"""SpikingLaminarV1V2Network: top-level composer for the spiking V1-V2 model.

Parallel to `src/model/network.py::LaminarV1V2Network`. This module is a
drop-in replacement that runs LIF/ALIF dynamics instead of rate dynamics and
exposes the exact same external API (`forward`, `step`, return dict keys),
so that `src/experiments/`, `src/analysis/`, `src/training/losses.py`, and
all scripts work without modification.

Key design decisions
--------------------
1. **Reused infrastructure (unchanged)**:
   - `src.model.populations.DeepTemplate` — pure algebra over (q_pred, pi),
     no rate-specific state; reusable as-is.
   - `src.model.feedback.EmergentFeedbackOperator` and `FeedbackMechanism` —
     operate only on V2 outputs `(q_pred, pi_eff)`, do not touch neural
     state, so they plug directly into the spiking pipeline.

2. **Same step dependency order**:
       L4 → PV → V2 (uses L2/3_{t-1}) → DeepTemplate → Feedback → SOM/VIP → L2/3
   This is verified line-for-line against `src/model/network.py::step`.

3. **Rate-compatible output dict**: every `r_*` key holds the filtered spike
   trace `x_*` so downstream rate-facing code works unchanged. Additionally
   exposed (new for spiking-aware callers):
       spike_l4_all, spike_l23_all, spike_som_all, spike_vip_all, spike_v2_all
       v_*_all (membrane potentials) — optional for analysis.

4. **Feedback branches** (mirror rate network):
   - `emergent + simple_feedback`: V2 head_feedback → relu(+) → center_exc,
     relu(-) → SOM drive, no EmergentFeedbackOperator call.
   - `emergent`: EmergentFeedbackOperator consumes (q_pred, pi_eff),
     produces (som_drive, vip_drive, apical_gain).
   - `fixed`: FeedbackMechanism hardcoded kernels; no VIP, L4 error signal.

5. **Oracle mode**: identical semantics to rate network (per-step and
   per-sequence modes both supported).

Evidence
--------
- Plan: `/home/vishnu/.claude/plans/quirky-humming-giraffe.md`
    * Architecture mapping (lines 17-26)
    * SpikingNetworkState layout (lines 100-128)
    * "Files unchanged (shared infrastructure)" list (lines 86-94) — confirms
      DeepTemplate and EmergentFeedbackOperator are reusable.
- Rate reference: `src/model/network.py` — API and step order copied verbatim.
- State container: `src/spiking/state.py::SpikingNetworkState` — 20 fields.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, SpikingConfig
from src.model.feedback import EmergentFeedbackOperator, FeedbackMechanism
from src.model.populations import DeepTemplate
from src.spiking.populations import (
    SpikingL23Ring,
    SpikingL4Ring,
    SpikingPVPool,
    SpikingSOMRing,
    SpikingVIPRing,
)
from src.spiking.state import SpikingNetworkState, initial_spiking_state
from src.spiking.v2_context import SpikingV2Context
from src.state import StepAux
from src.utils import circular_distance_abs


class SpikingLaminarV1V2Network(nn.Module):
    """Spiking laminar V1-V2 network.

    Composes: SpikingL4Ring, SpikingPVPool, SpikingL23Ring, DeepTemplate,
              SpikingSOMRing, SpikingVIPRing, SpikingV2Context,
              EmergentFeedbackOperator or FeedbackMechanism (reused).

    Dependency order per timestep:
        L4 -> PV -> V2 (uses x_l23_{t-1}) -> template -> feedback -> SOM/VIP -> L2/3

    Two feedback modes (`cfg.feedback_mode`):
        'fixed':    V2 outputs q_pred directly; FeedbackMechanism (hardcoded).
        'emergent': V2 outputs mu_pred; EmergentFeedbackOperator (learned).
                    With `cfg.simple_feedback=True`, the V2 head_feedback path
                    is used instead of EmergentFeedbackOperator (E/I split).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        spiking_cfg: SpikingConfig,
        delta_som: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.spiking_cfg = spiking_cfg

        # ---- Spiking populations ----
        self.l4 = SpikingL4Ring(cfg, spiking_cfg)
        self.pv = SpikingPVPool(cfg)
        self.l23 = SpikingL23Ring(cfg, spiking_cfg)
        self.som = SpikingSOMRing(cfg, spiking_cfg)
        self.vip = SpikingVIPRing(cfg, spiking_cfg)

        # ---- Shared (algebraic) reused from rate model ----
        self.deep_template = DeepTemplate(cfg)

        # ---- V2 context (LSNN) ----
        self.v2 = SpikingV2Context(cfg, spiking_cfg)

        # Learnable VIP→SOM gain (softplus-wrapped) — same as rate network.
        self.w_vip_som = nn.Parameter(torch.tensor(0.5))
        # Learnable template→L2/3 center excitation weight (init 0.0 = off).
        self.w_template_drive = nn.Parameter(torch.tensor(0.0))

        # Feedback: emergent (learned) or fixed (hardcoded) — both reused unchanged.
        if cfg.feedback_mode == "emergent":
            self.feedback = EmergentFeedbackOperator(cfg, delta_som=delta_som)
        else:
            self.feedback = FeedbackMechanism(cfg)

        # Feedback warmup scale (buffer so torch.compile can see it).
        self.register_buffer("feedback_scale", torch.tensor(1.0))

        # Oracle mode (bypass V2 with injected predictions) — same API as rate.
        self.oracle_mode = False
        self.oracle_q_pred: Tensor | None = None   # [B, N] or [B, T, N]
        self.oracle_pi_pred: Tensor | None = None  # [B, 1] or [B, T, 1]
        self._oracle_q_step: Tensor | None = None
        self._oracle_pi_step: Tensor | None = None

    # ------------------------------------------------------------------
    # Helpers (identical to rate network; copied verbatim for isolation)
    # ------------------------------------------------------------------

    def _decode_orientation(self, r_l4: Tensor) -> Tensor:
        """Population vector decode from L4 filtered trace (same math as rate)."""
        N = r_l4.shape[-1]
        step = self.cfg.orientation_range / N
        prefs = torch.arange(N, device=r_l4.device, dtype=torch.float32) * step
        angles_rad = prefs * (2.0 * math.pi / self.cfg.orientation_range)
        z = (r_l4 * torch.exp(1j * angles_rad.unsqueeze(0).to(torch.cfloat))).sum(dim=-1)
        theta = torch.angle(z) * (self.cfg.orientation_range / (2.0 * math.pi))
        return theta % self.cfg.orientation_range

    def _make_bump(self, theta: Tensor, sigma: float | None = None) -> Tensor:
        """Circular Gaussian bump at orientation theta (same math as rate)."""
        N = self.cfg.n_orientations
        step = self.cfg.orientation_range / N
        prefs = torch.arange(N, device=theta.device, dtype=torch.float32) * step
        dists = circular_distance_abs(
            theta.unsqueeze(-1), prefs.unsqueeze(0), self.cfg.orientation_range
        )
        sigma_val = sigma if sigma is not None else self.cfg.sigma_ff
        return torch.exp(-dists ** 2 / (2 * sigma_val ** 2))

    def _construct_q_pred(self, r_l4: Tensor, p_cw: Tensor) -> Tensor:
        """Construct q_pred from L4 trace + state belief (same math as rate)."""
        theta_current = self._decode_orientation(r_l4)
        step = self.cfg.transition_step
        q_cw = self._make_bump(theta_current + step)
        q_ccw = self._make_bump(theta_current - step)
        q_pred = p_cw * q_cw + (1 - p_cw) * q_ccw
        q_pred = q_pred / (q_pred.sum(dim=-1, keepdim=True) + 1e-8)
        return q_pred

    # ------------------------------------------------------------------
    # Per-population state packing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l4_state_from(state: SpikingNetworkState) -> dict:
        return {"v": state.v_l4, "z": state.z_l4, "x": state.x_l4, "b": state.adapt_l4}

    @staticmethod
    def _l23_state_from(state: SpikingNetworkState) -> dict:
        return {"v": state.v_l23, "z": state.z_l23, "x": state.x_l23}

    @staticmethod
    def _som_state_from(state: SpikingNetworkState) -> dict:
        return {"v": state.v_som, "z": state.z_som, "x": state.x_som}

    @staticmethod
    def _vip_state_from(state: SpikingNetworkState) -> dict:
        return {"v": state.v_vip, "z": state.z_vip, "x": state.x_vip}

    @staticmethod
    def _v2_state_from(state: SpikingNetworkState) -> dict:
        return {"v": state.v_v2, "z": state.z_v2, "x": state.x_v2, "b": state.b_v2}

    @staticmethod
    def _pv_state_from(state: SpikingNetworkState) -> dict:
        return {"r_pv": state.r_pv}

    # ------------------------------------------------------------------
    # One timestep
    # ------------------------------------------------------------------

    def step(
        self,
        stimulus: Tensor,
        cue: Tensor,
        task_state: Tensor,
        state: SpikingNetworkState,
    ) -> tuple[SpikingNetworkState, StepAux]:
        """One timestep of the full spiking network.

        Mirrors `LaminarV1V2Network.step` exactly for dependency order and
        branching logic. The only substitutions are:
          - rate populations -> spiking populations,
          - rate output (`r_*`) -> filtered spike trace (`x_*`).

        Args:
            stimulus: [B, N] population-coded, contrast-scaled grating.
            cue: [B, N] cue input (zeros by default).
            task_state: [B, 2] task relevance state.
            state: Previous SpikingNetworkState.

        Returns:
            new_state: Updated SpikingNetworkState.
            aux: StepAux with q_pred, pi_pred, pi_pred_eff, state_logits.
        """
        B = stimulus.shape[0]
        device = stimulus.device
        N = self.cfg.n_orientations

        # --------------------------------------------------------------
        # 1. L4 (uses PV from previous step) — spiking ALIF
        # --------------------------------------------------------------
        new_l4_state, z_l4, x_l4 = self.l4(
            stimulus, state.r_pv, self._l4_state_from(state)
        )

        # --------------------------------------------------------------
        # 2. PV (uses new x_l4, old x_l23) — rate-based leaky integrator
        # --------------------------------------------------------------
        new_pv_state, r_pv, _ = self.pv(
            x_l4, state.x_l23, self._pv_state_from(state)
        )

        # --------------------------------------------------------------
        # 3. V2 LSNN (uses x_l4 this step + x_l23 previous — 1-step feedback delay)
        # --------------------------------------------------------------
        if self.oracle_mode and self.oracle_q_pred is not None:
            # Oracle bypass: accept pre-shaped q/pi_pred from the outside.
            if self.oracle_q_pred.dim() == 3:
                q_pred = self._oracle_q_step
                pi_pred_raw = self._oracle_pi_step
            else:
                q_pred = self.oracle_q_pred
                pi_pred_raw = self.oracle_pi_pred
            state_logits = torch.zeros(B, 3, device=device)
            p_cw = torch.full((B, 1), 0.5, device=device)
            # In simple_feedback mode, V2's head_feedback must still run
            # so the feedback signal is produced — oracle only overrides q/pi.
            if self.cfg.simple_feedback and self.cfg.feedback_mode == "emergent":
                _, _, feedback_signal, new_v2_state = self.v2(
                    x_l4, state.x_l23, cue, task_state, self._v2_state_from(state)
                )
            else:
                new_v2_state = self._v2_state_from(state)
                feedback_signal = torch.zeros(B, N, device=device)
        elif self.cfg.feedback_mode == "emergent":
            mu_pred, pi_pred_raw, feedback_signal, new_v2_state = self.v2(
                x_l4, state.x_l23, cue, task_state, self._v2_state_from(state)
            )
            q_pred = mu_pred  # V2 directly outputs the prior distribution
            p_cw = torch.full((B, 1), 0.5, device=device)  # compat placeholder
            state_logits = torch.zeros(B, 3, device=device)
        else:
            q_pred, pi_pred_raw, state_logits, new_v2_state = self.v2(
                x_l4, state.x_l23, cue, task_state, self._v2_state_from(state)
            )
            p_cw = torch.full((B, 1), 0.5, device=device)
            feedback_signal = torch.zeros(B, N, device=device)

        # Effective precision for V1 feedback (scaled by warmup ramp).
        pi_pred_eff = pi_pred_raw * self.feedback_scale

        # --------------------------------------------------------------
        # 4. Deep template (uses effective precision) — reused rate module
        # --------------------------------------------------------------
        deep_tmpl = self.deep_template(q_pred, pi_pred_eff)

        # --------------------------------------------------------------
        # 5-6. Feedback pathway (branched by mode) + SOM/VIP + L2/3
        # --------------------------------------------------------------
        if self.cfg.feedback_mode == "emergent" and self.cfg.simple_feedback:
            # V2 direct feedback with E/I split (Dale's law):
            # positive → excitation to L2/3, negative → drives SOM interneurons.
            scaled_fb = feedback_signal * self.feedback_scale
            center_exc = torch.relu(scaled_fb)        # [B, N]
            som_drive_fb = torch.relu(-scaled_fb)     # [B, N]

            new_som_state, z_som, x_som = self.som(
                som_drive_fb, self._som_state_from(state)
            )
            # VIP inactive in simple_feedback: run a zero-drive LIF step so the
            # state advances with the same cadence as the other populations
            # (consistent with rate model where r_vip is reset to zeros).
            vip_zero_drive = torch.zeros(B, N, device=device)
            new_vip_state, z_vip, x_vip = self.vip(
                vip_zero_drive, self._vip_state_from(state)
            )

            new_l23_state, z_l23, x_l23 = self.l23(
                x_l4, center_exc, x_som, r_pv, self._l23_state_from(state),
                apical_gain=None,
            )
        elif self.cfg.feedback_mode == "emergent":
            # Emergent: learned operator outputs SOM + VIP drives + apical gain.
            som_drive, vip_drive, apical_gain = self.feedback(
                q_pred, pi_pred_eff, r_l4=None
            )
            new_vip_state, z_vip, x_vip = self.vip(
                vip_drive, self._vip_state_from(state)
            )
            # VIP inhibits SOM: reduce SOM drive where VIP is active.
            effective_som_drive = F.relu(
                som_drive - F.softplus(self.w_vip_som) * x_vip
            )
            center_exc = self.w_template_drive * deep_tmpl
            new_som_state, z_som, x_som = self.som(
                effective_som_drive, self._som_state_from(state)
            )
            new_l23_state, z_l23, x_l23 = self.l23(
                x_l4, center_exc, x_som, r_pv, self._l23_state_from(state),
                apical_gain=apical_gain,
            )
        else:
            # Fixed: mechanism-specific computation (VIP not active).
            # Run a zero-drive LIF step for VIP so its state still advances.
            vip_zero_drive = torch.zeros(B, N, device=device)
            new_vip_state, z_vip, x_vip = self.vip(
                vip_zero_drive, self._vip_state_from(state)
            )
            som_drive = self.feedback.compute_som_drive(q_pred, pi_pred_eff)
            new_som_state, z_som, x_som = self.som(
                som_drive, self._som_state_from(state)
            )
            center_exc = self.feedback.compute_center_excitation(q_pred, pi_pred_eff)
            # Error-signal L4 input: rate model calls
            # feedback.compute_error_signal(r_l4, deep_tmpl). We pass x_l4 as
            # the rate-equivalent trace.
            l4_to_l23 = self.feedback.compute_error_signal(x_l4, deep_tmpl)
            new_l23_state, z_l23, x_l23 = self.l23(
                l4_to_l23, center_exc, x_som, r_pv, self._l23_state_from(state),
                apical_gain=None,
            )

        # --------------------------------------------------------------
        # Assemble new state
        # --------------------------------------------------------------
        new_state = SpikingNetworkState(
            # L4 (ALIF)
            v_l4=new_l4_state["v"],
            z_l4=new_l4_state["z"],
            x_l4=new_l4_state["x"],
            adapt_l4=new_l4_state["b"],
            # PV (rate)
            r_pv=new_pv_state["r_pv"],
            # L2/3
            v_l23=new_l23_state["v"],
            z_l23=new_l23_state["z"],
            x_l23=new_l23_state["x"],
            # SOM
            v_som=new_som_state["v"],
            z_som=new_som_state["z"],
            x_som=new_som_state["x"],
            # VIP
            v_vip=new_vip_state["v"],
            z_vip=new_vip_state["z"],
            x_vip=new_vip_state["x"],
            # V2
            v_v2=new_v2_state["v"],
            z_v2=new_v2_state["z"],
            x_v2=new_v2_state["x"],
            b_v2=new_v2_state["b"],
            # Shared
            deep_template=deep_tmpl,
        )

        aux = StepAux(
            q_pred=q_pred,
            pi_pred=pi_pred_raw,
            pi_pred_eff=pi_pred_eff,
            state_logits=state_logits,
            p_cw=p_cw,
            center_exc=center_exc,
        )

        return new_state, aux

    # ------------------------------------------------------------------
    # Input packing (identical API to rate network)
    # ------------------------------------------------------------------

    @staticmethod
    def pack_inputs(
        stimulus_seq: Tensor,
        cue_seq: Tensor | None = None,
        task_state_seq: Tensor | None = None,
    ) -> Tensor:
        """Pack stimulus + cue + task_state into a single [B, T, N+N+2] tensor."""
        B, T, N = stimulus_seq.shape
        device = stimulus_seq.device
        if cue_seq is None:
            cue_seq = torch.zeros(B, T, N, device=device)
        if task_state_seq is None:
            task_state_seq = torch.zeros(B, T, 2, device=device)
        return torch.cat([stimulus_seq, cue_seq, task_state_seq], dim=-1)

    # ------------------------------------------------------------------
    # Full forward over T timesteps
    # ------------------------------------------------------------------

    def forward(
        self,
        packed_input: Tensor,
        state: SpikingNetworkState | None = None,
    ) -> tuple[Tensor, SpikingNetworkState, dict[str, Tensor]]:
        """Run a sequence of timesteps.

        Args:
            packed_input: [B, T, N+N+2] packed stimulus + cue + task_state
                (or raw [B, T, N] stimulus — cue and task_state default to zeros).
            state: Initial SpikingNetworkState or None (defaults to zeros).

        Returns:
            r_l23_all: [B, T, N] — filtered L2/3 spike trace (rate-compatible).
            final_state: SpikingNetworkState after the last timestep.
            aux: dict with the same keys as the rate network plus spike
                 trajectories. Rate keys (r_*_all) carry filtered traces (x_*):
                    q_pred_all, pi_pred_all, pi_pred_eff_all, state_logits_all,
                    deep_template_all, r_l4_all, r_pv_all, r_som_all, r_vip_all,
                    p_cw_all, center_exc_all
                 Spike-specific keys (new, for analysis):
                    spike_l4_all, spike_l23_all, spike_som_all, spike_vip_all,
                    spike_v2_all
        """
        N = self.cfg.n_orientations

        # Support both packed [B, T, N+N+2] and unpacked [B, T, N] inputs.
        if packed_input.shape[-1] == N:
            B, T, _ = packed_input.shape
            device = packed_input.device
            stimulus_seq = packed_input
            cue_seq = torch.zeros(B, T, N, device=device)
            task_state_seq = torch.zeros(B, T, 2, device=device)
        else:
            B, T, _ = packed_input.shape
            device = packed_input.device
            stimulus_seq = packed_input[:, :, :N]
            cue_seq = packed_input[:, :, N:2 * N]
            task_state_seq = packed_input[:, :, 2 * N:]

        if state is None:
            state = initial_spiking_state(
                B,
                n_orientations=self.cfg.n_orientations,
                v2_hidden_dim=self.spiking_cfg.n_lsnn_neurons,
                device=device,
            )

        # ---- Preallocate rate-compatible output tensors ----
        r_l23_all = torch.empty(B, T, N, device=device)
        r_l4_all = torch.empty(B, T, N, device=device)
        r_pv_all = torch.empty(B, T, 1, device=device)
        r_som_all = torch.empty(B, T, N, device=device)
        r_vip_all = torch.empty(B, T, N, device=device)
        q_pred_all = torch.empty(B, T, N, device=device)
        pi_pred_all = torch.empty(B, T, 1, device=device)
        pi_pred_eff_all = torch.empty(B, T, 1, device=device)
        state_logits_all = torch.empty(B, T, 3, device=device)
        deep_template_all = torch.empty(B, T, N, device=device)
        p_cw_all = torch.empty(B, T, 1, device=device)
        center_exc_all = torch.empty(B, T, N, device=device)

        # ---- Spike-specific trajectories (new for spiking analysis) ----
        spike_l4_all = torch.empty(B, T, N, device=device)
        spike_l23_all = torch.empty(B, T, N, device=device)
        spike_som_all = torch.empty(B, T, N, device=device)
        spike_vip_all = torch.empty(B, T, N, device=device)
        spike_v2_all = torch.empty(B, T, self.spiking_cfg.n_lsnn_neurons, device=device)

        # Oracle sequence-mode check.
        _oracle_seq = (
            self.oracle_mode
            and self.oracle_q_pred is not None
            and self.oracle_q_pred.dim() == 3
        )

        # Cache kernels once for all timesteps (params don't change within forward).
        self.l23.cache_kernels()
        if hasattr(self.feedback, "cache_kernels"):
            self.feedback.cache_kernels()
        try:
            for t in range(T):
                if _oracle_seq:
                    self._oracle_q_step = self.oracle_q_pred[:, t]
                    self._oracle_pi_step = self.oracle_pi_pred[:, t]

                state, aux_t = self.step(
                    stimulus_seq[:, t], cue_seq[:, t], task_state_seq[:, t], state
                )

                # Rate-compatible (filtered-trace) trajectories.
                r_l23_all[:, t] = state.x_l23
                r_l4_all[:, t] = state.x_l4
                r_pv_all[:, t] = state.r_pv
                r_som_all[:, t] = state.x_som
                r_vip_all[:, t] = state.x_vip
                q_pred_all[:, t] = aux_t.q_pred
                pi_pred_all[:, t] = aux_t.pi_pred
                pi_pred_eff_all[:, t] = aux_t.pi_pred_eff
                state_logits_all[:, t] = aux_t.state_logits
                deep_template_all[:, t] = state.deep_template
                p_cw_all[:, t] = aux_t.p_cw
                if aux_t.center_exc is not None:
                    center_exc_all[:, t] = aux_t.center_exc
                else:
                    center_exc_all[:, t] = 0.0

                # Spike trajectories.
                spike_l4_all[:, t] = state.z_l4
                spike_l23_all[:, t] = state.z_l23
                spike_som_all[:, t] = state.z_som
                spike_vip_all[:, t] = state.z_vip
                spike_v2_all[:, t] = state.z_v2
        finally:
            self.l23.uncache_kernels()
            if hasattr(self.feedback, "uncache_kernels"):
                self.feedback.uncache_kernels()

        aux = {
            # Rate-compatible keys (filtered traces under r_* names).
            "q_pred_all": q_pred_all,
            "pi_pred_all": pi_pred_all,
            "pi_pred_eff_all": pi_pred_eff_all,
            "state_logits_all": state_logits_all,
            "deep_template_all": deep_template_all,
            "r_l4_all": r_l4_all,
            "r_pv_all": r_pv_all,
            "r_som_all": r_som_all,
            "r_vip_all": r_vip_all,
            "p_cw_all": p_cw_all,
            "center_exc_all": center_exc_all,
            # Spike-specific trajectories (new).
            "spike_l4_all": spike_l4_all,
            "spike_l23_all": spike_l23_all,
            "spike_som_all": spike_som_all,
            "spike_vip_all": spike_vip_all,
            "spike_v2_all": spike_v2_all,
        }

        return r_l23_all, state, aux
