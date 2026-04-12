"""LaminarV1V2Network: top-level composer for the laminar V1-V2 model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig
from src.state import NetworkState, StepAux, initial_state
from src.model.populations import V1L4Ring, PVPool, V1L23Ring, SOMRing
from src.model.v2_context import V2ContextModule


class LaminarV1V2Network(nn.Module):
    """Complete laminar V1-V2 network.

    Composes: V1L4Ring, PVPool, V1L23Ring, SOMRing, V2ContextModule.

    V2 feedback uses a single Linear(16, 36) head with Dale's law E/I split:
        - relu(+feedback_signal) → center_exc (additive excitation to L2/3)
        - relu(-feedback_signal) → som_drive (drives SOM → subtractive inhibition)

    Dependency order per timestep:
        L4 → PV → V2 (uses L2/3_{t-1}) → feedback E/I split → SOM → L2/3
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.l4 = V1L4Ring(cfg)
        self.pv = PVPool(cfg)
        self.l23 = V1L23Ring(cfg)
        self.som = SOMRing(cfg)

        # Fix 1: dual V2 architecture — two independent V2 modules.
        self.use_dual_v2 = getattr(cfg, "use_dual_v2", False)
        if self.use_dual_v2:
            # Each regime gets its own V2 with independent GRU, head_mu,
            # head_pi, head_feedback. use_per_regime_feedback is irrelevant
            # since each V2 has exactly one head_feedback.
            cfg_single = ModelConfig(**{
                k: getattr(cfg, k) for k in cfg.__dataclass_fields__
                if k not in ('use_dual_v2', 'use_per_regime_feedback')
            }, use_dual_v2=False, use_per_regime_feedback=False)
            self.v2_focused = V2ContextModule(cfg_single)
            self.v2_routine = V2ContextModule(cfg_single)
            # Legacy self.v2 alias for code that references it (e.g. freeze_v2
            # in trainer.py). Points to v2_focused for parameter freezing.
            self.v2 = self.v2_focused
        else:
            self.v2 = V2ContextModule(cfg)

        # Feedback warmup scale: registered as buffer so torch.compile can see it
        self.register_buffer("feedback_scale", torch.tensor(1.0))

        # Phase 2: causal E/I gate on the feedback split.
        # When use_ei_gate is True, a tiny Linear(3, 2) maps
        # (task_state[:, :2], pi_pred_raw[:, :1]) → (g_E, g_I) via 2*sigmoid,
        # giving the network a direct multiplicative pathway for task_state
        # to bias the E/I split without compromising V2's feedback head.
        # Init: bias=0, weight N(0, 0.01) → sigmoid(~0)*2 ≈ 1.0 at step 0,
        # so the module is near-identity at initialization (preserves
        # pre-Phase-2 behavior within fp noise for regression tests).
        self.use_ei_gate = getattr(cfg, "use_ei_gate", False)
        if self.use_ei_gate:
            self.alpha_net = nn.Linear(2 + 1, 2)
            nn.init.zeros_(self.alpha_net.bias)
            nn.init.normal_(self.alpha_net.weight, std=0.01)

        # Oracle mode: bypass V2 with injected predictions
        self.oracle_mode = False
        self.oracle_q_pred = None   # set externally: [B, N] (per-step) or [B, T, N] (per-sequence)
        self.oracle_pi_pred = None  # set externally: [B, 1] (scalar) or [B, T, 1] (per-sequence)
        # Per-timestep oracle slices (set by forward() for sequence mode to avoid
        # storing a Python int counter that causes torch.compile recompilation)
        self._oracle_q_step: Tensor | None = None
        self._oracle_pi_step: Tensor | None = None

    def _make_bump(
        self,
        thetas: Tensor,
        sigma: float | None = None,
    ) -> Tensor:
        """Create Gaussian bumps centered at each orientation in ``thetas``.

        Used by oracle-template modes in Stage 2 training (``stage2_feedback.py``)
        to construct peaked prediction priors q_pred of shape [B, N].

        Args:
            thetas: [B] — center orientations in degrees.
            sigma: Width of the bump in degrees. Defaults to ``cfg.sigma_ff``.

        Returns:
            bumps: [B, N] — un-normalised Gaussian bumps (caller normalises).
        """
        N = self.cfg.n_orientations
        period = self.cfg.orientation_range
        step = period / N
        if sigma is None:
            sigma = self.cfg.sigma_ff
        pref = torch.arange(N, dtype=torch.float32, device=thetas.device) * step
        diff = (pref.unsqueeze(0) - thetas.unsqueeze(1) + period / 2) % period - period / 2
        return torch.exp(-diff ** 2 / (2 * sigma ** 2))

    def step(
        self,
        stimulus: Tensor,
        cue: Tensor,
        task_state: Tensor,
        state: NetworkState,
    ) -> tuple[NetworkState, StepAux]:
        """One timestep of the full network.

        Args:
            stimulus: [B, N] -- population-coded, contrast-scaled grating.
            cue: [B, N] -- cue input (zeros by default).
            task_state: [B, 2] -- task relevance state.
            state: Previous NetworkState.

        Returns:
            new_state: Updated NetworkState.
            aux: StepAux with q_pred, pi_pred, pi_pred_eff, state_logits.
        """
        B = stimulus.shape[0]
        N = self.cfg.n_orientations
        device = stimulus.device

        # 1. L4 (uses PV from previous step)
        r_l4, adaptation = self.l4(stimulus, state.r_l4, state.r_pv, state.adaptation)

        # 2. PV (uses new L4, old L2/3)
        r_pv = self.pv(r_l4, state.r_l23, state.r_pv)

        # 3. V2 (uses r_l4 from this step + old L2/3 -- one-step feedback delay)
        if self.oracle_mode and self.oracle_q_pred is not None:
            # Support both per-step [B, N] and per-sequence [B, T, N] oracle.
            if self.oracle_q_pred.dim() == 3:
                q_pred = self._oracle_q_step
                pi_pred_raw = self._oracle_pi_step
            else:
                q_pred = self.oracle_q_pred
                pi_pred_raw = self.oracle_pi_pred
            state_logits = torch.zeros(B, 3, device=device)
            p_cw = torch.full((B, 1), 0.5, device=device)
            # V2's head_feedback must still run to produce the feedback signal
            _, _, feedback_signal, h_v2 = self.v2(
                r_l4, state.r_l23, cue, task_state, state.h_v2
            )
        elif self.use_dual_v2:
            # Fix 1: run both V2 modules, blend outputs by task_state.
            # Each V2 gets its own h_v2 hidden state (packed in the first/second
            # half of state.h_v2, which is [B, 2*H] for dual V2).
            H = self.cfg.v2_hidden_dim
            h_v2_foc_prev = state.h_v2[:, :H]
            h_v2_rou_prev = state.h_v2[:, H:]
            mu_foc, pi_foc, fb_foc, h_v2_foc = self.v2_focused(
                r_l4, state.r_l23, cue, task_state, h_v2_foc_prev
            )
            mu_rou, pi_rou, fb_rou, h_v2_rou = self.v2_routine(
                r_l4, state.r_l23, cue, task_state, h_v2_rou_prev
            )
            # Blend by task_state: focused samples → v2_focused, routine → v2_routine.
            # For baseline (0,0), both contribute zero → falls back to zeros.
            w_foc = task_state[:, 0:1]  # [B, 1]
            w_rou = task_state[:, 1:2]  # [B, 1]
            # Normalise weights so pure focused/routine select exactly one module.
            # Baseline (0,0): both weights are 0 → use focused as default.
            w_sum = (w_foc + w_rou).clamp(min=1e-8)
            w_foc_n = w_foc / w_sum
            w_rou_n = w_rou / w_sum
            q_pred = w_foc_n * mu_foc + w_rou_n * mu_rou
            pi_pred_raw = w_foc_n * pi_foc + w_rou_n * pi_rou
            feedback_signal = w_foc_n * fb_foc + w_rou_n * fb_rou
            h_v2 = torch.cat([h_v2_foc, h_v2_rou], dim=-1)  # [B, 2*H]
            p_cw = torch.full((B, 1), 0.5, device=device)
            state_logits = torch.zeros(B, 3, device=device)
        else:
            mu_pred, pi_pred_raw, feedback_signal, h_v2 = self.v2(
                r_l4, state.r_l23, cue, task_state, state.h_v2
            )
            q_pred = mu_pred  # V2 directly outputs the prior distribution
            p_cw = torch.full((B, 1), 0.5, device=device)  # compat placeholder
            state_logits = torch.zeros(B, 3, device=device)

        # Effective precision for V1 feedback (scaled by warmup ramp during training)
        pi_pred_eff = pi_pred_raw * self.feedback_scale

        # 4. V2 direct feedback with E/I split (Dale's law):
        # positive → excitation to L2/3, negative → drives SOM interneurons
        if self.cfg.use_precision_gating:
            # Rescue 2: precision multiplicatively gates feedback strength.
            # precision_gate = pi_pred_raw / pi_max ∈ [0, 1].
            # At max precision (pi_pred_raw = pi_max), gate = 1.0 → feedback
            # unchanged. At zero precision, gate = 0 → feedback silenced.
            # The [0, 1] range means precision can only ATTENUATE, never
            # amplify. During burn-in (feedback_scale≈0), scaled_fb≈0
            # regardless, preserving the curriculum.
            precision_gate = pi_pred_raw / self.cfg.pi_max  # [B, 1], [0, 1]
            scaled_fb = feedback_signal * self.feedback_scale * precision_gate
        else:
            scaled_fb = feedback_signal * self.feedback_scale
        if self.use_ei_gate:
            # Phase 2 causal gate: multiplicative per-sample E/I scaling.
            # Gate input uses pi_pred_raw (not pi_pred_eff) so the gate can
            # receive nonzero gradient signal during the feedback_scale burn-in
            # when scaled_fb ≈ 0 would otherwise zero out the downstream path.
            gate_input = torch.cat([task_state, pi_pred_raw], dim=-1)  # [B, 3]
            gains = 2.0 * torch.sigmoid(self.alpha_net(gate_input))    # [B, 2], init ≈ 1.0
            g_E = gains[:, 0:1]                                         # [B, 1] → broadcast to [B, N]
            g_I = gains[:, 1:2]                                         # [B, 1]
            center_exc = g_E * torch.relu(scaled_fb)
            som_drive_fb = g_I * torch.relu(-scaled_fb)
        else:
            gains = None
            center_exc = torch.relu(scaled_fb)           # positive part → excitation
            som_drive_fb = torch.relu(-scaled_fb)        # negative part → SOM drive
        r_som = self.som(som_drive_fb, state.r_som)  # SOM integrates with tau_som

        # 5. L2/3 update
        r_l23 = self.l23(r_l4, state.r_l23, center_exc, r_som, r_pv)

        # Deep template placeholder (q_pred * pi_eff, retained for energy cost compatibility)
        deep_tmpl = q_pred * pi_pred_eff

        new_state = NetworkState(
            r_l4=r_l4,
            r_l23=r_l23,
            r_pv=r_pv,
            r_som=r_som,
            r_vip=torch.zeros(B, N, device=device),
            adaptation=adaptation,
            h_v2=h_v2,
            deep_template=deep_tmpl,
        )

        aux = StepAux(
            q_pred=q_pred,
            pi_pred=pi_pred_raw,
            pi_pred_eff=pi_pred_eff,
            state_logits=state_logits,
            p_cw=p_cw,
            center_exc=center_exc,
            som_drive_fb=som_drive_fb,
            gains=gains,
        )

        return new_state, aux

    @staticmethod
    def pack_inputs(
        stimulus_seq: Tensor,
        cue_seq: Tensor | None = None,
        task_state_seq: Tensor | None = None,
    ) -> Tensor:
        """Pack stimulus, cue, and task_state into a single tensor.

        Args:
            stimulus_seq: [B, T, N]
            cue_seq: [B, T, N] or None (zeros)
            task_state_seq: [B, T, 2] or None (zeros)

        Returns:
            packed: [B, T, N + N + 2] = [B, T, 74] for N=36.
        """
        B, T, N = stimulus_seq.shape
        device = stimulus_seq.device
        if cue_seq is None:
            cue_seq = torch.zeros(B, T, N, device=device)
        if task_state_seq is None:
            task_state_seq = torch.zeros(B, T, 2, device=device)
        return torch.cat([stimulus_seq, cue_seq, task_state_seq], dim=-1)

    def forward(
        self,
        packed_input: Tensor,
        state: NetworkState | None = None,
    ) -> tuple[Tensor, NetworkState, dict[str, Tensor]]:
        """Run a sequence of timesteps.

        Args:
            packed_input: [B, T, N+N+2] -- packed stimulus + cue + task_state.
                Use LaminarV1V2Network.pack_inputs() to create this, or pass
                a raw [B, T, N] stimulus (cue and task_state default to zeros).
            state: Initial NetworkState or None (defaults to zeros).

        Returns:
            r_l23_all: [B, T, N] -- L2/3 trajectory.
            final_state: NetworkState after last timestep.
            aux: dict with stacked trajectories:
                 q_pred_all [B, T, N], pi_pred_all [B, T, 1],
                 state_logits_all [B, T, 3], deep_template_all [B, T, N],
                 r_l4_all [B, T, N], r_pv_all [B, T, 1], r_som_all [B, T, N],
                 r_vip_all [B, T, N], p_cw_all [B, T, 1].
        """
        N = self.cfg.n_orientations

        # Support both packed [B, T, N+N+2] and unpacked [B, T, N] inputs
        if packed_input.shape[-1] == N:
            # Raw stimulus only -- generate zero cue and task_state
            B, T, _ = packed_input.shape
            device = packed_input.device
            stimulus_seq = packed_input
            cue_seq = torch.zeros(B, T, N, device=device)
            task_state_seq = torch.zeros(B, T, 2, device=device)
        else:
            B, T, _ = packed_input.shape
            device = packed_input.device
            stimulus_seq = packed_input[:, :, :N]
            cue_seq = packed_input[:, :, N:2*N]
            task_state_seq = packed_input[:, :, 2*N:]

        if state is None:
            # Fix 1: dual V2 packs both hidden states into h_v2 → [B, 2*H]
            h_dim = self.cfg.v2_hidden_dim * 2 if self.use_dual_v2 else self.cfg.v2_hidden_dim
            state = initial_state(
                B, self.cfg.n_orientations, h_dim, device=device
            )

        # Preallocate output tensors (avoids list appends + torch.stack overhead)
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
        # Phase 2.4: also track som_drive_fb trajectory so the routine_shape
        # loss can read it from aux. Populated in both gate-on and gate-off
        # paths (som_drive_fb is always computed in step()).
        som_drive_fb_all = torch.empty(B, T, N, device=device)
        # Phase 2: preallocate per-step gate output trajectory. Only populated
        # when use_ei_gate=True, otherwise left as zeros so downstream code
        # can assume the key exists and has a defined shape.
        gains_all = torch.zeros(B, T, 2, device=device)

        # Check if oracle mode uses sequence-length tensors [B, T, ...]
        _oracle_seq = (self.oracle_mode and self.oracle_q_pred is not None
                       and self.oracle_q_pred.dim() == 3)

        # Cache kernels once for all timesteps (params don't change within forward)
        self.l23.cache_kernels()
        try:
            for t in range(T):
                # Pre-slice oracle tensors per-timestep
                if _oracle_seq:
                    self._oracle_q_step = self.oracle_q_pred[:, t]
                    self._oracle_pi_step = self.oracle_pi_pred[:, t]

                state, aux_t = self.step(
                    stimulus_seq[:, t], cue_seq[:, t], task_state_seq[:, t], state
                )
                r_l23_all[:, t] = state.r_l23
                r_l4_all[:, t] = state.r_l4
                r_pv_all[:, t] = state.r_pv
                r_som_all[:, t] = state.r_som
                r_vip_all[:, t] = state.r_vip
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
                if aux_t.som_drive_fb is not None:
                    som_drive_fb_all[:, t] = aux_t.som_drive_fb
                else:
                    som_drive_fb_all[:, t] = 0.0
                if aux_t.gains is not None:
                    gains_all[:, t] = aux_t.gains
        finally:
            self.l23.uncache_kernels()

        aux = {
            "q_pred_all": q_pred_all,             # [B, T, N]
            "pi_pred_all": pi_pred_all,           # [B, T, 1]
            "pi_pred_eff_all": pi_pred_eff_all,   # [B, T, 1]
            "state_logits_all": state_logits_all,  # [B, T, 3]
            "deep_template_all": deep_template_all,  # [B, T, N]
            "r_l4_all": r_l4_all,                 # [B, T, N]
            "r_pv_all": r_pv_all,                 # [B, T, 1]
            "r_som_all": r_som_all,               # [B, T, N]
            "r_vip_all": r_vip_all,               # [B, T, N]
            "p_cw_all": p_cw_all,                 # [B, T, 1]
            "center_exc_all": center_exc_all,     # [B, T, N]
            "som_drive_fb_all": som_drive_fb_all, # [B, T, N]
            "gains_all": gains_all,               # [B, T, 2]  zeros when use_ei_gate=False
        }

        return r_l23_all, state, aux
