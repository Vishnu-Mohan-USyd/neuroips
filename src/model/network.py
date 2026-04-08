"""LaminarV1V2Network: top-level composer for the laminar V1-V2 model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig
from src.state import NetworkState, StepAux, initial_state
from src.model.populations import V1L4Ring, PVPool, V1L23Ring, DeepTemplate, SOMRing, VIPRing
from src.model.v2_context import V2ContextModule
from src.model.feedback import FeedbackMechanism, EmergentFeedbackOperator
from src.utils import circular_distance_abs, circular_gaussian


class LaminarV1V2Network(nn.Module):
    """Complete laminar V1-V2 network.

    Composes: V1L4Ring, PVPool, V1L23Ring, DeepTemplate, SOMRing,
              V2ContextModule, FeedbackMechanism or EmergentFeedbackOperator.

    Dependency order per timestep:
        L4 -> PV -> V2 (uses L2/3_{t-1}) -> template -> SOM -> L2/3

    Two feedback modes (cfg.feedback_mode):
        'fixed': V2 outputs q_pred directly; FeedbackMechanism (hardcoded A-E).
        'emergent': V2 outputs p_cw; q_pred constructed analytically from L4;
                    EmergentFeedbackOperator (learned basis functions).
    """

    def __init__(self, cfg: ModelConfig, delta_som: bool = False):
        super().__init__()
        self.cfg = cfg
        self.l4 = V1L4Ring(cfg)
        self.pv = PVPool(cfg)
        self.l23 = V1L23Ring(cfg)
        self.deep_template = DeepTemplate(cfg)
        self.som = SOMRing(cfg)
        self.v2 = V2ContextModule(cfg)

        # VIP interneurons (disinhibitory: VIP→SOM→L2/3)
        self.vip = VIPRing(cfg)
        # Learnable VIP→SOM gain (softplus-constrained to stay positive)
        self.w_vip_som = nn.Parameter(torch.tensor(0.5))
        # Branch C: learnable template→L2/3 center excitation weight (init 0.0 = off)
        self.w_template_drive = nn.Parameter(torch.tensor(0.0))

        # Feedback: emergent (learned) or fixed (hardcoded mechanism)
        if cfg.feedback_mode == 'emergent':
            self.feedback = EmergentFeedbackOperator(cfg, delta_som=delta_som)
        else:
            self.feedback = FeedbackMechanism(cfg)

        # Feedback warmup scale: registered as buffer so torch.compile can see it
        self.register_buffer("feedback_scale", torch.tensor(1.0))

        # Oracle mode: bypass V2 with injected predictions
        self.oracle_mode = False
        self.oracle_q_pred = None   # set externally: [B, N] (per-step) or [B, T, N] (per-sequence)
        self.oracle_pi_pred = None  # set externally: [B, 1] (scalar) or [B, T, 1] (per-sequence)
        # Per-timestep oracle slices (set by forward() for sequence mode to avoid
        # storing a Python int counter that causes torch.compile recompilation)
        self._oracle_q_step: Tensor | None = None
        self._oracle_pi_step: Tensor | None = None

    def _decode_orientation(self, r_l4: Tensor) -> Tensor:
        """Population vector decode from L4 firing rates.

        Uses complex exponential weighting for circular mean on the
        doubled-angle representation (180-deg periodic orientation space).

        Args:
            r_l4: [B, N] -- L4 population rates.

        Returns:
            theta: [B] -- decoded orientation in degrees, in [0, period).
        """
        N = r_l4.shape[-1]
        step = self.cfg.orientation_range / N
        prefs = torch.arange(N, device=r_l4.device, dtype=torch.float32) * step  # [N]

        # Map orientation to doubled angle: 0-180 deg -> 0-2pi radians
        angles_rad = prefs * (2.0 * math.pi / self.cfg.orientation_range)  # [N]

        # Complex exponential population vector
        z = (r_l4 * torch.exp(1j * angles_rad.unsqueeze(0).to(torch.cfloat))).sum(dim=-1)  # [B]

        # Convert back from doubled angle to orientation degrees
        theta = torch.angle(z) * (self.cfg.orientation_range / (2.0 * math.pi))  # [B]
        return theta % self.cfg.orientation_range

    def _make_bump(self, theta: Tensor, sigma: float | None = None) -> Tensor:
        """Create a population-coded Gaussian bump at orientation theta.

        By default uses the same sigma_ff as the feedforward tuning curves
        for consistency. Callers (e.g. the Phase 5 oracle-template
        construction) may pass a custom `sigma` to build a narrower or
        wider prior without affecting the feedforward pathway.

        Args:
            theta: [B] -- target orientation in degrees.
            sigma: width of the Gaussian (degrees). If None, defaults to
                `self.cfg.sigma_ff` (backward-compatible with all existing
                callers that pass only `theta`).

        Returns:
            bump: [B, N] -- circular Gaussian bump (unnormalized).
        """
        N = self.cfg.n_orientations
        step = self.cfg.orientation_range / N
        prefs = torch.arange(N, device=theta.device, dtype=torch.float32) * step  # [N]
        dists = circular_distance_abs(
            theta.unsqueeze(-1), prefs.unsqueeze(0), self.cfg.orientation_range
        )  # [B, N]
        sigma_val = sigma if sigma is not None else self.cfg.sigma_ff
        return torch.exp(-dists ** 2 / (2 * sigma_val ** 2))

    def _construct_q_pred(self, r_l4: Tensor, p_cw: Tensor) -> Tensor:
        """Construct q_pred analytically from L4 orientation + state belief.

        CW state: predicted next = current + transition_step
        CCW state: predicted next = current - transition_step
        q_pred = p_cw * bump(current + step) + (1 - p_cw) * bump(current - step)

        Args:
            r_l4: [B, N] -- L4 population rates (for decoding current orientation).
            p_cw: [B, 1] -- probability that the rule is CW.

        Returns:
            q_pred: [B, N] -- predicted next-orientation distribution (normalized).
        """
        theta_current = self._decode_orientation(r_l4)  # [B]
        step = self.cfg.transition_step

        q_cw = self._make_bump(theta_current + step)    # [B, N]
        q_ccw = self._make_bump(theta_current - step)   # [B, N]

        q_pred = p_cw * q_cw + (1 - p_cw) * q_ccw      # [B, N]
        q_pred = q_pred / (q_pred.sum(dim=-1, keepdim=True) + 1e-8)  # normalize
        return q_pred

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
        # 1. L4 (uses PV from previous step)
        r_l4, adaptation = self.l4(stimulus, state.r_l4, state.r_pv, state.adaptation)

        # 2. PV (uses new L4, old L2/3)
        r_pv = self.pv(r_l4, state.r_l23, state.r_pv)

        # 3. V2 (uses r_l4 from this step + old L2/3 -- one-step feedback delay)
        if self.oracle_mode and self.oracle_q_pred is not None:
            # Support both per-step [B, N] and per-sequence [B, T, N] oracle.
            # For sequence mode, forward() pre-slices into _oracle_q_step/pi_step
            # to avoid a Python int counter that triggers torch.compile recompilation.
            if self.oracle_q_pred.dim() == 3:
                q_pred = self._oracle_q_step
                pi_pred_raw = self._oracle_pi_step
            else:
                q_pred = self.oracle_q_pred
                pi_pred_raw = self.oracle_pi_pred
            state_logits = torch.zeros(stimulus.shape[0], 3, device=stimulus.device)
            p_cw = torch.full((stimulus.shape[0], 1), 0.5, device=stimulus.device)
            h_v2 = state.h_v2
        elif self.cfg.feedback_mode == 'emergent':
            mu_pred, pi_pred_raw, h_v2 = self.v2(
                r_l4, state.r_l23, cue, task_state, state.h_v2
            )
            q_pred = mu_pred  # V2 directly outputs the prior distribution
            p_cw = torch.full((stimulus.shape[0], 1), 0.5, device=stimulus.device)  # compat placeholder
            # No state_logits in emergent mode; use zeros placeholder for StepAux
            state_logits = torch.zeros(stimulus.shape[0], 3, device=stimulus.device)
        else:
            q_pred, pi_pred_raw, state_logits, h_v2 = self.v2(
                r_l4, state.r_l23, cue, task_state, state.h_v2
            )
            p_cw = torch.full((stimulus.shape[0], 1), 0.5, device=stimulus.device)

        # Effective precision for V1 feedback (scaled by warmup ramp during training)
        pi_pred_eff = pi_pred_raw * self.feedback_scale

        # 4. Deep template (uses effective precision)
        deep_tmpl = self.deep_template(q_pred, pi_pred_eff)

        # 5-6. Feedback pathway (branched by mode)
        if self.cfg.feedback_mode == 'emergent':
            # Emergent: learned operator outputs SOM + VIP drives + apical gain
            som_drive, vip_drive, apical_gain = self.feedback(q_pred, pi_pred_eff, r_l4=None)
            r_vip = self.vip(vip_drive, state.r_vip)
            # VIP inhibits SOM: reduce SOM drive where VIP is active
            effective_som_drive = F.relu(som_drive - F.softplus(self.w_vip_som) * r_vip)
            center_exc = self.w_template_drive * deep_tmpl
            r_som = self.som(effective_som_drive, state.r_som)
            l4_to_l23 = r_l4  # No error signal in emergent mode
            r_l23 = self.l23(l4_to_l23, state.r_l23, center_exc, r_som, r_pv,
                             apical_gain=apical_gain)
        else:
            # Fixed: mechanism-specific computation (VIP not active)
            r_vip = state.r_vip  # pass through unchanged
            som_drive = self.feedback.compute_som_drive(q_pred, pi_pred_eff)
            r_som = self.som(som_drive, state.r_som)
            template_modulation = self.feedback.compute_center_excitation(q_pred, pi_pred_eff)
            l4_to_l23 = self.feedback.compute_error_signal(r_l4, deep_tmpl)
            r_l23 = self.l23(l4_to_l23, state.r_l23, template_modulation, r_som, r_pv)

        new_state = NetworkState(
            r_l4=r_l4,
            r_l23=r_l23,
            r_pv=r_pv,
            r_som=r_som,
            r_vip=r_vip,
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
                 p_cw_all [B, T, 1].
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
            state = initial_state(
                B, self.cfg.n_orientations, self.cfg.v2_hidden_dim, device=device
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

        # Check if oracle mode uses sequence-length tensors [B, T, ...]
        _oracle_seq = (self.oracle_mode and self.oracle_q_pred is not None
                       and self.oracle_q_pred.dim() == 3)

        # Cache kernels once for all timesteps (params don't change within forward)
        self.l23.cache_kernels()
        if hasattr(self.feedback, 'cache_kernels'):
            self.feedback.cache_kernels()
        try:
            for t in range(T):
                # Pre-slice oracle tensors per-timestep (avoids Python int counter
                # on module that causes torch.compile recompilation)
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
        finally:
            self.l23.uncache_kernels()
            if hasattr(self.feedback, 'uncache_kernels'):
                self.feedback.uncache_kernels()

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
        }

        return r_l23_all, state, aux
