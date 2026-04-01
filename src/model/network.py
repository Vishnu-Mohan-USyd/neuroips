"""LaminarV1V2Network: top-level composer for the laminar V1-V2 model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.config import ModelConfig
from src.state import NetworkState, StepAux, initial_state
from src.model.populations import V1L4Ring, PVPool, V1L23Ring, DeepTemplate, SOMRing
from src.model.v2_context import V2ContextModule
from src.model.feedback import FeedbackMechanism


class LaminarV1V2Network(nn.Module):
    """Complete laminar V1-V2 network.

    Composes: V1L4Ring, PVPool, V1L23Ring, DeepTemplate, SOMRing,
              V2ContextModule, FeedbackMechanism.

    Dependency order per timestep:
        L4 -> PV -> V2 (uses L2/3_{t-1}) -> template -> SOM -> L2/3
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.l4 = V1L4Ring(cfg)
        self.pv = PVPool(cfg)
        self.l23 = V1L23Ring(cfg)
        self.deep_template = DeepTemplate(cfg)
        self.som = SOMRing(cfg)
        self.v2 = V2ContextModule(cfg)
        self.feedback = FeedbackMechanism(cfg)
        # Feedback warmup scale: registered as buffer so torch.compile can see it
        self.register_buffer("feedback_scale", torch.tensor(1.0))

    def step(
        self,
        stimulus: Tensor,
        cue: Tensor,
        task_state: Tensor,
        state: NetworkState,
    ) -> tuple[NetworkState, StepAux]:
        """One timestep of the full network.

        Args:
            stimulus: [B, N] — population-coded, contrast-scaled grating.
            cue: [B, N] — cue input (zeros by default).
            task_state: [B, 2] — task relevance state.
            state: Previous NetworkState.

        Returns:
            new_state: Updated NetworkState.
            aux: StepAux with q_pred, pi_pred, pi_pred_eff, state_logits.
        """
        # 1. L4 (uses PV from previous step)
        r_l4, adaptation = self.l4(stimulus, state.r_l4, state.r_pv, state.adaptation)

        # 2. PV (uses new L4, old L2/3)
        r_pv = self.pv(r_l4, state.r_l23, state.r_pv)

        # 3. V2 (uses old L2/3 — one-step feedback delay)
        q_pred, pi_pred_raw, state_logits, h_v2 = self.v2(
            state.r_l23, cue, task_state, state.h_v2
        )

        # Effective precision for V1 feedback (scaled by warmup ramp during training)
        pi_pred_eff = pi_pred_raw * self.feedback_scale

        # 4. Deep template (uses effective precision)
        deep_tmpl = self.deep_template(q_pred, pi_pred_eff)

        # 5. SOM (mechanism-dependent drive, uses effective precision)
        som_drive = self.feedback.compute_som_drive(q_pred, pi_pred_eff)
        r_som = self.som(som_drive, state.r_som)

        # 6. L2/3 (mechanism-dependent inputs, uses effective precision)
        template_modulation = self.feedback.compute_center_excitation(q_pred, pi_pred_eff)
        l4_to_l23 = self.feedback.compute_error_signal(r_l4, deep_tmpl)
        r_l23 = self.l23(l4_to_l23, state.r_l23, template_modulation, r_som, r_pv)

        new_state = NetworkState(
            r_l4=r_l4,
            r_l23=r_l23,
            r_pv=r_pv,
            r_som=r_som,
            adaptation=adaptation,
            h_v2=h_v2,
            deep_template=deep_tmpl,
        )

        aux = StepAux(
            q_pred=q_pred,
            pi_pred=pi_pred_raw,
            pi_pred_eff=pi_pred_eff,
            state_logits=state_logits,
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
            packed_input: [B, T, N+N+2] — packed stimulus + cue + task_state.
                Use LaminarV1V2Network.pack_inputs() to create this, or pass
                a raw [B, T, N] stimulus (cue and task_state default to zeros).
            state: Initial NetworkState or None (defaults to zeros).

        Returns:
            r_l23_all: [B, T, N] — L2/3 trajectory.
            final_state: NetworkState after last timestep.
            aux: dict with stacked trajectories:
                 q_pred_all [B, T, N], pi_pred_all [B, T, 1],
                 state_logits_all [B, T, 3], deep_template_all [B, T, N],
                 r_l4_all [B, T, N], r_pv_all [B, T, 1], r_som_all [B, T, N].
        """
        N = self.cfg.n_orientations

        # Support both packed [B, T, N+N+2] and unpacked [B, T, N] inputs
        if packed_input.shape[-1] == N:
            # Raw stimulus only — generate zero cue and task_state
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
        q_pred_all = torch.empty(B, T, N, device=device)
        pi_pred_all = torch.empty(B, T, 1, device=device)
        pi_pred_eff_all = torch.empty(B, T, 1, device=device)
        state_logits_all = torch.empty(B, T, 3, device=device)
        deep_template_all = torch.empty(B, T, N, device=device)

        # Cache kernels once for all timesteps (params don't change within forward)
        self.l23.cache_kernels()
        self.feedback.cache_kernels()
        try:
            for t in range(T):
                state, aux_t = self.step(
                    stimulus_seq[:, t], cue_seq[:, t], task_state_seq[:, t], state
                )
                r_l23_all[:, t] = state.r_l23
                r_l4_all[:, t] = state.r_l4
                r_pv_all[:, t] = state.r_pv
                r_som_all[:, t] = state.r_som
                q_pred_all[:, t] = aux_t.q_pred
                pi_pred_all[:, t] = aux_t.pi_pred
                pi_pred_eff_all[:, t] = aux_t.pi_pred_eff
                state_logits_all[:, t] = aux_t.state_logits
                deep_template_all[:, t] = state.deep_template
        finally:
            self.l23.uncache_kernels()
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
        }

        return r_l23_all, state, aux
