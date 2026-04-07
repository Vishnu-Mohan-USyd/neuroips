"""V2 context inference module: GRU-based latent-state inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig


class V2ContextModule(nn.Module):
    """V2 context inference module.

    NOT another orientation ring — a latent-state inference module.
    Inputs: L4 activity and/or L2/3 activity (previous step) + cue + task_state.

    Two modes controlled by cfg.feedback_mode:

    'fixed' (legacy): Outputs q_pred [B, N], pi_pred [B, 1],
        state_logits [B, 3], h_v2 [B, H].

    'emergent' (factorized): Outputs p_cw [B, 1] (sigmoid probability
        that rule is CW), pi_pred [B, 1], h_v2 [B, H].
        q_pred is constructed analytically in network.py from L4 + p_cw.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        n = cfg.n_orientations
        self.v2_input_mode = cfg.v2_input_mode
        self.hidden_dim = cfg.v2_hidden_dim
        self.pi_max = cfg.pi_max
        self.feedback_mode = cfg.feedback_mode
        self.som_regime_gate_enabled = cfg.som_regime_gate_enabled
        self.som_regime_gate_target = cfg.som_regime_gate_target
        self.som_regime_gate_beta = cfg.som_regime_gate_beta

        # Input dimension depends on mode
        if self.v2_input_mode == 'l23':
            input_dim = n + n + 2        # L2/3 + cue + task_state
        elif self.v2_input_mode == 'l4':
            input_dim = n + n + 2        # L4 + cue + task_state
        elif self.v2_input_mode == 'l4_l23':
            input_dim = n + n + n + 2    # L4 + L2/3 + cue + task_state
        else:
            raise ValueError(f"Unknown v2_input_mode: {self.v2_input_mode}")

        self.gru = nn.GRUCell(input_dim, cfg.v2_hidden_dim)

        # Output heads depend on feedback_mode
        if self.feedback_mode == 'emergent':
            # Factorized: binary CW probability + precision
            self.head_p_cw = nn.Linear(cfg.v2_hidden_dim, 1)  # -> sigmoid -> p_cw
            nn.init.constant_(self.head_p_cw.bias, 0.0)  # sigmoid(0) = 0.5 (uninformative)
        else:
            # Legacy: full orientation distribution + state logits
            self.head_q = nn.Linear(cfg.v2_hidden_dim, n)        # -> softmax -> q_pred
            self.head_state = nn.Linear(cfg.v2_hidden_dim, 3)    # -> raw logits

        # Precision head (shared by both modes)
        self.head_pi = nn.Linear(cfg.v2_hidden_dim, 1)       # -> softplus + clamp
        nn.init.constant_(self.head_pi.bias, 0.0)

        # Small scalar context head used only when cue-conditioned SOM gating is enabled.
        self.head_som_regime = nn.Linear(cfg.v2_hidden_dim, 1)
        nn.init.zeros_(self.head_som_regime.weight)
        nn.init.constant_(self.head_som_regime.bias, cfg.som_regime_gate_init_bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Allow older checkpoints to load without the new SOM-gate head keys."""
        # Older checkpoints predate the SOM-regime head entirely. Populate the
        # missing entries from the module's current parameters before the base
        # loader records them as missing keys under strict loading.
        for suffix, param in (
            ("weight", self.head_som_regime.weight),
            ("bias", self.head_som_regime.bias),
        ):
            key = prefix + f"head_som_regime.{suffix}"
            if key not in state_dict:
                state_dict[key] = param.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_som_regime_gate(self, h_v2: Tensor) -> Tensor:
        """Return a mild multiplicative gate on the learned SOM field.

        Args:
            h_v2: [B, H] current V2 hidden state.

        Returns:
            gate: [B, 1], equal to 1.0 when disabled and in [1, 1+beta] when enabled.
        """
        if not self.som_regime_gate_enabled or self.som_regime_gate_beta <= 0.0:
            return torch.ones(h_v2.shape[0], 1, device=h_v2.device, dtype=h_v2.dtype)
        if self.som_regime_gate_target != "alpha_inh":
            raise ValueError(
                f"Unsupported som_regime_gate_target={self.som_regime_gate_target!r}. "
                "Expected 'alpha_inh'."
            )
        gate_strength = torch.sigmoid(self.head_som_regime(h_v2))
        gate = 1.0 + self.som_regime_gate_beta * gate_strength
        assert torch.isfinite(gate).all(), "som regime gate must remain finite"
        assert (gate >= 1.0).all(), "som regime gate must be >= 1.0"
        return gate

    def forward(
        self,
        r_l4: Tensor,
        r_l23_prev: Tensor,
        cue: Tensor,
        task_state: Tensor,
        h_v2_prev: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """One step of V2 context inference.

        Args:
            r_l4: [B, N] -- current L4 rates (stable, pre-feedback).
            r_l23_prev: [B, N] -- L2/3 rates from PREVIOUS timestep.
            cue: [B, N] -- cue input (zeros by default).
            task_state: [B, 2] -- task relevance state.
            h_v2_prev: [B, H] -- previous GRU hidden state.

        Returns (feedback_mode == 'emergent'):
            p_cw: [B, 1] -- probability that rule is CW (sigmoid).
            pi_pred: [B, 1] -- prediction precision in [0, pi_max].
            h_v2: [B, H] -- updated GRU hidden state.

        Returns (feedback_mode == 'fixed'):
            q_pred: [B, N] -- predicted orientation distribution (sums to 1).
            pi_pred: [B, 1] -- prediction precision in [0, pi_max].
            state_logits: [B, 3] -- raw logits for CW/CCW/neutral.
            h_v2: [B, H] -- updated GRU hidden state.
        """
        if self.v2_input_mode == 'l23':
            v2_input = torch.cat([r_l23_prev, cue, task_state], dim=-1)
        elif self.v2_input_mode == 'l4':
            v2_input = torch.cat([r_l4, cue, task_state], dim=-1)
        elif self.v2_input_mode == 'l4_l23':
            v2_input = torch.cat([r_l4, r_l23_prev, cue, task_state], dim=-1)
        h_v2 = self.gru(v2_input, h_v2_prev)  # [B, H]

        pi_pred = torch.clamp(F.softplus(self.head_pi(h_v2)), max=self.pi_max)  # [B, 1]

        if self.feedback_mode == 'emergent':
            p_cw = torch.sigmoid(self.head_p_cw(h_v2))  # [B, 1]
            return p_cw, pi_pred, h_v2
        else:
            q_pred = F.softmax(self.head_q(h_v2), dim=-1)  # [B, N]
            state_logits = self.head_state(h_v2)             # [B, 3]
            return q_pred, pi_pred, state_logits, h_v2
