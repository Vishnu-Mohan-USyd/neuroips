"""Composite loss for the laminar V1-V2 model.

Components:
    1. Sensory readout: cross-entropy on L2/3 -> decoded orientation
    2. Prediction: cross-entropy on V2 q_pred -> next orientation
    3. Energy cost: L1 on population rates (E_excitatory and E_total variants)
    4. Homeostasis: penalizes mean L2/3 rate outside target range
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, TrainingConfig


class CompositeLoss(nn.Module):
    """Multi-objective loss for the V1-V2 network.

    L = lambda_sensory * sensory_readout_loss    (decode current orientation from L2/3)
      + lambda_pred   * prediction_loss           (V2 predicts next orientation)
      + lambda_energy * energy_cost               (activity penalty)
      + lambda_homeo  * homeostasis_penalty       (keep mean activity in range)
    """

    def __init__(self, cfg: TrainingConfig, model_cfg: ModelConfig):
        super().__init__()
        self.lambda_sensory = cfg.lambda_sensory   # 1.0
        self.lambda_pred = cfg.lambda_pred         # 0.5
        self.lambda_energy = cfg.lambda_energy     # 0.01
        self.lambda_homeo = cfg.lambda_homeo       # 1.0
        self.lambda_state = cfg.lambda_state       # 0.25

        N = model_cfg.n_orientations  # 36
        self.orient_step = model_cfg.orientation_range / N  # 5.0
        self.n_orient = N

        # Trainable linear decoder: L2/3 activity -> orientation logits
        self.orientation_decoder = nn.Linear(N, N)

        # Homeostasis target range
        self.target_min = 0.05
        self.target_max = 0.5

    def _theta_to_channel(self, theta: Tensor) -> Tensor:
        """Convert orientation degrees [...] to nearest channel index [...] (long)."""
        return (theta / self.orient_step).round().long() % self.n_orient

    def sensory_readout_loss(
        self, r_l23_windows: Tensor, true_theta_windows: Tensor
    ) -> Tensor:
        """Cross-entropy loss for orientation decoding from L2/3.

        Args:
            r_l23_windows: [B, n_windows, N] — L2/3 at readout timepoints.
            true_theta_windows: [B, n_windows] — true orientations in degrees.

        Returns:
            Scalar loss.
        """
        B, W, N = r_l23_windows.shape
        logits = self.orientation_decoder(r_l23_windows.reshape(B * W, N))
        targets = self._theta_to_channel(true_theta_windows).reshape(B * W)
        return F.cross_entropy(logits, targets)

    def prediction_loss(
        self, q_pred_windows: Tensor, true_next_theta_windows: Tensor
    ) -> Tensor:
        """NLL loss for V2 next-orientation prediction.

        Args:
            q_pred_windows: [B, n_windows, N] — V2 predicted dist (softmax).
            true_next_theta_windows: [B, n_windows] — actual next orientation degrees.

        Returns:
            Scalar loss.
        """
        B, W, N = q_pred_windows.shape
        log_q = torch.log(q_pred_windows.reshape(B * W, N) + 1e-8)
        targets = self._theta_to_channel(true_next_theta_windows).reshape(B * W)
        return F.nll_loss(log_q, targets)

    def energy_cost(self, outputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Compute energy costs from network output trajectories.

        Args:
            outputs: dict with 'r_l4' [B,T,N], 'r_l23' [B,T,N],
                     'r_pv' [B,T,1], 'r_som' [B,T,N], 'deep_template' [B,T,N].

        Returns:
            (E_excitatory, E_total)
        """
        e_exc = (
            outputs["r_l4"].abs().mean()
            + outputs["r_l23"].abs().mean()
            + outputs["deep_template"].abs().mean()
        )
        e_inh = (
            outputs["r_pv"].abs().mean()
            + outputs["r_som"].abs().mean()
        )
        return e_exc, e_exc + e_inh

    def homeostasis_penalty(self, r_l23: Tensor) -> Tensor:
        """Penalize mean L2/3 rate outside [target_min, target_max].

        Uses squared penalty for smooth gradients.

        Args:
            r_l23: [B, T, N] or [B, N] — L2/3 rates.

        Returns:
            Scalar penalty.
        """
        mean_act = r_l23.mean()
        penalty = (
            F.relu(mean_act - self.target_max) ** 2
            + F.relu(self.target_min - mean_act) ** 2
        )
        return penalty

    def state_classification_loss(
        self, state_logits_windows: Tensor, true_states_windows: Tensor
    ) -> Tensor:
        """Cross-entropy loss for HMM state classification.

        Args:
            state_logits_windows: [B, n_windows, 3] — raw logits for CW/CCW/neutral.
            true_states_windows: [B, n_windows] — true HMM state indices (long).

        Returns:
            Scalar loss.
        """
        B, W, C = state_logits_windows.shape
        return F.cross_entropy(
            state_logits_windows.reshape(B * W, C),
            true_states_windows.reshape(B * W),
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        true_theta_windows: Tensor,
        true_next_theta_windows: Tensor,
        r_l23_windows: Tensor,
        q_pred_windows: Tensor,
        state_logits_windows: Tensor | None = None,
        true_states_windows: Tensor | None = None,
        use_e_total: bool = True,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute composite loss.

        Args:
            outputs: full dict from network forward with rate trajectories.
            true_theta_windows: [B, n_windows] current orientations in degrees.
            true_next_theta_windows: [B, n_windows] next orientations in degrees.
            r_l23_windows: [B, n_windows, N] L2/3 at readout times.
            q_pred_windows: [B, n_windows, N] V2 predictions at readout times.
            state_logits_windows: [B, n_windows, 3] state logits (optional).
            true_states_windows: [B, n_windows] true HMM states (optional).
            use_e_total: If True, use E_total; else E_excitatory.

        Returns:
            (total_loss, loss_dict) where loss_dict has .item() scalar values.
        """
        l_sens = self.sensory_readout_loss(r_l23_windows, true_theta_windows)
        l_pred = self.prediction_loss(q_pred_windows, true_next_theta_windows)
        e_exc, e_total = self.energy_cost(outputs)
        l_homeo = self.homeostasis_penalty(outputs["r_l23"])

        l_energy = e_total if use_e_total else e_exc

        total = (
            self.lambda_sensory * l_sens
            + self.lambda_pred * l_pred
            + self.lambda_energy * l_energy
            + self.lambda_homeo * l_homeo
        )

        loss_dict = {
            "sensory": l_sens.item(),
            "prediction": l_pred.item(),
            "energy_exc": e_exc.item(),
            "energy_total": e_total.item(),
            "homeostasis": l_homeo.item(),
        }

        # State classification loss (Fix 4)
        if state_logits_windows is not None and true_states_windows is not None:
            l_state = self.state_classification_loss(state_logits_windows, true_states_windows)
            total = total + self.lambda_state * l_state
            loss_dict["state"] = l_state.item()
        else:
            loss_dict["state"] = 0.0

        loss_dict["total"] = total.item()

        return total, loss_dict
