"""Composite loss for the laminar V1-V2 model.

Components:
    1. Sensory readout: cross-entropy on L2/3 -> decoded orientation
    2. Prediction: cross-entropy on V2 q_pred -> next orientation (fixed mode)
    3. State BCE: binary cross-entropy on p_cw vs true CW label (emergent mode)
    4. State classification: cross-entropy on state_logits (fixed mode)
    5. Energy cost: L1 on population rates
    6. Homeostasis: penalizes mean L2/3 rate outside target range
    7. Feedback sparsity: L1 on emergent operator weights (emergent mode)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, TrainingConfig


class CompositeLoss(nn.Module):
    """Multi-objective loss for the V1-V2 network.

    Supports both feedback modes:

    Fixed mode:
        L = lambda_sensory * sensory + lambda_pred * prediction
          + lambda_state * state_CE + lambda_energy * energy
          + lambda_homeo * homeostasis

    Emergent mode:
        L = lambda_sensory * sensory + lambda_state * BCE(p_cw, true_cw)
          + lambda_energy * energy + lambda_homeo * homeostasis
          + lambda_fb * (|alpha_inh|_1 + |alpha_exc|_1)
    """

    def __init__(self, cfg: TrainingConfig, model_cfg: ModelConfig):
        super().__init__()
        self.lambda_sensory = cfg.lambda_sensory   # 1.0
        self.lambda_pred = cfg.lambda_pred         # 0.5
        self.lambda_energy = cfg.lambda_energy     # 0.01
        self.lambda_homeo = cfg.lambda_homeo       # 1.0
        self.lambda_state = cfg.lambda_state       # 0.25
        self.lambda_fb = cfg.lambda_fb             # 0.01
        self.lambda_surprise = cfg.lambda_surprise   # 0.0 (disabled by default)
        self.lambda_error = cfg.lambda_error         # 0.0 (disabled by default)
        self.lambda_detection = cfg.lambda_detection # 0.0 (disabled by default)

        self.feedback_mode = model_cfg.feedback_mode

        N = model_cfg.n_orientations  # 36
        self.orient_step = model_cfg.orientation_range / N  # 5.0
        self.n_orient = N
        self.period = model_cfg.orientation_range  # 180.0

        # Trainable linear decoder: L2/3 activity -> orientation logits
        self.orientation_decoder = nn.Linear(N, N)

        # Surprise detection head: L2/3 -> binary (expected vs unexpected)
        if self.lambda_surprise > 0:
            self.surprise_detector = nn.Linear(N, 1)

        # Prediction error readout: L2/3 -> 12 error bins (Experiment A)
        if self.lambda_error > 0:
            self.error_decoder = nn.Linear(N, 12)

        # Detection head: L2/3 -> binary "is expected orientation present?" (Experiment C)
        if self.lambda_detection > 0:
            self.detection_head = nn.Linear(N, 1)

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
            r_l23_windows: [B, n_windows, N] -- L2/3 at readout timepoints.
            true_theta_windows: [B, n_windows] -- true orientations in degrees.

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
        """KL divergence between q_pred and a circular Gaussian target distribution.

        Only used in fixed feedback mode. V2 gets partial credit for being close.

        Args:
            q_pred_windows: [B, n_windows, N] -- V2 predicted dist (softmax).
            true_next_theta_windows: [B, n_windows] -- actual next orientation degrees.

        Returns:
            Scalar loss.
        """
        B, W, N = q_pred_windows.shape

        # Build soft target: circular Gaussian centered on true next orientation
        step = self.orient_step  # 5.0
        preferred = torch.arange(N, device=q_pred_windows.device).float() * step  # [N]

        # Circular distance from each channel to the true next orientation
        true_theta = true_next_theta_windows.unsqueeze(-1)  # [B, W, 1]
        dists = torch.abs(preferred.unsqueeze(0).unsqueeze(0) - true_theta)  # [B, W, N]
        dists = torch.min(dists, 180.0 - dists)  # circular distance

        sigma_target = 10.0  # ~2 channels width
        target_dist = torch.exp(-dists**2 / (2 * sigma_target**2))
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)  # normalise

        # KL divergence: sum_i target_i * log(target_i / q_pred_i)
        log_q = torch.log(q_pred_windows.reshape(B * W, N) + 1e-8)
        kl = F.kl_div(log_q, target_dist.reshape(B * W, N), reduction='batchmean', log_target=False)

        return kl

    def state_bce_loss(
        self, p_cw_windows: Tensor, true_states_windows: Tensor
    ) -> Tensor:
        """Binary cross-entropy loss on p_cw vs true CW/CCW label.

        Used in emergent mode where V2 outputs p_cw (probability of CW).

        Args:
            p_cw_windows: [B, n_windows, 1] -- predicted CW probability (sigmoid).
            true_states_windows: [B, n_windows] -- true HMM state indices
                (0=CW, 1=CCW; in 2-state mode).

        Returns:
            Scalar loss.
        """
        # Target: 1.0 for CW (state==0), 0.0 for CCW (state==1)
        target = (true_states_windows == 0).float().unsqueeze(-1)  # [B, W, 1]
        return F.binary_cross_entropy(p_cw_windows, target)

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
            r_l23: [B, T, N] or [B, N] -- L2/3 rates.

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

        Used in fixed mode only.

        Args:
            state_logits_windows: [B, n_windows, 3] -- raw logits for CW/CCW/neutral.
            true_states_windows: [B, n_windows] -- true HMM state indices (long).

        Returns:
            Scalar loss.
        """
        B, W, C = state_logits_windows.shape
        return F.cross_entropy(
            state_logits_windows.reshape(B * W, C),
            true_states_windows.reshape(B * W),
        )

    def surprise_detection_loss(
        self, r_l23_windows: Tensor, is_expected: Tensor
    ) -> Tensor:
        """BCE loss on a binary 'expected vs unexpected' classifier reading L2/3.

        Args:
            r_l23_windows: [B, W, N] -- L2/3 at readout timepoints.
            is_expected: [B, W] -- binary (1=expected, 0=unexpected).

        Returns:
            Scalar loss.
        """
        B, W, N = r_l23_windows.shape
        logits = self.surprise_detector(r_l23_windows.reshape(B * W, N))  # [B*W, 1]
        targets = is_expected.reshape(B * W, 1).float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def prediction_error_readout_loss(
        self, r_l23_windows: Tensor, true_theta: Tensor, predicted_theta: Tensor
    ) -> Tensor:
        """L2/3 must encode the prediction error, not the stimulus itself.

        Args:
            r_l23_windows: [B, W, N] -- L2/3 at readout timepoints.
            true_theta: [B, W] -- true orientations in degrees.
            predicted_theta: [B, W] -- predicted next orientations in degrees.

        Returns:
            Scalar cross-entropy loss over 12 error bins.
        """
        from src.utils import circular_distance
        B, W, N = r_l23_windows.shape
        # Signed angular error in degrees
        error = circular_distance(true_theta, predicted_theta, self.period)  # [B, W]
        # Discretize into 12 bins: [-90, -75, ..., -15, 0, 15, ..., 75, 90] -> [0..11]
        error_channel = (error / 15.0).round().long()  # ~ [-6, 6]
        error_channel = (error_channel + 6) % 12  # shift to [0, 11]
        logits = self.error_decoder(r_l23_windows.reshape(B * W, N))  # [B*W, 12]
        return F.cross_entropy(logits, error_channel.reshape(B * W))

    def detection_confirmation_loss(
        self, r_l23_windows: Tensor, is_expected: Tensor
    ) -> Tensor:
        """Binary: is the expected orientation present in this stimulus?

        Args:
            r_l23_windows: [B, W, N] -- L2/3 at readout timepoints.
            is_expected: [B, W] -- binary (1=expected, 0=unexpected).

        Returns:
            Scalar BCE loss.
        """
        B, W, N = r_l23_windows.shape
        logits = self.detection_head(r_l23_windows.reshape(B * W, N))  # [B*W, 1]
        targets = is_expected.reshape(B * W, 1).float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def feedback_sparsity_loss(self, model: nn.Module) -> Tensor:
        """L1 sparsity penalty on emergent feedback operator weights.

        Args:
            model: The network (must have feedback.alpha_inh and feedback.alpha_exc).

        Returns:
            Scalar L1 penalty.
        """
        if not hasattr(model, 'feedback') or not hasattr(model.feedback, 'alpha_inh'):
            return torch.tensor(0.0)
        return model.feedback.alpha_inh.abs().sum() + model.feedback.alpha_exc.abs().sum()

    def forward(
        self,
        outputs: dict[str, Tensor],
        true_theta_windows: Tensor,
        true_next_theta_windows: Tensor,
        r_l23_windows: Tensor,
        q_pred_windows: Tensor,
        state_logits_windows: Tensor | None = None,
        true_states_windows: Tensor | None = None,
        p_cw_windows: Tensor | None = None,
        model: nn.Module | None = None,
        is_expected: Tensor | None = None,
        predicted_theta: Tensor | None = None,
        use_e_total: bool = True,
        fb_scale: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute composite loss.

        Args:
            outputs: full dict from network forward with rate trajectories.
            true_theta_windows: [B, n_windows] current orientations in degrees.
            true_next_theta_windows: [B, n_windows] next orientations in degrees.
            r_l23_windows: [B, n_windows, N] L2/3 at readout times.
            q_pred_windows: [B, n_windows, N] V2 predictions at readout times.
            state_logits_windows: [B, n_windows, 3] state logits (fixed mode).
            true_states_windows: [B, n_windows] true HMM states.
            p_cw_windows: [B, n_windows, 1] CW probability (emergent mode).
            model: Network module (for feedback sparsity loss).
            use_e_total: If True, use E_total; else E_excitatory.
            fb_scale: Feedback scale (0 during burn-in, ramps to 1). Scales
                L1 sparsity penalty to prevent alpha death during burn-in.

        Returns:
            (total_loss, loss_dict) where loss_dict has .item() scalar values.
        """
        l_sens = self.sensory_readout_loss(r_l23_windows, true_theta_windows)
        e_exc, e_total = self.energy_cost(outputs)
        l_homeo = self.homeostasis_penalty(outputs["r_l23"])
        l_energy = e_total if use_e_total else e_exc

        total = (
            self.lambda_sensory * l_sens
            + self.lambda_energy * l_energy
            + self.lambda_homeo * l_homeo
        )

        loss_dict = {
            "sensory": l_sens.item(),
            "energy_exc": e_exc.item(),
            "energy_total": e_total.item(),
            "homeostasis": l_homeo.item(),
        }

        if self.feedback_mode == 'emergent':
            # Emergent mode: BCE on p_cw + feedback sparsity
            loss_dict["prediction"] = 0.0  # No prediction KL in emergent mode

            if p_cw_windows is not None and true_states_windows is not None:
                l_state = self.state_bce_loss(p_cw_windows, true_states_windows)
                total = total + self.lambda_state * l_state
                loss_dict["state"] = l_state.item()
            else:
                loss_dict["state"] = 0.0

            if model is not None:
                l_fb = self.feedback_sparsity_loss(model)
                total = total + self.lambda_fb * fb_scale * l_fb
                loss_dict["fb_sparsity"] = l_fb.item()
            else:
                loss_dict["fb_sparsity"] = 0.0
        else:
            # Fixed mode: prediction KL + state classification
            l_pred = self.prediction_loss(q_pred_windows, true_next_theta_windows)
            total = total + self.lambda_pred * l_pred
            loss_dict["prediction"] = l_pred.item()

            if state_logits_windows is not None and true_states_windows is not None:
                l_state = self.state_classification_loss(state_logits_windows, true_states_windows)
                total = total + self.lambda_state * l_state
                loss_dict["state"] = l_state.item()
            else:
                loss_dict["state"] = 0.0

            loss_dict["fb_sparsity"] = 0.0

        # Surprise detection loss (when enabled)
        if self.lambda_surprise > 0 and is_expected is not None:
            l_surprise = self.surprise_detection_loss(r_l23_windows, is_expected)
            total = total + self.lambda_surprise * l_surprise
            loss_dict["surprise"] = l_surprise.item()
        else:
            loss_dict["surprise"] = 0.0

        # Prediction error readout loss (Experiment A)
        if self.lambda_error > 0 and predicted_theta is not None:
            l_error = self.prediction_error_readout_loss(
                r_l23_windows, true_theta_windows, predicted_theta
            )
            total = total + self.lambda_error * l_error
            loss_dict["error_readout"] = l_error.item()
        else:
            loss_dict["error_readout"] = 0.0

        # Detection confirmation loss (Experiment C)
        if self.lambda_detection > 0 and is_expected is not None:
            l_detect = self.detection_confirmation_loss(r_l23_windows, is_expected)
            total = total + self.lambda_detection * l_detect
            loss_dict["detection"] = l_detect.item()
        else:
            loss_dict["detection"] = 0.0

        loss_dict["total"] = total.item()

        return total, loss_dict
