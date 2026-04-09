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
          + lambda_fb * |alpha_inh|_1
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
        self.lambda_l4_sensory = cfg.lambda_l4_sensory  # 0.0 (disabled by default)
        self.lambda_mismatch = cfg.lambda_mismatch      # 0.0 (disabled by default)
        self.lambda_sharp = cfg.lambda_sharp            # 0.0 (disabled by default)
        self.lambda_local_disc = cfg.lambda_local_disc  # 0.0 (disabled by default)
        self.lambda_pred_suppress = cfg.lambda_pred_suppress  # 0.0 (disabled by default)
        self.lambda_fb_energy = cfg.lambda_fb_energy          # 0.0 (disabled by default)
        self.l2_energy = cfg.l2_energy                        # False (L1 by default)

        self.feedback_mode = model_cfg.feedback_mode

        N = model_cfg.n_orientations  # 36
        self.orient_step = model_cfg.orientation_range / N  # 5.0
        self.n_orient = N
        self.period = model_cfg.orientation_range  # 180.0

        # Trainable linear decoder: L2/3 activity -> orientation logits
        self.orientation_decoder = nn.Linear(N, N)

        # L4 sensory decoder: L4 activity -> orientation logits
        if self.lambda_l4_sensory > 0:
            self.l4_decoder = nn.Linear(N, N)

        # L2/3 mismatch detection head: L2/3 -> binary (expected vs deviant)
        if self.lambda_mismatch > 0:
            self.mismatch_head = nn.Linear(N, 1)

        # Surprise detection head: L2/3 -> binary (expected vs unexpected)
        if self.lambda_surprise > 0:
            self.surprise_detector = nn.Linear(N, 1)

        # Prediction error readout: L2/3 -> 12 error bins (Experiment A)
        if self.lambda_error > 0:
            self.error_decoder = nn.Linear(N, 12)

        # Detection head: L2/3 -> binary "is expected orientation present?" (Experiment C)
        if self.lambda_detection > 0:
            self.detection_head = nn.Linear(N, 1)

        # Local competitor discrimination head: 7 channels (center ±3 = ±15°) -> 7 classes (Phase 4)
        if self.lambda_local_disc > 0:
            self.local_disc_head = nn.Linear(7, 7)

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

    def l4_sensory_readout_loss(
        self, r_l4_windows: Tensor, true_theta_windows: Tensor
    ) -> Tensor:
        """Cross-entropy loss for orientation decoding from L4.

        Args:
            r_l4_windows: [B, n_windows, N] -- L4 at readout timepoints.
            true_theta_windows: [B, n_windows] -- true orientations in degrees.

        Returns:
            Scalar loss.
        """
        B, W, N = r_l4_windows.shape
        logits = self.l4_decoder(r_l4_windows.reshape(B * W, N))
        targets = self._theta_to_channel(true_theta_windows).reshape(B * W)
        return F.cross_entropy(logits, targets)

    def mismatch_detection_loss(
        self,
        r_l23_windows: Tensor,
        mismatch_labels: Tensor,
        mismatch_mask: Tensor | None = None,
    ) -> Tensor:
        """Weighted BCE loss for mismatch detection from L2/3.

        Args:
            r_l23_windows: [B, n_windows, N] -- L2/3 at readout timepoints.
            mismatch_labels: [B, n_windows] -- binary (1=mismatch, 0=expected).
            mismatch_mask: [B, n_windows] -- mask (1=valid, 0=exclude).
                Used to exclude first presentation where no prediction exists.

        Returns:
            Scalar loss.
        """
        B, W, N = r_l23_windows.shape
        logits = self.mismatch_head(r_l23_windows.reshape(B * W, N))  # [B*W, 1]
        targets = mismatch_labels.reshape(B * W, 1).float()

        # Compute pos_weight for class imbalance (~24% mismatch → weight ~3.2)
        n_pos = targets.sum().clamp(min=1.0)
        n_neg = (1.0 - targets).sum().clamp(min=1.0)
        pos_weight = n_neg / n_pos

        raw_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )  # [B*W, 1]

        if mismatch_mask is not None:
            mask = mismatch_mask.reshape(B * W, 1).float()
            return (raw_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return raw_loss.mean()

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

        Legacy: used in emergent mode when V2 output p_cw. Retained for
        backward compatibility with fixed mode and oracle mode.

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

    def prior_kl_loss(
        self, mu_pred_windows: Tensor, true_next_theta_windows: Tensor
    ) -> Tensor:
        """KL divergence between V2's predicted prior and true next orientation.

        Target: circular Gaussian bump at true next orientation, normalized
        to a proper distribution. V2's mu_pred (softmax output) is treated
        as the predicted distribution.

        Args:
            mu_pred_windows: [B, n_windows, N] -- V2's predicted prior (sums to 1).
            true_next_theta_windows: [B, n_windows] -- true next orientations (degrees).

        Returns:
            Scalar KL divergence loss.
        """
        B, W, N = mu_pred_windows.shape
        step = self.orient_step
        prefs = torch.arange(N, device=mu_pred_windows.device).float() * step  # [N]

        # Build target distribution: Gaussian bump at true next orientation
        true_degs = true_next_theta_windows.unsqueeze(-1)  # [B, W, 1]
        dists = torch.min(
            torch.abs(prefs.unsqueeze(0).unsqueeze(0) - true_degs),
            self.period - torch.abs(prefs.unsqueeze(0).unsqueeze(0) - true_degs)
        )  # [B, W, N]
        target = torch.exp(-dists ** 2 / (2 * 10.0 ** 2))  # sigma=10 deg
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

        # KL(target || mu_pred): F.kl_div expects log-input
        log_mu = (mu_pred_windows + 1e-8).log()
        return F.kl_div(
            log_mu.reshape(-1, N), target.reshape(-1, N),
            reduction='batchmean', log_target=False,
        )

    def tuning_sharpness_loss(
        self, r_l23_windows: Tensor, true_theta_windows: Tensor
    ) -> Tensor:
        """Penalize L2/3 activity proportional to angular distance from stimulus.

        Creates gradient pressure for sharpening: zero penalty at the stimulus
        channel, maximal penalty at the antipode. Through SOM, this pushes the
        feedback operator toward broad/surround inhibition (sharpening).

        Args:
            r_l23_windows: [B, W, N] L2/3 activity at readout windows.
            true_theta_windows: [B, W] true orientation in degrees.
        Returns:
            Scalar loss.
        """
        B, W, N = r_l23_windows.shape
        step = self.orient_step
        prefs = torch.arange(N, device=r_l23_windows.device).float() * step  # [N]
        true_degs = true_theta_windows.unsqueeze(-1)  # [B, W, 1]
        # Circular distance from each channel to the stimulus orientation
        dists = torch.min(
            torch.abs(prefs.unsqueeze(0).unsqueeze(0) - true_degs),
            self.period - torch.abs(prefs.unsqueeze(0).unsqueeze(0) - true_degs)
        )  # [B, W, N]
        # Weight: 0 at stimulus channel, 1 at antipode (90° away for 180° period)
        weight = dists / (self.period / 2)  # [0, 1]
        return (r_l23_windows * weight).mean()

    def local_discrimination_loss(
        self, r_l23_windows: Tensor, true_theta_windows: Tensor
    ) -> Tensor:
        """Local 7-way discrimination: expected channel vs ±1, ±2, ±3 neighbors.

        For each readout window, extract a 7-channel slice centered at the
        true orientation channel (with circular wrap for channels that fall
        off the ends). A dedicated 7-input linear head (`local_disc_head`)
        is trained to identify the center (true) class vs the six nearest
        neighbors (±3 channels = ±15°). This creates direct gradient
        pressure for L2/3 to separate the expected channel from its
        neighbors — aligned with the ambiguous_offset=15° used in training.

        Args:
            r_l23_windows: [B, W, N] L2/3 activity at readout timepoints.
            true_theta_windows: [B, W] true orientation in degrees.

        Returns:
            Scalar cross-entropy loss over the 7 local classes.
        """
        B, W, N = r_l23_windows.shape
        # Center channel for each (B, W) trial.
        c = self._theta_to_channel(true_theta_windows)  # [B, W], long
        # Offsets [-3, -2, -1, 0, 1, 2, 3] -- neighbours in orientation space.
        offsets = torch.tensor(
            [-3, -2, -1, 0, 1, 2, 3], device=r_l23_windows.device, dtype=torch.long,
        )
        # Broadcast to per-trial channel indices, wrapping circularly.
        # channels: [B, W, 7] -- indices into the N-channel L2/3 axis.
        channels = (c.unsqueeze(-1) + offsets.view(1, 1, 7)) % N
        # Gather activities at those 7 channels for every trial.
        r_flat = r_l23_windows.reshape(B * W, N)                 # [BW, N]
        ch_flat = channels.reshape(B * W, 7)                     # [BW, 7]
        local_r = torch.gather(r_flat, dim=1, index=ch_flat)     # [BW, 7]
        # Logits over the 7 classes.
        logits = self.local_disc_head(local_r)                   # [BW, 7]
        # Target is class 3 (the center / true channel).
        targets = torch.full(
            (B * W,), 3, device=r_l23_windows.device, dtype=torch.long,
        )
        return F.cross_entropy(logits, targets)

    def prediction_suppression_loss(
        self, r_l23_windows: Tensor, q_pred_windows: Tensor
    ) -> Tensor:
        """Penalize L2/3 activity that matches V2's prediction.

        Encourages dampening at predicted channels — the predictive coding
        objective. dot(r_l23, q_pred) is high when V1 is active where V2
        predicts, so minimizing this pushes V1 to suppress expected activity.

        Args:
            r_l23_windows: [B, W, N] -- L2/3 at readout timepoints.
            q_pred_windows: [B, W, N] -- V2 predicted orientation prior.

        Returns:
            Scalar loss.
        """
        return (r_l23_windows * q_pred_windows).sum(dim=-1).mean()

    def feedback_energy_loss(self, center_exc: Tensor) -> Tensor:
        """Penalize magnitude of excitatory feedback to L2/3.

        Specifically targets the center_exc signal that inflates amplitude.
        Higher penalty → V2 must use inhibitory (SOM) path more.

        Args:
            center_exc: [B, T, N] -- excitatory feedback trajectory.

        Returns:
            Scalar L1 penalty on center_exc magnitude.
        """
        return center_exc.abs().mean()

    def energy_cost(self, outputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Compute energy costs from network output trajectories.

        When self.l2_energy is True, the r_l23 term uses a quadratic (L2)
        penalty instead of L1. This penalizes high-amplitude L2/3 activity
        more aggressively, helping control global amplitude ratio.

        Args:
            outputs: dict with 'r_l4' [B,T,N], 'r_l23' [B,T,N],
                     'r_pv' [B,T,1], 'r_som' [B,T,N], 'deep_template' [B,T,N],
                     and optionally 'r_vip' [B,T,N].

        Returns:
            (E_excitatory, E_total)
        """
        r_l23_energy = outputs["r_l23"].pow(2).mean() if self.l2_energy else outputs["r_l23"].abs().mean()
        e_exc = (
            outputs["r_l4"].abs().mean()
            + r_l23_energy
            + outputs["deep_template"].abs().mean()
        )
        e_inh = (
            outputs["r_pv"].abs().mean()
            + outputs["r_som"].abs().mean()
        )
        if "r_vip" in outputs:
            e_inh = e_inh + outputs["r_vip"].abs().mean()
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

        Penalizes both SOM (alpha_inh) and VIP (alpha_vip) pathways.
        Additionally penalizes VIP magnitude exceeding SOM magnitude
        (norm-matching) to prevent runaway disinhibition.

        Args:
            model: The network (must have feedback.alpha_inh / alpha_vip).

        Returns:
            Scalar L1 penalty.
        """
        if not hasattr(model, 'feedback') or not hasattr(model.feedback, 'alpha_inh'):
            return torch.tensor(0.0)
        fb = model.feedback
        l1_inh = fb.alpha_inh.abs().sum()
        total = l1_inh
        if hasattr(fb, 'alpha_vip'):
            l1_vip = fb.alpha_vip.abs().sum()
            # Norm-matching: penalize VIP exceeding SOM magnitude
            vip_excess = F.relu(l1_vip - l1_inh)
            total = total + l1_vip + vip_excess
        if hasattr(fb, 'alpha_apical'):
            total = total + fb.alpha_apical.abs().sum()
        return total

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
        r_l4_windows: Tensor | None = None,
        mismatch_labels: Tensor | None = None,
        mismatch_mask: Tensor | None = None,
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
            # Emergent mode: prior KL on mu_pred (= q_pred) + feedback sparsity
            loss_dict["prediction"] = 0.0

            # Prior KL loss: q_pred_windows IS mu_pred in learned-prior mode
            if self.lambda_state > 0 and true_states_windows is not None:
                # true_states_windows is used as a flag: if not None, prior KL is active.
                # The actual target is true_next_theta_windows (passed via true_next_theta_windows arg).
                l_prior_kl = self.prior_kl_loss(q_pred_windows, true_next_theta_windows)
                total = total + self.lambda_state * l_prior_kl
                loss_dict["state"] = l_prior_kl.item()
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

        # L4 sensory readout loss (deviance objective)
        if self.lambda_l4_sensory > 0 and r_l4_windows is not None:
            l_l4_sens = self.l4_sensory_readout_loss(r_l4_windows, true_theta_windows)
            total = total + self.lambda_l4_sensory * l_l4_sens
            loss_dict["l4_sensory"] = l_l4_sens.item()
        else:
            loss_dict["l4_sensory"] = 0.0

        # L2/3 mismatch detection loss (deviance objective)
        if self.lambda_mismatch > 0 and mismatch_labels is not None:
            l_mismatch = self.mismatch_detection_loss(
                r_l23_windows, mismatch_labels, mismatch_mask
            )
            total = total + self.lambda_mismatch * l_mismatch
            loss_dict["mismatch"] = l_mismatch.item()
        else:
            loss_dict["mismatch"] = 0.0

        # Tuning sharpness loss (penalize flank activity)
        if self.lambda_sharp > 0:
            l_sharp = self.tuning_sharpness_loss(r_l23_windows, true_theta_windows)
            total = total + self.lambda_sharp * l_sharp
            loss_dict["sharp"] = l_sharp.item()
        else:
            loss_dict["sharp"] = 0.0

        # Phase 4: local competitor discrimination loss (expected vs +- 1, +- 2)
        if self.lambda_local_disc > 0:
            l_local = self.local_discrimination_loss(r_l23_windows, true_theta_windows)
            total = total + self.lambda_local_disc * l_local
            loss_dict["local_disc"] = l_local.item()
        else:
            loss_dict["local_disc"] = 0.0

        # Prediction suppression: penalize L2/3 activity matching V2 prediction
        if self.lambda_pred_suppress > 0:
            l_pred_sup = self.prediction_suppression_loss(r_l23_windows, q_pred_windows)
            total = total + self.lambda_pred_suppress * l_pred_sup
            loss_dict["pred_suppress"] = l_pred_sup.item()
        else:
            loss_dict["pred_suppress"] = 0.0

        # Feedback energy: penalize excitatory feedback magnitude
        if self.lambda_fb_energy > 0 and "center_exc" in outputs:
            l_fb_energy = self.feedback_energy_loss(outputs["center_exc"])
            total = total + self.lambda_fb_energy * l_fb_energy
            loss_dict["fb_energy"] = l_fb_energy.item()
        else:
            loss_dict["fb_energy"] = 0.0

        loss_dict["total"] = total.item()

        return total, loss_dict
