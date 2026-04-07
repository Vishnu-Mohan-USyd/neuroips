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
        self.lambda_local_rank = cfg.lambda_local_rank  # 0.0 (disabled by default)
        self.local_rank_offsets_deg = tuple(cfg.local_rank_offsets_deg)
        self.local_rank_margins = tuple(cfg.local_rank_margins)
        self.local_rank_weights = tuple(cfg.local_rank_weights)
        self.local_rank_late_weights = tuple(cfg.local_rank_late_weights)
        self.local_rank_ambiguous_only = cfg.local_rank_ambiguous_only

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

        # Local competitor discrimination head: 5 channels (center +- 1, +- 2) -> 5 classes (Phase 4)
        if self.lambda_local_disc > 0:
            self.local_disc_head = nn.Linear(5, 5)

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
        """Local 5-way discrimination: expected channel vs +- 1, +- 2 neighbors.

        For each readout window, extract a 5-channel slice centered at the
        true orientation channel (with circular wrap for channels that fall
        off the ends). A dedicated 5-input linear head (`local_disc_head`)
        is trained to identify the center (true) class vs the four nearest
        neighbors. This creates direct gradient pressure for L2/3 to
        separate the expected channel from its +- 1 and +- 2 neighbors --
        the signature of Kok-style sharpening that 36-way CE does not
        penalise.

        Args:
            r_l23_windows: [B, W, N] L2/3 activity at readout timepoints.
            true_theta_windows: [B, W] true orientation in degrees.

        Returns:
            Scalar cross-entropy loss over the 5 local classes.
        """
        B, W, N = r_l23_windows.shape
        # Center channel for each (B, W) trial.
        c = self._theta_to_channel(true_theta_windows)  # [B, W], long
        # Offsets [-2, -1, 0, 1, 2] -- neighbours in orientation space.
        offsets = torch.tensor(
            [-2, -1, 0, 1, 2], device=r_l23_windows.device, dtype=torch.long,
        )
        # Broadcast to per-trial channel indices, wrapping circularly.
        # channels: [B, W, 5] -- indices into the N-channel L2/3 axis.
        channels = (c.unsqueeze(-1) + offsets.view(1, 1, 5)) % N
        # Gather activities at those 5 channels for every trial.
        r_flat = r_l23_windows.reshape(B * W, N)                 # [BW, N]
        ch_flat = channels.reshape(B * W, 5)                     # [BW, 5]
        local_r = torch.gather(r_flat, dim=1, index=ch_flat)     # [BW, 5]
        # Logits over the 5 classes.
        logits = self.local_disc_head(local_r)                   # [BW, 5]
        # Target is class 2 (the center / true channel).
        targets = torch.full(
            (B * W,), 2, device=r_l23_windows.device, dtype=torch.long,
        )
        return F.cross_entropy(logits, targets)

    def _local_rank_channel_offsets(self) -> tuple[int, ...]:
        """Convert configured local ranking offsets from degrees to channel units."""
        channel_offsets: list[int] = []
        for offset_deg in self.local_rank_offsets_deg:
            channel = int(round(offset_deg / self.orient_step))
            assert channel > 0, (
                f"local_rank_offsets_deg must map to positive channel offsets; got {offset_deg}"
            )
            assert abs(channel * self.orient_step - offset_deg) < 1e-6, (
                f"local_rank offset {offset_deg} deg is not aligned to the orientation grid "
                f"(step={self.orient_step} deg)"
            )
            channel_offsets.append(channel)
        return tuple(channel_offsets)

    def local_ranking_loss(
        self,
        r_l23_late_windows: Tensor,
        true_theta_windows: Tensor,
        ambiguous_windows: Tensor | None = None,
    ) -> Tensor:
        """Margin ranking on raw late L2/3 activity for target > local competitors.

        Args:
            r_l23_late_windows: [B, W, T_late, N] raw late-window L2/3 activity.
            true_theta_windows: [B, W] true target orientation in degrees.
            ambiguous_windows: Optional [B, W] boolean/float mask. When provided
                and `local_rank_ambiguous_only` is True, the ranking loss applies
                only to ambiguous presentations.

        Returns:
            Scalar hinge-style ranking loss.
        """
        B, W, T_late, N = r_l23_late_windows.shape
        assert N == self.n_orient
        assert len(self.local_rank_offsets_deg) == len(self.local_rank_margins) == len(self.local_rank_weights), (
            "local ranking offsets, margins, and weights must have the same length"
        )
        assert len(self.local_rank_late_weights) == T_late, (
            f"local_rank_late_weights length {len(self.local_rank_late_weights)} "
            f"must match late window length {T_late}"
        )

        center = self._theta_to_channel(true_theta_windows)  # [B, W]
        center_idx = center.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_late, 1)
        target_act = torch.gather(r_l23_late_windows, dim=-1, index=center_idx).squeeze(-1)  # [B, W, T]

        late_weights = torch.as_tensor(
            self.local_rank_late_weights,
            dtype=r_l23_late_windows.dtype,
            device=r_l23_late_windows.device,
        ).view(1, 1, T_late)

        if ambiguous_windows is not None and self.local_rank_ambiguous_only:
            valid_mask = ambiguous_windows.to(dtype=r_l23_late_windows.dtype).unsqueeze(-1)
        else:
            valid_mask = torch.ones(B, W, 1, dtype=r_l23_late_windows.dtype, device=r_l23_late_windows.device)

        numerator = torch.zeros((), dtype=r_l23_late_windows.dtype, device=r_l23_late_windows.device)
        denominator = torch.zeros((), dtype=r_l23_late_windows.dtype, device=r_l23_late_windows.device)
        for offset_ch, margin, weight in zip(
            self._local_rank_channel_offsets(),
            self.local_rank_margins,
            self.local_rank_weights,
        ):
            plus_idx = ((center + offset_ch) % N).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_late, 1)
            minus_idx = ((center - offset_ch) % N).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_late, 1)
            plus_act = torch.gather(r_l23_late_windows, dim=-1, index=plus_idx).squeeze(-1)
            minus_act = torch.gather(r_l23_late_windows, dim=-1, index=minus_idx).squeeze(-1)

            hinge_plus = F.relu(margin - (target_act - plus_act))
            hinge_minus = F.relu(margin - (target_act - minus_act))
            pair_loss = 0.5 * (hinge_plus + hinge_minus)
            weighted_loss = pair_loss * late_weights * valid_mask
            numerator = numerator + float(weight) * weighted_loss.sum()
            denominator = denominator + float(weight) * (late_weights * valid_mask).sum()

        if denominator.item() == 0.0:
            return torch.zeros((), dtype=r_l23_late_windows.dtype, device=r_l23_late_windows.device)
        return numerator / denominator

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
            model: The network (must have feedback.alpha_inh).

        Returns:
            Scalar L1 penalty.
        """
        if not hasattr(model, 'feedback') or not hasattr(model.feedback, 'alpha_inh'):
            return torch.tensor(0.0)
        return model.feedback.alpha_inh.abs().sum()

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
        r_l23_late_windows: Tensor | None = None,
        ambiguous_windows: Tensor | None = None,
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

        if self.lambda_local_rank > 0 and r_l23_late_windows is not None:
            l_local_rank = self.local_ranking_loss(
                r_l23_late_windows,
                true_theta_windows,
                ambiguous_windows=ambiguous_windows,
            )
            total = total + self.lambda_local_rank * l_local_rank
            loss_dict["local_rank"] = l_local_rank.item()
        else:
            loss_dict["local_rank"] = 0.0

        loss_dict["total"] = total.item()

        return total, loss_dict
