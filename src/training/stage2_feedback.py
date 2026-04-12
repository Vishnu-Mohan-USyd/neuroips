"""Stage 2: V2 sequence learning + feedback training.

Goal: V2 learns transition statistics. Feedback learns to modulate V1.
Supports both 'fixed' (hardcoded mechanisms) and 'emergent' (learned operator)
feedback modes.

Separate LR groups (V2: 3e-4, feedback: 1e-4), gradient clip 1.0,
linear warmup + cosine decay.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from src.config import ModelConfig, TrainingConfig, StimulusConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.losses import CompositeLoss
from src.utils import circular_distance_abs
from src.training.trainer import (
    freeze_stage1,
    unfreeze_stage2,
    create_stage2_optimizer,
    make_warmup_cosine_scheduler,
    build_stimulus_sequence,
    compute_readout_indices,
    extract_readout_data,
)

logger = logging.getLogger(__name__)


def compute_mismatch_labels(
    metadata,
    transition_step: float = 15.0,
    mismatch_threshold_deg: float = 3.0,
    orientation_range: float = 180.0,
) -> tuple[Tensor, Tensor]:
    """Compute binary mismatch labels from generator ground truth.

    A presentation is "mismatch" if the actual orientation deviates from
    the expected orientation (given previous orientation and current state)
    by more than ``mismatch_threshold_deg`` degrees, or if the state is
    NEUTRAL.

    Args:
        metadata: SequenceMetadata with .orientations [B, S] and .states [B, S].
        transition_step: Expected step size in degrees for CW/CCW.
        mismatch_threshold_deg: Circular distance threshold for mismatch
            (degrees). Default 3.0 — tight enough to catch the 5° jitter
            scale used in the simple-dual-regime sweep. Lower than the old
            10° default (which was calibrated against a 15° transition_step).
        orientation_range: Period of orientation space (degrees).

    Returns:
        mismatch_labels: [B, S] binary labels (1=mismatch, 0=expected).
        mismatch_mask: [B, S] validity mask (0 for first presentation, 1 elsewhere).
    """
    orientations = metadata.orientations  # [B, S]
    states = metadata.states  # [B, S] — current state at each presentation

    prev_theta = orientations[:, :-1]  # [B, S-1]
    curr_theta = orientations[:, 1:]   # [B, S-1]
    curr_state = states[:, 1:]         # [B, S-1]

    # Expected orientation given state: CW → +step, CCW → -step
    expected_cw = (prev_theta + transition_step) % orientation_range
    expected_ccw = (prev_theta - transition_step) % orientation_range
    expected = torch.where(curr_state == 0, expected_cw, expected_ccw)

    # NEUTRAL state (state >= 2) is always mismatch (unpredictable)
    is_neutral = (curr_state >= 2)

    circ_dist = circular_distance_abs(curr_theta, expected, orientation_range)
    mismatch = ((circ_dist > mismatch_threshold_deg) | is_neutral).float()

    # Pad first presentation (no valid prediction for t=0)
    B = orientations.shape[0]
    device = orientations.device
    mismatch_labels = torch.cat([torch.zeros(B, 1, device=device), mismatch], dim=1)
    mismatch_mask = torch.cat([torch.zeros(B, 1, device=device),
                               torch.ones_like(mismatch)], dim=1)
    return mismatch_labels, mismatch_mask


@dataclass
class Stage2Result:
    """Result of Stage 2 training."""
    final_loss: float
    final_sensory_acc: float
    final_pred_acc: float
    loss_history: list[float]
    n_steps_trained: int


def run_stage2(
    net: LaminarV1V2Network,
    loss_fn: CompositeLoss,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    stim_cfg: StimulusConfig,
    device: torch.device | None = None,
    seed: int = 42,
    log_interval: int = 100,
    checkpoint_fn: Callable[[int], None] | None = None,
    checkpoint_steps: list[int] | None = None,
    output_dir: str | None = None,
) -> Stage2Result:
    """Run Stage 2 training: V2 + feedback with BPTT over HMM sequences.

    Args:
        net: LaminarV1V2Network (Stage 1 params should be frozen).
        loss_fn: CompositeLoss (shared with Stage 1 for decoder continuity).
        model_cfg: ModelConfig.
        train_cfg: TrainingConfig.
        stim_cfg: StimulusConfig.
        device: Device to train on.
        seed: Random seed.
        log_interval: Steps between log messages.
        checkpoint_fn: Optional callback called at each checkpoint step.
        checkpoint_steps: List of step numbers at which to call checkpoint_fn.
        output_dir: Directory for writing metrics.jsonl (optional).

    Returns:
        Stage2Result with training metrics.
    """
    dev = device or torch.device("cpu")
    net = net.to(dev)
    loss_fn = loss_fn.to(dev)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    N = model_cfg.n_orientations
    n_steps = train_cfg.stage2_n_steps
    batch_size = train_cfg.batch_size
    seq_length = train_cfg.seq_length
    feedback_mode = model_cfg.feedback_mode

    # Ensure correct freeze/unfreeze state
    freeze_stage1(net)
    unfreeze_stage2(net)

    # Freeze V2 when using oracle mode (must be before optimizer creation)
    if train_cfg.freeze_v2:
        for p in net.v2.parameters():
            p.requires_grad_(False)
        # Fix 1: also freeze v2_routine if present
        if hasattr(net, "v2_routine"):
            for p in net.v2_routine.parameters():
                p.requires_grad_(False)
        logger.info("V2 frozen (oracle/freeze mode): V2 parameters excluded from training")

    # Freeze orientation decoder for mechanistic analysis (must be before optimizer creation)
    if train_cfg.freeze_decoder or train_cfg.freeze_v2:
        for p in loss_fn.orientation_decoder.parameters():
            p.requires_grad_(False)
        logger.info("Froze orientation decoder for mechanistic analysis")

    # Step-level compile: fast (~13s), low RAM (~1.2GB), same throughput at T=600.
    net.step = torch.compile(net.step, mode='max-autotune-no-cudagraphs')
    compiled_net = net

    # Optimizer with separate LR groups
    optimizer = create_stage2_optimizer(net, loss_fn, train_cfg)

    # Scheduler
    scheduler = make_warmup_cosine_scheduler(
        optimizer, train_cfg.stage2_warmup_steps, n_steps
    )

    # HMM sequence generator
    hmm_gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=model_cfg.orientation_range,
        contrast_range=train_cfg.stage2_contrast_range,
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        task_p_switch=stim_cfg.task_p_switch,
    )

    # Readout indices (last 3 steps of ON period — L2/3 needs time to settle)
    window_start = max(0, train_cfg.steps_on - 3)
    window_end = train_cfg.steps_on - 1
    readout_indices = compute_readout_indices(
        seq_length, train_cfg.steps_on, train_cfg.steps_isi,
        window_start=window_start, window_end=window_end,
    )

    loss_history = []
    last_sensory_acc = 0.0
    last_pred_acc = 0.0

    # Reference baselines
    if feedback_mode == 'emergent':
        logger.info(
            "Baselines (emergent): prior_kl uninformative=~3.6 (uniform prior), "
            "sensory uniform=0.028 (1/36)"
        )
    else:
        logger.info(
            "Baselines: uniform=0.028 (1/36), same_as_current~0.20, "
            "oracle_with_state~0.75 (within +/-1 channel)"
        )

    # V2 curriculum: hard zero feedback for predictor burn-in, then ramp 0->1
    predictor_burnin_steps = train_cfg.stage2_burnin_steps
    feedback_ramp_steps = train_cfg.stage2_ramp_steps

    # Freeze-then-unfreeze W_rec gain (aligned with end of burn-in)
    gain_unfreeze_step = predictor_burnin_steps
    net.l23.gain_rec_raw.requires_grad_(False)  # Start frozen

    for step in range(n_steps):
        # V2 curriculum: hard zero during burn-in, then ramp
        if step < predictor_burnin_steps:
            net.feedback_scale.fill_(0.0)
        else:
            ramp_progress = (step - predictor_burnin_steps) / feedback_ramp_steps
            net.feedback_scale.fill_(min(1.0, ramp_progress))

        # Unfreeze gain_rec after burn-in
        if step == gain_unfreeze_step:
            net.l23.gain_rec_raw.requires_grad_(True)
            logger.info(f"Step {step}: unfreezing gain_rec_raw")

        optimizer.zero_grad()

        # Generate HMM sequence batch
        metadata = hmm_gen.generate(batch_size, seq_length, gen)

        # Build temporal stimulus sequence
        stim_seq, cue_seq, task_seq, true_thetas, true_next_thetas, true_states = (
            build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
        )
        # Add stimulus noise if configured (Stage 2 only)
        if train_cfg.stimulus_noise > 0.0:
            stim_seq = stim_seq + train_cfg.stimulus_noise * torch.randn_like(stim_seq)
            stim_seq = stim_seq.clamp(min=0.0)  # firing rates can't be negative

        stim_seq = stim_seq.to(dev, non_blocking=True)
        cue_seq = cue_seq.to(dev, non_blocking=True)
        task_seq = task_seq.to(dev, non_blocking=True)
        true_thetas = true_thetas.to(dev, non_blocking=True)
        true_next_thetas = true_next_thetas.to(dev, non_blocking=True)
        true_states = true_states.to(dev, non_blocking=True)

        # Fix 2: gradient isolation — override task_state so that each
        # isolation_period block sees only one regime. First half = focused
        # (1,0), second half = routine (0,1). HMM stimulus statistics are
        # unchanged; only the task_state routing is overridden.
        if train_cfg.gradient_isolation:
            period = train_cfg.isolation_period
            phase = (step // period) % 2  # 0 = focused, 1 = routine
            if phase == 0:
                override_ts = torch.tensor([1.0, 0.0])
            else:
                override_ts = torch.tensor([0.0, 1.0])
            # Override temporal task_seq [B, T_total, 2]
            task_seq = override_ts.expand_as(task_seq).clone().to(dev)
            # Override metadata.task_states [B, S, 2] for loss computation
            metadata.task_states = override_ts.expand_as(metadata.task_states).clone()

        # Oracle / freeze V2 mode: bypass V2 with ground-truth predictions
        if train_cfg.freeze_v2:
            net.oracle_mode = True
            steps_per_pres = train_cfg.steps_on + train_cfg.steps_isi
            T_total = seq_length * steps_per_pres

            oris = metadata.orientations.to(dev)  # [B, S]
            # Use true_states (shifted next-state from build_stimulus_sequence)
            # because states[t+1] determines the transition from ori[t] to ori[t+1].
            # metadata.states[t] is the state that generated ori[t] from ori[t-1].
            cur_states = true_states  # [B, S] — next state (shifted), already on dev
            step_deg = model_cfg.transition_step
            B_cur = oris.shape[0]
            template_mode = train_cfg.oracle_template

            if template_mode == "oracle_true":
                # Normal oracle: bump at next orientation given TRUE state
                #   CW → theta[s] + transition_step
                #   CCW → theta[s] - transition_step
                #   NEUTRAL → uniform (average of CW and CCW)
                theta_cw = (oris + step_deg) % model_cfg.orientation_range
                theta_ccw = (oris - step_deg) % model_cfg.orientation_range
                q_cw = net._make_bump(
                    theta_cw.reshape(-1), sigma=train_cfg.oracle_sigma
                ).reshape(B_cur, seq_length, N)
                q_ccw = net._make_bump(
                    theta_ccw.reshape(-1), sigma=train_cfg.oracle_sigma
                ).reshape(B_cur, seq_length, N)
                p_cw_oracle = (cur_states == 0).float().unsqueeze(-1)  # [B, S, 1]
                p_ccw_oracle = (cur_states == 1).float().unsqueeze(-1)
                p_neutral = (1.0 - p_cw_oracle - p_ccw_oracle).clamp(min=0)
                q_oracle = (p_cw_oracle * q_cw + p_ccw_oracle * q_ccw
                            + p_neutral * 0.5 * (q_cw + q_ccw))
                q_oracle = q_oracle / (q_oracle.sum(dim=-1, keepdim=True) + 1e-8)

            elif template_mode == "oracle_wrong":
                # Swap CW <-> CCW: use the OPPOSITE state's prediction.
                # Same peakedness and same set of predicted orientations as oracle_true,
                # but the prediction is anti-correlated with the true next stimulus.
                theta_cw = (oris + step_deg) % model_cfg.orientation_range
                theta_ccw = (oris - step_deg) % model_cfg.orientation_range
                q_cw = net._make_bump(
                    theta_cw.reshape(-1), sigma=train_cfg.oracle_sigma
                ).reshape(B_cur, seq_length, N)
                q_ccw = net._make_bump(
                    theta_ccw.reshape(-1), sigma=train_cfg.oracle_sigma
                ).reshape(B_cur, seq_length, N)
                # INVERTED: if state is CW use theta_ccw, if CCW use theta_cw
                p_cw_oracle = (cur_states == 1).float().unsqueeze(-1)  # inverted
                p_ccw_oracle = (cur_states == 0).float().unsqueeze(-1)  # inverted
                p_neutral = (1.0 - p_cw_oracle - p_ccw_oracle).clamp(min=0)
                q_oracle = (p_cw_oracle * q_cw + p_ccw_oracle * q_ccw
                            + p_neutral * 0.5 * (q_cw + q_ccw))
                q_oracle = q_oracle / (q_oracle.sum(dim=-1, keepdim=True) + 1e-8)

            elif template_mode == "oracle_random":
                # Bump at a randomly chosen orientation, independent of stimulus.
                # Same peakedness as oracle_true (via _make_bump) but random center.
                # Fresh random draw per (batch, presentation) — the "prediction"
                # is peaked but uncorrelated with the upcoming stimulus. This is
                # the correct null for "does the shape of the template matter?":
                # a peaked-but-wrong template has the same statistics as a
                # peaked-and-right template.
                random_thetas = (
                    torch.rand(B_cur, seq_length, device=dev, generator=None)
                    * model_cfg.orientation_range
                )
                q_rand = net._make_bump(
                    random_thetas.reshape(-1), sigma=train_cfg.oracle_sigma,
                ).reshape(B_cur, seq_length, N)
                q_oracle = q_rand / (q_rand.sum(dim=-1, keepdim=True) + 1e-8)

            elif template_mode == "oracle_uniform":
                # Flat distribution over orientations — no peak, no structure.
                # The feedback operator receives a uniform template at every
                # presentation. This is the correct null for "does the template
                # need to be peaked at all?".
                q_oracle = torch.full(
                    (B_cur, seq_length, N), 1.0 / N, device=dev
                )

            else:
                raise ValueError(
                    f"Unknown oracle_template: {template_mode!r}. "
                    "Must be one of: oracle_true, oracle_wrong, oracle_random, oracle_uniform."
                )

            # Phase 3: shift q_oracle by +1 along presentation dim so that the
            # oracle built from presentation s (which forecasts s+1) is applied
            # during presentation s+1. This converts the template from a
            # "same-step forecast" (redundant with the current stimulus) into
            # a "prior about the current item" (held over from the previous
            # step). The first presentation has no valid prior, so it receives
            # a uniform distribution.
            if train_cfg.oracle_shift_timing:
                uniform_first = torch.full(
                    (B_cur, 1, N), 1.0 / N, device=dev, dtype=q_oracle.dtype,
                )
                q_oracle = torch.cat([uniform_first, q_oracle[:, :-1, :]], dim=1)

            # Expand to all timesteps: [B, S, N] → [B, T_total, N]
            oracle_q = q_oracle.unsqueeze(2).expand(
                -1, -1, steps_per_pres, -1
            ).reshape(B_cur, T_total, N).to(dev)
            oracle_pi = torch.full(
                (B_cur, T_total, 1), train_cfg.oracle_pi, device=dev,
            )
            net.oracle_q_pred = oracle_q
            net.oracle_pi_pred = oracle_pi

        # Forward pass
        packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
        r_l23_all, final_state, aux = compiled_net(packed)

        # Reset oracle mode after forward pass
        if train_cfg.freeze_v2:
            net.oracle_mode = False
            net.oracle_q_pred = None
            net.oracle_pi_pred = None

        # Build outputs dict for loss computation
        outputs = {
            "r_l23": r_l23_all,
            "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"],
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
            "state_logits": aux["state_logits_all"],
            "p_cw": aux["p_cw_all"],
            "center_exc": aux["center_exc_all"],
            # Phase 2.4: som_drive_fb trajectory exposed for routine_shape loss
            # (per-sample E/I symmetry-break incentive). Always present in aux
            # even when gate is off — zeros feedback produces zero tensor.
            "som_drive_fb": aux["som_drive_fb_all"],
        }

        # Extract readout windows
        r_l23_windows, q_pred_windows, state_logits_windows = extract_readout_data(
            outputs, readout_indices,
            steps_on=train_cfg.steps_on, steps_isi=train_cfg.steps_isi,
        )

        # Extract p_cw windows for emergent mode (placeholder — p_cw is 0.5 in
        # learned-prior mode; retained for backward compat with loss_fn signature)
        p_cw_windows = None
        if feedback_mode == 'emergent':
            steps_per = train_cfg.steps_on + train_cfg.steps_isi
            _, ts_first = readout_indices[0]
            w_start = ts_first[0]
            w_end = ts_first[-1] + 1
            B_batch = aux["p_cw_all"].shape[0]
            S = len(readout_indices)
            p_cw_windows = (
                aux["p_cw_all"]
                .reshape(B_batch, S, steps_per, 1)[:, :, w_start:w_end]
                .mean(dim=2)
            )

        # Extract L4 readout windows (same extraction pattern as L2/3)
        r_l4_windows = None
        if train_cfg.lambda_l4_sensory > 0:
            steps_per = train_cfg.steps_on + train_cfg.steps_isi
            _, ts_first = readout_indices[0]
            w_start = ts_first[0]
            w_end = ts_first[-1] + 1
            B_batch = aux["r_l4_all"].shape[0]
            S = len(readout_indices)
            r_l4_windows = (
                aux["r_l4_all"]
                .reshape(B_batch, S, steps_per, N)[:, :, w_start:w_end]
                .mean(dim=2)
            )

        # Compute mismatch labels from ground truth
        mm_labels_windows = None
        mm_mask_windows = None
        if train_cfg.lambda_mismatch > 0 or train_cfg.lambda_expected_suppress > 0:
            mm_labels, mm_mask = compute_mismatch_labels(
                metadata,
                transition_step=stim_cfg.transition_step,
                mismatch_threshold_deg=3.0,
                orientation_range=model_cfg.orientation_range,
            )
            mm_labels_windows = mm_labels.to(dev)
            mm_mask_windows = mm_mask.to(dev)

        # Compute is_expected for surprise/detection losses (if enabled)
        is_expected = None
        if train_cfg.lambda_surprise > 0 or train_cfg.lambda_detection > 0:
            orient_step = model_cfg.orientation_range / N
            pred_next_ch = q_pred_windows[:, :-1].argmax(dim=-1)  # [B, W-1]
            true_next_ch = loss_fn._theta_to_channel(true_next_thetas[:, :-1])  # [B, W-1]
            ch_dist = (pred_next_ch - true_next_ch).abs() % N
            ch_dist = torch.min(ch_dist, N - ch_dist)
            is_expected_partial = (ch_dist <= 1).long()  # [B, W-1]
            is_expected = torch.cat([
                torch.ones(is_expected_partial.shape[0], 1, device=dev, dtype=torch.long),
                is_expected_partial,
            ], dim=1)  # [B, W]

        # Compute predicted_theta for error readout loss (if enabled)
        predicted_theta = None
        if train_cfg.lambda_error > 0:
            orient_step = model_cfg.orientation_range / N
            predicted_theta = q_pred_windows.argmax(dim=-1).float() * orient_step  # [B, W]

        # Compute loss (scale L1 sparsity by fb_scale to prevent alpha death during burn-in)
        # In oracle/freeze_v2 mode, p_cw is a placeholder (0.5) — skip state BCE loss
        states_for_loss = None if train_cfg.freeze_v2 else true_states
        p_cw_for_loss = None if train_cfg.freeze_v2 else p_cw_windows
        # Simple-dual-regime: per-presentation Markov task_state.
        # metadata.task_states is [B, S, 2] — one-hot regime for each
        # presentation. S == seq_length == W (the number of readout
        # windows). We pass this directly to CompositeLoss for per-
        # presentation gating of sensory / energy / mismatch.
        # CompositeLoss accepts either [B, 2] (legacy sequence-level) or
        # [B, W, 2] (simple-dual per-presentation) — detected by ndim.
        task_state_bw = metadata.task_states.to(dev, non_blocking=True)  # [B, W, 2]
        total_loss, loss_dict = loss_fn(
            outputs, true_thetas, true_next_thetas,
            r_l23_windows, q_pred_windows,
            state_logits_windows=state_logits_windows if feedback_mode == 'fixed' else None,
            true_states_windows=states_for_loss,
            p_cw_windows=p_cw_for_loss,
            model=net if feedback_mode == 'emergent' else None,
            is_expected=is_expected,
            predicted_theta=predicted_theta,
            fb_scale=net.feedback_scale.item(),
            r_l4_windows=r_l4_windows,
            mismatch_labels=mm_labels_windows,
            mismatch_mask=mm_mask_windows,
            task_state=task_state_bw,
            task_routing=train_cfg.task_routing,
        )

        total_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(net.parameters(), train_cfg.gradient_clip)

        optimizer.step()
        scheduler.step()

        loss_history.append(loss_dict["total"])

        # Logging
        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                # Sensory accuracy (global + per-regime)
                logits = loss_fn.orientation_decoder(r_l23_windows)
                B_W = logits.shape[0] * logits.shape[1]
                sens_correct = (
                    logits.reshape(B_W, N).argmax(dim=-1)
                    == loss_fn._theta_to_channel(true_thetas).reshape(-1)
                ).float()  # [B*W]
                sensory_acc = sens_correct.mean().item()
                last_sensory_acc = sensory_acc

                # Per-regime sensory accuracy using per-presentation task_state.
                # task_state_bw is [B, W, 2]; col 0 = focused/relevant,
                # col 1 = routine/irrelevant. Flatten to [B*W] masks.
                rel_mask = task_state_bw[..., 0].reshape(-1)  # [B*W] focused
                irr_mask = task_state_bw[..., 1].reshape(-1)  # [B*W] routine
                s_acc_rel = (
                    (sens_correct * rel_mask).sum() / rel_mask.sum().clamp(min=1.0)
                ).item()
                s_acc_irr = (
                    (sens_correct * irr_mask).sum() / irr_mask.sum().clamp(min=1.0)
                ).item()
                # Defaults for per-regime mismatch (populated in emergent-mode
                # branch below if mismatch head is enabled).
                mm_acc_rel = 0.0
                mm_acc_irr = 0.0

                # Gradient norms
                total_norm = 0.0
                for p in net.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                # pi_pred ceiling monitoring
                pi_ceiling_frac = (aux["pi_pred_all"] >= model_cfg.pi_max - 0.01).float().mean().item()

                orient_step = model_cfg.orientation_range / N

                if feedback_mode == 'emergent':
                    # Emergent mode metrics: prior KL accuracy (q_pred = mu_pred)
                    state_acc = 0.0  # No CW/CCW classification in learned-prior mode

                    # Prediction accuracy via V2's mu_pred (= q_pred)
                    pred_channels = q_pred_windows[:, :-1].argmax(dim=-1)
                    true_channels = loss_fn._theta_to_channel(true_next_thetas[:, :-1])
                    pred_acc = (pred_channels == true_channels).float().mean().item()
                    last_pred_acc = pred_acc

                    # Circular angular error
                    pred_theta = q_pred_windows[:, :-1].argmax(dim=-1).float() * orient_step
                    true_theta_next = true_next_thetas[:, :-1]
                    angular_error = circular_distance_abs(pred_theta, true_theta_next).mean().item()

                    # Feedback info
                    fb_info = ""

                    # L4 sensory accuracy (when enabled)
                    l4_info = ""
                    if r_l4_windows is not None and hasattr(loss_fn, 'l4_decoder'):
                        l4_logits = loss_fn.l4_decoder(r_l4_windows)
                        B_W_l4 = l4_logits.shape[0] * l4_logits.shape[1]
                        l4_acc = (
                            l4_logits.reshape(B_W_l4, N).argmax(dim=-1)
                            == loss_fn._theta_to_channel(true_thetas).reshape(-1)
                        ).float().mean().item()
                        l4_info = f"l4_acc={l4_acc:.3f}, "

                    # Mismatch accuracy (global + per-regime) when enabled
                    mm_info = ""
                    if mm_labels_windows is not None and hasattr(loss_fn, 'mismatch_head'):
                        mm_logits = loss_fn.mismatch_head(
                            r_l23_windows.reshape(-1, N)
                        ).squeeze(-1)
                        mm_preds = (mm_logits > 0).float()
                        mm_targets = mm_labels_windows.reshape(-1)
                        mm_valid = mm_mask_windows.reshape(-1).bool() if mm_mask_windows is not None else torch.ones_like(mm_targets).bool()
                        if mm_valid.any():
                            mm_correct = (mm_preds == mm_targets).float()  # [B*W]
                            mm_acc = mm_correct[mm_valid].mean().item()
                            # Per-regime: valid & relevant, valid & irrelevant
                            rel_valid = (rel_mask.bool() & mm_valid)
                            irr_valid = (irr_mask.bool() & mm_valid)
                            if rel_valid.any():
                                mm_acc_rel = mm_correct[rel_valid].mean().item()
                            if irr_valid.any():
                                mm_acc_irr = mm_correct[irr_valid].mean().item()
                            mm_info = f"mm_acc={mm_acc:.3f}, "

                    logger.info(
                        f"Stage 2 step {step+1}/{n_steps}: "
                        f"loss={loss_dict['total']:.4f}, "
                        f"sens={loss_dict['sensory']:.4f}, "
                        f"prior_kl={loss_dict['state']:.4f}, "
                        f"fb_sparse={loss_dict['fb_sparsity']:.4f}, "
                        f"energy={loss_dict['energy_total']:.4f}, "
                        f"homeo={loss_dict['homeostasis']:.4f}, "
                        f"s_acc={sensory_acc:.3f}, p_acc={pred_acc:.3f}, "
                        f"s_acc_rel={s_acc_rel:.3f}, s_acc_irr={s_acc_irr:.3f}, "
                        f"mm_acc_rel={mm_acc_rel:.3f}, mm_acc_irr={mm_acc_irr:.3f}, "
                        f"ang_err={angular_error:.1f}, "
                        f"{fb_info}{l4_info}{mm_info}"
                        f"grad_norm={total_norm:.3f}, "
                        f"pi_ceil={pi_ceiling_frac:.3f}, "
                        f"fb_scale={net.feedback_scale.item():.3f}"
                    )
                else:
                    # Fixed mode metrics (unchanged)
                    # Prediction accuracy
                    pred_channels = q_pred_windows[:, :-1].argmax(dim=-1)
                    true_channels = loss_fn._theta_to_channel(true_next_thetas[:, :-1])
                    pred_acc = (pred_channels == true_channels).float().mean().item()
                    last_pred_acc = pred_acc

                    # State accuracy
                    if state_logits_windows is not None:
                        state_pred = state_logits_windows.argmax(dim=-1)
                        state_acc = (state_pred == true_states).float().mean().item()
                    else:
                        state_acc = 0.0

                    # Circular angular error
                    pred_theta = q_pred_windows[:, :-1].argmax(dim=-1).float() * orient_step
                    true_theta_next = true_next_thetas[:, :-1]
                    angular_error = circular_distance_abs(pred_theta, true_theta_next).mean().item()

                    # Top-3 channel accuracy
                    top3 = q_pred_windows[:, :-1].topk(3, dim=-1).indices
                    true_ch = loss_fn._theta_to_channel(true_next_thetas[:, :-1]).unsqueeze(-1)
                    top3_acc = (top3 == true_ch).any(dim=-1).float().mean().item()

                    # 12-anchor accuracy
                    pred_ch = q_pred_windows[:, :-1].argmax(dim=-1)
                    true_anchor = ((true_ch.squeeze(-1).float() / 3.0).round().long() * 3) % N
                    anchor_acc = ((pred_ch - true_anchor).abs() % N).clamp(max=N // 2).le(1).float().mean().item()

                    logger.info(
                        f"Stage 2 step {step+1}/{n_steps}: "
                        f"loss={loss_dict['total']:.4f}, "
                        f"sens={loss_dict['sensory']:.4f}, "
                        f"pred={loss_dict['prediction']:.4f}, "
                        f"state={loss_dict['state']:.4f}, "
                        f"energy={loss_dict['energy_total']:.4f}, "
                        f"homeo={loss_dict['homeostasis']:.4f}, "
                        f"s_acc={sensory_acc:.3f}, p_acc={pred_acc:.3f}, "
                        f"state_acc={state_acc:.3f}, "
                        f"ang_err={angular_error:.1f}, "
                        f"top3={top3_acc:.3f}, "
                        f"anchor={anchor_acc:.3f}, "
                        f"grad_norm={total_norm:.3f}, "
                        f"pi_ceil={pi_ceiling_frac:.3f}, "
                        f"fb_scale={net.feedback_scale.item():.3f}"
                    )

                # Write metrics to JSONL file for monitoring
                if output_dir is not None:
                    metrics = {
                        'step': step + 1,
                        'loss': round(loss_dict['total'], 4),
                        'sens': round(loss_dict['sensory'], 4),
                        'pred': round(loss_dict.get('prediction', 0.0), 4),
                        'state': round(loss_dict['state'], 4),
                        'energy': round(loss_dict['energy_total'], 4),
                        'homeo': round(loss_dict['homeostasis'], 4),
                        's_acc': round(sensory_acc, 3),
                        'p_acc': round(pred_acc if 'pred_acc' in dir() else last_pred_acc, 3),
                        'state_acc': round(state_acc, 3),
                        # Simple-dual-regime: per-regime monitored metrics.
                        's_acc_rel': round(s_acc_rel, 3),
                        's_acc_irr': round(s_acc_irr, 3),
                        'mm_acc_rel': round(mm_acc_rel, 3),
                        'mm_acc_irr': round(mm_acc_irr, 3),
                        'ang_err': round(angular_error, 1),
                        'grad_norm': round(total_norm, 3),
                        'pi_ceil': round(pi_ceiling_frac, 3),
                        'fb_scale': round(net.feedback_scale.item(), 3),
                        'feedback_mode': feedback_mode,
                    }
                    if loss_dict.get('l4_sensory', 0.0) > 0:
                        metrics['l4_sensory'] = round(loss_dict['l4_sensory'], 4)
                    if loss_dict.get('mismatch', 0.0) > 0:
                        metrics['mismatch'] = round(loss_dict['mismatch'], 4)
                    if loss_dict.get('expected_suppress', 0.0) > 0:
                        metrics['expected_suppress'] = round(loss_dict['expected_suppress'], 4)
                    if train_cfg.gradient_isolation:
                        phase = (step // train_cfg.isolation_period) % 2
                        metrics['grad_iso_phase'] = 'focused' if phase == 0 else 'routine'
                    metrics_path = os.path.join(output_dir, 'metrics.jsonl')
                    with open(metrics_path, 'a') as f:
                        f.write(json.dumps(metrics) + '\n')

        # Checkpoint callback
        if checkpoint_fn and checkpoint_steps and (step + 1) in checkpoint_steps:
            checkpoint_fn(step + 1)

    return Stage2Result(
        final_loss=loss_history[-1] if loss_history else float("nan"),
        final_sensory_acc=last_sensory_acc,
        final_pred_acc=last_pred_acc,
        loss_history=loss_history,
        n_steps_trained=n_steps,
    )
