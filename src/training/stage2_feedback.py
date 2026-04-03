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
        n_states=stim_cfg.n_states,
    )

    # Readout indices (last half of ON period)
    window_start = max(1, train_cfg.steps_on // 2)
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
            "Baselines (emergent): CW_accuracy chance=0.50, "
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
            build_stimulus_sequence(metadata, model_cfg, train_cfg)
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

        # Forward pass
        packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
        r_l23_all, final_state, aux = compiled_net(packed)

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
        }

        # Extract readout windows
        r_l23_windows, q_pred_windows, state_logits_windows = extract_readout_data(
            outputs, readout_indices,
            steps_on=train_cfg.steps_on, steps_isi=train_cfg.steps_isi,
        )

        # Extract p_cw windows for emergent mode
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
        total_loss, loss_dict = loss_fn(
            outputs, true_thetas, true_next_thetas,
            r_l23_windows, q_pred_windows,
            state_logits_windows=state_logits_windows if feedback_mode == 'fixed' else None,
            true_states_windows=true_states,
            p_cw_windows=p_cw_windows,
            model=net if feedback_mode == 'emergent' else None,
            is_expected=is_expected,
            predicted_theta=predicted_theta,
            fb_scale=net.feedback_scale.item(),
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
                # Sensory accuracy
                logits = loss_fn.orientation_decoder(r_l23_windows)
                B_W = logits.shape[0] * logits.shape[1]
                sensory_acc = (
                    logits.reshape(B_W, N).argmax(dim=-1)
                    == loss_fn._theta_to_channel(true_thetas).reshape(-1)
                ).float().mean().item()
                last_sensory_acc = sensory_acc

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
                    # Emergent mode metrics: CW accuracy from p_cw
                    if p_cw_windows is not None:
                        cw_pred = (p_cw_windows.squeeze(-1) > 0.5).long()
                        cw_target = (true_states == 0).long()
                        state_acc = (cw_pred == cw_target).float().mean().item()
                    else:
                        state_acc = 0.0

                    # Prediction accuracy via analytically-constructed q_pred
                    pred_channels = q_pred_windows[:, :-1].argmax(dim=-1)
                    true_channels = loss_fn._theta_to_channel(true_next_thetas[:, :-1])
                    pred_acc = (pred_channels == true_channels).float().mean().item()
                    last_pred_acc = pred_acc

                    # Circular angular error
                    pred_theta = q_pred_windows[:, :-1].argmax(dim=-1).float() * orient_step
                    true_theta_next = true_next_thetas[:, :-1]
                    angular_error = circular_distance_abs(pred_theta, true_theta_next).mean().item()

                    # Feedback operator profile info
                    fb_info = ""
                    if hasattr(net.feedback, 'alpha_inh'):
                        a_inh_norm = net.feedback.alpha_inh.abs().sum().item()
                        a_exc_norm = net.feedback.alpha_exc.abs().sum().item()
                        fb_info = f"a_inh={a_inh_norm:.3f}, a_exc={a_exc_norm:.3f}, "

                    logger.info(
                        f"Stage 2 step {step+1}/{n_steps}: "
                        f"loss={loss_dict['total']:.4f}, "
                        f"sens={loss_dict['sensory']:.4f}, "
                        f"state_bce={loss_dict['state']:.4f}, "
                        f"fb_sparse={loss_dict['fb_sparsity']:.4f}, "
                        f"energy={loss_dict['energy_total']:.4f}, "
                        f"homeo={loss_dict['homeostasis']:.4f}, "
                        f"s_acc={sensory_acc:.3f}, p_acc={pred_acc:.3f}, "
                        f"cw_acc={state_acc:.3f}, "
                        f"ang_err={angular_error:.1f}, "
                        f"{fb_info}"
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
                        'ang_err': round(angular_error, 1),
                        'grad_norm': round(total_norm, 3),
                        'pi_ceil': round(pi_ceiling_frac, 3),
                        'fb_scale': round(net.feedback_scale.item(), 3),
                        'feedback_mode': feedback_mode,
                    }
                    if feedback_mode == 'emergent' and hasattr(net.feedback, 'alpha_inh'):
                        metrics['a_inh_norm'] = round(net.feedback.alpha_inh.abs().sum().item(), 4)
                        metrics['a_exc_norm'] = round(net.feedback.alpha_exc.abs().sum().item(), 4)
                        metrics['fb_sparsity'] = round(loss_dict.get('fb_sparsity', 0.0), 4)
                    if feedback_mode == 'fixed':
                        metrics['top3'] = round(top3_acc, 3)
                        metrics['anchor'] = round(anchor_acc, 3)
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
