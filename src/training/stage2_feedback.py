"""Stage 2: V2 sequence learning + feedback training.

Goal: V2 learns transition statistics. Feedback learns to modulate V1.
80K steps with BPTT over HMM sequences.
Separate LR groups (V2: 3e-4, feedback: 1e-4), gradient clip 1.0,
linear warmup + cosine decay.
"""

from __future__ import annotations

import logging
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
            Signature: checkpoint_fn(step) where step is the 1-indexed step number.
        checkpoint_steps: List of step numbers at which to call checkpoint_fn.

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

    # Ensure correct freeze/unfreeze state
    freeze_stage1(net)
    unfreeze_stage2(net)

    # Compile step function for reduced Python/CUDA kernel launch overhead.
    # Step-level compile avoids the 600-step graph unrolling that makes
    # full-model compile impractical for recurrent networks.
    net.step = torch.compile(net.step, mode='max-autotune-no-cudagraphs')

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
    )

    # Readout indices (timesteps 4-7 of each ON period)
    readout_indices = compute_readout_indices(
        seq_length, train_cfg.steps_on, train_cfg.steps_isi,
        window_start=4, window_end=7,
    )

    loss_history = []
    last_sensory_acc = 0.0
    last_pred_acc = 0.0

    # Reference baselines for interpreting metrics
    logger.info(
        "Baselines: uniform=0.028 (1/36), same_as_current≈0.20, "
        "oracle_with_state≈0.75 (within ±1 channel)"
    )

    # Fix E: V2 curriculum — hard zero feedback for predictor burn-in,
    # then ramp feedback from 0 to 1.
    predictor_burnin_steps = 5000
    feedback_ramp_steps = 5000  # ramp from 0→1 over this many steps after burn-in

    # Fix 5: Freeze-then-unfreeze W_rec gain (aligned with end of burn-in)
    gain_unfreeze_step = predictor_burnin_steps
    net.l23.gain_rec_raw.requires_grad_(False)  # Start frozen

    for step in range(n_steps):
        # Fix E: V2 curriculum — hard zero during burn-in, then ramp
        if step < predictor_burnin_steps:
            net.feedback_scale = 0.0
        else:
            ramp_progress = (step - predictor_burnin_steps) / feedback_ramp_steps
            net.feedback_scale = min(1.0, ramp_progress)

        # Fix 5: Unfreeze gain_rec after burn-in
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
        stim_seq = stim_seq.to(dev)
        cue_seq = cue_seq.to(dev)
        task_seq = task_seq.to(dev)
        true_thetas = true_thetas.to(dev)
        true_next_thetas = true_next_thetas.to(dev)
        true_states = true_states.to(dev)

        # Forward pass (packed single-tensor input)
        packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
        r_l23_all, final_state, aux = net(packed)

        # Build outputs dict for loss computation
        outputs = {
            "r_l23": r_l23_all,
            "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"],
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
            "state_logits": aux["state_logits_all"],
        }

        # Extract readout windows (now also extracts state_logits)
        r_l23_windows, q_pred_windows, state_logits_windows = extract_readout_data(
            outputs, readout_indices
        )

        # Compute loss (with state classification loss)
        total_loss, loss_dict = loss_fn(
            outputs, true_thetas, true_next_thetas,
            r_l23_windows, q_pred_windows,
            state_logits_windows=state_logits_windows,
            true_states_windows=true_states,
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

                # Prediction accuracy (skip last presentation)
                pred_channels = q_pred_windows[:, :-1].argmax(dim=-1)
                true_channels = loss_fn._theta_to_channel(true_next_thetas[:, :-1])
                pred_acc = (pred_channels == true_channels).float().mean().item()
                last_pred_acc = pred_acc

                # Gradient norms
                total_norm = 0.0
                for p in net.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                # Fix 4d: pi_pred ceiling monitoring
                pi_ceiling_frac = (aux["pi_pred_all"] >= model_cfg.pi_max - 0.01).float().mean().item()

                # Fix C: Better metrics
                orient_step = model_cfg.orientation_range / N

                # 1. Latent state accuracy (3-way: CW/CCW/neutral)
                if state_logits_windows is not None:
                    state_pred = state_logits_windows.argmax(dim=-1)
                    state_acc = (state_pred == true_states).float().mean().item()
                else:
                    state_acc = 0.0

                # 2. Circular angular error (degrees)
                pred_theta = q_pred_windows[:, :-1].argmax(dim=-1).float() * orient_step
                true_theta_next = true_next_thetas[:, :-1]
                angular_error = circular_distance_abs(pred_theta, true_theta_next).mean().item()

                # 3. Top-3 channel accuracy
                top3 = q_pred_windows[:, :-1].topk(3, dim=-1).indices
                true_ch = loss_fn._theta_to_channel(true_next_thetas[:, :-1]).unsqueeze(-1)
                top3_acc = (top3 == true_ch).any(dim=-1).float().mean().item()

                # 4. 12-anchor accuracy (within ±1 channel of nearest anchor)
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
                f"fb_scale={net.feedback_scale:.3f}"
            )

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
