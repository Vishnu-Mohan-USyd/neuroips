"""Shared training utilities: optimizer setup, scheduler, freeze/unfreeze,
stimulus sequence building, readout window extraction."""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import ModelConfig, TrainingConfig, StimulusConfig
from src.model.network import LaminarV1V2Network
from src.model.feedback import EmergentFeedbackOperator
from src.stimulus.gratings import generate_grating, make_ambiguous_stimulus
from src.stimulus.sequences import HMMSequenceGenerator


# ---------------------------------------------------------------------------
# Parameter grouping & freezing
# ---------------------------------------------------------------------------

def get_stage1_params(net: LaminarV1V2Network) -> Iterator[nn.Parameter]:
    """Trainable parameters for Stage 1: L2/3 + PV."""
    yield from net.l23.parameters()
    yield from net.pv.parameters()


def freeze_stage1(net: LaminarV1V2Network) -> None:
    """Freeze Stage 1 parameters after sensory scaffold training.

    Freezes: L4, PV, L2/3 W_l4_l23 (buffer, already frozen).
    Keeps W_rec (sigma_rec_raw, gain_rec_raw) trainable for Stage 2.
    """
    net.l4.requires_grad_(False)
    net.pv.requires_grad_(False)
    # Freeze L2/3 inhibitory gains but keep W_rec trainable
    net.l23.w_som.gain_raw.requires_grad_(False)
    net.l23.w_pv_l23.gain_raw.requires_grad_(False)


def unfreeze_stage2(net: LaminarV1V2Network) -> None:
    """Ensure Stage 2 parameters are trainable: V2, feedback, SOM, deep_template, W_rec."""
    for p in net.v2.parameters():
        p.requires_grad_(True)
    for p in net.feedback.parameters():
        p.requires_grad_(True)
    net.deep_template.gain_raw.requires_grad_(True)
    # W_rec stays trainable in Stage 2
    net.l23.sigma_rec_raw.requires_grad_(True)
    net.l23.gain_rec_raw.requires_grad_(True)


def create_stage2_optimizer(
    net: LaminarV1V2Network,
    loss_fn: nn.Module,
    cfg: TrainingConfig,
) -> AdamW:
    """AdamW with separate LR groups for Stage 2.

    Group 1: V2 params (lr_v2)
    Group 2: Feedback params (lr_feedback)
    Group 3: W_rec + deep_template (lr_feedback)
    Group 4: Decoder (stage1_lr)
    """
    # Feedback params: works for both FeedbackMechanism and EmergentFeedbackOperator
    feedback_params = [p for p in net.feedback.parameters() if p.requires_grad]

    param_groups = [
        {"params": list(net.v2.parameters()), "lr": cfg.stage2_lr_v2},
        {"params": feedback_params, "lr": cfg.stage2_lr_feedback},
        {
            "params": [
                net.l23.sigma_rec_raw,
                net.l23.gain_rec_raw,
                net.deep_template.gain_raw,
            ],
            "lr": cfg.stage2_lr_feedback,
        },
        {"params": list(loss_fn.orientation_decoder.parameters()), "lr": cfg.stage1_lr},
    ]
    # Add surprise detector params if present
    if hasattr(loss_fn, 'surprise_detector'):
        param_groups.append(
            {"params": list(loss_fn.surprise_detector.parameters()), "lr": cfg.stage1_lr}
        )
    # Filter empty groups
    param_groups = [g for g in param_groups if len(list(g["params"])) > 0]
    return AdamW(param_groups, weight_decay=cfg.stage2_weight_decay)


# ---------------------------------------------------------------------------
# Scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def make_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup for warmup_steps, then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Stimulus sequence building
# ---------------------------------------------------------------------------

def build_stimulus_sequence(
    metadata,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Convert HMM SequenceMetadata into temporal stimulus sequences.

    Each presentation occupies steps_on timesteps of grating + steps_isi
    timesteps of blank (ISI). Total timesteps = seq_length * (steps_on + steps_isi).

    Args:
        metadata: SequenceMetadata from HMMSequenceGenerator.
        model_cfg: ModelConfig for population coding params.
        train_cfg: TrainingConfig for temporal params.

    Returns:
        stimulus_seq: [B, T_total, N] — population-coded stimuli.
        cue_seq: [B, T_total, N] — cue inputs.
        task_state_seq: [B, T_total, 2] — task states.
        true_thetas: [B, seq_length] — true orientations in degrees.
        true_next_thetas: [B, seq_length] — next orientations in degrees.
        true_states: [B, seq_length] — HMM state indices (long).
    """
    B, S = metadata.orientations.shape
    N = model_cfg.n_orientations
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    T_total = S * steps_per

    # Generate all stimuli at once: flatten [B, S] → [B*S], one call to generate_grating
    oris_flat = metadata.orientations.reshape(-1)       # [B*S]
    contrasts_flat = metadata.contrasts.reshape(-1)      # [B*S]
    stim_all = generate_grating(
        oris_flat, contrasts_flat,
        n_orientations=N,
        sigma=model_cfg.sigma_ff,
        n=model_cfg.naka_rushton_n,
        c50=model_cfg.naka_rushton_c50,
        period=model_cfg.orientation_range,
    ).reshape(B, S, N)  # [B, S, N]

    # Handle ambiguous stimuli in batch
    is_amb_flat = metadata.is_ambiguous.reshape(-1)      # [B*S]
    if is_amb_flat.any():
        oris2_flat = (oris_flat + 15.0) % model_cfg.orientation_range
        stim_amb = make_ambiguous_stimulus(
            oris_flat[is_amb_flat], oris2_flat[is_amb_flat], contrasts_flat[is_amb_flat],
            n_orientations=N,
            sigma=model_cfg.sigma_ff,
            n=model_cfg.naka_rushton_n,
            c50=model_cfg.naka_rushton_c50,
            period=model_cfg.orientation_range,
        )
        stim_all.reshape(-1, N)[is_amb_flat] = stim_amb

    # Temporal expansion: [B, S, N] → [B, S, steps_per, N] → [B, T_total, N]
    # Stimulus fills first steps_on timesteps per presentation, ISI is zero
    temporal = torch.zeros(B, S, steps_per, N)
    temporal[:, :, :steps_on, :] = stim_all.unsqueeze(2).expand(-1, -1, steps_on, -1)
    stimulus_seq = temporal.reshape(B, T_total, N)

    # Cue: [B, S, N] → expand to all timesteps per presentation
    cue_seq = metadata.cues.unsqueeze(2).expand(-1, -1, steps_per, -1).reshape(B, T_total, N)

    # Task state: [B, S, 2] → expand to all timesteps per presentation
    task_state_seq = metadata.task_states.unsqueeze(2).expand(-1, -1, steps_per, -1).reshape(B, T_total, 2)

    # True orientations in degrees
    true_thetas = metadata.orientations  # [B, S]

    # Next-orientation: shifted by 1, last wraps to first
    true_next_thetas = torch.roll(metadata.orientations, -1, dims=1)  # [B, S]

    # Shift states by 1 to align with prediction targets: q_pred predicts
    # next orientation, so state_logits should predict next state.
    true_next_states = torch.roll(metadata.states, -1, dims=1)
    true_next_states[:, -1] = metadata.states[:, -1]  # last: keep current (no valid next)

    return stimulus_seq, cue_seq, task_state_seq, true_thetas, true_next_thetas, true_next_states


# ---------------------------------------------------------------------------
# Readout window extraction
# ---------------------------------------------------------------------------

def compute_readout_indices(
    seq_length: int,
    steps_on: int = 8,
    steps_isi: int = 4,
    window_start: int = 4,
    window_end: int = 7,
) -> list[tuple[int, list[int]]]:
    """Return timestep indices for readout windows.

    For each of seq_length presentations, the readout window is timesteps
    [window_start, window_end] relative to presentation onset (inclusive).

    Returns:
        list of (presentation_idx, list of timestep indices)
    """
    steps_per = steps_on + steps_isi
    indices = []
    for s in range(seq_length):
        t_onset = s * steps_per
        window_ts = list(range(t_onset + window_start, t_onset + window_end + 1))
        indices.append((s, window_ts))
    return indices


def extract_readout_data(
    outputs: dict[str, Tensor],
    readout_indices: list[tuple[int, list[int]]],
    steps_on: int = 8,
    steps_isi: int = 4,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Extract L2/3, q_pred, and state_logits at readout windows, averaged over each window.

    Uses reshape-based slicing instead of per-presentation loops:
    reshape [B, T, D] → [B, S, steps_per, D], slice the window, mean over dim=2.

    Args:
        outputs: dict with 'r_l23' [B, T, N], 'q_pred' [B, T, N],
                 and optionally 'state_logits' [B, T, 3].
        readout_indices: from compute_readout_indices (used to extract window bounds).
        steps_on: Timesteps per stimulus presentation.
        steps_isi: Inter-stimulus interval timesteps.

    Returns:
        r_l23_windows: [B, n_presentations, N]
        q_pred_windows: [B, n_presentations, N]
        state_logits_windows: [B, n_presentations, 3] or None
    """
    r_l23_all = outputs["r_l23"]       # [B, T, N]
    q_pred_all = outputs["q_pred"]     # [B, T, N]
    state_logits_all = outputs.get("state_logits")  # [B, T, 3] or None

    # Extract window bounds from readout_indices (all presentations use the same relative window)
    _, ts_first = readout_indices[0]
    steps_per = steps_on + steps_isi
    window_start = ts_first[0]  # Relative offset within first presentation
    window_end = ts_first[-1] + 1  # Exclusive end

    B, T, N = r_l23_all.shape
    S = len(readout_indices)

    # Reshape [B, T, D] → [B, S, steps_per, D], slice window, mean over window dim
    r_l23_windows = (
        r_l23_all.reshape(B, S, steps_per, N)[:, :, window_start:window_end].mean(dim=2)
    )
    q_pred_windows = (
        q_pred_all.reshape(B, S, steps_per, q_pred_all.shape[-1])[:, :, window_start:window_end].mean(dim=2)
    )

    state_logits_windows = None
    if state_logits_all is not None:
        state_logits_windows = (
            state_logits_all.reshape(B, S, steps_per, state_logits_all.shape[-1])[:, :, window_start:window_end].mean(dim=2)
        )

    return r_l23_windows, q_pred_windows, state_logits_windows
