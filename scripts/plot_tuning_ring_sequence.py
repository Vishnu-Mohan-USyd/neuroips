#!/usr/bin/env python3
"""Plot a matched-sequence tuning-ring triptych using circular ring icons.

This script loads a trained checkpoint, runs three matched single-trial
sequences that share the same context history, and renders one ring icon per
presentation column for:
1. Expected probe
2. Unexpected probe
3. Omitted probe

Each ring icon shows the mean L2/3 activity during that presentation's ON
window. The final column is highlighted as the probe or omission slot, and the
bottom row shows the actual stimulus orientations across the shared context and
branch-specific final presentation. The default view uses the aligned 5°
checkpoint, crops to the final three context presentations, puts the expected
probe at 0°, and uses a +90° unexpected branch.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import ModelConfig, TrainingConfig
from src.experiments.paradigm_base import TrialConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating


DEFAULT_RESULTS_DIR = "results/r12_fb24_sharp_050_width_075_rec11_aligned"
DEFAULT_CHECKPOINT = (
    f"{DEFAULT_RESULTS_DIR}/emergent_seed42/checkpoint.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_RESULTS_DIR}/tuning_ring_sequence_rings_dev90_last3_probe0_viridis.png",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device for the forward pass.",
    )
    parser.add_argument(
        "--start-ori",
        type=float,
        default=None,
        help="Starting orientation for the shared context sequence (deg). "
             "When omitted, the script chooses the start so the expected probe lands at 0°.",
    )
    parser.add_argument(
        "--direction",
        choices=["cw", "ccw"],
        default="ccw",
        help="Direction of the shared context progression.",
    )
    parser.add_argument(
        "--unexpected-offset",
        type=float,
        default=90.0,
        help="Unexpected probe offset relative to the expected probe (deg).",
    )
    parser.add_argument(
        "--show-last-n",
        type=int,
        default=3,
        help="How many context presentations to display before the probe column.",
    )
    parser.add_argument(
        "--metric-variant",
        choices=("none", "left-box", "probe-caption", "probe-inhole", "probe-side-strip"),
        default="none",
        help="Placement style for the final probe/omission-slot metrics.",
    )
    parser.add_argument(
        "--stimulus-row-style",
        choices=("mini-ring", "bead-ring"),
        default="mini-ring",
        help="Rendering style for the bottom stimulus row only.",
    )
    parser.add_argument(
        "--transition-step",
        type=float,
        default=None,
        help="Override transition_step if not in ModelConfig.",
    )
    return parser.parse_args()


def _filter_dataclass_kwargs(cls, raw: dict | None) -> dict:
    if raw is None:
        return {}
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}


def load_model_and_trial_cfg(checkpoint_path: Path, device: torch.device) -> tuple[LaminarV1V2Network, ModelConfig, TrialConfig]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_raw = dict(ckpt.get("config", {}).get("model", {}))
    for legacy_key in ("mechanism", "n_basis", "max_apical_gain", "tau_vip",
                       "simple_feedback", "template_gain"):
        model_raw.pop(legacy_key, None)
    train_raw = dict(ckpt.get("config", {}).get("training", {}))

    model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_raw))
    train_cfg = TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, train_raw))
    trial_cfg = TrialConfig(
        n_context=10,
        steps_on=train_cfg.steps_on,
        steps_isi=train_cfg.steps_isi,
        steps_post=12,
        contrast=0.8,
    )

    net = LaminarV1V2Network(model_cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, model_cfg, trial_cfg


def make_grating(model_cfg: ModelConfig, orientation_deg: float, contrast: float) -> torch.Tensor:
    return generate_grating(
        torch.tensor([orientation_deg], dtype=torch.float32),
        torch.tensor([contrast], dtype=torch.float32),
        n_orientations=model_cfg.n_orientations,
        sigma=model_cfg.sigma_ff,
        n=model_cfg.naka_rushton_n,
        c50=model_cfg.naka_rushton_c50,
        period=model_cfg.orientation_range,
    )


def build_single_trial(
    model_cfg: ModelConfig,
    trial_cfg: TrialConfig,
    context_oris: list[float],
    probe_ori: float,
    probe_contrast: float,
    relevant_task: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_steps = trial_cfg.n_context * (trial_cfg.steps_on + trial_cfg.steps_isi) + trial_cfg.steps_on + trial_cfg.steps_post
    stim = torch.zeros(1, total_steps, model_cfg.n_orientations)

    for idx, ori in enumerate(context_oris):
        onset = idx * (trial_cfg.steps_on + trial_cfg.steps_isi)
        stim[0, onset:onset + trial_cfg.steps_on] = make_grating(model_cfg, ori, trial_cfg.contrast)

    probe_onset = trial_cfg.n_context * (trial_cfg.steps_on + trial_cfg.steps_isi)
    stim[0, probe_onset:probe_onset + trial_cfg.steps_on] = make_grating(model_cfg, probe_ori, probe_contrast)

    task_state = torch.zeros(1, total_steps, 2)
    if relevant_task:
        task_state[0, :, 0] = 1.0
    else:
        task_state[0, :, 1] = 1.0
    return stim, task_state


def run_sequence(
    net: LaminarV1V2Network,
    stim: torch.Tensor,
    task_state: torch.Tensor,
) -> torch.Tensor:
    packed = LaminarV1V2Network.pack_inputs(
        stim.to(next(net.parameters()).device),
        None,
        task_state.to(next(net.parameters()).device),
    )
    with torch.no_grad():
        r_l23_all, _, _ = net.forward(packed)
    return r_l23_all[0].detach().cpu()


def presentation_windows(trial_cfg: TrialConfig) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    stride = trial_cfg.steps_on + trial_cfg.steps_isi
    for pres_idx in range(trial_cfg.n_context):
        onset = pres_idx * stride
        windows.append((onset, onset + trial_cfg.steps_on))
    probe_onset = trial_cfg.n_context * stride
    windows.append((probe_onset, probe_onset + trial_cfg.steps_on))
    return windows


def average_on_windows(r_l23: torch.Tensor, windows: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([r_l23[start:end].mean(dim=0).numpy() for start, end in windows], axis=0)


def build_context_sequence(start_ori: float, step_deg: float, n_context: int, direction: str, period: float) -> list[float]:
    sign = 1.0 if direction == "cw" else -1.0
    return [float((start_ori + sign * idx * step_deg) % period) for idx in range(n_context)]


def infer_start_orientation(
    transition_step: float,
    n_context: int,
    direction: str,
    period: float,
) -> float:
    sign = 1.0 if direction == "cw" else -1.0
    return float((-sign * n_context * transition_step) % period)


def plot_ring_icon(
    ax: plt.Axes,
    activity: np.ndarray,
    vmax: float,
    cmap,
    highlight: bool = False,
    show_cardinals: bool = True,
) -> None:
    n_channels = len(activity)
    theta_centers = np.arange(n_channels) * (2 * np.pi / n_channels)
    width = 2 * np.pi / n_channels
    inner_r = 0.68
    outer_r = 1.02
    height = outer_r - inner_r

    colors = cmap(Normalize(vmin=0.0, vmax=vmax)(activity))
    ax.bar(
        theta_centers,
        height,
        width=width,
        bottom=inner_r,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        align="center",
    )

    if show_cardinals:
        for deg in (0, 45, 90, 135):
            rad = deg * 2 * np.pi / 180.0
            ax.text(rad, 0.30, f"{deg}", ha="center", va="center", color="0.55", fontsize=6)

    if highlight:
        ax.bar(
            [0.0],
            [outer_r + 0.08],
            width=2 * np.pi,
            bottom=0.0,
            color="none",
            edgecolor="#cc8f00",
            linewidth=1.6,
            align="edge",
        )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 1.22)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)


def add_ring_caption(ax, text, color="0.25", fontsize=8.0):
    ax.text(0.5, -0.16, text, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize, color=color)


def plot_stimulus_bead_ring(ax, n_channels, highlight_idx, cmap, highlight_value=1.0, highlight=False, scale=1.0, show_axis_labels=True):
    theta = np.arange(n_channels) * (2 * np.pi / n_channels)
    radius = 1.0 * scale
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    for deg, label in ((0, "0"), (90, "90")):
        rad = deg * 2 * np.pi / 180.0
        ax.plot([0.0, 0.9 * radius * np.cos(rad)], [0.0, 0.9 * radius * np.sin(rad)], color="#e3e3e3", linewidth=0.9, zorder=0, clip_on=False)
        if show_axis_labels:
            ax.text(1.02 * radius * np.cos(rad), 1.02 * radius * np.sin(rad), label, ha="center", va="center", fontsize=6.2, color="#b3b3b3", zorder=0, clip_on=False)
    base_color = "#d7d7d7"
    ax.scatter(x, y, s=18 * scale, c=base_color, edgecolors="none", zorder=1, clip_on=False)
    if highlight_idx is not None:
        hi = int(highlight_idx) % n_channels
        color = cmap(Normalize(vmin=0.0, vmax=1.0)(highlight_value))
        ax.scatter([x[hi]], [y[hi]], s=116 * scale, c=[color], edgecolors="#4f4f4f", linewidths=1.0, zorder=3, clip_on=False)
        ax.scatter([x[hi]], [y[hi]], s=68 * scale, c=[color], edgecolors="white", linewidths=0.9, zorder=4, clip_on=False)
    if highlight:
        ring = plt.Circle((0.0, 0.0), 1.18 * scale, fill=False, edgecolor="#cc8f00", linewidth=1.6, clip_on=False)
        ax.add_patch(ring)
    lim = 1.38 * scale
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def format_probe_metrics(slot_at_zero, slot_total, compact=False):
    if compact:
        return f"0° {slot_at_zero:.3f}\nΣ {slot_total:.2f}"
    return f"slot @ 0° {slot_at_zero:.3f}\nslot total {slot_total:.2f}"


def branch_probe_slot_metrics(probe_curve, channel_at_zero_deg):
    return float(probe_curve[channel_at_zero_deg]), float(probe_curve.sum())


def add_probe_inhole_metrics(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=7.6, color="0.15", linespacing=1.15)


def add_probe_metric_strip(fig, metrics):
    ax = fig.add_axes([0.845, 0.23, 0.085, 0.54])
    ax.axis("off")
    ax.text(0.0, 1.02, "Final slot metrics", ha="left", va="bottom", fontsize=9.0, fontweight="bold", color="0.15")
    y_positions = [0.80, 0.50, 0.20]
    for (label, slot_at_zero, slot_total), y in zip(metrics, y_positions):
        ax.text(0.0, y, f"{label}\n0° {slot_at_zero:.3f}\nΣ {slot_total:.2f}", ha="left", va="center", fontsize=8.1, color="0.15", linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.82", alpha=0.96, linewidth=0.6))


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    net, model_cfg, trial_cfg = load_model_and_trial_cfg(checkpoint_path, device)

    # Resolve transition_step: CLI override takes precedence, else try config, else default 10.0
    transition_step = (
        args.transition_step
        if args.transition_step is not None
        else getattr(model_cfg, "transition_step", 10.0)
    )
    step_deg = model_cfg.orientation_range / model_cfg.n_orientations

    start_ori = (
        args.start_ori
        if args.start_ori is not None
        else infer_start_orientation(
            transition_step=transition_step,
            n_context=trial_cfg.n_context,
            direction=args.direction,
            period=model_cfg.orientation_range,
        )
    )
    context_oris = build_context_sequence(
        start_ori=start_ori,
        step_deg=transition_step,
        n_context=trial_cfg.n_context,
        direction=args.direction,
        period=model_cfg.orientation_range,
    )
    sign = 1.0 if args.direction == "cw" else -1.0
    expected_ori = float((context_oris[-1] + sign * transition_step) % model_cfg.orientation_range)
    unexpected_ori = float((expected_ori + args.unexpected_offset) % model_cfg.orientation_range)

    stim_exp, task_exp = build_single_trial(model_cfg, trial_cfg, context_oris, expected_ori, probe_contrast=trial_cfg.contrast)
    stim_unexp, task_unexp = build_single_trial(model_cfg, trial_cfg, context_oris, unexpected_ori, probe_contrast=trial_cfg.contrast)
    stim_omit, task_omit = build_single_trial(model_cfg, trial_cfg, context_oris, expected_ori, probe_contrast=0.0)

    windows = presentation_windows(trial_cfg)
    seq_expected = average_on_windows(run_sequence(net, stim_exp, task_exp), windows)
    seq_unexpected = average_on_windows(run_sequence(net, stim_unexp, task_unexp), windows)
    seq_omission = average_on_windows(run_sequence(net, stim_omit, task_omit), windows)
    show_last_n = max(1, min(args.show_last_n, trial_cfg.n_context))
    display_indices = list(range(trial_cfg.n_context - show_last_n, trial_cfg.n_context)) + [trial_cfg.n_context]
    display_context_oris = [context_oris[idx] for idx in display_indices[:-1]]

    seq_expected = seq_expected[display_indices]
    seq_unexpected = seq_unexpected[display_indices]
    seq_omission = seq_omission[display_indices]
    vmax = float(np.max([seq_expected.max(), seq_unexpected.max(), seq_omission.max()]))

    context_stimuli = [
        make_grating(model_cfg, ori, trial_cfg.contrast).squeeze(0).numpy()
        for ori in display_context_oris
    ]
    expected_stimulus = make_grating(model_cfg, expected_ori, trial_cfg.contrast).squeeze(0).numpy()
    unexpected_stimulus = make_grating(model_cfg, unexpected_ori, trial_cfg.contrast).squeeze(0).numpy()
    omission_stimulus = np.zeros_like(expected_stimulus)
    stim_vmax = float(np.max(context_stimuli + [expected_stimulus, unexpected_stimulus]))
    context_stim_indices = [int(round(ori / step_deg)) % model_cfg.n_orientations for ori in display_context_oris]
    expected_stim_idx = int(round(expected_ori / step_deg)) % model_cfg.n_orientations
    unexpected_stim_idx = int(round(unexpected_ori / step_deg)) % model_cfg.n_orientations

    cmap = matplotlib.colormaps["viridis"]
    row_specs = [
        ("Expected", seq_expected, f"expected probe {expected_ori:.0f}°"),
        ("Unexpected", seq_unexpected, f"unexpected probe +{args.unexpected_offset:.0f}°"),
        ("Omission", seq_omission, "probe omitted"),
    ]
    n_presentations = len(display_indices)
    zero_ch = int(round(0.0 / step_deg)) % model_cfg.n_orientations
    row_metrics: list[tuple[str, float, float]] = []

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(4, n_presentations, height_ratios=[1.0, 1.0, 1.0, 0.62], hspace=0.22, wspace=0.08)

    for row_idx, (row_label, seq_matrix, row_note) in enumerate(row_specs):
        slot_at_zero, slot_total = branch_probe_slot_metrics(seq_matrix[-1], zero_ch)
        row_metrics.append((row_label, slot_at_zero, slot_total))
        for col_idx in range(n_presentations):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection="polar")
            highlight = col_idx == n_presentations - 1
            show_cardinals = not (highlight and args.metric_variant == "probe-inhole")
            plot_ring_icon(ax, seq_matrix[col_idx], vmax, cmap, highlight=highlight, show_cardinals=show_cardinals)
            if highlight and args.metric_variant == "probe-caption":
                add_ring_caption(ax, format_probe_metrics(slot_at_zero, slot_total, compact=True), color="0.15", fontsize=7.7)
            elif highlight and args.metric_variant == "probe-inhole":
                add_probe_inhole_metrics(ax, format_probe_metrics(slot_at_zero, slot_total, compact=True))
            if row_idx == 0:
                header_idx = display_indices[col_idx] + 1
                header = f"P{header_idx}" if not highlight else "Probe"
                ax.set_title(header, y=1.14, fontsize=10, fontweight="bold" if highlight else None, color="#9a6700" if highlight else "black")
            if col_idx == 0:
                ax.text(-0.62, 0.53, row_label, transform=ax.transAxes, ha="right", va="center", fontsize=12, fontweight="bold")
                ax.text(-0.62, 0.19, row_note, transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color="0.35")
                if args.metric_variant == "left-box":
                    ax.text(-0.62, -0.07, format_probe_metrics(slot_at_zero, slot_total), transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color="0.15",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.82", alpha=0.95, linewidth=0.6))

    for col_idx, ori in enumerate(display_context_oris):
        if args.stimulus_row_style == "bead-ring":
            ax = fig.add_subplot(gs[3, col_idx])
            plot_stimulus_bead_ring(ax, n_channels=model_cfg.n_orientations, highlight_idx=context_stim_indices[col_idx], cmap=cmap, show_axis_labels=False)
        else:
            ax = fig.add_subplot(gs[3, col_idx], projection="polar")
            plot_ring_icon(ax, context_stimuli[col_idx], stim_vmax, cmap, show_cardinals=False)
        if args.stimulus_row_style != "bead-ring":
            add_ring_caption(ax, f"{ori:.0f}°")
        if col_idx == 0:
            ax.text(-0.78, 0.53, "Stimulus", transform=ax.transAxes, ha="right", va="center", fontsize=12, fontweight="bold")
            ax.text(-0.78, 0.18, "shown sequence", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color="0.35")

    probe_spec = gs[3, n_presentations - 1].subgridspec(3, 1, hspace=0.08)
    probe_rows = [
        ("E", expected_stimulus, expected_stim_idx, f"{expected_ori:.0f}°"),
        ("U", unexpected_stimulus, unexpected_stim_idx, f"{unexpected_ori:.0f}°"),
        ("O", omission_stimulus, None, "omit"),
    ]
    for probe_idx, (tag, stimulus, highlight_idx, caption) in enumerate(probe_rows):
        if args.stimulus_row_style == "bead-ring":
            ax = fig.add_subplot(probe_spec[probe_idx, 0])
            plot_stimulus_bead_ring(ax, n_channels=model_cfg.n_orientations, highlight_idx=highlight_idx, cmap=cmap, highlight=True, scale=1.18, show_axis_labels=False)
            ax.text(-0.24, 0.50, tag, transform=ax.transAxes, ha="right", va="center", fontsize=8.8, fontweight="bold", color="#9a6700", clip_on=False)
        else:
            ax = fig.add_subplot(probe_spec[probe_idx, 0], projection="polar")
            plot_ring_icon(ax, stimulus, stim_vmax, cmap, highlight=True, show_cardinals=False)
        if args.stimulus_row_style != "bead-ring":
            ax.text(-0.10, 0.52, tag, transform=ax.transAxes, ha="right", va="center", fontsize=8.5, fontweight="bold", color="#9a6700")
            add_ring_caption(ax, caption, color="0.35", fontsize=7.5)

    if args.metric_variant == "probe-side-strip":
        add_probe_metric_strip(fig, row_metrics)
        cax = fig.add_axes([0.945, 0.28, 0.015, 0.50])
    else:
        cax = fig.add_axes([0.915, 0.28, 0.015, 0.50])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Mean L2/3 activity during ON window")

    fig.suptitle("Matched sequence tuning ring heatmaps", fontsize=16)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(output_path)


if __name__ == "__main__":
    main()
