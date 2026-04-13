#!/usr/bin/env python3
"""Orientation ring heatmap — L2/3 activity at probe 60° (Task #23 redo).

Loads the Rescue 4 checkpoint, runs HMM sequences with FB ON, filters
presentations whose probe orientation is ~60° (channel 11 or 12), then
produces a 2x2 grid of polar ring heatmaps:

    rows:    task_state = Relevant (focused) / Irrelevant (routine)
    cols:    prediction bucket = Expected (pred_err<=10) / Unexpected (pred_err>20)
    wedge i: colored by mean r_l23 at late-ON (t=9) channel i

Decorations per panel:
    - Black arrow + "probe 60°" label pointing at channel 12 (probe).
    - Blue arrow + "predicted <deg>°" label pointing at the MODE predicted
      channel within the bucket — UNEXPECTED panels only.
    - Cardinal labels (0/45/90/135°) at plot-angle 0/90/180/270 deg.

Colormap YlOrRd, shared `vmax` across all panels. A single horizontal
colorbar at the bottom reports the activity scale.

Save path defaults to docs/figures/tuning_ring_heatmap.png (relative to
repo root). The polar axes use theta_zero_location='E' (0° at right)
and theta_direction=1 (counterclockwise) so that 60° orientation falls
at the upper-left, matching the reference image.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


# ------------------------------ helpers ------------------------------


def circular_distance(a: torch.Tensor, b: torch.Tensor, period: float = 180.0) -> torch.Tensor:
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def circular_mode(values: np.ndarray, N: int) -> int:
    """Integer mode on the orientation ring (values are channel indices)."""
    if len(values) == 0:
        return -1
    hist = np.bincount(values, minlength=N)
    return int(np.argmax(hist))


# ------------------------------ collect ------------------------------


def collect(config_path: str, checkpoint_path: str, device: torch.device,
            seed: int, n_batches: int, probe_channels: set[int]) -> tuple[dict, int]:
    """Run the trial loop and return (buckets_dict, N).

    buckets_dict:
        ('relevant',   'expected'):   dict with 'mean', 'n', 'pred_mode_ch'
        ('relevant',   'unexpected'): ...
        ('irrelevant', 'expected'):   ...
        ('irrelevant', 'unexpected'): ...
    """
    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    T_READ = 9

    gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
    )

    keys = [
        ("relevant",   "expected"),
        ("relevant",   "unexpected"),
        ("irrelevant", "expected"),
        ("irrelevant", "unexpected"),
    ]
    curves = {k: [] for k in keys}
    pred_ch = {k: [] for k in keys}

    rng = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for _ in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg,
            )
            stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)
            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net.forward(packed)
            q_pred_all = aux["q_pred_all"]

            B = r_l23_all.shape[0]
            true_ori = metadata.orientations.to(device)

            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1
                q_pred_isi = q_pred_all[:, t_isi_last, :]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)                      # [B]
                pred_ori = pred_peak_idx.float() * step_deg
                actual_ori = true_ori[:, pres_i]
                is_amb = metadata.is_ambiguous[:, pres_i].to(device)
                pred_error = circular_distance(pred_ori, actual_ori, period)

                is_exp = (pred_error <= 10.0) & (~is_amb)
                is_unexp = (pred_error > 20.0) & (~is_amb)

                true_ch = (actual_ori / step_deg).round().long() % N            # [B]
                t_readout = pres_i * steps_per + T_READ
                r_l23_t = r_l23_all[:, t_readout, :]                            # [B, N]

                ts_this = metadata.task_states[:, pres_i].to(device)            # [B, 2]
                regime_idx = ts_this.argmax(dim=-1)                             # [B]

                r_cpu = r_l23_t.cpu().numpy()
                pred_cpu = pred_peak_idx.cpu().numpy()
                true_cpu = true_ch.cpu().numpy()
                reg_cpu = regime_idx.cpu().numpy()
                exp_cpu = is_exp.cpu().numpy()
                unexp_cpu = is_unexp.cpu().numpy()

                for b in range(B):
                    if int(true_cpu[b]) not in probe_channels:
                        continue
                    regime_name = "relevant" if int(reg_cpu[b]) == 0 else "irrelevant"
                    bucket_name = "expected" if exp_cpu[b] else ("unexpected" if unexp_cpu[b] else None)
                    if bucket_name is None:
                        continue
                    key = (regime_name, bucket_name)
                    curves[key].append(r_cpu[b])
                    pred_ch[key].append(int(pred_cpu[b]))

    out: dict = {}
    for k in keys:
        arr = np.stack(curves[k]) if curves[k] else np.zeros((0, N))
        mean_curve = arr.mean(axis=0) if arr.shape[0] else np.zeros(N)
        mode_pred = circular_mode(np.array(pred_ch[k], dtype=int), N) if pred_ch[k] else -1
        out[k] = {"mean": mean_curve, "n": arr.shape[0], "pred_mode_ch": mode_pred}
    return out, N


# ------------------------------ plot ---------------------------------


def plot_ring(ax, activity: np.ndarray, vmax: float, cmap, step_deg: float,
              probe_ch: int, pred_ch: int | None, show_pred_arrow: bool) -> None:
    """Draw one polar ring heatmap into `ax`."""
    N = len(activity)
    theta_centers = np.arange(N) * (2 * np.pi / N)
    width = 2 * np.pi / N

    inner_r = 0.65
    outer_r = 1.00
    height = outer_r - inner_r

    norm = Normalize(vmin=0.0, vmax=vmax)
    colors = cmap(norm(activity))

    ax.bar(
        theta_centers, height, width=width, bottom=inner_r,
        color=colors, edgecolor="white", linewidth=0.6, align="center",
    )

    # Black probe arrow + label
    probe_theta = probe_ch * (2 * np.pi / N)
    ax.annotate(
        f"probe {int(round(probe_ch * step_deg))}°",
        xy=(probe_theta, outer_r + 0.02),
        xytext=(probe_theta, 1.42),
        ha="center", va="center", color="black", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )

    # Blue predicted arrow + label (unexpected only)
    if show_pred_arrow and pred_ch is not None and pred_ch >= 0:
        pred_theta = pred_ch * (2 * np.pi / N)
        ax.annotate(
            f"predicted {int(round(pred_ch * step_deg))}°",
            xy=(pred_theta, outer_r + 0.02),
            xytext=(pred_theta, 1.42),
            ha="center", va="center", color="tab:blue", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.4),
        )

    # Cardinal labels inside the ring (orientation degrees)
    for deg in [0, 45, 90, 135]:
        rad = deg * 2 * np.pi / 180.0  # orientation → plot angle (0..180° → 0..360°)
        ax.text(rad, 0.28, f"{deg}°", ha="center", va="center",
                color="gray", fontsize=9)

    # Lower-left annotation: total + peak activity (in axes fraction coords)
    total_act = float(np.sum(activity))
    peak_act = float(np.max(activity))
    ax.text(
        0.02, 0.02,
        f"total L2/3 activity: {total_act:.2f}\npeak: {peak_act:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85, linewidth=0.6),
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config/sweep/sweep_rescue_4.yaml")
    p.add_argument("--checkpoint",
                   default="/home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt")
    p.add_argument("--output", default="docs/figures/tuning_ring_heatmap.png")
    p.add_argument("--device", default=None)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe_channels = {11, 12}  # 55°–60° (±5° around 60°)
    PROBE_CH = 12  # draw the probe arrow exactly at 60°

    print(f"[collect] Rescue 4 — device={device} seed={args.seed} n_batches={args.n_batches}",
          flush=True)
    buckets, N = collect(
        args.config, args.checkpoint, device,
        seed=args.seed, n_batches=args.n_batches,
        probe_channels=probe_channels,
    )

    for (reg, bkt), entry in buckets.items():
        pred_deg = entry["pred_mode_ch"] * 5.0 if entry["pred_mode_ch"] >= 0 else float("nan")
        print(f"  {reg:10s} {bkt:10s} n={entry['n']:4d}  "
              f"peak={entry['mean'].max():.3f}  pred_mode_ch={entry['pred_mode_ch']}"
              f" ({pred_deg:.0f}°)", flush=True)

    vmax = max(entry["mean"].max() for entry in buckets.values())
    if vmax <= 0:
        vmax = 1.0
    cmap = cm.get_cmap("viridis")

    fig = plt.figure(figsize=(10, 10))
    row_titles = ["Relevant", "Irrelevant"]
    col_titles = ["Expected", "Unexpected"]
    regime_keys = ["relevant", "irrelevant"]
    bucket_keys = ["expected", "unexpected"]

    for i, regime in enumerate(regime_keys):
        for j, bucket in enumerate(bucket_keys):
            idx = i * 2 + j + 1
            ax = fig.add_subplot(2, 2, idx, projection="polar")
            entry = buckets[(regime, bucket)]
            plot_ring(
                ax, entry["mean"], vmax=vmax, cmap=cmap, step_deg=5.0,
                probe_ch=PROBE_CH,
                pred_ch=entry["pred_mode_ch"],
                show_pred_arrow=(bucket == "unexpected"),
            )
            ax.set_title(f"{row_titles[i]} {col_titles[j]}\n(n={entry['n']})",
                         fontsize=12, pad=18)

    # Shared horizontal colorbar
    cbar_ax = fig.add_axes([0.22, 0.045, 0.56, 0.020])
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("L2/3 activity", fontsize=10)

    fig.suptitle("Orientation ring — L2/3 activity (probe at 60°)",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.945,
        "Total activity summed across 36 channels shows expectation suppression; "
        "per-wedge distribution shows prediction-driven spatial shift.",
        ha="center", va="top", fontsize=10, style="italic", color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.output}", flush=True)


if __name__ == "__main__":
    main()
