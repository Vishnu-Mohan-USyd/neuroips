#!/usr/bin/env python3
"""Extended ring heatmaps for R4 checkpoint — two figures.

Figure 1 (tuning_ring_recentered.png):
    Every trial's r_l23 is rolled so its true-theta channel lands at index 18.
    2×2 grid (Relevant/Irrelevant × Expected/Unexpected) of viridis ring
    heatmaps, shared vmax. Lower-left annotation per panel shows n, total
    L2/3, peak at the true-orientation channel (index 18), and FWHM of the
    mean re-centered curve. Black arrow labels the "true orientation" at
    the channel-18 wedge.

Figure 2 (tuning_ring_allprobes.png):
    Same 4 buckets, but NO re-centering — mean r_l23 across every trial in
    the bucket regardless of stimulus orientation. 2×2 viridis, shared vmax.
    Lower-left box shows n, total L2/3, and mean per channel (total/36).
    No orientation-specific arrows.

Both figures use:
    - FB ON, late-ON readout (t=9)
    - Expected: V2 ISI pred-error ≤ 10°; Unexpected: pred-error > 20°
    - Ambiguous-stimulus trials excluded
    - Seed 42, n_batches=20 (default), cuda if available

If any Expected bucket has fewer than 500 trials, the script re-runs the
collection with n_batches doubled, up to once.
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


def fwhm_of_curve(curve: np.ndarray, step_deg: float) -> float:
    """FWHM of a 1D circular response profile via linear interpolation.

    Finds the fractional channel offsets at which the curve crosses
    ``half_max = baseline + 0.5 * (peak - baseline)`` on either side of
    the peak, then returns ``(right_frac + left_frac) * step_deg``. This
    resolves sub-bin precision (bin-counting snaps to multiples of
    step_deg and hides sub-5° shape differences).

    Returns NaN if a crossing cannot be found in either direction
    (e.g. the curve stays above half-max for the full ring). Returns 0
    if the curve is effectively flat (peak < 1e-8).
    """
    arr = np.asarray(curve, dtype=float)
    peak = float(arr.max())
    if peak < 1e-8:
        return 0.0
    peak_idx = int(np.argmax(arr))
    baseline = float(arr.min())
    half_max = baseline + 0.5 * (peak - baseline)
    n = len(arr)

    right_cross: float | None = None
    for i in range(1, n):
        idx = (peak_idx + i) % n
        prev_idx = (peak_idx + i - 1) % n
        if arr[idx] <= half_max < arr[prev_idx]:
            frac = (arr[prev_idx] - half_max) / (arr[prev_idx] - arr[idx])
            right_cross = (i - 1) + frac
            break

    left_cross: float | None = None
    for i in range(1, n):
        idx = (peak_idx - i) % n
        prev_idx = (peak_idx - i + 1) % n
        if arr[idx] <= half_max < arr[prev_idx]:
            frac = (arr[prev_idx] - half_max) / (arr[prev_idx] - arr[idx])
            left_cross = (i - 1) + frac
            break

    if right_cross is None or left_cross is None:
        return float("nan")
    return (right_cross + left_cross) * step_deg


# ------------------------------ collect ------------------------------


def collect(config_path: str, checkpoint_path: str, device: torch.device,
            seed: int, n_batches: int) -> tuple[dict, int, float]:
    """Run the trial loop and return (buckets_dict, N, step_deg).

    Each bucket holds raw per-trial r_l23[t=9] (numpy) and true-theta channel
    index. Keys:
        ('relevant',   'expected')
        ('relevant',   'unexpected')
        ('irrelevant', 'expected')
        ('irrelevant', 'unexpected')
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
    r_rows = {k: [] for k in keys}
    true_chs = {k: [] for k in keys}

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
                true_cpu = true_ch.cpu().numpy()
                reg_cpu = regime_idx.cpu().numpy()
                exp_cpu = is_exp.cpu().numpy()
                unexp_cpu = is_unexp.cpu().numpy()

                for b in range(B):
                    regime_name = "relevant" if int(reg_cpu[b]) == 0 else "irrelevant"
                    if exp_cpu[b]:
                        bucket_name = "expected"
                    elif unexp_cpu[b]:
                        bucket_name = "unexpected"
                    else:
                        continue
                    key = (regime_name, bucket_name)
                    r_rows[key].append(r_cpu[b])
                    true_chs[key].append(int(true_cpu[b]))

    out: dict = {}
    for k in keys:
        if r_rows[k]:
            r_arr = np.stack(r_rows[k])                 # [n, N]
            true_arr = np.array(true_chs[k], dtype=int)  # [n]
        else:
            r_arr = np.zeros((0, N))
            true_arr = np.zeros((0,), dtype=int)
        out[k] = {"r": r_arr, "true_ch": true_arr}
    return out, N, step_deg


# ------------------------------ derive means ------------------------------


CENTER_IDX = 18  # channel index where true orientation lands after re-centering


def recentered_mean(entry: dict, N: int) -> np.ndarray:
    r = entry["r"]; true_ch = entry["true_ch"]
    if r.shape[0] == 0:
        return np.zeros(N)
    rolled = np.stack([np.roll(r[i], shift=CENTER_IDX - int(true_ch[i])) for i in range(r.shape[0])])
    return rolled.mean(axis=0)


def pooled_mean(entry: dict, N: int) -> np.ndarray:
    r = entry["r"]
    if r.shape[0] == 0:
        return np.zeros(N)
    return r.mean(axis=0)


# ------------------------------ plot ---------------------------------


def _plot_ring_base(ax, activity: np.ndarray, vmax: float, cmap) -> None:
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

    # Cardinal labels inside the ring
    for deg in [0, 45, 90, 135]:
        rad = deg * 2 * np.pi / 180.0
        ax.text(rad, 0.28, f"{deg}°", ha="center", va="center",
                color="gray", fontsize=9)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)


def plot_ring_recentered(ax, curve: np.ndarray, vmax: float, cmap, step_deg: float,
                         n: int) -> None:
    _plot_ring_base(ax, curve, vmax, cmap)

    total = float(curve.sum())
    peak_at_true = float(curve[CENTER_IDX])
    fwhm = fwhm_of_curve(curve, step_deg)
    ax.text(
        0.02, 0.02,
        f"n = {n}\n"
        f"total L2/3: {total:.2f}\n"
        f"peak @ true (ch {CENTER_IDX}): {peak_at_true:.3f}\n"
        f"FWHM: {fwhm:.1f}°",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85, linewidth=0.6),
    )


def plot_ring_pooled(ax, curve: np.ndarray, vmax: float, cmap, step_deg: float,
                     n: int) -> None:
    N = len(curve)
    _plot_ring_base(ax, curve, vmax, cmap)
    total = float(curve.sum())
    mean_per_ch = total / N
    ax.text(
        0.02, 0.02,
        f"n = {n}\n"
        f"total L2/3: {total:.2f}\n"
        f"mean per channel: {mean_per_ch:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85, linewidth=0.6),
    )


# ------------------------------ figure builders ---------------------


def build_figure(
    buckets: dict, N: int, step_deg: float, mean_fn, plot_fn,
    vmax_src: dict, fig_path: str, suptitle: str, subtitle: str,
) -> list[tuple[str, int, float, float, float]]:
    """Render a 2×2 figure from either re-centered or pooled means.

    Returns a list of (label, n, total, peak, fwhm) stats — peak/fwhm computed
    on the mean vector. (For pooled, peak/fwhm use the pooled curve's max.)
    """
    cmap = cm.get_cmap("viridis")

    # shared vmax across all 4 panels
    vmax = max(float(vmax_src[k].max()) for k in buckets)
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(10, 10))
    row_titles = ["Relevant", "Irrelevant"]
    col_titles = ["Expected", "Unexpected"]
    regime_keys = ["relevant", "irrelevant"]
    bucket_keys = ["expected", "unexpected"]

    stats = []
    for i, regime in enumerate(regime_keys):
        for j, bucket in enumerate(bucket_keys):
            idx = i * 2 + j + 1
            ax = fig.add_subplot(2, 2, idx, projection="polar")
            entry = buckets[(regime, bucket)]
            curve = mean_fn(entry, N)
            n = int(entry["r"].shape[0])
            plot_fn(ax, curve, vmax, cmap, step_deg, n)
            ax.set_title(f"{row_titles[i]} {col_titles[j]}\n(n={n})",
                         fontsize=12, pad=18)
            stats.append((
                f"{row_titles[i]} {col_titles[j]}", n,
                float(curve.sum()), float(curve.max()),
                fwhm_of_curve(curve, step_deg),
            ))

    cbar_ax = fig.add_axes([0.22, 0.045, 0.56, 0.020])
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("L2/3 activity", fontsize=10)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.985)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top",
                 fontsize=10, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))

    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats


# ------------------------------ main ---------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config/sweep/sweep_rescue_4.yaml")
    p.add_argument("--checkpoint",
                   default="/home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt")
    p.add_argument("--output-dir", default="docs/figures")
    p.add_argument("--fig1-name", default="tuning_ring_recentered.png",
                   help="Filename (within --output-dir) for the re-centered figure.")
    p.add_argument("--fig1-title", default="Tuning curves re-centered on true orientation",
                   help="Main title for the re-centered figure.")
    p.add_argument("--skip-fig2", action="store_true",
                   help="Skip the all-probes-pooled figure (only produce re-centered).")
    p.add_argument("--device", default=None)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-exp-n", type=int, default=500,
                   help="Re-collect with 2× batches once if any Expected bucket has fewer trials.")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[collect] R4 — device={device} seed={args.seed} n_batches={args.n_batches}", flush=True)

    buckets, N, step_deg = collect(
        args.config, args.checkpoint, device,
        seed=args.seed, n_batches=args.n_batches,
    )

    n_rel_exp = int(buckets[("relevant", "expected")]["r"].shape[0])
    n_irr_exp = int(buckets[("irrelevant", "expected")]["r"].shape[0])
    print(f"  Expected n: relevant={n_rel_exp}, irrelevant={n_irr_exp}", flush=True)

    if min(n_rel_exp, n_irr_exp) < args.min_exp_n:
        bumped = args.n_batches * 2
        print(f"  below min-exp-n ({args.min_exp_n}); re-collecting with n_batches={bumped}", flush=True)
        buckets, N, step_deg = collect(
            args.config, args.checkpoint, device,
            seed=args.seed, n_batches=bumped,
        )
        n_rel_exp = int(buckets[("relevant", "expected")]["r"].shape[0])
        n_irr_exp = int(buckets[("irrelevant", "expected")]["r"].shape[0])
        print(f"  re-collected Expected n: relevant={n_rel_exp}, irrelevant={n_irr_exp}", flush=True)

    # Precompute both mean vectors (for vmax sharing within each figure)
    recentered = {k: recentered_mean(v, N) for k, v in buckets.items()}
    pooled = {k: pooled_mean(v, N) for k, v in buckets.items()}

    fig1_path = os.path.join(args.output_dir, args.fig1_name)
    fig2_path = os.path.join(args.output_dir, "tuning_ring_allprobes.png")

    stats1 = build_figure(
        buckets=buckets, N=N, step_deg=step_deg,
        mean_fn=recentered_mean, plot_fn=plot_ring_recentered,
        vmax_src=recentered, fig_path=fig1_path,
        suptitle=args.fig1_title,
        subtitle="",
    )
    print(f"[save] {fig1_path}", flush=True)
    print("[fig1 stats]  label               n    total   peak   fwhm(°)", flush=True)
    for label, n, total, peak, fwhm in stats1:
        print(f"  {label:22s} {n:5d}  {total:6.3f}  {peak:5.3f}  {fwhm:6.2f}", flush=True)

    if not args.skip_fig2:
        stats2 = build_figure(
            buckets=buckets, N=N, step_deg=step_deg,
            mean_fn=pooled_mean, plot_fn=plot_ring_pooled,
            vmax_src=pooled, fig_path=fig2_path,
            suptitle="Mean L2/3 activity (all probe orientations pooled)",
            subtitle=(
                "No re-centering — average over every trial in the bucket; "
                "reveals the overall expectation-suppression gap per bucket."
            ),
        )
        print(f"[save] {fig2_path}", flush=True)
        print("[fig2 stats]  label               n    total   peak   fwhm(°)", flush=True)
        for label, n, total, peak, fwhm in stats2:
            print(f"  {label:22s} {n:5d}  {total:6.3f}  {peak:5.3f}  {fwhm:6.2f}", flush=True)


if __name__ == "__main__":
    main()
