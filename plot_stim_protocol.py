"""Render the four #53 stim-protocol-check PNG artifacts from the JSON dumps
written by `v1_test --stim-protocol-check`.

Inputs (defaults; per task #53 spec):
    /tmp/stim_check_l4_rates.json        B+C source data + snapshot params
    /tmp/l23_responses_per_variant.json  D source data

Outputs:
    /tmp/stim_variants_snapshots.png      A: per-variant intensity snapshots
    /tmp/l4_transient_rates.png           B: per-variant L4 peak-rate histograms
    /tmp/l4_decorrelation.png             C: per-variant cross-HC correlation hist
    /tmp/l23_responses_per_variant.png    D: per-variant L2/3 response stats

Usage:
    python3 plot_stim_protocol.py
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRID = 32
PI = math.pi


def render_grating_intensity(theta_deg, f_cyc_per_px, v_tf_hz, drift_sign,
                             phi_phase_rad, x_origin, y_origin,
                             aperture_active, aperture_cx, aperture_cy,
                             aperture_sigma, t_s):
    """Compute the 32x32 stim intensity grid for diagnostic rendering."""
    th = theta_deg * PI / 180.0
    K = 2.0 * PI * f_cyc_per_px
    omega = 2.0 * PI * v_tf_hz * drift_sign
    xs = np.arange(GRID)
    ys = np.arange(GRID)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    phase = (K * ((XX - x_origin) * np.cos(th)
                  + (YY - y_origin) * np.sin(th))
             - omega * t_s + phi_phase_rad)
    intensity = 0.5 * (1.0 + np.cos(phase))   # ∈ [0, 1]
    if aperture_active:
        if aperture_sigma <= 0:
            mask = np.ones_like(intensity)
        else:
            inv_2sigma_sq = 1.0 / (2.0 * aperture_sigma**2)
            dxc = XX - aperture_cx
            dyc = YY - aperture_cy
            mask = np.exp(-(dxc**2 + dyc**2) * inv_2sigma_sq)
        intensity = intensity * mask
    return intensity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l4",   default="/tmp/stim_check_l4_rates.json",
                    help="artifact B + C source data")
    ap.add_argument("--l23",  default="/tmp/l23_responses_per_variant.json",
                    help="artifact D source data")
    ap.add_argument("--out",  default="/tmp")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d_l4  = json.loads(Path(args.l4).read_text())
    d_l23 = json.loads(Path(args.l23).read_text())
    variants = d_l4["variants"]
    l23_by_name = {v["name"]: v for v in d_l23["variants"]}

    # ---------- Artifact A: stim variant snapshots ----------
    n_var = len(variants)
    fig, axes = plt.subplots(n_var, 4, figsize=(11, 2.4 * n_var))
    axes = np.atleast_2d(axes)
    snapshot_times = [0.0, 0.250, 0.500, 0.750]
    for v_idx, v in enumerate(variants):
        params_list = v.get("snapshot_params", [])
        if not params_list:
            for j in range(4):
                axes[v_idx, j].axis("off")
            continue
        # Use the first snapshot's params, vary t.
        sp = params_list[0]
        for j, t_s in enumerate(snapshot_times):
            ax = axes[v_idx, j]
            img = render_grating_intensity(
                theta_deg=sp["theta_deg"],
                f_cyc_per_px=sp["f_cyc_per_px"],
                v_tf_hz=sp["v_tf_hz"],
                drift_sign=+1,
                phi_phase_rad=sp["phi_phase_rad"],
                x_origin=sp["x_origin"],
                y_origin=sp["y_origin"],
                aperture_active=sp["aperture_active"],
                aperture_cx=sp["aperture_cx"],
                aperture_cy=sp["aperture_cy"],
                aperture_sigma=sp["aperture_sigma"],
                t_s=t_s,
            )
            ax.imshow(img, vmin=0, vmax=1, cmap="gray", origin="lower")
            ax.set_xticks([0, 8, 16, 24, 31])
            ax.set_yticks([0, 8, 16, 24, 31])
            if j == 0:
                title = (f"{v['name']}: θ={sp['theta_deg']:.1f}°, "
                         f"f={sp['f_cyc_per_px']:.4f}, ϕ={sp['phi_phase_rad']:.2f}\n"
                         f"x0={sp['x_origin']:.2f}, y0={sp['y_origin']:.2f}"
                         + (f", ap=({sp['aperture_cx']:.1f},{sp['aperture_cy']:.1f}) σ={sp['aperture_sigma']:.1f}"
                            if sp["aperture_active"] else ", no aperture")
                         + f"\nt={t_s*1000:.0f} ms")
            else:
                title = f"t={t_s*1000:.0f} ms"
            ax.set_title(title, fontsize=7)
    fig.suptitle("A. Stim-variant intensity snapshots (t = 0, 250, 500, 750 ms)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "stim_variants_snapshots.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out/'stim_variants_snapshots.png'}")

    # ---------- Artifact B: L4 transient peak rates ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    bins_rate = np.linspace(0.0, 250.0, 60)
    for v_idx, v in enumerate(variants):
        ax = axes[v_idx]
        rates = np.asarray(v["peak_rate_per_cell_hz"], dtype=np.float64)
        ax.hist(rates, bins=bins_rate, color="C0", edgecolor="0.4", alpha=0.85)
        ax.axvline(v["peak_rate_mean_hz"], color="k", lw=1.2,
                   label=f"mean={v['peak_rate_mean_hz']:.1f} Hz")
        ax.axvline(v["peak_rate_median_hz"], color="0.4", lw=1.0, ls="--",
                   label=f"median={v['peak_rate_median_hz']:.1f} Hz")
        ax.axvline(v["peak_rate_p95_hz"], color="C1", lw=1.0, ls="-.",
                   label=f"p95={v['peak_rate_p95_hz']:.1f} Hz")
        ax.axvline(10.0, color="r", lw=1.0, ls=":", label="10 Hz threshold")
        ax.set_yscale("log")
        ax.set_xlabel("per-cell peak rate (Hz, 20 ms window, max over trials)")
        ax.set_ylabel("# L4 cells")
        ax.set_title(
            f"{v['name']}: n_trials={v['n_trials']}\n"
            f"frac>=10Hz={v['frac_cells_reaching_10hz']:.3f}, "
            f">=20Hz={v['frac_cells_reaching_20hz']:.3f}, "
            f"L4_mean={v['l4_mean_rate_hz']:.2f} Hz",
            fontsize=9,
        )
        ax.legend(fontsize=7)
    for k in range(len(variants), len(axes)):
        axes[k].axis("off")
    fig.suptitle("B. Per-variant L4 transient peak-rate histograms (across trials)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "l4_transient_rates.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l4_transient_rates.png'}")

    # ---------- Artifact C: cross-HC correlation histograms ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    bins_corr = np.linspace(-1.0, 1.0, 41)
    full_mean_abs = None
    for v_idx, v in enumerate(variants):
        if v["name"] == "full":
            full_mean_abs = v.get(
                "mean_pairwise_corr_across_hcols_abs",
                np.mean(np.abs(np.asarray(v["pairwise_corrs"]))),
            )
    for v_idx, v in enumerate(variants):
        ax = axes[v_idx]
        corrs = np.asarray(v["pairwise_corrs"], dtype=np.float64)
        mean_r    = v.get("mean_pairwise_corr_across_hcols", float(corrs.mean()))
        mean_absr = v.get("mean_pairwise_corr_across_hcols_abs",
                          float(np.abs(corrs).mean()))
        rel = mean_absr / full_mean_abs if full_mean_abs else 1.0
        ax.hist(corrs, bins=bins_corr, color="C2", edgecolor="0.4", alpha=0.85)
        ax.axvline(mean_r,    color="k", lw=1.5, label=f"mean(r)={mean_r:+.4f}")
        ax.axvline(mean_absr, color="C3", lw=1.2, ls="--",
                   label=f"mean(|r|)={mean_absr:.4f}")
        ax.axvline(-mean_absr, color="C3", lw=1.2, ls="--", alpha=0.4)
        ax.axvline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_xlabel("pairwise Pearson r (50 ms bins, concatenated across trials)")
        ax.set_ylabel("# pairs")
        ax.set_xlim(-1, 1)
        ax.set_title(
            f"{v['name']}: n_pairs={len(corrs)}, n_trials={v['n_trials']}\n"
            f"|r|/full_|r|={rel:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
    for k in range(len(variants), len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        "C. Per-variant L4 cross-hypercolumn pairwise correlation histograms\n"
        "Decorrelation metric = mean(|r|); aperture targets <50% of full's mean(|r|).",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "l4_decorrelation.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l4_decorrelation.png'}")

    # ---------- Artifact D: L2/3 responses per variant ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    names = [v["name"] for v in variants]
    # Pull from the dedicated l23 JSON for each variant.
    def _l23(name, key):
        return l23_by_name[name][key]
    means = np.asarray([_l23(n, "l23_mean_rate_hz") for n in names])
    p95s  = np.asarray([_l23(n, "l23_p95_rate_hz")  for n in names])
    maxes = np.asarray([_l23(n, "l23_max_rate_hz")  for n in names])
    silents = np.asarray([_l23(n, "l23_frac_silent") for n in names])
    x = np.arange(len(names))
    w = 0.22
    ax.bar(x - 1.5*w, means,   w, label="mean", color="C0")
    ax.bar(x - 0.5*w, p95s,    w, label="p95",  color="C1")
    ax.bar(x + 0.5*w, maxes,   w, label="max",  color="C2")
    ax2 = ax.twinx()
    ax2.bar(x + 1.5*w, silents, w, label="frac_silent (right axis)",
            color="C3", alpha=0.7)
    ax2.set_ylabel("frac_silent (rate < 0.1 Hz)")
    ax2.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("L2/3 firing rate (Hz)")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, max(50.0, maxes.max() * 1.1))
    ax.set_title(
        "D. L2/3 driven response per variant (averaged across trials)\n"
        "Pass: mixed mean ∈ [0.5, 5] Hz, no runaway",
        fontsize=11,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "l23_responses_per_variant.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_responses_per_variant.png'}")


if __name__ == "__main__":
    main()
