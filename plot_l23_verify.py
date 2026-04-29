"""Render the four L2/3 Phase A verify PNG artifacts from the JSON dumps
written by `v1_test --enable-l23 --verify`.

Inputs (default paths under /tmp):
    /tmp/l23_connectivity.json    fan-in / weights / partner profiles
    /tmp/l23_drive_summary.json   per-cell L2/3 firing rates

Outputs:
    /tmp/l23_connectivity.png     A: per-cell fan-in histogram (interior/edge)
    /tmp/l23_weights.png          B: lognormal EPSP-mV histogram (log-x)
    /tmp/l23_drive_theta0.png     C: L2/3 firing-rate histogram
    /tmp/l23_partner_profile.png  D: 8 sample cells x 8 ori bar plots

Usage:
    python3 plot_l23_verify.py
    python3 plot_l23_verify.py --conn /tmp/l23_connectivity.json \
                               --drive /tmp/l23_drive_summary.json \
                               --out /tmp
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn",  default="/tmp/l23_connectivity.json")
    ap.add_argument("--drive", default="/tmp/l23_drive_summary.json")
    ap.add_argument("--out",   default="/tmp")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conn = json.loads(Path(args.conn).read_text())
    drive = json.loads(Path(args.drive).read_text())

    # ---------- Artifact A: connectivity histogram ----------
    fanin_per_cell = np.asarray(conn["fanin"]["per_cell"], dtype=np.int32)
    n_l23 = conn["n_l23"]
    grid = int(math.sqrt(n_l23 // 16))   # 16 clones per hypercolumn
    assert grid * grid * 16 == n_l23, "expected 32x32x16 = 16384"

    is_edge = np.zeros(n_l23, dtype=bool)
    for i in range(n_l23):
        gx = (i // 16) % grid
        gy = (i // 16) // grid
        if gx in (0, grid - 1) or gy in (0, grid - 1):
            is_edge[i] = True
    interior_fanin = fanin_per_cell[~is_edge]
    edge_fanin = fanin_per_cell[is_edge]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.arange(0, fanin_per_cell.max() + 2)
    ax.hist(fanin_per_cell, bins=bins, color="0.7",
            label=f"all cells (n={n_l23})", edgecolor="0.4")
    ax.hist(interior_fanin, bins=bins, color="C0", alpha=0.55,
            label=f"interior (n={len(interior_fanin)})")
    ax.hist(edge_fanin, bins=bins, color="C1", alpha=0.55,
            label=f"edge (n={len(edge_fanin)})")
    ax.axvline(conn["fanin"]["mean"], color="k", lw=1.5,
               label=f"mean = {conn['fanin']['mean']:.1f}")
    ax.axvline(conn["fanin"]["mean_interior"], color="C0", lw=1.2,
               ls="--", label=f"interior mean = {conn['fanin']['mean_interior']:.1f}")
    ax.axvline(conn["fanin"]["mean_edge"], color="C1", lw=1.2,
               ls="--", label=f"edge mean = {conn['fanin']['mean_edge']:.1f}")
    target = conn["params"]["target_fanin"]
    ax.axvline(target, color="r", lw=1.2, ls=":",
               label=f"target ({target}) = {target}/{conn['params']['candidate_pool_interior']} interior")
    ax.set_xlabel("fan-in (# L4 partners)")
    ax.set_ylabel("# L2/3 cells")
    ax.set_title(
        f"A. L4→L2/3 fan-in distribution (Phase A static wiring)\n"
        f"  total_synapses={conn['total_synapses']}, "
        f"min={conn['fanin']['min']}, max={conn['fanin']['max']}, "
        f"p_connect={conn['params']['p_connect']:.4f}"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "l23_connectivity.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_connectivity.png'}")

    # ---------- Artifact B: weight distribution ----------
    weights_mV = np.asarray(conn["weights_mV"]["all"], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.logspace(np.log10(max(weights_mV.min(), 1e-3)),
                       np.log10(weights_mV.max() * 1.05), 60)
    ax.hist(weights_mV, bins=bins, color="C2", edgecolor="0.4", alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(conn["weights_mV"]["median"], color="k", lw=1.5,
               label=f"median = {conn['weights_mV']['median']:.3f} mV")
    ax.axvline(conn["weights_mV"]["mean"], color="0.4", lw=1.2, ls="--",
               label=f"mean = {conn['weights_mV']['mean']:.3f} mV")
    ax.axvline(conn["params"]["epsp_max_mV"], color="r", lw=1.2, ls=":",
               label=f"hard cap = {conn['params']['epsp_max_mV']} mV")
    ax.set_xlabel("EPSP per spike (mV, peak at rest)")
    ax.set_ylabel("# synapses")
    ax.set_title(
        f"B. L4→L2/3 weight distribution (lognormal init)\n"
        f"  μ_log=ln({conn['params']['epsp_median_mV']}), "
        f"σ_log={conn['params']['epsp_log_sigma']}, "
        f"clipped at {conn['params']['epsp_max_mV']} mV;  "
        f"n={conn['weights_mV']['n']:,}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "l23_weights.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_weights.png'}")

    # ---------- Artifact C: L2/3 firing-rate histogram ----------
    rates = np.asarray(drive["l23_rate_hz"], dtype=np.float64)
    g = drive["l23_global"]
    stim = drive["stim"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins_rate = np.linspace(0.0, max(1.0, rates.max() * 1.05), 80)
    ax.hist(rates, bins=bins_rate, color="C3", edgecolor="0.4", alpha=0.85)
    ax.axvline(g["mean_rate_hz"],   color="k", lw=1.5,
               label=f"mean = {g['mean_rate_hz']:.3f} Hz")
    ax.axvline(g["median_rate_hz"], color="0.4", lw=1.2, ls="--",
               label=f"median = {g['median_rate_hz']:.3f} Hz")
    ax.axvline(g["p95_rate_hz"],    color="C0", lw=1.2, ls="-.",
               label=f"p95 = {g['p95_rate_hz']:.3f} Hz")
    ax.axvline(g["max_rate_hz"],    color="C2", lw=1.2, ls=":",
               label=f"max = {g['max_rate_hz']:.3f} Hz")
    ax.set_xlabel("L2/3 firing rate (Hz)")
    ax.set_ylabel("# L2/3 cells")
    ax.set_yscale("log")
    ax.set_title(
        f"C. L2/3 firing rate at θ={stim['theta_deg']}°, f={stim['f_cyc_per_pixel']:.4f} cyc/px, "
        f"v={stim['v_tf_hz']} Hz, d={stim['d_drift_sign']:+d}\n"
        f"  duration={drive['duration_ms']} ms, n_l23={drive['n_l23']}, "
        f"frac_silent={g['frac_silent']:.4f}, "
        f"L4_mean_rate={drive['l4_global']['mean_rate_hz']:.2f} Hz"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "l23_drive_theta0.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_drive_theta0.png'}")

    # ---------- Artifact D: per-L2/3-cell L4-partner orientation profile ----------
    profiles = conn["partner_profiles"]
    n_p = len(profiles)
    nrows = (n_p + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    target = conn["params"]["target_fanin"] / 8  # equal share if unbiased
    for s, pp in enumerate(profiles):
        ax = axes[s // 4, s % 4]
        oris = np.arange(8)
        counts = np.asarray(pp["by_ori_count"], dtype=np.int32)
        ax.bar(oris, counts, color="C4", edgecolor="0.3")
        ax.axhline(counts.sum() / 8, color="k", lw=1.0, ls="--",
                   label=f"equal share = {counts.sum()/8:.1f}")
        ax.set_title(
            f"l23_idx={pp['l23_idx']} (gx={pp['gx']}, gy={pp['gy']})\n"
            f"  fanin={pp['fanin']}",
            fontsize=9,
        )
        ax.set_xlabel("L4 ori_idx (×22.5°)")
        ax.set_ylabel("# partners")
        ax.set_xticks(oris)
        ax.legend(fontsize=7, loc="upper right")
    # blank any unused subplots
    for s in range(n_p, nrows * 4):
        axes[s // 4, s % 4].axis("off")
    fig.suptitle(
        "D. Per-L2/3-cell L4-partner orientation profile\n"
        "(equal-across-orientations expected ⇒ no sampling bias)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "l23_partner_profile.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_partner_profile.png'}")


if __name__ == "__main__":
    main()
