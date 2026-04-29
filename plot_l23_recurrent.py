"""Render the five L2/3 recurrent (B1, task #3) PNG artifacts from the
JSON dumps written by `v1_test --enable-l23-recurrent
--load-trained-weights ...`.

Inputs (default paths under /tmp):
    /tmp/l23_recurrent_connectivity.json    (artifacts A, B, C, D source)
    /tmp/l23_recurrent_drive_summary.json   (artifact E source)

Outputs:
    /tmp/l23_recurrent_connectivity.png  A: per-cell fan-in histogram
    /tmp/l23_recurrent_weights.png       B: lognormal EPSP-mV histogram (log-x)
    /tmp/l23_recurrent_distance.png      C: distance histogram + p(d) decay
    /tmp/l23_recurrent_drive.png         E: L2/3 firing-rate histogram

Usage:
    python3 plot_l23_recurrent.py
    python3 plot_l23_recurrent.py --conn /tmp/l23_recurrent_connectivity.json \\
                                  --drive /tmp/l23_recurrent_drive_summary.json \\
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
    ap.add_argument("--conn",  default="/tmp/l23_recurrent_connectivity.json")
    ap.add_argument("--drive", default="/tmp/l23_recurrent_drive_summary.json")
    ap.add_argument("--out",   default="/tmp")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conn = json.loads(Path(args.conn).read_text())

    # ---------- Artifact A: connectivity / fan-in histogram ----------
    fanin = np.asarray(conn["fanin"]["per_cell"], dtype=np.int32)
    n_l23 = conn["n_l23"]
    grid = int(math.sqrt(n_l23 // 16))
    assert grid * grid * 16 == n_l23, "expected 32x32x16 = 16384"

    is_edge = np.zeros(n_l23, dtype=bool)
    for i in range(n_l23):
        gx = (i // 16) % grid
        gy = (i // 16) // grid
        if gx in (0, grid - 1) or gy in (0, grid - 1):
            is_edge[i] = True
    interior_fanin = fanin[~is_edge]
    edge_fanin = fanin[is_edge]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    bins = np.arange(0, fanin.max() + 2)
    ax.hist(fanin, bins=bins, color="0.7",
            label=f"all (n={n_l23})", edgecolor="0.4")
    ax.hist(interior_fanin, bins=bins, color="C0", alpha=0.55,
            label=f"interior (n={len(interior_fanin)})")
    ax.hist(edge_fanin, bins=bins, color="C1", alpha=0.55,
            label=f"edge (n={len(edge_fanin)})")
    ax.axvline(conn["fanin"]["mean"], color="k", lw=1.5,
               label=f"mean = {conn['fanin']['mean']:.1f}")
    ax.axvline(conn["fanin"]["mean_interior"], color="C0", lw=1.2, ls="--",
               label=f"interior mean = {conn['fanin']['mean_interior']:.1f}")
    ax.axvline(conn["fanin"]["mean_edge"], color="C1", lw=1.2, ls="--",
               label=f"edge mean = {conn['fanin']['mean_edge']:.1f}")
    p = conn["params"]
    ax.set_xlabel("recurrent fan-in (# L2/3 partners)")
    ax.set_ylabel("# L2/3 cells")
    ax.set_title(
        "A. L2/3→L2/3 recurrent fan-in distribution (B1 static wiring)\n"
        f"  total_synapses={conn['total_synapses']:,}, "
        f"min={conn['fanin']['min']}, max={conn['fanin']['max']}, "
        f"p(d)={p['p0']}·exp(-d/{p['length_hcol']}), d_max={p['dmax_hcol']}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "l23_recurrent_connectivity.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_recurrent_connectivity.png'}")

    # ---------- Artifact B: weight distribution ----------
    weights = np.asarray(conn["weights_mV"]["all"], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    bins = np.logspace(np.log10(max(weights.min(), 1e-3)),
                       np.log10(weights.max() * 1.05), 60)
    ax.hist(weights, bins=bins, color="C2", edgecolor="0.4", alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(conn["weights_mV"]["median"], color="k", lw=1.5,
               label=f"median = {conn['weights_mV']['median']:.3f} mV")
    ax.axvline(conn["weights_mV"]["mean"], color="0.4", lw=1.2, ls="--",
               label=f"mean = {conn['weights_mV']['mean']:.3f} mV")
    ax.axvline(p["epsp_max_mV"], color="r", lw=1.2, ls=":",
               label=f"hard cap = {p['epsp_max_mV']} mV")
    ax.set_xlabel("EPSP per spike (mV, peak at rest)")
    ax.set_ylabel("# synapses")
    ax.set_title(
        "B. L2/3→L2/3 weight distribution (lognormal init, static)\n"
        f"  μ_log=ln({p['epsp_median_mV']}), "
        f"σ_log={p['epsp_log_sigma']}, clipped at {p['epsp_max_mV']} mV; "
        f"n={conn['weights_mV']['n']:,}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "l23_recurrent_weights.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_recurrent_weights.png'}")

    # ---------- Artifact C: distance histogram + theoretical p(d) ----------
    hist = conn["distance_hist"]
    counts = np.asarray(hist["counts"], dtype=np.int64)
    bw = hist["bin_width_hcol"]
    centers = (np.arange(len(counts)) + 0.5) * bw
    edges_d = np.arange(len(counts) + 1) * bw

    # theoretical curve: count_per_bin ≈ N_post · n_neighbors_at_d · 2 · p_recip(d)
    # (factor 2 for the boost-induced rate doubling at the pair level).
    # The exact curve depends on the discrete lattice neighbours; we plot the
    # raw exp decay × N as a soft reference.
    d_curve = np.linspace(0, p["dmax_hcol"], 200)
    p_curve = p["p0"] * np.exp(-d_curve / p["length_hcol"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.bar(centers, counts, width=bw * 0.9, color="C4",
           edgecolor="0.3", alpha=0.85)
    ax.set_xlim(0, p["dmax_hcol"] * 1.05 + 0.25)
    ax.set_xlabel("inter-hypercolumn distance d (hypercolumns)")
    ax.set_ylabel("# directed edges")
    ax.set_title(
        f"C. Distance histogram (bin width {bw:g} hcol)\n"
        f"   total_synapses={conn['total_synapses']:,}, "
        f"d_max={p['dmax_hcol']}"
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(d_curve, p_curve, "C3-", lw=1.5,
            label=f"p(d) = {p['p0']}·exp(-d/{p['length_hcol']})")
    ax.axvline(p["dmax_hcol"], color="k", lw=1.0, ls=":",
               label=f"cutoff d_max = {p['dmax_hcol']}")
    ax.set_xlabel("d (hypercolumns)")
    ax.set_ylabel("p(d) (per-candidate)")
    ax.set_xlim(0, p["dmax_hcol"] * 1.05 + 0.25)
    ax.set_ylim(0, p["p0"] * 1.1)
    ax.set_title("C′. Theoretical connection probability vs. distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "l23_recurrent_distance.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_recurrent_distance.png'}")

    # ---------- Artifact D: reciprocity (text annotation in title) ----------
    rc = conn["reciprocity_check"]
    print(f"reciprocity (5000 MC): n_sampled={rc['n_sampled']}, "
          f"observed={rc['observed_rate']:.6f}, "
          f"chance={rc['chance_baseline_rate']:.6f}, "
          f"ratio={rc['ratio_to_chance']:.2f}x")
    print(f"reciprocity (global):  n_pairs={rc['global_n_candidate_unordered_pairs']}, "
          f"observed={rc['global_observed_rate']:.6f}, "
          f"chance={rc['global_chance_baseline_rate']:.6f}, "
          f"ratio={rc['global_ratio_to_chance']:.2f}x")

    # ---------- Artifact E: L2/3 firing-rate histogram during drive ----------
    drive = json.loads(Path(args.drive).read_text())
    rates = np.asarray(drive["l23_rate_hz"], dtype=np.float64)
    g = drive["l23_global"]
    stim = drive["stim"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
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
        "E. L2/3 rate during drive (B1: trained L4→L2/3 + recurrent L2/3→L2/3)\n"
        f"   θ={stim['theta_deg']}°, f={stim['f_cyc_per_pixel']:.4f} cyc/px, "
        f"v={stim['v_tf_hz']} Hz, d={stim['d_drift_sign']:+d}, "
        f"duration={drive['duration_ms']} ms, "
        f"frac_silent={g['frac_silent']:.4f}, "
        f"L4_mean={drive['l4_global']['mean_rate_hz']:.2f} Hz"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "l23_recurrent_drive.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out/'l23_recurrent_drive.png'}")


if __name__ == "__main__":
    main()
