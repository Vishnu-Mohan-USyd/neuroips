#!/usr/bin/env python3
"""Plot histogram of per-cell L4 OSI from /tmp/l4_osi.json -> /tmp/l4_osi.png.

Source: produced by `./build/v1_test --measure-l4-osi` (task #7).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    json_path = Path("/tmp/l4_osi.json")
    png_path = Path("/tmp/l4_osi.png")

    with json_path.open() as f:
        data = json.load(f)

    osi = np.asarray(data["osi_per_cell"], dtype=np.float64)
    rates = np.asarray(data["mean_rate_hz_per_cell"], dtype=np.float64)
    metrics = data["metrics"]
    n_cells = int(data["n_cells"])
    n_theta = int(data["n_theta"])
    n_reps = int(data["n_reps"])
    duration_ms = int(data["duration_ms"])
    thetas_deg = list(data["thetas_deg"])

    bins = np.linspace(0.0, 1.0, 51)  # 50 bins, width 0.02
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- Panel A: log-y histogram across all L4 cells ----
    ax = axes[0]
    ax.hist(osi, bins=bins, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("OSI (|Σ_θ R(θ) e^{2iθ}| / Σ_θ R(θ))")
    ax.set_ylabel("count (log scale)")
    ax.set_title(
        f"L4 OSI distribution — {n_cells} cells\n"
        f"{n_theta}θ × {n_reps} reps × {duration_ms} ms drift gratings"
    )
    median_osi = float(metrics["median_osi"])
    ax.axvline(median_osi, color="crimson", linestyle="--", linewidth=1.5,
               label=f"median = {median_osi:.3f}")
    ax.axvline(0.2, color="grey", linestyle=":", linewidth=1.0,
               label=(f"0.2 thr (frac> = {metrics['frac_gt_0.2']:.3f})"))
    ax.axvline(0.5, color="orange", linestyle=":", linewidth=1.0,
               label=(f"0.5 thr (frac> = {metrics['frac_gt_0.5']:.3f})"))
    ax.axvline(0.8, color="purple", linestyle=":", linewidth=1.0,
               label=(f"0.8 thr (frac> = {metrics['frac_gt_0.8']:.3f})"))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # ---- Panel B: per-ori_idx median OSI ----
    ax = axes[1]
    per_ori = data["per_ori_idx"]
    oris = [int(p["ori_idx"]) for p in per_ori]
    meds = [float(p["median_osi"]) for p in per_ori]
    ax.bar(oris, meds, color="cornflowerblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(oris)
    ax.set_xticklabels([f"{thetas_deg[o]:g}°" for o in oris], rotation=30)
    ax.set_xlabel("baked-in preferred orientation (ori_idx)")
    ax.set_ylabel("median OSI")
    ax.set_ylim(0.0, max(meds) * 1.15 if meds else 1.0)
    ax.set_title(
        f"per-ori_idx median OSI  (overall median {median_osi:.3f}, "
        f"silent {metrics['frac_silent']:.3f})"
    )
    for o, m in zip(oris, meds):
        ax.text(o, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"all-cell mean rate (Hz): mean={rates.mean():.3f} "
          f"median={np.median(rates):.3f} max={rates.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
