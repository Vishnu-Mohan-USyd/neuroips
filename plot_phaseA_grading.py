#!/usr/bin/env python3
"""Aggregate task #11 graded-iso L4→L2/3 sampling sweep results.

Reads:
    /tmp/phaseA_grading_{variant}/v1_v2_phaseA_v2_osi.json    (per-variant)
    /tmp/phaseA_grading_{variant}/grading_metadata.json       (per-variant)
    /tmp/phaseA_grading_{variant}/run.log                     (per-variant; for wall_s)
    /tmp/l4_osi.json                                          (L4 sanity ref)

Writes:
    /tmp/phaseA_grading_summary.json
    /tmp/phaseA_grading_summary.png

Usage:
    python3 plot_phaseA_grading.py [variants...]
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS_DEFAULT = ["random", "am", "sharp", "strict", "gentle"]
PASS_THRESHOLD = 0.516  # L4 reference median OSI from task #7

# Δθ-bin labels (degrees) for documentation in the JSON.
DELTA_THETA_DEG = [0.0, 22.5, 45.0, 67.5, 90.0]

# Mirror of grading_curve() in adex_v1.cu (task #11).
W_CURVES = {
    "random": [1.0, 1.0, 1.0, 1.0, 1.0],
    "am":     [0.55, 0.50, 0.20, 0.31, 0.0],
    "sharp":  [1.0, 0.40, 0.10, 0.05, 0.0],
    "strict": [1.0, 0.20, 0.0, 0.0, 0.0],
    "gentle": [1.0, 0.80, 0.50, 0.30, 0.10],
}

# Candidate counts per Δθ-bin for the full 3×3 patch (interior cell).
N_CAND = [144, 288, 288, 288, 144]
TARGET_FANIN = 40


def grading_params(variant: str) -> dict:
    """Mirror of C++ compute_grading_params(): returns scaling + p_connect_per_bin."""
    w = W_CURVES.get(variant)
    if w is None:
        return {}
    unnorm = sum(wi * ni for wi, ni in zip(w, N_CAND))
    if unnorm <= 0.0:
        return {"variant": variant, "w_curve": w, "scaling": 0.0,
                "p_connect_per_bin": [0.0] * 5, "expected_fanin_interior": 0.0}
    scaling = TARGET_FANIN / unnorm
    p_per = [min(wi * scaling, 1.0) for wi in w]
    fanin = sum(p * n for p, n in zip(p_per, N_CAND))
    return {
        "variant":                  variant,
        "w_curve":                  w,
        "scaling":                  scaling,
        "p_connect_per_bin":        p_per,
        "expected_fanin_interior":  fanin,
    }


def load_variant(variant: str, base_dir: Path) -> dict:
    var_dir = base_dir / f"phaseA_grading_{variant}"
    out = {"variant": variant, "dir": str(var_dir)}

    v2_path = var_dir / "v1_v2_phaseA_v2_osi.json"
    meta_path = var_dir / "grading_metadata.json"
    log_path = var_dir / "run.log"

    if not v2_path.exists():
        out["error"] = f"missing {v2_path}"
        return out

    with v2_path.open() as f:
        v2 = json.load(f)
    osi = np.asarray(v2["osi_per_cell"], dtype=np.float64)

    out["osi_median"]     = float(np.median(osi))
    out["frac_osi_gt_0p2"] = float((osi > 0.2).sum() / osi.size)
    out["frac_osi_gt_0p5"] = float((osi > 0.5).sum() / osi.size)
    out["frac_osi_gt_0p8"] = float((osi > 0.8).sum() / osi.size)
    out["frac_osi_eq_0"]   = float((osi <= 0.0).sum() / osi.size)
    out["n_l23_cells"]     = int(osi.size)
    out["osi_array"]       = osi   # kept for plotting; popped before JSON write

    # L2/3 silent fraction (rate < 0.1 Hz @ θ=0, 1s drifting grating).
    # Read from V5 diagnostic JSON: { "l23": { "frac_silent": ... } }.
    # Fall back to "frac of OSI == 0" if V5 is missing/malformed (strict subset).
    v5_path = var_dir / "v1_v2_phaseA_v5_diag.json"
    if v5_path.exists():
        try:
            with v5_path.open() as f:
                v5 = json.load(f)
            l23_block = v5.get("l23")
            if isinstance(l23_block, dict) and "frac_silent" in l23_block:
                out["frac_silent"] = float(l23_block["frac_silent"])
                out["frac_silent_source"] = "v5.l23.frac_silent (θ=0, 1s)"
            else:
                out["frac_silent"] = out["frac_osi_eq_0"]
                out["frac_silent_source"] = "fallback:osi==0 (v5 key missing)"
        except Exception as exc:
            out["frac_silent"] = out["frac_osi_eq_0"]
            out["frac_silent_source"] = f"v5_load_failed:{exc}"
    else:
        out["frac_silent"] = out["frac_osi_eq_0"]
        out["frac_silent_source"] = "fallback:osi==0 (v5 file missing)"

    # Variant grading params: deterministic from variant name (mirrors C++).
    gp = grading_params(variant)
    out.update({
        "w_curve":                 gp.get("w_curve"),
        "scaling":                 gp.get("scaling"),
        "p_per_bin":               gp.get("p_connect_per_bin"),
        "expected_fanin_interior": gp.get("expected_fanin_interior"),
    })

    # Pull additional fields (training wall, fanin, weight stats) from the
    # weights-sibling JSON (existing --save-trained-weights output).
    sibling_json = var_dir / "trained_weights.json"
    if sibling_json.exists():
        try:
            with sibling_json.open() as f:
                sib = json.load(f)
            out["training_wall_s"]  = sib.get("train_wall_s")
            out["n_train_trials"]   = sib.get("n_train_trials")
            out["train_stim_ms"]    = sib.get("train_stim_ms")
            out["train_iti_ms"]     = sib.get("train_iti_ms")
            out["fanin_actual"]     = sib.get("fanin")
            out["weights_nS_stats"] = sib.get("weights_nS")
            out["seed"]             = sib.get("seed")
            out["runaway"]          = sib.get("runaway")
        except Exception as exc:
            out["sibling_json_load_failed"] = str(exc)
    # Fallback: parse training_wall_s from log.
    if "training_wall_s" not in out and log_path.exists():
        m = re.search(r"training_wall_s=([0-9.]+)", log_path.read_text())
        if m:
            out["training_wall_s"] = float(m.group(1))

    # Per-variant L4 OSI sanity (run with --measure-l4-osi after this
    # variant's training, written to <var_dir>/l4_osi_sanity.json).
    l4_sanity_path = var_dir / "l4_osi_sanity.json"
    if l4_sanity_path.exists():
        try:
            with l4_sanity_path.open() as f:
                lj = json.load(f)
            out["l4_sanity_per_variant"] = {
                "median_osi":  float(lj["metrics"]["median_osi"]),
                "frac_gt_0p2": float(lj["metrics"]["frac_gt_0.2"]),
                "frac_gt_0p5": float(lj["metrics"]["frac_gt_0.5"]),
                "frac_gt_0p8": float(lj["metrics"]["frac_gt_0.8"]),
                "frac_silent": float(lj["metrics"]["frac_silent"]),
                "n_cells":     int(lj["n_cells"]),
            }
        except Exception as exc:
            out["l4_sanity_per_variant"] = {"error": str(exc)}
    return out


def load_l4_sanity() -> dict | None:
    p = Path("/tmp/l4_osi.json")
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    return {
        "median_osi":  float(d["metrics"]["median_osi"]),
        "frac_gt_0p2": float(d["metrics"]["frac_gt_0.2"]),
        "frac_gt_0p5": float(d["metrics"]["frac_gt_0.5"]),
        "frac_gt_0p8": float(d["metrics"]["frac_gt_0.8"]),
        "frac_silent": float(d["metrics"]["frac_silent"]),
        "n_cells":     int(d["n_cells"]),
    }


def main() -> int:
    variants = sys.argv[1:] if len(sys.argv) > 1 else VARIANTS_DEFAULT
    base = Path("/tmp")
    rows = [load_variant(v, base) for v in variants]
    l4 = load_l4_sanity()

    summary = {
        "pass_threshold_median_osi": PASS_THRESHOLD,
        "delta_theta_deg_per_bin":   DELTA_THETA_DEG,
        "l4_sanity": l4,
        "variants": [],
    }
    for r in rows:
        d = {k: v for k, v in r.items() if k != "osi_array"}
        d["pass_meets_l4_median"] = (
            "osi_median" in r and r["osi_median"] >= PASS_THRESHOLD
        )
        summary["variants"].append(d)

    out_json = base / "phaseA_grading_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")

    # ---- Plot ----
    valid = [r for r in rows if "osi_median" in r]
    if not valid:
        print("no valid variants to plot; skipping PNG")
        return 0

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel A: bar chart of OSI median + frac > 0.5 / 0.8 alongside.
    ax = axes[0]
    names = [r["variant"] for r in valid]
    meds  = [r["osi_median"]      for r in valid]
    f02   = [r["frac_osi_gt_0p2"] for r in valid]
    f05   = [r["frac_osi_gt_0p5"] for r in valid]
    f08   = [r["frac_osi_gt_0p8"] for r in valid]
    fsil  = [r["frac_silent"]     for r in valid]
    x = np.arange(len(names))
    bw = 0.18
    ax.bar(x - 2 * bw, meds, bw, color="steelblue",  label="median OSI")
    ax.bar(x - 1 * bw, f02,  bw, color="lightgreen", label="frac > 0.2")
    ax.bar(x + 0 * bw, f05,  bw, color="orange",     label="frac > 0.5")
    ax.bar(x + 1 * bw, f08,  bw, color="purple",     label="frac > 0.8")
    ax.bar(x + 2 * bw, fsil, bw, color="grey",       label="frac silent")
    ax.axhline(PASS_THRESHOLD, color="crimson", linestyle="--", linewidth=1.5,
               label=f"L4 ref = {PASS_THRESHOLD:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("Phase A graded-iso sampling sweep — L2/3 OSI metrics per variant")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for xi, m in zip(x, meds):
        ax.text(xi - 2 * bw, m + 0.01, f"{m:.3f}", ha="center", fontsize=7)

    # Panel B: per-cell OSI histograms overlaid (log-y).
    ax = axes[1]
    bins = np.linspace(0.0, 1.0, 51)
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "purple", "teal"]
    for i, r in enumerate(valid):
        ax.hist(r["osi_array"], bins=bins, histtype="step", linewidth=1.6,
                color=colors[i % len(colors)],
                label=f"{r['variant']} (median={r['osi_median']:.3f})")
    ax.axvline(PASS_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"L4 ref = {PASS_THRESHOLD:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("OSI")
    ax.set_ylabel("count (log)")
    ax.set_title("L2/3 per-cell OSI distribution (overlay)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_png = base / "phaseA_grading_summary.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")

    # ---- Console table ----
    print()
    print(f"{'variant':<8} {'median':>8} {'>0.2':>7} {'>0.5':>7} {'>0.8':>7} "
          f"{'silent':>7} {'fanin':>7} {'wall_s':>9} {'pass':>5}")
    for r in valid:
        passed = r["osi_median"] >= PASS_THRESHOLD
        print(f"{r['variant']:<8} "
              f"{r['osi_median']:>8.4f} "
              f"{r['frac_osi_gt_0p2']:>7.4f} "
              f"{r['frac_osi_gt_0p5']:>7.4f} "
              f"{r['frac_osi_gt_0p8']:>7.4f} "
              f"{r['frac_silent']:>7.4f} "
              f"{r.get('expected_fanin_interior', float('nan')):>7.2f} "
              f"{r.get('training_wall_s', float('nan')):>9.2f} "
              f"{'YES' if passed else 'no':>5}")
    if l4:
        print(f"{'L4 ref':<8} "
              f"{l4['median_osi']:>8.4f} "
              f"{l4['frac_gt_0p2']:>7.4f} "
              f"{l4['frac_gt_0p5']:>7.4f} "
              f"{l4['frac_gt_0p8']:>7.4f} "
              f"{l4['frac_silent']:>7.4f}")
    print()
    print("L4 OSI sanity per variant (should all match L4 ref across all 5):")
    print(f"{'variant':<8} {'L4_med':>8} {'>0.2':>7} {'>0.5':>7} {'>0.8':>7} {'silent':>7}")
    for r in valid:
        s = r.get("l4_sanity_per_variant")
        if s and "median_osi" in s:
            print(f"{r['variant']:<8} "
                  f"{s['median_osi']:>8.4f} "
                  f"{s['frac_gt_0p2']:>7.4f} "
                  f"{s['frac_gt_0p5']:>7.4f} "
                  f"{s['frac_gt_0p8']:>7.4f} "
                  f"{s['frac_silent']:>7.4f}")
        else:
            print(f"{r['variant']:<8}  (no l4_sanity_per_variant on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
