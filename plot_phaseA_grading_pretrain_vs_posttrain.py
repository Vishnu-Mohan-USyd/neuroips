#!/usr/bin/env python3
"""Task #14 aggregator — pre-train vs post-train L2/3 OSI per grading variant.

Reads:
    /tmp/phaseA_grading_<v>_pretrain/v1_v2_phaseA_v2_osi.json   (pre-train, no STDP)
    /tmp/phaseA_grading_<v>/v1_v2_phaseA_v2_osi.json            (post-train, from task #11)

Writes:
    /tmp/phaseA_grading_pretrain_vs_posttrain.png               (single PNG, comparison)
    /tmp/phaseA_grading_pretrain_vs_posttrain.json              (numeric summary)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["random", "am", "sharp", "strict", "gentle"]
PASS = 0.516  # L4 reference median


def load_v2(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        d = json.load(f)
    osi = np.asarray(d["osi_per_cell"], dtype=np.float64)
    return {
        "median":     float(np.median(osi)),
        "frac_gt_0p2": float((osi > 0.2).sum() / osi.size),
        "frac_gt_0p5": float((osi > 0.5).sum() / osi.size),
        "frac_gt_0p8": float((osi > 0.8).sum() / osi.size),
        "frac_eq_0":   float((osi <= 0.0).sum() / osi.size),
        "n_cells":     int(osi.size),
        "osi":         osi,
    }


def main() -> int:
    pre = {v: load_v2(Path(f"/tmp/phaseA_grading_{v}_pretrain/v1_v2_phaseA_v2_osi.json"))
           for v in VARIANTS}
    post = {v: load_v2(Path(f"/tmp/phaseA_grading_{v}/v1_v2_phaseA_v2_osi.json"))
            for v in VARIANTS}

    # ---- JSON summary ----
    summary = {"pass_threshold": PASS, "variants": []}
    print(f"{'variant':<8} {'pre':>8} {'post':>8} {'Δ':>8} "
          f"{'pre>0.5':>8} {'post>0.5':>9} {'pre>0.8':>8} {'post>0.8':>9}")
    for v in VARIANTS:
        a = pre[v]
        b = post[v]
        if not a or not b:
            print(f"{v:<8}  (missing)")
            continue
        delta = b["median"] - a["median"]
        # %-of-final-from-STDP = how much of the post-train value was contributed by STDP.
        # If post < pre (STDP decreased OSI), this is negative.
        # If post > pre (STDP boosted OSI), positive value indicates STDP's positive contribution.
        # Defined as (post - pre) / post for pre+post > 0.
        if abs(b["median"]) < 1e-9:
            stdp_share = 0.0
        else:
            stdp_share = (b["median"] - a["median"]) / b["median"]
        summary["variants"].append({
            "variant": v,
            "pre_train":  {k: a[k] for k in a if k != "osi"},
            "post_train": {k: b[k] for k in b if k != "osi"},
            "delta_median": delta,
            "stdp_share_of_post": stdp_share,
            "passes_l4_ref_pre":  a["median"] >= PASS,
            "passes_l4_ref_post": b["median"] >= PASS,
        })
        print(f"{v:<8} {a['median']:>8.4f} {b['median']:>8.4f} {delta:>+8.4f} "
              f"{a['frac_gt_0p5']:>8.4f} {b['frac_gt_0p5']:>9.4f} "
              f"{a['frac_gt_0p8']:>8.4f} {b['frac_gt_0p8']:>9.4f}")

    out_json = Path("/tmp/phaseA_grading_pretrain_vs_posttrain.json")
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_json}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    n = len(VARIANTS)
    x = np.arange(n)
    bw = 0.36

    # Panel A: bar comparison of medians
    ax = axes[0]
    pre_meds = [pre[v]["median"] for v in VARIANTS]
    post_meds = [post[v]["median"] for v in VARIANTS]
    ax.bar(x - bw/2, pre_meds, bw, label="pre-train (initial random weights)",
           color="darkorange", edgecolor="black")
    ax.bar(x + bw/2, post_meds, bw, label="post-train (1000 trial STDP, task #11)",
           color="steelblue", edgecolor="black")
    ax.axhline(PASS, color="crimson", linestyle="--", linewidth=1.5,
               label=f"L4 ref median = {PASS:.3f}")
    for i, (a_m, b_m) in enumerate(zip(pre_meds, post_meds)):
        ax.text(i - bw/2, a_m + 0.01, f"{a_m:.3f}", ha="center", fontsize=8)
        ax.text(i + bw/2, b_m + 0.01, f"{b_m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("L2/3 median OSI")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Pre-train (connectivity prior only) vs post-train (1000-trial STDP)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Panel B: per-cell histogram overlay (pre vs post) — focus on sharp variant.
    ax = axes[1]
    bins = np.linspace(0, 1, 51)
    colors_pre = {"random": "#fdae6b", "am": "#fd8d3c", "sharp": "#e6550d",
                  "strict": "#a63603", "gentle": "#fdd0a2"}
    colors_post = {"random": "#9ecae1", "am": "#6baed6", "sharp": "#3182bd",
                   "strict": "#08519c", "gentle": "#c6dbef"}
    # Plot pre-train as dotted, post-train as solid (sharp + random as the two anchors).
    for v in ("random", "sharp"):
        a = pre[v]
        b = post[v]
        ax.hist(a["osi"], bins=bins, histtype="step", linestyle=":", linewidth=1.6,
                color=colors_pre[v], label=f"{v} pre  (median={a['median']:.3f})")
        ax.hist(b["osi"], bins=bins, histtype="step", linestyle="-", linewidth=1.6,
                color=colors_post[v], label=f"{v} post (median={b['median']:.3f})")
    ax.axvline(PASS, color="crimson", linestyle="--", linewidth=1.0,
               label=f"L4 ref = {PASS:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("OSI")
    ax.set_ylabel("count (log)")
    ax.set_title("L2/3 per-cell OSI distribution — random + sharp variants")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_png = Path("/tmp/phaseA_grading_pretrain_vs_posttrain.png")
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
