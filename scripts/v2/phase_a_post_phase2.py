"""Task #74 Step 2: PHASE A sanity on a post-Phase-2 checkpoint.

Loads a Phase-2 checkpoint (Fix E substrate) and runs the same three
Phase A gates (orientation tuning / contrast response / surround
suppression) as ``phase_a_static_sanity.py`` so we can compare init
numbers to post-training numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.phase_a_static_sanity import (
    contrast_response,
    orientation_tuning,
    surround_suppression,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    bundle = load_checkpoint(args.checkpoint, seed=int(args.seed), device="cpu")
    cfg = bundle.cfg
    net = bundle.net
    net.eval()

    print("[phase_a_post2] orientation tuning ...")
    tune = orientation_tuning(net, cfg)
    print(
        f"  rate_median={tune['rate_median']:.3f} fwhm={tune['fwhm_median_deg']:.1f} "
        f"n_pref={tune['n_preferred_bins_populated']} r_som={tune['r_som_mean']:.2f}"
    )

    print("[phase_a_post2] contrast response ...")
    contrast = contrast_response(net, cfg)
    print(f"  r2_median={contrast['r2_median']:.3f}")

    print("[phase_a_post2] surround suppression ...")
    surround = surround_suppression(net, cfg)
    print(f"  si_median={surround['si_median_pooled']:.3f}")

    result = {
        "version": "phase_a_post_phase2_v1",
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "wall_seconds": float(time.time() - t0),
        "orientation_tuning": tune,
        "contrast_response": contrast,
        "surround_suppression": surround,
        "summary": {
            "rate_median": tune["rate_median"],
            "fwhm_median_deg": tune["fwhm_median_deg"],
            "n_preferred_bins": tune["n_preferred_bins_populated"],
            "r_som_mean_hz": tune["r_som_mean"],
            "contrast_R2_median": contrast["r2_median"],
            "si_median_pooled": surround["si_median_pooled"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[phase_a_post2] wrote {args.output}  wall={result['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
