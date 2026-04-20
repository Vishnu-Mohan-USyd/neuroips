"""Sprint 5d D5: H-off adaptation baseline.

Protocol (from SPRINT_5D_DIAG_PREREG.md §3 D5):
- build_frozen_network(with_v1_to_h="off", with_feedback_routes=False)
- Run Richter (120 trials) and Tang (360 items) at r=1.0, seeds {42,43,44}.
- Compute per-seed primary metrics:
    - Richter: center_delta (expected − unexpected, matched-channel), redist.
    - Tang: mean_delta_hz (deviant − expected).
    - Tang: svm_accuracy (rotational decoding).
- Retention = |effect_Hoff| / |effect_continuous_seed42_baseline|
- STOP condition: if retention ≥ 0.70 on ≥ 2 metrics (consistent across ≥ 2 of
  3 seeds) → Case C dominates verdict mapping.

Baseline: expectation_snn/data/checkpoints/sprint_5c_intact_r1.0_seed42_continuous_full.npz
  - richter.center_delta = 0.0844 Hz
  - richter.redist       = 0.0844
  - tang.mean_delta_hz   = -0.991 Hz
  - tang.svm_accuracy    = 0.856
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from brian2 import defaultclock, ms, prefs
from brian2 import seed as b2_seed

from expectation_snn.assays.runtime import build_frozen_network
from expectation_snn.assays.richter_crossover import (
    RichterConfig, run_richter_crossover,
)
from expectation_snn.assays.tang_rotating import TangConfig, run_tang_rotating

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms

OUT_DIR = Path("data/diag_sprint5d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline from sprint_5c continuous seed=42 full run (frozen, single-seed)
BASE_RICHTER_CENTER = 0.08444444444444521
BASE_RICHTER_REDIST = 0.08443274853801246
BASE_TANG_MEAN_DELTA = -0.9914937676609666
BASE_TANG_SVM_ACC = 0.8559999999999999


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=".",
        ).decode().strip()
    except Exception:
        return "unknown"


def _bootstrap_ci(x: np.ndarray, n_boot: int = 10_000,
                  seed: int = 99_999, ci: float = 95.0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def run_d5_richter(trial_seed: int) -> Dict[str, Any]:
    t0 = time.time()
    b2_seed(trial_seed)
    np.random.seed(trial_seed)
    bundle = build_frozen_network(
        h_kind="hr", seed=42, r=1.0, g_total=1.0,
        with_cue=False, with_v1_to_h="off", with_feedback_routes=False,
    )
    cfg = RichterConfig(seed=trial_seed)  # default: 30 exp pairs × 6 + 8 unexp × 24 ≈ 120 trials
    res = run_richter_crossover(bundle=bundle, cfg=cfg, seed=trial_seed, verbose=False)
    elapsed = time.time() - t0

    center_delta = float(res.center_vs_flank["center_delta"])
    flank_delta = float(res.center_vs_flank["flank_delta"])
    redist = float(res.center_vs_flank["redist"])

    out = {
        "assay": "richter",
        "trial_seed": trial_seed,
        "checkpoint_seed": 42,
        "center_delta_hoff": center_delta,
        "flank_delta_hoff": flank_delta,
        "redist_hoff": redist,
        "baseline_center_delta": BASE_RICHTER_CENTER,
        "baseline_redist": BASE_RICHTER_REDIST,
        "retention_center": abs(center_delta) / abs(BASE_RICHTER_CENTER)
                            if BASE_RICHTER_CENTER != 0 else float("nan"),
        "retention_redist": abs(redist) / abs(BASE_RICHTER_REDIST)
                            if BASE_RICHTER_REDIST != 0 else float("nan"),
        "raw_trailer_counts_e": np.asarray(res.raw["trailer_counts_e"]),
        "raw_theta_L": np.asarray(res.raw["theta_L"]),
        "raw_theta_T": np.asarray(res.raw["theta_T"]),
        "raw_cond_mask": np.asarray(res.raw["cond_mask"]),
        "raw_dtheta_step": np.asarray(res.raw["dtheta_step"]),
        "elapsed_s": elapsed,
        "git_sha": _git_sha(),
    }
    print(f"[D5/Richter seed={trial_seed}] done in {elapsed:.1f}s | "
          f"center_delta_Hoff={center_delta:+.4f} Hz | "
          f"retention_center={out['retention_center']:.3f} | "
          f"retention_redist={out['retention_redist']:.3f}")
    return out


def run_d5_tang(trial_seed: int) -> Dict[str, Any]:
    t0 = time.time()
    b2_seed(trial_seed)
    np.random.seed(trial_seed)
    bundle = build_frozen_network(
        h_kind="ht", seed=42, r=1.0, g_total=1.0,
        with_cue=False, with_v1_to_h="off", with_feedback_routes=False,
    )
    cfg = TangConfig(seed=trial_seed)
    res = run_tang_rotating(bundle=bundle, cfg=cfg, seed=trial_seed, verbose=False)
    elapsed = time.time() - t0

    mean_delta = float(res.cell_gain["mean_delta_hz"])
    svm_acc = float(res.svm["accuracy"])

    out = {
        "assay": "tang",
        "trial_seed": trial_seed,
        "checkpoint_seed": 42,
        "mean_delta_hz_hoff": mean_delta,
        "svm_accuracy_hoff": svm_acc,
        "baseline_mean_delta_hz": BASE_TANG_MEAN_DELTA,
        "baseline_svm_accuracy": BASE_TANG_SVM_ACC,
        "retention_mean_delta": abs(mean_delta) / abs(BASE_TANG_MEAN_DELTA),
        # SVM retention uses above-chance portion: (acc - 1/n_orient) / (base - 1/n_orient)
        "retention_svm_above_chance": (
            (svm_acc - 1.0 / 6.0) / (BASE_TANG_SVM_ACC - 1.0 / 6.0)
            if BASE_TANG_SVM_ACC > 1.0 / 6.0 else float("nan")
        ),
        "raw_counts_per_item": np.asarray(res.raw["counts_per_item"]),
        "raw_theta_per_item": np.asarray(res.raw["theta_per_item"]),
        "raw_deviant_mask": np.asarray(res.raw["deviant_mask"]),
        "raw_is_random": np.asarray(res.raw["is_random"]),
        "raw_dtheta_prev_step": np.asarray(res.raw["dtheta_prev_step"]),
        "elapsed_s": elapsed,
        "git_sha": _git_sha(),
    }
    print(f"[D5/Tang seed={trial_seed}] done in {elapsed:.1f}s | "
          f"mean_delta_Hoff={mean_delta:+.3f} Hz | "
          f"svm_acc_Hoff={svm_acc:.3f} | "
          f"retention_delta={out['retention_mean_delta']:.3f} | "
          f"retention_svm_above_chance={out['retention_svm_above_chance']:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--assays", type=str, default="richter,tang")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    assays = [a.strip() for a in args.assays.split(",") if a.strip()]

    sha = _git_sha()
    print(f"[D5] git={sha[:8]} seeds={seeds} assays={assays}")

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        if "richter" in assays:
            out_path = OUT_DIR / f"D5_richter_seed{seed}.npz"
            if out_path.exists():
                print(f"[D5/Richter seed={seed}] SKIP (already saved)")
                r = dict(np.load(out_path, allow_pickle=True))
                r = {k: (v.item() if v.ndim == 0 else v) for k, v in r.items()}
                r["assay"] = "richter"
            else:
                r = run_d5_richter(seed)
                np.savez(out_path, **r)
            results.append(r)
        if "tang" in assays:
            out_path = OUT_DIR / f"D5_tang_seed{seed}.npz"
            if out_path.exists():
                print(f"[D5/Tang seed={seed}] SKIP (already saved)")
                r = dict(np.load(out_path, allow_pickle=True))
                r = {k: (v.item() if v.ndim == 0 else v) for k, v in r.items()}
                r["assay"] = "tang"
            else:
                r = run_d5_tang(seed)
                np.savez(out_path, **r)
            results.append(r)

    elapsed = time.time() - t0
    print(f"[D5] ALL DONE in {elapsed:.1f}s")
    print(f"[D5] per-assay retention summary:")
    for assay in ("richter", "tang"):
        rows = [r for r in results if r["assay"] == assay]
        if not rows:
            continue
        if assay == "richter":
            rc = [r["retention_center"] for r in rows]
            rr = [r["retention_redist"] for r in rows]
            print(f"  richter : retention_center (per seed) = "
                  f"[{', '.join(f'{x:.3f}' for x in rc)}]")
            print(f"          : retention_redist (per seed) = "
                  f"[{', '.join(f'{x:.3f}' for x in rr)}]")
        else:
            rd = [r["retention_mean_delta"] for r in rows]
            rs = [r["retention_svm_above_chance"] for r in rows]
            print(f"  tang    : retention_mean_delta (per seed) = "
                  f"[{', '.join(f'{x:.3f}' for x in rd)}]")
            print(f"          : retention_svm_above_chance (per seed) = "
                  f"[{', '.join(f'{x:.3f}' for x in rs)}]")

    # Gate check: retention ≥ 0.70 on ≥ 2 metrics across ≥ 2 of 3 seeds → Case C dominates
    def _passes(vals: List[float]) -> int:
        return sum(1 for v in vals if np.isfinite(v) and v >= 0.70)

    metrics_passing = 0
    if any(r["assay"] == "richter" for r in results):
        rc = [r["retention_center"] for r in results if r["assay"] == "richter"]
        if _passes(rc) >= 2:
            metrics_passing += 1
        rr = [r["retention_redist"] for r in results if r["assay"] == "richter"]
        if _passes(rr) >= 2:
            metrics_passing += 1
    if any(r["assay"] == "tang" for r in results):
        rd = [r["retention_mean_delta"] for r in results if r["assay"] == "tang"]
        if _passes(rd) >= 2:
            metrics_passing += 1
        rs = [r["retention_svm_above_chance"] for r in results if r["assay"] == "tang"]
        if _passes(rs) >= 2:
            metrics_passing += 1

    print(f"[D5] STOP-CONDITION check: {metrics_passing} metrics show "
          f"retention ≥ 0.70 on ≥ 2 of 3 seeds.")
    if metrics_passing >= 2:
        print("[D5] VERDICT: Case C candidate — intrinsic V1 dominates on ≥2 metrics")
    else:
        print("[D5] VERDICT: Case C NOT supported — H feedback required for primary effects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
