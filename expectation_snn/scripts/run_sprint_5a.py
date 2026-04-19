"""Sprint 5a driver: run Kok + Richter + Tang at intact r=1.0, seed=42.

Task #27 Step 5. Executes the three primary-metric assays sequentially
on the frozen network (Stage 0 + Stage 1 H_R + Stage 1 H_T + Stage 2 cue
already trained and checkpointed), collects observed metrics, and
persists them to ``data/checkpoints/sprint_5a_intact_r1_seed42.npz``.

Per task #27:
    - seed = 42 only (multi-seed in Sprint 5b).
    - r = 1.0 (balanced), g_total = 1.0.
    - intact only (no ablations).

Wall-clock estimate (Lead dispatch):
    Kok  ~ 16 min,   Richter ~ 36 min,   Tang ~ 4 min.
    Total ~ 1 h at defaults.

Usage
-----
    python -m expectation_snn.scripts.run_sprint_5a
    # or with --dry-run for a quick pipeline check at tiny n:
    python -m expectation_snn.scripts.run_sprint_5a --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from expectation_snn.assays.runtime import build_frozen_network
from expectation_snn.assays.kok_passive import KokConfig, run_kok_passive
from expectation_snn.assays.richter_crossover import (
    RichterConfig, run_richter_crossover,
)
from expectation_snn.assays.tang_rotating import TangConfig, run_tang_rotating


DEFAULT_CKPT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "checkpoints"
)


def _flatten_for_npz(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested result dict for ``np.savez``.

    numpy.savez can only store array-like + scalar values; we pack nested
    dicts into dotted keys and drop non-serializable entries (bundle
    metadata strings are fine as 0-d arrays).
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_for_npz(full, v))
        elif isinstance(v, np.ndarray):
            out[full] = v
        elif isinstance(v, (list, tuple)):
            try:
                out[full] = np.asarray(v)
            except (ValueError, TypeError):
                out[full] = np.asarray(v, dtype=object)
        elif isinstance(v, (int, float, np.integer, np.floating, np.bool_)):
            out[full] = np.asarray(v)
        elif isinstance(v, bool):
            out[full] = np.asarray(v)
        elif isinstance(v, str):
            out[full] = np.asarray(v)
        elif v is None:
            out[full] = np.asarray(np.nan)
        else:
            # Final fallback: store as object array (pickle-backed).
            out[full] = np.asarray(v, dtype=object)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sprint 5a driver: Kok + Richter + Tang at r=1.0, seed=42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--g-total", type=float, default=1.0)
    parser.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npz path (default: sprint_5a_intact_r{r}_seed{seed}.npz)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tiny-n pipeline check (~1 min)")
    parser.add_argument("--skip-kok", action="store_true")
    parser.add_argument("--skip-richter", action="store_true")
    parser.add_argument("--skip-tang", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    seed = int(args.seed)
    r = float(args.r)
    g_total = float(args.g_total)

    if args.dry_run:
        kok_cfg = KokConfig(
            n_stim_trials=20, n_omission_trials=4,
            cue_ms=200.0, gap_ms=200.0, grating_ms=200.0, iti_ms=300.0,
            seed=seed,
        )
        rich_cfg = RichterConfig(
            n_trials=24, reps_per_pair=2,
            leader_ms=200.0, trailer_ms=200.0, iti_ms=300.0,
            seed=seed,
        )
        tang_cfg = TangConfig(
            n_items=80, item_ms=150.0, presettle_ms=200.0, seed=seed,
        )
        tag = "dryrun"
    else:
        kok_cfg = KokConfig(seed=seed)
        rich_cfg = RichterConfig(seed=seed)
        tang_cfg = TangConfig(seed=seed)
        tag = "full"

    out_path = args.out or str(
        DEFAULT_CKPT_DIR / f"sprint_5a_intact_r{r:.1f}_seed{seed}_{tag}.npz"
    )
    out_path = os.fspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    results: Dict[str, Any] = {}
    t_start = time.time()

    # Each assay builds its own fresh bundle (Brian2 group-name uniqueness
    # forbids re-use across separate networks in the same process).
    if not args.skip_kok:
        print(f"[sprint-5a] Kok ({tag}) — seed={seed} r={r} g_total={g_total}")
        t0 = time.time()
        bundle_kok = build_frozen_network(
            h_kind="hr", seed=seed, r=r, g_total=g_total,
            with_cue=True, ckpt_dir=args.ckpt_dir,
        )
        kok_res = run_kok_passive(
            bundle=bundle_kok, cfg=kok_cfg, seed=seed, verbose=args.verbose,
        )
        dt = time.time() - t0
        print(f"  Kok done in {dt/60:.1f} min  "
              f"mean_amp valid/invalid = "
              f"{kok_res.mean_amp['valid']['total_rate_hz']:.3f} / "
              f"{kok_res.mean_amp['invalid']['total_rate_hz']:.3f} Hz  "
              f"SVM={kok_res.svm['accuracy']:.3f}")
        results["kok"] = {
            "mean_amp_valid_hz": kok_res.mean_amp['valid']['total_rate_hz'],
            "mean_amp_valid_ci": np.asarray(kok_res.mean_amp['valid']['total_rate_hz_ci']),
            "mean_amp_invalid_hz": kok_res.mean_amp['invalid']['total_rate_hz'],
            "mean_amp_invalid_ci": np.asarray(kok_res.mean_amp['invalid']['total_rate_hz_ci']),
            "svm_accuracy": kok_res.svm['accuracy'],
            "svm_accuracy_ci": np.asarray(kok_res.svm['accuracy_ci']),
            "pref_rank_bin_delta": kok_res.pref_rank['bin_delta'],
            "pref_rank_bin_expected": kok_res.pref_rank['bin_expected'],
            "pref_rank_bin_unexpected": kok_res.pref_rank['bin_unexpected'],
            "omission_delta": kok_res.omission,
            "raw_grating_counts": kok_res.raw['trial_grating_counts'],
            "raw_cond_mask": kok_res.raw['cond_mask'],
            "raw_theta_per_trial": kok_res.raw['theta_per_trial'],
            "raw_is_omission": kok_res.raw['is_omission'].astype(np.int8),
            "raw_pref_rad": kok_res.raw['pref_rad'],
            "meta": {
                k: v for k, v in kok_res.meta.items()
                if k not in ("config", "bundle") and np.isscalar(v)
            },
            "wall_time_s": float(dt),
        }

    if not args.skip_richter:
        print(f"[sprint-5a] Richter ({tag}) — seed={seed} r={r}")
        t0 = time.time()
        bundle_rich = build_frozen_network(
            h_kind="hr", seed=seed, r=r, g_total=g_total,
            with_cue=False, ckpt_dir=args.ckpt_dir,
        )
        rich_res = run_richter_crossover(
            bundle=bundle_rich, cfg=rich_cfg, seed=seed, verbose=args.verbose,
        )
        dt = time.time() - t0
        print(f"  Richter done in {dt/60:.1f} min  "
              f"center-vs-flank redist = {rich_res.center_vs_flank['redist']:+.3f}  "
              f"bin0 Δ = {rich_res.pref_rank['bin_delta'][0]:+.3f}")
        results["richter"] = {
            "pref_rank_bin_delta": rich_res.pref_rank['bin_delta'],
            "pref_rank_bin_expected": rich_res.pref_rank['bin_expected'],
            "pref_rank_bin_unexpected": rich_res.pref_rank['bin_unexpected'],
            "feature_distance_grid": rich_res.feature_distance['grid'],
            "feature_distance_counts": rich_res.feature_distance['grid_counts'],
            "cell_type_rate_hz": rich_res.cell_type_gain['rate_hz'],
            "cell_type_delta_hz": rich_res.cell_type_gain['delta_hz'],
            "center_delta": rich_res.center_vs_flank['center_delta'],
            "flank_delta": rich_res.center_vs_flank['flank_delta'],
            "redist": rich_res.center_vs_flank['redist'],
            "voxel_families": sorted(rich_res.voxel_forward.keys()),
            "raw_trailer_counts_e": rich_res.raw['trailer_counts_e'],
            "raw_theta_L": rich_res.raw['theta_L'],
            "raw_theta_T": rich_res.raw['theta_T'],
            "raw_cond_mask": rich_res.raw['cond_mask'],
            "raw_pref_rad": rich_res.raw['pref_rad'],
            "meta": {
                k: v for k, v in rich_res.meta.items()
                if k not in ("config", "bundle") and np.isscalar(v)
            },
            "wall_time_s": float(dt),
        }
        # Store per-family baseline + predicted voxel tuning.
        for fam, out in rich_res.voxel_forward.items():
            results["richter"][f"voxel_{fam}_baseline"] = out["voxel_tuning_baseline"]
            results["richter"][f"voxel_{fam}_predicted"] = out["voxel_tuning_predicted"]

    if not args.skip_tang:
        print(f"[sprint-5a] Tang ({tag}) — seed={seed} r={r}")
        t0 = time.time()
        bundle_tang = build_frozen_network(
            h_kind="ht", seed=seed, r=r, g_total=g_total,
            with_cue=False, ckpt_dir=args.ckpt_dir,
        )
        tang_res = run_tang_rotating(
            bundle=bundle_tang, cfg=tang_cfg, seed=seed, verbose=args.verbose,
        )
        dt = time.time() - t0
        print(f"  Tang done in {dt/60:.1f} min  "
              f"mean cell-Δ = {tang_res.cell_gain['mean_delta_hz']:+.3f} Hz  "
              f"SVM={tang_res.svm['accuracy']:.3f}")
        results["tang"] = {
            "cell_delta_hz": tang_res.cell_gain['delta_hz'],
            "cell_rate_deviant_hz": tang_res.cell_gain['rate_deviant_hz'],
            "cell_rate_expected_hz": tang_res.cell_gain['rate_expected_hz'],
            "mean_delta_hz": tang_res.cell_gain['mean_delta_hz'],
            "mean_delta_hz_ci": np.asarray(tang_res.cell_gain['mean_delta_hz_ci']),
            "svm_accuracy": tang_res.svm['accuracy'],
            "svm_accuracy_ci": np.asarray(tang_res.svm['accuracy_ci']),
            "laminar_deviant_hz": tang_res.laminar['deviant_rate_hz'],
            "laminar_expected_hz": tang_res.laminar['expected_rate_hz'],
            "laminar_delta_hz": tang_res.laminar['delta_hz'],
            "tuning_expected_fwhm": tang_res.tuning['expected_fit']['fwhm_rad'],
            "tuning_deviant_fwhm": tang_res.tuning['deviant_fit']['fwhm_rad'],
            "tuning_expected_r2": tang_res.tuning['expected_fit']['r2'],
            "tuning_deviant_r2": tang_res.tuning['deviant_fit']['r2'],
            "raw_counts_per_item": tang_res.raw['counts_per_item'],
            "raw_theta_per_item": tang_res.raw['theta_per_item'],
            "raw_deviant_mask": tang_res.raw['deviant_mask'].astype(np.int8),
            "raw_pref_rad": tang_res.raw['pref_rad'],
            "meta": {
                k: v for k, v in tang_res.meta.items()
                if k not in ("config", "bundle") and np.isscalar(v)
            },
            "wall_time_s": float(dt),
        }

    total_dt = time.time() - t_start
    print(f"\n[sprint-5a] all assays done in {total_dt/60:.1f} min")
    print(f"[sprint-5a] saving to {out_path}")

    flat = _flatten_for_npz("", results)
    flat["_provenance.seed"] = np.asarray(seed)
    flat["_provenance.r"] = np.asarray(r)
    flat["_provenance.g_total"] = np.asarray(g_total)
    flat["_provenance.total_wall_min"] = np.asarray(total_dt / 60.0)
    flat["_provenance.tag"] = np.asarray(tag)
    np.savez_compressed(out_path, **flat)
    print(f"[sprint-5a] saved {len(flat)} arrays.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
