"""Sprint 5c driver: r=1.0 dual-mode rerun (continuous + context_only).

Task #39, Sprint 5c step 4. Re-runs the three updated assays
(R1 Richter / R2 Kok / R3 Tang) at r=1.0 seed=42 in TWO V1→H modes:

  * continuous   — V1→H feedforward always on (Sprint 5b mode).
  * context_only — V1→H gated to cue-only intervals (new toggle).

Tang is skipped in ``context_only`` mode (no natural context window —
items are 250 ms back-to-back; assay raises ValueError on the bundle).

Outputs (per mode, defaults at full n):
    data/checkpoints/sprint_5c_intact_r1.0_seed42_continuous.npz
    data/checkpoints/sprint_5c_intact_r1.0_seed42_context_only.npz

Both files are then summarised in ``docs/SPRINT_5C_DUAL_MODE_FINDINGS.md``
(written manually by the analyst — this script only emits the .npz).

Wall-clock estimate at full defaults (per mode):
    Kok      ~ 16-20 min   (240 stim + 48 omission, +MVPA subsampling)
    Richter  ~ 25-30 min   (372 deranged-permutation trials)
    Tang     ~ 8-10 min    (1000 items: 500 random + 500 rotating)
    Total   ~ 50-60 min in continuous, ~ 40-45 min in context_only (no Tang).

Usage
-----
    python -m expectation_snn.scripts.run_sprint_5c                    # full
    python -m expectation_snn.scripts.run_sprint_5c --dry-run          # ~3 min
    python -m expectation_snn.scripts.run_sprint_5c --modes continuous # only one
    python -m expectation_snn.scripts.run_sprint_5c --skip-richter
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
from expectation_snn.scripts.run_sprint_5a import _flatten_for_npz


DEFAULT_CKPT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "checkpoints"
)


# ---------------------------------------------------------------------------
# Result extractors — one per assay. Pull only the fields needed for
# downstream analysis; raw spike counts kept for re-analysis but otherwise
# trimmed to keep .npz size bounded.
# ---------------------------------------------------------------------------

def _extract_kok(res, dt: float) -> Dict[str, Any]:
    om = res.orientation_mvpa
    return {
        "mean_amp_valid_hz": res.mean_amp['valid']['total_rate_hz'],
        "mean_amp_valid_ci": np.asarray(res.mean_amp['valid']['total_rate_hz_ci']),
        "mean_amp_invalid_hz": res.mean_amp['invalid']['total_rate_hz'],
        "mean_amp_invalid_ci": np.asarray(res.mean_amp['invalid']['total_rate_hz_ci']),
        "svm_accuracy": res.svm['accuracy'],
        "svm_accuracy_ci": np.asarray(res.svm['accuracy_ci']),
        "pref_rank_bin_delta": res.pref_rank['bin_delta'],
        "pref_rank_bin_expected": res.pref_rank['bin_expected'],
        "pref_rank_bin_unexpected": res.pref_rank['bin_unexpected'],
        "omission_delta": res.omission,
        # R2 orientation MVPA
        "mvpa_delta_decoding": om['delta_decoding'],
        "mvpa_delta_decoding_ci": np.asarray(om['delta_decoding_ci']),
        "mvpa_acc_valid_mean": om['acc_valid_mean'],
        "mvpa_acc_invalid_mean": om['acc_invalid_mean'],
        "mvpa_delta_subs": om['delta_subs'],
        "mvpa_acc_valid_subs": om['acc_valid_subs'],
        "mvpa_acc_invalid_subs": om['acc_invalid_subs'],
        "mvpa_n_subsamples": int(om['n_subsamples']),
        "mvpa_n_per_class_subsample": int(om['n_per_class_subsample']),
        # raw
        "raw_grating_counts": res.raw['trial_grating_counts'],
        "raw_cond_mask": res.raw['cond_mask'],
        "raw_theta_per_trial": res.raw['theta_per_trial'],
        "raw_is_omission": res.raw['is_omission'].astype(np.int8),
        "raw_orient_labels": res.raw['orient_labels'],
        "raw_pref_rad": res.raw['pref_rad'],
        "meta": {k: v for k, v in res.meta.items()
                  if k not in ("config", "bundle") and np.isscalar(v)},
        "wall_time_s": float(dt),
    }


def _extract_richter(res, dt: float) -> Dict[str, Any]:
    out = {
        "pref_rank_bin_delta": res.pref_rank['bin_delta'],
        "pref_rank_bin_expected": res.pref_rank['bin_expected'],
        "pref_rank_bin_unexpected": res.pref_rank['bin_unexpected'],
        "feature_distance_grid": res.feature_distance['grid'],
        "feature_distance_counts": res.feature_distance['grid_counts'],
        "cell_type_rate_hz": res.cell_type_gain['rate_hz'],
        "cell_type_delta_hz": res.cell_type_gain['delta_hz'],
        "center_delta": res.center_vs_flank['center_delta'],
        "flank_delta": res.center_vs_flank['flank_delta'],
        "redist": res.center_vs_flank['redist'],
        "voxel_families": sorted(res.voxel_forward.keys()),
        "raw_trailer_counts_e": res.raw['trailer_counts_e'],
        "raw_theta_L": res.raw['theta_L'],
        "raw_theta_T": res.raw['theta_T'],
        "raw_cond_mask": res.raw['cond_mask'],
        "raw_dtheta_step": res.raw['dtheta_step'],
        "raw_pref_rad": res.raw['pref_rad'],
        "meta": {k: v for k, v in res.meta.items()
                  if k not in ("config", "bundle") and np.isscalar(v)},
        "wall_time_s": float(dt),
    }
    # R1 Δθ-stratified report
    ds = res.dtheta_stratified
    out["dtheta_n_trials_per_step"] = np.asarray(
        [ds["n_trials_per_step"].get(k, 0) for k in (1, 2, 3, 4, 5)],
    )
    out["dtheta_step_keys"] = np.asarray((1, 2, 3, 4, 5))
    for k in (1, 2, 3, 4, 5):
        bd = ds["bin_delta_by_step"].get(k)
        if bd is not None:
            out[f"dtheta_step{k}_bin_delta"] = bd
        rd = ds["redist_by_step"].get(k)
        if rd is not None:
            out[f"dtheta_step{k}_center_delta"] = rd["center_delta"]
            out[f"dtheta_step{k}_flank_delta"] = rd["flank_delta"]
            out[f"dtheta_step{k}_redist"] = rd["redist"]
    # voxel-forward per family
    for fam, vf in res.voxel_forward.items():
        out[f"voxel_{fam}_baseline"] = vf["voxel_tuning_baseline"]
        out[f"voxel_{fam}_predicted"] = vf["voxel_tuning_predicted"]
    return out


def _extract_tang(res, dt: float) -> Dict[str, Any]:
    out = {
        "cell_delta_hz": res.cell_gain['delta_hz'],
        "cell_rate_deviant_hz": res.cell_gain['rate_deviant_hz'],
        "cell_rate_expected_hz": res.cell_gain['rate_expected_hz'],
        "mean_delta_hz": res.cell_gain['mean_delta_hz'],
        "mean_delta_hz_ci": np.asarray(res.cell_gain['mean_delta_hz_ci']),
        "svm_accuracy": res.svm['accuracy'],
        "svm_accuracy_ci": np.asarray(
            res.svm.get('accuracy_ci', (np.nan, np.nan))),
        "laminar_deviant_hz": res.laminar['deviant_rate_hz'],
        "laminar_expected_hz": res.laminar['expected_rate_hz'],
        "laminar_delta_hz": res.laminar['delta_hz'],
        "tuning_expected_fwhm": res.tuning['expected_fit']['fwhm_rad'],
        "tuning_deviant_fwhm": res.tuning['deviant_fit']['fwhm_rad'],
        "raw_counts_per_item": res.raw['counts_per_item'],
        "raw_theta_per_item": res.raw['theta_per_item'],
        "raw_deviant_mask": res.raw['deviant_mask'].astype(np.int8),
        "raw_is_random": res.raw['is_random'].astype(np.int8),
        "raw_dtheta_prev_step": res.raw['dtheta_prev_step'],
        "raw_cond_codes": res.raw['cond_codes'],
        "raw_pref_rad": res.raw['pref_rad'],
        "meta": {k: v for k, v in res.meta.items()
                  if k not in ("config", "bundle") and np.isscalar(v)},
        "wall_time_s": float(dt),
    }
    # R3 three-condition matched-θ rate
    tc = res.three_condition
    for name in ("random", "rotating_expected", "rotating_deviant"):
        d = tc["per_cond"][name]
        out[f"3cond_{name}_rate_hz"] = d["mean_rate_hz"]
        out[f"3cond_{name}_rate_ci"] = np.asarray(d["ci"])
    for k, v in tc["deltas"].items():
        out[f"3cond_delta_{k}_hz"] = v["mean_delta_hz"]
        out[f"3cond_delta_{k}_ci"] = np.asarray(v["ci"])
    out["3cond_per_cell_rates"] = tc["per_cell_rates"]
    out["3cond_n_cells_with_data"] = int(tc["n_cells_with_data_all_conds"])
    # R3 Δθ_prev stratification
    dp = res.dtheta_prev
    out["dtheta_prev_n_trials_grid"] = dp["n_trials_grid"]
    rate_grid = np.full((3, 4), np.nan, dtype=np.float64)
    n_cells_grid = np.zeros((3, 4), dtype=np.int64)
    for ci, name in enumerate(("random", "rotating_expected",
                                 "rotating_deviant")):
        for step in (0, 1, 2, 3):
            ent = dp["by_cond_by_step"][name][step]
            rate_grid[ci, step] = ent["mean_rate_hz"]
            n_cells_grid[ci, step] = ent["n_cells"]
    out["dtheta_prev_rate_hz_grid"] = rate_grid
    out["dtheta_prev_n_cells_grid"] = n_cells_grid
    return out


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------

def _run_one_mode(
    mode: str, *, seed: int, r: float, g_total: float,
    ckpt_dir: str,
    kok_cfg: KokConfig, rich_cfg: RichterConfig, tang_cfg: TangConfig,
    skip_kok: bool, skip_richter: bool, skip_tang: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Run all three assays in one V1→H mode and return the result dict."""
    results: Dict[str, Any] = {}
    tag_print = f"[5c/{mode}]"

    if not skip_kok:
        print(f"{tag_print} Kok — seed={seed} r={r}")
        t0 = time.time()
        bundle = build_frozen_network(
            h_kind="hr", seed=seed, r=r, g_total=g_total,
            with_cue=True, with_v1_to_h=mode, ckpt_dir=ckpt_dir,
        )
        res = run_kok_passive(
            bundle=bundle, cfg=kok_cfg, seed=seed, verbose=verbose,
        )
        dt = time.time() - t0
        om = res.orientation_mvpa
        print(f"  Kok done {dt/60:.1f} min  "
              f"valid/inv = {res.mean_amp['valid']['total_rate_hz']:.3f} / "
              f"{res.mean_amp['invalid']['total_rate_hz']:.3f}  "
              f"SVM={res.svm['accuracy']:.3f}  "
              f"MVPA Δ={om['delta_decoding']:+.3f}")
        results["kok"] = _extract_kok(res, dt)

    if not skip_richter:
        print(f"{tag_print} Richter — seed={seed} r={r}")
        t0 = time.time()
        bundle = build_frozen_network(
            h_kind="hr", seed=seed, r=r, g_total=g_total,
            with_cue=False, with_v1_to_h=mode, ckpt_dir=ckpt_dir,
        )
        res = run_richter_crossover(
            bundle=bundle, cfg=rich_cfg, seed=seed, verbose=verbose,
        )
        dt = time.time() - t0
        print(f"  Richter done {dt/60:.1f} min  "
              f"redist={res.center_vs_flank['redist']:+.3f}  "
              f"bin0Δ={res.pref_rank['bin_delta'][0]:+.3f}")
        results["richter"] = _extract_richter(res, dt)

    if not skip_tang:
        if mode == "context_only":
            print(f"{tag_print} Tang — SKIPPED (context_only undefined for "
                  f"back-to-back items)")
        else:
            print(f"{tag_print} Tang — seed={seed} r={r}")
            t0 = time.time()
            bundle = build_frozen_network(
                h_kind="ht", seed=seed, r=r, g_total=g_total,
                with_cue=False, with_v1_to_h=mode, ckpt_dir=ckpt_dir,
            )
            res = run_tang_rotating(
                bundle=bundle, cfg=tang_cfg, seed=seed, verbose=verbose,
            )
            dt = time.time() - t0
            tc = res.three_condition["per_cond"]
            print(f"  Tang done {dt/60:.1f} min  "
                  f"3cond rand/exp/dev = "
                  f"{tc['random']['mean_rate_hz']:.3f} / "
                  f"{tc['rotating_expected']['mean_rate_hz']:.3f} / "
                  f"{tc['rotating_deviant']['mean_rate_hz']:.3f}")
            results["tang"] = _extract_tang(res, dt)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sprint 5c driver: dual-mode (continuous + context_only) "
                     "rerun of R1/R2/R3 assays at r=1.0 seed=42",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--g-total", type=float, default=1.0)
    ap.add_argument("--ckpt-dir", type=str, default=None,
                     help="Default: expectation_snn/data/checkpoints")
    ap.add_argument("--out-dir", type=str, default=None,
                     help="Default: same as ckpt-dir")
    ap.add_argument("--modes", type=str, default="continuous,context_only",
                     help="Comma-separated subset of {continuous,context_only}")
    ap.add_argument("--dry-run", action="store_true",
                     help="Tiny-n pipeline check (~3 min total)")
    ap.add_argument("--skip-kok", action="store_true")
    ap.add_argument("--skip-richter", action="store_true")
    ap.add_argument("--skip-tang", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    seed = int(args.seed)
    r = float(args.r)
    g_total = float(args.g_total)
    ckpt_dir = args.ckpt_dir or str(DEFAULT_CKPT_DIR)
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    verbose = not args.quiet

    modes: List[str] = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("continuous", "context_only"):
            print(f"ERROR: unknown mode {m!r}")
            return 2

    if args.dry_run:
        kok_cfg = KokConfig(
            n_stim_trials=40, n_omission_trials=4,
            cue_ms=200.0, gap_ms=200.0, grating_ms=200.0, iti_ms=300.0,
            mvpa_n_subsamples=3, mvpa_n_bootstrap=100, mvpa_cv=2,
            seed=seed,
        )
        rich_cfg = RichterConfig(
            reps_expected=2, reps_unexpected=1,
            leader_ms=200.0, trailer_ms=200.0, iti_ms=300.0,
            seed=seed,
        )
        tang_cfg = TangConfig(
            n_random=40, n_rotating=40, item_ms=150.0, presettle_ms=200.0,
            seed=seed,
        )
        tag = "dryrun"
    else:
        kok_cfg = KokConfig(seed=seed)
        rich_cfg = RichterConfig(seed=seed)
        tang_cfg = TangConfig(seed=seed)
        tag = "full"

    t_start_all = time.time()
    summary: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        print(f"\n========== mode = {mode} ==========")
        t0 = time.time()
        results = _run_one_mode(
            mode, seed=seed, r=r, g_total=g_total, ckpt_dir=ckpt_dir,
            kok_cfg=kok_cfg, rich_cfg=rich_cfg, tang_cfg=tang_cfg,
            skip_kok=args.skip_kok, skip_richter=args.skip_richter,
            skip_tang=args.skip_tang, verbose=verbose,
        )
        dt_mode = time.time() - t0
        out_path = os.path.join(
            out_dir,
            f"sprint_5c_intact_r{r:.1f}_seed{seed}_{mode}_{tag}.npz",
        )
        flat = _flatten_for_npz("", results)
        flat["_provenance.seed"] = np.asarray(seed)
        flat["_provenance.r"] = np.asarray(r)
        flat["_provenance.g_total"] = np.asarray(g_total)
        flat["_provenance.mode"] = np.asarray(mode)
        flat["_provenance.tag"] = np.asarray(tag)
        flat["_provenance.mode_wall_min"] = np.asarray(dt_mode / 60.0)
        np.savez_compressed(out_path, **flat)
        print(f"[5c/{mode}] saved {len(flat)} arrays → {out_path}")

        # Quick summary line (used by analyst to compare modes side-by-side)
        s: Dict[str, float] = {}
        if "kok" in results:
            s["kok_mvpa_delta"] = float(results["kok"]["mvpa_delta_decoding"])
            s["kok_svm_acc"] = float(results["kok"]["svm_accuracy"])
        if "richter" in results:
            s["richter_redist"] = float(results["richter"]["redist"])
            s["richter_bin0_delta"] = float(
                results["richter"]["pref_rank_bin_delta"][0])
        if "tang" in results:
            s["tang_mean_cell_delta"] = float(
                results["tang"]["mean_delta_hz"])
            s["tang_3cond_dev_minus_exp"] = float(
                results["tang"]["3cond_delta_deviant_minus_expected_hz"])
        summary[mode] = s

    total_dt = (time.time() - t_start_all) / 60.0

    print(f"\n========== Sprint 5c summary ==========")
    print(f"total wall: {total_dt:.1f} min   modes: {modes}")
    for mode, s in summary.items():
        print(f"  [{mode}]")
        for k, v in s.items():
            print(f"    {k:30s} = {v:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
