"""Task #73 Kok15 single-anchor fine-discrimination — 15° anchor only.

Lead's request: clean fine-discrim test at cue_mapping[0]=15° only, probes at
15°±10° = {5°, 25°}. Expected = cue_id=0 + probe in {5, 25}. Unexpected =
cue_id=1 (which predicts 75°) + probe in {5, 25} (mismatch). Decoder = SVM
trained on 12-orient localizer, tolerance=15°.

Also compute:
  - n_units preferring 15° (±15° tol) from localizer
  - Δfwhm (expected vs unexpected) on 15°-anchor trials, using
    response-weighted FWHM.
  - Δamp_pref (expected vs unexpected) on pref units at 15°.

Output packed as summary line.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.svm import LinearSVC
from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_kok import (
    _compute_localizer_stats, _score_fine_discrim,
    _response_weighted_fwhm, _bootstrap_ci_mean_diff,
    _permutation_p_two_sided,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed,
)
from scripts.v2.eval_kok import (
    run_kok_probe_trial, run_kok_localizer_trial,
)


def _ang_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "checkpoints/v2/phase3_kok_task73_at15/"
                   "phase3_kok_s42.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchor-deg", type=float, default=15.0)
    p.add_argument("--offset-deg", type=float, default=10.0,
                   help="probe offsets +/- this around the anchor")
    p.add_argument("--n-fine", type=int, default=120,
                   help="trials per (cue, offset) cell — 120 × 2 offsets × "
                        "2 cues = 480 fine trials total")
    p.add_argument("--n-loc-per", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--tol-deg", type=float, default=15.0)
    args = p.parse_args()

    bundle = load_checkpoint(args.ckpt, seed=int(args.seed), device="cpu")
    bundle.net.set_phase("phase3_kok")
    cue_mapping = bundle.meta.get("cue_mapping") or cue_mapping_from_seed(
        int(args.seed))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}
    cue_0_anchor = cue_mapping[0]   # expected: 15°
    cue_1_anchor = cue_mapping[1]   # 75°
    assert abs(cue_0_anchor - args.anchor_deg) < 1e-6, (
        f"expected ckpt cue_mapping[0]=={args.anchor_deg}, got {cue_0_anchor}")
    print(f"[Kok15_single_fine] anchor={args.anchor_deg}, offset=±{args.offset_deg}",
          file=sys.stderr, flush=True)

    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(int(args.seed))

    # 1. Localizer → SVM decoder + per-unit preferred_deg + fwhm
    n_orients = 12
    loc_orients = np.linspace(0.0, 180.0, n_orients, endpoint=False)
    loc_trials_list, loc_orient_list = [], []
    for o in loc_orients:
        for _ in range(int(args.n_loc_per)):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(o),
                timing=timing, noise_std=float(args.noise_std), generator=gen,
            )
            loc_trials_list.append(r.cpu().numpy().astype(np.float64))
            loc_orient_list.append(float(o))
    loc_trials = np.stack(loc_trials_list, axis=0)
    loc_orient = np.asarray(loc_orient_list, dtype=np.float64)
    loc_stats = _compute_localizer_stats(loc_trials, loc_orient, loc_orients)
    preferred_deg = loc_stats["preferred_deg"]
    fwhm_per_unit = loc_stats["fwhm_deg"]
    clf = LinearSVC(dual=False, max_iter=5000)
    clf.fit(loc_trials, loc_orient)
    train_acc = float(clf.score(loc_trials, loc_orient))
    print(f"[Kok15_single_fine] SVM train_acc={train_acc:.3f}",
          file=sys.stderr, flush=True)

    # Units that prefer the anchor (±tol°)
    pref_mask = _ang_diff(preferred_deg, float(args.anchor_deg)) < float(args.tol_deg)
    n_units_pref = int(pref_mask.sum())
    print(f"[Kok15_single_fine] n_units preferring {args.anchor_deg}°: {n_units_pref}",
          file=sys.stderr, flush=True)

    # 2. Fine trials — cue 0 (expected), cue 1 (unexpected), each × 2 offsets
    offsets = (-float(args.offset_deg), +float(args.offset_deg))
    fine_r, fine_true, fine_expected = [], [], []
    for cue_id, expected_flag in ((0, True), (1, False)):
        for offset in offsets:
            probe_deg = float((args.anchor_deg + offset) % 180.0)
            for _ in range(int(args.n_fine)):
                r = run_kok_probe_trial(
                    bundle, cue_id=int(cue_id),
                    probe_orientation_deg=probe_deg,
                    timing=timing, noise_std=float(args.noise_std),
                    generator=gen,
                )
                fine_r.append(r.cpu().numpy().astype(np.float64))
                fine_true.append(probe_deg)
                fine_expected.append(bool(expected_flag))
    fine_r = np.stack(fine_r, axis=0)                           # [N_fine, n_l23]
    fine_true = np.asarray(fine_true, dtype=np.float64)
    fine_expected = np.asarray(fine_expected, dtype=bool)

    # 3. SVM score + bootstrap + permutation on per-trial correctness
    score = _score_fine_discrim(
        clf, fine_r, fine_true, fine_expected,
        tolerance_deg=float(args.tol_deg),
    )
    correct = score["correct_per_trial"].astype(np.float64)
    boot_acc = _bootstrap_ci_mean_diff(
        correct[fine_expected], correct[~fine_expected],
        n_resamples=int(args.n_bootstrap), seed=0,
    )
    p_val_acc = _permutation_p_two_sided(
        correct[fine_expected], correct[~fine_expected],
        n_permutations=int(args.n_permutations), seed=0,
    )

    # 4. Δfwhm on single-anchor fine trials (expected vs unexpected)
    fwhm_block = _response_weighted_fwhm(fine_r, fine_expected, fwhm_per_unit)
    per_trial_fwhm = fwhm_block.pop("per_trial_fwhm_deg")
    boot_fwhm = _bootstrap_ci_mean_diff(
        per_trial_fwhm[fine_expected], per_trial_fwhm[~fine_expected],
        n_resamples=int(args.n_bootstrap), seed=1,
    )
    p_val_fwhm = _permutation_p_two_sided(
        per_trial_fwhm[fine_expected], per_trial_fwhm[~fine_expected],
        n_permutations=int(args.n_permutations), seed=1,
    )

    # 5. Δamp_pref on pref units (mean activity)
    if n_units_pref > 0:
        amp_pref_exp = float(fine_r[fine_expected][:, pref_mask].mean())
        amp_pref_unexp = float(fine_r[~fine_expected][:, pref_mask].mean())
        delta_amp_pref = amp_pref_exp - amp_pref_unexp
    else:
        amp_pref_exp = amp_pref_unexp = delta_amp_pref = float("nan")

    out = {
        "ckpt": str(args.ckpt),
        "anchor_deg": float(args.anchor_deg),
        "offset_deg": float(args.offset_deg),
        "n_fine_per_condition": int(args.n_fine),
        "n_units_pref_anchor": n_units_pref,
        "svm_train_acc": train_acc,
        "svm_acc_expected": score["acc_expected"],
        "svm_acc_unexpected": score["acc_unexpected"],
        "delta_svm_acc": score["delta_acc"],
        "delta_svm_acc_ci_lo": boot_acc["ci_lo"],
        "delta_svm_acc_ci_hi": boot_acc["ci_hi"],
        "p_svm_acc_two_sided": p_val_acc,
        "fwhm_expected_deg": fwhm_block["fwhm_expected_deg"],
        "fwhm_unexpected_deg": fwhm_block["fwhm_unexpected_deg"],
        "delta_fwhm_deg": fwhm_block["delta_fwhm_deg"],
        "delta_fwhm_ci_lo": boot_fwhm["ci_lo"],
        "delta_fwhm_ci_hi": boot_fwhm["ci_hi"],
        "p_fwhm_two_sided": p_val_fwhm,
        "amp_pref_expected": amp_pref_exp,
        "amp_pref_unexpected": amp_pref_unexp,
        "delta_amp_pref": delta_amp_pref,
        "n_expected_trials": int(fine_expected.sum()),
        "n_unexpected_trials": int((~fine_expected).sum()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
