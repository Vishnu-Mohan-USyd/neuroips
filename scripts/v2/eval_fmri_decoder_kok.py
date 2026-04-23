"""Task #74 fMRI-style Δdecode on the β-trained Kok checkpoint.

Two protocols, both on the same β checkpoint, both producing a single
Δ and verdict:

1. ``--protocol delta-per-mode`` (legacy, Task #74 first attempt).
   Three separate 5-fold CV runs — cue_mode ∈ {none, expected,
   unexpected} — each scoring its own decoder. Δ = CV-mean(expected)
   − CV-mean(unexpected). This protocol has a known confound: inside
   each single-mode run the cue is 100% correlated with the probe
   orientation, so the decoder latches onto the cue signature itself
   and ceilings trivially under noise. Kept for backward-comparison.

2. ``--protocol frozen-localizer`` (Task #74 redesign, default).
   Train the decoder ONCE on a cue-free localizer
   (``cue_mode="none"``, q_t=None throughout), pick C via 5-fold CV
   on the localizer, refit a single LinearSVC on the full localizer
   at that C, and LOCK it. Then apply the frozen decoder to two
   out-of-sample cued test sets:

     * ``cued + matched``     — q_t = q_cue whose mapping matches the
       probe orientation (β gate suppresses pref units of the probe
       orientation).
     * ``cued + mismatched``  — q_t = q_cue whose mapping is
       orthogonal to the probe (β gate suppresses pref units of the
       *other* orientation).

   Metric: Δ = acc_matched − acc_mismatched, where accuracy is
   computed trial-wise against the probe-orientation ground truth
   (NOT the cue). Because the decoder was never exposed to cue
   signatures, it can only read "probe orientation from V1
   activity" and cannot cheat through the cue.

Both protocols emit a bootstrap CI on Δ (resampling with replacement
1000×) and a paired-label permutation p-value (≥1000 permutations)
and dump a JSON with all block-level details.

Verdict rule (shared):
    |Δ| < eps  → null
    Δ > eps    → sharpening-sign
    Δ < -eps   → dampening-sign
where ``eps`` defaults to 0.02 (matches dispatch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_fmri_decoder import (
    build_voxel_pool_mask, collect_trials, pool_to_voxels,
    svm_5fold_cv_with_C,
)
from scripts.v2.eval_kok import _install_learned_w_q_gain
from scripts.v2.train_phase3_kok_learning import cue_mapping_from_seed


def _collect_cued_trials(
    bundle, *, cue_mode: str, cue_mapping: dict[int, float],
    n_trials_per_orient: int, noise_std: float,
    contrast_min: float, contrast_max: float,
    n_warmup: int, n_readout: int, seed: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around :func:`collect_trials` with a progress line."""
    print(
        f"[fmri_decoder_kok] {label}: collecting 2 orient × "
        f"{n_trials_per_orient} trials (cue_mode={cue_mode}, "
        f"noise_std={noise_std}, contrast∈[{contrast_min},{contrast_max}], "
        f"{n_warmup}+{n_readout} steps)...",
        flush=True,
    )
    X, y, _ = collect_trials(
        bundle, orientations_deg=[45.0, 135.0],
        n_trials_per_orient=int(n_trials_per_orient),
        noise_std=float(noise_std),
        contrast_min=float(contrast_min),
        contrast_max=float(contrast_max),
        n_warmup=int(n_warmup), n_readout=int(n_readout),
        seed=int(seed),
        cue_mode=cue_mode,
        cue_mapping=cue_mapping if cue_mode != "none" else None,
    )
    print(
        f"[fmri_decoder_kok] {label}: collected X{X.shape} y{y.shape}.",
        flush=True,
    )
    return X, y


def _bootstrap_delta_trial_ci(
    correct_matched: np.ndarray, correct_mismatched: np.ndarray,
    *, n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI on Δ = mean(correct_matched) − mean(correct_mismatched)
    via trial-level resampling with replacement within each condition.

    Returns ``(mean_diff, ci_lo, ci_hi)``.
    """
    a = np.asarray(correct_matched, dtype=np.float64)
    b = np.asarray(correct_mismatched, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n_a = len(a)
    n_b = len(b)
    diffs = np.empty(int(n_resamples), dtype=np.float64)
    for k in range(int(n_resamples)):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        diffs[k] = float(a[idx_a].mean() - b[idx_b].mean())
    mean_diff = float(a.mean() - b.mean())
    ci_lo = float(np.quantile(diffs, alpha / 2.0))
    ci_hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return mean_diff, ci_lo, ci_hi


def _trial_label_permutation_p(
    correct_matched: np.ndarray, correct_mismatched: np.ndarray,
    *, n_permutations: int = 1000, seed: int = 42,
) -> float:
    """Two-sided permutation p-value on Δ via trial-wise matched-vs-mismatched
    label shuffling.

    Pools the per-trial correctness indicators across both conditions,
    randomly reassigns condition membership while preserving condition
    sample counts, and recomputes Δ. Returns the right-inclusive
    two-sided p-value
    ``(1 + #{|Δ_perm| ≥ |Δ_obs|}) / (1 + n_permutations)``.
    """
    a = np.asarray(correct_matched, dtype=np.float64)
    b = np.asarray(correct_mismatched, dtype=np.float64)
    pool = np.concatenate([a, b])
    n_a = len(a)
    n_b = len(b)
    rng = np.random.default_rng(int(seed))
    delta_obs = float(a.mean() - b.mean())
    count_ge = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(n_a + n_b)
        ap = pool[perm[:n_a]]
        bp = pool[perm[n_a:]]
        delta_perm = float(ap.mean() - bp.mean())
        if abs(delta_perm) >= abs(delta_obs) - 1e-12:
            count_ge += 1
    return float((1 + count_ge) / (1 + int(n_permutations)))


def _fit_locked_localizer_decoder(
    X: np.ndarray, y: np.ndarray, *, seed: int,
    C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> tuple[Any, dict[str, Any]]:
    """Train a frozen LinearSVC on the full localizer.

    Returns ``(clf_locked, loc_cv)``:

      * ``loc_cv`` — output of :func:`svm_5fold_cv_with_C` on the
        localizer, used for reporting the localizer CV accuracy and
        the per-fold C choices.
      * ``clf_locked`` — ``LinearSVC`` fit on ALL localizer trials at
        the **modal** best-C across the 5 outer folds (the "locked"
        decoder to be applied to the cued test sets).

    Picking the modal C across folds is the simplest honest choice
    when CV reports five possibly-different best-C values: it
    respects the grid-search outcome without averaging across
    incomparable scales.
    """
    from collections import Counter
    from sklearn.svm import LinearSVC
    loc_cv = svm_5fold_cv_with_C(X, y, C_grid=C_grid, seed=int(seed))
    if "error" in loc_cv:
        raise RuntimeError(f"localizer CV failed: {loc_cv}")
    per_fold_C = loc_cv["per_fold_C"]
    modal_C = float(Counter(per_fold_C).most_common(1)[0][0])
    clf = LinearSVC(
        C=modal_C, random_state=int(seed), max_iter=10000, dual="auto",
    )
    clf.fit(X, y)
    return clf, loc_cv


def run_frozen_localizer_protocol(
    bundle, *, cue_mapping: dict[int, float], mask: np.ndarray,
    n_trials_localizer_per_orient: int,
    n_trials_test_per_orient: int,
    noise_std: float, contrast_min: float, contrast_max: float,
    n_warmup: int, n_readout: int,
    trial_seed: int, cv_seed: int,
    n_bootstrap: int = 1000, n_permutations: int = 1000,
    verdict_eps: float = 0.02,
) -> dict[str, Any]:
    """Run the Task #74 frozen-localizer protocol end-to-end.

    Three trial collections (localizer no-cue, cued matched, cued
    mismatched) are scored by a single locked decoder trained on the
    localizer. Returns a result dict with per-block accuracies, Δ,
    bootstrap CI, permutation p, and a final ``verdict`` ∈
    {``sharpening``, ``dampening``, ``null``}.
    """
    # --- Step 1: localizer data + locked decoder -------------------------
    X_loc, y_loc = _collect_cued_trials(
        bundle, cue_mode="none", cue_mapping=cue_mapping,
        n_trials_per_orient=n_trials_localizer_per_orient,
        noise_std=noise_std, contrast_min=contrast_min,
        contrast_max=contrast_max, n_warmup=n_warmup, n_readout=n_readout,
        seed=trial_seed, label="localizer (cue_mode=none)",
    )
    X_loc_vox = pool_to_voxels(X_loc, mask)
    clf, loc_cv = _fit_locked_localizer_decoder(
        X_loc_vox, y_loc, seed=cv_seed,
    )
    modal_C = float(clf.C)
    print(
        f"[fmri_decoder_kok] localizer CV acc={loc_cv['acc_mean']:.4f} "
        f"± {loc_cv['acc_std']:.4f}; locked_C={modal_C}.",
        flush=True,
    )

    # --- Step 2: cued test sets + frozen-decoder predictions -------------
    X_m, y_m = _collect_cued_trials(
        bundle, cue_mode="expected", cue_mapping=cue_mapping,
        n_trials_per_orient=n_trials_test_per_orient,
        noise_std=noise_std, contrast_min=contrast_min,
        contrast_max=contrast_max, n_warmup=n_warmup, n_readout=n_readout,
        seed=trial_seed + 1, label="cued_matched (cue_mode=expected)",
    )
    X_u, y_u = _collect_cued_trials(
        bundle, cue_mode="unexpected", cue_mapping=cue_mapping,
        n_trials_per_orient=n_trials_test_per_orient,
        noise_std=noise_std, contrast_min=contrast_min,
        contrast_max=contrast_max, n_warmup=n_warmup, n_readout=n_readout,
        seed=trial_seed + 2, label="cued_mismatched (cue_mode=unexpected)",
    )
    X_m_vox = pool_to_voxels(X_m, mask)
    X_u_vox = pool_to_voxels(X_u, mask)
    pred_m = clf.predict(X_m_vox)
    pred_u = clf.predict(X_u_vox)
    correct_m = (pred_m == y_m).astype(np.float64)
    correct_u = (pred_u == y_u).astype(np.float64)
    acc_m = float(correct_m.mean())
    acc_u = float(correct_u.mean())

    # --- Step 3: Δ, bootstrap CI, permutation p --------------------------
    delta_mean, ci_lo, ci_hi = _bootstrap_delta_trial_ci(
        correct_m, correct_u,
        n_resamples=int(n_bootstrap), seed=int(trial_seed),
    )
    p_two_sided = _trial_label_permutation_p(
        correct_m, correct_u,
        n_permutations=int(n_permutations), seed=int(trial_seed) + 17,
    )
    if abs(delta_mean) < float(verdict_eps):
        verdict = "null"
    elif delta_mean > 0.0:
        verdict = "sharpening"
    else:
        verdict = "dampening"

    return {
        "protocol": "frozen-localizer",
        "localizer": {
            "cv_acc_mean": float(loc_cv["acc_mean"]),
            "cv_acc_std": float(loc_cv["acc_std"]),
            "cv_per_fold_acc": [float(x) for x in loc_cv["per_fold_acc"]],
            "cv_per_fold_C": [float(x) for x in loc_cv["per_fold_C"]],
            "locked_C": modal_C,
            "n_trials_total": int(X_loc.shape[0]),
        },
        "cued_matched": {
            "acc": acc_m,
            "n_trials": int(len(correct_m)),
        },
        "cued_mismatched": {
            "acc": acc_u,
            "n_trials": int(len(correct_u)),
        },
        "delta_matched_minus_mismatched": {
            "mean": float(delta_mean),
            "ci": [float(ci_lo), float(ci_hi)],
            "n_bootstrap": int(n_bootstrap),
            "permutation_p_two_sided": float(p_two_sided),
            "n_permutations": int(n_permutations),
            "eps": float(verdict_eps),
        },
        "verdict": verdict,
    }


def _bootstrap_delta_ci(
    per_fold_exp: list[float], per_fold_unexp: list[float],
    *, n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI on Δ = mean(exp) − mean(unexp) via fold resampling.

    The 5 paired folds are resampled with replacement; for each
    resample the Δ of fold means is computed. The returned CI is the
    empirical ``[alpha/2, 1-alpha/2]`` percentile interval of the
    resample distribution.

    Returns ``(mean_diff, ci_lo, ci_hi)``.
    """
    exp = np.asarray(per_fold_exp, dtype=np.float64)
    unexp = np.asarray(per_fold_unexp, dtype=np.float64)
    if exp.shape != unexp.shape:
        raise ValueError("per-fold arrays must have the same shape")
    rng = np.random.default_rng(int(seed))
    n = len(exp)
    diffs = np.empty(int(n_resamples), dtype=np.float64)
    for b in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        diffs[b] = float(exp[idx].mean() - unexp[idx].mean())
    mean_diff = float(exp.mean() - unexp.mean())
    ci_lo = float(np.quantile(diffs, alpha / 2.0))
    ci_hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return mean_diff, ci_lo, ci_hi


def _paired_permutation_p(
    per_fold_exp: list[float], per_fold_unexp: list[float],
    *, n_permutations: int = 1000, seed: int = 42,
) -> float:
    """Two-sided paired-swap permutation p-value on Δ.

    Under H₀ (expected ≡ unexpected), per-fold expected/unexpected
    labels are interchangeable. For each permutation, each fold's
    pair is independently swapped with probability 0.5. Returns the
    standard right-inclusive two-sided p-value
    ``p = (1 + #{|Δ_perm| ≥ |Δ_obs|}) / (1 + n_permutations)``.
    """
    exp = np.asarray(per_fold_exp, dtype=np.float64)
    unexp = np.asarray(per_fold_unexp, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n = len(exp)
    delta_obs = float(exp.mean() - unexp.mean())
    count_ge = 0
    for _ in range(int(n_permutations)):
        swap = rng.integers(0, 2, size=n).astype(bool)
        a = np.where(swap, unexp, exp)
        b = np.where(swap, exp, unexp)
        delta_perm = float(a.mean() - b.mean())
        if abs(delta_perm) >= abs(delta_obs) - 1e-12:
            count_ge += 1
    return float((1 + count_ge) / (1 + int(n_permutations)))


def _run_condition(
    bundle, *, cue_mode: str, cue_mapping: dict[int, float],
    mask: np.ndarray, n_trials_per_orient: int,
    noise_std: float, contrast_min: float, contrast_max: float,
    n_warmup: int, n_readout: int, trial_seed: int, cv_seed: int,
) -> dict[str, Any]:
    """Collect trials under one cue_mode, score 5-fold CV, return block."""
    print(
        f"[fmri_decoder_kok] cue_mode={cue_mode}: collecting "
        f"2 orient × {n_trials_per_orient} trials (noise_std={noise_std}, "
        f"contrast∈[{contrast_min},{contrast_max}], "
        f"{n_warmup}+{n_readout} steps)...",
        flush=True,
    )
    orientations_deg = [45.0, 135.0]
    X, y, contrasts = collect_trials(
        bundle, orientations_deg=orientations_deg,
        n_trials_per_orient=int(n_trials_per_orient),
        noise_std=float(noise_std),
        contrast_min=float(contrast_min),
        contrast_max=float(contrast_max),
        n_warmup=int(n_warmup), n_readout=int(n_readout),
        seed=int(trial_seed),
        cue_mode=cue_mode, cue_mapping=cue_mapping,
    )
    print(
        f"[fmri_decoder_kok] cue_mode={cue_mode}: X{X.shape} y{y.shape} "
        "→ 5-fold CV (voxel-pooled).",
        flush=True,
    )
    X_vox = pool_to_voxels(X, mask)
    svm = svm_5fold_cv_with_C(X_vox, y, seed=int(cv_seed))
    return {
        "cue_mode": cue_mode,
        "orientations_deg": orientations_deg,
        "n_trials_total": int(X.shape[0]),
        "svm_voxel_pooled": svm,
        "contrast_mean": float(contrasts.mean()),
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="β-trained Phase-3 Kok checkpoint (must carry top-level W_q_gain).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-voxels", type=int, default=64)
    p.add_argument(
        "--protocol", choices=("delta-per-mode", "frozen-localizer"),
        default="frozen-localizer",
        help=(
            "delta-per-mode: three separate 5-fold CV runs, Δ of CV "
            "means (Task #74 original, has cue-probe confound). "
            "frozen-localizer (default): train decoder on cue-free "
            "localizer, lock it, apply to cued matched/mismatched "
            "test sets (Task #74 redesign — the correct Kok-style "
            "Δdecode protocol)."
        ),
    )
    p.add_argument(
        "--n-trials-per-orient", type=int, default=400,
        help=(
            "delta-per-mode: trials per orientation per cue_mode "
            "(400 → 800/mode). frozen-localizer: trials per "
            "orientation for each cued test set "
            "(matched + mismatched, 400 → 800/condition)."
        ),
    )
    p.add_argument(
        "--n-trials-localizer-per-orient", type=int, default=800,
        help=(
            "frozen-localizer only: trials per orientation for the "
            "cue-free localizer training set. Default 800 → 1600 "
            "total. Localizer needs enough trials to yield a stable "
            "locked decoder."
        ),
    )
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--contrast-min", type=float, default=0.1)
    p.add_argument("--contrast-max", type=float, default=0.3)
    p.add_argument("--n-warmup-steps", type=int, default=50)
    p.add_argument("--n-readout-steps", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument(
        "--verdict-eps", type=float, default=0.02,
        help="|Δ| < eps → null. Matches dispatch: eps=0.02.",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("logs/task74/eval_fmri_decoder_kok.json"),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)

    bundle = load_checkpoint(
        args.checkpoint, seed=int(args.seed), device=args.device,
    )
    # The ckpt is phase-3 Kok; task weights (W_qm_task, W_mh_task_exc,
    # W_mh_task_inh) must be live so cue_mode=expected/unexpected
    # exercises them end-to-end.
    bundle.net.set_phase("phase3_kok")
    _install_learned_w_q_gain(
        bundle, args.checkpoint, device=args.device,
    )
    bundle.net.eval()

    # cue_mapping comes from the ckpt meta if present, else from seed.
    if "cue_mapping" in bundle.meta:
        cue_mapping = {
            int(k): float(v) for k, v in bundle.meta["cue_mapping"].items()
        }
    else:
        cue_mapping = cue_mapping_from_seed(int(args.seed))
    print(f"[fmri_decoder_kok] cue_mapping={cue_mapping}", flush=True)

    mask = build_voxel_pool_mask(int(args.n_voxels))

    if args.protocol == "frozen-localizer":
        result = run_frozen_localizer_protocol(
            bundle, cue_mapping=cue_mapping, mask=mask,
            n_trials_localizer_per_orient=int(
                args.n_trials_localizer_per_orient
            ),
            n_trials_test_per_orient=int(args.n_trials_per_orient),
            noise_std=float(args.noise_std),
            contrast_min=float(args.contrast_min),
            contrast_max=float(args.contrast_max),
            n_warmup=int(args.n_warmup_steps),
            n_readout=int(args.n_readout_steps),
            trial_seed=int(args.seed),
            cv_seed=int(args.seed),
            n_bootstrap=int(args.n_bootstrap),
            n_permutations=int(args.n_permutations),
            verdict_eps=float(args.verdict_eps),
        )

        loc = result["localizer"]
        d = result["delta_matched_minus_mismatched"]
        header = (
            f"fmri_decoder (frozen-localizer readout, β-ckpt, "
            f"noise={args.noise_std}, readout={args.n_readout_steps}, "
            f"45°/135°)"
        )
        print(header)
        print(
            f"localizer_acc (q_t=None train/test split, 5-fold CV)   "
            f"= {loc['cv_acc_mean']:.4f} ± {loc['cv_acc_std']:.4f}"
        )
        print(
            f"acc_matched_cue     (cue=probe, probe=probe_truth)     "
            f"= {result['cued_matched']['acc']:.4f}"
        )
        print(
            f"acc_mismatched_cue  (cue⊥probe, probe=probe_truth)     "
            f"= {result['cued_mismatched']['acc']:.4f}"
        )
        print(
            f"Δ = acc_matched − acc_mismatched = {d['mean']:+.4f}   "
            f"ci=[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]   "
            f"p={d['permutation_p_two_sided']:.4f}"
        )
        print(
            f"verdict: {result['verdict']}  (eps={d['eps']:g})"
        )

        out = {
            "assay": "eval_fmri_decoder_kok",
            "protocol": "frozen-localizer",
            "checkpoint": str(args.checkpoint),
            "seed": int(args.seed),
            "cue_mapping": {str(k): float(v) for k, v in cue_mapping.items()},
            "n_voxels": int(args.n_voxels),
            "noise_std": float(args.noise_std),
            "contrast_range": [
                float(args.contrast_min), float(args.contrast_max),
            ],
            "n_warmup_steps": int(args.n_warmup_steps),
            "n_readout_steps": int(args.n_readout_steps),
            "result": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        print(f"[fmri_decoder_kok] wrote {args.output}", flush=True)
        return 0

    # --- Legacy delta-per-mode protocol (kept for backward-comparison) ---
    # Run the three cue-mode conditions with distinct trial RNG streams
    # but a shared CV seed so fold splits are aligned across conditions
    # (required for paired bootstrap / permutation on Δdecode).
    blocks: dict[str, dict[str, Any]] = {}
    for offset, mode in enumerate(("none", "expected", "unexpected")):
        blocks[mode] = _run_condition(
            bundle, cue_mode=mode, cue_mapping=cue_mapping,
            mask=mask,
            n_trials_per_orient=int(args.n_trials_per_orient),
            noise_std=float(args.noise_std),
            contrast_min=float(args.contrast_min),
            contrast_max=float(args.contrast_max),
            n_warmup=int(args.n_warmup_steps),
            n_readout=int(args.n_readout_steps),
            trial_seed=int(args.seed) + offset,
            cv_seed=int(args.seed),
        )

    exp_folds = blocks["expected"]["svm_voxel_pooled"]["per_fold_acc"]
    unexp_folds = blocks["unexpected"]["svm_voxel_pooled"]["per_fold_acc"]
    delta_mean, ci_lo, ci_hi = _bootstrap_delta_ci(
        exp_folds, unexp_folds,
        n_resamples=int(args.n_bootstrap), seed=int(args.seed),
    )
    p_two_sided = _paired_permutation_p(
        exp_folds, unexp_folds,
        n_permutations=int(args.n_permutations), seed=int(args.seed),
    )

    eps = float(args.verdict_eps)
    if abs(delta_mean) < eps:
        verdict = "null"
    elif delta_mean > 0:
        verdict = "sharpening"
    else:
        verdict = "dampening"

    # --- Stdout table -----------------------------------------------------
    header = (
        f"fmri_decoder Δdecode on β-trained Kok ckpt "
        f"(noise={args.noise_std}, readout={args.n_readout_steps}, "
        f"45°/135° binary)"
    )
    print(header)
    print("cue_mode        acc_mean   acc_std   n_trials")
    for mode in ("none", "expected", "unexpected"):
        s = blocks[mode]["svm_voxel_pooled"]
        n = blocks[mode]["n_trials_total"]
        print(
            f"{mode:<14}  {s['acc_mean']:.4f}     {s['acc_std']:.4f}    {n}"
        )
    print(
        f"Δdecode_exp_vs_unexp = {delta_mean:+.4f} "
        f"ci=[{ci_lo:+.4f},{ci_hi:+.4f}] p={p_two_sided:.4f}"
    )
    print(f"verdict: {verdict}  (eps={eps:g})")

    # --- Dump JSON --------------------------------------------------------
    out = {
        "assay": "eval_fmri_decoder_kok",
        "protocol": "delta-per-mode",
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "cue_mapping": {str(k): float(v) for k, v in cue_mapping.items()},
        "n_voxels": int(args.n_voxels),
        "noise_std": float(args.noise_std),
        "contrast_range": [float(args.contrast_min), float(args.contrast_max)],
        "n_warmup_steps": int(args.n_warmup_steps),
        "n_readout_steps": int(args.n_readout_steps),
        "blocks": blocks,
        "delta_decode": {
            "mean": float(delta_mean),
            "ci": [float(ci_lo), float(ci_hi)],
            "n_bootstrap": int(args.n_bootstrap),
            "permutation_p_two_sided": float(p_two_sided),
            "n_permutations": int(args.n_permutations),
            "eps": float(eps),
        },
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[fmri_decoder_kok] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
