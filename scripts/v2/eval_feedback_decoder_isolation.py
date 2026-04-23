"""Task #74 — feedback-vs-decoder isolation (Q1 + Q2 + Δdecode).

The previous ``eval_fmri_decoder_kok`` ``frozen-localizer`` smoke run
showed a saturated Δ≈-1.0 on a β checkpoint, which is suspicious —
before interpreting the Δ sign we need to know *mechanically*
whether the β feedback is actually perturbing the voxel
representation at the chosen operating point (noise=0.5, readout=5),
and whether the induced change lies along the direction the
localizer-trained linear decoder reads. This script runs the
three isolation blocks in one pass and emits the combined verdict.

**Q1 — feedback check.** Collect voxel-pooled L2/3 activity matrices
``Z_none``, ``Z_matched``, ``Z_mismatched`` on the β ckpt and report
per-voxel cue effect sizes in units of the baseline trial-std,
fraction of voxels with |Δ|>2σ_trial, and RSA-style correlations
between condition-mean voxel vectors.

**Q2 — decoder readout check.** Fit a LinearSVC on the localizer data
(5-fold CV acc + best-C refit on full localizer); extract the weight
vector ``w`` ∈ ℝ^n_voxels; compute trial-wise projections and
class-separation indices under each condition; report cosines
between the per-condition mean shifts and ``w``.

**Q3 — Δdecode.** Apply the frozen localizer-trained decoder to the
matched and mismatched test sets; report acc_matched, acc_mismatched,
Δ, trial-level bootstrap CI, and trial-level label-permutation p.

**Combined verdict** at the end routes through the 3-case matrix
defined by the lead (feedback_alive × cue_shift_along_decoder_axis
× |Δ|>ε → "real effect"; feedback_alive × orthogonal × Δ≈0 →
"decoder broken"; feedback_inert → "feedback broken"; other cases
reported verbatim).

All trial counts, noise, contrast, readout match the Task #74
operating point (noise=0.5, readout=5, n_voxels=64, 45°/135° binary,
localizer=800/orient, test=400/orient).
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
from scripts.v2.eval_fmri_decoder_kok import (
    _bootstrap_delta_trial_ci, _trial_label_permutation_p,
    _fit_locked_localizer_decoder,
)
from scripts.v2.eval_kok import _install_learned_w_q_gain
from scripts.v2.train_phase3_kok_learning import cue_mapping_from_seed


# ---------------------------------------------------------------------------
# Helpers (numpy-only; all trial collection is delegated to collect_trials)
# ---------------------------------------------------------------------------


def _q1_block(
    Z_none: np.ndarray, Z_matched: np.ndarray, Z_mismatched: np.ndarray,
    *, two_sigma_thresh: float = 2.0, inert_frac: float = 0.01,
    alive_frac: float = 0.05,
) -> dict[str, Any]:
    """Compute feedback-check summary statistics on voxel-pooled
    activity matrices.

    ``σ_trial`` is the per-voxel standard deviation across trials in
    the baseline ``Z_none`` (ddof=1 unbiased estimator). Cue-induced
    Δ magnitudes are reported as per-voxel absolute-mean-shifts in
    units of that baseline std, then aggregated by median and by
    "fraction of voxels with |Δ| > 2σ_trial".

    RSA-style voxel-wise correlations are Pearson correlations on the
    condition-mean voxel vectors (length = n_voxels).
    """
    sigma_trial = Z_none.std(axis=0, ddof=1)  # [n_voxels]
    n_voxels = Z_none.shape[1]

    mean_none = Z_none.mean(axis=0)
    mean_m = Z_matched.mean(axis=0)
    mean_u = Z_mismatched.mean(axis=0)

    # Guard against σ == 0 voxels (constant across trials). Extremely
    # unlikely under 0.5-std input noise but safe to check.
    safe_sigma = np.where(sigma_trial > 1e-12, sigma_trial, 1e-12)

    def norm_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a - b) / safe_sigma

    shift_m_vs_none = norm_shift(mean_m, mean_none)
    shift_u_vs_none = norm_shift(mean_u, mean_none)
    shift_m_vs_u = norm_shift(mean_m, mean_u)

    def frac_above(vals: np.ndarray) -> float:
        return float((vals > two_sigma_thresh).sum()) / float(n_voxels)

    corr_mu = float(np.corrcoef(mean_m, mean_u)[0, 1])
    corr_mn = float(np.corrcoef(mean_m, mean_none)[0, 1])
    corr_un = float(np.corrcoef(mean_u, mean_none)[0, 1])

    frac_m_none = frac_above(shift_m_vs_none)
    frac_u_none = frac_above(shift_u_vs_none)
    frac_m_u = frac_above(shift_m_vs_u)

    # Decision rule from the lead's dispatch. Use the union of matched
    # or mismatched vs baseline to judge "feedback alive".
    max_cue_vs_none = max(frac_m_none, frac_u_none)
    if max_cue_vs_none > alive_frac:
        verdict = "feedback_alive"
    elif max_cue_vs_none < inert_frac:
        verdict = "feedback_inert"
    else:
        verdict = "ambiguous"

    return {
        "baseline_mean_abs_per_voxel": float(np.abs(mean_none).mean()),
        "baseline_l1_over_nvox": float(
            np.abs(mean_none).sum() / float(n_voxels)
        ),
        "baseline_sigma_trial_median": float(np.median(sigma_trial)),
        "shift_median_matched_vs_none_over_sigma": float(
            np.median(shift_m_vs_none)
        ),
        "shift_median_mismatched_vs_none_over_sigma": float(
            np.median(shift_u_vs_none)
        ),
        "shift_median_matched_vs_mismatched_over_sigma": float(
            np.median(shift_m_vs_u)
        ),
        "frac_voxels_above_2sigma_matched_vs_none": frac_m_none,
        "frac_voxels_above_2sigma_mismatched_vs_none": frac_u_none,
        "frac_voxels_above_2sigma_matched_vs_mismatched": frac_m_u,
        "corr_matched_mismatched": corr_mu,
        "corr_matched_none": corr_mn,
        "corr_mismatched_none": corr_un,
        "n_voxels": int(n_voxels),
        "verdict": verdict,
    }


def _projections_stats(
    Z: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> dict[str, Any]:
    """Project trials onto the decoder axis and summarise."""
    proj = Z @ w  # [n_trials]
    classes = sorted(np.unique(y).tolist())
    per_class_means = {}
    for c in classes:
        per_class_means[int(c)] = float(proj[y == c].mean())
    pooled_std = float(proj.std(ddof=1))
    if len(classes) == 2:
        m_a = per_class_means[int(classes[0])]
        m_b = per_class_means[int(classes[1])]
        sep = float((m_a - m_b) / pooled_std) if pooled_std > 1e-12 else 0.0
    else:
        sep = float("nan")
    return {
        "mean": float(proj.mean()),
        "std": pooled_std,
        "per_class_means": {str(k): v for k, v in per_class_means.items()},
        "class_separation_index": sep,
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _q2_block(
    Z_none: np.ndarray, y_none: np.ndarray,
    Z_matched: np.ndarray, y_matched: np.ndarray,
    Z_mismatched: np.ndarray, y_mismatched: np.ndarray,
    *, localizer_cv: dict[str, Any], w: np.ndarray,
    orthogonal_cos_thresh: float = 0.15,
    too_small_frac_thresh: float = 0.001,
) -> dict[str, Any]:
    """Compute decoder-readout summary on voxel activity + weight vector.

    Decision rule (from the lead's dispatch, paraphrased):
      * ``cue_shift_along_decoder_axis`` when |cosine|>=0.15 on at
        least one of matched/mismatched shifts.
      * ``cue_shift_orthogonal_to_decoder`` when |cosine|<0.15 on
        both shifts AND the shift magnitude is nontrivial.
      * ``cue_shift_too_small_to_tell`` when the shift magnitude
        itself is a fraction of ``|w|`` too small to interpret a
        cosine against.
    """
    mean_none = Z_none.mean(axis=0)
    mean_m = Z_matched.mean(axis=0)
    mean_u = Z_mismatched.mean(axis=0)
    delta_m = mean_m - mean_none
    delta_u = mean_u - mean_none

    cos_m = _cosine(delta_m, w)
    cos_u = _cosine(delta_u, w)
    norm_w = float(np.linalg.norm(w))
    norm_delta_m = float(np.linalg.norm(delta_m))
    norm_delta_u = float(np.linalg.norm(delta_u))

    # Relative shift magnitude: ||Δ|| / ||w||_2 — purely informational.
    rel_m = norm_delta_m / norm_w if norm_w > 1e-12 else 0.0
    rel_u = norm_delta_u / norm_w if norm_w > 1e-12 else 0.0
    rel_max = max(rel_m, rel_u)

    if rel_max < too_small_frac_thresh:
        verdict = "cue_shift_too_small_to_tell"
    elif max(abs(cos_m), abs(cos_u)) >= orthogonal_cos_thresh:
        verdict = "cue_shift_along_decoder_axis"
    else:
        verdict = "cue_shift_orthogonal_to_decoder"

    return {
        "localizer_cv_acc_mean": float(localizer_cv["acc_mean"]),
        "localizer_cv_acc_std": float(localizer_cv["acc_std"]),
        "decoder_weight_l2_norm": norm_w,
        "projections": {
            "none": _projections_stats(Z_none, y_none, w),
            "matched": _projections_stats(Z_matched, y_matched, w),
            "mismatched": _projections_stats(Z_mismatched, y_mismatched, w),
        },
        "cosine_delta_matched_vs_w": cos_m,
        "cosine_delta_mismatched_vs_w": cos_u,
        "rel_delta_matched_over_w": float(rel_m),
        "rel_delta_mismatched_over_w": float(rel_u),
        "verdict": verdict,
    }


def _q3_block(
    Z_matched_test: np.ndarray, y_matched_test: np.ndarray,
    Z_mismatched_test: np.ndarray, y_mismatched_test: np.ndarray,
    *, clf, n_bootstrap: int = 1000, n_permutations: int = 1000,
    seed: int = 42, verdict_eps: float = 0.02,
) -> dict[str, Any]:
    """Apply the frozen decoder, compute Δdecode + CI + p."""
    pred_m = clf.predict(Z_matched_test)
    pred_u = clf.predict(Z_mismatched_test)
    correct_m = (pred_m == y_matched_test).astype(np.float64)
    correct_u = (pred_u == y_mismatched_test).astype(np.float64)
    acc_m = float(correct_m.mean())
    acc_u = float(correct_u.mean())
    delta_mean, ci_lo, ci_hi = _bootstrap_delta_trial_ci(
        correct_m, correct_u, n_resamples=int(n_bootstrap), seed=int(seed),
    )
    p_two_sided = _trial_label_permutation_p(
        correct_m, correct_u,
        n_permutations=int(n_permutations), seed=int(seed) + 17,
    )
    if abs(delta_mean) < float(verdict_eps):
        verdict = "null"
    elif delta_mean > 0.0:
        verdict = "sharpening"
    else:
        verdict = "dampening"
    return {
        "acc_matched": acc_m,
        "acc_mismatched": acc_u,
        "delta": {
            "mean": float(delta_mean),
            "ci": [float(ci_lo), float(ci_hi)],
            "n_bootstrap": int(n_bootstrap),
            "permutation_p_two_sided": float(p_two_sided),
            "n_permutations": int(n_permutations),
            "eps": float(verdict_eps),
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Combined verdict
# ---------------------------------------------------------------------------


def _combined_verdict(
    q1_verdict: str, q2_verdict: str, delta_mean: float, eps: float,
) -> str:
    """Route the three verdicts through the lead's decision matrix."""
    delta_big = abs(delta_mean) > float(eps)
    if q1_verdict == "feedback_alive":
        if q2_verdict == "cue_shift_along_decoder_axis" and delta_big:
            return "real_effect: feedback changes representation, decoder reads it, Δ sign is the scientific answer"
        if q2_verdict == "cue_shift_orthogonal_to_decoder" and not delta_big:
            return "decoder_broken: β changes representation but along a direction the voxel decoder ignores"
        return (
            f"feedback_alive but Q2={q2_verdict}, |Δ|{'>' if delta_big else '≤'}eps "
            "— report raw numbers; the three blocks disagree."
        )
    if q1_verdict == "feedback_inert":
        return "feedback_broken: β is not perturbing the voxel representation at this operating point"
    return f"ambiguous_Q1: {q1_verdict}; Q2={q2_verdict}; |Δ|{'>' if delta_big else '≤'}eps — report raw numbers"


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-voxels", type=int, default=64)
    p.add_argument("--n-trials-localizer-per-orient", type=int, default=800)
    p.add_argument("--n-trials-test-per-orient", type=int, default=400)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--contrast-min", type=float, default=0.1)
    p.add_argument("--contrast-max", type=float, default=0.3)
    p.add_argument("--n-warmup-steps", type=int, default=50)
    p.add_argument("--n-readout-steps", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--verdict-eps", type=float, default=0.02)
    p.add_argument(
        "--output", type=Path,
        default=Path("logs/task74/eval_feedback_decoder_isolation.json"),
    )
    return p


def _print_q1(q1: dict[str, Any]) -> None:
    nvox = q1["n_voxels"]
    print(f"voxel_representation_check (β-ckpt, n_voxels={nvox})")
    print(
        f"baseline (q_t=None) per-voxel mean, averaged over voxels: "
        f"||Z_none.mean(trial)||₁ / n_voxels = "
        f"{q1['baseline_l1_over_nvox']:.6f}"
    )
    print(
        f"                                                          "
        f"per-voxel std over trials, median = "
        f"{q1['baseline_sigma_trial_median']:.6f}"
    )
    print()
    print("cue effect sizes (per-voxel Δ in units of baseline σ_trial):")
    print(
        f"  |mean(Z_matched)    - mean(Z_none)| / σ_trial, median "
        f"across voxels = "
        f"{q1['shift_median_matched_vs_none_over_sigma']:.4f}"
    )
    print(
        f"  |mean(Z_mismatched) - mean(Z_none)| / σ_trial, median "
        f"across voxels = "
        f"{q1['shift_median_mismatched_vs_none_over_sigma']:.4f}"
    )
    print(
        f"  |mean(Z_matched)    - mean(Z_mismatched)|  / σ_trial, "
        f"median        = "
        f"{q1['shift_median_matched_vs_mismatched_over_sigma']:.4f}"
    )
    print()
    print("fraction of voxels with |Δ| > 2σ_trial:")
    f_mn = q1["frac_voxels_above_2sigma_matched_vs_none"] * nvox
    f_un = q1["frac_voxels_above_2sigma_mismatched_vs_none"] * nvox
    f_mu = q1["frac_voxels_above_2sigma_matched_vs_mismatched"] * nvox
    print(f"  matched vs none:       {f_mn:.0f} / {nvox}")
    print(f"  mismatched vs none:    {f_un:.0f} / {nvox}")
    print(f"  matched vs mismatched: {f_mu:.0f} / {nvox}")
    print()
    print(
        f"RSA-style: correlation( Z_matched.mean(trial), "
        f"Z_mismatched.mean(trial) ) across voxels = "
        f"{q1['corr_matched_mismatched']:.4f}"
    )
    print(
        f"          correlation( Z_matched.mean(trial), "
        f"Z_none.mean(trial) )       across voxels = "
        f"{q1['corr_matched_none']:.4f}"
    )
    print(
        f"          correlation( Z_mismatched.mean(trial), "
        f"Z_none.mean(trial) )       across voxels = "
        f"{q1['corr_mismatched_none']:.4f}"
    )
    print()
    print(f"VERDICT ON Q1: {q1['verdict']}")


def _print_q2(q2: dict[str, Any]) -> None:
    print("decoder_readout_check")
    print(
        f"localizer_decoder_acc (5-fold CV on cue=none data) = "
        f"{q2['localizer_cv_acc_mean']:.4f} ± "
        f"{q2['localizer_cv_acc_std']:.4f}"
    )
    print(
        f"decoder_weight_l2_norm = {q2['decoder_weight_l2_norm']:.4f}"
    )
    print()
    print("projections (trial-wise):")
    for tag in ("none", "matched", "mismatched"):
        pj = q2["projections"][tag]
        pcm = pj["per_class_means"]
        # Classes are orientation indices {0: 45°, 1: 135°} by collect_trials
        # label convention (0 = first orientation in list).
        cm_45 = pcm.get("0", float("nan"))
        cm_135 = pcm.get("1", float("nan"))
        print(
            f"  w · Z_{tag:<12}: mean={pj['mean']:+.4f}, "
            f"std={pj['std']:.4f}, per-class means "
            f"[45°={cm_45:+.4f}, 135°={cm_135:+.4f}]"
        )
    print()
    print("class-separation index = (mean_45 - mean_135) / pooled_std, "
          "on each condition:")
    sep_n = q2["projections"]["none"]["class_separation_index"]
    sep_m = q2["projections"]["matched"]["class_separation_index"]
    sep_u = q2["projections"]["mismatched"]["class_separation_index"]
    print(
        f"  none={sep_n:+.4f} matched={sep_m:+.4f} "
        f"mismatched={sep_u:+.4f}"
    )
    print()
    print(
        f"cosine( (mean(Z_matched)-mean(Z_none)),    w ) = "
        f"{q2['cosine_delta_matched_vs_w']:+.4f}"
    )
    print(
        f"cosine( (mean(Z_mismatched)-mean(Z_none)), w ) = "
        f"{q2['cosine_delta_mismatched_vs_w']:+.4f}"
    )
    print(
        f"rel ||Δmatched||/||w||={q2['rel_delta_matched_over_w']:.6f}   "
        f"rel ||Δmismatched||/||w||={q2['rel_delta_mismatched_over_w']:.6f}"
    )
    print()
    print(f"VERDICT ON Q2: {q2['verdict']}")


def _print_q3(q3: dict[str, Any]) -> None:
    print("Δdecode_block (frozen-localizer readout, β-ckpt)")
    print(
        f"acc_matched    = {q3['acc_matched']:.4f}"
    )
    print(
        f"acc_mismatched = {q3['acc_mismatched']:.4f}"
    )
    d = q3["delta"]
    print(
        f"Δ = acc_matched − acc_mismatched = {d['mean']:+.4f}   "
        f"ci=[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]   "
        f"p={d['permutation_p_two_sided']:.4f}"
    )
    print(f"verdict: {q3['verdict']}  (eps={d['eps']:g})")


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)

    bundle = load_checkpoint(
        args.checkpoint, seed=int(args.seed), device=args.device,
    )
    bundle.net.set_phase("phase3_kok")
    installed = _install_learned_w_q_gain(
        bundle, args.checkpoint, device=args.device,
    )
    if not installed:
        print(
            "[WARN] no W_q_gain key in ckpt — running on non-β ckpt? "
            "Results will reflect a W_q_gain=1.0 baseline.",
            flush=True,
        )
    bundle.net.eval()

    if "cue_mapping" in bundle.meta:
        cue_mapping = {
            int(k): float(v) for k, v in bundle.meta["cue_mapping"].items()
        }
    else:
        cue_mapping = cue_mapping_from_seed(int(args.seed))
    print(f"[iso] cue_mapping={cue_mapping}", flush=True)

    mask = build_voxel_pool_mask(int(args.n_voxels))

    # --- Collect three trial sets ----------------------------------------
    # Localizer: cue_mode=none. Used for BOTH Q1 baseline Z_none and Q2
    # decoder training (so the operating point is identical between
    # representation stats and decoder fit).
    print(
        f"[iso] collecting localizer (cue=none) "
        f"2 × {args.n_trials_localizer_per_orient} trials...",
        flush=True,
    )
    X_loc, y_loc, _ = collect_trials(
        bundle, orientations_deg=[45.0, 135.0],
        n_trials_per_orient=int(args.n_trials_localizer_per_orient),
        noise_std=float(args.noise_std),
        contrast_min=float(args.contrast_min),
        contrast_max=float(args.contrast_max),
        n_warmup=int(args.n_warmup_steps),
        n_readout=int(args.n_readout_steps),
        seed=int(args.seed), cue_mode="none", cue_mapping=None,
    )
    Z_none = pool_to_voxels(X_loc, mask)

    print(
        f"[iso] collecting matched cued test "
        f"2 × {args.n_trials_test_per_orient} trials...",
        flush=True,
    )
    X_m, y_m, _ = collect_trials(
        bundle, orientations_deg=[45.0, 135.0],
        n_trials_per_orient=int(args.n_trials_test_per_orient),
        noise_std=float(args.noise_std),
        contrast_min=float(args.contrast_min),
        contrast_max=float(args.contrast_max),
        n_warmup=int(args.n_warmup_steps),
        n_readout=int(args.n_readout_steps),
        seed=int(args.seed) + 1,
        cue_mode="expected", cue_mapping=cue_mapping,
    )
    Z_matched = pool_to_voxels(X_m, mask)

    print(
        f"[iso] collecting mismatched cued test "
        f"2 × {args.n_trials_test_per_orient} trials...",
        flush=True,
    )
    X_u, y_u, _ = collect_trials(
        bundle, orientations_deg=[45.0, 135.0],
        n_trials_per_orient=int(args.n_trials_test_per_orient),
        noise_std=float(args.noise_std),
        contrast_min=float(args.contrast_min),
        contrast_max=float(args.contrast_max),
        n_warmup=int(args.n_warmup_steps),
        n_readout=int(args.n_readout_steps),
        seed=int(args.seed) + 2,
        cue_mode="unexpected", cue_mapping=cue_mapping,
    )
    Z_mismatched = pool_to_voxels(X_u, mask)

    # --- Q1 ----------------------------------------------------------------
    print("[iso] computing Q1 (representation check)...", flush=True)
    q1 = _q1_block(Z_none, Z_matched, Z_mismatched)

    # --- Decoder fit (shared by Q2 + Q3) -----------------------------------
    print("[iso] fitting localizer decoder...", flush=True)
    clf, loc_cv = _fit_locked_localizer_decoder(
        Z_none, y_loc, seed=int(args.seed),
    )
    # LinearSVC (binary) stores shape (1, n_features); flatten.
    w = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    assert w.shape[0] == Z_none.shape[1], (
        f"decoder weight shape {w.shape} incompatible with voxel count "
        f"{Z_none.shape[1]}"
    )

    # --- Q2 ----------------------------------------------------------------
    q2 = _q2_block(
        Z_none, y_loc, Z_matched, y_m, Z_mismatched, y_u,
        localizer_cv=loc_cv, w=w,
    )

    # --- Q3 ----------------------------------------------------------------
    q3 = _q3_block(
        Z_matched, y_m, Z_mismatched, y_u,
        clf=clf,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
        verdict_eps=float(args.verdict_eps),
    )

    # --- Combined verdict --------------------------------------------------
    combined = _combined_verdict(
        q1["verdict"], q2["verdict"], q3["delta"]["mean"],
        float(args.verdict_eps),
    )

    # --- Stdout ------------------------------------------------------------
    print()
    _print_q1(q1)
    print()
    _print_q2(q2)
    print()
    _print_q3(q3)
    print()
    print(f"COMBINED: {combined}")

    # --- JSON dump ---------------------------------------------------------
    out = {
        "assay": "eval_feedback_decoder_isolation",
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "cue_mapping": {str(k): float(v) for k, v in cue_mapping.items()},
        "n_voxels": int(args.n_voxels),
        "noise_std": float(args.noise_std),
        "contrast_range": [float(args.contrast_min), float(args.contrast_max)],
        "n_warmup_steps": int(args.n_warmup_steps),
        "n_readout_steps": int(args.n_readout_steps),
        "n_trials_localizer_total": int(Z_none.shape[0]),
        "n_trials_matched": int(Z_matched.shape[0]),
        "n_trials_mismatched": int(Z_mismatched.shape[0]),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "combined_verdict": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[iso] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
