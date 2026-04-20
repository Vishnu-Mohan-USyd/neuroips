"""Phase-3 Kok evaluation — upgraded per Task #72 / critique §5.

Original Task #40 assays (retained for harness compatibility):
  1. **Mean L2/3 amplitude per cell (4 cells)** — 256 L2/3 E units split into 4
     non-overlapping groups of 64, per-condition probe-epoch means.
  2. **5-fold CV LinearSVC orientation decoding** (2-orientation; saturates).
  3. **Pref / non-pref asymmetry** (pref split by main-assay response).

New Task #72 upgrades:
  4. **Multi-orientation localizer** — 12 orientations × N trials, CUE-FREE.
     Yields per-unit preferred orientation (argmax across 12 orients) + per-unit
     FWHM (half-max width on rotated tuning curve, linear interpolation).
  5. **Fine-orientation discrimination** — anchor ± ``offset_deg`` probes, cue
     valid/invalid; SVM trained INDEPENDENTLY on localizer features then scored
     on fine-offset trials. Decoder discriminates small offsets (the sign, i.e.
     anchor−off vs anchor+off).
  6. **Localizer-split pref/non-pref** — assigns each unit its preferred
     orientation from the localizer, then uses orthogonal (pref + 90° mod 180°)
     as non-pref. Reports asymmetry on the two-orientation main-assay trials.
  7. **Response-weighted effective FWHM** per condition — per-trial
     ``Σ r_u · fwhm_u / Σ r_u`` (localizer FWHMs fixed per unit; condition
     modulates the weighting).
  8. **Bootstrap 95% CIs + permutation p-values** on fine-SVM Δ, asymmetry,
     response-weighted FWHM Δ.

Writes ``eval_kok.json`` next to the checkpoint. Exit 0 always.

Usage:
    python -m scripts.v2.eval_kok \\
        --checkpoint checkpoints/v2/phase3_kok/phase3_kok_s42.pt \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import (
    CheckpointBundle, load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


__all__ = [
    "run_kok_probe_trial", "run_kok_localizer_trial", "evaluate_kok",
]


# ---------------------------------------------------------------------------
# Single-trial forward (eval-only; no plasticity)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_kok_probe_trial(
    bundle: CheckpointBundle,
    *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    """Return the mean L2/3 rate vector during probe1 for one eval trial.

    Shape ``[n_l23_e]`` — used by every downstream metric
    (amp-per-cell-group, decoder, asymmetry).
    """
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    probe_rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, None
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=q_t)
        if delay_end <= t < probe1_end:
            probe_rates.append(info["r_l23"][0].clone())

    return torch.stack(probe_rates, dim=0).mean(dim=0)             # [n_l23_e]


@torch.no_grad()
def run_kok_localizer_trial(
    bundle: CheckpointBundle,
    *, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    """Cue-FREE probe trial for localizer tuning.

    Same timing as ``run_kok_probe_trial`` but ``q_t`` is ``None`` throughout —
    probe-epoch response is driven only by the sensory stimulus (no context-
    memory cue). Returns ``[n_l23_e]`` mean probe-epoch rate.
    """
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    probe_rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame = blank
        elif t < delay_end:
            frame = blank
        elif t < probe1_end:
            frame = probe
        elif t < blank2_end:
            frame = blank
        else:
            frame = probe
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=None)
        if delay_end <= t < probe1_end:
            probe_rates.append(info["r_l23"][0].clone())

    return torch.stack(probe_rates, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Metric blocks (Task #40 — retained)
# ---------------------------------------------------------------------------


def _per_group_means(
    r_l23: np.ndarray, n_groups: int = 4,
) -> list[float]:
    """Split ``[n_trials, n_l23]`` into ``n_groups`` column blocks; return
    the grand mean of each block (averaged over trials and units)."""
    n_units = r_l23.shape[1]
    if n_units % n_groups != 0:
        raise ValueError(
            f"n_l23={n_units} not divisible by n_groups={n_groups}"
        )
    block = n_units // n_groups
    return [
        float(r_l23[:, g * block:(g + 1) * block].mean())
        for g in range(n_groups)
    ]


def _svm_5fold_cv(
    X: np.ndarray, y: np.ndarray, *, seed: int,
) -> dict[str, Any]:
    """5-fold CV LinearSVC accuracy. Returns mean + per-fold scores."""
    try:
        from sklearn.model_selection import StratifiedKFold            # type: ignore
        from sklearn.svm import LinearSVC                              # type: ignore
    except ImportError:
        return {"error": "sklearn not available"}
    if X.shape[0] < 10 or len(np.unique(y)) < 2:
        return {"error": "insufficient data for 5-fold CV"}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    fold_acc: list[float] = []
    for tr, te in skf.split(X, y):
        clf = LinearSVC(random_state=int(seed), max_iter=5000, dual="auto")
        clf.fit(X[tr], y[tr])
        fold_acc.append(float(clf.score(X[te], y[te])))
    return {
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "per_fold": fold_acc,
    }


def _pref_nonpref_asymmetry(
    r_l23: np.ndarray,               # [n_trials, n_l23]
    orientation_deg: np.ndarray,     # [n_trials] — values in {45, 135}
    expected: np.ndarray,            # [n_trials] — bool, True when cue matches probe
) -> dict[str, Any]:
    """Split units by preferred orientation (from main-assay trials), then
    measure exp vs unexp effect. This is the Task #40 asymmetry — kept for
    the test harness. The Task-#72 localizer-based asymmetry is below."""
    orient_vals = tuple(np.unique(orientation_deg))
    if len(orient_vals) != 2:
        return {"error": "need exactly two orientations"}
    o_lo, o_hi = float(orient_vals[0]), float(orient_vals[1])

    r_lo = r_l23[orientation_deg == o_lo].mean(axis=0)              # [n_l23]
    r_hi = r_l23[orientation_deg == o_hi].mean(axis=0)
    pref_is_hi = r_hi >= r_lo                                       # [n_l23]

    def _cell_mean(mask_trials: np.ndarray, unit_mask: np.ndarray) -> float:
        sub = r_l23[np.ix_(mask_trials, unit_mask)]
        return float(sub.mean()) if sub.size else float("nan")

    pref_exp_lo = _cell_mean(
        (orientation_deg == o_lo) & expected, ~pref_is_hi,
    )
    pref_exp_hi = _cell_mean(
        (orientation_deg == o_hi) & expected, pref_is_hi,
    )
    pref_unexp_lo = _cell_mean(
        (orientation_deg == o_lo) & ~expected, ~pref_is_hi,
    )
    pref_unexp_hi = _cell_mean(
        (orientation_deg == o_hi) & ~expected, pref_is_hi,
    )
    nonpref_exp_lo = _cell_mean(
        (orientation_deg == o_lo) & expected, pref_is_hi,
    )
    nonpref_exp_hi = _cell_mean(
        (orientation_deg == o_hi) & expected, ~pref_is_hi,
    )
    nonpref_unexp_lo = _cell_mean(
        (orientation_deg == o_lo) & ~expected, pref_is_hi,
    )
    nonpref_unexp_hi = _cell_mean(
        (orientation_deg == o_hi) & ~expected, ~pref_is_hi,
    )

    pref_exp = float(np.nanmean([pref_exp_lo, pref_exp_hi]))
    pref_unexp = float(np.nanmean([pref_unexp_lo, pref_unexp_hi]))
    nonpref_exp = float(np.nanmean([nonpref_exp_lo, nonpref_exp_hi]))
    nonpref_unexp = float(np.nanmean([nonpref_unexp_lo, nonpref_unexp_hi]))
    asymmetry = (nonpref_unexp - nonpref_exp) - (pref_unexp - pref_exp)

    return {
        "pref_expected": pref_exp,
        "pref_unexpected": pref_unexp,
        "nonpref_expected": nonpref_exp,
        "nonpref_unexpected": nonpref_unexp,
        "asymmetry": float(asymmetry),
        "n_units_pref_hi": int(pref_is_hi.sum()),
        "n_units_pref_lo": int((~pref_is_hi).sum()),
    }


# ---------------------------------------------------------------------------
# Task #72: localizer-derived per-unit preferences + FWHMs
# ---------------------------------------------------------------------------


def _compute_localizer_stats(
    loc_trials: np.ndarray,           # [n_trials, n_l23]
    loc_orient: np.ndarray,           # [n_trials] deg
    localizer_orients: np.ndarray,    # [n_orients] deg (sorted ascending)
) -> dict[str, np.ndarray]:
    """Condense localizer trials into per-unit tuning curves + preferred
    orientation + FWHM.

    Returns (all np.ndarray):
      - ``tuning_curve``         [n_orients, n_l23]  mean rate per orient×unit
      - ``preferred_deg``        [n_l23]             argmax orient per unit
      - ``fwhm_deg``             [n_l23]             FWHM per unit (deg)
      - ``peak_rate``            [n_l23]             tuning peak per unit

    FWHM is computed on the curve rotated so the unit's preferred orient is
    at index 0, via linear interpolation to the half-max level on each side.
    If the curve doesn't cross half-max within the sampled range, FWHM
    clamps to the full sweep (180°).
    """
    n_orients = localizer_orients.size
    n_units = loc_trials.shape[1]
    tuning = np.zeros((n_orients, n_units), dtype=np.float64)
    for i, o in enumerate(localizer_orients):
        mask = np.isclose(loc_orient, o, atol=1e-6)
        if not mask.any():
            tuning[i] = np.nan
            continue
        tuning[i] = loc_trials[mask].mean(axis=0)
    argmax = np.argmax(tuning, axis=0)                              # [n_l23]
    preferred_deg = localizer_orients[argmax]
    peak_rate = tuning.max(axis=0)
    baseline = tuning.min(axis=0)
    half_max = 0.5 * (peak_rate + baseline)                         # [n_l23]

    # Rotate so preferred is at index 0, wrap-around padded.
    # Sweep spacing (assumes uniform)
    d_orient = float(localizer_orients[1] - localizer_orients[0]) if n_orients > 1 else 180.0
    sweep_deg = float(n_orients * d_orient)
    fwhm_deg = np.full(n_units, sweep_deg, dtype=np.float64)
    for u in range(n_units):
        if not np.isfinite(tuning[:, u]).all():
            fwhm_deg[u] = np.nan
            continue
        shift = int(argmax[u])
        curve = np.roll(tuning[:, u], -shift)                       # peak at 0
        hm = half_max[u]
        # Find half-max crossing to the right (index 1..n-1).
        # Curve wraps at index 0 again (peak); we search for the two nearest
        # crossings of hm on each "side" via angle from peak.
        # Sample symmetric distance from peak.
        # Crossing on right: first j>0 where curve[j] < hm, interpolate.
        def _cross(ordered_idx):
            prev_val = curve[0]
            prev_ang = 0.0
            for j in ordered_idx[1:]:
                ang = (j - 0) * d_orient if j > 0 else (n_orients - -j) * d_orient
                v = curve[j]
                if v <= hm:
                    # Linear interp between prev (prev_val, prev_ang) and this.
                    if prev_val == v:
                        return ang
                    frac = (prev_val - hm) / (prev_val - v + 1e-12)
                    return prev_ang + frac * (ang - prev_ang)
                prev_val, prev_ang = v, ang
            return ordered_idx[-1] * d_orient
        right = _cross(list(range(n_orients)))
        left_idx = list(range(n_orients))
        # search leftward: curve index descending from 0 (wrapping)
        left_path = [(n_orients - k) % n_orients for k in range(n_orients)]
        def _cross_left():
            prev_val = curve[0]
            prev_ang = 0.0
            for k in range(1, n_orients):
                j = (n_orients - k) % n_orients
                ang = k * d_orient
                v = curve[j]
                if v <= hm:
                    if prev_val == v:
                        return ang
                    frac = (prev_val - hm) / (prev_val - v + 1e-12)
                    return prev_ang + frac * (ang - prev_ang)
                prev_val, prev_ang = v, ang
            return (n_orients - 1) * d_orient
        left = _cross_left()
        fwhm_deg[u] = float(right + left)
    return {
        "tuning_curve": tuning,
        "preferred_deg": preferred_deg,
        "fwhm_deg": fwhm_deg,
        "peak_rate": peak_rate,
    }


# ---------------------------------------------------------------------------
# Task #72: localizer-derived pref/non-pref asymmetry on the anchor assay
# ---------------------------------------------------------------------------


def _pref_nonpref_from_localizer(
    r_l23: np.ndarray,
    orientation_deg: np.ndarray,
    expected: np.ndarray,
    localizer_preferred_deg: np.ndarray,
    anchor_tolerance_deg: float = 15.0,
) -> dict[str, Any]:
    """Pref / non-pref split on a two-anchor main assay, using localizer
    preferences.

    A unit is "pref the shown orient" if its localizer-preferred angle is
    within ``anchor_tolerance_deg`` of the shown anchor (mod 180°). It is
    "non-pref" if its preferred is within the tolerance of the orthogonal
    anchor (pref + 90° mod 180°). Units that fit neither are excluded from
    the split.
    """
    orient_vals = tuple(np.unique(orientation_deg))
    if len(orient_vals) != 2:
        return {"error": "need exactly two orientations"}
    o_lo, o_hi = float(orient_vals[0]), float(orient_vals[1])

    def _ang_diff(a, b):
        d = np.abs(a - b)
        d = np.minimum(d, 180.0 - d)
        return d

    pref_lo = _ang_diff(localizer_preferred_deg, o_lo) < anchor_tolerance_deg
    pref_hi = _ang_diff(localizer_preferred_deg, o_hi) < anchor_tolerance_deg

    def _cell_mean(mask_trials, unit_mask):
        if not unit_mask.any() or not mask_trials.any():
            return float("nan")
        sub = r_l23[np.ix_(mask_trials, unit_mask)]
        return float(sub.mean())

    # When shown o_lo: pref units = pref_lo, non-pref units = pref_hi.
    pref_exp_lo = _cell_mean((orientation_deg == o_lo) & expected, pref_lo)
    pref_exp_hi = _cell_mean((orientation_deg == o_hi) & expected, pref_hi)
    pref_unexp_lo = _cell_mean((orientation_deg == o_lo) & ~expected, pref_lo)
    pref_unexp_hi = _cell_mean((orientation_deg == o_hi) & ~expected, pref_hi)
    nonpref_exp_lo = _cell_mean((orientation_deg == o_lo) & expected, pref_hi)
    nonpref_exp_hi = _cell_mean((orientation_deg == o_hi) & expected, pref_lo)
    nonpref_unexp_lo = _cell_mean(
        (orientation_deg == o_lo) & ~expected, pref_hi,
    )
    nonpref_unexp_hi = _cell_mean(
        (orientation_deg == o_hi) & ~expected, pref_lo,
    )

    pref_exp = float(np.nanmean([pref_exp_lo, pref_exp_hi]))
    pref_unexp = float(np.nanmean([pref_unexp_lo, pref_unexp_hi]))
    nonpref_exp = float(np.nanmean([nonpref_exp_lo, nonpref_exp_hi]))
    nonpref_unexp = float(np.nanmean([nonpref_unexp_lo, nonpref_unexp_hi]))
    asymmetry = (nonpref_unexp - nonpref_exp) - (pref_unexp - pref_exp)

    return {
        "pref_expected": pref_exp,
        "pref_unexpected": pref_unexp,
        "nonpref_expected": nonpref_exp,
        "nonpref_unexpected": nonpref_unexp,
        "asymmetry": float(asymmetry),
        "n_units_pref_lo": int(pref_lo.sum()),
        "n_units_pref_hi": int(pref_hi.sum()),
        "anchor_tolerance_deg": float(anchor_tolerance_deg),
    }


# ---------------------------------------------------------------------------
# Task #72: fine-orientation discrimination (independent SVM trained on
# localizer; scored on fine-offset trials).
# ---------------------------------------------------------------------------


def _train_orientation_svm_on_localizer(
    loc_trials: np.ndarray, loc_orient: np.ndarray, *, seed: int,
) -> Any:
    """Multi-class LinearSVC trained on localizer (independent of main assay).

    Returns a sklearn classifier trained to predict orientation from
    ``[n_l23]`` probe response. Classes are the discrete orientations in
    ``loc_orient``.
    """
    from sklearn.svm import LinearSVC                                 # type: ignore
    labels = np.round(loc_orient).astype(np.int64) % 180
    clf = LinearSVC(random_state=int(seed), max_iter=5000, dual="auto")
    clf.fit(loc_trials, labels)
    return clf


def _score_fine_discrim(
    clf: Any,
    fine_trials: np.ndarray,
    fine_true_deg: np.ndarray,
    fine_expected: np.ndarray,
    *, tolerance_deg: float = 22.5,
) -> dict[str, Any]:
    """Score localizer-SVM on fine-offset trials.

    A trial counts as "correct" if the predicted orientation is within
    ``tolerance_deg`` of the true orientation (mod 180°). Default
    tolerance = 22.5° = half the localizer bin spacing (15°) + 7.5° margin.
    Reports expected vs unexpected accuracy + Δ.
    """
    pred = clf.predict(fine_trials).astype(np.float64)
    true = fine_true_deg.astype(np.float64)
    diff = np.abs(pred - true) % 180.0
    diff = np.minimum(diff, 180.0 - diff)
    correct = diff <= tolerance_deg
    exp_mask = fine_expected.astype(bool)
    if exp_mask.any():
        acc_exp = float(correct[exp_mask].mean())
    else:
        acc_exp = float("nan")
    if (~exp_mask).any():
        acc_unexp = float(correct[~exp_mask].mean())
    else:
        acc_unexp = float("nan")
    return {
        "tolerance_deg": float(tolerance_deg),
        "acc_expected": acc_exp,
        "acc_unexpected": acc_unexp,
        "delta_acc": float(acc_exp - acc_unexp),
        "n_expected": int(exp_mask.sum()),
        "n_unexpected": int((~exp_mask).sum()),
        "correct_per_trial": correct.astype(bool),
    }


# ---------------------------------------------------------------------------
# Task #72: response-weighted effective FWHM per condition
# ---------------------------------------------------------------------------


def _response_weighted_fwhm(
    r_l23: np.ndarray,
    expected: np.ndarray,
    fwhm_per_unit: np.ndarray,
) -> dict[str, Any]:
    """Per-condition mean of ``Σ r_u · fwhm_u / Σ r_u`` across trials.

    Interpretation: units that respond more strongly on a trial contribute
    their (static, localizer-derived) tuning width to the trial's effective
    FWHM. Sharpening predicts lower effective FWHM on expected trials
    (narrow-tuned units dominate).
    """
    # Guard against all-zero rows.
    r = np.clip(r_l23, a_min=0.0, a_max=None)
    denom = r.sum(axis=1)
    denom = np.where(denom > 1e-9, denom, np.nan)
    numer = (r * fwhm_per_unit[None, :]).sum(axis=1)
    per_trial_fwhm = numer / denom                                  # [n_trials]
    exp_mask = expected.astype(bool)
    fwhm_exp = float(np.nanmean(per_trial_fwhm[exp_mask])) if exp_mask.any() else float("nan")
    fwhm_unexp = float(np.nanmean(per_trial_fwhm[~exp_mask])) if (~exp_mask).any() else float("nan")
    return {
        "fwhm_expected_deg": fwhm_exp,
        "fwhm_unexpected_deg": fwhm_unexp,
        "delta_fwhm_deg": float(fwhm_exp - fwhm_unexp),
        "per_trial_fwhm_deg": per_trial_fwhm,  # kept for bootstrap; popped before JSON dump
    }


# ---------------------------------------------------------------------------
# Bootstrap + permutation helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci_mean_diff(
    values_a: np.ndarray, values_b: np.ndarray,
    *, n_resamples: int = 1000, seed: int = 0,
) -> dict[str, float]:
    """Trial-level bootstrap 95% CI for ``mean(a) − mean(b)``.

    Each resample draws ``len(a)`` with replacement from ``a`` and
    ``len(b)`` with replacement from ``b`` independently, computes the
    difference of means, and reports the percentile CI over the resamples.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            "mean_diff": float("nan"), "ci_lo": float("nan"),
            "ci_hi": float("nan"), "n_a": int(a.size), "n_b": int(b.size),
        }
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_resamples), dtype=np.float64)
    for k in range(int(n_resamples)):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        diffs[k] = aa.mean() - bb.mean()
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "n_a": int(a.size), "n_b": int(b.size),
        "n_resamples": int(n_resamples),
    }


def _permutation_p_two_sided(
    values_a: np.ndarray, values_b: np.ndarray,
    *, n_permutations: int = 1000, seed: int = 0,
) -> float:
    """Two-sided permutation p-value for difference of means.

    Pools a + b, shuffles labels ``n_permutations`` times, counts the fraction
    of shuffles with |Δmean| ≥ |Δobs|.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(int(seed))
    hits = 0
    for _ in range(int(n_permutations)):
        idx = rng.permutation(pool.size)
        aa = pool[idx[:n_a]]
        bb = pool[idx[n_a:]]
        if abs(aa.mean() - bb.mean()) >= obs:
            hits += 1
    return float((hits + 1) / (int(n_permutations) + 1))


# ---------------------------------------------------------------------------
# Top-level eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_kok(
    bundle: CheckpointBundle,
    *, n_trials_per_condition: int = 60,
    cue_mapping: Optional[dict[int, float]] = None,
    timing: Optional[KokTiming] = None,
    n_cell_groups: int = 4,
    noise_std: float = 0.01,
    seed: int = 42,
    n_localizer_orients: int = 12,
    n_localizer_trials: int = 20,
    fine_offsets_deg: tuple[float, ...] = (-10.0, 10.0),
    n_fine_trials_per_condition: int = 60,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    run_upgrades: bool = True,
) -> dict[str, Any]:
    """Run Kok expected/unexpected eval, return metrics dict for JSON dump.

    When ``run_upgrades`` is False the function runs only the Task #40
    assays (test-harness fast path). When True (default) it additionally
    runs the Task #72 localizer + fine-discrim + FWHM + bootstrap suite.
    """
    cfg = bundle.cfg
    timing = timing or KokTiming()
    cue_mapping = cue_mapping or cue_mapping_from_seed(seed)

    gen = torch.Generator(device=cfg.device); gen.manual_seed(int(seed))

    # --- Task #40 main assay: two anchors, expected vs unexpected --------
    trials_l23: list[Tensor] = []
    trials_orient: list[float] = []
    trials_cue: list[int] = []
    trials_expected: list[bool] = []

    for cue_id in (0, 1):
        cue_probe = cue_mapping[cue_id]
        other_probe = cue_mapping[1 - cue_id]
        for probe_deg, is_expected in (
            (cue_probe, True), (other_probe, False),
        ):
            for _ in range(int(n_trials_per_condition)):
                r = run_kok_probe_trial(
                    bundle, cue_id=cue_id,
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=float(noise_std),
                    generator=gen,
                )
                trials_l23.append(r)
                trials_orient.append(float(probe_deg))
                trials_cue.append(int(cue_id))
                trials_expected.append(bool(is_expected))

    r_arr = torch.stack(trials_l23, dim=0).cpu().numpy()            # [N, n_l23]
    orient = np.asarray(trials_orient, dtype=np.float64)
    expected = np.asarray(trials_expected, dtype=bool)

    per_cell = {}
    for tag, mask in (
        ("expected", expected),
        ("unexpected", ~expected),
        ("all", np.ones_like(expected, dtype=bool)),
    ):
        per_cell[tag] = _per_group_means(r_arr[mask], n_groups=int(n_cell_groups))

    uniq = np.unique(orient)
    y = np.where(orient == uniq[0], 0, 1).astype(np.int64)
    svm_all = _svm_5fold_cv(r_arr, y, seed=int(seed))
    svm_expected = _svm_5fold_cv(r_arr[expected], y[expected], seed=int(seed))
    svm_unexpected = _svm_5fold_cv(
        r_arr[~expected], y[~expected], seed=int(seed),
    )
    asymmetry_main = _pref_nonpref_asymmetry(r_arr, orient, expected)

    out: dict[str, Any] = {
        "assay": "eval_kok",
        "n_trials_per_condition": int(n_trials_per_condition),
        "cue_mapping": {int(k): float(v) for k, v in cue_mapping.items()},
        "orientations_deg": [float(x) for x in uniq],
        "per_cell_mean_l23": per_cell,
        "n_cell_groups": int(n_cell_groups),
        "svm": {
            "all": svm_all, "expected": svm_expected,
            "unexpected": svm_unexpected,
        },
        "pref_nonpref": asymmetry_main,
    }

    if not run_upgrades:
        return out

    # --- Task #72 localizer ----------------------------------------------
    localizer_orients = np.linspace(
        0.0, 180.0, int(n_localizer_orients), endpoint=False,
    ).astype(np.float64)
    loc_trials_list: list[np.ndarray] = []
    loc_orient_list: list[float] = []
    for o in localizer_orients:
        for _ in range(int(n_localizer_trials)):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(o),
                timing=timing, noise_std=float(noise_std),
                generator=gen,
            )
            loc_trials_list.append(r.cpu().numpy().astype(np.float64))
            loc_orient_list.append(float(o))
    loc_trials = np.stack(loc_trials_list, axis=0)                  # [N_loc, n_l23]
    loc_orient = np.asarray(loc_orient_list, dtype=np.float64)

    loc_stats = _compute_localizer_stats(
        loc_trials, loc_orient, localizer_orients,
    )
    preferred_deg = loc_stats["preferred_deg"]
    fwhm_per_unit = loc_stats["fwhm_deg"]

    # --- Localizer-based orientation decoder (INDEPENDENT of main assay) -
    try:
        from sklearn.svm import LinearSVC  # type: ignore             # noqa: F401
        loc_clf = _train_orientation_svm_on_localizer(
            loc_trials, loc_orient, seed=int(seed),
        )
        # Score the localizer itself (training-set accuracy proxy for quality).
        loc_train_pred = loc_clf.predict(loc_trials).astype(np.float64)
        loc_diff = np.abs(loc_train_pred - loc_orient) % 180.0
        loc_diff = np.minimum(loc_diff, 180.0 - loc_diff)
        tolerance_deg = float(180.0 / n_localizer_orients / 2.0 + 7.5)
        loc_train_acc = float((loc_diff <= tolerance_deg).mean())
        loc_svm_available = True
    except ImportError:
        loc_clf = None
        loc_train_acc = float("nan")
        tolerance_deg = float("nan")
        loc_svm_available = False

    # --- Fine-discrimination assay (anchor ± offset) ---------------------
    fine_trials_list: list[np.ndarray] = []
    fine_true_list: list[float] = []
    fine_expected_list: list[bool] = []
    fine_cue_list: list[int] = []
    fine_offset_list: list[float] = []
    for cue_id in (0, 1):
        anchor_expected = cue_mapping[cue_id]
        anchor_unexpected = cue_mapping[1 - cue_id]
        for expected_flag, anchor in (
            (True, anchor_expected), (False, anchor_unexpected),
        ):
            for offset in fine_offsets_deg:
                probe_deg = float((anchor + offset) % 180.0)
                for _ in range(int(n_fine_trials_per_condition)):
                    r = run_kok_probe_trial(
                        bundle, cue_id=int(cue_id),
                        probe_orientation_deg=probe_deg,
                        timing=timing, noise_std=float(noise_std),
                        generator=gen,
                    )
                    fine_trials_list.append(
                        r.cpu().numpy().astype(np.float64)
                    )
                    fine_true_list.append(probe_deg)
                    fine_expected_list.append(bool(expected_flag))
                    fine_cue_list.append(int(cue_id))
                    fine_offset_list.append(float(offset))
    fine_trials = np.stack(fine_trials_list, axis=0)                # [N_fine, n_l23]
    fine_true = np.asarray(fine_true_list, dtype=np.float64)
    fine_expected = np.asarray(fine_expected_list, dtype=bool)
    fine_offset = np.asarray(fine_offset_list, dtype=np.float64)

    if loc_svm_available:
        fine_result = _score_fine_discrim(
            loc_clf, fine_trials, fine_true, fine_expected,
            tolerance_deg=float(180.0 / n_localizer_orients / 2.0 + 7.5),
        )
    else:
        fine_result = {"error": "sklearn not available"}

    # --- Task-#72 localizer-based asymmetry ------------------------------
    asymmetry_task72 = _pref_nonpref_from_localizer(
        r_arr, orient, expected, preferred_deg,
    )

    # --- Response-weighted FWHM per condition ----------------------------
    fwhm_block = _response_weighted_fwhm(r_arr, expected, fwhm_per_unit)
    per_trial_fwhm = fwhm_block.pop("per_trial_fwhm_deg")

    # --- Bootstrap + permutation stats -----------------------------------
    # 1) Fine-discrim Δ accuracy (per-trial correct bool as a Bernoulli draw).
    if isinstance(fine_result, dict) and "correct_per_trial" in fine_result:
        correct_per_trial = fine_result.pop("correct_per_trial")
        a_corr = correct_per_trial[fine_expected].astype(np.float64)
        b_corr = correct_per_trial[~fine_expected].astype(np.float64)
        fine_ci = _bootstrap_ci_mean_diff(
            a_corr, b_corr, n_resamples=int(n_bootstrap), seed=int(seed),
        )
        fine_p = _permutation_p_two_sided(
            a_corr, b_corr, n_permutations=int(n_permutations), seed=int(seed),
        )
        fine_result["bootstrap_delta_acc"] = fine_ci
        fine_result["permutation_p_two_sided"] = fine_p

    # 2) Response-weighted FWHM Δ.
    fwhm_ci = _bootstrap_ci_mean_diff(
        per_trial_fwhm[expected], per_trial_fwhm[~expected],
        n_resamples=int(n_bootstrap), seed=int(seed),
    )
    fwhm_p = _permutation_p_two_sided(
        per_trial_fwhm[expected], per_trial_fwhm[~expected],
        n_permutations=int(n_permutations), seed=int(seed),
    )
    fwhm_block["bootstrap_delta_fwhm"] = fwhm_ci
    fwhm_block["permutation_p_two_sided"] = fwhm_p

    # 3) Asymmetry (cell-level) Δ bootstrap over units.
    #    For a per-unit bootstrap we need per-unit exp/unexp deltas. Derive:
    #    per-unit mean at (non-pref, exp/unexp) − per-unit mean at (pref, exp/unexp).
    #    Simpler proxy: two independent trial-level distributions feeding the
    #    asymmetry formula. Report a trial-level bootstrap of the scalar.
    def _bootstrap_asymmetry(seed_local: int) -> dict[str, float]:
        n_trials = r_arr.shape[0]
        rng_l = np.random.default_rng(int(seed_local))
        vals = np.empty(int(n_bootstrap), dtype=np.float64)
        for k in range(int(n_bootstrap)):
            idx = rng_l.choice(n_trials, size=n_trials, replace=True)
            a = _pref_nonpref_from_localizer(
                r_arr[idx], orient[idx], expected[idx], preferred_deg,
            )
            vals[k] = float(a.get("asymmetry", float("nan")))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {
                "mean_asymmetry": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
            }
        return {
            "mean_asymmetry": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
            "n_resamples": int(n_bootstrap),
        }

    asymmetry_task72["bootstrap_asymmetry"] = _bootstrap_asymmetry(int(seed))

    # --- Bundle Task #72 block -------------------------------------------
    out["localizer"] = {
        "n_orients": int(n_localizer_orients),
        "n_trials_per_orient": int(n_localizer_trials),
        "orients_deg": [float(x) for x in localizer_orients],
        "tuning_curve_mean_per_orient": [
            float(loc_stats["tuning_curve"][i].mean())
            for i in range(int(n_localizer_orients))
        ],
        "mean_fwhm_deg_population": float(np.nanmean(fwhm_per_unit)),
        "fraction_units_tuned": float(
            (loc_stats["peak_rate"] - loc_stats["tuning_curve"].min(axis=0)
             > 1e-6).mean()
        ),
        "orientation_decoder_train_acc": loc_train_acc,
        "orientation_decoder_tolerance_deg": tolerance_deg,
    }
    out["fine_discrim"] = {
        "offsets_deg": [float(x) for x in fine_offsets_deg],
        "n_trials_per_sub_condition": int(n_fine_trials_per_condition),
        **fine_result,
    }
    out["pref_nonpref_localizer"] = asymmetry_task72
    out["fwhm_per_condition"] = fwhm_block
    return out


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3 Kok evaluation (Task #72)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-per-condition", type=int, default=60)
    p.add_argument("--n-cell-groups", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--n-localizer-orients", type=int, default=12)
    p.add_argument("--n-localizer-trials", type=int, default=20)
    p.add_argument(
        "--fine-offsets-deg", type=float, nargs="+",
        default=[-10.0, 10.0],
    )
    p.add_argument("--n-fine-trials-per-condition", type=int, default=60)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument(
        "--skip-upgrades", action="store_true",
        help="run Task #40 assays only (fast path for harness tests)",
    )
    p.add_argument("--output", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    bundle = load_checkpoint(
        args.checkpoint, seed=int(args.seed), device=args.device,
    )
    bundle.net.set_phase("phase3_kok")
    cue_mapping = None
    if "cue_mapping" in bundle.meta:
        cue_mapping = {
            int(k): float(v) for k, v in bundle.meta["cue_mapping"].items()
        }
    results = evaluate_kok(
        bundle, n_trials_per_condition=int(args.n_trials_per_condition),
        cue_mapping=cue_mapping,
        n_cell_groups=int(args.n_cell_groups),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
        n_localizer_orients=int(args.n_localizer_orients),
        n_localizer_trials=int(args.n_localizer_trials),
        fine_offsets_deg=tuple(float(x) for x in args.fine_offsets_deg),
        n_fine_trials_per_condition=int(args.n_fine_trials_per_condition),
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        run_upgrades=not args.skip_upgrades,
    )
    out_path = args.output or (args.checkpoint.parent / "eval_kok.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
