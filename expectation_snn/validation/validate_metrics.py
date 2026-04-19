"""Functional validation for `assays.metrics` (Sprint 5a, Step 1).

Synthetic ground-truth tests for each of the 9 validator-signed metric
signatures: plant a known effect into the input, confirm the metric
recovers the correct direction / magnitude, and test edge cases
(empty bins, single-cell populations, perfectly balanced conditions).

Assays (1 per metric, plus edge-case assays):

 A1 suppression_vs_preference — enhance plant ⇒ bin-0 Δ > bin-last Δ
 A1b suppression_vs_preference — balanced plant ⇒ bin Δ ≈ 0 everywhere
 A2 suppression_vs_distance_from_expected — vertical structure when
     suppression tracks distance from expected
 A3 total_population_activity — empty mask returns 0, scale in Hz correct
 A4 preferred_channel_gain — per-cell peak recovered at own preferred θ
 A5 tuning_fit — von-Mises fit recovers FWHM of planted tuning
 A6 omission_subtracted_response — stim > omit ⇒ positive Δ
 A7 svm_decoding_accuracy — well-tuned population decodes θ well above chance
 A8 pseudo_voxel_forward_model — 6/6 families return valid shapes
 A9 evidence_package — combined gate

Run:
    python -m expectation_snn.validation.validate_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..assays.metrics import (
    _VOXEL_MODEL_FAMILIES,
    _wrap_pi,
    evidence_package,
    omission_subtracted_response,
    preferred_channel_gain,
    pseudo_voxel_forward_model,
    suppression_vs_distance_from_expected,
    suppression_vs_preference,
    svm_decoding_accuracy,
    total_population_activity,
    tuning_fit,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_CELLS = 48
N_THETAS = 12
SIGMA_TUNING_DEG = 15.0
PEAK_HZ = 10.0
BASELINE_HZ = 1.0


def _tiled_preferences(n_cells: int = N_CELLS) -> np.ndarray:
    """Uniformly tile preferences over the 0..π ring."""
    thetas = np.linspace(0.0, np.pi, N_THETAS, endpoint=False)
    return np.tile(thetas, n_cells // N_THETAS)[:n_cells]


def _thetas_grid() -> np.ndarray:
    return np.linspace(0.0, np.pi, N_THETAS, endpoint=False)


def _synth_tuning_counts(
    prefs: np.ndarray,
    presented_theta_per_trial: np.ndarray,
    gain_for_expected: float = 0.0,
    expected_mask: np.ndarray = None,
    peak_hz: float = PEAK_HZ,
    baseline_hz: float = BASELINE_HZ,
    sigma_deg: float = SIGMA_TUNING_DEG,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Per-cell Poisson spike counts from Gaussian tuning about preference.

    Peak at cell's preferred θ. Optional multiplicative gain on
    expected-condition trials.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_cells = len(prefs)
    n_trials = len(presented_theta_per_trial)
    sigma = np.deg2rad(sigma_deg)
    counts = np.zeros((n_cells, n_trials), dtype=np.float64)
    for t, th in enumerate(presented_theta_per_trial):
        d = _wrap_pi(prefs - th)
        tuning = peak_hz * np.exp(-0.5 * (d / sigma) ** 2) + baseline_hz
        if expected_mask is not None and bool(expected_mask[t]):
            tuning = tuning * (1.0 + gain_for_expected)
        counts[:, t] = rng.poisson(tuning)
    return counts


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class MetricsValidationReport:
    results: Tuple[Tuple[str, bool, str], ...]

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self) -> str:
        lines = ["metrics validation:"]
        for name, ok, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name:50s} {detail}")
        lines.append("  " + "-" * 40)
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assay runners
# ---------------------------------------------------------------------------

def _assay_pref_rank_enhance(seed: int = 0) -> Tuple[bool, str]:
    """A1: plant a positive bin-0 Δ (enhance); expect bin_delta[0] > bin_delta[-1]."""
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    n_trials = 120
    presented = rng.choice(thetas_g, size=n_trials)
    # Balance expected/unexpected per presented θ for a fair test.
    cond = rng.integers(0, 2, size=n_trials)
    counts = _synth_tuning_counts(
        prefs, presented,
        gain_for_expected=0.5, expected_mask=cond, rng=rng,
    )
    r = suppression_vs_preference(counts, prefs, presented, cond, n_bins=10)
    ok = r["bin_delta"][0] > r["bin_delta"][-1] and r["bin_delta"][0] > 0.2
    return ok, (f"bin0 Δ={r['bin_delta'][0]:+.2f}, "
                f"bin-last Δ={r['bin_delta'][-1]:+.2f}")


def _assay_pref_rank_balanced(seed: int = 1) -> Tuple[bool, str]:
    """A1b: balanced conditions (no expectation effect) ⇒ |bin Δ| small everywhere."""
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    n_trials = 240
    presented = rng.choice(thetas_g, size=n_trials)
    cond = rng.integers(0, 2, size=n_trials)
    # No gain modulation — expected and unexpected identical in expectation.
    counts = _synth_tuning_counts(
        prefs, presented, gain_for_expected=0.0, expected_mask=cond, rng=rng,
    )
    r = suppression_vs_preference(counts, prefs, presented, cond, n_bins=10)
    max_abs = float(np.max(np.abs(r["bin_delta"])))
    ok = max_abs < 1.0   # tolerable noise under Poisson with PEAK=10
    return ok, f"max|bin Δ|={max_abs:.3f}"


def _assay_dist_grid_structure(seed: int = 2) -> Tuple[bool, str]:
    """A2: plant distance-from-expected suppression; expect vertical structure.

    Shape: when suppression depends only on distance from expected (not
    distance from presented), grid rows [0] should show lower response
    than row [-1] (far from expected = stronger response), irrespective
    of column position.

    To actually isolate dist-from-expected dependence, we synthesize
    per-trial responses that are a function of d_from_expected only.
    """
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    n_trials = 240
    presented = rng.choice(thetas_g, size=n_trials)
    expected = rng.choice(thetas_g, size=n_trials)
    # Plant: response decreases with d_from_expected (stronger suppression
    # at preferred θ under expected condition, Kok-like).
    sigma = np.deg2rad(25.0)
    counts = np.zeros((len(prefs), n_trials))
    for t in range(n_trials):
        d_exp = _wrap_pi(prefs - expected[t])
        # Higher response AWAY from expected ⇒ row-0 < row-last.
        rate = 2.0 + 6.0 * (d_exp / (np.pi / 2.0))   # 2..8 Hz
        counts[:, t] = rng.poisson(rate)
    r = suppression_vs_distance_from_expected(
        counts, prefs, expected, presented, grid_bins=(8, 8),
    )
    # Average across columns (distance from presented) per row.
    row_means = np.nanmean(r["grid"], axis=1)
    ok = row_means[0] < row_means[-1]
    return ok, f"row_means[0]={row_means[0]:.2f} < row_means[-1]={row_means[-1]:.2f}"


def _assay_total_population_empty(seed: int = 3) -> Tuple[bool, str]:
    """A3: empty mask ⇒ rate=0; full mask ⇒ rate > 0 in correct Hz band."""
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    n_trials = 50
    presented = rng.choice(thetas_g, size=n_trials)
    counts = _synth_tuning_counts(
        prefs, presented, peak_hz=5.0, baseline_hz=1.0, rng=rng,
    )
    # With 500 ms window, peak tuning 5 Hz = 2.5 spikes/trial peak, baseline ~0.5.
    # For cells × window, rate per cell should be roughly ~2 Hz on average.
    r_empty = total_population_activity(
        counts, np.zeros(len(prefs), dtype=bool), window_ms=500.0,
    )
    r_full = total_population_activity(
        counts, np.ones(len(prefs), dtype=bool), window_ms=500.0,
    )
    ok = (
        r_empty["total_rate_hz"] == 0.0
        and r_empty["n_pop"] == 0
        and 0.5 < r_full["total_rate_hz"] < 10.0
        and r_full["n_pop"] == len(prefs)
    )
    return ok, (f"empty rate={r_empty['total_rate_hz']:.2f} Hz  "
                f"full rate={r_full['total_rate_hz']:.2f} Hz (n={r_full['n_pop']})")


def _assay_preferred_channel_gain(seed: int = 4) -> Tuple[bool, str]:
    """A4: per-cell peak recovered at own-pref θ (> overall mean)."""
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    # Per-θ mean tuning curve (deterministic: no Poisson noise).
    sigma = np.deg2rad(SIGMA_TUNING_DEG)
    sc_by_theta = np.zeros((len(prefs), len(thetas_g)))
    for i, th in enumerate(thetas_g):
        d = _wrap_pi(prefs - th)
        sc_by_theta[:, i] = PEAK_HZ * np.exp(-0.5 * (d / sigma) ** 2) + BASELINE_HZ
    r = preferred_channel_gain(sc_by_theta, prefs, thetas_g)
    # pref_ch_gain should be close to PEAK_HZ+BASELINE_HZ; overall mean is lower.
    ok = float(r.mean()) > 1.5 * float(sc_by_theta.mean()) and r.shape == (len(prefs),)
    return ok, (f"mean_pref_gain={r.mean():.2f} vs overall_mean="
                f"{sc_by_theta.mean():.2f}")


def _assay_tuning_fit_fwhm(seed: int = 5) -> Tuple[bool, str]:
    """A5: von-Mises fit recovers FWHM ~40° for planted Gaussian (σ=15°).

    Gaussian σ=15° → FWHM_gauss = 2·√(2 ln 2)·15° ≈ 35.3°. Von Mises fitted
    to this should return FWHM in a similar ballpark.
    """
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    sigma = np.deg2rad(SIGMA_TUNING_DEG)
    sc_by_theta = np.zeros((len(prefs), len(thetas_g)))
    for i, th in enumerate(thetas_g):
        d = _wrap_pi(prefs - th)
        sc_by_theta[:, i] = PEAK_HZ * np.exp(-0.5 * (d / sigma) ** 2) + BASELINE_HZ
    r = tuning_fit(sc_by_theta, thetas_g, fit="von_mises")
    good = r["r2"] > 0.8
    fwhm_deg = np.rad2deg(r["fwhm_rad"][good])
    med = float(np.nanmedian(fwhm_deg))
    ok = 20.0 < med < 60.0 and int(good.sum()) >= len(prefs) // 2
    return ok, (f"good_fits={int(good.sum())}/{len(prefs)}  "
                f"median FWHM={med:.1f}°")


def _assay_omission_subtraction(seed: int = 6) -> Tuple[bool, str]:
    """A6: stim > omit ⇒ positive per-cell Δ."""
    rng = np.random.default_rng(seed)
    n_cells = 20
    stim = rng.poisson(4.0, size=(n_cells, 30))
    omit = rng.poisson(1.0, size=(n_cells, 30))
    r = omission_subtracted_response(stim, omit)
    ok = float(r.mean()) > 1.5 and r.shape == (n_cells,)
    return ok, f"mean_delta={r.mean():.2f}  shape={r.shape}"


def _assay_svm_decoding(seed: int = 7) -> Tuple[bool, str]:
    """A7: decode presented θ from tuned population — accuracy ≫ chance."""
    rng = np.random.default_rng(seed)
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    # 20 trials per θ = 240 total.
    presented = np.repeat(thetas_g, 20)
    counts = _synth_tuning_counts(prefs, presented, rng=rng)
    labels = np.digitize(presented, thetas_g) - 1
    r = svm_decoding_accuracy(counts.T, labels, cv=5, seed=seed)
    # 12 classes → chance ≈ 0.083. Expect > 0.4 for this tuning.
    ok = r["accuracy"] > 0.4 and r["n_classes"] == 12
    return ok, f"acc={r['accuracy']:.3f}  CI={r['accuracy_ci']}"


def _assay_pseudo_voxel_shapes(seed: int = 8) -> Tuple[bool, str]:
    """A8: all 6 model families return correct shapes and preferred voxel."""
    prefs = _tiled_preferences()
    thetas_g = _thetas_grid()
    # Per-cell tuning curve.
    sigma = np.deg2rad(SIGMA_TUNING_DEG)
    sc_by_theta = np.zeros((len(prefs), len(thetas_g)))
    for i, th in enumerate(thetas_g):
        d = _wrap_pi(prefs - th)
        sc_by_theta[:, i] = PEAK_HZ * np.exp(-0.5 * (d / sigma) ** 2) + BASELINE_HZ
    # 4 pseudo-voxels by contiguous spatial-bin assignment.
    voxel_bins = (np.arange(len(prefs)) // (len(prefs) // 4)).astype(np.int64)
    details = []
    ok = True
    for fam in _VOXEL_MODEL_FAMILIES:
        r = pseudo_voxel_forward_model(
            sc_by_theta, voxel_bins, fam, thetas_g, effect_size=0.2,
        )
        shape_ok = (
            r["voxel_tuning_baseline"].shape == (4, len(thetas_g))
            and r["voxel_tuning_predicted"].shape == (4, len(thetas_g))
            and r["n_voxels"] == 4
        )
        # Direction of deformation at preferred voxel peak:
        pv = r["preferred_voxel"]
        base_peak = float(r["voxel_tuning_baseline"][pv].max())
        pred_peak = float(r["voxel_tuning_predicted"][pv].max())
        if fam == "local_gain_dampen":
            dir_ok = pred_peak < base_peak
        elif fam in ("local_gain_enhance", "global_gain"):
            dir_ok = pred_peak > base_peak - 1e-9
        else:
            dir_ok = True    # tuning/remote — just check shape
        ok = ok and shape_ok and dir_ok
        details.append(f"{fam}:{'OK' if shape_ok and dir_ok else 'BAD'}")
    return ok, " ".join(details)


def _assay_evidence_package_gate(seed: int = 9) -> Tuple[bool, str]:
    """A9: evidence_package passes with consistent +ve effect; fails on
    inconsistent ablation (no collapse)."""
    # Happy path: all seeds positive, ablation collapses, held-out matches.
    main = {42: 0.30, 7: 0.25, 123: 0.35, 2024: 0.28, 11: 0.31}
    ablate = {42: 0.02, 7: 0.05, 123: 0.01, 2024: 0.03, 11: 0.04}
    held = {99: 0.27, 314: 0.29}
    r_ok = evidence_package(main, held, ablate, pre_registered_direction=+1)
    # Fail path: ablation does NOT collapse.
    ablate_bad = {42: 0.28, 7: 0.26, 123: 0.32, 2024: 0.29, 11: 0.30}
    r_bad = evidence_package(main, held, ablate_bad, pre_registered_direction=+1)
    ok = r_ok["passed"] and not r_bad["passed"]
    return ok, (f"happy-path passed={r_ok['passed']} (d={r_ok['cohens_d']:.2f})  "
                f"no-collapse passed={r_bad['passed']} "
                f"(collapses={r_bad['ablation_collapses']})")


def _assay_single_cell_population(seed: int = 10) -> Tuple[bool, str]:
    """Edge: single-cell population — total_population_activity works."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(1, 30))
    mask = np.array([True])
    r = total_population_activity(counts, mask, window_ms=500.0)
    ok = r["n_pop"] == 1 and r["total_rate_hz"] > 0
    return ok, f"rate={r['total_rate_hz']:.2f} Hz  n_pop={r['n_pop']}"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_validation(verbose: bool = True) -> MetricsValidationReport:
    results = (
        ("A1  suppression_vs_preference (enhance)", *_assay_pref_rank_enhance()),
        ("A1b suppression_vs_preference (balanced)", *_assay_pref_rank_balanced()),
        ("A2  suppression_vs_distance_from_expected", *_assay_dist_grid_structure()),
        ("A3  total_population_activity (edge cases)", *_assay_total_population_empty()),
        ("A4  preferred_channel_gain", *_assay_preferred_channel_gain()),
        ("A5  tuning_fit (von Mises FWHM recovery)", *_assay_tuning_fit_fwhm()),
        ("A6  omission_subtracted_response", *_assay_omission_subtraction()),
        ("A7  svm_decoding_accuracy", *_assay_svm_decoding()),
        ("A8  pseudo_voxel_forward_model (6 families)", *_assay_pseudo_voxel_shapes()),
        ("A9  evidence_package (gate aggregation)", *_assay_evidence_package_gate()),
        ("edge  single-cell population", *_assay_single_cell_population()),
    )
    rep = MetricsValidationReport(results=results)
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
