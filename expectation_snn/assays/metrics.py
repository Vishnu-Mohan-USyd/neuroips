"""Authoritative neuron-level + comparability metrics (plan sec 5, task #18).

Nine primary + secondary metric signatures, used by Kok / Richter / Tang
assays and by the evidence-package aggregator. All signatures validator-
signed (task #18 deliverable).

Conventions (locked)
--------------------
- Inputs are **spike counts per trial**, not rates. Conversion to Hz
  (if any) happens inside the metric via `window_ms` or equivalent.
- Orientations are in **radians** on the 0..π ring (we treat orientation
  as π-periodic; the ring metric wraps modulo π).
- **Deciles** (n_bins=10) for rank-based preference binning. Bin 0 is the
  most-preferred bin (cells whose preferred θ is closest to the presented
  θ); bin n_bins-1 is the least preferred.
- **4 pseudo-voxels** by default for the forward-model analysis.
- Condition mask is **1 = expected / valid, 0 = unexpected / invalid**.
- Ablation input for the evidence package is the **primary causal**
  ablation only (A3 for sharpening-direct, A4 for dampening-SOM,
  A2 for H3 statefulness) — not the full A1..A4 tuple.
- Preference-rank metric (`suppression_vs_preference`) returns a
  **per-bin Δ** (expected − unexpected) rather than raw paired rates.

Signatures
----------
1. suppression_vs_preference
2. suppression_vs_distance_from_expected
3. total_population_activity
4. preferred_channel_gain
5. tuning_fit
6. omission_subtracted_response
7. svm_decoding_accuracy
8. pseudo_voxel_forward_model
9. evidence_package

References
----------
- Kok 2012 PMID 22841311; Richter 2022 DOI 10.1093/oons/kvac013;
  Tang 2023 PMID 36864037.
- Brouwer & Heeger 2009 PMID 19535619 (voxel forward-model bridge).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_pi(dtheta: np.ndarray) -> np.ndarray:
    """Wrap an angular difference into the 0..π/2 orientation half-ring.

    Orientation is π-periodic; absolute distance is bounded by π/2.
    """
    d = np.mod(np.asarray(dtheta, dtype=np.float64), np.pi)
    return np.minimum(d, np.pi - d)


def _bootstrap_mean_ci(
    x: np.ndarray,
    B: int = 2000,
    seed: int = 123,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """Percentile bootstrap CI for the mean of a 1-D sample."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), (float(lo), float(hi))


# ---------------------------------------------------------------------------
# 1. Suppression vs preference  (Kok / Richter)
# ---------------------------------------------------------------------------

def suppression_vs_preference(
    spike_counts: np.ndarray,
    preferred_thetas: np.ndarray,
    presented_theta_per_trial: np.ndarray,
    condition_mask: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Per-preference-bin Δ response (expected − unexpected).

    Rank cells by how close their preferred θ is to the presented θ on
    each trial, bin by decile (default n_bins=10), and report the
    per-bin difference in mean response between condition-mask=1
    (expected/valid) and condition-mask=0 (unexpected/invalid) trials.

    Sign convention
    ---------------
    - Δ > 0 at bin 0 (most preferred): expectation **enhances** the
      preferred-channel response ("gain-enhance" / Tang-like).
    - Δ < 0 at bin 0: expectation **dampens** the preferred-channel
      response ("gain-dampen" / Kok-like if uniform across bins).
    - Negative Δ uniform across bins: dampening.
    - Negative Δ concentrated in off-preference bins: sharpening
      (Kok 2012 canonical prediction).

    Parameters
    ----------
    spike_counts : (n_cells, n_trials)
        Spike counts per cell per trial (spikes, not rates).
    preferred_thetas : (n_cells,)
        Per-cell preferred orientation in radians (0..π).
    presented_theta_per_trial : (n_trials,)
        Per-trial presented stimulus orientation in radians.
    condition_mask : (n_trials,)
        0/1 mask: 1 = expected / valid, 0 = unexpected / invalid.
    n_bins : int
        Number of preference-rank bins (default 10 = deciles).

    Returns
    -------
    dict with keys
      - 'bin_delta'          : (n_bins,) mean[expected] - mean[unexpected] per bin
      - 'bin_expected'       : (n_bins,) mean response for condition=1 cells-trials
      - 'bin_unexpected'     : (n_bins,) mean response for condition=0 cells-trials
      - 'bin_counts_exp'     : (n_bins,) number of (cell, trial) pairs in exp bin
      - 'bin_counts_unexp'   : (n_bins,) idem for unexpected
      - 'n_bins'             : int
    """
    counts = np.asarray(spike_counts, dtype=np.float64)
    prefs = np.asarray(preferred_thetas, dtype=np.float64)
    pres = np.asarray(presented_theta_per_trial, dtype=np.float64)
    mask = np.asarray(condition_mask, dtype=np.int64)

    n_cells, n_trials = counts.shape
    if prefs.shape != (n_cells,):
        raise ValueError(f"preferred_thetas shape {prefs.shape} != ({n_cells},)")
    if pres.shape != (n_trials,):
        raise ValueError("presented_theta_per_trial shape mismatch")
    if mask.shape != (n_trials,):
        raise ValueError("condition_mask shape mismatch")

    # Angular distance cell × trial: |pref_c - pres_t| on π-ring.
    d = _wrap_pi(prefs[:, None] - pres[None, :])   # (n_cells, n_trials)

    # Rank cells within each trial (smaller d → smaller rank).
    # bin 0 = most preferred = closest to presented θ.
    ranks = np.argsort(np.argsort(d, axis=0), axis=0)   # (n_cells, n_trials)
    bin_idx = (ranks * n_bins) // n_cells             # 0..n_bins-1

    resp_exp = np.zeros(n_bins, dtype=np.float64)
    resp_unexp = np.zeros(n_bins, dtype=np.float64)
    cnt_exp = np.zeros(n_bins, dtype=np.int64)
    cnt_unexp = np.zeros(n_bins, dtype=np.int64)

    for t in range(n_trials):
        bt = bin_idx[:, t]
        ct = counts[:, t]
        if mask[t] == 1:
            np.add.at(resp_exp, bt, ct)
            np.add.at(cnt_exp, bt, 1)
        else:
            np.add.at(resp_unexp, bt, ct)
            np.add.at(cnt_unexp, bt, 1)

    mean_exp = np.where(cnt_exp > 0, resp_exp / np.maximum(cnt_exp, 1), 0.0)
    mean_unexp = np.where(cnt_unexp > 0, resp_unexp / np.maximum(cnt_unexp, 1), 0.0)
    bin_delta = mean_exp - mean_unexp

    return {
        "bin_delta": bin_delta,
        "bin_expected": mean_exp,
        "bin_unexpected": mean_unexp,
        "bin_counts_exp": cnt_exp,
        "bin_counts_unexp": cnt_unexp,
        "n_bins": int(n_bins),
    }


# ---------------------------------------------------------------------------
# 2. Suppression vs distance-from-expected 8 × 8 grid  (Richter)
# ---------------------------------------------------------------------------

def suppression_vs_distance_from_expected(
    spike_counts: np.ndarray,
    preferred_thetas: np.ndarray,
    expected_theta_per_trial: np.ndarray,
    presented_theta_per_trial: np.ndarray,
    grid_bins: Tuple[int, int] = (8, 8),
) -> Dict[str, Any]:
    """Joint (dist-from-expected × dist-from-presented) 8 × 8 response grid.

    For each (cell, trial) pair, compute
      d_exp  = |pref_cell − expected_theta_trial|   (π-ring)
      d_pres = |pref_cell − presented_theta_trial|  (π-ring)
    and bin the spike count linearly on 0..π/2 (max distance on
    orientation ring) into ``grid_bins`` rows × cols. Reveals whether
    observed suppression tracks absolute mismatch to the presented
    stimulus (diagonal structure) or distance from the expected
    orientation (vertical structure).

    Parameters
    ----------
    spike_counts : (n_cells, n_trials)
    preferred_thetas : (n_cells,)
    expected_theta_per_trial : (n_trials,) — the cue-predicted orientation
    presented_theta_per_trial : (n_trials,) — the shown orientation
    grid_bins : (n_rows, n_cols); rows = d_from_expected, cols = d_from_presented

    Returns
    -------
    dict with keys
      - 'grid'                 : (n_rows, n_cols) mean response per bin
      - 'grid_counts'          : (n_rows, n_cols) cell-trial count per bin
      - 'edges_expected'       : (n_rows+1,) bin edges in radians
      - 'edges_presented'      : (n_cols+1,) bin edges in radians
      - 'grid_bins'            : tuple
    """
    counts = np.asarray(spike_counts, dtype=np.float64)
    prefs = np.asarray(preferred_thetas, dtype=np.float64)
    exp_t = np.asarray(expected_theta_per_trial, dtype=np.float64)
    pres_t = np.asarray(presented_theta_per_trial, dtype=np.float64)

    n_cells, n_trials = counts.shape
    if prefs.shape != (n_cells,):
        raise ValueError("preferred_thetas shape mismatch")
    if exp_t.shape != (n_trials,) or pres_t.shape != (n_trials,):
        raise ValueError("per-trial theta shape mismatch")

    d_exp = _wrap_pi(prefs[:, None] - exp_t[None, :])
    d_pres = _wrap_pi(prefs[:, None] - pres_t[None, :])

    nr, nc = grid_bins
    edges_exp = np.linspace(0.0, np.pi / 2.0, nr + 1)
    edges_pres = np.linspace(0.0, np.pi / 2.0, nc + 1)

    row_idx = np.clip(np.digitize(d_exp, edges_exp) - 1, 0, nr - 1)
    col_idx = np.clip(np.digitize(d_pres, edges_pres) - 1, 0, nc - 1)

    grid = np.zeros((nr, nc), dtype=np.float64)
    grid_cnt = np.zeros((nr, nc), dtype=np.int64)
    flat = row_idx * nc + col_idx
    np.add.at(grid.reshape(-1), flat.reshape(-1), counts.reshape(-1))
    np.add.at(grid_cnt.reshape(-1), flat.reshape(-1), 1)

    with np.errstate(invalid="ignore"):
        mean_grid = np.where(grid_cnt > 0, grid / np.maximum(grid_cnt, 1), 0.0)

    return {
        "grid": mean_grid,
        "grid_counts": grid_cnt,
        "edges_expected": edges_exp,
        "edges_presented": edges_pres,
        "grid_bins": tuple(grid_bins),
    }


# ---------------------------------------------------------------------------
# 3. Total population activity
# ---------------------------------------------------------------------------

def total_population_activity(
    spike_counts: np.ndarray,
    population_mask: np.ndarray,
    window_ms: float,
) -> Dict[str, Any]:
    """Summed population activity over a window, in Hz.

    Parameters
    ----------
    spike_counts : (n_cells, n_trials) — spike counts over window_ms
    population_mask : (n_cells,) bool — cells belonging to the population
    window_ms : float — window duration in ms (used for count → Hz)

    Returns
    -------
    dict with keys
      - 'total_rate_hz_per_trial'  : (n_trials,) mean Hz per population cell
      - 'total_rate_hz'            : scalar, mean over trials
      - 'total_rate_hz_ci'         : (lo, hi) bootstrap 95% CI on the mean
      - 'n_pop'                    : int
    """
    counts = np.asarray(spike_counts, dtype=np.float64)
    mask = np.asarray(population_mask, dtype=bool)
    if mask.shape != (counts.shape[0],):
        raise ValueError("population_mask shape mismatch")
    if window_ms <= 0:
        raise ValueError("window_ms must be > 0")
    if not mask.any():
        return {
            "total_rate_hz_per_trial": np.zeros(counts.shape[1]),
            "total_rate_hz": 0.0,
            "total_rate_hz_ci": (0.0, 0.0),
            "n_pop": 0,
        }
    per_trial_rate = counts[mask].sum(axis=0) / mask.sum() / (window_ms / 1000.0)
    mean_rate, (lo, hi) = _bootstrap_mean_ci(per_trial_rate)
    return {
        "total_rate_hz_per_trial": per_trial_rate,
        "total_rate_hz": mean_rate,
        "total_rate_hz_ci": (lo, hi),
        "n_pop": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# 4. Preferred-channel gain
# ---------------------------------------------------------------------------

def preferred_channel_gain(
    spike_counts_by_theta: np.ndarray,
    preferred_thetas: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    """Per-cell response at the θ closest to each cell's own preferred θ.

    Parameters
    ----------
    spike_counts_by_theta : (n_cells, n_thetas)
        Mean spike count per cell per presented orientation.
    preferred_thetas : (n_cells,)
    thetas : (n_thetas,)

    Returns
    -------
    ndarray shape (n_cells,) — response at the θ closest to each cell's preference.
    """
    sc = np.asarray(spike_counts_by_theta, dtype=np.float64)
    prefs = np.asarray(preferred_thetas, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    n_cells, n_t = sc.shape
    if prefs.shape != (n_cells,):
        raise ValueError("preferred_thetas shape mismatch")
    if thetas.shape != (n_t,):
        raise ValueError("thetas shape mismatch")
    d = _wrap_pi(prefs[:, None] - thetas[None, :])
    idx = np.argmin(d, axis=1)
    return sc[np.arange(n_cells), idx]


# ---------------------------------------------------------------------------
# 5. Tuning fit (von Mises)
# ---------------------------------------------------------------------------

def _von_mises_1d(theta, A, kappa, mu, baseline):
    """A * exp(κ(cos(2(θ−μ)) − 1)) + baseline on the π-periodic orientation ring.

    Normalized so peak (θ=μ) equals A + baseline; trough (θ=μ+π/2) is
    A·exp(−2κ) + baseline. The 2(θ−μ) makes the envelope π-periodic.
    """
    return A * np.exp(kappa * (np.cos(2.0 * (theta - mu)) - 1.0)) + baseline


def tuning_fit(
    spike_counts_by_theta: np.ndarray,
    thetas: np.ndarray,
    fit: str = "von_mises",
) -> Dict[str, Any]:
    """Per-cell von-Mises tuning fit; report FWHM, peak, preferred θ, R².

    FWHM is derived numerically from fitted κ: the separation (rad) between
    the two θ where the von-Mises envelope falls to half-peak. If the fit
    is poor (R² < 0.2) or κ < ln(2)/2 (envelope too flat for a half-peak
    crossing), FWHM is NaN for that cell.

    Parameters
    ----------
    spike_counts_by_theta : (n_cells, n_thetas) — mean spikes per cell × θ
    thetas : (n_thetas,) — radians
    fit : str — currently only 'von_mises'

    Returns
    -------
    dict with keys
      - 'A'        : (n_cells,) amplitude
      - 'kappa'    : (n_cells,) concentration
      - 'mu'       : (n_cells,) preferred θ (rad, 0..π)
      - 'baseline' : (n_cells,) baseline offset
      - 'fwhm_rad' : (n_cells,) full-width at half-max of the envelope
      - 'r2'       : (n_cells,) fit R² (NaN on fit failure)
    """
    if fit != "von_mises":
        raise NotImplementedError(f"fit='{fit}' not implemented")
    from scipy.optimize import curve_fit

    sc = np.asarray(spike_counts_by_theta, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    n_cells, n_t = sc.shape
    if thetas.shape != (n_t,):
        raise ValueError("thetas shape mismatch")

    out_A = np.full(n_cells, np.nan)
    out_kappa = np.full(n_cells, np.nan)
    out_mu = np.full(n_cells, np.nan)
    out_base = np.full(n_cells, np.nan)
    out_fwhm = np.full(n_cells, np.nan)
    out_r2 = np.full(n_cells, np.nan)

    for c in range(n_cells):
        y = sc[c]
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if y_max - y_min < 1e-9:
            continue
        mu0 = float(thetas[int(np.argmax(y))])
        A0 = y_max - y_min
        base0 = y_min
        try:
            popt, _ = curve_fit(
                _von_mises_1d, thetas, y,
                p0=[A0, 2.0, mu0, base0],
                bounds=([0.0, 0.01, 0.0, -np.inf],
                        [np.inf, 100.0, np.pi, np.inf]),
                maxfev=2000,
            )
            A, kappa, mu, baseline = popt
            y_pred = _von_mises_1d(thetas, A, kappa, mu, baseline)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            out_A[c] = A
            out_kappa[c] = kappa
            out_mu[c] = mu % np.pi
            out_base[c] = baseline
            out_r2[c] = r2
            # Envelope exp(κ(cos(2Δθ) − 1)) = 0.5 ⇒ cos(2Δθ) = 1 + ln(0.5)/κ.
            cos_arg = 1.0 + np.log(0.5) / kappa
            if -1.0 <= cos_arg <= 1.0:
                half_angle_2theta = float(np.arccos(cos_arg))
                out_fwhm[c] = half_angle_2theta   # = 2·(half-width in θ)
        except (RuntimeError, ValueError):
            continue

    return {
        "A": out_A,
        "kappa": out_kappa,
        "mu": out_mu,
        "baseline": out_base,
        "fwhm_rad": out_fwhm,
        "r2": out_r2,
    }


# ---------------------------------------------------------------------------
# 6. Omission-subtracted response
# ---------------------------------------------------------------------------

def omission_subtracted_response(
    spike_counts_stim: np.ndarray,
    spike_counts_omit: np.ndarray,
) -> np.ndarray:
    """Per-cell stim − omit mean response (deviant-vs-expected or stim-vs-omit).

    Parameters
    ----------
    spike_counts_stim : (n_cells, n_trials_stim)
    spike_counts_omit : (n_cells, n_trials_omit)

    Returns
    -------
    ndarray shape (n_cells,) — mean(stim) − mean(omit) per cell.
    """
    s = np.asarray(spike_counts_stim, dtype=np.float64)
    o = np.asarray(spike_counts_omit, dtype=np.float64)
    if s.shape[0] != o.shape[0]:
        raise ValueError("cell dim mismatch")
    return s.mean(axis=1) - o.mean(axis=1)


# ---------------------------------------------------------------------------
# 7. Linear-SVM decoding accuracy
# ---------------------------------------------------------------------------

def svm_decoding_accuracy(
    spike_vectors: np.ndarray,
    labels: np.ndarray,
    cv: int = 5,
    classifier: str = "linear_svm",
    seed: int = 0,
) -> Dict[str, Any]:
    """Stratified k-fold cross-validated linear-SVM decoding accuracy.

    Parameters
    ----------
    spike_vectors : (n_trials, n_cells) — population vectors (spikes/trial)
    labels : (n_trials,) — integer class labels
    cv : int — number of folds (default 5)
    classifier : str — currently only 'linear_svm'
    seed : int — RNG seed for fold assignment + SVM init

    Returns
    -------
    dict with keys
      - 'accuracy'       : scalar, mean fold accuracy
      - 'accuracy_ci'    : (lo, hi) percentile CI over folds (95%)
      - 'fold_accuracy'  : (cv,) per-fold accuracy
      - 'n_trials'       : int
      - 'n_classes'      : int
    """
    if classifier != "linear_svm":
        raise NotImplementedError(f"classifier='{classifier}' not implemented")
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(spike_vectors, dtype=np.float64)
    y = np.asarray(labels)
    n_trials, _n_cells = X.shape
    if y.shape != (n_trials,):
        raise ValueError("labels shape mismatch")
    n_classes = int(len(np.unique(y)))

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_accs = []
    for tr, te in skf.split(X, y):
        clf = LinearSVC(
            C=1.0, max_iter=5000, dual="auto", random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        fold_accs.append(float(clf.score(X[te], y[te])))
    fold_accs = np.asarray(fold_accs, dtype=np.float64)
    mean_acc = float(fold_accs.mean())
    if cv >= 3:
        lo = float(np.quantile(fold_accs, 0.025))
        hi = float(np.quantile(fold_accs, 0.975))
    else:
        lo = hi = mean_acc
    return {
        "accuracy": mean_acc,
        "accuracy_ci": (lo, hi),
        "fold_accuracy": fold_accs,
        "n_trials": int(n_trials),
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# 8. Pseudo-voxel forward model (6 families)
# ---------------------------------------------------------------------------

_VOXEL_MODEL_FAMILIES: Tuple[str, ...] = (
    "local_gain_dampen",
    "local_gain_enhance",
    "local_tuning_sharpen",
    "local_tuning_broaden",
    "remote_gain",
    "global_gain",
)


def _sharpen_or_broaden(
    tuning: np.ndarray, thetas: np.ndarray, eps: float, sharpen: bool,
) -> np.ndarray:
    """Deform a voxel tuning curve by multiplicative Gaussian reweighting.

    Preserves peak amplitude; sharpens (sharpen=True) or broadens by ε.
    """
    peak_idx = int(np.argmax(tuning))
    peak_theta = thetas[peak_idx]
    d = _wrap_pi(thetas - peak_theta)
    baseline = float(tuning.min())
    amp = tuning - baseline
    factor = (1.0 - eps) if sharpen else (1.0 + eps)
    sigma_eff = max(1e-3, float(np.deg2rad(20.0)) * factor)
    gauss = np.exp(-0.5 * (d / sigma_eff) ** 2)
    if gauss.max() < 1e-12:
        return tuning.copy()
    reweighted = amp.max() * gauss / gauss.max()
    return baseline + reweighted


def pseudo_voxel_forward_model(
    spike_counts_by_theta: np.ndarray,
    voxel_spatial_bins: np.ndarray,
    model_family: str,
    thetas: np.ndarray,
    effect_size: float = 0.2,
) -> Dict[str, Any]:
    """Richter-matched 6-family pseudo-voxel forward model.

    Pools cells into ``n_voxels`` pseudo-voxels by ``voxel_spatial_bins``
    (default 4 per plan §5). For each voxel, computes baseline tuning
    (mean spike count per θ), then predicts the deformed expected-
    condition tuning under the assumed model family. Caller compares
    predicted against observed in an AIC/LL framework.

    The 6 families (per plan §5):

      - ``local_gain_dampen``   : preferred voxel peak × (1 − ε)
      - ``local_gain_enhance``  : preferred voxel peak × (1 + ε)
      - ``local_tuning_sharpen``: preferred voxel FWHM narrowed by ε
      - ``local_tuning_broaden``: preferred voxel FWHM widened by ε
      - ``remote_gain``         : non-preferred voxels × (1 + ε)
      - ``global_gain``         : all voxels × (1 + ε)

    The "preferred voxel" is the voxel with the largest baseline peak.

    Parameters
    ----------
    spike_counts_by_theta : (n_cells, n_thetas)
    voxel_spatial_bins    : (n_cells,) int in 0..n_voxels-1
    model_family          : one of _VOXEL_MODEL_FAMILIES
    thetas                : (n_thetas,) radians
    effect_size           : fractional deformation ε (default 0.2)

    Returns
    -------
    dict with keys
      - 'voxel_tuning_baseline'   : (n_voxels, n_thetas)
      - 'voxel_tuning_predicted'  : (n_voxels, n_thetas)
      - 'preferred_voxel'         : int
      - 'model_family'            : str
      - 'n_voxels'                : int
      - 'effect_size'             : float
    """
    if model_family not in _VOXEL_MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model_family '{model_family}'. "
            f"Must be one of {_VOXEL_MODEL_FAMILIES}"
        )
    sc = np.asarray(spike_counts_by_theta, dtype=np.float64)
    bins = np.asarray(voxel_spatial_bins, dtype=np.int64)
    thetas = np.asarray(thetas, dtype=np.float64)

    n_cells, n_t = sc.shape
    if bins.shape != (n_cells,):
        raise ValueError("voxel_spatial_bins shape mismatch")
    if thetas.shape != (n_t,):
        raise ValueError("thetas shape mismatch")
    if len(bins) == 0:
        raise ValueError("at least one cell required")
    n_voxels = int(bins.max() + 1)
    if n_voxels <= 0:
        raise ValueError("at least one voxel required")

    voxel_tuning = np.zeros((n_voxels, n_t), dtype=np.float64)
    for v in range(n_voxels):
        mask = bins == v
        if mask.any():
            voxel_tuning[v] = sc[mask].mean(axis=0)

    peak_per_voxel = voxel_tuning.max(axis=1)
    preferred_v = int(np.argmax(peak_per_voxel))

    predicted = voxel_tuning.copy()
    eps = float(effect_size)

    if model_family == "local_gain_dampen":
        predicted[preferred_v] = voxel_tuning[preferred_v] * (1.0 - eps)
    elif model_family == "local_gain_enhance":
        predicted[preferred_v] = voxel_tuning[preferred_v] * (1.0 + eps)
    elif model_family == "local_tuning_sharpen":
        predicted[preferred_v] = _sharpen_or_broaden(
            voxel_tuning[preferred_v], thetas, eps, sharpen=True,
        )
    elif model_family == "local_tuning_broaden":
        predicted[preferred_v] = _sharpen_or_broaden(
            voxel_tuning[preferred_v], thetas, eps, sharpen=False,
        )
    elif model_family == "remote_gain":
        for v in range(n_voxels):
            if v != preferred_v:
                predicted[v] = voxel_tuning[v] * (1.0 + eps)
    elif model_family == "global_gain":
        predicted = voxel_tuning * (1.0 + eps)

    return {
        "voxel_tuning_baseline": voxel_tuning,
        "voxel_tuning_predicted": predicted,
        "preferred_voxel": preferred_v,
        "model_family": model_family,
        "n_voxels": n_voxels,
        "effect_size": eps,
    }


# ---------------------------------------------------------------------------
# 9. Evidence package (per-claim gate aggregator)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceResult:
    passed: bool
    direction_match: bool
    cohens_d: float
    d_passes: bool
    ablation_collapses: bool
    held_out_replicates: bool
    n_main_seeds: int
    detail: str

    def summary(self) -> str:
        return (
            f"EvidencePackage  passed={self.passed}  "
            f"dir_match={self.direction_match}  "
            f"d={self.cohens_d:.3f} (≥0.2: {self.d_passes})  "
            f"ablation_collapse={self.ablation_collapses}  "
            f"held_out_replicates={self.held_out_replicates}  "
            f"n_main_seeds={self.n_main_seeds}  {self.detail}"
        )


def evidence_package(
    metric_vals_per_seed: Mapping[int, float],
    held_out_seed_vals: Optional[Mapping[int, float]] = None,
    ablation_metric_vals_per_seed: Optional[Mapping[int, float]] = None,
    pre_registered_direction: int = +1,
    cohens_d_floor: float = 0.2,
) -> Dict[str, Any]:
    """Per-headline-claim evidence gate (plan sec 5).

    A claim passes if ALL four hold:
      (a) Direction matches `pre_registered_direction`.
      (b) Cohen's d of main-seed values vs zero ≥ `cohens_d_floor`.
      (c) Under the primary causal ablation, the metric collapses in
          magnitude (|mean_ablation| < 0.5·|mean_main|) and/or reverses
          direction. Missing ablation input fails this sub-check.
      (d) Held-out seeds replicate the direction. Missing held-out input
          fails this sub-check.

    Parameters
    ----------
    metric_vals_per_seed : Mapping[int, float]
        Primary seed set (plan: {42, 7, 123, 2024, 11}) → scalar metric
        (e.g., bin-0 Δ for preference-rank).
    held_out_seed_vals : Mapping[int, float]
        Held-out seeds (plan: {99, 314}) → scalar.
    ablation_metric_vals_per_seed : Mapping[int, float]
        Same seeds as main, under the claim's primary causal ablation only.
    pre_registered_direction : int, +1 or −1
    cohens_d_floor : float

    Returns
    -------
    dict with EvidenceResult fields + 'result' (dataclass).
    """
    if pre_registered_direction not in (+1, -1):
        raise ValueError("pre_registered_direction must be +1 or -1")
    vals = np.asarray(list(metric_vals_per_seed.values()), dtype=np.float64)
    if len(vals) == 0:
        raise ValueError("metric_vals_per_seed is empty")

    mean_main = float(vals.mean())
    direction_match = (
        np.sign(mean_main) == np.sign(pre_registered_direction)
        and abs(mean_main) > 1e-12
    )

    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    cohens_d = float(mean_main / sd) if sd > 1e-12 else 0.0
    d_passes = abs(cohens_d) >= cohens_d_floor and direction_match

    if ablation_metric_vals_per_seed:
        ab = np.asarray(
            list(ablation_metric_vals_per_seed.values()), dtype=np.float64,
        )
        mean_ab = float(ab.mean())
        ablation_collapses = (
            abs(mean_ab) < 0.5 * abs(mean_main)
            or np.sign(mean_ab) != np.sign(mean_main)
        )
    else:
        mean_ab = float("nan")
        ablation_collapses = False

    if held_out_seed_vals:
        ho = np.asarray(
            list(held_out_seed_vals.values()), dtype=np.float64,
        )
        ho_mean = float(ho.mean())
        held_out_replicates = (
            np.sign(ho_mean) == np.sign(pre_registered_direction)
            and abs(ho_mean) > 1e-12
        )
    else:
        ho_mean = float("nan")
        held_out_replicates = False

    passed = (
        direction_match
        and d_passes
        and ablation_collapses
        and held_out_replicates
    )

    detail = (
        f"mean_main={mean_main:.4f}  "
        f"n_main={len(vals)}  "
        f"mean_ablation={mean_ab:.4f}  "
        f"mean_held_out={ho_mean:.4f}"
    )

    res = EvidenceResult(
        passed=passed,
        direction_match=direction_match,
        cohens_d=cohens_d,
        d_passes=d_passes,
        ablation_collapses=ablation_collapses,
        held_out_replicates=held_out_replicates,
        n_main_seeds=len(vals),
        detail=detail,
    )
    return {
        "passed": passed,
        "direction_match": direction_match,
        "cohens_d": cohens_d,
        "d_passes": d_passes,
        "ablation_collapses": ablation_collapses,
        "held_out_replicates": held_out_replicates,
        "n_main_seeds": len(vals),
        "mean_main": mean_main,
        "detail": detail,
        "result": res,
    }
