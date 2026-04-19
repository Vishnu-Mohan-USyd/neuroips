"""LOAD-BEARING for Phase 2 Gate 7: the hidden regime ``g`` must be
**invisible in a single frame** and **recoverable from history only**.

If a linear probe on a single frame's raw pixels could predict the regime,
the network would get g for free and context memory C would be redundant.
If rolling history statistics couldn't predict g either, the task would be
degenerate. We require both:

* Linear regression of the regime one-hot onto flattened pixels has
  **R² < 0.1** on held-out data.
* A 3-feature descriptor of the last 50 frames (mean Δθ, std Δθ, jump
  fraction) classifies regime **substantially above chance** (≥ 0.40 on
  4-way LinearSVC, chance = 0.25) — strictly higher than pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world import REGIMES, ProceduralWorld

sklearn = pytest.importorskip("sklearn")                    # noqa: F841


def _windowed_mean_std(arr: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean + std of ``arr`` over a window of ``w``."""
    n = arr.shape[0]
    out_m = np.zeros(n, dtype=np.float32)
    out_s = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - w + 1)
        s = arr[lo:i + 1]
        out_m[i] = float(s.mean())
        out_s[i] = float(s.std()) if s.shape[0] > 1 else 0.0
    return out_m, out_s


def _assemble_history_features(
    states: list, n_states: int, window: int,
) -> np.ndarray:
    """Build a [n_states, 3] rolling-window descriptor from ``states``."""
    thetas = np.array([s.theta for s in states], dtype=np.float32)
    zs = np.array([s.z for s in states], dtype=np.int32)
    dtheta = np.diff(thetas, prepend=thetas[0])
    dtheta = ((dtheta + 180.0) % 360.0 - 180.0).astype(np.float32)
    dz = np.diff(zs, prepend=zs[0])
    jumps = (dz != 0).astype(np.float32)
    m_dt, s_dt = _windowed_mean_std(dtheta, window)
    m_j, _ = _windowed_mean_std(jumps, window)
    return np.stack([m_dt, s_dt, m_j], axis=1)


def _r2_multivariate(
    X: np.ndarray, Y: np.ndarray, split: int,
) -> float:
    """Held-out R² of a multi-output OLS fit ``Y ≈ β X``."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X[:split], Y[:split])
    pred = model.predict(X[split:])
    ss_res = ((Y[split:] - pred) ** 2).sum()
    ss_tot = ((Y[split:] - Y[split:].mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def test_regime_invisible_to_single_frame_linear_probe(cfg) -> None:
    """Pixel → regime-one-hot R² on held-out data is below 0.1."""
    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, bank, seed_family="train")
    frames, states = world.trajectory(0, n_steps=8000)
    regimes = np.array([REGIMES.index(s.regime) for s in states])
    onehot = np.eye(4, dtype=np.float32)[regimes]

    # Warm-up: drop first 50 frames so the windowed comparison below has a
    # fair pair. (This test doesn't depend on the warmup, just keeps the
    # two tests internally consistent.)
    X_pix = frames[50:].view(frames.shape[0] - 50, -1).numpy()
    Y = onehot[50:]
    r2 = _r2_multivariate(X_pix, Y, split=X_pix.shape[0] // 2)
    assert r2 < 0.1, (
        f"single-frame pixel → regime R²={r2:.4f} — regime leaks into "
        f"individual frames (spec requires < 0.1)"
    )


def test_history_features_above_pixels(cfg) -> None:
    """Rolling-window θ-drift + jump-rate history beats pixel-only
    classification by a clear margin on 4-way regime classification."""
    from sklearn.svm import LinearSVC

    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, bank, seed_family="train")
    frames, states = world.trajectory(0, n_steps=8000)
    regimes = np.array([REGIMES.index(s.regime) for s in states])

    window = 50
    feats_hist = _assemble_history_features(states, len(states), window)

    # Drop warm-up
    X_pix = frames[window:].view(frames.shape[0] - window, -1).numpy()
    X_hist = feats_hist[window:]
    y = regimes[window:]
    split = y.shape[0] // 2

    clf_pix = LinearSVC(max_iter=5000, dual="auto", random_state=0)
    clf_pix.fit(X_pix[:split], y[:split])
    acc_pix = float(clf_pix.score(X_pix[split:], y[split:]))

    clf_hist = LinearSVC(max_iter=5000, dual="auto", random_state=0)
    clf_hist.fit(X_hist[:split], y[:split])
    acc_hist = float(clf_hist.score(X_hist[split:], y[split:]))

    assert acc_pix < 0.35, (
        f"pixel-only regime classifier acc {acc_pix:.4f} — regime leaks "
        f"into single-frame pixels (spec: near 0.25 chance)"
    )
    assert acc_hist >= 0.40, (
        f"history-feature regime classifier acc {acc_hist:.4f} — rolling "
        f"Δθ + jump-rate features should beat chance decisively"
    )
    assert acc_hist > acc_pix + 0.10, (
        f"history features ({acc_hist:.4f}) should beat pixels "
        f"({acc_pix:.4f}) by a clear margin"
    )


def test_windowed_features_prefer_drift_regimes(cfg) -> None:
    """Circular sanity check: the 50-frame mean(Δθ) cleanly separates CW
    from CCW drift. The window averages over ≈50 steps of which ~36 are
    in-regime (regime transition rate ~2 %/step), so the expected mean is
    attenuated below the nominal ±5°; we require only that the sign and
    ordering come through cleanly."""
    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, bank, seed_family="train")
    _, states = world.trajectory(0, n_steps=6000)
    regimes = np.array([REGIMES.index(s.regime) for s in states])
    feats = _assemble_history_features(states, len(states), window=50)
    mean_dt = feats[:, 0]

    cw_mean = float(mean_dt[regimes == REGIMES.index("CW-drift")].mean())
    ccw_mean = float(mean_dt[regimes == REGIMES.index("CCW-drift")].mean())
    low_mean = float(mean_dt[regimes == REGIMES.index("low-hazard")].mean())
    assert cw_mean > 1.0, (
        f"windowed mean(Δθ) for CW-drift is {cw_mean:.4f} — expected ≳ +1"
    )
    assert ccw_mean < -1.0, (
        f"windowed mean(Δθ) for CCW-drift is {ccw_mean:.4f} — expected ≲ -1"
    )
    # Hazard regime is zero-mean; it lies between CW and CCW.
    assert ccw_mean < low_mean < cw_mean, (
        f"ordering CCW ({ccw_mean:.4f}) < low-hazard ({low_mean:.4f}) < "
        f"CW ({cw_mean:.4f}) violated"
    )
