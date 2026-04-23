"""Known-answer tests for :mod:`scripts.v2.toy_alpha_subtractive`.

The integration is ``r_j = softplus(basal_j − apical_j)`` where
``softplus(x) = log(1 + e^x)`` is smooth with a non-negative floor:

  * ``softplus(0)     = log 2 ≈ 0.6931``
  * ``softplus(+1)    ≈ 1.3133``  (``log(1 + e)``)
  * ``softplus(-1)    ≈ 0.3133``  (``log(1 + e^{-1})``)

All α coefficients are baked into the pattern functions so the
integrator is parameter-free.

Tested contracts (all directly from the Gaussian-tuning + softplus
algebra, not from numerical experiment):

1. Pattern A at α=1, cue=probe: basal ≡ apical, so ``r_j ≡
   softplus(0) = log 2`` on every unit (uniform response — peak is
   algebraically cancelled).
2. Pattern B at α=1, cue=probe: at pref-of-cue ``basal ≈ 1``,
   ``apical ≈ 0``, ``r ≈ softplus(1) ≈ 1.313``; at nonpref-of-cue
   ``basal ≈ 0``, ``apical ≈ 1``, ``r ≈ softplus(-1) ≈ 0.313``.
3. Pattern C at α₀=α₁=0: apical is identically 0 so ``r ≡
   softplus(basal)`` — the no-modulation baseline.
4. α=0 (Pattern A and B): apical ≡ 0, ``r ≡ softplus(basal)``.
"""
from __future__ import annotations

import math

import numpy as np

from scripts.v2.toy_alpha_subtractive import (
    _pref_mask, apical_pattern_A, apical_pattern_B, apical_pattern_C,
    basal_drive, build_preferred_orientations, integrate,
)


LOG2 = math.log(2.0)
SOFTPLUS_PLUS1 = math.log1p(math.e)          # softplus(1) = log(1+e)
SOFTPLUS_MINUS1 = math.log1p(1.0 / math.e)   # softplus(-1) = log(1+e^-1)


def test_pattern_A_matched_uniformalises_response_at_alpha_one():
    """α=1, Pattern A, cue=probe: basal ≡ apical → r ≡ log 2 everywhere."""
    pref_deg = build_preferred_orientations(32)
    sigma = 15.0
    probe = 45.0
    basal = basal_drive(probe, pref_deg, sigma)
    apical = apical_pattern_A(probe, pref_deg, sigma, alpha=1.0)
    # Algebraically basal − apical = 0 under Pattern A with α=1 and
    # cue=probe, so softplus(0) = log 2 on every unit.
    assert np.allclose(basal, apical, atol=0.0, rtol=0.0), (
        "precondition: basal and α=1 Pattern A apical must be bit-equal "
        "when cue=probe"
    )
    r = integrate(basal, apical)
    assert np.allclose(r, LOG2, atol=1e-12), (
        f"Pattern A matched at α=1 should give r=log 2 everywhere, "
        f"got max|Δ|={float(np.max(np.abs(r - LOG2))):.2e}"
    )


def test_pattern_B_matched_alpha_one_known_endpoints():
    """α=1, Pattern B, cue=probe: pref ≈ softplus(1); nonpref ≈ softplus(-1)."""
    pref_deg = build_preferred_orientations(32)
    sigma = 15.0
    probe = 45.0
    basal = basal_drive(probe, pref_deg, sigma)
    apical = apical_pattern_B(probe, pref_deg, sigma, alpha=1.0)
    r = integrate(basal, apical)

    # Unit closest to cue: basal ≈ 1, apical ≈ 0, so r ≈ softplus(1).
    d_cue = np.abs(pref_deg - probe)
    d_cue = np.minimum(d_cue, 180.0 - d_cue)
    j_pref = int(np.argmin(d_cue))
    basal_star = float(basal[j_pref])
    apical_star = float(apical[j_pref])
    expected_pref = math.log1p(math.exp(basal_star - apical_star))
    assert abs(float(r[j_pref]) - expected_pref) < 1e-10, (
        f"pref-of-cue softplus mismatch: got {float(r[j_pref]):.6f}, "
        f"expected {expected_pref:.6f}"
    )
    # And the value is comfortably above log 2 (since basal - apical > 0
    # at pref-of-cue when basal is near 1).
    assert float(r[j_pref]) > LOG2 + 0.3, (
        f"pref-of-cue response {float(r[j_pref]):.4f} should be well "
        f"above log 2 = {LOG2:.4f} under Pattern B matched at α=1"
    )

    # Unit closest to nonpref-of-cue: basal ≈ 0, apical ≈ 1, so
    # r ≈ softplus(-1) ≈ 0.313.
    nonpref_center = (probe + 90.0) % 180.0
    d_np = np.abs(pref_deg - nonpref_center)
    d_np = np.minimum(d_np, 180.0 - d_np)
    j_np = int(np.argmin(d_np))
    basal_np = float(basal[j_np])
    apical_np = float(apical[j_np])
    expected_np = math.log1p(math.exp(basal_np - apical_np))
    assert abs(float(r[j_np]) - expected_np) < 1e-10
    # And the value is comfortably below log 2.
    assert float(r[j_np]) < LOG2 - 0.3, (
        f"nonpref-of-cue response {float(r[j_np]):.4f} should be well "
        f"below log 2 = {LOG2:.4f} under Pattern B matched at α=1"
    )
    # Additional diagnostic assertion: the numerical endpoints should
    # be close to softplus(±1) at the exact grid-nearest unit.
    assert abs(float(r[j_pref]) - SOFTPLUS_PLUS1) < 0.05, (
        f"pref-of-cue endpoint drifts from softplus(1)={SOFTPLUS_PLUS1:.4f} "
        f"by more than 0.05 at the closest grid unit"
    )
    assert abs(float(r[j_np]) - SOFTPLUS_MINUS1) < 0.05, (
        f"nonpref-of-cue endpoint drifts from softplus(-1)="
        f"{SOFTPLUS_MINUS1:.4f} by more than 0.05 at the closest grid unit"
    )


def test_pattern_C_zero_coeffs_is_softplus_of_basal():
    """Pattern C at α₀=α₁=0: apical ≡ 0 → r ≡ softplus(basal)."""
    pref_deg = build_preferred_orientations(32)
    sigma = 15.0
    probe = 45.0
    basal = basal_drive(probe, pref_deg, sigma)
    apical = apical_pattern_C(
        probe, pref_deg, sigma, alpha0=0.0, alpha1=0.0,
    )
    assert np.allclose(apical, 0.0, atol=0.0), "Pattern C α=0 must be zero"
    r = integrate(basal, apical)
    expected = np.log1p(np.exp(basal))
    assert np.allclose(r, expected, atol=1e-12), (
        "Pattern C with zero coefficients must equal softplus(basal)"
    )


def test_alpha_zero_is_identity_pass_through_on_A_and_B():
    """α=0 Pattern A or B: apical ≡ 0 → r ≡ softplus(basal)."""
    pref_deg = build_preferred_orientations(32)
    sigma = 15.0
    probe = 45.0
    basal = basal_drive(probe, pref_deg, sigma)
    expected = np.log1p(np.exp(basal))
    for cue in (probe, (probe + 90.0) % 180.0):
        for apical_fn in (apical_pattern_A, apical_pattern_B):
            apical = apical_fn(cue, pref_deg, sigma, alpha=0.0)
            assert np.allclose(apical, 0.0, atol=0.0)
            r = integrate(basal, apical)
            assert np.allclose(r, expected, atol=1e-12), (
                f"α=0 {apical_fn.__name__} cue={cue} not a pass-through "
                f"of softplus(basal); max|Δ|="
                f"{float(np.max(np.abs(r - expected))):.2e}"
            )


def test_pattern_C_uniform_floor_lowers_total_firing():
    """α₀>0 uniform floor pulls softplus(basal − α₀) < softplus(basal)."""
    pref_deg = build_preferred_orientations(32)
    sigma = 15.0
    probe = 45.0
    basal = basal_drive(probe, pref_deg, sigma)
    r_base = integrate(basal, np.zeros_like(basal))
    apical = apical_pattern_C(
        probe, pref_deg, sigma, alpha0=0.5, alpha1=0.0,
    )
    assert np.allclose(apical, 0.5, atol=0.0), (
        "Pattern C with α₁=0 and α₀=0.5 must be constant 0.5"
    )
    r_floor = integrate(basal, apical)
    # softplus is strictly increasing → subtracting 0.5 strictly lowers
    # every unit's response.
    assert np.all(r_floor < r_base), (
        "uniform α₀=0.5 did not lower every unit's response"
    )
    # Mean must drop meaningfully (softplus is Lipschitz-1, so the
    # drop is at most 0.5 per unit; at least some fraction of 0.5).
    assert (r_base.mean() - r_floor.mean()) > 0.2, (
        f"mean response drop {r_base.mean() - r_floor.mean():.4f} "
        f"smaller than expected under α₀=0.5 uniform floor"
    )
