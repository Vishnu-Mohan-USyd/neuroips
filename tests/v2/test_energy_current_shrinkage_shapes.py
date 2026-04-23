"""Shape / sign / monotonicity invariants for `current_weight_shrinkage`.

Implicit-Euler form (Task #62): `Δw_ij = −w_ij · s_i / (1 + s_i)` with
`s_i = β · mean_b(a_pre_i²)`. This guarantees `|Δw| ≤ |w|` regardless of
pre-activity magnitude, replacing the explicit-Euler form that overshot
for large `pre²` and drove oscillatory weight explosion in Phase-2.

  * Output shape matches weights.
  * Zero pre-activity ⇒ zero shrinkage.
  * Higher pre-activity magnitude ⇒ larger shrinkage magnitude (monotonic,
    approximately square-law when `s ≪ 1`).
  * Output sign is opposite to weight sign (shrinkage pulls toward zero).
  * β = 0 turns the penalty off entirely.
  * Bounded: `|Δw| ≤ |w|` for all finite pre.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.energy import EnergyPenalty


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------

def test_output_shape_matches_weights() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(8, 6)
    pre = torch.randn(4, 6)
    dw = energy.current_weight_shrinkage(weights, pre)
    assert dw.shape == weights.shape


def test_output_dtype_matches_weights() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(4, 3, dtype=torch.float64)
    pre = torch.randn(2, 3, dtype=torch.float64)
    dw = energy.current_weight_shrinkage(weights, pre)
    assert dw.dtype == torch.float64


# ---------------------------------------------------------------------------
# Zero-input invariants
# ---------------------------------------------------------------------------

def test_zero_pre_activity_gives_zero_shrinkage() -> None:
    """a_pre = 0 ⇒ mean(a_pre²) = 0 ⇒ ΔW = 0 for all weights."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(4, 3)
    pre = torch.zeros(2, 3)
    dw = energy.current_weight_shrinkage(weights, pre)
    assert torch.all(dw == 0.0)


def test_zero_weights_give_zero_shrinkage() -> None:
    """w = 0 ⇒ ΔW = 0 regardless of pre-activity."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.zeros(4, 3)
    pre = torch.rand(2, 3) + 0.1
    dw = energy.current_weight_shrinkage(weights, pre)
    assert torch.all(dw == 0.0)


def test_beta_zero_gives_zero_shrinkage() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=0.0)
    weights = torch.randn(4, 3)
    pre = torch.rand(2, 3) + 0.1
    dw = energy.current_weight_shrinkage(weights, pre)
    assert torch.all(dw == 0.0)


# ---------------------------------------------------------------------------
# Sign / monotonicity
# ---------------------------------------------------------------------------

def test_shrinkage_sign_opposite_weight_sign_when_pre_active() -> None:
    """Shrinkage pulls weight toward zero: sign(ΔW) = −sign(W) (for a_pre > 0, β > 0)."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    pre = torch.ones(2, 2)                                          # positive
    dw = energy.current_weight_shrinkage(weights, pre)
    # Where w > 0, ΔW < 0; where w < 0, ΔW > 0.
    assert (dw[weights > 0] < 0).all()
    assert (dw[weights < 0] > 0).all()


def test_higher_pre_magnitude_gives_larger_shrinkage_magnitude() -> None:
    """Doubling |a_pre| increases |ΔW| monotonically; in the `s ≪ 1` regime
    the ratio approaches the square-law 4× limit. Under the implicit-Euler
    form (Task #62), the ratio is exactly 4·(1+s_small)/(1+s_big) < 4."""
    beta = 1e-3
    energy = EnergyPenalty(alpha=0.0, beta=beta)
    weights = torch.ones(3, 2)
    pre_small = torch.ones(2, 2)
    pre_big = torch.full((2, 2), 2.0)
    dw_small = energy.current_weight_shrinkage(weights, pre_small)
    dw_big = energy.current_weight_shrinkage(weights, pre_big)
    # Monotone: doubling pre grows |dw|.
    assert torch.all(dw_big.abs() > dw_small.abs())
    # Exact implicit-Euler ratio: s_small = β·1, s_big = β·4.
    s_small = beta * 1.0
    s_big = beta * 4.0
    expected_ratio = (s_big / (1.0 + s_big)) / (s_small / (1.0 + s_small))
    ratio = (dw_big.abs() / dw_small.abs()).mean().item()
    assert abs(ratio - expected_ratio) < 1e-5
    # Approaches 4 in the s ≪ 1 limit.
    assert 3.9 < ratio < 4.0


def test_bounded_by_weight_magnitude_for_any_pre() -> None:
    """Implicit Euler guarantees |Δw| ≤ |w| even for huge pre-activities
    (Task #62: this bound is what prevents the Phase-2 weight runaway)."""
    energy = EnergyPenalty(alpha=0.0, beta=10.0)
    weights = torch.randn(6, 5) * 3.0
    pre = torch.full((4, 5), 1e3)                                   # extreme
    dw = energy.current_weight_shrinkage(weights, pre)
    assert torch.all(dw.abs() <= weights.abs() + 1e-8)


def test_analytic_formula_spot_check() -> None:
    """Hand-derived scalar check of −w · s / (1 + s), s = β · mean_b(a_pre²)."""
    beta = 0.1
    energy = EnergyPenalty(alpha=0.0, beta=beta)
    weights = torch.tensor([[2.0]])                                 # [1, 1]
    pre = torch.tensor([[1.0], [3.0]])                              # [B=2, n_pre=1]
    # mean_b(a_pre²) = mean(1, 9) = 5 ; s = 0.1 · 5 = 0.5
    # ΔW = -2 · 0.5 / (1 + 0.5) = -2/3
    dw = energy.current_weight_shrinkage(weights, pre)
    torch.testing.assert_close(
        dw, torch.tensor([[-2.0 / 3.0]]), atol=1e-7, rtol=0.0
    )


def test_sign_of_pre_activity_does_not_matter() -> None:
    """Formula uses |a_pre|² — sign of a_pre is irrelevant."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(4, 3)
    pre = torch.randn(2, 3)
    dw_pos = energy.current_weight_shrinkage(weights, pre)
    dw_neg = energy.current_weight_shrinkage(weights, -pre)
    torch.testing.assert_close(dw_pos, dw_neg, atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_wrong_pre_activity_n_pre() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(4, 3)
    pre = torch.randn(2, 5)                                         # wrong n_pre
    with pytest.raises(ValueError, match="pre_activity"):
        energy.current_weight_shrinkage(weights, pre)


def test_rejects_1d_weights() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    with pytest.raises(ValueError, match="weights"):
        energy.current_weight_shrinkage(torch.rand(5), torch.rand(2, 5))


def test_rejects_1d_pre_activity() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    with pytest.raises(ValueError, match="pre_activity"):
        energy.current_weight_shrinkage(torch.rand(5, 3), torch.rand(3))


# ---------------------------------------------------------------------------
# Task #74 Fix P — raw_prior kwarg (2026-04-22)
# ---------------------------------------------------------------------------

def test_raw_prior_none_matches_legacy_shrink_to_zero() -> None:
    """Passing ``raw_prior=None`` (default) must yield exactly the
    pre-Fix-P behaviour: Δw = -w · s / (1 + s)."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    torch.manual_seed(0)
    weights = torch.randn(6, 4)
    pre = torch.randn(8, 4)
    dw_default = energy.current_weight_shrinkage(weights, pre)
    dw_explicit_none = energy.current_weight_shrinkage(
        weights, pre, raw_prior=None,
    )
    torch.testing.assert_close(dw_default, dw_explicit_none, atol=0.0, rtol=0.0)

    # Closed-form: for raw_prior=None, Δw = -w · s / (1 + s).
    s = energy.beta * (pre * pre).mean(dim=0)
    expected = -weights * (s / (1.0 + s)).view(1, -1)
    torch.testing.assert_close(dw_default, expected)


def test_raw_prior_at_init_is_inert() -> None:
    """Shrinkage with raw_prior equal to the weight value itself must be
    exactly zero — the weight sits at its shrinkage target."""
    energy = EnergyPenalty(alpha=0.0, beta=5e-2)
    torch.manual_seed(1)
    init_value = torch.full((5, 3), -6.5)                      # Fix-L2 style
    pre = torch.randn(10, 3).abs()                             # non-zero pre
    dw = energy.current_weight_shrinkage(
        init_value, pre, raw_prior=init_value.clone(),
    )
    torch.testing.assert_close(dw, torch.zeros_like(dw), atol=0.0, rtol=0.0)


def test_raw_prior_pulls_toward_prior_not_zero() -> None:
    """With raw_prior = -8, a weight at -8 is inert; perturb it and Fix P
    should pull it back toward -8 (not toward 0)."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-1)
    prior = torch.full((4, 2), -8.0)
    perturbed = prior + 0.5                                    # weights at -7.5
    pre = torch.ones(3, 2)                                     # s = β · 1 = 0.1
    dw = energy.current_weight_shrinkage(
        perturbed, pre, raw_prior=prior,
    )
    # Expected: Δw = -(w - prior) · s / (1 + s) = -0.5 · 0.1 / 1.1
    expected = torch.full((4, 2), -0.5 * 0.1 / 1.1)
    torch.testing.assert_close(dw, expected)
    # Post-update weight moves toward prior (−8), not toward 0.
    w_new = perturbed + dw
    assert (w_new < perturbed).all()
    assert (w_new > prior).all()


def test_raw_prior_shape_mismatch_raises() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(4, 3)
    pre = torch.randn(2, 3)
    bad_prior = torch.randn(4, 5)
    with pytest.raises(ValueError, match="raw_prior"):
        energy.current_weight_shrinkage(weights, pre, raw_prior=bad_prior)


def test_raw_prior_non_tensor_raises() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(4, 3)
    pre = torch.randn(2, 3)
    with pytest.raises(ValueError, match="raw_prior"):
        # Mypy-wise this is deliberately wrong; runtime check catches it.
        energy.current_weight_shrinkage(weights, pre, raw_prior=-6.5)  # type: ignore[arg-type]


def test_raw_prior_mask_still_zeros_masked_entries() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(4, 3)
    pre = torch.randn(2, 3)
    prior = torch.zeros_like(weights)
    mask = torch.tensor([
        [True,  False, True],
        [False, True,  False],
        [True,  True,  False],
        [False, False, True],
    ])
    dw = energy.current_weight_shrinkage(
        weights, pre, mask=mask, raw_prior=prior,
    )
    masked_off = ~mask
    assert torch.all(dw[masked_off] == 0.0)
