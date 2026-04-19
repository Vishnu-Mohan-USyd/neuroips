"""Shape / sign / monotonicity invariants for `current_weight_shrinkage`.

Closed form: `Δw_ij = −β · mean_b(a_pre_i²) · w_ij`.

  * Output shape matches weights.
  * Zero pre-activity ⇒ zero shrinkage.
  * Higher pre-activity magnitude ⇒ larger shrinkage magnitude.
  * Output sign is opposite to weight sign (shrinkage pulls toward zero).
  * β = 0 turns the penalty off entirely.
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
    """Doubling |a_pre| quadruples |ΔW| (square law)."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.ones(3, 2)
    pre_small = torch.ones(2, 2)
    pre_big = torch.full((2, 2), 2.0)
    dw_small = energy.current_weight_shrinkage(weights, pre_small)
    dw_big = energy.current_weight_shrinkage(weights, pre_big)
    # Because the rule uses squared pre-activity, doubling gives 4×.
    torch.testing.assert_close(dw_big, 4.0 * dw_small, atol=1e-7, rtol=0.0)


def test_analytic_formula_spot_check() -> None:
    """Hand-derived scalar check of −β · mean_b(a_pre²) · w."""
    beta = 0.1
    energy = EnergyPenalty(alpha=0.0, beta=beta)
    weights = torch.tensor([[2.0]])                                 # [1, 1]
    pre = torch.tensor([[1.0], [3.0]])                              # [B=2, n_pre=1]
    # mean_b(a_pre²) = mean(1, 9) = 5 ; ΔW = -0.1 · 5 · 2 = -1.0
    dw = energy.current_weight_shrinkage(weights, pre)
    torch.testing.assert_close(
        dw, torch.tensor([[-1.0]]), atol=1e-7, rtol=0.0
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
