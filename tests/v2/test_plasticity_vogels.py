"""Vogels iSTDP homeostatic inhibitory plasticity.

Verifies:
  * post > ρ_target ⇒ ΔW > 0 (raw-weight growth ⇒ stronger inhibition);
  * post < ρ_target ⇒ ΔW < 0;
  * post == ρ_target ⇒ Hebbian term exactly zero;
  * weight-decay term acts correctly;
  * mask preservation;
  * input / init validation.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.plasticity import VogelsISTDPRule


# ---------------------------------------------------------------------------
# Sign behaviour
# ---------------------------------------------------------------------------

def test_post_above_target_grows_raw_weight() -> None:
    """a_post > ρ + pre > 0 + decay=0 ⇒ ΔW > 0."""
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.0)
    pre = torch.ones(2, 3)
    post = torch.full((2, 4), 2.0)                         # > target
    weights = torch.zeros(4, 3)
    dw = rule.delta(pre, post, weights)
    assert (dw > 0).all()


def test_post_below_target_shrinks_raw_weight() -> None:
    """a_post < ρ + pre > 0 + decay=0 ⇒ ΔW < 0."""
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.0)
    pre = torch.ones(2, 3)
    post = torch.full((2, 4), 0.5)                         # < target
    weights = torch.zeros(4, 3)
    dw = rule.delta(pre, post, weights)
    assert (dw < 0).all()


def test_post_equal_target_hebbian_zero() -> None:
    """a_post == ρ + decay=0 ⇒ ΔW == 0 exactly."""
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.0)
    pre = torch.rand(2, 3)
    post = torch.ones(2, 4)                                # == target
    weights = torch.rand(4, 3)
    dw = rule.delta(pre, post, weights)
    assert torch.all(dw == 0.0)


def test_hebbian_magnitude_matches_analytic_formula() -> None:
    """Scalar check against hand-derived value.

    lr kept small so analytic values stay inside the ±0.01 per-step clamp
    (Task #62) and the formula is verified unclipped.
    """
    rule = VogelsISTDPRule(lr=0.001, target_rate=0.5, weight_decay=0.0)
    pre = torch.tensor([[1.0, 2.0]])                       # [B=1, n_pre=2]
    post = torch.tensor([[1.0]])                           # [B=1, n_post=1]
    weights = torch.zeros(1, 2)
    # post_dev = [0.5]; hebb[0, 0] = 0.5 · 1 = 0.5; hebb[0, 1] = 0.5 · 2 = 1.0
    expected = torch.tensor([[0.001 * 0.5, 0.001 * 1.0]])
    dw = rule.delta(pre, post, weights)
    torch.testing.assert_close(dw, expected, atol=1e-6, rtol=0.0)


def test_delta_clamps_large_updates() -> None:
    """Per-step Δw magnitude is clamped to [-0.01, 0.01] (Task #62)."""
    rule = VogelsISTDPRule(lr=1.0, target_rate=0.0, weight_decay=0.0)
    pre = torch.tensor([[5.0, 5.0]])
    post = torch.tensor([[5.0]])
    weights = torch.zeros(1, 2)
    dw = rule.delta(pre, post, weights)
    assert float(dw.abs().max().item()) <= 0.01 + 1e-8


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------

def test_weight_decay_shrinks_existing_weights() -> None:
    """weight_decay kept small so the expected analytic value stays inside
    the ±0.01 per-step clamp (Task #62) and the decay arithmetic is verified
    unclipped."""
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.005)
    pre = torch.zeros(2, 3)
    post = torch.ones(2, 4)                                # post == target, Hebbian=0
    weights = torch.ones(4, 3)
    dw = rule.delta(pre, post, weights)
    torch.testing.assert_close(
        dw, torch.full_like(weights, -0.005), atol=1e-6, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Mask preservation (sanity)
# ---------------------------------------------------------------------------

def test_mask_zeros_disallowed_entries() -> None:
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.01)
    pre = torch.rand(2, 3)
    post = torch.rand(2, 4) + 0.5
    weights = torch.rand(4, 3)
    mask = torch.rand(4, 3) > 0.5
    dw = rule.delta(pre, post, weights, mask=mask)
    assert torch.all(dw[~mask] == 0.0)


# ---------------------------------------------------------------------------
# Input / init validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_lr", [0.0, -0.01])
def test_rejects_non_positive_lr(bad_lr: float) -> None:
    with pytest.raises(ValueError, match="lr"):
        VogelsISTDPRule(lr=bad_lr, target_rate=1.0)


def test_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=-0.1)


def test_rejects_negative_target_rate() -> None:
    with pytest.raises(ValueError, match="target_rate"):
        VogelsISTDPRule(lr=0.1, target_rate=-0.5)


def test_rejects_wrong_weights_shape() -> None:
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0)
    pre = torch.rand(2, 3)
    post = torch.rand(2, 4)
    weights = torch.zeros(3, 4)                            # transposed
    with pytest.raises(ValueError, match="weights"):
        rule.delta(pre, post, weights)


def test_rejects_mask_wrong_shape() -> None:
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0)
    pre = torch.rand(2, 3)
    post = torch.rand(2, 4)
    weights = torch.zeros(4, 3)
    bad_mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        rule.delta(pre, post, weights, mask=bad_mask)
