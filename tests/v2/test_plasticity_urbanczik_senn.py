"""Urbanczik & Senn apical-basal predictive-Hebbian rule.

Verifies:
  * sign of ΔW matches sign of ε = apical − basal (holding pre ≥ 0);
  * weight-decay term shrinks existing weights toward zero;
  * mask preservation (exact 0 on False entries);
  * zero ε (with zero decay) ⇒ zero update;
  * zero pre-activity (with zero decay) ⇒ zero update;
  * input validation rejects malformed shapes & bad init args.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.plasticity import UrbanczikSennRule


# ---------------------------------------------------------------------------
# Sign behaviour
# ---------------------------------------------------------------------------

def test_positive_epsilon_produces_positive_update() -> None:
    """ε > 0 + pre > 0 + decay=0 ⇒ ΔW > 0 entrywise."""
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.0)
    pre = torch.ones(2, 3)
    apical = torch.full((2, 4), 2.0)
    basal = torch.zeros(2, 4)
    weights = torch.zeros(4, 3)
    dw = rule.delta(pre, apical, basal, weights)
    assert (dw > 0).all()


def test_negative_epsilon_produces_negative_update() -> None:
    """ε < 0 + pre > 0 + decay=0 ⇒ ΔW < 0 entrywise."""
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.0)
    pre = torch.ones(2, 3)
    apical = torch.zeros(2, 4)
    basal = torch.full((2, 4), 2.0)
    weights = torch.zeros(4, 3)
    dw = rule.delta(pre, apical, basal, weights)
    assert (dw < 0).all()


def test_hebbian_magnitude_matches_analytic_formula() -> None:
    """Scalar check: Δw[j, i] == lr · mean_b(eps[b,j] · pre[b,i])."""
    rule = UrbanczikSennRule(lr=0.5, weight_decay=0.0)
    pre = torch.tensor([[1.0, 2.0], [3.0, 4.0]])           # [B=2, n_pre=2]
    apical = torch.tensor([[1.0], [2.0]])                  # [B=2, n_post=1]
    basal = torch.tensor([[0.5], [0.0]])
    weights = torch.zeros(1, 2)
    eps = apical - basal                                   # [[0.5], [2.0]]
    # hebb[0, 0] = mean(0.5·1, 2.0·3) = mean(0.5, 6.0) = 3.25
    # hebb[0, 1] = mean(0.5·2, 2.0·4) = mean(1.0, 8.0) = 4.5
    expected = torch.tensor([[0.5 * 3.25, 0.5 * 4.5]])
    dw = rule.delta(pre, apical, basal, weights)
    torch.testing.assert_close(dw, expected, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------

def test_weight_decay_shrinks_existing_weights() -> None:
    """With zero Hebbian drive and positive weights, decay pulls ΔW negative."""
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.05)
    pre = torch.zeros(2, 3)
    apical = torch.zeros(2, 4)
    basal = torch.zeros(2, 4)
    weights = torch.ones(4, 3)                             # positive weights
    dw = rule.delta(pre, apical, basal, weights)
    torch.testing.assert_close(
        dw, torch.full_like(weights, -0.05), atol=1e-6, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Zero-input invariants
# ---------------------------------------------------------------------------

def test_zero_epsilon_with_zero_decay_gives_zero_update() -> None:
    """apical == basal + no decay ⇒ ΔW exactly 0."""
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.0)
    pre = torch.rand(2, 3)
    apical = torch.rand(2, 4)
    basal = apical.clone()
    weights = torch.rand(4, 3)
    dw = rule.delta(pre, apical, basal, weights)
    assert torch.all(dw == 0.0)


def test_zero_pre_activity_with_zero_decay_gives_zero_update() -> None:
    """pre == 0 + no decay ⇒ ΔW exactly 0."""
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.0)
    pre = torch.zeros(2, 3)
    apical = torch.rand(2, 4)
    basal = torch.rand(2, 4)
    weights = torch.rand(4, 3)
    dw = rule.delta(pre, apical, basal, weights)
    assert torch.all(dw == 0.0)


# ---------------------------------------------------------------------------
# Mask preservation (sanity; full suite in test_plasticity_mask_preservation)
# ---------------------------------------------------------------------------

def test_mask_zeros_disallowed_entries() -> None:
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.01)
    pre = torch.rand(2, 3)
    apical = torch.rand(2, 4)
    basal = torch.rand(2, 4)
    weights = torch.rand(4, 3)
    mask = torch.tensor(
        [[True, False, True], [False, True, True],
         [True, True, False], [False, False, True]], dtype=torch.bool
    )
    dw = rule.delta(pre, apical, basal, weights, mask=mask)
    assert torch.all(dw[~mask] == 0.0)


# ---------------------------------------------------------------------------
# Input / init validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_lr", [0.0, -0.01])
def test_rejects_non_positive_lr(bad_lr: float) -> None:
    with pytest.raises(ValueError, match="lr"):
        UrbanczikSennRule(lr=bad_lr, weight_decay=0.0)


def test_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        UrbanczikSennRule(lr=0.1, weight_decay=-0.1)


def test_rejects_mismatched_apical_basal() -> None:
    rule = UrbanczikSennRule(lr=0.1)
    pre = torch.rand(2, 3)
    apical = torch.rand(2, 4)
    basal = torch.rand(2, 5)                               # wrong last dim
    weights = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="basal"):
        rule.delta(pre, apical, basal, weights)


def test_rejects_wrong_weights_shape() -> None:
    rule = UrbanczikSennRule(lr=0.1)
    pre = torch.rand(2, 3)
    apical = torch.rand(2, 4)
    basal = torch.rand(2, 4)
    weights = torch.zeros(3, 4)                            # transposed
    with pytest.raises(ValueError, match="weights"):
        rule.delta(pre, apical, basal, weights)


def test_rejects_mask_wrong_dtype() -> None:
    rule = UrbanczikSennRule(lr=0.1)
    pre = torch.rand(2, 3)
    apical = torch.rand(2, 4)
    basal = torch.rand(2, 4)
    weights = torch.zeros(4, 3)
    bad_mask = torch.ones(4, 3)                            # float, not bool
    with pytest.raises(ValueError, match="bool"):
        rule.delta(pre, apical, basal, weights, mask=bad_mask)


def test_rejects_mismatched_batch_size() -> None:
    rule = UrbanczikSennRule(lr=0.1)
    pre = torch.rand(2, 3)
    apical = torch.rand(3, 4)                              # B=3 vs pre B=2
    basal = torch.rand(3, 4)
    weights = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="batch"):
        rule.delta(pre, apical, basal, weights)
