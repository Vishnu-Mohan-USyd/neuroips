"""Threshold-drift homeostasis.

Verifies:
  * overactive unit (mean_b(a) > ρ_target) ⇒ θ rises;
  * underactive unit (mean_b(a) < ρ_target) ⇒ θ falls;
  * at-target activity ⇒ θ unchanged (fixed point);
  * repeated updates on above/below-target activity are monotonic;
  * batch averaging: two-sample activity uses the batch mean;
  * init/input validation (lr, n_units, activity shape).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.plasticity import ThresholdHomeostasis


# ---------------------------------------------------------------------------
# Sign behaviour
# ---------------------------------------------------------------------------

def test_overactive_unit_theta_rises() -> None:
    hom = ThresholdHomeostasis(lr=0.01, target_rate=1.0, n_units=4, init_theta=0.0)
    theta_before = hom.theta.clone()
    activity = torch.full((2, 4), 2.0)                     # above target
    hom.update(activity)
    assert torch.all(hom.theta > theta_before)


def test_underactive_unit_theta_falls() -> None:
    hom = ThresholdHomeostasis(lr=0.01, target_rate=1.0, n_units=4, init_theta=0.0)
    theta_before = hom.theta.clone()
    activity = torch.full((2, 4), 0.5)                     # below target
    hom.update(activity)
    assert torch.all(hom.theta < theta_before)


def test_at_target_theta_unchanged() -> None:
    """Fixed-point test: mean_b(a) == ρ ⇒ Δθ = 0 ⇒ θ stays pinned."""
    hom = ThresholdHomeostasis(lr=0.1, target_rate=1.0, n_units=4, init_theta=0.3)
    theta_before = hom.theta.clone()
    activity = torch.ones(5, 4)                            # mean = 1.0 = target
    hom.update(activity)
    torch.testing.assert_close(hom.theta, theta_before, atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# Analytic magnitude
# ---------------------------------------------------------------------------

def test_update_magnitude_matches_formula() -> None:
    """Δθ_j = lr · (mean_b(a_j) − ρ) per-unit."""
    hom = ThresholdHomeostasis(lr=0.1, target_rate=1.0, n_units=3, init_theta=0.0)
    activity = torch.tensor([[1.5, 0.5, 1.0], [2.5, 0.5, 1.0]])   # means [2.0, 0.5, 1.0]
    expected = torch.tensor([0.1 * (2.0 - 1.0), 0.1 * (0.5 - 1.0), 0.0])
    hom.update(activity)
    torch.testing.assert_close(hom.theta, expected, atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# Monotonic drift on constant inputs
# ---------------------------------------------------------------------------

def test_repeated_overactive_updates_are_monotonic() -> None:
    """θ grows monotonically across steps when activity stays above target."""
    hom = ThresholdHomeostasis(lr=0.01, target_rate=1.0, n_units=3, init_theta=0.0)
    activity = torch.full((2, 3), 2.0)
    thetas = [hom.theta.clone()]
    for _ in range(5):
        hom.update(activity)
        thetas.append(hom.theta.clone())
    for t_prev, t_next in zip(thetas[:-1], thetas[1:]):
        assert torch.all(t_next > t_prev)


def test_repeated_underactive_updates_are_monotonic() -> None:
    """θ decreases monotonically across steps when activity stays below target."""
    hom = ThresholdHomeostasis(lr=0.01, target_rate=1.0, n_units=3, init_theta=0.5)
    activity = torch.full((2, 3), 0.5)
    thetas = [hom.theta.clone()]
    for _ in range(5):
        hom.update(activity)
        thetas.append(hom.theta.clone())
    for t_prev, t_next in zip(thetas[:-1], thetas[1:]):
        assert torch.all(t_next < t_prev)


# ---------------------------------------------------------------------------
# Batch averaging
# ---------------------------------------------------------------------------

def test_batch_mean_is_used() -> None:
    """Per-unit update uses mean over batch, not individual samples."""
    hom = ThresholdHomeostasis(lr=0.1, target_rate=0.0, n_units=2, init_theta=0.0)
    activity = torch.tensor([[1.0, 2.0], [3.0, 4.0]])      # means [2.0, 3.0]
    hom.update(activity)
    expected = torch.tensor([0.1 * 2.0, 0.1 * 3.0])
    torch.testing.assert_close(hom.theta, expected, atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# Theta is a buffer
# ---------------------------------------------------------------------------

def test_theta_is_registered_buffer() -> None:
    hom = ThresholdHomeostasis(lr=0.1, target_rate=1.0, n_units=4)
    buffer_names = [name for name, _ in hom.named_buffers()]
    assert "theta" in buffer_names
    # And it's *not* a parameter (no grad).
    param_names = [name for name, _ in hom.named_parameters()]
    assert "theta" not in param_names


def test_theta_shape_and_init_value() -> None:
    hom = ThresholdHomeostasis(
        lr=0.1, target_rate=1.0, n_units=5, init_theta=0.25
    )
    assert hom.theta.shape == (5,)
    torch.testing.assert_close(
        hom.theta, torch.full((5,), 0.25), atol=1e-7, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Input / init validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_lr", [0.0, -0.01])
def test_rejects_non_positive_lr(bad_lr: float) -> None:
    with pytest.raises(ValueError, match="lr"):
        ThresholdHomeostasis(lr=bad_lr, target_rate=1.0, n_units=4)


def test_rejects_non_positive_n_units() -> None:
    with pytest.raises(ValueError, match="n_units"):
        ThresholdHomeostasis(lr=0.1, target_rate=1.0, n_units=0)


def test_rejects_negative_target_rate() -> None:
    with pytest.raises(ValueError, match="target_rate"):
        ThresholdHomeostasis(lr=0.1, target_rate=-0.1, n_units=4)


def test_rejects_wrong_activity_shape() -> None:
    hom = ThresholdHomeostasis(lr=0.1, target_rate=1.0, n_units=4)
    with pytest.raises(ValueError, match="n_units"):
        hom.update(torch.rand(2, 5))                       # wrong last dim


def test_rejects_non_2d_activity() -> None:
    hom = ThresholdHomeostasis(lr=0.1, target_rate=1.0, n_units=4)
    with pytest.raises(ValueError, match="2-D"):
        hom.update(torch.rand(4))                          # missing batch dim
