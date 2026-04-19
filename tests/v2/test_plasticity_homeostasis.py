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
    """Δθ_j = lr · tanh(error/scale) · scale with deadband (Task #54).

    error = mean_b(a_j) − ρ_target; deadband zeroes |error| < 0.2·|ρ|;
    scale = 0.1·|ρ| + 1e-3.
    """
    hom = ThresholdHomeostasis(
        lr=0.1, target_rate=1.0, n_units=3, init_theta=0.0,
        deadband_fraction=0.2,
    )
    # Means [2.0, 0.5, 1.0]. Errors [1.0, -0.5, 0.0]. Deadband = 0.2.
    # Unit 2 is at target (error=0, also within deadband) → no update.
    # Units 0 and 1 outside deadband → tanh-saturating update.
    activity = torch.tensor([[1.5, 0.5, 1.0], [2.5, 0.5, 1.0]])   # means [2.0, 0.5, 1.0]
    scale = 0.1 * 1.0 + 1e-3
    expected = torch.tensor([
        0.1 * torch.tanh(torch.tensor(1.0 / scale)).item() * scale,
        0.1 * torch.tanh(torch.tensor(-0.5 / scale)).item() * scale,
        0.0,
    ])
    hom.update(activity)
    torch.testing.assert_close(hom.theta, expected, atol=1e-7, rtol=0.0)


def test_deadband_zeros_small_errors() -> None:
    """Task #54: |error| < deadband_fraction·|ρ| → no update."""
    hom = ThresholdHomeostasis(
        lr=0.5, target_rate=1.0, n_units=2, init_theta=0.25,
        deadband_fraction=0.2,
    )
    theta_before = hom.theta.clone()
    # Error = 0.1 per unit ⇒ |0.1| < 0.2·1.0 = 0.2 deadband.
    activity = torch.full((3, 2), 1.1)
    hom.update(activity)
    torch.testing.assert_close(hom.theta, theta_before, atol=0.0, rtol=0.0)


def test_bounded_response_saturates() -> None:
    """Task #54: huge error does not produce huge Δθ — tanh saturates at ±lr·scale."""
    hom = ThresholdHomeostasis(
        lr=0.1, target_rate=1.0, n_units=2, init_theta=0.0,
        deadband_fraction=0.2,
    )
    scale = 0.1 * 1.0 + 1e-3  # 0.101
    # Error = 999 (huge). tanh → 1 ⇒ Δθ ≈ lr · scale = 0.0101.
    activity = torch.full((1, 2), 1000.0)
    hom.update(activity)
    saturation_bound = hom.lr * scale
    assert torch.all(hom.theta.abs() < 1.01 * saturation_bound)
    assert torch.all(hom.theta > 0.99 * saturation_bound)


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
    """Per-unit update uses mean over batch, not individual samples.

    With the Task #54 bounded rule: at target_rate=0 the deadband degenerates to
    zero, so any non-zero error drives a tanh-saturating update whose sign
    equals the sign of the batch mean per unit. The non-zero-mean units should
    receive a POSITIVE update (both batch means are positive here), and the
    magnitudes should saturate at ~lr·scale.
    """
    hom = ThresholdHomeostasis(
        lr=0.1, target_rate=0.0, n_units=2, init_theta=0.0,
        deadband_fraction=0.2,
    )
    activity = torch.tensor([[1.0, 2.0], [3.0, 4.0]])      # means [2.0, 3.0]
    hom.update(activity)
    # scale = 0.1*|0| + 1e-3 = 0.001; tanh(2000) ≈ tanh(3000) ≈ 1 ⇒ both saturate.
    scale = 1e-3
    expected = torch.tensor([
        0.1 * torch.tanh(torch.tensor(2.0 / scale)).item() * scale,
        0.1 * torch.tanh(torch.tensor(3.0 / scale)).item() * scale,
    ])
    torch.testing.assert_close(hom.theta, expected, atol=1e-8, rtol=0.0)
    # Both units move positively (their batch means are positive).
    assert torch.all(hom.theta > 0)


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
