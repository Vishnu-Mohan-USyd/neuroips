"""Optional inputs: c_bias=None ≡ c_bias=torch.zeros(...); same for l23_apical.

Passing ``None`` on an optional forward arg must produce bit-identical output
to passing an explicit zero tensor of the correct shape. The skip-matmul
implementation detail must not change the numerical result.
"""

from __future__ import annotations

import torch

from src.v2_model.prediction_head import PredictionHead


def _default() -> PredictionHead:
    return PredictionHead(
        n_l4_e=128, n_h_e=64, n_c_bias=48, n_l23_apical=256, seed=0,
    )


def test_none_c_bias_equals_explicit_zero() -> None:
    head = _default()
    B = 3
    torch.manual_seed(0)
    h_rate = torch.randn(B, 64)
    apical = torch.randn(B, 256)

    y_none = head(h_rate, None, apical)
    y_zero = head(h_rate, torch.zeros(B, 48), apical)
    torch.testing.assert_close(y_none, y_zero, atol=0.0, rtol=0.0)


def test_none_apical_equals_explicit_zero() -> None:
    head = _default()
    B = 4
    torch.manual_seed(0)
    h_rate = torch.randn(B, 64)
    c_bias = torch.randn(B, 48)

    y_none = head(h_rate, c_bias, None)
    y_zero = head(h_rate, c_bias, torch.zeros(B, 256))
    torch.testing.assert_close(y_none, y_zero, atol=0.0, rtol=0.0)


def test_all_optionals_none_equals_all_zero() -> None:
    head = _default()
    B = 2
    torch.manual_seed(0)
    h_rate = torch.randn(B, 64)

    y_none = head(h_rate, None, None)
    y_zero = head(h_rate, torch.zeros(B, 48), torch.zeros(B, 256))
    torch.testing.assert_close(y_none, y_zero, atol=0.0, rtol=0.0)


def test_h_only_call_equals_explicit_zeros_on_all_optionals() -> None:
    """Calling ``head(h_rate)`` must match ``head(h_rate, 0, 0)`` exactly."""
    head = _default()
    B = 5
    torch.manual_seed(1)
    h_rate = torch.randn(B, 64)

    y_default = head(h_rate)
    y_explicit = head(h_rate, torch.zeros(B, 48), torch.zeros(B, 256))
    torch.testing.assert_close(y_default, y_explicit, atol=0.0, rtol=0.0)


def test_skip_matmul_path_not_degenerate() -> None:
    """Sanity: the no-inputs path still uses W_pred_H + bias, not a zero output."""
    head = _default()
    h_rate = torch.randn(4, 64)
    y = head(h_rate)
    # Shouldn't be constant across batch items if W_pred_H is random.
    assert not torch.allclose(y[0], y[1])


def test_contributions_are_additive() -> None:
    """y_full − y_h_only should equal softplus(W_C) @ c_bias +
    softplus(W_apical) @ apical (pre-rectification this is exact; post-rect
    may differ if the rect mask flips — so check a strong-positive drive
    regime)."""
    head = _default()
    B = 2
    h_rate = 3.0 * torch.rand(B, 64)                               # ensures drive > 0
    c_bias = torch.rand(B, 48)
    apical = torch.rand(B, 256)

    y_h = head(h_rate)
    y_full = head(h_rate, c_bias, apical)
    # With non-negative inputs + non-negative (softplus) weights, full pred
    # must be elementwise ≥ h-only pred (rectified_softplus is monotonic).
    assert (y_full >= y_h - 1e-6).all()
