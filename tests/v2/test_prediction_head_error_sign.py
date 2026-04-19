"""``compute_error`` arithmetic: ε = x_actual − x_predicted (sign + identity)."""

from __future__ import annotations

import pytest
import torch

from src.v2_model.prediction_head import compute_error


def test_error_actual_minus_zero_returns_actual() -> None:
    x = torch.randn(3, 128)
    zero = torch.zeros_like(x)
    eps = compute_error(x, zero)
    torch.testing.assert_close(eps, x, atol=0.0, rtol=0.0)


def test_error_zero_minus_predicted_returns_negative_predicted() -> None:
    x = torch.randn(3, 128)
    zero = torch.zeros_like(x)
    eps = compute_error(zero, x)
    torch.testing.assert_close(eps, -x, atol=0.0, rtol=0.0)


def test_error_identical_tensors_returns_zero() -> None:
    x = torch.randn(2, 128)
    eps = compute_error(x, x)
    torch.testing.assert_close(
        eps, torch.zeros_like(x), atol=0.0, rtol=0.0,
    )


def test_error_preserves_shape() -> None:
    for B in (1, 2, 5, 16):
        for N in (4, 128, 256):
            a = torch.randn(B, N)
            b = torch.randn(B, N)
            assert compute_error(a, b).shape == (B, N)


def test_error_dtype_preserved() -> None:
    a = torch.randn(2, 128, dtype=torch.float64)
    b = torch.randn(2, 128, dtype=torch.float64)
    eps = compute_error(a, b)
    assert eps.dtype == torch.float64


def test_error_shape_mismatch_raises() -> None:
    a = torch.randn(2, 128)
    b = torch.randn(3, 128)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_error(a, b)

    a = torch.randn(2, 128)
    b = torch.randn(2, 64)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_error(a, b)


def test_error_arithmetic_composes() -> None:
    """ε(a, b) + ε(b, c) = ε(a, c)."""
    a = torch.randn(2, 128)
    b = torch.randn(2, 128)
    c = torch.randn(2, 128)
    lhs = compute_error(a, b) + compute_error(b, c)
    rhs = compute_error(a, c)
    torch.testing.assert_close(lhs, rhs, atol=1e-6, rtol=0.0)
