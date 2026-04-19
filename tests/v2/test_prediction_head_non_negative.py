"""Prediction output must be non-negative regardless of input sign.

``rectified_softplus`` enforces f(x) ≥ 0 at the output; Dale-excitatory
``softplus``-parameterised weights mean the drive itself is a non-negative
combination when inputs are non-negative — but the test also checks that
negative inputs (allowed at the module interface even if unusual biologically)
do not break the non-negativity guarantee.
"""

from __future__ import annotations

import torch

from src.v2_model.prediction_head import PredictionHead


def _default() -> PredictionHead:
    return PredictionHead(
        n_l4_e=128, n_h_e=64, n_c_bias=48, n_l23_apical=256, seed=0,
    )


def test_non_negative_with_positive_inputs() -> None:
    head = _default()
    x_hat = head(
        torch.rand(4, 64),
        torch.rand(4, 48),
        torch.rand(4, 256),
    )
    assert (x_hat >= 0.0).all()


def test_non_negative_with_negative_inputs() -> None:
    """Even with negative inputs (e.g. gradients probes), output stays ≥ 0."""
    head = _default()
    x_hat = head(
        -torch.rand(4, 64),
        -torch.rand(4, 48),
        -torch.rand(4, 256),
    )
    assert (x_hat >= 0.0).all()


def test_non_negative_with_mixed_inputs() -> None:
    head = _default()
    x_hat = head(
        torch.randn(8, 64),
        torch.randn(8, 48),
        torch.randn(8, 256),
    )
    assert (x_hat >= 0.0).all()


def test_non_negative_zero_inputs() -> None:
    """Zero inputs → output = rectified_softplus(softplus(b_pred_raw)) ≥ 0."""
    head = _default()
    x_hat = head(
        torch.zeros(2, 64),
        torch.zeros(2, 48),
        torch.zeros(2, 256),
    )
    assert (x_hat >= 0.0).all()


def test_non_negative_h_only_path() -> None:
    """None optionals → output must still be ≥ 0."""
    head = _default()
    x_hat = head(torch.randn(3, 64))
    assert (x_hat >= 0.0).all()


def test_non_negative_extreme_magnitudes() -> None:
    """Large |input| must not push output negative (no numerical aberration)."""
    head = _default()
    x_hat = head(
        100.0 * torch.randn(2, 64),
        100.0 * torch.randn(2, 48),
        100.0 * torch.randn(2, 256),
    )
    assert torch.isfinite(x_hat).all()
    assert (x_hat >= 0.0).all()
