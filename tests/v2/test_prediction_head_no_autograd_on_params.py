"""Backward through forward must never accumulate gradients into any
``nn.Parameter`` — all raw weights have ``requires_grad=False`` and are
invisible to autograd. Gradients may still flow into module inputs
(so BPTT-fallback paths remain functional — design note #15).
"""

from __future__ import annotations

import torch

from src.v2_model.prediction_head import PredictionHead


def _assert_params_have_no_grad(head: PredictionHead) -> None:
    for name, p in head.named_parameters():
        assert p.grad is None, (
            f"PredictionHead.{name}.grad was accumulated; expected None"
        )


def test_all_params_have_requires_grad_false() -> None:
    head = PredictionHead()
    for name, p in head.named_parameters():
        assert p.requires_grad is False, (
            f"PredictionHead.{name} has requires_grad=True"
        )


def test_backward_full_inputs_leaves_params_ungraded() -> None:
    head = PredictionHead(seed=0)
    B = 3
    h_rate = torch.randn(B, 64, requires_grad=True)
    c_bias = torch.randn(B, 48, requires_grad=True)
    apical = torch.randn(B, 256, requires_grad=True)
    x_hat = head(h_rate, c_bias, apical)
    x_hat.sum().backward()
    _assert_params_have_no_grad(head)
    assert h_rate.grad is not None
    assert c_bias.grad is not None
    assert apical.grad is not None


def test_backward_h_only_leaves_params_ungraded() -> None:
    head = PredictionHead(seed=0)
    h_rate = torch.randn(2, 64, requires_grad=True)
    x_hat = head(h_rate)
    x_hat.sum().backward()
    _assert_params_have_no_grad(head)
    assert h_rate.grad is not None


def test_backward_h_and_c_leaves_params_ungraded() -> None:
    head = PredictionHead(seed=0)
    h_rate = torch.randn(2, 64, requires_grad=True)
    c_bias = torch.randn(2, 48, requires_grad=True)
    x_hat = head(h_rate, c_bias)
    x_hat.sum().backward()
    _assert_params_have_no_grad(head)
    assert h_rate.grad is not None
    assert c_bias.grad is not None
