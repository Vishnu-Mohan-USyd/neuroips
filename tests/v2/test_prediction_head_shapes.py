"""Forward output shape [B, n_l4_e]; optional-input acceptance; config defaults."""

from __future__ import annotations

import pytest
import torch

from src.v2_model.prediction_head import PredictionHead


def _default() -> PredictionHead:
    return PredictionHead(
        n_l4_e=128, n_h_e=64, n_c_bias=48, n_l23_apical=256, seed=0,
    )


def test_forward_output_shape_full_inputs() -> None:
    head = _default()
    B = 3
    x_hat = head(
        torch.randn(B, 64),
        torch.randn(B, 48),
        torch.randn(B, 256),
    )
    assert x_hat.shape == (B, 128)
    assert x_hat.dtype == torch.float32


def test_forward_output_shape_h_only() -> None:
    head = _default()
    B = 5
    x_hat = head(torch.randn(B, 64))
    assert x_hat.shape == (B, 128)


def test_forward_output_shape_h_and_c_only() -> None:
    head = _default()
    B = 2
    x_hat = head(torch.randn(B, 64), torch.randn(B, 48))
    assert x_hat.shape == (B, 128)


def test_forward_output_shape_h_and_apical_only() -> None:
    head = _default()
    B = 4
    x_hat = head(
        torch.randn(B, 64),
        None,
        torch.randn(B, 256),
    )
    assert x_hat.shape == (B, 128)


def test_forward_batch_size_one() -> None:
    head = _default()
    x_hat = head(torch.randn(1, 64), torch.randn(1, 48), torch.randn(1, 256))
    assert x_hat.shape == (1, 128)


# ---------------------------------------------------------------------------
# Constructor defaults match arch config
# ---------------------------------------------------------------------------

def test_defaults_match_arch_config() -> None:
    from src.v2_model.config import ArchitectureConfig
    cfg = ArchitectureConfig()
    head = PredictionHead()
    assert head.n_l4_e == cfg.n_l4_e
    assert head.n_h_e == cfg.n_h_e
    assert head.n_c_bias == cfg.n_c
    assert head.n_l23_apical == cfg.n_l23_e


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------

def test_n_l4_e_zero_raises() -> None:
    with pytest.raises(ValueError, match="n_l4_e"):
        PredictionHead(n_l4_e=0, n_h_e=64)


def test_n_h_e_zero_raises() -> None:
    with pytest.raises(ValueError, match="n_h_e"):
        PredictionHead(n_l4_e=128, n_h_e=0)


def test_n_c_bias_zero_raises() -> None:
    with pytest.raises(ValueError, match="n_c_bias"):
        PredictionHead(n_c_bias=0)


def test_n_l23_apical_zero_raises() -> None:
    with pytest.raises(ValueError, match="n_l23_apical"):
        PredictionHead(n_l23_apical=0)


def test_wrong_h_rate_shape_raises() -> None:
    head = _default()
    with pytest.raises(ValueError, match="h_rate"):
        head(torch.randn(3, 32))                       # wrong feature dim


def test_wrong_c_bias_shape_raises() -> None:
    head = _default()
    with pytest.raises(ValueError, match="c_bias"):
        head(torch.randn(3, 64), torch.randn(3, 99))


def test_wrong_apical_shape_raises() -> None:
    head = _default()
    with pytest.raises(ValueError, match="l23_apical_summary"):
        head(torch.randn(3, 64), None, torch.randn(3, 99))


def test_c_bias_passed_when_weight_absent_raises() -> None:
    head = PredictionHead(n_c_bias=None, n_l23_apical=256)
    with pytest.raises(ValueError, match="c_bias"):
        head(torch.randn(2, 64), torch.randn(2, 48))


def test_apical_passed_when_weight_absent_raises() -> None:
    head = PredictionHead(n_c_bias=48, n_l23_apical=None)
    with pytest.raises(ValueError, match="l23_apical_summary"):
        head(torch.randn(2, 64), None, torch.randn(2, 256))
