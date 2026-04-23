"""Same seed → bit-identical weights; same inputs → bit-identical forward."""

from __future__ import annotations

import torch

from src.v2_model.prediction_head import PredictionHead


def test_same_seed_identical_weights() -> None:
    h1 = PredictionHead(seed=42)
    h2 = PredictionHead(seed=42)
    for name in h1._all_plastic_names:
        torch.testing.assert_close(
            getattr(h1, name), getattr(h2, name), atol=0.0, rtol=0.0,
        )


def test_different_seed_different_weights() -> None:
    h1 = PredictionHead(seed=0)
    h2 = PredictionHead(seed=1)
    assert not torch.equal(h1.W_pred_H_raw, h2.W_pred_H_raw)
    assert not torch.equal(h1.W_pred_C_raw, h2.W_pred_C_raw)
    assert not torch.equal(h1.W_pred_apical_raw, h2.W_pred_apical_raw)


def test_bias_init_deterministic_across_seeds() -> None:
    """``b_pred_raw`` is seed-independent (constant init at -8.0).

    Task #74 Fix O (2026-04-22): init changed from -10.0 → -8.0 to match
    the [-8, 8] raw clamp used by ``apply_plasticity_step``.
    """
    h1 = PredictionHead(seed=0)
    h2 = PredictionHead(seed=999)
    torch.testing.assert_close(
        h1.b_pred_raw, h2.b_pred_raw, atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        h1.b_pred_raw, torch.full_like(h1.b_pred_raw, -8.0),
        atol=0.0, rtol=0.0,
    )


def test_forward_bit_exact_across_instances_full_inputs() -> None:
    h1 = PredictionHead(seed=11)
    h2 = PredictionHead(seed=11)
    B = 3
    torch.manual_seed(0)
    h_rate = torch.randn(B, 64)
    c_bias = torch.randn(B, 48)
    apical = torch.randn(B, 256)
    y1 = h1(h_rate, c_bias, apical)
    y2 = h2(h_rate, c_bias, apical)
    torch.testing.assert_close(y1, y2, atol=0.0, rtol=0.0)


def test_forward_bit_exact_across_calls() -> None:
    head = PredictionHead(seed=7)
    B = 2
    h_rate = torch.randn(B, 64)
    c_bias = torch.randn(B, 48)
    apical = torch.randn(B, 256)
    y1 = head(h_rate, c_bias, apical)
    y2 = head(h_rate, c_bias, apical)
    torch.testing.assert_close(y1, y2, atol=0.0, rtol=0.0)


def test_forward_bit_exact_h_only_path() -> None:
    h1 = PredictionHead(seed=3)
    h2 = PredictionHead(seed=3)
    h_rate = torch.randn(4, 64)
    y1 = h1(h_rate)
    y2 = h2(h_rate)
    torch.testing.assert_close(y1, y2, atol=0.0, rtol=0.0)
