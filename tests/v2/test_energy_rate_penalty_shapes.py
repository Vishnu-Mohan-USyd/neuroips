"""Shape / sign / zero-input invariants for `rate_penalty_delta_drive`.

Per v4 D.18, the L1 rate penalty returns a subtractive contribution `−α`
for every firing excitatory unit and zero for silent units. The sign
convention is "add this to the drive".
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.energy import EnergyPenalty


def test_shape_matches_rate() -> None:
    energy = EnergyPenalty(alpha=0.1, beta=0.0)
    rate = torch.rand(3, 7)
    out = energy.rate_penalty_delta_drive(rate)
    assert out.shape == rate.shape


def test_dtype_matches_rate() -> None:
    energy = EnergyPenalty(alpha=0.1, beta=0.0)
    rate32 = torch.rand(5, dtype=torch.float32)
    rate64 = torch.rand(5, dtype=torch.float64)
    assert energy.rate_penalty_delta_drive(rate32).dtype == torch.float32
    assert energy.rate_penalty_delta_drive(rate64).dtype == torch.float64


def test_zero_rate_gives_zero_penalty() -> None:
    """Silent units: no rate, no cost."""
    energy = EnergyPenalty(alpha=0.1, beta=0.0)
    rate = torch.zeros(4, 8)
    out = energy.rate_penalty_delta_drive(rate)
    assert torch.all(out == 0.0)


def test_positive_rate_gives_minus_alpha() -> None:
    """Every firing unit contributes exactly `−α` (step-function)."""
    alpha = 0.05
    energy = EnergyPenalty(alpha=alpha, beta=0.0)
    rate = torch.tensor([[0.0, 0.01, 1.0, 100.0]])
    out = energy.rate_penalty_delta_drive(rate)
    expected = torch.tensor([[0.0, -alpha, -alpha, -alpha]])
    torch.testing.assert_close(out, expected, atol=1e-7, rtol=0.0)


def test_mixed_firing_silent_patterns() -> None:
    """Batched rate: firing mask applies per-entry independently."""
    energy = EnergyPenalty(alpha=0.2, beta=0.0)
    rate = torch.tensor([[1.0, 0.0, 2.5], [0.0, 0.3, 0.0]])
    out = energy.rate_penalty_delta_drive(rate)
    expected = torch.tensor([[-0.2, 0.0, -0.2], [0.0, -0.2, 0.0]])
    torch.testing.assert_close(out, expected, atol=1e-7, rtol=0.0)


def test_alpha_zero_gives_zero_everywhere() -> None:
    """α = 0 turns the penalty off entirely."""
    energy = EnergyPenalty(alpha=0.0, beta=0.0)
    rate = torch.rand(3, 5) + 0.1                                   # all firing
    out = energy.rate_penalty_delta_drive(rate)
    assert torch.all(out == 0.0)


def test_output_is_not_positive() -> None:
    """The L1 rate penalty only ever subtracts from the drive — never adds."""
    energy = EnergyPenalty(alpha=0.1, beta=0.0)
    rate = torch.rand(10, 20)                                       # ≥ 0 random
    out = energy.rate_penalty_delta_drive(rate)
    assert (out <= 0).all()


# ---------------------------------------------------------------------------
# Validation / init
# ---------------------------------------------------------------------------

def test_rejects_integer_rate() -> None:
    energy = EnergyPenalty(alpha=0.1, beta=0.0)
    with pytest.raises(ValueError, match="floating-point"):
        energy.rate_penalty_delta_drive(torch.zeros(4, dtype=torch.long))


def test_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        EnergyPenalty(alpha=-0.01, beta=0.0)


def test_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        EnergyPenalty(alpha=0.0, beta=-0.01)
