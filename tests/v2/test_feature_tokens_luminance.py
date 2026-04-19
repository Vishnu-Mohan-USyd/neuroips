"""Matched-mean-luminance invariant across the 12 identity tokens.

Spec: ``|mean(token_t) - TARGET_MEAN| < 1e-4`` for every token t.
The construction guarantees this via an exact-arithmetic DC shift
(``raw - raw.mean() + TARGET_MEAN``) immediately before the final clip,
so the realised tolerance is at float32 machine precision (~1e-7).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import (
    N_TOKENS,
    TARGET_MEAN,
    TokenBank,
)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_per_token_mean_equals_target(cfg, seed: int) -> None:
    """Every token's mean luminance sits within 1e-4 of ``TARGET_MEAN``."""
    bank = TokenBank(cfg, seed=seed)
    per_token_mean = bank.tokens.mean(dim=(-1, -2, -3))      # [12]
    assert per_token_mean.shape == (N_TOKENS,)
    diffs = (per_token_mean - TARGET_MEAN).abs()
    assert float(diffs.max()) < 1e-4, (
        f"worst-token mean-luminance error {float(diffs.max()):.2e} "
        f"exceeds 1e-4 tolerance"
    )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_cross_token_mean_spread(cfg, seed: int) -> None:
    """No pair of tokens differs in mean luminance by more than 1e-4."""
    bank = TokenBank(cfg, seed=seed)
    per_token_mean = bank.tokens.mean(dim=(-1, -2, -3))
    spread = float(per_token_mean.max() - per_token_mean.min())
    assert spread < 1e-4, f"inter-token luminance spread {spread:.2e} >= 1e-4"


def test_mean_luminance_report_via_balance_statistics(cfg) -> None:
    """``balance_statistics()[t]['mean_lum']`` agrees with direct computation."""
    bank = TokenBank(cfg, seed=0)
    stats = bank.balance_statistics()
    for t in range(N_TOKENS):
        direct = float(bank.tokens[t, 0].mean().item())
        reported = stats[t]["mean_lum"]
        assert isinstance(reported, float)
        assert abs(reported - direct) < 1e-7


def test_zero_center_identity_for_centered_token(cfg) -> None:
    """Sanity: ``mean(token - mean(token) + 0.5) = 0.5`` exactly (arithmetic
    identity the construction relies upon)."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 32)
    centred = x - x.mean() + TARGET_MEAN
    assert abs(float(centred.mean()) - TARGET_MEAN) < 1e-6
