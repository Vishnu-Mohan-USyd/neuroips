"""Matched RMS contrast across the 12 identity tokens (±5 %).

Spec: all tokens must have the same pixel standard deviation to within ±5 %
of the cross-token mean. The construction applies an exact per-token linear
rescale to ``TARGET_RMS`` immediately before the final clip — for the default
``TARGET_RMS=0.08`` the clip never triggers and the realised tolerance is at
float32 machine precision (~1e-7).

The 5 % tolerance is the acceptance bar; the tests assert the tight spec.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import (
    N_TOKENS,
    TARGET_RMS,
    TokenBank,
)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_per_token_rms_within_5pct_of_mean(cfg, seed: int) -> None:
    """Relative spread ``|rms_t - mean(rms)| / mean(rms) < 0.05`` for all t."""
    bank = TokenBank(cfg, seed=seed)
    rms = bank.tokens.std(dim=(-1, -2, -3))                  # [12]
    mean_rms = float(rms.mean())
    rel = (rms - mean_rms).abs() / mean_rms
    assert float(rel.max()) < 0.05, (
        f"worst-token RMS deviation {float(rel.max()):.4f} exceeds 5 %"
    )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_rms_close_to_target_rms(cfg, seed: int) -> None:
    """Each token's RMS lands within 5 % of ``TARGET_RMS`` (no clip triggers)."""
    bank = TokenBank(cfg, seed=seed)
    rms = bank.tokens.std(dim=(-1, -2, -3))
    rel = (rms - TARGET_RMS).abs() / TARGET_RMS
    assert float(rel.max()) < 0.05, (
        f"worst-token RMS - TARGET_RMS = "
        f"{float((rms - TARGET_RMS).abs().max()):.4f}"
    )


def test_rms_is_positive(cfg) -> None:
    """Every token has positive spatial variance (no degenerate constant tiles)."""
    bank = TokenBank(cfg, seed=0)
    rms = bank.tokens.std(dim=(-1, -2, -3))
    assert float(rms.min()) > 0.0


def test_rms_via_balance_statistics(cfg) -> None:
    """``balance_statistics()[t]['rms_contrast']`` matches direct .std()."""
    bank = TokenBank(cfg, seed=0)
    stats = bank.balance_statistics()
    for t in range(N_TOKENS):
        direct = float(bank.tokens[t, 0].std().item())
        reported = stats[t]["rms_contrast"]
        assert isinstance(reported, float)
        assert abs(reported - direct) < 1e-6
