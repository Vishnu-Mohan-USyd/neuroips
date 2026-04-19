"""Deterministic reconstruction from a fixed seed.

Two ``TokenBank(cfg, seed=s)`` instances built with the same seed must
return byte-identical ``tokens`` tensors (``atol=rtol=0``). Different seeds
must give different tensors. No global RNG state leaks: interleaving with
other ``torch.randn`` calls must not perturb the bank.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import N_TOKENS, TokenBank


@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_same_seed_reproduces_tokens_exactly(cfg, seed: int) -> None:
    """Two ``TokenBank`` instances with the same seed → identical tokens."""
    a = TokenBank(cfg, seed=seed)
    b = TokenBank(cfg, seed=seed)
    torch.testing.assert_close(a.tokens, b.tokens, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "s1,s2", [(0, 1), (1, 42), (0, 42), (7, 8), (0, 100)],
)
def test_different_seeds_differ(cfg, s1: int, s2: int) -> None:
    """Distinct seeds produce distinct banks (construction is seed-sensitive)."""
    a = TokenBank(cfg, seed=s1).tokens
    b = TokenBank(cfg, seed=s2).tokens
    assert not torch.allclose(a, b, atol=1e-6), (
        f"banks at seeds {s1} and {s2} are numerically identical"
    )


def test_rng_isolation_from_global_state(cfg) -> None:
    """Construction must not depend on ``torch.manual_seed``'s global state."""
    torch.manual_seed(12345)
    a = TokenBank(cfg, seed=0).tokens
    # Burn the global RNG
    _ = torch.randn(1000)
    torch.manual_seed(99999)
    _ = torch.randn(1000)
    # Same local seed should still reproduce the bank.
    b = TokenBank(cfg, seed=0).tokens
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_token_shape_preserved_across_seeds(cfg) -> None:
    """Different seeds don't change the output shape."""
    for seed in (0, 1, 42, 1234):
        bank = TokenBank(cfg, seed=seed)
        assert bank.tokens.shape == (N_TOKENS, 1, 32, 32)
