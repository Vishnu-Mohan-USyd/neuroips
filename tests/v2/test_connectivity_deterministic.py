"""Determinism: same seed + same inputs → bit-exact same mask.

`generate_sparse_mask` uses a local `torch.Generator` seeded by the `seed`
argument. No other RNG state leaks in. This file nails that contract down
with a suite of equivalences and non-equivalences.
"""

from __future__ import annotations

import torch

from src.v2_model.connectivity import generate_sparse_mask


def test_same_seed_same_inputs_bit_exact() -> None:
    """Two calls with identical args produce identical masks."""
    kwargs = dict(
        positions=None, features=None, n_units=128, sparsity=0.15, seed=2025
    )
    m1 = generate_sparse_mask(**kwargs)
    m2 = generate_sparse_mask(**kwargs)
    assert torch.equal(m1, m2)


def test_different_seed_different_masks() -> None:
    """Different seeds produce at least one differing entry."""
    m1 = generate_sparse_mask(
        positions=None, features=None, n_units=128, sparsity=0.15, seed=1
    )
    m2 = generate_sparse_mask(
        positions=None, features=None, n_units=128, sparsity=0.15, seed=2
    )
    assert not torch.equal(m1, m2)


def test_same_seed_retinotopic_is_bit_exact() -> None:
    """Determinism holds when positions + features are provided."""
    positions = torch.rand(64, 2) * 16.0
    features = torch.rand(64) * 180.0
    kwargs = dict(
        positions=positions, features=features, n_units=64, sparsity=0.15,
        sigma_position=4.0, sigma_feature=25.0, seed=99,
    )
    m1 = generate_sparse_mask(**kwargs)
    m2 = generate_sparse_mask(**kwargs)
    assert torch.equal(m1, m2)


def test_global_rng_does_not_affect_mask() -> None:
    """Mutating the global torch RNG between calls must not change the output
    — proves the generator is fully local to the function."""
    kwargs = dict(
        positions=None, features=None, n_units=64, sparsity=0.20, seed=123
    )
    torch.manual_seed(0)
    m1 = generate_sparse_mask(**kwargs)

    # Chew up the global RNG with unrelated draws.
    _ = torch.randn(10_000)
    torch.manual_seed(99999)
    _ = torch.randint(0, 100, (10_000,))

    m2 = generate_sparse_mask(**kwargs)
    assert torch.equal(m1, m2)


def test_order_independence_across_calls() -> None:
    """Interleaving calls with different seeds does not perturb either
    result — each call's RNG state is fully isolated."""
    m_a1 = generate_sparse_mask(
        positions=None, features=None, n_units=32, sparsity=0.25, seed=1
    )
    m_b1 = generate_sparse_mask(
        positions=None, features=None, n_units=32, sparsity=0.25, seed=2
    )
    m_b2 = generate_sparse_mask(
        positions=None, features=None, n_units=32, sparsity=0.25, seed=2
    )
    m_a2 = generate_sparse_mask(
        positions=None, features=None, n_units=32, sparsity=0.25, seed=1
    )
    assert torch.equal(m_a1, m_a2)
    assert torch.equal(m_b1, m_b2)
