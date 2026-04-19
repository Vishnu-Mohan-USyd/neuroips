"""Dale's-law compliance for `initialize_masked_weights`.

  * `dale_sign="excitatory"` → all weights ≥ 0, zeros where mask is False.
  * `dale_sign="inhibitory"` → all weights ≤ 0, zeros where mask is False.
  * `dale_sign=None`         → weights may be any sign, zeros where mask is False.
  * Non-edge entries are zero (bit-exact).
  * Determinism under seed.
  * Rejects malformed inputs.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.connectivity import (
    generate_sparse_mask,
    initialize_masked_weights,
)


def _sample_mask(n: int = 64, sparsity: float = 0.15, seed: int = 0) -> torch.Tensor:
    return generate_sparse_mask(
        positions=None, features=None, n_units=n, sparsity=sparsity, seed=seed
    )


# ---------------------------------------------------------------------------
# Sign constraints
# ---------------------------------------------------------------------------

def test_excitatory_weights_non_negative() -> None:
    mask = _sample_mask()
    w = initialize_masked_weights(
        mask=mask, scale=0.1, dale_sign="excitatory", seed=0
    )
    assert (w >= 0).all(), (
        f"excitatory init produced a negative weight: min={w.min().item()}"
    )


def test_inhibitory_weights_non_positive() -> None:
    mask = _sample_mask()
    w = initialize_masked_weights(
        mask=mask, scale=0.1, dale_sign="inhibitory", seed=0
    )
    assert (w <= 0).all(), (
        f"inhibitory init produced a positive weight: max={w.max().item()}"
    )


def test_none_dale_sign_allows_mixed_signs() -> None:
    mask = _sample_mask(n=128, sparsity=0.25, seed=11)
    w = initialize_masked_weights(
        mask=mask, scale=0.5, dale_sign=None, seed=0
    )
    # On edges (mask=True), raw Gaussian draws produce both positive and
    # negative entries with overwhelming probability.
    on_edges = w[mask]
    assert (on_edges > 0).any() and (on_edges < 0).any()


# ---------------------------------------------------------------------------
# Masked-off entries are zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sign", ["excitatory", "inhibitory", None])
def test_masked_off_entries_are_exactly_zero(sign) -> None:
    mask = _sample_mask(n=64, sparsity=0.2, seed=5)
    w = initialize_masked_weights(mask=mask, scale=0.3, dale_sign=sign, seed=0)
    off_edges = w[~mask]
    assert torch.all(off_edges == 0.0), (
        f"sign={sign}: found {(off_edges != 0).sum().item()} non-zero "
        f"entries where mask is False."
    )


@pytest.mark.parametrize("sign", ["excitatory", "inhibitory"])
def test_on_edges_nonzero_with_overwhelming_probability(sign) -> None:
    """For Dale-constrained signs, softplus(raw) is strictly > 0. So every
    mask=True cell has |w| > 0."""
    mask = _sample_mask(n=64, sparsity=0.2, seed=7)
    w = initialize_masked_weights(mask=mask, scale=0.3, dale_sign=sign, seed=0)
    on_edges = w[mask]
    assert (on_edges != 0.0).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_weights() -> None:
    mask = _sample_mask()
    w1 = initialize_masked_weights(
        mask=mask, scale=0.1, dale_sign="excitatory", seed=42
    )
    w2 = initialize_masked_weights(
        mask=mask, scale=0.1, dale_sign="excitatory", seed=42
    )
    torch.testing.assert_close(w1, w2, atol=0.0, rtol=0.0)


def test_different_seed_gives_different_weights() -> None:
    mask = _sample_mask()
    w1 = initialize_masked_weights(
        mask=mask, scale=0.1, dale_sign="excitatory", seed=1
    )
    w2 = initialize_masked_weights(
        mask=mask, scale=0.1, dale_sign="excitatory", seed=2
    )
    assert not torch.equal(w1, w2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_non_bool_mask() -> None:
    bad_mask = torch.zeros(8, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="bool"):
        initialize_masked_weights(
            mask=bad_mask, scale=0.1, dale_sign="excitatory", seed=0
        )


def test_rejects_non_positive_scale() -> None:
    mask = _sample_mask(n=16, sparsity=0.3, seed=0)
    with pytest.raises(ValueError, match="scale"):
        initialize_masked_weights(mask=mask, scale=0.0, dale_sign=None, seed=0)


def test_rejects_unknown_dale_sign() -> None:
    mask = _sample_mask(n=16, sparsity=0.3, seed=0)
    with pytest.raises(ValueError, match="dale_sign"):
        initialize_masked_weights(
            mask=mask, scale=0.1, dale_sign="disinhibitory", seed=0  # type: ignore[arg-type]
        )
