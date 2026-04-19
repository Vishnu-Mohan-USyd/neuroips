"""Mask / sparse-connectivity preservation for `current_weight_shrinkage`.

The shrinkage must not create a non-zero update at a position where the
connectivity mask says there is no synapse — applying the penalty to a
zero weight already gives zero, but an explicit mask check guards against
future changes that might break this invariant.
"""

from __future__ import annotations

import torch

from src.v2_model.connectivity import generate_sparse_mask
from src.v2_model.energy import EnergyPenalty


def _make_mask(n_post: int, n_pre: int, sparsity: float = 0.3, seed: int = 0) -> torch.Tensor:
    n = max(n_post, n_pre)
    full = generate_sparse_mask(
        positions=None, features=None, n_units=n,
        sparsity=sparsity, seed=seed,
    )
    return full[:n_post, :n_pre].contiguous()


def test_masked_off_entries_are_exactly_zero() -> None:
    """ΔW[~mask] == 0, even when pre and weights would naturally give non-zero."""
    n_post, n_pre = 8, 6
    mask = _make_mask(n_post, n_pre, sparsity=0.3, seed=1)
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(n_post, n_pre) + 0.5
    pre = torch.rand(4, n_pre) + 0.1                                # all active
    dw = energy.current_weight_shrinkage(weights, pre, mask=mask)

    # Precondition: the unmasked result must have non-zero entries at
    # masked-off positions, else the test is vacuous.
    dw_unmasked = energy.current_weight_shrinkage(weights, pre)
    assert (dw_unmasked[~mask] != 0.0).any()

    assert torch.all(dw[~mask] == 0.0)
    torch.testing.assert_close(dw[mask], dw_unmasked[mask], atol=0.0, rtol=0.0)


def test_all_false_mask_gives_zero_update() -> None:
    """Fully-masked-off ⇒ ΔW entirely zero."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(4, 3)
    pre = torch.rand(2, 3) + 0.1
    mask = torch.zeros(4, 3, dtype=torch.bool)
    dw = energy.current_weight_shrinkage(weights, pre, mask=mask)
    assert torch.all(dw == 0.0)


def test_all_true_mask_matches_unmasked() -> None:
    """Fully-True ⇒ ΔW identical to no-mask case."""
    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(4, 3)
    pre = torch.rand(2, 3) + 0.1
    dw_full = energy.current_weight_shrinkage(
        weights, pre, mask=torch.ones(4, 3, dtype=torch.bool)
    )
    dw_none = energy.current_weight_shrinkage(weights, pre)
    torch.testing.assert_close(dw_full, dw_none, atol=0.0, rtol=0.0)


def test_mask_wrong_shape_raises() -> None:
    import pytest

    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(4, 3)
    pre = torch.rand(2, 3)
    bad_mask = torch.ones(4, 4, dtype=torch.bool)                   # wrong shape
    with pytest.raises(ValueError, match="shape"):
        energy.current_weight_shrinkage(weights, pre, mask=bad_mask)


def test_mask_wrong_dtype_raises() -> None:
    import pytest

    energy = EnergyPenalty(alpha=0.0, beta=1e-2)
    weights = torch.randn(4, 3)
    pre = torch.rand(2, 3)
    bad_mask = torch.ones(4, 3, dtype=torch.float32)                # not bool
    with pytest.raises(ValueError, match="bool"):
        energy.current_weight_shrinkage(weights, pre, mask=bad_mask)
