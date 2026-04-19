"""Mask / sparse-connectivity preservation — one dedicated test per rule.

Per v4 plan §Learning rules: plasticity must not create edges where the
connectivity mask says there is no synapse. The returned ΔW must have an
exact zero at every entry where `mask[i, j]` is False.

This suite exercises realistic inputs (random tensors with a moderate-density
mask from `generate_sparse_mask`) rather than hand-constructed zeros, to make
sure the rule would *naturally* produce a non-zero update at the masked-off
entry if the mask were not enforced.
"""

from __future__ import annotations

import torch

from src.v2_model.connectivity import generate_sparse_mask
from src.v2_model.plasticity import (
    ThreeFactorRule,
    UrbanczikSennRule,
    VogelsISTDPRule,
)


def _make_mask(n_post: int, n_pre: int, sparsity: float = 0.3, seed: int = 0) -> torch.Tensor:
    """Build a random but deterministic mask with shape [n_post, n_pre].

    Uses `generate_sparse_mask` at n=max(n_post, n_pre) then crops. This is a
    straightforward way to get a realistic boolean mask matching the
    plasticity-API convention (rows = post, cols = pre).
    """
    n = max(n_post, n_pre)
    full = generate_sparse_mask(
        positions=None, features=None, n_units=n,
        sparsity=sparsity, seed=seed,
    )
    return full[:n_post, :n_pre].contiguous()


def test_urbanczik_senn_preserves_mask() -> None:
    n_pre, n_post = 6, 8
    mask = _make_mask(n_post, n_pre, sparsity=0.3, seed=1)
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.05)
    pre = torch.rand(4, n_pre) + 0.1
    apical = torch.rand(4, n_post) + 0.2
    basal = torch.rand(4, n_post) + 0.1
    weights = torch.rand(n_post, n_pre) + 0.5               # non-zero weights
    dw = rule.delta(pre, apical, basal, weights, mask=mask)

    # Without the mask the rule would produce non-zero entries at (nearly)
    # every location; with the mask enforced, masked-off entries must be 0.
    dw_unmasked = rule.delta(pre, apical, basal, weights)
    assert (dw_unmasked[~mask] != 0.0).any(), (
        "test precondition: unmasked ΔW should be non-zero at masked-off "
        "entries so the mask is actually being tested."
    )
    assert torch.all(dw[~mask] == 0.0)
    torch.testing.assert_close(dw[mask], dw_unmasked[mask], atol=0.0, rtol=0.0)


def test_vogels_preserves_mask() -> None:
    n_pre, n_post = 6, 8
    mask = _make_mask(n_post, n_pre, sparsity=0.3, seed=2)
    rule = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.05)
    pre = torch.rand(4, n_pre) + 0.1
    post = torch.rand(4, n_post) + 0.5                     # not all == target
    weights = torch.rand(n_post, n_pre) + 0.5
    dw = rule.delta(pre, post, weights, mask=mask)
    dw_unmasked = rule.delta(pre, post, weights)
    assert (dw_unmasked[~mask] != 0.0).any()
    assert torch.all(dw[~mask] == 0.0)
    torch.testing.assert_close(dw[mask], dw_unmasked[mask], atol=0.0, rtol=0.0)


def test_three_factor_qm_preserves_mask() -> None:
    n_pre, n_post = 6, 8
    mask = _make_mask(n_post, n_pre, sparsity=0.3, seed=3)
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.05)
    cue = torch.rand(4, n_pre) + 0.1
    memory = torch.rand(4, n_post) + 0.1
    memory_error = torch.rand(4, n_post) + 0.1
    weights = torch.rand(n_post, n_pre) + 0.5
    dw = rule.delta_qm(cue, memory, memory_error, weights, mask=mask)
    dw_unmasked = rule.delta_qm(cue, memory, memory_error, weights)
    assert (dw_unmasked[~mask] != 0.0).any()
    assert torch.all(dw[~mask] == 0.0)
    torch.testing.assert_close(dw[mask], dw_unmasked[mask], atol=0.0, rtol=0.0)


def test_three_factor_mh_preserves_mask() -> None:
    n_pre, n_post = 6, 8
    mask = _make_mask(n_post, n_pre, sparsity=0.3, seed=4)
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.05)
    memory = torch.rand(4, n_pre) + 0.1
    probe_error = torch.rand(4, n_post) + 0.1
    weights = torch.rand(n_post, n_pre) + 0.5
    dw = rule.delta_mh(memory, probe_error, weights, mask=mask)
    dw_unmasked = rule.delta_mh(memory, probe_error, weights)
    assert (dw_unmasked[~mask] != 0.0).any()
    assert torch.all(dw[~mask] == 0.0)
    torch.testing.assert_close(dw[mask], dw_unmasked[mask], atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Edge case: all-False mask ⇒ ΔW entirely zero
# ---------------------------------------------------------------------------

def test_all_false_mask_gives_zero_update_urbanczik() -> None:
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.05)
    weights = torch.rand(4, 3)
    dw = rule.delta(
        pre_activity=torch.rand(2, 3), apical=torch.rand(2, 4),
        basal=torch.rand(2, 4), weights=weights,
        mask=torch.zeros(4, 3, dtype=torch.bool),
    )
    assert torch.all(dw == 0.0)


def test_all_true_mask_matches_unmasked_urbanczik() -> None:
    rule = UrbanczikSennRule(lr=0.1, weight_decay=0.05)
    pre = torch.rand(2, 3)
    apical = torch.rand(2, 4)
    basal = torch.rand(2, 4)
    weights = torch.rand(4, 3)
    dw_all_true = rule.delta(
        pre, apical, basal, weights,
        mask=torch.ones(4, 3, dtype=torch.bool),
    )
    dw_none = rule.delta(pre, apical, basal, weights)
    torch.testing.assert_close(dw_all_true, dw_none, atol=0.0, rtol=0.0)
