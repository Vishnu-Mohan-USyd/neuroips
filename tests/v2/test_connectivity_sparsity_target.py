"""Sparsity tolerance: achieved density ≤ ±1 % of the target.

Per v4 spec §Connectivity: target sparsity ≈ 12 %. We verify that for a
range of target sparsities {5 %, 12 %, 20 %} the mask's achieved density is
within a tight tolerance of the target.

Design note: since the generator uses exact per-row multinomial sampling
without replacement, the achieved density is
    k · n / n² = k / n
where k = round(sparsity · n). For the `(n_units, sparsity)` pairs we
test, |achieved − target| ≤ 1 / n, which is well within the 1 % tolerance
at n = 256 (≤ 0.4 %).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.v2_model.connectivity import generate_sparse_mask


@pytest.mark.parametrize("target_sparsity", [0.05, 0.12, 0.20])
def test_uniform_mode_sparsity_tolerance(target_sparsity: float) -> None:
    """In the uniform (positions=features=None) mode, density is exact k/n."""
    n = 256
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n,
        sparsity=target_sparsity, seed=0
    )
    achieved = mask.float().mean().item()
    assert abs(achieved - target_sparsity) < 0.01, (
        f"target={target_sparsity}, achieved={achieved:.4f}, "
        f"delta={abs(achieved - target_sparsity):.4f} > 0.01 tolerance"
    )


@pytest.mark.parametrize("target_sparsity", [0.05, 0.12, 0.20])
def test_retinotopic_mode_sparsity_tolerance(target_sparsity: float) -> None:
    """Same tolerance for the full retinotopy + feature-structured mode."""
    n = 256
    positions = torch.rand(n, 2) * 32.0
    features = torch.rand(n) * 180.0
    mask = generate_sparse_mask(
        positions=positions, features=features, n_units=n,
        sparsity=target_sparsity,
        sigma_position=4.0, sigma_feature=25.0, seed=1
    )
    achieved = mask.float().mean().item()
    assert abs(achieved - target_sparsity) < 0.01, (
        f"target={target_sparsity}, achieved={achieved:.4f}"
    )


def test_per_row_density_constant_across_rows() -> None:
    """Per-row density is constant (exact-k sampling), so the histogram
    of row-degrees is a delta at k — variance should be zero."""
    n = 256
    target_sparsity = 0.12
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n,
        sparsity=target_sparsity, seed=3
    )
    row_degrees = mask.sum(dim=1).float()
    assert row_degrees.std().item() == 0.0, (
        f"row-degree stddev > 0: {row_degrees.std().item():.4f}"
    )
    # And it should equal round(sparsity · n):
    expected_k = round(target_sparsity * n)
    assert row_degrees[0].item() == expected_k


def test_edge_case_small_n_sparsity_tight() -> None:
    """Small n: k is clamped to [1, n-1]; density = k/n."""
    # sparsity=0.5 with n=10 → k=5, density=0.5, delta=0.0.
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=10, sparsity=0.5, seed=0
    )
    achieved = mask.float().mean().item()
    assert math.isclose(achieved, 0.5, abs_tol=1e-6)


def test_edge_case_sparsity_rounds_to_zero_k_is_clamped_to_one() -> None:
    """sparsity · n < 0.5 → round() → 0; generator must clamp to k=1."""
    # n=4, sparsity=0.1 → round(0.4) = 0; clamped to 1 edge per row.
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=4, sparsity=0.1, seed=0
    )
    row_degrees = mask.sum(dim=1)
    assert torch.all(row_degrees == 1)


def test_edge_case_sparsity_near_one_k_is_clamped_to_n_minus_one() -> None:
    """sparsity · n ≥ n → k clamped to n-1 (can't connect to self)."""
    # n=10, sparsity=0.99 → round(9.9) = 10; clamped to 9.
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=10, sparsity=0.99, seed=0
    )
    row_degrees = mask.sum(dim=1)
    assert torch.all(row_degrees == 9)
    # Diagonal must remain False.
    assert not mask.diagonal().any()
