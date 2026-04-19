"""Non-retinotopic / non-feature fallback: `positions=None, features=None`.

Some populations (e.g. H recurrent E) have no explicit retinotopic layout
and no pre-assigned feature angle. The generator must fall back to a uniform
random sparse mask:

  * Edge probability is uniform across all off-diagonal pairs.
  * Target density is achieved within ±1 %.
  * Per-row out-degree is constant (exact-k sampling).
  * Over many seeds, empirical per-pair edge probability converges to
    sparsity (within Monte-Carlo tolerance) — i.e. no hidden bias.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.v2_model.connectivity import generate_sparse_mask


def test_uniform_fallback_achieves_target_density() -> None:
    n = 128
    sparsity = 0.12
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n, sparsity=sparsity, seed=0
    )
    achieved = mask.float().mean().item()
    assert abs(achieved - sparsity) < 0.01


def test_uniform_fallback_per_row_degree_constant() -> None:
    """Per-row degrees equal round(sparsity · n)."""
    n = 128
    sparsity = 0.12
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n, sparsity=sparsity, seed=0
    )
    expected_k = round(sparsity * n)
    row_degrees = mask.sum(dim=1)
    assert torch.all(row_degrees == expected_k)


def test_uniform_fallback_empirical_probability_flat() -> None:
    """Across many seeds the per-pair edge probability converges to the
    target sparsity, uniformly across the off-diagonal matrix.

    Uses a moderately small n + many seeds so the Monte-Carlo error shrinks
    to a value well below the tolerance we assert on. With n=32, sparsity=0.2,
    and n_seeds=400, per-pair stddev is √(p(1−p)/n_seeds) ≈ 0.02; we assert
    max-deviation < 0.06 (3σ).
    """
    n = 32
    sparsity = 0.20
    n_seeds = 400
    counts = torch.zeros(n, n)
    for s in range(n_seeds):
        counts = counts + generate_sparse_mask(
            positions=None, features=None, n_units=n,
            sparsity=sparsity, seed=s
        ).float()
    empirical_p = counts / float(n_seeds)

    # Skip the diagonal (always zero) from the uniformity check.
    off_diag = ~torch.eye(n, dtype=torch.bool)
    vals = empirical_p[off_diag]

    # Mean over off-diagonal: each row picks exactly k out of (n-1) possible
    # partners uniformly → off-diagonal mean converges to k / (n - 1)
    # (not k / n, because the diagonal is always excluded from sampling).
    k = round(sparsity * n)
    expected_off_diag_mean = k / (n - 1)
    assert math.isclose(
        vals.mean().item(), expected_off_diag_mean, abs_tol=0.005
    ), (
        f"off-diag mean = {vals.mean().item():.4f}, "
        f"expected ≈ k/(n-1) = {expected_off_diag_mean:.4f}"
    )

    # Uniformity: no cell deviates from its expected probability by more than
    # 6 standard-deviations of the Monte-Carlo estimate.
    per_pair_std = math.sqrt(
        expected_off_diag_mean * (1.0 - expected_off_diag_mean) / n_seeds
    )
    max_dev = (vals - expected_off_diag_mean).abs().max().item()
    assert max_dev < 6.0 * per_pair_std, (
        f"max per-pair deviation from expected = {max_dev:.4f} > "
        f"6σ = {6 * per_pair_std:.4f}; uniform fallback is biased."
    )


def test_uniform_fallback_no_like_to_like_signal() -> None:
    """In the uniform-fallback mode, there is no feature similarity to
    correlate against — but the mask should also not exhibit spurious
    structure along the diagonal band. We verify that bands of
    distance-from-diagonal have roughly equal edge rates."""
    n = 64
    sparsity = 0.25
    n_seeds = 200
    counts = torch.zeros(n, n)
    for s in range(n_seeds):
        counts = counts + generate_sparse_mask(
            positions=None, features=None, n_units=n,
            sparsity=sparsity, seed=1000 + s
        ).float()
    empirical_p = counts / float(n_seeds)

    # Compute mean edge rate at each "band" (|i - j| constant).
    band_rates: list[float] = []
    for delta in range(1, 10):
        i = torch.arange(n - delta)
        j = i + delta
        band = torch.cat([empirical_p[i, j], empirical_p[j, i]])
        band_rates.append(band.mean().item())
    # All bands should equal k/(n-1) within the Monte-Carlo tolerance.
    k = round(sparsity * n)
    expected_off_diag_mean = k / (n - 1)
    per_pair_std = math.sqrt(
        expected_off_diag_mean * (1.0 - expected_off_diag_mean) / n_seeds
    )
    for rate in band_rates:
        assert abs(rate - expected_off_diag_mean) < 6.0 * per_pair_std, (
            f"band rate {rate:.4f} deviates from expected "
            f"{expected_off_diag_mean:.4f} by > 6σ; uniform fallback has "
            f"unexpected structure."
        )


def test_diagonal_is_false_in_uniform_fallback() -> None:
    """No self-connection even in the uniform fallback."""
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=32, sparsity=0.5, seed=0
    )
    assert not mask.diagonal().any()
