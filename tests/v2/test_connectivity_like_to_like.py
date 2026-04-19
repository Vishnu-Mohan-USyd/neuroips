"""Like-to-like preference: edge probability correlates positively with
feature similarity (v4 plan §Gate 7).

Construct a population with both retinotopic positions and orientation
features, generate many sparse masks with different seeds, measure the
empirical edge-presence rate per (i, j) pair, and verify Pearson
correlation with the log-propensity is positive and reasonably strong
(> 0.3).

Two complementary tests:
  (a) **Within-seed** — for a single mask, the correlation of
      is_edge[i, j] with log-propensity ranked across pairs must be
      positive (r > 0.3). This reflects the structure in the sampling.
  (b) **Cross-seed** — averaging the edge indicator over many seeds
      converges to a probability field whose correlation with the
      underlying log-propensity is strong (r > 0.5), because Monte-Carlo
      noise averages out. The bar is below 1.0 because the sampler
      normalises the propensity per row (each row draws exactly k), so
      empirical[i, j] tracks the *row-normalised* propensity rather than
      the raw log P across the full pair cloud.
"""

from __future__ import annotations

import math

import torch

from src.v2_model.connectivity import generate_sparse_mask
from src.v2_model.utils import circular_distance_abs


def _retino_feature_population(
    retino_side: int = 8, n_ori: int = 8, grid_span_px: float = 32.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A retinotopic × orientation population, tiled regularly.

    Retinotopic positions are laid out on a retino_side × retino_side grid
    across `grid_span_px`; each location has n_ori orientation-tuned units.

    Returns:
        positions: [n_units, 2] in px, where n_units = retino_side² · n_ori.
        features: [n_units] in degrees ∈ [0, 180).
    """
    n_units = retino_side * retino_side * n_ori
    spacing = grid_span_px / retino_side
    # All retinotopic positions replicated n_ori times per cell.
    coords = torch.arange(retino_side, dtype=torch.float32) * spacing
    ry, rx = torch.meshgrid(coords, coords, indexing="ij")
    rx = rx.reshape(-1).repeat_interleave(n_ori)  # [n_units]
    ry = ry.reshape(-1).repeat_interleave(n_ori)
    positions = torch.stack([rx, ry], dim=-1)  # [n_units, 2]

    thetas_deg = torch.arange(n_ori, dtype=torch.float32) * (180.0 / n_ori)
    features = thetas_deg.repeat(retino_side * retino_side)  # [n_units]

    return positions, features


def _log_propensity(
    positions: torch.Tensor, features: torch.Tensor,
    sigma_position: float, sigma_feature: float,
    feature_period_deg: float = 180.0,
) -> torch.Tensor:
    """Return log P_unnormalised[i, j] for test scoring."""
    d = torch.cdist(positions, positions, p=2.0)
    d2 = d * d
    delta = circular_distance_abs(
        features.unsqueeze(1), features.unsqueeze(0), period=feature_period_deg
    )
    log_p = (
        -d2 / (2.0 * sigma_position * sigma_position)
        - delta * delta / (2.0 * sigma_feature * sigma_feature)
    )
    return log_p


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two flat tensors of equal length."""
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum().item()
    den = (a.norm() * b.norm()).item()
    if den == 0.0:
        return 0.0
    return num / den


def _off_diag_mask(n: int) -> torch.Tensor:
    """Boolean [n, n] mask that is True off the diagonal (excludes self)."""
    m = torch.ones(n, n, dtype=torch.bool)
    m.fill_diagonal_(False)
    return m


def test_within_seed_like_to_like_correlation() -> None:
    """Single mask: edge indicator correlates positively with log-propensity.

    A comfortable margin (r > 0.3) is required by the plan v4 Gate 7.
    """
    positions, features = _retino_feature_population(
        retino_side=8, n_ori=8, grid_span_px=32.0
    )
    n_units = positions.shape[0]

    sigma_position, sigma_feature = 4.0, 25.0
    mask = generate_sparse_mask(
        positions=positions, features=features, n_units=n_units,
        sparsity=0.12, sigma_position=sigma_position,
        sigma_feature=sigma_feature, seed=42,
    )
    log_p = _log_propensity(
        positions, features, sigma_position, sigma_feature
    )
    off_diag = _off_diag_mask(n_units)
    r = _pearson(mask[off_diag], log_p[off_diag])
    assert r > 0.3, (
        f"within-seed like-to-like correlation r={r:.3f} ≤ 0.3 "
        f"(plan v4 Gate 7 threshold)."
    )


def test_cross_seed_empirical_probability_tracks_log_propensity() -> None:
    """Averaging edge indicators over many seeds → strong correlation."""
    positions, features = _retino_feature_population(
        retino_side=6, n_ori=8, grid_span_px=24.0
    )
    n_units = positions.shape[0]
    sigma_position, sigma_feature = 4.0, 25.0

    n_seeds = 64
    edge_count = torch.zeros(n_units, n_units)
    for s in range(n_seeds):
        mask = generate_sparse_mask(
            positions=positions, features=features, n_units=n_units,
            sparsity=0.12, sigma_position=sigma_position,
            sigma_feature=sigma_feature, seed=s,
        )
        edge_count = edge_count + mask.float()
    empirical_p = edge_count / float(n_seeds)

    log_p = _log_propensity(
        positions, features, sigma_position, sigma_feature
    )
    off_diag = _off_diag_mask(n_units)
    r = _pearson(empirical_p[off_diag], log_p[off_diag])
    assert r > 0.5, (
        f"cross-seed empirical-P vs log-propensity correlation "
        f"r={r:.3f} ≤ 0.5 after {n_seeds} seeds."
    )


def test_closer_pairs_more_likely_than_far_pairs() -> None:
    """Monotonicity check: high-similarity quartile has higher edge rate
    than the low-similarity quartile. Doesn't depend on assumed Pearson
    form — more robust to any non-linearity in the sampling."""
    positions, features = _retino_feature_population(
        retino_side=8, n_ori=8, grid_span_px=32.0
    )
    n_units = positions.shape[0]
    sigma_position, sigma_feature = 4.0, 25.0

    n_seeds = 32
    edge_count = torch.zeros(n_units, n_units)
    for s in range(n_seeds):
        mask = generate_sparse_mask(
            positions=positions, features=features, n_units=n_units,
            sparsity=0.12, sigma_position=sigma_position,
            sigma_feature=sigma_feature, seed=100 + s,
        )
        edge_count = edge_count + mask.float()
    empirical_p = edge_count / float(n_seeds)

    log_p = _log_propensity(
        positions, features, sigma_position, sigma_feature
    )
    off_diag = _off_diag_mask(n_units)
    emp_flat = empirical_p[off_diag]
    lp_flat = log_p[off_diag]

    # Stratify by quartile of log-propensity.
    q75 = torch.quantile(lp_flat, 0.75)
    q25 = torch.quantile(lp_flat, 0.25)
    high_sim_rate = emp_flat[lp_flat >= q75].mean().item()
    low_sim_rate = emp_flat[lp_flat <= q25].mean().item()
    assert high_sim_rate > low_sim_rate, (
        f"high-similarity quartile rate={high_sim_rate:.4f} ≤ "
        f"low-similarity quartile rate={low_sim_rate:.4f}; "
        f"like-to-like preference broken."
    )


def test_feature_only_mode_produces_positive_correlation() -> None:
    """With positions=None, orientation-only mode still gives r > 0.3
    on the feature-similarity axis."""
    n_ori = 16
    # Keep it simple: one retinotopic location × many orientations.
    features = torch.linspace(0.0, 180.0, steps=n_ori + 1)[:-1]
    # Replicate so there are enough per-feature partners.
    n_units = 64
    features = features.repeat(n_units // n_ori)[:n_units]

    mask = generate_sparse_mask(
        positions=None, features=features, n_units=n_units,
        sparsity=0.15, sigma_feature=25.0, seed=2024,
    )
    delta = circular_distance_abs(
        features.unsqueeze(1), features.unsqueeze(0), period=180.0
    )
    log_p = -delta * delta / (2.0 * 25.0 * 25.0)

    off_diag = _off_diag_mask(n_units)
    r = _pearson(mask[off_diag], log_p[off_diag])
    assert r > 0.3, (
        f"feature-only like-to-like correlation r={r:.3f} ≤ 0.3."
    )
