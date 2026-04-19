"""Matched total edge energy via Sobel-gradient magnitude (±5 %).

``total_edge_energy(t) = Σ_{h,w} √(G_x² + G_y²)`` where G_x, G_y are 3×3
Sobel-filter responses. This is a standard 'edge content' summary that
complements the band-limited Gabor histogram. The construction inherits
edge-energy matching from the RMS + orientation-balance matching (Sobel
is an isotropic local-gradient filter correlated with total contrast +
per-orientation energies), so the test tolerance of 5 % is comfortably met.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import (
    N_TOKENS,
    TokenBank,
    _edge_energy,
)


def _per_token_edges(bank: TokenBank) -> torch.Tensor:
    """Return ``[N_TOKENS]`` scalar Sobel-energy per token."""
    return torch.stack([
        _edge_energy(bank.tokens[t, 0]) for t in range(N_TOKENS)
    ])


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_edge_energy_matched_within_5pct(cfg, seed: int) -> None:
    """Relative spread of total Sobel edge energy across tokens < 5 %."""
    bank = TokenBank(cfg, seed=seed)
    edges = _per_token_edges(bank)
    mean_edges = float(edges.mean())
    rel = (edges - mean_edges).abs() / mean_edges
    assert float(rel.max()) < 0.05, (
        f"edge-energy spread {float(rel.max()):.4f} exceeds 5 %"
    )


def test_edge_energy_positive(cfg) -> None:
    """Every token has non-trivial edge content."""
    bank = TokenBank(cfg, seed=0)
    edges = _per_token_edges(bank)
    assert float(edges.min()) > 0.0


def test_edge_energy_is_float_scalar_per_token(cfg) -> None:
    """``_edge_energy(token)`` must return a scalar Tensor."""
    bank = TokenBank(cfg, seed=0)
    eg = _edge_energy(bank.tokens[0, 0])
    assert eg.dim() == 0


def test_edge_energy_via_balance_statistics(cfg) -> None:
    """``balance_statistics()[t]['total_edge_energy']`` matches direct."""
    bank = TokenBank(cfg, seed=0)
    stats = bank.balance_statistics()
    edges = _per_token_edges(bank)
    for t in range(N_TOKENS):
        reported = stats[t]["total_edge_energy"]
        assert isinstance(reported, float)
        assert abs(reported - float(edges[t])) < 1e-4
