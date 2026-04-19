"""LOAD-BEARING: matched orientation-energy histogram through the fixed LGN/
Gabor bank (±5 %).

For each token we compute the space-summed Gabor-quadrature energy
``Σ_{h,w} √(E_k² + O_k²)`` at every orientation k ∈ {0..N_ori-1}, using the
same DC-balanced Gabor kernels that ``LGNL4FrontEnd`` uses. The per-
orientation relative spread across the 12 tokens must be below 5 % — this is
the core matched-stimulus invariant the Richter-like assay relies on.

The construction achieves this via:
  * Point-symmetric diagonal position pairs (same-orientation stamps land at
    maximum spatial separation → near-zero same-ori cross-terms).
  * Per-(token, orientation) Adam optimisation that minimises the squared
    relative deviation from the cross-token-mean histogram; the module keeps
    the best-seen amplitude table because the loss is non-convex.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import (
    N_ORI,
    N_TOKENS,
    TokenBank,
    _build_lgn_gabor_bank,
    _orientation_energy_histogram,
)


def _per_token_orientation_histogram(bank: TokenBank) -> torch.Tensor:
    """[N_TOKENS, N_ORI] Gabor-quadrature energy summed over space."""
    evens, odds = _build_lgn_gabor_bank()
    return _orientation_energy_histogram(bank.tokens, evens, odds)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_orientation_energy_matched_within_5pct(cfg, seed: int) -> None:
    """Worst-token relative deviation per orientation is below 5 %."""
    bank = TokenBank(cfg, seed=seed)
    hist = _per_token_orientation_histogram(bank)            # [12, 8]
    mean_hist = hist.mean(dim=0, keepdim=True)               # [1, 8]
    rel = (hist - mean_hist) / mean_hist
    assert float(rel.abs().max()) < 0.05, (
        f"ori-energy spread {float(rel.abs().max()):.4f} exceeds 5 % "
        f"(worst ori {int(rel.abs().max(dim=0).values.argmax())})"
    )


@pytest.mark.parametrize("seed", [0])
def test_histogram_shape(cfg, seed: int) -> None:
    """Shape of the direct orientation-energy histogram."""
    bank = TokenBank(cfg, seed=seed)
    hist = _per_token_orientation_histogram(bank)
    assert hist.shape == (N_TOKENS, N_ORI)


def test_orientation_energy_non_negative(cfg) -> None:
    """Quadrature energy is non-negative by construction."""
    bank = TokenBank(cfg, seed=0)
    hist = _per_token_orientation_histogram(bank)
    assert float(hist.min()) >= 0.0


def test_ori_hist_via_balance_statistics(cfg) -> None:
    """``balance_statistics()[t]['orientation_energy_hist']`` == direct histogram."""
    bank = TokenBank(cfg, seed=0)
    stats = bank.balance_statistics()
    hist = _per_token_orientation_histogram(bank)
    for t in range(N_TOKENS):
        reported = stats[t]["orientation_energy_hist"]
        assert isinstance(reported, list)
        assert len(reported) == N_ORI
        torch.testing.assert_close(
            torch.tensor(reported, dtype=torch.float32),
            hist[t],
            atol=1e-4, rtol=1e-4,
        )
