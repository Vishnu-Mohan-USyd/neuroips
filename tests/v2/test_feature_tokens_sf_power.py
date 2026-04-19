"""Matched radial spatial-frequency power spectrum (±5 %).

For each token we compute the 1D radial average of ``|FFT(token)|²`` over 16
concentric annuli covering [0, Nyquist]. Radial power is matched across
tokens to within ±5 % on every annulus that holds meaningful energy (i.e.
exclude the near-silent near-Nyquist bins where floor numerical noise
dominates).

Why only the "valid" bins: the Gabor-stamp construction's 4-pixel wavelength
concentrates energy in the first few annuli; the highest-frequency bins hold
essentially only DC-filter quantisation noise (~1e-20 raw power) so any
relative metric there is meaningless.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import (
    N_TOKENS,
    TokenBank,
    _radial_sf_power,
)


def _all_token_radial_spectra(bank: TokenBank) -> torch.Tensor:
    """Stack per-token radial spectra → ``[N_TOKENS, n_bins]``."""
    return torch.stack([
        _radial_sf_power(bank.tokens[t, 0]) for t in range(N_TOKENS)
    ])


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_radial_sf_matched_on_valid_bins(cfg, seed: int) -> None:
    """Relative spread on bins above ``1e-3 × max(mean)`` < 5 %."""
    bank = TokenBank(cfg, seed=seed)
    spectra = _all_token_radial_spectra(bank)                 # [12, 16]
    mean_spec = spectra.mean(dim=0)                           # [16]
    valid = mean_spec > mean_spec.max() * 1e-3
    rel = (spectra[:, valid] - mean_spec[valid]) / mean_spec[valid]
    assert float(rel.abs().max()) < 0.05, (
        f"radial SF mismatch {float(rel.abs().max()):.4f} exceeds 5 %"
    )


@pytest.mark.parametrize("seed", [0, 42])
def test_total_sf_power_matched(cfg, seed: int) -> None:
    """Sum of radial power (= total |FFT|² / bin_count) matches across tokens."""
    bank = TokenBank(cfg, seed=seed)
    spectra = _all_token_radial_spectra(bank)
    totals = spectra.sum(dim=-1)                              # [12]
    mean_total = float(totals.mean())
    rel = (totals - mean_total).abs() / mean_total
    assert float(rel.max()) < 0.05, (
        f"total radial SF power spread {float(rel.max()):.4f} exceeds 5 %"
    )


def test_radial_power_non_negative(cfg) -> None:
    """Power spectrum values are non-negative by definition."""
    bank = TokenBank(cfg, seed=0)
    spectra = _all_token_radial_spectra(bank)
    assert float(spectra.min()) >= 0.0


def test_radial_sf_reported_via_balance_statistics(cfg) -> None:
    """``balance_statistics()[t]['sf_power_radial']`` equals direct."""
    bank = TokenBank(cfg, seed=0)
    stats = bank.balance_statistics()
    for t in range(N_TOKENS):
        direct = _radial_sf_power(bank.tokens[t, 0])
        reported = stats[t]["sf_power_radial"]
        assert isinstance(reported, list)
        assert len(reported) == direct.numel()
        torch.testing.assert_close(
            torch.tensor(reported, dtype=torch.float32),
            direct,
            atol=1e-6, rtol=1e-5,
        )
