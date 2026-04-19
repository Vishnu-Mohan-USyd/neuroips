"""LOAD-BEARING: tokens remain linearly discriminable through the LGN/L4
front end (verify_discriminability ≥ 0.25 on 12-way LinearSVC).

Chance level for a balanced 12-way classification is ``1/12 ≈ 0.083``; the
spec requires accuracy ≥ 0.25 (3× chance) on a held-out 30 % split of 200
noise-augmented samples per token. The construction achieves this through
the per-token (pair → orientation) permutation of the 4×4 retinotopic grid:
each token has a distinct retinotopic orientation map, so the 128-dim L4 E
rate vector discriminates them even though the spatially-summed orientation
histogram is matched to within 5 %.
"""

from __future__ import annotations

import pytest

from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.stimuli.feature_tokens import TokenBank

sklearn = pytest.importorskip("sklearn")                    # noqa: F841


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_discriminability_above_floor(cfg, seed: int) -> None:
    """12-way LinearSVC accuracy on L4 E rate features ≥ 0.25."""
    bank = TokenBank(cfg, seed=seed)
    lgn = LGNL4FrontEnd(cfg)
    acc = bank.verify_discriminability(lgn, n_noise_samples=200, seed=seed)
    assert acc >= 0.25, (
        f"discriminability {acc:.4f} < 0.25 (chance = 0.083)"
    )


def test_discriminability_returns_float(cfg) -> None:
    """``verify_discriminability`` returns a scalar Python float in [0, 1]."""
    bank = TokenBank(cfg, seed=0)
    lgn = LGNL4FrontEnd(cfg)
    acc = bank.verify_discriminability(lgn, n_noise_samples=200, seed=0)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_discriminability_well_above_chance(cfg) -> None:
    """A healthy bank should land well above the 3× chance floor — in fact
    closer to 1.0 on the default config. This catches silent collapses
    (e.g. rotation equivariance masking tokens) that technically scrape past
    0.25 but indicate a bug."""
    bank = TokenBank(cfg, seed=0)
    lgn = LGNL4FrontEnd(cfg)
    acc = bank.verify_discriminability(lgn, n_noise_samples=200, seed=0)
    assert acc >= 0.8, (
        f"discriminability {acc:.4f} — suspiciously low for the matched-"
        f"texture bank (expected ≳0.9)"
    )
