"""Train-family and eval-family trajectories at the same index must differ.

Train family uses seed base 42; eval family uses seed base 9000. A
``trajectory_seed`` of ``k`` therefore maps to seeds 42+k vs 9000+k, which
drive different ``torch.Generator`` states and distinct trajectories.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world import ProceduralWorld


@pytest.fixture(scope="module")
def bank():
    from src.v2_model.config import ModelConfig
    return TokenBank(ModelConfig(), seed=0)


@pytest.mark.parametrize("trajectory_seed", [0, 1, 7, 42, 100])
def test_train_eval_frames_differ(cfg, bank, trajectory_seed: int) -> None:
    """Same trajectory_seed, different seed_family → different frames."""
    train = ProceduralWorld(cfg, bank, seed_family="train")
    eval_ = ProceduralWorld(cfg, bank, seed_family="eval")
    f_tr, _ = train.trajectory(trajectory_seed, n_steps=50)
    f_ev, _ = eval_.trajectory(trajectory_seed, n_steps=50)
    assert not torch.equal(f_tr, f_ev), (
        f"train family and eval family produced identical frames at "
        f"trajectory_seed={trajectory_seed}"
    )
    # Non-trivial difference: distinguishable per-pixel
    mean_abs_diff = float((f_tr - f_ev).abs().mean())
    assert mean_abs_diff > 1e-3, (
        f"train/eval frames differ but by only {mean_abs_diff:.6f} — "
        f"trajectories are statistically indistinguishable"
    )


def test_train_eval_regime_sequences_differ(cfg, bank) -> None:
    """Regime sequences should diverge within the first ~50 steps."""
    train = ProceduralWorld(cfg, bank, seed_family="train")
    eval_ = ProceduralWorld(cfg, bank, seed_family="eval")
    _, s_tr = train.trajectory(0, n_steps=100)
    _, s_ev = eval_.trajectory(0, n_steps=100)
    regimes_tr = [s.regime for s in s_tr]
    regimes_ev = [s.regime for s in s_ev]
    assert regimes_tr != regimes_ev, (
        "train and eval regime sequences coincide at trajectory_seed=0"
    )


def test_different_trajectory_seeds_differ_within_family(cfg, bank) -> None:
    """Within the train family, two distinct trajectory seeds diverge."""
    train = ProceduralWorld(cfg, bank, seed_family="train")
    f0, _ = train.trajectory(0, n_steps=50)
    f1, _ = train.trajectory(1, n_steps=50)
    assert not torch.equal(f0, f1)
