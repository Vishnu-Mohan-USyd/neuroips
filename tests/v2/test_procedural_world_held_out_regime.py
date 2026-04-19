"""Train-mode ``held_out_regime`` removes a regime from the transition
support entirely.

Over 10 000 steps, the held-out regime must never appear — neither in the
initial reset draw nor in any transition. This is what makes the held-out
regime genuinely out-of-distribution during training and available as an
eval-time novel context.
"""

from __future__ import annotations

import pytest

from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world import REGIMES, ProceduralWorld


@pytest.fixture(scope="module")
def bank():
    from src.v2_model.config import ModelConfig
    return TokenBank(ModelConfig(), seed=0)


@pytest.mark.parametrize("held_out", list(REGIMES))
def test_held_out_never_appears(cfg, bank, held_out: str) -> None:
    """Over 10 000 steps, held-out regime is absent from every state."""
    world = ProceduralWorld(
        cfg, bank, seed_family="train", held_out_regime=held_out,
    )
    state = world.reset(42)
    assert state.regime != held_out, (
        f"reset drew the held-out regime {held_out}"
    )
    for _ in range(10_000):
        _, state, info = world.step(state)
        assert state.regime != held_out, (
            f"step produced held-out regime {held_out}"
        )
        assert info["regime"] != held_out


def test_held_out_exposes_three_regimes(cfg, bank) -> None:
    """The other three regimes do appear (non-trivial support)."""
    world = ProceduralWorld(
        cfg, bank, seed_family="train", held_out_regime="high-hazard",
    )
    state = world.reset(42)
    seen: set[str] = {state.regime}
    for _ in range(5000):
        _, state, _ = world.step(state)
        seen.add(state.regime)
    assert seen == {"CW-drift", "CCW-drift", "low-hazard"}


def test_eval_family_sees_all_regimes(cfg, bank) -> None:
    """Eval family (no held-out) eventually visits every regime."""
    world = ProceduralWorld(cfg, bank, seed_family="eval")
    state = world.reset(0)
    seen: set[str] = {state.regime}
    for _ in range(5000):
        _, state, _ = world.step(state)
        seen.add(state.regime)
    assert seen == set(REGIMES)
