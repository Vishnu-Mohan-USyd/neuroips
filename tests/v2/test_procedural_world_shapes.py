"""Shapes, dtype, and range of procedural-world outputs.

Every frame is a ``[1, 32, 32]`` float32 image in [0, 1], and every
:class:`WorldState` carries all seven latent fields.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world import REGIMES, ProceduralWorld, WorldState


@pytest.fixture
def world(cfg):
    bank = TokenBank(cfg, seed=0)
    return ProceduralWorld(cfg, bank, seed_family="train")


def test_reset_returns_world_state(world: ProceduralWorld) -> None:
    """``reset`` returns a :class:`WorldState` with all fields populated."""
    state = world.reset(0)
    assert isinstance(state, WorldState)
    for field in ("z", "theta", "position", "contrast",
                  "occluder", "regime", "step_idx"):
        assert hasattr(state, field)
    assert 0 <= state.z < 12
    assert 0.0 <= state.theta < 360.0
    py, px = state.position
    assert 0 <= py < 32 and 0 <= px < 32
    assert 0.1 <= state.contrast <= 1.0
    assert state.occluder in (0, 1)
    assert state.regime in REGIMES
    assert state.step_idx == 0


def test_step_frame_shape_dtype_range(world: ProceduralWorld) -> None:
    """Step frame is [1, 32, 32] float32 in [0, 1]."""
    state = world.reset(0)
    frame, state, info = world.step(state)
    assert frame.shape == (1, 32, 32)
    assert frame.dtype == torch.float32
    assert float(frame.min()) >= 0.0
    assert float(frame.max()) <= 1.0
    assert state.step_idx == 1
    assert info["regime"] in REGIMES
    assert isinstance(info["is_jump"], bool)


@pytest.mark.parametrize("n", [1, 5, 100])
def test_trajectory_shape(world: ProceduralWorld, n: int) -> None:
    """``trajectory(seed, n_steps)`` returns ``[n, 1, 32, 32]``."""
    frames, states = world.trajectory(0, n)
    assert frames.shape == (n, 1, 32, 32)
    assert frames.dtype == torch.float32
    assert float(frames.min()) >= 0.0
    assert float(frames.max()) <= 1.0
    assert len(states) == n
    for s in states:
        assert isinstance(s, WorldState)


def test_step_info_keys(world: ProceduralWorld) -> None:
    """``info`` dict carries the ground-truth latents + ``is_jump``."""
    state = world.reset(0)
    _, _, info = world.step(state)
    required = {"z", "theta", "position", "contrast", "occluder",
                "regime", "is_jump", "step_idx"}
    assert required.issubset(info.keys())


def test_step_before_reset_raises(world: ProceduralWorld) -> None:
    """A ``step`` call before :meth:`reset` errors explicitly."""
    fresh = ProceduralWorld(
        world.cfg, world.token_bank, seed_family="train",
    )
    dummy = WorldState(
        z=0, theta=0.0, position=(0, 0), contrast=0.5,
        occluder=0, regime=REGIMES[0], step_idx=0,
    )
    with pytest.raises(RuntimeError):
        fresh.step(dummy)


def test_seed_family_validation(cfg) -> None:
    """Bad ``seed_family`` raises; ``held_out_regime`` only legal with train."""
    bank = TokenBank(cfg, seed=0)
    with pytest.raises(ValueError):
        ProceduralWorld(cfg, bank, seed_family="foo")
    with pytest.raises(ValueError):
        ProceduralWorld(cfg, bank, seed_family="eval",
                        held_out_regime="CW-drift")
    with pytest.raises(ValueError):
        ProceduralWorld(cfg, bank, seed_family="train",
                        held_out_regime="nonsense-regime")
