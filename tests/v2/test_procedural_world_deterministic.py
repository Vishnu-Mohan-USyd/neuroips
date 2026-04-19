"""Deterministic reconstruction from ``trajectory_seed``.

Two ``ProceduralWorld`` instances built over the same token bank and
driven with the same trajectory seed must emit bit-identical frame
tensors (``torch.equal``) and identical regime sequences.

Concretely: :meth:`reset` creates a fresh :class:`torch.Generator` from
``seed_base + trajectory_seed``, and every random draw inside
:meth:`step` goes through that generator — so the global RNG state does
not leak into the trajectory.
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


@pytest.mark.parametrize("trajectory_seed", [0, 1, 7, 42, 1234])
def test_same_trajectory_seed_reproduces_exactly(
    cfg, bank, trajectory_seed: int,
) -> None:
    """Two worlds, same trajectory seed → identical frames."""
    w1 = ProceduralWorld(cfg, bank, seed_family="train")
    w2 = ProceduralWorld(cfg, bank, seed_family="train")
    f1, s1 = w1.trajectory(trajectory_seed, n_steps=60)
    f2, s2 = w2.trajectory(trajectory_seed, n_steps=60)
    torch.testing.assert_close(f1, f2, atol=0.0, rtol=0.0)
    regimes1 = [s.regime for s in s1]
    regimes2 = [s.regime for s in s2]
    assert regimes1 == regimes2
    latents1 = [(s.z, s.theta, s.position, s.contrast, s.occluder) for s in s1]
    latents2 = [(s.z, s.theta, s.position, s.contrast, s.occluder) for s in s2]
    assert latents1 == latents2


def test_different_trajectory_seeds_differ(cfg, bank) -> None:
    """Distinct trajectory seeds → distinct frames."""
    w = ProceduralWorld(cfg, bank, seed_family="train")
    f0, _ = w.trajectory(0, n_steps=50)
    f1, _ = w.trajectory(1, n_steps=50)
    assert not torch.equal(f0, f1)


def test_rng_isolation_from_global_state(cfg, bank) -> None:
    """Global RNG state must not perturb per-trajectory frames."""
    w = ProceduralWorld(cfg, bank, seed_family="train")

    torch.manual_seed(12345)
    f_a, _ = w.trajectory(0, n_steps=30)

    # Burn the global RNG
    _ = torch.randn(1000)
    torch.manual_seed(99999)
    _ = torch.randn(1000)

    # Same trajectory_seed → identical frames regardless of global RNG
    f_b, _ = w.trajectory(0, n_steps=30)
    torch.testing.assert_close(f_a, f_b, atol=0.0, rtol=0.0)


def test_step_sequence_is_deterministic(cfg, bank) -> None:
    """Stepping through :meth:`step` gives the same sequence as
    :meth:`trajectory`."""
    w1 = ProceduralWorld(cfg, bank, seed_family="train")
    w2 = ProceduralWorld(cfg, bank, seed_family="train")
    f_traj, _ = w1.trajectory(7, n_steps=20)

    state = w2.reset(7)
    frames_manual = [w2.render(state)]
    for _ in range(19):
        frame, state, _ = w2.step(state)
        frames_manual.append(frame)
    f_manual = torch.stack(frames_manual)
    torch.testing.assert_close(f_traj, f_manual, atol=0.0, rtol=0.0)
