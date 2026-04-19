"""V2Network forward is deterministic given (frames, state, seed).

Two ``V2Network`` instances built with the same ``seed`` must produce
bit-identical forward output on the same input. Also: global RNG state
must not leak into the forward pass — plasticity rules rely on this
contract to reproduce trace accumulation across runs.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.network import V2Network


def _run_forward(cfg, seed: int, frames: torch.Tensor) -> tuple:
    net = V2Network(cfg, token_bank=None, seed=seed)
    state = net.initial_state(batch_size=frames.shape[0])
    return net(frames, state)


def test_same_seed_same_output(cfg):
    """Two networks with the same seed → identical x_hat and every state field."""
    torch.manual_seed(0)
    frames = torch.randn(3, 1, cfg.arch.grid_h, cfg.arch.grid_w) * 0.1
    x1, s1, info1 = _run_forward(cfg, seed=42, frames=frames)
    x2, s2, info2 = _run_forward(cfg, seed=42, frames=frames)

    torch.testing.assert_close(x1, x2, atol=0.0, rtol=0.0)
    for field in ("r_l4", "r_l23", "r_pv", "r_som", "r_h", "h_pv", "m"):
        torch.testing.assert_close(
            getattr(s1, field), getattr(s2, field), atol=0.0, rtol=0.0,
        )
    for key in info1:
        torch.testing.assert_close(info1[key], info2[key], atol=0.0, rtol=0.0)


def test_different_seed_different_output(cfg):
    """Different seeds give different forward output (non-trivial RNG coupling).

    Two forward steps are needed: step 1 activates L4 from the frame, step 2
    lets the random L2/3 / H / C weights propagate from that activation into
    the prediction. With only one step, ``state.r_l4 = 0`` going in, every
    drive collapses to zero and the seed-dependent weights are multiplied by
    zero — ``x_hat`` is constant across seeds in that degenerate case.
    """
    torch.manual_seed(0)
    frames = torch.randn(2, 2, 1, cfg.arch.grid_h, cfg.arch.grid_w) * 0.1

    def _two_steps(seed):
        net = V2Network(cfg, token_bank=None, seed=seed)
        state = net.initial_state(batch_size=2)
        _, state, _ = net(frames[:, 0], state)
        x_hat, state, _ = net(frames[:, 1], state)
        return x_hat, state

    x1, _ = _two_steps(seed=42)
    x2, _ = _two_steps(seed=123)
    assert not torch.equal(x1, x2), "different seeds must not produce identical x_hat"


def test_global_rng_does_not_leak(cfg):
    """Burning the global RNG between forward calls must not change output."""
    torch.manual_seed(0)
    frames = torch.randn(2, 1, cfg.arch.grid_h, cfg.arch.grid_w) * 0.1

    net = V2Network(cfg, token_bank=None, seed=42)
    state = net.initial_state(batch_size=2)
    x1, _, _ = net(frames, state)

    _ = torch.randn(10_000)          # burn global RNG
    torch.manual_seed(99999)
    _ = torch.randn(10_000)          # burn again

    net2 = V2Network(cfg, token_bank=None, seed=42)
    state2 = net2.initial_state(batch_size=2)
    x2, _, _ = net2(frames, state2)

    torch.testing.assert_close(x1, x2, atol=0.0, rtol=0.0)
