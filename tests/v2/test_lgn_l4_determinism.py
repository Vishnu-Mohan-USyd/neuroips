"""Determinism of `LGNL4FrontEnd.forward`.

Per v4 spec §Learning rules — no autograd in main path; tests must be
bit-exact reproducible. The front end contains no RNG and no non-deterministic
ops (all conv2d, avg_pool2d, reshape, elementwise arithmetic, softplus).

This file nails that invariant down:

  * Two calls with the same `frames` and the same `state` produce identical
    outputs (bit-exact — atol=0, rtol=0).
  * Two independently constructed modules with the same `cfg` produce identical
    outputs on the same input.
  * Successive timesteps are a pure function of `(frames_t, state_{t-1})` —
    swapping the state resets the trajectory to the start.
"""

from __future__ import annotations

import torch

from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.state import initial_state


def test_forward_is_bit_exact_reproducible(cfg, batch_size) -> None:
    """Same input + same state → bit-exact same output on repeated calls."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_1, r_l4_1, _ = front(frames, state)
    lgn_2, r_l4_2, _ = front(frames, state)

    torch.testing.assert_close(lgn_1, lgn_2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(r_l4_1, r_l4_2, atol=0.0, rtol=0.0)


def test_two_instances_same_cfg_agree(cfg, batch_size) -> None:
    """Independently constructed modules with the same cfg agree bit-exactly."""
    front_a = LGNL4FrontEnd(cfg)
    front_b = LGNL4FrontEnd(cfg)

    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_a, r_l4_a, _ = front_a(frames, state)
    lgn_b, r_l4_b, _ = front_b(frames, state)

    torch.testing.assert_close(lgn_a, lgn_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(r_l4_a, r_l4_b, atol=0.0, rtol=0.0)


def test_forward_is_pure_function_of_state_and_frames(cfg, batch_size) -> None:
    """forward does not maintain hidden state: resetting (frames, state) resets output."""
    front = LGNL4FrontEnd(cfg)
    state0 = initial_state(cfg, batch_size=batch_size)
    frames_a = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)
    frames_b = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_a, r_l4_a, state_a = front(frames_a, state0)

    # Run something else...
    _ = front(frames_b, state_a)
    # ...then re-run the original. Must reproduce the first call exactly.
    lgn_a2, r_l4_a2, _ = front(frames_a, state0)

    torch.testing.assert_close(lgn_a, lgn_a2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(r_l4_a, r_l4_a2, atol=0.0, rtol=0.0)


def test_different_inputs_produce_different_outputs(cfg, batch_size) -> None:
    """Sanity: non-degenerate forward. Two distinct inputs give distinct outputs.

    Without this, "determinism" could be vacuously satisfied by a constant
    output. Uses a non-zero contrast pattern so the DC-balanced filters
    actually respond.
    """
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)

    # Two very different patterns: impulse vs anti-impulse.
    frames_a = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)
    frames_a[:, :, cfg.arch.grid_h // 2, cfg.arch.grid_w // 2] = 1.0

    frames_b = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)
    frames_b[:, :, 0, 0] = 1.0

    _, r_l4_a, _ = front(frames_a, state)
    _, r_l4_b, _ = front(frames_b, state)

    assert not torch.allclose(r_l4_a, r_l4_b, atol=1e-6), (
        "distinct inputs produced identical l4 rates — forward is degenerate."
    )
