"""Multi-step rollout stays finite and informative.

Without training, the recurrent E/I weights are random and the circuit
is not yet spectrally stabilised — rates can grow over a long rollout.
Phase 2 plasticity + homeostasis will pin the steady-state firing
rate; this test only asserts what must hold *before* training:

* All rates remain finite (no NaN/inf) over a short rollout.
* The L2/3 E state changes between consecutive steps (non-degenerate).
* Different frame sequences give different end-of-rollout L4 rates.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def test_5_step_rollout_stays_finite(cfg, net):
    """No NaN/inf over a short rollout; r_l23 evolves across steps."""
    a = cfg.arch
    torch.manual_seed(0)
    frames = torch.randn(2, 5, 1, a.grid_h, a.grid_w) * 0.2

    state = net.initial_state(batch_size=2)
    prev_l23 = state.r_l23.clone()
    seen_change = False
    for t in range(5):
        x_hat, state, _ = net(frames[:, t], state)
        for name, tensor in (
            ("r_l4", state.r_l4), ("r_l23", state.r_l23),
            ("r_h", state.r_h), ("x_hat", x_hat),
        ):
            assert torch.all(torch.isfinite(tensor)), (
                f"{name} NaN/inf at t={t}"
            )
        if not torch.equal(state.r_l23, prev_l23):
            seen_change = True
        prev_l23 = state.r_l23.clone()
    assert seen_change, "r_l23 never changed across 5 steps — state stuck"


def test_rollout_tracks_input(cfg, net):
    """Different frames → different end-of-rollout L4 rates."""
    a = cfg.arch
    torch.manual_seed(1)
    frames_a = torch.randn(1, 3, 1, a.grid_h, a.grid_w) * 0.2
    torch.manual_seed(2)
    frames_b = torch.randn(1, 3, 1, a.grid_h, a.grid_w) * 0.2

    state_a = net.initial_state(batch_size=1)
    state_b = net.initial_state(batch_size=1)
    for t in range(3):
        _, state_a, _ = net(frames_a[:, t], state_a)
        _, state_b, _ = net(frames_b[:, t], state_b)

    assert not torch.equal(state_a.r_l4, state_b.r_l4), (
        "different frame sequences produced identical r_l4 — L4 front end "
        "did not track input differences"
    )
