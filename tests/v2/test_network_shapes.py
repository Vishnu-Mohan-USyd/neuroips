"""V2Network forward-step tensor shapes match the architecture table.

Every field returned by :meth:`V2Network.forward` must have the exact
``[B, ...]`` shape spec'd in ``cfg.arch``. If any population's output
width drifts from the config, downstream plasticity rules (which
pre-allocate trace tensors from ``cfg.arch``) will silently mis-align.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_forward_shapes(cfg, net, B):
    """Every forward-returned tensor has the correct ``[B, ...]`` shape."""
    a = cfg.arch
    state = net.initial_state(batch_size=B)
    frames = torch.zeros(B, 1, a.grid_h, a.grid_w)
    x_hat, next_state, info = net(frames, state)

    assert x_hat.shape == (B, a.n_l4_e)
    assert next_state.r_l4.shape == (B, a.n_l4_e)
    assert next_state.r_l23.shape == (B, a.n_l23_e)
    assert next_state.r_pv.shape == (B, a.n_l23_pv)
    assert next_state.r_som.shape == (B, a.n_l23_som)
    assert next_state.r_h.shape == (B, a.n_h_e)
    assert next_state.h_pv.shape == (B, a.n_h_pv)
    assert next_state.m.shape == (B, a.n_c)
    assert next_state.regime_posterior.shape == (B, cfg.regime.n_regimes)

    # Info dict matches contract
    assert info["lgn_feature_map"].shape == (
        B, 2 + a.n_orientations, a.grid_h, a.grid_w,
    )
    assert info["r_l4"].shape == (B, a.n_l4_e)
    assert info["r_l23"].shape == (B, a.n_l23_e)
    assert info["r_pv"].shape == (B, a.n_l23_pv)
    assert info["r_som"].shape == (B, a.n_l23_som)
    assert info["r_h"].shape == (B, a.n_h_e)
    assert info["h_pv"].shape == (B, a.n_h_pv)
    assert info["m"].shape == (B, a.n_c)
    assert info["b_l23"].shape == (B, a.n_l23_e)
    assert info["x_hat_next"].shape == (B, a.n_l4_e)


def test_forward_rejects_wrong_frame_shape(cfg, net):
    """Forward raises on frame rank or channel-count mismatch."""
    state = net.initial_state(batch_size=2)
    with pytest.raises(ValueError, match=r"x_t must be \[B, 1, H, W\]"):
        net(torch.zeros(2, 3, 32, 32), state)
    with pytest.raises(ValueError, match=r"x_t must be \[B, 1, H, W\]"):
        net(torch.zeros(2, 32, 32), state)


def test_forward_rejects_batch_mismatch(cfg, net):
    """Forward raises when frame batch != state batch."""
    state = net.initial_state(batch_size=2)
    frames = torch.zeros(4, 1, cfg.arch.grid_h, cfg.arch.grid_w)
    with pytest.raises(ValueError, match="batch size"):
        net(frames, state)
