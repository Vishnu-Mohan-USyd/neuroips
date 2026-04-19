"""Shape + API contract for `src.v2_model.lgn_l4.LGNL4FrontEnd`.

Verifies the Task #9 module API:
  - forward returns (lgn_feature_map, l4_e_rate, updated_state).
  - lgn_feature_map: [B, 2 + N_ori, H, W]; channel layout is ON/OFF DoG first,
    then N_ori Gabor-energy channels.
  - l4_e_rate: [B, n_l4_e].
  - updated_state is a NetworkStateV2 with the new r_l4 spliced in and
    every other field left untouched (NamedTuple._replace semantics).

Also exercises input-shape validation (the forward contract raises on
grid mismatch and bad channel count).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.state import NetworkStateV2, initial_state


def test_forward_lgn_feature_shape(cfg, batch_size) -> None:
    """LGN feature map is [B, 2 + N_ori, H, W]."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_feat, _, _ = front(frames, state)

    expected = (batch_size, 2 + cfg.arch.n_orientations,
                cfg.arch.grid_h, cfg.arch.grid_w)
    assert lgn_feat.shape == expected


def test_forward_l4_rate_shape(cfg, batch_size) -> None:
    """L4 E rate tensor is [B, n_l4_e]."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    _, r_l4, _ = front(frames, state)
    assert r_l4.shape == (batch_size, cfg.arch.n_l4_e)


def test_forward_updated_state_is_named_tuple(cfg, batch_size) -> None:
    """Returned state is a NetworkStateV2 (not a plain tuple)."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    _, _, new_state = front(frames, state)
    assert isinstance(new_state, NetworkStateV2)


def test_forward_state_r_l4_updated(cfg, batch_size) -> None:
    """`new_state.r_l4` matches the returned l4_e_rate tensor."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    _, r_l4, new_state = front(frames, state)
    torch.testing.assert_close(new_state.r_l4, r_l4, atol=0.0, rtol=0.0)


def test_forward_other_state_fields_preserved(cfg, batch_size) -> None:
    """Only r_l4 changes — every other field of the input state is preserved
    by reference (NamedTuple._replace semantics)."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    _, _, new_state = front(frames, state)
    for name in ("r_l23", "r_pv", "r_som", "r_h", "h_pv", "m",
                 "regime_posterior"):
        assert getattr(new_state, name) is getattr(state, name), (
            f"field {name!r} was replaced; front-end must only update r_l4."
        )
    assert new_state.pre_traces is state.pre_traces
    assert new_state.post_traces is state.post_traces


def test_forward_dtype_preserved(cfg, batch_size) -> None:
    """Rate tensor keeps the default float32 dtype of `initial_state`."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    _, r_l4, _ = front(frames, state)
    assert r_l4.dtype == torch.float32


@pytest.mark.parametrize("bad_channels", [0, 2, 3])
def test_forward_rejects_wrong_channel_count(cfg, batch_size, bad_channels) -> None:
    """Input must have exactly 1 channel (grayscale)."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, bad_channels,
                         cfg.arch.grid_h, cfg.arch.grid_w)

    with pytest.raises(ValueError, match="frames"):
        front(frames, state)


def test_forward_rejects_wrong_spatial(cfg, batch_size) -> None:
    """Spatial dimensions must match cfg.arch.grid_h × grid_w."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h // 2, cfg.arch.grid_w)

    with pytest.raises(ValueError, match="spatial"):
        front(frames, state)


def test_forward_rejects_wrong_state_r_l4(cfg, batch_size) -> None:
    """state.r_l4 must have shape [B, n_l4_e]; mismatch must raise."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    bad_state = state._replace(
        r_l4=torch.zeros(batch_size, cfg.arch.n_l4_e + 1)
    )
    with pytest.raises(ValueError, match="r_l4"):
        front(frames, bad_state)


def test_retinotopic_layout_consistency(cfg) -> None:
    """Module computes retinotopic layout (4×4 × N_ori = 128; 4×4 = 16 PV)."""
    front = LGNL4FrontEnd(cfg)
    assert front.retino_side * front.retino_side == cfg.arch.n_l4_pv
    assert front.retino_side ** 2 * front.n_ori == cfg.arch.n_l4_e
    assert front.pool_h * front.retino_side == cfg.arch.grid_h
    assert front.pool_w * front.retino_side == cfg.arch.grid_w
