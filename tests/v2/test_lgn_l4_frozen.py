"""Frozen-by-construction invariants for `LGNL4FrontEnd`.

Per v4 spec §Architecture: LGN + L4 must never be trained. The module
expresses this by registering every filter bank as a `buffer`, never as
an `nn.Parameter`. These tests verify that:

  1. The module has zero `nn.Parameter`s.
  2. The filter-bank buffers are present and tagged as buffers (so they
     move with `.to(device)` but do not appear in `.parameters()` or any
     downstream optimizer).
  3. Enabling autograd on the *input* does not produce gradients for any
     module tensor (because there are no parameters to accumulate into).
  4. Default output has `requires_grad == False` (the rate fields in
     `initial_state` are leaves without `requires_grad=True`, and the
     module introduces no new trainable parameters).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.state import initial_state


def test_module_has_no_parameters(cfg) -> None:
    """No `nn.Parameter` anywhere in `LGNL4FrontEnd`."""
    front = LGNL4FrontEnd(cfg)
    params = list(front.parameters())
    assert params == [], (
        f"LGNL4FrontEnd must hold zero nn.Parameters; found {len(params)}: "
        f"{[n for n, _ in front.named_parameters()]}"
    )


def test_total_trainable_count_is_zero(cfg) -> None:
    """Sum of trainable-element counts is zero."""
    front = LGNL4FrontEnd(cfg)
    total = sum(p.numel() for p in front.parameters() if p.requires_grad)
    assert total == 0


@pytest.mark.parametrize(
    "buf_name", ["dog_kernel", "gabor_even", "gabor_odd", "sigma_norm_sq"]
)
def test_filter_bank_is_buffer(cfg, buf_name: str) -> None:
    """Each filter bank must be registered as a buffer."""
    front = LGNL4FrontEnd(cfg)
    buf_dict = dict(front.named_buffers())
    assert buf_name in buf_dict, f"missing buffer {buf_name!r}"
    assert not buf_dict[buf_name].requires_grad


def test_filter_bank_kernel_shapes(cfg) -> None:
    """DoG is [2,1,K,K]; Gabor even/odd are [N_ori,1,K,K]; all same K."""
    front = LGNL4FrontEnd(cfg)
    K = front.KERNEL_SIZE
    n_ori = cfg.arch.n_orientations
    assert front.dog_kernel.shape == (2, 1, K, K)
    assert front.gabor_even.shape == (n_ori, 1, K, K)
    assert front.gabor_odd.shape == (n_ori, 1, K, K)


def test_forward_output_does_not_require_grad(cfg, batch_size) -> None:
    """With a non-grad input, nothing the module returns should require grad."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.zeros(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w,
                         requires_grad=False)

    lgn_feat, r_l4, new_state = front(frames, state)
    assert not lgn_feat.requires_grad
    assert not r_l4.requires_grad
    assert not new_state.r_l4.requires_grad


def test_gradients_through_input_do_not_touch_module(cfg, batch_size) -> None:
    """Even with `frames.requires_grad=True`, the module itself exposes no
    parameter gradients, because it has no parameters.

    This is the strongest form of `frozen`: the optimizer cannot touch
    anything in this module, regardless of what the caller does upstream.
    """
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w,
                         requires_grad=True)

    _, r_l4, _ = front(frames, state)
    r_l4.sum().backward()

    # No parameter → nothing to check on the module side. Input grad exists
    # but that's the caller's tensor, not ours.
    assert frames.grad is not None
    assert list(front.parameters()) == []


def test_buffers_move_with_device(cfg) -> None:
    """Filter banks move when `.to(device)` is called (buffer semantics)."""
    front = LGNL4FrontEnd(cfg)
    # No CUDA assumed — round-trip via .to('cpu') is still a live code path.
    moved = front.to("cpu")
    for _, buf in moved.named_buffers():
        assert buf.device.type == "cpu"
