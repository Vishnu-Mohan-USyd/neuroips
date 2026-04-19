"""Load-bearing: context memory C must actually reach L2/3 apical.

If the ``ContextMemory`` → L2/3 E context_bias route is misrouted or
silently zeroed, the predictive-coding signal from H/C cannot modulate
L2/3 apical, and the full predictive-circuit story collapses. This
test asserts:

1. With the generic readout ``W_mh_gen`` ≠ 0, varying the memory state
   produces a measurably different L2/3 E rate (the bias is reaching
   the apical compartment and affecting the soma).
2. Setting the readout to exactly zero makes the memory state
   irrelevant — L2/3 E rate depends only on feedforward + recurrent
   inputs.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def _forward_with_custom_m(net, m_init, frames):
    """Run one forward step with a custom initial memory state."""
    state = net.initial_state(batch_size=frames.shape[0])
    state = state._replace(m=m_init)
    return net(frames, state)


def test_context_memory_affects_l23(cfg, net):
    """Different m_t values produce different r_l23 after one forward step."""
    a = cfg.arch
    frames = torch.randn(1, 1, a.grid_h, a.grid_w) * 0.2

    # Two distinct memory states (bounded positive — ContextMemory is
    # rectified so negative values would collapse to zero anyway).
    m_a = torch.zeros(1, a.n_c)
    m_b = torch.full((1, a.n_c), 2.0)

    _, state_a, _ = _forward_with_custom_m(net, m_a, frames)
    _, state_b, _ = _forward_with_custom_m(net, m_b, frames)

    # L2/3 E rates must differ — the only input that changes between the
    # two runs is m_t, which feeds L23 apical via W_mh_gen.
    diff = (state_a.r_l23 - state_b.r_l23).abs().max()
    assert float(diff) > 1e-3, (
        f"r_l23 did not respond to memory change (max-abs diff {float(diff):.6f}) "
        "— context_bias is not reaching L2/3 apical"
    )


def test_zeroing_readout_silences_context(cfg, net):
    """With W_mh_gen = 0 (and W_mh_task = 0 by init), m_t has no effect on r_l23."""
    a = cfg.arch
    with torch.no_grad():
        net.context_memory.W_mh_gen.zero_()
        net.context_memory.W_mh_task.zero_()  # zero by construction; belt + braces

    frames = torch.randn(1, 1, a.grid_h, a.grid_w) * 0.2

    m_a = torch.zeros(1, a.n_c)
    m_b = torch.full((1, a.n_c), 5.0)

    _, state_a, _ = _forward_with_custom_m(net, m_a, frames)
    _, state_b, _ = _forward_with_custom_m(net, m_b, frames)

    # With the readout zeroed, L23 sees an identical (zero) context_bias
    # regardless of m_t — r_l23 must be identical. (r_h also identical
    # since HE context_bias is unconditionally zero in this wiring.)
    torch.testing.assert_close(state_a.r_l23, state_b.r_l23, atol=0.0, rtol=0.0)
