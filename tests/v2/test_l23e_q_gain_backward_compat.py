"""Task #74 β-mechanism Step 1 — bit-exactness lock for L23E W_q_gain.

Pytest gate requested by Lead on 2026-04-23 for the frozen-weight
introduction of ``W_q_gain``. The three invariants below must hold so
that adding the per-cue gain pathway does not silently alter Phase-2
substrate or any existing eval:

1. ``W_q_gain`` is a non-trainable, non-persistent buffer.
2. With the default ``W_q_gain`` (all ones), calling ``L23E.forward``
   with any one-hot ``q_t`` produces a bit-exact-same output as calling
   with ``q_t=None``. (So existing eval code that now routes ``q_t``
   through V2Network → L23E sees no behavioural drift.)
3. When ``q_t=None``, L23E ignores ``W_q_gain`` entirely — legacy
   callers are completely untouched.

Each assertion uses ``torch.equal`` (exact equality, not tolerance) to
lock bit-exactness.
"""
from __future__ import annotations

import torch

from src.v2_model.layers import L23E


def _make_l23e(seed: int = 0) -> L23E:
    return L23E(
        n_units=64, n_l4_e=32, n_pv=8, n_som=8, n_h_e=16,
        n_cue=6, seed=seed, device="cpu",
    )


def _make_inputs(l23: L23E, *, batch: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    l4 = torch.rand(batch, l23.n_l4_e, generator=g)
    rec = torch.rand(batch, l23.n_units, generator=g) * 0.01
    som = torch.rand(batch, l23.n_som, generator=g) * 0.01
    pv = torch.rand(batch, l23.n_pv, generator=g) * 0.01
    h = torch.rand(batch, l23.n_h_e, generator=g) * 0.01
    ctx = torch.zeros(batch, l23.n_units)
    state = torch.zeros(batch, l23.n_units)
    return l4, rec, som, pv, h, ctx, state


def test_w_q_gain_is_non_trainable_buffer():
    l23 = _make_l23e()
    # Buffer present, right shape
    assert hasattr(l23, "W_q_gain")
    assert l23.W_q_gain.shape == (l23.n_cue, l23.n_units)
    # Not a Parameter
    assert "W_q_gain" not in dict(l23.named_parameters())
    assert "W_q_gain" in dict(l23.named_buffers())
    # No gradient tracking
    assert l23.W_q_gain.requires_grad is False
    # Non-persistent — not in state_dict
    assert "W_q_gain" not in l23.state_dict()
    # Default init: all ones
    assert torch.equal(l23.W_q_gain, torch.ones_like(l23.W_q_gain))


def test_forward_with_none_qt_bitexact_to_no_gain_path():
    """When q_t is None, the forward path must be byte-identical to the
    pre-change implementation. We simulate 'pre-change' by running with
    q_t=None and asserting the W_q_gain buffer was never read (evidenced
    by manually blowing up W_q_gain and re-running — same output)."""
    l23 = _make_l23e()
    l4, rec, som, pv, h, ctx, state = _make_inputs(l23)

    out_none_1, _ = l23(
        l4_input=l4, l23_recurrent_input=rec, som_input=som, pv_input=pv,
        h_apical_input=h, context_bias=ctx, state=state,
    )
    # Arbitrarily corrupt W_q_gain — result must be unchanged.
    with torch.no_grad():
        l23.W_q_gain.fill_(9999.0)
    out_none_2, _ = l23(
        l4_input=l4, l23_recurrent_input=rec, som_input=som, pv_input=pv,
        h_apical_input=h, context_bias=ctx, state=state,
    )
    assert torch.equal(out_none_1, out_none_2)


def test_forward_default_wqgain_onehot_qt_bitexact_to_none():
    """With the default (all-ones) W_q_gain, forward(q_t=one-hot) ==
    forward(q_t=None) bit-exactly — any one-hot vector selects a
    row of ones, and ``ff_l4 * ones == ff_l4``.
    """
    l23 = _make_l23e()
    l4, rec, som, pv, h, ctx, state = _make_inputs(l23)

    out_none, _ = l23(
        l4_input=l4, l23_recurrent_input=rec, som_input=som, pv_input=pv,
        h_apical_input=h, context_bias=ctx, state=state,
    )
    # Build batched one-hots over different cue ids per batch row.
    q_t = torch.zeros(l4.shape[0], l23.n_cue)
    for b in range(l4.shape[0]):
        q_t[b, b % l23.n_cue] = 1.0
    out_qt, _ = l23(
        l4_input=l4, l23_recurrent_input=rec, som_input=som, pv_input=pv,
        h_apical_input=h, context_bias=ctx, state=state, q_t=q_t,
    )
    assert torch.equal(out_none, out_qt)


def test_forward_nonones_wqgain_changes_output():
    """Sanity: if W_q_gain is populated with a non-ones pattern, the
    forward output with q_t=one-hot must differ from q_t=None — proves
    the new path is actually exercised.
    """
    l23 = _make_l23e()
    l4, rec, som, pv, h, ctx, state = _make_inputs(l23)
    with torch.no_grad():
        l23.W_q_gain.fill_(1.0)
        l23.W_q_gain[0].fill_(0.5)                 # cue 0 halves L4 drive
    q_t = torch.zeros(l4.shape[0], l23.n_cue)
    q_t[:, 0] = 1.0                                # all batch rows pick cue 0
    out_none, _ = l23(
        l4_input=l4, l23_recurrent_input=rec, som_input=som, pv_input=pv,
        h_apical_input=h, context_bias=ctx, state=state,
    )
    out_qt, _ = l23(
        l4_input=l4, l23_recurrent_input=rec, som_input=som, pv_input=pv,
        h_apical_input=h, context_bias=ctx, state=state, q_t=q_t,
    )
    assert not torch.equal(out_none, out_qt)
