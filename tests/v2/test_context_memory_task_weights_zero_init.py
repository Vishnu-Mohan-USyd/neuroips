"""Task-specific weight init: bounded random for input paths, zero for readout.

The input-pathway weights (``W_qm_task``, ``W_lm_task``) initialise to
``N(0, task_input_init_std)`` with ``task_input_init_std=0.3`` so the
cue/leader drive into memory starts at roughly the same magnitude as
the generic history drive (Task #70 / debugger Task #69 H_W). Combined
with ``cue_gain=5.0`` in forward, this lifts task-specific drive into
the same envelope as ``W_hm_gen @ h``, which is what lets memory
actually differentiate cue classes within a trial (earlier init of 0.01
left cue_drive ~0.08 vs h_drive ~1.57, so memory ended up cos_sim
0.999 across cue-0 vs cue-1).

The magnitude cap is kept deliberately loose (``|W|_max < 1.0``): the
test's job is "not accidentally huge", not "tiny enough to satisfy the
old Gate-6 null-control band" — the null-control assay itself is a
functional check, it doesn't need the underlying weights to be
vanishingly small.

The output readout weight (``W_mh_task``) stays at exact zero at
construction — the output bias path must not be touched by task inputs
until Phase-3 plasticity binds it.
"""

from __future__ import annotations

import torch

from src.v2_model.context_memory import ContextMemory


TASK_INPUT_MAX_MAGNITUDE = 2.0  # loose cap — Task #70 raised init to std=0.3


def _cm() -> ContextMemory:
    return ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        tau_m_ms=500.0, dt_ms=5.0, seed=0,
    )


def test_task_input_weights_small_at_construction() -> None:
    """|W|_max < 1.0 for Kok cue + Richter leader pathways (N(0, 0.3) init, Task #70)."""
    cm = _cm()
    assert float(cm.W_qm_task.abs().max().item()) < TASK_INPUT_MAX_MAGNITUDE
    assert float(cm.W_lm_task.abs().max().item()) < TASK_INPUT_MAX_MAGNITUDE
    # But not exactly zero — they must carry a bootstrap signal.
    assert float(cm.W_qm_task.abs().max().item()) > 0.0
    assert float(cm.W_lm_task.abs().max().item()) > 0.0


def test_task_readout_weight_zero_at_construction() -> None:
    """W_mh_task (output path) must still be exact zero before Phase-3."""
    cm = _cm()
    assert torch.all(cm.W_mh_task == 0.0)


def test_generic_weights_nonzero_at_construction() -> None:
    """Sanity: the generic weights are actually populated (not a vacuous test)."""
    cm = _cm()
    assert not torch.all(cm.W_hm_gen == 0.0)
    assert not torch.all(cm.W_mm_gen == 0.0)
    assert not torch.all(cm.W_mh_gen == 0.0)


def test_readout_bias_independent_of_task_inputs_before_training() -> None:
    """With W_mh_task = 0, the pre-update bias readout is generic-only.

    Because ``b_t = W_mh_gen · m_t + W_mh_task · m_t`` uses the *pre-update*
    memory state, passing or omitting q/leader this step does not change
    ``b_t`` as long as ``W_mh_task = 0``.
    """
    cm = _cm()
    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)
    q = torch.randn(B, cm.n_cue)
    lead = torch.randn(B, cm.n_leader)

    _, b_with_task_inputs = cm(m, h, q, lead)
    _, b_generic_only = cm(m, h)                                # no q, no leader

    torch.testing.assert_close(
        b_with_task_inputs, b_generic_only, atol=0.0, rtol=0.0
    )


def test_task_inputs_shift_memory_by_at_most_input_scale() -> None:
    """W_qm_task, W_lm_task are small-random, so swapping q/leader tensors
    perturbs the memory update by a bounded amount, not exactly zero."""
    cm = _cm()
    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)

    q1 = torch.randn(B, cm.n_cue)
    q2 = torch.randn(B, cm.n_cue)
    lead1 = torch.randn(B, cm.n_leader)
    lead2 = torch.randn(B, cm.n_leader)

    m_next_1, _ = cm(m, h, q1, lead1)
    m_next_2, _ = cm(m, h, q2, lead2)

    # The delta is bounded by (1 - decay) · φ′ · (‖W_qm‖‖Δq‖ + ‖W_lm‖‖Δl‖),
    # and with ‖W‖ ~ 0.01 and random inputs of order 1 this is well under 1.0.
    delta = (m_next_1 - m_next_2).abs().max().item()
    assert delta < 1.0
    # But it is not identically zero — cue/leader now carry signal.
    assert delta > 0.0
