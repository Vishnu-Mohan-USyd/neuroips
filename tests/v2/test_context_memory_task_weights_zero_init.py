"""Task-specific weights initialise to exact zero.

Required for Gate-6 null-expectation control: before Phase-3 plasticity
begins, the task-specific streams (Kok cue, Richter leader, task readout)
contribute exactly nothing to memory dynamics or to the output bias. Only
the generic pathway is active.
"""

from __future__ import annotations

import torch

from src.v2_model.context_memory import ContextMemory


def _cm() -> ContextMemory:
    return ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        tau_m_ms=500.0, dt_ms=5.0, seed=0,
    )


def test_task_weights_zero_at_construction() -> None:
    cm = _cm()
    assert torch.all(cm.W_qm_task == 0.0)
    assert torch.all(cm.W_lm_task == 0.0)
    assert torch.all(cm.W_mh_task == 0.0)


def test_generic_weights_nonzero_at_construction() -> None:
    """Sanity: the generic weights are actually populated (not a vacuous test)."""
    cm = _cm()
    assert not torch.all(cm.W_hm_gen == 0.0)
    assert not torch.all(cm.W_mm_gen == 0.0)
    assert not torch.all(cm.W_mh_gen == 0.0)


def test_task_weights_contribute_zero_to_bias_before_training() -> None:
    """With task weights at zero, `b_t` comes entirely from the generic readout."""
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


def test_task_inputs_do_not_affect_memory_before_training() -> None:
    """With W_qm_task / W_lm_task zero, the memory update is input-independent
    in q and leader."""
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
    m_next_none, _ = cm(m, h)

    torch.testing.assert_close(m_next_1, m_next_2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(m_next_1, m_next_none, atol=0.0, rtol=0.0)
