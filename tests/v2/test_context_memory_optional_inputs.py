"""Optional `q_t` and `leader_t` inputs.

Brief contract: when either is `None`, the corresponding term is *skipped
entirely* — equivalent to passing explicit zero tensors but avoiding the
matmul. These tests confirm both the equivalence and that the skip is
actually exploited (`None` path doesn't touch the task weight).
"""

from __future__ import annotations

import torch

from src.v2_model.context_memory import ContextMemory


def _cm() -> ContextMemory:
    return ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        tau_m_ms=500.0, dt_ms=5.0, seed=0,
    )


def test_none_equivalent_to_explicit_zero_q() -> None:
    cm = _cm()
    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)
    lead = torch.randn(B, cm.n_leader)

    m_a, b_a = cm(m, h, q_t=None, leader_t=lead)
    m_b, b_b = cm(m, h, q_t=torch.zeros(B, cm.n_cue), leader_t=lead)

    torch.testing.assert_close(m_a, m_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_a, b_b, atol=0.0, rtol=0.0)


def test_none_equivalent_to_explicit_zero_leader() -> None:
    cm = _cm()
    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)
    q = torch.randn(B, cm.n_cue)

    m_a, b_a = cm(m, h, q_t=q, leader_t=None)
    m_b, b_b = cm(m, h, q_t=q, leader_t=torch.zeros(B, cm.n_leader))

    torch.testing.assert_close(m_a, m_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_a, b_b, atol=0.0, rtol=0.0)


def test_both_none_equivalent_to_both_explicit_zero() -> None:
    cm = _cm()
    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)

    m_a, b_a = cm(m, h)
    m_b, b_b = cm(
        m, h,
        q_t=torch.zeros(B, cm.n_cue),
        leader_t=torch.zeros(B, cm.n_leader),
    )
    torch.testing.assert_close(m_a, m_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_a, b_b, atol=0.0, rtol=0.0)


def test_nontrivial_task_weights_reveal_input_effect() -> None:
    """Sanity: populate a task weight, confirm q actually does something.

    Without this, the None-vs-zero equivalence would be vacuous — it would be
    trivially satisfied by weights that are zero everywhere.
    """
    cm = _cm()
    with torch.no_grad():
        cm.W_qm_task.normal_(0.0, 0.1)                                 # make it non-trivial

    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)

    m_no_q, _ = cm(m, h)
    q = torch.rand(B, cm.n_cue) + 0.5                                  # strictly positive drive
    m_with_q, _ = cm(m, h, q_t=q)

    # Different drive ⇒ different next-memory (outside numerical noise).
    assert not torch.allclose(m_no_q, m_with_q, atol=1e-6)

    # But None ≡ zero tensor still holds under the same populated weights.
    m_zero_q, _ = cm(m, h, q_t=torch.zeros(B, cm.n_cue))
    torch.testing.assert_close(m_no_q, m_zero_q, atol=0.0, rtol=0.0)
