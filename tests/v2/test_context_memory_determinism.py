"""Determinism of `ContextMemory` — construction and forward.

Two requirements:
  * Construction with the same seed produces bit-identical weight tensors.
  * Forward on the same inputs produces bit-identical outputs across calls
    and across fresh instances.
"""

from __future__ import annotations

import torch

from src.v2_model.context_memory import ContextMemory


def _make(seed: int = 0) -> ContextMemory:
    return ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        tau_m_ms=500.0, dt_ms=5.0, seed=seed,
    )


WEIGHT_NAMES = (
    "W_hm_gen", "W_mm_gen", "W_mh_gen",
    "W_qm_task", "W_lm_task", "W_mh_task",
)


def test_same_seed_same_init() -> None:
    cm1 = _make(seed=42)
    cm2 = _make(seed=42)
    for name in WEIGHT_NAMES:
        p1 = getattr(cm1, name)
        p2 = getattr(cm2, name)
        torch.testing.assert_close(p1, p2, atol=0.0, rtol=0.0)


def test_different_seed_different_generic_init() -> None:
    """Seed must actually seed — two different seeds give different generic weights."""
    cm1 = _make(seed=0)
    cm2 = _make(seed=1)
    # Task-specific weights are zero regardless of seed — skip them.
    for name in ("W_hm_gen", "W_mm_gen", "W_mh_gen"):
        p1 = getattr(cm1, name)
        p2 = getattr(cm2, name)
        assert not torch.equal(p1, p2), f"{name} identical across seeds"


def test_task_input_weights_seed_dependent_and_small() -> None:
    """W_qm_task / W_lm_task are small-random init (N(0, 0.01), Task #58).

    They must vary with seed (so Phase-3 bootstrap gets a real signal) and
    their magnitudes must stay under the 0.05 cap used by the null-control
    tolerance. W_mh_task stays at exact zero regardless of seed.
    """
    cm_a = _make(seed=0)
    cm_b = _make(seed=1)
    # Input-path task weights are seed-dependent and bounded.
    for name in ("W_qm_task", "W_lm_task"):
        p_a = getattr(cm_a, name)
        p_b = getattr(cm_b, name)
        assert not torch.equal(p_a, p_b), f"{name} identical across seeds"
        assert float(p_a.abs().max().item()) < 0.05
        assert float(p_b.abs().max().item()) < 0.05
    # Output readout task weight stays at exact zero until Phase-3 plasticity.
    for seed in (0, 1, 42, 9999):
        cm = _make(seed=seed)
        assert torch.all(cm.W_mh_task == 0.0)


def test_forward_bit_exact_across_calls() -> None:
    cm = _make(seed=7)
    B = 3
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)
    q = torch.randn(B, cm.n_cue)
    lead = torch.randn(B, cm.n_leader)

    m1, b1 = cm(m, h, q, lead)
    m2, b2 = cm(m, h, q, lead)
    torch.testing.assert_close(m1, m2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b1, b2, atol=0.0, rtol=0.0)


def test_forward_bit_exact_across_instances() -> None:
    cm1 = _make(seed=11)
    cm2 = _make(seed=11)
    B = 3
    m = torch.randn(B, cm1.n_m)
    h = torch.randn(B, cm1.n_h)
    m1, b1 = cm1(m, h)
    m2, b2 = cm2(m, h)
    torch.testing.assert_close(m1, m2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b1, b2, atol=0.0, rtol=0.0)


def test_set_phase_does_not_perturb_forward() -> None:
    """set_phase is informational; it must not alter forward arithmetic."""
    cm = _make(seed=3)
    B = 2
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)

    ref_m_next, ref_b = cm(m, h)
    for phase in ("phase3_kok", "phase3_richter", "phase2"):
        cm.set_phase(phase)                                           # type: ignore[arg-type]
        mn, bn = cm(m, h)
        torch.testing.assert_close(mn, ref_m_next, atol=0.0, rtol=0.0)
        torch.testing.assert_close(bn, ref_b, atol=0.0, rtol=0.0)
