"""Per-phase plastic/frozen weight specification (v4 plan §Architecture + D.2).

Phase 2:          plastic = {W_hm_gen, W_mm_gen, W_mh_gen}
                  frozen  = {W_qm_task, W_lm_task,
                             W_mh_task_exc, W_mh_task_inh}
Phase 3 Kok:      plastic = {W_qm_task, W_mh_task_exc, W_mh_task_inh}
                  frozen  = {W_hm_gen, W_mm_gen, W_mh_gen, W_lm_task}
Phase 3 Richter:  plastic = {W_lm_task, W_mh_task_exc, W_mh_task_inh}
                  frozen  = {W_hm_gen, W_mm_gen, W_mh_gen, W_qm_task}

Task #74 Fix C-v2: the original ``W_mh_task`` was split into
``W_mh_task_exc`` (→ additive L23 E apical, secondary) and
``W_mh_task_inh`` (→ per-SOM-unit multiplicative gain on SOM→L23E
synapses, main apical-gain route). Both are plastic in Phase-3.
"""

from __future__ import annotations

import pytest

from src.v2_model.context_memory import ContextMemory


ALL_WEIGHTS = {
    "W_hm_gen", "W_mm_gen", "W_mh_gen",
    "W_qm_task", "W_lm_task", "W_mh_task_exc", "W_mh_task_inh",
}


def _make_cm() -> ContextMemory:
    return ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12, n_out_som=9,
        tau_m_ms=500.0, dt_ms=5.0, seed=0,
    )


@pytest.mark.parametrize(
    "phase,plastic,frozen",
    [
        ("phase2",
         {"W_hm_gen", "W_mm_gen", "W_mh_gen"},
         {"W_qm_task", "W_lm_task", "W_mh_task_exc", "W_mh_task_inh"}),
        ("phase3_kok",
         {"W_qm_task", "W_mh_task_exc", "W_mh_task_inh"},
         {"W_hm_gen", "W_mm_gen", "W_mh_gen", "W_lm_task"}),
        ("phase3_richter",
         {"W_lm_task", "W_mh_task_exc", "W_mh_task_inh"},
         {"W_hm_gen", "W_mm_gen", "W_mh_gen", "W_qm_task"}),
    ],
)
def test_phase_specs_match(phase: str, plastic: set[str], frozen: set[str]) -> None:
    cm = _make_cm()
    cm.set_phase(phase)                                                 # type: ignore[arg-type]
    assert set(cm.plastic_weight_names()) == plastic
    assert set(cm.frozen_weight_names()) == frozen


def test_plastic_and_frozen_disjoint() -> None:
    """No weight may be simultaneously plastic and frozen in any phase."""
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        cm = _make_cm()
        cm.set_phase(phase)                                             # type: ignore[arg-type]
        plastic = set(cm.plastic_weight_names())
        frozen = set(cm.frozen_weight_names())
        assert plastic.isdisjoint(frozen), f"phase {phase}: overlap {plastic & frozen}"


def test_plastic_and_frozen_cover_all_weights() -> None:
    """Every weight must appear in exactly one of {plastic, frozen} per phase."""
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        cm = _make_cm()
        cm.set_phase(phase)                                             # type: ignore[arg-type]
        covered = set(cm.plastic_weight_names()) | set(cm.frozen_weight_names())
        assert covered == ALL_WEIGHTS, (
            f"phase {phase}: missing {ALL_WEIGHTS - covered} or extra "
            f"{covered - ALL_WEIGHTS}"
        )


def test_named_parameters_match_weight_names() -> None:
    """All seven weight names must correspond to `nn.Parameter` attributes."""
    cm = _make_cm()
    param_names = {n for n, _ in cm.named_parameters()}
    assert ALL_WEIGHTS <= param_names, f"missing: {ALL_WEIGHTS - param_names}"


def test_parameters_have_requires_grad_false() -> None:
    """All weights are `nn.Parameter(..., requires_grad=False)` by v2 convention."""
    cm = _make_cm()
    for name, p in cm.named_parameters():
        assert p.requires_grad is False, f"{name} has requires_grad=True"


def test_switching_phase_does_not_mutate_weights() -> None:
    """Phase gating is informational — switching must not touch tensor data."""
    cm = _make_cm()
    snapshots = {n: p.detach().clone() for n, p in cm.named_parameters()}

    cm.set_phase("phase3_kok")
    cm.set_phase("phase3_richter")
    cm.set_phase("phase2")

    for n, p in cm.named_parameters():
        assert (p == snapshots[n]).all(), f"{n} changed after phase switches"
