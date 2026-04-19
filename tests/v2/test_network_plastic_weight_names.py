"""Aggregated plastic-weight manifest is correct per phase.

``V2Network.plastic_weight_names()`` returns ``(module_name, weight_name)``
pairs for every raw weight the current phase's plasticity rules may
mutate. The manifest must:

* Include every generic weight in Phase 2 (and exclude task-specific).
* Include only task-specific weights in Phase 3 variants.
* Never include any LGN/L4 weight (front end is frozen by construction).
"""

from __future__ import annotations

import pytest

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def test_phase2_includes_generic_weights(net):
    """Phase 2 manifest contains all L2/3, H, generic C, and prediction weights."""
    net.set_phase("phase2")
    names = set(net.plastic_weight_names())

    required = {
        ("l23_e", "W_l4_l23_raw"),
        ("l23_e", "W_rec_raw"),
        ("l23_e", "W_pv_l23_raw"),
        ("l23_e", "W_som_l23_raw"),
        ("l23_e", "W_fb_apical_raw"),
        ("l23_pv", "W_pre_raw"),
        ("l23_som", "W_l23_som_raw"),
        ("l23_som", "W_fb_som_raw"),
        ("h_e", "W_l23_h_raw"),
        ("h_e", "W_rec_raw"),
        ("h_e", "W_pv_h_raw"),
        ("h_pv", "W_pre_raw"),
        ("context_memory", "W_hm_gen"),
        ("context_memory", "W_mm_gen"),
        ("context_memory", "W_mh_gen"),
        ("prediction_head", "W_pred_H_raw"),
        ("prediction_head", "W_pred_C_raw"),
        ("prediction_head", "W_pred_apical_raw"),
        ("prediction_head", "b_pred_raw"),
    }
    missing = required - names
    assert not missing, f"phase2 manifest missing: {missing}"

    # No task-specific weights in Phase 2
    task_weights = {
        ("context_memory", "W_qm_task"),
        ("context_memory", "W_lm_task"),
        ("context_memory", "W_mh_task"),
    }
    leaks = task_weights & names
    assert not leaks, f"task-specific weights leaked into phase2 manifest: {leaks}"


def test_phase3_kok_only_task_weights(net):
    """Phase 3 Kok exposes only Kok task weights (+ shared W_mh_task)."""
    net.set_phase("phase3_kok")
    names = set(net.plastic_weight_names())
    assert names == {
        ("context_memory", "W_qm_task"),
        ("context_memory", "W_mh_task"),
    }, f"phase3_kok manifest wrong: {names}"


def test_phase3_richter_only_task_weights(net):
    """Phase 3 Richter exposes only Richter task weights (+ shared W_mh_task)."""
    net.set_phase("phase3_richter")
    names = set(net.plastic_weight_names())
    assert names == {
        ("context_memory", "W_lm_task"),
        ("context_memory", "W_mh_task"),
    }, f"phase3_richter manifest wrong: {names}"


def test_lgn_l4_never_plastic(net):
    """LGN/L4 weights never appear in any phase's plastic manifest."""
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        net.set_phase(phase)
        for module, _ in net.plastic_weight_names():
            assert module != "lgn_l4", (
                f"lgn_l4 weight listed as plastic in {phase!r} — violates "
                f"frozen-core contract"
            )
