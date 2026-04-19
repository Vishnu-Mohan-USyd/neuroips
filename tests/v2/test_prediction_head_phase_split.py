"""Phase 2 = all weights plastic; Phase 3 = empty plastic list; phase switch
is non-mutating (weights never change on ``set_phase``)."""

from __future__ import annotations

import pytest
import torch

from src.v2_model.prediction_head import PredictionHead


def _default() -> PredictionHead:
    return PredictionHead(seed=0)


def test_phase2_all_weights_plastic() -> None:
    head = _default()
    head.set_phase("phase2")
    expected = {
        "W_pred_H_raw", "W_pred_C_raw", "W_pred_apical_raw", "b_pred_raw",
    }
    assert set(head.plastic_weight_names()) == expected
    assert set(head.frozen_weight_names()) == set()


def test_phase3_kok_all_frozen() -> None:
    head = _default()
    head.set_phase("phase3_kok")
    expected = {
        "W_pred_H_raw", "W_pred_C_raw", "W_pred_apical_raw", "b_pred_raw",
    }
    assert set(head.plastic_weight_names()) == set()
    assert set(head.frozen_weight_names()) == expected


def test_phase3_richter_all_frozen() -> None:
    head = _default()
    head.set_phase("phase3_richter")
    expected = {
        "W_pred_H_raw", "W_pred_C_raw", "W_pred_apical_raw", "b_pred_raw",
    }
    assert set(head.plastic_weight_names()) == set()
    assert set(head.frozen_weight_names()) == expected


def test_phase_switch_does_not_mutate_weights() -> None:
    """Calling ``set_phase`` never touches the stored parameters."""
    head = _default()
    snapshot = {
        name: getattr(head, name).detach().clone()
        for name in head._all_plastic_names
    }
    for phase in ("phase2", "phase3_kok", "phase3_richter", "phase2"):
        head.set_phase(phase)
        for name, before in snapshot.items():
            after = getattr(head, name)
            torch.testing.assert_close(
                after, before, atol=0.0, rtol=0.0,
            )


def test_plastic_and_frozen_disjoint_all_phases() -> None:
    head = _default()
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        head.set_phase(phase)
        plastic = set(head.plastic_weight_names())
        frozen = set(head.frozen_weight_names())
        assert plastic.isdisjoint(frozen), f"{phase}: overlap {plastic & frozen}"


def test_plastic_and_frozen_cover_all_weights() -> None:
    head = _default()
    expected = set(head._all_plastic_names)
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        head.set_phase(phase)
        covered = (
            set(head.plastic_weight_names()) | set(head.frozen_weight_names())
        )
        assert covered == expected, (
            f"{phase}: missing {expected - covered} or extra {covered - expected}"
        )


def test_unknown_phase_raises() -> None:
    head = _default()
    with pytest.raises(ValueError, match="phase"):
        head.set_phase("phase4")                                  # type: ignore[arg-type]


def test_default_phase_is_phase2() -> None:
    head = _default()
    assert head.phase == "phase2"


# ---------------------------------------------------------------------------
# Optional-weight configurations still honour the phase split
# ---------------------------------------------------------------------------

def test_phase_split_without_c_weight() -> None:
    head = PredictionHead(n_c_bias=None, n_l23_apical=256, seed=0)
    head.set_phase("phase2")
    expected = {"W_pred_H_raw", "W_pred_apical_raw", "b_pred_raw"}
    assert set(head.plastic_weight_names()) == expected


def test_phase_split_without_apical_weight() -> None:
    head = PredictionHead(n_c_bias=48, n_l23_apical=None, seed=0)
    head.set_phase("phase3_kok")
    expected = {"W_pred_H_raw", "W_pred_C_raw", "b_pred_raw"}
    assert set(head.frozen_weight_names()) == expected


def test_phase_split_h_only_head() -> None:
    head = PredictionHead(n_c_bias=None, n_l23_apical=None, seed=0)
    head.set_phase("phase2")
    assert set(head.plastic_weight_names()) == {"W_pred_H_raw", "b_pred_raw"}
