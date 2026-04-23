"""Forward shape / dtype / phase-API sanity for `ContextMemory`."""

from __future__ import annotations

import pytest
import torch

from src.v2_model.context_memory import ContextMemory


def _make_cm(**overrides) -> ContextMemory:
    kw = dict(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12, n_out_som=9,
        tau_m_ms=500.0, dt_ms=5.0, seed=0,
    )
    kw.update(overrides)
    return ContextMemory(**kw)


def test_forward_output_shapes() -> None:
    cm = _make_cm()
    B = 4
    m = torch.randn(B, 16)
    h = torch.randn(B, 24)
    q = torch.randn(B, 6)
    lead = torch.randn(B, 7)
    m_next, b_exc, som_gain = cm(m, h, q, lead)
    assert m_next.shape == (B, 16)
    assert b_exc.shape == (B, 12)
    assert som_gain.shape == (B, 9)


def test_forward_output_dtype_matches_input() -> None:
    cm = _make_cm(dtype=torch.float64)
    B = 2
    m = torch.randn(B, 16, dtype=torch.float64)
    h = torch.randn(B, 24, dtype=torch.float64)
    m_next, b_exc, som_gain = cm(m, h)
    assert m_next.dtype == torch.float64
    assert b_exc.dtype == torch.float64
    assert som_gain.dtype == torch.float64


def test_single_batch_shape() -> None:
    cm = _make_cm()
    m = torch.randn(1, 16)
    h = torch.randn(1, 24)
    m_next, b_exc, som_gain = cm(m, h)
    assert m_next.shape == (1, 16)
    assert b_exc.shape == (1, 12)
    assert som_gain.shape == (1, 9)


def test_phase_api_default_phase2() -> None:
    cm = _make_cm()
    assert cm.phase == "phase2"
    assert set(cm.plastic_weight_names()) == {"W_hm_gen", "W_mm_gen", "W_mh_gen"}
    assert set(cm.frozen_weight_names()) == {
        "W_qm_task", "W_lm_task", "W_mh_task_exc", "W_mh_task_inh",
    }


def test_set_phase_switches_sets() -> None:
    cm = _make_cm()
    cm.set_phase("phase3_kok")
    assert cm.phase == "phase3_kok"
    assert set(cm.plastic_weight_names()) == {
        "W_qm_task", "W_mh_task_exc", "W_mh_task_inh",
    }

    cm.set_phase("phase3_richter")
    assert set(cm.plastic_weight_names()) == {
        "W_lm_task", "W_mh_task_exc", "W_mh_task_inh",
    }

    cm.set_phase("phase2")
    assert set(cm.plastic_weight_names()) == {"W_hm_gen", "W_mm_gen", "W_mh_gen"}


def test_set_phase_rejects_unknown() -> None:
    cm = _make_cm()
    with pytest.raises(ValueError, match="phase"):
        cm.set_phase("phase4")                                          # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_sizes(bad: int) -> None:
    with pytest.raises(ValueError):
        _make_cm(n_m=bad)
    with pytest.raises(ValueError):
        _make_cm(n_h=bad)
    with pytest.raises(ValueError):
        _make_cm(n_cue=bad)
    with pytest.raises(ValueError):
        _make_cm(n_leader=bad)
    with pytest.raises(ValueError):
        _make_cm(n_out=bad)


def test_rejects_non_positive_tau() -> None:
    with pytest.raises(ValueError, match="tau_m_ms"):
        _make_cm(tau_m_ms=0.0)
    with pytest.raises(ValueError, match="tau_m_ms"):
        _make_cm(tau_m_ms=-1.0)


def test_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt_ms"):
        _make_cm(dt_ms=0.0)
    with pytest.raises(ValueError, match="dt_ms"):
        _make_cm(dt_ms=-1.0)


def test_dt_ge_tau_still_constructs() -> None:
    """Exact-ODE leak ``exp(-dt/τ)`` is in (0, 1) for any positive dt/τ — no
    stability constraint. `dt >= tau` is unusual but legal."""
    cm = _make_cm(dt_ms=500.0, tau_m_ms=500.0)
    # exp(-1) ≈ 0.3679 — aggressive decay but well-defined.
    assert 0.36 < cm._decay < 0.38
    cm2 = _make_cm(dt_ms=600.0, tau_m_ms=500.0)
    # exp(-1.2) ≈ 0.3012 — still legal.
    assert 0.29 < cm2._decay < 0.31


def test_rejects_negative_init_std() -> None:
    with pytest.raises(ValueError, match="init_std"):
        _make_cm(init_std=-0.1)


# ---------------------------------------------------------------------------
# Forward-input shape validation
# ---------------------------------------------------------------------------

def test_rejects_wrong_m_shape() -> None:
    cm = _make_cm()
    with pytest.raises(ValueError, match="m_t"):
        cm(torch.randn(3, 7), torch.randn(3, 24))                       # wrong n_m
    with pytest.raises(ValueError, match="m_t"):
        cm(torch.randn(16), torch.randn(3, 24))                          # 1-D


def test_rejects_wrong_h_shape() -> None:
    cm = _make_cm()
    with pytest.raises(ValueError, match="h_t"):
        cm(torch.randn(3, 16), torch.randn(3, 99))                       # wrong n_h
    with pytest.raises(ValueError, match="h_t"):
        cm(torch.randn(3, 16), torch.randn(24))                          # 1-D


def test_rejects_batch_mismatch() -> None:
    cm = _make_cm()
    with pytest.raises(ValueError, match="batch"):
        cm(torch.randn(3, 16), torch.randn(5, 24))


def test_rejects_wrong_q_shape() -> None:
    cm = _make_cm()
    m = torch.randn(3, 16)
    h = torch.randn(3, 24)
    with pytest.raises(ValueError, match="q_t"):
        cm(m, h, q_t=torch.randn(3, 99))                                 # wrong n_cue
    with pytest.raises(ValueError, match="q_t"):
        cm(m, h, q_t=torch.randn(5, 6))                                  # wrong batch


def test_rejects_wrong_leader_shape() -> None:
    cm = _make_cm()
    m = torch.randn(3, 16)
    h = torch.randn(3, 24)
    with pytest.raises(ValueError, match="leader_t"):
        cm(m, h, leader_t=torch.randn(3, 99))                            # wrong n_leader
    with pytest.raises(ValueError, match="leader_t"):
        cm(m, h, leader_t=torch.randn(5, 7))                             # wrong batch
