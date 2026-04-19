"""Three-factor plasticity rules for Phase-3 cue-memory-probe binding.

Variant A — `delta_qm` (cue → memory, 3-factor):
  * Coincident positive (cue, memory, memory_error) ⇒ ΔW > 0.
  * Zero cue / zero memory / zero memory_error each kills the Hebbian term.
  * Sign of ΔW tracks sign of memory_error (holding cue, memory ≥ 0).

Variant B — `delta_mh` (memory → probe, 2-factor error-driven):
  * Positive probe_error + positive memory ⇒ ΔW > 0.
  * Sign flips with sign of probe_error.

Both variants: mask preservation, weight-decay behaviour, validation.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.plasticity import ThreeFactorRule


# ---------------------------------------------------------------------------
# delta_qm — three-factor
# ---------------------------------------------------------------------------

def test_delta_qm_coincident_factors_positive() -> None:
    """All three factors positive ⇒ ΔW > 0 entrywise (decay=0)."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    cue = torch.ones(2, 3)
    memory = torch.full((2, 4), 2.0)
    memory_error = torch.full((2, 4), 0.5)
    weights = torch.zeros(4, 3)
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    assert (dw > 0).all()


def test_delta_qm_zero_cue_zero_update() -> None:
    """cue = 0 + decay=0 ⇒ ΔW = 0."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    cue = torch.zeros(2, 3)
    memory = torch.rand(2, 4) + 0.1
    memory_error = torch.rand(2, 4) + 0.1
    weights = torch.rand(4, 3)
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    assert torch.all(dw == 0.0)


def test_delta_qm_zero_memory_zero_update() -> None:
    """memory = 0 + decay=0 ⇒ ΔW = 0."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    cue = torch.rand(2, 3) + 0.1
    memory = torch.zeros(2, 4)
    memory_error = torch.rand(2, 4) + 0.1
    weights = torch.rand(4, 3)
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    assert torch.all(dw == 0.0)


def test_delta_qm_zero_memory_error_zero_update() -> None:
    """memory_error = 0 + decay=0 ⇒ ΔW = 0."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    cue = torch.rand(2, 3) + 0.1
    memory = torch.rand(2, 4) + 0.1
    memory_error = torch.zeros(2, 4)
    weights = torch.rand(4, 3)
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    assert torch.all(dw == 0.0)


def test_delta_qm_sign_tracks_memory_error() -> None:
    """Flipping sign of memory_error flips the sign of ΔW."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    cue = torch.ones(2, 3)
    memory = torch.full((2, 4), 2.0)
    memory_error_pos = torch.full((2, 4), 0.5)
    memory_error_neg = -memory_error_pos
    weights = torch.zeros(4, 3)
    dw_pos = rule.delta_qm(cue, memory, memory_error_pos, weights)
    dw_neg = rule.delta_qm(cue, memory, memory_error_neg, weights)
    torch.testing.assert_close(dw_pos, -dw_neg, atol=1e-6, rtol=0.0)


def test_delta_qm_analytic_magnitude() -> None:
    """Scalar check: Δw[j, i] == lr · mean_b(memory[b,j] · memory_error[b,j] · cue[b,i]).

    lr kept small so analytic values stay inside the ±0.01 per-step clamp
    (Task #62) and the formula is verified unclipped.
    """
    rule = ThreeFactorRule(lr=0.001, weight_decay=0.0)
    cue = torch.tensor([[1.0, 2.0]])                       # [B=1, n_cue=2]
    memory = torch.tensor([[3.0]])                         # [B=1, n_m=1]
    memory_error = torch.tensor([[0.5]])
    weights = torch.zeros(1, 2)
    # gated = memory · memory_error = 1.5; hebb[0,0] = 1.5 · 1 = 1.5; hebb[0,1] = 3.0
    expected = torch.tensor([[0.001 * 1.5, 0.001 * 3.0]])
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    torch.testing.assert_close(dw, expected, atol=1e-6, rtol=0.0)


def test_delta_qm_clamps_large_updates() -> None:
    """Per-step Δw magnitude is clamped to [-0.01, 0.01] (Task #62).

    Rationale: plastic raw weights enter a positive-feedback explosion when
    individual Δw can reach O(10) — Phase-2 1000-step rolling training
    diverged by step ~150 before this clamp was introduced.
    """
    rule = ThreeFactorRule(lr=0.5, weight_decay=0.0)
    cue = torch.tensor([[1.0, 2.0]])
    memory = torch.tensor([[3.0]])
    memory_error = torch.tensor([[0.5]])
    weights = torch.zeros(1, 2)
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    assert float(dw.abs().max().item()) <= 0.01 + 1e-8
    # Both entries would analytically be [0.75, 1.5] — both clipped to 0.01.
    torch.testing.assert_close(
        dw, torch.tensor([[0.01, 0.01]]), atol=1e-8, rtol=0.0
    )


def test_delta_qm_mask_preservation() -> None:
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.01)
    cue = torch.rand(2, 3)
    memory = torch.rand(2, 4)
    memory_error = torch.rand(2, 4)
    weights = torch.rand(4, 3)
    mask = torch.rand(4, 3) > 0.5
    dw = rule.delta_qm(cue, memory, memory_error, weights, mask=mask)
    assert torch.all(dw[~mask] == 0.0)


# ---------------------------------------------------------------------------
# delta_mh — error-driven
# ---------------------------------------------------------------------------

def test_delta_mh_positive_probe_error_positive_update() -> None:
    """probe_error > 0 + memory > 0 + decay=0 ⇒ ΔW > 0."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    memory = torch.ones(2, 3)                              # [B=2, n_m=3]
    probe_error = torch.full((2, 4), 0.5)                  # [B=2, n_h=4]
    weights = torch.zeros(4, 3)
    dw = rule.delta_mh(memory, probe_error, weights)
    assert (dw > 0).all()


def test_delta_mh_sign_flips_with_probe_error() -> None:
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.0)
    memory = torch.ones(2, 3)
    pe_pos = torch.full((2, 4), 0.5)
    weights = torch.zeros(4, 3)
    dw_pos = rule.delta_mh(memory, pe_pos, weights)
    dw_neg = rule.delta_mh(memory, -pe_pos, weights)
    torch.testing.assert_close(dw_pos, -dw_neg, atol=1e-6, rtol=0.0)


def test_delta_mh_analytic_magnitude() -> None:
    """lr kept small so analytic values stay inside the ±0.01 per-step
    clamp (Task #62) and the formula is verified unclipped."""
    rule = ThreeFactorRule(lr=0.001, weight_decay=0.0)
    memory = torch.tensor([[1.0, 2.0]])                    # [B=1, n_m=2]
    probe_error = torch.tensor([[3.0]])                    # [B=1, n_h=1]
    weights = torch.zeros(1, 2)
    # hebb[0, 0] = 3 · 1 = 3; hebb[0, 1] = 3 · 2 = 6
    expected = torch.tensor([[0.001 * 3.0, 0.001 * 6.0]])
    dw = rule.delta_mh(memory, probe_error, weights)
    torch.testing.assert_close(dw, expected, atol=1e-6, rtol=0.0)


def test_delta_mh_clamps_large_updates() -> None:
    """Per-step Δw magnitude is clamped to [-0.01, 0.01] (Task #62)."""
    rule = ThreeFactorRule(lr=0.5, weight_decay=0.0)
    memory = torch.tensor([[1.0, 2.0]])
    probe_error = torch.tensor([[3.0]])
    weights = torch.zeros(1, 2)
    dw = rule.delta_mh(memory, probe_error, weights)
    assert float(dw.abs().max().item()) <= 0.01 + 1e-8
    torch.testing.assert_close(
        dw, torch.tensor([[0.01, 0.01]]), atol=1e-8, rtol=0.0
    )


def test_delta_mh_mask_preservation() -> None:
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.01)
    memory = torch.rand(2, 3)
    probe_error = torch.rand(2, 4)
    weights = torch.rand(4, 3)
    mask = torch.rand(4, 3) > 0.5
    dw = rule.delta_mh(memory, probe_error, weights, mask=mask)
    assert torch.all(dw[~mask] == 0.0)


# ---------------------------------------------------------------------------
# Shared weight-decay behaviour
# ---------------------------------------------------------------------------

def test_weight_decay_qm() -> None:
    """weight_decay kept small so the expected analytic value stays inside
    the ±0.01 per-step clamp (Task #62) and the decay arithmetic is verified
    unclipped."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.005)
    cue = torch.zeros(2, 3)                                # no Hebbian drive
    memory = torch.zeros(2, 4)
    memory_error = torch.zeros(2, 4)
    weights = torch.ones(4, 3)
    dw = rule.delta_qm(cue, memory, memory_error, weights)
    torch.testing.assert_close(
        dw, torch.full_like(weights, -0.005), atol=1e-6, rtol=0.0
    )


def test_weight_decay_mh() -> None:
    """weight_decay kept small so the expected analytic value stays inside
    the ±0.01 per-step clamp (Task #62)."""
    rule = ThreeFactorRule(lr=0.1, weight_decay=0.005)
    memory = torch.zeros(2, 3)
    probe_error = torch.zeros(2, 4)
    weights = torch.ones(4, 3)
    dw = rule.delta_mh(memory, probe_error, weights)
    torch.testing.assert_close(
        dw, torch.full_like(weights, -0.005), atol=1e-6, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Input / init validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_lr", [0.0, -0.01])
def test_rejects_non_positive_lr(bad_lr: float) -> None:
    with pytest.raises(ValueError, match="lr"):
        ThreeFactorRule(lr=bad_lr)


def test_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        ThreeFactorRule(lr=0.1, weight_decay=-0.01)


def test_delta_qm_rejects_mismatched_memory_error() -> None:
    rule = ThreeFactorRule(lr=0.1)
    cue = torch.rand(2, 3)
    memory = torch.rand(2, 4)
    memory_error = torch.rand(2, 5)                        # wrong last dim
    weights = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="memory_error"):
        rule.delta_qm(cue, memory, memory_error, weights)
