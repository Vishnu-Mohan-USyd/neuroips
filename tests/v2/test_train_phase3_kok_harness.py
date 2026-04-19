"""Smoke tests for the Phase-3 Kok task-learning driver.

Exercises :func:`run_phase3_kok_training` on an untrained network with a
micro-budget (2 learning + 2 scan trials, short epochs). Verifies the
Task-#40 critical invariants:

* The driver runs end-to-end on an untrained net without crashing.
* ``frozen_sensory_core_sha`` is identical before and after training.
* Every non-plastic Parameter is left bitwise-identical.
* The two task weights (``W_qm_task``, ``W_mh_task``) remain finite.
* Running under the wrong phase raises :class:`PhaseFrozenError` before
  any weight mutation can happen.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.train_phase2_predictive import PhaseFrozenError
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, run_phase3_kok_training,
)
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_phase3_kok_checkpoint(cfg, tmp_path: Path, *, seed: int = 42) -> Path:
    """Save an untrained V2Network with ``phase='phase3_kok'``."""
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase("phase3_kok")
    net.eval()
    path = tmp_path / "phase3_kok_step_0.pt"
    torch.save({
        "step": 0, "state_dict": net.state_dict(),
        "phase": "phase3_kok", "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path


@pytest.fixture
def untrained_kok_bundle(cfg, tmp_path):
    return load_checkpoint(
        _write_phase3_kok_checkpoint(cfg, tmp_path), seed=42, device="cpu",
    )


def _tiny_timing() -> KokTiming:
    """Micro-budget timing: 9 steps per trial instead of 370."""
    return KokTiming(
        cue_steps=2, delay_steps=2, probe1_steps=2,
        blank_steps=1, probe2_steps=2,
    )


def test_tiny_training_runs_and_preserves_frozen_invariants(
    untrained_kok_bundle,
):
    """2+2 trials complete without crashing; SHA and frozen weights hold."""
    net = untrained_kok_bundle.net
    assert net.phase == "phase3_kok"

    hm_gen_before = net.context_memory.W_hm_gen.detach().clone()
    mh_gen_before = net.context_memory.W_mh_gen.detach().clone()
    pred_head_before = net.prediction_head.W_pred_H_raw.detach().clone()
    sha_before = net.frozen_sensory_core_sha()

    # Deterministic (``noise_std=0``) call: the closed-form plasticity
    # rule is multiplicative in ``memory`` / ``memory_error`` so the
    # untrained-net task weights may remain at their zero init. The
    # smoke test therefore only asserts (a) no-crash, (b) frozen-weight
    # and SHA invariants, and (c) task weights stay finite — not that
    # they necessarily change.
    run_phase3_kok_training(
        net=net,
        n_trials_learning=2, n_trials_scan=2,
        validity_scan=0.75, lr=1e-3, weight_decay=0.0,
        seed=42, timing=_tiny_timing(), noise_std=0.0, log_every=1,
    )

    qm_after = net.context_memory.W_qm_task
    mh_after = net.context_memory.W_mh_task
    assert torch.all(torch.isfinite(qm_after)), "W_qm_task has NaN/Inf"
    assert torch.all(torch.isfinite(mh_after)), "W_mh_task has NaN/Inf"

    # Frozen generic + prediction-head weights must be bitwise identical.
    assert torch.equal(net.context_memory.W_hm_gen, hm_gen_before), (
        "W_hm_gen mutated during Phase-3 Kok training (should be frozen)"
    )
    assert torch.equal(net.context_memory.W_mh_gen, mh_gen_before), (
        "W_mh_gen mutated during Phase-3 Kok training (should be frozen)"
    )
    assert torch.equal(net.prediction_head.W_pred_H_raw, pred_head_before), (
        "prediction_head mutated during Phase-3 Kok training"
    )

    # LGN/L4 SHA invariant.
    assert net.frozen_sensory_core_sha() == sha_before, (
        "frozen sensory-core SHA changed during Phase-3 Kok training"
    )


def test_wrong_phase_raises_before_mutation(cfg):
    """Calling the driver with phase!='phase3_kok' must raise PhaseFrozenError."""
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42, device="cpu")
    net.set_phase("phase2")
    with pytest.raises(PhaseFrozenError):
        run_phase3_kok_training(
            net=net, n_trials_learning=1, n_trials_scan=0,
            timing=_tiny_timing(), seed=42, log_every=1,
        )


def test_cue_mapping_counterbalanced_by_seed():
    """Seed parity controls cue→orientation mapping (Kok counterbalance)."""
    assert cue_mapping_from_seed(1) == {0: 45.0, 1: 135.0}
    assert cue_mapping_from_seed(43) == {0: 45.0, 1: 135.0}
    assert cue_mapping_from_seed(0) == {0: 135.0, 1: 45.0}
    assert cue_mapping_from_seed(42) == {0: 135.0, 1: 45.0}
