"""Smoke tests for the Phase-3 Richter task-learning driver.

Exercises :func:`run_phase3_richter_training` on an untrained network with
a micro-budget (2 learning + 2 scan trials, short epochs). Verifies the
Task-#40 critical invariants:

* The driver runs end-to-end on an untrained net without crashing.
* ``frozen_sensory_core_sha`` is identical before and after training.
* Every non-plastic Parameter is left bitwise-identical.
* The two task weights (``W_lm_task``, ``W_mh_task``) remain finite.
* Running under the wrong phase raises :class:`PhaseFrozenError`.
* Permutation is seed-deterministic and uses all 6 trailer positions.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.train_phase2_predictive import PhaseFrozenError
from scripts.v2.train_phase3_richter_learning import (
    N_LEAD_TRAIL, RichterTiming, permutation_from_seed,
    run_phase3_richter_training,
)
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_phase3_richter_checkpoint(
    cfg, tmp_path: Path, *, seed: int = 42,
) -> Path:
    """Save an untrained V2Network with ``phase='phase3_richter'``."""
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase("phase3_richter")
    net.eval()
    path = tmp_path / "phase3_richter_step_0.pt"
    torch.save({
        "step": 0, "state_dict": net.state_dict(),
        "phase": "phase3_richter", "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path


@pytest.fixture
def untrained_richter_bundle(cfg, tmp_path):
    return load_checkpoint(
        _write_phase3_richter_checkpoint(cfg, tmp_path),
        seed=42, device="cpu",
    )


def _tiny_timing() -> RichterTiming:
    """Micro-budget timing: 4 steps per trial instead of 200."""
    return RichterTiming(leader_steps=2, trailer_steps=2, iti_steps=0)


def test_tiny_training_runs_and_preserves_frozen_invariants(
    untrained_richter_bundle,
):
    """2+2 trials complete without crashing; SHA and frozen weights hold."""
    net = untrained_richter_bundle.net
    bank = untrained_richter_bundle.bank
    assert net.phase == "phase3_richter"

    hm_gen_before = net.context_memory.W_hm_gen.detach().clone()
    mh_gen_before = net.context_memory.W_mh_gen.detach().clone()
    pred_head_before = net.prediction_head.W_pred_H_raw.detach().clone()
    sha_before = net.frozen_sensory_core_sha()

    # Deterministic call (``noise_std=0``) for a crash-free smoke test.
    # Closed-form plasticity is multiplicative in memory activity, so the
    # untrained net's task weights may stay at zero; the smoke test only
    # verifies (a) no crash, (b) SHA + frozen-weight invariants, and
    # (c) finite task weights.
    run_phase3_richter_training(
        net=net, bank=bank,
        n_trials_learning=2, n_trials_scan=2,
        reliability_scan=0.5, lr=1e-3, weight_decay=0.0,
        seed=42, timing=_tiny_timing(), noise_std=0.0, log_every=1,
    )

    lm_after = net.context_memory.W_lm_task
    mh_after = net.context_memory.W_mh_task
    assert torch.all(torch.isfinite(lm_after)), "W_lm_task has NaN/Inf"
    assert torch.all(torch.isfinite(mh_after)), "W_mh_task has NaN/Inf"

    # Frozen generic + prediction-head weights unchanged.
    assert torch.equal(net.context_memory.W_hm_gen, hm_gen_before), (
        "W_hm_gen mutated during Phase-3 Richter training"
    )
    assert torch.equal(net.context_memory.W_mh_gen, mh_gen_before), (
        "W_mh_gen mutated during Phase-3 Richter training"
    )
    assert torch.equal(net.prediction_head.W_pred_H_raw, pred_head_before), (
        "prediction_head mutated during Phase-3 Richter training"
    )

    assert net.frozen_sensory_core_sha() == sha_before, (
        "frozen sensory-core SHA changed during Phase-3 Richter training"
    )


def test_wrong_phase_raises_before_mutation(cfg):
    """Calling the driver with phase!='phase3_richter' raises PhaseFrozenError."""
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42, device="cpu")
    net.set_phase("phase2")
    with pytest.raises(PhaseFrozenError):
        run_phase3_richter_training(
            net=net, bank=bank,
            n_trials_learning=1, n_trials_scan=0,
            timing=_tiny_timing(), seed=42, log_every=1,
        )


def test_permutation_is_seed_deterministic_and_covers_all_positions():
    """σ(·) is a permutation of {0..5} and stable under fixed seed."""
    p1 = permutation_from_seed(42)
    p2 = permutation_from_seed(42)
    assert p1 == p2, "permutation is not seed-deterministic"
    assert len(p1) == N_LEAD_TRAIL
    assert set(p1) == set(range(N_LEAD_TRAIL)), (
        f"permutation is not a full bijection of {N_LEAD_TRAIL} positions: {p1}"
    )
    assert permutation_from_seed(7) != p1, (
        "different seeds produced identical permutations — RNG wiring broken"
    )
