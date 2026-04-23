"""Task #74 β-mechanism Step 3 — W_q_gain plasticity gate.

Pytest requested by Lead on 2026-04-23 to lock two invariants:

1. **Manifest**: ``l23_e.W_q_gain`` appears in ``plastic_weight_names``
   under ``phase3_kok`` and is absent under ``phase2``, ``phase3_richter``,
   and ``null_control``. This is the declarative contract the Step-3
   trainer relies on when gating ``apply_w_q_gain_update`` with a
   ``PhaseFrozenError`` check.

2. **Behavioural**: when ``enable_w_q_gain_rule=True`` and the phase is
   ``phase3_kok``, a 4-trial learning-sub-phase + 4-trial scan-sub-phase
   run actually mutates ``W_q_gain`` during learning AND leaves it
   bit-exact-unchanged across scan (mirroring the existing plasticity
   freeze of W_qm_task / W_mh_task_exc in the scan sub-phase).

Small trial counts keep this fast; we just need a signal that the path
is wired correctly — convergence behaviour is covered by the Step-2
toy and the Step-3 full-pipeline gate.
"""
from __future__ import annotations

import torch

from scripts.v2.train_phase3_kok_learning import (
    KokTiming,
    run_phase3_kok_training,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


_TIMING = KokTiming(
    cue_steps=4, delay_steps=4, probe1_steps=4, blank_steps=2, probe2_steps=4,
)


def _build_net(seed: int = 42, device: str = "cpu") -> V2Network:
    cfg = ModelConfig(seed=seed, device=device)
    bank = TokenBank(cfg, seed=0)
    return V2Network(cfg, token_bank=bank, seed=seed, device=device)


def test_w_q_gain_in_phase3_kok_plastic_manifest():
    net = _build_net()
    net.set_phase("phase3_kok")
    names = set(net.plastic_weight_names())
    assert ("l23_e", "W_q_gain") in names, (
        "Expected l23_e.W_q_gain in plastic_weight_names under phase3_kok; "
        f"got {sorted(names)}"
    )


def test_w_q_gain_frozen_outside_phase3_kok():
    net = _build_net()
    for phase in ("phase2", "phase3_richter"):
        net.set_phase(phase)
        names = set(net.plastic_weight_names())
        assert ("l23_e", "W_q_gain") not in names, (
            f"l23_e.W_q_gain must NOT be plastic under phase={phase!r}; "
            f"got {sorted(names)}"
        )


def test_w_q_gain_mutates_during_learning_and_frozen_during_scan():
    """Run a tiny phase3_kok schedule with the rule enabled and verify:
      * after the learning sub-phase W_q_gain ≠ ones (rule fired)
      * across the scan sub-phase W_q_gain is bit-exact unchanged
    """
    net = _build_net()
    net.set_phase("phase3_kok")

    # Snapshot the initial all-ones default.
    W0 = net.l23_e.W_q_gain.detach().clone()
    assert torch.equal(W0, torch.ones_like(W0))

    # Learning sub-phase only: 6 trials, 100% valid — matched=True every
    # trial, so ΔW = -lr · r_l23e · (+1); after 6 trials W_q_gain diverges
    # from ones on at least one cue row.
    run_phase3_kok_training(
        net=net, n_trials_learning=6, n_trials_scan=0,
        validity_scan=1.0, lr=1e-3, seed=42, timing=_TIMING,
        enable_w_q_gain_rule=True,
        disable_fix_j_mh_exc=True,
        w_q_gain_lr=1e-3,
        w_q_gain_clamp=(0.1, 1.0),
    )
    W_after_learn = net.l23_e.W_q_gain.detach().clone()
    assert not torch.equal(W_after_learn, W0), (
        "Expected W_q_gain to change during the learning sub-phase; "
        "unchanged after 6 matched trials."
    )

    # Scan sub-phase: apply_plasticity=False, so the rule is gated off.
    # Running another 6 trials as SCAN must leave W_q_gain bit-exact.
    run_phase3_kok_training(
        net=net, n_trials_learning=0, n_trials_scan=6,
        validity_scan=0.75, lr=1e-3, seed=43, timing=_TIMING,
        enable_w_q_gain_rule=True,
        disable_fix_j_mh_exc=True,
        w_q_gain_lr=1e-3,
        w_q_gain_clamp=(0.1, 1.0),
    )
    W_after_scan = net.l23_e.W_q_gain.detach().clone()
    assert torch.equal(W_after_scan, W_after_learn), (
        "Expected W_q_gain bit-exact-unchanged across the scan sub-phase; "
        "mutation observed."
    )
