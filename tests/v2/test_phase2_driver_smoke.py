"""10-step Phase-2 driver smoke test.

Exercises the full closed-form plasticity path on a micro-budget:
  * No NaN / inf in any rate or plastic weight over 10 steps.
  * At least one plastic weight changes measurably between start and end.
  * LGN/L4 frozen-core SHA is unchanged after training.
  * ``run_phase2_training`` honours the Phase-2 plastic manifest — no
    attempts to mutate a frozen weight (tested implicitly: an attempt
    would raise :class:`PhaseFrozenError`, failing the test).
"""

from __future__ import annotations

import pytest
import torch

from scripts.v2.train_phase2_predictive import (
    build_world,
    run_phase2_training,
)
from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


@pytest.fixture
def world(cfg):
    world, _bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    return world


def test_10_step_training_runs_and_mutates_weights(cfg, net, world):
    """10 steps produce finite rates, changed weights, unchanged LGN/L4."""
    # Snapshot every plastic raw weight before training.
    snap_before: dict[tuple[str, str], torch.Tensor] = {}
    for mod_name, wname in net.plastic_weight_names():
        w = getattr(getattr(net, mod_name), wname)
        snap_before[(mod_name, wname)] = w.detach().clone()

    sha_before = net.frozen_sensory_core_sha()

    history = run_phase2_training(
        net=net, world=world,
        n_steps=10, batch_size=2,
        lr_urbanczik=1e-3, lr_vogels=1e-3, lr_hebb=1e-3,
        weight_decay=1e-5, beta_syn=1e-4,
        log_every=1, checkpoint_every=0,
    )

    assert len(history) == 10, (
        f"expected 10 logged steps (log_every=1); got {len(history)}"
    )

    # Finite rates everywhere (sampled via the metrics record).
    for m in history:
        assert torch.isfinite(torch.tensor(m.loss_pred)), (
            f"non-finite loss at step {m.step}"
        )
        assert torch.isfinite(torch.tensor(m.eps_abs_mean))
        assert torch.isfinite(torch.tensor(m.r_l23_mean))
        assert torch.isfinite(torch.tensor(m.r_h_mean))

    # Finite plastic weights.
    for (mod_name, wname), _before in snap_before.items():
        w_after = getattr(getattr(net, mod_name), wname)
        assert torch.all(torch.isfinite(w_after)), (
            f"{mod_name}.{wname} contains NaN/inf after training"
        )

    # At least one weight must have moved. Every spec matters — record the
    # number that moved to guard against a silent "all rules are no-ops" bug.
    n_moved = 0
    for (mod_name, wname), before in snap_before.items():
        w_after = getattr(getattr(net, mod_name), wname)
        if not torch.equal(w_after, before):
            n_moved += 1
    assert n_moved > 0, (
        "no plastic weight was mutated in 10 steps — plasticity wiring broken"
    )

    # Frozen-core SHA unchanged.
    sha_after = net.frozen_sensory_core_sha()
    assert sha_after == sha_before, (
        "LGN/L4 frozen-core SHA changed during training — mutation leaked "
        "into the fixed sensory front end"
    )


def test_10_step_training_eps_is_finite_and_bounded(cfg, net, world):
    """|eps| is finite and of order the prediction-head init (~softplus(-5))
    at every step. A catastrophic divergence would blow past ~1e3 within
    a few steps at these LRs — guard against that regression."""
    history = run_phase2_training(
        net=net, world=world,
        n_steps=10, batch_size=2,
        lr_urbanczik=1e-3, lr_vogels=1e-3, lr_hebb=1e-3,
        weight_decay=1e-5, beta_syn=1e-4,
        log_every=1,
    )
    for m in history:
        assert 0.0 <= m.eps_abs_mean < 1e3, (
            f"|eps| at step {m.step} = {m.eps_abs_mean:.3e} outside bounds"
        )
        assert 0.0 <= m.r_l23_mean < 1e4, (
            f"r_l23 mean at step {m.step} = {m.r_l23_mean:.3e} outside bounds"
        )
