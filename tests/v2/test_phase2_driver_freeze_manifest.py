"""The Phase-2 driver refuses to update a weight that is frozen under the
current phase manifest.

* :func:`run_phase2_training` requires ``net.phase == "phase2"`` — a
  Phase-3 net is rejected at entry.
* :func:`apply_plasticity_step` — the per-step update kernel — raises
  :class:`PhaseFrozenError` if a weight listed as frozen by the current
  phase manifest is targeted for update.

This prevents the common failure mode of accidentally re-training generic
weights during Phase-3 task learning (or vice versa).
"""

from __future__ import annotations

import pytest
import torch

from scripts.v2.train_phase2_predictive import (
    PhaseFrozenError,
    PlasticityRuleBank,
    apply_plasticity_step,
    build_world,
    run_phase2_training,
    sample_batch_window,
)
from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


@pytest.fixture
def world(cfg):
    world, _bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    return world


def test_run_phase2_rejects_phase3_net(cfg, net, world):
    """Calling ``run_phase2_training`` on a Phase-3 net raises immediately."""
    net.set_phase("phase3_kok")
    with pytest.raises(PhaseFrozenError, match="phase2"):
        run_phase2_training(
            net=net, world=world, n_steps=1, batch_size=2, log_every=1,
        )


def test_apply_plasticity_rejects_frozen_weight_in_phase3_kok(cfg, net, world):
    """In phase3_kok the L2/3 E plastic manifest is empty; any attempt to
    run the Phase-2 update kernel must raise PhaseFrozenError at the
    first frozen-weight access (an L2/3 Urbanczik–Senn update)."""
    net.set_phase("phase3_kok")

    # Quick forward roll to build state0/1/2 + infos the kernel needs.
    frames = sample_batch_window(world, seeds=[0, 1], n_steps_per_window=2)
    state0 = net.initial_state(batch_size=frames.shape[0])
    x_hat_0, state1, info0 = net(frames[:, 0], state0)
    _, state2, info1 = net(frames[:, 1], state1)

    rules = PlasticityRuleBank.from_config(
        cfg=cfg,
        lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=1e-4,
    )
    with pytest.raises(PhaseFrozenError):
        apply_plasticity_step(
            net=net, rules=rules,
            state0=state0, state1=state1, state2=state2,
            info0=info0, info1=info1, x_hat_0=x_hat_0,
        )


def test_phase2_plastic_manifest_covers_every_weight_kernel_touches(
    cfg, net, world,
):
    """In phase2 the kernel runs to completion — no PhaseFrozenError.

    This locks the contract: every weight the Phase-2 update kernel asks
    about must appear in :meth:`V2Network.plastic_weight_names` under
    phase2. If a new plastic weight is added to the network without being
    wired into the driver (or vice-versa), this test fails immediately.
    """
    assert net.phase == "phase2"
    frames = sample_batch_window(world, seeds=[0, 1], n_steps_per_window=2)
    state0 = net.initial_state(batch_size=frames.shape[0])
    x_hat_0, state1, info0 = net(frames[:, 0], state0)
    _, state2, info1 = net(frames[:, 1], state1)

    rules = PlasticityRuleBank.from_config(
        cfg=cfg,
        lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=1e-4,
    )
    deltas = apply_plasticity_step(
        net=net, rules=rules,
        state0=state0, state1=state1, state2=state2,
        info0=info0, info1=info1, x_hat_0=x_hat_0,
    )
    # Every (module, weight) the driver touched must be in the plastic manifest.
    manifest = set(net.plastic_weight_names())
    for key in deltas:
        mod, wname = key.split(".", 1)
        assert (mod, wname) in manifest, (
            f"driver touched {key} but it is not in net.plastic_weight_names()"
        )


def test_checkpoint_survives_phase2_training_and_reloads(cfg, net, world, tmp_path):
    """A checkpoint saved mid-training reloads into a fresh network with
    bit-identical plastic weights — guards the ``state_dict`` round-trip
    required for long multi-process runs."""
    out = tmp_path / "ckpt"
    run_phase2_training(
        net=net, world=world,
        n_steps=4, batch_size=2,
        lr_urbanczik=1e-3, lr_vogels=1e-3, lr_hebb=1e-3,
        weight_decay=1e-5, beta_syn=1e-4,
        log_every=1, checkpoint_every=2,
        checkpoint_dir=out,
    )
    ckpt_path = out / "step_4.pt"
    assert ckpt_path.exists(), f"expected checkpoint at {ckpt_path}"
    payload = torch.load(ckpt_path, weights_only=False)
    assert payload["step"] == 4
    assert payload["phase"] == "phase2"

    fresh = V2Network(cfg, token_bank=None, seed=42)
    fresh.load_state_dict(payload["state_dict"])
    for mod_name, wname in fresh.plastic_weight_names():
        w_fresh = getattr(getattr(fresh, mod_name), wname)
        w_orig = getattr(getattr(net, mod_name), wname)
        torch.testing.assert_close(w_fresh, w_orig, atol=0.0, rtol=0.0)
