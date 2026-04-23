"""Task #74 Fix Q: ``l23_pv.W_pre_raw`` is frozen in Phase-2.

Invariant: during Phase-2 training, no plasticity rule writes to
``net.l23_pv.W_pre_raw``. It remains bit-exactly at its init value
end-to-end.

The previous wiring applied ``rules.vogels_ipop`` (Vogels iSTDP,
target=1 Hz) to this weight. Vogels iSTDP is designed for I→E synapses
— it forms the homeostatic update from ``(r_post − ρ) · r_pre`` with
the inhibitory sign convention baked into the caller. Applied to an
E→I synapse (``l23_pv.W_pre_raw``, pre=r_l23, post=r_pv), the
homeostatic sign inverts and the rule becomes anti-homeostatic: the
Level-10 Debugger (Task #74) isolated this as the driver of L23E
monotone collapse (0.328 → 0.097 Hz) and SOM exponential silencing.

Cleanest intervention — matching the Fix D-simpler pattern used for
L23SOM excitatory inputs (``test_som_excitatory_inputs_frozen_phase2.py``)
— is to freeze ``l23_pv.W_pre_raw`` at init end-to-end.

Checked two ways:
  1. :meth:`FastInhibitoryPopulation.plastic_weight_names` returns ``[]``
     for the ``l23_pv`` instance in every phase (it is constructed with
     ``freeze_W_pre=True`` in ``network.py``).
  2. A multi-step Phase-2 training run leaves ``l23_pv.W_pre_raw``
     bit-exactly equal to its init tensor.

Fix Q' (same dispatch) extended the same freeze to ``h_pv.W_pre_raw``
(HE→HPV, same E→I structural concern). Both PV-pre weights are now
frozen in Phase-2.
"""

from __future__ import annotations

import torch

from scripts.v2.train_phase2_predictive import (
    build_world,
    run_phase2_training,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


def test_l23_pv_plastic_names_empty_in_every_phase() -> None:
    """``l23_pv.plastic_weight_names()`` must be empty in every phase."""
    cfg = ModelConfig()
    net = V2Network(cfg, token_bank=None, seed=42)
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        net.set_phase(phase)
        assert net.l23_pv.plastic_weight_names() == [], (
            f"l23_pv.plastic_weight_names() non-empty in phase={phase!r}: "
            f"{net.l23_pv.plastic_weight_names()}"
        )
        top_level = [
            (m, w) for (m, w) in net.plastic_weight_names()
            if m == "l23_pv"
        ]
        assert top_level == [], (
            f"V2Network.plastic_weight_names() lists l23_pv weights in "
            f"phase={phase!r}: {top_level}"
        )


def test_phase2_training_leaves_l23_pv_W_pre_unchanged() -> None:
    """20 steps of Phase-2 training must not move ``l23_pv.W_pre_raw``."""
    cfg = ModelConfig()
    torch.manual_seed(cfg.seed)
    world, bank = build_world(cfg, seed_family="train")
    net = V2Network(cfg, token_bank=bank, seed=cfg.seed)
    net.set_phase("phase2")

    w_before = net.l23_pv.W_pre_raw.detach().clone()

    run_phase2_training(
        net=net, world=world,
        n_steps=20, batch_size=2,
        lr_urbanczik=1e-3, lr_vogels=1e-3, lr_hebb=1e-3,
        weight_decay=1e-5, beta_syn=1e-4,
        log_every=10, checkpoint_every=0,
    )

    # Bit-exact equality — any rule leaking onto this weight would fail.
    assert torch.equal(net.l23_pv.W_pre_raw, w_before), (
        "l23_pv.W_pre_raw moved during Phase-2 — must stay frozen. "
        f"max |Δ|={(net.l23_pv.W_pre_raw - w_before).abs().max().item():.3e}"
    )


def test_h_pv_W_pre_also_frozen_fix_q_prime() -> None:
    """Fix Q' extends the freeze to ``h_pv.W_pre_raw`` (HE→HPV, same E→I
    + Vogels structural bug). Must be empty in every phase and bit-exact
    across a Phase-2 training segment.
    """
    cfg = ModelConfig()
    net = V2Network(cfg, token_bank=None, seed=42)
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        net.set_phase(phase)
        assert net.h_pv.plastic_weight_names() == [], (
            f"h_pv.plastic_weight_names() non-empty in phase={phase!r}: "
            f"{net.h_pv.plastic_weight_names()}"
        )
        top_level = [
            (m, w) for (m, w) in net.plastic_weight_names()
            if m == "h_pv"
        ]
        assert top_level == [], (
            f"V2Network.plastic_weight_names() lists h_pv weights in "
            f"phase={phase!r}: {top_level}"
        )


def test_phase2_training_leaves_h_pv_W_pre_unchanged() -> None:
    """Fix Q': 20 steps of Phase-2 must not move ``h_pv.W_pre_raw``."""
    cfg = ModelConfig()
    torch.manual_seed(cfg.seed)
    world, bank = build_world(cfg, seed_family="train")
    net = V2Network(cfg, token_bank=bank, seed=cfg.seed)
    net.set_phase("phase2")

    w_before = net.h_pv.W_pre_raw.detach().clone()

    run_phase2_training(
        net=net, world=world,
        n_steps=20, batch_size=2,
        lr_urbanczik=1e-3, lr_vogels=1e-3, lr_hebb=1e-3,
        weight_decay=1e-5, beta_syn=1e-4,
        log_every=10, checkpoint_every=0,
    )

    assert torch.equal(net.h_pv.W_pre_raw, w_before), (
        "h_pv.W_pre_raw moved during Phase-2 — must stay frozen. "
        f"max |Δ|={(net.h_pv.W_pre_raw - w_before).abs().max().item():.3e}"
    )
