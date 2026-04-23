"""Task #74 Fix D-simpler: W_l23_som_raw and W_fb_som_raw are frozen in Phase-2.

Invariant: during Phase-2 training, no plasticity rule writes to
``net.l23_som.W_l23_som_raw`` or ``net.l23_som.W_fb_som_raw``. They remain
bit-exactly at their init values end-to-end.

This replaces the earlier Fix D attempt (Vogels iSTDP + Turrigiano synaptic
scaling on these weights). Multiple rules fighting on one weight drove r_som
to saturation (~500 Hz equivalent). Freezing eliminates the pathology; the
Phase-3 experiment question doesn't require plasticity on these inputs.

Checked two ways:
  1. :meth:`L23SOM.plastic_weight_names` returns ``[]`` in every phase.
  2. A multi-step Phase-2 training run leaves both raw weights bit-exactly
     equal to their init tensors.
"""

from __future__ import annotations

import torch

from scripts.v2.train_phase2_predictive import (
    build_world,
    run_phase2_training,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


def test_l23_som_plastic_names_empty_in_every_phase():
    """L23SOM must not declare any plastic weights in any phase."""
    cfg = ModelConfig()
    net = V2Network(cfg, token_bank=None, seed=42)
    for phase in ("phase2", "phase3_kok", "phase3_richter"):
        net.set_phase(phase)
        assert net.l23_som.plastic_weight_names() == [], (
            f"L23SOM.plastic_weight_names() non-empty in phase={phase!r}: "
            f"{net.l23_som.plastic_weight_names()}"
        )
        # Sanity: the module-level manifest at the network level likewise
        # excludes these two weights.
        top_level = [
            (m, w) for (m, w) in net.plastic_weight_names()
            if m == "l23_som"
        ]
        assert top_level == [], (
            f"V2Network.plastic_weight_names() lists l23_som weights in "
            f"phase={phase!r}: {top_level}"
        )


def test_phase2_training_leaves_som_excitatory_inputs_unchanged():
    """20 steps of Phase-2 training must not move either SOM-excitatory raw."""
    cfg = ModelConfig()
    torch.manual_seed(cfg.seed)
    world, bank = build_world(cfg, seed_family="train")
    net = V2Network(cfg, token_bank=bank, seed=cfg.seed)
    net.set_phase("phase2")

    w_l23_before = net.l23_som.W_l23_som_raw.detach().clone()
    w_fb_before = net.l23_som.W_fb_som_raw.detach().clone()

    run_phase2_training(
        net=net, world=world,
        n_steps=20, batch_size=2,
        lr_urbanczik=1e-3, lr_vogels=1e-3, lr_hebb=1e-3,
        weight_decay=1e-5, beta_syn=1e-4,
        log_every=10, checkpoint_every=0,
    )

    # Bit-exact equality — any rule leaking onto these weights would fail.
    # (The trainer itself also asserts this at end-of-training; this test
    # guards against a future refactor that silently removes that check.)
    assert torch.equal(net.l23_som.W_l23_som_raw, w_l23_before), (
        "W_l23_som_raw moved during Phase-2 — must stay frozen. max |Δ|="
        f"{(net.l23_som.W_l23_som_raw - w_l23_before).abs().max().item():.3e}"
    )
    assert torch.equal(net.l23_som.W_fb_som_raw, w_fb_before), (
        "W_fb_som_raw moved during Phase-2 — must stay frozen. max |Δ|="
        f"{(net.l23_som.W_fb_som_raw - w_fb_before).abs().max().item():.3e}"
    )
