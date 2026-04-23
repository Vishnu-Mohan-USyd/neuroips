"""Task #74 (Fix A_homeo) regression test — θ must be frozen in Phase-3.

Context: Task #74 CHECK 3 showed that 100 Phase-3-Kok trials silently drifted
``l23_e.homeostasis.theta`` (max|Δθ|=0.213) and ``h_e.homeostasis.theta``
(max|Δθ|=0.028), collapsing L2/3 orientation-coverage entropy from 1.636
(with homeostasis disabled) to 0.558 (with homeostasis active). The
Phase-3 learning drivers no longer invoke ``homeostasis.update`` (call-site
gate), and the trainers verify θ is bit-identical across training
(``_assert_theta_unchanged``, defense-in-depth).

This test asserts both legs:
  (a) The Kok and Richter drivers complete N≥10 trials with |Δθ|==0 on
      both excitatory-pop θ buffers (l23_e, h_e).
  (b) The pre-seeded non-zero θ pattern survives training identically —
      guarantees the test isn't trivially passing because θ began at zeros
      and no update would have moved it anyway.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, run_phase3_kok_training,
)
from scripts.v2.train_phase3_richter_learning import (
    RichterTiming, run_phase3_richter_training,
)
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_ckpt(cfg, tmp_path: Path, phase: str, *, seed: int = 42) -> Path:
    """Save an untrained V2Network with the requested Phase-3 phase."""
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase(phase)
    net.eval()
    path = tmp_path / f"{phase}_step_0.pt"
    torch.save(
        {
            "step": 0, "state_dict": net.state_dict(),
            "phase": phase, "frozen_sha": net.frozen_sensory_core_sha(),
        },
        path,
    )
    return path


def _seed_theta(net: V2Network, seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace θ on l23_e and h_e with a reproducible non-zero pattern.

    Guarantees the frozen-θ assertion is non-trivial — if the driver
    silently zeroed θ (e.g. by reassignment) we'd detect it.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        net.l23_e.homeostasis.theta.copy_(
            0.05 * torch.randn(
                net.l23_e.homeostasis.n_units, generator=g, dtype=torch.float32,
            ),
        )
        net.h_e.homeostasis.theta.copy_(
            0.05 * torch.randn(
                net.h_e.homeostasis.n_units, generator=g, dtype=torch.float32,
            ),
        )
    return (
        net.l23_e.homeostasis.theta.detach().clone(),
        net.h_e.homeostasis.theta.detach().clone(),
    )


def _tiny_kok_timing() -> KokTiming:
    return KokTiming(
        cue_steps=2, delay_steps=2, probe1_steps=2,
        blank_steps=1, probe2_steps=2,
    )


def _tiny_richter_timing() -> RichterTiming:
    return RichterTiming(leader_steps=2, trailer_steps=2, iti_steps=0)


def test_theta_frozen_across_phase3_kok_training(cfg, tmp_path):
    """10 Kok trials on non-zero θ → θ bit-identical (max|Δθ|==0)."""
    bundle = load_checkpoint(
        _write_ckpt(cfg, tmp_path, "phase3_kok"), seed=42, device="cpu",
    )
    net = bundle.net
    assert net.phase == "phase3_kok"

    theta_l23_before, theta_h_before = _seed_theta(net, seed=1234)

    run_phase3_kok_training(
        net=net,
        n_trials_learning=6, n_trials_scan=4,
        validity_scan=0.75, lr=1e-3, weight_decay=0.0,
        seed=42, timing=_tiny_kok_timing(), noise_std=0.0, log_every=1,
    )

    theta_l23_after = net.l23_e.homeostasis.theta.data
    theta_h_after = net.h_e.homeostasis.theta.data
    max_abs_delta_l23 = float((theta_l23_after - theta_l23_before).abs().max().item())
    max_abs_delta_h = float((theta_h_after - theta_h_before).abs().max().item())

    assert torch.equal(theta_l23_before, theta_l23_after), (
        f"l23_e.homeostasis.theta drifted during Phase-3 Kok training "
        f"(max|Δθ|={max_abs_delta_l23:.3e})"
    )
    assert torch.equal(theta_h_before, theta_h_after), (
        f"h_e.homeostasis.theta drifted during Phase-3 Kok training "
        f"(max|Δθ|={max_abs_delta_h:.3e})"
    )
    # Non-triviality: the seeded pattern was non-zero.
    assert float(theta_l23_before.abs().max().item()) > 1e-3
    assert float(theta_h_before.abs().max().item()) > 1e-3


def test_theta_frozen_across_phase3_richter_training(cfg, tmp_path):
    """10 Richter trials on non-zero θ → θ bit-identical (max|Δθ|==0)."""
    bundle = load_checkpoint(
        _write_ckpt(cfg, tmp_path, "phase3_richter"), seed=42, device="cpu",
    )
    net = bundle.net
    bank = bundle.bank
    assert net.phase == "phase3_richter"

    theta_l23_before, theta_h_before = _seed_theta(net, seed=5678)

    run_phase3_richter_training(
        net=net, bank=bank,
        n_trials_learning=6, n_trials_scan=4,
        reliability_scan=0.5, lr=1e-3, weight_decay=0.0,
        seed=42, timing=_tiny_richter_timing(), noise_std=0.0, log_every=1,
    )

    theta_l23_after = net.l23_e.homeostasis.theta.data
    theta_h_after = net.h_e.homeostasis.theta.data
    max_abs_delta_l23 = float((theta_l23_after - theta_l23_before).abs().max().item())
    max_abs_delta_h = float((theta_h_after - theta_h_before).abs().max().item())

    assert torch.equal(theta_l23_before, theta_l23_after), (
        f"l23_e.homeostasis.theta drifted during Phase-3 Richter training "
        f"(max|Δθ|={max_abs_delta_l23:.3e})"
    )
    assert torch.equal(theta_h_before, theta_h_after), (
        f"h_e.homeostasis.theta drifted during Phase-3 Richter training "
        f"(max|Δθ|={max_abs_delta_h:.3e})"
    )
    assert float(theta_l23_before.abs().max().item()) > 1e-3
    assert float(theta_h_before.abs().max().item()) > 1e-3
