"""Task #59 — instrumented debug harness for Phase-2 stateful plasticity explosion.

Reproduces Task #56's stateful-Phase-2 run with per-step instrumentation:
  * per-rule delta_abs_max
  * state field magnitudes
  * bookkeeping fields (pre_traces, post_traces, regime_posterior)
  * plastic weight norms
  * homeostatic theta drift

Exercises varied soft-reset configs to isolate the explosion mechanism.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import NetworkStateV2
from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank, apply_plasticity_step, build_world, sample_batch_window,
    _forward_window, _soft_reset_state,
)


# ============================================================================
# Instrumentation
# ============================================================================

@dataclass
class StepSnapshot:
    step: int
    r_l23_max: float
    r_l23_mean: float
    r_h_max: float
    r_h_mean: float
    m_norm: float
    theta_l23_mean: float
    theta_h_mean: float
    pre_traces_nkeys: int
    post_traces_nkeys: int
    regime_post_norm: float
    deltas: dict[str, float]
    weights: dict[str, float]
    any_nan: bool


def _snapshot_weights(net: V2Network) -> dict[str, float]:
    """Mean-abs of softplus(raw) for the interesting plastic weights."""
    out = {}
    # L2/3 E pyramidal weights (Urbanczik)
    for wname in ("W_l4_l23_raw", "W_rec_raw", "W_fb_apical_raw"):
        if hasattr(net.l23_e, wname):
            out[f"l23_e.{wname}"] = F.softplus(
                getattr(net.l23_e, wname)
            ).mean().item()
    # L2/3 inhibitory
    for wname in ("W_pv_l23_raw", "W_som_l23_raw"):
        if hasattr(net.l23_e, wname):
            out[f"l23_e.{wname}"] = F.softplus(
                getattr(net.l23_e, wname)
            ).mean().item()
    # H recurrent
    for wname in ("W_l23_h_raw", "W_rec_raw", "W_pv_h_raw"):
        if hasattr(net.h_e, wname):
            out[f"h_e.{wname}"] = F.softplus(
                getattr(net.h_e, wname)
            ).mean().item()
    # Prediction head
    for wname in ("W_pred_H_raw", "W_pred_C_raw", "W_pred_apical_raw",
                  "b_pred_raw"):
        if hasattr(net.prediction_head, wname):
            out[f"prediction_head.{wname}"] = F.softplus(
                getattr(net.prediction_head, wname)
            ).mean().item()
    return out


def _state_any_nan(state: NetworkStateV2) -> bool:
    for name in ("r_l4", "r_l23", "r_pv", "r_som", "r_h", "h_pv", "m"):
        v = getattr(state, name)
        if not torch.isfinite(v).all().item():
            return True
    return False


def _snapshot_step(
    step: int, state2: NetworkStateV2, net: V2Network,
    deltas: dict[str, float],
) -> StepSnapshot:
    theta_l23 = (
        net.l23_e.homeostasis.theta.mean().item()
        if hasattr(net.l23_e, "homeostasis") else float("nan")
    )
    theta_h = (
        net.h_e.homeostasis.theta.mean().item()
        if hasattr(net.h_e, "homeostasis") else float("nan")
    )
    return StepSnapshot(
        step=step,
        r_l23_max=float(state2.r_l23.abs().max().item()),
        r_l23_mean=float(state2.r_l23.mean().item()),
        r_h_max=float(state2.r_h.abs().max().item()),
        r_h_mean=float(state2.r_h.mean().item()),
        m_norm=float(state2.m.norm().item()),
        theta_l23_mean=theta_l23,
        theta_h_mean=theta_h,
        pre_traces_nkeys=len(state2.pre_traces),
        post_traces_nkeys=len(state2.post_traces),
        regime_post_norm=float(state2.regime_posterior.norm().item()),
        deltas=dict(deltas),
        weights=_snapshot_weights(net),
        any_nan=_state_any_nan(state2),
    )


# ============================================================================
# Instrumented training loop
# ============================================================================

def run_instrumented(
    seed: int = 42,
    n_steps: int = 300,
    batch_size: int = 4,
    warmup_steps: int = 30,
    segment_length: int = 50,
    soft_reset_scale: float = 0.1,
    lr_urbanczik: float = 1e-4,
    lr_vogels: float = 1e-4,
    lr_hebb: float = 1e-4,
    weight_decay: float = 1e-5,
    beta_syn: float = 1e-4,
) -> list[StepSnapshot]:
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")

    rules = PlasticityRuleBank.from_config(
        cfg=cfg,
        lr_urbanczik=lr_urbanczik, lr_vogels=lr_vogels, lr_hebb=lr_hebb,
        weight_decay=weight_decay, beta_syn=beta_syn,
    )

    state = net.initial_state(batch_size=batch_size)

    # Warmup
    for w in range(warmup_steps):
        seeds = [seed * 10_000 + w * batch_size + b for b in range(batch_size)]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)
        _, _, state, _, _, _, _ = _forward_window(net, frames, state)

    snapshots: list[StepSnapshot] = []
    for step in range(n_steps):
        seeds = [
            seed * 10_000 + (warmup_steps + step) * batch_size + b
            for b in range(batch_size)
        ]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)
        try:
            (
                state0, state1, state2, info0, info1, x_hat_0, _x_hat_1,
            ) = _forward_window(net, frames, state)
            delta_per_w = apply_plasticity_step(
                net, rules, state0, state1, state2, info0, info1, x_hat_0,
            )
        except Exception as e:
            print(f"[step {step}] EXCEPTION: {e}")
            break

        snap = _snapshot_step(step, state2, net, delta_per_w)
        snapshots.append(snap)

        state = state2
        if (
            segment_length > 0
            and (step + 1) % segment_length == 0
            and (step + 1) < n_steps
        ):
            state = _soft_reset_state(state, scale=soft_reset_scale)

        if snap.any_nan:
            print(f"[step {step}] state has NaN — halting")
            break

    return snapshots


# ============================================================================
# Printers
# ============================================================================

def print_run_summary(label: str, snaps: list[StepSnapshot]) -> None:
    print(f"\n{'='*76}")
    print(f"=== {label}   (steps run: {len(snaps)}) ===")
    print(f"{'='*76}")
    if not snaps:
        print("NO SNAPSHOTS")
        return

    print(f"\n{'step':>5}  {'|r_l23|max':>11}  {'|r_h|max':>11}  "
          f"{'m_norm':>11}  {'θ_l23':>11}  {'θ_h':>11}  "
          f"{'maxδ':>11}  {'nan':>4}")
    for i, s in enumerate(snaps):
        if i < 3 or i % 20 == 0 or i == len(snaps) - 1 or s.any_nan:
            maxd = max(s.deltas.values()) if s.deltas else 0.0
            print(f"{s.step:>5}  {s.r_l23_max:>11.3e}  {s.r_h_max:>11.3e}  "
                  f"{s.m_norm:>11.3e}  {s.theta_l23_mean:>+11.3e}  "
                  f"{s.theta_h_mean:>+11.3e}  {maxd:>11.3e}  "
                  f"{s.any_nan!s:>4}")

    # Bookkeeping summary — should stay constant
    first, last = snaps[0], snaps[-1]
    print(f"\n  bookkeeping:")
    print(f"    pre_traces keys:   t=0 → {first.pre_traces_nkeys}, "
          f"t={last.step} → {last.pre_traces_nkeys}")
    print(f"    post_traces keys:  t=0 → {first.post_traces_nkeys}, "
          f"t={last.step} → {last.post_traces_nkeys}")
    print(f"    regime_post.norm:  t=0 → {first.regime_post_norm:.4f}, "
          f"t={last.step} → {last.regime_post_norm:.4f}")

    # Per-rule δ progression — find first rule to exceed threshold
    print(f"\n  per-rule δ trajectory (max over all weights per rule type):")
    rule_keys = list(snaps[-1].deltas.keys()) if snaps[-1].deltas else []
    print(f"  {'step':>5}  " + "  ".join(f"{k[:24]:>24}" for k in rule_keys[:6]))
    for i, s in enumerate(snaps):
        if i < 3 or i % 40 == 0 or i == len(snaps) - 1 or s.any_nan:
            row = f"  {s.step:>5}  " + "  ".join(
                f"{s.deltas.get(k, 0.0):>24.3e}" for k in rule_keys[:6]
            )
            print(row)


def identify_first_explosive_rule(snaps: list[StepSnapshot]) -> None:
    """Walk per-rule deltas chronologically; identify first rule to exceed 1e3."""
    if not snaps:
        return
    print(f"\n  FIRST EXPLOSIVE RULE ANALYSIS (threshold = |δ| > 1e3):")
    thresh = 1e3
    by_rule: dict[str, Optional[int]] = {}
    for s in snaps:
        for rk, dv in s.deltas.items():
            if rk not in by_rule and abs(dv) > thresh:
                by_rule[rk] = s.step
    ranked = sorted(by_rule.items(), key=lambda kv: (kv[1] is None, kv[1]))
    for rk, st in ranked[:10]:
        print(f"    {rk:>35} first > 1e3 at step {st}")


def print_weight_evolution(label: str, snaps: list[StepSnapshot]) -> None:
    if not snaps:
        return
    print(f"\n  weight (softplus mean) evolution [{label}]:")
    keys = sorted(snaps[0].weights.keys())[:8]
    print(f"  {'step':>5}  " + "  ".join(f"{k[-20:]:>20}" for k in keys))
    for i, s in enumerate(snaps):
        if i < 3 or i % 40 == 0 or i == len(snaps) - 1 or s.any_nan:
            row = f"  {s.step:>5}  " + "  ".join(
                f"{s.weights.get(k, 0.0):>20.3e}" for k in keys
            )
            print(row)


# ============================================================================
# Main — run 4 configs, compare
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="all",
                        choices=["all", "default", "hard_reset", "no_reset",
                                 "short_segment"])
    parser.add_argument("--n-steps", type=int, default=300)
    args = parser.parse_args()

    configs = {
        "default": dict(segment_length=50, soft_reset_scale=0.1),
        "hard_reset": dict(segment_length=50, soft_reset_scale=0.0),
        "no_reset": dict(segment_length=50, soft_reset_scale=1.0),
        "short_segment": dict(segment_length=10, soft_reset_scale=0.1),
    }

    for name, extra in configs.items():
        if args.run not in ("all", name):
            continue
        print(f"\n\n{'#'*76}")
        print(f"# Config: {name}   params={extra}")
        print(f"{'#'*76}")
        snaps = run_instrumented(
            seed=42, n_steps=args.n_steps, batch_size=4,
            warmup_steps=30, **extra,
        )
        print_run_summary(name, snaps)
        identify_first_explosive_rule(snaps)
        print_weight_evolution(name, snaps)


if __name__ == "__main__":
    main()
