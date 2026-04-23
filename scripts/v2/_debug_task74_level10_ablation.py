"""Task#74 Level-10 ablation — isolate which plasticity rule drives collapse.

Runs 5 conditions × 500 steps each, measures r_l23 / r_som / r_hpv at
steps 1, 100, 300, 500. Uses the same Fix-K+L2+M+N substrate + wiring as
level_10_whole_network_stability.py.

Ablations:
  1. baseline:      all plasticity on (reproduces failure)
  2. no_urbanczik:  lr_urbanczik=0  (kills L23 apical/rec updates + pred head)
  3. no_vogels:     lr_vogels=0     (kills inhibitory plasticity)
  4. no_homeostasis: monkey-patch net.l23_e/h_e.homeostasis.update → no-op
  5. no_hebb:       lr_hebb=0       (kills C generic + H hebb updates)
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank, _clone_world, _forward_window, _soft_reset_state,
    apply_plasticity_step, build_world, step_persistent_batch,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


def run_condition(
    name: str, n_steps: int = 500, seed: int = 42,
    lr_urbanczik: float = 1e-4, lr_vogels: float = 1e-4, lr_hebb: float = 1e-4,
    disable_homeostasis: bool = False,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")

    if disable_homeostasis:
        net.l23_e.homeostasis.update = lambda *a, **k: None
        net.h_e.homeostasis.update = lambda *a, **k: None

    rules = PlasticityRuleBank.from_config(
        cfg=cfg, lr_urbanczik=lr_urbanczik, lr_vogels=lr_vogels,
        lr_hebb=lr_hebb, weight_decay=1e-5, beta_syn=1e-4,
    )

    batch_size = 4
    warmup_steps = 30
    segment_length = 50
    soft_reset_scale = 0.1

    worlds = [_clone_world(world) for _ in range(batch_size)]
    reset_counter = [0]
    def _reset_all_worlds():
        res = [worlds[b].reset(seed * 10_000 + reset_counter[0] * 10_000 + b)
               for b in range(batch_size)]
        reset_counter[0] += 1
        return res
    world_states = _reset_all_worlds()
    state = net.initial_state(batch_size=batch_size)

    for _ in range(warmup_steps):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        _s0, _s1, state, _i0, _i1, _x0, _x1 = _forward_window(
            net, frames, state)

    checkpoint_steps = [1, 100, 300, 500]
    snapshots: dict[int, dict] = {}

    for step in range(1, n_steps + 1):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        _delta = apply_plasticity_step(net, rules, s0, s1, s2, i0, i1, x_hat_0)
        state = s2
        if segment_length > 0 and step % segment_length == 0 and step < n_steps:
            state = _soft_reset_state(state, scale=soft_reset_scale)
            world_states = _reset_all_worlds()

        if step in checkpoint_steps:
            snapshots[step] = {
                "l23e":   float(s2.r_l23.mean().item()),
                "l23pv":  float(s2.r_pv.mean().item()),
                "l23som": float(s2.r_som.mean().item()),
                "he":     float(s2.r_h.mean().item()),
                "hpv":    float(s2.h_pv.mean().item()),
            }

        if not math.isfinite(float(s2.r_l23.abs().max().item())):
            snapshots[step] = {"nonfinite": True}
            break

    return {"name": name, "snapshots": snapshots}


def main():
    t0 = time.monotonic()
    # Use lr=1e-20 as effective-zero (constructor rejects lr<=0).
    EPS = 1e-20
    conditions = [
        ("baseline",       dict()),
        ("no_urbanczik",   dict(lr_urbanczik=EPS)),
        ("no_vogels",      dict(lr_vogels=EPS)),
        ("no_homeostasis", dict(disable_homeostasis=True)),
        ("no_hebb",        dict(lr_hebb=EPS)),
        ("FROZEN_ALL",     dict(lr_urbanczik=EPS, lr_vogels=EPS, lr_hebb=EPS,
                                disable_homeostasis=True)),
    ]
    results = []
    for name, kwargs in conditions:
        t_cond = time.monotonic()
        r = run_condition(name, n_steps=500, seed=42, **kwargs)
        dt = time.monotonic() - t_cond
        print(f"[{name}] wall={dt:.1f}s", file=sys.stderr)
        results.append(r)

    print("\n=== ABLATION RESULTS ===")
    for r in results:
        s = r["snapshots"]
        def _g(k, f):
            d = s.get(k)
            if d is None or d.get("nonfinite"):
                return "NF" if d else "NA"
            return f"{d[f]:.3f}"
        line = (
            f"ablation={r['name']} "
            f"l23e_rate_step1={_g(1,'l23e')} "
            f"l23e_rate_step100={_g(100,'l23e')} "
            f"l23e_rate_step300={_g(300,'l23e')} "
            f"l23e_rate_step500={_g(500,'l23e')} "
            f"l23som_step500={_g(500,'l23som')} "
            f"l23pv_step500={_g(500,'l23pv')} "
            f"he_step500={_g(500,'he')} "
            f"hpv_step500={_g(500,'hpv')}"
        )
        print(line)

    # Full snapshots for reference
    print("\n=== FULL SNAPSHOTS ===")
    for r in results:
        print(f"[{r['name']}]")
        for k in sorted(r["snapshots"]):
            print(f"  step={k}: {r['snapshots'][k]}")

    print(f"\n[total wall={time.monotonic()-t0:.1f}s]")

    out = Path("logs/task74/level10_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
