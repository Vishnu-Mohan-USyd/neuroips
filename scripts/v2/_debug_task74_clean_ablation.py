"""Task#74 — clean ablation after Fix O (lr=0 is honest no-op).

Previously, setting a rule's lr=1e-20 still applied weight_decay, energy
shrinkage, and clamp-to-[-8,8] inside `_apply_update`. Fix O makes
`_apply_update` a complete no-op when `rule_lr == 0.0`. This probe sets
each rule's `.lr = 0.0` post-construction (constructor still rejects 0)
to produce honest per-rule ablations.
"""
from __future__ import annotations
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


def run(name: str, *, zero: set[str], disable_homeo: bool, n_steps: int = 500, seed: int = 42) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")
    if disable_homeo:
        net.l23_e.homeostasis.update = lambda *a, **k: None
        net.h_e.homeostasis.update = lambda *a, **k: None

    rules = PlasticityRuleBank.from_config(
        cfg=cfg, lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=1e-4,
    )
    # Post-construction zeroing (constructor still rejects lr<=0).
    if "urbanczik" in zero:
        rules.urbanczik.lr = 0.0
    if "vogels" in zero:
        rules.vogels_l23.lr = 0.0
        rules.vogels_h.lr = 0.0
        rules.vogels_ipop.lr = 0.0
    if "hebb" in zero:
        rules.hebb.lr = 0.0

    batch_size = 4
    warmup_steps = 30
    segment_length = 50
    soft_reset_scale = 0.1

    worlds = [_clone_world(world) for _ in range(batch_size)]
    reset_counter = [0]
    def _reset_all():
        res = [worlds[b].reset(seed * 10_000 + reset_counter[0] * 10_000 + b)
               for b in range(batch_size)]
        reset_counter[0] += 1
        return res
    world_states = _reset_all()
    state = net.initial_state(batch_size=batch_size)

    for _ in range(warmup_steps):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        _s0, _s1, state, _i0, _i1, _x0, _x1 = _forward_window(net, frames, state)

    checkpoints = [1, 100, 300, 500]
    snapshots: dict[int, dict] = {}

    for step in range(1, n_steps + 1):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        _ = apply_plasticity_step(net, rules, s0, s1, s2, i0, i1, x_hat_0)
        state = s2
        if segment_length > 0 and step % segment_length == 0 and step < n_steps:
            state = _soft_reset_state(state, scale=soft_reset_scale)
            world_states = _reset_all()

        if step in checkpoints:
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
    # collapse threshold for L23E at step 500: must stay > 0.2 Hz.
    conditions = [
        ("all_on",        dict(zero=set(),                   disable_homeo=False)),
        ("no_urbanczik",  dict(zero={"urbanczik"},           disable_homeo=False)),
        ("no_vogels",     dict(zero={"vogels"},              disable_homeo=False)),
        ("no_homeo",      dict(zero=set(),                   disable_homeo=True)),
        ("no_cgen",       dict(zero={"hebb"},                disable_homeo=False)),
    ]
    results = []
    for name, kwargs in conditions:
        tc = time.monotonic()
        r = run(name, **kwargs)
        print(f"[{name}] wall={time.monotonic()-tc:.1f}s", file=sys.stderr)
        results.append(r)

    print("\n=== CLEAN ABLATION RESULTS (Fix-O-era, lr=0 honest) ===")
    COLLAPSE_THRESH = 0.2
    for r in results:
        s = r["snapshots"]
        def _g(k, f):
            d = s.get(k)
            if d is None or d.get("nonfinite"):
                return "NF" if d else "NA"
            return f"{d[f]:.3f}"
        d500 = s.get(500, {})
        l23e_500 = d500.get("l23e", 0.0) if not d500.get("nonfinite") else 0.0
        prevented = "T" if l23e_500 > COLLAPSE_THRESH else "F"
        print(
            f"ablation={r['name']} "
            f"l23e_s1={_g(1,'l23e')} l23e_s100={_g(100,'l23e')} "
            f"l23e_s300={_g(300,'l23e')} l23e_s500={_g(500,'l23e')} "
            f"som_s500={_g(500,'l23som')} pv_s500={_g(500,'l23pv')} "
            f"he_s500={_g(500,'he')} hpv_s500={_g(500,'hpv')} "
            f"collapse_prevented={prevented}"
        )

    print("\n=== FULL ===")
    for r in results:
        print(f"[{r['name']}]")
        for k in sorted(r["snapshots"]):
            print(f"  step={k}: {r['snapshots'][k]}")

    print(f"\n[total wall={time.monotonic()-t0:.1f}s]")
    import json
    out = Path("logs/task74/level10_clean_ablation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
