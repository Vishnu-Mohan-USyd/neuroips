"""Task#74 — per-Vogels-weight isolation (post Fix O+P).

Five Vogels-plastic weights across three rule instances:
  vogels_l23 (ρ=3.0): W_pv_l23_raw (pre=r_pv), W_som_l23_raw (pre=r_som)
  vogels_h   (ρ=0.1): W_pv_h_raw   (pre=h_pv)
  vogels_ipop(ρ=1.0): l23_pv.W_pre_raw (pre=r_l23), h_pv.W_pre_raw (pre=r_h)

"Only W" conditions: keep that weight plastic; restore all others to their
pre-step value every step so apply_plasticity_step sees them but their
learning is nullified. Also capture step-1 dw sign per weight (measured
from snapshot diff after one real apply_plasticity_step).
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
import torch.nn.functional as F

from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank, _clone_world, _forward_window, _soft_reset_state,
    apply_plasticity_step, build_world, step_persistent_batch,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


# All five Vogels-plastic weights as (module, attr) tuples.
VOGELS_WEIGHTS = [
    ("l23_e",  "W_pv_l23_raw"),   # PV→L23E inhibition
    ("l23_e",  "W_som_l23_raw"),  # SOM→L23E inhibition
    ("h_e",    "W_pv_h_raw"),     # HPV→HE inhibition
    ("l23_pv", "W_pre_raw"),      # L23PV self-reg (E-pre→PV? see below)
    ("h_pv",   "W_pre_raw"),      # HPV self-reg
]


def _get_w(net, module, attr):
    return getattr(getattr(net, module), attr)


def run(name: str, *, keep_plastic: set, n_steps: int = 500, seed: int = 42) -> dict:
    """Run with only the named (module, attr) weights plastic; restore all
    other Vogels-plastic weights to their pre-step snapshot each step."""
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")
    rules = PlasticityRuleBank.from_config(
        cfg=cfg, lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=1e-4,
    )

    batch_size = 4
    worlds = [_clone_world(world) for _ in range(batch_size)]
    reset_counter = [0]
    def _reset_all():
        res = [worlds[b].reset(seed * 10_000 + reset_counter[0] * 10_000 + b)
               for b in range(batch_size)]
        reset_counter[0] += 1
        return res
    world_states = _reset_all()
    state = net.initial_state(batch_size=batch_size)
    for _ in range(30):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        _s0, _s1, state, _i0, _i1, _x0, _x1 = _forward_window(net, frames, state)

    init_raw = {(m, a): _get_w(net, m, a).mean().item() for m, a in VOGELS_WEIGHTS}
    init_eff = {(m, a): F.softplus(_get_w(net, m, a)).mean().item()
                for m, a in VOGELS_WEIGHTS}

    checkpoints = [1, 100, 300, 500]
    snap: dict = {}
    dw_step1 = {}

    for step in range(1, n_steps + 1):
        # Snapshot the weights to be frozen BEFORE the step.
        pre_freeze = {
            (m, a): _get_w(net, m, a).data.clone()
            for m, a in VOGELS_WEIGHTS if (m, a) not in keep_plastic
        }
        # Also snapshot the to-be-plastic weight's raw mean for step-1 dw capture
        step1_pre = {
            (m, a): _get_w(net, m, a).data.clone()
            for m, a in VOGELS_WEIGHTS if (m, a) in keep_plastic
        } if step == 1 else {}

        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        _ = apply_plasticity_step(net, rules, s0, s1, s2, i0, i1, x_hat_0)

        # Restore the non-plastic weights to their pre-step values.
        for (m, a), w_pre in pre_freeze.items():
            _get_w(net, m, a).data.copy_(w_pre)
        if step == 1 and step1_pre:
            for (m, a), w_pre in step1_pre.items():
                w_now = _get_w(net, m, a).data
                dw_step1[(m, a)] = {
                    "dw_mean": float((w_now - w_pre).mean().item()),
                    "dw_abs_mean": float((w_now - w_pre).abs().mean().item()),
                    "dw_sign": "+" if (w_now - w_pre).mean().item() > 0 else "-",
                }

        state = s2
        if step % 50 == 0 and step < n_steps:
            state = _soft_reset_state(state, scale=0.1)
            world_states = _reset_all()

        if step in checkpoints:
            snap[step] = {
                "l23e": float(s2.r_l23.mean().item()),
                "r_pv": float(s2.r_pv.mean().item()),
                "r_som": float(s2.r_som.mean().item()),
                "r_h": float(s2.r_h.mean().item()),
                "h_pv": float(s2.h_pv.mean().item()),
                **{
                    f"raw_{m}.{a}": _get_w(net, m, a).mean().item()
                    for m, a in VOGELS_WEIGHTS
                },
                **{
                    f"eff_{m}.{a}": F.softplus(_get_w(net, m, a)).mean().item()
                    for m, a in VOGELS_WEIGHTS
                },
            }
        if not math.isfinite(float(s2.r_l23.abs().max().item())):
            snap[step] = {"nonfinite": True}; break

    return {"name": name, "init_raw": init_raw, "init_eff": init_eff,
            "dw_step1": dw_step1, "snap": snap}


def main():
    t0 = time.monotonic()
    allw = set(VOGELS_WEIGHTS)
    conditions = [
        ("all_on",             dict(keep_plastic=allw)),
        ("vogels_pv_l23_only", dict(keep_plastic={("l23_e", "W_pv_l23_raw")})),
        ("vogels_som_l23_only",dict(keep_plastic={("l23_e", "W_som_l23_raw")})),
        ("vogels_pv_h_only",   dict(keep_plastic={("h_e",   "W_pv_h_raw")})),
        ("vogels_ipop_l23_only",dict(keep_plastic={("l23_pv","W_pre_raw")})),
        ("vogels_ipop_h_only", dict(keep_plastic={("h_pv",  "W_pre_raw")})),
        ("vogels_all_off",     dict(keep_plastic=set())),
    ]
    results = []
    for name, kwargs in conditions:
        tc = time.monotonic()
        r = run(name, **kwargs)
        print(f"[{name}] wall={time.monotonic()-tc:.1f}s", file=sys.stderr)
        results.append(r)

    print("\n=== INIT (post-warmup) ===")
    for (m, a), v in results[0]["init_raw"].items():
        e = results[0]["init_eff"][(m, a)]
        print(f"  {m}.{a}: raw_mean={v:+.4f} eff_mean={e:.5f}")

    print("\n=== STEP-1 dw SIGN (per isolated-active-weight condition) ===")
    for r in results:
        if not r["dw_step1"]:
            continue
        for (m, a), d in r["dw_step1"].items():
            print(f"  {r['name']:<22} {m}.{a}: dw_mean={d['dw_mean']:+.6f} "
                  f"sign={d['dw_sign']} |dw|={d['dw_abs_mean']:.6f}")

    print("\n=== L23E TRAJECTORY + collapse verdict ===")
    for r in results:
        s = r["snap"]
        def _g(k):
            d = s.get(k, {})
            if d.get("nonfinite"):
                return "NF"
            return f"{d.get('l23e', float('nan')):.3f}"
        d500 = s.get(500, {})
        drives = "T" if d500.get("l23e", 0) < 0.2 else "F"
        print(f"  ablation={r['name']:<22} l23e_s1={_g(1)} s100={_g(100)} "
              f"s300={_g(300)} s500={_g(500)} drives_collapse={drives}")

    print("\n=== EFFECTIVE WEIGHT CHANGE s1→s500 (all_on) ===")
    s1 = results[0]["snap"].get(1, {})
    s500 = results[0]["snap"].get(500, {})
    for m, a in VOGELS_WEIGHTS:
        key = f"eff_{m}.{a}"
        e1 = s1.get(key); e500 = s500.get(key)
        if e1 is not None and e500 is not None:
            ratio = e500 / e1 if e1 > 0 else float("inf")
            print(f"  {m}.{a}: eff_s1={e1:.5f} eff_s500={e500:.5f} ratio={ratio:.3f}×")

    print(f"\n[total wall={time.monotonic()-t0:.1f}s]")
    import json
    out = Path("logs/task74/level10_vogels_per_weight.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
