"""Task#74 — Vogels sub-rule isolation + per-weight trajectory.

Three Vogels rule instances: vogels_l23 (target=3.0 Hz, pushes L23E up),
vogels_h (target=0.1 Hz, pushes HE), vogels_ipop (target=1.0, PV self-reg).
vogels_l23 applies to W_pv_l23 and W_som_l23; vogels_h to W_pv_h;
vogels_ipop to l23_pv.W_pre and h_pv.W_pre.

Ablate each sub-rule individually, plus (W_pv_l23_only, W_som_l23_only).
Also track per-step dw magnitude + raw-weight mean for W_pv_l23 and W_som_l23
in all_on and dissect which weight drives the collapse.
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


def _capture_vogels_subweight_delta(net: V2Network, rules: PlasticityRuleBank,
                                    s1, s2) -> dict:
    """Compute the Vogels dw (before apply) for W_pv_l23 and W_som_l23."""
    from scripts.v2.train_phase2_predictive import _raw_prior
    out = {}
    for wname, pre in (("W_pv_l23_raw", s1.r_pv), ("W_som_l23_raw", s1.r_som)):
        w = net.l23_e.__getattr__(wname)
        dw = rules.vogels_l23.delta(
            pre_activity=pre, post_activity=s2.r_l23, weights=w,
            raw_prior=_raw_prior(net, "l23_e", wname, w),
        )
        out[wname] = {
            "dw_mean": float(dw.mean().item()),
            "dw_abs_mean": float(dw.abs().mean().item()),
            "dw_sign_pos_frac": float((dw > 0).float().mean().item()),
            "w_raw_mean": float(w.mean().item()),
            "w_eff_mean": float(F.softplus(w).mean().item()),
        }
    return out


def run(name: str, *, zero_subrules: set[str], zero_weights: set[str],
        capture_dw: bool = False, n_steps: int = 500, seed: int = 42) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")

    rules = PlasticityRuleBank.from_config(
        cfg=cfg, lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=1e-4,
    )
    if "vogels_l23" in zero_subrules:
        rules.vogels_l23.lr = 0.0
    if "vogels_h" in zero_subrules:
        rules.vogels_h.lr = 0.0
    if "vogels_ipop" in zero_subrules:
        rules.vogels_ipop.lr = 0.0

    # Per-weight gating: freeze the specified raw weight by overriding .lr=0
    # on the rule-side is coarse. Finer: monkey-patch the delta to return zeros
    # for those specific calls. Simplest: clone the original weight and restore
    # after each apply_plasticity_step.
    frozen_snaps = {}
    for wname in zero_weights:
        w = net.l23_e.__getattr__(wname)
        frozen_snaps[wname] = w.data.clone()

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

    # Capture init state (post-warmup) for mechanism trace
    init_snap = {
        "r_l23_mean": float(state.r_l23.mean().item()),
        "r_pv_mean": float(state.r_pv.mean().item()),
        "r_som_mean": float(state.r_som.mean().item()),
        "W_pv_l23_raw_mean": float(net.l23_e.W_pv_l23_raw.mean().item()),
        "W_pv_l23_eff_mean": float(F.softplus(net.l23_e.W_pv_l23_raw).mean().item()),
        "W_som_l23_raw_mean": float(net.l23_e.W_som_l23_raw.mean().item()),
        "W_som_l23_eff_mean": float(F.softplus(net.l23_e.W_som_l23_raw).mean().item()),
    }

    checkpoints = [1, 100, 300, 500]
    snapshots: dict = {"init": init_snap}
    dw_step1 = None

    for step in range(1, n_steps + 1):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        if capture_dw and step == 1:
            dw_step1 = _capture_vogels_subweight_delta(net, rules, s1, s2)
        _ = apply_plasticity_step(net, rules, s0, s1, s2, i0, i1, x_hat_0)
        # Restore frozen weights
        for wname, wsnap in frozen_snaps.items():
            net.l23_e.__getattr__(wname).data.copy_(wsnap)
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
                "W_pv_l23_raw_mean": float(net.l23_e.W_pv_l23_raw.mean().item()),
                "W_pv_l23_eff_mean": float(F.softplus(net.l23_e.W_pv_l23_raw).mean().item()),
                "W_som_l23_raw_mean": float(net.l23_e.W_som_l23_raw.mean().item()),
                "W_som_l23_eff_mean": float(F.softplus(net.l23_e.W_som_l23_raw).mean().item()),
            }
        if not math.isfinite(float(s2.r_l23.abs().max().item())):
            snapshots[step] = {"nonfinite": True}
            break

    return {"name": name, "dw_step1": dw_step1, "snapshots": snapshots}


def main():
    t0 = time.monotonic()
    conditions = [
        ("all_on",          dict(zero_subrules=set(), zero_weights=set(), capture_dw=True)),
        ("no_vogels_l23",   dict(zero_subrules={"vogels_l23"}, zero_weights=set())),
        ("no_vogels_h",     dict(zero_subrules={"vogels_h"}, zero_weights=set())),
        ("no_vogels_ipop",  dict(zero_subrules={"vogels_ipop"}, zero_weights=set())),
        ("freeze_W_pv_l23", dict(zero_subrules=set(), zero_weights={"W_pv_l23_raw"})),
        ("freeze_W_som_l23", dict(zero_subrules=set(), zero_weights={"W_som_l23_raw"})),
        ("freeze_both_l23inh", dict(zero_subrules=set(),
                                   zero_weights={"W_pv_l23_raw", "W_som_l23_raw"})),
    ]
    results = []
    for name, kwargs in conditions:
        tc = time.monotonic()
        r = run(name, **kwargs)
        print(f"[{name}] wall={time.monotonic()-tc:.1f}s", file=sys.stderr)
        results.append(r)

    print("\n=== INIT STATE (post-warmup) ===")
    init = results[0]["snapshots"]["init"]
    for k, v in init.items():
        print(f"  {k}={v:.5f}")

    print("\n=== STEP-1 VOGELS dw (for vogels_l23 weights) ===")
    for k, v in (results[0].get("dw_step1") or {}).items():
        print(f"  {k}: dw_mean={v['dw_mean']:+.6f} dw_abs_mean={v['dw_abs_mean']:.6f} "
              f"frac_pos={v['dw_sign_pos_frac']:.3f} w_raw_mean={v['w_raw_mean']:+.4f} "
              f"w_eff_mean={v['w_eff_mean']:.5f}")

    print("\n=== SUB-RULE ABLATION + WEIGHT-FREEZE RESULTS ===")
    COLLAPSE_THRESH = 0.2
    for r in results:
        s = r["snapshots"]
        def _g(k, f):
            d = s.get(k)
            if d is None or (isinstance(d, dict) and d.get("nonfinite")):
                return "NF" if d else "NA"
            return f"{d[f]:.3f}"
        d500 = s.get(500, {}) if not s.get(500, {}).get("nonfinite") else {}
        l23e_500 = d500.get("l23e", 0.0)
        prevented = "T" if l23e_500 > COLLAPSE_THRESH else "F"
        print(
            f"ablation={r['name']:<20} l23e_s500={_g(500,'l23e')} som_s500={_g(500,'l23som')} "
            f"pv_s500={_g(500,'l23pv')} he_s500={_g(500,'he')} hpv_s500={_g(500,'hpv')} "
            f"prevented={prevented}"
        )

    print("\n=== WEIGHT TRAJECTORY (all_on condition) ===")
    for step in [1, 100, 300, 500]:
        d = results[0]["snapshots"].get(step, {})
        if d and not d.get("nonfinite"):
            print(f"  step={step:<4} W_pv_l23_raw={d.get('W_pv_l23_raw_mean', float('nan')):+.4f} "
                  f"W_pv_l23_eff={d.get('W_pv_l23_eff_mean', float('nan')):.5f}   "
                  f"W_som_l23_raw={d.get('W_som_l23_raw_mean', float('nan')):+.4f} "
                  f"W_som_l23_eff={d.get('W_som_l23_eff_mean', float('nan')):.5f}")

    print(f"\n[total wall={time.monotonic()-t0:.1f}s]")
    import json
    out = Path("logs/task74/level10_vogels_subrule.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
