"""Task#74 — diff FROZEN forward-only vs apply_plasticity_step(lr=1e-20).

Both runs: homeostasis disabled, procedural driver, 500 steps, same seed.
The apply_plasticity_step path collapses (l23e=0.073) while forward-only
is stable (l23e=0.337). This probe identifies WHICH state variables
diverge — the mutation that apply_plasticity_step performs beyond pure
weight learning (since lrs are effectively zero).
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


def _snapshot(net: V2Network, state) -> dict:
    """Capture all mutable state that could differ between runs."""
    with torch.no_grad():
        return {
            # Network rates
            "r_l23": float(state.r_l23.mean().item()),
            "r_som": float(state.r_som.mean().item()),
            "r_pv": float(state.r_pv.mean().item()),
            "r_h": float(state.r_h.mean().item()),
            "h_pv": float(state.h_pv.mean().item()),
            "m": float(state.m.mean().item()),
            # Thresholds (θ homeostasis)
            "theta_l23e_mean": float(net.l23_e.theta.mean().item()),
            "theta_l23e_abs_max": float(net.l23_e.theta.abs().max().item()),
            "theta_he_mean": float(net.h_e.theta.mean().item()),
            "theta_he_abs_max": float(net.h_e.theta.abs().max().item()),
            # Weight norms (pre-softplus raw)
            "W_rec_l23_norm": float(net.l23_e.W_rec_raw.norm().item()),
            "W_l4_l23_norm": float(net.l23_e.W_l4_l23_raw.norm().item()),
            "W_fb_apical_norm": float(net.l23_e.W_fb_apical_raw.norm().item()),
            "W_pv_l23_norm": float(net.l23_e.W_pv_l23_raw.norm().item()),
            "W_som_l23_norm": float(net.l23_e.W_som_l23_raw.norm().item()),
            "W_pred_H_norm": float(net.prediction_head.W_pred_H_raw.norm().item()),
            "W_pred_C_norm": float(net.prediction_head.W_pred_C_raw.norm().item()),
            "W_pred_apical_norm": float(net.prediction_head.W_pred_apical_raw.norm().item()),
            "b_pred_mean": float(net.prediction_head.b_pred_raw.mean().item()),
            "b_pred_abs_max": float(net.prediction_head.b_pred_raw.abs().max().item()),
            "W_hm_gen_norm": float(net.context_memory.W_hm_gen.norm().item()),
            "W_mm_gen_norm": float(net.context_memory.W_mm_gen.norm().item()),
            "W_mh_gen_norm": float(net.context_memory.W_mh_gen.norm().item()),
            "W_rec_h_norm": float(net.h_e.W_rec_raw.norm().item()),
            "W_l23_h_norm": float(net.h_e.W_l23_h_raw.norm().item()),
            "W_pv_h_norm": float(net.h_e.W_pv_h_raw.norm().item()),
            "W_pv_pre_l23_norm": float(net.l23_pv.W_pre_raw.norm().item()),
            "W_pv_pre_h_norm": float(net.h_pv.W_pre_raw.norm().item()),
        }


def run(name: str, *, use_plasticity: bool, n_steps: int = 500, seed: int = 42) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")
    # Homeostasis monkey-patched off in BOTH conditions (the difference is only
    # whether we call apply_plasticity_step).
    net.l23_e.homeostasis.update = lambda *a, **k: None
    net.h_e.homeostasis.update = lambda *a, **k: None

    rules = None
    if use_plasticity:
        EPS = 1e-20
        rules = PlasticityRuleBank.from_config(
            cfg=cfg, lr_urbanczik=EPS, lr_vogels=EPS, lr_hebb=EPS,
            weight_decay=1e-5, beta_syn=1e-4,
        )

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

    checkpoints = [0, 100, 300, 500]
    snapshots: dict[int, dict] = {0: _snapshot(net, state)}

    for step in range(1, n_steps + 1):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2)
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        if use_plasticity:
            _ = apply_plasticity_step(net, rules, s0, s1, s2, i0, i1, x_hat_0)
        state = s2
        if segment_length > 0 and step % segment_length == 0 and step < n_steps:
            state = _soft_reset_state(state, scale=soft_reset_scale)
            world_states = _reset_all()
        if step in checkpoints:
            snapshots[step] = _snapshot(net, state)
        if not math.isfinite(float(s2.r_l23.abs().max().item())):
            snapshots[step] = {"nonfinite": True}
            break

    return {"name": name, "snapshots": snapshots}


def main():
    t0 = time.monotonic()
    r_forward = run("forward_only", use_plasticity=False)
    print(f"[forward_only] wall={time.monotonic()-t0:.1f}s", file=sys.stderr)
    t1 = time.monotonic()
    r_plast = run("plast_lr_eps", use_plasticity=True)
    print(f"[plast_lr_eps] wall={time.monotonic()-t1:.1f}s", file=sys.stderr)

    # Compute divergence per-variable at each checkpoint
    print("\n=== DIVERGENCE TABLE (plast - forward) / max(|forward|, 1e-12) ===")
    keys = sorted(r_forward["snapshots"][0].keys())
    print(f"{'var':<25} {'init':>10} {'fw_s500':>10} {'pl_s500':>10} {'abs_diff':>10} {'rel_diff':>10}")
    divergent = []
    for k in keys:
        init_v = r_forward["snapshots"][0].get(k)
        fw = r_forward["snapshots"].get(500, {}).get(k)
        pl = r_plast["snapshots"].get(500, {}).get(k)
        if fw is None or pl is None:
            continue
        abs_d = pl - fw
        rel_d = abs_d / max(abs(fw), 1e-12)
        flag = ""
        if abs(rel_d) > 0.01:
            flag = "  <<<"
            divergent.append((k, rel_d, fw, pl))
        print(f"{k:<25} {init_v:>10.4f} {fw:>10.4f} {pl:>10.4f} {abs_d:>+10.4f} {rel_d:>+10.3%}{flag}")

    # Trajectory for the most-divergent variables
    print("\n=== TRAJECTORY OF DIVERGENT VARS (|rel_diff|>1%) ===")
    for k, rel_d, fw500, pl500 in sorted(divergent, key=lambda x: -abs(x[1]))[:10]:
        print(f"\n[{k}]  fw_s500={fw500:.4f}  pl_s500={pl500:.4f}  rel_diff={rel_d:+.2%}")
        for step in [0, 100, 300, 500]:
            fw = r_forward["snapshots"].get(step, {}).get(k)
            pl = r_plast["snapshots"].get(step, {}).get(k)
            if fw is not None and pl is not None:
                print(f"  step={step:<4} fw={fw:>+12.5f}  pl={pl:>+12.5f}  Δ={pl-fw:>+12.5f}")

    print(f"\n[total wall={time.monotonic()-t0:.1f}s]")

    import json
    out = Path("logs/task74/level10_diff_probe.json")
    out.write_text(json.dumps({"forward_only": r_forward, "plast_lr_eps": r_plast}, indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
