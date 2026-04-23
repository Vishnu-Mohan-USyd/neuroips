"""Task #61b debug harness: instrument apply_plasticity_step to log |dw|_max
per rule-call and |w|_max per weight across steps, so we can localise WHICH
rule's Δw first explodes and whether raw-weight decay (weight_decay) at
default (1e-5) vs ×1000 (1e-1) stabilises divergence.

Not imported by tests. Standalone script.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2 import train_phase2_predictive as T


def run(weight_decay: float, n_steps: int, log_path: Path) -> dict:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = T.build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")

    rules = T.PlasticityRuleBank.from_config(
        cfg=cfg, lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=weight_decay, beta_syn=1e-4,
    )

    state = net.initial_state(batch_size=4)
    # Warmup 30 steps no-plasticity (same as default)
    for w in range(30):
        seeds = [42 * 10_000 + w * 4 + b for b in range(4)]
        frames = T.sample_batch_window(world, seeds, n_steps_per_window=2)
        _s0, _s1, state, _i0, _i1, _x0, _x1 = T._forward_window(net, frames, state)

    # Monkey-patch _apply_update to record per-call |dw|_max and |total|_max.
    # Keep state dict of last call info.
    per_call_log: list[dict] = []
    orig_apply = T._apply_update

    def record_apply(net, module, weight, dw, energy, pre, mask=None):
        T._assert_plastic(net, module, weight)
        w = T._get_weight(net, module, weight)
        shrink = energy.current_weight_shrinkage(w, pre, mask=mask)
        total = dw + shrink
        dw_abs_max = float(dw.abs().max().item())
        shrink_abs_max = float(shrink.abs().max().item())
        total_abs_max = float(total.abs().max().item())
        w_abs_max_pre = float(w.data.abs().max().item())
        per_call_log.append({
            "module": module, "weight": weight,
            "dw_max": dw_abs_max, "shrink_max": shrink_abs_max,
            "total_max": total_abs_max, "w_max_pre": w_abs_max_pre,
        })
        w.data.add_(total)
        return float(total.abs().mean().item())

    T._apply_update = record_apply
    try:
        result = {"divergence_step": None, "first_explosion": None,
                  "history": []}
        for step in range(n_steps):
            seeds = [42 * 10_000 + (30 + step) * 4 + b for b in range(4)]
            frames = T.sample_batch_window(world, seeds, n_steps_per_window=2)
            s0, s1, s2, i0, i1, xh0, _ = T._forward_window(net, frames, state)
            per_call_log.clear()
            try:
                T.apply_plasticity_step(net, rules, s0, s1, s2, i0, i1, xh0)
            except Exception as e:
                result["divergence_step"] = step
                result["exception"] = str(e)
                break
            # Record per-step per-weight max
            step_info = {"step": step,
                         "per_call": list(per_call_log),
                         "r_l23_max": float(s2.r_l23.abs().max().item()),
                         "r_h_max": float(s2.r_h.abs().max().item())}
            result["history"].append(step_info)
            # Check for explosion — dw_max > 1 is already huge given lr=1e-4
            for c in per_call_log:
                if c["dw_max"] > 100.0 and result["first_explosion"] is None:
                    result["first_explosion"] = {"step": step, **c}
            state = s2
            # Segment soft reset every 50
            if (step + 1) % 50 == 0 and (step + 1) < n_steps:
                state = T._soft_reset_state(state, scale=0.1)
            # Check for nonfinite
            if not math.isfinite(s2.r_l23.abs().max().item()):
                result["divergence_step"] = step
                result["nonfinite"] = True
                break
    finally:
        T._apply_update = orig_apply

    with log_path.open("w") as f:
        for h in result["history"]:
            f.write(json.dumps(h) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-steps", type=int, default=260)
    p.add_argument("--log", type=Path, default=Path("/tmp/task61b_per_rule.jsonl"))
    args = p.parse_args()
    r = run(args.weight_decay, args.n_steps, args.log)
    print(f"weight_decay= {args.weight_decay}")
    print(f"divergence_step= {r['divergence_step']}")
    print(f"first_explosion= {r['first_explosion']}")
    # Also summarise max-dw per (module,weight) at the explosion step, if any.
    if r["first_explosion"] is not None:
        step_target = r["first_explosion"]["step"]
        for h in r["history"]:
            if h["step"] == step_target:
                print(f"--- per-call snapshot at explosion step {step_target} ---")
                for c in h["per_call"]:
                    print(f"  {c['module']}.{c['weight']}: dw_max={c['dw_max']:.3e} shrink_max={c['shrink_max']:.3e} total_max={c['total_max']:.3e} w_max_pre={c['w_max_pre']:.3e}")
                break
