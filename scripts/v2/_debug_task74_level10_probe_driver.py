"""Task#74 — probe FROZEN network: gratings vs procedural, with/without soft-reset."""
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
    _clone_world, _forward_window, _soft_reset_state,
    build_world, step_persistent_batch,
)
from scripts.v2._gates_common import make_grating_frame
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


def run(
    name: str, *, driver: str, soft_reset: bool, n_steps: int = 500,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")
    # FROZEN: disable homeostasis too (we never call apply_plasticity_step)
    net.l23_e.homeostasis.update = lambda *a, **k: None
    net.h_e.homeostasis.update = lambda *a, **k: None

    batch_size = 4
    warmup_steps = 30
    segment_length = 50
    soft_reset_scale = 0.1

    if driver == "procedural":
        worlds = [_clone_world(world) for _ in range(batch_size)]
        reset_counter = [0]
        def _reset_all():
            res = [worlds[b].reset(seed * 10_000 + reset_counter[0] * 10_000 + b)
                   for b in range(batch_size)]
            reset_counter[0] += 1
            return res
        world_states = _reset_all()

        def _get_frames():
            return step_persistent_batch(
                worlds, world_states, n_steps_per_window=2)
    elif driver == "gratings":
        # Single sustained grating, oriented at 45°, full contrast.
        grating = make_grating_frame(45.0, 1.0, cfg, batch_size=batch_size)
        # [B,1,H,W] → replicate to [B,2,1,H,W]
        two_frame = torch.stack([grating, grating], dim=1)
        def _get_frames():
            return two_frame, None
    else:
        raise ValueError(driver)

    state = net.initial_state(batch_size=batch_size)

    # warmup (no plasticity, no soft-reset)
    for _ in range(warmup_steps):
        if driver == "procedural":
            frames, world_states_new = _get_frames()
            world_states[:] = world_states_new  # note: world_states is a list
        else:
            frames, _ = _get_frames()
        _s0, _s1, state, _i0, _i1, _x0, _x1 = _forward_window(net, frames, state)

    checkpoint_steps = [1, 100, 300, 500]
    snapshots: dict[int, dict] = {}

    for step in range(1, n_steps + 1):
        if driver == "procedural":
            frames, world_states_new = _get_frames()
            world_states[:] = world_states_new
        else:
            frames, _ = _get_frames()
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        state = s2

        if soft_reset and segment_length > 0 and step % segment_length == 0 and step < n_steps:
            state = _soft_reset_state(state, scale=soft_reset_scale)
            if driver == "procedural":
                world_states[:] = [
                    worlds[b].reset(seed * 10_000 + (step // segment_length) * 10_000 + b)
                    for b in range(batch_size)
                ]

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
    conditions = [
        ("FROZEN_on_procedural",        dict(driver="procedural", soft_reset=True)),
        ("FROZEN_on_gratings",          dict(driver="gratings",   soft_reset=True)),
        ("FROZEN_no_softreset_proc",    dict(driver="procedural", soft_reset=False)),
        ("FROZEN_no_softreset_grat",    dict(driver="gratings",   soft_reset=False)),
    ]
    results = []
    for name, kwargs in conditions:
        tc = time.monotonic()
        r = run(name, **kwargs)
        print(f"[{name}] wall={time.monotonic()-tc:.1f}s", file=sys.stderr)
        results.append(r)

    print("\n=== PROBE RESULTS ===")
    for r in results:
        s = r["snapshots"]
        def _g(k, f):
            d = s.get(k)
            if d is None or d.get("nonfinite"):
                return "NF" if d else "NA"
            return f"{d[f]:.3f}"
        print(
            f"ablation={r['name']} "
            f"l23e_s1={_g(1,'l23e')} l23e_s100={_g(100,'l23e')} "
            f"l23e_s300={_g(300,'l23e')} l23e_s500={_g(500,'l23e')} "
            f"som_s500={_g(500,'l23som')} pv_s500={_g(500,'l23pv')} "
            f"hpv_s500={_g(500,'hpv')}"
        )

    print("\n=== FULL ===")
    for r in results:
        print(f"[{r['name']}]")
        for k in sorted(r["snapshots"]):
            print(f"  step={k}: {r['snapshots'][k]}")

    print(f"\n[total wall={time.monotonic()-t0:.1f}s]")

    import json
    out = Path("logs/task74/level10_probe_driver.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
