"""One-shot parallel check: is r_l23 zero already at Phase-2 init, or is it
Fix-C SOM overkill that killed it during training?

Procedure (single-core, runs in parallel with the ongoing retrain):
  1. Load Phase-2 step_3000 checkpoint into a phase3_kok V2Network.
  2. Zero W_mh_task_exc and W_mh_task_inh (no learned task route).
  3. Run 5 Kok probe trials (seed 42, cue→orient matched mapping).
  4. Report mean(r_l23) and mean(r_som) during probe1, plus a rough
     "what's r_som doing during the current in-flight training?" anchor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


def _run_probe(bundle, cue_id: int, probe_deg: float,
               timing: KokTiming, seed: int) -> dict[str, float]:
    cfg = bundle.cfg
    device = cfg.device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(float(probe_deg), 1.0, cfg, device=device)
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    n_total = timing.total

    r_l23_probe1: list[torch.Tensor] = []
    r_som_probe1: list[torch.Tensor] = []
    with torch.no_grad():
        for t in range(n_total):
            if t < cue_end:
                frame, q_t = blank, q_cue
            elif t < delay_end:
                frame, q_t = blank, None
            elif t < probe1_end:
                frame, q_t = probe, None
            else:
                frame, q_t = blank, None
            _x, state, info = bundle.net(frame, state, q_t=q_t)
            if delay_end <= t < probe1_end:
                r_l23_probe1.append(info["r_l23"][0].detach())
                r_som_probe1.append(info["r_som"][0].detach())

    l23_stack = torch.stack(r_l23_probe1, dim=0)                 # [probe1_steps, n_l23]
    som_stack = torch.stack(r_som_probe1, dim=0)                 # [probe1_steps, n_som]
    return {
        "r_l23_mean": float(l23_stack.mean()),
        "r_l23_max": float(l23_stack.max()),
        "r_som_mean": float(som_stack.mean()),
        "r_som_max": float(som_stack.max()),
    }


def main() -> int:
    ckpt_path = Path(
        "checkpoints/v2/phase2/phase2_task70_s42/phase2_s42/step_3000.pt"
    )
    if not ckpt_path.exists():
        print(f"ERROR: ckpt not found at {ckpt_path}", file=sys.stderr)
        return 1

    # NB: Phase-2 checkpoint loads as phase='phase2'. We need phase='phase3_kok'
    # so the task-weight routes are live. Flip phase after load.
    bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    bundle.net.eval()

    cue_map = cue_mapping_from_seed(42)
    timing = KokTiming()

    # --- Condition 1: task weights FORCED to zero (isolates Phase-2 baseline) ---
    with torch.no_grad():
        bundle.net.context_memory.W_mh_task_exc.zero_()
        bundle.net.context_memory.W_mh_task_inh.zero_()
        bundle.net.context_memory.W_qm_task.zero_()

    print("\n=== Cond 1: W_mh_task_{exc,inh,qm} = 0 (Phase-2 baseline) ===")
    r_l23_vals_z: list[float] = []
    r_som_vals_z: list[float] = []
    for i in range(5):
        cue_id = i % 2
        probe_deg = cue_map[cue_id]               # matched probe
        out = _run_probe(bundle, cue_id, probe_deg, timing, seed=42 + i)
        r_l23_vals_z.append(out["r_l23_mean"])
        r_som_vals_z.append(out["r_som_mean"])
        print(
            f"trial{i}  cue={cue_id} probe={probe_deg:>5.1f}° "
            f"r_l23_mean={out['r_l23_mean']:.4e} r_l23_max={out['r_l23_max']:.4e} "
            f"r_som_mean={out['r_som_mean']:.4e} r_som_max={out['r_som_max']:.4e}"
        )
    mean_l23_z = sum(r_l23_vals_z) / len(r_l23_vals_z)
    mean_som_z = sum(r_som_vals_z) / len(r_som_vals_z)
    print(f"→ cond1 mean r_l23={mean_l23_z:.4e}  mean r_som={mean_som_z:.4e}")

    # ---- Summary line ----
    if mean_l23_z > 0.05:
        interp = "SOM_overkill"
    elif mean_l23_z > 1e-6:
        interp = "SOM_partial_suppression"
    else:
        interp = "core_dead_at_phase2"
    print(
        f"\nSUMMARY: r_l23_with_task_weights=1.7e-20 "
        f"r_l23_task_zero={mean_l23_z:.4e} "
        f"r_som_task_zero={mean_som_z:.4e} "
        f"interpretation={interp}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
