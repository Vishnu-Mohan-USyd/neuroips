"""Phase-2 Gate 7 — context-memory load-bearing check (plan v4 / Task #39).

Compares next-step prediction quality with vs without the context-memory
output bias ``b_l23``. The ablation zeroes both readout weights —
``W_mh_gen`` and ``W_mh_task`` — so the memory integration still runs
but its projection to L2/3 apical is silenced.

Pass criterion (plan v4):
    (MSE_ablated − MSE_with_C) / MSE_with_C ≥ 0.05

If the degradation is below that floor, ``C`` is dead weight: the
procedural-world hidden regime is too easy for L4+L2/3+H alone and
Phase-2 must be iterated before Phase-3 can meaningfully train the
task readout.

Usage:
    python -m scripts.v2.run_c_load_bearing_check \\
        --checkpoint checkpoints/v2/phase2/phase2_s42/step_1000.pt \\
        --seed 42

Writes ``gate_7_c_load_bearing.json`` next to the checkpoint. Exit 0 on
pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from scripts.v2._gates_common import CheckpointBundle, load_checkpoint
from src.v2_model.world.procedural import ProceduralWorld


__all__ = [
    "ablate_context_memory",
    "measure_prediction_mse",
    "run_gate_7_c_load_bearing",
]


# ---------------------------------------------------------------------------
# Ablation context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def ablate_context_memory(bundle: CheckpointBundle) -> Iterator[None]:
    """Temporarily zero ``W_mh_gen`` + ``W_mh_task`` on the context memory.

    Restores the originals on exit. This keeps memory integration (the
    ``m_t`` recurrence) intact but silences the readout to L2/3 apical —
    i.e. ``b_l23 ≡ 0`` for the duration of the context.
    """
    cm = bundle.net.context_memory
    saved_gen = cm.W_mh_gen.data.detach().clone()
    saved_task = cm.W_mh_task.data.detach().clone()
    try:
        cm.W_mh_gen.data.zero_()
        cm.W_mh_task.data.zero_()
        yield
    finally:
        cm.W_mh_gen.data.copy_(saved_gen)
        cm.W_mh_task.data.copy_(saved_task)


# ---------------------------------------------------------------------------
# Prediction MSE on eval trajectories
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure_prediction_mse(
    bundle: CheckpointBundle,
    *, n_trajectories: int = 8, n_steps_per_traj: int = 20,
    seed_family: str = "eval",
) -> float:
    """Mean next-step prediction MSE on procedural-world trajectories.

    Uses the same evaluation loop structure as gate 4 in
    :mod:`scripts.v2.eval_gates`: for each trajectory we step the
    network, caching ``x_hat`` at step ``t`` and comparing it to the
    observed ``r_l4`` at step ``t+1``. The first step is discarded
    because there's no prior prediction to compare against.
    """
    world = ProceduralWorld(bundle.cfg, bundle.bank, seed_family=seed_family)
    errs: list[float] = []
    for traj_id in range(int(n_trajectories)):
        frames_seq, _states = world.trajectory(
            trajectory_seed=traj_id, n_steps=int(n_steps_per_traj),
        )
        frames_seq = frames_seq.unsqueeze(1)                     # [T, 1, 1, H, W]
        state = bundle.net.initial_state(batch_size=1)
        prev_x_hat: Tensor | None = None
        for t in range(frames_seq.shape[0]):
            x_hat, state, info = bundle.net(frames_seq[t], state)
            if prev_x_hat is not None:
                errs.append(float(F.mse_loss(prev_x_hat, info["r_l4"]).item()))
            prev_x_hat = x_hat
    return float(np.mean(errs)) if errs else float("nan")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gate_7_c_load_bearing(
    bundle: CheckpointBundle,
    *, n_trajectories: int = 8, n_steps_per_traj: int = 20,
    degradation_floor: float = 0.05,
    phase: str = "phase2",
    seed_family: str = "eval",
) -> dict[str, Any]:
    """Run the Gate-7 ablation + with/without comparison.

    ``phase`` is the phase to run under; we default to Phase-2 (i.e.
    keep the Phase-2 book-keeping) since the ablation check is about
    the learned structure, not the Phase-3 task weights. The context
    manager restores ``W_mh_*`` so the call is idempotent.
    """
    bundle.net.set_phase(phase)
    with_c = measure_prediction_mse(
        bundle, n_trajectories=n_trajectories,
        n_steps_per_traj=n_steps_per_traj, seed_family=seed_family,
    )
    with ablate_context_memory(bundle):
        without_c = measure_prediction_mse(
            bundle, n_trajectories=n_trajectories,
            n_steps_per_traj=n_steps_per_traj, seed_family=seed_family,
        )
    if with_c > 1e-12:
        degradation = (without_c - with_c) / with_c
    else:
        degradation = float("nan")
    return {
        "gate": "7_c_load_bearing",
        "n_trajectories": int(n_trajectories),
        "n_steps_per_traj": int(n_steps_per_traj),
        "mse_with_c": with_c,
        "mse_without_c": without_c,
        "relative_degradation": float(degradation),
        "degradation_floor": float(degradation_floor),
        "passed": bool(
            np.isfinite(degradation) and degradation >= degradation_floor
        ),
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-2 Gate 7 C-load-bearing")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trajectories", type=int, default=8)
    p.add_argument("--n-steps-per-traj", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device=args.device)
    results = run_gate_7_c_load_bearing(
        bundle,
        n_trajectories=args.n_trajectories,
        n_steps_per_traj=args.n_steps_per_traj,
    )
    out_path = args.output or (args.checkpoint.parent / "gate_7_c_load_bearing.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
