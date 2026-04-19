"""Task #60 verification (scratch driver).

Phase-2 1000-step rolling training with default hyperparams (seed=42).
Uses the production ``run_phase2_training`` driver with log_every=1 so
the returned history covers every step. Reports:
- |eps| at t = 0, 100, 200, 500, 1000
- max over all steps of delta_mean across plastic weights (as proxy for
  plasticity-delta explosion)
- any_nan across logged history
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import build_world, run_phase2_training


def main() -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")

    metrics_path = Path("/tmp/task60_metrics.jsonl")
    history = run_phase2_training(
        net=net, world=world,
        n_steps=1000, batch_size=4, seed_offset=42 * 10_000,
        log_every=1,
        metrics_path=metrics_path,
        warmup_steps=30, segment_length=50, soft_reset_scale=0.1,
    )

    # history is indexed by step index == wall index.
    def eps_at(t: int) -> float:
        return float(history[min(t, len(history) - 1)].eps_abs_mean)

    for t in (0, 100, 200, 500, 999):  # t=1000 requires index 999
        label = "1000" if t == 999 else str(t)
        print(f"phase2_1000step_eps_at_t{label} = {eps_at(t):.6e}")

    delta_max = max(h.delta_mean for h in history)
    print(f"phase2_1000step_delta_max = {delta_max:.6e}")

    any_nan = any(
        (not math.isfinite(h.loss_pred))
        or (not math.isfinite(h.eps_abs_mean))
        or (not math.isfinite(h.r_l23_mean))
        or (not math.isfinite(h.r_h_mean))
        or (not math.isfinite(h.delta_mean))
        for h in history
    )
    print(f"phase2_1000step_any_nan = {any_nan}")


if __name__ == "__main__":
    main()
