"""Decisive ablation: run with beta_syn=0 (no energy shrinkage). If divergence
disappears, the energy-shrinkage Euler overshoot is confirmed as the root cause.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))

import math
import torch
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import build_world, run_phase2_training


def run(beta: float, n_steps: int = 260) -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")
    hist = run_phase2_training(
        net=net, world=world, n_steps=n_steps, batch_size=4,
        seed_offset=42*10_000,
        lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=beta,
        log_every=1, warmup_steps=30, segment_length=50, soft_reset_scale=0.1,
    )
    max_delta = max(h.delta_mean for h in hist)
    any_nan = any(
        (not math.isfinite(h.delta_mean)) or (not math.isfinite(h.loss_pred))
        for h in hist)
    print(f"beta={beta} steps={len(hist)} max_delta_mean={max_delta:.3e} any_nan={any_nan}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--n-steps", type=int, default=260)
    args = p.parse_args()
    run(args.beta, args.n_steps)
