"""Init-mean shift test: raise W_pv_l23_raw init_mean by 1 unit (-5.0 → -4.0)
and rerun Phase-2 training. Reports divergence step for comparison with
default (step 250). 'earlier'/'later'/'same' verdict.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import torch
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import build_world, run_phase2_training


def run(shift: float, n_steps: int = 260) -> int | None:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")
    # Shift W_pv_l23_raw init: add `shift` in-place, update raw_init_means.
    with torch.no_grad():
        net.l23_e.W_pv_l23_raw.add_(shift)
    net.l23_e.raw_init_means["W_pv_l23_raw"] = \
        net.l23_e.raw_init_means.get("W_pv_l23_raw", -5.0) + shift
    try:
        hist = run_phase2_training(
            net=net, world=world, n_steps=n_steps, batch_size=4,
            seed_offset=42*10_000,
            lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
            weight_decay=1e-5, beta_syn=1e-4,
            log_every=1, warmup_steps=30, segment_length=50, soft_reset_scale=0.1,
        )
    except RuntimeError as e:
        msg = str(e)
        if "non-finite" in msg:
            # Extract "at step N"
            import re
            m = re.search(r"step (\d+)", msg)
            return int(m.group(1)) if m else -1
        raise
    return None  # no divergence within n_steps


if __name__ == "__main__":
    for shift in (0.0, 1.0):
        d = run(shift, 260)
        print(f"shift={shift:+.1f} divergence_step={d}")
