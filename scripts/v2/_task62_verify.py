"""Task #62 verification (scratch driver).

Phase-2 1000-step rolling training with default hyperparams (seed=42),
with plasticity-delta and raw-weight clamps enabled. Reports:
- |eps| at t = 0, 100, 200, 500, 1000
- max over the run of delta_mean (proxy for plasticity-delta size)
- max over the run of |raw w|_max over all plastic Phase-2 weights
- any_nan across logged history
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import build_world, run_phase2_training


PLASTIC_WEIGHTS = (
    ("l23_e", "W_l4_l23_raw"),
    ("l23_e", "W_rec_raw"),
    ("l23_e", "W_fb_apical_raw"),
    ("l23_pv", "W_e_pv_raw"),
    ("l23_pv", "W_pv_e_raw"),
    ("l23_som", "W_e_som_raw"),
    ("l23_som", "W_som_e_raw"),
    ("pred_head", "W_pred_raw"),
    ("prediction_head", "b_pred_raw"),
)


def _w_max(net: V2Network) -> float:
    m = 0.0
    for mod_name, w_name in PLASTIC_WEIGHTS:
        mod = getattr(net, mod_name, None)
        if mod is None:
            continue
        w = getattr(mod, w_name, None)
        if w is None:
            continue
        m = max(m, float(w.detach().abs().max().item()))
    return m


def main() -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")

    # Snapshot initial w_max (should be well within [-8, 8]).
    w_max_init = _w_max(net)

    metrics_path = Path("/tmp/task62_metrics.jsonl")
    history = run_phase2_training(
        net=net, world=world,
        n_steps=1000, batch_size=4, seed_offset=42 * 10_000,
        log_every=1,
        metrics_path=metrics_path,
        warmup_steps=30, segment_length=50, soft_reset_scale=0.1,
    )

    w_max_final = _w_max(net)

    def eps_at(t: int) -> float:
        return float(history[min(t, len(history) - 1)].eps_abs_mean)

    for t in (0, 100, 200, 500, 999):
        label = "1000" if t == 999 else str(t)
        print(f"phase2_1000step_eps_at_t{label} = {eps_at(t):.6e}")

    delta_max = max(h.delta_mean for h in history)
    print(f"phase2_1000step_delta_max = {delta_max:.6e}")
    print(f"phase2_1000step_w_max_init = {w_max_init:.6e}")
    print(f"phase2_1000step_w_max_final = {w_max_final:.6e}")

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
