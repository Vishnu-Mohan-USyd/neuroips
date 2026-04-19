"""Task #56 verification (scratch driver).

1. Phase-2 200-step smoke with rolling state — report eps_start / eps_end.
2. m.norm() trajectory over 200 steps.
3. Full-state Jacobian on post-training net at blank input.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import (
    build_world, run_phase2_training, sample_batch_window, _forward_window,
)
from scripts.v2._calibrate_task52 import _compute_jacobian_radius


def main() -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")

    metrics_path = Path("/tmp/task56_metrics.jsonl")
    history = run_phase2_training(
        net=net, world=world,
        n_steps=200, batch_size=4, seed_offset=42 * 10_000,
        log_every=1,
        metrics_path=metrics_path,
        warmup_steps=30,
        segment_length=50,
        soft_reset_scale=0.1,
    )
    eps_start = history[0].eps_abs_mean
    eps_end = history[-1].eps_abs_mean
    print(f"phase2_200step_eps_start = {eps_start:.4e}")
    print(f"phase2_200step_eps_end   = {eps_end:.4e}")

    # --- m.norm trajectory — mirror training-loop state policy ------------
    torch.manual_seed(42)
    cfg2 = ModelConfig(seed=42, device="cpu")
    world2, bank2 = build_world(cfg2, seed_family="train", held_out_regime=None)
    net2 = V2Network(cfg2, token_bank=bank2, seed=42)
    net2.load_state_dict(net.state_dict())
    net2.set_phase("phase2")

    state = net2.initial_state(batch_size=4)
    with torch.no_grad():
        # Warmup 30.
        for w in range(30):
            seeds = [42 * 10_000 + w * 4 + b for b in range(4)]
            frames = sample_batch_window(world2, seeds, n_steps_per_window=2)
            _, _, state, _, _, _, _ = _forward_window(net2, frames, state)

        m_norms = []
        for step in range(200):
            seeds = [42 * 10_000 + (30 + step) * 4 + b for b in range(4)]
            frames = sample_batch_window(world2, seeds, n_steps_per_window=2)
            _, _, state, _, _, _, _ = _forward_window(net2, frames, state)
            m_norms.append(float(state.m.norm().item()))
            if (step + 1) % 50 == 0 and (step + 1) < 200:
                state = state._replace(
                    r_l4=state.r_l4 * 0.1, r_l23=state.r_l23 * 0.1,
                    r_pv=state.r_pv * 0.1, r_som=state.r_som * 0.1,
                    r_h=state.r_h * 0.1, h_pv=state.h_pv * 0.1,
                    m=state.m * 0.1,
                )
    print(f"m_norm_at_t50  = {m_norms[49]:.4e}")
    print(f"m_norm_at_t100 = {m_norms[99]:.4e}")
    print(f"m_norm_at_t150 = {m_norms[149]:.4e}")
    print(f"m_norm_at_t200 = {m_norms[199]:.4e}")

    # --- Full-state Jacobian on POST-TRAINING net at blank input ----------
    torch.manual_seed(42)
    cfg3 = ModelConfig(seed=42, device="cpu")
    net3 = V2Network(cfg3, token_bank=None, seed=42, device="cpu")
    sd = {k: v for k, v in net.state_dict().items() if not k.startswith("token_bank.")}
    net3.load_state_dict(sd, strict=False)
    net3.eval()
    a = cfg3.arch
    blank_x = torch.zeros(1, 1, a.grid_h, a.grid_w, dtype=torch.float32)
    state = net3.initial_state(batch_size=1, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(500):
            _, state, _ = net3(blank_x, state)
    rho_full, rho_nm = _compute_jacobian_radius(net3, state, cfg3, blank_x)
    print(f"jacobian_lmax_after_training_blank = {rho_full:.4f}")
    print(f"jacobian_lmax_no_memory            = {rho_nm:.4f}")


if __name__ == "__main__":
    main()
