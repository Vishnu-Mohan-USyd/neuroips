"""Task #59 — isolate the Vogels iSTDP δ magnitude source.

At step 0 of instrumented Phase-2, mean|Δw| on W_pv_l23_raw = 12.77. With
`η=1e-4`, that would require `(1/B)·mean|post.T @ pre|` ≈ 1.28e5.

This probe decomposes the Δw into (lr·hebb) vs (wd·decay) vs (energy·shrink)
components and reports each term's max / mean magnitudes, plus the actual
pre / post rate statistics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank, build_world, sample_batch_window, _forward_window,
    _raw_prior,
)


def main() -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")
    rules = PlasticityRuleBank.from_config(
        cfg=cfg, lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=1e-5, beta_syn=1e-4,
    )

    state = net.initial_state(batch_size=4)
    # Warmup 30
    for w in range(30):
        seeds = [42 * 10_000 + w * 4 + b for b in range(4)]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)
        _, _, state, _, _, _, _ = _forward_window(net, frames, state)

    # Step 0: produce state0, state1, state2 and dissect the Vogels update
    seeds = [42 * 10_000 + 30 * 4 + b for b in range(4)]
    frames = sample_batch_window(world, seeds, n_steps_per_window=2)
    state0, state1, state2, info0, info1, x_hat_0, _ = _forward_window(
        net, frames, state
    )

    print("=== Rate statistics at the update point ===")
    for lbl, t in [
        ("state1.r_l4", state1.r_l4), ("state1.r_l23", state1.r_l23),
        ("state1.r_pv", state1.r_pv), ("state1.r_som", state1.r_som),
        ("state1.r_h", state1.r_h),
        ("state2.r_l23", state2.r_l23), ("state2.r_pv", state2.r_pv),
        ("state2.r_som", state2.r_som), ("state2.r_h", state2.r_h),
    ]:
        print(f"  {lbl:>18}: shape={tuple(t.shape)} mean={t.mean().item():.4f}"
              f"  max={t.abs().max().item():.4f}  "
              f"pos_frac={(t > 0).float().mean().item():.3f}")

    # --- Vogels applied to W_pv_l23_raw -------------------------------------
    pre = state1.r_pv
    post = state2.r_l23
    w = net.l23_e.W_pv_l23_raw
    target = cfg.plasticity.vogels_target_l23e_hz
    lr = 1e-4
    wd = 1e-5
    beta = 1e-4

    print(f"\n=== Vogels on W_pv_l23_raw ===")
    print(f"  W shape: {tuple(w.shape)}")
    print(f"  W raw:  mean={w.mean().item():.4f} max={w.abs().max().item():.4f}"
          f" softplus={F.softplus(w).mean().item():.4f}")
    print(f"  target_rate_hz ρ = {target}")
    print(f"  lr = {lr}   wd = {wd}   beta_syn = {beta}")

    # Hebb term: (1/B) * post_dev.T @ pre
    post_dev = post - target                              # [B, 256]
    hebb = post_dev.t() @ pre / float(post.shape[0])      # [256, 64]
    print(f"\n  hebb = (1/B) · post_dev.T @ pre:  "
          f"shape={tuple(hebb.shape)} mean={hebb.abs().mean().item():.4e} "
          f"max={hebb.abs().max().item():.4e}")
    print(f"  lr · hebb:  mean={ (lr*hebb).abs().mean().item():.4e} "
          f"max={ (lr*hebb).abs().max().item():.4e}")

    # Decay term
    prior = _raw_prior(net, "l23_e", "W_pv_l23_raw", w)
    decay = wd * (w - prior)
    print(f"\n  raw_prior mean = {prior.mean().item():.4f}")
    print(f"  wd · (w - prior):  mean={decay.abs().mean().item():.4e} "
          f"max={decay.abs().max().item():.4e}")

    # Energy shrinkage term
    pre_sq = (pre * pre).mean(dim=0)        # [n_pre=64]
    shrink = -beta * pre_sq * w             # broadcasts to [256, 64]
    print(f"\n  pre² mean across batch, max={pre_sq.max().item():.4e} "
          f"mean={pre_sq.mean().item():.4e}")
    print(f"  -beta · pre² · w:  mean={shrink.abs().mean().item():.4e} "
          f"max={shrink.abs().max().item():.4e}")

    # dw from Vogels.delta: lr*hebb - wd*(w - prior)
    dw = lr * hebb - decay
    print(f"\n  dw (Vogels) = lr·hebb - wd·(w-prior):  "
          f"mean={dw.abs().mean().item():.4e} "
          f"max={dw.abs().max().item():.4e}")

    # total applied: dw + shrink
    total = dw + shrink
    print(f"\n  total = dw + shrink (what gets added to w):  "
          f"mean={total.abs().mean().item():.4e} "
          f"max={total.abs().max().item():.4e}")

    # Dominant term check
    terms = {
        "lr*hebb": (lr * hebb).abs().mean().item(),
        "wd*decay": decay.abs().mean().item(),
        "shrink": shrink.abs().mean().item(),
    }
    print(f"\n  Dominant-term ranking (by mean|·|):")
    for k, v in sorted(terms.items(), key=lambda kv: -kv[1]):
        print(f"    {k:>10}  {v:.4e}")

    # Ratio of decay to hebb
    print(f"\n  wd*decay / lr*hebb  = {decay.abs().mean().item() / max(1e-30, (lr*hebb).abs().mean().item()):.2e}")
    print(f"  shrink / lr*hebb    = {shrink.abs().mean().item() / max(1e-30, (lr*hebb).abs().mean().item()):.2e}")


if __name__ == "__main__":
    main()
