"""Diagnostic harness for Task #34 H10-H14 hypothesis tests.

H10: fixed-input (repeat single frame) at LRs≈0 — does |eps| still diverge?
H11: log theta_L23E and theta_HE trajectories at LRs≈0.
H12: 10000 steps at LRs≈0 — does r_l23 stabilise or keep growing?
H13: spectral radius of softplus(W_rec_raw)*mask at init for L23E + HE.
H14: state-reset-every-step — answered by reading driver code (line 505
     unconditionally sets state0 = net.initial_state per window, so the
     baseline regime IS reset-every-step; baseline diverges therefore
     reset-every-step diverges).

All experiments at LRs = 1e-12 (UrbanczikSennRule/VogelsISTDPRule require
lr > 0, so we use a numerically-negligible value instead of literal zero).
wd = 0, beta_syn = 0 so the update path is purely the plasticity rule.

Usage:
    python scripts/v2/_debug_phase2_h10_h14.py --experiment h13
    python scripts/v2/_debug_phase2_h10_h14.py --experiment h10 --n-steps 1000
    python scripts/v2/_debug_phase2_h10_h14.py --experiment h11 --n-steps 1000
    python scripts/v2/_debug_phase2_h10_h14.py --experiment h12 --n-steps 10000

This harness shells out to the production driver's component functions to
avoid copy-drift — it does NOT reimplement the training loop except for
the optional fixed-frame swap and per-step logging.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Optional

import torch
import torch.nn.functional as F

from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank,
    _forward_window,
    apply_plasticity_step,
    build_world,
    sample_batch_window,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


LR_TINY = 1e-12  # proxy for zero (UrbanczikSennRule rejects lr <= 0)


def _mk_net_world(seed: int = 42):
    cfg = ModelConfig()
    torch.manual_seed(seed)
    net = V2Network(cfg, token_bank=None, seed=seed)
    net.set_phase("phase2")
    world, _bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    return net, world


def spectral_radius_probe(net: V2Network) -> dict:
    """H13: compute spectral radius of softplus(W_rec_raw)*mask at init."""
    l23_W = (F.softplus(net.l23_e.W_rec_raw.detach()) * net.l23_e.mask_rec)
    he_W = (F.softplus(net.h_e.W_rec_raw.detach()) * net.h_e.mask_rec)
    # L23 mask is like-to-like sparse (shape 256,256).
    # HE mask is uniform-random sparse (shape 64,64).
    l23_eigs = torch.linalg.eigvals(l23_W).abs()
    he_eigs = torch.linalg.eigvals(he_W).abs()

    # Full linearised Jacobian at a fixed point where phi'(drive-theta) is
    # approximated by the slope of softplus (~0.5 at zero); the stability
    # criterion is |eig((1-dt/tau)*I + phi'*W_rec)| < 1.
    l23_leak = float(net.l23_e._leak)
    he_leak = float(net.h_e._leak)
    n_l23 = l23_W.shape[0]
    n_he = he_W.shape[0]
    I_l23 = torch.eye(n_l23, dtype=l23_W.dtype)
    I_he = torch.eye(n_he, dtype=he_W.dtype)
    # phi' upper-bound is 1 (rectified softplus slope saturates at 1 for large
    # positive argument). Use phi'=1 for worst-case stability.
    J_l23 = l23_leak * I_l23 + 1.0 * l23_W
    J_he = he_leak * I_he + 1.0 * he_W
    J_l23_eigs = torch.linalg.eigvals(J_l23).abs()
    J_he_eigs = torch.linalg.eigvals(J_he).abs()

    return {
        "l23e_W_rec_max_abs_eig": float(l23_eigs.max().item()),
        "l23e_W_rec_mean_abs_eig": float(l23_eigs.mean().item()),
        "he_W_rec_max_abs_eig": float(he_eigs.max().item()),
        "he_W_rec_mean_abs_eig": float(he_eigs.mean().item()),
        "l23e_leak_1_minus_dt_over_tau": l23_leak,
        "he_leak_1_minus_dt_over_tau": he_leak,
        "l23e_full_jacobian_phi_prime_1_max_abs_eig": float(
            J_l23_eigs.max().item()
        ),
        "he_full_jacobian_phi_prime_1_max_abs_eig": float(
            J_he_eigs.max().item()
        ),
        "l23e_mask_rec_density": float(net.l23_e.mask_rec.mean().item()),
        "he_mask_rec_density": float(net.h_e.mask_rec.mean().item()),
        "l23e_softplus_W_rec_max": float(l23_W.max().item()),
        "l23e_softplus_W_rec_mean_where_mask": float(
            (l23_W.sum() / net.l23_e.mask_rec.sum()).item()
        ),
        "he_softplus_W_rec_max": float(he_W.max().item()),
        "he_softplus_W_rec_mean_where_mask": float(
            (he_W.sum() / net.h_e.mask_rec.sum()).item()
        ),
    }


def custom_loop(
    net: V2Network,
    world,
    n_steps: int,
    *,
    lr: float = LR_TINY,
    fixed_frame: bool = False,
    fixed_seed_base: int = 0,
    batch_size: int = 2,
    seed_offset: int = 0,
    log_every: int = 1,
) -> list[dict]:
    """Replicate run_phase2_training's loop body with per-step logging.

    If fixed_frame=True, draws ONE batch of frames at startup and reuses
    the same tensor for every step (tests H10).
    """
    rules = PlasticityRuleBank.from_config(
        cfg=net.cfg,
        lr_urbanczik=lr,
        lr_vogels=lr,
        lr_hebb=lr,
        weight_decay=0.0,
        beta_syn=0.0,
    )

    fixed_frames_tensor: Optional[torch.Tensor] = None
    if fixed_frame:
        fixed_seeds = [fixed_seed_base + b for b in range(batch_size)]
        fixed_frames_tensor = sample_batch_window(
            world, fixed_seeds, n_steps_per_window=2
        )

    history: list[dict] = []
    for step in range(n_steps):
        if fixed_frame:
            assert fixed_frames_tensor is not None
            frames = fixed_frames_tensor
        else:
            seeds = [seed_offset + step * batch_size + b for b in range(batch_size)]
            frames = sample_batch_window(world, seeds, n_steps_per_window=2)

        # EXACTLY mirrors driver line 505:
        state0 = net.initial_state(batch_size=batch_size)
        (
            state0, state1, state2, info0, info1, x_hat_0, _x_hat_1,
        ) = _forward_window(net, frames, state0)

        delta_per_w = apply_plasticity_step(
            net, rules, state0, state1, state2, info0, info1, x_hat_0,
        )

        if step % log_every == 0 or step == n_steps - 1:
            eps = state2.r_l4 - x_hat_0
            entry = {
                "step": step,
                "eps_abs_mean": float(eps.abs().mean().item()),
                "r_l4_mean": float(state2.r_l4.mean().item()),
                "r_l23_mean": float(state2.r_l23.mean().item()),
                "r_h_mean": float(state2.r_h.mean().item()),
                "r_pv_mean": float(state2.r_pv.mean().item()),
                "r_som_mean": float(state2.r_som.mean().item()),
                "theta_l23e_mean": float(
                    net.l23_e.homeostasis.theta.mean().item()
                ),
                "theta_he_mean": float(
                    net.h_e.homeostasis.theta.mean().item()
                ),
                "theta_l23e_max": float(
                    net.l23_e.homeostasis.theta.max().item()
                ),
                "theta_he_max": float(net.h_e.homeostasis.theta.max().item()),
                "theta_l23e_min": float(
                    net.l23_e.homeostasis.theta.min().item()
                ),
                "theta_he_min": float(net.h_e.homeostasis.theta.min().item()),
                "delta_mean": float(
                    sum(delta_per_w.values()) / max(len(delta_per_w), 1)
                ),
            }
            history.append(entry)

        if not math.isfinite(state2.r_l23.abs().max().item()):
            history.append({
                "step": step,
                "error": f"non-finite r_l23 at step {step}",
            })
            break

    return history


def summarise(hist: list[dict], label: str) -> str:
    """One-line-per-metric summary for console output."""
    if not hist:
        return f"[{label}] no history"
    h0 = hist[0]
    hN = hist[-1]
    if "error" in hN:
        return f"[{label}] CRASH at step {hN.get('step')} — {hN.get('error')}"
    steps_logged = len(hist)
    eps_trace = [h["eps_abs_mean"] for h in hist if "eps_abs_mean" in h]
    r_l23_trace = [h["r_l23_mean"] for h in hist if "r_l23_mean" in h]
    theta_l23_trace = [h["theta_l23e_mean"] for h in hist if "theta_l23e_mean" in h]
    theta_he_trace = [h["theta_he_mean"] for h in hist if "theta_he_mean" in h]
    r_h_trace = [h["r_h_mean"] for h in hist if "r_h_mean" in h]

    # Monotonicity check on theta (adjacent differences)
    def _mono(trace: list[float]) -> str:
        if len(trace) < 2:
            return "N/A"
        diffs = [trace[i+1] - trace[i] for i in range(len(trace) - 1)]
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        return f"{pos} up / {neg} down / {len(diffs)} total"

    lines = [
        f"[{label}] n_logged={steps_logged} (step {h0['step']}-{hN['step']})",
        f"  eps     start={eps_trace[0]:.4e} end={eps_trace[-1]:.4e} "
        f"max={max(eps_trace):.4e} min={min(eps_trace):.4e} ratio={eps_trace[-1]/max(eps_trace[0], 1e-12):.3f}x",
        f"  r_l23   start={r_l23_trace[0]:.4e} end={r_l23_trace[-1]:.4e} "
        f"max={max(r_l23_trace):.4e}",
        f"  r_h     start={r_h_trace[0]:.4e} end={r_h_trace[-1]:.4e} "
        f"max={max(r_h_trace):.4e}",
        f"  theta_L23E  start={theta_l23_trace[0]:.4e} end={theta_l23_trace[-1]:.4e} "
        f"Δ={theta_l23_trace[-1]-theta_l23_trace[0]:+.4e} mono={_mono(theta_l23_trace)}",
        f"  theta_HE    start={theta_he_trace[0]:.4e} end={theta_he_trace[-1]:.4e} "
        f"Δ={theta_he_trace[-1]-theta_he_trace[0]:+.4e} mono={_mono(theta_he_trace)}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment", type=str, required=True,
        choices=["h10", "h11", "h12", "h13", "all"],
    )
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument(
        "--log-sparse-every", type=int, default=100,
        help="For H12 (10k steps), log only every N steps to keep output size sane.",
    )
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    results = {"seed": args.seed}

    if args.experiment in ("h13", "all"):
        net, _ = _mk_net_world(seed=args.seed)
        results["h13"] = spectral_radius_probe(net)
        print("=== H13: spectral radius at init ===")
        for k, v in results["h13"].items():
            print(f"  {k}: {v}")

    if args.experiment in ("h10", "all"):
        print(f"\n=== H10: fixed-frame {args.n_steps} steps at LRs={LR_TINY} ===")
        net, world = _mk_net_world(seed=args.seed)
        hist = custom_loop(
            net, world, n_steps=args.n_steps,
            lr=LR_TINY, fixed_frame=True,
            log_every=args.log_every,
        )
        results["h10"] = {
            "n_steps": args.n_steps,
            "trace": hist,
        }
        print(summarise(hist, label="H10 fixed-frame"))

    if args.experiment in ("h11", "all"):
        print(f"\n=== H11: normal frames {args.n_steps} steps at LRs={LR_TINY}, log theta every step ===")
        net, world = _mk_net_world(seed=args.seed)
        hist = custom_loop(
            net, world, n_steps=args.n_steps,
            lr=LR_TINY, fixed_frame=False,
            log_every=args.log_every,
        )
        results["h11"] = {
            "n_steps": args.n_steps,
            "trace": hist,
        }
        print(summarise(hist, label="H11 normal-frames"))

    if args.experiment in ("h12", "all"):
        print(
            f"\n=== H12: normal frames {args.n_steps} steps at LRs={LR_TINY} "
            f"(long-run, log every {args.log_sparse_every}) ==="
        )
        net, world = _mk_net_world(seed=args.seed)
        hist = custom_loop(
            net, world, n_steps=args.n_steps,
            lr=LR_TINY, fixed_frame=False,
            log_every=args.log_sparse_every,
        )
        results["h12"] = {
            "n_steps": args.n_steps,
            "trace": hist,
        }
        print(summarise(hist, label="H12 long-run"))

    if args.out_json is not None:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.out_json}")


if __name__ == "__main__":
    main()
