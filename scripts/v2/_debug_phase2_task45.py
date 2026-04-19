"""Diagnostic harness for Task #45 — Phase 2 regression post Task #44.

Context:
    Task #44 scaled excitatory init_means down (row-sums dropped ~100x).
    Phase 2 1000-step |eps| regressed from slope -7.5e-6 -> +2.2e-4 (8x growth).
    Task #38 lowered target_rates (L23E=0.5, HE=0.1), Task #43 widened theta
    clamp (±10). Task #44 did NOT touch inhibitory weights
    (W_pv_l23/W_som_l23/W_pv_h still init_mean=0, softplus≈0.69/weight).

Experiments (team-lead brief):
    E1 LR sweep + HOMEO-OFF isolation: same seed=42, fresh net each run,
       1000 steps, log_every=50:
         A: plast=1e-4, homeo=1e-5 (default, reproduces +2.2e-4 slope)
         B: plast=1e-5, homeo=1e-5
         C: plast=1e-6, homeo=1e-5
         D: plast=1e-7, homeo=1e-5
         E: plast=1e-12, homeo=1e-5  (LR=0 proxy, homeo ON)
         F: plast=1e-4, homeo=0      (HOMEO-OFF)
         G: plast=1e-12, homeo=0     (all-zero baseline)

    E2 Weight trajectory logging (done inside E1 via per-step record of
       softplus(W_pv_l23_raw).max(), softplus(W_pred_H_raw).max()).

    E3 Rate/theta trajectory logging (done inside E1 via per-step record of
       theta_L23E mean, theta_HE mean, r_l23, r_h, r_pv, r_som).

Evidence-only report. No fix recommendations per Debugger protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Optional

import numpy as np
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


LR_TINY = 1e-12


def _mk_net_world(seed: int = 42):
    cfg = ModelConfig()
    torch.manual_seed(seed)
    net = V2Network(cfg, token_bank=None, seed=seed)
    net.set_phase("phase2")
    world, _bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    return net, world


def zero_homeostasis(net: V2Network) -> None:
    """Set L23E and HE homeostasis lr to 0 (bypasses constructor validation)."""
    net.l23_e.homeostasis.lr = 0.0
    net.h_e.homeostasis.lr = 0.0


def init_state_probe(net: V2Network, world, seed: int = 0) -> dict:
    """Single-frame forward from init state to capture rate-vs-target landscape."""
    frames = sample_batch_window(world, [seed, seed + 1], n_steps_per_window=2)
    state0 = net.initial_state(batch_size=2)
    _, state1, state2, _, _, x_hat_0, _ = _forward_window(net, frames, state0)
    l23e_target = float(net.l23_e.homeostasis.target_rate)
    he_target = float(net.h_e.homeostasis.target_rate)
    return {
        "frames_mean": float(frames.mean().item()),
        "r_l4_mean": float(state2.r_l4.mean().item()),
        "r_l23_mean": float(state2.r_l23.mean().item()),
        "r_l23_target": l23e_target,
        "r_l23_minus_target": float(state2.r_l23.mean().item()) - l23e_target,
        "r_h_mean": float(state2.r_h.mean().item()),
        "r_h_target": he_target,
        "r_h_minus_target": float(state2.r_h.mean().item()) - he_target,
        "r_pv_mean": float(state2.r_pv.mean().item()),
        "r_som_mean": float(state2.r_som.mean().item()),
        "x_hat_mean": float(x_hat_0.mean().item()),
        "eps_mean": float((state2.r_l4 - x_hat_0).abs().mean().item()),
        "W_l4_l23_rowsum_mean": float(
            F.softplus(net.l23_e.W_l4_l23_raw.detach()).sum(dim=1).mean().item()
        ),
        "W_pv_l23_max_abs": float(
            F.softplus(net.l23_e.W_pv_l23_raw.detach()).max().item()
        ),
        "W_som_l23_max_abs": float(
            F.softplus(net.l23_e.W_som_l23_raw.detach()).max().item()
        ),
        "W_pv_h_max_abs": float(
            F.softplus(net.h_e.W_pv_h_raw.detach()).max().item()
        ),
        "W_rec_l23_max_abs_eig": float(
            torch.linalg.eigvals(
                F.softplus(net.l23_e.W_rec_raw.detach()) * net.l23_e.mask_rec
            ).abs().max().item()
        ),
        "W_rec_he_max_abs_eig": float(
            torch.linalg.eigvals(
                F.softplus(net.h_e.W_rec_raw.detach()) * net.h_e.mask_rec
            ).abs().max().item()
        ),
    }


def custom_loop(
    net: V2Network,
    world,
    n_steps: int,
    *,
    lr_plast: float,
    lr_homeo_zero: bool = False,
    batch_size: int = 2,
    log_every: int = 50,
) -> list[dict]:
    if lr_homeo_zero:
        zero_homeostasis(net)

    rules = PlasticityRuleBank.from_config(
        cfg=net.cfg,
        lr_urbanczik=lr_plast,
        lr_vogels=lr_plast,
        lr_hebb=lr_plast,
        weight_decay=0.0,
        beta_syn=0.0,
    )

    history: list[dict] = []
    for step in range(n_steps):
        seeds = [step * batch_size + b for b in range(batch_size)]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)
        state0 = net.initial_state(batch_size=batch_size)
        state0, state1, state2, info0, info1, x_hat_0, _ = _forward_window(
            net, frames, state0
        )
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
                "r_l23_max": float(state2.r_l23.max().item()),
                "r_h_mean": float(state2.r_h.mean().item()),
                "r_h_max": float(state2.r_h.max().item()),
                "r_pv_mean": float(state2.r_pv.mean().item()),
                "r_som_mean": float(state2.r_som.mean().item()),
                "x_hat_mean": float(x_hat_0.mean().item()),
                "x_hat_max": float(x_hat_0.max().item()),
                "theta_l23e_mean": float(net.l23_e.homeostasis.theta.mean().item()),
                "theta_l23e_min": float(net.l23_e.homeostasis.theta.min().item()),
                "theta_he_mean": float(net.h_e.homeostasis.theta.mean().item()),
                "theta_he_min": float(net.h_e.homeostasis.theta.min().item()),
                "w_pv_l23_max": float(
                    F.softplus(net.l23_e.W_pv_l23_raw.detach()).max().item()
                ),
                "w_pv_l23_mean": float(
                    F.softplus(net.l23_e.W_pv_l23_raw.detach()).mean().item()
                ),
                "w_som_l23_max": float(
                    F.softplus(net.l23_e.W_som_l23_raw.detach()).max().item()
                ),
                "w_pred_H_max": float(
                    F.softplus(net.prediction_head.W_pred_H_raw.detach()).max().item()
                ),
                "w_pred_H_mean": float(
                    F.softplus(net.prediction_head.W_pred_H_raw.detach()).mean().item()
                ),
                "b_pred_sp_mean": float(
                    F.softplus(net.prediction_head.b_pred_raw.detach()).mean().item()
                ),
                "delta_mean": float(
                    sum(delta_per_w.values()) / max(len(delta_per_w), 1)
                ),
            }
            history.append(entry)

        if not math.isfinite(state2.r_l23.abs().max().item()):
            history.append({"step": step, "error": f"non-finite r_l23 at step {step}"})
            break

    return history


def slope_stats(hist: list[dict]) -> dict:
    """polyfit-deg1 slope + start/end of eps_abs_mean."""
    eps = np.array([h["eps_abs_mean"] for h in hist if "eps_abs_mean" in h])
    steps = np.array([h["step"] for h in hist if "eps_abs_mean" in h], dtype=float)
    if len(eps) < 2:
        return {"slope": 0.0, "start": 0.0, "end": 0.0, "n": int(len(eps))}
    slope, intercept = np.polyfit(steps, eps, 1)
    return {
        "slope": float(slope),
        "start": float(eps[0]),
        "end": float(eps[-1]),
        "ratio": float(eps[-1] / max(eps[0], 1e-12)),
        "n": int(len(eps)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument(
        "--variants", type=str, default="A,B,C,D,E,F,G",
        help="Comma-list of variants to run.",
    )
    args = ap.parse_args()

    results = {
        "seed": args.seed,
        "n_steps": args.n_steps,
    }

    # === Init-state probe ===
    net, world = _mk_net_world(seed=args.seed)
    init_probe = init_state_probe(net, world)
    results["init_probe"] = init_probe
    print("=== INIT-STATE PROBE (fresh net, single frame from zero state) ===")
    for k, v in init_probe.items():
        print(f"  {k}: {v}")

    # === Variant specs ===
    variants = {
        "A": dict(lr_plast=1e-4, lr_homeo_zero=False,
                  label="A: plast=1e-4, homeo=1e-5 (DEFAULT)"),
        "B": dict(lr_plast=1e-5, lr_homeo_zero=False,
                  label="B: plast=1e-5, homeo=1e-5"),
        "C": dict(lr_plast=1e-6, lr_homeo_zero=False,
                  label="C: plast=1e-6, homeo=1e-5"),
        "D": dict(lr_plast=1e-7, lr_homeo_zero=False,
                  label="D: plast=1e-7, homeo=1e-5"),
        "E": dict(lr_plast=LR_TINY, lr_homeo_zero=False,
                  label="E: plast=1e-12, homeo=1e-5 (LR=0 proxy)"),
        "F": dict(lr_plast=1e-4, lr_homeo_zero=True,
                  label="F: plast=1e-4, homeo=0 (HOMEO-OFF)"),
        "G": dict(lr_plast=LR_TINY, lr_homeo_zero=True,
                  label="G: plast=1e-12, homeo=0 (ALL-ZERO)"),
    }

    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]

    for key in wanted:
        spec = variants[key]
        print(f"\n=== Variant {spec['label']} ===")
        net_i, world_i = _mk_net_world(seed=args.seed)
        hist = custom_loop(
            net_i, world_i, n_steps=args.n_steps,
            lr_plast=spec["lr_plast"], lr_homeo_zero=spec["lr_homeo_zero"],
            log_every=args.log_every,
        )
        stats = slope_stats(hist)
        results[f"variant_{key}"] = {"trace": hist, "stats": stats, "label": spec["label"]}
        h0 = hist[0]
        hN = hist[-1]
        print(f"  eps  slope={stats['slope']:+.4e}  start={stats['start']:.4e} "
              f"end={stats['end']:.4e}  ratio={stats['ratio']:.3f}x  n={stats['n']}")
        print(f"  theta_L23E_mean: {h0['theta_l23e_mean']:+.4e} -> {hN['theta_l23e_mean']:+.4e}")
        print(f"  theta_HE_mean  : {h0['theta_he_mean']:+.4e} -> {hN['theta_he_mean']:+.4e}")
        print(f"  theta_L23E_min : {h0['theta_l23e_min']:+.4e} -> {hN['theta_l23e_min']:+.4e}")
        print(f"  theta_HE_min   : {h0['theta_he_min']:+.4e} -> {hN['theta_he_min']:+.4e}")
        print(f"  r_l23          : {h0['r_l23_mean']:.4e} -> {hN['r_l23_mean']:.4e}")
        print(f"  r_h            : {h0['r_h_mean']:.4e} -> {hN['r_h_mean']:.4e}")
        print(f"  x_hat_mean     : {h0['x_hat_mean']:.4e} -> {hN['x_hat_mean']:.4e}")
        print(f"  w_pv_l23_max   : {h0['w_pv_l23_max']:.4e} -> {hN['w_pv_l23_max']:.4e}")
        print(f"  w_som_l23_max  : {h0['w_som_l23_max']:.4e} -> {hN['w_som_l23_max']:.4e}")
        print(f"  w_pred_H_max   : {h0['w_pred_H_max']:.4e} -> {hN['w_pred_H_max']:.4e}")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'variant':<8} {'lr_plast':>10} {'homeo':>6} "
          f"{'slope':>14} {'start':>12} {'end':>12} {'ratio':>8}")
    for key in wanted:
        spec = variants[key]
        s = results[f"variant_{key}"]["stats"]
        homeo = "0" if spec["lr_homeo_zero"] else "1e-5"
        print(f"{key:<8} {spec['lr_plast']:>10.0e} {homeo:>6} "
              f"{s['slope']:>14.4e} {s['start']:>12.4e} {s['end']:>12.4e} "
              f"{s['ratio']:>8.3f}x")

    if args.out_json is not None:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.out_json}")


if __name__ == "__main__":
    main()
