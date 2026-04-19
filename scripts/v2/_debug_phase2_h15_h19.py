"""Diagnostic harness for Task #37 H15-H19 hypothesis tests.

Post-fix investigation: spectral radius fix (Task #36, layers.py init_mean=-3.5)
dropped max|eig| from 21 to 0.93 but |eps| slope is virtually unchanged
(+6.85e-5 -> +6.93e-5). Need to find the actual driver.

Team-lead hypothesis (key insight): prior h10-h14 harness passed
lr_urbanczik=lr_vogels=lr_hebb=1e-12, but NOT lr_homeostasis. Homeostasis
runs on its own lr (default 1e-5) configured at Population construction;
ThresholdHomeostasis.update() is called unconditionally by the driver
(scripts/v2/train_phase2_predictive.py:374-375) regardless of the plasticity
rule bank. So prior H11 result (theta monotone drift) was NOT actually
isolated from "no plasticity" -- homeostasis was still active.

Experiments:

H15 HOMEO-OFF (critical isolating test):
    Three subruns at identical init seed=42, 1000 steps each:
      A: default LRs (1e-4 plast, 1e-5 homeo) -> reproduce divergence.
      B: default plasticity LRs + homeostasis lr=0 -> if flat, theta driver confirmed.
      C: plasticity LRs=1e-12 + homeostasis lr=0 -> true all-zero baseline.
    Measure polyfit-deg1 slope of |eps| over 1000 steps.

H16 target rate mismatch:
    Fresh net. Single procedural frame from zero state. Record actual
    r_l4, r_l23, r_h mean rates vs target_rate_hz=1.0.

H17 b_pred_raw drift:
    Default-LR run. Log softplus(b_pred_raw).mean() every step.

H18 softplus(W_pred_H_raw) growth:
    Default-LR run. Log softplus(W_pred_H_raw).max() and .mean() every step.

H19 spectral radius verify (post-fix):
    softplus(W_rec_raw) * mask spectral radius for L23E + HE.
    Expected: max|eig| < 1 per Coder's claim.

Bypass mechanism for homeostasis=0:
    ThresholdHomeostasis.__init__ rejects lr<=0, but we can set
    ``net.<pop>.homeostasis.lr = 0.0`` AFTER construction. update() at
    src/v2_model/plasticity.py:285 uses self.lr so this zeros the update.

All experiments use seed=42, batch_size=2 -- same as prior harness.
Tested evidence only; no fix recommendations in report.
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
LR_DEFAULT_PLAST = 1e-4
LR_DEFAULT_HOMEO = 1e-5


def _mk_net_world(seed: int = 42):
    cfg = ModelConfig()
    torch.manual_seed(seed)
    net = V2Network(cfg, token_bank=None, seed=seed)
    net.set_phase("phase2")
    world, _bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    return net, world


def zero_homeostasis(net: V2Network) -> None:
    """Set L23E and HE homeostasis lr to 0 (bypasses constructor validation).

    Effect: ``ThresholdHomeostasis.update()`` becomes ``theta.add_(0)`` -> no-op.
    """
    net.l23_e.homeostasis.lr = 0.0
    net.h_e.homeostasis.lr = 0.0


def spectral_radius_post_fix(net: V2Network) -> dict:
    l23_W = (F.softplus(net.l23_e.W_rec_raw.detach()) * net.l23_e.mask_rec)
    he_W = (F.softplus(net.h_e.W_rec_raw.detach()) * net.h_e.mask_rec)
    l23_eigs = torch.linalg.eigvals(l23_W).abs()
    he_eigs = torch.linalg.eigvals(he_W).abs()
    l23_leak = float(net.l23_e._leak)
    he_leak = float(net.h_e._leak)
    n_l23 = l23_W.shape[0]
    n_he = he_W.shape[0]
    I_l23 = torch.eye(n_l23, dtype=l23_W.dtype)
    I_he = torch.eye(n_he, dtype=he_W.dtype)
    J_l23 = l23_leak * I_l23 + 1.0 * l23_W
    J_he = he_leak * I_he + 1.0 * he_W
    return {
        "l23e_W_rec_max_abs_eig": float(l23_eigs.max().item()),
        "l23e_W_rec_rowsum_mean": float(l23_W.sum(dim=1).mean().item()),
        "he_W_rec_max_abs_eig": float(he_eigs.max().item()),
        "he_W_rec_rowsum_mean": float(he_W.sum(dim=1).mean().item()),
        "l23e_leak_1_minus_dt_over_tau": l23_leak,
        "he_leak_1_minus_dt_over_tau": he_leak,
        "l23e_full_jacobian_phi_prime_1_max_abs_eig": float(
            torch.linalg.eigvals(J_l23).abs().max().item()
        ),
        "he_full_jacobian_phi_prime_1_max_abs_eig": float(
            torch.linalg.eigvals(J_he).abs().max().item()
        ),
        "l23e_homeostasis_lr": float(net.l23_e.homeostasis.lr),
        "l23e_homeostasis_target_rate": float(net.l23_e.homeostasis.target_rate),
        "he_homeostasis_lr": float(net.h_e.homeostasis.lr),
        "he_homeostasis_target_rate": float(net.h_e.homeostasis.target_rate),
    }


def first_frame_rate_probe(net: V2Network, world, seed: int = 0) -> dict:
    """H16: single-frame forward from initial_state, record actual rates."""
    frames = sample_batch_window(world, [seed, seed + 1], n_steps_per_window=2)
    state0 = net.initial_state(batch_size=2)
    _, state1, state2, _, _, x_hat_0, _ = _forward_window(net, frames, state0)
    return {
        "frames_mean": float(frames.mean().item()),
        "frames_max": float(frames.max().item()),
        "r_l4_mean_after_step1": float(state1.r_l4.mean().item()),
        "r_l4_max_after_step1": float(state1.r_l4.max().item()),
        "r_l23_mean_after_step2": float(state2.r_l23.mean().item()),
        "r_l23_max_after_step2": float(state2.r_l23.max().item()),
        "r_h_mean_after_step2": float(state2.r_h.mean().item()),
        "r_h_max_after_step2": float(state2.r_h.max().item()),
        "r_pv_mean_after_step2": float(state2.r_pv.mean().item()),
        "r_som_mean_after_step2": float(state2.r_som.mean().item()),
        "x_hat_0_mean": float(x_hat_0.mean().item()),
        "x_hat_0_max": float(x_hat_0.max().item()),
        "eps_mean": float((state2.r_l4 - x_hat_0).abs().mean().item()),
        "l23e_target_rate_hz": float(net.l23_e.homeostasis.target_rate),
        "he_target_rate_hz": float(net.h_e.homeostasis.target_rate),
    }


def custom_loop(
    net: V2Network,
    world,
    n_steps: int,
    *,
    lr_plast: float = LR_DEFAULT_PLAST,
    lr_homeo_zero: bool = False,
    batch_size: int = 2,
    seed_offset: int = 0,
    log_every: int = 1,
) -> list[dict]:
    """Replicate run_phase2_training's loop body with per-step logging.

    Args:
        lr_plast: shared lr for urbanczik/vogels/hebb rules.
        lr_homeo_zero: if True, set homeostasis.lr = 0 post-construction.
    """
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
        seeds = [seed_offset + step * batch_size + b for b in range(batch_size)]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)

        state0 = net.initial_state(batch_size=batch_size)
        (
            state0, state1, state2, info0, info1, x_hat_0, _x_hat_1,
        ) = _forward_window(net, frames, state0)

        delta_per_w = apply_plasticity_step(
            net, rules, state0, state1, state2, info0, info1, x_hat_0,
        )

        if step % log_every == 0 or step == n_steps - 1:
            eps = state2.r_l4 - x_hat_0
            b_pred_sp = F.softplus(net.prediction_head.b_pred_raw.detach())
            w_ph_sp = F.softplus(net.prediction_head.W_pred_H_raw.detach())
            entry = {
                "step": step,
                "eps_abs_mean": float(eps.abs().mean().item()),
                "r_l4_mean": float(state2.r_l4.mean().item()),
                "r_l23_mean": float(state2.r_l23.mean().item()),
                "r_h_mean": float(state2.r_h.mean().item()),
                "x_hat_mean": float(x_hat_0.mean().item()),
                "x_hat_max": float(x_hat_0.max().item()),
                "theta_l23e_mean": float(net.l23_e.homeostasis.theta.mean().item()),
                "theta_he_mean": float(net.h_e.homeostasis.theta.mean().item()),
                "theta_l23e_max": float(net.l23_e.homeostasis.theta.max().item()),
                "theta_he_min": float(net.h_e.homeostasis.theta.min().item()),
                "b_pred_sp_mean": float(b_pred_sp.mean().item()),
                "b_pred_sp_max": float(b_pred_sp.max().item()),
                "w_pred_H_sp_mean": float(w_ph_sp.mean().item()),
                "w_pred_H_sp_max": float(w_ph_sp.max().item()),
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


def slope_and_stats(hist: list[dict]) -> dict:
    """Compute polyfit-deg1 slope + start/end of eps_abs_mean."""
    steps = np.array([h["step"] for h in hist if "eps_abs_mean" in h], dtype=float)
    eps = np.array([h["eps_abs_mean"] for h in hist if "eps_abs_mean" in h])
    if len(steps) < 2:
        return {"slope": 0.0, "start": 0.0, "end": 0.0, "n": len(steps)}
    slope, intercept = np.polyfit(steps, eps, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "start": float(eps[0]),
        "end": float(eps[-1]),
        "ratio": float(eps[-1] / max(eps[0], 1e-12)),
        "max": float(eps.max()),
        "min": float(eps.min()),
        "n": int(len(steps)),
    }


def summarise(hist: list[dict], label: str) -> str:
    stats = slope_and_stats(hist)
    if stats["n"] == 0:
        return f"[{label}] EMPTY"
    h0 = hist[0]
    hN = hist[-1]
    if "error" in hN:
        return f"[{label}] CRASH at step {hN.get('step')} — {hN.get('error')}"
    theta_l23_trace = [h["theta_l23e_mean"] for h in hist if "theta_l23e_mean" in h]
    theta_he_trace = [h["theta_he_mean"] for h in hist if "theta_he_mean" in h]
    b_pred_trace = [h["b_pred_sp_mean"] for h in hist if "b_pred_sp_mean" in h]
    w_ph_max_trace = [h["w_pred_H_sp_max"] for h in hist if "w_pred_H_sp_max" in h]
    r_l23_trace = [h["r_l23_mean"] for h in hist if "r_l23_mean" in h]
    r_h_trace = [h["r_h_mean"] for h in hist if "r_h_mean" in h]
    return (
        f"[{label}] n={stats['n']}\n"
        f"  eps  slope={stats['slope']:+.4e}  start={stats['start']:.4e}  "
        f"end={stats['end']:.4e}  ratio={stats['ratio']:.3f}x\n"
        f"  theta_L23E: {theta_l23_trace[0]:+.4e} -> {theta_l23_trace[-1]:+.4e} "
        f"(delta {theta_l23_trace[-1]-theta_l23_trace[0]:+.4e})\n"
        f"  theta_HE  : {theta_he_trace[0]:+.4e} -> {theta_he_trace[-1]:+.4e} "
        f"(delta {theta_he_trace[-1]-theta_he_trace[0]:+.4e})\n"
        f"  b_pred_sp_mean: {b_pred_trace[0]:.4e} -> {b_pred_trace[-1]:.4e}\n"
        f"  W_pred_H_sp_max: {w_ph_max_trace[0]:.4e} -> {w_ph_max_trace[-1]:.4e}\n"
        f"  r_l23_mean: {r_l23_trace[0]:.4e} -> {r_l23_trace[-1]:.4e}\n"
        f"  r_h_mean  : {r_h_trace[0]:.4e} -> {r_h_trace[-1]:.4e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment", type=str, required=True,
        choices=["h15", "h16", "h19", "all"],
    )
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    results = {"seed": args.seed, "n_steps": args.n_steps}

    if args.experiment in ("h19", "all"):
        net, _ = _mk_net_world(seed=args.seed)
        r = spectral_radius_post_fix(net)
        results["h19"] = r
        print("=== H19: spectral radius post-fix ===")
        for k, v in r.items():
            print(f"  {k}: {v}")

    if args.experiment in ("h16", "all"):
        net, world = _mk_net_world(seed=args.seed)
        r = first_frame_rate_probe(net, world)
        results["h16"] = r
        print("\n=== H16: single-frame rate probe ===")
        for k, v in r.items():
            print(f"  {k}: {v}")

    if args.experiment in ("h15", "all"):
        # Three subruns with fresh nets (must be fresh — parameters mutate).
        # Runs h17/h18 trajectories naturally fall out of A.
        print(f"\n=== H15 Run A: default LRs (plast=1e-4, homeo=1e-5) ===")
        net_a, world_a = _mk_net_world(seed=args.seed)
        hist_a = custom_loop(
            net_a, world_a, n_steps=args.n_steps,
            lr_plast=LR_DEFAULT_PLAST, lr_homeo_zero=False,
            log_every=args.log_every,
        )
        results["h15_A_default"] = {
            "trace": hist_a, "stats": slope_and_stats(hist_a),
        }
        print(summarise(hist_a, label="H15 A default"))

        print(f"\n=== H15 Run B: default plast LRs, HOMEOSTASIS LR = 0 ===")
        net_b, world_b = _mk_net_world(seed=args.seed)
        hist_b = custom_loop(
            net_b, world_b, n_steps=args.n_steps,
            lr_plast=LR_DEFAULT_PLAST, lr_homeo_zero=True,
            log_every=args.log_every,
        )
        results["h15_B_homeo_off"] = {
            "trace": hist_b, "stats": slope_and_stats(hist_b),
        }
        print(summarise(hist_b, label="H15 B homeo=0"))

        print(f"\n=== H15 Run C: plast LRs=1e-12 + HOMEOSTASIS LR = 0 (all-zero) ===")
        net_c, world_c = _mk_net_world(seed=args.seed)
        hist_c = custom_loop(
            net_c, world_c, n_steps=args.n_steps,
            lr_plast=LR_TINY, lr_homeo_zero=True,
            log_every=args.log_every,
        )
        results["h15_C_all_zero"] = {
            "trace": hist_c, "stats": slope_and_stats(hist_c),
        }
        print(summarise(hist_c, label="H15 C all-zero"))

        print("\n=== H15 comparison table ===")
        print(f"{'variant':<18} {'slope':>14} {'start':>12} {'end':>12} {'ratio':>8}")
        for k in ("h15_A_default", "h15_B_homeo_off", "h15_C_all_zero"):
            s = results[k]["stats"]
            print(f"{k:<18} {s['slope']:>14.4e} {s['start']:>12.4e} "
                  f"{s['end']:>12.4e} {s['ratio']:>8.3f}x")

    if args.out_json is not None:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.out_json}")


if __name__ == "__main__":
    main()
