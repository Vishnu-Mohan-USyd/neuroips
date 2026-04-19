"""Diagnostic harness for Task #33 debugging — not a test.

Runs Phase-2 training at a chosen lr_urbanczik and logs per-step state
(|eps|, rates, homeostatic thresholds, raw-weight stats) so hypotheses can
be isolated. Shells out to `run_phase2_training` to avoid copy-drift from
the production driver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.v2.train_phase2_predictive import build_world, run_phase2_training
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


_PROBE_WEIGHTS = [
    ("prediction_head", "W_pred_H_raw"),
    ("prediction_head", "W_pred_C_raw"),
    ("prediction_head", "W_pred_apical_raw"),
    ("prediction_head", "b_pred_raw"),
    ("h_e", "W_l23_h_raw"),
    ("h_e", "W_rec_raw"),
    ("l23_e", "W_pv_l23_raw"),
    ("l23_e", "W_som_l23_raw"),
]


def _snapshot(net: V2Network) -> dict[str, dict[str, float]]:
    snap: dict[str, dict[str, float]] = {}
    # Raw weights.
    for mod_name, w_name in _PROBE_WEIGHTS:
        mod = getattr(net, mod_name, None)
        if mod is None:
            continue
        w = getattr(mod, w_name, None)
        if w is None:
            continue
        t = w.detach()
        snap[f"{mod_name}.{w_name}"] = {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "abs_mean": float(t.abs().mean().item()),
        }
    # Homeostatic thresholds.
    for mod_name in ("l23_e", "h_e"):
        mod = getattr(net, mod_name, None)
        if mod is None or not hasattr(mod, "homeostasis"):
            continue
        homeo = mod.homeostasis
        theta = getattr(homeo, "theta", None)
        if theta is None:
            continue
        t = theta.detach()
        snap[f"{mod_name}.homeostasis.theta"] = {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
        }
    return snap


def run_with_probes(
    lr_urbanczik: float,
    n_steps: int,
    probe_every: int,
    beta_syn: float = 1e-4,
    lr_vogels: float = 5e-4,
    lr_hebb: float = 5e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
) -> dict:
    cfg = ModelConfig()
    torch.manual_seed(seed)
    net = V2Network(cfg, token_bank=None, seed=seed)
    world, _bank = build_world(cfg, seed_family="train", token_bank_seed=0)

    # Probe schedule: snapshot before, then every `probe_every` steps.
    out = {"config": {
        "lr_urbanczik": lr_urbanczik,
        "lr_vogels": lr_vogels,
        "lr_hebb": lr_hebb,
        "weight_decay": weight_decay,
        "beta_syn": beta_syn,
        "n_steps": n_steps,
        "seed": seed,
    }, "snapshots": [], "eps_trace": []}

    out["snapshots"].append({"step": 0, "state": _snapshot(net)})

    # Walk the run by chunks of `probe_every`.
    cum_history = []
    step = 0
    while step < n_steps:
        chunk = min(probe_every, n_steps - step)
        hist = run_phase2_training(
            net=net, world=world,
            n_steps=chunk, batch_size=2,
            seed_offset=step * 2,
            lr_urbanczik=lr_urbanczik,
            lr_vogels=lr_vogels,
            lr_hebb=lr_hebb,
            weight_decay=weight_decay,
            beta_syn=beta_syn,
            log_every=1,
        )
        cum_history.extend(hist)
        step += chunk
        out["snapshots"].append({"step": step, "state": _snapshot(net)})

    eps = np.asarray([m.eps_abs_mean for m in cum_history], dtype=np.float64)
    r_l23 = np.asarray([m.r_l23_mean for m in cum_history], dtype=np.float64)
    r_h = np.asarray([m.r_h_mean for m in cum_history], dtype=np.float64)
    out["eps_trace"] = eps.tolist()
    out["r_l23_trace"] = r_l23.tolist()
    out["r_h_trace"] = r_h.tolist()
    # Also compute linreg slope for report.
    x = np.arange(len(eps), dtype=np.float64)
    slope, _ = np.polyfit(x, eps, deg=1)
    out["eps_slope"] = float(slope)
    out["eps_start"] = float(eps[0])
    out["eps_end"] = float(eps[-1])
    out["eps_max"] = float(eps.max())
    out["eps_min"] = float(eps.min())
    out["r_l23_start"] = float(r_l23[0])
    out["r_l23_end"] = float(r_l23[-1])
    out["r_l23_max"] = float(r_l23.max())
    out["r_h_start"] = float(r_h[0])
    out["r_h_end"] = float(r_h[-1])
    out["r_h_max"] = float(r_h.max())
    return out


def _fmt(o: dict) -> str:
    # Compact summary for stdout.
    lines = []
    lines.append(f"cfg {json.dumps(o['config'])}")
    lines.append(
        f"eps: start={o['eps_start']:.4e} end={o['eps_end']:.4e} "
        f"max={o['eps_max']:.4e} min={o['eps_min']:.4e} slope={o['eps_slope']:.4e}"
    )
    lines.append(
        f"r_l23: start={o['r_l23_start']:.4e} end={o['r_l23_end']:.4e} max={o['r_l23_max']:.4e}"
    )
    lines.append(
        f"r_h:   start={o['r_h_start']:.4e} end={o['r_h_end']:.4e} max={o['r_h_max']:.4e}"
    )
    for snap in o["snapshots"]:
        lines.append(f"  step={snap['step']}")
        for k, v in snap["state"].items():
            if "theta" in k:
                lines.append(
                    f"    {k}: mean={v['mean']:.4e} min={v['min']:.4e} max={v['max']:.4e}"
                )
            else:
                lines.append(
                    f"    {k}: abs_mean={v['abs_mean']:.4e} min={v['min']:.4e} max={v['max']:.4e}"
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-urbanczik", type=float, default=1e-3)
    ap.add_argument("--lr-vogels", type=float, default=5e-4)
    ap.add_argument("--lr-hebb", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--beta-syn", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = run_with_probes(
        lr_urbanczik=args.lr_urbanczik,
        n_steps=args.n_steps,
        probe_every=args.probe_every,
        beta_syn=args.beta_syn,
        lr_vogels=args.lr_vogels,
        lr_hebb=args.lr_hebb,
        weight_decay=args.weight_decay,
    )

    print(_fmt(out))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
