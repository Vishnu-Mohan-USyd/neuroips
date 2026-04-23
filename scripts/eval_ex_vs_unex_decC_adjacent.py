#!/usr/bin/env python3
"""Task #19 — adjacent-channel suppression analysis on NEW eval (ex vs unex).

Extends Task #12/#13 by saving per-trial 36-ch r_l23 readout vectors, then
re-centering each trial so the probe-true channel sits at ch18, AND signing
the channel offset by march direction (positive offsets = "ahead in march
direction"). Aggregates across all 2400 trials per branch and reports the
full signed-offset tuning curve, an adjacent-channel suppression table, and
a march-symmetry diagnostic.

Sample: identical to Task #12/#13 (seed_base=42, n_values={4..15}, n_trials=200).
Forward machinery and trial generation are imported verbatim from
eval_ex_vs_unex_decC.py — no re-filtering, no design changes.

Outputs:
  - results/eval_ex_vs_unex_decC_adjacent.json
  - docs/figures/eval_ex_vs_unex_decC_tuning_curves.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

from src.config import load_config
from src.model.network import LaminarV1V2Network

# Re-use the original eval's batch builder and constants
from eval_ex_vs_unex_decC import (
    build_trial_batch,
    READOUT_WIN,
    N_VALUES_DEFAULT,
    N_TRIALS_DEFAULT,
    SEED_BASE_DEFAULT,
    CKPT_PATH_DEFAULT,
    DECODER_C_PATH_DEFAULT,
    CONFIG_PATH_DEFAULT,
)


OUT_JSON_DEFAULT = "results/eval_ex_vs_unex_decC_adjacent.json"
OUT_FIG_DEFAULT = "docs/figures/eval_ex_vs_unex_decC_tuning_curves.png"


# -----------------------------------------------------------------------------
# Core: signed-offset re-centering
# -----------------------------------------------------------------------------

def signed_offset_curves(r_probe: np.ndarray,
                         true_ch: np.ndarray,
                         sign: np.ndarray,
                         n_ori: int) -> tuple[np.ndarray, np.ndarray]:
    """Re-center each trial's tuning curve and sign offsets by march direction.

    For each trial i, returns a 36-channel vector indexed by SIGNED offset
    relative to the probe-true channel:
        signed[i, j] = r_probe[i, (true_ch[i] + sign[i] * offsets[j]) % n_ori]
    where offsets = [-n_ori/2, -n_ori/2 + 1, ..., n_ori/2 - 1].

    For CW trials (sign=+1):  offset +k is k channels CW of probe (= +k*5° in march dir).
    For CCW trials (sign=-1): offset +k is k channels CCW of probe (= +k*5° in march dir).

    So positive offsets are ALWAYS "ahead in march direction" and negative offsets
    are ALWAYS "behind / toward march history".

    Args:
        r_probe: [n_trials, n_ori] — readout-window mean L2/3.
        true_ch: [n_trials] long — true probe orientation channel.
        sign:    [n_trials] in {+1, -1}.
        n_ori:   number of orientation channels (= 36).

    Returns:
        signed:  [n_trials, n_ori] — signed-offset curves.
        offsets: [n_ori] — signed offsets in channels (= [-n_ori/2, ..., n_ori/2 - 1]).
    """
    n = r_probe.shape[0]
    half = n_ori // 2
    offsets = np.arange(-half, half, dtype=np.int64)         # [-18, -17, ..., +17]
    sign_int = sign.astype(np.int64).reshape(n, 1)
    true_ch = true_ch.astype(np.int64).reshape(n, 1)
    ch_idx = (true_ch + sign_int * offsets.reshape(1, -1)) % n_ori
    signed = np.take_along_axis(r_probe, ch_idx, axis=1)
    return signed, offsets


# -----------------------------------------------------------------------------
# Forward pass — extract per-trial 36-ch r_l23 only
# -----------------------------------------------------------------------------

def run_one_N_adjacent(N: int, n_trials: int, seed_base: int,
                       net: LaminarV1V2Network,
                       model_cfg, train_cfg,
                       device: torch.device) -> dict[str, Any]:
    """Forward pass for one march length N. Returns per-trial 36-ch r_probe vectors
    + per-trial true_ch and sign for both branches.
    """
    bd = build_trial_batch(N, n_trials, seed_base, model_cfg, train_cfg, device)
    win_lo = bd["probe_onset"] + READOUT_WIN[0]
    win_hi = bd["probe_onset"] + READOUT_WIN[1]
    assert READOUT_WIN[1] - READOUT_WIN[0] == 2

    packed_ex = net.pack_inputs(bd["stim_ex"], bd["cue"], bd["ts"])
    r_l23_ex, _, _ = net.forward(packed_ex)
    r_probe_ex = r_l23_ex[:, win_lo:win_hi, :].mean(dim=1)        # [B, n_ori]

    packed_unex = net.pack_inputs(bd["stim_unex"], bd["cue"], bd["ts"])
    r_l23_unex, _, _ = net.forward(packed_unex)
    r_probe_unex = r_l23_unex[:, win_lo:win_hi, :].mean(dim=1)

    return dict(
        N=int(N),
        n_trials=int(n_trials),
        r_probe_ex=r_probe_ex.detach().cpu().numpy().astype(np.float64),
        r_probe_unex=r_probe_unex.detach().cpu().numpy().astype(np.float64),
        ex_ch=bd["ex_ch"].detach().cpu().numpy().astype(np.int64),
        unex_ch=bd["unex_ch"].detach().cpu().numpy().astype(np.int64),
        sign=bd["sign"].detach().cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------

def per_offset_mean_sem(signed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean + SEM across trials for each signed offset. Both shape [n_ori]."""
    n = signed.shape[0]
    m = signed.mean(axis=0)
    s = signed.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
    return m, s


def paired_asymmetry(signed: np.ndarray, offsets: np.ndarray, k: int
                     ) -> tuple[float, float]:
    """Per-trial paired asymmetry: signed[+k] - signed[-k]; mean, paired SEM."""
    n = signed.shape[0]
    j_pos = int(np.where(offsets == k)[0][0])
    j_neg = int(np.where(offsets == -k)[0][0])
    diff = signed[:, j_pos] - signed[:, j_neg]
    m = float(diff.mean())
    s = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, s


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=CKPT_PATH_DEFAULT)
    p.add_argument("--decoder-c", default=DECODER_C_PATH_DEFAULT)
    p.add_argument("--config", default=CONFIG_PATH_DEFAULT)
    p.add_argument("--output-json", default=OUT_JSON_DEFAULT)
    p.add_argument("--output-fig", default=OUT_FIG_DEFAULT)
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    p.add_argument("--n-values", type=int, nargs="+", default=N_VALUES_DEFAULT)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] N values={args.n_values}", flush=True)
    print(f"[setup] n_trials/N={args.n_trials}  seed_base={args.seed_base}", flush=True)

    # Load config + model
    model_cfg, train_cfg, _ = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    print(f"[cfg] n_orientations={n_ori}  period={period}  step_deg={step_deg}", flush=True)

    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    print("[setup] network loaded", flush=True)

    # ----------------------------- Forward sweep -----------------------------
    pool_signed_ex, pool_signed_unex = [], []
    per_N_summary = []
    with torch.no_grad():
        for N_val in args.n_values:
            print(f"[run] N={N_val}", flush=True)
            res = run_one_N_adjacent(N_val, args.n_trials, args.seed_base,
                                     net, model_cfg, train_cfg, device)
            signed_ex, offsets = signed_offset_curves(
                res["r_probe_ex"], res["ex_ch"], res["sign"], n_ori)
            signed_unex, _ = signed_offset_curves(
                res["r_probe_unex"], res["unex_ch"], res["sign"], n_ori)
            pool_signed_ex.append(signed_ex)
            pool_signed_unex.append(signed_unex)
            per_N_summary.append({
                "N": N_val,
                "n_trials": args.n_trials,
                "n_cw": int((res["sign"] > 0).sum()),
                "n_ccw": int((res["sign"] < 0).sum()),
            })
            print(f"  n_cw={int((res['sign'] > 0).sum())}  "
                  f"n_ccw={int((res['sign'] < 0).sum())}  "
                  f"signed_ex.shape={signed_ex.shape}", flush=True)

    pool_signed_ex = np.concatenate(pool_signed_ex, axis=0)         # [2400, n_ori]
    pool_signed_unex = np.concatenate(pool_signed_unex, axis=0)
    n_total = pool_signed_ex.shape[0]
    print(f"\n[pool] n_total per branch = {n_total}", flush=True)

    # ----------------------------- Per-offset stats -----------------------------
    mean_ex, sem_ex = per_offset_mean_sem(pool_signed_ex)
    mean_unex, sem_unex = per_offset_mean_sem(pool_signed_unex)
    delta = mean_ex - mean_unex                                     # [n_ori]

    # SEM of paired delta per offset (paired across trials → use pooled SE; here
    # the sample is the same trial-set per offset, so paired SEM at offset j is
    # std(signed_ex[:,j] - signed_unex[:,j]) / sqrt(n)
    paired_diff = pool_signed_ex - pool_signed_unex
    delta_sem = paired_diff.std(axis=0, ddof=1) / np.sqrt(n_total)

    # Sanity check: peak-at-probe (offset 0) should equal the original eval's
    # pooled.peak_at_stim_*_mean (within float precision).
    half = n_ori // 2
    peak_ex_check = float(mean_ex[half])
    peak_unex_check = float(mean_unex[half])
    print(f"[sanity] peak_at_stim_ex(@offset 0)   = {peak_ex_check:.6f}", flush=True)
    print(f"[sanity] peak_at_stim_unex(@offset 0) = {peak_unex_check:.6f}", flush=True)
    print(f"[sanity] (originals: 0.773378 ex / 0.626342 unex)", flush=True)

    # ----------------------------- Adjacent table -----------------------------
    adj_ks = list(range(-5, 6))
    adj_table = []
    for k in adj_ks:
        j = int(np.where(offsets == k)[0][0])
        adj_table.append({
            "offset_ch": k,
            "offset_deg": k * step_deg,
            "ex_mean": float(mean_ex[j]),
            "ex_sem": float(sem_ex[j]),
            "unex_mean": float(mean_unex[j]),
            "unex_sem": float(sem_unex[j]),
            "delta_ex_minus_unex": float(delta[j]),
            "delta_sem_paired": float(delta_sem[j]),
        })
    print("\n[adjacent table] (k = channel offset, signed by march direction)", flush=True)
    print(f"  {'k':>3}  {'r_ex (mean ± SEM)':>22}  {'r_unex (mean ± SEM)':>22}  "
          f"{'Δ (ex − unex)':>15}", flush=True)
    for row in adj_table:
        print(f"  {row['offset_ch']:>+3}  "
              f"{row['ex_mean']:>10.5f} ± {row['ex_sem']:>7.5f}  "
              f"{row['unex_mean']:>10.5f} ± {row['unex_sem']:>7.5f}  "
              f"{row['delta_ex_minus_unex']:>+10.5f} ± {row['delta_sem_paired']:.5f}",
              flush=True)

    # --------------------- Symmetry diagnostic (k = 1, 2, 3) ---------------------
    # Δ_ex(+k) − Δ_ex(−k): paired difference within the ex curve at signed offsets.
    # Same for unex.
    sym_ex, sym_unex = [], []
    for k in (1, 2, 3):
        m_ex, s_ex = paired_asymmetry(pool_signed_ex, offsets, k)
        m_un, s_un = paired_asymmetry(pool_signed_unex, offsets, k)
        sym_ex.append({"k": k, "mean": m_ex, "sem_paired": s_ex})
        sym_unex.append({"k": k, "mean": m_un, "sem_paired": s_un})
    print("\n[symmetry] ex curve: Δ_ex(+k) − Δ_ex(−k)  "
          "(positive = bias toward 'ahead in march direction')", flush=True)
    for row in sym_ex:
        print(f"  k=+{row['k']}:  {row['mean']:>+10.5f} ± {row['sem_paired']:.5f}",
              flush=True)
    print("[symmetry] unex curve: Δ_unex(+k) − Δ_unex(−k)", flush=True)
    for row in sym_unex:
        print(f"  k=+{row['k']}:  {row['mean']:>+10.5f} ± {row['sem_paired']:.5f}",
              flush=True)

    # ----------------------------- Full curve (offset −17..+17) -----------------------------
    full_curve = []
    for k in range(-(half - 1), half):                     # -17..+17 inclusive
        j = int(np.where(offsets == k)[0][0])
        full_curve.append({
            "offset_ch": k,
            "offset_deg": k * step_deg,
            "ex_mean": float(mean_ex[j]),
            "ex_sem": float(sem_ex[j]),
            "unex_mean": float(mean_unex[j]),
            "unex_sem": float(sem_unex[j]),
            "delta_ex_minus_unex": float(delta[j]),
            "delta_sem_paired": float(delta_sem[j]),
        })

    # ----------------------------- Save JSON -----------------------------
    out = {
        "label": "Task #19: adjacent-channel suppression on R1+R2 NEW eval, march-signed offsets",
        "checkpoint": args.checkpoint,
        "decoder_c": args.decoder_c,
        "config": args.config,
        "design": {
            "n_values": list(args.n_values),
            "n_trials_per_N": args.n_trials,
            "seed_base": args.seed_base,
            "n_total_per_branch": n_total,
            "readout_window_steps": list(READOUT_WIN),
            "n_orientations": n_ori,
            "period_deg": period,
            "step_deg": step_deg,
            "centering": "per-trial: roll so probe true_ch lands at ch18; "
                         "then sign-flip offsets by march direction (CCW → flip).",
            "march_sign_convention": "+1 CW (ch_idx ↑), -1 CCW (ch_idx ↓); "
                                     "after sign-flip, +offset = ahead in march direction",
        },
        "per_N_counts": per_N_summary,
        "sanity_check_peak_at_offset_0": {
            "ex_mean": peak_ex_check,
            "unex_mean": peak_unex_check,
            "original_ex_mean": 0.7733782676359018,
            "original_unex_mean": 0.6263418252579868,
        },
        "full_signed_offset_curve": full_curve,
        "adjacent_table": adj_table,
        "symmetry_diagnostic_ex": sym_ex,
        "symmetry_diagnostic_unex": sym_unex,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[json] wrote {args.output_json}", flush=True)

    # ----------------------------- Plot -----------------------------
    plot_offsets = np.array([row["offset_ch"] for row in full_curve])
    plot_ex_m = np.array([row["ex_mean"] for row in full_curve])
    plot_ex_s = np.array([row["ex_sem"] for row in full_curve])
    plot_unex_m = np.array([row["unex_mean"] for row in full_curve])
    plot_unex_s = np.array([row["unex_sem"] for row in full_curve])

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.0))
    ax.plot(plot_offsets, plot_ex_m, "-o", color="C0", lw=1.6, ms=4,
            label=f"Expected   (n={n_total})")
    ax.fill_between(plot_offsets, plot_ex_m - plot_ex_s, plot_ex_m + plot_ex_s,
                    color="C0", alpha=0.20)
    ax.plot(plot_offsets, plot_unex_m, "-s", color="C1", lw=1.6, ms=4,
            label=f"Unexpected (n={n_total})")
    ax.fill_between(plot_offsets, plot_unex_m - plot_unex_s, plot_unex_m + plot_unex_s,
                    color="C1", alpha=0.20)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", label="probe (offset 0)")
    ax.set_xlabel("Channel offset from probe (signed by march direction; + = ahead)")
    ax.set_ylabel("Mean L2/3 readout (re-centered + march-signed)")
    ax.set_title("Re-centered tuning curves: ex vs unex on R1+R2 (Decoder C eval sample)\n"
                 f"Δ peak (offset 0) = {peak_ex_check - peak_unex_check:+.4f}",
                 fontsize=11)
    ax.set_xticks(np.arange(-17, 18, 2))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(args.output_fig))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {args.output_fig}", flush=True)


if __name__ == "__main__":
    main()
