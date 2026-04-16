#!/usr/bin/env python3
"""Matched-quality Expected vs Unexpected analysis on R1+R2 (Task #30).

Hypothesis under test
---------------------
The re-centered "Expected < Unexpected at stim channel" result (0.58 vs 0.66
on R1+R2) might be a statistical artifact of trial heterogeneity — Expected
trials may be enriched with low-confidence/low-precision presentations that
drag the mean down, while Unexpected trials happen to land in higher-quality
regimes. If the gap is real, it should PERSIST when both buckets are
restricted to (a) tight prediction errors and (b) the same top-quartile
pi_pred_eff distribution.

Method
------
Per presentation (skipping the first of each sequence and ambiguous trials),
record:
  - pred_err            : circular distance between argmax(q_pred) at the LAST
                          ISI step and the true orientation, in degrees.
  - pi_pred_eff         : V2 effective precision (= pi_pred_raw * feedback_scale)
                          at the LAST ISI step BEFORE stimulus onset
                          (t_isi_last = pres_i * steps_per - 1).
  - r_l23_window_avg    : mean of r_l23 over the [9, 11] readout window of this
                          presentation (matches the trained decoder's window).
  - r_l23_rolled        : window-averaged r_l23 rolled so true_theta lands at
                          channel N//2 (=18 for N=36).
  - decoder_top1        : argmax of orientation_decoder(r_l23_window_avg).
  - true_theta_idx      : ground truth channel index.
  - task_state          : 0 = focused, 1 = routine.

Buckets
-------
  unfiltered Expected   : pred_err <= 10°  (legacy criterion)
  unfiltered Unexpected : pred_err >  20°  (legacy criterion)
  matched   Expected    : pred_err <= 2.5° AND pi_pred_eff >= POOLED 75th pct
  matched   Unexpected  : pred_err >  20°  AND pi_pred_eff >= POOLED 75th pct

If matched_Exp n < 200, widen Exp pred_err tolerance to <= 5°. If still < 200,
report underpowered and stop.

Reported per bucket
-------------------
  n, mean pi_pred_eff, mean pred_err
  decoding accuracy (mean + 95% bootstrap CI, 1000 resamples)
  activity @ stim channel = mean over trials of r_l23_rolled[:, 18]
  activity total          = mean over trials of r_l23_rolled.sum(-1)
  peak-channel of MEAN r_l23_rolled (sanity)

Plus per-task-state (focused / routine) breakdown.

Sanity check
------------
KS-2samp test (or histogram peak compare if scipy unavailable) on pi_pred_eff
distributions between matched_Exp and matched_Unexp. If means differ by > 10%,
flag the matching as not tight.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circular_distance(a: torch.Tensor, b: torch.Tensor, period: float = 180.0) -> torch.Tensor:
    """Absolute circular distance on [0, period)."""
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def _load_decoder(ckpt: dict, N: int, device: torch.device) -> nn.Linear:
    """Instantiate and load the trained orientation decoder.

    Supports new (`loss_heads["orientation_decoder"]`) and legacy
    (`decoder_state`) checkpoint formats.
    """
    decoder = nn.Linear(N, N).to(device)
    if "loss_heads" in ckpt and isinstance(ckpt["loss_heads"], dict) \
            and "orientation_decoder" in ckpt["loss_heads"]:
        decoder.load_state_dict(ckpt["loss_heads"]["orientation_decoder"])
    elif "decoder_state" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state"])
    else:
        raise RuntimeError(
            "Checkpoint has no orientation_decoder weights "
            "(tried ckpt['loss_heads']['orientation_decoder'] and ckpt['decoder_state'])"
        )
    decoder.eval()
    return decoder


def bootstrap_acc_ci(correct: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> tuple[float | None, float | None]:
    """95% bootstrap CI on a binary accuracy array (percentile method)."""
    n = correct.shape[0]
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    accs = correct[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def roll_to_center(r_l23: np.ndarray, true_ch: np.ndarray, center_idx: int) -> np.ndarray:
    """Vectorised version of np.roll(r[i], shift=center_idx-true_ch[i]) per row.

    r_l23: [n, N] float
    true_ch: [n] int
    center_idx: scalar (typically N//2)

    Returns: [n, N] where rolled[i, center_idx] == r_l23[i, true_ch[i]].
    """
    n, N = r_l23.shape
    shifts = (center_idx - true_ch.astype(np.int64)) % N        # [n]
    # gather index: rolled[i, j] = r_l23[i, (j - shifts[i]) % N]
    cols = (np.arange(N)[None, :] - shifts[:, None]) % N        # [n, N]
    rows = np.arange(n)[:, None]                                # [n, 1]
    return r_l23[rows, cols]


def ks_2sample(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """Two-sample KS test. Uses scipy if available, else reports None."""
    try:
        from scipy.stats import ks_2samp
        ks = ks_2samp(a, b)
        return {"D": float(ks.statistic), "p": float(ks.pvalue)}
    except Exception:
        return {"D": None, "p": None}


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

def collect_records(args, device: torch.device) -> tuple[dict[str, np.ndarray], dict]:
    """Run forward passes, collect per-presentation arrays."""
    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N

    decoder = _load_decoder(ckpt, N, device)

    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi

    # Stage-2 readout window (matches stage2_feedback.py:225-226 and
    # decoding_by_expected.py)
    W_START, W_END = 9, 11
    assert W_END < steps_on, f"Window [{W_START},{W_END}] outside steps_on={steps_on}"

    gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
    )

    rng = torch.Generator().manual_seed(args.rng_seed)

    # Buffers (lists of numpy arrays — concatenated at the end)
    pred_err_buf: list[np.ndarray] = []
    pi_pred_eff_buf: list[np.ndarray] = []
    r_l23_win_buf: list[np.ndarray] = []
    true_ch_buf: list[np.ndarray] = []
    decoder_top1_buf: list[np.ndarray] = []
    task_state_buf: list[np.ndarray] = []

    n_total_pres = 0
    n_ambiguous = 0

    with torch.no_grad():
        for batch_i in range(args.n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net.forward(packed)              # [B, T, N]
            q_pred_all = aux["q_pred_all"]                       # [B, T, N]
            pi_pred_eff_all = aux["pi_pred_eff_all"]             # [B, T, 1]

            B = r_l23_all.shape[0]
            true_ori = metadata.orientations.to(device)          # [B, S]
            is_amb_all = metadata.is_ambiguous.to(device)        # [B, S] bool
            ts_meta = metadata.task_states.to(device)            # [B, S, 2]

            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1

                # V2 prediction at last ISI step
                q_pred_isi = q_pred_all[:, t_isi_last, :]        # [B, N]
                pi_isi = pi_pred_eff_all[:, t_isi_last, 0]       # [B]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg      # [B]

                actual_ori = true_ori[:, pres_i]                 # [B]
                pred_err = circular_distance(pred_ori, actual_ori, period)  # [B]

                # r_l23 window-average over [pres_i*steps_per+9, +11]
                t0 = pres_i * steps_per + W_START
                t1 = pres_i * steps_per + W_END                  # inclusive
                r_l23_win = r_l23_all[:, t0:t1 + 1, :].mean(dim=1)  # [B, N]

                # True channel + decoder top-1
                true_ch = (actual_ori / step_deg).round().long() % N  # [B]
                logits = decoder(r_l23_win)
                pred_ch = logits.argmax(dim=-1)                  # [B]

                # task_state one-hot [..., 0]=focused, [..., 1]=routine
                ts_this = ts_meta[:, pres_i, :]                  # [B, 2]
                regime_idx = ts_this.argmax(dim=-1)              # [B] long

                # Filter out ambiguous BEFORE recording (bookkeeping)
                is_amb = is_amb_all[:, pres_i]
                keep = ~is_amb
                n_total_pres += B
                n_ambiguous += int(is_amb.sum().item())

                if keep.any():
                    keep_cpu = keep.cpu().numpy()
                    pred_err_buf.append(pred_err[keep].cpu().numpy())
                    pi_pred_eff_buf.append(pi_isi[keep].cpu().numpy())
                    r_l23_win_buf.append(r_l23_win[keep].cpu().numpy())
                    true_ch_buf.append(true_ch[keep].cpu().numpy())
                    decoder_top1_buf.append(pred_ch[keep].cpu().numpy())
                    task_state_buf.append(regime_idx[keep].cpu().numpy())

    records = {
        "pred_err": np.concatenate(pred_err_buf, axis=0).astype(np.float32),
        "pi_pred_eff": np.concatenate(pi_pred_eff_buf, axis=0).astype(np.float32),
        "r_l23_win": np.concatenate(r_l23_win_buf, axis=0).astype(np.float32),
        "true_ch": np.concatenate(true_ch_buf, axis=0).astype(np.int64),
        "decoder_top1": np.concatenate(decoder_top1_buf, axis=0).astype(np.int64),
        "task_state": np.concatenate(task_state_buf, axis=0).astype(np.int64),
    }
    # Roll to centre
    records["r_l23_rolled"] = roll_to_center(records["r_l23_win"], records["true_ch"], center_idx=N // 2)

    meta = {
        "N": int(N),
        "step_deg": float(step_deg),
        "center_idx": int(N // 2),
        "seq_length": int(seq_length),
        "batch_size": int(batch_size),
        "steps_on": int(steps_on),
        "steps_isi": int(steps_isi),
        "n_total_pres_post_pres0": int(n_total_pres),
        "n_ambiguous_excluded": int(n_ambiguous),
        "n_records": int(records["pred_err"].shape[0]),
        "feedback_scale": 1.0,
        "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "readout_window": {"start": W_START, "end": W_END, "inclusive": True},
    }
    return records, meta


# ---------------------------------------------------------------------------
# Bucket aggregation
# ---------------------------------------------------------------------------

def summarise_bucket(records: dict[str, np.ndarray], mask: np.ndarray, center_idx: int, name: str) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {
            "label": name, "n": 0,
            "mean_pi_pred_eff": None, "mean_pred_err": None,
            "decoding_acc": None, "decoding_acc_ci95": [None, None],
            "activity_at_stim_ch": None, "activity_at_stim_ch_sem": None,
            "activity_total": None, "activity_total_sem": None,
            "peak_channel_of_mean": None, "peak_value_of_mean": None,
        }
    pred_err = records["pred_err"][mask]
    pi = records["pi_pred_eff"][mask]
    correct = (records["decoder_top1"][mask] == records["true_ch"][mask]).astype(np.float64)
    rolled = records["r_l23_rolled"][mask]                     # [n, N]
    at_ch = rolled[:, center_idx]                              # [n]
    totals = rolled.sum(axis=1)                                # [n]
    mean_curve = rolled.mean(axis=0)                           # [N]
    peak_idx = int(np.argmax(mean_curve))
    peak_val = float(mean_curve[peak_idx])
    acc = float(correct.mean())
    lo, hi = bootstrap_acc_ci(correct)
    return {
        "label": name,
        "n": n,
        "mean_pi_pred_eff": float(pi.mean()),
        "mean_pred_err": float(pred_err.mean()),
        "decoding_acc": acc,
        "decoding_acc_ci95": [lo, hi],
        "activity_at_stim_ch": float(at_ch.mean()),
        "activity_at_stim_ch_sem": float(at_ch.std(ddof=1) / math.sqrt(max(n, 2))) if n >= 2 else None,
        "activity_total": float(totals.mean()),
        "activity_total_sem": float(totals.std(ddof=1) / math.sqrt(max(n, 2))) if n >= 2 else None,
        "peak_channel_of_mean": peak_idx,
        "peak_value_of_mean": peak_val,
    }


def build_buckets(records: dict[str, np.ndarray], meta: dict, exp_pred_err_max: float) -> tuple[dict, float, str]:
    """Build the four buckets and return summaries + chosen pi threshold + Exp criterion."""
    pi_threshold = float(np.percentile(records["pi_pred_eff"], 75.0))
    pred_err = records["pred_err"]
    pi = records["pi_pred_eff"]

    unfilt_exp_mask = pred_err <= 10.0
    unfilt_unexp_mask = pred_err > 20.0
    matched_exp_mask = (pred_err <= exp_pred_err_max) & (pi >= pi_threshold)
    matched_unexp_mask = (pred_err > 20.0) & (pi >= pi_threshold)

    cidx = meta["center_idx"]
    return {
        "unfiltered_expected":   summarise_bucket(records, unfilt_exp_mask, cidx, f"Unfiltered Exp (pred_err<=10°)"),
        "unfiltered_unexpected": summarise_bucket(records, unfilt_unexp_mask, cidx, f"Unfiltered Unexp (pred_err>20°)"),
        "matched_expected":      summarise_bucket(records, matched_exp_mask, cidx, f"Matched Exp (pred_err<={exp_pred_err_max}° & pi>=Q75)"),
        "matched_unexpected":    summarise_bucket(records, matched_unexp_mask, cidx, f"Matched Unexp (pred_err>20° & pi>=Q75)"),
    }, pi_threshold, f"pred_err<={exp_pred_err_max}°"


def per_task_state_split(records: dict[str, np.ndarray], meta: dict, exp_pred_err_max: float, pi_threshold: float) -> dict:
    pred_err = records["pred_err"]; pi = records["pi_pred_eff"]
    ts = records["task_state"]
    cidx = meta["center_idx"]
    out = {}
    for name, idx in [("focused", 0), ("routine", 1)]:
        ts_mask = ts == idx
        out[name] = {
            "unfiltered_expected":   summarise_bucket(records, ts_mask & (pred_err <= 10.0), cidx, f"{name} Unfilt Exp"),
            "unfiltered_unexpected": summarise_bucket(records, ts_mask & (pred_err > 20.0), cidx, f"{name} Unfilt Unexp"),
            "matched_expected":      summarise_bucket(records, ts_mask & (pred_err <= exp_pred_err_max) & (pi >= pi_threshold), cidx, f"{name} Matched Exp"),
            "matched_unexpected":    summarise_bucket(records, ts_mask & (pred_err > 20.0) & (pi >= pi_threshold), cidx, f"{name} Matched Unexp"),
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_ci(ci: list) -> str:
    if ci[0] is None or ci[1] is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def print_table(buckets: dict) -> None:
    keys = ["unfiltered_expected", "unfiltered_unexpected", "matched_expected", "matched_unexpected"]
    short = ["unfilt Exp", "unfilt Unexp", "matched Exp", "matched Unexp"]
    headers = ["metric"] + short
    rows = [
        ("n",                lambda b: f"{b['n']}"),
        ("mean pi_pred_eff", lambda b: f"{b['mean_pi_pred_eff']:.3f}" if b['mean_pi_pred_eff'] is not None else "n/a"),
        ("mean pred_err",    lambda b: f"{b['mean_pred_err']:.2f}°" if b['mean_pred_err'] is not None else "n/a"),
        ("decoding acc",     lambda b: f"{b['decoding_acc']:.3f} {fmt_ci(b['decoding_acc_ci95'])}" if b['decoding_acc'] is not None else "n/a"),
        ("activity @ stim",  lambda b: f"{b['activity_at_stim_ch']:.3f}±{b['activity_at_stim_ch_sem']:.4f}" if b['activity_at_stim_ch'] is not None else "n/a"),
        ("activity total",   lambda b: f"{b['activity_total']:.3f}±{b['activity_total_sem']:.4f}" if b['activity_total'] is not None else "n/a"),
        ("peak ch (mean)",   lambda b: f"ch{b['peak_channel_of_mean']} ({b['peak_value_of_mean']:.3f})" if b['peak_channel_of_mean'] is not None else "n/a"),
    ]
    col_widths = [max(len(h), 18) for h in headers]
    for r in rows:
        for i, k in enumerate(keys):
            val = r[1](buckets[k])
            col_widths[i + 1] = max(col_widths[i + 1], len(val))

    def fmt_row(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for label, fn in rows:
        print(fmt_row([label] + [fn(buckets[k]) for k in keys]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="Sweep YAML matching the checkpoint's architecture.")
    p.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt file.")
    p.add_argument("--output", required=True, help="Path for JSON output.")
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--label", default="", help="Human label (default: checkpoint basename).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[setup] config={args.config}  checkpoint={args.checkpoint}  device={device}  n_batches={args.n_batches}  seed={args.rng_seed}")

    records, meta = collect_records(args, device)
    print(f"[collect] n_records={meta['n_records']}  ambiguous_excluded={meta['n_ambiguous_excluded']}/{meta['n_total_pres_post_pres0']}")

    # --- First pass: matched Exp at pred_err <= 2.5° ---
    buckets, pi_threshold, exp_criterion = build_buckets(records, meta, exp_pred_err_max=2.5)
    matched_exp_n = buckets["matched_expected"]["n"]

    # Widen if underpowered
    widened = False
    if matched_exp_n < 200:
        widened = True
        print(f"[matching] matched_Exp n={matched_exp_n} < 200 — widening pred_err to <=5°")
        buckets, pi_threshold, exp_criterion = build_buckets(records, meta, exp_pred_err_max=5.0)
        matched_exp_n = buckets["matched_expected"]["n"]

    underpowered = matched_exp_n < 200

    # --- Sanity: KS / mean comparison of pi_pred_eff between matched buckets ---
    pi_arr = records["pi_pred_eff"]
    pred_err_arr = records["pred_err"]
    exp_pred_err_max = 2.5 if not widened else 5.0
    me_mask = (pred_err_arr <= exp_pred_err_max) & (pi_arr >= pi_threshold)
    mu_mask = (pred_err_arr > 20.0) & (pi_arr >= pi_threshold)
    me_pi = pi_arr[me_mask]; mu_pi = pi_arr[mu_mask]
    matching_check = ks_2sample(me_pi, mu_pi)
    if me_pi.size and mu_pi.size:
        mean_diff_pct = abs(float(me_pi.mean()) - float(mu_pi.mean())) / max(float(me_pi.mean()), 1e-9) * 100
    else:
        mean_diff_pct = None
    matching_check["mean_pi_pct_diff"] = mean_diff_pct
    matching_check["matched_exp_mean_pi"] = float(me_pi.mean()) if me_pi.size else None
    matching_check["matched_unexp_mean_pi"] = float(mu_pi.mean()) if mu_pi.size else None
    matching_check["mean_pi_diff_warn"] = (mean_diff_pct is not None and mean_diff_pct > 10.0)

    # --- Per-task-state split ---
    per_ts = per_task_state_split(records, meta, exp_pred_err_max, pi_threshold)

    result = {
        "label": args.label or os.path.basename(args.checkpoint),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "meta": meta,
        "pi_pred_eff_threshold_q75": pi_threshold,
        "exp_criterion": exp_criterion,
        "exp_pred_err_max_used": exp_pred_err_max,
        "widened_due_to_low_n": widened,
        "underpowered_warning": underpowered,
        "buckets": buckets,
        "matching_check": matching_check,
        "by_task_state": per_ts,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # --- Console summary ---
    print()
    print(f"[matching] pi_pred_eff Q75 threshold = {pi_threshold:.3f}")
    print(f"[matching] matched_Exp mean pi = {matching_check['matched_exp_mean_pi']}")
    print(f"[matching] matched_Unexp mean pi = {matching_check['matched_unexp_mean_pi']}")
    print(f"[matching] mean diff (%) = {matching_check['mean_pi_pct_diff']}")
    print(f"[matching] KS D = {matching_check['D']}, p = {matching_check['p']}")
    if matching_check["mean_pi_diff_warn"]:
        print(f"[WARN] mean pi_pred_eff differs between matched buckets by > 10% — matching not tight")
    if underpowered:
        print(f"[WARN] matched_Exp n = {matched_exp_n} < 200 even after widening — underpowered")
    print()
    print("=== POOLED ===")
    print_table(result["buckets"])
    for ts_name, ts_buckets in per_ts.items():
        print()
        print(f"=== {ts_name.upper()} ===")
        print_table(ts_buckets)
    print()
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
