#!/usr/bin/env python3
"""Matched-quality 3-row averaged ring heatmap (Task #34).

Builds an averaged tuning-ring figure for R1+R2 with three buckets:
  - Expected   (stim present, pred_err <= 5°,  pi_pred_eff >= pooled Q75)
  - Unexpected (stim present, pred_err >  20°, pi_pred_eff >= pooled Q75)
  - Omission   (stim absent at target presentation, pi_pred_eff >= pooled Q75)

The point is to reflect the HMM training distribution (not a single matched-
sequence trial), control for trial quality across all 3 buckets, and visualise
the energy / decoding-accuracy findings on R1+R2.

Method
------
For each batch of `batch_size` HMM sequences (seq_length=25), run TWO forward
passes:

  Pass A — stimulus present (normal). Use the stimulus tensor as generated.
           Yields Expected / Unexpected data (split by V2 prediction error
           against the actual stimulus orientation).
  Pass B — stimulus zeroed at ONE target presentation. The target index is
           fixed across all sequences (default: last presentation of the
           sequence, index = seq_length - 1). The ON window of that
           presentation gets `stim[:, onset:onset+steps_on, :] = 0`. Everything
           else is identical (same seed, same context). Yields Omission data.

Per-trial measurements:
  - r_l23 averaged over the readout window [9, 11] (Stage-2 trained window).
  - pi_pred_eff at the LAST ISI step BEFORE the relevant ON window
    (`t_isi = pres_i * steps_per - 1`).
  - For Pass A trials: pred_err vs the ACTUAL stimulus orientation; true_ch
    from `metadata.orientations[:, pres_i]`.
  - For Pass B (Omission): no pred_err split; true_ch is the WOULD-HAVE-BEEN
    target orientation (from `metadata.orientations[:, target_idx]`).
  - decoder_top1 from the trained orientation_decoder; decoder_correct =
    decoder_top1 == true_ch (for Omission this should be ~chance ~1/N).

Pi quality cut: Q75 of the union of (Pass A all post-pres-0 pi values) ∪
(Pass B target pi values) is used as the matching threshold. Bucket masks:
  - Expected: pred_err <= 5°  AND pi >= Q75 AND ~ambiguous (Pass A records)
  - Unexpected: pred_err > 20° AND pi >= Q75 AND ~ambiguous (Pass A records)
  - Omission: pi >= Q75 AND ~ambiguous_at_target (Pass B records)

If any bucket is < 200 trials, the Expected pred_err tolerance is widened to
10° (and reported); if still < 200, the Q75 threshold is dropped to Q50 (and
reported). The widened thresholds and counts are surfaced in the JSON.

Aggregation per bucket:
  - re-centered r_l23 vector (np.roll(r, 18 - true_ch)) → mean over trials =
    "the ring" (a [N] vector).
  - peak @ true = mean_ring[18]; total = mean_ring.sum(); FWHM via interpolated
    half-max (helper from plot_tuning_ring_extended.py).
  - decoder accuracy with 95% bootstrap CI; mean pi_pred_eff.

Figure: 3 rows, 1 column polar wedge bar plots, viridis cmap, shared vmax,
right-side colorbar, per-row annotation box. Saved to the path given by
`--output-fig`.

Design notes
------------
* Seed (42) and HMM generator state are reset BEFORE each pass within a batch
  so that Pass A and Pass B see IDENTICAL stimulus sequences modulo the
  zeroed target window. The two forward calls share the same HMM metadata.
* `feedback_scale = 1.0` (FB ON), oracle_mode = False, eval mode.
* All ambiguous trials are dropped (Pass A: ambiguous at the recorded pres_i;
  Pass B: ambiguous at the target index).
* The decoder for Omission is reported but flagged "n/a" in the figure
  annotation since "correctness" against an absent stimulus is meaningless.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

# Add repo root + scripts/ to sys.path so we can import from src/ and other scripts/
_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence

# Reused helpers
from matched_quality_sim import (
    circular_distance,
    _load_decoder,
    bootstrap_acc_ci,
    roll_to_center,
    ks_2sample,
)
from plot_tuning_ring_extended import (
    fwhm_of_curve,
    _plot_ring_base,
)


# ---------------------------------------------------------------------------
# Two-pass record collection
# ---------------------------------------------------------------------------

def collect_records(args, device: torch.device) -> tuple[dict[str, Any], dict]:
    """Run paired forward passes on each batch and collect per-trial records.

    Returns
    -------
    records : dict with keys
        "passA" : dict[np.ndarray]
            Pass A (stim present) per-record arrays:
              pred_err, pi_pred_eff, r_l23_win, r_l23_rolled, true_ch,
              decoder_top1.  Length = n_total Pass-A presentations after
              ambiguous exclusion.
        "passB_target" : dict[np.ndarray]
            Pass B (stim absent at target_idx) records, ONE per sequence
            (after target-ambiguous exclusion):
              pi_pred_eff (at pre-target ISI), r_l23_win, r_l23_rolled,
              true_ch (planned), decoder_top1.
    meta : dict with N, step_deg, target_idx, seq_length, batch_size, etc.
    """
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

    W_START, W_END = 9, 11
    assert W_END < steps_on, f"Window [{W_START},{W_END}] outside steps_on={steps_on}"

    # Target presentation index (default = last presentation = seq_length - 1)
    target_idx = (seq_length - 1) if args.target_idx is None else int(args.target_idx)
    assert 1 <= target_idx <= seq_length - 1, \
        f"target_idx={target_idx} must be in [1, seq_length-1={seq_length - 1}]"
    target_onset = target_idx * steps_per
    target_isi_pre = target_onset - 1               # last ISI step before target ON

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

    # Pass-A buffers (all post-pres-0 records from the stim-present forward)
    A_pred_err: list[np.ndarray] = []
    A_pi: list[np.ndarray] = []
    A_r_win: list[np.ndarray] = []
    A_true_ch: list[np.ndarray] = []
    A_decoder_top1: list[np.ndarray] = []

    # Pass-B buffers (one record per sequence at target_idx, stim zeroed there)
    B_pi: list[np.ndarray] = []
    B_r_win: list[np.ndarray] = []
    B_true_ch: list[np.ndarray] = []
    B_decoder_top1: list[np.ndarray] = []

    n_total_A_pres = 0
    n_amb_A = 0
    n_total_B_seq = 0
    n_amb_B = 0

    with torch.no_grad():
        for batch_i in range(args.n_batches):
            # ONE metadata for both passes (same seed, same HMM realisation)
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            true_ori = metadata.orientations.to(device)        # [B, S]
            is_amb_all = metadata.is_ambiguous.to(device)      # [B, S] bool

            # --- Pass A: forward with normal stim ---
            packed_A = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_A, _, aux_A = net.forward(packed_A)          # [B, T, N]
            q_pred_A = aux_A["q_pred_all"]                     # [B, T, N]
            pi_eff_A = aux_A["pi_pred_eff_all"]                # [B, T, 1]

            B = r_l23_A.shape[0]

            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1
                q_pred_isi = q_pred_A[:, t_isi_last, :]        # [B, N]
                pi_isi = pi_eff_A[:, t_isi_last, 0]            # [B]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg
                actual_ori = true_ori[:, pres_i]
                pred_err = circular_distance(pred_ori, actual_ori, period)

                t0 = pres_i * steps_per + W_START
                t1 = pres_i * steps_per + W_END                # inclusive
                r_win = r_l23_A[:, t0:t1 + 1, :].mean(dim=1)   # [B, N]
                true_ch = (actual_ori / step_deg).round().long() % N

                logits = decoder(r_win)
                pred_ch = logits.argmax(dim=-1)

                is_amb = is_amb_all[:, pres_i]
                keep = ~is_amb
                n_total_A_pres += B
                n_amb_A += int(is_amb.sum().item())

                if keep.any():
                    A_pred_err.append(pred_err[keep].cpu().numpy())
                    A_pi.append(pi_isi[keep].cpu().numpy())
                    A_r_win.append(r_win[keep].cpu().numpy())
                    A_true_ch.append(true_ch[keep].cpu().numpy())
                    A_decoder_top1.append(pred_ch[keep].cpu().numpy())

            # --- Pass B: zero stim at target's ON window, forward again ---
            stim_B = stim_seq.clone()
            stim_B[:, target_onset:target_onset + steps_on, :] = 0.0
            packed_B = net.pack_inputs(stim_B, cue_seq, ts_seq)
            r_l23_B, _, aux_B = net.forward(packed_B)
            pi_eff_B = aux_B["pi_pred_eff_all"]                # [B, T, 1]

            # Pi at PRE-TARGET ISI step (same step in both passes; here we
            # take pass-B's value to be explicit per the task brief)
            pi_target_B = pi_eff_B[:, target_isi_pre, 0]       # [B]

            t0 = target_onset + W_START
            t1 = target_onset + W_END                          # inclusive
            r_win_B = r_l23_B[:, t0:t1 + 1, :].mean(dim=1)     # [B, N]

            actual_ori_B = true_ori[:, target_idx]             # [B]
            true_ch_B = (actual_ori_B / step_deg).round().long() % N

            logits_B = decoder(r_win_B)
            pred_ch_B = logits_B.argmax(dim=-1)

            is_amb_B = is_amb_all[:, target_idx]
            keep_B = ~is_amb_B
            n_total_B_seq += B
            n_amb_B += int(is_amb_B.sum().item())

            if keep_B.any():
                B_pi.append(pi_target_B[keep_B].cpu().numpy())
                B_r_win.append(r_win_B[keep_B].cpu().numpy())
                B_true_ch.append(true_ch_B[keep_B].cpu().numpy())
                B_decoder_top1.append(pred_ch_B[keep_B].cpu().numpy())

    # Concat
    passA = {
        "pred_err": np.concatenate(A_pred_err, axis=0).astype(np.float32),
        "pi_pred_eff": np.concatenate(A_pi, axis=0).astype(np.float32),
        "r_l23_win": np.concatenate(A_r_win, axis=0).astype(np.float32),
        "true_ch": np.concatenate(A_true_ch, axis=0).astype(np.int64),
        "decoder_top1": np.concatenate(A_decoder_top1, axis=0).astype(np.int64),
    }
    passA["r_l23_rolled"] = roll_to_center(passA["r_l23_win"], passA["true_ch"], center_idx=N // 2)

    passB = {
        "pi_pred_eff": np.concatenate(B_pi, axis=0).astype(np.float32),
        "r_l23_win": np.concatenate(B_r_win, axis=0).astype(np.float32),
        "true_ch": np.concatenate(B_true_ch, axis=0).astype(np.int64),
        "decoder_top1": np.concatenate(B_decoder_top1, axis=0).astype(np.int64),
    }
    passB["r_l23_rolled"] = roll_to_center(passB["r_l23_win"], passB["true_ch"], center_idx=N // 2)

    meta = {
        "N": int(N),
        "step_deg": float(step_deg),
        "center_idx": int(N // 2),
        "seq_length": int(seq_length),
        "batch_size": int(batch_size),
        "steps_on": int(steps_on),
        "steps_isi": int(steps_isi),
        "target_idx": int(target_idx),
        "target_onset_step": int(target_onset),
        "target_isi_pre_step": int(target_isi_pre),
        "n_passA_records": int(passA["pred_err"].shape[0]),
        "n_passA_total_pres": int(n_total_A_pres),
        "n_passA_ambiguous_excluded": int(n_amb_A),
        "n_passB_records": int(passB["pi_pred_eff"].shape[0]),
        "n_passB_total_seq": int(n_total_B_seq),
        "n_passB_ambiguous_excluded": int(n_amb_B),
        "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "feedback_scale": 1.0,
        "readout_window": {"start": W_START, "end": W_END, "inclusive": True},
    }
    return {"passA": passA, "passB_target": passB}, meta


# ---------------------------------------------------------------------------
# Bucket masking and summarisation
# ---------------------------------------------------------------------------

def make_buckets(records: dict, exp_pred_err_max: float, pi_q_pct: float
                 ) -> tuple[dict[str, dict], float]:
    """Construct the 3 bucket subsets.

    Pi threshold = `pi_q_pct` percentile of the UNION of Pass-A pi values and
    Pass-B target pi values.

    Returns
    -------
    buckets : dict
        Each value is a dict with keys "r_rolled", "true_ch", "decoder_top1",
        "pi_pred_eff", "pred_err" (None for omission), "is_omission" (bool).
    pi_threshold : float
    """
    A = records["passA"]
    B = records["passB_target"]

    pooled = np.concatenate([A["pi_pred_eff"], B["pi_pred_eff"]], axis=0)
    pi_threshold = float(np.percentile(pooled, pi_q_pct))

    A_pi_ok = A["pi_pred_eff"] >= pi_threshold
    A_pred_err = A["pred_err"]
    exp_mask = A_pi_ok & (A_pred_err <= exp_pred_err_max)
    unexp_mask = A_pi_ok & (A_pred_err > 20.0)
    B_pi_ok = B["pi_pred_eff"] >= pi_threshold

    def slice_A(mask: np.ndarray) -> dict:
        return {
            "r_rolled": A["r_l23_rolled"][mask],
            "r_win": A["r_l23_win"][mask],
            "true_ch": A["true_ch"][mask],
            "decoder_top1": A["decoder_top1"][mask],
            "pi_pred_eff": A["pi_pred_eff"][mask],
            "pred_err": A["pred_err"][mask],
            "is_omission": False,
        }

    def slice_B(mask: np.ndarray) -> dict:
        return {
            "r_rolled": B["r_l23_rolled"][mask],
            "r_win": B["r_l23_win"][mask],
            "true_ch": B["true_ch"][mask],
            "decoder_top1": B["decoder_top1"][mask],
            "pi_pred_eff": B["pi_pred_eff"][mask],
            "pred_err": None,
            "is_omission": True,
        }

    return {
        "expected": slice_A(exp_mask),
        "unexpected": slice_A(unexp_mask),
        "omission": slice_B(B_pi_ok),
    }, pi_threshold


def summarise(bucket: dict, center_idx: int, step_deg: float) -> dict[str, Any]:
    n = int(bucket["r_rolled"].shape[0])
    if n == 0:
        return {
            "n": 0, "mean_ring": None, "peak_at_true": None, "total": None,
            "fwhm_deg": None, "decoder_acc": None, "decoder_acc_ci95": [None, None],
            "mean_pi_pred_eff": None, "mean_pred_err": None,
        }
    mean_ring = bucket["r_rolled"].mean(axis=0)
    peak = float(mean_ring[center_idx])
    total = float(mean_ring.sum())
    fwhm = float(fwhm_of_curve(mean_ring, step_deg))
    correct = (bucket["decoder_top1"] == bucket["true_ch"]).astype(np.float64)
    acc = float(correct.mean())
    lo, hi = bootstrap_acc_ci(correct)
    return {
        "n": n,
        "mean_ring": mean_ring.astype(np.float32).tolist(),
        "peak_at_true": peak,
        "total": total,
        "fwhm_deg": fwhm,
        "decoder_acc": acc,
        "decoder_acc_ci95": [lo, hi],
        "mean_pi_pred_eff": float(bucket["pi_pred_eff"].mean()),
        "mean_pred_err": (None if bucket["pred_err"] is None
                          else float(bucket["pred_err"].mean())),
    }


def pairwise_pi_check(buckets: dict) -> dict[str, dict]:
    """KS + mean-diff between every pair of buckets' pi_pred_eff."""
    names = ["expected", "unexpected", "omission"]
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = buckets[names[i]]["pi_pred_eff"]
            b = buckets[names[j]]["pi_pred_eff"]
            if a.size == 0 or b.size == 0:
                out[f"{names[i]}_vs_{names[j]}"] = {
                    "ks": {"D": None, "p": None},
                    "mean_a": None, "mean_b": None, "mean_pct_diff": None,
                }
                continue
            ks = ks_2sample(a, b)
            mean_a = float(a.mean()); mean_b = float(b.mean())
            mean_pct = abs(mean_a - mean_b) / max(mean_a, 1e-9) * 100.0
            out[f"{names[i]}_vs_{names[j]}"] = {
                "ks": ks,
                "mean_a": mean_a, "mean_b": mean_b,
                "mean_pct_diff": mean_pct,
                "warn_pct_gt_10": mean_pct > 10.0,
            }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_3row_figure(summaries: dict[str, dict], meta: dict, fig_path: str,
                     title: str) -> None:
    """3-row × 1-col polar wedge ring figure (Expected / Unexpected / Omission).

    Shared vmax across all 3 panels (max of mean_ring across buckets). Viridis.
    Annotation box (lower-left) per panel: n, decoding acc, peak @ true,
    total, FWHM, mean pi.
    """
    cmap = cm.get_cmap("viridis")
    N = meta["N"]
    center_idx = meta["center_idx"]
    step_deg = meta["step_deg"]

    # vmax over the 3 mean rings
    rings = []
    for name in ["expected", "unexpected", "omission"]:
        r = summaries[name]["mean_ring"]
        if r is None:
            rings.append(np.zeros(N))
        else:
            rings.append(np.array(r, dtype=float))
    vmax = max(float(r.max()) for r in rings)
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(6.5, 14.0))
    row_titles = ["Expected", "Unexpected", "Omission"]
    bucket_keys = ["expected", "unexpected", "omission"]

    for i, (key, label) in enumerate(zip(bucket_keys, row_titles)):
        ax = fig.add_subplot(3, 1, i + 1, projection="polar")
        s = summaries[key]
        ring = rings[i]
        _plot_ring_base(ax, ring, vmax, cmap)

        if s["n"] == 0:
            ax.set_title(f"{label} — empty bucket", fontsize=12, pad=14)
            continue

        if key == "omission":
            dec_str = "n/a (no stim)"
        else:
            ci = s["decoder_acc_ci95"]
            ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                      if ci[0] is not None else "n/a")
            dec_str = f"{s['decoder_acc']:.3f} {ci_str}"

        ax.set_title(f"{label} (n={s['n']})", fontsize=12, pad=14)
        ax.text(
            0.02, 0.02,
            f"n = {s['n']}\n"
            f"decoding acc = {dec_str}\n"
            f"peak @ true (ch {center_idx}) = {s['peak_at_true']:.3f}\n"
            f"total L2/3 = {s['total']:.2f}\n"
            f"FWHM = {s['fwhm_deg']:.1f}°\n"
            f"mean pi = {s['mean_pi_pred_eff']:.3f}",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85, linewidth=0.6),
        )

    # Shared right-side colorbar
    cbar_ax = fig.add_axes([0.86, 0.10, 0.025, 0.78])
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("L2/3 activity", fontsize=10)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.84, 0.97))

    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-fig", required=True, help="PNG path for the 3-row figure.")
    p.add_argument("--output-json", required=True, help="JSON path for stats output.")
    p.add_argument("--label", default="", help="Label used in figure title.")
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--target-idx", type=int, default=None,
                   help="Presentation index for Pass-B omission. Default = "
                        "seq_length - 1 (last presentation).")
    p.add_argument("--exp-pred-err-max", type=float, default=5.0,
                   help="Initial Expected pred_err tolerance (deg). Widens to "
                        "10° if Expected n < 200.")
    p.add_argument("--pi-pct", type=float, default=75.0,
                   help="Initial pi pooled percentile cutoff. Drops to 50 if "
                        "any bucket still < 200 after Exp widening.")
    p.add_argument("--min-bucket-n", type=int, default=200,
                   help="Soft floor for bucket sizes. Triggers widening cascade.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    label = args.label or os.path.basename(args.checkpoint)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] device={device}  n_batches={args.n_batches}  seed={args.rng_seed}", flush=True)

    records, meta = collect_records(args, device)
    print(
        f"[collect] passA: {meta['n_passA_records']} records "
        f"({meta['n_passA_ambiguous_excluded']}/{meta['n_passA_total_pres']} ambiguous excluded)",
        flush=True,
    )
    print(
        f"[collect] passB(target_idx={meta['target_idx']}): {meta['n_passB_records']} records "
        f"({meta['n_passB_ambiguous_excluded']}/{meta['n_passB_total_seq']} ambiguous excluded)",
        flush=True,
    )

    # --- Initial bucketing at exp_pred_err_max=5°, Q75 ---
    exp_max = args.exp_pred_err_max
    pi_pct = args.pi_pct
    widening_log: list[str] = []

    buckets, pi_threshold = make_buckets(records, exp_pred_err_max=exp_max, pi_q_pct=pi_pct)
    n_exp = buckets["expected"]["r_rolled"].shape[0]
    n_unexp = buckets["unexpected"]["r_rolled"].shape[0]
    n_om = buckets["omission"]["r_rolled"].shape[0]
    print(f"[init] pi_threshold(Q{pi_pct:g})={pi_threshold:.4f}  "
          f"n_exp={n_exp}  n_unexp={n_unexp}  n_om={n_om}", flush=True)

    # Cascade 1: widen Expected pred_err to 10°
    if min(n_exp, n_unexp, n_om) < args.min_bucket_n:
        msg = (f"min bucket n {min(n_exp, n_unexp, n_om)} < {args.min_bucket_n}; "
               f"widening Expected pred_err {exp_max}°→10°")
        print(f"[widen] {msg}", flush=True)
        widening_log.append(msg)
        exp_max = 10.0
        buckets, pi_threshold = make_buckets(records, exp_pred_err_max=exp_max, pi_q_pct=pi_pct)
        n_exp = buckets["expected"]["r_rolled"].shape[0]
        n_unexp = buckets["unexpected"]["r_rolled"].shape[0]
        n_om = buckets["omission"]["r_rolled"].shape[0]
        print(f"[widen-1] n_exp={n_exp}  n_unexp={n_unexp}  n_om={n_om}", flush=True)

    # Cascade 2: drop pi cut to Q50
    if min(n_exp, n_unexp, n_om) < args.min_bucket_n:
        msg = (f"min bucket n {min(n_exp, n_unexp, n_om)} < {args.min_bucket_n}; "
               f"dropping pi pooled cut Q{pi_pct:g}→Q50")
        print(f"[widen] {msg}", flush=True)
        widening_log.append(msg)
        pi_pct = 50.0
        buckets, pi_threshold = make_buckets(records, exp_pred_err_max=exp_max, pi_q_pct=pi_pct)
        n_exp = buckets["expected"]["r_rolled"].shape[0]
        n_unexp = buckets["unexpected"]["r_rolled"].shape[0]
        n_om = buckets["omission"]["r_rolled"].shape[0]
        print(f"[widen-2] n_exp={n_exp}  n_unexp={n_unexp}  n_om={n_om}", flush=True)

    underpowered = min(n_exp, n_unexp, n_om) < args.min_bucket_n

    # --- Aggregate ---
    summaries = {
        name: summarise(b, center_idx=meta["center_idx"], step_deg=meta["step_deg"])
        for name, b in buckets.items()
    }

    # --- Pi distribution check across the 3 buckets ---
    pi_check = pairwise_pi_check(buckets)

    # --- Print report table ---
    print()
    print(f"[matching] final exp_pred_err_max = {exp_max}°  pi_pct = Q{pi_pct:g}  "
          f"pi_threshold = {pi_threshold:.4f}", flush=True)
    if widening_log:
        print(f"[matching] widening cascade: {widening_log}", flush=True)
    if underpowered:
        print(f"[WARN] underpowered after widening (min bucket n = "
              f"{min(n_exp, n_unexp, n_om)} < {args.min_bucket_n})", flush=True)

    print()
    headers = ["bucket", "n", "decoding acc (95% CI)", "peak@true", "total", "FWHM(°)", "mean pi", "mean pred_err"]
    rows = []
    for name in ["expected", "unexpected", "omission"]:
        s = summaries[name]
        if s["n"] == 0:
            rows.append([name, "0", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        ci = s["decoder_acc_ci95"]
        ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a")
        dec_cell = (f"{s['decoder_acc']:.3f} {ci_str}"
                    if name != "omission"
                    else f"{s['decoder_acc']:.3f} {ci_str} (n/a interp.)")
        pred_err_cell = (f"{s['mean_pred_err']:.2f}°"
                         if s["mean_pred_err"] is not None else "n/a")
        rows.append([
            name, str(s["n"]), dec_cell,
            f"{s['peak_at_true']:.3f}", f"{s['total']:.2f}",
            f"{s['fwhm_deg']:.2f}", f"{s['mean_pi_pred_eff']:.3f}", pred_err_cell,
        ])
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in col_w))
    for r in rows:
        print(fmt.format(*r))

    print()
    print("[pi-comparability]")
    for pair_name, info in pi_check.items():
        ks = info["ks"]
        warn = " [WARN >10%]" if info.get("warn_pct_gt_10") else ""
        ks_str = (f"D={ks['D']:.3f} p={ks['p']:.3g}"
                  if ks["D"] is not None else "scipy unavailable")
        print(f"  {pair_name:<28s} mean_a={info['mean_a']:.4f} "
              f"mean_b={info['mean_b']:.4f} pct_diff={info['mean_pct_diff']:.2f}%  {ks_str}{warn}",
              flush=True)

    # --- Render figure ---
    title = f"Matched-quality averaged tuning rings — {label}\n(HMM, n_trials matched on pi_pred)"
    plot_3row_figure(summaries, meta, args.output_fig, title=title)
    print(f"\n[fig] wrote {args.output_fig}", flush=True)

    # --- JSON ---
    result = {
        "label": label,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "meta": meta,
        "exp_pred_err_max_used": exp_max,
        "pi_pct_used": pi_pct,
        "pi_threshold": pi_threshold,
        "widening_cascade": widening_log,
        "min_bucket_n_floor": int(args.min_bucket_n),
        "underpowered": bool(underpowered),
        "buckets": {name: summaries[name] for name in ["expected", "unexpected", "omission"]},
        "pi_comparability": pi_check,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
