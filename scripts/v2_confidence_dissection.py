#!/usr/bin/env python3
"""V2 confidence (pi_pred_eff) dissection: 3 focused tests (Task #40).

Empirically tests whether V2 confidence (pi_pred_eff) causally explains
the direction flip between normal HMM Expected-vs-Unexpected
(Expected < Unexpected at stim channel, Task #30) and clean-march
(Expected > Unexpected, Task #37).

Test 1 — pi_pred distribution comparison
    4 groups: Normal-HMM-Expected, Normal-HMM-Unexpected,
    Clean-march-Expected, Clean-march-Unexpected.
    Note: clean-march groups share the same pre-probe state in the
    3-pass design, so their pi_pred distributions are identical.

Test 2 — quartile-binned Expected vs Unexpected delta
    Bin ALL trials by pi_pred quartiles (computed on pooled Exp+Unexp).
    Within each quartile, compute peak@true and decoding accuracy for
    Expected (pred_err ≤ 10°) and Unexpected (pred_err > 20°).
    Key question: does the Δpeak direction FLIP at high pi?

Test 3 — nearest-neighbor pi_pred matched pairs
    1-to-1 matching without replacement. After matching, re-measure
    peak@true and decoding accuracy.
    Key question: with pi tightly matched, does Expected < Unexpected
    persist?

Bonus — preceding aligned run length
    How many consecutive preceding presentations involved ≤ 5° changes.
    Reported per group and per quartile.

Output
------
Figure  : ``--output-fig``   (default docs/figures/v2_confidence_dissection_r1_2.png)
JSON    : ``--output-json``  (default results/v2_confidence_dissection_r1_2.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence

from matched_quality_sim import (
    circular_distance,
    _load_decoder,
    bootstrap_acc_ci,
    roll_to_center,
    ks_2sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signed_circ_delta(b: np.ndarray, a: np.ndarray, period: float) -> np.ndarray:
    """Signed circular delta b − a in (−period/2, period/2]."""
    return ((b - a + period / 2.0) % period) - period / 2.0


def compute_run_lengths(orientations: np.ndarray, period: float,
                        step_tol: float) -> np.ndarray:
    """For each presentation, count consecutive preceding smooth transitions.

    Parameters
    ----------
    orientations : [B, S] float, degrees
    period : orientation period (180)
    step_tol : maximum |delta| to count as "smooth" (5.0)

    Returns
    -------
    run_lengths : [B, S] int
        run_lengths[b, s] = how many consecutive transitions ending at s
        have |signed_circ_delta| ≤ step_tol.
    """
    B, S = orientations.shape
    rl = np.zeros((B, S), dtype=np.int32)
    # deltas[b, s] = signed_circ_delta(ori[b, s], ori[b, s-1])  for s ≥ 1
    deltas = signed_circ_delta(orientations[:, 1:], orientations[:, :-1], period)
    smooth = np.abs(deltas) <= step_tol + 1e-6  # [B, S-1], smooth[b, s-1] for transition s-1→s

    for s in range(1, S):
        for b in range(B):
            count = 0
            k = s - 1  # index into deltas: transition from pres k to pres k+1
            while k >= 0 and smooth[b, k]:
                count += 1
                k -= 1
            rl[b, s] = count
    return rl


def compute_march_flags(orientations: np.ndarray, period: float,
                        transition_step: float, step_tol: float = 0.5
                        ) -> np.ndarray:
    """Flag presentations that qualify as clean 3-step march.

    A presentation s qualifies if:
    - s ≥ 2
    - |signed_delta(ori[s-1], ori[s-2])| ∈ [transition_step - tol, transition_step + tol]
    - signed_delta(ori[s], ori[s-1]) matches signed_delta(ori[s-1], ori[s-2])
      in sign and magnitude (within tol)

    Returns [B, S] bool.
    """
    B, S = orientations.shape
    flags = np.zeros((B, S), dtype=bool)
    if S < 3:
        return flags

    for s in range(2, S):
        d_prev = signed_circ_delta(orientations[:, s - 1], orientations[:, s - 2], period)
        d_curr = signed_circ_delta(orientations[:, s], orientations[:, s - 1], period)
        prev_ok = np.abs(np.abs(d_prev) - transition_step) <= step_tol
        same_dir = np.sign(d_curr) == np.sign(d_prev)
        same_mag = np.abs(np.abs(d_curr) - np.abs(d_prev)) <= step_tol
        flags[:, s] = prev_ok & same_dir & same_mag
    return flags


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_hmm_pool(args, device: torch.device) -> tuple[dict[str, np.ndarray], dict]:
    """Run n_batches HMM forward passes, collect per-presentation arrays.

    Returns per-presentation records (excluding s=0 and ambiguous):
        pred_err, pi_pred_eff, r_l23_rolled, true_ch, decoder_top1,
        task_state, is_march, run_length
    Plus meta dict.
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
    transition_step = stim_cfg.transition_step

    W_START, W_END = 9, 11
    assert W_END < steps_on

    gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=transition_step,
        period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        task_p_switch=stim_cfg.task_p_switch,
    )

    rng = torch.Generator().manual_seed(args.rng_seed)

    # Buffers
    pred_err_buf = []
    pi_pred_eff_buf = []
    r_l23_win_buf = []
    true_ch_buf = []
    decoder_top1_buf = []
    task_state_buf = []
    is_march_buf = []
    run_length_buf = []

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
            r_l23_all, _, aux = net.forward(packed)
            q_pred_all = aux["q_pred_all"]
            pi_pred_eff_all = aux["pi_pred_eff_all"]

            B = r_l23_all.shape[0]
            true_ori = metadata.orientations.cpu().numpy()        # [B, S]
            is_amb_all = metadata.is_ambiguous.cpu().numpy()       # [B, S]
            ts_meta = metadata.task_states.cpu().numpy()           # [B, S, 2]

            # Compute run lengths and march flags for this batch
            rl_batch = compute_run_lengths(true_ori, period, transition_step)
            march_batch = compute_march_flags(true_ori, period, transition_step)

            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1

                q_pred_isi = q_pred_all[:, t_isi_last, :]
                pi_isi = pi_pred_eff_all[:, t_isi_last, 0]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg

                actual_ori_t = torch.from_numpy(true_ori[:, pres_i]).float().to(device)
                pred_err_t = circular_distance(pred_ori, actual_ori_t, period)

                t0 = pres_i * steps_per + W_START
                t1 = pres_i * steps_per + W_END
                r_l23_win = r_l23_all[:, t0:t1 + 1, :].mean(dim=1)

                true_ch_t = (actual_ori_t / step_deg).round().long() % N
                logits = decoder(r_l23_win)
                pred_ch = logits.argmax(dim=-1)

                regime_idx = torch.from_numpy(ts_meta[:, pres_i, :]).float().argmax(dim=-1)

                is_amb = is_amb_all[:, pres_i].astype(bool)
                keep = ~is_amb
                n_total_pres += B
                n_ambiguous += int(is_amb.sum())

                if keep.any():
                    pred_err_buf.append(pred_err_t[keep].cpu().numpy())
                    pi_pred_eff_buf.append(pi_isi[keep].cpu().numpy())
                    r_l23_win_buf.append(r_l23_win[keep].cpu().numpy())
                    true_ch_buf.append(true_ch_t[keep].cpu().numpy())
                    decoder_top1_buf.append(pred_ch[keep].cpu().numpy())
                    task_state_buf.append(regime_idx[keep].numpy())
                    is_march_buf.append(march_batch[keep, pres_i])
                    run_length_buf.append(rl_batch[keep, pres_i])

            if (batch_i + 1) % 10 == 0:
                print(f"  batch {batch_i + 1}/{args.n_batches}")

    records = {
        "pred_err": np.concatenate(pred_err_buf).astype(np.float32),
        "pi_pred_eff": np.concatenate(pi_pred_eff_buf).astype(np.float32),
        "r_l23_win": np.concatenate(r_l23_win_buf).astype(np.float32),
        "true_ch": np.concatenate(true_ch_buf).astype(np.int64),
        "decoder_top1": np.concatenate(decoder_top1_buf).astype(np.int64),
        "task_state": np.concatenate(task_state_buf).astype(np.int64),
        "is_march": np.concatenate(is_march_buf).astype(bool),
        "run_length": np.concatenate(run_length_buf).astype(np.int32),
    }
    records["r_l23_rolled"] = roll_to_center(
        records["r_l23_win"], records["true_ch"], center_idx=N // 2,
    )

    meta = {
        "N": int(N),
        "step_deg": float(step_deg),
        "center_idx": int(N // 2),
        "period": float(period),
        "seq_length": int(seq_length),
        "batch_size": int(batch_size),
        "steps_on": int(steps_on),
        "steps_isi": int(steps_isi),
        "transition_step": float(transition_step),
        "n_total_pres": int(n_total_pres),
        "n_ambiguous_excluded": int(n_ambiguous),
        "n_records": int(records["pred_err"].shape[0]),
        "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "readout_window": {"start": W_START, "end": W_END},
    }
    return records, meta


# ---------------------------------------------------------------------------
# Test 1: pi_pred distribution comparison
# ---------------------------------------------------------------------------

def pi_distribution_stats(pi: np.ndarray, label: str) -> dict[str, Any]:
    """Descriptive statistics for a pi_pred_eff array."""
    n = len(pi)
    if n == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": n,
        "mean": float(np.mean(pi)),
        "median": float(np.median(pi)),
        "std": float(np.std(pi, ddof=1)) if n > 1 else 0.0,
        "p10": float(np.percentile(pi, 10)),
        "p25": float(np.percentile(pi, 25)),
        "p75": float(np.percentile(pi, 75)),
        "p90": float(np.percentile(pi, 90)),
    }


def test1(records: dict, meta: dict) -> dict:
    """Test 1: pi_pred distribution comparison across 4 groups."""
    pred_err = records["pred_err"]
    pi = records["pi_pred_eff"]
    is_march = records["is_march"]

    mask_hmm_exp = pred_err <= 10.0
    mask_hmm_unexp = pred_err > 20.0
    # Clean march: structurally defined (march flag), not by pred_err.
    # In 3-pass design, Expected and Unexpected share the same pre-probe
    # state, so pi_pred is identical. We report one distribution.
    mask_march = is_march

    groups = {
        "hmm_expected": pi_distribution_stats(pi[mask_hmm_exp], "Normal HMM Expected (pred_err≤10°)"),
        "hmm_unexpected": pi_distribution_stats(pi[mask_hmm_unexp], "Normal HMM Unexpected (pred_err>20°)"),
        "march_expected": pi_distribution_stats(pi[mask_march], "Clean march Expected"),
        "march_unexpected": pi_distribution_stats(pi[mask_march], "Clean march Unexpected (= Expected by construction)"),
    }

    # KS test: HMM Expected vs HMM Unexpected
    ks_hmm = ks_2sample(pi[mask_hmm_exp], pi[mask_hmm_unexp])
    # KS test: HMM Expected vs Clean march
    ks_hmm_march = ks_2sample(pi[mask_hmm_exp], pi[mask_march])

    return {
        "groups": groups,
        "ks_hmm_exp_vs_unexp": ks_hmm,
        "ks_hmm_exp_vs_march": ks_hmm_march,
        "note": ("Clean march Expected and Unexpected have identical pi_pred "
                 "because they share the same pre-probe network state in the "
                 "3-pass design (stim diverges only at probe ON window)."),
        # Store raw arrays for plotting (not in JSON)
        "_pi_hmm_exp": pi[mask_hmm_exp],
        "_pi_hmm_unexp": pi[mask_hmm_unexp],
        "_pi_march": pi[mask_march],
    }


# ---------------------------------------------------------------------------
# Test 2: quartile-binned Expected vs Unexpected delta
# ---------------------------------------------------------------------------

def bucket_stats(records: dict, mask: np.ndarray, center_idx: int,
                 label: str) -> dict[str, Any]:
    """Compute peak@true, total, decoding accuracy for a subset."""
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "n": 0, "peak_at_true": None,
                "total": None, "dec_acc": None, "dec_ci95": [None, None],
                "mean_pi": None, "mean_pred_err": None, "mean_run_length": None}

    rolled = records["r_l23_rolled"][mask]
    peak = float(rolled[:, center_idx].mean())
    total = float(rolled.sum(axis=1).mean())
    correct = (records["decoder_top1"][mask] == records["true_ch"][mask]).astype(np.float64)
    acc = float(correct.mean())
    lo, hi = bootstrap_acc_ci(correct)
    mean_pi = float(records["pi_pred_eff"][mask].mean())
    mean_pe = float(records["pred_err"][mask].mean())
    mean_rl = float(records["run_length"][mask].mean())
    return {
        "label": label, "n": n, "peak_at_true": peak, "total": total,
        "dec_acc": acc, "dec_ci95": [lo, hi],
        "mean_pi": mean_pi, "mean_pred_err": mean_pe, "mean_run_length": mean_rl,
    }


def test2(records: dict, meta: dict) -> dict:
    """Test 2: quartile-binned Expected vs Unexpected delta."""
    pred_err = records["pred_err"]
    pi = records["pi_pred_eff"]
    center_idx = meta["center_idx"]

    # Compute quartile thresholds on POOLED (Exp + Unexp) pi distribution
    mask_exp = pred_err <= 10.0
    mask_unexp = pred_err > 20.0
    pooled_pi = pi[mask_exp | mask_unexp]
    q_edges = [float(np.percentile(pooled_pi, p)) for p in [0, 25, 50, 75, 100]]

    quartiles = []
    for qi in range(4):
        lo_pct = qi * 25
        hi_pct = (qi + 1) * 25
        q_lo = q_edges[qi]
        q_hi = q_edges[qi + 1]

        if qi < 3:
            q_mask = (pi >= q_lo) & (pi < q_hi)
        else:
            q_mask = (pi >= q_lo) & (pi <= q_hi)

        exp_mask = mask_exp & q_mask
        unexp_mask = mask_unexp & q_mask

        exp_stats = bucket_stats(records, exp_mask, center_idx,
                                 f"Q{qi+1} Expected")
        unexp_stats = bucket_stats(records, unexp_mask, center_idx,
                                   f"Q{qi+1} Unexpected")

        delta_peak = None
        delta_dec = None
        if exp_stats["peak_at_true"] is not None and unexp_stats["peak_at_true"] is not None:
            delta_peak = exp_stats["peak_at_true"] - unexp_stats["peak_at_true"]
        if exp_stats["dec_acc"] is not None and unexp_stats["dec_acc"] is not None:
            delta_dec = exp_stats["dec_acc"] - unexp_stats["dec_acc"]

        quartiles.append({
            "quartile": f"Q{qi+1}",
            "pi_range": [q_lo, q_hi],
            "pct_range": [lo_pct, hi_pct],
            "expected": exp_stats,
            "unexpected": unexp_stats,
            "delta_peak": delta_peak,
            "delta_dec": delta_dec,
        })

    return {"quartile_edges": q_edges, "quartiles": quartiles}


# ---------------------------------------------------------------------------
# Test 3: nearest-neighbor pi_pred matched pairs
# ---------------------------------------------------------------------------

def test3(records: dict, meta: dict) -> dict:
    """Test 3: nearest-neighbor 1-to-1 pi_pred matching (without replacement)."""
    pred_err = records["pred_err"]
    pi = records["pi_pred_eff"]
    center_idx = meta["center_idx"]

    exp_idx = np.where(pred_err <= 10.0)[0]
    unexp_idx = np.where(pred_err > 20.0)[0]

    # Smaller set drives matching
    if len(exp_idx) <= len(unexp_idx):
        driver_idx = exp_idx
        pool_idx = unexp_idx.copy()
        driver_is_exp = True
    else:
        driver_idx = unexp_idx
        pool_idx = exp_idx.copy()
        driver_is_exp = False

    pi_driver = pi[driver_idx]
    pi_pool = pi[pool_idx]

    # Greedy nearest-neighbor without replacement
    matched_driver = []
    matched_pool = []
    available = np.ones(len(pool_idx), dtype=bool)

    # Sort driver by pi to improve matching quality
    sort_order = np.argsort(pi_driver)

    for di in sort_order:
        if not available.any():
            break
        avail_mask = available
        avail_pi = pi_pool[avail_mask]
        avail_positions = np.where(avail_mask)[0]
        diffs = np.abs(avail_pi - pi_driver[di])
        best_local = np.argmin(diffs)
        best_pool = avail_positions[best_local]
        matched_driver.append(di)
        matched_pool.append(best_pool)
        available[best_pool] = False

    matched_driver = np.array(matched_driver)
    matched_pool = np.array(matched_pool)
    n_pairs = len(matched_driver)

    if driver_is_exp:
        exp_global = driver_idx[matched_driver]
        unexp_global = pool_idx[matched_pool]
    else:
        unexp_global = driver_idx[matched_driver]
        exp_global = pool_idx[matched_pool]

    # Matching quality
    pi_exp_matched = pi[exp_global]
    pi_unexp_matched = pi[unexp_global]
    mean_abs_diff = float(np.abs(pi_exp_matched - pi_unexp_matched).mean())
    max_abs_diff = float(np.abs(pi_exp_matched - pi_unexp_matched).max())

    # Metrics on matched subset
    rolled = records["r_l23_rolled"]
    rl = records["run_length"]

    exp_peak = float(rolled[exp_global, center_idx].mean())
    unexp_peak = float(rolled[unexp_global, center_idx].mean())
    exp_total = float(rolled[exp_global].sum(axis=1).mean())
    unexp_total = float(rolled[unexp_global].sum(axis=1).mean())

    exp_correct = (records["decoder_top1"][exp_global] == records["true_ch"][exp_global]).astype(np.float64)
    unexp_correct = (records["decoder_top1"][unexp_global] == records["true_ch"][unexp_global]).astype(np.float64)
    exp_acc = float(exp_correct.mean())
    unexp_acc = float(unexp_correct.mean())
    exp_ci = bootstrap_acc_ci(exp_correct)
    unexp_ci = bootstrap_acc_ci(unexp_correct)

    exp_rl_mean = float(rl[exp_global].mean())
    unexp_rl_mean = float(rl[unexp_global].mean())

    return {
        "n_pairs": n_pairs,
        "driver_is_exp": driver_is_exp,
        "matching_quality": {
            "mean_abs_pi_diff": mean_abs_diff,
            "max_abs_pi_diff": max_abs_diff,
            "mean_pi_exp": float(pi_exp_matched.mean()),
            "mean_pi_unexp": float(pi_unexp_matched.mean()),
        },
        "expected": {
            "peak_at_true": exp_peak,
            "total": exp_total,
            "dec_acc": exp_acc,
            "dec_ci95": list(exp_ci),
            "mean_run_length": exp_rl_mean,
        },
        "unexpected": {
            "peak_at_true": unexp_peak,
            "total": unexp_total,
            "dec_acc": unexp_acc,
            "dec_ci95": list(unexp_ci),
            "mean_run_length": unexp_rl_mean,
        },
        "delta_peak": exp_peak - unexp_peak,
        "delta_total": exp_total - unexp_total,
        "delta_dec": exp_acc - unexp_acc,
    }


# ---------------------------------------------------------------------------
# Bonus: run length analysis
# ---------------------------------------------------------------------------

def bonus_run_length(records: dict, meta: dict, test2_result: dict) -> dict:
    """Run length statistics per group and per quartile."""
    pred_err = records["pred_err"]
    rl = records["run_length"]

    exp_mask = pred_err <= 10.0
    unexp_mask = pred_err > 20.0

    overall = {
        "expected": {
            "mean": float(rl[exp_mask].mean()),
            "median": float(np.median(rl[exp_mask])),
            "std": float(rl[exp_mask].std(ddof=1)),
        },
        "unexpected": {
            "mean": float(rl[unexp_mask].mean()),
            "median": float(np.median(rl[unexp_mask])),
            "std": float(rl[unexp_mask].std(ddof=1)),
        },
    }

    # Per quartile (already computed in test2_result)
    per_quartile = []
    for q in test2_result["quartiles"]:
        per_quartile.append({
            "quartile": q["quartile"],
            "expected_run_length": q["expected"]["mean_run_length"],
            "unexpected_run_length": q["unexpected"]["mean_run_length"],
        })

    return {"overall": overall, "per_quartile": per_quartile}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure(test1_result: dict, test2_result: dict, test3_result: dict,
                fig_path: str, label: str = "R1+R2") -> None:
    """4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: pi_pred histograms (Test 1) ---
    ax = axes[0, 0]
    pi_hmm_exp = test1_result["_pi_hmm_exp"]
    pi_hmm_unexp = test1_result["_pi_hmm_unexp"]
    pi_march = test1_result["_pi_march"]

    bins = np.linspace(0, 1.0, 50)
    ax.hist(pi_hmm_exp, bins=bins, alpha=0.5, density=True, color="#1f77b4",
            label=f"HMM Expected (n={len(pi_hmm_exp)})")
    ax.hist(pi_hmm_unexp, bins=bins, alpha=0.5, density=True, color="#d62728",
            label=f"HMM Unexpected (n={len(pi_hmm_unexp)})")
    ax.hist(pi_march, bins=bins, alpha=0.4, density=True, color="#2ca02c",
            edgecolor="#2ca02c", linewidth=1.5, histtype="step",
            label=f"Clean march (n={len(pi_march)})")
    ax.set_xlabel("pi_pred_eff")
    ax.set_ylabel("density")
    ax.set_title("A. pi_pred distributions (Test 1)", fontweight="bold")
    ax.legend(fontsize=8)

    # Annotate KS stats
    ks_he = test1_result["ks_hmm_exp_vs_unexp"]
    ks_hm = test1_result["ks_hmm_exp_vs_march"]
    ax.text(0.98, 0.85, f"KS HMM-Exp vs Unexp: D={ks_he.get('D', 'n/a'):.3f}",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="0.3")
    ax.text(0.98, 0.78, f"KS HMM-Exp vs March: D={ks_hm.get('D', 'n/a'):.3f}",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="0.3")

    # --- Panel B: Δ peak vs quartile (Test 2) ---
    ax = axes[0, 1]
    labels = [q["quartile"] for q in test2_result["quartiles"]]
    deltas = [q["delta_peak"] for q in test2_result["quartiles"]]
    colors = ["#d62728" if d is not None and d < 0 else "#2ca02c"
              for d in deltas]
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, [d if d is not None else 0 for d in deltas],
                  color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ peak @ true (Exp − Unexp)")
    ax.set_title("B. Δ peak by pi quartile (Test 2)", fontweight="bold")
    # Annotate n counts
    for i, q in enumerate(test2_result["quartiles"]):
        ne = q["expected"]["n"]
        nu = q["unexpected"]["n"]
        ax.text(i, 0, f"n={ne}/{nu}", ha="center", va="bottom" if (deltas[i] or 0) < 0 else "top",
                fontsize=7, color="0.4")

    # --- Panel C: Δ decoding acc vs quartile (Test 2) ---
    ax = axes[1, 0]
    dec_deltas = [q["delta_dec"] for q in test2_result["quartiles"]]
    colors_dec = ["#d62728" if d is not None and d < 0 else "#2ca02c"
                  for d in dec_deltas]
    bars = ax.bar(x_pos, [d if d is not None else 0 for d in dec_deltas],
                  color=colors_dec, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ decoding acc (Exp − Unexp)")
    ax.set_title("C. Δ decoding by pi quartile (Test 2)", fontweight="bold")

    # --- Panel D: matched-pair comparison (Test 3) ---
    ax = axes[1, 1]
    t3 = test3_result
    x_bar = np.array([0, 1, 3, 4])
    vals = [
        t3["expected"]["peak_at_true"],
        t3["unexpected"]["peak_at_true"],
        t3["expected"]["dec_acc"],
        t3["unexpected"]["dec_acc"],
    ]
    bar_colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]
    bar_labels = ["Exp peak", "Unexp peak", "Exp dec", "Unexp dec"]
    ax.bar(x_bar, vals, color=bar_colors, alpha=0.8, edgecolor="black",
           linewidth=0.5)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title(f"D. Matched pairs (Test 3, n={t3['n_pairs']})",
                 fontweight="bold")
    # Annotate deltas
    ax.text(0.5, max(vals[0], vals[1]) * 1.05,
            f"Δ={t3['delta_peak']:.4f}", ha="center", fontsize=8, color="0.3")
    ax.text(3.5, max(vals[2], vals[3]) * 1.05,
            f"Δ={t3['delta_dec']:.3f}", ha="center", fontsize=8, color="0.3")
    # Matching quality annotation
    mq = t3["matching_quality"]
    ax.text(0.98, 0.05,
            f"mean|Δpi|={mq['mean_abs_pi_diff']:.4f}\nmax|Δpi|={mq['max_abs_pi_diff']:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="0.4")

    fig.suptitle(f"V2 confidence (pi_pred_eff) dissection — {label}",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {fig_path}")


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_test1(t1: dict) -> None:
    """Print Test 1 table."""
    print("\n" + "=" * 80)
    print("TEST 1: pi_pred_eff distribution comparison")
    print("=" * 80)
    headers = ["group", "n", "mean", "median", "SD", "p10", "p25", "p75", "p90"]
    col_w = [40, 8, 8, 8, 8, 8, 8, 8, 8]
    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_w))
    for key in ["hmm_expected", "hmm_unexpected", "march_expected", "march_unexpected"]:
        g = t1["groups"][key]
        if g["n"] == 0:
            print(fmt_row([g["label"], 0] + ["n/a"] * 7))
        else:
            print(fmt_row([
                g["label"], g["n"],
                f"{g['mean']:.4f}", f"{g['median']:.4f}", f"{g['std']:.4f}",
                f"{g['p10']:.4f}", f"{g['p25']:.4f}", f"{g['p75']:.4f}", f"{g['p90']:.4f}",
            ]))
    print(f"\nKS HMM-Exp vs HMM-Unexp: D={t1['ks_hmm_exp_vs_unexp'].get('D')}, p={t1['ks_hmm_exp_vs_unexp'].get('p')}")
    print(f"KS HMM-Exp vs March: D={t1['ks_hmm_exp_vs_march'].get('D')}, p={t1['ks_hmm_exp_vs_march'].get('p')}")
    print(f"Note: {t1['note']}")


def print_test2(t2: dict) -> None:
    """Print Test 2 table."""
    print("\n" + "=" * 80)
    print("TEST 2: quartile-binned Expected vs Unexpected")
    print("=" * 80)
    headers = ["pi Q", "n_exp", "n_unexp", "peak_exp", "peak_unexp", "Δpeak",
               "dec_exp", "dec_unexp", "Δdec", "rl_exp", "rl_unexp"]
    col_w = [6, 7, 9, 10, 12, 10, 9, 11, 10, 8, 10]
    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_w))
    for q in t2["quartiles"]:
        e = q["expected"]
        u = q["unexpected"]
        dp = f"{q['delta_peak']:.4f}" if q['delta_peak'] is not None else "n/a"
        dd = f"{q['delta_dec']:.4f}" if q['delta_dec'] is not None else "n/a"
        print(fmt_row([
            q["quartile"],
            e["n"], u["n"],
            f"{e['peak_at_true']:.4f}" if e['peak_at_true'] is not None else "n/a",
            f"{u['peak_at_true']:.4f}" if u['peak_at_true'] is not None else "n/a",
            dp,
            f"{e['dec_acc']:.4f}" if e['dec_acc'] is not None else "n/a",
            f"{u['dec_acc']:.4f}" if u['dec_acc'] is not None else "n/a",
            dd,
            f"{e['mean_run_length']:.2f}" if e['mean_run_length'] is not None else "n/a",
            f"{u['mean_run_length']:.2f}" if u['mean_run_length'] is not None else "n/a",
        ]))
    print(f"\nQuartile edges: {[f'{e:.4f}' for e in t2['quartile_edges']]}")


def print_test3(t3: dict) -> None:
    """Print Test 3 table."""
    print("\n" + "=" * 80)
    print("TEST 3: nearest-neighbor pi_pred matched pairs")
    print("=" * 80)
    mq = t3["matching_quality"]
    print(f"n_pairs = {t3['n_pairs']}  (driver_is_exp = {t3['driver_is_exp']})")
    print(f"Matching quality: mean|Δpi| = {mq['mean_abs_pi_diff']:.6f}, "
          f"max|Δpi| = {mq['max_abs_pi_diff']:.6f}")
    print(f"  mean pi_exp = {mq['mean_pi_exp']:.4f}, mean pi_unexp = {mq['mean_pi_unexp']:.4f}")
    print()
    headers = ["group", "peak@true", "total", "dec_acc", "dec_ci95", "run_length"]
    col_w = [12, 10, 10, 10, 22, 12]
    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_w))
    for key in ["expected", "unexpected"]:
        d = t3[key]
        ci = d.get("dec_ci95", [None, None])
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a"
        print(fmt_row([
            key.capitalize(),
            f"{d['peak_at_true']:.4f}", f"{d['total']:.4f}",
            f"{d['dec_acc']:.4f}", ci_str,
            f"{d['mean_run_length']:.2f}",
        ]))
    print(f"\nΔ peak (Exp − Unexp) = {t3['delta_peak']:.4f}")
    print(f"Δ total (Exp − Unexp) = {t3['delta_total']:.4f}")
    print(f"Δ dec   (Exp − Unexp) = {t3['delta_dec']:.4f}")


def print_bonus(bonus: dict) -> None:
    """Print bonus run-length analysis."""
    print("\n" + "=" * 80)
    print("BONUS: preceding aligned run length")
    print("=" * 80)
    o = bonus["overall"]
    print(f"Expected:   mean={o['expected']['mean']:.2f}  median={o['expected']['median']:.1f}  "
          f"SD={o['expected']['std']:.2f}")
    print(f"Unexpected: mean={o['unexpected']['mean']:.2f}  median={o['unexpected']['median']:.1f}  "
          f"SD={o['unexpected']['std']:.2f}")
    print("\nPer quartile:")
    for q in bonus["per_quartile"]:
        erl = q["expected_run_length"]
        url = q["unexpected_run_length"]
        erl_s = f"{erl:.2f}" if erl is not None else "n/a"
        url_s = f"{url:.2f}" if url is not None else "n/a"
        print(f"  {q['quartile']}: exp_rl={erl_s}  unexp_rl={url_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-fig", default="docs/figures/v2_confidence_dissection_r1_2.png")
    p.add_argument("--output-json", default="results/v2_confidence_dissection_r1_2.json")
    p.add_argument("--label", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[setup] checkpoint={args.checkpoint}")
    print(f"[setup] config={args.config}")
    print(f"[setup] device={device}  n_batches={args.n_batches}  seed={args.rng_seed}")

    # Collect data
    records, meta = collect_hmm_pool(args, device)
    print(f"[collect] n_records={meta['n_records']}  "
          f"ambiguous_excluded={meta['n_ambiguous_excluded']}/{meta['n_total_pres']}")
    print(f"[collect] n_march_qualifying={int(records['is_march'].sum())}")

    # Run tests
    t1 = test1(records, meta)
    t2 = test2(records, meta)
    t3 = test3(records, meta)
    bonus = bonus_run_length(records, meta, t2)

    # Console output
    print_test1(t1)
    print_test2(t2)
    print_test3(t3)
    print_bonus(bonus)

    # Plot
    plot_figure(t1, t2, t3, args.output_fig,
                label=args.label or os.path.basename(args.checkpoint))

    # JSON output (strip numpy arrays)
    t1_json = {k: v for k, v in t1.items() if not k.startswith("_")}
    result = {
        "label": args.label or os.path.basename(args.checkpoint),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "meta": meta,
        "test1": t1_json,
        "test2": t2,
        "test3": t3,
        "bonus_run_length": bonus,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] wrote {args.output_json}")


if __name__ == "__main__":
    main()
