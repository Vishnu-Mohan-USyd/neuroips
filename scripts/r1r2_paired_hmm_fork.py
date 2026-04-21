#!/usr/bin/env python3
"""Task #20 — paired-state HMM fork × 3 branches × 3 conditions headline figure.

For each of 3 conditions (Focused+native cue, Routine+native cue, Focused+neutral cue)
generate 1000 native HMM sequences (seq_length=25, seed 42), then run THREE parallel
probe forwards from a shared pre-probe context (presentations 0..23):
  - ex   : probe stim = HMM natural continuation at presentation 24
  - unex : probe stim = HMM continuation rotated by exactly +90° (= +18 ch shift on the
           36-channel ring; matches P3P convention)
  - omi  : probe stim zeroed

The cue and task_state are IDENTICAL across the 3 branches per trial — only the probe
stim differs — so pre-probe (presentations 0..23) activity is bit-identical and the
"three forwards over the full sequence" trick is equivalent to forking from the saved
NetworkState at end-of-presentation-23.

Per branch per condition:
  - readout window: r_l23 mean over [probe_onset+9 : probe_onset+11]
  - Decoder C: argmax over checkpoints/decoder_c.pt (frozen Linear(36,36))
  - Decoder B: 5-fold CV nearest-centroid on r_probe vs probe true_ch (seed 42)
  - Decoder-free metrics on per-trial re-centered tuning curve:
      * peak at probe true_ch (for omi: use ex's true_ch as the "expected" reference)
      * net L2/3 (sum across 36 channels)
      * FWHM (linear-interp at half-max of max-normalised re-centered curve)
  - Full re-centered tuning curve: 36-ch mean ± per-channel SEM

Outputs:
  - results/r1r2_paired_hmm_fork.json
  - docs/figures/r1r2_paired_hmm_fork_headline.png
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence
from src.analysis.decoding import nearest_centroid_decode

from matched_quality_sim import roll_to_center            # noqa: E402
from plot_tuning_ring_extended import fwhm_of_curve       # noqa: E402


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N_TRIALS_DEFAULT = 1000
SEQ_LENGTH = 25
SEED_DEFAULT = 42
READOUT_WIN = (9, 11)
N_FOLDS_DEC_B = 5

CKPT_PATH_DEFAULT = "results/simple_dual/emergent_seed42/checkpoint.pt"
DECODER_C_PATH_DEFAULT = "checkpoints/decoder_c.pt"
CONFIG_PATH_DEFAULT = "config/sweep/sweep_rescue_1_2.yaml"
OUT_JSON_DEFAULT = "results/r1r2_paired_hmm_fork.json"
OUT_FIG_DEFAULT = "docs/figures/r1r2_paired_hmm_fork_headline.png"

CONDITIONS = [
    ("C1_focused_native",       "Focused + HMM cue",   [1.0, 0.0], False),
    ("C2_routine_native",       "Routine + HMM cue",   [0.0, 1.0], False),
    ("C3_focused_neutralcue",   "Focused + neutral",   [1.0, 0.0], True),
]
BRANCHES = ["ex", "unex", "omi"]


# -----------------------------------------------------------------------------
# Per-trial signed-offset re-centering (no march-direction sign-flip — task #19's
# additional flip was march-specific; here we just re-center on probe true_ch).
# -----------------------------------------------------------------------------

def per_trial_recenter(r_probe: np.ndarray, true_ch: np.ndarray, n_ori: int) -> np.ndarray:
    """Roll each trial's r_probe so true_ch lands at center_idx = n_ori // 2."""
    return roll_to_center(r_probe.astype(np.float64),
                          true_ch.astype(np.int64),
                          center_idx=n_ori // 2)


def per_trial_peak_net_fwhm(rolled: np.ndarray, n_ori: int, step_deg: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial: peak-at-center (offset 0), net (sum across 36 ch), FWHM (deg)."""
    center = n_ori // 2
    peak = rolled[:, center].astype(np.float64)
    net = rolled.sum(axis=1).astype(np.float64)
    fwhm = np.array([fwhm_of_curve(rolled[i], step_deg) for i in range(rolled.shape[0])],
                    dtype=np.float64)
    return peak, net, fwhm


def cv_nearest_centroid_per_fold(patterns_t: torch.Tensor, labels_t: torch.Tensor,
                                 n_folds: int = 5, seed: int = 42) -> list[float]:
    """5-fold CV nearest-centroid; returns per-fold accuracies (length n_folds)."""
    gen = torch.Generator().manual_seed(seed)
    n = patterns_t.shape[0]
    perm = torch.randperm(n, generator=gen)
    fold_size = n // n_folds
    accs: list[float] = []
    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = torch.cat([perm[:fold * fold_size], perm[(fold + 1) * fold_size:]])
        acc = nearest_centroid_decode(
            patterns_t[train_idx], labels_t[train_idx],
            patterns_t[test_idx], labels_t[test_idx],
        )
        accs.append(float(acc))
    return accs


def mean_sem(arr: np.ndarray) -> tuple[float, float]:
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    s = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, s


def nan_mean_sem(arr: np.ndarray) -> tuple[float, float, int]:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std(ddof=1) / np.sqrt(len(valid))) \
        if len(valid) > 1 else 0.0, int(len(valid))


# -----------------------------------------------------------------------------
# Per-condition forward sweep
# -----------------------------------------------------------------------------

def run_condition(cond_id: str, cond_label: str, ts_value: list[float], zero_cue: bool,
                  metadata_base, model_cfg, train_cfg, stim_cfg,
                  net: LaminarV1V2Network, decoder_c: nn.Linear,
                  device: torch.device, n_trials: int, seed: int) -> dict[str, Any]:
    """One condition: build temporal seqs (with task_state and cue overrides), then
    run 3 probe-branch forwards."""
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    steps_on = int(train_cfg.steps_on)
    steps_isi = int(train_cfg.steps_isi)
    steps_per = steps_on + steps_isi
    probe_idx = SEQ_LENGTH - 1
    probe_onset = probe_idx * steps_per
    win_lo = probe_onset + READOUT_WIN[0]
    win_hi = probe_onset + READOUT_WIN[1]

    # Deep-copy metadata so each condition gets a clean slate (task_state and cue
    # overrides mutate fields)
    md = copy.deepcopy(metadata_base)

    # --- Override task_state across ALL presentations ---
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = float(ts_value[0])
    new_ts[..., 1] = float(ts_value[1])
    md.task_states = new_ts

    # --- Cue override (C3 only): zero ALL ISI cue bumps ---
    if zero_cue:
        md.cues = torch.zeros_like(md.cues)

    # Build temporal sequences
    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
        md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device)
    cue_seq = cue_seq.to(device)
    ts_seq = ts_seq.to(device)
    assert stim_seq.shape[0] == n_trials

    # Probe true_ch (ex). For unex it's ex_ch + 18 mod 36 (= +90° on 180° ring)
    true_ori_ex = md.orientations[:, probe_idx].to(device)              # [B] deg
    true_ch_ex = (true_ori_ex / step_deg).round().long() % n_ori        # [B]
    true_ch_unex = (true_ch_ex + n_ori // 2) % n_ori                    # [B] = +18 shift

    # Build branch stims: only the probe slot (ON window, steps_on timesteps) differs
    stim_ex = stim_seq                                                  # natural HMM
    stim_unex = stim_seq.clone()
    # +90° rotation on 180° period == +n_ori/2 channel shift on the population code
    # Roll along the channel axis = orientation shift; works for unimodal AND ambig
    stim_unex[:, probe_onset:probe_onset + steps_on, :] = torch.roll(
        stim_seq[:, probe_onset:probe_onset + steps_on, :],
        shifts=n_ori // 2, dims=-1)
    stim_omi = stim_seq.clone()
    stim_omi[:, probe_onset:probe_onset + steps_on, :] = 0.0

    # Forward 3 branches. Pre-probe (t < probe_onset) is bit-identical across
    # branches by construction → asserted below.
    print(f"[{cond_id}] forward ex...", flush=True)
    with torch.no_grad():
        packed_ex = net.pack_inputs(stim_ex, cue_seq, ts_seq)
        r_l23_ex, _, _ = net.forward(packed_ex)
        print(f"[{cond_id}] forward unex...", flush=True)
        packed_unex = net.pack_inputs(stim_unex, cue_seq, ts_seq)
        r_l23_unex, _, _ = net.forward(packed_unex)
        print(f"[{cond_id}] forward omi...", flush=True)
        packed_omi = net.pack_inputs(stim_omi, cue_seq, ts_seq)
        r_l23_omi, _, _ = net.forward(packed_omi)

    # Pre-probe smoke check
    pre_max_ex_unex = float((r_l23_ex[:, :probe_onset, :] - r_l23_unex[:, :probe_onset, :]).abs().max())
    pre_max_ex_omi = float((r_l23_ex[:, :probe_onset, :] - r_l23_omi[:, :probe_onset, :]).abs().max())
    print(f"[{cond_id}] pre-probe max|ex-unex|={pre_max_ex_unex:.2e}  "
          f"max|ex-omi|={pre_max_ex_omi:.2e}  (should be ~0)", flush=True)

    # Readout window
    r_probe_ex = r_l23_ex[:, win_lo:win_hi, :].mean(dim=1)              # [B, n_ori]
    r_probe_unex = r_l23_unex[:, win_lo:win_hi, :].mean(dim=1)
    r_probe_omi = r_l23_omi[:, win_lo:win_hi, :].mean(dim=1)

    # ---------- Decoder C (frozen) ----------
    pred_c_ex = decoder_c(r_probe_ex).argmax(dim=-1)
    pred_c_unex = decoder_c(r_probe_unex).argmax(dim=-1)
    pred_c_omi = decoder_c(r_probe_omi).argmax(dim=-1)
    decC_correct_ex = (pred_c_ex == true_ch_ex).cpu().numpy().astype(np.float64)
    decC_correct_unex = (pred_c_unex == true_ch_unex).cpu().numpy().astype(np.float64)
    # For omi, "correct" means decoder still predicts the EX channel (the "expected" ori)
    decC_correct_omi_vs_ex = (pred_c_omi == true_ch_ex).cpu().numpy().astype(np.float64)

    # ---------- Decoder B (5-fold CV nearest centroid, fit within slice) ----------
    # Fit on (r_probe_ex, true_ch_ex) and (r_probe_unex, true_ch_unex).
    # Don't fit on omi (no true_ch).
    rB_ex = r_probe_ex.detach().cpu()
    rB_unex = r_probe_unex.detach().cpu()
    yB_ex = true_ch_ex.detach().cpu()
    yB_unex = true_ch_unex.detach().cpu()
    decB_folds_ex = cv_nearest_centroid_per_fold(rB_ex, yB_ex, n_folds=N_FOLDS_DEC_B, seed=seed)
    decB_folds_unex = cv_nearest_centroid_per_fold(rB_unex, yB_unex, n_folds=N_FOLDS_DEC_B, seed=seed)

    # ---------- Decoder-free metrics (re-centered) ----------
    r_probe_ex_np = r_probe_ex.detach().cpu().numpy().astype(np.float64)
    r_probe_unex_np = r_probe_unex.detach().cpu().numpy().astype(np.float64)
    r_probe_omi_np = r_probe_omi.detach().cpu().numpy().astype(np.float64)
    true_ch_ex_np = true_ch_ex.detach().cpu().numpy().astype(np.int64)
    true_ch_unex_np = true_ch_unex.detach().cpu().numpy().astype(np.int64)

    rolled_ex = per_trial_recenter(r_probe_ex_np, true_ch_ex_np, n_ori)
    rolled_unex = per_trial_recenter(r_probe_unex_np, true_ch_unex_np, n_ori)
    # Omi: re-center on EX's true_ch (the expected orientation reference)
    rolled_omi = per_trial_recenter(r_probe_omi_np, true_ch_ex_np, n_ori)

    peak_ex, net_ex, fwhm_ex = per_trial_peak_net_fwhm(rolled_ex, n_ori, step_deg)
    peak_unex, net_unex, fwhm_unex = per_trial_peak_net_fwhm(rolled_unex, n_ori, step_deg)
    peak_omi, net_omi, fwhm_omi = per_trial_peak_net_fwhm(rolled_omi, n_ori, step_deg)

    # ---------- Tuning curves (mean ± SEM per channel) ----------
    n = rolled_ex.shape[0]
    tc_mean_ex = rolled_ex.mean(axis=0)
    tc_sem_ex = rolled_ex.std(axis=0, ddof=1) / np.sqrt(n)
    tc_mean_unex = rolled_unex.mean(axis=0)
    tc_sem_unex = rolled_unex.std(axis=0, ddof=1) / np.sqrt(n)
    tc_mean_omi = rolled_omi.mean(axis=0)
    tc_sem_omi = rolled_omi.std(axis=0, ddof=1) / np.sqrt(n)

    # ---------- Pack per-branch summary ----------
    def _branch(decC_corr, decB_folds, peak, netv, fwhm, tc_m, tc_s):
        m_decC, s_decC = mean_sem(decC_corr)
        decB_arr = np.array(decB_folds) if decB_folds is not None else None
        m_decB = float(decB_arr.mean()) if decB_arr is not None else None
        s_decB = float(decB_arr.std(ddof=1) / np.sqrt(len(decB_arr))) \
            if decB_arr is not None and len(decB_arr) > 1 else None
        m_peak, s_peak = mean_sem(peak)
        m_net, s_net = mean_sem(netv)
        m_fwhm, s_fwhm, n_fwhm = nan_mean_sem(fwhm)
        return {
            "n_trials": int(n),
            "decC_acc_mean": m_decC,
            "decC_acc_sem": s_decC,
            "decB_acc_mean": m_decB,
            "decB_acc_sem_across_folds": s_decB,
            "decB_per_fold_acc": decB_folds,
            "peak_mean": m_peak,
            "peak_sem": s_peak,
            "net_mean": m_net,
            "net_sem": s_net,
            "fwhm_mean": m_fwhm,
            "fwhm_sem": s_fwhm,
            "fwhm_n_valid": n_fwhm,
            "tuning_curve_mean": tc_m.tolist(),
            "tuning_curve_sem": tc_s.tolist(),
        }

    out = {
        "id": cond_id,
        "label": cond_label,
        "task_state": list(ts_value),
        "cue_neutralized": bool(zero_cue),
        "pre_probe_max_abs_diff_ex_unex": pre_max_ex_unex,
        "pre_probe_max_abs_diff_ex_omi": pre_max_ex_omi,
        "branches": {
            "ex": _branch(decC_correct_ex, decB_folds_ex,
                          peak_ex, net_ex, fwhm_ex, tc_mean_ex, tc_sem_ex),
            "unex": _branch(decC_correct_unex, decB_folds_unex,
                            peak_unex, net_unex, fwhm_unex, tc_mean_unex, tc_sem_unex),
            "omi": _branch(decC_correct_omi_vs_ex, None,
                           peak_omi, net_omi, fwhm_omi, tc_mean_omi, tc_sem_omi),
        },
    }
    return out


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
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] decoder_c={args.decoder_c}", flush=True)
    print(f"[setup] n_trials={args.n_trials}  seed={args.seed}", flush=True)

    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    print(f"[cfg] n_orientations={n_ori}  period={period}  "
          f"steps_on={train_cfg.steps_on}  steps_isi={train_cfg.steps_isi}  "
          f"seq_length={SEQ_LENGTH}  cue_valid_fraction={stim_cfg.cue_valid_fraction}",
          flush=True)

    # Load network
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    print(f"[setup] network loaded (oracle_mode={net.oracle_mode}, "
          f"feedback_scale={float(net.feedback_scale.item()):.3f})", flush=True)

    # Decoder C
    dec_ckpt = torch.load(args.decoder_c, map_location=device, weights_only=False)
    decoder_c = nn.Linear(n_ori, n_ori).to(device)
    decoder_c.load_state_dict(dec_ckpt["state_dict"])
    decoder_c.eval()
    best_val = dec_ckpt.get("train_meta", {}).get("best_val_acc", float("nan"))
    print(f"[setup] Decoder C loaded (best_val_acc={best_val:.4f})", flush=True)

    # Generate ONE HMM batch with config-derived params (cue_valid_fraction = 0.75)
    gen = HMMSequenceGenerator(
        n_orientations=n_ori,
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
        # task_p_switch left at HMM default (we override task_state per condition)
    )
    rng = torch.Generator().manual_seed(args.seed)
    print(f"[hmm] generating {args.n_trials} trials × seq_length={SEQ_LENGTH}...",
          flush=True)
    metadata_base = gen.generate(args.n_trials, SEQ_LENGTH, generator=rng)
    print(f"[hmm] generated: orientations.shape={tuple(metadata_base.orientations.shape)}  "
          f"cues.shape={tuple(metadata_base.cues.shape)}  "
          f"task_states.shape={tuple(metadata_base.task_states.shape)}  "
          f"is_ambiguous.sum()={int(metadata_base.is_ambiguous.sum())}", flush=True)

    # Run all 3 conditions
    cond_results = []
    t0 = time.time()
    for cond_id, cond_label, ts_val, zero_cue in CONDITIONS:
        ts_init = time.time()
        print(f"\n=== Condition {cond_id} ({cond_label}) ===", flush=True)
        res = run_condition(cond_id, cond_label, ts_val, zero_cue,
                            metadata_base, model_cfg, train_cfg, stim_cfg,
                            net, decoder_c, device, args.n_trials, args.seed)
        cond_results.append(res)
        # Print summary table for this condition
        print(f"\n[{cond_id}] summary:")
        print(f"  {'branch':>6} {'decC_acc':>12} {'decB_acc':>12} {'peak':>8} {'net':>8} "
              f"{'FWHM°':>8}", flush=True)
        for br_id, br in res["branches"].items():
            decB = f"{br['decB_acc_mean']:.4f}" if br['decB_acc_mean'] is not None else "n/a"
            print(f"  {br_id:>6} {br['decC_acc_mean']:>10.4f}± {br['decC_acc_sem']:.4f} "
                  f" {decB}  "
                  f" {br['peak_mean']:>6.4f}  {br['net_mean']:>6.4f}  "
                  f" {br['fwhm_mean']:>6.2f}", flush=True)
        print(f"[{cond_id}] elapsed: {time.time() - ts_init:.1f}s", flush=True)
    print(f"\n[total] elapsed: {time.time() - t0:.1f}s", flush=True)

    # Pack JSON
    out = {
        "label": "Task #20: paired-state HMM fork × 3 branches × 3 conditions on R1+R2 (Decoder B + C + decoder-free)",
        "checkpoint": args.checkpoint,
        "decoder_c": args.decoder_c,
        "config": args.config,
        "design": {
            "n_trials_per_condition": args.n_trials,
            "seq_length": SEQ_LENGTH,
            "seed": args.seed,
            "n_orientations": n_ori,
            "period_deg": period,
            "step_deg": period / n_ori,
            "steps_on": int(train_cfg.steps_on),
            "steps_isi": int(train_cfg.steps_isi),
            "readout_window_steps": list(READOUT_WIN),
            "cue_valid_fraction": float(stim_cfg.cue_valid_fraction),
            "unex_rotation_deg": 90.0,
            "unex_channel_shift": n_ori // 2,
            "decoder_b_n_folds": N_FOLDS_DEC_B,
            "decoder_b_method": "nearest-centroid 5-fold CV (fit within slice)",
            "decoder_c_method": "frozen Linear(36,36) argmax top-1",
            "fwhm_method": "linear-interp at half-max of max-normalised re-centered curve",
            "omi_branch_recentered_on": "ex true_ch (expected orientation as reference)",
        },
        "conditions": cond_results,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[json] wrote {args.output_json}", flush=True)

    # ----------------------------- Plot -----------------------------
    fig = plt.figure(figsize=(15.0, 10.5))
    outer = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                     height_ratios=[1.0, 1.1])

    # Compute consistent y-limits across conditions
    all_tc = []
    for r in cond_results:
        for br in r["branches"].values():
            tc = np.array(br["tuning_curve_mean"])
            sem = np.array(br["tuning_curve_sem"])
            all_tc.append(tc + sem)
            all_tc.append(tc - sem)
    tc_min = min(arr.min() for arr in all_tc) - 0.02
    tc_max = max(arr.max() for arr in all_tc) + 0.02

    half = n_ori // 2
    offsets = np.arange(-half, half)                                # [-18..+17]
    keep_mask = (offsets >= -17) & (offsets <= 17)
    plot_offsets = offsets[keep_mask]                               # [-17..+17]

    branch_styles = {
        "ex":   dict(color="C0", marker="o", label="Expected"),
        "unex": dict(color="C1", marker="s", label="Unexpected"),
        "omi":  dict(color="C2", marker="^", label="Omission"),
    }

    # ---- Row 1: tuning curves overlay per condition ----
    for col, r in enumerate(cond_results):
        ax = fig.add_subplot(outer[0, col])
        for br_id in ["ex", "unex", "omi"]:
            br = r["branches"][br_id]
            tc = np.array(br["tuning_curve_mean"])
            sem = np.array(br["tuning_curve_sem"])
            tc_p = tc[keep_mask]
            sem_p = sem[keep_mask]
            st = branch_styles[br_id]
            ax.plot(plot_offsets, tc_p, "-", marker=st["marker"], color=st["color"],
                    lw=1.5, ms=3.5, label=st["label"])
            ax.fill_between(plot_offsets, tc_p - sem_p, tc_p + sem_p,
                            color=st["color"], alpha=0.18)
        ax.axvline(0.0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("Channel offset from probe true_ch")
        if col == 0:
            ax.set_ylabel("L2/3 readout (re-centered)")
        ax.set_title(f"{r['label']}\nn={r['branches']['ex']['n_trials']}", fontsize=11)
        ax.set_xticks(np.arange(-15, 16, 5))
        ax.set_ylim(tc_min, tc_max)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=9, loc="upper right")

    # ---- Row 2: 5 metric mini-bar-charts per condition (subgridspec) ----
    metric_specs = [
        ("decC_acc_mean", "decC_acc_sem",       "Decoder C acc",   None),
        ("decB_acc_mean", "decB_acc_sem_across_folds", "Decoder B acc (5-fold)", None),
        ("peak_mean",     "peak_sem",           "Peak @ true_ch",  "omi"),
        ("net_mean",      "net_sem",            "Net L2/3",        "omi"),
        ("fwhm_mean",     "fwhm_sem",           "FWHM (deg)",      "omi"),
    ]

    # Compute per-metric global y-limits across conditions for visual comparability
    metric_ylim = {}
    for m_key, _, _, _ in metric_specs:
        vals = []
        for r in cond_results:
            for br_id, br in r["branches"].items():
                v = br.get(m_key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
        if vals:
            lo = min(vals)
            hi = max(vals)
            pad = 0.10 * (hi - lo if hi > lo else max(abs(hi), 1.0))
            metric_ylim[m_key] = (max(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)

    for col, r in enumerate(cond_results):
        sub = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1, col],
                                      wspace=0.55)
        for mi, (m_key, sem_key, m_lab, ref_branch) in enumerate(metric_specs):
            ax = fig.add_subplot(sub[0, mi])
            br_ex = r["branches"]["ex"]
            br_unex = r["branches"]["unex"]
            v_ex = br_ex.get(m_key, None)
            v_unex = br_unex.get(m_key, None)
            s_ex = br_ex.get(sem_key, None) or 0.0
            s_unex = br_unex.get(sem_key, None) or 0.0
            if v_ex is None or v_unex is None or (isinstance(v_ex, float) and np.isnan(v_ex)):
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes,
                        fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(m_lab, fontsize=8)
                continue
            xs = np.arange(2)
            vals = [v_ex, v_unex]
            errs = [s_ex, s_unex]
            colors = [branch_styles["ex"]["color"], branch_styles["unex"]["color"]]
            bars = ax.bar(xs, vals, yerr=errs, color=colors, capsize=3,
                          edgecolor="black", linewidth=0.5)
            # Reference line for omi where applicable
            if ref_branch is not None:
                v_ref = r["branches"][ref_branch].get(m_key, None)
                if v_ref is not None and not (isinstance(v_ref, float) and np.isnan(v_ref)):
                    ax.axhline(v_ref, color=branch_styles["omi"]["color"],
                               lw=1.2, ls="--", label="omi")
            # Tight value labels above bars
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.3f}" if abs(v) < 1.0 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xs)
            ax.set_xticklabels(["ex", "unex"], fontsize=7)
            if m_key in metric_ylim:
                ax.set_ylim(metric_ylim[m_key])
            ax.set_title(m_lab, fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("R1+R2 paired-state HMM fork: ex vs unex vs omi across 3 task/cue conditions  "
                 f"(n={cond_results[0]['branches']['ex']['n_trials']} per cell, seed={args.seed})",
                 fontsize=12)
    out_dir = os.path.dirname(os.path.abspath(args.output_fig))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {args.output_fig}", flush=True)


if __name__ == "__main__":
    main()
