#!/usr/bin/env python3
"""Task #22 — paradigm × readout matrix on R1+R2.

Builds 4 paired-state HMM-fork conditions on R1+R2 + reads pre-existing Task #19 NEW
march-signed eval JSON for a 5th column. For each condition, computes:

  - Decoder C accuracy (frozen Linear(36,36) argmax)
  - Decoder B accuracy (5-fold CV nearest-centroid)
  - Decoder-free per-trial peak / net / FWHM (re-centered tuning curves)
  - S(d) = r_unex(d) − r_ex(d) profile (paired SEM)
  - Paired bootstrap 95% CIs on Δ(ex − unex) for peak/net/FWHM/DecB/DecC (n=1000, seed=42)

Conditions:
  - C1  Focused + native HMM cue       (task_state=[1,0], cue native)
  - C2  Routine + native HMM cue       (task_state=[0,1], cue native)
  - C3  Focused + neutral cue          (task_state=[1,0], cue zeroed)
  - C4  Routine + neutral cue          (task_state=[0,1], cue zeroed)  ← Richter-style primary

NEW assay panel: Task #19 results/eval_ex_vs_unex_decC_adjacent.json (march-signed).

Outputs (all NEW; Task #20 outputs are preserved):
  - results/r1r2_paradigm_readout.json
  - docs/figures/r1r2_paradigm_readout.png
  - logs/r1r2_paradigm_readout.log  (written by the runner shell, not this script)

Multi-seed: only seed 42 is available for R1+R2 (no sibling seed_dirs). Bootstrap is
trial-only (hierarchical seed+trial bootstrap is reduced to trial bootstrap with n_seeds=1
— flagged in JSON metadata.flags).
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
N_BOOT_DEFAULT = 1000

CKPT_PATH_DEFAULT = "results/simple_dual/emergent_seed42/checkpoint.pt"
DECODER_C_PATH_DEFAULT = "checkpoints/decoder_c.pt"
CONFIG_PATH_DEFAULT = "config/sweep/sweep_rescue_1_2.yaml"
NEW_ASSAY_JSON_DEFAULT = "results/eval_ex_vs_unex_decC_adjacent.json"
OUT_JSON_DEFAULT = "results/r1r2_paradigm_readout.json"
OUT_FIG_DEFAULT = "docs/figures/r1r2_paradigm_readout.png"

# Conditions as 4-tuples: (cond_id, cond_label, task_state[2], zero_cue)
CONDITIONS = [
    ("C1_focused_native",     "Focused + HMM cue",  [1.0, 0.0], False),
    ("C2_routine_native",     "Routine + HMM cue",  [0.0, 1.0], False),
    ("C3_focused_neutralcue", "Focused + neutral",  [1.0, 0.0], True),
    ("C4_routine_neutralcue", "Routine + neutral",  [0.0, 1.0], True),
]


# -----------------------------------------------------------------------------
# Per-trial helpers
# -----------------------------------------------------------------------------

def per_trial_recenter(r_probe: np.ndarray, true_ch: np.ndarray, n_ori: int) -> np.ndarray:
    """Roll each trial's r_probe so true_ch lands at center_idx = n_ori // 2.

    Args:
        r_probe: [n_trials, n_ori] float64.
        true_ch: [n_trials] int64 channel index ∈ [0, n_ori).
        n_ori:   ring length (36).
    Returns:
        rolled: [n_trials, n_ori] with rolled[i, n_ori//2] == r_probe[i, true_ch[i]].
    """
    return roll_to_center(r_probe.astype(np.float64),
                          true_ch.astype(np.int64),
                          center_idx=n_ori // 2)


def per_trial_peak_net_fwhm(rolled: np.ndarray, n_ori: int, step_deg: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial: peak-at-center (offset 0), net (sum across n_ori), FWHM (deg, NaN if no clear peak)."""
    center = n_ori // 2
    peak = rolled[:, center].astype(np.float64)
    net = rolled.sum(axis=1).astype(np.float64)
    fwhm = np.array([fwhm_of_curve(rolled[i], step_deg) for i in range(rolled.shape[0])],
                    dtype=np.float64)
    return peak, net, fwhm


def cv_nearest_centroid_per_trial(patterns: torch.Tensor, labels: torch.Tensor,
                                  n_folds: int = 5, seed: int = 42
                                  ) -> tuple[list[float], np.ndarray]:
    """5-fold CV nearest-centroid; returns (per_fold_acc list, per_trial_correct float64 array).

    per_trial_correct is 0.0/1.0 for every trial assigned to a test fold; trials that
    fall outside the n_folds*fold_size range (when n is not divisible by n_folds) are NaN
    and excluded from bootstrap.

    Args:
        patterns: [n, d] float tensor.
        labels:   [n] int tensor (channel index).
        n_folds:  k.
        seed:     PyTorch generator seed for trial permutation (deterministic).
    """
    gen = torch.Generator().manual_seed(seed)
    n = patterns.shape[0]
    perm = torch.randperm(n, generator=gen)
    fold_size = n // n_folds

    per_trial_correct = np.full(n, np.nan, dtype=np.float64)
    accs: list[float] = []
    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = torch.cat([perm[:fold * fold_size], perm[(fold + 1) * fold_size:]])
        train_X = patterns[train_idx]
        train_y = labels[train_idx]
        test_X = patterns[test_idx]
        test_y = labels[test_idx]

        # Compute centroids per class present in training set
        classes_present = train_y.unique()
        # Build [n_classes_present, d] centroid matrix for vectorised distance
        cent_list = []
        cent_labels = []
        for c in classes_present.tolist():
            mask = train_y == c
            if mask.any():
                cent_list.append(train_X[mask].mean(dim=0))
                cent_labels.append(c)
        centroids = torch.stack(cent_list, dim=0)            # [C, d]
        cent_labels_t = torch.tensor(cent_labels, dtype=test_y.dtype, device=test_X.device)

        # Squared distance from each test point to each centroid
        d2 = ((test_X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=-1)   # [n_test, C]
        nearest = d2.argmin(dim=-1)                                              # [n_test]
        preds = cent_labels_t[nearest]
        correct = (preds == test_y).float().cpu().numpy().astype(np.float64)

        accs.append(float(correct.mean()))
        # Map per-trial correctness back to original trial indices
        for j, ti in enumerate(test_idx.tolist()):
            per_trial_correct[ti] = correct[j]

    return accs, per_trial_correct


def mean_sem(arr: np.ndarray) -> tuple[float, float, int]:
    """Mean and SEM (ddof=1), ignoring NaN. Returns (mean, sem, n_used)."""
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(valid.mean())
    s = float(valid.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, s, n


def paired_bootstrap_diff(arr_ex: np.ndarray, arr_unex: np.ndarray,
                          n_resamples: int = 1000, seed: int = 42
                          ) -> dict[str, Any]:
    """Paired bootstrap 95% CI on Δ = mean(ex − unex). NaN-pairs dropped jointly.

    Returns dict with delta_mean, delta_ci_lo, delta_ci_hi, n_pairs_used, n_resamples,
    and the raw paired-trial mean/sem of ex and unex post-NaN-drop for sanity.
    """
    valid = ~(np.isnan(arr_ex) | np.isnan(arr_unex))
    ex = arr_ex[valid]
    unex = arr_unex[valid]
    n = len(ex)
    if n == 0:
        return dict(delta_mean=float("nan"), delta_ci_lo=float("nan"),
                    delta_ci_hi=float("nan"), n_pairs_used=0, n_resamples=int(n_resamples),
                    ex_mean=float("nan"), ex_sem=float("nan"),
                    unex_mean=float("nan"), unex_sem=float("nan"))
    diffs = ex - unex
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return dict(
        delta_mean=float(diffs.mean()),
        delta_ci_lo=float(lo),
        delta_ci_hi=float(hi),
        n_pairs_used=int(n),
        n_resamples=int(n_resamples),
        ex_mean=float(ex.mean()),
        ex_sem=float(ex.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        unex_mean=float(unex.mean()),
        unex_sem=float(unex.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
    )


def s_of_d_profile(rolled_ex: np.ndarray, rolled_unex: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S(d) = mean_t(rolled_unex[t, d] − rolled_ex[t, d]) per offset d.

    Args:
        rolled_ex:   [n_trials, n_ori] re-centered (true_ch → center).
        rolled_unex: [n_trials, n_ori] re-centered.
    Returns:
        s_mean: [n_ori] mean paired S(d).
        s_sem:  [n_ori] paired SEM (across trials).
        diffs:  [n_trials, n_ori] per-trial paired diffs (saved for downstream bootstrap).
    """
    diffs = rolled_unex.astype(np.float64) - rolled_ex.astype(np.float64)
    n = diffs.shape[0]
    s_mean = diffs.mean(axis=0)
    s_sem = diffs.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(s_mean)
    return s_mean, s_sem, diffs


# -----------------------------------------------------------------------------
# Per-condition forward sweep + per-trial extraction
# -----------------------------------------------------------------------------

def run_condition(cond_id: str, cond_label: str, ts_value: list[float], zero_cue: bool,
                  metadata_base, model_cfg, train_cfg, stim_cfg,
                  net: LaminarV1V2Network, decoder_c: nn.Linear,
                  device: torch.device, n_trials: int, seed: int, n_boot: int,
                  decoder_a: nn.Linear | None = None,
                  pertrial_dump_dir: str | None = None,
                  decoder_d_raw: nn.Linear | None = None,
                  decoder_d_shape: nn.Linear | None = None,
                  ) -> dict[str, Any]:
    """One condition: build temporal seqs + run 3 probe-branch forwards. Returns
    summary dict with per-trial arrays preserved for bootstrap + S(d) profile."""
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

    md = copy.deepcopy(metadata_base)
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = float(ts_value[0])
    new_ts[..., 1] = float(ts_value[1])
    md.task_states = new_ts
    if zero_cue:
        md.cues = torch.zeros_like(md.cues)

    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
        md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device)
    cue_seq = cue_seq.to(device)
    ts_seq = ts_seq.to(device)
    assert stim_seq.shape[0] == n_trials

    true_ori_ex = md.orientations[:, probe_idx].to(device)
    true_ch_ex = (true_ori_ex / step_deg).round().long() % n_ori
    true_ch_unex = (true_ch_ex + n_ori // 2) % n_ori

    stim_ex = stim_seq
    stim_unex = stim_seq.clone()
    stim_unex[:, probe_onset:probe_onset + steps_on, :] = torch.roll(
        stim_seq[:, probe_onset:probe_onset + steps_on, :],
        shifts=n_ori // 2, dims=-1)

    print(f"[{cond_id}] forward ex...", flush=True)
    with torch.no_grad():
        packed_ex = net.pack_inputs(stim_ex, cue_seq, ts_seq)
        r_l23_ex, _, _ = net.forward(packed_ex)
        print(f"[{cond_id}] forward unex...", flush=True)
        packed_unex = net.pack_inputs(stim_unex, cue_seq, ts_seq)
        r_l23_unex, _, _ = net.forward(packed_unex)

    pre_max_ex_unex = float((r_l23_ex[:, :probe_onset, :]
                             - r_l23_unex[:, :probe_onset, :]).abs().max())
    print(f"[{cond_id}] pre-probe max|ex-unex|={pre_max_ex_unex:.2e}  (should be ~0)",
          flush=True)

    r_probe_ex = r_l23_ex[:, win_lo:win_hi, :].mean(dim=1)
    r_probe_unex = r_l23_unex[:, win_lo:win_hi, :].mean(dim=1)

    # ---------- Decoder C (frozen, per-trial correctness) ----------
    pred_c_ex = decoder_c(r_probe_ex).argmax(dim=-1)
    pred_c_unex = decoder_c(r_probe_unex).argmax(dim=-1)
    decC_correct_ex = (pred_c_ex == true_ch_ex).cpu().numpy().astype(np.float64)
    decC_correct_unex = (pred_c_unex == true_ch_unex).cpu().numpy().astype(np.float64)

    # ---------- Decoder A (frozen orientation_decoder from network ckpt) ----------
    # Task #26 addition: apply Dec A on the same r_probe to enable cross-decoder matrix.
    if decoder_a is not None:
        with torch.no_grad():
            pred_a_ex = decoder_a(r_probe_ex).argmax(dim=-1)
            pred_a_unex = decoder_a(r_probe_unex).argmax(dim=-1)
        decA_correct_ex = (pred_a_ex == true_ch_ex).cpu().numpy().astype(np.float64)
        decA_correct_unex = (pred_a_unex == true_ch_unex).cpu().numpy().astype(np.float64)
    else:
        decA_correct_ex = None
        decA_correct_unex = None

    # ---------- Decoder D-raw and D-shape (Task #4 — neutral FB-off linear) ----------
    # D-raw: direct on r_l23. D-shape: r_l23 normalised by row sum.
    if decoder_d_raw is not None and decoder_d_shape is not None:
        with torch.no_grad():
            pred_draw_ex = decoder_d_raw(r_probe_ex).argmax(dim=-1)
            pred_draw_unex = decoder_d_raw(r_probe_unex).argmax(dim=-1)
            r_shape_ex = r_probe_ex / (r_probe_ex.sum(dim=1, keepdim=True) + 1e-8)
            r_shape_unex = r_probe_unex / (r_probe_unex.sum(dim=1, keepdim=True) + 1e-8)
            pred_dshape_ex = decoder_d_shape(r_shape_ex).argmax(dim=-1)
            pred_dshape_unex = decoder_d_shape(r_shape_unex).argmax(dim=-1)
        decD_raw_correct_ex = (pred_draw_ex == true_ch_ex).cpu().numpy().astype(np.float64)
        decD_raw_correct_unex = (pred_draw_unex == true_ch_unex).cpu().numpy().astype(np.float64)
        decD_shape_correct_ex = (pred_dshape_ex == true_ch_ex).cpu().numpy().astype(np.float64)
        decD_shape_correct_unex = (pred_dshape_unex == true_ch_unex).cpu().numpy().astype(np.float64)
    else:
        decD_raw_correct_ex = None
        decD_raw_correct_unex = None
        decD_shape_correct_ex = None
        decD_shape_correct_unex = None

    # ---------- Decoder B (5-fold CV, per-trial correctness) ----------
    rB_ex = r_probe_ex.detach().cpu()
    rB_unex = r_probe_unex.detach().cpu()
    yB_ex = true_ch_ex.detach().cpu()
    yB_unex = true_ch_unex.detach().cpu()
    decB_folds_ex, decB_pertrial_ex = cv_nearest_centroid_per_trial(
        rB_ex, yB_ex, n_folds=N_FOLDS_DEC_B, seed=seed)
    decB_folds_unex, decB_pertrial_unex = cv_nearest_centroid_per_trial(
        rB_unex, yB_unex, n_folds=N_FOLDS_DEC_B, seed=seed)

    # ---------- Decoder-free metrics (re-centered) ----------
    r_probe_ex_np = r_probe_ex.detach().cpu().numpy().astype(np.float64)
    r_probe_unex_np = r_probe_unex.detach().cpu().numpy().astype(np.float64)
    true_ch_ex_np = true_ch_ex.detach().cpu().numpy().astype(np.int64)
    true_ch_unex_np = true_ch_unex.detach().cpu().numpy().astype(np.int64)

    rolled_ex = per_trial_recenter(r_probe_ex_np, true_ch_ex_np, n_ori)
    rolled_unex = per_trial_recenter(r_probe_unex_np, true_ch_unex_np, n_ori)

    peak_ex, net_ex_v, fwhm_ex = per_trial_peak_net_fwhm(rolled_ex, n_ori, step_deg)
    peak_unex, net_unex_v, fwhm_unex = per_trial_peak_net_fwhm(rolled_unex, n_ori, step_deg)

    # ---------- S(d) profile (paired) ----------
    s_mean, s_sem, _ = s_of_d_profile(rolled_ex, rolled_unex)
    half = n_ori // 2
    offsets = np.arange(-half, half, dtype=np.int64)              # [-18..+17] (length n_ori)
    # Re-order rolled curves so index [half+d] corresponds to offset d
    # (roll_to_center places true_ch at center_idx = half = 18; so rolled[:, half+d]
    # is the response d channels away from the probe true_ch (unsigned offset on the
    # ring; absolute offset since HMM unex is +n_ori/2 shift, not march-direction).)
    # offsets array already aligns with rolled axis.

    # ---------- Tuning curves (per-channel mean ± SEM, for completeness) ----------
    n = rolled_ex.shape[0]
    tc_mean_ex = rolled_ex.mean(axis=0)
    tc_sem_ex = rolled_ex.std(axis=0, ddof=1) / np.sqrt(n)
    tc_mean_unex = rolled_unex.mean(axis=0)
    tc_sem_unex = rolled_unex.std(axis=0, ddof=1) / np.sqrt(n)

    # ---------- Bootstrap CIs ----------
    # Note: dec B trials have NaN for trials outside test folds (when n % n_folds != 0).
    boot_decC = paired_bootstrap_diff(decC_correct_ex, decC_correct_unex, n_boot, seed=seed)
    boot_decB = paired_bootstrap_diff(decB_pertrial_ex, decB_pertrial_unex, n_boot, seed=seed)
    boot_peak = paired_bootstrap_diff(peak_ex, peak_unex, n_boot, seed=seed)
    boot_net = paired_bootstrap_diff(net_ex_v, net_unex_v, n_boot, seed=seed)
    boot_fwhm = paired_bootstrap_diff(fwhm_ex, fwhm_unex, n_boot, seed=seed)
    if decA_correct_ex is not None:
        boot_decA = paired_bootstrap_diff(decA_correct_ex, decA_correct_unex, n_boot, seed=seed)
    else:
        boot_decA = None

    # ---------- Per-trial NPZ dump for downstream cross-decoder reuse ----------
    if pertrial_dump_dir is not None:
        import os as _os
        _os.makedirs(pertrial_dump_dir, exist_ok=True)
        _dump_path = _os.path.join(pertrial_dump_dir, f"{cond_id}_pertrial.npz")
        np.savez(
            _dump_path,
            r_probe_ex=r_probe_ex_np.astype(np.float32),
            r_probe_unex=r_probe_unex_np.astype(np.float32),
            true_ch_ex=true_ch_ex_np,
            true_ch_unex=true_ch_unex_np,
            decC_correct_ex=decC_correct_ex,
            decC_correct_unex=decC_correct_unex,
            decB_pertrial_ex=decB_pertrial_ex,
            decB_pertrial_unex=decB_pertrial_unex,
            decA_correct_ex=(decA_correct_ex if decA_correct_ex is not None
                              else np.full_like(decC_correct_ex, np.nan)),
            decA_correct_unex=(decA_correct_unex if decA_correct_unex is not None
                                else np.full_like(decC_correct_unex, np.nan)),
            cond_id=np.array(cond_id),
        )
        print(f"[{cond_id}] dumped per-trial NPZ → {_dump_path}", flush=True)

    # ---------- Bootstrap CIs at S(d=0) and S(d=±3) ----------
    # S(d) per-trial = unex - ex (note sign: opposite of Δ above which uses ex - unex)
    diffs_full = rolled_unex.astype(np.float64) - rolled_ex.astype(np.float64)   # [n_trials, n_ori]
    rng_s = np.random.default_rng(seed)
    n_t = diffs_full.shape[0]
    idx_s = rng_s.integers(0, n_t, size=(n_boot, n_t))
    boot_s_full = diffs_full[idx_s].mean(axis=1)                                  # [n_boot, n_ori]
    s_lo = np.percentile(boot_s_full, 2.5, axis=0)
    s_hi = np.percentile(boot_s_full, 97.5, axis=0)

    def _s_at(d):
        ix = half + d                        # offset d → array index
        return dict(
            d=int(d),
            s_mean=float(s_mean[ix]),
            s_sem=float(s_sem[ix]),
            ci_lo=float(s_lo[ix]),
            ci_hi=float(s_hi[ix]),
        )

    s_at_offsets = {f"d={d}": _s_at(d) for d in [-3, 0, 3]}

    # ---------- Pack ----------
    def _branch_summary(decC_corr, decB_folds, decB_pertrial, decA_corr, peak, netv, fwhm,
                        tc_m, tc_s, decD_raw_corr=None, decD_shape_corr=None):
        m_decC, s_decC, _ = mean_sem(decC_corr)
        decB_arr = np.array(decB_folds, dtype=np.float64)
        m_decB = float(decB_arr.mean())
        s_decB_folds = (float(decB_arr.std(ddof=1) / np.sqrt(len(decB_arr)))
                        if len(decB_arr) > 1 else 0.0)
        m_decB_pt, s_decB_pt, n_decB_pt = mean_sem(decB_pertrial)
        m_peak, s_peak, _ = mean_sem(peak)
        m_net, s_net, _ = mean_sem(netv)
        m_fwhm, s_fwhm, n_fwhm = mean_sem(fwhm)
        if decA_corr is not None:
            m_decA, s_decA, _ = mean_sem(decA_corr)
        else:
            m_decA, s_decA = None, None
        if decD_raw_corr is not None:
            m_decDraw, s_decDraw, _ = mean_sem(decD_raw_corr)
        else:
            m_decDraw, s_decDraw = None, None
        if decD_shape_corr is not None:
            m_decDshape, s_decDshape, _ = mean_sem(decD_shape_corr)
        else:
            m_decDshape, s_decDshape = None, None
        return {
            "n_trials": int(n),
            "decC_acc_mean": m_decC,
            "decC_acc_sem": s_decC,
            "decB_acc_mean": m_decB,
            "decB_acc_sem_across_folds": s_decB_folds,
            "decB_per_fold_acc": list(decB_folds),
            "decB_pertrial_acc_mean": m_decB_pt,
            "decB_pertrial_acc_sem": s_decB_pt,
            "decB_pertrial_n_used": n_decB_pt,
            "decA_acc_mean": m_decA,
            "decA_acc_sem": s_decA,
            "decD_raw_acc_mean": m_decDraw,
            "decD_raw_acc_sem": s_decDraw,
            "decD_shape_acc_mean": m_decDshape,
            "decD_shape_acc_sem": s_decDshape,
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
        "branches": {
            "ex":   _branch_summary(decC_correct_ex, decB_folds_ex, decB_pertrial_ex,
                                    decA_correct_ex,
                                    peak_ex, net_ex_v, fwhm_ex, tc_mean_ex, tc_sem_ex,
                                    decD_raw_corr=decD_raw_correct_ex,
                                    decD_shape_corr=decD_shape_correct_ex),
            "unex": _branch_summary(decC_correct_unex, decB_folds_unex, decB_pertrial_unex,
                                    decA_correct_unex,
                                    peak_unex, net_unex_v, fwhm_unex, tc_mean_unex, tc_sem_unex,
                                    decD_raw_corr=decD_raw_correct_unex,
                                    decD_shape_corr=decD_shape_correct_unex),
        },
        "deltas_ex_minus_unex_bootstrap": {
            "decC_acc": boot_decC,
            "decB_acc": boot_decB,
            "decA_acc": boot_decA,
            "peak":     boot_peak,
            "net":      boot_net,
            "fwhm_deg": boot_fwhm,
        },
        "s_of_d_profile": {
            "offsets": offsets.tolist(),
            "s_mean": s_mean.tolist(),
            "s_sem":  s_sem.tolist(),
            "s_ci_lo": s_lo.tolist(),
            "s_ci_hi": s_hi.tolist(),
            "s_at_key_offsets": s_at_offsets,
            "convention": "S(d) = r_unex(d) − r_ex(d) at re-centered offset d (probe true_ch → 0); HMM is unsigned (no march direction).",
        },
    }
    return out


# -----------------------------------------------------------------------------
# NEW assay loader (Task #19 JSON)
# -----------------------------------------------------------------------------

def build_new_assay_panel(path: str) -> dict[str, Any]:
    """Load Task #19 results/eval_ex_vs_unex_decC_adjacent.json and build S(d) profile
    plus headline scalars matching HMM panels' annotation schema.

    Sign conventions matched:
      - S(d) = unex_mean − ex_mean = −delta_ex_minus_unex
      - S_sem = delta_sem_paired (sign-invariant)
    """
    with open(path, "r") as f:
        d19 = json.load(f)
    rows = d19["full_signed_offset_curve"]
    offsets = np.array([r["offset_ch"] for r in rows], dtype=np.int64)
    ex_mean = np.array([r["ex_mean"] for r in rows], dtype=np.float64)
    unex_mean = np.array([r["unex_mean"] for r in rows], dtype=np.float64)
    delta_paired_sem = np.array([r["delta_sem_paired"] for r in rows], dtype=np.float64)
    s_mean = unex_mean - ex_mean                                # = −delta_ex_minus_unex
    s_sem = delta_paired_sem                                    # paired SEM is sign-invariant

    # Headline: peak Δ at d=0 (ex - unex), reuse stored delta_ex_minus_unex if present
    def _row(d):
        for r in rows:
            if int(r["offset_ch"]) == d:
                return r
        return None

    headline = {}
    for d in [-3, 0, 3]:
        r = _row(d)
        if r is None:
            continue
        idx = int(np.where(offsets == d)[0][0])
        headline[f"d={d}"] = dict(
            d=int(d),
            ex_mean=float(r["ex_mean"]),
            unex_mean=float(r["unex_mean"]),
            delta_ex_minus_unex=float(r["delta_ex_minus_unex"]),
            delta_sem_paired=float(r["delta_sem_paired"]),
            s_mean=float(s_mean[idx]),
            s_sem=float(s_sem[idx]),
        )

    return {
        "id": "NEW_assay",
        "label": "NEW eval (Task #19 march-signed)",
        "source": path,
        "n_total_per_branch": int(d19["design"]["n_total_per_branch"]),
        "centering": d19["design"]["centering"],
        "march_sign_convention": d19["design"]["march_sign_convention"],
        "s_of_d_profile": {
            "offsets": offsets.tolist(),
            "s_mean": s_mean.tolist(),
            "s_sem":  s_sem.tolist(),
            "s_at_key_offsets": headline,
            "convention": "S(d) = r_unex(d) − r_ex(d), march-signed (offset>0 = ahead in march direction).",
        },
        "sanity": {
            "peak_at_stim_ex_d0":   float(d19.get("sanity", {}).get("peak_at_stim_ex", float("nan"))),
            "peak_at_stim_unex_d0": float(d19.get("sanity", {}).get("peak_at_stim_unex", float("nan"))),
        },
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _format_delta_with_ci(boot: dict[str, Any], fmt: str = ".3f") -> str:
    if boot is None:
        return "n/a"
    m = boot.get("delta_mean", float("nan"))
    lo = boot.get("delta_ci_lo", float("nan"))
    hi = boot.get("delta_ci_hi", float("nan"))
    if np.isnan(m):
        return "n/a"
    return f"{m:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def make_figure(cond_results: list[dict[str, Any]], new_panel: dict[str, Any],
                fig_path: str, n_trials: int, seed: int, n_seeds: int):
    """5-panel figure: NEW + C1/C2/C3/C4. Each panel shows S(d) ± SEM with vertical
    lines at d=0 and d=±3, plus a small annotation block of headline Δ stats."""
    fig = plt.figure(figsize=(20.0, 5.5))
    gs = GridSpec(1, 5, figure=fig, wspace=0.28)

    panels = [new_panel] + cond_results

    # Compute consistent y-limits across panels for visual comparability
    all_lo, all_hi = [], []
    for p in panels:
        sm = np.array(p["s_of_d_profile"]["s_mean"])
        ss = np.array(p["s_of_d_profile"]["s_sem"])
        all_lo.append((sm - ss).min())
        all_hi.append((sm + ss).max())
    y_lo = float(min(all_lo)) - 0.02
    y_hi = float(max(all_hi)) + 0.02

    for col, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        offsets = np.array(p["s_of_d_profile"]["offsets"])
        s_mean = np.array(p["s_of_d_profile"]["s_mean"])
        s_sem = np.array(p["s_of_d_profile"]["s_sem"])
        # Restrict to [-17, +17] (drop -18 if present)
        keep = (offsets >= -17) & (offsets <= 17)
        offsets_p = offsets[keep]
        s_mean_p = s_mean[keep]
        s_sem_p = s_sem[keep]

        ax.plot(offsets_p, s_mean_p, "-", color="C3", lw=1.6, marker="o", ms=3.5,
                label="S(d) = unex − ex")
        ax.fill_between(offsets_p, s_mean_p - s_sem_p, s_mean_p + s_sem_p,
                        color="C3", alpha=0.20)

        # Reference lines at d=0 and d=±3
        ax.axhline(0.0, color="black", lw=0.6, ls=":")
        ax.axvline(0.0, color="gray", lw=0.9, ls="--", label="d=0 (center)")
        ax.axvline(+3.0, color="dodgerblue", lw=0.7, ls="--", label="d=±3 (flank)")
        ax.axvline(-3.0, color="dodgerblue", lw=0.7, ls="--")

        ax.set_xlabel("Channel offset d from probe true_ch")
        if col == 0:
            ax.set_ylabel("S(d) = r_unex(d) − r_ex(d)")
        ax.set_xticks(np.arange(-15, 16, 5))
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        ax.set_title(p["label"], fontsize=10)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")

        # Annotation block: per-panel Δ summary
        if col == 0:
            # NEW assay: use stored delta_ex_minus_unex at key offsets (no DecB/DecC here)
            key = p["s_of_d_profile"]["s_at_key_offsets"]
            d0 = key.get("d=0", {})
            d3 = key.get("d=3", {})
            dm3 = key.get("d=-3", {})
            txt = (
                f"NEW (n={p['n_total_per_branch']}/branch)\n"
                f"Δ peak (d=0):  {d0.get('delta_ex_minus_unex', float('nan')):+.3f} "
                f"(±{d0.get('delta_sem_paired', float('nan')):.3f} SEM)\n"
                f"Δ ex−unex (d=+3): {d3.get('delta_ex_minus_unex', float('nan')):+.3f}\n"
                f"Δ ex−unex (d=−3): {dm3.get('delta_ex_minus_unex', float('nan')):+.3f}"
            )
        else:
            # HMM panel
            db = p["deltas_ex_minus_unex_bootstrap"]
            n_t = p["branches"]["ex"]["n_trials"]
            txt = (
                f"HMM (n={n_t}/branch)\n"
                f"Δ peak:    {_format_delta_with_ci(db['peak'], '.3f')}\n"
                f"Δ net:     {_format_delta_with_ci(db['net'], '.2f')}\n"
                f"Δ FWHM°:   {_format_delta_with_ci(db['fwhm_deg'], '.2f')}\n"
                f"Δ Dec B:   {_format_delta_with_ci(db['decB_acc'], '.3f')}\n"
                f"Δ Dec C:   {_format_delta_with_ci(db['decC_acc'], '.3f')}"
            )
        # Top-right anchored, in-axes
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.92))

    fig.suptitle(
        "R1+R2 paradigm × readout: S(d) = r_unex(d) − r_ex(d) profile across NEW + 4 HMM-fork conditions  "
        f"(HMM: n={n_trials}/branch, seed={seed}, n_seeds={n_seeds}; bootstrap n_resamples=1000, 95% CI)",
        fontsize=11, y=1.04)
    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {fig_path}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=CKPT_PATH_DEFAULT)
    p.add_argument("--decoder-c", default=DECODER_C_PATH_DEFAULT)
    p.add_argument("--config", default=CONFIG_PATH_DEFAULT)
    p.add_argument("--new-assay-json", default=NEW_ASSAY_JSON_DEFAULT)
    p.add_argument("--output-json", default=OUT_JSON_DEFAULT)
    p.add_argument("--output-fig", default=OUT_FIG_DEFAULT)
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--device", default=None)
    p.add_argument("--pertrial-dump-dir", default=None,
                   help="If set, dumps per-condition r_probe + decoder correctness arrays as NPZs.")
    p.add_argument("--conditions", default=None,
                   help="Comma-separated list of condition IDs to run (default=all).")
    p.add_argument("--decoder-d-raw-path", default=None,
                   help="Task #4: Dec D-raw ckpt path (Linear(N, N)+bias). If set "
                        "together with --decoder-d-shape-path, Δ_D columns are "
                        "computed per branch.")
    p.add_argument("--decoder-d-shape-path", default=None,
                   help="Task #4: Dec D-shape ckpt path (Linear(N, N)+bias). "
                        "Applied to r_l23 / (r_l23.sum + 1e-8).")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] decoder_c={args.decoder_c}", flush=True)
    print(f"[setup] n_trials={args.n_trials}  seed={args.seed}  n_boot={args.n_boot}",
          flush=True)
    print(f"[setup] new_assay_json={args.new_assay_json}", flush=True)

    # NEW assay panel
    new_panel = build_new_assay_panel(args.new_assay_json)
    print(f"[new-assay] loaded NEW assay (n_total_per_branch="
          f"{new_panel['n_total_per_branch']})", flush=True)

    # Load model + config
    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    print(f"[cfg] n_orientations={n_ori}  period={period}  "
          f"steps_on={train_cfg.steps_on}  steps_isi={train_cfg.steps_isi}  "
          f"seq_length={SEQ_LENGTH}  cue_valid_fraction={stim_cfg.cue_valid_fraction}",
          flush=True)

    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Task #26: strict=False to support legacy three-regimes ckpts (a1/b1/c1/e1)
    res = net.load_state_dict(ckpt["model_state"], strict=False)
    if res.unexpected_keys:
        print(f"[setup] PORT: {len(res.unexpected_keys)} unexpected keys ignored "
              f"(legacy ckpt)", flush=True)
    if res.missing_keys:
        print(f"[setup] PORT: {len(res.missing_keys)} missing keys default-init",
              flush=True)
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    print(f"[setup] network loaded (oracle_mode={net.oracle_mode}, "
          f"feedback_scale={float(net.feedback_scale.item()):.3f})", flush=True)

    dec_ckpt = torch.load(args.decoder_c, map_location=device, weights_only=False)
    decoder_c = nn.Linear(n_ori, n_ori).to(device)
    decoder_c.load_state_dict(dec_ckpt["state_dict"])
    decoder_c.eval()
    best_val = dec_ckpt.get("train_meta", {}).get("best_val_acc", float("nan"))
    print(f"[setup] Decoder C loaded (best_val_acc={best_val:.4f})", flush=True)

    # Task #26: load Decoder A from network ckpt's orientation_decoder head
    decoder_a = None
    decA_state = None
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        decA_state = ckpt["loss_heads"]["orientation_decoder"]
    elif "decoder_state" in ckpt:
        decA_state = ckpt["decoder_state"]
    if decA_state is not None:
        decoder_a = nn.Linear(n_ori, n_ori).to(device)
        decoder_a.load_state_dict(decA_state)
        decoder_a.eval()
        print(f"[setup] Decoder A loaded from ckpt['loss_heads' or 'decoder_state']",
              flush=True)
    else:
        print(f"[setup] WARN: Decoder A not found in ckpt; will skip Dec A", flush=True)

    # Task #4 — load Dec D-raw / D-shape if both paths provided
    decoder_d_raw = None
    decoder_d_shape = None
    if args.decoder_d_raw_path and args.decoder_d_shape_path:
        dR_ckpt = torch.load(args.decoder_d_raw_path, map_location=device, weights_only=False)
        dS_ckpt = torch.load(args.decoder_d_shape_path, map_location=device, weights_only=False)
        decoder_d_raw = nn.Linear(n_ori, n_ori, bias=True).to(device)
        decoder_d_raw.load_state_dict(dR_ckpt['state_dict'] if isinstance(dR_ckpt, dict) and 'state_dict' in dR_ckpt else dR_ckpt)
        decoder_d_raw.eval()
        decoder_d_shape = nn.Linear(n_ori, n_ori, bias=True).to(device)
        decoder_d_shape.load_state_dict(dS_ckpt['state_dict'] if isinstance(dS_ckpt, dict) and 'state_dict' in dS_ckpt else dS_ckpt)
        decoder_d_shape.eval()
        print(f"[setup] Decoder D-raw   loaded from {args.decoder_d_raw_path}", flush=True)
        print(f"[setup] Decoder D-shape loaded from {args.decoder_d_shape_path}", flush=True)
    elif args.decoder_d_raw_path or args.decoder_d_shape_path:
        raise ValueError("--decoder-d-raw-path and --decoder-d-shape-path must be set together")

    # Build ONE HMM batch with sweep_rescue_1_2 params
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
    )
    rng = torch.Generator().manual_seed(args.seed)
    print(f"[hmm] generating {args.n_trials} trials × seq_length={SEQ_LENGTH}...",
          flush=True)
    metadata_base = gen.generate(args.n_trials, SEQ_LENGTH, generator=rng)
    print(f"[hmm] generated: orientations.shape={tuple(metadata_base.orientations.shape)}  "
          f"cues.shape={tuple(metadata_base.cues.shape)}  "
          f"task_states.shape={tuple(metadata_base.task_states.shape)}  "
          f"is_ambiguous.sum()={int(metadata_base.is_ambiguous.sum())}", flush=True)

    # Run all 4 conditions (or filter via --conditions)
    cond_filter = None
    if getattr(args, 'conditions', None):
        cond_filter = set(s.strip() for s in args.conditions.split(','))
        print(f"[setup] condition filter: {sorted(cond_filter)}", flush=True)
    cond_results = []
    t0 = time.time()
    for cond_id, cond_label, ts_val, zero_cue in CONDITIONS:
        if cond_filter is not None and cond_id not in cond_filter:
            print(f"[skip] {cond_id} not in --conditions filter", flush=True)
            continue
        ts_init = time.time()
        print(f"\n=== Condition {cond_id} ({cond_label}) ===", flush=True)
        res = run_condition(cond_id, cond_label, ts_val, zero_cue,
                            metadata_base, model_cfg, train_cfg, stim_cfg,
                            net, decoder_c, device, args.n_trials, args.seed,
                            args.n_boot,
                            decoder_a=decoder_a,
                            pertrial_dump_dir=getattr(args, 'pertrial_dump_dir', None),
                            decoder_d_raw=decoder_d_raw,
                            decoder_d_shape=decoder_d_shape)
        cond_results.append(res)

        # Per-condition headline summary
        print(f"\n[{cond_id}] summary (Δ = ex − unex; bootstrap 95% CI):", flush=True)
        db = res["deltas_ex_minus_unex_bootstrap"]
        print(f"  Δ Dec C:   {_format_delta_with_ci(db['decC_acc'], '.4f')}", flush=True)
        print(f"  Δ Dec B:   {_format_delta_with_ci(db['decB_acc'], '.4f')}", flush=True)
        print(f"  Δ peak:    {_format_delta_with_ci(db['peak'], '.4f')}", flush=True)
        print(f"  Δ net:     {_format_delta_with_ci(db['net'], '.4f')}", flush=True)
        print(f"  Δ FWHM°:   {_format_delta_with_ci(db['fwhm_deg'], '.3f')}", flush=True)
        s0 = res["s_of_d_profile"]["s_at_key_offsets"]["d=0"]
        sp3 = res["s_of_d_profile"]["s_at_key_offsets"]["d=3"]
        sm3 = res["s_of_d_profile"]["s_at_key_offsets"]["d=-3"]
        print(f"  S(0)  = {s0['s_mean']:+.4f}  [{s0['ci_lo']:+.4f}, {s0['ci_hi']:+.4f}]",
              flush=True)
        print(f"  S(+3) = {sp3['s_mean']:+.4f}  [{sp3['ci_lo']:+.4f}, {sp3['ci_hi']:+.4f}]",
              flush=True)
        print(f"  S(−3) = {sm3['s_mean']:+.4f}  [{sm3['ci_lo']:+.4f}, {sm3['ci_hi']:+.4f}]",
              flush=True)
        print(f"[{cond_id}] elapsed: {time.time() - ts_init:.1f}s", flush=True)
    print(f"\n[total] HMM elapsed: {time.time() - t0:.1f}s", flush=True)

    # Pack JSON
    out = {
        "label": "Task #22: paradigm × readout matrix on R1+R2 (4 HMM conditions + NEW assay column)",
        "checkpoint": args.checkpoint,
        "decoder_c": args.decoder_c,
        "config": args.config,
        "new_assay_source": args.new_assay_json,
        "design": {
            "n_trials_per_condition": args.n_trials,
            "seq_length": SEQ_LENGTH,
            "seed": args.seed,
            "n_seeds_available": 1,
            "n_boot": args.n_boot,
            "bootstrap_method": "trial-only (paired); hierarchical seed+trial reduces to trial bootstrap because only seed=42 is available",
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
            "decoder_b_method": "nearest-centroid 5-fold CV (fit within slice; per-trial correctness for bootstrap)",
            "decoder_c_method": "frozen Linear(36,36) argmax top-1",
            "fwhm_method": "linear-interp at half-max of max-normalised re-centered curve",
            "delta_sign_convention": "Δ X = mean(X_ex) − mean(X_unex); positive Δ peak/net = expected has higher response",
            "S_of_d_sign_convention": "S(d) = r_unex(d) − r_ex(d); positive S(d) = unexpected is larger at offset d (= 'expected suppression' at d)",
        },
        "flags": {
            "single_seed_only": True,
            "single_seed_reason": "results/simple_dual/ contains only emergent_seed42 for R1+R2; no sibling seed_* dirs found.",
            "bootstrap_reduced": "hierarchical (seed, trial) bootstrap reduces to trial-only bootstrap with n_seeds=1.",
        },
        "new_assay_panel": new_panel,
        "conditions": cond_results,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[json] wrote {args.output_json}", flush=True)

    # Skip figure if subset of conditions (figure expects all 4)
    if cond_filter is None:
        make_figure(cond_results, new_panel, args.output_fig,
                    n_trials=args.n_trials, seed=args.seed, n_seeds=1)
    else:
        print(f"[fig] skipped (--conditions filter active)", flush=True)


if __name__ == "__main__":
    main()
