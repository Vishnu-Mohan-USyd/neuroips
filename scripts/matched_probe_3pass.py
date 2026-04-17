#!/usr/bin/env python3
"""Matched-probe 3-pass HMM ring figure (Task #37).

Fixes the matched-context bug in Task #35/#36: those scripts averaged
**different** trials per row, so the P-2/P-1 context rings were not
truly matched across rows. Here we run **THREE forward passes per
qualifying sequence**, all sharing the **same context up to the probe**:

* **Pass A (Expected)**: original stim — probe is the natural march
  destination.
* **Pass B (Unexpected)**: same stim except probe ON window replaced
  with a grating at ``(march_destination + 90°) mod orientation_range``.
  Same contrast as Pass A.
* **Pass C (Omission)**: same stim except probe ON window zeroed out.
  task_state and cue are NOT zeroed — only stim.

Because the stim diverges *only* at the probe ON window (target_idx ON
steps), the network's pre-probe state — and therefore r_l23 at P-2 and
P-1 readout windows — are **exactly equal across all three passes** by
construction. Only the probe ring differs.

Filter (applied on Pass A only — same subset of trials used in B & C)
---------------------------------------------------------------------
* Clean 3-step march at target slot (S-1):
  ``signed_delta(true[t-1], true[t-2]) == ±transition_step`` AND
  ``signed_delta(true[t], true[t-1])`` matches in same direction +
  magnitude (within ``--step-tol``).
* ``is_ambiguous[target] == False``.
* ``pi_pred_eff(target ISI) ≥ Q_p`` (default Q75) of the **broad** pool
  of all ¬amb pass-A target pi values (BEFORE structural filter), per
  Task #36's pi-pool fix.
* ``pred_err(target) ≤ --exp-pred-err-max`` (default 5°) on Pass A —
  V2 must actually predict the upcoming probe correctly.

If n_expected < ``--min-bucket-n`` after the 5° cut, the pred_err
threshold cascades 5°→10°→15° (per Lead's brief). Pi threshold is held.

The Unexpected and Omission passes are **NOT** filtered separately —
they re-use exactly the same subset of trials as Pass A. The whole
point is "holding context matched, how does the probe response differ
per branch?"

Re-centering (per trial)
------------------------
``shift = N//2 - march_destination_channel = 18 - true_ch[target]``,
applied identically to r_l23 at P-2, P-1, Probe windows for ALL three
passes.

Mirror-flip CCW trials about centre (toggle ``--no-flip-ccw``):
  ``out[j] = rolled[(2*center - j) % N]`` for trials with
  ``signed_delta(t-1, t-2) < 0``.

Bead positions (re-centered + CW-flipped frame)
------------------------------------------------
* P-2 column bead: ch ``(N//2 - 2 * step_ch) % N`` (single bead)
* P-1 column bead: ch ``(N//2 - 1 * step_ch) % N`` (single bead)
* Probe column has THREE sub-cells (E, U, O stacked):
  * E (Expected): bead at ``N//2`` (= 18)
  * U (Unexpected): bead at ``(N//2 + 90°/step_deg) % N`` (= 0 for N=36)
  * O (Omission): no bead

Outputs
-------
PNG to ``--output-fig`` and JSON stats to ``--output-json``.
"""
from __future__ import annotations

import argparse
import json
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
from src.stimulus.gratings import generate_grating
from src.training.trainer import build_stimulus_sequence

from matched_quality_sim import (
    circular_distance,
    _load_decoder,
    bootstrap_acc_ci,
    roll_to_center,
    ks_2sample,
)
from plot_tuning_ring_extended import fwhm_of_curve
from plot_tuning_ring_sequence import (
    plot_ring_icon,
    plot_stimulus_bead_ring,
)
from matched_hmm_ring_sequence import (
    signed_circ_delta,
    _flip_ccw_about_center,
)


# ---------------------------------------------------------------------------
# Three-pass collection
# ---------------------------------------------------------------------------

def collect_records(args, device: torch.device) -> tuple[dict[str, Any], dict]:
    """Run THREE forward passes per batch (Expected / Unexpected / Omission),
    all sharing identical context. Filter on Pass A; B and C use the same
    subset.

    Returns
    -------
    records : dict
        Per-trial arrays (one row per BATCH-trial; total = n_batches*B):
          "is_clean_march": bool         — 3-march at target slot
          "is_amb_target":  bool
          "ctx_dir":        int8         — +1 CW, -1 CCW, 0 other
          "target_true_ch": int64
          "target_true_ori": float32
          "pi_target":      float32      — pi_pred_eff at target_isi_pre
          "pred_err_A":     float32      — Pass A pred_err at target slot (deg)
          "r_pm2":          float32 [n,N]
          "r_pm1":          float32 [n,N]
          "r_probe_A":      float32 [n,N]
          "r_probe_B":      float32 [n,N]
          "r_probe_C":      float32 [n,N]
          "decoder_top1_A": int64
          "decoder_top1_B": int64
          "decoder_top1_C": int64
          "unexp_probe_ori": float32     — orientation used for Pass B (deg)
          "unexp_probe_ch":  int64       — channel argmax of Pass B grating
    meta : dict
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
    center_idx = N // 2

    decoder = _load_decoder(ckpt, N, device)

    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi

    W_START, W_END = 9, 11
    assert W_END < steps_on, f"Window [{W_START},{W_END}] outside steps_on={steps_on}"

    target_idx = (seq_length - 1) if args.target_idx is None else int(args.target_idx)
    assert 2 <= target_idx <= seq_length - 1, \
        f"target_idx={target_idx} must be in [2, seq_length-1={seq_length - 1}]"
    target_onset = target_idx * steps_per
    target_isi_pre = target_onset - 1
    transition_step = float(stim_cfg.transition_step)
    step_tol = float(args.step_tol)
    assert step_tol > 0.0, "step_tol must be positive"

    # Window slices: (start_inclusive, end_inclusive) for P-2, P-1, Probe
    win_pm2 = (
        (target_idx - 2) * steps_per + W_START,
        (target_idx - 2) * steps_per + W_END,
    )
    win_pm1 = (
        (target_idx - 1) * steps_per + W_START,
        (target_idx - 1) * steps_per + W_END,
    )
    win_target = (
        target_idx * steps_per + W_START,
        target_idx * steps_per + W_END,
    )

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

    # Per-trial buffers (one row per batch-trial)
    buf_is_clean: list[np.ndarray] = []
    buf_is_amb: list[np.ndarray] = []
    buf_ctx_dir: list[np.ndarray] = []
    buf_tgt_ch: list[np.ndarray] = []
    buf_tgt_ori: list[np.ndarray] = []
    buf_pi: list[np.ndarray] = []
    buf_pred_err: list[np.ndarray] = []
    buf_r_pm2: list[np.ndarray] = []
    buf_r_pm1: list[np.ndarray] = []
    buf_r_pA: list[np.ndarray] = []
    buf_r_pB: list[np.ndarray] = []
    buf_r_pC: list[np.ndarray] = []
    buf_dec_A: list[np.ndarray] = []
    buf_dec_B: list[np.ndarray] = []
    buf_dec_C: list[np.ndarray] = []
    buf_unexp_ori: list[np.ndarray] = []
    buf_unexp_ch: list[np.ndarray] = []

    # Track P-2/P-1 identity diagnostic across A/B/C
    # (bool: max abs diff < tol). Computed per batch, broadcast to trials.
    diag_max_abs_pm2_A_vs_B: list[float] = []
    diag_max_abs_pm2_A_vs_C: list[float] = []
    diag_max_abs_pm1_A_vs_B: list[float] = []
    diag_max_abs_pm1_A_vs_C: list[float] = []

    n_total = 0

    with torch.no_grad():
        for batch_i in range(args.n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            true_ori = metadata.orientations.to(device)            # [B, S]
            is_amb_all = metadata.is_ambiguous.to(device)           # [B, S]
            true_ch_all = (true_ori / step_deg).round().long() % N  # [B, S]
            contrasts = metadata.contrasts.to(device)               # [B, S]

            B = stim_seq.shape[0]

            # --- Pass A: original stim ---
            packed_A = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_A, _, aux_A = net.forward(packed_A)               # [B, T, N]
            q_pred_A = aux_A["q_pred_all"]                           # [B, T, N]
            pi_eff_A = aux_A["pi_pred_eff_all"]                      # [B, T, 1]

            # Filter computables (Pass A) at target slot
            ori_pm0 = true_ori[:, target_idx]
            ori_pm1 = true_ori[:, target_idx - 1]
            ori_pm2 = true_ori[:, target_idx - 2]
            d_ctx = signed_circ_delta(ori_pm1, ori_pm2, period)      # [B]
            d_probe = signed_circ_delta(ori_pm0, ori_pm1, period)    # [B]
            ctx_match_step = (d_ctx.abs() - transition_step).abs() <= step_tol
            probe_match_step = (d_probe.abs() - transition_step).abs() <= step_tol
            same_dir = (torch.sign(d_ctx) == torch.sign(d_probe)) & (d_ctx.abs() > 1e-6)
            is_clean_march = ctx_match_step & probe_match_step & same_dir

            ctx_dir = torch.zeros_like(d_ctx)
            ctx_dir[ctx_match_step & (d_ctx > 0)] = 1.0
            ctx_dir[ctx_match_step & (d_ctx < 0)] = -1.0

            pi_target = pi_eff_A[:, target_isi_pre, 0]               # [B]
            q_pred_target_isi = q_pred_A[:, target_isi_pre, :]
            pred_peak_idx = q_pred_target_isi.argmax(dim=-1)
            pred_ori = pred_peak_idx.float() * step_deg
            actual_ori = true_ori[:, target_idx]
            pred_err = circular_distance(pred_ori, actual_ori, period)

            target_ch = true_ch_all[:, target_idx]
            is_amb_target = is_amb_all[:, target_idx]

            # --- Pass B: replace probe ON window with +90° grating ---
            unexp_ori = (actual_ori + period / 2.0) % period         # [B] (deg)
            grating_B = generate_grating(
                unexp_ori.cpu(), contrasts[:, target_idx].cpu(),
                n_orientations=N,
                sigma=model_cfg.sigma_ff,
                n=model_cfg.naka_rushton_n,
                c50=model_cfg.naka_rushton_c50,
                period=period,
            ).to(device)                                              # [B, N]
            stim_B = stim_seq.clone()
            # Replace [target_onset:target_onset+steps_on] with broadcast grating
            stim_B[:, target_onset:target_onset + steps_on, :] = grating_B.unsqueeze(1)
            packed_B = net.pack_inputs(stim_B, cue_seq, ts_seq)
            r_l23_B, _, _ = net.forward(packed_B)

            # --- Pass C: zero probe ON window ---
            stim_C = stim_seq.clone()
            stim_C[:, target_onset:target_onset + steps_on, :] = 0.0
            packed_C = net.pack_inputs(stim_C, cue_seq, ts_seq)
            r_l23_C, _, _ = net.forward(packed_C)

            # --- Per-trial readout windows (mean over [9,11]) ---
            r_pm2 = r_l23_A[:, win_pm2[0]:win_pm2[1] + 1, :].mean(dim=1)
            r_pm1 = r_l23_A[:, win_pm1[0]:win_pm1[1] + 1, :].mean(dim=1)
            r_probe_A = r_l23_A[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)
            r_probe_B = r_l23_B[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)
            r_probe_C = r_l23_C[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)

            # Diagnostic: confirm P-2/P-1 windows are bit-identical across A/B/C
            r_pm2_B = r_l23_B[:, win_pm2[0]:win_pm2[1] + 1, :].mean(dim=1)
            r_pm2_C = r_l23_C[:, win_pm2[0]:win_pm2[1] + 1, :].mean(dim=1)
            r_pm1_B = r_l23_B[:, win_pm1[0]:win_pm1[1] + 1, :].mean(dim=1)
            r_pm1_C = r_l23_C[:, win_pm1[0]:win_pm1[1] + 1, :].mean(dim=1)
            diag_max_abs_pm2_A_vs_B.append(float((r_pm2 - r_pm2_B).abs().max()))
            diag_max_abs_pm2_A_vs_C.append(float((r_pm2 - r_pm2_C).abs().max()))
            diag_max_abs_pm1_A_vs_B.append(float((r_pm1 - r_pm1_B).abs().max()))
            diag_max_abs_pm1_A_vs_C.append(float((r_pm1 - r_pm1_C).abs().max()))

            # Decoder readouts on each probe
            dec_A = decoder(r_probe_A).argmax(dim=-1)
            dec_B = decoder(r_probe_B).argmax(dim=-1)
            dec_C = decoder(r_probe_C).argmax(dim=-1)

            unexp_ch = (unexp_ori / step_deg).round().long() % N

            buf_is_clean.append(is_clean_march.cpu().numpy())
            buf_is_amb.append(is_amb_target.cpu().numpy())
            buf_ctx_dir.append(ctx_dir.cpu().numpy())
            buf_tgt_ch.append(target_ch.cpu().numpy())
            buf_tgt_ori.append(actual_ori.cpu().numpy())
            buf_pi.append(pi_target.cpu().numpy())
            buf_pred_err.append(pred_err.cpu().numpy())
            buf_r_pm2.append(r_pm2.cpu().numpy())
            buf_r_pm1.append(r_pm1.cpu().numpy())
            buf_r_pA.append(r_probe_A.cpu().numpy())
            buf_r_pB.append(r_probe_B.cpu().numpy())
            buf_r_pC.append(r_probe_C.cpu().numpy())
            buf_dec_A.append(dec_A.cpu().numpy())
            buf_dec_B.append(dec_B.cpu().numpy())
            buf_dec_C.append(dec_C.cpu().numpy())
            buf_unexp_ori.append(unexp_ori.cpu().numpy())
            buf_unexp_ch.append(unexp_ch.cpu().numpy())
            n_total += B

    records = {
        "is_clean_march": np.concatenate(buf_is_clean, axis=0).astype(bool),
        "is_amb_target":  np.concatenate(buf_is_amb, axis=0).astype(bool),
        "ctx_dir":        np.concatenate(buf_ctx_dir, axis=0).astype(np.int8),
        "target_true_ch": np.concatenate(buf_tgt_ch, axis=0).astype(np.int64),
        "target_true_ori": np.concatenate(buf_tgt_ori, axis=0).astype(np.float32),
        "pi_target":      np.concatenate(buf_pi, axis=0).astype(np.float32),
        "pred_err_A":     np.concatenate(buf_pred_err, axis=0).astype(np.float32),
        "r_pm2":          np.concatenate(buf_r_pm2, axis=0).astype(np.float32),
        "r_pm1":          np.concatenate(buf_r_pm1, axis=0).astype(np.float32),
        "r_probe_A":      np.concatenate(buf_r_pA, axis=0).astype(np.float32),
        "r_probe_B":      np.concatenate(buf_r_pB, axis=0).astype(np.float32),
        "r_probe_C":      np.concatenate(buf_r_pC, axis=0).astype(np.float32),
        "decoder_top1_A": np.concatenate(buf_dec_A, axis=0).astype(np.int64),
        "decoder_top1_B": np.concatenate(buf_dec_B, axis=0).astype(np.int64),
        "decoder_top1_C": np.concatenate(buf_dec_C, axis=0).astype(np.int64),
        "unexp_probe_ori": np.concatenate(buf_unexp_ori, axis=0).astype(np.float32),
        "unexp_probe_ch":  np.concatenate(buf_unexp_ch, axis=0).astype(np.int64),
    }

    meta = {
        "N": int(N),
        "step_deg": float(step_deg),
        "center_idx": int(center_idx),
        "period": float(period),
        "seq_length": int(seq_length),
        "batch_size": int(batch_size),
        "steps_on": int(steps_on),
        "steps_isi": int(steps_isi),
        "steps_per_pres": int(steps_per),
        "target_idx": int(target_idx),
        "target_onset_step": int(target_onset),
        "target_isi_pre_step": int(target_isi_pre),
        "win_pm2": list(win_pm2),
        "win_pm1": list(win_pm1),
        "win_target": list(win_target),
        "transition_step_deg": float(transition_step),
        "step_tol_deg": float(step_tol),
        "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "feedback_scale": 1.0,
        "readout_window": {"start": W_START, "end": W_END, "inclusive": True},
        "n_total_trials": int(n_total),
        "n_clean_march_unfiltered": int(records["is_clean_march"].sum()),
        "n_amb_target": int(records["is_amb_target"].sum()),
        "diag_max_abs_pm2_A_vs_B": float(max(diag_max_abs_pm2_A_vs_B)),
        "diag_max_abs_pm2_A_vs_C": float(max(diag_max_abs_pm2_A_vs_C)),
        "diag_max_abs_pm1_A_vs_B": float(max(diag_max_abs_pm1_A_vs_B)),
        "diag_max_abs_pm1_A_vs_C": float(max(diag_max_abs_pm1_A_vs_C)),
    }
    return records, meta


# ---------------------------------------------------------------------------
# Filtering, re-centering, summarising
# ---------------------------------------------------------------------------

def apply_filter(records: dict, pi_q_pct: float,
                 exp_pred_err_max: float
                 ) -> tuple[np.ndarray, float]:
    """Compute pi threshold (broad pool) and the qualifying-trial mask.

    Pi pool = ALL ¬amb pass-A target pi values BEFORE structural filter.
    Mask    = is_clean_march AND ¬is_amb_target AND pi >= threshold AND
              pred_err_A <= exp_pred_err_max.
    """
    keep_amb = ~records["is_amb_target"]
    pool = records["pi_target"][keep_amb]
    if pool.size == 0:
        pi_threshold = float("inf")
    else:
        pi_threshold = float(np.percentile(pool, pi_q_pct))
    mask = (
        records["is_clean_march"]
        & keep_amb
        & (records["pi_target"] >= pi_threshold)
        & (records["pred_err_A"] <= float(exp_pred_err_max))
    )
    return mask, pi_threshold


def recenter_three_passes(records: dict, mask: np.ndarray,
                          center_idx: int, flip_ccw: bool,
                          ) -> dict[str, np.ndarray]:
    """Roll all rings by per-trial shift = center - target_true_ch, then
    optionally mirror-flip CCW trials about the centre."""
    if mask.sum() == 0:
        return {k: None for k in
                ("mean_pm2", "mean_pm1",
                 "mean_probe_A", "mean_probe_B", "mean_probe_C",
                 "n_cw", "n_ccw", "n_other")}
    target_ch = records["target_true_ch"][mask]
    ctx_dir = records["ctx_dir"][mask]
    rolled = {
        "pm2": roll_to_center(records["r_pm2"][mask], target_ch, center_idx),
        "pm1": roll_to_center(records["r_pm1"][mask], target_ch, center_idx),
        "pA":  roll_to_center(records["r_probe_A"][mask], target_ch, center_idx),
        "pB":  roll_to_center(records["r_probe_B"][mask], target_ch, center_idx),
        "pC":  roll_to_center(records["r_probe_C"][mask], target_ch, center_idx),
    }
    if flip_ccw:
        for k in list(rolled.keys()):
            rolled[k] = _flip_ccw_about_center(rolled[k], ctx_dir, center_idx)
    return {
        "mean_pm2":     rolled["pm2"].mean(axis=0),
        "mean_pm1":     rolled["pm1"].mean(axis=0),
        "mean_probe_A": rolled["pA"].mean(axis=0),
        "mean_probe_B": rolled["pB"].mean(axis=0),
        "mean_probe_C": rolled["pC"].mean(axis=0),
        "n_cw": int((ctx_dir == 1).sum()),
        "n_ccw": int((ctx_dir == -1).sum()),
        "n_other": int(((ctx_dir != 1) & (ctx_dir != -1)).sum()),
    }


def per_pass_decoder_acc(records: dict, mask: np.ndarray,
                         pass_key: str, target_ch_key: str
                         ) -> tuple[float | None, list[float | None]]:
    """Return decoder top-1 accuracy + 95% bootstrap CI vs the per-pass
    'true' target channel.

    For Pass A: target_ch_key = "target_true_ch" (true probe channel).
    For Pass B: target_ch_key = "unexp_probe_ch"  (the +90° grating channel).
    For Pass C: target_ch_key = "target_true_ch" (the would-be probe; flagged
                in the figure as n/a interp.).
    """
    if mask.sum() == 0:
        return None, [None, None]
    pred = records[f"decoder_top1_{pass_key}"][mask]
    truth = records[target_ch_key][mask]
    correct = (pred == truth).astype(np.float64)
    acc = float(correct.mean())
    lo, hi = bootstrap_acc_ci(correct)
    return acc, [lo, hi]


def summarise_bucket(name: str, records: dict, mask: np.ndarray,
                     rolled: dict, center_idx: int, step_deg: float
                     ) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {
            "name": name, "n": 0,
            "mean_pm2": None, "mean_pm1": None, "mean_probe": None,
            "peak_at_true_probe": None, "total_probe": None,
            "fwhm_probe_deg": None,
            "decoder_acc": None, "decoder_acc_ci95": [None, None],
            "mean_pi_pred_eff": None, "mean_pred_err_A": None,
            "n_cw": 0, "n_ccw": 0, "n_other": 0,
        }
    if name == "expected":
        mp = rolled["mean_probe_A"]
        acc, ci = per_pass_decoder_acc(records, mask, "A", "target_true_ch")
    elif name == "unexpected":
        mp = rolled["mean_probe_B"]
        acc, ci = per_pass_decoder_acc(records, mask, "B", "unexp_probe_ch")
    elif name == "omission":
        mp = rolled["mean_probe_C"]
        # No actual stim — measure decoder top-1 vs would-be target (n/a interp.)
        acc, ci = per_pass_decoder_acc(records, mask, "C", "target_true_ch")
    else:
        raise ValueError(name)
    return {
        "name": name, "n": n,
        "mean_pm2":   rolled["mean_pm2"].astype(np.float32).tolist(),
        "mean_pm1":   rolled["mean_pm1"].astype(np.float32).tolist(),
        "mean_probe": mp.astype(np.float32).tolist(),
        "peak_at_true_probe": float(mp[center_idx]),
        "total_probe": float(mp.sum()),
        "fwhm_probe_deg": float(fwhm_of_curve(mp, step_deg)),
        "decoder_acc": acc,
        "decoder_acc_ci95": ci,
        "mean_pi_pred_eff": float(records["pi_target"][mask].mean()),
        "mean_pred_err_A": float(records["pred_err_A"][mask].mean()),
        "n_cw": rolled["n_cw"],
        "n_ccw": rolled["n_ccw"],
        "n_other": rolled["n_other"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bead_indices(center_idx: int, N: int, step_deg: float,
                  transition_step_deg: float, period: float
                  ) -> dict[str, int | None]:
    """Canonical bead positions in re-centered + CW-flipped frame.

    P-2 : center - 2*step_ch
    P-1 : center - 1*step_ch
    Probe E : center
    Probe U : (center + 90°/step_deg) mod N
    Probe O : None
    """
    step_ch = int(round(transition_step_deg / step_deg))
    deg_90_ch = int(round((period / 2.0) / step_deg))
    return {
        "pm2": (center_idx - 2 * step_ch) % N,
        "pm1": (center_idx - 1 * step_ch) % N,
        "probe_E": center_idx,
        "probe_U": (center_idx + deg_90_ch) % N,
        "probe_O": None,
    }


def plot_figure(summaries: dict[str, dict], meta: dict, fig_path: str,
                title: str) -> None:
    """4-row × 3-col matched-probe HMM ring figure.

    Row 0 : Expected   P-2 / P-1 / Probe (rings)
    Row 1 : Unexpected P-2 / P-1 / Probe (rings)
    Row 2 : Omission   P-2 / P-1 / Probe (rings)
    Row 3 : Stimulus bead row — single bead per cell in P-2 / P-1 columns;
            three sub-cells (E, U, O stacked) in the Probe column.
    """
    cmap = matplotlib.colormaps["viridis"]
    N = meta["N"]
    center_idx = meta["center_idx"]
    step_deg = meta["step_deg"]
    period = meta["period"]
    transition_step_deg = meta["transition_step_deg"]

    rings_per_bucket = []
    for name in ("expected", "unexpected", "omission"):
        s = summaries[name]
        if s["mean_probe"] is None:
            rings_per_bucket.append([np.zeros(N), np.zeros(N), np.zeros(N)])
        else:
            rings_per_bucket.append([np.array(s[k], dtype=float)
                                     for k in ("mean_pm2", "mean_pm1", "mean_probe")])
    vmax = max(float(r.max()) for triple in rings_per_bucket for r in triple)
    if vmax <= 0:
        vmax = 1.0

    n_cols = 3
    fig = plt.figure(figsize=(11.5, 9.4))
    gs = fig.add_gridspec(
        4, n_cols, height_ratios=[1.0, 1.0, 1.0, 0.75],
        hspace=0.30, wspace=0.10,
    )

    bucket_keys = ["expected", "unexpected", "omission"]
    row_labels = ["Expected", "Unexpected", "Omission"]
    row_notes = [
        "Pass A — original probe (march dest)",
        "Pass B — probe replaced (+90°)",
        "Pass C — probe zeroed",
    ]
    col_headers = ["P-2", "P-1", "Probe"]

    # --- Ring rows ---
    for row_idx, key in enumerate(bucket_keys):
        s = summaries[key]
        rings = rings_per_bucket[row_idx]
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection="polar")
            highlight = (col_idx == n_cols - 1)
            plot_ring_icon(ax, rings[col_idx], vmax, cmap, highlight=highlight,
                           show_cardinals=True)
            if row_idx == 0:
                header = "Probe" if highlight else col_headers[col_idx]
                color = "#9a6700" if highlight else "black"
                weight = "bold" if highlight else None
                ax.set_title(header, y=1.14, fontsize=11, color=color,
                             fontweight=weight)
            if col_idx == 0:
                ax.text(-0.55, 0.62, row_labels[row_idx], transform=ax.transAxes,
                        ha="right", va="center", fontsize=12, fontweight="bold")
                ax.text(-0.55, 0.40, row_notes[row_idx], transform=ax.transAxes,
                        ha="right", va="center", fontsize=8.5, color="0.35")
                if s["n"] > 0:
                    ci = s["decoder_acc_ci95"]
                    ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                              if ci[0] is not None else "n/a")
                    if key == "omission":
                        dec_str = (f"{s['decoder_acc']:.3f} {ci_str} "
                                   f"(n/a interp.)")
                    else:
                        dec_str = f"{s['decoder_acc']:.3f} {ci_str}"
                    ann = (f"n = {s['n']}\n"
                           f"dec acc = {dec_str}\n"
                           f"peak @ true = {s['peak_at_true_probe']:.3f}\n"
                           f"total L2/3 = {s['total_probe']:.2f}\n"
                           f"FWHM = {s['fwhm_probe_deg']:.1f}°\n"
                           f"mean pi = {s['mean_pi_pred_eff']:.3f}")
                else:
                    ann = "n = 0"
                ax.text(-0.55, 0.02, ann, transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.8, color="0.15",
                        linespacing=1.20,
                        bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                                  edgecolor="0.82", alpha=0.96, linewidth=0.6))

    # --- Bead row ---
    bead = _bead_indices(center_idx, N, step_deg, transition_step_deg, period)
    # P-2 column: 1 bead at bead["pm2"]
    # P-1 column: 1 bead at bead["pm1"]
    # Probe column: 3 sub-cells (E, U, O) stacked
    for col_idx in range(n_cols):
        if col_idx < 2:
            ax = fig.add_subplot(gs[3, col_idx])
            highlight_idx = bead["pm2"] if col_idx == 0 else bead["pm1"]
            plot_stimulus_bead_ring(
                ax, n_channels=N, highlight_idx=highlight_idx, cmap=cmap,
                highlight_value=0.85, highlight=False,
                scale=0.78, show_axis_labels=(col_idx == 0),
            )
            if col_idx == 0:
                ax.text(-0.30, 0.50, "ctx", transform=ax.transAxes,
                        ha="right", va="center", fontsize=9,
                        fontweight="bold", color="0.25", clip_on=False)
        else:
            sub = gs[3, col_idx].subgridspec(3, 1, hspace=0.05)
            for sub_idx, (key, hi) in enumerate([
                ("E", bead["probe_E"]),
                ("U", bead["probe_U"]),
                ("O", bead["probe_O"]),
            ]):
                ax = fig.add_subplot(sub[sub_idx, 0])
                plot_stimulus_bead_ring(
                    ax, n_channels=N, highlight_idx=hi, cmap=cmap,
                    highlight_value=0.85, highlight=True,
                    scale=0.78, show_axis_labels=False,
                )
                ax.text(-0.30, 0.50, key, transform=ax.transAxes,
                        ha="right", va="center", fontsize=9,
                        fontweight="bold", color="#9a6700", clip_on=False)

    # Bead-row label (figure-coord text — avoids overlapping the bead axes)
    fig.text(0.045, 0.04, "Stimulus bead (re-centered + CW-flipped frame)",
             ha="left", va="bottom", fontsize=9, color="0.30")

    # Shared right-side colorbar
    cbar_ax = fig.add_axes([0.93, 0.40, 0.018, 0.42])
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Mean L2/3 activity (re-centered)", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))

    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-fig", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--target-idx", type=int, default=None,
                   help="Probe presentation index. Default = seq_length-1.")
    p.add_argument("--pi-pct", type=float, default=75.0,
                   help="Pi percentile cut on the broad ¬amb pool of pass-A "
                        "target pi values.")
    p.add_argument("--exp-pred-err-max", type=float, default=5.0,
                   help="Max pred_err on Pass A (deg). Cascade widens "
                        "5°→10°→15° if n_qualifying < --min-bucket-n. Pi held.")
    p.add_argument("--step-tol", type=float, default=1.0,
                   help="Tolerance (deg) on |delta| matching transition_step.")
    p.add_argument("--min-bucket-n", type=int, default=100,
                   help="Soft floor on the qualifying subset. Triggers "
                        "exp_pred_err cascade.")
    p.add_argument("--no-flip-ccw", action="store_true", default=False,
                   help="Disable mirror-flip of CCW trials (useful for "
                        "verifying CW/CCW pooling effects).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    label = args.label or os.path.basename(args.checkpoint)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] device={device}  n_batches={args.n_batches}  seed={args.rng_seed}",
          flush=True)
    print(f"[setup] flip_ccw={not args.no_flip_ccw}", flush=True)

    records, meta = collect_records(args, device)
    print(f"[collect] n_total={meta['n_total_trials']}  "
          f"clean_march(unfiltered)={meta['n_clean_march_unfiltered']}  "
          f"amb_target={meta['n_amb_target']}  "
          f"target_idx={meta['target_idx']}", flush=True)
    print(f"[diag] context-identity max abs (should be ~0):  "
          f"P-2 A vs B = {meta['diag_max_abs_pm2_A_vs_B']:.2e}  "
          f"P-2 A vs C = {meta['diag_max_abs_pm2_A_vs_C']:.2e}  "
          f"P-1 A vs B = {meta['diag_max_abs_pm1_A_vs_B']:.2e}  "
          f"P-1 A vs C = {meta['diag_max_abs_pm1_A_vs_C']:.2e}", flush=True)

    pi_pct = float(args.pi_pct)
    exp_pred_err_max_used = float(args.exp_pred_err_max)
    widening_log: list[str] = []

    mask, pi_threshold = apply_filter(records, pi_pct, exp_pred_err_max_used)
    n_q = int(mask.sum())
    print(f"[init] pi_threshold(Q{pi_pct:g})={pi_threshold:.4f}  "
          f"exp_pred_err_max={exp_pred_err_max_used:g}°  "
          f"n_qualifying={n_q}", flush=True)

    cascade = [10.0, 15.0]
    for step in cascade:
        if n_q >= args.min_bucket_n:
            break
        msg = (f"n_q={n_q} < {args.min_bucket_n}; widening "
               f"exp_pred_err_max {exp_pred_err_max_used:g}°→{step:g}° "
               f"(pi held at Q{pi_pct:g})")
        print(f"[widen-pred] {msg}", flush=True)
        widening_log.append(msg)
        exp_pred_err_max_used = float(step)
        mask, pi_threshold = apply_filter(records, pi_pct, exp_pred_err_max_used)
        n_q = int(mask.sum())

    underpowered = n_q < args.min_bucket_n
    if underpowered:
        print(f"[CAVEAT] n_qualifying={n_q} still < {args.min_bucket_n} at "
              f"exp_pred_err_max={exp_pred_err_max_used:g}°", flush=True)

    rolled = recenter_three_passes(
        records, mask,
        center_idx=meta["center_idx"],
        flip_ccw=(not args.no_flip_ccw),
    )

    summaries = {
        name: summarise_bucket(name, records, mask, rolled,
                               center_idx=meta["center_idx"],
                               step_deg=meta["step_deg"])
        for name in ("expected", "unexpected", "omission")
    }

    # Verify P-2 / P-1 across rows are bit-identical (allclose on means).
    # (They MUST be, by construction — same trials, same shifts. Smoke check.)
    if rolled["mean_pm2"] is not None:
        # Each summary points to the SAME rolled["mean_pm2"]; allclose-self.
        identical_pm2 = np.allclose(
            np.array(summaries["expected"]["mean_pm2"]),
            np.array(summaries["unexpected"]["mean_pm2"])
        ) and np.allclose(
            np.array(summaries["expected"]["mean_pm2"]),
            np.array(summaries["omission"]["mean_pm2"])
        )
        identical_pm1 = np.allclose(
            np.array(summaries["expected"]["mean_pm1"]),
            np.array(summaries["unexpected"]["mean_pm1"])
        ) and np.allclose(
            np.array(summaries["expected"]["mean_pm1"]),
            np.array(summaries["omission"]["mean_pm1"])
        )
    else:
        identical_pm2 = identical_pm1 = False

    print(f"[verify] P-2 mean identical across rows: {identical_pm2}  "
          f"P-1 mean identical across rows: {identical_pm1}", flush=True)

    # --- Print report table ---
    print()
    print(f"[matching] pi_pct={pi_pct:g}  pi_threshold={pi_threshold:.4f}  "
          f"exp_pred_err_max={exp_pred_err_max_used:g}°  "
          f"flip_ccw={not args.no_flip_ccw}", flush=True)
    if widening_log:
        print(f"[matching] widening cascade: {widening_log}", flush=True)
    if underpowered:
        print(f"[WARN] underpowered (n_q={n_q} < {args.min_bucket_n})", flush=True)

    print()
    headers = ["row", "n", "decoder acc (95% CI)", "peak @ true",
               "total L2/3", "FWHM°", "mean pi", "mean pred_err_A"]
    rows = []
    for name in ("expected", "unexpected", "omission"):
        s = summaries[name]
        if s["n"] == 0:
            rows.append([name, "0", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        ci = s["decoder_acc_ci95"]
        ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a")
        dec_cell = (f"{s['decoder_acc']:.3f} {ci_str}"
                    if name != "omission"
                    else f"{s['decoder_acc']:.3f} {ci_str} (n/a interp.)")
        rows.append([
            name, str(s["n"]), dec_cell,
            f"{s['peak_at_true_probe']:.3f}",
            f"{s['total_probe']:.2f}",
            f"{s['fwhm_probe_deg']:.2f}",
            f"{s['mean_pi_pred_eff']:.3f}",
            f"{s['mean_pred_err_A']:.2f}°",
        ])
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in col_w))
    for r in rows:
        print(fmt.format(*r))

    # Render figure
    sub = (f"(re-centered: probe → ch {meta['center_idx']}; "
           f"3 passes share context; "
           f"pi broad-pool Q{pi_pct:g}; "
           f"exp pred_err≤{exp_pred_err_max_used:g}°; "
           f"flip_ccw={not args.no_flip_ccw})")
    title = f"Matched-probe 3-pass HMM ring figure — {label}\n{sub}"
    plot_figure(summaries, meta, args.output_fig, title=title)
    print(f"\n[fig] wrote {args.output_fig}", flush=True)

    # JSON
    result = {
        "label": label,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "meta": meta,
        "pi_pct_used": pi_pct,
        "pi_threshold": pi_threshold,
        "exp_pred_err_max_used": exp_pred_err_max_used,
        "flip_ccw": (not args.no_flip_ccw),
        "widening_cascade": widening_log,
        "min_bucket_n_floor": int(args.min_bucket_n),
        "underpowered": bool(underpowered),
        "n_qualifying": int(n_q),
        "context_identical_pm2_across_rows": bool(identical_pm2),
        "context_identical_pm1_across_rows": bool(identical_pm1),
        "buckets": summaries,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
