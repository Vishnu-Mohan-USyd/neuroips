#!/usr/bin/env python3
"""Matched-quality 3-row × 3-col HMM-averaged ring sequence figure (Task #35).

Extends ``matched_3row_ring.py`` (Task #34) to the matched-sequence layout
(``plot_tuning_ring_sequence.py``). For each of three structurally-defined
buckets — Expected, Unexpected, Omission — averages tuning rings at three
sequence positions (P-2, P-1, Probe), with **per-trial re-centering** so the
probe's true orientation always lands at channel ``N // 2`` (= 18 for N=36).

Sequence-structure buckets (orientation-trajectory based)
---------------------------------------------------------
This HMM's 3 latent states (CW / CCW / NEUTRAL) and high p_self produce
**marches**, not literal "stays" — under each non-NEUTRAL state, every step
advances the orientation by ±transition_step. We therefore detect structural
classes from the **signed circular orientation deltas** rather than identical-
``true_ch`` comparison (which would never match an HMM with non-zero
``transition_step``). With ``transition_step = 5°`` and ``step_deg = 5°``
(N=36, period=180°), a CW march advances by exactly 1 channel per step.

Define:
  ``delta(b,a) = ((b - a + period/2) % period) - period/2`` ∈ [-90°, 90°]
  ``ctx_dir = sign(delta(probe-1, probe-2))`` if |delta| matches transition_step
              within ``--step-tol`` (default 1.0°), else 0 (excluded)

For each batch of HMM sequences (seq_length=25), we run TWO forward passes:

Pass A (stim present, normal forward)
  Iterate ``probe_idx ∈ [2, seq_length-1]``. For each candidate:
    * Expected   ↔ ctx_dir != 0 AND
                   sign(delta(probe, probe-1)) == ctx_dir AND
                   |delta(probe, probe-1) - ctx_dir * transition_step| <= step_tol
                   → 3-step constant-direction march (P-2 → P-1 → Probe).
    * Unexpected ↔ ctx_dir != 0 AND
                   |delta(probe, probe-1)| >= --jump-min-deg (default 75°)
                   → context is a march, but probe jumps ≈90° away.

Pass B (stim zeroed at the *target* presentation only)
  Target index defaults to ``seq_length - 1`` (matches Task #34). Structure:
    * Omission ↔ ctx_dir(target-1, target-2) != 0  (i.e., 2-step march context)
                 The stim at ``target_idx`` is zeroed so the probe ring is
                 the network's response to absent sensory drive after a
                 predictable 2-step march.

CW / CCW pooling
----------------
After re-centering (rolling so probe → channel ``N//2``), CW trials place
context at channels (center-2, center-1, center) and CCW trials at
(center+2, center+1, center). To pool both directions into a single
visualisation with bumps on one side of the probe, we **mirror-flip CCW
trials** about the centre channel:
  ``rolled_flipped[j] = rolled[(2*center - j) % N]``
This is an engineering approximation that conflates CW and CCW context
patterns, justified for visualisation clarity (a single "march template").
The flip is recorded in JSON metadata.

Quality matching
----------------
``pi_pred_eff`` is read at the LAST ISI step BEFORE the relevant ON window
(``t_isi = probe_idx * (steps_on+steps_isi) - 1``). For matched quality
across the three structural buckets we use a single pooled pi threshold
(``--pi-pct``, default Q75). Ambiguous trials (at the probe slot in pass A,
target slot in pass B) are dropped before pooling.

Two pooling modes:

* **Default (Task #35 behaviour)**: Pool pi from the THREE STRUCTURALLY
  FILTERED populations (3-march, march+jump, march-context-omission) and
  take the pooled percentile.
* **``--tight-expected`` (Task #36 behaviour)**: Pool pi from the
  ``BROAD`` union of all ¬amb pass-A and ¬amb pass-B records BEFORE any
  structural filtering. This matches Task #34's matched-3row pi pool and
  prevents the threshold from drifting when downstream filters (pred_err)
  shift the structural population.

If any bucket falls below ``--min-bucket-n`` after pi filtering, the
default mode drops the pi cut to Q50 and records the widening cascade in
the JSON output.

Tight Expected mode (``--tight-expected``)
------------------------------------------
Adds two structural filters that enforce the V2 prediction quality contract:

* **Expected**: ``pred_err(Probe) ≤ --exp-pred-err-max`` (default 5°). The
  V2 head must actually predict the upcoming probe orientation correctly,
  not merely have high precision π. If ``n_expected`` falls below
  ``--min-bucket-n``, the pred_err threshold is progressively widened
  5° → 10° → 15° → 20° (pi threshold is **not** widened — Q75 is the
  matched-quality criterion). At 20° a caveat is recorded.
* **Unexpected**: ``pred_err(Probe) > --unexp-pred-err-min`` (default 60°).
  Was missing in the Task #35 implementation; Task #36 enforces it.

Omission is unchanged.

Re-centering and averaging
--------------------------
For each surviving trial we compute ``shift = (N // 2) - probe_true_ch``,
then ``np.roll(r, shift)`` is applied to the readout windows at ALL THREE
columns (P-2, P-1, Probe) using the same per-trial shift. Bucket means are
taken over the rolled rings.

Readout window: ``[9, 11]`` inclusive (last 3 ON steps of the 12-step ON
window — Stage-2 decoder's training window). Same window used at every
sequence position.

Decoder accuracy is computed at the probe slot (Pass A: vs actual probe
orientation; Pass B / Omission: vs *planned* probe orientation, flagged
"n/a interp." in the figure).

Figure layout
-------------
4 rows × 3 cols ``GridSpec``:
  Row 0 — Expected     P-2 / P-1 / Probe  (rings)
  Row 1 — Unexpected   P-2 / P-1 / Probe  (rings)
  Row 2 — Omission     P-2 / P-1 / Probe  (rings)
  Row 3 — Stimulus bead row, 3-sub-row per column, one bead per bucket per
          column. Highlight at the channel the stimulus should fall at AFTER
          per-trial re-centering:
            Expected   : P-2 18, P-1 18, Probe 18
            Unexpected : P-2  0, P-1  0, Probe 18
            Omission   : P-2 18, P-1 18, Probe none

The probe column (col 2) gets an orange outline highlight on every ring
icon. Vmax is shared across the 9 rings. A right-side colorbar shows the
L2/3 activity scale.

Outputs
-------
PNG to ``--output-fig`` and JSON stats to ``--output-json``.
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
)
from plot_tuning_ring_sequence import (
    plot_ring_icon,
    plot_stimulus_bead_ring,
)


# ---------------------------------------------------------------------------
# Data collection (two-pass)
# ---------------------------------------------------------------------------

def _readout_windows_for(probe_idx: int, steps_per: int, w_start: int, w_end: int
                         ) -> list[tuple[int, int]]:
    """Return [(t0, t1_inclusive)] for P-2, P-1, Probe given a probe index."""
    out: list[tuple[int, int]] = []
    for offset in (-2, -1, 0):
        i = probe_idx + offset
        t0 = i * steps_per + w_start
        t1 = i * steps_per + w_end
        out.append((t0, t1))
    return out


def signed_circ_delta(b: torch.Tensor, a: torch.Tensor, period: float) -> torch.Tensor:
    """Signed circular delta b - a wrapped to [-period/2, +period/2]."""
    return ((b - a + period / 2) % period) - period / 2


def collect_records(args, device: torch.device) -> tuple[dict[str, Any], dict]:
    """Run paired forward passes per batch and collect per-trial records.

    Returns
    -------
    records : dict
        "passA" : per-(trial × probe_idx) records — keys
                  "is_3stay", "is_stay_jump90", "is_amb_probe",
                  "probe_true_ch", "ctx_m1_true_ch", "ctx_m2_true_ch",
                  "r_pm2", "r_pm1", "r_probe" (each [n, N]),
                  "pi", "pred_err", "decoder_top1".
        "passB_target" : per-trial records for Pass B at fixed target_idx —
                  "is_stay_stay_target", "is_amb_target", "target_true_ch",
                  "ctx_m1_true_ch", "ctx_m2_true_ch",
                  "r_pm2", "r_pm1", "r_target", "pi",
                  "decoder_top1".
    meta : dict with N, step_deg, target_idx, etc.
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
    jump_min_deg = float(args.jump_min_deg)
    assert 0.0 <= jump_min_deg <= 90.0, \
        f"jump_min_deg={jump_min_deg} must be in [0, 90] (period=180°, max distance=90)"
    transition_step = float(stim_cfg.transition_step)
    step_tol = float(args.step_tol)
    assert step_tol > 0.0, "step_tol must be positive"

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

    # Pass A buffers (one row per (trial × probe_idx) candidate, probe_idx ∈ [2, S-1])
    A_is_3march: list[np.ndarray] = []     # 3-step constant-direction march
    A_is_march_jump: list[np.ndarray] = [] # 2-step march context + ≥jump_min_deg probe
    A_ctx_dir: list[np.ndarray] = []       # +1 CW, -1 CCW, 0 not-march context
    A_is_amb: list[np.ndarray] = []
    A_probe_true_ch: list[np.ndarray] = []
    A_r_pm2: list[np.ndarray] = []
    A_r_pm1: list[np.ndarray] = []
    A_r_probe: list[np.ndarray] = []
    A_pi: list[np.ndarray] = []
    A_pred_err: list[np.ndarray] = []
    A_dec_top1: list[np.ndarray] = []
    A_delta_probe: list[np.ndarray] = []   # signed delta(probe, probe-1) — diagnostic

    # Pass B buffers (one row per trial at fixed target_idx)
    B_is_march_ctx: list[np.ndarray] = []
    B_ctx_dir: list[np.ndarray] = []
    B_is_amb: list[np.ndarray] = []
    B_target_true_ch: list[np.ndarray] = []
    B_r_pm2: list[np.ndarray] = []
    B_r_pm1: list[np.ndarray] = []
    B_r_target: list[np.ndarray] = []
    B_pi: list[np.ndarray] = []
    B_dec_top1: list[np.ndarray] = []

    n_total_A_pres = 0
    n_total_B_seq = 0

    with torch.no_grad():
        for batch_i in range(args.n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            true_ori = metadata.orientations.to(device)        # [B, S]
            is_amb_all = metadata.is_ambiguous.to(device)      # [B, S] bool
            true_ch_all = (true_ori / step_deg).round().long() % N  # [B, S]

            # --- Pass A: forward with normal stim ---
            packed_A = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_A, _, aux_A = net.forward(packed_A)          # [B, T, N]
            q_pred_A = aux_A["q_pred_all"]                     # [B, T, N]
            pi_eff_A = aux_A["pi_pred_eff_all"]                # [B, T, 1]
            B = r_l23_A.shape[0]

            for probe_idx in range(2, seq_length):
                t_isi_last = probe_idx * steps_per - 1
                pi_isi = pi_eff_A[:, t_isi_last, 0]            # [B]
                q_pred_isi = q_pred_A[:, t_isi_last, :]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg
                actual_ori = true_ori[:, probe_idx]
                pred_err = circular_distance(pred_ori, actual_ori, period)

                # Trajectory-based classification via signed circular deltas.
                ori_pm0 = true_ori[:, probe_idx]
                ori_pm1 = true_ori[:, probe_idx - 1]
                ori_pm2 = true_ori[:, probe_idx - 2]
                d_ctx = signed_circ_delta(ori_pm1, ori_pm2, period)   # [B]
                d_probe = signed_circ_delta(ori_pm0, ori_pm1, period) # [B]

                ctx_match_step = (d_ctx.abs() - transition_step).abs() <= step_tol
                probe_match_step = (d_probe.abs() - transition_step).abs() <= step_tol
                same_dir = (torch.sign(d_ctx) == torch.sign(d_probe)) & (d_ctx.abs() > 1e-6)

                is_3march = ctx_match_step & probe_match_step & same_dir
                jump90 = circular_distance(ori_pm0, ori_pm1, period) >= jump_min_deg
                is_march_jump = ctx_match_step & jump90  # context is march, probe jumps far

                # ctx_dir: +1 CW, -1 CCW, 0 = ctx not a clean march
                ctx_dir = torch.zeros_like(d_ctx)
                ctx_dir[ctx_match_step & (d_ctx > 0)] = 1.0
                ctx_dir[ctx_match_step & (d_ctx < 0)] = -1.0

                probe_ch = true_ch_all[:, probe_idx]
                wins = _readout_windows_for(probe_idx, steps_per, W_START, W_END)
                r_pm2 = r_l23_A[:, wins[0][0]:wins[0][1] + 1, :].mean(dim=1)
                r_pm1 = r_l23_A[:, wins[1][0]:wins[1][1] + 1, :].mean(dim=1)
                r_probe = r_l23_A[:, wins[2][0]:wins[2][1] + 1, :].mean(dim=1)

                logits = decoder(r_probe)
                pred_ch = logits.argmax(dim=-1)

                is_amb = is_amb_all[:, probe_idx]
                A_is_3march.append(is_3march.cpu().numpy())
                A_is_march_jump.append(is_march_jump.cpu().numpy())
                A_ctx_dir.append(ctx_dir.cpu().numpy())
                A_is_amb.append(is_amb.cpu().numpy())
                A_probe_true_ch.append(probe_ch.cpu().numpy())
                A_r_pm2.append(r_pm2.cpu().numpy())
                A_r_pm1.append(r_pm1.cpu().numpy())
                A_r_probe.append(r_probe.cpu().numpy())
                A_pi.append(pi_isi.cpu().numpy())
                A_pred_err.append(pred_err.cpu().numpy())
                A_dec_top1.append(pred_ch.cpu().numpy())
                A_delta_probe.append(d_probe.cpu().numpy())
                n_total_A_pres += B

            # --- Pass B: zero stim at target ON window, forward again ---
            stim_B = stim_seq.clone()
            stim_B[:, target_onset:target_onset + steps_on, :] = 0.0
            packed_B = net.pack_inputs(stim_B, cue_seq, ts_seq)
            r_l23_B, _, aux_B = net.forward(packed_B)
            pi_eff_B = aux_B["pi_pred_eff_all"]                # [B, T, 1]
            pi_target_B = pi_eff_B[:, target_isi_pre, 0]       # [B]

            wins_B = _readout_windows_for(target_idx, steps_per, W_START, W_END)
            r_pm2_B = r_l23_B[:, wins_B[0][0]:wins_B[0][1] + 1, :].mean(dim=1)
            r_pm1_B = r_l23_B[:, wins_B[1][0]:wins_B[1][1] + 1, :].mean(dim=1)
            r_target_B = r_l23_B[:, wins_B[2][0]:wins_B[2][1] + 1, :].mean(dim=1)

            ori_t = true_ori[:, target_idx]
            ori_tm1 = true_ori[:, target_idx - 1]
            ori_tm2 = true_ori[:, target_idx - 2]
            d_ctx_B = signed_circ_delta(ori_tm1, ori_tm2, period)
            ctx_match_step_B = (d_ctx_B.abs() - transition_step).abs() <= step_tol
            ctx_dir_B = torch.zeros_like(d_ctx_B)
            ctx_dir_B[ctx_match_step_B & (d_ctx_B > 0)] = 1.0
            ctx_dir_B[ctx_match_step_B & (d_ctx_B < 0)] = -1.0
            is_march_ctx_B = ctx_match_step_B

            target_ch_B = true_ch_all[:, target_idx]
            is_amb_B = is_amb_all[:, target_idx]

            logits_B = decoder(r_target_B)
            pred_ch_B = logits_B.argmax(dim=-1)

            B_is_march_ctx.append(is_march_ctx_B.cpu().numpy())
            B_ctx_dir.append(ctx_dir_B.cpu().numpy())
            B_is_amb.append(is_amb_B.cpu().numpy())
            B_target_true_ch.append(target_ch_B.cpu().numpy())
            B_r_pm2.append(r_pm2_B.cpu().numpy())
            B_r_pm1.append(r_pm1_B.cpu().numpy())
            B_r_target.append(r_target_B.cpu().numpy())
            B_pi.append(pi_target_B.cpu().numpy())
            B_dec_top1.append(pred_ch_B.cpu().numpy())
            n_total_B_seq += B

    passA = {
        "is_3march": np.concatenate(A_is_3march, axis=0).astype(bool),
        "is_march_jump": np.concatenate(A_is_march_jump, axis=0).astype(bool),
        "ctx_dir": np.concatenate(A_ctx_dir, axis=0).astype(np.int8),
        "is_amb_probe": np.concatenate(A_is_amb, axis=0).astype(bool),
        "probe_true_ch": np.concatenate(A_probe_true_ch, axis=0).astype(np.int64),
        "r_pm2": np.concatenate(A_r_pm2, axis=0).astype(np.float32),
        "r_pm1": np.concatenate(A_r_pm1, axis=0).astype(np.float32),
        "r_probe": np.concatenate(A_r_probe, axis=0).astype(np.float32),
        "pi": np.concatenate(A_pi, axis=0).astype(np.float32),
        "pred_err": np.concatenate(A_pred_err, axis=0).astype(np.float32),
        "decoder_top1": np.concatenate(A_dec_top1, axis=0).astype(np.int64),
        "delta_probe": np.concatenate(A_delta_probe, axis=0).astype(np.float32),
    }
    passB = {
        "is_march_ctx": np.concatenate(B_is_march_ctx, axis=0).astype(bool),
        "ctx_dir": np.concatenate(B_ctx_dir, axis=0).astype(np.int8),
        "is_amb_target": np.concatenate(B_is_amb, axis=0).astype(bool),
        "target_true_ch": np.concatenate(B_target_true_ch, axis=0).astype(np.int64),
        "r_pm2": np.concatenate(B_r_pm2, axis=0).astype(np.float32),
        "r_pm1": np.concatenate(B_r_pm1, axis=0).astype(np.float32),
        "r_target": np.concatenate(B_r_target, axis=0).astype(np.float32),
        "pi": np.concatenate(B_pi, axis=0).astype(np.float32),
        "decoder_top1": np.concatenate(B_dec_top1, axis=0).astype(np.int64),
    }

    meta = {
        "N": int(N),
        "step_deg": float(step_deg),
        "center_idx": int(center_idx),
        "seq_length": int(seq_length),
        "batch_size": int(batch_size),
        "steps_on": int(steps_on),
        "steps_isi": int(steps_isi),
        "steps_per_pres": int(steps_per),
        "target_idx": int(target_idx),
        "target_onset_step": int(target_onset),
        "target_isi_pre_step": int(target_isi_pre),
        "jump_min_deg": float(jump_min_deg),
        "transition_step_deg": float(transition_step),
        "step_tol_deg": float(step_tol),
        "n_passA_records": int(passA["pi"].shape[0]),
        "n_passA_total_pres": int(n_total_A_pres),
        "n_passB_records": int(passB["pi"].shape[0]),
        "n_passB_total_seq": int(n_total_B_seq),
        "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "feedback_scale": 1.0,
        "readout_window": {"start": W_START, "end": W_END, "inclusive": True},
        "n_3march_passA_unmasked": int(passA["is_3march"].sum()),
        "n_3march_cw_passA_unmasked": int(((passA["is_3march"]) &
                                           (passA["ctx_dir"] == 1)).sum()),
        "n_3march_ccw_passA_unmasked": int(((passA["is_3march"]) &
                                            (passA["ctx_dir"] == -1)).sum()),
        "n_march_jump_passA_unmasked": int(passA["is_march_jump"].sum()),
        "n_march_ctx_passB_unmasked": int(passB["is_march_ctx"].sum()),
        "n_amb_passA_probe": int(passA["is_amb_probe"].sum()),
        "n_amb_passB_target": int(passB["is_amb_target"].sum()),
    }
    return {"passA": passA, "passB_target": passB}, meta


# ---------------------------------------------------------------------------
# Bucket masking, re-centering, and aggregation
# ---------------------------------------------------------------------------

def make_buckets(records: dict, pi_q_pct: float,
                 tight_expected: bool = False,
                 exp_pred_err_max: float = 5.0,
                 unexp_pred_err_min: float = 60.0,
                 ) -> tuple[dict[str, dict], float]:
    """Build the 3 structural buckets, applying pooled-Q matching on pi.

    Parameters
    ----------
    records : dict
        Output of :func:`collect_records`.
    pi_q_pct : float
        Percentile cut on pi (e.g. 75 → Q75).
    tight_expected : bool, default False
        If True (Task #36 mode):
          * Pi pool is the **broad** union of ALL ¬amb pass-A and ¬amb
            pass-B records (BEFORE any structural / pred_err filtering),
            matching Task #34's matched-3row pool.
          * Expected adds ``pred_err(Probe) <= exp_pred_err_max``.
          * Unexpected adds ``pred_err(Probe) >  unexp_pred_err_min``.
        If False (Task #35 mode, default):
          * Pi pool is the union of the three STRUCTURALLY FILTERED
            populations.
          * No pred_err filter applied.
    exp_pred_err_max : float, default 5.0
        Upper bound on Expected pred_err (deg). Only used if
        ``tight_expected=True``.
    unexp_pred_err_min : float, default 60.0
        Lower bound on Unexpected pred_err (deg). Only used if
        ``tight_expected=True``.
    """
    A = records["passA"]
    B = records["passB_target"]

    # Structure + ¬amb candidate masks
    A_keep = ~A["is_amb_probe"]
    A_3march = A["is_3march"] & A_keep
    A_jump = A["is_march_jump"] & A_keep
    B_marchctx = B["is_march_ctx"] & ~B["is_amb_target"]

    if tight_expected:
        # Broad pool: all ¬amb pass-A and ¬amb pass-B BEFORE structural filter.
        # This is the Task #34 matched-3row pi pool, applied here so the
        # threshold does not drift when pred_err filters shift bucket sizes.
        pool = np.concatenate([A["pi"][A_keep], B["pi"][~B["is_amb_target"]]],
                              axis=0)
    else:
        # Default Task #35 behaviour: pool from structurally filtered pops.
        pool = np.concatenate([A["pi"][A_3march], A["pi"][A_jump],
                               B["pi"][B_marchctx]], axis=0)
    if pool.size == 0:
        pi_threshold = float("inf")
    else:
        pi_threshold = float(np.percentile(pool, pi_q_pct))

    exp_mask = A_3march & (A["pi"] >= pi_threshold)
    unexp_mask = A_jump & (A["pi"] >= pi_threshold)
    om_mask = B_marchctx & (B["pi"] >= pi_threshold)

    if tight_expected:
        exp_mask = exp_mask & (A["pred_err"] <= float(exp_pred_err_max))
        unexp_mask = unexp_mask & (A["pred_err"] > float(unexp_pred_err_min))

    def slice_A(mask: np.ndarray, name: str) -> dict:
        return {
            "name": name,
            "n": int(mask.sum()),
            "is_omission": False,
            "pred_err": A["pred_err"][mask],
            "pi": A["pi"][mask],
            "probe_true_ch": A["probe_true_ch"][mask],
            "ctx_dir": A["ctx_dir"][mask],
            "r_pm2": A["r_pm2"][mask],
            "r_pm1": A["r_pm1"][mask],
            "r_probe": A["r_probe"][mask],
            "decoder_top1": A["decoder_top1"][mask],
        }

    om = {
        "name": "omission",
        "n": int(om_mask.sum()),
        "is_omission": True,
        "pred_err": None,
        "pi": B["pi"][om_mask],
        "probe_true_ch": B["target_true_ch"][om_mask],
        "ctx_dir": B["ctx_dir"][om_mask],
        "r_pm2": B["r_pm2"][om_mask],
        "r_pm1": B["r_pm1"][om_mask],
        "r_target": B["r_target"][om_mask],
        "decoder_top1": B["decoder_top1"][om_mask],
    }

    return {
        "expected": slice_A(exp_mask, "expected"),
        "unexpected": slice_A(unexp_mask, "unexpected"),
        "omission": om,
    }, pi_threshold


def _flip_ccw_about_center(rolled: np.ndarray, ctx_dir: np.ndarray,
                           center_idx: int) -> np.ndarray:
    """Mirror-flip CCW rows (ctx_dir == -1) about center channel.

    rolled: [n, N] — already rolled so probe → center.
    ctx_dir: [n] int8 — +1 CW, -1 CCW, 0 other.
    Returns [n, N] with CCW rows flipped, others untouched.
    """
    n, N = rolled.shape
    out = rolled.copy()
    ccw = (ctx_dir == -1)
    if not ccw.any():
        return out
    # mirror: out_ccw[j] = rolled_ccw[(2*center - j) % N]
    cols = (2 * center_idx - np.arange(N)) % N    # [N]
    out[ccw] = rolled[ccw][:, cols]
    return out


def recenter_and_average(bucket: dict, center_idx: int) -> dict[str, np.ndarray | None]:
    """Roll P-2/P-1/Probe rings by per-trial shift = center - probe_true_ch,
    then mirror-flip CCW trials about the centre so contexts pool as if all
    trials were CW marches (P-2 at center-2, P-1 at center-1, Probe at center).
    """
    n = bucket["n"]
    if n == 0:
        return {"mean_pm2": None, "mean_pm1": None, "mean_probe": None,
                "n_cw": 0, "n_ccw": 0, "n_other_dir": 0}
    probe_ch = bucket["probe_true_ch"]
    ctx_dir = bucket["ctx_dir"]
    r_pm2 = bucket["r_pm2"]
    r_pm1 = bucket["r_pm1"]
    r_probe = bucket["r_probe"] if not bucket["is_omission"] else bucket["r_target"]
    rolled_pm2 = roll_to_center(r_pm2, probe_ch, center_idx=center_idx)
    rolled_pm1 = roll_to_center(r_pm1, probe_ch, center_idx=center_idx)
    rolled_probe = roll_to_center(r_probe, probe_ch, center_idx=center_idx)
    rolled_pm2 = _flip_ccw_about_center(rolled_pm2, ctx_dir, center_idx)
    rolled_pm1 = _flip_ccw_about_center(rolled_pm1, ctx_dir, center_idx)
    rolled_probe = _flip_ccw_about_center(rolled_probe, ctx_dir, center_idx)
    return {
        "mean_pm2": rolled_pm2.mean(axis=0),
        "mean_pm1": rolled_pm1.mean(axis=0),
        "mean_probe": rolled_probe.mean(axis=0),
        "n_cw": int((ctx_dir == 1).sum()),
        "n_ccw": int((ctx_dir == -1).sum()),
        "n_other_dir": int(((ctx_dir != 1) & (ctx_dir != -1)).sum()),
    }


def summarise(bucket: dict, rolled: dict, center_idx: int, step_deg: float
              ) -> dict[str, Any]:
    n = bucket["n"]
    if n == 0:
        return {
            "n": 0, "n_cw": 0, "n_ccw": 0, "n_other_dir": 0,
            "mean_pm2": None, "mean_pm1": None, "mean_probe": None,
            "peak_at_true_probe": None,
            "total_probe": None,
            "fwhm_probe_deg": None,
            "decoder_acc": None,
            "decoder_acc_ci95": [None, None],
            "mean_pi_pred_eff": None,
            "mean_pred_err": None,
        }
    mp2 = rolled["mean_pm2"]; mp1 = rolled["mean_pm1"]; mpr = rolled["mean_probe"]
    correct = (bucket["decoder_top1"] == bucket["probe_true_ch"]).astype(np.float64)
    acc = float(correct.mean())
    lo, hi = bootstrap_acc_ci(correct)
    return {
        "n": n,
        "n_cw": int(rolled["n_cw"]),
        "n_ccw": int(rolled["n_ccw"]),
        "n_other_dir": int(rolled["n_other_dir"]),
        "mean_pm2": mp2.astype(np.float32).tolist(),
        "mean_pm1": mp1.astype(np.float32).tolist(),
        "mean_probe": mpr.astype(np.float32).tolist(),
        "peak_at_true_probe": float(mpr[center_idx]),
        "total_probe": float(mpr.sum()),
        "fwhm_probe_deg": float(fwhm_of_curve(mpr, step_deg)),
        "decoder_acc": acc,
        "decoder_acc_ci95": [lo, hi],
        "mean_pi_pred_eff": float(bucket["pi"].mean()),
        "mean_pred_err": (None if bucket["pred_err"] is None
                          else float(bucket["pred_err"].mean())),
    }


def pairwise_pi_check(buckets: dict) -> dict[str, dict]:
    names = ["expected", "unexpected", "omission"]
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = buckets[names[i]]["pi"]
            b = buckets[names[j]]["pi"]
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

def _bead_indices_for_bucket(name: str, center_idx: int, N: int,
                             transition_step_deg: float, step_deg: float
                             ) -> tuple[int | None, int | None, int | None]:
    """Bead positions per column (P-2, P-1, Probe) in the re-centered+CW-flipped
    frame. Probe always at ``center_idx``. Context columns sit at one and two
    transition_step channels before centre (CW march template).

    Expected   : (center-2step, center-1step, center)
    Unexpected : context bumps at the same march positions; probe at center
                 (the probe was 90° away in raw orientation, but re-centering
                 puts it back at centre regardless of original location).
    Omission   : context same as Expected; probe = None (no stim).
    """
    step_ch = int(round(transition_step_deg / step_deg))
    pm2 = (center_idx - 2 * step_ch) % N
    pm1 = (center_idx - 1 * step_ch) % N
    if name == "expected":
        return (pm2, pm1, center_idx)
    if name == "unexpected":
        return (pm2, pm1, center_idx)
    if name == "omission":
        return (pm2, pm1, None)
    raise ValueError(f"unknown bucket name {name}")


def plot_figure(summaries: dict[str, dict], meta: dict, fig_path: str,
                title: str) -> None:
    """4-row × 3-col matched-sequence-style HMM ring figure."""
    cmap = matplotlib.colormaps["viridis"]
    N = meta["N"]
    center_idx = meta["center_idx"]
    step_deg = meta["step_deg"]

    # Shared vmax over all 9 ring panels
    rings_per_bucket = []
    for name in ["expected", "unexpected", "omission"]:
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
        "stay-stay-stay",
        "stay-stay-jump≈90°",
        "stay-stay-(stim absent)",
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
                # Per-bucket annotation block (n / dec acc / peak / FWHM / pi)
                if s["n"] > 0:
                    if key == "omission":
                        dec_str = "n/a (no stim)"
                    else:
                        ci = s["decoder_acc_ci95"]
                        ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                                  if ci[0] is not None else "n/a")
                        dec_str = f"{s['decoder_acc']:.3f} {ci_str}"
                    ann = (f"n = {s['n']}\n"
                           f"dec acc = {dec_str}\n"
                           f"peak@true = {s['peak_at_true_probe']:.3f}\n"
                           f"Σ probe = {s['total_probe']:.2f}\n"
                           f"FWHM = {s['fwhm_probe_deg']:.1f}°\n"
                           f"mean π = {s['mean_pi_pred_eff']:.3f}")
                else:
                    ann = "n = 0"
                ax.text(-0.55, 0.02, ann, transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.8, color="0.15",
                        linespacing=1.20,
                        bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                                  edgecolor="0.82", alpha=0.96, linewidth=0.6))

    # --- Bead row: 3-sub-row per column (E / U / O) ---
    bead_indices_per_bucket = {
        name: _bead_indices_for_bucket(name, center_idx, N,
                                       meta["transition_step_deg"], step_deg)
        for name in bucket_keys
    }
    for col_idx in range(n_cols):
        sub = gs[3, col_idx].subgridspec(3, 1, hspace=0.05)
        for sub_idx, key in enumerate(bucket_keys):
            ax = fig.add_subplot(sub[sub_idx, 0])
            highlight_idx = bead_indices_per_bucket[key][col_idx]
            highlight_outline = (col_idx == n_cols - 1)
            plot_stimulus_bead_ring(
                ax, n_channels=N, highlight_idx=highlight_idx, cmap=cmap,
                highlight_value=0.85, highlight=highlight_outline,
                scale=0.78, show_axis_labels=(sub_idx == 0 and col_idx == 0),
            )
            if col_idx == 0:
                tag = key[:1].upper()
                ax.text(-0.30, 0.50, tag, transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, fontweight="bold",
                        color="#9a6700" if highlight_outline else "0.25",
                        clip_on=False)
        # Column footer: bead row label
        sub_ax = fig.add_subplot(gs[3, col_idx])
        sub_ax.axis("off")
        # Place bead label below bead column (use figure coords would be cleaner,
        # but a sub-axes text is simpler)
        if col_idx == 0:
            sub_ax.text(-0.05, -0.10, "Stimulus bead (re-centered frame)",
                        transform=sub_ax.transAxes, ha="left", va="top",
                        fontsize=9, color="0.30")

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
                   help="Pass-B target presentation index. Default = seq_length-1.")
    p.add_argument("--pi-pct", type=float, default=75.0,
                   help="Pi pooled percentile (drops to 50 if any bucket < min-bucket-n).")
    p.add_argument("--jump-min-deg", type=float, default=75.0,
                   help="Minimum circular distance for the Unexpected jump (deg, "
                        "period=180 → max 90; default 75 → any jump in [75, 90]).")
    p.add_argument("--step-tol", type=float, default=1.0,
                   help="Tolerance (deg) on |delta| matching transition_step for "
                        "march detection. Defaults to 1.0° (HMM has zero jitter, "
                        "so deltas are exact; tolerance is a safety margin).")
    p.add_argument("--min-bucket-n", type=int, default=100,
                   help="Soft floor for bucket sizes; triggers Q75→Q50 widening "
                        "(default mode) or pred_err widening (tight mode).")
    p.add_argument("--tight-expected", action="store_true", default=False,
                   help="Task #36 mode: enforce pred_err(Probe) <= --exp-pred-err-max "
                        "on Expected and pred_err(Probe) > --unexp-pred-err-min on "
                        "Unexpected; pi pool becomes the broad ¬amb union (BEFORE "
                        "structural filter) — matches Task #34's matched-3row "
                        "pool. Default off preserves Task #35 behaviour.")
    p.add_argument("--exp-pred-err-max", type=float, default=5.0,
                   help="Max Expected pred_err (deg) when --tight-expected is set. "
                        "Cascade widens 5°→10°→15°→20° if n_exp < --min-bucket-n. "
                        "Pi threshold is NOT widened (kept at --pi-pct).")
    p.add_argument("--unexp-pred-err-min", type=float, default=60.0,
                   help="Min Unexpected pred_err (deg) when --tight-expected is "
                        "set. Not part of the cascade.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    label = args.label or os.path.basename(args.checkpoint)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] device={device}  n_batches={args.n_batches}  seed={args.rng_seed}",
          flush=True)

    records, meta = collect_records(args, device)
    print(
        f"[collect] passA records: {meta['n_passA_records']}  "
        f"3march={meta['n_3march_passA_unmasked']} "
        f"(cw={meta['n_3march_cw_passA_unmasked']}, "
        f"ccw={meta['n_3march_ccw_passA_unmasked']})  "
        f"march_jump={meta['n_march_jump_passA_unmasked']}  "
        f"amb_excluded(probe)={meta['n_amb_passA_probe']}", flush=True)
    print(
        f"[collect] passB records: {meta['n_passB_records']}  "
        f"march_ctx={meta['n_march_ctx_passB_unmasked']}  "
        f"amb_excluded(target)={meta['n_amb_passB_target']}  "
        f"target_idx={meta['target_idx']}", flush=True)

    pi_pct = args.pi_pct
    widening_log: list[str] = []
    tight_caveat: str | None = None
    exp_pred_err_max_used = float(args.exp_pred_err_max)
    unexp_pred_err_min_used = float(args.unexp_pred_err_min)

    def _bk(pi_pct_: float, exp_max_: float) -> tuple[dict, float, int, int, int]:
        buckets_, pi_thr_ = make_buckets(
            records, pi_q_pct=pi_pct_,
            tight_expected=bool(args.tight_expected),
            exp_pred_err_max=exp_max_,
            unexp_pred_err_min=unexp_pred_err_min_used,
        )
        return (buckets_, pi_thr_,
                buckets_["expected"]["n"],
                buckets_["unexpected"]["n"],
                buckets_["omission"]["n"])

    buckets, pi_threshold, n_exp, n_unexp, n_om = _bk(pi_pct, exp_pred_err_max_used)
    if args.tight_expected:
        print(f"[init|tight] pi_threshold(Q{pi_pct:g})={pi_threshold:.4f}  "
              f"exp_pred_err_max={exp_pred_err_max_used:g}°  "
              f"unexp_pred_err_min={unexp_pred_err_min_used:g}°  "
              f"n_exp={n_exp}  n_unexp={n_unexp}  n_om={n_om}", flush=True)
    else:
        print(f"[init] pi_threshold(Q{pi_pct:g})={pi_threshold:.4f}  "
              f"n_exp={n_exp}  n_unexp={n_unexp}  n_om={n_om}", flush=True)

    if args.tight_expected:
        # Pred_err cascade on Expected only. Pi threshold held at --pi-pct.
        cascade = [10.0, 15.0, 20.0]
        for step in cascade:
            if n_exp >= args.min_bucket_n:
                break
            msg = (f"n_exp={n_exp} < {args.min_bucket_n}; widening "
                   f"exp_pred_err_max {exp_pred_err_max_used:g}°→{step:g}° "
                   f"(pi held at Q{pi_pct:g})")
            print(f"[widen-pred] {msg}", flush=True)
            widening_log.append(msg)
            exp_pred_err_max_used = float(step)
            buckets, pi_threshold, n_exp, n_unexp, n_om = _bk(
                pi_pct, exp_pred_err_max_used)
        if n_exp < args.min_bucket_n:
            tight_caveat = (
                f"CAVEAT: n_exp={n_exp} still < {args.min_bucket_n} at "
                f"exp_pred_err_max={exp_pred_err_max_used:g}°. The V2 "
                f"predictions in this checkpoint do not reliably land "
                f"near the march endpoint."
            )
            print(f"[CAVEAT] {tight_caveat}", flush=True)
        elif exp_pred_err_max_used >= 20.0:
            tight_caveat = (
                f"CAVEAT: required exp_pred_err_max={exp_pred_err_max_used:g}° "
                f"to reach n_exp >= {args.min_bucket_n}. V2 predictions in "
                f"this checkpoint are not reliably tight near the march "
                f"endpoint."
            )
            print(f"[CAVEAT] {tight_caveat}", flush=True)
    else:
        # Default Task #35 pi cascade.
        if min(n_exp, n_unexp, n_om) < args.min_bucket_n:
            msg = (f"min bucket n {min(n_exp, n_unexp, n_om)} < "
                   f"{args.min_bucket_n}; dropping pi cut Q{pi_pct:g}→Q50")
            print(f"[widen] {msg}", flush=True)
            widening_log.append(msg)
            pi_pct = 50.0
            buckets, pi_threshold, n_exp, n_unexp, n_om = _bk(
                pi_pct, exp_pred_err_max_used)
            print(f"[widen-1] n_exp={n_exp}  n_unexp={n_unexp}  n_om={n_om}",
                  flush=True)

    underpowered = min(n_exp, n_unexp, n_om) < args.min_bucket_n

    # Re-center and aggregate per bucket
    rolled = {name: recenter_and_average(b, center_idx=meta["center_idx"])
              for name, b in buckets.items()}
    summaries = {name: summarise(b, rolled[name],
                                 center_idx=meta["center_idx"],
                                 step_deg=meta["step_deg"])
                 for name, b in buckets.items()}

    pi_check = pairwise_pi_check(buckets)

    # --- Print report table ---
    print()
    print(f"[matching] final pi_pct = Q{pi_pct:g}  pi_threshold = {pi_threshold:.4f}",
          flush=True)
    if widening_log:
        print(f"[matching] widening cascade: {widening_log}", flush=True)
    if underpowered:
        print(f"[WARN] underpowered (min bucket n = {min(n_exp, n_unexp, n_om)} "
              f"< {args.min_bucket_n})", flush=True)

    print()
    headers = ["bucket", "n", "decoding acc (95% CI)", "peak@true probe",
               "Σ probe", "FWHM probe(°)", "mean π", "mean pred_err"]
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
            f"{s['peak_at_true_probe']:.3f}",
            f"{s['total_probe']:.2f}",
            f"{s['fwhm_probe_deg']:.2f}",
            f"{s['mean_pi_pred_eff']:.3f}", pred_err_cell,
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
        if info["mean_a"] is None:
            print(f"  {pair_name:<28s} (empty bucket)")
            continue
        ks_str = (f"D={ks['D']:.3f} p={ks['p']:.3g}"
                  if ks["D"] is not None else "scipy unavailable")
        print(f"  {pair_name:<28s} mean_a={info['mean_a']:.4f} "
              f"mean_b={info['mean_b']:.4f} pct_diff={info['mean_pct_diff']:.2f}%  "
              f"{ks_str}{warn}", flush=True)

    # Render figure
    if args.tight_expected:
        sub = (f"(re-centered: probe → ch {meta['center_idx']}; "
               f"pi broad-pool Q{pi_pct:g}; "
               f"Exp pred_err≤{exp_pred_err_max_used:g}°, "
               f"Unexp pred_err>{unexp_pred_err_min_used:g}°)")
    else:
        sub = (f"(re-centered: probe → ch {meta['center_idx']}; "
               f"pi pooled Q{pi_pct:g})")
    title = f"Matched-quality HMM ring sequence — {label}\n{sub}"
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
        "tight_expected_used": bool(args.tight_expected),
        "exp_pred_err_max_used": float(exp_pred_err_max_used),
        "unexp_pred_err_min_used": float(unexp_pred_err_min_used),
        "tight_caveat": tight_caveat,
        "widening_cascade": widening_log,
        "min_bucket_n_floor": int(args.min_bucket_n),
        "underpowered": bool(underpowered),
        "buckets": summaries,
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
