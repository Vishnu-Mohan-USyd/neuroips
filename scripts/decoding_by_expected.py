#!/usr/bin/env python3
"""Decoding accuracy split by expected vs unexpected trials (Task #20).

Loads a trained checkpoint, replays HMM sequences with feedback ON, and
classifies each post-first presentation as EXPECTED or UNEXPECTED using
the V2 prediction from the last ISI timestep (same classifier as
`scripts/debug_expected_vs_unexpected.py`). Then runs the trained
orientation decoder on r_l23 averaged over the Stage-2 training readout
window [9, 11] and records top-1 correctness + softmax top-1 margin per
trial. Aggregates overall accuracy and per-task_state (focused vs
routine) breakdown, plus a two-proportion z-test on Δ accuracy.

Inputs
------
--config       Path to the sweep YAML for this checkpoint (must match the
               architecture the checkpoint was saved from).
--checkpoint   Path to the `.pt` checkpoint produced by `scripts/train.py`.
--output       Path for the JSON summary output.

Output JSON schema
------------------
{
  "label": <str>,
  "checkpoint": <str>,
  "config": <str>,
  "feedback_scale": 1.0,
  "readout_window": {"start": 9, "end": 11, "inclusive": true},
  "buckets": {
      "expected":   {"n": int, "acc": float, "top1_margin": float},
      "unexpected": {"n": int, "acc": float, "top1_margin": float},
  },
  "delta_acc": float,            # acc_expected - acc_unexpected
  "z_test": {"z": float, "p": float},  # two-proportion, two-sided
  "by_task_state": {
      "focused":  {"expected": {...}, "unexpected": {...}},
      "routine":  {"expected": {...}, "unexpected": {...}},
  },
  "n_total_classified_trials": int
}

Design notes
------------
* Uses the Stage-2 trained readout window [t=9, t=11] inclusive, averaged
  — same window the orientation_decoder was trained on (last 3 ON steps
  of the 12-step ON period, from stage2_feedback.py:225-226).
* `feedback_scale = 1.0` matches the user's "FB ON only" constraint.
* Ambiguous trials excluded (same as debug_expected_vs_unexpected).
* Seed 42 (matches `debug_expected_vs_unexpected.py` default), 10 batches
  of 32 × 25 trials by default — ~8000 presentations before excl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def circular_distance(a: torch.Tensor, b: torch.Tensor, period: float = 180.0) -> torch.Tensor:
    """Absolute circular distance on [0, period)."""
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> dict | None:
    """Two-sided two-proportion z-test.

    Returns {'z': float, 'p': float} or None if inputs are degenerate.
    Pooled-variance formulation: se = sqrt(p*(1-p)*(1/n1 + 1/n2)),
    p = (k1 + k2) / (n1 + n2), z = (p1 - p2) / se.
    Two-sided p = erfc(|z| / sqrt(2)).
    """
    if n1 <= 0 or n2 <= 0:
        return None
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if var <= 0.0:
        return {"z": 0.0, "p": 1.0}
    z = (p1 - p2) / math.sqrt(var)
    pval = math.erfc(abs(z) / math.sqrt(2.0))
    return {"z": float(z), "p": float(pval)}


def _load_decoder(ckpt: dict, N: int, device: torch.device) -> nn.Linear:
    """Instantiate an nn.Linear(N, N) and load the trained orientation decoder.

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


def _summary(subset: list[dict]) -> dict:
    """Aggregate {n, acc, top1_margin} on a list of per-trial records."""
    n = len(subset)
    if n == 0:
        return {"n": 0, "acc": None, "top1_margin": None}
    acc = sum(int(r["correct"]) for r in subset) / n
    marg = sum(r["margin"] for r in subset) / n
    return {"n": int(n), "acc": float(acc), "top1_margin": float(marg)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="Path to sweep YAML matching the checkpoint's architecture.")
    p.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt file.")
    p.add_argument("--output", required=True, help="Path for JSON output.")
    p.add_argument("--label", default="", help="Human label for this run (default: checkpoint basename).")
    p.add_argument("--device", default=None, help="Torch device; default cuda if available, else cpu.")
    p.add_argument("--rng-seed", type=int, default=42, help="Seed for HMM stimulus generation (default 42).")
    p.add_argument("--n-batches", type=int, default=10, help="Number of batches of (batch_size x seq_length) trials (default 10).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Config + model ---
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

    # --- Decoder (orientation_decoder) ---
    decoder = _load_decoder(ckpt, N, device)

    # --- Trial / window config ---
    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    # Stage-2 training readout window: last 3 ON steps (matching
    # stage2_feedback.py:225-226: window_start=max(0, steps_on-3)=9,
    # window_end=steps_on-1=11, inclusive).
    W_START, W_END = 9, 11
    assert W_END < steps_on, (
        f"Readout window [{W_START}, {W_END}] must fall inside steps_on={steps_on}"
    )

    # --- HMM stimulus generator (same args as debug_expected_vs_unexpected.py) ---
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

    records: list[dict] = []
    rng = torch.Generator().manual_seed(args.rng_seed)

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
            r_l23_all, _, aux = net.forward(packed)  # [B, T, N]
            q_pred_all = aux["q_pred_all"]            # [B, T, N]

            B = r_l23_all.shape[0]
            # Window-averaged r_l23 per presentation: reshape → slice → mean
            r_l23_win = (
                r_l23_all.reshape(B, seq_length, steps_per, N)
                         [:, :, W_START:W_END + 1, :]
                         .mean(dim=2)
            )  # [B, S, N]

            logits = decoder(r_l23_win.reshape(B * seq_length, N))    # [B*S, N]
            probs = F.softmax(logits, dim=-1)
            pred_ch = logits.argmax(dim=-1).reshape(B, seq_length)    # [B, S]

            # Top-1 margin over softmax (p_top1 - p_top2)
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            top1_margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).reshape(B, seq_length)

            # Target channel: same binning as loss_fn._theta_to_channel
            true_ori = metadata.orientations.to(device)                 # [B, S]
            true_ch = (true_ori / step_deg).round().long() % N           # [B, S]
            correct = (pred_ch == true_ch)                               # [B, S] bool

            # Per-presentation classification (expected / unexpected / middle)
            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1
                q_pred_isi = q_pred_all[:, t_isi_last, :]                # [B, N]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg              # [B]
                actual_ori = true_ori[:, pres_i]                         # [B]
                is_amb = metadata.is_ambiguous[:, pres_i].to(device)     # [B] bool
                pred_error = circular_distance(pred_ori, actual_ori, period)  # [B]

                # task_state one-hot: [..., 0] = focused, [..., 1] = routine
                ts_this = metadata.task_states[:, pres_i].to(device)     # [B, 2]
                regime_idx = ts_this.argmax(dim=-1)                      # [B] long

                # Bucket each sample
                is_exp = (pred_error <= 10.0) & (~is_amb)
                is_unexp = (pred_error > 20.0) & (~is_amb)

                exp_idx = is_exp.nonzero(as_tuple=False).flatten().tolist()
                unexp_idx = is_unexp.nonzero(as_tuple=False).flatten().tolist()

                for b in exp_idx:
                    records.append({
                        "bucket": "expected",
                        "correct": bool(correct[b, pres_i].item()),
                        "margin": float(top1_margin[b, pres_i].item()),
                        "regime": int(regime_idx[b].item()),
                    })
                for b in unexp_idx:
                    records.append({
                        "bucket": "unexpected",
                        "correct": bool(correct[b, pres_i].item()),
                        "margin": float(top1_margin[b, pres_i].item()),
                        "regime": int(regime_idx[b].item()),
                    })

    # ---- Aggregate ----
    def filt(pred: Callable[[dict], bool]) -> list[dict]:
        return [r for r in records if pred(r)]

    exp_recs = filt(lambda r: r["bucket"] == "expected")
    unexp_recs = filt(lambda r: r["bucket"] == "unexpected")
    exp_sum = _summary(exp_recs)
    unexp_sum = _summary(unexp_recs)

    # Per-regime
    by_regime = {}
    for name, idx in [("focused", 0), ("routine", 1)]:
        by_regime[name] = {
            "expected": _summary(filt(lambda r, idx=idx: r["bucket"] == "expected" and r["regime"] == idx)),
            "unexpected": _summary(filt(lambda r, idx=idx: r["bucket"] == "unexpected" and r["regime"] == idx)),
        }

    # Significance
    exp_corr = sum(1 for r in exp_recs if r["correct"])
    unexp_corr = sum(1 for r in unexp_recs if r["correct"])
    ztest = two_prop_z(exp_corr, exp_sum["n"], unexp_corr, unexp_sum["n"])

    delta_acc = None
    if exp_sum["acc"] is not None and unexp_sum["acc"] is not None:
        delta_acc = exp_sum["acc"] - unexp_sum["acc"]

    result = {
        "label": args.label or os.path.basename(args.checkpoint),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "feedback_scale": 1.0,
        "readout_window": {"start": W_START, "end": W_END, "inclusive": True},
        "buckets": {"expected": exp_sum, "unexpected": unexp_sum},
        "delta_acc": delta_acc,
        "z_test": ztest,
        "by_task_state": by_regime,
        "n_total_classified_trials": len(records),
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Also stdout a one-line summary for terse sweep logs
    print(f"[{result['label']}] "
          f"exp={exp_sum['n']}  acc_exp={exp_sum['acc']}  "
          f"unexp={unexp_sum['n']}  acc_unexp={unexp_sum['acc']}  "
          f"Δ={delta_acc}  "
          f"z={(ztest or {}).get('z')}  p={(ztest or {}).get('p')}")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
