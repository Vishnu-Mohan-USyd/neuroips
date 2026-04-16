#!/usr/bin/env python3
"""Stim vs pred bump amplitudes on Unexpected trials, R1+R2 (Task #31).

Hypothesis under test
---------------------
If the trained orientation_decoder gets ~80% of Unexpected (`pred_err > 20°`)
trials right, the L2/3 stimulus bump at `true_theta_idx` should consistently
be the TALLEST point in `r_l23_window_avg` on those trials. If it is not —
i.e., the V2 prediction bump at `pred_theta_idx` often exceeds the stim bump —
then decoder accuracy on Unexpected is being driven by something other than a
"stim wins peak" mechanism.

Per-trial measurements (Unexpected `pred_err > 20°`)
----------------------------------------------------
- stim_amp     = r_l23_window_avg[true_theta_idx]
- pred_amp     = r_l23_window_avg[pred_theta_idx]
- max_amp      = r_l23_window_avg.max()
- argmax_ch    = argmax index of r_l23_window_avg
- pi_pred_eff  = V2 effective precision at the LAST ISI step before stim
- decoder_correct = decoder top-1 == true_theta_idx

For Expected (`pred_err <= 10°`): same fields, but `pred_theta_idx` is at
most 2 channels off `true_theta_idx` (step_deg = 5°), so stim_amp ≈ pred_amp.

Metrics on Unexpected
---------------------
- Distribution of stim_amp, pred_amp, margin (= stim - pred): mean, SD, p10/p50/p90
- Fraction stim_amp > pred_amp
- Fraction argmax_ch == true_theta_idx           ("stim wins peak")
- Fraction argmax_ch == pred_theta_idx           ("pred wins peak")
- Fraction argmax_ch is some_other_ch            ("noise wins peak")
- Decoder accuracy
- All of the above split by Unexpected-internal pi_pred_eff quartile (Q1..Q4)

For Expected: report stim_amp distribution only.

Reuse trial generation, network forward, and pi_pred extraction from
matched_quality_sim.py. Same seed (42), same n_batches (40), same HMM,
same [9, 11] readout window.

Don't re-center. All measurements are at known channels.
"""
from __future__ import annotations

import argparse
import json
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
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def _load_decoder(ckpt: dict, N: int, device: torch.device) -> nn.Linear:
    decoder = nn.Linear(N, N).to(device)
    if "loss_heads" in ckpt and isinstance(ckpt["loss_heads"], dict) \
            and "orientation_decoder" in ckpt["loss_heads"]:
        decoder.load_state_dict(ckpt["loss_heads"]["orientation_decoder"])
    elif "decoder_state" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state"])
    else:
        raise RuntimeError("No orientation_decoder weights found in checkpoint.")
    decoder.eval()
    return decoder


def dist_summary(arr: np.ndarray) -> dict[str, float | None]:
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(args, device: torch.device) -> tuple[dict[str, np.ndarray], dict]:
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

    bufs = {k: [] for k in (
        "pred_err", "pi_pred_eff", "stim_amp", "pred_amp", "max_amp",
        "argmax_ch", "true_ch", "pred_ch", "decoder_top1", "task_state",
    )}

    n_total = 0; n_amb = 0

    with torch.no_grad():
        for _ in range(args.n_batches):
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
            true_ori = metadata.orientations.to(device)
            is_amb_all = metadata.is_ambiguous.to(device)
            ts_meta = metadata.task_states.to(device)

            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1
                t0 = pres_i * steps_per + W_START
                t1 = pres_i * steps_per + W_END

                q_pred_isi = q_pred_all[:, t_isi_last, :]              # [B, N]
                pi_isi = pi_pred_eff_all[:, t_isi_last, 0]             # [B]
                pred_ch = q_pred_isi.argmax(dim=-1)                    # [B]
                pred_ori = pred_ch.float() * step_deg
                actual_ori = true_ori[:, pres_i]
                pred_err = circular_distance(pred_ori, actual_ori, period)

                r_l23_win = r_l23_all[:, t0:t1 + 1, :].mean(dim=1)     # [B, N]
                true_ch = (actual_ori / step_deg).round().long() % N   # [B]

                # Activity at known channels
                stim_amp = r_l23_win.gather(1, true_ch.unsqueeze(1)).squeeze(1)  # [B]
                pred_amp = r_l23_win.gather(1, pred_ch.unsqueeze(1)).squeeze(1)  # [B]
                max_amp, argmax_ch = r_l23_win.max(dim=-1)                       # [B], [B]

                # Trained decoder
                logits = decoder(r_l23_win)
                decoder_top1 = logits.argmax(dim=-1)                             # [B]

                ts_this = ts_meta[:, pres_i, :]
                regime_idx = ts_this.argmax(dim=-1)
                is_amb = is_amb_all[:, pres_i]
                keep = ~is_amb
                n_total += B
                n_amb += int(is_amb.sum().item())

                if keep.any():
                    bufs["pred_err"].append(pred_err[keep].cpu().numpy())
                    bufs["pi_pred_eff"].append(pi_isi[keep].cpu().numpy())
                    bufs["stim_amp"].append(stim_amp[keep].cpu().numpy())
                    bufs["pred_amp"].append(pred_amp[keep].cpu().numpy())
                    bufs["max_amp"].append(max_amp[keep].cpu().numpy())
                    bufs["argmax_ch"].append(argmax_ch[keep].cpu().numpy())
                    bufs["true_ch"].append(true_ch[keep].cpu().numpy())
                    bufs["pred_ch"].append(pred_ch[keep].cpu().numpy())
                    bufs["decoder_top1"].append(decoder_top1[keep].cpu().numpy())
                    bufs["task_state"].append(regime_idx[keep].cpu().numpy())

    records = {}
    dtypes = {
        "pred_err": np.float32, "pi_pred_eff": np.float32,
        "stim_amp": np.float32, "pred_amp": np.float32, "max_amp": np.float32,
        "argmax_ch": np.int64, "true_ch": np.int64, "pred_ch": np.int64,
        "decoder_top1": np.int64, "task_state": np.int64,
    }
    for k, v in bufs.items():
        records[k] = np.concatenate(v, axis=0).astype(dtypes[k])

    meta = {
        "N": int(N), "step_deg": float(step_deg),
        "seq_length": int(seq_length), "batch_size": int(batch_size),
        "steps_on": int(steps_on), "steps_isi": int(steps_isi),
        "n_total_pres_post_pres0": int(n_total),
        "n_ambiguous_excluded": int(n_amb),
        "n_records": int(records["pred_err"].shape[0]),
        "feedback_scale": 1.0, "rng_seed": int(args.rng_seed),
        "n_batches": int(args.n_batches),
        "readout_window": {"start": W_START, "end": W_END, "inclusive": True},
    }
    return records, meta


# ---------------------------------------------------------------------------
# Bucket metrics
# ---------------------------------------------------------------------------

def unexpected_bucket_metrics(records: dict[str, np.ndarray], mask: np.ndarray, label: str) -> dict[str, Any]:
    """Full metric set for an Unexpected-style bucket (where stim and pred channels differ)."""
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "n": 0}
    stim = records["stim_amp"][mask]
    pred = records["pred_amp"][mask]
    margin = stim - pred
    argmax_ch = records["argmax_ch"][mask]
    true_ch = records["true_ch"][mask]
    pred_ch = records["pred_ch"][mask]
    decoder_top1 = records["decoder_top1"][mask]
    pi = records["pi_pred_eff"][mask]
    pred_err = records["pred_err"][mask]

    frac_stim_gt_pred = float((stim > pred).mean())
    frac_argmax_eq_stim = float((argmax_ch == true_ch).mean())
    frac_argmax_eq_pred = float((argmax_ch == pred_ch).mean())
    frac_argmax_other = 1.0 - frac_argmax_eq_stim - frac_argmax_eq_pred
    decoder_acc = float((decoder_top1 == true_ch).mean())

    return {
        "label": label,
        "n": n,
        "mean_pi_pred_eff": float(pi.mean()),
        "mean_pred_err": float(pred_err.mean()),
        "stim_amp_dist": dist_summary(stim),
        "pred_amp_dist": dist_summary(pred),
        "margin_dist": dist_summary(margin),         # stim - pred
        "frac_stim_gt_pred": frac_stim_gt_pred,
        "frac_argmax_eq_stim": frac_argmax_eq_stim,
        "frac_argmax_eq_pred": frac_argmax_eq_pred,
        "frac_argmax_other": frac_argmax_other,
        "decoder_acc": decoder_acc,
    }


def expected_bucket_metrics(records: dict[str, np.ndarray], mask: np.ndarray, label: str) -> dict[str, Any]:
    """Metric set for Expected bucket: stim_amp distribution + ancillary."""
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "n": 0}
    stim = records["stim_amp"][mask]
    pred = records["pred_amp"][mask]
    argmax_ch = records["argmax_ch"][mask]
    true_ch = records["true_ch"][mask]
    decoder_top1 = records["decoder_top1"][mask]
    pi = records["pi_pred_eff"][mask]
    pred_err = records["pred_err"][mask]
    return {
        "label": label,
        "n": n,
        "mean_pi_pred_eff": float(pi.mean()),
        "mean_pred_err": float(pred_err.mean()),
        "stim_amp_dist": dist_summary(stim),
        "pred_amp_dist": dist_summary(pred),
        "frac_stim_eq_pred_channel": float((records["true_ch"][mask] == records["pred_ch"][mask]).mean()),
        "frac_argmax_eq_stim": float((argmax_ch == true_ch).mean()),
        "decoder_acc": float((decoder_top1 == true_ch).mean()),
    }


def quartile_split_unexpected(records: dict[str, np.ndarray], unexp_mask: np.ndarray) -> dict[str, Any]:
    """Split the Unexpected bucket into Q1..Q4 by Unexpected-internal pi_pred_eff."""
    pi_unexp = records["pi_pred_eff"][unexp_mask]
    if pi_unexp.size < 4:
        return {f"Q{i+1}": {"n": 0} for i in range(4)}
    edges = np.percentile(pi_unexp, [25, 50, 75])
    q1 = pi_unexp <= edges[0]
    q2 = (pi_unexp > edges[0]) & (pi_unexp <= edges[1])
    q3 = (pi_unexp > edges[1]) & (pi_unexp <= edges[2])
    q4 = pi_unexp > edges[2]

    # Apply each quartile mask to the global records via reconstruction
    unexp_indices = np.where(unexp_mask)[0]
    out = {}
    for label, qmask, edge_lo, edge_hi in [
        ("Q1", q1, float(pi_unexp.min()), float(edges[0])),
        ("Q2", q2, float(edges[0]),       float(edges[1])),
        ("Q3", q3, float(edges[1]),       float(edges[2])),
        ("Q4", q4, float(edges[2]),       float(pi_unexp.max())),
    ]:
        global_mask = np.zeros_like(unexp_mask)
        global_mask[unexp_indices[qmask]] = True
        m = unexpected_bucket_metrics(records, global_mask, f"Unexpected {label} (pi in [{edge_lo:.3f}, {edge_hi:.3f}])")
        m["pi_lo"] = edge_lo
        m["pi_hi"] = edge_hi
        out[label] = m
    out["pi_quartile_edges"] = [float(edges[0]), float(edges[1]), float(edges[2])]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_dist(d: dict | None) -> str:
    if d is None or d.get("n", 0) == 0:
        return "n/a"
    return f"mean={d['mean']:.3f}  sd={d['sd']:.3f}  p10/50/90={d['p10']:.3f}/{d['p50']:.3f}/{d['p90']:.3f}"


def print_unexpected_block(b: dict) -> None:
    print(f"  n              = {b['n']}")
    print(f"  mean pi        = {b.get('mean_pi_pred_eff'):.3f}")
    print(f"  mean pred_err  = {b.get('mean_pred_err'):.2f}°")
    print(f"  stim_amp       = {fmt_dist(b.get('stim_amp_dist'))}")
    print(f"  pred_amp       = {fmt_dist(b.get('pred_amp_dist'))}")
    print(f"  margin (s-p)   = {fmt_dist(b.get('margin_dist'))}")
    print(f"  frac stim>pred = {b.get('frac_stim_gt_pred'):.3f}")
    print(f"  frac argmax==stim = {b.get('frac_argmax_eq_stim'):.3f}")
    print(f"  frac argmax==pred = {b.get('frac_argmax_eq_pred'):.3f}")
    print(f"  frac argmax==other = {b.get('frac_argmax_other'):.3f}")
    print(f"  decoder acc    = {b.get('decoder_acc'):.3f}")


def print_expected_block(b: dict) -> None:
    print(f"  n                       = {b['n']}")
    print(f"  mean pi                 = {b.get('mean_pi_pred_eff'):.3f}")
    print(f"  mean pred_err           = {b.get('mean_pred_err'):.2f}°")
    print(f"  stim_amp                = {fmt_dist(b.get('stim_amp_dist'))}")
    print(f"  pred_amp                = {fmt_dist(b.get('pred_amp_dist'))}")
    print(f"  frac stim_ch==pred_ch   = {b.get('frac_stim_eq_pred_channel'):.3f}")
    print(f"  frac argmax==stim       = {b.get('frac_argmax_eq_stim'):.3f}")
    print(f"  decoder acc             = {b.get('decoder_acc'):.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--label", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] config={args.config}  ckpt={args.checkpoint}  device={device}  n_batches={args.n_batches}  seed={args.rng_seed}")

    records, meta = collect(args, device)
    print(f"[collect] n_records={meta['n_records']}  ambiguous_excluded={meta['n_ambiguous_excluded']}/{meta['n_total_pres_post_pres0']}")

    pred_err = records["pred_err"]
    unexp_mask = pred_err > 20.0
    exp_mask = pred_err <= 10.0

    unexp = unexpected_bucket_metrics(records, unexp_mask, "Unexpected (pred_err > 20°)")
    exp = expected_bucket_metrics(records, exp_mask, "Expected (pred_err <= 10°)")
    quartile = quartile_split_unexpected(records, unexp_mask)

    result = {
        "label": args.label or os.path.basename(args.checkpoint),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "meta": meta,
        "unexpected_pooled": unexp,
        "expected_pooled": exp,
        "unexpected_by_pi_quartile": quartile,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Console summary
    print()
    print("=== UNEXPECTED (pred_err > 20°) ===")
    print_unexpected_block(unexp)
    print()
    print("=== EXPECTED (pred_err <= 10°) ===")
    print_expected_block(exp)
    print()
    print("=== UNEXPECTED by pi_pred_eff quartile (Unexp-internal) ===")
    edges = quartile.get("pi_quartile_edges")
    if edges is not None:
        print(f"  pi quartile edges: {[round(e, 3) for e in edges]}")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        print(f"-- {q} --")
        print_unexpected_block(quartile[q])
    print()
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
