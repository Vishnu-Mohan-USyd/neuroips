"""Task #5 — Stratified eval for Dec E on 10k natural HMM.

Mirrors ``scripts/eval_decoder_a_prime_stratified.py`` but adds Dec E (from
``checkpoints/decoder_e.pt``) alongside Dec A, Dec A′, Dec C, Dec D-raw,
Dec D-shape for apples-to-apples comparison on the same 10k HMM stream
(seed 42, Task #25 design). Same strata: ambiguous/clean, pi_low_Q1/pi_high_Q4,
low_pred_err_le5deg/high_pred_err_gt20deg, focused/routine, march_smooth/jump.

NOTE: evaluation uses the same 50/50 focused/routine batching convention as
Task #25, NOT the HMM-own-stochastic distribution Dec E was trained on. This
mirrors the established Task #25 protocol; Dec E's effective performance on
its own training distribution is separately reported via its val pool.

Outputs:
  results/decoder_e_stratified_eval.json
  /tmp/decE_stratified_pertrial.npz (optional per-trial arrays)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _THIS_DIR)

from src.config import load_config, MechanismType  # noqa: F401
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


N_ORI = 36
SEQ_LENGTH = 25
READOUT_WIN = (9, 11)
SEED = 42

R1R2_CKPT = os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt")
DEC_E_CKPT = os.path.join(_REPO, "checkpoints/decoder_e_r1r2.pt")
DEC_A_PRIME_CKPT = os.path.join(_REPO, "checkpoints/decoder_a_prime.pt")
DEC_C_CKPT = os.path.join(_REPO, "checkpoints/decoder_c.pt")
DEC_D_RAW_CKPT = os.path.join(_REPO, "checkpoints/decoder_d_fbON_neutral_raw_r1r2.pt")
DEC_D_SHAPE_CKPT = os.path.join(_REPO, "checkpoints/decoder_d_fbON_neutral_shape_r1r2.pt")
DEFAULT_CONFIG = os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")
OUT_JSON = os.path.join(_REPO, "results/decoder_e_stratified_eval.json")
OUT_NPZ = "/tmp/decE_stratified_pertrial.npz"

# Per-network Dec E / Dec D paths for the --net-name arg path.
PER_NET_DEC_E = {
    "r1r2": os.path.join(_REPO, "checkpoints/decoder_e_r1r2.pt"),
    "a1":   os.path.join(_REPO, "checkpoints/decoder_e_a1.pt"),
    "b1":   os.path.join(_REPO, "checkpoints/decoder_e_b1.pt"),
    "c1":   os.path.join(_REPO, "checkpoints/decoder_e_c1.pt"),
    "e1":   os.path.join(_REPO, "checkpoints/decoder_e_e1.pt"),
}
PER_NET_DEC_D_RAW = {k: os.path.join(_REPO, f"checkpoints/decoder_d_fbON_neutral_raw_{k}.pt")
                     for k in PER_NET_DEC_E}
PER_NET_DEC_D_SHAPE = {k: os.path.join(_REPO, f"checkpoints/decoder_d_fbON_neutral_shape_{k}.pt")
                       for k in PER_NET_DEC_E}
PER_NET_CKPT = {
    "r1r2": R1R2_CKPT,
    "a1":   "/tmp/remote_ckpts/a1/checkpoint.pt",
    "b1":   "/tmp/remote_ckpts/b1/checkpoint.pt",
    "c1":   "/tmp/remote_ckpts/c1/checkpoint.pt",
    "e1":   "/tmp/remote_ckpts/e1/checkpoint.pt",
}
PER_NET_CFG = {
    "r1r2": os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml"),
    **{k: os.path.join(_REPO, f"config/sweep/sweep_{k}.yaml") for k in ["a1", "b1", "c1", "e1"]},
}


def circular_distance(a: np.ndarray, b: np.ndarray, n: int = N_ORI) -> np.ndarray:
    d = np.abs((a.astype(np.int64) - b.astype(np.int64)) % n)
    return np.minimum(d, n - d)


def tolerance_table(pred: np.ndarray, true: np.ndarray) -> dict:
    d = circular_distance(pred, true)
    return {
        "top1": float((d == 0).mean()),
        "within1": float((d <= 1).mean()),
        "within2": float((d <= 2).mean()),
        "within3": float((d <= 3).mean()),
        "mae_ch": float(d.mean()),
        "std_ch": float(d.std()),
        "n": int(len(d)),
    }


def stratum_table(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {k: float("nan") for k in ("top1", "within1", "within2", "within3",
                                           "mae_ch", "std_ch")} | {"n": 0}
    return tolerance_table(pred[mask], true[mask])


def load_decoder_linear(path: str, n_ori: int, device: torch.device) -> nn.Linear | None:
    from pathlib import Path as _Path
    if not path or not _Path(path).exists():
        return None
    d = torch.load(path, map_location=device, weights_only=False)
    sd = d.get("state_dict", d)
    dec = nn.Linear(n_ori, n_ori, bias=True).to(device)
    dec.load_state_dict(sd)
    dec.eval()
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-name", default=None,
                    help="If set (r1r2|a1|b1|c1|e1), per-net Dec E / Dec D / ckpt / "
                         "config paths are inferred. Individual --decoder-* / --ckpt / "
                         "--config overrides still work.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--decoder-e", default=None)
    ap.add_argument("--decoder-a-prime", default=DEC_A_PRIME_CKPT)
    ap.add_argument("--decoder-c", default=DEC_C_CKPT)
    ap.add_argument("--decoder-d-raw", default=None)
    ap.add_argument("--decoder-d-shape", default=None)
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args()

    # Resolve per-net defaults
    net = args.net_name or "r1r2"
    if args.ckpt is None:
        args.ckpt = PER_NET_CKPT.get(net, R1R2_CKPT)
    if args.config is None:
        args.config = PER_NET_CFG.get(net, DEFAULT_CONFIG)
    if args.decoder_e is None:
        args.decoder_e = PER_NET_DEC_E.get(net, DEC_E_CKPT)
    if args.decoder_d_raw is None:
        args.decoder_d_raw = PER_NET_DEC_D_RAW.get(net, DEC_D_RAW_CKPT)
    if args.decoder_d_shape is None:
        args.decoder_d_shape = PER_NET_DEC_D_SHAPE.get(net, DEC_D_SHAPE_CKPT)
    if args.out_json is None:
        args.out_json = os.path.join(_REPO, f"results/decoder_e_stratified_eval_{net}.json")
    if args.out_npz is None:
        args.out_npz = f"/tmp/decE_stratified_pertrial_{net}.npz"
    # Dec A' only applies to R1+R2 (trained on R1+R2 L2/3).
    if net != "r1r2":
        args.decoder_a_prime = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    model_cfg, train_cfg, stim_cfg = load_config(args.config)
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
    pre_probe_isi_idx = probe_onset - 1

    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    res = net.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[net] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
          flush=True)
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    for p in net.parameters():
        p.requires_grad_(False)

    decA = nn.Linear(n_ori, n_ori).to(device)
    decA_state = None
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        decA_state = ckpt["loss_heads"]["orientation_decoder"]
    elif "decoder_state" in ckpt:
        decA_state = ckpt["decoder_state"]
    else:
        raise RuntimeError(f"No Dec A in ckpt {args.ckpt} (neither loss_heads nor decoder_state)")
    decA.load_state_dict(decA_state)
    decA.eval()

    decE = load_decoder_linear(args.decoder_e, n_ori, device)
    decAp = load_decoder_linear(args.decoder_a_prime, n_ori, device)
    decC = load_decoder_linear(args.decoder_c, n_ori, device)
    decDr = load_decoder_linear(args.decoder_d_raw, n_ori, device)
    decDs = load_decoder_linear(args.decoder_d_shape, n_ori, device)

    print(f"[decA]  loaded from ckpt", flush=True)
    print(f"[decE]  loaded? {decE is not None} ({args.decoder_e})", flush=True)
    print(f"[decA'] loaded? {decAp is not None}", flush=True)
    print(f"[decC]  loaded? {decC is not None}", flush=True)
    print(f"[decD-raw]   loaded? {decDr is not None}", flush=True)
    print(f"[decD-shape] loaded? {decDs is not None}", flush=True)

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

    n_batches = args.n_trials // args.batch_size
    print(f"[run] {n_batches} batches × bs={args.batch_size} = {n_batches*args.batch_size} trials",
          flush=True)

    buckets = {k: [] for k in
               ["true", "A", "E", "Ap", "C", "Dr", "Ds",
                "pi", "pred_err_v2_deg", "amb", "focused", "jump", "contrast", "prev_deg"]}
    rng = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    with torch.no_grad():
        for bi in range(n_batches):
            md = gen.generate(args.batch_size, SEQ_LENGTH, generator=rng)

            # 50/50 focused/routine (Task #25 convention)
            ts_mode = (torch.arange(args.batch_size) < args.batch_size // 2).long()
            new_ts = torch.zeros_like(md.task_states)
            new_ts[..., 0] = ts_mode.float().unsqueeze(-1)
            new_ts[..., 1] = (1 - ts_mode).float().unsqueeze(-1)
            md.task_states = new_ts

            torch.manual_seed(SEED + bi)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED + bi)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                md, model_cfg, train_cfg, stim_cfg)

            packed = net.pack_inputs(stim_seq.to(device), cue_seq.to(device), ts_seq.to(device))
            r_l23_all, _, aux = net.forward(packed)
            q_pred_all = aux["q_pred_all"]
            pi_pred_all = aux.get("pi_pred_all", None)

            r_probe = r_l23_all[:, win_lo:win_hi, :].mean(dim=1)

            true_ori = md.orientations[:, probe_idx].to(device)
            true_ch = (true_ori / step_deg).round().long() % n_ori

            q_pred_isi = q_pred_all[:, pre_probe_isi_idx, :]
            v2_pred_ch = q_pred_isi.argmax(dim=-1)
            v2_pred_err = circular_distance(v2_pred_ch.cpu().numpy(),
                                            true_ch.cpu().numpy())
            if pi_pred_all is not None:
                pi_at = pi_pred_all[:, win_lo:win_hi, 0].mean(dim=1).cpu().numpy()
            else:
                pi_at = np.full(args.batch_size, np.nan, dtype=np.float64)

            buckets["true"].append(true_ch.cpu().numpy().astype(np.int64))
            buckets["A"].append(decA(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decE is not None:
                buckets["E"].append(decE(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decAp is not None:
                buckets["Ap"].append(decAp(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decC is not None:
                buckets["C"].append(decC(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decDr is not None:
                buckets["Dr"].append(decDr(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decDs is not None:
                r_shape = r_probe / (r_probe.sum(dim=1, keepdim=True) + 1e-8)
                buckets["Ds"].append(decDs(r_shape).argmax(-1).cpu().numpy().astype(np.int64))
            buckets["pi"].append(pi_at)
            buckets["pred_err_v2_deg"].append(v2_pred_err.astype(np.float64) * step_deg)
            is_amb = md.is_ambiguous[:, probe_idx].cpu().numpy().astype(np.bool_)
            buckets["amb"].append(is_amb)
            buckets["focused"].append((ts_mode == 1).numpy())
            prev_ori = md.orientations[:, probe_idx - 1].cpu().numpy()
            cur_ori = md.orientations[:, probe_idx].cpu().numpy()
            d_deg = np.abs(prev_ori - cur_ori)
            d_deg = np.minimum(d_deg, period - d_deg)
            buckets["jump"].append(d_deg > 30.0)
            buckets["contrast"].append(md.contrasts[:, probe_idx].cpu().numpy())
            buckets["prev_deg"].append(d_deg)
            if (bi + 1) % 100 == 0 or bi == 0:
                print(f"  batch {bi+1}/{n_batches} elapsed {time.time()-t0:.1f}s",
                      flush=True)

    true_ch = np.concatenate(buckets["true"])
    preds = {}
    for k in ["A", "E", "Ap", "C", "Dr", "Ds"]:
        if buckets[k]:
            preds[k] = np.concatenate(buckets[k])

    pi_arr = np.concatenate(buckets["pi"])
    pred_err_deg = np.concatenate(buckets["pred_err_v2_deg"])
    is_amb = np.concatenate(buckets["amb"])
    is_focused = np.concatenate(buckets["focused"])
    is_jump = np.concatenate(buckets["jump"])
    contrast = np.concatenate(buckets["contrast"])
    prev_deg = np.concatenate(buckets["prev_deg"])

    print(f"\n[done] total trials={len(true_ch)} elapsed={time.time()-t0:.1f}s", flush=True)

    if args.out_npz:
        np.savez(args.out_npz,
                 true_ch=true_ch,
                 **{f"{k}_pred": v for k, v in preds.items()},
                 pi=pi_arr, pred_err_v2_deg=pred_err_deg,
                 is_ambiguous=is_amb, is_focused=is_focused, is_jump=is_jump,
                 contrast=contrast, prev_deg=prev_deg,
                 step_deg=step_deg, n_ori=n_ori)
        print(f"[npz] wrote {args.out_npz}", flush=True)

    # Summary
    decoder_names = {"A": "decA", "E": "decE", "Ap": "decA_prime",
                     "C": "decC", "Dr": "decD_raw", "Ds": "decD_shape"}
    summary = {
        "design": {
            "n_trials": int(len(true_ch)),
            "batch_size": args.batch_size,
            "seed": SEED,
            "ckpt": args.ckpt,
            "decoder_e": args.decoder_e,
            "decoder_a_prime": args.decoder_a_prime,
            "decoder_c": args.decoder_c,
            "decoder_d_raw": args.decoder_d_raw,
            "decoder_d_shape": args.decoder_d_shape,
            "config": args.config,
            "readout_window": list(READOUT_WIN),
            "step_deg": step_deg,
            "n_ori": n_ori,
            "task_state_convention_at_eval": "50/50 focused/routine (Task #25)",
        },
        "overall": {decoder_names[k]: tolerance_table(v, true_ch) for k, v in preds.items()},
    }
    if "E" in preds and "A" in preds:
        summary["agreement"] = {
            "frac_same_pred_A_vs_E": float((preds["A"] == preds["E"]).mean()),
            "frac_same_pred_E_vs_C": (float((preds["E"] == preds["C"]).mean())
                                       if "C" in preds else None),
            "frac_same_pred_E_vs_Ap": (float((preds["E"] == preds["Ap"]).mean())
                                        if "Ap" in preds else None),
            "mean_circ_dist_A_E_ch": float(circular_distance(preds["A"], preds["E"]).mean()),
        }

    pi_finite = np.isfinite(pi_arr)
    if pi_finite.sum() > 100:
        pi_q1 = np.quantile(pi_arr[pi_finite], 0.25)
        pi_q3 = np.quantile(pi_arr[pi_finite], 0.75)
    else:
        pi_q1 = pi_q3 = float("nan")
    strata = {
        "ambiguous": is_amb,
        "clean": ~is_amb,
        "pi_low_Q1":  pi_finite & (pi_arr <= pi_q1) if pi_finite.any() else np.zeros_like(is_amb),
        "pi_high_Q4": pi_finite & (pi_arr >= pi_q3) if pi_finite.any() else np.zeros_like(is_amb),
        "low_pred_err_le5deg":  pred_err_deg <= 5.0,
        "high_pred_err_gt20deg": pred_err_deg > 20.0,
        "focused": is_focused,
        "routine": ~is_focused,
        "march_smooth": ~is_jump,
        "jump": is_jump,
    }
    summary["strata_thresholds"] = {
        "pi_q1": float(pi_q1) if not np.isnan(pi_q1) else None,
        "pi_q3": float(pi_q3) if not np.isnan(pi_q3) else None,
        "jump_threshold_deg": 30.0,
        "low_pred_err_threshold_deg": 5.0,
        "high_pred_err_threshold_deg": 20.0,
    }
    summary["strata"] = {}
    for sname, smask in strata.items():
        summary["strata"][sname] = {"n": int(smask.sum())} | {
            decoder_names[k]: stratum_table(preds[k], true_ch, smask) for k in preds
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] wrote {args.out_json}", flush=True)

    # Headline
    print("\n========== HEADLINE ==========")
    for k, d in summary["overall"].items():
        print(f"{k:<10s}  top-1={d['top1']:.4f}  ±1={d['within1']:.4f}  MAE_ch={d['mae_ch']:.3f}")
    if "agreement" in summary:
        a = summary["agreement"]
        print(f"frac_same_pred A vs E: {a['frac_same_pred_A_vs_E']:.4f}")
        if a.get("frac_same_pred_E_vs_Ap") is not None:
            print(f"frac_same_pred E vs A': {a['frac_same_pred_E_vs_Ap']:.4f}")
        if a.get("frac_same_pred_E_vs_C") is not None:
            print(f"frac_same_pred E vs C: {a['frac_same_pred_E_vs_C']:.4f}")


if __name__ == "__main__":
    main()
