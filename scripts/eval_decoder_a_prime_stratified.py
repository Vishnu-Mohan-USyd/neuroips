"""Task #1/#2 — Stratified eval: Dec A vs Dec A' vs Dec C on r1r2/a1/b1/c1/e1.

Mirrors ``/tmp/task25_decoder_av_c_validation.py`` exactly (same n_trials, batch size,
seeding, strata) — but adds Dec A' (loaded from per-net ``checkpoints/decoder_a_prime_{net}.pt``)
alongside Dec A and Dec C for apples-to-apples comparison.

Strata (identical to Task #25):
  - ambiguous / clean
  - pi_low_Q1 / pi_high_Q4   (pi_pred at probe readout window)
  - low_pred_err_le5deg / high_pred_err_gt20deg (V2 prediction error at pre-probe ISI)
  - focused / routine (50/50 split per batch)
  - march_smooth / jump (circular probe-vs-prev-anchor distance > 30°)

Per-net support (Task #2): pass ``--net-name r1r2|a1|b1|c1|e1`` and per-net defaults
for the network ckpt, config, Dec A' ckpt and output path are inferred. Dec A is
loaded via back-compat fallback: ``ckpt['loss_heads']['orientation_decoder']`` →
``ckpt['decoder_state']`` (legacy ckpts) → skip.

Outputs:
  results/decoder_a_prime_stratified_eval_{net}.json
  (optionally /tmp/decA_prime_stratified_pertrial_{net}.npz with raw arrays)

Hard rules: fixed seed=42, no ex/unex manipulation, pure inference.
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
DEC_A_PRIME_CKPT = os.path.join(_REPO, "checkpoints/decoder_a_prime.pt")
DEC_C_CKPT = os.path.join(_REPO, "checkpoints/decoder_c.pt")
DEFAULT_CONFIG = os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")
OUT_JSON = os.path.join(_REPO, "results/decoder_a_prime_stratified_eval.json")
OUT_NPZ = "/tmp/decA_prime_stratified_pertrial.npz"

# Task #2: per-net routing tables. Mirror scripts/eval_decoder_e_stratified.py.
PER_NET_CKPT = {
    "r1r2": R1R2_CKPT,
    "a1":   "/tmp/remote_ckpts/a1/checkpoint.pt",
    "b1":   "/tmp/remote_ckpts/b1/checkpoint.pt",
    "c1":   "/tmp/remote_ckpts/c1/checkpoint.pt",
    "e1":   "/tmp/remote_ckpts/e1/checkpoint.pt",
}
PER_NET_CFG = {
    "r1r2": DEFAULT_CONFIG,
    **{k: os.path.join(_REPO, f"config/sweep/sweep_{k}.yaml") for k in ("a1", "b1", "c1", "e1")},
}
PER_NET_DEC_A_PRIME = {
    "r1r2": DEC_A_PRIME_CKPT,
    **{k: os.path.join(_REPO, f"checkpoints/decoder_a_prime_{k}.pt")
       for k in ("a1", "b1", "c1", "e1")},
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
        return {"top1": float("nan"), "within1": float("nan"),
                "within2": float("nan"), "within3": float("nan"),
                "mae_ch": float("nan"), "std_ch": float("nan"), "n": 0}
    return tolerance_table(pred[mask], true[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-name", default=None,
                    help="If set (r1r2|a1|b1|c1|e1), per-net Dec A' / ckpt / config / "
                         "out-json paths are inferred. Individual overrides still work.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--decoder-a-prime", default=None)
    ap.add_argument("--decoder-c", default=DEC_C_CKPT)
    ap.add_argument("--n-trials", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args()

    net_name = args.net_name or "r1r2"
    if args.ckpt is None:
        args.ckpt = PER_NET_CKPT.get(net_name, R1R2_CKPT)
    if args.config is None:
        args.config = PER_NET_CFG.get(net_name, DEFAULT_CONFIG)
    if args.decoder_a_prime is None:
        args.decoder_a_prime = PER_NET_DEC_A_PRIME.get(net_name, DEC_A_PRIME_CKPT)
    if args.out_json is None:
        args.out_json = os.path.join(
            _REPO, f"results/decoder_a_prime_stratified_eval_{net_name}.json")
    if args.out_npz is None:
        args.out_npz = f"/tmp/decA_prime_stratified_pertrial_{net_name}.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  net_name={net_name}", flush=True)

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

    # ---- Network ----
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

    # ---- Dec A (original, from ckpt) — legacy-safe fallback ----
    decA = nn.Linear(n_ori, n_ori).to(device)
    decA_state = None
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        decA_state = ckpt["loss_heads"]["orientation_decoder"]
        decA_src = "ckpt.loss_heads.orientation_decoder"
    elif "decoder_state" in ckpt:
        decA_state = ckpt["decoder_state"]
        decA_src = "ckpt.decoder_state (legacy ckpt fallback)"
    else:
        raise RuntimeError(
            f"No Dec A in ckpt {args.ckpt} (neither loss_heads nor decoder_state)")
    decA.load_state_dict(decA_state)
    decA.eval()
    print(f"[decA] loaded from {decA_src}", flush=True)

    # ---- Dec A' (trained on frozen L2/3) ----
    dap_ckpt = torch.load(args.decoder_a_prime, map_location=device, weights_only=False)
    decAp = nn.Linear(n_ori, n_ori).to(device)
    decAp.load_state_dict(dap_ckpt["state_dict"])
    decAp.eval()
    print(f"[decA'] loaded from {args.decoder_a_prime} "
          f"(trained {dap_ckpt.get('n_steps','?')} steps, "
          f"final_val_acc={dap_ckpt.get('final_val_acc','?')})", flush=True)

    # ---- Dec C ----
    dc_ckpt = torch.load(args.decoder_c, map_location=device, weights_only=False)
    decC = nn.Linear(n_ori, n_ori).to(device)
    decC.load_state_dict(dc_ckpt["state_dict"])
    decC.eval()
    print(f"[decC] loaded from {args.decoder_c}", flush=True)

    # ---- HMM generator ----
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

    all_true, all_decA, all_decAp, all_decC = [], [], [], []
    all_pi, all_pred_err_v2, all_amb = [], [], []
    all_task_state, all_jump, all_contrast, all_prev_dist = [], [], [], []

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
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net.forward(packed)
            q_pred_all = aux["q_pred_all"]
            pi_pred_all = aux.get("pi_pred_all", None)

            r_probe = r_l23_all[:, win_lo:win_hi, :].mean(dim=1)  # [B, N]

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

            decA_pred = decA(r_probe).argmax(dim=-1).cpu().numpy().astype(np.int64)
            decAp_pred = decAp(r_probe).argmax(dim=-1).cpu().numpy().astype(np.int64)
            decC_pred = decC(r_probe).argmax(dim=-1).cpu().numpy().astype(np.int64)

            is_amb = md.is_ambiguous[:, probe_idx].cpu().numpy().astype(np.bool_)
            task_state_focused = (ts_mode == 1).numpy()

            prev_ori = md.orientations[:, probe_idx - 1].cpu().numpy()
            cur_ori = md.orientations[:, probe_idx].cpu().numpy()
            d_deg = np.abs(prev_ori - cur_ori)
            d_deg = np.minimum(d_deg, period - d_deg)
            is_jump = (d_deg > 30.0)

            contrast = md.contrasts[:, probe_idx].cpu().numpy()

            all_true.append(true_ch.cpu().numpy().astype(np.int64))
            all_decA.append(decA_pred)
            all_decAp.append(decAp_pred)
            all_decC.append(decC_pred)
            all_pi.append(pi_at)
            all_pred_err_v2.append(v2_pred_err.astype(np.float64))
            all_amb.append(is_amb)
            all_task_state.append(task_state_focused)
            all_jump.append(is_jump)
            all_contrast.append(contrast)
            all_prev_dist.append(d_deg)

            if (bi + 1) % 50 == 0 or bi == 0:
                print(f"  batch {bi+1}/{n_batches} elapsed {time.time()-t0:.1f}s",
                      flush=True)

    true_ch = np.concatenate(all_true)
    decA_p = np.concatenate(all_decA)
    decAp_p = np.concatenate(all_decAp)
    decC_p = np.concatenate(all_decC)
    pi_arr = np.concatenate(all_pi)
    pred_err_v2 = np.concatenate(all_pred_err_v2)
    is_amb = np.concatenate(all_amb)
    is_focused = np.concatenate(all_task_state)
    is_jump = np.concatenate(all_jump)
    contrast = np.concatenate(all_contrast)
    prev_dist_deg = np.concatenate(all_prev_dist)
    pred_err_v2_deg = pred_err_v2 * step_deg

    print(f"\n[done] total trials={len(true_ch)} elapsed={time.time()-t0:.1f}s",
          flush=True)

    if args.out_npz:
        np.savez(args.out_npz,
                 true_ch=true_ch,
                 decA_pred=decA_p, decAp_pred=decAp_p, decC_pred=decC_p,
                 pi=pi_arr, pred_err_v2_deg=pred_err_v2_deg,
                 is_ambiguous=is_amb, is_focused=is_focused, is_jump=is_jump,
                 contrast=contrast, prev_dist_deg=prev_dist_deg,
                 step_deg=step_deg, n_ori=n_ori)
        print(f"[npz] wrote {args.out_npz}", flush=True)

    # ---------- Summary ----------
    summary = {
        "design": {
            "n_trials": int(len(true_ch)),
            "batch_size": args.batch_size,
            "seed": SEED,
            "ckpt": args.ckpt,
            "decoder_a_prime": args.decoder_a_prime,
            "decoder_c": args.decoder_c,
            "config": args.config,
            "readout_window": list(READOUT_WIN),
            "step_deg": step_deg,
            "n_ori": n_ori,
        },
        "overall": {
            "decA": tolerance_table(decA_p, true_ch),
            "decA_prime": tolerance_table(decAp_p, true_ch),
            "decC": tolerance_table(decC_p, true_ch),
        },
        "agreement": {
            "frac_same_pred_A_vs_Ap":  float((decA_p == decAp_p).mean()),
            "frac_same_pred_A_vs_C":   float((decA_p == decC_p).mean()),
            "frac_same_pred_Ap_vs_C":  float((decAp_p == decC_p).mean()),
            "mean_circ_dist_A_Ap_ch":  float(circular_distance(decA_p, decAp_p).mean()),
            "mean_circ_dist_A_C_ch":   float(circular_distance(decA_p, decC_p).mean()),
            "mean_circ_dist_Ap_C_ch":  float(circular_distance(decAp_p, decC_p).mean()),
        },
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
        "low_pred_err_le5deg":  pred_err_v2_deg <= 5.0,
        "high_pred_err_gt20deg": pred_err_v2_deg > 20.0,
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
        summary["strata"][sname] = {
            "n": int(smask.sum()),
            "decA":       stratum_table(decA_p,  true_ch, smask),
            "decA_prime": stratum_table(decAp_p, true_ch, smask),
            "decC":       stratum_table(decC_p,  true_ch, smask),
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] wrote {args.out_json}", flush=True)

    # Headline
    a = summary["overall"]["decA"]
    ap = summary["overall"]["decA_prime"]
    c = summary["overall"]["decC"]
    print("\n========== HEADLINE ==========")
    print(f"DecA  top-1: {a['top1']:.4f}  ±1: {a['within1']:.4f}  MAE_ch: {a['mae_ch']:.3f}")
    print(f"DecA' top-1: {ap['top1']:.4f}  ±1: {ap['within1']:.4f}  MAE_ch: {ap['mae_ch']:.3f}")
    print(f"DecC  top-1: {c['top1']:.4f}  ±1: {c['within1']:.4f}  MAE_ch: {c['mae_ch']:.3f}")
    print(f"Frac same pred A vs A': {summary['agreement']['frac_same_pred_A_vs_Ap']:.4f}")


if __name__ == "__main__":
    main()
