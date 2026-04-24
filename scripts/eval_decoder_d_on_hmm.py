"""Task #4 — Evaluate Dec D-template / D-raw / D-shape on 10k natural HMM per network.

Mirrors `scripts/build_decoder_d_template.py` eval loop but adds Dec D-raw and
Dec D-shape (trained by `scripts/train_decoder_d_neutral.py`) alongside the
template-cosine decoder and the original Dec A / Dec A′ (R1+R2 only) / Dec C.

Used for the Pass-2 headline table (per-net top-1 across 6 decoders on 10k
natural HMM, seed 42).

Outputs:
  results/decoder_d_all_eval.json     (per-network tolerance tables)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

NETWORKS = [
    ("r1r2", os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt"),
     os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")),
    ("a1", "/tmp/remote_ckpts/a1/checkpoint.pt",
     os.path.join(_REPO, "config/sweep/sweep_a1.yaml")),
    ("b1", "/tmp/remote_ckpts/b1/checkpoint.pt",
     os.path.join(_REPO, "config/sweep/sweep_b1.yaml")),
    ("c1", "/tmp/remote_ckpts/c1/checkpoint.pt",
     os.path.join(_REPO, "config/sweep/sweep_c1.yaml")),
    ("e1", "/tmp/remote_ckpts/e1/checkpoint.pt",
     os.path.join(_REPO, "config/sweep/sweep_e1.yaml")),
]

DEC_A_PRIME_CKPT = os.path.join(_REPO, "checkpoints/decoder_a_prime.pt")
DEC_C_CKPT = os.path.join(_REPO, "checkpoints/decoder_c.pt")


def circular_distance(a: np.ndarray, b: np.ndarray, n: int = N_ORI) -> np.ndarray:
    d = np.abs((a.astype(np.int64) - b.astype(np.int64)) % n)
    return np.minimum(d, n - d)


def tolerance_table(pred: np.ndarray, true: np.ndarray) -> dict:
    d = circular_distance(pred, true)
    return {
        "top1": float((d == 0).mean()),
        "within1": float((d <= 1).mean()),
        "within2": float((d <= 2).mean()),
        "mae_ch": float(d.mean()),
        "n": int(len(d)),
    }


def load_decoder_linear(path: str, n_ori: int, device: torch.device) -> nn.Linear:
    d = torch.load(path, map_location=device, weights_only=False)
    sd = d.get("state_dict", d)
    dec = nn.Linear(n_ori, n_ori, bias=True).to(device)
    dec.load_state_dict(sd)
    dec.eval()
    return dec


def load_decoder_a(ckpt: dict, n_ori: int, device: torch.device):
    dec = nn.Linear(n_ori, n_ori).to(device)
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        dec.load_state_dict(ckpt["loss_heads"]["orientation_decoder"])
    elif "decoder_state" in ckpt:
        dec.load_state_dict(ckpt["decoder_state"])
    else:
        return None
    dec.eval()
    return dec


@torch.no_grad()
def decode_cosine(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    R_n = F.normalize(R, dim=-1)
    T_n = F.normalize(T, dim=-1)
    return (R_n @ T_n.t()).argmax(dim=-1)


def eval_all(
    net, T_template, decA, decAp, decC, dec_d_raw, dec_d_shape,
    model_cfg, train_cfg, stim_cfg, device,
    n_trials: int = 10000, batch_size: int = 16,
) -> dict:
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
    n_batches = n_trials // batch_size
    rng = torch.Generator().manual_seed(SEED)

    buckets = {k: [] for k in ["true", "D_tpl", "D_raw", "D_shape", "A", "Ap", "C"]}
    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            md = gen.generate(batch_size, SEQ_LENGTH, generator=rng)
            ts_mode = (torch.arange(batch_size) < batch_size // 2).long()
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
            r_l23_all, _, _ = net.forward(packed)
            r_probe = r_l23_all[:, win_lo:win_hi, :].mean(dim=1)
            true_ori = md.orientations[:, probe_idx].to(device)
            true_ch = (true_ori / step_deg).round().long() % n_ori
            buckets["true"].append(true_ch.cpu().numpy().astype(np.int64))
            # Dec D-template (optional — set to None in FB-ON-only mode)
            if T_template is not None:
                buckets["D_tpl"].append(decode_cosine(r_probe, T_template).cpu().numpy().astype(np.int64))
            # Dec D-raw
            if dec_d_raw is not None:
                buckets["D_raw"].append(dec_d_raw(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if dec_d_shape is not None:
                r_shape = r_probe / (r_probe.sum(dim=1, keepdim=True) + 1e-8)
                buckets["D_shape"].append(dec_d_shape(r_shape).argmax(-1).cpu().numpy().astype(np.int64))
            if decA is not None:
                buckets["A"].append(decA(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decAp is not None:
                buckets["Ap"].append(decAp(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if decC is not None:
                buckets["C"].append(decC(r_probe).argmax(-1).cpu().numpy().astype(np.int64))
            if (bi + 1) % 100 == 0 or bi == 0:
                print(f"    batch {bi+1}/{n_batches} elapsed {time.time()-t0:.1f}s", flush=True)

    true_ch_arr = np.concatenate(buckets["true"])
    out = {}
    for k in ["D_tpl", "D_raw", "D_shape", "A", "Ap", "C"]:
        if buckets[k]:
            out[k] = tolerance_table(np.concatenate(buckets[k]), true_ch_arr)
    out["elapsed_s"] = float(time.time() - t0)
    out["n_trials"] = int(len(true_ch_arr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(_REPO, "results/decoder_d_all_eval.json"))
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nets", nargs="+", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    selected = [n for n in NETWORKS if args.nets is None or n[0] in args.nets]
    decAp_shared = None
    if Path(DEC_A_PRIME_CKPT).exists():
        decAp_shared = load_decoder_linear(DEC_A_PRIME_CKPT, N_ORI, device)
    decC_shared = None
    if Path(DEC_C_CKPT).exists():
        decC_shared = load_decoder_linear(DEC_C_CKPT, N_ORI, device)

    summary = {
        "label": "Task #4 — 10k HMM eval: D-template, D-raw, D-shape vs A/A'/C per network",
        "design": {
            "n_trials": args.n_trials,
            "batch_size": args.batch_size,
            "seed": SEED,
            "readout_window": list(READOUT_WIN),
        },
        "per_network": {},
    }

    for name, ckpt_path, cfg_path in selected:
        print(f"\n=== {name} ===", flush=True)
        model_cfg, train_cfg, stim_cfg = load_config(cfg_path)
        net = LaminarV1V2Network(model_cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        res = net.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"  [net] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
              flush=True)
        net.eval()
        net.oracle_mode = False
        net.feedback_scale.fill_(1.0)
        for p in net.parameters():
            p.requires_grad_(False)

        # Load Dec D-raw + D-shape (FB-ON neutral). Dec D-template is deprecated.
        raw_path = os.path.join(_REPO, f"checkpoints/decoder_d_fbON_neutral_raw_{name}.pt")
        shape_path = os.path.join(_REPO, f"checkpoints/decoder_d_fbON_neutral_shape_{name}.pt")
        T_tpl = None   # no template in the FB-ON retrain
        dec_d_raw = None
        dec_d_shape = None
        if Path(raw_path).exists():
            dec_d_raw = load_decoder_linear(raw_path, N_ORI, device)
            print(f"  [D-raw] loaded {raw_path}", flush=True)
        if Path(shape_path).exists():
            dec_d_shape = load_decoder_linear(shape_path, N_ORI, device)
            print(f"  [D-shape] loaded {shape_path}", flush=True)
        if dec_d_raw is None and dec_d_shape is None:
            print(f"  [skip] no D-raw / D-shape ckpts at {raw_path} / {shape_path}", flush=True)
            continue

        decA = load_decoder_a(ckpt, N_ORI, device)
        decAp_for_net = decAp_shared if name == "r1r2" else None

        ev = eval_all(
            net, T_tpl, decA, decAp_for_net, decC_shared, dec_d_raw, dec_d_shape,
            model_cfg, train_cfg, stim_cfg, device,
            n_trials=args.n_trials, batch_size=args.batch_size,
        )
        summary["per_network"][name] = ev
        parts = []
        if "D_tpl" in ev: parts.append(f"D_tpl={ev['D_tpl']['top1']:.4f}")
        if "D_raw" in ev: parts.append(f"D_raw={ev['D_raw']['top1']:.4f}")
        if "D_shape" in ev: parts.append(f"D_shape={ev['D_shape']['top1']:.4f}")
        if "A" in ev: parts.append(f"A={ev['A']['top1']:.4f}")
        if "Ap" in ev: parts.append(f"A'={ev['Ap']['top1']:.4f}")
        if "C" in ev: parts.append(f"C={ev['C']['top1']:.4f}")
        print(f"  [eval @{name}] " + "  ".join(parts), flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[json] wrote {args.out_json}", flush=True)

    # Headline (D_tpl column optional — omitted in FB-ON-only mode)
    print("\n========== HEADLINE: top-1 per network / decoder (10k HMM) ==========")
    header_cols = ['net', 'D_raw', 'D_shape', 'A', 'A′', 'C']
    print("  ".join(f"{c:>7s}" for c in header_cols))
    for name, _, _ in selected:
        if name not in summary["per_network"]:
            continue
        e = summary["per_network"][name]
        def g(k): return (f"{e[k]['top1']:>7.4f}" if k in e else f"{'—':>7s}")
        print(f"{name:<7s}  {g('D_raw')}  {g('D_shape')}  {g('A')}  {g('Ap')}  {g('C')}")


if __name__ == "__main__":
    main()
