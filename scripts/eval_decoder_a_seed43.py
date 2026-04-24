"""Task #4 — 10k HMM top-1 eval for Dec A seed 43.

For each of the 5 nets trained fresh at seed 43 (via scripts/train.py full
Stage-1 + Stage-2 pipeline), this script:

1. Loads the new seed-43 checkpoint (`checkpoints/net_seed43_{net}.pt`).
2. Extracts Dec A from `ckpt['loss_heads']['orientation_decoder']`.
3. Runs 10k HMM eval (625 batches × bs=16 × seq=25 = 250k readouts) with the
   SAME protocol used in Task #3's post-training eval: seed 42 for the eval
   stream, 50/50 focused/routine task_state, feedback_scale=1.0, readout
   window [9:11] per presentation.
4. Reports Dec A seed-43 top-1 alongside Dec A seed-42 (loaded from the
   original ckpt) for side-by-side comparison.
5. Writes `results/decoder_a_seed43_eval_{net}.json`.

Per-net usage:
    python3 scripts/eval_decoder_a_seed43.py --net-name r1r2
    python3 scripts/eval_decoder_a_seed43.py --net-name a1
    ...
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

SEED_EVAL = 42
EVAL_N_TRIALS = 10_000
EVAL_BATCH_SIZE = 16
SEQ_LENGTH = 25
READOUT_WIN = (9, 11)

DEFAULT_CONFIG = os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")

PER_NET_SEED42_CKPT = {
    "r1r2": os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt"),
    "a1":   "/tmp/remote_ckpts/a1/checkpoint.pt",
    "b1":   "/tmp/remote_ckpts/b1/checkpoint.pt",
    "c1":   "/tmp/remote_ckpts/c1/checkpoint.pt",
    "e1":   "/tmp/remote_ckpts/e1/checkpoint.pt",
}
PER_NET_SEED43_CKPT = {
    k: os.path.join(_REPO, f"checkpoints/net_seed43_{k}.pt") for k in PER_NET_SEED42_CKPT
}
PER_NET_CFG = {
    "r1r2": DEFAULT_CONFIG,
    **{k: os.path.join(_REPO, f"config/sweep/sweep_{k}.yaml") for k in ("a1", "b1", "c1", "e1")},
}


def build_generator(model_cfg, train_cfg, stim_cfg) -> HMMSequenceGenerator:
    return HMMSequenceGenerator(
        n_orientations=model_cfg.n_orientations,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=model_cfg.orientation_range,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        task_p_switch=getattr(stim_cfg, "task_p_switch", 0.0),
    )


def extract_decA(ckpt: dict, n_ori: int, device) -> nn.Linear | None:
    """Load Dec A from ckpt's loss_heads (new) or decoder_state (legacy fallback)."""
    decA_state = None
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        decA_state = ckpt["loss_heads"]["orientation_decoder"]
    elif "decoder_state" in ckpt:
        decA_state = ckpt["decoder_state"]
    if decA_state is None:
        return None
    dec = nn.Linear(n_ori, n_ori, bias=True).to(device)
    dec.load_state_dict(decA_state)
    dec.eval()
    return dec


def set_50_50_task_state(md, batch_size: int) -> None:
    ts_mode = (torch.arange(batch_size) < batch_size // 2).long()
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = ts_mode.float().unsqueeze(-1)
    new_ts[..., 1] = (1 - ts_mode).float().unsqueeze(-1)
    md.task_states = new_ts


@torch.no_grad()
def run_eval(
    net, gen, *,
    decA_s42: nn.Linear | None, decA_s43: nn.Linear,
    n_batches: int, batch_size: int, seq_length: int,
    steps_per: int, win_lo: int, win_hi: int,
    step_deg: float, n_ori: int,
    device, model_cfg, train_cfg, stim_cfg,
) -> dict:
    """Run n_batches × batch_size × seq_length readouts; return top-1 per decoder."""
    rng = torch.Generator().manual_seed(SEED_EVAL)
    correct_s42 = correct_s43 = total = 0
    t0 = time.time()
    for bi in range(n_batches):
        md = gen.generate(batch_size, seq_length, generator=rng)
        set_50_50_task_state(md, batch_size)

        torch.manual_seed(SEED_EVAL + bi)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED_EVAL + bi)
        stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
            md, model_cfg, train_cfg, stim_cfg)
        stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

        packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
        r_l23_all, _, _ = net.forward(packed)

        B, S, N = batch_size, seq_length, n_ori
        for s in range(S):
            onset = s * steps_per
            r_probe = r_l23_all[:, onset + win_lo:onset + win_hi, :].mean(dim=1)  # [B, N]
            true_ori = md.orientations[:, s].to(device)
            true_ch = (true_ori / step_deg).round().long() % n_ori  # [B]

            pred_s43 = decA_s43(r_probe).argmax(-1)
            correct_s43 += int((pred_s43 == true_ch).sum())
            if decA_s42 is not None:
                pred_s42 = decA_s42(r_probe).argmax(-1)
                correct_s42 += int((pred_s42 == true_ch).sum())
            total += int(r_probe.shape[0])

        if (bi + 1) % 100 == 0 or bi == 0:
            print(f"  batch {bi+1}/{n_batches}  elapsed={time.time()-t0:.1f}s", flush=True)
    top1_s43 = correct_s43 / max(total, 1)
    top1_s42 = (correct_s42 / max(total, 1)) if decA_s42 is not None else None
    return {
        "n_readouts": total,
        "decA_seed43_top1": top1_s43,
        "decA_seed42_top1": top1_s42,
        "gap_seed43_minus_seed42": (top1_s43 - top1_s42) if top1_s42 is not None else None,
        "wall_clock_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-name", required=True, choices=list(PER_NET_SEED42_CKPT.keys()))
    ap.add_argument("--ckpt-seed43", default=None)
    ap.add_argument("--ckpt-seed42", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    net_name = args.net_name
    if args.ckpt_seed43 is None:
        args.ckpt_seed43 = PER_NET_SEED43_CKPT[net_name]
    if args.ckpt_seed42 is None:
        args.ckpt_seed42 = PER_NET_SEED42_CKPT[net_name]
    if args.config is None:
        args.config = PER_NET_CFG[net_name]
    if args.out_json is None:
        args.out_json = os.path.join(
            _REPO, f"results/decoder_a_seed43_eval_{net_name}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  net={net_name}", flush=True)
    print(f"[setup] seed43 ckpt: {args.ckpt_seed43}", flush=True)
    print(f"[setup] seed42 ckpt: {args.ckpt_seed42}", flush=True)
    print(f"[setup] config:      {args.config}", flush=True)

    # Config
    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    steps_per = int(train_cfg.steps_on) + int(train_cfg.steps_isi)
    win_lo, win_hi = READOUT_WIN

    # Load seed-43 net (BPTT network + Dec A head)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt_s43 = torch.load(args.ckpt_seed43, map_location=device, weights_only=False)
    res = net.load_state_dict(ckpt_s43["model_state"], strict=False)
    print(f"[net_s43] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
          flush=True)
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    for p in net.parameters():
        p.requires_grad_(False)

    decA_s43 = extract_decA(ckpt_s43, n_ori, device)
    assert decA_s43 is not None, f"No Dec A in seed-43 ckpt {args.ckpt_seed43}"
    print(f"[decA_s43] loaded Linear({n_ori},{n_ori})+bias from seed-43 ckpt", flush=True)

    # Load seed-42 Dec A (from original ckpt) — note: this uses seed-42 NETWORK
    # for the decoder but we evaluate on seed-43 NETWORK's r_l23. That's a
    # head-swap eval — intentionally shows how the same decoder performs on
    # the new L2/3. For apples-to-apples Dec A_seed42_top1, we'd re-eval on
    # the seed-42 net, which Task #3 already did (0.5464 on r1r2 etc).
    # Here we prioritize the HEAD-ON eval: both decoders scored against SAME
    # r_l23 stream (seed-43 net). Orig seed-42 number reported separately
    # for comparison.
    ckpt_s42 = torch.load(args.ckpt_seed42, map_location="cpu", weights_only=False)
    decA_s42_samepool = extract_decA(ckpt_s42, n_ori, device)
    if decA_s42_samepool is not None:
        print(f"[decA_s42] loaded from original ckpt (will eval on seed-43 net's r_l23 for "
              f"same-pool comparison)", flush=True)
    else:
        print(f"[decA_s42] WARN: original ckpt lacks loss_heads and decoder_state; "
              f"skipping same-pool comparison", flush=True)

    # 10k HMM eval
    gen = build_generator(model_cfg, train_cfg, stim_cfg)
    n_batches = EVAL_N_TRIALS // EVAL_BATCH_SIZE
    print(f"\n[eval] {n_batches} batches × {EVAL_BATCH_SIZE} × {SEQ_LENGTH} = "
          f"{n_batches * EVAL_BATCH_SIZE * SEQ_LENGTH} readouts (seed={SEED_EVAL}, 50/50, FB=1.0)",
          flush=True)
    t0 = time.time()
    result_samepool = run_eval(
        net, gen,
        decA_s42=decA_s42_samepool, decA_s43=decA_s43,
        n_batches=n_batches, batch_size=EVAL_BATCH_SIZE, seq_length=SEQ_LENGTH,
        steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
        step_deg=step_deg, n_ori=n_ori,
        device=device, model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
    )
    print(f"[eval samepool done in {time.time()-t0:.1f}s]", flush=True)

    # Print headline
    print(f"\n========== HEADLINE =========")
    print(f"Net = {net_name}")
    print(f"Dec A seed-43 (new ckpt head, eval on seed-43 net r_l23):  "
          f"top1 = {result_samepool['decA_seed43_top1']:.4f}")
    if result_samepool["decA_seed42_top1"] is not None:
        print(f"Dec A seed-42 (orig ckpt head, SAME seed-43 net r_l23):    "
              f"top1 = {result_samepool['decA_seed42_top1']:.4f}")
        print(f"gap (seed43 − seed42 head, same pool):                      "
              f"{result_samepool['gap_seed43_minus_seed42']:+.4f}")

    # Dump
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "task": "Task #4 — Dec A seed 43 reproducibility eval",
            "net_name": net_name,
            "ckpt_seed43": args.ckpt_seed43,
            "ckpt_seed42": args.ckpt_seed42,
            "config": args.config,
            "eval_protocol": {
                "seed": SEED_EVAL,
                "n_trials": EVAL_N_TRIALS,
                "batch_size": EVAL_BATCH_SIZE,
                "seq_length": SEQ_LENGTH,
                "readout_window": list(READOUT_WIN),
                "task_state": "50/50 focused/routine",
                "feedback_scale": 1.0,
            },
            "samepool": result_samepool,
        }, f, indent=2)
    print(f"\n[save] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
