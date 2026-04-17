#!/usr/bin/env python3
"""Debugger instrumentation for Task #6 (HMM side).

Reproduces matched_hmm_ring_sequence's Pass A readout and saves per-trial
decoder outputs + r_probe so quantisation behavior can be compared against
the clean-march pipeline (_debug_clean_march_per_trial.py).

Also runs an FB-OFF variant for direct comparison with the clean-march
FB-OFF Pass B.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence

from matched_quality_sim import _load_decoder, circular_distance
from matched_hmm_ring_sequence import signed_circ_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--fb-off", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(0.0 if args.fb_off else 1.0)

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

    bufs = {k: [] for k in [
        "probe_true_ch", "probe_true_ori",
        "ctx_m1_true_ch", "ctx_m2_true_ch",
        "is_amb_probe",
        "pi_probe", "pred_err_probe",
        "r_probe",
        "dec",
        "is_3march", "is_march_jump",
        "pred_argmax_probe",
    ]}

    # Collect at EVERY presentation slot after ctx has stabilised (S >= 2).
    # matched_hmm_ring_sequence uses probe_idx sweep over range(S_WARMUP, seq_length)
    # where S_WARMUP is typically 2 (needs 2 context frames for is_3march).
    S_WARMUP = 2

    with torch.no_grad():
        for batch_i in range(args.n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)
            true_ori = metadata.orientations.to(device)
            is_amb = metadata.is_ambiguous.to(device)
            true_ch_all = (true_ori / step_deg).round().long() % N
            B, S = true_ori.shape

            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23, _, aux = net.forward(packed)
            q_pred = aux["q_pred_all"]
            pi_eff = aux["pi_pred_eff_all"]

            for probe_idx in range(S_WARMUP, S):
                # Window
                w_start = probe_idx * steps_per + W_START
                w_end = probe_idx * steps_per + W_END
                r_probe = r_l23[:, w_start:w_end + 1, :].mean(dim=1)
                dec = decoder(r_probe).argmax(dim=-1)

                probe_ch = true_ch_all[:, probe_idx]
                probe_ori = true_ori[:, probe_idx]
                ctx_m1 = true_ch_all[:, probe_idx - 1]
                ctx_m2 = true_ch_all[:, probe_idx - 2]
                amb_probe = is_amb[:, probe_idx]

                # Context structure checks (3-march and march_jump like matched_hmm_ring_sequence)
                ori_m2 = true_ori[:, probe_idx - 2]
                ori_m1 = true_ori[:, probe_idx - 1]
                ori_p0 = probe_ori
                d_ctx = signed_circ_delta(ori_m1, ori_m2, period)
                d_probe = signed_circ_delta(ori_p0, ori_m1, period)
                t_step = float(stim_cfg.transition_step)
                tol = 1.0
                ctx_match = (d_ctx.abs() - t_step).abs() <= tol
                probe_match_step = (d_probe.abs() - t_step).abs() <= tol
                same_dir = (torch.sign(d_ctx) == torch.sign(d_probe)) & (d_ctx.abs() > 1e-6)
                is_3march = ctx_match & probe_match_step & same_dir

                # march_jump: ctx is march, probe deviates ~90° (jump_min_deg is typically 80-90°)
                jump_min = float(getattr(stim_cfg, 'jump_min_deg', 80.0))
                is_jump = ctx_match & (d_probe.abs() >= jump_min - 1.0) & (d_probe.abs() <= 100.0)

                pi_probe = pi_eff[:, probe_idx * steps_per - 1, 0]
                q_pred_ti = q_pred[:, probe_idx * steps_per - 1, :]
                pred_argmax = q_pred_ti.argmax(dim=-1)
                pred_ori = pred_argmax.float() * step_deg
                pred_err = circular_distance(pred_ori, probe_ori, period)

                bufs["probe_true_ch"].append(probe_ch.cpu().numpy())
                bufs["probe_true_ori"].append(probe_ori.cpu().numpy())
                bufs["ctx_m1_true_ch"].append(ctx_m1.cpu().numpy())
                bufs["ctx_m2_true_ch"].append(ctx_m2.cpu().numpy())
                bufs["is_amb_probe"].append(amb_probe.cpu().numpy())
                bufs["pi_probe"].append(pi_probe.cpu().numpy())
                bufs["pred_err_probe"].append(pred_err.cpu().numpy())
                bufs["r_probe"].append(r_probe.cpu().numpy())
                bufs["dec"].append(dec.cpu().numpy())
                bufs["is_3march"].append(is_3march.cpu().numpy())
                bufs["is_march_jump"].append(is_jump.cpu().numpy())
                bufs["pred_argmax_probe"].append(pred_argmax.cpu().numpy())

            if (batch_i + 1) % 10 == 0 or batch_i == 0:
                print(f"[batch {batch_i + 1}/{args.n_batches}]", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in bufs.items()}
    out["meta_N"] = np.array([N])
    out["meta_step_deg"] = np.array([step_deg])
    out["meta_center_idx"] = np.array([N // 2])
    out["meta_period"] = np.array([period])
    out["meta_fb_off"] = np.array([int(bool(args.fb_off))])

    np.savez_compressed(args.output, **out)
    print(f"[done] saved {args.output}  n_records={len(out['dec'])}", flush=True)


if __name__ == "__main__":
    main()
