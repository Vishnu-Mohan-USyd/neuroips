#!/usr/bin/env python3
"""Debugger instrumentation for Task #6.

Reproduces matched_probe_3pass Pass A/B/C on R1+R2, saves PER-TRIAL arrays,
and adds an FB-OFF Pass B variant. Intended to be run REMOTELY on reuben-ml
(where the R1+R2 emergent_seed42 checkpoint lives). Output is a single .npz.

Tests:
  - Does decoder_top1_B pick target_true_ch (march_dest / prediction)
    or unexp_ch (actual +90° probe) on Unexpected trials?
  - Does the per-trial r_probe_B profile peak at march_dest or at unexp_ch?
  - Does feedback_scale=0 during Pass B change dec_acc against unexp_ch?
  - Same dist of pi_target and pred_err, same qualifying mask.
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
from src.stimulus.gratings import generate_grating
from src.training.trainer import build_stimulus_sequence

from matched_quality_sim import (
    circular_distance,
    _load_decoder,
    roll_to_center,
)
from matched_hmm_ring_sequence import signed_circ_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batches", type=int, default=40)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

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

    target_idx = seq_length - 1
    target_onset = target_idx * steps_per
    target_isi_pre = target_idx * steps_per - 1

    win_pm2 = ((target_idx - 2) * steps_per + W_START, (target_idx - 2) * steps_per + W_END)
    win_pm1 = ((target_idx - 1) * steps_per + W_START, (target_idx - 1) * steps_per + W_END)
    win_target = (target_idx * steps_per + W_START, target_idx * steps_per + W_END)

    transition_step = float(stim_cfg.transition_step)
    step_tol = 1.0

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
        "is_clean_march", "is_amb", "ctx_dir", "target_true_ch", "target_true_ori",
        "pi_target", "pred_err_A", "unexp_ori", "unexp_ch",
        "r_probe_A", "r_probe_B", "r_probe_C", "r_probe_B_fboff",
        "dec_A", "dec_B", "dec_C", "dec_B_fboff",
        "r_pm2_A", "r_pm1_A",
        "q_pred_argmax_target_isi",
    ]}

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
            is_amb_all = metadata.is_ambiguous.to(device)
            true_ch_all = (true_ori / step_deg).round().long() % N
            contrasts = metadata.contrasts.to(device)
            B = stim_seq.shape[0]

            # Pass A (original, FB on)
            net.feedback_scale.fill_(1.0)
            packed_A = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_A, _, aux_A = net.forward(packed_A)
            q_pred_A = aux_A["q_pred_all"]
            pi_eff_A = aux_A["pi_pred_eff_all"]

            ori_pm0 = true_ori[:, target_idx]
            ori_pm1 = true_ori[:, target_idx - 1]
            ori_pm2 = true_ori[:, target_idx - 2]
            d_ctx = signed_circ_delta(ori_pm1, ori_pm2, period)
            d_probe = signed_circ_delta(ori_pm0, ori_pm1, period)
            ctx_match_step = (d_ctx.abs() - transition_step).abs() <= step_tol
            probe_match_step = (d_probe.abs() - transition_step).abs() <= step_tol
            same_dir = (torch.sign(d_ctx) == torch.sign(d_probe)) & (d_ctx.abs() > 1e-6)
            is_clean_march = ctx_match_step & probe_match_step & same_dir

            ctx_dir = torch.zeros_like(d_ctx)
            ctx_dir[ctx_match_step & (d_ctx > 0)] = 1.0
            ctx_dir[ctx_match_step & (d_ctx < 0)] = -1.0

            pi_target = pi_eff_A[:, target_isi_pre, 0]
            q_pred_target_isi = q_pred_A[:, target_isi_pre, :]
            pred_peak_idx = q_pred_target_isi.argmax(dim=-1)
            pred_ori = pred_peak_idx.float() * step_deg
            actual_ori = true_ori[:, target_idx]
            pred_err = circular_distance(pred_ori, actual_ori, period)

            target_ch = true_ch_all[:, target_idx]
            is_amb_target = is_amb_all[:, target_idx]

            # Pass B (+90° probe, FB on)
            unexp_ori = (actual_ori + period / 2.0) % period
            grating_B = generate_grating(
                unexp_ori.cpu(), contrasts[:, target_idx].cpu(),
                n_orientations=N, sigma=model_cfg.sigma_ff,
                n=model_cfg.naka_rushton_n, c50=model_cfg.naka_rushton_c50,
                period=period,
            ).to(device)
            stim_B = stim_seq.clone()
            stim_B[:, target_onset:target_onset + steps_on, :] = grating_B.unsqueeze(1)
            packed_B = net.pack_inputs(stim_B, cue_seq, ts_seq)
            r_l23_B, _, _ = net.forward(packed_B)

            # Pass C (zero probe, FB on)
            stim_C = stim_seq.clone()
            stim_C[:, target_onset:target_onset + steps_on, :] = 0.0
            packed_C = net.pack_inputs(stim_C, cue_seq, ts_seq)
            r_l23_C, _, _ = net.forward(packed_C)

            # Pass B (FB OFF)
            net.feedback_scale.fill_(0.0)
            packed_B_off = net.pack_inputs(stim_B, cue_seq, ts_seq)
            r_l23_B_off, _, _ = net.forward(packed_B_off)
            net.feedback_scale.fill_(1.0)

            r_pm2 = r_l23_A[:, win_pm2[0]:win_pm2[1] + 1, :].mean(dim=1)
            r_pm1 = r_l23_A[:, win_pm1[0]:win_pm1[1] + 1, :].mean(dim=1)
            r_probe_A = r_l23_A[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)
            r_probe_B = r_l23_B[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)
            r_probe_C = r_l23_C[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)
            r_probe_B_off = r_l23_B_off[:, win_target[0]:win_target[1] + 1, :].mean(dim=1)

            dec_A = decoder(r_probe_A).argmax(dim=-1)
            dec_B = decoder(r_probe_B).argmax(dim=-1)
            dec_C = decoder(r_probe_C).argmax(dim=-1)
            dec_B_off = decoder(r_probe_B_off).argmax(dim=-1)

            unexp_ch = (unexp_ori / step_deg).round().long() % N

            bufs["is_clean_march"].append(is_clean_march.cpu().numpy())
            bufs["is_amb"].append(is_amb_target.cpu().numpy())
            bufs["ctx_dir"].append(ctx_dir.cpu().numpy())
            bufs["target_true_ch"].append(target_ch.cpu().numpy())
            bufs["target_true_ori"].append(actual_ori.cpu().numpy())
            bufs["pi_target"].append(pi_target.cpu().numpy())
            bufs["pred_err_A"].append(pred_err.cpu().numpy())
            bufs["unexp_ori"].append(unexp_ori.cpu().numpy())
            bufs["unexp_ch"].append(unexp_ch.cpu().numpy())
            bufs["r_probe_A"].append(r_probe_A.cpu().numpy())
            bufs["r_probe_B"].append(r_probe_B.cpu().numpy())
            bufs["r_probe_C"].append(r_probe_C.cpu().numpy())
            bufs["r_probe_B_fboff"].append(r_probe_B_off.cpu().numpy())
            bufs["dec_A"].append(dec_A.cpu().numpy())
            bufs["dec_B"].append(dec_B.cpu().numpy())
            bufs["dec_C"].append(dec_C.cpu().numpy())
            bufs["dec_B_fboff"].append(dec_B_off.cpu().numpy())
            bufs["r_pm2_A"].append(r_pm2.cpu().numpy())
            bufs["r_pm1_A"].append(r_pm1.cpu().numpy())
            bufs["q_pred_argmax_target_isi"].append(pred_peak_idx.cpu().numpy())

            if (batch_i + 1) % 10 == 0 or batch_i == 0:
                print(f"[batch {batch_i + 1}/{args.n_batches}] B={B}", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in bufs.items()}
    out["meta_N"] = np.array([N])
    out["meta_step_deg"] = np.array([step_deg])
    out["meta_center_idx"] = np.array([N // 2])
    out["meta_period"] = np.array([period])

    np.savez_compressed(args.output, **out)
    print(f"[done] saved {args.output}; n_trials={len(out['dec_A'])}", flush=True)


if __name__ == "__main__":
    main()
