"""Task #13 — Channel-resolved suppression measure on HMM C1 (R1+R2).

Extension of scripts/_h11_strat_diag.py per validator NO-GO. Adds 4 scalar
measures per trial (computed at the same readout window [9:11) as Δ_decC):

  r_at_predch_A:  L2/3 activity at V2-predicted channel, Pass A.
  r_at_predch_B:  L2/3 activity at V2-predicted channel, Pass B (probe rolled +90°).
  r_at_stimch_A:  L2/3 activity at Pass A's actual stim channel.
  r_at_stimch_B:  L2/3 activity at Pass B's actual stim channel.

Per-stratum means saved alongside global r_norm and decoder accuracies. Output
to /tmp/h11_strat_diag_v2.json. Per-trial arrays to /tmp/h11_pertrial_v2.npz.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.training.trainer import build_stimulus_sequence
from src.stimulus.sequences import HMMSequenceGenerator
from src.utils import circular_distance

CKPT = "/mnt/c/Users/User/codingproj/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt"
DEC_C = "/mnt/c/Users/User/codingproj/freshstart/checkpoints/decoder_c.pt"
CONFIG = "/mnt/c/Users/User/codingproj/freshstart/config/sweep/sweep_rescue_1_2.yaml"

SEED = 42
N_TRIALS = 4000
SEQ_LENGTH = 25
TASK_STATE_FOCUSED = (1.0, 0.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] device={device}", flush=True)
model_cfg, train_cfg, stim_cfg = load_config(CONFIG)
n_ori = int(model_cfg.n_orientations)
period = float(model_cfg.orientation_range)
step_deg = period / n_ori
steps_on = int(train_cfg.steps_on); steps_isi = int(train_cfg.steps_isi)
steps_per = steps_on + steps_isi
probe_idx = SEQ_LENGTH - 1
probe_onset = probe_idx * steps_per
isi_pre_probe = probe_onset - 1

# Network
net = LaminarV1V2Network(model_cfg).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval(); net.oracle_mode = False; net.feedback_scale.fill_(1.0)
for p in net.parameters():
    p.requires_grad_(False)

# Decoder C
dC_state = torch.load(DEC_C, map_location=device, weights_only=False)
if isinstance(dC_state, dict) and "state_dict" in dC_state:
    dC_state = dC_state["state_dict"]
decC = nn.Linear(n_ori, n_ori, bias=True).to(device)
decC.load_state_dict(dC_state)
decC.eval()
print(f"[setup] Decoder C loaded; ||W||={decC.weight.norm().item():.2f}", flush=True)


def run_paradigm(task_state, zero_cue, label, n_trials=N_TRIALS, batch=1000):
    print(f"\n========== {label}  (task_state={task_state}, zero_cue={zero_cue}) ==========",
          flush=True)
    gen = HMMSequenceGenerator(
        n_orientations=n_ori,
        p_self=stim_cfg.p_self, p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors, jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step, period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim, n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        task_p_switch=getattr(stim_cfg, "task_p_switch", 0.0),
    )
    rng = torch.Generator().manual_seed(SEED)

    all_pe_A = []; all_pe_B = []
    all_r_probe_A = []; all_r_probe_B = []
    all_true_ch_ex = []; all_true_ch_unex = []
    all_pred_ch = []
    all_pi = []; all_keep = []
    # Channel-resolved scalars
    all_r_predch_A = []; all_r_predch_B = []
    all_r_stimch_A = []; all_r_stimch_B = []

    n_done = 0
    while n_done < n_trials:
        b = min(batch, n_trials - n_done)
        md = gen.generate(b, SEQ_LENGTH, generator=rng)
        new_ts = torch.zeros_like(md.task_states)
        new_ts[..., 0] = float(task_state[0]); new_ts[..., 1] = float(task_state[1])
        md.task_states = new_ts
        if zero_cue:
            md.cues = torch.zeros_like(md.cues)
        stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
        stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

        true_ori_ex = md.orientations[:, probe_idx].to(device)
        true_ch_ex = (true_ori_ex / step_deg).round().long() % n_ori
        true_ch_unex = (true_ch_ex + n_ori // 2) % n_ori

        stim_ex = stim_seq
        stim_unex = stim_seq.clone()
        stim_unex[:, probe_onset:probe_onset + steps_on, :] = torch.roll(
            stim_seq[:, probe_onset:probe_onset + steps_on, :], shifts=n_ori // 2, dims=-1)

        is_amb = md.is_ambiguous[:, probe_idx].to(device)

        with torch.no_grad():
            packed_A = net.pack_inputs(stim_ex, cue_seq, ts_seq)
            r_l23_A, _, aux_A = net.forward(packed_A)
            packed_B = net.pack_inputs(stim_unex, cue_seq, ts_seq)
            r_l23_B, _, _ = net.forward(packed_B)

        q_pred_isi = aux_A["q_pred_all"][:, isi_pre_probe, :]   # [B, N]
        pi_isi = aux_A["pi_pred_eff_all"][:, isi_pre_probe, 0]  # [B]
        pred_peak = q_pred_isi.argmax(dim=-1)                   # [B]  V2-predicted channel
        pred_ori = pred_peak.float() * step_deg

        actual_ori_A = true_ori_ex
        actual_ori_B = (true_ori_ex + period / 2.0) % period
        pe_A = circular_distance(pred_ori, actual_ori_A, period).abs()
        pe_B = circular_distance(pred_ori, actual_ori_B, period).abs()

        # Mean over readout window — keeps full channel dim
        r_probe_A = r_l23_A[:, probe_onset+9:probe_onset+11, :].mean(dim=1)  # [B, N]
        r_probe_B = r_l23_B[:, probe_onset+9:probe_onset+11, :].mean(dim=1)  # [B, N]

        # Channel-resolved scalars: gather one channel per trial
        idx_pred = pred_peak.long()
        idx_stim_A = true_ch_ex.long()
        idx_stim_B = true_ch_unex.long()
        # arange index for batch dim
        bi_arange = torch.arange(b, device=device)
        r_predch_A = r_probe_A[bi_arange, idx_pred]    # [B]
        r_predch_B = r_probe_B[bi_arange, idx_pred]
        r_stimch_A = r_probe_A[bi_arange, idx_stim_A]
        r_stimch_B = r_probe_B[bi_arange, idx_stim_B]

        all_pe_A.append(pe_A.cpu().numpy())
        all_pe_B.append(pe_B.cpu().numpy())
        all_r_probe_A.append(r_probe_A.cpu().numpy())
        all_r_probe_B.append(r_probe_B.cpu().numpy())
        all_true_ch_ex.append(true_ch_ex.cpu().numpy())
        all_true_ch_unex.append(true_ch_unex.cpu().numpy())
        all_pred_ch.append(pred_peak.cpu().numpy())
        all_pi.append(pi_isi.cpu().numpy())
        all_keep.append((~is_amb).cpu().numpy())
        all_r_predch_A.append(r_predch_A.cpu().numpy())
        all_r_predch_B.append(r_predch_B.cpu().numpy())
        all_r_stimch_A.append(r_stimch_A.cpu().numpy())
        all_r_stimch_B.append(r_stimch_B.cpu().numpy())
        n_done += b

    pe_A_arr = np.concatenate(all_pe_A)
    pe_B_arr = np.concatenate(all_pe_B)
    r_probe_A = np.concatenate(all_r_probe_A)
    r_probe_B = np.concatenate(all_r_probe_B)
    true_ch_ex = np.concatenate(all_true_ch_ex)
    true_ch_unex = np.concatenate(all_true_ch_unex)
    pred_ch = np.concatenate(all_pred_ch)
    pi_arr = np.concatenate(all_pi)
    keep = np.concatenate(all_keep)
    r_predch_A_arr = np.concatenate(all_r_predch_A)
    r_predch_B_arr = np.concatenate(all_r_predch_B)
    r_stimch_A_arr = np.concatenate(all_r_stimch_A)
    r_stimch_B_arr = np.concatenate(all_r_stimch_B)

    print(f"[N] total={pe_A_arr.shape[0]}, kept={int(keep.sum())}", flush=True)

    # Decoder predictions
    with torch.no_grad():
        rA_t = torch.tensor(r_probe_A).float().to(device)
        rB_t = torch.tensor(r_probe_B).float().to(device)
        pred_A = decC(rA_t).argmax(-1).cpu().numpy()
        pred_B = decC(rB_t).argmax(-1).cpu().numpy()
    correct_A = (pred_A == true_ch_ex).astype(np.float64)
    correct_B = (pred_B == true_ch_unex).astype(np.float64)

    # Stratify
    pe = pe_A_arr.copy(); pe[~keep] = np.nan
    print(f"[V2 pred_err A] mean={np.nanmean(pe):.2f}° median={np.nanmedian(pe):.2f}°  "
          f"q25={np.nanpercentile(pe, 25):.2f}°  q50={np.nanpercentile(pe, 50):.2f}°  "
          f"q75={np.nanpercentile(pe, 75):.2f}°", flush=True)

    fixed_bins = {
        "fixed_pe<=5":      keep & (pe_A_arr <= 5.0),
        "fixed_5<pe<=15":   keep & (pe_A_arr > 5.0) & (pe_A_arr <= 15.0),
        "fixed_15<pe<=30":  keep & (pe_A_arr > 15.0) & (pe_A_arr <= 30.0),
        "fixed_30<pe<=60":  keep & (pe_A_arr > 30.0) & (pe_A_arr <= 60.0),
        "fixed_pe>60":      keep & (pe_A_arr > 60.0),
    }
    print("\n[Channel-resolved per stratum]")
    print(f"  {'stratum':18s}  {'n':>5s}  {'ex_acc':>7s}  {'unex_acc':>9s}  {'Δ_decC':>9s}  "
          f"{'r_predch_A':>11s}  {'r_predch_B':>11s}  {'r_stimch_A':>11s}  {'r_stimch_B':>11s}  "
          f"{'r_norm_A':>9s}  {'r_norm_B':>9s}  {'pe_B_mean':>9s}")
    out = {}
    for name, mask in fixed_bins.items():
        n = int(mask.sum())
        if n < 20:
            print(f"  {name:18s}  n={n:4d}  (skipped, too few)")
            out[name] = {"n": n}
            continue
        ex_acc = float(correct_A[mask].mean())
        un_acc = float(correct_B[mask].mean())
        d = ex_acc - un_acc
        rn_A = float(np.linalg.norm(r_probe_A[mask], axis=1).mean())
        rn_B = float(np.linalg.norm(r_probe_B[mask], axis=1).mean())
        rpA = float(r_predch_A_arr[mask].mean())
        rpB = float(r_predch_B_arr[mask].mean())
        rsA = float(r_stimch_A_arr[mask].mean())
        rsB = float(r_stimch_B_arr[mask].mean())
        peB = float(pe_B_arr[mask].mean())
        print(f"  {name:18s}  {n:5d}  {ex_acc:7.4f}  {un_acc:9.4f}  {d:+9.4f}  "
              f"{rpA:11.4f}  {rpB:11.4f}  {rsA:11.4f}  {rsB:11.4f}  "
              f"{rn_A:9.4f}  {rn_B:9.4f}  {peB:9.2f}")
        out[name] = {
            "n": n, "ex_acc": ex_acc, "unex_acc": un_acc, "delta_decC": d,
            "r_at_predch_A_mean": rpA, "r_at_predch_B_mean": rpB,
            "r_at_stimch_A_mean": rsA, "r_at_stimch_B_mean": rsB,
            "r_norm_A_mean": rn_A, "r_norm_B_mean": rn_B,
            "pe_B_mean_deg": peB,
        }

    n_keep = int(keep.sum())
    overall = {
        "n_keep": n_keep,
        "ex_acc": float(correct_A[keep].mean()),
        "unex_acc": float(correct_B[keep].mean()),
        "delta_decC": float(correct_A[keep].mean() - correct_B[keep].mean()),
        "r_at_predch_A_mean": float(r_predch_A_arr[keep].mean()),
        "r_at_predch_B_mean": float(r_predch_B_arr[keep].mean()),
        "r_at_stimch_A_mean": float(r_stimch_A_arr[keep].mean()),
        "r_at_stimch_B_mean": float(r_stimch_B_arr[keep].mean()),
        "r_norm_A_mean": float(np.linalg.norm(r_probe_A[keep], axis=1).mean()),
        "r_norm_B_mean": float(np.linalg.norm(r_probe_B[keep], axis=1).mean()),
        "pe_A_mean_deg": float(pe_A_arr[keep].mean()),
        "pe_B_mean_deg": float(pe_B_arr[keep].mean()),
    }
    print(f"\n[overall {label}] n_keep={n_keep}  ex={overall['ex_acc']:.4f}  "
          f"unex={overall['unex_acc']:.4f}  Δ={overall['delta_decC']:+.4f}", flush=True)

    pertrial = {
        "pe_A": pe_A_arr, "pe_B": pe_B_arr,
        "pi": pi_arr, "keep": keep,
        "true_ch_ex": true_ch_ex, "true_ch_unex": true_ch_unex,
        "pred_ch": pred_ch,
        "r_at_predch_A": r_predch_A_arr, "r_at_predch_B": r_predch_B_arr,
        "r_at_stimch_A": r_stimch_A_arr, "r_at_stimch_B": r_stimch_B_arr,
        "correct_A": correct_A, "correct_B": correct_B,
    }
    return overall, out, pertrial


# Run on HMM C1
res_C1, strat_C1, pt_C1 = run_paradigm(TASK_STATE_FOCUSED, zero_cue=False, label="HMM C1 (focused + HMM cue)")
# Also run HMM C3 for completeness (keeps v1's coverage)
res_C3, strat_C3, pt_C3 = run_paradigm(TASK_STATE_FOCUSED, zero_cue=True,  label="HMM C3 (focused + zero cue)")

with open("/tmp/h11_strat_diag_v2.json", "w") as f:
    json.dump({
        "HMM_C1": {"overall": res_C1, "strata": strat_C1},
        "HMM_C3": {"overall": res_C3, "strata": strat_C3},
    }, f, indent=2)
print("\n[save] /tmp/h11_strat_diag_v2.json")

np.savez("/tmp/h11_pertrial_v2.npz",
         **{f"C1_{k}": v for k, v in pt_C1.items()},
         **{f"C3_{k}": v for k, v in pt_C3.items()})
print("[save] /tmp/h11_pertrial_v2.npz")
