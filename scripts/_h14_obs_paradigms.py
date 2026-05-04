"""Task #14 Phase 4 — Channel-resolved diagnostic on observational paradigms.

For each of (M3R native, HMS native, HMS-T native, VCD-test3 native) we apply
the published filter to a single shared forward pass and compute the
channel-resolved measures r_at_predch / r_at_stimch on the ex vs unex buckets
plus decoder accuracy.

Then Exp 4.5 reconstructs HMS-T as paired-fork: take the HMS-T-qualifying
trials (clean 3-march context with V2 pred_err ≤ 5°) and apply the +90° roll
construction. Compare Δ_decC paired-fork vs Δ_decC observational on the same
trials.

Network: R1+R2 at results/simple_dual/emergent_seed42/checkpoint.pt
HMM stream: 80 batches × bs (N_trials = 80 * train_cfg.batch_size).
"""
from __future__ import annotations
import os, sys, json
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
N_BATCHES = 80
SEQ_LENGTH = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cfg, train_cfg, stim_cfg = load_config(CONFIG)
n_ori = int(model_cfg.n_orientations)
period = float(model_cfg.orientation_range)
step_deg = period / n_ori
steps_on = int(train_cfg.steps_on); steps_isi = int(train_cfg.steps_isi)
steps_per = steps_on + steps_isi
batch_size = int(train_cfg.batch_size)

net = LaminarV1V2Network(model_cfg).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval(); net.oracle_mode = False; net.feedback_scale.fill_(1.0)
for p in net.parameters():
    p.requires_grad_(False)

dC_state = torch.load(DEC_C, map_location=device, weights_only=False)
if isinstance(dC_state, dict) and "state_dict" in dC_state:
    dC_state = dC_state["state_dict"]
decC = nn.Linear(n_ori, n_ori, bias=True).to(device)
decC.load_state_dict(dC_state); decC.eval()

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

# Per-presentation data buffers (record at every pres_i in [1, 24])
buf = {k: [] for k in (
    "r_probe", "pred_ch", "true_ch", "pred_err", "pi",
    "is_amb", "actual_ori", "pred_ori", "decoder_top1",
    "r_stimch", "r_predch", "r_norm",
    # paired-fork buffers (only recorded for last presentation per trial)
    "is_last_pres", "pf_r_probe_B", "pf_r_stimch_B", "pf_r_predch_B",
    "pf_decoder_top1_B", "pf_correct_B",
    # Trajectory features for HMS filter
    "ori_minus2", "ori_minus1",   # for HMS 3-march detection
)}

print(f"[forward] {N_BATCHES} batches × bs={batch_size} = {N_BATCHES * batch_size} HMM trials", flush=True)

for batch_i in range(N_BATCHES):
    md = gen.generate(batch_size, SEQ_LENGTH, generator=rng)
    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

    with torch.no_grad():
        packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
        r_l23, _, aux = net.forward(packed)
        q_pred_all = aux["q_pred_all"]
        pi_all = aux["pi_pred_eff_all"]

    # For paired-fork only at probe_idx=24
    probe_idx = SEQ_LENGTH - 1
    probe_onset = probe_idx * steps_per
    stim_unex = stim_seq.clone()
    stim_unex[:, probe_onset:probe_onset + steps_on, :] = torch.roll(
        stim_seq[:, probe_onset:probe_onset + steps_on, :], shifts=n_ori // 2, dims=-1)
    with torch.no_grad():
        packed_B = net.pack_inputs(stim_unex, cue_seq, ts_seq)
        r_l23_B, _, _ = net.forward(packed_B)

    true_ori = md.orientations.to(device)   # [B, S]
    is_amb_all = md.is_ambiguous.to(device) # [B, S]

    for pres_i in range(1, SEQ_LENGTH):
        t_isi_last = pres_i * steps_per - 1
        q_pred_isi = q_pred_all[:, t_isi_last, :]
        pi_isi = pi_all[:, t_isi_last, 0]
        pred_peak = q_pred_isi.argmax(dim=-1)
        pred_ori = pred_peak.float() * step_deg

        actual_ori = true_ori[:, pres_i]
        pe = circular_distance(pred_ori, actual_ori, period).abs()
        true_ch = (actual_ori / step_deg).round().long() % n_ori

        t0 = pres_i * steps_per + 9
        t1 = pres_i * steps_per + 11      # inclusive
        r_win = r_l23[:, t0:t1+1, :].mean(dim=1)
        bi_arange = torch.arange(batch_size, device=device)
        r_stimch = r_win[bi_arange, true_ch.long()]
        r_predch = r_win[bi_arange, pred_peak.long()]
        r_norm = r_win.norm(dim=1)

        decoder_pred = decC(r_win).argmax(-1)

        # ori_minus1, ori_minus2 for HMS 3-march detection
        ori_m1 = true_ori[:, pres_i-1] if pres_i >= 1 else torch.full_like(actual_ori, -999.0)
        ori_m2 = true_ori[:, pres_i-2] if pres_i >= 2 else torch.full_like(actual_ori, -999.0)

        is_amb = is_amb_all[:, pres_i]

        is_last = (pres_i == probe_idx)
        # Pass-B values at last presentation
        if is_last:
            r_win_B = r_l23_B[:, t0:t1+1, :].mean(dim=1)
            stim_ch_B = (true_ch + n_ori // 2) % n_ori
            r_stimch_B = r_win_B[bi_arange, stim_ch_B.long()]
            r_predch_B = r_win_B[bi_arange, pred_peak.long()]
            decoder_pred_B = decC(r_win_B).argmax(-1)
            correct_B = (decoder_pred_B == stim_ch_B).float()
        else:
            r_win_B = torch.zeros_like(r_win)
            r_stimch_B = torch.zeros_like(r_stimch)
            r_predch_B = torch.zeros_like(r_predch)
            decoder_pred_B = torch.zeros_like(decoder_pred)
            correct_B = torch.zeros_like(r_stimch)

        buf["r_probe"].append(r_win.cpu().numpy())
        buf["pred_ch"].append(pred_peak.cpu().numpy())
        buf["true_ch"].append(true_ch.cpu().numpy())
        buf["pred_err"].append(pe.cpu().numpy())
        buf["pi"].append(pi_isi.cpu().numpy())
        buf["is_amb"].append(is_amb.cpu().numpy())
        buf["actual_ori"].append(actual_ori.cpu().numpy())
        buf["pred_ori"].append(pred_ori.cpu().numpy())
        buf["decoder_top1"].append(decoder_pred.cpu().numpy())
        buf["r_stimch"].append(r_stimch.cpu().numpy())
        buf["r_predch"].append(r_predch.cpu().numpy())
        buf["r_norm"].append(r_norm.cpu().numpy())
        buf["ori_minus1"].append(ori_m1.cpu().numpy())
        buf["ori_minus2"].append(ori_m2.cpu().numpy())
        buf["is_last_pres"].append(np.full(batch_size, is_last))
        buf["pf_r_probe_B"].append(r_win_B.cpu().numpy())
        buf["pf_r_stimch_B"].append(r_stimch_B.cpu().numpy())
        buf["pf_r_predch_B"].append(r_predch_B.cpu().numpy())
        buf["pf_decoder_top1_B"].append(decoder_pred_B.cpu().numpy())
        buf["pf_correct_B"].append(correct_B.cpu().numpy())

    if (batch_i + 1) % 20 == 0:
        print(f"  batch {batch_i+1}/{N_BATCHES}  presentations collected", flush=True)

# Concatenate
data = {}
for k, lst in buf.items():
    if k == "r_probe" or k == "pf_r_probe_B":
        data[k] = np.concatenate([a for a in lst], axis=0)  # [N_records, n_ori]
    else:
        data[k] = np.concatenate([a.flatten() if a.ndim == 1 else a for a in lst], axis=0)

# correct top-1 vs true_ch
data["correct"] = (data["decoder_top1"] == data["true_ch"]).astype(np.float64)
print(f"[N] total per-presentation records: {data['correct'].shape[0]}")

# ============================================================================
# Define each paradigm's ex/unex selection on this shared data
# ============================================================================
keep = ~data["is_amb"].astype(bool)

# Pi quartile pool computed on kept records
pi_q75_global = float(np.percentile(data["pi"][keep], 75))
print(f"[pi Q75 global, kept] = {pi_q75_global:.4f}")

# Helper: HMS trajectory features
def signed_circ_delta(a, b, period):
    diff = (a - b) % period
    diff = np.where(diff > period / 2, diff - period, diff)
    return diff

d_ctx = signed_circ_delta(data["ori_minus1"], data["ori_minus2"], period)
d_probe = signed_circ_delta(data["actual_ori"], data["ori_minus1"], period)
ctx_match_step = np.abs(np.abs(d_ctx) - 5.0) <= 1.0
probe_match_step = np.abs(np.abs(d_probe) - 5.0) <= 1.0
same_dir = (np.sign(d_ctx) == np.sign(d_probe)) & (np.abs(d_ctx) > 1e-6)
is_3march = ctx_match_step & probe_match_step & same_dir
is_march_jump = ctx_match_step & (np.abs(d_probe) >= 75.0)

paradigms = {
    "Row 10  M3R native":        {
        "ex":   keep & (data["pred_err"] <= 5.0)  & (data["pi"] >= pi_q75_global),
        "unex": keep & (data["pred_err"] > 20.0) & (data["pi"] >= pi_q75_global),
    },
    "Row 11  HMS native":        {
        # No pred_err filter; trajectory only + pi Q75
        "ex":   keep & is_3march      & (data["pi"] >= pi_q75_global),
        "unex": keep & is_march_jump  & (data["pi"] >= pi_q75_global),
    },
    "Row 12  HMS-T native":      {
        "ex":   keep & is_3march      & (data["pred_err"] <= 5.0)  & (data["pi"] >= pi_q75_global),
        "unex": keep & is_march_jump  & (data["pred_err"] > 60.0) & (data["pi"] >= pi_q75_global),
    },
    "Row 14  VCD-test3 native":  {
        "ex":   keep & (data["pred_err"] <= 10.0),
        "unex": keep & (data["pred_err"] > 20.0),
    },
}

print(f"\n========== Exp 4.4 — Channel-resolved diagnostic on observational paradigms ==========")
print(f"  {'paradigm':28s}  {'n_ex':>5s}  {'n_unex':>6s}  {'ex_acc':>7s}  {'unex_acc':>9s}  "
      f"{'Δ_decC':>9s}  {'r_stimch_ex':>11s}  {'r_stimch_unex':>13s}  {'Δr_stimch':>10s}  "
      f"{'pe_ex':>6s}  {'pe_unex':>7s}")
exp44_out = {}
for name, sel in paradigms.items():
    n_ex = int(sel["ex"].sum()); n_unex = int(sel["unex"].sum())
    if n_ex < 20 or n_unex < 20:
        print(f"  {name:28s}  n_ex={n_ex:4d}  n_unex={n_unex:4d}  (insufficient)")
        exp44_out[name] = {"n_ex": n_ex, "n_unex": n_unex, "insufficient": True}
        continue
    ex_acc = float(data["correct"][sel["ex"]].mean())
    un_acc = float(data["correct"][sel["unex"]].mean())
    rs_ex = float(data["r_stimch"][sel["ex"]].mean())
    rs_un = float(data["r_stimch"][sel["unex"]].mean())
    rp_ex = float(data["r_predch"][sel["ex"]].mean())
    rp_un = float(data["r_predch"][sel["unex"]].mean())
    pe_ex = float(data["pred_err"][sel["ex"]].mean())
    pe_un = float(data["pred_err"][sel["unex"]].mean())
    rn_ex = float(data["r_norm"][sel["ex"]].mean())
    rn_un = float(data["r_norm"][sel["unex"]].mean())
    print(f"  {name:28s}  {n_ex:5d}  {n_unex:6d}  {ex_acc:7.4f}  {un_acc:9.4f}  "
          f"{ex_acc-un_acc:+9.4f}  {rs_ex:11.4f}  {rs_un:13.4f}  {rs_ex-rs_un:+10.4f}  "
          f"{pe_ex:6.2f}  {pe_un:7.2f}")
    exp44_out[name] = {
        "n_ex": n_ex, "n_unex": n_unex,
        "ex_acc": ex_acc, "unex_acc": un_acc, "delta_decC": ex_acc - un_acc,
        "r_stimch_ex": rs_ex, "r_stimch_unex": rs_un, "delta_r_stimch": rs_ex - rs_un,
        "r_predch_ex": rp_ex, "r_predch_unex": rp_un, "delta_r_predch": rp_ex - rp_un,
        "r_norm_ex": rn_ex, "r_norm_unex": rn_un,
        "pe_ex_mean_deg": pe_ex, "pe_unex_mean_deg": pe_un,
    }

# ============================================================================
# Exp 4.5 — Paired-fork on HMS-T-qualifying trials (3-march context + V2 pred_err ≤ 5°)
# Take the LAST PRESENTATION's record (same as paired-fork normally uses).
# Apply +90° roll. Compare Δ_decC paired-fork vs the observational HMS-T result above.
# ============================================================================
print(f"\n========== Exp 4.5 — Paired-fork on HMS-T-qualifying subset (last pres only) ==========")
last_pres_mask = data["is_last_pres"].astype(bool)
hmst_qual = (
    last_pres_mask & keep & is_3march & (data["pred_err"] <= 5.0) & (data["pi"] >= pi_q75_global)
)
n_q = int(hmst_qual.sum())
print(f"  HMS-T-qualifying trials at last pres: n={n_q}")
if n_q >= 20:
    # Pass A acc on these trials
    pf_ex_acc = float(data["correct"][hmst_qual].mean())
    # Pass B acc (paired-fork +90°)
    pf_un_acc = float(data["pf_correct_B"][hmst_qual].mean())
    rs_pf_ex = float(data["r_stimch"][hmst_qual].mean())
    rs_pf_un = float(data["pf_r_stimch_B"][hmst_qual].mean())
    print(f"  Paired-fork on HMS-T qual: n={n_q}  ex={pf_ex_acc:.4f}  unex={pf_un_acc:.4f}  Δ={pf_ex_acc - pf_un_acc:+.4f}")
    print(f"  Channel-resolved: r_stimch_A={rs_pf_ex:.4f}  r_stimch_B={rs_pf_un:.4f}  Δr={rs_pf_ex-rs_pf_un:+.4f}")
    exp45 = {
        "n_qualifying": n_q,
        "paired_fork_ex_acc": pf_ex_acc, "paired_fork_unex_acc": pf_un_acc,
        "paired_fork_delta_decC": pf_ex_acc - pf_un_acc,
        "paired_fork_r_stimch_A": rs_pf_ex, "paired_fork_r_stimch_B": rs_pf_un,
        "paired_fork_delta_r_stimch": rs_pf_ex - rs_pf_un,
    }
    # Compare against observational HMS-T on the SAME last-pres set
    # (note: observational HMS-T pools across all presentations; here we use
    #  only last-pres subset for an apples-to-apples paired-fork comparison)
    obs_ex_lp = last_pres_mask & paradigms["Row 12  HMS-T native"]["ex"]
    obs_un_lp = last_pres_mask & paradigms["Row 12  HMS-T native"]["unex"]
    print(f"  (observational HMS-T restricted to last-pres: n_ex={int(obs_ex_lp.sum())} "
          f"n_unex={int(obs_un_lp.sum())})")
    if obs_ex_lp.sum() >= 20 and obs_un_lp.sum() >= 20:
        oex = float(data["correct"][obs_ex_lp].mean())
        oun = float(data["correct"][obs_un_lp].mean())
        print(f"  Observational HMS-T on last-pres: ex={oex:.4f} unex={oun:.4f} Δ={oex-oun:+.4f}")
        exp45["obs_HMS_T_last_pres_delta"] = oex - oun
        exp45["obs_HMS_T_last_pres_n_ex"] = int(obs_ex_lp.sum())
        exp45["obs_HMS_T_last_pres_n_unex"] = int(obs_un_lp.sum())
else:
    exp45 = {"n_qualifying": n_q, "insufficient": True}

with open("/tmp/h14_obs_paradigms.json", "w") as f:
    json.dump({"exp_4_4": exp44_out, "exp_4_5": exp45}, f, indent=2)
print("\n[save] /tmp/h14_obs_paradigms.json")
