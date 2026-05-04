"""Task #11 follow-up: stratify paired-fork Δ_decC by V2 pred_err quartile.

If H2 is correct (V2 pred_err is the discriminator):
- low pred_err trials (V2 predicted well) → paired-fork Δ_decC < 0 (dampening)
- high pred_err trials (V2 predicted poorly) → paired-fork Δ_decC > 0 (sharpening)
- The full HMM C1 pool averages over all → small net Δ.

Also tests the same on a larger trial pool (to overcome n=153 small-sample
issue) by running 4× more HMM batches.
"""
from __future__ import annotations
import os, sys, json, copy, time
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
N_TRIALS = 4000   # 4× larger for stratified n
SEQ_LENGTH = 25
TASK_STATE_FOCUSED = (1.0, 0.0)
TASK_STATE_ROUTINE = (0.0, 1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cfg, train_cfg, stim_cfg = load_config(CONFIG)
n_ori = int(model_cfg.n_orientations)
period = float(model_cfg.orientation_range)
step_deg = period / n_ori
steps_on = int(train_cfg.steps_on); steps_isi = int(train_cfg.steps_isi)
steps_per = steps_on + steps_isi
probe_idx = SEQ_LENGTH - 1
probe_onset = probe_idx * steps_per
isi_pre_probe = probe_onset - 1

# Load network
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

    all_pred_err_A = []; all_pred_err_B = []
    all_r_probe_A = []; all_r_probe_B = []
    all_true_ch_ex = []; all_true_ch_unex = []
    all_pi = []; all_keep = []

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

        q_pred_isi = aux_A["q_pred_all"][:, isi_pre_probe, :]
        pi_isi = aux_A["pi_pred_eff_all"][:, isi_pre_probe, 0]
        pred_peak = q_pred_isi.argmax(dim=-1)
        pred_ori = pred_peak.float() * step_deg

        actual_ori_A = true_ori_ex
        actual_ori_B = (true_ori_ex + period / 2.0) % period
        pe_A = circular_distance(pred_ori, actual_ori_A, period).abs()
        pe_B = circular_distance(pred_ori, actual_ori_B, period).abs()

        r_probe_A = r_l23_A[:, probe_onset+9:probe_onset+11, :].mean(dim=1)
        r_probe_B = r_l23_B[:, probe_onset+9:probe_onset+11, :].mean(dim=1)

        all_pred_err_A.append(pe_A.cpu().numpy())
        all_pred_err_B.append(pe_B.cpu().numpy())
        all_r_probe_A.append(r_probe_A.cpu().numpy())
        all_r_probe_B.append(r_probe_B.cpu().numpy())
        all_true_ch_ex.append(true_ch_ex.cpu().numpy())
        all_true_ch_unex.append(true_ch_unex.cpu().numpy())
        all_pi.append(pi_isi.cpu().numpy())
        all_keep.append((~is_amb).cpu().numpy())
        n_done += b

    pred_err_A = np.concatenate(all_pred_err_A)
    pred_err_B = np.concatenate(all_pred_err_B)
    r_probe_A = np.concatenate(all_r_probe_A)
    r_probe_B = np.concatenate(all_r_probe_B)
    true_ch_ex = np.concatenate(all_true_ch_ex)
    true_ch_unex = np.concatenate(all_true_ch_unex)
    pi = np.concatenate(all_pi)
    keep = np.concatenate(all_keep)

    print(f"[N] total={pred_err_A.shape[0]}, kept={int(keep.sum())}", flush=True)

    # Per-trial decoder predictions
    with torch.no_grad():
        rA_t = torch.tensor(r_probe_A).float().to(device)
        rB_t = torch.tensor(r_probe_B).float().to(device)
        pred_A = decC(rA_t).argmax(-1).cpu().numpy()
        pred_B = decC(rB_t).argmax(-1).cpu().numpy()

    correct_A = (pred_A == true_ch_ex).astype(np.float64)
    correct_B = (pred_B == true_ch_unex).astype(np.float64)

    # Stratify by V2 pred_err on Pass A (computed at pre-probe ISI; same in both passes)
    pe = pred_err_A.copy()
    pe[~keep] = np.nan
    print(f"[V2 pred_err A] mean={np.nanmean(pe):.2f}° median={np.nanmedian(pe):.2f}°  "
          f"q25={np.nanpercentile(pe, 25):.2f}°  q50={np.nanpercentile(pe, 50):.2f}°  "
          f"q75={np.nanpercentile(pe, 75):.2f}°", flush=True)

    # Define 4 strata by quartile + a fixed-bin variant
    q1, q2, q3 = np.nanpercentile(pe, [25, 50, 75])
    strata = {
        f"q1_pe<={q1:.1f}":       keep & (pred_err_A <= q1),
        f"q2_{q1:.1f}<pe<={q2:.1f}": keep & (pred_err_A > q1) & (pred_err_A <= q2),
        f"q3_{q2:.1f}<pe<={q3:.1f}": keep & (pred_err_A > q2) & (pred_err_A <= q3),
        f"q4_pe>{q3:.1f}":        keep & (pred_err_A > q3),
    }
    fixed_bins = {
        "fixed_pe<=5":      keep & (pred_err_A <= 5.0),
        "fixed_5<pe<=15":   keep & (pred_err_A > 5.0) & (pred_err_A <= 15.0),
        "fixed_15<pe<=30":  keep & (pred_err_A > 15.0) & (pred_err_A <= 30.0),
        "fixed_30<pe<=60":  keep & (pred_err_A > 30.0) & (pred_err_A <= 60.0),
        "fixed_pe>60":      keep & (pred_err_A > 60.0),
    }
    print("\n[Paired-fork Δ_decC stratified by V2 pred_err]")
    print(f"  {'stratum':30s}  {'n':>5s}  {'ex_acc':>7s}  {'unex_acc':>9s}  {'Δ':>9s}  {'r_norm_ex':>10s}  {'r_norm_unex':>11s}")
    out = {}
    for name, mask in {**strata, **fixed_bins}.items():
        n = int(mask.sum())
        if n < 20:
            print(f"  {name:30s}  n={n:4d}  (skipped, too few)")
            out[name] = {"n": n}
            continue
        ex_acc = correct_A[mask].mean()
        un_acc = correct_B[mask].mean()
        d = ex_acc - un_acc
        rn_ex = np.linalg.norm(r_probe_A[mask], axis=1).mean()
        rn_un = np.linalg.norm(r_probe_B[mask], axis=1).mean()
        print(f"  {name:30s}  {n:5d}  {ex_acc:7.4f}  {un_acc:9.4f}  {d:+9.4f}  {rn_ex:10.4f}  {rn_un:11.4f}")
        out[name] = {"n": n, "ex_acc": float(ex_acc), "unex_acc": float(un_acc),
                     "delta": float(d), "r_norm_ex": float(rn_ex), "r_norm_unex": float(rn_un)}

    # Also overall:
    n_keep = int(keep.sum())
    overall = {
        "n_keep": n_keep,
        "ex_acc": float(correct_A[keep].mean()),
        "unex_acc": float(correct_B[keep].mean()),
        "delta": float(correct_A[keep].mean() - correct_B[keep].mean()),
        "r_norm_ex_mean": float(np.linalg.norm(r_probe_A[keep], axis=1).mean()),
        "r_norm_unex_mean": float(np.linalg.norm(r_probe_B[keep], axis=1).mean()),
    }
    print(f"\n[overall paired-fork {label}] n_keep={n_keep} ex={overall['ex_acc']:.4f} "
          f"unex={overall['unex_acc']:.4f} Δ={overall['delta']:+.4f}", flush=True)

    # Observational pred_err split for the same data
    obs_thresholds = [(5.0, 20.0), (5.0, 60.0), (10.0, 20.0), (10.0, 30.0)]
    print(f"\n[Observational pred_err split on Pass A only, decoded against true_ch_ex]")
    for lo_thr, hi_thr in obs_thresholds:
        m_lo = keep & (pred_err_A <= lo_thr)
        m_hi = keep & (pred_err_A > hi_thr)
        if m_lo.sum() < 20 or m_hi.sum() < 20:
            print(f"  pe<={lo_thr} vs pe>{hi_thr}: insufficient data")
            continue
        a_lo = correct_A[m_lo].mean(); a_hi = correct_A[m_hi].mean()
        rn_lo = np.linalg.norm(r_probe_A[m_lo], axis=1).mean()
        rn_hi = np.linalg.norm(r_probe_A[m_hi], axis=1).mean()
        # pi-filter
        pi_q75 = np.percentile(pi[keep], 75)
        m_lo_pi = m_lo & (pi >= pi_q75); m_hi_pi = m_hi & (pi >= pi_q75)
        a_lo_pi = correct_A[m_lo_pi].mean() if m_lo_pi.sum() >= 20 else float("nan")
        a_hi_pi = correct_A[m_hi_pi].mean() if m_hi_pi.sum() >= 20 else float("nan")
        print(f"  pe<={lo_thr:5.1f} (n={int(m_lo.sum()):4d}, acc={a_lo:.4f}, ||r||={rn_lo:.4f}) "
              f"vs pe>{hi_thr:5.1f} (n={int(m_hi.sum()):4d}, acc={a_hi:.4f}, ||r||={rn_hi:.4f})  "
              f"Δ={a_lo - a_hi:+.4f}  | pi-Q75: lo n={int(m_lo_pi.sum())} acc={a_lo_pi:.4f}, "
              f"hi n={int(m_hi_pi.sum())} acc={a_hi_pi:.4f}, Δ={a_lo_pi - a_hi_pi:+.4f}")

    return overall, out

# Run for HMM C1 (focused + HMM cue)
res_C1, strat_C1 = run_paradigm(TASK_STATE_FOCUSED, zero_cue=False, label="HMM C1 (focused + HMM cue)")

# Run for HMM C3 (focused + zero cue)
res_C3, strat_C3 = run_paradigm(TASK_STATE_FOCUSED, zero_cue=True,  label="HMM C3 (focused + zero cue)")

# Save
with open("/tmp/h11_strat_diag.json", "w") as f:
    json.dump({"HMM_C1": {"overall": res_C1, "strata": strat_C1},
               "HMM_C3": {"overall": res_C3, "strata": strat_C3}}, f, indent=2)
print("\n[save] /tmp/h11_strat_diag.json")
