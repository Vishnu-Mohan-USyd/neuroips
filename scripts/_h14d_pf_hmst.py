"""Task #16 Phase 4d Exp 4.8 — Paired-fork HMS-T at proper n (≥500).

Take HMS-T-qualifying trials at the LAST presentation (3-march context +
V2 pred_err ≤5° + pi ≥ Q75 + ¬ambiguous at probe). Apply paired-fork +90°
roll for unex branch. Compare Δ_decC paired-fork on this subset.

Predicted by mechanism: paired-fork should give small/null Δ_decC like HMM C1's
overall (+0.012). NOT the strong −0.30 that observational HMS-T reports.

Falsification: paired-fork HMS-T STILL shows strong negative Δ_decC → mechanism
doesn't capture HMS-T's dampening, OR the construction transformation has
its own confound.

Run 50000 HMM trials to populate qualifying bucket (~500 expected at 1% rate).
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
N_BATCHES = 50    # × bs ≈ 1600 trials … not enough; use bigger batch
BATCH_SIZE = 1024
N_BATCHES = 50    # 50 × 1024 = 51200 trials
SEQ_LENGTH = 25

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

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
net = LaminarV1V2Network(model_cfg).to(device)
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

# Per-trial buffers — last presentation only (paired-fork comparison)
buf = {k: [] for k in (
    "pe_A", "pe_B", "pred_ch", "true_ch_ex", "true_ch_unex",
    "is_amb_probe", "pi_isi",
    "ori_minus1", "ori_minus2", "actual_ori_probe",
    "r_stimch_A", "r_stimch_B",
    "r_predch_A", "r_predch_B",
    "correct_A", "correct_B",
)}

print(f"[forward] {N_BATCHES} batches × bs={BATCH_SIZE} = {N_BATCHES * BATCH_SIZE} trials  "
      f"(paired-fork at probe_idx={probe_idx})", flush=True)
for bi_b in range(N_BATCHES):
    md = gen.generate(BATCH_SIZE, SEQ_LENGTH, generator=rng)
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = 1.0; new_ts[..., 1] = 0.0
    md.task_states = new_ts
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

    bi = torch.arange(BATCH_SIZE, device=device)
    r_predch_A = r_probe_A[bi, pred_peak.long()]
    r_predch_B = r_probe_B[bi, pred_peak.long()]
    r_stimch_A = r_probe_A[bi, true_ch_ex.long()]
    r_stimch_B = r_probe_B[bi, true_ch_unex.long()]

    pred_dec_A = decC(r_probe_A).argmax(-1)
    pred_dec_B = decC(r_probe_B).argmax(-1)
    correct_A = (pred_dec_A == true_ch_ex).float()
    correct_B = (pred_dec_B == true_ch_unex).float()

    # Trajectory features for HMS-T qualifier
    ori_m1 = md.orientations[:, probe_idx - 1].to(device)
    ori_m2 = md.orientations[:, probe_idx - 2].to(device)

    buf["pe_A"].append(pe_A.cpu().numpy())
    buf["pe_B"].append(pe_B.cpu().numpy())
    buf["pred_ch"].append(pred_peak.cpu().numpy())
    buf["true_ch_ex"].append(true_ch_ex.cpu().numpy())
    buf["true_ch_unex"].append(true_ch_unex.cpu().numpy())
    buf["is_amb_probe"].append(is_amb.cpu().numpy())
    buf["pi_isi"].append(pi_isi.cpu().numpy())
    buf["ori_minus1"].append(ori_m1.cpu().numpy())
    buf["ori_minus2"].append(ori_m2.cpu().numpy())
    buf["actual_ori_probe"].append(actual_ori_A.cpu().numpy())
    buf["r_stimch_A"].append(r_stimch_A.cpu().numpy())
    buf["r_stimch_B"].append(r_stimch_B.cpu().numpy())
    buf["r_predch_A"].append(r_predch_A.cpu().numpy())
    buf["r_predch_B"].append(r_predch_B.cpu().numpy())
    buf["correct_A"].append(correct_A.cpu().numpy())
    buf["correct_B"].append(correct_B.cpu().numpy())

    if (bi_b + 1) % 10 == 0:
        print(f"  batch {bi_b+1}/{N_BATCHES} done", flush=True)

data = {k: np.concatenate(v) for k, v in buf.items()}
N_total = data["pe_A"].shape[0]
print(f"[N] total trials: {N_total}", flush=True)

# Apply HMS-T qualifier to LAST presentation
def signed_circ(a, b, p):
    d = (a - b) % p
    return np.where(d > p / 2, d - p, d)

d_ctx = signed_circ(data["ori_minus1"], data["ori_minus2"], period)
d_probe = signed_circ(data["actual_ori_probe"], data["ori_minus1"], period)
ctx_match_step = np.abs(np.abs(d_ctx) - 5.0) <= 1.0
probe_match_step = np.abs(np.abs(d_probe) - 5.0) <= 1.0
same_dir = (np.sign(d_ctx) == np.sign(d_probe)) & (np.abs(d_ctx) > 1e-6)
is_3march_last = ctx_match_step & probe_match_step & same_dir
keep = ~data["is_amb_probe"].astype(bool)
pi_q75 = float(np.percentile(data["pi_isi"][keep], 75))
print(f"[pi Q75 global, kept] = {pi_q75:.4f}", flush=True)

hmst_qual = (
    keep & is_3march_last & (data["pe_A"] <= 5.0) & (data["pi_isi"] >= pi_q75)
)
n_q = int(hmst_qual.sum())
print(f"\nHMS-T-qualifying paired-fork trials at probe_idx={probe_idx}: n={n_q}", flush=True)

# Also compute the reverse: 3-march context + jump≥75° (= paired-fork-of-unex condition,
# would be needed for full paired-fork-of-HMS-T but here we use the standard +90° roll
# for the "unex" branch since paired-fork's unex IS the +90° rolled stim).
is_jump_last = ctx_match_step & (np.abs(d_probe) >= 75.0)
n_jump_q = int((keep & is_jump_last & (data["pe_A"] > 60.0) & (data["pi_isi"] >= pi_q75)).sum())
print(f"HMS-T 'unex' (jump≥75 + pe>60) at probe_idx: n={n_jump_q}  "
      f"(for reference; paired-fork uses +90° roll regardless)", flush=True)

# Paired-fork on HMS-T-qualifying
if n_q >= 50:
    ex_acc = float(data["correct_A"][hmst_qual].mean())
    un_acc = float(data["correct_B"][hmst_qual].mean())
    delta = ex_acc - un_acc
    rs_A = float(data["r_stimch_A"][hmst_qual].mean())
    rs_B = float(data["r_stimch_B"][hmst_qual].mean())
    rp_A = float(data["r_predch_A"][hmst_qual].mean())
    rp_B = float(data["r_predch_B"][hmst_qual].mean())
    print(f"\n========== Exp 4.8 — Paired-fork HMS-T (n={n_q}) ==========")
    print(f"  ex_acc(A)  = {ex_acc:.4f}")
    print(f"  unex_acc(B) = {un_acc:.4f}")
    print(f"  Δ_decC paired-fork HMS-T = {delta:+.4f}")
    print(f"  r_stimch_A = {rs_A:.4f}  r_stimch_B = {rs_B:.4f}  Δr_stimch = {rs_A-rs_B:+.4f}")
    print(f"  r_predch_A = {rp_A:.4f}  r_predch_B = {rp_B:.4f}  Δr_predch = {rp_A-rp_B:+.4f}")

    # 95% CI via bootstrap on the paired-fork delta
    np.random.seed(0)
    boot = []
    diffs = data["correct_A"][hmst_qual] - data["correct_B"][hmst_qual]
    for _ in range(2000):
        boot.append(np.random.choice(diffs, size=len(diffs), replace=True).mean())
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI for Δ_decC: [{lo:+.4f}, {hi:+.4f}]")

    # Compare to observational HMS-T (single forward pass — read from prior /tmp/h14d_hms_diag.json
    # if exists)
    obs_path = "/tmp/h14d_hms_diag.json"
    obs_delta = None
    if os.path.exists(obs_path):
        with open(obs_path) as f:
            obs = json.load(f)
        if "exp_4_9" in obs and "HMS-T native" in obs["exp_4_9"]:
            obs_delta = obs["exp_4_9"]["HMS-T native"].get("delta_decC_intact")
            print(f"\n  For comparison: observational HMS-T native Δ_decC = {obs_delta:+.4f}  "
                  f"(from /tmp/h14d_hms_diag.json)")
            if obs_delta is not None:
                if abs(delta) < 0.5 * abs(obs_delta):
                    verdict = "CONSISTENT WITH MECHANISM — paired-fork Δ much smaller than observational"
                elif (delta < 0) and abs(delta) >= 0.7 * abs(obs_delta):
                    verdict = "MECHANISM FALSIFIED — paired-fork still shows strong dampening"
                else:
                    verdict = "INTERMEDIATE — partial reduction"
                print(f"  Verdict: {verdict}")

    out = {
        "n_qualifying": n_q,
        "ex_acc": ex_acc, "unex_acc": un_acc,
        "delta_decC": delta, "delta_decC_95CI": [float(lo), float(hi)],
        "r_stimch_A": rs_A, "r_stimch_B": rs_B, "delta_r_stimch": rs_A - rs_B,
        "r_predch_A": rp_A, "r_predch_B": rp_B, "delta_r_predch": rp_A - rp_B,
        "obs_HMS_T_delta_for_comparison": obs_delta,
        "n_total_trials": N_total,
    }
else:
    out = {"n_qualifying": n_q, "insufficient": True}

with open("/tmp/h14d_pf_hmst.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n[save] /tmp/h14d_pf_hmst.json")
