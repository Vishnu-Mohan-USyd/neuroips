"""Phase 5 Open Q2 — polarity flip between HMM C1 paired-fork pe≤5° and
HMS-T paired-fork qualifying. Both are V2-predictable subsets giving Δ_decC
≈ −0.10 but channel-resolved Δr_stimch has opposite signs:
  HMM C1 pe≤5°:        Δr_stimch = −0.044  (suppressive at V2-pred channel)
  HMS-T paired-fork:   Δr_stimch = +0.179  (broadband-excitatory)

Hypotheses:
  H_pi:      V2 confidence (pi) differs between the subsets.
  H_context: pre-probe context length / V2 prediction strength differs.

Method: single forward pass on R1+R2, paired-fork at probe_idx=24 on N=51200
HMM trials. Apply both subset filters offline. Compare pi distributions, V2
prediction-magnitude distributions; match on pi and recompute Δr_stimch.
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
N_BATCHES = 50
BATCH_SIZE = 1024
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

KEYS = ("pe_A", "pi_isi", "is_amb_probe",
        "actual_ori_probe", "ori_minus1", "ori_minus2",
        "true_ch_ex", "true_ch_unex", "pred_ch",
        "r_stimch_A", "r_stimch_B", "r_predch_A", "r_predch_B",
        "correct_A", "correct_B",
        "q_pred_max_at_isi",   # max-channel value of V2's q_pred (prediction-strength proxy)
        "q_pred_entropy",      # negative-entropy of q_pred (low = sharp)
        )
buf = {k: [] for k in KEYS}

print(f"[forward] {N_BATCHES} batches × bs={BATCH_SIZE} = {N_BATCHES * BATCH_SIZE} trials  "
      f"paired-fork at probe_idx={probe_idx}", flush=True)
for bi_b in range(N_BATCHES):
    md = gen.generate(BATCH_SIZE, SEQ_LENGTH, generator=rng)
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = 1.0; new_ts[..., 1] = 0.0   # focused for HMM C1
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
        pkg_A = net.pack_inputs(stim_ex, cue_seq, ts_seq)
        r_l23_A, _, aux_A = net.forward(pkg_A)
        pkg_B = net.pack_inputs(stim_unex, cue_seq, ts_seq)
        r_l23_B, _, _ = net.forward(pkg_B)

    q_pred_isi = aux_A["q_pred_all"][:, isi_pre_probe, :]
    pi_isi = aux_A["pi_pred_eff_all"][:, isi_pre_probe, 0]
    pred_peak = q_pred_isi.argmax(dim=-1)
    pred_ori = pred_peak.float() * step_deg
    pe_A = circular_distance(pred_ori, true_ori_ex, period).abs()

    # Prediction-strength proxies
    q_max = q_pred_isi.max(dim=-1).values   # peak height (higher = sharper)
    q_norm = q_pred_isi / (q_pred_isi.sum(dim=-1, keepdim=True) + 1e-8)
    q_entropy = -(q_norm * torch.log(q_norm + 1e-8)).sum(dim=-1)   # entropy (lower = sharper)

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

    ori_m1 = md.orientations[:, probe_idx - 1].to(device)
    ori_m2 = md.orientations[:, probe_idx - 2].to(device)

    buf["pe_A"].append(pe_A.cpu().numpy())
    buf["pi_isi"].append(pi_isi.cpu().numpy())
    buf["is_amb_probe"].append(is_amb.cpu().numpy())
    buf["actual_ori_probe"].append(true_ori_ex.cpu().numpy())
    buf["ori_minus1"].append(ori_m1.cpu().numpy())
    buf["ori_minus2"].append(ori_m2.cpu().numpy())
    buf["true_ch_ex"].append(true_ch_ex.cpu().numpy())
    buf["true_ch_unex"].append(true_ch_unex.cpu().numpy())
    buf["pred_ch"].append(pred_peak.cpu().numpy())
    buf["r_stimch_A"].append(r_stimch_A.cpu().numpy())
    buf["r_stimch_B"].append(r_stimch_B.cpu().numpy())
    buf["r_predch_A"].append(r_predch_A.cpu().numpy())
    buf["r_predch_B"].append(r_predch_B.cpu().numpy())
    buf["correct_A"].append(correct_A.cpu().numpy())
    buf["correct_B"].append(correct_B.cpu().numpy())
    buf["q_pred_max_at_isi"].append(q_max.cpu().numpy())
    buf["q_pred_entropy"].append(q_entropy.cpu().numpy())
    if (bi_b + 1) % 10 == 0:
        print(f"  batch {bi_b+1}/{N_BATCHES}", flush=True)

data = {k: np.concatenate(v) for k, v in buf.items()}
N = data["pe_A"].shape[0]
print(f"[N] total trials: {N}", flush=True)

# Subset definitions
keep = ~data["is_amb_probe"].astype(bool)
pi_q75 = float(np.percentile(data["pi_isi"][keep], 75))

def signed_circ(a, b, p):
    d = (a - b) % p
    return np.where(d > p / 2, d - p, d)
d_ctx = signed_circ(data["ori_minus1"], data["ori_minus2"], period)
d_probe = signed_circ(data["actual_ori_probe"], data["ori_minus1"], period)
ctx_match_step = np.abs(np.abs(d_ctx) - 5.0) <= 1.0
probe_match_step = np.abs(np.abs(d_probe) - 5.0) <= 1.0
same_dir = (np.sign(d_ctx) == np.sign(d_probe)) & (np.abs(d_ctx) > 1e-6)
is_3march = ctx_match_step & probe_match_step & same_dir

S_C1   = keep & (data["pe_A"] <= 5.0)                                                 # HMM C1 pe≤5° subset
S_HMST = keep & is_3march & (data["pe_A"] <= 5.0) & (data["pi_isi"] >= pi_q75)        # HMS-T paired-fork qual

n_C1   = int(S_C1.sum()); n_HMST = int(S_HMST.sum())
print(f"\n[subset sizes] HMM C1 pe<=5°: n={n_C1}  |  HMS-T paired-fork qual: n={n_HMST}", flush=True)

def report(name, mask):
    n = int(mask.sum())
    pi = data["pi_isi"][mask]
    qmax = data["q_pred_max_at_isi"][mask]
    qent = data["q_pred_entropy"][mask]
    rsA = float(data["r_stimch_A"][mask].mean()); rsB = float(data["r_stimch_B"][mask].mean())
    rpA = float(data["r_predch_A"][mask].mean()); rpB = float(data["r_predch_B"][mask].mean())
    ex_acc = float(data["correct_A"][mask].mean()); un_acc = float(data["correct_B"][mask].mean())
    print(f"\n[{name}] n={n}")
    print(f"  pi:               mean={pi.mean():.4f}  median={np.median(pi):.4f}  q25={np.percentile(pi, 25):.4f}  q75={np.percentile(pi, 75):.4f}")
    print(f"  q_pred_max:       mean={qmax.mean():.4f}  median={np.median(qmax):.4f}")
    print(f"  q_pred_entropy:   mean={qent.mean():.4f}  median={np.median(qent):.4f}  (lower = sharper prediction)")
    print(f"  r_stimch_A={rsA:.4f}  r_stimch_B={rsB:.4f}  Δr_stimch = {rsA - rsB:+.4f}")
    print(f"  r_predch_A={rpA:.4f}  r_predch_B={rpB:.4f}  Δr_predch = {rpA - rpB:+.4f}")
    print(f"  ex_acc(A)={ex_acc:.4f}  unex_acc(B)={un_acc:.4f}  Δ_decC = {ex_acc - un_acc:+.4f}")
    return {"n": n, "pi_mean": float(pi.mean()), "pi_median": float(np.median(pi)),
            "pi_q25": float(np.percentile(pi, 25)), "pi_q75": float(np.percentile(pi, 75)),
            "q_max_mean": float(qmax.mean()), "q_entropy_mean": float(qent.mean()),
            "delta_r_stimch": rsA - rsB, "delta_r_predch": rpA - rpB,
            "delta_decC": ex_acc - un_acc, "ex_acc": ex_acc, "unex_acc": un_acc}

print("\n========== Open Q2 — Subset characteristics ==========")
out_C1 = report("HMM C1 pe<=5°", S_C1)
out_HMST = report("HMS-T paired-fork qual (3-march + pe<=5° + pi-Q75)", S_HMST)

# H_pi: are pi distributions different?
pi_diff = abs(out_HMST["pi_mean"] - out_C1["pi_mean"])
print(f"\n========== H_pi — pi distributions ==========")
print(f"  |pi_HMST - pi_C1| (mean) = {pi_diff:.4f}")
if pi_diff > 0.10:
    print(f"  → pi distributions differ substantially (Δmean > 0.10)")
else:
    print(f"  → pi distributions are similar (Δmean ≤ 0.10)")

# H_pi test: match on pi (use pi-Q75 of GLOBAL pool to threshold both subsets)
S_C1_piQ75 = S_C1 & (data["pi_isi"] >= pi_q75)
n_C1_pi = int(S_C1_piQ75.sum())
print(f"\n  HMM C1 pe<=5° + pi-Q75 (matched to HMS-T qual filter): n={n_C1_pi}")
if n_C1_pi >= 30:
    rsA = float(data["r_stimch_A"][S_C1_piQ75].mean()); rsB = float(data["r_stimch_B"][S_C1_piQ75].mean())
    rpA = float(data["r_predch_A"][S_C1_piQ75].mean()); rpB = float(data["r_predch_B"][S_C1_piQ75].mean())
    ex_acc = float(data["correct_A"][S_C1_piQ75].mean()); un_acc = float(data["correct_B"][S_C1_piQ75].mean())
    print(f"  r_stimch_A={rsA:.4f}  r_stimch_B={rsB:.4f}  Δr_stimch = {rsA-rsB:+.4f}")
    print(f"  r_predch_A={rpA:.4f}  r_predch_B={rpB:.4f}  Δr_predch = {rpA-rpB:+.4f}")
    print(f"  Δ_decC = {ex_acc - un_acc:+.4f}")
    if (rsA - rsB) * out_HMST["delta_r_stimch"] > 0:   # same sign as HMS-T?
        h_pi = "CONFIRMED — pi-matching on HMM C1 makes Δr_stimch sign FLIP to match HMS-T"
    elif abs(rsA - rsB) < 0.5 * abs(out_C1["delta_r_stimch"]):
        h_pi = "PARTIAL — pi-matching shrinks Δr_stimch but doesn't fully flip"
    else:
        h_pi = "FALSIFIED — pi-matching does NOT change Δr_stimch sign"
    print(f"  H_pi verdict: {h_pi}")
else:
    h_pi = "INSUFFICIENT n for pi-matching test"
    print(f"  {h_pi}")

# H_context: does V2 prediction strength (q_max / q_entropy) differ between subsets?
print(f"\n========== H_context — V2 prediction strength comparison ==========")
qmax_diff = out_HMST["q_max_mean"] - out_C1["q_max_mean"]
qent_diff = out_HMST["q_entropy_mean"] - out_C1["q_entropy_mean"]
print(f"  q_pred_max:     HMM C1 = {out_C1['q_max_mean']:.4f}  vs  HMS-T = {out_HMST['q_max_mean']:.4f}  "
      f"(diff = {qmax_diff:+.4f})")
print(f"  q_pred_entropy: HMM C1 = {out_C1['q_entropy_mean']:.4f}  vs  HMS-T = {out_HMST['q_entropy_mean']:.4f}  "
      f"(diff = {qent_diff:+.4f}) — lower = sharper")
if abs(qmax_diff) > 0.02 or abs(qent_diff) > 0.10:
    h_context = (f"CONFIRMED — V2 prediction strength DIFFERS substantially "
                 f"(q_max Δ={qmax_diff:+.3f}, entropy Δ={qent_diff:+.3f})")
else:
    h_context = "FALSIFIED — V2 prediction strength is similar between subsets"
print(f"  H_context verdict: {h_context}")

# Save
out = {
    "HMM_C1_pe_leq_5": out_C1,
    "HMS_T_paired_fork_qual": out_HMST,
    "H_pi_verdict": h_pi if 'h_pi' in dir() else "n/a",
    "H_context_verdict": h_context,
    "n_C1_pi_Q75_matched": int(n_C1_pi) if 'n_C1_pi' in dir() else 0,
    "config": {"N_total_trials": N, "pi_Q75_global": pi_q75},
}
with open("/tmp/h15_q2_polarity_flip.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n[save] /tmp/h15_q2_polarity_flip.json")
