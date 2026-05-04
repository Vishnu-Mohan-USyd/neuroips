"""Task #14 Phase 4 — Causal interventions on channel-resolved suppression mechanism.

Experiments:
  4.1 V2-feedback ablation: set net.feedback_scale = 0 at inference; check
      whether channel-resolved suppression (r_at_predch, r_at_stimch deltas)
      vanishes.
  4.2 Channel perturbation: artificially multiply r_l23 at V2-predicted channel
      by 0.70 (~30% suppression matching empirical magnitude); compare resulting
      Δ_decC pattern to the natural one.
  4.3 Confound regression: per-trial logistic regression of decoder-correct on
      r_stimch + pe + difficulty proxies; report partial R²/coefficients.

Network: R1+R2 at results/simple_dual/emergent_seed42/checkpoint.pt
Paradigm: HMM C1 (focused, HMM cue), N=4000 trials, seed 42.
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
N_TRIALS = 4000
SEQ_LENGTH = 25
PERTURB_FACTOR = 0.70   # 30% suppression at V2-pred channel for Exp 4.2

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
def make_net(fb_scale):
    n = LaminarV1V2Network(model_cfg).to(device)
    n.load_state_dict(ckpt["model_state"], strict=False)
    n.eval(); n.oracle_mode = False; n.feedback_scale.fill_(float(fb_scale))
    for p in n.parameters():
        p.requires_grad_(False)
    return n
net_intact = make_net(1.0)
net_ablated = make_net(0.0)
print(f"[setup] intact fb_scale={float(net_intact.feedback_scale):.2f}; "
      f"ablated fb_scale={float(net_ablated.feedback_scale):.2f}", flush=True)

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

# Aggregate buffers
all_data = {
    cond: {k: [] for k in ("pe_A", "pe_B", "pred_ch", "true_ch_ex", "true_ch_unex",
                            "r_predch_A", "r_predch_B", "r_stimch_A", "r_stimch_B",
                            "r_norm_A", "r_norm_B", "correct_A", "correct_B",
                            "r_probe_A", "r_probe_B",
                            # 4.2 perturbed
                            "correct_A_perturbed", "correct_B_perturbed",
                            "pi", "keep")}
    for cond in ("intact", "ablated")
}

n_done = 0
batch = 1000
while n_done < N_TRIALS:
    b = min(batch, N_TRIALS - n_done)
    md = gen.generate(b, SEQ_LENGTH, generator=rng)
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

    for cond_name, net in [("intact", net_intact), ("ablated", net_ablated)]:
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

        bi_arange = torch.arange(b, device=device)
        idx_pred = pred_peak.long()
        r_predch_A = r_probe_A[bi_arange, idx_pred]
        r_predch_B = r_probe_B[bi_arange, idx_pred]
        r_stimch_A = r_probe_A[bi_arange, true_ch_ex.long()]
        r_stimch_B = r_probe_B[bi_arange, true_ch_unex.long()]

        # Decoder accuracies
        pred_dec_A = decC(r_probe_A).argmax(-1)
        pred_dec_B = decC(r_probe_B).argmax(-1)
        correct_A = (pred_dec_A == true_ch_ex).float()
        correct_B = (pred_dec_B == true_ch_unex).float()

        # ========== Exp 4.2 — channel perturbation ==========
        # Apply ONLY in intact condition (perturbing already-ablated network is meaningless)
        if cond_name == "intact":
            r_pert_A = r_probe_A.clone()
            r_pert_B = r_probe_B.clone()
            # Multiply V2-pred channel by PERTURB_FACTOR for both branches
            r_pert_A[bi_arange, idx_pred] = r_pert_A[bi_arange, idx_pred] * PERTURB_FACTOR
            r_pert_B[bi_arange, idx_pred] = r_pert_B[bi_arange, idx_pred] * PERTURB_FACTOR
            pred_pert_A = decC(r_pert_A).argmax(-1)
            pred_pert_B = decC(r_pert_B).argmax(-1)
            correct_A_p = (pred_pert_A == true_ch_ex).float()
            correct_B_p = (pred_pert_B == true_ch_unex).float()
        else:
            # placeholder zeros (won't be used for ablated)
            correct_A_p = torch.zeros(b, device=device)
            correct_B_p = torch.zeros(b, device=device)

        d = all_data[cond_name]
        d["pe_A"].append(pe_A.cpu().numpy())
        d["pe_B"].append(pe_B.cpu().numpy())
        d["pred_ch"].append(pred_peak.cpu().numpy())
        d["true_ch_ex"].append(true_ch_ex.cpu().numpy())
        d["true_ch_unex"].append(true_ch_unex.cpu().numpy())
        d["r_predch_A"].append(r_predch_A.cpu().numpy())
        d["r_predch_B"].append(r_predch_B.cpu().numpy())
        d["r_stimch_A"].append(r_stimch_A.cpu().numpy())
        d["r_stimch_B"].append(r_stimch_B.cpu().numpy())
        d["r_norm_A"].append(r_probe_A.norm(dim=1).cpu().numpy())
        d["r_norm_B"].append(r_probe_B.norm(dim=1).cpu().numpy())
        d["r_probe_A"].append(r_probe_A.cpu().numpy())
        d["r_probe_B"].append(r_probe_B.cpu().numpy())
        d["correct_A"].append(correct_A.cpu().numpy())
        d["correct_B"].append(correct_B.cpu().numpy())
        d["correct_A_perturbed"].append(correct_A_p.cpu().numpy())
        d["correct_B_perturbed"].append(correct_B_p.cpu().numpy())
        d["pi"].append(pi_isi.cpu().numpy())
        d["keep"].append((~is_amb).cpu().numpy())

    n_done += b
    print(f"  collected {n_done}/{N_TRIALS}", flush=True)

for cond in all_data:
    for k in all_data[cond]:
        all_data[cond][k] = np.concatenate(all_data[cond][k])

# ===========================================================================
# Per-stratum analyses
# ===========================================================================
def stratify_print(label, d):
    print(f"\n========== {label} ==========")
    pe = d["pe_A"].copy(); pe[~d["keep"]] = np.nan
    print(f"V2 pred_err mean={np.nanmean(pe):.2f}° median={np.nanmedian(pe):.2f}°")
    bins = {
        "pe<=5":     d["keep"] & (d["pe_A"] <= 5.0),
        "5<pe<=15":  d["keep"] & (d["pe_A"] > 5.0)  & (d["pe_A"] <= 15.0),
        "15<pe<=30": d["keep"] & (d["pe_A"] > 15.0) & (d["pe_A"] <= 30.0),
        "30<pe<=60": d["keep"] & (d["pe_A"] > 30.0) & (d["pe_A"] <= 60.0),
        "pe>60":     d["keep"] & (d["pe_A"] > 60.0),
    }
    print(f"  {'stratum':12s}  {'n':>5s}  {'r_predch_A':>11s}  {'r_predch_B':>11s}  "
          f"{'r_stimch_A':>11s}  {'r_stimch_B':>11s}  {'Δr_stimch':>10s}  "
          f"{'ex_acc':>7s}  {'unex_acc':>9s}  {'Δ_decC':>9s}")
    out = {}
    for name, m in bins.items():
        n = int(m.sum())
        if n < 20:
            continue
        rpA = float(d["r_predch_A"][m].mean())
        rpB = float(d["r_predch_B"][m].mean())
        rsA = float(d["r_stimch_A"][m].mean())
        rsB = float(d["r_stimch_B"][m].mean())
        ex = float(d["correct_A"][m].mean())
        un = float(d["correct_B"][m].mean())
        print(f"  {name:12s}  {n:5d}  {rpA:11.4f}  {rpB:11.4f}  "
              f"{rsA:11.4f}  {rsB:11.4f}  {rsA-rsB:+10.4f}  "
              f"{ex:7.4f}  {un:9.4f}  {ex-un:+9.4f}")
        out[name] = {"n": n, "r_predch_A": rpA, "r_predch_B": rpB,
                     "r_stimch_A": rsA, "r_stimch_B": rsB,
                     "delta_r_stimch": rsA - rsB,
                     "ex_acc": ex, "unex_acc": un, "delta_decC": ex - un}
    return out

intact_out = stratify_print("Exp 4.1 — Intact (fb_scale=1.0)", all_data["intact"])
ablated_out = stratify_print("Exp 4.1 — Ablated (fb_scale=0.0)", all_data["ablated"])

# ===========================================================================
# Exp 4.2 — channel perturbation results
# ===========================================================================
print("\n\n========== Exp 4.2 — Manual channel perturbation (intact net, r_l23 at pred_ch × 0.70) ==========")
d = all_data["intact"]
print(f"  {'stratum':12s}  {'n':>5s}  {'natural_Δ':>10s}  {'perturbed_Δ':>11s}  "
      f"{'natural_ex':>10s}  {'natural_un':>10s}  {'perturb_ex':>10s}  {'perturb_un':>10s}")
exp42_out = {}
bins = {
    "pe<=5":     d["keep"] & (d["pe_A"] <= 5.0),
    "5<pe<=15":  d["keep"] & (d["pe_A"] > 5.0)  & (d["pe_A"] <= 15.0),
    "15<pe<=30": d["keep"] & (d["pe_A"] > 15.0) & (d["pe_A"] <= 30.0),
    "30<pe<=60": d["keep"] & (d["pe_A"] > 30.0) & (d["pe_A"] <= 60.0),
    "pe>60":     d["keep"] & (d["pe_A"] > 60.0),
}
for name, m in bins.items():
    n = int(m.sum())
    if n < 20: continue
    nat_ex = float(d["correct_A"][m].mean()); nat_un = float(d["correct_B"][m].mean())
    pert_ex = float(d["correct_A_perturbed"][m].mean()); pert_un = float(d["correct_B_perturbed"][m].mean())
    print(f"  {name:12s}  {n:5d}  {nat_ex-nat_un:+10.4f}  {pert_ex-pert_un:+11.4f}  "
          f"{nat_ex:10.4f}  {nat_un:10.4f}  {pert_ex:10.4f}  {pert_un:10.4f}")
    exp42_out[name] = {"n": n, "natural_delta": nat_ex - nat_un, "perturb_delta": pert_ex - pert_un,
                        "natural_ex": nat_ex, "natural_un": nat_un,
                        "perturb_ex": pert_ex, "perturb_un": pert_un}

# ===========================================================================
# Exp 4.3 — Confound regression
# ===========================================================================
print("\n\n========== Exp 4.3 — Per-trial regression: decoder-correct ~ r_stimch + pe + r_norm ==========")
from sklearn.linear_model import LogisticRegression
d = all_data["intact"]
m = d["keep"]
# Pool A and B trials together (Pass A: target=true_ch_ex, Pass B: target=true_ch_unex)
# Predictors: r_stimch (own branch), pe (V2 pred_err vs own branch's stim), r_norm.
# Pass A's pe is pe_A; Pass B's pe (V2 vs Pass B's stim) is pe_B.
X_A = np.column_stack([d["r_stimch_A"][m], d["pe_A"][m], d["r_norm_A"][m]])
X_B = np.column_stack([d["r_stimch_B"][m], d["pe_B"][m], d["r_norm_B"][m]])
y_A = d["correct_A"][m].astype(int)
y_B = d["correct_B"][m].astype(int)
X = np.vstack([X_A, X_B]); y = np.concatenate([y_A, y_B])
# Standardize
X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
clf = LogisticRegression(max_iter=2000)
clf.fit(X_norm, y)
print(f"  n_pooled={len(y)} (A+B Pass union), classes={np.bincount(y)}")
print(f"  Standardized coeffs (z-scored predictors):")
for name, c in zip(["r_stimch", "pe_to_own_stim", "r_norm"], clf.coef_[0]):
    print(f"    {name:18s}: {c:+.4f}")
print(f"  intercept: {clf.intercept_[0]:+.4f}")
acc_full = (clf.predict(X_norm) == y).mean()
print(f"  full-model accuracy on training data: {acc_full:.4f}")
# Drop r_stimch
X2 = X_norm[:, [1, 2]]
clf2 = LogisticRegression(max_iter=2000); clf2.fit(X2, y)
acc_no_rstim = (clf2.predict(X2) == y).mean()
print(f"  without r_stimch: {acc_no_rstim:.4f}  → r_stimch contributes Δacc = {acc_full - acc_no_rstim:+.4f}")
# Drop pe
X3 = X_norm[:, [0, 2]]
clf3 = LogisticRegression(max_iter=2000); clf3.fit(X3, y)
acc_no_pe = (clf3.predict(X3) == y).mean()
print(f"  without pe:       {acc_no_pe:.4f}  → pe contributes Δacc      = {acc_full - acc_no_pe:+.4f}")

# ===========================================================================
# Save
# ===========================================================================
out = {
    "exp_4_1_intact": intact_out,
    "exp_4_1_ablated": ablated_out,
    "exp_4_2_perturbation": exp42_out,
    "exp_4_3_regression": {
        "coeffs_standardized": {n: float(c) for n, c in zip(["r_stimch", "pe_to_own_stim", "r_norm"], clf.coef_[0])},
        "intercept": float(clf.intercept_[0]),
        "full_acc": float(acc_full),
        "no_rstim_acc": float(acc_no_rstim),
        "no_pe_acc": float(acc_no_pe),
    },
    "perturb_factor": PERTURB_FACTOR,
}
with open("/tmp/h14_causal.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n[save] /tmp/h14_causal.json")
np.savez("/tmp/h14_causal_pertrial.npz",
         **{f"intact_{k}": v for k, v in all_data["intact"].items() if k not in ("r_probe_A", "r_probe_B")},
         **{f"ablated_{k}": v for k, v in all_data["ablated"].items() if k not in ("r_probe_A", "r_probe_B")})
print("[save] /tmp/h14_causal_pertrial.npz")
