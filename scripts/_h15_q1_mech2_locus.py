"""Phase 5 Open Q1 — Mech 2 circuit-level locus on HMS / HMS-T.

Three hypotheses to test:
  H_L4   : L4 representation is already weaker on rare-jump stims (BU-only).
  H_PV   : PV ablation reduces Mech 2 dampening (PV laterally suppresses
           rare-jump trials more).
  H_DecC : Dec C's training distribution is biased against rare-jump stims.

Method: 3 ablation conditions on R1+R2 forward pass.
  C0: intact (V2 on, PV on)
  C1: V2 ablated (fb_scale=0)               — isolates non-V2 dampening
  C2: V2 + PV both ablated                  — isolates non-V2-non-PV dampening
For each, record per-presentation r_l4 at stim channel + r_l23 at stim channel
+ decoder accuracy. Apply HMS / HMS-T filters offline.

For H_DecC: train sklearn LogisticRegression on r_l23 from a held-out HMM pool
using V2-ablated condition; re-evaluate HMS / HMS-T splits.
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
print(f"[setup] device={device}", flush=True)
model_cfg, train_cfg, stim_cfg = load_config(CONFIG)
n_ori = int(model_cfg.n_orientations)
period = float(model_cfg.orientation_range)
step_deg = period / n_ori
steps_on = int(train_cfg.steps_on); steps_isi = int(train_cfg.steps_isi)
steps_per = steps_on + steps_isi
batch_size = int(train_cfg.batch_size)

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
def make_net(fb_scale: float, ablate_pv: bool):
    n = LaminarV1V2Network(model_cfg).to(device)
    n.load_state_dict(ckpt["model_state"], strict=False)
    n.eval(); n.oracle_mode = False; n.feedback_scale.fill_(float(fb_scale))
    for p in n.parameters():
        p.requires_grad_(False)
    if ablate_pv:
        # Monkey-patch PV forward to return zeros — kills divisive PV in L4 and subtractive PV in L23
        original_pv_forward = n.pv.forward
        def zero_pv(self, r_l4, r_l23, r_pv_prev):
            return torch.zeros_like(r_pv_prev)
        n.pv.forward = zero_pv.__get__(n.pv, type(n.pv))
    return n

net_C0 = make_net(1.0, False)   # intact
net_C1 = make_net(0.0, False)   # V2 ablated only
net_C2 = make_net(0.0, True)    # V2 + PV ablated

dC_state = torch.load(DEC_C, map_location=device, weights_only=False)
if isinstance(dC_state, dict) and "state_dict" in dC_state:
    dC_state = dC_state["state_dict"]
decC = nn.Linear(n_ori, n_ori, bias=True).to(device)
decC.load_state_dict(dC_state); decC.eval()

# PV-ablation smoke test will be checked from real forward-pass outputs below
# (read r_pv_all from aux2 in the main loop).

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

# Per-presentation buffers — separate for each condition
KEYS = ("pred_err", "pi", "is_amb", "actual_ori", "ori_minus1", "ori_minus2", "true_ch",
        "C0_correct", "C1_correct", "C2_correct",
        "C0_r_l4_stimch", "C1_r_l4_stimch", "C2_r_l4_stimch",
        "C0_r_l23_stimch", "C1_r_l23_stimch", "C2_r_l23_stimch",
        "C0_r_l23", "C1_r_l23", "C2_r_l23",   # full [B, N] for Dec C / sklearn
        "pred_ch")
buf = {k: [] for k in KEYS}
rng = torch.Generator().manual_seed(SEED)

print(f"[forward] {N_BATCHES} batches × bs={batch_size} = {N_BATCHES * batch_size} HMM trials  "
      f"(C0=intact, C1=V2-abl, C2=V2+PV-abl)", flush=True)
for bi_b in range(N_BATCHES):
    md = gen.generate(batch_size, SEQ_LENGTH, generator=rng)
    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

    with torch.no_grad():
        pkg0 = net_C0.pack_inputs(stim_seq, cue_seq, ts_seq)
        r0_l23, _, aux0 = net_C0.forward(pkg0)
        r0_l4 = aux0["r_l4_all"]   # [B, T, N]
        q_pred_all = aux0["q_pred_all"]
        pi_all = aux0["pi_pred_eff_all"]

        pkg1 = net_C1.pack_inputs(stim_seq, cue_seq, ts_seq)
        r1_l23, _, aux1 = net_C1.forward(pkg1)
        r1_l4 = aux1["r_l4_all"]

        pkg2 = net_C2.pack_inputs(stim_seq, cue_seq, ts_seq)
        r2_l23, _, aux2 = net_C2.forward(pkg2)
        r2_l4 = aux2["r_l4_all"]
        if bi_b == 0:
            pv_max = aux2["r_pv_all"].abs().max().item()
            print(f"[smoke C2 PV ablation] r_pv_all max in batch 0 = {pv_max:.6f} (expect 0.0)",
                  flush=True)

    true_ori = md.orientations.to(device)
    is_amb_all = md.is_ambiguous.to(device)
    bi = torch.arange(batch_size, device=device)

    for pres_i in range(1, SEQ_LENGTH):
        t_isi_last = pres_i * steps_per - 1
        q_pred_isi = q_pred_all[:, t_isi_last, :]
        pi_isi = pi_all[:, t_isi_last, 0]
        pred_peak = q_pred_isi.argmax(dim=-1)
        pred_ori = pred_peak.float() * step_deg
        actual_ori = true_ori[:, pres_i]
        pe = circular_distance(pred_ori, actual_ori, period).abs()
        true_ch = (actual_ori / step_deg).round().long() % n_ori

        t0 = pres_i * steps_per + 9; t1 = pres_i * steps_per + 11
        # L23 readout window
        r0_l23_w = r0_l23[:, t0:t1+1, :].mean(dim=1)
        r1_l23_w = r1_l23[:, t0:t1+1, :].mean(dim=1)
        r2_l23_w = r2_l23[:, t0:t1+1, :].mean(dim=1)
        # L4 readout window (same window)
        r0_l4_w = r0_l4[:, t0:t1+1, :].mean(dim=1)
        r1_l4_w = r1_l4[:, t0:t1+1, :].mean(dim=1)
        r2_l4_w = r2_l4[:, t0:t1+1, :].mean(dim=1)

        c0_top1 = decC(r0_l23_w).argmax(-1)
        c1_top1 = decC(r1_l23_w).argmax(-1)
        c2_top1 = decC(r2_l23_w).argmax(-1)
        c0_corr = (c0_top1 == true_ch).float()
        c1_corr = (c1_top1 == true_ch).float()
        c2_corr = (c2_top1 == true_ch).float()

        c0_l4_stimch = r0_l4_w[bi, true_ch.long()]
        c1_l4_stimch = r1_l4_w[bi, true_ch.long()]
        c2_l4_stimch = r2_l4_w[bi, true_ch.long()]
        c0_l23_stimch = r0_l23_w[bi, true_ch.long()]
        c1_l23_stimch = r1_l23_w[bi, true_ch.long()]
        c2_l23_stimch = r2_l23_w[bi, true_ch.long()]

        ori_m1 = true_ori[:, pres_i-1]
        ori_m2 = true_ori[:, pres_i-2] if pres_i >= 2 else torch.full_like(actual_ori, -999.0)

        buf["pred_err"].append(pe.cpu().numpy())
        buf["pi"].append(pi_isi.cpu().numpy())
        buf["is_amb"].append(is_amb_all[:, pres_i].cpu().numpy())
        buf["actual_ori"].append(actual_ori.cpu().numpy())
        buf["ori_minus1"].append(ori_m1.cpu().numpy())
        buf["ori_minus2"].append(ori_m2.cpu().numpy())
        buf["true_ch"].append(true_ch.cpu().numpy())
        buf["C0_correct"].append(c0_corr.cpu().numpy())
        buf["C1_correct"].append(c1_corr.cpu().numpy())
        buf["C2_correct"].append(c2_corr.cpu().numpy())
        buf["C0_r_l4_stimch"].append(c0_l4_stimch.cpu().numpy())
        buf["C1_r_l4_stimch"].append(c1_l4_stimch.cpu().numpy())
        buf["C2_r_l4_stimch"].append(c2_l4_stimch.cpu().numpy())
        buf["C0_r_l23_stimch"].append(c0_l23_stimch.cpu().numpy())
        buf["C1_r_l23_stimch"].append(c1_l23_stimch.cpu().numpy())
        buf["C2_r_l23_stimch"].append(c2_l23_stimch.cpu().numpy())
        buf["C0_r_l23"].append(r0_l23_w.cpu().numpy())
        buf["C1_r_l23"].append(r1_l23_w.cpu().numpy())
        buf["C2_r_l23"].append(r2_l23_w.cpu().numpy())
        buf["pred_ch"].append(pred_peak.cpu().numpy())

    if (bi_b + 1) % 20 == 0:
        print(f"  batch {bi_b+1}/{N_BATCHES}", flush=True)

# Concat
data = {}
for k, v in buf.items():
    if k.endswith("_r_l23"):
        data[k] = np.concatenate(v, axis=0)   # [N_records, n_ori]
    else:
        data[k] = np.concatenate(v, axis=0)

print(f"[N] per-presentation records: {data['C0_correct'].shape[0]}", flush=True)

# ============================================================================
# Define HMS / HMS-T filters (same as Phase 4d / 4e)
# ============================================================================
def signed_circ(a, b, p):
    d = (a - b) % p
    return np.where(d > p / 2, d - p, d)
keep = ~data["is_amb"].astype(bool)
pi_q75 = float(np.percentile(data["pi"][keep], 75))
print(f"[pi Q75 global, kept] = {pi_q75:.4f}", flush=True)
d_ctx = signed_circ(data["ori_minus1"], data["ori_minus2"], period)
d_probe = signed_circ(data["actual_ori"], data["ori_minus1"], period)
ctx_match_step = np.abs(np.abs(d_ctx) - 5.0) <= 1.0
probe_match_step = np.abs(np.abs(d_probe) - 5.0) <= 1.0
same_dir = (np.sign(d_ctx) == np.sign(d_probe)) & (np.abs(d_ctx) > 1e-6)
is_3march = ctx_match_step & probe_match_step & same_dir
is_march_jump = ctx_match_step & (np.abs(d_probe) >= 75.0)

paradigms = {
    "HMS native":   {"ex": keep & is_3march & (data["pi"] >= pi_q75),
                     "unex": keep & is_march_jump & (data["pi"] >= pi_q75)},
    "HMS-T native": {"ex": keep & is_3march & (data["pred_err"] <= 5.0) & (data["pi"] >= pi_q75),
                     "unex": keep & is_march_jump & (data["pred_err"] > 60.0) & (data["pi"] >= pi_q75)},
    "M3R native":   {"ex": keep & (data["pred_err"] <= 5.0) & (data["pi"] >= pi_q75),
                     "unex": keep & (data["pred_err"] > 20.0) & (data["pi"] >= pi_q75)},
}

# ============================================================================
# H_L4 — L4 representation strength on rare-jump stims (in V2-ablated C1)
# ============================================================================
print(f"\n========== H_L4 — L4 representation on ex vs unex (V2-ablated C1) ==========")
print(f"  {'paradigm':14s}  {'n_ex':>5s}  {'n_un':>5s}  "
      f"{'L4_stimch_ex':>12s}  {'L4_stimch_un':>12s}  {'ΔL4_stim':>9s}  "
      f"{'L23_stimch_ex':>13s}  {'L23_stimch_un':>13s}  {'ΔL23_stim':>10s}")
hL4_out = {}
for name, sel in paradigms.items():
    n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
    if n_ex < 20 or n_un < 20:
        print(f"  {name:14s}  insufficient")
        continue
    # Use V2-ablated condition (C1) — isolates intrinsic L4 / L23 representation
    l4_ex = float(data["C1_r_l4_stimch"][sel["ex"]].mean())
    l4_un = float(data["C1_r_l4_stimch"][sel["unex"]].mean())
    l23_ex = float(data["C1_r_l23_stimch"][sel["ex"]].mean())
    l23_un = float(data["C1_r_l23_stimch"][sel["unex"]].mean())
    print(f"  {name:14s}  {n_ex:5d}  {n_un:5d}  "
          f"{l4_ex:12.4f}  {l4_un:12.4f}  {l4_ex-l4_un:+9.4f}  "
          f"{l23_ex:13.4f}  {l23_un:13.4f}  {l23_ex-l23_un:+10.4f}")
    hL4_out[name] = {
        "n_ex": n_ex, "n_unex": n_un,
        "C1_l4_stimch_ex": l4_ex, "C1_l4_stimch_unex": l4_un,
        "C1_delta_l4_stim": l4_ex - l4_un,
        "C1_l23_stimch_ex": l23_ex, "C1_l23_stimch_unex": l23_un,
        "C1_delta_l23_stim": l23_ex - l23_un,
    }

# H_L4 verdict per paradigm
print("\n  H_L4 per-paradigm verdict (with V2 ablated, isolating non-V2):")
for name, d in hL4_out.items():
    if d["C1_delta_l4_stim"] > 0.001:   # ex has higher L4 stim activity than unex by ≥ 0.001
        v = "CONFIRMED — L4 rep stronger on ex (3-march) than unex (rare-jump)"
    elif abs(d["C1_delta_l4_stim"]) <= 0.001:
        v = "FALSIFIED — L4 rep approximately equal between ex and unex"
    else:
        v = "INVERTED — L4 unex > ex (opposite of H_L4)"
    print(f"    {name:14s}: ΔL4_stim = {d['C1_delta_l4_stim']:+.4f} → {v}")

# ============================================================================
# H_PV — PV ablation (C2) compared to V2-only ablation (C1)
# ============================================================================
print(f"\n========== H_PV — PV ablation effect on Mech 2 dampening ==========")
print(f"  {'paradigm':14s}  {'C0_Δ_decC':>9s}  {'C1_Δ_decC':>9s}  {'C2_Δ_decC':>9s}  "
      f"{'V2_contrib':>10s}  {'PV_contrib_after_V2_abl':>22s}  {'H_PV':>30s}")
hPV_out = {}
for name, sel in paradigms.items():
    n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
    if n_ex < 20 or n_un < 20: continue
    c0_ex = float(data["C0_correct"][sel["ex"]].mean()); c0_un = float(data["C0_correct"][sel["unex"]].mean())
    c1_ex = float(data["C1_correct"][sel["ex"]].mean()); c1_un = float(data["C1_correct"][sel["unex"]].mean())
    c2_ex = float(data["C2_correct"][sel["ex"]].mean()); c2_un = float(data["C2_correct"][sel["unex"]].mean())
    d0 = c0_ex - c0_un; d1 = c1_ex - c1_un; d2 = c2_ex - c2_un
    v2_contrib = d0 - d1
    pv_contrib_after_v2 = d1 - d2   # change in Δ from C1 to C2 (= PV contribution AFTER V2 already ablated)
    abs_d1, abs_d2 = abs(d1), abs(d2)
    if abs_d1 > 1e-6:
        pv_change_pct = (abs_d1 - abs_d2) / abs_d1 * 100
    else:
        pv_change_pct = 0
    if pv_change_pct >= 30:
        h_pv = f"CONFIRMED — PV ablation reduces |Δ_decC| by {pv_change_pct:.0f}%"
    elif pv_change_pct <= -30:
        h_pv = f"INVERTED — PV ablation amplifies |Δ_decC| by {-pv_change_pct:.0f}%"
    else:
        h_pv = f"FALSIFIED — PV ablation changes |Δ_decC| by {pv_change_pct:+.0f}%"
    print(f"  {name:14s}  {d0:+9.4f}  {d1:+9.4f}  {d2:+9.4f}  "
          f"{v2_contrib:+10.4f}  {pv_contrib_after_v2:+22.4f}  {h_pv}")
    hPV_out[name] = {
        "C0_delta_decC": d0, "C1_delta_decC": d1, "C2_delta_decC": d2,
        "V2_contribution": v2_contrib,
        "PV_contribution_after_V2": pv_contrib_after_v2,
        "PV_change_pct": pv_change_pct, "H_PV_verdict": h_pv,
    }

# ============================================================================
# H_DecC — sklearn LBFGS retrained on r_l23 from same eval pool (V2-ablated)
# ============================================================================
print(f"\n========== H_DecC — Retrained LBFGS classifier on V2-ablated r_l23 ==========")
from sklearn.linear_model import LogisticRegression
# Use the V2-ablated C1 r_l23 + actual stim channel as labels.
# Train/test split: 50/50, random.
X_all = data["C1_r_l23"]   # [N_records, n_ori]
y_all = data["true_ch"]
n_total = X_all.shape[0]
np.random.seed(0)
perm = np.random.permutation(n_total)
n_train = n_total // 2
tr_idx = perm[:n_train]; te_idx = perm[n_train:]
print(f"  Training sklearn LBFGS on {n_train} records (V2-ablated r_l23)... ", end="", flush=True)
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
clf.fit(X_all[tr_idx], y_all[tr_idx])
acc_test = (clf.predict(X_all[te_idx]) == y_all[te_idx]).mean()
print(f"test_acc={acc_test:.4f}", flush=True)

# Apply to HMS / HMS-T splits using ABLATED-C1 r_l23 features (intact features would
# include V2 effects, but we want the underlying non-V2 representation).
print("\n  H_DecC results — sklearn LBFGS-on-pool vs Dec C, on V2-ablated (C1) features:")
print(f"  {'paradigm':14s}  {'n_ex':>5s}  {'n_un':>5s}  "
      f"{'DecC_Δ':>8s}  {'LBFGS_Δ':>8s}  {'change':>8s}  {'H_DecC':>30s}")
hDecC_out = {}
for name, sel in paradigms.items():
    n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
    if n_ex < 20 or n_un < 20: continue
    decC_corr_ex = data["C1_correct"][sel["ex"]].mean()
    decC_corr_un = data["C1_correct"][sel["unex"]].mean()
    decC_d = float(decC_corr_ex - decC_corr_un)
    lbfgs_pred_ex = clf.predict(X_all[sel["ex"]])
    lbfgs_pred_un = clf.predict(X_all[sel["unex"]])
    lbfgs_corr_ex = (lbfgs_pred_ex == y_all[sel["ex"]]).mean()
    lbfgs_corr_un = (lbfgs_pred_un == y_all[sel["unex"]]).mean()
    lbfgs_d = float(lbfgs_corr_ex - lbfgs_corr_un)
    if abs(decC_d) > 1e-6:
        change = (abs(decC_d) - abs(lbfgs_d)) / abs(decC_d) * 100
    else:
        change = 0
    if change >= 50:
        v = f"CONFIRMED — LBFGS reduces |Δ| by {change:.0f}% (DecC training distrib was source)"
    elif abs(change) < 30:
        v = f"FALSIFIED — LBFGS Δ ≈ DecC Δ (DecC training is NOT the source)"
    else:
        v = f"PARTIAL — LBFGS reduces |Δ| by {change:.0f}%"
    print(f"  {name:14s}  {n_ex:5d}  {n_un:5d}  "
          f"{decC_d:+8.4f}  {lbfgs_d:+8.4f}  {change:+7.0f}%  {v:>30s}")
    hDecC_out[name] = {
        "decC_delta": decC_d, "lbfgs_delta": lbfgs_d,
        "change_pct": change, "H_DecC_verdict": v,
    }

# Save
out = {
    "H_L4": hL4_out,
    "H_PV": hPV_out,
    "H_DecC": hDecC_out,
    "config": {
        "n_records": int(data["C0_correct"].shape[0]),
        "pi_Q75": pi_q75,
        "lbfgs_test_acc_on_V2_ablated_pool": float(acc_test),
    },
}
with open("/tmp/h15_q1_mech2_locus.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n[save] /tmp/h15_q1_mech2_locus.json")
