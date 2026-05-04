"""Phase 7 Section 1 + 3 — Dec A cross-check for V2-ablation on observational
paradigms and W_rec amplifier ablation. R1+R2 only.

Reruns:
  Section 1: V2-ablation on M3R / HMS / HMS-T / VCD-test3 (native + modified)
             with both Dec C and Dec A readouts. Verify Mech 1 vs Mech 2
             verdict per paradigm under Dec A.
  Section 3: V2 + W_rec ablation on HMS / HMS-T / M3R control under both
             Dec C and Dec A readouts. Verify W_rec amplifier verdict under
             Dec A.

Five forward passes:
  P1: native input,   fb_scale=1.0  (intact)
  P2: native input,   fb_scale=0.0  (V2 ablated only)
  P3: native input,   fb_scale=0.0 + W_rec zeroed   (Section 3 test condition)
  P4: modified input, fb_scale=1.0  (intact)
  P5: modified input, fb_scale=0.0  (V2 ablated only)
"""
from __future__ import annotations
import os, sys, json, types
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.training.trainer import build_stimulus_sequence
from src.stimulus.sequences import HMMSequenceGenerator
from src.utils import circular_distance
from src.model.populations import rectified_softplus
from scripts.cross_decoder_eval import apply_input_overrides

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


def make_net(fb: float, ablate_wrec: bool = False):
    n = LaminarV1V2Network(model_cfg).to(device)
    n.load_state_dict(ckpt["model_state"], strict=False)
    n.eval(); n.oracle_mode = False; n.feedback_scale.fill_(float(fb))
    for p in n.parameters():
        p.requires_grad_(False)
    if ablate_wrec:
        # Patch L23.forward to skip W_rec recurrence term
        def skip_wrec_forward(self_l23, r_l4, r_l23_prev, template_modulation, r_som, r_pv):
            ff = F.linear(r_l4, self_l23.W_l4_to_l23)
            excitatory_drive = ff + template_modulation   # NO rec
            som_term = self_l23.w_som(r_som)
            l23_drive = excitatory_drive - som_term - self_l23.w_pv_l23(r_pv)
            r_l23 = r_l23_prev + (self_l23.dt / self_l23.tau_l23) * (
                -r_l23_prev + rectified_softplus(l23_drive)
            )
            return r_l23
        n.l23.forward = types.MethodType(skip_wrec_forward, n.l23)
    return n


# Decoders
dC_state = torch.load(DEC_C, map_location=device, weights_only=False)
if isinstance(dC_state, dict) and "state_dict" in dC_state:
    dC_state = dC_state["state_dict"]
decC = nn.Linear(n_ori, n_ori, bias=True).to(device)
decC.load_state_dict(dC_state); decC.eval()

decA = nn.Linear(n_ori, n_ori, bias=True).to(device)
decA.load_state_dict(ckpt["loss_heads"]["orientation_decoder"])
decA.eval()
print(f"[setup] Dec C ||W||={decC.weight.norm().item():.2f}  "
      f"Dec A ||W||={decA.weight.norm().item():.2f}", flush=True)

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


def run_pass(net, variant: str, label: str):
    """variant ∈ {'native', 'modified'}. Collect per-presentation records."""
    print(f"\n[forward {label}]  variant={variant}  N_batches={N_BATCHES}  bs={batch_size}", flush=True)
    rng = torch.Generator().manual_seed(SEED)
    KEYS = ("pred_err", "pi", "is_amb", "actual_ori", "ori_minus1", "ori_minus2", "true_ch",
            "decC_correct", "decA_correct",
            "r_l23_stimch", "pred_ch")
    buf = {k: [] for k in KEYS}
    for bi_b in range(N_BATCHES):
        md = gen.generate(batch_size, SEQ_LENGTH, generator=rng)
        if variant == "modified":
            apply_input_overrides(md, period, n_ori)
        stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
        stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

        with torch.no_grad():
            pkg = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23, _, aux = net.forward(pkg)
            q_pred_all = aux["q_pred_all"]
            pi_all = aux["pi_pred_eff_all"]

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
            r_w = r_l23[:, t0:t1+1, :].mean(dim=1)
            decC_top1 = decC(r_w).argmax(-1)
            decA_top1 = decA(r_w).argmax(-1)
            decC_corr = (decC_top1 == true_ch).float()
            decA_corr = (decA_top1 == true_ch).float()
            r_stim = r_w[bi, true_ch.long()]
            ori_m1 = true_ori[:, pres_i-1]
            ori_m2 = true_ori[:, pres_i-2] if pres_i >= 2 else torch.full_like(actual_ori, -999.0)

            buf["pred_err"].append(pe.cpu().numpy())
            buf["pi"].append(pi_isi.cpu().numpy())
            buf["is_amb"].append(is_amb_all[:, pres_i].cpu().numpy())
            buf["actual_ori"].append(actual_ori.cpu().numpy())
            buf["ori_minus1"].append(ori_m1.cpu().numpy())
            buf["ori_minus2"].append(ori_m2.cpu().numpy())
            buf["true_ch"].append(true_ch.cpu().numpy())
            buf["decC_correct"].append(decC_corr.cpu().numpy())
            buf["decA_correct"].append(decA_corr.cpu().numpy())
            buf["r_l23_stimch"].append(r_stim.cpu().numpy())
            buf["pred_ch"].append(pred_peak.cpu().numpy())
        if (bi_b + 1) % 20 == 0:
            print(f"  [{label}] batch {bi_b+1}/{N_BATCHES}", flush=True)
    return {k: np.concatenate(v) for k, v in buf.items()}


# 5 forward passes
data_P1 = run_pass(make_net(1.0),                     "native",   "P1=native_intact")
data_P2 = run_pass(make_net(0.0),                     "native",   "P2=native_V2abl")
data_P3 = run_pass(make_net(0.0, ablate_wrec=True),   "native",   "P3=native_V2+Wrec_abl")
data_P4 = run_pass(make_net(1.0),                     "modified", "P4=modified_intact")
data_P5 = run_pass(make_net(0.0),                     "modified", "P5=modified_V2abl")

# Filters built on each variant separately
def signed_circ(a, b, p):
    d = (a - b) % p
    return np.where(d > p / 2, d - p, d)

def build_filters(d):
    keep = ~d["is_amb"].astype(bool)
    pi_q75 = float(np.percentile(d["pi"][keep], 75))
    d_ctx = signed_circ(d["ori_minus1"], d["ori_minus2"], period)
    d_probe = signed_circ(d["actual_ori"], d["ori_minus1"], period)
    ctx_match_step = np.abs(np.abs(d_ctx) - 5.0) <= 1.0
    probe_match_step = np.abs(np.abs(d_probe) - 5.0) <= 1.0
    same_dir = (np.sign(d_ctx) == np.sign(d_probe)) & (np.abs(d_ctx) > 1e-6)
    is_3march = ctx_match_step & probe_match_step & same_dir
    is_march_jump = ctx_match_step & (np.abs(d_probe) >= 75.0)
    return {
        "M3R":       {"ex": keep & (d["pred_err"] <= 5.0) & (d["pi"] >= pi_q75),
                      "unex": keep & (d["pred_err"] > 20.0) & (d["pi"] >= pi_q75)},
        "HMS":       {"ex": keep & is_3march & (d["pi"] >= pi_q75),
                      "unex": keep & is_march_jump & (d["pi"] >= pi_q75)},
        "HMS-T":     {"ex": keep & is_3march & (d["pred_err"] <= 5.0) & (d["pi"] >= pi_q75),
                      "unex": keep & is_march_jump & (d["pred_err"] > 60.0) & (d["pi"] >= pi_q75)},
        "VCD-test3": {"ex": keep & (d["pred_err"] <= 10.0),
                      "unex": keep & (d["pred_err"] > 20.0)},
    }, pi_q75


def section1_table(data_intact, data_ablated, variant_label):
    filters, pi_q75 = build_filters(data_intact)
    print(f"\n========== Phase 7 §1 — {variant_label} (Dec A vs Dec C side-by-side) ==========")
    print(f"  pi_Q75 (intact pool) = {pi_q75:.4f}")
    print(f"  {'paradigm':28s}  {'n_ex':>5s}  {'n_un':>5s}  "
          f"{'decC_int':>9s}  {'decC_abl':>9s}  {'decC_chg':>9s}  "
          f"{'decA_int':>9s}  {'decA_abl':>9s}  {'decA_chg':>9s}  {'agree?':>7s}")
    out = {}
    for pname in ["M3R", "HMS", "HMS-T", "VCD-test3"]:
        sel = filters[pname]
        n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
        if n_ex < 20 or n_un < 20:
            print(f"  {pname:28s}  insufficient")
            continue
        # Dec C
        dC_int = float(data_intact["decC_correct"][sel["ex"]].mean() - data_intact["decC_correct"][sel["unex"]].mean())
        dC_abl = float(data_ablated["decC_correct"][sel["ex"]].mean() - data_ablated["decC_correct"][sel["unex"]].mean())
        # Dec A
        dA_int = float(data_intact["decA_correct"][sel["ex"]].mean() - data_intact["decA_correct"][sel["unex"]].mean())
        dA_abl = float(data_ablated["decA_correct"][sel["ex"]].mean() - data_ablated["decA_correct"][sel["unex"]].mean())
        # Verdict via sign of V2 contribution = (intact − ablated)
        v2_C = dC_int - dC_abl
        v2_A = dA_int - dA_abl
        # Mech 1 if V2 contrib in dampening direction (negative); Mech 2 if positive
        mech_C = "Mech1" if v2_C < -0.005 else ("Mech2" if v2_C > 0.005 else "INC")
        mech_A = "Mech1" if v2_A < -0.005 else ("Mech2" if v2_A > 0.005 else "INC")
        agree = "YES" if mech_C == mech_A else "NO"
        # Compute % change
        chg_C = (abs(dC_int) - abs(dC_abl)) / max(abs(dC_int), 1e-6) * 100
        chg_A = (abs(dA_int) - abs(dA_abl)) / max(abs(dA_int), 1e-6) * 100
        print(f"  {pname+' '+variant_label:28s}  {n_ex:5d}  {n_un:5d}  "
              f"{dC_int:+9.4f}  {dC_abl:+9.4f}  {chg_C:+8.0f}%  "
              f"{dA_int:+9.4f}  {dA_abl:+9.4f}  {chg_A:+8.0f}%  {agree:>7s}  "
              f"[{mech_C}/{mech_A}]")
        out[pname] = {"n_ex": n_ex, "n_unex": n_un,
                       "decC_intact": dC_int, "decC_ablated": dC_abl,
                       "decA_intact": dA_int, "decA_ablated": dA_abl,
                       "v2_contrib_decC": v2_C, "v2_contrib_decA": v2_A,
                       "mech_decC": mech_C, "mech_decA": mech_A, "agree": agree}
    return out


s1_native = section1_table(data_P1, data_P2, "native")
s1_modified = section1_table(data_P4, data_P5, "modified")

# Section 3: W_rec ablation effect on Δ_decA + Δ_decC for HMS / HMS-T / M3R
def section3_table(data_C1, data_C_Wrec):
    """C1 = V2 ablated only. C_Wrec = V2 + W_rec ablated.
    Both forward passes use V2-ablated network's pi/pred_err for filtering."""
    filters, pi_q75 = build_filters(data_C1)
    print(f"\n========== Phase 7 §3 — W_rec ablation (Dec A vs Dec C side-by-side) ==========")
    print(f"  pi_Q75 (V2-abl pool) = {pi_q75:.4f}")
    print(f"  {'paradigm':14s}  {'n_ex':>5s}  {'n_un':>5s}  "
          f"{'decC_C1':>9s}  {'decC_Wrec':>9s}  {'decC_chg':>9s}  "
          f"{'decA_C1':>9s}  {'decA_Wrec':>9s}  {'decA_chg':>9s}  {'agree?':>7s}")
    out = {}
    for pname in ["HMS", "HMS-T", "M3R"]:
        sel = filters[pname]
        n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
        if n_ex < 20 or n_un < 20: continue
        dC_c1 = float(data_C1["decC_correct"][sel["ex"]].mean() - data_C1["decC_correct"][sel["unex"]].mean())
        dC_wr = float(data_C_Wrec["decC_correct"][sel["ex"]].mean() - data_C_Wrec["decC_correct"][sel["unex"]].mean())
        dA_c1 = float(data_C1["decA_correct"][sel["ex"]].mean() - data_C1["decA_correct"][sel["unex"]].mean())
        dA_wr = float(data_C_Wrec["decA_correct"][sel["ex"]].mean() - data_C_Wrec["decA_correct"][sel["unex"]].mean())
        chg_C = (abs(dC_c1) - abs(dC_wr)) / max(abs(dC_c1), 1e-6) * 100
        chg_A = (abs(dA_c1) - abs(dA_wr)) / max(abs(dA_c1), 1e-6) * 100
        # Channel-resolved ΔL23_stim under each condition
        l23_C1_ex = float(data_C1["r_l23_stimch"][sel["ex"]].mean()); l23_C1_un = float(data_C1["r_l23_stimch"][sel["unex"]].mean())
        l23_Wr_ex = float(data_C_Wrec["r_l23_stimch"][sel["ex"]].mean()); l23_Wr_un = float(data_C_Wrec["r_l23_stimch"][sel["unex"]].mean())
        d_l23_C1 = l23_C1_ex - l23_C1_un; d_l23_Wr = l23_Wr_ex - l23_Wr_un
        l23_chg = (abs(d_l23_C1) - abs(d_l23_Wr)) / max(abs(d_l23_C1), 1e-6) * 100
        signflip_C = "FLIP" if (dC_c1 * dC_wr) < 0 else "same"
        signflip_A = "FLIP" if (dA_c1 * dA_wr) < 0 else "same"
        agree_signflip = "YES" if signflip_C == signflip_A else "NO"
        print(f"  {pname:14s}  {n_ex:5d}  {n_un:5d}  "
              f"{dC_c1:+9.4f}  {dC_wr:+9.4f}  {chg_C:+8.0f}%  "
              f"{dA_c1:+9.4f}  {dA_wr:+9.4f}  {chg_A:+8.0f}%  {agree_signflip:>7s}  "
              f"signflip C={signflip_C} A={signflip_A}  ΔL23_chg={l23_chg:+.0f}%")
        out[pname] = {"n_ex": n_ex, "n_unex": n_un,
                      "decC_C1": dC_c1, "decC_Wrec": dC_wr,
                      "decA_C1": dA_c1, "decA_Wrec": dA_wr,
                      "decC_change_pct": chg_C, "decA_change_pct": chg_A,
                      "delta_L23_stim_C1": d_l23_C1, "delta_L23_stim_Wrec": d_l23_Wr,
                      "delta_L23_change_pct": l23_chg,
                      "decC_signflip": signflip_C, "decA_signflip": signflip_A,
                      "agree": agree_signflip}
    return out


s3 = section3_table(data_P2, data_P3)

# Save
with open("/tmp/h17a_decA_obs_wrec.json", "w") as f:
    json.dump({"section1_native": s1_native, "section1_modified": s1_modified,
               "section3_Wrec": s3}, f, indent=2)
print("\n[save] /tmp/h17a_decA_obs_wrec.json")
