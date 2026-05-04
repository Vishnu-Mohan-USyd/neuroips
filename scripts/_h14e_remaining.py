"""Task Phase 4e — V2-ablation on remaining R1+R2 dampening paradigms.

Tests V2-feedback causality (intact fb_scale=1.0 vs ablated fb_scale=0.0) on:
  - VCD-test3 native
  - M3R modified (focused task_state pinned + march-continuation cue)
  - HMS modified
  - HMS-T modified
  - VCD-test3 modified

Modified variants: apply `apply_input_overrides(metadata, period, n_ori)`
from scripts/cross_decoder_eval.py before forward pass — pins task_state to
focused [1,0], replaces cues with deterministic march-continuation Gaussian
bumps σ=10°.

Two forward-pass conditions × two input variants (native + modified) on the
same HMM batch ⇒ 4 forward passes × N_batches × bs records.

Verdict per paradigm:
  Mech 1   : ablation FLIPS Δ_decC sign OR reduces |Δ_decC| by ≥50%
  Mech 2   : ablation AMPLIFIES |Δ_decC| OR V2 contribution is sharpening (sign opposite to dampening)
  INCONCLUSIVE: otherwise
"""
from __future__ import annotations
import os, sys, json, copy
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.training.trainer import build_stimulus_sequence
from src.stimulus.sequences import HMMSequenceGenerator
from src.utils import circular_distance
from scripts.cross_decoder_eval import apply_input_overrides

CKPT = "/mnt/c/Users/User/codingproj/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt"
DEC_C = "/mnt/c/Users/User/codingproj/freshstart/checkpoints/decoder_c.pt"
CONFIG = "/mnt/c/Users/User/codingproj/freshstart/config/sweep/sweep_rescue_1_2.yaml"
SEED = 42
N_BATCHES = 80   # × bs=32 = 2560 trials × 24 presentations = 61,440 records per condition
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
def make_net(fb):
    n = LaminarV1V2Network(model_cfg).to(device)
    n.load_state_dict(ckpt["model_state"], strict=False)
    n.eval(); n.oracle_mode = False; n.feedback_scale.fill_(float(fb))
    for p in n.parameters():
        p.requires_grad_(False)
    return n
net_intact = make_net(1.0); net_ablated = make_net(0.0)

dC_state = torch.load(DEC_C, map_location=device, weights_only=False)
if isinstance(dC_state, dict) and "state_dict" in dC_state:
    dC_state = dC_state["state_dict"]
decC = nn.Linear(n_ori, n_ori, bias=True).to(device)
decC.load_state_dict(dC_state); decC.eval()

base_gen = HMMSequenceGenerator(
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

KEYS = ("pred_err", "pi", "is_amb", "actual_ori", "ori_minus1", "ori_minus2", "true_ch",
        "intact_correct", "ablated_correct",
        "intact_r_stimch", "ablated_r_stimch",
        "intact_r_predch", "ablated_r_predch", "pred_ch")


def run_input_variant(variant: str):
    """variant ∈ {'native', 'modified'}. Returns dict of concat per-presentation arrays."""
    print(f"\n[forward] variant={variant}  N_batches={N_BATCHES}  bs={batch_size}  "
          f"intact + ablated", flush=True)
    rng = torch.Generator().manual_seed(SEED)  # SAME seed for native and modified for matched stim
    buf = {k: [] for k in KEYS}
    for bi_b in range(N_BATCHES):
        md = base_gen.generate(batch_size, SEQ_LENGTH, generator=rng)
        if variant == "modified":
            apply_input_overrides(md, period, n_ori)
        # else: native HMM cue + native task_state, untouched
        stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
        stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

        with torch.no_grad():
            packed = net_intact.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_intact, _, aux = net_intact.forward(packed)
            q_pred_all = aux["q_pred_all"]
            pi_all = aux["pi_pred_eff_all"]
            packed_a = net_ablated.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_ablated, _, _ = net_ablated.forward(packed_a)

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
            r_intact_w = r_intact[:, t0:t1+1, :].mean(dim=1)
            r_ablated_w = r_ablated[:, t0:t1+1, :].mean(dim=1)
            intact_top1 = decC(r_intact_w).argmax(-1)
            ablated_top1 = decC(r_ablated_w).argmax(-1)
            intact_corr = (intact_top1 == true_ch).float()
            ablated_corr = (ablated_top1 == true_ch).float()
            intact_r_stimch = r_intact_w[bi, true_ch.long()]
            ablated_r_stimch = r_ablated_w[bi, true_ch.long()]
            intact_r_predch = r_intact_w[bi, pred_peak.long()]
            ablated_r_predch = r_ablated_w[bi, pred_peak.long()]
            ori_m1 = true_ori[:, pres_i-1]
            ori_m2 = true_ori[:, pres_i-2] if pres_i >= 2 else torch.full_like(actual_ori, -999.0)

            buf["pred_err"].append(pe.cpu().numpy())
            buf["pi"].append(pi_isi.cpu().numpy())
            buf["is_amb"].append(is_amb_all[:, pres_i].cpu().numpy())
            buf["actual_ori"].append(actual_ori.cpu().numpy())
            buf["ori_minus1"].append(ori_m1.cpu().numpy())
            buf["ori_minus2"].append(ori_m2.cpu().numpy())
            buf["true_ch"].append(true_ch.cpu().numpy())
            buf["intact_correct"].append(intact_corr.cpu().numpy())
            buf["ablated_correct"].append(ablated_corr.cpu().numpy())
            buf["intact_r_stimch"].append(intact_r_stimch.cpu().numpy())
            buf["ablated_r_stimch"].append(ablated_r_stimch.cpu().numpy())
            buf["intact_r_predch"].append(intact_r_predch.cpu().numpy())
            buf["ablated_r_predch"].append(ablated_r_predch.cpu().numpy())
            buf["pred_ch"].append(pred_peak.cpu().numpy())
        if (bi_b + 1) % 20 == 0:
            print(f"  {variant} batch {bi_b+1}/{N_BATCHES}", flush=True)
    return {k: np.concatenate(v) for k, v in buf.items()}


data_native = run_input_variant("native")
data_modified = run_input_variant("modified")
print(f"[N] per-presentation records (native): {data_native['intact_correct'].shape[0]}", flush=True)
print(f"[N] per-presentation records (modified): {data_modified['intact_correct'].shape[0]}", flush=True)

# ============================================================================
# Build per-paradigm filters on each variant
# ============================================================================
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
        "M3R":         {"ex":   keep & (d["pred_err"] <= 5.0)  & (d["pi"] >= pi_q75),
                        "unex": keep & (d["pred_err"] > 20.0) & (d["pi"] >= pi_q75)},
        "HMS":         {"ex":   keep & is_3march & (d["pi"] >= pi_q75),
                        "unex": keep & is_march_jump & (d["pi"] >= pi_q75)},
        "HMS-T":       {"ex":   keep & is_3march & (d["pred_err"] <= 5.0) & (d["pi"] >= pi_q75),
                        "unex": keep & is_march_jump & (d["pred_err"] > 60.0) & (d["pi"] >= pi_q75)},
        "VCD-test3":   {"ex":   keep & (d["pred_err"] <= 10.0),
                        "unex": keep & (d["pred_err"] > 20.0)},
    }, pi_q75


print("\n========== Phase 4e: V2-ablation on remaining R1+R2 paradigms ==========")
print(f"  {'paradigm':28s}  {'n_ex':>5s}  {'n_un':>5s}  "
      f"{'Δ_decC_int':>11s}  {'Δ_decC_abl':>11s}  {'V2 contrib':>11s}  {'Verdict':>20s}")

results = {}

# Run each (paradigm, variant)
def run_one(name, variant, sel, d):
    n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
    if n_ex < 20 or n_un < 20:
        print(f"  {name:28s}  n_ex={n_ex} n_un={n_un}  insufficient")
        return {"insufficient": True, "n_ex": n_ex, "n_unex": n_un}
    ex_int = float(d["intact_correct"][sel["ex"]].mean()); un_int = float(d["intact_correct"][sel["unex"]].mean())
    ex_abl = float(d["ablated_correct"][sel["ex"]].mean()); un_abl = float(d["ablated_correct"][sel["unex"]].mean())
    d_int = ex_int - un_int; d_abl = ex_abl - un_abl
    v2_contrib = d_int - d_abl  # positive = V2 makes Δ_decC more positive (sharpening direction)

    abs_int, abs_abl = abs(d_int), abs(d_abl)
    if abs_int > 1e-6:
        # MECH 1: V2 ablation reduces |Δ_decC| by ≥50% OR flips sign of Δ_decC AND |Δ| reduces ≥30%
        sign_flip = np.sign(d_int) != np.sign(d_abl) and abs_int > 1e-3
        red50 = (abs_int - abs_abl) / abs_int >= 0.50
        # MECH 2: ablation AMPLIFIES |Δ_decC| (i.e., abs_abl > abs_int) by ≥30% AND no sign flip
        ampl30 = (abs_abl - abs_int) / abs_int >= 0.30
        if (sign_flip and (abs_int - abs_abl) / abs_int >= 0.30) or red50:
            verdict = "Mech 1 (V2 causal)"
        elif ampl30:
            verdict = "Mech 2 (NON-V2)"
        else:
            verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE (intact ~0)"
    print(f"  {name:28s}  {n_ex:5d}  {n_un:5d}  "
          f"{d_int:+11.4f}  {d_abl:+11.4f}  {v2_contrib:+11.4f}  {verdict:>20s}")
    return {
        "n_ex": n_ex, "n_unex": n_un,
        "delta_decC_intact": d_int, "delta_decC_ablated": d_abl,
        "v2_contribution": v2_contrib, "verdict": verdict,
        "ex_acc_intact": ex_int, "unex_acc_intact": un_int,
        "ex_acc_ablated": ex_abl, "unex_acc_ablated": un_abl,
    }


# Native paradigms
filters_native, piQ75_n = build_filters(data_native)
print(f"\n[native, pi Q75 = {piQ75_n:.4f}]")
for paradigm in ["M3R", "HMS", "HMS-T", "VCD-test3"]:
    name = f"{paradigm} native"
    results[name] = run_one(name, "native", filters_native[paradigm], data_native)

# Modified paradigms
filters_modified, piQ75_m = build_filters(data_modified)
print(f"\n[modified, pi Q75 = {piQ75_m:.4f}]")
for paradigm in ["M3R", "HMS", "HMS-T", "VCD-test3"]:
    name = f"{paradigm} modified"
    results[name] = run_one(name, "modified", filters_modified[paradigm], data_modified)


# ============================================================================
# Final R1+R2 paradigm classification table
# ============================================================================
print("\n========== R1+R2 paradigm classification (V2-ablation verdicts) ==========")
print(f"{'Paradigm':30s}  {'Δ_decC_intact':>14s}  {'Δ_decC_ablated':>15s}  {'Verdict':>20s}")
# Pre-existing results from Phase 4d Exp 4.9 — re-emit for completeness
prior_phase4d = {}
prior_path = "/tmp/h14d_hms_diag.json"
if os.path.exists(prior_path):
    with open(prior_path) as f:
        pd = json.load(f).get("exp_4_9", {})
    for k in ("M3R native", "HMS native", "HMS-T native"):
        if k in pd and not pd[k].get("insufficient"):
            prior_phase4d[k] = pd[k]
            print(f"{k:30s}  {pd[k]['delta_decC_intact']:+14.4f}  "
                  f"{pd[k]['delta_decC_ablated']:+15.4f}  {pd[k].get('verdict','—'):>20s}")

for k, v in results.items():
    if v.get("insufficient"):
        print(f"{k:30s}  insufficient (n_ex={v['n_ex']}, n_un={v['n_unex']})")
        continue
    print(f"{k:30s}  {v['delta_decC_intact']:+14.4f}  {v['delta_decC_ablated']:+15.4f}  {v['verdict']:>20s}")

# Save
with open("/tmp/h14e_remaining.json", "w") as f:
    json.dump({"phase4e_results": results,
               "phase4d_replay_for_completeness": prior_phase4d,
               "config": {"n_records_per_variant": int(data_native["intact_correct"].shape[0]),
                          "pi_Q75_native": piQ75_n,
                          "pi_Q75_modified": piQ75_m}}, f, indent=2)
print("\n[save] /tmp/h14e_remaining.json")
