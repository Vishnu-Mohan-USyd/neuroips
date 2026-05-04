"""Task #16 Phase 4d Exp 4.7 + 4.9 — HMS / HMS-T diagnostics on R1+R2.

Two forward passes on the same HMM stream (intact fb=1.0 + ablated fb=0.0).
Per-presentation records collected at every pres_i ∈ [1, 24].

  Exp 4.9: apply HMS, HMS-T filters to BOTH conditions; compare Δ_decC intact
           vs ablated. Verdict per paradigm: A (V2 causal) / B (not) / MIXED.

  Exp 4.7: stratify HMS / HMS-T trials by stim-decodability (= correct on
           ablated condition for the same stim ⇒ "BU-only decodable"). Within
           the same-decodability stratum, recompute Δ_decC and Δr_stimch on
           intact data. Test whether the channel-resolved suppression
           mechanism reappears once stim-difficulty is controlled.
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
N_BATCHES = 80   # × bs=32 = 2560 HMM trials × 24 presentations = 61440 records
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

KEYS = ("pred_err", "pi", "is_amb", "actual_ori", "ori_minus1", "ori_minus2",
        "true_ch", "intact_correct", "ablated_correct",
        "intact_r_stimch", "ablated_r_stimch",
        "intact_r_predch", "ablated_r_predch",
        "intact_r_norm", "ablated_r_norm",
        "intact_decoder_top1", "ablated_decoder_top1",
        "pred_ch")
buf = {k: [] for k in KEYS}

print(f"[forward] {N_BATCHES} batches × bs={batch_size} = {N_BATCHES * batch_size} HMM trials  "
      f"(intact + ablated)", flush=True)
for bi_b in range(N_BATCHES):
    md = gen.generate(batch_size, SEQ_LENGTH, generator=rng)
    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

    with torch.no_grad():
        packed = net_intact.pack_inputs(stim_seq, cue_seq, ts_seq)
        r_intact, _, aux = net_intact.forward(packed)
        q_pred_all = aux["q_pred_all"]
        pi_all = aux["pi_pred_eff_all"]

        packed_a = net_ablated.pack_inputs(stim_seq, cue_seq, ts_seq)
        r_ablated, _, _ = net_ablated.forward(packed_a)

    true_ori = md.orientations.to(device)   # [B, S]
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

        t0 = pres_i * steps_per + 9
        t1 = pres_i * steps_per + 11
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
        buf["intact_r_norm"].append(r_intact_w.norm(dim=1).cpu().numpy())
        buf["ablated_r_norm"].append(r_ablated_w.norm(dim=1).cpu().numpy())
        buf["intact_decoder_top1"].append(intact_top1.cpu().numpy())
        buf["ablated_decoder_top1"].append(ablated_top1.cpu().numpy())
        buf["pred_ch"].append(pred_peak.cpu().numpy())

    if (bi_b + 1) % 20 == 0:
        print(f"  batch {bi_b+1}/{N_BATCHES} done", flush=True)

data = {k: np.concatenate(v) for k, v in buf.items()}
print(f"[N] total per-presentation records: {data['intact_correct'].shape[0]}", flush=True)

keep = ~data["is_amb"].astype(bool)
pi_q75 = float(np.percentile(data["pi"][keep], 75))
print(f"[pi Q75 global, kept] = {pi_q75:.4f}", flush=True)

# HMS trajectory features
def signed_circ(a, b, p):
    d = (a - b) % p
    return np.where(d > p / 2, d - p, d)
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
    # M3R for comparison (pred_err split, mechanism CONFIRMED in Phase 4)
    "M3R native":   {"ex": keep & (data["pred_err"] <= 5.0) & (data["pi"] >= pi_q75),
                     "unex": keep & (data["pred_err"] > 20.0) & (data["pi"] >= pi_q75)},
}

print(f"\n========== Exp 4.9 — V2 ablation effect on HMS / HMS-T / M3R Δ_decC ==========")
print(f"  {'paradigm':14s}  {'n_ex':>5s}  {'n_unex':>6s}  "
      f"{'Δ_decC_int':>11s}  {'Δ_decC_abl':>11s}  {'|Δ| change':>12s}  "
      f"{'r_stimch_ex_int':>15s}  {'r_stimch_unex_int':>17s}  {'Δr_stimch_int':>13s}")
exp49 = {}
for name, sel in paradigms.items():
    n_ex = int(sel["ex"].sum()); n_unex = int(sel["unex"].sum())
    if n_ex < 20 or n_unex < 20:
        print(f"  {name:14s}  n_ex={n_ex} n_unex={n_unex}  insufficient")
        exp49[name] = {"insufficient": True}
        continue
    ex_int = float(data["intact_correct"][sel["ex"]].mean())
    un_int = float(data["intact_correct"][sel["unex"]].mean())
    ex_abl = float(data["ablated_correct"][sel["ex"]].mean())
    un_abl = float(data["ablated_correct"][sel["unex"]].mean())
    d_int = ex_int - un_int; d_abl = ex_abl - un_abl
    chg = (abs(d_int) - abs(d_abl)) / abs(d_int) if abs(d_int) > 1e-6 else 0.0
    rs_ex_int = float(data["intact_r_stimch"][sel["ex"]].mean())
    rs_un_int = float(data["intact_r_stimch"][sel["unex"]].mean())
    rs_ex_abl = float(data["ablated_r_stimch"][sel["ex"]].mean())
    rs_un_abl = float(data["ablated_r_stimch"][sel["unex"]].mean())
    print(f"  {name:14s}  {n_ex:5d}  {n_unex:6d}  "
          f"{d_int:+11.4f}  {d_abl:+11.4f}  {chg*100:+11.1f}%  "
          f"{rs_ex_int:15.4f}  {rs_un_int:17.4f}  {rs_ex_int-rs_un_int:+13.4f}")
    if chg >= 0.50: verdict = "OUTCOME A — V2 causal"
    elif abs(chg) < 0.30: verdict = "OUTCOME B — V2 not causal"
    else: verdict = "MIXED / partial"
    exp49[name] = {
        "n_ex": n_ex, "n_unex": n_unex,
        "delta_decC_intact": d_int, "delta_decC_ablated": d_abl,
        "abs_change_pct": chg * 100,
        "intact_r_stimch_ex": rs_ex_int, "intact_r_stimch_unex": rs_un_int,
        "intact_delta_r_stimch": rs_ex_int - rs_un_int,
        "ablated_r_stimch_ex": rs_ex_abl, "ablated_r_stimch_unex": rs_un_abl,
        "ablated_delta_r_stimch": rs_ex_abl - rs_un_abl,
        "verdict": verdict,
    }
    print(f"    → {verdict}")

# ============================================================================
# Exp 4.7 — stratify HMS / HMS-T by stim-decodability (= ablated_correct on
# the same stim). Within {decodable, undecodable} buckets, recompute Δ_decC
# and Δr_stimch on the INTACT condition.
# ============================================================================
print(f"\n========== Exp 4.7 — Stim-decodability stratification (HMS / HMS-T) ==========")
print("  Stim-decodability proxy: ablated_correct (=BU-only decoder correctness on the same stim).\n"
      "  Subject-control: within each decodability bucket, the ex vs unex stim sets are\n"
      "  matched on whether their stim was BU-decodable. If the channel-resolved\n"
      "  suppression mechanism is real for HMS/HMS-T, Δr_stimch should turn\n"
      "  NEGATIVE within the 'decodable' bucket once decodability confound is removed.")

for name, sel in [("HMS native", paradigms["HMS native"]),
                  ("HMS-T native", paradigms["HMS-T native"]),
                  ("M3R native (control)", paradigms["M3R native"])]:
    print(f"\n  [{name}]")
    for label, dec_mask in [("decodable",   data["ablated_correct"].astype(bool)),
                             ("undecodable", ~data["ablated_correct"].astype(bool))]:
        ex_b = sel["ex"] & dec_mask
        un_b = sel["unex"] & dec_mask
        n_ex = int(ex_b.sum()); n_un = int(un_b.sum())
        if n_ex < 20 or n_un < 20:
            print(f"    {label:12s}: n_ex={n_ex} n_un={n_un}  insufficient")
            continue
        ex_acc = float(data["intact_correct"][ex_b].mean())
        un_acc = float(data["intact_correct"][un_b].mean())
        rs_ex = float(data["intact_r_stimch"][ex_b].mean())
        rs_un = float(data["intact_r_stimch"][un_b].mean())
        rp_ex = float(data["intact_r_predch"][ex_b].mean())
        rp_un = float(data["intact_r_predch"][un_b].mean())
        pe_ex = float(data["pred_err"][ex_b].mean())
        pe_un = float(data["pred_err"][un_b].mean())
        print(f"    {label:12s}: n_ex={n_ex:4d}  n_un={n_un:4d}  "
              f"ex_acc={ex_acc:.4f}  un_acc={un_acc:.4f}  Δ={ex_acc-un_acc:+.4f}  "
              f"r_stimch_ex={rs_ex:.4f}  r_stimch_un={rs_un:.4f}  Δr={rs_ex-rs_un:+.4f}  "
              f"r_predch_ex={rp_ex:.4f}  r_predch_un={rp_un:.4f}  pe_ex={pe_ex:.1f} pe_un={pe_un:.1f}")

# Save
with open("/tmp/h14d_hms_diag.json", "w") as f:
    json.dump({"exp_4_9": exp49,
               "config": {"n_trials_per_condition": N_BATCHES * batch_size,
                          "n_records_total": int(data["intact_correct"].shape[0]),
                          "pi_Q75": pi_q75}},
              f, indent=2)
print("\n[save] /tmp/h14d_hms_diag.json")
np.savez("/tmp/h14d_hms_pertrial.npz", **{k: v for k, v in data.items()})
print("[save] /tmp/h14d_hms_pertrial.npz")
