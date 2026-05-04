"""Phase 6 — Identify the L4→L23 amplifier of Mech 2 dampening on R1+R2.

Phase 5 found a small L4 bias (ΔL4_stimch ≈ +0.004) on HMS / HMS-T 3-march
trials, amplified ~25× to ΔL23_stimch ≈ +0.10. PV and Dec C falsified as the
amplifier. Phase 6 tests three remaining candidates.

Conditions (all with V2 ablated, fb_scale=0):
  C1   : V2 ablated only — Phase 5 baseline
  C_SOM: V2 + SOM→L23 inhibition zeroed (monkey-patch w_som to return zero)
  C_L4a: V2 + L4 adaptation zeroed (monkey-patch L4 forward to use zero adaptation)
  C_Wrec: V2 + L2/3 W_rec zeroed (monkey-patch L23 forward to skip recurrence)

For each condition, record per-presentation r_l4 and r_l23 at stim_ch + decoder
accuracy. Apply HMS / HMS-T / M3R filters offline. Compute ΔL4_stimch,
ΔL23_stimch, amplification ratio = ΔL23 / ΔL4.

Verdict per hypothesis:
  CONFIRMED: with the candidate ablated, ΔL23_stimch shrinks by ≥30% relative
             to C1 baseline (or amplification ratio drops to <10×).
  FALSIFIED: ΔL23_stimch and amplification ratio essentially unchanged (<10%).
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config
from src.model.network import LaminarV1V2Network
# L23Ring/L4Pool not directly imported; access via net.l23 / net.l4
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


def make_net(label: str):
    """Build a fresh network and apply the ablation specified by label.
    label ∈ {'C1', 'C_SOM', 'C_L4a', 'C_Wrec'}.
    All conditions have V2 feedback ablated (feedback_scale=0) — isolate Mech 2."""
    n = LaminarV1V2Network(model_cfg).to(device)
    n.load_state_dict(ckpt["model_state"], strict=False)
    n.eval(); n.oracle_mode = False; n.feedback_scale.fill_(0.0)
    for p in n.parameters():
        p.requires_grad_(False)

    if label == "C_SOM":
        # Monkey-patch L23.w_som.forward to return zeros — kills SOM inhibition into L23
        import types
        def zero_wsom_forward(self_w, r_som):
            return torch.zeros_like(r_som)
        n.l23.w_som.forward = types.MethodType(zero_wsom_forward, n.l23.w_som)
        # Also override __call__-bypass via forward pre-hook is unnecessary — Module.__call__
        # uses self.forward, so this monkey-patch covers w_som(r_som) calls.

    elif label == "C_L4a":
        # Monkey-patch L4Pool.forward to use ZERO adaptation each timestep
        original_l4_forward = n.l4.forward
        def zero_adapt_forward(self_l4, stimulus, r_l4_prev, r_pv_prev, adaptation_prev):
            zero_adapt = torch.zeros_like(adaptation_prev)
            return original_l4_forward(stimulus, r_l4_prev, r_pv_prev, zero_adapt)
        # Bind as method
        import types
        n.l4.forward = types.MethodType(zero_adapt_forward, n.l4)

    elif label == "C_Wrec":
        # Monkey-patch L23.forward to skip W_rec contribution
        original_l23_forward = n.l23.forward
        def skip_wrec_forward(self_l23, r_l4, r_l23_prev, template_modulation, r_som, r_pv):
            ff = F.linear(r_l4, self_l23.W_l4_to_l23)
            # Skip recurrence: rec = 0
            excitatory_drive = ff + template_modulation
            # Use whichever w_som is currently bound (could be patched zero in C_SOM
            # but for pure C_Wrec it's the original)
            if callable(self_l23.w_som):
                som_term = self_l23.w_som(r_som)
            else:
                som_term = self_l23.w_som(r_som)
            l23_drive = excitatory_drive - som_term - self_l23.w_pv_l23(r_pv)
            r_l23 = r_l23_prev + (self_l23.dt / self_l23.tau_l23) * (
                -r_l23_prev + F.softplus(l23_drive) * (l23_drive > 0).float()
            )
            return r_l23
        # Wait — rectified_softplus is what's used. Let me import.
        from src.model.populations import rectified_softplus
        def skip_wrec_forward2(self_l23, r_l4, r_l23_prev, template_modulation, r_som, r_pv):
            ff = F.linear(r_l4, self_l23.W_l4_to_l23)
            excitatory_drive = ff + template_modulation   # NO rec
            som_term = self_l23.w_som(r_som)
            l23_drive = excitatory_drive - som_term - self_l23.w_pv_l23(r_pv)
            r_l23 = r_l23_prev + (self_l23.dt / self_l23.tau_l23) * (
                -r_l23_prev + rectified_softplus(l23_drive)
            )
            return r_l23
        import types
        n.l23.forward = types.MethodType(skip_wrec_forward2, n.l23)

    return n


# Build all 4 condition nets
print("[setup] building 4 nets: C1 (V2-abl), C_SOM, C_L4a, C_Wrec", flush=True)
net_C1   = make_net("C1")
net_SOM  = make_net("C_SOM")
net_L4a  = make_net("C_L4a")
net_Wrec = make_net("C_Wrec")

# Smoke tests (run once to confirm ablations work)
print("\n[smoke tests]", flush=True)
sample_md = HMMSequenceGenerator(
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
).generate(2, SEQ_LENGTH, generator=torch.Generator().manual_seed(0))
ss, cs, ts, _, _, _ = build_stimulus_sequence(sample_md, model_cfg, train_cfg, stim_cfg)
ss = ss.to(device); cs = cs.to(device); ts = ts.to(device)
with torch.no_grad():
    r1, _, aux1 = net_C1.forward(net_C1.pack_inputs(ss, cs, ts))
    r2, _, aux2 = net_SOM.forward(net_SOM.pack_inputs(ss, cs, ts))
    r3, _, aux3 = net_L4a.forward(net_L4a.pack_inputs(ss, cs, ts))
    r4, _, aux4 = net_Wrec.forward(net_Wrec.pack_inputs(ss, cs, ts))
    diffs = {
        "C1 vs C_SOM (r_l23 max diff)": (r1 - r2).abs().max().item(),
        "C1 vs C_L4a (r_l4 max diff)": (aux1["r_l4_all"] - aux3["r_l4_all"]).abs().max().item(),
        "C1 vs C_Wrec (r_l23 max diff)": (r1 - r4).abs().max().item(),
    }
    for k, v in diffs.items():
        print(f"  {k}: {v:.6f} (expect > 0 if ablation took effect)")

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

# Buffers — per condition, per record
COND_NAMES = ["C1", "C_SOM", "C_L4a", "C_Wrec"]
KEYS_PER_COND = ("correct", "r_l4_stimch", "r_l23_stimch")
SHARED_KEYS = ("pred_err", "pi", "is_amb", "actual_ori", "ori_minus1", "ori_minus2", "true_ch")
buf = {k: [] for k in SHARED_KEYS}
for c in COND_NAMES:
    for k in KEYS_PER_COND:
        buf[f"{c}_{k}"] = []

print(f"\n[forward] {N_BATCHES} batches × bs={batch_size} = {N_BATCHES * batch_size} HMM trials  "
      f"(4 conditions)", flush=True)
nets_dict = {"C1": net_C1, "C_SOM": net_SOM, "C_L4a": net_L4a, "C_Wrec": net_Wrec}
for bi_b in range(N_BATCHES):
    md = gen.generate(batch_size, SEQ_LENGTH, generator=rng)
    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)

    cond_data = {}
    with torch.no_grad():
        for cname, cnet in nets_dict.items():
            r_l23, _, aux = cnet.forward(cnet.pack_inputs(stim_seq, cue_seq, ts_seq))
            r_l4 = aux["r_l4_all"]
            cond_data[cname] = {"r_l23": r_l23, "r_l4": r_l4}
            if cname == "C1":
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
        for cname, cd in cond_data.items():
            r_l23_w = cd["r_l23"][:, t0:t1+1, :].mean(dim=1)
            r_l4_w = cd["r_l4"][:, t0:t1+1, :].mean(dim=1)
            top1 = decC(r_l23_w).argmax(-1)
            corr = (top1 == true_ch).float()
            buf[f"{cname}_correct"].append(corr.cpu().numpy())
            buf[f"{cname}_r_l4_stimch"].append(r_l4_w[bi, true_ch.long()].cpu().numpy())
            buf[f"{cname}_r_l23_stimch"].append(r_l23_w[bi, true_ch.long()].cpu().numpy())

        ori_m1 = true_ori[:, pres_i-1]
        ori_m2 = true_ori[:, pres_i-2] if pres_i >= 2 else torch.full_like(actual_ori, -999.0)
        buf["pred_err"].append(pe.cpu().numpy())
        buf["pi"].append(pi_isi.cpu().numpy())
        buf["is_amb"].append(is_amb_all[:, pres_i].cpu().numpy())
        buf["actual_ori"].append(actual_ori.cpu().numpy())
        buf["ori_minus1"].append(ori_m1.cpu().numpy())
        buf["ori_minus2"].append(ori_m2.cpu().numpy())
        buf["true_ch"].append(true_ch.cpu().numpy())

    if (bi_b + 1) % 20 == 0:
        print(f"  batch {bi_b+1}/{N_BATCHES}", flush=True)

data = {k: np.concatenate(v) for k, v in buf.items()}
print(f"[N] per-presentation records: {data['C1_correct'].shape[0]}", flush=True)

# ============================================================================
# Filters
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
# Per-paradigm × per-condition: ΔL4_stimch, ΔL23_stimch, ratio, Δ_decC
# ============================================================================
def measure(cond, sel):
    n_ex = int(sel["ex"].sum()); n_un = int(sel["unex"].sum())
    if n_ex < 20 or n_un < 20:
        return None
    l4_ex = float(data[f"{cond}_r_l4_stimch"][sel["ex"]].mean())
    l4_un = float(data[f"{cond}_r_l4_stimch"][sel["unex"]].mean())
    l23_ex = float(data[f"{cond}_r_l23_stimch"][sel["ex"]].mean())
    l23_un = float(data[f"{cond}_r_l23_stimch"][sel["unex"]].mean())
    ex_acc = float(data[f"{cond}_correct"][sel["ex"]].mean())
    un_acc = float(data[f"{cond}_correct"][sel["unex"]].mean())
    d_l4 = l4_ex - l4_un
    d_l23 = l23_ex - l23_un
    d_decC = ex_acc - un_acc
    ratio = (d_l23 / d_l4) if abs(d_l4) > 1e-6 else float("nan")
    return {"n_ex": n_ex, "n_unex": n_un,
            "L4_ex": l4_ex, "L4_un": l4_un, "delta_L4_stim": d_l4,
            "L23_ex": l23_ex, "L23_un": l23_un, "delta_L23_stim": d_l23,
            "ex_acc": ex_acc, "unex_acc": un_acc, "delta_decC": d_decC,
            "amplification_ratio": ratio}


print("\n========== Per-paradigm × per-condition table ==========")
print(f"  {'paradigm':14s}  {'cond':>7s}  {'n_ex':>5s}  {'n_un':>5s}  "
      f"{'ΔL4':>9s}  {'ΔL23':>9s}  {'ratio':>8s}  {'Δ_decC':>9s}")
results = {}
for pname, sel in paradigms.items():
    results[pname] = {}
    for cond in COND_NAMES:
        r = measure(cond, sel)
        if r is None:
            print(f"  {pname:14s}  {cond:>7s}  insufficient")
            continue
        ratio_str = f"{r['amplification_ratio']:.1f}" if np.isfinite(r['amplification_ratio']) else "—"
        print(f"  {pname:14s}  {cond:>7s}  {r['n_ex']:5d}  {r['n_unex']:5d}  "
              f"{r['delta_L4_stim']:+9.4f}  {r['delta_L23_stim']:+9.4f}  {ratio_str:>8s}  "
              f"{r['delta_decC']:+9.4f}")
        results[pname][cond] = r

# ============================================================================
# Verdicts
# ============================================================================
print("\n========== Hypothesis verdicts (per paradigm) ==========")
verdicts = {}
for pname in ["HMS native", "HMS-T native", "M3R native"]:
    if pname not in results or "C1" not in results[pname]: continue
    base_l23 = results[pname]["C1"]["delta_L23_stim"]
    base_dec = results[pname]["C1"]["delta_decC"]
    base_l4 = results[pname]["C1"]["delta_L4_stim"]
    print(f"\n  [{pname}] baseline (C1): ΔL4={base_l4:+.4f}  ΔL23={base_l23:+.4f}  "
          f"Δ_decC={base_dec:+.4f}  ratio={base_l23/base_l4 if abs(base_l4)>1e-6 else float('nan'):.1f}")
    verdicts[pname] = {}
    for cond_name, h_label in [("C_SOM", "H_SOM"), ("C_L4a", "H_L4adapt"), ("C_Wrec", "H_W_rec")]:
        if cond_name not in results[pname]: continue
        r = results[pname][cond_name]
        l23_change = (abs(base_l23) - abs(r["delta_L23_stim"])) / max(abs(base_l23), 1e-6) * 100
        dec_change = (abs(base_dec) - abs(r["delta_decC"])) / max(abs(base_dec), 1e-6) * 100
        if l23_change >= 30:
            v = f"CONFIRMED — {h_label} reduces |ΔL23_stimch| by {l23_change:.0f}%"
        elif l23_change <= -30:
            v = f"INVERTED — {h_label} amplifies |ΔL23_stimch| by {-l23_change:.0f}%"
        elif abs(l23_change) < 10:
            v = f"FALSIFIED — |ΔL23| change {l23_change:+.0f}% (<10%)"
        else:
            v = f"PARTIAL — |ΔL23| change {l23_change:+.0f}%"
        print(f"    {h_label:11s}: ΔL23={r['delta_L23_stim']:+.4f} (vs {base_l23:+.4f})  "
              f"|ΔL23|→{l23_change:+.0f}%  Δ_decC={r['delta_decC']:+.4f} (|Δ_decC|→{dec_change:+.0f}%)  → {v}")
        verdicts[pname][h_label] = {"|ΔL23|_change_pct": l23_change,
                                     "|Δ_decC|_change_pct": dec_change,
                                     "verdict_string": v,
                                     "ablated_delta_L23": r["delta_L23_stim"],
                                     "ablated_delta_decC": r["delta_decC"]}

with open("/tmp/h16_amplifier_locus.json", "w") as f:
    json.dump({"per_paradigm_per_condition": results,
               "verdicts": verdicts,
               "config": {"n_records": int(data["C1_correct"].shape[0]),
                          "pi_Q75": pi_q75}}, f, indent=2)
print("\n[save] /tmp/h16_amplifier_locus.json")
