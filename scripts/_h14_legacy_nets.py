"""Task #14 Phase 4 Exp 4.6 — Channel-resolved diagnostic on legacy nets a1/b1/c1/e1.

For each of the 4 legacy network checkpoints, run HMM C1 paired-fork forward
pass and compute the same channel-resolved measures (r_at_predch, r_at_stimch
per branch, Δ_decC) per V2 pred_err stratum. Confirm whether the
channel-resolved suppression mechanism generalises across networks.

Each net uses its own sweep config (sweep_a1.yaml etc.). Decoder C is shared
across networks (Linear(36,36) frozen).

Networks:
  a1: /tmp/remote_ckpts/a1/checkpoint.pt   config/sweep/sweep_a1.yaml   (dampening)
  b1: /tmp/remote_ckpts/b1/checkpoint.pt   config/sweep/sweep_b1.yaml   (dampening)
  c1: /tmp/remote_ckpts/c1/checkpoint.pt   config/sweep/sweep_c1.yaml   (sharpening)
  e1: /tmp/remote_ckpts/e1/checkpoint.pt   config/sweep/sweep_e1.yaml   (sharpening)
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

REPO = "/mnt/c/Users/User/codingproj/freshstart"
DEC_C = f"{REPO}/checkpoints/decoder_c.pt"
SEED = 42
N_TRIALS = 4000
SEQ_LENGTH = 25

NETS = {
    "a1": {"ckpt": "/tmp/remote_ckpts/a1/checkpoint.pt",
           "config": f"{REPO}/config/sweep/sweep_a1.yaml"},
    "b1": {"ckpt": "/tmp/remote_ckpts/b1/checkpoint.pt",
           "config": f"{REPO}/config/sweep/sweep_b1.yaml"},
    "c1": {"ckpt": "/tmp/remote_ckpts/c1/checkpoint.pt",
           "config": f"{REPO}/config/sweep/sweep_c1.yaml"},
    "e1": {"ckpt": "/tmp/remote_ckpts/e1/checkpoint.pt",
           "config": f"{REPO}/config/sweep/sweep_e1.yaml"},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dC_state = torch.load(DEC_C, map_location=device, weights_only=False)
if isinstance(dC_state, dict) and "state_dict" in dC_state:
    dC_state = dC_state["state_dict"]
decC = nn.Linear(36, 36, bias=True).to(device)
decC.load_state_dict(dC_state); decC.eval()


def run_net(net_name: str, ckpt_path: str, cfg_path: str):
    print(f"\n========== {net_name}  (ckpt={ckpt_path})  ==========", flush=True)
    model_cfg, train_cfg, stim_cfg = load_config(cfg_path)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    steps_on = int(train_cfg.steps_on); steps_isi = int(train_cfg.steps_isi)
    steps_per = steps_on + steps_isi
    probe_idx = SEQ_LENGTH - 1
    probe_onset = probe_idx * steps_per
    isi_pre_probe = probe_onset - 1

    net = LaminarV1V2Network(model_cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    res = net.load_state_dict(ck["model_state"], strict=False)
    print(f"  load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    net.eval(); net.oracle_mode = False; net.feedback_scale.fill_(1.0)
    for p in net.parameters():
        p.requires_grad_(False)

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

    buf = {k: [] for k in (
        "pe_A", "pe_B", "pred_ch", "true_ch_ex", "true_ch_unex",
        "r_predch_A", "r_predch_B", "r_stimch_A", "r_stimch_B",
        "r_norm_A", "r_norm_B", "correct_A", "correct_B", "keep")}

    n_done = 0; batch = 1000
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

        with torch.no_grad():
            packed_A = net.pack_inputs(stim_ex, cue_seq, ts_seq)
            r_l23_A, _, aux_A = net.forward(packed_A)
            packed_B = net.pack_inputs(stim_unex, cue_seq, ts_seq)
            r_l23_B, _, _ = net.forward(packed_B)
        q_pred_isi = aux_A["q_pred_all"][:, isi_pre_probe, :]
        pred_peak = q_pred_isi.argmax(dim=-1)
        pred_ori = pred_peak.float() * step_deg
        actual_ori_A = true_ori_ex
        actual_ori_B = (true_ori_ex + period / 2.0) % period
        pe_A = circular_distance(pred_ori, actual_ori_A, period).abs()
        pe_B = circular_distance(pred_ori, actual_ori_B, period).abs()
        r_probe_A = r_l23_A[:, probe_onset+9:probe_onset+11, :].mean(dim=1)
        r_probe_B = r_l23_B[:, probe_onset+9:probe_onset+11, :].mean(dim=1)

        bi = torch.arange(b, device=device)
        r_predch_A = r_probe_A[bi, pred_peak.long()]
        r_predch_B = r_probe_B[bi, pred_peak.long()]
        r_stimch_A = r_probe_A[bi, true_ch_ex.long()]
        r_stimch_B = r_probe_B[bi, true_ch_unex.long()]

        pred_dec_A = decC(r_probe_A).argmax(-1)
        pred_dec_B = decC(r_probe_B).argmax(-1)
        correct_A = (pred_dec_A == true_ch_ex).float()
        correct_B = (pred_dec_B == true_ch_unex).float()

        buf["pe_A"].append(pe_A.cpu().numpy())
        buf["pe_B"].append(pe_B.cpu().numpy())
        buf["pred_ch"].append(pred_peak.cpu().numpy())
        buf["true_ch_ex"].append(true_ch_ex.cpu().numpy())
        buf["true_ch_unex"].append(true_ch_unex.cpu().numpy())
        buf["r_predch_A"].append(r_predch_A.cpu().numpy())
        buf["r_predch_B"].append(r_predch_B.cpu().numpy())
        buf["r_stimch_A"].append(r_stimch_A.cpu().numpy())
        buf["r_stimch_B"].append(r_stimch_B.cpu().numpy())
        buf["r_norm_A"].append(r_probe_A.norm(dim=1).cpu().numpy())
        buf["r_norm_B"].append(r_probe_B.norm(dim=1).cpu().numpy())
        buf["correct_A"].append(correct_A.cpu().numpy())
        buf["correct_B"].append(correct_B.cpu().numpy())
        buf["keep"].append((~is_amb).cpu().numpy())
        n_done += b
    for k in buf:
        buf[k] = np.concatenate(buf[k])

    pe = buf["pe_A"].copy(); pe[~buf["keep"]] = np.nan
    print(f"  pred_err mean={np.nanmean(pe):.2f}° median={np.nanmedian(pe):.2f}°  N_keep={int(buf['keep'].sum())}")
    bins = {
        "pe<=5":     buf["keep"] & (buf["pe_A"] <= 5.0),
        "5<pe<=15":  buf["keep"] & (buf["pe_A"] > 5.0)  & (buf["pe_A"] <= 15.0),
        "15<pe<=30": buf["keep"] & (buf["pe_A"] > 15.0) & (buf["pe_A"] <= 30.0),
        "30<pe<=60": buf["keep"] & (buf["pe_A"] > 30.0) & (buf["pe_A"] <= 60.0),
        "pe>60":     buf["keep"] & (buf["pe_A"] > 60.0),
    }
    print(f"  {'stratum':12s}  {'n':>5s}  {'r_predch_A':>11s}  {'r_predch_B':>11s}  "
          f"{'r_stimch_A':>11s}  {'r_stimch_B':>11s}  {'Δr_stimch':>10s}  "
          f"{'ex_acc':>7s}  {'unex_acc':>9s}  {'Δ_decC':>9s}")
    out = {}
    for name, m in bins.items():
        n = int(m.sum())
        if n < 20:
            print(f"  {name:12s}  n={n:4d}  (skipped, too few)")
            continue
        rpA = float(buf["r_predch_A"][m].mean()); rpB = float(buf["r_predch_B"][m].mean())
        rsA = float(buf["r_stimch_A"][m].mean()); rsB = float(buf["r_stimch_B"][m].mean())
        ex = float(buf["correct_A"][m].mean()); un = float(buf["correct_B"][m].mean())
        print(f"  {name:12s}  {n:5d}  {rpA:11.4f}  {rpB:11.4f}  "
              f"{rsA:11.4f}  {rsB:11.4f}  {rsA-rsB:+10.4f}  "
              f"{ex:7.4f}  {un:9.4f}  {ex-un:+9.4f}")
        out[name] = {"n": n, "r_predch_A": rpA, "r_predch_B": rpB,
                     "r_stimch_A": rsA, "r_stimch_B": rsB, "delta_r_stimch": rsA - rsB,
                     "ex_acc": ex, "unex_acc": un, "delta_decC": ex - un}
    overall = {
        "n_keep": int(buf["keep"].sum()),
        "ex_acc": float(buf["correct_A"][buf["keep"]].mean()),
        "unex_acc": float(buf["correct_B"][buf["keep"]].mean()),
        "delta_decC": float(buf["correct_A"][buf["keep"]].mean() - buf["correct_B"][buf["keep"]].mean()),
        "pe_mean_deg": float(np.nanmean(pe)),
    }
    print(f"  [overall] ex={overall['ex_acc']:.4f}  unex={overall['unex_acc']:.4f}  "
          f"Δ={overall['delta_decC']:+.4f}", flush=True)
    return overall, out


all_results = {}
for name, info in NETS.items():
    if not os.path.exists(info["ckpt"]):
        print(f"  [skip] {name}: ckpt missing at {info['ckpt']}", flush=True)
        continue
    if not os.path.exists(info["config"]):
        print(f"  [skip] {name}: config missing at {info['config']}", flush=True)
        continue
    overall, strata = run_net(name, info["ckpt"], info["config"])
    all_results[name] = {"overall": overall, "strata": strata}

with open("/tmp/h14_legacy_nets.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\n[save] /tmp/h14_legacy_nets.json")
