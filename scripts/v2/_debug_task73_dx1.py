"""Task #73 Dx1 — expectation-signal audit.

On Kok Task#70 checkpoint, for each eval trial:
  - collect m, b_task=W_mh_task@m, b_gen=W_mh_gen@m, r_l23 at t=delay_end-1
    (after cue, before probe)
  - label = cue_id (equivalent to expected orientation)
Then:
  - decode accuracy per feature (5-fold CV)
  - ||b_task|| / ||b_gen|| ratio
  - correlation of b_task with localizer tuning at the expected orientation
    (template-replay signature).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
)


def run_pre_probe_trial(bundle, cue_id, timing):
    cfg = bundle.cfg
    device = cfg.device
    q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device=device)
    blank = make_blank_frame(1, cfg, device=device)
    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    m_end = None; r_l23_end = None
    for t in range(delay_end):
        frame = blank
        q_t = q_cue if t < cue_end else None
        _x, state, info = bundle.net(frame, state, q_t=q_t)
        bundle.net.l23_e.homeostasis.update(state.r_l23)
        bundle.net.h_e.homeostasis.update(state.r_h)
        if t == delay_end - 1:
            m_end = state.m.detach().clone()
            r_l23_end = state.r_l23.detach().clone()
    return m_end, r_l23_end


def run_localizer_trials(bundle, timing, n_orients=12, n_per=5):
    cfg = bundle.cfg
    device = cfg.device
    orients = np.linspace(0.0, 180.0, n_orients, endpoint=False)
    blank = make_blank_frame(1, cfg, device=device)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    curves = []
    for o in orients:
        probe = make_grating_frame(float(o), 1.0, cfg, device=device)
        per_trial = []
        for _ in range(n_per):
            state = bundle.net.initial_state(batch_size=1)
            probe_rs = []
            for t in range(probe1_end):
                if t < cue_end:
                    frame = blank
                elif t < delay_end:
                    frame = blank
                else:
                    frame = probe
                _x, state, info = bundle.net(frame, state, q_t=None)
                bundle.net.l23_e.homeostasis.update(state.r_l23)
                bundle.net.h_e.homeostasis.update(state.r_h)
                if delay_end <= t < probe1_end:
                    probe_rs.append(info["r_l23"][0].clone())
            per_trial.append(torch.stack(probe_rs, dim=0).mean(dim=0))
        curves.append(torch.stack(per_trial, dim=0).mean(dim=0))
    return orients, torch.stack(curves, dim=0)  # [n_orients, n_l23]


def main(ckpt_path: Path, n_trials_per_cue: int = 100, n_localizer_per: int = 5):
    bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cue_mapping = bundle.meta.get("cue_mapping", cue_mapping_from_seed(42))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}
    cm = bundle.net.context_memory
    W_mh_gen = cm.W_mh_gen.data
    W_mh_task = cm.W_mh_task.data

    timing = KokTiming()

    # Collect features per trial
    m_feats, b_task_feats, b_gen_feats, r_l23_feats, labels = [], [], [], [], []
    orient_label = []
    for cue_id in (0, 1):
        for _ in range(n_trials_per_cue):
            m, r = run_pre_probe_trial(bundle, cue_id, timing)
            b_task = F.linear(m, W_mh_task)
            b_gen = F.linear(m, W_mh_gen)
            m_feats.append(m.reshape(-1).numpy())
            b_task_feats.append(b_task.reshape(-1).numpy())
            b_gen_feats.append(b_gen.reshape(-1).numpy())
            r_l23_feats.append(r.reshape(-1).numpy())
            labels.append(cue_id)
            orient_label.append(cue_mapping[cue_id])

    X_m = np.stack(m_feats); X_bt = np.stack(b_task_feats)
    X_bg = np.stack(b_gen_feats); X_r = np.stack(r_l23_feats)
    y = np.array(labels)

    def dec(X):
        clf = LogisticRegression(max_iter=2000)
        s = cross_val_score(clf, X, y, cv=5)
        return float(s.mean()), float(s.std())
    acc_m = dec(X_m); acc_bt = dec(X_bt); acc_bg = dec(X_bg); acc_r = dec(X_r)

    # Norm ratios per trial
    bt_norm = np.linalg.norm(X_bt, axis=1)
    bg_norm = np.linalg.norm(X_bg, axis=1)
    ratio = bt_norm / np.maximum(bg_norm, 1e-12)

    # Localizer — for template-replay check
    orients, tuning = run_localizer_trials(
        bundle, timing, n_orients=12, n_per=n_localizer_per)
    tuning_np = tuning.numpy()  # [n_orients, n_l23]

    # Mean b_task per cue
    bt_c0 = X_bt[y == 0].mean(axis=0)
    bt_c1 = X_bt[y == 1].mean(axis=0)
    # Expected orientation for each cue: cue_mapping[cue_id]
    def nearest_orient_idx(target_deg):
        return int(np.argmin(np.abs(orients - target_deg)))

    exp_idx_c0 = nearest_orient_idx(cue_mapping[0])
    exp_idx_c1 = nearest_orient_idx(cue_mapping[1])
    # Correlate b_task(cue=c) with localizer_r_l23(orient=expected[c])
    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    corr_c0_expected = pearson(bt_c0, tuning_np[exp_idx_c0])
    corr_c0_unexpected = pearson(bt_c0, tuning_np[exp_idx_c1])
    corr_c1_expected = pearson(bt_c1, tuning_np[exp_idx_c1])
    corr_c1_unexpected = pearson(bt_c1, tuning_np[exp_idx_c0])

    # All orientations correlation
    corrs_c0_all = [pearson(bt_c0, tuning_np[i]) for i in range(len(orients))]
    corrs_c1_all = [pearson(bt_c1, tuning_np[i]) for i in range(len(orients))]
    argmax_c0 = int(np.argmax(corrs_c0_all))
    argmax_c1 = int(np.argmax(corrs_c1_all))

    out = {
        "n_trials_per_cue": n_trials_per_cue,
        "cue_mapping": cue_mapping,
        "decode_accuracy": {
            "m_(memory)": acc_m,
            "b_task": acc_bt,
            "b_gen": acc_bg,
            "r_l23_pre_probe": acc_r,
        },
        "W_mh_task_norm": float(W_mh_task.norm().item()),
        "W_mh_gen_norm": float(W_mh_gen.norm().item()),
        "b_task_norm_mean": float(bt_norm.mean()),
        "b_gen_norm_mean": float(bg_norm.mean()),
        "b_task_over_b_gen_ratio_mean": float(ratio.mean()),
        "b_task_over_b_gen_ratio_median": float(np.median(ratio)),
        "template_replay_check": {
            "cue0_b_task_corr_with_expected_orient_localizer": corr_c0_expected,
            "cue0_b_task_corr_with_unexpected_orient_localizer": corr_c0_unexpected,
            "cue1_b_task_corr_with_expected_orient_localizer": corr_c1_expected,
            "cue1_b_task_corr_with_unexpected_orient_localizer": corr_c1_unexpected,
            "cue0_argmax_orient_from_corr_deg": float(orients[argmax_c0]),
            "cue1_argmax_orient_from_corr_deg": float(orients[argmax_c1]),
            "cue0_expected_orient_deg": cue_mapping[0],
            "cue1_expected_orient_deg": cue_mapping[1],
            "corrs_c0_by_orient": {
                float(orients[i]): float(corrs_c0_all[i])
                for i in range(len(orients))
            },
            "corrs_c1_by_orient": {
                float(orients[i]): float(corrs_c1_all[i])
                for i in range(len(orients))
            },
        },
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt"))
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--n-localizer", type=int, default=5)
    args = p.parse_args()
    main(args.ckpt, args.n_trials, args.n_localizer)
