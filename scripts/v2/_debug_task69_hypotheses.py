"""Task #69 — diagnose why cue→memory pathway doesn't produce cue-specific
content after Phase-3 Kok training.

Tests four hypotheses with tested evidence:

H-W: cue_0 and cue_1 drive the same memory pattern (W_qm_task learns a
     cue-independent direction).
H-M: memory discriminates but W_mh_task projects both to same bias.
H-T: three-factor rule is structurally cue-invariant in current configuration.
H-R: 500 learning trials is too few; extend to 2000 and re-measure C-decode.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
    make_blank_frame, make_grating_frame, run_kok_trial,
    run_phase3_kok_training,
)


def _load_net_from_ckpt(ckpt_path: Path) -> tuple:
    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase3_kok")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["state_dict"], strict=False)
    cue_mapping = blob.get("cue_mapping", cue_mapping_from_seed(42))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}
    return cfg, net, cue_mapping


def _run_cue_trial_collect_m_end(
    net, cfg, cue_id: int, probe_deg: float, timing: KokTiming,
):
    """Run trial (no plasticity); return m at delay_end-1 and at cue_end-1."""
    q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
    blank = make_blank_frame(1, cfg, device="cpu")
    probe = make_grating_frame(probe_deg, 1.0, cfg, device="cpu")
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    state = net.initial_state(batch_size=1)
    m_cueend = None
    m_delay = None
    with torch.no_grad():
        for t in range(delay_end):
            if t < cue_end:
                frame, q_t = blank, q_cue
            else:
                frame, q_t = blank, None
            _x, state, info = net(frame, state, q_t=q_t)
            net.l23_e.homeostasis.update(state.r_l23)
            net.h_e.homeostasis.update(state.r_h)
            if t == cue_end - 1:
                m_cueend = state.m.detach().clone()
            if t == delay_end - 1:
                m_delay = state.m.detach().clone()
    return m_cueend, m_delay


def h_w_and_h_m(ckpt_path: Path) -> dict:
    """H-W / H-M: load trained net, measure m_end per cue + W_mh @ m_end."""
    cfg, net, cue_mapping = _load_net_from_ckpt(ckpt_path)
    timing = KokTiming()
    m_c0_cueend, m_c0_delay = _run_cue_trial_collect_m_end(
        net, cfg, 0, cue_mapping[0], timing)
    m_c1_cueend, m_c1_delay = _run_cue_trial_collect_m_end(
        net, cfg, 1, cue_mapping[1], timing)

    def cos(a, b):
        a = a.flatten(); b = b.flatten()
        return float(torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item())

    # H-W: m_end_cue0 vs m_end_cue1
    c0_norm = float(m_c0_delay.norm().item())
    c1_norm = float(m_c1_delay.norm().item())
    c_cos = cos(m_c0_delay, m_c1_delay)
    c_cueend_cos = cos(m_c0_cueend, m_c1_cueend)

    # H-M: W_mh_task @ m_end
    W_mh_task = net.context_memory.W_mh_task.data  # [n_out, n_m]
    b_c0 = torch.nn.functional.linear(m_c0_delay, W_mh_task)  # [1, n_out]
    b_c1 = torch.nn.functional.linear(m_c1_delay, W_mh_task)
    b_c0_norm = float(b_c0.norm().item())
    b_c1_norm = float(b_c1.norm().item())
    b_cos = cos(b_c0, b_c1)

    # W_mh_task structure
    W_mh_task_norm = float(W_mh_task.norm().item())
    W_mh_task_max = float(W_mh_task.abs().max().item())
    # Total bias (generic + task)
    W_mh_gen = net.context_memory.W_mh_gen.data
    b_gen_c0 = torch.nn.functional.linear(m_c0_delay, W_mh_gen)
    b_gen_c1 = torch.nn.functional.linear(m_c1_delay, W_mh_gen)
    b_total_c0 = b_gen_c0 + b_c0
    b_total_c1 = b_gen_c1 + b_c1
    b_total_cos = cos(b_total_c0, b_total_c1)

    return {
        "H_W": {
            "cue_0_memory_state_norm": c0_norm,
            "cue_1_memory_state_norm": c1_norm,
            "cue_differentiation_cosine_delay": c_cos,
            "cue_differentiation_cosine_cueend": c_cueend_cos,
            "m_absdiff_max_delay": float(
                (m_c0_delay - m_c1_delay).abs().max().item()),
            "m_absdiff_mean_delay": float(
                (m_c0_delay - m_c1_delay).abs().mean().item()),
            "verdict": "CONFIRMED" if c_cos > 0.99 else "FALSIFIED",
        },
        "H_M": {
            "b_l23_cue0_norm": b_c0_norm,
            "b_l23_cue1_norm": b_c1_norm,
            "b_l23_differentiation_cosine_taskonly": b_cos,
            "b_total_differentiation_cosine": b_total_cos,
            "W_mh_task_norm": W_mh_task_norm,
            "W_mh_task_abs_max": W_mh_task_max,
            "verdict": "CONFIRMED" if b_cos > 0.99 else "FALSIFIED",
        },
        "W_qm_task_norm": float(net.context_memory.W_qm_task.data.norm().item()),
    }


def h_t_rule_symmetry() -> dict:
    """H-T: run short instrumented training from scratch; collect dw_qm by cue."""
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42)
    # We need a Phase-2 pretrained net for realism, but for H-T (which only
    # asks about *rule structure*) a fresh net is sufficient: the same random
    # init produces the same cue/memory activations for a fixed cue_id, so
    # per-cue dw_qm statistics isolate rule structure from learned state.
    net.set_phase("phase3_kok")
    cue_mapping = cue_mapping_from_seed(42)
    timing = KokTiming()
    from src.v2_model.plasticity import ThreeFactorRule
    rule = ThreeFactorRule(lr=1e-3, weight_decay=1e-5)

    # Log per-cue dw_qm matrices — we want to see whether updates differ
    # between cue_A (cue=0) and cue_B (cue=1).
    rows_cue0 = []
    rows_cue1 = []
    dw_qm_full_cue0 = []
    dw_qm_full_cue1 = []
    # Instrument run_kok_trial by patching ThreeFactorRule.delta_qm.
    orig_delta_qm = rule.delta_qm
    captured = {"last": None}

    def patched_delta_qm(*args, **kwargs):
        dw = orig_delta_qm(*args, **kwargs)
        captured["last"] = dw.detach().clone()
        return dw
    rule.delta_qm = patched_delta_qm

    for trial in range(40):
        cue_id = trial % 2  # alternate
        probe_deg = float(cue_mapping[cue_id])
        info = run_kok_trial(
            net, cfg,
            cue_id=cue_id, probe_orientation_deg=probe_deg,
            timing=timing, rule=rule,
            noise_std=0.0, device="cpu", apply_plasticity=True,
        )
        dw_full = captured["last"]
        # dw_full: [n_m, n_cue]. Columns 0 and 1 are the cue slots.
        col0_absmean = float(dw_full[:, 0].abs().mean().item())
        col1_absmean = float(dw_full[:, 1].abs().mean().item())
        col0_max = float(dw_full[:, 0].abs().max().item())
        col1_max = float(dw_full[:, 1].abs().max().item())
        row = {
            "trial": trial, "cue_id": cue_id,
            "dw_qm_abs_mean": float(info["dw_qm_abs_mean"].item()),
            "dw_qm_col0_absmean": col0_absmean,
            "dw_qm_col1_absmean": col1_absmean,
            "dw_qm_col0_max": col0_max,
            "dw_qm_col1_max": col1_max,
        }
        if cue_id == 0:
            rows_cue0.append(row)
            dw_qm_full_cue0.append(dw_full.clone())
        else:
            rows_cue1.append(row)
            dw_qm_full_cue1.append(dw_full.clone())

    # Aggregate: mean dw_qm magnitude across trials for each cue.
    # Also: does dw_qm[:, cue_id] > dw_qm[:, 1-cue_id]? (structurally it should.)
    dw_cue0_mean = float(np.mean([r["dw_qm_abs_mean"] for r in rows_cue0]))
    dw_cue1_mean = float(np.mean([r["dw_qm_abs_mean"] for r in rows_cue1]))
    # Column-wise: for cue_id=0 trials, col0 should be nonzero, col1 should be zero.
    c0_col0_mean = float(np.mean([r["dw_qm_col0_absmean"] for r in rows_cue0]))
    c0_col1_mean = float(np.mean([r["dw_qm_col1_absmean"] for r in rows_cue0]))
    c1_col0_mean = float(np.mean([r["dw_qm_col0_absmean"] for r in rows_cue1]))
    c1_col1_mean = float(np.mean([r["dw_qm_col1_absmean"] for r in rows_cue1]))
    # Symmetry test: |dw_cue0 - dw_cue1| / max(dw_cue0, dw_cue1)
    if max(dw_cue0_mean, dw_cue1_mean) > 0:
        sym_ratio = abs(dw_cue0_mean - dw_cue1_mean) / max(dw_cue0_mean, dw_cue1_mean)
    else:
        sym_ratio = 0.0
    symmetric = sym_ratio < 0.1  # updates within 10% → rule is cue-invariant

    # Cross-cue column separation: the key structural test.
    # If the rule is cue-specific by construction, c0_col1_mean should be ~0.
    cross_col_leakage = max(c0_col1_mean, c1_col0_mean) / max(
        c0_col0_mean, c1_col1_mean, 1e-20)

    return {
        "n_trials_per_cue": len(rows_cue0),
        "dw_cue_A_mean": dw_cue0_mean,
        "dw_cue_B_mean": dw_cue1_mean,
        "symmetric_within_10pct": bool(symmetric),
        "symmetry_ratio": sym_ratio,
        "cue0_col0_mean": c0_col0_mean,
        "cue0_col1_mean": c0_col1_mean,
        "cue1_col0_mean": c1_col0_mean,
        "cue1_col1_mean": c1_col1_mean,
        "cross_column_leakage_ratio": cross_col_leakage,
        "verdict": "CONFIRMED" if symmetric and cross_col_leakage < 0.01 else "FALSIFIED",
    }


def h_r_longer_training(
    phase2_ckpt: Path, n_trials: int, seed: int = 42,
) -> dict:
    """H-R: rebuild net from Phase-2 checkpoint, run n_trials learning, measure decode."""
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    blob = torch.load(phase2_ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["state_dict"], strict=False)
    net.set_phase("phase3_kok")
    cue_mapping = cue_mapping_from_seed(seed)

    t0 = time.monotonic()
    run_phase3_kok_training(
        net=net, n_trials_learning=n_trials, n_trials_scan=0,
        validity_scan=1.0, lr=1e-3, weight_decay=1e-5, seed=seed,
        cue_mapping=cue_mapping, metrics_path=None, log_every=max(n_trials // 10, 1),
    )
    train_secs = time.monotonic() - t0

    # Now measure C-decode.
    timing = KokTiming()
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    n_total = timing.total
    C_feats = []
    labels = []
    np_rng = np.random.default_rng(321)
    for k in range(300):
        cue_id = int(np_rng.integers(0, 2))
        probe_deg = float(cue_mapping[cue_id])
        q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
        blank = make_blank_frame(1, cfg, device="cpu")
        probe = make_grating_frame(probe_deg, 1.0, cfg, device="cpu")
        state = net.initial_state(batch_size=1)
        c_delay = None
        with torch.no_grad():
            for t in range(n_total):
                if t < cue_end:
                    frame, q_t = blank, q_cue
                elif t < delay_end:
                    frame, q_t = blank, None
                else:
                    frame, q_t = probe, None
                _x, state, info = net(frame, state, q_t=q_t)
                net.l23_e.homeostasis.update(state.r_l23)
                net.h_e.homeostasis.update(state.r_h)
                if t == delay_end - 1:
                    c_delay = state.m.detach().clone().reshape(-1).numpy()
                    break
        C_feats.append(c_delay)
        labels.append(cue_id)
    X = np.stack(C_feats); y = np.array(labels)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, X, y, cv=5)
    y_shuf = np.random.default_rng(0).permutation(y)
    shuf = float(cross_val_score(
        LogisticRegression(max_iter=2000), X, y_shuf, cv=5).mean())
    return {
        "n_trials": n_trials,
        "train_seconds": train_secs,
        "c_decode_acc": float(scores.mean()),
        "c_decode_std": float(scores.std()),
        "shuffle_ctrl": shuf,
        "label_balance_p1": float((y == 1).mean()),
        "W_qm_task_norm": float(net.context_memory.W_qm_task.data.norm().item()),
        "W_mh_task_norm": float(net.context_memory.W_mh_task.data.norm().item()),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok/phase3_kok_s42.pt"))
    p.add_argument("--phase2-ckpt", type=Path,
                   default=Path("checkpoints/v2/phase2/phase2_s42/step_3000.pt"))
    p.add_argument("--hw-hm", action="store_true")
    p.add_argument("--ht", action="store_true")
    p.add_argument("--hr", action="store_true")
    p.add_argument("--hr-n-trials", type=int, default=2000)
    args = p.parse_args()
    out = {}
    if args.hw_hm:
        out["H_W_and_H_M"] = h_w_and_h_m(args.ckpt)
    if args.ht:
        out["H_T"] = h_t_rule_symmetry()
    if args.hr:
        out["H_R"] = h_r_longer_training(args.phase2_ckpt, args.hr_n_trials)
    print(json.dumps(out, indent=2, default=str))
