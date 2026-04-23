"""Task #67 — verify critique claims C1 (phase-2 temporal discontinuity) and
C5 (expected-orientation decode from C/H/b_l23 after cue, before probe).

C1: run 20 Phase-2 training steps; record m.norm() per step and explicitly
check world trajectory reseeding behaviour. We do not need to modify the
trainer — we just instantiate the world the same way and observe: each step
gets seeds = [offset + (warmup+step)*B + b for b in B] → each (step,b) is a
fresh seed → fresh world.trajectory call → no regime continuity.

C5: load Kok checkpoint, run 300 trials (100% validity, 2 cues, fixed
preferred probe per cue), collect C/H/b_l23 at t=delay_end-1 (after cue,
before probe), and the cue_id label (which determines the expected
orientation). Train a linear decoder (LogisticRegression) per feature and
report accuracy via 5-fold CV.
"""
from __future__ import annotations
import sys, json, math
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
from scripts.v2.train_phase2_predictive import build_world, sample_batch_window, _forward_window
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
    make_blank_frame, make_grating_frame,
)


# --- C1 -----------------------------------------------------------------
def c1_check() -> dict:
    torch.manual_seed(42)
    cfg = ModelConfig(seed=42, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase2")
    seed_offset = 42 * 10_000
    batch_size = 4
    warmup = 0  # skip warmup for speed; does not affect the C1 question
    state = net.initial_state(batch_size=batch_size)
    m_norms = []
    seeds_used = []
    world_states_hashes = []
    for step in range(20):
        seeds = [seed_offset + (warmup + step) * batch_size + b
                 for b in range(batch_size)]
        seeds_used.append(list(seeds))
        # sample_batch_window returns just frames; to introspect regime
        # state we must call world.trajectory directly.
        regime_tags = []
        for s in seeds:
            _frames, states = world.trajectory(s, 2)
            # states may carry regime info; try common attribute names
            rid = None
            for attr in ("regime_id", "regime", "regime_index",
                         "current_regime"):
                if hasattr(states[-1], attr):
                    rid = getattr(states[-1], attr)
                    break
            if rid is None and isinstance(states[-1], dict):
                rid = states[-1].get("regime_id", None)
            regime_tags.append(str(rid))
        world_states_hashes.append(regime_tags)

        frames = sample_batch_window(world, seeds, n_steps_per_window=2)
        s0, s1, s2, i0, i1, xh0, _ = _forward_window(net, frames, state)
        m_norms.append(float(s2.m.norm().item()))
        state = s2

    # Question: is world regime persistent across steps for fixed batch[b]?
    # If seeds differ every step, each (step,b) draws an independent
    # trajectory → no persistence.
    seed_persistence = all(
        seeds_used[t][b] != seeds_used[t+1][b]
        for t in range(len(seeds_used) - 1) for b in range(batch_size)
    )
    world_state_persistent = not seed_persistence
    return {
        "seeds_step0": seeds_used[0],
        "seeds_step1": seeds_used[1],
        "seeds_differ_each_step_for_same_b": seed_persistence,
        "world_state_persistent_across_steps": world_state_persistent,
        "m_norm_first_5": m_norms[:5],
        "m_norm_last_5": m_norms[-5:],
    }


# --- C5 -----------------------------------------------------------------
def c5_check(ckpt_path: Path, n_trials: int = 300) -> dict:
    torch.manual_seed(123)
    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase3_kok")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    net.load_state_dict(sd, strict=False)
    cue_mapping = blob.get("cue_mapping", cue_mapping_from_seed(42))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}

    timing = KokTiming()
    n_total = timing.total
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps

    C_feats, H_feats, B_feats, labels = [], [], [], []
    np_rng = np.random.default_rng(321)
    for k in range(n_trials):
        cue_id = int(np_rng.integers(0, 2))
        probe_deg = float(cue_mapping[cue_id])  # 100% valid
        q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
        blank = make_blank_frame(1, cfg, device="cpu")
        probe = make_grating_frame(probe_deg, 1.0, cfg, device="cpu")
        state = net.initial_state(batch_size=1)
        c_delay, h_delay, b_delay = None, None, None
        c_cueend, h_cueend = None, None
        for t in range(n_total):
            if t < cue_end:
                frame, q_t = blank, q_cue
            elif t < delay_end:
                frame, q_t = blank, None
            else:
                frame, q_t = probe, None
            _x_hat, state, info = net(frame, state, q_t=q_t)
            with torch.no_grad():
                net.l23_e.homeostasis.update(state.r_l23)
                net.h_e.homeostasis.update(state.r_h)
            if t == cue_end - 1:
                c_cueend = state.m.detach().clone().reshape(-1).numpy()
                h_cueend = state.r_h.detach().clone().reshape(-1).numpy()
            if t == delay_end - 1:
                c_delay = state.m.detach().clone().reshape(-1).numpy()
                h_delay = state.r_h.detach().clone().reshape(-1).numpy()
                b_delay = info["b_l23"].detach().clone().reshape(-1).numpy()
                break
        C_feats.append((c_cueend, c_delay))
        H_feats.append((h_cueend, h_delay))
        B_feats.append(b_delay)
        labels.append(cue_id)

    X_c_cueend = np.stack([c[0] for c in C_feats])
    X_c_delay = np.stack([c[1] for c in C_feats])
    X_h_cueend = np.stack([h[0] for h in H_feats])
    X_h_delay = np.stack([h[1] for h in H_feats])
    X_b_delay = np.stack(B_feats)
    y = np.array(labels)
    def acc(X):
        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, X, y, cv=5)
        return float(scores.mean()), float(scores.std())
    a_c_cueend = acc(X_c_cueend)
    a_c_delay = acc(X_c_delay)
    a_h_cueend = acc(X_h_cueend)
    a_h_delay = acc(X_h_delay)
    a_b_delay = acc(X_b_delay)
    # Diagnostic: per-feature variance across trials, class-conditional means
    def diagnose(X, y_arr, name):
        x0 = X[y_arr == 0]; x1 = X[y_arr == 1]
        var_total = float(X.var(axis=0).mean())
        mu_diff = float(np.abs(x0.mean(axis=0) - x1.mean(axis=0)).mean())
        mu_diff_max = float(np.abs(x0.mean(axis=0) - x1.mean(axis=0)).max())
        return {"name": name, "var_across_trials_mean": var_total,
                "class_mean_diff_mean": mu_diff,
                "class_mean_diff_max": mu_diff_max,
                "X_min": float(X.min()), "X_max": float(X.max()),
                "X_std": float(X.std())}
    diag_c = diagnose(X_c_delay, y, "c_delay")
    diag_h = diagnose(X_h_delay, y, "h_delay")
    diag_b = diagnose(X_b_delay, y, "b_l23_delay")
    # Sanity control: label shuffle (chance baseline)
    y_shuf = np.random.default_rng(0).permutation(y)
    clf = LogisticRegression(max_iter=2000)
    shuf = float(cross_val_score(clf, X_c_delay, y_shuf, cv=5).mean())
    # Balance check
    p_label1 = float((y == 1).mean())
    return {
        "n_trials": n_trials,
        "label_balance_p1": p_label1,
        "shuffle_control_acc": shuf,
        "c_decode_at_cue_end": a_c_cueend,
        "c_expected_decode_acc": a_c_delay[0],
        "c_expected_decode_std": a_c_delay[1],
        "h_decode_at_cue_end": a_h_cueend,
        "h_expected_decode_acc": a_h_delay[0],
        "b_l23_expected_decode_acc": a_b_delay[0],
        "W_qm_task_norm": float(net.context_memory.W_qm_task.norm().item()),
        "W_mh_task_norm": float(net.context_memory.W_mh_task.norm().item()),
        "diag_c": diag_c, "diag_h": diag_h, "diag_b": diag_b,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok/phase3_kok_s42.pt"))
    p.add_argument("--c1", action="store_true")
    p.add_argument("--c5", action="store_true")
    p.add_argument("--n-trials", type=int, default=300)
    args = p.parse_args()
    out = {}
    if args.c1:
        out["c1"] = c1_check()
    if args.c5:
        out["c5"] = c5_check(args.ckpt, n_trials=args.n_trials)
    print(json.dumps(out, indent=2))
