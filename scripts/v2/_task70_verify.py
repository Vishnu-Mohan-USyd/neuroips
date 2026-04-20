"""Task #70 verification probe.

Runs Kok trial simulation from a Phase-3 Kok checkpoint, collects the
memory state ``m`` at delay_end−1 for cue_id=0 vs cue_id=1, and reports:

- ``cos_sim_cue0_cue1`` : cosine similarity between class-mean m vectors
  (Target <0.95; pre-Task-#70 was 0.999)
- ``c_expected_decode_acc`` : LogReg 5-fold CV accuracy of expected
  orientation from raw m vectors (Target >0.70; pre-Task-#70 was 0.52)
- ``h_expected_decode_acc`` / ``b_l23_expected_decode_acc`` for reference
- ``W_qm_task_norm`` / ``W_mh_task_norm`` for weight size context
- ``cue_gain`` (should be 5.0 post-Task-#70)

Reuses the c5 trial loop from ``_debug_task67_c1_c5``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
    make_blank_frame, make_grating_frame,
)


def collect_features(
    ckpt_path: Path, n_trials: int = 300, seed: int = 42,
) -> dict:
    torch.manual_seed(123)
    cfg = ModelConfig(seed=seed, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase3_kok")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    net.load_state_dict(sd, strict=False)
    cue_mapping = blob.get("cue_mapping", cue_mapping_from_seed(seed))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}

    timing = KokTiming()
    n_total = timing.total
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps

    m_delay: list[np.ndarray] = []
    h_delay: list[np.ndarray] = []
    b_delay: list[np.ndarray] = []
    labels: list[int] = []
    np_rng = np.random.default_rng(321)
    for _ in range(n_trials):
        cue_id = int(np_rng.integers(0, 2))
        probe_deg = float(cue_mapping[cue_id])
        q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
        blank = make_blank_frame(1, cfg, device="cpu")
        probe = make_grating_frame(probe_deg, 1.0, cfg, device="cpu")
        state = net.initial_state(batch_size=1)
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
            if t == delay_end - 1:
                m_delay.append(state.m.detach().clone().reshape(-1).numpy())
                h_delay.append(state.r_h.detach().clone().reshape(-1).numpy())
                b_delay.append(
                    info["b_l23"].detach().clone().reshape(-1).numpy()
                )
                break
        labels.append(cue_id)

    y = np.asarray(labels)
    M = np.stack(m_delay)          # [n_trials, n_m]
    H = np.stack(h_delay)
    B = np.stack(b_delay)

    def _acc(X: np.ndarray) -> tuple[float, float]:
        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, X, y, cv=5)
        return float(scores.mean()), float(scores.std())

    def _cos(m: np.ndarray) -> float:
        num = float((m[0] * m[1]).sum())
        denom = float(np.linalg.norm(m[0]) * np.linalg.norm(m[1]) + 1e-12)
        return num / denom

    class_mean_m = np.stack([M[y == k].mean(axis=0) for k in (0, 1)])
    cos_sim = _cos(class_mean_m)
    c_acc, c_std = _acc(M)
    h_acc, h_std = _acc(H)
    b_acc, b_std = _acc(B)

    return {
        "n_trials": int(n_trials),
        "label_balance_p1": float((y == 1).mean()),
        "cos_sim_cue0_cue1": cos_sim,
        "c_expected_decode_acc": c_acc,
        "c_expected_decode_std": c_std,
        "h_expected_decode_acc": h_acc,
        "b_l23_expected_decode_acc": b_acc,
        "W_qm_task_norm": float(net.context_memory.W_qm_task.norm().item()),
        "W_mh_task_norm": float(net.context_memory.W_mh_task.norm().item()),
        "W_lm_task_norm": float(net.context_memory.W_lm_task.norm().item()),
        "cue_gain": float(net.context_memory.cue_gain),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Task #70 cue-pathway verification")
    p.add_argument(
        "--ckpt", type=Path,
        default=Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt"),
    )
    p.add_argument("--n-trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    out = collect_features(args.ckpt, n_trials=args.n_trials, seed=args.seed)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
