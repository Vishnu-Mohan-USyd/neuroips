"""Task #70: decode leader token identity from ContextMemory m state
at the end of the leader window (t = leader_end - 1).

Reports 5-fold CV accuracy of a LogisticRegression classifier
trained on m ∈ R^n_m → leader_pos ∈ {0, ..., 5}.
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
from scripts.v2.train_phase3_richter_learning import (
    RichterTiming, build_leader_tensor, N_LEAD_TRAIL,
    LEADER_TOKEN_IDX, TRAILER_TOKEN_IDX,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", type=Path,
        default=Path("checkpoints/v2/phase3_richter_task70/phase3_richter_s42.pt"),
    )
    p.add_argument("--n-trials-per-token", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    cfg = ModelConfig(seed=int(args.seed), device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=int(args.seed))
    net.set_phase("phase3_richter")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["state_dict"], strict=False)

    timing = RichterTiming()
    leader_end = timing.leader_steps

    m_vecs: list[np.ndarray] = []
    labels: list[int] = []
    for pos in range(N_LEAD_TRAIL):
        leader_tok_idx = LEADER_TOKEN_IDX[pos]
        leader_frame = bank.tokens[leader_tok_idx:leader_tok_idx + 1].unsqueeze(
            0
        )                                          # [1, 1, H, W] — but tokens are [N,1,H,W]
        leader_frame = bank.tokens[
            leader_tok_idx:leader_tok_idx + 1
        ]                                          # [1, 1, H, W]
        l_cue = build_leader_tensor(pos, cfg.arch.n_h_e, device="cpu")
        for trial in range(args.n_trials_per_token):
            state = net.initial_state(batch_size=1)
            for t in range(leader_end):
                _x_hat, state, _info = net(
                    leader_frame, state, leader_t=l_cue,
                )
                if t == leader_end - 1:
                    m_vecs.append(
                        state.m.detach().clone().reshape(-1).numpy()
                    )
                    labels.append(int(pos))
                    break

    M = np.stack(m_vecs)
    y = np.asarray(labels)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    acc = float(cross_val_score(clf, M, y, cv=5).mean())
    print(json.dumps({
        "n_trials_total": int(M.shape[0]),
        "n_classes": int(y.max() + 1),
        "richter_lead_decode_from_m": acc,
        "chance": 1.0 / N_LEAD_TRAIL,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
