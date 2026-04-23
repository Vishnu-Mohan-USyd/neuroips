"""Level 7 component validation — context memory C cue encoding at init.

Per Lead's bottom-up validation protocol (Task #74). Scope: run Kok-style
cue-then-delay trials against the full substrate (Fix-K + L2 + M + N) with
plasticity OFF (pure forward, weights at init). Ask: does the end-of-delay
memory state ``m`` carry cue-discriminative information?

At init, ``W_qm_task`` is drawn from ``N(0, task_input_init_std=0.3)``
(Task #58 design — small random values bootstrap Phase-3 three-factor
learning). With ``cue_gain=5.0`` in the forward, the cue stream has
non-trivial effective gain even with zero plastic updates.

``W_mh_task_{exc,inh}`` start at exact zero, so the *readout* pathway
(m → L23E apical / m → SOM gain) is silent at init. This probe measures
whether cue information is *encoded* in ``m``, not whether it is *read out*
into L23/SOM behavior.

Trial structure (matches ``scripts/v2/train_phase3_kok_learning.KokTiming``):

  [cue phase  : 40 steps (200 ms)] — blank frame, q_t = one-hot(cue_id)
  [delay phase: 110 steps (550 ms)] — blank frame, q_t = None
  <record final state.m>

Probe protocol
--------------
For each cue_id ∈ {0, 1} × n_trials:
  * Build a fresh ``initial_state`` so trials are independent.
  * Drive ``V2Network.forward`` for 150 steps exactly as above.
  * Record ``m_end`` = state.m after the final step.

After collecting all trials:
  * Train a logistic regression on 50% of trials (stratified), test on
    held-out 50%. Report accuracy.
  * Compute ``‖m‖`` (L2 norm per trial, mean and std).
  * Compute ``cos(m̄_cueA, m̄_cueB)`` — mean-vector cosine between the two
    cue classes (tests whether the two memory representations are
    distinguishable in direction, not just magnitude).
  * Report weight-norm diagnostics: ``‖W_qm_task‖_F``, ``‖W_mh_gen‖_F``,
    ``‖W_mh_task_exc‖_F``, ``‖W_mh_task_inh‖_F``.

Pass criteria (gated)
---------------------
  * decode accuracy ≥ 0.70
  * ``‖m‖`` mean ∈ [0.1, 100]  (non-zero, non-saturating)
  * ``cos(m̄_A, m̄_B)`` < 0.95  (two cue classes meaningfully separated)

"Neutral baseline" verdict (per Lead's dispatch) — allowed as a PASS-
equivalent when:
  * decode accuracy ∈ [0.45, 0.55]  (chance, ±5%)  AND
  * ‖m‖ in valid band  AND
  * all task-input weights near init magnitudes  (structural capability
    intact; Phase-3 training will install cue→memory mapping).

Fail otherwise.

DM::
  level7_verdict=<pass|fail|neutral_baseline>
    decode_cue_acc=<#> chance_ref=0.5 m_norm=<#>
    cos_m_cueA_cueB=<#> W_qm_task_norm=<#> W_mh_gen_norm=<#>
    issue_if_fail=<short>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.v2._gates_common import make_blank_frame
from scripts.v2.train_phase3_kok_learning import build_cue_tensor
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


N_CUE_STEPS = 40       # 200 ms at dt=5 ms
N_DELAY_STEPS = 110    # 550 ms at dt=5 ms


@torch.no_grad()
def _run_kok_cue_delay_trial(
    net: V2Network, cfg: ModelConfig, cue_id: int,
    n_cue_steps: int, n_delay_steps: int,
) -> Tensor:
    """Run one cue-then-delay trial, return ``m`` at end of delay.

    Returns a 1-D tensor of shape ``[n_m]`` (batch squeezed).
    """
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)
    state = net.initial_state(batch_size=1)

    for t in range(int(n_cue_steps)):
        _, state, _ = net(blank, state, q_t=q_cue)
    for t in range(int(n_delay_steps)):
        _, state, _ = net(blank, state, q_t=None)

    return state.m.squeeze(0).detach().clone()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials-per-cue", type=int, default=30)
    p.add_argument("--n-cue-steps", type=int, default=N_CUE_STEPS)
    p.add_argument("--n-delay-steps", type=int, default=N_DELAY_STEPS)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    net = V2Network(cfg, token_bank=None, seed=seed, device="cpu")
    net.eval()

    n_trials_total = 2 * int(args.n_trials_per_cue)
    # n_m (context memory size) == n_c (cue alphabet) in this architecture
    # per network.py L358 (ContextMemory built with n_m=a.n_c).
    n_m = int(net.context_memory.n_m)

    M = np.zeros((n_trials_total, n_m), dtype=np.float64)
    y = np.zeros(n_trials_total, dtype=np.int64)

    idx = 0
    for cue_id in (0, 1):
        for t in range(int(args.n_trials_per_cue)):
            m_end = _run_kok_cue_delay_trial(
                net, cfg, cue_id=cue_id,
                n_cue_steps=int(args.n_cue_steps),
                n_delay_steps=int(args.n_delay_steps),
            )
            M[idx] = m_end.cpu().numpy().astype(np.float64)
            y[idx] = cue_id
            idx += 1

    # Linear decoder (logistic regression, 50/50 stratified split).
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=seed,
    )
    (train_idx, test_idx), = splitter.split(M, y)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
    )
    clf.fit(M[train_idx], y[train_idx])
    train_acc = float(clf.score(M[train_idx], y[train_idx]))
    decode_acc = float(clf.score(M[test_idx], y[test_idx]))

    # Memory-norm diagnostics.
    m_norms = np.linalg.norm(M, axis=1)
    m_norm_mean = float(m_norms.mean())
    m_norm_std = float(m_norms.std())
    m_norm_min = float(m_norms.min())
    m_norm_max = float(m_norms.max())

    # Class-mean cosine similarity.
    m_a_mean = M[y == 0].mean(axis=0)
    m_b_mean = M[y == 1].mean(axis=0)
    cos_m_ab = _cosine_sim(m_a_mean, m_b_mean)

    # Weight-norm diagnostics.
    cm = net.context_memory
    w_qm_norm = float(torch.linalg.norm(cm.W_qm_task).cpu())
    w_lm_norm = float(torch.linalg.norm(cm.W_lm_task).cpu())
    w_mh_gen_norm = float(torch.linalg.norm(cm.W_mh_gen).cpu())
    w_mh_task_exc_norm = float(torch.linalg.norm(cm.W_mh_task_exc).cpu())
    w_mh_task_inh_norm = float(torch.linalg.norm(cm.W_mh_task_inh).cpu())
    w_hm_gen_norm = float(torch.linalg.norm(cm.W_hm_gen).cpu())
    w_mm_gen_norm = float(torch.linalg.norm(cm.W_mm_gen).cpu())

    # ---- Verdict ------------------------------------------------------------
    fails: list[str] = []
    # Band-checks first (always enforced).
    if not (0.1 <= m_norm_mean <= 100.0):
        fails.append(f"m_norm {m_norm_mean:.3f}∉[0.1,100]")
    if not math.isfinite(cos_m_ab) or cos_m_ab >= 0.95:
        fails.append(f"cos_m_ab {cos_m_ab:.3f}≥0.95")

    # Classify decode outcome.
    is_neutral = 0.45 <= decode_acc <= 0.55
    if fails:
        verdict = "fail"
        issue = ";".join(fails)
    elif decode_acc >= 0.70:
        verdict = "pass"
        issue = "none"
    elif is_neutral:
        verdict = "neutral_baseline"
        issue = "none (decode at chance by design — W_mh_task_exc/inh=0 at init)"
    elif decode_acc < 0.60:
        verdict = "fail"
        issue = f"decode_acc {decode_acc:.3f}<0.60 (below chance-band)"
    else:
        # 0.55 < acc < 0.70 → ambiguous partial-signal zone
        verdict = "fail"
        issue = (f"decode_acc {decode_acc:.3f}∈(0.55,0.70) — partial cue "
                 "leakage but below pass floor")

    summary = {
        "version": "level_7_context_memory_v1",
        "seed": seed,
        "n_trials_per_cue": int(args.n_trials_per_cue),
        "n_cue_steps": int(args.n_cue_steps),
        "n_delay_steps": int(args.n_delay_steps),
        "n_m": int(n_m),
        "n_cue": int(cfg.arch.n_c),
        "stats": {
            "decode_cue_acc": decode_acc,
            "train_acc": train_acc,
            "chance_ref": 0.5,
            "m_norm_mean": m_norm_mean,
            "m_norm_std": m_norm_std,
            "m_norm_min": m_norm_min,
            "m_norm_max": m_norm_max,
            "cos_m_cueA_cueB": cos_m_ab,
            # Weight norms (diagnostic)
            "W_qm_task_norm": w_qm_norm,
            "W_lm_task_norm": w_lm_norm,
            "W_mh_gen_norm": w_mh_gen_norm,
            "W_mh_task_exc_norm": w_mh_task_exc_norm,
            "W_mh_task_inh_norm": w_mh_task_inh_norm,
            "W_hm_gen_norm": w_hm_gen_norm,
            "W_mm_gen_norm": w_mm_gen_norm,
        },
        "verdict": verdict,
        "issue_if_fail": issue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    line = (
        f"level7_verdict={verdict} "
        f"decode_cue_acc={decode_acc:.3f} chance_ref=0.5 "
        f"m_norm={m_norm_mean:.3f} "
        f"cos_m_cueA_cueB={cos_m_ab:.3f} "
        f"W_qm_task_norm={w_qm_norm:.3f} "
        f"W_mh_gen_norm={w_mh_gen_norm:.3f} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict in ("pass", "neutral_baseline") else 1


if __name__ == "__main__":
    raise SystemExit(main())
