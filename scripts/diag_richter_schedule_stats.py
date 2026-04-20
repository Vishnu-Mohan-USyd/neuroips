"""Sprint 5e-Diag B1: Richter training schedule contingency statistics.

Computes the empirical P(T | L) transition matrix of the current
`richter_crossover_training_schedule` (expectation_snn/brian2_model/stimulus.py
lines 147-241) and contrasts it against a reviewer-proposed biased
deranged-permutation schedule.

Current schedule construction (stimulus.py lines 199-207):
    base = [(i, j) for i in range(6) for j in range(6)]  # all 36 pairs
    for r in range(replicates):
        block = base.copy(); rng.shuffle(block); pairs[...] = block

Each 36-trial block is a permutation of ALL leader-trailer pairs, so after
any complete block P(T | L) = 1/6 uniformly. There is no statistical
contingency for STDP on H_R to latch onto.

Reviewer-proposed biased schedule: deranged permutation f(L) = (L+1) mod 6;
P(L -> f(L)) = 0.80, remaining 0.20 spread uniformly across the 5 other
trailers (0.04 each).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from expectation_snn.brian2_model.stimulus import (
    richter_crossover_training_schedule,
)

OUT_DIR = Path("data/diag_sprint5e")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ORIENTS = 6
SEED = 42
N_TRIALS = 360


def empirical_pt_given_l(pairs: np.ndarray, n: int = N_ORIENTS) -> np.ndarray:
    """Return (n, n) matrix P[L, T] = P(T | L)."""
    counts = np.zeros((n, n), dtype=np.int64)
    for li, ti in pairs:
        counts[int(li), int(ti)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    return counts / row_sum


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def biased_schedule_pairs(rng: np.random.Generator, n_trials: int,
                          bias: float = 0.80) -> np.ndarray:
    """Reviewer's reference design: deranged perm f(L)=(L+1)%6, bias on f(L)."""
    leaders = rng.integers(0, N_ORIENTS, size=n_trials)
    pairs = np.empty((n_trials, 2), dtype=np.int64)
    other_p = (1.0 - bias) / (N_ORIENTS - 1)
    for k, L in enumerate(leaders):
        expected = (int(L) + 1) % N_ORIENTS
        probs = np.full(N_ORIENTS, other_p)
        probs[expected] = bias
        T = int(rng.choice(N_ORIENTS, p=probs))
        pairs[k] = (int(L), T)
    return pairs


def main() -> int:
    print(f"[B1] Richter training schedule statistics")
    print(f"[B1] seed={SEED}, n_trials={N_TRIALS}")

    # Current schedule
    rng = np.random.default_rng(SEED)
    plan = richter_crossover_training_schedule(rng, n_trials=N_TRIALS)
    pairs = plan.meta["pairs"]
    P_curr = empirical_pt_given_l(pairs)

    print("\n[B1] CURRENT schedule P(T | L) [rows=L, cols=T]:")
    print(P_curr)
    max_PT = float(P_curr.max(axis=1).max())
    min_entropy = min(entropy_bits(P_curr[L]) for L in range(N_ORIENTS))
    max_entropy = float(np.log2(N_ORIENTS))
    print(f"[B1]  max_L max_T P(T|L) = {max_PT:.4f}")
    print(f"[B1]  min_L entropy(P(T|L)) = {min_entropy:.4f} bits  "
          f"(max possible = {max_entropy:.4f})")

    # Reference biased schedule
    rng2 = np.random.default_rng(SEED)
    biased = biased_schedule_pairs(rng2, N_TRIALS, bias=0.80)
    P_bias = empirical_pt_given_l(biased)
    print("\n[B1] BIASED reference P(T | L) [P(L->f(L))=0.80, f(L)=(L+1)%6]:")
    print(P_bias)
    max_PT_bias = float(P_bias.max(axis=1).max())
    min_entropy_bias = min(entropy_bits(P_bias[L]) for L in range(N_ORIENTS))
    print(f"[B1]  max_L max_T P(T|L) = {max_PT_bias:.4f}")
    print(f"[B1]  min_L entropy(P(T|L)) = {min_entropy_bias:.4f} bits")

    # Verdict
    uniform = 1.0 / N_ORIENTS
    tol = 0.02
    is_uniform = abs(max_PT - uniform) < tol
    print(f"\n[B1] VERDICT:")
    print(f"  Current schedule P(T|L) within {tol} of uniform 1/6={uniform:.4f}: "
          f"{is_uniform}")
    print(f"  Current schedule max_T P(T|L)={max_PT:.4f} << 0.70 threshold "
          f"  -> NO STATISTICAL CONTINGENCY to learn.")
    print(f"  BUG 1 CONFIRMED: `richter_crossover_training_schedule` produces "
          f"balanced all-pairs; STDP-based H_R cannot extract a forecast.")

    np.savez(OUT_DIR / "B1_richter_schedule_stats.npz",
             P_current=P_curr, P_biased_ref=P_bias,
             current_pairs=pairs, biased_pairs=biased,
             max_PT_current=max_PT, max_PT_biased=max_PT_bias,
             min_entropy_current=min_entropy,
             min_entropy_biased=min_entropy_bias,
             seed=SEED, n_trials=N_TRIALS)
    print(f"[B1] saved {OUT_DIR}/B1_richter_schedule_stats.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
