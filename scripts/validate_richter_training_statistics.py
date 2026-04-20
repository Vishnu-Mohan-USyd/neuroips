"""Sprint 5e-Diag B4a: reusable Richter schedule validator.

Given any TrialPlan from `richter_crossover_training_schedule` (or any
schedule exposing `meta['pairs']`), asserts:
  max_L max_T P(T | L) >= THRESHOLD_MAX_PT  (reviewer: 0.70)

This validator MUST fail on the current balanced all-pairs schedule
(max_T P(T|L) = 1/6 ≈ 0.167) and PASS on the reviewer's biased
deranged-permutation reference.

Scope: no retraining, no Brian2 — pure schedule introspection.
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

THRESHOLD_MAX_PT = 0.70
ENTROPY_MARGIN_BITS = 0.5  # max entropy log2(6)=2.585; biased should be < ~2.08
N_ORIENTS = 6
SEED = 42
N_TRIALS = 360


def compute_pt_given_l(pairs: np.ndarray, n: int = N_ORIENTS):
    counts = np.zeros((n, n), dtype=np.int64)
    for li, ti in pairs:
        counts[int(li), int(ti)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    return counts / row_sum


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def validate(pairs: np.ndarray, label: str, expect_pass: bool) -> bool:
    P = compute_pt_given_l(pairs)
    max_PT = float(P.max(axis=1).max())
    min_ent = min(entropy_bits(P[L]) for L in range(N_ORIENTS))
    max_ent = float(np.log2(N_ORIENTS))
    passed_max = max_PT >= THRESHOLD_MAX_PT
    passed_ent = min_ent <= (max_ent - ENTROPY_MARGIN_BITS)
    passed = passed_max and passed_ent
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}")
    print(f"       max_T P(T|L) = {max_PT:.4f}   (threshold >= {THRESHOLD_MAX_PT})")
    print(f"       min_L entropy = {min_ent:.4f} bits  "
          f"(threshold <= {max_ent - ENTROPY_MARGIN_BITS:.4f})")
    if passed != expect_pass:
        print(f"       UNEXPECTED: expected pass={expect_pass}, got {passed}")
    return passed


def biased_deranged_perm_pairs(rng: np.random.Generator, n: int,
                                bias: float = 0.80) -> np.ndarray:
    leaders = rng.integers(0, N_ORIENTS, size=n)
    other_p = (1.0 - bias) / (N_ORIENTS - 1)
    pairs = np.empty((n, 2), dtype=np.int64)
    for k, L in enumerate(leaders):
        expected = (int(L) + 1) % N_ORIENTS
        probs = np.full(N_ORIENTS, other_p)
        probs[expected] = bias
        pairs[k] = (int(L), int(rng.choice(N_ORIENTS, p=probs)))
    return pairs


def main() -> int:
    rng_a = np.random.default_rng(SEED)
    current = richter_crossover_training_schedule(rng_a, n_trials=N_TRIALS)
    rng_b = np.random.default_rng(SEED)
    biased = biased_deranged_perm_pairs(rng_b, N_TRIALS, bias=0.80)

    print("=== validate_richter_training_statistics ===")
    print(f"threshold: max_T P(T|L) >= {THRESHOLD_MAX_PT}   "
          f"min_L entropy <= log2({N_ORIENTS}) - {ENTROPY_MARGIN_BITS}\n")
    pass_curr = validate(current.meta["pairs"],
                         "current richter_crossover_training_schedule",
                         expect_pass=False)
    print()
    pass_bias = validate(biased,
                         "biased deranged permutation (reference design)",
                         expect_pass=True)

    exit_code = 0
    if pass_curr:
        print("\nERROR: current schedule UNEXPECTEDLY PASSED — review the code.")
        exit_code = 1
    if not pass_bias:
        print("\nERROR: reference biased schedule UNEXPECTEDLY FAILED.")
        exit_code = 1
    if exit_code == 0:
        print("\n[B4a] Validator behaves as specified:")
        print("  current schedule FAILS (no contingency) — as expected.")
        print("  biased reference PASSES (has contingency) — as expected.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
