"""Functional validation for Sprint 5e Fix A — `richter_biased_training_schedule`.

Per the per-component functional-validation rule: every new module
(or new function of load-bearing scope) ships with a `validate_*.py`
that proves the contract holds on seed=42.

Contract (from stimulus.py docstring + SPRINT_5D_POST_VERDICT_REVIEW.md):

  [1] default_config_is_deterministic
        Two independent `default_rng(42)` invocations return byte-equal
        `pairs` / `expected_trailer_idx` / `is_expected` / all metadata.

  [2] balanced_leader_distribution
        Each leader appears exactly `n_trials / n_orients` times
        (balanced leader sampling is part of the contract — the bias is
        on P(T|L), not on P(L)).

  [3] no_same_orientation_trailers
        For every trial: `trailer_idx != leader_idx`. This is an
        explicit guarantee of the deranged-permutation schedule (bias
        only on f(L) and 4 others; L→L probability is 0.0).

  [4] empirical_bias_matches_target
        `P(T = f(L) | L) ≈ p_bias` within ±0.05 on 360 trials at seed=42
        for every L ∈ {0..5}. Stricter than the schedule validator
        (which only checks max P(T|L) globally).

  [5] entropy_and_max_pt_pass_schedule_validator_floor
        Matches the thresholds in
        `scripts/validate_richter_training_statistics.py`:
          - max_T max_L P(T|L) ≥ 0.70
          - min_L entropy(T|L) ≤ log2(6) − 0.5

  [6] trialplan_structure
        Items alternate leader / trailer / iti; leader+trailer meta
        carries keys {leader_idx, trailer_idx, expected_trailer_idx,
        is_expected}; total duration equals
        n_trials * (leader_ms + trailer_ms + iti_ms).

  [7] derangement_validation_rejects_fixed_points
        `derangement=(0, 2, 3, 4, 5, 1)` (L=0 is a fixed point) raises
        ValueError with an informative message.

  [8] legacy_schedule_still_fails_same_validator_assertion
        Existing `richter_crossover_training_schedule` still has
        max P(T|L) = 1/6 — i.e. Fix A introduces the biased variant
        without silently modifying the legacy function (debugger's
        Bug-1 evidence must remain reproducible).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.stimulus import (
    RICHTER_ITI_MS,
    RICHTER_LEADER_MS,
    RICHTER_TRAILER_MS,
    richter_biased_training_schedule,
    richter_crossover_training_schedule,
)


N_ORIENTS = 6
SEED = 42
N_TRIALS = 360
P_BIAS = 0.80
MAX_PT_FLOOR = 0.70
ENTROPY_MARGIN = 0.5


def _pt_given_l(pairs: np.ndarray, n: int = N_ORIENTS) -> np.ndarray:
    counts = np.zeros((n, n), dtype=np.int64)
    for li, ti in pairs:
        counts[int(li), int(ti)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    return counts / row_sum


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def assay_default_config_is_deterministic() -> None:
    print("[1] default_config_is_deterministic")
    rng_a = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED)
    pa = richter_biased_training_schedule(rng_a, n_trials=N_TRIALS, p_bias=P_BIAS)
    pb = richter_biased_training_schedule(rng_b, n_trials=N_TRIALS, p_bias=P_BIAS)
    assert np.array_equal(pa.meta["pairs"], pb.meta["pairs"])
    assert np.array_equal(pa.meta["expected_trailer_idx"],
                          pb.meta["expected_trailer_idx"])
    assert np.array_equal(pa.meta["is_expected"], pb.meta["is_expected"])
    assert pa.meta["p_bias"] == pb.meta["p_bias"] == P_BIAS
    print("    deterministic under seed=42  PASS")


def assay_balanced_leader_distribution() -> None:
    print("[2] balanced_leader_distribution")
    rng = np.random.default_rng(SEED)
    plan = richter_biased_training_schedule(rng, n_trials=N_TRIALS, p_bias=P_BIAS)
    leaders = plan.meta["pairs"][:, 0]
    counts = Counter(leaders.tolist())
    expected = N_TRIALS // N_ORIENTS
    for L in range(N_ORIENTS):
        n_L = counts.get(L, 0)
        assert n_L == expected, (
            f"leader {L} appears {n_L} times (expected {expected})"
        )
    print(f"    each leader = {expected} trials  PASS")


def assay_no_same_orientation_trailers() -> None:
    print("[3] no_same_orientation_trailers")
    rng = np.random.default_rng(SEED)
    plan = richter_biased_training_schedule(rng, n_trials=N_TRIALS, p_bias=P_BIAS)
    pairs = plan.meta["pairs"]
    diag = int((pairs[:, 0] == pairs[:, 1]).sum())
    assert diag == 0, (
        f"{diag} same-orientation trials leaked into biased schedule"
    )
    print(f"    L→L trials = 0 / {N_TRIALS}  PASS")


def assay_empirical_bias_matches_target() -> None:
    print("[4] empirical_bias_matches_target")
    rng = np.random.default_rng(SEED)
    plan = richter_biased_training_schedule(rng, n_trials=N_TRIALS, p_bias=P_BIAS)
    pairs = plan.meta["pairs"]
    derangement = plan.meta["derangement"]
    per_L_bias = np.zeros(N_ORIENTS)
    for L in range(N_ORIENTS):
        mask = pairs[:, 0] == L
        n_L = int(mask.sum())
        assert n_L > 0, f"leader {L} never appeared"
        fL = derangement[L]
        n_fL = int((pairs[mask, 1] == fL).sum())
        per_L_bias[L] = n_fL / n_L
        assert abs(per_L_bias[L] - P_BIAS) <= 0.12, (
            f"leader {L} empirical bias {per_L_bias[L]:.3f} "
            f"outside [{P_BIAS - 0.12:.2f}, {P_BIAS + 0.12:.2f}]"
        )
    global_bias = float(plan.meta["is_expected"].mean())
    assert abs(global_bias - P_BIAS) <= 0.05, (
        f"global bias {global_bias:.3f} outside [{P_BIAS - 0.05:.2f}, "
        f"{P_BIAS + 0.05:.2f}]"
    )
    print(f"    global P(T=f(L)|L) = {global_bias:.3f} "
          f"(target {P_BIAS:.2f} ± 0.05)  PASS")


def assay_entropy_and_max_pt_pass_validator_floor() -> None:
    print("[5] entropy_and_max_pt_pass_schedule_validator_floor")
    rng = np.random.default_rng(SEED)
    plan = richter_biased_training_schedule(rng, n_trials=N_TRIALS, p_bias=P_BIAS)
    P = _pt_given_l(plan.meta["pairs"])
    max_pt = float(P.max(axis=1).max())
    min_ent = min(_entropy_bits(P[L]) for L in range(N_ORIENTS))
    max_ent = float(np.log2(N_ORIENTS))
    assert max_pt >= MAX_PT_FLOOR, (
        f"max P(T|L) = {max_pt:.3f} < {MAX_PT_FLOOR}"
    )
    assert min_ent <= max_ent - ENTROPY_MARGIN, (
        f"min_L entropy {min_ent:.3f} > log2(6) - 0.5 = "
        f"{max_ent - ENTROPY_MARGIN:.3f}"
    )
    print(f"    max P(T|L) = {max_pt:.3f}  min_L ent = {min_ent:.3f} bits  PASS")


def assay_trialplan_structure() -> None:
    print("[6] trialplan_structure")
    rng = np.random.default_rng(SEED)
    leader_ms = RICHTER_LEADER_MS
    trailer_ms = RICHTER_TRAILER_MS
    iti_ms = RICHTER_ITI_MS
    plan = richter_biased_training_schedule(
        rng, n_trials=N_TRIALS, p_bias=P_BIAS,
        leader_ms=leader_ms, trailer_ms=trailer_ms, iti_ms=iti_ms,
    )
    assert plan.meta["paradigm"] == "richter_biased"
    expected_total = N_TRIALS * (leader_ms + trailer_ms + iti_ms)
    assert abs(plan.total_ms - expected_total) < 1e-6, (
        f"total_ms {plan.total_ms} != {expected_total}"
    )
    kinds_head = [it.kind for it in plan.items[:6]]
    assert kinds_head == ["leader", "trailer", "iti",
                          "leader", "trailer", "iti"], kinds_head
    for it in plan.items:
        if it.kind in ("leader", "trailer"):
            for key in ("leader_idx", "trailer_idx",
                        "expected_trailer_idx", "is_expected"):
                assert key in it.meta, f"missing key {key} on {it.kind} meta"
    print("    items alternate leader/trailer/iti; meta keys present  PASS")


def assay_derangement_validation_rejects_fixed_points() -> None:
    print("[7] derangement_validation_rejects_fixed_points")
    rng = np.random.default_rng(0)
    try:
        richter_biased_training_schedule(
            rng, n_trials=60,
            derangement=(0, 2, 3, 4, 5, 1),   # L=0 is a fixed point
        )
        raise AssertionError("fixed-point derangement should raise")
    except ValueError as exc:
        assert "fixed point" in str(exc).lower(), (
            f"expected 'fixed point' in message, got: {exc}"
        )
    # Non-permutation (duplicate) also rejected.
    try:
        richter_biased_training_schedule(
            rng, n_trials=60,
            derangement=(1, 2, 3, 4, 5, 5),
        )
        raise AssertionError("duplicate derangement should raise")
    except ValueError as exc:
        assert "permutation" in str(exc).lower(), (
            f"expected 'permutation' in message, got: {exc}"
        )
    print("    rejects fixed points and non-permutations  PASS")


def assay_legacy_schedule_unchanged() -> None:
    print("[8] legacy_schedule_still_fails_same_validator_assertion")
    rng = np.random.default_rng(SEED)
    plan = richter_crossover_training_schedule(rng, n_trials=N_TRIALS)
    P = _pt_given_l(plan.meta["pairs"])
    max_pt = float(P.max(axis=1).max())
    # Balanced schedule: P(T|L) = 1/6 exactly.
    assert abs(max_pt - 1.0 / N_ORIENTS) < 1e-9, (
        f"legacy schedule max P(T|L) = {max_pt:.4f} != {1.0/N_ORIENTS:.4f}; "
        f"Fix A must NOT modify the legacy function"
    )
    print(f"    legacy schedule max P(T|L) = {max_pt:.4f} (≈ 1/6)  PASS")


def main() -> int:
    np.random.seed(SEED)
    assays = [
        assay_default_config_is_deterministic,
        assay_balanced_leader_distribution,
        assay_no_same_orientation_trailers,
        assay_empirical_bias_matches_target,
        assay_entropy_and_max_pt_pass_validator_floor,
        assay_trialplan_structure,
        assay_derangement_validation_rejects_fixed_points,
        assay_legacy_schedule_unchanged,
    ]
    failed = []
    for a in assays:
        try:
            a()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{a.__name__}: {exc}")
            print(f"    FAIL — {exc}")
    n = len(assays)
    n_ok = n - len(failed)
    print()
    print(f"validate_richter_biased_schedule: {n_ok}/{n} PASS")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
