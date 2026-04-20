"""Functional validation for Sprint 5e Fix B — pre-trailer forecast gate.

Per the per-component functional-validation rule: every new module
(or new function of load-bearing scope) ships with a `validate_*.py`
that proves the contract holds on seed=42.

Contract (from SPRINT_5D_POST_VERDICT_REVIEW.md + dispatch + `check_h_transition_mi`
docstring after Fix B):

  [1] post_trailer_mode_preserves_legacy_mi
        Calling with `gate_window="post_trailer"` (or the legacy default
        for callers that predate Fix B) emits the SAME MI value as the
        original Sprint 5a/5c gate. Regression guard so the legacy
        ablation path stays identical.

  [2] pre_trailer_perfect_forecast_passes
        When `h_argmax == expected_trailer_idx` on every trial, the
        pre_trailer gate returns prob = 1.0, passed = True, and the
        check name is `h_preprobe_forecast_prob`.

  [3] pre_trailer_leader_locked_fails
        When `h_argmax == leader_idx` every trial (the exact pathology
        debugger measured on Sprint 5d D1 seeds 42/43/44 and on the
        H-only forecast unit test B5), the pre-probe prob equals
        P(leader == expected_trailer) ≈ 0 for a proper derangement, so
        the gate FAILs. Memorialises the Bug-2 pathology.

  [4] pre_trailer_chance_fails
        Uniformly-random h_argmax gives prob ≈ 1/6 ≈ 0.167, which is
        BELOW the 0.25 threshold (1.5× chance). Threshold discriminates
        trained-with-prior from chance.

  [5] pre_trailer_at_threshold_boundary
        Synthesising exactly `ceil(n * 0.25)` hits yields passed=True
        (gate is `>=`), one fewer yields False. Confirms the decision
        rule is not accidentally strict-greater-than.

  [6] pre_trailer_requires_expected_trailer_idx
        Calling with `gate_window="pre_trailer"` but no
        `expected_trailer_idx` must raise ValueError with an informative
        message (not a silent pass / silent AttributeError).

  [7] pre_trailer_shape_mismatch_raises
        `h_argmax.shape != expected_trailer_idx.shape` must ValueError.

  [8] unknown_gate_window_raises
        Typos or unsupported modes must ValueError, not silently use a
        default.

  [9] empty_input_returns_fail_no_crash
        Zero-trial pre_trailer call must return a clean FAIL with
        prob=0.0, not divide-by-zero.

 [10] check_name_dispatches_on_mode
        Post-trailer → "h_transition_mi_bits"; pre-trailer →
        "h_preprobe_forecast_prob". Stage-1 report labelling depends on
        this.

 [11] schedule_integration_with_biased_generator
        Given a biased schedule from Fix A, deriving
        `expected_trailer_idx = derangement[leader_idx]` reproduces the
        generator's `plan.meta["expected_trailer_idx"]` exactly, and a
        perfect-forecast network passes on that schedule.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.stimulus import (
    richter_biased_training_schedule,
)
from expectation_snn.validation.stage_1_gate import (
    PREPROBE_FORECAST_MIN_PROB,
    TRANSITION_MI_MIN_BITS,
    _joint_hist_mi,
    check_h_transition_mi,
)


N_ORIENTS = 6
SEED = 42
N_TRIALS = 360


def assay_post_trailer_mode_preserves_legacy_mi() -> None:
    print("[1] post_trailer_mode_preserves_legacy_mi")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    # Simulate a "post-trailer bump tracks trailer" scenario under biased
    # contingency: h_argmax agrees with trailer 80% of the time.
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    hits = rng.random(N_TRIALS) < 0.80
    h_argmax = np.where(hits, derangement[leader],
                        rng.integers(0, N_ORIENTS, size=N_TRIALS))
    # Legacy-metric ground truth: plain MI estimator.
    mi_truth = _joint_hist_mi(leader, h_argmax, N_ORIENTS)
    # New API, post_trailer mode explicit.
    res_post = check_h_transition_mi(
        leader, h_argmax, N_ORIENTS, gate_window="post_trailer",
    )
    assert res_post.name == "h_transition_mi_bits"
    assert abs(res_post.value - mi_truth) < 1e-12
    # Default (unspecified gate_window) must also be post_trailer for
    # backward compatibility with Sprint 5c callers.
    res_default = check_h_transition_mi(leader, h_argmax, N_ORIENTS)
    assert res_default.name == "h_transition_mi_bits"
    assert abs(res_default.value - mi_truth) < 1e-12
    assert res_post.passed == (mi_truth >= TRANSITION_MI_MIN_BITS)
    print(f"    MI legacy={mi_truth:.4f}  MI new_api={res_post.value:.4f}  "
          f"name={res_post.name}  PASS")


def assay_pre_trailer_perfect_forecast_passes() -> None:
    print("[2] pre_trailer_perfect_forecast_passes")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    expected = derangement[leader]
    res = check_h_transition_mi(
        leader, expected.copy(), N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected,
    )
    assert res.name == "h_preprobe_forecast_prob"
    assert res.passed
    assert abs(res.value - 1.0) < 1e-12, res.value
    print(f"    prob={res.value:.3f} threshold={PREPROBE_FORECAST_MIN_PROB}  "
          f"name={res.name}  PASS")


def assay_pre_trailer_leader_locked_fails() -> None:
    print("[3] pre_trailer_leader_locked_fails")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    expected = derangement[leader]
    # h_argmax == leader (debugger's B5 pathology). For a proper
    # derangement, P(leader == expected_trailer) == 0 exactly.
    res = check_h_transition_mi(
        leader, leader.copy(), N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected,
    )
    assert not res.passed
    assert res.value == 0.0, res.value
    print(f"    prob={res.value:.3f} (leader-locked)  FAIL as expected — "
          f"matches debugger B4b seeds 42/43/44  PASS")


def assay_pre_trailer_chance_fails() -> None:
    print("[4] pre_trailer_chance_fails")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    expected = derangement[leader]
    # Uniformly random argmax, independent of leader/expected.
    h_random = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    res = check_h_transition_mi(
        leader, h_random, N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected,
    )
    assert not res.passed, res.summary()
    # Chance = 1/6 ≈ 0.167 < 0.25 threshold.
    chance = 1.0 / N_ORIENTS
    assert res.value < PREPROBE_FORECAST_MIN_PROB, res.value
    # Should be close to chance within sampling noise (±0.05 at N=360).
    assert abs(res.value - chance) <= 0.05, (res.value, chance)
    print(f"    prob={res.value:.3f}  chance={chance:.3f}  "
          f"threshold={PREPROBE_FORECAST_MIN_PROB}  PASS")


def assay_pre_trailer_at_threshold_boundary() -> None:
    print("[5] pre_trailer_at_threshold_boundary")
    rng = np.random.default_rng(SEED)
    n = N_TRIALS
    leader = rng.integers(0, N_ORIENTS, size=n)
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    expected = derangement[leader]
    threshold = PREPROBE_FORECAST_MIN_PROB
    # Exactly `ceil(threshold * n)` hits → prob >= threshold → PASS.
    n_hits_pass = int(math.ceil(threshold * n))
    h_pass = rng.integers(0, N_ORIENTS, size=n)  # filled with random first
    # Force first n_hits_pass to be hits; force the rest to be non-hits.
    h_pass[:n_hits_pass] = expected[:n_hits_pass]
    for k in range(n_hits_pass, n):
        L = int(leader[k])
        fL = int(expected[k])
        # Any orient other than fL is a miss.
        for cand in range(N_ORIENTS):
            if cand != fL:
                h_pass[k] = cand
                break
    # Sanity: exactly n_hits_pass hits.
    n_hits_actual = int((h_pass == expected).sum())
    assert n_hits_actual == n_hits_pass, (n_hits_actual, n_hits_pass)
    res_pass = check_h_transition_mi(
        leader, h_pass, N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected,
    )
    assert res_pass.passed, res_pass.summary()
    assert res_pass.value >= threshold
    # Exactly n_hits_pass - 1 → should FAIL (decision rule is `>=`).
    h_fail = h_pass.copy()
    # Flip one hit to a miss (use index 0 which we set as a hit above).
    L0 = int(leader[0]); fL0 = int(expected[0])
    for cand in range(N_ORIENTS):
        if cand != fL0:
            h_fail[0] = cand
            break
    assert int((h_fail == expected).sum()) == n_hits_pass - 1
    res_fail = check_h_transition_mi(
        leader, h_fail, N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected,
    )
    assert not res_fail.passed, res_fail.summary()
    print(f"    n={n}  hits={n_hits_pass} → prob={res_pass.value:.4f} PASS  "
          f"hits={n_hits_pass-1} → prob={res_fail.value:.4f} FAIL  PASS")


def assay_pre_trailer_requires_expected_trailer_idx() -> None:
    print("[6] pre_trailer_requires_expected_trailer_idx")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    try:
        check_h_transition_mi(
            leader, leader.copy(), N_ORIENTS, gate_window="pre_trailer",
        )
    except ValueError as exc:
        assert "expected_trailer_idx" in str(exc), str(exc)
        print(f"    raised ValueError: {exc}  PASS")
        return
    raise AssertionError(
        "pre_trailer without expected_trailer_idx should raise ValueError"
    )


def assay_pre_trailer_shape_mismatch_raises() -> None:
    print("[7] pre_trailer_shape_mismatch_raises")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    expected = derangement[leader]
    # h_argmax has wrong shape.
    h_short = rng.integers(0, N_ORIENTS, size=N_TRIALS - 1)
    try:
        check_h_transition_mi(
            leader, h_short, N_ORIENTS,
            gate_window="pre_trailer",
            expected_trailer_idx=expected,
        )
    except ValueError as exc:
        assert "shape" in str(exc).lower(), str(exc)
        print(f"    raised ValueError: {exc}  PASS")
        return
    raise AssertionError("shape mismatch should raise ValueError")


def assay_unknown_gate_window_raises() -> None:
    print("[8] unknown_gate_window_raises")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    try:
        check_h_transition_mi(
            leader, leader.copy(), N_ORIENTS, gate_window="post_settle",
        )
    except ValueError as exc:
        assert "gate_window" in str(exc), str(exc)
        print(f"    raised ValueError: {exc}  PASS")
        return
    raise AssertionError("unknown gate_window should raise ValueError")


def assay_empty_input_returns_fail_no_crash() -> None:
    print("[9] empty_input_returns_fail_no_crash")
    empty_int = np.array([], dtype=np.int64)
    res = check_h_transition_mi(
        empty_int, empty_int, N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=empty_int,
    )
    assert not res.passed
    assert res.value == 0.0
    assert "n_trials=0" in res.detail
    print(f"    empty input → {res.summary()}  PASS")


def assay_check_name_dispatches_on_mode() -> None:
    print("[10] check_name_dispatches_on_mode")
    rng = np.random.default_rng(SEED)
    leader = rng.integers(0, N_ORIENTS, size=N_TRIALS)
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    expected = derangement[leader]
    res_post = check_h_transition_mi(
        leader, leader.copy(), N_ORIENTS, gate_window="post_trailer",
    )
    res_pre = check_h_transition_mi(
        leader, expected.copy(), N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected,
    )
    assert res_post.name == "h_transition_mi_bits"
    assert res_pre.name == "h_preprobe_forecast_prob"
    assert "post_trailer" in res_post.detail
    assert "pre_trailer" in res_pre.detail
    print(f"    post_trailer→{res_post.name}  pre_trailer→{res_pre.name}  PASS")


def assay_schedule_integration_with_biased_generator() -> None:
    print("[11] schedule_integration_with_biased_generator")
    rng = np.random.default_rng(SEED)
    plan = richter_biased_training_schedule(
        rng, n_trials=N_TRIALS, p_bias=0.80,
    )
    pairs = plan.meta["pairs"]
    derangement = np.asarray(plan.meta["derangement"], dtype=np.int64)
    # Schedule-emitted expected_trailer_idx must equal derangement[leader]
    # elementwise. This is the contract the Fix B driver relies on.
    expected_sched = np.asarray(plan.meta["expected_trailer_idx"], dtype=np.int64)
    expected_derived = derangement[pairs[:, 0].astype(np.int64)]
    assert np.array_equal(expected_sched, expected_derived), (
        "schedule's expected_trailer_idx must equal derangement[leader]"
    )
    # A perfect-forecast network on this schedule → prob=1.0, PASS.
    leader = pairs[:, 0].astype(np.int64)
    res = check_h_transition_mi(
        leader, expected_sched.copy(), N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected_sched,
    )
    assert res.passed, res.summary()
    assert abs(res.value - 1.0) < 1e-12
    # A network that forecasts f(L) 80% of the time (matches the training
    # contingency) — still well above threshold.
    n = N_TRIALS
    hits = np.random.default_rng(SEED + 1).random(n) < 0.80
    h_80 = np.where(hits, expected_sched, pairs[:, 0].astype(np.int64))
    res_80 = check_h_transition_mi(
        leader, h_80, N_ORIENTS,
        gate_window="pre_trailer",
        expected_trailer_idx=expected_sched,
    )
    assert res_80.passed, res_80.summary()
    assert res_80.value >= 0.70, res_80.value
    print(f"    perfect prob={res.value:.3f} PASS  "
          f"80%-forecast prob={res_80.value:.3f} PASS")


def main() -> int:
    np.random.seed(SEED)
    assays = [
        assay_post_trailer_mode_preserves_legacy_mi,
        assay_pre_trailer_perfect_forecast_passes,
        assay_pre_trailer_leader_locked_fails,
        assay_pre_trailer_chance_fails,
        assay_pre_trailer_at_threshold_boundary,
        assay_pre_trailer_requires_expected_trailer_idx,
        assay_pre_trailer_shape_mismatch_raises,
        assay_unknown_gate_window_raises,
        assay_empty_input_returns_fail_no_crash,
        assay_check_name_dispatches_on_mode,
        assay_schedule_integration_with_biased_generator,
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
    print(f"validate_preprobe_forecast_gate: {n_ok}/{n} PASS")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
