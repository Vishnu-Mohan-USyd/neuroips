"""Validate a bounded native Stage1 ctx_pred training trial slice.

The slice is intentionally controlled and small: H_context emits pre events
only in the leader phase, H_prediction emits teacher/post events only in the
trailer phase, and the ctx_pred gate fires at trailer offset before ITI
quiescence. The synapse matrix uses the real 192 x 192 ctx_pred shape.
"""
from __future__ import annotations

import numpy as np

from expectation_snn.cuda_sim.native import (
    backend_info,
    run_ctx_pred_training_trial_slice_test,
)


TOL = 1e-10
GAMMA = 1e-4
W_TARGET = 0.0075
DT_TRIAL_S = 2.5


def _edge(pre: int, post: int, n_post: int) -> int:
    return int(pre) * int(n_post) + int(post)


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _assert_cpu_cuda_match(result: dict) -> None:
    assert _max_error(result) <= TOL, result["max_abs_error"]
    for key in (
        "w_ctx_pred_final",
        "elig_before_gate",
        "elig_after_iti",
        "xpre_after_iti",
        "xpost_after_iti",
        "row_sums",
    ):
        cpu = np.asarray(result[f"cpu_{key}"], dtype=np.float64)
        cuda = np.asarray(result[f"cuda_{key}"], dtype=np.float64)
        assert np.allclose(cpu, cuda, atol=TOL, rtol=0.0), key
    assert int(result["cpu_n_capped"]) == int(result["cuda_n_capped"])


def _assert_repeat_deterministic(a: dict, b: dict) -> None:
    assert int(a["seed"]) == int(b["seed"]) == 42
    for key in (
        "initial_w_ctx_pred",
        "cpu_w_ctx_pred_final",
        "cuda_w_ctx_pred_final",
        "cpu_elig_before_gate",
        "cuda_elig_before_gate",
        "cpu_row_sums",
        "cuda_row_sums",
    ):
        assert np.array_equal(
            np.asarray(a[key], dtype=np.float64),
            np.asarray(b[key], dtype=np.float64),
        ), key


def main() -> int:
    result = run_ctx_pred_training_trial_slice_test(seed=42)
    repeat = run_ctx_pred_training_trial_slice_test(seed=42)

    assert int(result["n_pre"]) == 192
    assert int(result["n_post"]) == 192
    assert int(result["n_syn"]) == 192 * 192
    assert np.isclose(float(result["dt_ms"]), 0.1, atol=0.0, rtol=0.0)
    assert int(result["gate_step"]) == int(result["phase_steps"]["trailer_end_step"])
    assert result["phase_steps"] == {
        "leader_start_step": 0,
        "leader_end_step": 80,
        "trailer_start_step": 80,
        "trailer_end_step": 180,
        "iti_start_step": 180,
        "iti_end_step": 220,
    }

    hctx_steps = np.asarray(result["hctx_pre_event_steps"], dtype=np.int32)
    hpred_steps = np.asarray(result["hpred_post_event_steps"], dtype=np.int32)
    assert hctx_steps.tolist() == [10, 79]
    assert hpred_steps.tolist() == [80, 179]
    assert np.all(hctx_steps >= result["phase_steps"]["leader_start_step"])
    assert np.all(hctx_steps < result["phase_steps"]["leader_end_step"])
    assert np.all(hpred_steps >= result["phase_steps"]["trailer_start_step"])
    assert np.all(hpred_steps < result["phase_steps"]["trailer_end_step"])
    assert result["event_counts"] == {
        "hctx_pre.leader": 2,
        "hctx_pre.trailer": 0,
        "hctx_pre.iti": 0,
        "hctx_pre.outside": 0,
        "hpred_post.leader": 0,
        "hpred_post.trailer": 2,
        "hpred_post.iti": 0,
        "hpred_post.outside": 0,
    }

    _assert_cpu_cuda_match(result)
    _assert_repeat_deterministic(result, repeat)

    n_post = int(result["n_post"])
    leader_pre = int(result["leader_pre"])
    boundary_pre = int(result["boundary_pre"])
    trailer_post = int(result["trailer_post"])
    late_trailer_post = int(result["late_trailer_post"])
    capped_pre = int(result["capped_pre"])
    silent_idx = _edge(int(result["silent_pre"]), int(result["silent_post"]), n_post)
    leader_target_idx = _edge(leader_pre, trailer_post, n_post)
    boundary_target_idx = _edge(boundary_pre, trailer_post, n_post)
    late_target_idx = _edge(boundary_pre, late_trailer_post, n_post)
    leader_copy_idx = _edge(leader_pre, leader_pre, n_post)

    initial_w = np.asarray(result["initial_w_ctx_pred"], dtype=np.float64)
    final_w = np.asarray(result["cpu_w_ctx_pred_final"], dtype=np.float64)
    elig_before = np.asarray(result["cpu_elig_before_gate"], dtype=np.float64)
    elig_after = np.asarray(result["cpu_elig_after_iti"], dtype=np.float64)
    row_sums = np.asarray(result["cpu_row_sums"], dtype=np.float64)

    assert initial_w.shape == (36864,)
    assert final_w.shape == (36864,)
    assert elig_before.shape == (36864,)

    assert elig_before[leader_target_idx] > 0.0
    assert elig_before[boundary_target_idx] > elig_before[leader_target_idx]
    assert elig_before[late_target_idx] > 0.0
    assert np.isclose(elig_before[leader_copy_idx], 0.0, atol=TOL, rtol=0.0)

    leader_delta = final_w[leader_target_idx] - initial_w[leader_target_idx]
    boundary_delta = final_w[boundary_target_idx] - initial_w[boundary_target_idx]
    late_delta = final_w[late_target_idx] - initial_w[late_target_idx]
    silent_delta = final_w[silent_idx] - initial_w[silent_idx]
    assert leader_delta > silent_delta
    assert boundary_delta > leader_delta
    assert late_delta > silent_delta

    expected_silent = initial_w[silent_idx] - GAMMA * (
        initial_w[silent_idx] - W_TARGET
    ) * DT_TRIAL_S
    assert np.isclose(final_w[silent_idx], expected_silent, atol=TOL, rtol=0.0)
    assert np.isclose(
        final_w[leader_copy_idx],
        initial_w[leader_copy_idx]
        - GAMMA * (initial_w[leader_copy_idx] - W_TARGET) * DT_TRIAL_S,
        atol=TOL,
        rtol=0.0,
    )

    assert int(result["cpu_n_capped"]) == 1
    assert np.isclose(row_sums[capped_pre], 3.0, atol=5e-12, rtol=0.0)
    assert float(row_sums.max()) <= 3.0 + 5e-12
    assert np.all(final_w >= -TOL)
    assert np.all(final_w <= 1.0 + TOL)
    assert np.allclose(elig_after, 0.0, atol=TOL, rtol=0.0)

    print(
        "validate_native_ctx_pred_training_trial_slice: PASS",
        f"backend_info={backend_info()}",
        "phases=leader:[0,80),trailer:[80,180),iti:[180,220)",
        f"gate_step={result['gate_step']}",
        f"event_counts={result['event_counts']}",
        f"leader_delta={leader_delta:.6e}",
        f"boundary_delta={boundary_delta:.6e}",
        f"late_delta={late_delta:.6e}",
        f"silent_delta={silent_delta:.6e}",
        f"capped_row_sum={row_sums[capped_pre]:.12f}",
        f"max_err={_max_error(result):.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
