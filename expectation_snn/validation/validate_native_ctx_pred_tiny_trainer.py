"""Validate controlled multi-trial native Stage1 ctx_pred trainer primitive."""
from __future__ import annotations

import numpy as np

from expectation_snn.cuda_sim.native import (
    backend_info,
    run_ctx_pred_tiny_trainer_test,
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
        "ctx_ee_w_final",
        "pred_ee_w_final",
        "elig_after_training",
        "xpre_after_training",
        "xpost_after_training",
        "row_sums",
        "gate_w_before",
        "gate_w_after",
        "gate_dw_sum",
        "gate_elig_mean",
        "gate_elig_max",
        "gate_row_sum_max",
    ):
        cpu = np.asarray(result[f"cpu_{key}"], dtype=np.float64)
        cuda = np.asarray(result[f"cuda_{key}"], dtype=np.float64)
        assert np.allclose(cpu, cuda, atol=TOL, rtol=0.0), key
    assert np.array_equal(
        np.asarray(result["cpu_gate_n_capped"], dtype=np.int32),
        np.asarray(result["cuda_gate_n_capped"], dtype=np.int32),
    )


def _assert_repeat_deterministic(a: dict, b: dict) -> None:
    assert int(a["seed"]) == int(b["seed"]) == 42
    assert int(a["schedule_variant"]) == int(b["schedule_variant"]) == 0
    for key in (
        "initial_w_ctx_pred",
        "cpu_w_ctx_pred_final",
        "cuda_w_ctx_pred_final",
        "cpu_gate_w_before",
        "cpu_gate_w_after",
        "cpu_gate_dw_sum",
        "cpu_gate_elig_mean",
        "cpu_gate_elig_max",
        "cpu_gate_row_sum_max",
        "cpu_gate_n_capped",
    ):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


def _assert_common_schema(result: dict) -> None:
    assert int(result["n_trials"]) == 5
    assert int(result["n_pre"]) == 192
    assert int(result["n_post"]) == 192
    assert int(result["n_syn"]) == 192 * 192
    assert int(result["h_ee_n_syn"]) == 192 * 191
    assert int(result["trial_steps"]) == 220
    assert int(result["n_steps"]) == 5 * 220
    assert np.isclose(float(result["dt_ms"]), 0.1, atol=0.0, rtol=0.0)
    assert result["phase_steps"] == {
        "leader_start_step": 0,
        "leader_end_step": 80,
        "trailer_start_step": 80,
        "trailer_end_step": 180,
        "iti_start_step": 180,
        "iti_end_step": 220,
    }
    assert np.asarray(result["gate_steps"], dtype=np.int32).tolist() == [
        180,
        400,
        620,
        840,
        1060,
    ]
    assert result["event_counts"] == {
        "hctx_pre.leader": 10,
        "hctx_pre.trailer": 0,
        "hctx_pre.iti": 0,
        "hctx_pre.outside": 0,
        "hpred_post.leader": 0,
        "hpred_post.trailer": 10,
        "hpred_post.iti": 0,
        "hpred_post.outside": 0,
    }


def main() -> int:
    result = run_ctx_pred_tiny_trainer_test(seed=42, schedule_variant=0)
    repeat = run_ctx_pred_tiny_trainer_test(seed=42, schedule_variant=0)
    shifted = run_ctx_pred_tiny_trainer_test(seed=42, schedule_variant=1)

    _assert_common_schema(result)
    _assert_common_schema(shifted)
    assert int(result["schedule_variant"]) == 0
    assert int(shifted["schedule_variant"]) == 1
    assert np.asarray(result["trial_leader_pre_cells"], dtype=np.int32).tolist() == [
        20,
        21,
        22,
        23,
        24,
    ]
    assert np.asarray(shifted["trial_leader_pre_cells"], dtype=np.int32).tolist() == [
        90,
        91,
        92,
        93,
        94,
    ]

    _assert_cpu_cuda_match(result)
    _assert_cpu_cuda_match(shifted)
    _assert_repeat_deterministic(result, repeat)

    initial = np.asarray(result["initial_w_ctx_pred"], dtype=np.float64)
    final = np.asarray(result["cpu_w_ctx_pred_final"], dtype=np.float64)
    shifted_final = np.asarray(shifted["cpu_w_ctx_pred_final"], dtype=np.float64)
    ctx_ee = np.asarray(result["cpu_ctx_ee_w_final"], dtype=np.float64)
    pred_ee = np.asarray(result["cpu_pred_ee_w_final"], dtype=np.float64)
    row_sums = np.asarray(result["cpu_row_sums"], dtype=np.float64)
    elig_after = np.asarray(result["cpu_elig_after_training"], dtype=np.float64)
    gate_w_before = np.asarray(result["cpu_gate_w_before"], dtype=np.float64)
    gate_w_after = np.asarray(result["cpu_gate_w_after"], dtype=np.float64)
    gate_dw_sum = np.asarray(result["cpu_gate_dw_sum"], dtype=np.float64)
    gate_elig_mean = np.asarray(result["cpu_gate_elig_mean"], dtype=np.float64)
    gate_elig_max = np.asarray(result["cpu_gate_elig_max"], dtype=np.float64)
    gate_n_capped = np.asarray(result["cpu_gate_n_capped"], dtype=np.int32)

    assert initial.shape == (36864,)
    assert final.shape == (36864,)
    assert ctx_ee.shape == (36672,)
    assert pred_ee.shape == (36672,)
    assert gate_w_before.shape == (5,)
    assert gate_w_after.shape == (5,)
    assert gate_dw_sum.shape == (5,)
    assert gate_elig_mean.shape == (5,)
    assert gate_elig_max.shape == (5,)
    assert gate_n_capped.shape == (5,)
    assert np.all(gate_elig_max > 0.0)
    assert np.all(gate_elig_mean >= 0.0)
    assert gate_n_capped[0] == 1
    assert int(gate_n_capped.sum()) >= 1

    n_post = int(result["n_post"])
    primary_idx = _edge(20, 30, n_post)
    shifted_primary_idx = _edge(90, 30, n_post)
    silent_idx = _edge(5, 6, n_post)
    primary_delta = final[primary_idx] - initial[primary_idx]
    shifted_delta_same_edge = shifted_final[primary_idx] - initial[primary_idx]
    shifted_primary_delta = shifted_final[shifted_primary_idx] - initial[shifted_primary_idx]
    silent_factor = (1.0 - GAMMA * DT_TRIAL_S) ** int(result["n_trials"])
    expected_silent = W_TARGET + (initial[silent_idx] - W_TARGET) * silent_factor
    silent_delta = final[silent_idx] - initial[silent_idx]

    assert primary_delta > silent_delta
    assert shifted_primary_delta > shifted_delta_same_edge
    assert primary_delta > shifted_delta_same_edge
    assert not np.array_equal(final, shifted_final)
    assert np.isclose(final[silent_idx], expected_silent, atol=TOL, rtol=0.0)
    assert np.all(final >= -TOL)
    assert np.all(final <= 1.0 + TOL)
    assert float(row_sums.max()) <= 3.0 + 5e-12
    assert np.allclose(elig_after, 0.0, atol=TOL, rtol=0.0)

    print(
        "validate_native_ctx_pred_tiny_trainer: PASS",
        f"backend_info={backend_info()}",
        f"n_trials={result['n_trials']}",
        f"gate_steps={np.asarray(result['gate_steps']).tolist()}",
        f"gate_n_capped={gate_n_capped.tolist()}",
        f"gate_dw_sum={gate_dw_sum.tolist()}",
        f"primary_delta={primary_delta:.6e}",
        f"shifted_same_edge_delta={shifted_delta_same_edge:.6e}",
        f"shifted_primary_delta={shifted_primary_delta:.6e}",
        f"silent_delta={silent_delta:.6e}",
        f"max_err={_max_error(result):.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
