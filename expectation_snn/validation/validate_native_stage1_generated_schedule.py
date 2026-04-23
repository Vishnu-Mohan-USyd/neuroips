"""Validate small generated Stage-1 schedule through native trainer boundary."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.native import backend_info
from expectation_snn.cuda_sim.train_stage1_native import (
    N_CTX_PRED_SYN,
    N_H_EE_SYN,
    build_small_generated_stage1_schedule,
    write_generated_schedule_stage1_checkpoint,
)


SEED = 42
TOL = 1e-10


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _assert_schedule_mapping(schedule: dict) -> None:
    n_trials = int(schedule["n_trials"])
    pairs = np.asarray(schedule["pairs"], dtype=np.int32)
    leader = np.asarray(schedule["leader_idx"], dtype=np.int32)
    trailer = np.asarray(schedule["trailer_idx"], dtype=np.int32)
    expected = np.asarray(schedule["expected_trailer_idx"], dtype=np.int32)
    is_expected = np.asarray(schedule["is_expected"], dtype=np.bool_)
    leader_cells = np.asarray(schedule["leader_pre_cells"], dtype=np.int32)
    trailer_cells = np.asarray(schedule["trailer_post_cells"], dtype=np.int32)
    expected_cells = np.asarray(
        schedule["expected_trailer_post_cells"], dtype=np.int32,
    )
    assert pairs.shape == (n_trials, 2)
    assert leader.shape == (n_trials,)
    assert trailer.shape == (n_trials,)
    assert expected.shape == (n_trials,)
    assert is_expected.shape == (n_trials,)
    assert leader_cells.shape == (n_trials,)
    assert trailer_cells.shape == (n_trials,)
    assert expected_cells.shape == (n_trials,)
    assert np.array_equal(pairs[:, 0], leader)
    assert np.array_equal(pairs[:, 1], trailer)
    assert np.all(leader != trailer)
    assert np.all((leader_cells >= 0) & (leader_cells < 192))
    assert np.all((trailer_cells >= 0) & (trailer_cells < 192))
    assert np.array_equal(leader_cells // 32, leader)
    assert np.array_equal(trailer_cells // 32, trailer)
    assert np.array_equal(expected_cells // 32, expected)


def _assert_native_result(result: dict, schedule: dict) -> None:
    n_trials = int(schedule["n_trials"])
    assert int(result["schedule_variant"]) == -1
    assert int(result["n_trials"]) == n_trials
    assert int(result["n_syn"]) == N_CTX_PRED_SYN
    assert int(result["h_ee_n_syn"]) == N_H_EE_SYN
    assert np.asarray(result["gate_steps"], dtype=np.int32).shape == (n_trials,)
    assert np.array_equal(
        np.asarray(result["trial_leader_pre_cells"], dtype=np.int32),
        np.asarray(schedule["leader_pre_cells"], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(result["trial_trailer_post_cells"], dtype=np.int32),
        np.asarray(schedule["trailer_post_cells"], dtype=np.int32),
    )
    assert result["event_counts"] == {
        "hctx_pre.leader": 2 * n_trials,
        "hctx_pre.trailer": 0,
        "hctx_pre.iti": 0,
        "hctx_pre.outside": 0,
        "hpred_post.leader": 0,
        "hpred_post.trailer": 2 * n_trials,
        "hpred_post.iti": 0,
        "hpred_post.outside": 0,
    }
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
    gate_dw_sum = np.asarray(result["cpu_gate_dw_sum"], dtype=np.float64)
    assert gate_dw_sum.shape == (n_trials,)
    assert np.any(np.abs(gate_dw_sum) > 0.0)


def _assert_checkpoint(path: Path, result: dict, schedule: dict) -> None:
    with np.load(path, allow_pickle=False) as data:
        assert data["ctx_ee_w_final"].shape == (N_H_EE_SYN,)
        assert data["pred_ee_w_final"].shape == (N_H_EE_SYN,)
        assert data["W_ctx_pred_final"].shape == (N_CTX_PRED_SYN,)
        assert int(data["n_trials"]) == int(schedule["n_trials"])
        assert bool(data["passed"]) is False
        assert bool(data["native_placeholder_h_recurrent_arrays"]) is True
        assert np.array_equal(data["leader_idx"], schedule["leader_idx"])
        assert np.array_equal(data["trailer_idx"], schedule["trailer_idx"])
        assert np.array_equal(
            data["expected_trailer_idx"],
            schedule["expected_trailer_idx"],
        )
        assert np.array_equal(data["is_expected"], schedule["is_expected"])
        assert np.array_equal(
            data["native_trial_leader_pre_cells"],
            schedule["leader_pre_cells"],
        )
        assert np.array_equal(
            data["native_trial_trailer_post_cells"],
            schedule["trailer_post_cells"],
        )
        assert np.array_equal(
            data["gate_dw_sum"],
            np.asarray(result["cpu_gate_dw_sum"], dtype=np.float64),
        )


def main() -> int:
    schedule = build_small_generated_stage1_schedule(seed=SEED, n_trials=12)
    repeat_schedule = build_small_generated_stage1_schedule(seed=SEED, n_trials=12)
    shifted_schedule = build_small_generated_stage1_schedule(seed=SEED + 1, n_trials=12)
    _assert_schedule_mapping(schedule)
    _assert_schedule_mapping(repeat_schedule)
    _assert_schedule_mapping(shifted_schedule)
    assert np.array_equal(schedule["pairs"], repeat_schedule["pairs"])
    assert not np.array_equal(schedule["pairs"], shifted_schedule["pairs"])

    with tempfile.TemporaryDirectory(prefix="native_stage1_generated_") as tmp_s:
        out_path = Path(tmp_s) / "stage_1_ctx_pred_seed42_generated.npz"
        checkpoint, result, written_schedule = write_generated_schedule_stage1_checkpoint(
            out_path,
            seed=SEED,
            n_trials=12,
        )
        _assert_schedule_mapping(written_schedule)
        _assert_native_result(result, written_schedule)
        _assert_checkpoint(checkpoint, result, written_schedule)

        print(
            "validate_native_stage1_generated_schedule: PASS",
            f"backend_info={backend_info()}",
            f"checkpoint={checkpoint}",
            f"n_trials={result['n_trials']}",
            f"pairs={np.asarray(written_schedule['pairs']).tolist()}",
            f"gate_dw_sum={np.asarray(result['cpu_gate_dw_sum']).tolist()}",
            f"max_err={_max_error(result):.3e}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
