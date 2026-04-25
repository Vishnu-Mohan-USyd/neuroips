"""Validate native Stage-1 ctx_pred weights concentrate on biased target channels."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.train_stage1_native import (
    N_CTX_PRED_SYN,
    N_H_E,
    write_native_stage1_n72_checkpoint,
)


SEED = 42
N_TRIALS = 72
N_ORIENTATIONS = 6
H_E_PER_CHANNEL = 16
H_RICHTER_CHANNEL_STRIDE = 2
TOL = 1e-12


def _channel_cells(orientation_idx: int) -> np.ndarray:
    h_channel = int(orientation_idx) * H_RICHTER_CHANNEL_STRIDE
    start = h_channel * H_E_PER_CHANNEL
    return np.arange(start, start + H_E_PER_CHANNEL, dtype=np.int32)


def _block_scores(weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weights, dtype=np.float64).reshape(N_H_E, N_H_E)
    scores = np.zeros((N_ORIENTATIONS, N_ORIENTATIONS), dtype=np.float64)
    for leader in range(N_ORIENTATIONS):
        pre = _channel_cells(leader)
        for trailer in range(N_ORIENTATIONS):
            post = _channel_cells(trailer)
            scores[leader, trailer] = float(matrix[np.ix_(pre, post)].sum())
    return scores


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_stage1_concentration_") as tmp_s:
        path, result, schedule = write_native_stage1_n72_checkpoint(
            Path(tmp_s) / "stage_1_ctx_pred_seed42_native_n72.npz",
            seed=SEED,
        )
        with np.load(path, allow_pickle=False) as data:
            w_init = np.asarray(data["W_ctx_pred_init"], dtype=np.float64)
            w_final = np.asarray(data["W_ctx_pred_final"], dtype=np.float64)

        assert w_init.shape == (N_CTX_PRED_SYN,)
        assert w_final.shape == (N_CTX_PRED_SYN,)
        assert np.allclose(w_init, 0.0, atol=TOL, rtol=0.0)
        assert int(result["n_trials"]) == N_TRIALS

        pairs = np.asarray(schedule["pairs"], dtype=np.int32)
        expected = np.asarray(schedule["expected_trailer_idx"], dtype=np.int32)
        leader_cells = np.asarray(schedule["leader_pre_cells"], dtype=np.int32)
        trailer_cells = np.asarray(schedule["trailer_post_cells"], dtype=np.int32)
        assert pairs.shape == (N_TRIALS, 2)
        assert np.array_equal((leader_cells // H_E_PER_CHANNEL) // 2, pairs[:, 0])
        assert np.array_equal((trailer_cells // H_E_PER_CHANNEL) // 2, pairs[:, 1])
        assert np.all(((leader_cells // H_E_PER_CHANNEL) % 2) == 0)
        assert np.all(((trailer_cells // H_E_PER_CHANNEL) % 2) == 0)

        delta_scores = _block_scores(w_final - w_init)
        final_scores = _block_scores(w_final)
        expected_by_leader = np.empty((N_ORIENTATIONS,), dtype=np.int32)
        for leader in range(N_ORIENTATIONS):
            mask = pairs[:, 0] == leader
            assert np.any(mask), leader
            vals = np.unique(expected[mask])
            assert vals.shape == (1,), (leader, vals)
            expected_by_leader[leader] = vals[0]

        delta_argmax = np.argmax(delta_scores, axis=1).astype(np.int32)
        final_argmax = np.argmax(final_scores, axis=1).astype(np.int32)
        assert np.array_equal(delta_argmax, expected_by_leader), (
            delta_argmax.tolist(),
            expected_by_leader.tolist(),
            delta_scores.tolist(),
        )
        assert np.array_equal(final_argmax, expected_by_leader), (
            final_argmax.tolist(),
            expected_by_leader.tolist(),
            final_scores.tolist(),
        )

        expected_delta = delta_scores[np.arange(N_ORIENTATIONS), expected_by_leader]
        masked = delta_scores.copy()
        masked[np.arange(N_ORIENTATIONS), expected_by_leader] = -np.inf
        next_best = np.max(masked, axis=1)
        assert np.all(expected_delta > next_best + TOL), (
            expected_delta.tolist(),
            next_best.tolist(),
        )

        target_trial_argmax = int(
            np.count_nonzero(delta_argmax[pairs[:, 0]] == expected)
        )
        assert target_trial_argmax == N_TRIALS, target_trial_argmax

        print(
            "validate_native_stage1_ctx_pred_concentration: PASS",
            f"path={path}",
            f"delta_argmax={delta_argmax.tolist()}",
            f"expected_by_leader={expected_by_leader.tolist()}",
            f"target_trial_argmax={target_trial_argmax}/{N_TRIALS}",
            f"expected_delta={expected_delta.tolist()}",
            f"next_best_delta={next_best.tolist()}",
            f"delta_abs_sum={float(np.abs(w_final - w_init).sum()):.12e}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
