"""Validate bounded GPU-native n=72 Stage-1 ctx_pred checkpoint generation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.train_stage1_native import (
    DEFAULT_NATIVE_N72_CHECKPOINT,
    N_CTX_PRED_SYN,
    N_H_E,
    N_H_EE_SYN,
    sha256_file,
    stable_npz_content_hash,
    write_native_stage1_n72_checkpoint,
)


SEED = 42
N_TRIALS = 72
TOL = 1e-10


def _decode_bytes(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _assert_checkpoint_schema(path: Path, schedule: dict, result: dict) -> None:
    assert path.is_file(), path
    with np.load(path, allow_pickle=False) as data:
        required = {
            "ctx_ee_w_final",
            "pred_ee_w_final",
            "W_ctx_pred_init",
            "W_ctx_pred_final",
            "native_schedule_pairs",
            "native_schedule_pairs_sha256",
            "gate_step",
            "gate_w_before",
            "gate_w_after",
            "gate_dw_sum",
            "gate_elig_mean",
            "gate_elig_max",
            "gate_n_capped",
            "gate_row_sum_max",
            "row_sum_final",
            "row_cap_limit",
            "row_cap_ok",
            "finite_ok",
            "max_cpu_cuda_error",
            "W_ctx_pred_delta_abs_sum",
            "native_wall_seconds",
            "native_schedule_wall_seconds",
            "native_train_wall_seconds",
            "native_gate_eval_wall_seconds",
            "native_checkpoint_write_wall_seconds",
            "native_total_wall_seconds",
            "native_backend_info",
            "native_stable_content_sha256",
            "native_gate_metrics_schema_version",
            "native_gate_metrics_provisional",
            "native_gate_metrics_all_pass",
            "native_gate_thresholds_all_pass",
            "native_gate_metrics_source",
            "native_gate_h_dynamics_cpu_cuda_max_abs_error",
            "h_gate_n_trials",
            "h_gate_n_steps_per_trial",
            "h_gate_leader_channels",
            "h_gate_trailer_channels",
            "h_gate_ctx_persistence_ms_by_trial",
            "h_gate_pred_pretrailer_target_counts",
            "h_context_persistence_ms",
            "h_context_persistence_pass",
            "h_prediction_pretrailer_forecast_probability",
            "h_prediction_pretrailer_forecast_pass",
            "h_prediction_pretrailer_forecast_trial_count",
            "h_prediction_pretrailer_target_spikes",
            "h_prediction_pretrailer_start_step",
            "h_prediction_pretrailer_end_step",
            "ctx_pred_gate_drive_amp_pA",
            "h_context_population_rate_hz",
            "h_prediction_population_rate_hz",
            "h_context_inh_population_rate_hz",
            "h_prediction_inh_population_rate_hz",
            "h_context_max_native_h_rate_hz",
            "h_prediction_max_native_h_rate_hz",
            "h_context_max_channel_rate_hz",
            "h_prediction_max_channel_rate_hz",
            "no_runaway_max_rate_hz",
            "no_runaway_population_max_rate_hz",
            "no_runaway_max_cell_rate_hz",
            "no_runaway_max_channel_rate_hz",
            "no_runaway_pass",
            "passed",
            "native_scientific_stage1_passed",
            "native_placeholder_h_recurrent_arrays",
        }
        missing = sorted(required.difference(data.files))
        assert not missing, missing

        assert int(data["seed"]) == SEED
        assert int(data["n_trials"]) == N_TRIALS
        assert data["ctx_ee_w_final"].shape == (N_H_EE_SYN,)
        assert data["pred_ee_w_final"].shape == (N_H_EE_SYN,)
        assert data["W_ctx_pred_init"].shape == (N_CTX_PRED_SYN,)
        assert data["W_ctx_pred_final"].shape == (N_CTX_PRED_SYN,)
        assert data["native_schedule_pairs"].shape == (N_TRIALS, 2)
        assert data["gate_step"].shape == (N_TRIALS,)
        assert data["gate_dw_sum"].shape == (N_TRIALS,)
        assert data["gate_elig_mean"].shape == (N_TRIALS,)
        assert data["gate_elig_max"].shape == (N_TRIALS,)
        assert data["gate_n_capped"].shape == (N_TRIALS,)
        assert data["gate_row_sum_max"].shape == (N_TRIALS,)
        assert data["row_sum_final"].shape == (N_H_E,)
        assert np.array_equal(data["native_schedule_pairs"], schedule["pairs"])
        assert _decode_bytes(data["native_schedule_pairs_sha256"]) == (
            schedule["pairs_sha256"]
        )
        assert np.array_equal(data["leader_idx"], schedule["leader_idx"])
        assert np.array_equal(data["trailer_idx"], schedule["trailer_idx"])
        assert np.array_equal(
            data["expected_trailer_idx"],
            schedule["expected_trailer_idx"],
        )
        assert np.array_equal(data["is_expected"], schedule["is_expected"])

        W_init = np.asarray(data["W_ctx_pred_init"], dtype=np.float64)
        W_final = np.asarray(data["W_ctx_pred_final"], dtype=np.float64)
        assert np.allclose(W_init, 0.0, atol=TOL, rtol=0.0)
        row_sums = np.asarray(data["row_sum_final"], dtype=np.float64)
        gate_dw_sum = np.asarray(data["gate_dw_sum"], dtype=np.float64)
        gate_elig_max = np.asarray(data["gate_elig_max"], dtype=np.float64)
        gate_row_sum_max = np.asarray(data["gate_row_sum_max"], dtype=np.float64)

        assert bool(data["passed"]) is False
        assert bool(data["native_scientific_stage1_passed"]) is False
        assert bool(data["native_placeholder_h_recurrent_arrays"]) is True
        assert bool(data["finite_ok"]) is True
        assert bool(data["row_cap_ok"]) is True
        assert np.all(np.isfinite(W_final))
        assert np.all(np.isfinite(row_sums))
        assert np.all(W_final >= -TOL)
        assert np.all(W_final <= 1.0 + TOL)
        assert float(row_sums.max()) <= float(data["row_cap_limit"]) + 5e-12
        assert float(gate_row_sum_max.max()) <= float(data["row_cap_limit"]) + 5e-12
        assert float(data["W_ctx_pred_delta_abs_sum"]) > 1e-6
        assert np.any(np.abs(gate_dw_sum) > 0.0)
        assert np.any(gate_elig_max > 0.0)
        assert float(data["native_wall_seconds"]) > 0.0
        assert float(data["native_schedule_wall_seconds"]) >= 0.0
        assert float(data["native_train_wall_seconds"]) > 0.0
        assert float(data["native_gate_eval_wall_seconds"]) > 0.0
        assert float(data["native_checkpoint_write_wall_seconds"]) > 0.0
        assert float(data["native_total_wall_seconds"]) >= (
            float(data["native_train_wall_seconds"])
            + float(data["native_gate_eval_wall_seconds"])
        )
        assert np.isclose(
            float(data["native_wall_seconds"]),
            float(data["native_total_wall_seconds"]),
        )
        assert "device_count" in _decode_bytes(data["native_backend_info"])
        assert _decode_bytes(data["native_stable_content_sha256"]) == (
            stable_npz_content_hash(path)
        )
        assert bool(data["native_gate_metrics_provisional"]) is True
        assert _decode_bytes(data["native_gate_metrics_source"]) == (
            "native_h_recurrent_dynamics_schedule_eval"
        )
        assert float(data["native_gate_h_dynamics_cpu_cuda_max_abs_error"]) <= TOL
        assert int(data["h_gate_n_trials"]) == N_TRIALS
        assert int(data["h_gate_n_steps_per_trial"]) == 6000
        assert data["h_gate_leader_channels"].shape == (N_TRIALS,)
        assert data["h_gate_trailer_channels"].shape == (N_TRIALS,)
        assert data["h_gate_ctx_persistence_ms_by_trial"].shape == (N_TRIALS,)
        assert data["h_gate_pred_pretrailer_target_counts"].shape == (N_TRIALS,)
        assert int(data["h_prediction_pretrailer_start_step"]) == 2000
        assert int(data["h_prediction_pretrailer_end_step"]) == 3000
        assert np.isclose(float(data["ctx_pred_gate_drive_amp_pA"]), 400.0)
        assert np.isclose(float(data["ctx_pred_drive_amp_ctx_pred_pA"]), 400.0)
        assert np.isclose(float(data["ctx_pred_pred_e_uniform_bias_pA"]), 100.0)
        cfg = json.loads(_decode_bytes(data["ctx_pred_config_json"]))
        assert np.isclose(float(cfg["drive_amp_ctx_pred_pA"]), 400.0)
        assert np.isclose(float(cfg["pred_e_uniform_bias_pA"]), 100.0)
        assert np.isclose(float(cfg["w_init_frac"]), 0.0)
        assert np.array_equal(
            data["h_gate_leader_channels"],
            np.asarray(schedule["leader_pre_cells"], dtype=np.int32) // 16,
        )
        assert np.array_equal(
            data["h_gate_trailer_channels"],
            np.asarray(schedule["trailer_post_cells"], dtype=np.int32) // 16,
        )
        assert np.isclose(
            float(data["h_context_persistence_ms"]),
            float(np.mean(data["h_gate_ctx_persistence_ms_by_trial"])),
        )
        assert np.isclose(
            float(data["h_prediction_pretrailer_forecast_probability"]),
            float(np.count_nonzero(data["h_gate_pred_pretrailer_target_counts"]))
            / float(N_TRIALS),
        )
        h_persist_pass = (
            200.0 <= float(data["h_context_persistence_ms"]) <= 500.0
        )
        forecast_pass = (
            float(data["h_prediction_pretrailer_forecast_probability"]) >= 0.25
        )
        population_rates = (
            float(data["h_context_population_rate_hz"]),
            float(data["h_prediction_population_rate_hz"]),
            float(data["h_context_inh_population_rate_hz"]),
            float(data["h_prediction_inh_population_rate_hz"]),
        )
        no_runaway_pass = float(data["no_runaway_max_rate_hz"]) <= 80.0
        assert np.isclose(float(data["no_runaway_max_rate_hz"]), max(population_rates))
        assert np.isclose(
            float(data["no_runaway_population_max_rate_hz"]),
            float(data["no_runaway_max_rate_hz"]),
        )
        assert float(data["no_runaway_max_cell_rate_hz"]) >= float(
            data["no_runaway_max_rate_hz"]
        )
        assert float(data["no_runaway_max_channel_rate_hz"]) >= float(
            data["no_runaway_max_rate_hz"]
        )
        assert bool(data["h_context_persistence_pass"]) == h_persist_pass
        assert bool(data["h_prediction_pretrailer_forecast_pass"]) == forecast_pass
        assert bool(data["no_runaway_pass"]) == no_runaway_pass
        thresholds_all_pass = bool(h_persist_pass and forecast_pass and no_runaway_pass)
        assert bool(data["native_gate_thresholds_all_pass"]) == thresholds_all_pass
        assert bool(data["native_gate_metrics_all_pass"]) is False
        assert float(data["max_cpu_cuda_error"]) <= TOL
        assert np.array_equal(
            data["gate_dw_sum"],
            np.asarray(result["cpu_gate_dw_sum"], dtype=np.float64),
        )
        assert not np.array_equal(W_init, W_final)


def _assert_same_seed_reproducible(a_path: Path, b_path: Path) -> None:
    a = _load_arrays(a_path)
    b = _load_arrays(b_path)
    deterministic_keys = (
        "native_schedule_pairs",
        "leader_idx",
        "trailer_idx",
        "expected_trailer_idx",
        "is_expected",
        "native_trial_leader_pre_cells",
        "native_trial_trailer_post_cells",
        "gate_step",
        "gate_w_before",
        "gate_w_after",
        "gate_dw_sum",
        "gate_elig_mean",
        "gate_elig_max",
        "gate_n_capped",
        "gate_row_sum_max",
        "row_sum_final",
        "W_ctx_pred_init",
        "W_ctx_pred_final",
        "ctx_ee_w_final",
        "pred_ee_w_final",
        "native_gate_metrics_provisional",
        "native_gate_metrics_all_pass",
        "native_gate_thresholds_all_pass",
        "native_gate_metrics_source",
        "native_gate_h_dynamics_cpu_cuda_max_abs_error",
        "h_context_persistence_ms",
        "h_context_persistence_pass",
        "h_gate_ctx_persistence_ms_by_trial",
        "h_gate_pred_pretrailer_target_counts",
        "h_gate_leader_channels",
        "h_gate_trailer_channels",
        "h_prediction_pretrailer_forecast_probability",
        "h_prediction_pretrailer_forecast_pass",
        "h_prediction_pretrailer_start_step",
        "h_prediction_pretrailer_end_step",
        "ctx_pred_gate_drive_amp_pA",
        "h_context_population_rate_hz",
        "h_prediction_population_rate_hz",
        "h_context_inh_population_rate_hz",
        "h_prediction_inh_population_rate_hz",
        "h_context_max_native_h_rate_hz",
        "h_prediction_max_native_h_rate_hz",
        "h_context_max_channel_rate_hz",
        "h_prediction_max_channel_rate_hz",
        "no_runaway_max_rate_hz",
        "no_runaway_population_max_rate_hz",
        "no_runaway_max_cell_rate_hz",
        "no_runaway_max_channel_rate_hz",
        "no_runaway_pass",
    )
    for key in deterministic_keys:
        assert np.array_equal(a[key], b[key]), key
    assert _decode_bytes(a["native_schedule_pairs_sha256"]) == _decode_bytes(
        b["native_schedule_pairs_sha256"],
    )
    assert _decode_bytes(a["native_stable_content_sha256"]) == _decode_bytes(
        b["native_stable_content_sha256"],
    )
    assert stable_npz_content_hash(a_path) == stable_npz_content_hash(b_path)


def main() -> int:
    path, result, schedule = write_native_stage1_n72_checkpoint(
        DEFAULT_NATIVE_N72_CHECKPOINT,
        seed=SEED,
    )
    _assert_checkpoint_schema(path, schedule, result)

    with tempfile.TemporaryDirectory(prefix="native_stage1_n72_repeat_") as tmp_s:
        repeat_path = Path(tmp_s) / "stage_1_ctx_pred_seed42_native_n72_repeat.npz"
        repeat, repeat_result, repeat_schedule = write_native_stage1_n72_checkpoint(
            repeat_path,
            seed=SEED,
        )
        _assert_checkpoint_schema(repeat, repeat_schedule, repeat_result)
        _assert_same_seed_reproducible(path, repeat)

    with np.load(path, allow_pickle=False) as data:
        gate_dw_sum = np.asarray(data["gate_dw_sum"], dtype=np.float64)
        row_sums = np.asarray(data["row_sum_final"], dtype=np.float64)
        digest = sha256_file(path)
        stable_digest = stable_npz_content_hash(path)
        print(
            "validate_native_stage1_n72_checkpoint: PASS",
            f"path={path}",
            f"sha256={digest}",
            f"stable_content_sha256={stable_digest}",
            f"n_trials={int(data['n_trials'])}",
            f"pairs_sha256={_decode_bytes(data['native_schedule_pairs_sha256'])}",
            f"gate_dw_sum_total={float(gate_dw_sum.sum()):.12e}",
            f"gate_elig_max={float(np.max(data['gate_elig_max'])):.12e}",
            f"gate_n_capped_sum={int(np.sum(data['gate_n_capped']))}",
            f"row_sum_max={float(row_sums.max()):.12f}",
            f"delta_abs_sum={float(data['W_ctx_pred_delta_abs_sum']):.12e}",
            f"W_init_max={float(np.max(data['W_ctx_pred_init'])):.12e}",
            f"native_wall_seconds={float(data['native_wall_seconds']):.6f}",
            f"max_cpu_cuda_error={float(data['max_cpu_cuda_error']):.3e}",
            f"h_context_persistence_ms={float(data['h_context_persistence_ms']):.6f}",
            f"h_context_persistence_pass={bool(data['h_context_persistence_pass'])}",
            f"forecast_probability={float(data['h_prediction_pretrailer_forecast_probability']):.6f}",
            f"forecast_pass={bool(data['h_prediction_pretrailer_forecast_pass'])}",
            f"forecast_target_spikes={int(data['h_prediction_pretrailer_target_spikes'])}",
            f"forecast_trial_count={int(data['h_prediction_pretrailer_forecast_trial_count'])}",
            f"no_runaway_max_rate_hz={float(data['no_runaway_max_rate_hz']):.6f}",
            f"no_runaway_pass={bool(data['no_runaway_pass'])}",
            f"pred_max_rate_hz={float(data['h_prediction_max_native_h_rate_hz']):.6f}",
            f"ctx_pred_gate_drive_amp_pA={float(data['ctx_pred_gate_drive_amp_pA']):.1f}",
            f"native_gate_metrics_all_pass={bool(data['native_gate_metrics_all_pass'])}",
            "passed=False",
            "native_scientific_stage1_passed=False",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
