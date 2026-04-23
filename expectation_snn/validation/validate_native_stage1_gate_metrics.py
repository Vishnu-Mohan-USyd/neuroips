"""Validate native Stage-1 gate metrics and stable content hashing."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.native import run_h_ring_dynamics_test
from expectation_snn.cuda_sim.train_stage1_native import (
    H_CONTEXT_PERSISTENCE_MAX_MS,
    H_CONTEXT_PERSISTENCE_MIN_MS,
    H_PRED_FORECAST_PROB_MIN,
    NO_RUNAWAY_MAX_RATE_HZ,
    N_CTX_PRED_SYN,
    stable_npz_content_hash,
    write_native_stage1_n72_checkpoint,
)


SEED = 42
TOL = 1e-12


def _decode_bytes(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_json(value: object) -> dict:
    return json.loads(_decode_bytes(value))


def _assert_gate_logic(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "native_gate_metrics_schema_version",
            "native_gate_metrics_provisional",
            "native_gate_metrics_all_pass",
            "native_gate_thresholds_all_pass",
            "native_gate_metrics_json",
            "native_gate_metrics_source",
            "native_gate_h_dynamics_cpu_cuda_max_abs_error",
            "h_gate_n_trials",
            "h_gate_n_steps_per_trial",
            "h_gate_leader_channels",
            "h_gate_trailer_channels",
            "h_gate_ctx_persistence_ms_by_trial",
            "h_gate_pred_pretrailer_target_counts",
            "h_context_persistence_ms",
            "h_context_persistence_min_ms",
            "h_context_persistence_max_ms",
            "h_context_persistence_pass",
            "h_prediction_pretrailer_forecast_probability",
            "h_prediction_pretrailer_forecast_threshold",
            "h_prediction_pretrailer_forecast_pass",
            "h_prediction_pretrailer_start_step",
            "h_prediction_pretrailer_end_step",
            "ctx_pred_gate_drive_amp_pA",
            "no_runaway_max_rate_hz",
            "no_runaway_threshold_hz",
            "no_runaway_pass",
            "passed",
            "native_scientific_stage1_passed",
            "native_stable_content_sha256",
        }
        missing = sorted(required.difference(data.files))
        assert not missing, missing

        metrics = _decode_json(data["native_gate_metrics_json"])
        assert int(data["native_gate_metrics_schema_version"]) == 1
        assert metrics["schema_version"] == 1
        assert bool(data["native_gate_metrics_provisional"]) is True
        assert metrics["provisional"] is True
        assert metrics["metric_source"] == "native_h_recurrent_dynamics_schedule_eval"
        assert _decode_bytes(data["native_gate_metrics_source"]) == metrics["metric_source"]
        assert "native H recurrent/inhibitory dynamics" in metrics["reason"]
        assert float(data["native_gate_h_dynamics_cpu_cuda_max_abs_error"]) <= 1e-8
        assert int(data["h_gate_n_trials"]) == 72
        assert int(data["h_gate_n_steps_per_trial"]) == 6000
        assert data["h_gate_leader_channels"].shape == (72,)
        assert data["h_gate_trailer_channels"].shape == (72,)
        assert data["h_gate_ctx_persistence_ms_by_trial"].shape == (72,)
        assert data["h_gate_pred_pretrailer_target_counts"].shape == (72,)
        assert np.all(data["h_gate_ctx_persistence_ms_by_trial"] >= 0.0)
        assert np.all(data["h_gate_pred_pretrailer_target_counts"] >= 0)
        assert int(data["h_prediction_pretrailer_start_step"]) == 2000
        assert int(data["h_prediction_pretrailer_end_step"]) == 3000
        assert np.isclose(float(data["ctx_pred_gate_drive_amp_pA"]), 400.0)
        assert metrics["h_prediction_pretrailer_start_step"] == 2000
        assert metrics["h_prediction_pretrailer_end_step"] == 3000
        assert np.isclose(metrics["ctx_pred_gate_drive_amp_pA"], 400.0)

        h_persist = float(data["h_context_persistence_ms"])
        h_persist_pass = bool(data["h_context_persistence_pass"])
        forecast = float(data["h_prediction_pretrailer_forecast_probability"])
        forecast_pass = bool(data["h_prediction_pretrailer_forecast_pass"])
        no_runaway_rate = float(data["no_runaway_max_rate_hz"])
        no_runaway_pass = bool(data["no_runaway_pass"])

        assert np.isclose(
            float(data["h_context_persistence_min_ms"]),
            H_CONTEXT_PERSISTENCE_MIN_MS,
        )
        assert np.isclose(
            float(data["h_context_persistence_max_ms"]),
            H_CONTEXT_PERSISTENCE_MAX_MS,
        )
        assert np.isclose(
            float(data["h_prediction_pretrailer_forecast_threshold"]),
            H_PRED_FORECAST_PROB_MIN,
        )
        assert np.isclose(float(data["no_runaway_threshold_hz"]), NO_RUNAWAY_MAX_RATE_HZ)

        assert h_persist_pass == (
            H_CONTEXT_PERSISTENCE_MIN_MS
            <= h_persist
            <= H_CONTEXT_PERSISTENCE_MAX_MS
        )
        assert forecast_pass == (forecast >= H_PRED_FORECAST_PROB_MIN)
        assert no_runaway_pass == (no_runaway_rate <= NO_RUNAWAY_MAX_RATE_HZ)

        thresholds_all_pass = bool(h_persist_pass and forecast_pass and no_runaway_pass)
        assert bool(data["native_gate_thresholds_all_pass"]) == thresholds_all_pass
        assert metrics["thresholds_all_pass"] == thresholds_all_pass
        assert bool(data["native_gate_metrics_all_pass"]) is False
        assert metrics["all_pass"] is False
        assert bool(data["passed"]) is False
        assert bool(data["native_scientific_stage1_passed"]) is False

        # Metrics now come from native H dynamics, but the checkpoint remains
        # provisional until the semantics are independently accepted.
        assert np.isclose(
            h_persist,
            float(np.mean(data["h_gate_ctx_persistence_ms_by_trial"])),
        )
        assert np.isclose(forecast, (
            float(np.count_nonzero(data["h_gate_pred_pretrailer_target_counts"]))
            / float(int(data["h_gate_n_trials"]))
        ))
        assert no_runaway_pass is True

        stable_hash = stable_npz_content_hash(path)
        assert _decode_bytes(data["native_stable_content_sha256"]) == stable_hash
        assert data["W_ctx_pred_final"].shape == (N_CTX_PRED_SYN,)

        return {
            "stable_hash": stable_hash,
            "h_context_persistence_ms": h_persist,
            "forecast_probability": forecast,
            "no_runaway_max_rate_hz": no_runaway_rate,
            "thresholds_all_pass": thresholds_all_pass,
            "source": metrics["metric_source"],
        }


def _assert_h_ring_dynamics_gate_primitive() -> dict[str, float]:
    result = run_h_ring_dynamics_test(seed=SEED)
    max_error = max(float(v) for v in result["max_abs_error"].values())
    assert max_error <= 1e-8, result["max_abs_error"]
    for stem in (
        "ctx_total_counts",
        "pred_total_counts",
        "ctx_inh_total_counts",
        "pred_inh_total_counts",
    ):
        assert np.array_equal(result[f"cpu_{stem}"], result[f"cuda_{stem}"]), stem

    metrics = {str(k): float(v) for k, v in result["metrics"].items()}
    assert metrics["ctx_leader_total_spikes"] > 0.0, metrics
    assert metrics["ctx_persistence_total_spikes"] > 0.0, metrics
    assert metrics["pred_leader_total_spikes"] == 0.0, metrics
    assert metrics["pred_trailer_total_spikes"] > 0.0, metrics
    assert metrics["no_runaway_pass"] == 1.0, metrics
    assert metrics["max_rate_hz"] <= NO_RUNAWAY_MAX_RATE_HZ, metrics
    assert (
        H_CONTEXT_PERSISTENCE_MIN_MS
        <= metrics["ctx_persistence_ms"]
        <= H_CONTEXT_PERSISTENCE_MAX_MS
    ), metrics
    assert metrics["ctx_persistence_window_pass"] == 1.0, metrics
    return {
        "h_ring_max_error": max_error,
        "h_ring_ctx_persistence_ms": metrics["ctx_persistence_ms"],
        "h_ring_max_rate_hz": metrics["max_rate_hz"],
    }


def _assert_same_seed_stable_hash(path_a: Path, path_b: Path) -> None:
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        hash_a = _decode_bytes(a["native_stable_content_sha256"])
        hash_b = _decode_bytes(b["native_stable_content_sha256"])
        assert hash_a == hash_b
        assert stable_npz_content_hash(path_a) == stable_npz_content_hash(path_b)
        assert stable_npz_content_hash(path_a) == hash_a
        assert np.array_equal(a["native_schedule_pairs"], b["native_schedule_pairs"])
        assert np.array_equal(a["W_ctx_pred_final"], b["W_ctx_pred_final"])
        assert np.array_equal(a["gate_dw_sum"], b["gate_dw_sum"])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_stage1_gate_metrics_") as tmp_s:
        root = Path(tmp_s)
        path_a, _, _ = write_native_stage1_n72_checkpoint(
            root / "stage_1_ctx_pred_seed42_native_n72_a.npz",
            seed=SEED,
        )
        path_b, _, _ = write_native_stage1_n72_checkpoint(
            root / "stage_1_ctx_pred_seed42_native_n72_b.npz",
            seed=SEED,
        )
        metrics = _assert_gate_logic(path_a)
        _assert_gate_logic(path_b)
        _assert_same_seed_stable_hash(path_a, path_b)
        h_metrics = _assert_h_ring_dynamics_gate_primitive()

        print(
            "validate_native_stage1_gate_metrics: PASS",
            f"path={path_a}",
            f"stable_content_sha256={metrics['stable_hash']}",
            f"h_context_persistence_ms={metrics['h_context_persistence_ms']:.6f}",
            f"forecast_probability={metrics['forecast_probability']:.6f}",
            f"no_runaway_max_rate_hz={metrics['no_runaway_max_rate_hz']:.6f}",
            f"thresholds_all_pass={metrics['thresholds_all_pass']}",
            f"source={metrics['source']}",
            f"h_ring_ctx_persistence_ms={h_metrics['h_ring_ctx_persistence_ms']:.6f}",
            f"h_ring_max_rate_hz={h_metrics['h_ring_max_rate_hz']:.6f}",
            "native_gate_metrics_all_pass=False",
            "passed=False",
            "native_scientific_stage1_passed=False",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
