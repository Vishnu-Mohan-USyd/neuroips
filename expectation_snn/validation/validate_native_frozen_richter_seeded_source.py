"""Validate seeded native frozen-Richter source generation parity."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_frozen_richter_seeded_source_test,
)
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-10
RAW_COUNT_KEYS = (
    "v1_e.leader",
    "v1_e.preprobe",
    "v1_e.trailer",
    "v1_error.trailer",
    "v1_error_neg.trailer",
    "hctx_e.leader",
    "hctx_e.preprobe",
    "hctx_e.trailer",
    "hpred_e.leader",
    "hpred_e.preprobe",
    "hpred_e.trailer",
)
RATE_KEYS = (
    "v1_e.leader",
    "v1_e.preprobe",
    "v1_e.trailer",
    "hctx_e.preprobe",
    "hctx_e.trailer",
    "hpred_e.preprobe",
    "hpred_e.trailer",
)
SOURCE_COUNT_KEYS = (
    "source.events_by_step",
    "source.events_by_afferent",
    "source.events_by_channel",
    "source.events_by_phase",
)


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _as_int_array(result: dict, side: str, group: str, key: str) -> np.ndarray:
    return np.asarray(result[f"{side}_{group}"][key], dtype=np.int32)


def _assert_int_maps_equal(result: dict, group: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        assert np.array_equal(
            _as_int_array(result, "cpu", group, key),
            _as_int_array(result, "cuda", group, key),
        ), (group, key)


def _assert_rate_maps_equal(result: dict) -> None:
    for key in RATE_KEYS:
        cpu = np.asarray(result["cpu_diagnostic_rates_hz"][key], dtype=np.float64)
        cuda = np.asarray(result["cuda_diagnostic_rates_hz"][key], dtype=np.float64)
        assert np.allclose(cpu, cuda, atol=TOL, rtol=0.0), key


def _assert_same_seed_equal(a: dict, b: dict) -> None:
    assert a["seed"] == b["seed"]
    assert a["source_event_counts"] == b["source_event_counts"]
    for key in RAW_COUNT_KEYS:
        assert np.array_equal(a["cpu_raw_counts"][key], b["cpu_raw_counts"][key]), key
    for key in SOURCE_COUNT_KEYS:
        assert np.array_equal(
            a["cpu_source_counts"][key],
            b["cpu_source_counts"][key],
        ), key


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_seeded_source_") as tmp_s:
        root = Path(tmp_s)
        ckpt_dir = root / "checkpoints"
        manifest_path = root / "ctx_pred_manifest.npz"
        _write_synthetic_checkpoints(ckpt_dir)
        export_ctx_pred_manifest(
            ckpt_dir=ckpt_dir,
            out_path=manifest_path,
            seed=SEED,
            r=1.0,
            g_total=1.0,
            v1_to_h_mode="context_only",
            with_feedback_routes=True,
        )
        with np.load(manifest_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}

        kwargs = dict(
            expected_channel=0,
            unexpected_channel=1,
            grating_rate_hz=500.0,
            baseline_rate_hz=0.0,
            n_steps=100,
            leader_start_step=0,
            leader_end_step=30,
            preprobe_start_step=30,
            preprobe_end_step=38,
            trailer_start_step=38,
            trailer_end_step=78,
            iti_start_step=78,
            iti_end_step=100,
        )
        result_a = run_frozen_richter_seeded_source_test(arrays, seed=12345, **kwargs)
        result_b = run_frozen_richter_seeded_source_test(arrays, seed=12345, **kwargs)
        result_c = run_frozen_richter_seeded_source_test(arrays, seed=54321, **kwargs)
        result_error = run_frozen_richter_seeded_source_test(
            arrays,
            seed=12345,
            v1_error_comparator_mode_id=1,
            v1_error_sensory_gain=1.0,
            v1_error_prediction_gain=1.0,
            **kwargs,
        )
        result_signed_error = run_frozen_richter_seeded_source_test(
            arrays,
            seed=12345,
            v1_error_comparator_mode_id=2,
            v1_error_sensory_gain=1.0,
            v1_error_prediction_gain=1.0,
            **kwargs,
        )

    assert result_a["seed"] == 12345
    assert result_a["n_steps"] == 100
    assert np.isclose(result_a["dt_ms"], 0.1, atol=0.0, rtol=0.0)
    assert result_a["expected_channel"] == 0
    assert result_a["unexpected_channel"] == 1
    assert result_a["phase_steps"] == {
        "leader_start_step": 0,
        "leader_end_step": 30,
        "preprobe_start_step": 30,
        "preprobe_end_step": 38,
        "trailer_start_step": 38,
        "trailer_end_step": 78,
        "iti_start_step": 78,
        "iti_end_step": 100,
    }
    assert result_a["edge_counts"] == {
        "v1_stim_to_e": 3840,
        "v1_to_h_ctx": 21504,
        "ctx_to_pred": 36864,
        "fb_pred_to_v1e_apical": 3072,
        "fb_pred_to_v1som": 3072,
    }
    assert result_a["rates_hz"] == {
        "grating": 500.0,
        "baseline": 0.0,
        "v1_stim_sigma_deg": 22.0,
    }

    _assert_int_maps_equal(result_a, "raw_counts", RAW_COUNT_KEYS)
    _assert_int_maps_equal(result_a, "source_counts", SOURCE_COUNT_KEYS)
    _assert_rate_maps_equal(result_a)
    _assert_same_seed_equal(result_a, result_b)
    _assert_int_maps_equal(result_error, "raw_counts", RAW_COUNT_KEYS)
    assert _max_error(result_error) <= TOL
    _assert_int_maps_equal(result_signed_error, "raw_counts", RAW_COUNT_KEYS)
    assert _max_error(result_signed_error) <= TOL

    source_by_step = _as_int_array(
        result_a, "cpu", "source_counts", "source.events_by_step"
    )
    source_by_afferent = _as_int_array(
        result_a, "cpu", "source_counts", "source.events_by_afferent"
    )
    source_by_channel = _as_int_array(
        result_a, "cpu", "source_counts", "source.events_by_channel"
    )
    source_by_phase = _as_int_array(
        result_a, "cpu", "source_counts", "source.events_by_phase"
    )
    assert source_by_step.shape == (100,)
    assert source_by_afferent.shape == (240,)
    assert source_by_channel.shape == (12,)
    assert source_by_phase.shape == (5,)
    assert int(source_by_phase[4]) == int(result_a["source_event_counts"]["total"])
    assert int(source_by_step.sum()) == int(result_a["source_event_counts"]["total"])
    assert int(source_by_afferent.sum()) == int(result_a["source_event_counts"]["total"])
    assert int(source_by_channel.sum()) == int(result_a["source_event_counts"]["total"])
    assert int(result_a["source_event_counts"]["leader"]) > 0
    assert int(result_a["source_event_counts"]["preprobe"]) > 0
    assert int(result_a["source_event_counts"]["trailer"]) > 0
    assert int(result_a["source_event_counts"]["iti"]) == 0
    assert int(result_a["source_event_counts"]["total"]) > 0

    for key in RAW_COUNT_KEYS:
        values = np.asarray(result_a["cpu_raw_counts"][key], dtype=np.int32)
        assert values.shape == (192,), key
    for key in RATE_KEYS:
        values = np.asarray(
            result_a["cpu_diagnostic_rates_hz"][key],
            dtype=np.float64,
        )
        assert values.shape == (192,), key

    different_seed_source_by_step = np.asarray(
        result_c["cpu_source_counts"]["source.events_by_step"],
        dtype=np.int32,
    )
    different_seed_source_by_afferent = np.asarray(
        result_c["cpu_source_counts"]["source.events_by_afferent"],
        dtype=np.int32,
    )
    assert (
        not np.array_equal(source_by_step, different_seed_source_by_step)
        or not np.array_equal(source_by_afferent, different_seed_source_by_afferent)
    )

    max_err = _max_error(result_a)
    assert max_err <= TOL, result_a["max_abs_error"]
    for key, cpu_values in result_a["cpu_final_state"].items():
        assert np.allclose(
            cpu_values,
            result_a["cuda_final_state"][key],
            atol=TOL,
            rtol=0.0,
        ), key

    print(
        "validate_native_frozen_richter_seeded_source: PASS",
        f"backend_info={backend_info()}",
        "phases=leader:[0,30),preprobe:[30,38),trailer:[38,78),iti:[78,100)",
        "schedule=expected_channel:0 leader/preprobe,unexpected_channel:1 trailer",
        f"source_events={result_a['source_event_counts']}",
        f"source_by_channel={source_by_channel.tolist()}",
        f"max_err={max_err:.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
