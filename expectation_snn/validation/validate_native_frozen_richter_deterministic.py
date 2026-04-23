"""Validate deterministic native bounded frozen-Richter trial scheduling."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_frozen_richter_deterministic_trial_test,
)
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-10
BOUNDARY_CELLS = (100, 101, 102, 103)


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _count_array(result: dict, side: str, key: str) -> np.ndarray:
    return np.asarray(result[f"{side}_raw_counts"][key], dtype=np.int32)


def _count_total(result: dict, side: str, key: str) -> int:
    return int(_count_array(result, side, key).sum())


def _assert_count_equal(result: dict, key: str) -> None:
    assert np.array_equal(
        _count_array(result, "cpu", key),
        _count_array(result, "cuda", key),
    ), key


def _assert_phase_boundary(result: dict, pop: str) -> None:
    leader = _count_array(result, "cpu", f"{pop}.leader")
    preprobe = _count_array(result, "cpu", f"{pop}.preprobe")
    trailer = _count_array(result, "cpu", f"{pop}.trailer")
    assert int(leader[100]) == 1
    assert int(preprobe[100]) == 0
    assert int(trailer[100]) == 0
    assert int(leader[101]) == 0
    assert int(preprobe[101]) == 1
    assert int(trailer[101]) == 0
    assert int(leader[102]) == 0
    assert int(preprobe[102]) == 0
    assert int(trailer[102]) == 1
    assert int(leader[103]) == 0
    assert int(preprobe[103]) == 0
    assert int(trailer[103]) == 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_frozen_richter_det_") as tmp_s:
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

        result = run_frozen_richter_deterministic_trial_test(
            arrays,
            expected_stim_pre_index=0,
            unexpected_stim_pre_index=20,
            stim_period_steps=5,
            n_steps=120,
            leader_start_step=0,
            leader_end_step=30,
            preprobe_start_step=30,
            preprobe_end_step=60,
            trailer_start_step=60,
            trailer_end_step=100,
            iti_start_step=100,
            iti_end_step=120,
        )

    assert result["n_steps"] == 120
    assert result["expected_stim_pre_index"] == 0
    assert result["unexpected_stim_pre_index"] == 20
    assert result["stim_period_steps"] == 5
    assert result["phase_steps"] == {
        "leader_start_step": 0,
        "leader_end_step": 30,
        "preprobe_start_step": 30,
        "preprobe_end_step": 60,
        "trailer_start_step": 60,
        "trailer_end_step": 100,
        "iti_start_step": 100,
        "iti_end_step": 120,
        "first_stim_step": 2,
        "v1_force_step": 34,
        "hctx_force_step": 66,
        "hpred_force_step": 88,
    }
    assert result["edge_counts"] == {
        "v1_stim_to_e": 3840,
        "v1_to_h_ctx": 21504,
        "ctx_to_pred": 36864,
        "fb_pred_to_v1e_apical": 3072,
        "fb_pred_to_v1som": 3072,
    }
    assert result["source_fanouts"] == {
        "v1_stim_to_e.expected": 16,
        "v1_stim_to_e.unexpected": 16,
        "v1_to_h_ctx": 112,
        "ctx_to_pred": 192,
        "fb_pred_to_v1e_apical": 16,
        "fb_pred_to_v1som": 16,
    }
    assert result["source_event_counts"] == {
        "expected.leader": 6,
        "expected.preprobe": 6,
        "expected.trailer": 0,
        "unexpected.leader": 0,
        "unexpected.preprobe": 0,
        "unexpected.trailer": 8,
        "total": 20,
    }
    assert result["drive_amps"] == {
        "v1_stim_to_e": 35.0,
        "v1_to_h_ctx": 80.0,
        "ctx_to_pred": 25.0,
        "fb_pred_to_v1e_apical": 30.0,
        "fb_pred_to_v1som": 40.0,
    }
    assert np.isclose(
        result["event_sums"]["v1_stim_to_e.expected_target"],
        35.0,
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["event_sums"]["v1_stim_to_e.unexpected_first_target"],
        35.0,
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["event_sums"]["ctx_to_pred"],
        0.125,
        atol=TOL,
        rtol=0.0,
    )
    for key in ("v1_to_h_ctx", "fb_pred_to_v1e_apical", "fb_pred_to_v1som"):
        assert float(result["event_sums"][key]) > 0.0, key

    assert int(result["v1e_index"]) == 0
    assert int(result["hctx_index"]) not in BOUNDARY_CELLS
    assert int(result["hpred_index"]) not in BOUNDARY_CELLS
    assert int(result["feedback_v1e_index"]) not in BOUNDARY_CELLS
    assert 0 <= int(result["hctx_index"]) < 192
    assert 0 <= int(result["hpred_index"]) < 192
    assert 0 <= int(result["feedback_v1e_index"]) < 192
    assert 0 <= int(result["feedback_som_index"]) < 48

    for pop in ("v1_e", "hctx_e", "hpred_e"):
        for phase in ("leader", "preprobe", "trailer"):
            _assert_count_equal(result, f"{pop}.{phase}")
        _assert_phase_boundary(result, pop)

    assert _count_total(result, "cpu", "v1_e.leader") == 1
    assert _count_total(result, "cpu", "v1_e.preprobe") == 2
    assert _count_total(result, "cpu", "v1_e.trailer") == 1
    assert _count_total(result, "cpu", "hctx_e.leader") == 1
    assert _count_total(result, "cpu", "hctx_e.preprobe") == 1
    assert _count_total(result, "cpu", "hctx_e.trailer") == 2
    assert _count_total(result, "cpu", "hpred_e.leader") == 1
    assert _count_total(result, "cpu", "hpred_e.preprobe") == 1
    assert _count_total(result, "cpu", "hpred_e.trailer") == 2

    assert int(_count_array(result, "cpu", "v1_e.preprobe")[0]) == 1
    assert int(
        _count_array(result, "cpu", "hctx_e.trailer")[int(result["hctx_index"])]
    ) == 1
    assert int(
        _count_array(result, "cpu", "hpred_e.trailer")[int(result["hpred_index"])]
    ) == 1

    ordering = result["ordering_deltas"]
    assert float(ordering["stim_same_step_abs"]) <= TOL
    assert float(ordering["v1_h_same_step_abs"]) <= TOL
    assert float(ordering["ctx_pred_same_step_abs"]) <= TOL
    assert float(ordering["feedback_same_step_abs"]) <= TOL
    assert float(ordering["stim_next_delta"]) > 0.0
    assert float(ordering["v1_h_next_delta"]) > 0.0
    assert float(ordering["ctx_pred_next_delta"]) > 0.0
    assert float(ordering["feedback_apical_next_delta"]) > 0.0
    assert float(ordering["feedback_som_next_delta"]) > 0.0
    assert float(ordering["feedback_soma_late_delta"]) > 0.0

    max_err = _max_error(result)
    assert max_err <= TOL, result["max_abs_error"]
    for key, cpu_values in result["cpu_final_state"].items():
        assert np.allclose(
            cpu_values,
            result["cuda_final_state"][key],
            atol=TOL,
            rtol=0.0,
        ), key

    print(
        "validate_native_frozen_richter_deterministic: PASS",
        f"backend_info={backend_info()}",
        "phases=leader:[0,30),preprobe:[30,60),trailer:[60,100),iti:[100,120)",
        "source_events=expected:12,unexpected:8",
        "chain=v1_stim_to_e->v1_to_h_ctx->ctx_to_pred->feedback",
        f"counts=v1:{[_count_total(result, 'cpu', f'v1_e.{p}') for p in ('leader', 'preprobe', 'trailer')]}",
        f"hctx:{[_count_total(result, 'cpu', f'hctx_e.{p}') for p in ('leader', 'preprobe', 'trailer')]}",
        f"hpred:{[_count_total(result, 'cpu', f'hpred_e.{p}') for p in ('leader', 'preprobe', 'trailer')]}",
        f"max_err={max_err:.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
