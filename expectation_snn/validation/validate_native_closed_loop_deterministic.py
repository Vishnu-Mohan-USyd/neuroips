"""Validate bounded deterministic native V1/H_ctx/H_pred closed-loop scheduler."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_closed_loop_deterministic_count_test,
)
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-11


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _assert_close(a: float, b: float, *, label: str) -> None:
    assert np.isclose(float(a), float(b), atol=TOL, rtol=0.0), (
        label,
        a,
        b,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_closed_loop_det_") as tmp_s:
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

        result = run_closed_loop_deterministic_count_test(
            arrays,
            stim_pre_index=0,
            stim_step=2,
            v1_force_step=4,
            hctx_force_step=26,
            hpred_force_step=28,
            n_steps=35,
            window_start_step=4,
            window_end_step=32,
        )

    assert result["n_steps"] == 35
    assert result["window_start_step"] == 4
    assert result["window_end_step"] == 32
    assert result["stim_step"] == 2
    assert result["v1_force_step"] == 4
    assert result["hctx_force_step"] == 26
    assert result["hpred_force_step"] == 28
    assert result["stim_pre_index"] == 0
    assert result["v1e_index"] == 0
    assert 0 <= int(result["hctx_index"]) < 192
    assert 0 <= int(result["hpred_index"]) < 192
    assert 0 <= int(result["feedback_v1e_index"]) < 192
    assert 0 <= int(result["feedback_som_index"]) < 48

    assert result["edge_counts"] == {
        "v1_stim_to_e": 3840,
        "v1_to_h_ctx": 21504,
        "ctx_to_pred": 36864,
        "fb_pred_to_v1e_apical": 3072,
        "fb_pred_to_v1som": 3072,
    }
    assert result["source_fanouts"] == {
        "v1_stim_to_e": 16,
        "v1_to_h_ctx": 112,
        "ctx_to_pred": 192,
        "fb_pred_to_v1e_apical": 16,
        "fb_pred_to_v1som": 16,
    }
    assert result["drive_amps"] == {
        "v1_stim_to_e": 35.0,
        "v1_to_h_ctx": 80.0,
        "ctx_to_pred": 25.0,
        "fb_pred_to_v1e_apical": 30.0,
        "fb_pred_to_v1som": 40.0,
    }
    assert np.isclose(
        result["event_sums"]["v1_stim_to_e"],
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
    for key in (
        "v1_to_h_ctx",
        "fb_pred_to_v1e_apical",
        "fb_pred_to_v1som",
    ):
        assert float(result["event_sums"][key]) > 0.0, key

    cpu_v1_counts = np.asarray(result["cpu_v1_counts"], dtype=np.int32)
    cuda_v1_counts = np.asarray(result["cuda_v1_counts"], dtype=np.int32)
    cpu_hctx_counts = np.asarray(result["cpu_hctx_counts"], dtype=np.int32)
    cuda_hctx_counts = np.asarray(result["cuda_hctx_counts"], dtype=np.int32)
    cpu_hpred_counts = np.asarray(result["cpu_hpred_counts"], dtype=np.int32)
    cuda_hpred_counts = np.asarray(result["cuda_hpred_counts"], dtype=np.int32)
    assert np.array_equal(cpu_v1_counts, cuda_v1_counts)
    assert np.array_equal(cpu_hctx_counts, cuda_hctx_counts)
    assert np.array_equal(cpu_hpred_counts, cuda_hpred_counts)
    assert int(result["cpu_total_v1_window_spikes"]) == 1
    assert int(result["cuda_total_v1_window_spikes"]) == 1
    assert int(result["cpu_total_hctx_window_spikes"]) == 1
    assert int(result["cuda_total_hctx_window_spikes"]) == 1
    assert int(result["cpu_total_hpred_window_spikes"]) == 1
    assert int(result["cuda_total_hpred_window_spikes"]) == 1
    assert int(cpu_v1_counts[int(result["v1e_index"])]) == 1
    assert int(cpu_hctx_counts[int(result["hctx_index"])]) == 1
    assert int(cpu_hpred_counts[int(result["hpred_index"])]) == 1

    max_err = _max_error(result)
    assert max_err <= TOL, result["max_abs_error"]
    for key, cpu_values in result["cpu_final_state"].items():
        assert np.allclose(
            cpu_values,
            result["cuda_final_state"][key],
            atol=TOL,
            rtol=0.0,
        ), key

    stim_same_step_delta = abs(
        float(result["cpu_v1_soma_after_stim_step"])
        - float(result["cpu_no_stim_v1_soma_after_stim_step"])
    )
    assert stim_same_step_delta <= TOL
    _assert_close(
        result["cpu_v1_soma_after_stim_step"],
        result["cuda_v1_soma_after_stim_step"],
        label="stim same-step CPU/CUDA",
    )
    _assert_close(
        result["cpu_v1_i_e_after_stim_scatter"],
        result["event_sums"]["v1_stim_to_e"],
        label="stim scatter",
    )
    _assert_close(
        result["cuda_v1_i_e_after_stim_scatter"],
        result["event_sums"]["v1_stim_to_e"],
        label="stim scatter cuda",
    )
    stim_next_delta = (
        float(result["cpu_v1_soma_after_stim_next_step"])
        - float(result["cpu_no_stim_v1_soma_after_stim_next_step"])
    )
    assert stim_next_delta > 0.0
    _assert_close(
        result["cpu_v1_soma_after_stim_next_step"],
        result["cuda_v1_soma_after_stim_next_step"],
        label="stim next-step CPU/CUDA",
    )

    v1_h_same_step_delta = abs(
        float(result["cpu_hctx_v_after_v1_step"])
        - float(result["cpu_no_v1_hctx_v_after_v1_step"])
    )
    assert v1_h_same_step_delta <= TOL
    _assert_close(
        result["cpu_hctx_i_e_after_v1_scatter"],
        result["event_sums"]["v1_to_h_ctx"],
        label="V1->H scatter",
    )
    v1_h_next_delta = (
        float(result["cpu_hctx_v_after_v1_next_step"])
        - float(result["cpu_no_v1_hctx_v_after_v1_next_step"])
    )
    assert v1_h_next_delta > 0.0
    _assert_close(
        result["cpu_hctx_v_after_v1_next_step"],
        result["cuda_hctx_v_after_v1_next_step"],
        label="V1->H next-step CPU/CUDA",
    )

    ctx_pred_same_step_delta = abs(
        float(result["cpu_hpred_v_after_hctx_step"])
        - float(result["cpu_no_ctx_hpred_v_after_hctx_step"])
    )
    assert ctx_pred_same_step_delta <= TOL
    _assert_close(
        result["cpu_hpred_i_e_after_hctx_scatter"],
        result["event_sums"]["ctx_to_pred"],
        label="ctx->pred scatter",
    )
    ctx_pred_next_delta = (
        float(result["cpu_hpred_v_after_hctx_next_step"])
        - float(result["cpu_no_ctx_hpred_v_after_hctx_next_step"])
    )
    assert ctx_pred_next_delta > 0.0
    _assert_close(
        result["cpu_hpred_v_after_hctx_next_step"],
        result["cuda_hpred_v_after_hctx_next_step"],
        label="ctx->pred next-step CPU/CUDA",
    )

    feedback_same_step_delta = abs(
        float(result["cpu_v1_soma_after_hpred_step"])
        - float(result["cpu_no_fb_v1_soma_after_hpred_step"])
    )
    assert feedback_same_step_delta <= TOL
    _assert_close(
        result["cpu_v1_i_ap_after_fb_scatter"],
        result["event_sums"]["fb_pred_to_v1e_apical"],
        label="feedback direct scatter",
    )
    _assert_close(
        result["cpu_som_i_e_after_fb_scatter"],
        result["event_sums"]["fb_pred_to_v1som"],
        label="feedback SOM scatter",
    )
    feedback_apical_next_delta = (
        float(result["cpu_v1_ap_after_fb_next_step"])
        - float(result["cpu_no_fb_v1_ap_after_fb_next_step"])
    )
    feedback_som_next_delta = (
        float(result["cpu_som_v_after_fb_next_step"])
        - float(result["cpu_no_fb_som_v_after_fb_next_step"])
    )
    feedback_soma_late_delta = (
        float(result["cpu_v1_soma_after_fb_late_step"])
        - float(result["cpu_no_fb_v1_soma_after_fb_late_step"])
    )
    assert feedback_apical_next_delta > 0.0
    assert feedback_som_next_delta > 0.0
    assert feedback_soma_late_delta > 0.0
    _assert_close(
        result["cpu_v1_ap_after_fb_next_step"],
        result["cuda_v1_ap_after_fb_next_step"],
        label="feedback V1 apical CPU/CUDA",
    )
    _assert_close(
        result["cpu_som_v_after_fb_next_step"],
        result["cuda_som_v_after_fb_next_step"],
        label="feedback SOM CPU/CUDA",
    )
    _assert_close(
        result["cpu_v1_soma_after_fb_late_step"],
        result["cuda_v1_soma_after_fb_late_step"],
        label="feedback V1 soma late CPU/CUDA",
    )

    print(
        "validate_native_closed_loop_deterministic: PASS",
        f"backend_info={backend_info()}",
        "chain=v1_stim_to_e->v1_to_h_ctx->ctx_to_pred->feedback",
        "steps=stim:2,v1:4,hctx:26,hpred:28",
        "window=[4,32)",
        f"counts=v1:{int(result['cpu_total_v1_window_spikes'])}",
        f"hctx:{int(result['cpu_total_hctx_window_spikes'])}",
        f"hpred:{int(result['cpu_total_hpred_window_spikes'])}",
        f"stim_next_delta={stim_next_delta:.7f}",
        f"v1_h_next_delta={v1_h_next_delta:.7f}",
        f"ctx_pred_next_delta={ctx_pred_next_delta:.7f}",
        f"fb_apical_next_delta={feedback_apical_next_delta:.7f}",
        f"fb_som_next_delta={feedback_som_next_delta:.7f}",
        f"fb_soma_late_delta={feedback_soma_late_delta:.7f}",
        f"max_err={max_err:.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
