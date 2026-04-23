"""Validate deterministic multi-step H_pred -> V1 native feedback primitive."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_feedback_v1_count_test,
)
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-12


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_feedback_v1_counts_") as tmp_s:
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

        result = run_feedback_v1_count_test(
            arrays,
            direct_bank_name="fb_pred_to_v1e_apical",
            som_bank_name="fb_pred_to_v1som",
            pre_index=7,
            event_steps=(2, 3),
            n_steps=20,
            window_start_step=5,
            window_end_step=10,
        )

    cpu_counts = np.asarray(result["cpu_counts"], dtype=np.int32)
    cuda_counts = np.asarray(result["cuda_counts"], dtype=np.int32)
    assert result["direct_bank_name"] == "fb_pred_to_v1e_apical"
    assert result["som_bank_name"] == "fb_pred_to_v1som"
    assert int(result["direct_edge_count"]) == 3072
    assert int(result["som_edge_count"]) == 3072
    assert int(result["direct_edges_for_source"]) == 16
    assert int(result["som_edges_for_source"]) == 16
    assert 0 <= int(result["target_v1e_index"]) < 192
    assert 0 <= int(result["target_som_index"]) < 48
    assert result["n_steps"] == 20
    assert result["window_start_step"] == 5
    assert result["window_end_step"] == 10
    assert float(result["direct_drive_amp"]) == 30.0
    assert float(result["som_drive_amp"]) == 40.0
    assert float(result["direct_event_sum_to_target"]) > 0.0
    assert float(result["som_event_sum_to_target"]) > 0.0

    assert np.array_equal(cpu_counts, cuda_counts)
    assert int(result["cpu_total_window_spikes"]) == int(result["cuda_total_window_spikes"])
    assert int(result["cpu_total_window_spikes"]) == 1
    assert int(cpu_counts[101]) == 1          # start is included: [start, end)
    assert int(cpu_counts[100]) == 0          # start - 1 is excluded
    assert int(cpu_counts[102]) == 0          # end is excluded

    max_err = _max_error(result)
    assert max_err <= TOL, result["max_abs_error"]
    for key, cpu_values in result["cpu_final_state"].items():
        assert np.allclose(
            cpu_values,
            result["cuda_final_state"][key],
            atol=TOL,
            rtol=0.0,
        ), key

    same_step_soma_delta = abs(
        float(result["cpu_v1e_soma_after_event_step"])
        - float(result["cpu_no_event_v1e_soma_after_event_step"])
    )
    assert same_step_soma_delta <= TOL, result
    assert np.isclose(
        result["cpu_v1e_soma_after_event_step"],
        result["cuda_v1e_soma_after_event_step"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cpu_v1e_i_ap_after_event_scatter"],
        result["direct_event_sum_to_target"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cuda_v1e_i_ap_after_event_scatter"],
        result["direct_event_sum_to_target"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cpu_v1som_i_e_after_event_scatter"],
        result["som_event_sum_to_target"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cuda_v1som_i_e_after_event_scatter"],
        result["som_event_sum_to_target"],
        atol=TOL,
        rtol=0.0,
    )

    apical_next_delta = (
        float(result["cpu_v1e_ap_after_next_step"])
        - float(result["cpu_no_event_v1e_ap_after_next_step"])
    )
    assert apical_next_delta > 0.0, result
    assert np.isclose(
        result["cpu_v1e_ap_after_next_step"],
        result["cuda_v1e_ap_after_next_step"],
        atol=TOL,
        rtol=0.0,
    )
    som_next_delta = (
        float(result["cpu_v1som_v_after_next_step"])
        - float(result["cpu_no_event_v1som_v_after_next_step"])
    )
    assert som_next_delta > 0.0, result
    assert np.isclose(
        result["cpu_v1som_v_after_next_step"],
        result["cuda_v1som_v_after_next_step"],
        atol=TOL,
        rtol=0.0,
    )
    soma_late_delta = (
        float(result["cpu_v1e_soma_after_late_step"])
        - float(result["cpu_no_event_v1e_soma_after_late_step"])
    )
    assert soma_late_delta > 0.0, result
    assert np.isclose(
        result["cpu_v1e_soma_after_late_step"],
        result["cuda_v1e_soma_after_late_step"],
        atol=TOL,
        rtol=0.0,
    )

    print(
        "validate_native_feedback_v1_counts: PASS",
        f"backend_info={backend_info()}",
        "direct_bank=fb_pred_to_v1e_apical",
        "som_bank=fb_pred_to_v1som",
        "pre_index=7",
        "event_steps=2,3",
        "window=[5,10)",
        f"window_spikes={int(result['cpu_total_window_spikes'])}",
        f"same_step_soma_delta={same_step_soma_delta:.3e}",
        f"apical_next_delta={apical_next_delta:.7f}",
        f"som_next_delta={som_next_delta:.7f}",
        f"soma_late_delta={soma_late_delta:.7f}",
        f"max_err={max_err:.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
