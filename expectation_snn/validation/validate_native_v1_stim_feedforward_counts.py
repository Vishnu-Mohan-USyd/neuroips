"""Validate deterministic V1 stimulus and V1->H_ctx native scheduler slice."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_v1_stim_feedforward_count_test,
)
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-12


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_v1_stim_ff_counts_") as tmp_s:
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

        result = run_v1_stim_feedforward_count_test(
            arrays,
            stim_bank_name="v1_stim_to_e",
            feedforward_bank_name="v1_to_h_ctx",
            stim_pre_index=0,
            stim_event_steps=(2,),
            force_v1e_step=4,
            n_steps=20,
            window_start_step=4,
            window_end_step=10,
        )

    cpu_v1_counts = np.asarray(result["cpu_v1_counts"], dtype=np.int32)
    cuda_v1_counts = np.asarray(result["cuda_v1_counts"], dtype=np.int32)
    cpu_h_counts = np.asarray(result["cpu_h_counts"], dtype=np.int32)
    cuda_h_counts = np.asarray(result["cuda_h_counts"], dtype=np.int32)
    forced_v1 = int(result["forced_v1e_index"])

    assert result["stim_bank_name"] == "v1_stim_to_e"
    assert result["feedforward_bank_name"] == "v1_to_h_ctx"
    assert int(result["stim_edge_count"]) == 3840
    assert int(result["feedforward_edge_count"]) == 21504
    assert int(result["stim_edges_for_source"]) == 16
    assert int(result["feedforward_edges_for_source"]) == 112
    assert forced_v1 == 0
    assert 0 <= int(result["target_h_index"]) < 192
    assert result["window_start_step"] == 4
    assert result["window_end_step"] == 10
    assert result["force_v1e_step"] == 4
    assert float(result["stim_drive_amp"]) == 35.0
    assert float(result["feedforward_drive_amp"]) == 80.0
    assert np.isclose(
        result["stim_event_sum_to_v1e_target"],
        35.0,
        atol=TOL,
        rtol=0.0,
    )
    assert float(result["feedforward_event_sum_to_h_target"]) > 0.0

    assert np.array_equal(cpu_v1_counts, cuda_v1_counts)
    assert np.array_equal(cpu_h_counts, cuda_h_counts)
    assert int(result["cpu_total_v1_window_spikes"]) == int(
        result["cuda_total_v1_window_spikes"]
    )
    assert int(result["cpu_total_h_window_spikes"]) == int(
        result["cuda_total_h_window_spikes"]
    )
    assert int(result["cpu_total_v1_window_spikes"]) == 1
    assert int(cpu_v1_counts[forced_v1]) == 1
    assert int(result["cpu_total_h_window_spikes"]) == 1
    assert int(cpu_h_counts[101]) == 1        # start is included: [start, end)
    assert int(cpu_h_counts[100]) == 0        # start - 1 is excluded
    assert int(cpu_h_counts[102]) == 0        # end is excluded

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
        float(result["cpu_v1e_soma_after_stim_step"])
        - float(result["cpu_no_stim_v1e_soma_after_stim_step"])
    )
    assert stim_same_step_delta <= TOL, result
    assert np.isclose(
        result["cpu_v1e_soma_after_stim_step"],
        result["cuda_v1e_soma_after_stim_step"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cpu_v1e_i_e_after_stim_scatter"],
        result["stim_event_sum_to_v1e_target"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cuda_v1e_i_e_after_stim_scatter"],
        result["stim_event_sum_to_v1e_target"],
        atol=TOL,
        rtol=0.0,
    )
    stim_next_delta = (
        float(result["cpu_v1e_soma_after_stim_next_step"])
        - float(result["cpu_no_stim_v1e_soma_after_stim_next_step"])
    )
    assert stim_next_delta > 0.0, result
    assert np.isclose(stim_next_delta, 0.0175, atol=TOL, rtol=0.0)
    assert np.isclose(
        result["cpu_v1e_soma_after_stim_next_step"],
        result["cuda_v1e_soma_after_stim_next_step"],
        atol=TOL,
        rtol=0.0,
    )

    h_same_step_delta = abs(
        float(result["cpu_h_v_after_force_v1e_step"])
        - float(result["cpu_no_ff_h_v_after_force_v1e_step"])
    )
    assert h_same_step_delta <= TOL, result
    assert np.isclose(
        result["cpu_h_v_after_force_v1e_step"],
        result["cuda_h_v_after_force_v1e_step"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cpu_h_i_e_after_ff_scatter"],
        result["feedforward_event_sum_to_h_target"],
        atol=TOL,
        rtol=0.0,
    )
    assert np.isclose(
        result["cuda_h_i_e_after_ff_scatter"],
        result["feedforward_event_sum_to_h_target"],
        atol=TOL,
        rtol=0.0,
    )
    h_next_delta = (
        float(result["cpu_h_v_after_ff_next_step"])
        - float(result["cpu_no_ff_h_v_after_ff_next_step"])
    )
    assert h_next_delta > 0.0, result
    assert np.isclose(
        result["cpu_h_v_after_ff_next_step"],
        result["cuda_h_v_after_ff_next_step"],
        atol=TOL,
        rtol=0.0,
    )

    print(
        "validate_native_v1_stim_feedforward_counts: PASS",
        f"backend_info={backend_info()}",
        "stim_bank=v1_stim_to_e",
        "feedforward_bank=v1_to_h_ctx",
        "stim_pre_index=0",
        "stim_steps=2",
        "force_v1e_step=4",
        "window=[4,10)",
        f"v1_window_spikes={int(result['cpu_total_v1_window_spikes'])}",
        f"h_window_spikes={int(result['cpu_total_h_window_spikes'])}",
        f"stim_same_step_delta={stim_same_step_delta:.3e}",
        f"stim_next_delta={stim_next_delta:.7f}",
        f"h_same_step_delta={h_same_step_delta:.3e}",
        f"h_next_delta={h_next_delta:.7f}",
        f"max_err={max_err:.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
