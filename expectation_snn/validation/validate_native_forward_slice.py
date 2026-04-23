"""Validate first deterministic native frozen-forward event ordering slice."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import backend_info, run_event_ordering_slice
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-12


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_forward_slice_") as tmp_s:
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

        result = run_event_ordering_slice(
            arrays,
            bank_name="ctx_to_pred",
            pre_index=7,
        )

    assert result["bank_name"] == "ctx_to_pred"
    assert result["n_edges_for_source"] == 192
    assert 0 <= int(result["target_index"]) < 192
    assert float(result["drive_amp"]) == 25.0
    assert np.isclose(result["event_sum_to_target"], 0.125, atol=TOL, rtol=0.0)
    assert np.isclose(result["cpu_i_e_after_scatter"], 0.125, atol=TOL, rtol=0.0)
    assert np.isclose(result["cuda_i_e_after_scatter"], 0.125, atol=TOL, rtol=0.0)
    assert abs(float(result["max_abs_error"])) <= TOL, result
    assert int(result["cpu_total_spikes"]) == 0
    assert int(result["cuda_total_spikes"]) == 0

    same_step_delta = abs(
        float(result["cpu_v_after_step0"])
        - float(result["cpu_no_event_v_after_step0"])
    )
    assert same_step_delta <= TOL, result
    assert np.isclose(
        result["cuda_v_after_step0"],
        result["cpu_no_event_v_after_step0"],
        atol=TOL,
        rtol=0.0,
    )
    next_step_delta = (
        float(result["cpu_v_after_step1"])
        - float(result["cpu_no_event_v_after_step1"])
    )
    assert next_step_delta > 0.0, result
    assert np.isclose(next_step_delta, 0.0000625, atol=1e-12, rtol=0.0)

    print(
        "validate_native_forward_slice: PASS",
        f"backend_info={backend_info()}",
        "bank=ctx_to_pred",
        "pre_index=7",
        f"target_index={int(result['target_index'])}",
        f"same_step_delta={same_step_delta:.3e}",
        f"next_step_delta={next_step_delta:.7f}",
        f"max_err={float(result['max_abs_error']):.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
