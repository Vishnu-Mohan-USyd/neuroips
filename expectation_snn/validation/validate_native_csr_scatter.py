"""Validate native CSR event scatter against an exported synapse bank."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import backend_info, run_csr_scatter_test
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-12


def _assert_duplicate_target_scatter() -> dict:
    arrays = {
        "syn_duplicate_pre": np.asarray([0, 0, 0, 1], dtype=np.int32),
        "syn_duplicate_post": np.asarray([2, 2, 2, 1], dtype=np.int32),
        "syn_duplicate_w": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "syn_duplicate_drive_amp_pA": np.asarray(5.0, dtype=np.float64),
    }
    result = run_csr_scatter_test(arrays, bank_name="duplicate", pre_index=0)
    cpu_target = np.asarray(result["cpu_target"], dtype=np.float64)
    cuda_target = np.asarray(result["cuda_target"], dtype=np.float64)
    assert result["n_edges"] == 4
    assert result["n_pre"] == 2
    assert result["n_target"] == 3
    assert result["n_edges_for_source"] == 3
    assert float(result["max_abs_error"]) <= TOL, result
    assert np.allclose(cpu_target, cuda_target, atol=TOL, rtol=0.0)
    assert np.isclose(cuda_target[2], 3.0, atol=TOL, rtol=0.0)
    assert np.count_nonzero(cuda_target) == 1
    return result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_csr_scatter_") as tmp_s:
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

        result = run_csr_scatter_test(
            arrays,
            bank_name="ctx_to_pred",
            pre_index=7,
        )
        cpu_target = np.asarray(result["cpu_target"], dtype=np.float64)
        cuda_target = np.asarray(result["cuda_target"], dtype=np.float64)
        assert result["n_edges"] == 36864
        assert result["n_pre"] == 192
        assert result["n_target"] == 192
        assert result["n_edges_for_source"] == 192
        assert float(result["drive_amp"]) == 25.0
        assert float(result["max_abs_error"]) <= TOL, result
        assert np.allclose(cpu_target, cuda_target, atol=TOL, rtol=0.0)
        assert int(np.count_nonzero(cuda_target)) == 192
        assert np.isclose(cuda_target.sum(), 24.0, atol=TOL, rtol=0.0)
        duplicate = _assert_duplicate_target_scatter()

    print(
        "validate_native_csr_scatter: PASS",
        f"backend_info={backend_info()}",
        "bank=ctx_to_pred",
        "pre_index=7",
        "n_edges_for_source=192",
        f"max_err={float(result['max_abs_error']):.3e}",
        f"target_sum={float(cuda_target.sum()):.6f}",
        f"duplicate_target_sum={float(duplicate['cuda_target'][2]):.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
