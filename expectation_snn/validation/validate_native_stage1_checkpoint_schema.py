"""Validate native Stage-1 ctx_pred checkpoint schema compatibility.

This validator writes a temporary checkpoint from the native CUDA tiny
trainer, then verifies the existing frozen-runtime checkpoint loader accepts
the artifact.  It is a schema/runtime-loading validation only: the native
tiny trainer's H recurrent arrays are placeholders and the checkpoint marks
``passed=False`` to avoid confusing this with a scientific Stage-1 pass.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from brian2 import Network, defaultclock, ms, prefs, start_scope
from brian2 import seed as b2_seed

from expectation_snn.assays.runtime import (
    _expected_ctx_pred_shape,
    _expected_h_ee_shape,
    _expected_v1_pv_to_e_shape,
    _load_stage1_ctx_pred_into,
    build_frozen_network,
)
from expectation_snn.brian2_model.h_context_prediction import (
    H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
    build_h_context_prediction,
    h_context_prediction_config_from_json,
)
from expectation_snn.brian2_model.v1_ring import build_v1_ring
from expectation_snn.cuda_sim.native import backend_info
from expectation_snn.cuda_sim.train_stage1_native import (
    N_CTX_PRED_SYN,
    N_H_EE_SYN,
    build_small_generated_stage1_schedule,
    write_tiny_trainer_stage1_checkpoint,
)


SEED = 42


def _decode_json_bytes(value: object) -> dict:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"cannot decode JSON payload from {type(value)!r}")


def _write_stage0_checkpoint(ckpt_dir: Path) -> Path:
    """Write the minimal Stage-0 checkpoint required by build_frozen_network."""
    start_scope()
    v1 = build_v1_ring(name_prefix="native_schema_stage0_shape")
    net = Network(*v1.groups)
    net.run(0 * ms)
    expected = _expected_v1_pv_to_e_shape(v1)
    assert expected == (6144,), expected
    path = ckpt_dir / f"stage_0_seed{SEED}.npz"
    np.savez(
        path,
        bias_pA=np.float64(0.0),
        pv_to_e_w=np.zeros(expected, dtype=np.float64),
    )
    return path


def _assert_stage1_npz_schema(path: Path, native_result: dict) -> None:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "ctx_ee_w_final",
            "pred_ee_w_final",
            "W_ctx_pred_final",
            "ctx_pred_config_schema_version",
            "ctx_pred_config_json",
            "ctx_pred_drive_amp_ctx_pred_pA",
            "ctx_pred_pred_e_uniform_bias_pA",
            "seed",
            "n_trials",
            "gate_w_before",
            "gate_w_after",
            "gate_dw_sum",
            "gate_elig_mean",
            "gate_elig_max",
            "gate_n_capped",
            "passed",
            "native_report_json",
            "native_placeholder_h_recurrent_arrays",
            "native_scientific_stage1_passed",
        }
        missing = sorted(required.difference(data.files))
        assert not missing, missing
        assert data["ctx_ee_w_final"].shape == (N_H_EE_SYN,)
        assert data["pred_ee_w_final"].shape == (N_H_EE_SYN,)
        assert data["W_ctx_pred_final"].shape == (N_CTX_PRED_SYN,)
        assert data["elig_final"].shape == (N_CTX_PRED_SYN,)
        assert data["gate_w_before"].shape == (int(native_result["n_trials"]),)
        assert data["gate_n_capped"].shape == (int(native_result["n_trials"]),)
        assert int(data["ctx_pred_config_schema_version"]) == (
            H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION
        )
        cfg = h_context_prediction_config_from_json(data["ctx_pred_config_json"])
        assert np.isclose(
            float(data["ctx_pred_drive_amp_ctx_pred_pA"]),
            cfg.drive_amp_ctx_pred_pA,
        )
        assert np.isclose(
            float(data["ctx_pred_pred_e_uniform_bias_pA"]),
            cfg.pred_e_uniform_bias_pA,
        )
        assert bool(data["passed"]) is False
        assert bool(data["native_scientific_stage1_passed"]) is False
        assert bool(data["native_placeholder_h_recurrent_arrays"]) is True
        report = _decode_json_bytes(data["native_report_json"])
        assert report["scientific_stage1_passed"] is False
        assert "placeholder" in report["h_recurrent_arrays"]


def _assert_direct_loader(path: Path) -> dict:
    start_scope()
    bundle = build_h_context_prediction(
        ctx_name="native_schema_ctx",
        pred_name="native_schema_pred",
    )
    net = Network(*bundle.groups)
    net.run(0 * ms)
    h_shape = _expected_h_ee_shape(bundle.ctx)
    cp_shape = _expected_ctx_pred_shape(bundle)
    assert h_shape == (N_H_EE_SYN,), h_shape
    assert cp_shape == (N_CTX_PRED_SYN,), cp_shape
    meta = _load_stage1_ctx_pred_into(bundle, str(path))
    assert meta["n_ctx_ee_w"] == N_H_EE_SYN
    assert meta["n_pred_ee_w"] == N_H_EE_SYN
    assert meta["n_ctx_pred_w"] == N_CTX_PRED_SYN
    with np.load(path, allow_pickle=False) as data:
        assert np.allclose(bundle.ctx.ee.w[:], data["ctx_ee_w_final"])
        assert np.allclose(bundle.pred.ee.w[:], data["pred_ee_w_final"])
        assert np.allclose(bundle.ctx_pred.w[:], data["W_ctx_pred_final"])
    return meta


def _assert_frozen_runtime_loads(ckpt_dir: Path) -> dict:
    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)
    bundle = build_frozen_network(
        architecture="ctx_pred",
        seed=SEED,
        ckpt_dir=str(ckpt_dir),
        with_cue=False,
        with_v1_to_h="context_only",
        with_feedback_routes=True,
    )
    assert bundle.meta["architecture"] == "ctx_pred"
    assert bundle.meta["ctx_pred_config_source"] == "checkpoint"
    assert bundle.meta["n_ctx_ee_w"] == N_H_EE_SYN
    assert bundle.meta["n_pred_ee_w"] == N_H_EE_SYN
    assert bundle.meta["n_ctx_pred_w"] == N_CTX_PRED_SYN
    return dict(bundle.meta)


def _assert_generated_schedule_metadata() -> dict:
    schedule = build_small_generated_stage1_schedule(seed=SEED, n_trials=12)
    pairs = np.asarray(schedule["pairs"], dtype=np.int32)
    leader = np.asarray(schedule["leader_idx"], dtype=np.int32)
    trailer = np.asarray(schedule["trailer_idx"], dtype=np.int32)
    expected = np.asarray(schedule["expected_trailer_idx"], dtype=np.int32)
    is_expected = np.asarray(schedule["is_expected"], dtype=np.bool_)
    leader_cells = np.asarray(schedule["leader_pre_cells"], dtype=np.int32)
    trailer_cells = np.asarray(schedule["trailer_post_cells"], dtype=np.int32)
    assert pairs.shape == (12, 2)
    assert np.array_equal(pairs[:, 0], leader)
    assert np.array_equal(pairs[:, 1], trailer)
    assert expected.shape == (12,)
    assert is_expected.shape == (12,)
    assert np.all(leader != trailer)
    assert np.all((leader_cells >= 0) & (leader_cells < 192))
    assert np.all((trailer_cells >= 0) & (trailer_cells < 192))
    assert np.array_equal(leader_cells // 32, leader)
    assert np.array_equal(trailer_cells // 32, trailer)
    return schedule


def main() -> int:
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)

    with tempfile.TemporaryDirectory(prefix="native_stage1_schema_") as tmp_s:
        ckpt_dir = Path(tmp_s)
        stage0_path = _write_stage0_checkpoint(ckpt_dir)
        stage1_path, native_result = write_tiny_trainer_stage1_checkpoint(
            ckpt_dir / f"stage_1_ctx_pred_seed{SEED}.npz",
            seed=SEED,
            schedule_variant=0,
        )
        _assert_stage1_npz_schema(stage1_path, native_result)
        direct_meta = _assert_direct_loader(stage1_path)
        runtime_meta = _assert_frozen_runtime_loads(ckpt_dir)
        schedule = _assert_generated_schedule_metadata()

        with np.load(stage1_path, allow_pickle=False) as data:
            gate_dw_sum = np.asarray(data["gate_dw_sum"], dtype=np.float64)
            n_trials = int(data["n_trials"])
            passed = bool(data["passed"])
            placeholder = bool(data["native_placeholder_h_recurrent_arrays"])

        print(
            "validate_native_stage1_checkpoint_schema: PASS",
            f"backend_info={backend_info()}",
            f"stage0={stage0_path}",
            f"stage1={stage1_path}",
            f"n_trials={n_trials}",
            f"passed={passed}",
            f"placeholder_h_recurrent={placeholder}",
            f"gate_dw_sum={gate_dw_sum.tolist()}",
            f"direct_loader={direct_meta}",
            f"runtime_n_ctx_pred={runtime_meta['n_ctx_pred_w']}",
            f"generated_schedule_n_trials={schedule['n_trials']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
