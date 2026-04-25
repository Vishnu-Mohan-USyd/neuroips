"""Checkpoint helpers for native CUDA Stage-1 ctx_pred training.

This module bridges the native CUDA training primitives to the existing
Python/Brian2 runtime checkpoint contract.  The first writer intentionally
targets schema compatibility only: the native tiny trainer exercises
ctx_pred plasticity on controlled events, but its H recurrent arrays are
placeholders and must not be interpreted as a scientific Stage-1 pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from expectation_snn.brian2_model.h_context_prediction import (
    H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
    HContextPredictionConfig,
    h_context_prediction_config_to_json,
)
from expectation_snn.brian2_model.stimulus import (
    RICHTER_ORIENTATIONS_DEG,
    richter_biased_training_schedule,
)
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_ctx_pred_generated_schedule_test,
    run_ctx_pred_tiny_trainer_test,
    run_stage1_h_gate_dynamics_test,
)


N_H_E = 192
N_H_EE_SYN = N_H_E * (N_H_E - 1)
N_CTX_PRED_SYN = N_H_E * N_H_E
N_H_CHANNELS = 12
N_H_E_PER_CHANNEL = 16
N_ORIENTATIONS = 6
H_RICHTER_CHANNEL_STRIDE = N_H_CHANNELS // N_ORIENTATIONS
N_CELLS_PER_ORIENTATION = N_H_E_PER_CHANNEL
NATIVE_STAGE1_CTX_PRED_DRIVE_PA = 400.0
NATIVE_STAGE1_PRED_E_UNIFORM_BIAS_PA = 100.0
NATIVE_STAGE1_W_INIT_FRAC = 0.0
NATIVE_STAGE1_CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_DT_MS = 0.1
DEFAULT_NATIVE_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2] / "expectation_snn" / "data"
    / "checkpoints_native"
)
DEFAULT_NATIVE_N72_CHECKPOINT = (
    DEFAULT_NATIVE_CHECKPOINT_DIR / "stage_1_ctx_pred_seed42_native_n72.npz"
)
STABLE_HASH_EXCLUDE_KEYS = frozenset({
    "native_backend_info",
    "native_stable_content_sha256",
    "native_wall_seconds",
    "native_schedule_wall_seconds",
    "native_train_wall_seconds",
    "native_gate_eval_wall_seconds",
    "native_checkpoint_write_wall_seconds",
    "native_total_wall_seconds",
})
H_CONTEXT_PERSISTENCE_MIN_MS = 200.0
H_CONTEXT_PERSISTENCE_MAX_MS = 500.0
H_PRED_FORECAST_PROB_MIN = 0.25
NO_RUNAWAY_MAX_RATE_HZ = 80.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_bytes(payload: Mapping[str, Any]) -> np.bytes_:
    return np.bytes_(json.dumps(_jsonable(dict(payload)), sort_keys=True))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    header = f"{arr.dtype.str}:{arr.shape}".encode("utf-8")
    return _sha256_bytes(header + arr.tobytes())


def _stable_arrays_hash(
    arrays: Mapping[str, np.ndarray],
    *,
    exclude_keys: frozenset[str] = STABLE_HASH_EXCLUDE_KEYS,
) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        if key in exclude_keys:
            continue
        arr = np.ascontiguousarray(np.asarray(arrays[key]))
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(arr.dtype.str.encode("utf-8"))
        h.update(b"\0")
        h.update(str(arr.shape).encode("utf-8"))
        h.update(b"\0")
        h.update(arr.tobytes())
        h.update(b"\0")
    return h.hexdigest()


def stable_npz_content_hash(
    path: str | Path,
    *,
    exclude_keys: frozenset[str] = STABLE_HASH_EXCLUDE_KEYS,
) -> str:
    """Hash NPZ payload content while excluding runtime-volatile keys."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return _stable_arrays_hash(arrays, exclude_keys=exclude_keys)


def _write_npz_with_stable_hash(path: Path, payload: Mapping[str, object]) -> str:
    np.savez(path, **payload)
    stable_hash = stable_npz_content_hash(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["native_stable_content_sha256"] = np.bytes_(stable_hash)
    np.savez(path, **arrays)
    check_hash = stable_npz_content_hash(path)
    if check_hash != stable_hash:
        raise AssertionError(
            "stable content hash changed after embedding hash field: "
            f"{stable_hash} != {check_hash}"
        )
    return stable_hash


def _array(
    result: Mapping[str, Any],
    key: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    if key not in result:
        raise KeyError(f"native trainer result missing required key {key!r}")
    return np.asarray(result[key], dtype=dtype)


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != expected {shape}")
    return arr


def _infer_trial_orientation_indices(cells: np.ndarray) -> np.ndarray:
    """Map H-cell ids to Richter orientation bins for metadata only.

    Native generated schedules embed the six Richter orientations on the
    12-channel H ring as even H channels: orientation ``k`` maps to H channel
    ``2*k`` and each H channel has 16 excitatory cells.  Older controlled
    tiny fixtures may use arbitrary H cells; for those, return the raw H
    channel so metadata remains honest rather than inventing a 6-bin mapping.
    """
    cells = np.asarray(cells, dtype=np.int32)
    if np.any((cells < 0) | (cells >= N_H_E)):
        raise ValueError("controlled H-cell ids must be in [0, 192)")
    channels = np.floor_divide(cells, N_H_E_PER_CHANNEL).astype(np.int32)
    if np.all((channels % H_RICHTER_CHANNEL_STRIDE) == 0):
        return np.floor_divide(channels, H_RICHTER_CHANNEL_STRIDE).astype(np.int32)
    return channels


def compute_native_stage1_gate_metrics(
    native_result: Mapping[str, Any],
    schedule_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute native Stage-1 gates from H recurrent/inhibitory dynamics.

    The current trainer still uses controlled events for plasticity, but this
    gate pass runs the generated Stage-1 schedule through the native H ring
    dynamics and final ``W_ctx_pred``.  Metrics remain provisional until the
    semantics are independently accepted against the intended Brian2 Stage-1
    gate.
    """
    n_trials = int(native_result["n_trials"])
    leader_cells = _require_shape(
        "schedule leader_pre_cells",
        np.asarray(schedule_metadata["leader_pre_cells"], dtype=np.int32),
        (n_trials,),
    )
    trailer_cells = _require_shape(
        "schedule trailer_post_cells",
        np.asarray(schedule_metadata["trailer_post_cells"], dtype=np.int32),
        (n_trials,),
    )
    w_ctx_pred = _require_shape(
        "W_ctx_pred_final",
        _array(native_result, "cpu_w_ctx_pred_final", dtype=np.float64),
        (N_CTX_PRED_SYN,),
    )
    h_gate = run_stage1_h_gate_dynamics_test(
        seed=int(native_result.get("seed", 0)),
        leader_cells=leader_cells,
        trailer_cells=trailer_cells,
        w_ctx_pred=w_ctx_pred,
    )
    h_gate_errors = {
        str(k): float(v) for k, v in h_gate["max_abs_error"].items()
    }
    h_gate_max_error = max(h_gate_errors.values()) if h_gate_errors else 0.0
    if h_gate_max_error > 1e-8:
        raise AssertionError(
            f"native H gate CPU/CUDA mismatch {h_gate_max_error}: "
            f"{h_gate_errors}"
        )
    h_metrics = {str(k): float(v) for k, v in h_gate["metrics"].items()}

    h_context_persistence_ms = float(h_metrics["h_context_persistence_ms"])
    h_context_persistence_pass = (
        H_CONTEXT_PERSISTENCE_MIN_MS
        <= h_context_persistence_ms
        <= H_CONTEXT_PERSISTENCE_MAX_MS
    )
    forecast_probability = float(
        h_metrics["h_prediction_pretrailer_forecast_probability"],
    )
    forecast_pass = forecast_probability >= H_PRED_FORECAST_PROB_MIN

    no_runaway_max_rate = float(h_metrics["no_runaway_max_rate_hz"])
    no_runaway_pass = no_runaway_max_rate <= NO_RUNAWAY_MAX_RATE_HZ

    threshold_all_pass = bool(
        h_context_persistence_pass and forecast_pass and no_runaway_pass
    )
    provisional = True
    ctx_rate = float(h_metrics["ctx_max_cell_rate_hz"])
    pred_rate = float(h_metrics["pred_max_cell_rate_hz"])
    ctx_channel_rate = float(h_metrics["ctx_max_channel_rate_hz"])
    pred_channel_rate = float(h_metrics["pred_max_channel_rate_hz"])
    no_runaway_max_cell_rate = float(
        h_metrics.get("no_runaway_max_cell_rate_hz", max(ctx_rate, pred_rate))
    )
    no_runaway_max_channel_rate = float(
        h_metrics.get("no_runaway_max_channel_rate_hz", max(ctx_channel_rate, pred_channel_rate))
    )
    ctx_population_rate = float(h_metrics["ctx_population_rate_hz"])
    pred_population_rate = float(h_metrics["pred_population_rate_hz"])
    ctx_inh_population_rate = float(h_metrics["ctx_inh_population_rate_hz"])
    pred_inh_population_rate = float(h_metrics["pred_inh_population_rate_hz"])
    return {
        "schema_version": 1,
        "provisional": provisional,
        "reason": (
            "native H recurrent/inhibitory dynamics gate pass over the "
            "generated Stage-1 schedule; provisional pending independent "
            "validation of exact Brian2 Stage-1 gate semantics"
        ),
        "metric_source": "native_h_recurrent_dynamics_schedule_eval",
        "h_gate_cpu_cuda_max_abs_error": h_gate_max_error,
        "h_gate_phase_steps": dict(h_gate["phase_steps"]),
        "h_gate_n_trials": int(h_gate["n_trials"]),
        "h_gate_n_steps_per_trial": int(h_gate["n_steps_per_trial"]),
        "h_gate_dt_ms": float(h_gate["dt_ms"]),
        "h_gate_leader_channels": np.asarray(
            h_gate["leader_channels"], dtype=np.int32,
        ),
        "h_gate_trailer_channels": np.asarray(
            h_gate["trailer_channels"], dtype=np.int32,
        ),
        "h_gate_ctx_persistence_ms_by_trial": np.asarray(
            h_gate["cpu_ctx_persistence_ms_by_trial"], dtype=np.float64,
        ),
        "h_gate_pred_pretrailer_target_counts": np.asarray(
            h_gate["cpu_pred_pretrailer_target_counts"], dtype=np.int32,
        ),
        "h_context_persistence_ms": h_context_persistence_ms,
        "h_context_persistence_min_ms": H_CONTEXT_PERSISTENCE_MIN_MS,
        "h_context_persistence_max_ms": H_CONTEXT_PERSISTENCE_MAX_MS,
        "h_context_persistence_pass": bool(h_context_persistence_pass),
        "h_prediction_pretrailer_forecast_probability": forecast_probability,
        "h_prediction_pretrailer_forecast_threshold": H_PRED_FORECAST_PROB_MIN,
        "h_prediction_pretrailer_forecast_pass": bool(forecast_pass),
        "no_runaway_max_rate_hz": no_runaway_max_rate,
        "no_runaway_population_max_rate_hz": float(
            h_metrics.get("no_runaway_population_max_rate_hz", no_runaway_max_rate)
        ),
        "no_runaway_max_cell_rate_hz": no_runaway_max_cell_rate,
        "no_runaway_max_channel_rate_hz": no_runaway_max_channel_rate,
        "no_runaway_threshold_hz": NO_RUNAWAY_MAX_RATE_HZ,
        "no_runaway_pass": bool(no_runaway_pass),
        "h_context_population_rate_hz": ctx_population_rate,
        "h_prediction_population_rate_hz": pred_population_rate,
        "h_context_inh_population_rate_hz": ctx_inh_population_rate,
        "h_prediction_inh_population_rate_hz": pred_inh_population_rate,
        "h_context_max_native_h_rate_hz": ctx_rate,
        "h_prediction_max_native_h_rate_hz": pred_rate,
        "h_context_max_controlled_event_rate_hz": ctx_rate,
        "h_prediction_max_controlled_event_rate_hz": pred_rate,
        "h_context_max_channel_rate_hz": ctx_channel_rate,
        "h_prediction_max_channel_rate_hz": pred_channel_rate,
        "h_prediction_pretrailer_forecast_trial_count": int(
            h_metrics["forecast_trial_count"],
        ),
        "h_prediction_pretrailer_target_spikes": int(
            h_metrics["pred_pretrailer_target_spikes"],
        ),
        "h_prediction_pretrailer_start_step": int(
            h_metrics["h_prediction_pretrailer_start_step"],
        ),
        "h_prediction_pretrailer_end_step": int(
            h_metrics["h_prediction_pretrailer_end_step"],
        ),
        "ctx_pred_gate_drive_amp_pA": float(
            h_metrics["ctx_pred_gate_drive_amp_pA"],
        ),
        "thresholds_all_pass": threshold_all_pass,
        "all_pass": bool(threshold_all_pass and not provisional),
        "schedule_pairs_sha256": schedule_metadata.get("pairs_sha256", ""),
    }


def build_small_generated_stage1_schedule(
    *,
    seed: int = 42,
    n_trials: int = 12,
    p_bias: float = 0.80,
) -> dict[str, Any]:
    """Build a Richter-biased Stage-1 schedule for native training.

    The returned schedule is metadata plus controlled H-cell event ids.  This
    helper keeps the existing schedule builder as the source of truth for
    leader/trailer pairing semantics while the native trainer consumes only
    explicit H_context/H_prediction cell ids.
    """
    rng = np.random.default_rng(int(seed))
    plan = richter_biased_training_schedule(
        rng,
        n_trials=int(n_trials),
        p_bias=float(p_bias),
    )
    pairs = np.asarray(plan.meta["pairs"], dtype=np.int32)
    expected = np.asarray(plan.meta["expected_trailer_idx"], dtype=np.int32)
    is_expected = np.asarray(plan.meta["is_expected"], dtype=np.bool_)
    if pairs.shape != (int(n_trials), 2):
        raise AssertionError(f"unexpected generated pairs shape {pairs.shape}")

    offsets = np.arange(int(n_trials), dtype=np.int32) % N_H_E_PER_CHANNEL
    leader_h_channels = pairs[:, 0] * H_RICHTER_CHANNEL_STRIDE
    trailer_h_channels = pairs[:, 1] * H_RICHTER_CHANNEL_STRIDE
    expected_trailer_h_channels = expected * H_RICHTER_CHANNEL_STRIDE
    leader_cells = leader_h_channels * N_H_E_PER_CHANNEL + offsets
    trailer_cells = trailer_h_channels * N_H_E_PER_CHANNEL + offsets
    expected_trailer_cells = expected_trailer_h_channels * N_H_E_PER_CHANNEL + offsets

    pairs_sha256 = _sha256_array(pairs)
    return {
        "schedule_name": "richter_biased_generated",
        "seed": int(seed),
        "n_trials": int(n_trials),
        "p_bias": float(p_bias),
        "orientations_deg": np.asarray(RICHTER_ORIENTATIONS_DEG, dtype=np.float64),
        "pairs_sha256": pairs_sha256,
        "pairs": pairs,
        "leader_idx": pairs[:, 0].astype(np.int32),
        "trailer_idx": pairs[:, 1].astype(np.int32),
        "expected_trailer_idx": expected,
        "is_expected": is_expected,
        "leader_pre_cells": leader_cells.astype(np.int32),
        "trailer_post_cells": trailer_cells.astype(np.int32),
        "expected_trailer_post_cells": expected_trailer_cells.astype(np.int32),
        "metadata_json": {
            "source": "richter_biased_training_schedule",
            "controlled_event_mapping": (
                "cell = (orientation_idx * 2) * 16 + trial_offset_mod_16"
            ),
            "pairs_sha256": pairs_sha256,
            "native_generated_trainer_status": "consumed_by_native_channel_window_trainer",
            "h_channels": "12 channels x 16 E cells; Richter orientations use even H channels",
            "bounded_trace_boundary": (
                "generated trainer clears xpre/xpost after each delayed gate "
                "to approximate production long-ITI coincidence-trace decay"
            ),
        },
    }


def _base_schedule_metadata(native_result: Mapping[str, Any]) -> dict[str, Any]:
    leader_cells = _array(native_result, "trial_leader_pre_cells", dtype=np.int32)
    trailer_cells = _array(native_result, "trial_trailer_post_cells", dtype=np.int32)
    n_trials = int(native_result["n_trials"])
    _require_shape("trial_leader_pre_cells", leader_cells, (n_trials,))
    _require_shape("trial_trailer_post_cells", trailer_cells, (n_trials,))
    leader_idx = _infer_trial_orientation_indices(leader_cells)
    trailer_idx = _infer_trial_orientation_indices(trailer_cells)
    return {
        "schedule_name": "native_tiny_controlled",
        "leader_idx": leader_idx,
        "trailer_idx": trailer_idx,
        "expected_trailer_idx": trailer_idx.copy(),
        "is_expected": np.ones((n_trials,), dtype=np.bool_),
        "leader_pre_cells": leader_cells,
        "trailer_post_cells": trailer_cells,
        "expected_trailer_post_cells": trailer_cells.copy(),
        "metadata_json": {
            "source": "native_fixed_tiny_trainer",
            "controlled_event_mapping": "native kernel fixed trial cells",
        },
    }


def write_native_stage1_ctx_pred_checkpoint(
    native_result: Mapping[str, Any],
    out_path: str | Path,
    *,
    seed: int | None = None,
    schedule_metadata: Mapping[str, Any] | None = None,
    ctx_pred_cfg: HContextPredictionConfig | None = None,
) -> Path:
    """Write a runtime-compatible Stage-1 ctx_pred checkpoint.

    Parameters
    ----------
    native_result:
        Result dictionary from ``run_ctx_pred_tiny_trainer_test`` or a future
        native Stage-1 trainer with the same checkpoint-shaped output keys.
    out_path:
        Destination ``.npz`` path.
    seed:
        Optional checkpoint seed metadata. Defaults to ``native_result['seed']``.
    schedule_metadata:
        Optional generated schedule metadata. If omitted, controlled tiny
        trainer trial-cell metadata is inferred from ``native_result``.
    ctx_pred_cfg:
        Optional ctx_pred config to serialize. Defaults to the current model
        defaults, matching the runtime checkpoint metadata contract.

    Returns
    -------
    pathlib.Path
        The written checkpoint path.
    """
    cfg = ctx_pred_cfg or HContextPredictionConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_seed = int(native_result["seed"] if seed is None else seed)
    n_trials = int(native_result["n_trials"])
    dt_ms = float(native_result.get("dt_ms", DEFAULT_DT_MS))
    phase_steps = dict(native_result.get("phase_steps", {}))
    event_counts = dict(native_result.get("event_counts", {}))
    schedule = (
        dict(schedule_metadata)
        if schedule_metadata is not None
        else _base_schedule_metadata(native_result)
    )
    pairs = np.asarray(
        schedule.get(
            "pairs",
            np.column_stack((schedule["leader_idx"], schedule["trailer_idx"])),
        ),
        dtype=np.int32,
    )
    if pairs.shape != (n_trials, 2):
        raise ValueError(f"schedule pairs shape {pairs.shape} != {(n_trials, 2)}")
    pairs_sha256 = str(schedule.get("pairs_sha256", _sha256_array(pairs)))

    W_ctx_pred = _require_shape(
        "W_ctx_pred_final",
        _array(native_result, "cpu_w_ctx_pred_final", dtype=np.float64),
        (N_CTX_PRED_SYN,),
    )
    W_ctx_pred_init = _require_shape(
        "W_ctx_pred_init",
        _array(native_result, "initial_w_ctx_pred", dtype=np.float64),
        (N_CTX_PRED_SYN,),
    )
    ctx_ee = _require_shape(
        "ctx_ee_w_final",
        _array(native_result, "cpu_ctx_ee_w_final", dtype=np.float64),
        (N_H_EE_SYN,),
    )
    pred_ee = _require_shape(
        "pred_ee_w_final",
        _array(native_result, "cpu_pred_ee_w_final", dtype=np.float64),
        (N_H_EE_SYN,),
    )
    elig = _require_shape(
        "elig_final",
        np.asarray(
            native_result.get("cpu_elig_after_training", np.zeros(N_CTX_PRED_SYN)),
            dtype=np.float64,
        ),
        (N_CTX_PRED_SYN,),
    )

    gate_w_before = _array(native_result, "cpu_gate_w_before", dtype=np.float64)
    gate_w_after = _array(native_result, "cpu_gate_w_after", dtype=np.float64)
    gate_dw_sum = _array(native_result, "cpu_gate_dw_sum", dtype=np.float64)
    gate_elig_mean = _array(native_result, "cpu_gate_elig_mean", dtype=np.float64)
    gate_elig_max = _array(native_result, "cpu_gate_elig_max", dtype=np.float64)
    gate_n_capped = _array(native_result, "cpu_gate_n_capped", dtype=np.int32)
    gate_row_sum_max = _array(
        native_result,
        "cpu_gate_row_sum_max",
        dtype=np.float64,
    )
    gate_steps = _array(native_result, "gate_steps", dtype=np.int32)
    n_gate = int(gate_steps.shape[0])
    for name, arr in (
        ("gate_w_before", gate_w_before),
        ("gate_w_after", gate_w_after),
        ("gate_dw_sum", gate_dw_sum),
        ("gate_elig_mean", gate_elig_mean),
        ("gate_elig_max", gate_elig_max),
        ("gate_n_capped", gate_n_capped),
        ("gate_row_sum_max", gate_row_sum_max),
    ):
        _require_shape(name, arr, (n_gate,))

    max_abs_error = native_result.get("max_abs_error", {})
    row_sums = _require_shape(
        "row_sums",
        _array(native_result, "cpu_row_sums", dtype=np.float64),
        (N_H_E,),
    )
    delta = W_ctx_pred - W_ctx_pred_init
    max_cpu_cuda_error = (
        max(float(v) for v in max_abs_error.values())
        if max_abs_error
        else float("nan")
    )
    finite_ok = bool(
        np.all(np.isfinite(W_ctx_pred))
        and np.all(np.isfinite(ctx_ee))
        and np.all(np.isfinite(pred_ee))
        and np.all(np.isfinite(gate_dw_sum))
        and np.all(np.isfinite(row_sums))
    )
    row_cap_limit = 3.0
    row_cap_ok = bool(float(row_sums.max()) <= row_cap_limit + 5e-12)
    gate_t0 = time.perf_counter()
    gate_metrics = compute_native_stage1_gate_metrics(native_result, schedule)
    native_gate_wall = float(time.perf_counter() - gate_t0)
    native_scientific_passed = bool(gate_metrics["all_pass"])
    native_schedule_wall = float(native_result.get("native_schedule_wall_seconds", 0.0))
    native_train_wall = float(
        native_result.get("native_train_wall_seconds", native_result.get("native_wall_seconds", 0.0))
    )
    report = {
        "artifact_kind": "native_stage1_ctx_pred_schema_checkpoint",
        "scientific_stage1_passed": native_scientific_passed,
        "schema_checkpoint_written": True,
        "h_recurrent_arrays": (
            "placeholder arrays from native tiny trainer, not scientific H "
            "recurrent training output"
        ),
        "ctx_pred_weights": (
            "controlled native CUDA tiny trainer output; suitable for schema "
            "and runtime-loading validation only"
        ),
        "max_abs_error": _jsonable(max_abs_error),
        "finite_ok": finite_ok,
        "row_cap_ok": row_cap_ok,
        "max_cpu_cuda_error": max_cpu_cuda_error,
        "schedule_pairs_sha256": pairs_sha256,
        "gate_metrics": gate_metrics,
    }
    cfg_json = h_context_prediction_config_to_json(cfg)
    payload = dict(
        native_stage1_checkpoint_schema_version=np.int32(
            NATIVE_STAGE1_CHECKPOINT_SCHEMA_VERSION,
        ),
        grammar=np.bytes_("native_ctx_pred_controlled"),
        architecture=np.bytes_("ctx_pred"),
        schedule=np.bytes_(str(schedule.get("schedule_name", "native_controlled"))),
        seed=np.int32(checkpoint_seed),
        n_trials=np.int32(n_trials),
        dt_ms=np.float64(dt_ms),
        native_wall_seconds=np.float64(native_result.get("native_wall_seconds", 0.0)),
        native_schedule_wall_seconds=np.float64(native_schedule_wall),
        native_train_wall_seconds=np.float64(native_train_wall),
        native_gate_eval_wall_seconds=np.float64(native_gate_wall),
        native_checkpoint_write_wall_seconds=np.float64(0.0),
        native_total_wall_seconds=np.float64(0.0),
        native_backend_info=np.bytes_(str(native_result.get("native_backend_info", ""))),
        passed=np.bool_(native_scientific_passed),
        native_schema_checkpoint_written=np.bool_(True),
        native_scientific_stage1_passed=np.bool_(native_scientific_passed),
        native_placeholder_h_recurrent_arrays=np.bool_(True),
        native_schedule_variant=np.int32(native_result.get("schedule_variant", -1)),
        native_report_json=_json_bytes(report),
        native_phase_steps_json=_json_bytes(phase_steps),
        native_event_counts_json=_json_bytes(event_counts),
        native_schedule_metadata_json=_json_bytes(schedule.get("metadata_json", {})),
        native_schedule_pairs=pairs,
        native_schedule_pairs_sha256=np.bytes_(pairs_sha256),
        native_schedule_p_bias=np.float64(schedule.get("p_bias", np.nan)),
        native_schedule_orientations_deg=np.asarray(
            schedule.get("orientations_deg", RICHTER_ORIENTATIONS_DEG),
            dtype=np.float64,
        ),
        leader_idx=np.asarray(schedule["leader_idx"], dtype=np.int32),
        trailer_idx=np.asarray(schedule["trailer_idx"], dtype=np.int32),
        expected_trailer_idx=np.asarray(
            schedule["expected_trailer_idx"], dtype=np.int32,
        ),
        is_expected=np.asarray(schedule["is_expected"], dtype=np.bool_),
        native_trial_leader_pre_cells=np.asarray(
            schedule["leader_pre_cells"], dtype=np.int32,
        ),
        native_trial_trailer_post_cells=np.asarray(
            schedule["trailer_post_cells"], dtype=np.int32,
        ),
        native_trial_expected_trailer_post_cells=np.asarray(
            schedule["expected_trailer_post_cells"], dtype=np.int32,
        ),
        W_ctx_pred_init=W_ctx_pred_init,
        W_ctx_pred_final=W_ctx_pred,
        ctx_ee_w_final=ctx_ee,
        pred_ee_w_final=pred_ee,
        elig_final=elig,
        gate_k=np.arange(n_gate, dtype=np.int32),
        gate_step=gate_steps,
        gate_t_ms=gate_steps.astype(np.float64) * dt_ms,
        gate_w_before=gate_w_before,
        gate_w_after=gate_w_after,
        gate_dw_sum=gate_dw_sum,
        gate_elig_mean=gate_elig_mean,
        gate_elig_max=gate_elig_max,
        gate_n_capped=gate_n_capped,
        gate_row_sum_max=gate_row_sum_max,
        row_sum_final=row_sums,
        row_cap_limit=np.float64(row_cap_limit),
        row_cap_ok=np.bool_(row_cap_ok),
        finite_ok=np.bool_(finite_ok),
        max_cpu_cuda_error=np.float64(max_cpu_cuda_error),
        native_gate_metrics_schema_version=np.int32(gate_metrics["schema_version"]),
        native_gate_metrics_provisional=np.bool_(gate_metrics["provisional"]),
        native_gate_metrics_all_pass=np.bool_(gate_metrics["all_pass"]),
        native_gate_thresholds_all_pass=np.bool_(gate_metrics["thresholds_all_pass"]),
        native_gate_metrics_json=_json_bytes(gate_metrics),
        native_gate_metrics_source=np.bytes_(gate_metrics["metric_source"]),
        native_gate_h_dynamics_cpu_cuda_max_abs_error=np.float64(
            gate_metrics["h_gate_cpu_cuda_max_abs_error"],
        ),
        h_gate_n_trials=np.int32(gate_metrics["h_gate_n_trials"]),
        h_gate_n_steps_per_trial=np.int32(gate_metrics["h_gate_n_steps_per_trial"]),
        h_gate_dt_ms=np.float64(gate_metrics["h_gate_dt_ms"]),
        h_gate_leader_channels=np.asarray(
            gate_metrics["h_gate_leader_channels"], dtype=np.int32,
        ),
        h_gate_trailer_channels=np.asarray(
            gate_metrics["h_gate_trailer_channels"], dtype=np.int32,
        ),
        h_gate_ctx_persistence_ms_by_trial=np.asarray(
            gate_metrics["h_gate_ctx_persistence_ms_by_trial"], dtype=np.float64,
        ),
        h_gate_pred_pretrailer_target_counts=np.asarray(
            gate_metrics["h_gate_pred_pretrailer_target_counts"], dtype=np.int32,
        ),
        h_context_persistence_ms=np.float64(
            gate_metrics["h_context_persistence_ms"],
        ),
        h_context_persistence_min_ms=np.float64(
            gate_metrics["h_context_persistence_min_ms"],
        ),
        h_context_persistence_max_ms=np.float64(
            gate_metrics["h_context_persistence_max_ms"],
        ),
        h_context_persistence_pass=np.bool_(
            gate_metrics["h_context_persistence_pass"],
        ),
        h_prediction_pretrailer_forecast_probability=np.float64(
            gate_metrics["h_prediction_pretrailer_forecast_probability"],
        ),
        h_prediction_pretrailer_forecast_threshold=np.float64(
            gate_metrics["h_prediction_pretrailer_forecast_threshold"],
        ),
        h_prediction_pretrailer_forecast_pass=np.bool_(
            gate_metrics["h_prediction_pretrailer_forecast_pass"],
        ),
        no_runaway_max_rate_hz=np.float64(gate_metrics["no_runaway_max_rate_hz"]),
        no_runaway_population_max_rate_hz=np.float64(
            gate_metrics["no_runaway_population_max_rate_hz"],
        ),
        no_runaway_max_cell_rate_hz=np.float64(
            gate_metrics["no_runaway_max_cell_rate_hz"],
        ),
        no_runaway_max_channel_rate_hz=np.float64(
            gate_metrics["no_runaway_max_channel_rate_hz"],
        ),
        no_runaway_threshold_hz=np.float64(gate_metrics["no_runaway_threshold_hz"]),
        no_runaway_pass=np.bool_(gate_metrics["no_runaway_pass"]),
        h_context_population_rate_hz=np.float64(
            gate_metrics["h_context_population_rate_hz"],
        ),
        h_prediction_population_rate_hz=np.float64(
            gate_metrics["h_prediction_population_rate_hz"],
        ),
        h_context_inh_population_rate_hz=np.float64(
            gate_metrics["h_context_inh_population_rate_hz"],
        ),
        h_prediction_inh_population_rate_hz=np.float64(
            gate_metrics["h_prediction_inh_population_rate_hz"],
        ),
        h_context_max_controlled_event_rate_hz=np.float64(
            gate_metrics["h_context_max_controlled_event_rate_hz"],
        ),
        h_prediction_max_controlled_event_rate_hz=np.float64(
            gate_metrics["h_prediction_max_controlled_event_rate_hz"],
        ),
        h_context_max_native_h_rate_hz=np.float64(
            gate_metrics["h_context_max_native_h_rate_hz"],
        ),
        h_prediction_max_native_h_rate_hz=np.float64(
            gate_metrics["h_prediction_max_native_h_rate_hz"],
        ),
        h_context_max_channel_rate_hz=np.float64(
            gate_metrics["h_context_max_channel_rate_hz"],
        ),
        h_prediction_max_channel_rate_hz=np.float64(
            gate_metrics["h_prediction_max_channel_rate_hz"],
        ),
        h_prediction_pretrailer_forecast_trial_count=np.int32(
            gate_metrics["h_prediction_pretrailer_forecast_trial_count"],
        ),
        h_prediction_pretrailer_target_spikes=np.int32(
            gate_metrics["h_prediction_pretrailer_target_spikes"],
        ),
        h_prediction_pretrailer_start_step=np.int32(
            gate_metrics["h_prediction_pretrailer_start_step"],
        ),
        h_prediction_pretrailer_end_step=np.int32(
            gate_metrics["h_prediction_pretrailer_end_step"],
        ),
        ctx_pred_gate_drive_amp_pA=np.float64(
            gate_metrics["ctx_pred_gate_drive_amp_pA"],
        ),
        W_ctx_pred_mean_init=np.float64(W_ctx_pred_init.mean()),
        W_ctx_pred_mean_final=np.float64(W_ctx_pred.mean()),
        W_ctx_pred_min_final=np.float64(W_ctx_pred.min()),
        W_ctx_pred_max_final=np.float64(W_ctx_pred.max()),
        W_ctx_pred_delta_abs_sum=np.float64(np.abs(delta).sum()),
        W_ctx_pred_delta_l2=np.float64(np.linalg.norm(delta)),
        ctx_pred_config_schema_version=np.int32(
            H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
        ),
        ctx_pred_config_json=np.bytes_(cfg_json),
        ctx_pred_drive_amp_ctx_pred_pA=np.float64(cfg.drive_amp_ctx_pred_pA),
        ctx_pred_pred_e_uniform_bias_pA=np.float64(
            cfg.pred_e_uniform_bias_pA,
        ),
    )
    write_t0 = time.perf_counter()
    stable_hash = _write_npz_with_stable_hash(out_path, payload)
    checkpoint_write_wall = float(time.perf_counter() - write_t0)
    total_wall = (
        native_schedule_wall
        + native_train_wall
        + native_gate_wall
        + checkpoint_write_wall
    )
    with np.load(out_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["native_checkpoint_write_wall_seconds"] = np.float64(
        checkpoint_write_wall,
    )
    arrays["native_total_wall_seconds"] = np.float64(total_wall)
    arrays["native_wall_seconds"] = np.float64(total_wall)
    np.savez(out_path, **arrays)
    check_hash = stable_npz_content_hash(out_path)
    if check_hash != stable_hash:
        raise AssertionError(
            "stable content hash changed after timing telemetry update: "
            f"{stable_hash} != {check_hash}"
        )
    return out_path


def write_tiny_trainer_stage1_checkpoint(
    out_path: str | Path,
    *,
    seed: int = 42,
    schedule_variant: int = 0,
) -> tuple[Path, dict[str, Any]]:
    """Run the native tiny trainer and write a schema checkpoint."""
    result = run_ctx_pred_tiny_trainer_test(
        seed=int(seed),
        schedule_variant=int(schedule_variant),
    )
    path = write_native_stage1_ctx_pred_checkpoint(
        result,
        out_path,
        seed=int(seed),
    )
    return path, result


def write_generated_schedule_stage1_checkpoint(
    out_path: str | Path,
    *,
    seed: int = 42,
    n_trials: int = 12,
    p_bias: float = 0.80,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Run the native generated-schedule smoke trainer and write a checkpoint."""
    schedule_t0 = time.perf_counter()
    schedule = build_small_generated_stage1_schedule(
        seed=int(seed),
        n_trials=int(n_trials),
        p_bias=float(p_bias),
    )
    schedule_wall = float(time.perf_counter() - schedule_t0)
    native_backend = backend_info()
    train_t0 = time.perf_counter()
    result = run_ctx_pred_generated_schedule_test(
        seed=int(seed),
        leader_pre_cells=np.asarray(schedule["leader_pre_cells"], dtype=np.int32),
        trailer_post_cells=np.asarray(schedule["trailer_post_cells"], dtype=np.int32),
    )
    train_wall = float(time.perf_counter() - train_t0)
    result["native_schedule_wall_seconds"] = schedule_wall
    result["native_train_wall_seconds"] = train_wall
    result["native_gate_eval_wall_seconds"] = 0.0
    result["native_wall_seconds"] = train_wall
    result["native_backend_info"] = native_backend
    if int(result["n_trials"]) != int(schedule["n_trials"]):
        raise AssertionError(
            f"native result n_trials={result['n_trials']} does not match "
            f"schedule n_trials={schedule['n_trials']}"
        )
    ctx_pred_cfg = HContextPredictionConfig(
        drive_amp_ctx_pred_pA=NATIVE_STAGE1_CTX_PRED_DRIVE_PA,
        pred_e_uniform_bias_pA=NATIVE_STAGE1_PRED_E_UNIFORM_BIAS_PA,
        w_init_frac=NATIVE_STAGE1_W_INIT_FRAC,
    )
    path = write_native_stage1_ctx_pred_checkpoint(
        result,
        out_path,
        seed=int(seed),
        schedule_metadata=schedule,
        ctx_pred_cfg=ctx_pred_cfg,
    )
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "native_schedule_wall_seconds",
            "native_train_wall_seconds",
            "native_gate_eval_wall_seconds",
            "native_checkpoint_write_wall_seconds",
            "native_total_wall_seconds",
            "native_wall_seconds",
        ):
            result[key] = float(data[key])
    return path, result, schedule


def write_native_stage1_n72_checkpoint(
    out_path: str | Path | None = None,
    *,
    seed: int = 42,
    p_bias: float = 0.80,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Generate the bounded native n=72 Stage-1 ctx_pred checkpoint."""
    return write_generated_schedule_stage1_checkpoint(
        DEFAULT_NATIVE_N72_CHECKPOINT if out_path is None else out_path,
        seed=int(seed),
        n_trials=72,
        p_bias=float(p_bias),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate native CUDA Stage-1 ctx_pred checkpoint artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=72)
    parser.add_argument("--p-bias", type=float, default=0.80)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_NATIVE_N72_CHECKPOINT,
        help="Output .npz checkpoint path.",
    )
    args = parser.parse_args(argv)
    path, result, schedule = write_generated_schedule_stage1_checkpoint(
        args.out,
        seed=args.seed,
        n_trials=args.n_trials,
        p_bias=args.p_bias,
    )
    digest = sha256_file(path)
    gate_dw_sum = np.asarray(result["cpu_gate_dw_sum"], dtype=np.float64)
    row_sums = np.asarray(result["cpu_row_sums"], dtype=np.float64)
    with np.load(path, allow_pickle=False) as data:
        gate_all_pass = bool(data["native_gate_metrics_all_pass"])
        passed = bool(data["passed"])
        scientific_passed = bool(data["native_scientific_stage1_passed"])
    print(
        "train_stage1_native: wrote checkpoint",
        f"path={path}",
        f"sha256={digest}",
        f"stable_content_sha256={stable_npz_content_hash(path)}",
        f"seed={args.seed}",
        f"n_trials={args.n_trials}",
        f"pairs_sha256={schedule['pairs_sha256']}",
        f"native_wall_seconds={float(result['native_wall_seconds']):.6f}",
        f"gate_dw_sum_total={float(gate_dw_sum.sum()):.12e}",
        f"row_sum_max={float(row_sums.max()):.12f}",
        f"max_cpu_cuda_error={max(float(v) for v in result['max_abs_error'].values()):.3e}",
        f"gate_all_pass={gate_all_pass}",
        f"passed={passed}",
        f"native_scientific_stage1_passed={scientific_passed}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
