"""Python boundary for the native CUDA module.

The native extension intentionally does not parse NPZ files at this stage.
Python loads the manifest with NumPy and passes arrays across the pybind
boundary for C++ validation/inspection.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


class NativeModuleUnavailable(ImportError):
    """Raised when the native CUDA extension has not been built/importable."""


def _load_native_module():
    module_dir = os.environ.get("EXPECTATION_SNN_NATIVE_MODULE_DIR")
    if module_dir:
        module_path = str(Path(module_dir).expanduser())
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
    try:
        return importlib.import_module("_native_cuda")
    except ImportError as exc:
        hint = (
            "Native CUDA module '_native_cuda' is not importable. Build it with "
            "`cmake -S cpp_cuda -B <build-dir>` and "
            "`cmake --build <build-dir> --parallel 1`, then set "
            "EXPECTATION_SNN_NATIVE_MODULE_DIR=<build-dir> or add the build "
            "directory to PYTHONPATH."
        )
        raise NativeModuleUnavailable(hint) from exc


def backend_info() -> str:
    """Return a short native/CUDA runtime info string."""
    return str(_load_native_module().backend_info())


def inspect_manifest_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    """Inspect a manifest already loaded as NumPy arrays."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(_load_native_module().inspect_manifest_arrays(prepared))


def inspect_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a schema-v1 manifest NPZ in Python and inspect it natively."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return inspect_manifest_arrays(arrays)


def run_decay_test(
    population: str,
    *,
    n_steps: int = 10,
    threshold_case: bool = False,
) -> dict[str, Any]:
    """Run the native CPU/CUDA deterministic neuron decay primitive."""
    return dict(
        _load_native_module().run_decay_test(
            str(population), int(n_steps), bool(threshold_case),
        )
    )


def run_h_ring_dynamics_test(*, seed: int = 42) -> dict[str, Any]:
    """Run bounded native H recurrent/inhibitory dynamics with CPU/CUDA parity."""
    return dict(_load_native_module().run_h_ring_dynamics_test(int(seed)))


def run_csr_scatter_test(
    arrays: dict[str, Any],
    *,
    bank_name: str,
    pre_index: int = 0,
) -> dict[str, Any]:
    """Run one-source-spike CSR scatter for an exported synapse bank."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_csr_scatter_test(
            prepared, str(bank_name), int(pre_index),
        )
    )


def run_event_ordering_slice(
    arrays: dict[str, Any],
    *,
    bank_name: str,
    pre_index: int = 0,
) -> dict[str, Any]:
    """Run the deterministic one-event forward-ordering slice."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_event_ordering_slice(
            prepared, str(bank_name), int(pre_index),
        )
    )


def run_ctx_to_pred_count_test(
    arrays: dict[str, Any],
    *,
    bank_name: str = "ctx_to_pred",
    pre_index: int = 7,
    event_steps: list[int] | tuple[int, ...] = (2, 3),
    n_steps: int = 20,
    window_start_step: int = 5,
    window_end_step: int = 10,
) -> dict[str, Any]:
    """Run deterministic H_ctx forced-spike to H_pred count-window primitive."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_ctx_to_pred_count_test(
            prepared,
            str(bank_name),
            int(pre_index),
            [int(step) for step in event_steps],
            int(n_steps),
            int(window_start_step),
            int(window_end_step),
        )
    )


def run_feedback_v1_count_test(
    arrays: dict[str, Any],
    *,
    direct_bank_name: str = "fb_pred_to_v1e_apical",
    som_bank_name: str = "fb_pred_to_v1som",
    pre_index: int = 7,
    event_steps: list[int] | tuple[int, ...] = (2, 3),
    n_steps: int = 20,
    window_start_step: int = 5,
    window_end_step: int = 10,
) -> dict[str, Any]:
    """Run deterministic H_pred forced-spike to V1 feedback count primitive."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_feedback_v1_count_test(
            prepared,
            str(direct_bank_name),
            str(som_bank_name),
            int(pre_index),
            [int(step) for step in event_steps],
            int(n_steps),
            int(window_start_step),
            int(window_end_step),
        )
    )


def run_v1_stim_feedforward_count_test(
    arrays: dict[str, Any],
    *,
    stim_bank_name: str = "v1_stim_to_e",
    feedforward_bank_name: str = "v1_to_h_ctx",
    stim_pre_index: int = 0,
    stim_event_steps: list[int] | tuple[int, ...] = (2,),
    force_v1e_step: int = 4,
    n_steps: int = 20,
    window_start_step: int = 4,
    window_end_step: int = 10,
) -> dict[str, Any]:
    """Run deterministic stimulus->V1 and V1->H feedforward count primitive."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_v1_stim_feedforward_count_test(
            prepared,
            str(stim_bank_name),
            str(feedforward_bank_name),
            int(stim_pre_index),
            [int(step) for step in stim_event_steps],
            int(force_v1e_step),
            int(n_steps),
            int(window_start_step),
            int(window_end_step),
        )
    )


def run_closed_loop_deterministic_count_test(
    arrays: dict[str, Any],
    *,
    stim_pre_index: int = 0,
    stim_step: int = 2,
    v1_force_step: int = 4,
    hctx_force_step: int = 26,
    hpred_force_step: int = 28,
    n_steps: int = 35,
    window_start_step: int = 4,
    window_end_step: int = 32,
) -> dict[str, Any]:
    """Run the deterministic V1->H_ctx->H_pred->V1 closed-loop primitive."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_closed_loop_deterministic_count_test(
            prepared,
            int(stim_pre_index),
            int(stim_step),
            int(v1_force_step),
            int(hctx_force_step),
            int(hpred_force_step),
            int(n_steps),
            int(window_start_step),
            int(window_end_step),
        )
    )


def run_frozen_richter_deterministic_trial_test(
    arrays: dict[str, Any],
    *,
    expected_stim_pre_index: int = 0,
    unexpected_stim_pre_index: int = 20,
    stim_period_steps: int = 5,
    n_steps: int = 120,
    leader_start_step: int = 0,
    leader_end_step: int = 30,
    preprobe_start_step: int = 30,
    preprobe_end_step: int = 60,
    trailer_start_step: int = 60,
    trailer_end_step: int = 100,
    iti_start_step: int = 100,
    iti_end_step: int = 120,
) -> dict[str, Any]:
    """Run the deterministic bounded frozen-Richter trial scheduler primitive."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_frozen_richter_deterministic_trial_test(
            prepared,
            int(expected_stim_pre_index),
            int(unexpected_stim_pre_index),
            int(stim_period_steps),
            int(n_steps),
            int(leader_start_step),
            int(leader_end_step),
            int(preprobe_start_step),
            int(preprobe_end_step),
            int(trailer_start_step),
            int(trailer_end_step),
            int(iti_start_step),
            int(iti_end_step),
        )
    )


def run_frozen_richter_seeded_source_test(
    arrays: dict[str, Any],
    *,
    seed: int = 12345,
    expected_channel: int = 0,
    unexpected_channel: int = 1,
    grating_rate_hz: float = 500.0,
    baseline_rate_hz: float = 0.0,
    n_steps: int = 120,
    leader_start_step: int = 0,
    leader_end_step: int = 30,
    preprobe_start_step: int = 30,
    preprobe_end_step: int = 60,
    trailer_start_step: int = 60,
    trailer_end_step: int = 100,
    iti_start_step: int = 100,
    iti_end_step: int = 120,
) -> dict[str, Any]:
    """Run seeded source generation with CPU/CUDA parity and diagnostic counts."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_frozen_richter_seeded_source_test(
            prepared,
            int(seed),
            int(expected_channel),
            int(unexpected_channel),
            float(grating_rate_hz),
            float(baseline_rate_hz),
            int(n_steps),
            int(leader_start_step),
            int(leader_end_step),
            int(preprobe_start_step),
            int(preprobe_end_step),
            int(trailer_start_step),
            int(trailer_end_step),
            int(iti_start_step),
            int(iti_end_step),
        )
    )


def run_frozen_richter_controlled_source_test(
    arrays: dict[str, Any],
    *,
    event_steps: list[int] | tuple[int, ...],
    event_sources: list[int] | tuple[int, ...],
    expected_channel: int = 0,
    unexpected_channel: int = 1,
    n_steps: int = 120,
    leader_start_step: int = 0,
    leader_end_step: int = 30,
    preprobe_start_step: int = 30,
    preprobe_end_step: int = 60,
    trailer_start_step: int = 60,
    trailer_end_step: int = 100,
    iti_start_step: int = 100,
    iti_end_step: int = 120,
) -> dict[str, Any]:
    """Run bounded frozen-Richter with explicit stimulus source events."""
    prepared = {str(k): np.asarray(v) for k, v in arrays.items()}
    return dict(
        _load_native_module().run_frozen_richter_controlled_source_test(
            prepared,
            [int(step) for step in event_steps],
            [int(source) for source in event_sources],
            int(expected_channel),
            int(unexpected_channel),
            int(n_steps),
            int(leader_start_step),
            int(leader_end_step),
            int(preprobe_start_step),
            int(preprobe_end_step),
            int(trailer_start_step),
            int(trailer_end_step),
            int(iti_start_step),
            int(iti_end_step),
        )
    )


def run_ctx_pred_plasticity_test(
    *,
    seed: int = 42,
    n_steps: int = 640,
) -> dict[str, Any]:
    """Run native CPU/CUDA ctx_pred eligibility and delayed gate primitive."""
    return dict(
        _load_native_module().run_ctx_pred_plasticity_test(
            int(seed),
            int(n_steps),
        )
    )


def run_ctx_pred_training_trial_slice_test(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Run controlled native Stage1 ctx_pred trial-slice training primitive."""
    return dict(
        _load_native_module().run_ctx_pred_training_trial_slice_test(int(seed))
    )


def run_ctx_pred_tiny_trainer_test(
    *,
    seed: int = 42,
    schedule_variant: int = 0,
) -> dict[str, Any]:
    """Run controlled native multi-trial Stage1 ctx_pred trainer primitive."""
    return dict(
        _load_native_module().run_ctx_pred_tiny_trainer_test(
            int(seed),
            int(schedule_variant),
        )
    )


def run_ctx_pred_generated_schedule_test(
    *,
    seed: int = 42,
    leader_pre_cells: list[int] | tuple[int, ...] | np.ndarray,
    trailer_post_cells: list[int] | tuple[int, ...] | np.ndarray,
) -> dict[str, Any]:
    """Run native Stage1 ctx_pred training on explicit generated trial cells."""
    return dict(
        _load_native_module().run_ctx_pred_generated_schedule_test(
            int(seed),
            [int(cell) for cell in np.asarray(leader_pre_cells).ravel()],
            [int(cell) for cell in np.asarray(trailer_post_cells).ravel()],
        )
    )


def run_stage1_h_gate_dynamics_test(
    *,
    seed: int = 42,
    leader_cells: list[int] | tuple[int, ...] | np.ndarray,
    trailer_cells: list[int] | tuple[int, ...] | np.ndarray,
    w_ctx_pred: list[float] | tuple[float, ...] | np.ndarray,
) -> dict[str, Any]:
    """Run native H recurrent/inhibitory gate metrics over a Stage-1 schedule."""
    return dict(
        _load_native_module().run_stage1_h_gate_dynamics_test(
            int(seed),
            [int(cell) for cell in np.asarray(leader_cells).ravel()],
            [int(cell) for cell in np.asarray(trailer_cells).ravel()],
            [float(w) for w in np.asarray(w_ctx_pred, dtype=np.float64).ravel()],
        )
    )
