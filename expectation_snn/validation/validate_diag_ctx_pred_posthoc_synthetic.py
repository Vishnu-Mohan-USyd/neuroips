"""Validate diag ctx_pred Richter posthoc counting with synthetic checkpoints.

This is a construction/lifecycle smoke for
``scripts.diag_ctx_pred_richter_balance.run_condition``. It uses temporary
Stage-0 and Stage-1 ctx_pred checkpoints with deterministic shapes and config
metadata, then runs the diagnostic in ``count_mode='posthoc'`` under both
NumPy and local ``cpp_standalone`` backends by default. Pass
``--backend cuda`` on a Brian2CUDA host to run the same synthetic diagnostic
under CUDA standalone.

Synthetic checkpoint values are not scientifically meaningful.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from expectation_snn.brian2_model.h_context_prediction import (
    H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
    HContextPredictionConfig,
    h_context_prediction_config_to_json,
)
from expectation_snn.brian2_model.h_ring import (
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER_CHANNEL,
)
from expectation_snn.brian2_model.v1_ring import (
    N_E_PER_CHANNEL as V1_N_E_PER_CHANNEL,
    N_CHANNELS as V1_N_CHANNELS,
    N_PV_POOL,
)
from scripts.diag_ctx_pred_richter_balance import _make_schedule, run_condition


SEED = 42
MODULE = "expectation_snn.validation.validate_diag_ctx_pred_posthoc_synthetic"
DEFAULT_BACKENDS = ("numpy", "cpp_standalone")
BACKENDS = ("numpy", "cpp_standalone", "cuda", "cuda_standalone")


def _expected_condition_backend(backend: str) -> str:
    return "cuda_standalone" if backend == "cuda" else backend


def _write_synthetic_checkpoints(ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_v1_e = V1_N_CHANNELS * V1_N_E_PER_CHANNEL
    n_h_e = H_N_CHANNELS * H_N_E_PER_CHANNEL
    n_v1_pv_to_e = N_PV_POOL * n_v1_e
    n_h_ee = n_h_e * (n_h_e - 1)
    n_ctx_pred = n_h_e * n_h_e

    cfg = HContextPredictionConfig(
        drive_amp_ctx_pred_pA=25.0,
        pred_e_uniform_bias_pA=0.0,
    )
    cfg_json = h_context_prediction_config_to_json(cfg)

    np.savez(
        ckpt_dir / f"stage_0_seed{SEED}.npz",
        bias_pA=np.float64(0.0),
        pv_to_e_w=np.zeros((n_v1_pv_to_e,), dtype=np.float64),
    )
    np.savez(
        ckpt_dir / f"stage_1_ctx_pred_seed{SEED}.npz",
        ctx_ee_w_final=np.zeros((n_h_ee,), dtype=np.float64),
        pred_ee_w_final=np.zeros((n_h_ee,), dtype=np.float64),
        W_ctx_pred_final=np.full((n_ctx_pred,), 0.005, dtype=np.float64),
        ctx_pred_config_schema_version=np.int32(
            H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
        ),
        ctx_pred_config_json=np.bytes_(cfg_json),
        ctx_pred_drive_amp_ctx_pred_pA=np.float64(cfg.drive_amp_ctx_pred_pA),
        ctx_pred_pred_e_uniform_bias_pA=np.float64(cfg.pred_e_uniform_bias_pA),
    )


def _run_worker(backend: str, ckpt_dir: Path) -> int:
    schedule = _make_schedule(n_trials=2, seed=SEED)
    result = run_condition(
        seed=SEED,
        ckpt_dir=ckpt_dir,
        r=1.0,
        g_total=0.0,
        feedback_routes=True,
        v1_to_h_mode="context_only",
        ctx_pred_drive_pA=None,
        pred_bias_pA=0.0,
        schedule=schedule,
        leader_ms=2.0,
        trailer_ms=1.0,
        iti_ms=0.5,
        preprobe_window_ms=1.0,
        contrast=0.0,
        count_mode="posthoc",
    )

    raw = result["raw"]
    condition = result["condition"]
    assert condition["backend"] == _expected_condition_backend(backend), condition
    assert condition["count_mode"] == "posthoc", condition
    assert raw["leader_windows_ms"].shape == (2, 2)
    assert raw["preprobe_windows_ms"].shape == (2, 2)
    assert raw["trailer_windows_ms"].shape == (2, 2)
    assert raw["trial_windows_ms"].shape == (2, 2)
    assert raw["leader_counts_e"].shape == (V1_N_CHANNELS * V1_N_E_PER_CHANNEL, 2)
    assert raw["trailer_counts_e"].shape == (V1_N_CHANNELS * V1_N_E_PER_CHANNEL, 2)
    assert raw["trailer_counts_som"].shape == (V1_N_CHANNELS * 4, 2)
    assert raw["trailer_counts_pv"].shape == (N_PV_POOL, 2)
    assert raw["h_ctx_preprobe_rate_hz"].shape == (2, H_N_CHANNELS)
    assert raw["h_pred_preprobe_rate_hz"].shape == (2, H_N_CHANNELS)
    assert raw["h_ctx_trailer_rate_hz"].shape == (2, H_N_CHANNELS)
    assert raw["h_pred_trailer_rate_hz"].shape == (2, H_N_CHANNELS)
    assert np.all(np.diff(raw["leader_windows_ms"], axis=1).ravel() > 0.0)
    assert np.all(np.diff(raw["preprobe_windows_ms"], axis=1).ravel() > 0.0)
    assert np.all(np.diff(raw["trailer_windows_ms"], axis=1).ravel() > 0.0)
    assert np.all(np.diff(raw["trial_windows_ms"], axis=1).ravel() > 0.0)

    print(
        "validate_diag_ctx_pred_posthoc_synthetic worker:",
        f"backend={backend}",
        f"standalone_dir={condition['standalone_dir']}",
        f"leader_counts_shape={raw['leader_counts_e'].shape}",
        f"h_pred_shape={raw['h_pred_preprobe_rate_hz'].shape}",
    )
    return 0


def _run_backend_subprocess(backend: str, root: Path) -> None:
    ckpt_dir = root / f"{backend}_ckpt"
    standalone_dir = root / f"{backend}_standalone"
    _write_synthetic_checkpoints(ckpt_dir)

    env = os.environ.copy()
    env["EXPECTATION_SNN_BACKEND"] = backend
    if backend == "numpy":
        env.pop("EXPECTATION_SNN_STANDALONE_DIR", None)
    else:
        env["EXPECTATION_SNN_STANDALONE_DIR"] = str(standalone_dir)

    cmd = [
        sys.executable,
        "-m",
        MODULE,
        "--worker",
        "--backend",
        backend,
        "--ckpt-dir",
        str(ckpt_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"{MODULE} worker failed for backend={backend} "
            f"with exit code {completed.returncode}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help=(
            "Run one backend. Without this option, runs NumPy and local "
            "cpp_standalone as a CUDA lifecycle stand-in."
        ),
    )
    parser.add_argument("--ckpt-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.worker:
        if args.backend is None or args.ckpt_dir is None:
            raise ValueError("--worker requires --backend and --ckpt-dir")
        return _run_worker(args.backend, args.ckpt_dir)

    with tempfile.TemporaryDirectory(prefix="diag_ctx_pred_posthoc_") as tmp_s:
        root = Path(tmp_s)
        backends = (args.backend,) if args.backend is not None else DEFAULT_BACKENDS
        for backend in backends:
            _run_backend_subprocess(backend, root)
    print("validate_diag_ctx_pred_posthoc_synthetic: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
