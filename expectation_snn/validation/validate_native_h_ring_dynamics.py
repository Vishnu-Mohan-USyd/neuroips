"""Validate bounded native H recurrent/inhibitory dynamics.

This validator exercises a deterministic H_context/H_prediction ring primitive
that includes H E cells, H inhibitory cells, generated recurrent E->E
AMPA/NMDA, generated E->I/I->E routing, and online gate-relevant counts.  It
does not mark Stage-1 checkpoints scientifically passed; it verifies that the
bounded native dynamics can produce measurable persistence/teacher drive while
preserving CPU/CUDA parity.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from expectation_snn.cuda_sim.native import backend_info, run_h_ring_dynamics_test
from expectation_snn.cuda_sim.train_stage1_native import (
    H_CONTEXT_PERSISTENCE_MAX_MS,
    H_CONTEXT_PERSISTENCE_MIN_MS,
    NO_RUNAWAY_MAX_RATE_HZ,
)


SEED = 42
N_H_E = 192
N_H_INH = 16
DT_MS = 0.1
TOL = 1e-8


def _sum(result: dict[str, Any], key: str) -> int:
    return int(np.asarray(result[key], dtype=np.int32).sum())


def _assert_equal_counts(result: dict[str, Any], stem: str) -> None:
    cpu = np.asarray(result[f"cpu_{stem}"], dtype=np.int32)
    cuda = np.asarray(result[f"cuda_{stem}"], dtype=np.int32)
    assert cpu.shape == cuda.shape, (stem, cpu.shape, cuda.shape)
    assert np.array_equal(cpu, cuda), (stem, int(np.max(np.abs(cpu - cuda))))


def _max_error(result: dict[str, Any]) -> float:
    return max(float(value) for value in result["max_abs_error"].values())


def main() -> int:
    result = run_h_ring_dynamics_test(seed=SEED)
    repeat = run_h_ring_dynamics_test(seed=SEED)

    assert int(result["seed"]) == SEED
    assert int(result["n_e"]) == N_H_E
    assert int(result["n_inh"]) == N_H_INH
    assert np.isclose(float(result["dt_ms"]), DT_MS, atol=0.0, rtol=0.0)
    assert int(result["n_steps"]) > 0
    assert result["phase_steps"] == repeat["phase_steps"]

    for stem in (
        "ctx_leader_counts",
        "ctx_persistence_counts",
        "ctx_late_counts",
        "ctx_total_counts",
        "pred_leader_counts",
        "pred_pretrailer_counts",
        "pred_trailer_counts",
        "pred_total_counts",
        "ctx_inh_total_counts",
        "pred_inh_total_counts",
    ):
        _assert_equal_counts(result, stem)
        _assert_equal_counts(repeat, stem)
        assert np.array_equal(
            np.asarray(result[f"cpu_{stem}"], dtype=np.int32),
            np.asarray(repeat[f"cpu_{stem}"], dtype=np.int32),
        ), stem

    max_err = _max_error(result)
    repeat_max_err = _max_error(repeat)
    assert max_err <= TOL, result["max_abs_error"]
    assert repeat_max_err <= TOL, repeat["max_abs_error"]

    metrics = {str(k): float(v) for k, v in result["metrics"].items()}
    repeat_metrics = {str(k): float(v) for k, v in repeat["metrics"].items()}
    assert metrics == repeat_metrics

    ctx_leader = _sum(result, "cpu_ctx_leader_counts")
    ctx_persistence = _sum(result, "cpu_ctx_persistence_counts")
    pred_leader = _sum(result, "cpu_pred_leader_counts")
    pred_pretrailer = _sum(result, "cpu_pred_pretrailer_counts")
    pred_trailer = _sum(result, "cpu_pred_trailer_counts")
    ctx_inh = _sum(result, "cpu_ctx_inh_total_counts")
    pred_inh = _sum(result, "cpu_pred_inh_total_counts")

    assert ctx_leader > 0, metrics
    assert ctx_persistence > 0, metrics
    assert ctx_inh > 0, metrics
    assert pred_leader == 0, metrics
    assert pred_pretrailer == 0, metrics
    assert pred_trailer > 0, metrics
    assert pred_inh > 0, metrics
    assert metrics["pred_silent_leader_pass"] == 1.0, metrics
    assert metrics["pred_trailer_driven_pass"] == 1.0, metrics
    assert metrics["no_runaway_pass"] == 1.0, metrics
    assert metrics["max_rate_hz"] <= NO_RUNAWAY_MAX_RATE_HZ, metrics

    # This bounded primitive is meant to expose whether the generated H ring
    # can support Stage-1-like persistence.  If this fails, the measured metric
    # is a scientific/modeling decision point rather than a reason to mark a
    # checkpoint passed.
    ctx_persistence_ms = metrics["ctx_persistence_ms"]
    assert (
        H_CONTEXT_PERSISTENCE_MIN_MS
        <= ctx_persistence_ms
        <= H_CONTEXT_PERSISTENCE_MAX_MS
    ), metrics
    assert metrics["ctx_persistence_window_pass"] == 1.0, metrics

    print(
        "validate_native_h_ring_dynamics: PASS",
        f"backend_info={backend_info()}",
        f"max_error={max_err:.3e}",
        f"ctx_leader_spikes={ctx_leader}",
        f"ctx_persistence_spikes={ctx_persistence}",
        f"ctx_persistence_ms={ctx_persistence_ms:.3f}",
        f"pred_trailer_spikes={pred_trailer}",
        f"max_rate_hz={metrics['max_rate_hz']:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
