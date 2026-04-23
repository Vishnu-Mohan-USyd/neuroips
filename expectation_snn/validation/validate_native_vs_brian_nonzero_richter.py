"""Validate nonzero Brian/native Richter count parity under controlled forcing.

This is an engineering equivalence smoke, not a claim that the native simulator
already reproduces spontaneous Brian2 stimulus-driven threshold crossings.  It
uses the real exported stimulus/feedforward/feedback banks and explicit V1
stimulus afferent events, but the nonzero spikes are produced by matching
forced-threshold voltage injections in Brian2 and in the native deterministic
trial primitive.  The purpose is to validate raw count arrays, [start,end)
window semantics, and per-cell rates when spikes are present.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from brian2 import (
    Hz,
    Network,
    NetworkOperation,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    mV,
    ms,
    pA,
    prefs,
    start_scope,
)
from brian2 import seed as brian_seed

from expectation_snn.assays.runtime import (
    build_frozen_network,
    count_spikes_in_window,
)
from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import run_frozen_richter_deterministic_trial_test
from expectation_snn.cuda_sim.richter_native import RAW_COUNT_KEYS
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-10
EXPECTED_STIM_PRE_INDEX = 0
UNEXPECTED_STIM_PRE_INDEX = 20
STIM_PERIOD_STEPS = 5
PHASE_KWARGS = dict(
    n_steps=120,
    leader_start_step=0,
    leader_end_step=30,
    preprobe_start_step=30,
    preprobe_end_step=60,
    trailer_start_step=60,
    trailer_end_step=100,
    iti_start_step=100,
    iti_end_step=120,
)


def _phase_event_steps(start_step: int, end_step: int) -> list[int]:
    first = start_step + 2
    return list(range(first, end_step, STIM_PERIOD_STEPS))


def _source_events() -> tuple[np.ndarray, np.ndarray]:
    leader = _phase_event_steps(
        PHASE_KWARGS["leader_start_step"],
        PHASE_KWARGS["leader_end_step"],
    )
    preprobe = _phase_event_steps(
        PHASE_KWARGS["preprobe_start_step"],
        PHASE_KWARGS["preprobe_end_step"],
    )
    trailer = _phase_event_steps(
        PHASE_KWARGS["trailer_start_step"],
        PHASE_KWARGS["trailer_end_step"],
    )
    steps = np.asarray(leader + preprobe + trailer, dtype=np.int32)
    sources = np.asarray(
        [EXPECTED_STIM_PRE_INDEX] * (len(leader) + len(preprobe))
        + [UNEXPECTED_STIM_PRE_INDEX] * len(trailer),
        dtype=np.int32,
    )
    return steps, sources


def _as_count(result: dict, side: str, key: str) -> np.ndarray:
    return np.asarray(result[f"{side}_raw_counts"][key], dtype=np.int32)


def _window_steps(key: str) -> tuple[int, int]:
    phase = key.split(".")[-1]
    return (
        PHASE_KWARGS[f"{phase}_start_step"],
        PHASE_KWARGS[f"{phase}_end_step"],
    )


def _rates_hz(counts: np.ndarray, key: str, dt_ms: float = 0.1) -> np.ndarray:
    start_step, end_step = _window_steps(key)
    duration_s = float(end_step - start_step) * dt_ms / 1000.0
    return counts.astype(np.float64) / duration_s


def _brian_counts(
    monitor: SpikeMonitor,
    *,
    n_cells: int,
    key: str,
) -> np.ndarray:
    start_step, end_step = _window_steps(key)
    return count_spikes_in_window(
        np.asarray(monitor.i[:], dtype=np.int64),
        np.asarray(monitor.t[:] / ms, dtype=np.float64),
        n_cells,
        float(start_step) * 0.1,
        float(end_step) * 0.1,
    ).astype(np.int32)


def _run_brian_forced_nonzero(
    *,
    ckpt_dir: Path,
    arrays: dict[str, np.ndarray],
    native_result: dict,
) -> tuple[dict[str, np.ndarray], float]:
    """Run Brian2 with matching source events and forced-threshold spikes."""
    start_scope()
    brian_seed(SEED)
    np.random.seed(SEED)
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    bundle = build_frozen_network(
        architecture="ctx_pred",
        seed=SEED,
        r=1.0,
        g_total=1.0,
        with_cue=False,
        with_v1_to_h="context_only",
        with_feedback_routes=True,
        ckpt_dir=str(ckpt_dir),
    )
    bundle.v1_ring.stim.rates = 0 * Hz
    bundle.silence_tang_direction()

    event_steps, event_sources = _source_events()
    source = SpikeGeneratorGroup(
        int(arrays["pop_v1_stim_n"]),
        indices=event_sources,
        times=event_steps.astype(np.float64) * 0.1 * ms,
        name="native_vs_brian_nonzero_source",
    )
    stim_syn = Synapses(
        source,
        bundle.v1_ring.e,
        model="w : 1",
        on_pre=f"I_e_post += w * {float(arrays['syn_v1_stim_to_e_drive_amp_pA'])}*pA",
        name="native_vs_brian_nonzero_stim_to_e",
    )
    stim_syn.connect(
        i=np.asarray(arrays["syn_v1_stim_to_e_pre"], dtype=np.int32),
        j=np.asarray(arrays["syn_v1_stim_to_e_post"], dtype=np.int32),
    )
    stim_syn.w[:] = np.asarray(arrays["syn_v1_stim_to_e_w"], dtype=np.float64)

    v1_force_cell = int(native_result["v1e_index"])
    hctx_force_cell = int(native_result["hctx_index"])
    hpred_force_cell = int(native_result["hpred_index"])
    v1_force_step = int(native_result["phase_steps"]["v1_force_step"])
    hctx_force_step = int(native_result["phase_steps"]["hctx_force_step"])
    hpred_force_step = int(native_result["phase_steps"]["hpred_force_step"])

    boundary_steps = {
        PHASE_KWARGS["preprobe_start_step"] - 1: 100,
        PHASE_KWARGS["preprobe_start_step"]: 101,
        PHASE_KWARGS["preprobe_end_step"]: 102,
        PHASE_KWARGS["trailer_end_step"]: 103,
    }

    def force_thresholds(t):
        step = int(np.rint(float(t / defaultclock.dt)))
        if step in boundary_steps:
            cell = int(boundary_steps[step])
            bundle.v1_ring.e.V_soma[cell] = -49.0 * mV
            bundle.ctx_pred.ctx.e.V[cell] = -49.0 * mV
            bundle.ctx_pred.pred.e.V[cell] = -49.0 * mV
        if step == v1_force_step:
            bundle.v1_ring.e.V_soma[v1_force_cell] = -49.0 * mV
        if step == hctx_force_step:
            bundle.ctx_pred.ctx.e.V[hctx_force_cell] = -49.0 * mV
        if step == hpred_force_step:
            bundle.ctx_pred.pred.e.V[hpred_force_cell] = -49.0 * mV

    force_op = NetworkOperation(
        force_thresholds,
        when="start",
        name="native_vs_brian_nonzero_forced_thresholds",
    )

    v1_mon = SpikeMonitor(bundle.v1_ring.e, name="native_vs_brian_nonzero_v1_e")
    hctx_mon = SpikeMonitor(bundle.ctx_pred.ctx.e, name="native_vs_brian_nonzero_hctx_e")
    hpred_mon = SpikeMonitor(bundle.ctx_pred.pred.e, name="native_vs_brian_nonzero_hpred_e")
    net = Network(*bundle.groups, source, stim_syn, force_op, v1_mon, hctx_mon, hpred_mon)

    start = time.perf_counter()
    net.run(float(PHASE_KWARGS["n_steps"]) * 0.1 * ms)
    brian_wall_s = time.perf_counter() - start

    return {
        "v1_e.leader": _brian_counts(v1_mon, n_cells=int(arrays["pop_v1_e_n"]), key="v1_e.leader"),
        "v1_e.preprobe": _brian_counts(v1_mon, n_cells=int(arrays["pop_v1_e_n"]), key="v1_e.preprobe"),
        "v1_e.trailer": _brian_counts(v1_mon, n_cells=int(arrays["pop_v1_e_n"]), key="v1_e.trailer"),
        "hctx_e.leader": _brian_counts(hctx_mon, n_cells=int(arrays["pop_ctx_e_n"]), key="hctx_e.leader"),
        "hctx_e.preprobe": _brian_counts(hctx_mon, n_cells=int(arrays["pop_ctx_e_n"]), key="hctx_e.preprobe"),
        "hctx_e.trailer": _brian_counts(hctx_mon, n_cells=int(arrays["pop_ctx_e_n"]), key="hctx_e.trailer"),
        "hpred_e.leader": _brian_counts(hpred_mon, n_cells=int(arrays["pop_pred_e_n"]), key="hpred_e.leader"),
        "hpred_e.preprobe": _brian_counts(hpred_mon, n_cells=int(arrays["pop_pred_e_n"]), key="hpred_e.preprobe"),
        "hpred_e.trailer": _brian_counts(hpred_mon, n_cells=int(arrays["pop_pred_e_n"]), key="hpred_e.trailer"),
    }, brian_wall_s


def main() -> int:
    event_steps, event_sources = _source_events()
    with tempfile.TemporaryDirectory(prefix="native_vs_brian_nonzero_") as tmp_s:
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

        start = time.perf_counter()
        native_result = run_frozen_richter_deterministic_trial_test(
            arrays,
            expected_stim_pre_index=EXPECTED_STIM_PRE_INDEX,
            unexpected_stim_pre_index=UNEXPECTED_STIM_PRE_INDEX,
            stim_period_steps=STIM_PERIOD_STEPS,
            **PHASE_KWARGS,
        )
        native_wall_s = time.perf_counter() - start
        brian_counts, brian_wall_s = _run_brian_forced_nonzero(
            ckpt_dir=ckpt_dir,
            arrays=arrays,
            native_result=native_result,
        )

    assert native_result["source_event_counts"] == {
        "expected.leader": 6,
        "expected.preprobe": 6,
        "expected.trailer": 0,
        "unexpected.leader": 0,
        "unexpected.preprobe": 0,
        "unexpected.trailer": 8,
        "total": int(event_steps.size),
    }
    assert int(event_steps.size) == 20
    assert int(np.count_nonzero(event_sources == EXPECTED_STIM_PRE_INDEX)) == 12
    assert int(np.count_nonzero(event_sources == UNEXPECTED_STIM_PRE_INDEX)) == 8

    max_err = max(float(v) for v in native_result["max_abs_error"].values())
    assert max_err <= TOL, native_result["max_abs_error"]
    for key in RAW_COUNT_KEYS:
        cpu = _as_count(native_result, "cpu", key)
        cuda = _as_count(native_result, "cuda", key)
        assert np.array_equal(cpu, cuda), ("native_cpu_cuda", key)
        assert np.array_equal(cuda, brian_counts[key]), ("brian_native", key)
        assert np.allclose(
            _rates_hz(cuda, key),
            _rates_hz(brian_counts[key], key),
            atol=TOL,
            rtol=0.0,
        ), ("rates", key)

    totals = {
        pop: [
            int(_as_count(native_result, "cuda", f"{pop}.{phase}").sum())
            for phase in ("leader", "preprobe", "trailer")
        ]
        for pop in ("v1_e", "hctx_e", "hpred_e")
    }
    assert totals == {
        "v1_e": [1, 2, 1],
        "hctx_e": [1, 1, 2],
        "hpred_e": [1, 1, 2],
    }
    assert sum(sum(values) for values in totals.values()) > 0

    print(
        "validate_native_vs_brian_nonzero_richter: PASS",
        "spike_mode=forced_threshold_engineering_smoke",
        "source_events=explicit_v1_stim_afferents expected:12 unexpected:8",
        f"counts={totals}",
        f"native_wall_s={native_wall_s:.6f}",
        f"brian_wall_s={brian_wall_s:.6f}",
        f"max_native_cpu_cuda_err={max_err:.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
