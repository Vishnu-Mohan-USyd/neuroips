"""Validate native diagnostic schema against a tiny controlled Brian2 run."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
from brian2 import (
    Hz,
    Network,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
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
from expectation_snn.cuda_sim.richter_native import (
    RAW_COUNT_KEYS,
    controlled_diagnostic,
    load_manifest,
    write_diagnostic,
)
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


TOL = 1e-10
PHASE_KWARGS = dict(
    n_steps=40,
    leader_start_step=0,
    leader_end_step=10,
    preprobe_start_step=10,
    preprobe_end_step=20,
    trailer_start_step=20,
    trailer_end_step=30,
    iti_start_step=30,
    iti_end_step=40,
)
EVENT_STEPS = np.asarray([1, 2, 11, 21, 31], dtype=np.int32)
EVENT_SOURCES = np.asarray([0, 1, 0, 20, 21], dtype=np.int32)


def _as_int_array(result: dict, side: str, group: str, key: str) -> np.ndarray:
    return np.asarray(result[f"{side}_{group}"][key], dtype=np.int32)


def _assert_native_cpu_cuda_equal(result: dict) -> None:
    for key in RAW_COUNT_KEYS:
        assert np.array_equal(
            _as_int_array(result, "cpu", "raw_counts", key),
            _as_int_array(result, "cuda", "raw_counts", key),
        ), key
    for key in (
        "source.events_by_step",
        "source.events_by_afferent",
        "source.events_by_channel",
        "source.events_by_phase",
    ):
        assert np.array_equal(
            _as_int_array(result, "cpu", "source_counts", key),
            _as_int_array(result, "cuda", "source_counts", key),
        ), key
    max_err = max(float(v) for v in result["max_abs_error"].values())
    assert max_err <= TOL, result["max_abs_error"]


def _brian_counts(
    monitor: SpikeMonitor,
    *,
    n_cells: int,
    start_step: int,
    end_step: int,
    dt_ms: float,
) -> np.ndarray:
    return count_spikes_in_window(
        np.asarray(monitor.i[:], dtype=np.int64),
        np.asarray(monitor.t[:] / ms, dtype=np.float64),
        n_cells,
        float(start_step) * dt_ms,
        float(end_step) * dt_ms,
    ).astype(np.int32)


def _run_brian_controlled(
    *,
    ckpt_dir: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], float]:
    """Run a minimal Brian2 reference using the same explicit source events."""
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

    source = SpikeGeneratorGroup(
        int(arrays["pop_v1_stim_n"]),
        indices=EVENT_SOURCES,
        times=EVENT_STEPS.astype(np.float64) * 0.1 * ms,
        name="native_vs_brian_controlled_source",
    )
    stim_syn = Synapses(
        source,
        bundle.v1_ring.e,
        model="w : 1",
        on_pre=f"I_e_post += w * {float(arrays['syn_v1_stim_to_e_drive_amp_pA'])}*pA",
        name="native_vs_brian_controlled_stim_to_e",
    )
    stim_syn.connect(
        i=np.asarray(arrays["syn_v1_stim_to_e_pre"], dtype=np.int32),
        j=np.asarray(arrays["syn_v1_stim_to_e_post"], dtype=np.int32),
    )
    stim_syn.w[:] = np.asarray(arrays["syn_v1_stim_to_e_w"], dtype=np.float64)

    v1_mon = SpikeMonitor(bundle.v1_ring.e, name="native_vs_brian_v1_e")
    hctx_mon = SpikeMonitor(bundle.ctx_pred.ctx.e, name="native_vs_brian_hctx_e")
    hpred_mon = SpikeMonitor(bundle.ctx_pred.pred.e, name="native_vs_brian_hpred_e")
    net = Network(*bundle.groups, source, stim_syn, v1_mon, hctx_mon, hpred_mon)

    start = time.perf_counter()
    net.run(float(PHASE_KWARGS["n_steps"]) * 0.1 * ms)
    brian_wall_s = time.perf_counter() - start

    return {
        "v1_e.leader": _brian_counts(
            v1_mon,
            n_cells=int(arrays["pop_v1_e_n"]),
            start_step=PHASE_KWARGS["leader_start_step"],
            end_step=PHASE_KWARGS["leader_end_step"],
            dt_ms=0.1,
        ),
        "v1_e.preprobe": _brian_counts(
            v1_mon,
            n_cells=int(arrays["pop_v1_e_n"]),
            start_step=PHASE_KWARGS["preprobe_start_step"],
            end_step=PHASE_KWARGS["preprobe_end_step"],
            dt_ms=0.1,
        ),
        "v1_e.trailer": _brian_counts(
            v1_mon,
            n_cells=int(arrays["pop_v1_e_n"]),
            start_step=PHASE_KWARGS["trailer_start_step"],
            end_step=PHASE_KWARGS["trailer_end_step"],
            dt_ms=0.1,
        ),
        "hctx_e.leader": _brian_counts(
            hctx_mon,
            n_cells=int(arrays["pop_ctx_e_n"]),
            start_step=PHASE_KWARGS["leader_start_step"],
            end_step=PHASE_KWARGS["leader_end_step"],
            dt_ms=0.1,
        ),
        "hctx_e.preprobe": _brian_counts(
            hctx_mon,
            n_cells=int(arrays["pop_ctx_e_n"]),
            start_step=PHASE_KWARGS["preprobe_start_step"],
            end_step=PHASE_KWARGS["preprobe_end_step"],
            dt_ms=0.1,
        ),
        "hctx_e.trailer": _brian_counts(
            hctx_mon,
            n_cells=int(arrays["pop_ctx_e_n"]),
            start_step=PHASE_KWARGS["trailer_start_step"],
            end_step=PHASE_KWARGS["trailer_end_step"],
            dt_ms=0.1,
        ),
        "hpred_e.leader": _brian_counts(
            hpred_mon,
            n_cells=int(arrays["pop_pred_e_n"]),
            start_step=PHASE_KWARGS["leader_start_step"],
            end_step=PHASE_KWARGS["leader_end_step"],
            dt_ms=0.1,
        ),
        "hpred_e.preprobe": _brian_counts(
            hpred_mon,
            n_cells=int(arrays["pop_pred_e_n"]),
            start_step=PHASE_KWARGS["preprobe_start_step"],
            end_step=PHASE_KWARGS["preprobe_end_step"],
            dt_ms=0.1,
        ),
        "hpred_e.trailer": _brian_counts(
            hpred_mon,
            n_cells=int(arrays["pop_pred_e_n"]),
            start_step=PHASE_KWARGS["trailer_start_step"],
            end_step=PHASE_KWARGS["trailer_end_step"],
            dt_ms=0.1,
        ),
    }, brian_wall_s


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_vs_brian_small_") as tmp_s:
        root = Path(tmp_s)
        ckpt_dir = root / "checkpoints"
        manifest_path = root / "ctx_pred_manifest.npz"
        out_json = root / "native_small_richter.json"
        out_npz = root / "native_small_richter.npz"
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
        arrays = load_manifest(manifest_path)
        payload = controlled_diagnostic(
            arrays,
            event_steps=EVENT_STEPS,
            event_sources=EVENT_SOURCES,
            condition={
                "name": "small_controlled_brian_comparison",
                "source_mode": "explicit_events",
                "expected_channel": 0,
                "unexpected_channel": 1,
            },
            expected_channel=0,
            unexpected_channel=1,
            **PHASE_KWARGS,
        )
        write_diagnostic(payload, out_json=out_json, out_npz=out_npz)
        native_result = payload["native_result"]
        brian_counts, brian_wall_s = _run_brian_controlled(
            ckpt_dir=ckpt_dir,
            arrays=arrays,
        )

        loaded_json = json.loads(out_json.read_text())
        with np.load(out_npz) as raw:
            assert raw["cuda_v1_e_leader_counts"].shape == (192, 1)
            assert raw["cuda_hctx_e_preprobe_counts"].shape == (192, 1)
            assert raw["cuda_hpred_e_trailer_counts"].shape == (192, 1)
            assert np.array_equal(raw["source_event_steps"], EVENT_STEPS)
            assert np.array_equal(raw["source_event_sources"], EVENT_SOURCES)

    _assert_native_cpu_cuda_equal(native_result)
    assert loaded_json["meta"]["schema"] == "native_richter_subset_v1"
    assert loaded_json["backend"]["name"] == "native_cuda"
    assert loaded_json["source_events"]["n_events"] == int(EVENT_STEPS.size)
    assert native_result["source_event_counts"] == {
        "leader": 2,
        "preprobe": 1,
        "trailer": 1,
        "iti": 1,
        "total": 5,
    }
    source_by_phase = _as_int_array(
        native_result,
        "cuda",
        "source_counts",
        "source.events_by_phase",
    )
    assert np.array_equal(source_by_phase, np.asarray([2, 1, 1, 1, 5], dtype=np.int32))
    source_by_channel = _as_int_array(
        native_result,
        "cuda",
        "source_counts",
        "source.events_by_channel",
    )
    assert int(source_by_channel[0]) == 3
    assert int(source_by_channel[1]) == 2
    assert int(source_by_channel.sum()) == 5

    for key in RAW_COUNT_KEYS:
        native_counts = _as_int_array(native_result, "cuda", "raw_counts", key)
        assert np.array_equal(native_counts, brian_counts[key]), key

    native_wall_s = float(payload["diagnostic"]["performance"]["native_wall_s"])
    total_native_spikes = sum(
        int(np.asarray(native_result["cuda_raw_counts"][key], dtype=np.int32).sum())
        for key in RAW_COUNT_KEYS
    )
    total_brian_spikes = sum(int(values.sum()) for values in brian_counts.values())
    print(
        "validate_native_vs_brian_small_richter: PASS",
        "source_events=leader:2,preprobe:1,trailer:1,iti:1",
        f"native_wall_s={native_wall_s:.6f}",
        f"brian_wall_s={brian_wall_s:.6f}",
        f"native_spikes={total_native_spikes}",
        f"brian_spikes={total_brian_spikes}",
        f"json_schema={loaded_json['meta']['schema']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
