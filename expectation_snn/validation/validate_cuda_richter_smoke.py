"""Single-build Richter scheduling smoke for CUDA standalone.

This validator is a narrow bridge toward a fast frozen Richter evaluation:
it proves that all Richter-like leader/trailer/ITI epochs can be queued before
one explicit Brian2 standalone build, with spike counts computed afterwards
from monitors. It intentionally does not require trained ctx_pred checkpoints.

Run:
    EXPECTATION_SNN_BACKEND=numpy python -m expectation_snn.validation.validate_cuda_richter_smoke
    EXPECTATION_SNN_BACKEND=cuda python -m expectation_snn.validation.validate_cuda_richter_smoke
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from brian2 import Network, SpikeMonitor, defaultclock, device, mV, ms, nS, pA
from brian2 import seed as b2_seed, start_scope

from expectation_snn.assays.richter_crossover import (
    RichterConfig,
    build_richter_schedule,
)
from expectation_snn.brian2_model.backend import configure_backend, selected_backend
from expectation_snn.brian2_model.feedback_routes import (
    build_feedback_routes,
    ctx_pred_feedback_config,
)
from expectation_snn.brian2_model.h_ring import (
    N_CHANNELS as H_N_CHANNELS,
    build_h_r,
    pulse_channel,
    silence_cue,
)
from expectation_snn.brian2_model.v1_ring import build_v1_ring, set_stimulus


SEED = 42
DT_MS = 0.1


def _standalone_dir(backend_name: str) -> Path:
    root = os.environ.get("EXPECTATION_SNN_STANDALONE_DIR")
    if root:
        return Path(root).expanduser()
    return Path("/tmp") / "expectation_snn_brian2" / f"{backend_name}_{os.getpid()}"


def _setup_backend():
    start_scope()
    backend_name = selected_backend()
    if backend_name == "numpy":
        backend_cfg = configure_backend()
    else:
        backend_cfg = configure_backend(
            build_on_run=False,
            directory=_standalone_dir(backend_name),
        )
    defaultclock.dt = DT_MS * ms
    b2_seed(SEED)
    np.random.seed(SEED)
    return backend_cfg


def _freeze_plasticity(h, v1) -> None:
    """Freeze mutable plasticity while leaving spike-driven currents active."""
    v1.pv_to_e.active = False
    h.ee.namespace["A_plus_eff"] = 0.0
    h.ee.namespace["A_minus_eff"] = 0.0
    h.inh_to_e.namespace["eta_eff"] = 0.0


def _reset_state(h, v1) -> None:
    """Reset fast state before each scheduled trial."""
    h.e.V = -70.0 * mV
    h.e.I_e = 0 * pA
    h.e.I_i = 0 * pA
    h.e.g_nmda_h = 0 * nS
    h.inh.V = -65.0 * mV
    h.inh.I_e = 0 * pA
    h.inh.I_i = 0 * pA

    v1.e.V_soma = -70.0 * mV
    v1.e.V_ap = -70.0 * mV
    v1.e.I_e = 0 * pA
    v1.e.I_i = 0 * pA
    v1.e.I_ap_e = 0 * pA
    v1.e.w_adapt = 0 * pA
    v1.som.V = -65.0 * mV
    v1.som.I_e = 0 * pA
    v1.som.I_i = 0 * pA
    v1.pv.V = -65.0 * mV
    v1.pv.I_e = 0 * pA
    v1.pv.I_i = 0 * pA


def _nearest_channel(theta_rad: float, n_channels: int = H_N_CHANNELS) -> int:
    chans = np.arange(n_channels, dtype=np.float64) * (np.pi / n_channels)
    d = np.abs(chans - float(theta_rad))
    d = np.minimum(d, np.pi - d)
    return int(np.argmin(d))


def _expected_channel(theta_l_rad: float) -> int:
    """Expected Richter trailer: one 30-degree step after leader."""
    leader_ch = _nearest_channel(theta_l_rad, H_N_CHANNELS)
    return int((leader_ch + 2) % H_N_CHANNELS)


def _count_window(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    n_cells: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    mask = (spike_t_ms >= start_ms) & (spike_t_ms < end_ms)
    if not mask.any():
        return np.zeros(n_cells, dtype=np.int64)
    return np.bincount(spike_i[mask], minlength=n_cells).astype(np.int64)


def _count_windows(mon: SpikeMonitor, n_cells: int, windows: Sequence[tuple[float, float]]) -> np.ndarray:
    spike_i = np.asarray(mon.i[:], dtype=np.int64)
    spike_t_ms = np.asarray(mon.t / ms, dtype=np.float64)
    out = np.zeros((n_cells, len(windows)), dtype=np.int64)
    for k, (start_ms, end_ms) in enumerate(windows):
        out[:, k] = _count_window(spike_i, spike_t_ms, n_cells, start_ms, end_ms)
    return out


def _queue_schedule(
    *,
    net: Network,
    h,
    v1,
    schedule: list[dict],
    leader_ms: float,
    trailer_ms: float,
    iti_ms: float,
    contrast: float,
    h_cue_rate_hz: float,
) -> list[tuple[float, float]]:
    trailer_windows: list[tuple[float, float]] = []
    t_ms = 0.0
    for item in schedule:
        _reset_state(h, v1)

        set_stimulus(v1, theta_rad=float(item["theta_L"]), contrast=float(contrast))
        pulse_channel(h, channel=_expected_channel(float(item["theta_L"])), rate_hz=h_cue_rate_hz)
        net.run(float(leader_ms) * ms)
        t_ms += float(leader_ms)

        set_stimulus(v1, theta_rad=float(item["theta_T"]), contrast=float(contrast))
        silence_cue(h)
        trailer_start = t_ms
        net.run(float(trailer_ms) * ms)
        t_ms += float(trailer_ms)
        trailer_windows.append((trailer_start, t_ms))

        set_stimulus(v1, theta_rad=0.0, contrast=0.0)
        silence_cue(h)
        if iti_ms > 0.0:
            net.run(float(iti_ms) * ms)
            t_ms += float(iti_ms)
    return trailer_windows


def _make_smoke_schedule(n_trials: int) -> list[dict]:
    """Return a small Richter schedule containing both conditions when possible."""
    cfg = RichterConfig(seed=SEED, reps_expected=1, reps_unexpected=1)
    full = build_richter_schedule(cfg, np.random.default_rng(SEED))
    expected = [item for item in full if int(item["condition"]) == 1]
    unexpected = [item for item in full if int(item["condition"]) == 0]
    if n_trials == 1:
        return [dict(expected[0])]
    out: list[dict] = []
    while len(out) < n_trials:
        out.append(dict(expected[len(out) % len(expected)]))
        if len(out) >= n_trials:
            break
        out.append(dict(unexpected[len(out) % len(unexpected)]))
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--leader-ms", type=float, default=80.0)
    parser.add_argument("--trailer-ms", type=float, default=80.0)
    parser.add_argument("--iti-ms", type=float, default=40.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--h-cue-rate-hz", type=float, default=250.0)
    parser.add_argument("--r", type=float, default=1.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    backend_cfg = _setup_backend()
    standalone = backend_cfg.name != "numpy"

    h = build_h_r()
    v1 = build_v1_ring()
    _freeze_plasticity(h, v1)
    fb = build_feedback_routes(h, v1, ctx_pred_feedback_config(r=float(args.r)))

    e_mon = SpikeMonitor(v1.e, name="cuda_richter_smoke_v1_e")
    h_mon = SpikeMonitor(h.e, name="cuda_richter_smoke_h_e")
    net = Network(*h.groups, *v1.groups, *fb.groups, e_mon, h_mon)

    schedule = _make_smoke_schedule(int(args.n_trials))
    trailer_windows = _queue_schedule(
        net=net,
        h=h,
        v1=v1,
        schedule=schedule,
        leader_ms=float(args.leader_ms),
        trailer_ms=float(args.trailer_ms),
        iti_ms=float(args.iti_ms),
        contrast=float(args.contrast),
        h_cue_rate_hz=float(args.h_cue_rate_hz),
    )

    try:
        if standalone:
            if backend_cfg.directory is None:
                raise RuntimeError("standalone backend requires a build directory")
            device.build(directory=str(backend_cfg.directory), compile=True, run=True)

        trailer_counts_e = _count_windows(e_mon, int(v1.e.N), trailer_windows)
        h_counts_all = int(h_mon.num_spikes)
        v1_counts_all = int(e_mon.num_spikes)
        trailer_rate_hz = (
            trailer_counts_e.sum(axis=0)
            / float(v1.e.N)
            / (float(args.trailer_ms) * 1e-3)
        )
        cond = np.asarray([int(item["condition"]) for item in schedule], dtype=np.int64)
        expected_mean = float(trailer_rate_hz[cond == 1].mean()) if np.any(cond == 1) else float("nan")
        unexpected_mean = float(trailer_rate_hz[cond == 0].mean()) if np.any(cond == 0) else float("nan")

        passed = (
            trailer_counts_e.shape == (int(v1.e.N), len(schedule))
            and np.all(np.isfinite(trailer_rate_hz))
            and v1_counts_all > 0
            and h_counts_all > 0
        )
        print(
            "validate_cuda_richter_smoke:",
            f"backend={backend_cfg.name}",
            f"n_trials={len(schedule)}",
            f"standalone_dir={backend_cfg.directory}",
        )
        print(
            "validate_cuda_richter_smoke:",
            f"v1_total_spikes={v1_counts_all}",
            f"h_total_spikes={h_counts_all}",
            f"trailer_rate_hz_mean={float(trailer_rate_hz.mean()):.3f}",
            f"expected_mean={expected_mean:.3f}",
            f"unexpected_mean={unexpected_mean:.3f}",
        )
        print("validate_cuda_richter_smoke: PASS" if passed else "validate_cuda_richter_smoke: FAIL")
        return 0 if passed else 1
    finally:
        if standalone:
            device.reinit()
            device.activate()


if __name__ == "__main__":
    raise SystemExit(main())
