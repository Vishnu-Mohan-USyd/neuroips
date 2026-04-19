"""Calibrate g_total + per-route drive amplitudes for feedback_routes.

Target (Lead dispatch for Sprint 4.5):
    at r=1.0 (balanced), with the matched-channel H bump firing at ~50 Hz
    concurrently with a grating at the matched orientation:
      - direct-only (r=inf): +5-15 % V1 matched-channel gain vs no-feedback.
      - SOM-only    (r=0):   -5-15 % V1 matched-channel gain vs no-feedback.

This script sweeps (drive_amp_h_to_v1e_apical_pA,
drive_amp_h_to_v1som_pA) at fixed g_total=1.0 and sigma_channels=1.0,
reporting the (direct, SOM, balanced) gains for each pair. Use the
printed table to pick the defaults committed to feedback_routes.py.

Usage
-----
    python -m expectation_snn.scripts.calibrate_feedback_routes

or
    python expectation_snn/scripts/calibrate_feedback_routes.py

Reports one row per (drive_amp_direct, drive_amp_som) pair with
matched-channel V1_E rate relative to no-feedback baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import numpy as np

from brian2 import (
    Network, SpikeMonitor, defaultclock, prefs, ms, Hz,
    seed as b2_seed,
)

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from expectation_snn.brian2_model.h_ring import (
    build_h_r, pulse_channel, silence_cue,
    N_CHANNELS as H_N_CHANNELS,
)
from expectation_snn.brian2_model.v1_ring import (
    build_v1_ring, set_stimulus,
)
from expectation_snn.brian2_model.feedback_routes import (
    build_feedback_routes, FeedbackRoutesConfig,
)


SEED = 42
RUN_PRE_MS = 200.0   # warm-up, no feedback pulse yet
RUN_TRIAL_MS = 1500.0 # measurement window (long -> spike-count resolution)
H_PULSE_RATE = 300.0   # cue rate -> H matched ~ 50-90 Hz (Lead target)


def _run_once(
    drive_direct_pA: float,
    drive_som_pA: float,
    r_val: float,
    grating_on: bool,
    h_pulse_on: bool,
) -> dict:
    """Return V1_E matched-channel rate (Hz) + baseline unmatched rate.

    Build a fresh network per call so Brian2 state is isolated. Seed is
    re-set deterministically so only the config changes between calls.
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)

    h = build_h_r()
    v = build_v1_ring()
    cfg = FeedbackRoutesConfig(
        g_total=1.0,
        r=r_val,
        drive_amp_h_to_v1e_apical_pA=drive_direct_pA,
        drive_amp_h_to_v1som_pA=drive_som_pA,
        sigma_channels=1.0,
    )
    fb = build_feedback_routes(h, v, cfg)

    if grating_on:
        set_stimulus(v, theta_rad=0.0, contrast=1.0)   # grating at ch0
    else:
        set_stimulus(v, theta_rad=0.0, contrast=0.0)

    e_mon = SpikeMonitor(v.e)
    net = Network(*h.groups, *v.groups, *fb.groups, e_mon)

    # Warm-up (no H pulse yet)
    silence_cue(h)
    net.run(RUN_PRE_MS * ms)
    n_pre = e_mon.num_spikes

    # Trial window: optionally pulse H ch0 to drive bump
    if h_pulse_on:
        pulse_channel(h, channel=0, rate_hz=H_PULSE_RATE)
    net.run(RUN_TRIAL_MS * ms)
    silence_cue(h)
    n_trial = e_mon.num_spikes - n_pre

    # Per-channel V1_E spike counts within the trial window only
    t = np.asarray(e_mon.t / ms)
    i = np.asarray(e_mon.i[:])
    in_trial = t >= RUN_PRE_MS
    ch = v.e_channel[i[in_trial]]
    per_ch = np.bincount(ch, minlength=12)
    n_per_ch_cells = 16
    rate_per_ch = per_ch / (n_per_ch_cells * RUN_TRIAL_MS * 1e-3)
    return {
        "matched_hz": float(rate_per_ch[0]),
        "unmatched_hz": float(rate_per_ch[6]),    # orthogonal
        "full_profile_hz": rate_per_ch.tolist(),
        "total_trial_spikes": int(n_trial),
    }


def calibrate() -> None:
    # Measure no-feedback baseline first (r=0, drives both amps to 0 gain
    # via g_SOM=g_total=1 but drive_amp_h_to_v1som=0 kills route 2 too).
    # Cleanest: use g_total=0 via a special config path; we instead set
    # both drives to 0, which zeros route effect entirely.
    baseline = _run_once(
        drive_direct_pA=0.0, drive_som_pA=0.0,
        r_val=1.0, grating_on=True, h_pulse_on=True,
    )
    base_matched = baseline["matched_hz"]
    print(f"Baseline (no feedback, grating+H pulse):")
    print(f"  matched_hz={base_matched:.3f}  "
          f"unmatched_hz={baseline['unmatched_hz']:.3f}")
    print()

    print(f"{'drive_direct':>12} {'drive_som':>10} | "
          f"{'dir-only (+%)':>13} | {'som-only (-%)':>13} | "
          f"{'balanced (%)':>12}")
    print("-" * 75)

    # Sweep drive amplitudes. Sigma & g_total fixed. With H pulsed at
    # ~50-90 Hz, we need apical drive amps ~ 50-100 pA and SOM drives
    # ~20-60 pA (SOM is disynaptic: H -> SOM -> E, already amplified by
    # existing w_som_e=0.5 in v1_ring).
    for dd in (40.0, 60.0, 80.0, 100.0, 140.0):
        for ds in (15.0, 25.0, 40.0, 60.0, 90.0):
            # Direct-only: r=infinity. For this sweep use r=100 (near-inf).
            dir_only = _run_once(
                drive_direct_pA=dd, drive_som_pA=ds,
                r_val=100.0, grating_on=True, h_pulse_on=True,
            )
            # SOM-only: r=0.01 (near-zero).
            som_only = _run_once(
                drive_direct_pA=dd, drive_som_pA=ds,
                r_val=0.01, grating_on=True, h_pulse_on=True,
            )
            # Balanced r=1.0
            balanced = _run_once(
                drive_direct_pA=dd, drive_som_pA=ds,
                r_val=1.0, grating_on=True, h_pulse_on=True,
            )
            d_gain = (dir_only["matched_hz"] - base_matched) / max(
                base_matched, 1e-6) * 100.0
            s_gain = (som_only["matched_hz"] - base_matched) / max(
                base_matched, 1e-6) * 100.0
            b_gain = (balanced["matched_hz"] - base_matched) / max(
                base_matched, 1e-6) * 100.0
            print(f"{dd:>12.1f} {ds:>10.1f} | "
                  f"{d_gain:>+13.2f} | {s_gain:>+13.2f} | "
                  f"{b_gain:>+12.2f}")

    print()
    print("Target at r=1.0: direct-only in +5..+15, SOM-only in -5..-15")


if __name__ == "__main__":
    calibrate()
