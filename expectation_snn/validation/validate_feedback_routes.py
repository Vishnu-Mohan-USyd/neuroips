"""Functional validation for `brian2_model.feedback_routes`.

Per-component validation rule: must pass before Sprint 5a assays use it.

Five assays per Lead Sprint 4.5 dispatch:

  [1] matched/unmatched V1 gain ratio > 1.1 at r=1.0
      (feature-matched feedback preserves/enhances channel preference)
  [2] ctx_pred route primitive: low-r dampens the local center+d1/d2
      neighborhood, while r=4 produces strict center-up and local-flank-down
      sharpening
  [3] total-off (g_total=0): no feedback modulation
  [4] sub-threshold claim: H bump + grating OFF -> V1_E < 1 Hz
  [5] balance sweep preview: monotonic local-neighborhood shift across
      {0.25, 0.5, 1, 2, 4}

Also retained from the prior topological checks:

  [6] topology determinism: ctx_pred direct route is center-only, SOM
      route is wrapped d1/d2 surround, and both rows are normalized.

Uses seed=42, Brian2 numpy codegen, dt=0.1 ms. Each assay re-builds the
network cleanly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

from brian2 import (
    Network, SpikeMonitor, defaultclock, device, ms, Hz,
    seed as b2_seed, start_scope,
)

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.h_ring import (
    build_h_r, pulse_channel, silence_cue,
)
from expectation_snn.brian2_model.backend import configure_backend, selected_backend
from expectation_snn.brian2_model.v1_ring import (
    build_v1_ring, set_stimulus,
)
from expectation_snn.brian2_model.feedback_routes import (
    build_feedback_routes, balance_weights, ctx_pred_feedback_config,
)


SEED = 42
DT_MS = 0.1
RUN_PRE_MS = 200.0
RUN_TRIAL_MS = 1500.0
H_PULSE_RATE_HZ = 300.0

# One V1_E spike in one channel over the 1.5 s window is 1/(16*1.5) Hz.
MIN_ROUTE_DELTA_HZ = 0.04
_STANDALONE_BUILD_INDEX = 0


# -- helpers ----------------------------------------------------------------

def _next_standalone_dir(backend_name: str) -> Path:
    """Return a fresh standalone build directory for this validator process."""
    global _STANDALONE_BUILD_INDEX
    _STANDALONE_BUILD_INDEX += 1
    root = os.environ.get("EXPECTATION_SNN_STANDALONE_DIR")
    if root:
        base = Path(root).expanduser()
    else:
        base = Path("/tmp") / "expectation_snn_brian2" / f"{backend_name}_{os.getpid()}"
    return base / f"build_{_STANDALONE_BUILD_INDEX:03d}"


def _setup_brian():
    start_scope()
    backend_name = selected_backend()
    if backend_name == "numpy":
        backend_cfg = configure_backend()
    else:
        backend_cfg = configure_backend(
            build_on_run=False,
            directory=_next_standalone_dir(backend_name),
        )
    defaultclock.dt = DT_MS * ms
    b2_seed(SEED)
    np.random.seed(SEED)
    return backend_cfg


def _is_standalone(backend_cfg) -> bool:
    return backend_cfg.name != "numpy"


def _build_standalone(backend_cfg) -> None:
    if backend_cfg.directory is None:
        raise RuntimeError("Standalone backend requires a build directory")
    device.build(directory=str(backend_cfg.directory), compile=True, run=True)


def _reset_standalone_device() -> None:
    device.reinit()
    device.activate()


def _trial_rates(e_mon, som_mon, h_mon, v, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Per-channel rates over the TRIAL window only.
    def _per_ch(mon, ch_map, n_per_ch):
        t = np.asarray(mon.t / ms)
        i = np.asarray(mon.i[:])
        intr = t >= RUN_PRE_MS
        counts = np.bincount(ch_map[i[intr]], minlength=12)
        return counts / (n_per_ch * RUN_TRIAL_MS * 1e-3)

    return (
        _per_ch(e_mon, v.e_channel, 16),
        _per_ch(som_mon, v.som_channel, 4),
        _per_ch(h_mon, h.e_channel, 16),
    )


def _run_trial(
    *,
    drive_direct_pA: float = 30.0,
    drive_som_pA: float = 40.0,
    r_val: float = 1.0,
    g_total: float = 1.0,
    grating_on: bool = True,
    h_pulse_on: bool = True,
    grating_contrast: float = 1.0,
) -> dict:
    """Build network, run trial, return per-channel V1_E + V1_SOM counts."""
    backend_cfg = _setup_brian()
    standalone = _is_standalone(backend_cfg)
    h = build_h_r()
    v = build_v1_ring()
    cfg = ctx_pred_feedback_config(
        g_total=g_total, r=r_val,
        drive_amp_h_to_v1e_apical_pA=drive_direct_pA,
        drive_amp_h_to_v1som_pA=drive_som_pA,
    )
    fb = build_feedback_routes(h, v, cfg)

    set_stimulus(v, theta_rad=0.0,
                 contrast=grating_contrast if grating_on else 0.0)
    e_mon = SpikeMonitor(v.e)
    som_mon = SpikeMonitor(v.som)
    h_mon = SpikeMonitor(h.e)
    net = Network(*h.groups, *v.groups, *fb.groups,
                  e_mon, som_mon, h_mon)

    silence_cue(h)
    net.run(RUN_PRE_MS * ms)
    if not standalone:
        n_e_pre = e_mon.num_spikes
        n_s_pre = som_mon.num_spikes
        n_h_pre = h_mon.num_spikes

    if h_pulse_on:
        pulse_channel(h, channel=0, rate_hz=H_PULSE_RATE_HZ)
    net.run(RUN_TRIAL_MS * ms)
    silence_cue(h)

    if standalone:
        try:
            _build_standalone(backend_cfg)
            rate_e, rate_s, rate_h = _trial_rates(e_mon, som_mon, h_mon, v, h)
        finally:
            _reset_standalone_device()
    else:
        rate_e, rate_s, rate_h = _trial_rates(e_mon, som_mon, h_mon, v, h)


    return {
        "v1_e_hz": rate_e,
        "v1_som_hz": rate_s,
        "h_e_hz": rate_h,
        "v1_e_matched_hz": float(rate_e[0]),
        "v1_e_unmatched_hz": float(rate_e[3]),   # 45° off (ch3)
        "v1_e_orth_hz": float(rate_e[6]),        # 90° orthogonal
        "fb_g_direct": float(fb.g_direct),
        "fb_g_som": float(fb.g_SOM),
    }


def _gain_pct(trial_hz: float, baseline_hz: float) -> float:
    if baseline_hz <= 1e-6:
        return float("inf") if trial_hz > 0 else 0.0
    return (trial_hz / baseline_hz - 1.0) * 100.0


def _local_flank_mean(rate: np.ndarray) -> float:
    """Mean of the d1/d2 local flank channels around predicted channel 0."""
    return float(np.mean(rate[[1, 11, 2, 10]]))


def _local_neighborhood_mean(rate: np.ndarray) -> float:
    """Mean of center plus d1/d2 local surround around predicted channel 0."""
    return float(np.mean(rate[[0, 1, 11, 2, 10]]))


# -- assays -----------------------------------------------------------------

def assay_1_matched_unmatched_ratio_at_r1() -> bool:
    """[1] V1 E matched/unmatched > 1.1 at r=1.0 (feature selectivity)."""
    r = _run_trial(r_val=1.0)
    matched = r["v1_e_matched_hz"]
    unmatched = r["v1_e_unmatched_hz"]
    if unmatched <= 1e-6:
        # No firing at ch3; the grating tuning already makes unmatched
        # channels quiet, so ratio is trivially > 1.1.
        passed = matched > 0.0
        detail = (f"matched={matched:.2f}, unmatched(ch3)={unmatched:.2f}"
                  f" (unmatched ~ 0 under tight grating tuning; "
                  f"selectivity preserved)")
    else:
        ratio = matched / unmatched
        passed = ratio > 1.1
        detail = f"matched={matched:.2f}, unmatched={unmatched:.2f}, " \
                 f"ratio={ratio:.3f}"
    print(f"[1] matched_unmatched_ratio_r1: "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


def assay_2_ctx_pred_dampen_and_sharpen_primitive() -> bool:
    """[2] Low-r local dampening; r=4 center-up/local-flank-down.

    The d1/d2 SOM kernel has zero center weight for a delta-function
    prediction. Therefore low-r suppression is evaluated over the local
    center+d1/d2 neighborhood. Strict sharpening at r=4 is evaluated as
    center-up together with local-flank-down.
    """
    baseline = _run_trial(drive_direct_pA=0.0, drive_som_pA=0.0,
                          r_val=1.0, g_total=0.0)
    low_r = _run_trial(r_val=0.25)
    high_r = _run_trial(r_val=4.0)

    baseline_center = baseline["v1_e_matched_hz"]
    baseline_local = _local_neighborhood_mean(baseline["v1_e_hz"])
    low_local_delta = _local_neighborhood_mean(low_r["v1_e_hz"]) - baseline_local
    high_center_delta = high_r["v1_e_matched_hz"] - baseline_center
    baseline_flank = _local_flank_mean(baseline["v1_e_hz"])
    high_flank_delta = _local_flank_mean(high_r["v1_e_hz"]) - baseline_flank

    passed = (
        low_local_delta <= -MIN_ROUTE_DELTA_HZ
        and high_center_delta >= MIN_ROUTE_DELTA_HZ
        and high_flank_delta <= -MIN_ROUTE_DELTA_HZ
    )
    detail = (
        f"low-r local_neighborhood_delta={low_local_delta:+.3f} Hz; "
        f"r=4 center_delta={high_center_delta:+.3f} Hz, "
        f"local_flank_delta={high_flank_delta:+.3f} Hz; "
        f"baseline center/flank/local="
        f"{baseline_center:.3f}/{baseline_flank:.3f}/{baseline_local:.3f} Hz"
    )
    print(f"[2] ctx_pred_route_primitive:  "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


def assay_3_total_off_no_modulation() -> bool:
    """[3] g_total=0 (no feedback): V1 E matched rate == baseline."""
    # g_total=0 -> both g_direct=0 and g_SOM=0 regardless of r.
    no_fb = _run_trial(g_total=0.0, r_val=1.0)
    # Compare to the "baseline" that uses drive_amp=0 (truly no feedback).
    baseline = _run_trial(drive_direct_pA=0.0, drive_som_pA=0.0,
                          r_val=1.0, g_total=0.0)
    # Both should be identical (feedback Synapses exist but weights are 0).
    diff_matched = abs(no_fb["v1_e_matched_hz"] - baseline["v1_e_matched_hz"])
    rel_diff = diff_matched / max(baseline["v1_e_matched_hz"], 1e-6)
    passed = rel_diff < 0.05
    detail = (f"no_fb_matched={no_fb['v1_e_matched_hz']:.3f}  "
              f"baseline_matched={baseline['v1_e_matched_hz']:.3f}  "
              f"rel_diff={rel_diff*100:.2f}%")
    print(f"[3] total_off_no_modulation:   "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


def assay_4_sub_threshold_no_firing() -> bool:
    """[4] H bump + grating OFF -> V1 E < 1 Hz (apical is modulatory)."""
    r = _run_trial(grating_on=True, h_pulse_on=True, grating_contrast=0.0)
    # grating_on=True, contrast=0.0 means stimulus afferents fire at 0 Hz.
    m = r["v1_e_matched_hz"]
    u = r["v1_e_unmatched_hz"]
    o = r["v1_e_orth_hz"]
    passed = max(m, u, o) < 1.0
    detail = (f"grating OFF, H pulse ON (r=1.0): matched={m:.3f}  "
              f"unmatched={u:.3f}  orth={o:.3f}  H_ch0={r['h_e_hz'][0]:.2f} Hz")
    print(f"[4] sub_threshold_no_firing:   "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


def assay_5_monotonic_balance_sweep() -> bool:
    """[5] local-neighborhood gain monotone non-decreasing across r sweep."""
    baseline = _run_trial(drive_direct_pA=0.0, drive_som_pA=0.0,
                          r_val=1.0, g_total=0.0)
    bm = _local_neighborhood_mean(baseline["v1_e_hz"])
    r_vals = [0.25, 0.50, 1.00, 2.00, 4.00]
    gains = []
    for rv in r_vals:
        t = _run_trial(r_val=rv)
        gains.append(_gain_pct(_local_neighborhood_mean(t["v1_e_hz"]), bm))
    # Monotone non-decreasing (discrete resolution causes ties)
    mono = all(gains[i + 1] >= gains[i] - 1e-6 for i in range(len(gains) - 1))
    # First and last must have opposite signs (SOM-dominant vs direct-dominant)
    opposite = (gains[0] < 0) and (gains[-1] > 0)
    passed = mono and opposite
    detail = ("  ".join(f"r={rv:.2f}:{g:+.2f}%"
                         for rv, g in zip(r_vals, gains)))
    print(f"[5] monotonic_balance_sweep:   "
          f"{'PASS' if passed else 'FAIL'} -- local_baseline={bm:.2f}  {detail}")
    return passed


def assay_6_topology_determinism() -> bool:
    """[6] topology: center direct route + wrapped d1/d2 SOM surround."""
    backend_cfg = _setup_brian()
    standalone = _is_standalone(backend_cfg)
    h = build_h_r()
    v = build_v1_ring()
    fb = build_feedback_routes(h, v, ctx_pred_feedback_config(g_total=1.0, r=1.0))
    _net = Network(*h.groups, *v.groups, *fb.groups)
    _net.run(0 * ms)
    try:
        if standalone:
            _build_standalone(backend_cfg)

        kd = fb.kernel_direct
        ks = fb.kernel_som
        expected_direct = np.eye(12)
        expected_som = np.zeros((12, 12), dtype=np.float64)
        for ci in range(12):
            expected_som[ci, (ci - 1) % 12] = 0.4
            expected_som[ci, (ci + 1) % 12] = 0.4
            expected_som[ci, (ci - 2) % 12] = 0.1
            expected_som[ci, (ci + 2) % 12] = 0.1
        passed_kernel = (
            np.allclose(kd, expected_direct)
            and np.allclose(ks, expected_som)
            and np.allclose(kd.sum(axis=1), 1.0)
            and np.allclose(ks.sum(axis=1), 1.0)
        )

        # Direct route connects only same-channel H_E -> V1_E.
        i1 = np.asarray(fb.hr_to_v1e.i[:])
        j1 = np.asarray(fb.hr_to_v1e.j[:])
        ci = h.e_channel[i1]
        cj = v.e_channel[j1]
        passed_direct_topology = bool(np.all(ci == cj))

        # SOM route connects exactly d1/d2 wrapped targets and excludes center.
        i2 = np.asarray(fb.hr_to_v1som.i[:])
        j2 = np.asarray(fb.hr_to_v1som.j[:])
        ci2 = h.e_channel[i2]
        cj2 = v.som_channel[j2]
        d = np.abs(ci2 - cj2)
        d = np.minimum(d, 12 - d)
        passed_som_topology = bool(
            np.all(np.isin(d, [1, 2]))
            and not np.any(d == 0)
            and int(fb.kernel_w_som.size) == 12 * 16 * 4 * 4
        )
        passed = passed_kernel and passed_direct_topology and passed_som_topology
        detail = (
            f"direct center_only={passed_direct_topology} n={fb.kernel_w_direct.size}; "
            f"SOM d={sorted(set(d.tolist()))} n={fb.kernel_w_som.size}; "
            f"row_sums direct/SOM={kd.sum(axis=1).min():.1f}/"
            f"{ks.sum(axis=1).min():.1f}; row0 SOM={ks[0].tolist()}"
        )
        print(f"[6] topology_determinism:      "
              f"{'PASS' if passed else 'FAIL'} -- {detail}")
        return passed
    finally:
        if standalone:
            _reset_standalone_device()


# -- runner -----------------------------------------------------------------

def main() -> int:
    assays = [
        ("matched_unmatched_ratio_r1", assay_1_matched_unmatched_ratio_at_r1),
        ("ctx_pred_route_primitive",   assay_2_ctx_pred_dampen_and_sharpen_primitive),
        ("total_off_no_modulation",    assay_3_total_off_no_modulation),
        ("sub_threshold_no_firing",    assay_4_sub_threshold_no_firing),
        ("monotonic_balance_sweep",    assay_5_monotonic_balance_sweep),
        ("topology_determinism",       assay_6_topology_determinism),
    ]
    n_pass = 0
    for name, fn in assays:
        try:
            ok = fn()
        except Exception as exc:
            print(f"[X] {name}: EXCEPTION {type(exc).__name__}: {exc}")
            ok = False
        if ok:
            n_pass += 1
    total = len(assays)
    print(f"\n--- validate_feedback_routes: {n_pass}/{total} PASS ---")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
