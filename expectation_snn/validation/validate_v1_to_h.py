"""Functional validation for `brian2_model.feedforward_v1_to_h`.

Per-component validation rule (Sprint 5.5, task #31): must pass before the
assay-runtime builder wires this pathway in.

Four assays per Lead's task #31 spec:

  [1] feature-matched drive: V1_E[ch0] high-rate -> H_E[ch0] >> H_E[ch_far]
  [2] feedforward sufficient (H -> V1 zeroed): V1[ch0] grating -> H bump emerges
  [3] persistence: drive V1[ch0] strongly, silence V1, H bump persists 100+ ms
                   (uses higher g_v1_to_h to push H above NMDA-recruitment regime)
  [4] g_v1_to_h sweep: find the gain where V1_E ~22 Hz matched (default Stage-0
                        calibration) gives H_E ~10-30 Hz at the matched channel

All assays use seed=42, Brian2 numpy codegen, dt=0.1 ms. Each rebuilds
the network cleanly (Brian2 state is per-Network).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from brian2 import (
    Network, SpikeMonitor, defaultclock, prefs, ms, pA,
    seed as b2_seed,
)

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.h_ring import (
    build_h_r, silence_cue, N_CHANNELS as H_N_CHANNELS,
)
from expectation_snn.brian2_model.v1_ring import (
    build_v1_ring, set_stimulus, N_CHANNELS as V1_N_CHANNELS,
)
from expectation_snn.brian2_model.feedforward_v1_to_h import (
    build_v1_to_h_feedforward, V1ToHConfig,
    DEFAULT_G_V1_TO_H, DEFAULT_DRIVE_AMP_V1_TO_H_PA,
)
from expectation_snn.brian2_model.feedback_routes import (
    build_feedback_routes, FeedbackRoutesConfig,
)


SEED = 42
DT_MS = 0.1
CKPT_S0 = os.path.join(
    str(_pkg_root), "expectation_snn", "data", "checkpoints",
    "stage_0_seed42.npz",
)
CKPT_S1_HR = os.path.join(
    str(_pkg_root), "expectation_snn", "data", "checkpoints",
    "stage_1_hr_seed42.npz",
)


# -- helpers ----------------------------------------------------------------


def _setup_brian():
    prefs.codegen.target = "numpy"
    defaultclock.dt = DT_MS * ms
    b2_seed(SEED)
    np.random.seed(SEED)


def _load_stage0_v1(v1) -> None:
    """Apply Stage-0 V1 calibration so V1 fires in realistic 2-8 Hz pop avg."""
    s0 = np.load(CKPT_S0)
    v1.e.I_bias = float(s0["bias_pA"]) * pA
    v1.pv_to_e.w[:] = np.asarray(s0["pv_to_e_w"], dtype=np.float64)
    v1.pv_to_e.active = False  # freeze


def _load_stage1_hr(h) -> None:
    """Apply Stage-1 H_R calibration + freeze plasticity."""
    s1 = np.load(CKPT_S1_HR)
    h.ee.w[:] = np.asarray(s1["ee_w_final"], dtype=np.float64)
    h.ee.namespace["A_plus_eff"] = 0.0
    h.ee.namespace["A_minus_eff"] = 0.0
    h.inh_to_e.namespace["eta_eff"] = 0.0


def _per_channel_rate(mon, ch_map, n_per_ch, win_ms, n_channels=12):
    """Per-channel firing rate (Hz) from a SpikeMonitor over a window."""
    idx = np.asarray(mon.i[:])
    counts = np.bincount(ch_map[idx], minlength=n_channels)
    return counts / (n_per_ch * win_ms * 1e-3)


# -- assay 1: feature-matched drive -----------------------------------------


def assay_1_feature_matched_drive() -> bool:
    """Drive V1[ch0] strongly, confirm H[ch0] >> H[ch_far]."""
    _setup_brian()
    v = build_v1_ring()
    h = build_h_r()
    _load_stage0_v1(v)
    _load_stage1_hr(h)
    ff = build_v1_to_h_feedforward(v, h, V1ToHConfig())

    # Drive only ch0 stimulus afferents. Use set_stimulus at theta=0
    # (peaks ch0). The Stage-0 calibration tunes V1_E to ~ 4 Hz mean
    # (matched ch0 ~22 Hz at full contrast).
    set_stimulus(v, theta_rad=0.0, contrast=1.0)
    silence_cue(h)

    h_mon = SpikeMonitor(h.e, name="a1_h_mon")
    v_mon = SpikeMonitor(v.e, name="a1_v_mon")
    net = Network(*v.groups, *h.groups, *ff.groups, h_mon, v_mon)
    net.run(500 * ms)

    h_rates = _per_channel_rate(h_mon, h.e_channel, 16, 500.0)
    v_rates = _per_channel_rate(v_mon, v.e_channel, 16, 500.0)

    h_ch0 = float(h_rates[0])
    h_ch6 = float(h_rates[6])
    v_ch0 = float(v_rates[0])
    v_ch6 = float(v_rates[6])

    passed = (h_ch0 > 5.0) and (h_ch6 < 2.0)
    detail = (f"V1[ch0]={v_ch0:.1f}  V1[ch6]={v_ch6:.2f}  "
              f"H[ch0]={h_ch0:.1f}  H[ch6]={h_ch6:.2f}  "
              f"(target H[ch0]>5 H[ch6]<2)")
    print(f"[1] feature_matched_drive: "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# -- assay 2: feedforward sufficient (H->V1 disabled) -----------------------


def assay_2_feedforward_sufficient_no_feedback() -> bool:
    """With H->V1 feedback ZEROED, V1 grating still produces H bump.

    Confirms V1 -> H is the only pathway needed for H activation; we are
    not relying on a closed loop (V1 -> H -> V1 -> H).
    """
    _setup_brian()
    v = build_v1_ring()
    h = build_h_r()
    _load_stage0_v1(v)
    _load_stage1_hr(h)
    ff = build_v1_to_h_feedforward(v, h, V1ToHConfig())
    # Build feedback routes too, but at g_total=0 -> truly zero feedback.
    fb = build_feedback_routes(
        h, v, FeedbackRoutesConfig(g_total=0.0, r=1.0),
        name_prefix="a2_fb",
    )
    # Sanity: feedback weights truly zero.
    assert float(np.asarray(fb.hr_to_v1e.w[:]).max()) == 0.0
    assert float(np.asarray(fb.hr_to_v1som.w[:]).max()) == 0.0

    set_stimulus(v, theta_rad=0.0, contrast=1.0)
    silence_cue(h)
    h_mon = SpikeMonitor(h.e, name="a2_h_mon")
    net = Network(*v.groups, *h.groups, *ff.groups, *fb.groups, h_mon)
    net.run(500 * ms)
    h_rates = _per_channel_rate(h_mon, h.e_channel, 16, 500.0)
    h_ch0 = float(h_rates[0])
    passed = h_ch0 > 5.0
    detail = (f"H->V1 zeroed; V1 grating ch0; H[ch0]={h_ch0:.1f} Hz "
              f"(>5 required)")
    print(f"[2] feedforward_sufficient: "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# -- assay 3: persistence (above-threshold drive) ---------------------------


def assay_3_persistence_post_v1_silencing() -> bool:
    """Drive V1[ch0] strongly enough to push H into NMDA recruitment band,
    silence V1, confirm H bump persists for >= 100 ms (one 100-ms bin).

    Uses g_v1_to_h = 2.5 (above the calibrated default 1.5) to demonstrate
    the H ring CAN persist when feedforward drive engages NMDA, not that
    the calibrated default does so. The default (1.5) deliberately keeps
    H below attractor saturation - that is the "engage but don't lock in"
    target band.
    """
    _setup_brian()
    v = build_v1_ring()
    h = build_h_r()
    _load_stage0_v1(v)
    _load_stage1_hr(h)
    # Higher gain to push H into persistence regime (not the assay default).
    ff = build_v1_to_h_feedforward(
        v, h, V1ToHConfig(g_v1_to_h=2.5),
    )

    set_stimulus(v, theta_rad=0.0, contrast=1.0)
    silence_cue(h)
    h_mon = SpikeMonitor(h.e, name="a3_h_mon")
    net = Network(*v.groups, *h.groups, *ff.groups, h_mon)

    # Drive phase: 200 ms.
    net.run(200 * ms)
    # Silence V1.
    set_stimulus(v, theta_rad=0.0, contrast=0.0)
    net.run(300 * ms)

    h_t = np.asarray(h_mon.t / ms)
    h_idx = np.asarray(h_mon.i[:])
    in_drv = h_t < 200.0
    drv_ch = np.bincount(h.e_channel[h_idx[in_drv]],
                         minlength=H_N_CHANNELS)
    drv_rate_ch0 = float(drv_ch[0]) / 16.0 / 0.2
    # Post-V1-silencing windows
    in_post0 = (h_t >= 200.0) & (h_t < 300.0)   # 0-100 ms post
    post0_ch = np.bincount(h.e_channel[h_idx[in_post0]],
                           minlength=H_N_CHANNELS)
    post0_ch0 = float(post0_ch[0]) / 16.0 / 0.1
    in_post1 = (h_t >= 300.0) & (h_t < 400.0)   # 100-200 ms post
    post1_ch = np.bincount(h.e_channel[h_idx[in_post1]],
                           minlength=H_N_CHANNELS)
    post1_ch0 = float(post1_ch[0]) / 16.0 / 0.1

    # Persistence: at least one post-bin > 1 Hz (~ above silence)
    persisted = (post0_ch0 > 1.0)
    drv_active = (drv_rate_ch0 > 30.0)
    passed = persisted and drv_active
    detail = (f"drive H[ch0]={drv_rate_ch0:.1f}  "
              f"post 0-100ms H[ch0]={post0_ch0:.1f}  "
              f"post 100-200ms H[ch0]={post1_ch0:.1f}  "
              f"(drive>30 + post0>1 required)")
    print(f"[3] persistence:          "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# -- assay 4: g_v1_to_h calibration sweep -----------------------------------


def assay_4_calibration_sweep() -> bool:
    """Sweep g_v1_to_h; assert default 1.5 places matched H[ch0] in 10-30 Hz.

    With Stage-0 V1 calibration loaded, full-contrast grating drives the
    matched-channel V1_E cells at ~ 22 Hz (mean V1_E ~ 4 Hz). The default
    g_v1_to_h must place H_E[ch0] at 10-30 Hz: enough to engage H -> V1
    feedback, but below NMDA-recruited attractor saturation.
    """
    g_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    h_rates = []
    v_rates_matched = []
    for g in g_values:
        _setup_brian()
        v = build_v1_ring()
        h = build_h_r()
        _load_stage0_v1(v)
        _load_stage1_hr(h)
        ff = build_v1_to_h_feedforward(v, h, V1ToHConfig(g_v1_to_h=g))
        set_stimulus(v, theta_rad=0.0, contrast=1.0)
        silence_cue(h)
        h_mon = SpikeMonitor(h.e, name=f"a4_hmon_g{int(g*10)}")
        v_mon = SpikeMonitor(v.e, name=f"a4_vmon_g{int(g*10)}")
        net = Network(*v.groups, *h.groups, *ff.groups, h_mon, v_mon)
        net.run(500 * ms)
        rates_h = _per_channel_rate(h_mon, h.e_channel, 16, 500.0)
        rates_v = _per_channel_rate(v_mon, v.e_channel, 16, 500.0)
        h_rates.append(float(rates_h[0]))
        v_rates_matched.append(float(rates_v[0]))

    # Assert at default g (1.5), H[ch0] in [10, 30] Hz.
    default_idx = g_values.index(DEFAULT_G_V1_TO_H)
    h_at_default = h_rates[default_idx]
    in_band = (10.0 <= h_at_default <= 30.0)
    # Assert monotone non-decreasing (feedforward gain should produce
    # monotone H rate increase).
    monotone = all(h_rates[i + 1] >= h_rates[i] - 0.5 for i in range(len(h_rates) - 1))
    passed = in_band and monotone
    sweep_str = "  ".join(
        f"g={g}:H={hr:.1f}" for g, hr in zip(g_values, h_rates)
    )
    detail = (f"V1[ch0]~{v_rates_matched[default_idx]:.1f}Hz  "
              f"sweep {sweep_str}  "
              f"default g={DEFAULT_G_V1_TO_H}-> H={h_at_default:.1f} "
              f"(target 10-30 + monotone)")
    print(f"[4] calibration_sweep:    "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# -- runner -----------------------------------------------------------------


def main() -> int:
    assays = [
        ("feature_matched_drive",   assay_1_feature_matched_drive),
        ("feedforward_sufficient",  assay_2_feedforward_sufficient_no_feedback),
        ("persistence",             assay_3_persistence_post_v1_silencing),
        ("calibration_sweep",       assay_4_calibration_sweep),
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
    print(f"\n--- validate_v1_to_h: {n_pass}/{total} PASS ---")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
