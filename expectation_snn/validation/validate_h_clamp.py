"""Functional validation for the Sprint 5d diagnostic H-clamp pathway.

Sprint 5d-infra step 2 (task #41). Validates the HClamp module and its
runtime wiring:

  [1] clamp_activates_target_channel_standalone
        Build a standalone H_R ring + HClamp (ch=0, 200 Hz, no V1→H),
        run a 3-phase trial (pre / clamp on / post). Expect:
          - pre-window  H_E[ch0] < 2 Hz
          - on-window   H_E[ch0] > 20 Hz

  [2] clamp_off_mode_is_no_op
        Build HClamp but never call set_active(True). Target-channel
        rate remains < 2 Hz (same as the no-clamp baseline).

  [3] clamp_with_v1_to_h_off_drives_h_from_clamp_alone
        build_frozen_network(with_v1_to_h="off",
                             with_h_clamp=HClampConfig(target_channel=0,
                                                       clamp_rate_hz=200)).
        Enable clamp; with V1→H OFF and no cue, H_E[ch0] must still
        fire > 20 Hz — confirms clamp works independently of V1→H.

  [4] clamp_set_active_roundtrip
        Toggle set_active(True) → (False) → (True); weights must round-
        trip exactly to (active_w → 0 → active_w).

  [5] clamp_independent_of_feedback_routes_toggle
        build_frozen_network(with_feedback_routes=False,
                             with_h_clamp=HClampConfig(...)) — confirm
        fb.g_direct == fb.g_SOM == 0 AND H_E[ch0] still rises on clamp
        (clamp is independent of the feedback-routes toggle).

All assays use seed=42, Brian2 numpy codegen, dt=0.1 ms. Fresh network
per assay (Brian2 state is per-Network).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from brian2 import (
    Network, SpikeMonitor, defaultclock, prefs, ms,
    seed as b2_seed,
)

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.assays.runtime import build_frozen_network
from expectation_snn.brian2_model.h_clamp import (
    HClampConfig, build_h_clamp,
)
from expectation_snn.brian2_model.h_ring import (
    build_h_r, silence_cue, N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER_CHANNEL,
)


SEED = 42
DT_MS = 0.1


def _setup_brian():
    prefs.codegen.target = "numpy"
    defaultclock.dt = DT_MS * ms
    b2_seed(SEED)
    np.random.seed(SEED)


def _channel_rate_hz(spike_i, spike_t_ms, channel_of, n_per_ch,
                     target_ch, t0_ms, t1_ms):
    """Mean firing rate (Hz) of `target_ch`'s E cells inside [t0,t1)."""
    if len(spike_t_ms) == 0:
        return 0.0
    m = (spike_t_ms >= t0_ms) & (spike_t_ms < t1_ms)
    if not m.any():
        return 0.0
    ch_hits = channel_of[spike_i[m]]
    n_sp = int(np.sum(ch_hits == target_ch))
    dur_s = (t1_ms - t0_ms) / 1000.0
    return float(n_sp) / (n_per_ch * dur_s)


# --------------------------------------------------------------------------
# Assay 1 — standalone clamp drives target channel
# --------------------------------------------------------------------------

def assay_1_clamp_activates_target_channel_standalone() -> bool:
    _setup_brian()
    h = build_h_r()
    silence_cue(h)
    cfg = HClampConfig(target_channel=0, clamp_rate_hz=200.0)
    clamp = build_h_clamp(h, cfg)

    h_mon = SpikeMonitor(h.e, name="c1_h_mon")
    net = Network(*h.groups, *clamp.groups, h_mon)

    # Phase A: 200 ms pre-window (clamp off)
    net.run(200 * ms)
    # Phase B: 200 ms clamp on
    clamp.set_active(True)
    net.run(200 * ms)
    # Phase C: 200 ms post-window (clamp off)
    clamp.set_active(False)
    net.run(200 * ms)

    si = np.asarray(h_mon.i[:])
    st = np.asarray(h_mon.t / ms)
    n_per_ch = H_N_E_PER_CHANNEL
    r_pre  = _channel_rate_hz(si, st, h.e_channel, n_per_ch, 0, 0.0, 200.0)
    r_on   = _channel_rate_hz(si, st, h.e_channel, n_per_ch, 0, 200.0, 400.0)
    r_post = _channel_rate_hz(si, st, h.e_channel, n_per_ch, 0, 400.0, 600.0)

    pre_ok = r_pre < 2.0
    on_ok  = r_on > 20.0
    passed = pre_ok and on_ok
    detail = (f"pre={r_pre:.2f} Hz (<2)  on={r_on:.2f} Hz (>20)  "
              f"post={r_post:.2f} Hz")
    print(f"[1] clamp_activates_target_channel:         "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 2 — clamp OFF mode is a no-op
# --------------------------------------------------------------------------

def assay_2_clamp_off_mode_is_no_op() -> bool:
    _setup_brian()
    h = build_h_r()
    silence_cue(h)
    cfg = HClampConfig(target_channel=0, clamp_rate_hz=200.0)
    clamp = build_h_clamp(h, cfg)

    h_mon = SpikeMonitor(h.e, name="c2_h_mon")
    net = Network(*h.groups, *clamp.groups, h_mon)
    # Never activate — clamp synapses stay w=0 for full 400 ms.
    net.run(400 * ms)

    si = np.asarray(h_mon.i[:])
    st = np.asarray(h_mon.t / ms)
    n_per_ch = H_N_E_PER_CHANNEL
    r_full = _channel_rate_hz(si, st, h.e_channel, n_per_ch, 0, 0.0, 400.0)
    # Clamp weights must have stayed at 0 throughout
    w_final = np.asarray(clamp.clamp_to_he.w[:])
    w_all_zero = bool(np.all(w_final == 0.0))

    passed = (r_full < 2.0) and w_all_zero
    detail = (f"ch0 full-window rate={r_full:.2f} Hz (<2); "
              f"weights all zero={w_all_zero}")
    print(f"[2] clamp_off_mode_is_no_op:                "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 3 — clamp drives H with V1→H OFF (no V1 amplifier)
# --------------------------------------------------------------------------

def assay_3_clamp_with_v1_to_h_off() -> bool:
    """build_frozen_network with with_v1_to_h='off' + h_clamp → H fires
    only due to the clamp, not V1→H."""
    _setup_brian()
    cfg = HClampConfig(target_channel=0, clamp_rate_hz=200.0)
    bundle = build_frozen_network(
        h_kind="hr", seed=SEED, r=1.0, g_total=1.0,
        with_cue=False,
        with_v1_to_h="off",
        with_h_clamp=cfg,
    )
    assert bundle.v1_to_h is None, "V1→H should be absent in off mode"
    assert bundle.h_clamp is not None, "h_clamp must be built"

    h_mon = SpikeMonitor(bundle.h_ring.e, name="c3_h_mon")
    net = Network(*bundle.groups, h_mon)
    bundle.reset_all()

    # Phase A: 200 ms silent (clamp off)
    net.run(200 * ms)
    # Phase B: 200 ms clamp on
    bundle.h_clamp.set_active(True)
    net.run(200 * ms)
    # Phase C: 200 ms silent (clamp off again)
    bundle.h_clamp.set_active(False)
    net.run(200 * ms)

    si = np.asarray(h_mon.i[:])
    st = np.asarray(h_mon.t / ms)
    n_per_ch = H_N_E_PER_CHANNEL
    r_pre  = _channel_rate_hz(si, st, bundle.h_ring.e_channel, n_per_ch, 0,
                              0.0, 200.0)
    r_on   = _channel_rate_hz(si, st, bundle.h_ring.e_channel, n_per_ch, 0,
                              200.0, 400.0)
    r_post = _channel_rate_hz(si, st, bundle.h_ring.e_channel, n_per_ch, 0,
                              400.0, 600.0)

    pre_ok = r_pre < 2.0
    on_ok = r_on > 20.0
    passed = pre_ok and on_ok
    detail = (f"V1→H=off clamp ch0: pre={r_pre:.2f} on={r_on:.2f} "
              f"post={r_post:.2f} (pre<2 & on>20 required); "
              f"meta target_channel={bundle.meta.get('h_clamp_target_channel')}")
    print(f"[3] clamp_with_v1_to_h_off:                 "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 4 — set_active round-trip exactness
# --------------------------------------------------------------------------

def assay_4_set_active_roundtrip() -> bool:
    _setup_brian()
    h = build_h_r()
    cfg = HClampConfig(target_channel=3, clamp_rate_hz=150.0)
    clamp = build_h_clamp(h, cfg)

    w0 = np.asarray(clamp.clamp_to_he.w[:]).copy()
    eq_off = bool(np.all(w0 == 0.0))

    clamp.set_active(True)
    w_on = np.asarray(clamp.clamp_to_he.w[:])
    eq_on = bool(np.allclose(w_on, clamp.active_w))
    all_targeted = (len(w_on) > 0) and bool(np.all(w_on > 0.0))

    clamp.set_active(False)
    w_off = np.asarray(clamp.clamp_to_he.w[:])
    eq_off_after = bool(np.all(w_off == 0.0))

    clamp.set_active(True)
    w_on2 = np.asarray(clamp.clamp_to_he.w[:])
    eq_on_after = bool(np.allclose(w_on2, clamp.active_w))

    passed = eq_off and eq_on and all_targeted and eq_off_after and eq_on_after
    detail = (f"init-zero={eq_off}  on-matches-active_w={eq_on}  "
              f"on-nonzero={all_targeted}  off-after-on={eq_off_after}  "
              f"second-on={eq_on_after}  active_w={clamp.active_w:.3f}")
    print(f"[4] clamp_set_active_roundtrip:             "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 5 — clamp independent of feedback_routes toggle
# --------------------------------------------------------------------------

def assay_5_clamp_independent_of_feedback_routes() -> bool:
    """with_feedback_routes=False zeroes H→V1 routes; clamp still works."""
    _setup_brian()
    cfg = HClampConfig(target_channel=0, clamp_rate_hz=200.0)
    bundle = build_frozen_network(
        h_kind="hr", seed=SEED, r=1.0, g_total=1.0,
        with_cue=False,
        with_v1_to_h="off",
        with_feedback_routes=False,
        with_h_clamp=cfg,
    )
    gd = bundle.fb.g_direct
    gs = bundle.fb.g_SOM
    fb_zeroed = (gd == 0.0) and (gs == 0.0)
    meta_flag = (bundle.meta.get("with_feedback_routes") is False)
    has_clamp = bundle.h_clamp is not None

    h_mon = SpikeMonitor(bundle.h_ring.e, name="c5_h_mon")
    net = Network(*bundle.groups, h_mon)
    bundle.reset_all()

    net.run(200 * ms)             # pre-window
    bundle.h_clamp.set_active(True)
    net.run(200 * ms)             # clamp on
    bundle.h_clamp.set_active(False)

    si = np.asarray(h_mon.i[:])
    st = np.asarray(h_mon.t / ms)
    n_per_ch = H_N_E_PER_CHANNEL
    r_pre = _channel_rate_hz(si, st, bundle.h_ring.e_channel, n_per_ch, 0,
                             0.0, 200.0)
    r_on = _channel_rate_hz(si, st, bundle.h_ring.e_channel, n_per_ch, 0,
                            200.0, 400.0)
    clamp_ok = (r_pre < 2.0) and (r_on > 20.0)

    passed = fb_zeroed and meta_flag and has_clamp and clamp_ok
    detail = (f"fb zeroed={fb_zeroed} (g_direct={gd:.3f} g_SOM={gs:.3f})  "
              f"meta flag={meta_flag}  clamp present={has_clamp}  "
              f"ch0 pre={r_pre:.2f} on={r_on:.2f}")
    print(f"[5] clamp_independent_of_feedback_routes:   "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

_ASSAYS = [
    ("clamp_activates_target_channel",      assay_1_clamp_activates_target_channel_standalone),
    ("clamp_off_mode_is_no_op",             assay_2_clamp_off_mode_is_no_op),
    ("clamp_with_v1_to_h_off",              assay_3_clamp_with_v1_to_h_off),
    ("clamp_set_active_roundtrip",          assay_4_set_active_roundtrip),
    ("clamp_independent_of_feedback_routes", assay_5_clamp_independent_of_feedback_routes),
]


def main() -> int:
    n_pass = 0
    for name, fn in _ASSAYS:
        try:
            ok = fn()
        except Exception as exc:
            print(f"[X] {name}: EXCEPTION {type(exc).__name__}: {exc}")
            ok = False
        if ok:
            n_pass += 1
    total = len(_ASSAYS)
    print(f"\n--- validate_h_clamp: {n_pass}/{total} PASS ---")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
