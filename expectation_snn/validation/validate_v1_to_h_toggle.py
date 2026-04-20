"""Functional validation for the Sprint 5c V1->H runtime toggle.

Sprint 5c step 2 (task #37). Validates the new with_v1_to_h modes:

  [1] set_active_in_place
        Build a V1ToH; call set_active(False) then set_active(True);
        confirm weights round-trip to kernel_w * g_v1_to_h exactly.

  [2] continuous_mode_h_active_during_probe
        build_frozen_network(with_v1_to_h="continuous") followed by a
        Kok-shape trial: H_R fires above floor during the grating window.
        Establishes the baseline behaviour matched by Sprint 5a/5b.

  [3] off_mode_h_silent_during_probe
        build_frozen_network(with_v1_to_h="off"): bundle.v1_to_h is None
        and H_R is silent (< 1 Hz) during the grating window. Matches the
        pre-V1->H Sprint 5b r-invariant artefact.

  [4] context_only_mode_h_active_in_cue_silent_in_probe
        build_frozen_network(with_v1_to_h="context_only") followed by a
        Kok-shape trial: H_R fires above floor during cue+gap (V1->H on),
        then drops to silence during the grating window after the assay
        loop calls set_active(False), then recovers under set_active(True).
        This is the "true prior vs amplifier" separation diagnostic for
        Sprint 5c.

  [5] tang_rejects_context_only
        run_tang_rotating with a context_only bundle raises ValueError.
        (Tang has no natural context window between back-to-back items.)

All assays use seed=42, Brian2 numpy codegen, dt=0.1 ms. Fresh network
per assay (Brian2 state is per-Network).
"""
from __future__ import annotations

import os
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

from expectation_snn.assays.runtime import (
    build_frozen_network, set_grating, STAGE2_CUE_ACTIVE_HZ,
)
from expectation_snn.assays.tang_rotating import (
    run_tang_rotating, TangConfig,
)
from expectation_snn.brian2_model.feedforward_v1_to_h import (
    build_v1_to_h_feedforward, V1ToHConfig,
)
from expectation_snn.brian2_model.h_ring import (
    build_h_r, N_CHANNELS as H_N_CHANNELS,
)
from expectation_snn.brian2_model.v1_ring import (
    build_v1_ring, N_CHANNELS as V1_N_CHANNELS,
)


SEED = 42
DT_MS = 0.1


def _setup_brian():
    prefs.codegen.target = "numpy"
    defaultclock.dt = DT_MS * ms
    b2_seed(SEED)
    np.random.seed(SEED)


def _per_channel_rate_hz(spike_i, channel_of, n_per_ch, win_ms,
                          n_channels=H_N_CHANNELS):
    """Per-channel firing rate (Hz) from raw spike indices over a window."""
    counts = np.bincount(channel_of[spike_i], minlength=n_channels)
    return counts.astype(np.float64) / (n_per_ch * win_ms * 1e-3)


def _window_rate_hz(spike_i, spike_t_ms, channel_of, n_per_ch,
                    t0_ms, t1_ms, n_channels=H_N_CHANNELS):
    """Mean per-channel rate (Hz) inside [t0,t1)."""
    if len(spike_t_ms) == 0:
        return np.zeros(n_channels)
    m = (spike_t_ms >= t0_ms) & (spike_t_ms < t1_ms)
    if not m.any():
        return np.zeros(n_channels)
    counts = np.bincount(channel_of[spike_i[m]], minlength=n_channels)
    dur_s = (t1_ms - t0_ms) / 1000.0
    return counts.astype(np.float64) / (n_per_ch * dur_s)


# --------------------------------------------------------------------------
# Assay 1 — set_active in-place toggle round-trips exactly
# --------------------------------------------------------------------------

def assay_1_set_active_in_place() -> bool:
    """Toggle weights off then on; assert round-trip exactness."""
    _setup_brian()
    v = build_v1_ring()
    h = build_h_r()
    cfg = V1ToHConfig()              # default g_v1_to_h = 1.5
    ff = build_v1_to_h_feedforward(v, h, cfg)

    w0 = np.asarray(ff.v1_to_he.w[:]).copy()
    expected_w_active = ff.kernel_w * ff.g_v1_to_h
    eq_initial = np.allclose(w0, expected_w_active)

    ff.set_active(False)
    w_off = np.asarray(ff.v1_to_he.w[:])
    all_zero = bool(np.all(w_off == 0.0))

    ff.set_active(True)
    w_on = np.asarray(ff.v1_to_he.w[:])
    eq_after = np.allclose(w_on, expected_w_active)

    # gain attribute should not have moved
    gain_unchanged = (ff.g_v1_to_h == cfg.g_v1_to_h)

    passed = eq_initial and all_zero and eq_after and gain_unchanged
    detail = (f"initial={eq_initial}  "
              f"off-zero={all_zero}  "
              f"on-restored={eq_after}  "
              f"gain={ff.g_v1_to_h:.3f} (cfg {cfg.g_v1_to_h:.3f})")
    print(f"[1] set_active_in_place:                  "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Helper: Kok-shape single trial measuring H rates per epoch
# --------------------------------------------------------------------------

def _run_one_kok_trial(bundle, *, cue, theta_rad, cue_ms=500.0, gap_ms=500.0,
                       grating_ms=500.0, contrast=1.0,
                       toggle_ff_during_grating=False):
    """Run one Kok-shape trial and return per-epoch H_R rates (Hz)."""
    h_mon = SpikeMonitor(bundle.h_ring.e, name="vt_h_mon")
    net = Network(*bundle.groups, h_mon)
    bundle.reset_all()

    t_pre = float(net.t / ms)
    set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
    net.run(200 * ms)

    t_cue_start = float(net.t / ms)
    bundle.cue_on(cue, rate_hz=STAGE2_CUE_ACTIVE_HZ)
    set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
    net.run(cue_ms * ms)

    t_gap_start = float(net.t / ms)
    bundle.cue_off()
    set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
    net.run(gap_ms * ms)

    t_grat_start = float(net.t / ms)
    set_grating(bundle.v1_ring, theta_rad=theta_rad, contrast=contrast)
    if toggle_ff_during_grating and bundle.v1_to_h is not None:
        bundle.v1_to_h.set_active(False)
    net.run(grating_ms * ms)
    if toggle_ff_during_grating and bundle.v1_to_h is not None:
        bundle.v1_to_h.set_active(True)
    t_grat_end = float(net.t / ms)

    spike_i = np.asarray(h_mon.i[:])
    spike_t = np.asarray(h_mon.t[:] / ms)
    cue_ch = 3 if cue == "A" else 9
    matched_ch = int(round((theta_rad / np.pi) * H_N_CHANNELS)) % H_N_CHANNELS
    n_per_ch = int(bundle.h_ring.e.N // H_N_CHANNELS)

    cue_rates = _window_rate_hz(spike_i, spike_t, bundle.h_ring.e_channel,
                                 n_per_ch, t_cue_start, t_gap_start)
    gap_rates = _window_rate_hz(spike_i, spike_t, bundle.h_ring.e_channel,
                                 n_per_ch, t_gap_start, t_grat_start)
    grat_rates = _window_rate_hz(spike_i, spike_t, bundle.h_ring.e_channel,
                                  n_per_ch, t_grat_start, t_grat_end)
    return {
        "cue_ch": int(cue_ch),
        "matched_ch": int(matched_ch),
        "cue_rate_cue_ch_hz": float(cue_rates[cue_ch]),
        "gap_rate_cue_ch_hz": float(gap_rates[cue_ch]),
        "grating_rate_matched_ch_hz": float(grat_rates[matched_ch]),
    }


# --------------------------------------------------------------------------
# Assay 2 — continuous mode: H active during the grating
# --------------------------------------------------------------------------

def assay_2_continuous_mode_h_active_during_probe() -> bool:
    """Default continuous mode: H_R matched-channel >= 5 Hz during grating."""
    _setup_brian()
    bundle = build_frozen_network(
        h_kind="hr", seed=SEED, r=1.0, g_total=1.0,
        with_cue=True, with_v1_to_h="continuous",
    )
    mode_meta = bundle.meta.get("v1_to_h_mode")
    has_v1_to_h = bundle.v1_to_h is not None
    out = _run_one_kok_trial(
        bundle, cue="A", theta_rad=np.pi * 3.0 / 12.0,
        toggle_ff_during_grating=False,
    )
    grat_active = out["grating_rate_matched_ch_hz"] >= 5.0
    passed = (mode_meta == "continuous") and has_v1_to_h and grat_active
    detail = (f"mode={mode_meta!r} v1_to_h={'present' if has_v1_to_h else 'NONE'}  "
              f"H[matched ch{out['matched_ch']}] grating="
              f"{out['grating_rate_matched_ch_hz']:.1f} Hz (>=5 required)")
    print(f"[2] continuous_h_active_during_probe:     "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 3 — off mode: H silent during the grating
# --------------------------------------------------------------------------

def assay_3_off_mode_h_silent_during_probe() -> bool:
    """Off mode: bundle.v1_to_h is None, H matched-channel < 1 Hz."""
    _setup_brian()
    bundle = build_frozen_network(
        h_kind="hr", seed=SEED, r=1.0, g_total=1.0,
        with_cue=True, with_v1_to_h="off",
    )
    mode_meta = bundle.meta.get("v1_to_h_mode")
    no_v1_to_h = bundle.v1_to_h is None
    # No v1_to_h_meta keys when off
    no_v1_to_h_meta_g = "v1_to_h_g" not in bundle.meta

    out = _run_one_kok_trial(
        bundle, cue="A", theta_rad=np.pi * 3.0 / 12.0,
        toggle_ff_during_grating=False,
    )
    grat_silent = out["grating_rate_matched_ch_hz"] < 1.0
    passed = (mode_meta == "off") and no_v1_to_h and no_v1_to_h_meta_g and grat_silent
    detail = (f"mode={mode_meta!r} v1_to_h={'NONE' if no_v1_to_h else 'present'}  "
              f"H[matched] grating={out['grating_rate_matched_ch_hz']:.2f} Hz "
              f"(<1 required)")
    print(f"[3] off_h_silent_during_probe:            "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 4 — context_only mode: H active in cue, silent in grating
# --------------------------------------------------------------------------

def assay_4_context_only_mode() -> bool:
    """Context-only mode: H_R has non-zero cue/gap rate (V1->H disabled
    matters less here because cue is the primary driver into H), but
    importantly the matched-channel H rate during grating drops to ~0
    after set_active(False) — confirming the assay loop's toggle hook
    really does silence the V1->H amplifier path.
    """
    _setup_brian()
    bundle = build_frozen_network(
        h_kind="hr", seed=SEED, r=1.0, g_total=1.0,
        with_cue=True, with_v1_to_h="context_only",
    )
    mode_meta = bundle.meta.get("v1_to_h_mode")
    has_v1_to_h = bundle.v1_to_h is not None
    out = _run_one_kok_trial(
        bundle, cue="A", theta_rad=np.pi * 3.0 / 12.0,
        toggle_ff_during_grating=True,
    )
    cue_active = out["cue_rate_cue_ch_hz"] > 5.0   # cue input drives H
    grat_silent = out["grating_rate_matched_ch_hz"] < 1.0
    # Confirm weights are restored after the trial (set_active(True) at end)
    w_after = np.asarray(bundle.v1_to_h.v1_to_he.w[:])
    expected_w = bundle.v1_to_h.kernel_w * bundle.v1_to_h.g_v1_to_h
    weights_restored = bool(np.allclose(w_after, expected_w))

    passed = ((mode_meta == "context_only") and has_v1_to_h
              and cue_active and grat_silent and weights_restored)
    detail = (f"mode={mode_meta!r}  cue rate ch{out['cue_ch']}="
              f"{out['cue_rate_cue_ch_hz']:.1f}  "
              f"grating rate matched ch{out['matched_ch']}="
              f"{out['grating_rate_matched_ch_hz']:.2f}  "
              f"restored={weights_restored}")
    print(f"[4] context_only_h_silent_in_probe:       "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Assay 5 — Tang rejects context_only
# --------------------------------------------------------------------------

def assay_5_tang_rejects_context_only() -> bool:
    """Tang assay raises ValueError on a context_only bundle."""
    _setup_brian()
    bundle = build_frozen_network(
        h_kind="ht", seed=SEED, r=1.0, g_total=1.0,
        with_cue=False, with_v1_to_h="context_only",
    )
    cfg = TangConfig(n_items=4, item_ms=50.0, presettle_ms=0.0,
                     block_len_range=(2, 3), seed=SEED)
    raised = False
    msg = ""
    try:
        run_tang_rotating(bundle=bundle, cfg=cfg, seed=SEED)
    except ValueError as e:
        raised = True
        msg = str(e)
    passed = raised and "context_only" in msg
    detail = (f"raised ValueError={raised}  msg has 'context_only'="
              f"{'context_only' in msg}")
    print(f"[5] tang_rejects_context_only:            "
          f"{'PASS' if passed else 'FAIL'} -- {detail}")
    return passed


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def main() -> int:
    assays = [
        ("set_active_in_place",                    assay_1_set_active_in_place),
        ("continuous_h_active_during_probe",       assay_2_continuous_mode_h_active_during_probe),
        ("off_h_silent_during_probe",              assay_3_off_mode_h_silent_during_probe),
        ("context_only_h_silent_in_probe",         assay_4_context_only_mode),
        ("tang_rejects_context_only",              assay_5_tang_rejects_context_only),
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
    print(f"\n--- validate_v1_to_h_toggle: {n_pass}/{total} PASS ---")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
