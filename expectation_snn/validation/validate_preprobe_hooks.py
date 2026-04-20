"""Functional validation for Sprint 5d pre-probe H instrumentation (task #41 step 4).

Why this validator exists
-------------------------
Sprint 5c meta-review identifies Case A (H learning/memory broken) and
Case B (feedback routing broken). Both need to compare H firing *just
before* the probe window (pre-probe) against H during the probe — and
that requires a SpikeMonitor on H_E that records per-channel rates in
the last ``preprobe_window_ms`` of the epoch preceding the probe.

This module wires that instrumentation through three assays:

  - Kok:     last 100 ms of gap (before grating onset)
  - Richter: last 100 ms of leader (before trailer onset)
  - Tang:    last 50 ms of item t (before item t+1 onset)

This validator proves the plumbing is correct. It does NOT evaluate any
science claim about H firing distributions — that's the Debugger's
job (task #42). The three functional checks per assay are:

  [1] {assay}_preprobe_off_returns_none
        Default build (with_preprobe_h_mon=False). Expect:
          - bundle.h_e_mon is None
          - raw["h_preprobe_rate_hz"] is None
          - snapshot_h_counts(bundle) raises RuntimeError

  [2] {assay}_preprobe_on_reports_finite_rates
        Build with with_preprobe_h_mon=True. Expect:
          - bundle.h_e_mon is a SpikeMonitor on h.e
          - raw["h_preprobe_rate_hz"].shape == (n_trials, 12)
          - all rates are finite, >= 0, with at least some > 0 (sanity:
            V1→H is on in the default "continuous" mode, so H fires)
          - raw["preprobe_window_ms"] == configured value

  [3] preprobe_h_rate_hz_helper_math
        Unit test of the rate conversion: fake counts_before/counts_after
        arrays, known h_ring, known window — expected Hz per channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.assays.kok_passive import KokConfig, run_kok_passive
from expectation_snn.assays.richter_crossover import (
    RichterConfig, run_richter_crossover,
)
from expectation_snn.assays.tang_rotating import TangConfig, run_tang_rotating
from expectation_snn.assays.runtime import (
    build_frozen_network, snapshot_h_counts, preprobe_h_rate_hz,
)


H_N_CHANNELS = 12


# ---------------------------------------------------------------------------
# Mini configs (short paradigms — all < 60 s wall time each)
# ---------------------------------------------------------------------------

def _mini_kok(**overrides) -> KokConfig:
    # n_stim_trials must be even (cue-balanced) and the 75/25 validity split
    # drops to "all valid" at n_stim_trials=4 (per_cue=2 → valid=2 invalid=0).
    # 12 keeps some invalid trials (per_cue=6 → valid=5 invalid=1 × 2 cues).
    base = dict(
        n_stim_trials=12, n_omission_trials=0,
        cue_ms=200.0, gap_ms=200.0, grating_ms=200.0, iti_ms=100.0,
        mvpa_n_subsamples=2, mvpa_n_bootstrap=20, mvpa_cv=2,
        preprobe_window_ms=100.0,
        seed=42,
    )
    base.update(overrides)
    return KokConfig(**base)


def _mini_richter(**overrides) -> RichterConfig:
    # Richter n_trials is derived from reps_expected (6 pair types) +
    # reps_unexpected (24 pair types). Use reps_expected=1 to get 6 trials
    # and reps_unexpected=0 for a minimal run.
    base = dict(
        reps_expected=1, reps_unexpected=0,
        leader_ms=300.0, trailer_ms=200.0, iti_ms=100.0,
        preprobe_window_ms=100.0,
        seed=42,
    )
    base.update(overrides)
    return RichterConfig(**base)


def _mini_tang(**overrides) -> TangConfig:
    base = dict(
        n_random=4, n_rotating=0, item_ms=150.0, presettle_ms=100.0,
        preprobe_window_ms=50.0, seed=42,
    )
    base.update(overrides)
    return TangConfig(**base)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def _assert_preprobe_on_shape(
    raw, n_trials: int, win_ms: float, label: str,
    require_active_h: bool,
) -> None:
    """Shape + finiteness + optional non-silence assertion.

    ``require_active_h`` is True for assays whose preprobe window lies
    *during* an active stimulus (Richter leader, Tang item) — H should
    fire. It is False for Kok (preprobe is the gap between cue and
    grating, i.e. no stimulus and no cue active): here H may legitimately
    be silent; that silence is the scientific question for diagnostic
    D1/D2, not an infrastructure defect.
    """
    arr = raw.get("h_preprobe_rate_hz")
    assert arr is not None, f"{label}: raw.h_preprobe_rate_hz is None"
    arr = np.asarray(arr)
    assert arr.shape == (n_trials, H_N_CHANNELS), (
        f"{label}: h_preprobe_rate_hz shape={arr.shape}, "
        f"expected ({n_trials}, {H_N_CHANNELS})"
    )
    assert np.all(np.isfinite(arr)), (
        f"{label}: h_preprobe_rate_hz has non-finite entries: {arr}"
    )
    assert np.all(arr >= 0.0), (
        f"{label}: h_preprobe_rate_hz has negative entries: {arr.min()}"
    )
    if require_active_h:
        assert arr.max() > 0.0, (
            f"{label}: H entirely silent during a stimulus-active window — "
            f"V1→H feedforward may be broken. rate array = {arr}"
        )
    assert abs(float(raw["preprobe_window_ms"]) - win_ms) < 1e-9, (
        f"{label}: preprobe_window_ms={raw['preprobe_window_ms']}, "
        f"expected {win_ms}"
    )


def _assert_preprobe_off(raw, label: str) -> None:
    arr = raw.get("h_preprobe_rate_hz", None)
    # Off-mode contract: field is explicitly None (not absent, since assays
    # always populate the raw dict).
    assert arr is None, (
        f"{label}: expected None h_preprobe_rate_hz with preprobe OFF, "
        f"got {type(arr).__name__}"
    )


# ---------------------------------------------------------------------------
# Assays
# ---------------------------------------------------------------------------

def assay_kok_preprobe_off_returns_none() -> None:
    print("[1] kok_preprobe_off_returns_none")
    bundle = build_frozen_network(
        h_kind="hr", seed=42, with_cue=True, with_preprobe_h_mon=False,
    )
    assert bundle.h_e_mon is None, "bundle.h_e_mon should be None when off"
    # Ensure helper rejects the off case explicitly.
    try:
        snapshot_h_counts(bundle)
    except RuntimeError as exc:
        assert "with_preprobe_h_mon=True" in str(exc), str(exc)
    else:
        raise AssertionError(
            "snapshot_h_counts should raise RuntimeError when h_e_mon is None"
        )

    cfg = _mini_kok()
    res = run_kok_passive(bundle=bundle, cfg=cfg, verbose=False)
    _assert_preprobe_off(res.raw, "Kok[off]")
    print("    bundle.h_e_mon=None  raw.h_preprobe_rate_hz=None  PASS")


def assay_kok_preprobe_on_reports_finite_rates() -> None:
    print("[2] kok_preprobe_on_reports_finite_rates")
    bundle = build_frozen_network(
        h_kind="hr", seed=42, with_cue=True, with_preprobe_h_mon=True,
    )
    assert bundle.h_e_mon is not None, "bundle.h_e_mon should be live when on"
    # snapshot returns a (h.e.N,) int64 array of cumulative counts.
    snap = snapshot_h_counts(bundle)
    assert snap.dtype == np.int64 and snap.ndim == 1, (snap.dtype, snap.shape)

    cfg = _mini_kok()
    res = run_kok_passive(bundle=bundle, cfg=cfg, verbose=False)
    n_trials_kok = cfg.n_stim_trials + cfg.n_omission_trials
    # Kok preprobe is the last 100 ms of the gap (blank between cue and
    # grating); H may be silent. Only check shape & finiteness here.
    _assert_preprobe_on_shape(
        res.raw, n_trials=n_trials_kok, win_ms=cfg.preprobe_window_ms,
        label="Kok[on]", require_active_h=False,
    )
    # Defensive: sanity-check max rate is not absurd (<1 kHz).
    assert np.asarray(res.raw["h_preprobe_rate_hz"]).max() < 1000.0
    print(f"    preprobe_rate shape={res.raw['h_preprobe_rate_hz'].shape}  "
          f"max={np.asarray(res.raw['h_preprobe_rate_hz']).max():.2f}Hz  PASS")


def assay_richter_preprobe_off_returns_none() -> None:
    print("[3] richter_preprobe_off_returns_none")
    bundle = build_frozen_network(
        h_kind="hr", seed=42, with_cue=False, with_preprobe_h_mon=False,
    )
    assert bundle.h_e_mon is None
    cfg = _mini_richter()
    res = run_richter_crossover(bundle=bundle, cfg=cfg, verbose=False)
    _assert_preprobe_off(res.raw, "Richter[off]")
    print("    n_trials={}  raw.h_preprobe_rate_hz=None  PASS"
          .format(int(res.meta['n_trials'])))


def assay_richter_preprobe_on_reports_finite_rates() -> None:
    print("[4] richter_preprobe_on_reports_finite_rates")
    bundle = build_frozen_network(
        h_kind="hr", seed=42, with_cue=False, with_preprobe_h_mon=True,
    )
    assert bundle.h_e_mon is not None
    cfg = _mini_richter()
    res = run_richter_crossover(bundle=bundle, cfg=cfg, verbose=False)
    n_trials_ric = int(res.meta["n_trials"])
    # Richter preprobe is the last 100 ms of the leader grating → H driven.
    _assert_preprobe_on_shape(
        res.raw, n_trials=n_trials_ric, win_ms=cfg.preprobe_window_ms,
        label="Richter[on]", require_active_h=True,
    )
    print(f"    preprobe_rate shape={res.raw['h_preprobe_rate_hz'].shape}  "
          f"max={np.asarray(res.raw['h_preprobe_rate_hz']).max():.2f}Hz  PASS")


def assay_tang_preprobe_off_returns_none() -> None:
    print("[5] tang_preprobe_off_returns_none")
    bundle = build_frozen_network(
        h_kind="ht", seed=42, with_cue=False, with_preprobe_h_mon=False,
    )
    assert bundle.h_e_mon is None
    cfg = _mini_tang()
    res = run_tang_rotating(bundle=bundle, cfg=cfg, verbose=False)
    _assert_preprobe_off(res.raw, "Tang[off]")
    print("    raw.h_preprobe_rate_hz=None  PASS")


def assay_tang_preprobe_on_reports_finite_rates() -> None:
    print("[6] tang_preprobe_on_reports_finite_rates")
    bundle = build_frozen_network(
        h_kind="ht", seed=42, with_cue=False, with_preprobe_h_mon=True,
    )
    assert bundle.h_e_mon is not None
    cfg = _mini_tang()
    res = run_tang_rotating(bundle=bundle, cfg=cfg, verbose=False)
    n_items = int(cfg.n_random) + int(cfg.n_rotating)
    # Tang preprobe is the last 50 ms of item t (stimulus is on) → H driven.
    _assert_preprobe_on_shape(
        res.raw, n_trials=n_items, win_ms=cfg.preprobe_window_ms,
        label="Tang[on]", require_active_h=True,
    )
    print(f"    preprobe_rate shape={res.raw['h_preprobe_rate_hz'].shape}  "
          f"max={np.asarray(res.raw['h_preprobe_rate_hz']).max():.2f}Hz  PASS")


def assay_preprobe_h_rate_hz_helper_math() -> None:
    """Unit test of the per-H-channel rate conversion math."""
    print("[7] preprobe_h_rate_hz_helper_math")
    # Minimal fake h_ring: 12 channels × 16 cells; ch=0 gets a spike each
    # into two cells; ch=1 gets three spikes; rest silent.
    class _FakeE:
        def __init__(self, N):
            self.N = int(N)

    class _FakeHRing:
        def __init__(self):
            self.e = _FakeE(192)
            self.e_channel = np.repeat(np.arange(12), 16).astype(np.int64)

    h_ring = _FakeHRing()
    cnt_before = np.zeros(192, dtype=np.int64)
    cnt_after = np.zeros(192, dtype=np.int64)
    cnt_after[0] = 1; cnt_after[1] = 1                   # ch 0: 2 spikes / 16 cells
    cnt_after[16] = 2; cnt_after[17] = 1                 # ch 1: 3 spikes / 16 cells
    window_ms = 100.0
    rate = preprobe_h_rate_hz(cnt_before, cnt_after, h_ring, window_ms)
    assert rate.shape == (12,), rate.shape
    # ch 0: 2 spikes / (16 cells × 0.1 s) = 1.25 Hz
    # ch 1: 3 spikes / (16 cells × 0.1 s) = 1.875 Hz
    # others: 0
    assert abs(rate[0] - 1.25) < 1e-9, rate[0]
    assert abs(rate[1] - 1.875) < 1e-9, rate[1]
    assert np.all(rate[2:] == 0.0), rate
    # Error branches: window_ms <= 0
    try:
        preprobe_h_rate_hz(cnt_before, cnt_after, h_ring, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("window_ms=0 should raise")
    print("    ch0=1.25Hz ch1=1.875Hz  error branches OK  PASS")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    np.random.seed(42)
    assays = [
        assay_preprobe_h_rate_hz_helper_math,   # pure unit — fastest, run first
        assay_kok_preprobe_off_returns_none,
        assay_kok_preprobe_on_reports_finite_rates,
        assay_richter_preprobe_off_returns_none,
        assay_richter_preprobe_on_reports_finite_rates,
        assay_tang_preprobe_off_returns_none,
        assay_tang_preprobe_on_reports_finite_rates,
    ]
    failed: list[str] = []
    for a in assays:
        try:
            a()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{a.__name__}: {exc}")
            print(f"    FAIL — {exc}")
    n = len(assays)
    print()
    print(f"validate_preprobe_hooks: {n - len(failed)}/{n} PASS")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
