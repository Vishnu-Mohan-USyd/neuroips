"""Functional component validation for the Kok passive assay.

Per-component rule (task #27): passes BEFORE integration into Sprint 5a.

Checks (all local, no full 288-trial run):

1. Schedule builder: 240 stim + 48 omission at default, balanced A/B.
2. Validity split: 180 valid / 60 invalid at 0.75 validity.
3. Orientations: cue_A expected = 45°, cue_B expected = 135°.
4. Invalid trials: theta differs from expected by π/2 (orthogonal).
5. Omission trials: theta is NaN, grating is blank.
6. End-to-end smoke: tiny config runs, all 4 primary metrics return
   well-formed dicts with the right shapes + finite numbers.

Usage
-----
    python -m expectation_snn.validation.validate_kok_passive
"""
from __future__ import annotations

import numpy as np

from expectation_snn.assays.kok_passive import (
    KokConfig, KokResult,
    build_kok_schedule, run_kok_passive,
)
from expectation_snn.assays.runtime import STAGE2_CUE_CHANNELS
from expectation_snn.brian2_model.h_ring import N_CHANNELS as H_N_CHANNELS


EXPECTED_A_RAD = float(STAGE2_CUE_CHANNELS[0]) * (np.pi / H_N_CHANNELS)  # 45°
EXPECTED_B_RAD = float(STAGE2_CUE_CHANNELS[1]) * (np.pi / H_N_CHANNELS)  # 135°


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def test_schedule_totals() -> None:
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    n_total = cfg.n_stim_trials + cfg.n_omission_trials
    assert len(sch) == n_total, f"expected {n_total}, got {len(sch)}"
    n_stim = sum(1 for t in sch if not t["is_omission"])
    n_om = sum(1 for t in sch if t["is_omission"])
    assert n_stim == cfg.n_stim_trials, (n_stim, cfg.n_stim_trials)
    assert n_om == cfg.n_omission_trials, (n_om, cfg.n_omission_trials)


def test_cue_balance() -> None:
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    a = sum(1 for t in sch if t["cue"] == "A")
    b = sum(1 for t in sch if t["cue"] == "B")
    n = cfg.n_stim_trials + cfg.n_omission_trials
    assert a == b == n // 2, f"cue split A={a} B={b} (n={n})"


def test_validity_split() -> None:
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    stim = [t for t in sch if not t["is_omission"]]
    n_valid = sum(1 for t in stim if t["condition"] == 1)
    n_invalid = len(stim) - n_valid
    expected_valid = int(round(cfg.n_stim_trials * cfg.validity))
    assert n_valid == expected_valid, (n_valid, expected_valid)
    assert n_invalid == cfg.n_stim_trials - expected_valid, (n_invalid,)


def test_cue_to_orientation_map() -> None:
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    for t in sch:
        if t["cue"] == "A":
            assert _approx(t["expected_rad"], EXPECTED_A_RAD), t
        else:
            assert _approx(t["expected_rad"], EXPECTED_B_RAD), t


def test_valid_vs_invalid_orientation() -> None:
    """Valid: theta = expected. Invalid: theta = orthogonal (Δθ = π/2)."""
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    for t in sch:
        if t["is_omission"]:
            continue
        dtheta = abs(t["theta_rad"] - t["expected_rad"])
        dtheta_wrapped = min(dtheta, np.pi - dtheta)
        if t["condition"] == 1:
            assert _approx(dtheta_wrapped, 0.0), t
        else:
            assert _approx(dtheta_wrapped, np.pi / 2.0), t


def test_omission_theta_is_nan() -> None:
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    for t in sch:
        if t["is_omission"]:
            assert np.isnan(t["theta_rad"]), t
        else:
            assert not np.isnan(t["theta_rad"]), t


def test_per_cue_validity_balance() -> None:
    """75 % valid / 25 % invalid *within each cue* (90/30 at defaults)."""
    cfg = KokConfig()
    sch = build_kok_schedule(cfg)
    for cue in ("A", "B"):
        stim = [t for t in sch if t["cue"] == cue and not t["is_omission"]]
        n_valid = sum(1 for t in stim if t["condition"] == 1)
        per_cue = cfg.n_stim_trials // 2
        expected = int(round(per_cue * cfg.validity))
        assert n_valid == expected, (cue, n_valid, expected)


def test_end_to_end_smoke() -> None:
    """Tiny run — asserts all 4 primary metrics populate correctly."""
    cfg = KokConfig(
        n_stim_trials=16, n_omission_trials=4,
        cue_ms=200.0, gap_ms=200.0, grating_ms=200.0, iti_ms=300.0,
        seed=42,
    )
    r = run_kok_passive(cfg=cfg)
    assert isinstance(r, KokResult)
    # Metric 1: mean amp valid + invalid dicts
    for cond in ("valid", "invalid"):
        d = r.mean_amp[cond]
        assert "total_rate_hz" in d and np.isfinite(d["total_rate_hz"]), d
        assert "total_rate_hz_ci" in d
    # Metric 2: SVM
    assert 0.0 <= r.svm["accuracy"] <= 1.0
    # Metric 3: preference rank
    assert r.pref_rank["bin_delta"].shape == (10,)
    # Metric 4: omission-subtracted per cell
    n_e = r.raw["trial_grating_counts"].shape[0]
    assert r.omission.shape == (n_e,)
    # Shape sanity
    gc = r.raw["trial_grating_counts"]
    assert gc.shape == (n_e, cfg.n_stim_trials + cfg.n_omission_trials)
    assert (gc >= 0).all()


_CHECKS = [
    ("schedule_totals", test_schedule_totals),
    ("cue_balance", test_cue_balance),
    ("validity_split", test_validity_split),
    ("cue_to_orientation_map", test_cue_to_orientation_map),
    ("valid_vs_invalid_orientation", test_valid_vs_invalid_orientation),
    ("omission_theta_is_nan", test_omission_theta_is_nan),
    ("per_cue_validity_balance", test_per_cue_validity_balance),
    ("end_to_end_smoke", test_end_to_end_smoke),
]


def main() -> int:
    n_pass = 0
    n_fail = 0
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            n_fail += 1
    total = n_pass + n_fail
    print(f"\nvalidate_kok_passive: {n_pass}/{total} PASS")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
