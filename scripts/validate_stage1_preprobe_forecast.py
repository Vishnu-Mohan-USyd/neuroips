"""Sprint 5e-Diag B4b: Stage-1 pre-trailer forecast validator.

The current Stage-1 gate (expectation_snn/validation/stage_1_gate.py line 161,
expectation_snn/brian2_model/train.py lines 721-740) measures H_R argmax at
`trailer_offset + 500 ms` — a POST-trailer window. A post-trailer argmax
that aligns with the leader channel is MEMORY (the leader bump persisting),
not FORECAST. A forecast gate must measure H_R BEFORE trailer onset.

This validator reuses the Sprint 5d D1 pre-probe data: `h_preprobe_rate_hz`
is the per-channel rate over the LAST 100 ms of the leader (i.e. immediately
pre-trailer) for each trial. We therefore test the already-trained
checkpoint (seed=42) against a pre-trailer forecast gate at zero additional
compute cost.

Gate:
  Under the Richter assay's biased schedule, the "expected trailer" for
  leader L is (L + 30°) which maps to H channel (ref_ch + 2) mod 12. A
  forecast-carrying H_R should point argmax at that expected-trailer
  channel significantly above chance (1/6).

  PASS  iff   P(argmax_pre_trailer == expected_trailer_ch) >= 0.25
             (>= 1.5× chance, safely above finite-sample noise band)

Current Stage-1 checkpoint trained on the balanced all-pairs schedule
(Bug 1) MUST fail this gate: without contingency to learn and with a gate
that measures memory instead of forecast, argmax locks on the leader
channel throughout the pre-trailer window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

IN_DIR = Path("data/diag_sprint5d")
OUT_DIR = Path("data/diag_sprint5e")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_H_CHANNELS = 12
# H uses 12 channels; Richter/Tang use 6 orientations at 30° spacing that
# map to alternating H channels [0, 2, 4, 6, 8, 10]. Orient step = 1 in
# 6-orient idx ↔ channel step = 2 in 12-channel idx.
H_ORIENT_CHANNELS = np.arange(0, N_H_CHANNELS, 2)  # (6,)
H_ORIENT_STEP = 2
FORECAST_MIN_FRAC = 0.25        # >= 1.5 × (1/6) chance
CHANCE = 1.0 / 6.0
SEEDS = [42, 43, 44]


def joint_hist_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.size == 0:
        return 0.0
    edges = np.arange(n_bins + 1) - 0.5
    pxy, _, _ = np.histogram2d(x, y, bins=(edges, edges))
    pxy = pxy / pxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    iy, ix = np.nonzero(pxy.T)
    for j, i in zip(iy, ix):
        p_ij = pxy[i, j]
        denom = px[i] * py[j]
        if denom > 0 and p_ij > 0:
            mi += p_ij * np.log2(p_ij / denom)
    return float(mi)


def theta_to_orient_idx(theta_rad: np.ndarray, orientations_deg=None) -> np.ndarray:
    """Map theta_rad to the 6-orient index used in richter/tang (30° grid)."""
    if orientations_deg is None:
        orientations_deg = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
    o_rad = np.deg2rad(orientations_deg)
    t = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    t_mod = np.mod(t, np.pi)
    d = np.abs(t_mod[:, None] - o_rad[None, :])
    d = np.minimum(d, np.pi - d)
    return np.argmin(d, axis=1).astype(np.int64)


def validate_richter_seed(seed: int) -> dict:
    path = IN_DIR / f"D1_richter_seed{seed}.npz"
    d = dict(np.load(path, allow_pickle=True))
    rates = np.asarray(d["h_preprobe_rate_hz"])              # (N, 12)
    ref_ch = np.asarray(d["ref_ch"]).astype(np.int64)        # leader, 12-ch idx
    # Expected trailer under Richter assay bias = leader + 30°
    # (orient step = 1) -> 12-channel step = 2.
    expected_ch = (ref_ch + H_ORIENT_STEP) % N_H_CHANNELS
    amax12 = rates.argmax(axis=1).astype(np.int64)

    n_active = int((rates.max(axis=1) > 0.5).sum())
    frac_forecast = float(np.mean(amax12 == expected_ch))
    frac_memory = float(np.mean(amax12 == ref_ch))

    # Auxiliary MI (reported for context, not the gate).
    leader_idx = ref_ch // H_ORIENT_STEP
    trailer_idx = np.asarray(d["exp_ch"]).astype(np.int64) // H_ORIENT_STEP
    rates_6 = rates[:, H_ORIENT_CHANNELS]
    amax6 = rates_6.argmax(axis=1).astype(np.int64)
    mi_forecast_bits = joint_hist_mi(trailer_idx, amax6, 6)
    mi_memory_bits = joint_hist_mi(leader_idx, amax6, 6)

    passed = frac_forecast >= FORECAST_MIN_FRAC
    return dict(
        seed=int(seed),
        n_trials=int(rates.shape[0]),
        n_active=n_active,
        P_argmax_eq_expected_trailer=frac_forecast,
        P_argmax_eq_leader=frac_memory,
        mi_forecast_bits=mi_forecast_bits,
        mi_memory_bits=mi_memory_bits,
        passed=passed,
    )


def main() -> int:
    print("=== validate_stage1_preprobe_forecast ===")
    print(f"  gate: P(argmax(H[pre_trailer_100ms]) == expected_trailer_ch) "
          f">= {FORECAST_MIN_FRAC}  (chance = 1/6 = {CHANCE:.3f})")
    print(f"  data: Sprint 5d D1 richter pre-probe rates (last 100 ms of leader)")
    print()
    rows = []
    for seed in SEEDS:
        try:
            r = validate_richter_seed(seed)
        except FileNotFoundError as exc:
            print(f"  seed={seed}: MISSING ({exc})")
            continue
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{status}] seed={seed}  n={r['n_trials']}  "
            f"n_active={r['n_active']}  "
            f"P(amax=expected_trailer)={r['P_argmax_eq_expected_trailer']:.3f}  "
            f"P(amax=leader)={r['P_argmax_eq_leader']:.3f}  "
            f"MI_fore={r['mi_forecast_bits']:.3f}b  "
            f"MI_mem={r['mi_memory_bits']:.3f}b"
        )
        rows.append(r)

    if not rows:
        print("\nERROR: no D1 seeds loaded — re-run Sprint 5d D1 first.")
        return 2

    any_pass = any(r["passed"] for r in rows)
    print()
    if any_pass:
        print("[B4b] AT LEAST ONE SEED PASSED pre-trailer forecast gate — "
              "unexpected under balanced-schedule training.")
    else:
        print("[B4b] ALL SEEDS FAIL pre-trailer forecast gate.")
        print("      Current Stage-1 checkpoint does NOT carry a forecast "
              "into the pre-trailer window.")
        print("      Argmax locks on the LEADER channel (memory), with "
              "P(amax=expected_trailer) well below chance 1/6.")
        print("      This is consistent with Bug 1 (no contingency to learn) "
              "and Bug 2 (training gate measures memory, not forecast).")

    summary = np.array(rows, dtype=object)
    np.savez(OUT_DIR / "B4b_preprobe_forecast_validation.npz", rows=summary)
    return 0 if not any_pass else 1


if __name__ == "__main__":
    sys.exit(main())
