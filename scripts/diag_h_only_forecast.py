"""Sprint 5e-Diag B5: H-only forecast unit test.

Question (critical): given a training schedule that DOES carry
statistical contingency (biased deranged permutation, reviewer's B1 fix)
and a pre-trailer measurement window (B2 fix), can the H_R ring ALONE —
no V1, no V1→H, no H→V1, no assays — learn to point its pre-probe
argmax at the expected trailer channel?

Protocol
--------
- Build H_R via `build_h_r` (Sprint-3 Wang-style bump attractor).
- Monkey-patch `richter_crossover_training_schedule` to emit a biased
  deranged-permutation schedule: f(L)=(L+1) mod 6, P(L→f(L))=0.80, 0.04
  for each of the other 5 trailers. 360 trials total.
- Attach a SpikeMonitor, pre-settle 10 s, then run the biased schedule
  with E↔E STDP plastic (same cfg as production Stage-1).
- For each trial in the last half, compute per-channel H_R rate over the
  last 100 ms of the leader (pre-trailer window), identical to the
  reviewer's pre-trailer forecast gate.
- Report:
    - P(argmax == leader)     (memory)
    - P(argmax == f(L) = expected trailer)     (forecast)
    - P(argmax == actual trailer)              (ground truth incl 20% off-bias)
    - MI(argmax, leader)  and MI(argmax, expected trailer)
- Also compute POST-trailer (+500 ms, matching current Stage-1 gate) so
  that pre- vs post-trailer comparison is available in the same run.

Tang direction
--------------
H_R / H_T have only 12 orientation channels (`h_ring.py::N_CHANNELS=12`,
no direction variable — `grep direction brian2_model/h_ring.py` returns
0 matches). Tang forecast requires a direction state (Bug 3). An H-only
Tang forecast test is therefore structurally impossible with the current
architecture: reported as "direction state absent — Bug 3 blocks Tang
forecast test even in isolation."

Verdict rules
-------------
If Richter biased-schedule pre-probe P(amax == f(L)) > 0.25 on seed 42:
    → architecture is OK; schedule + gate were the bugs (Bug 1 + Bug 2).
If P(amax == f(L)) stays ≤ chance (1/6) despite biased training:
    → architecture is the deeper bug; context/prediction split is needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from brian2 import (
    Hz,
    Network,
    NetworkOperation,
    SpikeMonitor,
    defaultclock,
    ms,
    prefs,
    seed as b2_seed,
)

import expectation_snn.brian2_model.stimulus as _stim_mod
import expectation_snn.brian2_model.train as _train_mod

from expectation_snn.brian2_model.stimulus import (
    RICHTER_LEADER_MS,
    RICHTER_TRAILER_MS,
    TrialItem,
    TrialPlan,
)
from expectation_snn.brian2_model.h_ring import build_h_r, N_CHANNELS as H_N_CHANNELS
from expectation_snn.brian2_model.train import (
    H_ORIENT_CHANNELS,
    _drive_h_broad_noise,
    _drive_h_cue_gaussian,
    _make_postsyn_normalizer,
    _per_channel_rate_in_window,
    _stage1_h_cfg,
    silence_cue,
)

OUT_DIR = _ROOT / "data" / "diag_sprint5e"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_TRIALS = 360
LEADER_MS = RICHTER_LEADER_MS
TRAILER_MS = RICHTER_TRAILER_MS
ITI_MS = 1500.0                 # shorter ITI for budget; keeps bump decay band
PRESETTLE_MS = 10_000.0
PRESETTLE_NOISE_HZ = 40.0
CUE_PEAK_HZ = 300.0
CUE_SIGMA_DEG = 15.0
NORMALIZE_DT_MS = 200.0

N_ORIENTS = 6
BIAS = 0.80                     # reviewer-specified contingency strength
H_ORIENT_STEP = 2               # 6-orient idx step 1 ↔ 12-ch step 2
ORIENTATIONS_RAD = np.deg2rad(np.array([0, 30, 60, 90, 120, 150]))


def biased_richter_schedule(
    rng: np.random.Generator, n_trials: int = N_TRIALS,
    leader_ms: float = LEADER_MS, trailer_ms: float = TRAILER_MS,
    iti_ms: float = ITI_MS, contrast: float = 1.0,
    orientations_deg=None,
) -> TrialPlan:
    """Biased deranged-permutation schedule f(L)=(L+1)%6, P(L→f(L))=BIAS."""
    pairs = np.empty((n_trials, 2), dtype=np.int64)
    other_p = (1.0 - BIAS) / (N_ORIENTS - 1)
    for k in range(n_trials):
        L = int(rng.integers(0, N_ORIENTS))
        expected = (L + 1) % N_ORIENTS
        probs = np.full(N_ORIENTS, other_p)
        probs[expected] = BIAS
        T = int(rng.choice(N_ORIENTS, p=probs))
        pairs[k] = (L, T)
    thetas_rad = ORIENTATIONS_RAD
    items = []
    for k, (li, ti) in enumerate(pairs):
        items.append(TrialItem(
            theta_rad=float(thetas_rad[li]),
            contrast=contrast, duration_ms=leader_ms,
            kind="leader",
            meta={"trial": k, "leader_idx": int(li), "trailer_idx": int(ti)},
        ))
        items.append(TrialItem(
            theta_rad=float(thetas_rad[ti]),
            contrast=contrast, duration_ms=trailer_ms,
            kind="trailer",
            meta={"trial": k, "leader_idx": int(li), "trailer_idx": int(ti)},
        ))
        if iti_ms > 0:
            items.append(TrialItem(
                theta_rad=None, contrast=0.0, duration_ms=iti_ms,
                kind="iti", meta={"trial": k},
            ))
    return TrialPlan(
        items=items,
        meta={"paradigm": "richter_biased",
              "n_trials": n_trials, "pairs": pairs,
              "orientations_deg": tuple(np.rad2deg(thetas_rad)),
              "bias": BIAS},
    )


def run_h_only_forecast(seed: int = SEED) -> dict:
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    plan = biased_richter_schedule(rng, n_trials=N_TRIALS)
    pairs = plan.meta["pairs"]
    cfg = _stage1_h_cfg(None)
    ring = build_h_r(config=cfg)
    silence_cue(ring)

    e_mon = SpikeMonitor(ring.e, name="b5_e")
    ee_norm = _make_postsyn_normalizer(
        ring.ee, target_sum=cfg.target_postsyn_sum,
        dt_ms=NORMALIZE_DT_MS, name="b5_norm",
    )
    net = Network(*ring.groups, e_mon, ee_norm)

    if PRESETTLE_MS > 0:
        _drive_h_broad_noise(ring, mean_rate_hz=PRESETTLE_NOISE_HZ)
        net.run(PRESETTLE_MS * ms)
        silence_cue(ring)

    schedule_start_abs_ms = float(net.t / ms)
    t0 = time.time()
    for item in plan.items:
        if item.kind == "iti" or item.theta_rad is None:
            silence_cue(ring)
        else:
            _drive_h_cue_gaussian(
                ring, item.theta_rad,
                peak_rate_hz=CUE_PEAK_HZ, sigma_deg=CUE_SIGMA_DEG,
            )
        net.run(item.duration_ms * ms)
    sim_wall_s = time.time() - t0

    spike_i = np.asarray(e_mon.i[:], dtype=np.int64)
    spike_t = np.asarray(e_mon.t / ms, dtype=np.float64)
    e_ch = ring.e_channel
    n_e_per = len(ring.e) // H_N_CHANNELS

    trial_ms = LEADER_MS + TRAILER_MS + ITI_MS
    trial_start = schedule_start_abs_ms + np.arange(N_TRIALS) * trial_ms
    leader_end = trial_start + LEADER_MS
    trailer_end = leader_end + TRAILER_MS

    # Pre-trailer window: last 100 ms of leader
    pre_start = leader_end - 100.0
    pre_end = leader_end
    # Post-trailer window: +500 ms ± 50 ms (matches current Stage-1 gate)
    post_mid = trailer_end + 500.0
    post_start = post_mid - 50.0
    post_end = post_mid + 50.0

    start_k = N_TRIALS // 2  # use last half (plasticity settled)
    n_use = N_TRIALS - start_k

    pre_amax12 = np.empty(n_use, dtype=np.int64)
    post_amax12 = np.empty(n_use, dtype=np.int64)
    leader_idx6 = np.empty(n_use, dtype=np.int64)
    trailer_idx6 = np.empty(n_use, dtype=np.int64)
    expected_idx6 = np.empty(n_use, dtype=np.int64)
    for kk, k in enumerate(range(start_k, N_TRIALS)):
        pre_rates = _per_channel_rate_in_window(
            spike_i, spike_t, e_ch, pre_start[k], pre_end[k],
            H_N_CHANNELS, n_e_per,
        )
        post_rates = _per_channel_rate_in_window(
            spike_i, spike_t, e_ch, post_start[k], post_end[k],
            H_N_CHANNELS, n_e_per,
        )
        pre_amax12[kk] = int(pre_rates.argmax())
        post_amax12[kk] = int(post_rates.argmax())
        li, ti = int(pairs[k, 0]), int(pairs[k, 1])
        leader_idx6[kk] = li
        trailer_idx6[kk] = ti
        expected_idx6[kk] = (li + 1) % N_ORIENTS

    leader_ch12 = leader_idx6 * H_ORIENT_STEP
    trailer_ch12 = trailer_idx6 * H_ORIENT_STEP
    expected_ch12 = expected_idx6 * H_ORIENT_STEP

    def frac(a, b):
        return float(np.mean(a == b))

    results = dict(
        seed=seed,
        n_use=int(n_use),
        sim_wall_s=sim_wall_s,
        # pre-trailer (forecast window)
        pre_P_amax_eq_leader=frac(pre_amax12, leader_ch12),
        pre_P_amax_eq_expected_trailer=frac(pre_amax12, expected_ch12),
        pre_P_amax_eq_actual_trailer=frac(pre_amax12, trailer_ch12),
        # post-trailer (current Stage-1 gate window)
        post_P_amax_eq_leader=frac(post_amax12, leader_ch12),
        post_P_amax_eq_expected_trailer=frac(post_amax12, expected_ch12),
        post_P_amax_eq_actual_trailer=frac(post_amax12, trailer_ch12),
        pre_amax12=pre_amax12,
        post_amax12=post_amax12,
        leader_idx6=leader_idx6,
        trailer_idx6=trailer_idx6,
        expected_idx6=expected_idx6,
        pairs=pairs,
    )
    return results


def main() -> int:
    print("=== diag_h_only_forecast  (B5) ===")
    print(f"seed={SEED}  n_trials={N_TRIALS}  bias={BIAS}  "
          f"ITI={ITI_MS} ms  presettle={PRESETTLE_MS} ms")
    print("Tang: SKIPPED — h_ring.N_CHANNELS=12 has no direction state; "
          "Bug 3 blocks Tang forecast test even in isolation.")
    print()
    r = run_h_only_forecast(SEED)
    print(f"\n[B5] simulated in {r['sim_wall_s']:.1f}s  n_use={r['n_use']}  "
          f"chance P=1/6={1/6:.3f}")
    print(f"\nPre-trailer (last 100 ms of leader) — FORECAST window:")
    print(f"  P(amax == leader)              = "
          f"{r['pre_P_amax_eq_leader']:.3f}   (memory)")
    print(f"  P(amax == expected_trailer=f(L)) = "
          f"{r['pre_P_amax_eq_expected_trailer']:.3f}   (forecast)")
    print(f"  P(amax == actual_trailer)      = "
          f"{r['pre_P_amax_eq_actual_trailer']:.3f}")
    print(f"\nPost-trailer (+500 ms) — current gate window:")
    print(f"  P(amax == leader)              = {r['post_P_amax_eq_leader']:.3f}")
    print(f"  P(amax == expected_trailer)    = "
          f"{r['post_P_amax_eq_expected_trailer']:.3f}")
    print(f"  P(amax == actual_trailer)      = "
          f"{r['post_P_amax_eq_actual_trailer']:.3f}")

    pre_fc = r["pre_P_amax_eq_expected_trailer"]
    print()
    if pre_fc > 0.25:
        print(f"[B5] VERDICT: PRE-PROBE FORECAST PRESENT "
              f"(P={pre_fc:.3f} > 0.25). Architecture OK; Bug 1 + Bug 2 are "
              f"the real problem.")
    elif pre_fc > 1 / 6 * 1.1:
        print(f"[B5] VERDICT: MARGINAL (P={pre_fc:.3f}). Biased schedule "
              f"pushes forecast slightly above chance but not robustly.")
    else:
        print(f"[B5] VERDICT: NO FORECAST (P={pre_fc:.3f} <= chance 1/6). "
              f"Even with biased schedule + correct pre-trailer measurement, "
              f"H_R argmax doesn't predict f(L). Architecture (single-ring, "
              f"no context/prediction split) is also a bug.")

    np.savez(
        OUT_DIR / "B5_h_only_forecast_richter.npz",
        **{k: v for k, v in r.items() if not isinstance(v, (list, dict))},
    )
    print(f"\n[B5] saved {OUT_DIR}/B5_h_only_forecast_richter.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
