"""Stage 1 gate -- stability and learnability ONLY.

Per plan sec 3, Stage-1:

- `check_h_bump_persistence` : a leader/rotation-evoked bump in H_R (or
  H_T) persists 200-500 ms after input offset before decaying to baseline.
- `check_h_transition_mi` : MI(leader_idx, H_R_argmax_at_+500ms) > 0 —
  H_R has learned to predict the trailer orientation from the leader
  (Richter incidental-pair co-occurrence statistic).
- `check_h_rotation_mi` : MI(expected_next_idx, H_T_argmax_at_+end_of_item)
  > 0 — H_T has learned the rotation statistic.
- `check_no_runaway` : no layer exceeds a ceiling rate (reuse Stage 0).

NO V1 / Kok / Richter / Tang phenomenon checked here (that is the role of
the held-out Stage-2 / Stage-3 assays). Stage 1 is purely about whether H
recurrence is stable and MI is distinguishable from zero.

Current-stage seed policy (pre-registration, research_log.md): seed=42
only. Multi-seed replication is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# -- bands / thresholds (lift to module consts so the driver can override) --

BUMP_PERSISTENCE_BAND_MS = (200.0, 500.0)
BUMP_RATE_FLOOR_HZ = 2.0          # "bump present" when peak-channel rate > floor
TRANSITION_MI_MIN_BITS = 0.05     # pragmatic ">0" threshold above finite-sample bias
ROTATION_MI_MIN_BITS = 0.05
RUNAWAY_CEILING_HZ = 80.0
# Sprint 5e Fix B — pre-trailer forecast probability gate. Chance = 1/6
# on 6 orientations ≈ 0.167; threshold 0.25 ≈ 1.5× chance is realistic
# for a network that carries a pre-trailer prior (see
# SPRINT_5D_POST_VERDICT_REVIEW.md Bug 2 + debugger's B4b seeds 42/43/44
# which hit 0.015/0.015/0.030 under post-trailer gate).
PREPROBE_FORECAST_MIN_PROB = 0.25


@dataclass
class CheckResult:
    """Uniform result container (mirrors Stage-0)."""
    name: str
    passed: bool
    value: Optional[float] = None
    band: Optional[Tuple[float, float]] = None
    detail: str = ""

    def summary(self) -> str:
        val = f"{self.value:.3f}" if self.value is not None else "n/a"
        band = f"[{self.band[0]:.3f}, {self.band[1]:.3f}]" if self.band else ""
        return (f"[{'PASS' if self.passed else 'FAIL'}] {self.name:30s} "
                f"value={val} band={band} {self.detail}")


def _in_band(x: float, band: Tuple[float, float]) -> bool:
    return band[0] <= x <= band[1]


# -- bump persistence -------------------------------------------------------

def compute_bump_persistence_ms(
    h_peak_rate_ms: np.ndarray,
    offset_idx: int,
    dt_ms: float = 1.0,
    floor_hz: float = BUMP_RATE_FLOOR_HZ,
) -> float:
    """Return how long the peak-channel rate stays >= `floor_hz` after offset.

    Parameters
    ----------
    h_peak_rate_ms : np.ndarray, shape (T,)
        Per-ms rate (Hz) of the peak H channel (e.g., smoothed spike count
        from a SpikeMonitor). The peak channel is picked at / before
        `offset_idx`.
    offset_idx : int
        Sample index at which the driving input turned off.
    dt_ms : float
        Sampling interval of `h_peak_rate_ms`.
    floor_hz : float
        Rate below which the bump is considered extinguished.

    Returns
    -------
    persistence_ms : float
        `0.0` if the peak never crossed the floor at offset (no bump);
        `len(h_peak_rate_ms) - offset_idx` * dt_ms if the bump never decayed
        during the recording window (upper-bounded); otherwise the elapsed
        time (ms) from offset until the first sample below `floor_hz`.
    """
    h = np.asarray(h_peak_rate_ms, dtype=np.float64)
    T = len(h)
    if offset_idx < 0 or offset_idx >= T:
        return float("nan")
    if h[offset_idx] < floor_hz:
        return 0.0
    tail = h[offset_idx:]
    below = np.where(tail < floor_hz)[0]
    if len(below) == 0:
        return float((T - offset_idx - 1) * dt_ms)
    return float(below[0] * dt_ms)


def check_h_bump_persistence(
    h_peak_rate_ms: np.ndarray,
    offset_idx: int,
    band: Tuple[float, float] = BUMP_PERSISTENCE_BAND_MS,
    floor_hz: float = BUMP_RATE_FLOOR_HZ,
    dt_ms: float = 1.0,
) -> CheckResult:
    """Bump persistence (ms after input offset) must lie in `band`."""
    p = compute_bump_persistence_ms(h_peak_rate_ms, offset_idx,
                                    dt_ms=dt_ms, floor_hz=floor_hz)
    passed = (not np.isnan(p)) and _in_band(p, band)
    return CheckResult("h_bump_persistence_ms", passed, p, band,
                       f"floor={floor_hz:.1f} Hz offset_idx={offset_idx}")


# -- mutual information -----------------------------------------------------

def _joint_hist_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Plug-in MI estimator in BITS for two discrete rvs.

    Uses the joint-histogram / sample-frequency estimator (biased toward
    positive values in finite samples; we compensate by requiring MI to
    exceed `TRANSITION_MI_MIN_BITS` rather than strict >0).
    """
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape:
        raise ValueError(f"MI shape mismatch: {x.shape} vs {y.shape}")
    if len(x) == 0:
        return 0.0
    edges = np.arange(n_bins + 1) - 0.5  # integer-safe: bin k is [k-.5, k+.5)
    pxy, _, _ = np.histogram2d(x, y, bins=(edges, edges))
    pxy = pxy / pxy.sum()
    px = pxy.sum(axis=1)    # (n_bins,)
    py = pxy.sum(axis=0)    # (n_bins,)
    # Sum only over cells with joint mass > 0 to avoid log(0).
    mi = 0.0
    iy, ix = np.nonzero(pxy.T)   # (k, i) pairs
    for j, i in zip(iy, ix):
        p_ij = pxy[i, j]
        denom = px[i] * py[j]
        if denom > 0.0 and p_ij > 0.0:
            mi += p_ij * np.log2(p_ij / denom)
    return float(mi)


def check_h_transition_mi(
    leader_idx_per_trial: np.ndarray,
    h_argmax_per_trial: np.ndarray,
    n_orient: int,
    min_bits: float = TRANSITION_MI_MIN_BITS,
    gate_window: str = "post_trailer",
    expected_trailer_idx: Optional[np.ndarray] = None,
    min_prob: float = PREPROBE_FORECAST_MIN_PROB,
) -> CheckResult:
    """H_R must carry information about the trailer forecast.

    Two gate modes (Sprint 5e Fix B).

    Parameters
    ----------
    leader_idx_per_trial : np.ndarray, shape (n_trials,), int
        Orientation index of the leader for each trial. Used only in
        ``post_trailer`` mode.
    h_argmax_per_trial : np.ndarray, shape (n_trials,), int
        argmax channel of H_R rate in the gate window. Index in
        ``[0, n_orient)``.
    n_orient : int
        Number of possible orientation categories.
    min_bits : float
        Finite-sample floor for "> 0" MI (post_trailer mode, default 0.05 bits).
    gate_window : {"post_trailer", "pre_trailer"}
        Which probe window the ``h_argmax_per_trial`` came from, and which
        metric to apply.

        - ``"post_trailer"`` (legacy, Sprint 5a/5c):
              metric = ``MI(leader_idx, h_argmax_at_+500ms_after_trailer)``;
              pass iff MI >= ``min_bits``. The probe window sits 500 ms past
              trailer offset, so a network that simply tracks the *current*
              input (no forecast) can pass this gate as long as the training
              schedule has L→T contingency. Kept for backward compatibility
              / ablation; the driver should pick this mode only when the
              post-trailer bump is the claim under test.
        - ``"pre_trailer"`` (new, default for the forecast claim):
              metric = ``P(h_argmax_in_last_100ms_of_leader == expected_trailer_idx)``;
              pass iff probability >= ``min_prob`` (default 0.25 ≈ 1.5× chance
              on 6 orientations). This is the Richter / Kok incidental-pair
              empirical signal: the network must express the forecast
              **before** the trailer arrives. Requires ``expected_trailer_idx``
              (derangement[leader_idx]) so the check knows what the expected
              forecast is per trial.
    expected_trailer_idx : np.ndarray, shape (n_trials,), int, optional
        Expected trailer orientation per trial (derived from the schedule's
        deranged permutation). Required when ``gate_window="pre_trailer"``.
    min_prob : float
        Probability floor for pre_trailer mode (default 0.25 = 1.5 × 1/6
        chance).

    Returns
    -------
    CheckResult
        ``name="h_transition_mi_bits"`` in ``post_trailer`` mode, and
        ``name="h_preprobe_forecast_prob"`` in ``pre_trailer`` mode, so
        Stage-1 reports distinguish the two metrics at a glance.
    """
    if gate_window == "post_trailer":
        mi = _joint_hist_mi(leader_idx_per_trial, h_argmax_per_trial, n_orient)
        return CheckResult("h_transition_mi_bits", mi >= min_bits, mi,
                           (min_bits, np.log2(n_orient)),
                           f"n_trials={len(leader_idx_per_trial)} "
                           f"window=post_trailer")
    if gate_window == "pre_trailer":
        if expected_trailer_idx is None:
            raise ValueError(
                "gate_window='pre_trailer' requires expected_trailer_idx "
                "(derangement[leader_idx]) — the pre-probe forecast gate "
                "needs a per-trial target."
            )
        h_arr = np.asarray(h_argmax_per_trial, dtype=np.int64)
        e_arr = np.asarray(expected_trailer_idx, dtype=np.int64)
        if h_arr.shape != e_arr.shape:
            raise ValueError(
                f"pre_trailer gate shape mismatch: "
                f"h_argmax {h_arr.shape} vs expected_trailer {e_arr.shape}"
            )
        n = int(h_arr.size)
        if n == 0:
            return CheckResult("h_preprobe_forecast_prob", False, 0.0,
                               (min_prob, 1.0),
                               "n_trials=0 window=pre_trailer")
        n_hit = int((h_arr == e_arr).sum())
        prob = float(n_hit) / float(n)
        return CheckResult("h_preprobe_forecast_prob", prob >= min_prob, prob,
                           (min_prob, 1.0),
                           f"n_trials={n} hits={n_hit} "
                           f"chance={1.0/float(n_orient):.3f} "
                           f"window=pre_trailer")
    raise ValueError(
        f"gate_window must be 'post_trailer' or 'pre_trailer', got {gate_window!r}"
    )


def check_h_rotation_mi(
    expected_next_idx: np.ndarray,
    h_argmax_per_item: np.ndarray,
    n_orient: int,
    min_bits: float = ROTATION_MI_MIN_BITS,
) -> CheckResult:
    """H_T must carry information about the expected next rotation orient."""
    mi = _joint_hist_mi(expected_next_idx, h_argmax_per_item, n_orient)
    return CheckResult("h_rotation_mi_bits", mi >= min_bits, mi,
                       (min_bits, np.log2(n_orient)),
                       f"n_items={len(expected_next_idx)}")


# -- runaway guard ----------------------------------------------------------

def check_no_runaway(rates_per_layer: Dict[str, float],
                     ceiling_hz: float = RUNAWAY_CEILING_HZ) -> CheckResult:
    overs = {k: v for k, v in rates_per_layer.items() if v > ceiling_hz}
    ok = len(overs) == 0
    detail = "OK" if ok else f"over: {overs}"
    return CheckResult(
        "no_runaway", ok,
        max(rates_per_layer.values()) if rates_per_layer else 0.0,
        (0.0, ceiling_hz), detail,
    )


# -- aggregator -------------------------------------------------------------

@dataclass
class Stage1Report:
    results: Dict[str, CheckResult]
    passed: bool

    def summary(self) -> str:
        lines = [r.summary() for r in self.results.values()]
        verdict = "ALL PASS" if self.passed else "FAILED"
        lines.append(f"--- Stage 1 gate: {verdict} ---")
        return "\n".join(lines)


def aggregate(checks: Dict[str, CheckResult]) -> Stage1Report:
    all_ok = all(r.passed for r in checks.values())
    return Stage1Report(results=checks, passed=all_ok)


# -- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    # 1) Persistence: bump stays at 10 Hz for 300 ms after offset, then decays.
    dt = 1.0
    T = 1000
    h = np.zeros(T)
    h[200:500] = 15.0      # driven bump
    h[500:800] = np.linspace(15.0, 1.0, 300)   # decay over 300 ms
    h[800:] = 0.5
    p = compute_bump_persistence_ms(h, offset_idx=500, dt_ms=dt, floor_hz=2.0)
    assert 200 <= p <= 320, f"persistence {p} ms"
    print(f"stage_1 smoke: persistence = {p:.1f} ms (expected 200-320)")

    # 2) Persistence FAIL (extinguishes in 100 ms).
    h_fast = np.zeros(T)
    h_fast[200:500] = 15.0
    h_fast[500:600] = np.linspace(15.0, 0.0, 100)
    p_fast = compute_bump_persistence_ms(h_fast, offset_idx=500)
    assert p_fast < 200, p_fast

    # 3) MI: perfect transition (leader == H_argmax) gives MI = H(X).
    rng = np.random.default_rng(42)
    leader = rng.integers(0, 6, size=300)
    h_perfect = leader.copy()
    mi_perfect = _joint_hist_mi(leader, h_perfect, 6)
    # With finite samples marginals aren't exactly uniform; MI == H(leader).
    marg = np.bincount(leader, minlength=6) / 300.0
    h_true = -np.sum(marg[marg > 0] * np.log2(marg[marg > 0]))
    assert abs(mi_perfect - h_true) < 1e-9, (mi_perfect, h_true)

    # 4) MI: independent -> ~0 (up to finite-sample bias).
    h_random = rng.integers(0, 6, size=300)
    mi_null = _joint_hist_mi(leader, h_random, 6)
    assert mi_null < 0.15, mi_null
    print(f"stage_1 smoke: MI perfect={mi_perfect:.3f}, null={mi_null:.3f}")

    # 4b) pre_trailer gate — perfect forecast (h_argmax == expected_trailer).
    # derangement f(L) = (L+1) % 6 matches Sprint 5e Fix A default.
    derangement = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    leader_pre = rng.integers(0, 6, size=300)
    expected_pre = derangement[leader_pre]
    h_forecast_ok = expected_pre.copy()
    check_pre_ok = check_h_transition_mi(
        leader_pre, h_forecast_ok, 6,
        gate_window="pre_trailer",
        expected_trailer_idx=expected_pre,
    )
    assert check_pre_ok.passed, check_pre_ok.summary()
    assert check_pre_ok.name == "h_preprobe_forecast_prob"
    assert abs(check_pre_ok.value - 1.0) < 1e-9
    # pre_trailer gate — argmax locked on leader (chance-level on expected trailer).
    check_pre_fail = check_h_transition_mi(
        leader_pre, leader_pre.copy(), 6,
        gate_window="pre_trailer",
        expected_trailer_idx=expected_pre,
    )
    assert not check_pre_fail.passed, check_pre_fail.summary()
    # pre_trailer gate requires expected_trailer_idx.
    try:
        check_h_transition_mi(
            leader_pre, h_forecast_ok, 6,
            gate_window="pre_trailer",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pre_trailer gate must require expected_trailer_idx")
    # Unknown gate_window must raise.
    try:
        check_h_transition_mi(
            leader_pre, h_forecast_ok, 6, gate_window="post_settle",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown gate_window must raise")
    print(f"stage_1 smoke: pre_trailer perfect={check_pre_ok.value:.3f} "
          f"fail={check_pre_fail.value:.3f}")

    # 5) Aggregation: all-pass case.
    res = {
        "h_bump_persistence_ms": check_h_bump_persistence(h, offset_idx=500),
        "h_transition_mi_bits": check_h_transition_mi(leader, h_perfect, 6),
        "h_rotation_mi_bits": check_h_rotation_mi(leader, h_perfect, 6),
        "no_runaway": check_no_runaway({"h_e": 12.0, "h_inh": 20.0}),
    }
    rep = aggregate(res)
    print(rep.summary())
    assert rep.passed

    # 6) Fail case.
    res_bad = dict(res)
    res_bad["h_bump_persistence_ms"] = check_h_bump_persistence(
        h_fast, offset_idx=500,
    )
    res_bad["h_transition_mi_bits"] = check_h_transition_mi(leader, h_random, 6)
    rep_bad = aggregate(res_bad)
    assert not rep_bad.passed
    print("stage_1 smoke: PASS")
