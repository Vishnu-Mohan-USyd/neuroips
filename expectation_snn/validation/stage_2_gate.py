"""Stage 2 gate -- cue learning on H_R (plan sec 3).

Checks (per plan + Lead dispatch for Sprint 4):

1. **Cue-alone orientation selectivity d >= 0.2, bootstrap 95% CI > 0.**
   On held-out cue-alone probe trials, measure the difference between the
   matched-channel H_R rate and the unmatched-channel H_R rate, normalised
   by the pooled standard deviation. Across trials, d must clear 0.2, and
   the bootstrap 95% CI must exclude zero.

2. **Cue-alone evokes H bump in >= 80 pct of valid-cue probe trials.**
   A probe "evokes a bump" if the matched-channel H rate in the delay
   window exceeds `BUMP_RATE_FLOOR_HZ` AND the matched channel is the
   population peak channel. Pooled over both cues' probes.

3. **H_R recurrent weights unchanged pre/post training.**
   Plasticity on `ee` is frozen during Stage 2; this check just verifies
   freezing actually worked. max(|w_after - w_before|) / max(|w_before|)
   must be < 0.01.

4. **No runaway.** Mean H_E + H_inh rates during training stay below the
   ceiling (same as Stage-0 / Stage-1, 80 Hz).

Seed policy: pre-registered seed=42 first-pass. Multi-seed replication
({7, 123, 2024, 11} + held-out {99, 314}) deferred until paradigm-level
findings warrant it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# -- bands / thresholds -----------------------------------------------------

SELECTIVITY_D_MIN = 0.2                  # Cohen's d floor (per plan)
BUMP_RATE_FLOOR_HZ = 5.0                 # matched-channel rate floor (Hz)
BUMP_TRIAL_FRAC_MIN = 0.80               # >= 80 pct of probes evoke bump
HR_WEIGHT_DRIFT_MAX = 0.01               # |Delta w|/w_max floor (plasticity
                                         # frozen -> should be ~0)
RUNAWAY_CEILING_HZ = 80.0
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 123


# -- result container -------------------------------------------------------

@dataclass
class CheckResult:
    """Uniform result container (mirrors Stage-0 / Stage-1)."""
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


# -- selectivity ------------------------------------------------------------

def cohens_d(matched_rates: np.ndarray, unmatched_rates: np.ndarray) -> float:
    """Cohen's d between matched- and unmatched-channel rate samples.

    Pooled-sd denominator. Returns 0 if pooled sd is zero (both arrays
    constant).
    """
    a = np.asarray(matched_rates, dtype=np.float64)
    b = np.asarray(unmatched_rates, dtype=np.float64)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    va = float(np.var(a, ddof=1)) if na > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if nb > 1 else 0.0
    s = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if s < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def bootstrap_d_ci(matched: np.ndarray, unmatched: np.ndarray,
                   B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
                   ) -> Tuple[float, Tuple[float, float]]:
    """Return (d, (d_lo95, d_hi95)) via paired resampling when lengths match.

    When matched and unmatched rates come from the same set of probes (one
    per probe), we resample probe-indices jointly to preserve the pairing.
    Otherwise we bootstrap each group independently.
    """
    a = np.asarray(matched, dtype=np.float64)
    b = np.asarray(unmatched, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d = cohens_d(a, b)
    n = len(a)
    if n == 0 or n != len(b):
        return d, (float("nan"), float("nan"))
    ds = np.empty(B, dtype=np.float64)
    for k in range(B):
        idx = rng.integers(0, n, size=n)
        ds[k] = cohens_d(a[idx], b[idx])
    lo, hi = np.quantile(ds, [0.025, 0.975])
    return d, (float(lo), float(hi))


def check_cue_selectivity(matched_rates: np.ndarray,
                          unmatched_rates: np.ndarray,
                          d_min: float = SELECTIVITY_D_MIN,
                          ) -> CheckResult:
    """Cue-alone orientation selectivity d >= d_min with bootstrap CI > 0."""
    d, (lo, hi) = bootstrap_d_ci(matched_rates, unmatched_rates)
    passed = (d >= d_min) and (lo > 0.0)
    return CheckResult(
        "cue_selectivity_d", passed, d,
        (d_min, float("inf")),
        f"bootstrap 95%% CI=[{lo:.3f}, {hi:.3f}]  "
        f"n_probe={len(matched_rates)}",
    )


# -- bump-evocation fraction ------------------------------------------------

def check_bump_fraction(matched_rates: np.ndarray,
                        peak_channels: np.ndarray,
                        matched_channels: np.ndarray,
                        rate_floor_hz: float = BUMP_RATE_FLOOR_HZ,
                        frac_min: float = BUMP_TRIAL_FRAC_MIN,
                        ) -> CheckResult:
    """Fraction of probes that evoke a matched-channel H bump >= frac_min.

    Each probe k evokes a bump iff
        matched_rates[k] > rate_floor_hz  AND  peak_channels[k] == matched_channels[k].
    """
    a = np.asarray(matched_rates, dtype=np.float64)
    peak = np.asarray(peak_channels, dtype=np.int64)
    m = np.asarray(matched_channels, dtype=np.int64)
    n = len(a)
    if n == 0:
        return CheckResult("bump_evocation_frac", False, 0.0, (frac_min, 1.0),
                           "n_probe=0")
    evoked = (a > rate_floor_hz) & (peak == m)
    frac = float(evoked.mean())
    passed = frac >= frac_min
    return CheckResult(
        "bump_evocation_frac", passed, frac, (frac_min, 1.0),
        f"n_evoked={int(evoked.sum())}/{n}  "
        f"floor={rate_floor_hz:.1f} Hz",
    )


# -- H_R recurrent weights unchanged ----------------------------------------

def check_hr_weights_unchanged(w_before: np.ndarray, w_after: np.ndarray,
                               drift_max: float = HR_WEIGHT_DRIFT_MAX,
                               ) -> CheckResult:
    """Max absolute drift / max(|w_before|) must be < drift_max.

    With `ee` plasticity frozen (A_plus_eff = A_minus_eff = 0), the
    weights should literally not change; this check just verifies
    freezing actually worked.
    """
    before = np.asarray(w_before, dtype=np.float64)
    after = np.asarray(w_after, dtype=np.float64)
    if before.shape != after.shape:
        return CheckResult(
            "hr_weights_unchanged", False, float("nan"),
            (0.0, drift_max),
            f"shape mismatch {before.shape} vs {after.shape}",
        )
    denom = max(float(np.max(np.abs(before))), 1e-12)
    drift = float(np.max(np.abs(after - before))) / denom
    passed = drift <= drift_max
    return CheckResult(
        "hr_weights_unchanged", passed, drift, (0.0, drift_max),
        f"max|dw|={drift * denom:.3e}  denom_wmax={denom:.3e}",
    )


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
class Stage2Report:
    results: Dict[str, CheckResult]
    passed: bool

    def summary(self) -> str:
        lines = [r.summary() for r in self.results.values()]
        verdict = "ALL PASS" if self.passed else "FAILED"
        lines.append(f"--- Stage 2 gate: {verdict} ---")
        return "\n".join(lines)


def aggregate(checks: Dict[str, CheckResult]) -> Stage2Report:
    all_ok = all(r.passed for r in checks.values())
    return Stage2Report(results=checks, passed=all_ok)


# -- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) Selectivity: matched 10 +- 1 Hz, unmatched 2 +- 1 Hz over 20 probes.
    matched = rng.normal(10.0, 1.0, size=20)
    unmatched = rng.normal(2.0, 1.0, size=20)
    r_sel = check_cue_selectivity(matched, unmatched)
    assert r_sel.passed and r_sel.value > 5.0, r_sel.summary()
    print(r_sel.summary())

    # 2) Fraction: 19/20 probes hit matched channel at >5 Hz, 1 is unmatched.
    n = 20
    matched_rates = np.full(n, 12.0)
    matched_rates[0] = 2.0  # one weak probe
    peak = np.array([3] * 18 + [9] + [3])   # 1 unmatched-peak probe
    mch = np.full(n, 3)
    r_frac = check_bump_fraction(matched_rates, peak, mch)
    # 18/20 = 0.90 >= 0.80 -> pass
    assert r_frac.passed, r_frac.summary()
    print(r_frac.summary())

    # 3) Weight stability: identical -> drift = 0 -> pass.
    w = rng.uniform(0, 1.5, size=1000)
    r_w = check_hr_weights_unchanged(w, w)
    assert r_w.passed and r_w.value == 0.0, r_w.summary()
    print(r_w.summary())

    # 4) No runaway.
    r_run = check_no_runaway({"h_e": 25.0, "h_inh": 18.0, "v1_e": 3.5})
    assert r_run.passed, r_run.summary()
    print(r_run.summary())

    rep = aggregate({
        "cue_selectivity_d": r_sel,
        "bump_evocation_frac": r_frac,
        "hr_weights_unchanged": r_w,
        "no_runaway": r_run,
    })
    print(rep.summary())
    assert rep.passed
    print("stage_2_gate smoke: PASS")
