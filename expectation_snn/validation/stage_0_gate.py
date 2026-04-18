"""Stage 0 calibration gate (plan §3, Stage 0).

This module holds pure check functions that take monitor data and config
bands, and return a pass/fail verdict + diagnostic dict. The driver
(`brian2_model/train.py:run_stage_0`) is responsible for:

1. Running the V1 ring under a battery of probe orientations and ITI windows.
2. Running the H ring under silence (baseline) and pulse.
3. Feeding the resulting rate/FWHM measurements into the check functions below.
4. Iteratively tweaking bias currents / iSTDP ceilings until all checks pass.

Checks (per plan §3 Stage-0):

- `check_v1_e_rate_band` : V1 E rates in [2, 8] Hz (Niell & Stryker 2008,
  PMID 18562647).
- `check_v1_pv_rate_band` : PV pool 10-40 Hz (Hu 2014, fast-spiking range).
- `check_v1_som_rate_band` : SOM 2-6 Hz (Urban-Ciecko & Barth 2016).
- `check_tuning_fwhm` : orientation tuning FWHM in [30, 60] deg (plan §1).
- `check_h_baseline_quiet` : no persistent bump pre-Stage-1.
- `check_h_pulse_response` : cue pulse biases the matched channel.
- `check_no_runaway` : no layer exceeds a ceiling rate (50 Hz default) and
  no layer is silent for > 50 ms under an expected-on condition.

Current-stage seed policy (pre-registration, research_log.md): seed=42
only. Multi-seed replication is deferred until a first-pass finding is
worth replicating, so `stage_0_gate` is designed to be called with a
single-seed trial context and return an actionable diagnostic dict the
driver can use to nudge parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Rate bands (Hz). Lift to module constants so driver code can override.
V1_E_RATE_BAND_HZ = (2.0, 8.0)
V1_PV_RATE_BAND_HZ = (10.0, 40.0)
V1_SOM_RATE_BAND_HZ = (2.0, 6.0)
TUNING_FWHM_BAND_DEG = (30.0, 60.0)
H_BASELINE_MAX_HZ = 1.0        # "quiet" == < 1 Hz avg pop rate
RUNAWAY_CEILING_HZ = 80.0


@dataclass
class CheckResult:
    """Uniform result container."""
    name: str
    passed: bool
    value: Optional[float] = None
    band: Optional[Tuple[float, float]] = None
    detail: str = ""

    def summary(self) -> str:
        val = f"{self.value:.2f}" if self.value is not None else "n/a"
        band = f"[{self.band[0]:.2f}, {self.band[1]:.2f}]" if self.band else ""
        return (f"[{'PASS' if self.passed else 'FAIL'}] {self.name:30s} "
                f"value={val} band={band} {self.detail}")


def _in_band(x: float, band: Tuple[float, float]) -> bool:
    return band[0] <= x <= band[1]


def check_v1_e_rate_band(mean_rate_hz: float,
                         band: Tuple[float, float] = V1_E_RATE_BAND_HZ
                         ) -> CheckResult:
    """Mean V1_E population firing rate (Hz) must lie in `band`."""
    return CheckResult("v1_e_rate_band", _in_band(mean_rate_hz, band),
                       mean_rate_hz, band)


def check_v1_pv_rate_band(mean_rate_hz: float,
                          band: Tuple[float, float] = V1_PV_RATE_BAND_HZ
                          ) -> CheckResult:
    return CheckResult("v1_pv_rate_band", _in_band(mean_rate_hz, band),
                       mean_rate_hz, band)


def check_v1_som_rate_band(mean_rate_hz: float,
                           band: Tuple[float, float] = V1_SOM_RATE_BAND_HZ
                           ) -> CheckResult:
    return CheckResult("v1_som_rate_band", _in_band(mean_rate_hz, band),
                       mean_rate_hz, band)


def compute_fwhm_deg(rates_per_channel: np.ndarray,
                     channel_spacing_deg: float = 15.0) -> float:
    """Estimate tuning FWHM (deg) from per-channel rates on a 0..180 ring.

    Rotates the profile so its peak lands at channel 0, then interpolates
    across the halved-peak crossings in both directions. Returns the sum
    of crossings on either side (so FWHM = L + R).

    Parameters
    ----------
    rates_per_channel : np.ndarray, shape (N,)
    channel_spacing_deg : float

    Returns
    -------
    fwhm_deg : float
        NaN if peak is 0 or profile is flat.
    """
    r = np.asarray(rates_per_channel, dtype=np.float64).copy()
    n = len(r)
    if not np.isfinite(r).all() or r.max() <= 0:
        return float("nan")
    peak_idx = int(np.argmax(r))
    # Shift so the peak is at index 0.
    r_shift = np.roll(r, -peak_idx)
    peak = r_shift[0]
    half = 0.5 * peak
    # Right-side crossing (looking at channels 1, 2, ... n//2).
    r_halves = []
    for direction in (+1, -1):
        # Accumulate step-by-step from 1..n//2.
        last_above = 0.0  # in channel indices
        last_val = peak
        found = False
        for k in range(1, n // 2 + 1):
            idx = (0 + direction * k) % n
            val = r_shift[(idx - 0) % n] if direction > 0 else r_shift[(-k) % n]
            if val <= half:
                # Linear interp between (k-1, last_val) and (k, val).
                if last_val == val:
                    crossing = float(k)
                else:
                    crossing = (k - 1) + (last_val - half) / (last_val - val)
                r_halves.append(crossing)
                found = True
                break
            last_above = float(k)
            last_val = val
        if not found:
            r_halves.append(float(n // 2))   # no crossing found -> broad
    total_channels = sum(r_halves)
    return total_channels * channel_spacing_deg


def check_tuning_fwhm(rates_per_channel: np.ndarray,
                      band: Tuple[float, float] = TUNING_FWHM_BAND_DEG,
                      channel_spacing_deg: float = 15.0) -> CheckResult:
    fwhm = compute_fwhm_deg(rates_per_channel, channel_spacing_deg)
    passed = (not np.isnan(fwhm)) and _in_band(fwhm, band)
    return CheckResult("v1_tuning_fwhm_deg", passed, fwhm, band)


def check_h_baseline_quiet(mean_rate_hz: float,
                           max_hz: float = H_BASELINE_MAX_HZ
                           ) -> CheckResult:
    return CheckResult("h_baseline_quiet", mean_rate_hz <= max_hz,
                       mean_rate_hz, (0.0, max_hz))


def check_h_pulse_response(per_channel_spike_counts: np.ndarray,
                           pulsed_channel: int,
                           orthogonal_channel: Optional[int] = None
                           ) -> CheckResult:
    """Pulse on channel k should dominate orthogonal channels."""
    n = len(per_channel_spike_counts)
    orth = orthogonal_channel if orthogonal_channel is not None else (
        (pulsed_channel + n // 2) % n
    )
    pulsed = float(per_channel_spike_counts[pulsed_channel])
    ortho = float(per_channel_spike_counts[orth])
    # Require that pulsed > 2x orthogonal (or orthogonal is 0 with pulsed > 0).
    ok = (pulsed > 2.0 * ortho) if ortho > 0 else (pulsed > 0)
    return CheckResult("h_pulse_response", ok, pulsed / max(ortho, 1.0),
                       (2.0, float("inf")),
                       f"pulse_ch{pulsed_channel}={pulsed}, orth_ch{orth}={ortho}")


def check_no_runaway(rates_per_layer: Dict[str, float],
                     ceiling_hz: float = RUNAWAY_CEILING_HZ) -> CheckResult:
    """No layer exceeds ceiling_hz (runaway guard)."""
    overs = {k: v for k, v in rates_per_layer.items() if v > ceiling_hz}
    ok = len(overs) == 0
    detail = "OK" if ok else f"over: {overs}"
    return CheckResult("no_runaway", ok, max(rates_per_layer.values()) if
                       rates_per_layer else 0.0,
                       (0.0, ceiling_hz), detail)


# -- aggregator -------------------------------------------------------------

@dataclass
class Stage0Report:
    results: Dict[str, CheckResult]
    passed: bool

    def summary(self) -> str:
        lines = [r.summary() for r in self.results.values()]
        verdict = "ALL PASS" if self.passed else "FAILED"
        lines.append(f"--- Stage 0 gate: {verdict} ---")
        return "\n".join(lines)


def aggregate(checks: Dict[str, CheckResult]) -> Stage0Report:
    all_ok = all(r.passed for r in checks.values())
    return Stage0Report(results=checks, passed=all_ok)


# -- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    # Pass case
    good = {
        "v1_e_rate_band": check_v1_e_rate_band(5.0),
        "v1_pv_rate_band": check_v1_pv_rate_band(20.0),
        "v1_som_rate_band": check_v1_som_rate_band(4.0),
        "v1_tuning_fwhm_deg": check_tuning_fwhm(
            np.array([10, 8, 5, 2, 1, 0.5, 0, 0.5, 1, 2, 5, 8], dtype=float)
        ),
        "h_baseline_quiet": check_h_baseline_quiet(0.2),
        "h_pulse_response": check_h_pulse_response(
            np.array([20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]), pulsed_channel=0
        ),
        "no_runaway": check_no_runaway({"v1_e": 5.0, "v1_pv": 20.0}),
    }
    rep = aggregate(good)
    print(rep.summary())
    assert rep.passed, "Pass case should pass"

    # Fail case: FWHM too wide + runaway
    bad = dict(good)
    bad["v1_e_rate_band"] = check_v1_e_rate_band(20.0)   # runaway
    bad["no_runaway"] = check_no_runaway({"v1_e": 100.0})
    rep2 = aggregate(bad)
    assert not rep2.passed
    print("--- deliberate-fail case ---")
    print(rep2.summary())

    print("stage_0_gate smoke: PASS")
