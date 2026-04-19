"""Component-level validation for the V1 ring.

Scope (Sprint-3 backfill, 2026-04-19): TUNING-PRESERVATION ONLY.

The PV rheobase Poisson-noise smoothing claim (Sprint-3 commit
`ec2a2ac fix(v1_ring): smooth PV rheobase with Poisson background`)
is NOT asserted in this file. The right metric + threshold for the
smoothing property is under active investigation by the debugger
(see docs/phase_gate_evidence.md under "Sprint 3 backfill -- per-
component functional validation"). Once the debugger proposes a
tested metric, this file will be amended with:

  TODO(debugger/Sprint-3-backfill):
    - PV rheobase continuity: sweep bias in
      [bias_cal - 20, bias_cal + 20] pA at 2 pA steps; assert no
      jump > 15 Hz between adjacent biases (anti-cliff).
    - CV-ISI > 0.5 with Poisson background enabled
      (physiological irregularity under noise).

For now we validate only the tuning-preservation claim: drifting a
grating into the full V1 ring (E + PV + SOM, within-channel and
cross-channel wiring active) places the V1 E peak on the driven
channel and the FWHM falls in the Tang-observed 30-60 deg band.

Assays
------

1. **Peak channel**: per-channel V1 E population rate peaks at the
   channel closest to the driven theta.

2. **FWHM in band**: full-width at half-maximum of the per-channel
   V1 E tuning curve lies in [30, 60] deg (Tang 2023 / plan sec 3
   Stage-0 gate band).

3. **Orthogonal suppression**: V1 E rate at orthogonal channel
   (theta + 90 deg) is < half of peak-channel rate (basic tuning
   sharpness).

Run:
    python -m expectation_snn.validation.validate_v1_ring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from brian2 import (
    Network,
    SpikeMonitor,
    defaultclock,
    ms,
    prefs,
    seed as b2_seed,
)

from ..brian2_model.v1_ring import (
    N_CHANNELS,
    N_E_PER_CHANNEL,
    build_v1_ring,
    set_stimulus,
)


FWHM_BAND_DEG: Tuple[float, float] = (30.0, 60.0)
PROBE_DURATION_MS = 500.0
PROBE_SEED = 42


# --- helpers ---------------------------------------------------------------

def _fwhm_deg_ring(per_channel_rate: np.ndarray) -> float:
    """Full-width at half-maximum (degrees) of a ring tuning curve.

    Rolls so the peak sits at the array center, linearly interpolates
    the half-crossings on either side, and returns their separation in
    degrees (channel spacing = 180 / N_CHANNELS = 15 deg).
    """
    rate = np.asarray(per_channel_rate, dtype=np.float64)
    n = len(rate)
    peak_idx = int(np.argmax(rate))
    centered = np.roll(rate, n // 2 - peak_idx)
    peak = float(centered.max())
    if peak <= 0.0:
        return float("nan")
    half = 0.5 * peak
    # Find the two half-crossings left/right of the center peak.
    peak_c = n // 2
    # Left crossing
    left = peak_c
    for k in range(peak_c, 0, -1):
        if centered[k] >= half and centered[k - 1] < half:
            # linear interp
            frac = (centered[k] - half) / (centered[k] - centered[k - 1])
            left = k - frac
            break
        if k - 1 == 0 and centered[k - 1] >= half:
            left = 0.0
    # Right crossing
    right = peak_c
    for k in range(peak_c, n - 1):
        if centered[k] >= half and centered[k + 1] < half:
            frac = (centered[k] - half) / (centered[k] - centered[k + 1])
            right = k + frac
            break
        if k + 1 == n - 1 and centered[k + 1] >= half:
            right = float(n - 1)
    channel_spacing_deg = 180.0 / N_CHANNELS
    return float((right - left) * channel_spacing_deg)


# --- report ----------------------------------------------------------------

@dataclass
class V1RingValidationReport:
    per_channel_rate_hz: np.ndarray
    peak_channel: int
    expected_peak_channel: int
    peak_rate_hz: float
    orthogonal_rate_hz: float
    fwhm_deg: float
    fwhm_band: Tuple[float, float]
    passed_peak: bool
    passed_fwhm: bool
    passed_orth: bool

    @property
    def passed(self) -> bool:
        return self.passed_peak and self.passed_fwhm and self.passed_orth

    def summary(self) -> str:
        lines = ["V1 ring tuning-preservation validation:"]
        lines.append("  per-channel V1 E rates (Hz):")
        for c, r in enumerate(self.per_channel_rate_hz):
            mark = "  <- peak" if c == self.peak_channel else ""
            mark += "  <- driven" if c == self.expected_peak_channel else ""
            lines.append(f"    ch{c:02d}  {r:6.2f} Hz{mark}")
        lines.append("  -----------------------------------")
        lines.append(
            f"  peak channel = {self.peak_channel} "
            f"(expected {self.expected_peak_channel})  "
            f"{'PASS' if self.passed_peak else 'FAIL'}"
        )
        lines.append(
            f"  FWHM = {self.fwhm_deg:.2f} deg  "
            f"band={self.fwhm_band}  "
            f"{'PASS' if self.passed_fwhm else 'FAIL'}"
        )
        lines.append(
            f"  orth rate = {self.orthogonal_rate_hz:.2f} Hz "
            f"(< 0.5 x peak = {0.5 * self.peak_rate_hz:.2f})  "
            f"{'PASS' if self.passed_orth else 'FAIL'}"
        )
        lines.append("  -----------------------------------")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def run_v1_ring_validation(verbose: bool = True) -> V1RingValidationReport:
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(PROBE_SEED); np.random.seed(PROBE_SEED)

    ring = build_v1_ring()
    set_stimulus(ring, theta_rad=0.0, contrast=1.0)
    e_mon = SpikeMonitor(ring.e, name="v1e_mon_validate")

    net = Network(*ring.groups, e_mon)
    net.run(PROBE_DURATION_MS * ms)

    e_idx = np.asarray(e_mon.i[:])
    per_ch = np.bincount(ring.e_channel[e_idx], minlength=N_CHANNELS) / (
        N_E_PER_CHANNEL * (PROBE_DURATION_MS / 1000.0)
    )

    peak_c = int(np.argmax(per_ch))
    expected_peak = 0          # theta_rad=0 -> channel 0
    orth_c = N_CHANNELS // 2   # 90 deg = channel 6
    fwhm_deg = _fwhm_deg_ring(per_ch)

    passed_peak = peak_c == expected_peak
    passed_fwhm = FWHM_BAND_DEG[0] <= fwhm_deg <= FWHM_BAND_DEG[1]
    passed_orth = float(per_ch[orth_c]) < 0.5 * float(per_ch[peak_c])

    rep = V1RingValidationReport(
        per_channel_rate_hz=per_ch,
        peak_channel=peak_c,
        expected_peak_channel=expected_peak,
        peak_rate_hz=float(per_ch[peak_c]),
        orthogonal_rate_hz=float(per_ch[orth_c]),
        fwhm_deg=fwhm_deg,
        fwhm_band=FWHM_BAND_DEG,
        passed_peak=passed_peak,
        passed_fwhm=passed_fwhm,
        passed_orth=passed_orth,
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_v1_ring_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
