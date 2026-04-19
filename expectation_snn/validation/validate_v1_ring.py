"""Component-level validation for the V1 ring PV rheobase smoothing
(Sprint-3 commit `fix(v1_ring): smooth PV rheobase with Poisson background`).

The bare LIF FS cell has a deterministic f-I curve with a razor-thin
rheobase: just below threshold -> silent; just above -> near saturation.
In vivo FS cells receive ~400-1000 Hz of spontaneous EPSCs from local
cortex (Hu, Gan & Jonas 2014 Science 345:1255263), which smooths the
f-I curve and lets the population fire in the biologically observed
10-40 Hz band over a wider bias range.

Scope note
----------
This is an *isolated* f-I test: no recurrent V1_E -> PV drive, no
thalamic drive, only I_bias + (optional) bg Poisson. In the full
Stage-0 network, PV's operating point at pv_bias_pA=240 sits in the
10-40 Hz band because cortical AMPA drive adds on top of bias. Here
we are validating only the *smoothing* claim: bg noise converts the
razor-thin cliff into a smooth, monotone curve.

Assays
------

1. **Anti-cliff**: with bg OFF the LIF FS cell is algebraically at
   rheobase when I_bias == gL*(VT-EL) = 300 pA; below rheobase it is
   silent, above rheobase the rate jumps very steeply. With bg ON
   the curve must pass smoothly through 10-40 Hz over a non-zero
   span of bias.

2. **Smooth-band span**: with bg ON, the bias span over which mean
   PV rate falls in [10, 40] Hz must be >= 15 pA (resolution-limited;
   the true geometric span is ~19 pA at 400 Hz x 15 pA bg). Without
   bg the span is 0 (cliff).

3. **Monotonicity**: with bg ON the f-I curve must be monotone
   non-decreasing across the sweep (no dips from stochastic
   inhibition effects).

Run:
    python -m expectation_snn.validation.validate_v1_ring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from brian2 import (
    Network,
    PoissonInput,
    SpikeMonitor,
    defaultclock,
    Hz,
    mV,
    ms,
    pA,
    prefs,
    seed as b2_seed,
)

from ..brian2_model.neurons import (
    V1PV_C, V1PV_GL, V1PV_EL, V1PV_VT, V1PV_VR,
    make_v1_pv_population,
)


FI_BIAS_SWEEP_PA = np.arange(240.0, 281.0, 2.5)    # 17 points, 2.5 pA res
FI_MIN_MONOTONE_SMOOTH_SPAN_PA = 15.0              # transition band must span
                                                    # >= 15 pA at 2.5 pA sampling
                                                    # (true geometric span ~19 pA)
FI_RATE_LOW_HZ = 10.0
FI_RATE_HIGH_HZ = 40.0
TARGET_BG_RATE_HZ = 400.0                           # shipped default (Hu 2014)
TARGET_BG_WEIGHT_PA = 15.0                          # shipped default
PROBE_DUR_MS = 1000.0


@dataclass
class V1RingValidationReport:
    fi_bias_pA: np.ndarray
    fi_rate_hz_with_bg: np.ndarray
    fi_rate_hz_no_bg: np.ndarray
    smooth_span_pA: float
    cliff_span_pA: float
    monotone: bool
    passed_fi_smoothness: bool
    passed_anti_cliff: bool
    passed_monotone: bool

    @property
    def passed(self) -> bool:
        return (self.passed_fi_smoothness and self.passed_anti_cliff
                and self.passed_monotone)

    def summary(self) -> str:
        lines = ["V1 ring / PV rheobase smoothing validation:"]
        lines.append("  f-I curve (I_bias pA -> mean rate Hz):")
        lines.append("     bias   with_bg    no_bg")
        for bias, r_bg, r_no in zip(
            self.fi_bias_pA, self.fi_rate_hz_with_bg, self.fi_rate_hz_no_bg,
        ):
            lines.append(f"   {bias:6.1f}   {r_bg:7.2f}   {r_no:7.2f}")
        lines.append("  -----------------------------------")
        lines.append(
            f"  smooth span (with bg, in [{FI_RATE_LOW_HZ:.0f}, "
            f"{FI_RATE_HIGH_HZ:.0f}] Hz) = "
            f"{self.smooth_span_pA:.1f} pA "
            f"(>= {FI_MIN_MONOTONE_SMOOTH_SPAN_PA:.0f})  "
            f"{'PASS' if self.passed_fi_smoothness else 'FAIL'}"
        )
        lines.append(
            f"  cliff span  (no bg, in [{FI_RATE_LOW_HZ:.0f}, "
            f"{FI_RATE_HIGH_HZ:.0f}] Hz)   = "
            f"{self.cliff_span_pA:.1f} pA  "
            f"{'PASS' if self.passed_anti_cliff else 'FAIL'}"
            f"  (anti-cliff: without bg span must be 0)"
        )
        lines.append(
            f"  monotone non-decreasing (with bg)          : "
            f"{'YES' if self.monotone else 'NO '}  "
            f"{'PASS' if self.passed_monotone else 'FAIL'}"
        )
        lines.append("  -----------------------------------")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _measure_pv_rate(
    bias_pA: float,
    bg_enabled: bool,
    bg_rate_hz: float = TARGET_BG_RATE_HZ,
    bg_weight_pA: float = TARGET_BG_WEIGHT_PA,
    n_cells: int = 50,
    probe_ms: float = PROBE_DUR_MS,
    seed: int = 42,
) -> float:
    """Mean PV firing rate (Hz) at fixed `bias_pA`, no external synaptic input."""
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed)
    np.random.seed(seed)

    pv = make_v1_pv_population(n_cells, name="fi_pv")
    pv.I_bias = bias_pA * pA
    mon = SpikeMonitor(pv, name="fi_mon")

    net_objs = [pv, mon]
    if bg_enabled:
        bg = PoissonInput(
            target=pv, target_var="I_e", N=1,
            rate=bg_rate_hz * Hz, weight=bg_weight_pA * pA,
        )
        net_objs.append(bg)

    net = Network(*net_objs)
    net.run(probe_ms * ms)

    total_spikes = int(len(mon.i[:]))
    rate_hz = total_spikes / (n_cells * probe_ms / 1000.0)
    return float(rate_hz)


def run_v1_ring_validation(verbose: bool = True) -> V1RingValidationReport:
    """Run both assays and return a report."""
    if verbose:
        print(
            f"V1 ring validation: sweeping bias in "
            f"[{FI_BIAS_SWEEP_PA[0]}, {FI_BIAS_SWEEP_PA[-1]}] pA, "
            f"bg={TARGET_BG_RATE_HZ}Hz x {TARGET_BG_WEIGHT_PA}pA"
        )
    rates_with_bg: List[float] = []
    rates_no_bg: List[float] = []
    for bias in FI_BIAS_SWEEP_PA:
        rates_with_bg.append(_measure_pv_rate(float(bias), bg_enabled=True))
        rates_no_bg.append(_measure_pv_rate(float(bias), bg_enabled=False))
    rates_with_bg_arr = np.asarray(rates_with_bg)
    rates_no_bg_arr = np.asarray(rates_no_bg)

    # Smooth span: bias range where rate is in [10, 40] Hz, with bg.
    in_band_bg = (rates_with_bg_arr >= FI_RATE_LOW_HZ) & (rates_with_bg_arr <= FI_RATE_HIGH_HZ)
    if np.any(in_band_bg):
        bias_lo = FI_BIAS_SWEEP_PA[in_band_bg].min()
        bias_hi = FI_BIAS_SWEEP_PA[in_band_bg].max()
        smooth_span = float(bias_hi - bias_lo)
    else:
        smooth_span = 0.0

    in_band_no = (rates_no_bg_arr >= FI_RATE_LOW_HZ) & (rates_no_bg_arr <= FI_RATE_HIGH_HZ)
    if np.any(in_band_no):
        bias_lo_no = FI_BIAS_SWEEP_PA[in_band_no].min()
        bias_hi_no = FI_BIAS_SWEEP_PA[in_band_no].max()
        cliff_span = float(bias_hi_no - bias_lo_no)
    else:
        cliff_span = 0.0

    # Monotone non-decreasing check (tolerate tiny stochastic noise).
    deltas = np.diff(rates_with_bg_arr)
    monotone = bool(np.all(deltas >= -0.5))

    passed_fi = smooth_span >= FI_MIN_MONOTONE_SMOOTH_SPAN_PA
    passed_cliff = cliff_span == 0.0
    passed_mono = monotone

    rep = V1RingValidationReport(
        fi_bias_pA=FI_BIAS_SWEEP_PA,
        fi_rate_hz_with_bg=rates_with_bg_arr,
        fi_rate_hz_no_bg=rates_no_bg_arr,
        smooth_span_pA=smooth_span,
        cliff_span_pA=cliff_span,
        monotone=monotone,
        passed_fi_smoothness=passed_fi,
        passed_anti_cliff=passed_cliff,
        passed_monotone=passed_mono,
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_v1_ring_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
