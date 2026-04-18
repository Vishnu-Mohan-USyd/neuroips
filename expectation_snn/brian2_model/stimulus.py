"""Stimulus generators.

One primary function now (Stage 0 needs only this):

- `drifting_grating_input(theta_rad, contrast, duration_ms, dt_ms)`:
    returns per-channel Poisson rate arrays (N_CHANNELS,) over the window.
    The caller applies these to `V1Ring.stim` via `v1_ring.set_stimulus`.

Assay stubs — bodies filled out in a later sprint once calibration is
green:

- `kok_trial(...)` — 500 ms cue -> 500 ms gap -> 500 ms grating
    (45 deg or 135 deg, 75 pct validity, ~3.5 s ITI, plus omission).
    Source: Kok 2012 PMID 22841311.
- `richter_pair(...)` — 500 ms leader -> 500 ms trailer, no ISI,
    ~5 s ITI. Source: Richter 2022 DOI 10.1093/oons/kvac013.
- `tang_rotating(...)` — 6 orientations at 30 deg steps, 4 Hz / 250 ms
    / no ISI, rotating blocks of 5-9 items before 12 pct deviants.
    Source: Tang 2023 PMID 36864037. Timing verified from paper PDF,
    see docs/research_log.md.

All assay stubs return `TrialPlan` dataclasses describing the sequence of
(theta_rad, contrast, duration_ms) entries; the driver module
(brian2_model/train.py or validation/*) consumes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Shared ring geometry: keep in sync with v1_ring / h_ring.
N_CHANNELS = 12
CHANNEL_SPACING_RAD = np.pi / N_CHANNELS  # 15 deg
DEFAULT_STIM_PEAK_HZ = 80.0
DEFAULT_STIM_SIGMA_DEG = 15.0


@dataclass
class TrialItem:
    """One timed stimulus epoch."""
    theta_rad: Optional[float]   # None -> blank / ISI
    contrast: float
    duration_ms: float
    kind: str = "grating"        # "grating", "blank", "cue", "deviant"


@dataclass
class TrialPlan:
    """A full trial = list of TrialItems + per-trial metadata."""
    items: List[TrialItem]
    meta: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return float(sum(it.duration_ms for it in self.items))


def drifting_grating_input(
    theta_rad: float,
    contrast: float,
    duration_ms: float,
    dt_ms: float = 1.0,
    peak_rate_hz: float = DEFAULT_STIM_PEAK_HZ,
    sigma_deg: float = DEFAULT_STIM_SIGMA_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel Poisson rates for a drifting grating over a window.

    Parameters
    ----------
    theta_rad : float
        Grating orientation in radians (0..pi).
    contrast : float
        Linear scale on peak rate; 0..1 typical.
    duration_ms : float
        Stimulus duration in ms.
    dt_ms : float
        Sampling interval for the returned time axis.
    peak_rate_hz : float
        Peak Poisson rate at preferred channel at full contrast.
    sigma_deg : float
        Gaussian tuning width (degrees) over the ring.

    Returns
    -------
    t_ms : np.ndarray, shape (n_steps,)
        Time axis starting at 0.
    rates_hz : np.ndarray, shape (n_steps, N_CHANNELS)
        Per-channel Poisson rate at each time step. Constant across the
        window for drifting gratings in plan v5 (we don't model phase /
        temporal-frequency modulation at the rate level).
    """
    n_steps = max(1, int(round(duration_ms / dt_ms)))
    t_ms = np.arange(n_steps, dtype=np.float64) * dt_ms
    thetas = np.arange(N_CHANNELS) * CHANNEL_SPACING_RAD
    d = np.abs(thetas - float(theta_rad))
    d = np.minimum(d, np.pi - d)  # wrap-around on 0..pi ring
    sigma_rad = np.deg2rad(sigma_deg)
    per_channel = peak_rate_hz * float(contrast) * np.exp(
        -0.5 * (d / sigma_rad) ** 2
    )
    rates = np.broadcast_to(per_channel[None, :], (n_steps, N_CHANNELS)).copy()
    return t_ms, rates


# ------------------- assay stubs (to be fleshed out later) ------------------

def kok_trial(
    rng: np.random.Generator,
    expected_theta_rad: float,
    validity: float = 0.75,
    cue_ms: float = 500.0,
    gap_ms: float = 500.0,
    grating_ms: float = 500.0,
    iti_ms: float = 3500.0,
    include_omission: bool = False,
) -> TrialPlan:
    """Kok 2012 cue -> gap -> grating (45 or 135 deg), with omission.

    NOT IMPLEMENTED. Returns NotImplementedError. Will be filled out
    after Stage 0 gate is green.
    """
    raise NotImplementedError("kok_trial() — deferred to post-stage-0 sprint")


def richter_pair(
    rng: np.random.Generator,
    leader_idx: int,
    trailer_idx: int,
    leader_ms: float = 500.0,
    trailer_ms: float = 500.0,
    iti_ms: float = 5000.0,
) -> TrialPlan:
    """Richter 2022 6 x 6 leader -> trailer cross-over (no ISI between).

    NOT IMPLEMENTED. Deferred.
    """
    raise NotImplementedError("richter_pair() — deferred to post-stage-0 sprint")


def tang_rotating(
    rng: np.random.Generator,
    n_items: int = 1000,
    item_ms: float = 250.0,
    block_len_range: Tuple[int, int] = (5, 9),
    deviant_per_block: bool = True,
    orientations_deg: Sequence[float] = (0, 30, 60, 90, 120, 150),
) -> TrialPlan:
    """Tang 2023 rotating-orientation paradigm (verified in research_log.md).

    NOT IMPLEMENTED. Deferred.
    """
    raise NotImplementedError("tang_rotating() — deferred to post-stage-0 sprint")


# ------------------- self-check / smoke -------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) drifting_grating_input sanity: peak at preferred channel, Gaussian.
    theta = 0.0  # 0 deg -> channel 0
    t, rates = drifting_grating_input(theta, contrast=1.0, duration_ms=250.0)
    assert rates.shape == (250, N_CHANNELS), rates.shape
    assert np.argmax(rates[0]) == 0, (
        f"Expected peak at ch0, got ch{int(np.argmax(rates[0]))}"
    )
    # Stationary in time for drifting grating (no phase modulation modelled).
    assert np.allclose(rates[0], rates[-1]), "rates should be constant in time"

    peak0 = rates[0, 0]
    ch6 = rates[0, 6]   # 90 deg away -> near zero
    assert peak0 > DEFAULT_STIM_PEAK_HZ * 0.9, (
        f"Peak too small: {peak0:.2f}"
    )
    assert ch6 < 1e-3, f"Orthogonal channel should be ~zero, got {ch6:.4f}"
    print(f"stimulus smoke: drifting_grating at theta=0deg -> "
          f"ch0={peak0:.2f} Hz, ch6={ch6:.2e} Hz")

    # 2) Contrast scales linearly.
    _, low = drifting_grating_input(theta, contrast=0.25, duration_ms=10.0)
    assert abs(low[0, 0] - 0.25 * DEFAULT_STIM_PEAK_HZ) < 1e-6
    print(f"stimulus smoke: contrast scaling OK")

    # 3) Orientation wrap-around: theta just below pi is near ch0 (since
    #    pi-eps wraps to ~0 on the orientation ring 0..pi).
    _, nearwrap = drifting_grating_input(
        np.pi - 1e-3, contrast=1.0, duration_ms=10.0
    )
    assert np.argmax(nearwrap[0]) == 0, (
        f"Near-wrap should peak at ch0, got ch{int(np.argmax(nearwrap[0]))}"
    )
    print(f"stimulus smoke: wrap-around at theta=(pi-eps) peaks at ch0")

    # 4) Assay stubs raise NotImplementedError.
    for fn, args in [
        (kok_trial, (rng, 0.0)),
        (richter_pair, (rng, 0, 1)),
        (tang_rotating, (rng,)),
    ]:
        try:
            fn(*args)
            raise AssertionError(f"{fn.__name__} should raise NotImplementedError")
        except NotImplementedError:
            print(f"stimulus smoke: {fn.__name__} stub raises as expected")

    print("stimulus smoke: PASS")
