"""Primitive validation for the V1-local E/PV/SOM inhibitory motif.

This validator is intentionally bottom-up: it builds only fresh V1 rings,
loads the existing Stage-0 V1 calibration state, drives a fixed channel-0
grating, and checks that local inhibitory circuitry keeps V1 in the
Stage-0 rate bands. It does not build H, train plasticity, or run an assay.

Run:
    python -m expectation_snn.validation.validate_v1_inhibitory_motif
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from brian2 import (
    Network,
    SpikeMonitor,
    defaultclock,
    ms,
    pA,
    prefs,
    seed as b2_seed,
    start_scope,
)

from ..brian2_model.v1_ring import (
    N_CHANNELS,
    N_E_PER_CHANNEL,
    build_v1_ring,
    set_stimulus,
)
from .stage_0_gate import (
    RUNAWAY_CEILING_HZ,
    V1_E_RATE_BAND_HZ,
    V1_PV_RATE_BAND_HZ,
    V1_SOM_RATE_BAND_HZ,
)


SEED = 42
PROBE_MS = 500.0
THETA_RAD = 0.0
CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "checkpoints"
    / f"stage_0_seed{SEED}.npz"
)

# The Stage-0-calibrated PV weights are modest over a 500 ms window, so the
# causal check uses two deterministic, spike-count-resolvable thresholds:
# at least +0.10 Hz population-wide and +1 Hz on the preferred E channel.
MIN_POP_E_DISINHIBITION_HZ = 0.10
MIN_PEAK_E_DISINHIBITION_HZ = 1.00


@dataclass
class MotifRates:
    """Compact V1 rate report for one short grating probe."""

    e_rate_hz: float
    pv_rate_hz: float
    som_rate_hz: float
    per_channel_e_hz: np.ndarray

    @property
    def peak_e_rate_hz(self) -> float:
        return float(self.per_channel_e_hz.max())


def _load_stage0_v1_state(ring) -> Tuple[float, int]:
    """Apply Stage-0 V1 E bias and PV->E weights to a fresh V1 ring."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"missing Stage-0 checkpoint: {CHECKPOINT_PATH}")
    data = np.load(CHECKPOINT_PATH)
    bias_pA = float(data["bias_pA"])
    pv_w = np.asarray(data["pv_to_e_w"], dtype=np.float64)
    live = np.asarray(ring.pv_to_e.w[:], dtype=np.float64)
    if pv_w.shape != live.shape:
        raise ValueError(
            f"Stage-0 pv_to_e_w shape {pv_w.shape} != live {live.shape}"
        )
    ring.e.I_bias = bias_pA * pA
    ring.pv_to_e.w[:] = pv_w
    ring.pv_to_e.active = False
    return bias_pA, int(pv_w.size)


def _run_probe(*, ablate_inhibition: bool, name_prefix: str) -> MotifRates:
    """Build one fresh V1 ring and measure a 500 ms channel-0 grating probe."""
    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)

    ring = build_v1_ring(name_prefix=name_prefix)
    _load_stage0_v1_state(ring)
    if ablate_inhibition:
        ring.pv_to_e.w[:] = 0.0
        ring.som_to_e.w[:] = 0.0

    set_stimulus(ring, theta_rad=THETA_RAD, contrast=1.0)
    e_mon = SpikeMonitor(ring.e, name=f"{name_prefix}_e_mon")
    pv_mon = SpikeMonitor(ring.pv, name=f"{name_prefix}_pv_mon")
    som_mon = SpikeMonitor(ring.som, name=f"{name_prefix}_som_mon")
    net = Network(*ring.groups, e_mon, pv_mon, som_mon)
    net.run(PROBE_MS * ms)

    dur_s = PROBE_MS * 1e-3
    e_rate = float(e_mon.num_spikes) / (int(ring.e.N) * dur_s)
    pv_rate = float(pv_mon.num_spikes) / (int(ring.pv.N) * dur_s)
    som_rate = float(som_mon.num_spikes) / (int(ring.som.N) * dur_s)
    e_idx = np.asarray(e_mon.i[:], dtype=np.int64)
    per_ch = np.bincount(
        ring.e_channel[e_idx],
        minlength=N_CHANNELS,
    ).astype(np.float64) / (N_E_PER_CHANNEL * dur_s)
    return MotifRates(e_rate, pv_rate, som_rate, per_ch)


def _assert_band(name: str, value: float, band: Tuple[float, float]) -> None:
    assert band[0] <= value <= band[1], (
        f"{name}={value:.3f} Hz outside band [{band[0]:.3f}, {band[1]:.3f}]"
    )


def main() -> int:
    intact = _run_probe(ablate_inhibition=False, name_prefix="v1_motif_intact")
    ablated = _run_probe(ablate_inhibition=True, name_prefix="v1_motif_ablated")

    _assert_band("V1_E intact", intact.e_rate_hz, V1_E_RATE_BAND_HZ)
    _assert_band("V1_PV intact", intact.pv_rate_hz, V1_PV_RATE_BAND_HZ)
    _assert_band("V1_SOM intact", intact.som_rate_hz, V1_SOM_RATE_BAND_HZ)
    assert max(intact.e_rate_hz, intact.pv_rate_hz, intact.som_rate_hz) <= RUNAWAY_CEILING_HZ, (
        f"runaway rate under intact V1: E={intact.e_rate_hz:.3f}, "
        f"PV={intact.pv_rate_hz:.3f}, SOM={intact.som_rate_hz:.3f} Hz"
    )

    pop_delta = ablated.e_rate_hz - intact.e_rate_hz
    peak_delta = ablated.peak_e_rate_hz - intact.peak_e_rate_hz
    assert pop_delta >= MIN_POP_E_DISINHIBITION_HZ, (
        f"PV/SOM ablation should increase population V1_E by at least "
        f"{MIN_POP_E_DISINHIBITION_HZ:.2f} Hz; got {pop_delta:.3f} Hz"
    )
    assert peak_delta >= MIN_PEAK_E_DISINHIBITION_HZ, (
        f"PV/SOM ablation should increase preferred-channel V1_E by at least "
        f"{MIN_PEAK_E_DISINHIBITION_HZ:.2f} Hz; got {peak_delta:.3f} Hz"
    )
    assert max(ablated.e_rate_hz, ablated.pv_rate_hz, ablated.som_rate_hz) <= RUNAWAY_CEILING_HZ, (
        f"runaway rate under PV/SOM ablation: E={ablated.e_rate_hz:.3f}, "
        f"PV={ablated.pv_rate_hz:.3f}, SOM={ablated.som_rate_hz:.3f} Hz"
    )

    print("validate_v1_inhibitory_motif: PASS")
    print(
        "  intact rates: "
        f"E={intact.e_rate_hz:.3f} Hz "
        f"PV={intact.pv_rate_hz:.3f} Hz "
        f"SOM={intact.som_rate_hz:.3f} Hz "
        f"peak_E={intact.peak_e_rate_hz:.3f} Hz"
    )
    print(
        "  PV/SOM ablated: "
        f"E={ablated.e_rate_hz:.3f} Hz "
        f"peak_E={ablated.peak_e_rate_hz:.3f} Hz "
        f"delta_E={pop_delta:.3f} Hz "
        f"delta_peak_E={peak_delta:.3f} Hz"
    )
    print(
        "  bands: "
        f"E={V1_E_RATE_BAND_HZ} PV={V1_PV_RATE_BAND_HZ} "
        f"SOM={V1_SOM_RATE_BAND_HZ} runaway<={RUNAWAY_CEILING_HZ:.1f} Hz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
