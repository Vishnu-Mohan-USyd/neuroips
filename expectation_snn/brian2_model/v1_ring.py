"""V1 fixed feature-tuned ring (plan §1 v5).

Architecture
------------
- 12 orientation channels at 15° spacing over 0-165° (in radians:
  `thetas_rad = arange(12) * pi/12`).
- Per channel: 16 E cells (LIF soma + passive apical + SFA, from
  `neurons.make_v1_e_population`) and 4 co-tuned SOM cells.
- One broad PV pool: 32 cells, shared across all 12 channels.
- Weak nearest-neighbor E<->E ring (cross-channel, wrap-around) for
  mild competition.
- Stimulus drives V1_E somatic `I_e` via a fixed Gaussian feature-tuning
  profile of width `sigma_stim_deg` (default 15°).
- PV -> E synapses: Vogels iSTDP stabilizer (Stage-0 trained, frozen
  afterwards). E -> PV: sparse fixed excitation.
- SOM <-> E within-channel: SOM is co-tuned to its channel; E -> SOM
  excites, SOM -> E inhibits.

All non-stabilizer synapses are fixed once calibrated. PV iSTDP is
active only during Stage-0 calibration and frozen afterwards
(caller-level responsibility).

References
----------
- Niell & Stryker 2008 (PMID 18562647) — V1 E rate band 2-8 Hz awake.
- Hu, Gan, Jonas 2014 (Science 345:1255263) — PV fast-spiking kinetics.
- Urban-Ciecko & Barth 2016 (Nat Rev Neurosci 17:401) — SOM Martinotti.
- Vogels 2011 (PMID 22075724) — iSTDP rate control on inhibitory synapses.
- Ko et al. 2011 (Nature 473:87) — feature-specific local E-E connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brian2 import (
    NeuronGroup,
    PoissonGroup,
    Synapses,
    Hz,
    ms,
    mV,
    pA,
)

from .neurons import (
    make_v1_e_population,
    make_v1_pv_population,
    make_v1_som_population,
)
from .plasticity import vogels_istdp

# -- constants ---------------------------------------------------------------

N_CHANNELS = 12
CHANNEL_SPACING_RAD = np.pi / N_CHANNELS  # 15° in radians
N_E_PER_CHANNEL = 16
N_SOM_PER_CHANNEL = 4
N_PV_POOL = 32

# Default stimulus/tuning parameters (overridable via V1RingConfig).
DEFAULT_SIGMA_STIM_DEG = 15.0
DEFAULT_STIM_MAX_RATE_HZ = 80.0
DEFAULT_N_STIM_AFFERENTS_PER_CHANNEL = 20


@dataclass
class V1RingConfig:
    """Tunable V1 ring configuration.

    Parameters
    ----------
    sigma_stim_deg : float
        Gaussian width of stimulus -> V1_E feature tuning, in degrees.
    stim_max_rate_hz : float
        Peak Poisson rate on a stimulus-afferent channel at the preferred
        orientation and full contrast.
    n_stim_afferents_per_channel : int
        Independent Poisson afferents per feature channel. Each afferent
        contacts all 16 E cells of its channel with a fixed weight.
    w_stim_e : float
        Weight (dimensionless multiplier) on stimulus -> V1_E synapses.
        Combined with `drive_amp_stim_pA` this sets soma drive per spike.
    drive_amp_stim_pA : float
        Current amplitude deposited into `I_e` per afferent spike (pA).
    w_ee_ring : float
        Weight of nearest-neighbour V1 E -> E cross-channel synapse.
    drive_amp_ee_pA : float
        Current amplitude on local E -> E spike.
    w_e_pv : float
        E -> PV weight (fixed).
    p_e_pv : float
        E -> PV connection probability.
    drive_amp_e_pv_pA : float
        Current amplitude for E -> PV drive.
    w_pv_e_init : float
        Initial PV -> E weight before iSTDP.
    drive_amp_pv_e_pA : float
        Current amplitude for PV -> E inhibitory drive.
    pv_rho_hz : float
        Target postsynaptic E rate for PV iSTDP (Stage 0 calibration).
    pv_eta : float
        PV iSTDP learning rate.
    w_e_som : float
        E -> SOM weight (co-tuned within-channel).
    drive_amp_e_som_pA : float
        Current amplitude for E -> SOM drive.
    w_som_e : float
        SOM -> E weight (co-tuned).
    drive_amp_som_e_pA : float
        Current amplitude for SOM -> E inhibitory drive.
    """

    sigma_stim_deg: float = DEFAULT_SIGMA_STIM_DEG
    stim_max_rate_hz: float = DEFAULT_STIM_MAX_RATE_HZ
    n_stim_afferents_per_channel: int = DEFAULT_N_STIM_AFFERENTS_PER_CHANNEL
    w_stim_e: float = 1.0
    drive_amp_stim_pA: float = 35.0

    w_ee_ring: float = 0.05
    drive_amp_ee_pA: float = 5.0

    w_e_pv: float = 0.3
    p_e_pv: float = 0.25
    drive_amp_e_pv_pA: float = 15.0

    w_pv_e_init: float = 0.5
    drive_amp_pv_e_pA: float = 25.0
    pv_rho_hz: float = 5.0
    pv_eta: float = 5e-3

    w_e_som: float = 0.3
    drive_amp_e_som_pA: float = 15.0
    w_som_e: float = 0.5
    drive_amp_som_e_pA: float = 20.0


@dataclass
class V1Ring:
    """Container returned by `build_v1_ring`."""
    # groups
    e: NeuronGroup                      # flat (12 * 16) E cells, channel-major
    som: NeuronGroup                    # flat (12 * 4) SOM, channel-major
    pv: NeuronGroup                     # 32 broad PV
    stim: PoissonGroup                  # flat (12 * n_aff) afferents, channel-major
    # synapses
    stim_to_e: Synapses
    ee_ring: Synapses
    e_to_pv: Synapses
    pv_to_e: Synapses                   # Vogels iSTDP handle (plastic in Stage 0)
    e_to_som: Synapses
    som_to_e: Synapses
    # config + channel ids
    config: V1RingConfig
    thetas_rad: np.ndarray              # (N_CHANNELS,) channel preferred orientations
    e_channel: np.ndarray               # (n_e,) channel index per E cell
    som_channel: np.ndarray             # (n_som,) channel index per SOM cell
    stim_channel: np.ndarray            # (n_stim,) channel index per afferent
    # brian2 objects the caller needs for Network(...) add
    groups: List[object] = field(default_factory=list)


# -- helpers ------------------------------------------------------------------

def _ring_distance_rad(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Minimum wrap-around distance on a 0..pi ring (orientation is mod pi)."""
    d = np.abs(theta_a - theta_b)
    return np.minimum(d, np.pi - d)


def stimulus_tuning_profile(theta_rad: float, config: V1RingConfig) -> np.ndarray:
    """Per-channel Poisson rate (Hz) for a grating at orientation `theta_rad`.

    Gaussian on ring distance with sigma = sigma_stim_deg. Returns shape
    (N_CHANNELS,).
    """
    thetas = np.arange(N_CHANNELS) * CHANNEL_SPACING_RAD
    d_rad = _ring_distance_rad(thetas, np.asarray(theta_rad))
    sigma_rad = np.deg2rad(config.sigma_stim_deg)
    return config.stim_max_rate_hz * np.exp(-0.5 * (d_rad / sigma_rad) ** 2)


# -- builder -----------------------------------------------------------------

def build_v1_ring(config: Optional[V1RingConfig] = None,
                  name_prefix: str = "v1") -> V1Ring:
    """Construct the full V1 ring circuit.

    Returns a V1Ring dataclass holding NeuronGroups, Synapses, and channel
    labellings. The caller owns the Network; the returned `groups` list can
    be splatted into `Network(*ring.groups, ...)`.

    Notes
    -----
    - E and SOM cells are channel-major-ordered: cell `i` has channel
      `i // N_{E,SOM}_PER_CHANNEL`.
    - All explicit post-spike currents are driven in pA (the plasticity
      factories embed the scale; here we also include explicit `on_pre`
      expressions for the non-plastic Synapses objects).
    """
    cfg = config or V1RingConfig()
    thetas_rad = np.arange(N_CHANNELS) * CHANNEL_SPACING_RAD

    n_e = N_CHANNELS * N_E_PER_CHANNEL
    n_som = N_CHANNELS * N_SOM_PER_CHANNEL
    n_stim = N_CHANNELS * cfg.n_stim_afferents_per_channel

    e_channel = np.repeat(np.arange(N_CHANNELS), N_E_PER_CHANNEL)
    som_channel = np.repeat(np.arange(N_CHANNELS), N_SOM_PER_CHANNEL)
    stim_channel = np.repeat(np.arange(N_CHANNELS),
                             cfg.n_stim_afferents_per_channel)

    # --- neuron groups
    e = make_v1_e_population(n_e, name=f"{name_prefix}_e")
    som = make_v1_som_population(n_som, name=f"{name_prefix}_som")
    pv = make_v1_pv_population(N_PV_POOL, name=f"{name_prefix}_pv")

    # Stimulus afferents: `rates` is set by the caller via stimulus.py.
    stim = PoissonGroup(n_stim, rates=0 * Hz, name=f"{name_prefix}_stim")

    # --- stimulus -> E: afferent channel c contacts every E cell in channel c.
    stim_to_e = Synapses(
        stim, e,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_stim_pA}*pA",
        name=f"{name_prefix}_stim_to_e",
    )
    i_stim: list[int] = []
    j_e: list[int] = []
    for k in range(n_stim):
        c = int(stim_channel[k])
        for j in range(c * N_E_PER_CHANNEL, (c + 1) * N_E_PER_CHANNEL):
            i_stim.append(k)
            j_e.append(j)
    stim_to_e.connect(i=np.asarray(i_stim, dtype=np.int64),
                      j=np.asarray(j_e, dtype=np.int64))
    stim_to_e.w = cfg.w_stim_e

    # --- nearest-neighbour E<->E cross-channel ring (wrap-around).
    # For each source E cell in channel c, connect to every E cell in
    # channels c-1 and c+1 (mod N_CHANNELS); exclude same-channel pairs.
    ee_ring = Synapses(
        e, e,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_ee_pA}*pA",
        name=f"{name_prefix}_ee_ring",
    )
    i_src, j_tgt = [], []
    for ci in range(N_CHANNELS):
        for dc in (-1, +1):
            cj = (ci + dc) % N_CHANNELS
            for ii in range(ci * N_E_PER_CHANNEL, (ci + 1) * N_E_PER_CHANNEL):
                for jj in range(cj * N_E_PER_CHANNEL, (cj + 1) * N_E_PER_CHANNEL):
                    i_src.append(ii)
                    j_tgt.append(jj)
    ee_ring.connect(i=np.asarray(i_src, dtype=np.int64),
                    j=np.asarray(j_tgt, dtype=np.int64))
    ee_ring.w = cfg.w_ee_ring

    # --- E -> PV: sparse random with probability p_e_pv.
    e_to_pv = Synapses(
        e, pv,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_e_pv_pA}*pA",
        name=f"{name_prefix}_e_to_pv",
    )
    e_to_pv.connect(p=cfg.p_e_pv)
    e_to_pv.w = cfg.w_e_pv

    # --- PV -> E: Vogels iSTDP (plastic in Stage 0).
    pv_to_e = vogels_istdp(
        pv, e,
        connectivity="True",
        w_init=cfg.w_pv_e_init,
        w_max=10.0,
        eta=cfg.pv_eta,
        rho=cfg.pv_rho_hz * Hz,
        tau=20 * ms,
        drive_amp_pA=cfg.drive_amp_pv_e_pA,
        name=f"{name_prefix}_pv_to_e",
    )

    # --- E -> SOM within-channel co-tuning.
    e_to_som = Synapses(
        e, som,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_e_som_pA}*pA",
        name=f"{name_prefix}_e_to_som",
    )
    i_es, j_es = [], []
    for c in range(N_CHANNELS):
        for ii in range(c * N_E_PER_CHANNEL, (c + 1) * N_E_PER_CHANNEL):
            for jj in range(c * N_SOM_PER_CHANNEL, (c + 1) * N_SOM_PER_CHANNEL):
                i_es.append(ii)
                j_es.append(jj)
    e_to_som.connect(i=np.asarray(i_es, dtype=np.int64),
                     j=np.asarray(j_es, dtype=np.int64))
    e_to_som.w = cfg.w_e_som

    # --- SOM -> E within-channel (co-tuned suppression).
    som_to_e = Synapses(
        som, e,
        model="w : 1",
        on_pre=f"I_i_post += w * {cfg.drive_amp_som_e_pA}*pA",
        name=f"{name_prefix}_som_to_e",
    )
    i_se, j_se = [], []
    for c in range(N_CHANNELS):
        for ii in range(c * N_SOM_PER_CHANNEL, (c + 1) * N_SOM_PER_CHANNEL):
            for jj in range(c * N_E_PER_CHANNEL, (c + 1) * N_E_PER_CHANNEL):
                i_se.append(ii)
                j_se.append(jj)
    som_to_e.connect(i=np.asarray(i_se, dtype=np.int64),
                     j=np.asarray(j_se, dtype=np.int64))
    som_to_e.w = cfg.w_som_e

    ring = V1Ring(
        e=e, som=som, pv=pv, stim=stim,
        stim_to_e=stim_to_e, ee_ring=ee_ring,
        e_to_pv=e_to_pv, pv_to_e=pv_to_e,
        e_to_som=e_to_som, som_to_e=som_to_e,
        config=cfg, thetas_rad=thetas_rad,
        e_channel=e_channel, som_channel=som_channel,
        stim_channel=stim_channel,
    )
    ring.groups = [
        e, som, pv, stim,
        stim_to_e, ee_ring, e_to_pv, pv_to_e, e_to_som, som_to_e,
    ]
    return ring


def set_stimulus(ring: V1Ring, theta_rad: float, contrast: float = 1.0) -> None:
    """Set Poisson afferent rates for a grating at `theta_rad`, `contrast`.

    Parameters
    ----------
    ring : V1Ring
    theta_rad : float
        Orientation in radians (0..pi).
    contrast : float
        Scales Poisson peak rate linearly; 1.0 = full contrast.

    Notes
    -----
    Every afferent in channel c gets the same rate; variability across
    the afferent pool is the Poisson draw at simulation time.
    """
    per_channel = stimulus_tuning_profile(theta_rad, ring.config) * contrast
    rates = per_channel[ring.stim_channel]
    ring.stim.rates = rates * Hz


# -- self-check / smoke --------------------------------------------------------

if __name__ == "__main__":
    from brian2 import Network, SpikeMonitor, defaultclock, prefs, seed as b2_seed

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42); np.random.seed(42)

    ring = build_v1_ring()
    set_stimulus(ring, theta_rad=0.0, contrast=1.0)

    e_mon = SpikeMonitor(ring.e)
    pv_mon = SpikeMonitor(ring.pv)
    som_mon = SpikeMonitor(ring.som)
    net = Network(*ring.groups, e_mon, pv_mon, som_mon)

    net.run(500 * ms)
    e_rate = e_mon.num_spikes / (len(ring.e) * 0.5)
    pv_rate = pv_mon.num_spikes / (len(ring.pv) * 0.5)
    som_rate = som_mon.num_spikes / (len(ring.som) * 0.5)

    # Report by channel for E cells (so we can see orientation tuning).
    e_idx = np.asarray(e_mon.i[:])
    per_ch = np.bincount(ring.e_channel[e_idx], minlength=N_CHANNELS) / (
        N_E_PER_CHANNEL * 0.5)

    print(f"v1_ring smoke: pop rates  E={e_rate:.2f} Hz  "
          f"PV={pv_rate:.2f} Hz  SOM={som_rate:.2f} Hz")
    print(f"v1_ring smoke: per-channel E rates:")
    for c, r in enumerate(per_ch):
        mark = "  <- preferred" if c == 0 else ""
        print(f"  ch{c:02d} theta={np.rad2deg(ring.thetas_rad[c]):5.1f}  {r:6.2f} Hz{mark}")
    peak_c = int(np.argmax(per_ch))
    assert peak_c == 0, f"Expected peak at ch0 (0 deg), got ch{peak_c}"
    assert per_ch[0] > per_ch[6], "Preferred > orthogonal"
    print("v1_ring smoke: PASS (peak at preferred channel)")
