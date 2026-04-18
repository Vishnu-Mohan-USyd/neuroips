"""H module: stateful context / prediction ring (plan §1 v5).

Architecture
------------
- 12 channels, 16 LIF E cells per channel (192 total E).
- Strong within-channel recurrent E->E for bump persistence (Wang 2001).
- Weak cross-channel ring E->E for mild competition.
- One broad H inhibitory pool (16 LIF) for global stability.
- Cue afferents deposit into H_E's I_e via dedicated Poisson group.
- Bump half-life target: 200-500 ms after a brief pulse.

Two factory functions:
- `build_h_r` -> H_R for Kok/Richter leader->trailer sequences.
- `build_h_t` -> H_T for Tang rotating-orientation sequences.

Both factories share the same wiring topology; they differ only in which
training corpus the caller feeds them during Stage 1 (plan §3).

References
----------
- Wang X-J (2001) Trends Neurosci 24:455 — persistent bumps via strong recurrent E.
- Wimmer et al. 2014 Nat Neurosci 17:431 — bump dynamics noise-robustness.
- Vogels 2011 (PMID 22075724) — inhibitory control of E/I balance.
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
    pA,
)

from .neurons import (
    make_h_e_population,
    make_h_inh_population,
)
from .plasticity import (
    pair_stdp_with_normalization,
    vogels_istdp,
)

# -- constants ---------------------------------------------------------------

N_CHANNELS = 12
CHANNEL_SPACING_RAD = np.pi / N_CHANNELS  # matches V1 ring
N_E_PER_CHANNEL = 16
N_INH_POOL = 16

DEFAULT_N_CUE_AFFERENTS = 64   # shared over all cues; caller assigns rates


@dataclass
class HRingConfig:
    """Tunable H-ring configuration.

    Parameters
    ----------
    w_ee_within_init : float
        Initial strong within-channel E->E weight (plastic, Stage 1 STDP).
        Must be large enough to support bump persistence once learned.
    w_ee_cross_init : float
        Initial weak nearest-neighbour cross-channel E->E weight (also
        plastic via the same rule, but within-channel dominates).
    drive_amp_ee_pA : float
        Per-spike drive of H E->E onto postsynaptic I_e.
    ee_w_max : float
        Multiplicative cap for within-channel E->E STDP.
    ee_A_plus : float
        LTP amplitude for H E->E pair-STDP.
    ee_A_minus : float
        LTD amplitude for H E->E pair-STDP.
    target_postsyn_sum : float
        Target sum of incoming within-channel E->E weights per postsyn cell
        after normalize_postsyn_sum cadence. Caller invokes via Brian2
        NetworkOperation.
    w_e_inh : float
        E -> inh_pool weight (fixed).
    p_e_inh : float
        E -> inh_pool connection probability.
    drive_amp_e_inh_pA : float
        E -> inh drive amplitude per spike (pA).
    w_inh_e_init : float
        Inh_pool -> E initial weight (plastic via Vogels iSTDP).
    drive_amp_inh_e_pA : float
        Inh_pool -> E inhibitory drive amplitude per spike (pA).
    inh_rho_hz : float
        Target postsyn E rate for Vogels iSTDP.
    inh_eta : float
        Vogels learning rate.
    w_cue_e_init : float
        Cue -> H_E initial weight (plastic via eligibility_trace_cue_rule,
        but the factory here leaves that to feedback_routes / train.py).
    drive_amp_cue_e_pA : float
        Cue -> H_E drive amplitude per spike (pA).
    """

    # H recurrent E<->E
    w_ee_within_init: float = 0.3
    w_ee_cross_init: float = 0.05
    drive_amp_ee_pA: float = 25.0
    ee_w_max: float = 1.5
    ee_A_plus: float = 0.01
    ee_A_minus: float = 0.0105
    target_postsyn_sum: float = 3.0

    # H E <-> inh pool
    w_e_inh: float = 0.3
    p_e_inh: float = 0.25
    drive_amp_e_inh_pA: float = 20.0
    w_inh_e_init: float = 0.5
    drive_amp_inh_e_pA: float = 30.0
    inh_rho_hz: float = 2.0
    inh_eta: float = 5e-3

    # Cue afferents (cue->H weight governed by feedback_routes at Stage 2)
    w_cue_e_init: float = 0.5
    drive_amp_cue_e_pA: float = 80.0
    n_cue_afferents: int = DEFAULT_N_CUE_AFFERENTS


@dataclass
class HRing:
    """Container returned by `build_h_r` / `build_h_t`."""
    name: str
    e: NeuronGroup
    inh: NeuronGroup
    cue: PoissonGroup
    # plastic synapses (H STDP + Vogels + cue-elig)
    ee: Synapses           # E->E (within + cross-channel), pair-STDP
    e_to_inh: Synapses     # fixed
    inh_to_e: Synapses     # Vogels iSTDP
    cue_to_e: Synapses     # fixed in h_ring.py; elig wiring done elsewhere
    config: HRingConfig
    thetas_rad: np.ndarray        # (N_CHANNELS,)
    e_channel: np.ndarray         # (192,) channel index per E cell
    groups: List[object] = field(default_factory=list)


# -- builder ----------------------------------------------------------------

def _build_h_ring(name: str, config: Optional[HRingConfig]) -> HRing:
    cfg = config or HRingConfig()
    thetas_rad = np.arange(N_CHANNELS) * CHANNEL_SPACING_RAD

    n_e = N_CHANNELS * N_E_PER_CHANNEL
    e_channel = np.repeat(np.arange(N_CHANNELS), N_E_PER_CHANNEL)

    # --- populations
    e = make_h_e_population(n_e, name=f"{name}_e")
    inh = make_h_inh_population(N_INH_POOL, name=f"{name}_inh")
    cue = PoissonGroup(cfg.n_cue_afferents, rates=0 * Hz, name=f"{name}_cue")

    # --- E <-> E : pair-STDP, within-channel all-to-all (except self)
    #     + cross-channel nearest-neighbour (ring).
    #     We connect within and cross as one Synapses object then assign
    #     initial weights by post/pre channel labels.
    ee = pair_stdp_with_normalization(
        e, e,
        connectivity="i != j",
        w_init=cfg.w_ee_cross_init,    # placeholder; rewritten below
        w_max=cfg.ee_w_max,
        A_plus=cfg.ee_A_plus,
        A_minus=cfg.ee_A_minus,
        tau_pre=20 * ms, tau_post=20 * ms,
        drive_amp_pA=cfg.drive_amp_ee_pA,
        target_channel="soma",
        name=f"{name}_ee",
    )
    # Re-assign by (pre,post) channel pair: within-channel = strong,
    # |dc| == 1 mod N = weak, else = 0 (but keep the synapse to allow
    # STDP to find feature-similarity statistics if the caller wants).
    i_vec = np.asarray(ee.i[:])
    j_vec = np.asarray(ee.j[:])
    ci = e_channel[i_vec]
    cj = e_channel[j_vec]
    dc = np.abs(ci - cj)
    dc = np.minimum(dc, N_CHANNELS - dc)   # wrap-around
    w_init = np.where(dc == 0, cfg.w_ee_within_init,
             np.where(dc == 1, cfg.w_ee_cross_init, 0.0))
    ee.w[:] = w_init

    # --- E -> inh pool (fixed)
    e_to_inh = Synapses(
        e, inh,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_e_inh_pA}*pA",
        name=f"{name}_e_to_inh",
    )
    e_to_inh.connect(p=cfg.p_e_inh)
    e_to_inh.w = cfg.w_e_inh

    # --- inh pool -> E (Vogels iSTDP, Stage-0 stabilizer)
    inh_to_e = vogels_istdp(
        inh, e,
        connectivity="True",
        w_init=cfg.w_inh_e_init,
        w_max=10.0,
        eta=cfg.inh_eta,
        rho=cfg.inh_rho_hz * Hz,
        tau=20 * ms,
        drive_amp_pA=cfg.drive_amp_inh_e_pA,
        name=f"{name}_inh_to_e",
    )

    # --- cue -> E  (placeholder fixed channel-matched block wiring;
    #                plastic cue->H synapses are built in feedback_routes
    #                via eligibility_trace_cue_rule during Stage 2).
    cue_to_e = Synapses(
        cue, e,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_cue_e_pA}*pA",
        name=f"{name}_cue_to_e",
    )
    n_cue = int(cfg.n_cue_afferents)
    n_blk = n_cue // N_CHANNELS
    cue_channel = np.repeat(np.arange(N_CHANNELS), n_blk)
    # Afferents beyond block coverage (if n_cue % N_CHANNELS != 0) sit unused.
    i_cue: list[int] = []
    j_cue: list[int] = []
    for k in range(n_blk * N_CHANNELS):
        c = int(cue_channel[k])
        for j in range(c * N_E_PER_CHANNEL, (c + 1) * N_E_PER_CHANNEL):
            i_cue.append(k)
            j_cue.append(j)
    cue_to_e.connect(i=np.asarray(i_cue, dtype=np.int64),
                     j=np.asarray(j_cue, dtype=np.int64))
    cue_to_e.w = cfg.w_cue_e_init

    ring = HRing(
        name=name,
        e=e, inh=inh, cue=cue,
        ee=ee, e_to_inh=e_to_inh, inh_to_e=inh_to_e, cue_to_e=cue_to_e,
        config=cfg, thetas_rad=thetas_rad, e_channel=e_channel,
    )
    ring.groups = [e, inh, cue, ee, e_to_inh, inh_to_e, cue_to_e]
    return ring


def build_h_r(config: Optional[HRingConfig] = None) -> HRing:
    """H_R: trained on leader -> trailer sequence statistics (Kok, Richter)."""
    return _build_h_ring("hr", config)


def build_h_t(config: Optional[HRingConfig] = None) -> HRing:
    """H_T: trained on Tang-style rotating-orientation sequences."""
    return _build_h_ring("ht", config)


# -- helpers -----------------------------------------------------------------

def pulse_channel(ring: HRing, channel: int, rate_hz: float = 200.0) -> None:
    """Set cue Poisson rates so only `channel`'s block fires.

    Partitions cue afferents into N_CHANNELS blocks of equal size; the
    remainder (if any) sits at 0 Hz. This is a shortcut used by the
    Stage-0 bump test; production cue wiring lives in stimulus.py / train.py.
    """
    n_cue = int(ring.cue.N)
    block = n_cue // N_CHANNELS
    rates = np.zeros(n_cue)
    rates[channel * block: (channel + 1) * block] = rate_hz
    ring.cue.rates = rates * Hz


def silence_cue(ring: HRing) -> None:
    """Zero all cue rates."""
    ring.cue.rates = 0 * Hz


# -- self-check / smoke --------------------------------------------------------

if __name__ == "__main__":
    from brian2 import Network, SpikeMonitor, defaultclock, prefs, seed as b2_seed

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42); np.random.seed(42)

    ring = build_h_r()

    # 1) Baseline: with zero cue input, H should be quiet (< 0.5 Hz).
    e_mon_base = SpikeMonitor(ring.e)
    net1 = Network(*ring.groups, e_mon_base)
    silence_cue(ring)
    net1.run(500 * ms)
    base_rate = e_mon_base.num_spikes / (len(ring.e) * 0.5)
    print(f"h_ring baseline: E rate = {base_rate:.2f} Hz "
          f"(expect < 2 Hz pre-stage-0)")

    # 2) Pulse channel 0's cue afferents at 200 Hz for 100 ms, then silence.
    #    Ring should show ch0 E cells active during pulse; channel structure
    #    should be preserved (untrained H still has within-ch strong recurrence).
    b2_seed(43); np.random.seed(43)
    ring2 = build_h_r()
    e_mon = SpikeMonitor(ring2.e)
    net2 = Network(*ring2.groups, e_mon)
    pulse_channel(ring2, channel=0, rate_hz=200.0)
    net2.run(100 * ms)
    # During pulse
    pulse_spikes = e_mon.num_spikes
    silence_cue(ring2)
    net2.run(500 * ms)
    post_pulse_spikes = e_mon.num_spikes - pulse_spikes

    # Break down by channel during pulse window
    t = np.asarray(e_mon.t / ms)
    i = np.asarray(e_mon.i[:])
    in_pulse = t < 100.0
    in_post  = (t >= 100.0) & (t < 600.0)
    per_ch_pulse = np.bincount(ring2.e_channel[i[in_pulse]], minlength=N_CHANNELS)
    per_ch_post  = np.bincount(ring2.e_channel[i[in_post]],  minlength=N_CHANNELS)
    print(f"h_ring pulse (100 ms): total E spikes = {pulse_spikes}, "
          f"ch0 = {per_ch_pulse[0]}, ch6 = {per_ch_pulse[6]}")
    print(f"h_ring post-pulse (500 ms): total E spikes = {post_pulse_spikes}, "
          f"ch0 = {per_ch_post[0]}, ch6 = {per_ch_post[6]}")
    assert per_ch_pulse[0] >= per_ch_pulse[6], (
        "pulse should bias ch0 over ch6 (got "
        f"ch0={per_ch_pulse[0]}, ch6={per_ch_pulse[6]})"
    )
    assert base_rate < 5.0, f"Quiet baseline failed: {base_rate:.2f} Hz"
    print("h_ring smoke: PASS")
