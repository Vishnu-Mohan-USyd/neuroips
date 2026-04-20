"""Diagnostic H-clamp pathway (Sprint 5d, task #41 step 2).

Why this module exists
----------------------
Sprint 5c meta-review diagnosed that H is acting as a sensory amplifier
driven by V1, rather than a predictive module with a pre-probe prior. To
localise whether the failure is in the H learning/memory module (Case A)
or in the H -> V1 feedback interface (Case B), Sprint 5d needs to bypass
H learning entirely and **externally drive** the "expected" H channel
before a probe window, measuring what V1 does in response.

This module implements that: a fixed-channel Poisson "clamp" of
configurable rate + window, with AMPA drive into H_E's I_e, toggled
in-place via synapse weight zeroing (mirrors
`feedforward_v1_to_h.py` set_active pattern — no Network rebuild between
trials).

Topology
--------
One PoissonGroup of N afferents firing continuously at `clamp_rate_hz`,
all-to-all connected to the N_E_PER_CHANNEL E cells of `target_channel`
only. Weights set to `active_w` when `set_active(True)`, 0 otherwise.

Drive port
----------
AMPA-like: `I_e_post += w * drive_amp_pA * pA` on each pre-spike. No
NMDA — this is a diagnostic injector, not a recurrent/learning pathway.

Not attached to training
------------------------
Only the assay-time frozen-network builder wires this in. Stage-0/1/2
training paths construct their own networks and are unchanged.

References
----------
- Cossell L et al. (2015) Nature 518:399 — functional targeting of
  pyramidal dendrites; sanity check for biological feature-matched
  clamp afferents.
- Sprint 5c meta-review `expectation_snn/docs/SPRINT_5C_META_REVIEW.md`
  — decision-tree for failure localisation via D3 H-clamp test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brian2 import (
    Hz,
    pA,
    PoissonGroup,
    Synapses,
)

from .h_ring import (
    HRing,
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER_CHANNEL,
)


# -- constants --------------------------------------------------------------

#: Default Poisson afferent pool size per clamp. 32 gives a well-averaged
#: event stream at 200 Hz without lock-step artifacts while keeping the
#: synapse count small (32 * 16 = 512).
DEFAULT_N_CLAMP_AFFERENTS = 32

#: Default per-spike AMPA drive amplitude (pA) deposited into target H_E's
#: `I_e_post`. Matches order-of-magnitude of Stage-2 cue drive (20 pA) so
#: 200 Hz clamp comfortably elevates H_E[target] above ~20 Hz while
#: remaining sub-attractor.
DEFAULT_CLAMP_DRIVE_AMP_PA = 20.0


# -- config -----------------------------------------------------------------

@dataclass
class HClampConfig:
    """Diagnostic H-clamp configuration.

    Parameters
    ----------
    target_channel : int
        H_E channel index [0, N_CHANNELS) to inject into.
    clamp_rate_hz : float
        Continuous Poisson rate of the clamp afferent pool (Hz). The
        PoissonGroup runs at this rate for the lifetime of the bundle;
        gating happens via `set_active` on the Synapses weights.
    window_start_ms, window_end_ms : float
        *Metadata only* — informs the assay runner when to call
        `set_active(True)` / `set_active(False)`. Not enforced by this
        module (Brian2 `network_operation` or explicit `run` segment
        sequencing is the caller's job).
    n_afferents : int
        Size of the Poisson afferent pool.
    drive_amp_pA : float
        Per-spike AMPA current into I_e_post.
    """

    target_channel: int = 0
    clamp_rate_hz: float = 200.0
    window_start_ms: float = 0.0
    window_end_ms: float = 0.0
    n_afferents: int = DEFAULT_N_CLAMP_AFFERENTS
    drive_amp_pA: float = DEFAULT_CLAMP_DRIVE_AMP_PA


# -- container --------------------------------------------------------------

@dataclass
class HClamp:
    """Container returned by :func:`build_h_clamp`.

    Attributes
    ----------
    clamp : PoissonGroup
        Afferent pool firing at `config.clamp_rate_hz`.
    clamp_to_he : Synapses
        Poisson -> H_E[target_channel] synapses, weight-gated AMPA drive.
    config : HClampConfig
        Echoed-back config.
    active_w : float
        Per-synapse weight applied when `set_active(True)`.
    groups : list
        Brian2 objects to splat into Network(...).
    """
    clamp: PoissonGroup
    clamp_to_he: Synapses
    config: HClampConfig
    active_w: float
    groups: List[object] = field(default_factory=list)

    def set_active(self, active: bool) -> None:
        """Enable / silence the H-clamp in place.

        Mirrors :meth:`V1ToH.set_active` — weights flip between
        `active_w` and 0 to toggle drive without rebuilding the Network.
        The PoissonGroup keeps firing at `clamp_rate_hz`; only the
        deposit amplitude changes.
        """
        if active:
            self.clamp_to_he.w[:] = self.active_w
        else:
            self.clamp_to_he.w[:] = 0.0


# -- builder ----------------------------------------------------------------

def build_h_clamp(
    h_ring: HRing,
    config: Optional[HClampConfig] = None,
    name_prefix: str = "h_clamp",
) -> HClamp:
    """Construct a diagnostic Poisson clamp targeting one H_E channel.

    Parameters
    ----------
    h_ring : HRing
        Target ring (H_R or H_T). Only ``h_ring.e`` and
        ``h_ring.e_channel`` are read.
    config : HClampConfig, optional
        Clamp parameters. Defaults to `target_channel=0`,
        `clamp_rate_hz=200`, `n_afferents=32`, `drive_amp_pA=20`.
    name_prefix : str
        Brian2 object name namespace.

    Returns
    -------
    HClamp
        Container with PoissonGroup + Synapses; start inactive
        (weights=0). Caller invokes :meth:`HClamp.set_active(True)` to
        turn on.

    Notes
    -----
    The clamp starts **inactive**: `set_active(True)` must be called by
    the assay runner at the desired window start. This matches the
    "construct at build time with weights set to 0" pattern from the
    Sprint 5d-infra dispatch.
    """
    cfg = config or HClampConfig()

    if not (0 <= cfg.target_channel < H_N_CHANNELS):
        raise ValueError(
            f"target_channel must be in [0, {H_N_CHANNELS}), got "
            f"{cfg.target_channel!r}"
        )
    if cfg.n_afferents < 1:
        raise ValueError(f"n_afferents must be >= 1, got {cfg.n_afferents}")
    if cfg.clamp_rate_hz < 0.0:
        raise ValueError(
            f"clamp_rate_hz must be >= 0, got {cfg.clamp_rate_hz}"
        )
    assert len(h_ring.e_channel) == len(h_ring.e), (
        "h_ring.e_channel length must match h_ring.e.N"
    )

    clamp = PoissonGroup(
        int(cfg.n_afferents),
        rates=float(cfg.clamp_rate_hz) * Hz,
        name=f"{name_prefix}_{h_ring.name}_poisson",
    )

    clamp_to_he = Synapses(
        clamp, h_ring.e,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_pA}*pA",
        name=f"{name_prefix}_{h_ring.name}_syn",
    )

    e_tgt_idx = np.flatnonzero(h_ring.e_channel == cfg.target_channel)
    if e_tgt_idx.size == 0:
        raise RuntimeError(
            f"No H_E cells found for target_channel={cfg.target_channel} "
            f"(h_ring.e_channel has values "
            f"{sorted(set(h_ring.e_channel.tolist()))})"
        )

    n_tgt = int(e_tgt_idx.size)
    n_aff = int(cfg.n_afferents)
    i_src = np.repeat(np.arange(n_aff, dtype=np.int64), n_tgt)
    j_tgt = np.tile(e_tgt_idx.astype(np.int64), n_aff)
    clamp_to_he.connect(i=i_src, j=j_tgt)

    # Start inactive (matches step-2 spec: "construct at build time with
    # weights set to 0, toggle active during the window only").
    clamp_to_he.w[:] = 0.0

    obj = HClamp(
        clamp=clamp,
        clamp_to_he=clamp_to_he,
        config=cfg,
        active_w=1.0,
        groups=[clamp, clamp_to_he],
    )
    return obj


# -- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    from brian2 import (
        Network, SpikeMonitor, defaultclock, prefs, ms,
        seed as b2_seed,
    )
    from .h_ring import build_h_r, silence_cue

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42)
    np.random.seed(42)

    h = build_h_r()
    cfg = HClampConfig(target_channel=0, clamp_rate_hz=200.0)
    clamp = build_h_clamp(h, cfg)
    silence_cue(h)

    h_mon = SpikeMonitor(h.e)
    net = Network(*h.groups, *clamp.groups, h_mon)

    # Phase A: 200 ms silent (clamp inactive)
    net.run(200 * ms)

    # Phase B: 200 ms clamp active
    clamp.set_active(True)
    net.run(200 * ms)

    # Phase C: 200 ms silent again
    clamp.set_active(False)
    net.run(200 * ms)

    i = np.asarray(h_mon.i[:])
    t = np.asarray(h_mon.t / ms)
    ch = h.e_channel[i]
    n_per_ch = H_N_E_PER_CHANNEL

    def _rate(win0, win1):
        m = (t >= win0) & (t < win1)
        return float(np.sum(ch[m] == cfg.target_channel)) / (n_per_ch * (win1 - win0) * 1e-3)

    r_pre  = _rate(0.0, 200.0)
    r_on   = _rate(200.0, 400.0)
    r_post = _rate(400.0, 600.0)
    print(f"h_clamp smoke: ch{cfg.target_channel} "
          f"pre={r_pre:.2f} Hz  on={r_on:.2f} Hz  post={r_post:.2f} Hz")
    assert r_on > 20.0, f"clamp should drive ch{cfg.target_channel} > 20 Hz, got {r_on:.2f}"
    assert r_pre < 2.0, f"pre-window should be < 2 Hz, got {r_pre:.2f}"
    print("h_clamp smoke: PASS")
