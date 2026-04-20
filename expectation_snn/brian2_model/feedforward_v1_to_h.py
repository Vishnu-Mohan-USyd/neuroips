"""Fixed-weight bottom-up V1 -> H feedforward (Sprint 5.5, task #31).

Why this module exists
----------------------
Sprint 5b discovered that without a bottom-up driver into H, every primary
assay was r-invariant: H rings emit ~0 spikes during all measurement
windows, so `g_direct * H_spikes == g_SOM * H_spikes == 0` and the H -> V1
balance ratio cannot influence V1. This module adds the missing afferent.

Topology
--------
V1_E[channel c] -> H_E[channel c'] feature-matched Gaussian over channel
distance, with sigma = `sigma_channels` (default 1.0 = 15 deg). Identical
kernel to feedback_routes (same wrap-around minimum distance), so V1 and
H share the same orientation labelling.

The pathway is built per H ring (one builder call -> one Synapses bank).
The assay-runtime caller invokes it once per assay with the relevant ring
(H_R for Kok / Richter, H_T for Tang).

Drive port
----------
V1_E spikes deposit into H_E's `I_e_post` (AMPA-like, tau_e=5 ms). NMDA
co-release is reserved for the *recurrent* H E->E pathway (Wang 2001
attractor) and is intentionally NOT used here -- this is a feedforward
afferent, not a recurrent excitation.

Assay-time only
---------------
Stage-2 cue learning relied on H being driven *only* by the cue, with no
V1 -> H crosstalk; otherwise the cue->H eligibility trace would lock onto
V1's grating tuning instead of the cue identity. To preserve that
isolation, this pathway is attached only inside the assay-runtime frozen
network builder (option ii in the task #31 spec). Training paths
(stage_0/1/2) build their own networks and are unchanged.

Calibration
-----------
At grating drive (V1_E ~4 Hz sustained at the matched channel), the
per-spike drive amplitude `drive_amp_v1_to_h_pA` is sized so the H bump
reaches ~10-30 Hz - enough to engage H -> V1 feedback at non-trivial
balance ratios, but well below attractor saturation. The default value
DEFAULT_DRIVE_AMP_V1_TO_H_PA is selected from a sweep done in
`validation/validate_v1_to_h.py` Assay 4.

References
----------
- Felleman DJ, Van Essen DC (1991) Cereb Cortex 1:1 - layered cortical
  feedforward target conventions.
- Stratford KJ et al. (1996) Nature 382:258 - feature-matched layer-4
  feedforward to layer 2/3.
- Wang X-J (2001) Trends Neurosci 24:455 - recurrent NMDA reserved for
  intra-area persistence, not feedforward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brian2 import (
    Synapses,
    pA,
)

from .h_ring import HRing, N_CHANNELS as H_N_CHANNELS
from .v1_ring import (
    V1Ring,
    N_CHANNELS as V1_N_CHANNELS,
)


# -- constants --------------------------------------------------------------

# Per-spike AMPA drive amplitude (pA) deposited into H_E's I_e on each
# pre (V1_E) spike. AMPA EPSC magnitude scale.
DEFAULT_DRIVE_AMP_V1_TO_H_PA = 80.0

# Topology: Gaussian over channel distance. sigma=1 channel = 15 deg.
DEFAULT_SIGMA_CHANNELS = 1.0
# Drop kernel tail below this fraction of peak to keep connectivity bounded.
DEFAULT_CONNECTIVITY_FLOOR = 1e-3
# Default scalar gain on the row-normalized kernel. Calibrated via
# `validation/validate_v1_to_h.py` Assay 4 (with Stage-0 V1 + Stage-1 H_R
# loaded): at full-contrast grating (matched-channel V1_E ~ 22 Hz, mean
# population V1_E ~ 4 Hz), this gain places the matched-channel H_E
# bump at ~30 Hz - top of the 10-30 Hz "engage feedback, sub-attractor"
# target band. Dropping below ~1.0 silences H; pushing above ~2.0 risks
# attractor lock-in.
DEFAULT_G_V1_TO_H = 1.5


# -- config -----------------------------------------------------------------

@dataclass
class V1ToHConfig:
    """Fixed-weight V1 -> H feedforward configuration.

    Parameters
    ----------
    g_v1_to_h : float
        Scalar gain on the feedforward route. Multiplies the
        Gaussian-kernel weight (which sums to 1 per source channel).
    drive_amp_v1_to_h_pA : float
        Per-spike current deposited into `I_e_post` on H_E.
    sigma_channels : float
        Std. dev. of Gaussian feature-matched topology, in channels.
        Default 1.0 (15 deg).
    connectivity_floor : float
        Relative kernel weight below which synapses are dropped
        (determinism, keeps connection counts bounded).
    """

    g_v1_to_h: float = DEFAULT_G_V1_TO_H
    drive_amp_v1_to_h_pA: float = DEFAULT_DRIVE_AMP_V1_TO_H_PA
    sigma_channels: float = DEFAULT_SIGMA_CHANNELS
    connectivity_floor: float = DEFAULT_CONNECTIVITY_FLOOR


# -- container --------------------------------------------------------------

@dataclass
class V1ToH:
    """Container returned by `build_v1_to_h_feedforward`.

    Attributes
    ----------
    v1_to_he : Synapses
        V1_E -> H_E, fixed-weight, AMPA-only.
    config : V1ToHConfig
        Echoed-back config so the caller can audit defaults.
    g_v1_to_h : float
        Resolved scalar gain (echoes config; kept as a top-level attribute
        for parity with FeedbackRoutes).
    kernel : np.ndarray
        (N_CHANNELS, N_CHANNELS) row-normalized Gaussian kernel.
    kernel_w : np.ndarray
        Per-synapse kernel weight pre-scaling (so callers can re-scale via
        `set_v1_to_h_gain` analogous to `feedback_routes.set_balance`).
    groups : list[Synapses]
        Brian2 objects to splat into Network(...).
    """
    v1_to_he: Synapses
    config: V1ToHConfig
    g_v1_to_h: float
    kernel: np.ndarray = field(default_factory=lambda: np.array([]))
    kernel_w: np.ndarray = field(default_factory=lambda: np.array([]))
    groups: List[object] = field(default_factory=list)

    def set_active(self, active: bool) -> None:
        """Enable / silence the V1 -> H pathway in place.

        Sprint 5c context_only mode: at probe onset the assay loop calls
        ``bundle.v1_to_h.set_active(False)`` to remove the same-trial
        amplifier; at ITI it restores via ``set_active(True)``. Avoids a
        Network rebuild between trials by mutating the synapse weight
        vector directly (same pattern as :func:`set_v1_to_h_gain`).

        ``g_v1_to_h`` is unchanged across toggles, so re-enabling restores
        the calibrated drive amplitude exactly.
        """
        if active:
            self.v1_to_he.w[:] = self.kernel_w * self.g_v1_to_h
        else:
            self.v1_to_he.w[:] = 0.0


# -- builder ----------------------------------------------------------------

def build_v1_to_h_feedforward(
    v1_ring: V1Ring,
    h_ring: HRing,
    config: Optional[V1ToHConfig] = None,
    name_prefix: str = "ff_v1h",
) -> V1ToH:
    """Construct V1_E -> H_E fixed-weight feedforward synapses.

    Requires that V1 and H rings share the same channel layout (12 chs
    at 15 deg spacing). Asserts this explicitly.

    Parameters
    ----------
    v1_ring : V1Ring
        Source ring. `v1_ring.e` and `v1_ring.e_channel` are read.
    h_ring : HRing
        Target ring (H_R or H_T). `h_ring.e` and `h_ring.e_channel`
        are read.
    config : V1ToHConfig, optional
        Feedforward params. Defaults to g_v1_to_h=1.0, drive=60 pA,
        sigma=1.0 channels.
    name_prefix : str
        Used to namespace the Brian2 Synapse object name.

    Returns
    -------
    V1ToH
        Container with the Synapses + groups list ready for Network(...).
    """
    cfg = config or V1ToHConfig()

    assert V1_N_CHANNELS == H_N_CHANNELS, (
        f"V1 and H ring channel counts must match "
        f"(got V1={V1_N_CHANNELS}, H={H_N_CHANNELS})."
    )
    assert len(v1_ring.e_channel) == len(v1_ring.e), (
        "v1_ring.e_channel length must match v1_ring.e.N"
    )
    assert len(h_ring.e_channel) == len(h_ring.e), (
        "h_ring.e_channel length must match h_ring.e.N"
    )

    # Pre-compute Gaussian kernel over channel distance (wrap-around).
    # Identical structure to feedback_routes.build_feedback_routes, so
    # that V1->H and H->V1 use the same orientation alignment.
    ch = np.arange(V1_N_CHANNELS)
    d = np.abs(ch[:, None] - ch[None, :])
    d = np.minimum(d, V1_N_CHANNELS - d).astype(np.float64)
    kernel = np.exp(-0.5 * (d / cfg.sigma_channels) ** 2)
    kernel[kernel < cfg.connectivity_floor] = 0.0
    # Row-normalize: total deposit per V1_E spike is invariant to sigma.
    row_sum = kernel.sum(axis=1, keepdims=True)
    assert np.all(row_sum > 0), "All-zero row in kernel -- sigma too small."
    kernel /= row_sum

    # --- V1_E -> H_E (AMPA-only, fixed weight) -----------------------
    v1_to_he = Synapses(
        v1_ring.e, h_ring.e,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_v1_to_h_pA}*pA",
        name=f"{name_prefix}_{h_ring.name}",
    )
    i_src: list[int] = []
    j_tgt: list[int] = []
    w_src: list[float] = []
    for ci in range(V1_N_CHANNELS):
        v1_src_idx = np.flatnonzero(v1_ring.e_channel == ci)
        for cj in range(V1_N_CHANNELS):
            k = float(kernel[ci, cj])
            if k == 0.0:
                continue
            h_tgt_idx = np.flatnonzero(h_ring.e_channel == cj)
            for i_v in v1_src_idx:
                for j_h in h_tgt_idx:
                    i_src.append(int(i_v))
                    j_tgt.append(int(j_h))
                    w_src.append(k)
    v1_to_he.connect(
        i=np.asarray(i_src, dtype=np.int64),
        j=np.asarray(j_tgt, dtype=np.int64),
    )
    v1_to_he.w[:] = np.asarray(w_src, dtype=np.float64) * cfg.g_v1_to_h

    ff = V1ToH(
        v1_to_he=v1_to_he,
        config=cfg,
        g_v1_to_h=float(cfg.g_v1_to_h),
        kernel=kernel,
        kernel_w=np.asarray(w_src, dtype=np.float64),
    )
    ff.groups = [v1_to_he]
    return ff


def set_v1_to_h_gain(ff: V1ToH, g_v1_to_h: float) -> None:
    """Re-scale the V1 -> H gain in-place (no Network rebuild).

    Useful for calibration sweeps: build once, vary g_v1_to_h between
    runs without paying the Synapses-construction cost.

    Parameters
    ----------
    ff : V1ToH
        Container from `build_v1_to_h_feedforward`.
    g_v1_to_h : float
        New scalar gain (multiplies the row-normalized kernel weights).
    """
    ff.v1_to_he.w[:] = ff.kernel_w * float(g_v1_to_h)
    ff.g_v1_to_h = float(g_v1_to_h)
    ff.config.g_v1_to_h = float(g_v1_to_h)


# -- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    from brian2 import (
        Network, SpikeMonitor, defaultclock, prefs, ms,
        seed as b2_seed,
    )
    from .h_ring import build_h_r, silence_cue
    from .v1_ring import build_v1_ring, set_stimulus

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42)
    np.random.seed(42)

    h = build_h_r()
    v = build_v1_ring()
    ff = build_v1_to_h_feedforward(v, h, V1ToHConfig(g_v1_to_h=1.0))

    # Drive V1 with grating at ch0; expect H_R[ch0] to fire above baseline.
    set_stimulus(v, theta_rad=0.0, contrast=1.0)
    silence_cue(h)
    h_mon = SpikeMonitor(h.e)
    v_mon = SpikeMonitor(v.e)
    net = Network(*h.groups, *v.groups, *ff.groups, h_mon, v_mon)
    net.run(500 * ms)

    h_idx = np.asarray(h_mon.i[:])
    h_per_ch = np.bincount(h.e_channel[h_idx], minlength=H_N_CHANNELS)
    v_idx = np.asarray(v_mon.i[:])
    v_per_ch = np.bincount(v.e_channel[v_idx], minlength=V1_N_CHANNELS)
    print(f"feedforward_v1_to_h smoke: V1 grating ch0 -> H per-ch counts:")
    for c in range(H_N_CHANNELS):
        mark = "  <- driven" if c == 0 else ""
        print(f"  ch{c:02d}  V1={v_per_ch[c]:4d}  H={h_per_ch[c]:4d}{mark}")
    assert h_per_ch[0] >= h_per_ch[6], (
        f"matched ch0 should beat orth ch6: got ch0={h_per_ch[0]}, "
        f"ch6={h_per_ch[6]}"
    )
    print("feedforward_v1_to_h smoke: PASS")
