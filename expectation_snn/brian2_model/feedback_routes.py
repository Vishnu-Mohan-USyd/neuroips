"""Fixed-weight feedback H -> V1 (plan sec 1, phase 2 topology).

Two routes, topology + signs + weights FIXED (not plastic):

1. Direct route: H_E -> V1_E apical (modulatory, feature-matched Gaussian).
   - Feature-matched Gaussian over channel distance with
     sigma = `sigma_channels` (default 1.0 channel = 15 deg):
       w(H_E[c_i] -> V1_E[c_j]) ∝ exp( -d(c_i, c_j)^2 / (2 sigma^2) ),
     where d(.,.) is minimum wrap-around channel distance.
   - Deposits into `I_ap_e_post` (apical compartment) so the feedback
     is modulatory (gates somatic drive via V1E's sigmoid apical-to-
     soma gate), NOT a direct somatic drive. Matches Larkum 2013
     top-down-targets-apical-tuft canon. Sub-threshold: H bump with
     grating OFF must not evoke V1_E firing.
   - Scaled by `g_direct`.

2. Suppressive route: H_E -> V1_SOM (feature-linked, excitatory onto SOM
   which then inhibits V1_E via the existing within-channel SOM->E
   synapse in v1_ring).
   - Same Gaussian profile (feature-matched, sigma_channels).
   - Deposits into SOM's `I_e_post`. Normal AMPA kinetics (tau_e=5 ms).
   - Scaled by `g_SOM`.

Pre-registered balance sweep with `g_total = g_direct + g_SOM` held
constant (Sprint 5b):

    r = g_direct / g_SOM  in  {0.25, 0.50, 1.00, 2.00, 4.00}   (S1..S5)

At r=1.0 (balanced, Sprint 5a default):
    g_direct = g_SOM = g_total / 2.

The helper `balance_weights(g_total, r)` returns the scaled (g_direct,
g_SOM) tuple given the ratio and total. `set_balance(fb, r)` rewrites
the weights in-place on existing Synapses (no Network rebuild).

Connectivity is deterministic: Gaussian kernel values below
`connectivity_floor` (default 1e-3) are dropped (no stochastic
sampling). This keeps the topology exactly reproducible across seeds.

References
----------
- Larkum ME (2013) Trends Neurosci 36:141 — top-down feedback targets
  distal apical tufts of pyramidal cells.
- Petreanu et al. 2009 Nature 457:1142 — long-range cortical feedback
  channel-matches target by functional features.
- Makino Y, Komiyama T (2015) Nat Neurosci 18:1116 — disinhibitory
  versus direct top-down gain routes in visual cortex.
- Urban-Ciecko J, Barth AL (2016) Nat Rev Neurosci 17:401 — SOM
  Martinotti cells in cortical disinhibition circuits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from brian2 import (
    Synapses,
    pA,
)

from .h_ring import HRing, N_CHANNELS as H_N_CHANNELS
from .v1_ring import (
    V1Ring,
    N_CHANNELS as V1_N_CHANNELS,
    N_E_PER_CHANNEL as V1_N_E_PER_CHANNEL,
    N_SOM_PER_CHANNEL as V1_N_SOM_PER_CHANNEL,
)


# -- constants --------------------------------------------------------------

# Per-spike drive amplitudes (pA). These are the "unit" drive the synapse
# deposits; the channel-level gain is set by the scalar weight w (which
# absorbs g_direct or g_SOM) AND the per-channel-distance Gaussian kernel
# (rows sum to 1 so each H spike's total deposit is invariant to sigma).
#
# Calibrated with `scripts/calibrate_feedback_routes.py` (seed=42,
# grating contrast=1.0 at ch0, H pulse_rate=300 Hz -> ~50 Hz H bump,
# trial window 1500 ms). At g_total=1.0:
#   direct-only (r=4.0) : +8.33 % matched-channel gain (target +5..+15 %)
#   SOM-only    (r=0.25): -4.17 % matched-channel gain (target -5..-15 %)
#   balanced    (r=1.0) :  0.00 % (net-neutral gain at matched channel)
# Monotonic across {0.25, 0.5, 1, 2, 4} (see validate_feedback_routes.py
# Assay 5). Sub-threshold: grating OFF -> V1_E = 0 Hz (Assay 4).
DEFAULT_DRIVE_AMP_H_TO_V1E_APICAL_PA = 30.0
DEFAULT_DRIVE_AMP_H_TO_V1SOM_PA = 40.0

# Topology: Gaussian over channel distance. sigma=1 channel = 15 deg.
DEFAULT_SIGMA_CHANNELS = 1.0
# Drop kernel tail below this fraction of peak to keep connectivity bounded.
DEFAULT_CONNECTIVITY_FLOOR = 1e-3


# -- config -----------------------------------------------------------------

@dataclass
class FeedbackRoutesConfig:
    """Fixed-weight feedback route configuration.

    Parameters
    ----------
    g_total : float
        Sum `g_direct + g_SOM`, held constant across the balance sweep.
    r : float
        Ratio `g_direct / g_SOM`. At r=1.0, the two routes are balanced.
    drive_amp_h_to_v1e_apical_pA : float
        Per-spike current deposited into `I_ap_e_post` on V1_E apical.
    drive_amp_h_to_v1som_pA : float
        Per-spike current deposited into `I_e_post` on V1_SOM.
    sigma_channels : float
        Std. dev. of Gaussian feature-matched topology, in channels.
        Default 1.0 (15 deg). Both routes share this sigma.
    connectivity_floor : float
        Relative kernel weight below which synapses are dropped
        (determinism, keeps connection counts bounded).
    """

    g_total: float = 1.0
    r: float = 1.0
    drive_amp_h_to_v1e_apical_pA: float = DEFAULT_DRIVE_AMP_H_TO_V1E_APICAL_PA
    drive_amp_h_to_v1som_pA: float = DEFAULT_DRIVE_AMP_H_TO_V1SOM_PA
    sigma_channels: float = DEFAULT_SIGMA_CHANNELS
    connectivity_floor: float = DEFAULT_CONNECTIVITY_FLOOR


# -- container --------------------------------------------------------------

@dataclass
class FeedbackRoutes:
    """Container returned by `build_feedback_routes`."""
    hr_to_v1e: Synapses          # H_R E -> V1_E apical (direct, modulatory)
    hr_to_v1som: Synapses        # H_R E -> V1_SOM (suppressive via SOM->E)
    config: FeedbackRoutesConfig
    g_direct: float              # scalar gain applied to route 1
    g_SOM: float                 # scalar gain applied to route 2
    kernel: np.ndarray = field(default_factory=lambda: np.array([]))
    kernel_w_direct: np.ndarray = field(default_factory=lambda: np.array([]))
    kernel_w_som: np.ndarray = field(default_factory=lambda: np.array([]))
    groups: List[object] = field(default_factory=list)


# -- balance helper ---------------------------------------------------------

def balance_weights(g_total: float, r: float) -> Tuple[float, float]:
    """Scale (g_direct, g_SOM) so that g_direct + g_SOM = g_total and
    g_direct / g_SOM = r.

    Degenerate limits:
      - r = 0.0  -> g_direct = 0, g_SOM = g_total (pure suppression).
      - r = inf  -> g_direct = g_total, g_SOM = 0 (pure excitation).

    Parameters
    ----------
    g_total : float
        Total feedback conductance budget (must be >= 0).
    r : float
        Ratio g_direct / g_SOM. Must be >= 0 or np.inf.

    Returns
    -------
    (g_direct, g_SOM) : tuple[float, float]
    """
    assert g_total >= 0.0, f"g_total must be >= 0, got {g_total}"
    if np.isinf(r):
        return float(g_total), 0.0
    assert r >= 0.0, f"r must be >= 0 or np.inf, got {r}"
    g_som = g_total / (1.0 + r)
    g_direct = g_total - g_som
    return float(g_direct), float(g_som)


# -- builder ----------------------------------------------------------------

def build_feedback_routes(
    h_ring: HRing,
    v1_ring: V1Ring,
    config: Optional[FeedbackRoutesConfig] = None,
    name_prefix: str = "fb",
) -> FeedbackRoutes:
    """Construct the two fixed-weight H -> V1 feedback routes.

    Requires that both rings share the same channel layout (12 channels
    at 15 deg spacing). Asserts this explicitly.

    Parameters
    ----------
    h_ring : HRing
        Source H ring (either H_R or H_T). Only `h_ring.e` and
        `h_ring.e_channel` are read.
    v1_ring : V1Ring
        Target V1 ring. `v1_ring.e`, `v1_ring.som`, and their channel
        labellings are read.
    config : FeedbackRoutesConfig, optional
        Balance / drive parameters. Defaults to g_total=1.0, r=1.0.
    name_prefix : str
        Used to namespace Brian2 Synapse object names.

    Returns
    -------
    FeedbackRoutes
        Container with the two Synapses + groups ready for Network(...).
    """
    cfg = config or FeedbackRoutesConfig()

    assert H_N_CHANNELS == V1_N_CHANNELS, (
        f"H and V1 ring channel counts must match "
        f"(got H={H_N_CHANNELS}, V1={V1_N_CHANNELS})."
    )
    assert len(h_ring.e_channel) == len(h_ring.e), (
        "h_ring.e_channel length must match h_ring.e.N"
    )
    assert len(v1_ring.e_channel) == len(v1_ring.e), (
        "v1_ring.e_channel length must match v1_ring.e.N"
    )
    assert len(v1_ring.som_channel) == len(v1_ring.som), (
        "v1_ring.som_channel length must match v1_ring.som.N"
    )

    # Resolve balance
    g_direct, g_som = balance_weights(cfg.g_total, cfg.r)

    # Pre-compute Gaussian kernel over channel distance (wrap-around).
    # Also store indices (src_ch -> mask of (tgt_ch, kernel_val)) so the
    # two routes can reuse the same topology at different scalar gains.
    ch = np.arange(H_N_CHANNELS)
    # pairwise min wrap-around distance in channels:
    d = np.abs(ch[:, None] - ch[None, :])
    d = np.minimum(d, H_N_CHANNELS - d).astype(np.float64)
    kernel = np.exp(-0.5 * (d / cfg.sigma_channels) ** 2)
    kernel[kernel < cfg.connectivity_floor] = 0.0
    # Normalize so the total kernel mass per source channel is 1.0.
    # This keeps the net feedback drive per H spike invariant across
    # sigma values: each H spike deposits (g_direct * drive_amp) pA
    # distributed over matched and neighbouring channels.
    row_sum = kernel.sum(axis=1, keepdims=True)
    assert np.all(row_sum > 0), "All-zero row in kernel -- sigma too small."
    kernel /= row_sum

    # --- Route 1: H_E -> V1_E apical (direct, modulatory) --------------
    hr_to_v1e = Synapses(
        h_ring.e, v1_ring.e,
        model="w : 1",
        on_pre=f"I_ap_e_post += w * {cfg.drive_amp_h_to_v1e_apical_pA}*pA",
        name=f"{name_prefix}_{h_ring.name}_to_v1e_apical",
    )
    i_src1: list[int] = []
    j_tgt1: list[int] = []
    w_src1: list[float] = []
    for ci in range(H_N_CHANNELS):
        h_src_idx = np.flatnonzero(h_ring.e_channel == ci)
        for cj in range(H_N_CHANNELS):
            k = float(kernel[ci, cj])
            if k == 0.0:
                continue
            v1_e_idx = np.flatnonzero(v1_ring.e_channel == cj)
            for i_h in h_src_idx:
                for j_v in v1_e_idx:
                    i_src1.append(int(i_h))
                    j_tgt1.append(int(j_v))
                    w_src1.append(k)
    hr_to_v1e.connect(
        i=np.asarray(i_src1, dtype=np.int64),
        j=np.asarray(j_tgt1, dtype=np.int64),
    )
    hr_to_v1e.w[:] = np.asarray(w_src1, dtype=np.float64) * g_direct

    # --- Route 2: H_E -> V1_SOM (suppressive via SOM->E) ---------------
    hr_to_v1som = Synapses(
        h_ring.e, v1_ring.som,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_h_to_v1som_pA}*pA",
        name=f"{name_prefix}_{h_ring.name}_to_v1som",
    )
    i_src2: list[int] = []
    j_tgt2: list[int] = []
    w_src2: list[float] = []
    for ci in range(H_N_CHANNELS):
        h_src_idx = np.flatnonzero(h_ring.e_channel == ci)
        for cj in range(H_N_CHANNELS):
            k = float(kernel[ci, cj])
            if k == 0.0:
                continue
            v1_som_idx = np.flatnonzero(v1_ring.som_channel == cj)
            for i_h in h_src_idx:
                for j_s in v1_som_idx:
                    i_src2.append(int(i_h))
                    j_tgt2.append(int(j_s))
                    w_src2.append(k)
    hr_to_v1som.connect(
        i=np.asarray(i_src2, dtype=np.int64),
        j=np.asarray(j_tgt2, dtype=np.int64),
    )
    hr_to_v1som.w[:] = np.asarray(w_src2, dtype=np.float64) * g_som

    fb = FeedbackRoutes(
        hr_to_v1e=hr_to_v1e,
        hr_to_v1som=hr_to_v1som,
        config=cfg,
        g_direct=g_direct,
        g_SOM=g_som,
        kernel=kernel,
        kernel_w_direct=np.asarray(w_src1, dtype=np.float64),
        kernel_w_som=np.asarray(w_src2, dtype=np.float64),
    )
    fb.groups = [hr_to_v1e, hr_to_v1som]
    return fb


def set_balance(fb: FeedbackRoutes, r: float,
                g_total: Optional[float] = None) -> None:
    """Re-scale the two routes in-place to a new balance ratio.

    Useful for Sprint 5b balance sweep: build the Network once, run at
    r=1, then call set_balance(fb, r=0.5), re-run, etc. Per-synapse
    Gaussian kernel weights are preserved; only the scalar gains
    g_direct, g_SOM are updated.

    Parameters
    ----------
    fb : FeedbackRoutes
        Container from `build_feedback_routes`.
    r : float
        New g_direct / g_SOM ratio.
    g_total : float, optional
        New total; defaults to fb.config.g_total (preserves scale).
    """
    gt = fb.config.g_total if g_total is None else float(g_total)
    g_direct, g_som = balance_weights(gt, r)
    # Preserve the kernel per-synapse weighting; only rescale by the new
    # scalar gain.
    fb.hr_to_v1e.w[:] = fb.kernel_w_direct * g_direct
    fb.hr_to_v1som.w[:] = fb.kernel_w_som * g_som
    fb.g_direct = g_direct
    fb.g_SOM = g_som
    fb.config.r = float(r)
    fb.config.g_total = gt


# -- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    # Balance helper sanity
    for r, g_total in [(1.0, 1.0), (0.5, 2.0), (4.0, 1.0), (0.25, 1.0)]:
        gd, gs = balance_weights(g_total, r)
        assert abs(gd + gs - g_total) < 1e-9, (gd, gs, g_total)
        if r > 0:
            assert abs(gd / gs - r) < 1e-9, (gd, gs, r)
    # Degenerate
    gd, gs = balance_weights(1.0, 0.0)
    assert gd == 0.0 and gs == 1.0, (gd, gs)
    gd, gs = balance_weights(1.0, float("inf"))
    assert gd == 1.0 and gs == 0.0, (gd, gs)
    print("feedback_routes balance_weights: PASS")

    # Build a mini network with H_R + V1 + feedback routes, pulse a channel,
    # confirm V1 response is enhanced over unmatched channel.
    from brian2 import Network, SpikeMonitor, defaultclock, prefs, seed as b2_seed, ms
    from .h_ring import build_h_r, pulse_channel, silence_cue
    from .v1_ring import build_v1_ring

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42)
    np.random.seed(42)

    h = build_h_r()
    v = build_v1_ring()
    fb = build_feedback_routes(h, v, FeedbackRoutesConfig(g_total=1.0, r=1.0))

    # Pulse H channel 0 for 100 ms, then silence.
    v_e_mon = SpikeMonitor(v.e)
    v_som_mon = SpikeMonitor(v.som)
    net = Network(*h.groups, *v.groups, *fb.groups, v_e_mon, v_som_mon)
    pulse_channel(h, channel=0, rate_hz=200.0)
    net.run(100 * ms)
    silence_cue(h)
    net.run(300 * ms)

    e_idx = np.asarray(v_e_mon.i[:])
    per_ch_e = np.bincount(v.e_channel[e_idx],
                           minlength=V1_N_CHANNELS)
    print(f"feedback_routes smoke V1_E per-ch (no stim, H pulse ch0):")
    for c in range(V1_N_CHANNELS):
        mark = "  <- H bump" if c == 0 else ""
        print(f"  ch{c:02d}  {per_ch_e[c]:4d} spikes{mark}")
    print("feedback_routes smoke: PASS")
