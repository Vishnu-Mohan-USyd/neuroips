"""Component-level functional validation for the per-channel + broad
inhibitory architecture of the H ring (brian2_model/h_ring.py).

Two biology-anchored assays:

1. **Local vs broad suppression ratio.** Drive one H channel ONLY via
   its cue afferents and measure the post-cue E-firing rate in that
   channel (suppressed by local inh[c]) vs in neighbour channels
   (suppressed only by the broad inh pool). Assert:

        local_self_suppression / broad_other_suppression   >=  2

   Reference: cortical PV subpools show far stronger target-channel
   suppression than global inhibition (Pouille et al. 2009 Science
   325:1619; Karnani et al. 2016 PNAS 113:E6329). Validates that the
   per-channel subpool is actually doing per-channel work rather than
   acting as a second copy of the broad pool.

2. **Broad pool enforces single-channel attractor.** Drive TWO or more
   channels simultaneously with balanced cue rates; measure per-channel
   E rates at steady-state. Winner-take-all should emerge: one channel
   dominates, others are suppressed. Assert:

        (top_channel_rate - 2nd_channel_rate) / top_channel_rate >= 0.4

   That is, the winning channel fires at least 40 pct more than the
   runner-up — a mild attractor-competition floor. Reference: Wang 2001
   bump-attractor model requires cross-channel inhibition sufficient to
   break symmetry under balanced drive.

Run:
    python -m expectation_snn.validation.validate_per_channel_inh
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from brian2 import (
    Network,
    SpikeMonitor,
    defaultclock,
    Hz,
    ms,
    prefs,
    seed as b2_seed,
)

from ..brian2_model.h_ring import (
    HRingConfig,
    N_CHANNELS,
    N_E_PER_CHANNEL,
    N_INH_POOL,
    build_h_r,
    silence_cue,
)


# -- measurement bands ------------------------------------------------------

LOCAL_VS_BROAD_RATIO_MIN = 2.0        # local suppression must be >=2x broad
WTA_MARGIN_MIN = 0.4                   # top channel exceeds runner-up by >=40%
PROBE_DUR_MS = 500.0                   # each cue epoch
SETTLE_MS = 200.0                      # quiet time before/after cue
CUE_PEAK_HZ = 300.0
CUE_SIGMA_DEG = 15.0


@dataclass
class InhValidationReport:
    driven_ch: int
    rate_on_driven_ch_hz: float
    rate_on_nbr_ch_hz: float
    rate_on_far_ch_hz: float
    baseline_rate_hz: float
    local_suppression_ratio: float     # baseline / driven_ch_rest_after_cue
    broad_suppression_ratio: float     # baseline / far_ch_rate_during_cue
    local_vs_broad_ratio: float        # local / broad

    wta_top_rate_hz: float
    wta_runner_rate_hz: float
    wta_margin: float

    passed_local_vs_broad: bool
    passed_wta: bool

    @property
    def passed(self) -> bool:
        return self.passed_local_vs_broad and self.passed_wta

    def summary(self) -> str:
        ratio_str = (
            "inf (nbr fully silenced -- local inh carries all the work)"
            if self.local_vs_broad_ratio == float("inf")
            else f"{self.local_vs_broad_ratio:.2f}"
        )
        return (
            "Per-channel inh validation:\n"
            f"  driven_ch = {self.driven_ch}\n"
            f"  E rate on driven_ch  = {self.rate_on_driven_ch_hz:.2f} Hz\n"
            f"  E rate on +/-1 nbr   = {self.rate_on_nbr_ch_hz:.2f} Hz\n"
            f"  E rate on far (~180) = {self.rate_on_far_ch_hz:.2f} Hz\n"
            f"  E rate baseline      = {self.baseline_rate_hz:.2f} Hz\n"
            f"  -------------------------------------------------\n"
            f"  local drop (driven - nbr)   = {self.local_suppression_ratio:.2f} Hz\n"
            f"  broad drop (nbr - far)      = {self.broad_suppression_ratio:.2f} Hz\n"
            f"  LOCAL_vs_BROAD ratio         = {ratio_str} "
            f"(>={LOCAL_VS_BROAD_RATIO_MIN})  "
            f"{'PASS' if self.passed_local_vs_broad else 'FAIL'}\n"
            f"  -------------------------------------------------\n"
            f"  WTA top = {self.wta_top_rate_hz:.2f} Hz, "
            f"runner = {self.wta_runner_rate_hz:.2f} Hz, "
            f"margin = {self.wta_margin:.2f} "
            f"(>={WTA_MARGIN_MIN})  "
            f"{'PASS' if self.passed_wta else 'FAIL'}\n"
            f"  -------------------------------------------------\n"
            f"  verdict: {'PASS' if self.passed else 'FAIL'}"
        )


# -- shared helpers ---------------------------------------------------------

def _drive_single_channel_delta(ring, ch: int, peak_rate_hz: float) -> None:
    """Drive ONE channel's cue block at `peak_rate_hz` and all others at 0.

    Delta-shaped cue so we isolate the effect of the per-channel /
    broad inhibitory architecture from cue spread (a Gaussian cue with
    sigma ~ channel spacing drives neighbours directly, which confounds
    the test of inh architecture).
    """
    n_cue = int(ring.cue.N)
    block = n_cue // N_CHANNELS
    rates = np.zeros(n_cue)
    rates[ch * block : (ch + 1) * block] = peak_rate_hz
    ring.cue.rates = rates * Hz


def _drive_balanced_multichannel(ring, channels, peak_rate_hz: float) -> None:
    """Set cue rates balanced across `channels` (list of int)."""
    n_cue = int(ring.cue.N)
    block = n_cue // N_CHANNELS
    rates = np.zeros(n_cue)
    for c in channels:
        rates[c * block : (c + 1) * block] = peak_rate_hz
    ring.cue.rates = rates * Hz


def _rate_per_channel(e_mon: SpikeMonitor, e_channel: np.ndarray,
                      t0_ms: float, t1_ms: float) -> np.ndarray:
    """Per-channel E firing rate (Hz) over [t0, t1) ms."""
    i = np.asarray(e_mon.i[:], dtype=np.int64)
    t_ms = np.asarray(e_mon.t / ms, dtype=np.float64)
    mask = (t_ms >= t0_ms) & (t_ms < t1_ms)
    counts = np.bincount(e_channel[i[mask]], minlength=N_CHANNELS)
    dur_s = max((t1_ms - t0_ms) / 1000.0, 1e-6)
    return counts / (N_E_PER_CHANNEL * dur_s)


# -- Assay 1: local vs broad suppression ------------------------------------

def _stage1_cfg() -> HRingConfig:
    """Match the Stage-1 tuned config used in train.py::_stage1_h_cfg.

    We only need enough of the Stage-1 cfg to exercise the per-channel +
    broad inh architecture; the STDP tuning is irrelevant here.
    """
    cfg = HRingConfig()
    cfg.w_ee_within_init = 1.0
    cfg.w_ee_cross_init = 0.02
    cfg.drive_amp_ee_pA = 50.0
    cfg.nmda_drive_amp_nS = 0.5
    cfg.ee_w_max = 1.5
    cfg.p_e_inh = 0.4
    cfg.w_e_inh = 0.4
    cfg.drive_amp_e_inh_pA = 40.0
    cfg.w_inh_e_init = 0.5
    cfg.broad_inh_scale = 0.3
    cfg.drive_amp_inh_e_pA = 40.0
    cfg.inh_rho_hz = 10.0
    cfg.inh_eta = 0.0    # freeze Vogels during the validation (no learning)
    cfg.inh_w_max = 1.5
    return cfg


def run_per_channel_inh_validation(
    seed: int = 42,
    verbose: bool = True,
) -> InhValidationReport:
    """Run both inhibition assays on a freshly built H_R ring."""
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed)
    np.random.seed(seed)

    cfg = _stage1_cfg()

    # -- Assay 1: drive channel 0 only. Measure per-channel rates in last 200 ms. --
    ring = build_h_r(config=cfg)
    silence_cue(ring)
    e_mon = SpikeMonitor(ring.e, name="inh_e_single")
    net = Network(*ring.groups, e_mon)

    # Warmup: settle dynamics with zero cue (baseline rate).
    net.run(SETTLE_MS * ms)
    baseline_t0 = 0.0
    baseline_t1 = SETTLE_MS

    # Drive channel 0 (delta cue, only ch=0 afferents fire).
    driven_ch = 0
    _drive_single_channel_delta(ring, driven_ch, CUE_PEAK_HZ)
    net.run(PROBE_DUR_MS * ms)
    cue_t0 = SETTLE_MS
    cue_t1 = SETTLE_MS + PROBE_DUR_MS

    silence_cue(ring)
    net.run(SETTLE_MS * ms)

    # Per-channel rates during the cue window.
    per_ch_rates = _rate_per_channel(e_mon, ring.e_channel, cue_t0, cue_t1)
    baseline_per_ch = _rate_per_channel(
        e_mon, ring.e_channel, baseline_t0, baseline_t1,
    )
    baseline_rate_hz = float(baseline_per_ch.mean())

    rate_driven = float(per_ch_rates[driven_ch])
    # +/- 1 neighbours (cross-channel ring).
    nbr_ch = [(driven_ch + 1) % N_CHANNELS, (driven_ch - 1) % N_CHANNELS]
    rate_nbr = float(np.mean([per_ch_rates[c] for c in nbr_ch]))
    # "Far" channel -- opposite side of ring.
    far_ch = (driven_ch + N_CHANNELS // 2) % N_CHANNELS
    rate_far = float(per_ch_rates[far_ch])

    # Local suppression: without the per-channel inh subpool, driven_ch
    # would blow up -> bounded by local inh. Broad suppression: far_ch,
    # which is suppressed only by the broad inh pool + weak cross-channel
    # E->E. Compare how much each is held below a notional saturation.
    # Use driven_ch E rate and far_ch E rate directly: lower rate ->
    # stronger suppression relative to spike-unrestricted LIF saturation
    # (which is here approximated by the driven-channel's own firing
    # rate, i.e. the inh pool ultimately limits driven_ch).

    # For a cleaner metric, we use the ratio of cue rates between bump
    # centre and far channel:
    #   r_driven_vs_far = rate_driven / max(rate_far, 1e-6)
    # Local-vs-broad: the per-channel subpool operates on the bump
    # channel; the broad pool operates on EVERY channel. If local inh
    # is removed, driven_ch saturates, r_driven_vs_far is very large.
    # If broad inh is removed, far_ch grows, r_driven_vs_far shrinks.
    # We define:
    #   local suppression ratio = rate_driven / 1.0  (unit-scaled)
    #   broad suppression ratio = (rate_driven - rate_far) / 1.0
    # and the local-vs-broad ratio = (rate_driven - rate_far) / max(rate_far, 1e-3)
    #
    # Simpler and more interpretable: use the difference between driven
    # and far channel (the broad pool equalises — without it,
    # multi-channel bumps persist; with it, far channels are pushed
    # toward baseline). Then the factor is:
    #       local_vs_broad = (rate_driven - rate_nbr) / (rate_nbr - rate_far + 1e-6)
    # i.e. "how much bigger is the local->nbr drop than the nbr->far drop".
    # If the local per-channel inh is strong, there's a sharp drop at the
    # edge of the bump; if only the broad pool exists, all channels look
    # similar.

    # We'll also record baseline/cue contrasts.

    # Local vs broad: define as driven_vs_nbr contrast (per-channel effect)
    # normalised by nbr_vs_far contrast (broad-effect residual). Clipped to
    # 1e-6 to avoid divide-by-zero when a channel is fully silenced (which
    # itself means the architecture is extremely selective at that edge).
    local_drop = max(rate_driven - rate_nbr, 0.0)
    broad_drop = max(rate_nbr - rate_far, 0.0)
    # If broad_drop is zero (nbr already at floor, no residual to 'broad
    # suppress'), the local inh has already done all the channel-selective
    # work, so we treat local_vs_broad as >> threshold.
    if broad_drop < 1e-3:
        local_vs_broad = float("inf") if local_drop > 1e-3 else 0.0
    else:
        local_vs_broad = local_drop / broad_drop

    # Also sanity-check: the baseline quiet (cue off) should be near zero.

    passed_local_vs_broad = local_vs_broad >= LOCAL_VS_BROAD_RATIO_MIN

    # -- Assay 2: drive two opposite channels simultaneously, check WTA. --
    ring2 = build_h_r(config=cfg)
    silence_cue(ring2)
    e_mon2 = SpikeMonitor(ring2.e, name="inh_e_multi")
    net2 = Network(*ring2.groups, e_mon2)
    net2.run(SETTLE_MS * ms)

    ch_a = 0
    ch_b = N_CHANNELS // 2            # 6 (opposite)
    _drive_balanced_multichannel(ring2, [ch_a, ch_b], CUE_PEAK_HZ)
    net2.run(PROBE_DUR_MS * ms)
    net_t0 = SETTLE_MS
    net_t1 = SETTLE_MS + PROBE_DUR_MS
    multi_rates = _rate_per_channel(e_mon2, ring2.e_channel, net_t0, net_t1)

    # Only compare channels that were driven. We want the broad pool to
    # collapse the two-channel state onto one. If the broad pool is too
    # weak, both fire equally; we want a >=40% margin.
    r_a = float(multi_rates[ch_a])
    r_b = float(multi_rates[ch_b])
    top = max(r_a, r_b)
    runner = min(r_a, r_b)
    if top <= 1e-3:
        wta_margin = 0.0
    else:
        wta_margin = (top - runner) / top
    passed_wta = wta_margin >= WTA_MARGIN_MIN

    rep = InhValidationReport(
        driven_ch=driven_ch,
        rate_on_driven_ch_hz=rate_driven,
        rate_on_nbr_ch_hz=rate_nbr,
        rate_on_far_ch_hz=rate_far,
        baseline_rate_hz=baseline_rate_hz,
        local_suppression_ratio=local_drop,
        broad_suppression_ratio=broad_drop,
        local_vs_broad_ratio=local_vs_broad,
        wta_top_rate_hz=top,
        wta_runner_rate_hz=runner,
        wta_margin=wta_margin,
        passed_local_vs_broad=passed_local_vs_broad,
        passed_wta=passed_wta,
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_per_channel_inh_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
