"""Three-stage training drivers (plan §3).

Stage 0 (this file):
    calibration — build V1 ring, drive it with grating probes across all 12
    orientations, let PV iSTDP find its target band, adjust V1_E `I_bias` by
    bisection until the mean V1_E rate sits in the Niell & Stryker band. After
    iSTDP settles, freeze PV->E weights. Verify H baseline is quiet and that
    a cue pulse biases the matched channel. Save checkpoint.

Stage 1 (deferred): incidental context learning on H recurrent E<->E only.
Stage 2 (deferred): cue -> H via teacher-forced eligibility-trace learning.

Seeds per pre-registration: seed=42 only for the current stage. Multi-seed
replication is deferred until a first-pass finding warrants it.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from brian2 import (
    Network,
    NetworkOperation,
    PoissonGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    Hz,
    ms,
    mV,
    nS,
    pA,
    prefs,
    seed as b2_seed,
)

from .v1_ring import (
    V1Ring,
    V1RingConfig,
    build_v1_ring,
    set_stimulus,
    N_CHANNELS as V1_N_CHANNELS,
    N_E_PER_CHANNEL as V1_N_E_PER,
)
from .h_ring import (
    HRing,
    HRingConfig,
    build_h_r,
    build_h_t,
    pulse_channel,
    silence_cue,
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
)
from .stimulus import (
    richter_biased_training_schedule,
    richter_crossover_training_schedule,
    tang_rotating_sequence,
    RICHTER_ORIENTATIONS_DEG,
    TANG_ORIENTATIONS_DEG,
)
from .plasticity import (
    normalize_postsyn_sum,
    eligibility_trace_cue_rule,
)
from ..validation.stage_0_gate import (  # type: ignore[relative-beyond-top-level]
    check_v1_e_rate_band,
    check_v1_pv_rate_band,
    check_v1_som_rate_band,
    check_tuning_fwhm,
    check_h_baseline_quiet,
    check_h_pulse_response,
    check_no_runaway,
    compute_fwhm_deg,
    aggregate,
    Stage0Report,
)
from ..validation.stage_1_gate import (  # type: ignore[relative-beyond-top-level]
    check_h_bump_persistence,
    check_h_transition_mi,
    check_h_rotation_mi,
    check_no_runaway as check_no_runaway_s1,
    aggregate as aggregate_s1,
    Stage1Report,
    compute_bump_persistence_ms,
)
from ..validation.stage_2_gate import (  # type: ignore[relative-beyond-top-level]
    check_cue_selectivity,
    check_bump_fraction,
    check_hr_weights_unchanged,
    check_no_runaway as check_no_runaway_s2,
    aggregate as aggregate_s2,
    Stage2Report,
)


# Default paths
CHECKPOINT_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "checkpoints",
)


@dataclass
class Stage0Result:
    """Return value of `run_stage_0`."""
    seed: int
    report: Stage0Report
    v1_cfg: V1RingConfig
    h_cfg: HRingConfig
    diagnostics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


# -- V1 probe ---------------------------------------------------------------

def _probe_v1_single_orientation(
    v1_cfg: V1RingConfig,
    theta_rad: float,
    probe_ms: float,
    seed_val: int,
    name_suffix: str,
    istdp_enabled: bool,
    bias_pA: float = 0.0,
) -> Tuple[float, float, float, np.ndarray, float]:
    """Build a V1 ring, run one orientation probe, return population rates.

    Parameters
    ----------
    v1_cfg : V1RingConfig
    theta_rad : float
        Orientation to probe.
    probe_ms : float
        Window duration for rate integration.
    seed_val : int
    name_suffix : str
        Unique Brian2 group-name suffix (stops accidental re-use errors).
    istdp_enabled : bool
        If False, PV->E Vogels synapses are set `active=False` so weights
        freeze at their current values.
    bias_pA : float
        I_bias value applied uniformly to all V1 E cells in pA.

    Returns
    -------
    (e_rate_hz, pv_rate_hz, som_rate_hz, per_channel_e_rate, fwhm_deg)
    """
    b2_seed(seed_val); np.random.seed(seed_val)

    ring = build_v1_ring(config=v1_cfg, name_prefix=f"v1_{name_suffix}")
    ring.e.I_bias = bias_pA * pA
    set_stimulus(ring, theta_rad, contrast=1.0)
    if not istdp_enabled:
        ring.pv_to_e.active = False

    e_mon = SpikeMonitor(ring.e)
    pv_mon = SpikeMonitor(ring.pv)
    som_mon = SpikeMonitor(ring.som)
    net = Network(*ring.groups, e_mon, pv_mon, som_mon)
    net.run(probe_ms * ms)

    dur_s = probe_ms / 1000.0
    e_rate = e_mon.num_spikes / (len(ring.e) * dur_s)
    pv_rate = pv_mon.num_spikes / (len(ring.pv) * dur_s)
    som_rate = som_mon.num_spikes / (len(ring.som) * dur_s)

    e_idx = np.asarray(e_mon.i[:])
    per_ch_counts = np.bincount(ring.e_channel[e_idx], minlength=V1_N_CHANNELS)
    per_ch_rate = per_ch_counts / (V1_N_E_PER * dur_s)
    fwhm = float(compute_fwhm_deg(per_ch_rate, channel_spacing_deg=15.0))

    return e_rate, pv_rate, som_rate, per_ch_rate, fwhm


# -- H probe ---------------------------------------------------------------

def _probe_h_baseline_and_pulse(
    h_cfg: HRingConfig,
    seed_val: int,
    baseline_ms: float = 500.0,
    pulse_ms: float = 200.0,
    post_pulse_ms: float = 500.0,
    pulse_rate_hz: float = 400.0,
) -> Tuple[float, np.ndarray]:
    """Run one baseline window + one pulse window on H_R.

    Returns
    -------
    (baseline_rate_hz, per_channel_pulse_counts)
    """
    b2_seed(seed_val); np.random.seed(seed_val)
    ring = build_h_r(config=h_cfg)
    silence_cue(ring)
    e_mon = SpikeMonitor(ring.e)
    net = Network(*ring.groups, e_mon)

    # Baseline
    net.run(baseline_ms * ms)
    baseline_spikes = int(e_mon.num_spikes)
    baseline_rate = baseline_spikes / (len(ring.e) * baseline_ms / 1000.0)

    # Pulse ch0
    pulse_channel(ring, channel=0, rate_hz=pulse_rate_hz)
    net.run(pulse_ms * ms)
    # Post-pulse
    silence_cue(ring)
    net.run(post_pulse_ms * ms)

    t = np.asarray(e_mon.t / ms)
    i = np.asarray(e_mon.i[:])
    in_pulse = (t >= baseline_ms) & (t < baseline_ms + pulse_ms)
    per_ch = np.bincount(ring.e_channel[i[in_pulse]], minlength=H_N_CHANNELS)
    return baseline_rate, per_ch


# -- Calibration passes -----------------------------------------------------

def _calibrate_v1_bias(
    v1_cfg: V1RingConfig,
    target_e_rate: float,
    probe_ms: float,
    seed_val: int,
    max_passes: int = 6,
    bias_low_pA: float = 0.0,
    bias_high_pA: float = 150.0,
    bias_tol_pA: float = 5.0,
    log: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """Bisection on uniform V1_E `I_bias` to hit `target_e_rate` Hz.

    Runs one probe per pass at theta=0 rad with PV iSTDP DISABLED. Vogels
    iSTDP calibration is performed after the bias is locked (otherwise the
    two adaptive loops race).

    Returns
    -------
    (best_bias_pA, best_e_rate_hz)
    """
    lo, hi = bias_low_pA, bias_high_pA
    last_rate = float("nan")
    best_bias = lo
    for k in range(max_passes):
        mid = 0.5 * (lo + hi)
        e_rate, pv_rate, som_rate, _per_ch, fwhm = _probe_v1_single_orientation(
            v1_cfg, theta_rad=0.0, probe_ms=probe_ms,
            seed_val=seed_val + k, name_suffix=f"biasprobe_{k}",
            istdp_enabled=False, bias_pA=mid,
        )
        if log is not None:
            log.append(f"  bias probe {k}: bias={mid:6.1f} pA  "
                       f"E={e_rate:.2f} Hz  PV={pv_rate:.2f} Hz  "
                       f"SOM={som_rate:.2f} Hz  FWHM={fwhm:.1f} deg")
        last_rate = e_rate
        best_bias = mid
        if e_rate > target_e_rate:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < bias_tol_pA:
            break
    return best_bias, last_rate


def _run_pv_istdp_settling(
    v1_cfg: V1RingConfig,
    bias_pA: float,
    settle_ms: float,
    seed_val: int,
    log: Optional[List[str]] = None,
) -> Tuple[V1Ring, Network, List[SpikeMonitor], Dict[str, float]]:
    """Run the ring at the calibrated bias with iSTDP ENABLED to let PV->E
    weights converge to the target post-rate.

    Returns the live ring + the still-active network + the SpikeMonitors
    attached to that network, so the caller can swap in fresh monitors for
    a subsequent probe window on the same ring (Brian2 forbids adding a
    pre-run NeuronGroup to a second Network).
    """
    b2_seed(seed_val); np.random.seed(seed_val)

    ring = build_v1_ring(config=v1_cfg, name_prefix="v1_settle")
    ring.e.I_bias = bias_pA * pA
    set_stimulus(ring, theta_rad=0.0, contrast=1.0)

    e_mon = SpikeMonitor(ring.e, name="settle_e_mon")
    pv_mon = SpikeMonitor(ring.pv, name="settle_pv_mon")
    som_mon = SpikeMonitor(ring.som, name="settle_som_mon")
    net = Network(*ring.groups, e_mon, pv_mon, som_mon)
    net.run(settle_ms * ms)

    dur_s = settle_ms / 1000.0
    diag = {
        "e_rate_hz": e_mon.num_spikes / (len(ring.e) * dur_s),
        "pv_rate_hz": pv_mon.num_spikes / (len(ring.pv) * dur_s),
        "som_rate_hz": som_mon.num_spikes / (len(ring.som) * dur_s),
        "pv_to_e_w_mean": float(np.asarray(ring.pv_to_e.w[:]).mean()),
        "pv_to_e_w_max": float(np.asarray(ring.pv_to_e.w[:]).max()),
    }
    if log is not None:
        log.append(f"  iSTDP settle {settle_ms:.0f} ms: E={diag['e_rate_hz']:.2f} Hz"
                   f"  PV={diag['pv_rate_hz']:.2f} Hz  SOM={diag['som_rate_hz']:.2f} Hz"
                   f"  pv_to_e w mean/max = {diag['pv_to_e_w_mean']:.3f}/"
                   f"{diag['pv_to_e_w_max']:.3f}")
    return ring, net, [e_mon, pv_mon, som_mon], diag


# -- Main Stage-0 driver ----------------------------------------------------

def run_stage_0(
    seed: int = 42,
    target_e_rate: float = 4.0,
    bias_probe_ms: float = 600.0,
    istdp_settle_ms: float = 3000.0,
    final_probe_ms: float = 600.0,
    checkpoint_dir: Optional[str] = None,
    v1_cfg: Optional[V1RingConfig] = None,
    h_cfg: Optional[HRingConfig] = None,
    verbose: bool = True,
) -> Stage0Result:
    """Full Stage-0 calibration driver.

    Workflow
    --------
    1. Bisection on V1_E I_bias (iSTDP OFF) to hit target_e_rate.
    2. Let PV->E Vogels iSTDP settle at the calibrated bias.
    3. Non-destructive per-orientation probe -> FWHM.
    4. H baseline + pulse probe.
    5. Check all gates; save checkpoint if all pass.
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    v1_cfg = v1_cfg or V1RingConfig()
    h_cfg = h_cfg or HRingConfig()
    log: List[str] = []

    t0 = time.time()
    log.append(f"Stage-0 driver: seed={seed}, target_e_rate={target_e_rate} Hz")

    # 1) Bias bisection
    bias_pA_, rate_at_bias = _calibrate_v1_bias(
        v1_cfg, target_e_rate, bias_probe_ms, seed_val=seed, log=log,
    )
    log.append(f"bias calibrated: {bias_pA_:.1f} pA  (rate {rate_at_bias:.2f} Hz)")

    # 2) Vogels iSTDP settle
    ring, net_settle, settle_mons, settle_diag = _run_pv_istdp_settling(
        v1_cfg, bias_pA_, istdp_settle_ms, seed_val=seed + 100, log=log,
    )
    # Freeze PV->E after settling.
    ring.pv_to_e.active = False

    # 3) Final per-orientation probe -> population rates + FWHM @ theta=0.
    #    Reuse the SAME network. Remove stale settle monitors, add fresh ones.
    for m in settle_mons:
        net_settle.remove(m)
    e_mon2 = SpikeMonitor(ring.e, name="final_e_mon")
    pv_mon2 = SpikeMonitor(ring.pv, name="final_pv_mon")
    som_mon2 = SpikeMonitor(ring.som, name="final_som_mon")
    net_settle.add(e_mon2, pv_mon2, som_mon2)
    set_stimulus(ring, theta_rad=0.0, contrast=1.0)
    net_settle.run(final_probe_ms * ms)
    dur_s = final_probe_ms / 1000.0
    final_e_rate = e_mon2.num_spikes / (len(ring.e) * dur_s)
    final_pv_rate = pv_mon2.num_spikes / (len(ring.pv) * dur_s)
    final_som_rate = som_mon2.num_spikes / (len(ring.som) * dur_s)
    e_idx2 = np.asarray(e_mon2.i[:])
    per_ch_final = np.bincount(ring.e_channel[e_idx2], minlength=V1_N_CHANNELS) / (
        V1_N_E_PER * dur_s
    )
    fwhm_final = float(compute_fwhm_deg(per_ch_final, 15.0))
    log.append(f"final V1 @ theta=0: E={final_e_rate:.2f} Hz  PV={final_pv_rate:.2f} Hz  "
               f"SOM={final_som_rate:.2f} Hz  FWHM={fwhm_final:.1f} deg  "
               f"per-ch peak={per_ch_final.max():.1f} Hz")

    # 4) H probe
    h_baseline, h_pulse_counts = _probe_h_baseline_and_pulse(h_cfg, seed_val=seed + 200)
    log.append(f"H baseline={h_baseline:.2f} Hz  "
               f"H pulse per-ch: ch0={h_pulse_counts[0]}, ch6={h_pulse_counts[6]}")

    # 5) Gate aggregation
    checks = {
        "v1_e_rate_band": check_v1_e_rate_band(final_e_rate),
        "v1_pv_rate_band": check_v1_pv_rate_band(final_pv_rate),
        "v1_som_rate_band": check_v1_som_rate_band(final_som_rate),
        "v1_tuning_fwhm_deg": check_tuning_fwhm(per_ch_final),
        "h_baseline_quiet": check_h_baseline_quiet(h_baseline),
        "h_pulse_response": check_h_pulse_response(h_pulse_counts, pulsed_channel=0),
        "no_runaway": check_no_runaway({
            "v1_e": final_e_rate, "v1_pv": final_pv_rate, "v1_som": final_som_rate,
        }),
    }
    report = aggregate(checks)
    log.append(report.summary())

    elapsed = time.time() - t0
    log.append(f"Stage-0 elapsed: {elapsed:.1f} s")
    if verbose:
        print("\n".join(log))

    ckpt_path: Optional[str] = None
    if report.passed:
        ckpt_dir = checkpoint_dir or CHECKPOINT_DIR_DEFAULT
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"stage_0_seed{seed}.npz")
        np.savez(
            ckpt_path,
            seed=np.int32(seed),
            target_e_rate=np.float64(target_e_rate),
            bias_pA=np.float64(bias_pA_),
            final_e_rate=np.float64(final_e_rate),
            final_pv_rate=np.float64(final_pv_rate),
            final_som_rate=np.float64(final_som_rate),
            fwhm_deg=np.float64(fwhm_final),
            per_ch_rate=np.asarray(per_ch_final, dtype=np.float64),
            pv_to_e_w=np.asarray(ring.pv_to_e.w[:], dtype=np.float64),
            h_baseline_rate=np.float64(h_baseline),
            h_pulse_per_ch=np.asarray(h_pulse_counts, dtype=np.int64),
            v1_drive_amp_stim_pA=np.float64(v1_cfg.drive_amp_stim_pA),
            v1_pv_rho_hz=np.float64(v1_cfg.pv_rho_hz),
        )
        if verbose:
            print(f"checkpoint saved: {ckpt_path}")

    diagnostics = {
        "bias_pA": bias_pA_,
        "final_e_rate": final_e_rate,
        "final_pv_rate": final_pv_rate,
        "final_som_rate": final_som_rate,
        "fwhm_deg": fwhm_final,
        "h_baseline": h_baseline,
        "elapsed_s": elapsed,
        **{f"settle_{k}": v for k, v in settle_diag.items()},
    }
    return Stage0Result(
        seed=seed, report=report,
        v1_cfg=v1_cfg, h_cfg=h_cfg,
        diagnostics=diagnostics, checkpoint_path=ckpt_path,
    )


# -- Stage-1: H recurrent incidental learning ------------------------------

# H has 12 channels; 6 Tang/Richter orientations map to channels 0,2,4,6,8,10
# (every other channel, matching 30° vs 15° channel spacing).
H_ORIENT_CHANNELS = np.arange(0, H_N_CHANNELS, 2)   # (6,) -> [0, 2, 4, 6, 8, 10]


def _drive_h_cue_gaussian(
    ring: HRing,
    theta_rad: float,
    peak_rate_hz: float = 150.0,
    sigma_deg: float = 15.0,
) -> None:
    """Set H cue Poisson rates to a Gaussian over channels centred at theta.

    Parameters
    ----------
    ring : HRing
    theta_rad : float
        Stimulus orientation on the 0..pi ring.
    peak_rate_hz : float
        Per-afferent rate of the centrally tuned channel's block.
    sigma_deg : float
        Gaussian width in degrees.
    """
    n_cue = int(ring.cue.N)
    block = n_cue // H_N_CHANNELS
    thetas = np.arange(H_N_CHANNELS) * (np.pi / H_N_CHANNELS)
    d = np.abs(thetas - float(theta_rad))
    d = np.minimum(d, np.pi - d)     # wrap on 0..pi ring
    sigma_rad = np.deg2rad(sigma_deg)
    per_ch = peak_rate_hz * np.exp(-0.5 * (d / sigma_rad) ** 2)
    rates = np.zeros(n_cue)
    for c in range(H_N_CHANNELS):
        rates[c * block : (c + 1) * block] = per_ch[c]
    ring.cue.rates = rates * Hz


def _run_h_schedule(
    ring: HRing,
    net: Network,
    plan,
    cue_peak_hz: float,
    cue_sigma_deg: float,
) -> None:
    """Drive H's cue through every `TrialItem` in `plan`; blanks silence cue."""
    for item in plan.items:
        if item.kind == "iti" or item.theta_rad is None:
            silence_cue(ring)
        else:
            _drive_h_cue_gaussian(
                ring, item.theta_rad,
                peak_rate_hz=cue_peak_hz, sigma_deg=cue_sigma_deg,
            )
        net.run(item.duration_ms * ms)


def _stage1_h_cfg(base: Optional[HRingConfig] = None) -> HRingConfig:
    """Tuned HRing config for Stage 1 (Sprint-3 architecture).

    The Sprint-3 revision adds NMDA slow recurrent on H_E (Wang 2001) and
    per-channel inhibition. Configuration below targets a post-training
    bump that

    (a) fires at biologically plausible rates during the cue pulse
        (30-60 Hz per cell on the peak channel),
    (b) persists for 200-500 ms after cue offset (Stage-1 gate band,
        matching Kok / Richter paradigms where the predictive bump must
        carry into the trailer window but not beyond), and
    (c) decays to baseline before the next trial begins.

    AMPA drive (50 pA) is moderate so AMPA alone cannot sustain the bump;
    NMDA (nmda_drive_amp_nS=0.2) provides the slow (tau_nmda=100 ms)
    recurrent conductance that holds the bump open, with Mg2+ block
    ensuring the NMDA channel only conducts when E cells are depolarized
    (i.e. inside the bump). Per-channel inhibition prevents cross-channel
    spread, and Vogels iSTDP (rho=5 Hz, eta=5e-3) stabilises the overall
    E/I balance during training.
    """
    cfg = base or HRingConfig()
    # H recurrent E->E AMPA (fast, plastic) + NMDA (slow, co-released).
    cfg.w_ee_within_init = 1.0
    cfg.w_ee_cross_init = 0.02
    cfg.drive_amp_ee_pA = 50.0          # moderate AMPA kick (Wang-style)
    cfg.nmda_drive_amp_nS = 0.5         # NMDA per-spike conductance (Wang 2001, frozen probe threshold at 0.3)
    cfg.ee_w_max = 1.5
    cfg.ee_A_plus = 2e-4                # 5x smaller than before; prevents cross-nbr LTP runaway in 90s schedule
    cfg.ee_A_minus = 2.1e-4
    cfg.target_postsyn_sum = 16.0       # ~= init sum (16 within-ch * 1.0 + 32 cross-nbr * 0.02)
    # Per-channel + broad inhibition (split inh pool in h_ring.py).
    cfg.p_e_inh = 0.4                   # E -> broad inh sparsity
    cfg.w_e_inh = 0.4
    cfg.drive_amp_e_inh_pA = 40.0
    cfg.w_inh_e_init = 0.5              # per-channel local inh->E init weight
    cfg.broad_inh_scale = 0.3           # broad inh->E scale factor
    cfg.drive_amp_inh_e_pA = 40.0
    cfg.inh_rho_hz = 10.0               # Vogels target E rate (target bump @ 20-30 Hz
                                        # selected, 1-2 Hz elsewhere, avg 10 Hz)
    cfg.inh_eta = 5e-3                  # standard Vogels rate; capped by inh_w_max
    cfg.inh_w_max = 1.5                 # Vogels weight ceiling. Caps max inhibition
                                        # at 3x init (w_inh_e_init=0.5); prevents
                                        # schedule-length-dependent over-suppression
                                        # of the post-pulse bump.
    return cfg


def _drive_h_broad_noise(ring: HRing, mean_rate_hz: float = 40.0) -> None:
    """Uniform low Poisson rate across all cue afferents (for pre-settle)."""
    n_cue = int(ring.cue.N)
    ring.cue.rates = np.full(n_cue, mean_rate_hz) * Hz


def _make_postsyn_normalizer(
    syn, target_sum: float, dt_ms: float = 200.0, name: str = "norm_ee",
) -> "NetworkOperation":
    """Return a NetworkOperation that rescales postsyn-sum of `syn` at cadence."""
    def _op():
        normalize_postsyn_sum(syn, target_sum=target_sum)
    return NetworkOperation(_op, dt=dt_ms * ms, name=name)


def _per_channel_rate_in_window(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    e_channel: np.ndarray,
    t_start_ms: float,
    t_end_ms: float,
    n_channels: int,
    n_per_channel: int,
) -> np.ndarray:
    """Per-channel firing rate (Hz) over [t_start_ms, t_end_ms)."""
    mask = (spike_t_ms >= t_start_ms) & (spike_t_ms < t_end_ms)
    per_ch = np.bincount(e_channel[spike_i[mask]], minlength=n_channels)
    dur_s = max((t_end_ms - t_start_ms) / 1000.0, 1e-6)
    return per_ch / (n_per_channel * dur_s)


def _peak_channel_ms_series(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    e_channel: np.ndarray,
    t0_ms: float,
    t1_ms: float,
    peak_ch: int,
    n_per_channel: int,
    bin_ms: float = 10.0,
) -> np.ndarray:
    """Per-bin firing-rate series (Hz) of `peak_ch` over [t0_ms, t1_ms).

    Returns (n_bins,) where bin duration is `bin_ms`.
    """
    n_bins = max(1, int(round((t1_ms - t0_ms) / bin_ms)))
    mask = ((spike_t_ms >= t0_ms) & (spike_t_ms < t1_ms)
            & (e_channel[spike_i] == peak_ch))
    if not np.any(mask):
        return np.zeros(n_bins, dtype=np.float64)
    t_rel = spike_t_ms[mask] - t0_ms
    bins = np.linspace(0.0, n_bins * bin_ms, n_bins + 1)
    counts, _ = np.histogram(t_rel, bins=bins)
    # rate Hz = counts / (n_per_channel * bin_s)
    bin_s = bin_ms / 1000.0
    return counts.astype(np.float64) / (n_per_channel * bin_s)


@dataclass
class Stage1Result:
    """Return value of `run_stage_1_hr` / `run_stage_1_ht`."""
    grammar: str
    seed: int
    report: Stage1Report
    h_cfg: HRingConfig
    diagnostics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


# -- Stage-1: Richter H_R --------------------------------------------------

def run_stage_1_hr(
    seed: int = 42,
    n_trials: int = 72,
    leader_ms: float = 500.0,
    trailer_ms: float = 500.0,
    iti_ms: float = 1500.0,
    probe_delay_ms: float = 500.0,
    probe_window_ms: float = 100.0,
    cue_peak_hz: float = 300.0,
    cue_sigma_deg: float = 15.0,
    presettle_ms: float = 10000.0,
    presettle_noise_hz: float = 40.0,
    normalize_dt_ms: float = 200.0,
    trials_used_tail: Optional[int] = None,
    persist_probe_theta_rad: float = 0.0,
    persist_pulse_ms: float = 300.0,
    persist_post_ms: float = 1000.0,
    checkpoint_dir: Optional[str] = None,
    h_cfg: Optional[HRingConfig] = None,
    verbose: bool = True,
    *,
    biased: bool = True,
    p_bias: float = 0.80,
    derangement: Optional[Tuple[int, ...]] = None,
    gate_window: str = "pre_trailer",
) -> Stage1Result:
    """Stage-1 driver for H_R (Richter crossover grammar).

    Workflow
    --------
    1. Build H_R, freeze Vogels iSTDP (inh->E frozen at init weights).
    2. Generate Richter schedule. With ``biased=True`` (Sprint 5e Fix A
       default) this is the biased deranged-permutation schedule with
       ``P(L → f(L)) = p_bias``; ``biased=False`` falls back to the
       legacy balanced 36-pair schedule (ablation / assay-time generator).
    3. Drive cue with per-item Gaussian-over-channels rate; H E<->E STDP
       is plastic throughout (Hebbian pair rule, `pair_stdp_with_normalization`).
    4. Post-hoc, per trial: compute H_R argmax in the gate-window probe.
       With ``gate_window="pre_trailer"`` (Sprint 5e Fix B default) the
       window is the last ``probe_window_ms`` ms of the leader —
       ``[leader_end - probe_window_ms, leader_end]`` — and the gate metric
       is ``P(argmax == expected_trailer_idx)``, i.e. the pre-trailer
       forecast. With ``gate_window="post_trailer"`` the window is the
       legacy ``trailer_end + probe_delay_ms ± probe_window_ms/2`` and the
       metric is ``MI(leader, argmax)`` (post-trailer bump tracking).
    5. Aggregate gate checks: bump_persistence, transition_MI /
       preprobe_forecast, no_runaway.

    The rotation MI check is not computed for this grammar (N/A for Richter).
    """
    if gate_window not in ("pre_trailer", "post_trailer"):
        raise ValueError(
            f"gate_window must be 'pre_trailer' or 'post_trailer', "
            f"got {gate_window!r}"
        )
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    h_cfg = _stage1_h_cfg(h_cfg)
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate schedule. Fix A default: biased deranged permutation with
    # P(L → f(L)) = p_bias (default 0.80). Legacy balanced schedule is
    # retained behind biased=False for ablation / assay use.
    if biased:
        if derangement is None:
            derangement = (1, 2, 3, 4, 5, 0)
        plan = richter_biased_training_schedule(
            rng, n_trials=n_trials, p_bias=p_bias, derangement=derangement,
            leader_ms=leader_ms, trailer_ms=trailer_ms, iti_ms=iti_ms,
        )
        schedule_name = "richter_biased"
    else:
        plan = richter_crossover_training_schedule(
            rng, n_trials=n_trials,
            leader_ms=leader_ms, trailer_ms=trailer_ms, iti_ms=iti_ms,
        )
        schedule_name = "richter_crossover"
    pairs = plan.meta["pairs"]               # (n_trials, 2)
    # Per-trial expected trailer for the pre-probe gate (from derangement).
    # On the legacy balanced schedule `expected_trailer_idx` is not produced
    # by the generator, so we synthesize one for consistency: default f(L)
    # = (L + 1) % 6, matching the reviewer's reference derangement. It is
    # only consulted when gate_window="pre_trailer".
    if "expected_trailer_idx" in plan.meta:
        expected_trailer_all = np.asarray(
            plan.meta["expected_trailer_idx"], dtype=np.int64,
        )
    else:
        default_deran = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
        expected_trailer_all = default_deran[pairs[:, 0].astype(np.int64)]

    log: List[str] = [
        f"Stage-1 H_R driver: seed={seed}, n_trials={n_trials}, "
        f"ITI={iti_ms:.0f}ms, cue_peak={cue_peak_hz:.0f}Hz "
        f"sigma={cue_sigma_deg:.0f}deg",
        f"  schedule: {schedule_name} biased={biased} p_bias={p_bias:.2f}  "
        f"{plan.total_ms/1000.0:.1f} s sim time, "
        f"{len(plan.items)} items",
        f"  gate_window={gate_window}  probe_window={probe_window_ms:.0f}ms"
        + (f" probe_delay={probe_delay_ms:.0f}ms" if gate_window == "post_trailer" else ""),
        f"  presettle={presettle_ms:.0f}ms @{presettle_noise_hz:.0f}Hz, "
        f"norm_dt={normalize_dt_ms:.0f}ms target_sum={h_cfg.target_postsyn_sum:.1f}",
    ]

    # Build H_R. Vogels iSTDP stays plastic (caps runaway under strong cue),
    # but learns slowly (see cfg.inh_eta in `_stage1_h_cfg`) so inh->E does
    # not over-strengthen and kill the post-pulse bump across the schedule.
    ring = build_h_r(config=h_cfg)
    silence_cue(ring)

    e_mon = SpikeMonitor(ring.e, name="s1_hr_e")
    inh_mon = SpikeMonitor(ring.inh, name="s1_hr_inh")
    # Post-syn sum normalizer on E->E (prevents STDP-LTD erosion of recurrent
    # weights during uncorrelated pre-settle + schedule). Cadence=200 ms.
    ee_norm = _make_postsyn_normalizer(
        ring.ee, target_sum=h_cfg.target_postsyn_sum,
        dt_ms=normalize_dt_ms, name="s1_hr_ee_norm",
    )
    net = Network(*ring.groups, e_mon, inh_mon, ee_norm)

    # Pre-settle: ee recurrent drive + ee STDP plastic. Vogels iSTDP is
    # frozen (see above); pre-settle just lets Poisson noise fill buffers
    # and allows ee STDP to find a mild equilibrium under broadband drive.
    if presettle_ms > 0:
        _drive_h_broad_noise(ring, mean_rate_hz=presettle_noise_hz)
        net.run(presettle_ms * ms)
        silence_cue(ring)

    # Mark plan start time (schedule trials start *after* pre-settle in abs time).
    schedule_start_abs_ms = float(net.t / ms)

    # Run full schedule.
    t0 = time.time()
    _run_h_schedule(ring, net, plan, cue_peak_hz, cue_sigma_deg)
    sim_wall_s = time.time() - t0

    # Extract spikes.
    spike_i = np.asarray(e_mon.i[:], dtype=np.int64)
    spike_t_ms = np.asarray(e_mon.t / ms, dtype=np.float64)
    e_channel = ring.e_channel

    # Trial time bookkeeping. Schedule items started at `schedule_start_abs_ms`.
    trial_ms = leader_ms + trailer_ms + iti_ms
    trial_start_abs = schedule_start_abs_ms + np.arange(n_trials) * trial_ms
    leader_end_abs  = trial_start_abs + leader_ms                 # (n_trials,)
    trailer_end_abs = trial_start_abs + leader_ms + trailer_ms    # (n_trials,)
    # Gate-window selection. Sprint 5e Fix B: the pre-trailer gate asks
    # whether H_R expresses the forecast BEFORE the trailer arrives, so
    # the probe sits in the last `probe_window_ms` ms of the leader.
    if gate_window == "pre_trailer":
        probe_win_start = leader_end_abs - probe_window_ms
        probe_win_end   = leader_end_abs.copy()
    else:  # post_trailer — legacy
        probe_win_start = trailer_end_abs + probe_delay_ms - probe_window_ms / 2.0
        probe_win_end   = trailer_end_abs + probe_delay_ms + probe_window_ms / 2.0

    # Use the LAST half of trials (plasticity has kicked in).
    if trials_used_tail is None:
        trials_used_tail = max(n_trials // 2, 36)
    start_k = max(0, n_trials - trials_used_tail)

    # Per-trial H argmax in the gate window, mapped to 6 orients.
    h_argmax_per_trial_6 = np.empty(trials_used_tail, dtype=np.int64)
    leader_idx_used = np.empty(trials_used_tail, dtype=np.int64)
    expected_trailer_used = np.empty(trials_used_tail, dtype=np.int64)
    for kk, k in enumerate(range(start_k, n_trials)):
        rates_12 = _per_channel_rate_in_window(
            spike_i, spike_t_ms, e_channel,
            probe_win_start[k], probe_win_end[k],
            H_N_CHANNELS, H_N_E_PER,
        )
        rates_6 = rates_12[H_ORIENT_CHANNELS]      # (6,)
        h_argmax_per_trial_6[kk] = int(np.argmax(rates_6))
        leader_idx_used[kk] = int(pairs[k, 0])
        expected_trailer_used[kk] = int(expected_trailer_all[k])

    # Bump persistence: dedicated post-training probe (like H_T). Single
    # isolated pulse -> silence -> measure persistence on peak channel.
    # Cleaner than mid-schedule ITI because ITI is contaminated by the next
    # trial's leader arriving soon after.
    silence_cue(ring)
    net.run(200 * ms)                       # brief quiet before probe
    t_pulse_start = float(net.t / ms)
    _drive_h_cue_gaussian(
        ring, persist_probe_theta_rad, cue_peak_hz, cue_sigma_deg,
    )
    net.run(persist_pulse_ms * ms)
    t_pulse_end = float(net.t / ms)
    silence_cue(ring)
    net.run(persist_post_ms * ms)

    spike_i2 = np.asarray(e_mon.i[:], dtype=np.int64)
    spike_t_ms2 = np.asarray(e_mon.t / ms, dtype=np.float64)

    # Peak channel from the last 100 ms of pulse.
    peak_rates_probe = _per_channel_rate_in_window(
        spike_i2, spike_t_ms2, e_channel,
        t_pulse_end - 100.0, t_pulse_end,
        H_N_CHANNELS, H_N_E_PER,
    )
    peak_ch_probe = int(np.argmax(peak_rates_probe))
    bin_ms = 10.0
    probe_series = _peak_channel_ms_series(
        spike_i2, spike_t_ms2, e_channel,
        t_pulse_end, t_pulse_end + persist_post_ms,
        peak_ch_probe, H_N_E_PER, bin_ms=bin_ms,
    )
    persistence_med = compute_bump_persistence_ms(
        probe_series, offset_idx=0, dt_ms=bin_ms, floor_hz=2.0,
    )
    log.append(f"  post-training probe: peak_ch={peak_ch_probe} "
               f"(cued theta={np.rad2deg(persist_probe_theta_rad):.0f}deg), "
               f"persistence={persistence_med:.1f} ms")

    # Overall layer rates during SCHEDULE ONLY (exclude pre-settle).
    total_sim_s = plan.total_ms / 1000.0
    e_t_ms_all = np.asarray(e_mon.t / ms, dtype=np.float64)
    inh_t_ms_all = np.asarray(inh_mon.t / ms, dtype=np.float64)
    n_e_sched = int(((e_t_ms_all >= schedule_start_abs_ms)).sum())
    n_inh_sched = int(((inh_t_ms_all >= schedule_start_abs_ms)).sum())
    e_rate = float(n_e_sched) / (len(ring.e) * total_sim_s)
    inh_rate = float(n_inh_sched) / (len(ring.inh) * total_sim_s)
    log.append(f"  mean H rates during schedule: E={e_rate:.2f} Hz, "
               f"inh={inh_rate:.2f} Hz")

    # Plasticity diagnostics.
    w_ee = np.asarray(ring.ee.w[:], dtype=np.float64)
    ee_i = np.asarray(ring.ee.i[:], dtype=np.int64)
    ee_j = np.asarray(ring.ee.j[:], dtype=np.int64)
    ci = e_channel[ee_i]; cj = e_channel[ee_j]
    dc = np.abs(ci - cj); dc = np.minimum(dc, H_N_CHANNELS - dc)
    log.append(f"  ee weights: mean={w_ee.mean():.3f}  max={w_ee.max():.3f} "
               f"within-ch mean={w_ee[dc==0].mean():.3f} "
               f"cross-nbr mean={w_ee[dc==1].mean():.3f}")

    # Gate aggregation.
    # For bump persistence, synthesize a tiny rate array for the checker:
    # the checker is designed for a per-ms series, but we pass the median
    # as a 2-element "hi then 0" series so _in_band(p, band) works cleanly.
    # Cleaner: construct a representative series: `persistence_med` ms high,
    # then low.
    fake_series_bin_ms = 10.0
    fake_high_bins = max(1, int(round(persistence_med / fake_series_bin_ms))) if not np.isnan(persistence_med) else 0
    fake_series = np.concatenate([
        np.full(fake_high_bins, 5.0),
        np.zeros(200),
    ])
    bump_check = check_h_bump_persistence(
        fake_series, offset_idx=0, dt_ms=fake_series_bin_ms,
    )

    transition_check = check_h_transition_mi(
        leader_idx_used, h_argmax_per_trial_6, n_orient=6,
        gate_window=gate_window,
        expected_trailer_idx=expected_trailer_used,
    )
    runaway_check = check_no_runaway_s1(
        {"h_e": e_rate, "h_inh": inh_rate},
    )
    # The gate metric name differs per mode: pre_trailer → probability,
    # post_trailer → MI in bits. Use the name the check emitted so the
    # Stage-1 report labels the value correctly.
    checks = {
        "h_bump_persistence_ms": bump_check,
        transition_check.name: transition_check,
        "no_runaway": runaway_check,
    }
    report = aggregate_s1(checks)
    log.append(report.summary())
    log.append(f"Stage-1 H_R elapsed: {sim_wall_s:.1f} s sim wall-clock")
    if verbose:
        print("\n".join(log))

    # Checkpoint.
    ckpt_path: Optional[str] = None
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR_DEFAULT
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"stage_1_hr_seed{seed}.npz")
    np.savez(
        ckpt_path,
        seed=np.int32(seed),
        grammar=np.bytes_("richter"),
        n_trials=np.int32(n_trials),
        leader_idx=leader_idx_used,
        h_argmax=h_argmax_per_trial_6,
        expected_trailer_idx=expected_trailer_used,
        gate_window=np.bytes_(gate_window),
        schedule=np.bytes_(schedule_name),
        p_bias=np.float64(p_bias),
        derangement=np.asarray(
            derangement if derangement is not None else (1, 2, 3, 4, 5, 0),
            dtype=np.int64,
        ),
        persistence_med_ms=np.float64(persistence_med),
        e_rate_hz=np.float64(e_rate),
        inh_rate_hz=np.float64(inh_rate),
        ee_w_final=w_ee,
        passed=np.bool_(report.passed),
    )
    if verbose:
        print(f"checkpoint saved: {ckpt_path}")

    return Stage1Result(
        grammar="richter", seed=seed,
        report=report, h_cfg=h_cfg,
        diagnostics={
            "persistence_med_ms": persistence_med,
            "e_rate_hz": e_rate, "inh_rate_hz": inh_rate,
            "ee_w_mean": float(w_ee.mean()),
            "ee_w_max": float(w_ee.max()),
            "sim_wall_s": sim_wall_s,
        },
        checkpoint_path=ckpt_path,
    )


# -- Stage-1: Tang H_T -----------------------------------------------------

def run_stage_1_ht(
    seed: int = 42,
    n_items: int = 500,
    item_ms: float = 250.0,
    probe_window_ms: float = 50.0,
    cue_peak_hz: float = 300.0,
    cue_sigma_deg: float = 15.0,
    presettle_ms: float = 10000.0,
    presettle_noise_hz: float = 40.0,
    normalize_dt_ms: float = 200.0,
    items_used_tail: Optional[int] = None,
    persist_probe_theta_rad: float = 0.0,
    persist_pulse_ms: float = 300.0,
    persist_post_ms: float = 1000.0,
    checkpoint_dir: Optional[str] = None,
    h_cfg: Optional[HRingConfig] = None,
    verbose: bool = True,
) -> Stage1Result:
    """Stage-1 driver for H_T (Tang rotating-orientation grammar).

    Workflow
    --------
    1. Build H_T, freeze Vogels iSTDP.
    2. Tang schedule (n_items items, 4 Hz, back-to-back, blocks 5..9).
    3. Drive plastic H E<->E throughout; post-hoc per-item argmax at item end.
    4. Compute MI(expected_next, argmax) over the last `items_used_tail`
       items (excluding items that are the last in their block, where the
       rotation rule no longer constrains "next").
    5. Add a brief post-training probe for bump persistence (single pulse
       -> silence -> measure peak-channel persistence).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    h_cfg = _stage1_h_cfg(h_cfg)
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    plan = tang_rotating_sequence(rng, n_items=n_items, item_ms=item_ms)
    n_items_actual = plan.meta["n_items"]
    deviant_mask = plan.meta["deviant_mask"]       # (n_items,)
    block_ids = plan.meta["block_ids"]             # (n_items,)
    rotation_dir = plan.meta["rotation_dir"]       # (n_items,) +-1

    # Orient indices for each item (before deviant substitution we can't know,
    # but meta in item contains "orient_idx" per item).
    orient_idx_per_item = np.asarray(
        [it.meta["orient_idx"] for it in plan.items], dtype=np.int64,
    )
    pos_in_block = np.asarray(
        [it.meta["pos_in_block"] for it in plan.items], dtype=np.int64,
    )
    block_len_per_item = np.asarray(
        [it.meta["block_len"] for it in plan.items], dtype=np.int64,
    )

    log: List[str] = [
        f"Stage-1 H_T driver: seed={seed}, n_items={n_items_actual}, "
        f"item_ms={item_ms:.0f}ms, cue_peak={cue_peak_hz:.0f}Hz "
        f"sigma={cue_sigma_deg:.0f}deg",
        f"  schedule: {plan.total_ms/1000.0:.1f} s sim time",
    ]

    # Build H_T. Vogels iSTDP plastic with slow learning rate (same rationale
    # as run_stage_1_hr).
    ring = build_h_t(config=h_cfg)
    silence_cue(ring)

    e_mon = SpikeMonitor(ring.e, name="s1_ht_e")
    inh_mon = SpikeMonitor(ring.inh, name="s1_ht_inh")
    # Post-syn sum normalizer on E->E (prevents STDP-LTD erosion during the
    # long schedule). Cadence=200 ms. Same as H_R.
    ee_norm = _make_postsyn_normalizer(
        ring.ee, target_sum=h_cfg.target_postsyn_sum,
        dt_ms=normalize_dt_ms, name="s1_ht_ee_norm",
    )
    net = Network(*ring.groups, e_mon, inh_mon, ee_norm)

    if presettle_ms > 0:
        _drive_h_broad_noise(ring, mean_rate_hz=presettle_noise_hz)
        net.run(presettle_ms * ms)
        silence_cue(ring)

    schedule_start_abs_ms = float(net.t / ms)

    # Run full schedule.
    t0 = time.time()
    _run_h_schedule(ring, net, plan, cue_peak_hz, cue_sigma_deg)
    sim_wall_s = time.time() - t0

    # Extract spikes.
    spike_i = np.asarray(e_mon.i[:], dtype=np.int64)
    spike_t_ms = np.asarray(e_mon.t / ms, dtype=np.float64)
    e_channel = ring.e_channel

    # Item boundaries (offset by pre-settle duration).
    item_end_abs = schedule_start_abs_ms + (np.arange(n_items_actual) + 1) * item_ms
    probe_win_start = item_end_abs - probe_window_ms
    probe_win_end   = item_end_abs

    # Per-item argmax -> 6 orients.
    h_argmax_per_item_6 = np.empty(n_items_actual, dtype=np.int64)
    for k in range(n_items_actual):
        rates_12 = _per_channel_rate_in_window(
            spike_i, spike_t_ms, e_channel,
            probe_win_start[k], probe_win_end[k],
            H_N_CHANNELS, H_N_E_PER,
        )
        rates_6 = rates_12[H_ORIENT_CHANNELS]
        h_argmax_per_item_6[k] = int(np.argmax(rates_6))

    # Compute expected-next per item from rotation rule.
    # For an item at block position k with direction `d`, the expected next
    # orientation (at position k+1) is:
    #     (start + d*(k+1)) % 6
    # where `start` is the first item's orient_idx in that block.
    expected_next_per_item = np.empty(n_items_actual, dtype=np.int64)
    # Rebuild `start` per block from first-occurrence orient_idx.
    block_start_orient: Dict[int, int] = {}
    for k in range(n_items_actual):
        bid = int(block_ids[k])
        if bid not in block_start_orient:
            block_start_orient[bid] = int(orient_idx_per_item[k])
    for k in range(n_items_actual):
        start = block_start_orient[int(block_ids[k])]
        d = int(rotation_dir[k])
        kpos = int(pos_in_block[k])
        expected_next_per_item[k] = (start + d * (kpos + 1)) % 6

    # Mask: exclude the last item of each block (next belongs to next block;
    # rule-predicted continuation is ill-defined from H's perspective).
    last_in_block_mask = (pos_in_block == (block_len_per_item - 1))
    # Also restrict to last items_used_tail to give plasticity time to work.
    if items_used_tail is None:
        items_used_tail = max(n_items_actual // 2, 200)
    tail_start = max(0, n_items_actual - items_used_tail)
    tail_mask = np.arange(n_items_actual) >= tail_start
    use_mask = tail_mask & (~last_in_block_mask)

    expected_used = expected_next_per_item[use_mask]
    argmax_used = h_argmax_per_item_6[use_mask]
    log.append(f"  MI samples: n_used={int(use_mask.sum())} "
               f"(tail_start={tail_start}, last-in-block excluded: "
               f"{int(last_in_block_mask.sum())})")

    rotation_check = check_h_rotation_mi(
        expected_used, argmax_used, n_orient=6,
    )

    # Overall rates during SCHEDULE ONLY (exclude pre-settle).
    total_sim_s = plan.total_ms / 1000.0
    e_t_all_ms = np.asarray(e_mon.t / ms, dtype=np.float64)
    inh_t_all_ms = np.asarray(inh_mon.t / ms, dtype=np.float64)
    schedule_end_abs_ms = schedule_start_abs_ms + plan.total_ms
    m_e = (e_t_all_ms >= schedule_start_abs_ms) & (e_t_all_ms < schedule_end_abs_ms)
    m_inh = (inh_t_all_ms >= schedule_start_abs_ms) & (inh_t_all_ms < schedule_end_abs_ms)
    e_rate = float(int(m_e.sum())) / (len(ring.e) * total_sim_s)
    inh_rate = float(int(m_inh.sum())) / (len(ring.inh) * total_sim_s)

    # Plasticity diagnostics.
    w_ee = np.asarray(ring.ee.w[:], dtype=np.float64)
    ee_i = np.asarray(ring.ee.i[:], dtype=np.int64)
    ee_j = np.asarray(ring.ee.j[:], dtype=np.int64)
    ci = e_channel[ee_i]; cj = e_channel[ee_j]
    dc = np.abs(ci - cj); dc = np.minimum(dc, H_N_CHANNELS - dc)
    log.append(f"  ee weights: mean={w_ee.mean():.3f}  max={w_ee.max():.3f} "
               f"within-ch mean={w_ee[dc==0].mean():.3f} "
               f"cross-nbr mean={w_ee[dc==1].mean():.3f}")
    log.append(f"  mean H rates: E={e_rate:.2f} Hz, inh={inh_rate:.2f} Hz")

    # Bump persistence: one probe pulse after training.
    # Add an extra 200ms baseline to let the network settle.
    silence_cue(ring)
    net.run(200 * ms)
    t_pulse_start = float(net.t / ms)
    _drive_h_cue_gaussian(
        ring, persist_probe_theta_rad, cue_peak_hz, cue_sigma_deg,
    )
    net.run(persist_pulse_ms * ms)
    t_pulse_end = float(net.t / ms)
    silence_cue(ring)
    net.run(persist_post_ms * ms)

    # Re-extract spikes from the probe window.
    spike_i2 = np.asarray(e_mon.i[:], dtype=np.int64)
    spike_t_ms2 = np.asarray(e_mon.t / ms, dtype=np.float64)

    # Peak channel: last 100ms of pulse.
    peak_rates = _per_channel_rate_in_window(
        spike_i2, spike_t_ms2, e_channel,
        t_pulse_end - 100.0, t_pulse_end,
        H_N_CHANNELS, H_N_E_PER,
    )
    peak_ch_probe = int(np.argmax(peak_rates))
    # Rate series 10ms bins, from pulse offset to end of post-probe window.
    bin_ms = 10.0
    probe_series = _peak_channel_ms_series(
        spike_i2, spike_t_ms2, e_channel,
        t_pulse_end, t_pulse_end + persist_post_ms,
        peak_ch_probe, H_N_E_PER, bin_ms=bin_ms,
    )
    persistence_probe = compute_bump_persistence_ms(
        probe_series, offset_idx=0, dt_ms=bin_ms, floor_hz=2.0,
    )
    log.append(f"  bump probe: peak_ch={peak_ch_probe}, "
               f"persistence={persistence_probe:.1f} ms "
               f"(cue at theta={np.rad2deg(persist_probe_theta_rad):.0f}deg)")

    # Gate aggregation.
    fake_series_bin_ms = 10.0
    fake_high_bins = max(1, int(round(persistence_probe / fake_series_bin_ms))) \
        if not np.isnan(persistence_probe) else 0
    fake_series = np.concatenate([
        np.full(fake_high_bins, 5.0),
        np.zeros(200),
    ])
    bump_check = check_h_bump_persistence(
        fake_series, offset_idx=0, dt_ms=fake_series_bin_ms,
    )
    runaway_check = check_no_runaway_s1(
        {"h_e": e_rate, "h_inh": inh_rate},
    )
    checks = {
        "h_bump_persistence_ms": bump_check,
        "h_rotation_mi_bits": rotation_check,
        "no_runaway": runaway_check,
    }
    report = aggregate_s1(checks)
    log.append(report.summary())
    log.append(f"Stage-1 H_T elapsed: {sim_wall_s:.1f} s sim wall-clock")
    if verbose:
        print("\n".join(log))

    ckpt_path: Optional[str] = None
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR_DEFAULT
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"stage_1_ht_seed{seed}.npz")
    np.savez(
        ckpt_path,
        seed=np.int32(seed),
        grammar=np.bytes_("tang"),
        n_items=np.int32(n_items_actual),
        expected_next=expected_next_per_item,
        h_argmax=h_argmax_per_item_6,
        persistence_probe_ms=np.float64(persistence_probe),
        e_rate_hz=np.float64(e_rate),
        inh_rate_hz=np.float64(inh_rate),
        ee_w_final=w_ee,
        passed=np.bool_(report.passed),
    )
    if verbose:
        print(f"checkpoint saved: {ckpt_path}")

    return Stage1Result(
        grammar="tang", seed=seed,
        report=report, h_cfg=h_cfg,
        diagnostics={
            "persistence_probe_ms": persistence_probe,
            "e_rate_hz": e_rate, "inh_rate_hz": inh_rate,
            "ee_w_mean": float(w_ee.mean()),
            "ee_w_max": float(w_ee.max()),
            "sim_wall_s": sim_wall_s,
        },
        checkpoint_path=ckpt_path,
    )


# -- Stage-2: cue learning on H_R ------------------------------------------

# Stage-2 paradigm constants (plan §3; Lead dispatch for Sprint 4).
# H and V1 share 12-channel geometry at 15° spacing, so channel c has pref
# orientation c * 15°. 45° → channel 3; 135° → channel 9.
STAGE2_CUE_CHANNELS: Tuple[int, int] = (3, 9)
STAGE2_N_CUE_AFFERENTS: int = 32
STAGE2_CUE_ACTIVE_HZ: float = 80.0       # per-afferent rate when cue "on"
# Cue drive is tuned so the cue alone is BELOW rheobase in the untrained
# ring -- LTP only happens when the teacher fires matched-channel H_E
# cells during the grating epoch (which pairs with still-elevated elig
# from the cue). Without this, the broad all-to-all cue LTPs every
# cue->H synapse, saturating w_max and washing out selectivity.
STAGE2_CUE_DRIVE_PA: float = 20.0        # H_E rheobase ~200 pA. Cue-alone tonic:
                                         # 32 * 80 Hz * w * 20 pA * 5 ms = w * 256 pA.
                                         # At w_init=0.1: 25.6 pA (clearly sub);
                                         # at trained matched w≈1.0: 256 pA (supra);
                                         # at trained unmatched w≈0.4: 102 pA (sub).
STAGE2_TAU_ELIG_MS: float = 1500.0       # elig decay (plan; validated in Sprint-4 Step 1)
STAGE2_LR: float = 0.0002                # Per matched synapse, per valid trial:
                                         # ~15 post_spikes/cell * 0.43 avg elig *
                                         # 0.0002 = +0.0013 LTP. 150 valid trials -->
                                         # +0.19 LTP. Plus positive-feedback from
                                         # cue-alone firing (grows nonlinearly once
                                         # w crosses ~1.2). Target matched w ~1.5.
                                         # Unmatched 50-trial LTP ~0.065, no feedback.
STAGE2_W_INIT: float = 0.1               # Low init so untrained cue is sub-rheobase
                                         # (25.6 pA tonic << 200 pA rheobase).
STAGE2_W_MAX: float = 2.0
STAGE2_TEACHER_W: float = 1.0            # V1 -> H_R fixed teacher weight (unused
                                         # after direct-injection refactor)
STAGE2_TEACHER_DRIVE_PA: float = 150.0   # V1 -> H_R per-spike drive (unused)
STAGE2_TEACHER_BIAS_PA: float = 300.0    # DC bias injected on matched-channel H_E
                                         # cells during grating epoch. Replaces the
                                         # V1 -> H_R Poisson teacher: cleaner (no
                                         # V1 tuning spillover to adjacent channels)
                                         # and faster (no V1 physics). H_E rheobase
                                         # is I_rheo=gL*(Vt-EL)=10*20=200 pA; 300 pA
                                         # is 1.5x rheobase -> sustained ~30 Hz.
STAGE2_VALID_FRAC: float = 0.75          # 75% of training trials = cue-matched
STAGE2_CUE_MS: float = 500.0
STAGE2_GAP_MS: float = 500.0
STAGE2_GRATING_MS: float = 500.0
STAGE2_ITI_MS: float = 2500.0
STAGE2_N_PROBES_PER_CUE: int = 20
STAGE2_PROBE_GAP_MS: float = 500.0


@dataclass
class Stage2Result:
    """Return value of `run_stage_2_cue`."""
    seed: int
    report: Stage2Report
    v1_cfg: V1RingConfig
    h_cfg: HRingConfig
    diagnostics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


def _freeze_v1_ring_plasticity(ring: V1Ring) -> None:
    """Freeze Vogels iSTDP on V1 PV→E (post-Stage-0)."""
    ring.pv_to_e.active = False


def _freeze_h_ring_plasticity(ring: HRing) -> None:
    """Freeze H-ring plasticity (pair-STDP on ee; Vogels on inh→E).

    Synaptic drive (AMPA + NMDA co-release on ee; inhibitory current on
    inh→E) still flows; only the weight-update terms are zeroed via
    namespace mutation. This preserves the Stage-1 bump-attractor
    dynamics while keeping the weights fixed as Lead's gate requires
    (`|Δw|/max(|w|) < 0.01`).
    """
    ring.ee.namespace["A_plus_eff"] = 0.0
    ring.ee.namespace["A_minus_eff"] = 0.0
    ring.inh_to_e.namespace["eta_eff"] = 0.0


def _load_stage0_v1(ring: V1Ring, ckpt_path: str) -> Tuple[float, int]:
    """Restore V1 E bias + PV→E weights from Stage-0 checkpoint.

    Returns
    -------
    (bias_pA, n_pv_to_e_synapses)
    """
    data = np.load(ckpt_path)
    bias = float(data["bias_pA"])
    ring.e.I_bias = bias * pA
    pv_w = np.asarray(data["pv_to_e_w"], dtype=np.float64)
    live = np.asarray(ring.pv_to_e.w[:])
    if pv_w.shape != live.shape:
        raise ValueError(
            f"Stage-0 pv_to_e_w shape {pv_w.shape} != live {live.shape}"
        )
    ring.pv_to_e.w[:] = pv_w
    return bias, pv_w.size


def _load_stage1_hr(ring: HRing, ckpt_path: str) -> int:
    """Restore H_R E→E weights from Stage-1 H_R checkpoint."""
    data = np.load(ckpt_path)
    ee_w = np.asarray(data["ee_w_final"], dtype=np.float64)
    live = np.asarray(ring.ee.w[:])
    if ee_w.shape != live.shape:
        raise ValueError(
            f"Stage-1 H_R ee_w_final shape {ee_w.shape} != live {live.shape}"
        )
    ring.ee.w[:] = ee_w
    return ee_w.size


def _build_v1_to_hr_teacher(
    v1_ring: V1Ring,
    h_ring: HRing,
    teacher_w: float = STAGE2_TEACHER_W,
    drive_pA: float = STAGE2_TEACHER_DRIVE_PA,
) -> Synapses:
    """Fixed channel-matched V1 E → H_R E teacher projection.

    Each V1 E cell in channel c connects to every H_R E cell in the same
    channel c (V1 and H share 12-channel 15°-spacing geometry). Weights
    are fixed (not plastic). During the grating epoch, V1 activity at
    the grating θ drives the matched H_R channel strongly enough to
    evoke a bump; during cue / gap / ITI windows V1 is silent so the
    teacher contributes nothing.
    """
    teacher = Synapses(
        v1_ring.e, h_ring.e,
        model="w : 1",
        on_pre=f"I_e_post += w * {drive_pA}*pA",
        name="s2_v1_to_hr_teacher",
    )
    i_list: List[int] = []
    j_list: List[int] = []
    for c in range(V1_N_CHANNELS):
        for i_v in range(c * V1_N_E_PER, (c + 1) * V1_N_E_PER):
            for j_h in range(c * H_N_E_PER, (c + 1) * H_N_E_PER):
                i_list.append(i_v)
                j_list.append(j_h)
    teacher.connect(
        i=np.asarray(i_list, dtype=np.int64),
        j=np.asarray(j_list, dtype=np.int64),
    )
    teacher.w = teacher_w
    return teacher


def _build_stage2_cue_pathway(
    h_ring: HRing,
) -> Tuple[Tuple[PoissonGroup, PoissonGroup], Tuple[Synapses, Synapses]]:
    """Build the two cue populations + plastic eligibility-trace synapses.

    cue_A (Poisson, 32 afferents) and cue_B (Poisson, 32 afferents) each
    project all-to-all to all 192 H_R E cells via
    `eligibility_trace_cue_rule` (tau_elig=1500 ms, lr=0.08, w_init=0.2,
    w_max=2.0). Learning is driven by post-spikes: the V1→H_R teacher
    fires the matched channel on valid trials while elig from the
    earlier cue is still non-zero, producing LTP on the matched edges.
    """
    cue_A = PoissonGroup(
        STAGE2_N_CUE_AFFERENTS, rates=0 * Hz, name="s2_cue_A",
    )
    cue_B = PoissonGroup(
        STAGE2_N_CUE_AFFERENTS, rates=0 * Hz, name="s2_cue_B",
    )
    elig_A = eligibility_trace_cue_rule(
        cue_A, h_ring.e,
        connectivity="True",
        w_init=STAGE2_W_INIT,
        w_max=STAGE2_W_MAX,
        tau_elig=STAGE2_TAU_ELIG_MS * ms,
        learning_rate=STAGE2_LR,
        drive_amp_pA=STAGE2_CUE_DRIVE_PA,
        name="s2_cue_elig_A",
    )
    elig_B = eligibility_trace_cue_rule(
        cue_B, h_ring.e,
        connectivity="True",
        w_init=STAGE2_W_INIT,
        w_max=STAGE2_W_MAX,
        tau_elig=STAGE2_TAU_ELIG_MS * ms,
        learning_rate=STAGE2_LR,
        drive_amp_pA=STAGE2_CUE_DRIVE_PA,
        name="s2_cue_elig_B",
    )
    return (cue_A, cue_B), (elig_A, elig_B)


def _stage2_trial_plan(
    rng: np.random.Generator,
    n_trials: int,
    valid_frac: float = STAGE2_VALID_FRAC,
) -> Dict[str, np.ndarray]:
    """Balanced training schedule: 50% cue_A / 50% cue_B, shuffled.

    Per cue: `valid_frac` of trials present the cue-matched grating θ
    (e.g. cue_A + 45° grating); the rest present the orthogonal θ
    (e.g. cue_A + 135° grating).
    """
    if n_trials % 2 != 0:
        raise ValueError(f"n_trials must be even (got {n_trials})")
    half = n_trials // 2
    cue_label = np.concatenate([
        np.zeros(half, dtype=np.int64),
        np.ones(half, dtype=np.int64),
    ])
    valid = np.zeros(n_trials, dtype=bool)
    n_valid_per_cue = int(round(half * valid_frac))
    idx_A = np.where(cue_label == 0)[0]
    idx_B = np.where(cue_label == 1)[0]
    valid[rng.choice(idx_A, size=n_valid_per_cue, replace=False)] = True
    valid[rng.choice(idx_B, size=n_valid_per_cue, replace=False)] = True
    perm = rng.permutation(n_trials)
    return {
        "cue_label": cue_label[perm],
        "valid": valid[perm],
    }


def run_stage_2_cue(
    seed: int = 42,
    n_train_trials: int = 200,
    stage0_ckpt: Optional[str] = None,
    stage1_hr_ckpt: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    v1_cfg: Optional[V1RingConfig] = None,
    h_cfg: Optional[HRingConfig] = None,
    verbose: bool = True,
) -> Stage2Result:
    """Stage-2 driver: cue learning on H_R (plan §3, Sprint-4).

    Workflow
    --------
    1. Build V1 ring + H_R ring; load Stage-0 (V1 bias + PV→E weights)
       and Stage-1 H_R (E→E recurrent weights) from checkpoints.
    2. Freeze ALL existing plasticity:
        - V1 PV→E Vogels iSTDP  →  `pv_to_e.active=False`
        - H_R E→E pair-STDP     →  `A_plus_eff = A_minus_eff = 0`
        - H_R inh→E Vogels iSTDP→  `eta_eff = 0`
    3. Wire plastic cue pathway: 2 Poisson cue populations → H_R E
       all-to-all via `eligibility_trace_cue_rule` (τ=1500 ms, lr=0.08).
    4. Wire fixed V1 → H_R teacher: channel-matched within-channel
       all-to-all (V1 E ch c → H_R E ch c).
    5. Run `n_train_trials` trials of
       (500 ms cue → 500 ms gap → 500 ms grating → 2500 ms ITI).
       cue_A → 45° (ch 3); cue_B → 135° (ch 9). 75% of trials are valid
       (grating matches cue's predicted θ); 25% are invalid (orthogonal).
    6. Post-training: 40 held-out cue-alone probes (20 per cue) measure
       matched- vs unmatched-channel H_R rates over the 500 ms cue
       window.
    7. Aggregate Stage-2 gate:
        - cue_selectivity_d (d ≥ 0.2, bootstrap 95% CI > 0)
        - bump_evocation_frac (matched rate > 5 Hz AND matched = peak,
          ≥ 80 pct of probes)
        - hr_weights_unchanged (max|Δw|/max|w_before| < 0.01)
        - no_runaway (H_E + H_inh rates < 80 Hz during training)
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    v1_cfg = v1_cfg or V1RingConfig()
    h_cfg = _stage1_h_cfg(h_cfg)

    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR_DEFAULT
    stage0_ckpt = stage0_ckpt or os.path.join(
        ckpt_dir, f"stage_0_seed{seed}.npz",
    )
    stage1_hr_ckpt = stage1_hr_ckpt or os.path.join(
        ckpt_dir, f"stage_1_hr_seed{seed}.npz",
    )
    for p in (stage0_ckpt, stage1_hr_ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Stage-2 requires upstream checkpoint: {p}"
            )

    log: List[str] = [
        f"Stage-2 driver: seed={seed}, n_train_trials={n_train_trials}",
        f"  Stage-0 ckpt: {stage0_ckpt}",
        f"  Stage-1 H_R ckpt: {stage1_hr_ckpt}",
        f"  Trial: {STAGE2_CUE_MS:.0f}ms cue → {STAGE2_GAP_MS:.0f}ms gap → "
        f"{STAGE2_GRATING_MS:.0f}ms grating → {STAGE2_ITI_MS:.0f}ms ITI",
        f"  Cues: A→{STAGE2_CUE_CHANNELS[0]*15}° (ch{STAGE2_CUE_CHANNELS[0]}), "
        f"B→{STAGE2_CUE_CHANNELS[1]*15}° (ch{STAGE2_CUE_CHANNELS[1]});  "
        f"valid_frac={STAGE2_VALID_FRAC:.2f}",
    ]

    # ---- Build + load + freeze ---------------------------------------
    v1_ring = build_v1_ring(config=v1_cfg, name_prefix="v1_s2")
    h_ring = build_h_r(config=h_cfg)
    silence_cue(h_ring)   # built-in fixed cue pathway silenced all run

    bias_pA_, pv_w_n = _load_stage0_v1(v1_ring, stage0_ckpt)
    ee_n = _load_stage1_hr(h_ring, stage1_hr_ckpt)
    log.append(f"  loaded: V1 I_bias={bias_pA_:.1f} pA, pv→e.w n={pv_w_n}; "
               f"H_R ee.w n={ee_n}")

    _freeze_v1_ring_plasticity(v1_ring)
    _freeze_h_ring_plasticity(h_ring)
    log.append("  froze: V1 pv→e iSTDP, H_R ee pair-STDP, H_R inh→e Vogels")

    # ---- Plastic cue pathway -----------------------------------------
    (cue_A, cue_B), (elig_A, elig_B) = _build_stage2_cue_pathway(h_ring)

    # Teacher: direct DC bias injection on matched-channel H_E cells
    # during the grating epoch. V1→H_R Poisson teacher (broad due to V1
    # ±15° tuning width) is replaced here because it produced cross-
    # channel LTP that saturated weights. V1 ring is still built/loaded/
    # frozen but dropped from the simulation Network.
    matched_cells_by_cue = {
        0: np.arange(STAGE2_CUE_CHANNELS[0] * H_N_E_PER,
                     (STAGE2_CUE_CHANNELS[0] + 1) * H_N_E_PER, dtype=np.int64),
        1: np.arange(STAGE2_CUE_CHANNELS[1] * H_N_E_PER,
                     (STAGE2_CUE_CHANNELS[1] + 1) * H_N_E_PER, dtype=np.int64),
    }

    # Snapshot H_R ee.w BEFORE training for drift check.
    ee_w_before = np.asarray(h_ring.ee.w[:], dtype=np.float64).copy()

    # ---- Monitors + Network ------------------------------------------
    h_e_mon = SpikeMonitor(h_ring.e, name="s2_h_e")
    h_inh_mon = SpikeMonitor(h_ring.inh, name="s2_h_inh")

    # V1 ring NOT in Network: teacher is direct injection.
    net = Network(
        *h_ring.groups,
        cue_A, cue_B, elig_A, elig_B,
        h_e_mon, h_inh_mon,
    )

    # ---- Training schedule -------------------------------------------
    plan = _stage2_trial_plan(rng, n_train_trials, STAGE2_VALID_FRAC)
    trial_ms = (STAGE2_CUE_MS + STAGE2_GAP_MS
                + STAGE2_GRATING_MS + STAGE2_ITI_MS)
    log.append(f"  trials: {n_train_trials} "
               f"(valid={int(plan['valid'].sum())}, "
               f"invalid={int((~plan['valid']).sum())}); "
               f"sim-time budget = {trial_ms*n_train_trials/1000.0:.0f} s")

    train_start_abs_ms = float(net.t / ms)
    t_wall0 = time.time()
    for k in range(n_train_trials):
        cue_idx = int(plan["cue_label"][k])
        is_valid = bool(plan["valid"][k])
        theta_cue_rad = STAGE2_CUE_CHANNELS[cue_idx] * (np.pi / H_N_CHANNELS)
        theta_orth_rad = STAGE2_CUE_CHANNELS[1 - cue_idx] * (np.pi / H_N_CHANNELS)
        grating_theta_rad = theta_cue_rad if is_valid else theta_orth_rad

        # (0) per-trial reset: kill any lingering bump + elig from prior
        # trial so each trial starts with H_R at resting state and no
        # residual eligibility (prevents cross-trial LTP bleed-through).
        h_ring.e.V = -70.0 * mV
        h_ring.e.I_e = 0 * pA
        h_ring.e.I_i = 0 * pA
        h_ring.e.g_nmda_h = 0 * nS
        h_ring.inh.V = -65.0 * mV
        h_ring.inh.I_e = 0 * pA
        h_ring.inh.I_i = 0 * pA
        elig_A.elig = 0.0
        elig_B.elig = 0.0

        # (1) cue epoch: fire the selected cue; teacher off
        if cue_idx == 0:
            cue_A.rates = STAGE2_CUE_ACTIVE_HZ * Hz
            cue_B.rates = 0 * Hz
        else:
            cue_A.rates = 0 * Hz
            cue_B.rates = STAGE2_CUE_ACTIVE_HZ * Hz
        net.run(STAGE2_CUE_MS * ms)

        # (2) gap: everything silent
        cue_A.rates = 0 * Hz
        cue_B.rates = 0 * Hz
        net.run(STAGE2_GAP_MS * ms)

        # (3) grating: teacher fires matched channel via DC bias injection
        # (channel determined by validity: valid -> cue's matched channel,
        # invalid -> orthogonal channel)
        matched_ch_this_trial = (
            STAGE2_CUE_CHANNELS[cue_idx] if is_valid
            else STAGE2_CUE_CHANNELS[1 - cue_idx]
        )
        teacher_cells = np.arange(
            matched_ch_this_trial * H_N_E_PER,
            (matched_ch_this_trial + 1) * H_N_E_PER, dtype=np.int64,
        )
        h_ring.e.I_bias[teacher_cells] = STAGE2_TEACHER_BIAS_PA * pA
        net.run(STAGE2_GRATING_MS * ms)

        # (4) ITI: teacher off
        h_ring.e.I_bias[teacher_cells] = 0 * pA
        net.run(STAGE2_ITI_MS * ms)

        step = max(1, n_train_trials // 10)
        if verbose and (k + 1) % step == 0:
            elapsed = time.time() - t_wall0
            # diagnostic: mean cue_A weight on matched (ch3) vs unmatched (ch9)
            w_A = np.asarray(elig_A.w[:], dtype=np.float64)
            w_A_mat = w_A.reshape(STAGE2_N_CUE_AFFERENTS, -1)
            w_A_per_h = w_A_mat.mean(axis=0)
            wA_match = float(w_A_per_h[h_ring.e_channel == STAGE2_CUE_CHANNELS[0]].mean())
            wA_unmatch = float(w_A_per_h[h_ring.e_channel == STAGE2_CUE_CHANNELS[1]].mean())
            print(f"    trial {k+1:4d}/{n_train_trials}  wall={elapsed:6.0f}s  "
                  f"cue={'A' if cue_idx == 0 else 'B'}  valid={is_valid}  "
                  f"grating_θ={np.rad2deg(grating_theta_rad):5.1f}°  "
                  f"wA_matched={wA_match:.3f}  wA_unmatched={wA_unmatch:.3f}")

    train_wall_s = time.time() - t_wall0
    train_end_abs_ms = float(net.t / ms)
    log.append(f"  training wall-clock: {train_wall_s:.0f} s "
               f"({train_wall_s / n_train_trials:.2f} s/trial)")

    # ---- Training-window runaway guard -------------------------------
    e_t = np.asarray(h_e_mon.t / ms, dtype=np.float64)
    inh_t = np.asarray(h_inh_mon.t / ms, dtype=np.float64)
    train_dur_s = max((train_end_abs_ms - train_start_abs_ms) / 1000.0, 1e-6)
    n_e_train = int(((e_t >= train_start_abs_ms) & (e_t < train_end_abs_ms)).sum())
    n_inh_train = int(((inh_t >= train_start_abs_ms) & (inh_t < train_end_abs_ms)).sum())
    h_e_rate = n_e_train / (len(h_ring.e) * train_dur_s)
    h_inh_rate = n_inh_train / (len(h_ring.inh) * train_dur_s)
    log.append(f"  train H rates: E={h_e_rate:.2f} Hz, inh={h_inh_rate:.2f} Hz")

    # ---- Cue-alone probes --------------------------------------------
    n_probes_total = 2 * STAGE2_N_PROBES_PER_CUE
    probe_cue_order = np.concatenate([
        np.zeros(STAGE2_N_PROBES_PER_CUE, dtype=np.int64),
        np.ones(STAGE2_N_PROBES_PER_CUE, dtype=np.int64),
    ])
    probe_cue_order = probe_cue_order[rng.permutation(n_probes_total)]

    probe_starts: List[float] = []
    probe_labels: List[int] = []
    for k in range(n_probes_total):
        # Quiet gap before probe to drain any lingering bump, then full
        # state reset so prior probe's bump cannot contaminate this one.
        cue_A.rates = 0 * Hz
        cue_B.rates = 0 * Hz
        net.run(STAGE2_PROBE_GAP_MS * ms)
        h_ring.e.V = -70.0 * mV
        h_ring.e.I_e = 0 * pA
        h_ring.e.I_i = 0 * pA
        h_ring.e.g_nmda_h = 0 * nS
        h_ring.inh.V = -65.0 * mV
        h_ring.inh.I_e = 0 * pA
        h_ring.inh.I_i = 0 * pA
        t_probe_start = float(net.t / ms)
        cue_idx = int(probe_cue_order[k])
        if cue_idx == 0:
            cue_A.rates = STAGE2_CUE_ACTIVE_HZ * Hz
        else:
            cue_B.rates = STAGE2_CUE_ACTIVE_HZ * Hz
        net.run(STAGE2_CUE_MS * ms)
        cue_A.rates = 0 * Hz
        cue_B.rates = 0 * Hz
        probe_starts.append(t_probe_start)
        probe_labels.append(cue_idx)

    probe_starts_arr = np.asarray(probe_starts, dtype=np.float64)
    probe_labels_arr = np.asarray(probe_labels, dtype=np.int64)

    # ---- Per-probe per-channel H_R rates ------------------------------
    e_spike_i = np.asarray(h_e_mon.i[:], dtype=np.int64)
    e_spike_t = np.asarray(h_e_mon.t / ms, dtype=np.float64)
    e_channel = h_ring.e_channel

    matched_rates = np.zeros(n_probes_total, dtype=np.float64)
    unmatched_rates = np.zeros(n_probes_total, dtype=np.float64)
    peak_channels = np.zeros(n_probes_total, dtype=np.int64)
    matched_channels = np.zeros(n_probes_total, dtype=np.int64)

    for k in range(n_probes_total):
        t0 = probe_starts_arr[k]
        t1 = t0 + STAGE2_CUE_MS
        cue_idx = int(probe_labels_arr[k])
        matched_ch = STAGE2_CUE_CHANNELS[cue_idx]
        unmatched_ch = STAGE2_CUE_CHANNELS[1 - cue_idx]
        per_ch_rate = _per_channel_rate_in_window(
            e_spike_i, e_spike_t, e_channel,
            t0, t1, H_N_CHANNELS, H_N_E_PER,
        )
        matched_rates[k] = per_ch_rate[matched_ch]
        unmatched_rates[k] = per_ch_rate[unmatched_ch]
        peak_channels[k] = int(np.argmax(per_ch_rate))
        matched_channels[k] = matched_ch

    log.append(
        f"  probes: {n_probes_total} total "
        f"(A={int((probe_labels_arr == 0).sum())}, "
        f"B={int((probe_labels_arr == 1).sum())}); "
        f"matched mean={matched_rates.mean():.2f} Hz, "
        f"unmatched mean={unmatched_rates.mean():.2f} Hz"
    )

    # ---- Weight drift ------------------------------------------------
    ee_w_after = np.asarray(h_ring.ee.w[:], dtype=np.float64)
    max_drift = float(np.max(np.abs(ee_w_after - ee_w_before)))
    log.append(f"  ee.w drift: max|Δw| = {max_drift:.3e}")

    # ---- Cue weight diagnostics --------------------------------------
    cue_A_w = np.asarray(elig_A.w[:], dtype=np.float64)
    cue_B_w = np.asarray(elig_B.w[:], dtype=np.float64)
    # eligibility cue→H is all-to-all, flat array N_cue * N_h_e.
    # Reshape as (N_cue, N_h_e) to inspect channel-specific means.
    cue_A_w_mat = cue_A_w.reshape(STAGE2_N_CUE_AFFERENTS, -1)
    cue_B_w_mat = cue_B_w.reshape(STAGE2_N_CUE_AFFERENTS, -1)
    # Mean incoming cue→H weight per post-H-cell, re-indexed by channel.
    mean_w_A_per_h = cue_A_w_mat.mean(axis=0)
    mean_w_B_per_h = cue_B_w_mat.mean(axis=0)
    mean_w_A_per_ch = np.array([
        mean_w_A_per_h[e_channel == c].mean() for c in range(H_N_CHANNELS)
    ])
    mean_w_B_per_ch = np.array([
        mean_w_B_per_h[e_channel == c].mean() for c in range(H_N_CHANNELS)
    ])
    log.append(
        f"  cue_A w per-ch: matched ch{STAGE2_CUE_CHANNELS[0]}="
        f"{mean_w_A_per_ch[STAGE2_CUE_CHANNELS[0]]:.3f}, "
        f"unmatched ch{STAGE2_CUE_CHANNELS[1]}="
        f"{mean_w_A_per_ch[STAGE2_CUE_CHANNELS[1]]:.3f}"
    )
    log.append(
        f"  cue_B w per-ch: matched ch{STAGE2_CUE_CHANNELS[1]}="
        f"{mean_w_B_per_ch[STAGE2_CUE_CHANNELS[1]]:.3f}, "
        f"unmatched ch{STAGE2_CUE_CHANNELS[0]}="
        f"{mean_w_B_per_ch[STAGE2_CUE_CHANNELS[0]]:.3f}"
    )

    # ---- Gate checks + aggregation -----------------------------------
    sel_check = check_cue_selectivity(matched_rates, unmatched_rates)
    bump_check = check_bump_fraction(
        matched_rates, peak_channels, matched_channels,
    )
    weights_check = check_hr_weights_unchanged(ee_w_before, ee_w_after)
    runaway_check = check_no_runaway_s2(
        {"h_e": h_e_rate, "h_inh": h_inh_rate},
    )
    checks = {
        "cue_selectivity_d": sel_check,
        "bump_evocation_frac": bump_check,
        "hr_weights_unchanged": weights_check,
        "no_runaway": runaway_check,
    }
    report = aggregate_s2(checks)
    log.append(report.summary())

    total_wall = time.time() - t_wall0
    log.append(f"Stage-2 elapsed: {total_wall:.0f} s wall-clock")
    if verbose:
        print("\n".join(log))

    # ---- Checkpoint --------------------------------------------------
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"stage_2_seed{seed}.npz")
    np.savez(
        ckpt_path,
        seed=np.int32(seed),
        n_train_trials=np.int32(n_train_trials),
        cue_label=plan["cue_label"],
        valid_mask=plan["valid"],
        cue_A_w_final=cue_A_w,
        cue_B_w_final=cue_B_w,
        cue_A_w_per_ch=mean_w_A_per_ch,
        cue_B_w_per_ch=mean_w_B_per_ch,
        matched_rates=matched_rates,
        unmatched_rates=unmatched_rates,
        peak_channels=peak_channels,
        matched_channels=matched_channels,
        probe_cue_labels=probe_labels_arr,
        ee_w_before=ee_w_before,
        ee_w_after=ee_w_after,
        h_e_rate_hz=np.float64(h_e_rate),
        h_inh_rate_hz=np.float64(h_inh_rate),
        max_ee_w_drift=np.float64(max_drift),
        passed=np.bool_(report.passed),
    )
    if verbose:
        print(f"checkpoint saved: {ckpt_path}")

    return Stage2Result(
        seed=seed, report=report,
        v1_cfg=v1_cfg, h_cfg=h_cfg,
        diagnostics={
            "h_e_rate_hz": h_e_rate,
            "h_inh_rate_hz": h_inh_rate,
            "max_ee_w_drift": max_drift,
            "mean_matched_rate_hz": float(matched_rates.mean()),
            "mean_unmatched_rate_hz": float(unmatched_rates.mean()),
            "sim_wall_s": total_wall,
        },
        checkpoint_path=ckpt_path,
    )


# -- CLI entry -------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training drivers (Stage 0 / Stage 1 / Stage 2)")
    p.add_argument("--stage", type=str, default="0",
                   choices=["0", "1_hr", "1_ht", "1_both", "2"])
    p.add_argument("--seed", type=int, default=42)
    # Stage 0 args
    p.add_argument("--target_e_rate", type=float, default=4.0)
    p.add_argument("--bias_probe_ms", type=float, default=600.0)
    p.add_argument("--istdp_settle_ms", type=float, default=3000.0)
    p.add_argument("--final_probe_ms", type=float, default=600.0)
    # Stage 1 args
    p.add_argument("--n_trials", type=int, default=72,
                   help="Stage-1 H_R trial count (multiple of 36)")
    p.add_argument("--n_items", type=int, default=500,
                   help="Stage-1 H_T item count")
    p.add_argument("--iti_ms", type=float, default=1500.0,
                   help="Stage-1 H_R ITI (<< paper 5000 for wall-clock)")
    # Stage 2 args
    p.add_argument("--n_train_trials", type=int, default=200,
                   help="Stage-2 cue training trial count (even)")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    if args.stage == "0":
        result = run_stage_0(
            seed=args.seed,
            target_e_rate=args.target_e_rate,
            bias_probe_ms=args.bias_probe_ms,
            istdp_settle_ms=args.istdp_settle_ms,
            final_probe_ms=args.final_probe_ms,
            checkpoint_dir=args.checkpoint_dir,
            verbose=True,
        )
        passed = result.report.passed
        name = "Stage-0"
    elif args.stage == "1_hr":
        result = run_stage_1_hr(
            seed=args.seed, n_trials=args.n_trials,
            iti_ms=args.iti_ms,
            checkpoint_dir=args.checkpoint_dir, verbose=True,
        )
        passed = result.report.passed
        name = "Stage-1 H_R"
    elif args.stage == "1_ht":
        result = run_stage_1_ht(
            seed=args.seed, n_items=args.n_items,
            checkpoint_dir=args.checkpoint_dir, verbose=True,
        )
        passed = result.report.passed
        name = "Stage-1 H_T"
    elif args.stage == "1_both":
        r1 = run_stage_1_hr(
            seed=args.seed, n_trials=args.n_trials,
            iti_ms=args.iti_ms,
            checkpoint_dir=args.checkpoint_dir, verbose=True,
        )
        r2 = run_stage_1_ht(
            seed=args.seed, n_items=args.n_items,
            checkpoint_dir=args.checkpoint_dir, verbose=True,
        )
        passed = r1.report.passed and r2.report.passed
        name = "Stage-1 both"
    elif args.stage == "2":
        result = run_stage_2_cue(
            seed=args.seed,
            n_train_trials=args.n_train_trials,
            checkpoint_dir=args.checkpoint_dir,
            verbose=True,
        )
        passed = result.report.passed
        name = "Stage-2"
    else:
        raise SystemExit(f"Unknown stage: {args.stage}")
    if passed:
        print(f"{name} GATE: ALL PASS")
        raise SystemExit(0)
    print(f"{name} GATE: FAILED")
    raise SystemExit(2)
