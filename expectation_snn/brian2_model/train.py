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
    SpikeMonitor,
    defaultclock,
    Hz,
    ms,
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
    richter_crossover_training_schedule,
    tang_rotating_sequence,
    RICHTER_ORIENTATIONS_DEG,
    TANG_ORIENTATIONS_DEG,
)
from .plasticity import normalize_postsyn_sum
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
) -> Stage1Result:
    """Stage-1 driver for H_R (Richter crossover grammar).

    Workflow
    --------
    1. Build H_R, freeze Vogels iSTDP (inh->E frozen at init weights).
    2. Generate balanced Richter schedule (n_trials must be multiple of 36).
    3. Drive cue with per-item Gaussian-over-channels rate; H E<->E STDP
       is plastic throughout (Hebbian pair rule, `pair_stdp_with_normalization`).
    4. Post-hoc, per trial: H argmax at `trailer_offset + probe_delay_ms`.
    5. Aggregate gate checks: bump_persistence, transition_MI, no_runaway.

    The rotation MI check is not computed for this grammar (N/A for Richter).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    h_cfg = _stage1_h_cfg(h_cfg)
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate schedule.
    plan = richter_crossover_training_schedule(
        rng, n_trials=n_trials,
        leader_ms=leader_ms, trailer_ms=trailer_ms, iti_ms=iti_ms,
    )
    pairs = plan.meta["pairs"]               # (n_trials, 2)

    log: List[str] = [
        f"Stage-1 H_R driver: seed={seed}, n_trials={n_trials}, "
        f"ITI={iti_ms:.0f}ms, cue_peak={cue_peak_hz:.0f}Hz "
        f"sigma={cue_sigma_deg:.0f}deg",
        f"  schedule: {plan.total_ms/1000.0:.1f} s sim time, "
        f"{len(plan.items)} items",
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
    trailer_end_abs = trial_start_abs + leader_ms + trailer_ms    # (n_trials,)
    probe_win_start = trailer_end_abs + probe_delay_ms - probe_window_ms / 2.0
    probe_win_end   = trailer_end_abs + probe_delay_ms + probe_window_ms / 2.0

    # Use the LAST half of trials (plasticity has kicked in).
    if trials_used_tail is None:
        trials_used_tail = max(n_trials // 2, 36)
    start_k = max(0, n_trials - trials_used_tail)

    # Per-trial H argmax at +500ms after trailer offset, mapped to 6 orients.
    h_argmax_per_trial_6 = np.empty(trials_used_tail, dtype=np.int64)
    leader_idx_used = np.empty(trials_used_tail, dtype=np.int64)
    for kk, k in enumerate(range(start_k, n_trials)):
        rates_12 = _per_channel_rate_in_window(
            spike_i, spike_t_ms, e_channel,
            probe_win_start[k], probe_win_end[k],
            H_N_CHANNELS, H_N_E_PER,
        )
        rates_6 = rates_12[H_ORIENT_CHANNELS]      # (6,)
        h_argmax_per_trial_6[kk] = int(np.argmax(rates_6))
        leader_idx_used[kk] = int(pairs[k, 0])

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
    )
    runaway_check = check_no_runaway_s1(
        {"h_e": e_rate, "h_inh": inh_rate},
    )
    checks = {
        "h_bump_persistence_ms": bump_check,
        "h_transition_mi_bits": transition_check,
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


# -- CLI entry -------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training drivers (Stage 0 / Stage 1)")
    p.add_argument("--stage", type=str, default="0",
                   choices=["0", "1_hr", "1_ht", "1_both"])
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
    else:
        raise SystemExit(f"Unknown stage: {args.stage}")
    if passed:
        print(f"{name} GATE: ALL PASS")
        raise SystemExit(0)
    print(f"{name} GATE: FAILED")
    raise SystemExit(2)
