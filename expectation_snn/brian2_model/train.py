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
    pulse_channel,
    silence_cue,
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
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


# -- CLI entry -------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-0 calibration driver")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_e_rate", type=float, default=4.0)
    p.add_argument("--bias_probe_ms", type=float, default=600.0)
    p.add_argument("--istdp_settle_ms", type=float, default=3000.0)
    p.add_argument("--final_probe_ms", type=float, default=600.0)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    result = run_stage_0(
        seed=args.seed,
        target_e_rate=args.target_e_rate,
        bias_probe_ms=args.bias_probe_ms,
        istdp_settle_ms=args.istdp_settle_ms,
        final_probe_ms=args.final_probe_ms,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True,
    )
    if result.report.passed:
        print("Stage-0 GATE: ALL PASS")
        raise SystemExit(0)
    print("Stage-0 GATE: FAILED")
    raise SystemExit(2)
