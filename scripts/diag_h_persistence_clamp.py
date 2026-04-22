"""Diagnostic H-ring persistence clamp for Stage-1 failure isolation.

This script does not edit production code. It reproduces the H-only
Stage-1-style schedule under several inhibition configurations, then probes
post-pulse bump persistence. The goal is to test whether the 10 ms
persistence failure is caused by inhibitory plasticity / config overrides or
by the recurrent substrate itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from brian2 import (  # noqa: E402
    Hz,
    Network,
    SpikeMonitor,
    defaultclock,
    ms,
    prefs,
    seed as b2_seed,
)

from expectation_snn.brian2_model.h_ring import (  # noqa: E402
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
    HRingConfig,
    build_h_r,
    silence_cue,
)
from expectation_snn.brian2_model.stimulus import (  # noqa: E402
    richter_biased_training_schedule,
)
from expectation_snn.brian2_model.train import (  # noqa: E402
    _drive_h_broad_noise,
    _drive_h_cue_gaussian,
    _make_postsyn_normalizer,
    _peak_channel_ms_series,
    _per_channel_rate_in_window,
    _stage1_h_cfg,
)
from expectation_snn.validation.stage_1_gate import (  # noqa: E402
    compute_bump_persistence_ms,
)


def _jsonify(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def make_cfg(name: str) -> HRingConfig:
    """Return one diagnostic H-ring configuration."""
    cfg = _stage1_h_cfg(HRingConfig())
    if name == "helper_current":
        return cfg
    if name == "helper_current_inh_eta0":
        cfg.inh_eta = 0.0
        return cfg
    if name == "intended_20_10":
        cfg.inh_rho_hz = 20.0
        cfg.inh_w_max = 10.0
        return cfg
    raise ValueError(f"unknown config {name!r}")


def run_config(
    *,
    seed: int,
    config_name: str,
    n_trials: int,
    presettle_ms: float,
    leader_ms: float,
    trailer_ms: float,
    iti_ms: float,
    pulse_ms: float,
    post_ms: float,
) -> Dict[str, object]:
    """Run one H-only persistence diagnostic arm."""
    from brian2 import start_scope

    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    cfg = make_cfg(config_name)
    plan = richter_biased_training_schedule(
        rng,
        n_trials=n_trials,
        p_bias=0.80,
        derangement=(1, 2, 3, 4, 5, 0),
        leader_ms=leader_ms,
        trailer_ms=trailer_ms,
        iti_ms=iti_ms,
    )

    ring = build_h_r(config=cfg)
    silence_cue(ring)
    e_mon = SpikeMonitor(ring.e, name=f"{config_name}_e_mon")
    inh_mon = SpikeMonitor(ring.inh, name=f"{config_name}_inh_mon")
    ee_norm = _make_postsyn_normalizer(
        ring.ee,
        target_sum=cfg.target_postsyn_sum,
        dt_ms=200.0,
        name=f"{config_name}_ee_norm",
    )
    net = Network(*ring.groups, e_mon, inh_mon, ee_norm)

    if presettle_ms > 0:
        _drive_h_broad_noise(ring, mean_rate_hz=40.0)
        net.run(presettle_ms * ms)
        silence_cue(ring)

    schedule_start_abs_ms = float(net.t / ms)
    t_wall0 = time.time()
    for item in plan.items:
        if item.kind == "iti" or item.theta_rad is None:
            silence_cue(ring)
        else:
            _drive_h_cue_gaussian(
                ring,
                item.theta_rad,
                peak_rate_hz=300.0,
                sigma_deg=15.0,
            )
        net.run(item.duration_ms * ms)
    sim_wall_s = time.time() - t_wall0
    silence_cue(ring)

    sched_end_ms = schedule_start_abs_ms + plan.total_ms
    total_sim_s = plan.total_ms / 1000.0
    e_t_ms = np.asarray(e_mon.t / ms, dtype=np.float64)
    inh_t_ms = np.asarray(inh_mon.t / ms, dtype=np.float64)
    e_sched = int(((e_t_ms >= schedule_start_abs_ms) & (e_t_ms < sched_end_ms)).sum())
    inh_sched = int(((inh_t_ms >= schedule_start_abs_ms) & (inh_t_ms < sched_end_ms)).sum())
    e_rate = float(e_sched) / (len(ring.e) * total_sim_s)
    inh_rate = float(inh_sched) / (len(ring.inh) * total_sim_s)

    net.run(200 * ms)
    t_pulse_start = float(net.t / ms)
    _drive_h_cue_gaussian(ring, 0.0, peak_rate_hz=300.0, sigma_deg=15.0)
    net.run(pulse_ms * ms)
    t_pulse_end = float(net.t / ms)
    silence_cue(ring)
    net.run(post_ms * ms)

    spike_i = np.asarray(e_mon.i[:], dtype=np.int64)
    spike_t_ms = np.asarray(e_mon.t / ms, dtype=np.float64)
    peak_rates = _per_channel_rate_in_window(
        spike_i,
        spike_t_ms,
        ring.e_channel,
        t_pulse_end - 100.0,
        t_pulse_end,
        H_N_CHANNELS,
        H_N_E_PER,
    )
    peak_ch = int(np.argmax(peak_rates))
    bin_ms = 10.0
    series = _peak_channel_ms_series(
        spike_i,
        spike_t_ms,
        ring.e_channel,
        t_pulse_end,
        t_pulse_end + post_ms,
        peak_ch,
        H_N_E_PER,
        bin_ms=bin_ms,
    )
    persistence_ms = compute_bump_persistence_ms(
        series,
        offset_idx=0,
        dt_ms=bin_ms,
        floor_hz=2.0,
    )
    inh_w = np.asarray(ring.inh_to_e.w[:], dtype=np.float64)
    ee_w = np.asarray(ring.ee.w[:], dtype=np.float64)

    return {
        "config": config_name,
        "seed": seed,
        "n_trials": n_trials,
        "presettle_ms": presettle_ms,
        "schedule_total_ms": float(plan.total_ms),
        "sim_wall_s": float(sim_wall_s),
        "h_cfg": {
            "inh_rho_hz": float(cfg.inh_rho_hz),
            "inh_eta": float(cfg.inh_eta),
            "inh_w_max": float(cfg.inh_w_max),
            "w_ee_within_init": float(cfg.w_ee_within_init),
            "w_ee_cross_init": float(cfg.w_ee_cross_init),
            "nmda_drive_amp_nS": float(cfg.nmda_drive_amp_nS),
            "target_postsyn_sum": float(cfg.target_postsyn_sum),
        },
        "schedule_rates_hz": {
            "e": e_rate,
            "inh": inh_rate,
        },
        "probe": {
            "pulse_start_ms": t_pulse_start,
            "pulse_end_ms": t_pulse_end,
            "peak_ch": peak_ch,
            "peak_rate_last100ms_hz": float(peak_rates[peak_ch]),
            "persistence_ms": float(persistence_ms),
            "post_series_first20_hz": series[:20].tolist(),
        },
        "weights": {
            "inh_to_e_mean": float(inh_w.mean()),
            "inh_to_e_max": float(inh_w.max()),
            "ee_mean": float(ee_w.mean()),
            "ee_max": float(ee_w.max()),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=72)
    p.add_argument(
        "--configs",
        type=str,
        default="helper_current,helper_current_inh_eta0,intended_20_10",
    )
    p.add_argument("--presettle-ms", type=float, default=1000.0)
    p.add_argument("--leader-ms", type=float, default=500.0)
    p.add_argument("--trailer-ms", type=float, default=500.0)
    p.add_argument("--iti-ms", type=float, default=1500.0)
    p.add_argument("--pulse-ms", type=float, default=300.0)
    p.add_argument("--post-ms", type=float, default=1000.0)
    p.add_argument("--out", type=str, default="data/diag_h_persistence_clamp_seed42.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_trials % 6 != 0:
        raise ValueError("--n-trials must be a multiple of 6")
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    results = []
    for config_name in configs:
        print(f"=== H persistence config: {config_name} ===")
        res = run_config(
            seed=args.seed,
            config_name=config_name,
            n_trials=args.n_trials,
            presettle_ms=args.presettle_ms,
            leader_ms=args.leader_ms,
            trailer_ms=args.trailer_ms,
            iti_ms=args.iti_ms,
            pulse_ms=args.pulse_ms,
            post_ms=args.post_ms,
        )
        print(
            "  persistence_ms="
            f"{res['probe']['persistence_ms']:.1f} "
            "peak_rate="
            f"{res['probe']['peak_rate_last100ms_hz']:.2f}Hz "
            "schedule_E="
            f"{res['schedule_rates_hz']['e']:.2f}Hz "
            "inh_w_mean/max="
            f"{res['weights']['inh_to_e_mean']:.3f}/"
            f"{res['weights']['inh_to_e_max']:.3f}"
        )
        results.append(res)

    payload = {
        "script": "diag_h_persistence_clamp.py",
        "args": vars(args),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=_jsonify) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
