"""Diagnostic matrix for Stage-1 ctx->pred timing failure modes.

This script does not edit production code. It runs short Stage-1-like
ctx_pred training arms and compares:

- V1->H_pred teacher timing: always on vs trailer-only.
- Modulatory gate timing: current implementation vs onset-or-after gate.

The goal is to test whether H_prediction is leader-locked because it is
directly taught by the leader and/or because the M gate consumes eligibility
before trailer-driven post spikes are available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from brian2 import (  # noqa: E402
    Hz,
    Network,
    NetworkOperation,
    SpikeMonitor,
    defaultclock,
    ms,
    prefs,
    seed as b2_seed,
)

from expectation_snn.brian2_model.feedforward_v1_to_h import (  # noqa: E402
    V1ToHConfig,
    build_v1_to_h_feedforward,
)
from expectation_snn.brian2_model.h_context_prediction import (  # noqa: E402
    HContextPrediction,
    HContextPredictionConfig,
    apply_modulatory_update,
    build_h_context_prediction,
    make_modulatory_gate_operation,
    silence_direction,
)
from expectation_snn.brian2_model.h_ring import (  # noqa: E402
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
    HRingConfig,
    silence_cue,
)
from expectation_snn.brian2_model.stimulus import (  # noqa: E402
    richter_biased_training_schedule,
)
from expectation_snn.brian2_model.train import (  # noqa: E402
    CHECKPOINT_DIR_DEFAULT,
    _drive_h_broad_noise,
    _drive_h_cue_gaussian,
    _freeze_v1_ring_plasticity,
    _load_stage0_v1,
    _make_postsyn_normalizer,
    _per_channel_rate_in_window,
    _stage1_h_cfg,
)
from expectation_snn.brian2_model.v1_ring import (  # noqa: E402
    V1RingConfig,
    build_v1_ring,
    set_stimulus,
)


H_ORIENT_CHANNELS = np.arange(0, H_N_CHANNELS, 2, dtype=np.int64)


def _jsonify(x):
    """Convert numpy scalars/arrays into JSON-serializable objects."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _make_onset_or_after_gate_operation(
    bundle: HContextPrediction,
    trailer_onsets_ms: Sequence[float],
    *,
    dt_trial_s: float,
    dt_op_ms: float,
    log: List[dict],
) -> NetworkOperation:
    """Diagnostic gate that cannot fire before trailer onset."""
    cfg = bundle.config
    m_integral = cfg.m_amplitude * (2.0 * cfg.m_window_ms) * 1e-3
    onsets = np.asarray(trailer_onsets_ms, dtype=np.float64)
    consumed = np.zeros(onsets.shape, dtype=bool)
    half_w = float(cfg.m_window_ms)

    def _op() -> None:
        t_now = float(defaultclock.t / ms)
        remaining = np.flatnonzero(~consumed)
        if remaining.size == 0:
            return
        for k in remaining:
            t_on = float(onsets[k])
            if t_on <= t_now <= t_on + half_w:
                stats = apply_modulatory_update(
                    bundle,
                    m_integral=m_integral,
                    dt_trial_s=dt_trial_s,
                )
                log.append({"k": int(k), "t_ms": t_now, **stats})
                consumed[k] = True
                return
            if t_now < t_on:
                break

    return NetworkOperation(
        _op,
        dt=dt_op_ms * ms,
        when="end",
        name=f"{bundle.ctx.name}_{bundle.pred.name}_diag_onset_gate",
    )


def _weight_mapping_stats(
    bundle: HContextPrediction,
    derangement: Sequence[int],
) -> Dict[str, float]:
    """Summarize ctx->pred weights for leader->leader vs leader->expected."""
    syn = bundle.ctx_pred
    w = np.asarray(syn.w[:], dtype=np.float64)
    i_pre = np.asarray(syn.i[:], dtype=np.int64)
    j_post = np.asarray(syn.j[:], dtype=np.int64)
    ctx_ch = bundle.ctx.e_channel[i_pre]
    pred_ch = bundle.pred.e_channel[j_post]

    leader_means = []
    expected_means = []
    for leader_idx, expected_idx in enumerate(derangement):
        leader_ch = int(H_ORIENT_CHANNELS[leader_idx])
        expected_ch = int(H_ORIENT_CHANNELS[int(expected_idx)])
        leader_mask = (ctx_ch == leader_ch) & (pred_ch == leader_ch)
        expected_mask = (ctx_ch == leader_ch) & (pred_ch == expected_ch)
        leader_means.append(float(w[leader_mask].mean()))
        expected_means.append(float(w[expected_mask].mean()))

    return {
        "w_leader_to_leader_mean": float(np.mean(leader_means)),
        "w_leader_to_expected_mean": float(np.mean(expected_means)),
        "w_expected_minus_leader": float(np.mean(expected_means) - np.mean(leader_means)),
        "w_ctx_pred_mean": float(w.mean()),
        "w_ctx_pred_max": float(w.max()),
    }


def run_arm(
    *,
    seed: int,
    n_trials: int,
    teacher_mode: str,
    gate_mode: str,
    presettle_ms: float,
    leader_ms: float,
    trailer_ms: float,
    iti_ms: float,
    probe_window_ms: float,
    p_bias: float,
    v1_to_hctx_g: float,
    v1_to_hpred_g: float,
    m_gate_dt_ms: float,
    stage0_ckpt: str,
) -> Dict[str, object]:
    """Run one diagnostic arm and return compact evidence."""
    if teacher_mode not in {"always", "trailer_only"}:
        raise ValueError(f"unknown teacher_mode {teacher_mode!r}")
    if gate_mode not in {"current", "onset_or_after"}:
        raise ValueError(f"unknown gate_mode {gate_mode!r}")

    from brian2 import start_scope

    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    h_cfg = _stage1_h_cfg(HRingConfig())
    ctx_pred_cfg = HContextPredictionConfig(ctx_cfg=h_cfg, pred_cfg=h_cfg)
    v1_cfg = V1RingConfig()
    derangement = (1, 2, 3, 4, 5, 0)

    plan = richter_biased_training_schedule(
        rng,
        n_trials=n_trials,
        p_bias=p_bias,
        derangement=derangement,
        leader_ms=leader_ms,
        trailer_ms=trailer_ms,
        iti_ms=iti_ms,
    )
    pairs = np.asarray(plan.meta["pairs"], dtype=np.int64)
    expected_all = np.asarray(plan.meta["expected_trailer_idx"], dtype=np.int64)

    v1_ring = build_v1_ring(config=v1_cfg, name_prefix="v1_diag_cp")
    bias_pA, pv_n = _load_stage0_v1(v1_ring, stage0_ckpt)
    _freeze_v1_ring_plasticity(v1_ring)
    v1_ring.stim.rates = 0 * Hz

    bundle = build_h_context_prediction(
        config=ctx_pred_cfg,
        rng=rng,
        ctx_name="h_ctx_diag",
        pred_name="h_pred_diag",
    )
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)

    v1_hctx = build_v1_to_h_feedforward(
        v1_ring,
        bundle.ctx,
        config=V1ToHConfig(g_v1_to_h=v1_to_hctx_g),
        name_prefix="diag_v1_hctx",
    )
    v1_hpred = build_v1_to_h_feedforward(
        v1_ring,
        bundle.pred,
        config=V1ToHConfig(g_v1_to_h=v1_to_hpred_g),
        name_prefix="diag_v1_hpred",
    )
    if teacher_mode == "trailer_only":
        v1_hpred.set_active(False)

    ctx_e_mon = SpikeMonitor(bundle.ctx.e, name="diag_ctx_e_mon")
    ctx_inh_mon = SpikeMonitor(bundle.ctx.inh, name="diag_ctx_inh_mon")
    pred_e_mon = SpikeMonitor(bundle.pred.e, name="diag_pred_e_mon")
    pred_inh_mon = SpikeMonitor(bundle.pred.inh, name="diag_pred_inh_mon")
    ctx_ee_norm = _make_postsyn_normalizer(
        bundle.ctx.ee,
        target_sum=h_cfg.target_postsyn_sum,
        dt_ms=200.0,
        name="diag_ctx_ee_norm",
    )
    pred_ee_norm = _make_postsyn_normalizer(
        bundle.pred.ee,
        target_sum=h_cfg.target_postsyn_sum,
        dt_ms=200.0,
        name="diag_pred_ee_norm",
    )
    net = Network(
        *v1_ring.groups,
        *bundle.groups,
        *v1_hctx.groups,
        *v1_hpred.groups,
        ctx_ee_norm,
        pred_ee_norm,
        ctx_e_mon,
        ctx_inh_mon,
        pred_e_mon,
        pred_inh_mon,
    )

    if presettle_ms > 0:
        _drive_h_broad_noise(bundle.ctx, mean_rate_hz=40.0)
        net.run(presettle_ms * ms)
        silence_cue(bundle.ctx)

    schedule_start_abs_ms = float(net.t / ms)
    trial_ms = leader_ms + trailer_ms + iti_ms
    trailer_onsets_ms = (
        schedule_start_abs_ms
        + np.arange(n_trials, dtype=np.float64) * trial_ms
        + leader_ms
    )
    gate_log: List[dict] = []
    if gate_mode == "current":
        gate = make_modulatory_gate_operation(
            bundle,
            trailer_onsets_ms,
            dt_trial_s=trial_ms / 1000.0,
            dt_op_ms=m_gate_dt_ms,
            log=gate_log,
        )
    else:
        gate = _make_onset_or_after_gate_operation(
            bundle,
            trailer_onsets_ms,
            dt_trial_s=trial_ms / 1000.0,
            dt_op_ms=m_gate_dt_ms,
            log=gate_log,
        )
    net.add(gate)

    t_wall0 = time.time()
    for item in plan.items:
        if item.kind == "iti" or item.theta_rad is None:
            silence_cue(bundle.ctx)
            v1_ring.stim.rates = 0 * Hz
            if teacher_mode == "trailer_only":
                v1_hpred.set_active(False)
        else:
            _drive_h_cue_gaussian(
                bundle.ctx,
                item.theta_rad,
                peak_rate_hz=300.0,
                sigma_deg=15.0,
            )
            set_stimulus(v1_ring, theta_rad=item.theta_rad, contrast=1.0)
            if teacher_mode == "trailer_only":
                v1_hpred.set_active(item.kind == "trailer")
        net.run(item.duration_ms * ms)
    sim_wall_s = time.time() - t_wall0
    silence_cue(bundle.ctx)
    v1_ring.stim.rates = 0 * Hz
    if teacher_mode == "trailer_only":
        v1_hpred.set_active(False)

    sched_end_ms = schedule_start_abs_ms + plan.total_ms
    total_sim_s = plan.total_ms / 1000.0

    def _sched_rate(mon: SpikeMonitor, n_cells: int) -> float:
        t_ms = np.asarray(mon.t / ms, dtype=np.float64)
        n = int(((t_ms >= schedule_start_abs_ms) & (t_ms < sched_end_ms)).sum())
        return float(n) / (float(n_cells) * total_sim_s)

    ctx_rate = _sched_rate(ctx_e_mon, len(bundle.ctx.e))
    pred_rate = _sched_rate(pred_e_mon, len(bundle.pred.e))
    ctx_inh_rate = _sched_rate(ctx_inh_mon, len(bundle.ctx.inh))
    pred_inh_rate = _sched_rate(pred_inh_mon, len(bundle.pred.inh))

    pred_i = np.asarray(pred_e_mon.i[:], dtype=np.int64)
    pred_t = np.asarray(pred_e_mon.t / ms, dtype=np.float64)
    ctx_i = np.asarray(ctx_e_mon.i[:], dtype=np.int64)
    ctx_t = np.asarray(ctx_e_mon.t / ms, dtype=np.float64)

    trial_start_abs = schedule_start_abs_ms + np.arange(n_trials) * trial_ms
    leader_end_abs = trial_start_abs + leader_ms
    probe_start = leader_end_abs - probe_window_ms
    probe_end = leader_end_abs.copy()
    trials_used_tail = max(n_trials // 2, 36)
    start_k = max(0, n_trials - trials_used_tail)

    pred_arg = np.empty(n_trials - start_k, dtype=np.int64)
    ctx_arg = np.empty(n_trials - start_k, dtype=np.int64)
    leader_used = pairs[start_k:n_trials, 0].astype(np.int64)
    expected_used = expected_all[start_k:n_trials].astype(np.int64)
    for kk, k in enumerate(range(start_k, n_trials)):
        rates_pred_12 = _per_channel_rate_in_window(
            pred_i,
            pred_t,
            bundle.pred.e_channel,
            probe_start[k],
            probe_end[k],
            H_N_CHANNELS,
            H_N_E_PER,
        )
        rates_ctx_12 = _per_channel_rate_in_window(
            ctx_i,
            ctx_t,
            bundle.ctx.e_channel,
            probe_start[k],
            probe_end[k],
            H_N_CHANNELS,
            H_N_E_PER,
        )
        pred_arg[kk] = int(np.argmax(rates_pred_12[H_ORIENT_CHANNELS]))
        ctx_arg[kk] = int(np.argmax(rates_ctx_12[H_ORIENT_CHANNELS]))

    if gate_log:
        gate_k = np.asarray([g["k"] for g in gate_log], dtype=np.int64)
        gate_t = np.asarray([g["t_ms"] for g in gate_log], dtype=np.float64)
        gate_delta = gate_t - trailer_onsets_ms[gate_k]
    else:
        gate_delta = np.zeros(0, dtype=np.float64)

    out: Dict[str, object] = {
        "seed": seed,
        "n_trials": n_trials,
        "teacher_mode": teacher_mode,
        "gate_mode": gate_mode,
        "stage0_ckpt": stage0_ckpt,
        "stage0_bias_pA": float(bias_pA),
        "stage0_pv_weight_count": int(pv_n),
        "schedule_total_ms": float(plan.total_ms),
        "sim_wall_s": float(sim_wall_s),
        "n_gate_updates": int(len(gate_log)),
        "gate_delta_ms": {
            "min": float(gate_delta.min()) if gate_delta.size else None,
            "mean": float(gate_delta.mean()) if gate_delta.size else None,
            "max": float(gate_delta.max()) if gate_delta.size else None,
            "first10": gate_delta[:10].tolist(),
        },
        "rates_hz": {
            "ctx_e": ctx_rate,
            "pred_e": pred_rate,
            "ctx_inh": ctx_inh_rate,
            "pred_inh": pred_inh_rate,
        },
        "forecast": {
            "n_probe_trials": int(pred_arg.size),
            "p_pred_argmax_eq_leader": float(np.mean(pred_arg == leader_used)),
            "p_pred_argmax_eq_expected": float(np.mean(pred_arg == expected_used)),
            "p_ctx_argmax_eq_leader": float(np.mean(ctx_arg == leader_used)),
            "p_ctx_argmax_eq_expected": float(np.mean(ctx_arg == expected_used)),
            "pred_argmax_counts": np.bincount(pred_arg, minlength=6).tolist(),
            "ctx_argmax_counts": np.bincount(ctx_arg, minlength=6).tolist(),
        },
    }
    out.update(_weight_mapping_stats(bundle, derangement))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=72)
    p.add_argument("--teacher-modes", type=str, default="always,trailer_only")
    p.add_argument("--gate-modes", type=str, default="current,onset_or_after")
    p.add_argument("--presettle-ms", type=float, default=1000.0)
    p.add_argument("--leader-ms", type=float, default=500.0)
    p.add_argument("--trailer-ms", type=float, default=500.0)
    p.add_argument("--iti-ms", type=float, default=1500.0)
    p.add_argument("--probe-window-ms", type=float, default=100.0)
    p.add_argument("--p-bias", type=float, default=0.80)
    p.add_argument("--v1-to-hctx-g", type=float, default=1.5)
    p.add_argument("--v1-to-hpred-g", type=float, default=1.5)
    p.add_argument("--m-gate-dt-ms", type=float, default=5.0)
    p.add_argument("--stage0-ckpt", type=str, default="")
    p.add_argument("--out", type=str, default="data/diag_ctx_pred_timing_matrix_seed42.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_trials % 6 != 0:
        raise ValueError("--n-trials must be a multiple of 6")
    stage0_ckpt = args.stage0_ckpt or os.path.join(
        CHECKPOINT_DIR_DEFAULT,
        f"stage_0_seed{args.seed}.npz",
    )
    if not os.path.exists(stage0_ckpt):
        raise FileNotFoundError(stage0_ckpt)

    teacher_modes = [x.strip() for x in args.teacher_modes.split(",") if x.strip()]
    gate_modes = [x.strip() for x in args.gate_modes.split(",") if x.strip()]
    results = []
    for teacher_mode in teacher_modes:
        for gate_mode in gate_modes:
            print(f"=== ctx_pred timing arm: teacher={teacher_mode} gate={gate_mode} ===")
            res = run_arm(
                seed=args.seed,
                n_trials=args.n_trials,
                teacher_mode=teacher_mode,
                gate_mode=gate_mode,
                presettle_ms=args.presettle_ms,
                leader_ms=args.leader_ms,
                trailer_ms=args.trailer_ms,
                iti_ms=args.iti_ms,
                probe_window_ms=args.probe_window_ms,
                p_bias=args.p_bias,
                v1_to_hctx_g=args.v1_to_hctx_g,
                v1_to_hpred_g=args.v1_to_hpred_g,
                m_gate_dt_ms=args.m_gate_dt_ms,
                stage0_ckpt=stage0_ckpt,
            )
            print(
                "  gate_delta_ms="
                f"{res['gate_delta_ms']['min']}..{res['gate_delta_ms']['max']} "
                "pred=leader "
                f"{res['forecast']['p_pred_argmax_eq_leader']:.3f} "
                "pred=expected "
                f"{res['forecast']['p_pred_argmax_eq_expected']:.3f} "
                "w_exp-minus-leader "
                f"{res['w_expected_minus_leader']:.6f}"
            )
            results.append(res)

    payload = {
        "script": "diag_ctx_pred_timing_matrix.py",
        "args": vars(args),
        "stage0_ckpt": stage0_ckpt,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=_jsonify) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
