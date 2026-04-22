"""Regression primitive for Stage-1 ctx_pred delayed-gate learning.

This is intentionally bottom-up: it builds H_context + H_prediction only,
with no V1, no full Stage-1 schedule, and no retrain. One leader cue drives
H_context channel 0 while H_prediction receives no teacher. Then one trailer
cue drives H_prediction channel 2. The ctx_pred M-gate is consumed at trailer
offset/end, after trailer H_pred spikes have occurred.

Pass condition: under the production candidate config
``drive_amp_ctx_pred_pA=400``, ``pred_e_uniform_bias_pA=100``, and
``w_init_frac=0``, the mean weight increase for
ctxLeader -> predTrailer must exceed ctxLeader -> predLeader. H_prediction
must also remain silent during the leader window, proving the leader-copy
eligibility is not being created by initial ctx->pred drive.

Run with:
    python -m expectation_snn.validation.validate_ctx_pred_delayed_gate_learning
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs, seed as b2_seed, start_scope

from expectation_snn.brian2_model.h_context_prediction import (
    HContextPrediction,
    HContextPredictionConfig,
    build_h_context_prediction,
    make_modulatory_gate_operation,
    silence_direction,
)
from expectation_snn.brian2_model.h_ring import (
    HRingConfig,
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
    pulse_channel,
    silence_cue,
)
from expectation_snn.brian2_model.train import _stage1_h_cfg


SEED = 42
BASELINE_MS = 100.0
LEADER_MS = 300.0
TRAILER_MS = 300.0
POST_GATE_MS = 100.0
PULSE_RATE_HZ = 400.0
LEADER_CHANNEL = 0
TRAILER_CHANNEL = 2
LEADER_PRED_SILENCE_HZ = 2.0
MIN_DW_MARGIN = 1e-8
M_GATE_DT_MS = 5.0


def _production_candidate_cfg() -> HContextPredictionConfig:
    """Return the Stage-1 production candidate used by the full retrain script."""
    h_cfg = _stage1_h_cfg(HRingConfig())
    pred_h_cfg = _stage1_h_cfg(HRingConfig())
    pred_h_cfg.w_inh_e_init = 1.0
    pred_h_cfg.inh_w_max = 3.0
    return HContextPredictionConfig(
        ctx_cfg=h_cfg,
        pred_cfg=pred_h_cfg,
        drive_amp_ctx_pred_pA=400.0,
        pred_e_uniform_bias_pA=100.0,
        w_init_frac=0.0,
    )


def _per_channel_rate_hz(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    e_channel: np.ndarray,
    t0_ms: float,
    t1_ms: float,
) -> np.ndarray:
    """Return per-channel E-cell rates in Hz for ``[t0_ms, t1_ms)``."""
    mask = (spike_t_ms >= t0_ms) & (spike_t_ms < t1_ms)
    counts = np.bincount(e_channel[spike_i[mask]], minlength=H_N_CHANNELS)
    dur_s = max((t1_ms - t0_ms) / 1000.0, 1e-12)
    return counts.astype(np.float64) / (H_N_E_PER * dur_s)


def _population_rate_hz(
    spike_t_ms: np.ndarray,
    t0_ms: float,
    t1_ms: float,
    n_cells: int,
) -> float:
    """Return population mean firing rate in Hz for ``[t0_ms, t1_ms)``."""
    n_spikes = int(((spike_t_ms >= t0_ms) & (spike_t_ms < t1_ms)).sum())
    dur_s = max((t1_ms - t0_ms) / 1000.0, 1e-12)
    return n_spikes / (n_cells * dur_s)


def _weight_delta_mean(
    bundle: HContextPrediction,
    w_before: np.ndarray,
    w_after: np.ndarray,
    *,
    pre_channel: int,
    post_channel: int,
) -> float:
    """Mean ``delta w`` for one ctx channel to one pred channel block."""
    i_pre = np.asarray(bundle.ctx_pred.i[:], dtype=np.int64)
    j_post = np.asarray(bundle.ctx_pred.j[:], dtype=np.int64)
    pre_ch = np.asarray(bundle.ctx.e_channel, dtype=np.int64)
    post_ch = np.asarray(bundle.pred.e_channel, dtype=np.int64)
    pre_cells = np.flatnonzero(pre_ch == pre_channel)
    post_cells = np.flatnonzero(post_ch == post_channel)
    mask = np.isin(i_pre, pre_cells) & np.isin(j_post, post_cells)
    assert mask.any(), (
        f"no ctx_pred synapses for pre channel {pre_channel} -> "
        f"post channel {post_channel}"
    )
    return float((w_after[mask] - w_before[mask]).mean())


def run_probe() -> dict:
    """Run the one-episode primitive and return assertion metrics."""
    prefs.codegen.target = "numpy"
    start_scope()
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)

    cfg = _production_candidate_cfg()
    bundle = build_h_context_prediction(
        config=cfg,
        rng=np.random.default_rng(SEED),
        ctx_name="ctx_delay_gate_val",
        pred_name="pred_delay_gate_val",
    )
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)

    w_before = np.asarray(bundle.ctx_pred.w[:], dtype=np.float64).copy()
    assert np.allclose(w_before, 0.0), (
        f"production candidate should start W_ctx_pred at zero; "
        f"range=[{w_before.min():.6g}, {w_before.max():.6g}]"
    )

    leader_start_ms = BASELINE_MS
    leader_end_ms = leader_start_ms + LEADER_MS
    trailer_start_ms = leader_end_ms
    trailer_end_ms = trailer_start_ms + TRAILER_MS

    gate_log: list[dict] = []
    mgate = make_modulatory_gate_operation(
        bundle,
        trailer_onsets_ms=[trailer_end_ms],
        dt_trial_s=(LEADER_MS + TRAILER_MS) / 1000.0,
        dt_op_ms=M_GATE_DT_MS,
        log=gate_log,
    )
    ctx_mon = SpikeMonitor(bundle.ctx.e, name="delay_gate_ctx_e")
    pred_mon = SpikeMonitor(bundle.pred.e, name="delay_gate_pred_e")
    net = Network(*bundle.groups, ctx_mon, pred_mon, mgate)

    net.run(BASELINE_MS * ms)

    pulse_channel(bundle.ctx, channel=LEADER_CHANNEL, rate_hz=PULSE_RATE_HZ)
    silence_cue(bundle.pred)
    net.run(LEADER_MS * ms)

    silence_cue(bundle.ctx)
    pulse_channel(bundle.pred, channel=TRAILER_CHANNEL, rate_hz=PULSE_RATE_HZ)
    net.run(TRAILER_MS * ms)

    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    net.run(POST_GATE_MS * ms)

    assert len(gate_log) == 1, f"expected one delayed M-gate update, got {len(gate_log)}"
    gate_t_ms = float(gate_log[0]["t_ms"])
    gate_delta_ms = gate_t_ms - trailer_end_ms
    assert gate_delta_ms >= -1e-9, (
        f"M-gate fired before trailer offset: gate_t={gate_t_ms:.3f} ms "
        f"trailer_end={trailer_end_ms:.3f} ms"
    )
    assert gate_delta_ms <= M_GATE_DT_MS + float(defaultclock.dt / ms) + 1e-9, (
        f"M-gate should fire within one operation tick of trailer offset; "
        f"delta={gate_delta_ms:.3f} ms"
    )

    pred_i = np.asarray(pred_mon.i[:], dtype=np.int64)
    pred_t_ms = np.asarray(pred_mon.t / ms, dtype=np.float64)
    pred_channel = np.asarray(bundle.pred.e_channel, dtype=np.int64)
    n_pred = int(bundle.pred.e.N)

    leader_pred_pop_rate = _population_rate_hz(
        pred_t_ms, leader_start_ms, leader_end_ms, n_pred,
    )
    leader_pred_ch_rates = _per_channel_rate_hz(
        pred_i, pred_t_ms, pred_channel, leader_start_ms, leader_end_ms,
    )
    leader_pred_max_ch_rate = float(leader_pred_ch_rates.max())
    assert leader_pred_pop_rate <= LEADER_PRED_SILENCE_HZ, (
        f"H_pred was not silent during leader: population={leader_pred_pop_rate:.3f} Hz "
        f"> {LEADER_PRED_SILENCE_HZ:.3f} Hz"
    )
    assert leader_pred_max_ch_rate <= LEADER_PRED_SILENCE_HZ, (
        f"H_pred leader max channel={leader_pred_max_ch_rate:.3f} Hz "
        f"> {LEADER_PRED_SILENCE_HZ:.3f} Hz"
    )

    trailer_pred_ch_rates = _per_channel_rate_hz(
        pred_i, pred_t_ms, pred_channel, trailer_start_ms, trailer_end_ms,
    )
    trailer_rate = float(trailer_pred_ch_rates[TRAILER_CHANNEL])
    assert trailer_rate > LEADER_PRED_SILENCE_HZ, (
        f"trailer channel did not produce H_pred response; rate={trailer_rate:.3f} Hz"
    )

    w_after = np.asarray(bundle.ctx_pred.w[:], dtype=np.float64).copy()
    dw_desired = _weight_delta_mean(
        bundle, w_before, w_after,
        pre_channel=LEADER_CHANNEL,
        post_channel=TRAILER_CHANNEL,
    )
    dw_copy = _weight_delta_mean(
        bundle, w_before, w_after,
        pre_channel=LEADER_CHANNEL,
        post_channel=LEADER_CHANNEL,
    )
    dw_margin = dw_desired - dw_copy
    assert dw_desired > dw_copy + MIN_DW_MARGIN, (
        f"desired ctxLeader->predTrailer delta ({dw_desired:.9f}) must exceed "
        f"leader-copy ctxLeader->predLeader delta ({dw_copy:.9f}) by "
        f">{MIN_DW_MARGIN:.1e}; margin={dw_margin:.9f}"
    )

    return {
        "drive_amp_ctx_pred_pA": cfg.drive_amp_ctx_pred_pA,
        "pred_e_uniform_bias_pA": cfg.pred_e_uniform_bias_pA,
        "w_init_frac": cfg.w_init_frac,
        "leader_start_ms": leader_start_ms,
        "leader_end_ms": leader_end_ms,
        "trailer_start_ms": trailer_start_ms,
        "trailer_end_ms": trailer_end_ms,
        "gate_t_ms": gate_t_ms,
        "gate_delta_ms": gate_delta_ms,
        "leader_pred_pop_rate_hz": leader_pred_pop_rate,
        "leader_pred_max_ch_rate_hz": leader_pred_max_ch_rate,
        "trailer_pred_ch_rate_hz": trailer_rate,
        "dw_desired": dw_desired,
        "dw_copy": dw_copy,
        "dw_margin": dw_margin,
        "gate_w_mean_before": float(gate_log[0]["w_mean_before"]),
        "gate_w_mean_after": float(gate_log[0]["w_mean_after"]),
    }


def main() -> int:
    metrics = run_probe()
    print("validate_ctx_pred_delayed_gate_learning: PASS")
    print(
        "  config: "
        f"drive_amp_ctx_pred={metrics['drive_amp_ctx_pred_pA']:.1f} pA "
        f"pred_e_uniform_bias={metrics['pred_e_uniform_bias_pA']:.1f} pA "
        f"w_init_frac={metrics['w_init_frac']:.3f}"
    )
    print(
        "  timing: "
        f"leader=[{metrics['leader_start_ms']:.1f}, {metrics['leader_end_ms']:.1f}) ms "
        f"trailer=[{metrics['trailer_start_ms']:.1f}, {metrics['trailer_end_ms']:.1f}) ms "
        f"gate_t={metrics['gate_t_ms']:.1f} ms "
        f"delta_from_trailer_offset={metrics['gate_delta_ms']:.1f} ms"
    )
    print(
        "  leader-window H_pred silence: "
        f"population={metrics['leader_pred_pop_rate_hz']:.3f} Hz "
        f"max_channel={metrics['leader_pred_max_ch_rate_hz']:.3f} Hz "
        f"threshold<={LEADER_PRED_SILENCE_HZ:.1f} Hz"
    )
    print(
        "  trailer response: "
        f"pred_channel_{TRAILER_CHANNEL}={metrics['trailer_pred_ch_rate_hz']:.3f} Hz"
    )
    print(
        "  delta_w: "
        f"ctxLeader->predTrailer={metrics['dw_desired']:.9f} "
        f"ctxLeader->predLeader={metrics['dw_copy']:.9f} "
        f"margin={metrics['dw_margin']:.9f}"
    )
    print(
        "  gate weight mean: "
        f"before={metrics['gate_w_mean_before']:.9f} "
        f"after={metrics['gate_w_mean_after']:.9f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
