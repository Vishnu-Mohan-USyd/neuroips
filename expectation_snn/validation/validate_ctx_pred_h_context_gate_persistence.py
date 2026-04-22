"""Focused regression for Stage-1 ctx_pred H_context trailer overwrite.

This validator runs the actual ``run_stage_1_ctx_pred`` training path on a
small deterministic schedule. It asserts that H_context is not externally
overwritten by the actual trailer at the delayed M-gate: the saved gate-window
H_context argmax/rates must favor the leader over the trailer. It also checks
that learned aggregate ctxLeader -> predExpected deltas exceed both
leader-copy and trailer-self deltas, and that the ctx_pred Stage-1 persistence
gate is scored from H_context memory with recurrent normalizers inactive,
rather than H_prediction autonomous decay. It also asserts the Stage-1 ITI
fix: after the trailer-end gate, H_context is quiesced and ctx->pred
transmission is disabled only for the blank/ITI tail.

Run with:
    python -m expectation_snn.validation.validate_ctx_pred_h_context_gate_persistence
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.h_context_prediction import HContextPredictionConfig
from expectation_snn.brian2_model.h_ring import HRingConfig, N_CHANNELS, N_E_PER_CHANNEL
from expectation_snn.brian2_model.train import (
    CHECKPOINT_DIR_DEFAULT,
    H_ORIENT_CHANNELS,
    _stage1_h_cfg,
    run_stage_1_ctx_pred,
)


SEED = 42
N_TRIALS = 6
LEADER_MS = 500.0
TRAILER_MS = 500.0
ITI_MS = 100.0
PROBE_WINDOW_MS = 100.0
PERSIST_PULSE_MS = 300.0
PERSIST_POST_MS = 1000.0
P_BIAS = 0.999999999999
MIN_LEADER_GATE_MATCH = 0.5
MAX_TRAILER_GATE_MATCH = 0.34
MIN_RATE_MARGIN_HZ = 1.0
MIN_DW_MARGIN = 1e-7
PERSISTENCE_BAND_MS = (200.0, 500.0)
MAX_ITI_CTX_E_RATE_HZ = 1.0
MAX_DISABLED_CTX_PRED_W = 1e-12


def _production_candidate_cfg() -> tuple[HRingConfig, HContextPredictionConfig]:
    """Return the production candidate config for the small Stage-1 run."""
    h_cfg = _stage1_h_cfg(HRingConfig())
    pred_h_cfg = _stage1_h_cfg(HRingConfig())
    pred_h_cfg.w_inh_e_init = 1.0
    pred_h_cfg.inh_w_max = 3.0
    ctx_pred_cfg = HContextPredictionConfig(
        ctx_cfg=h_cfg,
        pred_cfg=pred_h_cfg,
        drive_amp_ctx_pred_pA=400.0,
        pred_e_uniform_bias_pA=100.0,
        w_init_frac=0.0,
    )
    return h_cfg, ctx_pred_cfg


def _transition_delta_mean(
    w_init: np.ndarray,
    w_final: np.ndarray,
    pre_orient_idx: np.ndarray,
    post_orient_idx: np.ndarray,
) -> float:
    """Mean delta w over orientation-index transition blocks.

    ``W_ctx_pred`` is all-to-all over H E cells. Brian2 stores the all-to-all
    ``connect(True)`` matrix in pre-major order for this synapse, matching a
    ``(n_pre, n_post)`` reshape.
    """
    n_e = N_CHANNELS * N_E_PER_CHANNEL
    dw_matrix = (w_final - w_init).reshape((n_e, n_e))
    values: list[np.ndarray] = []
    for pre_idx, post_idx in zip(pre_orient_idx, post_orient_idx):
        pre_ch = int(H_ORIENT_CHANNELS[int(pre_idx)])
        post_ch = int(H_ORIENT_CHANNELS[int(post_idx)])
        pre_cells = np.arange(
            pre_ch * N_E_PER_CHANNEL,
            (pre_ch + 1) * N_E_PER_CHANNEL,
            dtype=np.int64,
        )
        post_cells = np.arange(
            post_ch * N_E_PER_CHANNEL,
            (post_ch + 1) * N_E_PER_CHANNEL,
            dtype=np.int64,
        )
        values.append(dw_matrix[np.ix_(pre_cells, post_cells)].ravel())
    return float(np.concatenate(values).mean())


def run_probe() -> dict:
    """Run the small actual Stage-1 path and return assertion metrics."""
    stage0_ckpt = os.path.join(CHECKPOINT_DIR_DEFAULT, f"stage_0_seed{SEED}.npz")
    if not os.path.exists(stage0_ckpt):
        raise FileNotFoundError(
            f"Stage-0 checkpoint required for this validator: {stage0_ckpt}"
        )

    h_cfg, ctx_pred_cfg = _production_candidate_cfg()
    with tempfile.TemporaryDirectory(prefix="ctx_pred_gate_persist_") as tmpdir:
        res = run_stage_1_ctx_pred(
            seed=SEED,
            n_trials=N_TRIALS,
            leader_ms=LEADER_MS,
            trailer_ms=TRAILER_MS,
            iti_ms=ITI_MS,
            probe_window_ms=PROBE_WINDOW_MS,
            presettle_ms=0.0,
            persist_pulse_ms=PERSIST_PULSE_MS,
            persist_post_ms=PERSIST_POST_MS,
            trials_used_tail=N_TRIALS,
            checkpoint_dir=tmpdir,
            stage0_ckpt=stage0_ckpt,
            h_cfg=h_cfg,
            ctx_pred_cfg=ctx_pred_cfg,
            p_bias=P_BIAS,
            verbose=False,
        )
        if res.checkpoint_path is None:
            raise AssertionError("run_stage_1_ctx_pred did not produce a checkpoint")
        with np.load(res.checkpoint_path, allow_pickle=False) as ck:
            leader_idx = np.asarray(ck["leader_idx"], dtype=np.int64)
            trailer_idx = np.asarray(ck["trailer_idx"], dtype=np.int64)
            expected_idx = np.asarray(ck["expected_trailer_idx"], dtype=np.int64)
            ctx_argmax_gate = np.asarray(ck["h_argmax_ctx_at_gate"], dtype=np.int64)
            leader_rate = np.asarray(ck["ctx_gate_leader_rate_hz"], dtype=np.float64)
            trailer_rate = np.asarray(ck["ctx_gate_trailer_rate_hz"], dtype=np.float64)
            expected_rate = np.asarray(ck["ctx_gate_expected_rate_hz"], dtype=np.float64)
            gate_t_ms = np.asarray(ck["gate_t_ms"], dtype=np.float64)
            gate_target_ms = np.asarray(ck["ctx_pred_gate_times_ms"], dtype=np.float64)
            w_init = np.asarray(ck["W_ctx_pred_init"], dtype=np.float64)
            w_final = np.asarray(ck["W_ctx_pred_final"], dtype=np.float64)
            persistence_med_ms = float(np.asarray(ck["persistence_med_ms"]).item())
            h_context_persistence_ms = float(
                np.asarray(ck["h_context_persistence_ms"]).item()
            )
            h_pred_autonomous_persistence_ms = float(
                np.asarray(ck["h_pred_autonomous_persistence_ms"]).item()
            )
            source_raw = np.asarray(ck["h_bump_persistence_source"]).item()
            if isinstance(source_raw, (bytes, np.bytes_)):
                persistence_source = source_raw.decode("utf-8")
            else:
                persistence_source = str(source_raw)
            recurrent_norm_active = bool(
                np.asarray(
                    ck["recurrent_normalizers_active_during_post_probes"],
                ).item()
            )
            ctx_norm_active = bool(
                np.asarray(ck["ctx_ee_norm_active_during_post_probes"]).item()
            )
            pred_norm_active = bool(
                np.asarray(ck["pred_ee_norm_active_during_post_probes"]).item()
            )
            iti_gate_flush_ms = float(np.asarray(ck["iti_gate_flush_ms"]).item())
            iti_ctx_e_rate_hz = np.asarray(
                ck["iti_ctx_e_rate_hz"], dtype=np.float64,
            )
            iti_pred_e_rate_hz = np.asarray(
                ck["iti_pred_e_rate_hz"], dtype=np.float64,
            )
            iti_ctx_e_rate_mean_hz = float(
                np.asarray(ck["iti_ctx_e_rate_mean_hz"]).item()
            )
            iti_pred_e_rate_mean_hz = float(
                np.asarray(ck["iti_pred_e_rate_mean_hz"]).item()
            )
            ctx_quiesced_iti_count = int(
                np.asarray(ck["ctx_quiesced_iti_count"]).item()
            )
            ctx_pred_disabled_iti_count = int(
                np.asarray(
                    ck["ctx_pred_transmission_disabled_iti_count"],
                ).item()
            )
            ctx_pred_disabled_iti_ms = float(
                np.asarray(ck["ctx_pred_transmission_disabled_iti_ms"]).item()
            )
            ctx_pred_iti_disabled_w_max = float(
                np.asarray(ck["ctx_pred_iti_disabled_w_max"]).item()
            )
            ctx_pred_restored_after_schedule = bool(
                np.asarray(ck["ctx_pred_restored_after_schedule"]).item()
            )

    assert leader_idx.size == N_TRIALS, leader_idx
    assert np.array_equal(trailer_idx, expected_idx), (
        f"high-bias deterministic mini-run expected actual trailers to equal "
        f"expected trailers; got trailer={trailer_idx.tolist()} "
        f"expected={expected_idx.tolist()}"
    )
    assert gate_t_ms.size == N_TRIALS, (
        f"expected {N_TRIALS} M-gate updates, got {gate_t_ms.size}"
    )
    gate_delta_ms = gate_t_ms - gate_target_ms
    assert np.all(gate_delta_ms >= -1e-9), (
        f"M-gate fired before target gate time: delta_ms={gate_delta_ms.tolist()}"
    )

    leader_match = float(np.mean(ctx_argmax_gate == leader_idx))
    trailer_match = float(np.mean(ctx_argmax_gate == trailer_idx))
    expected_match = float(np.mean(ctx_argmax_gate == expected_idx))
    leader_rate_mean = float(leader_rate.mean())
    trailer_rate_mean = float(trailer_rate.mean())
    expected_rate_mean = float(expected_rate.mean())
    assert leader_match >= MIN_LEADER_GATE_MATCH, (
        f"H_context gate argmax did not favor leader often enough: "
        f"leader_match={leader_match:.3f}, threshold={MIN_LEADER_GATE_MATCH:.3f}, "
        f"ctx_gate={ctx_argmax_gate.tolist()}, leader={leader_idx.tolist()}"
    )
    assert trailer_match <= MAX_TRAILER_GATE_MATCH, (
        f"H_context looks trailer-overwritten at gate: "
        f"trailer_match={trailer_match:.3f}, threshold={MAX_TRAILER_GATE_MATCH:.3f}, "
        f"ctx_gate={ctx_argmax_gate.tolist()}, trailer={trailer_idx.tolist()}"
    )
    assert leader_rate_mean > trailer_rate_mean + MIN_RATE_MARGIN_HZ, (
        f"H_context gate rates do not favor leader over trailer: "
        f"leader={leader_rate_mean:.3f} Hz trailer={trailer_rate_mean:.3f} Hz "
        f"margin_threshold={MIN_RATE_MARGIN_HZ:.3f} Hz"
    )

    assert persistence_source == "h_context", (
        f"ctx_pred h_bump_persistence source must be h_context, got "
        f"{persistence_source!r}"
    )
    assert np.isclose(persistence_med_ms, h_context_persistence_ms), (
        f"legacy persistence_med_ms should equal h_context_persistence_ms "
        f"for ctx_pred; got {persistence_med_ms:.3f} vs "
        f"{h_context_persistence_ms:.3f}"
    )
    report_persistence = float(res.report.results["h_bump_persistence_ms"].value)
    assert np.isclose(report_persistence, h_context_persistence_ms), (
        f"report h_bump_persistence_ms should be H_context persistence; "
        f"report={report_persistence:.3f}, ctx={h_context_persistence_ms:.3f}"
    )
    lo, hi = PERSISTENCE_BAND_MS
    context_band_pass = bool(lo <= h_context_persistence_ms <= hi)
    report_band_pass = bool(res.report.results["h_bump_persistence_ms"].passed)
    assert report_band_pass == context_band_pass, (
        f"report persistence pass flag should be based on H_context band "
        f"[{lo:.1f}, {hi:.1f}] ms; report_pass={report_band_pass}, "
        f"context_pass={context_band_pass}, "
        f"context={h_context_persistence_ms:.3f} ms"
    )
    assert context_band_pass, (
        f"H_context persistence should land in Stage-1 memory band "
        f"[{lo:.1f}, {hi:.1f}] ms with recurrent normalizers inactive; "
        f"got {h_context_persistence_ms:.3f} ms"
    )
    assert not recurrent_norm_active and not ctx_norm_active and not pred_norm_active, (
        f"post-training probes must run with recurrent normalizers inactive; "
        f"combined={recurrent_norm_active}, ctx={ctx_norm_active}, "
        f"pred={pred_norm_active}"
    )

    assert ctx_quiesced_iti_count == N_TRIALS, (
        f"H_context should be quiesced once per ITI after the delayed gate; "
        f"got {ctx_quiesced_iti_count}, expected {N_TRIALS}"
    )
    assert ctx_pred_disabled_iti_count == N_TRIALS, (
        f"ctx->pred transmission should be disabled once per ITI tail; "
        f"got {ctx_pred_disabled_iti_count}, expected {N_TRIALS}"
    )
    expected_disabled_ms = N_TRIALS * max(ITI_MS - iti_gate_flush_ms, 0.0)
    assert np.isclose(ctx_pred_disabled_iti_ms, expected_disabled_ms), (
        f"ctx->pred disabled duration should equal post-gate ITI tail; "
        f"got {ctx_pred_disabled_iti_ms:.3f} ms, "
        f"expected {expected_disabled_ms:.3f} ms"
    )
    assert ctx_pred_iti_disabled_w_max <= MAX_DISABLED_CTX_PRED_W, (
        f"ctx->pred weights must be zero while disabled in ITI; "
        f"max_disabled_w={ctx_pred_iti_disabled_w_max:.12g}"
    )
    assert ctx_pred_restored_after_schedule, (
        "ctx->pred weights were not restored after ITI suppression"
    )
    assert np.all(np.isfinite(iti_ctx_e_rate_hz)), iti_ctx_e_rate_hz.tolist()
    assert float(np.max(iti_ctx_e_rate_hz)) <= MAX_ITI_CTX_E_RATE_HZ, (
        f"H_context should be quiet in the post-gate ITI tail; "
        f"max={float(np.max(iti_ctx_e_rate_hz)):.3f} Hz, "
        f"threshold={MAX_ITI_CTX_E_RATE_HZ:.3f} Hz, "
        f"rates={iti_ctx_e_rate_hz.tolist()}"
    )

    desired_dw = _transition_delta_mean(w_init, w_final, leader_idx, expected_idx)
    leader_copy_dw = _transition_delta_mean(w_init, w_final, leader_idx, leader_idx)
    trailer_self_dw = _transition_delta_mean(w_init, w_final, trailer_idx, trailer_idx)
    assert desired_dw > leader_copy_dw + MIN_DW_MARGIN, (
        f"desired leader->expected delta {desired_dw:.9f} must exceed "
        f"leader-copy {leader_copy_dw:.9f}"
    )
    assert desired_dw > trailer_self_dw + MIN_DW_MARGIN, (
        f"desired leader->expected delta {desired_dw:.9f} must exceed "
        f"trailer-self {trailer_self_dw:.9f}"
    )

    return {
        "n_trials": N_TRIALS,
        "gate_updates": int(gate_t_ms.size),
        "gate_delta_min_ms": float(gate_delta_ms.min()),
        "gate_delta_max_ms": float(gate_delta_ms.max()),
        "leader_match": leader_match,
        "trailer_match": trailer_match,
        "expected_match": expected_match,
        "leader_rate_mean": leader_rate_mean,
        "trailer_rate_mean": trailer_rate_mean,
        "expected_rate_mean": expected_rate_mean,
        "desired_dw": desired_dw,
        "leader_copy_dw": leader_copy_dw,
        "trailer_self_dw": trailer_self_dw,
        "persistence_source": persistence_source,
        "persistence_med_ms": persistence_med_ms,
        "h_context_persistence_ms": h_context_persistence_ms,
        "h_pred_autonomous_persistence_ms": h_pred_autonomous_persistence_ms,
        "report_persistence_ms": report_persistence,
        "context_band_pass": context_band_pass,
        "report_band_pass": report_band_pass,
        "recurrent_norm_active": recurrent_norm_active,
        "ctx_norm_active": ctx_norm_active,
        "pred_norm_active": pred_norm_active,
        "iti_gate_flush_ms": iti_gate_flush_ms,
        "iti_ctx_e_rate_mean_hz": iti_ctx_e_rate_mean_hz,
        "iti_ctx_e_rate_max_hz": float(np.max(iti_ctx_e_rate_hz)),
        "iti_pred_e_rate_mean_hz": iti_pred_e_rate_mean_hz,
        "iti_pred_e_rate_max_hz": float(np.max(iti_pred_e_rate_hz)),
        "ctx_quiesced_iti_count": ctx_quiesced_iti_count,
        "ctx_pred_disabled_iti_count": ctx_pred_disabled_iti_count,
        "ctx_pred_disabled_iti_ms": ctx_pred_disabled_iti_ms,
        "ctx_pred_iti_disabled_w_max": ctx_pred_iti_disabled_w_max,
        "ctx_pred_restored_after_schedule": ctx_pred_restored_after_schedule,
        "leader_idx": leader_idx,
        "trailer_idx": trailer_idx,
        "ctx_argmax_gate": ctx_argmax_gate,
        "passed_stage1_report": bool(res.report.passed),
    }


def main() -> int:
    metrics = run_probe()
    print("validate_ctx_pred_h_context_gate_persistence: PASS")
    print(
        "  mini actual Stage-1 path: "
        f"n_trials={metrics['n_trials']} gate_updates={metrics['gate_updates']} "
        f"stage1_report_passed={metrics['passed_stage1_report']}"
    )
    print(
        "  gate timing: "
        f"delta_ms=[{metrics['gate_delta_min_ms']:.1f}, "
        f"{metrics['gate_delta_max_ms']:.1f}]"
    )
    print(
        "  H_context at delayed gate: "
        f"leader_match={metrics['leader_match']:.3f} "
        f"trailer_match={metrics['trailer_match']:.3f} "
        f"expected_match={metrics['expected_match']:.3f}; "
        f"rates leader/trailer/expected="
        f"{metrics['leader_rate_mean']:.3f}/"
        f"{metrics['trailer_rate_mean']:.3f}/"
        f"{metrics['expected_rate_mean']:.3f} Hz"
    )
    print(
        "  delta_w aggregate: "
        f"leader->expected={metrics['desired_dw']:.9f} "
        f"leader->leader={metrics['leader_copy_dw']:.9f} "
        f"trailer->trailer={metrics['trailer_self_dw']:.9f}"
    )
    print(
        "  persistence metric: "
        f"source={metrics['persistence_source']} "
        f"h_bump/report={metrics['report_persistence_ms']:.1f} ms "
        f"h_context={metrics['h_context_persistence_ms']:.1f} ms "
        f"h_pred_autonomous={metrics['h_pred_autonomous_persistence_ms']:.1f} ms "
        f"context_band_pass={metrics['context_band_pass']} "
        f"normalizers_active={metrics['recurrent_norm_active']}"
    )
    print(
        "  ITI suppression: "
        f"gate_flush={metrics['iti_gate_flush_ms']:.1f} ms "
        f"Hctx_E mean/max="
        f"{metrics['iti_ctx_e_rate_mean_hz']:.3f}/"
        f"{metrics['iti_ctx_e_rate_max_hz']:.3f} Hz "
        f"Hpred_E mean/max="
        f"{metrics['iti_pred_e_rate_mean_hz']:.3f}/"
        f"{metrics['iti_pred_e_rate_max_hz']:.3f} Hz "
        f"ctx_quiesced={metrics['ctx_quiesced_iti_count']} "
        f"ctx_pred_disabled={metrics['ctx_pred_disabled_iti_count']} "
        f"disabled_ms={metrics['ctx_pred_disabled_iti_ms']:.1f} "
        f"max_disabled_w={metrics['ctx_pred_iti_disabled_w_max']:.3g} "
        f"restored={metrics['ctx_pred_restored_after_schedule']}"
    )
    print(
        "  gate argmax sequence: "
        f"ctx={metrics['ctx_argmax_gate'].tolist()} "
        f"leader={metrics['leader_idx'].tolist()} "
        f"trailer={metrics['trailer_idx'].tolist()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
