"""Sprint 5e-Fix E: full Stage-1 ctx_pred retrain (n_trials=360, seed=42).

Launches the H_context + H_prediction Stage-1 driver in its consolidation
regime. The brief (n=60) validator run produced forecast@chance and
over-long bump-persistence; the 360-trial schedule is the regime where
the three-factor W_ctx_pred rule is expected to consolidate a predictive
mapping.

Outputs
-------
- Checkpoint (overwrites any prior n=60 run)::

      expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz

- Log (this script's stdout, tee'd by launcher)::

      logs/train_stage1_ctx_pred_full_seed42.log

- Evidence-log entry printed to stdout with the 3 primary gate values:
  ``h_bump_persistence_ms``, ``h_preprobe_forecast_prob``, ``no_runaway``.

Usage
-----
::

    python scripts/train_stage1_ctx_pred_full.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[1]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.train import (
    Stage1CtxPredResult, run_stage_1_ctx_pred, CHECKPOINT_DIR_DEFAULT,
    _stage1_h_cfg,
)
from expectation_snn.brian2_model.h_ring import HRingConfig
from expectation_snn.brian2_model.h_context_prediction import HContextPredictionConfig


SEED = 42
N_TRIALS = 360

# --- attempt #4, Sprint 5e-Fix E, 2026-04-21 --------------------------------
# Attempt #1 (commit 5317540): FAIL. bump_persistence=10ms, forecast=0.0.
# Attempt #2 (commit 31e6e98): bit-identical no-op (inh_w_max override
#   squashed by _stage1_h_cfg).
# Attempt #3 (commit 55d0f3f, fix 5ef422b): FAIL. w_init_frac 0.05 -> 0.015
#   fix landed (init mean 0.025 -> 0.0075, confirmed in evidence-log), but
#   persistence stayed at ~10 ms, W_ctx_pred collapsed onto the 3/192
#   row-cap floor after trial ~35, and pred_argmax == leader 99.4%
#   (amplifier signature). Debugger (task #47) delivered compound H1+H3+H4
#   verdict.
#
# Attempt #4 (THIS run) — three commits:
#   0f30cd2 Fix A: DEFAULT_TAU_COINC_MS 20 -> 500 ms (leader->trailer gap).
#   b5d8fd7 Fix B(i): HRingConfig.inh_rho_hz 2 -> 20 Hz (Vogels target).
#   b5ba400 Fix C: DEFAULT_W_TARGET 0.05 -> 0.0075 (match post-fix init).
#
#   Pre-check: H3 sanity (bg b1od0brws) confirmed inh_eta=0 raises
#   persistence 10 ms -> 990 ms at n=24, proving Vogels was the crusher.
#
#   Go/no-go thresholds (Lead dispatch 2026-04-21):
#       h_bump_persistence_ms in [200, 500]
#       h_preprobe_forecast_prob >= 0.25
#       no_runaway <= 80 Hz
#
#   This is the LAST iteration budget. Zero remaining if attempt #4 fails.
H_CFG = _stage1_h_cfg(HRingConfig())
PRED_H_CFG = _stage1_h_cfg(HRingConfig())
PRED_H_CFG.w_inh_e_init = 1.0
PRED_H_CFG.inh_w_max = 3.0
# Production candidate calibrated by Lorentz's combined diagnostic:
#   - 400 pA ctx->pred per-spike drive restores useful forecast drive.
#   - 100 pA uniform H_prediction E bias is label-blind excitability, not
#     transition content.
#   - stronger local inhibition is prediction-ring only here; H_context
#     stays on the existing Stage-1 H_CFG.
#   - zero initial ctx->pred weights keep H_prediction silent during the
#     leader when the V1->H_pred teacher is off; learning is assigned by
#     the delayed trailer-offset M-gate. This candidate passed saved Stage-1
#     and no-override feedback assays; future full checkpoints should
#     serialize these values in ctx_pred config metadata.
CTX_PRED_CFG = HContextPredictionConfig(
    ctx_cfg=H_CFG,
    pred_cfg=PRED_H_CFG,
    drive_amp_ctx_pred_pA=400.0,
    pred_e_uniform_bias_pA=100.0,
    w_init_frac=0.0,
)
ATTEMPT = 4


def _fmt_check(name: str, cr) -> str:
    band = cr.band
    return (
        f"  [{('PASS' if cr.passed else 'FAIL')}] {name:<30s} "
        f"value={cr.value:.3f} band=[{band[0]:.3f}, {band[1]:.3f}]"
    )


def main() -> int:
    t0 = time.time()
    print(f"=== Stage-1 ctx_pred FULL retrain (seed={SEED}, n_trials={N_TRIALS}, attempt={ATTEMPT}) ===")
    print(f"checkpoint dir: {CHECKPOINT_DIR_DEFAULT}")
    print(f"wall-start    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"attempt-#4 fixes (task #47 Debugger compound verdict):")
    print(f"  Fix A (0f30cd2): tau_coinc = {CTX_PRED_CFG.tau_coinc_ms:.1f} ms  "
          f"(was 20.0; span leader->trailer gap)")
    print(f"  Fix B(i) (b5d8fd7): HRingConfig.inh_rho_hz = {H_CFG.inh_rho_hz:.1f} Hz  "
          f"(was 2.0; match observed ctx-E rate)")
    print(f"  Fix C (b5ba400): w_target = {CTX_PRED_CFG.w_target:.5f}  "
          f"(was 0.05; match post-fix init mean)")
    print(f"  Carry-over (5ef422b): w_init_frac = {CTX_PRED_CFG.w_init_frac:.3f}  "
          f"(production candidate zero-init; avoid leader-window copy drive)")
    print(f"  Lorentz timing fix: Stage-1 ctx_pred M-gate fires at trailer offset/end, "
          f"after trailer H_pred response")
    print(f"  Production candidate: drive_amp_ctx_pred = "
          f"{CTX_PRED_CFG.drive_amp_ctx_pred_pA:.1f} pA; "
          f"pred_e_uniform_bias = {CTX_PRED_CFG.pred_e_uniform_bias_pA:.1f} pA "
          f"(Lorentz combined diagnostic)")
    print(f"  Prediction ring inhibition: w_inh_e_init = "
          f"{PRED_H_CFG.w_inh_e_init:.2f}; inh_w_max = "
          f"{PRED_H_CFG.inh_w_max:.2f} "
          f"(context w_inh_e_init={H_CFG.w_inh_e_init:.2f}, "
          f"context inh_w_max={H_CFG.inh_w_max:.2f})")
    print(f"init row sum = "
          f"{CTX_PRED_CFG.w_init_frac * CTX_PRED_CFG.w_max / 2 * 192:.3f} "
          f"(cap = {CTX_PRED_CFG.w_row_max:.1f})")
    print(f"go/no-go: bump_persistence in [200, 500] ms, "
          f"forecast >= 0.25, no_runaway <= 80 Hz")
    sys.stdout.flush()

    res: Stage1CtxPredResult = run_stage_1_ctx_pred(
        seed=SEED,
        n_trials=N_TRIALS,
        h_cfg=H_CFG,
        ctx_pred_cfg=CTX_PRED_CFG,
        verbose=True,
    )

    t1 = time.time()
    wall_s = t1 - t0
    print(f"\n=== Stage-1 ctx_pred full retrain completed in {wall_s:.1f} s "
          f"({wall_s / 60:.1f} min) ===")

    rep = res.report
    print("\n--- Stage-1 gate report ---")
    for name, cr in rep.results.items():
        print(_fmt_check(name, cr))
    print(f"  verdict: {'PASS' if rep.passed else 'FAIL'}")

    # Evidence-log entry (JSON for easy downstream parsing).
    evidence = {
        "stage": "stage_1_ctx_pred_full",
        "attempt": ATTEMPT,
        "overrides": {
            "HRingConfig.inh_rho_hz": H_CFG.inh_rho_hz,
            "HContextPredictionConfig.tau_coinc_ms": CTX_PRED_CFG.tau_coinc_ms,
            "HContextPredictionConfig.w_target": CTX_PRED_CFG.w_target,
            "HContextPredictionConfig.w_init_frac": CTX_PRED_CFG.w_init_frac,
            "HContextPredictionConfig.w_row_max": CTX_PRED_CFG.w_row_max,
            "HContextPredictionConfig.drive_amp_ctx_pred_pA": CTX_PRED_CFG.drive_amp_ctx_pred_pA,
            "HContextPredictionConfig.pred_e_uniform_bias_pA": CTX_PRED_CFG.pred_e_uniform_bias_pA,
            "HContextPredictionConfig.pred_cfg.w_inh_e_init": PRED_H_CFG.w_inh_e_init,
            "HContextPredictionConfig.pred_cfg.inh_w_max": PRED_H_CFG.inh_w_max,
            "HContextPredictionConfig.ctx_cfg.w_inh_e_init": H_CFG.w_inh_e_init,
            "HContextPredictionConfig.ctx_cfg.inh_w_max": H_CFG.inh_w_max,
            "init_row_sum_expected": CTX_PRED_CFG.w_init_frac * CTX_PRED_CFG.w_max / 2 * 192,
        },
        "seed": SEED,
        "n_trials": N_TRIALS,
        "wall_s": wall_s,
        "checkpoint_path": res.checkpoint_path,
        "passed": bool(rep.passed),
        "checks": {
            name: {
                "value": float(cr.value),
                "passed": bool(cr.passed),
                "band": [float(cr.band[0]), float(cr.band[1])],
                "detail": cr.detail,
            }
            for name, cr in rep.results.items()
        },
        "diagnostics": {k: float(v) for k, v in res.diagnostics.items()},
    }
    print("\n--- evidence-log JSON ---")
    print(json.dumps(evidence, indent=2, default=str))
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
