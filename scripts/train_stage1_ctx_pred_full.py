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
)
from expectation_snn.brian2_model.h_ring import HRingConfig
from expectation_snn.brian2_model.h_context_prediction import HContextPredictionConfig


SEED = 42
N_TRIALS = 360

# --- attempt #3 (real attempt #2), Sprint 5e-Fix E, 2026-04-21 -------------
# Attempt #1 (commit 5317540): FAIL. bump_persistence=10ms, forecast=0.0.
# Attempt #2 (commit 31e6e98): bit-identical no-op. The script override
#   HRingConfig(inh_w_max=2.0) was silently overwritten by train.py's
#   _stage1_h_cfg (forces inh_w_max=1.5 unconditionally).
# Attempt #3 (THIS run) — fix commit 5ef422b:
#   Snapshot (docs/stage1_ctx_pred_attempt1_snapshot.md, commit 4493fa8)
#   showed W_ctx_pred collapsed to w_row_max / n_pre = 3.0 / 192 = 0.015625
#   on trial 1 because init row sum (0.025 * 192 = 4.80) exceeded
#   w_row_max = 3.0. Fix: shrink DEFAULT_W_INIT_FRAC 0.05 -> 0.015 so init
#   row sum = 0.0075 * 192 = 1.44, leaving headroom for LTP before cap.
# HRingConfig(inh_w_max=2.0) retained here for trace continuity — it is
# still a no-op under _stage1_h_cfg but documents intent.
H_CFG = HRingConfig(inh_w_max=2.0)
CTX_PRED_CFG = HContextPredictionConfig(ctx_cfg=H_CFG, pred_cfg=H_CFG)
ATTEMPT = 3


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
    print(f"fix (commit 5ef422b) : DEFAULT_W_INIT_FRAC 0.05 -> 0.015  "
          f"(row-cap collapse fix; docs/stage1_ctx_pred_attempt1_snapshot.md)")
    print(f"               effective W_ctx_pred init row sum = "
          f"{CTX_PRED_CFG.w_init_frac * CTX_PRED_CFG.w_max / 2 * 192:.3f} "
          f"(cap = {CTX_PRED_CFG.w_row_max:.1f})")
    print(f"intended HRingConfig.inh_w_max = {H_CFG.inh_w_max} "
          f"(no-op under train._stage1_h_cfg; effective = 1.5)")
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
            "HRingConfig.inh_w_max_intended": H_CFG.inh_w_max,
            "HContextPredictionConfig.w_init_frac": CTX_PRED_CFG.w_init_frac,
            "HContextPredictionConfig.w_row_max": CTX_PRED_CFG.w_row_max,
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
