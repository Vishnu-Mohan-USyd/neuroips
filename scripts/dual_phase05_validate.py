"""Phase 0.5 validation: run M7 three times with the three task_state values.

This is a no-code-impact validator that directly calls
``metric_match_vs_near_miss_decoding`` with ``task_state`` set to
``(0,0)``, ``(1,0)``, ``(0,1)`` and writes the δ=10° deltas to JSON.

Expected result on the e1 checkpoint (V2 ignores task_state until Phase 1A
is trained):
    m7_baseline δ=10°  ≈ +0.1037  (matches task #23 regression baseline ±0.01)
    m7_focused  δ=10°  ≈ +0.1037  (within ±0.01 of m7_baseline)
    m7_routine  δ=10°  ≈ +0.1037  (within ±0.01 of m7_baseline)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make sure we can import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_representation import (  # noqa: E402
    load_model,
    metric_match_vs_near_miss_decoding,
)


def _m7_call(net, device, task_state: tuple[float, float]) -> dict:
    return metric_match_vs_near_miss_decoding(
        net, device,
        n_train=800, n_test=200,
        noise_std=0.3, readout_noise_std=0.3,
        seed=42, oracle_theta=90.0,
        task_state=task_state,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)

    net, _model_cfg, _train_cfg = load_model(args.checkpoint, args.config, device)

    print("Running M7 baseline  (task_state=(0,0))...")
    m7_baseline = _m7_call(net, device, (0.0, 0.0))
    print(f"  δ=10° delta_acc = {m7_baseline['delta_10']['delta_acc']:+.6f}")

    print("Running M7 focused   (task_state=(1,0))...")
    m7_focused = _m7_call(net, device, (1.0, 0.0))
    print(f"  δ=10° delta_acc = {m7_focused['delta_10']['delta_acc']:+.6f}")

    print("Running M7 routine   (task_state=(0,1))...")
    m7_routine = _m7_call(net, device, (0.0, 1.0))
    print(f"  δ=10° delta_acc = {m7_routine['delta_10']['delta_acc']:+.6f}")

    # --- Consistency checks ---
    legacy_expected = 0.1037
    legacy_tol = 0.01
    agreement_tol = 0.01

    d10_baseline = m7_baseline["delta_10"]["delta_acc"]
    d10_focused = m7_focused["delta_10"]["delta_acc"]
    d10_routine = m7_routine["delta_10"]["delta_acc"]

    legacy_diff = d10_baseline - legacy_expected
    legacy_pass = abs(legacy_diff) <= legacy_tol
    focused_vs_baseline = d10_focused - d10_baseline
    routine_vs_baseline = d10_routine - d10_baseline
    focused_pass = abs(focused_vs_baseline) <= agreement_tol
    routine_pass = abs(routine_vs_baseline) <= agreement_tol
    overall_pass = legacy_pass and focused_pass and routine_pass

    out = {
        "paradigm": "task_state_m7_scaffolding_noop",
        "checkpoint": args.checkpoint,
        "n_train_per_anchor": 800,
        "n_test_per_anchor": 200,
        "noise_std": 0.3,
        "readout_noise_std": 0.3,
        "seed": 42,
        "m7_baseline": m7_baseline,
        "m7_focused": m7_focused,
        "m7_routine": m7_routine,
        "delta10_baseline": d10_baseline,
        "delta10_focused": d10_focused,
        "delta10_routine": d10_routine,
        "delta10_focused_vs_baseline": focused_vs_baseline,
        "delta10_routine_vs_baseline": routine_vs_baseline,
        "legacy_expected_delta10": legacy_expected,
        "legacy_tol": legacy_tol,
        "agreement_tol": agreement_tol,
        "legacy_PASS": legacy_pass,
        "focused_agreement_PASS": focused_pass,
        "routine_agreement_PASS": routine_pass,
        "overall_verdict": "PASS" if overall_pass else "FAIL",
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print()
    print("=" * 70)
    print(f"m7_baseline δ=10°  : {d10_baseline:+.6f}  "
          f"(legacy expected {legacy_expected:+.6f}, diff {legacy_diff:+.6f}) "
          f"{'PASS' if legacy_pass else 'FAIL'}")
    print(f"m7_focused  δ=10°  : {d10_focused:+.6f}  "
          f"(vs baseline {focused_vs_baseline:+.6f}) "
          f"{'PASS' if focused_pass else 'FAIL'}")
    print(f"m7_routine  δ=10°  : {d10_routine:+.6f}  "
          f"(vs baseline {routine_vs_baseline:+.6f}) "
          f"{'PASS' if routine_pass else 'FAIL'}")
    print("=" * 70)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
