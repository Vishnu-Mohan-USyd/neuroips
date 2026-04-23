"""Task #74 Phase-0 — Check 2: eval-time task-state contamination.

Compares L2/3 orientation-coverage entropy under two eval variants on the
Task#70 Phase-3-Kok ckpt:
  (a) Normal localizer probes (as in baseline).
  (b) Same probes BUT with W_{mh,qm,lm}_task zeroed before measurement.

If entropy_b recovers Phase-2's value, the measurement is contaminated by
task-bias routing into L2/3 — i.e. not a genuine tuning collapse.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.task74_diagnostics import compute_coverage, LOCALIZER_ORIENTS_12


def main() -> int:
    ckpt = Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt")

    print("CHECK 2a: normal localizer (baseline redo for parity)", flush=True)
    bundle_a = load_checkpoint(ckpt, seed=42, device="cpu")
    out_a = compute_coverage(
        bundle_a, orients=LOCALIZER_ORIENTS_12,
        n_trials_per_orient=8, noise_std=0.0, seed=42,
    )
    print(
        f"  entropy_a = {out_a['entropy_nats']:.3f} nats  "
        f"n_bins_a = {out_a['n_bins_geq_5pct']}/12",
        flush=True,
    )

    print(
        "CHECK 2b: zero task state (W_mh_task=0, W_qm_task=0, W_lm_task=0)",
        flush=True,
    )
    bundle_b = load_checkpoint(ckpt, seed=42, device="cpu")
    cm = bundle_b.net.context_memory
    with torch.no_grad():
        cm.W_mh_task.zero_()
        cm.W_qm_task.zero_()
        cm.W_lm_task.zero_()
    print(
        f"  after zero: |W_mh_task|max={cm.W_mh_task.abs().max().item():.3e} "
        f"|W_qm_task|max={cm.W_qm_task.abs().max().item():.3e} "
        f"|W_lm_task|max={cm.W_lm_task.abs().max().item():.3e}",
        flush=True,
    )
    out_b = compute_coverage(
        bundle_b, orients=LOCALIZER_ORIENTS_12,
        n_trials_per_orient=8, noise_std=0.0, seed=42,
    )
    print(
        f"  entropy_b = {out_b['entropy_nats']:.3f} nats  "
        f"n_bins_b = {out_b['n_bins_geq_5pct']}/12",
        flush=True,
    )

    print()
    print("=== SUMMARY ===", flush=True)
    print(
        f"entropy_a={out_a['entropy_nats']:.3f} "
        f"entropy_b={out_b['entropy_nats']:.3f} "
        f"n_bins_a={out_a['n_bins_geq_5pct']} "
        f"n_bins_b={out_b['n_bins_geq_5pct']}",
        flush=True,
    )

    out = {
        "check": "check2_zero_task_state",
        "checkpoint": str(ckpt),
        "a_normal": out_a,
        "b_zeroed_task_weights": out_b,
        "entropy_a": out_a["entropy_nats"],
        "entropy_b": out_b["entropy_nats"],
        "n_bins_a": out_a["n_bins_geq_5pct"],
        "n_bins_b": out_b["n_bins_geq_5pct"],
    }
    Path("logs/task74/check2_zero_task_state.json").write_text(
        json.dumps(out, indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
