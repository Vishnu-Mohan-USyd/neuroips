"""Task #74 Phase-0 — Check 3: homeostasis-θ-drift causal test.

Isolates whether Phase-3 collapse is caused by θ drift on sensory-core
populations (l23_e, h_e) during Phase-3 training.

Procedure
---------
1. Load Phase-2 step_3000 ckpt (untainted by Phase-3).
2. Measure baseline coverage entropy (INIT_P2).
3. Clone the net and run ``run_phase3_kok_training`` for ``n_trials=100``
   with the homeostasis update methods MONKEY-PATCHED to no-op on
   sensory-core populations. All other Phase-3 mechanics (W_qm_task /
   W_mh_task updates) run normally.
4. Measure coverage entropy on the resulting net (NO_HOMEO).
5. Compare NO_HOMEO vs baseline Phase-3-Kok entropy (0.558):
   - NO_HOMEO ≈ 0.558 → θ drift is NOT causal (mechanism = W_mh_task / W_qm_task side-effect)
   - NO_HOMEO ≫ 0.558 (e.g. ≈1.7) → θ drift IS causal.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.task74_diagnostics import compute_coverage, LOCALIZER_ORIENTS_12
from scripts.v2.train_phase3_kok_learning import (
    run_phase3_kok_training, cue_mapping_from_seed,
)


def main() -> int:
    p2_path = Path("checkpoints/v2/phase2/phase2_task70_s42/phase2_s42/step_3000.pt")
    print(f"loading phase-2 step_3000 ckpt: {p2_path}", flush=True)

    bundle = load_checkpoint(p2_path, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")

    # Monkey-patch homeostasis.update to a no-op on sensory-core pops.
    def _noop(*_a, **_kw):
        return None
    bundle.net.l23_e.homeostasis.update = _noop  # type: ignore[assignment]
    bundle.net.h_e.homeostasis.update = _noop   # type: ignore[assignment]
    print("homeostasis.update disabled on l23_e, h_e", flush=True)

    cue_mapping = cue_mapping_from_seed(42)
    t0 = time.monotonic()
    n_trials = 100
    print(f"running run_phase3_kok_training with n_trials_learning={n_trials}", flush=True)
    run_phase3_kok_training(
        net=bundle.net,
        n_trials_learning=n_trials,
        n_trials_scan=0,
        validity_scan=1.0,
        lr=1e-3,
        weight_decay=0.0,
        seed=42,
        noise_std=0.0,
        cue_mapping=cue_mapping,
        metrics_path=None,
        log_every=max(n_trials // 10, 1),
    )
    print(f"training done in {time.monotonic()-t0:.1f}s", flush=True)

    print("computing coverage entropy on resulting (no-homeo-drift) net", flush=True)
    out = compute_coverage(
        bundle, orients=LOCALIZER_ORIENTS_12,
        n_trials_per_orient=15, noise_std=0.0, seed=42,
    )
    print(
        f"  entropy_no_homeo_drift = {out['entropy_nats']:.3f} nats  "
        f"n_bins = {out['n_bins_geq_5pct']}/12",
        flush=True,
    )

    # Compare against Task#70 Phase-3-Kok baseline (0.558) and Phase-2
    # step_3000 baseline (computed in parallel job)
    print()
    print("=== SUMMARY ===", flush=True)
    print(
        f"entropy_no_homeo_drift={out['entropy_nats']:.3f} "
        f"n_bins={out['n_bins_geq_5pct']}/12 "
        f"(compare: phase3_kok_baseline=0.558, phase2_task70=<see parallel job>)",
        flush=True,
    )

    record = {
        "check": "check3_no_homeo_drift",
        "starting_ckpt": str(p2_path),
        "n_trials_learning": n_trials,
        "entropy_no_homeo_drift": out["entropy_nats"],
        "n_bins_no_homeo_drift": out["n_bins_geq_5pct"],
        "coverage": out,
    }
    Path("logs/task74/check3_no_homeo_drift.json").write_text(
        json.dumps(record, indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
