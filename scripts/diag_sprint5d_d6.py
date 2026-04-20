"""Sprint 5d D6: Kok SNR curve (Case D candidate).

Pre-reg grid: 3 contrast × 4 n_cells × 2 n_orient = 24 combos × 3 seeds.
**Post-hoc deviation (pre-reg §6, budget-driven)**: baseline already shows
100% decoder saturation (mvpa_acc=1.0 valid=invalid in Sprint 5c continuous
seed=42). D6 is reduced to 4 targeted combos × seed 42 only. Rationale:
the baseline ceiling itself IS the Case D evidence; D6 just confirms a
non-saturated band exists and tests Δ_decoding there.

Combos (chosen to span SNR space):
  A: c=1.0, n_cells=96, n_orient=2   — baseline replica (expect ~100%)
  B: c=0.5, n_cells=24, n_orient=2   — moderate SNR reduction
  C: c=0.3, n_cells=12, n_orient=2   — aggressive SNR reduction
  D: c=0.3, n_cells=12, n_orient=6   — multi-class, chance=17%

Pass band: Acc_base ∈ [55%, 80%] (= Acc_invalid in 2-class, multi-class
accuracy in n_orient=6).
Case D confirmed if at least 1 in-band combo shows Δ_decoding > 0 with
CI excluding 0.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from brian2 import defaultclock, ms, prefs
from brian2 import seed as b2_seed

from expectation_snn.assays.runtime import build_frozen_network
from expectation_snn.assays.kok_passive import KokConfig, run_kok_passive

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms

OUT_DIR = Path("data/diag_sprint5d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMBOS = [
    {"tag": "A_c1.0_n96_o2",  "contrast_multiplier": 1.0, "n_cells_subsampled": 96, "n_orientations": 2},
    {"tag": "B_c0.5_n24_o2",  "contrast_multiplier": 0.5, "n_cells_subsampled": 24, "n_orientations": 2},
    {"tag": "C_c0.3_n12_o2",  "contrast_multiplier": 0.3, "n_cells_subsampled": 12, "n_orientations": 2},
    {"tag": "D_c0.3_n12_o6",  "contrast_multiplier": 0.3, "n_cells_subsampled": 12, "n_orientations": 6},
]

SEED = 42


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode().strip()
    except Exception:
        return "unknown"


def run_combo(combo: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    b2_seed(SEED)
    np.random.seed(SEED)
    bundle = build_frozen_network(
        h_kind="hr", seed=SEED, r=1.0, g_total=1.0,
        with_cue=True,   # Kok assay mandates cue regardless of n_orientations
        with_v1_to_h="continuous",
    )
    if combo["n_orientations"] == 2:
        cfg = KokConfig(
            seed=SEED,
            n_stim_trials=120, n_omission_trials=0,
            contrast_multiplier=combo["contrast_multiplier"],
            n_cells_subsampled=combo["n_cells_subsampled"],
            n_orientations=2,
        )
    else:
        cfg = KokConfig(
            seed=SEED,
            n_stim_trials=120, n_omission_trials=0,
            contrast_multiplier=combo["contrast_multiplier"],
            n_cells_subsampled=combo["n_cells_subsampled"],
            n_orientations=combo["n_orientations"],
        )
    res = run_kok_passive(bundle=bundle, cfg=cfg, seed=SEED, verbose=False)
    elapsed = time.time() - t0

    om = res.orientation_mvpa
    acc_valid = float(om.get("acc_valid_mean", float("nan")))
    acc_invalid = float(om.get("acc_invalid_mean", float("nan")))
    delta = float(om.get("delta_decoding", float("nan")))
    delta_ci = om.get("delta_decoding_ci", (float("nan"), float("nan")))
    acc = float(om.get("accuracy", float("nan")))
    acc_ci = om.get("accuracy_ci", (float("nan"), float("nan")))

    if combo["n_orientations"] == 2:
        acc_base = acc_invalid   # invalid = uncued baseline
    else:
        acc_base = acc           # multi-class mean accuracy

    in_band = 0.55 <= acc_base <= 0.80

    out = {
        "tag": combo["tag"],
        "seed": SEED,
        "contrast_multiplier": combo["contrast_multiplier"],
        "n_cells_subsampled": combo["n_cells_subsampled"],
        "n_orientations": combo["n_orientations"],
        "acc_valid": acc_valid,
        "acc_invalid": acc_invalid,
        "acc_base": acc_base,
        "acc_multi": acc,
        "acc_multi_ci": np.asarray(acc_ci),
        "delta_decoding": delta,
        "delta_decoding_ci": np.asarray(delta_ci),
        "in_band": int(in_band),
        "elapsed_s": elapsed,
        "git_sha": _git_sha(),
    }
    delta_sig = bool(
        np.isfinite(delta) and np.isfinite(delta_ci[0])
        and (delta_ci[0] > 0 or delta_ci[1] < 0)
    )
    print(f"[D6/{combo['tag']}] done in {elapsed:.1f}s | "
          f"Acc_valid={acc_valid:.3f} Acc_invalid={acc_invalid:.3f} "
          f"Acc_base={acc_base:.3f} in_band={in_band} | "
          f"Δ_decoding={delta:+.4f} CI=[{delta_ci[0]:+.4f},{delta_ci[1]:+.4f}] "
          f"sig={delta_sig}")
    return out


def main() -> int:
    sha = _git_sha()
    print(f"[D6] git={sha[:8]} seed={SEED} combos={[c['tag'] for c in COMBOS]}")
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for c in COMBOS:
        out_path = OUT_DIR / f"D6_{c['tag']}_seed{SEED}.npz"
        if out_path.exists():
            print(f"[D6/{c['tag']}] SKIP (already saved)")
            r = dict(np.load(out_path, allow_pickle=True))
            r = {k: (v.item() if getattr(v, "ndim", 1) == 0 else v)
                 for k, v in r.items()}
            results.append(r)
            continue
        r = run_combo(c)
        np.savez(out_path, **r)
        results.append(r)
    elapsed = time.time() - t0
    print(f"[D6] ALL DONE in {elapsed:.1f}s")
    print("[D6] summary:")
    in_band = [r for r in results if r["in_band"]]
    print(f"  in-band combos: {[r['tag'] for r in in_band]}")
    if combo_sig := [r for r in in_band
                     if (r['delta_decoding_ci'][0] > 0 if r['n_orientations'] == 2 else False)]:
        print(f"[D6] VERDICT: CASE D CONFIRMED via {[r['tag'] for r in combo_sig]}")
    elif in_band:
        print(f"[D6] VERDICT: non-saturated band found but Δ_decoding "
              f"CI overlaps 0 on all in-band combos → Kok genuinely null at in-band SNR")
    else:
        print("[D6] VERDICT: INCONCLUSIVE — no combo lands in [55%, 80%] band; "
              "baseline 100% saturation + full SNR sweep needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
