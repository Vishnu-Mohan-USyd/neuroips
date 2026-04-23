"""Task #74 β-mechanism — Step 3: Level 11 gate with LEARNED W_q_gain.

Loads a phase3_kok checkpoint that was trained with the Step-2-validated
three-factor rule on ``l23_e.W_q_gain`` (``--enable-w-q-gain-rule``), and
reports the 4-condition Level-11 assay **without** re-initialising
``W_q_gain``. This is the Step-3 verdict from the Lead's bottom-up
β-mechanism protocol: if the learned gain drives the correct expectation
effect (Δr negative, pref-units-suppressed-more asymmetry), the whole
β-pipeline (mechanism + rule + integration) is validated end-to-end.

Emits DM-ready single-line summary::

    step3_full_pipeline: final_mean_gain_pref=<#> final_mean_gain_nonpref=<#>
        Δr_matched_vs_mismatched=<#> Δr_sign=<correct|inverse|null>
        magnitude=<#>

Key contract differences from ``step1_beta_level11.py``:

1. ``W_q_gain`` is **loaded** from the ckpt's explicit key (added by the
   trainer in Step 3; buffer is non-persistent so it is saved separately
   from ``state_dict``) and installed into ``bundle.net.l23_e.W_q_gain``.
   The hand-crafting step is skipped.
2. Task weights are **not** zeroed — the trainer already trained
   ``W_qm_task`` and left ``W_mh_task_exc`` at zero (Fix J disabled). In
   this ckpt ``W_mh_task_exc`` ≡ 0 so the task_exc apical readout delivers
   zero drive; any expectation effect observed here is attributable to
   ``W_q_gain`` alone.
3. ``final_mean_gain_pref`` / ``final_mean_gain_nonpref`` are computed
   against per-unit preferred orientation from the same cue-free localiser
   used in Step 1, to let Lead compare the learned gain against the
   hand-crafted 1 − g0 = 0.7 target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.step1_beta_level11 import (
    _NULL_EPS, _run_localizer_trial, _run_probe_trial_with_gate,
    _sign_label, _unit_pref_mask,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, cue_mapping_from_seed,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="phase3_kok ckpt produced with "
                         "--enable-w-q-gain-rule + --disable-fix-j-mh-exc.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials-per-cell", type=int, default=30)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-localizer-orients", type=int, default=36)
    ap.add_argument("--n-localizer-trials", type=int, default=8)
    ap.add_argument("--pref-tol-deg", type=float, default=15.0)
    ap.add_argument(
        "--output", type=Path,
        default=Path("logs/task77/step3_beta_level11.json"),
    )
    args = ap.parse_args()

    # --- Load ckpt + install LEARNED W_q_gain --------------------------
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device="cpu")
    bundle.net.set_phase("phase3_kok")

    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "W_q_gain" not in raw_ckpt:
        raise KeyError(
            "checkpoint missing 'W_q_gain' key — retrain with "
            "--enable-w-q-gain-rule using the Step-3 trainer."
        )
    W_q_gain_learned = raw_ckpt["W_q_gain"].to(
        dtype=bundle.net.l23_e.W_q_gain.dtype,
        device=bundle.net.l23_e.W_q_gain.device,
    )
    if W_q_gain_learned.shape != bundle.net.l23_e.W_q_gain.shape:
        raise ValueError(
            f"W_q_gain shape mismatch: ckpt={tuple(W_q_gain_learned.shape)} "
            f"vs net={tuple(bundle.net.l23_e.W_q_gain.shape)}"
        )
    with torch.no_grad():
        bundle.net.l23_e.W_q_gain.data.copy_(W_q_gain_learned)

    timing = KokTiming()
    cue_mapping = cue_mapping_from_seed(args.seed)

    # --- Localiser (cue-free) → per-unit preferred orientation ---------
    orients = np.linspace(0.0, 180.0, args.n_localizer_orients, endpoint=False)
    gen_loc = torch.Generator().manual_seed(args.seed + 1)
    loc_rates: list[np.ndarray] = []
    loc_y: list[float] = []
    for theta in orients:
        for _ in range(args.n_localizer_trials):
            r = _run_localizer_trial(
                bundle, probe_orientation_deg=float(theta),
                timing=timing, noise_std=args.noise_std, generator=gen_loc,
            )
            loc_rates.append(r.cpu().numpy())
            loc_y.append(float(theta))
    L = np.stack(loc_rates, axis=0)
    L_y = np.asarray(loc_y)
    orient_mean = np.stack([
        L[np.abs(L_y - o) < 1e-6].mean(axis=0) for o in orients
    ], axis=0)
    unit_pref_idx = np.argmax(orient_mean, axis=0)
    unit_pref_deg = orients[unit_pref_idx]

    # --- Summarise learned W_q_gain on pref / non-pref units ----------
    # Mirror of step1 hand-craft mask: "pref for cue c" = units whose
    # preferred orientation is within tol of cue_mapping[c]. Everything
    # else is "non-pref" for that cue. We average the learned gain over
    # each set and report per-cue plus grand-mean.
    pref_gains: list[float] = []
    nonpref_gains: list[float] = []
    per_cue: dict[int, dict[str, float]] = {}
    for c in (0, 1):
        exp_deg = float(cue_mapping[int(c)])
        pref_m = _unit_pref_mask(unit_pref_deg, exp_deg, tol=args.pref_tol_deg)
        nonpref_m = ~pref_m
        row = bundle.net.l23_e.W_q_gain[c].detach().cpu().numpy()
        g_pref = float(row[pref_m].mean()) if pref_m.any() else float("nan")
        g_nonpref = float(row[nonpref_m].mean()) if nonpref_m.any() else float("nan")
        pref_gains.append(g_pref)
        nonpref_gains.append(g_nonpref)
        per_cue[c] = {
            "n_pref": int(pref_m.sum()),
            "g_pref": g_pref,
            "g_nonpref": g_nonpref,
        }
    final_mean_gain_pref = float(np.mean(pref_gains))
    final_mean_gain_nonpref = float(np.mean(nonpref_gains))

    # --- Level 11 main assay with β gate active during probe ----------
    gen = torch.Generator().manual_seed(args.seed)
    R_list: list[np.ndarray] = []
    y_probe: list[int] = []
    y_matched: list[int] = []
    for cue_id in (0, 1):
        for probe_deg in CUE_ORIENTATIONS_DEG:
            matched = int(
                abs(cue_mapping[int(cue_id)] - float(probe_deg)) < 1e-6
            )
            for _ in range(args.n_trials_per_cell):
                r = _run_probe_trial_with_gate(
                    bundle, cue_id=int(cue_id),
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=args.noise_std, generator=gen,
                )
                R_list.append(r.cpu().numpy())
                y_probe.append(
                    0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6
                    else 1
                )
                y_matched.append(matched)
    R = np.stack(R_list, axis=0)
    y_matched_arr = np.asarray(y_matched, dtype=np.int64)
    y_probe_arr = np.asarray(y_probe, dtype=np.int64)

    mean_r_matched = float(R[y_matched_arr == 1].mean())
    mean_r_mismatched = float(R[y_matched_arr == 0].mean())
    delta_r = mean_r_matched - mean_r_mismatched
    r_sign = _sign_label(delta_r, correct_is_negative=True)

    # --- Pref / non-pref asymmetry (same as step1) --------------------
    pref_exp_l, pref_unexp_l = [], []
    nonpref_exp_l, nonpref_unexp_l = [], []
    for probe_deg in CUE_ORIENTATIONS_DEG:
        pc = 0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6 else 1
        other = (
            CUE_ORIENTATIONS_DEG[1] if pc == 0 else CUE_ORIENTATIONS_DEG[0]
        )
        pref_m = _unit_pref_mask(unit_pref_deg, float(probe_deg))
        nonpref_m = _unit_pref_mask(unit_pref_deg, float(other))
        sel_m = (y_probe_arr == pc) & (y_matched_arr == 1)
        sel_u = (y_probe_arr == pc) & (y_matched_arr == 0)
        if pref_m.any():
            pref_exp_l.append(float(R[np.ix_(sel_m, pref_m)].mean()))
            pref_unexp_l.append(float(R[np.ix_(sel_u, pref_m)].mean()))
        if nonpref_m.any():
            nonpref_exp_l.append(float(R[np.ix_(sel_m, nonpref_m)].mean()))
            nonpref_unexp_l.append(float(R[np.ix_(sel_u, nonpref_m)].mean()))
    pe = float(np.mean(pref_exp_l))
    pu = float(np.mean(pref_unexp_l))
    ne = float(np.mean(nonpref_exp_l))
    nu = float(np.mean(nonpref_unexp_l))
    pref_ok = pe <= pu + _NULL_EPS
    nonpref_ok = ne + _NULL_EPS >= nu
    if abs(pe - pu) < _NULL_EPS and abs(ne - nu) < _NULL_EPS:
        asym_sign = "null"
    elif pref_ok and nonpref_ok:
        asym_sign = "correct"
    else:
        asym_sign = "inverse"

    verdict = (
        "pass" if (r_sign == "correct" and asym_sign == "correct") else "fail"
    )

    line = (
        f"step3_full_pipeline: "
        f"final_mean_gain_pref={final_mean_gain_pref:.4f} "
        f"final_mean_gain_nonpref={final_mean_gain_nonpref:.4f} "
        f"delta_r_matched_vs_mismatched={delta_r:+.5f} "
        f"delta_r_sign={r_sign} "
        f"magnitude={abs(delta_r):.5f} "
        f"asym_sign={asym_sign} verdict={verdict}"
    )
    print(line)
    print(
        f"  mean_r_matched={mean_r_matched:.5f} "
        f"mean_r_mismatched={mean_r_mismatched:.5f}"
    )
    print(
        f"  pref_exp={pe:.4f} pref_unexp={pu:.4f} "
        f"nonpref_exp={ne:.4f} nonpref_unexp={nu:.4f}"
    )
    for c in (0, 1):
        print(
            f"  cue{c}: g_pref={per_cue[c]['g_pref']:.4f} "
            f"g_nonpref={per_cue[c]['g_nonpref']:.4f} "
            f"(n_pref={per_cue[c]['n_pref']})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "ckpt": str(args.checkpoint),
            "seed": args.seed,
            "cue_mapping": {str(k): v for k, v in cue_mapping.items()},
            "final_mean_gain_pref": final_mean_gain_pref,
            "final_mean_gain_nonpref": final_mean_gain_nonpref,
            "per_cue": {str(k): v for k, v in per_cue.items()},
            "mean_r_matched": mean_r_matched,
            "mean_r_mismatched": mean_r_mismatched,
            "delta_r": delta_r,
            "delta_r_sign": r_sign,
            "magnitude": abs(delta_r),
            "pref_exp": pe, "pref_unexp": pu,
            "nonpref_exp": ne, "nonpref_unexp": nu,
            "asym_sign": asym_sign,
            "verdict": verdict,
        }, f, indent=2)
    print(f"  JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
