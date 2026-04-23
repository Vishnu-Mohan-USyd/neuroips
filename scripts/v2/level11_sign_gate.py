"""Task #74 Level 11 — expectation-sign gate on Phase-2 substrate.

Tests whether the Phase-2 substrate, WITH ALL TASK WEIGHTS ZEROED
(W_qm_task = W_mh_task_exc = W_mh_task_inh = 0), produces the
predictive-coding sign: matched cue+probe should show LOWER L2/3E
population amplitude than mismatched.

Design: SAME probe orientation × DIFFERENT cue contexts. Isolates
expectation from cue–orient binding. For each probe∈{45°,135°} and
cue_id∈{0,1}: 30 trials. Trial is "matched" when cue_mapping[cue_id]
equals the probe orientation, else "mismatched".

Metrics (ALL computed across n_l23_e units):
  * mean_r_matched   (grand mean L23E probe-epoch rate, matched trials)
  * mean_r_mismatched
  * decode_matched, decode_mismatched — 5-fold CV LinearSVC on probe
    orientation using only trials of each condition (can we read
    probe orient from L23E under matched vs mismatched cues?)
  * pref/non-pref asymmetry from a CUE-FREE localizer (36 orients);
    pass criterion is pref_matched ≤ pref_mismatched AND
    nonpref_matched ≥ nonpref_mismatched.

Verdict (printed as single DM-ready line):
    level11_sign_gate: mean_r_exp=<#> mean_r_unexp=<#>
      Δr_sign=<correct|inverse|null>
      decode_exp=<#> decode_unexp=<#>
      Δdecode_sign=<correct|inverse|null>
      asym_sign=<correct|inverse|null>
      verdict=<pass|fail>

correct = matches PC prediction (exp<unexp for energy; exp>unexp for
decode; pref_exp<pref_unexp AND nonpref_exp>=nonpref_unexp for asym).
inverse = opposite sign. null = |Δ| < 1e-4 (effectively no signal —
expected when task weights are zero and cue cannot reach the circuit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_kok import (
    run_kok_localizer_trial, run_kok_probe_trial,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, cue_mapping_from_seed,
)


_NULL_EPS = 1e-4


def _sign_label(delta: float, *, correct_is_negative: bool) -> str:
    """Classify Δ = matched − mismatched against PC prediction.

    For energy: PC predicts matched < mismatched → correct when Δ<0.
    For decode: PC predicts matched > mismatched → correct when Δ>0.
    """
    if abs(delta) < _NULL_EPS:
        return "null"
    if correct_is_negative:
        return "correct" if delta < 0 else "inverse"
    return "correct" if delta > 0 else "inverse"


def _svm_5fold(X: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    if X.shape[0] < 10 or len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    accs = []
    for tr, te in skf.split(X, y):
        clf = LinearSVC(random_state=int(seed), max_iter=5000, dual="auto")
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return float(np.mean(accs))


def run_gate(
    *, ckpt_path: Path, seed: int, n_trials_per_cell: int,
    noise_std: float, n_localizer_orients: int,
    n_localizer_trials: int, out_path: Path,
) -> dict:
    bundle = load_checkpoint(ckpt_path, seed=seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    # --- ZERO the three task weights ------------------------------------
    cm = bundle.net.context_memory
    with torch.no_grad():
        w_qm_norm_before = float(cm.W_qm_task.detach().norm().item())
        w_mhe_norm_before = float(cm.W_mh_task_exc.detach().norm().item())
        w_mhi_norm_before = float(cm.W_mh_task_inh.detach().norm().item())
        cm.W_qm_task.data.zero_()
        cm.W_mh_task_exc.data.zero_()
        cm.W_mh_task_inh.data.zero_()
        assert float(cm.W_qm_task.detach().abs().max().item()) == 0.0
        assert float(cm.W_mh_task_exc.detach().abs().max().item()) == 0.0
        assert float(cm.W_mh_task_inh.detach().abs().max().item()) == 0.0

    timing = KokTiming()
    cue_mapping = cue_mapping_from_seed(seed)  # {0: orient, 1: orient}

    gen = torch.Generator().manual_seed(seed)

    # --- Main assay: 4 conditions × n_trials ----------------------------
    # cue_id ∈ {0,1} × probe_orient ∈ {45,135}. matched = cue_mapping[cue]==probe.
    conditions = []
    r_mat: list[np.ndarray] = []     # [n_cond_trials, n_l23]
    y_probe: list[int] = []          # probe orient label (0=45,1=135)
    y_matched: list[int] = []        # 1 if matched else 0
    for cue_id in (0, 1):
        for probe_deg in CUE_ORIENTATIONS_DEG:
            matched = int(
                abs(cue_mapping[int(cue_id)] - float(probe_deg)) < 1e-6
            )
            for _ in range(n_trials_per_cell):
                r = run_kok_probe_trial(
                    bundle, cue_id=int(cue_id),
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=noise_std, generator=gen,
                )
                r_mat.append(r.cpu().numpy())
                y_probe.append(
                    0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6
                    else 1
                )
                y_matched.append(matched)
            conditions.append(
                {"cue_id": int(cue_id), "probe_deg": float(probe_deg),
                 "matched": bool(matched)}
            )
    R = np.stack(r_mat, axis=0)                # [n_all_trials, n_l23]
    y_probe_arr = np.asarray(y_probe, dtype=np.int64)
    y_matched_arr = np.asarray(y_matched, dtype=np.int64)

    # Population-mean energy
    mean_r_matched = float(R[y_matched_arr == 1].mean())
    mean_r_mismatched = float(R[y_matched_arr == 0].mean())
    delta_r = mean_r_matched - mean_r_mismatched

    # Decode probe orientation within matched vs within mismatched
    decode_matched = _svm_5fold(
        R[y_matched_arr == 1], y_probe_arr[y_matched_arr == 1], seed=seed,
    )
    decode_mismatched = _svm_5fold(
        R[y_matched_arr == 0], y_probe_arr[y_matched_arr == 0], seed=seed,
    )
    delta_dec = float(decode_matched - decode_mismatched)

    # --- Localizer (cue-free) for pref assignment -----------------------
    orients = np.linspace(
        0.0, 180.0, n_localizer_orients, endpoint=False,
    )
    loc_mat: list[np.ndarray] = []
    loc_y: list[float] = []
    gen_loc = torch.Generator().manual_seed(seed + 1)
    for theta in orients:
        for _ in range(n_localizer_trials):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(theta),
                timing=timing, noise_std=noise_std, generator=gen_loc,
            )
            loc_mat.append(r.cpu().numpy())
            loc_y.append(float(theta))
    L = np.stack(loc_mat, axis=0)
    L_y = np.asarray(loc_y)

    # Per-unit pref from localizer: argmax over orient means
    orient_mean_per_unit = np.stack([
        L[np.abs(L_y - o) < 1e-6].mean(axis=0) for o in orients
    ], axis=0)                                   # [n_orients, n_l23]
    unit_pref_idx = np.argmax(orient_mean_per_unit, axis=0)   # [n_l23]
    unit_pref_deg = orients[unit_pref_idx]                    # [n_l23]

    # For each of the two main-assay probe orientations, find units
    # preferring that orientation (within ±15° tolerance).
    def _unit_mask(anchor_deg: float, tol: float = 15.0) -> np.ndarray:
        d = np.abs(
            ((unit_pref_deg - anchor_deg + 90.0) % 180.0) - 90.0
        )
        return d <= tol

    pref_vals_exp: list[float] = []
    pref_vals_unexp: list[float] = []
    nonpref_vals_exp: list[float] = []
    nonpref_vals_unexp: list[float] = []
    for probe_deg in CUE_ORIENTATIONS_DEG:
        probe_cls = (
            0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6 else 1
        )
        pref_m = _unit_mask(float(probe_deg))
        other = (
            CUE_ORIENTATIONS_DEG[1] if probe_cls == 0 else CUE_ORIENTATIONS_DEG[0]
        )
        nonpref_m = _unit_mask(float(other))
        trial_sel_matched = (y_probe_arr == probe_cls) & (y_matched_arr == 1)
        trial_sel_mism = (y_probe_arr == probe_cls) & (y_matched_arr == 0)
        if pref_m.any():
            pref_vals_exp.append(
                float(R[np.ix_(trial_sel_matched, pref_m)].mean())
            )
            pref_vals_unexp.append(
                float(R[np.ix_(trial_sel_mism, pref_m)].mean())
            )
        if nonpref_m.any():
            nonpref_vals_exp.append(
                float(R[np.ix_(trial_sel_matched, nonpref_m)].mean())
            )
            nonpref_vals_unexp.append(
                float(R[np.ix_(trial_sel_mism, nonpref_m)].mean())
            )

    pref_exp = float(np.mean(pref_vals_exp)) if pref_vals_exp else float("nan")
    pref_unexp = (
        float(np.mean(pref_vals_unexp)) if pref_vals_unexp else float("nan")
    )
    nonpref_exp = (
        float(np.mean(nonpref_vals_exp)) if nonpref_vals_exp else float("nan")
    )
    nonpref_unexp = (
        float(np.mean(nonpref_vals_unexp)) if nonpref_vals_unexp else float("nan")
    )

    # Asym pass: pref_exp ≤ pref_unexp AND nonpref_exp ≥ nonpref_unexp
    pref_ok = pref_exp <= pref_unexp + _NULL_EPS
    nonpref_ok = nonpref_exp + _NULL_EPS >= nonpref_unexp
    if (
        abs(pref_exp - pref_unexp) < _NULL_EPS
        and abs(nonpref_exp - nonpref_unexp) < _NULL_EPS
    ):
        asym_sign = "null"
    elif pref_ok and nonpref_ok:
        asym_sign = "correct"
    else:
        asym_sign = "inverse"

    r_sign = _sign_label(delta_r, correct_is_negative=True)
    dec_sign = _sign_label(delta_dec, correct_is_negative=False)

    # Pass if ALL three signals are "correct"
    verdict = (
        "pass" if (r_sign == "correct"
                   and dec_sign == "correct"
                   and asym_sign == "correct") else "fail"
    )

    result = {
        "level": 11,
        "ckpt": str(ckpt_path),
        "seed": seed,
        "n_trials_per_cell": n_trials_per_cell,
        "noise_std": noise_std,
        "cue_mapping": {str(k): v for k, v in cue_mapping.items()},
        "task_weight_norms_before_zero": {
            "W_qm_task": w_qm_norm_before,
            "W_mh_task_exc": w_mhe_norm_before,
            "W_mh_task_inh": w_mhi_norm_before,
        },
        "mean_r_matched": mean_r_matched,
        "mean_r_mismatched": mean_r_mismatched,
        "delta_r": delta_r,
        "r_sign": r_sign,
        "decode_matched": decode_matched,
        "decode_mismatched": decode_mismatched,
        "delta_decode": delta_dec,
        "decode_sign": dec_sign,
        "pref_expected": pref_exp,
        "pref_unexpected": pref_unexp,
        "nonpref_expected": nonpref_exp,
        "nonpref_unexpected": nonpref_unexp,
        "asym_sign": asym_sign,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Emit DM-ready one-liner
    print(
        f"level11_sign_gate: "
        f"mean_r_exp={mean_r_matched:.4f} "
        f"mean_r_unexp={mean_r_mismatched:.4f} "
        f"Δr_sign={r_sign} "
        f"decode_exp={decode_matched:.3f} "
        f"decode_unexp={decode_mismatched:.3f} "
        f"Δdecode_sign={dec_sign} "
        f"asym_sign={asym_sign} "
        f"verdict={verdict}"
    )
    print(f"  pref_exp={pref_exp:.4f} pref_unexp={pref_unexp:.4f} "
          f"nonpref_exp={nonpref_exp:.4f} nonpref_unexp={nonpref_unexp:.4f}")
    print(f"  delta_r={delta_r:+.4e} delta_dec={delta_dec:+.4f}")
    print(f"  JSON: {out_path}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials-per-cell", type=int, default=30)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-localizer-orients", type=int, default=36)
    ap.add_argument("--n-localizer-trials", type=int, default=8)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    run_gate(
        ckpt_path=args.checkpoint, seed=args.seed,
        n_trials_per_cell=args.n_trials_per_cell,
        noise_std=args.noise_std,
        n_localizer_orients=args.n_localizer_orients,
        n_localizer_trials=args.n_localizer_trials,
        out_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
