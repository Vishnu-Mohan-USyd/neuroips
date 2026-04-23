"""Task #73 Dx2 — same-state causal ablation + injection on Kok ckpt.

Runs lightweight Kok eval under 4 conditions on the Task#70 checkpoint:
  A. baseline
  B. W_mh_task = 0 (ablate the learned template)
  C. W_mh_task × 5 (amplify)
  D. W_mh_task × -1 (invert sign)

Route-to-SOM (E) is NOT tested here — requires network-forward modification.

Per condition: mean r_l23 at probe1 for expected/unexpected, Δamp, Kok
asymmetry (pref − nonpref anchor-based, 15° tolerance), and svm decode
(expected vs unexpected label from r_l23 probe).

Reports a concise JSON dict.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_kok import (
    run_kok_probe_trial, run_kok_localizer_trial,
    _pref_nonpref_from_localizer, _compute_localizer_stats,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed,
)


def run_condition(ckpt_path: Path, mod_fn, n_trials=20, n_loc=4):
    bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cue_mapping = bundle.meta.get("cue_mapping", cue_mapping_from_seed(42))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}
    # Apply modification to W_mh_task
    mod_fn(bundle.net.context_memory.W_mh_task.data)

    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(42)

    trials_l23, orient, expected = [], [], []
    for cue_id in (0, 1):
        cue_probe = cue_mapping[cue_id]
        other_probe = cue_mapping[1 - cue_id]
        for probe_deg, is_exp in ((cue_probe, True), (other_probe, False)):
            for _ in range(n_trials):
                r = run_kok_probe_trial(
                    bundle, cue_id=cue_id,
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=0.01, generator=gen,
                )
                trials_l23.append(r.cpu().numpy())
                orient.append(probe_deg); expected.append(is_exp)
    r_arr = np.stack(trials_l23, axis=0)
    orient = np.asarray(orient); expected = np.asarray(expected)

    amp_exp = float(r_arr[expected].mean())
    amp_unexp = float(r_arr[~expected].mean())

    # Localizer for asymmetry
    loc_orients = np.linspace(0.0, 180.0, 12, endpoint=False)
    loc_trials_list = []
    loc_orient_list = []
    for o in loc_orients:
        for _ in range(n_loc):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(o),
                timing=timing, noise_std=0.01, generator=gen,
            )
            loc_trials_list.append(r.cpu().numpy())
            loc_orient_list.append(float(o))
    loc_trials = np.stack(loc_trials_list, axis=0)
    loc_orient = np.asarray(loc_orient_list)
    loc_stats = _compute_localizer_stats(loc_trials, loc_orient, loc_orients)
    asym = _pref_nonpref_from_localizer(
        r_arr, orient, expected, loc_stats["preferred_deg"])

    # SVM decode expected vs unexpected from r_l23 probe response
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold
    y_exp = expected.astype(np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(r_arr, y_exp):
        clf = LinearSVC(random_state=42, max_iter=5000, dual="auto")
        clf.fit(r_arr[tr], y_exp[tr])
        accs.append(float(clf.score(r_arr[te], y_exp[te])))
    svm_exp_vs_unexp = float(np.mean(accs))

    # SVM decode orientation within each condition
    uniq = np.unique(orient)
    y_ori = np.where(orient == uniq[0], 0, 1).astype(np.int64)
    accs_ori = []
    for tr, te in skf.split(r_arr, y_ori):
        clf = LinearSVC(random_state=42, max_iter=5000, dual="auto")
        clf.fit(r_arr[tr], y_ori[tr])
        accs_ori.append(float(clf.score(r_arr[te], y_ori[te])))
    svm_ori = float(np.mean(accs_ori))

    return {
        "amp_expected": amp_exp,
        "amp_unexpected": amp_unexp,
        "delta_amp": amp_exp - amp_unexp,
        "asymmetry": float(asym.get("asymmetry", float("nan"))),
        "n_pref_lo": int(asym.get("n_units_pref_lo", 0)),
        "n_pref_hi": int(asym.get("n_units_pref_hi", 0)),
        "svm_exp_vs_unexp": svm_exp_vs_unexp,
        "svm_orientation": svm_ori,
        "W_mh_task_norm_after_mod": float(
            bundle.net.context_memory.W_mh_task.data.norm().item()),
    }


def main(ckpt_path: Path, n_trials: int = 20):
    conditions = {
        "A_baseline": lambda w: None,
        "B_zero_W_mh_task": lambda w: w.zero_(),
        "C_x5_W_mh_task": lambda w: w.mul_(5.0),
        "D_invert_W_mh_task": lambda w: w.mul_(-1.0),
    }
    results = {}
    for name, mod in conditions.items():
        print(f"Running {name}...", file=sys.stderr, flush=True)
        results[name] = run_condition(ckpt_path, mod, n_trials=n_trials, n_loc=3)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt"))
    p.add_argument("--n-trials", type=int, default=20)
    args = p.parse_args()
    main(args.ckpt, args.n_trials)
