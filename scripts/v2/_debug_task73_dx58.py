"""Task #73 Dx5+Dx8 — FIXED-anchor per-bin audit + W_mh_task scale sweep.

DECISIVE experiment to discriminate between surviving hypotheses:
  H1 template-weak-positive : W_mh_task cue-matched; ×scale→monotone Δamp;
                              Δr concentrated on "expected-preferred" bin
  H2 wrong-sign-template    : ×scale→monotone NEGATIVE Δamp (suppresses exp)
  H3 target-layer-mismatch  : ×scale boosts all preferred-orient bins UNIFORMLY
  H-falsified pure DC gain  : ×scale amplifies amp_exp and amp_unexp equally

Fixes Dx2's cross-condition anchor-set confound by computing localizer ONCE
with the UNMODIFIED trained W_mh_task, then freezing preferred_deg[u] and
scoring every scale condition against that fixed anchor set.

Scale sweep: [-3, -1, -0.5, 0, 0.5, 1, 3, 5] × 25 trials × 4 sub-conds = 800
main trials + 12-orient × 12-trial localizer.

Per-scale measurements:
  amp_exp, amp_unexp, Δamp
  Δr_at_expected_bin  (units with |pref_orient - expected| < 15°)
  Δr_at_unexpected_bin
  Δr_at_other_bin
  svm_exp_vs_unex (5-fold CV LinearSVC)
  svm_orient_decode
  ‖b_task‖ at pre-probe (sanity check of scale)
  n_units_at_expected, n_units_at_unexpected (from fixed anchor set)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.eval_kok import (
    run_kok_probe_trial, run_kok_localizer_trial, _compute_localizer_stats,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
)


def compute_fixed_anchor_set(ckpt_path, n_loc_per=12, noise_std=0.01, seed=42):
    """Compute localizer tuning + per-unit preferred_deg WITH UNMODIFIED
    W_mh_task. Returns preferred_deg [n_l23], tuning_curve [n_orient, n_l23].
    """
    bundle = load_checkpoint(ckpt_path, seed=seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    orients = np.linspace(0.0, 180.0, 12, endpoint=False)
    loc_trials, loc_orient = [], []
    for o in orients:
        for _ in range(n_loc_per):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(o),
                timing=timing, noise_std=noise_std, generator=gen,
            )
            loc_trials.append(r.cpu().numpy())
            loc_orient.append(float(o))
    loc_trials = np.stack(loc_trials, axis=0)
    loc_orient = np.asarray(loc_orient)
    stats = _compute_localizer_stats(loc_trials, loc_orient, orients)
    return {
        "preferred_deg": stats["preferred_deg"],
        "tuning_curve": stats["tuning_curve"],
        "peak_rate": stats["peak_rate"],
        "W_mh_task_original": bundle.net.context_memory.W_mh_task.data.clone(),
        "cue_mapping": bundle.meta.get(
            "cue_mapping", cue_mapping_from_seed(seed)),
        "localizer_orients": orients,
    }


def get_b_task_at_pre_probe(bundle, cue_mapping, timing, n_trials=10):
    """Mean ‖W_mh_task@m‖ at t=delay_end-1 across cue0/cue1."""
    cfg = bundle.cfg
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    blank = make_blank_frame(1, cfg, device="cpu")
    b_norms = []
    W = bundle.net.context_memory.W_mh_task.data
    for cue_id in (0, 1):
        q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
        for _ in range(n_trials):
            state = bundle.net.initial_state(batch_size=1)
            for t in range(delay_end):
                q_t = q_cue if t < cue_end else None
                _x, state, info = bundle.net(blank, state, q_t=q_t)
                bundle.net.l23_e.homeostasis.update(state.r_l23)
                bundle.net.h_e.homeostasis.update(state.r_h)
                if t == delay_end - 1:
                    b = (W @ state.m[0]).cpu().numpy()
                    b_norms.append(float(np.linalg.norm(b)))
                    break
    return float(np.mean(b_norms))


def run_scale_condition(ckpt_path, scale, W_original, preferred_deg,
                         cue_mapping, n_trials=25, noise_std=0.01, seed=42):
    """One scale point of the sweep. Loads fresh bundle, sets W_mh_task =
    scale * W_original, runs Kok main assay with fixed anchor set."""
    bundle = load_checkpoint(ckpt_path, seed=seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    bundle.net.context_memory.W_mh_task.data.copy_(scale * W_original)
    W_norm_after = float(
        bundle.net.context_memory.W_mh_task.data.norm().item())

    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)

    trials_l23, orient, expected, cue_ids = [], [], [], []
    for cue_id in (0, 1):
        cue_probe = cue_mapping[cue_id]
        other_probe = cue_mapping[1 - cue_id]
        for probe_deg, is_exp in ((cue_probe, True), (other_probe, False)):
            for _ in range(n_trials):
                r = run_kok_probe_trial(
                    bundle, cue_id=cue_id,
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=noise_std, generator=gen,
                )
                trials_l23.append(r.cpu().numpy())
                orient.append(probe_deg); expected.append(is_exp)
                cue_ids.append(cue_id)
    r_arr = np.stack(trials_l23, axis=0)        # [N, n_l23]
    orient = np.asarray(orient); expected = np.asarray(expected)
    cue_arr = np.asarray(cue_ids)

    # Global amp
    amp_exp = float(r_arr[expected].mean())
    amp_unexp = float(r_arr[~expected].mean())

    # Per-cue delta_amp with expected-bin
    def _ang_diff(a, b):
        d = np.abs(a - b)
        return np.minimum(d, 180.0 - d)

    # For each trial: compute r in 3 bins based on FIXED preferred_deg
    # relative to the EXPECTED orientation (per cue), not the probe orient.
    # We want to know: do units prefering the expected orient see more of
    # the template bias than units preferring unexpected orient?
    expected_per_trial = np.array(
        [cue_mapping[c] for c in cue_arr])
    unexpected_per_trial = np.array(
        [cue_mapping[1 - c] for c in cue_arr])

    # Anchor mask against the CUE-EXPECTED orient (varies per trial, but
    # per-unit preferred_deg is fixed). We'll average per-bin rates per trial.
    TOL = 15.0
    n_units = preferred_deg.shape[0]
    bin_at_expected = np.zeros(r_arr.shape[0])
    bin_at_unexpected = np.zeros(r_arr.shape[0])
    bin_other = np.zeros(r_arr.shape[0])
    count_bins_per_cue = {}
    for cue_id in (0, 1):
        exp_o = cue_mapping[cue_id]
        unexp_o = cue_mapping[1 - cue_id]
        mask_exp = _ang_diff(preferred_deg, exp_o) < TOL
        mask_unexp = _ang_diff(preferred_deg, unexp_o) < TOL
        mask_other = ~(mask_exp | mask_unexp)
        count_bins_per_cue[cue_id] = {
            "n_at_expected": int(mask_exp.sum()),
            "n_at_unexpected": int(mask_unexp.sum()),
            "n_other": int(mask_other.sum()),
        }
        trial_idx = np.where(cue_arr == cue_id)[0]
        for ti in trial_idx:
            r = r_arr[ti]
            bin_at_expected[ti] = (
                float(r[mask_exp].mean()) if mask_exp.any() else np.nan)
            bin_at_unexpected[ti] = (
                float(r[mask_unexp].mean()) if mask_unexp.any() else np.nan)
            bin_other[ti] = (
                float(r[mask_other].mean()) if mask_other.any() else np.nan)

    def _nanmean_safe(arr):
        if arr.size == 0 or not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanmean(arr))

    delta_exp_bin = _nanmean_safe(bin_at_expected[expected]) - \
                    _nanmean_safe(bin_at_expected[~expected])
    delta_unexp_bin = _nanmean_safe(bin_at_unexpected[expected]) - \
                      _nanmean_safe(bin_at_unexpected[~expected])
    delta_other_bin = _nanmean_safe(bin_other[expected]) - \
                      _nanmean_safe(bin_other[~expected])

    # SVMs
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _svm_mean(X, y):
        accs = []
        for tr, te in skf.split(X, y):
            clf = LinearSVC(random_state=42, max_iter=5000, dual="auto")
            clf.fit(X[tr], y[tr])
            accs.append(float(clf.score(X[te], y[te])))
        return float(np.mean(accs))

    y_exp = expected.astype(np.int64)
    svm_exp_vs_unex = _svm_mean(r_arr, y_exp)
    uniq = np.unique(orient)
    y_ori = np.where(orient == uniq[0], 0, 1).astype(np.int64)
    svm_orient = _svm_mean(r_arr, y_ori)

    # ||b_task|| at pre-probe (sanity check of scale)
    b_task_norm = get_b_task_at_pre_probe(
        bundle, cue_mapping, timing, n_trials=5)

    return {
        "scale": scale,
        "W_mh_task_norm_after_mod": W_norm_after,
        "b_task_norm_pre_probe": b_task_norm,
        "amp_expected": amp_exp,
        "amp_unexpected": amp_unexp,
        "delta_amp_global": amp_exp - amp_unexp,
        "delta_r_at_expected_bin": delta_exp_bin,
        "delta_r_at_unexpected_bin": delta_unexp_bin,
        "delta_r_at_other_bin": delta_other_bin,
        "svm_exp_vs_unex": svm_exp_vs_unex,
        "svm_orient": svm_orient,
        "bin_sizes": count_bins_per_cue,
    }


def main(ckpt_path, n_trials=25, n_loc_per=12):
    print(f"[Dx5+Dx8] Computing FIXED anchor set (unmodified W_mh_task)...",
          file=sys.stderr, flush=True)
    anchor = compute_fixed_anchor_set(
        ckpt_path, n_loc_per=n_loc_per, noise_std=0.01, seed=42)
    cue_mapping = {int(k): float(v) for k, v in anchor["cue_mapping"].items()}
    W_original = anchor["W_mh_task_original"]
    preferred = anchor["preferred_deg"]
    orients = anchor["localizer_orients"]
    n_l23 = preferred.shape[0]
    print(f"[Dx5+Dx8] Fixed anchors computed: n_units={n_l23}, "
          f"‖W_original‖={W_original.norm().item():.4f}",
          file=sys.stderr, flush=True)
    # Report anchor distribution
    hist = np.zeros(len(orients))
    for u in range(n_l23):
        i = int(np.argmin(np.abs(orients - preferred[u])))
        hist[i] += 1
    print(f"[Dx5+Dx8] Preferred-orient histogram (units per 15° bin): "
          f"{dict(zip([float(o) for o in orients], [int(h) for h in hist]))}",
          file=sys.stderr, flush=True)

    scales = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0]
    results = []
    for s in scales:
        print(f"[Dx5+Dx8] Running scale={s}...",
              file=sys.stderr, flush=True)
        r = run_scale_condition(
            ckpt_path, s, W_original, preferred, cue_mapping,
            n_trials=n_trials)
        results.append(r)

    # Also record the fixed anchor info
    out = {
        "cue_mapping": cue_mapping,
        "W_mh_task_original_norm": float(W_original.norm().item()),
        "anchor_preferred_histogram": dict(zip(
            [float(o) for o in orients], [int(h) for h in hist])),
        "n_trials_per_subcond": n_trials,
        "scales": scales,
        "per_scale": results,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt"))
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--n-loc-per", type=int, default=12)
    args = p.parse_args()
    main(args.ckpt, args.n_trials, args.n_loc_per)
