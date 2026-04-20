"""Sprint 5d D1 — pre-probe prior index.

Pre-reg: expectation_snn/docs/SPRINT_5D_DIAG_PREREG.md §3 D1.
Writes data/diag_sprint5d/D1_{kok,richter,tang}_seed{42,43,44}.npz.
"""
from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import numpy as np

# Brian2 imports (avoided at module level when the file is imported for tests)
from expectation_snn.assays.runtime import (
    build_frozen_network, STAGE2_CUE_CHANNELS,
)
from expectation_snn.assays.kok_passive import KokConfig, run_kok_passive
from expectation_snn.assays.richter_crossover import (
    RichterConfig, run_richter_crossover,
)
from expectation_snn.assays.tang_rotating import TangConfig, run_tang_rotating
from expectation_snn.brian2_model.stimulus import TANG_ORIENTATIONS_DEG

N_CHANNELS = 12
OUT_DIR = "data/diag_sprint5d"
os.makedirs(OUT_DIR, exist_ok=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def theta_rad_to_channel(theta_rad: float) -> int:
    """Map θ ∈ [0, π) → nearest of 12 H channels (orientations = ch · π/12)."""
    ch = int(np.round((float(theta_rad) % np.pi) / (np.pi / N_CHANNELS))) % N_CHANNELS
    return ch


def far_channel(ch: int) -> int:
    return (ch + N_CHANNELS // 2) % N_CHANNELS


def _cohens_d(x: np.ndarray) -> float:
    s = np.std(x, ddof=1) if x.size > 1 else 0.0
    if s == 0.0:
        return 0.0
    return float(np.mean(x) / s)


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    return _cohens_d(d)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 99_999,
                  ci: float = 95.0):
    """Return (lo, hi) of bootstrap mean CI."""
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Kok D1
# ---------------------------------------------------------------------------

def run_d1_kok(trial_seed: int, n_stim: int = 60):
    """Kok pre-probe prior: 100 ms end-of-gap window."""
    print(f"[D1/Kok seed={trial_seed}] building bundle + running assay ...",
          flush=True)
    t0 = time.time()
    bundle = build_frozen_network(
        h_kind="hr", seed=42, r=1.0, g_total=1.0, with_cue=True,
        with_v1_to_h="continuous", with_preprobe_h_mon=True,
    )
    cfg = KokConfig(
        n_stim_trials=n_stim, n_omission_trials=0,
        seed=trial_seed, preprobe_window_ms=100.0,
    )
    res = run_kok_passive(bundle=bundle, cfg=cfg, seed=trial_seed)

    raw = res.raw
    hp = np.asarray(raw["h_preprobe_rate_hz"])  # (n_trials, 12)
    cue_per_trial = np.asarray(raw["cue_per_trial"])  # 'A'/'B' strings
    theta_per_trial = np.asarray(raw["theta_per_trial"])  # radians
    expected_per_trial = np.asarray(raw["expected_per_trial"])  # radians

    ch_A, ch_B = int(STAGE2_CUE_CHANNELS[0]), int(STAGE2_CUE_CHANNELS[1])

    # Expected channel = the channel predicted by the cue (not the stim).
    # Reference = the channel of the OTHER cue (distractor).
    n_trials = hp.shape[0]
    exp_ch = np.zeros(n_trials, dtype=np.int64)
    ref_ch = np.zeros(n_trials, dtype=np.int64)
    far_ch_arr = np.zeros(n_trials, dtype=np.int64)
    current_stim_ch = np.zeros(n_trials, dtype=np.int64)
    for k in range(n_trials):
        if cue_per_trial[k] == "A":
            exp_ch[k] = ch_A
            ref_ch[k] = ch_B
        else:
            exp_ch[k] = ch_B
            ref_ch[k] = ch_A
        far_ch_arr[k] = far_channel(exp_ch[k])
        # During the gap window the stimulus is OFF — but the impending stim
        # is theta_per_trial. For the "current" comparison in the specificity
        # check we use the cue target (= expected_per_trial channel) vs the
        # invalid-cue channel (= reference). The cue is the only "current"
        # signal during the gap.
        current_stim_ch[k] = theta_rad_to_channel(theta_per_trial[k])

    PI_trial = hp[np.arange(n_trials), exp_ch] - hp[np.arange(n_trials), ref_ch]
    H_exp_abs = hp[np.arange(n_trials), exp_ch]
    H_ref = hp[np.arange(n_trials), ref_ch]
    H_far = hp[np.arange(n_trials), far_ch_arr]

    d_exp_vs_ref = _cohens_d_paired(H_exp_abs, H_ref)
    d_exp_vs_far = _cohens_d_paired(H_exp_abs, H_far)

    PI_mean = float(np.mean(PI_trial))
    PI_sd = float(np.std(PI_trial, ddof=1) if PI_trial.size > 1 else 0.0)
    PI_d = _cohens_d(PI_trial)
    PI_ci = _bootstrap_ci(PI_trial)

    out = {
        "assay": "kok",
        "trial_seed": trial_seed,
        "checkpoint_seed": 42,
        "n_trials": n_trials,
        "preprobe_window_ms": 100.0,
        "h_preprobe_rate_hz": hp,
        "exp_ch": exp_ch,
        "ref_ch": ref_ch,
        "far_ch": far_ch_arr,
        "current_stim_ch": current_stim_ch,
        "cue_per_trial": cue_per_trial,
        "PI_trial": PI_trial,
        "PI_mean": PI_mean,
        "PI_sd": PI_sd,
        "PI_cohens_d": PI_d,
        "PI_ci_lo": PI_ci[0],
        "PI_ci_hi": PI_ci[1],
        "H_expected_abs_mean": float(np.mean(H_exp_abs)),
        "H_ref_mean": float(np.mean(H_ref)),
        "H_far_mean": float(np.mean(H_far)),
        "max_H_any_channel": float(np.max(hp)),
        "d_expected_vs_current": float(d_exp_vs_ref),
        "d_expected_vs_far": float(d_exp_vs_far),
        "elapsed_s": time.time() - t0,
        "git_sha": _git_sha(),
    }
    # Pre-registered 4-check gate
    out["check_abs"] = bool(out["H_expected_abs_mean"] >= 5.0)
    out["check_PI_mag"] = bool(
        (out["PI_mean"] >= 2.0)
        and (PI_ci[0] > 0.0)
    )
    out["check_d_PI"] = bool(out["PI_cohens_d"] >= 0.5)
    out["check_specificity"] = bool(
        (out["d_expected_vs_current"] >= 0.5)
        and (out["d_expected_vs_far"] >= 0.5)
    )
    out["prior_pass"] = bool(
        out["check_abs"] and out["check_PI_mag"]
        and out["check_d_PI"] and out["check_specificity"]
    )
    fp = f"{OUT_DIR}/D1_kok_seed{trial_seed}.npz"
    np.savez_compressed(fp, **{
        k: v for k, v in out.items()
        if not isinstance(v, (dict, list, tuple)) or k in ("cue_per_trial",)
    })
    print(
        f"[D1/Kok seed={trial_seed}] done in {out['elapsed_s']:.1f}s | "
        f"H_exp_abs={out['H_expected_abs_mean']:.2f} Hz | "
        f"PI_mean={PI_mean:+.3f} CI=[{PI_ci[0]:+.3f},{PI_ci[1]:+.3f}] | "
        f"d_PI={PI_d:+.3f} | d(exp-ref)={d_exp_vs_ref:+.3f} | "
        f"d(exp-far)={d_exp_vs_far:+.3f} | PASS={out['prior_pass']}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Richter D1
# ---------------------------------------------------------------------------

def run_d1_richter(trial_seed: int, reps_exp: int = 10, reps_unexp: int = 3):
    """Richter pre-probe prior: last 100 ms of leader."""
    print(f"[D1/Richter seed={trial_seed}] building bundle + running ...",
          flush=True)
    t0 = time.time()
    bundle = build_frozen_network(
        h_kind="hr", seed=42, r=1.0, g_total=1.0, with_cue=False,
        with_v1_to_h="continuous", with_preprobe_h_mon=True,
    )
    cfg = RichterConfig(
        reps_expected=reps_exp, reps_unexpected=reps_unexp,
        seed=trial_seed, preprobe_window_ms=100.0,
    )
    res = run_richter_crossover(bundle=bundle, cfg=cfg, seed=trial_seed)

    raw = res.raw
    hp = np.asarray(raw["h_preprobe_rate_hz"])  # (n_trials, 12)
    theta_L = np.asarray(raw["theta_L"])
    theta_T = np.asarray(raw["theta_T"])
    cond_mask = np.asarray(raw["cond_mask"])  # 1=expected, 0=unexpected

    n_trials = hp.shape[0]
    exp_ch = np.array([theta_rad_to_channel(t) for t in theta_T])
    ref_ch = np.array([theta_rad_to_channel(t) for t in theta_L])
    far_ch_arr = np.array([far_channel(c) for c in ref_ch])

    # Per pre-reg §3 D1: Richter PI = H[expected_trailer] − H[leader].
    PI_trial = hp[np.arange(n_trials), exp_ch] - hp[np.arange(n_trials), ref_ch]
    H_exp_abs = hp[np.arange(n_trials), exp_ch]
    H_ref = hp[np.arange(n_trials), ref_ch]
    H_far = hp[np.arange(n_trials), far_ch_arr]

    # Split by condition (expected vs unexpected) — both contribute to the
    # primary gate, but report separately too.
    exp_mask = cond_mask == 1
    unexp_mask = cond_mask == 0

    out = {
        "assay": "richter",
        "trial_seed": trial_seed,
        "checkpoint_seed": 42,
        "n_trials": n_trials,
        "preprobe_window_ms": 100.0,
        "h_preprobe_rate_hz": hp,
        "exp_ch": exp_ch,
        "ref_ch": ref_ch,
        "far_ch": far_ch_arr,
        "theta_L": theta_L,
        "theta_T": theta_T,
        "cond_mask": cond_mask,
        "PI_trial": PI_trial,
        "PI_mean": float(np.mean(PI_trial)),
        "PI_sd": float(np.std(PI_trial, ddof=1) if PI_trial.size > 1 else 0.0),
        "PI_cohens_d": _cohens_d(PI_trial),
        "PI_ci_lo": _bootstrap_ci(PI_trial)[0],
        "PI_ci_hi": _bootstrap_ci(PI_trial)[1],
        "H_expected_abs_mean": float(np.mean(H_exp_abs)),
        "H_ref_mean": float(np.mean(H_ref)),
        "H_far_mean": float(np.mean(H_far)),
        "max_H_any_channel": float(np.max(hp)),
        "d_expected_vs_current": _cohens_d_paired(H_exp_abs, H_ref),
        "d_expected_vs_far": _cohens_d_paired(H_exp_abs, H_far),
        # Stratified
        "PI_mean_expected_trials": float(np.mean(PI_trial[exp_mask])) if exp_mask.any() else float("nan"),
        "PI_mean_unexpected_trials": float(np.mean(PI_trial[unexp_mask])) if unexp_mask.any() else float("nan"),
        "elapsed_s": time.time() - t0,
        "git_sha": _git_sha(),
    }
    out["check_abs"] = bool(out["H_expected_abs_mean"] >= 5.0)
    out["check_PI_mag"] = bool(
        (out["PI_mean"] >= 2.0) and (out["PI_ci_lo"] > 0.0)
    )
    out["check_d_PI"] = bool(out["PI_cohens_d"] >= 0.5)
    out["check_specificity"] = bool(
        (out["d_expected_vs_current"] >= 0.5)
        and (out["d_expected_vs_far"] >= 0.5)
    )
    out["prior_pass"] = bool(
        out["check_abs"] and out["check_PI_mag"]
        and out["check_d_PI"] and out["check_specificity"]
    )
    fp = f"{OUT_DIR}/D1_richter_seed{trial_seed}.npz"
    np.savez_compressed(fp, **{k: v for k, v in out.items()
                               if not isinstance(v, (dict,))})
    print(
        f"[D1/Richter seed={trial_seed}] done in {out['elapsed_s']:.1f}s | "
        f"H_exp_abs={out['H_expected_abs_mean']:.2f} Hz | "
        f"PI={out['PI_mean']:+.3f} CI=[{out['PI_ci_lo']:+.3f},{out['PI_ci_hi']:+.3f}] | "
        f"d_PI={out['PI_cohens_d']:+.3f} | d(exp-curr)={out['d_expected_vs_current']:+.3f} "
        f"d(exp-far)={out['d_expected_vs_far']:+.3f} | PASS={out['prior_pass']}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Tang D1
# ---------------------------------------------------------------------------

def run_d1_tang(trial_seed: int, n_random: int = 40, n_rotating: int = 80):
    """Tang pre-probe prior: last 100 ms of each item (per §8 resolution)."""
    print(f"[D1/Tang seed={trial_seed}] building bundle + running ...",
          flush=True)
    t0 = time.time()
    bundle = build_frozen_network(
        h_kind="ht", seed=42, r=1.0, g_total=1.0, with_cue=False,
        with_v1_to_h="continuous", with_preprobe_h_mon=True,
    )
    cfg = TangConfig(
        n_random=n_random, n_rotating=n_rotating, item_ms=250.0,
        presettle_ms=200.0, seed=trial_seed, preprobe_window_ms=100.0,
    )
    res = run_tang_rotating(bundle=bundle, cfg=cfg, seed=trial_seed)

    raw = res.raw
    hp = np.asarray(raw["h_preprobe_rate_hz"])  # (n_items, 12)
    theta_per_item = np.asarray(raw["theta_per_item"])
    deviant_mask = np.asarray(raw["deviant_mask"])
    is_random = np.asarray(raw["is_random"])
    rotation_dir = np.asarray(raw["rotation_dir"])  # +1 / -1

    # Tang orientations are at 30° spacing → channels are {0, 2, 4, 6, 8, 10}.
    # Expected next = current + rotation_dir * 2 (mod 12) for rotating items.
    tang_orient_rad = np.deg2rad(np.asarray(TANG_ORIENTATIONS_DEG))
    n_items = hp.shape[0]
    current_ch = np.array([theta_rad_to_channel(t) for t in theta_per_item])
    # Predicted next-item channel under the rotation hypothesis:
    # rotation_dir * 1 step of 30° = 2 channels (because each channel = 15°).
    exp_next_ch = np.zeros(n_items, dtype=np.int64)
    for k in range(n_items):
        step = 2 * int(rotation_dir[k]) if not is_random[k] else 0
        exp_next_ch[k] = (current_ch[k] + step) % N_CHANNELS
    far_ch_arr = np.array([far_channel(c) for c in current_ch])

    # Restrict PI computation to rotating (non-random) items where "next" is
    # well-defined, and not to the last item of a block (no "next"). Simplest
    # proxy: all non-random items; we lose 1 per block but that's negligible.
    valid = (~is_random).astype(bool)
    PI_trial = np.full(n_items, np.nan, dtype=np.float64)
    PI_trial[valid] = (
        hp[valid, exp_next_ch[valid]] - hp[valid, current_ch[valid]]
    )
    PI_valid = PI_trial[~np.isnan(PI_trial)]

    H_exp_abs = hp[valid, exp_next_ch[valid]]
    H_ref = hp[valid, current_ch[valid]]
    H_far = hp[valid, far_ch_arr[valid]]

    out = {
        "assay": "tang",
        "trial_seed": trial_seed,
        "checkpoint_seed": 42,
        "n_items": n_items,
        "preprobe_window_ms": 100.0,
        "h_preprobe_rate_hz": hp,
        "current_ch": current_ch,
        "exp_next_ch": exp_next_ch,
        "far_ch": far_ch_arr,
        "theta_per_item": theta_per_item,
        "deviant_mask": deviant_mask,
        "is_random": is_random,
        "rotation_dir": rotation_dir,
        "PI_trial": PI_valid,
        "PI_mean": float(np.mean(PI_valid)) if PI_valid.size else float("nan"),
        "PI_sd": float(np.std(PI_valid, ddof=1)) if PI_valid.size > 1 else 0.0,
        "PI_cohens_d": _cohens_d(PI_valid),
        "PI_ci_lo": _bootstrap_ci(PI_valid)[0],
        "PI_ci_hi": _bootstrap_ci(PI_valid)[1],
        "H_expected_abs_mean": float(np.mean(H_exp_abs)) if H_exp_abs.size else 0.0,
        "H_ref_mean": float(np.mean(H_ref)) if H_ref.size else 0.0,
        "H_far_mean": float(np.mean(H_far)) if H_far.size else 0.0,
        "max_H_any_channel": float(np.max(hp)),
        "d_expected_vs_current": _cohens_d_paired(H_exp_abs, H_ref),
        "d_expected_vs_far": _cohens_d_paired(H_exp_abs, H_far),
        "elapsed_s": time.time() - t0,
        "git_sha": _git_sha(),
    }
    out["check_abs"] = bool(out["H_expected_abs_mean"] >= 5.0)
    out["check_PI_mag"] = bool(
        (out["PI_mean"] >= 2.0) and (out["PI_ci_lo"] > 0.0)
    )
    out["check_d_PI"] = bool(out["PI_cohens_d"] >= 0.5)
    out["check_specificity"] = bool(
        (out["d_expected_vs_current"] >= 0.5)
        and (out["d_expected_vs_far"] >= 0.5)
    )
    out["prior_pass"] = bool(
        out["check_abs"] and out["check_PI_mag"]
        and out["check_d_PI"] and out["check_specificity"]
    )
    fp = f"{OUT_DIR}/D1_tang_seed{trial_seed}.npz"
    np.savez_compressed(fp, **{k: v for k, v in out.items()
                               if not isinstance(v, dict)})
    print(
        f"[D1/Tang seed={trial_seed}] done in {out['elapsed_s']:.1f}s | "
        f"H_exp_abs={out['H_expected_abs_mean']:.2f} Hz | "
        f"PI={out['PI_mean']:+.3f} CI=[{out['PI_ci_lo']:+.3f},{out['PI_ci_hi']:+.3f}] | "
        f"d_PI={out['PI_cohens_d']:+.3f} | PASS={out['prior_pass']}",
        flush=True,
    )
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--assays", type=str, default="kok,richter,tang")
    ap.add_argument("--n-stim-kok", type=int, default=60)
    ap.add_argument("--reps-exp-richter", type=int, default=10)
    ap.add_argument("--reps-unexp-richter", type=int, default=3)
    ap.add_argument("--n-random-tang", type=int, default=40)
    ap.add_argument("--n-rotating-tang", type=int, default=80)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    assays = [a.strip() for a in args.assays.split(",")]

    print(f"[D1] git={_git_sha()[:8]}  seeds={seeds}  assays={assays}")
    t_all = time.time()
    summary = {}
    for a in assays:
        for s in seeds:
            if a == "kok":
                summary.setdefault("kok", []).append(
                    run_d1_kok(s, n_stim=args.n_stim_kok)
                )
            elif a == "richter":
                summary.setdefault("richter", []).append(
                    run_d1_richter(
                        s, reps_exp=args.reps_exp_richter,
                        reps_unexp=args.reps_unexp_richter,
                    )
                )
            elif a == "tang":
                summary.setdefault("tang", []).append(
                    run_d1_tang(
                        s, n_random=args.n_random_tang,
                        n_rotating=args.n_rotating_tang,
                    )
                )
    print(f"[D1] ALL DONE in {time.time() - t_all:.1f}s")
    print("[D1] per-assay PASS fractions:")
    for a, lst in summary.items():
        n_pass = sum(int(x["prior_pass"]) for x in lst)
        print(f"  {a:8s}: {n_pass}/{len(lst)} seeds PASS "
              f"(PI_mean per seed: {[f'{x[chr(80)+chr(73)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:+.2f}' for x in lst]})")
