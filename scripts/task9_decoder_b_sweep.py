"""Task #9: Decoder B (5-fold CV nearest-centroid) sweep on Pass A/B and HMM march_jump.

Pure CPU on stored per-trial NPZ arrays. No model loads, no GPU.

Decoder B: src.analysis.decoding.cross_validated_decoding (5-fold, nearest-centroid,
linear in trial-pattern space). The library function returns only the mean accuracy
across folds at ±0 tolerance, so I re-use its KFold split logic in a wrapper that
also reports per-fold variance and ±1/±2 circular tolerance. Core function is NOT
modified — wrapper duplicates only the fold-split convention (perm with seed=42).

Decoder A reference: the per-trial argmax already stored in the NPZ (`dec_A`/`dec_B`
for clean-march Pass A/B, `dec` for HMM probe). Computed on the SAME filtered
sample as Decoder B for apples-to-apples comparison.

Output: stdout-only table + saves a JSON summary at results/task9_decoder_b_sweep_r1_2.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.decoding import nearest_centroid_decode

ROOT = Path("/mnt/c/Users/User/codingproj/freshstart")
NPZ_CM = Path("/tmp/debug_cleanmarch_per_trial.npz")
NPZ_HMM = Path("/tmp/debug_hmm_per_trial.npz")
OUT_JSON = ROOT / "results" / "task9_decoder_b_sweep_r1_2.json"

N_CH = 36
N_FOLDS = 5
SEED = 42


def circular_dist(pred: np.ndarray, true: np.ndarray, n_ch: int = N_CH) -> np.ndarray:
    """Wrap-around channel distance on a ring of n_ch slots."""
    d = np.abs(pred - true)
    return np.minimum(d, n_ch - d)


def acc_with_tolerance(pred: np.ndarray, true: np.ndarray, tol: int) -> float:
    """Fraction of trials where circular |pred - true| <= tol channels."""
    if len(true) == 0:
        return float("nan")
    d = circular_dist(pred, true)
    return float((d <= tol).mean())


def cv_decoder_b_perfold(
    patterns: np.ndarray,
    labels: np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    """Replicates `cross_validated_decoding` fold split (seed-matched perm) but
    keeps per-fold predictions so we can report ±0/±1/±2 tolerance + variance.

    Returns dict with keys:
      n_total, n_per_fold, fold_accs_t0/t1/t2 (list of floats),
      mean_t0/t1/t2, std_t0/t1/t2, all_preds, all_trues (concat across folds in
      test-fold order — caller can re-index if needed).
    """
    # Reproduce src.analysis.decoding.cross_validated_decoding's fold split exactly.
    gen = torch.Generator()
    gen.manual_seed(seed)
    n = patterns.shape[0]
    perm = torch.randperm(n, generator=gen).numpy()
    fold_size = n // n_folds

    pat_t = torch.as_tensor(patterns, dtype=torch.float32)
    lab_t = torch.as_tensor(labels, dtype=torch.long)

    fold_t0, fold_t1, fold_t2 = [], [], []
    all_preds, all_trues = [], []

    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate(
            [perm[:fold * fold_size], perm[(fold + 1) * fold_size:]]
        )

        train_X = pat_t[train_idx]
        train_y = lab_t[train_idx]
        test_X = pat_t[test_idx]
        test_y = lab_t[test_idx]

        # Build centroids
        classes = train_y.unique()
        centroids = {}
        for c in classes:
            m = train_y == c.item()
            if m.sum() > 0:
                centroids[c.item()] = train_X[m].mean(dim=0)

        # Predict
        preds = []
        for i in range(len(test_X)):
            dists = {c: ((test_X[i] - cent) ** 2).sum().item() for c, cent in centroids.items()}
            preds.append(min(dists, key=dists.get))
        preds_np = np.asarray(preds, dtype=np.int64)
        trues_np = test_y.numpy().astype(np.int64)

        fold_t0.append(acc_with_tolerance(preds_np, trues_np, 0))
        fold_t1.append(acc_with_tolerance(preds_np, trues_np, 1))
        fold_t2.append(acc_with_tolerance(preds_np, trues_np, 2))

        all_preds.append(preds_np)
        all_trues.append(trues_np)

    return {
        "n_total": int(n),
        "n_per_fold": int(fold_size),
        "n_folds": int(n_folds),
        "trials_used": int(fold_size * n_folds),  # may be < n if n%n_folds != 0
        "fold_accs_t0": [float(a) for a in fold_t0],
        "fold_accs_t1": [float(a) for a in fold_t1],
        "fold_accs_t2": [float(a) for a in fold_t2],
        "mean_t0": float(np.mean(fold_t0)),
        "std_t0": float(np.std(fold_t0, ddof=1)) if n_folds > 1 else 0.0,
        "mean_t1": float(np.mean(fold_t1)),
        "std_t1": float(np.std(fold_t1, ddof=1)) if n_folds > 1 else 0.0,
        "mean_t2": float(np.mean(fold_t2)),
        "std_t2": float(np.std(fold_t2, ddof=1)) if n_folds > 1 else 0.0,
        "n_unique_classes_in_full_sample": int(len(np.unique(labels))),
    }


def decoder_a_acc(pred_chs: np.ndarray, true_chs: np.ndarray) -> dict:
    """Decoder A accuracy at ±0/±1/±2 circular tolerance over the full sample."""
    return {
        "n": int(len(true_chs)),
        "acc_t0": acc_with_tolerance(pred_chs, true_chs, 0),
        "acc_t1": acc_with_tolerance(pred_chs, true_chs, 1),
        "acc_t2": acc_with_tolerance(pred_chs, true_chs, 2),
    }


def run_clean_march_variant(cm, keep_mask: np.ndarray, label: str) -> dict:
    """Run Decoder B + Decoder A sweep on the given clean-march mask. Returns
    {pass_A_b, pass_B_b, pass_A_a, pass_B_a, n} with the standard fields.
    """
    n_keep = int(keep_mask.sum())
    print(f"\n--- Clean-march variant: {label}  (n={n_keep}) ---")
    r_probe_A = cm["r_probe_A"][keep_mask]
    r_probe_B = cm["r_probe_B"][keep_mask]
    target_true_ch = cm["target_true_ch"][keep_mask]
    unexp_ch = cm["unexp_ch"][keep_mask]
    dec_A_arr = cm["dec_A"][keep_mask]
    dec_B_arr = cm["dec_B"][keep_mask]
    print(f"  Pass A: r={r_probe_A.shape}, label classes (target_true_ch) = {len(np.unique(target_true_ch))}")
    print(f"  Pass B: r={r_probe_B.shape}, label classes (unexp_ch)       = {len(np.unique(unexp_ch))}")
    db_passA = cv_decoder_b_perfold(r_probe_A, target_true_ch)
    db_passB = cv_decoder_b_perfold(r_probe_B, unexp_ch)
    da_passA = decoder_a_acc(dec_A_arr, target_true_ch)
    da_passB = decoder_a_acc(dec_B_arr, unexp_ch)
    return dict(
        n=n_keep, label=label,
        db_passA=db_passA, db_passB=db_passB,
        da_passA=da_passA, da_passB=da_passB,
    )


def print_variant_table(label: str, variant: dict, hmm_block: dict | None = None) -> None:
    print()
    print("=" * 86)
    print(f"VARIANT: {label}")
    print("=" * 86)
    print(f"{'condition':<26}{'decoder':<10}{'n':>6}{'  ±0':>10}{'  ±1':>10}{'  ±2':>10}")
    print("-" * 86)
    rows = [
        ("Clean-march Pass A", "B (CV)", variant["db_passA"]["n_total"], variant["db_passA"]["mean_t0"], variant["db_passA"]["mean_t1"], variant["db_passA"]["mean_t2"]),
        ("Clean-march Pass A", "A (lin)", variant["da_passA"]["n"], variant["da_passA"]["acc_t0"], variant["da_passA"]["acc_t1"], variant["da_passA"]["acc_t2"]),
        ("Clean-march Pass B", "B (CV)", variant["db_passB"]["n_total"], variant["db_passB"]["mean_t0"], variant["db_passB"]["mean_t1"], variant["db_passB"]["mean_t2"]),
        ("Clean-march Pass B", "A (lin)", variant["da_passB"]["n"], variant["da_passB"]["acc_t0"], variant["da_passB"]["acc_t1"], variant["da_passB"]["acc_t2"]),
    ]
    if hmm_block is not None:
        rows.extend([
            ("HMM march_jump", "B (CV)", hmm_block["db"]["n_total"], hmm_block["db"]["mean_t0"], hmm_block["db"]["mean_t1"], hmm_block["db"]["mean_t2"]),
            ("HMM march_jump", "A (lin)", hmm_block["da"]["n"], hmm_block["da"]["acc_t0"], hmm_block["da"]["acc_t1"], hmm_block["da"]["acc_t2"]),
        ])
    for cond, dec, n, t0, t1, t2 in rows:
        print(f"{cond:<26}{dec:<10}{n:>6d}{t0:>10.4f}{t1:>10.4f}{t2:>10.4f}")
    print("-" * 86)
    delta_b_t0 = variant["db_passA"]["mean_t0"] - variant["db_passB"]["mean_t0"]
    delta_b_t1 = variant["db_passA"]["mean_t1"] - variant["db_passB"]["mean_t1"]
    delta_b_t2 = variant["db_passA"]["mean_t2"] - variant["db_passB"]["mean_t2"]
    delta_a_t0 = variant["da_passA"]["acc_t0"] - variant["da_passB"]["acc_t0"]
    delta_a_t1 = variant["da_passA"]["acc_t1"] - variant["da_passB"]["acc_t1"]
    delta_a_t2 = variant["da_passA"]["acc_t2"] - variant["da_passB"]["acc_t2"]
    print(f"{'Δ(Pass A − Pass B)':<26}{'B (CV)':<10}{'-':>6}{delta_b_t0:>+10.4f}{delta_b_t1:>+10.4f}{delta_b_t2:>+10.4f}")
    print(f"{'Δ(Pass A − Pass B)':<26}{'A (lin)':<10}{'-':>6}{delta_a_t0:>+10.4f}{delta_a_t1:>+10.4f}{delta_a_t2:>+10.4f}")
    print("=" * 86)


def main():
    # ---- Clean-march: BOTH filter variants ----
    cm = np.load(NPZ_CM)
    is_clean = cm["is_clean_march"]
    is_amb = cm["is_amb"]
    keep_535 = is_clean & (~is_amb)
    keep_774 = is_clean.copy()
    print(f"Clean-march variant (i)  is_clean_march only           : n={int(keep_774.sum())}  (validator-compatible, brief asks for 774)")
    print(f"Clean-march variant (ii) is_clean_march & ~is_amb      : n={int(keep_535.sum())}  (downstream-pipeline filter)")
    var_535 = run_clean_march_variant(cm, keep_535, label="(ii) is_clean_march & ~is_amb")
    var_774 = run_clean_march_variant(cm, keep_774, label="(i) is_clean_march only [validator-compatible]")

    # ---- HMM march_jump (unchanged across clean-march variants) ----
    hmm = np.load(NPZ_HMM)
    is_jump = hmm["is_march_jump"]
    is_amb_hmm = hmm["is_amb_probe"]
    keep_hmm = is_jump & (~is_amb_hmm)
    n_keep_hmm = int(keep_hmm.sum())
    print(f"\nHMM march_jump (is_march_jump & ~is_amb_probe): n={n_keep_hmm}")
    r_probe_hmm = hmm["r_probe"][keep_hmm]
    probe_true_hmm = hmm["probe_true_ch"][keep_hmm]
    dec_hmm = hmm["dec"][keep_hmm]
    print(f"  HMM march_jump: r shape={r_probe_hmm.shape}, label classes (probe_true_ch) = {len(np.unique(probe_true_hmm))}")

    db_hmmJump = cv_decoder_b_perfold(r_probe_hmm, probe_true_hmm)
    da_hmmJump = decoder_a_acc(dec_hmm, probe_true_hmm)
    hmm_block = {"db": db_hmmJump, "da": da_hmmJump}

    # ---- Print both variant tables (HMM block appears once, with the n=535 table) ----
    print_variant_table("(ii) is_clean_march & ~is_amb  [n=535, downstream pipeline]", var_535, hmm_block=hmm_block)
    print_variant_table("(i)  is_clean_march only       [n=774, validator-compatible]", var_774, hmm_block=None)

    # ---- Decoder B per-fold breakdown for both variants ----
    for label, var in [("(ii) n=535", var_535), ("(i) n=774", var_774)]:
        print(f"\nDECODER B per-fold breakdown — {label}:")
        for name, db in [("Clean-march Pass A", var["db_passA"]), ("Clean-march Pass B", var["db_passB"])]:
            print(f"\n  {name}  (n={db['n_total']}, n/fold={db['n_per_fold']}, classes_in_sample={db['n_unique_classes_in_full_sample']})")
            print(f"    ±0 tol  per-fold: {[f'{a:.4f}' for a in db['fold_accs_t0']]}  mean={db['mean_t0']:.4f}  std={db['std_t0']:.4f}")
            print(f"    ±1 tol  per-fold: {[f'{a:.4f}' for a in db['fold_accs_t1']]}  mean={db['mean_t1']:.4f}  std={db['std_t1']:.4f}")
            print(f"    ±2 tol  per-fold: {[f'{a:.4f}' for a in db['fold_accs_t2']]}  mean={db['mean_t2']:.4f}  std={db['std_t2']:.4f}")
    print(f"\nDECODER B per-fold breakdown — HMM march_jump (filter held constant):")
    print(f"\n  HMM march_jump  (n={db_hmmJump['n_total']}, n/fold={db_hmmJump['n_per_fold']}, classes_in_sample={db_hmmJump['n_unique_classes_in_full_sample']})")
    print(f"    ±0 tol  per-fold: {[f'{a:.4f}' for a in db_hmmJump['fold_accs_t0']]}  mean={db_hmmJump['mean_t0']:.4f}  std={db_hmmJump['std_t0']:.4f}")
    print(f"    ±1 tol  per-fold: {[f'{a:.4f}' for a in db_hmmJump['fold_accs_t1']]}  mean={db_hmmJump['mean_t1']:.4f}  std={db_hmmJump['std_t1']:.4f}")
    print(f"    ±2 tol  per-fold: {[f'{a:.4f}' for a in db_hmmJump['fold_accs_t2']]}  mean={db_hmmJump['mean_t2']:.4f}  std={db_hmmJump['std_t2']:.4f}")

    # Re-bind "current" alias for old code paths writing JSON
    db_passA = var_535["db_passA"]; db_passB = var_535["db_passB"]
    da_passA = var_535["da_passA"]; da_passB = var_535["da_passB"]
    delta_b_t0 = db_passA["mean_t0"] - db_passB["mean_t0"]
    delta_b_t1 = db_passA["mean_t1"] - db_passB["mean_t1"]
    delta_b_t2 = db_passA["mean_t2"] - db_passB["mean_t2"]
    delta_a_t0 = da_passA["acc_t0"] - da_passB["acc_t0"]
    delta_a_t1 = da_passA["acc_t1"] - da_passB["acc_t1"]
    delta_a_t2 = da_passA["acc_t2"] - da_passB["acc_t2"]

    # ---- Save JSON (both variants side-by-side) ----
    def variant_to_dict(var: dict) -> dict:
        delta_b = {
            "t0": var["db_passA"]["mean_t0"] - var["db_passB"]["mean_t0"],
            "t1": var["db_passA"]["mean_t1"] - var["db_passB"]["mean_t1"],
            "t2": var["db_passA"]["mean_t2"] - var["db_passB"]["mean_t2"],
        }
        delta_a = {
            "t0": var["da_passA"]["acc_t0"] - var["da_passB"]["acc_t0"],
            "t1": var["da_passA"]["acc_t1"] - var["da_passB"]["acc_t1"],
            "t2": var["da_passA"]["acc_t2"] - var["da_passB"]["acc_t2"],
        }
        return {
            "n": var["n"], "label": var["label"],
            "decoder_b_passA": var["db_passA"], "decoder_b_passB": var["db_passB"],
            "decoder_a_passA": var["da_passA"], "decoder_a_passB": var["da_passB"],
            "delta_passA_minus_passB_decoder_b": delta_b,
            "delta_passA_minus_passB_decoder_a": delta_a,
        }

    out = {
        "label": "Task #9 Decoder B 5-fold CV vs Decoder A reference (R1+R2 seed 42) — BOTH filter variants",
        "checkpoint": "/home/vishnu/neuroips/rescue_1_2/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt",
        "data_sources": {
            "clean_march": str(NPZ_CM),
            "hmm": str(NPZ_HMM),
        },
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_channels": N_CH,
        "tolerance_step_deg": 5.0,
        "filters": {
            "variant_535_downstream_pipeline": "is_clean_march & ~is_amb",
            "variant_774_validator_compatible": "is_clean_march only (no amb filter)",
            "hmm_march_jump": "is_march_jump & ~is_amb_probe",
        },
        "variant_535": variant_to_dict(var_535),
        "variant_774": variant_to_dict(var_774),
        "hmm_march_jump_decoder_b": db_hmmJump,
        "hmm_march_jump_decoder_a": da_hmmJump,
        "caveats": [
            f"variant_535: n={int(keep_535.sum())} after `is_clean_march & ~is_amb` — matches downstream matched_probe_3pass pipeline.",
            f"variant_774: n={int(keep_774.sum())} after `is_clean_march` only — matches validator's reproduction sample.",
            "5-fold CV on n=535 leaves ~107 per fold, on n=774 leaves ~154 per fold; both below v1.0 §1-E 1500 floor.",
            "Decoder B nearest-centroid uses sample-mean centroid per class on each train fold — no class-balance reweighting.",
            "n%n_folds may drop the last (n mod n_folds) trials from CV evaluation; reported in trials_used.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
