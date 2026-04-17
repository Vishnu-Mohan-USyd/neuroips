"""
Task #10: Empirically characterise Decoder A (loss_fn.orientation_decoder).

Pure CPU. Read-only on the model. Produces:
  results/decA_probe_r1_2.json
  docs/figures/decA_weights_r1_2.png
  docs/figures/decA_compound_sweep_r1_2.png
  logs/decA_probe.log (via tee on caller side)

Sections:
  Part 1 — raw weight inspection (heatmap, FWHM, side-lobes, bias)
  Part 2 — single-bump amplitude sweep (sigma=3, mu=18)
  Part 3 — single-bump width sweep      (A=0.5, mu=18)
  Part 4 — compound-bump sweep (primary + secondary at offset Delta)
  Part 5 — Pass A / Pass B emulated synthetic profiles
  Part 6 — noise floor (1000 random draws)
  Part 7 — real-data cross-reference (Pass A/B from
           /tmp/debug_cleanmarch_per_trial.npz)

No mechanism interpretation — facts only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N = 36                       # ring channels
STEP_DEG = 5.0
PERIOD = 180.0
CENTER = 18                  # canonical re-centered probe channel
RNG_SEED = 42

CKPT_PATH = Path("results/simple_dual/emergent_seed42/checkpoint.pt")
OUT_JSON = Path("results/decA_probe_r1_2.json")
FIG_WEIGHTS = Path("docs/figures/decA_weights_r1_2.png")
FIG_COMPOUND = Path("docs/figures/decA_compound_sweep_r1_2.png")
PER_TRIAL_NPZ = Path("/tmp/debug_cleanmarch_per_trial.npz")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def circ_dist(a: int, b: int, n: int = N) -> int:
    """Circular distance on a ring of n channels."""
    d = abs(int(a) - int(b))
    return min(d, n - d)


def gaussian_bump(mu: float, sigma: float, A: float = 1.0, n: int = N) -> np.ndarray:
    """Wrapped Gaussian on the ring (sums two side reflections so wrap is exact)."""
    chs = np.arange(n)
    diff = (chs - mu + n / 2) % n - n / 2     # signed, in [-n/2, n/2)
    return (A * np.exp(-0.5 * (diff / sigma) ** 2)).astype(np.float64)


def compound_bump(
    ch_P: int, ch_S: int, A_P: float, A_S: float, sigma: float = 3.0
) -> np.ndarray:
    return gaussian_bump(ch_P, sigma, A_P) + gaussian_bump(ch_S, sigma, A_S)


def forward_decA(W: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """logits = W @ x + b, where x can be (N,) or (B, N)."""
    if x.ndim == 1:
        return W @ x + b
    return x @ W.T + b


def fwhm_circular(row: np.ndarray, peak_idx: int) -> float:
    """FWHM of a 1-D row (length N) around peak_idx, on a ring.
    Returns width in channels (interpolated between samples) where row drops to
    half its peak. NaN if no half-max crossing found within ±N/2.
    """
    n = len(row)
    peak = float(row[peak_idx])
    if peak <= 0:
        return float("nan")
    half = peak / 2.0

    # Walk left and right on the ring until row drops below half-max.
    def walk(direction: int) -> float:
        prev_val = peak
        prev_offset = 0.0
        for k in range(1, n):
            idx = (peak_idx + direction * k) % n
            val = float(row[idx])
            if val < half:
                # linear interpolation between (k-1, prev_val) and (k, val)
                if prev_val == val:
                    return float(k)
                t = (prev_val - half) / (prev_val - val)
                return float(k - 1 + t)
            prev_val = val
            prev_offset = float(k)
        return float("nan")

    left = walk(-1)
    right = walk(+1)
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return left + right


# -----------------------------------------------------------------------------
# Part 1 — weight inspection
# -----------------------------------------------------------------------------
def part1_weights(W: np.ndarray, b: np.ndarray) -> dict:
    """Inspect W and b. Diagonal-ness, FWHM per row, side-lobe energy, bias."""
    print("\n=== Part 1: raw weights ===")

    diag = np.array([W[i, i] for i in range(N)])
    print(f"Diagonal weights: min={diag.min():.4f}  max={diag.max():.4f}  "
          f"mean={diag.mean():.4f}  std={diag.std():.4f}")

    # Per-row argmax / FWHM / off-diagonal extrema / ±90° mean.
    per_row = []
    for i in range(N):
        row = W[i]
        argmax = int(row.argmax())
        argmin = int(row.argmin())
        fwhm = fwhm_circular(row, argmax)
        off = np.delete(row.copy(), argmax)
        opp1 = (i + N // 4) % N      # +90 deg = +9 ch
        opp2 = (i - N // 4) % N      # -90 deg
        per_row.append({
            "row": i,
            "argmax": argmax,
            "argmax_at_diag": argmax == i,
            "diag_value": float(W[i, i]),
            "row_max": float(row.max()),
            "row_min": float(row.min()),
            "fwhm_ch": float(fwhm),
            "off_diag_min": float(off.min()),
            "off_diag_max": float(off.max()),
            "value_plus_90deg": float(row[opp1]),
            "value_minus_90deg": float(row[opp2]),
        })

    pct_diag = sum(1 for r in per_row if r["argmax_at_diag"]) / N
    print(f"Rows whose argmax is at diagonal: {int(pct_diag*N)}/{N}  "
          f"({100*pct_diag:.1f}%)")

    fwhms = np.array([r["fwhm_ch"] for r in per_row])
    fwhms_finite = fwhms[np.isfinite(fwhms)]
    print(f"FWHM (ch): mean={fwhms_finite.mean():.2f}  median={np.median(fwhms_finite):.2f}  "
          f"min={fwhms_finite.min():.2f}  max={fwhms_finite.max():.2f}  "
          f"NaN={(~np.isfinite(fwhms)).sum()}")

    plus90 = np.array([r["value_plus_90deg"] for r in per_row])
    minus90 = np.array([r["value_minus_90deg"] for r in per_row])
    print(f"Mean weight at +90deg from diagonal: {plus90.mean():.4f}  "
          f"(min={plus90.min():.4f}, max={plus90.max():.4f})")
    print(f"Mean weight at -90deg from diagonal: {minus90.mean():.4f}  "
          f"(min={minus90.min():.4f}, max={minus90.max():.4f})")

    # Bias
    print(f"Bias b: min={b.min():.4f}  max={b.max():.4f}  "
          f"mean={b.mean():.4f}  std={b.std():.4f}")
    bmax_idx = int(b.argmax())
    bmin_idx = int(b.argmin())
    print(f"Bias argmax = ch{bmax_idx}  ({b[bmax_idx]:+.4f})")
    print(f"Bias argmin = ch{bmin_idx}  ({b[bmin_idx]:+.4f})")

    # Heatmap figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                             gridspec_kw={"width_ratios": [3, 1, 1]})
    im = axes[0].imshow(W, cmap="RdBu_r", aspect="equal",
                        vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    axes[0].set_xlabel("Input ch (L2/3 channel)")
    axes[0].set_ylabel("Output ch (orientation class)")
    axes[0].set_title("Decoder A weight matrix W (Linear 36x36)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(diag, marker="o", color="#1f77b4", lw=1)
    axes[1].set_xlabel("Class index")
    axes[1].set_ylabel("Diagonal weight W[i,i]")
    axes[1].set_title("Diagonal")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(N), b, color="#d62728")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlabel("Class index")
    axes[2].set_ylabel("Bias b[i]")
    axes[2].set_title("Bias")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    FIG_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_WEIGHTS, dpi=130)
    plt.close(fig)
    print(f"Saved heatmap -> {FIG_WEIGHTS}")

    return {
        "diag_stats": {
            "min": float(diag.min()),
            "max": float(diag.max()),
            "mean": float(diag.mean()),
            "std": float(diag.std()),
        },
        "fwhm_stats": {
            "mean": float(fwhms_finite.mean()),
            "median": float(np.median(fwhms_finite)),
            "min": float(fwhms_finite.min()),
            "max": float(fwhms_finite.max()),
            "n_nan": int((~np.isfinite(fwhms)).sum()),
        },
        "side_lobe_stats": {
            "mean_plus_90deg": float(plus90.mean()),
            "min_plus_90deg": float(plus90.min()),
            "max_plus_90deg": float(plus90.max()),
            "mean_minus_90deg": float(minus90.mean()),
            "min_minus_90deg": float(minus90.min()),
            "max_minus_90deg": float(minus90.max()),
        },
        "argmax_at_diag_pct": float(pct_diag),
        "bias": {
            "values": [float(v) for v in b],
            "min": float(b.min()),
            "max": float(b.max()),
            "mean": float(b.mean()),
            "argmax": bmax_idx,
            "argmin": bmin_idx,
        },
        "per_row": per_row,
    }


# -----------------------------------------------------------------------------
# Part 2 — single-bump amplitude sweep
# -----------------------------------------------------------------------------
def part2_amplitude(W: np.ndarray, b: np.ndarray) -> dict:
    print("\n=== Part 2: single-bump amplitude sweep (mu=18, sigma=3) ===")
    sigma = 3.0
    A_list = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    rows = []
    print(f"{'A':>6}  {'argmax':>7}  {'top1_logit':>11}  {'top2_logit':>11}  {'margin':>8}")
    for A in A_list:
        x = gaussian_bump(CENTER, sigma, A)
        z = forward_decA(W, b, x)
        order = np.argsort(z)[::-1]
        top1, top2 = order[0], order[1]
        margin = float(z[top1] - z[top2])
        print(f"{A:>6.2f}  {int(top1):>7d}  {float(z[top1]):>11.4f}  "
              f"{float(z[top2]):>11.4f}  {margin:>8.4f}")
        rows.append({
            "A": float(A),
            "argmax": int(top1),
            "top2_class": int(top2),
            "top1_logit": float(z[top1]),
            "top2_logit": float(z[top2]),
            "margin": margin,
            "all_logits": [float(v) for v in z],
        })
    argmaxes = [r["argmax"] for r in rows]
    unique_argmax = sorted(set(argmaxes))
    print(f"Unique argmax across A: {unique_argmax}")
    return {"rows": rows, "unique_argmax": unique_argmax}


# -----------------------------------------------------------------------------
# Part 3 — single-bump width sweep
# -----------------------------------------------------------------------------
def part3_width(W: np.ndarray, b: np.ndarray) -> dict:
    print("\n=== Part 3: single-bump width sweep (mu=18, A=0.5) ===")
    A = 0.5
    sigma_list = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    rows = []
    print(f"{'sigma':>6}  {'argmax':>7}  {'top1_logit':>11}  {'top2_logit':>11}  {'margin':>8}")
    for sigma in sigma_list:
        x = gaussian_bump(CENTER, sigma, A)
        z = forward_decA(W, b, x)
        order = np.argsort(z)[::-1]
        top1, top2 = order[0], order[1]
        margin = float(z[top1] - z[top2])
        print(f"{sigma:>6.1f}  {int(top1):>7d}  {float(z[top1]):>11.4f}  "
              f"{float(z[top2]):>11.4f}  {margin:>8.4f}")
        rows.append({
            "sigma": float(sigma),
            "argmax": int(top1),
            "top2_class": int(top2),
            "top1_logit": float(z[top1]),
            "top2_logit": float(z[top2]),
            "margin": margin,
            "all_logits": [float(v) for v in z],
        })
    argmaxes = [r["argmax"] for r in rows]
    unique_argmax = sorted(set(argmaxes))
    print(f"Unique argmax across sigma: {unique_argmax}")
    return {"rows": rows, "unique_argmax": unique_argmax}


# -----------------------------------------------------------------------------
# Part 4 — compound-bump sweep
# -----------------------------------------------------------------------------
def part4_compound(W: np.ndarray, b: np.ndarray) -> dict:
    print("\n=== Part 4: compound-bump sweep (ch_P=18) ===")
    sigma = 3.0
    ch_P = CENTER
    Delta_list = [9, 18, 27]
    A_pairs = [(0.5, 0.0), (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5)]

    by_delta = {}
    for Delta in Delta_list:
        ch_S = (ch_P + Delta) % N
        print(f"\n  Delta = {Delta} ch  ({Delta * STEP_DEG:.0f} deg) -> ch_P={ch_P}, ch_S={ch_S}")
        print(f"    {'A_P':>5}  {'A_S':>5}  {'ratio':>6}  {'argmax':>7}  "
              f"{'==ch_P':>7}  {'==ch_S':>7}  {'top1':>9}  {'top2':>9}  {'margin':>8}")
        rows = []
        flip_ratio = None
        for A_P, A_S in A_pairs:
            x = compound_bump(ch_P, ch_S, A_P, A_S, sigma=sigma)
            z = forward_decA(W, b, x)
            order = np.argsort(z)[::-1]
            top1, top2 = order[0], order[1]
            margin = float(z[top1] - z[top2])
            ratio = A_S / A_P if A_P > 0 else float("nan")
            is_P = bool(int(top1) == int(ch_P))
            is_S = bool(int(top1) == int(ch_S))
            print(f"    {A_P:>5.2f}  {A_S:>5.2f}  {ratio:>6.2f}  {int(top1):>7d}  "
                  f"{str(is_P):>7}  {str(is_S):>7}  {float(z[top1]):>9.4f}  "
                  f"{float(z[top2]):>9.4f}  {margin:>8.4f}")
            if flip_ratio is None and not is_P:
                flip_ratio = ratio
            rows.append({
                "A_P": float(A_P),
                "A_S": float(A_S),
                "ratio": float(ratio),
                "argmax": int(top1),
                "argmax_at_ch_P": is_P,
                "argmax_at_ch_S": is_S,
                "top1_logit": float(z[top1]),
                "top2_logit": float(z[top2]),
                "margin": margin,
                "all_logits": [float(v) for v in z],
            })
        by_delta[str(Delta)] = {
            "ch_P": int(ch_P),
            "ch_S": int(ch_S),
            "rows": rows,
            "flip_ratio": flip_ratio,
        }
        print(f"  -> first ratio at which argmax leaves ch_P: {flip_ratio}")

    # Figure: argmax vs ratio for each Delta
    fig, axes = plt.subplots(1, len(Delta_list), figsize=(15, 4), sharey=True)
    for ax, Delta in zip(axes, Delta_list):
        rows = by_delta[str(Delta)]["rows"]
        ratios = [r["ratio"] for r in rows]
        argmaxes = [r["argmax"] for r in rows]
        margins = [r["margin"] for r in rows]
        ch_P = by_delta[str(Delta)]["ch_P"]
        ch_S = by_delta[str(Delta)]["ch_S"]
        ax.plot(ratios, argmaxes, "o-", color="#1f77b4", label="argmax")
        ax.axhline(ch_P, color="green", ls="--", alpha=0.6, label=f"ch_P={ch_P}")
        ax.axhline(ch_S, color="red", ls="--", alpha=0.6, label=f"ch_S={ch_S}")
        ax.set_xlabel("A_S / A_P")
        ax.set_ylabel("argmax channel")
        ax.set_title(f"Delta={Delta} ch ({Delta*STEP_DEG:.0f} deg)")
        ax.set_ylim(-1, N)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(ratios, margins, "s--", color="#d62728", alpha=0.6, label="margin")
        ax2.set_ylabel("top1-top2 margin", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("Decoder A: compound-bump argmax vs amplitude ratio")
    fig.tight_layout()
    FIG_COMPOUND.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_COMPOUND, dpi=130)
    plt.close(fig)
    print(f"Saved compound-sweep figure -> {FIG_COMPOUND}")

    return by_delta


# -----------------------------------------------------------------------------
# Part 5 — Pass A / Pass B emulation
# -----------------------------------------------------------------------------
def part5_passAB(W: np.ndarray, b: np.ndarray) -> dict:
    print("\n=== Part 5: Pass A / Pass B emulation ===")
    sigma = 3.0
    # Pass A: single bump A=0.68, mu=18, sigma=3
    xA = gaussian_bump(CENTER, sigma, 0.68)
    # Pass B: primary at mu=18 A=0.44 + secondary at mu=0 (=ring-opposite, +90deg) A=0.20
    xB = gaussian_bump(CENTER, sigma, 0.44) + gaussian_bump(0, sigma, 0.20)

    out = {}
    for label, x in [("pass_A_synth", xA), ("pass_B_synth", xB)]:
        z = forward_decA(W, b, x)
        order = np.argsort(z)[::-1]
        top1, top2, top3 = order[0], order[1], order[2]
        m12 = float(z[top1] - z[top2])
        m23 = float(z[top2] - z[top3])
        print(f"  {label}:")
        print(f"    argmax = ch{int(top1)}  (logit {float(z[top1]):+.4f})")
        print(f"    top2   = ch{int(top2)}  (logit {float(z[top2]):+.4f})")
        print(f"    top3   = ch{int(top3)}  (logit {float(z[top3]):+.4f})")
        print(f"    margin top1-top2 = {m12:.4f}")
        print(f"    margin top2-top3 = {m23:.4f}")
        print(f"    full input  (3x12 rows):")
        for k in range(3):
            print("     ", " ".join(f"{x[12*k+j]:+.3f}" for j in range(12)))
        print(f"    full logits (3x12 rows):")
        for k in range(3):
            print("     ", " ".join(f"{z[12*k+j]:+.3f}" for j in range(12)))
        out[label] = {
            "input": [float(v) for v in x],
            "logits": [float(v) for v in z],
            "argmax": int(top1),
            "top1_class": int(top1),
            "top2_class": int(top2),
            "top3_class": int(top3),
            "top1_logit": float(z[top1]),
            "top2_logit": float(z[top2]),
            "top3_logit": float(z[top3]),
            "margin_top1_top2": m12,
            "margin_top2_top3": m23,
        }
    return out


# -----------------------------------------------------------------------------
# Part 6 — noise floor
# -----------------------------------------------------------------------------
def part6_noise(W: np.ndarray, b: np.ndarray, n_draws: int = 1000) -> dict:
    print("\n=== Part 6: noise floor (1000 random draws) ===")
    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(0, 0.1, size=(n_draws, N)) + 0.05
    Z = forward_decA(W, b, X)
    argmaxes = Z.argmax(axis=1)
    counts = np.bincount(argmaxes, minlength=N)
    pct = counts / float(n_draws)
    top1_logits = Z[np.arange(n_draws), argmaxes]
    print(f"argmax counts (per class):")
    for k in range(3):
        print("  ", " ".join(f"{int(counts[12*k+j]):>4d}" for j in range(12)))
    print(f"Counts: min={int(counts.min())}  max={int(counts.max())}  "
          f"mean={float(counts.mean()):.2f}  std={float(counts.std()):.2f}")
    print(f"Mean top1 logit under noise: {float(top1_logits.mean()):.4f}  "
          f"(std {float(top1_logits.std()):.4f})")
    # uniform expected count = 1000/36 = 27.78
    expected = n_draws / N
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    print(f"Chi^2 vs uniform = {chi2:.2f}  (df=35, "
          f"~p<0.05 if chi2>49.8; ~p<0.001 if chi2>66.6)")
    return {
        "n_draws": int(n_draws),
        "counts_per_class": [int(c) for c in counts],
        "pct_per_class": [float(p) for p in pct],
        "argmax_count_min": int(counts.min()),
        "argmax_count_max": int(counts.max()),
        "argmax_count_mean": float(counts.mean()),
        "argmax_count_std": float(counts.std()),
        "mean_top1_logit": float(top1_logits.mean()),
        "std_top1_logit": float(top1_logits.std()),
        "chi2_vs_uniform": chi2,
        "expected_count_per_class": float(expected),
    }


# -----------------------------------------------------------------------------
# Part 7 — real-data cross-reference
# -----------------------------------------------------------------------------
def part7_realdata(W: np.ndarray, b: np.ndarray) -> dict:
    print("\n=== Part 7: real-data cross-reference (clean-march, pi-Q75 + pred_err<=15) ===")
    if not PER_TRIAL_NPZ.exists():
        print(f"  npz not found at {PER_TRIAL_NPZ} — skipping.")
        return {"available": False, "path": str(PER_TRIAL_NPZ)}

    d = np.load(PER_TRIAL_NPZ, allow_pickle=False)
    keys = list(d.keys())
    print(f"  npz keys: {keys}")

    is_cm = d["is_clean_march"].astype(bool)
    is_amb = d["is_amb"].astype(bool)
    target_true_ch = d["target_true_ch"].astype(int)
    unexp_ch = d["unexp_ch"].astype(int)
    rA = d["r_probe_A"].astype(np.float64)
    rB = d["r_probe_B"].astype(np.float64)
    if "pi_target" in d.files:
        pi = d["pi_target"].astype(np.float64)
    elif "pi_pred" in d.files:
        pi = d["pi_pred"].astype(np.float64)
    else:
        pi = None
    if "pred_err_A" in d.files:
        pred_err = d["pred_err_A"].astype(np.float64)
    elif "pred_err_deg" in d.files:
        pred_err = d["pred_err_deg"].astype(np.float64)
    else:
        pred_err = None

    base_mask = is_cm & ~is_amb
    print(f"  is_clean_march & ~is_amb -> n={int(base_mask.sum())}")
    if pi is not None and pred_err is not None:
        # Replicate Pass A (target_true) and Pass B (unexp) high-confidence sample.
        pi_keep = pi[base_mask]
        if pi_keep.size > 0:
            q75 = float(np.quantile(pi_keep, 0.75))
        else:
            q75 = float("nan")
        # The Task #7 sample was "pi-Q75 + pred_err<=15".
        keep_pe = pred_err <= 15.0
        keep_pi = pi >= q75
        sel = base_mask & keep_pe & keep_pi
        print(f"  pi-Q75 + pred_err<=15 -> n={int(sel.sum())}  (q75={q75:.4f})")
    else:
        # If pi/pred_err not present, fall back to the base mask.
        print("  pi_pred/pred_err not present in npz — using base mask")
        sel = base_mask
        q75 = None

    idx = np.where(sel)[0]
    if idx.size == 0:
        return {"available": True, "n": 0, "note": "no rows survive selection"}

    out = {}
    for label, r, true_ch in [("pass_A", rA, target_true_ch),
                               ("pass_B", rB, unexp_ch)]:
        x = r[idx]                               # [n, 36]
        tc = true_ch[idx]                        # [n]
        z = forward_decA(W, b, x)                # [n, 36]
        order = np.argsort(z, axis=1)[:, ::-1]
        top1 = order[:, 0]
        top2 = order[:, 1]
        m12 = z[np.arange(z.shape[0]), top1] - z[np.arange(z.shape[0]), top2]
        # circular displacement of argmax from true channel
        d_signed = ((top1 - tc + N // 2) % N) - N // 2
        d_circ = np.minimum(np.abs(top1 - tc), N - np.abs(top1 - tc))
        correct0 = d_circ == 0
        correct1 = d_circ <= 1
        correct2 = d_circ <= 2
        acc0 = float(correct0.mean())
        acc1 = float(correct1.mean())
        acc2 = float(correct2.mean())

        # Buckets on |displacement|
        n = x.shape[0]
        b_at = int((d_circ == 0).sum())
        b_pm1 = int((d_circ == 1).sum())
        b_pm2 = int((d_circ == 2).sum())
        b_other = int((d_circ > 2).sum())

        # Correlation of margin with correctness
        if n > 1:
            corr = float(np.corrcoef(m12, correct0.astype(float))[0, 1])
        else:
            corr = float("nan")
        mean_margin_correct = float(m12[correct0].mean()) if correct0.any() else float("nan")
        mean_margin_wrong = float(m12[~correct0].mean()) if (~correct0).any() else float("nan")

        print(f"  {label}: n={n}")
        print(f"    Decoder A acc  +-0={acc0:.4f}   +-1={acc1:.4f}   +-2={acc2:.4f}")
        print(f"    argmax distance buckets: at_true={b_at}  +-1={b_pm1}  "
              f"+-2={b_pm2}  other={b_other}")
        print(f"    mean top1-top2 margin: correct={mean_margin_correct:.4f}  "
              f"wrong={mean_margin_wrong:.4f}")
        print(f"    corr(margin, correct@0): {corr:+.4f}")
        out[label] = {
            "n": int(n),
            "acc_pm0": acc0,
            "acc_pm1": acc1,
            "acc_pm2": acc2,
            "argmax_at_true": b_at,
            "argmax_at_pm1": b_pm1,
            "argmax_at_pm2": b_pm2,
            "argmax_other": b_other,
            "mean_margin_correct": mean_margin_correct,
            "mean_margin_wrong": mean_margin_wrong,
            "corr_margin_correct": corr,
            "mean_top1_logit": float(z[np.arange(n), top1].mean()),
            "mean_margin_overall": float(m12.mean()),
        }
    out["selection"] = {
        "n_selected": int(idx.size),
        "filter": "is_clean_march & ~is_amb & (pi >= pi_Q75) & (pred_err <= 15)",
        "pi_Q75": q75,
    }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ck["loss_heads"]["orientation_decoder"]
    W = sd["weight"].cpu().numpy().astype(np.float64)   # (36, 36)
    b = sd["bias"].cpu().numpy().astype(np.float64)     # (36,)
    print(f"W shape={W.shape}, b shape={b.shape}")
    print(f"W stats: min={W.min():+.4f}  max={W.max():+.4f}  "
          f"mean={W.mean():+.4f}  fro={np.linalg.norm(W):.4f}")

    out = {
        "label": "Task #10 — Decoder A empirical probe (R1+R2 emergent_seed42)",
        "checkpoint": str(CKPT_PATH),
        "decoder_source": "loss_heads.orientation_decoder",
        "N": int(N),
        "step_deg": float(STEP_DEG),
        "center_idx": int(CENTER),
        "rng_seed": int(RNG_SEED),
    }
    out["part1_weights"] = part1_weights(W, b)
    out["part2_amplitude"] = part2_amplitude(W, b)
    out["part3_width"] = part3_width(W, b)
    out["part4_compound"] = part4_compound(W, b)
    out["part5_passAB"] = part5_passAB(W, b)
    out["part6_noise"] = part6_noise(W, b)
    out["part7_realdata"] = part7_realdata(W, b)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
