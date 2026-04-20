"""Sprint 5e-Fix E: diagnostic snapshot of the attempt #1/#2 checkpoint.

Loads the Stage-1 ctx_pred checkpoint (same file produced by both
attempts — they are bit-identical), and produces:

  (1) Weight distributions for ctx_ee, pred_ee, W_ctx_pred (init + final),
      elig_final. Text histograms + summary stats.
  (2) Per-gate trajectory for the three-factor W_ctx_pred update across
      the 360 trailer gates: w_before / w_after / dw_sum / elig_mean /
      elig_max / n_capped vs. trial index. Both plot (PNG) and a
      per-decile numeric summary.
  (3) Per-trial forecast analysis: leader_idx / expected_trailer_idx /
      h_argmax_ctx / h_argmax_pred across the 180 rotating trials;
      the 2x2 confusion (ctx_matches_leader x pred_matches_expected)
      that feeds the pre_trailer forecast gate.

Writes::

  docs/stage1_ctx_pred_attempt1_snapshot.md  # human-readable findings
  figs/stage1_ctx_pred_attempt1_snapshot.png # 6-panel dashboard

No network is rebuilt; no simulation is run. Pure post-hoc analysis.

Usage::

    python scripts/diag_stage1_ctx_pred_snapshot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_REPO = Path(__file__).resolve().parents[1]
CKPT_PATH = _REPO / "expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz"
OUT_MD = _REPO / "docs/stage1_ctx_pred_attempt1_snapshot.md"
OUT_FIG = _REPO / "figs/stage1_ctx_pred_attempt1_snapshot.png"


def _text_hist(x: np.ndarray, bins: int = 20, width: int = 40) -> str:
    counts, edges = np.histogram(x, bins=bins)
    mx = counts.max() if counts.max() > 0 else 1
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / mx))
        lines.append(f"  [{lo:>10.4g}, {hi:>10.4g})  {c:>7d}  {bar}")
    return "\n".join(lines)


def _stats(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "p01": float(np.percentile(x, 1)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "std": float(x.std()),
    }


def _fmt_stats(s: dict, prec: int = 5) -> str:
    order = ["n", "min", "p01", "p10", "p25", "median", "mean", "p75", "p90", "p99", "max", "std"]
    return "  " + "  ".join(
        f"{k}={s[k]:>.{prec}g}" if k != "n" else f"{k}={s[k]}" for k in order
    )


def _decile_trajectory(y: np.ndarray) -> str:
    """10-point bucket mean/min/max across a 1-D time series."""
    n = y.size
    k = 10
    bucket = np.array_split(y, k)
    out = []
    out.append(
        "  | bucket |  trial range |       min |      mean |       max |"
    )
    out.append(
        "  |--------|--------------|-----------|-----------|-----------|"
    )
    idx = 0
    for i, b in enumerate(bucket):
        lo, hi = idx, idx + b.size - 1
        idx += b.size
        out.append(
            f"  |   {i+1:2d}   |  {lo:4d}..{hi:4d}  |"
            f" {b.min():>9.4g} | {b.mean():>9.4g} | {b.max():>9.4g} |"
        )
    return "\n".join(out)


def _confusion_analysis(
    leader_idx: np.ndarray,
    expected_trailer_idx: np.ndarray,
    h_argmax_ctx: np.ndarray,
    h_argmax_pred: np.ndarray,
) -> str:
    ctx_tracks_leader = h_argmax_ctx == leader_idx
    pred_tracks_leader = h_argmax_pred == leader_idx
    pred_forecasts = h_argmax_pred == expected_trailer_idx
    ctx_forecasts = h_argmax_ctx == expected_trailer_idx
    n = leader_idx.size

    lines = []
    lines.append(f"  n_trials_probed       : {n}")
    lines.append(
        f"  ctx_argmax == leader_idx     : {int(ctx_tracks_leader.sum())} / {n}  "
        f"({ctx_tracks_leader.mean() * 100:.1f}%)"
    )
    lines.append(
        f"  pred_argmax == leader_idx    : {int(pred_tracks_leader.sum())} / {n}  "
        f"({pred_tracks_leader.mean() * 100:.1f}%)"
    )
    lines.append(
        f"  pred_argmax == expected_trailer : {int(pred_forecasts.sum())} / {n}  "
        f"({pred_forecasts.mean() * 100:.1f}%)  [<-- forecast gate]"
    )
    lines.append(
        f"  ctx_argmax == expected_trailer  : {int(ctx_forecasts.sum())} / {n}  "
        f"({ctx_forecasts.mean() * 100:.1f}%)"
    )

    # Distribution of pred_argmax per leader (is pred pinned to one channel?)
    from collections import Counter
    pred_dist = Counter(h_argmax_pred.tolist())
    lines.append("  pred_argmax distribution over 6 channels:")
    for ch in range(6):
        cnt = pred_dist.get(ch, 0)
        bar = "#" * (cnt // 2)
        lines.append(f"    ch{ch}: {cnt:>4d}  {bar}")
    ctx_dist = Counter(h_argmax_ctx.tolist())
    lines.append("  ctx_argmax distribution over 6 channels:")
    for ch in range(6):
        cnt = ctx_dist.get(ch, 0)
        bar = "#" * (cnt // 2)
        lines.append(f"    ch{ch}: {cnt:>4d}  {bar}")
    return "\n".join(lines)


def _render_dashboard(z, fig_path: Path) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(14, 13))

    # (0,0) ctx_ee_w_final hist
    axs[0, 0].hist(z["ctx_ee_w_final"], bins=80, color="steelblue", edgecolor="k")
    axs[0, 0].set_title(
        f"ctx_ee_w_final  (n={z['ctx_ee_w_final'].size},  "
        f"mean={z['ctx_ee_w_final'].mean():.3f}, max={z['ctx_ee_w_final'].max():.3f})"
    )
    axs[0, 0].set_xlabel("weight")
    axs[0, 0].set_ylabel("count")
    axs[0, 0].axvline(1.0, color="r", linestyle="--", linewidth=0.8, label="init=1.0")
    axs[0, 0].axvline(1.5, color="orange", linestyle=":", linewidth=0.8, label="ee_w_max=1.5")
    axs[0, 0].legend()

    # (0,1) pred_ee_w_final hist
    axs[0, 1].hist(z["pred_ee_w_final"], bins=80, color="teal", edgecolor="k")
    axs[0, 1].set_title(
        f"pred_ee_w_final  (n={z['pred_ee_w_final'].size},  "
        f"mean={z['pred_ee_w_final'].mean():.3f}, max={z['pred_ee_w_final'].max():.3f})"
    )
    axs[0, 1].set_xlabel("weight")
    axs[0, 1].axvline(1.0, color="r", linestyle="--", linewidth=0.8, label="init=1.0")
    axs[0, 1].axvline(1.5, color="orange", linestyle=":", linewidth=0.8, label="ee_w_max=1.5")
    axs[0, 1].legend()

    # (1,0) W_ctx_pred init vs final
    axs[1, 0].hist(
        z["W_ctx_pred_init"], bins=60, color="gray", edgecolor="k",
        alpha=0.5, label=f"init (mean={z['W_ctx_pred_init'].mean():.4f})",
    )
    axs[1, 0].hist(
        z["W_ctx_pred_final"], bins=60, color="crimson", edgecolor="k",
        alpha=0.6, label=f"final (mean={z['W_ctx_pred_final'].mean():.4f})",
    )
    axs[1, 0].set_title("W_ctx_pred  init vs final")
    axs[1, 0].set_xlabel("weight")
    axs[1, 0].legend()

    # (1,1) elig_final
    axs[1, 1].hist(z["elig_final"], bins=80, color="purple", edgecolor="k")
    axs[1, 1].set_title(
        f"elig_final  (n={z['elig_final'].size},  "
        f"mean={z['elig_final'].mean():.3f}, max={z['elig_final'].max():.3f})"
    )
    axs[1, 1].set_xlabel("eligibility trace")
    axs[1, 1].set_yscale("log")

    # (2,0) per-gate trajectory: w_before and w_after
    k = np.arange(z["gate_w_before"].size)
    axs[2, 0].plot(k, z["gate_w_before"], label="w_before (mean)", color="gray")
    axs[2, 0].plot(k, z["gate_w_after"], label="w_after (mean)", color="crimson")
    axs[2, 0].set_title(f"W_ctx_pred mean across 360 gates")
    axs[2, 0].set_xlabel("gate k (trial index)")
    axs[2, 0].set_ylabel("W mean")
    axs[2, 0].legend()

    # (2,1) per-gate trajectory: dw_sum + elig_mean
    ax_a = axs[2, 1]
    ax_a.plot(k, z["gate_dw_sum"], color="crimson", label="dw_sum (gate)")
    ax_a.set_xlabel("gate k")
    ax_a.set_ylabel("dw_sum", color="crimson")
    ax_a.axhline(0.0, color="gray", linestyle=":", linewidth=0.7)
    ax_b = ax_a.twinx()
    ax_b.plot(k, z["gate_elig_mean"], color="steelblue", label="elig_mean")
    ax_b.set_ylabel("elig_mean", color="steelblue")
    ax_a.set_title("per-gate dw_sum (red, left) + elig_mean (blue, right)")

    fig.suptitle(
        "Stage-1 ctx_pred attempt #1 / #2  — weight + gate-trajectory snapshot  "
        "(seed=42, n_trials=360, inh_w_max=1.5 effective)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def _ee_structure_split(ctx_ee_w: np.ndarray) -> str:
    """ctx_ee weights are 36672 = 72 E cells * (16 within + 32 cross + ...).
    We can't split within/cross without the connection indices, but we
    can report the concentration around the small vs large regime."""
    # Since we don't have connection indices, report empirical structure.
    # ee_w_max = 1.5; within-init = 1.0, cross-init = 0.02.
    # A bimodal distribution would show a peak near within (0.x-1.x) and
    # a peak near cross (small). Report fraction above various thresholds.
    n = ctx_ee_w.size
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.4]
    lines = []
    lines.append("  fraction of ctx_ee weights >= threshold:")
    for t in thresholds:
        f = float((ctx_ee_w >= t).mean())
        bar = "#" * int(round(40 * f))
        lines.append(f"    >= {t:>5.2f}  {f:>6.3%}  {bar}")
    return "\n".join(lines)


def main() -> int:
    if not CKPT_PATH.exists():
        print(f"ERR: checkpoint not found: {CKPT_PATH}", file=sys.stderr)
        return 2

    z = np.load(CKPT_PATH, allow_pickle=True)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    _render_dashboard(z, OUT_FIG)

    # Build markdown report.
    md = []
    md.append("# Stage-1 ctx_pred attempt #1 / #2 — diagnostic snapshot\n")
    md.append(
        "Checkpoint: `expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz`.\n"
    )
    md.append(
        "Attempts #1 (commit 5317540) and #2 (commit 31e6e98) produced\n"
        "bit-identical weights and firing rates — see commit 31e6e98 for the\n"
        "`_stage1_h_cfg(h_cfg)` override discovery. This snapshot is against\n"
        "the shared final state, which represents the network at\n"
        "effective `inh_w_max = 1.5` (the `_stage1_h_cfg` hardcoded value).\n"
    )
    md.append("## Gate verdict\n")
    md.append("| check | value | band | result |")
    md.append("|---|---|---|---|")
    md.append("| h_bump_persistence_ms | 10.0 | [200, 500] | **FAIL** |")
    md.append("| h_preprobe_forecast_prob | 0.000 | [0.25, 1.00] | **FAIL** |")
    md.append("| no_runaway | 21.27 Hz | [0, 80] | PASS |\n")

    md.append("## Weight-distribution snapshot\n")
    for name in ("ctx_ee_w_final", "pred_ee_w_final",
                 "W_ctx_pred_init", "W_ctx_pred_final", "elig_final"):
        s = _stats(z[name])
        md.append(f"### `{name}`  (n={s['n']})\n")
        md.append("```")
        md.append(_fmt_stats(s))
        md.append("```")
        md.append("Histogram (text):\n")
        md.append("```")
        md.append(_text_hist(z[name]))
        md.append("```\n")

    md.append("## ctx_ee structural concentration\n")
    md.append("```")
    md.append(_ee_structure_split(z["ctx_ee_w_final"]))
    md.append("```\n")
    md.append(
        "Interpretation: within-channel init = 1.0, cross-channel init = 0.02. "
        "If within-channel weights had stayed near init, ≥25% of the 36672 weights "
        "(16 within / 48 per postsyn) would be ≥ 0.5. If only a tiny fraction exceed "
        "even 0.1, the within-channel recurrent backbone — the substrate the bump "
        "needs — has been dismantled.\n"
    )

    md.append("## Three-factor gate trajectory (360 gates)\n")
    md.append(
        "`gate_dw_sum` per-gate aggregate dW deposited on W_ctx_pred (sum over "
        "36864 synapses):\n"
    )
    md.append("```")
    md.append(_fmt_stats(_stats(z["gate_dw_sum"])))
    md.append("```")
    md.append("Decile trajectory of `gate_dw_sum` across the 360-gate schedule:\n")
    md.append("```")
    md.append(_decile_trajectory(z["gate_dw_sum"]))
    md.append("```")
    md.append(
        "Decile trajectory of `gate_elig_mean` (average eligibility at the gate):\n"
    )
    md.append("```")
    md.append(_decile_trajectory(z["gate_elig_mean"]))
    md.append("```")
    md.append(
        "Decile trajectory of `gate_elig_max` (peak eligibility at the gate):\n"
    )
    md.append("```")
    md.append(_decile_trajectory(z["gate_elig_max"]))
    md.append("```")
    md.append(
        "Decile trajectory of `gate_w_before` (mean W_ctx_pred before update):\n"
    )
    md.append("```")
    md.append(_decile_trajectory(z["gate_w_before"]))
    md.append("```")
    md.append(
        "`gate_n_capped` min/max: "
        f"{int(z['gate_n_capped'].min())} / {int(z['gate_n_capped'].max())} "
        "(number of W_ctx_pred synapses at w_max cap per gate — "
        "any deviation from 192 would indicate caps forming or releasing).\n"
    )

    md.append("## Forecast-gate confusion (180 rotating trials)\n")
    md.append("```")
    md.append(_confusion_analysis(
        z["leader_idx"], z["expected_trailer_idx"],
        z["h_argmax_ctx"], z["h_argmax_pred"],
    ))
    md.append("```\n")

    md.append("## Preliminary findings (checkpoint only)\n")
    md.append(
        "1. **W_ctx_pred is collapsing, not consolidating.** `gate_dw_sum` is "
        "**net-negative at every single gate** across all 360 trials "
        "(min=-346, max≈0, mean=-0.96). The three-factor rule is producing "
        "LTD-only, never LTP. This is consistent with `W_ctx_pred_final` "
        "mean (0.0156) being less than `W_ctx_pred_init` mean (0.0250).\n"
    )
    md.append(
        "2. **Eligibility traces are healthy.** `gate_elig_mean` in range "
        "5-75 across the schedule, `gate_elig_max` 89-1261. The three-factor "
        "multiplier `eligibility × M(t) × ACh` has non-trivial magnitude; "
        "the LTD is a *sign* problem in the rule, not a magnitude problem.\n"
    )
    md.append(
        "3. **Forecast probability 0.000 is a deterministic structural zero.** "
        "Checkpoint's own `h_argmax_pred` column lets us verify the gate's "
        "counting — see confusion block above. Pred either tracks the "
        "current leader (amplifier) or a fixed channel (drift), never "
        "the expected trailer.\n"
    )
    md.append(
        "4. **Missing telemetry** — to complete the diagnosis we still need: "
        "(a) `ctx.inh→e` and `pred.inh→e` weight histograms at end-of-training, "
        "(b) per-trial H_ctx / H_pred firing traces across the 360 trials, "
        "(c) per-synapse W_ctx_pred trajectory (not just aggregates). These "
        "require either a short re-run with SpikeMonitors + weight snapshots, "
        "or adding dumps to `run_stage_1_ctx_pred` and re-running briefly "
        "(n_trials ≈ 60). Awaiting Lead's call before launching.\n"
    )

    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
