"""Sprint 5e-Fix E attempt #4: diagnostic snapshot of the attempt-#4 checkpoint.

Attempt #4 (commit d8ca59f driver + Fix A/B(i)/C commits 0f30cd2 / b5d8fd7 /
b5ba400, log `logs/train_stage1_ctx_pred_full_seed42_attempt4.log`) FAILED
the Stage-1 gate: `h_bump_persistence_ms=10`, `h_preprobe_forecast_prob=
0.039`, `no_runaway=PASS 21.27 Hz`.

A critical finding emerged during analysis: **Fix B(i) was silently
reverted by `expectation_snn/brian2_model/train.py:_stage1_h_cfg:553`**.
The dataclass-default change `HRingConfig.inh_rho_hz = 20.0` was
overwritten in-place to `10.0` by the Stage-1 helper before the Brian2
Synapses were built. The startup banner printed 20.0 (read at driver
entry); the end-of-run evidence-log JSON printed 10.0 on the *same*
`H_CFG` object. This matches attempt #2's failure pattern (inh_w_max
clobbered by the same helper at line 556).

Fix A (`tau_coinc 20 -> 500 ms`) and Fix C (`w_target 0.05 -> 0.0075`)
DID land correctly — they live on `HContextPredictionConfig`, which the
helper does not touch. So the attempt #4 run was effectively "Fix A +
Fix C with Fix B(i) reverted to baseline 10 Hz".

This script produces:

  (1) Weight distributions for ctx_ee, pred_ee, W_ctx_pred (init/final),
      elig_final. Text histograms + summary stats.
  (2) Per-gate trajectory: w_before / w_after / dw_sum / elig_mean /
      elig_max / n_capped vs. trial index, decile tables + PNG.
  (3) Per-trial forecast confusion: leader/expected_trailer/
      h_argmax_ctx/h_argmax_pred across the 180 rotating trials.

Writes::

  docs/stage1_ctx_pred_attempt4_snapshot.md
  figs/stage1_ctx_pred_attempt4_snapshot.png

No network is rebuilt; no simulation is run. Pure post-hoc analysis of
`expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz`.

Usage::

    python scripts/diag_stage1_ctx_pred_attempt4_snapshot.py
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
OUT_MD = _REPO / "docs/stage1_ctx_pred_attempt4_snapshot.md"
OUT_FIG = _REPO / "figs/stage1_ctx_pred_attempt4_snapshot.png"


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
        f"  ctx_argmax == leader_idx        : {int(ctx_tracks_leader.sum())} / {n}  "
        f"({ctx_tracks_leader.mean() * 100:.1f}%)"
    )
    lines.append(
        f"  pred_argmax == leader_idx       : {int(pred_tracks_leader.sum())} / {n}  "
        f"({pred_tracks_leader.mean() * 100:.1f}%)  [amplifier signature]"
    )
    lines.append(
        f"  pred_argmax == expected_trailer : {int(pred_forecasts.sum())} / {n}  "
        f"({pred_forecasts.mean() * 100:.1f}%)  [<-- forecast gate]"
    )
    lines.append(
        f"  ctx_argmax == expected_trailer  : {int(ctx_forecasts.sum())} / {n}  "
        f"({ctx_forecasts.mean() * 100:.1f}%)"
    )
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

    axs[0, 1].hist(z["pred_ee_w_final"], bins=80, color="teal", edgecolor="k")
    axs[0, 1].set_title(
        f"pred_ee_w_final  (n={z['pred_ee_w_final'].size},  "
        f"mean={z['pred_ee_w_final'].mean():.3f}, max={z['pred_ee_w_final'].max():.3f})"
    )
    axs[0, 1].set_xlabel("weight")
    axs[0, 1].axvline(1.0, color="r", linestyle="--", linewidth=0.8, label="init=1.0")
    axs[0, 1].axvline(1.5, color="orange", linestyle=":", linewidth=0.8, label="ee_w_max=1.5")
    axs[0, 1].legend()

    axs[1, 0].hist(
        z["W_ctx_pred_init"], bins=60, color="gray", edgecolor="k",
        alpha=0.5, label=f"init (mean={z['W_ctx_pred_init'].mean():.4f})",
    )
    axs[1, 0].hist(
        z["W_ctx_pred_final"], bins=60, color="crimson", edgecolor="k",
        alpha=0.6, label=f"final (mean={z['W_ctx_pred_final'].mean():.4f})",
    )
    axs[1, 0].axvline(
        3.0 / 192, color="black", linestyle=":", linewidth=0.8,
        label="row-cap floor 3/192",
    )
    axs[1, 0].set_title("W_ctx_pred  init vs final")
    axs[1, 0].set_xlabel("weight")
    axs[1, 0].legend()

    axs[1, 1].hist(z["elig_final"], bins=80, color="purple", edgecolor="k")
    axs[1, 1].set_title(
        f"elig_final  (n={z['elig_final'].size},  "
        f"mean={z['elig_final'].mean():.3f}, max={z['elig_final'].max():.3f})"
    )
    axs[1, 1].set_xlabel("eligibility trace")
    axs[1, 1].set_yscale("log")

    k = np.arange(z["gate_w_before"].size)
    axs[2, 0].plot(k, z["gate_w_before"], label="w_before (mean)", color="gray")
    axs[2, 0].plot(k, z["gate_w_after"], label="w_after (mean)", color="crimson")
    axs[2, 0].axhline(3.0 / 192, color="black", linestyle=":", linewidth=0.7)
    axs[2, 0].set_title("W_ctx_pred mean across 360 gates")
    axs[2, 0].set_xlabel("gate k (trial index)")
    axs[2, 0].set_ylabel("W mean")
    axs[2, 0].legend()

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
        "Stage-1 ctx_pred attempt #4  — weight + gate-trajectory snapshot  "
        "(seed=42, n_trials=360, Fix A+C active, Fix B(i) reverted by _stage1_h_cfg)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def _ee_structure_split(ctx_ee_w: np.ndarray) -> str:
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

    md = []
    md.append("# Stage-1 ctx_pred attempt #4 — diagnostic snapshot\n")
    md.append(
        "Checkpoint: `expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz`.\n"
        "Log: `logs/train_stage1_ctx_pred_full_seed42_attempt4.log`.\n"
    )
    md.append(
        "Attempt #4 (driver commit `d8ca59f`, Fix A `0f30cd2`, Fix B(i)\n"
        "`b5d8fd7`, Fix C `b5ba400`) was Lead's compound-fix pass following\n"
        "the task-#47 Debugger H1+H3+H4 verdict. The three intended\n"
        "simulator changes were:\n\n"
        "- Fix A: `DEFAULT_TAU_COINC_MS` 20 → 500 ms (span leader→trailer gap).\n"
        "- Fix B(i): `HRingConfig.inh_rho_hz` 2 → 20 Hz (Vogels target rate).\n"
        "- Fix C: `DEFAULT_W_TARGET` 0.05 → 0.0075 (match post-init mean).\n"
    )

    md.append("## Gate verdict — FAIL\n")
    md.append("| check | value | band | result |")
    md.append("|---|---|---|---|")
    md.append("| h_bump_persistence_ms | 10.0 (probe 0.0) | [200, 500] | **FAIL** |")
    md.append("| h_preprobe_forecast_prob | 0.039 | [0.25, 1.00] | **FAIL** |")
    md.append("| no_runaway (ctx_e_rate) | 21.27 Hz | [0, 80] | PASS |\n")

    md.append("## Critical finding: Fix B(i) was silently reverted\n")
    md.append(
        "The attempt-#4 evidence-log JSON reports\n"
        "`\"HRingConfig.inh_rho_hz\": 10.0` for the same `H_CFG` object whose\n"
        "startup banner (earlier in the same run) printed `20.0 Hz`. Root\n"
        "cause: `expectation_snn/brian2_model/train.py` line 553 has\n"
        "`cfg.inh_rho_hz = 10.0` inside `_stage1_h_cfg`, and the helper\n"
        "mutates the passed `cfg` in-place (line 536: `cfg = base or\n"
        "HRingConfig()`), so the driver's `H_CFG` got its field overwritten\n"
        "before Brian2 Synapses were built.\n\n"
        "This is the **same failure pattern** as attempt #2, where\n"
        "`HRingConfig(inh_w_max=2.0)` was clobbered to `1.5` by\n"
        "`_stage1_h_cfg` line 556. Two silent overrides in four attempts —\n"
        "it is a pattern, not a one-off. The Stage-1 helper is treated as\n"
        "canonical by every Stage-1 driver; config changes intended for\n"
        "Stage-1 must land inside the helper, not at the dataclass default.\n\n"
        "Implication for the task-#47 Debugger verdict: the Debugger\n"
        "assumed attempts #1-#3 ran with `inh_rho_hz = 2.0` (the dataclass\n"
        "default). They actually ran with `10.0` (the helper override). So\n"
        "the observed target/actual mismatch was 10 Hz / 21 Hz ≈ 2×, not\n"
        "2 Hz / 21 Hz ≈ 10×. H3 (Vogels iSTDP runaway) is still the right\n"
        "family of hypothesis — the H3 sanity run (bg `b1od0brws`, commit-\n"
        "local monkeypatch `inh_eta = 0`) saw persistence lift from ~10 ms\n"
        "to 990 ms with the 2.0 dataclass default active — but the\n"
        "*quantitative* expectation for Fix B(i) was built on a stale\n"
        "baseline.\n\n"
        "Fix A and Fix C landed correctly: the runtime log line\n"
        "`ctx_pred: tau_coinc=500ms tau_elig=1000ms eta=1.00e-03\n"
        "w_target=0.007` confirms both reached the Brian2 build. These\n"
        "fields live on `HContextPredictionConfig`, which `_stage1_h_cfg`\n"
        "does not touch.\n"
    )

    md.append("## Comparison across attempts\n")
    md.append(
        "| metric | #1/#2 | #3 | #4 |\n"
        "|---|---|---|---|\n"
        "| `W_ctx_pred_init` mean | 0.02502 | 0.00751 | 0.00751 |\n"
        "| `W_ctx_pred_final` mean | 0.01562 (=3/192) | 0.01562 (=3/192) | 0.01562 (=3/192) |\n"
        "| `W_ctx_pred_final` max | ~0.05 | 0.0707 | 0.0758 |\n"
        "| `elig_final` mean | ~1 | 3.03 | 34.12 |\n"
        "| `ctx_e_rate_hz` | 21.27 | 21.27 | 21.27 |\n"
        "| `pred_e_rate_hz` | ~19 | 19.23 | 17.89 |\n"
        "| `h_bump_persistence_ms` | 10 | 10 | 10 |\n"
        "| `h_preprobe_forecast_prob` | 0.000 | 0.139 | **0.039** |\n"
    )
    md.append(
        "Readouts across the table:\n\n"
        "1. The row-cap floor `3.0/192 = 0.015625` is the dominant attractor\n"
        "   for `W_ctx_pred_final` mean across all four attempts — Fix A\n"
        "   (tau_coinc 500 ms) and Fix C (w_target 0.0075) did not release\n"
        "   the mean from it, although they widened the upper tail\n"
        "   (max 0.05 → 0.076).\n"
        "2. `elig_final` mean grew ~11× from #3 → #4, consistent with the\n"
        "   25× longer τ_coinc (Fix A) producing more overlap between\n"
        "   `x_pre` residuals and post-spikes. Eligibility is not the\n"
        "   bottleneck.\n"
        "3. `ctx_e_rate_hz = 21.27` is **bit-identical across all four\n"
        "   attempts** — direct confirmation that the ring's E/I state was\n"
        "   not perturbed by any fix. Fix B(i) would have shifted this\n"
        "   rate (target 20 Hz vs helper-overridden 10 Hz); it didn't.\n"
        "4. Forecast **dropped** from 0.139 (#3) → 0.039 (#4). Fix A + Fix\n"
        "   C together made the forecast worse. Hypothesis (untested):\n"
        "   with τ_coinc spanning the full leader epoch, the rule\n"
        "   accumulates more *leader-leader* coincidence eligibility\n"
        "   because the V1 → H_pred teacher lights up the leader channel\n"
        "   in pred during the leader window. The gate then consolidates\n"
        "   the leader → leader mapping, not leader → expected-trailer.\n"
        "   The V1 → H_pred always-on teacher is the candidate\n"
        "   architectural culprit.\n"
    )

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

    md.append("## Three-factor gate trajectory (360 gates)\n")
    md.append("`gate_dw_sum` per-gate aggregate dW:\n")
    md.append("```")
    md.append(_fmt_stats(_stats(z["gate_dw_sum"])))
    md.append("```")
    md.append("Decile trajectory of `gate_dw_sum`:\n")
    md.append("```")
    md.append(_decile_trajectory(z["gate_dw_sum"]))
    md.append("```")
    md.append("Decile trajectory of `gate_elig_mean`:\n")
    md.append("```")
    md.append(_decile_trajectory(z["gate_elig_mean"]))
    md.append("```")
    md.append("Decile trajectory of `gate_elig_max`:\n")
    md.append("```")
    md.append(_decile_trajectory(z["gate_elig_max"]))
    md.append("```")
    md.append("Decile trajectory of `gate_w_before`:\n")
    md.append("```")
    md.append(_decile_trajectory(z["gate_w_before"]))
    md.append("```")
    md.append(
        f"`gate_n_capped` min/max: "
        f"{int(z['gate_n_capped'].min())} / {int(z['gate_n_capped'].max())} "
        "(presyn rows whose outgoing row-sum triggered the `w_row_max = 3.0` "
        "hard rescale that gate).\n"
    )

    md.append("## Forecast-gate confusion (180 rotating trials)\n")
    md.append("```")
    md.append(_confusion_analysis(
        z["leader_idx"], z["expected_trailer_idx"],
        z["h_argmax_ctx"], z["h_argmax_pred"],
    ))
    md.append("```\n")

    md.append("## Iteration status\n")
    md.append(
        "Per Lead's attempt-#4 dispatch: **0 iterations remaining** after\n"
        "this run. No attempt #5 will be launched without user approval.\n"
        "Lead is escalating the full evidence snapshot to the user; this\n"
        "document is the artefact of record for that escalation.\n"
    )

    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
