"""Aggregate Sprint 5b balance-sweep results into summary table + figure.

Sprint 5b (task #29): sweep r ∈ {0.25, 0.5, 1.0, 2.0, 4.0} (intact, seed=42),
collect primary metrics from each of the three assays (Kok / Richter / Tang),
and produce:

  (a) a 6-panel figure showing each primary metric vs r;
  (b) a markdown table for the evidence log;
  (c) the H1 verdict (regime-dependency: opposite signs at the extremes
      of r, where r=0.25 is SOM-dominated feedback and r=4.0 is
      direct-apical-dominated feedback).

Inputs
------
``expectation_snn/data/checkpoints/sprint_5b_intact_r{r}_seed42.npz`` for
each r in the sweep (the r=1.0 file is symlinked from the Sprint 5a
result to avoid re-running).

Usage
-----
    python -m expectation_snn.analysis.balance_sweep

Writes:
    expectation_snn/data/figures/sprint_5b_balance_sweep.png
    expectation_snn/data/figures/sprint_5b_summary.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


BALANCE_R_VALUES: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]
SEED = 42
CKPT_DIR = Path(__file__).resolve().parents[1] / "data" / "checkpoints"
FIG_DIR = Path(__file__).resolve().parents[1] / "data" / "figures"


def _npz_path(r: float) -> Path:
    # Natural float repr ("0.25", "0.5", "1.0", "2.0", "4.0") — matches the
    # filenames written by run_sprint_5a.py with --out. Using `:.1f` would
    # round 0.25 -> "0.2" and collide with r=0.2 (not in the sweep today,
    # but cheap to future-proof).
    return CKPT_DIR / f"sprint_5b_intact_r{r}_seed{SEED}.npz"


def load_sweep() -> Dict[float, Dict[str, np.ndarray]]:
    """Load the sweep into a dict keyed by r, each value a dict of arrays."""
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for r in BALANCE_R_VALUES:
        p = _npz_path(r)
        if not p.exists():
            raise FileNotFoundError(f"missing sweep result for r={r}: {p}")
        z = np.load(p, allow_pickle=True)
        out[r] = {k: z[k] for k in z.files}
    return out


def summarize_primary_metrics(
    sweep: Dict[float, Dict[str, np.ndarray]],
) -> Dict[str, Dict[float, float]]:
    """Pick out the headline scalar per primary metric per r."""
    rows: Dict[str, Dict[float, float]] = {
        "kok_amp_delta_hz":     {},  # valid - invalid, total rate
        "kok_omission_mean_hz": {},  # mean over neurons of omission delta
        "kok_svm":              {},  # SVM validity decoding accuracy
        "kok_bin0_delta":       {},  # pref-rank bin 0 delta

        "richter_redist":            {},
        "richter_center_delta":      {},
        "richter_flank_delta":       {},
        "richter_E_local_delta":     {},  # cell_type[E=0, local=0]

        "tang_mean_delta_hz":    {},
        "tang_svm":              {},
        "tang_laminar_delta_hz": {},
        "tang_fwhm_expected":    {},  # median fwhm, r^2 > 0.5
        "tang_fwhm_deviant":     {},

        "wall_time_min_total":   {},
    }

    for r, data in sweep.items():
        # Kok
        rows["kok_amp_delta_hz"][r] = float(
            data["kok.mean_amp_valid_hz"] - data["kok.mean_amp_invalid_hz"]
        )
        om = data["kok.omission_delta"]
        om = om[np.isfinite(om)]
        rows["kok_omission_mean_hz"][r] = float(np.nanmean(om)) if om.size else float("nan")
        rows["kok_svm"][r] = float(data["kok.svm_accuracy"])
        rows["kok_bin0_delta"][r] = float(data["kok.pref_rank_bin_delta"][0])

        # Richter
        rows["richter_redist"][r] = float(data["richter.redist"])
        rows["richter_center_delta"][r] = float(data["richter.center_delta"])
        rows["richter_flank_delta"][r] = float(data["richter.flank_delta"])
        ct = data["richter.cell_type_delta_hz"]  # (3 pops, 3 dists)
        rows["richter_E_local_delta"][r] = float(ct[0, 0])

        # Tang
        rows["tang_mean_delta_hz"][r] = float(data["tang.mean_delta_hz"])
        rows["tang_svm"][r] = float(data["tang.svm_accuracy"])
        rows["tang_laminar_delta_hz"][r] = float(data["tang.laminar_delta_hz"])
        fwhm_e = data["tang.tuning_expected_fwhm"]
        r2_e = data["tang.tuning_expected_r2"]
        fwhm_d = data["tang.tuning_deviant_fwhm"]
        r2_d = data["tang.tuning_deviant_r2"]
        ok_e = np.isfinite(fwhm_e) & (r2_e > 0.5)
        ok_d = np.isfinite(fwhm_d) & (r2_d > 0.5)
        rows["tang_fwhm_expected"][r] = (
            float(np.median(fwhm_e[ok_e])) if ok_e.any() else float("nan")
        )
        rows["tang_fwhm_deviant"][r] = (
            float(np.median(fwhm_d[ok_d])) if ok_d.any() else float("nan")
        )

        # Wall
        total_min = float(
            data.get(
                "_provenance.total_wall_min",
                np.asarray(
                    (
                        float(data["kok.wall_time_s"])
                        + float(data["richter.wall_time_s"])
                        + float(data["tang.wall_time_s"])
                    )
                    / 60.0
                ),
            )
        )
        rows["wall_time_min_total"][r] = total_min

    return rows


def write_summary_md(
    rows: Dict[str, Dict[float, float]],
    r_values: List[float],
    out_path: Path,
) -> None:
    """Write a clean markdown table to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["metric"] + [f"r={r:g}" for r in r_values]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for mname, per_r in rows.items():
        cells = [mname]
        for r in r_values:
            v = per_r.get(r, float("nan"))
            if mname == "wall_time_min_total":
                cells.append(f"{v:.1f}")
            elif abs(v) >= 1.0 or v == 0.0:
                cells.append(f"{v:+.3f}")
            else:
                cells.append(f"{v:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def classify_h1(rows: Dict[str, Dict[float, float]]) -> Dict[str, str]:
    """Return per-metric H1 regime-dependency verdict.

    H1 predicts opposite signs at the r={0.25, 4.0} extremes for metrics
    that are genuinely feedback-dependent (regime-switch). A metric is
    marked:
      - "REGIME-SWITCH" if sign(r=0.25) != sign(r=4.0) AND both |v| > 0.01
      - "MONOTONIC"     if sign(r=0.25) == sign(r=4.0) AND |v(4.0)| > |v(0.25)|
      - "NULL"          if max abs over sweep < 0.01
      - "NONMONOTONIC"  otherwise.
    """
    verdicts: Dict[str, str] = {}
    for mname, per_r in rows.items():
        if mname == "wall_time_min_total":
            continue
        vs = [per_r[r] for r in BALANCE_R_VALUES]
        v_lo, v_hi = per_r[0.25], per_r[4.0]
        max_abs = max(abs(v) for v in vs)
        if max_abs < 0.01:
            verdicts[mname] = "NULL"
        elif np.sign(v_lo) != np.sign(v_hi) and abs(v_lo) > 0.01 and abs(v_hi) > 0.01:
            verdicts[mname] = "REGIME-SWITCH"
        elif np.sign(v_lo) == np.sign(v_hi) and abs(v_hi) > abs(v_lo):
            verdicts[mname] = "MONOTONIC"
        else:
            verdicts[mname] = "NONMONOTONIC"
    return verdicts


def plot_sweep(
    rows: Dict[str, Dict[float, float]],
    r_values: List[float],
    out_path: Path,
) -> None:
    """6-panel overview figure of primary metrics vs r."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()

    def _plot(ax, metric_names: List[str], title: str, ylabel: str) -> None:
        for mname in metric_names:
            ys = [rows[mname][r] for r in r_values]
            ax.plot(r_values, ys, marker="o", label=mname)
        ax.set_xscale("log")
        ax.set_xticks(r_values)
        ax.set_xticklabels([f"{r:g}" for r in r_values])
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel("r = g_direct / g_SOM")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    _plot(axes[0], ["kok_amp_delta_hz", "kok_bin0_delta"],
          "Kok amp modulation", "Δ valid − invalid (Hz)")
    _plot(axes[1], ["kok_omission_mean_hz"],
          "Kok omission response", "mean Δ over neurons (Hz)")
    _plot(axes[2], ["kok_svm", "tang_svm"],
          "SVM decoding accuracies", "accuracy")

    _plot(axes[3], ["richter_redist", "richter_center_delta", "richter_flank_delta"],
          "Richter center-vs-flank", "Δ rate (Hz)")
    _plot(axes[4], ["richter_E_local_delta"],
          "Richter E local Δ (exp − unexp)", "Δ rate (Hz)")
    _plot(axes[5], ["tang_mean_delta_hz", "tang_laminar_delta_hz"],
          "Tang per-cell & laminar Δ (dev − exp)", "Δ rate (Hz)")

    fig.suptitle(
        f"Sprint 5b: balance sweep (intact, seed={SEED})\n"
        f"r={r_values}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_tuning_widths(
    rows: Dict[str, Dict[float, float]],
    r_values: List[float],
    out_path: Path,
) -> None:
    """Tang tuning FWHM (H3 null) across the sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fwhm_e = [rows["tang_fwhm_expected"][r] for r in r_values]
    fwhm_d = [rows["tang_fwhm_deviant"][r] for r in r_values]
    ax.plot(r_values, fwhm_e, "o-", label="expected")
    ax.plot(r_values, fwhm_d, "s-", label="deviant")
    ax.set_xscale("log")
    ax.set_xticks(r_values)
    ax.set_xticklabels([f"{r:g}" for r in r_values])
    ax.set_xlabel("r = g_direct / g_SOM")
    ax.set_ylabel("median FWHM (rad)")
    ax.set_title("Tang tuning width (r^2 > 0.5 only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    print(f"[sprint-5b] loading sweep from {CKPT_DIR}")
    sweep = load_sweep()
    print(f"[sprint-5b] found {len(sweep)} r-values: "
          f"{sorted(sweep.keys())}")

    rows = summarize_primary_metrics(sweep)
    verdicts = classify_h1(rows)

    # Print summary to stdout.
    print("\n=== Primary metrics vs r ===")
    col_w = 18
    hdr = f"{'metric':<28} " + " ".join(f"r={r:<6g}" for r in BALANCE_R_VALUES) + " verdict"
    print(hdr)
    print("-" * len(hdr))
    for mname, per_r in rows.items():
        if mname == "wall_time_min_total":
            continue
        vals = " ".join(f"{per_r[r]:+7.3f}" for r in BALANCE_R_VALUES)
        print(f"{mname:<28} {vals}  {verdicts.get(mname, '')}")
    print()

    # Write markdown summary table.
    md_path = FIG_DIR / "sprint_5b_summary.md"
    write_summary_md(rows, BALANCE_R_VALUES, md_path)
    print(f"[sprint-5b] wrote summary md -> {md_path}")

    # Write H1 verdict file.
    verdict_path = FIG_DIR / "sprint_5b_h1_verdict.md"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    vlines = [
        "# Sprint 5b — H1 regime-dependency verdict",
        "",
        "H1 predicts: opposite signs at r={0.25, 4.0} for metrics genuinely",
        "regulated by the feedback balance (SOM-dominated vs direct-apical-dominated).",
        "",
        "| metric | verdict |",
        "|---|---|",
    ]
    for mname, v in verdicts.items():
        vlines.append(f"| {mname} | {v} |")
    verdict_path.write_text("\n".join(vlines) + "\n")
    print(f"[sprint-5b] wrote H1 verdict -> {verdict_path}")

    # Figures.
    fig_main = FIG_DIR / "sprint_5b_balance_sweep.png"
    plot_sweep(rows, BALANCE_R_VALUES, fig_main)
    print(f"[sprint-5b] wrote main figure -> {fig_main}")
    fig_fwhm = FIG_DIR / "sprint_5b_tang_fwhm.png"
    plot_tuning_widths(rows, BALANCE_R_VALUES, fig_fwhm)
    print(f"[sprint-5b] wrote FWHM figure -> {fig_fwhm}")

    # Verdict summary.
    n_switch = sum(1 for v in verdicts.values() if v == "REGIME-SWITCH")
    n_mono   = sum(1 for v in verdicts.values() if v == "MONOTONIC")
    n_null   = sum(1 for v in verdicts.values() if v == "NULL")
    n_non    = sum(1 for v in verdicts.values() if v == "NONMONOTONIC")
    print(f"\nH1 summary: {n_switch} REGIME-SWITCH, {n_mono} MONOTONIC, "
          f"{n_null} NULL, {n_non} NONMONOTONIC (of {len(verdicts)} metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
