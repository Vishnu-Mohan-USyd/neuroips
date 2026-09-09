"""Recreate the two learned-temporal response and training-control figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ("8", "9", "10")
COLORS = {"early": "#078291", "late": "#CE5F27", "baseline": "#65727C",
          "preferred": "#2867AA", "broad": "#D77C19", "near": "#8B5BAE"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "figures" / "temporal_v11" / "results.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "temporal_v11_figures")
    return parser.parse_args()


def curves(results, condition):
    probes = [results["runs"][f"fitted_{condition}"][seed]["probe"] for seed in SEEDS]
    return {
        "offsets": np.asarray(probes[0]["offset_degrees"]),
        "times": np.asarray(probes[0]["times"]),
        "expected": np.asarray([probe["expected"] for probe in probes]),
        "baseline": np.asarray([probe["baseline"] for probe in probes]),
        **{f"{window}_{kind}": np.asarray([probe["windows"][window][kind] for probe in probes])
           for window in ("early", "late") for kind in ("expected", "baseline")},
    }


def band(ax, x, values, color, label, linestyle="-", early_points=False):
    mean = values.mean(0)
    ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.16, linewidth=0)
    line, = ax.plot(x, mean, color=color, lw=2.5, ls=linestyle, label=label)
    if early_points:
        indices = np.flatnonzero(np.isclose(x, .1) | np.isclose(x, .2))
        ax.scatter(x[indices], mean[indices], color=color, s=17,
                   edgecolors="white", linewidths=.5, zorder=4)
    return line


def style_axis(ax, *, time=False, percent=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#91A1AE")
    ax.tick_params(colors="#5F7285", labelsize=11)
    ax.grid(axis="y", color="#E3EAF0", linewidth=.8)
    ax.set_axisbelow(True)
    if time:
        ax.set_xlim(0, 4)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Time (relative units)")
        ax.axvspan(.1, .2, color=COLORS["early"], alpha=.07)
        ax.axvspan(3.1, 4, color=COLORS["late"], alpha=.05)
    else:
        ax.set_xlim(-40, 40)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_xlabel("Orientation offset (°)")
    if percent:
        ax.set_ylim(-100, 112)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:+.0f}" if value > 0 else f"{value:.0f}"))
        ax.axhline(0, color=COLORS["baseline"], lw=1.1, ls="--")
        ax.set_ylabel("Change from baseline (%)")


def temporal_panel(ax, data, near=False):
    offsets = data["offsets"]
    groups = [(offsets == 0, "preferred", "Preferred: 0°", "-"),
              ((np.abs(offsets) >= 15) & (np.abs(offsets) <= 30),
               "broad", "Flanks: ±15–30°", "-")]
    if near:
        groups.append((np.abs(offsets) == 15, "near", "Flanks: ±15°", "--"))
    lines = []
    for mask, color, label, linestyle in groups:
        # Normalize each seed's group aggregate before averaging across seeds.
        values = 100 * (data["expected"][:, :, mask].mean(-1)
                        / data["baseline"][:, :, mask].mean(-1) - 1)
        lines.append(band(ax, data["times"], values, COLORS[color], label,
                          linestyle, early_points=True))
    style_axis(ax, time=True, percent=True)
    return lines


def save_figure(fig, out_dir, name):
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    results = json.loads(args.results.read_text())
    data = {condition: curves(results, condition)
            for condition in ("joint", "accuracy_only", "energy_only", "sustained", "fast_sst")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 13, "axes.labelcolor": "#203D53",
                         "text.color": "#183348", "legend.fontsize": 11,
                         "axes.titlesize": 15, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.25))
    fig.subplots_adjust(left=.078, right=.985, bottom=.09, top=.895, wspace=.25, hspace=.35)
    fig.suptitle("Expected stimulus · early accuracy + energy", x=.078, y=.973,
                 ha="left", fontsize=19, fontweight="bold")
    joint = data["joint"]
    mask = np.abs(joint["offsets"]) <= 40
    offsets = joint["offsets"][mask]
    ax = axes[0, 0]
    band(ax, offsets, joint["early_expected"][:, mask], COLORS["early"], "Early: t = 0.1, 0.2")
    band(ax, offsets, joint["late_expected"][:, mask], COLORS["late"], "Late: t = 3.1–4.0")
    band(ax, offsets, joint["early_baseline"][:, mask], COLORS["baseline"], "Baseline", "--")
    style_axis(ax)
    ax.set_ylim(0, 1.1 * max(joint[f"{window}_{kind}"][:, mask].max()
                            for window in ("early", "late") for kind in ("expected", "baseline")))
    ax.set_ylabel("Rate (a.u.)")
    ax.set_title("A  Response shape", loc="left", pad=14)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    for window in ("early", "late"):
        ratio = 100 * (joint[f"{window}_expected"] / joint[f"{window}_baseline"] - 1)
        band(ax, offsets, ratio[:, mask], COLORS[window], window.capitalize())
    style_axis(ax, percent=True)
    ax.set_title("B  Relative response", loc="left", pad=14)
    ax.legend(frameon=False, loc="upper right")

    for ax, condition, title in ((axes[1, 0], "joint", "C  Equal initial kinetics"),
                                 (axes[1, 1], "fast_sst", "D  SST initially 10× faster")):
        temporal_panel(ax, data[condition], near=True)
        ax.set_title(title, loc="left", pad=14)
        ax.legend(frameon=False, loc="upper right")
    fig.text(.985, .015, "Mean and range · 3 seeds", ha="right", color="#6C7A86", fontsize=10)
    save_figure(fig, args.out_dir, "temporal_response_minimal")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.25))
    fig.subplots_adjust(left=.078, right=.985, bottom=.09, top=.84, wspace=.25, hspace=.36)
    fig.suptitle("Expected stimulus · training controls", x=.078, y=.973,
                 ha="left", fontsize=19, fontweight="bold")
    for ax, condition, title in ((axes[0, 0], "joint", "A  Early accuracy + energy"),
                                 (axes[0, 1], "accuracy_only", "B  Accuracy only"),
                                 (axes[1, 0], "sustained", "C  Sustained accuracy + energy"),
                                 (axes[1, 1], "energy_only", "D  Energy only")):
        lines = temporal_panel(ax, data[condition])
        ax.set_title(title, loc="left", pad=14)
    fig.legend(lines, [line.get_label() for line in lines], loc="upper center",
               bbox_to_anchor=(.57, .937), ncol=2, frameon=False)
    fig.text(.985, .015, "Mean and range · 3 seeds", ha="right", color="#6C7A86", fontsize=10)
    save_figure(fig, args.out_dir, "temporal_controls_minimal")
    print(args.out_dir)


if __name__ == "__main__":
    main()
