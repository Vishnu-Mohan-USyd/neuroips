#!/usr/bin/env python3
"""Plot the saved three-seed, high-alpha temporal response profiles."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=root / "figures" / "temporal_v10",
                        help="Directory containing seed_8/9/10/temporal_assay.json.")
    parser.add_argument("--out", type=Path, default=root / "outputs" / "temporal_shape_profile",
                        help="Output file stem; writes .png and .svg.")
    args = parser.parse_args()

    records = [json.loads((args.results_root / f"seed_{s}" / "temporal_assay.json").read_text())
               for s in (8, 9, 10)]
    probes = [d["per_alpha"]["0.7"]["probe"] for d in records]
    times = np.asarray(probes[0]["times"])
    offsets = np.asarray(probes[0]["offset_degrees"])
    expected = np.asarray([p["expected"] for p in probes])
    unexpected = np.asarray([p["unexpected"] for p in probes])
    baseline = np.asarray([p["baseline"] for p in probes])
    early = np.isclose(times, .1) | np.isclose(times, .2)
    late = (times > 3.000001) & (times <= 4.000001)
    visible = np.abs(offsets) <= 30
    x = offsets[visible]
    center = np.flatnonzero(offsets == 0)[0]
    flanks = np.flatnonzero(np.abs(offsets) == 15)
    ratios = expected / baseline
    center_ratios = ratios[:, :, center]
    flank_ratios = ratios[:, :, flanks].mean(axis=-1)
    percent_change = 100 * (ratios - 1)

    ink = "#173047"
    muted = "#5C6C79"
    blue = "#1465A1"
    orange = "#CE7135"
    teal = "#168678"
    gray = "#65717A"
    early_fill = "#FFF0C8"
    late_fill = "#ECE7F6"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 13,
        "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": "#A5AFB8", "axes.linewidth": .8,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(15.2, 11.6), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=.077, right=.939, bottom=.183, top=.853,
                           wspace=.255, hspace=.43)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    a, b, c, d = axes
    fig.text(.077, .949, "Early sharpening → late central suppression", fontsize=23,
             weight="bold", color=ink)
    fig.text(.077, .914,
             "One held stimulus  ·  α = 0.70  ·  mean of seeds 8, 9 and 10  ·  shaded bands: seed range (min–max)",
             fontsize=11.7, color=muted)
    fig.text(.077, .885,
             "Each seed curve averages 216 matched continuation / reversal histories.",
             fontsize=10.5, color=muted)

    def style(ax, letter, title):
        ax.set_title(f"{letter}   {title}", loc="left", weight="bold", pad=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=4, width=.7)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E9ED", linewidth=.65)

    def band(ax, xx, values, color, label=None, linestyle="-", marker=None, width=2.35):
        mean = values.mean(axis=0)
        ax.fill_between(xx, values.min(axis=0), values.max(axis=0), color=color,
                        alpha=.19, linewidth=0, zorder=2)
        ax.plot(xx, mean, color=color, linewidth=width, linestyle=linestyle, label=label,
                marker=marker, markersize=3.8, markeredgewidth=.6,
                markeredgecolor="white", zorder=3)
        return mean

    for ax, mask, letter, title in (
        (a, early, "A", "Early · mean at t = 0.1 and 0.2"),
        (b, late, "B", "Late · mean over 3 < t ≤ 4"),
    ):
        style(ax, letter, title)
        band(ax, x, baseline[:, mask].mean(axis=1)[:, visible], gray,
             "First-stimulus baseline", "--", width=1.8)
        band(ax, x, unexpected[:, mask].mean(axis=1)[:, visible], orange,
             "Unexpected (reversal)")
        band(ax, x, expected[:, mask].mean(axis=1)[:, visible], blue,
             "Expected (continuation)", marker="o")
        ax.set_xlim(-31, 31)
        ax.set_ylim(0, 1.76)
        ax.set_xticks([-30, -15, 0, 15, 30])
        ax.set_yticks([0, .4, .8, 1.2, 1.6])
        ax.set_xlabel("Preferred-orientation offset from stimulus (°)")
        ax.set_ylabel("E response (arbitrary rate units)")
        ax.axvline(0, color="#C9D1D8", linewidth=.8, linestyle=":", zorder=1)

    legend_handles = [
        Line2D([0], [0], color=blue, lw=2.35, marker="o", ms=4, label="Expected"),
        Line2D([0], [0], color=orange, lw=2.35, label="Unexpected"),
        Line2D([0], [0], color=gray, lw=1.8, ls="--", label="First-stimulus baseline"),
    ]
    a.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9.4,
             handlelength=2.3, borderaxespad=.2, labelspacing=.5)
    a.annotate("Raised 0° response", xy=(0, expected[:, early, center].mean()),
               xytext=(9, 1.58), color=blue, fontsize=10,
               arrowprops={"arrowstyle": "-", "color": blue, "lw": .9},
               ha="left", va="center")
    b.text(.045, .88, "Broad central trough\nwith a small 0° bump", transform=b.transAxes,
           color=blue, fontsize=10.5, va="top", linespacing=1.35)
    b.add_patch(Rectangle((-30, 0), 60, .26, facecolor="none", edgecolor=blue,
                          linewidth=.9, linestyle=(0, (3, 3)), alpha=.65, zorder=4))
    inset = b.inset_axes([.692, .585, .30, .35])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color("#9BB2C2")
        spine.set_linewidth(.8)
    band(inset, x, expected[:, late].mean(axis=1)[:, visible], blue, marker="o", width=1.8)
    inset.set_xlim(-30, 30)
    inset.set_ylim(0, .26)
    inset.set_xticks([-20, 0, 20])
    inset.set_yticks([0, .1, .2])
    inset.tick_params(labelsize=8, length=2.5, pad=2)
    inset.set_title("Expected · expanded y", fontsize=8.8, pad=5, color=blue)
    inset.grid(axis="y", color="#E8EDF1", lw=.5)
    inset.set_xlabel("Offset (°)", fontsize=8, labelpad=1)

    style(c, "C", "Expected center and flanks through time")
    c.axvspan(.1, .2, color=early_fill, zorder=0)
    c.axvspan(3, 4, color=late_fill, zorder=0)
    band(c, times, center_ratios, blue, "Center: 0°")
    band(c, times, flank_ratios, teal, "Flanks: mean of −15° and +15°")
    for values, color in ((center_ratios, blue), (flank_ratios, teal)):
        c.plot(times[early], values.mean(axis=0)[early], "o", color=color,
               markeredgecolor="white", markeredgewidth=.8, markersize=6, zorder=5)
    c.axhline(1, color=gray, linestyle=(0, (4, 3)), linewidth=1.15, zorder=1)
    c.text(1.94, 1.035, "First-stimulus baseline = 1", color=gray, fontsize=9.3,
           bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": .8})
    c.set_xlim(0, 4)
    c.set_ylim(0, max(1.75, 1.08 * center_ratios.max()))
    c.set_xticks([0, 1, 2, 3, 4])
    c.set_yticks([0, .5, 1, 1.5])
    c.set_xlabel("Time since stimulus onset (relative units)")
    c.set_ylabel("E response / channel baseline")
    c.legend(loc="upper center", bbox_to_anchor=(.55, .94), frameon=False,
             fontsize=9.6, handlelength=2.2)
    c.text(.15, .982, "early", transform=c.get_xaxis_transform(), ha="left", va="top",
           fontsize=9, color="#927018")
    c.text(3.5, .982, "late", transform=c.get_xaxis_transform(), ha="center", va="top",
           fontsize=9, color="#6C5A8B")

    style(d, "D", "Expected response change across orientation")
    cmap = LinearSegmentedColormap.from_list(
        "suppression_enhancement", ["#184D82", "#75A9CC", "#F8F7F1", "#E7A087", "#A33D32"])
    mesh = d.pcolormesh(times, x, percent_change.mean(axis=0)[:, visible].T,
                        cmap=cmap, norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100),
                        shading="nearest", rasterized=True)
    d.set_xlim(0, 4)
    d.set_ylim(-32.5, 32.5)
    d.set_xticks([0, 1, 2, 3, 4])
    d.set_yticks([-30, -15, 0, 15, 30])
    d.set_xlabel("Time since stimulus onset (relative units)")
    d.set_ylabel("Preferred-orientation offset (°)")
    d.grid(False)
    for time in (.1, .2):
        d.axvline(time, color="#E5C46E", linewidth=.95, linestyle=(0, (2, 2)), alpha=.9)
    d.axvline(3, color="white", linewidth=1.1, linestyle=(0, (4, 3)), alpha=.9)
    d.text(.31, 28.5, "early samples", fontsize=8.8, color="#18324A",
           bbox={"facecolor": "white", "edgecolor": "none", "alpha": .72, "pad": 2})
    d.text(3.49, 28.5, "late", fontsize=9, color="#18324A", ha="center",
           bbox={"facecolor": "white", "edgecolor": "none", "alpha": .72, "pad": 2})
    cbar_ax = d.inset_axes([1.025, 0, .034, 1])
    cbar = fig.colorbar(mesh, cax=cbar_ax, ticks=[-100, -50, 0, 50, 100])
    cbar.ax.set_yticklabels(["−100", "−50", "0", "+50", "+100"])
    cbar.ax.tick_params(labelsize=9, length=3)
    cbar.set_label("Change from channel baseline (%)", fontsize=9.8, labelpad=9)
    cbar.outline.set_linewidth(.6)

    fig.text(.077, .119,
             "Baseline: sequence-first E response before prior feedback, aligned to its own stimulus; it includes the local sensory circuit.",
             fontsize=10, color=muted)
    fig.text(.077, .094,
             "C–D normalize each channel within each seed before averaging. Bands show the three-seed range, not a confidence interval.",
             fontsize=10, color=muted)
    fig.text(.077, .069,
             "Time is relative, with fixed SST τ = 1. Low-alpha (α = 0.07) networks also show the transition; this figure shows α = 0.70.",
             fontsize=10, color=muted)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png, svg = args.out.with_suffix(".png"), args.out.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=.18)
    fig.savefig(svg, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
