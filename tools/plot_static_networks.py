"""plot the two static networks from their measured results."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "static" / "main.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "static_figures")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    blue, orange = '#2867A3', '#D58927'
    ink, grey, light = '#243A4B', '#7C858D', '#ADB5BB'
    colors = {'expected': blue, 'unexpected': orange}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'text.color': ink, 'axes.labelcolor': ink,
        'axes.edgecolor': grey, 'axes.linewidth': .75,
        'xtick.color': grey, 'ytick.color': grey,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(12.6, 8.7), facecolor='white')
    left, right, width = .087, .583, .365
    shape_axes = [fig.add_axes([x, .554, width, .305]) for x in (left, right)]
    cost_ax = fig.add_axes([left, .133, width, .257])
    accuracy_ax = fig.add_axes([right, .133, width, .257])

    def tufte(ax, xticks, yticks, *, bounds=None):
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        for side in ('left', 'bottom'):
            ax.spines[side].set_position(('outward', 7))
            ax.spines[side].set_linewidth(.75)
            ax.spines[side].set_color(grey)
        ax.spines['bottom'].set_bounds(min(xticks), max(xticks))
        ax.spines['left'].set_bounds(min(yticks), max(yticks))
        ax.tick_params(direction='out', length=3.5, width=.75, pad=6, labelsize=11.5)
        ax.grid(False)

    fig.legend(handles=[
        Line2D([], [], color=blue, lw=2.5, label='expected'),
        Line2D([], [], color=orange, lw=2.5, label='unexpected'),
        Line2D([], [], color=grey, lw=1.5, ls=(0, (4, 3)), label='baseline'),
    ], loc='upper center', bbox_to_anchor=(.523, .985), ncol=3,
        frameon=False, handlelength=2.3, columnspacing=2.5, fontsize=12.5)

    for i, (ax, kind, title) in enumerate(zip(
            shape_axes, ('sharpening', 'dampening'),
            ('task-prioritized', 'energy-prioritized'))):
        records = data['records'][kind]
        offsets = np.asarray(records[0]['offset_degrees'])
        mask = abs(offsets) <= 45
        reference = np.asarray([row['baseline_curve'] for row in records])
        ax.plot(offsets[mask], reference.mean(0)[mask], color=grey,
                lw=1.5, ls=(0, (4, 3)), zorder=2)
        for condition in ('unexpected', 'expected'):
            curves = np.asarray([row[condition]['curve'] for row in records])
            ax.fill_between(offsets[mask], curves.min(0)[mask], curves.max(0)[mask],
                            color=colors[condition], alpha=.17, linewidth=0, zorder=3)
            ax.plot(offsets[mask], curves.mean(0)[mask], color=colors[condition],
                    lw=2.5, solid_capstyle='round', zorder=4)
        ax.set_xlim(-45, 45)
        ax.set_ylim(0, 1.75)
        tufte(ax, [-40, -20, 0, 20, 40], [0, .5, 1, 1.5])
        ax.set_yticklabels(['0', '0.5', '1.0', '1.5'])
        ax.set_xlabel('orientation offset (°)', fontsize=13, labelpad=11)
        ax.set_ylabel('response (a.u.)', fontsize=13, labelpad=13)
        ax.set_title(title, loc='left', fontsize=18, weight='bold', pad=16)
        ax.text(1, 1.063, f'α = {records[0]["alpha"]:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12.5, color=grey)
        ax.text(-.14, 1.075, chr(97+i), transform=ax.transAxes,
                fontsize=21, weight='bold', va='bottom')

    positions = [0, 1]
    jitter = np.asarray([-.055, 0, .055])
    def paired_dots(ax, metric):
        for group, kind in zip(positions, ('sharpening', 'dampening')):
            records = data['records'][kind]
            expected = np.asarray([row['expected'][metric] for row in records])
            unexpected = np.asarray([row['unexpected'][metric] for row in records])
            for j in range(3):
                ax.plot([group-.20+jitter[j], group+.20+jitter[j]],
                        [expected[j], unexpected[j]], color='#C1C8CD', lw=.9, zorder=1)
            for shift, values, color in ((-.20, expected, blue), (.20, unexpected, orange)):
                ax.scatter(group+shift+jitter, values, s=28, color=color,
                           edgecolors='white', linewidth=.55, zorder=4)
                ax.plot([group+shift-.085, group+shift+.085], [values.mean()]*2,
                        color=color, lw=1.5, zorder=3)
        ax.set_xlim(-.40, 1.40)
        ax.set_xticks(positions, ['task-prioritized', 'energy-prioritized'])
        ax.tick_params(axis='x', length=0, pad=12, labelsize=12)
        ax.spines['bottom'].set_visible(False)

    cost_ax.set_ylim(0, 1.1)
    tufte(cost_ax, positions, [0, .5, 1])
    paired_dots(cost_ax, 'normalized_activity')
    cost_ax.set_yticklabels(['0', '0.5', '1.0'])
    cost_ax.axhline(1, color=light, lw=.9, ls=(0, (4, 3)), zorder=0)
    cost_ax.set_ylabel('activity / reference', fontsize=13, labelpad=13)
    cost_ax.set_title('energy proxy', loc='left', fontsize=18, weight='bold', pad=18)
    cost_ax.text(-.14, 1.09, 'c', transform=cost_ax.transAxes,
                 fontsize=21, weight='bold', va='bottom')

    accuracy_ax.set_ylim(0, 100)
    tufte(accuracy_ax, positions, [0, 25, 50, 75, 100])
    paired_dots(accuracy_ax, 'accuracy_percent')
    accuracy_ax.axhline(100/36, color=light, lw=.9, ls=(0, (4, 3)), zorder=0)
    accuracy_ax.text(1.38, 100/36+1.5, 'chance', ha='right', va='bottom',
                     color=grey, fontsize=10.5)
    accuracy_ax.set_ylabel('accuracy (%)', fontsize=13, labelpad=13)
    accuracy_ax.set_title('decoding', loc='left', fontsize=18, weight='bold', pad=18)
    accuracy_ax.text(-.14, 1.09, 'd', transform=accuracy_ax.transAxes,
                     fontsize=21, weight='bold', va='bottom')

    fig.text(.948, .038, '3 seeds', ha='right', color=grey, fontsize=10.5)
    for extension in ('png', 'svg', 'pdf'):
        destination = out / f'figure1_static_networks.{extension}'
        fig.savefig(destination, dpi=300)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
