"""Publication-quality plotting functions.

Uses matplotlib for all figures. Functions accept analysis results
and return figure objects for saving or display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib required for plotting")


def plot_suppression_profile(
    delta_theta,
    suppression,
    surprise=None,
    title: str = "Suppression Profile",
) -> "Figure":
    """Plot suppression (and optionally surprise) by tuning distance.

    Args:
        delta_theta: [n_bins] angular offsets in degrees.
        suppression: [n_bins] suppression values.
        surprise: [n_bins] optional surprise values.
        title: Figure title.
    """
    _check_mpl()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = delta_theta.numpy() if hasattr(delta_theta, "numpy") else delta_theta
    s = suppression.numpy() if hasattr(suppression, "numpy") else suppression
    ax.plot(x, s, "b-o", label="Suppression (exp - neutral)", markersize=4)
    if surprise is not None:
        surp = surprise.numpy() if hasattr(surprise, "numpy") else surprise
        ax.plot(x, surp, "r-s", label="Surprise (unexp - neutral)", markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Angular distance from expected (deg)")
    ax.set_ylabel("Response difference")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_temporal_timecourse(
    time_course: dict[str, dict[str, "torch.Tensor"]],
    layer: str = "r_l23",
    conditions: list[str] | None = None,
    title: str = "Temporal Time Course",
) -> "Figure":
    """Plot temporal time course of mean activity for selected conditions.

    Args:
        time_course: {condition: {layer: [T] time series}}.
        layer: Which layer to plot.
        conditions: Which conditions to include (None = all).
    """
    _check_mpl()
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    conds = conditions or list(time_course.keys())
    for cond in conds:
        if cond in time_course and layer in time_course[cond]:
            tc = time_course[cond][layer]
            y = tc.numpy() if hasattr(tc, "numpy") else tc
            ax.plot(y, label=cond)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Mean {layer} activity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pareto_frontier(
    points: list,
    frontier: list,
    title: str = "Pareto Frontier: Accuracy vs Energy",
) -> "Figure":
    """Plot accuracy vs energy with Pareto frontier.

    Args:
        points: List of ParetoPoint with .energy and .accuracy.
        frontier: Pareto-optimal subset.
    """
    _check_mpl()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    mechs = set(p.mechanism for p in points)
    colors = {m: c for m, c in zip(sorted(mechs), plt.cm.tab10.colors)}

    for p in points:
        ax.scatter(p.energy, p.accuracy, c=[colors.get(p.mechanism, "gray")],
                   alpha=0.5, s=30)

    # Frontier line
    if frontier:
        fe = [p.energy for p in frontier]
        fa = [p.accuracy for p in frontier]
        ax.plot(fe, fa, "k--", linewidth=1.5, label="Pareto frontier")

    # Legend for mechanisms
    for m in sorted(mechs):
        ax.scatter([], [], c=[colors[m]], label=m, s=40)

    ax.set_xlabel("Energy (total activity)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rdm(
    rdm,
    title: str = "Representational Dissimilarity Matrix",
) -> "Figure":
    """Plot a representational dissimilarity matrix as a heatmap."""
    _check_mpl()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    r = rdm.numpy() if hasattr(rdm, "numpy") else rdm
    im = ax.imshow(r, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Distance")
    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Stimulus")
    ax.set_title(title)
    fig.tight_layout()
    return fig
