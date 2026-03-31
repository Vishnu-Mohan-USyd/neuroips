"""Analysis 7: Energy analysis and Pareto frontier.

Total activity per correct inference.
Efficiency = accuracy / activity.
Pareto frontier: accuracy vs energy for each mechanism across lambda_energy sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.experiments.paradigm_base import ConditionData, ExperimentResult


@dataclass
class EnergyResult:
    """Energy analysis for one model."""
    total_activity: float              # mean total activity
    excitatory_activity: float         # mean excitatory (L4 + L2/3)
    inhibitory_activity: float         # mean inhibitory (PV + SOM)
    efficiency: float | None           # accuracy / activity (if accuracy provided)
    per_condition: dict[str, float]    # total activity per condition


@dataclass
class ParetoPoint:
    """One point on the Pareto frontier."""
    lambda_energy: float
    accuracy: float
    energy: float
    mechanism: str


def compute_energy(
    result: ExperimentResult,
    window: str = "sustained",
    accuracy: float | None = None,
) -> EnergyResult:
    """Compute total neural activity (energy proxy).

    Energy = sum of absolute firing rates across all populations.

    Args:
        result: ExperimentResult from a paradigm.
        window: Temporal window.
        accuracy: Optional decoding accuracy for efficiency computation.
    """
    start, end = result.temporal_windows[window]
    per_condition: dict[str, float] = {}

    all_exc = []
    all_inh = []

    for cond_name, cd in result.conditions.items():
        r_l4 = cd.r_l4[:, start:end].abs().mean().item()
        r_l23 = cd.r_l23[:, start:end].abs().mean().item()
        r_pv = cd.r_pv[:, start:end].abs().mean().item()
        r_som = cd.r_som[:, start:end].abs().mean().item()

        exc = r_l4 + r_l23
        inh = r_pv + r_som
        per_condition[cond_name] = exc + inh
        all_exc.append(exc)
        all_inh.append(inh)

    mean_exc = sum(all_exc) / max(len(all_exc), 1)
    mean_inh = sum(all_inh) / max(len(all_inh), 1)
    total = mean_exc + mean_inh

    efficiency = accuracy / max(total, 1e-6) if accuracy is not None else None

    return EnergyResult(
        total_activity=total,
        excitatory_activity=mean_exc,
        inhibitory_activity=mean_inh,
        efficiency=efficiency,
        per_condition=per_condition,
    )


def compute_pareto_frontier(
    points: list[ParetoPoint],
) -> list[ParetoPoint]:
    """Extract Pareto-optimal points (maximize accuracy, minimize energy).

    Args:
        points: List of (lambda_energy, accuracy, energy, mechanism) tuples.

    Returns:
        Pareto-optimal subset, sorted by energy.
    """
    sorted_pts = sorted(points, key=lambda p: p.energy)
    frontier = []
    best_acc = -float("inf")

    for pt in sorted_pts:
        if pt.accuracy > best_acc:
            frontier.append(pt)
            best_acc = pt.accuracy

    return frontier
