"""Analysis 1+2: Mean response analysis and suppression/surprise-by-tuning profiles.

KEY DIAGNOSTIC: suppression profile shape distinguishes mechanisms.
Also computes surprise profile and their difference (Feuerriegel 2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils import circular_distance_abs
from src.experiments.paradigm_base import ConditionData, ExperimentResult


# ---------------------------------------------------------------------------
# Mean response analysis (Analysis 1)
# ---------------------------------------------------------------------------

@dataclass
class MeanResponseResult:
    """Mean response per population per condition."""
    condition_means: dict[str, dict[str, float]]  # {condition: {layer: mean}}
    populations: list[str]


def compute_mean_responses(
    result: ExperimentResult,
    window: str = "sustained",
) -> MeanResponseResult:
    """Compute mean response per population for each condition.

    Args:
        result: ExperimentResult from a paradigm.
        window: Temporal window name.

    Returns:
        MeanResponseResult with condition x layer means.
    """
    start, end = result.temporal_windows[window]
    populations = ["r_l4", "r_l23", "r_som", "deep_template"]
    condition_means: dict[str, dict[str, float]] = {}

    for cond_name, cd in result.conditions.items():
        means: dict[str, float] = {}
        for pop in populations:
            data = getattr(cd, pop)[:, start:end]  # [n_trials, W, N]
            means[pop] = data.mean().item()
        condition_means[cond_name] = means

    return MeanResponseResult(condition_means=condition_means, populations=populations)


# ---------------------------------------------------------------------------
# Suppression / surprise profiles (Analysis 2)
# ---------------------------------------------------------------------------

@dataclass
class SuppressionProfileResult:
    """Suppression and surprise profiles from trained model data."""
    delta_theta: Tensor          # [n_bins] angular offsets
    suppression: Tensor          # [n_bins] expected - neutral
    surprise: Tensor             # [n_bins] unexpected - neutral
    difference: Tensor           # [n_bins] suppression - surprise
    raw_expected: Tensor         # [n_bins]
    raw_unexpected: Tensor       # [n_bins]
    raw_neutral: Tensor          # [n_bins]


def compute_suppression_profile_from_experiment(
    result: ExperimentResult,
    expected_ori: float,
    window: str = "sustained",
    layer: str = "r_l23",
    n_orientations: int = 36,
    period: float = 180.0,
) -> SuppressionProfileResult:
    """Compute suppression and surprise profiles from experiment results.

    Expects conditions named like 'cw_dev0', 'cw_dev45', 'neutral_dev0', etc.
    Averages CW and CCW rules. Groups units by |pref_theta - expected_theta|.

    Args:
        result: ExperimentResult from P1 (hidden_state).
        expected_ori: The expected orientation (degrees).
        window: Which temporal window to average over.
        layer: Which population to analyse.

    Returns:
        SuppressionProfileResult.
    """
    start, end = result.temporal_windows[window]
    step = period / n_orientations
    pref_oris = torch.arange(n_orientations, dtype=torch.float32) * step
    dists = circular_distance_abs(pref_oris, torch.tensor(expected_ori), period=period)
    unique_dists = dists.unique(sorted=True)

    # Collect expected (dev=0) and neutral (dev=0) conditions
    expected_resp = _average_condition(result, ["cw_dev0", "ccw_dev0"], start, end, layer)
    neutral_resp = _average_condition(result, ["neutral_dev0"], start, end, layer)

    # For surprise, use large deviations (e.g., dev45 or dev90)
    surprise_conds = []
    for name in result.conditions:
        if "dev45" in name and "neutral" not in name:
            surprise_conds.append(name)
    if not surprise_conds:
        for name in result.conditions:
            if "dev90" in name and "neutral" not in name:
                surprise_conds.append(name)
    unexpected_resp = _average_condition(
        result, surprise_conds if surprise_conds else ["cw_dev0"], start, end, layer)

    # Average symmetric units
    avg_exp, avg_unexp, avg_neut = [], [], []
    for d in unique_dists:
        mask = (dists - d).abs() < 0.1
        avg_exp.append(expected_resp[mask].mean())
        avg_unexp.append(unexpected_resp[mask].mean())
        avg_neut.append(neutral_resp[mask].mean())

    avg_exp = torch.stack(avg_exp)
    avg_unexp = torch.stack(avg_unexp)
    avg_neut = torch.stack(avg_neut)

    suppression = avg_exp - avg_neut
    surprise = avg_unexp - avg_neut

    return SuppressionProfileResult(
        delta_theta=unique_dists,
        suppression=suppression,
        surprise=surprise,
        difference=suppression - surprise,
        raw_expected=avg_exp,
        raw_unexpected=avg_unexp,
        raw_neutral=avg_neut,
    )


def _average_condition(
    result: ExperimentResult,
    condition_names: list[str],
    start: int, end: int,
    layer: str,
) -> Tensor:
    """Average response across conditions and trials within a window. Returns [N]."""
    responses = []
    for name in condition_names:
        if name in result.conditions:
            data = getattr(result.conditions[name], layer)[:, start:end]
            responses.append(data.mean(dim=(0, 1)))  # mean over trials and time
    if not responses:
        return torch.zeros(36)
    return torch.stack(responses).mean(dim=0)
