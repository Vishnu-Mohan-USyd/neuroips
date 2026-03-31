"""Analysis 8: Ambiguity bias.

Population vector decode on P3 stimuli.
Signed bias toward expected orientation under ambiguous probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils import circular_distance
from src.experiments.paradigm_base import ExperimentResult


@dataclass
class BiasResult:
    """Perceptual bias analysis result."""
    mean_bias_mixture: float        # signed bias (degrees) for mixture probes
    mean_bias_low_contrast: float   # signed bias for low-contrast probes
    mean_bias_clear: float          # bias for clear probes (should be ~0)
    bias_per_rule: dict[str, float] # per rule (CW/CCW)


def population_vector_decode(
    responses: Tensor,
    n_orientations: int = 36,
    period: float = 180.0,
) -> Tensor:
    """Decode orientation from population response via center of mass.

    Args:
        responses: [n_trials, N] population responses.

    Returns:
        decoded: [n_trials] decoded orientations in degrees.
    """
    step = period / n_orientations
    prefs = torch.arange(n_orientations, dtype=torch.float32) * step  # [N]

    # Convert to complex for circular mean
    angles_rad = prefs * 2 * torch.pi / period
    weights = torch.exp(1j * angles_rad.to(torch.cfloat))
    z = responses.to(torch.cfloat) @ weights
    decoded_rad = z.angle()
    decoded_deg = (decoded_rad.float() * period / (2 * torch.pi)) % period
    return decoded_deg


def compute_bias(
    decoded: Tensor,
    expected_ori: float,
    period: float = 180.0,
) -> float:
    """Compute signed bias toward expected orientation.

    Positive = attraction toward expected.
    Negative = repulsion away from expected.

    Args:
        decoded: [n_trials] decoded orientations.
        expected_ori: expected orientation in degrees.

    Returns:
        Mean signed bias in degrees.
    """
    dist = circular_distance(decoded, torch.tensor(expected_ori), period)
    # Negative distance means decoded is closer to expected than the stimulus
    return -dist.mean().item()  # negate so positive = attraction


def run_bias_analysis(
    result: ExperimentResult,
    expected_ori: float = 45.0,
    window: str = "sustained",
    n_orientations: int = 36,
    period: float = 180.0,
) -> BiasResult:
    """Run ambiguity bias analysis on P3 experiment results.

    Args:
        result: ExperimentResult from AmbiguousParadigm.
        expected_ori: The expected orientation (degrees).
    """
    start, end = result.temporal_windows[window]

    biases: dict[str, list[float]] = {"mixture": [], "low_contrast": [], "clear": []}
    per_rule: dict[str, float] = {}

    for cond_name, cd in result.conditions.items():
        r_l23 = cd.r_l23[:, start:end].mean(dim=1)  # [n_trials, N]
        decoded = population_vector_decode(r_l23, n_orientations, period)
        bias = compute_bias(decoded, expected_ori, period)

        if "mixture" in cond_name:
            biases["mixture"].append(bias)
        elif "low_contrast" in cond_name:
            biases["low_contrast"].append(bias)
        elif "clear" in cond_name:
            biases["clear"].append(bias)

        per_rule[cond_name] = bias

    def _mean(lst: list[float]) -> float:
        return sum(lst) / max(len(lst), 1) if lst else 0.0

    return BiasResult(
        mean_bias_mixture=_mean(biases["mixture"]),
        mean_bias_low_contrast=_mean(biases["low_contrast"]),
        mean_bias_clear=_mean(biases["clear"]),
        bias_per_rule=per_rule,
    )
