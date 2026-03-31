"""Analysis 11: Temporal analysis across windows.

Time x layer x mechanism is a PRIMARY result, not supplementary.
Expectation effects can reverse with time (McDermott 2026, Todorovic & de Lange 2012).

Analyses across five windows:
    - Prestimulus (before probe onset)
    - Early/transient (first 2-3 timesteps of probe)
    - Sustained (steady-state, timesteps 4-7)
    - Late/off-response (post-stimulus)
    - Omission window (if available)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.experiments.paradigm_base import ConditionData, ExperimentResult


@dataclass
class TemporalWindowResult:
    """Results for one temporal window."""
    window_name: str
    mean_response: dict[str, dict[str, float]]  # {condition: {layer: mean}}
    suppression_index: dict[str, float]          # {layer: expected - neutral mean}


@dataclass
class TemporalAnalysisResult:
    """Full temporal analysis across all windows."""
    windows: dict[str, TemporalWindowResult]
    layers: list[str]
    time_course: dict[str, dict[str, Tensor]]  # {condition: {layer: [T] mean across trials}}


def compute_window_responses(
    result: ExperimentResult,
    window_name: str,
    layers: list[str] | None = None,
) -> TemporalWindowResult:
    """Compute mean responses per condition per layer for one window.

    Args:
        result: ExperimentResult from any paradigm.
        window_name: Name of temporal window.
        layers: Which layers to analyse.
    """
    if layers is None:
        layers = ["r_l4", "r_l23", "r_som", "deep_template"]

    if window_name not in result.temporal_windows:
        return TemporalWindowResult(
            window_name=window_name,
            mean_response={}, suppression_index={},
        )

    start, end = result.temporal_windows[window_name]
    mean_response: dict[str, dict[str, float]] = {}

    for cond_name, cd in result.conditions.items():
        layer_means: dict[str, float] = {}
        for layer in layers:
            data = getattr(cd, layer)[:, start:end]
            layer_means[layer] = data.mean().item()
        mean_response[cond_name] = layer_means

    # Suppression index: find expected and neutral conditions
    suppression: dict[str, float] = {}
    exp_conds = [n for n in result.conditions if "dev0" in n and "neutral" not in n]
    neut_conds = [n for n in result.conditions if "neutral" in n and "dev0" in n]

    if exp_conds and neut_conds:
        for layer in layers:
            exp_mean = sum(mean_response[c][layer] for c in exp_conds) / len(exp_conds)
            neut_mean = sum(mean_response[c][layer] for c in neut_conds) / len(neut_conds)
            suppression[layer] = exp_mean - neut_mean

    return TemporalWindowResult(
        window_name=window_name,
        mean_response=mean_response,
        suppression_index=suppression,
    )


def run_temporal_analysis(
    result: ExperimentResult,
    layers: list[str] | None = None,
) -> TemporalAnalysisResult:
    """Run full temporal analysis across all available windows.

    Args:
        result: ExperimentResult from any paradigm.
    """
    if layers is None:
        layers = ["r_l4", "r_l23", "r_som", "deep_template"]

    windows: dict[str, TemporalWindowResult] = {}
    for wname in result.temporal_windows:
        windows[wname] = compute_window_responses(result, wname, layers)

    # Full time course: mean across trials for each condition x layer
    time_course: dict[str, dict[str, Tensor]] = {}
    for cond_name, cd in result.conditions.items():
        layer_tc: dict[str, Tensor] = {}
        for layer in layers:
            data = getattr(cd, layer)  # [n_trials, T, ...]
            if data.ndim == 3:
                layer_tc[layer] = data.mean(dim=(0, 2))  # [T]
            else:
                layer_tc[layer] = data.mean(dim=0).squeeze(-1)  # [T]
        time_course[cond_name] = layer_tc

    return TemporalAnalysisResult(
        windows=windows, layers=layers, time_course=time_course,
    )
