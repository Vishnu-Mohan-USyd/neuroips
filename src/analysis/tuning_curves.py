"""Analysis 3: Tuning curve analysis.

Von Mises fits per unit per condition.
Extracts amplitude, width, preferred orientation.
Separates gain vs width vs shift effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils import circular_distance_abs


@dataclass
class TuningFit:
    """Von Mises fit for one unit in one condition."""
    amplitude: float       # peak - baseline
    baseline: float        # minimum response
    preferred_ori: float   # degrees
    width: float           # concentration parameter (kappa)
    r_squared: float


@dataclass
class TuningAnalysisResult:
    """Tuning curve analysis across conditions."""
    fits: dict[str, list[TuningFit]]  # {condition: [fit per unit]}
    mean_amplitude: dict[str, float]
    mean_width: dict[str, float]
    mean_shift: dict[str, float]  # shift relative to neutral preferred


def fit_von_mises(
    responses: Tensor,
    orientations: Tensor,
    period: float = 180.0,
) -> TuningFit:
    """Fit a Von Mises-like function to a tuning curve via grid search.

    Args:
        responses: [n_orientations] mean response at each orientation.
        orientations: [n_orientations] orientation values in degrees.

    Returns:
        TuningFit with best parameters.
    """
    n = len(responses)
    best_r2 = -float("inf")
    best_params = {"amplitude": 0.0, "baseline": 0.0, "preferred": 0.0, "kappa": 1.0}

    baseline_est = responses.min().item()
    amp_est = (responses.max() - responses.min()).item()
    pref_est = orientations[responses.argmax()].item()

    # Grid search over kappa (width)
    for kappa in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        # For each kappa, optimize preferred via grid
        for pref_offset in [-5.0, -2.5, 0.0, 2.5, 5.0]:
            pref = (pref_est + pref_offset) % period
            dists = circular_distance_abs(orientations, torch.tensor(pref), period)
            # Von Mises-like: baseline + amp * exp(kappa * (cos(2*pi*dist/period) - 1))
            cos_arg = torch.cos(2 * torch.pi * dists / period)
            curve = baseline_est + amp_est * torch.exp(kappa * (cos_arg - 1.0))

            ss_res = ((responses - curve) ** 2).sum()
            ss_tot = ((responses - responses.mean()) ** 2).sum()
            r2 = (1.0 - ss_res / max(ss_tot, 1e-12)).item()

            if r2 > best_r2:
                best_r2 = r2
                best_params = {
                    "amplitude": amp_est, "baseline": baseline_est,
                    "preferred": pref, "kappa": kappa,
                }

    return TuningFit(
        amplitude=best_params["amplitude"],
        baseline=best_params["baseline"],
        preferred_ori=best_params["preferred"],
        width=best_params["kappa"],
        r_squared=best_r2,
    )


def analyse_tuning_curves(
    condition_responses: dict[str, Tensor],
    n_orientations: int = 36,
    period: float = 180.0,
) -> TuningAnalysisResult:
    """Analyse tuning curves across conditions.

    Args:
        condition_responses: {condition_name: [n_units, n_orientations]} mean responses.

    Returns:
        TuningAnalysisResult.
    """
    step = period / n_orientations
    orientations = torch.arange(n_orientations, dtype=torch.float32) * step

    fits: dict[str, list[TuningFit]] = {}
    mean_amp: dict[str, float] = {}
    mean_width: dict[str, float] = {}

    for cond_name, resp in condition_responses.items():
        n_units = resp.shape[0]
        cond_fits = []
        for u in range(n_units):
            cond_fits.append(fit_von_mises(resp[u], orientations, period))
        fits[cond_name] = cond_fits
        mean_amp[cond_name] = sum(f.amplitude for f in cond_fits) / max(n_units, 1)
        mean_width[cond_name] = sum(f.width for f in cond_fits) / max(n_units, 1)

    # Compute shift relative to neutral
    mean_shift: dict[str, float] = {}
    neutral_prefs = None
    for name in fits:
        if "neutral" in name:
            neutral_prefs = [f.preferred_ori for f in fits[name]]
            break

    for cond_name, cond_fits in fits.items():
        if neutral_prefs is not None and len(neutral_prefs) == len(cond_fits):
            shifts = [
                circular_distance_abs(
                    torch.tensor(f.preferred_ori),
                    torch.tensor(np), period,
                ).item()
                for f, np in zip(cond_fits, neutral_prefs)
            ]
            mean_shift[cond_name] = sum(shifts) / max(len(shifts), 1)
        else:
            mean_shift[cond_name] = 0.0

    return TuningAnalysisResult(
        fits=fits, mean_amplitude=mean_amp,
        mean_width=mean_width, mean_shift=mean_shift,
    )
