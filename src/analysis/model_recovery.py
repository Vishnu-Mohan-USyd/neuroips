"""Model recovery / sensitivity analysis.

In the current architecture, there is a single feedback mechanism (V2 GRU
with head_feedback and Dale's law E/I split). The original three-mechanism
recovery pipeline (dampening / sharpening / center-surround) is no longer
applicable.

This module is kept as a namespace placeholder. The parametric fitting
utilities (SuppressionProfile, FitResult, fit_parametric_models) remain
available for post-training profile analysis if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Suppression-by-tuning profile (mechanism-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class SuppressionProfile:
    """Suppression and surprise profiles as function of tuning distance."""
    delta_theta: Tensor      # [n_bins] angular distances from expected (degrees)
    suppression: Tensor      # [n_bins] Δ_supp = response(expected) - response(neutral)
    surprise: Tensor         # [n_bins] Δ_surp = response(unexpected) - response(neutral)
    raw_expected: Tensor     # [n_bins] mean response in expected condition
    raw_unexpected: Tensor   # [n_bins] mean response in unexpected condition
    raw_neutral: Tensor      # [n_bins] mean response in neutral condition


@dataclass
class FitResult:
    """Result of fitting a parametric model to a suppression profile."""
    model_name: str       # "gaussian_trough", "mexican_hat", "offset_gaussian"
    r_squared: float      # R² goodness of fit
    params: dict          # Fitted parameters
    fitted_curve: Tensor  # [n_points] fitted values


def _r_squared(y_true: Tensor, y_pred: Tensor) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def fit_parametric_models(
    profile: SuppressionProfile,
) -> list[FitResult]:
    """Fit three parametric models to the normalized suppression profile.

    Models:
        1. gaussian_trough: A * exp(-x²/(2σ²)), A < 0
        2. mexican_hat: A_n * exp(-x²/(2σ_n²)) + A_b * exp(-x²/(2σ_b²))
        3. offset_gaussian: C + A * exp(-x²/(2σ²)), C < 0, A > 0

    Normalizes the profile to [-1, 1] for grid search, then rescales.
    """
    x = profile.delta_theta
    y = profile.suppression

    y_range = max(y.abs().max().item(), 1e-6)
    y_norm = y / y_range

    results = []

    # 1. Gaussian trough
    best_r2 = -float("inf")
    best_params = {}
    best_curve = torch.zeros_like(y_norm)
    for a in [v * 0.05 for v in range(-20, 1)]:
        for sigma in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]:
            curve = a * torch.exp(-x ** 2 / (2 * sigma ** 2))
            r2 = _r_squared(y_norm, curve)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"amplitude": a * y_range, "sigma": sigma}
                best_curve = curve * y_range
    results.append(FitResult("gaussian_trough", best_r2, best_params, best_curve))

    # 2. Mexican hat
    best_r2 = -float("inf")
    best_params = {}
    best_curve = torch.zeros_like(y_norm)
    for a_n in [v * 0.1 for v in range(1, 16)]:
        for s_n in [5.0, 8.0, 10.0, 12.0, 15.0]:
            for a_b in [v * 0.1 for v in range(-10, 0)]:
                for s_b in [20.0, 25.0, 30.0, 40.0, 50.0, 60.0]:
                    if s_b <= s_n:
                        continue
                    curve = (a_n * torch.exp(-x ** 2 / (2 * s_n ** 2))
                             + a_b * torch.exp(-x ** 2 / (2 * s_b ** 2)))
                    r2 = _r_squared(y_norm, curve)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {
                            "a_narrow": a_n * y_range, "sigma_narrow": s_n,
                            "a_broad": a_b * y_range, "sigma_broad": s_b,
                        }
                        best_curve = curve * y_range
    results.append(FitResult("mexican_hat", best_r2, best_params, best_curve))

    # 3. Offset Gaussian
    best_r2 = -float("inf")
    best_params = {}
    best_curve = torch.zeros_like(y_norm)
    for c in [v * 0.05 for v in range(-20, 1)]:
        for a in [v * 0.1 for v in range(1, 21)]:
            for sigma in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
                curve = c + a * torch.exp(-x ** 2 / (2 * sigma ** 2))
                r2 = _r_squared(y_norm, curve)
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        "offset": c * y_range, "amplitude": a * y_range, "sigma": sigma,
                    }
                    best_curve = curve * y_range
    results.append(FitResult("offset_gaussian", best_r2, best_params, best_curve))

    return results


def identify_mechanism(fit_results: list[FitResult]) -> str:
    """Identify which model best explains the suppression profile.

    Returns the model_name with the highest R².
    """
    best = max(fit_results, key=lambda f: f.r_squared)
    return best.model_name
