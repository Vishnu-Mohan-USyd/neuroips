"""Tests for model recovery / sensitivity analysis.

The original three-mechanism recovery pipeline (dampening / sharpening /
center-surround) has been removed. The current architecture uses a single
emergent feedback mechanism (V2 GRU with head_feedback and Dale's law
E/I split).

Parametric fitting utilities are still available in
src.analysis.model_recovery for post-training profile analysis.
"""

from __future__ import annotations

import pytest
import torch

from src.analysis.model_recovery import (
    SuppressionProfile,
    FitResult,
    fit_parametric_models,
    identify_mechanism,
)


class TestParametricFitting:
    """Tests for the mechanism-agnostic parametric fitting utilities."""

    @pytest.fixture
    def synthetic_trough_profile(self):
        """A Gaussian trough profile (like dampening)."""
        x = torch.arange(19, dtype=torch.float32) * 5.0  # 0 to 90 degrees
        suppression = -0.5 * torch.exp(-x ** 2 / (2 * 12.0 ** 2))
        return SuppressionProfile(
            delta_theta=x,
            suppression=suppression,
            surprise=torch.zeros_like(x),
            raw_expected=1.0 + suppression,
            raw_unexpected=torch.ones_like(x),
            raw_neutral=torch.ones_like(x),
        )

    def test_fit_returns_three_models(self, synthetic_trough_profile):
        fits = fit_parametric_models(synthetic_trough_profile)
        assert len(fits) == 3

    def test_fit_result_fields(self, synthetic_trough_profile):
        fits = fit_parametric_models(synthetic_trough_profile)
        for fit in fits:
            assert isinstance(fit, FitResult)
            assert isinstance(fit.model_name, str)
            assert isinstance(fit.r_squared, float)
            assert isinstance(fit.params, dict)
            assert isinstance(fit.fitted_curve, torch.Tensor)

    def test_trough_identified_as_gaussian_trough(self, synthetic_trough_profile):
        fits = fit_parametric_models(synthetic_trough_profile)
        identified = identify_mechanism(fits)
        assert identified == "gaussian_trough"

    def test_best_fit_r2_above_threshold(self, synthetic_trough_profile):
        fits = fit_parametric_models(synthetic_trough_profile)
        best_r2 = max(f.r_squared for f in fits)
        assert best_r2 > 0.7
