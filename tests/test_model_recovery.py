"""Tests for model recovery / sensitivity analysis (Phase 5.5).

Covers:
    - Synthetic response generation (shapes, conditions)
    - Suppression profile computation (symmetric averaging, shape invariants)
    - Parametric model fitting (correct identification for all 3 mechanisms)
    - Observation model (voxel pooling, noise injection, trial counts)
    - MVPA classification (above chance at high SNR)
    - Full recovery pipeline (hard gate: all mechanisms correctly identified)
"""

from __future__ import annotations

import pytest
import torch

from src.config import MechanismType
from src.analysis.model_recovery import (
    generate_synthetic_responses,
    compute_suppression_profile,
    fit_parametric_models,
    identify_mechanism,
    make_observation_model,
    mvpa_classify,
    run_recovery,
    MECHANISM_TO_FIT_MODEL,
    SyntheticResponses,
    SuppressionProfile,
    FitResult,
    RecoveryResult,
)


# ---------------------------------------------------------------------------
# Synthetic response generation
# ---------------------------------------------------------------------------

class TestSyntheticResponses:
    """Tests for generate_synthetic_responses."""

    @pytest.fixture(scope="class")
    def responses_dampening(self):
        return generate_synthetic_responses(MechanismType.DAMPENING, seed=42)

    def test_returns_correct_type(self, responses_dampening):
        assert isinstance(responses_dampening, SyntheticResponses)

    def test_shapes_match(self, responses_dampening):
        r = responses_dampening
        N = 36
        assert r.expected.shape == (N,)
        assert r.unexpected.shape == (N,)
        assert r.neutral.shape == (N,)
        assert r.pref_oris.shape == (N,)

    def test_finite_responses(self, responses_dampening):
        """L2/3 responses should be finite (no NaN/Inf)."""
        r = responses_dampening
        assert torch.isfinite(r.expected).all()
        assert torch.isfinite(r.unexpected).all()
        assert torch.isfinite(r.neutral).all()

    def test_nonzero_responses(self, responses_dampening):
        """Responses should not be all-zero — network should be active."""
        r = responses_dampening
        assert r.expected.abs().sum() > 0
        assert r.neutral.abs().sum() > 0

    def test_expected_and_neutral_differ(self, responses_dampening):
        """With prediction active, expected condition should differ from neutral."""
        r = responses_dampening
        diff = (r.expected - r.neutral).abs().sum()
        assert diff > 1e-4

    def test_mechanism_stored(self, responses_dampening):
        assert responses_dampening.mechanism == MechanismType.DAMPENING

    def test_orientations_stored(self, responses_dampening):
        r = responses_dampening
        assert r.expected_ori == 45.0
        assert r.unexpected_ori == 90.0

    def test_all_mechanisms_generate(self):
        """All three mechanisms should produce valid responses."""
        for mech in [MechanismType.DAMPENING, MechanismType.SHARPENING, MechanismType.CENTER_SURROUND]:
            r = generate_synthetic_responses(mech, seed=99)
            assert r.expected.shape == (36,)
            assert r.expected.abs().sum() > 0


# ---------------------------------------------------------------------------
# Suppression profile
# ---------------------------------------------------------------------------

class TestSuppressionProfile:
    """Tests for compute_suppression_profile."""

    @pytest.fixture(scope="class")
    def profile_dampening(self):
        r = generate_synthetic_responses(MechanismType.DAMPENING, seed=42)
        return compute_suppression_profile(r)

    def test_returns_correct_type(self, profile_dampening):
        assert isinstance(profile_dampening, SuppressionProfile)

    def test_unique_bins(self, profile_dampening):
        """Symmetric averaging should produce <= N/2 + 1 unique bins."""
        p = profile_dampening
        assert len(p.delta_theta) <= 19  # 36/2 + 1

    def test_bins_sorted(self, profile_dampening):
        """Delta-theta bins should be sorted ascending."""
        p = profile_dampening
        diffs = p.delta_theta[1:] - p.delta_theta[:-1]
        assert (diffs > 0).all()

    def test_first_bin_is_zero(self, profile_dampening):
        """First bin should be at distance = 0°."""
        assert profile_dampening.delta_theta[0].item() == pytest.approx(0.0, abs=0.5)

    def test_suppression_at_center_negative_for_dampening(self, profile_dampening):
        """Dampening should suppress at the expected orientation (negative suppression)."""
        p = profile_dampening
        assert p.suppression[0].item() < 0

    def test_raw_responses_stored(self, profile_dampening):
        p = profile_dampening
        assert len(p.raw_expected) == len(p.delta_theta)
        assert len(p.raw_neutral) == len(p.delta_theta)

    def test_suppression_is_expected_minus_neutral(self, profile_dampening):
        """Suppression should be expected - neutral."""
        p = profile_dampening
        computed = p.raw_expected - p.raw_neutral
        assert torch.allclose(p.suppression, computed, atol=1e-6)


# ---------------------------------------------------------------------------
# Parametric model fitting
# ---------------------------------------------------------------------------

class TestParametricFitting:
    """Tests for fit_parametric_models and identify_mechanism."""

    @pytest.fixture(scope="class")
    def all_fits(self):
        """Fit parametric models for all 3 mechanisms."""
        fits = {}
        for mech in [MechanismType.DAMPENING, MechanismType.SHARPENING, MechanismType.CENTER_SURROUND]:
            r = generate_synthetic_responses(mech, seed=42)
            p = compute_suppression_profile(r)
            fits[mech] = fit_parametric_models(p)
        return fits

    def test_returns_three_fits(self, all_fits):
        for mech, fit_list in all_fits.items():
            assert len(fit_list) == 3

    def test_fit_result_fields(self, all_fits):
        for fit_list in all_fits.values():
            for fit in fit_list:
                assert isinstance(fit, FitResult)
                assert isinstance(fit.model_name, str)
                assert isinstance(fit.r_squared, float)
                assert isinstance(fit.params, dict)
                assert isinstance(fit.fitted_curve, torch.Tensor)

    def test_dampening_identified(self, all_fits):
        """Dampening should be identified as gaussian_trough."""
        identified = identify_mechanism(all_fits[MechanismType.DAMPENING])
        assert identified == "gaussian_trough"

    def test_sharpening_identified(self, all_fits):
        """Sharpening should be identified as mexican_hat."""
        identified = identify_mechanism(all_fits[MechanismType.SHARPENING])
        assert identified == "mexican_hat"

    def test_center_surround_identified(self, all_fits):
        """Center-surround should be identified as offset_gaussian."""
        identified = identify_mechanism(all_fits[MechanismType.CENTER_SURROUND])
        assert identified == "offset_gaussian"

    def test_best_fit_r2_above_threshold(self, all_fits):
        """Best-fit R² should be substantial (> 0.7) for each mechanism."""
        for mech, fit_list in all_fits.items():
            best_r2 = max(f.r_squared for f in fit_list)
            assert best_r2 > 0.7, f"{mech.value} best R² = {best_r2}"

    def test_winning_model_beats_runners_up(self, all_fits):
        """The winning model should have clearly higher R² than alternatives."""
        for mech, fit_list in all_fits.items():
            r2s = sorted([f.r_squared for f in fit_list], reverse=True)
            margin = r2s[0] - r2s[1]
            assert margin > 0.005, f"{mech.value} margin too small: {margin:.4f}"


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class TestObservationModel:
    """Tests for make_observation_model."""

    @pytest.fixture(scope="class")
    def obs_result(self):
        r = generate_synthetic_responses(MechanismType.DAMPENING, seed=42)
        return make_observation_model(r, n_voxels=8, snr=5.0, n_trials=100, seed=42)

    def test_returns_tuple_of_three(self, obs_result):
        voxel_resp, patterns, labels = obs_result
        assert patterns.ndim == 2
        assert labels.ndim == 1

    def test_voxel_count(self, obs_result):
        voxel_resp, patterns, labels = obs_result
        assert voxel_resp.n_voxels == 8
        assert patterns.shape[1] == 8

    def test_trial_count(self, obs_result):
        """100 trials × 3 conditions = 300 total trials."""
        _, patterns, labels = obs_result
        assert patterns.shape[0] == 300
        assert labels.shape[0] == 300

    def test_labels_balanced(self, obs_result):
        _, _, labels = obs_result
        for c in [0, 1, 2]:
            assert (labels == c).sum().item() == 100

    def test_noise_present(self, obs_result):
        """With SNR=5, patterns should have some variance across trials."""
        _, patterns, labels = obs_result
        expected_trials = patterns[labels == 0]
        trial_var = expected_trials.var(dim=0).mean()
        assert trial_var > 1e-8


# ---------------------------------------------------------------------------
# MVPA classification
# ---------------------------------------------------------------------------

class TestMVPA:
    """Tests for mvpa_classify."""

    @pytest.fixture(scope="class")
    def mvpa_result(self):
        r = generate_synthetic_responses(MechanismType.DAMPENING, seed=42)
        _, patterns, labels = make_observation_model(r, n_voxels=8, snr=5.0, n_trials=100, seed=42)
        return mvpa_classify(patterns, labels, seed=42)

    def test_returns_dict_with_keys(self, mvpa_result):
        assert "acc_2way" in mvpa_result
        assert "acc_3way" in mvpa_result

    def test_accuracy_in_range(self, mvpa_result):
        assert 0.0 <= mvpa_result["acc_2way"] <= 1.0
        assert 0.0 <= mvpa_result["acc_3way"] <= 1.0

    def test_2way_above_chance(self, mvpa_result):
        """At SNR=5, 2-way classification should be above chance (0.5)."""
        assert mvpa_result["acc_2way"] > 0.55

    def test_3way_above_chance(self, mvpa_result):
        """At SNR=5, 3-way classification should be above chance (0.33)."""
        assert mvpa_result["acc_3way"] > 0.40


# ---------------------------------------------------------------------------
# Full recovery pipeline (hard gate)
# ---------------------------------------------------------------------------

class TestRecoveryPipeline:
    """Full recovery pipeline: must correctly identify all 3 mechanisms."""

    @pytest.fixture(scope="class", params=[
        MechanismType.DAMPENING,
        MechanismType.SHARPENING,
        MechanismType.CENTER_SURROUND,
    ])
    def recovery_result(self, request):
        return run_recovery(request.param, seed=42)

    def test_returns_correct_type(self, recovery_result):
        assert isinstance(recovery_result, RecoveryResult)

    def test_mechanism_correctly_identified(self, recovery_result):
        """HARD GATE: each mechanism must be correctly recovered."""
        r = recovery_result
        expected = MECHANISM_TO_FIT_MODEL[r.mechanism]
        assert r.identified_mechanism == expected, (
            f"{r.mechanism.value}: expected {expected}, got {r.identified_mechanism}"
        )
        assert r.correctly_identified

    def test_voxel_results_populated(self, recovery_result):
        """Should have results for 3 voxel counts × 3 SNR levels = 9 combos."""
        assert len(recovery_result.voxel_results) == 9

    def test_high_snr_mvpa_above_chance(self, recovery_result):
        """At high SNR, MVPA should beat chance for all mechanisms."""
        high_snr_key = "vox8_snr20.0"
        if high_snr_key in recovery_result.voxel_results:
            mvpa = recovery_result.voxel_results[high_snr_key]
            assert mvpa["acc_3way"] > 0.40


# ---------------------------------------------------------------------------
# Mapping consistency
# ---------------------------------------------------------------------------

class TestMechanismMapping:
    """Tests for MECHANISM_TO_FIT_MODEL consistency."""

    def test_all_mechanisms_mapped(self):
        for mech in [MechanismType.DAMPENING, MechanismType.SHARPENING, MechanismType.CENTER_SURROUND]:
            assert mech in MECHANISM_TO_FIT_MODEL

    def test_model_names_are_valid(self):
        valid_names = {"gaussian_trough", "mexican_hat", "offset_gaussian"}
        for model_name in MECHANISM_TO_FIT_MODEL.values():
            assert model_name in valid_names

    def test_mapping_is_one_to_one(self):
        values = list(MECHANISM_TO_FIT_MODEL.values())
        assert len(values) == len(set(values)), "Mapping is not one-to-one"
