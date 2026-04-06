"""Tests for the analysis suite (Phase 7).

Covers all 13 analysis modules plus plotting.
Uses a shared fixture with a small network and minimal experiment runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from src.config import ModelConfig, MechanismType
from src.model.network import LaminarV1V2Network
from src.experiments.hidden_state import HiddenStateParadigm
from src.experiments.omission import OmissionParadigm
from src.experiments.ambiguous import AmbiguousParadigm
from src.experiments.paradigm_base import ExperimentResult

from src.analysis.suppression_profile import (
    compute_mean_responses, compute_suppression_profile_from_experiment,
    MeanResponseResult, SuppressionProfileResult,
)
from src.analysis.tuning_curves import (
    fit_von_mises, analyse_tuning_curves, TuningFit, TuningAnalysisResult,
)
from src.analysis.decoding import (
    nearest_centroid_decode, cross_validated_decoding,
    compute_d_prime, compute_fisher_information,
)
from src.analysis.rsa import compute_rdm, kendall_tau, run_rsa, RSAResult
from src.analysis.energy import compute_energy, compute_pareto_frontier, EnergyResult, ParetoPoint
from src.analysis.observation_model import (
    pool_to_voxels, run_observation_model, ObservationModelResult,
)
from src.analysis.omission_analysis import (
    run_omission_analysis, TemplateDecodingResult,
    decode_orientation_from_template, compute_template_fidelity,
)
from src.analysis.bias_analysis import (
    population_vector_decode, compute_bias, run_bias_analysis, BiasResult,
)
from src.analysis.temporal_analysis import (
    run_temporal_analysis, TemporalAnalysisResult,
)
from src.analysis.v2_probes import (
    compute_q_pred_entropy, run_v2_probes, V2ProbeResult,
)
from src.analysis.ablations import run_ablation, AblationResult


def _load_analyze_representation():
    """Load the analysis script as a module for helper-level testing."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_representation.py"
    spec = importlib.util.spec_from_file_location("analyze_representation_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def net_and_cfg():
    torch.manual_seed(0)
    cfg = ModelConfig(mechanism=MechanismType.DAMPENING, n_orientations=36, feedback_mode='fixed')
    net = LaminarV1V2Network(cfg)
    net.eval()
    return net, cfg


@pytest.fixture(scope="module")
def p1_result(net_and_cfg):
    net, cfg = net_and_cfg
    p = HiddenStateParadigm(net, cfg, probe_deviations=[0.0, 45.0, 90.0])
    return p.run(n_trials=3, seed=42, batch_size=4)


@pytest.fixture(scope="module")
def p2_result(net_and_cfg):
    net, cfg = net_and_cfg
    p = OmissionParadigm(net, cfg)
    return p.run(n_trials=3, seed=42, batch_size=4)


@pytest.fixture(scope="module")
def p3_result(net_and_cfg):
    net, cfg = net_and_cfg
    p = AmbiguousParadigm(net, cfg)
    return p.run(n_trials=3, seed=42, batch_size=4)


@pytest.fixture(scope="module")
def analyze_script():
    return _load_analyze_representation()


@pytest.fixture(scope="module")
def emergent_vip_net():
    torch.manual_seed(0)
    cfg = ModelConfig(
        mechanism=MechanismType.CENTER_SURROUND,
        n_orientations=36,
        feedback_mode='emergent',
        vip_enabled=True,
        vip_gain=2.0,
    )
    net = LaminarV1V2Network(cfg)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Analysis 1+2: Mean responses + Suppression profiles
# ---------------------------------------------------------------------------

class TestMeanResponses:
    def test_returns_correct_type(self, p1_result):
        result = compute_mean_responses(p1_result)
        assert isinstance(result, MeanResponseResult)

    def test_all_conditions_present(self, p1_result):
        result = compute_mean_responses(p1_result)
        assert len(result.condition_means) == len(p1_result.conditions)

    def test_populations_listed(self, p1_result):
        result = compute_mean_responses(p1_result)
        assert "r_l23" in result.populations


class TestSuppressionProfile:
    def test_returns_correct_type(self, p1_result):
        result = compute_suppression_profile_from_experiment(p1_result, expected_ori=45.0)
        assert isinstance(result, SuppressionProfileResult)

    def test_has_all_fields(self, p1_result):
        result = compute_suppression_profile_from_experiment(p1_result, expected_ori=45.0)
        assert result.delta_theta is not None
        assert result.suppression is not None
        assert result.surprise is not None
        assert result.difference is not None

    def test_bins_sorted(self, p1_result):
        result = compute_suppression_profile_from_experiment(p1_result, expected_ori=45.0)
        diffs = result.delta_theta[1:] - result.delta_theta[:-1]
        assert (diffs > 0).all()


# ---------------------------------------------------------------------------
# Analysis 3: Tuning curves
# ---------------------------------------------------------------------------

class TestTuningCurves:
    def test_fit_von_mises(self):
        oris = torch.arange(36, dtype=torch.float32) * 5
        # Gaussian-like tuning curve peaking at 45 degrees
        dists = ((oris - 45.0 + 90) % 180 - 90).abs()
        resp = torch.exp(-dists ** 2 / (2 * 12.0 ** 2))
        fit = fit_von_mises(resp, oris)
        assert isinstance(fit, TuningFit)
        assert fit.r_squared > 0.5
        assert abs(fit.preferred_ori - 45.0) < 10.0

    def test_analyse_tuning_curves(self):
        oris = torch.arange(36, dtype=torch.float32) * 5
        dists = ((oris - 45.0 + 90) % 180 - 90).abs()
        resp_exp = torch.exp(-dists ** 2 / (2 * 12.0 ** 2))
        resp_neut = torch.exp(-dists ** 2 / (2 * 15.0 ** 2))
        responses = {"expected": resp_exp.unsqueeze(0), "neutral": resp_neut.unsqueeze(0)}
        result = analyse_tuning_curves(responses)
        assert isinstance(result, TuningAnalysisResult)
        assert "expected" in result.fits
        assert len(result.fits["expected"]) == 1


# ---------------------------------------------------------------------------
# Analysis 4: Decoding
# ---------------------------------------------------------------------------

class TestDecoding:
    def test_nearest_centroid(self):
        train_X = torch.randn(20, 5)
        train_y = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        train_X[:10] += 3  # separate clusters
        acc = nearest_centroid_decode(train_X, train_y, train_X, train_y)
        assert acc > 0.8

    def test_cross_validated(self):
        X = torch.randn(40, 5)
        y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        X[:20] += 3
        acc = cross_validated_decoding(X, y)
        assert acc > 0.6

    def test_d_prime(self):
        a = torch.randn(50, 10) + 2
        b = torch.randn(50, 10) - 2
        dp = compute_d_prime(a, b)
        assert dp > 1.0

    def test_fisher_information(self):
        oris = torch.arange(36, dtype=torch.float32) * 5
        # Simple tuning curves for 10 units
        responses = torch.stack([
            torch.exp(-((oris - i * 18.0 + 90) % 180 - 90).pow(2) / (2 * 12.0 ** 2))
            for i in range(10)
        ], dim=1)  # [36, 10]
        fisher = compute_fisher_information(responses, oris)
        assert fisher.shape == (36,)
        assert (fisher >= 0).all()


# ---------------------------------------------------------------------------
# Analysis 5: RSA
# ---------------------------------------------------------------------------

class TestRSA:
    def test_compute_rdm(self):
        resp = torch.randn(10, 36)
        rdm = compute_rdm(resp)
        assert rdm.shape == (10, 10)
        assert rdm.diag().abs().sum() < 1e-6  # diagonal is zero

    def test_kendall_tau_self(self):
        rdm = torch.randn(10, 10)
        rdm = (rdm + rdm.T) / 2
        tau = kendall_tau(rdm, rdm)
        assert abs(tau - 1.0) < 0.01

    def test_run_rsa(self):
        data = {"cond_a": torch.randn(10, 36), "cond_b": torch.randn(10, 36)}
        result = run_rsa(data)
        assert isinstance(result, RSAResult)
        assert len(result.rdms) == 2
        assert len(result.kendall_tau) == 1


# ---------------------------------------------------------------------------
# Analysis 7: Energy
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_compute_energy(self, p1_result):
        result = compute_energy(p1_result)
        assert isinstance(result, EnergyResult)
        assert result.total_activity >= 0

    def test_pareto_frontier(self):
        pts = [
            ParetoPoint(0.01, 0.9, 10.0, "A"),
            ParetoPoint(0.01, 0.8, 5.0, "A"),
            ParetoPoint(0.01, 0.95, 20.0, "B"),
            ParetoPoint(0.01, 0.7, 3.0, "B"),
        ]
        frontier = compute_pareto_frontier(pts)
        assert len(frontier) >= 2
        # Frontier should be monotonically increasing in both
        for i in range(len(frontier) - 1):
            assert frontier[i].energy <= frontier[i + 1].energy
            assert frontier[i].accuracy <= frontier[i + 1].accuracy


# ---------------------------------------------------------------------------
# Analysis 9: Observation model
# ---------------------------------------------------------------------------

class TestObservationModel:
    def test_pool_to_voxels(self):
        r = torch.randn(10, 36)
        v = pool_to_voxels(r, n_voxels=4)
        assert v.shape == (10, 4)

    def test_run_observation_model(self):
        data = {
            "expected": torch.randn(20, 36) + 1,
            "unexpected": torch.randn(20, 36) - 1,
            "neutral": torch.randn(20, 36),
        }
        result = run_observation_model(data, n_voxels=4, snr=5.0)
        assert isinstance(result, ObservationModelResult)
        assert 0 <= result.mvpa_accuracy_3way <= 1.0


# ---------------------------------------------------------------------------
# Analysis 6: Omission + template decoding
# ---------------------------------------------------------------------------

class TestOmissionAnalysis:
    def test_template_fidelity(self):
        template = torch.zeros(36)
        template[9] = 1.0  # peak at channel 9 = 45 degrees
        fid = compute_template_fidelity(template, 9)
        assert fid > 0.4

    def test_run_omission_analysis(self, p2_result):
        result = run_omission_analysis(p2_result)
        assert isinstance(result, TemplateDecodingResult)


# ---------------------------------------------------------------------------
# Analysis 8: Bias
# ---------------------------------------------------------------------------

class TestBiasAnalysis:
    def test_population_vector_decode(self):
        # Response peaking at channel 9 (45 degrees)
        resp = torch.zeros(3, 36)
        resp[:, 9] = 1.0
        resp[:, 8] = 0.5
        resp[:, 10] = 0.5
        decoded = population_vector_decode(resp)
        assert decoded.shape == (3,)

    def test_run_bias_analysis(self, p3_result):
        result = run_bias_analysis(p3_result, expected_ori=45.0)
        assert isinstance(result, BiasResult)


# ---------------------------------------------------------------------------
# Analysis 11: Temporal
# ---------------------------------------------------------------------------

class TestTemporalAnalysis:
    def test_run_temporal_analysis(self, p1_result):
        result = run_temporal_analysis(p1_result)
        assert isinstance(result, TemporalAnalysisResult)
        assert len(result.windows) == len(p1_result.temporal_windows)

    def test_time_course_has_all_conditions(self, p1_result):
        result = run_temporal_analysis(p1_result)
        assert len(result.time_course) == len(p1_result.conditions)

    def test_time_course_shape(self, p1_result):
        result = run_temporal_analysis(p1_result)
        cond = next(iter(result.time_course.values()))
        T = next(iter(p1_result.conditions.values())).r_l23.shape[1]
        for layer_tc in cond.values():
            assert layer_tc.shape == (T,)


# ---------------------------------------------------------------------------
# Analysis 12: V2 probes
# ---------------------------------------------------------------------------

class TestV2Probes:
    def test_q_pred_entropy(self):
        q = torch.ones(5, 10, 36) / 36  # uniform -> max entropy
        ent = compute_q_pred_entropy(q)
        assert ent.shape == (5, 10)
        assert ent.mean() > 3.0  # ln(36) ~ 3.58

    def test_run_v2_probes(self, p1_result):
        result = run_v2_probes(p1_result)
        assert isinstance(result, V2ProbeResult)
        assert len(result.q_pred_entropy) == len(p1_result.conditions)
        assert len(result.pi_pred_mean) == len(p1_result.conditions)


# ---------------------------------------------------------------------------
# Analysis 13: Ablations
# ---------------------------------------------------------------------------

class TestAblations:
    def test_run_single_ablation(self, net_and_cfg):
        net, cfg = net_and_cfg
        result = run_ablation(net, cfg, "zero_som", n_trials=2,
                              probe_deviations=[0.0, 45.0])
        assert isinstance(result, AblationResult)
        assert result.ablation_name == "zero_som"
        assert isinstance(result.experiment_result, ExperimentResult)

    def test_ablation_restores_params(self, net_and_cfg):
        """After ablation, model parameters should be restored."""
        net, cfg = net_and_cfg
        # Record original SOM params
        orig_params = {n: p.clone() for n, p in net.som.named_parameters()}
        run_ablation(net, cfg, "zero_som", n_trials=2, probe_deviations=[0.0])
        # Verify restored
        for n, p in net.som.named_parameters():
            assert torch.allclose(p, orig_params[n]), f"SOM param {n} not restored"

    def test_clamp_pi_ablation(self, net_and_cfg):
        net, cfg = net_and_cfg
        result = run_ablation(net, cfg, "clamp_pi", n_trials=2,
                              probe_deviations=[0.0])
        assert result.ablation_name == "clamp_pi"


# ---------------------------------------------------------------------------
# Script-level analysis helpers: cue-aware Option B support
# ---------------------------------------------------------------------------

class TestAnalyzeRepresentationCueSupport:
    def test_feedback_disabled_zeroes_all_top_down_branch_gains(self, analyze_script):
        cfg = ModelConfig(
            mechanism=MechanismType.CENTER_SURROUND,
            n_orientations=36,
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_center_support_enabled=True,
            emergent_center_support_gain=0.12,
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_beta=0.15,
            apical_gain_enabled=True,
            apical_gain_beta=0.08,
        )
        net = LaminarV1V2Network(cfg)
        fb = net.feedback

        saved = (
            fb.alpha_inh.detach().clone(),
            fb.som_baseline.detach().clone() if hasattr(fb, "som_baseline") else None,
            fb.center_support_gain,
            fb.recurrent_gain_beta,
            fb.apical_gain_beta,
            net.cfg.vip_gain,
        )
        with analyze_script.feedback_disabled(net):
            assert torch.allclose(fb.alpha_inh, torch.zeros_like(fb.alpha_inh))
            if hasattr(fb, "som_baseline"):
                assert torch.allclose(fb.som_baseline, torch.zeros_like(fb.som_baseline))
            assert fb.center_support_gain == 0.0
            assert fb.recurrent_gain_beta == 0.0
            assert fb.apical_gain_beta == 0.0
            assert net.cfg.vip_gain == 0.0

        assert torch.allclose(fb.alpha_inh, saved[0])
        if saved[1] is not None:
            assert torch.allclose(fb.som_baseline, saved[1])
        assert fb.center_support_gain == saved[2]
        assert fb.recurrent_gain_beta == saved[3]
        assert fb.apical_gain_beta == saved[4]
        assert net.cfg.vip_gain == saved[5]

    def test_sanity_check_ablation_reports_all_zeroed_branches(self, analyze_script):
        cfg = ModelConfig(
            mechanism=MechanismType.CENTER_SURROUND,
            n_orientations=36,
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_center_support_enabled=True,
            emergent_center_support_gain=0.12,
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_beta=0.15,
            apical_gain_enabled=True,
            apical_gain_beta=0.08,
        )
        net = LaminarV1V2Network(cfg)
        sanity = analyze_script.sanity_check_ablation(net, torch.device("cpu"))

        assert sanity["center_off_max_abs"] < 1e-7
        assert sanity["recurrent_off_max_abs"] < 1e-7
        assert sanity["apical_off_max_abs"] < 1e-7
        assert sanity["vip_gain_off"] == 0.0
        assert sanity["ablation_zero"] is True

    def test_parse_args_zero_cue_defaults(self, analyze_script, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["analyze_representation.py", "--checkpoint", "a.pt", "--config", "b.yaml"],
        )
        args = analyze_script.parse_args()
        assert args.cue_mode == "none"
        assert args.cue_contrast == pytest.approx(analyze_script.EVAL_CONTRAST)
        assert args.cue_prestimulus_steps == 0
        assert args.cue_offset == 0.0

    def test_run_trials_default_matches_explicit_zero_cue(self, analyze_script, emergent_vip_net):
        device = torch.device("cpu")
        stim = torch.tensor([90.0, 95.0], device=device)
        oracle = torch.tensor([90.0, 90.0], device=device)

        out_default = analyze_script.run_trials(emergent_vip_net, stim, oracle, device)
        out_explicit = analyze_script.run_trials(
            emergent_vip_net, stim, oracle, device,
            cue_cfg=analyze_script.CueConfig(),
        )
        assert torch.allclose(out_default, out_explicit, atol=1e-6)

    def test_metric_time_resolved_exposes_vip_when_cued(self, analyze_script, emergent_vip_net):
        device = torch.device("cpu")
        cue_cfg = analyze_script.CueConfig(mode="oracle", prestimulus_steps=4, contrast=1.0)

        uncued = analyze_script.metric_time_resolved(
            emergent_vip_net, device, oracle_theta=90.0, stim_theta=90.0,
        )
        cued = analyze_script.metric_time_resolved(
            emergent_vip_net, device, oracle_theta=90.0, stim_theta=90.0,
            cue_cfg=cue_cfg,
        )

        assert len(cued["timesteps"]) == analyze_script.T_STEPS + cue_cfg.prestimulus_steps
        assert max(uncued["vip_mean_on"]) < 1e-8
        assert max(cued["vip_mean_on"][:cue_cfg.prestimulus_steps]) > 0.0

    def test_metric_local_dprime_accepts_cue_cfg(self, analyze_script, emergent_vip_net):
        device = torch.device("cpu")
        cue_cfg = analyze_script.CueConfig(mode="oracle", prestimulus_steps=4, contrast=1.0)
        result = analyze_script.metric_local_dprime(
            emergent_vip_net, device, n_trials=2, noise_std=0.0, seed=7, cue_cfg=cue_cfg,
        )
        assert "delta_5" in result
        assert "delta_10" in result
        assert result["n_trials_per_class"] == 2


# ---------------------------------------------------------------------------
# Plotting (import check only — no display)
# ---------------------------------------------------------------------------

class TestPlotting:
    def test_imports(self):
        from src.analysis.plotting import (
            plot_suppression_profile, plot_temporal_timecourse,
            plot_pareto_frontier, plot_rdm,
        )

    def test_plot_suppression_profile(self):
        try:
            from src.analysis.plotting import plot_suppression_profile
            fig = plot_suppression_profile(
                torch.arange(10, dtype=torch.float32) * 10,
                torch.randn(10),
                surprise=torch.randn(10),
            )
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")
