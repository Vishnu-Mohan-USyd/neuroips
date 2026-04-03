"""Tests for V2 Context Module, Feedback Mechanism, Full Network.

Includes mechanism-specific SOM pattern tests, golden trial regression tests,
gradient flow, state detachment, and emergent feedback operator tests.
"""

import math

import torch
import pytest

from src.config import ModelConfig, MechanismType
from src.utils import shifted_softplus
from src.state import initial_state, NetworkState
from src.stimulus.gratings import generate_grating
from src.model.v2_context import V2ContextModule
from src.model.feedback import FeedbackMechanism, EmergentFeedbackOperator
from src.model.network import LaminarV1V2Network


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    """Default config with fixed feedback mode (legacy tests)."""
    return ModelConfig(feedback_mode='fixed')


@pytest.fixture
def cfg_emergent():
    """Config with emergent feedback mode."""
    return ModelConfig(feedback_mode='emergent')


def _cfg_with_mechanism(mechanism: MechanismType) -> ModelConfig:
    return ModelConfig(mechanism=mechanism, feedback_mode='fixed')


def _one_hot_q(channel: int, n: int = 36) -> torch.Tensor:
    """Create a batch-1 one-hot q_pred at the given channel."""
    q = torch.zeros(1, n)
    q[0, channel] = 1.0
    return q


# ── V2ContextModule Tests (fixed mode) ──────────────────────────────────

class TestV2Context:
    def test_output_shapes(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        q_pred, pi_pred, state_logits, h_v2_new = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

        assert q_pred.shape == (B, N)
        assert pi_pred.shape == (B, 1)
        assert state_logits.shape == (B, 3)
        assert h_v2_new.shape == (B, H)

    def test_q_pred_sums_to_one(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        q_pred, _, _, _ = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

        assert torch.allclose(q_pred.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_q_pred_non_negative(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        q_pred, _, _, _ = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

        assert (q_pred >= 0).all()

    def test_pi_pred_bounded(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim

        # Run multiple random inputs to stress-test bounds
        for _ in range(5):
            r_l23 = torch.randn(B, N) * 10
            cue = torch.zeros(B, N)
            task_state = torch.zeros(B, 2)
            h_v2 = torch.randn(B, H) * 5

            _, pi_pred, _, _ = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

            assert (pi_pred >= 0).all()
            assert (pi_pred <= cfg.pi_max).all()

    def test_state_logits_are_raw(self, cfg):
        """state_logits should be raw (no softmax), so they can be negative."""
        v2 = V2ContextModule(cfg)
        B, N, H = 8, cfg.n_orientations, cfg.v2_hidden_dim
        r_l23 = torch.randn(B, N) * 5
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.randn(B, H) * 3

        _, _, state_logits, _ = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

        # Raw logits should NOT sum to 1 in general
        sums = state_logits.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones(B), atol=0.1)

    def test_gru_hidden_state_updates(self, cfg):
        """Hidden state should change after a step."""
        v2 = V2ContextModule(cfg)
        B, N, H = 2, cfg.n_orientations, cfg.v2_hidden_dim
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        _, _, _, h_v2_new = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

        assert not torch.allclose(h_v2, h_v2_new)


# ── V2ContextModule Tests (emergent mode) ────────────────────────────────

class TestV2ContextEmergent:
    def test_output_shapes(self, cfg_emergent):
        v2 = V2ContextModule(cfg_emergent)
        B, N, H = 4, cfg_emergent.n_orientations, cfg_emergent.v2_hidden_dim
        r_l4 = torch.randn(B, N).abs()
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        p_cw, pi_pred, h_v2_new = v2(r_l4, r_l23, cue, task_state, h_v2)

        assert p_cw.shape == (B, 1)
        assert pi_pred.shape == (B, 1)
        assert h_v2_new.shape == (B, H)

    def test_p_cw_bounded_0_1(self, cfg_emergent):
        """p_cw should be in [0, 1] (sigmoid output)."""
        v2 = V2ContextModule(cfg_emergent)
        B, N, H = 8, cfg_emergent.n_orientations, cfg_emergent.v2_hidden_dim

        for _ in range(5):
            r_l4 = torch.randn(B, N) * 10
            r_l23 = torch.randn(B, N) * 10
            cue = torch.zeros(B, N)
            task_state = torch.zeros(B, 2)
            h_v2 = torch.randn(B, H) * 5

            p_cw, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

            assert (p_cw >= 0).all()
            assert (p_cw <= 1).all()

    def test_initial_p_cw_near_half(self, cfg_emergent):
        """With zero inputs and zero hidden state, p_cw should be ~0.5."""
        v2 = V2ContextModule(cfg_emergent)
        B, N, H = 1, cfg_emergent.n_orientations, cfg_emergent.v2_hidden_dim

        r_l4 = torch.zeros(B, N)
        r_l23 = torch.zeros(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        p_cw, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

        # sigmoid(0) = 0.5, but GRU output may shift this slightly
        assert abs(p_cw.item() - 0.5) < 0.2


# ── FeedbackMechanism Tests (fixed mode, unchanged) ─────────────────────

class TestFeedbackMechanismBasics:
    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_som_drive_shape(self, mech):
        cfg = _cfg_with_mechanism(mech)
        fb = FeedbackMechanism(cfg)
        q_pred = torch.softmax(torch.randn(4, 36), dim=-1)
        pi_pred = torch.ones(4, 1) * 3.0

        som_drive = fb.compute_som_drive(q_pred, pi_pred)
        assert som_drive.shape == (4, 36)

    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_center_excitation_shape(self, mech):
        cfg = _cfg_with_mechanism(mech)
        fb = FeedbackMechanism(cfg)
        q_pred = torch.softmax(torch.randn(4, 36), dim=-1)
        pi_pred = torch.ones(4, 1) * 3.0

        ce = fb.compute_center_excitation(q_pred, pi_pred)
        assert ce.shape == (4, 36)

    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_error_signal_shape(self, mech):
        cfg = _cfg_with_mechanism(mech)
        fb = FeedbackMechanism(cfg)
        r_l4 = torch.randn(4, 36).abs()
        template = torch.randn(4, 36).abs()

        error = fb.compute_error_signal(r_l4, template)
        assert error.shape == (4, 36)


class TestMechanismSpecificSOM:
    """Test that SOM drive patterns match the expected mechanism behavior."""

    def _get_som_drive(self, mech: MechanismType, channel: int = 9) -> torch.Tensor:
        """Compute SOM drive for a one-hot q_pred at given channel."""
        cfg = _cfg_with_mechanism(mech)
        fb = FeedbackMechanism(cfg)
        q_pred = _one_hot_q(channel, cfg.n_orientations)
        pi_pred = torch.tensor([[3.0]])
        return fb.compute_som_drive(q_pred, pi_pred).squeeze(0)

    def test_dampening_peaks_at_expected(self):
        """Model A: SOM inhibits AT the expected orientation."""
        som = self._get_som_drive(MechanismType.DAMPENING, channel=9)
        assert som.argmax().item() == 9
        assert som[9] > som[0]

    def test_sharpening_lowest_at_expected(self):
        """Model B: SOM is LOWEST at expected orientation (DoG center-sparing)."""
        som = self._get_som_drive(MechanismType.SHARPENING, channel=9)
        assert som.argmin().item() == 9
        for offset in [3, 4, 5, 6]:
            flank_ch = (9 + offset) % 36
            assert som[flank_ch] > som[9]

    def test_center_surround_low_center_high_flanks(self):
        """Model C: SOM is low near expected, high at flanks."""
        som = self._get_som_drive(MechanismType.CENTER_SURROUND, channel=9)
        flank_channel = 15
        assert som[flank_channel] > som[9]

    def test_adaptation_only_zero_som(self):
        """Model D: SOM drive is exactly zero."""
        som = self._get_som_drive(MechanismType.ADAPTATION_ONLY, channel=9)
        assert torch.allclose(som, torch.zeros_like(som))

    def test_predictive_error_zero_som(self):
        """Model E: SOM drive is exactly zero."""
        som = self._get_som_drive(MechanismType.PREDICTIVE_ERROR, channel=9)
        assert torch.allclose(som, torch.zeros_like(som))


class TestModelCNetEffect:
    def test_center_surround_net_effect(self):
        cfg = _cfg_with_mechanism(MechanismType.CENTER_SURROUND)
        fb = FeedbackMechanism(cfg)
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])

        som = fb.compute_som_drive(q_pred, pi_pred).squeeze(0)
        ce = fb.compute_center_excitation(q_pred, pi_pred).squeeze(0)

        net_center = ce[9] - som[9]
        assert net_center > 0

        for offset in [6, 7, 8]:
            flank_ch = (9 + offset) % 36
            net_flank = ce[flank_ch] - som[flank_ch]
            assert net_flank < 0


class TestCenterExcitation:
    def test_only_model_c_has_center_excitation(self):
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])

        for mech in MechanismType:
            cfg = _cfg_with_mechanism(mech)
            fb = FeedbackMechanism(cfg)
            ce = fb.compute_center_excitation(q_pred, pi_pred)

            if mech == MechanismType.CENTER_SURROUND:
                assert ce.abs().sum() > 0
            else:
                assert torch.allclose(ce, torch.zeros_like(ce))


class TestErrorSignal:
    def test_model_e_suppresses_predicted(self):
        cfg = _cfg_with_mechanism(MechanismType.PREDICTIVE_ERROR)
        fb = FeedbackMechanism(cfg)

        r_l4 = torch.ones(1, 36)
        template = torch.zeros(1, 36)
        template[0, 9] = 3.0

        error = fb.compute_error_signal(r_l4, template)
        assert error[0, 9] < 0.0
        assert error[0, 0] > 0.5

    def test_non_e_mechanisms_pass_through_l4(self):
        r_l4 = torch.randn(4, 36).abs()
        template = torch.randn(4, 36).abs()

        for mech in [MechanismType.DAMPENING, MechanismType.SHARPENING,
                     MechanismType.CENTER_SURROUND, MechanismType.ADAPTATION_ONLY]:
            cfg = _cfg_with_mechanism(mech)
            fb = FeedbackMechanism(cfg)
            result = fb.compute_error_signal(r_l4, template)
            assert torch.allclose(result, r_l4)


class TestWidthConstraints:
    def test_dampening_width_clamped_max_15(self):
        cfg = _cfg_with_mechanism(MechanismType.DAMPENING)
        fb = FeedbackMechanism(cfg)
        with torch.no_grad():
            fb.surround_width_raw.fill_(10.0)
        assert fb.surround_width.item() <= 15.0 + 1e-5

    def test_sharpening_broad_exceeds_narrow_by_10(self):
        cfg = _cfg_with_mechanism(MechanismType.SHARPENING)
        fb = FeedbackMechanism(cfg)
        with torch.no_grad():
            fb.surround_width_raw.fill_(-10.0)
        narrow = fb.center_width.item()
        broad = fb.surround_width.item()
        assert broad >= narrow + 10.0 - 1e-5


# ── EmergentFeedbackOperator Tests ──────────────────────────────────────

class TestEmergentFeedbackOperator:
    def test_output_shapes(self, cfg_emergent):
        fb = EmergentFeedbackOperator(cfg_emergent)
        B, N = 4, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0

        som_drive, center_exc = fb(q_pred, pi_eff)

        assert som_drive.shape == (B, N)
        assert center_exc.shape == (B, N)

    def test_zero_init_means_zero_output(self, cfg_emergent):
        """When alpha_inh and alpha_exc are zero, output should be zero."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        B, N = 2, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0

        som_drive, center_exc = fb(q_pred, pi_eff)

        assert torch.allclose(som_drive, torch.zeros_like(som_drive), atol=1e-6)
        assert torch.allclose(center_exc, torch.zeros_like(center_exc), atol=1e-6)

    def test_non_negative_outputs(self, cfg_emergent):
        """SOM drive and center_exc should always be >= 0 (ReLU)."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        # Set random weights
        with torch.no_grad():
            fb.alpha_inh.normal_()
            fb.alpha_exc.normal_()

        B, N = 8, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0

        som_drive, center_exc = fb(q_pred, pi_eff)

        assert (som_drive >= -1e-6).all()
        assert (center_exc >= -1e-6).all()

    def test_basis_shape(self, cfg_emergent):
        """Basis should be [K, N] with K ~ 7."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        K, N = fb.basis.shape
        assert N == cfg_emergent.n_orientations
        assert K == 7  # 4 Gaussians + 1 Mexican hat + 1 constant + 1 odd

    def test_profiles_shape(self, cfg_emergent):
        fb = EmergentFeedbackOperator(cfg_emergent)
        K_inh, K_exc = fb.get_profiles()
        assert K_inh.shape == (cfg_emergent.n_orientations,)
        assert K_exc.shape == (cfg_emergent.n_orientations,)

    def test_kernel_caching(self, cfg_emergent):
        """Cached and uncached should give same results."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        with torch.no_grad():
            fb.alpha_inh.normal_(std=0.5)
            fb.alpha_exc.normal_(std=0.5)

        B, N = 4, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0

        # Without cache
        som1, exc1 = fb(q_pred, pi_eff)

        # With cache
        fb.cache_kernels()
        som2, exc2 = fb(q_pred, pi_eff)
        fb.uncache_kernels()

        assert torch.allclose(som1, som2, atol=1e-6)
        assert torch.allclose(exc1, exc2, atol=1e-6)

    def test_manual_dampening_profile(self, cfg_emergent):
        """Setting alpha_inh to mimic narrow Gaussian should produce dampening-like profile."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        # Basis[0] is narrow (sigma=5) Gaussian
        with torch.no_grad():
            fb.alpha_inh.zero_()
            fb.alpha_inh[0] = 1.0  # narrow Gaussian only

        K_inh, K_exc = fb.get_profiles()
        # K_inh should peak at channel 0 (the center)
        assert K_inh.argmax().item() == 0
        # K_exc should be zero
        assert torch.allclose(K_exc, torch.zeros_like(K_exc), atol=1e-6)


# ── Full Network Tests (fixed mode) ─────────────────────────────────────

class TestNetworkForward:
    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_forward_all_mechanisms_no_error(self, mech):
        """Forward pass for all 5 mechanisms (fixed mode): B=4, T=20, no errors."""
        cfg = _cfg_with_mechanism(mech)
        net = LaminarV1V2Network(cfg)
        B, T, N = 4, 20, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, final_state, aux = net(stim)

        assert r_l23_all.shape == (B, T, N)
        assert final_state.r_l4.shape == (B, N)
        assert final_state.r_l23.shape == (B, N)
        assert final_state.r_pv.shape == (B, 1)
        assert final_state.r_som.shape == (B, N)
        assert final_state.h_v2.shape == (B, cfg.v2_hidden_dim)
        assert aux["q_pred_all"].shape == (B, T, N)
        assert aux["pi_pred_all"].shape == (B, T, 1)
        assert aux["state_logits_all"].shape == (B, T, 3)
        assert aux["deep_template_all"].shape == (B, T, N)
        assert aux["p_cw_all"].shape == (B, T, 1)

    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_forward_no_nan(self, mech):
        """No NaN in any output after 20 steps (fixed mode)."""
        cfg = _cfg_with_mechanism(mech)
        net = LaminarV1V2Network(cfg)
        B, T, N = 4, 20, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, final_state, aux = net(stim)

        assert not torch.isnan(r_l23_all).any()
        for field in final_state:
            assert not torch.isnan(field).any()
        for v in aux.values():
            assert not torch.isnan(v).any()

    def test_step_returns_state_and_aux(self, cfg):
        """step() should return (NetworkState, StepAux)."""
        from src.state import StepAux
        net = LaminarV1V2Network(cfg)
        B, N = 2, cfg.n_orientations
        state = initial_state(B, N, cfg.v2_hidden_dim)
        stim = torch.randn(B, N).abs()
        cue = torch.zeros(B, N)
        task = torch.zeros(B, 2)

        result = net.step(stim, cue, task, state)
        assert isinstance(result, tuple) and len(result) == 2
        new_state, aux = result
        assert isinstance(new_state, NetworkState)
        assert isinstance(aux, StepAux)
        assert aux.q_pred.shape == (B, N)
        assert aux.pi_pred.shape == (B, 1)
        assert aux.state_logits.shape == (B, 3)
        assert aux.p_cw.shape == (B, 1)

    def test_default_cue_and_task_state(self, cfg):
        """forward() should work with cue_seq=None and task_state_seq=None."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        r_l23_all, _, _ = net(stim)
        assert r_l23_all.shape == (B, T, N)


# ── Full Network Tests (emergent mode) ──────────────────────────────────

class TestNetworkEmergent:
    def test_forward_emergent_no_error(self, cfg_emergent):
        """Forward pass in emergent mode: B=4, T=20, no errors."""
        net = LaminarV1V2Network(cfg_emergent)
        B, T, N = 4, 20, cfg_emergent.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, final_state, aux = net(stim)

        assert r_l23_all.shape == (B, T, N)
        assert aux["q_pred_all"].shape == (B, T, N)
        assert aux["pi_pred_all"].shape == (B, T, 1)
        assert aux["p_cw_all"].shape == (B, T, 1)

    def test_forward_emergent_no_nan(self, cfg_emergent):
        """No NaN in emergent mode outputs."""
        net = LaminarV1V2Network(cfg_emergent)
        B, T, N = 4, 20, cfg_emergent.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, final_state, aux = net(stim)

        assert not torch.isnan(r_l23_all).any()
        for field in final_state:
            assert not torch.isnan(field).any()
        for v in aux.values():
            assert not torch.isnan(v).any()

    def test_q_pred_construction_cw(self, cfg_emergent):
        """With p_cw=1.0 and theta=45, q_pred should peak at 60 deg."""
        net = LaminarV1V2Network(cfg_emergent)
        N = cfg_emergent.n_orientations

        # Create L4 activity peaked at 45 deg (channel 9)
        theta = torch.tensor([45.0])
        r_l4 = generate_grating(theta, torch.tensor([0.8]), N)  # [1, N]

        p_cw = torch.tensor([[1.0]])
        q_pred = net._construct_q_pred(r_l4, p_cw)

        # Should peak at 60 deg = channel 12
        assert q_pred.argmax(dim=-1).item() == 12

    def test_q_pred_construction_ccw(self, cfg_emergent):
        """With p_cw=0.0 and theta=45, q_pred should peak at 30 deg."""
        net = LaminarV1V2Network(cfg_emergent)
        N = cfg_emergent.n_orientations

        theta = torch.tensor([45.0])
        r_l4 = generate_grating(theta, torch.tensor([0.8]), N)

        p_cw = torch.tensor([[0.0]])
        q_pred = net._construct_q_pred(r_l4, p_cw)

        # Should peak at 30 deg = channel 6
        assert q_pred.argmax(dim=-1).item() == 6

    def test_q_pred_construction_uncertain(self, cfg_emergent):
        """With p_cw=0.5, q_pred should be bimodal."""
        net = LaminarV1V2Network(cfg_emergent)
        N = cfg_emergent.n_orientations

        theta = torch.tensor([45.0])
        r_l4 = generate_grating(theta, torch.tensor([0.8]), N)

        p_cw = torch.tensor([[0.5]])
        q_pred = net._construct_q_pred(r_l4, p_cw)

        # Should have significant mass at both 30 and 60 deg (channels 6 and 12)
        assert q_pred[0, 6] > 0.01
        assert q_pred[0, 12] > 0.01

    def test_q_pred_normalized(self, cfg_emergent):
        """Constructed q_pred should sum to ~1."""
        net = LaminarV1V2Network(cfg_emergent)
        N = cfg_emergent.n_orientations

        theta = torch.tensor([90.0])
        r_l4 = generate_grating(theta, torch.tensor([0.8]), N)

        for p in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p_cw = torch.tensor([[p]])
            q_pred = net._construct_q_pred(r_l4, p_cw)
            assert torch.allclose(q_pred.sum(), torch.tensor(1.0), atol=1e-5)

    def test_zero_feedback_at_init(self, cfg_emergent):
        """Emergent feedback should be zero at initialization (alpha=0)."""
        net = LaminarV1V2Network(cfg_emergent)
        B, T, N = 2, 5, cfg_emergent.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        # SOM should be ~0 since feedback operator outputs zero
        # (alpha_inh and alpha_exc are initialized to zero)
        # But SOM has dynamics, so check it's very small
        assert aux["r_som_all"].abs().max() < 0.1


# ── Gradient Flow Tests ──────────────────────────────────────────────────

class TestGradientFlowNetwork:
    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_gradients_through_full_network(self, mech):
        """Forward 5 steps, backward, verify all trainable params have gradients (fixed mode)."""
        cfg = _cfg_with_mechanism(mech)
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 5, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        loss = (r_l23_all.sum()
                + aux["state_logits_all"].sum()
                + aux["deep_template_all"].sum()
                + aux["q_pred_all"].sum()
                + aux["pi_pred_all"].sum())
        loss.backward()

        unused_params = {"l4.pv_gain.gain_raw"}
        for name, param in net.named_parameters():
            if param.requires_grad and name not in unused_params:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradients_emergent_mode(self, cfg_emergent):
        """Gradient flow through emergent feedback operator."""
        net = LaminarV1V2Network(cfg_emergent)
        B, T, N = 2, 5, cfg_emergent.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        loss = (r_l23_all.sum()
                + aux["deep_template_all"].sum()
                + aux["q_pred_all"].sum()
                + aux["pi_pred_all"].sum()
                + aux["p_cw_all"].sum())
        loss.backward()

        # Check emergent-specific params
        assert net.feedback.alpha_inh.grad is not None
        assert net.feedback.alpha_exc.grad is not None
        # Check V2 p_cw head
        assert net.v2.head_p_cw.weight.grad is not None

        unused_params = {"l4.pv_gain.gain_raw"}
        for name, param in net.named_parameters():
            if param.requires_grad and name not in unused_params:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestStateDetachment:
    def test_no_gradient_leakage_between_trials(self, cfg):
        """Gradients from trial 2 should NOT flow back through trial 1."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 5, cfg.n_orientations

        stim1 = torch.randn(B, T, N).abs() * 0.5
        _, state1, _ = net(stim1)

        state_detached = NetworkState(*[s.detach() for s in state1])

        stim2 = torch.randn(B, T, N).abs() * 0.5
        r_l23_2, _, _ = net(stim2, state=state_detached)

        loss = r_l23_2.sum()
        loss.backward()

        assert stim1.grad is None or torch.allclose(stim1.grad, torch.zeros_like(stim1))


# ── Golden Trial Tests (fixed mode, unchanged) ──────────────────────────

class TestGoldenTrials:
    def _golden_som_drive(self, mech: MechanismType) -> torch.Tensor:
        torch.manual_seed(42)
        cfg = _cfg_with_mechanism(mech)
        fb = FeedbackMechanism(cfg)
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])
        return fb.compute_som_drive(q_pred, pi_pred)

    def test_golden_dampening_som(self):
        som = self._golden_som_drive(MechanismType.DAMPENING)
        som2 = self._golden_som_drive(MechanismType.DAMPENING)
        assert torch.allclose(som, som2, atol=1e-6)

    def test_golden_sharpening_som(self):
        som = self._golden_som_drive(MechanismType.SHARPENING)
        som2 = self._golden_som_drive(MechanismType.SHARPENING)
        assert torch.allclose(som, som2, atol=1e-6)
        assert som[0].argmin().item() == 9

    def test_golden_center_surround_som(self):
        som = self._golden_som_drive(MechanismType.CENTER_SURROUND)
        som2 = self._golden_som_drive(MechanismType.CENTER_SURROUND)
        assert torch.allclose(som, som2, atol=1e-6)

    def test_golden_full_network_dampening(self):
        """Golden trial: full network, dampening, 10 steps, fixed seed."""
        torch.manual_seed(42)
        cfg = _cfg_with_mechanism(MechanismType.DAMPENING)
        net = LaminarV1V2Network(cfg)
        B, T, N = 1, 10, cfg.n_orientations

        ori = torch.tensor([45.0])
        contrast = torch.tensor([0.8])
        stim_single = generate_grating(ori, contrast, N)
        stim_seq = stim_single.unsqueeze(1).expand(B, T, N)

        r_l23_all, final_state, aux = net(stim_seq)

        torch.manual_seed(42)
        net2 = LaminarV1V2Network(cfg)
        r_l23_all2, _, _ = net2(stim_seq)
        assert torch.allclose(r_l23_all, r_l23_all2, atol=1e-6)
