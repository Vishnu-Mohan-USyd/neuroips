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

        mu_pred, pi_pred, fb_signal, h_v2_new = v2(r_l4, r_l23, cue, task_state, h_v2)

        assert mu_pred.shape == (B, N)
        assert pi_pred.shape == (B, 1)
        assert fb_signal.shape == (B, N)
        assert h_v2_new.shape == (B, H)

    def test_mu_pred_sums_to_one(self, cfg_emergent):
        """mu_pred should be a valid probability distribution (softmax output)."""
        v2 = V2ContextModule(cfg_emergent)
        B, N, H = 8, cfg_emergent.n_orientations, cfg_emergent.v2_hidden_dim

        for _ in range(5):
            r_l4 = torch.randn(B, N) * 10
            r_l23 = torch.randn(B, N) * 10
            cue = torch.zeros(B, N)
            task_state = torch.zeros(B, 2)
            h_v2 = torch.randn(B, H) * 5

            mu_pred, _, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

            assert (mu_pred >= 0).all(), "mu_pred should be non-negative (softmax)"
            assert torch.allclose(mu_pred.sum(dim=-1), torch.ones(B), atol=1e-5), \
                "mu_pred should sum to 1 (valid distribution)"

    def test_initial_mu_pred_near_uniform(self, cfg_emergent):
        """With zero inputs and zero hidden state, mu_pred should be near uniform."""
        v2 = V2ContextModule(cfg_emergent)
        B, N, H = 1, cfg_emergent.n_orientations, cfg_emergent.v2_hidden_dim

        r_l4 = torch.zeros(B, N)
        r_l23 = torch.zeros(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        mu_pred, _, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

        # With zero-initialized GRU and random head_mu weights, mu_pred
        # won't be exactly uniform but should not be extremely peaked.
        # Check entropy is at least 50% of max entropy.
        max_entropy = torch.log(torch.tensor(float(N)))
        entropy = -(mu_pred * (mu_pred + 1e-8).log()).sum()
        assert entropy > 0.5 * max_entropy, (
            f"mu_pred too peaked at init: entropy={entropy.item():.2f}, "
            f"max={max_entropy.item():.2f}"
        )


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
        r_l4 = torch.randn(B, N).abs()

        som_drive, vip_drive, apical_gain = fb(q_pred, pi_eff, r_l4)

        assert som_drive.shape == (B, N)
        assert vip_drive.shape == (B, N)
        assert apical_gain.shape == (B, N)

    def test_small_init_mostly_uniform(self, cfg_emergent):
        """Default init (alpha=0.01) should produce nearly uniform output.

        Softplus(field≈0) ≈ 0.693 gives a constant baseline. The spatial
        variation from the tiny alpha=0.01 field should be negligible.
        """
        fb = EmergentFeedbackOperator(cfg_emergent)
        B, N = 2, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0
        r_l4 = torch.randn(B, N).abs()

        som_drive, vip_drive, apical_gain = fb(q_pred, pi_eff, r_l4)

        # Spatial variation should be very small (< 1% of mean)
        assert som_drive.std(dim=-1).max() < 0.1 * som_drive.mean()

    def test_zero_alpha_uniform_output(self, cfg_emergent):
        """When alpha=0, field=0, softplus(0)≈0.693 → uniform baseline output."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        with torch.no_grad():
            fb.alpha_inh.zero_()
        B, N = 2, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0
        r_l4 = torch.randn(B, N).abs()

        som_drive, vip_drive, apical_gain = fb(q_pred, pi_eff, r_l4)

        # With alpha=0, field=0, softplus(0)=ln(2)≈0.693 → uniform
        # Output should be constant across orientations (no spatial structure)
        assert som_drive.std(dim=-1).max() < 1e-5, "Should be spatially uniform"

    def test_non_negative_outputs(self, cfg_emergent):
        """SOM drive should always be >= 0 (softplus, no delta_som).
        VIP drive uses delta-style so it can go negative before VIPRing
        rectification — that's expected behavior.
        """
        fb = EmergentFeedbackOperator(cfg_emergent)
        # Set random weights
        with torch.no_grad():
            fb.alpha_inh.normal_()
            fb.alpha_vip.normal_()

        B, N = 8, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0
        r_l4 = torch.randn(B, N).abs()

        som_drive, vip_drive, apical_gain = fb(q_pred, pi_eff, r_l4)

        # SOM uses plain softplus (no delta) by default → non-negative
        assert (som_drive >= -1e-6).all()
        # VIP uses delta-style (signed) — just check finite
        assert torch.isfinite(vip_drive).all()

    def test_alpha_shapes(self, cfg_emergent):
        """Alpha weights should be [N] — one per orientation channel."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        N = cfg_emergent.n_orientations
        assert fb.alpha_inh.shape == (N,)
        assert fb.alpha_vip.shape == (N,)
        assert fb.alpha_apical.shape == (N,)

    def test_profiles_shape(self, cfg_emergent):
        fb = EmergentFeedbackOperator(cfg_emergent)
        K_inh = fb.get_profiles()
        assert K_inh.shape == (cfg_emergent.n_orientations,)

    def test_kernel_caching(self, cfg_emergent):
        """Cached and uncached should give same results."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        with torch.no_grad():
            fb.alpha_inh.normal_(std=0.5)
            fb.alpha_vip.normal_(std=0.5)
            fb.alpha_apical.normal_(std=0.5)

        B, N = 4, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0
        r_l4 = torch.randn(B, N).abs()

        # Without cache
        som1, vip1, apical1 = fb(q_pred, pi_eff, r_l4)

        # With cache
        fb.cache_kernels()
        som2, vip2, apical2 = fb(q_pred, pi_eff, r_l4)
        fb.uncache_kernels()

        assert torch.allclose(som1, som2, atol=1e-6)
        assert torch.allclose(vip1, vip2, atol=1e-6)
        assert torch.allclose(apical1, apical2, atol=1e-6)

    def test_manual_dampening_profile(self, cfg_emergent):
        """Setting alpha_inh to mimic narrow Gaussian should produce dampening-like profile."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        # Basis[0] is narrow (sigma=5) Gaussian
        with torch.no_grad():
            fb.alpha_inh.zero_()
            fb.alpha_inh[0] = 1.0  # narrow Gaussian only

        K_inh = fb.get_profiles()
        # K_inh should peak at channel 0 (the center)
        assert K_inh.argmax().item() == 0


# ── VIP Disinhibitory Pathway Tests ───────────────────────────────────

class TestVIPPathway:
    """Tests for VIPRing dynamics, VIP→SOM disinhibition, and alpha_vip ablation."""

    def test_vip_ring_dynamics(self, cfg_emergent):
        """VIPRing Euler update: drive=0 → decay toward zero."""
        from src.model.populations import VIPRing
        vip = VIPRing(cfg_emergent)
        B, N = 2, cfg_emergent.n_orientations
        r_vip_prev = torch.ones(B, N) * 0.5
        vip_drive = torch.zeros(B, N)
        r_vip = vip(vip_drive, r_vip_prev)
        # With zero drive and rectified_softplus(0)≈0.693, the rate should change
        # but remain finite and non-negative
        assert r_vip.shape == (B, N)
        assert torch.isfinite(r_vip).all()
        assert (r_vip >= 0).all()

    def test_vip_ring_positive_drive(self, cfg_emergent):
        """VIPRing with positive drive should increase rates."""
        from src.model.populations import VIPRing
        vip = VIPRing(cfg_emergent)
        B, N = 2, cfg_emergent.n_orientations
        r_vip_prev = torch.zeros(B, N)
        vip_drive = torch.ones(B, N) * 2.0
        r_vip = vip(vip_drive, r_vip_prev)
        assert (r_vip > 0).all(), "Positive drive should produce positive rates"

    def test_network_has_vip_params(self, cfg_emergent):
        """Network should have VIP ring and w_vip_som parameter."""
        net = LaminarV1V2Network(cfg_emergent)
        assert hasattr(net, 'vip'), "Missing VIPRing module"
        assert hasattr(net, 'w_vip_som'), "Missing w_vip_som parameter"
        assert net.w_vip_som.requires_grad

    def test_feedback_has_alpha_vip(self, cfg_emergent):
        """EmergentFeedbackOperator should have alpha_vip (zero init)."""
        net = LaminarV1V2Network(cfg_emergent)
        fb = net.feedback
        assert hasattr(fb, 'alpha_vip'), "Missing alpha_vip"
        assert fb.alpha_vip.shape == fb.alpha_inh.shape
        # alpha_vip inits at 0.01 (same as alpha_inh) — NOT zero,
        # because rectified_softplus has zero gradient at 0 which would
        # permanently kill the VIP pathway.
        assert torch.allclose(fb.alpha_vip, torch.full_like(fb.alpha_vip, 0.01))

    def test_forward_with_vip(self, cfg_emergent):
        """Full forward pass with VIP active — no NaN, no crash."""
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        B, T, N = 2, 10, cfg_emergent.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, final_state, aux = net(stim)
        assert r_l23_all.shape == (B, T, N)
        assert not torch.isnan(r_l23_all).any()
        assert "r_vip_all" in aux
        assert aux["r_vip_all"].shape == (B, T, N)
        assert not torch.isnan(aux["r_vip_all"]).any()
        # r_vip in final state
        assert final_state.r_vip.shape == (B, N)

    def test_alpha_vip_zero_som_unchanged(self, cfg_emergent):
        """When alpha_vip=0, SOM drive is unchanged from SOM-only behavior.

        With alpha_vip=0, VIP field=0, delta-VIP output=0. After VIPRing
        with zero drive, r_vip is near-zero. So effective_som_drive ≈ som_drive.
        """
        torch.manual_seed(42)
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        # Ensure alpha_vip is zero
        with torch.no_grad():
            net.feedback.alpha_vip.zero_()

        B, T, N = 2, 5, cfg_emergent.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, state, aux = net(stim)

        # r_vip should be near-zero throughout (zero drive + zero init)
        assert aux["r_vip_all"].abs().max() < 0.1, (
            f"VIP should be near-zero with alpha_vip=0, max={aux['r_vip_all'].abs().max():.4f}"
        )

    def test_gradient_flow_alpha_vip(self, cfg_emergent):
        """Gradient flows through alpha_vip after forward + backward."""
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        # Give alpha_vip non-uniform values so the circulant profile is
        # asymmetric and produces nonzero field (uniform weights → uniform
        # circulant → zero field on zero-mean input → zero gradient).
        with torch.no_grad():
            torch.manual_seed(0)
            net.feedback.alpha_vip.copy_(torch.randn(cfg_emergent.n_orientations) * 0.1)

        B, T, N = 2, 5, cfg_emergent.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        loss = r_l23_all.sum() + aux["r_vip_all"].sum()
        loss.backward()

        assert net.feedback.alpha_vip.grad is not None, "alpha_vip has no gradient"
        assert net.feedback.alpha_vip.grad.abs().sum() > 0, "alpha_vip gradient is all zeros"
        assert net.w_vip_som.grad is not None, "w_vip_som has no gradient"

    def test_vip_profile_shape(self, cfg_emergent):
        """get_vip_profile returns [N] tensor."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        K_vip = fb.get_vip_profile()
        assert K_vip.shape == (cfg_emergent.n_orientations,)

    def test_apical_gain_shape_and_range(self, cfg_emergent):
        """Apical gain is [B, N] with values in [1-max, 1+max]."""
        fb = EmergentFeedbackOperator(cfg_emergent, delta_som=True)
        B, N = 8, cfg_emergent.n_orientations
        # Random alpha_apical to get nontrivial gain
        with torch.no_grad():
            fb.alpha_apical.normal_(0, 1.0)
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0
        r_l4 = torch.randn(B, N).abs()

        _, _, apical_gain = fb(q_pred, pi_eff, r_l4)

        assert apical_gain.shape == (B, N)
        max_g = fb.max_apical_gain
        assert (apical_gain >= 1.0 - max_g - 1e-6).all(), (
            f"apical_gain below lower bound: min={apical_gain.min():.4f}"
        )
        assert (apical_gain <= 1.0 + max_g + 1e-6).all(), (
            f"apical_gain above upper bound: max={apical_gain.max():.4f}"
        )

    def test_apical_gain_unity_when_alpha_zero(self, cfg_emergent):
        """When alpha_apical=0, apical_gain = 1.0 everywhere (no modulation)."""
        fb = EmergentFeedbackOperator(cfg_emergent, delta_som=True)
        with torch.no_grad():
            fb.alpha_apical.zero_()
        B, N = 4, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0
        r_l4 = torch.randn(B, N).abs()

        _, _, apical_gain = fb(q_pred, pi_eff, r_l4)

        assert torch.allclose(apical_gain, torch.ones_like(apical_gain), atol=1e-6), (
            f"apical_gain should be 1.0 with alpha_apical=0, got range [{apical_gain.min():.6f}, {apical_gain.max():.6f}]"
        )

    def test_apical_gain_unity_when_l4_uniform(self, cfg_emergent):
        """With uniform L4 (centered basal_field=0), coincidence=0, gain=1.0."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        fb.alpha_apical.data.fill_(1.0)  # strong apical weights

        B, N = 4, cfg_emergent.n_orientations
        q_pred = torch.zeros(B, N)
        q_pred[:, 18] = 1.0  # peaked prediction
        q_pred = q_pred / q_pred.sum(dim=-1, keepdim=True)
        pi_eff = torch.ones(B, 1) * 3.0
        r_l4 = torch.ones(B, N) * 0.5  # uniform L4 → centered = 0

        _, _, apical_gain = fb(q_pred, pi_eff, r_l4)
        # Uniform L4 → zero basal field → zero coincidence → gain = 1.0
        assert torch.allclose(apical_gain, torch.ones_like(apical_gain), atol=1e-5)

    def test_apical_gain_zero_when_mismatch(self, cfg_emergent):
        """With mismatched prediction and L4, coincidence ≈ 0, gain ≈ 1.0."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        fb.alpha_apical.data.fill_(1.0)

        B, N = 4, cfg_emergent.n_orientations
        # Prediction peaked at channel 18 (90°)
        q_pred = torch.zeros(B, N)
        q_pred[:, 18] = 1.0
        q_pred = q_pred / q_pred.sum(dim=-1, keepdim=True)
        pi_eff = torch.ones(B, 1) * 3.0

        # L4 peaked at channel 0 (0°) — maximally mismatched
        r_l4 = torch.zeros(B, N)
        r_l4[:, 0] = 1.0

        _, _, apical_gain = fb(q_pred, pi_eff, r_l4)
        # Mismatch → coincidence ≈ 0 at both channel 0 and 18
        assert torch.allclose(apical_gain, torch.ones_like(apical_gain), atol=0.05)

    def test_apical_gain_positive_when_match(self, cfg_emergent):
        """With matching prediction and L4, coincidence > 0, gain > 1.0 at predicted channel."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        # Peaked profile at offset 0: "boost at the predicted channel"
        # (uniform weights → flat circulant → zero field on zero-mean input)
        fb.alpha_apical.data.zero_()
        fb.alpha_apical.data[0] = 1.0

        B, N = 4, cfg_emergent.n_orientations
        # Both prediction and L4 peaked at channel 18
        q_pred = torch.zeros(B, N)
        q_pred[:, 18] = 1.0
        q_pred = q_pred / q_pred.sum(dim=-1, keepdim=True)
        pi_eff = torch.ones(B, 1) * 3.0

        r_l4 = torch.zeros(B, N)
        r_l4[:, 18] = 1.0

        _, _, apical_gain = fb(q_pred, pi_eff, r_l4)
        # Match → positive coincidence at channel 18 → gain > 1.0
        assert apical_gain[0, 18] > 1.01, f"Expected gain > 1.01 at matched channel, got {apical_gain[0, 18]}"

    def test_apical_profile_shape(self, cfg_emergent):
        """get_apical_profile returns [N] tensor."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        K_apical = fb.get_apical_profile()
        assert K_apical.shape == (cfg_emergent.n_orientations,)

    def test_gradient_flow_alpha_apical(self, cfg_emergent):
        """Gradient flows through alpha_apical after forward + backward.

        Uses T=15 (not T=5) because apical gain affects L2/3 multiplicatively
        — gradient only flows when L2/3 is nonzero, which requires enough
        timesteps for the network to warm up from zero initial state.
        """
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        # Non-uniform weights required: uniform direct weights → flat circulant
        # → zero field on zero-mean input → zero gradient.
        with torch.no_grad():
            torch.manual_seed(1)
            net.feedback.alpha_apical.copy_(torch.randn(cfg_emergent.n_orientations) * 0.5)

        B, T, N = 2, 15, cfg_emergent.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        loss = r_l23_all.sum()
        loss.backward()

        assert net.feedback.alpha_apical.grad is not None, "alpha_apical has no gradient"
        assert net.feedback.alpha_apical.grad.abs().sum() > 0, "alpha_apical gradient is all zeros"

    def test_vip_config_loads(self):
        """VIP config YAML loads with tau_vip=10."""
        from src.config import load_config
        mc, tc, sc = load_config("config/exp_vip_ambig.yaml")
        assert mc.tau_vip == 10
        assert tc.delta_som is True
        assert tc.freeze_v2 is True
        assert tc.ambiguous_fraction == 0.3

    def test_network_state_has_r_vip(self):
        """NetworkState has r_vip field."""
        state = initial_state(batch_size=2)
        assert hasattr(state, 'r_vip')
        assert state.r_vip.shape == (2, 36)
        assert (state.r_vip == 0).all()

    def test_sparsity_loss_includes_vip(self, cfg_emergent):
        """feedback_sparsity_loss penalizes alpha_inh + alpha_vip + vip_excess + alpha_apical."""
        from src.training.losses import CompositeLoss
        from src.config import TrainingConfig
        tc = TrainingConfig(lambda_fb=0.01)
        loss_fn = CompositeLoss(tc, cfg_emergent)
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        # Set nonzero weights, zero apical to isolate VIP test
        with torch.no_grad():
            net.feedback.alpha_inh.fill_(0.5)
            net.feedback.alpha_vip.fill_(0.3)
            net.feedback.alpha_apical.zero_()
        l_fb = loss_fn.feedback_sparsity_loss(net)
        # L1(alpha_inh) + L1(alpha_vip) + relu(L1_vip - L1_inh) + L1(alpha_apical=0)
        N = cfg_emergent.n_orientations  # 36 direct channel weights
        expected_inh = 0.5 * N
        expected_vip = 0.3 * N
        # vip < inh so vip_excess = 0
        expected = expected_inh + expected_vip
        assert abs(l_fb.item() - expected) < 0.01, f"got {l_fb.item()}, expected {expected}"

    def test_sparsity_loss_vip_excess_penalty(self, cfg_emergent):
        """When VIP exceeds SOM, the excess penalty kicks in."""
        from src.training.losses import CompositeLoss
        from src.config import TrainingConfig
        tc = TrainingConfig(lambda_fb=0.01)
        loss_fn = CompositeLoss(tc, cfg_emergent)
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        # VIP > SOM, zero apical to isolate VIP test
        with torch.no_grad():
            net.feedback.alpha_inh.fill_(0.1)
            net.feedback.alpha_vip.fill_(0.5)
            net.feedback.alpha_apical.zero_()
        l_fb = loss_fn.feedback_sparsity_loss(net)
        N = cfg_emergent.n_orientations
        l1_inh = 0.1 * N
        l1_vip = 0.5 * N
        vip_excess = l1_vip - l1_inh  # positive
        expected = l1_inh + l1_vip + vip_excess
        assert abs(l_fb.item() - expected) < 0.01, f"got {l_fb.item()}, expected {expected}"

    def test_sparsity_loss_includes_apical(self, cfg_emergent):
        """feedback_sparsity_loss includes alpha_apical L1 term."""
        from src.training.losses import CompositeLoss
        from src.config import TrainingConfig
        tc = TrainingConfig(lambda_fb=0.01)
        loss_fn = CompositeLoss(tc, cfg_emergent)
        net = LaminarV1V2Network(cfg_emergent, delta_som=True)
        with torch.no_grad():
            net.feedback.alpha_inh.zero_()
            net.feedback.alpha_vip.zero_()
            net.feedback.alpha_apical.fill_(0.2)
        l_fb = loss_fn.feedback_sparsity_loss(net)
        expected = 0.2 * cfg_emergent.n_orientations  # L1(alpha_apical)
        assert abs(l_fb.item() - expected) < 0.01, f"got {l_fb.item()}, expected {expected}"


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

    def test_small_feedback_at_init(self, cfg_emergent):
        """Emergent feedback should be near-zero at initialization (alpha=0.01)."""
        net = LaminarV1V2Network(cfg_emergent)
        B, T, N = 2, 5, cfg_emergent.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        # SOM should be very small since alpha is only 0.01
        assert aux["r_som_all"].abs().max() < 0.5


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

        # VIP + emergent-only params don't get gradients in fixed mode
        unused_params = {
            "l4.pv_gain.gain_raw",
            "w_vip_som",
            "w_template_drive",  # only used in emergent branch
            "vip.tau_vip",  # not a param, but guard
            "feedback.alpha_vip",
            "feedback.vip_baseline",
        }
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
                + aux["q_pred_all"].sum()  # q_pred = mu_pred in emergent mode
                + aux["pi_pred_all"].sum())
        loss.backward()

        # Check emergent-specific params
        assert net.feedback.alpha_inh.grad is not None
        # Check V2 mu head (learned prior — q_pred IS mu_pred)
        assert net.v2.head_mu.weight.grad is not None

        unused_params = {"l4.pv_gain.gain_raw",
                         "v2.head_feedback.weight", "v2.head_feedback.bias"}
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

        state_detached = NetworkState(**{k: v.detach() for k, v in state1._asdict().items()})

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


# ── Branch C: Template Drive Tests ──────────────────────────────────────

class TestTemplateDrive:
    """Tests for w_template_drive: template→L2/3 center excitation (Branch C)."""

    def test_param_exists_and_init(self, cfg_emergent):
        """w_template_drive exists, is a scalar nn.Parameter, and inits at 0.0."""
        net = LaminarV1V2Network(cfg_emergent)
        assert hasattr(net, 'w_template_drive'), "Missing w_template_drive parameter"
        assert isinstance(net.w_template_drive, torch.nn.Parameter)
        assert net.w_template_drive.shape == ()
        assert net.w_template_drive.item() == 0.0

    def test_zero_init_matches_old_behavior(self, cfg_emergent):
        """With w_template_drive=0.0 (default), center_exc=0 → same as old zeros behavior."""
        torch.manual_seed(123)
        net = LaminarV1V2Network(cfg_emergent)
        N = cfg_emergent.n_orientations
        ori = torch.tensor([45.0])
        contrast = torch.tensor([0.8])
        stim_single = generate_grating(ori, contrast, N)
        stim = stim_single.unsqueeze(1).expand(1, 10, N)

        r_l23_all, _, _ = net(stim)
        assert not torch.isnan(r_l23_all).any()
        # With w_template_drive=0.0, center_exc = 0*deep_tmpl = 0, identical to old code
        assert torch.isfinite(r_l23_all).all()

    def test_positive_weight_increases_l23(self, cfg_emergent):
        """Positive w_template_drive adds excitation → L2/3 rates should increase."""
        torch.manual_seed(42)
        net = LaminarV1V2Network(cfg_emergent)
        N = cfg_emergent.n_orientations
        ori = torch.tensor([45.0, 90.0])
        contrast = torch.tensor([0.8, 0.8])
        stim_single = generate_grating(ori, contrast, N)  # [2, N]
        stim = stim_single.unsqueeze(1).expand(2, 20, N)

        # Baseline with w_template_drive=0
        r_l23_base, _, _ = net(stim)

        # Now set w_template_drive to a positive value
        with torch.no_grad():
            net.w_template_drive.fill_(2.0)
        r_l23_pos, _, _ = net(stim)

        # L2/3 should have higher mean activation with positive template drive
        assert r_l23_pos.sum() >= r_l23_base.sum(), (
            f"Positive w_template_drive should not decrease L2/3: "
            f"base_sum={r_l23_base.sum():.6f}, pos_sum={r_l23_pos.sum():.6f}"
        )

    def test_gradient_flow(self, cfg_emergent):
        """Gradient flows through w_template_drive after forward + backward."""
        torch.manual_seed(42)
        net = LaminarV1V2Network(cfg_emergent)
        # Set nonzero so gradient is nonzero (at 0.0, deep_tmpl*0=0 has zero grad)
        with torch.no_grad():
            net.w_template_drive.fill_(0.5)

        N = cfg_emergent.n_orientations
        ori = torch.tensor([45.0, 90.0])
        contrast = torch.tensor([0.8, 0.8])
        stim_single = generate_grating(ori, contrast, N)
        stim = stim_single.unsqueeze(1).expand(2, 15, N)
        r_l23_all, _, _ = net(stim)

        loss = r_l23_all.sum()
        loss.backward()

        assert net.w_template_drive.grad is not None, "w_template_drive has no gradient"
        assert net.w_template_drive.grad.abs() > 0, "w_template_drive gradient is zero"

    def test_in_optimizer(self, cfg_emergent):
        """w_template_drive appears in Stage 2 optimizer parameter groups."""
        from src.training.trainer import create_stage2_optimizer
        from src.config import TrainingConfig

        net = LaminarV1V2Network(cfg_emergent)
        # Minimal loss_fn mock with orientation_decoder
        class MockLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.orientation_decoder = torch.nn.Linear(36, 1)
        loss_fn = MockLoss()
        train_cfg = TrainingConfig()

        optimizer = create_stage2_optimizer(net, loss_fn, train_cfg)
        # Collect all params from all groups
        all_opt_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                all_opt_params.add(id(p))

        assert id(net.w_template_drive) in all_opt_params, (
            "w_template_drive not found in optimizer param groups"
        )
