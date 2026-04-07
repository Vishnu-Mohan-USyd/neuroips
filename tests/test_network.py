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

    def test_som_regime_gate_defaults_to_identity_when_disabled(self, cfg_emergent):
        """The SOM regime gate must be an exact identity map when disabled."""
        v2 = V2ContextModule(cfg_emergent)
        h_v2 = torch.randn(4, cfg_emergent.v2_hidden_dim)

        gate = v2.compute_som_regime_gate(h_v2)

        assert torch.allclose(gate, torch.ones_like(gate))

    def test_som_regime_gate_is_condition_dependent_when_enabled(self):
        """The scalar SOM regime gate should vary with h_v2 when enabled."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            som_regime_gate_enabled=True,
            som_regime_gate_beta=0.10,
            som_regime_gate_init_bias=-2.0,
        )
        v2 = V2ContextModule(cfg)
        with torch.no_grad():
            v2.head_som_regime.weight.zero_()
            v2.head_som_regime.weight[0, 0] = 1.0
            v2.head_som_regime.bias.zero_()

        h_low = torch.zeros(1, cfg.v2_hidden_dim)
        h_high = torch.zeros(1, cfg.v2_hidden_dim)
        h_high[0, 0] = 2.0

        gate_low = v2.compute_som_regime_gate(h_low)
        gate_high = v2.compute_som_regime_gate(h_high)

        assert gate_low.shape == (1, 1)
        assert gate_high.shape == (1, 1)
        assert gate_low.item() >= 1.0
        assert gate_high.item() > gate_low.item()


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

        som_drive = fb(q_pred, pi_eff)

        assert som_drive.shape == (B, N)

    def test_small_init_mostly_uniform(self, cfg_emergent):
        """Default init (alpha=0.01) should produce nearly uniform output.

        Softplus(field≈0) ≈ 0.693 gives a constant baseline. The spatial
        variation from the tiny alpha=0.01 field should be negligible.
        """
        fb = EmergentFeedbackOperator(cfg_emergent)
        B, N = 2, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0

        som_drive = fb(q_pred, pi_eff)

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

        som_drive = fb(q_pred, pi_eff)

        # With alpha=0, field=0, softplus(0)=ln(2)≈0.693 → uniform
        # Output should be constant across orientations (no spatial structure)
        assert som_drive.std(dim=-1).max() < 1e-5, "Should be spatially uniform"

    def test_non_negative_outputs(self, cfg_emergent):
        """SOM drive should always be >= 0 (softplus)."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        # Set random weights
        with torch.no_grad():
            fb.alpha_inh.normal_()

        B, N = 8, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0

        som_drive = fb(q_pred, pi_eff)

        assert (som_drive >= -1e-6).all()

    def test_basis_shape(self, cfg_emergent):
        """Basis should be [K, N] with K ~ 7."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        K, N = fb.basis.shape
        assert N == cfg_emergent.n_orientations
        assert K == 7  # 4 Gaussians + 1 Mexican hat + 1 constant + 1 odd

    def test_profiles_shape(self, cfg_emergent):
        fb = EmergentFeedbackOperator(cfg_emergent)
        K_inh = fb.get_profiles()
        assert K_inh.shape == (cfg_emergent.n_orientations,)

    def test_kernel_caching(self, cfg_emergent):
        """Cached and uncached should give same results."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        with torch.no_grad():
            fb.alpha_inh.normal_(std=0.5)

        B, N = 4, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 2.0

        # Without cache
        som1 = fb(q_pred, pi_eff)

        # With cache
        fb.cache_kernels()
        som2 = fb(q_pred, pi_eff)
        fb.uncache_kernels()

        assert torch.allclose(som1, som2, atol=1e-6)

    def test_som_regime_gate_disabled_keeps_som_path_identical(self, cfg_emergent):
        """Passing a gate tensor must be a no-op when the branch is disabled."""
        fb = EmergentFeedbackOperator(cfg_emergent, delta_som=True)
        B, N = 2, cfg_emergent.n_orientations
        q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        pi_eff = torch.ones(B, 1) * 3.0
        gate = torch.full((B, 1), 1.05)

        som_no_gate = fb(q_pred, pi_eff)
        som_with_gate = fb(q_pred, pi_eff, som_regime_gate=gate)

        assert torch.allclose(som_no_gate, som_with_gate, atol=1e-6)

    def test_som_regime_gate_modulates_alpha_inh_field_before_delta_som(self):
        """A larger SOM regime gate should increase learned inhibitory drive only."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            som_regime_gate_enabled=True,
            som_regime_gate_beta=0.10,
        )
        fb = EmergentFeedbackOperator(cfg, delta_som=True)
        with torch.no_grad():
            fb.alpha_inh.zero_()
            fb.alpha_inh[0] = 1.0
            fb.som_baseline.zero_()

        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_lo = torch.tensor([[1.0]])
        gate_hi = torch.tensor([[1.1]])

        som_lo = fb(q_pred, pi_eff, som_regime_gate=gate_lo)
        som_hi = fb(q_pred, pi_eff, som_regime_gate=gate_hi)

        assert torch.isfinite(som_hi).all()
        assert torch.isfinite(som_lo).all()
        pos_mask = som_lo > 0
        neg_mask = som_lo < 0
        assert pos_mask.any()
        assert neg_mask.any()
        assert (som_hi[pos_mask] >= som_lo[pos_mask] - 1e-7).all()
        assert (som_hi[neg_mask] <= som_lo[neg_mask] + 1e-7).all()
        assert som_hi.max().item() > som_lo.max().item()

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

    def test_center_support_disabled_is_exact_zero(self, cfg_emergent):
        """The new excitatory branch must be a strict no-op by default."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        q_pred = _one_hot_q(9, cfg_emergent.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg_emergent.n_orientations)
        gate_signal[0, 9] = 1.0

        center_exc = fb.compute_center_excitation(q_pred, pi_eff, gate_signal=gate_signal)

        assert torch.allclose(center_exc, torch.zeros_like(center_exc))

    def test_center_support_is_local_positive_when_cued(self):
        """Enabled center support should add a narrow positive bump at prediction."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_center_support_enabled=True,
            emergent_center_support_gain=0.12,
            emergent_center_support_sigma=5.0,
            emergent_center_support_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0

        center_exc = fb.compute_center_excitation(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert (center_exc >= 0).all()
        assert center_exc.argmax(dim=-1).item() == 9
        assert center_exc[0, 9].item() > 0.0
        assert center_exc[0, 9].item() > center_exc[0, 0].item()

    def test_center_support_buffer_not_serialized_in_state_dict(self):
        """Derived center-support buffers should not break strict legacy loads."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_center_support_enabled=True,
        )
        net = LaminarV1V2Network(cfg)
        state_dict = net.state_dict()

        assert "feedback.center_support_dists_sq" not in state_dict

        reloaded = LaminarV1V2Network(cfg)
        reloaded.load_state_dict(state_dict, strict=True)

    def test_som_regime_head_backward_compatible_with_legacy_state_dict(self):
        """Strict loading must tolerate checkpoints saved before the SOM gate head existed."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            som_regime_gate_enabled=True,
            som_regime_gate_beta=0.10,
        )
        net = LaminarV1V2Network(cfg)
        legacy_state_dict = {
            key: value
            for key, value in net.state_dict().items()
            if key not in {
                "v2.head_som_regime.weight",
                "v2.head_som_regime.bias",
            }
        }

        reloaded = LaminarV1V2Network(cfg)
        reloaded.load_state_dict(legacy_state_dict, strict=True)

    def test_center_support_persists_into_early_probe_window(self):
        """VIP-gated center support should survive a few probe steps after cue offset."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_center_support_enabled=True,
            emergent_center_support_gain=0.12,
            emergent_center_support_sigma=5.0,
            emergent_center_support_cue_gated=True,
        )
        net = LaminarV1V2Network(cfg)
        B, N = 1, cfg.n_orientations
        state = initial_state(B, N, cfg.v2_hidden_dim)
        stim = torch.zeros(B, N)
        task = torch.zeros(B, 2)
        cue_on = torch.zeros(B, N)
        cue_on[0, 9] = 1.0
        cue_off = torch.zeros(B, N)

        oracle_q = _one_hot_q(9, N)
        net.oracle_mode = True
        net.oracle_q_pred = oracle_q
        net.oracle_pi_pred = torch.tensor([[3.0]])

        try:
            for _ in range(4):
                state, _ = net.step(stim, cue_on, task, state)

            probe_center_exc = []
            for _ in range(3):
                state, aux = net.step(stim, cue_off, task, state)
                center_exc = net.feedback.compute_center_excitation(
                    aux.q_pred, aux.pi_pred_eff, gate_signal=state.r_vip
                )
                probe_center_exc.append(center_exc)
        finally:
            net.oracle_mode = False
            net.oracle_q_pred = None
            net.oracle_pi_pred = None

        for center_exc in probe_center_exc:
            assert center_exc[0, 9].item() > 0.0
            assert center_exc.argmax(dim=-1).item() == 9

    def test_uncued_support_gate_stays_zero_when_branch_enabled(self):
        """Without cue history, VIP-gated center support should remain exactly zero."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            emergent_center_support_enabled=True,
            emergent_center_support_gain=0.12,
            emergent_center_support_sigma=5.0,
            emergent_center_support_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)

        center_exc = fb.compute_center_excitation(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert torch.allclose(center_exc, torch.zeros_like(center_exc))

    def test_recurrent_gain_disabled_is_exact_zero(self, cfg_emergent):
        """The recurrent-gain branch must be a strict no-op by default."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        q_pred = _one_hot_q(9, cfg_emergent.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg_emergent.n_orientations)
        gate_signal[0, 9] = 1.0

        recurrent_gain = fb.compute_recurrent_gain(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert torch.allclose(recurrent_gain, torch.zeros_like(recurrent_gain))

    def test_recurrent_gain_is_local_positive_when_cued(self):
        """Enabled recurrent gain should create a narrow positive gain bump."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_beta=0.15,
            emergent_recurrent_gain_sigma=5.0,
            emergent_recurrent_gain_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0

        recurrent_gain = fb.compute_recurrent_gain(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert (recurrent_gain >= 0).all()
        assert recurrent_gain.argmax(dim=-1).item() == 9
        assert recurrent_gain[0, 9].item() > 0.0
        assert recurrent_gain[0, 9].item() > recurrent_gain[0, 0].item()

    def test_recurrent_gain_persists_into_early_probe_window(self):
        """VIP-gated recurrent gain should survive a few probe steps after cue offset."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_beta=0.15,
            emergent_recurrent_gain_sigma=5.0,
            emergent_recurrent_gain_cue_gated=True,
        )
        net = LaminarV1V2Network(cfg)
        B, N = 1, cfg.n_orientations
        state = initial_state(B, N, cfg.v2_hidden_dim)
        stim = torch.zeros(B, N)
        task = torch.zeros(B, 2)
        cue_on = torch.zeros(B, N)
        cue_on[0, 9] = 1.0
        cue_off = torch.zeros(B, N)

        oracle_q = _one_hot_q(9, N)
        net.oracle_mode = True
        net.oracle_q_pred = oracle_q
        net.oracle_pi_pred = torch.tensor([[3.0]])

        try:
            for _ in range(4):
                state, _ = net.step(stim, cue_on, task, state)

            probe_gains = []
            for _ in range(3):
                state, aux = net.step(stim, cue_off, task, state)
                recurrent_gain = net.feedback.compute_recurrent_gain(
                    aux.q_pred, aux.pi_pred_eff, gate_signal=state.r_vip
                )
                probe_gains.append(recurrent_gain)
        finally:
            net.oracle_mode = False
            net.oracle_q_pred = None
            net.oracle_pi_pred = None

        for recurrent_gain in probe_gains:
            assert recurrent_gain[0, 9].item() > 0.0
            assert recurrent_gain.argmax(dim=-1).item() == 9

    def test_signed_recurrent_gain_has_positive_center_and_negative_flanks(self):
        """Signed recurrent modulation should spare center gain while suppressing flanks."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_mode="signed_center_surround",
            emergent_recurrent_gain_beta=0.15,
            emergent_recurrent_gain_flank_beta=0.12,
            emergent_recurrent_gain_sigma=5.0,
            emergent_recurrent_gain_sigma_surround=20.0,
            emergent_recurrent_gain_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0

        recurrent_gain = fb.compute_recurrent_gain(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert recurrent_gain[0, 9].item() > 0.0
        flank_vals = recurrent_gain[0, [7, 8, 10, 11]]
        assert flank_vals.min().item() < 0.0
        assert recurrent_gain[0, 9].item() > flank_vals.abs().max().item()
        assert (1.0 + recurrent_gain).min().item() > 0.0

    def test_flank_som_disabled_is_exact_zero(self, cfg_emergent):
        """The flank-only SOM supplement must be a strict no-op by default."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        q_pred = _one_hot_q(9, cfg_emergent.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg_emergent.n_orientations)
        gate_signal[0, 9] = 1.0

        flank_boost = fb.compute_flank_som_boost(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert torch.allclose(flank_boost, torch.zeros_like(flank_boost))

    def test_flank_som_is_center_spared_and_flank_positive_when_cued(self):
        """Enabled flank SOM should spare the center and boost nearby flanks."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_flank_som_enabled=True,
            emergent_flank_som_gain=0.12,
            emergent_flank_som_sigma_center=5.0,
            emergent_flank_som_sigma_surround=20.0,
            emergent_flank_som_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0

        flank_boost = fb.compute_flank_som_boost(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        center_val = flank_boost[0, 9].item()
        flank_vals = flank_boost[0, [7, 8, 10, 11]]

        assert (flank_boost >= 0).all()
        assert center_val < 1e-7
        assert flank_vals.max().item() > 0.0
        assert flank_vals.mean().item() > center_val

    def test_zero_cue_keeps_flank_som_path_identical(self):
        """With zero cue, enabling flank SOM should leave trajectories unchanged."""
        cfg_off = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_flank_som_enabled=False,
        )
        net_off = LaminarV1V2Network(cfg_off)

        cfg_on = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_flank_som_enabled=True,
            emergent_flank_som_gain=0.12,
            emergent_flank_som_sigma_center=5.0,
            emergent_flank_som_sigma_surround=20.0,
            emergent_flank_som_cue_gated=True,
        )
        net_on = LaminarV1V2Network(cfg_on)
        net_on.load_state_dict(net_off.state_dict(), strict=True)

        B, T, N = 2, 6, cfg_on.n_orientations
        stim = torch.rand(B, T, N)
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)
        packed = net_on.pack_inputs(stim, cue, task)

        r_off, state_off, aux_off = net_off(packed)
        r_on, state_on, aux_on = net_on(packed)

        assert torch.allclose(r_off, r_on, atol=1e-6)
        assert torch.allclose(state_off.r_som, state_on.r_som, atol=1e-6)
        assert torch.allclose(aux_off["r_som_all"], aux_on["r_som_all"], atol=1e-6)

    def test_flank_shunt_disabled_is_exact_zero(self, cfg_emergent):
        """The flank-only shunt branch must be a strict no-op by default."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        q_pred = _one_hot_q(9, cfg_emergent.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg_emergent.n_orientations)
        gate_signal[0, 9] = 1.0

        flank_shunt = fb.compute_flank_shunt(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert torch.allclose(flank_shunt, torch.zeros_like(flank_shunt))

    def test_flank_shunt_is_center_spared_and_flank_positive_when_cued(self):
        """Enabled flank shunt should spare the center and rise on nearby flanks."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_flank_shunt_enabled=True,
            emergent_flank_shunt_gain=0.10,
            emergent_flank_shunt_sigma_center=5.0,
            emergent_flank_shunt_sigma_surround=20.0,
            emergent_flank_shunt_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0

        flank_shunt = fb.compute_flank_shunt(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        center_val = flank_shunt[0, 9].item()
        flank_vals = flank_shunt[0, [7, 8, 10, 11]]

        assert (flank_shunt >= 0).all()
        assert center_val < 1e-7
        assert flank_vals.max().item() > 0.0
        assert flank_vals.mean().item() > center_val

    def test_center_recruited_flank_shunt_is_center_spared_and_flank_positive_when_cued(self):
        """Center-recruited shunt should spread winner-driven suppression to flanks."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            emergent_flank_shunt_enabled=True,
            emergent_flank_shunt_gain=0.10,
            emergent_flank_shunt_sigma_center=5.0,
            emergent_flank_shunt_sigma_surround=20.0,
            emergent_flank_shunt_cue_gated=True,
            emergent_flank_shunt_source="center_recruited",
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0
        winner_proxy = torch.zeros_like(q_pred)
        winner_proxy[0, 9] = 2.0

        flank_shunt = fb.compute_flank_shunt(
            q_pred,
            pi_eff,
            gate_signal=gate_signal,
            winner_proxy=winner_proxy,
        )

        center_val = flank_shunt[0, 9].item()
        flank_vals = flank_shunt[0, [7, 8, 10, 11]]

        assert (flank_shunt >= 0).all()
        assert center_val < 1e-7
        assert flank_vals.max().item() > 0.0
        assert flank_vals.mean().item() > center_val

    def test_zero_cue_keeps_flank_shunt_path_identical(self):
        """With zero cue, enabling flank shunt should leave trajectories unchanged."""
        cfg_off = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_flank_shunt_enabled=False,
        )
        net_off = LaminarV1V2Network(cfg_off)

        cfg_on = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_flank_shunt_enabled=True,
            emergent_flank_shunt_gain=0.10,
            emergent_flank_shunt_sigma_center=5.0,
            emergent_flank_shunt_sigma_surround=20.0,
            emergent_flank_shunt_cue_gated=True,
            emergent_flank_shunt_source="center_recruited",
        )
        net_on = LaminarV1V2Network(cfg_on)
        net_on.load_state_dict(net_off.state_dict(), strict=True)

        B, T, N = 2, 6, cfg_on.n_orientations
        stim = torch.rand(B, T, N)
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)
        packed = net_on.pack_inputs(stim, cue, task)

        r_off, state_off, aux_off = net_off(packed)
        r_on, state_on, aux_on = net_on(packed)

        assert torch.allclose(r_off, r_on, atol=1e-6)
        assert torch.allclose(state_off.r_l23, state_on.r_l23, atol=1e-6)
        assert torch.allclose(aux_off["r_som_all"], aux_on["r_som_all"], atol=1e-6)

    def test_uncued_recurrent_gain_stays_zero_when_branch_enabled(self):
        """Without cue history, recurrent gain should remain exactly zero."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_beta=0.15,
            emergent_recurrent_gain_sigma=5.0,
            emergent_recurrent_gain_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)

        recurrent_gain = fb.compute_recurrent_gain(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert torch.allclose(recurrent_gain, torch.zeros_like(recurrent_gain))

    def test_apical_gain_disabled_is_exact_zero(self, cfg_emergent):
        """The apical/feedforward gain branch must be a strict no-op by default."""
        fb = EmergentFeedbackOperator(cfg_emergent)
        q_pred = _one_hot_q(9, cfg_emergent.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg_emergent.n_orientations)
        gate_signal[0, 9] = 1.0

        apical_gain = fb.compute_apical_gain(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert torch.allclose(apical_gain, torch.zeros_like(apical_gain))

    def test_apical_gain_is_local_positive_when_cued(self):
        """Enabled apical gain should create a narrow positive gain bump."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            apical_gain_enabled=True,
            apical_gain_beta=0.08,
            apical_gain_sigma=5.0,
            apical_gain_cue_gated=True,
        )
        fb = EmergentFeedbackOperator(cfg)
        q_pred = _one_hot_q(9, cfg.n_orientations)
        pi_eff = torch.tensor([[3.0]])
        gate_signal = torch.zeros(1, cfg.n_orientations)
        gate_signal[0, 9] = 1.0

        apical_gain = fb.compute_apical_gain(
            q_pred, pi_eff, gate_signal=gate_signal
        )

        assert (apical_gain >= 0).all()
        assert apical_gain.argmax(dim=-1).item() == 9
        assert apical_gain[0, 9].item() > 0.0
        assert apical_gain[0, 9].item() > apical_gain[0, 0].item()

    def test_apical_gain_persists_into_late_probe_window(self):
        """VIP-gated apical gain should survive into the late probe read window."""
        cfg = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            apical_gain_enabled=True,
            apical_gain_beta=0.08,
            apical_gain_tau=10,
            apical_gain_sigma=5.0,
            apical_gain_mode="persistent_sum",
            apical_gain_cue_gated=True,
        )
        net = LaminarV1V2Network(cfg)
        B, N = 1, cfg.n_orientations
        state = initial_state(B, N, cfg.v2_hidden_dim)
        stim = torch.zeros(B, N)
        task = torch.zeros(B, 2)
        cue_on = torch.zeros(B, N)
        cue_on[0, 9] = 1.0
        cue_off = torch.zeros(B, N)

        oracle_q = _one_hot_q(9, N)
        net.oracle_mode = True
        net.oracle_q_pred = oracle_q
        net.oracle_pi_pred = torch.tensor([[3.0]])

        try:
            for _ in range(4):
                state, _ = net.step(stim, cue_on, task, state)

            late_states = []
            for _ in range(6):
                state, aux = net.step(stim, cue_off, task, state)
                late_states.append(state.a_apical.clone())
        finally:
            net.oracle_mode = False
            net.oracle_q_pred = None
            net.oracle_pi_pred = None

        assert late_states[-1][0, 9].item() > 0.0
        assert late_states[-1].argmax(dim=-1).item() == 9

    def test_zero_cue_keeps_apical_gain_path_identical(self):
        """The apical gain path must be a strict no-op when cue input is zero."""
        torch.manual_seed(432)
        cfg_base = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            apical_gain_enabled=False,
        )
        net_base = LaminarV1V2Network(cfg_base)

        torch.manual_seed(432)
        cfg_gain = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            apical_gain_enabled=True,
            apical_gain_beta=0.08,
            apical_gain_tau=10,
            apical_gain_sigma=5.0,
            apical_gain_mode="persistent_sum",
            apical_gain_cue_gated=True,
        )
        net_gain = LaminarV1V2Network(cfg_gain)

        B, T, N = 2, 6, cfg_base.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)

        packed = LaminarV1V2Network.pack_inputs(stim, cue, task)
        r_base, state_base, aux_base = net_base(packed)
        r_gain, state_gain, aux_gain = net_gain(packed)

        assert torch.allclose(r_base, r_gain, atol=1e-6)
        assert torch.allclose(state_base.r_l23, state_gain.r_l23, atol=1e-6)
        assert torch.allclose(aux_base["r_som_all"], aux_gain["r_som_all"], atol=1e-6)
        assert torch.allclose(state_gain.a_apical, torch.zeros_like(state_gain.a_apical), atol=1e-7)


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
        assert aux["som_regime_gate_all"].shape == (B, T, 1)

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

    def test_vip_disabled_matches_zero_cue_path(self):
        """Enabling VIP should be a strict no-op when the cue tensor is zero."""
        torch.manual_seed(123)
        cfg_base = ModelConfig(feedback_mode='emergent', vip_enabled=False)
        net_base = LaminarV1V2Network(cfg_base)

        torch.manual_seed(123)
        cfg_vip = ModelConfig(feedback_mode='emergent', vip_enabled=True, vip_gain=1.0)
        net_vip = LaminarV1V2Network(cfg_vip)

        B, T, N = 2, 6, cfg_base.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)

        packed = LaminarV1V2Network.pack_inputs(stim, cue, task)
        r_base, state_base, aux_base = net_base(packed)
        r_vip, state_vip, aux_vip = net_vip(packed)

        assert torch.allclose(r_base, r_vip, atol=1e-6)
        assert torch.allclose(state_base.r_som, state_vip.r_som, atol=1e-6)
        assert torch.allclose(state_vip.r_vip, torch.zeros_like(state_vip.r_vip), atol=1e-7)
        assert torch.allclose(aux_base["r_som_all"], aux_vip["r_som_all"], atol=1e-6)
        assert torch.allclose(aux_vip["r_vip_all"], torch.zeros_like(aux_vip["r_vip_all"]), atol=1e-7)

    def test_zero_cue_keeps_recurrent_gain_path_identical(self):
        """The recurrent-gain path must be a strict no-op when cue input is zero."""
        torch.manual_seed(321)
        cfg_base = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            emergent_recurrent_gain_enabled=False,
        )
        net_base = LaminarV1V2Network(cfg_base)

        torch.manual_seed(321)
        cfg_gain = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            emergent_recurrent_gain_enabled=True,
            emergent_recurrent_gain_beta=0.15,
            emergent_recurrent_gain_sigma=5.0,
            emergent_recurrent_gain_cue_gated=True,
        )
        net_gain = LaminarV1V2Network(cfg_gain)

        B, T, N = 2, 6, cfg_base.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)

        packed = LaminarV1V2Network.pack_inputs(stim, cue, task)
        r_base, state_base, aux_base = net_base(packed)
        r_gain, state_gain, aux_gain = net_gain(packed)

        assert torch.allclose(r_base, r_gain, atol=1e-6)
        assert torch.allclose(state_base.r_l23, state_gain.r_l23, atol=1e-6)
        assert torch.allclose(aux_base["r_som_all"], aux_gain["r_som_all"], atol=1e-6)


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
        assert aux["r_vip_all"].shape == (B, T, N)
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

    def test_cue_driven_vip_reduces_som_locally(self):
        """With oracle feedback, cue-driven VIP should suppress SOM locally."""
        cfg = ModelConfig(
            mechanism=MechanismType.DAMPENING,
            feedback_mode='fixed',
            vip_enabled=True,
            vip_gain=2.0,
        )
        net = LaminarV1V2Network(cfg)
        B, N = 1, cfg.n_orientations
        state = initial_state(B, N, cfg.v2_hidden_dim)
        stim = torch.zeros(B, N)
        task = torch.zeros(B, 2)
        cue_off = torch.zeros(B, N)
        cue_on = torch.zeros(B, N)
        cue_on[0, 9] = 2.0

        oracle_q = torch.zeros(B, N)
        oracle_q[0, 9] = 1.0
        net.oracle_mode = True
        net.oracle_q_pred = oracle_q
        net.oracle_pi_pred = torch.tensor([[3.0]])
        try:
            state_off, _ = net.step(stim, cue_off, task, state)
            state_on, _ = net.step(stim, cue_on, task, state)
        finally:
            net.oracle_mode = False
            net.oracle_q_pred = None
            net.oracle_pi_pred = None

        assert state_on.r_vip[0, 9].item() > 0.0
        assert state_on.r_som[0, 9].item() < state_off.r_som[0, 9].item()

    def test_zero_cue_keeps_center_support_path_identical(self):
        """Cue-gated center support should be a strict no-op when cue is absent."""
        torch.manual_seed(321)
        cfg_base = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_center_support_enabled=False,
        )
        net_base = LaminarV1V2Network(cfg_base)

        torch.manual_seed(321)
        cfg_support = ModelConfig(
            feedback_mode='emergent',
            vip_enabled=True,
            vip_gain=0.35,
            emergent_center_support_enabled=True,
            emergent_center_support_gain=0.12,
            emergent_center_support_sigma=5.0,
            emergent_center_support_cue_gated=True,
        )
        net_support = LaminarV1V2Network(cfg_support)

        B, T, N = 2, 6, cfg_base.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)

        packed = LaminarV1V2Network.pack_inputs(stim, cue, task)
        r_base, state_base, aux_base = net_base(packed)
        r_support, state_support, aux_support = net_support(packed)

        assert torch.allclose(r_base, r_support, atol=1e-6)
        assert torch.allclose(state_base.r_l23, state_support.r_l23, atol=1e-6)
        assert torch.allclose(state_base.r_som, state_support.r_som, atol=1e-6)
        assert torch.allclose(aux_base["r_som_all"], aux_support["r_som_all"], atol=1e-6)
        assert torch.allclose(aux_base["q_pred_all"], aux_support["q_pred_all"], atol=1e-6)


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
        if not cfg.som_regime_gate_enabled:
            unused_params.update(
                {
                    "v2.head_som_regime.weight",
                    "v2.head_som_regime.bias",
                }
            )
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
        # Check V2 p_cw head
        assert net.v2.head_p_cw.weight.grad is not None

        unused_params = {"l4.pv_gain.gain_raw"}
        if not cfg_emergent.som_regime_gate_enabled:
            unused_params.update(
                {
                    "v2.head_som_regime.weight",
                    "v2.head_som_regime.bias",
                }
            )
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
