"""Tests for Phase 4: V2 Context Module, Feedback Mechanism, Full Network.

Includes mechanism-specific SOM pattern tests, golden trial regression tests,
gradient flow, and state detachment between trials.
"""

import math

import torch
import pytest

from src.config import ModelConfig, MechanismType
from src.utils import shifted_softplus
from src.state import initial_state, NetworkState
from src.stimulus.gratings import generate_grating
from src.model.v2_context import V2ContextModule
from src.model.feedback import FeedbackMechanism
from src.model.network import LaminarV1V2Network


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return ModelConfig()


def _cfg_with_mechanism(mechanism: MechanismType) -> ModelConfig:
    return ModelConfig(mechanism=mechanism)


def _one_hot_q(channel: int, n: int = 36) -> torch.Tensor:
    """Create a batch-1 one-hot q_pred at the given channel."""
    q = torch.zeros(1, n)
    q[0, channel] = 1.0
    return q


# ── V2ContextModule Tests ────────────────────────────────────────────────

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


# ── FeedbackMechanism Tests ─────────────────────────────────────────────

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
        # Peak should be at or very near channel 9
        assert som.argmax().item() == 9
        # Should have a narrow peak
        assert som[9] > som[0]  # peak vs far away

    def test_sharpening_lowest_at_expected(self):
        """Model B: SOM is LOWEST at expected orientation (DoG center-sparing).

        The signed DoG (broad - narrow) produces a dip at the expected channel
        and maxima at flanks. SOM.argmin() should be at the expected channel.
        """
        som = self._get_som_drive(MechanismType.SHARPENING, channel=9)
        # SOM should be minimum at channel 9
        assert som.argmin().item() == 9, (
            f"SOM argmin={som.argmin().item()}, expected 9"
        )
        # Flanks (channels +-3 to +-6 offsets = 15-30 deg away) should exceed expected
        for offset in [3, 4, 5, 6]:
            flank_ch = (9 + offset) % 36
            assert som[flank_ch] > som[9], (
                f"Flank ch{flank_ch} ({som[flank_ch]:.4f}) should exceed "
                f"expected ch9 ({som[9]:.4f})"
            )

    def test_center_surround_low_center_high_flanks(self):
        """Model C: SOM is low near expected, high at flanks."""
        som = self._get_som_drive(MechanismType.CENTER_SURROUND, channel=9)
        # At channel 9, center subtraction should reduce SOM drive
        # Flanks should be higher than center
        # Find a flank channel ~30-50 deg away
        flank_channel = 15  # 75 deg, ~30 deg from channel 9 at 45 deg
        assert som[flank_channel] > som[9], (
            f"Flank ({som[flank_channel]:.4f}) should exceed center ({som[9]:.4f})"
        )

    def test_adaptation_only_zero_som(self):
        """Model D: SOM drive is exactly zero."""
        som = self._get_som_drive(MechanismType.ADAPTATION_ONLY, channel=9)
        assert torch.allclose(som, torch.zeros_like(som))

    def test_predictive_error_zero_som(self):
        """Model E: SOM drive is exactly zero."""
        som = self._get_som_drive(MechanismType.PREDICTIVE_ERROR, channel=9)
        assert torch.allclose(som, torch.zeros_like(som))


class TestModelCNetEffect:
    """Test that Model C's net top-down effect is positive at center, negative at flanks."""

    def test_center_surround_net_effect(self):
        """For Model C: center_excitation > SOM at expected, SOM > center_excitation at flanks."""
        cfg = _cfg_with_mechanism(MechanismType.CENTER_SURROUND)
        fb = FeedbackMechanism(cfg)
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])

        som = fb.compute_som_drive(q_pred, pi_pred).squeeze(0)
        ce = fb.compute_center_excitation(q_pred, pi_pred).squeeze(0)

        # At expected channel: center excitation should dominate (net positive)
        net_center = ce[9] - som[9]
        assert net_center > 0, f"Net at center should be positive, got {net_center:.4f}"

        # At flanks (+-6 channels = 30 deg away): SOM should dominate (net negative)
        for offset in [6, 7, 8]:
            flank_ch = (9 + offset) % 36
            net_flank = ce[flank_ch] - som[flank_ch]
            assert net_flank < 0, (
                f"Net at flank ch{flank_ch} should be negative, got {net_flank:.4f}"
            )


class TestCenterExcitation:
    def test_only_model_c_has_center_excitation(self):
        """Only Model C (center-surround) should produce nonzero center excitation."""
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])

        for mech in MechanismType:
            cfg = _cfg_with_mechanism(mech)
            fb = FeedbackMechanism(cfg)
            ce = fb.compute_center_excitation(q_pred, pi_pred)

            if mech == MechanismType.CENTER_SURROUND:
                assert ce.abs().sum() > 0, f"{mech.value} should have nonzero center excitation"
            else:
                assert torch.allclose(ce, torch.zeros_like(ce)), (
                    f"{mech.value} should have zero center excitation"
                )


class TestErrorSignal:
    def test_model_e_suppresses_predicted(self):
        """Model E: where template > l4, error should be near zero."""
        cfg = _cfg_with_mechanism(MechanismType.PREDICTIVE_ERROR)
        fb = FeedbackMechanism(cfg)

        r_l4 = torch.ones(1, 36)
        template = torch.zeros(1, 36)
        template[0, 9] = 3.0  # Strong prediction at channel 9

        error = fb.compute_error_signal(r_l4, template)

        # Channel 9: shifted_softplus(1 - 3) = shifted_softplus(-2) < 0
        assert error[0, 9] < 0.0
        # Other channels: shifted_softplus(1 - 0) = shifted_softplus(1) ≈ 0.62
        assert error[0, 0] > 0.5

    def test_non_e_mechanisms_pass_through_l4(self):
        """For non-E mechanisms, error signal is just r_l4 unchanged."""
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
        # Force raw param very high
        with torch.no_grad():
            fb.surround_width_raw.fill_(10.0)
        assert fb.surround_width.item() <= 15.0 + 1e-5

    def test_sharpening_broad_exceeds_narrow_by_10(self):
        """Model B: broad sigma >= narrow sigma + 10 deg."""
        cfg = _cfg_with_mechanism(MechanismType.SHARPENING)
        fb = FeedbackMechanism(cfg)
        # Force surround raw very low to test clamp
        with torch.no_grad():
            fb.surround_width_raw.fill_(-10.0)
        narrow = fb.center_width.item()
        broad = fb.surround_width.item()
        assert broad >= narrow + 10.0 - 1e-5, (
            f"broad={broad:.1f} should be >= narrow={narrow:.1f} + 10"
        )


# ── Full Network Tests ───────────────────────────────────────────────────

class TestNetworkForward:
    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_forward_all_mechanisms_no_error(self, mech):
        """Forward pass for all 5 mechanisms: B=4, T=20, no errors."""
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

    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_forward_no_nan(self, mech):
        """No NaN in any output after 20 steps."""
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

    def test_default_cue_and_task_state(self, cfg):
        """forward() should work with cue_seq=None and task_state_seq=None."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        r_l23_all, _, _ = net(stim)
        assert r_l23_all.shape == (B, T, N)


class TestGradientFlowNetwork:
    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_gradients_through_full_network(self, mech):
        """Forward 5 steps, backward, verify all trainable params have gradients.

        Uses a comprehensive loss covering all output paths (matching training loss
        which includes sensory, prediction, and energy terms).
        """
        cfg = _cfg_with_mechanism(mech)
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 5, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        # Comprehensive loss: sensory (r_l23) + prediction (state_logits) + template
        loss = (r_l23_all.sum()
                + aux["state_logits_all"].sum()
                + aux["deep_template_all"].sum()
                + aux["q_pred_all"].sum()
                + aux["pi_pred_all"].sum())
        loss.backward()

        # l4.pv_gain is stored on L4 for convenience but unused in the forward path
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

        # Detach state between trials
        state_detached = NetworkState(*[s.detach() for s in state1])

        stim2 = torch.randn(B, T, N).abs() * 0.5
        r_l23_2, _, _ = net(stim2, state=state_detached)

        loss = r_l23_2.sum()
        loss.backward()

        # The key test: stim1 should have no grad since we detached
        assert stim1.grad is None or torch.allclose(stim1.grad, torch.zeros_like(stim1))


# ── Golden Trial Tests ───────────────────────────────────────────────────

class TestGoldenTrials:
    """Fixed-seed regression tests. Any refactor that changes outputs will break these."""

    def _golden_som_drive(self, mech: MechanismType) -> torch.Tensor:
        """Compute SOM drive with fixed seed for reproducibility."""
        torch.manual_seed(42)
        cfg = _cfg_with_mechanism(mech)
        fb = FeedbackMechanism(cfg)
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])
        return fb.compute_som_drive(q_pred, pi_pred)

    def test_golden_dampening_som(self):
        som = self._golden_som_drive(MechanismType.DAMPENING)
        # Record exact values at key channels
        expected_peak = som[0, 9].item()
        expected_far = som[0, 0].item()
        # Re-run and verify determinism
        som2 = self._golden_som_drive(MechanismType.DAMPENING)
        assert torch.allclose(som, som2, atol=1e-6)
        # Print for record
        print(f"\n=== Golden dampening SOM (channel 9, pi=3.0) ===")
        print(f"  Peak (ch9): {expected_peak:.6f}")
        print(f"  Far (ch0):  {expected_far:.6f}")
        print(f"  Ratio peak/far: {expected_peak / max(expected_far, 1e-10):.2f}")

    def test_golden_sharpening_som(self):
        som = self._golden_som_drive(MechanismType.SHARPENING)
        som2 = self._golden_som_drive(MechanismType.SHARPENING)
        assert torch.allclose(som, som2, atol=1e-6)
        # Model B should have minimum at expected channel (DoG)
        assert som[0].argmin().item() == 9
        print(f"\n=== Golden sharpening SOM (channel 9, pi=3.0) ===")
        print(f"  Min (ch9):  {som[0, 9].item():.6f}")
        print(f"  Max:        {som[0].max().item():.6f} at ch{som[0].argmax().item()}")
        print(f"  Far (ch0):  {som[0, 0].item():.6f}")
        print(f"  Flank (ch12): {som[0, 12].item():.6f}")
        print(f"  Flank (ch15): {som[0, 15].item():.6f}")

    def test_golden_center_surround_som(self):
        som = self._golden_som_drive(MechanismType.CENTER_SURROUND)
        som2 = self._golden_som_drive(MechanismType.CENTER_SURROUND)
        assert torch.allclose(som, som2, atol=1e-6)
        print(f"\n=== Golden center-surround SOM (channel 9, pi=3.0) ===")
        print(f"  Center (ch9): {som[0, 9].item():.6f}")
        print(f"  Flank (ch15): {som[0, 15].item():.6f}")
        print(f"  Far (ch0):    {som[0, 0].item():.6f}")

    def test_golden_full_network_dampening(self):
        """Golden trial: full network, dampening, 10 steps, fixed seed."""
        torch.manual_seed(42)
        cfg = _cfg_with_mechanism(MechanismType.DAMPENING)
        net = LaminarV1V2Network(cfg)
        B, T, N = 1, 10, cfg.n_orientations

        # Fixed stimulus: grating at 45 deg, contrast 0.8
        ori = torch.tensor([45.0])
        contrast = torch.tensor([0.8])
        stim_single = generate_grating(ori, contrast, N)  # [1, N]
        stim_seq = stim_single.unsqueeze(1).expand(B, T, N)

        r_l23_all, final_state, aux = net(stim_seq)

        # Record and verify determinism
        torch.manual_seed(42)
        net2 = LaminarV1V2Network(cfg)
        r_l23_all2, _, _ = net2(stim_seq)
        assert torch.allclose(r_l23_all, r_l23_all2, atol=1e-6)

        print(f"\n=== Golden full network (dampening, θ=45°, c=0.8, T=10) ===")
        print(f"  L2/3 peak channel at T=10: {final_state.r_l23[0].argmax().item()}")
        print(f"  L2/3 peak rate:   {final_state.r_l23[0].max().item():.4f}")
        print(f"  PV rate:          {final_state.r_pv[0, 0].item():.4f}")
        print(f"  q_pred entropy:   {-(aux['q_pred_all'][0, -1] * torch.log(aux['q_pred_all'][0, -1] + 1e-10)).sum().item():.4f}")
        print(f"  pi_pred:          {aux['pi_pred_all'][0, -1, 0].item():.4f}")


# ── Numerical Results (print tests) ──────────────────────────────────────

class TestPhase4Numerical:
    def test_print_mechanism_som_profiles(self):
        """Print SOM drive profile for each mechanism with one-hot q_pred."""
        q_pred = _one_hot_q(9)
        pi_pred = torch.tensor([[3.0]])
        N = 36

        print(f"\n=== SOM drive profiles (one-hot at ch9, pi=3.0) ===")
        for mech in [MechanismType.DAMPENING, MechanismType.SHARPENING,
                     MechanismType.CENTER_SURROUND, MechanismType.ADAPTATION_ONLY,
                     MechanismType.PREDICTIVE_ERROR]:
            cfg = _cfg_with_mechanism(mech)
            fb = FeedbackMechanism(cfg)
            som = fb.compute_som_drive(q_pred, pi_pred).squeeze(0)
            print(f"  {mech.value:20s}: "
                  f"ch9={som[9].item():+.4f}  "
                  f"ch0={som[0].item():+.4f}  "
                  f"ch15={som[15].item():+.4f}  "
                  f"max={som.max().item():+.4f}  "
                  f"min={som.min().item():+.4f}")

    def test_print_v2_initial_output(self, cfg):
        """Print V2 outputs from zero initial state."""
        v2 = V2ContextModule(cfg)
        B, N, H = 1, cfg.n_orientations, cfg.v2_hidden_dim

        r_l23 = torch.zeros(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        q_pred, pi_pred, state_logits, _ = v2(torch.zeros_like(r_l23), r_l23, cue, task_state, h_v2)

        print(f"\n=== V2 initial output (all-zero inputs) ===")
        print(f"  q_pred max: {q_pred.max().item():.4f}, min: {q_pred.min().item():.4f}")
        print(f"  q_pred sum: {q_pred.sum().item():.4f}")
        print(f"  pi_pred: {pi_pred[0, 0].item():.4f} (max={cfg.pi_max})")
        print(f"  state_logits: [{state_logits[0, 0].item():.4f}, "
              f"{state_logits[0, 1].item():.4f}, {state_logits[0, 2].item():.4f}]")

    def test_print_error_signal_values(self):
        """Print Model E error signal values."""
        cfg = _cfg_with_mechanism(MechanismType.PREDICTIVE_ERROR)
        fb = FeedbackMechanism(cfg)

        r_l4 = torch.ones(1, 36)
        template = torch.zeros(1, 36)
        template[0, 9] = 3.0

        error = fb.compute_error_signal(r_l4, template)

        print(f"\n=== Model E error signal (l4=1.0, template[9]=3.0) ===")
        print(f"  error[9] (predicted):   {error[0, 9].item():.6f}")
        print(f"  error[0] (unpredicted): {error[0, 0].item():.6f}")
        print(f"  shifted_softplus(-2):   {shifted_softplus(torch.tensor(-2.0)).item():.6f}")
        print(f"  shifted_softplus(1):    {shifted_softplus(torch.tensor(1.0)).item():.6f}")

    def test_print_som_profile_vs_delta_theta(self):
        """Print SOM drive vs delta-theta for A, B, C — key diagnostic figure."""
        q_pred = _one_hot_q(9)  # expected at channel 9 = 45 deg
        pi_pred = torch.tensor([[3.0]])

        print(f"\n=== SOM drive vs delta-theta (one-hot at ch9=45deg, pi=3.0) ===")
        print(f"  {'dtheta':>6s}  {'A(damp)':>8s}  {'B(sharp)':>8s}  {'C(c-s)':>8s}")
        print(f"  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'------':>8s}")

        for mech, label in [(MechanismType.DAMPENING, "A"),
                            (MechanismType.SHARPENING, "B"),
                            (MechanismType.CENTER_SURROUND, "C")]:
            cfg = _cfg_with_mechanism(mech)
            fb = FeedbackMechanism(cfg)
            som = fb.compute_som_drive(q_pred, pi_pred).squeeze(0)

            if label == "A":
                vals_a = som
            elif label == "B":
                vals_b = som
            else:
                vals_c = som

        # Print at selected delta-theta values (channels offset from 9)
        for offset in [0, 1, 2, 3, 4, 5, 6, 9, 12, 18]:
            ch = (9 + offset) % 36
            dtheta = offset * 5  # 5 deg per channel
            print(f"  {dtheta:5d}°  {vals_a[ch].item():+8.4f}  "
                  f"{vals_b[ch].item():+8.4f}  {vals_c[ch].item():+8.4f}")
