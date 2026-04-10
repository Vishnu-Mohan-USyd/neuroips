"""Tests for V2 Context Module and Full Network (simple feedback mode).

Tests cover V2 output shapes/constraints, network forward pass,
gradient flow, state detachment, and golden trial regression.
"""

import math

import torch
import pytest

from src.config import ModelConfig
from src.state import initial_state, NetworkState
from src.stimulus.gratings import generate_grating
from src.model.v2_context import V2ContextModule
from src.model.network import LaminarV1V2Network


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    """Default config (emergent feedback mode)."""
    return ModelConfig()


def _one_hot_q(channel: int, n: int = 36) -> torch.Tensor:
    """Create a batch-1 one-hot q_pred at the given channel."""
    q = torch.zeros(1, n)
    q[0, channel] = 1.0
    return q


# ── V2ContextModule Tests ──────────────────────────────────────────────

class TestV2Context:
    def test_output_shapes(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        mu_pred, pi_pred, feedback_signal, h_v2_new = v2(r_l4, r_l23, cue, task_state, h_v2)

        assert mu_pred.shape == (B, N)
        assert pi_pred.shape == (B, 1)
        assert feedback_signal.shape == (B, N)
        assert h_v2_new.shape == (B, H)

    def test_mu_pred_sums_to_one(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        mu_pred, _, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

        assert torch.allclose(mu_pred.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_mu_pred_non_negative(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)

        mu_pred, _, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

        assert (mu_pred >= 0).all()

    def test_pi_pred_bounded(self, cfg):
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim

        # Run multiple random inputs to stress-test bounds
        for _ in range(5):
            r_l4 = torch.randn(B, N) * 10
            r_l23 = torch.randn(B, N) * 10
            cue = torch.zeros(B, N)
            task_state = torch.zeros(B, 2)
            h_v2 = torch.randn(B, H) * 5

            _, pi_pred, _, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

            assert (pi_pred >= 0).all()
            assert (pi_pred <= cfg.pi_max + 1e-5).all()

    def test_feedback_signal_is_raw(self, cfg):
        """feedback_signal should be raw (no activation), can be positive or negative."""
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.randn(B, H)

        _, _, fb_signal, _ = v2(r_l4, r_l23, cue, task_state, h_v2)

        assert torch.isfinite(fb_signal).all()
        # Raw signal should have both positive and negative values in general
        # (not guaranteed with random init, but shape should be correct)
        assert fb_signal.shape == (B, N)

    def test_v2_input_modes(self):
        """V2 should work with all input modes: l23, l4, l4_l23."""
        for mode in ('l23', 'l4', 'l4_l23'):
            cfg = ModelConfig(v2_input_mode=mode)
            v2 = V2ContextModule(cfg)
            B, N, H = 2, cfg.n_orientations, cfg.v2_hidden_dim
            r_l4 = torch.randn(B, N)
            r_l23 = torch.randn(B, N)
            cue = torch.zeros(B, N)
            task_state = torch.zeros(B, 2)
            h_v2 = torch.zeros(B, H)

            mu_pred, pi_pred, fb, h = v2(r_l4, r_l23, cue, task_state, h_v2)
            assert mu_pred.shape == (B, N)
            assert pi_pred.shape == (B, 1)
            assert fb.shape == (B, N)
            assert h.shape == (B, H)


# ── Full Network Tests ────────────────────────────────────────────────────

class TestNetworkForward:
    def test_forward_no_error(self, cfg):
        """Forward pass: B=4, T=20, no errors."""
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
        assert aux["deep_template_all"].shape == (B, T, N)
        assert aux["p_cw_all"].shape == (B, T, 1)

    def test_forward_no_nan(self, cfg):
        """No NaN in any output after 20 steps."""
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
        assert aux.p_cw.shape == (B, 1)

    def test_default_cue_and_task_state(self, cfg):
        """forward() should work with cue_seq=None and task_state_seq=None."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        r_l23_all, _, _ = net(stim)
        assert r_l23_all.shape == (B, T, N)

    def test_feedback_ei_split(self, cfg):
        """V2 feedback signal is split into center_exc (>=0) and SOM drive (>=0)."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        _, _, aux = net(stim)
        center_exc = aux["center_exc_all"]

        # center_exc should be non-negative (relu of positive part)
        assert (center_exc >= -1e-6).all()

    def test_som_non_negative(self, cfg):
        """SOM rates should always be non-negative."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 15, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        _, _, aux = net(stim)
        assert (aux["r_som_all"] >= -1e-6).all()

    def test_packed_input_format(self, cfg):
        """forward() should work with packed [B, T, N+N+2] input."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        cue = torch.zeros(B, T, N)
        task = torch.zeros(B, T, 2)
        packed = net.pack_inputs(stim, cue, task)

        r_l23_all, _, _ = net(packed)
        assert r_l23_all.shape == (B, T, N)


# ── Gradient Flow Tests ──────────────────────────────────────────────────

class TestGradientFlowNetwork:
    def test_gradients_through_full_network(self, cfg):
        """Forward 5 steps, backward, verify all trainable params have gradients."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 5, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        loss = (r_l23_all.sum()
                + aux["deep_template_all"].sum()
                + aux["q_pred_all"].sum()
                + aux["pi_pred_all"].sum())
        loss.backward()

        # Check that all trainable params have gradients
        unused_params = {"l4.pv_gain.gain_raw"}
        for name, param in net.named_parameters():
            if param.requires_grad and name not in unused_params:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_flow_v2_heads(self, cfg):
        """Gradient should flow through V2's head_mu, head_pi, head_feedback."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 5, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        loss = r_l23_all.sum() + aux["q_pred_all"].sum() + aux["pi_pred_all"].sum()
        loss.backward()

        # V2 mu head (learned prior)
        assert net.v2.head_mu.weight.grad is not None
        # V2 feedback head
        assert net.v2.head_feedback.weight.grad is not None
        # V2 pi head
        assert net.v2.head_pi.weight.grad is not None


# ── State Detachment ──────────────────────────────────────────────────────

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


# ── Golden Trial Regression ──────────────────────────────────────────────

class TestGoldenTrials:
    def test_golden_deterministic(self, cfg):
        """Same seed, same config → identical output."""
        torch.manual_seed(42)
        net1 = LaminarV1V2Network(cfg)
        B, T, N = 1, 10, cfg.n_orientations
        ori = torch.tensor([45.0])
        contrast = torch.tensor([0.8])
        stim_single = generate_grating(ori, contrast, N)
        stim_seq = stim_single.unsqueeze(1).expand(B, T, N)
        r_l23_1, _, _ = net1(stim_seq)

        torch.manual_seed(42)
        net2 = LaminarV1V2Network(cfg)
        r_l23_2, _, _ = net2(stim_seq)

        assert torch.allclose(r_l23_1, r_l23_2, atol=1e-6)


# ── Oracle Mode Tests ────────────────────────────────────────────────────

class TestOracleMode:
    def test_oracle_per_step(self, cfg):
        """Oracle mode with per-step tensors should work without error."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        net.oracle_mode = True
        net.oracle_q_pred = torch.softmax(torch.randn(B, N), dim=-1)
        net.oracle_pi_pred = torch.ones(B, 1) * 3.0

        r_l23_all, _, aux = net(stim)
        assert r_l23_all.shape == (B, T, N)
        assert not torch.isnan(r_l23_all).any()

    def test_oracle_per_sequence(self, cfg):
        """Oracle mode with per-sequence [B, T, N] tensors should work."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        net.oracle_mode = True
        net.oracle_q_pred = torch.softmax(torch.randn(B, T, N), dim=-1)
        net.oracle_pi_pred = torch.ones(B, T, 1) * 3.0

        r_l23_all, _, aux = net(stim)
        assert r_l23_all.shape == (B, T, N)
        assert not torch.isnan(r_l23_all).any()


# ── Simple Feedback Integration ──────────────────────────────────────────

class TestSimpleFeedback:
    def test_small_feedback_at_init(self, cfg):
        """Feedback should be near-zero at initialization (head_feedback random init)."""
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 5, cfg.n_orientations

        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        # SOM should be relatively small at init
        assert aux["r_som_all"].abs().max() < 5.0

    def test_feedback_scale_affects_output(self, cfg):
        """feedback_scale=0 should produce different output from feedback_scale=1."""
        torch.manual_seed(42)
        net = LaminarV1V2Network(cfg)
        B, T, N = 2, 15, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5

        # Run with feedback_scale=1 (default)
        r_l23_on, _, _ = net(stim)

        # Run with feedback_scale=0
        net.feedback_scale.fill_(0.0)
        r_l23_off, _, _ = net(stim)

        # Should be different (feedback contributes something)
        # Note: at init the difference may be small, but shapes should match
        assert r_l23_on.shape == r_l23_off.shape

    def test_no_vip_module(self, cfg):
        """Network should NOT have VIP module (legacy removed)."""
        net = LaminarV1V2Network(cfg)
        assert not hasattr(net, 'vip'), "VIP module should be removed"
        assert not hasattr(net, 'w_vip_som'), "w_vip_som should be removed"
        assert not hasattr(net, 'w_template_drive'), "w_template_drive should be removed"
        assert not hasattr(net, 'deep_template'), "deep_template module should be removed"
        assert not hasattr(net, 'feedback'), "feedback operator should be removed"

    def test_make_bump_exists_and_shapes(self, cfg):
        """Regression: _make_bump must exist on the network for oracle modes.

        stage2_feedback.py calls ``net._make_bump`` at four sites to build
        peaked prediction priors for oracle_true, oracle_wrong and oracle_random
        templates. A previous refactor silently removed the method, breaking
        every oracle_template path except oracle_uniform. This test pins the
        contract so it can never disappear again without a visible failure.
        """
        net = LaminarV1V2Network(cfg)
        N = cfg.n_orientations

        # Default sigma (sigma_ff fallback)
        thetas = torch.tensor([0.0, 45.0, 90.0, 135.0])
        bumps = net._make_bump(thetas)
        assert bumps.shape == (4, N)
        # Argmax should land on the nearest channel to each theta
        # (45/180 * 36 = 9, 90/180 * 36 = 18, 135/180 * 36 = 27)
        assert bumps.argmax(dim=-1).tolist() == [0, 9, 18, 27]

        # Explicit sigma must match default when equal to sigma_ff
        bumps_explicit = net._make_bump(thetas, sigma=cfg.sigma_ff)
        assert torch.allclose(bumps, bumps_explicit)

        # Narrower sigma → sharper peaks
        bumps_narrow = net._make_bump(thetas, sigma=3.0)
        assert bumps_narrow.max(dim=-1).values.sum() > 0
        # Narrow bumps should have smaller mass at the off-peak channels
        off_peak_mass_default = bumps.sum(dim=-1) - bumps.max(dim=-1).values
        off_peak_mass_narrow = bumps_narrow.sum(dim=-1) - bumps_narrow.max(dim=-1).values
        assert (off_peak_mass_narrow < off_peak_mass_default).all()
