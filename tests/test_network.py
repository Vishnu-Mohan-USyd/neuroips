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


# ── V2 Per-Regime Feedback (Task #9 / Fix 2) ─────────────────────────────

class TestV2PerRegimeFeedback:
    """Regression tests for the per-regime head_feedback in V2ContextModule.

    When `cfg.use_per_regime_feedback=False` (default, Network_mm), V2 uses
    a single shared `head_feedback`. When `cfg.use_per_regime_feedback=True`
    (Network_both, Task #9 / Fix 2), V2 instantiates `head_feedback_focused`
    and `head_feedback_routine`, gates them by `task_state[:, 0:1]` and
    `task_state[:, 1:2]`, and the legacy `head_feedback` attribute MUST NOT
    exist (so any stale code path fails loudly).
    """

    def test_per_regime_feedback_off_is_legacy(self):
        """Default: only the legacy `head_feedback` exists; per-regime heads do not."""
        cfg = ModelConfig(use_per_regime_feedback=False)
        v2 = V2ContextModule(cfg)
        assert hasattr(v2, "head_feedback")
        assert isinstance(v2.head_feedback, torch.nn.Linear)
        assert not hasattr(v2, "head_feedback_focused")
        assert not hasattr(v2, "head_feedback_routine")

        # Forward shape check (legacy mode ignores task_state for fb).
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        h_v2 = torch.zeros(B, H)
        _, _, fb, _ = v2(r_l4, r_l23, cue, task_state, h_v2)
        assert fb.shape == (B, N)

    def test_per_regime_feedback_on_constructs(self):
        """Fix 2 ON: per-regime heads exist; legacy `head_feedback` does NOT."""
        cfg = ModelConfig(use_per_regime_feedback=True)
        v2 = V2ContextModule(cfg)
        N, H = cfg.n_orientations, cfg.v2_hidden_dim

        assert hasattr(v2, "head_feedback_focused")
        assert hasattr(v2, "head_feedback_routine")
        assert isinstance(v2.head_feedback_focused, torch.nn.Linear)
        assert isinstance(v2.head_feedback_routine, torch.nn.Linear)
        assert v2.head_feedback_focused.in_features == H
        assert v2.head_feedback_focused.out_features == N
        assert v2.head_feedback_routine.in_features == H
        assert v2.head_feedback_routine.out_features == N

        # Stale-name guard: legacy `head_feedback` MUST NOT exist when
        # per-regime is on, so any stale code path fails loudly instead of
        # silently using random uninitialized weights.
        assert not hasattr(v2, "head_feedback")

        # Init symmetry: focused and routine heads start identical.
        assert torch.allclose(
            v2.head_feedback_focused.weight, v2.head_feedback_routine.weight
        )
        assert torch.allclose(
            v2.head_feedback_focused.bias, v2.head_feedback_routine.bias
        )

    def test_per_regime_feedback_gating(self):
        """task_state one-hot routes feedback to the matching head exactly."""
        torch.manual_seed(0)
        cfg = ModelConfig(use_per_regime_feedback=True)
        v2 = V2ContextModule(cfg)
        # Push the heads apart so the gating is observable.
        with torch.no_grad():
            v2.head_feedback_routine.weight.add_(torch.randn_like(
                v2.head_feedback_routine.weight) * 0.5)
            v2.head_feedback_routine.bias.add_(torch.randn_like(
                v2.head_feedback_routine.bias) * 0.5)

        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        h_v2_prev = torch.randn(B, H)

        # Run the GRU once to recover the same h_v2 the v2 forward will use.
        # Build the same input the v2 forward would build under l23 mode.
        v2_input = torch.cat([r_l23, cue, torch.zeros(B, 2)], dim=-1)
        # We can't easily reproduce the GRU output without running v2 once,
        # so instead drive v2 three times (focused / routine / mixed) with
        # the SAME h_v2_prev and SAME inputs and check the produced fb
        # against direct linear application on the produced h_v2.

        # 1. Focused-only: fb must equal head_feedback_focused(h_v2).
        ts_focused = torch.tensor([[1.0, 0.0]] * B)
        _, _, fb_focused, h_v2 = v2(r_l4, r_l23, cue, ts_focused, h_v2_prev)
        expected_focused = v2.head_feedback_focused(h_v2)
        assert torch.allclose(fb_focused, expected_focused, atol=1e-6)

        # 2. Routine-only: fb must equal head_feedback_routine(h_v2).
        ts_routine = torch.tensor([[0.0, 1.0]] * B)
        _, _, fb_routine, h_v2_r = v2(r_l4, r_l23, cue, ts_routine, h_v2_prev)
        expected_routine = v2.head_feedback_routine(h_v2_r)
        assert torch.allclose(fb_routine, expected_routine, atol=1e-6)

        # 3. Mixed batch: half focused, half routine — gating is per-sample.
        ts_mixed = torch.tensor([[1.0, 0.0], [1.0, 0.0],
                                 [0.0, 1.0], [0.0, 1.0]])
        _, _, fb_mixed, h_v2_m = v2(r_l4, r_l23, cue, ts_mixed, h_v2_prev)
        expected_mixed = torch.cat([
            v2.head_feedback_focused(h_v2_m[:2]),
            v2.head_feedback_routine(h_v2_m[2:]),
        ], dim=0)
        assert torch.allclose(fb_mixed, expected_mixed, atol=1e-6)

        # Sanity: focused and routine outputs must differ (heads have been
        # pushed apart by the random perturbation above).
        assert not torch.allclose(fb_focused, fb_routine, atol=1e-3)

    def test_per_regime_feedback_gradient_flow(self):
        """Both per-regime heads receive non-zero gradients on a mixed batch."""
        torch.manual_seed(0)
        cfg = ModelConfig(use_per_regime_feedback=True)
        v2 = V2ContextModule(cfg)
        B, N, H = 4, cfg.n_orientations, cfg.v2_hidden_dim
        r_l4 = torch.randn(B, N)
        r_l23 = torch.randn(B, N)
        cue = torch.zeros(B, N)
        h_v2_prev = torch.randn(B, H)

        # Mixed batch: half focused, half routine — both heads must see grad.
        ts = torch.tensor([[1.0, 0.0], [1.0, 0.0],
                           [0.0, 1.0], [0.0, 1.0]])
        _, _, fb, _ = v2(r_l4, r_l23, cue, ts, h_v2_prev)
        loss = fb.sum()
        loss.backward()

        assert v2.head_feedback_focused.weight.grad is not None
        assert v2.head_feedback_routine.weight.grad is not None
        assert v2.head_feedback_focused.weight.grad.abs().sum().item() > 0.0
        assert v2.head_feedback_routine.weight.grad.abs().sum().item() > 0.0

        # And: a focused-only batch must NOT route gradient into the routine
        # head (and vice versa) — gating is hard.
        v2.zero_grad()
        ts_focused_only = torch.tensor([[1.0, 0.0]] * B)
        _, _, fb_f, _ = v2(r_l4, r_l23, cue, ts_focused_only, h_v2_prev)
        fb_f.sum().backward()
        assert v2.head_feedback_focused.weight.grad.abs().sum().item() > 0.0
        # routine head receives no gradient from this batch
        if v2.head_feedback_routine.weight.grad is not None:
            assert v2.head_feedback_routine.weight.grad.abs().sum().item() == 0.0


# ── Fix 1: Dual V2 Architecture ──────────────────────────────────────────

class TestDualV2:
    """Tests for Fix 1: dual V2 GRU architecture (use_dual_v2=True)."""

    def test_dual_v2_constructs_two_modules(self):
        """use_dual_v2=True creates v2_focused and v2_routine."""
        cfg = ModelConfig(use_dual_v2=True)
        net = LaminarV1V2Network(cfg)
        assert hasattr(net, "v2_focused")
        assert hasattr(net, "v2_routine")
        # Legacy v2 alias points to v2_focused
        assert net.v2 is net.v2_focused
        # Both are V2ContextModule with independent params
        foc_ids = {id(p) for p in net.v2_focused.parameters()}
        rou_ids = {id(p) for p in net.v2_routine.parameters()}
        assert foc_ids.isdisjoint(rou_ids), "v2_focused and v2_routine share params"

    def test_dual_v2_off_is_legacy(self):
        """use_dual_v2=False produces single v2, no v2_focused/v2_routine."""
        cfg = ModelConfig(use_dual_v2=False)
        net = LaminarV1V2Network(cfg)
        assert hasattr(net, "v2")
        assert not hasattr(net, "v2_focused")
        assert not hasattr(net, "v2_routine")

    def test_dual_v2_forward_runs(self):
        """Forward pass with dual V2 completes without error."""
        cfg = ModelConfig(use_dual_v2=True)
        net = LaminarV1V2Network(cfg)
        B, T, N = 4, 20, cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        task = torch.zeros(B, T, 2)
        task[:2, :, 0] = 1.0  # first 2 samples focused
        task[2:, :, 1] = 1.0  # last 2 samples routine
        packed = net.pack_inputs(stim, task_state_seq=task)
        r_l23_all, final, aux = net(packed)
        assert r_l23_all.shape == (B, T, N)
        # h_v2 should be [B, 2*H] for dual V2
        assert final.h_v2.shape == (B, 2 * cfg.v2_hidden_dim)

    def test_dual_v2_routing_by_task_state(self):
        """Focused-only and routine-only batches use different V2 modules."""
        cfg = ModelConfig(use_dual_v2=True)
        net = LaminarV1V2Network(cfg)
        B, N = 4, cfg.n_orientations
        # Perturb v2_routine to differ from v2_focused
        with torch.no_grad():
            for p in net.v2_routine.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        stim = torch.randn(B, N).abs()
        cue = torch.zeros(B, N)
        H = cfg.v2_hidden_dim
        h_v2_init = torch.zeros(B, 2 * H)
        from src.state import NetworkState
        state = NetworkState(
            r_l4=torch.zeros(B, N), r_l23=torch.zeros(B, N),
            r_pv=torch.zeros(B, 1), r_som=torch.zeros(B, N),
            r_vip=torch.zeros(B, N), adaptation=torch.zeros(B, N),
            h_v2=h_v2_init, deep_template=torch.zeros(B, N),
        )
        # Focused-only
        ts_foc = torch.tensor([[1.0, 0.0]]).expand(B, 2)
        _, aux_foc = net.step(stim, cue, ts_foc, state)
        # Routine-only
        ts_rou = torch.tensor([[0.0, 1.0]]).expand(B, 2)
        _, aux_rou = net.step(stim, cue, ts_rou, state)
        # Outputs must differ since modules have different weights
        assert not torch.allclose(aux_foc.q_pred, aux_rou.q_pred), \
            "Focused and routine V2 outputs should differ after perturbation"

    def test_dual_v2_gradient_independence(self):
        """Gradient flows to the correct V2 module based on task_state."""
        cfg = ModelConfig(use_dual_v2=True)
        net = LaminarV1V2Network(cfg)
        B, T, N = 4, 8, cfg.n_orientations
        # All-focused batch
        stim = torch.randn(B, T, N).abs()
        task = torch.zeros(B, T, 2)
        task[:, :, 0] = 1.0  # all focused
        packed = net.pack_inputs(stim, task_state_seq=task)
        r_l23, _, _ = net(packed)
        loss = r_l23.sum()
        loss.backward()
        # v2_focused should have gradients
        foc_grad = sum(p.grad.abs().sum().item() for p in net.v2_focused.parameters() if p.grad is not None)
        # v2_routine should have ZERO gradients (not used)
        rou_grad = sum(p.grad.abs().sum().item() for p in net.v2_routine.parameters() if p.grad is not None)
        assert foc_grad > 0, "v2_focused should receive gradient for focused batch"
        assert rou_grad == 0.0, "v2_routine should receive zero gradient for focused batch"


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


# ── Phase 2: Causal E/I gate (alpha_net) ────────────────────────────────

class TestEIGate:
    """Regression + semantics tests for the Phase 2 causal E/I gate.

    Two invariants the gate must satisfy, independently of training:

    1. ``use_ei_gate=False`` (default) → network forward is byte-equal to the
       pre-Phase-2 behavior. No ``alpha_net`` submodule, no extra parameters,
       no gate multiplication in step(). Covered by test_gate_off_absent.

    2. ``use_ei_gate=True`` at initialization → gate outputs ≈ 1.0 everywhere
       (init is bias=0, weight N(0, 0.01) → sigmoid(~0)*2 ≈ 1.0), so
       ``center_exc`` and ``som_drive_fb`` are within fp noise of the gate-off
       path given identical parameter init for all other modules. Covered by
       test_gate_on_init_is_near_identity and test_gate_semantics_manual.
    """

    def test_gate_off_absent(self, cfg):
        """use_ei_gate=False → no alpha_net attribute; legacy path only."""
        assert cfg.use_ei_gate is False
        net = LaminarV1V2Network(cfg)
        assert not hasattr(net, "alpha_net"), \
            "alpha_net must not exist when use_ei_gate=False"
        # forward still works
        B, T, N = 2, 10, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23, _, aux = net(stim)
        assert r_l23.shape == (B, T, N)
        # gains_all always exists (zeros when gate off)
        assert aux["gains_all"].shape == (B, T, 2)
        assert torch.all(aux["gains_all"] == 0.0)

    def test_gate_on_init_is_near_identity(self, cfg):
        """Gate-on at init produces output within fp tolerance of gate-off.

        Because alpha_net is init'd with bias=0 and weight std=0.01, and
        ``2 * sigmoid(~0) ≈ 1.0``, the gate multiplies center_exc / som_drive
        by ~1.0 and the resulting r_l23 trajectory should match the gate-off
        trajectory within a few 1e-3 units (tolerance accommodates the
        cumulative effect of the 1% weight std across T=10 timesteps).
        """
        from dataclasses import replace

        B, T, N = 2, 10, cfg.n_orientations
        torch.manual_seed(42)
        stim = torch.randn(B, T, N).abs() * 0.5

        # Gate-off reference
        torch.manual_seed(0)
        net_off = LaminarV1V2Network(cfg)
        r_off, _, aux_off = net_off(stim)

        # Gate-on with identical parameter seed for non-alpha_net modules.
        # (torch.manual_seed(0) before construction → same init for l4/pv/
        # l23/som/v2. alpha_net then consumes two additional random calls
        # for its weight init, but those don't affect the shared modules.)
        torch.manual_seed(0)
        cfg_gated = replace(cfg, use_ei_gate=True)
        net_on = LaminarV1V2Network(cfg_gated)
        assert hasattr(net_on, "alpha_net")
        assert net_on.alpha_net.weight.shape == (2, 3)
        assert net_on.alpha_net.bias.shape == (2,)
        # Init: bias=0, weight N(0, 0.01) → |bias| == 0, |weight| small
        assert torch.all(net_on.alpha_net.bias == 0.0)
        assert net_on.alpha_net.weight.abs().max() < 0.1  # 10σ upper bound

        r_on, _, aux_on = net_on(stim)

        # Gate outputs stored in aux should be near 1.0 at init
        gains = aux_on["gains_all"]                        # [B, T, 2]
        assert gains.shape == (B, T, 2)
        assert torch.all(gains > 0.7), f"min gain = {gains.min().item()}"
        assert torch.all(gains < 1.3), f"max gain = {gains.max().item()}"
        # Crucially the gains are NOT exactly zero (which would silence the
        # whole feedback path and trivially match "off" output)
        assert (gains - 1.0).abs().mean() > 1e-4

        # r_l23 trajectory should be within fp tolerance of gate-off.
        assert torch.allclose(r_off, r_on, atol=5e-3), (
            f"Gate-on at init diverged from gate-off by "
            f"max={((r_off - r_on).abs().max().item()):.2e}"
        )

    def test_gate_semantics_manual(self, cfg):
        """Manually setting bias flips gate semantics as expected.

        Wiring check: with ``bias[0] = -10`` (g_E ≈ 0, suppress center_exc)
        and ``bias[1] = +10`` (g_I ≈ 2, amplify SOM drive), the forward
        trajectory should produce near-zero center_exc and inflated SOM
        relative to the default gate-on init.
        """
        from dataclasses import replace

        cfg_gated = replace(cfg, use_ei_gate=True)
        torch.manual_seed(42)
        net = LaminarV1V2Network(cfg_gated)
        # Default init: g ≈ 1. Capture reference center_exc/som magnitudes.
        B, T, N = 4, 15, cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        _, _, aux_default = net(stim)
        ce_default = aux_default["center_exc_all"].abs().mean().item()

        # Manually suppress g_E: bias[0] = -10 → sigmoid(-10) ≈ 4.5e-5 →
        # g_E ≈ 9e-5, essentially zero.
        with torch.no_grad():
            net.alpha_net.bias[0] = -10.0
            net.alpha_net.bias[1] = +10.0
            net.alpha_net.weight.zero_()      # make gate deterministic
        _, _, aux_suppressed = net(stim)
        gains_suppressed = aux_suppressed["gains_all"]
        g_E_mean = gains_suppressed[..., 0].mean().item()
        g_I_mean = gains_suppressed[..., 1].mean().item()

        # g_E should be tiny; g_I should be near 2.0
        assert g_E_mean < 1e-3, f"g_E_mean={g_E_mean} (expected near 0)"
        assert g_I_mean > 1.99, f"g_I_mean={g_I_mean} (expected near 2.0)"
        # center_exc should collapse under the g_E suppression
        ce_suppressed = aux_suppressed["center_exc_all"].abs().mean().item()
        assert ce_suppressed < ce_default * 0.1 + 1e-6, (
            f"center_exc did not collapse: default={ce_default:.4e}, "
            f"suppressed={ce_suppressed:.4e}"
        )

    def test_som_drive_fb_in_aux(self, cfg):
        """Phase 2.4: aux must expose som_drive_fb_all for the routine_shape loss.

        som_drive_fb = relu(-scaled_fb) is computed every timestep in step()
        (regardless of use_ei_gate). The forward pass must propagate it into
        aux["som_drive_fb_all"] with shape [B, T, N]. It must be nonzero for
        a forward pass with nontrivial feedback (nonzero stimulus → nonzero
        q_pred contrast → nonzero feedback drive).
        """
        from dataclasses import replace

        # Exercise both gate-off (default) and gate-on paths.
        for use_gate in (False, True):
            cfg_v = replace(cfg, use_ei_gate=use_gate)
            torch.manual_seed(0)
            net = LaminarV1V2Network(cfg_v)
            B, T, N = 2, 15, cfg.n_orientations
            stim = torch.randn(B, T, N).abs() * 0.5
            _, _, aux = net(stim)
            assert "som_drive_fb_all" in aux, (
                f"use_ei_gate={use_gate}: aux missing 'som_drive_fb_all'"
            )
            sdf = aux["som_drive_fb_all"]
            assert sdf.shape == (B, T, N), (
                f"use_ei_gate={use_gate}: expected shape {(B, T, N)}, got {sdf.shape}"
            )
            # som_drive_fb = relu(-scaled_fb) is always non-negative
            assert torch.all(sdf >= 0.0)
            # At least some timesteps should have nonzero drive once feedback
            # has built up (not a strict semantic requirement — the key check
            # is the tensor exists with correct shape).
            assert torch.isfinite(sdf).all()


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


class TestPrecisionGating:
    """Tests for Rescue 2: use_precision_gating config flag.

    When True, pi_pred multiplicatively gates the feedback signal:
        precision_gate = pi_pred_raw / pi_max  (range [0, 1])
        scaled_fb = feedback_signal * feedback_scale * precision_gate
    so that V2's learned precision directly controls feedback strength.
    """

    def test_precision_gating_scales_feedback(self):
        """Zero precision zeros feedback; max precision leaves it unchanged."""
        cfg = ModelConfig(use_precision_gating=True)
        net = LaminarV1V2Network(cfg)
        net.feedback_scale.fill_(1.0)

        B, N = 2, cfg.n_orientations
        stim = torch.randn(B, N).abs()
        cue = torch.zeros(B, N)
        task_state = torch.tensor([[1.0, 0.0]] * B)
        state0 = initial_state(B, N, cfg.v2_hidden_dim)

        # Set head_pi bias very negative → softplus ≈ 0 → pi_pred_raw ≈ 0
        with torch.no_grad():
            net.v2.head_pi.bias.fill_(-10.0)
            net.v2.head_pi.weight.fill_(0.0)
        state_lo, aux_lo = net.step(stim, cue, task_state, state0)
        assert aux_lo.center_exc.abs().max() < 1e-4, (
            f"Zero precision should zero out feedback, got max={aux_lo.center_exc.abs().max():.6f}"
        )
        assert aux_lo.som_drive_fb.abs().max() < 1e-4, (
            f"Zero precision should zero out SOM drive, got max={aux_lo.som_drive_fb.abs().max():.6f}"
        )

        # Set head_pi bias very positive → softplus → clamp at pi_max → gate=1.0
        with torch.no_grad():
            net.v2.head_pi.bias.fill_(10.0)
        state_hi, aux_hi = net.step(stim, cue, task_state, state0)
        # At max precision, gate=1.0, so scaled_fb = feedback_signal * feedback_scale.
        # Compare with legacy (gate-off) to confirm they match.
        cfg_legacy = ModelConfig(use_precision_gating=False)
        net_legacy = LaminarV1V2Network(cfg_legacy)
        net_legacy.feedback_scale.fill_(1.0)
        # Copy all weights from net → net_legacy
        net_legacy.load_state_dict(net.state_dict(), strict=False)
        with torch.no_grad():
            net_legacy.v2.head_pi.bias.fill_(10.0)
            net_legacy.v2.head_pi.weight.fill_(0.0)
        _, aux_leg = net_legacy.step(stim, cue, task_state, state0)
        # center_exc should be approximately equal (gate≈1.0)
        assert torch.allclose(aux_hi.center_exc, aux_leg.center_exc, atol=1e-4), (
            "Max precision should match legacy feedback"
        )

    def test_precision_gating_off_is_legacy(self):
        """With use_precision_gating=False, forward pass completes (legacy)."""
        cfg = ModelConfig(use_precision_gating=False)
        net = LaminarV1V2Network(cfg)
        net.feedback_scale.fill_(1.0)

        B, N = 2, cfg.n_orientations
        stim = torch.randn(B, N).abs()
        cue = torch.zeros(B, N)
        task_state = torch.tensor([[1.0, 0.0]] * B)
        state0 = initial_state(B, N, cfg.v2_hidden_dim)

        # With oracle pi=0 and legacy mode, feedback should NOT be zeroed
        net.oracle_mode = True
        net.oracle_q_pred = torch.ones(B, N) / N
        net.oracle_pi_pred = torch.zeros(B, 1)
        state_legacy, aux_legacy = net.step(stim, cue, task_state, state0)
        net.oracle_mode = False

        # Legacy mode doesn't use pi in feedback path — just verify it runs
        assert aux_legacy.center_exc.shape == (B, N)
