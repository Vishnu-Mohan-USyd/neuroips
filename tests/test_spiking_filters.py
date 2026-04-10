"""Unit tests for the exponential spike trace filter (src/spiking/filters.py).

Covers:
    1. tau <-> alpha conversions (forward and round-trip).
    2. Single-step update formula.
    3. Sequence filter correctness:
       - impulse response is an exponential decay,
       - constant input steady state matches 1/(1-alpha),
       - shape, dtype, and device preservation,
       - equivalence with per-step update.
    4. Module wrapper:
       - fixed tau stores alpha correctly,
       - learnable tau is a Parameter and gradient flows,
       - Module forward matches the functional form.
    5. torch.compile compatibility.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.spiking.filters import (
    ExponentialTraceFilter,
    alpha_from_tau,
    exp_filter_sequence,
    exp_filter_step,
    tau_from_alpha,
)


# ----------------------------------------------------------------------------
# 1. tau <-> alpha conversions
# ----------------------------------------------------------------------------

class TestAlphaTauConversion:
    def test_plan_values(self):
        """Tau = 10 -> alpha ~= 0.905 (plan line 48)."""
        alpha = alpha_from_tau(10.0)
        assert abs(alpha - math.exp(-0.1)) < 1e-10
        assert abs(alpha - 0.9048374180359595) < 1e-10  # exact value

    def test_dt_scaling(self):
        # alpha = exp(-dt/tau): halving dt halves the effective exponent
        a1 = alpha_from_tau(10.0, dt=1.0)
        a2 = alpha_from_tau(10.0, dt=0.5)
        assert a2 > a1  # smaller dt -> alpha closer to 1 (less decay per step)
        assert abs(a2 - math.exp(-0.05)) < 1e-10

    def test_round_trip(self):
        for tau in [1.0, 5.0, 10.0, 100.0, 500.0]:
            alpha = alpha_from_tau(tau)
            tau_back = tau_from_alpha(alpha)
            assert abs(tau_back - tau) < 1e-8

    def test_rejects_nonpositive_tau(self):
        with pytest.raises(ValueError, match="tau must be > 0"):
            alpha_from_tau(0.0)
        with pytest.raises(ValueError, match="tau must be > 0"):
            alpha_from_tau(-1.0)

    def test_rejects_nonpositive_dt(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            alpha_from_tau(10.0, dt=0.0)

    def test_rejects_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            tau_from_alpha(0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            tau_from_alpha(1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            tau_from_alpha(1.5)


# ----------------------------------------------------------------------------
# 2. Single-step update
# ----------------------------------------------------------------------------

class TestExpFilterStep:
    def test_basic_update(self):
        x_prev = torch.tensor([1.0, 2.0, 3.0])
        z = torch.tensor([0.0, 1.0, 0.0])
        alpha = 0.9
        x_new = exp_filter_step(x_prev, z, alpha)
        expected = torch.tensor([0.9, 1.8 + 1.0, 2.7])
        torch.testing.assert_close(x_new, expected)

    def test_preserves_shape(self):
        for shape in [(4,), (2, 8), (4, 12, 36)]:
            x_prev = torch.randn(*shape)
            z = torch.zeros(*shape)
            out = exp_filter_step(x_prev, z, 0.9)
            assert out.shape == shape

    def test_preserves_dtype(self):
        for dtype in [torch.float32, torch.float64]:
            x_prev = torch.randn(10, dtype=dtype)
            z = torch.zeros(10, dtype=dtype)
            out = exp_filter_step(x_prev, z, 0.9)
            assert out.dtype == dtype


# ----------------------------------------------------------------------------
# 3. Sequence filter correctness
# ----------------------------------------------------------------------------

class TestExpFilterSequence:
    def test_impulse_response_is_exponential_decay(self):
        """Input [1, 0, 0, 0, ...] -> output [1, alpha, alpha^2, alpha^3, ...]."""
        alpha = 0.9
        T = 20
        spikes = torch.zeros(1, T, 1)
        spikes[0, 0, 0] = 1.0
        out = exp_filter_sequence(spikes, alpha)
        expected = torch.tensor([alpha**t for t in range(T)]).reshape(1, T, 1)
        torch.testing.assert_close(out, expected, atol=1e-7, rtol=1e-7)

    def test_constant_input_steady_state(self):
        """Constant z[t]=1 drives x -> 1/(1-alpha) as T -> infinity."""
        alpha = 0.9
        T = 500  # long enough to saturate
        spikes = torch.ones(1, T, 1)
        out = exp_filter_sequence(spikes, alpha)
        steady_state = 1.0 / (1.0 - alpha)  # = 10.0
        # Last timestep should be very close to steady state
        assert abs(out[0, -1, 0].item() - steady_state) < 1e-4

    def test_steady_state_plan_tau_10(self):
        """Plan value: tau=10, alpha~0.905 -> x* ~ 1/(1-0.905) ~ 10.52."""
        alpha = alpha_from_tau(10.0)
        T = 1000
        spikes = torch.ones(2, T, 4)
        out = exp_filter_sequence(spikes, alpha)
        expected_ss = 1.0 / (1.0 - alpha)
        torch.testing.assert_close(
            out[:, -1, :],
            torch.full((2, 4), expected_ss),
            atol=1e-4, rtol=1e-4,
        )

    def test_step_sequence_equivalence(self):
        """exp_filter_sequence matches iterated exp_filter_step."""
        alpha = 0.8
        B, T, N = 2, 15, 5
        spikes = (torch.rand(B, T, N) > 0.5).float()

        out_seq = exp_filter_sequence(spikes, alpha)

        x = torch.zeros(B, N)
        out_iter = torch.empty_like(spikes)
        for t in range(T):
            x = exp_filter_step(x, spikes[:, t], alpha)
            out_iter[:, t] = x

        torch.testing.assert_close(out_seq, out_iter)

    def test_initial_state_respected(self):
        """Non-zero x0 propagates forward."""
        alpha = 0.5
        B, T, N = 1, 3, 2
        spikes = torch.zeros(B, T, N)  # no spikes
        x0 = torch.tensor([[1.0, 2.0]])
        out = exp_filter_sequence(spikes, alpha, x0=x0)
        # With no spikes: x[t] = alpha * x[t-1] -> decays from x0
        expected = torch.tensor([
            [[0.5, 1.0]],       # t=0: 0.5 * (1.0, 2.0)
            [[0.25, 0.5]],      # t=1: 0.5 * previous
            [[0.125, 0.25]],    # t=2
        ]).transpose(0, 1)
        torch.testing.assert_close(out, expected)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"spikes must be \[B, T, N\]"):
            exp_filter_sequence(torch.randn(4, 36), alpha=0.9)

    def test_rejects_wrong_x0_shape(self):
        spikes = torch.zeros(2, 5, 3)
        with pytest.raises(ValueError, match="x0 must have shape"):
            exp_filter_sequence(spikes, alpha=0.9, x0=torch.zeros(4, 3))


# ----------------------------------------------------------------------------
# 4. Module wrapper
# ----------------------------------------------------------------------------

class TestExponentialTraceFilterFixed:
    def test_alpha_matches_plan(self):
        f = ExponentialTraceFilter(tau=10.0)
        assert abs(f.alpha.item() - math.exp(-0.1)) < 1e-7

    def test_tau_is_buffer_not_parameter(self):
        f = ExponentialTraceFilter(tau=10.0, learnable=False)
        params = list(f.parameters())
        assert len(params) == 0, "fixed mode should have no learnable params"
        # Moves with .to()
        assert f._tau_fixed.dtype == torch.float32

    def test_forward_matches_functional(self):
        tau = 10.0
        alpha = alpha_from_tau(tau)
        f = ExponentialTraceFilter(tau=tau)

        x_prev = torch.randn(4, 36)
        spikes = (torch.rand(4, 36) > 0.5).float()
        out_mod = f(spikes, x_prev)
        out_fn = exp_filter_step(x_prev, spikes, alpha)
        torch.testing.assert_close(out_mod, out_fn)

    def test_repr_contains_tau_and_alpha(self):
        f = ExponentialTraceFilter(tau=10.0)
        r = repr(f)
        assert "tau=10" in r
        assert "alpha=0.90" in r


class TestExponentialTraceFilterLearnable:
    def test_learnable_creates_parameter(self):
        f = ExponentialTraceFilter(tau=10.0, learnable=True)
        params = list(f.parameters())
        assert len(params) == 1
        assert params[0].requires_grad

    def test_initial_tau_correct(self):
        f = ExponentialTraceFilter(tau=10.0, learnable=True)
        assert abs(f.tau.item() - 10.0) < 1e-4

    def test_gradient_flows_through_alpha(self):
        """Updating tau via backprop works end-to-end."""
        f = ExponentialTraceFilter(tau=10.0, learnable=True)

        spikes = torch.ones(1, 1) * 2.0
        x_prev = torch.zeros(1, 1)
        out = f(spikes, x_prev)  # = alpha * 0 + 2 = 2 (no tau dep here)

        # Use a case where output depends on tau
        x_prev = torch.ones(1, 1) * 3.0
        spikes = torch.zeros(1, 1)
        out = f(spikes, x_prev)  # = alpha * 3 + 0 = 3 * exp(-1/tau)
        out.sum().backward()

        assert f.tau_raw.grad is not None
        assert torch.isfinite(f.tau_raw.grad).all()
        assert f.tau_raw.grad.abs().item() > 0

    def test_softplus_keeps_tau_positive(self):
        """Even if tau_raw goes very negative, tau stays > 0."""
        f = ExponentialTraceFilter(tau=10.0, learnable=True)
        with torch.no_grad():
            f.tau_raw.fill_(-50.0)
        # softplus(-50) ~ 1.9e-22 > 0
        assert f.tau.item() > 0.0


# ----------------------------------------------------------------------------
# 5. torch.compile compatibility
# ----------------------------------------------------------------------------

class TestTorchCompile:
    def test_step_compiles(self):
        alpha = 0.9

        def fn(x_prev, spikes):
            return exp_filter_step(x_prev, spikes, alpha)

        compiled = torch.compile(fn, fullgraph=True)
        x_prev = torch.randn(4, 36)
        spikes = torch.zeros(4, 36)
        out = compiled(x_prev, spikes)

        torch.testing.assert_close(out, exp_filter_step(x_prev, spikes, alpha))

    def test_module_compiles(self):
        f = ExponentialTraceFilter(tau=10.0)
        compiled = torch.compile(f, fullgraph=True)

        x_prev = torch.randn(2, 36)
        spikes = (torch.rand(2, 36) > 0.5).float()
        out = compiled(spikes, x_prev)

        torch.testing.assert_close(out, f(spikes, x_prev))
