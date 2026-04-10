"""Unit tests for the ATan surrogate gradient (src/spiking/surrogate.py).

Covers the four acceptance criteria from task #11:
    1. Forward returns binary {0, 1} values at threshold crossings.
    2. Backward returns finite, non-zero gradients near threshold.
    3. Gradient magnitude matches the analytical ATan derivative within 1e-5.
    4. Works under torch.compile without errors.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.spiking.surrogate import ATanSurrogate, atan_surrogate


# ----------------------------------------------------------------------------
# 1. Forward pass: binary {0, 1}
# ----------------------------------------------------------------------------

class TestForwardBinary:
    def test_output_is_binary(self):
        """Output values lie strictly in {0, 1}."""
        x = torch.randn(1000) * 3.0
        out = atan_surrogate(x, slope=25.0)
        unique = set(out.unique().tolist())
        assert unique.issubset({0.0, 1.0}), f"non-binary outputs: {unique}"

    def test_positive_inputs_spike(self):
        x = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])
        out = atan_surrogate(x, slope=25.0)
        assert torch.all(out == 1.0)

    def test_negative_inputs_silent(self):
        x = torch.tensor([-0.01, -0.1, -1.0, -10.0, -100.0])
        out = atan_surrogate(x, slope=25.0)
        assert torch.all(out == 0.0)

    def test_exact_threshold_is_zero(self):
        """At x == 0 we emit 0 (strict `>`, matches snnTorch ATan line 190)."""
        x = torch.zeros(10)
        out = atan_surrogate(x, slope=25.0)
        assert torch.all(out == 0.0)

    def test_output_shape_preserved(self):
        for shape in [(4,), (4, 36), (8, 12, 36), (2, 3, 4, 5)]:
            x = torch.randn(*shape)
            out = atan_surrogate(x, slope=25.0)
            assert out.shape == x.shape

    def test_output_dtype_preserved(self):
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(10, dtype=dtype)
            out = atan_surrogate(x, slope=25.0)
            assert out.dtype == dtype


# ----------------------------------------------------------------------------
# 2. Backward pass: finite, non-zero near threshold
# ----------------------------------------------------------------------------

class TestBackwardFiniteNonZero:
    def test_gradient_finite_everywhere(self):
        x = torch.linspace(-10.0, 10.0, 201, requires_grad=True)
        out = atan_surrogate(x, slope=25.0)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_gradient_nonzero_near_threshold(self):
        """Within the active region (|x| small relative to 1/(pi*slope)), gradient > 0."""
        slope = 25.0
        # Active half-width at which gradient drops to half max:
        # 1/(1 + (pi*slope*x)^2) = 0.5  =>  x = 1/(pi*slope)
        hw = 1.0 / (math.pi * slope)
        x = torch.linspace(-hw, hw, 21, requires_grad=True)
        out = atan_surrogate(x, slope=slope)
        out.sum().backward()
        assert (x.grad > 0).all()

    def test_gradient_max_at_zero(self):
        """Gradient peak is at x = 0 and equals 1/pi, independent of slope."""
        for slope in [0.5, 1.0, 5.0, 25.0, 100.0]:
            x = torch.zeros(1, requires_grad=True)
            out = atan_surrogate(x, slope=slope)
            out.sum().backward()
            torch.testing.assert_close(
                x.grad, torch.tensor([1.0 / math.pi]), atol=1e-6, rtol=1e-6
            )


# ----------------------------------------------------------------------------
# 3. Gradient matches analytical formula within 1e-5
# ----------------------------------------------------------------------------

def analytical_atan_grad(x: torch.Tensor, slope: float) -> torch.Tensor:
    """Reference formula: (1/pi) * 1 / (1 + (pi * slope * x)^2)."""
    return (1.0 / math.pi) / (1.0 + (math.pi * slope * x).pow(2))


class TestAnalyticalGradientMatch:
    @pytest.mark.parametrize("slope", [1.0, 2.0, 5.0, 25.0])
    def test_matches_analytical_formula(self, slope):
        x = torch.linspace(-2.0, 2.0, 201, requires_grad=True)
        out = atan_surrogate(x, slope=slope)
        out.sum().backward()

        expected = analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)

    def test_matches_analytical_formula_2d(self):
        """[B, N] shape (main use case: V_mem - V_thresh for a population)."""
        slope = 25.0
        x = torch.randn(4, 36, requires_grad=True)
        out = atan_surrogate(x, slope=slope)
        out.sum().backward()

        expected = analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)

    def test_chain_rule_composition(self):
        """Gradient composes correctly when the surrogate output feeds further ops."""
        slope = 25.0
        x = torch.randn(8, requires_grad=True)
        z = atan_surrogate(x, slope=slope)
        # Downstream: weighted sum -> loss
        w = torch.arange(1, 9, dtype=torch.float32)
        loss = (w * z).sum()
        loss.backward()

        # By chain rule: dloss/dx = w * d(z)/dx = w * analytical_grad
        expected = w * analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------
# 4. torch.compile compatibility
# ----------------------------------------------------------------------------

class TestTorchCompile:
    def test_functional_compiles_and_runs(self):
        """atan_surrogate survives torch.compile forward pass."""
        slope = 25.0

        def fn(x: torch.Tensor) -> torch.Tensor:
            return atan_surrogate(x, slope=slope)

        compiled = torch.compile(fn, fullgraph=True)
        x = torch.randn(4, 36)
        out = compiled(x)

        # Correctness vs. eager
        ref = fn(x)
        torch.testing.assert_close(out, ref)

    def test_functional_compiles_backward(self):
        """Backward pass works end-to-end under torch.compile."""
        slope = 25.0

        def fn(x: torch.Tensor) -> torch.Tensor:
            return atan_surrogate(x, slope=slope).sum()

        compiled = torch.compile(fn, fullgraph=True)
        x = torch.randn(4, 36, requires_grad=True)
        out = compiled(x)
        out.backward()

        assert torch.isfinite(x.grad).all()
        expected = analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)

    def test_module_compiles(self):
        """ATanSurrogate nn.Module survives torch.compile."""
        mod = ATanSurrogate(slope=25.0)
        compiled = torch.compile(mod, fullgraph=True)

        x = torch.randn(4, 36, requires_grad=True)
        out = compiled(x)
        loss = out.sum()
        loss.backward()

        assert torch.isfinite(x.grad).all()
        expected = analytical_atan_grad(x.detach(), 25.0)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------
# Module wrapper parity
# ----------------------------------------------------------------------------

class TestModuleWrapper:
    def test_module_matches_functional(self):
        slope = 7.5
        mod = ATanSurrogate(slope=slope)
        x = torch.linspace(-1, 1, 21, requires_grad=True)

        out_mod = mod(x)
        loss_mod = out_mod.sum()
        loss_mod.backward()
        grad_mod = x.grad.clone()

        x2 = x.detach().clone().requires_grad_(True)
        out_fn = atan_surrogate(x2, slope=slope)
        loss_fn = out_fn.sum()
        loss_fn.backward()
        grad_fn = x2.grad

        torch.testing.assert_close(out_mod.detach(), out_fn.detach())
        torch.testing.assert_close(grad_mod, grad_fn)

    def test_module_repr_contains_slope(self):
        mod = ATanSurrogate(slope=25.0)
        assert "slope=25" in repr(mod)


# ----------------------------------------------------------------------------
# 5. Dampening (Bellec 2018 §3)
# ----------------------------------------------------------------------------

class TestSurrogateDampening:
    """Bellec 2018 §3: multiply the pseudo-derivative by dampen < 1 for BPTT
    stability on recurrent spiking nets with long unroll horizons. The forward
    pass must be unchanged; only the gradient is scaled."""

    def test_dampen_one_is_baseline(self):
        """dampen=1.0 reproduces the original (undamped) gradient exactly."""
        slope = 25.0
        x = torch.linspace(-2.0, 2.0, 201, requires_grad=True)
        out = atan_surrogate(x, slope=slope, dampen=1.0)
        out.sum().backward()

        expected = analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)

    def test_dampen_scales_gradient_exactly(self):
        """For any dampen value, gradient = dampen * baseline (pointwise)."""
        slope = 25.0
        for dampen in [0.1, 0.3, 0.5, 0.7]:
            x = torch.linspace(-2.0, 2.0, 101, requires_grad=True)
            out = atan_surrogate(x, slope=slope, dampen=dampen)
            out.sum().backward()

            expected = dampen * analytical_atan_grad(x.detach(), slope)
            torch.testing.assert_close(x.grad, expected, atol=1e-6, rtol=1e-6)

    def test_dampen_forward_unchanged(self):
        """Forward Heaviside output is identical regardless of dampen value."""
        x = torch.randn(100) * 3.0
        ref = atan_surrogate(x, slope=25.0, dampen=1.0)
        for dampen in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
            out = atan_surrogate(x, slope=25.0, dampen=dampen)
            torch.testing.assert_close(out, ref)

    def test_dampen_zero_kills_gradient(self):
        """dampen=0 produces identically-zero gradients (useful for stop-grad)."""
        x = torch.linspace(-1.0, 1.0, 21, requires_grad=True)
        out = atan_surrogate(x, slope=25.0, dampen=0.0)
        out.sum().backward()
        assert torch.all(x.grad == 0.0)

    def test_dampen_default_preserves_existing_behavior(self):
        """The default value must be 1.0 so existing call sites are unaffected."""
        slope = 25.0
        x = torch.randn(50, requires_grad=True)
        out_default = atan_surrogate(x, slope=slope)
        out_default.sum().backward()
        default_grad = x.grad.clone()

        x2 = x.detach().clone().requires_grad_(True)
        out_explicit = atan_surrogate(x2, slope=slope, dampen=1.0)
        out_explicit.sum().backward()

        torch.testing.assert_close(default_grad, x2.grad, atol=0.0, rtol=0.0)

    def test_dampen_compiles_fullgraph(self):
        """Dampening survives torch.compile(fullgraph=True) forward+backward."""
        slope = 25.0
        dampen = 0.3

        def fn(x: torch.Tensor) -> torch.Tensor:
            return atan_surrogate(x, slope=slope, dampen=dampen).sum()

        compiled = torch.compile(fn, fullgraph=True)
        x = torch.randn(4, 36, requires_grad=True)
        out = compiled(x)
        out.backward()

        assert torch.isfinite(x.grad).all()
        expected = dampen * analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)

    def test_module_dampen_parameter(self):
        """ATanSurrogate(dampen=0.3) scales gradients by 0.3."""
        slope = 25.0
        dampen = 0.3
        mod = ATanSurrogate(slope=slope, dampen=dampen)

        x = torch.randn(8, 36, requires_grad=True)
        out = mod(x)
        out.sum().backward()

        expected = dampen * analytical_atan_grad(x.detach(), slope)
        torch.testing.assert_close(x.grad, expected, atol=1e-6, rtol=1e-6)

    def test_module_repr_contains_dampen(self):
        mod = ATanSurrogate(slope=25.0, dampen=0.3)
        r = repr(mod)
        assert "dampen=0.3" in r
        assert "slope=25" in r
