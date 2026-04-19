"""Bit-exact determinism of `EnergyPenalty` — no stochastic draws anywhere.

Both methods (`rate_penalty_delta_drive`, `current_weight_shrinkage`) are
closed-form and `@torch.no_grad()`. Repeated calls with identical inputs
must therefore return tensors that compare bit-equal (``atol=rtol=0``).
Two fresh instances with the same (α, β) must produce identical outputs
on the same inputs, and interleaving calls must not contaminate state.
"""

from __future__ import annotations

import torch

from src.v2_model.energy import EnergyPenalty


# ---------------------------------------------------------------------------
# Same call → same tensor (bit-exact)
# ---------------------------------------------------------------------------

def test_rate_penalty_bit_exact_across_calls() -> None:
    energy = EnergyPenalty(alpha=0.1, beta=0.0)
    rate = torch.rand(5, 7)
    out1 = energy.rate_penalty_delta_drive(rate)
    out2 = energy.rate_penalty_delta_drive(rate)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


def test_current_shrinkage_bit_exact_across_calls() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(6, 4)
    pre = torch.randn(3, 4)
    dw1 = energy.current_weight_shrinkage(weights, pre)
    dw2 = energy.current_weight_shrinkage(weights, pre)
    torch.testing.assert_close(dw1, dw2, atol=0.0, rtol=0.0)


def test_current_shrinkage_with_mask_bit_exact_across_calls() -> None:
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)
    weights = torch.randn(4, 5)
    pre = torch.randn(2, 5)
    mask = torch.tensor(
        [[True, False, True, True, False],
         [False, True, True, False, True],
         [True, True, False, True, False],
         [False, False, True, True, True]]
    )
    dw1 = energy.current_weight_shrinkage(weights, pre, mask=mask)
    dw2 = energy.current_weight_shrinkage(weights, pre, mask=mask)
    torch.testing.assert_close(dw1, dw2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Two instances with same config → same output
# ---------------------------------------------------------------------------

def test_two_instances_same_config_match() -> None:
    """Fresh EnergyPenalty(α, β) objects produce identical output on same inputs."""
    e1 = EnergyPenalty(alpha=0.05, beta=2e-3)
    e2 = EnergyPenalty(alpha=0.05, beta=2e-3)

    rate = torch.rand(4, 6)
    weights = torch.randn(6, 5)
    pre = torch.randn(3, 5)

    torch.testing.assert_close(
        e1.rate_penalty_delta_drive(rate),
        e2.rate_penalty_delta_drive(rate),
        atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        e1.current_weight_shrinkage(weights, pre),
        e2.current_weight_shrinkage(weights, pre),
        atol=0.0, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# Interleaved calls don't contaminate state
# ---------------------------------------------------------------------------

def test_interleaved_calls_do_not_contaminate_state() -> None:
    """Calling the rate path between two shrinkage calls (and vice versa)
    leaves outputs bit-identical — the module carries no per-call state."""
    energy = EnergyPenalty(alpha=0.1, beta=1e-3)

    weights_a = torch.randn(4, 3)
    pre_a = torch.randn(2, 3)
    rate_b = torch.rand(5, 4)
    weights_c = torch.randn(6, 5)
    pre_c = torch.randn(3, 5)

    ref_a = energy.current_weight_shrinkage(weights_a, pre_a)
    ref_c = energy.current_weight_shrinkage(weights_c, pre_c)
    ref_b = energy.rate_penalty_delta_drive(rate_b)

    # Now interleave: rate ↔ shrinkage ↔ rate ↔ shrinkage.
    _ = energy.rate_penalty_delta_drive(rate_b)
    second_a = energy.current_weight_shrinkage(weights_a, pre_a)
    _ = energy.rate_penalty_delta_drive(rate_b)
    second_c = energy.current_weight_shrinkage(weights_c, pre_c)
    second_b = energy.rate_penalty_delta_drive(rate_b)

    torch.testing.assert_close(second_a, ref_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(second_c, ref_c, atol=0.0, rtol=0.0)
    torch.testing.assert_close(second_b, ref_b, atol=0.0, rtol=0.0)


def test_no_parameters_to_track() -> None:
    """EnergyPenalty has no `nn.Parameter`s — α, β are plain floats."""
    energy = EnergyPenalty(alpha=0.1, beta=1e-3)
    assert list(energy.parameters()) == []
    assert list(energy.buffers()) == []
