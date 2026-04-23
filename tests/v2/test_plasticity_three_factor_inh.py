"""Closed-form tests for ``ThreeFactorRule.delta_mh_inh`` (Task #74 Fix C).

The SOM-routed task readout uses a three-factor rule:

    dw[j, i] = η · mean_b(som_modulator[b, j] · memory[b, i])
             − λ · (weights[j, i] − raw_prior[j, i])

then clamped to ±0.01 and masked. These tests confirm:

* shape/dtype preservation;
* exact agreement with the closed-form on a hand-constructed batch;
* the ±0.01 clamp is actually applied;
* ``weight_decay`` shrinks toward ``raw_prior`` (or 0 if absent);
* the optional sparsity mask zeroes the specified entries.
"""

from __future__ import annotations

import torch

from src.v2_model.plasticity import ThreeFactorRule


def _rule(lr: float = 0.1, weight_decay: float = 0.0) -> ThreeFactorRule:
    return ThreeFactorRule(lr=lr, weight_decay=weight_decay)


def test_shape_and_dtype_preserved() -> None:
    rule = _rule()
    B, n_m, n_som = 4, 5, 3
    memory = torch.randn(B, n_m, dtype=torch.float32)
    som_mod = torch.randn(B, n_som, dtype=torch.float32)
    weights = torch.zeros(n_som, n_m, dtype=torch.float32)
    dw = rule.delta_mh_inh(memory, som_mod, weights)
    assert dw.shape == (n_som, n_m)
    assert dw.dtype == torch.float32


def test_closed_form_matches_batch_outer_mean() -> None:
    """dw = η · mean_b(som_mod[b] ⊗ memory[b]) when λ = 0 and |dw| < 0.01."""
    rule = _rule(lr=0.01, weight_decay=0.0)                # small η keeps |dw| < clamp
    B, n_m, n_som = 3, 4, 2
    torch.manual_seed(0)
    memory = torch.randn(B, n_m) * 0.1                     # small ⇒ no clamp hit
    som_mod = torch.randn(B, n_som) * 0.1
    weights = torch.zeros(n_som, n_m)
    dw = rule.delta_mh_inh(memory, som_mod, weights)

    # Expected via explicit batch outer-product mean.
    expected = torch.zeros(n_som, n_m)
    for b in range(B):
        expected += torch.outer(som_mod[b], memory[b])
    expected = rule.lr * expected / B

    torch.testing.assert_close(dw, expected, atol=1e-7, rtol=1e-6)


def test_clamp_applied_at_pm_001() -> None:
    """Large η × large activity ⇒ result must saturate at ±0.01 per element."""
    rule = _rule(lr=100.0, weight_decay=0.0)
    B, n_m, n_som = 2, 3, 2
    memory = torch.ones(B, n_m)
    som_mod = torch.ones(B, n_som)                         # all-positive ⇒ hits +0.01
    weights = torch.zeros(n_som, n_m)
    dw = rule.delta_mh_inh(memory, som_mod, weights)
    assert torch.all(dw == 0.01)

    som_mod_neg = -torch.ones(B, n_som)                    # negative ⇒ -0.01
    dw_neg = rule.delta_mh_inh(memory, som_mod_neg, weights)
    assert torch.all(dw_neg == -0.01)


def test_weight_decay_shrinks_toward_zero_without_prior() -> None:
    """η = 0, λ > 0, no raw_prior ⇒ dw = -λ · weights (decay toward 0)."""
    lr = 1e-6                                              # effectively 0
    rule = ThreeFactorRule(lr=lr, weight_decay=0.05)
    B, n_m, n_som = 2, 3, 2
    memory = torch.zeros(B, n_m)                           # kills η term
    som_mod = torch.zeros(B, n_som)
    weights = torch.full((n_som, n_m), 0.1)                # below clamp after × 0.05
    dw = rule.delta_mh_inh(memory, som_mod, weights)
    expected = -0.05 * weights
    torch.testing.assert_close(dw, expected, atol=1e-7, rtol=0.0)


def test_weight_decay_shrinks_toward_raw_prior() -> None:
    """η = 0, λ > 0, raw_prior given ⇒ dw = -λ · (weights − raw_prior)."""
    lr = 1e-6
    rule = ThreeFactorRule(lr=lr, weight_decay=0.05)
    B, n_m, n_som = 2, 3, 2
    memory = torch.zeros(B, n_m)
    som_mod = torch.zeros(B, n_som)
    weights = torch.full((n_som, n_m), 0.15)
    raw_prior = torch.full((n_som, n_m), 0.05)
    dw = rule.delta_mh_inh(memory, som_mod, weights, raw_prior=raw_prior)
    expected = -0.05 * (weights - raw_prior)
    torch.testing.assert_close(dw, expected, atol=1e-7, rtol=0.0)


def test_mask_zeros_specified_entries() -> None:
    """Masked positions must be 0 regardless of Hebbian / decay contribution."""
    rule = _rule(lr=0.01, weight_decay=0.0)
    B, n_m, n_som = 3, 4, 2
    torch.manual_seed(1)
    memory = torch.randn(B, n_m) * 0.1
    som_mod = torch.randn(B, n_som) * 0.1
    weights = torch.zeros(n_som, n_m)
    mask = torch.ones(n_som, n_m, dtype=torch.bool)
    mask[0, 0] = False                                     # block this entry
    mask[1, 2] = False
    dw = rule.delta_mh_inh(memory, som_mod, weights, mask=mask)
    assert dw[0, 0] == 0.0
    assert dw[1, 2] == 0.0


def test_sign_convention_positive_modulator_strengthens() -> None:
    """Positive modulator + positive memory ⇒ positive weight update (strengthen)."""
    rule = _rule(lr=0.01, weight_decay=0.0)
    B, n_m, n_som = 4, 3, 2
    memory = torch.full((B, n_m), 0.1)
    som_mod = torch.full((B, n_som), 0.1)
    weights = torch.zeros(n_som, n_m)
    dw = rule.delta_mh_inh(memory, som_mod, weights)
    assert torch.all(dw > 0.0), "positive modulator × positive memory must strengthen"
