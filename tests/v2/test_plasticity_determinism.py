"""Determinism: identical inputs ⇒ bit-identical outputs for every rule.

The plasticity rules draw no random numbers internally, so each call is a
pure function of its inputs. The tests here assert bit-exact equality
(atol = rtol = 0) across repeated calls and across independently-constructed
rule instances.
"""

from __future__ import annotations

import torch

from src.v2_model.plasticity import (
    ThreeFactorRule,
    ThresholdHomeostasis,
    UrbanczikSennRule,
    VogelsISTDPRule,
)


# ---------------------------------------------------------------------------
# UrbanczikSennRule
# ---------------------------------------------------------------------------

def test_urbanczik_same_inputs_bit_exact() -> None:
    pre = torch.randn(3, 4)
    apical = torch.randn(3, 5)
    basal = torch.randn(3, 5)
    weights = torch.randn(5, 4)
    r = UrbanczikSennRule(lr=0.1, weight_decay=0.01)
    dw1 = r.delta(pre, apical, basal, weights)
    dw2 = r.delta(pre, apical, basal, weights)
    torch.testing.assert_close(dw1, dw2, atol=0.0, rtol=0.0)


def test_urbanczik_two_instances_agree_bit_exact() -> None:
    pre = torch.randn(3, 4)
    apical = torch.randn(3, 5)
    basal = torch.randn(3, 5)
    weights = torch.randn(5, 4)
    r1 = UrbanczikSennRule(lr=0.1, weight_decay=0.01)
    r2 = UrbanczikSennRule(lr=0.1, weight_decay=0.01)
    torch.testing.assert_close(
        r1.delta(pre, apical, basal, weights),
        r2.delta(pre, apical, basal, weights),
        atol=0.0, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# VogelsISTDPRule
# ---------------------------------------------------------------------------

def test_vogels_same_inputs_bit_exact() -> None:
    pre = torch.randn(3, 4)
    post = torch.randn(3, 5)
    weights = torch.randn(5, 4)
    r = VogelsISTDPRule(lr=0.1, target_rate=1.0, weight_decay=0.01)
    dw1 = r.delta(pre, post, weights)
    dw2 = r.delta(pre, post, weights)
    torch.testing.assert_close(dw1, dw2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# ThreeFactorRule
# ---------------------------------------------------------------------------

def test_three_factor_qm_same_inputs_bit_exact() -> None:
    cue = torch.randn(3, 4)
    memory = torch.randn(3, 5)
    memory_error = torch.randn(3, 5)
    weights = torch.randn(5, 4)
    r = ThreeFactorRule(lr=0.1, weight_decay=0.01)
    dw1 = r.delta_qm(cue, memory, memory_error, weights)
    dw2 = r.delta_qm(cue, memory, memory_error, weights)
    torch.testing.assert_close(dw1, dw2, atol=0.0, rtol=0.0)


def test_three_factor_mh_same_inputs_bit_exact() -> None:
    memory = torch.randn(3, 4)
    probe_error = torch.randn(3, 5)
    weights = torch.randn(5, 4)
    r = ThreeFactorRule(lr=0.1, weight_decay=0.01)
    dw1 = r.delta_mh(memory, probe_error, weights)
    dw2 = r.delta_mh(memory, probe_error, weights)
    torch.testing.assert_close(dw1, dw2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# ThresholdHomeostasis — tests end-state θ after N identical updates
# ---------------------------------------------------------------------------

def test_homeostasis_multi_step_bit_exact() -> None:
    """Independently constructed instances, N identical updates, same θ."""
    activity_seq = [torch.randn(3, 4) for _ in range(5)]
    h1 = ThresholdHomeostasis(lr=0.01, target_rate=1.0, n_units=4)
    h2 = ThresholdHomeostasis(lr=0.01, target_rate=1.0, n_units=4)
    for a in activity_seq:
        h1.update(a)
        h2.update(a)
    torch.testing.assert_close(h1.theta, h2.theta, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# No state-leakage between rules
# ---------------------------------------------------------------------------

def test_unrelated_rule_call_does_not_perturb_output() -> None:
    """A call to one rule must not perturb another rule's output: each rule
    call is a pure function of its own arguments."""
    pre = torch.randn(3, 4)
    apical = torch.randn(3, 5)
    basal = torch.randn(3, 5)
    weights = torch.randn(5, 4)
    us = UrbanczikSennRule(lr=0.1, weight_decay=0.01)
    dw_before = us.delta(pre, apical, basal, weights)

    # Interleaved unrelated rule call:
    vg = VogelsISTDPRule(lr=0.05, target_rate=2.0)
    _ = vg.delta(torch.randn(10, 20), torch.randn(10, 7), torch.randn(7, 20))

    dw_after = us.delta(pre, apical, basal, weights)
    torch.testing.assert_close(dw_before, dw_after, atol=0.0, rtol=0.0)
