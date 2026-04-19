"""Dale-sign preservation under softplus parameterisation.

Documented invariant (see `plasticity.py` module docstring):

    w_effective_excitatory  =  softplus(raw)   ≥ 0
    w_effective_inhibitory  = −softplus(raw)   ≤ 0

Because softplus is strictly monotonic, any real-valued raw-weight update
Δraw keeps the effective weight on the correct side of zero. This file
pins the invariant down with explicit tests:

  1. softplus monotonicity itself (sign of Δraw ≡ sign of Δsoftplus).
  2. Applying an Urbanczik-Senn update to excitatory raw weights:
       softplus(raw + dw) ≥ 0 always; direction tracks ε.
  3. Applying a Vogels update to inhibitory raw weights:
       −softplus(raw + dw) ≤ 0 always; above-target post ⇒ stronger inhibition.
  4. Applying a ThreeFactor update to inhibitory/excitatory raw weights:
       sign of effective weight is preserved after the step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.v2_model.plasticity import (
    ThreeFactorRule,
    UrbanczikSennRule,
    VogelsISTDPRule,
)


# ---------------------------------------------------------------------------
# 1. Softplus monotonicity (the underlying invariant)
# ---------------------------------------------------------------------------

def test_softplus_is_strictly_monotonic() -> None:
    """softplus(raw + Δ) − softplus(raw) has the same sign as Δ (for |Δ|>0)."""
    raw = torch.randn(200)
    dpos = torch.rand(200) * 0.2 + 1e-3                    # all positive
    dneg = -dpos
    assert (F.softplus(raw + dpos) > F.softplus(raw)).all()
    assert (F.softplus(raw + dneg) < F.softplus(raw)).all()


# ---------------------------------------------------------------------------
# 2. Excitatory Dale preservation — UrbanczikSenn
# ---------------------------------------------------------------------------

def test_urbanczik_excitatory_weight_remains_non_negative() -> None:
    """softplus(raw + dw) ≥ 0 always, regardless of sign of dw."""
    torch.manual_seed(0)
    raw = torch.randn(8, 5)                                # [n_post, n_pre]
    rule = UrbanczikSennRule(lr=0.2, weight_decay=0.05)
    pre = torch.rand(4, 5)
    # Run with both signs of ε to cover positive & negative Δraw.
    for sign in (+1.0, -1.0):
        apical = torch.rand(4, 8) + 0.1
        basal = apical + sign * (torch.rand(4, 8) + 0.1)
        dw = rule.delta(pre, apical, basal, raw)
        w_after = F.softplus(raw + dw)
        assert (w_after >= 0).all()


def test_urbanczik_positive_epsilon_grows_excitatory_weight() -> None:
    """Positive ε ⇒ softplus(raw_new) > softplus(raw_old) entrywise."""
    torch.manual_seed(1)
    raw = torch.randn(8, 5)
    rule = UrbanczikSennRule(lr=0.2, weight_decay=0.0)
    pre = torch.ones(4, 5)                                 # positive
    apical = torch.full((4, 8), 2.0)
    basal = torch.zeros(4, 8)                              # ε = 2 > 0
    dw = rule.delta(pre, apical, basal, raw)
    assert (dw > 0).all()
    w_before = F.softplus(raw)
    w_after = F.softplus(raw + dw)
    assert (w_after > w_before).all()


# ---------------------------------------------------------------------------
# 3. Inhibitory Dale preservation — Vogels iSTDP
# ---------------------------------------------------------------------------

def test_vogels_inhibitory_weight_remains_non_positive() -> None:
    """−softplus(raw + dw) ≤ 0 always."""
    torch.manual_seed(2)
    raw = torch.randn(8, 5)
    rule = VogelsISTDPRule(lr=0.2, target_rate=1.0, weight_decay=0.05)
    pre = torch.rand(4, 5)
    for post_scale in (0.2, 2.0):                          # below + above target
        post = torch.full((4, 8), post_scale)
        dw = rule.delta(pre, post, raw)
        w_after = -F.softplus(raw + dw)
        assert (w_after <= 0).all()


def test_vogels_post_above_target_strengthens_inhibition() -> None:
    """a_post > ρ ⇒ softplus(raw) grows ⇒ −softplus(raw) more negative."""
    torch.manual_seed(3)
    raw = torch.randn(8, 5)
    rule = VogelsISTDPRule(lr=0.2, target_rate=1.0, weight_decay=0.0)
    pre = torch.ones(4, 5)                                 # positive
    post = torch.full((4, 8), 2.0)                         # above target
    dw = rule.delta(pre, post, raw)
    assert (dw > 0).all()
    w_before = -F.softplus(raw)                            # ≤ 0
    w_after = -F.softplus(raw + dw)
    # Stronger inhibition = more negative = strictly less than before.
    assert (w_after < w_before).all()


def test_vogels_post_below_target_weakens_inhibition() -> None:
    """a_post < ρ ⇒ softplus(raw) shrinks ⇒ −softplus(raw) less negative."""
    torch.manual_seed(4)
    raw = torch.randn(8, 5)
    rule = VogelsISTDPRule(lr=0.2, target_rate=1.0, weight_decay=0.0)
    pre = torch.ones(4, 5)
    post = torch.full((4, 8), 0.5)                         # below target
    dw = rule.delta(pre, post, raw)
    assert (dw < 0).all()
    w_before = -F.softplus(raw)
    w_after = -F.softplus(raw + dw)
    assert (w_after > w_before).all()                      # closer to zero


# ---------------------------------------------------------------------------
# 4. ThreeFactor Dale preservation
# ---------------------------------------------------------------------------

def test_three_factor_excitatory_weight_stays_non_negative() -> None:
    torch.manual_seed(5)
    raw = torch.randn(8, 5)                                # interpret as excitatory raw
    rule = ThreeFactorRule(lr=0.2, weight_decay=0.05)
    cue = torch.rand(4, 5) + 0.1
    memory = torch.rand(4, 8) + 0.1
    for err_sign in (+1.0, -1.0):
        memory_error = err_sign * (torch.rand(4, 8) + 0.1)
        dw = rule.delta_qm(cue, memory, memory_error, raw)
        assert (F.softplus(raw + dw) >= 0).all()


def test_three_factor_mh_preserves_inhibitory_sign() -> None:
    torch.manual_seed(6)
    raw = torch.randn(8, 5)                                # interpret as inhibitory raw
    rule = ThreeFactorRule(lr=0.2, weight_decay=0.05)
    memory = torch.rand(4, 5) + 0.1
    for err_sign in (+1.0, -1.0):
        probe_error = err_sign * (torch.rand(4, 8) + 0.1)
        dw = rule.delta_mh(memory, probe_error, raw)
        assert ((-F.softplus(raw + dw)) <= 0).all()
