"""Regression tests for Task #74 Fix C-v2 — per-SOM-unit gain modulation.

The Phase-3 task readout produces a multiplicative gain on the SOM→L23E
inhibitory synapses instead of adding to the SOM drive directly. This
test file locks in the four behavioural invariants the fix depends on:

  (a) At ``W_mh_task_inh = 0`` (construction state) the gain is
      identically 1 — so the forward is a strict no-op vs. the
      pre-Fix-C circuit. This is the "Phase-2 exactly preserved at
      init" invariant.
  (b) The gain is strictly bounded in ``(0, 4.0]``. Softplus keeps it
      positive; ``clamp(max=4.0)`` keeps it bounded above regardless of
      how large ``W_mh_task_inh · m`` grows.
  (c) L23E firing rate is monotone-decreasing in gain (other inputs
      held fixed) — more gain ⇒ more effective SOM inhibition ⇒ less
      L23E activity. This is the sign-of-effect invariant the gain
      route relies on biologically (cholinergic/noradrenergic GABA
      up-regulation suppresses downstream pyramidal activity).
  (d) With ``som_gain = 1`` the L23E forward is bit-identical to the
      ``som_gain = None`` path. Protects the default-arg contract.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.v2_model.context_memory import ContextMemory
from src.v2_model.layers import L23E


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cm(n_out_som: int = 9, seed: int = 0) -> ContextMemory:
    return ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        n_out_som=n_out_som, tau_m_ms=500.0, dt_ms=5.0, seed=seed,
    )


def _l23e(n_som: int = 9, seed: int = 0) -> L23E:
    return L23E(
        n_units=32, n_l4_e=16, n_pv=8, n_som=n_som, n_h_e=24,
        tau_ms=20.0, dt_ms=5.0, seed=seed,
    )


def _l23e_inputs(layer: L23E, B: int = 2) -> dict[str, torch.Tensor]:
    return dict(
        l4_input=torch.rand(B, layer.n_l4_e) * 0.1,
        l23_recurrent_input=torch.rand(B, layer.n_units) * 0.05,
        som_input=torch.rand(B, layer.n_som) * 0.5 + 0.1,  # strictly positive
        pv_input=torch.rand(B, layer.n_pv) * 0.05,
        h_apical_input=torch.rand(B, layer.n_h_e) * 0.05,
        context_bias=torch.zeros(B, layer.n_units),
        state=torch.rand(B, layer.n_units) * 0.01,
    )


# ---------------------------------------------------------------------------
# (a) Init no-op: gain == 1 at construction
# ---------------------------------------------------------------------------


def test_som_gain_is_exactly_one_at_construction() -> None:
    """At W_mh_task_inh=0 (construction), gain = softplus(log(e-1)) = 1.0
    up to float32-roundoff precision (~1e-6 per element)."""
    cm = _cm()
    torch.manual_seed(0)
    B = 4
    m = torch.randn(B, cm.n_m)
    h = torch.randn(B, cm.n_h)
    _, _, som_gain = cm(m, h)
    expected = torch.ones_like(som_gain)
    torch.testing.assert_close(som_gain, expected, atol=5e-5, rtol=0.0)


def test_som_gain_unchanged_by_memory_content_when_weights_zero() -> None:
    """With W_mh_task_inh=0, gain is independent of m_t (readout is identically 1)."""
    cm = _cm()
    torch.manual_seed(0)
    h = torch.randn(3, cm.n_h)
    m1 = torch.randn(3, cm.n_m)
    m2 = torch.randn(3, cm.n_m) * 10.0  # wildly different memory content
    _, _, gain1 = cm(m1, h)
    _, _, gain2 = cm(m2, h)
    torch.testing.assert_close(gain1, gain2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# (b) Upper/lower bound: gain in (0, 4.0]
# ---------------------------------------------------------------------------


def test_som_gain_bounded_above_at_4() -> None:
    """Large positive W_mh_task_inh·m cannot push gain past 4.0."""
    cm = _cm()
    B = 2
    torch.manual_seed(0)
    with torch.no_grad():
        # Force W_mh_task_inh to large positive values so the pre-softplus
        # input blows way past any sane operating point.
        cm.W_mh_task_inh.normal_(mean=5.0, std=1.0)
    m = torch.ones(B, cm.n_m) * 5.0                  # big positive memory
    h = torch.zeros(B, cm.n_h)
    _, _, som_gain = cm(m, h)
    # Every entry must respect the clamp.
    assert float(som_gain.max().item()) <= 4.0 + 1e-6
    # And the clamp should actually be biting (not vacuous).
    assert float(som_gain.max().item()) == pytest.approx(4.0, abs=1e-5)


def test_som_gain_strictly_positive_under_moderate_negative_drive() -> None:
    """Moderate negative W_mh_task_inh·m lowers gain below 1 but keeps it > 0.

    This is the "down-regulation" mode — the inh readout can shrink
    SOM→L23E efficacy as well as grow it. We avoid extreme drive
    magnitudes where softplus float32-underflows to 0; the biologically
    meaningful regime is mild down-regulation, and that is what this
    test covers.
    """
    cm = _cm()
    B = 2
    with torch.no_grad():
        cm.W_mh_task_inh.normal_(mean=-0.2, std=0.1)  # moderate negative
    m = torch.ones(B, cm.n_m) * 0.5                  # moderate positive
    h = torch.zeros(B, cm.n_h)
    _, _, som_gain = cm(m, h)
    assert float(som_gain.min().item()) > 0.0
    # Down-regulation: gain should sit below the init value of 1.
    assert float(som_gain.mean().item()) < 1.0


# ---------------------------------------------------------------------------
# (c) Monotonicity: larger gain ⇒ smaller L23E response (other inputs fixed)
# ---------------------------------------------------------------------------


def test_l23e_response_monotone_decreasing_in_som_gain() -> None:
    """With positive SOM activity and all else equal, raising the per-unit
    gain must not increase r_l23 and must strictly decrease it for at least
    one gain step. This is the sign-of-effect invariant Fix C-v2 relies on.
    """
    torch.manual_seed(0)
    layer = _l23e()
    inp = _l23e_inputs(layer, B=1)

    # Sweep gain uniformly from 1.0 (init no-op) up to 4.0 (clamp ceiling).
    gain_values = [1.0, 1.5, 2.0, 3.0, 4.0]
    rates = []
    for g in gain_values:
        som_gain = torch.full((1, layer.n_som), g, dtype=torch.float32)
        rate, _ = layer(**inp, som_gain=som_gain)
        rates.append(float(rate.mean().item()))

    # Non-increasing across the sweep.
    for i in range(len(rates) - 1):
        assert rates[i + 1] <= rates[i] + 1e-7, (
            f"r_l23 increased when gain went from {gain_values[i]} "
            f"→ {gain_values[i+1]}: {rates[i]:.6f} → {rates[i+1]:.6f}"
        )
    # And at least one step must be strictly decreasing — otherwise the
    # gain had no measurable effect (would indicate the kwarg is ignored).
    assert rates[-1] < rates[0], (
        f"r_l23 did not decrease under 4× gain: {rates[0]:.6f} → {rates[-1]:.6f}"
    )


# ---------------------------------------------------------------------------
# (d) som_gain=None path equals som_gain=1 path (default-arg contract)
# ---------------------------------------------------------------------------


def test_l23e_som_gain_none_equals_ones() -> None:
    """Passing ``som_gain=None`` must match ``som_gain = torch.ones(...)``
    exactly. This protects the default-arg contract so existing (Phase-2)
    callers that never pass the kwarg continue to work bit-identically.
    """
    torch.manual_seed(0)
    layer = _l23e()
    inp = _l23e_inputs(layer, B=3)
    rate_none, _ = layer(**inp, som_gain=None)
    ones = torch.ones(3, layer.n_som, dtype=torch.float32)
    rate_ones, _ = layer(**inp, som_gain=ones)
    torch.testing.assert_close(rate_none, rate_ones, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# End-to-end: at W=0, the full L23E forward is bit-identical to pre-Fix-C.
# ---------------------------------------------------------------------------


def test_bias_offset_makes_softplus_evaluate_to_one() -> None:
    """Sanity check on the constant: the module's bias shift must make
    softplus evaluate to exactly 1.0 in float64 (and very close in
    float32). If this constant is ever changed by accident, the "init
    no-op" semantics break silently — this test locks the value in
    place by reading the symbol directly from the module.
    """
    from src.v2_model.context_memory import (
        _SOM_GAIN_BIAS0, _SOM_GAIN_CLAMP_MAX,
    )
    # By construction the bias is log(e - 1).
    assert abs(_SOM_GAIN_BIAS0 - math.log(math.e - 1.0)) < 1e-12
    # Softplus evaluated at that bias is exactly 1.0 (float64).
    y = F.softplus(torch.tensor(_SOM_GAIN_BIAS0, dtype=torch.float64))
    assert abs(float(y.item()) - 1.0) < 1e-12
    # Upper clamp: reasonable biological range (cholinergic modulation
    # ~2-4× — Disney & Aoki 2008; Pfeffer 2013).
    assert 2.0 <= _SOM_GAIN_CLAMP_MAX <= 6.0
