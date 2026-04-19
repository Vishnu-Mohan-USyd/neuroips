"""θ drift on E populations via ``ThresholdHomeostasis``; I populations carry
``target_rate_hz`` instead (no homeostasis submodule).
"""

from __future__ import annotations

import torch

from src.v2_model.layers import HE, HPV, L23E, L23PV, L23SOM


# ---------------------------------------------------------------------------
# E populations own a ``ThresholdHomeostasis`` submodule + θ buffer
# ---------------------------------------------------------------------------

def test_l23e_has_homeostasis_submodule() -> None:
    pop = L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
    )
    assert hasattr(pop, "homeostasis")
    assert pop.theta.shape == (pop.n_units,)


def test_he_has_homeostasis_submodule() -> None:
    pop = HE(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0)
    assert hasattr(pop, "homeostasis")
    assert pop.theta.shape == (pop.n_units,)


# ---------------------------------------------------------------------------
# θ drifts with activity — exact closed-form check
# ---------------------------------------------------------------------------

def test_l23e_theta_drifts_with_activity() -> None:
    """Δθ = lr · tanh(error/scale) · scale (Task #54 bounded rule)."""
    pop = L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
        target_rate=0.0, lr_homeostasis=1.0, init_theta=0.0,
    )
    activity = torch.ones(4, pop.n_units)
    pop.homeostasis.update(activity)
    # target=0 ⇒ deadband=0; error=1.0; scale = 0.1·|0| + 1e-3 = 1e-3;
    # tanh(1.0/1e-3) ≈ 1 ⇒ Δθ ≈ 1.0 · 1 · 1e-3 = 1e-3.
    scale = 1e-3
    expected_val = 1.0 * torch.tanh(torch.tensor(1.0 / scale)).item() * scale
    torch.testing.assert_close(
        pop.theta, torch.full((pop.n_units,), expected_val), atol=1e-8, rtol=0.0,
    )
    # Sign must be positive (overactive ⇒ θ rises).
    assert torch.all(pop.theta > 0)


def test_he_theta_drifts_with_activity() -> None:
    pop = HE(
        n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0,
        target_rate=0.0, lr_homeostasis=0.5, init_theta=0.0,
    )
    activity = torch.full((3, pop.n_units), 2.0)
    pop.homeostasis.update(activity)
    # target=0 ⇒ deadband=0; scale=1e-3; saturated update 0.5·1·1e-3 = 5e-4.
    scale = 1e-3
    expected_val = 0.5 * torch.tanh(torch.tensor(2.0 / scale)).item() * scale
    torch.testing.assert_close(
        pop.theta, torch.full((pop.n_units,), expected_val), atol=1e-8, rtol=0.0,
    )
    assert torch.all(pop.theta > 0)


def test_l23e_theta_under_target_drifts_negative() -> None:
    """Activity below target should push θ downward (homeostatic direction)."""
    pop = L23E(
        n_units=6, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
        target_rate=2.0, lr_homeostasis=0.5, init_theta=0.0,
    )
    activity = torch.zeros(4, pop.n_units)                          # mean_b = 0
    pop.homeostasis.update(activity)
    # error = 0 − 2 = −2. deadband = 0.2·2.0 = 0.4; |−2|>0.4 ⇒ kept.
    # scale = 0.1·2 + 1e-3 = 0.201. Δθ = 0.5 · tanh(−2/0.201) · 0.201.
    scale = 0.1 * 2.0 + 1e-3
    expected_val = 0.5 * torch.tanh(torch.tensor(-2.0 / scale)).item() * scale
    torch.testing.assert_close(
        pop.theta, torch.full((pop.n_units,), expected_val), atol=1e-8, rtol=0.0,
    )
    # Sign check: below-target activity ⇒ θ drops.
    assert torch.all(pop.theta < 0)


# ---------------------------------------------------------------------------
# I populations: no homeostasis submodule, but do have ``target_rate_hz``
# ---------------------------------------------------------------------------

def test_l23pv_has_no_homeostasis_submodule() -> None:
    pop = L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0)
    assert not hasattr(pop, "homeostasis")
    assert isinstance(pop.target_rate_hz, float)


def test_l23som_has_no_homeostasis_submodule() -> None:
    pop = L23SOM(n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=0)
    assert not hasattr(pop, "homeostasis")
    assert isinstance(pop.target_rate_hz, float)


def test_hpv_has_no_homeostasis_submodule() -> None:
    pop = HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=0)
    assert not hasattr(pop, "homeostasis")
    assert isinstance(pop.target_rate_hz, float)
