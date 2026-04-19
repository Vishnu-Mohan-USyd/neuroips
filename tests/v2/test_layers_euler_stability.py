"""Linear-Euler stability guard: ``dt_ms < tau_ms`` strictly enforced on every
population; forward produces finite output for bounded input.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.layers import HE, HPV, L23E, L23PV, L23SOM


def _size_kwargs(cls) -> dict:
    """Minimal population-size kwargs for each class (excluding τ / dt)."""
    if cls is L23E:
        return dict(n_units=4, n_l4_e=3, n_pv=2, n_som=2, n_h_e=2)
    if cls is L23PV:
        return dict(n_units=3, n_l23_e=4)
    if cls is L23SOM:
        return dict(n_units=3, n_l23_e=4, n_h_e=2)
    if cls is HE:
        return dict(n_units=4, n_l23_e=4, n_h_pv=2)
    if cls is HPV:
        return dict(n_units=2, n_h_e=4)
    raise AssertionError(cls)


ALL_POPS = [L23E, L23PV, L23SOM, HE, HPV]


@pytest.mark.parametrize("cls", ALL_POPS)
def test_dt_less_than_tau_constructs(cls) -> None:
    """dt < τ must succeed — mid-band point."""
    cls(tau_ms=10.0, dt_ms=5.0, seed=0, **_size_kwargs(cls))


@pytest.mark.parametrize("cls", ALL_POPS)
def test_dt_equal_tau_raises(cls) -> None:
    """dt == τ violates strict stability guard."""
    with pytest.raises(ValueError, match=r"dt_ms.*must be < tau_ms"):
        cls(tau_ms=5.0, dt_ms=5.0, seed=0, **_size_kwargs(cls))


@pytest.mark.parametrize("cls", ALL_POPS)
def test_dt_greater_tau_raises(cls) -> None:
    """dt > τ violates stability guard."""
    with pytest.raises(ValueError, match=r"dt_ms.*must be < tau_ms"):
        cls(tau_ms=5.0, dt_ms=10.0, seed=0, **_size_kwargs(cls))


@pytest.mark.parametrize("cls", ALL_POPS)
def test_zero_or_negative_tau_raises(cls) -> None:
    for tau in (0.0, -1.0):
        with pytest.raises(ValueError, match=r"tau_ms must be > 0"):
            cls(tau_ms=tau, dt_ms=1.0, seed=0, **_size_kwargs(cls))


@pytest.mark.parametrize("cls", ALL_POPS)
def test_zero_or_negative_dt_raises(cls) -> None:
    for dt in (0.0, -1.0):
        with pytest.raises(ValueError, match=r"dt_ms must be > 0"):
            cls(tau_ms=10.0, dt_ms=dt, seed=0, **_size_kwargs(cls))


# ---------------------------------------------------------------------------
# Bounded-input → bounded-output sanity (Euler step does not blow up)
# ---------------------------------------------------------------------------

def test_l23e_bounded_input_gives_finite_output() -> None:
    pop = L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
    )
    B = 3
    rate, _ = pop(
        torch.ones(B, pop.n_l4_e),
        torch.ones(B, pop.n_units),
        torch.ones(B, pop.n_som),
        torch.ones(B, pop.n_pv),
        torch.ones(B, pop.n_h_e),
        torch.zeros(B, pop.n_units),
        torch.ones(B, pop.n_units),
    )
    assert torch.isfinite(rate).all()


def test_l23pv_bounded_input_gives_finite_output() -> None:
    pop = L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0)
    rate, _ = pop(torch.ones(2, pop.n_l23_e), torch.ones(2, pop.n_units))
    assert torch.isfinite(rate).all()


def test_he_bounded_input_gives_finite_output() -> None:
    pop = HE(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0)
    rate, _ = pop(
        torch.ones(2, pop.n_l23_e),
        torch.ones(2, pop.n_units),
        torch.ones(2, pop.n_h_pv),
        torch.zeros(2, pop.n_units),
        torch.ones(2, pop.n_units),
    )
    assert torch.isfinite(rate).all()
