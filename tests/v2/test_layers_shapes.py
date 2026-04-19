"""Forward shape / phase-manifest sanity for the five `layers.py` populations."""

from __future__ import annotations

import torch

from src.v2_model.layers import HE, HPV, L23E, L23PV, L23SOM


def _l23e(**kw) -> L23E:
    return L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0, **kw,
    )


def _l23pv(**kw) -> L23PV:
    return L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0, **kw)


def _l23som(**kw) -> L23SOM:
    return L23SOM(
        n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=0, **kw,
    )


def _he(**kw) -> HE:
    return HE(
        n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0, **kw,
    )


def _hpv(**kw) -> HPV:
    return HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=0, **kw)


# ---------------------------------------------------------------------------
# Forward output shapes
# ---------------------------------------------------------------------------

def test_l23e_forward_shape() -> None:
    pop = _l23e()
    B = 4
    rate, state = pop(
        torch.randn(B, pop.n_l4_e),
        torch.randn(B, pop.n_units),
        torch.randn(B, pop.n_som),
        torch.randn(B, pop.n_pv),
        torch.randn(B, pop.n_h_e),
        torch.randn(B, pop.n_units),
        torch.randn(B, pop.n_units),
    )
    assert rate.shape == (B, pop.n_units)
    assert state.shape == (B, pop.n_units)


def test_l23pv_forward_shape() -> None:
    pop = _l23pv()
    B = 3
    rate, state = pop(
        torch.randn(B, pop.n_l23_e), torch.randn(B, pop.n_units)
    )
    assert rate.shape == (B, pop.n_units)
    assert state.shape == (B, pop.n_units)


def test_l23som_forward_shape() -> None:
    pop = _l23som()
    B = 3
    rate, state = pop(
        torch.randn(B, pop.n_l23_e),
        torch.randn(B, pop.n_h_e),
        torch.randn(B, pop.n_units),
    )
    assert rate.shape == (B, pop.n_units)
    assert state.shape == (B, pop.n_units)


def test_he_forward_shape() -> None:
    pop = _he()
    B = 2
    rate, state = pop(
        torch.randn(B, pop.n_l23_e),
        torch.randn(B, pop.n_units),
        torch.randn(B, pop.n_h_pv),
        torch.randn(B, pop.n_units),
        torch.randn(B, pop.n_units),
    )
    assert rate.shape == (B, pop.n_units)
    assert state.shape == (B, pop.n_units)


def test_hpv_forward_shape() -> None:
    pop = _hpv()
    B = 2
    rate, state = pop(
        torch.randn(B, pop.n_h_e), torch.randn(B, pop.n_units)
    )
    assert rate.shape == (B, pop.n_units)
    assert state.shape == (B, pop.n_units)


def test_forward_dtype_matches_input() -> None:
    pop = L23PV(
        n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0,
        dtype=torch.float64,
    )
    B = 2
    rate, _ = pop(
        torch.randn(B, pop.n_l23_e, dtype=torch.float64),
        torch.randn(B, pop.n_units, dtype=torch.float64),
    )
    assert rate.dtype == torch.float64


# ---------------------------------------------------------------------------
# plastic_weight_names() manifest
# ---------------------------------------------------------------------------

def test_l23e_plastic_weight_names() -> None:
    assert set(_l23e().plastic_weight_names()) == {
        "W_l4_l23_raw", "W_rec_raw", "W_pv_l23_raw",
        "W_som_l23_raw", "W_fb_apical_raw",
    }


def test_l23pv_plastic_weight_names() -> None:
    assert set(_l23pv().plastic_weight_names()) == {"W_l23_pv_raw"}


def test_l23som_plastic_weight_names() -> None:
    assert set(_l23som().plastic_weight_names()) == {
        "W_l23_som_raw", "W_fb_som_raw",
    }


def test_he_plastic_weight_names() -> None:
    assert set(_he().plastic_weight_names()) == {
        "W_l23_h_raw", "W_rec_raw", "W_pv_h_raw",
    }


def test_hpv_plastic_weight_names() -> None:
    assert set(_hpv().plastic_weight_names()) == {"W_h_pv_raw"}
