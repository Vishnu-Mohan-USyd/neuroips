"""Dale's law: E raw → ``+softplus``; I raw → ``-softplus``; masked-off = 0."""

from __future__ import annotations

import torch

from src.v2_model.layers import (
    HE, HPV, L23E, L23PV, L23SOM,
    _excitatory_eff, _inhibitory_eff,
)


# ---------------------------------------------------------------------------
# Helper contract
# ---------------------------------------------------------------------------

def test_excitatory_eff_always_nonnegative() -> None:
    raw = torch.randn(8, 12) * 5.0                                  # ±large raw
    w = _excitatory_eff(raw)
    assert (w >= 0).all()


def test_inhibitory_eff_always_nonpositive() -> None:
    raw = torch.randn(8, 12) * 5.0
    w = _inhibitory_eff(raw)
    assert (w <= 0).all()


def test_mask_zero_entries_yield_exact_zero() -> None:
    raw = torch.randn(4, 4)
    mask = torch.tensor(
        [[0., 1., 0., 1.],
         [1., 0., 1., 0.],
         [0., 0., 1., 1.],
         [1., 1., 0., 0.]]
    )
    w_e = _excitatory_eff(raw, mask)
    w_i = _inhibitory_eff(raw, mask)
    zero_positions = (mask == 0)
    assert (w_e[zero_positions] == 0).all()
    assert (w_i[zero_positions] == 0).all()


# ---------------------------------------------------------------------------
# L23E — Dale classes
# ---------------------------------------------------------------------------

def _mini_l23e() -> L23E:
    return L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
    )


def test_l23e_excitatory_weights_nonnegative() -> None:
    pop = _mini_l23e()
    for name in ("W_l4_l23_raw", "W_rec_raw", "W_fb_apical_raw"):
        w = _excitatory_eff(getattr(pop, name))
        assert (w >= 0).all(), f"{name} has a negative effective entry"


def test_l23e_inhibitory_weights_nonpositive() -> None:
    pop = _mini_l23e()
    for name in ("W_pv_l23_raw", "W_som_l23_raw"):
        w = _inhibitory_eff(getattr(pop, name))
        assert (w <= 0).all(), f"{name} has a positive effective entry"


def test_l23e_masked_recurrent_exact_zero() -> None:
    pop = _mini_l23e()
    w = _excitatory_eff(pop.W_rec_raw, pop.mask_rec)
    off_positions = (pop.mask_rec == 0)
    assert (w[off_positions] == 0).all()


# ---------------------------------------------------------------------------
# L23PV, L23SOM — excitatory presynaptic (E→I) input stays non-negative
# ---------------------------------------------------------------------------

def test_l23pv_input_weights_nonnegative() -> None:
    pop = L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0)
    w = _excitatory_eff(pop.W_l23_pv_raw)
    assert (w >= 0).all()


def test_l23som_input_weights_nonnegative() -> None:
    pop = L23SOM(n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=0)
    for name in ("W_l23_som_raw", "W_fb_som_raw"):
        w = _excitatory_eff(getattr(pop, name))
        assert (w >= 0).all(), f"{name} has a negative effective entry"


# ---------------------------------------------------------------------------
# HE — Dale classes + masked exact zero
# ---------------------------------------------------------------------------

def _mini_he() -> HE:
    return HE(
        n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0,
    )


def test_he_excitatory_weights_nonnegative() -> None:
    pop = _mini_he()
    for name in ("W_l23_h_raw", "W_rec_raw"):
        w = _excitatory_eff(getattr(pop, name))
        assert (w >= 0).all(), f"{name} has a negative effective entry"


def test_he_inhibitory_pv_nonpositive() -> None:
    pop = _mini_he()
    w = _inhibitory_eff(pop.W_pv_h_raw)
    assert (w <= 0).all()


def test_he_masked_recurrent_exact_zero() -> None:
    pop = _mini_he()
    w = _excitatory_eff(pop.W_rec_raw, pop.mask_rec)
    assert (w[pop.mask_rec == 0] == 0).all()


# ---------------------------------------------------------------------------
# HPV
# ---------------------------------------------------------------------------

def test_hpv_input_weights_nonnegative() -> None:
    pop = HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=0)
    w = _excitatory_eff(pop.W_h_pv_raw)
    assert (w >= 0).all()
