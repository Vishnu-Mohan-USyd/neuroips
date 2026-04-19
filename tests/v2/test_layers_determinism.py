"""Determinism: same seed → bit-identical weights; same inputs → bit-identical forward."""

from __future__ import annotations

import pytest
import torch

from src.v2_model.layers import HE, HPV, L23E, L23PV, L23SOM


def _spec_for(cls):
    if cls is L23E:
        return (
            dict(n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
                 tau_ms=20.0, dt_ms=5.0),
            ("W_l4_l23_raw", "W_rec_raw", "W_pv_l23_raw",
             "W_som_l23_raw", "W_fb_apical_raw"),
        )
    if cls is L23PV:
        return (
            dict(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0),
            ("W_l23_pv_raw",),
        )
    if cls is L23SOM:
        return (
            dict(n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0),
            ("W_l23_som_raw", "W_fb_som_raw"),
        )
    if cls is HE:
        return (
            dict(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0),
            ("W_l23_h_raw", "W_rec_raw", "W_pv_h_raw"),
        )
    if cls is HPV:
        return (
            dict(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0),
            ("W_h_pv_raw",),
        )
    raise AssertionError(cls)


ALL_POPS = [L23E, L23PV, L23SOM, HE, HPV]


@pytest.mark.parametrize("cls", ALL_POPS)
def test_same_seed_same_init(cls) -> None:
    kw, names = _spec_for(cls)
    pop1 = cls(**kw, seed=42)
    pop2 = cls(**kw, seed=42)
    for name in names:
        torch.testing.assert_close(
            getattr(pop1, name), getattr(pop2, name),
            atol=0.0, rtol=0.0,
        )


@pytest.mark.parametrize("cls", ALL_POPS)
def test_different_seed_different_init(cls) -> None:
    kw, names = _spec_for(cls)
    pop1 = cls(**kw, seed=0)
    pop2 = cls(**kw, seed=1)
    for name in names:
        assert not torch.equal(
            getattr(pop1, name), getattr(pop2, name)
        ), f"{cls.__name__}.{name} identical across seeds"


# ---------------------------------------------------------------------------
# Per-population forward determinism
# ---------------------------------------------------------------------------

def test_l23e_forward_bit_exact_across_instances() -> None:
    kw = dict(n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
              tau_ms=20.0, dt_ms=5.0)
    p1 = L23E(**kw, seed=11)
    p2 = L23E(**kw, seed=11)
    B = 3
    inputs = (
        torch.randn(B, p1.n_l4_e),
        torch.randn(B, p1.n_units),
        torch.randn(B, p1.n_som),
        torch.randn(B, p1.n_pv),
        torch.randn(B, p1.n_h_e),
        torch.randn(B, p1.n_units),
        torch.randn(B, p1.n_units),
    )
    r1, s1 = p1(*inputs)
    r2, s2 = p2(*inputs)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)


def test_l23pv_forward_bit_exact_across_calls() -> None:
    p = L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=7)
    B = 2
    x = torch.randn(B, p.n_l23_e)
    state = torch.randn(B, p.n_units)
    r1, s1 = p(x, state)
    r2, s2 = p(x, state)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)


def test_he_forward_bit_exact_across_instances() -> None:
    kw = dict(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0)
    p1 = HE(**kw, seed=3)
    p2 = HE(**kw, seed=3)
    B = 2
    inputs = (
        torch.randn(B, p1.n_l23_e),
        torch.randn(B, p1.n_units),
        torch.randn(B, p1.n_h_pv),
        torch.randn(B, p1.n_units),
        torch.randn(B, p1.n_units),
    )
    r1, s1 = p1(*inputs)
    r2, s2 = p2(*inputs)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)


def test_l23som_forward_bit_exact_across_calls() -> None:
    p = L23SOM(n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=5)
    B = 2
    x = torch.randn(B, p.n_l23_e)
    y = torch.randn(B, p.n_h_e)
    state = torch.randn(B, p.n_units)
    r1, s1 = p(x, y, state)
    r2, s2 = p(x, y, state)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)


def test_hpv_forward_bit_exact_across_calls() -> None:
    p = HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=9)
    B = 2
    x = torch.randn(B, p.n_h_e)
    state = torch.randn(B, p.n_units)
    r1, s1 = p(x, state)
    r2, s2 = p(x, state)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)
