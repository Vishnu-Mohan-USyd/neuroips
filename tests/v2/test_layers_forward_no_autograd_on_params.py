"""Backward through forward must never accumulate gradients into any
``nn.Parameter`` — all plastic weights have ``requires_grad=False`` and
are therefore invisible to autograd. Gradients may still flow into
module *inputs* (so BPTT-fallback paths keep working — design note #15).
"""

from __future__ import annotations

import torch

from src.v2_model.layers import HE, HPV, L23E, L23PV, L23SOM


def _assert_params_have_no_grad(mod) -> None:
    for name, p in mod.named_parameters():
        assert p.grad is None, (
            f"{type(mod).__name__}.{name}.grad was accumulated; expected None"
        )


# ---------------------------------------------------------------------------
# requires_grad sanity first (structural invariant)
# ---------------------------------------------------------------------------

def test_all_params_have_requires_grad_false() -> None:
    pops = [
        L23E(n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
             tau_ms=20.0, dt_ms=5.0, seed=0),
        L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0),
        L23SOM(n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=0),
        HE(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0),
        HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=0),
    ]
    for pop in pops:
        for name, p in pop.named_parameters():
            assert p.requires_grad is False, (
                f"{type(pop).__name__}.{name} has requires_grad=True"
            )


# ---------------------------------------------------------------------------
# backward() through forward leaves parameter .grad as None
# ---------------------------------------------------------------------------

def test_l23e_backward_leaves_params_ungraded() -> None:
    pop = L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
    )
    B = 3
    l4_in = torch.randn(B, pop.n_l4_e, requires_grad=True)
    rate, _ = pop(
        l4_in,
        torch.randn(B, pop.n_units),
        torch.randn(B, pop.n_som),
        torch.randn(B, pop.n_pv),
        torch.randn(B, pop.n_h_e),
        torch.zeros(B, pop.n_units),
        torch.zeros(B, pop.n_units),
    )
    rate.sum().backward()
    _assert_params_have_no_grad(pop)
    assert l4_in.grad is not None                                    # input grad OK


def test_l23pv_backward_leaves_params_ungraded() -> None:
    pop = L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0)
    x = torch.randn(2, pop.n_l23_e, requires_grad=True)
    rate, _ = pop(x, torch.zeros(2, pop.n_units))
    rate.sum().backward()
    _assert_params_have_no_grad(pop)
    assert x.grad is not None


def test_l23som_backward_leaves_params_ungraded() -> None:
    pop = L23SOM(
        n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=0,
    )
    x = torch.randn(2, pop.n_l23_e, requires_grad=True)
    y = torch.randn(2, pop.n_h_e, requires_grad=True)
    rate, _ = pop(x, y, torch.zeros(2, pop.n_units))
    rate.sum().backward()
    _assert_params_have_no_grad(pop)


def test_he_backward_leaves_params_ungraded() -> None:
    pop = HE(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0)
    x = torch.randn(2, pop.n_l23_e, requires_grad=True)
    rate, _ = pop(
        x,
        torch.zeros(2, pop.n_units),
        torch.zeros(2, pop.n_h_pv),
        torch.zeros(2, pop.n_units),
        torch.zeros(2, pop.n_units),
    )
    rate.sum().backward()
    _assert_params_have_no_grad(pop)


def test_hpv_backward_leaves_params_ungraded() -> None:
    pop = HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=0)
    x = torch.randn(2, pop.n_h_e, requires_grad=True)
    rate, _ = pop(x, torch.zeros(2, pop.n_units))
    rate.sum().backward()
    _assert_params_have_no_grad(pop)
