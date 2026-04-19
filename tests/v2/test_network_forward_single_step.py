"""Single-step forward semantics.

Every rate produced by one forward step must be non-negative (all
populations rectify via ``rectified_softplus``) and finite. ``x̂_{t+1}``
must also be non-negative — the prediction head is a sum of
``softplus``-weighted non-negative terms rectified again at the output.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def test_rates_non_negative_and_finite(cfg, net):
    """All population rates and x̂ are non-negative + finite after one step."""
    a = cfg.arch
    B = 2
    state = net.initial_state(batch_size=B)
    frames = torch.randn(B, 1, a.grid_h, a.grid_w) * 0.2
    x_hat, ns, info = net(frames, state)

    for name, tensor in (
        ("r_l4", ns.r_l4), ("r_l23", ns.r_l23), ("r_pv", ns.r_pv),
        ("r_som", ns.r_som), ("r_h", ns.r_h), ("h_pv", ns.h_pv),
        ("m", ns.m), ("x_hat", x_hat),
    ):
        assert torch.all(tensor >= 0.0), f"{name} has negative entries"
        assert torch.all(torch.isfinite(tensor)), f"{name} has non-finite entries"
    # LGN feature map can be zero on blank input, but is always non-negative
    # (rectified DoG + quadrature Gabor energy)
    assert torch.all(info["lgn_feature_map"] >= 0.0)


def test_forward_no_autograd_on_parameters(cfg, net):
    """No nn.Parameter in the network requires grad (pure local-plasticity)."""
    for p in net.parameters():
        assert p.requires_grad is False, (
            f"parameter requires_grad=True — v2 uses pure local plasticity, "
            f"no gradients should accumulate on any Parameter"
        )


def test_lgn_l4_has_no_parameters(cfg, net):
    """LGN/L4 front end is frozen by construction (zero nn.Parameters)."""
    assert sum(1 for _ in net.lgn_l4.parameters()) == 0, (
        "LGN/L4 must hold zero nn.Parameters — the frozen-core contract"
    )
