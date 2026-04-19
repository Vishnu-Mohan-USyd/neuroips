"""``bridge.extract_activations`` contract: stack all populations over time.

The analysis modules in ``src/analysis/*`` consume a dict keyed by
population name with ``[B, T, ...]`` tensors. ``extract_activations``
rolls out the network for ``T`` steps and returns this dict.

Tests:

* Every documented key is present.
* Tensor shapes are ``[B, T, ...]`` with the right trailing dims.
* Rolling twice on the same stimulus gives identical dicts
  (deterministic).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.bridge import extract_activations
from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def test_extract_activations_shape_contract(cfg, net):
    """Every key has the correct ``[B, T, ...]`` shape."""
    a = cfg.arch
    B, T = 2, 5
    torch.manual_seed(0)
    stim = torch.randn(B, T, 1, a.grid_h, a.grid_w) * 0.2

    acts = extract_activations(net, stim)

    expected_shapes: dict[str, tuple[int, ...]] = {
        "r_l4": (B, T, a.n_l4_e),
        "r_l23": (B, T, a.n_l23_e),
        "r_pv": (B, T, a.n_l23_pv),
        "r_som": (B, T, a.n_l23_som),
        "r_h": (B, T, a.n_h_e),
        "h_pv": (B, T, a.n_h_pv),
        "m": (B, T, a.n_c),
        "regime_posterior": (B, T, cfg.regime.n_regimes),
        "feature_map": (B, T, 2 + a.n_orientations, a.grid_h, a.grid_w),
    }
    for key, shape in expected_shapes.items():
        assert key in acts, f"key {key!r} missing from activations dict"
        assert tuple(acts[key].shape) == shape, (
            f"{key!r} shape {tuple(acts[key].shape)} != expected {shape}"
        )


def test_extract_activations_deterministic(cfg, net):
    """Running twice on the same stimulus gives identical activations."""
    a = cfg.arch
    torch.manual_seed(1)
    stim = torch.randn(1, 4, 1, a.grid_h, a.grid_w) * 0.2

    acts1 = extract_activations(net, stim)
    acts2 = extract_activations(net, stim)
    for key in acts1:
        torch.testing.assert_close(acts1[key], acts2[key], atol=0.0, rtol=0.0)


def test_extract_activations_rejects_bad_shape(cfg, net):
    """Mis-shaped stimulus (missing channel axis) raises ValueError."""
    bad = torch.zeros(1, 4, cfg.arch.grid_h, cfg.arch.grid_w)
    with pytest.raises(ValueError, match=r"stim must be \[B, T, 1, H, W\]"):
        extract_activations(net, bad)
