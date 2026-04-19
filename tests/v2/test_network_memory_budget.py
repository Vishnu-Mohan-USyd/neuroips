"""Total parameter count stays within an order-of-magnitude sanity budget.

Architecture counts (n_l23_e=256, n_h_e=64, n_c=48, ...) imply the
network holds ~0.25 M plastic Parameters. This budget-tracking test
catches accidental expansion — e.g. if someone registers a large dense
weight by mistake, the count jumps by 10× and this test fails.

Upper bound: 1 M Parameters. Actual today is ≈ 245 k.
"""

from __future__ import annotations

import pytest

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


def test_total_parameter_count_within_budget(net):
    """Total nn.Parameter count is under 1 M (actual ≈ 245k)."""
    total = sum(p.numel() for p in net.parameters())
    assert 100_000 < total < 1_000_000, (
        f"parameter count {total:,} is out of expected range "
        "(100k–1M); something structural changed"
    )


def test_lgn_l4_zero_parameters(net):
    """LGN/L4 front end must hold zero nn.Parameters (frozen by construction)."""
    assert sum(1 for _ in net.lgn_l4.parameters()) == 0


def test_no_parameter_requires_grad(net):
    """Every Parameter has ``requires_grad=False`` — pure local plasticity."""
    for p in net.parameters():
        assert p.requires_grad is False
