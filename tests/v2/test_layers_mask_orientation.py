"""Sparse recurrence mask is stored ``[n_post, n_pre]`` — design note #8.

``connectivity.generate_sparse_mask`` returns ``[n_pre, n_post]``; each
population must transpose-and-make-contiguous at init so that
``F.linear`` (which expects weight-orientation ``[out, in]``) sees the
mask on the same axes as the weight tensor.
"""

from __future__ import annotations

import torch

from src.v2_model.connectivity import generate_sparse_mask
from src.v2_model.layers import HE, L23E


# ---------------------------------------------------------------------------
# L23E
# ---------------------------------------------------------------------------

def test_l23e_mask_shape_is_n_post_n_pre() -> None:
    pop = L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
    )
    assert pop.mask_rec.shape == (pop.n_units, pop.n_units)


def test_l23e_mask_is_transpose_of_generate_sparse_mask() -> None:
    """Reproduce the raw mask and confirm the pop stored its transpose."""
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
         [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    )
    features = torch.linspace(0.0, 157.5, 8)
    pop = L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0,
        positions=positions, features=features,
        sparsity=0.5, sigma_position=1.0, sigma_feature=20.0, seed=42,
    )
    raw_mask = generate_sparse_mask(
        positions=positions, features=features, n_units=8,
        sparsity=0.5, sigma_position=1.0, sigma_feature=20.0, seed=42,
    )
    expected = raw_mask.t().contiguous().to(dtype=pop.mask_rec.dtype)
    assert torch.equal(pop.mask_rec, expected)


def test_l23e_mask_density_matches_configured_sparsity() -> None:
    """Sparsity is preserved through the transpose (column-sums ≡ row-sums of raw)."""
    pop = L23E(
        n_units=32, n_l4_e=8, n_pv=4, n_som=4, n_h_e=4,
        tau_ms=20.0, dt_ms=5.0,
        sparsity=0.25, seed=0,
    )
    expected_edges_per_row = round(0.25 * 32)
    # After transpose, rows are post-cells; each column is one pre-cell that
    # still has exactly ``expected_edges_per_row`` outgoing edges.
    col_sums = pop.mask_rec.sum(dim=0)
    assert torch.all(col_sums == float(expected_edges_per_row))


# ---------------------------------------------------------------------------
# HE
# ---------------------------------------------------------------------------

def test_he_mask_shape_is_n_post_n_pre() -> None:
    pop = HE(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0)
    assert pop.mask_rec.shape == (pop.n_units, pop.n_units)


def test_he_mask_is_transpose_of_generate_sparse_mask() -> None:
    pop = HE(
        n_units=6, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0,
        sparsity=0.5, seed=7,
    )
    raw_mask = generate_sparse_mask(
        positions=None, features=None, n_units=6, sparsity=0.5, seed=7,
    )
    expected = raw_mask.t().contiguous().to(dtype=pop.mask_rec.dtype)
    assert torch.equal(pop.mask_rec, expected)
