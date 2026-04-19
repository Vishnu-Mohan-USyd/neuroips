"""Shape, dtype, and diagonal invariants for
`src.v2_model.connectivity.generate_sparse_mask`.

Covers:
  * returned mask is [n_units, n_units] bool tensor.
  * diagonal is all False (no self-connections).
  * mask is asymmetric in content (`i → j` does not imply `j → i`).
  * per-row out-degree is exactly round(sparsity · n_units).
  * ValueError is raised for malformed inputs.
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.connectivity import generate_sparse_mask


# ---------------------------------------------------------------------------
# Happy-path shape / dtype / diagonal
# ---------------------------------------------------------------------------

def test_mask_shape(cfg) -> None:
    """[n, n] output."""
    n = 32
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n, sparsity=0.12, seed=0
    )
    assert mask.shape == (n, n)


def test_mask_dtype_is_bool() -> None:
    """torch.bool output."""
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=32, sparsity=0.12, seed=0
    )
    assert mask.dtype == torch.bool


def test_mask_diagonal_false() -> None:
    """No self-connections — diagonal is all False."""
    n = 64
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n, sparsity=0.2, seed=0
    )
    assert not mask.diagonal().any(), (
        f"diagonal contains {int(mask.diagonal().sum().item())} True entries"
    )


def test_per_row_out_degree_is_exact_k() -> None:
    """Each row has exactly round(sparsity · n_units) True entries."""
    n = 128
    sparsity = 0.12
    expected_k = round(sparsity * n)
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=n, sparsity=sparsity, seed=0
    )
    row_degrees = mask.sum(dim=1)
    assert torch.all(row_degrees == expected_k), (
        f"row degrees not all {expected_k}: "
        f"min={row_degrees.min().item()}, max={row_degrees.max().item()}"
    )


def test_mask_content_asymmetric_in_general() -> None:
    """With retinotopy + orientation structure, i→j ≠ j→i in general.

    Constructs a large-enough population that random sampling diverges the
    two directions; asserts at least one asymmetric pair exists. If the
    sampler ever became secretly symmetric this test catches it.
    """
    n = 32
    positions = torch.rand(n, 2) * 16.0
    features = torch.rand(n) * 180.0
    mask = generate_sparse_mask(
        positions=positions, features=features, n_units=n, sparsity=0.2,
        sigma_position=4.0, sigma_feature=25.0, seed=7
    )
    mismatch = (mask != mask.T).any().item()
    assert mismatch, "mask content is unexpectedly symmetric."


def test_device_default_is_cpu() -> None:
    """Default device is CPU when no device is passed."""
    mask = generate_sparse_mask(
        positions=None, features=None, n_units=16, sparsity=0.25, seed=0
    )
    assert mask.device.type == "cpu"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_sparsity", [0.0, 1.0, -0.01, 1.5])
def test_rejects_out_of_range_sparsity(bad_sparsity: float) -> None:
    with pytest.raises(ValueError, match="sparsity"):
        generate_sparse_mask(
            positions=None, features=None, n_units=32, sparsity=bad_sparsity, seed=0
        )


def test_rejects_n_units_too_small() -> None:
    with pytest.raises(ValueError, match="n_units"):
        generate_sparse_mask(
            positions=None, features=None, n_units=1, sparsity=0.5, seed=0
        )


def test_rejects_bad_positions_shape() -> None:
    bad_pos = torch.zeros(10, 3)  # should be [n, 2]
    with pytest.raises(ValueError, match="positions"):
        generate_sparse_mask(
            positions=bad_pos, features=None, n_units=10, sparsity=0.2,
            sigma_position=4.0, seed=0
        )


def test_rejects_bad_features_shape() -> None:
    bad_feat = torch.zeros(10, 2)  # should be [n]
    with pytest.raises(ValueError, match="features"):
        generate_sparse_mask(
            positions=None, features=bad_feat, n_units=10, sparsity=0.2,
            sigma_feature=25.0, seed=0
        )


def test_rejects_missing_sigma_position() -> None:
    pos = torch.zeros(8, 2)
    with pytest.raises(ValueError, match="sigma_position"):
        generate_sparse_mask(
            positions=pos, features=None, n_units=8, sparsity=0.25,
            sigma_position=None, seed=0
        )


def test_rejects_missing_sigma_feature() -> None:
    feat = torch.zeros(8)
    with pytest.raises(ValueError, match="sigma_feature"):
        generate_sparse_mask(
            positions=None, features=feat, n_units=8, sparsity=0.25,
            sigma_feature=None, seed=0
        )
