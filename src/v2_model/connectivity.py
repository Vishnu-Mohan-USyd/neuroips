"""Sparse like-to-like connectivity generator (Phase 1 step 5 / Task #11).

Per v4 spec §Connectivity:
    P(i → j) ∝ exp(−d_retinotopy² / 2σ_r²) · exp(−Δθ² / 2σ_θ²)
    target sparsity ≈ 12 %; plasticity preserves the mask (only existing edges
    can be updated).

This module is a pure utility — no `nn.Module`, no state, no forward pass.
It returns tensors that downstream layers consume.

Conventions
-----------
**Mask orientation**: `mask[i, j] = True` iff there is a directed edge from
*presynaptic* unit `i` to *postsynaptic* unit `j`. The returned matrix is
asymmetric in content: `i → j` does not imply `j → i`. The diagonal is
always `False` (no self-connections).

**Exact row-wise sparsity**: each presynaptic row samples exactly
`k = round(sparsity · n_units)` postsynaptic partners via multinomial sampling
without replacement, weighted by the Gaussian-falloff propensity. This yields
sparsity within ±1 % of the target deterministically (no rejection sampling
loops), plus a deterministic per-row out-degree that is convenient for tests.

**Storage format**: dense `torch.bool` `[n_units, n_units]` tensor. For v2
population sizes (L2/3 E = 256 → 64 KB, H E = 64 → 4 KB), a dense bool is
simpler and faster than CSR; the memory saving of sparse storage is
negligible at this scale, and most downstream ops (Hadamard with a weight
tensor) prefer dense masks.

**Dale constraint**: `initialize_masked_weights` uses the same softplus
parameterization as `src/utils.py::ExcitatoryLinear`: raw ∼ 𝒩(0, scale²)
then weight = ±softplus(raw). Excitatory → +softplus ≥ 0; inhibitory →
−softplus ≤ 0; `None` → raw (unconstrained sign). Non-edge entries are
exactly zero.

Out of scope
------------
L2/3 and H population classes (→ `layers.py`), plasticity updates
(→ `plasticity.py`), any network step (→ `network.py`).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.utils import circular_distance_abs

__all__ = ["generate_sparse_mask", "initialize_masked_weights"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pairwise_squared_distance(positions: Tensor) -> Tensor:
    """Pairwise squared Euclidean distance.

    Args:
        positions: [n, D] coordinates.

    Returns:
        [n, n] squared distance tensor (diagonal = 0).
    """
    # Use torch.cdist for numerical stability and to avoid manual subtraction.
    d = torch.cdist(positions, positions, p=2.0)
    return d * d


def _build_propensity(
    positions: Optional[Tensor],
    features: Optional[Tensor],
    n_units: int,
    sigma_position: Optional[float],
    sigma_feature: Optional[float],
    feature_period_deg: float,
    device: torch.device,
) -> Tensor:
    """Build the unnormalised propensity matrix P[i, j] ≥ 0.

    If both `positions` and `features` are None, returns an all-ones matrix
    (uniform-random fallback). Diagonal is always zeroed to forbid self-edges.
    """
    P = torch.ones(n_units, n_units, device=device, dtype=torch.float32)

    if positions is not None:
        if sigma_position is None or sigma_position <= 0.0:
            raise ValueError(
                f"positions provided; sigma_position must be > 0 (got {sigma_position})"
            )
        pos = positions.to(device=device, dtype=torch.float32)
        d2 = _pairwise_squared_distance(pos)
        P = P * torch.exp(-d2 / (2.0 * float(sigma_position) ** 2))

    if features is not None:
        if sigma_feature is None or sigma_feature <= 0.0:
            raise ValueError(
                f"features provided; sigma_feature must be > 0 (got {sigma_feature})"
            )
        feat = features.to(device=device, dtype=torch.float32)
        delta = circular_distance_abs(
            feat.unsqueeze(1), feat.unsqueeze(0), period=feature_period_deg
        )
        P = P * torch.exp(-delta * delta / (2.0 * float(sigma_feature) ** 2))

    # Forbid self-connections by construction.
    P.fill_diagonal_(0.0)
    return P


# ---------------------------------------------------------------------------
# Public API: mask generator
# ---------------------------------------------------------------------------

def generate_sparse_mask(
    positions: Optional[Tensor],
    features: Optional[Tensor],
    n_units: int,
    sparsity: float,
    sigma_position: Optional[float] = None,
    sigma_feature: Optional[float] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
    feature_period_deg: float = 180.0,
) -> Tensor:
    """Build a sparse, like-to-like, Dale-agnostic boolean connectivity mask.

    Sampling: for each presynaptic row *i*, draw exactly
    `k = round(sparsity · n_units)` postsynaptic partners without replacement
    with probability ∝ `exp(−d² / 2σ_r²) · exp(−Δθ² / 2σ_θ²)`. Self-connection
    is excluded because the diagonal of the propensity is zero.

    Args:
        positions: `[n_units, 2]` retinotopic coordinates (pixels) or `None`
            for non-retinotopic populations (e.g. H).
        features: `[n_units]` feature angle in *degrees* (e.g. orientation
            preference) or `None`. Period controlled by `feature_period_deg`.
        n_units: Number of units (rows = columns).
        sparsity: Target density ∈ (0, 1). Exactly
            `round(sparsity · n_units)` edges per row.
        sigma_position: Gaussian width in pixels; required if `positions` is not None.
        sigma_feature: Gaussian width in degrees; required if `features` is not None.
        seed: RNG seed for `torch.multinomial` → determinism.
        device: Torch device for the returned mask. Defaults to CPU.
        feature_period_deg: Period for the circular distance over `features`.
            Default 180° (orientation). Use 360° for direction-tuned features.

    Returns:
        `mask: [n_units, n_units]` `torch.bool`. `mask[i, j] = True` iff an
        edge `i → j` exists. Diagonal is `False`. Each row sums to exactly
        `round(sparsity · n_units)` (clamped to `[1, n_units - 1]`).

    Raises:
        ValueError: on invalid sparsity, bad tensor shapes, missing sigma.
    """
    if not (0.0 < sparsity < 1.0):
        raise ValueError(f"sparsity must be in (0, 1); got {sparsity}")
    if n_units < 2:
        raise ValueError(f"n_units must be ≥ 2; got {n_units}")
    if positions is not None and positions.shape != (n_units, 2):
        raise ValueError(
            f"positions must have shape [{n_units}, 2]; got {tuple(positions.shape)}"
        )
    if features is not None and features.shape != (n_units,):
        raise ValueError(
            f"features must have shape [{n_units}]; got {tuple(features.shape)}"
        )

    dev = device if device is not None else torch.device("cpu")

    P = _build_propensity(
        positions=positions,
        features=features,
        n_units=n_units,
        sigma_position=sigma_position,
        sigma_feature=sigma_feature,
        feature_period_deg=feature_period_deg,
        device=dev,
    )

    # Clamp per-row out-degree to a valid range.
    k = int(round(sparsity * n_units))
    k = max(1, min(k, n_units - 1))

    # If some row has all-zero propensity (Gaussian collapse for very narrow σ),
    # fall back to a uniform prior for that row so multinomial is well-defined.
    row_sums = P.sum(dim=1, keepdim=True)
    dead_rows = row_sums.squeeze(-1) < 1e-30
    if dead_rows.any():
        uniform = torch.ones_like(P)
        uniform.fill_diagonal_(0.0)
        P = torch.where(dead_rows.unsqueeze(-1), uniform, P)

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    # Vectorised row-wise multinomial without replacement.
    indices = torch.multinomial(
        P, num_samples=k, replacement=False, generator=gen
    )  # [n_units, k]

    mask = torch.zeros(n_units, n_units, dtype=torch.bool, device=dev)
    row_idx = torch.arange(n_units, device=dev).unsqueeze(-1).expand(-1, k)
    mask[row_idx, indices] = True
    return mask


# ---------------------------------------------------------------------------
# Public API: masked weight initialiser
# ---------------------------------------------------------------------------

def initialize_masked_weights(
    mask: Tensor,
    scale: float,
    dale_sign: Optional[Literal["excitatory", "inhibitory"]],
    seed: int = 0,
) -> Tensor:
    """Initialise a Dale-constrained weight tensor consistent with a mask.

    Parameterisation matches `src/utils.py::ExcitatoryLinear`: draw raw weights
    `raw ∼ 𝒩(0, scale²)` and map them through softplus (excitatory) or
    negated softplus (inhibitory). Non-edge entries are zeroed exactly.

    Args:
        mask: `[M, N]` `torch.bool` (typically from `generate_sparse_mask`).
        scale: Std-dev of the raw-weight prior (pre-softplus).
        dale_sign: `"excitatory"` → all nonzero weights ≥ 0;
            `"inhibitory"` → all nonzero weights ≤ 0;
            `None` → weights keep the raw Gaussian sign (no Dale constraint).
        seed: RNG seed for reproducibility.

    Returns:
        `weights: [M, N]` `torch.float32`. Zero wherever `mask` is `False`.

    Raises:
        ValueError: on invalid `mask.dtype` or unknown `dale_sign`.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be torch.bool; got {mask.dtype}")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0; got {scale}")

    dev = mask.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    raw = torch.empty(mask.shape, dtype=torch.float32, device=dev)
    raw.normal_(mean=0.0, std=float(scale), generator=gen)

    if dale_sign == "excitatory":
        w = F.softplus(raw)
    elif dale_sign == "inhibitory":
        w = -F.softplus(raw)
    elif dale_sign is None:
        w = raw
    else:
        raise ValueError(
            f"dale_sign must be 'excitatory', 'inhibitory', or None; "
            f"got {dale_sign!r}"
        )

    zero = torch.zeros((), dtype=w.dtype, device=dev)
    return torch.where(mask, w, zero)
