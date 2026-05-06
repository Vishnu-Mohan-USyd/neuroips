"""Analysis helpers for orientation tuning and connectivity summaries."""

from __future__ import annotations

import math

import torch

from .connectivity import SparseProjection
from .layout import NetworkLayout


def circular_orientation_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute unsigned orientation distance on a pi-periodic manifold."""

    delta = torch.remainder(a - b + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    return delta.abs()


def compute_osi(rates: torch.Tensor, orientations_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute OSI and preferred orientation from per-stimulus firing rates."""

    complex_kernel = torch.exp(2j * orientations_rad.to(torch.complex64))
    numerator = (rates.to(torch.complex64) * complex_kernel[:, None]).sum(dim=0)
    denominator = rates.sum(dim=0).clamp_min(1e-6)
    vector_strength = numerator / denominator.to(torch.complex64)
    osi = vector_strength.abs().to(torch.float32)
    pref = 0.5 * torch.angle(vector_strength).to(torch.float32)
    return osi, torch.remainder(pref, math.pi)


def projection_orientation_bias(projection: SparseProjection, layout: NetworkLayout) -> float:
    """Measure the mean orientation mismatch across a projection's sampled edges."""

    pre = layout.populations[projection.source]
    post = layout.populations[projection.target]
    pre_orientation = pre.preferred_orientation.index_select(0, projection.pre_indices)
    post_orientation = post.preferred_orientation.index_select(0, projection.post_indices)
    mask = torch.isfinite(pre_orientation) & torch.isfinite(post_orientation)
    if not torch.any(mask):
        return float("nan")
    mismatch = circular_orientation_distance(pre_orientation[mask], post_orientation[mask])
    return float(mismatch.mean().item())
