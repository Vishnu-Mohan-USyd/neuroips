"""Synthetic orientation stimulus helpers."""

from __future__ import annotations

import torch

from .geometry import OrientationGeometry


def orientation_to_one_hot(orientations: torch.Tensor, n_orientations: int) -> torch.Tensor:
    safe = orientations.clamp_min(0)
    one_hot = torch.nn.functional.one_hot(safe, num_classes=n_orientations).to(torch.float32)
    blank_mask = orientations.lt(0).unsqueeze(-1)
    return torch.where(blank_mask, torch.zeros_like(one_hot), one_hot)


def orientation_to_population_code(
    orientations: torch.Tensor,
    n_orientations: int,
    ambiguity_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    base = orientation_to_one_hot(orientations, n_orientations)
    if ambiguity_weights is None:
        return base
    if ambiguity_weights.shape != base.shape:
        raise ValueError("ambiguity_weights must match the one-hot population shape")
    normalized = ambiguity_weights.to(dtype=torch.float32)
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    blank_mask = orientations.lt(0).unsqueeze(-1)
    return torch.where(blank_mask, torch.zeros_like(normalized), normalized)


def broad_orientation_bank(geometry: OrientationGeometry, width_deg: float) -> torch.Tensor:
    kernels = [geometry.gaussian_kernel(center=index, width_deg=width_deg) for index in range(geometry.n_orientations)]
    return torch.stack(kernels, dim=0)
