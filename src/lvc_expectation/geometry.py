"""Orientation geometry helpers for 180-degree periodic stimuli."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OrientationGeometry:
    n_orientations: int
    periodicity_deg: float = 180.0

    def __post_init__(self) -> None:
        if self.n_orientations <= 0:
            raise ValueError("n_orientations must be positive")
        if self.periodicity_deg != 180.0:
            raise ValueError("phase-1 geometry uses 180-degree periodic orientation")

    @property
    def step_deg(self) -> float:
        return self.periodicity_deg / self.n_orientations

    @property
    def bin_centers_deg(self) -> torch.Tensor:
        return torch.arange(self.n_orientations, dtype=torch.float32) * self.step_deg

    def wrap_index(self, index: int | torch.Tensor) -> int | torch.Tensor:
        if torch.is_tensor(index):
            return torch.remainder(index, self.n_orientations)
        return index % self.n_orientations

    def circular_distance_bins(self, src: int | torch.Tensor, dst: int | torch.Tensor) -> int | torch.Tensor:
        src_wrapped = self.wrap_index(src)
        dst_wrapped = self.wrap_index(dst)
        if torch.is_tensor(src_wrapped) or torch.is_tensor(dst_wrapped):
            reference = src_wrapped if torch.is_tensor(src_wrapped) else dst_wrapped
            assert torch.is_tensor(reference)
            src_tensor = torch.as_tensor(src_wrapped, device=reference.device, dtype=reference.dtype)
            dst_tensor = torch.as_tensor(dst_wrapped, device=reference.device, dtype=reference.dtype)
            diff = torch.abs(dst_tensor - src_tensor)
            return torch.minimum(diff, self.n_orientations - diff)
        diff = abs(dst_wrapped - src_wrapped)
        return min(diff, self.n_orientations - diff)

    def circular_distance_deg(self, src: int | torch.Tensor, dst: int | torch.Tensor) -> float | torch.Tensor:
        distances = self.circular_distance_bins(src, dst)
        if torch.is_tensor(distances):
            return distances.to(dtype=torch.float32) * self.step_deg
        return distances * self.step_deg

    def circular_offsets(self, center: int, device: torch.device | None = None) -> torch.Tensor:
        indices = torch.arange(self.n_orientations, device=device, dtype=torch.int64)
        center_idx = torch.full_like(indices, self.wrap_index(center))
        diffs = (indices - center_idx).abs()
        wrapped = torch.minimum(diffs, self.n_orientations - diffs)
        return wrapped.to(dtype=torch.float32) * self.step_deg

    def circular_shift(self, values: torch.Tensor, shift: int, dim: int = -1) -> torch.Tensor:
        return torch.roll(values, shifts=shift, dims=dim)

    def gaussian_kernel(self, center: int, width_deg: float, device: torch.device | None = None) -> torch.Tensor:
        distances = self.circular_offsets(center=center, device=device)
        kernel = torch.exp(-0.5 * (distances / max(width_deg, 1e-6)) ** 2)
        return kernel / kernel.sum().clamp_min(1e-6)

    def shifted_indices(self, shift: int) -> torch.Tensor:
        base = torch.arange(self.n_orientations, dtype=torch.int64)
        return self.circular_shift(base, shift=shift)
