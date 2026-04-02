"""Fixed observation pooling for held-out evaluation."""

from __future__ import annotations

import torch

from .config import ExperimentConfig
from .geometry import OrientationGeometry
from .stimuli import broad_orientation_bank


class ObservationPool:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        geometry = OrientationGeometry(config.geometry.n_orientations)
        self.bank = broad_orientation_bank(geometry, config.observation.bank_width_deg)

    def apply(self, activity: torch.Tensor, scheme: str) -> torch.Tensor:
        if scheme == "identity":
            return activity.clone()
        if scheme == "gaussian_orientation_bank":
            return torch.einsum("bto,po->btp", activity, self.bank.to(activity.device))
        raise ValueError(f"unknown observation scheme: {scheme}")

    def apply_all(self, activity: torch.Tensor) -> dict[str, torch.Tensor]:
        return {scheme: self.apply(activity, scheme) for scheme in self.config.observation.schemes}


class ObservationPooler(ObservationPool):
    """Compatibility alias used by the model contract."""


class ObservationProjector:
    """Fixed projector used by held-out evaluation helpers."""

    def __init__(self, geometry: OrientationGeometry, width_deg: float = 30.0) -> None:
        self.geometry = geometry
        self.bank = broad_orientation_bank(geometry, width_deg)

    def project_all(self, activity: torch.Tensor) -> dict[str, torch.Tensor]:
        broad_pool = activity.mean(dim=-1, keepdim=True).expand_as(activity)
        return {
            "identity": activity,
            "broad_orientation_pool": broad_pool,
            "gaussian_orientation_bank": torch.einsum("bto,po->btp", activity, self.bank.to(activity.device)),
        }
