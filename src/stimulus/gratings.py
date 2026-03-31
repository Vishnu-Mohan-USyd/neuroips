"""Population-coded grating generator with Naka-Rushton contrast response."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils import circular_distance_abs, circular_gaussian

__all__ = ["population_code", "naka_rushton", "generate_grating", "make_ambiguous_stimulus"]


def population_code(
    orientation: Tensor,
    n_orientations: int = 36,
    sigma: float = 12.0,
    period: float = 180.0,
) -> Tensor:
    """Generate population-coded response to a grating orientation.

    Args:
        orientation: Stimulus orientation in degrees, shape [B] or scalar.
        n_orientations: Number of orientation channels.
        sigma: Tuning width (degrees) for circular Gaussian.
        period: Orientation period (180 for orientation).

    Returns:
        Population vector [B, n_orientations] with peak at preferred orientation
        closest to stimulus.
    """
    if orientation.dim() == 0:
        orientation = orientation.unsqueeze(0)

    step = period / n_orientations
    # Preferred orientations: [n_orientations]
    prefs = torch.arange(n_orientations, dtype=torch.float32, device=orientation.device) * step
    # Circular distances: [B, n_orientations]
    dists = circular_distance_abs(
        orientation.unsqueeze(-1),  # [B, 1]
        prefs.unsqueeze(0),         # [1, n_orientations]
        period=period,
    )
    return circular_gaussian(dists, sigma)


def naka_rushton(
    contrast: Tensor,
    n: float = 2.0,
    c50: float = 0.3,
) -> Tensor:
    """Naka-Rushton contrast response function.

    gain = contrast^n / (contrast^n + c50^n)

    Args:
        contrast: Contrast values in [0, 1], shape [B] or scalar.
        n: Exponent (default 2).
        c50: Half-saturation contrast (default 0.3).

    Returns:
        Contrast gain, same shape as input.
    """
    cn = contrast ** n
    return cn / (cn + c50 ** n)


def generate_grating(
    orientation: Tensor,
    contrast: Tensor,
    n_orientations: int = 36,
    sigma: float = 12.0,
    n: float = 2.0,
    c50: float = 0.3,
    period: float = 180.0,
) -> Tensor:
    """Generate a contrast-scaled population-coded grating stimulus.

    Args:
        orientation: Stimulus orientation in degrees [B].
        contrast: Stimulus contrast in [0, 1] [B].
        n_orientations: Number of orientation channels.
        sigma: Tuning width (degrees).
        n: Naka-Rushton exponent.
        c50: Naka-Rushton half-saturation contrast.
        period: Orientation period.

    Returns:
        Stimulus tensor [B, n_orientations].
    """
    pop = population_code(orientation, n_orientations, sigma, period)
    gain = naka_rushton(contrast, n, c50)
    # gain: [B], pop: [B, n_orientations]
    return gain.unsqueeze(-1) * pop


def make_ambiguous_stimulus(
    theta1: Tensor,
    theta2: Tensor,
    contrast: Tensor,
    n_orientations: int = 36,
    sigma: float = 12.0,
    n: float = 2.0,
    c50: float = 0.3,
    period: float = 180.0,
    weight: float = 0.5,
) -> Tensor:
    """Create an ambiguous stimulus as a mixture of two orientations.

    Args:
        theta1: First orientation [B].
        theta2: Second orientation [B].
        contrast: Contrast [B].
        weight: Mixing weight for theta1 (1-weight for theta2).

    Returns:
        Mixed stimulus [B, n_orientations].
    """
    s1 = generate_grating(theta1, contrast, n_orientations, sigma, n, c50, period)
    s2 = generate_grating(theta2, contrast, n_orientations, sigma, n, c50, period)
    return weight * s1 + (1.0 - weight) * s2
