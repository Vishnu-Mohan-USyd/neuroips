"""Utility functions: circular distance, circular Gaussian kernels,
shifted softplus, sign-constrained layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Circular distance
# ---------------------------------------------------------------------------

def circular_distance(a: Tensor, b: Tensor, period: float = 180.0) -> Tensor:
    """Signed shortest-arc distance on a circle of given period.

    Returns values in [-period/2, period/2].
    """
    diff = (a - b) % period
    return torch.where(diff > period / 2, diff - period, diff)


def circular_distance_abs(a: Tensor, b: Tensor, period: float = 180.0) -> Tensor:
    """Unsigned shortest-arc distance on a circle."""
    return circular_distance(a, b, period).abs()


# ---------------------------------------------------------------------------
# Circular Gaussian kernels
# ---------------------------------------------------------------------------

def circular_gaussian(delta: Tensor, sigma: float) -> Tensor:
    """Evaluate circular Gaussian: exp(-delta^2 / (2*sigma^2)).

    Args:
        delta: Angular distances (degrees).
        sigma: Width parameter (degrees).
    """
    return torch.exp(-delta ** 2 / (2.0 * sigma ** 2))


def make_circular_gaussian_kernel(
    n: int, sigma: float, period: float = 180.0, row_normalise: bool = False,
) -> Tensor:
    """Build an [n, n] circulant kernel matrix from a circular Gaussian.

    Entry (i, j) = exp(-circ_dist(theta_i, theta_j)^2 / (2*sigma^2)).

    Args:
        n: Number of orientation channels.
        sigma: Kernel width (degrees).
        period: Orientation period.
        row_normalise: If True, each row sums to 1.
    """
    step = period / n
    thetas = torch.arange(n, dtype=torch.float32) * step  # [n]
    # Pairwise circular distances [n, n]
    diffs = circular_distance_abs(
        thetas.unsqueeze(1), thetas.unsqueeze(0), period=period
    )
    K = circular_gaussian(diffs, sigma)
    if row_normalise:
        K = K / K.sum(dim=1, keepdim=True)
    return K


def circular_gaussian_fwhm(sigma: float) -> float:
    """Analytically compute FWHM of a Gaussian with given sigma.

    FWHM = 2 * sigma * sqrt(2 * ln(2)) ~ 2.355 * sigma.
    """
    return 2.0 * sigma * math.sqrt(2.0 * math.log(2.0))


# ---------------------------------------------------------------------------
# Shifted softplus
# ---------------------------------------------------------------------------

_SOFTPLUS_ZERO = F.softplus(torch.zeros(1)).item()


def shifted_softplus(x: Tensor) -> Tensor:
    """Softplus shifted so that shifted_softplus(0) = 0 exactly.

    NOTE: Can return negative values for x < 0 (down to -0.693).
    Use rectified_softplus for firing rate activations.
    """
    return F.softplus(x) - _SOFTPLUS_ZERO


def rectified_softplus(x: Tensor) -> Tensor:
    """Shifted softplus clamped to non-negative. f(0)=0, smooth for x>0."""
    return F.relu(F.softplus(x) - _SOFTPLUS_ZERO)


# ---------------------------------------------------------------------------
# Sign-constrained layers (Dale's law)
# ---------------------------------------------------------------------------

class ExcitatoryLinear(nn.Module):
    """Linear layer with non-negative weights (Dale's law excitatory).

    Stores raw (unconstrained) weights; applies softplus in forward pass.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Initialize so that softplus(raw) ~ small positive values
        nn.init.normal_(self.weight_raw, mean=0.0, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        w = F.softplus(self.weight_raw)
        return F.linear(x, w, self.bias)


class InhibitoryGain(nn.Module):
    """Learnable non-negative scalar gain for inhibitory connections.

    Output: gain * input, where gain = softplus(gain_raw).
    Used with subtraction at the call site: drive = ... - inhib_gain(pooled_activity).
    """

    def __init__(self, init_gain: float = 1.0):
        super().__init__()
        # Invert softplus to initialize at desired gain
        raw_init = math.log(math.exp(init_gain) - 1.0)
        self.gain_raw = nn.Parameter(torch.tensor(raw_init))

    @property
    def gain(self) -> Tensor:
        return F.softplus(self.gain_raw)

    def forward(self, x: Tensor) -> Tensor:
        return self.gain * x
