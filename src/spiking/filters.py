"""Exponential spike trace filters for SNN populations.

Converts binary spike trains into smooth continuous-valued traces via a
first-order IIR:

    x[t] = alpha * x[t-1] + z[t]

where `alpha = exp(-dt / tau_filter)` is the discrete decay factor.

The filtered traces `x_*` serve as the interface to all downstream consumers
(losses, decoders, V2 input, analysis). They are a smoothed, continuous-valued
surrogate for instantaneous firing rate and carry gradient through the
surrogate-gradient spike nonlinearity during BPTT.

Evidence / rationale
--------------------
Formulation copied verbatim from the Phase 1 port plan:

    plans/quirky-humming-giraffe.md line 34:
        x[t] = alpha * x[t-1] + z[t]                         # filtered trace
    plans/quirky-humming-giraffe.md line 35:
        Where beta = exp(-1/tau_mem), alpha = exp(-1/tau_filter).

Time constants from the plan's time constant mapping table
(plans/quirky-humming-giraffe.md lines 46-55):

    Population       | Filter alpha (tau_filter, dt=1)
    -----------------+---------------------------------
    L4               | 0.905   (tau_filter = 10)
    L2/3             | 0.905   (tau = 10)
    SOM              | 0.905   (tau = 10)
    VIP              | 0.905   (tau = 10)
    V2 LSNN          | 0.905   (tau = 10)

Project convention: `dt = 1.0` timestep, same as the rate model
(`src/model/populations.py` uses `self.dt = cfg.dt` with the same unit).

Steady-state behaviour
----------------------
For a constant spike train `z[t] == 1`, the steady state is
    x* = 1 / (1 - alpha) ~= tau_filter      (for alpha close to 1)
For alpha = 0.905 this gives x* ≈ 10.5, which is tau_filter in the limit
of small `(1 - alpha)`. This unnormalized form (no `(1 - alpha)` prefactor on
the spike input) matches the plan verbatim; normalization is left to the
downstream decoder, which learns its own scale.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Tau <-> alpha conversion
# ---------------------------------------------------------------------------

def alpha_from_tau(tau: float, dt: float = 1.0) -> float:
    """Convert a filter time constant to its discrete-time decay factor.

    Args:
        tau: Time constant in the same units as `dt`. Must be > 0.
        dt: Simulation timestep. Default 1.0 (project convention).

    Returns:
        `alpha = exp(-dt / tau)`, in (0, 1). Larger alpha = slower decay.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return math.exp(-dt / tau)


def tau_from_alpha(alpha: float, dt: float = 1.0) -> float:
    """Inverse of `alpha_from_tau`: recover tau from a decay factor.

    Args:
        alpha: Decay factor in (0, 1).
        dt: Simulation timestep. Default 1.0.

    Returns:
        `tau = -dt / log(alpha)`.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return -dt / math.log(alpha)


# ---------------------------------------------------------------------------
# Stateless step + batch sequence filter
# ---------------------------------------------------------------------------

def exp_filter_step(
    x_prev: Tensor,
    spikes: Tensor,
    alpha: float | Tensor,
) -> Tensor:
    """One timestep of the exponential trace filter.

        x[t] = alpha * x[t-1] + z[t]

    Args:
        x_prev: Previous trace values. Any shape.
        spikes: Current spike input (typically binary {0, 1}),
            broadcastable to `x_prev`.
        alpha: Decay factor in (0, 1). Scalar float or tensor broadcastable
            to `x_prev` (e.g. per-channel learnable alphas).

    Returns:
        Updated trace with the same shape as `x_prev`.
    """
    return alpha * x_prev + spikes


def exp_filter_sequence(
    spikes: Tensor,
    alpha: float,
    x0: Optional[Tensor] = None,
) -> Tensor:
    """Apply the exponential trace filter to an entire spike-train sequence.

    Iterates `x[t] = alpha * x[t-1] + z[t]` over the time dimension.

    Intended for offline analysis/testing. Inside the step-wise network
    forward pass, use `exp_filter_step` (the iteration is driven by the
    outer simulation loop).

    Args:
        spikes: Spike train of shape `[B, T, N]` (batch, time, channels).
        alpha: Decay factor in (0, 1).
        x0: Initial state `[B, N]`. Defaults to zeros.

    Returns:
        Filtered traces with the same shape as `spikes`.
    """
    if spikes.ndim != 3:
        raise ValueError(
            f"spikes must be [B, T, N], got shape {tuple(spikes.shape)}"
        )
    B, T, N = spikes.shape
    if x0 is None:
        x = torch.zeros(B, N, dtype=spikes.dtype, device=spikes.device)
    else:
        if x0.shape != (B, N):
            raise ValueError(
                f"x0 must have shape [B, N]=[{B}, {N}], got {tuple(x0.shape)}"
            )
        x = x0
    out = torch.empty_like(spikes)
    for t in range(T):
        x = alpha * x + spikes[:, t]
        out[:, t] = x
    return out


# ---------------------------------------------------------------------------
# Module wrapper (fixed or learnable tau)
# ---------------------------------------------------------------------------

class ExponentialTraceFilter(nn.Module):
    """First-order exponential trace filter as an `nn.Module`.

    Implements `x[t] = alpha * x[t-1] + z[t]` with `alpha = exp(-dt/tau)`.

    tau can be fixed (stored as a buffer) or learnable (parameterised through
    softplus so it stays strictly positive under gradient updates).

    The Module is **stateless across calls** — state `x_prev` is passed in
    explicitly, matching the snnTorch convention and the rate model's
    per-step signatures.

    Args:
        tau: Filter time constant (same units as `dt`). Must be > 0.
        dt: Simulation timestep. Default 1.0.
        learnable: If True, `tau` becomes a learnable parameter via softplus.
            Default False.
    """

    def __init__(self, tau: float, dt: float = 1.0, learnable: bool = False):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.dt = float(dt)
        self.learnable = bool(learnable)

        if learnable:
            # Invert softplus so that softplus(raw_init) == tau.
            raw_init = math.log(math.expm1(tau))
            self.tau_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            # Fixed tau stored as a buffer (moves with .to(device))
            self.register_buffer(
                "_tau_fixed", torch.tensor(tau, dtype=torch.float32)
            )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def tau(self) -> Tensor:
        """Current tau value (scalar tensor)."""
        if self.learnable:
            return F.softplus(self.tau_raw)
        return self._tau_fixed

    @property
    def alpha(self) -> Tensor:
        """Current alpha = exp(-dt/tau)."""
        return torch.exp(-self.dt / self.tau)

    # ------------------------------------------------------------------
    # Forward: one step
    # ------------------------------------------------------------------

    def forward(self, spikes: Tensor, x_prev: Tensor) -> Tensor:
        """One timestep update.

        Args:
            spikes: Current spike input. Any shape broadcastable to `x_prev`.
            x_prev: Previous trace state.

        Returns:
            Updated trace of the same shape as `x_prev`.
        """
        return self.alpha * x_prev + spikes

    def extra_repr(self) -> str:
        mode = "learnable" if self.learnable else "fixed"
        return (
            f"tau={self.tau.item():.4f} ({mode}), "
            f"dt={self.dt}, alpha={self.alpha.item():.4f}"
        )
