"""ATan surrogate gradient for spiking neurons (torch.compile compatible).

Implements the Heaviside step function with an analytical ATan surrogate
gradient via the straight-through estimator (STE) pattern built from pure
tensor ops. There is NO custom `torch.autograd.Function` here, so the
operation composes cleanly under `torch.compile(..., fullgraph=True)`.

Evidence / citations
--------------------
Formula (backward pass, with optional BPTT dampening):

    dS/dU = dampen * (1/pi) * 1 / (1 + (pi * slope * U)^2)

This matches the analytical derivative documented in snnTorch's ATan
surrogate docstring:

    snntorch/surrogate.py, class ATan, lines 170-173
        S     ≈ (1/pi) * arctan(pi * U * alpha/2)
        dS/dU = (1/pi) * 1 / (1 + (pi * U * alpha/2)^2)

Our `slope` parameter corresponds to snnTorch's `alpha/2`. Peak gradient is
`dampen * 1/pi`, attained at U = 0, regardless of slope; `slope` controls the
width of the active region (larger slope -> narrower peak).

**Dampening**: Bellec et al. 2018 §3 (NeurIPS 2018, page 3) explicitly
reduces the pseudo-derivative amplitude by a factor < 1 to stabilize BPTT
through recurrent spiking nets over "several 1000 layers" — necessary when
the unrolled horizon is in the hundreds to low thousands of steps. The
LSNN-official reference implementation uses `dampening_factor = 0.3` for
both the sMNIST tutorial (`bin/tutorial_sequential_mnist_with_LSNN.py`) and
the `lsnn/spiking_models.py` ALIF class default. Our Stage-2 BPTT window is
600 timesteps, so dampening is enabled by default via `SpikingConfig.
surrogate_dampen = 0.3`.

Original reference for the surrogate form:

    W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, Y. Tian (2021).
    "Incorporating Learnable Membrane Time Constants to Enhance Learning
    of Spiking Neural Networks." Proc. IEEE/CVF ICCV, pp. 2661-2671.

Default slope of 25 taken from the Phase 1 SNN port plan
(`plans/quirky-humming-giraffe.md`, "Surrogate Gradient" section, line 213).

Forward pass convention
-----------------------
`z = 1` if `x > 0` else `0` (strict inequality — at exact threshold we emit
no spike). This matches snnTorch's `ATan.forward`:

    snntorch/surrogate.py, line 190
        out = (input_ > 0).float()

Why pure tensor ops (not torch.autograd.Function)
-------------------------------------------------
`torch.compile` with `fullgraph=True` cannot trace custom autograd.Functions
in general and must fall back to eager mode for the wrapped op. Using the
STE identity `y = forward + (surrogate - surrogate.detach())` keeps the
whole op inside the compiled graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def atan_surrogate(x: Tensor, slope: float = 25.0, dampen: float = 1.0) -> Tensor:
    """Heaviside step function with (dampened) ATan surrogate gradient.

    Args:
        x: Pre-threshold input, e.g. `V_mem - V_thresh`. Any shape.
        slope: Width parameter for the gradient. Larger slope -> narrower,
            sharper peak. Peak value (at `x=0`) without dampening is `1/pi`
            independent of `slope`; `slope` controls only the width of the
            active region. Default 25.0 matches the Phase 1 plan.
        dampen: Multiplicative factor on the surrogate gradient. Default 1.0
            (no dampening). Set to 0.3 (Bellec 2018 §3, LSNN-official
            implementation) for long BPTT windows to keep backprop stable.
            The forward pass is unchanged; only the backward gradient is
            scaled.

    Returns:
        Binary tensor (same shape and dtype as `x`) with values in {0, 1}.
        Forward value is exactly Heaviside(x); backward gradient is
        `dampen * (1/pi) * 1 / (1 + (pi * slope * x)^2)`.

    Notes:
        STE trick: pick a smooth antiderivative `g(x)` whose derivative is the
        desired gradient, then return `heaviside + (g - g.detach())`. The
        `g - g.detach()` term is identically zero in forward but carries the
        desired gradient in backward. The Heaviside term has no gradient
        (comparison op), so autograd sees only `g`.

        Antiderivative used (with dampening folded into the prefactor):
            g(x) = (dampen / (pi^2 * slope)) * atan(pi * slope * x)
        Check:
            dg/dx = (dampen / (pi^2 * slope)) * (pi * slope) / (1 + (pi*slope*x)^2)
                  = dampen * (1/pi) * 1 / (1 + (pi * slope * x)^2)   [desired]
    """
    pi = math.pi
    # Forward: Heaviside step (no gradient — comparison + cast).
    heaviside = (x > 0).to(x.dtype)
    # Smooth antiderivative carrying the desired (dampened) gradient.
    g = (dampen / (pi * pi * slope)) * torch.atan(pi * slope * x)
    # Straight-through estimator.
    return heaviside + (g - g.detach())


class ATanSurrogate(nn.Module):
    """`nn.Module` wrapper around `atan_surrogate` for use as a spike function.

    Suitable as a drop-in `spike_grad` callable for snnTorch neurons, or as a
    standalone spiking nonlinearity inside a custom LIF/ALIF implementation.

    Args:
        slope: See `atan_surrogate`. Default 25.0.
        dampen: See `atan_surrogate`. Default 1.0 (no dampening). Set to 0.3
            for Bellec 2018 §3 BPTT stabilization on long windows.
    """

    def __init__(self, slope: float = 25.0, dampen: float = 1.0):
        super().__init__()
        self.slope = float(slope)
        self.dampen = float(dampen)

    def forward(self, x: Tensor) -> Tensor:
        return atan_surrogate(x, self.slope, self.dampen)

    def extra_repr(self) -> str:
        return f"slope={self.slope}, dampen={self.dampen}"
