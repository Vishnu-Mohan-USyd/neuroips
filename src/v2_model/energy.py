"""Circuit-wide metabolic / energy penalty (v4 plan D.18).

Two terms, both **pathway-agnostic** (applied uniformly to feedforward,
recurrent, and feedback synapses). This uniformity is the load-bearing v4
patch: an earlier draft made the L2 current penalty feedback-specific,
which would have biased the model against the very mechanism under
study. The API therefore exposes no pathway argument, and
`test_energy_is_global.py` proves that identical statistics yield
identical shrinkage regardless of which synaptic class they represent.

Two terms
---------
1. **L1 rate penalty on E populations**
   Adds a subtractive contribution `−α` to the drive of every firing
   excitatory unit (and zero to silent units). At the call site in
   `network.py` the return value is *added* to the drive, so a negative
   output implements a metabolic "cost to firing".

2. **L2 total-synaptic-current penalty on all weights**
   Shrinks each weight `w_ij` in proportion to the squared presynaptic
   activity it is carrying:
       Δw_ij = −β · mean_b(a_pre_i²) · w_ij
   Derivation: the continuous energy term `E_L2 = ½ · Σ_b Σ_ij (a_pre_i · w_ij)²`
   has ∂E/∂w_ij = Σ_b a_pre_i² · w_ij (a_pre is independent of w_ij within
   one forward pass, so no chain-rule surprise). Averaging over the batch
   and scaling by β yields the expression above. This interpretation of
   "|I_pre|" is the per-synapse contribution to the summed squared
   current, consistent with the v4 spec's `β · ‖I_syn‖²` cost term.

Shared invariants
-----------------
* Deterministic — no stochastic draws inside the module.
* No autograd — both methods wrapped in `@torch.no_grad()`.
* No `nn.Parameter`s — α, β are plain Python floats stored as attributes.
* Mask-preserving on `current_weight_shrinkage`: masked-off entries of
  the returned ΔW are exactly zero.

Out of scope
------------
Integration with the forward pass (`network.py`), per-population α / β
values (chosen by `EnergyConfig` already scaffolded in `config.py`), and
the restriction "rate penalty applies only to E, not I" (enforced at the
call site, not inside this module).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["EnergyPenalty"]


# ---------------------------------------------------------------------------
# Shared helper: mask-zeroing (small, kept local to avoid cross-module coupling)
# ---------------------------------------------------------------------------

def _apply_mask(dw: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Zero out entries of `dw` at positions where `mask` is False."""
    if mask is None:
        return dw
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be torch.bool; got {mask.dtype}")
    if mask.shape != dw.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match dw shape "
            f"{tuple(dw.shape)}"
        )
    zero = torch.zeros((), dtype=dw.dtype, device=dw.device)
    return torch.where(mask, dw, zero)


# ---------------------------------------------------------------------------
# EnergyPenalty
# ---------------------------------------------------------------------------

class EnergyPenalty(nn.Module):
    """Circuit-wide L1 (rate) + L2 (current) metabolic penalty.

    Both terms are pathway-agnostic: no feedforward / recurrent / feedback
    distinction inside this module. Callers supply the weight tensor and
    pre-activity; the returned tensors are added into the network-level
    drive or raw-weight update at the call site.

    Attributes
    ----------
    alpha : float
        Coefficient for the L1 rate penalty on firing units.
    beta : float
        Coefficient for the L2 synaptic-current penalty on weights.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        if alpha < 0.0:
            raise ValueError(f"alpha must be ≥ 0; got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be ≥ 0; got {beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def rate_penalty_delta_drive(self, rate: Tensor) -> Tensor:
        """Subtractive drive contribution for firing excitatory units.

        Returns `−α` where the unit is firing (rate > 0), else 0. The sign
        convention is "add this to the drive"; a negative value therefore
        implements a metabolic cost.

        Args:
            rate: `[B, n_units]` (or any shape) tensor of non-negative
                firing rates.

        Returns:
            Tensor of identical shape and dtype, containing
            `−α · I(rate > 0)` where `I(·)` is the firing indicator.
        """
        if rate.dtype.is_complex or not rate.dtype.is_floating_point:
            raise ValueError(
                f"rate must be a floating-point tensor; got dtype={rate.dtype}"
            )
        firing = (rate > 0).to(dtype=rate.dtype)
        return -self.alpha * firing

    @torch.no_grad()
    def current_weight_shrinkage(
        self,
        weights: Tensor,
        pre_activity: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-weight shrinkage proportional to squared presynaptic activity.

        Closed form: ``Δw_ij = −β · mean_b(a_pre_i²) · w_ij``.

        Args:
            weights:      `[n_post, n_pre]` current weight tensor. Any real
                values accepted (raw pre-softplus, effective, excitatory,
                inhibitory — the formula is scale- and sign-agnostic).
            pre_activity: `[B, n_pre]` presynaptic activity over the batch.
            mask:         Optional `[n_post, n_pre]` boolean mask. Entries
                where `mask` is False become exactly 0 in the output.

        Returns:
            `[n_post, n_pre]` tensor of weight-update contributions.

        Raises:
            ValueError: on shape mismatches or a non-bool mask.
        """
        if weights.ndim != 2:
            raise ValueError(f"weights must be 2-D; got ndim={weights.ndim}")
        if pre_activity.ndim != 2:
            raise ValueError(
                f"pre_activity must be 2-D [B, n_pre]; got ndim={pre_activity.ndim}"
            )
        n_pre = weights.shape[1]
        if pre_activity.shape[1] != n_pre:
            raise ValueError(
                f"pre_activity last dim {pre_activity.shape[1]} must equal "
                f"weights n_pre = {n_pre}"
            )
        # mean over batch of a_pre² → shape [n_pre]; broadcast to [n_post, n_pre].
        # Implicit-Euler-equivalent shrinkage (Task #62): the explicit-Euler
        # form dw = -β·pre²·w overshoots for large pre² and drives oscillatory
        # weight explosion. Equivalent implicit form w_new = w / (1 + β·pre²)
        # gives dw = -w · shrink / (1 + shrink), which is always bounded
        # |dw| ≤ |w| for any non-negative shrink_factor.
        pre_sq_mean = (pre_activity * pre_activity).mean(dim=0)        # [n_pre]
        shrink_factor = self.beta * pre_sq_mean.view(1, -1)            # [1, n_pre]
        dw = -weights * shrink_factor / (1.0 + shrink_factor)          # [n_post, n_pre]
        return _apply_mask(dw, mask)
