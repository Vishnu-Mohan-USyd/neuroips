"""Prediction head — predicts next-step L4 E rate (plan v4 step 10 / Task #23).

Readout used by the Phase-2 apical-basal prediction error signal. Produces
``x̂_{t+1}`` ∈ ℝ^{n_l4_e=128}, compared against the *actual* next-step L4 E
rate by :func:`compute_error` to give the per-unit residual ε consumed by
Urbanczik–Senn plasticity on the L2/3 apical compartment.

Prediction target rationale
----------------------------
Target is the **L4 E rate** (128 dims), NOT the raw LGN feature map
(10·32·32 = 10240 dims). L4 E is the first post-LGN stage where plastic
circuitry operates, and 10k → 64 pure-local inversion is intractable.
The apical-basal error compares (predicted next-L4 | via apical pathway)
vs (actual next-L4 | via basal pathway) at the L2/3 E soma.

Prediction sources
------------------
We include all three streams (see the Task-23 completion DM):

  1. ``h_rate``              [B, n_h_e=64]        — H state (required).
  2. ``c_bias``              [B, n_c_bias=48]     — context-memory bias
                                                    (optional; latent-regime
                                                    cue).
  3. ``l23_apical_summary``  [B, n_l23_apical=256]— L2/3 E apical summary
                                                    (optional; supplies the
                                                    apical-compartment
                                                    prediction source per
                                                    Urbanczik–Senn).

``None`` on an optional input skips the corresponding matmul (numerically
identical to passing the zero tensor — see
``test_prediction_head_optional_inputs.py``).

Dynamics
--------
    drive = softplus(W_pred_H_raw)        @ h_rate
          + softplus(W_pred_C_raw)        @ c_bias              (if provided)
          + softplus(W_pred_apical_raw)   @ l23_apical_summary  (if provided)
          + softplus(b_pred_raw)
    x_hat_next = rectified_softplus(drive)

All ``softplus`` applications enforce Dale's-law-excitatory weights — the
prediction is a monotonic non-negative combination of the input streams.

Phase gating
-------------
All weights are plastic in Phase 2 (head is learnt with the rest of the
generic circuit) and frozen in both Phase-3 variants (task-specific
modulation lives in the task-weight split inside ``context_memory.py``).

Autograd
--------
All raw weights are ``nn.Parameter(..., requires_grad=False)``. No
``@torch.no_grad()`` on forward — gradients flow through inputs for the
BPTT-fallback path (design note #15); they never accumulate into Parameters.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.utils import rectified_softplus


__all__ = ["PredictionHead", "compute_error", "PhaseLiteral"]


PhaseLiteral = Literal["phase2", "phase3_kok", "phase3_richter"]
_VALID_PHASES: tuple[str, ...] = ("phase2", "phase3_kok", "phase3_richter")


def _validate_size(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value}")


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def compute_error(x_actual: Tensor, x_predicted: Tensor) -> Tensor:
    """Element-wise prediction residual ``ε = x_actual − x_predicted``.

    Parameters
    ----------
    x_actual, x_predicted : Tensor [B, n_l4_e]
        Must have identical shape.

    Returns
    -------
    Tensor [B, n_l4_e]
        ε is positive when the prediction under-shoots the actual rate
        (basal > apical at the L2/3 E soma) and negative when over-shooting.
    """
    if x_actual.shape != x_predicted.shape:
        raise ValueError(
            f"compute_error shape mismatch: actual {tuple(x_actual.shape)} "
            f"vs predicted {tuple(x_predicted.shape)}"
        )
    return x_actual - x_predicted


# ---------------------------------------------------------------------------
# PredictionHead module
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """Linear readout → non-negative predicted L4 E rate.

    Parameters (all ``nn.Parameter(..., requires_grad=False)``):
      * ``W_pred_H_raw``       [n_l4_e, n_h_e]         — required.
      * ``W_pred_C_raw``       [n_l4_e, n_c_bias]       — iff ``n_c_bias!=None``.
      * ``W_pred_apical_raw``  [n_l4_e, n_l23_apical]   — iff ``n_l23_apical!=None``.
      * ``b_pred_raw``         [n_l4_e]                — small non-negative
                                                         bias (init at -8.0 so
                                                         ``softplus(b_pred_raw)
                                                         ≈ 3.35e-4`` per unit;
                                                         Task #52 calibration,
                                                         Task #74 Fix O bump
                                                         to match [-8, 8] raw
                                                         clamp).

    Parameters
    ----------
    n_l4_e : int
        Output dim — L4 E rate vector length (default 128).
    n_h_e : int
        Required input dim — H excitatory state length (default 64).
    n_c_bias : int | None
        Context-memory bias input dim. ``None`` disables the C stream
        entirely (weight not created; forward rejects non-None c_bias).
    n_l23_apical : int | None
        L2/3 E apical summary input dim. ``None`` disables the apical stream
        entirely.
    init_scale : float
        Std of the normal init for raw weights. Default 0.1.
    seed : int
        RNG seed used for deterministic weight init. Default 0.
    """

    def __init__(
        self,
        n_l4_e: int = 128,
        n_h_e: int = 64,
        n_c_bias: Optional[int] = 48,
        n_l23_apical: Optional[int] = 256,
        *,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _validate_size("n_l4_e", n_l4_e)
        _validate_size("n_h_e", n_h_e)
        if n_c_bias is not None:
            _validate_size("n_c_bias", n_c_bias)
        if n_l23_apical is not None:
            _validate_size("n_l23_apical", n_l23_apical)
        if init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0; got {init_scale}")

        self.n_l4_e = int(n_l4_e)
        self.n_h_e = int(n_h_e)
        self.n_c_bias: Optional[int] = (
            int(n_c_bias) if n_c_bias is not None else None
        )
        self.n_l23_apical: Optional[int] = (
            int(n_l23_apical) if n_l23_apical is not None else None
        )
        self._init_scale = float(init_scale)
        self._seed = int(seed)
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._dtype = dtype

        gen = torch.Generator(device=self._device)
        gen.manual_seed(self._seed)

        # Task #50 — per-weight init-mean registry for raw-prior weight decay.
        # Mirrors `_BasePopulation.raw_init_means`; the Phase-2 driver reads
        # this to anchor weight-decay at init magnitude rather than 0.
        self.raw_init_means: dict[str, float] = {}

        def _make_raw(
            shape: tuple[int, ...],
            *,
            init_mean: float = 0.0,
            name: Optional[str] = None,
        ) -> nn.Parameter:
            t = torch.empty(*shape, device=self._device, dtype=self._dtype)
            with torch.no_grad():
                t.normal_(
                    mean=float(init_mean), std=self._init_scale, generator=gen,
                )
            if name is not None:
                self.raw_init_means[name] = float(init_mean)
            return nn.Parameter(t, requires_grad=False)

        # Task #52 — T29 calibration: pred-head weights init at -8.0.
        # Task #74 Fix O (2026-04-22): bumped from -10.0 to -8.0 to match
        # the [-8, 8] raw clamp in ``apply_plasticity_step`` (Task #64).
        # The previous -10 init was outside the clamp band and got snapped
        # to -8 on the first plasticity step — a 7.4× softplus gain jump
        # (softplus(-10)=4.54e-5 → softplus(-8)=3.35e-4) that propagated
        # through the prediction pathway and destabilised L23E.
        # At -8, drive per unit ≈ softplus(-8) × (1 + sum_k mean-inputs_k),
        # still small enough to keep x̂ within a decade of r_l4 at blank,
        # and now stable under the plasticity clamp contract.
        self.W_pred_H_raw = _make_raw(
            (self.n_l4_e, self.n_h_e), init_mean=-8.0, name="W_pred_H_raw",
        )

        b_init = torch.full(
            (self.n_l4_e,), -8.0, device=self._device, dtype=self._dtype,
        )
        self.b_pred_raw = nn.Parameter(b_init, requires_grad=False)
        self.raw_init_means["b_pred_raw"] = -8.0

        if self.n_c_bias is not None:
            self.W_pred_C_raw = _make_raw(
                (self.n_l4_e, self.n_c_bias),
                init_mean=-8.0, name="W_pred_C_raw",
            )
        if self.n_l23_apical is not None:
            self.W_pred_apical_raw = _make_raw(
                (self.n_l4_e, self.n_l23_apical),
                init_mean=-8.0, name="W_pred_apical_raw",
            )

        # Plastic-in-phase-2 manifest (stable iteration order: H, C, apical, b).
        names: list[str] = ["W_pred_H_raw"]
        if self.n_c_bias is not None:
            names.append("W_pred_C_raw")
        if self.n_l23_apical is not None:
            names.append("W_pred_apical_raw")
        names.append("b_pred_raw")
        self._all_plastic_names: tuple[str, ...] = tuple(names)

        self._phase: str = "phase2"

    # ---- phase gating API -----------------------------------------------

    def set_phase(self, phase: PhaseLiteral) -> None:
        """Declare current training phase. Purely informational (no weight
        mutation — training drivers read the manifest before stepping rules)."""
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"phase must be one of {list(_VALID_PHASES)}; got {phase!r}"
            )
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase

    def plastic_weight_names(self) -> list[str]:
        """Names of raw-weight Parameters plastic in the current phase.

        Phase 2  → all weights.
        Phase 3  → empty (task-specific modulation lives in context_memory).
        """
        return list(self._all_plastic_names) if self._phase == "phase2" else []

    def frozen_weight_names(self) -> list[str]:
        """Names of raw-weight Parameters frozen in the current phase."""
        return [] if self._phase == "phase2" else list(self._all_plastic_names)

    # ---- forward ---------------------------------------------------------

    def forward(
        self,
        h_rate: Tensor,
        c_bias: Optional[Tensor] = None,
        l23_apical_summary: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict next-step L4 E rate ``x̂_{t+1}``.

        Parameters
        ----------
        h_rate : Tensor [B, n_h_e]
            Primary input: H excitatory rate.
        c_bias : Tensor [B, n_c_bias] | None
            Context-memory bias. ``None`` ≡ zero contribution. Raises if the
            head was constructed without ``n_c_bias``.
        l23_apical_summary : Tensor [B, n_l23_apical] | None
            L2/3 apical summary. ``None`` ≡ zero contribution. Raises if the
            head was constructed without ``n_l23_apical``.

        Returns
        -------
        x_hat_next : Tensor [B, n_l4_e]
            Non-negative predicted next-step L4 E rate.
        """
        if h_rate.ndim != 2 or h_rate.shape[1] != self.n_h_e:
            raise ValueError(
                f"h_rate must be [B, {self.n_h_e}]; got {tuple(h_rate.shape)}"
            )
        B = h_rate.shape[0]

        drive = F.linear(h_rate, F.softplus(self.W_pred_H_raw))

        if c_bias is not None:
            if self.n_c_bias is None:
                raise ValueError(
                    "c_bias passed but PredictionHead was constructed with "
                    "n_c_bias=None (W_pred_C_raw does not exist)"
                )
            if c_bias.ndim != 2 or tuple(c_bias.shape) != (B, self.n_c_bias):
                raise ValueError(
                    f"c_bias must be [{B}, {self.n_c_bias}]; "
                    f"got {tuple(c_bias.shape)}"
                )
            drive = drive + F.linear(c_bias, F.softplus(self.W_pred_C_raw))

        if l23_apical_summary is not None:
            if self.n_l23_apical is None:
                raise ValueError(
                    "l23_apical_summary passed but PredictionHead was "
                    "constructed with n_l23_apical=None "
                    "(W_pred_apical_raw does not exist)"
                )
            if (
                l23_apical_summary.ndim != 2
                or tuple(l23_apical_summary.shape) != (B, self.n_l23_apical)
            ):
                raise ValueError(
                    f"l23_apical_summary must be [{B}, {self.n_l23_apical}]; "
                    f"got {tuple(l23_apical_summary.shape)}"
                )
            drive = drive + F.linear(
                l23_apical_summary, F.softplus(self.W_pred_apical_raw),
            )

        drive = drive + F.softplus(self.b_pred_raw)
        return rectified_softplus(drive)
