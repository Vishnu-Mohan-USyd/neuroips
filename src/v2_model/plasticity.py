"""Local plasticity rules — closed-form updates with no autograd.

Four rules from the v4 plan §Learning rules. All updates are closed-form
outer-product expressions in `torch.no_grad` context; no `.backward()`.

  1. `UrbanczikSennRule` — apical-basal predictive-Hebbian.
     Δw_ij = η · a_pre_i · ε_post_j − λ · w_ij,  ε = apical − basal.
     Urbanczik & Senn (2014) *Neuron* 81:521;
     Halvagal & Zenke (2023) *Nat Neurosci* 26:1906.
  2. `VogelsISTDPRule` — homeostatic inhibitory plasticity.
     Δw_inh = η · a_pre · (a_post − ρ) − λ · w_inh.
     Vogels, Sprekeler, Zenke, Clopath, Gerstner (2011) *Science* 334:1569.
  3. `ThresholdHomeostasis` — per-unit threshold drift on a state buffer.
     Δθ_j = η · (a_j − ρ). Turrigiano et al. (1998) *Nature* 391:892.
  4. `ThreeFactorRule` — Phase-3 Kok/Richter cue-memory-probe binding.
     `delta_qm`: Δw ∝ cue · memory · memory_error − λ w (3-factor).
     `delta_mh`: Δw ∝ memory · probe_error − λ w        (error-driven).
     Frémaux & Gerstner (2016) *Front Neural Circuits* 9:85.

Shared invariants
-----------------
* All ΔW tensors are intended to be added to *raw pre-softplus* weights
  (see `src/utils.py::ExcitatoryLinear`). Because softplus is strictly
  monotonic, raw-weight updates transmit sign faithfully: positive Δraw ⇒
  stronger excitation (softplus(raw)) or stronger inhibition (−softplus(raw)).
* Every rule accepts an optional boolean `mask` with the same shape as
  `weights`. Entries where `mask` is False are set to exactly zero in ΔW
  (sparse-connectivity preservation).
* `weight_decay · weights` is subtracted from the *raw* update. Under
  softplus Dale parameterisation, raw → 0 corresponds to softplus(0) ≈ 0.693,
  not to zero effective strength. Standard "regularise the parameter" treatment.
* Deterministic: identical inputs always produce identical ΔW. No stochastic
  draws inside a rule.

Out of scope: wiring rules into populations (`layers.py` / `network.py`),
integrating with `connectivity.py` masks at build time, the training driver.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "UrbanczikSennRule",
    "VogelsISTDPRule",
    "ThresholdHomeostasis",
    "ThreeFactorRule",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _batch_outer_mean(post: Tensor, pre: Tensor) -> Tensor:
    """Batch-averaged outer product yielding a `[n_post, n_pre]` Hebbian term.

    Args:
        post: `[B, n_post]` (per-sample postsynaptic signal).
        pre:  `[B, n_pre]`  (per-sample presynaptic signal).

    Returns:
        `[n_post, n_pre]` tensor whose `(j, i)` entry equals
        `mean_b(post[b, j] * pre[b, i])`.
    """
    # post.t() @ pre has shape [n_post, n_pre] and sums over the batch axis.
    return post.t() @ pre / float(post.shape[0])


def _apply_mask(dw: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Zero out entries of ΔW at positions where `mask` is False.

    Args:
        dw: Candidate weight-update tensor.
        mask: Optional boolean tensor of the same shape as `dw`. If `None`,
            `dw` is returned unchanged.

    Returns:
        `dw` with masked-off entries replaced by exact zero.
    """
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


def _validate_pair_shapes(
    pre: Tensor, post: Tensor, weights: Tensor,
    pre_name: str = "pre", post_name: str = "post",
) -> None:
    """Check that `pre[B, n_pre]`, `post[B, n_post]`, `weights[n_post, n_pre]`."""
    if pre.ndim != 2:
        raise ValueError(f"{pre_name} must be 2-D [B, n_{pre_name}]; got ndim={pre.ndim}")
    if post.ndim != 2:
        raise ValueError(f"{post_name} must be 2-D [B, n_{post_name}]; got ndim={post.ndim}")
    if pre.shape[0] != post.shape[0]:
        raise ValueError(
            f"batch dim mismatch: {pre_name} B={pre.shape[0]} vs "
            f"{post_name} B={post.shape[0]}"
        )
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D; got ndim={weights.ndim}")
    expected = (post.shape[1], pre.shape[1])
    if weights.shape != expected:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} must equal "
            f"(n_{post_name}, n_{pre_name}) = {expected}"
        )


# ---------------------------------------------------------------------------
# 1. Urbanczik–Senn apical-basal predictive-Hebbian
# ---------------------------------------------------------------------------

class UrbanczikSennRule(nn.Module):
    """Apical-basal predictive-Hebbian rule (Urbanczik & Senn 2014).

    `Δw_ij = η · a_pre_i · ε_post_j − λ · w_ij`, with
    `ε_j = apical_current_j − basal_current_j`: the mismatch between top-down
    apical prediction and bottom-up basal drive. Positive ε pushes the weight
    upward, aligning basal drive with the apical prediction.
    """

    def __init__(self, lr: float, weight_decay: float = 0.0) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be ≥ 0; got {weight_decay}")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @torch.no_grad()
    def delta(
        self,
        pre_activity: Tensor,
        apical: Tensor,
        basal: Tensor,
        weights: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the ΔW tensor.

        Args:
            pre_activity: `[B, n_pre]` presynaptic rates.
            apical:       `[B, n_post]` postsynaptic apical current.
            basal:        `[B, n_post]` postsynaptic basal current.
            weights:      `[n_post, n_pre]` current raw weights.
            mask:         Optional `[n_post, n_pre]` boolean connectivity mask.

        Returns:
            `[n_post, n_pre]` update tensor. Entries where `mask` is False
            are exactly zero.
        """
        _validate_pair_shapes(pre_activity, apical, weights, pre_name="pre", post_name="post")
        if basal.shape != apical.shape:
            raise ValueError(
                f"basal shape {tuple(basal.shape)} must equal apical "
                f"shape {tuple(apical.shape)}"
            )
        epsilon = apical - basal                                      # [B, n_post]
        hebb = _batch_outer_mean(epsilon, pre_activity)               # [n_post, n_pre]
        dw = self.lr * hebb - self.weight_decay * weights
        return _apply_mask(dw, mask)


# ---------------------------------------------------------------------------
# 2. Vogels iSTDP (homeostatic inhibitory plasticity)
# ---------------------------------------------------------------------------

class VogelsISTDPRule(nn.Module):
    """Homeostatic inhibitory plasticity (Vogels et al. 2011).

    `Δw_inh = η · a_pre · (a_post − ρ_target) − λ · w_inh`. Post-cell firing
    above ρ_target grows the raw inhibitory weight; under softplus Dale
    parameterisation this makes `−softplus(raw)` more negative (stronger
    inhibition). Under-active post ⇒ weaker inhibition.
    """

    def __init__(
        self,
        lr: float,
        target_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be ≥ 0; got {weight_decay}")
        if target_rate < 0.0:
            raise ValueError(f"target_rate must be ≥ 0; got {target_rate}")
        self.lr = float(lr)
        self.target_rate = float(target_rate)
        self.weight_decay = float(weight_decay)

    @torch.no_grad()
    def delta(
        self,
        pre_activity: Tensor,
        post_activity: Tensor,
        weights: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the ΔW tensor.

        Args:
            pre_activity:  `[B, n_pre]` presynaptic (inhibitory-neuron) rates.
            post_activity: `[B, n_post]` postsynaptic rates.
            weights:       `[n_post, n_pre]` current raw weights.
            mask:          Optional `[n_post, n_pre]` boolean mask.
        """
        _validate_pair_shapes(pre_activity, post_activity, weights)
        post_dev = post_activity - self.target_rate                  # [B, n_post]
        hebb = _batch_outer_mean(post_dev, pre_activity)             # [n_post, n_pre]
        dw = self.lr * hebb - self.weight_decay * weights
        return _apply_mask(dw, mask)


# ---------------------------------------------------------------------------
# 3. Threshold homeostasis (per-unit drift)
# ---------------------------------------------------------------------------

class ThresholdHomeostasis(nn.Module):
    """Per-unit threshold-drift homeostasis.

    Maintains `theta: [n_units]` as an `nn.Module` buffer (moves with
    `.to(device)`, is serialised by `state_dict`). In-place update:
    `Δθ_j = η · (mean_b(a_j) − ρ_target)`. Over-active units accrete higher
    thresholds; under-active lose theirs. Expected wiring at the call site:
    `drive = input − theta`, so the network settles at a_j ≈ ρ_target.
    """

    def __init__(
        self,
        lr: float,
        target_rate: float,
        n_units: int,
        init_theta: float = 0.0,
    ) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if n_units < 1:
            raise ValueError(f"n_units must be ≥ 1; got {n_units}")
        if target_rate < 0.0:
            raise ValueError(f"target_rate must be ≥ 0; got {target_rate}")
        self.lr = float(lr)
        self.target_rate = float(target_rate)
        self.n_units = int(n_units)
        self.register_buffer(
            "theta",
            torch.full((n_units,), float(init_theta), dtype=torch.float32),
        )

    @torch.no_grad()
    def update(self, activity: Tensor) -> None:
        """In-place threshold update from a batch of activity.

        Args:
            activity: `[B, n_units]` per-batch unit rates.

        Side effects:
            `self.theta += lr · (mean_b(activity) − ρ_target)`.
        """
        if activity.ndim != 2:
            raise ValueError(f"activity must be 2-D [B, n_units]; got ndim={activity.ndim}")
        if activity.shape[1] != self.n_units:
            raise ValueError(
                f"activity has n_units={activity.shape[1]}, module was "
                f"constructed with n_units={self.n_units}"
            )
        mean_a = activity.mean(dim=0)                                # [n_units]
        self.theta.add_(self.lr * (mean_a - self.target_rate))
        # Safety clamp — Task #43 widened ±1 → ±10 so θ has room to track
        # large-gain transients during long-trajectory Phase-3 assays.
        # Normal operation (well-balanced E/I) keeps |θ| well under 1;
        # the widened bound only kicks in during high-activity bursts.
        self.theta.clamp_(min=-10.0, max=10.0)


# ---------------------------------------------------------------------------
# 4. Three-factor rule (Phase-3 Kok/Richter cue-memory-probe binding)
# ---------------------------------------------------------------------------

class ThreeFactorRule(nn.Module):
    """Phase-3 Kok/Richter cue-memory-probe binding. Two variants.

    * `delta_qm` — three-factor cue→memory binding:
          `Δw ∝ cue · memory · memory_error − λ w`.
      `memory_error` (≡ `a_post_memory − ρ_target` for Kok cue-binding)
      acts as a neuromodulatory gate: learning only when presynaptic cue,
      postsynaptic memory activity, and memory-error all coincide.

    * `delta_mh` — two-factor error-driven memory→probe:
          `Δw ∝ memory · probe_error − λ w`,
      with `probe_error = h*_probe − ĥ_preprobe` carrying the task error.
      Grouped here because both rules are used in the Phase-3 training recipe.
    """

    def __init__(self, lr: float, weight_decay: float = 0.0) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be ≥ 0; got {weight_decay}")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @torch.no_grad()
    def delta_qm(
        self,
        cue: Tensor,
        memory: Tensor,
        memory_error: Tensor,
        weights: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Three-factor update for W_qm^task (cue → memory).

        Args:
            cue:          `[B, n_cue]` presynaptic cue activity.
            memory:       `[B, n_m]`   postsynaptic memory activity.
            memory_error: `[B, n_m]`   postsynaptic modulatory / error signal.
            weights:      `[n_m, n_cue]` current raw weights.
            mask:         Optional `[n_m, n_cue]` boolean mask.
        """
        _validate_pair_shapes(cue, memory, weights, pre_name="cue", post_name="m")
        if memory_error.shape != memory.shape:
            raise ValueError(
                f"memory_error shape {tuple(memory_error.shape)} must equal "
                f"memory shape {tuple(memory.shape)}"
            )
        gated = memory * memory_error                                # [B, n_m]
        hebb = _batch_outer_mean(gated, cue)                         # [n_m, n_cue]
        dw = self.lr * hebb - self.weight_decay * weights
        return _apply_mask(dw, mask)

    @torch.no_grad()
    def delta_mh(
        self,
        memory: Tensor,
        probe_error: Tensor,
        weights: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Error-driven update for W_mh^task (memory → probe prediction).

        Args:
            memory:      `[B, n_m]` presynaptic memory activity.
            probe_error: `[B, n_h]` postsynaptic probe error.
            weights:     `[n_h, n_m]` current raw weights.
            mask:        Optional `[n_h, n_m]` boolean mask.
        """
        _validate_pair_shapes(memory, probe_error, weights, pre_name="m", post_name="h")
        hebb = _batch_outer_mean(probe_error, memory)                # [n_h, n_m]
        dw = self.lr * hebb - self.weight_decay * weights
        return _apply_mask(dw, mask)
