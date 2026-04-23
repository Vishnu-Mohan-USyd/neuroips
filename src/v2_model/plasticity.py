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
* Weight decay uses a **raw-prior** anchor (Task #50). Passing
  ``raw_prior=None`` (default) subtracts ``weight_decay · weights`` — raw→0,
  the legacy behaviour. Passing ``raw_prior=<tensor>`` subtracts
  ``weight_decay · (weights − raw_prior)`` so raw drifts back toward the
  weight's init value rather than toward zero. Why this matters: under
  softplus Dale with strongly negative inits (e.g. raw=-5.85), pulling raw
  toward 0 *increases* effective softplus(raw) — anti-shrinkage. The
  raw-prior form restores the intended "keep weights near init magnitude"
  regularisation regardless of the init sign.
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
        raw_prior: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the ΔW tensor.

        Args:
            pre_activity: `[B, n_pre]` presynaptic rates.
            apical:       `[B, n_post]` postsynaptic apical current.
            basal:        `[B, n_post]` postsynaptic basal current.
            weights:      `[n_post, n_pre]` current raw weights.
            mask:         Optional `[n_post, n_pre]` boolean connectivity mask.
            raw_prior:    Optional `[n_post, n_pre]` anchor for weight decay.
                          `None` → decay pulls toward 0 (legacy behaviour).
                          Tensor → decay pulls toward `raw_prior` — the weight's
                          init value — avoiding the anti-shrinkage pathology
                          for strongly negative raw inits.

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
        shrink_target = weights if raw_prior is None else (weights - raw_prior)
        dw = self.lr * hebb - self.weight_decay * shrink_target
        dw.clamp_(min=-0.01, max=0.01)
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
        raw_prior: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the ΔW tensor.

        Args:
            pre_activity:  `[B, n_pre]` presynaptic (inhibitory-neuron) rates.
            post_activity: `[B, n_post]` postsynaptic rates.
            weights:       `[n_post, n_pre]` current raw weights.
            mask:          Optional `[n_post, n_pre]` boolean mask.
            raw_prior:     Optional `[n_post, n_pre]` anchor for weight decay;
                           see module docstring. `None` ⇒ decay toward 0.
        """
        _validate_pair_shapes(pre_activity, post_activity, weights)
        post_dev = post_activity - self.target_rate                  # [B, n_post]
        hebb = _batch_outer_mean(post_dev, pre_activity)             # [n_post, n_pre]
        shrink_target = weights if raw_prior is None else (weights - raw_prior)
        dw = self.lr * hebb - self.weight_decay * shrink_target
        dw.clamp_(min=-0.01, max=0.01)
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
        deadband_fraction: float = 0.2,
    ) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if n_units < 1:
            raise ValueError(f"n_units must be ≥ 1; got {n_units}")
        if target_rate < 0.0:
            raise ValueError(f"target_rate must be ≥ 0; got {target_rate}")
        if deadband_fraction < 0.0:
            raise ValueError(
                f"deadband_fraction must be ≥ 0; got {deadband_fraction}"
            )
        self.lr = float(lr)
        self.target_rate = float(target_rate)
        self.n_units = int(n_units)
        self.deadband_fraction = float(deadband_fraction)
        self.register_buffer(
            "theta",
            torch.full((n_units,), float(init_theta), dtype=torch.float32),
        )

    @torch.no_grad()
    def update(self, activity: Tensor) -> None:
        """In-place bounded threshold update with deadband (Task #54).

        Implements a saturating, deadband-gated homeostatic rule:

            error = mean_b(activity) − ρ_target
            if |error| < deadband_fraction · |ρ_target|  →  error ≔ 0
            Δθ = lr · tanh(error / scale) · scale
                 (scale = 0.1·|ρ_target| + 1e-3)

        The tanh saturates at ±lr·scale, preventing runaway θ during
        high-activity transients. The deadband prevents monotonic drift
        when activity is already near the target (complement to the
        Task #52 operating-point fix — homeostasis is now a gentle
        maintainer, not an operating-point creator).

        Args:
            activity: `[B, n_units]` per-batch unit rates.

        Side effects:
            `self.theta` is updated in place, then clamped to [-10, 10].
        """
        if activity.ndim != 2:
            raise ValueError(f"activity must be 2-D [B, n_units]; got ndim={activity.ndim}")
        if activity.shape[1] != self.n_units:
            raise ValueError(
                f"activity has n_units={activity.shape[1]}, module was "
                f"constructed with n_units={self.n_units}"
            )
        mean_a = activity.mean(dim=0)                                # [n_units]
        error = mean_a - self.target_rate
        # Deadband: no update inside ±deadband_fraction · |target_rate|.
        deadband = self.deadband_fraction * abs(self.target_rate)
        in_band = torch.abs(error) < deadband
        error = torch.where(in_band, torch.zeros_like(error), error)
        # Bounded (saturating) response — tanh prevents runaway drift.
        scale = 0.1 * abs(self.target_rate) + 1e-3
        update = self.lr * torch.tanh(error / scale) * scale
        self.theta.add_(update)
        # Safety clamp — Task #43 widened ±1 → ±10 so θ has room to track
        # large-gain transients during long-trajectory Phase-3 assays.
        # With the Task #54 tanh saturation, this clamp is rarely reached;
        # retained as a hard safety net.
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
        raw_prior: Optional[Tensor] = None,
    ) -> Tensor:
        """Three-factor update for W_qm^task (cue → memory).

        Args:
            cue:          `[B, n_cue]` presynaptic cue activity.
            memory:       `[B, n_m]`   postsynaptic memory activity.
            memory_error: `[B, n_m]`   postsynaptic modulatory / error signal.
            weights:      `[n_m, n_cue]` current raw weights.
            mask:         Optional `[n_m, n_cue]` boolean mask.
            raw_prior:    Optional `[n_m, n_cue]` anchor for weight decay;
                          see module docstring. `None` ⇒ decay toward 0
                          (matches task-weight zero-init default).
        """
        _validate_pair_shapes(cue, memory, weights, pre_name="cue", post_name="m")
        if memory_error.shape != memory.shape:
            raise ValueError(
                f"memory_error shape {tuple(memory_error.shape)} must equal "
                f"memory shape {tuple(memory.shape)}"
            )
        gated = memory * memory_error                                # [B, n_m]
        hebb = _batch_outer_mean(gated, cue)                         # [n_m, n_cue]
        shrink_target = weights if raw_prior is None else (weights - raw_prior)
        dw = self.lr * hebb - self.weight_decay * shrink_target
        dw.clamp_(min=-0.01, max=0.01)
        return _apply_mask(dw, mask)

    @torch.no_grad()
    def delta_mh(
        self,
        memory: Tensor,
        probe_error: Tensor,
        weights: Tensor,
        mask: Optional[Tensor] = None,
        raw_prior: Optional[Tensor] = None,
    ) -> Tensor:
        """Error-driven update for W_mh^task (memory → probe prediction).

        Args:
            memory:      `[B, n_m]` presynaptic memory activity.
            probe_error: `[B, n_h]` postsynaptic probe error.
            weights:     `[n_h, n_m]` current raw weights.
            mask:        Optional `[n_h, n_m]` boolean mask.
            raw_prior:   Optional `[n_h, n_m]` anchor for weight decay;
                         see module docstring. `None` ⇒ decay toward 0.
        """
        _validate_pair_shapes(memory, probe_error, weights, pre_name="m", post_name="h")
        hebb = _batch_outer_mean(probe_error, memory)                # [n_h, n_m]
        shrink_target = weights if raw_prior is None else (weights - raw_prior)
        dw = self.lr * hebb - self.weight_decay * shrink_target
        dw.clamp_(min=-0.01, max=0.01)
        return _apply_mask(dw, mask)

    @torch.no_grad()
    def delta_mh_inh(
        self,
        memory: Tensor,
        som_modulator: Tensor,
        weights: Tensor,
        mask: Optional[Tensor] = None,
        raw_prior: Optional[Tensor] = None,
    ) -> Tensor:
        """Three-factor update for W_mh^task_inh (memory → SOM gain, Fix C-v2).

        Task #74 Fix C-v2: the task-specific readout was originally a
        single excitatory-driven path (``W_mh_task @ m`` → L23E apical).
        Empirical fault: direction metric cos(b_task, sensory_loc)=0.136
        ≪ 0.30. The biologically grounded alternative is to route the
        MAIN task bias through SOM, which provides apical gain control
        of L23 E dendrites (Urbanczik & Senn 2014; Larkum 1999).

        An earlier Fix-C variant added ``W_mh_task_inh @ m`` to the SOM
        drive directly; this silenced L23E (r_l23 → 1e-20) because the
        Phase-2 SOM baseline is already saturated. Fix C-v2 replaces that
        additive route with *per-SOM-unit gain modulation* on the
        SOM→L23E synapses:
            som_gain = softplus(W_mh_task_inh · m + 0.5413).clamp(max=4.0)
        which scales SOM→L23E efficacy multiplicatively (biologically:
        cholinergic / noradrenergic modulation of GABAergic transmission,
        Disney & Aoki 2008; Pfeffer 2013). The three-factor rule below is
        UNCHANGED — only the application site of the resulting signal is
        different. The modulator/memory correlation structure, learning
        rate, weight decay, clamp, and mask semantics are preserved
        verbatim.

        Closed-form:
            dw[j, i] = η · mean_b(som_modulator[b, j] · memory[b, i])
                     − λ · (weights[j, i] − raw_prior[j, i])

        with ``som_modulator[b, j] = r_som_expected_j − r_som_unexpected_j``
        for unit ``j`` under the current batch's (cue × probe) condition.
        Sign convention: positive modulator ⇒ SOM fires MORE for expected
        ⇒ strengthen W to recapitulate expected-time inhibition via memory.

        Args:
            memory:        `[B, n_m]` presynaptic memory activity
                           (eligibility ≡ pre-probe memory snapshot).
            som_modulator: `[B, n_som]` post-synaptic signed modulator —
                           contrast r_som_expected − r_som_unexpected
                           matched to this trial's probe stimulus.
            weights:       `[n_som, n_m]` current raw weights.
            mask:          Optional `[n_som, n_m]` boolean mask.
            raw_prior:     Optional `[n_som, n_m]` anchor for weight decay;
                           see module docstring. `None` ⇒ decay toward 0.

        Returns:
            `[n_som, n_m]` ΔW tensor, already clamped to ±0.01 and masked.
        """
        _validate_pair_shapes(
            memory, som_modulator, weights, pre_name="m", post_name="som",
        )
        hebb = _batch_outer_mean(som_modulator, memory)              # [n_som, n_m]
        shrink_target = weights if raw_prior is None else (weights - raw_prior)
        dw = self.lr * hebb - self.weight_decay * shrink_target
        dw.clamp_(min=-0.01, max=0.01)
        return _apply_mask(dw, mask)


