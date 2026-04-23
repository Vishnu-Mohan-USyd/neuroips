"""Phase-2 pure-local predictive training driver (plan v4 step 14 / Task #31).

Runs closed-form local plasticity on :class:`V2Network` over random 2-frame
trajectories from :class:`ProceduralWorld`, minimising

    L = λ_pred · ‖r_l4_{t+1} − x̂_{t+1}‖²     (predictive term)
      + λ_rate · ‖r_E‖₁                        (metabolic rate penalty)
      + λ_syn  · ‖I_syn‖₂²                     (metabolic current penalty)

without ever calling ``backward``. Plasticity is wired per-weight:

* **Urbanczik–Senn** (apical-basal) on the L2/3 pyramidal weights
  ``(W_l4_l23_raw, W_rec_raw, W_fb_apical_raw)`` using
  ``ε = apical_drive − basal_drive`` at each L2/3 E neuron.
* **Urbanczik–Senn** on the four prediction-head weights using
  ``ε = r_l4_next − x̂_next`` (the predictive coding residual).
* **Vogels iSTDP** on every inhibitory-to-E synapse with target ρ
  ``= cfg.plasticity.target_rate_hz``.
* **Hebbian** (post-activity as the driving signal, target=0 so the rule
  degenerates to outer-product) on the remaining generic E-to-E and
  E-to-I / memory weights. These pathways have no Urbanczik–Senn-style
  apical-basal split in the current wiring; pure rate-correlation keeps
  them biologically local without introducing a global error signal.
* **ThresholdHomeostasis** updated in-place on :class:`L23E` and
  :class:`HE` per batch to pin mean firing rate at ρ.
* **Energy shrinkage** (``β · mean_b(a_pre²) · W``) applied to every
  plastic weight as an additive update — this realises the
  ``‖I_syn‖²`` gradient term in closed form.

Design choices:
  * 2-frame windows (``T = 2``). Step 0 produces ``x̂`` from the initial
    state; step 1 reveals the actual ``r_l4`` that the prediction targeted.
    Longer windows would require accumulating a prediction history; a
    short window keeps the update rule a clean one-step transform.
  * Frozen-sensory-core invariance is asserted via
    :meth:`V2Network.frozen_sensory_core_sha` at every checkpoint.
  * Determinism: ``torch.manual_seed(cfg.seed)`` at driver start; each
    trajectory uses ``(step * batch_size + b)`` as its trajectory seed,
    so ``run_phase2_training`` with the same CLI args is bit-identical.

Runs:
  python -m scripts.v2.train_phase2_predictive --seed 42 --n-steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.energy import EnergyPenalty
from src.v2_model.network import V2Network
from src.v2_model.plasticity import UrbanczikSennRule, VogelsISTDPRule
from src.v2_model.prediction_head import compute_error
from src.v2_model.state import NetworkStateV2
from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world.procedural import ProceduralWorld, WorldState

__all__ = [
    "PhaseFrozenError",
    "PlasticityRuleBank",
    "build_world",
    "run_phase2_training",
    "sample_batch_window",
    "step_persistent_batch",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PhaseFrozenError(RuntimeError):
    """Raised when a training step tries to update a weight that is frozen
    under the current phase manifest. Guards Phase-3 drivers from accidentally
    touching generic weights (and vice-versa)."""


# ---------------------------------------------------------------------------
# Rule bank
# ---------------------------------------------------------------------------

@dataclass
class PlasticityRuleBank:
    """Container for the closed-form rules consumed by the driver.

    One instance per driver invocation. Learning rates come from
    ``cfg.plasticity``; the caller may override any LR via CLI.

    Task #74 Fix H (2026-04-21) — Vogels targets are split by the E-pop
    whose rate the rule is pushing toward ρ:

    * ``vogels_l23``  (target = ``cfg.plasticity.vogels_target_l23e_hz``)
      used for L23E-targeting inhibitory synapses — ``W_pv_l23_raw``,
      ``W_som_l23_raw``. Pushes L23E toward ~3 Hz.
    * ``vogels_h``    (target = ``cfg.plasticity.vogels_target_h_hz``)
      used for HE-targeting inhibitory synapse — ``W_pv_h_raw``.
      Pushes HE toward ~0.1 Hz (matches HE θ-homeostasis).
    * ``vogels_ipop`` (target = ``cfg.plasticity.target_rate_hz``, = 1.0)
      used for I-pop self-regulation — ``l23_pv.W_pre_raw`` and
      ``h_pv.W_pre_raw``. Not tied to any E-pop target; retained at 1.0
      for backwards compatibility with Task #52 tuning of PV/HPV
      self-regulation.
    """
    urbanczik: UrbanczikSennRule
    vogels_l23: VogelsISTDPRule
    vogels_h: VogelsISTDPRule
    vogels_ipop: VogelsISTDPRule
    hebb: VogelsISTDPRule           # Vogels with target_rate=0 ≡ Hebbian
    energy: EnergyPenalty

    @classmethod
    def from_config(
        cls,
        cfg: ModelConfig,
        lr_urbanczik: float,
        lr_vogels: float,
        lr_hebb: float,
        weight_decay: float,
        beta_syn: float,
    ) -> "PlasticityRuleBank":
        return cls(
            urbanczik=UrbanczikSennRule(
                lr=lr_urbanczik, weight_decay=weight_decay,
            ),
            vogels_l23=VogelsISTDPRule(
                lr=lr_vogels,
                target_rate=cfg.plasticity.vogels_target_l23e_hz,
                weight_decay=weight_decay,
            ),
            vogels_h=VogelsISTDPRule(
                lr=lr_vogels,
                target_rate=cfg.plasticity.vogels_target_h_hz,
                weight_decay=weight_decay,
            ),
            vogels_ipop=VogelsISTDPRule(
                lr=lr_vogels,
                target_rate=cfg.plasticity.target_rate_hz,
                weight_decay=weight_decay,
            ),
            hebb=VogelsISTDPRule(
                lr=lr_hebb, target_rate=0.0, weight_decay=weight_decay,
            ),
            energy=EnergyPenalty(alpha=0.0, beta=float(beta_syn)),
        )


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def build_world(
    cfg: ModelConfig,
    *,
    seed_family: str = "train",
    held_out_regime: Optional[str] = None,
    token_bank_seed: int = 0,
) -> tuple[ProceduralWorld, TokenBank]:
    """Instantiate a :class:`TokenBank` and the matching procedural world."""
    bank = TokenBank(cfg, seed=token_bank_seed)
    world = ProceduralWorld(
        cfg, bank, seed_family=seed_family, held_out_regime=held_out_regime,
    )
    return world, bank


def sample_batch_window(
    world: ProceduralWorld, seeds: list[int], n_steps_per_window: int = 2,
) -> Tensor:
    """Build a ``[B, T, 1, H, W]`` batch of frames from ``B = len(seeds)``
    independent trajectories. ``T = n_steps_per_window`` (default 2).

    Each trajectory draws from its own ``torch.Generator`` seeded from
    ``seed_family + seeds[b]``, so runs are reproducible given the seed list.

    NOTE (critique C1 / Task #68): this helper re-creates a fresh N-frame
    window per call, so using it inside a training loop yields *independent*
    per-step windows rather than *persistent* trajectories. Phase-2
    training uses :func:`step_persistent_batch` instead; this function is
    retained for gates / tests that genuinely want independent windows.
    """
    if n_steps_per_window < 1:
        raise ValueError(
            f"n_steps_per_window must be ≥ 1; got {n_steps_per_window}"
        )
    tracks: list[Tensor] = []
    for s in seeds:
        frames, _states = world.trajectory(s, n_steps_per_window)
        tracks.append(frames)                                      # [T,1,H,W]
    return torch.stack(tracks, dim=0)                             # [B,T,1,H,W]


def _clone_world(world: ProceduralWorld) -> ProceduralWorld:
    """Create a sibling :class:`ProceduralWorld` sharing cfg + TokenBank but
    with an independent internal ``_rng`` (set on the first ``reset`` call).

    Per-batch-element persistent trajectories (critique C1 / Task #68)
    require one :class:`ProceduralWorld` instance per batch element,
    because ``ProceduralWorld`` stores its generator as instance state
    (shared generator across concurrent states would interleave draws
    in an order-dependent way, breaking determinism).
    """
    return ProceduralWorld(
        world.cfg, world.token_bank,
        seed_family=world.seed_family,
        held_out_regime=world.held_out_regime,
    )


def step_persistent_batch(
    worlds: list[ProceduralWorld], states: list[WorldState],
    n_steps_per_window: int = 2,
) -> tuple[Tensor, list[WorldState]]:
    """Advance each of ``B = len(worlds)`` persistent trajectories by
    ``n_steps_per_window`` frames; return the batched frame tensor and the
    updated per-element ``WorldState`` list.

    Critique C1 / Task #68: Phase-2 training carries BOTH network state
    and world state forward across training steps. Each call advances
    the world in lock-step with the network (which consumes ``T = 2``
    frames per training step).

    Parameters
    ----------
    worlds
        List of ``B`` :class:`ProceduralWorld` instances, each already
        reset to its own trajectory seed.
    states
        Per-element :class:`WorldState` (matches ``worlds`` by index).
    n_steps_per_window
        Number of frames to advance per call (default 2 — one training
        step's worth).

    Returns
    -------
    frames
        Shape ``[B, n_steps_per_window, 1, H, W]``.
    new_states
        Updated :class:`WorldState` list (same length as ``worlds``).
    """
    if n_steps_per_window < 1:
        raise ValueError(
            f"n_steps_per_window must be ≥ 1; got {n_steps_per_window}"
        )
    if len(worlds) != len(states):
        raise ValueError(
            f"len(worlds)={len(worlds)} must match len(states)={len(states)}"
        )
    tracks: list[Tensor] = []
    new_states: list[WorldState] = []
    for w, s in zip(worlds, states):
        frames_b: list[Tensor] = []
        cur = s
        for _ in range(n_steps_per_window):
            f, cur, _info = w.step(cur)
            frames_b.append(f)
        tracks.append(torch.stack(frames_b, dim=0))                # [T,1,H,W]
        new_states.append(cur)
    return torch.stack(tracks, dim=0), new_states                  # [B,T,1,H,W]


# ---------------------------------------------------------------------------
# Apical / basal current helpers for the L23 Urbanczik–Senn updates
# ---------------------------------------------------------------------------

def _l23_apical_basal_drives(
    net: V2Network, state: NetworkStateV2, b_l23: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``(apical, basal)`` pre-update drives at the L2/3 E soma.

    * ``apical`` = H feedback + context-memory bias (top-down prediction).
    * ``basal``  = L4 feedforward + L2/3 recurrent E (bottom-up evidence).

    Inhibitory contributions are *not* included in the error signal: they
    act as a gain-control constraint, not a prediction target.
    """
    l = net.l23_e
    w_fb = F.softplus(l.W_fb_apical_raw)                           # [256,64]
    w_l4 = F.softplus(l.W_l4_l23_raw)                              # [256,128]
    w_rec = F.softplus(l.W_rec_raw) * l.mask_rec                   # [256,256]
    apical = F.linear(state.r_h, w_fb) + b_l23                     # [B,256]
    basal = F.linear(state.r_l4, w_l4) + F.linear(state.r_l23, w_rec)
    return apical, basal


# ---------------------------------------------------------------------------
# Plasticity application (one training micro-step)
# ---------------------------------------------------------------------------

def _assert_plastic(net: V2Network, module: str, weight: str) -> None:
    """Raise :class:`PhaseFrozenError` if ``(module, weight)`` is not plastic
    in the current phase. Centralised guard for every update."""
    manifest = net.plastic_weight_names()
    if (module, weight) not in manifest:
        raise PhaseFrozenError(
            f"refusing to update {module}.{weight}: not plastic under "
            f"phase={net.phase!r}"
        )


def _get_weight(net: V2Network, module: str, weight: str) -> Tensor:
    return getattr(getattr(net, module), weight)


def _raw_prior(net: V2Network, module: str, weight: str, w: Tensor) -> Tensor:
    """Return the raw-prior anchor tensor for ``module.weight`` (Task #50).

    Reads ``module.raw_init_means[weight]`` (float), or falls back to ``0.0``
    if the module does not publish a registry or the weight name is absent.
    Shape / dtype / device match ``w`` so the plasticity rules can subtract
    directly without broadcasting surprises.
    """
    mean = getattr(getattr(net, module), "raw_init_means", {}).get(weight, 0.0)
    return torch.full_like(w, float(mean))


def _apply_update(
    net: V2Network, module: str, weight: str, dw: Tensor,
    energy: EnergyPenalty, pre: Tensor, mask: Optional[Tensor] = None,
    *, rule_lr: Optional[float] = None,
) -> float:
    """Apply ``dw`` plus the energy-shrinkage term to ``module.weight``.

    Returns the absolute mean of the applied update (for logging).

    Task #74 Fix O (2026-04-22): if ``rule_lr == 0.0``, this is a complete
    no-op — no weight decay, no energy shrinkage, no clamp. This makes
    per-rule ablation (set rule.lr=0) honest: zeroing a rule's LR must
    leave its weights bit-identical to init. Previously, the clamp was
    applied regardless of lr, which silently mutated any weight initialised
    outside [-8, 8] on the first training step even under "lr=0" ablation.

    Task #74 Fix P (2026-04-22): the L2 energy shrinkage is now anchored
    at each weight's ``raw_init_means`` value (matches the weight-decay
    anchor already used by the plasticity rules). Shrinking toward zero
    pulled strongly-negative raw weights (e.g. ``W_pv_l23_raw`` init ≈
    -6.5, pred-head weights -8, Dale inhibitory -5) *upward* in raw
    magnitude under any plasticity-driving pre-activity, which *grew*
    their softplus effective weight and destabilised E/I balance.
    """
    _assert_plastic(net, module, weight)
    if rule_lr is not None and float(rule_lr) == 0.0:
        return 0.0
    w = _get_weight(net, module, weight)
    prior = _raw_prior(net, module, weight, w)
    shrink = energy.current_weight_shrinkage(w, pre, mask=mask, raw_prior=prior)
    total = dw + shrink
    w.data.add_(total)
    w.data.clamp_(min=-8.0, max=8.0)
    return float(total.abs().mean().item())


def apply_plasticity_step(
    net: V2Network,
    rules: PlasticityRuleBank,
    state0: NetworkStateV2,
    state1: NetworkStateV2,
    state2: NetworkStateV2,
    info0: dict[str, Tensor],
    info1: dict[str, Tensor],
    x_hat_0: Tensor,
) -> dict[str, float]:
    """Apply every plastic-weight update for a single 2-frame window.

    Signal sources:
      * L23 Urbanczik–Senn runs at step 1 (pre-update state1, drives computed
        from the same pre-update quantities L2/3 used in its forward).
      * Prediction-head Urbanczik–Senn uses
        ``ε = state2.r_l4 − x_hat_0`` (predictive residual) — apical=actual,
        basal=predicted.
      * All Vogels / Hebbian updates use the pre-update presynaptic state
        and the post-update post rate (i.e. the rate that actually fired
        inside the step).

    Returns a ``{weight_key: |Δw|_mean}`` dict for logging.
    """
    out: dict[str, float] = {}

    # ---- L2/3 Urbanczik–Senn (apical-basal at L23E, uses step 1) ---------
    apical_l23, basal_l23 = _l23_apical_basal_drives(
        net, state1, info1["b_l23"],
    )
    for wname, pre_tensor, mask in (
        ("W_l4_l23_raw", state1.r_l4, None),
        ("W_rec_raw", state1.r_l23, net.l23_e.mask_rec.to(torch.bool)),
        ("W_fb_apical_raw", state1.r_h, None),
    ):
        w = _get_weight(net, "l23_e", wname)
        dw = rules.urbanczik.delta(
            pre_activity=pre_tensor, apical=apical_l23, basal=basal_l23,
            weights=w, mask=mask,
            raw_prior=_raw_prior(net, "l23_e", wname, w),
        )
        out[f"l23_e.{wname}"] = _apply_update(
            net, "l23_e", wname, dw, rules.energy, pre_tensor, mask=mask,
            rule_lr=rules.urbanczik.lr,
        )

    # ---- L2/3 inhibitory → E (Vogels iSTDP, target=vogels_target_l23e_hz)
    for wname, pre_tensor in (
        ("W_pv_l23_raw", state1.r_pv),
        ("W_som_l23_raw", state1.r_som),
    ):
        w = _get_weight(net, "l23_e", wname)
        dw = rules.vogels_l23.delta(
            pre_activity=pre_tensor, post_activity=state2.r_l23, weights=w,
            raw_prior=_raw_prior(net, "l23_e", wname, w),
        )
        out[f"l23_e.{wname}"] = _apply_update(
            net, "l23_e", wname, dw, rules.energy, pre_tensor,
            rule_lr=rules.vogels_l23.lr,
        )

    # ---- E → L2/3 PV (frozen in Phase-2) ---------------------------------
    # Task #74 Fix Q (2026-04-22): ``l23_pv.W_pre_raw`` is the L23E→L23PV
    # excitatory synapse. The previous wiring applied ``rules.vogels_ipop``
    # (Vogels iSTDP, target=1 Hz) here, but Vogels iSTDP is designed for
    # I→E synapses: applied to an E→I synapse the homeostatic sign inverts
    # and becomes anti-homeostatic, driving L23E to monotone collapse
    # (Level 10, Task #74 Debugger). Cleanest intervention is to freeze
    # this weight at init end-to-end in Phase-2 — matching the
    # Fix D-simpler pattern used for L23SOM excitatory inputs below.
    # Enforcement is two-sided: (i) ``FastInhibitoryPopulation`` is built
    # with ``freeze_W_pre=True`` so ``plastic_weight_names()`` excludes
    # ``W_pre_raw``; (ii) no ``_apply_update`` call touches it here.

    # Task #74 Fix D-simpler: W_l23_som_raw and W_fb_som_raw are frozen
    # at init in Phase-2 (no Vogels above, no Turrigiano scaling, no other
    # rule). This replaces the earlier Fix D attempt at multi-rule
    # regulation which drove r_som to saturation. Freeze is enforced by
    # the Phase-2 plastic-weight manifest + never calling `_apply_update`
    # on these weights here.

    # ---- H recurrent E-to-E and L2/3 → H (Hebbian, target=0) -------------
    for module, wname, pre_tensor, post_tensor, mask in (
        ("h_e", "W_l23_h_raw", state1.r_l23, state2.r_h, None),
        (
            "h_e", "W_rec_raw", state1.r_h, state2.r_h,
            net.h_e.mask_rec.to(torch.bool),
        ),
    ):
        w = _get_weight(net, module, wname)
        dw = rules.hebb.delta(
            pre_activity=pre_tensor, post_activity=post_tensor, weights=w,
            mask=mask,
            raw_prior=_raw_prior(net, module, wname, w),
        )
        out[f"{module}.{wname}"] = _apply_update(
            net, module, wname, dw, rules.energy, pre_tensor, mask=mask,
            rule_lr=rules.hebb.lr,
        )

    # ---- H inhibitory (Vogels on I→E; target = vogels_target_h_hz) -------
    # Task #74 Fix Q' (2026-04-22): ``h_pv.W_pre_raw`` (HE→HPV, E→I
    # excitatory) is frozen at init for the same reason as ``l23_pv.W_pre_raw``
    # under Fix Q — Vogels iSTDP inverts sign on E→I via the ``-softplus``
    # Dale wrapper and becomes anti-homeostatic. Enforcement: `h_pv` is built
    # with ``freeze_W_pre=True`` so its manifest excludes ``W_pre_raw``; the
    # per-weight update below does not list it.
    for module, wname, pre_tensor, post_tensor, rule in (
        ("h_e", "W_pv_h_raw", state1.h_pv, state2.r_h, rules.vogels_h),
    ):
        w = _get_weight(net, module, wname)
        dw = rule.delta(
            pre_activity=pre_tensor, post_activity=post_tensor, weights=w,
            raw_prior=_raw_prior(net, module, wname, w),
        )
        out[f"{module}.{wname}"] = _apply_update(
            net, module, wname, dw, rules.energy, pre_tensor,
            rule_lr=rule.lr,
        )

    # ---- Context memory generic weights (Hebbian, target=0) --------------
    # These inits are normal(mean=0, std=0.1) so raw_prior=0 ≡ legacy decay;
    # passing raw_prior=None keeps the call site symmetric with other rules.
    for wname, pre_tensor, post_tensor in (
        ("W_hm_gen", state1.r_h, state2.m),
        ("W_mm_gen", state1.m, state2.m),
        ("W_mh_gen", state1.m, info1["b_l23"]),
    ):
        w = _get_weight(net, "context_memory", wname)
        dw = rules.hebb.delta(
            pre_activity=pre_tensor, post_activity=post_tensor, weights=w,
            raw_prior=_raw_prior(net, "context_memory", wname, w),
        )
        out[f"context_memory.{wname}"] = _apply_update(
            net, "context_memory", wname, dw, rules.energy, pre_tensor,
            rule_lr=rules.hebb.lr,
        )

    # ---- Prediction head: Urbanczik–Senn with ε = r_l4_next − x̂_prev ----
    # The streams entering x_hat_0 at step 0 were (h_rate=state1.r_h,
    # c_bias=state0.m, l23_apical=state1.r_l23) — see network.forward().
    for wname, pre_tensor in (
        ("W_pred_H_raw", state1.r_h),
        ("W_pred_C_raw", state0.m),
        ("W_pred_apical_raw", state1.r_l23),
    ):
        w = _get_weight(net, "prediction_head", wname)
        dw = rules.urbanczik.delta(
            pre_activity=pre_tensor, apical=state2.r_l4, basal=x_hat_0,
            weights=w,
            raw_prior=_raw_prior(net, "prediction_head", wname, w),
        )
        out[f"prediction_head.{wname}"] = _apply_update(
            net, "prediction_head", wname, dw, rules.energy, pre_tensor,
            rule_lr=rules.urbanczik.lr,
        )

    # ---- Prediction-head scalar bias b_pred_raw --------------------------
    # Closed-form update: db = lr · mean_b(ε) − wd · (b − b_prior). No "pre"
    # activity — bias has no pre-synaptic partner — so energy shrinkage is
    # omitted (its derivation requires pre-activity). Task #50: anchor decay
    # at the init value so decay doesn't push softplus(b) *upward*.
    # Task #74 Fix O (2026-04-22): honour lr=0 as a complete no-op — no
    # weight decay, no clamp. Mirrors the ``_apply_update`` contract so
    # ablation (``rules.urbanczik.lr = 0``) leaves ``b_pred_raw`` at init.
    _assert_plastic(net, "prediction_head", "b_pred_raw")
    b = net.prediction_head.b_pred_raw
    if float(rules.urbanczik.lr) == 0.0:
        out["prediction_head.b_pred_raw"] = 0.0
    else:
        eps_pred = state2.r_l4 - x_hat_0                           # [B, n_l4]
        b_prior = _raw_prior(net, "prediction_head", "b_pred_raw", b)
        db = (
            rules.urbanczik.lr * eps_pred.mean(dim=0)
            - rules.urbanczik.weight_decay * (b.data - b_prior)
        )
        b.data.add_(db)
        # Task #64: same raw clamp as _apply_update — covers this bias too,
        # which bypasses the helper because it has no presynaptic partner
        # (energy shrinkage is pre-activity-dependent and is therefore omitted).
        b.data.clamp_(min=-8.0, max=8.0)
        out["prediction_head.b_pred_raw"] = float(db.abs().mean().item())

    # ---- Threshold homeostasis -------------------------------------------
    net.l23_e.homeostasis.update(state2.r_l23)
    net.h_e.homeostasis.update(state2.r_h)
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainStepMetrics:
    step: int
    loss_pred: float
    eps_abs_mean: float
    r_l23_mean: float
    r_h_mean: float
    delta_mean: float
    wall_time_s: float


def _forward_window(
    net: V2Network, frames: Tensor, state0: NetworkStateV2,
) -> tuple[NetworkStateV2, NetworkStateV2, NetworkStateV2,
           dict[str, Tensor], dict[str, Tensor], Tensor, Tensor]:
    """Push a ``[B, 2, 1, H, W]`` batch through the net starting from ``state0``.

    Returns ``(state0, state1, state2, info0, info1, x_hat_0, x_hat_1)``.
    ``state0`` is the caller-supplied entry state (typically the previous
    window's ``state2`` to preserve circuit warm-up across training steps).
    ``state1`` and ``state2`` are the post-step-0 and post-step-1 states
    respectively. ``x_hat_0`` is the prediction made at step 0 about
    ``r_l4`` at step 1; the driver's predictive error uses
    ``state2.r_l4 − x_hat_0``.

    Zero-state ``state0`` leaves the prediction-head pre-activities
    (``state1.r_h``, ``state0.m``, ``state1.r_l23``) identically zero for
    the first forward (due to ``rectified_softplus(0)=0``), so the L2/3
    and context-memory streams never receive non-trivial learning signal.
    The training driver therefore **warms up** the state before beginning
    plasticity updates (see ``run_phase2_training``).
    """
    B = frames.shape[0]
    x_hat_0, state1, info0 = net(frames[:, 0], state0)
    x_hat_1, state2, info1 = net(frames[:, 1], state1)
    return state0, state1, state2, info0, info1, x_hat_0, x_hat_1


def _soft_reset_state(state: NetworkStateV2, scale: float = 0.1) -> NetworkStateV2:
    """Scale rate tensors by ``scale`` and reset plasticity bookkeeping.

    Used at Task #56 segment boundaries — soft reset keeps the circuit
    warm (non-zero pre-activities for plasticity) but damps accumulated
    recurrent drift.

    Task #60 fix: in addition to scaling the rate tensors, also damp the
    eligibility-trace dictionaries (``pre_traces`` / ``post_traces``) by the
    same ``scale`` and reset the regime posterior to uniform. Without this,
    traces accumulate across segments — validator Task #57 showed plasticity
    deltas exploding ~20 orders of magnitude between steps 110-250 even
    though rate tensors stayed bounded, because ``Δw ∝ pre_trace × post_trace``
    compounds when either trace is left carrying multi-segment history
    while the driver resets the "epsilon" reference point at each segment.
    Scaling traces alongside rates keeps the trace:rate ratio invariant
    across the boundary, which is the invariant the closed-form plasticity
    rules were derived under.

    ``regime_posterior`` is explicitly reset to uniform (1/n_regimes) rather
    than scaled, since a probability distribution must not be rescaled —
    uniform is the natural "no information" starting point at a boundary.
    """
    new_pre = {k: v * scale for k, v in state.pre_traces.items()}
    new_post = {k: v * scale for k, v in state.post_traces.items()}

    rp = state.regime_posterior
    n_reg = int(rp.shape[-1])
    uniform_posterior = torch.full_like(rp, 1.0 / float(n_reg))

    return state._replace(
        r_l4=state.r_l4 * scale,
        r_l23=state.r_l23 * scale,
        r_pv=state.r_pv * scale,
        r_som=state.r_som * scale,
        r_h=state.r_h * scale,
        h_pv=state.h_pv * scale,
        m=state.m * scale,
        pre_traces=new_pre,
        post_traces=new_post,
        regime_posterior=uniform_posterior,
    )


def run_phase2_training(
    net: V2Network,
    world: ProceduralWorld,
    *,
    n_steps: int,
    batch_size: int,
    seed_offset: int = 0,
    lr_urbanczik: float = 1e-4,
    lr_vogels: float = 1e-4,
    lr_hebb: float = 1e-4,
    weight_decay: float = 1e-5,
    beta_syn: float = 1e-4,
    log_every: int = 10,
    checkpoint_every: int = 0,
    metrics_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    warmup_steps: int = 30,
    segment_length: int = 50,
    soft_reset_scale: float = 0.1,
) -> list[TrainStepMetrics]:
    """Run Phase-2 predictive training in-process.

    The caller owns ``net`` / ``world``; this function mutates ``net``'s
    plastic weights in place and returns a list of
    :class:`TrainStepMetrics` for every logged step. Checkpoints (if
    configured) are written via :func:`torch.save` to
    ``checkpoint_dir / "step_{N}.pt"``.

    State continuity (Task #56 rolling-window update)
    -------------------------------------------------
    Previously the driver reset state to zero every 2-frame window, which
    destroyed the temporal structure that context memory (m) needs to
    learn regime dynamics (Task #49 claim 3). The new rolling policy:

      * One initial ``net.initial_state(...)`` call at entry.
      * Warmup for ``warmup_steps`` forwards with **no plasticity** —
        lets activity propagate through the circuit and settle into the
        stimulus-driven operating point.
      * Main loop carries state forward: each step's exit state becomes
        the next step's entry state, preserving temporal context.
      * Every ``segment_length`` main-loop steps, a **soft reset**
        multiplies all rate tensors by ``soft_reset_scale`` (default 0.1)
        to damp accumulated drift without losing circuit warmth.

    Set ``warmup_steps=0`` and ``segment_length=0`` to fully disable
    warmup and soft reset respectively (the latter produces a single
    unbroken rolling trajectory).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥ 1; got {n_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1; got {batch_size}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be ≥ 0; got {warmup_steps}")
    if segment_length < 0:
        raise ValueError(f"segment_length must be ≥ 0; got {segment_length}")
    if not (0.0 <= soft_reset_scale <= 1.0):
        raise ValueError(
            f"soft_reset_scale must be in [0, 1]; got {soft_reset_scale}"
        )
    if net.phase != "phase2":
        raise PhaseFrozenError(
            f"run_phase2_training requires net.phase == 'phase2'; "
            f"got {net.phase!r}"
        )

    cfg = net.cfg
    rules = PlasticityRuleBank.from_config(
        cfg=cfg,
        lr_urbanczik=lr_urbanczik,
        lr_vogels=lr_vogels,
        lr_hebb=lr_hebb,
        weight_decay=weight_decay,
        beta_syn=beta_syn,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_fh = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = metrics_path.open("w")

    sha_at_start = net.frozen_sensory_core_sha()
    # Task #74 Fix D-simpler: W_l23_som_raw and W_fb_som_raw are frozen at
    # init in Phase-2 (no plasticity rule writes to them). Snapshot here so
    # we can assert bit-exact equality at end-of-training.
    w_l23_som_init = net.l23_som.W_l23_som_raw.detach().clone()
    w_fb_som_init = net.l23_som.W_fb_som_raw.detach().clone()
    t_start = time.monotonic()
    history: list[TrainStepMetrics] = []

    # One initial state at driver entry — carries forward across the
    # warmup loop and the main training loop (Task #56 rolling-window).
    state = net.initial_state(batch_size=batch_size)

    # Critique C1 / Task #68: build B persistent world instances (one
    # per batch element) and keep per-element :class:`WorldState` across
    # training steps. Each training step advances each world by two
    # frames — matching the net, which consumes two frames per step and
    # thereby advances its own internal state by two. At segment
    # boundaries both the network state AND all world states are reset.
    worlds: list[ProceduralWorld] = [_clone_world(world) for _ in range(batch_size)]
    reset_counter = 0

    def _reset_all_world_states() -> list[WorldState]:
        nonlocal reset_counter
        states_new = [
            worlds[b].reset(seed_offset + reset_counter * 10_000 + b)
            for b in range(batch_size)
        ]
        reset_counter += 1
        return states_new

    world_states: list[WorldState] = _reset_all_world_states()

    try:
        # ---- Warmup: forward-only, no plasticity ----------------------
        for w in range(warmup_steps):
            frames, world_states = step_persistent_batch(
                worlds, world_states, n_steps_per_window=2,
            )
            _s0, _s1, state, _i0, _i1, _x0, _x1 = _forward_window(
                net, frames, state,
            )
            if not math.isfinite(state.r_l23.abs().max().item()):
                raise RuntimeError(
                    f"non-finite r_l23 during warmup step {w} — diverged"
                )

        # ---- Main loop: forward + plasticity, carrying state forward -
        for step in range(n_steps):
            frames, world_states = step_persistent_batch(
                worlds, world_states, n_steps_per_window=2,
            )

            (
                state0, state1, state2, info0, info1, x_hat_0, _x_hat_1,
            ) = _forward_window(net, frames, state)

            # Update plasticity *before* computing loss for logging — the
            # update depends on the just-observed error and is closed-form,
            # so ordering relative to logging is irrelevant for fidelity.
            delta_per_w = apply_plasticity_step(
                net, rules, state0, state1, state2, info0, info1, x_hat_0,
            )

            # Carry state forward to next step (Task #56 rolling policy).
            state = state2
            if (
                segment_length > 0
                and (step + 1) % segment_length == 0
                and (step + 1) < n_steps
            ):
                state = _soft_reset_state(state, scale=soft_reset_scale)
                # Critique C1 / Task #68: reset world states at segment
                # boundaries so new trajectories begin alongside the soft
                # network reset (damps accumulated drift without losing
                # circuit warmth).
                world_states = _reset_all_world_states()

            if step % log_every == 0 or step == n_steps - 1:
                eps = state2.r_l4 - x_hat_0
                m = TrainStepMetrics(
                    step=step,
                    loss_pred=float((eps * eps).mean().item()),
                    eps_abs_mean=float(eps.abs().mean().item()),
                    r_l23_mean=float(state2.r_l23.mean().item()),
                    r_h_mean=float(state2.r_h.mean().item()),
                    delta_mean=(
                        sum(delta_per_w.values()) / max(len(delta_per_w), 1)
                    ),
                    wall_time_s=time.monotonic() - t_start,
                )
                history.append(m)
                if metrics_fh is not None:
                    metrics_fh.write(json.dumps(m.__dict__) + "\n")
                    metrics_fh.flush()

            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and (step + 1) % checkpoint_every == 0
            ):
                _save_checkpoint(net, step + 1, checkpoint_dir, sha_at_start)

            # Invariance check: LGN/L4 never mutates.
            if not math.isfinite(state2.r_l23.abs().max().item()):
                raise RuntimeError(
                    f"non-finite r_l23 at step {step} — training diverged"
                )
    finally:
        if metrics_fh is not None:
            metrics_fh.close()

    sha_at_end = net.frozen_sensory_core_sha()
    if sha_at_end != sha_at_start:
        raise RuntimeError(
            "frozen sensory core SHA changed during training — "
            "LGN/L4 was mutated (forbidden)"
        )
    # Task #74 Fix D-simpler: assert SOM excitatory inputs unchanged.
    if not torch.equal(net.l23_som.W_l23_som_raw, w_l23_som_init):
        raise RuntimeError(
            "W_l23_som_raw changed during Phase-2 — must be frozen. "
            f"max |Δ|={(net.l23_som.W_l23_som_raw - w_l23_som_init).abs().max().item():.3e}"
        )
    if not torch.equal(net.l23_som.W_fb_som_raw, w_fb_som_init):
        raise RuntimeError(
            "W_fb_som_raw changed during Phase-2 — must be frozen. "
            f"max |Δ|={(net.l23_som.W_fb_som_raw - w_fb_som_init).abs().max().item():.3e}"
        )
    return history


def _save_checkpoint(
    net: V2Network, step: int, out_dir: Path, sha_at_start: dict[str, str],
) -> Path:
    """Write ``step_{step}.pt`` with network state_dict + provenance."""
    sha_now = net.frozen_sensory_core_sha()
    if sha_now != sha_at_start:
        raise RuntimeError(
            f"frozen sensory core SHA diverged at step {step} — aborting "
            "checkpoint"
        )
    path = out_dir / f"step_{step}.pt"
    torch.save(
        {
            "step": step,
            "state_dict": net.state_dict(),
            "phase": net.phase,
            "frozen_sha": sha_now,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-2 pure-local predictive training driver (V2).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("checkpoints/v2/phase2"),
    )
    p.add_argument(
        "--held-out-regime", type=str, default=None,
        choices=("CW-drift", "CCW-drift", "low-hazard", "high-hazard"),
    )
    p.add_argument("--lr-urbanczik", type=float, default=1e-4)
    p.add_argument("--lr-vogels", type=float, default=1e-4)
    p.add_argument("--lr-hebb", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--beta-syn", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu")
    # Task #56: rolling-window state policy.
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--segment-length", type=int, default=50)
    p.add_argument("--soft-reset-scale", type=float, default=0.1)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    torch.manual_seed(int(args.seed))

    cfg = ModelConfig(seed=int(args.seed), device=args.device)
    world, bank = build_world(
        cfg, seed_family="train", held_out_regime=args.held_out_regime,
    )
    net = V2Network(cfg, token_bank=bank, seed=int(args.seed))
    net.set_phase("phase2")

    seed_dir = args.out_dir / f"phase2_s{int(args.seed)}"
    metrics_path = seed_dir / "metrics.jsonl"
    history = run_phase2_training(
        net=net,
        world=world,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        seed_offset=int(args.seed) * 10_000,
        lr_urbanczik=float(args.lr_urbanczik),
        lr_vogels=float(args.lr_vogels),
        lr_hebb=float(args.lr_hebb),
        weight_decay=float(args.weight_decay),
        beta_syn=float(args.beta_syn),
        log_every=int(args.log_every),
        checkpoint_every=int(args.checkpoint_every),
        metrics_path=metrics_path,
        warmup_steps=int(args.warmup_steps),
        segment_length=int(args.segment_length),
        soft_reset_scale=float(args.soft_reset_scale),
        checkpoint_dir=seed_dir,
    )

    if history:
        last = history[-1]
        print(
            f"step={last.step} loss_pred={last.loss_pred:.4e} "
            f"|eps|={last.eps_abs_mean:.4e} r_l23={last.r_l23_mean:.3f} "
            f"r_h={last.r_h_mean:.3f} wall={last.wall_time_s:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
