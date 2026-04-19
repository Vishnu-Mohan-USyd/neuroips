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
from src.v2_model.world.procedural import ProceduralWorld

__all__ = [
    "PhaseFrozenError",
    "PlasticityRuleBank",
    "build_world",
    "run_phase2_training",
    "sample_batch_window",
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
    """Container for the three closed-form rules consumed by the driver.

    One instance per driver invocation. Learning rates come from
    ``cfg.plasticity``; the caller may override any LR via CLI.
    """
    urbanczik: UrbanczikSennRule
    vogels: VogelsISTDPRule
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
            vogels=VogelsISTDPRule(
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


def _apply_update(
    net: V2Network, module: str, weight: str, dw: Tensor,
    energy: EnergyPenalty, pre: Tensor, mask: Optional[Tensor] = None,
) -> float:
    """Apply ``dw`` plus the energy-shrinkage term to ``module.weight``.

    Returns the absolute mean of the applied update (for logging).
    """
    _assert_plastic(net, module, weight)
    w = _get_weight(net, module, weight)
    shrink = energy.current_weight_shrinkage(w, pre, mask=mask)
    total = dw + shrink
    w.data.add_(total)
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
        )
        out[f"l23_e.{wname}"] = _apply_update(
            net, "l23_e", wname, dw, rules.energy, pre_tensor, mask=mask,
        )

    # ---- L2/3 inhibitory → E (Vogels iSTDP, target=ρ) --------------------
    for wname, pre_tensor in (
        ("W_pv_l23_raw", state1.r_pv),
        ("W_som_l23_raw", state1.r_som),
    ):
        w = _get_weight(net, "l23_e", wname)
        dw = rules.vogels.delta(
            pre_activity=pre_tensor, post_activity=state2.r_l23, weights=w,
        )
        out[f"l23_e.{wname}"] = _apply_update(
            net, "l23_e", wname, dw, rules.energy, pre_tensor,
        )

    # ---- E → L2/3 PV, L2/3 SOM (Vogels on the I-pop side, target=ρ) -----
    for module, wname, pre_tensor, post_tensor in (
        ("l23_pv", "W_pre_raw", state1.r_l23, state2.r_pv),
        ("l23_som", "W_l23_som_raw", state1.r_l23, state2.r_som),
        ("l23_som", "W_fb_som_raw", state1.r_h, state2.r_som),
    ):
        w = _get_weight(net, module, wname)
        dw = rules.vogels.delta(
            pre_activity=pre_tensor, post_activity=post_tensor, weights=w,
        )
        out[f"{module}.{wname}"] = _apply_update(
            net, module, wname, dw, rules.energy, pre_tensor,
        )

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
        )
        out[f"{module}.{wname}"] = _apply_update(
            net, module, wname, dw, rules.energy, pre_tensor, mask=mask,
        )

    # ---- H inhibitory and H PV (Vogels, target=ρ) ------------------------
    for module, wname, pre_tensor, post_tensor in (
        ("h_e", "W_pv_h_raw", state1.h_pv, state2.r_h),
        ("h_pv", "W_pre_raw", state1.r_h, state2.h_pv),
    ):
        w = _get_weight(net, module, wname)
        dw = rules.vogels.delta(
            pre_activity=pre_tensor, post_activity=post_tensor, weights=w,
        )
        out[f"{module}.{wname}"] = _apply_update(
            net, module, wname, dw, rules.energy, pre_tensor,
        )

    # ---- Context memory generic weights (Hebbian, target=0) --------------
    for wname, pre_tensor, post_tensor in (
        ("W_hm_gen", state1.r_h, state2.m),
        ("W_mm_gen", state1.m, state2.m),
        ("W_mh_gen", state1.m, info1["b_l23"]),
    ):
        w = _get_weight(net, "context_memory", wname)
        dw = rules.hebb.delta(
            pre_activity=pre_tensor, post_activity=post_tensor, weights=w,
        )
        out[f"context_memory.{wname}"] = _apply_update(
            net, "context_memory", wname, dw, rules.energy, pre_tensor,
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
        )
        out[f"prediction_head.{wname}"] = _apply_update(
            net, "prediction_head", wname, dw, rules.energy, pre_tensor,
        )

    # ---- Prediction-head scalar bias b_pred_raw --------------------------
    # Closed-form update: db = lr · mean_b(ε) − wd · b_pred_raw. No "pre"
    # activity — bias has no pre-synaptic partner — so energy shrinkage is
    # omitted (its derivation requires pre-activity).
    _assert_plastic(net, "prediction_head", "b_pred_raw")
    b = net.prediction_head.b_pred_raw
    eps_pred = state2.r_l4 - x_hat_0                               # [B, n_l4]
    db = (
        rules.urbanczik.lr * eps_pred.mean(dim=0)
        - rules.urbanczik.weight_decay * b.data
    )
    b.data.add_(db)
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
    warmup_steps: int = 10,
) -> list[TrainStepMetrics]:
    """Run Phase-2 predictive training in-process.

    The caller owns ``net`` / ``world``; this function mutates ``net``'s
    plastic weights in place and returns a list of
    :class:`TrainStepMetrics` for every logged step. Checkpoints (if
    configured) are written via :func:`torch.save` to
    ``checkpoint_dir / "step_{N}.pt"``.

    State continuity
    ----------------
    Activity propagates through the circuit with a one-step delay per
    population (LGN/L4 → L2/3 → H → C). From a zero initial state the
    L2/3, H and C rates are exactly zero for the first forward because
    ``rectified_softplus(0) = 0``, which would null every learning signal
    that uses those rates as pre-activity. The driver therefore
      * runs ``warmup_steps`` forwards with no plasticity to propagate
        activity through the full circuit before any update fires, and
      * carries the **end state** of each 2-frame training window forward
        as the entry state of the next window, so the circuit operates at
        its natural steady state rather than resetting every step.
    Set ``warmup_steps=0`` to disable both (tests that construct state
    themselves rely on the callable units — see
    ``test_phase2_driver_freeze_manifest.py``).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥ 1; got {n_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1; got {batch_size}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be ≥ 0; got {warmup_steps}")
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
    t_start = time.monotonic()
    history: list[TrainStepMetrics] = []

    # Reset state each window — the V2 circuit is unstable under long
    # no-plasticity rollouts from zero state (E-to-E recurrence amplifies
    # before homeostasis/iSTDP can react). A two-frame window starting at
    # state=0 is well within the stable regime.
    try:
        for step in range(n_steps):
            seeds = [seed_offset + step * batch_size + b for b in range(batch_size)]
            frames = sample_batch_window(world, seeds, n_steps_per_window=2)

            state0 = net.initial_state(batch_size=batch_size)
            (
                state0, state1, state2, info0, info1, x_hat_0, _x_hat_1,
            ) = _forward_window(net, frames, state0)

            # Update plasticity *before* computing loss for logging — the
            # update depends on the just-observed error and is closed-form,
            # so ordering relative to logging is irrelevant for fidelity.
            delta_per_w = apply_plasticity_step(
                net, rules, state0, state1, state2, info0, info1, x_hat_0,
            )

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
