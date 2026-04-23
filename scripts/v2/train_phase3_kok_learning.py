"""Phase-3 Kok task-learning driver (plan v4 / Task #40; Task #74 Fix B+C; Fix J).

Freezes the sensory core and every *generic* circuit weight; only the
task-specific context-memory weights ``W_qm_task`` and ``W_mh_task_exc``
adapt, via :class:`ThreeFactorRule`. ``W_mh_task_inh`` is preserved in
the state-dict for backward compatibility but is inert under Fix J
(never updated; stays at its zero init → ``som_gain`` ≡ 1.0).

Task #74 Fix J (2026-04-21): the Fix-C-v2 SOM-gain path is deprecated in
favor of an additive L23E-space path with a DIFFERENTIAL modulator:

  * ``W_mh_task_exc`` [n_l23_e, n_m] — main (now only) task readout to
    L2/3 E apical, applied with unit gain (task_exc_gain=1.0 in
    network.py, up from 0.1) so the task bias actually modulates L23E.
    Three-factor update:
        ΔW_mh_task_exc[i, c] ∝ lr · l23e_mod[i] · m_start_probe[c]
    where
        l23e_mod = EMA_l23e[cue, expected] − EMA_l23e[cue, unexpected]
    is a per-L23E-unit signed contrast built from 4 running EMAs over
    (cue × is_expected) of the probe-1 r_l23 mean.
  * ``W_mh_task_inh`` — NOT updated under Fix J. Stays at zero init, so
    ``som_gain = softplus(0·m + 0.5413).clamp(max=4.0)`` ≡ 1.0. The
    SOM-gain route becomes a no-op, matching Fix-E init behavior.

Task #74 Fix B (magnitude): the outer-loop multiplier ``lr_mh_scale``
(default 20.0) scales the learning rate on both W_mh_task_{exc,inh}
updates, and the resulting weights are clamped to ±8.0 per element per
trial as a runaway safeguard.

Kok 2012 trial timing (``dt=5 ms``):
  * 200 ms cue    → 40 steps    (cue token ``q_t`` active, blank frame)
  * 550 ms delay  → 110 steps   (blank frame, zero cue)
  * 500 ms probe1 → 100 steps   (oriented grating, zero cue)
  * 100 ms blank  → 20 steps    (blank frame)
  * 500 ms probe2 → 100 steps   (oriented grating, zero cue)

Cue-mapping counterbalance: seed parity decides which of the two cues
points at which orientation (odd → cue0=45°, cue1=135°; even → reversed),
so the mapping is not confounded with rate-space asymmetries in the
post-Phase-2 network.

Learning sub-phase: deterministic (cue predicts probe orientation 100 %).
Scan sub-phase:     probabilistic (75 % valid, 25 % mismatched).

Plasticity (closed-form, no ``backward``):
  * ``W_qm_task`` via :meth:`ThreeFactorRule.delta_qm` with
    ``cue = q_t``, ``memory = m_end_cue``,
    ``memory_error = m_end_cue − m_pre_cue`` (cue-evoked change).
  * ``W_mh_task_exc`` via :meth:`ThreeFactorRule.delta_mh` with
    ``memory = m_start_probe``,
    ``probe_error = l23e_modulator = ema_exp[cue] − ema_unexp[cue]``
    (L23E-space differential EMA; Fix J).
  * ``W_mh_task_inh`` — NOT UPDATED under Fix J (inert; see module
    docstring). The zero init guarantees ``som_gain ≡ 1``.

Frozen-sensory-core SHA is asserted identical before/after training; every
non-plastic Parameter under the current phase manifest is also checksummed
to catch accidental writes outside the two task weights.

Usage:
    python -m scripts.v2.train_phase3_kok_learning \\
        --phase2-checkpoint checkpoints/v2/phase2/phase2_s42/step_1000.pt \\
        --seed 42 --n-trials-learning 5000 --n-trials-scan 10000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import make_blank_frame, make_grating_frame
from scripts.v2.train_phase2_predictive import PhaseFrozenError
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.plasticity import ThreeFactorRule
from src.v2_model.stimuli.feature_tokens import TokenBank


__all__ = [
    "CUE_ORIENTATIONS_DEG", "KokTiming", "cue_mapping_from_seed",
    "build_cue_tensor", "run_kok_trial", "run_phase3_kok_training",
]


# ---------------------------------------------------------------------------
# Paradigm constants
# ---------------------------------------------------------------------------

CUE_ORIENTATIONS_DEG: tuple[float, float] = (45.0, 135.0)


@dataclass
class KokTiming:
    """Per-trial step budget (integer ``dt=5 ms`` steps)."""
    cue_steps: int = 40          # 200 ms
    delay_steps: int = 110       # 550 ms
    probe1_steps: int = 100      # 500 ms
    blank_steps: int = 20        # 100 ms
    probe2_steps: int = 100      # 500 ms

    @property
    def total(self) -> int:
        return (self.cue_steps + self.delay_steps + self.probe1_steps
                + self.blank_steps + self.probe2_steps)


def cue_mapping_from_seed(seed: int) -> dict[int, float]:
    """Cue-id → orientation (deg) map. Counterbalanced by seed parity.

    Odd seed  ⇒ {0: 45°, 1: 135°}.
    Even seed ⇒ {0: 135°, 1: 45°}.
    """
    if int(seed) % 2 == 1:
        return {0: CUE_ORIENTATIONS_DEG[0], 1: CUE_ORIENTATIONS_DEG[1]}
    return {0: CUE_ORIENTATIONS_DEG[1], 1: CUE_ORIENTATIONS_DEG[0]}


def build_cue_tensor(cue_id: int, n_cue: int, device: str = "cpu") -> Tensor:
    """One-hot cue tensor ``[1, n_cue]`` with a single 1.0 at position ``cue_id``.

    Only the first two slots (0, 1) are used by the 2-class Kok paradigm;
    the remaining ``n_cue - 2`` dims stay exactly zero so W_qm_task does
    not accrue spurious outer-product entries outside the used subspace.
    """
    if cue_id not in (0, 1):
        raise ValueError(f"cue_id must be 0 or 1; got {cue_id}")
    q = torch.zeros(1, int(n_cue), device=device, dtype=torch.float32)
    q[0, int(cue_id)] = 1.0
    return q


# ---------------------------------------------------------------------------
# Frozen-weight snapshot (for the end-of-training integrity check)
# ---------------------------------------------------------------------------


def _frozen_params(net: V2Network) -> list[tuple[str, str]]:
    """Every named Parameter *not* in the current phase's plastic manifest.

    LGN/L4 is skipped — it has no Parameters (frozen-by-construction) and
    is covered by :meth:`V2Network.frozen_sensory_core_sha`.
    """
    plastic = set(net.plastic_weight_names())
    out: list[tuple[str, str]] = []
    for mod_name, child in net.named_children():
        if mod_name == "lgn_l4":
            continue
        for p_name, _p in child.named_parameters(recurse=False):
            if (mod_name, p_name) in plastic:
                continue
            out.append((mod_name, p_name))
    return out


def _snapshot_weights(
    net: V2Network, keys: list[tuple[str, str]],
) -> dict[tuple[str, str], Tensor]:
    snaps: dict[tuple[str, str], Tensor] = {}
    for mod, wname in keys:
        snaps[(mod, wname)] = (
            getattr(getattr(net, mod), wname).detach().clone()
        )
    return snaps


def _assert_snapshot_unchanged(
    net: V2Network, snaps: dict[tuple[str, str], Tensor],
) -> None:
    for (mod, wname), w_before in snaps.items():
        w_after = getattr(getattr(net, mod), wname).data
        if not torch.equal(w_before, w_after):
            raise RuntimeError(
                f"frozen weight {mod}.{wname} was mutated during Phase-3 "
                f"Kok training (phase={net.phase!r})"
            )


# Task #74 (Fix A_homeo): homeostasis θ is a non-Parameter buffer on the
# excitatory populations' ThresholdHomeostasis submodule. The original
# `_assert_snapshot_unchanged` only covers nn.Parameters, so θ drift slipped
# through. These helpers snapshot + verify θ stays bit-identical across
# Phase-3 training — a defense-in-depth invariant in addition to the
# Phase-3 call-site removal of `homeostasis.update()`.

def _snapshot_theta(net: V2Network) -> dict[str, Tensor]:
    """Snapshot sensory-core excitatory θ buffers (L2/3 E, H E)."""
    return {
        "l23_e": net.l23_e.homeostasis.theta.detach().clone(),
        "h_e": net.h_e.homeostasis.theta.detach().clone(),
    }


def _assert_theta_unchanged(
    net: V2Network, snaps: dict[str, Tensor],
) -> None:
    for pop_name, theta_before in snaps.items():
        theta_after = getattr(net, pop_name).homeostasis.theta.data
        if not torch.equal(theta_before, theta_after):
            max_abs_delta = float((theta_after - theta_before).abs().max().item())
            raise RuntimeError(
                f"frozen homeostasis θ on {pop_name} was mutated during "
                f"Phase-3 Kok training (phase={net.phase!r}, "
                f"max|Δθ|={max_abs_delta:.3e})"
            )


def _assert_plastic(net: V2Network, module: str, wname: str) -> None:
    if (module, wname) not in net.plastic_weight_names():
        raise PhaseFrozenError(
            f"refusing to update {module}.{wname}: not plastic under "
            f"phase={net.phase!r}"
        )


# ---------------------------------------------------------------------------
# Trial execution + plasticity
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_kok_trial(
    net: V2Network, cfg: ModelConfig,
    *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, rule: ThreeFactorRule,
    noise_std: float = 0.0, device: str = "cpu",
    apply_plasticity: bool = True,
    l23e_modulator: Optional[Tensor] = None,
    som_modulator: Optional[Tensor] = None,  # DEPRECATED (Fix J): ignored.
    lr_mh_scale: float = 1.0,
    lr_qm_scale: float = 1.0,
    weight_clamp: float = 8.0,
    weight_clamp_inh: float = 8.0,  # DEPRECATED (Fix J): no-op; W_mh_task_inh is inert.
    disable_fix_j_mh_exc: bool = False,  # Task #74 β-Step-3: zero out Fix J.
) -> dict[str, Tensor]:
    """Run one Kok trial through the network; apply plasticity at end.

    Returns a dict with the signals consumed by the update so the
    caller can log them; the update has already been applied in-place
    when ``apply_plasticity=True`` (the default).

    Task #74 Fix J (behaviour-affecting args):
      * ``l23e_modulator``: optional ``[1, n_l23_e]`` signed contrast
        ``EMA_l23e[cue_id, True] − EMA_l23e[cue_id, False]`` built by
        the caller from 4 running per-(cue × is_expected) EMAs of the
        probe-1 ``r_l23`` mean. Passed as the ``probe_error`` to
        :meth:`ThreeFactorRule.delta_mh` on ``W_mh_task_exc``. If
        ``None``, the update degenerates to pure weight-decay (no Hebb
        term). The EMA-differential is the three-factor modulator; the
        eligibility trace is ``memory = m_start_probe``.
      * ``som_modulator`` — DEPRECATED. Accepted for backward compat
        but ignored; ``W_mh_task_inh`` is no longer updated under
        Fix J (stays at zero init → ``som_gain`` ≡ 1.0).
      * ``lr_mh_scale``: scales the learning rate on ``W_mh_task_exc``
        (Fix B; kept at 1.0 by default).
      * ``lr_qm_scale``: scales the learning rate on ``W_qm_task``.
      * ``weight_clamp``: per-element |W| cap for ``W_mh_task_exc``
        (Fix B runaway safeguard, default 8.0).
      * ``weight_clamp_inh`` — DEPRECATED no-op under Fix J.

    The trial also returns ``r_l23_probe1_mean_vec`` ``[1, n_l23_e]``
    (mean r_l23 across the probe1 epoch) so the caller can update its
    per-(cue × is_expected) EMA buffers. ``r_som_probe1_mean`` is still
    returned for diagnostic logging but no longer feeds the rule.

    Critique C2 / Task #68: the scan sub-phase evaluates responses on
    already-learned weights. Leaving plasticity active during scan
    continues mutating task weights from random trial labels, diluting
    the signal. Callers in the scan sub-phase pass
    ``apply_plasticity=False`` so the updates are computed for logging
    but **not** written to the parameters. Signals (``dw_*_abs_mean``)
    still reflect what the rule WOULD have produced.
    """
    _assert_plastic(net, "context_memory", "W_qm_task")
    _assert_plastic(net, "context_memory", "W_mh_task_exc")
    # W_mh_task_inh remains in the phase3_kok "plastic" manifest for
    # state-dict compat, but Fix J does not update it. We still check
    # it is declared plastic so a future manifest change doesn't
    # silently reintroduce the SOM-gain path.
    _assert_plastic(net, "context_memory", "W_mh_task_inh")

    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = net.initial_state(batch_size=1)
    m_pre_cue = state.m.clone()                                 # [1, n_m]
    m_end_cue: Optional[Tensor] = None
    m_start_probe: Optional[Tensor] = None
    b_l23_pre_probe: Optional[Tensor] = None
    probe1_l23: list[Tensor] = []
    probe1_som: list[Tensor] = []

    n_total = timing.total
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps

    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, None
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn_like(frame)

        _x_hat, state, info = net(frame, state, q_t=q_t)

        # Task #74 (Fix A_homeo) — θ homeostasis is DISABLED during Phase-3.
        # Rationale: CHECK 3 on Task#70 ckpt showed 100 Phase-3-Kok trials
        # with `homeostasis.update` no-op'd recovers L2/3 coverage entropy
        # from 0.558 → 1.636 nats. θ drift in Phase-3 (a non-Parameter
        # buffer, outside `_assert_snapshot_unchanged`'s original scope)
        # was silently collapsing the sensory tuning. Phase-2 drivers
        # still call `homeostasis.update` — this gate is Phase-3-only.

        if t == cue_end - 1:
            m_end_cue = state.m.clone()
        if t == delay_end - 1:
            m_start_probe = state.m.clone()
        if t == delay_end:
            b_l23_pre_probe = info["b_l23"].clone()
        if delay_end <= t < probe1_end:
            probe1_l23.append(info["r_l23"].clone())
            probe1_som.append(info["r_som"].clone())

    assert m_end_cue is not None and m_start_probe is not None
    assert b_l23_pre_probe is not None and probe1_l23 and probe1_som

    # --- W_qm_task: cue → memory (three-factor, delta_qm) ---------------
    memory_error_qm = m_end_cue - m_pre_cue                      # [1, n_m]
    eff_lr_qm = float(rule.lr) * float(lr_qm_scale)
    qm_rule = ThreeFactorRule(lr=eff_lr_qm, weight_decay=float(rule.weight_decay))
    dw_qm = qm_rule.delta_qm(
        cue=q_cue, memory=m_end_cue, memory_error=memory_error_qm,
        weights=net.context_memory.W_qm_task,
    )
    if apply_plasticity:
        net.context_memory.W_qm_task.data.add_(dw_qm)
        # Runaway safeguard: cap |W_qm_task| at 1.0 per element to prevent
        # the cue → memory → cue positive-feedback loop from diverging during
        # early Phase-3 training (Task #58 / debugger Task #49 Claim 4).
        net.context_memory.W_qm_task.data.clamp_(min=-1.0, max=1.0)

    # --- W_mh_task_{exc,inh}: memory → L2/3 (Task #74 Fix B+C) -----------
    # Fix B: scale the lr on the mh updates by lr_mh_scale (default 20×).
    eff_lr_mh = float(rule.lr) * float(lr_mh_scale)
    mh_rule = ThreeFactorRule(lr=eff_lr_mh, weight_decay=float(rule.weight_decay))

    r_l23_probe1_mean = torch.stack(probe1_l23, dim=0).mean(dim=0)  # [1, n_l23]
    r_som_probe1_mean = torch.stack(probe1_som, dim=0).mean(dim=0)  # [1, n_som]

    # Fix J: single readout to L23 E apical. probe_error is the caller-built
    # L23E-space differential EMA modulator (shape [1, n_l23_e]). If the
    # caller did not provide one (first-trial bootstrap, or a test harness),
    # fall back to zeros so the update degenerates to pure weight-decay.
    n_l23_e = int(net.context_memory.n_out)
    if l23e_modulator is None:
        probe_error_mh = torch.zeros(1, n_l23_e, device=device, dtype=torch.float32)
    else:
        probe_error_mh = l23e_modulator
    dw_mh_exc = mh_rule.delta_mh(
        memory=m_start_probe, probe_error=probe_error_mh,
        weights=net.context_memory.W_mh_task_exc,
    )
    # Task #74 β-Step-3 (2026-04-23): when ``disable_fix_j_mh_exc`` is set,
    # zero out the W_mh_task_exc update so Fix J's L23E-differential EMA
    # path is inert. Any expectation effect observed under this setting is
    # therefore attributable to W_q_gain alone. The dw_mh_exc tensor is
    # still returned for schema/metrics compat, but the parameter is not
    # mutated. Because W_mh_task_exc initialises at zero, leaving it
    # unupdated keeps the task_exc apical readout at exactly zero drive.
    if apply_plasticity and not disable_fix_j_mh_exc:
        net.context_memory.W_mh_task_exc.data.add_(dw_mh_exc)
        net.context_memory.W_mh_task_exc.data.clamp_(
            min=-float(weight_clamp), max=float(weight_clamp),
        )

    # Fix J: W_mh_task_inh is INERT — no update, no clamp, no delta_mh_inh
    # call. It stays at its zero init, so ``som_gain ≡ 1.0`` everywhere.
    # We still emit a zero ``dw_mh_inh_abs_mean`` for metrics/schema compat.
    dw_mh_inh_abs_mean = torch.zeros((), dtype=torch.float32)

    return {
        "dw_qm_abs_mean": dw_qm.abs().mean().detach(),
        "dw_mh_exc_abs_mean": dw_mh_exc.abs().mean().detach(),
        "dw_mh_inh_abs_mean": dw_mh_inh_abs_mean,
        "probe1_r_l23_mean": r_l23_probe1_mean.mean().detach(),
        "r_l23_probe1_mean_vec": r_l23_probe1_mean.detach(),  # [1, n_l23_e] (Fix J EMA source)
        "r_som_probe1_mean": r_som_probe1_mean.detach(),      # [1, n_som] (diagnostic)
        "m_end_cue_mean": m_end_cue.mean().detach(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainStepMetrics:
    phase_name: str
    trial: int
    cue_id: int
    probe_orientation_deg: float
    is_expected: bool
    dw_qm_abs_mean: float
    dw_mh_exc_abs_mean: float
    dw_mh_inh_abs_mean: float
    probe1_r_l23_mean: float
    wall_time_s: float


@dataclass
class KokHistory:
    steps: list[TrainStepMetrics] = field(default_factory=list)


@torch.no_grad()
def apply_w_q_gain_update(
    net: V2Network,
    *, cue_id: int,
    r_l23e_vec: Tensor,
    matched: bool,
    lr: float,
    gain_min: float = 0.1,
    gain_max: float = 1.0,
) -> float:
    """Apply the Step-2 three-factor rule to ``l23_e.W_q_gain`` in place.

    ΔW_q_gain[cue_id, :] = −lr · cue_on · r_l23e · sign(matched)
        where sign = +1 if ``matched`` else −1.
    Then clamp to ``[gain_min, gain_max]``. Returns the L2 norm of the
    actual (post-clamp) delta for logging.

    Args
    ----
    net           : V2Network (phase must be ``phase3_kok``).
    cue_id        : int in [0, n_cue) — the presented cue on this trial.
    r_l23e_vec    : Tensor of shape ``[n_l23_e]`` or ``[1, n_l23_e]`` —
                    per-unit probe-1 mean L2/3 E rate from this trial.
    matched       : bool. True if the probe orientation equals
                    ``cue_mapping[cue_id]`` (expected trial); False on
                    unexpected trials.
    lr            : scalar learning rate (typically 1e-3).
    gain_min/max  : post-update clamp (defaults match Step 2 toy).

    Raises
    ------
    PhaseFrozenError : if the phase manifest does not declare W_q_gain
                       plastic (e.g. caller accidentally flipped phase).
    """
    if ("l23_e", "W_q_gain") not in net.plastic_weight_names():
        raise PhaseFrozenError(
            "refusing to update l23_e.W_q_gain: not plastic under "
            f"phase={net.phase!r}"
        )
    r = r_l23e_vec
    if r.dim() == 2:
        r = r.squeeze(0)
    sign = 1.0 if matched else -1.0
    delta = -float(lr) * float(sign) * r
    W = net.l23_e.W_q_gain                                      # [n_cue, n_l23_e]
    new_row = torch.clamp(W[cue_id] + delta, float(gain_min), float(gain_max))
    actual_delta = new_row - W[cue_id]
    W[cue_id].copy_(new_row)
    return float(actual_delta.norm().item())


def _pick_scan_probe(
    cue_id: int, cue_mapping: dict[int, float], validity: float,
    np_rng: np.random.Generator,
) -> float:
    """Sample a probe orientation: ``validity`` probability of matching cue."""
    if float(validity) >= 1.0 or np_rng.random() < float(validity):
        return float(cue_mapping[cue_id])
    other = 1 - int(cue_id)
    return float(cue_mapping[other])


def run_phase3_kok_training(
    net: V2Network,
    *, n_trials_learning: int,
    n_trials_scan: int,
    validity_scan: float = 0.75,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    timing: Optional[KokTiming] = None,
    noise_std: float = 0.0,
    cue_mapping: Optional[dict[int, float]] = None,
    metrics_path: Optional[Path] = None,
    log_every: int = 50,
    lr_mh_scale: float = 1.0,
    lr_qm_scale: float = 1.0,
    l23e_ema_alpha: float = 0.01,
    som_ema_alpha: Optional[float] = None,  # DEPRECATED alias for l23e_ema_alpha.
    weight_clamp: float = 8.0,
    weight_clamp_inh: float = 8.0,  # DEPRECATED no-op under Fix J.
    # Task #74 β-mechanism Step 3 (2026-04-23):
    disable_fix_j_mh_exc: bool = False,
    enable_w_q_gain_rule: bool = False,
    w_q_gain_lr: float = 1e-3,
    w_q_gain_clamp: tuple[float, float] = (0.1, 1.0),
) -> KokHistory:
    """Run the two-sub-phase Kok trainer in-place on ``net``.

    * Learning sub-phase — ``n_trials_learning`` trials, 100 % valid.
    * Scan sub-phase     — ``n_trials_scan`` trials, ``validity_scan`` valid.

    The caller owns ``net``; this function mutates ``W_qm_task`` and
    ``W_mh_task_exc`` in place. ``W_mh_task_inh`` is preserved unchanged
    (Fix J). No other Parameter is touched — an integrity check at exit
    asserts this.

    Task #74 Fix J: maintains 4 running EMAs of probe-1 r_l23 activity
    (shape ``[n_l23_e]``) over (cue × is_expected). The modulator passed
    to :meth:`ThreeFactorRule.delta_mh` on trial ``t`` is
    ``EMA_l23e[cue_t, True] − EMA_l23e[cue_t, False]`` (a ``[1, n_l23_e]``
    differential contrast — the three-factor modulator). Eligibility is
    ``m_start_probe``. ``l23e_ema_alpha`` (default 0.01) is the EMA
    coefficient. ``som_ema_alpha`` is a deprecated alias kept for
    backward-compat callers.
    """
    if som_ema_alpha is not None:
        import warnings
        warnings.warn(
            "som_ema_alpha is deprecated under Fix J; use l23e_ema_alpha. "
            "The value is being passed through to l23e_ema_alpha.",
            DeprecationWarning,
            stacklevel=2,
        )
        l23e_ema_alpha = float(som_ema_alpha)
    if net.phase != "phase3_kok":
        raise PhaseFrozenError(
            f"run_phase3_kok_training requires phase='phase3_kok'; "
            f"got {net.phase!r}"
        )
    if n_trials_learning < 0 or n_trials_scan < 0:
        raise ValueError("trial counts must be ≥ 0")
    if not (0.0 <= validity_scan <= 1.0):
        raise ValueError(f"validity_scan must be in [0, 1]; got {validity_scan}")

    cfg = net.cfg
    timing = timing or KokTiming()
    cue_mapping = cue_mapping or cue_mapping_from_seed(seed)
    rule = ThreeFactorRule(lr=float(lr), weight_decay=float(weight_decay))

    sha_at_start = net.frozen_sensory_core_sha()
    frozen_keys = _frozen_params(net)
    frozen_snaps = _snapshot_weights(net, frozen_keys)
    theta_snaps = _snapshot_theta(net)  # Task #74 Fix A_homeo

    np_rng = np.random.default_rng(int(seed))
    history = KokHistory()
    metrics_fh = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = metrics_path.open("w")

    # Task #74: cumulative-trial counter for stdout progress prints across
    # the two sub-phases (so the progress cadence is global, not per-phase).
    total_all = int(n_trials_learning) + int(n_trials_scan)
    global_trial = 0

    # Task #74 Fix J: per-(cue × is_expected) running EMAs of probe-1 r_l23.
    # Four buffers total; each shape [n_l23_e]. Zero-initialised; update
    # as ema ← (1−α)·ema + α·r_l23 after every trial. The L23E modulator
    # passed into delta_mh on W_mh_task_exc is
    #   l23e_mod = ema_exp[cue] − ema_unexp[cue].
    n_l23_e = int(net.context_memory.n_out)
    n_som = int(net.context_memory.n_out_som)
    device = str(cfg.device)
    l23e_ema: dict[tuple[int, bool], Tensor] = {
        (c, is_exp): torch.zeros(n_l23_e, device=device, dtype=torch.float32)
        for c in (0, 1) for is_exp in (True, False)
    }
    alpha = float(l23e_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"l23e_ema_alpha must be in (0, 1]; got {alpha}")

    t_start = time.monotonic()
    try:
        # Critique C2 / Task #68: plasticity is ON in the learning sub-phase
        # (trains the cue→memory→L23 mapping) and OFF in the scan sub-phase
        # (which evaluates the already-learned weights against probabilistic
        # labels). Keeping plasticity on during scan would dilute the
        # learned signal with random label-driven updates.
        for sub_phase, n_trials, validity, apply_plast in (
            ("learning", int(n_trials_learning), 1.0,                   True),
            ("scan",     int(n_trials_scan),     float(validity_scan),  False),
        ):
            for k in range(n_trials):
                cue_id = int(np_rng.integers(0, 2))
                probe_deg = _pick_scan_probe(cue_id, cue_mapping, validity, np_rng)
                is_expected = (
                    float(probe_deg) == float(cue_mapping[cue_id])
                )

                # Fix J: L23E modulator = ema_exp[cue] − ema_unexp[cue],
                # broadcast to [1, n_l23_e] for the batch axis.
                l23e_mod = (
                    l23e_ema[(cue_id, True)] - l23e_ema[(cue_id, False)]
                ).unsqueeze(0)

                info = run_kok_trial(
                    net, cfg,
                    cue_id=cue_id, probe_orientation_deg=probe_deg,
                    timing=timing, rule=rule,
                    noise_std=float(noise_std), device=str(cfg.device),
                    apply_plasticity=apply_plast,
                    l23e_modulator=l23e_mod,
                    lr_mh_scale=float(lr_mh_scale),
                    lr_qm_scale=float(lr_qm_scale),
                    weight_clamp=float(weight_clamp),
                    weight_clamp_inh=float(weight_clamp_inh),
                    disable_fix_j_mh_exc=bool(disable_fix_j_mh_exc),
                )

                # Task #74 β-mechanism Step 3: apply the Step-2-validated
                # three-factor rule on l23_e.W_q_gain, using the same
                # ``matched = is_expected`` scalar modulator and the
                # per-unit probe-1 r_l23 vector returned by run_kok_trial.
                # Gated by apply_plast so the scan sub-phase freezes
                # W_q_gain exactly like every other plastic weight.
                if enable_w_q_gain_rule and apply_plast:
                    apply_w_q_gain_update(
                        net,
                        cue_id=int(cue_id),
                        r_l23e_vec=info["r_l23_probe1_mean_vec"],
                        matched=bool(is_expected),
                        lr=float(w_q_gain_lr),
                        gain_min=float(w_q_gain_clamp[0]),
                        gain_max=float(w_q_gain_clamp[1]),
                    )

                # Fix J: update the running r_l23 EMA for this trial's
                # (cue, is_expected) condition. Updated in BOTH sub-phases
                # (unexpected probes only appear in scan). The EMA is a
                # read-only conditioning signal — it never mutates a
                # Parameter, so plasticity gating is unchanged.
                r_l23_vec = info["r_l23_probe1_mean_vec"].squeeze(0)  # [n_l23_e]
                key = (cue_id, bool(is_expected))
                l23e_ema[key] = (1.0 - alpha) * l23e_ema[key] + alpha * r_l23_vec

                if k % max(int(log_every), 1) == 0 or k == n_trials - 1:
                    m = TrainStepMetrics(
                        phase_name=sub_phase, trial=k, cue_id=cue_id,
                        probe_orientation_deg=float(probe_deg),
                        is_expected=bool(is_expected),
                        dw_qm_abs_mean=float(info["dw_qm_abs_mean"].item()),
                        dw_mh_exc_abs_mean=float(
                            info["dw_mh_exc_abs_mean"].item()
                        ),
                        dw_mh_inh_abs_mean=float(
                            info["dw_mh_inh_abs_mean"].item()
                        ),
                        probe1_r_l23_mean=float(info["probe1_r_l23_mean"].item()),
                        wall_time_s=time.monotonic() - t_start,
                    )
                    history.steps.append(m)
                    if metrics_fh is not None:
                        metrics_fh.write(json.dumps(m.__dict__) + "\n")
                        metrics_fh.flush()
                    # Task #74: stdout progress line for long runs.
                    elapsed_min = (time.monotonic() - t_start) / 60.0
                    qm_norm = float(
                        net.context_memory.W_qm_task.detach().norm().item()
                    )
                    mh_exc_norm = float(
                        net.context_memory.W_mh_task_exc.detach().norm().item()
                    )
                    mh_inh_norm = float(
                        net.context_memory.W_mh_task_inh.detach().norm().item()
                    )
                    print(
                        f"[train] phase={sub_phase} "
                        f"trial={global_trial}/{total_all} "
                        f"elapsed={elapsed_min:.1f}min "
                        f"W_qm={qm_norm:.4f} "
                        f"W_mh_exc={mh_exc_norm:.4f} "
                        f"W_mh_inh={mh_inh_norm:.4f}",
                        flush=True,
                    )
                global_trial += 1
    finally:
        if metrics_fh is not None:
            metrics_fh.close()

    sha_at_end = net.frozen_sensory_core_sha()
    if sha_at_end != sha_at_start:
        raise RuntimeError(
            "frozen sensory core SHA changed during Phase-3 Kok training — "
            "LGN/L4 mutated (forbidden)"
        )
    _assert_snapshot_unchanged(net, frozen_snaps)
    _assert_theta_unchanged(net, theta_snaps)  # Task #74 Fix A_homeo
    return history


def _save_checkpoint(
    net: V2Network, trial: int, out_path: Path, cue_mapping: dict[int, float],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Task #74 Fix A_homeo: record SHA-256 of the homeostasis θ buffers
    # (L23E, HE) as belt-and-braces evidence that θ didn't drift during
    # Phase-3 training. These are non-Parameter buffers that the
    # _assert_snapshot_unchanged invariant didn't originally cover.
    import hashlib
    theta_shas = {
        pop: hashlib.sha256(
            getattr(net, pop).homeostasis.theta.detach().cpu().numpy().tobytes(),
        ).hexdigest()
        for pop in ("l23_e", "h_e")
    }
    torch.save(
        {
            "step": int(trial),
            "state_dict": net.state_dict(),
            "phase": net.phase,
            "frozen_sha": net.frozen_sensory_core_sha(),
            "homeostasis_theta_sha256": theta_shas,
            "cue_mapping": {int(k): float(v) for k, v in cue_mapping.items()},
            # Task #74 β-Step-3: W_q_gain is a non-persistent buffer so it
            # is not part of net.state_dict(). Persist it explicitly so
            # Level-11 evaluators can reinstall the LEARNED gain.
            "W_q_gain": net.l23_e.W_q_gain.detach().cpu().clone(),
        },
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3 Kok task-learning driver.")
    p.add_argument("--phase2-checkpoint", type=Path, default=None,
                   help="Load a Phase-2 checkpoint as the starting net (optional).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-learning", type=int, default=30_000)
    p.add_argument("--n-trials-scan", type=int, default=10_000)
    p.add_argument("--validity-scan", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    # Task #74 Fix B + Fix C-v2:
    p.add_argument("--lr-mh-scale", type=float, default=1.0,
                   help="Fix B: outer multiplier on the W_mh_task_{exc,inh} lr. "
                        "Fix C-v2: default reset to 1.0 since the gain-modulation "
                        "redesign converts W_mh_task_inh into a softplus-bounded "
                        "gain signal; raise to 5-10 only if mid-run Δsvm signal "
                        "is weak.")
    p.add_argument("--lr-qm-scale", type=float, default=1.0,
                   help="Outer multiplier on the W_qm_task lr.")
    p.add_argument("--l23e-ema-alpha", type=float, default=0.01,
                   help="Fix J: EMA rate for per-(cue, is_expected) r_l23 buffers.")
    p.add_argument("--som-ema-alpha", type=float, default=None,
                   help="DEPRECATED (Fix J): alias for --l23e-ema-alpha. "
                        "Triggers a DeprecationWarning; still honored for "
                        "backward compat with older launch scripts.")
    p.add_argument("--weight-clamp", type=float, default=8.0,
                   help="Fix B: |W_mh_task_exc| per-element cap.")
    # Task #74 β-mechanism Step 3 (2026-04-23):
    p.add_argument("--disable-fix-j-mh-exc", action="store_true",
                   help="β-Step-3: zero out Fix J's W_mh_task_exc update. "
                        "Use with --enable-w-q-gain-rule to isolate the "
                        "W_q_gain path as the sole expectation mechanism.")
    p.add_argument("--enable-w-q-gain-rule", action="store_true",
                   help="β-Step-3: apply the Step-2-validated three-factor "
                        "rule on l23_e.W_q_gain during the learning sub-"
                        "phase. Buffer is frozen during scan.")
    p.add_argument("--w-q-gain-lr", type=float, default=1e-3,
                   help="β-Step-3: learning rate for the W_q_gain rule "
                        "(matches Step-2 toy default).")
    p.add_argument("--w-q-gain-min", type=float, default=0.1,
                   help="β-Step-3: post-update clamp floor on W_q_gain.")
    p.add_argument("--w-q-gain-max", type=float, default=1.0,
                   help="β-Step-3: post-update clamp ceiling on W_q_gain.")
    p.add_argument("--weight-clamp-inh", type=float, default=8.0,
                   help="DEPRECATED (Fix J): no-op. W_mh_task_inh is inert "
                        "under Fix J (never updated), so a per-element cap "
                        "on it has no effect. Kept as a CLI flag only to "
                        "avoid breaking scripts that pass it.")
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("checkpoints/v2/phase3_kok"),
    )
    p.add_argument(
        "--diagnostics-out", type=Path, default=None,
        help=(
            "If set, run the Task #74 3-metric suite on the final ckpt "
            "and write results to this JSON path."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    torch.manual_seed(int(args.seed))

    cfg = ModelConfig(seed=int(args.seed), device=args.device)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=int(args.seed), device=args.device)
    if args.phase2_checkpoint is not None:
        ckpt = torch.load(args.phase2_checkpoint, map_location=args.device,
                          weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
    net.set_phase("phase3_kok")

    cue_mapping = cue_mapping_from_seed(int(args.seed))
    out_path = args.out_dir / f"phase3_kok_s{int(args.seed)}.pt"
    metrics_path = out_path.parent / f"phase3_kok_s{int(args.seed)}_metrics.jsonl"
    run_phase3_kok_training(
        net=net,
        n_trials_learning=int(args.n_trials_learning),
        n_trials_scan=int(args.n_trials_scan),
        validity_scan=float(args.validity_scan),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        noise_std=float(args.noise_std),
        cue_mapping=cue_mapping,
        metrics_path=metrics_path,
        log_every=int(args.log_every),
        lr_mh_scale=float(args.lr_mh_scale),
        lr_qm_scale=float(args.lr_qm_scale),
        l23e_ema_alpha=float(args.l23e_ema_alpha),
        som_ema_alpha=(
            float(args.som_ema_alpha) if args.som_ema_alpha is not None else None
        ),
        weight_clamp=float(args.weight_clamp),
        weight_clamp_inh=float(args.weight_clamp_inh),
        disable_fix_j_mh_exc=bool(args.disable_fix_j_mh_exc),
        enable_w_q_gain_rule=bool(args.enable_w_q_gain_rule),
        w_q_gain_lr=float(args.w_q_gain_lr),
        w_q_gain_clamp=(float(args.w_q_gain_min), float(args.w_q_gain_max)),
    )
    _save_checkpoint(
        net, args.n_trials_learning + args.n_trials_scan,
        out_path, cue_mapping,
    )
    print(f"phase3_kok checkpoint written to {out_path}", flush=True)

    # Task #74 (Fix A_homeo): run the 3 standard diagnostics on the final
    # ckpt and write a single JSON next to the training log.
    if args.diagnostics_out is not None:
        from scripts.v2.task74_diagnostics import run_diagnostics
        print(
            f"[diagnostics] running 3-metric suite on {out_path}...",
            flush=True,
        )
        diag = run_diagnostics(
            out_path, paradigm="kok", seed=int(args.seed), device=args.device,
        )
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_out.write_text(json.dumps(diag, indent=2))
        cov_ent = diag.get("coverage", {}).get("entropy_nats")
        rm = diag.get("rule_magnitude", {})
        ra = diag.get("readout_alignment", {})
        print(
            f"[diagnostics] entropy_nats={cov_ent} "
            f"rule_ratio={rm.get('ratio')} "
            f"cos_mean_same={ra.get('mean_cos_same')} "
            f"→ {args.diagnostics_out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
