#!/usr/bin/env python3
"""Train one common task initialization and the requested task-activity arms.

All arms clone one seed-specific task pretrain and differ only in ``alpha`` for
``L=(1-alpha)*T+alpha*R/R_ref``. ``T`` combines normalized next-channel cross
entropy with noisy current-orientation population-vector precision; ``R`` is a
dimensionless conditional modeled-population activity proxy. The fixed-step
finals, not validation-selected checkpoints, are the experiment endpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import simple_net as simple  # noqa: E402
import tuned_emergence_lib as tuned  # noqa: E402


N = 36
STEP_DEG = 5.0
ALPHAS = (0.004, 0.5)
TRAINING_COMPATIBILITY_VERSION = "shared_divisive_som_population_activity_v8"
FIXED_CANONICAL_VIP_MOTIF_GAINS = {
    "w_vd": 0.1,
    "w_sv": 0.1,
}
MODEL_CONFIG = {
    "hidden": 64,
    "ff_sigma_channels": 1.1,
    "ff_gain": 1.6,
    "decoder_gain": 8.0,
    "readout": "population_vector",
    "population_normalize": True,
    "pred_inhib_strength": 0.04,
    "pred_inhib_sigma_channels": 2.0,
    "som_input_sigma_channels": 2.0,
    "som_output_sigma_channels": math.sqrt(2.0) * 2.0,
    "vip_som_sigma_channels": 2.0,
    "pred_feature_supp_strength": 0.0,
    "rate_saturation_r_max": 0.0,
    "rate_saturation_r_half": 1.0,
    "adapt_strength": 0.0,
    "adapt_decay": 0.85,
    "adapt_sigma_channels": 1.0,
    "local_comp_strength": 0.0,
    "local_comp_sigma_channels": 2.0,
    "local_comp_power": 1.0,
    "local_comp_mode": "divisive",
    "local_comp_trainable": False,
    "recurrent_cell": "rnn_tanh",
    "m_fixed_parameterization": tuned.M_FIXED_MODE_SOFTPLUS_RAW,
    "fixed_intrinsic_rheobases": True,
    "model_architecture_version": tuned.MODEL_ARCHITECTURE_VERSION,
    "training_compatibility_version": TRAINING_COMPATIBILITY_VERSION,
    "fixed_canonical_vip_motif_gains": FIXED_CANONICAL_VIP_MOTIF_GAINS,
}
MODELED_ACTIVITY_WEIGHTS = {
    "final_e": 5.0 / 6.0,
    "pv": 37.0 / 480.0,
    "som": 1.0 / 20.0,
    "vip": 19.0 / 480.0,
}
CONSTRAINED_DUAL_STEP_SIZE = 1e-3


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    simple.device = selected
    simple.prefs = torch.arange(N, device=selected).float() * STEP_DEG
    tuned.device = selected
    return selected


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def centered_feedback_property_check(device: torch.device) -> dict[str, float]:
    raw = torch.linspace(-2.0, 2.0, 4 * N, device=device).reshape(4, N)
    shifts = torch.tensor((3.25, -2.5, 0.125, 1.75), device=device)[:, None]
    baseline = tuned.predictive_feedback_evidence(raw, True)
    shifted = tuned.predictive_feedback_evidence(raw + shifts, True)
    max_shift_difference = float((baseline - shifted).abs().max().item())
    common_shift = torch.zeros((), device=device, requires_grad=True)
    energy = tuned.predictive_feedback_evidence(raw + common_shift, True).mean()
    common_shift_gradient = float(
        torch.autograd.grad(energy, common_shift)[0].item()
    )
    if max_shift_difference > 1e-6 or abs(common_shift_gradient) > 1e-6:
        raise RuntimeError(
            "centered feedback failed scalar-shift invariance or gradient check"
        )
    return {
        "max_scalar_shift_feedback_difference": max_shift_difference,
        "common_shift_energy_gradient": common_shift_gradient,
    }


def posterior_feedback_property_check(
    device: torch.device,
    alphas: list[float],
    feedback_mode: str,
) -> dict:
    """Verify the shared posterior-family feedback transform."""

    raw = torch.linspace(-2.0, 2.0, 4 * N, device=device).reshape(4, N)
    raw_before = raw.clone()
    shifts = torch.tensor((3.25, -2.5, 0.125, 1.75), device=device)[:, None]
    baseline = tuned.predictive_feedback_evidence(
        raw,
        feedback_mode=feedback_mode,
    )
    shifted = tuned.predictive_feedback_evidence(
        raw + shifts,
        feedback_mode=feedback_mode,
    )
    uniform = tuned.predictive_feedback_evidence(
        torch.full((4, N), 2.75, device=device),
        feedback_mode=feedback_mode,
    )
    max_shift_difference = float((baseline - shifted).abs().max().item())
    finite = bool(torch.isfinite(baseline).all().item())
    minimum = float(baseline.min().item())
    maximum = float(baseline.max().item())
    raw_logits_unchanged = torch.equal(raw, raw_before)
    arm_modes = [feedback_mode for _ in alphas]
    identical_mode_all_arms = (
        bool(arm_modes)
        and len(set(arm_modes)) == 1
        and arm_modes[0] == feedback_mode
    )
    if feedback_mode == tuned.FEEDBACK_MODE_POSTERIOR:
        uniform_target = torch.full_like(uniform, 1.0 / float(N))
        uniform_ok = bool(torch.allclose(uniform, uniform_target, atol=1e-7))
        row_sum = baseline.sum(dim=-1)
        row_sum_ok = bool(
            torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6)
        )
        maximum_ok = maximum <= 1.0
        theoretical_maximum = 1.0
    elif feedback_mode == tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS:
        uniform_ok = torch.count_nonzero(uniform).item() == 0
        row_sum_ok = True
        maximum_ok = maximum <= float(N - 1)
        theoretical_maximum = float(N - 1)
    else:
        raise RuntimeError(f"unexpected posterior feedback mode {feedback_mode!r}")
    if (
        max_shift_difference >= 1e-6
        or not uniform_ok
        or not row_sum_ok
        or not finite
        or minimum < 0.0
        or not maximum_ok
        or not raw_logits_unchanged
        or not identical_mode_all_arms
    ):
        raise RuntimeError("posterior feedback property check failed")
    return {
        "max_scalar_shift_feedback_difference": max_shift_difference,
        "uniform_logits_ok": uniform_ok,
        "row_sum_ok": row_sum_ok,
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "theoretical_maximum": theoretical_maximum,
        "raw_logits_unchanged": raw_logits_unchanged,
        "identical_mode_all_arms": identical_mode_all_arms,
        "arm_feedback_modes": arm_modes,
    }


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def restore_generator_state(
    generator: torch.Generator,
    state: torch.Tensor,
) -> None:
    """Restore a generator from a one-dimensional uint8 state tensor."""

    if not isinstance(state, torch.Tensor):
        raise TypeError("generator state must be a torch.Tensor")
    if state.dtype != torch.uint8:
        raise TypeError("generator state tensor must have dtype torch.uint8")
    if state.ndim != 1:
        raise ValueError("generator state tensor must be one-dimensional")
    generator.set_state(state.detach().cpu().contiguous())


def momentum_batch(
    batch: int,
    sequence_length: int,
    device: torch.device,
    generator: torch.Generator,
    *,
    p_stay: float = 0.9,
    vmax: int = 4,
    mismatch_prob: float = 0.0,
    mismatch_stats: dict | None = None,
    return_mismatch_mask: bool = False,
):
    """Generate ordinary momentum sequences using one isolated RNG stream.

    Phase 2b (DESIGN_PHASE2B sections 3.2 and 6): with probability
    ``mismatch_prob`` per velocity transition t (1 <= t <= sequence_length-2)
    and ONLY when |velocity[t-1]| >= 2, the applied step ``velocity[t]`` is
    halted to zero for that single step (the visual-flow-halt analogue,
    Vasilevskaya 2023 / Widmer 2022). The halt is applied AFTER the velocity
    recursion is built and BEFORE the cumulative sum, so the pre-halt velocity
    resumes automatically at t+1. The extra RNG draw is guarded behind
    ``mismatch_prob > 0.0`` and placed after every existing draw, so at 0.0
    the generator consumes bit-identical draws to the unpatched code.
    ``mismatch_stats`` (if given) accumulates realized event / eligible /
    transition counts in place; note they are NOT persisted in checkpoints,
    so a resumed run undercounts (our runs are single-shot).
    ``return_mismatch_mask`` additionally returns the [batch, S-2] halt mask
    (columns j correspond to halted applied-step index t = j+1), or None
    when ``mismatch_prob == 0.0``.
    """

    acceleration_values = torch.tensor((-1, 0, 1), device=device)
    acceleration_index = torch.empty(
        batch, sequence_length, dtype=torch.long, device=device
    )
    acceleration_index[:, 0] = torch.randint(
        0, 3, (batch,), device=device, generator=generator
    )
    for time_index in range(1, sequence_length):
        stay = torch.rand(batch, device=device, generator=generator) < p_stay
        replacement = torch.randint(
            0, 3, (batch,), device=device, generator=generator
        )
        acceleration_index[:, time_index] = torch.where(
            stay, acceleration_index[:, time_index - 1], replacement
        )
    accelerations = acceleration_values[acceleration_index]
    velocity = torch.empty(batch, sequence_length, dtype=torch.long, device=device)
    velocity[:, 0] = torch.randint(
        -vmax, vmax + 1, (batch,), device=device, generator=generator
    )
    for time_index in range(1, sequence_length):
        velocity[:, time_index] = (
            velocity[:, time_index - 1] + accelerations[:, time_index - 1]
        ).clamp(-vmax, vmax)
    initial = torch.randint(
        0, N, (batch, 1), dtype=torch.long, device=device, generator=generator
    )
    halt = None
    applied_velocity = velocity
    if mismatch_prob > 0.0 and sequence_length > 2:
        eligible = velocity[:, :-2].abs() >= 2  # |v_{t-1}| for t = 1..S-2
        draws = torch.rand(
            batch, sequence_length - 2, device=device, generator=generator
        )
        halt = eligible & (draws < mismatch_prob)
        applied_velocity = velocity.clone()
        applied_velocity[:, 1:-1] = applied_velocity[:, 1:-1].masked_fill(
            halt, 0
        )
        if mismatch_stats is not None:
            mismatch_stats["events"] = (
                mismatch_stats.get("events", 0) + int(halt.sum().item())
            )
            mismatch_stats["eligible"] = (
                mismatch_stats.get("eligible", 0) + int(eligible.sum().item())
            )
            mismatch_stats["transitions"] = (
                mismatch_stats.get("transitions", 0) + int(eligible.numel())
            )
    offsets = torch.cat(
        (
            torch.zeros(batch, 1, dtype=torch.long, device=device),
            torch.cumsum(applied_velocity[:, :-1], dim=1),
        ),
        dim=1,
    )
    channels = (initial + offsets) % N
    if return_mismatch_mask:
        return channels.float() * STEP_DEG, channels, halt
    return channels.float() * STEP_DEG, channels


def mismatch_accounting(prob: float, stats: dict) -> dict:
    """Realized halt-rate accounting (DESIGN_PHASE2B section 6.4) — measured."""

    events = stats.get("events", 0)
    eligible = stats.get("eligible", 0)
    transitions = stats.get("transitions", 0)
    return {
        "mismatch_prob": prob,
        "realized_events": events,
        "eligible_transitions": eligible,
        "total_transitions": transitions,
        "realized_fraction_of_eligible": (
            events / eligible if eligible else 0.0
        ),
        "realized_fraction_of_all": (
            events / transitions if transitions else 0.0
        ),
        "eligibility_fraction": (
            eligible / transitions if transitions else 0.0
        ),
    }


def pv_scalar_from_pre_pv(
    net: tuned.SimpleTunedNet,
    pre_pv_rate: torch.Tensor,
) -> torch.Tensor:
    """Return the actual broad PV gain scalar ``w_pv * mean(pre_pv_rate)``."""

    w_pv = net.circuit_gains()[tuned.CIRC_INDEX["w_pv"]]
    return w_pv * pre_pv_rate.mean(dim=-1, keepdim=True)


def modeled_population_activity_components(
    final_e: torch.Tensor,
    som: torch.Tensor,
    vip: torch.Tensor,
    pv_scalar: torch.Tensor,
    som_gain: torch.Tensor | None = None,
    exc_feedback_work: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return the conditional modeled-population activity components.

    The scalar is dimensionless and conditional on modeled class means:
    ``R=(5/6)E + (37/480)PV + (1/20)SOM + (19/480)VIP``, where ``E`` is the
    raw E class mean ``final_e.mean()``. It is not a spike, ATP, or
    metabolic-energy estimate. ``som_gain`` and
    ``exc_feedback_work`` are optional reporting-only internals and do not
    enter ``R``.
    """

    final_e_mean = final_e.mean()
    som_mean = som.mean()
    vip_mean = vip.mean()
    pv_mean = pv_scalar.mean()
    if som_gain is None:
        som_gain_mean = final_e_mean.new_zeros(())
    else:
        som_gain_mean = som_gain.mean()
    if exc_feedback_work is None:
        exc_feedback_work_mean = final_e_mean.new_zeros(())
    else:
        exc_feedback_work_mean = exc_feedback_work.mean()
    numerator = (
        MODELED_ACTIVITY_WEIGHTS["final_e"] * final_e_mean
        + MODELED_ACTIVITY_WEIGHTS["pv"] * pv_mean
        + MODELED_ACTIVITY_WEIGHTS["som"] * som_mean
        + MODELED_ACTIVITY_WEIGHTS["vip"] * vip_mean
    )
    return {
        "modeled_population_activity_final_e": final_e_mean,
        "modeled_population_activity_pv": pv_mean,
        "modeled_population_activity_som": som_mean,
        "modeled_population_activity_vip": vip_mean,
        "modeled_population_activity_som_gain_report": som_gain_mean,
        "modeled_population_activity_exc_feedback_report": exc_feedback_work_mean,
        "modeled_population_activity_numerator": numerator,
    }


@torch.no_grad()
def reference_values(net: tuned.SimpleTunedNet, device: torch.device) -> dict[str, float]:
    channels = torch.arange(N, device=device, dtype=torch.long)
    theta = channels.float() * STEP_DEG
    l4 = tuned.l4_code(theta)
    zeros = torch.zeros(N, N, device=device)
    rates, internals = net.l23(
        l4,
        zeros,
        torch.zeros_like(zeros),
        return_internals=True,
    )
    (
        som,
        vip,
        som_gain,
        pre_pv_rate,
        post_pv_rate,
        exc_feedback_work,
        _,
        _,
    ) = internals
    pv_scalar = pv_scalar_from_pre_pv(net, pre_pv_rate)
    activity = modeled_population_activity_components(
        rates,
        som,
        vip,
        pv_scalar,
        som_gain,
        exc_feedback_work,
    )
    r_ref = activity["modeled_population_activity_numerator"]
    maxima = rates.max(dim=1).values.sort().values
    a_ref = 0.5 * (maxima[N // 2 - 1] + maxima[N // 2])
    if not torch.isfinite(r_ref) or not r_ref > 0:
        raise RuntimeError("R_ref must be finite and positive")
    if not torch.isfinite(a_ref) or not a_ref > 0:
        raise RuntimeError("A_ref must be finite and positive")
    return {
        "R_ref": float(r_ref.item()),
        "modeled_population_activity_ref": float(r_ref.item()),
        "A_ref": float(a_ref.item()),
        "sigma_train": float((0.25 * a_ref).item()),
        "modeled_population_activity_ref_final_e": float(
            activity["modeled_population_activity_final_e"].item()
        ),
        "modeled_population_activity_ref_pv": float(
            activity["modeled_population_activity_pv"].item()
        ),
        "modeled_population_activity_ref_som": float(
            activity["modeled_population_activity_som"].item()
        ),
        "modeled_population_activity_ref_vip": float(
            activity["modeled_population_activity_vip"].item()
        ),
        "modeled_population_activity_ref_som_gain_report": float(
            activity["modeled_population_activity_som_gain_report"].item()
        ),
        "modeled_population_activity_ref_exc_feedback_report": float(
            activity["modeled_population_activity_exc_feedback_report"].item()
        ),
        # Deprecated aliases: the values are modeled population activity,
        # not work, ATP, or metabolic energy.
        "activity_work_ref": float(r_ref.item()),
    }


def confidence_weighted_current_orientation_ce(
    net: tuned.SimpleTunedNet,
    rates: torch.Tensor,
    channels: torch.Tensor,
    noise_generator: torch.Generator,
    references: dict[str, float],
    current_decoder_noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return audited confidence-weighted circular current-orientation CE.

    The resultant uses doubled orientation angles over the 36-channel basis.
    Confidence scales only the decoded direction logits; decoder gain is
    detached so this term does not train by simply inflating readout gain.
    """

    if current_decoder_noise is None:
        noise = torch.randn(
            rates.shape,
            device=rates.device,
            dtype=rates.dtype,
            generator=noise_generator,
        ) * references["sigma_train"]
    else:
        if (
            current_decoder_noise.shape != rates.shape
            or current_decoder_noise.device != rates.device
            or current_decoder_noise.dtype != rates.dtype
        ):
            raise ValueError(
                "current_decoder_noise must match the rates shape, device, and dtype"
            )
        noise = current_decoder_noise
    activity = F.relu(rates + noise)
    angles = 2.0 * math.pi * torch.arange(N, device=rates.device) / float(N)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    x_component = (activity * cos_angles).sum(dim=-1)
    y_component = (activity * sin_angles).sum(dim=-1)
    magnitude = torch.sqrt(x_component.square() + y_component.square())
    eps = rates.new_tensor(1e-8 * N * references["R_ref"])
    direction = (
        x_component.unsqueeze(-1) * cos_angles
        + y_component.unsqueeze(-1) * sin_angles
    ) / (magnitude.unsqueeze(-1) + eps)
    sigma_vec = rates.new_tensor(references["sigma_train"] * math.sqrt(N / 2.0))
    confidence = magnitude / (magnitude + sigma_vec + eps)
    logits = (
        F.softplus(net.decoder_gain_raw).detach()
        * confidence.unsqueeze(-1)
        * direction
    )
    current_ce = F.cross_entropy(
        logits.reshape(-1, N),
        channels.reshape(-1),
    )
    return {
        "current_ce": current_ce,
        "current_ce_normalized": current_ce / math.log(N),
        "current_confidence": confidence.mean(),
        "current_resultant_magnitude": magnitude.mean(),
    }


def task_activity_losses(
    net: tuned.SimpleTunedNet,
    theta: torch.Tensor,
    channels: torch.Tensor,
    noise_generator: torch.Generator,
    references: dict[str, float],
    center_feedback: bool = False,
    feedback_mode: str | None = None,
    current_decoder_noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute task and normalized modeled-population activity pressure.

    Parameters
    ----------
    theta:
        Degree-valued orientation bins with shape ``[B,S]``. One channel is
        ``5`` nominal degrees.
    channels:
        Matching zero-based integer channel labels with shape ``[B,S]``.
    noise_generator:
        Arm-local generator used for exactly one ``[B,S,36]`` Gaussian draw
        unless ``current_decoder_noise`` supplies that draw explicitly.
    current_decoder_noise:
        Optional fixed ``[B,S,36]`` noise tensor for paired evaluations.
    references:
        Positive scalar activity references in arbitrary units, including
        ``R_ref`` and ``sigma_train``.

    Returns
    -------
    dict[str, torch.Tensor]
        Scalar dimensionless tensors. With logits ``[B,S,36]`` and raw L2/3
        rates ``r[B,S,36]`` in arbitrary activity units, the returned terms are
        ``task=0.5*next_ce/log(36)+0.5*current_ce/log(36)`` and
        ``modeled_population_activity=R/R_ref`` where the E term in ``R`` is
        the raw E class mean ``r.mean()``. This is a dimensionless conditional
        modeled-population activity pressure, not spikes, ATP, or metabolic
        energy.
        Feedback computed after time ``t`` affects L2/3 only at ``t+1``; the
        first response has zero feedback state.
    """

    predictions, rates, internals = tuned.forward_seq_tuned(
        net,
        theta,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
        return_internals=True,
    )
    (
        som_seq,
        vip_seq,
        som_gain_seq,
        pre_pv_seq,
        post_pv_seq,
        exc_feedback_work_seq,
        _,
        _,
    ) = internals
    next_ce = F.cross_entropy(
        predictions[:, :-1, :].reshape(-1, N), channels[:, 1:].reshape(-1)
    )
    current = confidence_weighted_current_orientation_ce(
        net,
        rates,
        channels,
        noise_generator,
        references,
        current_decoder_noise=current_decoder_noise,
    )
    task = 0.5 * next_ce / math.log(N) + 0.5 * current["current_ce"] / math.log(N)
    pv_scalar_seq = pv_scalar_from_pre_pv(net, pre_pv_seq)
    activity = modeled_population_activity_components(
        rates,
        som_seq,
        vip_seq,
        pv_scalar_seq,
        som_gain_seq,
        exc_feedback_work_seq,
    )
    modeled_activity = (
        activity["modeled_population_activity_numerator"] / references["R_ref"]
    )
    return {
        "next_ce": next_ce,
        **current,
        "task": task,
        "modeled_population_activity": modeled_activity,
        # Deprecated alias retained for callers that still index "energy".
        # This is R/R_ref modeled population activity, not metabolic energy.
        "energy": modeled_activity,
        **activity,
    }


def task_energy_losses(*args, **kwargs) -> dict[str, torch.Tensor]:
    """Deprecated alias for ``task_activity_losses``.

    Returned ``energy`` is itself a deprecated alias for normalized
    modeled-population activity, not metabolic energy.
    """

    return task_activity_losses(*args, **kwargs)


def paired_constrained_task_losses(
    candidate_net: tuned.SimpleTunedNet,
    frozen_reference_net: tuned.SimpleTunedNet,
    theta: torch.Tensor,
    channels: torch.Tensor,
    noise_generator: torch.Generator,
    references: dict[str, float],
    center_feedback: bool = False,
    feedback_mode: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Evaluate candidate and frozen reference on one batch and noise tensor."""

    parameter = next(candidate_net.parameters())
    current_decoder_noise = torch.randn(
        (*theta.shape, N),
        device=theta.device,
        dtype=parameter.dtype,
        generator=noise_generator,
    ) * references["sigma_train"]
    candidate_losses = task_activity_losses(
        candidate_net,
        theta,
        channels,
        noise_generator,
        references,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
        current_decoder_noise=current_decoder_noise,
    )
    with torch.no_grad():
        reference_losses = task_activity_losses(
            frozen_reference_net,
            theta,
            channels,
            noise_generator,
            references,
            center_feedback=center_feedback,
            feedback_mode=feedback_mode,
            current_decoder_noise=current_decoder_noise,
        )
    reference_losses = {
        name: value.detach() for name, value in reference_losses.items()
    }
    return candidate_losses, reference_losses, current_decoder_noise


def constrained_objective_terms(
    candidate_losses: dict[str, torch.Tensor],
    reference_losses: dict[str, torch.Tensor],
    lambda_next: torch.Tensor,
    lambda_current: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the two constraints and their activity Lagrangian."""

    candidate_next = candidate_losses["next_ce"] / math.log(N)
    reference_next = reference_losses["next_ce"].detach() / math.log(N)
    candidate_current = candidate_losses["current_ce_normalized"]
    reference_current = reference_losses["current_ce_normalized"].detach()
    constraint_next = candidate_next - reference_next
    constraint_current = candidate_current - reference_current
    objective = (
        candidate_losses["modeled_population_activity"]
        + lambda_next * constraint_next
        + lambda_current * constraint_current
    )
    return {
        "objective": objective,
        "constraint_next": constraint_next,
        "constraint_current": constraint_current,
        "candidate_next": candidate_next,
        "reference_next": reference_next,
        "candidate_current": candidate_current,
        "reference_current": reference_current,
    }


def projected_dual_ascent(
    lambda_next: torch.Tensor,
    lambda_current: torch.Tensor,
    constraint_next: torch.Tensor,
    constraint_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply separate nonnegative dual updates with the fixed step size."""

    with torch.no_grad():
        updated_next = torch.clamp_min(
            lambda_next
            + CONSTRAINED_DUAL_STEP_SIZE * constraint_next.detach(),
            0.0,
        )
        updated_current = torch.clamp_min(
            lambda_current
            + CONSTRAINED_DUAL_STEP_SIZE * constraint_current.detach(),
            0.0,
        )
    return updated_next, updated_current


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class EventLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")

    def write(self, event: dict) -> None:
        row = {"wall_time": time.time(), **event}
        text = json.dumps(row, sort_keys=True)
        print(text, flush=True)
        self.stream.write(text + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def set_pretrain_parameter_policy(net: tuned.SimpleTunedNet) -> list[torch.nn.Parameter]:
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    parameters = list(net.gru.parameters()) + list(net.W_fb.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def set_axis_parameter_policy(
    net: tuned.SimpleTunedNet, freeze_local_comp: bool = False
) -> list[torch.nn.Parameter]:
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    parameters = list(net.gru.parameters()) + list(net.W_fb.parameters())
    parameters.append(net.w_sf_fixed)
    # Shared anatomy is initialized/settled, not learned during common
    # pretraining or alpha arms. Axis fitting trains only the tanh-RNN, W_fb,
    # and the existing nonnegative prediction-to-SOM coupling w_sf_fixed.
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def checkpoint_payload(
    *,
    stage: str,
    seed: int,
    step: int,
    target_steps: int,
    net: tuned.SimpleTunedNet,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    noise_generator: torch.Generator,
    references: dict[str, float],
    alpha: float | None,
    task_weight: float | None = None,
    freeze_local_comp: bool = False,
    center_feedback: bool = False,
    feedback_mode: str | None = None,
    mismatch_prob: float = 0.0,
) -> dict:
    resolved_feedback_mode = tuned.resolve_feedback_mode(
        center_feedback,
        feedback_mode,
    )
    return {
        "stage": stage,
        "mismatch_prob": mismatch_prob,
        "seed": seed,
        "alpha": alpha,
        "task_weight": task_weight,
        "step": step,
        "target_steps": target_steps,
        "state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "data_generator_state": data_generator.get_state(),
        "noise_generator_state": noise_generator.get_state(),
        "tuned_net_config": MODEL_CONFIG,
        "model_architecture_version": tuned.MODEL_ARCHITECTURE_VERSION,
        "training_compatibility_version": TRAINING_COMPATIBILITY_VERSION,
        "fixed_canonical_vip_motif_gains": FIXED_CANONICAL_VIP_MOTIF_GAINS,
        "references": references,
        "freeze_local_comp": freeze_local_comp,
        "center_feedback": (
            resolved_feedback_mode == tuned.FEEDBACK_MODE_CENTERED
        ),
        "feedback_mode": resolved_feedback_mode,
    }


def run_pretrain(
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    event_log: EventLog,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    seed_everything(args.seed)
    net = tuned.build_tuned_from_config(MODEL_CONFIG).to(device)
    enforce_fixed_vip_motif(net)
    assert_fixed_vip_motif(net)
    references = reference_values(net, device)
    net.ref_rate.fill_(references["R_ref"])
    optimizer = torch.optim.Adam(
        set_pretrain_parameter_policy(net), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )
    data_generator = make_generator(device, 200000 + args.seed)
    noise_generator = make_generator(device, 300000 + args.seed)
    mismatch_stats: dict[str, int] = {
        "events": 0,
        "eligible": 0,
        "transitions": 0,
    }
    latest_path = run_dir / "common_pretrain_latest.pt"
    final_path = run_dir / "common_pretrain_final.pt"
    start_step = 1
    resume_path = final_path if final_path.exists() else latest_path
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device)
        saved_feedback_mode = tuned.resolve_feedback_mode(
            bool(saved.get("center_feedback", False)),
            saved.get("feedback_mode"),
        )
        if (
            saved["target_steps"] != args.pretrain_steps
            or saved["seed"] != args.seed
            or saved_feedback_mode != args.feedback_mode
            or float(saved.get("mismatch_prob", 0.0)) != args.mismatch_prob
            or saved.get("model_architecture_version")
            != tuned.MODEL_ARCHITECTURE_VERSION
            or saved.get("training_compatibility_version")
            != TRAINING_COMPATIBILITY_VERSION
        ):
            raise RuntimeError("pretrain checkpoint metadata does not match this run")
        net.load_state_dict(saved["state_dict"])
        assert_fixed_vip_motif(net)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        restore_generator_state(data_generator, saved["data_generator_state"])
        restore_generator_state(noise_generator, saved["noise_generator_state"])
        references = saved["references"]
        start_step = int(saved["step"]) + 1
        event_log.write({"event": "pretrain_resume", "step": start_step - 1})
    net.train()
    for step in range(start_step, args.pretrain_steps + 1):
        theta, channels = momentum_batch(
            args.batch,
            args.sequence_length,
            device,
            data_generator,
            mismatch_prob=args.mismatch_prob,
            mismatch_stats=mismatch_stats,
        )
        losses = task_activity_losses(
            net,
            theta,
            channels,
            noise_generator,
            references,
            center_feedback=args.center_feedback,
            feedback_mode=args.feedback_mode,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["task"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            set_pretrain_parameter_policy(net), args.clip
        )
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.pretrain_steps:
            event_log.write(
                {
                    "event": "pretrain_step",
                    "step": step,
                    "task": float(losses["task"].item()),
                    "next_ce": float(losses["next_ce"].item()),
                    "current_ce": float(losses["current_ce"].item()),
                    "current_ce_normalized": float(
                        losses["current_ce_normalized"].item()
                    ),
                    "current_confidence": float(
                        losses["current_confidence"].item()
                    ),
                    "current_resultant_magnitude": float(
                        losses["current_resultant_magnitude"].item()
                    ),
                    "modeled_population_activity_descriptive": float(
                        losses["modeled_population_activity"].item()
                    ),
                    "modeled_population_activity_final_e": float(
                        losses["modeled_population_activity_final_e"].item()
                    ),
                    "modeled_population_activity_pv": float(
                        losses["modeled_population_activity_pv"].item()
                    ),
                    "modeled_population_activity_som": float(
                        losses["modeled_population_activity_som"].item()
                    ),
                    "modeled_population_activity_vip": float(
                        losses["modeled_population_activity_vip"].item()
                    ),
                    "modeled_population_activity_som_gain_report": float(
                        losses[
                            "modeled_population_activity_som_gain_report"
                        ].item()
                    ),
                    "modeled_population_activity_exc_feedback_report": float(
                        losses[
                            "modeled_population_activity_exc_feedback_report"
                        ].item()
                    ),
                    "modeled_population_activity_numerator": float(
                        losses["modeled_population_activity_numerator"].item()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "mismatch_prob": args.mismatch_prob,
                    "mismatch_events": mismatch_stats["events"],
                    "mismatch_eligible": mismatch_stats["eligible"],
                    "mismatch_transitions": mismatch_stats["transitions"],
                }
            )
        if step % args.checkpoint_every == 0 or step == args.pretrain_steps:
            payload = checkpoint_payload(
                stage="common_pretrain",
                seed=args.seed,
                step=step,
                target_steps=args.pretrain_steps,
                net=net,
                optimizer=optimizer,
                data_generator=data_generator,
                noise_generator=noise_generator,
                references=references,
                alpha=None,
                center_feedback=args.center_feedback,
                feedback_mode=args.feedback_mode,
                mismatch_prob=args.mismatch_prob,
            )
            atomic_torch_save(payload, latest_path)
            if step == args.pretrain_steps:
                atomic_torch_save(payload, final_path)
    common_state = copy.deepcopy(net.state_dict())
    event_log.write(
        {
            "event": "pretrain_complete",
            "state_sha256": state_sha256(common_state),
            "references": references,
            "mismatch_accounting": mismatch_accounting(
                args.mismatch_prob, mismatch_stats
            ),
        }
    )
    return common_state, references


def alpha_tag(alpha: float | str) -> str:
    decimal = Decimal(str(alpha)).normalize()
    text = format(decimal, "f")
    if "." not in text:
        text = f"{text}.0"
    if text.startswith("."):
        text = f"0{text}"
    elif text.startswith("-."):
        text = text.replace("-.", "-0.", 1)
    return text


def alpha_slug(alpha: float | str) -> str:
    return alpha_tag(alpha).replace(".", "p")


def validate_unique_alpha_slugs(alphas: list[float]) -> None:
    slugs = [alpha_slug(alpha) for alpha in alphas]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"alphas produce duplicate checkpoint slugs: {slugs}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def optional_tensor_sha256(tensor: torch.Tensor | None) -> str | None:
    return None if tensor is None else tensor_sha256(tensor)


def optional_tensor_equal(
    left: torch.Tensor | None, right: torch.Tensor | None
) -> bool:
    if left is None or right is None:
        return left is right
    return torch.equal(left, right)


def fixed_vip_motif_indices() -> torch.Tensor:
    return torch.tensor(
        [
            tuned.CIRC_INDEX[name]
            for name in FIXED_CANONICAL_VIP_MOTIF_GAINS
        ],
        dtype=torch.long,
        device=tuned.device,
    )


def fixed_vip_motif_raw_values(net: tuned.SimpleTunedNet) -> torch.Tensor:
    return torch.tensor(
        [
            tuned.softplus_inverse(value)
            for value in FIXED_CANONICAL_VIP_MOTIF_GAINS.values()
        ],
        dtype=net.circ_raw.dtype,
        device=net.circ_raw.device,
    )


def enforce_fixed_vip_motif(net: tuned.SimpleTunedNet) -> None:
    with torch.no_grad():
        indices = fixed_vip_motif_indices().to(net.circ_raw.device)
        net.circ_raw[indices] = fixed_vip_motif_raw_values(net)


def assert_fixed_vip_motif(net: tuned.SimpleTunedNet) -> None:
    indices = fixed_vip_motif_indices().to(net.circ_raw.device)
    expected = fixed_vip_motif_raw_values(net)
    current = net.circ_raw.detach().index_select(0, indices)
    if not torch.equal(current, expected):
        raise RuntimeError("fixed canonical VIP motif gains changed")


def mask_fixed_vip_motif_grad(net: tuned.SimpleTunedNet) -> None:
    if net.circ_raw.grad is None:
        return
    indices = fixed_vip_motif_indices().to(net.circ_raw.grad.device)
    net.circ_raw.grad.detach().index_fill_(0, indices, 0.0)


def zero_fixed_vip_motif_optimizer_state(
    optimizer: torch.optim.Optimizer,
    net: tuned.SimpleTunedNet,
) -> None:
    state = optimizer.state.get(net.circ_raw, {})
    indices = fixed_vip_motif_indices().to(net.circ_raw.device)
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.shape == net.circ_raw.shape:
            value.detach().index_fill_(0, indices, 0.0)


def mechanism_statistics(net: tuned.SimpleTunedNet) -> dict[str, float]:
    """Pure-parameter readouts of the population circuit (kcontext DESIGN 7.8).

    The old static formula g3 - g4*relu(g1 - g2*g0) no longer exists; nothing
    in the forward pass computes a feedback coefficient. Logged instead: the 8
    Dale-positive synaptic magnitudes by name (softplus of circ_raw in
    tuned.CIRC_INDEX order), the broad-blanket scalar m_fixed effective value
    (cv_m = 0 structurally), the SOM route product w_sf*m_fixed, and
    the local competition strength.
    Firing fractions need a forward pass and are logged per arm by
    feedback_statistics. Logging only; no training-path effect.
    """
    gains = net.circuit_gains()
    named = {
        f"gain_{name}": float(gains[index].item())
        for name, index in tuned.CIRC_INDEX.items()
    }
    mean_m = float(net.m_fixed_effective().detach().item())
    raw_m = float(net.m_fixed.detach().item())
    w_sf = float(net.w_sf_effective().detach().item())
    cv_m = 0.0
    return {
        **named,
        "mean_m": mean_m,
        "m_fixed_raw": raw_m,
        "w_sf": w_sf,
        "w_sf_raw": float(net.w_sf_fixed.detach().item()),
        "m_fixed_parameterization": net.m_fixed_parameterization,
        "fixed_intrinsic_rheobases": net.fixed_intrinsic_rheobases,
        "cv_m": cv_m,
        "som_route_strength_times_mean_m": w_sf * mean_m,
        "local_comp_strength": float(
            net.local_comp_effective_strength().detach().cpu().item()
        ),
    }


@torch.no_grad()
def feedback_statistics(
    net: tuned.SimpleTunedNet,
    seed: int,
    device: torch.device,
    center_feedback: bool = False,
    feedback_mode: str | None = None,
) -> dict[str, float]:
    generator = make_generator(device, 800000 + seed)
    theta, _ = momentum_batch(256, 12, device, generator)
    predictions, _, internals = tuned.forward_seq_tuned(
        net,
        theta,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
        return_internals=True,
    )
    som_seq, vip_seq, _, _, _, _, _, _ = internals
    positive = tuned.predictive_feedback_evidence(
        predictions,
        center_feedback,
        feedback_mode,
    )
    return {
        "feedback_positive_mean": float(positive.mean().item()),
        "feedback_positive_fraction": float((positive > 0).float().mean().item()),
        "som_firing_fraction": float((som_seq > 0).float().mean().item()),
        "vip_firing_fraction": float((vip_seq > 0).float().mean().item()),
        "W_fb_bias_mean": float(net.W_fb.bias.mean().item()),
        "W_fb_bias_positive_fraction": float(
            (net.W_fb.bias > 0).float().mean().item()
        ),
        **mechanism_statistics(net),
    }


def run_alpha(
    alpha: float,
    common_state: dict[str, torch.Tensor],
    references: dict[str, float],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    event_log: EventLog,
) -> dict:
    net = tuned.build_tuned_from_config(MODEL_CONFIG).to(device)
    net.load_state_dict(copy.deepcopy(common_state))
    assert_fixed_vip_motif(net)
    common_state_hash = state_sha256(common_state)
    loaded_state_hash = state_sha256(net.state_dict())
    if loaded_state_hash != common_state_hash:
        raise RuntimeError(f"alpha {alpha} did not load the identical common state")
    initial_local_comp_param = getattr(net, "local_comp_strength_raw", None)
    initial_local_comp_raw = (
        None
        if initial_local_comp_param is None
        else initial_local_comp_param.detach().clone()
    )
    initial_local_comp_sha256 = optional_tensor_sha256(initial_local_comp_raw)
    task_weight = 1.0 - alpha
    optimizer = torch.optim.Adam(
        set_axis_parameter_policy(net, args.freeze_local_comp),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    data_generator = make_generator(device, 400000 + args.seed)
    noise_generator = make_generator(device, 500000 + args.seed)
    mismatch_stats: dict[str, int] = {
        "events": 0,
        "eligible": 0,
        "transitions": 0,
    }
    slug = alpha_slug(alpha)
    latest_path = run_dir / f"alpha_{slug}_latest.pt"
    final_path = run_dir / f"alpha_{slug}_final.pt"
    start_step = 1
    resume_path = final_path if final_path.exists() else latest_path
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device)
        saved_feedback_mode = tuned.resolve_feedback_mode(
            bool(saved.get("center_feedback", False)),
            saved.get("feedback_mode"),
        )
        if (
            saved["target_steps"] != args.axis_steps
            or saved["seed"] != args.seed
            or float(saved["alpha"]) != alpha
            or saved.get("task_weight", None) is None
            or float(saved["task_weight"]) != task_weight
            or bool(saved.get("freeze_local_comp", False))
            != args.freeze_local_comp
            or saved_feedback_mode != args.feedback_mode
            or float(saved.get("mismatch_prob", 0.0)) != args.mismatch_prob
            or saved.get("model_architecture_version")
            != tuned.MODEL_ARCHITECTURE_VERSION
            or saved.get("training_compatibility_version")
            != TRAINING_COMPATIBILITY_VERSION
        ):
            raise RuntimeError(f"alpha {alpha} checkpoint metadata does not match")
        net.load_state_dict(saved["state_dict"])
        assert_fixed_vip_motif(net)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        zero_fixed_vip_motif_optimizer_state(optimizer, net)
        restore_generator_state(data_generator, saved["data_generator_state"])
        restore_generator_state(noise_generator, saved["noise_generator_state"])
        start_step = int(saved["step"]) + 1
        event_log.write(
            {"event": "alpha_resume", "alpha": alpha, "step": start_step - 1}
        )
    current_local_comp_param = getattr(net, "local_comp_strength_raw", None)
    current_local_comp_raw = (
        None
        if current_local_comp_param is None
        else current_local_comp_param.detach()
    )
    if args.freeze_local_comp and not optional_tensor_equal(
        current_local_comp_raw, initial_local_comp_raw
    ):
        raise RuntimeError(f"alpha {alpha} resumed with changed frozen local competition")
    event_log.write(
        {
            "event": "alpha_start",
            "alpha": alpha,
            "task_weight": task_weight,
            "common_state_sha256": common_state_hash,
            "loaded_state_sha256": loaded_state_hash,
            "freeze_local_comp": args.freeze_local_comp,
            "center_feedback": args.center_feedback,
            "feedback_mode": args.feedback_mode,
            "local_comp_raw_initial_sha256": initial_local_comp_sha256,
            **mechanism_statistics(net),
        }
    )
    net.train()
    parameters = set_axis_parameter_policy(net, args.freeze_local_comp)
    for step in range(start_step, args.axis_steps + 1):
        theta, channels = momentum_batch(
            args.batch,
            args.sequence_length,
            device,
            data_generator,
            mismatch_prob=args.mismatch_prob,
            mismatch_stats=mismatch_stats,
        )
        losses = task_activity_losses(
            net,
            theta,
            channels,
            noise_generator,
            references,
            center_feedback=args.center_feedback,
            feedback_mode=args.feedback_mode,
        )
        objective = (
            task_weight * losses["task"]
            + alpha * losses["modeled_population_activity"]
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        mask_fixed_vip_motif_grad(net)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        mask_fixed_vip_motif_grad(net)
        zero_fixed_vip_motif_optimizer_state(optimizer, net)
        optimizer.step()
        enforce_fixed_vip_motif(net)
        zero_fixed_vip_motif_optimizer_state(optimizer, net)
        assert_fixed_vip_motif(net)
        if step == 1 or step % args.log_every == 0 or step == args.axis_steps:
            event_log.write(
                {
                    "event": "alpha_step",
                    "alpha": alpha,
                    "task_weight": task_weight,
                    "step": step,
                    "objective": float(objective.item()),
                    "task": float(losses["task"].item()),
                    "modeled_population_activity": float(
                        losses["modeled_population_activity"].item()
                    ),
                    "modeled_population_activity_final_e": float(
                        losses["modeled_population_activity_final_e"].item()
                    ),
                    "modeled_population_activity_pv": float(
                        losses["modeled_population_activity_pv"].item()
                    ),
                    "modeled_population_activity_som": float(
                        losses["modeled_population_activity_som"].item()
                    ),
                    "modeled_population_activity_vip": float(
                        losses["modeled_population_activity_vip"].item()
                    ),
                    "modeled_population_activity_som_gain_report": float(
                        losses[
                            "modeled_population_activity_som_gain_report"
                        ].item()
                    ),
                    "modeled_population_activity_exc_feedback_report": float(
                        losses[
                            "modeled_population_activity_exc_feedback_report"
                        ].item()
                    ),
                    "modeled_population_activity_numerator": float(
                        losses["modeled_population_activity_numerator"].item()
                    ),
                    "next_ce": float(losses["next_ce"].item()),
                    "current_ce": float(losses["current_ce"].item()),
                    "current_ce_normalized": float(
                        losses["current_ce_normalized"].item()
                    ),
                    "current_confidence": float(
                        losses["current_confidence"].item()
                    ),
                    "current_resultant_magnitude": float(
                        losses["current_resultant_magnitude"].item()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "gains": net.circuit_gains().detach().cpu().tolist(),
                    "mismatch_prob": args.mismatch_prob,
                    "mismatch_events": mismatch_stats["events"],
                    "mismatch_eligible": mismatch_stats["eligible"],
                    "mismatch_transitions": mismatch_stats["transitions"],
                    **mechanism_statistics(net),
                }
            )
        if step % args.checkpoint_every == 0 or step == args.axis_steps:
            payload = checkpoint_payload(
                stage="alpha_axis",
                seed=args.seed,
                step=step,
                target_steps=args.axis_steps,
                net=net,
                optimizer=optimizer,
                data_generator=data_generator,
                noise_generator=noise_generator,
                references=references,
                alpha=alpha,
                task_weight=task_weight,
                freeze_local_comp=args.freeze_local_comp,
                center_feedback=args.center_feedback,
                feedback_mode=args.feedback_mode,
                mismatch_prob=args.mismatch_prob,
            )
            atomic_torch_save(payload, latest_path)
            if step * 2 == args.axis_steps:
                # kcontext DESIGN section 5: gates are evaluated at the mid-arm
                # checkpoint (step 4000/8000); preserve it as a numbered file.
                # Artifact-only, no training-path effect.
                atomic_torch_save(
                    payload, run_dir / f"alpha_{slug}_step{step:05d}.pt"
                )
            if step == args.axis_steps:
                atomic_torch_save(payload, final_path)
    net.eval()
    final_local_comp_param = getattr(net, "local_comp_strength_raw", None)
    final_local_comp_raw = (
        None if final_local_comp_param is None else final_local_comp_param.detach()
    )
    if args.freeze_local_comp and not optional_tensor_equal(
        final_local_comp_raw, initial_local_comp_raw
    ):
        raise RuntimeError(f"alpha {alpha} changed frozen local competition")
    final_stats = feedback_statistics(
        net,
        args.seed,
        device,
        center_feedback=args.center_feedback,
        feedback_mode=args.feedback_mode,
    )
    result = {
        "alpha": alpha,
        "task_weight": task_weight,
        "checkpoint": str(final_path),
        "state_sha256": state_sha256(net.state_dict()),
        "feedback_statistics": final_stats,
        "gains": net.circuit_gains().detach().cpu().tolist(),
        "common_state_sha256": common_state_hash,
        "loaded_state_sha256": loaded_state_hash,
        "freeze_local_comp": args.freeze_local_comp,
        "center_feedback": args.center_feedback,
        "feedback_mode": args.feedback_mode,
        "local_comp_raw_initial_sha256": initial_local_comp_sha256,
        "local_comp_raw_final_sha256": optional_tensor_sha256(final_local_comp_raw),
        "local_comp_raw_byte_stable": optional_tensor_equal(
            final_local_comp_raw, initial_local_comp_raw
        ),
        "mismatch_prob": args.mismatch_prob,
        "mismatch_accounting": mismatch_accounting(
            args.mismatch_prob, mismatch_stats
        ),
        **mechanism_statistics(net),
    }
    event_log.write({"event": "alpha_complete", **result})
    return result


def canonical_json_sha256(payload: object) -> str:
    """Return a stable SHA-256 for JSON-compatible checkpoint metadata."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def constrained_checkpoint_integrity(
    common_state_hash: str,
    references: dict[str, float],
    data_generator_backend: str,
    noise_generator_backend: str,
) -> dict[str, str]:
    """Return immutable theta0, reference, config, and anatomy hashes."""

    generator_backends = {
        "data": data_generator_backend,
        "noise": noise_generator_backend,
    }
    if any(backend not in {"cpu", "cuda"} for backend in generator_backends.values()):
        raise ValueError("generator backends must be cpu or cuda")
    anatomy = {
        "model_architecture_version": tuned.MODEL_ARCHITECTURE_VERSION,
        "fixed_canonical_vip_motif_gains": FIXED_CANONICAL_VIP_MOTIF_GAINS,
    }
    return {
        "frozen_theta0_sha256": common_state_hash,
        "reference_values_sha256": canonical_json_sha256(references),
        "model_config_sha256": canonical_json_sha256(MODEL_CONFIG),
        "anatomy_sha256": canonical_json_sha256(anatomy),
        "generator_backends_sha256": canonical_json_sha256(generator_backends),
    }


def generator_backend(generator: torch.Generator) -> str:
    """Return the supported backend type of a PyTorch generator."""

    backend = generator.device.type
    if backend not in {"cpu", "cuda"}:
        raise RuntimeError(f"unsupported generator backend {backend!r}")
    return backend


def validate_constrained_generator_states(
    saved: dict,
    data_generator: torch.Generator,
    noise_generator: torch.Generator,
    *,
    legacy: bool,
) -> dict[str, str]:
    """Reject cross-backend or wrong-length generator state before restore."""

    requested = {
        "data": generator_backend(data_generator),
        "noise": generator_backend(noise_generator),
    }
    states = {
        "data": saved["data_generator_state"],
        "noise": saved["noise_generator_state"],
    }
    generators = {"data": data_generator, "noise": noise_generator}
    if legacy:
        for name, state in states.items():
            if state.numel() != 16:
                raise RuntimeError(
                    f"legacy constrained {name} RNG state must be 16-byte CUDA state"
                )
            if requested[name] != "cuda":
                raise RuntimeError(
                    f"legacy constrained {name} RNG state is CUDA-only; "
                    f"requested {requested[name]}"
                )
        saved_backends = {"data": "cuda", "noise": "cuda"}
    else:
        backend_fields = {
            "data": saved.get("data_generator_backend"),
            "noise": saved.get("noise_generator_backend"),
        }
        if any(backend not in {"cpu", "cuda"} for backend in backend_fields.values()):
            raise RuntimeError("constrained checkpoint RNG backend metadata is missing")
        saved_backends = backend_fields
        for name in ("data", "noise"):
            if saved_backends[name] != requested[name]:
                raise RuntimeError(
                    f"constrained {name} RNG backend mismatch: "
                    f"saved {saved_backends[name]}, requested {requested[name]}"
                )
    for name, state in states.items():
        fresh = torch.Generator(device=generators[name].device)
        expected_length = fresh.get_state().numel()
        if state.numel() != expected_length:
            raise RuntimeError(
                f"constrained {name} RNG state length {state.numel()} does not "
                f"match fresh {requested[name]} length {expected_length}"
            )
    return saved_backends


def validated_mismatch_stats(value: object) -> dict[str, int]:
    """Return exact nonnegative constrained-stream mismatch counters."""

    keys = ("events", "eligible", "transitions")
    if not isinstance(value, dict) or set(value) != set(keys):
        raise RuntimeError("constrained checkpoint mismatch counters are missing")
    counters: dict[str, int] = {}
    for key in keys:
        counter = value[key]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise RuntimeError("constrained checkpoint mismatch counters are invalid")
        counters[key] = counter
    return counters


def recover_legacy_constrained_mismatch_stats(
    saved: dict,
    training_log_path: Path,
) -> dict[str, int]:
    """Recover counters only from one exact legacy same-run/step log event."""

    if not training_log_path.is_file():
        raise RuntimeError("legacy constrained mismatch counters are unavailable")
    active_run: dict | None = None
    matches: list[dict[str, int]] = []
    with training_log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "run_start":
                active_run = row
                continue
            if row.get("event") != "constrained_step" or active_run is None:
                continue
            exact_run = (
                active_run.get("training_mode")
                == "constrained_efficient_coding"
                and active_run.get("seed") == saved["seed"]
                and active_run.get("axis_steps") == saved["target_steps"]
                and active_run.get("feedback_mode") == saved["feedback_mode"]
                and active_run.get("freeze_local_comp")
                == saved["freeze_local_comp"]
                and active_run.get("model_config") == saved["tuned_net_config"]
                and active_run.get("fixed_canonical_vip_motif_gains")
                == saved["fixed_canonical_vip_motif_gains"]
                and active_run.get("training_compatibility_version")
                == saved["training_compatibility_version"]
            )
            exact_step = (
                row.get("step") == saved["step"]
                and row.get("mismatch_prob") == saved["mismatch_prob"]
                and row.get("lambda_next") == saved["lambda_next"]
                and row.get("lambda_current") == saved["lambda_current"]
            )
            if exact_run and exact_step:
                matches.append(
                    validated_mismatch_stats(
                        {
                            "events": row.get("mismatch_events"),
                            "eligible": row.get("mismatch_eligible"),
                            "transitions": row.get("mismatch_transitions"),
                        }
                    )
                )
    if len(matches) != 1:
        raise RuntimeError(
            "legacy constrained mismatch counters lack one exact run/step match"
        )
    return matches[0]


def validate_constrained_resume_checkpoint(
    saved: dict,
    *,
    requested_target_steps: int,
    seed: int,
    freeze_local_comp: bool,
    feedback_mode: str,
    mismatch_prob: float,
    common_state_hash: str,
    references: dict[str, float],
    data_generator: torch.Generator,
    noise_generator: torch.Generator,
    legacy_training_log_path: Path,
) -> dict[str, object]:
    """Validate an exact resume, allowing only monotonic target extension."""

    required = {
        "stage",
        "step",
        "target_steps",
        "seed",
        "state_dict",
        "optimizer_state_dict",
        "data_generator_state",
        "noise_generator_state",
        "lambda_next",
        "lambda_current",
        "references",
        "tuned_net_config",
        "model_architecture_version",
        "training_compatibility_version",
        "fixed_canonical_vip_motif_gains",
        "common_state_sha256",
        "dual_step_size",
        "freeze_local_comp",
        "center_feedback",
        "feedback_mode",
        "mismatch_prob",
    }
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise RuntimeError("constrained checkpoint state is incomplete")
    saved_step = saved["step"]
    saved_target = saved["target_steps"]
    if (
        isinstance(saved_step, bool)
        or not isinstance(saved_step, int)
        or isinstance(saved_target, bool)
        or not isinstance(saved_target, int)
        or saved_step < 1
        or saved_target < saved_step
        or requested_target_steps < saved_target
    ):
        raise RuntimeError("constrained checkpoint target extension is not monotonic")
    optimizer_state = saved["optimizer_state_dict"]
    if (
        not isinstance(saved["state_dict"], dict)
        or not saved["state_dict"]
        or not isinstance(optimizer_state, dict)
        or not optimizer_state.get("state")
        or not optimizer_state.get("param_groups")
        or not isinstance(saved["data_generator_state"], torch.Tensor)
        or not isinstance(saved["noise_generator_state"], torch.Tensor)
        or saved["data_generator_state"].dtype != torch.uint8
        or saved["noise_generator_state"].dtype != torch.uint8
        or saved["data_generator_state"].ndim != 1
        or saved["noise_generator_state"].ndim != 1
    ):
        raise RuntimeError("constrained checkpoint training state is incomplete")
    saved_feedback_mode = tuned.resolve_feedback_mode(
        bool(saved.get("center_feedback", False)),
        saved["feedback_mode"],
    )
    new_format_fields = {
        "mismatch_stats",
        "frozen_theta0_sha256",
        "reference_values_sha256",
        "model_config_sha256",
        "anatomy_sha256",
        "generator_backends_sha256",
        "candidate_state_sha256",
        "data_generator_state_sha256",
        "noise_generator_state_sha256",
        "data_generator_backend",
        "noise_generator_backend",
    }
    present_new_format_fields = new_format_fields.intersection(saved)
    if present_new_format_fields and present_new_format_fields != new_format_fields:
        raise RuntimeError("constrained checkpoint state is incomplete")
    legacy = not present_new_format_fields
    saved_generator_backends = validate_constrained_generator_states(
        saved,
        data_generator,
        noise_generator,
        legacy=legacy,
    )
    integrity = constrained_checkpoint_integrity(
        common_state_hash,
        references,
        saved_generator_backends["data"],
        saved_generator_backends["noise"],
    )
    metadata_matches = (
        saved["stage"] == "constrained_efficient_coding"
        and saved["seed"] == seed
        and saved["common_state_sha256"] == common_state_hash
        and saved["references"] == references
        and saved["tuned_net_config"] == MODEL_CONFIG
        and saved["fixed_canonical_vip_motif_gains"]
        == FIXED_CANONICAL_VIP_MOTIF_GAINS
        and saved["model_architecture_version"]
        == tuned.MODEL_ARCHITECTURE_VERSION
        and saved["training_compatibility_version"]
        == TRAINING_COMPATIBILITY_VERSION
        and float(saved["dual_step_size"]) == CONSTRAINED_DUAL_STEP_SIZE
        and bool(saved["freeze_local_comp"]) == freeze_local_comp
        and saved_feedback_mode == feedback_mode
        and float(saved["mismatch_prob"]) == mismatch_prob
    )
    if not metadata_matches:
        raise RuntimeError("constrained checkpoint metadata does not match")
    if legacy:
        mismatch_stats = recover_legacy_constrained_mismatch_stats(
            saved,
            legacy_training_log_path,
        )
    else:
        mismatch_stats = validated_mismatch_stats(saved["mismatch_stats"])
        if any(saved.get(name) != value for name, value in integrity.items()):
            raise RuntimeError("constrained checkpoint integrity hash mismatch")
        if saved.get("candidate_state_sha256") != state_sha256(saved["state_dict"]):
            raise RuntimeError("constrained checkpoint candidate hash mismatch")
        if saved.get("data_generator_state_sha256") != tensor_sha256(
            saved["data_generator_state"]
        ):
            raise RuntimeError("constrained checkpoint data RNG hash mismatch")
        if saved.get("noise_generator_state_sha256") != tensor_sha256(
            saved["noise_generator_state"]
        ):
            raise RuntimeError("constrained checkpoint noise RNG hash mismatch")
    for name in ("lambda_next", "lambda_current"):
        value = float(saved[name])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("constrained checkpoint dual state is invalid")
    return {
        "step": saved_step,
        "saved_target_steps": saved_target,
        "mismatch_stats": mismatch_stats,
        "integrity": integrity,
        "generator_backends": saved_generator_backends,
        "legacy": legacy,
    }


def restore_constrained_training_state(
    saved: dict,
    candidate_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    noise_generator: torch.Generator,
    restored_mismatch_stats: dict[str, int],
    mismatch_stats: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore all mutable constrained state without resetting any component."""

    candidate_net.load_state_dict(saved["state_dict"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    restore_generator_state(data_generator, saved["data_generator_state"])
    restore_generator_state(noise_generator, saved["noise_generator_state"])
    restored_stats = validated_mismatch_stats(restored_mismatch_stats)
    mismatch_stats.clear()
    mismatch_stats.update(restored_stats)
    dtype = next(candidate_net.parameters()).dtype
    lambda_next = torch.as_tensor(saved["lambda_next"], device=device, dtype=dtype)
    lambda_current = torch.as_tensor(
        saved["lambda_current"],
        device=device,
        dtype=dtype,
    )
    return lambda_next, lambda_current


def load_constrained_resume_checkpoint(
    latest_path: Path,
    final_path: Path,
    device: torch.device,
) -> tuple[Path, dict] | None:
    """Load the constrained artifact with the greatest persisted global step."""

    loaded: list[tuple[Path, dict]] = []
    for path in (latest_path, final_path):
        if not path.exists():
            continue
        saved = torch.load(path, map_location=device)
        if (
            not isinstance(saved, dict)
            or isinstance(saved.get("step"), bool)
            or not isinstance(saved.get("step"), int)
        ):
            raise RuntimeError("constrained checkpoint global step is missing")
        loaded.append((path, saved))
    if not loaded:
        return None
    return max(
        loaded,
        key=lambda item: (item[1]["step"], item[0] == final_path),
    )


def run_constrained_efficient_coding(
    common_state: dict[str, torch.Tensor],
    references: dict[str, float],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    event_log: EventLog,
) -> dict:
    """Train one activity candidate under two paired task constraints."""

    candidate_net = tuned.build_tuned_from_config(MODEL_CONFIG).to(device)
    candidate_net.load_state_dict(copy.deepcopy(common_state))
    frozen_reference_net = tuned.build_tuned_from_config(MODEL_CONFIG).to(device)
    frozen_reference_net.load_state_dict(copy.deepcopy(common_state))
    assert_fixed_vip_motif(candidate_net)
    assert_fixed_vip_motif(frozen_reference_net)
    common_state_hash = state_sha256(common_state)
    if (
        state_sha256(candidate_net.state_dict()) != common_state_hash
        or state_sha256(frozen_reference_net.state_dict()) != common_state_hash
    ):
        raise RuntimeError("constrained candidate/reference common-state mismatch")
    for parameter in frozen_reference_net.parameters():
        parameter.requires_grad_(False)

    parameters = set_axis_parameter_policy(
        candidate_net,
        args.freeze_local_comp,
    )
    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    data_generator = make_generator(device, 400000 + args.seed)
    noise_generator = make_generator(device, 500000 + args.seed)
    data_generator_backend = generator_backend(data_generator)
    noise_generator_backend = generator_backend(noise_generator)
    integrity = constrained_checkpoint_integrity(
        common_state_hash,
        references,
        data_generator_backend,
        noise_generator_backend,
    )
    mismatch_stats: dict[str, int] = {
        "events": 0,
        "eligible": 0,
        "transitions": 0,
    }
    lambda_next = torch.zeros((), device=device)
    lambda_current = torch.zeros((), device=device)
    latest_path = run_dir / "constrained_efficient_coding_latest.pt"
    final_path = run_dir / "constrained_efficient_coding_final.pt"
    start_step = 1
    resume_checkpoint = load_constrained_resume_checkpoint(
        latest_path,
        final_path,
        device,
    )
    if resume_checkpoint is not None:
        resume_path, saved = resume_checkpoint
        validated_resume = validate_constrained_resume_checkpoint(
            saved,
            requested_target_steps=args.axis_steps,
            seed=args.seed,
            freeze_local_comp=args.freeze_local_comp,
            feedback_mode=args.feedback_mode,
            mismatch_prob=args.mismatch_prob,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=data_generator,
            noise_generator=noise_generator,
            legacy_training_log_path=run_dir / "training.jsonl",
        )
        lambda_next, lambda_current = restore_constrained_training_state(
            saved,
            candidate_net,
            optimizer,
            data_generator,
            noise_generator,
            validated_resume["mismatch_stats"],
            mismatch_stats,
            device,
        )
        assert_fixed_vip_motif(candidate_net)
        zero_fixed_vip_motif_optimizer_state(optimizer, candidate_net)
        start_step = int(validated_resume["step"]) + 1
        event_log.write(
            {
                "event": "constrained_resume",
                "checkpoint": str(resume_path),
                "step": start_step - 1,
                "saved_target_steps": validated_resume["saved_target_steps"],
                "target_steps": args.axis_steps,
                "legacy_counter_recovery": validated_resume["legacy"],
                "lambda_next": float(lambda_next.item()),
                "lambda_current": float(lambda_current.item()),
                "mismatch_stats": mismatch_stats,
            }
        )

    event_log.write(
        {
            "event": "constrained_start",
            "common_state_sha256": common_state_hash,
            "lambda_next": float(lambda_next.item()),
            "lambda_current": float(lambda_current.item()),
            "dual_step_size": CONSTRAINED_DUAL_STEP_SIZE,
            "freeze_local_comp": args.freeze_local_comp,
            "center_feedback": args.center_feedback,
            "feedback_mode": args.feedback_mode,
            **mechanism_statistics(candidate_net),
        }
    )
    candidate_net.train()
    frozen_reference_net.train()
    for step in range(start_step, args.axis_steps + 1):
        theta, channels = momentum_batch(
            args.batch,
            args.sequence_length,
            device,
            data_generator,
            mismatch_prob=args.mismatch_prob,
            mismatch_stats=mismatch_stats,
        )
        candidate_losses, reference_losses, _ = paired_constrained_task_losses(
            candidate_net,
            frozen_reference_net,
            theta,
            channels,
            noise_generator,
            references,
            center_feedback=args.center_feedback,
            feedback_mode=args.feedback_mode,
        )
        constrained = constrained_objective_terms(
            candidate_losses,
            reference_losses,
            lambda_next,
            lambda_current,
        )
        optimizer.zero_grad(set_to_none=True)
        constrained["objective"].backward()
        mask_fixed_vip_motif_grad(candidate_net)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        mask_fixed_vip_motif_grad(candidate_net)
        zero_fixed_vip_motif_optimizer_state(optimizer, candidate_net)
        optimizer.step()
        enforce_fixed_vip_motif(candidate_net)
        zero_fixed_vip_motif_optimizer_state(optimizer, candidate_net)
        assert_fixed_vip_motif(candidate_net)
        lambda_next, lambda_current = projected_dual_ascent(
            lambda_next,
            lambda_current,
            constrained["constraint_next"],
            constrained["constraint_current"],
        )
        if step == 1 or step % args.log_every == 0 or step == args.axis_steps:
            event_log.write(
                {
                    "event": "constrained_step",
                    "step": step,
                    "objective": float(constrained["objective"].item()),
                    "modeled_population_activity": float(
                        candidate_losses["modeled_population_activity"].item()
                    ),
                    "constraint_next": float(
                        constrained["constraint_next"].item()
                    ),
                    "constraint_current": float(
                        constrained["constraint_current"].item()
                    ),
                    "candidate_next": float(constrained["candidate_next"].item()),
                    "reference_next": float(constrained["reference_next"].item()),
                    "candidate_current": float(
                        constrained["candidate_current"].item()
                    ),
                    "reference_current": float(
                        constrained["reference_current"].item()
                    ),
                    "lambda_next": float(lambda_next.item()),
                    "lambda_current": float(lambda_current.item()),
                    "gradient_norm": float(gradient_norm),
                    "mismatch_prob": args.mismatch_prob,
                    "mismatch_events": mismatch_stats["events"],
                    "mismatch_eligible": mismatch_stats["eligible"],
                    "mismatch_transitions": mismatch_stats["transitions"],
                }
            )
        if step % args.checkpoint_every == 0 or step == args.axis_steps:
            payload = checkpoint_payload(
                stage="constrained_efficient_coding",
                seed=args.seed,
                step=step,
                target_steps=args.axis_steps,
                net=candidate_net,
                optimizer=optimizer,
                data_generator=data_generator,
                noise_generator=noise_generator,
                references=references,
                alpha=None,
                freeze_local_comp=args.freeze_local_comp,
                center_feedback=args.center_feedback,
                feedback_mode=args.feedback_mode,
                mismatch_prob=args.mismatch_prob,
            )
            payload.update(
                {
                    "common_state_sha256": common_state_hash,
                    "dual_step_size": CONSTRAINED_DUAL_STEP_SIZE,
                    "lambda_next": float(lambda_next.item()),
                    "lambda_current": float(lambda_current.item()),
                    "mismatch_stats": dict(mismatch_stats),
                    "data_generator_backend": data_generator_backend,
                    "noise_generator_backend": noise_generator_backend,
                    "candidate_state_sha256": state_sha256(
                        payload["state_dict"]
                    ),
                    "data_generator_state_sha256": tensor_sha256(
                        payload["data_generator_state"]
                    ),
                    "noise_generator_state_sha256": tensor_sha256(
                        payload["noise_generator_state"]
                    ),
                    **integrity,
                }
            )
            atomic_torch_save(payload, latest_path)
            if step == args.axis_steps:
                atomic_torch_save(payload, final_path)

    candidate_net.eval()
    if state_sha256(frozen_reference_net.state_dict()) != common_state_hash:
        raise RuntimeError("frozen constrained reference state changed")
    result = {
        "checkpoint": str(final_path),
        "state_sha256": state_sha256(candidate_net.state_dict()),
        "common_state_sha256": common_state_hash,
        "target_steps": args.axis_steps,
        "lambda_next": float(lambda_next.item()),
        "lambda_current": float(lambda_current.item()),
        "dual_step_size": CONSTRAINED_DUAL_STEP_SIZE,
        "freeze_local_comp": args.freeze_local_comp,
        "center_feedback": args.center_feedback,
        "feedback_mode": args.feedback_mode,
        "mismatch_prob": args.mismatch_prob,
        "mismatch_accounting": mismatch_accounting(
            args.mismatch_prob,
            mismatch_stats,
        ),
        "mismatch_stats": dict(mismatch_stats),
        "data_generator_backend": data_generator_backend,
        "noise_generator_backend": noise_generator_backend,
        **integrity,
        "feedback_statistics": feedback_statistics(
            candidate_net,
            args.seed,
            device,
            center_feedback=args.center_feedback,
            feedback_mode=args.feedback_mode,
        ),
        **mechanism_statistics(candidate_net),
    }
    event_log.write({"event": "constrained_complete", **result})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Experimental seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "emergent_task_activity_axis",
        help="Parent output directory; artifacts are written below seed_<seed>/.",
    )
    parser.add_argument(
        "--pretrain-steps", type=int, default=3000, help="Common task-only steps."
    )
    parser.add_argument(
        "--axis-steps",
        type=int,
        default=32000,
        help="Steps in every alpha arm or the constrained candidate.",
    )
    parser.add_argument(
        "--constrained-efficient-coding",
        action="store_true",
        help=(
            "Train one activity-minimizing candidate with separate paired "
            "next/current task constraints instead of the alpha sweep."
        ),
    )
    parser.add_argument("--batch", type=int, default=128, help="Sequence batch size.")
    parser.add_argument(
        "--sequence-length", type=int, default=12, help="Abstract time steps per sequence."
    )
    parser.add_argument(
        "--mismatch-prob",
        type=float,
        default=0.02,
        help=(
            "Phase-2b halt probability p_mm per eligible transition "
            "(|v_{t-1}| >= 2); pass 0.0 explicitly for the historical "
            "no-mismatch Phase-2 stream."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--clip", type=float, default=5.0, help="Gradient-norm clip.")
    parser.add_argument(
        "--log-every", type=int, default=100, help="Training JSONL log cadence."
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=250, help="Latest-checkpoint cadence."
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(ALPHAS),
        help="Unique task-activity coordinates in [0,1].",
    )
    parser.add_argument(
        "--task-weight",
        type=float,
        default=None,
        help="Deprecated; task weight is fixed to 1 - alpha.",
    )
    parser.add_argument(
        "--freeze-local-comp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep local competition identical to the common pretrain.",
    )
    parser.add_argument(
        "--center-feedback",
        action="store_true",
        help="Legacy alias selecting centered feedback evidence.",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=tuned.FEEDBACK_MODES,
        default=argparse.SUPPRESS,
        help=(
            "Omitting this option selects posterior unless "
            "--center-feedback is used."
        ),
    )
    parser.add_argument(
        "--recurrent-cell",
        choices=("gru", "rnn_tanh"),
        default="rnn_tanh",
        help="Recurrent predictor cell class.",
    )
    args = parser.parse_args()
    if not hasattr(args, "feedback_mode"):
        args.feedback_mode = None
    if args.pretrain_steps < 1 or args.axis_steps < 1:
        parser.error("step counts must be positive")
    if not args.alphas or len(args.alphas) != len(set(args.alphas)):
        parser.error("alphas must be a nonempty unique list")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        parser.error("alphas must lie in [0,1]")
    if args.task_weight is not None:
        parser.error("--task-weight is deprecated; objective uses 1 - alpha")
    try:
        validate_unique_alpha_slugs(args.alphas)
    except ValueError as error:
        parser.error(str(error))
    if args.center_feedback:
        if args.feedback_mode not in (None, tuned.FEEDBACK_MODE_CENTERED):
            parser.error(
                "--center-feedback conflicts with --feedback-mode "
                f"{args.feedback_mode}"
            )
        args.feedback_mode = tuned.FEEDBACK_MODE_CENTERED
    elif args.feedback_mode is None:
        args.feedback_mode = tuned.FEEDBACK_MODE_POSTERIOR
    args.center_feedback = args.feedback_mode == tuned.FEEDBACK_MODE_CENTERED
    return args


def main() -> None:
    args = parse_args()
    MODEL_CONFIG["recurrent_cell"] = args.recurrent_cell
    device = choose_device(args.device)
    training_mode = (
        "constrained_efficient_coding"
        if args.constrained_efficient_coding
        else "alpha_axis"
    )
    run_dir = args.out / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(run_dir / "training.jsonl")
    try:
        centered_feedback_property = (
            centered_feedback_property_check(device)
            if args.feedback_mode == tuned.FEEDBACK_MODE_CENTERED
            else None
        )
        posterior_feedback_property = (
            posterior_feedback_property_check(
                device,
                [0.0] if args.constrained_efficient_coding else args.alphas,
                args.feedback_mode,
            )
            if args.feedback_mode
            in (
                tuned.FEEDBACK_MODE_POSTERIOR,
                tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS,
            )
            else None
        )
        event_log.write(
            {
                "event": "run_start",
                "seed": args.seed,
                "device": str(device),
                "training_mode": training_mode,
                "alphas": None if args.constrained_efficient_coding else args.alphas,
                "task_weight": args.task_weight,
                "freeze_local_comp": args.freeze_local_comp,
                "center_feedback": args.center_feedback,
                "feedback_mode": args.feedback_mode,
                "training_compatibility_version": TRAINING_COMPATIBILITY_VERSION,
                "fixed_canonical_vip_motif_gains": FIXED_CANONICAL_VIP_MOTIF_GAINS,
                "centered_feedback_property_check": centered_feedback_property,
                "posterior_feedback_property_check": posterior_feedback_property,
                "pretrain_steps": args.pretrain_steps,
                "axis_steps": args.axis_steps,
                "batch": args.batch,
                "sequence_length": args.sequence_length,
                "model_config": MODEL_CONFIG,
            }
        )
        common_state, references = run_pretrain(
            args, run_dir, device, event_log
        )
        if args.constrained_efficient_coding:
            constrained_result = run_constrained_efficient_coding(
                common_state,
                references,
                args,
                run_dir,
                device,
                event_log,
            )
            alpha_results = None
        else:
            constrained_result = None
            alpha_results = [
                run_alpha(
                    alpha,
                    common_state,
                    references,
                    args,
                    run_dir,
                    device,
                    event_log,
                )
                for alpha in args.alphas
            ]
        summary = {
            "seed": args.seed,
            "device": str(device),
            "training_mode": training_mode,
            "freeze_local_comp": args.freeze_local_comp,
            "center_feedback": args.center_feedback,
            "feedback_mode": args.feedback_mode,
            "training_compatibility_version": TRAINING_COMPATIBILITY_VERSION,
            "fixed_canonical_vip_motif_gains": FIXED_CANONICAL_VIP_MOTIF_GAINS,
            "centered_feedback_property_check": centered_feedback_property,
            "posterior_feedback_property_check": posterior_feedback_property,
            "references": references,
            "common_pretrain_state_sha256": state_sha256(common_state),
        }
        if constrained_result is None:
            summary["alphas"] = alpha_results
        else:
            summary["constrained_efficient_coding"] = constrained_result
        atomic_json_save(summary, run_dir / "training_summary.json")
        event_log.write({"event": "run_complete", "summary": summary})
    finally:
        event_log.close()


if __name__ == "__main__":
    main()
