#!/usr/bin/env python3
"""Train one common task initialization and six task–energy alpha arms.

All arms clone one seed-specific task pretrain and differ only in ``alpha`` for
``L=(1-alpha)*T+alpha*E``. ``T`` combines normalized next-channel cross entropy
with noisy current-orientation population-vector precision; ``E`` is the
normalized L2/3 mean-rate proxy. The fixed-step finals, not validation-selected
checkpoints, are the experiment endpoints.

SURROUND-STUDY LINEAGE (this copy)
----------------------------------
This file is the surround-inhibition study harness. It is byte-derived from the
frozen heatmap-sweep harness
(``/home/vishnu/neuroips_analysis/heatmap_sweep_20260818/harness/train_sweep.py``,
sha256 cdd71a11cbd254aa452f3b60f4f9da4350fe9fd85f7dcdf95cd35513435c250e) and
differs from it by EXACTLY the two surround constants in ``MODEL_CONFIG``:

    pred_inhib_strength        0.0  -> 0.05   (this file's default)
    pred_inhib_sigma_channels  0.65 -> 4.0    (4 ch x 5 deg/ch = 20 deg)

Everything else — objectives, data generation, unroll, checkpoint format,
determinism — is bitwise the proven trainer (the validator's A/A control showed
an 11,000-step run reproduces the frozen no-surround cell BITWISE when the
strength is zeroed; see docs/study_record/VERDICT.md, Check 3).

Two operating points were studied (change only ``pred_inhib_strength``):
  * s = 0.05 — the delivered sharpening result (alpha=0.0; four-seed PASS).
  * s = 0.04 — the joint-window dose at which BOTH regimes (alpha=0.0
    sharpening and alpha=0.5 dampening) pass their pre-registered bars on the
    anchor seed; the dampening M band remained contested on 2/4 seeds
    (verdict O2 — see docs/ARCHITECTURE_AND_SCIENCE.md section 5).
The dose is bounded above by blanket arithmetic: the row-normalized kernel
subtracts a mean of s*sum(f)/36 from every channel, and with sum(f) ~ 25-32 at
the trained operating point this must stay below the off-center drive
(~0.46 peak flank drive) or the ring floors at the relu — s = 0.5 collapses the
circuit (proven root cause, docs/study_record/DIAGNOSTIC_REPORT.md).

Reproduction (envelope used throughout the study):
    PYTHONHASHSEED=0 python3 -B train_sweep.py --out <run_dir> --seed 8 \
        --alphas 0.0 --recurrent-cell rnn_tanh --device cuda:0
Fresh pretrain per run dir; deterministic algorithms are enabled below, so a
given (seed, config, device) reproduces bitwise.
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
ALPHAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
MODEL_CONFIG = {
    "hidden": 64,
    "ff_sigma_channels": 1.1,
    "ff_gain": 1.6,
    "decoder_gain": 8.0,
    "readout": "population_vector",
    "population_normalize": True,
    # --- THE SURROUND MECHANISM (the study's entire 2-constant diff) ---
    # Feedback-recruited subtractive surround in orientation space:
    # l23() computes  pred_inhib = s * (relu(fb) @ K.T)  and subtracts it
    # pre-relu (tuned_emergence_lib.SimpleTunedNet.l23). K is a row-normalized
    # circular Gaussian over the 36 channels; sigma = 4 ch = 20 deg puts 42.8%
    # of each unit of recruited inhibition into the +/-15-30 deg flank band
    # while the learned center gain g3 (boost) stays narrow — boost-narrow /
    # inhibit-broad is what sharpens. sigma is bio-fixed (SOM surround ~4x
    # broader than the pyramidal center; Adesnik 2012, Zhang 2014 note 37:
    # 200 um ~ 20 deg), never trained; only its recruitment (via W_fb and the
    # gains) is learned. Strength s is the dose knob: 0.05 = delivered
    # sharpening study; 0.04 = family joint window (see module docstring).
    "pred_inhib_strength": 0.05,
    "pred_inhib_sigma_channels": 4.0,
    "pred_feature_supp_strength": 0.0,
    "rate_saturation_r_max": 0.0,
    "rate_saturation_r_half": 1.0,
    "adapt_strength": 0.0,
    "adapt_decay": 0.85,
    "adapt_sigma_channels": 1.0,
    "local_comp_strength": math.log(2.0),
    "local_comp_sigma_channels": 2.0,
    "local_comp_power": 1.0,
    "local_comp_mode": "divisive",
    "local_comp_trainable": True,
    "recurrent_cell": "gru",
}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    simple.device = selected
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


def posterior_prior_excess_property_check(
    device: torch.device,
    alphas: list[float],
    feedback_mode: str,
) -> dict:
    """Verify the shared posterior-over-uniform-prior feedback transform."""

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
    uniform_exactly_zero = torch.count_nonzero(uniform).item() == 0
    finite = bool(torch.isfinite(baseline).all().item())
    minimum = float(baseline.min().item())
    maximum = float(baseline.max().item())
    raw_logits_unchanged = torch.equal(raw, raw_before)
    arm_modes = [feedback_mode for _ in alphas]
    identical_mode_all_arms = (
        bool(arm_modes)
        and len(set(arm_modes)) == 1
        and arm_modes[0] == tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS
    )
    if (
        max_shift_difference >= 1e-6
        or not uniform_exactly_zero
        or not finite
        or minimum < 0.0
        or maximum > float(N - 1)
        or not raw_logits_unchanged
        or not identical_mode_all_arms
    ):
        raise RuntimeError("posterior_prior_excess feedback property check failed")
    return {
        "max_scalar_shift_feedback_difference": max_shift_difference,
        "uniform_logits_exactly_zero": uniform_exactly_zero,
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "theoretical_maximum": float(N - 1),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ordinary momentum sequences using one isolated RNG stream."""

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
    offsets = torch.cat(
        (
            torch.zeros(batch, 1, dtype=torch.long, device=device),
            torch.cumsum(velocity[:, :-1], dim=1),
        ),
        dim=1,
    )
    channels = (initial + offsets) % N
    return channels.float() * STEP_DEG, channels


@torch.no_grad()
def reference_values(net: tuned.SimpleTunedNet, device: torch.device) -> dict[str, float]:
    channels = torch.arange(N, device=device, dtype=torch.long)
    theta = channels.float() * STEP_DEG
    l4 = tuned.l4_code(theta)
    zeros = torch.zeros(N, N, device=device)
    rates = net.l23(l4, zeros, torch.zeros_like(zeros))
    r_ref = rates.mean()
    maxima = rates.max(dim=1).values.sort().values
    a_ref = 0.5 * (maxima[N // 2 - 1] + maxima[N // 2])
    if not torch.isfinite(r_ref) or not r_ref > 0:
        raise RuntimeError("R_ref must be finite and positive")
    if not torch.isfinite(a_ref) or not a_ref > 0:
        raise RuntimeError("A_ref must be finite and positive")
    return {
        "R_ref": float(r_ref.item()),
        "A_ref": float(a_ref.item()),
        "sigma_train": float((0.25 * a_ref).item()),
    }


def task_energy_losses(
    net: tuned.SimpleTunedNet,
    theta: torch.Tensor,
    channels: torch.Tensor,
    noise_generator: torch.Generator,
    references: dict[str, float],
    center_feedback: bool = False,
    feedback_mode: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the task and normalized mean-rate terms for one sequence batch.

    Parameters
    ----------
    theta:
        Degree-valued orientation bins with shape ``[B,S]``. One channel is
        ``5`` nominal degrees.
    channels:
        Matching zero-based integer channel labels with shape ``[B,S]``.
    noise_generator:
        Arm-local generator used for exactly one ``[B,S,36]`` Gaussian draw.
    references:
        Positive scalar activity references in arbitrary units: ``R_ref`` and
        ``sigma_train``.

    Returns
    -------
    dict[str, torch.Tensor]
        Scalar dimensionless tensors. With logits ``[B,S,36]`` and raw L2/3
        rates ``r[B,S,36]`` in arbitrary activity units,
        ``Lpred=CE(logits[:,:-1], channels[:,1:])`` and noisy rectified rates
        define ``Lpv=mean(1-circular_alignment)``. The returned terms are
        ``task=0.5*Lpred/log(36)+0.5*Lpv/2`` and
        ``energy=mean(r)/R_ref``. Feedback computed after time ``t`` affects
        L2/3 only at ``t+1``; the first response has zero feedback state.
    """

    predictions, rates = tuned.forward_seq_tuned(
        net,
        theta,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
    )
    next_ce = F.cross_entropy(
        predictions[:, :-1, :].reshape(-1, N), channels[:, 1:].reshape(-1)
    )
    noise = torch.randn(
        rates.shape,
        device=rates.device,
        dtype=rates.dtype,
        generator=noise_generator,
    ) * references["sigma_train"]
    activity = F.relu(rates + noise)
    angles = 2.0 * math.pi * torch.arange(N, device=rates.device) / float(N)
    x_component = (activity * torch.cos(angles)).sum(dim=-1)
    y_component = (activity * torch.sin(angles)).sum(dim=-1)
    target_angles = 2.0 * math.pi * channels.float() / float(N)
    aligned = (
        x_component * torch.cos(target_angles)
        + y_component * torch.sin(target_angles)
    )
    magnitude = torch.sqrt(x_component.square() + y_component.square())
    epsilon = 1e-8 * N * references["R_ref"]
    population_vector_loss = (1.0 - aligned / (magnitude + epsilon)).mean()
    task = 0.5 * next_ce / math.log(N) + 0.5 * population_vector_loss / 2.0
    energy = rates.mean() / references["R_ref"]
    return {
        "next_ce": next_ce,
        "population_vector_loss": population_vector_loss,
        "task": task,
        "energy": energy,
    }


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
    parameters.append(net.circ_raw)
    if not freeze_local_comp:
        parameters.append(net.local_comp_strength_raw)
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
) -> dict:
    resolved_feedback_mode = tuned.resolve_feedback_mode(
        center_feedback,
        feedback_mode,
    )
    return {
        "stage": stage,
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
    references = reference_values(net, device)
    optimizer = torch.optim.Adam(
        set_pretrain_parameter_policy(net), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )
    data_generator = make_generator(device, 200000 + args.seed)
    noise_generator = make_generator(device, 300000 + args.seed)
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
        ):
            raise RuntimeError("pretrain checkpoint metadata does not match this run")
        net.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        restore_generator_state(data_generator, saved["data_generator_state"])
        restore_generator_state(noise_generator, saved["noise_generator_state"])
        references = saved["references"]
        start_step = int(saved["step"]) + 1
        event_log.write({"event": "pretrain_resume", "step": start_step - 1})
    net.train()
    for step in range(start_step, args.pretrain_steps + 1):
        theta, channels = momentum_batch(
            args.batch, args.sequence_length, device, data_generator
        )
        losses = task_energy_losses(
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
                    "population_vector_loss": float(
                        losses["population_vector_loss"].item()
                    ),
                    "energy_descriptive": float(losses["energy"].item()),
                    "gradient_norm": float(gradient_norm),
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
        }
    )
    return common_state, references


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.1f}".replace(".", "p")


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def mechanism_statistics(net: tuned.SimpleTunedNet) -> dict[str, float]:
    gains = F.softplus(net.circ_raw)
    som_margin = gains[1] - gains[2] * gains[0]
    effective = gains[3] - gains[4] * F.relu(som_margin)
    return {
        "som_margin_gs_minus_gsv_times_gv": float(som_margin.item()),
        "effective_net_som_vip_feedback_coefficient": float(effective.item()),
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
    predictions, _ = tuned.forward_seq_tuned(
        net,
        theta,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
    )
    positive = tuned.predictive_feedback_evidence(
        predictions,
        center_feedback,
        feedback_mode,
    )
    return {
        "feedback_positive_mean": float(positive.mean().item()),
        "feedback_positive_fraction": float((positive > 0).float().mean().item()),
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
    common_state_hash = state_sha256(common_state)
    loaded_state_hash = state_sha256(net.state_dict())
    if loaded_state_hash != common_state_hash:
        raise RuntimeError(f"alpha {alpha} did not load the identical common state")
    initial_local_comp_raw = net.local_comp_strength_raw.detach().clone()
    initial_local_comp_sha256 = tensor_sha256(initial_local_comp_raw)
    task_weight = (1.0 - alpha) if args.task_weight is None else args.task_weight
    optimizer = torch.optim.Adam(
        set_axis_parameter_policy(net, args.freeze_local_comp),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    data_generator = make_generator(device, 400000 + args.seed)
    noise_generator = make_generator(device, 500000 + args.seed)
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
        ):
            raise RuntimeError(f"alpha {alpha} checkpoint metadata does not match")
        net.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        restore_generator_state(data_generator, saved["data_generator_state"])
        restore_generator_state(noise_generator, saved["noise_generator_state"])
        start_step = int(saved["step"]) + 1
        event_log.write(
            {"event": "alpha_resume", "alpha": alpha, "step": start_step - 1}
        )
    if args.freeze_local_comp and not torch.equal(
        net.local_comp_strength_raw.detach(), initial_local_comp_raw
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
            args.batch, args.sequence_length, device, data_generator
        )
        losses = task_energy_losses(
            net,
            theta,
            channels,
            noise_generator,
            references,
            center_feedback=args.center_feedback,
            feedback_mode=args.feedback_mode,
        )
        objective = task_weight * losses["task"] + alpha * losses["energy"]
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.axis_steps:
            event_log.write(
                {
                    "event": "alpha_step",
                    "alpha": alpha,
                    "task_weight": task_weight,
                    "step": step,
                    "objective": float(objective.item()),
                    "task": float(losses["task"].item()),
                    "energy": float(losses["energy"].item()),
                    "next_ce": float(losses["next_ce"].item()),
                    "population_vector_loss": float(
                        losses["population_vector_loss"].item()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "gains": F.softplus(net.circ_raw).detach().cpu().tolist(),
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
            )
            atomic_torch_save(payload, latest_path)
            if step == args.axis_steps:
                atomic_torch_save(payload, final_path)
    net.eval()
    final_local_comp_raw = net.local_comp_strength_raw.detach()
    if args.freeze_local_comp and not torch.equal(
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
        "gains": F.softplus(net.circ_raw).detach().cpu().tolist(),
        "common_state_sha256": common_state_hash,
        "loaded_state_sha256": loaded_state_hash,
        "freeze_local_comp": args.freeze_local_comp,
        "center_feedback": args.center_feedback,
        "feedback_mode": args.feedback_mode,
        "local_comp_raw_initial_sha256": initial_local_comp_sha256,
        "local_comp_raw_final_sha256": tensor_sha256(final_local_comp_raw),
        "local_comp_raw_byte_stable": torch.equal(
            final_local_comp_raw, initial_local_comp_raw
        ),
        **mechanism_statistics(net),
    }
    event_log.write({"event": "alpha_complete", **result})
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
        default=ROOT / "outputs" / "emergent_task_energy_axis",
        help="Parent output directory; artifacts are written below seed_<seed>/.",
    )
    parser.add_argument(
        "--pretrain-steps", type=int, default=3000, help="Common task-only steps."
    )
    parser.add_argument(
        "--axis-steps", type=int, default=8000, help="Steps in every alpha arm."
    )
    parser.add_argument("--batch", type=int, default=128, help="Sequence batch size.")
    parser.add_argument(
        "--sequence-length", type=int, default=12, help="Abstract time steps per sequence."
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
        help="Unique task–energy coordinates in [0,1].",
    )
    parser.add_argument(
        "--task-weight",
        type=float,
        default=None,
        help="Task-term weight; omitted (None) reproduces the historical 1 - alpha.",
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
            "Omitting this option selects posterior_prior_excess unless "
            "--center-feedback is used."
        ),
    )
    parser.add_argument(
        "--recurrent-cell",
        choices=("gru", "rnn_tanh"),
        default="gru",
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
    if args.center_feedback:
        if args.feedback_mode not in (None, tuned.FEEDBACK_MODE_CENTERED):
            parser.error(
                "--center-feedback conflicts with --feedback-mode "
                f"{args.feedback_mode}"
            )
        args.feedback_mode = tuned.FEEDBACK_MODE_CENTERED
    elif args.feedback_mode is None:
        args.feedback_mode = tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS
    args.center_feedback = args.feedback_mode == tuned.FEEDBACK_MODE_CENTERED
    return args


def main() -> None:
    args = parse_args()
    MODEL_CONFIG["recurrent_cell"] = args.recurrent_cell
    device = choose_device(args.device)
    run_dir = args.out / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(run_dir / "training.jsonl")
    try:
        centered_feedback_property = (
            centered_feedback_property_check(device)
            if args.feedback_mode == tuned.FEEDBACK_MODE_CENTERED
            else None
        )
        posterior_prior_excess_property = (
            posterior_prior_excess_property_check(
                device,
                args.alphas,
                args.feedback_mode,
            )
            if args.feedback_mode
            == tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS
            else None
        )
        event_log.write(
            {
                "event": "run_start",
                "seed": args.seed,
                "device": str(device),
                "alphas": args.alphas,
                "task_weight": args.task_weight,
                "freeze_local_comp": args.freeze_local_comp,
                "center_feedback": args.center_feedback,
                "feedback_mode": args.feedback_mode,
                "centered_feedback_property_check": centered_feedback_property,
                "posterior_prior_excess_property_check": (
                    posterior_prior_excess_property
                ),
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
            "alphas": alpha_results,
            "freeze_local_comp": args.freeze_local_comp,
            "center_feedback": args.center_feedback,
            "feedback_mode": args.feedback_mode,
            "centered_feedback_property_check": centered_feedback_property,
            "posterior_prior_excess_property_check": (
                posterior_prior_excess_property
            ),
            "references": references,
            "common_pretrain_state_sha256": state_sha256(common_state),
        }
        atomic_json_save(summary, run_dir / "training_summary.json")
        event_log.write({"event": "run_complete", "summary": summary})
    finally:
        event_log.close()


if __name__ == "__main__":
    main()
