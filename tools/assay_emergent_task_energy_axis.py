#!/usr/bin/env python3
"""Measure fixed operational continuation/reversal pairs for alpha arms.

For every final channel and six signed velocities, condition A continues a
constant velocity and condition B reverses at its final transition. B's final
velocity change is outside the training acceleration support. The three primary
readout families are final L2/3 mean rate, noise-held-out condition-blind
decoding, and aligned center/flank shape. This executable reports measurements;
it does not apply a phenotype acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import simple_net as simple  # noqa: E402
import train_sweep  # noqa: E402
import tuned_emergence_lib as tuned  # noqa: E402


N = 36
STEP_DEG = 5.0
ALPHAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
VELOCITIES = (-3, -2, -1, 1, 2, 3)
OFFSETS = tuple(range(-18, 18))
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
DECODER_TRAIN_REPEATS = 32
DECODER_TEST_REPEATS = 32
DECODER_TRAIN_NOISE_SEED = 910001
DECODER_TEST_NOISE_SEED = 910002


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


def alpha_tag(alpha: float | str) -> str:
    return train_sweep.alpha_tag(alpha)


def alpha_slug(alpha: float | str) -> str:
    return train_sweep.alpha_slug(alpha)


def matched_pairs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 216 matched length-five A/B histories and their final channels.

    A is ``[y-4v,y-3v,y-2v,y-v,y]`` (operational continuation) and B is
    ``[y+2v,y+v,y,y-v,y]`` (operational OOD reversal), for all 36 ``y`` and
    ``v in {-3,-2,-1,1,2,3}``. Degree-valued histories have shape ``[216,5]``;
    final channel indices have shape ``[216]``.
    """

    rows_a: list[list[int]] = []
    rows_b: list[list[int]] = []
    final_channels: list[int] = []
    for final_channel in range(N):
        for velocity in VELOCITIES:
            rows_a.append(
                [
                    (final_channel - 4 * velocity) % N,
                    (final_channel - 3 * velocity) % N,
                    (final_channel - 2 * velocity) % N,
                    (final_channel - velocity) % N,
                    final_channel,
                ]
            )
            rows_b.append(
                [
                    (final_channel + 2 * velocity) % N,
                    (final_channel + velocity) % N,
                    final_channel,
                    (final_channel - velocity) % N,
                    final_channel,
                ]
            )
            final_channels.append(final_channel)
    channels_a = torch.tensor(rows_a, dtype=torch.long, device=device)
    channels_b = torch.tensor(rows_b, dtype=torch.long, device=device)
    finals = torch.tensor(final_channels, dtype=torch.long, device=device)
    return channels_a.float() * STEP_DEG, channels_b.float() * STEP_DEG, finals


def summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu")
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def align_rates(rates: torch.Tensor, finals: torch.Tensor) -> torch.Tensor:
    offsets = torch.tensor(OFFSETS, dtype=torch.long, device=rates.device)
    indices = (finals[:, None] + offsets[None, :]) % N
    return rates.gather(1, indices)


def population_alignment(
    rates: torch.Tensor, finals: torch.Tensor, r_ref: float
) -> torch.Tensor:
    activity = rates.clamp_min(0.0).to(torch.float64)
    angles = 2.0 * math.pi * torch.arange(
        N, dtype=torch.float64, device=rates.device
    ) / float(N)
    x_component = (activity * torch.cos(angles)).sum(dim=1)
    y_component = (activity * torch.sin(angles)).sum(dim=1)
    target_angles = 2.0 * math.pi * finals.to(torch.float64) / float(N)
    numerator = (
        x_component * torch.cos(target_angles)
        + y_component * torch.sin(target_angles)
    )
    denominator = torch.sqrt(x_component.square() + y_component.square())
    return numerator / (denominator + 1e-8 * N * r_ref)


def normalize_noisy_trials(values: torch.Tensor) -> torch.Tensor:
    """Rectify, then normalize each trial by L1 followed by L2 norm."""

    values = F.relu(values)
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.normalize(values, p=2, dim=-1, eps=1e-12)


def paired_noisy_features(
    rates_a: torch.Tensor,
    rates_b: torch.Tensor,
    sigma: float,
    repeats: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use one fixed noise table for paired A/B trials in one split."""

    generator = torch.Generator(device=rates_a.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        (repeats, *rates_a.shape),
        dtype=rates_a.dtype,
        device=rates_a.device,
        generator=generator,
    ) * sigma
    features_a = normalize_noisy_trials(rates_a.unsqueeze(0) + noise)
    features_b = normalize_noisy_trials(rates_b.unsqueeze(0) + noise)
    return features_a.flatten(0, 1), features_b.flatten(0, 1)


def fit_balanced_cosine_centroids(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    labels: torch.Tensor,
    repeats: int,
) -> torch.Tensor:
    repeated_labels = labels.repeat(repeats)
    pooled_features = torch.cat((features_a, features_b), dim=0)
    pooled_labels = torch.cat((repeated_labels, repeated_labels), dim=0)
    centroids = torch.stack(
        [pooled_features[pooled_labels == label].mean(dim=0) for label in range(N)]
    )
    return F.normalize(centroids, p=2, dim=-1, eps=1e-12)


def top1_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    repeats: int,
    centroids: torch.Tensor,
) -> float:
    predictions = (features @ centroids.t()).argmax(dim=-1)
    return float((predictions == labels.repeat(repeats)).float().mean().item())


def condition_blind_held_out_decoding(
    rates_a: torch.Tensor,
    rates_b: torch.Tensor,
    finals: torch.Tensor,
    sigma: float,
) -> dict[str, float | int | str]:
    """Fit one pooled A+B centroid readout and test independent noise draws.

    The underlying 216 histories occur in both splits. Only additive noise is
    held out; this is not stimulus-held-out or history-held-out decoding.
    Rates have shape ``[216,36]`` in arbitrary activity units and ``finals``
    contains their 36-class targets.
    """

    train_a, train_b = paired_noisy_features(
        rates_a,
        rates_b,
        sigma,
        DECODER_TRAIN_REPEATS,
        DECODER_TRAIN_NOISE_SEED,
    )
    centroids = fit_balanced_cosine_centroids(
        train_a, train_b, finals, DECODER_TRAIN_REPEATS
    )
    test_a, test_b = paired_noisy_features(
        rates_a,
        rates_b,
        sigma,
        DECODER_TEST_REPEATS,
        DECODER_TEST_NOISE_SEED,
    )
    accuracy_a = top1_accuracy(
        test_a, finals, DECODER_TEST_REPEATS, centroids
    )
    accuracy_b = top1_accuracy(
        test_b, finals, DECODER_TEST_REPEATS, centroids
    )
    return {
        "expected_A_held_out_top1_accuracy": accuracy_a,
        "unexpected_B_held_out_top1_accuracy": accuracy_b,
        "expected_A_minus_unexpected_B_accuracy": accuracy_a - accuracy_b,
        "classes": N,
        "train_repeats_per_pair_condition": DECODER_TRAIN_REPEATS,
        "test_repeats_per_pair_condition": DECODER_TEST_REPEATS,
        "training_noise_sigma": sigma,
        "normalization": "per_trial_rectify_then_L1_then_L2",
        "classifier": "one_condition_blind_balanced_pooled_A_plus_B_cosine_nearest_centroid",
    }


def synthetic_construct_check(device: torch.device) -> dict[str, float]:
    """Show alignment ignores symmetric scale while noisy accuracy does not."""

    labels = torch.arange(N, dtype=torch.long, device=device)
    full = torch.eye(N, dtype=torch.float32, device=device)
    attenuated = 0.05 * full
    alignment_full = population_alignment(full, labels, r_ref=1e-3)
    alignment_attenuated = population_alignment(attenuated, labels, r_ref=1e-3)
    alignment_difference = float(
        (alignment_full - alignment_attenuated).abs().max().item()
    )
    train_a, train_b = paired_noisy_features(
        full, full, sigma=0.2, repeats=16, seed=920001
    )
    centroids = fit_balanced_cosine_centroids(train_a, train_b, labels, 16)
    test_full, test_attenuated = paired_noisy_features(
        full, attenuated, sigma=0.2, repeats=64, seed=920002
    )
    full_accuracy = top1_accuracy(test_full, labels, 64, centroids)
    attenuated_accuracy = top1_accuracy(test_attenuated, labels, 64, centroids)
    if alignment_difference > 1e-6 or not full_accuracy > attenuated_accuracy:
        raise RuntimeError("synthetic decoding construct check failed")
    return {
        "max_alignment_change_under_symmetric_amplitude_loss": alignment_difference,
        "full_amplitude_held_out_accuracy": full_accuracy,
        "attenuated_held_out_accuracy": attenuated_accuracy,
        "accuracy_drop": full_accuracy - attenuated_accuracy,
    }


def shape_quantities(
    rates: torch.Tensor, finals: torch.Tensor, r_ref: float
) -> dict[str, torch.Tensor]:
    """Return per-history aligned center, flank, normalized flank, and Q.

    Raw rates have shape ``[216,36]`` in arbitrary activity units. Profiles are
    circularly aligned to ``finals``. Center uses offsets ``{-1,0,+1}`` and
    flank uses ``{±3,±4,±5,±6}`` channels. ``Fq`` and ``Q`` are dimensionless;
    center and flank retain activity units.
    """

    aligned = align_rates(rates, finals).to(torch.float64)
    offset_to_index = {offset: index for index, offset in enumerate(OFFSETS)}
    center_indices = torch.tensor(
        [offset_to_index[offset] for offset in CENTER_OFFSETS],
        dtype=torch.long,
        device=rates.device,
    )
    flank_indices = torch.tensor(
        [offset_to_index[offset] for offset in FLANK_OFFSETS],
        dtype=torch.long,
        device=rates.device,
    )
    center = aligned.index_select(1, center_indices).mean(dim=1)
    flank = aligned.index_select(1, flank_indices).mean(dim=1)
    normalized = aligned / (aligned.sum(dim=1, keepdim=True) + 1e-8 * N * r_ref)
    center_q = normalized.index_select(1, center_indices).mean(dim=1)
    flank_q = normalized.index_select(1, flank_indices).mean(dim=1)
    q_score = (center_q - flank_q) / (center_q + flank_q + 1e-8)
    return {"center": center, "flank": flank, "Fq": flank_q, "Q": q_score}


def load_arm(path: Path, device: torch.device) -> tuple[tuned.SimpleTunedNet, dict]:
    checkpoint = torch.load(path, map_location=device)
    architecture = checkpoint.get("model_architecture_version")
    if architecture not in (
        tuned.MODEL_ARCHITECTURE_VERSION,
        tuned.TEMPORAL_MODEL_ARCHITECTURE_VERSION,
        tuned.LEARNED_TEMPORAL_MODEL_ARCHITECTURE_VERSION,
    ):
        raise RuntimeError(
            "checkpoint architecture does not match current tuned circuit"
        )
    if checkpoint["tuned_net_config"].get("model_architecture_version") != architecture:
        raise RuntimeError("checkpoint architecture and model configuration disagree")
    net = tuned.build_tuned_from_config(checkpoint["tuned_net_config"]).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint


@torch.no_grad()
def assay_arm(
    path: Path,
    device: torch.device,
    common_local_comp_raw: torch.Tensor | None,
) -> tuple[dict, dict]:
    """Replay one final checkpoint and summarize its three assay families.

    Both A and B use the checkpoint's normal feedback-on unroll. Only final-step
    ``[216,36]`` L2/3 rates enter the mean-rate, decoder, and shape summaries.
    The common local-competition tensor is checked when the checkpoint declares
    it frozen.
    """

    net, checkpoint = load_arm(path, device)
    center_feedback = bool(checkpoint.get("center_feedback", False))
    feedback_mode = tuned.resolve_feedback_mode(
        center_feedback,
        checkpoint.get("feedback_mode"),
    )
    freeze_local_comp = bool(checkpoint.get("freeze_local_comp", False))
    local_comp_raw = checkpoint["state_dict"].get("local_comp_strength_raw")
    local_comp_matches_common = (
        local_comp_raw is None
        and common_local_comp_raw is None
    ) or (
        local_comp_raw is not None
        and common_local_comp_raw is not None
        and torch.equal(local_comp_raw, common_local_comp_raw)
    )
    if freeze_local_comp and not local_comp_matches_common:
        raise RuntimeError("frozen local competition differs from common pretrain")
    theta_a, theta_b, finals = matched_pairs(device)
    _, rates_a_all = tuned.forward_seq_tuned(
        net,
        theta_a,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
    )
    _, rates_b_all = tuned.forward_seq_tuned(
        net,
        theta_b,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
    )
    rates_a = rates_a_all[:, -1, :]
    rates_b = rates_b_all[:, -1, :]
    r_ref = float(checkpoint["references"]["R_ref"])
    epsilon = 1e-8 * N * r_ref

    mean_rate_a = rates_a.to(torch.float64).mean(dim=1)
    mean_rate_b = rates_b.to(torch.float64).mean(dim=1)
    paired_rate_difference = mean_rate_b - mean_rate_a
    mean_rate_difference = paired_rate_difference.mean()
    relative_saving_ratio_of_means = mean_rate_difference / (
        mean_rate_b.mean() + epsilon
    )

    alignment_a = population_alignment(rates_a, finals, r_ref)
    alignment_b = population_alignment(rates_b, finals, r_ref)
    alignment_contrast = alignment_a - alignment_b

    shape_a = shape_quantities(rates_a, finals, r_ref)
    shape_b = shape_quantities(rates_b, finals, r_ref)
    center_contrast = (shape_a["center"] - shape_b["center"]) / r_ref
    flank_contrast = (shape_a["flank"] - shape_b["flank"]) / r_ref
    flank_q_contrast = shape_a["Fq"] - shape_b["Fq"]
    q_contrast = shape_a["Q"] - shape_b["Q"]
    held_out_decoding = condition_blind_held_out_decoding(
        rates_a,
        rates_b,
        finals,
        sigma=float(checkpoint["references"]["sigma_train"]),
    )

    result = {
        "mean_rate_energy_saving": {
            "condition_a_mean_rate": summary(mean_rate_a),
            "condition_b_mean_rate": summary(mean_rate_b),
            "paired_mean_difference_unexpected_B_minus_expected_A": float(
                mean_rate_difference.item()
            ),
            "relative_saving_ratio_of_means": float(
                relative_saving_ratio_of_means.item()
            ),
            "paired_median_difference_unexpected_B_minus_expected_A": float(
                paired_rate_difference.median().item()
            ),
            "fraction_pairs_unexpected_B_greater_expected_A": float(
                (paired_rate_difference > 0).float().mean().item()
            ),
        },
        "circular_population_vector_alignment": {
            "expected_A_alignment": summary(alignment_a),
            "unexpected_B_alignment": summary(alignment_b),
            "expected_A_minus_unexpected_B_alignment": summary(
                alignment_contrast
            ),
        },
        "condition_blind_held_out_36_class_decoding": held_out_decoding,
        "aligned_center_flank_Q_shape_contrasts": {
            "center_a_minus_b_over_R_ref": summary(center_contrast),
            "flank_a_minus_b_over_R_ref": summary(flank_contrast),
            "Fq_a_minus_b": summary(flank_q_contrast),
            "Q_a_minus_b": summary(q_contrast),
        },
    }
    execution_config = {
        "center_feedback": center_feedback,
        "feedback_mode": feedback_mode,
        "freeze_local_comp": freeze_local_comp,
        "local_comp_matches_common_pretrain": local_comp_matches_common,
        "local_comp_strength": float(
            net.local_comp_effective_strength().detach().cpu().item()
        ),
    }
    return result, execution_config


def temporal_window_indices(times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    early = torch.stack([
        torch.nonzero(torch.isclose(times, times.new_tensor(t), atol=1e-6),
                      as_tuple=False).flatten()[0]
        for t in (0.1, 0.2)
    ])
    late = torch.nonzero((times > 3.0 + 1e-6) & (times <= 4.0 + 1e-6),
                         as_tuple=False).flatten()
    return early, late


def temporal_current_metrics(net, rates, labels, noise, references, generator):
    """Use the training readout; positive confidence/gain leave top-1 unchanged."""
    losses = train_sweep.confidence_weighted_current_orientation_ce(
        net, rates, labels, generator, references, current_decoder_noise=noise,
        current_readout="population_vector",
    )
    activity = F.relu(rates + noise)
    x = activity @ net.readout_cos
    y = activity @ net.readout_sin
    predictions = (x.unsqueeze(-1) * net.readout_cos
                   + y.unsqueeze(-1) * net.readout_sin).argmax(dim=-1)
    return {
        "current_ce": float(losses["current_ce"].item()),
        "current_accuracy": float((predictions == labels).double().mean().item()),
    }


@torch.no_grad()
def temporal_held_out(net, checkpoint, device):
    """Eight fixed new history batches; two instantaneous early decisions each."""
    data = train_sweep.make_generator(device, 930001)
    noise_generator = train_sweep.make_generator(device, 930002)
    references = checkpoint["references"]
    totals = {}
    for _ in range(8):
        theta, channels = train_sweep.momentum_batch(
            128, 12, device, data, mismatch_prob=0.02,
        )
        predictions, _, trace = tuned.forward_seq_tuned(
            net, theta, feedback_mode=checkpoint["feedback_mode"],
            return_timecourse=True,
        )
        early, _ = temporal_window_indices(trace["times"])
        rates = trace["rates"].index_select(2, early)
        labels = channels.unsqueeze(-1).expand(rates.shape[:-1])
        noise = torch.randn(rates.shape, device=device, generator=noise_generator,
                            dtype=rates.dtype) * references["sigma_train"]
        metrics = temporal_current_metrics(
            net, rates, labels, noise, references, noise_generator,
        )
        metrics["next_ce"] = float(F.cross_entropy(
            predictions[:, :-1].reshape(-1, N), channels[:, 1:].reshape(-1),
        ).item())
        metrics["next_accuracy"] = float(
            (predictions[:, :-1].argmax(-1) == channels[:, 1:]).double().mean().item()
        )
        integrals = train_sweep.temporal_activity_integrals(net, trace)
        for key in ("on_integral", "gap_integral", "cycle_integral"):
            metrics[key] = float(integrals[key].double().mean().item())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value / 8.0
    totals["normalized_cycle_activity"] = totals["cycle_integral"] / (
        4.0 * references["R_ref"]
    )
    return totals


@torch.no_grad()
def temporal_probe(net, checkpoint, device, *, clamp=False, measurements=True):
    theta_a, theta_b, finals = matched_pairs(device)
    theta = torch.cat((theta_a, theta_b))
    _, _, trace = tuned.forward_seq_tuned(
        net, theta, feedback_mode=checkpoint["feedback_mode"],
        return_timecourse=True, clamp_final_probe_sst=clamp,
    )
    integrals = train_sweep.temporal_activity_integrals(net, trace)
    count = len(finals)
    cycle = integrals["cycle_integral"][:, -1].double()
    result = {"cycle_integral": {
        "expected": float(cycle[:count].mean().item()),
        "unexpected": float(cycle[count:].mean().item()),
        "pooled": float(cycle.mean().item()),
    }}
    # Cumulative on-period cost and exact decay during the terminal blank.
    times = trace["times"]
    internals = trace["internals"]
    populations = {
        "final_e": trace["rates"][:, -1].mean(-1),
        "som": internals[0][:, -1].mean(-1),
        "vip": internals[1][:, -1].mean(-1),
        "pv": train_sweep.pv_scalar_from_pre_pv(net, internals[3][:, -1]).squeeze(-1),
    }
    result["cycle_activity_components"] = {}
    for name, activity in populations.items():
        component = torch.trapezoid(activity, times, dim=-1) * train_sweep.MODELED_ACTIVITY_WEIGHTS[name]
        if name == "som":
            component = component + integrals["gap_integral"][:, -1]
        result["cycle_activity_components"][name] = {
            "expected": float(component[:count].double().mean().item()),
            "unexpected": float(component[count:].double().mean().item()),
            "pooled": float(component.double().mean().item()),
        }
    instantaneous = integrals["instantaneous"][:, -1]
    increments = 0.5 * (instantaneous[:, 1:] + instantaneous[:, :-1]) * times.diff()
    cumulative = torch.cat((torch.zeros_like(increments[:, :1]),
                            increments.cumsum(dim=-1)), dim=-1)
    blank_times = torch.arange(1, 51, device=device, dtype=times.dtype) / 10.0
    if net.learn_temporal_kinetics:
        _, tau_p = net.temporal_time_constants()
        blank_fraction = -torch.expm1(-blank_times / tau_p) / (
            -torch.expm1(-net.temporal_protocol["gap_duration"] / tau_p)
        )
    else:
        blank_fraction = -torch.expm1(-blank_times) / (-math.expm1(-5.0))
    cumulative = torch.cat((cumulative, cumulative[:, -1:]
                            + integrals["gap_integral"][:, -1:] * blank_fraction), dim=-1)
    result["cycle_times"] = torch.cat((times, 4.0 + blank_times)).cpu().tolist()
    result["cumulative_activity"] = cumulative.double().mean(0).cpu().tolist()
    if not measurements:
        return result

    rates = trace["rates"][:, -1]
    steps = rates.shape[1]
    labels = torch.cat((finals, finals))
    aligned = align_rates(rates.flatten(0, 1), labels.repeat_interleave(steps))
    aligned = aligned.reshape(2 * count, steps, N).double()
    first_labels = (theta[:, 0] / STEP_DEG).round().long() % N
    first = align_rates(trace["rates"][:, 0].flatten(0, 1),
                        first_labels.repeat_interleave(steps))
    baseline = first.reshape(2 * count, steps, N).double().mean(0)
    expected, unexpected = aligned[:count].mean(0), aligned[count:].mean(0)
    early, late = temporal_window_indices(times)
    result.update({
        "times": times.cpu().tolist(),
        "offset_degrees": [offset * STEP_DEG for offset in OFFSETS],
        "expected": expected.cpu().tolist(),
        "unexpected": unexpected.cpu().tolist(),
        "baseline": baseline.cpu().tolist(),
        "windows": {},
    })
    for name, indices in (("early", early), ("late", late)):
        e, u, b = (curve.index_select(0, indices).mean(0)
                   for curve in (expected, unexpected, baseline))
        def mean_offsets(curve, offsets):
            return float(curve[[OFFSETS.index(i) for i in offsets]].mean().item())
        center, flank15 = mean_offsets(e, (0,)), mean_offsets(e, (-3, 3))
        result["windows"][name] = {
            "expected": e.cpu().tolist(), "unexpected": u.cpu().tolist(),
            "baseline": b.cpu().tolist(),
            "preferred_ratio": center / mean_offsets(b, (0,)),
            "shoulder_ratio": mean_offsets(e, (-1, 1)) / mean_offsets(b, (-1, 1)),
            "flank_ratio": mean_offsets(e, FLANK_OFFSETS) / mean_offsets(b, FLANK_OFFSETS),
            "center_rate": center, "flank15_rate": flank15,
            "expected_mean": float(e.mean().item()),
            "unexpected_mean": float(u.mean().item()),
            "expectation_suppression_percent": 100.0 * float((1.0 - e.mean()/u.mean()).item()),
        }
    # The same fixed decoder/noise table is used for both conditions at every time.
    generator = train_sweep.make_generator(device, DECODER_TEST_NOISE_SEED)
    noise = torch.randn((32, count, N), device=device, generator=generator,
                        dtype=rates.dtype) * checkpoint["references"]["sigma_train"]
    result["decoding"] = {}
    for name, condition in (("expected", rates[:count]), ("unexpected", rates[count:])):
        values = [temporal_current_metrics(
            net, condition[:, k].unsqueeze(0).expand_as(noise),
            finals.unsqueeze(0).expand(noise.shape[:-1]), noise,
            checkpoint["references"], generator,
        ) for k in range(steps)]
        result["decoding"][name] = {key: [row[key] for row in values] for key in values[0]}
    return result


def temporal_shape_gates(probe):
    early, late = probe["windows"]["early"], probe["windows"]["late"]
    return {
        "early_sharpening": (early["preferred_ratio"] > 1.02
                             and early["shoulder_ratio"] < 0.98
                             and early["flank_ratio"] < 0.98),
        "late_dampening": (late["preferred_ratio"] < 0.98
                           and late["center_rate"] <= 0.95 * late["flank15_rate"]
                           and late["expected_mean"] < early["expected_mean"]),
    }


def plot_temporal_assay(result, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    high = result["per_alpha"]["0.7"]
    low = result["per_alpha"]["0.07"]
    probe = high["probe"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, window in zip(axes[0], ("early", "late")):
        for field, color, style in (("baseline", "0.5", "--"),
                                    ("unexpected", "#dd8b3a", "-"),
                                    ("expected", "#b64944", "-")):
            ax.plot(probe["offset_degrees"], probe["windows"][window][field],
                    color=color, linestyle=style, label=field)
        ax.set(title=f"High pressure: {window}", xlim=(-60, 60),
               xlabel="Preferred orientation − stimulus (°)", ylabel="E activity")
        ax.legend(frameon=False, fontsize=8)
    for arm, color, label in ((low, "#236da8", "α=0.07"), (high, "#b64944", "α=0.70")):
        for condition, style in (("expected", "-"), ("unexpected", "--")):
            axes[1, 0].plot(arm["probe"]["times"],
                            arm["probe"]["decoding"][condition]["current_accuracy"],
                            color=color, linestyle=style, label=f"{label} {condition}")
        axes[1, 1].plot(arm["probe"]["cycle_times"],
                        arm["probe"]["cumulative_activity"], color=color, label=label)
    axes[1, 1].plot(high["clamped_probe"]["cycle_times"],
                    high["clamped_probe"]["cumulative_activity"],
                    color="0.3", linestyle="--", label="α=0.70, SST held after 0.2")
    axes[1, 0].set(xlabel="Time after onset (relative)", ylabel="Fixed-readout accuracy",
                   ylim=(0, 1))
    axes[1, 1].set(xlabel="Time after onset (relative)", ylabel="Cumulative activity proxy")
    axes[1, 1].axvline(4, color="0.7", linestyle=":")
    for ax in axes[1]:
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Temporal SST experiment · seed {result['metadata']['seed']}")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


@torch.no_grad()
def assay_temporal_run(args, device, output_path):
    if set(args.alphas) != {0.07, 0.70}:
        raise ValueError("the temporal experiment requires paired --alphas 0.07 0.70")
    common_net, common = load_arm(args.run_dir / "common_pretrain_final.pt", device)
    result = {
        "metadata": {
            "seed": common["seed"], "device": str(device),
            "architecture": common["model_architecture_version"],
            "temporal_protocol": common_net.temporal_protocol,
            "references": common["references"],
            "pair_count": 216, "held_out_batches": 8, "held_out_batch_size": 128,
            "held_out_sequence_length": 12, "held_out_mismatch_prob": 0.02,
            "held_out_data_seed": 930001, "held_out_noise_seed": 930002,
            "probe_noise_seed": DECODER_TEST_NOISE_SEED, "probe_noise_repeats": 32,
            "readout": "fixed_population_vector; two instantaneous early decisions",
            "activity_reference": 4.0 * common["references"]["R_ref"],
        },
        "common_pretrain": {
            "probe": temporal_probe(common_net, common, device),
            "held_out": temporal_held_out(common_net, common, device),
        },
        "per_alpha": {},
    }
    for alpha in (0.07, 0.70):
        path = args.run_dir / f"alpha_{alpha_slug(alpha)}_final.pt"
        net, checkpoint = load_arm(path, device)
        if (net.temporal_protocol != common_net.temporal_protocol
                or checkpoint["references"] != common["references"]
                or checkpoint["feedback_mode"] != common["feedback_mode"]):
            raise RuntimeError("temporal pair differs from common protocol/references/feedback")
        row = {"checkpoint": str(path), "probe": temporal_probe(net, checkpoint, device),
               "held_out": temporal_held_out(net, checkpoint, device)}
        fine_config = tuned.model_config(net)
        fine_config["temporal_protocol"] = dict(net.temporal_protocol, dt=0.05)
        fine = tuned.build_tuned_from_config(fine_config).to(device)
        fine.load_state_dict(net.state_dict(), strict=True)
        fine.eval()
        row["fine_held_out"] = temporal_held_out(fine, checkpoint, device)
        row["fine_probe"] = temporal_probe(fine, checkpoint, device, measurements=False)
        if alpha == 0.70:
            row["clamped_probe"] = temporal_probe(net, checkpoint, device, clamp=True,
                                                   measurements=False)
            row["fine_clamped_probe"] = temporal_probe(fine, checkpoint, device, clamp=True,
                                                        measurements=False)
        result["per_alpha"][alpha_tag(alpha)] = row
        print(f"alpha {alpha}: temporal assay complete", flush=True)
    low, high = result["per_alpha"]["0.07"], result["per_alpha"]["0.7"]
    saving = low["held_out"]["cycle_integral"] - high["held_out"]["cycle_integral"]
    error = sum(abs(row["held_out"]["cycle_integral"]
                    - row["fine_held_out"]["cycle_integral"]) for row in (low, high))
    normal_j = high["probe"]["cycle_integral"]["pooled"]
    clamp_j = high["clamped_probe"]["cycle_integral"]["pooled"]
    control_error = (abs(normal_j - high["fine_probe"]["cycle_integral"]["pooled"])
                     + abs(clamp_j - high["fine_clamped_probe"]["cycle_integral"]["pooled"]))
    gates = temporal_shape_gates(high["probe"])
    gates.update({
        "early_accuracy_preserved": high["held_out"]["current_accuracy"]
        >= low["held_out"]["current_accuracy"] - 0.05,
        "activity_saving_vs_low_pressure": saving > error,
        "activity_saving_vs_clamped_sst": clamp_j - normal_j > control_error,
    })
    result["acceptance"] = {
        "gates": gates, "all_pass": all(gates.values()),
        "common_pretrain_shape_gates": temporal_shape_gates(result["common_pretrain"]["probe"]),
        "early_accuracy_drop_percentage_points": 100.0 * (
            low["held_out"]["current_accuracy"] - high["held_out"]["current_accuracy"]),
        "activity_saving_percent": 100.0 * saving / low["held_out"]["cycle_integral"],
        "activity_resolution_difference_bound": error,
        "clamp_activity_saving_percent": 100.0 * (clamp_j - normal_j) / clamp_j,
        "clamp_resolution_difference_bound": control_error,
    }
    atomic_json_save(result, output_path)
    plot_temporal_assay(result, output_path)
    print(json.dumps(result["acceptance"], indent=2), flush=True)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="One seed directory containing common and alpha final checkpoints.",
    )
    parser.add_argument(
        "--device", default="auto", help="PyTorch device, for example cuda:0, cpu, or auto."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output JSON path; defaults to <run-dir>/assay.json.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(ALPHAS),
        help="Unique checkpoint alpha values to assay.",
    )
    args = parser.parse_args()
    if not args.alphas or len(args.alphas) != len(set(args.alphas)):
        parser.error("alphas must be a nonempty unique list")
    try:
        train_sweep.validate_unique_alpha_slugs(args.alphas)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_path = args.out or args.run_dir / "assay.json"
    construct_check = synthetic_construct_check(device)
    per_alpha = {}
    checkpoint_config = {}
    common_checkpoint = torch.load(
        args.run_dir / "common_pretrain_final.pt", map_location=device
    )
    if common_checkpoint.get("model_architecture_version") in (
        tuned.TEMPORAL_MODEL_ARCHITECTURE_VERSION,
        tuned.LEARNED_TEMPORAL_MODEL_ARCHITECTURE_VERSION,
    ):
        assay_temporal_run(args, device, output_path)
        return
    common_local_comp_raw = common_checkpoint["state_dict"].get(
        "local_comp_strength_raw"
    )
    common_center_feedback = bool(
        common_checkpoint.get("center_feedback", False)
    )
    common_feedback_mode = tuned.resolve_feedback_mode(
        common_center_feedback,
        common_checkpoint.get("feedback_mode"),
    )
    for alpha in args.alphas:
        alpha_key = alpha_tag(alpha)
        checkpoint_path = args.run_dir / f"alpha_{alpha_slug(alpha)}_final.pt"
        metrics, execution_config = assay_arm(
            checkpoint_path, device, common_local_comp_raw
        )
        if execution_config["feedback_mode"] != common_feedback_mode:
            raise RuntimeError(
                "axis checkpoint feedback mode differs from common pretrain"
            )
        per_alpha[alpha_key] = metrics
        checkpoint_config[alpha_key] = execution_config
    result = {
        "metadata": {
            "device": str(device),
            "pair_count": N * len(VELOCITIES),
            "final_channels": N,
            "signed_transition_velocities": list(VELOCITIES),
            "condition_a_history": "[y-4v,y-3v,y-2v,y-v,y] mod 36",
            "condition_b_history": "[y+2v,y+v,y,y-v,y] mod 36",
            "condition_a_label": "expected",
            "condition_b_label": "unexpected",
            "held_out_decoding_protocol": {
                "train_repeats_per_pair_condition": DECODER_TRAIN_REPEATS,
                "test_repeats_per_pair_condition": DECODER_TEST_REPEATS,
                "train_noise_seed": DECODER_TRAIN_NOISE_SEED,
                "test_noise_seed": DECODER_TEST_NOISE_SEED,
                "splits": "disjoint_independent_fixed_noise_generators",
                "paired_conditions": "same_noise_table_for_A_and_B_within_each_split",
                "fit": "one_balanced_condition_blind_pooled_A_plus_B_cosine_nearest_centroid",
                "features": "feedback_on_final_L23_rectified_additive_noise_then_per_trial_L1_L2",
            },
            "synthetic_construct_check": construct_check,
            "checkpoint_execution_config": checkpoint_config,
        },
        "per_alpha": per_alpha,
    }
    atomic_json_save(result, output_path)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
