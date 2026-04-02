"""Primary metric helpers for held-out assays."""

from __future__ import annotations

import torch


def mean_response_delta(expected: torch.Tensor, unexpected: torch.Tensor) -> torch.Tensor:
    return unexpected.mean() - expected.mean()


def template_specificity(activity: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
    gathered = activity.gather(dim=-1, index=target_bins.unsqueeze(-1)).squeeze(-1)
    mean_other = (activity.sum(dim=-1) - gathered) / (activity.shape[-1] - 1)
    return gathered - mean_other


def prestimulus_template_specificity(activity: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
    return template_specificity(activity, target_bins)


def prestimulus_condition_report(
    activity: torch.Tensor,
    target_bins: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    n_steps = int(mask.sum().item())
    n_trials = int(mask.any(dim=1).sum().item())
    if n_steps == 0:
        return {
            "n_steps": 0,
            "n_trials": n_trials,
            "prestimulus_template_specificity": 0.0,
            "template_peak": 0.0,
        }
    masked_activity = activity[mask]
    masked_targets = target_bins[mask]
    return {
        "n_steps": n_steps,
        "n_trials": n_trials,
        "prestimulus_template_specificity": float(template_specificity(masked_activity, masked_targets).mean().item()),
        "template_peak": float(masked_activity.max(dim=-1).values.mean().item()),
    }


def masked_template_specificity(
    activity: torch.Tensor,
    target_bins: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return activity.new_tensor(0.0)
    return template_specificity(activity[mask], target_bins[mask]).mean()


def comparator_nonuniformity(
    comparator: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return comparator.new_tensor(0.0)
    masked = comparator[mask]
    return (masked - masked.mean(dim=-1, keepdim=True)).abs().mean()


def omission_specificity(activity: torch.Tensor, omission_targets: torch.Tensor) -> torch.Tensor:
    valid = omission_targets.ge(0)
    if not valid.any():
        return torch.tensor(0.0, dtype=activity.dtype)
    target = omission_targets[valid]
    valid_activity = activity[valid]
    return template_specificity(valid_activity, target).mean()


def simple_decoder_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=-1)
    return predictions.eq(labels).to(torch.float32).mean()


def trajectory_mean_response_delta(
    trajectories: torch.Tensor,
    expected_mask: torch.Tensor,
    unexpected_mask: torch.Tensor,
) -> torch.Tensor:
    expected = trajectories[expected_mask]
    unexpected = trajectories[unexpected_mask]
    return mean_response_delta(expected=expected, unexpected=unexpected)


def decoder_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for simple_decoder_accuracy."""

    return simple_decoder_accuracy(logits, labels)
