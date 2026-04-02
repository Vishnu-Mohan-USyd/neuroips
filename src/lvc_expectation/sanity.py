"""Generator sanity checks for nuisance control and predictive structure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .paradigms import CONDITION_CODES
from .types import TrialBatch


@dataclass
class SanityResult:
    accuracy: float
    n_examples: int


def _balanced_majority_accuracy(features: list[tuple[int, ...]], labels: list[int]) -> float:
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for feature, label in zip(features, labels):
        buckets[feature].append(label)
    class_totals: dict[int, int] = defaultdict(int)
    class_correct: dict[int, int] = defaultdict(int)
    for feature, label in zip(features, labels):
        majority = max(set(buckets[feature]), key=buckets[feature].count)
        class_totals[label] += 1
        class_correct[label] += int(label == majority)
    if not class_totals:
        return 0.0
    return sum(class_correct[label] / class_totals[label] for label in class_totals) / len(class_totals)


def _collect_examples(batch: TrialBatch) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[int]]:
    nuisance_features: list[tuple[int, ...]] = []
    predictive_features: list[tuple[int, ...]] = []
    labels: list[int] = []
    condition_codes = batch.metadata["condition_codes"]
    for trial_idx in range(batch.orientations.shape[0]):
        for step in range(batch.orientations.shape[1]):
            condition = int(condition_codes[trial_idx, step].item())
            if condition not in (
                CONDITION_CODES["expected"],
                CONDITION_CODES["unexpected"],
                CONDITION_CODES["neutral"],
            ):
                continue
            nuisance_features.append(
                (
                    int(batch.task_mode[trial_idx].item()),
                    int(batch.prestim_mode[trial_idx].item()),
                    int(batch.blank_mask[trial_idx, step].item()),
                    int(batch.orthogonal_events[trial_idx, step].item()),
                )
            )
            predictive_features.append(
                (
                    int(batch.metadata["source_orientations"][trial_idx, step].item()),
                    int(batch.context_ids[trial_idx].item()),
                    int(batch.metadata["visible_step_index"][trial_idx, step].item()),
                )
            )
            labels.append(condition)
    return nuisance_features, predictive_features, labels


def nuisance_only_decoder(batch: TrialBatch) -> SanityResult:
    return run_nuisance_only_failure_test(batch)


def run_nuisance_only_failure_test(batch: TrialBatch) -> SanityResult:
    nuisance_features, _, labels = _collect_examples(batch)
    return SanityResult(accuracy=_balanced_majority_accuracy(nuisance_features, labels), n_examples=len(labels))


def predictive_structure_decoder(batch: TrialBatch) -> SanityResult:
    return run_predictive_structure_success_test(batch)


def run_predictive_structure_success_test(batch: TrialBatch) -> SanityResult:
    _, predictive_features, labels = _collect_examples(batch)
    return SanityResult(accuracy=_balanced_majority_accuracy(predictive_features, labels), n_examples=len(labels))
