"""Stage-3 task-definition helpers for mixed-offset complementary pair-set generalization.

This module is intentionally additive. It does not modify the frozen Stage-0/1/2
metrics, learner, or scaffold. Stage-3 here is oracle-only task-definition
execution over the existing local-global probe scaffold.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import torch

from .diagnostics import (
    DIAGNOSTIC_CLASSIFICATION_RULE_VERSION,
    DIAGNOSTIC_METRIC_SCHEMA_VERSION,
    DIAGNOSTIC_SCHEMA_VERSION,
    HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
    HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT,
    build_hidden_state_diagnostics_v2,
    build_hidden_state_probe_table_rows_from_payload,
    metric_versions_v2_payload,
    write_hidden_state_diagnostics_v2_json,
    write_hidden_state_probe_table_v2,
)
from .metrics import template_specificity
from .provenance import BENCHMARK_REGISTRY_VERSION, FROZEN_BENCHMARK_METRIC_VERSION
from .types import ContextPrediction, SimulationOutput, TrialBatch


PRIMARY_SEED_PANEL = (
    (101, 10101),
    (202, 10202),
    (303, 10303),
    (404, 10404),
    (505, 10505),
)
CONFIRMATION_SEED_PANEL = (
    (606, 10606),
    (707, 10707),
    (808, 10808),
    (909, 10909),
    (1001, 11001),
)
FULL_CONTROLLED_SOURCES = [0, 1, 2, 3, 4, 5]

STAGE3_TASK_VERSION = "2026-04-01.stage3.task-definition.v2"
STAGE3_METRIC_SCHEMA_VERSION = "2026-04-01.stage3.metric-schema.v2"
STAGE3_CLASSIFICATION_RULE_VERSION = "2026-04-01.stage3.task-definition.rules.v2"

TASK_DEFINITION_MANIFEST_V2_ARTIFACT = "task_definition_manifest.v2.json"
SUPPORT_TABLE_V2_ARTIFACT = "support_table.v2.csv"
SUPPORT_SUMMARY_V2_ARTIFACT = "support_summary.v2.json"
SHORTCUT_BASELINE_REPORT_V2_ARTIFACT = "shortcut_baseline_report.v2.json"
ORACLE_TASK_FEASIBILITY_REPORT_V2_ARTIFACT = "oracle_task_feasibility_report.v2.json"
TASK_DEFINITION_VERDICT_V2_ARTIFACT = "task_definition_verdict.v2.json"

STAGE3_DIRECTION_SPECS = {
    "splitA_to_splitB": {
        "train_split": "splitA",
        "eval_split": "splitB",
    },
    "splitB_to_splitA": {
        "train_split": "splitB",
        "eval_split": "splitA",
    },
}

SUPPORT_TABLE_FIELDS = (
    "direction_id",
    "seed_panel",
    "split_role",
    "pair_id",
    "symmetry_mate_pair_id",
    "source_orientation",
    "target_orientation",
    "expected_target_orientation",
    "local_offset_family",
    "context_id",
    "global_expected",
    "visible_step_index",
    "task_mode",
    "prestim_mode",
    "controlled_source_step",
    "n_rows",
    "n_unique_seeds",
)

_SOURCE_GROUPS = ((0, 1), (2, 3), (4, 5))
_SPLIT_A_SELECTION_PATTERNS = (
    (1, 0, 1, 0, 1, 0),
    (0, 1, 1, 0, 1, 0),
    (1, 0, 0, 1, 1, 0),
    (1, 0, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 1),
)


def stage3_metadata(
    *,
    metric_versions: dict[str, str],
    source_metric_versions: list[str],
) -> dict[str, object]:
    return {
        "metric_schema_version": STAGE3_METRIC_SCHEMA_VERSION,
        "metric_versions": metric_versions,
        "classification_rule_version": STAGE3_CLASSIFICATION_RULE_VERSION,
        "source_metric_versions": source_metric_versions,
        "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
        "stage3_task_version": STAGE3_TASK_VERSION,
    }


def _pattern_for_index(schedule_index: int) -> tuple[int, ...]:
    return _SPLIT_A_SELECTION_PATTERNS[schedule_index % len(_SPLIT_A_SELECTION_PATTERNS)]


def build_stage3_split_definition(schedule_index: int) -> dict[str, dict[str, object]]:
    """Return complementary splitA/splitB pair sets for a deterministic schedule index."""

    pattern = _pattern_for_index(schedule_index)
    split_a_pair_ids: list[int] = []
    split_b_pair_ids: list[int] = []
    split_a_symmetry_mates: dict[int, int] = {}
    split_b_symmetry_mates: dict[int, int] = {}

    for left_source, right_source in _SOURCE_GROUPS:
        split_a_group: list[int] = []
        split_b_group: list[int] = []
        for source_orientation in (left_source, right_source):
            use_plus_family = bool(pattern[source_orientation])
            split_a_pair_id = source_orientation * 2 + (0 if use_plus_family else 1)
            split_b_pair_id = source_orientation * 2 + (1 if use_plus_family else 0)
            split_a_group.append(split_a_pair_id)
            split_b_group.append(split_b_pair_id)
        split_a_pair_ids.extend(split_a_group)
        split_b_pair_ids.extend(split_b_group)
        split_a_symmetry_mates[split_a_group[0]] = split_a_group[1]
        split_a_symmetry_mates[split_a_group[1]] = split_a_group[0]
        split_b_symmetry_mates[split_b_group[0]] = split_b_group[1]
        split_b_symmetry_mates[split_b_group[1]] = split_b_group[0]

    assert len(split_a_pair_ids) == 6
    assert len(split_b_pair_ids) == 6
    assert len(set(split_a_pair_ids)) == 6
    assert len(set(split_b_pair_ids)) == 6
    assert set(split_a_pair_ids).isdisjoint(split_b_pair_ids)
    assert set(split_a_pair_ids) | set(split_b_pair_ids) == set(range(12))

    return {
        "splitA": {
            "pair_ids": tuple(split_a_pair_ids),
            "symmetry_mates": split_a_symmetry_mates,
        },
        "splitB": {
            "pair_ids": tuple(split_b_pair_ids),
            "symmetry_mates": split_b_symmetry_mates,
        },
    }


def get_direction_split_specs(
    *,
    direction_id: str,
    schedule_index: int,
) -> dict[str, dict[str, object]]:
    split_definition = build_stage3_split_definition(schedule_index)
    direction_spec = STAGE3_DIRECTION_SPECS[direction_id]
    return {
        "train": split_definition[direction_spec["train_split"]],
        "eval": split_definition[direction_spec["eval_split"]],
    }


def build_stage3_support_rows(
    batch: TrialBatch,
    *,
    direction_id: str,
    seed_panel: str,
    split_role: str,
    run_index: int,
    seed_value: int,
    symmetry_mate_lookup: dict[int, int],
) -> list[dict[str, object]]:
    probe_mask = batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]
    probe_indices = torch.nonzero(probe_mask, as_tuple=False)
    task_mode_names = batch.metadata["task_mode_names"]
    prestim_mode_names = batch.metadata["prestim_mode_names"]

    rows: list[dict[str, object]] = []
    for trial_idx_tensor, step_idx_tensor in probe_indices:
        trial_idx = int(trial_idx_tensor.item())
        step_idx = int(step_idx_tensor.item())
        pair_id = int(batch.metadata["probe_pair_id"][trial_idx].item())
        source_orientation = int(batch.metadata["probe_source_orientation"][trial_idx].item())
        target_orientation = int(batch.metadata["probe_target_orientation"][trial_idx].item())
        rows.append(
            {
                "direction_id": direction_id,
                "seed_panel": seed_panel,
                "split_role": split_role,
                "pair_id": pair_id,
                "symmetry_mate_pair_id": int(symmetry_mate_lookup[pair_id]),
                "source_orientation": source_orientation,
                "target_orientation": target_orientation,
                # Stage-3 support labels are defined by the paired target orientation.
                "expected_target_orientation": target_orientation,
                "local_offset_family": int(batch.metadata["probe_local_offset_bins"][trial_idx].item()),
                "context_id": int(batch.context_ids[trial_idx].item()),
                "global_expected": bool(batch.metadata["probe_global_expected_mask"][trial_idx, step_idx].item()),
                "visible_step_index": int(batch.metadata["visible_step_index"][trial_idx, step_idx].item()),
                "task_mode": task_mode_names[int(batch.task_mode[trial_idx].item())],
                "prestim_mode": prestim_mode_names[int(batch.prestim_mode[trial_idx].item())],
                "controlled_source_step": bool(batch.metadata["controlled_source_steps"][trial_idx, step_idx].item()),
                "_run_index": run_index,
                "_seed_value": seed_value,
                "_is_blank": bool(batch.blank_mask[trial_idx, step_idx].item()),
            }
        )
    return rows


def aggregate_stage3_support_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    group_fields = tuple(field for field in SUPPORT_TABLE_FIELDS if field not in {"n_rows", "n_unique_seeds"})
    for row in raw_rows:
        key = tuple(row[field] for field in group_fields)
        if key not in grouped:
            grouped[key] = {
                field: row[field] for field in group_fields
            }
            grouped[key]["_seed_values"] = set()
            grouped[key]["n_rows"] = 0
        grouped[key]["_seed_values"].add(int(row["_seed_value"]))
        grouped[key]["n_rows"] += 1

    aggregated_rows: list[dict[str, object]] = []
    for payload in grouped.values():
        aggregated_row = {field: payload[field] for field in group_fields}
        aggregated_row["n_rows"] = int(payload["n_rows"])
        aggregated_row["n_unique_seeds"] = int(len(payload["_seed_values"]))
        aggregated_rows.append(aggregated_row)

    aggregated_rows.sort(
        key=lambda row: (
            row["direction_id"],
            row["seed_panel"],
            row["split_role"],
            row["pair_id"],
            row["context_id"],
            row["global_expected"],
        )
    )
    return aggregated_rows


def _rows_by_run_and_role(
    raw_rows: list[dict[str, object]],
) -> dict[tuple[int, str], list[dict[str, object]]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((int(row["_run_index"]), str(row["split_role"])), []).append(row)
    return grouped


def _full_source_set_present(rows: list[dict[str, object]]) -> bool:
    return {int(row["source_orientation"]) for row in rows} == set(FULL_CONTROLLED_SOURCES)


def _both_offset_families_present(rows: list[dict[str, object]]) -> bool:
    counts = {int(row["local_offset_family"]) for row in rows}
    return counts == {-1, 1}


def _contexts_balanced(rows: list[dict[str, object]]) -> bool:
    counts = {0: 0, 1: 0}
    for row in rows:
        counts[int(row["context_id"])] += 1
    return counts[0] == counts[1] and counts[0] > 0


def _expected_unexpected_balanced(rows: list[dict[str, object]]) -> bool:
    expected_rows = sum(1 for row in rows if bool(row["global_expected"]))
    unexpected_rows = sum(1 for row in rows if not bool(row["global_expected"]))
    return expected_rows == unexpected_rows and expected_rows > 0


def _symmetry_mates_complete(rows: list[dict[str, object]]) -> bool:
    pair_ids = {int(row["pair_id"]) for row in rows}
    return all(int(row["symmetry_mate_pair_id"]) in pair_ids for row in rows)


def _source_context_ambiguous(rows: list[dict[str, object]]) -> bool:
    grouped: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        key = (int(row["source_orientation"]), int(row["context_id"]))
        grouped.setdefault(key, set()).add(int(row["expected_target_orientation"]))
    return bool(grouped) and all(len(targets) >= 2 for targets in grouped.values())


def build_stage3_support_summary(raw_rows: list[dict[str, object]]) -> dict[str, object]:
    grouped = _rows_by_run_and_role(raw_rows)
    run_indices = sorted({int(row["_run_index"]) for row in raw_rows})
    train_rows = [row for row in raw_rows if row["split_role"] == "train"]
    eval_rows = [row for row in raw_rows if row["split_role"] == "eval"]

    all_six_sources_present_in_train = all(_full_source_set_present(grouped[(run_index, "train")]) for run_index in run_indices)
    all_six_sources_present_in_eval = all(_full_source_set_present(grouped[(run_index, "eval")]) for run_index in run_indices)
    both_offset_families_present_in_train = all(
        _both_offset_families_present(grouped[(run_index, "train")]) for run_index in run_indices
    )
    both_offset_families_present_in_eval = all(
        _both_offset_families_present(grouped[(run_index, "eval")]) for run_index in run_indices
    )
    train_eval_pair_disjoint = all(
        {
            int(row["pair_id"]) for row in grouped[(run_index, "train")]
        }.isdisjoint(
            {int(row["pair_id"]) for row in grouped[(run_index, "eval")]}
        )
        for run_index in run_indices
    )
    pair_sets_complementary = all(
        (
            {int(row["pair_id"]) for row in grouped[(run_index, "train")]}
            | {int(row["pair_id"]) for row in grouped[(run_index, "eval")]}
        ) == set(range(12))
        for run_index in run_indices
    ) and train_eval_pair_disjoint
    contexts_balanced = all(_contexts_balanced(grouped[(run_index, role)]) for run_index in run_indices for role in ("train", "eval"))
    expected_unexpected_balanced = all(
        _expected_unexpected_balanced(grouped[(run_index, role)]) for run_index in run_indices for role in ("train", "eval")
    )
    visible_step_values = {int(row["visible_step_index"]) for row in raw_rows}
    task_values = {str(row["task_mode"]) for row in raw_rows}
    prestim_values = {str(row["prestim_mode"]) for row in raw_rows}
    controlled_source_values = {int(row["source_orientation"]) for row in raw_rows}

    return {
        "all_six_sources_present_in_train": all_six_sources_present_in_train,
        "all_six_sources_present_in_eval": all_six_sources_present_in_eval,
        "both_offset_families_present_in_train": both_offset_families_present_in_train,
        "both_offset_families_present_in_eval": both_offset_families_present_in_eval,
        "train_eval_pair_disjoint": train_eval_pair_disjoint,
        "pair_sets_complementary": pair_sets_complementary,
        "source_context_ambiguous_in_train": _source_context_ambiguous(train_rows),
        "source_context_ambiguous_in_eval": _source_context_ambiguous(eval_rows),
        "symmetry_mates_complete_in_train": all(
            _symmetry_mates_complete(grouped[(run_index, "train")]) for run_index in run_indices
        ),
        "symmetry_mates_complete_in_eval": all(
            _symmetry_mates_complete(grouped[(run_index, "eval")]) for run_index in run_indices
        ),
        "contexts_balanced": contexts_balanced,
        "expected_unexpected_balanced": expected_unexpected_balanced,
        "visible_step_fixed": visible_step_values == {1},
        "task_fixed_orientation_relevant": task_values == {"orientation_relevant"},
        "prestim_fixed_none": prestim_values == {"none"},
        "controlled_sources_fixed": controlled_source_values == set(FULL_CONTROLLED_SOURCES),
        "no_blank_scored_rows": not any(bool(row["_is_blank"]) for row in raw_rows),
        "no_uncontrolled_scored_rows": all(bool(row["controlled_source_step"]) for row in raw_rows),
    }


def _evaluate_lookup_baseline(
    *,
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    key_fields: tuple[str, ...],
) -> dict[str, float]:
    counts_by_key: dict[tuple[object, ...], dict[int, int]] = {}
    fallback_counts: dict[int, int] = {}
    for row in train_rows:
        key = tuple(row[field] for field in key_fields)
        target = int(row["expected_target_orientation"])
        counts_by_key.setdefault(key, {})[target] = counts_by_key.setdefault(key, {}).get(target, 0) + 1
        fallback_counts[target] = fallback_counts.get(target, 0) + 1

    if not fallback_counts:
        return {
            "expected_target_top1_rate": 0.0,
            "correct_pair_flip_rate": 0.0,
        }

    fallback_target = max(sorted(fallback_counts), key=lambda key: (fallback_counts[key], -key))

    predictions: list[int] = []
    for row in eval_rows:
        key = tuple(row[field] for field in key_fields)
        if key in counts_by_key:
            predicted_target = max(
                sorted(counts_by_key[key]),
                key=lambda target: (counts_by_key[key][target], -target),
            )
        else:
            predicted_target = fallback_target
        predictions.append(int(predicted_target))

    correct_top1 = [
        int(prediction == int(row["expected_target_orientation"]))
        for prediction, row in zip(predictions, eval_rows)
    ]
    pair_rows: dict[int, dict[int, tuple[int, int]]] = {}
    for prediction, row in zip(predictions, eval_rows):
        pair_rows.setdefault(int(row["pair_id"]), {})[int(row["context_id"])] = (
            int(prediction),
            int(row["expected_target_orientation"]),
        )

    correct_pair_flip_values: list[float] = []
    for context_rows in pair_rows.values():
        if 0 not in context_rows or 1 not in context_rows:
            continue
        predicted_0, expected_0 = context_rows[0]
        predicted_1, expected_1 = context_rows[1]
        correct_pair_flip_values.append(
            float(
                predicted_0 == expected_0
                and predicted_1 == expected_1
                and predicted_0 != predicted_1
            )
        )
    correct_pair_flip_rate = (
        float(sum(correct_pair_flip_values) / len(correct_pair_flip_values))
        if correct_pair_flip_values
        else 0.0
    )
    return {
        "expected_target_top1_rate": float(sum(correct_top1) / len(correct_top1)) if correct_top1 else 0.0,
        "correct_pair_flip_rate": correct_pair_flip_rate,
    }


def evaluate_source_only_lookup(
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
) -> dict[str, float]:
    return _evaluate_lookup_baseline(
        train_rows=train_rows,
        eval_rows=eval_rows,
        key_fields=("source_orientation",),
    )


def evaluate_source_context_lookup(
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
) -> dict[str, float]:
    return _evaluate_lookup_baseline(
        train_rows=train_rows,
        eval_rows=eval_rows,
        key_fields=("source_orientation", "context_id"),
    )


def _baseline_summary(seedwise: list[dict[str, object]]) -> dict[str, float]:
    top1_rates = [float(row["expected_target_top1_rate"]) for row in seedwise]
    flip_rates = [float(row["correct_pair_flip_rate"]) for row in seedwise]
    return {
        "mean_expected_target_top1_rate": float(sum(top1_rates) / len(top1_rates)) if top1_rates else 0.0,
        "max_seedwise_expected_target_top1_rate": float(max(top1_rates)) if top1_rates else 0.0,
        "mean_correct_pair_flip_rate": float(sum(flip_rates) / len(flip_rates)) if flip_rates else 0.0,
        "max_seedwise_correct_pair_flip_rate": float(max(flip_rates)) if flip_rates else 0.0,
    }


def build_stage3_shortcut_baseline_report(
    *,
    direction_id: str,
    seed_panel: str,
    source_only_seedwise: list[dict[str, object]],
    source_context_seedwise: list[dict[str, object]],
) -> dict[str, object]:
    source_only_summary = _baseline_summary(source_only_seedwise)
    source_context_summary = _baseline_summary(source_context_seedwise)
    source_only_passes = (
        source_only_summary["mean_expected_target_top1_rate"] <= 0.55
        and source_only_summary["max_seedwise_expected_target_top1_rate"] <= 0.60
    )
    source_context_passes = (
        source_context_summary["mean_expected_target_top1_rate"] <= 0.55
        and source_context_summary["max_seedwise_expected_target_top1_rate"] <= 0.60
        and source_context_summary["mean_correct_pair_flip_rate"] <= 0.25
        and source_context_summary["max_seedwise_correct_pair_flip_rate"] <= 0.35
    )
    return {
        **stage3_metadata(
            metric_versions={
                "source_only_lookup_v1.expected_target_top1_rate": "v1",
                "source_only_lookup_v1.correct_pair_flip_rate": "v1",
                "source_context_lookup_v1.expected_target_top1_rate": "v1",
                "source_context_lookup_v1.correct_pair_flip_rate": "v1",
            },
            source_metric_versions=[STAGE3_TASK_VERSION],
        ),
        "direction_id": direction_id,
        "seed_panel": seed_panel,
        "baselines": {
            "source_only_lookup_v1": {
                "seedwise": source_only_seedwise,
                "summary": source_only_summary,
                "passes": source_only_passes,
            },
            "source_context_lookup_v1": {
                "seedwise": source_context_seedwise,
                "summary": source_context_summary,
                "passes": source_context_passes,
            },
        },
        "passes": source_only_passes and source_context_passes,
    }


def build_oracle_alignment_metrics(
    batch: TrialBatch,
    oracle_prediction: ContextPrediction,
) -> dict[str, float]:
    probe_mask = batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]
    expected_distribution = batch.metadata["expected_distribution"][probe_mask].to(torch.float32).clamp_min(1e-8)
    log_expected = expected_distribution.log()
    log_predicted = torch.log_softmax(oracle_prediction.orientation_logits[probe_mask], dim=-1)
    return {
        "oracle_probe_alignment_kl": float((expected_distribution * (log_expected - log_predicted)).sum(dim=-1).mean().item()),
        "oracle_probe_expected_logprob": float((expected_distribution * log_predicted).sum(dim=-1).mean().item()),
    }


def build_stage3_oracle_probe_table(
    batch: TrialBatch,
    oracle_simulation: SimulationOutput,
    symmetry_mate_lookup: dict[int, int],
) -> list[dict[str, object]]:
    probe_mask = batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]
    probe_indices = torch.nonzero(probe_mask, as_tuple=False)
    task_mode_names = batch.metadata["task_mode_names"]
    prestim_mode_names = batch.metadata["prestim_mode_names"]
    comparator = oracle_simulation.states["context_comparator"]
    pooled = oracle_simulation.observations.get("gaussian_orientation_bank")
    if pooled is None:
        pooled = oracle_simulation.observations.get("identity", oracle_simulation.states["l23_readout"])

    rows: list[dict[str, object]] = []
    for trial_idx_tensor, step_idx_tensor in probe_indices:
        trial_idx = int(trial_idx_tensor.item())
        step_idx = int(step_idx_tensor.item())
        target_orientation = int(batch.metadata["probe_target_orientation"][trial_idx].item())
        target_tensor = torch.tensor([target_orientation], dtype=torch.long)
        l23_row = oracle_simulation.states["l23_readout"][trial_idx, step_idx].unsqueeze(0)
        pooled_row = pooled[trial_idx, step_idx].unsqueeze(0)
        comparator_row = comparator[trial_idx, step_idx]
        pair_id = int(batch.metadata["probe_pair_id"][trial_idx].item())
        rows.append(
            {
                "trial_index": trial_idx,
                "step_index": step_idx,
                "pair_id": pair_id,
                "symmetry_mate_pair_id": int(symmetry_mate_lookup[pair_id]),
                "source_orientation": int(batch.metadata["probe_source_orientation"][trial_idx].item()),
                "target_orientation": target_orientation,
                "expected_target_orientation": target_orientation,
                "local_offset_family": int(batch.metadata["probe_local_offset_bins"][trial_idx].item()),
                "context_id": int(batch.context_ids[trial_idx].item()),
                "global_expected": bool(batch.metadata["probe_global_expected_mask"][trial_idx, step_idx].item()),
                "task_mode": task_mode_names[int(batch.task_mode[trial_idx].item())],
                "prestim_mode": prestim_mode_names[int(batch.prestim_mode[trial_idx].item())],
                "controlled_source_step": bool(batch.metadata["controlled_source_steps"][trial_idx, step_idx].item()),
                "oracle_l23_target_aligned_specificity": float(template_specificity(l23_row, target_tensor).item()),
                "oracle_pooled_target_aligned_specificity": float(template_specificity(pooled_row, target_tensor).item()),
                "oracle_context_comparator_nonuniformity": float(
                    (comparator_row - comparator_row.mean()).abs().mean().item()
                ),
            }
        )
    return rows


def _oracle_symmetry_consistency(hidden_state_rows: list[dict[str, object]]) -> bool:
    grouped: dict[tuple[int, int], dict[bool, float]] = {}
    for row in hidden_state_rows:
        key = (int(row["_symmetry_group_id"]), int(row["context_id"]))
        grouped.setdefault(key, {})[bool(row["global_expected"])] = float(row["l23_target_specificity__v2"])
    return bool(grouped) and all(
        True in values and False in values and values[True] > values[False]
        for values in grouped.values()
    )


def build_oracle_task_feasibility_report(
    *,
    direction_id: str,
    seed_panel: str,
    per_run_reports: list[dict[str, object]],
) -> dict[str, object]:
    latent_contrasts = [
        float(report["oracle_probe_metrics"]["probe_target_aligned_specificity_contrast"])
        for report in per_run_reports
    ]
    pooled_contrasts = [
        float(report["oracle_probe_metrics"]["probe_pooled_target_aligned_specificity_contrast"])
        for report in per_run_reports
    ]
    checks = {
        "oracle_alignment_kl_all_seeds": all(
            float(report["oracle_alignment"]["oracle_probe_alignment_kl"]) <= 1e-6
            for report in per_run_reports
        ),
        "latent_contrast_positive_all_seeds": all(value > 0.0 for value in latent_contrasts),
        "pooled_contrast_positive_all_seeds": all(value > 0.0 for value in pooled_contrasts),
        "oracle_symmetry_consistency_all_seeds": all(
            bool(report["oracle_symmetry_consistency"]) for report in per_run_reports
        ),
        "context0_localization_positive_all_seeds": all(
            report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__context0_v2"] is not None
            and float(report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__context0_v2"]) > 0.0
            for report in per_run_reports
        ),
        "context1_localization_positive_all_seeds": all(
            report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__context1_v2"] is not None
            and float(report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__context1_v2"]) > 0.0
            for report in per_run_reports
        ),
        "block_early_localization_positive_all_seeds": all(
            report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__block_early_v2"] is not None
            and float(report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__block_early_v2"]) > 0.0
            for report in per_run_reports
        ),
        "block_late_localization_positive_all_seeds": all(
            report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__block_late_v2"] is not None
            and float(report["hidden_state_diagnostics"]["probe_target_aligned_specificity_contrast__block_late_v2"]) > 0.0
            for report in per_run_reports
        ),
        "mean_oracle_latent_contrast_floor": (sum(latent_contrasts) / len(latent_contrasts)) >= 0.03927,
        "mean_oracle_pooled_contrast_floor": (sum(pooled_contrasts) / len(pooled_contrasts)) >= 0.00309,
        "design_invariants_all_seeds": all(
            int(report["probe_design_report"]["n_probe_rows"]) == 12
            and int(report["probe_design_report"]["n_probe_global_expected_rows"]) == 6
            and int(report["probe_design_report"]["n_probe_global_unexpected_rows"]) == 6
            and int(report["probe_design_report"]["n_probe_pairs_total"]) == 6
            and int(report["probe_design_report"]["n_probe_pairs_scored"]) == 6
            for report in per_run_reports
        ),
    }
    return {
        **stage3_metadata(
            metric_versions={
                "oracle_probe_alignment_kl": "v1",
                "probe_target_aligned_specificity_contrast": "v1",
                "probe_pooled_target_aligned_specificity_contrast": "v1",
                "probe_target_aligned_specificity_contrast__context0_v2": "v2",
                "probe_target_aligned_specificity_contrast__context1_v2": "v2",
                "probe_target_aligned_specificity_contrast__block_early_v2": "v2",
                "probe_target_aligned_specificity_contrast__block_late_v2": "v2",
                "oracle_symmetry_consistency": "v2",
            },
            source_metric_versions=[FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
        ),
        "direction_id": direction_id,
        "seed_panel": seed_panel,
        "n_runs": len(per_run_reports),
        "seedwise": [
            {
                "run_id": report["run_id"],
                "train_seed": int(report["train_seed"]),
                "eval_seed": int(report["eval_seed"]),
                "oracle_probe_alignment_kl": float(report["oracle_alignment"]["oracle_probe_alignment_kl"]),
                "probe_target_aligned_specificity_contrast": float(
                    report["oracle_probe_metrics"]["probe_target_aligned_specificity_contrast"]
                ),
                "probe_pooled_target_aligned_specificity_contrast": float(
                    report["oracle_probe_metrics"]["probe_pooled_target_aligned_specificity_contrast"]
                ),
                "oracle_symmetry_consistency": bool(report["oracle_symmetry_consistency"]),
                "probe_target_aligned_specificity_contrast__context0_v2": report["hidden_state_diagnostics"][
                    "probe_target_aligned_specificity_contrast__context0_v2"
                ],
                "probe_target_aligned_specificity_contrast__context1_v2": report["hidden_state_diagnostics"][
                    "probe_target_aligned_specificity_contrast__context1_v2"
                ],
                "probe_target_aligned_specificity_contrast__block_early_v2": report["hidden_state_diagnostics"][
                    "probe_target_aligned_specificity_contrast__block_early_v2"
                ],
                "probe_target_aligned_specificity_contrast__block_late_v2": report["hidden_state_diagnostics"][
                    "probe_target_aligned_specificity_contrast__block_late_v2"
                ],
            }
            for report in per_run_reports
        ],
        "summary": {
            "mean_oracle_probe_alignment_kl": float(
                sum(float(report["oracle_alignment"]["oracle_probe_alignment_kl"]) for report in per_run_reports)
                / len(per_run_reports)
            ),
            "mean_oracle_latent_contrast": float(sum(latent_contrasts) / len(latent_contrasts)),
            "mean_oracle_pooled_contrast": float(sum(pooled_contrasts) / len(pooled_contrasts)),
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def build_task_definition_manifest_payload(
    *,
    direction_id: str,
    seed_panel: str,
    seed_pairs: tuple[tuple[int, int], ...],
    run_ids: list[str],
) -> dict[str, object]:
    return {
        **stage3_metadata(
            metric_versions={"task_definition_manifest": "v2"},
            source_metric_versions=[STAGE3_TASK_VERSION],
        ),
        "direction_id": direction_id,
        "seed_panel": seed_panel,
        "subcase": "dampening",
        "task_mode_fixed_state": "orientation_relevant",
        "prestim_fixed_state": "none",
        "probe_visible_step_index": 1,
        "controlled_sources": FULL_CONTROLLED_SOURCES,
        "seed_pairs": [
            {"train_seed": int(train_seed), "eval_seed": int(eval_seed)}
            for train_seed, eval_seed in seed_pairs
        ],
        "run_ids": run_ids,
        "split_definition": "mixed_offset_complementary_pair_sets",
    }


def build_task_definition_verdict(
    *,
    panel_outputs: list[dict[str, object]],
) -> dict[str, object]:
    panel_results: list[dict[str, object]] = []
    passes = True
    for panel_output in panel_outputs:
        support_passes = all(
            bool(value)
            for key, value in panel_output["support_summary"].items()
            if isinstance(value, bool)
        )
        baseline_passes = bool(panel_output["shortcut_baseline_report"]["passes"])
        oracle_passes = bool(panel_output["oracle_task_feasibility_report"]["passes"])
        panel_passes = support_passes and baseline_passes and oracle_passes
        passes &= panel_passes
        panel_results.append(
            {
                "direction_id": panel_output["direction_id"],
                "seed_panel": panel_output["seed_panel"],
                "support_passes": support_passes,
                "shortcut_baseline_passes": baseline_passes,
                "oracle_feasibility_passes": oracle_passes,
                "passes": panel_passes,
                "output_dir": panel_output["output_dir"],
            }
        )
    return {
        **stage3_metadata(
            metric_versions={"task_definition_verdict": "v2"},
            source_metric_versions=[STAGE3_TASK_VERSION, FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
        ),
        "panel_results": panel_results,
        "passes": passes,
    }


def write_stage3_support_table(rows: list[dict[str, object]], output_path: str | Path) -> None:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUPPORT_TABLE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in SUPPORT_TABLE_FIELDS})


def write_stage3_hidden_state_artifacts(
    *,
    run_id: str,
    batch_payload: dict[str, object],
    oracle_simulation_payload: dict[str, object],
    symmetry_mate_lookup: dict[int, int],
    output_dir: str | Path,
) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "batch": batch_payload,
        "learned_logits": None,
        "oracle_logits": oracle_simulation_payload["context_predictions"],
        "learned_precision": None,
        "oracle_precision": oracle_simulation_payload["precision"],
        "l23_readout": oracle_simulation_payload["states"]["l23_readout"],
        "gaussian_orientation_bank": oracle_simulation_payload["observations"].get("gaussian_orientation_bank"),
        "context_comparator": oracle_simulation_payload["states"]["context_comparator"],
    }
    rows = build_hidden_state_probe_table_rows_from_payload(payload)
    for row in rows:
        pair_id = int(row["pair_id"])
        row["_symmetry_group_id"] = min(pair_id, int(symmetry_mate_lookup[pair_id]))

    diagnostics = build_hidden_state_diagnostics_v2(rows)
    diagnostics_payload = {
        **stage3_metadata(
            metric_versions=metric_versions_v2_payload(),
            source_metric_versions=[DIAGNOSTIC_SCHEMA_VERSION, FROZEN_BENCHMARK_METRIC_VERSION],
        ),
        "run_id": run_id,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        **diagnostics,
    }
    resolved_output_dir = Path(output_dir)
    write_hidden_state_probe_table_v2(rows, resolved_output_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT)
    write_hidden_state_diagnostics_v2_json(
        diagnostics_payload,
        resolved_output_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
    )
    return {
        "rows": rows,
        "diagnostics": diagnostics_payload,
        "oracle_symmetry_consistency": _oracle_symmetry_consistency(rows),
    }
