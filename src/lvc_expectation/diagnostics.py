"""Stage-2 hidden-state diagnostics over archived local-global probe artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import statistics

import torch

from .metrics import template_specificity


DIAGNOSTIC_SCHEMA_VERSION = "stage2_hidden_state_diagnostics_v2"
DIAGNOSTIC_METRIC_SCHEMA_VERSION = "stage2_metric_schema_v2"
DIAGNOSTIC_CLASSIFICATION_RULE_VERSION = "stage2_diagnostic_calibration_rules_v1"
HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT = "hidden_state_probe_table.v2.csv"
HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT = "hidden_state_diagnostics.v2.json"

HIDDEN_STATE_PROBE_TABLE_FIELDS = (
    "run_id",
    "trial_index",
    "step_index",
    "pair_id",
    "context_id",
    "global_expected",
    "context_bin__v2",
    "block_position_bin__v2",
    "learned_expected_target_margin__v2",
    "oracle_expected_target_margin__v2",
    "learned_expected_target_confidence__v2",
    "oracle_expected_target_confidence__v2",
    "learned_top1_confidence__v2",
    "oracle_top1_confidence__v2",
    "learned_precision_gap__v2",
    "oracle_precision_gap__v2",
    "l23_target_specificity__v2",
    "pooled_target_specificity__v2",
    "context_comparator_nonuniformity__v2",
)

HIDDEN_STATE_DIAGNOSTIC_METRICS_V2 = (
    "learned_probe_expected_target_margin__v2",
    "oracle_probe_expected_target_margin__v2",
    "learned_probe_expected_target_confidence__v2",
    "oracle_probe_expected_target_confidence__v2",
    "learned_probe_top1_confidence__v2",
    "oracle_probe_top1_confidence__v2",
    "learned_probe_precision_gap__v2",
    "oracle_probe_precision_gap__v2",
    "probe_target_aligned_specificity_contrast__context0_v2",
    "probe_target_aligned_specificity_contrast__context1_v2",
    "probe_target_aligned_specificity_contrast__block_early_v2",
    "probe_target_aligned_specificity_contrast__block_late_v2",
    "probe_within_pair_mass__v2",
    "probe_correct_pair_flip_rate__v2",
    "probe_source_bin_top1__v2",
    "probe_source_bin_kl__v2",
    "probe_collapse_index__v2",
    "probe_maxprob_correct_gap__v2",
    "probe_entropy_correct_gap__v2",
    "probe_within_pair_mass_correct_gap__v2",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_scoring_rows(batch_payload: dict[str, object]) -> torch.Tensor:
    metadata = batch_payload["metadata"]
    return torch.nonzero(
        metadata["probe_step_mask"] & metadata["probe_valid_mask"],
        as_tuple=False,
    )


def _resolve_subcase_payload(
    serialized_simulations: dict[str, dict[str, object]],
    subcase: str,
) -> dict[str, object]:
    if subcase in serialized_simulations:
        return serialized_simulations[subcase]
    if len(serialized_simulations) == 1:
        return next(iter(serialized_simulations.values()))
    raise KeyError(f"subcase {subcase!r} not present in serialized simulation payload")


def _precision_confidence_gap(
    precision_tensor: torch.Tensor | None,
    trial_index: int,
    step_index: int,
    top1_confidence: float | None,
) -> float | None:
    if top1_confidence is None:
        return None
    if precision_tensor is None:
        return 0.5 - top1_confidence
    precision_value = float(torch.sigmoid(precision_tensor[trial_index, step_index, 0]).item())
    return precision_value - top1_confidence


def _row_prediction_metrics(
    *,
    logits: torch.Tensor | None,
    precision: torch.Tensor | None,
    trial_index: int,
    step_index: int,
    expected_distribution: torch.Tensor,
    source_bin: int,
) -> dict[str, float | int]:
    expected_bin = int(torch.argmax(expected_distribution).item())
    if logits is None:
        return {
            "expected_bin": expected_bin,
            "expected_target_margin": None,
            "expected_target_confidence": None,
            "top1_confidence": None,
            "top1_bin": None,
            "precision_gap": None,
            "within_pair_mass": None,
            "source_bin_top1": None,
            "source_bin_kl": None,
            "entropy": None,
            "correct_expected_target": None,
        }
    probabilities = torch.softmax(logits[trial_index, step_index].to(torch.float32), dim=-1)
    support_bins = torch.nonzero(expected_distribution.gt(0), as_tuple=False).squeeze(-1)
    support_bin_indices = [int(item) for item in support_bins.tolist()]
    non_expected_support_bins = [item for item in support_bin_indices if item != expected_bin]
    if non_expected_support_bins:
        competitor_probability = max(float(probabilities[item].item()) for item in non_expected_support_bins)
    else:
        competitor_probability = float(
            torch.cat((probabilities[:expected_bin], probabilities[expected_bin + 1 :])).max().item()
        )
    top1_confidence = float(probabilities.max().item())
    top1_bin = int(torch.argmax(probabilities).item())
    within_pair_mass = float(probabilities[support_bins].sum().item()) if support_bins.numel() else 0.0
    entropy = float((-(probabilities * probabilities.clamp_min(1e-8).log()).sum()).item())
    expected_target_confidence = float(probabilities[expected_bin].item())
    return {
        "expected_bin": expected_bin,
        "expected_target_margin": expected_target_confidence - competitor_probability,
        "expected_target_confidence": expected_target_confidence,
        "top1_confidence": top1_confidence,
        "top1_bin": top1_bin,
        "precision_gap": _precision_confidence_gap(precision, trial_index, step_index, top1_confidence),
        "within_pair_mass": within_pair_mass,
        "source_bin_top1": int(top1_bin == int(source_bin)),
        "source_bin_kl": float(-torch.log(probabilities[int(source_bin)].clamp_min(1e-8)).item()),
        "entropy": entropy,
        "correct_expected_target": int(top1_bin == expected_bin),
    }


def _row_state_metrics(
    *,
    run_payload: dict[str, object],
    trial_index: int,
    step_index: int,
    target_bin: int,
) -> dict[str, float | None]:
    if run_payload["l23_readout"] is None:
        return {
            "l23_target_specificity": None,
            "pooled_target_specificity": None,
            "context_comparator_nonuniformity": None,
        }
    l23_row = run_payload["l23_readout"][trial_index, step_index].unsqueeze(0)
    pooled_row = run_payload["gaussian_orientation_bank"][trial_index, step_index].unsqueeze(0)
    comparator_row = run_payload["context_comparator"][trial_index, step_index]
    target_tensor = torch.tensor([target_bin], dtype=torch.long, device=l23_row.device)
    return {
        "l23_target_specificity": float(template_specificity(l23_row, target_tensor).item()),
        "pooled_target_specificity": float(template_specificity(pooled_row, target_tensor).item()),
        "context_comparator_nonuniformity": float(
            (comparator_row - comparator_row.mean()).abs().mean().item()
        ),
    }


def load_probe_run_payload(
    run_dir: str | Path,
    *,
    subcase: str = "dampening",
) -> dict[str, object]:
    """Load the saved tensors needed for Stage-2 probe diagnostics."""

    resolved_run_dir = Path(run_dir)
    batch_payload = torch.load(resolved_run_dir / "eval" / "heldout_batch.pt")
    manifest = _read_json(resolved_run_dir / "manifest.json")

    learned_logits: torch.Tensor
    oracle_logits: torch.Tensor
    learned_precision: torch.Tensor | None
    oracle_precision: torch.Tensor | None
    l23_readout: torch.Tensor | None = None
    pooled_readout: torch.Tensor | None = None
    context_comparator: torch.Tensor | None = None

    if (resolved_run_dir / "eval" / "full_trajectories.pt").exists():
        learned_serialized = torch.load(resolved_run_dir / "eval" / "full_trajectories.pt")
        oracle_serialized = torch.load(resolved_run_dir / "eval" / "oracle_full_trajectories.pt")
        learned_run = _resolve_subcase_payload(learned_serialized, subcase)
        oracle_run = _resolve_subcase_payload(oracle_serialized, subcase)
        learned_logits = learned_run["context_predictions"]
        oracle_logits = oracle_run["context_predictions"]
        learned_precision = learned_run["precision"]
        oracle_precision = oracle_run["precision"]
        l23_readout = learned_run["states"]["l23_readout"]
        pooled_readout = learned_run["observations"]["gaussian_orientation_bank"]
        context_comparator = learned_run["states"]["context_comparator"]
    else:
        prediction_payload = torch.load(resolved_run_dir / "eval" / "probe_context_predictions.pt")
        learned_logits = prediction_payload["learned"]["orientation_logits"]
        oracle_logits = prediction_payload["oracle"]["orientation_logits"]
        learned_precision = prediction_payload["learned"]["precision_logit"]
        oracle_precision = prediction_payload["oracle"]["precision_logit"]

    return {
        "run_id": manifest["run_id"],
        "manifest": manifest,
        "batch": batch_payload,
        "learned_logits": learned_logits,
        "oracle_logits": oracle_logits,
        "learned_precision": learned_precision,
        "oracle_precision": oracle_precision,
        "l23_readout": l23_readout,
        "gaussian_orientation_bank": pooled_readout,
        "context_comparator": context_comparator,
    }


def build_hidden_state_probe_table_rows(
    run_dir: str | Path,
    *,
    subcase: str = "dampening",
) -> list[dict[str, object]]:
    """Build row-level Stage-2 probe diagnostics from a saved run directory."""

    payload = load_probe_run_payload(run_dir, subcase=subcase)
    return build_hidden_state_probe_table_rows_from_payload(payload)


def build_hidden_state_probe_table_rows_from_payload(
    payload: dict[str, object],
) -> list[dict[str, object]]:
    """Build Stage-2 probe-diagnostic rows from an explicit tensor payload."""

    batch = payload["batch"]
    metadata = batch["metadata"]
    probe_rows = _probe_scoring_rows(batch)
    pair_descriptors = metadata["probe_report"]["pair_descriptors"]
    total_probe_rows = int(probe_rows.shape[0])
    rows: list[dict[str, object]] = []

    for row_ordinal, (trial_index_tensor, step_index_tensor) in enumerate(probe_rows):
        trial_index = int(trial_index_tensor.item())
        step_index = int(step_index_tensor.item())
        context_id = int(batch["context_ids"][trial_index].item())
        pair_id = int(metadata["probe_pair_id"][trial_index].item())
        source_bin = int(metadata["probe_source_orientation"][trial_index].item())
        target_bin = int(metadata["probe_target_orientation"][trial_index].item())
        expected_distribution = metadata["expected_distribution"][trial_index, step_index].to(torch.float32)
        learned_prediction = _row_prediction_metrics(
            logits=payload["learned_logits"],
            precision=payload["learned_precision"],
            trial_index=trial_index,
            step_index=step_index,
            expected_distribution=expected_distribution,
            source_bin=source_bin,
        )
        oracle_prediction = _row_prediction_metrics(
            logits=payload["oracle_logits"],
            precision=payload["oracle_precision"],
            trial_index=trial_index,
            step_index=step_index,
            expected_distribution=expected_distribution,
            source_bin=source_bin,
        )
        state_metrics = _row_state_metrics(
            run_payload=payload,
            trial_index=trial_index,
            step_index=step_index,
            target_bin=target_bin,
        )
        block_position_bin = "early" if row_ordinal < max(1, total_probe_rows // 2) else "late"
        rows.append(
            {
                "run_id": payload["run_id"],
                "trial_index": trial_index,
                "step_index": step_index,
                "pair_id": pair_id,
                "context_id": context_id,
                "global_expected": bool(metadata["probe_global_expected_mask"][trial_index, step_index].item()),
                "context_bin__v2": f"context{context_id}",
                "block_position_bin__v2": block_position_bin,
                "learned_expected_target_margin__v2": learned_prediction["expected_target_margin"],
                "oracle_expected_target_margin__v2": oracle_prediction["expected_target_margin"],
                "learned_expected_target_confidence__v2": learned_prediction["expected_target_confidence"],
                "oracle_expected_target_confidence__v2": oracle_prediction["expected_target_confidence"],
                "learned_top1_confidence__v2": learned_prediction["top1_confidence"],
                "oracle_top1_confidence__v2": oracle_prediction["top1_confidence"],
                "learned_precision_gap__v2": learned_prediction["precision_gap"],
                "oracle_precision_gap__v2": oracle_prediction["precision_gap"],
                "l23_target_specificity__v2": state_metrics["l23_target_specificity"],
                "pooled_target_specificity__v2": state_metrics["pooled_target_specificity"],
                "context_comparator_nonuniformity__v2": state_metrics["context_comparator_nonuniformity"],
                "_expected_bin": learned_prediction["expected_bin"],
                "_learned_top1_bin": learned_prediction["top1_bin"],
                "_oracle_top1_bin": oracle_prediction["top1_bin"],
                "_learned_within_pair_mass": learned_prediction["within_pair_mass"],
                "_oracle_within_pair_mass": oracle_prediction["within_pair_mass"],
                "_learned_source_bin_top1": learned_prediction["source_bin_top1"],
                "_learned_source_bin_kl": learned_prediction["source_bin_kl"],
                "_learned_entropy": learned_prediction["entropy"],
                "_learned_correct_expected_target": learned_prediction["correct_expected_target"],
                "_source_bin": source_bin,
                "_target_bin": target_bin,
                "_symmetry_group_id": min(
                    pair_id,
                    int(pair_descriptors[str(pair_id)]["symmetry_mate_pair_id"]),
                ),
            }
        )
    return rows


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _correct_gap(
    rows: list[dict[str, object]],
    value_key: str,
) -> float | None:
    correct_values = [
        float(row[value_key])
        for row in rows
        if row[value_key] is not None and row["_learned_correct_expected_target"] == 1
    ]
    incorrect_values = [
        float(row[value_key])
        for row in rows
        if row[value_key] is not None and row["_learned_correct_expected_target"] == 0
    ]
    if not correct_values or not incorrect_values:
        return None
    return float(sum(correct_values) / len(correct_values) - sum(incorrect_values) / len(incorrect_values))


def _pair_flip_rate(rows: list[dict[str, object]], *, require_correct: bool) -> float | None:
    pair_context_rows: dict[int, dict[int, dict[str, object]]] = {}
    for row in rows:
        pair_context_rows.setdefault(int(row["pair_id"]), {})[int(row["context_id"])] = row

    pair_values: list[float] = []
    for context_rows in pair_context_rows.values():
        if 0 not in context_rows or 1 not in context_rows:
            continue
        context0 = context_rows[0]
        context1 = context_rows[1]
        if context0["_learned_top1_bin"] is None or context1["_learned_top1_bin"] is None:
            continue
        if require_correct:
            correct_pair_flip = (
                int(context0["_learned_top1_bin"]) == int(context0["_expected_bin"])
                and int(context1["_learned_top1_bin"]) == int(context1["_expected_bin"])
                and int(context0["_expected_bin"]) != int(context1["_expected_bin"])
            )
            pair_values.append(float(correct_pair_flip))
        else:
            pair_values.append(float(int(context0["_learned_top1_bin"]) != int(context1["_learned_top1_bin"])))
    return float(sum(pair_values) / len(pair_values)) if pair_values else None


def _paired_group_contrast(
    rows: list[dict[str, object]],
    *,
    value_key: str,
    group_key: str,
) -> float | None:
    grouped: dict[int, dict[bool, list[float]]] = {}
    for row in rows:
        value = row[value_key]
        if value is None:
            continue
        group_identifier = int(row[group_key])
        grouped.setdefault(group_identifier, {True: [], False: []})[bool(row["global_expected"])].append(float(value))

    contrasts: list[float] = []
    for group_rows in grouped.values():
        if not group_rows[True] or not group_rows[False]:
            continue
        expected_mean = sum(group_rows[True]) / len(group_rows[True])
        unexpected_mean = sum(group_rows[False]) / len(group_rows[False])
        contrasts.append(expected_mean - unexpected_mean)
    return _mean_or_none(contrasts)


def build_hidden_state_diagnostics_v2(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize Stage-2 hidden-state diagnostics from row-level probe diagnostics."""

    diagnostics = {
        "n_probe_rows": int(len(rows)),
        "learned_probe_expected_target_margin__v2": _mean_or_none(
            [float(row["learned_expected_target_margin__v2"]) for row in rows if row["learned_expected_target_margin__v2"] is not None]
        ),
        "oracle_probe_expected_target_margin__v2": _mean_or_none(
            [float(row["oracle_expected_target_margin__v2"]) for row in rows]
        ),
        "learned_probe_expected_target_confidence__v2": _mean_or_none(
            [
                float(row["learned_expected_target_confidence__v2"])
                for row in rows
                if row["learned_expected_target_confidence__v2"] is not None
            ]
        ),
        "oracle_probe_expected_target_confidence__v2": _mean_or_none(
            [float(row["oracle_expected_target_confidence__v2"]) for row in rows]
        ),
        "learned_probe_top1_confidence__v2": _mean_or_none(
            [float(row["learned_top1_confidence__v2"]) for row in rows if row["learned_top1_confidence__v2"] is not None]
        ),
        "oracle_probe_top1_confidence__v2": _mean_or_none(
            [float(row["oracle_top1_confidence__v2"]) for row in rows]
        ),
        "learned_probe_precision_gap__v2": _mean_or_none(
            [float(row["learned_precision_gap__v2"]) for row in rows if row["learned_precision_gap__v2"] is not None]
        ),
        "oracle_probe_precision_gap__v2": _mean_or_none(
            [float(row["oracle_precision_gap__v2"]) for row in rows]
        ),
        "probe_target_aligned_specificity_contrast__context0_v2": _paired_group_contrast(
            [row for row in rows if int(row["context_id"]) == 0],
            value_key="l23_target_specificity__v2",
            group_key="_symmetry_group_id",
        ),
        "probe_target_aligned_specificity_contrast__context1_v2": _paired_group_contrast(
            [row for row in rows if int(row["context_id"]) == 1],
            value_key="l23_target_specificity__v2",
            group_key="_symmetry_group_id",
        ),
        "probe_target_aligned_specificity_contrast__block_early_v2": _paired_group_contrast(
            [row for row in rows if row["block_position_bin__v2"] == "early"],
            value_key="l23_target_specificity__v2",
            group_key="_symmetry_group_id",
        ),
        "probe_target_aligned_specificity_contrast__block_late_v2": _paired_group_contrast(
            [row for row in rows if row["block_position_bin__v2"] == "late"],
            value_key="l23_target_specificity__v2",
            group_key="_symmetry_group_id",
        ),
        "probe_within_pair_mass__v2": _mean_or_none(
            [float(row["_learned_within_pair_mass"]) for row in rows if row["_learned_within_pair_mass"] is not None]
        ),
        "probe_correct_pair_flip_rate__v2": _pair_flip_rate(rows, require_correct=True),
        "probe_source_bin_top1__v2": _mean_or_none(
            [float(row["_learned_source_bin_top1"]) for row in rows if row["_learned_source_bin_top1"] is not None]
        ),
        "probe_source_bin_kl__v2": _mean_or_none(
            [float(row["_learned_source_bin_kl"]) for row in rows if row["_learned_source_bin_kl"] is not None]
        ),
        "probe_maxprob_correct_gap__v2": _correct_gap(rows, "learned_top1_confidence__v2"),
        "probe_entropy_correct_gap__v2": _correct_gap(rows, "_learned_entropy"),
        "probe_within_pair_mass_correct_gap__v2": _correct_gap(rows, "_learned_within_pair_mass"),
    }
    if diagnostics["probe_source_bin_top1__v2"] is None or diagnostics["probe_correct_pair_flip_rate__v2"] is None:
        diagnostics["probe_collapse_index__v2"] = None
    else:
        diagnostics["probe_collapse_index__v2"] = (
            float(diagnostics["probe_source_bin_top1__v2"])
            - float(diagnostics["probe_correct_pair_flip_rate__v2"])
        )
    return diagnostics


def metric_versions_v2_payload() -> dict[str, str]:
    return {metric_key: "v2" for metric_key in HIDDEN_STATE_DIAGNOSTIC_METRICS_V2}


def write_hidden_state_probe_table_v2(
    rows: list[dict[str, object]],
    output_path: str | Path,
) -> None:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HIDDEN_STATE_PROBE_TABLE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: row.get(field_name) for field_name in HIDDEN_STATE_PROBE_TABLE_FIELDS})


def write_hidden_state_diagnostics_v2_json(
    diagnostics_payload: dict[str, object],
    output_path: str | Path,
) -> None:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def summarize_metric(values: list[float | None]) -> dict[str, float | int | None]:
    present_values = [float(value) for value in values if value is not None]
    if not present_values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(present_values),
        "mean": float(sum(present_values) / len(present_values)),
        "median": float(statistics.median(present_values)),
        "min": float(min(present_values)),
        "max": float(max(present_values)),
    }
