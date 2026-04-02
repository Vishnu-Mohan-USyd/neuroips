"""Deterministic seed-aggregated reporting for archived run bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import statistics

import torch

from .diagnostics import (
    DIAGNOSTIC_CLASSIFICATION_RULE_VERSION as STAGE2_CLASSIFICATION_RULE_VERSION,
    DIAGNOSTIC_METRIC_SCHEMA_VERSION as STAGE2_DIAGNOSTIC_METRIC_SCHEMA_VERSION,
    DIAGNOSTIC_SCHEMA_VERSION as STAGE2_DIAGNOSTIC_SCHEMA_VERSION,
    HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
    HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT,
    HIDDEN_STATE_DIAGNOSTIC_METRICS_V2,
    build_hidden_state_diagnostics_v2,
    build_hidden_state_probe_table_rows,
    metric_versions_v2_payload,
    summarize_metric,
    write_hidden_state_diagnostics_v2_json,
    write_hidden_state_probe_table_v2,
)

REPORT_ARTIFACT_VERSION = "stage1_reporting_v1"
METRIC_SCHEMA_VERSION = "stage1_metric_schema_v1"
CLASSIFICATION_RULE_VERSION = "stage1_classification_rules_v1"
FROZEN_BENCHMARK_METRIC_VERSION = "benchmark_v1_frozen"
LEGACY_BENCHMARK_METRIC_VERSION = "benchmark_v0_legacy"
DEFAULT_BOOTSTRAP_SEED = 20260401
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

CLASS_PHASE1_POSTSTIM_POSITIVE = "phase-1 post-stim positive"
CLASS_PHASE1_PRESTIM_NEGATIVE = "phase-1 prestim negative"
CLASS_PHASE15_POSITIVE = "phase-1.5 decisive positive"
CLASS_PHASE2_CLEAN_NEGATIVE = "phase-2 source-heldout primary negative"
CLASS_CHALLENGER_NO_ADMISSION = "phase-2 challenger-selection negative"
CLASS_UNCLASSIFIED = "unclassified"
SEEDWISE_CSV_ARTIFACT = "seedwise_metrics.v1.csv"
STATS_JSON_ARTIFACT = "stats_report.v1.json"
VERDICT_JSON_ARTIFACT = "verdict_report.v1.json"
DIAGNOSTIC_CALIBRATION_REPORT_V2_ARTIFACT = "diagnostic_calibration_report.v2.json"


@dataclass(frozen=True)
class ArchivedRunBundle:
    run_dir: Path
    run_id: str
    manifest: dict[str, Any]
    notes: dict[str, Any]
    available_files: tuple[str, ...]
    classification: str
    source_metric_version: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def classify_archived_bundle(
    *,
    notes: dict[str, Any],
    available_files: tuple[str, ...],
) -> str:
    file_set = set(available_files)
    if "challenger_candidate" in notes:
        return CLASS_CHALLENGER_NO_ADMISSION
    if "prestim_gate.json" in file_set or notes.get("gate") == "context_only_prestim_template":
        return CLASS_PHASE1_PRESTIM_NEGATIVE
    if "probe_metrics.json" in file_set:
        if notes.get("phase2_regime") == "p2a_source_heldout_probe_generalization":
            return CLASS_PHASE2_CLEAN_NEGATIVE
        return CLASS_PHASE15_POSITIVE
    if "primary_metrics.json" in file_set:
        return CLASS_PHASE1_POSTSTIM_POSITIVE
    return CLASS_UNCLASSIFIED


def _detect_primary_metric_version(primary_metrics: dict[str, Any]) -> str:
    dampening = primary_metrics.get("dampening", {})
    if "poststimulus_l23_template_specificity" in dampening:
        return FROZEN_BENCHMARK_METRIC_VERSION
    return LEGACY_BENCHMARK_METRIC_VERSION


def detect_source_metric_version(run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    primary_metrics = _maybe_read_json(run_path / "primary_metrics.json")
    if primary_metrics is not None:
        return _detect_primary_metric_version(primary_metrics)
    if (run_path / "prestim_gate.json").exists():
        return FROZEN_BENCHMARK_METRIC_VERSION
    if (run_path / "probe_metrics.json").exists():
        return FROZEN_BENCHMARK_METRIC_VERSION
    if (run_path / "probe_context_alignment_report.json").exists():
        return FROZEN_BENCHMARK_METRIC_VERSION
    return LEGACY_BENCHMARK_METRIC_VERSION


def load_archived_run_bundle(run_dir: str | Path) -> ArchivedRunBundle:
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "manifest.json")
    notes = dict(manifest.get("notes", {}))
    available_files = tuple(sorted(path.name for path in run_path.iterdir() if path.is_file()))
    classification = classify_archived_bundle(notes=notes, available_files=available_files)
    source_metric_version = detect_source_metric_version(run_path)
    return ArchivedRunBundle(
        run_dir=run_path,
        run_id=run_path.name,
        manifest=manifest,
        notes=notes,
        available_files=available_files,
        classification=classification,
        source_metric_version=source_metric_version,
    )


def _family_descriptor(bundle: ArchivedRunBundle) -> dict[str, Any]:
    notes = bundle.notes
    return {
        "classification": bundle.classification,
        "gate": notes.get("gate", "tranche1" if "primary_metrics.json" in bundle.available_files else None),
        "controlled_sources": notes.get("controlled_sources"),
        "phase2_regime": notes.get("phase2_regime"),
        "challenger_candidate": notes.get("challenger_candidate"),
        "source_metric_version": bundle.source_metric_version,
    }


def discover_archived_run_families(root_dir: str | Path) -> dict[str, list[ArchivedRunBundle]]:
    root = Path(root_dir)
    families: dict[str, list[ArchivedRunBundle]] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        bundle = load_archived_run_bundle(run_dir)
        key = json.dumps(_family_descriptor(bundle), sort_keys=True)
        families.setdefault(key, []).append(bundle)
    return families


def _extract_bundle_metric(
    bundle: ArchivedRunBundle,
    metric_name: str,
    *,
    subcase: str,
) -> tuple[float, float | None]:
    run_dir = bundle.run_dir
    primary_metrics = _maybe_read_json(run_dir / "primary_metrics.json")
    if primary_metrics is not None and subcase in primary_metrics and metric_name in primary_metrics[subcase]:
        oracle_primary = _maybe_read_json(run_dir / "oracle_primary_metrics.json")
        oracle_value = None
        if oracle_primary is not None and subcase in oracle_primary and metric_name in oracle_primary[subcase]:
            oracle_value = float(oracle_primary[subcase][metric_name])
        return float(primary_metrics[subcase][metric_name]), oracle_value

    probe_metrics = _maybe_read_json(run_dir / "probe_metrics.json")
    if probe_metrics is not None and metric_name in probe_metrics:
        oracle_probe = _maybe_read_json(run_dir / "oracle_probe_metrics.json")
        oracle_value = None
        if oracle_probe is not None and metric_name in oracle_probe:
            oracle_value = float(oracle_probe[metric_name])
        return float(probe_metrics[metric_name]), oracle_value

    alignment_report = _maybe_read_json(run_dir / "probe_context_alignment_report.json")
    if alignment_report is not None:
        learned_key = metric_name if metric_name.startswith("learned_") else f"learned_{metric_name}"
        oracle_key = metric_name if metric_name.startswith("oracle_") else f"oracle_{metric_name}"
        if learned_key in alignment_report:
            oracle_value = float(alignment_report[oracle_key]) if oracle_key in alignment_report else None
            return float(alignment_report[learned_key]), oracle_value

    prestim_gate = _maybe_read_json(run_dir / "prestim_gate.json")
    if prestim_gate is not None:
        gate_name = bundle.notes.get("gate", "")
        gate_condition = "context_only"
        if isinstance(gate_name, str) and gate_name.endswith("_prestim_template"):
            gate_condition = gate_name.removesuffix("_prestim_template")
        intact_section = prestim_gate.get("intact", {}).get(gate_condition, {})
        if metric_name in intact_section:
            return float(intact_section[metric_name]), None

    raise KeyError(f"metric {metric_name!r} not found for archived bundle {bundle.run_id}")


def _sign_counts(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(1 for value in values if value > 0.0),
        "negative": sum(1 for value in values if value < 0.0),
        "zero": sum(1 for value in values if value == 0.0),
    }


def _bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 1:
        point = float(tensor.item())
        return {
            "seed": seed,
            "resamples": resamples,
            "low": point,
            "high": point,
        }
    generator = torch.Generator().manual_seed(seed)
    sample_indices = torch.randint(
        low=0,
        high=tensor.numel(),
        size=(resamples, tensor.numel()),
        generator=generator,
    )
    sample_means = tensor[sample_indices].mean(dim=1)
    quantiles = torch.quantile(sample_means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "seed": seed,
        "resamples": resamples,
        "low": float(quantiles[0].item()),
        "high": float(quantiles[1].item()),
    }


def _summarize_values(
    values: list[float],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if not values:
        raise ValueError("seed aggregation requires at least one value")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "n": len(values),
        "mean": float(tensor.mean().item()),
        "median": float(statistics.median(values)),
        "std": float(tensor.std(unbiased=False).item()),
        "sign_counts": _sign_counts(values),
        "bootstrap_mean_ci_95": _bootstrap_mean_ci(
            values,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
    }


def _metric_versions(entries: list[dict[str, Any]], metric_name: str) -> dict[str, str]:
    versions = {metric_name: "v1"}
    if any(entry["oracle_value"] is not None for entry in entries):
        versions[f"oracle::{metric_name}"] = "v1"
    if any(entry["oracle_normalized_ratio"] is not None for entry in entries):
        versions[f"{metric_name}__oracle_normalized_ratio"] = "reporting_v1"
    return versions


def build_seed_aggregate_report(
    bundles: list[ArchivedRunBundle] | list[str] | list[Path],
    *,
    metric_name: str,
    subcase: str = "dampening",
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    resolved_bundles = [
        bundle if isinstance(bundle, ArchivedRunBundle) else load_archived_run_bundle(bundle)
        for bundle in bundles
    ]
    if not resolved_bundles:
        raise ValueError("build_seed_aggregate_report requires at least one archived bundle")

    entries: list[dict[str, Any]] = []
    for bundle in resolved_bundles:
        value, oracle_value = _extract_bundle_metric(bundle, metric_name, subcase=subcase)
        heldout_seed = bundle.notes.get("heldout_seed")
        ratio = None
        if oracle_value is not None and oracle_value != 0.0:
            ratio = value / oracle_value
        entries.append(
            {
                "run_id": bundle.run_id,
                "heldout_seed": int(heldout_seed) if heldout_seed is not None else None,
                "classification": bundle.classification,
                "source_metric_version": bundle.source_metric_version,
                "value": value,
                "oracle_value": oracle_value,
                "oracle_normalized_ratio": ratio,
            }
        )

    entries.sort(key=lambda item: (item["heldout_seed"] is None, item["heldout_seed"], item["run_id"]))
    values = [float(entry["value"]) for entry in entries]
    ratios = [float(entry["oracle_normalized_ratio"]) for entry in entries if entry["oracle_normalized_ratio"] is not None]
    classifications = sorted({bundle.classification for bundle in resolved_bundles})
    source_metric_versions = sorted({bundle.source_metric_version for bundle in resolved_bundles})
    family_descriptors = sorted(
        json.dumps(_family_descriptor(bundle), sort_keys=True)
        for bundle in resolved_bundles
    )
    metric_versions = _metric_versions(entries, metric_name)

    return {
        "report_version": REPORT_ARTIFACT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "classification_rule_version": CLASSIFICATION_RULE_VERSION,
        "benchmark_metric_version": FROZEN_BENCHMARK_METRIC_VERSION,
        "metric_versions": metric_versions,
        "metric_name": metric_name,
        "subcase": subcase,
        "classification": classifications[0] if len(classifications) == 1 else "mixed",
        "source_metric_versions": source_metric_versions,
        "family_descriptors": family_descriptors,
        "seedwise": entries,
        "summary": _summarize_values(
            values,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "oracle_normalized_ratio_summary": (
            _summarize_values(
                ratios,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            if ratios
            else None
        ),
    }


def write_seed_aggregate_report(report: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _stats_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_version": report["report_version"],
        "metric_schema_version": report["metric_schema_version"],
        "classification_rule_version": report["classification_rule_version"],
        "benchmark_metric_version": report["benchmark_metric_version"],
        "metric_versions": report["metric_versions"],
        "metric_name": report["metric_name"],
        "subcase": report["subcase"],
        "classification": report["classification"],
        "source_metric_versions": report["source_metric_versions"],
        "family_descriptors": report["family_descriptors"],
        "summary": report["summary"],
        "oracle_normalized_ratio_summary": report["oracle_normalized_ratio_summary"],
    }


def _verdict_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_version": report["report_version"],
        "metric_schema_version": report["metric_schema_version"],
        "classification_rule_version": report["classification_rule_version"],
        "benchmark_metric_version": report["benchmark_metric_version"],
        "metric_versions": report["metric_versions"],
        "metric_name": report["metric_name"],
        "subcase": report["subcase"],
        "classification": report["classification"],
        "n_seedwise": len(report["seedwise"]),
        "sign_counts": report["summary"]["sign_counts"],
        "mean": report["summary"]["mean"],
        "median": report["summary"]["median"],
        "std": report["summary"]["std"],
        "bootstrap_mean_ci_95": report["summary"]["bootstrap_mean_ci_95"],
    }


def _write_seedwise_csv(report: dict[str, Any], output_path: Path) -> None:
    fieldnames = [
        "run_id",
        "heldout_seed",
        "classification",
        "source_metric_version",
        "value",
        "oracle_value",
        "oracle_normalized_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report["seedwise"]:
            writer.writerow({field: entry.get(field) for field in fieldnames})


def write_seed_aggregate_artifact_trio(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seedwise_path = output_root / SEEDWISE_CSV_ARTIFACT
    stats_path = output_root / STATS_JSON_ARTIFACT
    verdict_path = output_root / VERDICT_JSON_ARTIFACT
    _write_seedwise_csv(report, seedwise_path)
    stats_path.write_text(json.dumps(_stats_payload(report), indent=2, sort_keys=True), encoding="utf-8")
    verdict_path.write_text(json.dumps(_verdict_payload(report), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "seedwise_csv": seedwise_path,
        "stats_report": stats_path,
        "verdict_report": verdict_path,
    }


def _load_frozen_benchmark_registry(artifacts_root: str | Path) -> dict[str, Any]:
    return _read_json(Path(artifacts_root) / "benchmarks" / "benchmark_registry.v1.json")


def write_hidden_state_diagnostic_artifacts_v2(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    benchmark_anchor_id: str | None = None,
    benchmark_registry_version: str | None = None,
    artifacts_root: str | Path = "artifacts",
    subcase: str = "dampening",
) -> dict[str, Any]:
    """Write Stage-2 row-level and run-level hidden-state diagnostics for one archived probe run."""

    bundle = load_archived_run_bundle(run_dir)
    resolved_benchmark_registry_version = (
        benchmark_registry_version
        if benchmark_registry_version is not None
        else _load_frozen_benchmark_registry(artifacts_root)["benchmark_registry_version"]
    )
    rows = build_hidden_state_probe_table_rows(run_dir, subcase=subcase)
    diagnostics = build_hidden_state_diagnostics_v2(rows)
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(run_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    table_path = resolved_output_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT
    diagnostics_path = resolved_output_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT
    write_hidden_state_probe_table_v2(rows, table_path)
    diagnostics_payload = {
        "metric_schema_version": STAGE2_DIAGNOSTIC_METRIC_SCHEMA_VERSION,
        "metric_versions": metric_versions_v2_payload(),
        "classification_rule_version": STAGE2_CLASSIFICATION_RULE_VERSION,
        "source_metric_versions": [bundle.source_metric_version],
        "diagnostic_schema_version": STAGE2_DIAGNOSTIC_SCHEMA_VERSION,
        "benchmark_registry_version": resolved_benchmark_registry_version,
        "run_id": bundle.run_id,
        "classification": bundle.classification,
        "benchmark_anchor_id": benchmark_anchor_id,
        **diagnostics,
    }
    write_hidden_state_diagnostics_v2_json(diagnostics_payload, diagnostics_path)
    return {
        "rows": rows,
        "diagnostics": diagnostics_payload,
        "table_path": table_path,
        "diagnostics_path": diagnostics_path,
    }


def write_diagnostic_calibration_report_v2(
    benchmark_anchor_id: str,
    *,
    artifacts_root: str | Path = "artifacts",
    output_dir: str | Path | None = None,
    subcase: str = "dampening",
) -> dict[str, Any]:
    """Rescore one frozen benchmark family with Stage-2 hidden-state diagnostics."""

    registry_payload = _load_frozen_benchmark_registry(artifacts_root)
    benchmark_registry_version = registry_payload["benchmark_registry_version"]
    anchor_payload = registry_payload["anchors"][benchmark_anchor_id]
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(artifacts_root) / "benchmarks" / benchmark_anchor_id
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    per_run_outputs: list[dict[str, Any]] = []
    for run_id in anchor_payload["run_ids"]:
        run_dir = Path(artifacts_root) / "runs" / run_id
        run_output_dir = resolved_output_dir / "runs" / run_id
        per_run_outputs.append(
            write_hidden_state_diagnostic_artifacts_v2(
                run_dir,
                output_dir=run_output_dir,
                benchmark_anchor_id=benchmark_anchor_id,
                benchmark_registry_version=benchmark_registry_version,
                subcase=subcase,
            )
        )

    source_metric_versions = sorted(
        {
            version
            for item in per_run_outputs
            for version in item["diagnostics"]["source_metric_versions"]
        }
    )
    seedwise = []
    for item in per_run_outputs:
        diagnostics_payload = item["diagnostics"]
        seedwise.append(
            {
                "run_id": diagnostics_payload["run_id"],
                **{
                    metric_key: diagnostics_payload.get(metric_key)
                    for metric_key in HIDDEN_STATE_DIAGNOSTIC_METRICS_V2
                },
            }
        )
    metric_summaries = {
        metric_key: summarize_metric([entry.get(metric_key) for entry in seedwise])
        for metric_key in HIDDEN_STATE_DIAGNOSTIC_METRICS_V2
    }
    report_payload = {
        "metric_schema_version": STAGE2_DIAGNOSTIC_METRIC_SCHEMA_VERSION,
        "metric_versions": metric_versions_v2_payload(),
        "classification_rule_version": STAGE2_CLASSIFICATION_RULE_VERSION,
        "source_metric_versions": source_metric_versions,
        "diagnostic_schema_version": STAGE2_DIAGNOSTIC_SCHEMA_VERSION,
        "benchmark_registry_version": benchmark_registry_version,
        "benchmark_anchor_id": benchmark_anchor_id,
        "benchmark_classification": anchor_payload["classification"],
        "run_ids": list(anchor_payload["run_ids"]),
        "seedwise": seedwise,
        "summary": metric_summaries,
    }
    report_path = resolved_output_dir / DIAGNOSTIC_CALIBRATION_REPORT_V2_ARTIFACT
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "report": report_payload,
        "report_path": report_path,
        "per_run_outputs": per_run_outputs,
    }
