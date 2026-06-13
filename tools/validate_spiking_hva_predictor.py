#!/usr/bin/env python3
"""Validate explicit spiking-HVA future L2/3 population prediction.

This validator is intentionally separate from the older host-side HVA
predictor checks. It requires artifacts whose names include
``spiking_hva`` and it fails closed when the no-feedback/no-leak rows are
missing. The primary unit of prediction is a held-out population vector
over retinotopic tiles, not exact cell identity.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PREDICTION_COLUMNS = {
    "prediction_index",
    "repeat_index",
    "frame_index",
    "target_frame_index",
    "tile_id",
    "split",
    "target_state_norm",
    "predicted_state_norm",
    "persistence_pred_state_norm",
    "train_mean_pred_state_norm",
    "no_learning_pred_state_norm",
    "temporal_block_shift_pred_state_norm",
    "spatial_tile_shuffle_pred_state_norm",
}


class ValidationError(ValueError):
    """Raised when the artifact set fails schema or biological gates."""


@dataclass(frozen=True)
class PredictionRow:
    prediction_index: int
    repeat_index: int
    frame_index: int
    target_frame_index: int
    tile_id: int
    split: str
    target: float
    model: float
    persistence: float
    train_mean: float
    no_learning: float
    temporal_shuffle: float
    spatial_shuffle: float


def parse_float(raw: str, path: Path, row_index: int, column: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError(f"{path} row {row_index} invalid float {column}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValidationError(f"{path} row {row_index} non-finite {column}={raw!r}")
    return value


def parse_int(raw: str, path: Path, row_index: int, column: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(f"{path} row {row_index} invalid int {column}={raw!r}") from exc


def load_predictions(path: Path) -> list[PredictionRow]:
    if not path.is_file():
        raise ValidationError(f"Missing spiking-HVA prediction file: {path}")
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"{path} has no header")
        missing = sorted(PREDICTION_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValidationError(f"{path} missing required columns: {missing}")
        for row_index, row in enumerate(reader, start=2):
            rows.append(
                PredictionRow(
                    prediction_index=parse_int(row["prediction_index"], path, row_index, "prediction_index"),
                    repeat_index=parse_int(row["repeat_index"], path, row_index, "repeat_index"),
                    frame_index=parse_int(row["frame_index"], path, row_index, "frame_index"),
                    target_frame_index=parse_int(row["target_frame_index"], path, row_index, "target_frame_index"),
                    tile_id=parse_int(row["tile_id"], path, row_index, "tile_id"),
                    split=(row["split"] or "").strip(),
                    target=parse_float(row["target_state_norm"], path, row_index, "target_state_norm"),
                    model=parse_float(row["predicted_state_norm"], path, row_index, "predicted_state_norm"),
                    persistence=parse_float(
                        row["persistence_pred_state_norm"],
                        path,
                        row_index,
                        "persistence_pred_state_norm",
                    ),
                    train_mean=parse_float(
                        row["train_mean_pred_state_norm"],
                        path,
                        row_index,
                        "train_mean_pred_state_norm",
                    ),
                    no_learning=parse_float(
                        row["no_learning_pred_state_norm"],
                        path,
                        row_index,
                        "no_learning_pred_state_norm",
                    ),
                    temporal_shuffle=parse_float(
                        row["temporal_block_shift_pred_state_norm"],
                        path,
                        row_index,
                        "temporal_block_shift_pred_state_norm",
                    ),
                    spatial_shuffle=parse_float(
                        row["spatial_tile_shuffle_pred_state_norm"],
                        path,
                        row_index,
                        "spatial_tile_shuffle_pred_state_norm",
                    ),
                )
            )
    if not rows:
        raise ValidationError(f"{path} contains no prediction rows")
    return rows


def parse_summary(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise ValidationError(f"Missing summary file: {path}")
    metrics: dict[str, float] = {}
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "metric" not in reader.fieldnames or "value" not in reader.fieldnames:
                raise ValidationError(f"{path} must contain metric,value columns")
            for row_index, row in enumerate(reader, start=2):
                metric = (row["metric"] or "").strip()
                if not metric:
                    continue
                metrics[metric] = parse_float(row["value"], path, row_index, "value")
    else:
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                if "=" not in line:
                    continue
                metric, value = line.rstrip("\n").split("=", 1)
                try:
                    metrics[metric] = float(value)
                except ValueError:
                    continue
    return metrics


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def vector_corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValidationError("correlation received empty or size-mismatched vectors")
    ma = mean(a)
    mb = mean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = math.sqrt(dot(da, da) * dot(db, db))
    return 0.0 if denom <= 0.0 else dot(da, db) / denom


def vector_cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(dot(a, a) * dot(b, b))
    return 0.0 if denom <= 0.0 else dot(a, b) / denom


def mse(a: list[float], b: list[float]) -> float:
    return mean((x - y) ** 2 for x, y in zip(a, b, strict=True))


def ndcg_at_k(target: list[float], score: list[float], k: int) -> float:
    if not target or k <= 0:
        return 0.0
    k = min(k, len(target))
    ranked = sorted(range(len(score)), key=lambda idx: score[idx], reverse=True)[:k]
    ideal = sorted(range(len(target)), key=lambda idx: target[idx], reverse=True)[:k]
    dcg = sum(target[idx] / math.log2(rank + 2.0) for rank, idx in enumerate(ranked))
    idcg = sum(target[idx] / math.log2(rank + 2.0) for rank, idx in enumerate(ideal))
    return 0.0 if idcg <= 0.0 else dcg / idcg


def grouped_vectors(rows: list[PredictionRow]) -> list[dict[str, list[float]]]:
    grouped: dict[tuple[int, int, int], dict[int, PredictionRow]] = defaultdict(dict)
    for row in rows:
        if row.split != "heldout":
            continue
        key = (row.repeat_index, row.frame_index, row.target_frame_index)
        grouped[key][row.tile_id] = row
    samples: list[dict[str, list[float]]] = []
    for key in sorted(grouped):
        by_tile = grouped[key]
        tile_ids = sorted(by_tile)
        if not tile_ids:
            continue
        expected = list(range(tile_ids[0], tile_ids[-1] + 1))
        if tile_ids != expected:
            raise ValidationError(f"Incomplete tile vector for sample {key}: first/last={tile_ids[0]}/{tile_ids[-1]}")
        ordered = [by_tile[tile_id] for tile_id in tile_ids]
        samples.append(
            {
                "target": [row.target for row in ordered],
                "model": [row.model for row in ordered],
                "persistence": [row.persistence for row in ordered],
                "train_mean": [row.train_mean for row in ordered],
                "no_learning": [row.no_learning for row in ordered],
                "temporal_shuffle": [row.temporal_shuffle for row in ordered],
                "spatial_shuffle": [row.spatial_shuffle for row in ordered],
            }
        )
    if not samples:
        raise ValidationError("No heldout prediction samples found")
    return samples


def compute_metrics(samples: list[dict[str, list[float]]], *, top_k: int) -> dict[str, float]:
    metrics: dict[str, float] = {"sample_count": float(len(samples))}
    for name in ["model", "persistence", "train_mean", "no_learning", "temporal_shuffle", "spatial_shuffle"]:
        metrics[f"{name}_vector_corr_mean"] = mean(vector_corr(sample["target"], sample[name]) for sample in samples)
        metrics[f"{name}_vector_cosine_mean"] = mean(vector_cosine(sample["target"], sample[name]) for sample in samples)
        metrics[f"{name}_mse_mean"] = mean(mse(sample["target"], sample[name]) for sample in samples)
        metrics[f"{name}_ndcg_at_{top_k}_mean"] = mean(ndcg_at_k(sample["target"], sample[name], top_k) for sample in samples)
    metrics["model_vs_persistence_vector_corr_delta"] = (
        metrics["model_vector_corr_mean"] - metrics["persistence_vector_corr_mean"]
    )
    metrics["model_vs_no_learning_vector_corr_delta"] = (
        metrics["model_vector_corr_mean"] - metrics["no_learning_vector_corr_mean"]
    )
    metrics["model_vs_train_mean_vector_corr_delta"] = (
        metrics["model_vector_corr_mean"] - metrics["train_mean_vector_corr_mean"]
    )
    metrics["model_vs_persistence_mse_reduction"] = (
        (metrics["persistence_mse_mean"] - metrics["model_mse_mean"]) / max(metrics["persistence_mse_mean"], 1.0e-12)
    )
    metrics["model_vs_no_learning_mse_reduction"] = (
        (metrics["no_learning_mse_mean"] - metrics["model_mse_mean"]) / max(metrics["no_learning_mse_mean"], 1.0e-12)
    )
    metrics["model_vs_train_mean_mse_reduction"] = (
        (metrics["train_mean_mse_mean"] - metrics["model_mse_mean"]) / max(metrics["train_mean_mse_mean"], 1.0e-12)
    )
    return metrics


def bootstrap_delta_ci(
    samples: list[dict[str, list[float]]],
    *,
    metric: str,
    baseline: str,
    seed: int,
    bootstrap_samples: int,
) -> tuple[float, float]:
    if bootstrap_samples <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(bootstrap_samples):
        draw = [samples[rng.randrange(len(samples))] for _ in samples]
        if metric == "corr":
            model = mean(vector_corr(sample["target"], sample["model"]) for sample in draw)
            control = mean(vector_corr(sample["target"], sample[baseline]) for sample in draw)
            deltas.append(model - control)
        elif metric == "mse_reduction":
            model_mse = mean(mse(sample["target"], sample["model"]) for sample in draw)
            control_mse = mean(mse(sample["target"], sample[baseline]) for sample in draw)
            deltas.append((control_mse - model_mse) / max(control_mse, 1.0e-12))
        else:
            raise AssertionError(metric)
    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return lo, hi


def require_summary_gate(metrics: dict[str, float], key: str, expected: float) -> None:
    if key not in metrics:
        raise ValidationError(f"Missing no-cheat summary row: {key}")
    if metrics[key] != expected:
        raise ValidationError(f"No-cheat gate failed: {key}={metrics[key]} expected {expected}")


def validate_no_cheat(summary: dict[str, float]) -> None:
    required_equal = {
        "spiking_hva_enabled": 1.0,
        "spiking_hva_scaffold_only": 0.0,
        "spiking_hva_prediction_learning_enabled": 1.0,
        "spiking_hva_feedback_to_v1_enabled": 0.0,
        "spiking_hva_hva_to_v1_connection_count": 0.0,
        "spiking_hva_hva_to_v1_current_enabled": 0.0,
        "spiking_hva_external_v1_input_l23e_only": 1.0,
        "spiking_hva_uses_l4_input": 0.0,
        "spiking_hva_uses_future_features": 0.0,
        "spiking_hva_heldout_updates_applied": 0.0,
    }
    for key, expected in required_equal.items():
        require_summary_gate(summary, key, expected)


def _summary_value(summary: dict[str, float], key: str) -> float:
    if key not in summary:
        raise ValidationError(f"Missing architecture-contract summary row: {key}")
    return summary[key]


def _any_enabled(summary: dict[str, float], keys: list[str]) -> bool:
    return any(summary.get(key, 0.0) == 1.0 for key in keys)


def validate_architecture_contract(summary: dict[str, float]) -> None:
    """Fail closed unless prediction comes from explicit HVA spikes/state.

    This is intentionally separate from metric gates. A direct host readout
    from L2/3 tile rates can score well numerically but is not a higher-area
    predictor. The contract therefore requires an explicit source flag and
    rejects known direct-readout source declarations.
    """

    explicit_hva_source_flags = [
        "spiking_hva_predictor_prediction_source_explicit_hva_spikes",
        "spiking_hva_predictor_prediction_source_explicit_hva_state",
    ]
    if not _any_enabled(summary, explicit_hva_source_flags):
        raise ValidationError(
            "Architecture contract failed: prediction source must be explicit HVA spikes/state; "
            f"missing one of {explicit_hva_source_flags}"
        )

    disallowed_source_flags = [
        "spiking_hva_predictor_prediction_source_direct_l23_readout",
        "spiking_hva_predictor_prediction_source_direct_l23e_tile_rates",
        "spiking_hva_predictor_prediction_source_host_readout",
        "spiking_hva_predictor_prediction_source_algorithmic_ei_readout",
        "spiking_hva_predictor_prediction_source_hva_e_host_readout",
        "spiking_hva_predictor_direct_l23e_tile_rate_readout",
        "spiking_hva_predictor_algorithmic_ei_readout_head",
        "spiking_hva_predictor_hva_e_to_prediction_local_synapse_standin",
        "spiking_hva_predictor_host_softmax_listwise_readout_enabled",
    ]
    enabled_disallowed = [key for key in disallowed_source_flags if summary.get(key, 0.0) != 0.0]
    if enabled_disallowed:
        raise ValidationError(
            "Architecture contract failed: disallowed direct/readout prediction source enabled: "
            + ", ".join(enabled_disallowed)
        )

    multi_primary = summary.get(
        "spiking_hva_predictor_multi_timescale_state_primary_prediction",
        0.0,
    )
    multi_host = summary.get(
        "spiking_hva_predictor_multi_timescale_state_host_side_reconstruction",
        0.0,
    )
    multi_actual_genn = summary.get(
        "spiking_hva_predictor_multi_timescale_state_actual_genn_state",
        0.0,
    )
    if multi_primary == 1.0 and multi_host == 1.0:
        raise ValidationError(
            "Architecture contract failed: multi-timescale primary prediction is host-side reconstruction"
        )
    if multi_primary == 1.0 and multi_actual_genn != 1.0:
        raise ValidationError(
            "Architecture contract failed: multi-timescale primary prediction must be actual GeNN state"
        )

    required_equal = {
        "spiking_hva_predictor_no_future_features_at_prediction": 1.0,
        "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
        "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
        "spiking_hva_predictor_heldout_update_count": 0.0,
        "spiking_hva_predictor_feedback_to_v1_enabled": 0.0,
        "spiking_hva_predictor_hva_to_v1_connection_count": 0.0,
        "spiking_hva_predictor_lower_v1_mutation_enabled": 0.0,
    }
    for key, expected in required_equal.items():
        require_summary_gate(summary, key, expected)

    predictor_l4_input = summary.get("spiking_hva_predictor_uses_l4_input")
    if predictor_l4_input is None:
        predictor_l4_input = _summary_value(summary, "spiking_hva_uses_l4_input")
    if predictor_l4_input != 0.0:
        raise ValidationError(
            f"Architecture contract failed: predictor uses L4 input ({predictor_l4_input})"
        )


def validate_physiology_safety(summary: dict[str, float]) -> None:
    """Reject hidden HVA context runaway when transition source export is enabled."""

    transition_enabled = summary.get("spiking_hva_ctx_transition_enabled", 0.0) == 1.0
    transition_source_enabled = (
        summary.get("spiking_hva_predictor_hva_ctx_transition_state_source_export_enabled", 0.0) == 1.0
        or summary.get("spiking_hva_ctx_transition_state_source_export_enabled", 0.0) == 1.0
    )
    if not (transition_enabled and transition_source_enabled):
        return

    ctx_mean_key = "spiking_hva_ctx_transition_prediction_hva_ctx_mean_rate_hz"
    state_p99_key = "spiking_hva_ctx_transition_transition_state_p99_norm"
    ctx_mean_hz = _summary_value(summary, ctx_mean_key)
    transition_state_p99 = _summary_value(summary, state_p99_key)
    if ctx_mean_hz > 20.0:
        raise ValidationError(
            f"Physiology safety failed: {ctx_mean_key}={ctx_mean_hz:.6f} > 20 Hz"
        )
    if transition_state_p99 >= 1.95:
        raise ValidationError(
            f"Physiology safety failed: {state_p99_key}={transition_state_p99:.6f} >= 1.95"
        )


def validate_prediction_gates(
    metrics: dict[str, float],
    *,
    min_corr: float,
    min_delta: float,
    min_train_mean_delta: float,
) -> None:
    gates = {
        "model_vector_corr_mean": min_corr,
        "model_vector_cosine_mean": 0.75,
        "model_vs_persistence_vector_corr_delta": min_delta,
        "model_vs_no_learning_vector_corr_delta": min_delta,
        "model_vs_train_mean_vector_corr_delta": min_train_mean_delta,
        "model_vs_persistence_mse_reduction": 0.15,
        "model_vs_no_learning_mse_reduction": 0.15,
        "model_vs_train_mean_mse_reduction": 0.05,
    }
    failures = [
        f"{key}={metrics.get(key, float('nan')):.6f} < {threshold:.6f}"
        for key, threshold in gates.items()
        if metrics.get(key, -math.inf) < threshold
    ]
    if failures:
        raise ValidationError("Prediction gates failed: " + "; ".join(failures))


def find_summary_path(genn_dir: Path, prefix: str) -> Path:
    csv_path = genn_dir / f"{prefix}_summary.csv"
    txt_path = genn_dir / f"{prefix}_summary.txt"
    if csv_path.is_file():
        return csv_path
    return txt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genn-dir", type=Path, required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--primary-delay-frames", type=int, default=5)
    parser.add_argument("--secondary-delay-frames", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-corr", type=float, default=0.70)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--min-train-mean-delta", type=float, default=0.03)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    summary = parse_summary(find_summary_path(args.genn_dir, args.full))
    validate_no_cheat(summary)
    validate_architecture_contract(summary)
    validate_physiology_safety(summary)

    delay_list = [args.primary_delay_frames]
    if args.secondary_delay_frames:
        delay_list.extend(int(part) for part in args.secondary_delay_frames.split(",") if part.strip())

    all_metrics: dict[str, dict[str, float]] = {}
    for delay in delay_list:
        predictions = load_predictions(args.genn_dir / f"{args.full}_spiking_hva_predictions_delay{delay}.csv")
        samples = grouped_vectors(predictions)
        metrics = compute_metrics(samples, top_k=args.top_k)
        ci_corr = bootstrap_delta_ci(
            samples,
            metric="corr",
            baseline="persistence",
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        ci_mse = bootstrap_delta_ci(
            samples,
            metric="mse_reduction",
            baseline="persistence",
            seed=args.seed + 1,
            bootstrap_samples=args.bootstrap_samples,
        )
        metrics["bootstrap_corr_delta_vs_persistence_ci95_low"] = ci_corr[0]
        metrics["bootstrap_corr_delta_vs_persistence_ci95_high"] = ci_corr[1]
        metrics["bootstrap_mse_reduction_vs_persistence_ci95_low"] = ci_mse[0]
        metrics["bootstrap_mse_reduction_vs_persistence_ci95_high"] = ci_mse[1]
        all_metrics[f"delay{delay}"] = metrics
        if delay == args.primary_delay_frames:
            validate_prediction_gates(
                metrics,
                min_corr=args.min_corr,
                min_delta=args.min_delta,
                min_train_mean_delta=args.min_train_mean_delta,
            )
            if ci_corr[0] <= 0.0:
                raise ValidationError(
                    "Prediction gate failed: bootstrap_corr_delta_vs_persistence_ci95_low "
                    f"{ci_corr[0]:.6f} <= 0"
                )
            if ci_mse[0] <= 0.0:
                raise ValidationError(
                    "Prediction gate failed: bootstrap_mse_reduction_vs_persistence_ci95_low "
                    f"{ci_mse[0]:.6f} <= 0"
                )

    for delay_name, metrics in all_metrics.items():
        for key in sorted(metrics):
            print(f"{delay_name}.{key}={metrics[key]:.6f}")
    print("PASS spiking_hva_predictor")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"FAIL spiking_hva_predictor: {exc}", file=sys.stderr)
        raise SystemExit(1)
