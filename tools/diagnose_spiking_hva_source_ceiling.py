#!/usr/bin/env python3
"""Offline source-ceiling diagnostics for explicit spiking-HVA artifacts.

This tool estimates how much future L2/3 population-state information is
present in exported source states such as current L2/3, HVA_E, HVA_CTX, and
HVA_PRED. It is intentionally diagnostic only: train-only ridge/linear fits
use held-out targets for evaluation, but they are not model predictions and
must not be reported as HVA model success.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


RATE_KEY_COLUMNS = {"repeat_index", "frame_index", "tile_id"}
PREDICTION_COLUMNS = {
    "repeat_index",
    "frame_index",
    "target_frame_index",
    "tile_id",
    "split",
    "target_state_norm",
}


class DiagnosticError(ValueError):
    """Raised when source-ceiling inputs are missing or malformed."""


@dataclass(frozen=True)
class PredictionSample:
    repeat_index: int
    frame_index: int
    target_frame_index: int
    split: str
    tile_ids: tuple[int, ...]
    target: np.ndarray


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    state_columns: tuple[str, ...]
    rate_column: str | None
    history_source_column: str | None = None
    history_rate_column: str | None = None
    source_guidance_allowed: bool = True
    predictor_output_state_source: bool = False


def parse_float(raw: str, path: Path, row_number: int, column: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise DiagnosticError(f"{path} row {row_number} invalid float {column}={raw!r}") from exc
    if not math.isfinite(value):
        raise DiagnosticError(f"{path} row {row_number} non-finite {column}={raw!r}")
    return value


def parse_int(raw: str, path: Path, row_number: int, column: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise DiagnosticError(f"{path} row {row_number} invalid int {column}={raw!r}") from exc


def parse_summary(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    metrics: dict[str, float] = {}
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "metric" not in reader.fieldnames or "value" not in reader.fieldnames:
                return metrics
            for row in reader:
                metric = (row.get("metric") or "").strip()
                if not metric:
                    continue
                try:
                    metrics[metric] = float(row.get("value", "nan"))
                except ValueError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                metric, raw_value = line.rstrip("\n").split("=", 1)
                try:
                    metrics[metric] = float(raw_value)
                except ValueError:
                    continue
    return metrics


def resolve_prefix(genn_dir: Path, prefix: str | None) -> str:
    if prefix is not None:
        return prefix
    matches = sorted(genn_dir.glob("*_spiking_hva_rates.csv"))
    if len(matches) != 1:
        raise DiagnosticError(
            f"{genn_dir} contains {len(matches)} *_spiking_hva_rates.csv files; pass --prefix."
        )
    suffix = "_spiking_hva_rates.csv"
    return matches[0].name[: -len(suffix)]


def load_rates(path: Path) -> tuple[dict[tuple[int, int, int], dict[str, float]], list[str]]:
    if not path.is_file():
        raise DiagnosticError(f"Missing spiking-HVA rates file: {path}")
    rows: dict[tuple[int, int, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"{path} has no header")
        missing = sorted(RATE_KEY_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise DiagnosticError(f"{path} missing required columns: {missing}")
        fieldnames = list(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            repeat_index = parse_int(row["repeat_index"], path, row_number, "repeat_index")
            frame_index = parse_int(row["frame_index"], path, row_number, "frame_index")
            tile_id = parse_int(row["tile_id"], path, row_number, "tile_id")
            key = (repeat_index, frame_index, tile_id)
            if key in rows:
                raise DiagnosticError(f"{path} duplicate rate row for key={key}")
            parsed: dict[str, float] = {}
            for column in fieldnames:
                if column in {"sample_index", "repeat_index", "frame_index", "tile_id"}:
                    continue
                raw = row.get(column, "")
                if raw == "":
                    continue
                parsed[column] = parse_float(raw, path, row_number, column)
            rows[key] = parsed
    if not rows:
        raise DiagnosticError(f"{path} contains no rate rows")
    return rows, fieldnames


def load_prediction_samples(path: Path) -> list[PredictionSample]:
    if not path.is_file():
        raise DiagnosticError(f"Missing spiking-HVA prediction file: {path}")
    grouped: dict[tuple[int, int, int, str], dict[int, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"{path} has no header")
        missing = sorted(PREDICTION_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise DiagnosticError(f"{path} missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            repeat_index = parse_int(row["repeat_index"], path, row_number, "repeat_index")
            frame_index = parse_int(row["frame_index"], path, row_number, "frame_index")
            target_frame_index = parse_int(row["target_frame_index"], path, row_number, "target_frame_index")
            tile_id = parse_int(row["tile_id"], path, row_number, "tile_id")
            split = (row["split"] or "").strip()
            target = parse_float(row["target_state_norm"], path, row_number, "target_state_norm")
            grouped.setdefault((repeat_index, frame_index, target_frame_index, split), {})[tile_id] = target
    samples: list[PredictionSample] = []
    for key in sorted(grouped):
        by_tile = grouped[key]
        tile_ids = tuple(sorted(by_tile))
        if tile_ids != tuple(range(tile_ids[0], tile_ids[-1] + 1)):
            raise DiagnosticError(f"{path} incomplete tile vector for sample={key}")
        samples.append(
            PredictionSample(
                repeat_index=key[0],
                frame_index=key[1],
                target_frame_index=key[2],
                split=key[3],
                tile_ids=tile_ids,
                target=np.asarray([by_tile[tile_id] for tile_id in tile_ids], dtype=float),
            )
        )
    if not samples:
        raise DiagnosticError(f"{path} contains no prediction samples")
    return samples


def available_sources(
    fieldnames: Sequence[str],
    *,
    include_predictor_output_sources: bool = False,
) -> list[SourceDefinition]:
    fields = set(fieldnames)
    candidates = [
        SourceDefinition("l23e_current", ("l23e_state_norm",), "l23e_rate_hz"),
        SourceDefinition("hva_e", ("hva_e_state_norm",), "hva_e_rate_hz"),
        SourceDefinition("hva_ctx_slow", ("hva_e_slow_context_state_norm",), "hva_e_slow_context_rate_hz"),
        SourceDefinition("hva_ctx_transition", ("hva_ctx_transition_state_norm",), "hva_e_slow_context_rate_hz"),
        SourceDefinition(
            "hva_ctx_combined",
            ("hva_e_slow_context_state_norm", "hva_ctx_transition_state_norm"),
            "hva_e_slow_context_rate_hz",
        ),
        SourceDefinition(
            "hva_pred_membrane",
            ("hva_pred_e_membrane_state_norm",),
            "hva_pred_e_spike_rate_hz",
            source_guidance_allowed=False,
            predictor_output_state_source=True,
        ),
        SourceDefinition(
            "hva_pred_synaptic_current",
            ("hva_pred_e_synaptic_current_state_norm",),
            "hva_pred_e_spike_rate_hz",
            source_guidance_allowed=False,
            predictor_output_state_source=True,
        ),
        SourceDefinition(
            "hva_pred_signed_residual",
            ("hva_pred_e_signed_residual_state_norm",),
            "hva_pred_e_spike_rate_hz",
            source_guidance_allowed=False,
            predictor_output_state_source=True,
        ),
        SourceDefinition(
            "hva_pred_all_state",
            (
                "hva_pred_e_membrane_state_norm",
                "hva_pred_e_synaptic_current_state_norm",
                "hva_pred_e_signed_residual_state_norm",
            ),
            "hva_pred_e_spike_rate_hz",
            source_guidance_allowed=False,
            predictor_output_state_source=True,
        ),
        SourceDefinition("l23e_history", tuple(), "l23e_rate_hz", "l23e_state_norm", "l23e_rate_hz"),
    ]
    return [
        source
        for source in candidates
        if (include_predictor_output_sources or not source.predictor_output_state_source)
        and (
            (
                source.history_source_column is not None
                and source.history_source_column in fields
            )
            or all(column in fields for column in source.state_columns)
        )
    ]


def clip_id_for_frame(frame_index: int, clip_length_frames: int) -> int:
    if clip_length_frames <= 0:
        return 0
    return frame_index // clip_length_frames


def same_clip(frame_a: int, frame_b: int, clip_length_frames: int) -> bool:
    return clip_id_for_frame(frame_a, clip_length_frames) == clip_id_for_frame(frame_b, clip_length_frames)


def filter_cross_clip_samples(
    samples: Sequence[PredictionSample],
    clip_length_frames: int,
) -> tuple[list[PredictionSample], int]:
    if clip_length_frames <= 0:
        return list(samples), 0
    filtered = [
        sample
        for sample in samples
        if same_clip(sample.frame_index, sample.target_frame_index, clip_length_frames)
    ]
    return filtered, len(samples) - len(filtered)


def tile_positions(
    sample: PredictionSample,
    rates: dict[tuple[int, int, int], dict[str, float]],
) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    for tile_id in sample.tile_ids:
        row = rates.get((sample.repeat_index, sample.frame_index, tile_id), {})
        if "tile_x" in row and "tile_y" in row:
            positions[tile_id] = (int(row["tile_x"]), int(row["tile_y"]))
    if len(positions) == len(sample.tile_ids):
        return positions

    tile_count = len(sample.tile_ids)
    grid_side = int(round(math.sqrt(tile_count)))
    if grid_side * grid_side == tile_count:
        return {tile_id: (index % grid_side, index // grid_side) for index, tile_id in enumerate(sample.tile_ids)}
    return {tile_id: (index, 0) for index, tile_id in enumerate(sample.tile_ids)}


def sample_source_vector(
    sample: PredictionSample,
    rates: dict[tuple[int, int, int], dict[str, float]],
    source: SourceDefinition,
    *,
    history_lags: Sequence[int],
    clip_length_frames: int,
) -> tuple[np.ndarray, list[float]]:
    features: list[float] = []
    source_rates_hz: list[float] = []
    if source.history_source_column is not None:
        for lag in history_lags:
            source_frame = sample.frame_index - lag
            for tile_id in sample.tile_ids:
                if source_frame < 0 or not same_clip(source_frame, sample.frame_index, clip_length_frames):
                    features.append(0.0)
                    continue
                row = rates.get((sample.repeat_index, source_frame, tile_id), {})
                features.append(row.get(source.history_source_column, 0.0))
                if lag == 0 and source.history_rate_column is not None and source.history_rate_column in row:
                    source_rates_hz.append(row[source.history_rate_column])
        return np.asarray(features, dtype=float), source_rates_hz

    for column in source.state_columns:
        for tile_id in sample.tile_ids:
            row = rates.get((sample.repeat_index, sample.frame_index, tile_id), {})
            features.append(row.get(column, 0.0))
            if source.rate_column is not None and source.rate_column in row:
                source_rates_hz.append(row[source.rate_column])
    return np.asarray(features, dtype=float), source_rates_hz


def design_matrix(
    samples: Sequence[PredictionSample],
    rates: dict[tuple[int, int, int], dict[str, float]],
    source: SourceDefinition,
    *,
    history_lags: Sequence[int],
    clip_length_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    source_rate_rows: list[float] = []
    for sample in samples:
        source_vector, source_rates = sample_source_vector(
            sample,
            rates,
            source,
            history_lags=history_lags,
            clip_length_frames=clip_length_frames,
        )
        x_rows.append(source_vector)
        y_rows.append(sample.target)
        source_rate_rows.extend(source_rates)
    if not x_rows:
        raise DiagnosticError(f"Source {source.name} has no samples")
    return np.vstack(x_rows), np.vstack(y_rows), np.asarray(source_rate_rows, dtype=float)


def local_source_vector(
    sample: PredictionSample,
    target_tile_id: int,
    rates: dict[tuple[int, int, int], dict[str, float]],
    source: SourceDefinition,
    *,
    history_lags: Sequence[int],
    clip_length_frames: int,
    local_radius_tiles: int,
) -> tuple[np.ndarray, list[float]]:
    positions = tile_positions(sample, rates)
    target_x, target_y = positions[target_tile_id]
    tile_by_position = {position: tile_id for tile_id, position in positions.items()}
    offsets = [
        (dx, dy)
        for dy in range(-local_radius_tiles, local_radius_tiles + 1)
        for dx in range(-local_radius_tiles, local_radius_tiles + 1)
    ]
    features: list[float] = []
    source_rates_hz: list[float] = []

    if source.history_source_column is not None:
        for lag in history_lags:
            source_frame = sample.frame_index - lag
            for dx, dy in offsets:
                source_tile_id = tile_by_position.get((target_x + dx, target_y + dy))
                if (
                    source_tile_id is None
                    or source_frame < 0
                    or not same_clip(source_frame, sample.frame_index, clip_length_frames)
                ):
                    features.append(0.0)
                    continue
                row = rates.get((sample.repeat_index, source_frame, source_tile_id), {})
                features.append(row.get(source.history_source_column, 0.0))
                if lag == 0 and source.history_rate_column is not None and source.history_rate_column in row:
                    source_rates_hz.append(row[source.history_rate_column])
        return np.asarray(features, dtype=float), source_rates_hz

    for column in source.state_columns:
        for dx, dy in offsets:
            source_tile_id = tile_by_position.get((target_x + dx, target_y + dy))
            if source_tile_id is None:
                features.append(0.0)
                continue
            row = rates.get((sample.repeat_index, sample.frame_index, source_tile_id), {})
            features.append(row.get(column, 0.0))
            if source.rate_column is not None and source.rate_column in row:
                source_rates_hz.append(row[source.rate_column])
    return np.asarray(features, dtype=float), source_rates_hz


def fit_local_models(
    train_samples: Sequence[PredictionSample],
    rates: dict[tuple[int, int, int], dict[str, float]],
    source: SourceDefinition,
    *,
    history_lags: Sequence[int],
    clip_length_frames: int,
    local_radius_tiles: int,
    ridge_alpha: float,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    if not train_samples:
        raise DiagnosticError("Cannot fit local source ceiling without train samples")
    tile_ids = train_samples[0].tile_ids
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    feature_dim = 0
    for target_index, target_tile_id in enumerate(tile_ids):
        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        for sample in train_samples:
            if sample.tile_ids != tile_ids:
                raise DiagnosticError("Prediction samples use inconsistent tile vectors")
            source_vector, _ = local_source_vector(
                sample,
                target_tile_id,
                rates,
                source,
                history_lags=history_lags,
                clip_length_frames=clip_length_frames,
                local_radius_tiles=local_radius_tiles,
            )
            x_rows.append(source_vector)
            y_rows.append(float(sample.target[target_index]))
        train_x = np.vstack(x_rows)
        train_y = np.asarray(y_rows, dtype=float).reshape((-1, 1))
        weights, mean, std = fit_ridge(train_x, train_y, ridge_alpha)
        models[target_tile_id] = (weights, mean, std)
        feature_dim = train_x.shape[1]
    return models, feature_dim


def predict_local_models(
    samples: Sequence[PredictionSample],
    rates: dict[tuple[int, int, int], dict[str, float]],
    source: SourceDefinition,
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    history_lags: Sequence[int],
    clip_length_frames: int,
    local_radius_tiles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise DiagnosticError(f"Source {source.name} has no local samples")
    tile_ids = samples[0].tile_ids
    y_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    source_rate_rows: list[float] = []
    for sample in samples:
        if sample.tile_ids != tile_ids:
            raise DiagnosticError("Prediction samples use inconsistent tile vectors")
        y_rows.append(sample.target)
        pred = np.zeros(len(tile_ids), dtype=float)
        for target_index, target_tile_id in enumerate(tile_ids):
            source_vector, source_rates = local_source_vector(
                sample,
                target_tile_id,
                rates,
                source,
                history_lags=history_lags,
                clip_length_frames=clip_length_frames,
                local_radius_tiles=local_radius_tiles,
            )
            weights, mean, std = models[target_tile_id]
            pred[target_index] = float(predict_ridge(source_vector.reshape((1, -1)), weights, mean, std)[0, 0])
            source_rate_rows.extend(source_rates)
        pred_rows.append(pred)
    return np.vstack(y_rows), np.vstack(pred_rows), np.asarray(source_rate_rows, dtype=float)


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_x.shape[0] == 0:
        raise DiagnosticError("Cannot fit ridge source ceiling without train samples")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1.0e-9] = 1.0
    z = (train_x - mean) / std
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ train_y
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ rhs
    return weights, mean, std


def predict_ridge(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    return design @ weights


def vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered)))
    return 0.0 if denom <= 0.0 else float(np.dot(a_centered, b_centered) / denom)


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return 0.0 if denom <= 0.0 else float(np.dot(a, b) / denom)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def safe_p99(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.percentile(values, 99.0))


def metric_row(
    *,
    delay: int,
    source: SourceDefinition,
    readout_mode: str,
    split: str,
    clip_id: str,
    samples: Sequence[PredictionSample],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_rates: np.ndarray,
    feature_dim: int,
    history_lags: Sequence[int],
    local_radius_tiles: int,
    skipped_cross_clip_count: int,
) -> dict[str, str]:
    corrs = [vector_corr(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
    cosines = [vector_cosine(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
    mses = [float(np.mean((y_true[i] - y_pred[i]) ** 2)) for i in range(y_true.shape[0])]
    return {
        "delay_frames": str(delay),
        "source": source.name,
        "readout_mode": readout_mode,
        "split": split,
        "clip_id": clip_id,
        "sample_count": str(len(samples)),
        "tile_count": str(y_true.shape[1] if y_true.ndim == 2 else 0),
        "feature_dim": str(feature_dim),
        "vector_corr_mean": f"{safe_mean(corrs):.9f}",
        "vector_cosine_mean": f"{safe_mean(cosines):.9f}",
        "mse_mean": f"{safe_mean(mses):.9f}",
        "target_mean_norm": f"{float(np.mean(y_true)) if y_true.size else 0.0:.9f}",
        "prediction_mean_norm": f"{float(np.mean(y_pred)) if y_pred.size else 0.0:.9f}",
        "source_rate_mean_hz": f"{float(np.mean(source_rates)) if source_rates.size else 0.0:.9f}",
        "source_rate_p99_hz": f"{safe_p99(source_rates):.9f}",
        "diagnostic_only": "1",
        "offline_ridge_ceiling": "1",
        "global_offline_ridge": "1" if readout_mode == "global_ridge" else "0",
        "local_window_readout": "1" if readout_mode == "local_window_ridge" else "0",
        "architecture_realizable_source_ceiling": "1" if readout_mode == "local_window_ridge" else "0",
        "primary_model_success_claim": "0",
        "uses_train_targets": "1",
        "heldout_updates_applied": "0",
        "skipped_cross_clip_count": str(skipped_cross_clip_count),
        "local_radius_tiles": str(local_radius_tiles if readout_mode == "local_window_ridge" else 0),
        "source_guidance_allowed": "1" if source.source_guidance_allowed else "0",
        "predictor_output_state_source": "1" if source.predictor_output_state_source else "0",
        "source_prohibited_for_source_guidance": "0" if source.source_guidance_allowed else "1",
        "source_columns": "|".join(source.state_columns) if source.state_columns else source.history_source_column or "",
        "history_lags": "|".join(str(lag) for lag in history_lags) if source.history_source_column else "",
    }


def rows_for_source_delay(
    *,
    delay: int,
    samples: Sequence[PredictionSample],
    rates: dict[tuple[int, int, int], dict[str, float]],
    source: SourceDefinition,
    history_lags: Sequence[int],
    clip_length_frames: int,
    local_radius_tiles: int,
    ridge_alpha: float,
    skipped_cross_clip_count: int,
) -> list[dict[str, str]]:
    train_samples = [sample for sample in samples if sample.split == "train"]
    heldout_samples = [sample for sample in samples if sample.split == "heldout"]
    if not train_samples:
        raise DiagnosticError(f"Delay {delay} source {source.name} has no train samples after clip filtering")
    train_x, train_y, train_source_rates = design_matrix(
        train_samples,
        rates,
        source,
        history_lags=history_lags,
        clip_length_frames=clip_length_frames,
    )
    weights, mean, std = fit_ridge(train_x, train_y, ridge_alpha)

    output_rows: list[dict[str, str]] = []
    for split, split_samples in [("train", train_samples), ("heldout", heldout_samples)]:
        if not split_samples:
            continue
        x, y, source_rates = design_matrix(
            split_samples,
            rates,
            source,
            history_lags=history_lags,
            clip_length_frames=clip_length_frames,
        )
        pred = predict_ridge(x, weights, mean, std)
        output_rows.append(
            metric_row(
                delay=delay,
                source=source,
                readout_mode="global_ridge",
                split=split,
                clip_id="all",
                samples=split_samples,
                y_true=y,
                y_pred=pred,
                source_rates=source_rates if source_rates.size else train_source_rates,
                feature_dim=x.shape[1],
                history_lags=history_lags,
                local_radius_tiles=local_radius_tiles,
                skipped_cross_clip_count=skipped_cross_clip_count,
            )
        )
        by_clip: dict[int, list[int]] = {}
        for index, sample in enumerate(split_samples):
            by_clip.setdefault(clip_id_for_frame(sample.frame_index, clip_length_frames), []).append(index)
        for clip_id, indices in sorted(by_clip.items()):
            clip_samples = [split_samples[index] for index in indices]
            clip_x, clip_y, clip_source_rates = design_matrix(
                clip_samples,
                rates,
                source,
                history_lags=history_lags,
                clip_length_frames=clip_length_frames,
            )
            clip_pred = predict_ridge(clip_x, weights, mean, std)
            output_rows.append(
                metric_row(
                    delay=delay,
                    source=source,
                    readout_mode="global_ridge",
                    split=split,
                    clip_id=str(clip_id),
                    samples=clip_samples,
                    y_true=clip_y,
                    y_pred=clip_pred,
                    source_rates=clip_source_rates if clip_source_rates.size else source_rates,
                    feature_dim=clip_x.shape[1],
                    history_lags=history_lags,
                    local_radius_tiles=local_radius_tiles,
                    skipped_cross_clip_count=skipped_cross_clip_count,
                )
            )

    local_models, local_feature_dim = fit_local_models(
        train_samples,
        rates,
        source,
        history_lags=history_lags,
        clip_length_frames=clip_length_frames,
        local_radius_tiles=local_radius_tiles,
        ridge_alpha=ridge_alpha,
    )
    for split, split_samples in [("train", train_samples), ("heldout", heldout_samples)]:
        if not split_samples:
            continue
        y, pred, source_rates = predict_local_models(
            split_samples,
            rates,
            source,
            local_models,
            history_lags=history_lags,
            clip_length_frames=clip_length_frames,
            local_radius_tiles=local_radius_tiles,
        )
        output_rows.append(
            metric_row(
                delay=delay,
                source=source,
                readout_mode="local_window_ridge",
                split=split,
                clip_id="all",
                samples=split_samples,
                y_true=y,
                y_pred=pred,
                source_rates=source_rates if source_rates.size else train_source_rates,
                feature_dim=local_feature_dim,
                history_lags=history_lags,
                local_radius_tiles=local_radius_tiles,
                skipped_cross_clip_count=skipped_cross_clip_count,
            )
        )
        by_clip: dict[int, list[int]] = {}
        for index, sample in enumerate(split_samples):
            by_clip.setdefault(clip_id_for_frame(sample.frame_index, clip_length_frames), []).append(index)
        for clip_id, indices in sorted(by_clip.items()):
            clip_samples = [split_samples[index] for index in indices]
            clip_y, clip_pred, clip_source_rates = predict_local_models(
                clip_samples,
                rates,
                source,
                local_models,
                history_lags=history_lags,
                clip_length_frames=clip_length_frames,
                local_radius_tiles=local_radius_tiles,
            )
            output_rows.append(
                metric_row(
                    delay=delay,
                    source=source,
                    readout_mode="local_window_ridge",
                    split=split,
                    clip_id=str(clip_id),
                    samples=clip_samples,
                    y_true=clip_y,
                    y_pred=clip_pred,
                    source_rates=clip_source_rates if clip_source_rates.size else source_rates,
                    feature_dim=local_feature_dim,
                    history_lags=history_lags,
                    local_radius_tiles=local_radius_tiles,
                    skipped_cross_clip_count=skipped_cross_clip_count,
                )
            )
    return output_rows


def write_metrics_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise DiagnosticError("No diagnostic rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "delay_frames",
        "source",
        "readout_mode",
        "split",
        "clip_id",
        "sample_count",
        "tile_count",
        "feature_dim",
        "vector_corr_mean",
        "vector_cosine_mean",
        "mse_mean",
        "target_mean_norm",
        "prediction_mean_norm",
        "source_rate_mean_hz",
        "source_rate_p99_hz",
        "diagnostic_only",
        "offline_ridge_ceiling",
        "global_offline_ridge",
        "local_window_readout",
        "architecture_realizable_source_ceiling",
        "primary_model_success_claim",
        "uses_train_targets",
        "heldout_updates_applied",
        "skipped_cross_clip_count",
        "local_radius_tiles",
        "source_guidance_allowed",
        "predictor_output_state_source",
        "source_prohibited_for_source_guidance",
        "source_columns",
        "history_lags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise DiagnosticError("Expected at least one integer value")
    return values


def run_diagnostics(
    *,
    genn_dir: Path,
    prefix: str | None,
    delays: Sequence[int],
    ridge_alpha: float,
    history_lags: Sequence[int],
    clip_length_frames: int | None,
    local_radius_tiles: int,
    include_predictor_output_sources: bool,
    output_path: Path | None,
) -> Path:
    resolved_prefix = resolve_prefix(genn_dir, prefix)
    summary_path = genn_dir / f"{resolved_prefix}_summary.csv"
    summary = parse_summary(summary_path if summary_path.is_file() else genn_dir / f"{resolved_prefix}_summary.txt")
    resolved_clip_length = (
        int(summary.get("video_clip_length_frames", 0.0))
        if clip_length_frames is None
        else clip_length_frames
    )
    rates, fieldnames = load_rates(genn_dir / f"{resolved_prefix}_spiking_hva_rates.csv")
    sources = available_sources(
        fieldnames,
        include_predictor_output_sources=include_predictor_output_sources,
    )
    if not sources:
        raise DiagnosticError("No source columns found in spiking-HVA rates CSV")

    rows: list[dict[str, str]] = []
    for delay in delays:
        all_samples = load_prediction_samples(genn_dir / f"{resolved_prefix}_spiking_hva_predictions_delay{delay}.csv")
        samples, skipped_cross_clip_count = filter_cross_clip_samples(all_samples, resolved_clip_length)
        if not samples:
            raise DiagnosticError(f"Delay {delay} has no prediction samples after clip filtering")
        for source in sources:
            rows.extend(
                rows_for_source_delay(
                    delay=delay,
                    samples=samples,
                    rates=rates,
                    source=source,
                    history_lags=history_lags,
                    clip_length_frames=resolved_clip_length,
                    local_radius_tiles=local_radius_tiles,
                    ridge_alpha=ridge_alpha,
                    skipped_cross_clip_count=skipped_cross_clip_count,
                )
            )

    metrics_path = output_path or genn_dir / f"{resolved_prefix}_spiking_hva_source_ceiling_metrics.csv"
    write_metrics_csv(metrics_path, rows)
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genn-dir", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--delays", default="1,3,5")
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument("--history-lags", default="0,1,2,3,4")
    parser.add_argument("--clip-length-frames", type=int)
    parser.add_argument("--local-radius-tiles", type=int, default=2)
    parser.add_argument(
        "--include-predictor-output-sources",
        action="store_true",
        help="Include HVA_PRED output-state sources; rows are marked prohibited for source guidance.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.ridge_alpha < 0.0 or not math.isfinite(args.ridge_alpha):
        raise DiagnosticError("--ridge-alpha must be finite and non-negative")
    if args.local_radius_tiles < 0:
        raise DiagnosticError("--local-radius-tiles must be non-negative")
    output_path = run_diagnostics(
        genn_dir=args.genn_dir,
        prefix=args.prefix,
        delays=parse_int_list(args.delays),
        ridge_alpha=args.ridge_alpha,
        history_lags=parse_int_list(args.history_lags),
        clip_length_frames=args.clip_length_frames,
        local_radius_tiles=args.local_radius_tiles,
        include_predictor_output_sources=args.include_predictor_output_sources,
        output_path=args.output,
    )
    print(f"wrote {output_path}")
    print("diagnostic_only=1 offline_ridge_ceiling=1 primary_model_success_claim=0")
    print("global_offline_ridge rows are not architecture-realizable; use local_window_ridge for local source ceilings")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"FAIL spiking_hva_source_ceiling: {exc}", file=sys.stderr)
        raise SystemExit(1)
