#!/usr/bin/env python3
"""Validate L2/3 population-state reliability from exported video site rates.

The primary input is the existing GeNN natural-video site-rate export:

    repeat_index,frame_index,population,site_id,rate_hz

No biological pass thresholds are assumed. The tool reports measured metrics,
deterministic shuffle controls, and explicit missing-input diagnostics for
metrics that cannot be computed from the supplied artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SITE_RATE_COLUMNS = {"repeat_index", "frame_index", "population", "site_id", "rate_hz"}
FRAME_SUMMARY_COLUMNS = {"repeat_index", "frame_index", "frame_start_ms", "frame_end_ms"}
DEFAULT_SHUFFLE_COUNT = 100


class InputError(ValueError):
    """Raised when an artifact is missing required schema or data."""


@dataclass(frozen=True)
class ArtifactInput:
    """Resolved video site-rate artifact and optional matching frame summary."""

    site_rates_path: Path
    frame_summary_path: Path | None
    expected_frame_summary_path: Path | None


@dataclass(frozen=True)
class PopulationActivity:
    """Dense population activity matrix loaded from one site-rate CSV.

    Attributes:
        rates_hz: Activity matrix with shape ``[repeat, frame, site]``.
        spike_counts: Optional spike-count matrix with the same shape, loaded
            only if the site-rate CSV includes a complete ``spike_count`` column.
    """

    source_path: Path
    population: str
    repeats: list[int]
    frames: list[int]
    sites: list[int]
    rates_hz: np.ndarray
    spike_counts: np.ndarray | None


@dataclass(frozen=True)
class MissingMetric:
    """A metric that could not be computed and the exact missing input."""

    metric: str
    reason: str
    required_file: str | None = None
    required_columns: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"metric": self.metric, "reason": self.reason}
        if self.required_file is not None:
            payload["required_file"] = self.required_file
        if self.required_columns is not None:
            payload["required_columns"] = self.required_columns
        return payload


@dataclass(frozen=True)
class Threshold:
    """User-supplied optional threshold check."""

    metric: str
    operator: str
    value: float


def parse_int(value: str, path: Path, row_number: int, column: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise InputError(f"{path} row {row_number} has invalid integer {column}={value!r}") from exc


def parse_float(value: str, path: Path, row_number: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise InputError(f"{path} row {row_number} has invalid float {column}={value!r}") from exc
    if not math.isfinite(parsed):
        raise InputError(f"{path} row {row_number} has non-finite {column}={value!r}")
    return parsed


def infer_frame_summary_path(site_rates_path: Path) -> Path | None:
    """Infer the matching ``*_video_frame_summary.csv`` path from a site-rate path."""
    name = site_rates_path.name
    suffix = "_video_site_rates.csv"
    if not name.endswith(suffix):
        return None
    return site_rates_path.with_name(name[: -len(suffix)] + "_video_frame_summary.csv")


def resolve_one_input(raw_path: Path, prefix: str | None) -> ArtifactInput:
    """Resolve a file, run directory, or artifact prefix into a site-rate CSV."""
    if raw_path.is_dir():
        if prefix is not None:
            site_rates_path = raw_path / f"{prefix}_video_site_rates.csv"
            expected_frame_summary_path = raw_path / f"{prefix}_video_frame_summary.csv"
            if not site_rates_path.is_file():
                raise InputError(f"Missing video site-rate file: {site_rates_path}")
        else:
            matches = sorted(raw_path.glob("*_video_site_rates.csv"))
            if len(matches) != 1:
                raise InputError(
                    f"{raw_path} contains {len(matches)} *_video_site_rates.csv files; "
                    "pass the CSV file directly or use --prefix."
                )
            site_rates_path = matches[0]
            expected_frame_summary_path = infer_frame_summary_path(site_rates_path)
    elif raw_path.is_file():
        site_rates_path = raw_path
        expected_frame_summary_path = infer_frame_summary_path(site_rates_path)
    else:
        candidate = Path(str(raw_path) + "_video_site_rates.csv")
        if not candidate.is_file():
            raise InputError(f"Missing video site-rate file: {raw_path}; also tried {candidate}")
        site_rates_path = candidate
        expected_frame_summary_path = infer_frame_summary_path(site_rates_path)

    frame_summary_path = (
        expected_frame_summary_path
        if expected_frame_summary_path is not None and expected_frame_summary_path.is_file()
        else None
    )
    return ArtifactInput(
        site_rates_path=site_rates_path,
        frame_summary_path=frame_summary_path,
        expected_frame_summary_path=expected_frame_summary_path,
    )


def resolve_inputs(
    raw_paths: Sequence[Path],
    prefix: str | None,
    frame_summary_override: Path | None,
) -> list[ArtifactInput]:
    if not raw_paths:
        raise InputError("At least one site-rate CSV, artifact directory, or artifact prefix is required.")
    if frame_summary_override is not None and len(raw_paths) != 1:
        raise InputError("--frame-summary can only be used with a single input artifact.")

    artifacts = [resolve_one_input(path, prefix) for path in raw_paths]
    if frame_summary_override is not None:
        if not frame_summary_override.is_file():
            raise InputError(f"Missing frame-summary file: {frame_summary_override}")
        artifacts = [
            ArtifactInput(
                site_rates_path=artifacts[0].site_rates_path,
                frame_summary_path=frame_summary_override,
                expected_frame_summary_path=frame_summary_override,
            )
        ]
    return artifacts


def load_population_activity(path: Path, population: str) -> PopulationActivity:
    """Load one population into a dense ``[repeat, frame, site]`` matrix."""
    records: dict[tuple[int, int, int], float] = {}
    count_records: dict[tuple[int, int, int], float] = {}
    repeats: set[int] = set()
    frames: set[int] = set()
    sites: set[int] = set()
    populations: set[str] = set()
    saw_spike_count_column = False
    missing_spike_count_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InputError(f"Missing header in {path}")
        missing_columns = sorted(SITE_RATE_COLUMNS.difference(reader.fieldnames))
        if missing_columns:
            raise InputError(
                f"Missing video site-rate columns in {path}: {missing_columns}; "
                f"required columns are {sorted(SITE_RATE_COLUMNS)}"
            )
        saw_spike_count_column = "spike_count" in reader.fieldnames

        for row_number, row in enumerate(reader, start=2):
            row_population = (row.get("population") or "").strip()
            if row_population:
                populations.add(row_population)
            if row_population != population:
                continue
            repeat = parse_int(row["repeat_index"], path, row_number, "repeat_index")
            frame = parse_int(row["frame_index"], path, row_number, "frame_index")
            site = parse_int(row["site_id"], path, row_number, "site_id")
            rate = parse_float(row["rate_hz"], path, row_number, "rate_hz")
            key = (repeat, frame, site)
            if key in records:
                raise InputError(
                    f"Duplicate video site-rate row in {path}: "
                    f"repeat_index={repeat}, frame_index={frame}, site_id={site}, population={population!r}"
                )
            records[key] = rate
            repeats.add(repeat)
            frames.add(frame)
            sites.add(site)
            if saw_spike_count_column:
                raw_count = row.get("spike_count", "")
                if raw_count == "":
                    missing_spike_count_rows += 1
                else:
                    count_records[key] = parse_float(raw_count, path, row_number, "spike_count")

    if not records:
        raise InputError(
            f"{path} contains no rows for population={population!r}; "
            f"available populations={sorted(populations)}"
        )

    repeat_list = sorted(repeats)
    frame_list = sorted(frames)
    site_list = sorted(sites)
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeat_list)}
    frame_to_index = {frame: index for index, frame in enumerate(frame_list)}
    site_to_index = {site: index for index, site in enumerate(site_list)}

    rates = np.full((len(repeat_list), len(frame_list), len(site_list)), np.nan, dtype=np.float64)
    spike_counts: np.ndarray | None = None
    if saw_spike_count_column and not missing_spike_count_rows and len(count_records) == len(records):
        spike_counts = np.full_like(rates, np.nan)

    for (repeat, frame, site), rate in records.items():
        index = (repeat_to_index[repeat], frame_to_index[frame], site_to_index[site])
        rates[index] = rate
        if spike_counts is not None:
            spike_counts[index] = count_records[(repeat, frame, site)]

    missing = np.argwhere(~np.isfinite(rates))
    if missing.size:
        first = missing[0]
        raise InputError(
            f"Incomplete video site-rate matrix for population={population!r} in {path}: "
            f"missing {len(missing)} rows; first missing "
            f"repeat_index={repeat_list[int(first[0])]}, "
            f"frame_index={frame_list[int(first[1])]}, "
            f"site_id={site_list[int(first[2])]}"
        )
    if spike_counts is not None and np.any(~np.isfinite(spike_counts)):
        spike_counts = None

    return PopulationActivity(
        source_path=path,
        population=population,
        repeats=repeat_list,
        frames=frame_list,
        sites=site_list,
        rates_hz=rates,
        spike_counts=spike_counts,
    )


def crop_population_activity(
    activity: PopulationActivity,
    *,
    sheet_side: int | None,
    core_side: int | None,
) -> PopulationActivity:
    """Return a central validation-core crop of a row-major sheet activity tensor.

    Site IDs are interpreted as ``site_id = y * sheet_side + x``. The crop is
    centered using integer offsets, matching the model's validation-core
    convention for the 40x40 sheet with a 32x32 core.
    """

    if core_side is None:
        return activity
    if sheet_side is None:
        raise InputError("--core-side requires --sheet-side")
    if sheet_side <= 0:
        raise InputError("--sheet-side must be positive")
    if core_side <= 0 or core_side > sheet_side:
        raise InputError("--core-side must be in (0, sheet_side]")

    offset = (sheet_side - core_side) // 2
    keep_sites = {
        (y * sheet_side) + x
        for y in range(offset, offset + core_side)
        for x in range(offset, offset + core_side)
    }
    keep_indices = [index for index, site in enumerate(activity.sites) if site in keep_sites]
    if not keep_indices:
        raise InputError(
            f"No site IDs survived core crop: sheet_side={sheet_side}, "
            f"core_side={core_side}, source={activity.source_path}"
        )

    rates = activity.rates_hz[:, :, keep_indices]
    spike_counts = activity.spike_counts[:, :, keep_indices] if activity.spike_counts is not None else None
    sites = [activity.sites[index] for index in keep_indices]
    return PopulationActivity(
        source_path=activity.source_path,
        population=activity.population,
        repeats=activity.repeats,
        frames=activity.frames,
        sites=sites,
        rates_hz=rates,
        spike_counts=spike_counts,
    )


def load_frame_durations_seconds(path: Path, repeats: Sequence[int], frames: Sequence[int]) -> np.ndarray:
    """Load frame durations aligned to ``[repeat, frame]`` from a frame-summary CSV."""
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeats)}
    frame_to_index = {frame: index for index, frame in enumerate(frames)}
    durations = np.full((len(repeats), len(frames)), np.nan, dtype=np.float64)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InputError(f"Missing header in {path}")
        missing_columns = sorted(FRAME_SUMMARY_COLUMNS.difference(reader.fieldnames))
        if missing_columns:
            raise InputError(
                f"Missing video frame-summary columns in {path}: {missing_columns}; "
                f"required columns are {sorted(FRAME_SUMMARY_COLUMNS)}"
            )
        for row_number, row in enumerate(reader, start=2):
            repeat = parse_int(row["repeat_index"], path, row_number, "repeat_index")
            frame = parse_int(row["frame_index"], path, row_number, "frame_index")
            if repeat not in repeat_to_index or frame not in frame_to_index:
                continue
            start_ms = parse_float(row["frame_start_ms"], path, row_number, "frame_start_ms")
            end_ms = parse_float(row["frame_end_ms"], path, row_number, "frame_end_ms")
            duration_s = (end_ms - start_ms) / 1000.0
            if duration_s <= 0.0:
                raise InputError(
                    f"{path} row {row_number} has non-positive frame duration: "
                    f"frame_start_ms={start_ms}, frame_end_ms={end_ms}"
                )
            durations[repeat_to_index[repeat], frame_to_index[frame]] = duration_s

    missing = np.argwhere(~np.isfinite(durations))
    if missing.size:
        first = missing[0]
        raise InputError(
            f"Incomplete video frame-summary durations in {path}: missing {len(missing)} rows; "
            f"first missing repeat_index={repeats[int(first[0])]}, frame_index={frames[int(first[1])]}"
        )
    return durations


def finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def finite_std(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.std(finite, ddof=1)) if len(finite) >= 2 else math.nan


def finite_min(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.min(finite)) if finite else math.nan


def finite_max(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.max(finite)) if finite else math.nan


def finite_array_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else math.nan


def finite_array_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else math.nan


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for degenerate vectors."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for correlation: {a.shape} != {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 2:
        return math.nan
    x = a[mask].astype(np.float64, copy=False)
    y = b[mask].astype(np.float64, copy=False)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(x, y) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity, or NaN for zero vectors."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for cosine similarity: {a.shape} != {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 1:
        return math.nan
    x = a[mask].astype(np.float64, copy=False)
    y = b[mask].astype(np.float64, copy=False)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(x, y) / denom)


def summarize_values(prefix: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": finite_mean(values),
        f"{prefix}_std": finite_std(values),
        f"{prefix}_min": finite_min(values),
        f"{prefix}_max": finite_max(values),
        f"{prefix}_n": float(sum(1 for value in values if math.isfinite(value))),
    }


def compute_repeat_reliability(
    rates: np.ndarray,
    rng: np.random.Generator,
    shuffle_count: int,
) -> tuple[dict[str, float], list[MissingMetric]]:
    """Compute repeat-pair population-vector correlations and shuffle controls."""
    repeat_count, frame_count, site_count = rates.shape
    missing: list[MissingMetric] = []
    if repeat_count < 2:
        missing.append(
            MissingMetric(
                metric="population_vector_repeat_correlation",
                reason=f"need at least 2 repeats; found {repeat_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, missing
    if site_count < 2:
        missing.append(
            MissingMetric(
                metric="population_vector_repeat_correlation",
                reason=f"need at least 2 sites for vector correlation; found {site_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, missing

    same_frame_corrs: list[float] = []
    flat_corrs: list[float] = []
    same_frame_cosines: list[float] = []
    frame_shuffle_means: list[float] = []
    site_shuffle_means: list[float] = []

    for first in range(repeat_count):
        for second in range(first + 1, repeat_count):
            flat_corrs.append(pearson_corr(rates[first].reshape(-1), rates[second].reshape(-1)))
            for frame in range(frame_count):
                same_frame_corrs.append(pearson_corr(rates[first, frame], rates[second, frame]))
                same_frame_cosines.append(cosine_similarity(rates[first, frame], rates[second, frame]))

            if frame_count >= 2:
                for _ in range(shuffle_count):
                    permutation = rng.permutation(frame_count)
                    shuffled = [
                        pearson_corr(rates[first, frame], rates[second, int(permutation[frame])])
                        for frame in range(frame_count)
                    ]
                    frame_shuffle_means.append(finite_mean(shuffled))

            if site_count >= 2:
                for _ in range(shuffle_count):
                    permutation = rng.permutation(site_count)
                    shuffled = [
                        pearson_corr(rates[first, frame], rates[second, frame, permutation])
                        for frame in range(frame_count)
                    ]
                    site_shuffle_means.append(finite_mean(shuffled))

    metrics: dict[str, float] = {}
    metrics.update(summarize_values("repeat_vector_corr", same_frame_corrs))
    metrics.update(summarize_values("repeat_flat_corr", flat_corrs))
    metrics.update(summarize_values("repeat_vector_cosine", same_frame_cosines))
    metrics.update(summarize_values("repeat_frame_shuffle_corr", frame_shuffle_means))
    metrics.update(summarize_values("repeat_site_shuffle_corr", site_shuffle_means))
    metrics["repeat_frame_shuffle_gap_mean"] = (
        metrics["repeat_vector_corr_mean"] - metrics["repeat_frame_shuffle_corr_mean"]
    )
    metrics["repeat_site_shuffle_gap_mean"] = (
        metrics["repeat_vector_corr_mean"] - metrics["repeat_site_shuffle_corr_mean"]
    )
    if metrics["repeat_vector_corr_n"] == 0.0:
        missing.append(
            MissingMetric(
                metric="population_vector_repeat_correlation",
                reason=(
                    "no finite per-frame repeat correlations; population vectors are "
                    "zero or constant across sites for all compared frames"
                ),
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
    return metrics, missing


def frame_rsm(frame_site_matrix: np.ndarray) -> np.ndarray:
    """Return frame-by-frame Pearson representational similarity matrix."""
    frame_count = frame_site_matrix.shape[0]
    rsm = np.full((frame_count, frame_count), np.nan, dtype=np.float64)
    for first in range(frame_count):
        for second in range(first, frame_count):
            corr = pearson_corr(frame_site_matrix[first], frame_site_matrix[second])
            rsm[first, second] = corr
            rsm[second, first] = corr
    return rsm


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def rsm_correlation(first_rsm: np.ndarray, second_rsm: np.ndarray) -> tuple[float, int]:
    first_values = upper_triangle_values(first_rsm)
    second_values = upper_triangle_values(second_rsm)
    mask = np.isfinite(first_values) & np.isfinite(second_values)
    if int(np.sum(mask)) < 2:
        return math.nan, int(np.sum(mask))
    return pearson_corr(first_values[mask], second_values[mask]), int(np.sum(mask))


def compute_odd_even_rsm(
    rates: np.ndarray,
    repeats: Sequence[int],
    rng: np.random.Generator,
    shuffle_count: int,
) -> tuple[dict[str, float], dict[str, Any], list[MissingMetric]]:
    """Compute odd/even repeat-split RSM reliability and controls."""
    repeat_count, frame_count, site_count = rates.shape
    missing: list[MissingMetric] = []
    details: dict[str, Any] = {}
    if repeat_count < 2:
        missing.append(
            MissingMetric(
                metric="odd_even_rsm_correlation",
                reason=f"need at least 2 repeats; found {repeat_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, details, missing
    if frame_count < 3:
        missing.append(
            MissingMetric(
                metric="odd_even_rsm_correlation",
                reason=f"need at least 3 frames for an RSM correlation; found {frame_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, details, missing
    if site_count < 2:
        missing.append(
            MissingMetric(
                metric="odd_even_rsm_correlation",
                reason=f"need at least 2 sites for frame RSMs; found {site_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, details, missing

    even_indices = list(range(0, repeat_count, 2))
    odd_indices = list(range(1, repeat_count, 2))
    if not even_indices or not odd_indices:
        missing.append(
            MissingMetric(
                metric="odd_even_rsm_correlation",
                reason=f"could not form non-empty alternating repeat splits from {repeat_count} repeats",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, details, missing

    even_template = np.mean(rates[even_indices], axis=0)
    odd_template = np.mean(rates[odd_indices], axis=0)
    even_rsm = frame_rsm(even_template)
    odd_rsm = frame_rsm(odd_template)
    observed_corr, pair_count = rsm_correlation(even_rsm, odd_rsm)

    frame_shuffle_corrs: list[float] = []
    site_shuffle_corrs: list[float] = []
    for _ in range(shuffle_count):
        permutation = rng.permutation(frame_count)
        shuffled_rsm = frame_rsm(odd_template[permutation])
        corr, _ = rsm_correlation(even_rsm, shuffled_rsm)
        frame_shuffle_corrs.append(corr)

    if site_count >= 2:
        for _ in range(shuffle_count):
            shuffled_odd = np.empty_like(odd_template)
            for frame in range(frame_count):
                shuffled_odd[frame] = odd_template[frame, rng.permutation(site_count)]
            corr, _ = rsm_correlation(even_rsm, frame_rsm(shuffled_odd))
            site_shuffle_corrs.append(corr)

    details["odd_even_split_even_repeats"] = [int(repeats[index]) for index in even_indices]
    details["odd_even_split_odd_repeats"] = [int(repeats[index]) for index in odd_indices]
    metrics = {
        "odd_even_rsm_corr": observed_corr,
        "odd_even_rsm_pair_count": float(pair_count),
    }
    metrics.update(summarize_values("odd_even_rsm_frame_shuffle_corr", frame_shuffle_corrs))
    metrics.update(summarize_values("odd_even_rsm_site_shuffle_corr", site_shuffle_corrs))
    metrics["odd_even_rsm_frame_shuffle_gap_mean"] = (
        metrics["odd_even_rsm_corr"] - metrics["odd_even_rsm_frame_shuffle_corr_mean"]
    )
    metrics["odd_even_rsm_site_shuffle_gap_mean"] = (
        metrics["odd_even_rsm_corr"] - metrics["odd_even_rsm_site_shuffle_corr_mean"]
    )
    if not math.isfinite(observed_corr):
        missing.append(
            MissingMetric(
                metric="odd_even_rsm_correlation",
                reason=(
                    "no finite odd/even RSM correlation; frame population vectors are "
                    "zero or constant across sites, or the RSM upper triangles have no variance"
                ),
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
    return metrics, details, missing


def row_normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 0.0
    normalized = np.zeros_like(matrix, dtype=np.float64)
    normalized[valid] = matrix[valid] / norms[valid, None]
    return normalized, valid


def decoder_metrics_from_scores(
    score_matrix: np.ndarray,
    sample_valid: np.ndarray,
    template_valid: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    decoded_count = 0
    top1 = 0
    topk = 0
    ranks: list[int] = []
    same_values: list[float] = []
    different_values: list[float] = []
    frame_count = score_matrix.shape[0]

    for frame in range(frame_count):
        if not sample_valid[frame] or not template_valid[frame]:
            continue
        valid_scores = np.array(score_matrix[frame], copy=True)
        valid_scores[~template_valid] = -np.inf
        true_score = float(valid_scores[frame])
        if not math.isfinite(true_score):
            continue
        rank = int(1 + np.sum(valid_scores > true_score))
        decoded_count += 1
        top1 += int(rank == 1)
        topk += int(rank <= top_k)
        ranks.append(rank)
        same_values.append(true_score)
        for other in range(frame_count):
            if other != frame and template_valid[other] and math.isfinite(float(valid_scores[other])):
                different_values.append(float(valid_scores[other]))

    return {
        "decoded_count": float(decoded_count),
        "top1_accuracy": (top1 / decoded_count) if decoded_count else math.nan,
        "topk_accuracy": (topk / decoded_count) if decoded_count else math.nan,
        "mean_rank": float(np.mean(ranks)) if ranks else math.nan,
        "median_rank": float(np.median(ranks)) if ranks else math.nan,
        "same_similarity": finite_mean(same_values),
        "different_similarity": finite_mean(different_values),
        "same_different_gap": (
            finite_mean(same_values) - finite_mean(different_values)
            if math.isfinite(finite_mean(same_values)) and math.isfinite(finite_mean(different_values))
            else math.nan
        ),
    }


def compute_heldout_decoder(
    rates: np.ndarray,
    rng: np.random.Generator,
    top_k: int,
    shuffle_count: int,
) -> tuple[dict[str, float], list[MissingMetric]]:
    """Decode frame identity from held-out repeats using nearest cosine template."""
    repeat_count, frame_count, _ = rates.shape
    missing: list[MissingMetric] = []
    if repeat_count < 2:
        missing.append(
            MissingMetric(
                metric="heldout_decoder",
                reason=f"need at least 2 repeats; found {repeat_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, missing
    if frame_count < 2:
        missing.append(
            MissingMetric(
                metric="heldout_decoder",
                reason=f"need at least 2 frames for frame decoding; found {frame_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, missing
    if top_k < 1:
        raise InputError(f"top_k must be >= 1; got {top_k}")
    effective_top_k = min(top_k, frame_count)

    decoded_counts: list[float] = []
    top1_values: list[float] = []
    topk_values: list[float] = []
    mean_ranks: list[float] = []
    median_ranks: list[float] = []
    same_values: list[float] = []
    different_values: list[float] = []
    gap_values: list[float] = []
    shuffle_top1_values: list[float] = []
    shuffle_topk_values: list[float] = []

    for heldout in range(repeat_count):
        train_indices = [index for index in range(repeat_count) if index != heldout]
        templates = np.mean(rates[train_indices], axis=0)
        samples = rates[heldout]
        samples_norm, sample_valid = row_normalize(samples)
        templates_norm, template_valid = row_normalize(templates)
        score_matrix = samples_norm @ templates_norm.T
        observed = decoder_metrics_from_scores(score_matrix, sample_valid, template_valid, effective_top_k)
        decoded_counts.append(observed["decoded_count"])
        top1_values.append(observed["top1_accuracy"])
        topk_values.append(observed["topk_accuracy"])
        mean_ranks.append(observed["mean_rank"])
        median_ranks.append(observed["median_rank"])
        same_values.append(observed["same_similarity"])
        different_values.append(observed["different_similarity"])
        gap_values.append(observed["same_different_gap"])

        for _ in range(shuffle_count):
            permutation = rng.permutation(frame_count)
            shuffled_templates = templates_norm[permutation]
            shuffled_template_valid = template_valid[permutation]
            shuffled_scores = samples_norm @ shuffled_templates.T
            shuffled = decoder_metrics_from_scores(
                shuffled_scores,
                sample_valid,
                shuffled_template_valid,
                effective_top_k,
            )
            shuffle_top1_values.append(shuffled["top1_accuracy"])
            shuffle_topk_values.append(shuffled["topk_accuracy"])

    metrics = {
        "heldout_decoder_decoded_count": float(np.sum(decoded_counts)),
        "heldout_decoder_top1_accuracy": finite_mean(top1_values),
        f"heldout_decoder_top{effective_top_k}_accuracy": finite_mean(topk_values),
        "heldout_decoder_mean_rank": finite_mean(mean_ranks),
        "heldout_decoder_median_rank": finite_mean(median_ranks),
        "heldout_decoder_same_similarity": finite_mean(same_values),
        "heldout_decoder_different_similarity": finite_mean(different_values),
        "heldout_decoder_same_different_gap": finite_mean(gap_values),
        "heldout_decoder_chance_top1": 1.0 / frame_count,
        f"heldout_decoder_chance_top{effective_top_k}": effective_top_k / frame_count,
    }
    metrics.update(summarize_values("heldout_decoder_frame_shuffle_top1", shuffle_top1_values))
    metrics.update(summarize_values(f"heldout_decoder_frame_shuffle_top{effective_top_k}", shuffle_topk_values))
    metrics["heldout_decoder_top1_shuffle_gap_mean"] = (
        metrics["heldout_decoder_top1_accuracy"] - metrics["heldout_decoder_frame_shuffle_top1_mean"]
    )
    metrics[f"heldout_decoder_top{effective_top_k}_shuffle_gap_mean"] = (
        metrics[f"heldout_decoder_top{effective_top_k}_accuracy"]
        - metrics[f"heldout_decoder_frame_shuffle_top{effective_top_k}_mean"]
    )
    if metrics["heldout_decoder_decoded_count"] == 0.0:
        missing.append(
            MissingMetric(
                metric="heldout_decoder",
                reason=(
                    "no held-out samples could be decoded; held-out or template population "
                    "vectors are zero for all frames"
                ),
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
    return metrics, missing


def resolve_spike_counts(
    activity: PopulationActivity,
    artifact: ArtifactInput,
) -> tuple[np.ndarray | None, str | None, MissingMetric | None]:
    """Return counts for Fano metrics, or a precise missing-input diagnostic."""
    if activity.spike_counts is not None:
        return activity.spike_counts, "spike_count", None
    if artifact.frame_summary_path is None:
        expected = artifact.expected_frame_summary_path
        required_file = str(expected) if expected is not None else None
        return (
            None,
            None,
            MissingMetric(
                metric="fano_variability_quenching",
                reason=(
                    "requires spike counts or frame durations; site-rate CSV has no complete "
                    "spike_count column and matching frame-summary file was not found"
                ),
                required_file=required_file,
                required_columns=sorted(FRAME_SUMMARY_COLUMNS),
            ),
        )
    try:
        durations_s = load_frame_durations_seconds(
            artifact.frame_summary_path,
            activity.repeats,
            activity.frames,
        )
    except InputError as exc:
        return (
            None,
            None,
            MissingMetric(
                metric="fano_variability_quenching",
                reason=str(exc),
                required_file=str(artifact.frame_summary_path),
                required_columns=sorted(FRAME_SUMMARY_COLUMNS),
            ),
        )
    return activity.rates_hz * durations_s[:, :, None], "rate_hz_x_frame_duration", None


def compute_fano_metrics(
    counts: np.ndarray,
    count_source: str,
    quench_window_frames: int | None,
) -> tuple[dict[str, float], dict[str, Any], list[MissingMetric]]:
    """Compute across-repeat Fano factors and early/late variability change."""
    repeat_count, frame_count, site_count = counts.shape
    missing: list[MissingMetric] = []
    details: dict[str, Any] = {"fano_count_source": count_source}
    if repeat_count < 2:
        missing.append(
            MissingMetric(
                metric="fano_variability_quenching",
                reason=f"need at least 2 repeats for across-repeat variance; found {repeat_count}",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, details, missing

    means = np.mean(counts, axis=0)
    variances = np.var(counts, axis=0, ddof=1)
    valid = means > 1.0e-12
    fano = np.full((frame_count, site_count), np.nan, dtype=np.float64)
    fano[valid] = variances[valid] / means[valid]
    valid_values = fano[np.isfinite(fano)]
    if valid_values.size == 0:
        missing.append(
            MissingMetric(
                metric="fano_variability_quenching",
                reason="all frame/site mean spike counts are zero; Fano denominator is undefined",
                required_columns=sorted(SITE_RATE_COLUMNS),
            )
        )
        return {}, details, missing

    if quench_window_frames is None:
        window = max(1, min(frame_count // 4, 20))
    else:
        window = quench_window_frames
    window = max(1, min(window, frame_count))
    early = fano[:window]
    late = fano[-window:]
    early_mean = finite_array_mean(early)
    late_mean = finite_array_mean(late)

    metrics = {
        "fano_mean": finite_array_mean(fano),
        "fano_median": finite_array_median(fano),
        "fano_valid_fraction": float(valid_values.size / fano.size),
        "fano_window_frames": float(window),
        "fano_early_mean": early_mean,
        "fano_late_mean": late_mean,
        "fano_early_minus_late": early_mean - late_mean,
        "fano_late_over_early": late_mean / early_mean if early_mean > 0.0 else math.nan,
    }
    return metrics, details, missing


def parse_threshold(raw: str) -> Threshold:
    for operator in (">=", "<="):
        if operator in raw:
            name, value = raw.split(operator, 1)
            return Threshold(name.strip(), operator, float(value.strip()))
    if "=" in raw:
        name, value = raw.split("=", 1)
        return Threshold(name.strip(), ">=", float(value.strip()))
    raise argparse.ArgumentTypeError(
        f"Invalid threshold {raw!r}; use metric>=value, metric<=value, or metric=value."
    )


def check_thresholds(metrics: dict[str, float], thresholds: Sequence[Threshold]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for threshold in thresholds:
        actual = metrics.get(threshold.metric, math.nan)
        if threshold.operator == ">=":
            passed = math.isfinite(actual) and actual >= threshold.value
        elif threshold.operator == "<=":
            passed = math.isfinite(actual) and actual <= threshold.value
        else:
            raise ValueError(f"unsupported threshold operator: {threshold.operator}")
        checks.append(
            {
                "metric": threshold.metric,
                "operator": threshold.operator,
                "threshold": threshold.value,
                "actual": actual,
                "passed": passed,
            }
        )
    return checks


def validate_artifact(
    artifact: ArtifactInput,
    population: str,
    seed: int,
    shuffle_count: int,
    top_k: int,
    quench_window_frames: int | None,
    thresholds: Sequence[Threshold],
    sheet_side: int | None = None,
    core_side: int | None = None,
) -> dict[str, Any]:
    """Compute all available population-state metrics for one artifact."""
    activity = load_population_activity(artifact.site_rates_path, population)
    activity = crop_population_activity(activity, sheet_side=sheet_side, core_side=core_side)
    rng = np.random.default_rng(seed)
    metrics: dict[str, float] = {
        "repeat_count": float(len(activity.repeats)),
        "frame_count": float(len(activity.frames)),
        "site_count": float(len(activity.sites)),
    }
    details: dict[str, Any] = {
        "source": str(artifact.site_rates_path),
        "population": population,
        "seed": seed,
        "shuffle_count": shuffle_count,
        "top_k": top_k,
        "repeat_indices": [int(value) for value in activity.repeats],
        "frame_index_first_last": [int(activity.frames[0]), int(activity.frames[-1])],
        "site_id_first_last": [int(activity.sites[0]), int(activity.sites[-1])],
        "sheet_side": sheet_side,
        "core_side": core_side,
        "frame_summary": str(artifact.frame_summary_path) if artifact.frame_summary_path is not None else None,
    }
    missing_metrics: list[MissingMetric] = []

    repeat_metrics, repeat_missing = compute_repeat_reliability(activity.rates_hz, rng, shuffle_count)
    metrics.update(repeat_metrics)
    missing_metrics.extend(repeat_missing)

    rsm_metrics, rsm_details, rsm_missing = compute_odd_even_rsm(
        activity.rates_hz,
        activity.repeats,
        rng,
        shuffle_count,
    )
    metrics.update(rsm_metrics)
    details.update(rsm_details)
    missing_metrics.extend(rsm_missing)

    decoder_metrics, decoder_missing = compute_heldout_decoder(
        activity.rates_hz,
        rng,
        top_k,
        shuffle_count,
    )
    metrics.update(decoder_metrics)
    missing_metrics.extend(decoder_missing)

    counts, count_source, fano_missing = resolve_spike_counts(activity, artifact)
    if fano_missing is not None:
        missing_metrics.append(fano_missing)
    elif counts is not None and count_source is not None:
        fano_metrics, fano_details, fano_compute_missing = compute_fano_metrics(
            counts,
            count_source,
            quench_window_frames,
        )
        metrics.update(fano_metrics)
        details.update(fano_details)
        missing_metrics.extend(fano_compute_missing)

    threshold_checks = check_thresholds(metrics, thresholds)
    return {
        "details": details,
        "metrics": metrics,
        "missing_metrics": [missing.to_json() for missing in missing_metrics],
        "threshold_checks": threshold_checks,
        "thresholds_passed": all(check["passed"] for check in threshold_checks),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.12g}"
    return str(value)


def emit_text(result: dict[str, Any]) -> None:
    details = result["details"]
    metrics = result["metrics"]
    print(f"source={details['source']}")
    print(f"population={details['population']}")
    print(f"repeat_indices={details['repeat_indices']}")
    print(f"frame_index_first_last={details['frame_index_first_last']}")
    print(f"site_id_first_last={details['site_id_first_last']}")
    print(f"sheet_side={details['sheet_side']}")
    print(f"core_side={details['core_side']}")
    print(f"frame_summary={details['frame_summary']}")
    if "odd_even_split_even_repeats" in details:
        print(f"odd_even_split_even_repeats={details['odd_even_split_even_repeats']}")
        print(f"odd_even_split_odd_repeats={details['odd_even_split_odd_repeats']}")
    if "fano_count_source" in details:
        print(f"fano_count_source={details['fano_count_source']}")
    for key in sorted(metrics):
        print(f"{key}={format_value(metrics[key])}")
    if result["threshold_checks"]:
        print("threshold_checks:")
        for check in result["threshold_checks"]:
            print(
                "  "
                f"{check['metric']} {check['operator']} {format_value(check['threshold'])}: "
                f"actual={format_value(check['actual'])} passed={check['passed']}"
            )
    if result["missing_metrics"]:
        print("missing_metrics:")
        for item in result["missing_metrics"]:
            required_file = f" required_file={item['required_file']}" if "required_file" in item else ""
            required_columns = (
                f" required_columns={item['required_columns']}" if "required_columns" in item else ""
            )
            print(f"  {item['metric']}: {item['reason']}{required_file}{required_columns}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate L2/3 population-state repeat reliability from exported "
            "*_video_site_rates.csv artifacts."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "One or more *_video_site_rates.csv files, artifact directories containing one "
            "such file, or artifact prefixes without the _video_site_rates.csv suffix."
        ),
    )
    parser.add_argument("--population", default="l23e", help="Population to validate, default: l23e.")
    parser.add_argument("--prefix", help="Artifact prefix to use when an input is a directory.")
    parser.add_argument(
        "--sheet-side",
        type=int,
        help="Full row-major sheet side used to interpret site IDs for --core-side cropping.",
    )
    parser.add_argument(
        "--core-side",
        type=int,
        help="Optional centered validation-core side to crop before computing metrics.",
    )
    parser.add_argument(
        "--frame-summary",
        type=Path,
        help=(
            "Optional matching *_video_frame_summary.csv for one input. If omitted, the "
            "tool auto-detects a sibling file when the site-rate name follows the export convention."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for shuffle controls.")
    parser.add_argument(
        "--shuffle-count",
        type=int,
        default=DEFAULT_SHUFFLE_COUNT,
        help=f"Number of deterministic shuffle samples, default: {DEFAULT_SHUFFLE_COUNT}.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k frame decoder accuracy to report.")
    parser.add_argument(
        "--quench-window-frames",
        type=int,
        help=(
            "Frame count for early-vs-late Fano summaries. Default uses the first/last "
            "quartile capped at 20 frames."
        ),
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=parse_threshold,
        default=[],
        help=(
            "Optional threshold check. Use metric>=value or metric<=value. "
            "metric=value is treated as metric>=value. No thresholds are applied by default."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument("--output-json", type=Path, help="Write JSON results to this path as well.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.shuffle_count < 0:
        parser.error("--shuffle-count must be >= 0")
    if args.quench_window_frames is not None and args.quench_window_frames < 1:
        parser.error("--quench-window-frames must be >= 1")
    if args.core_side is not None and args.sheet_side is None:
        parser.error("--core-side requires --sheet-side")

    try:
        artifacts = resolve_inputs(args.inputs, args.prefix, args.frame_summary)
        results = [
            validate_artifact(
                artifact=artifact,
                population=args.population,
                seed=args.seed,
                shuffle_count=args.shuffle_count,
                top_k=args.top_k,
                quench_window_frames=args.quench_window_frames,
                thresholds=args.threshold,
                sheet_side=args.sheet_side,
                core_side=args.core_side,
            )
            for artifact in artifacts
        ]
    except InputError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    payload = {
        "validator": "validate_l23_population_state",
        "result_count": len(results),
        "results": results,
    }
    ready_payload = json_ready(payload)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(ready_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(ready_payload, indent=2, sort_keys=True))
    else:
        for index, result in enumerate(results):
            if index:
                print()
            emit_text(result)

    return 1 if any(not result["thresholds_passed"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
