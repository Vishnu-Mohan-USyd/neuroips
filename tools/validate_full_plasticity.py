#!/usr/bin/env python3
"""Strict validator for full-plasticity GeNN export artifacts.

This script validates three experiment prefixes:

- full: the default full-plasticity run
- control: a no-learning or disabled-plasticity control
- somoff: a SOM-output ablation run

It enforces Dalton's post-run gates over OSI, recurrent/inhibitory weight
changes, rate sanity, broad SOM suppression, and VIP-learning exclusion.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ValidationError(RuntimeError):
    """Raised when required outputs are missing or malformed."""


@dataclass(frozen=True)
class WeightSpec:
    """Configuration for a validated weight family."""

    name: str
    before_suffix: str
    after_suffix: str
    lower: float
    upper: float
    sign: str


@dataclass
class WeightSeries:
    """Parsed sparse weight vector indexed by synapse order."""

    indices: list[int]
    values: list[float]


@dataclass
class WeightMetrics:
    """Summary statistics comparing before/after weights."""

    active_count: int
    threshold: float
    changed_fraction: float
    p95_abs_change: float
    lower_fraction: float
    upper_fraction: float
    max_abs_change: float
    min_nonzero: float | None
    max_nonzero: float | None


@dataclass
class ContextRow:
    """Center-vs-broad validation row for one population."""

    condition: str
    population: str
    site_id: int
    som_output_scale: float
    mean_rate_hz: float
    rates_by_deg: dict[float, float]


@dataclass(frozen=True)
class SizeTuningRow:
    """One central-site size tuning response."""

    radius_sites: float
    population: str
    site_id: int
    som_output_scale: float
    orientation_deg: float
    rate_hz: float


@dataclass(frozen=True)
class CellTuningRow:
    """One L23E cell response vector across orientation."""

    cell_id: int
    site_id: int
    site_pref_deg: float
    pref_deg: float
    rates_by_deg: dict[float, float]
    mean_rate_hz: float
    peak_rate_hz: float
    osi: float
    recurrent_output_scale: float | None


@dataclass(frozen=True)
class SpecificityRow:
    """One active L23E->L23E synapse annotated by orientation preference."""

    synapse_index: int
    pre_id: int
    post_id: int
    pre_site: int
    post_site: int
    distance_sites: float
    pre_pref_deg: float
    post_pref_deg: float
    delta_pref_deg: float
    w_before: float
    w_after: float
    delta_w: float
    pre_peak_hz: float
    post_peak_hz: float
    response_corr: float


@dataclass
class RateMetrics:
    """Per-population site-rate sanity metrics."""

    median_hz: float
    frac_below_1hz: float
    p99_hz: float


@dataclass
class RunData:
    """All parsed artifacts for one experiment prefix."""

    prefix: str
    summary: dict[str, float]
    context_rows: dict[tuple[str, str], ContextRow]
    post_site_rates: dict[str, list[float]]
    weights: dict[str, tuple[WeightSeries, WeightSeries]]
    vip_weight_files: list[Path]
    size_tuning_rows: list[SizeTuningRow] | None = None
    specificity_rows: list[SpecificityRow] | None = None


WEIGHT_SPECS = (
    WeightSpec(
        name="l23ee",
        before_suffix="_l23ee_weights_before.csv",
        after_suffix="_l23ee_weights_after.csv",
        lower=0.001,
        upper=0.010,
        sign="positive",
    ),
    WeightSpec(
        name="l23pv_to_l23e",
        before_suffix="_l23pv_to_l23e_weights_before.csv",
        after_suffix="_l23pv_to_l23e_weights_after.csv",
        lower=-0.050,
        upper=-0.002,
        sign="negative",
    ),
    WeightSpec(
        name="l23som_to_l23e",
        before_suffix="_l23som_to_l23e_weights_before.csv",
        after_suffix="_l23som_to_l23e_weights_after.csv",
        lower=-0.040,
        upper=-0.001,
        sign="negative",
    ),
)

POST_SITE_SUFFIXES = {
    "l23e": "_post_l23_sites.csv",
    "l23pv": "_post_l23pv_sites.csv",
    "l23som": "_post_l23som_sites.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GeNN full-plasticity outputs against strict gates.",
    )
    parser.add_argument(
        "--genn-dir",
        required=True,
        type=Path,
        help="Directory containing GeNN CSV outputs.",
    )
    parser.add_argument(
        "--full",
        required=True,
        help="Prefix for the full-plasticity run, e.g. v1_fp_full.",
    )
    parser.add_argument(
        "--control",
        required=True,
        help="Prefix for the no-learning control run, e.g. v1_fp_control.",
    )
    parser.add_argument(
        "--somoff",
        required=True,
        help="Prefix for the SOM-output ablation run, e.g. v1_fp_somoff.",
    )
    parser.add_argument(
        "--recoff",
        help="Optional prefix for recurrence-context ablation run, e.g. v1_fp_recoff.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValidationError(f"Missing required file: {path}")
    return path


def parse_float(raw: str, path: Path, row_number: int, column: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Non-numeric value in {path} row {row_number} column {column}: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValidationError(
            f"Non-finite value in {path} row {row_number} column {column}: {raw!r}"
        )
    return value


def parse_int(raw: str, path: Path, row_number: int, column: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Non-integer value in {path} row {row_number} column {column}: {raw!r}"
        ) from exc
    return value


def parse_summary_csv(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["metric", "value"]:
            raise ValidationError(f"Unexpected summary schema in {path}: {reader.fieldnames}")
        summary: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            metric = (row.get("metric") or "").strip()
            if not metric:
                raise ValidationError(f"Missing metric name in {path} row {row_number}")
            summary[metric] = parse_float(row["value"], path, row_number, "value")
    if not summary:
        raise ValidationError(f"Summary file is empty: {path}")
    return summary


def parse_site_rates_csv(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "mean_rate_hz" not in reader.fieldnames:
            raise ValidationError(f"Missing mean_rate_hz column in {path}")
        rates: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            for column in reader.fieldnames:
                raw = row.get(column)
                if raw is None:
                    raise ValidationError(f"Missing value in {path} row {row_number} column {column}")
                parse_float(raw, path, row_number, column)
            rates.append(float(row["mean_rate_hz"]))
    if not rates:
        raise ValidationError(f"Post site file is empty: {path}")
    return rates


def parse_rate_column_name(column: str, path: Path) -> float:
    prefix = "rate_"
    suffix = "deg_hz"
    if not column.startswith(prefix) or not column.endswith(suffix):
        raise ValidationError(f"Unexpected rate column in {path}: {column}")
    return float(column[len(prefix) : -len(suffix)])


def parse_context_csv(path: Path) -> dict[tuple[str, str], ContextRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {"condition", "population", "site_id", "som_output_scale", "mean_rate_hz"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing context columns in {path}: {sorted(missing)}")

        rate_columns = [column for column in reader.fieldnames if column.startswith("rate_")]
        if not rate_columns:
            raise ValidationError(f"No orientation rate columns found in {path}")

        rows: dict[tuple[str, str], ContextRow] = {}
        for row_number, row in enumerate(reader, start=2):
            condition = (row.get("condition") or "").strip()
            population = (row.get("population") or "").strip()
            if not condition or not population:
                raise ValidationError(f"Missing condition/population in {path} row {row_number}")

            site_id = parse_int(row["site_id"], path, row_number, "site_id")
            som_output_scale = parse_float(row["som_output_scale"], path, row_number, "som_output_scale")
            mean_rate_hz = parse_float(row["mean_rate_hz"], path, row_number, "mean_rate_hz")

            rates_by_deg: dict[float, float] = {}
            for column in rate_columns:
                rates_by_deg[parse_rate_column_name(column, path)] = parse_float(
                    row[column], path, row_number, column
                )

            key = (condition, population)
            rows[key] = ContextRow(
                condition=condition,
                population=population,
                site_id=site_id,
                som_output_scale=som_output_scale,
                mean_rate_hz=mean_rate_hz,
                rates_by_deg=rates_by_deg,
            )

    expected_keys = {
        ("center_only", "l23e"),
        ("center_only", "l23pv"),
        ("center_only", "l23som"),
        ("broad_field", "l23e"),
        ("broad_field", "l23pv"),
        ("broad_field", "l23som"),
    }
    missing_keys = expected_keys.difference(rows)
    if missing_keys:
        raise ValidationError(f"Missing context rows in {path}: {sorted(missing_keys)}")
    return rows


def parse_size_tuning_csv(path: Path) -> list[SizeTuningRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "radius_sites",
            "population",
            "site_id",
            "som_output_scale",
            "orientation_deg",
            "rate_hz",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing size tuning columns in {path}: {sorted(missing)}")

        rows: list[SizeTuningRow] = []
        for row_number, row in enumerate(reader, start=2):
            population = (row.get("population") or "").strip()
            if not population:
                raise ValidationError(f"Missing population in {path} row {row_number}")
            rows.append(
                SizeTuningRow(
                    radius_sites=parse_float(row["radius_sites"], path, row_number, "radius_sites"),
                    population=population,
                    site_id=parse_int(row["site_id"], path, row_number, "site_id"),
                    som_output_scale=parse_float(row["som_output_scale"], path, row_number, "som_output_scale"),
                    orientation_deg=parse_float(row["orientation_deg"], path, row_number, "orientation_deg"),
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                )
            )

    if not rows:
        raise ValidationError(f"Size tuning file is empty: {path}")
    return rows


def parse_cell_tuning_csv(path: Path) -> dict[int, CellTuningRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {"cell_id", "site_id", "site_pref_deg", "pref_deg", "mean_rate_hz", "peak_rate_hz", "osi"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing cell tuning columns in {path}: {sorted(missing)}")

        rate_columns = [column for column in reader.fieldnames if column.startswith("rate_")]
        if not rate_columns:
            raise ValidationError(f"No orientation rate columns found in {path}")

        rows: dict[int, CellTuningRow] = {}
        for row_number, row in enumerate(reader, start=2):
            cell_id = parse_int(row["cell_id"], path, row_number, "cell_id")
            if cell_id in rows:
                raise ValidationError(f"Duplicate cell_id {cell_id} in {path}")

            rates_by_deg = {
                parse_rate_column_name(column, path): parse_float(row[column], path, row_number, column)
                for column in rate_columns
            }
            recurrent_output_scale = None
            if "recurrent_output_scale" in reader.fieldnames:
                recurrent_output_scale = parse_float(
                    row["recurrent_output_scale"],
                    path,
                    row_number,
                    "recurrent_output_scale",
                )

            rows[cell_id] = CellTuningRow(
                cell_id=cell_id,
                site_id=parse_int(row["site_id"], path, row_number, "site_id"),
                site_pref_deg=parse_float(row["site_pref_deg"], path, row_number, "site_pref_deg"),
                pref_deg=parse_float(row["pref_deg"], path, row_number, "pref_deg"),
                rates_by_deg=rates_by_deg,
                mean_rate_hz=parse_float(row["mean_rate_hz"], path, row_number, "mean_rate_hz"),
                peak_rate_hz=parse_float(row["peak_rate_hz"], path, row_number, "peak_rate_hz"),
                osi=parse_float(row["osi"], path, row_number, "osi"),
                recurrent_output_scale=recurrent_output_scale,
            )

    if not rows:
        raise ValidationError(f"Cell tuning file is empty: {path}")
    return rows


def parse_specificity_csv(path: Path) -> list[SpecificityRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "synapse_index",
            "pre_id",
            "post_id",
            "pre_site",
            "post_site",
            "distance_sites",
            "pre_pref_deg",
            "post_pref_deg",
            "delta_pref_deg",
            "w_before",
            "w_after",
            "delta_w",
            "pre_peak_hz",
            "post_peak_hz",
            "response_corr",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing specificity columns in {path}: {sorted(missing)}")

        rows: list[SpecificityRow] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                SpecificityRow(
                    synapse_index=parse_int(row["synapse_index"], path, row_number, "synapse_index"),
                    pre_id=parse_int(row["pre_id"], path, row_number, "pre_id"),
                    post_id=parse_int(row["post_id"], path, row_number, "post_id"),
                    pre_site=parse_int(row["pre_site"], path, row_number, "pre_site"),
                    post_site=parse_int(row["post_site"], path, row_number, "post_site"),
                    distance_sites=parse_float(row["distance_sites"], path, row_number, "distance_sites"),
                    pre_pref_deg=parse_float(row["pre_pref_deg"], path, row_number, "pre_pref_deg"),
                    post_pref_deg=parse_float(row["post_pref_deg"], path, row_number, "post_pref_deg"),
                    delta_pref_deg=parse_float(row["delta_pref_deg"], path, row_number, "delta_pref_deg"),
                    w_before=parse_float(row["w_before"], path, row_number, "w_before"),
                    w_after=parse_float(row["w_after"], path, row_number, "w_after"),
                    delta_w=parse_float(row["delta_w"], path, row_number, "delta_w"),
                    pre_peak_hz=parse_float(row["pre_peak_hz"], path, row_number, "pre_peak_hz"),
                    post_peak_hz=parse_float(row["post_peak_hz"], path, row_number, "post_peak_hz"),
                    response_corr=parse_float(row["response_corr"], path, row_number, "response_corr"),
                )
            )

    if not rows:
        raise ValidationError(f"Specificity file is empty: {path}")
    return rows


def parse_weight_csv(path: Path) -> WeightSeries:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "synapse_index" not in reader.fieldnames or "g" not in reader.fieldnames:
            raise ValidationError(f"Unexpected weight schema in {path}: {reader.fieldnames}")

        indices: list[int] = []
        values: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            indices.append(parse_int(row["synapse_index"], path, row_number, "synapse_index"))
            values.append(parse_float(row["g"], path, row_number, "g"))
            for column in reader.fieldnames:
                if column in {"synapse_index", "g"}:
                    continue
                raw = row.get(column)
                if raw is None:
                    raise ValidationError(f"Missing value in {path} row {row_number} column {column}")
                parse_float(raw, path, row_number, column)

    if not values:
        raise ValidationError(f"Weight file is empty: {path}")
    return WeightSeries(indices=indices, values=values)


def percentile(values: Iterable[float], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValidationError("Cannot compute a percentile of an empty collection.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (q / 100.0)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    fraction = position - lower_index
    return ((1.0 - fraction) * lower) + (fraction * upper)


def median(values: Iterable[float]) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValidationError("Cannot compute median of an empty collection.")
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return 0.5 * (sorted_values[midpoint - 1] + sorted_values[midpoint])


def compare_weight_series(before: WeightSeries, after: WeightSeries, lower: float, upper: float) -> WeightMetrics:
    if before.indices != after.indices:
        raise ValidationError("Weight CSV synapse_index columns do not align between before/after files.")
    if len(before.values) != len(after.values):
        raise ValidationError("Weight CSV lengths do not match between before/after files.")

    active_changes: list[float] = []
    active_after_values: list[float] = []
    nonzero_values: list[float] = []
    tolerance = 1.0e-9
    threshold = 0.01 * abs(upper - lower)
    lower_hits = 0
    upper_hits = 0

    for before_g, after_g in zip(before.values, after.values):
        if before_g != 0.0 or after_g != 0.0:
            active_changes.append(abs(after_g - before_g))
            active_after_values.append(after_g)
            if abs(after_g - lower) <= tolerance:
                lower_hits += 1
            if abs(after_g - upper) <= tolerance:
                upper_hits += 1
            if before_g != 0.0:
                nonzero_values.append(before_g)
            if after_g != 0.0:
                nonzero_values.append(after_g)

    if not active_changes:
        raise ValidationError("Weight comparison has no active rows.")

    active_count = len(active_changes)
    changed_count = sum(change >= threshold for change in active_changes)
    return WeightMetrics(
        active_count=active_count,
        threshold=threshold,
        changed_fraction=changed_count / active_count,
        p95_abs_change=percentile(active_changes, 95.0),
        lower_fraction=lower_hits / active_count,
        upper_fraction=upper_hits / active_count,
        max_abs_change=max(active_changes),
        min_nonzero=min(nonzero_values) if nonzero_values else None,
        max_nonzero=max(nonzero_values) if nonzero_values else None,
    )


def load_run(
    genn_dir: Path,
    prefix: str,
    *,
    require_size_tuning: bool = False,
    require_specificity: bool = False,
) -> RunData:
    summary = parse_summary_csv(require_file(genn_dir / f"{prefix}_summary.csv"))
    context_rows = parse_context_csv(require_file(genn_dir / f"{prefix}_som_context_validation.csv"))

    post_site_rates = {
        population: parse_site_rates_csv(require_file(genn_dir / f"{prefix}{suffix}"))
        for population, suffix in POST_SITE_SUFFIXES.items()
    }

    weights: dict[str, tuple[WeightSeries, WeightSeries]] = {}
    for spec in WEIGHT_SPECS:
        before = parse_weight_csv(require_file(genn_dir / f"{prefix}{spec.before_suffix}"))
        after = parse_weight_csv(require_file(genn_dir / f"{prefix}{spec.after_suffix}"))
        weights[spec.name] = (before, after)

    vip_weight_files = sorted(genn_dir.glob(f"{prefix}*vip*weights*.csv"))
    size_tuning_rows = (
        parse_size_tuning_csv(require_file(genn_dir / f"{prefix}_size_tuning.csv"))
        if require_size_tuning
        else None
    )
    specificity_rows = (
        parse_specificity_csv(require_file(genn_dir / f"{prefix}_l23ee_specificity.csv"))
        if require_specificity
        else None
    )

    return RunData(
        prefix=prefix,
        summary=summary,
        context_rows=context_rows,
        post_site_rates=post_site_rates,
        weights=weights,
        vip_weight_files=vip_weight_files,
        size_tuning_rows=size_tuning_rows,
        specificity_rows=specificity_rows,
    )


def require_summary_metric(run: RunData, metric: str) -> float:
    if metric not in run.summary:
        raise ValidationError(f"Missing summary metric {metric!r} in prefix {run.prefix}")
    value = run.summary[metric]
    if not math.isfinite(value):
        raise ValidationError(f"Non-finite summary metric {metric!r} in prefix {run.prefix}")
    return value


def compute_rate_metrics(rates: list[float]) -> RateMetrics:
    if not rates:
        raise ValidationError("Rate sanity requested for an empty rate vector.")
    return RateMetrics(
        median_hz=median(rates),
        frac_below_1hz=sum(rate < 1.0 for rate in rates) / len(rates),
        p99_hz=percentile(rates, 99.0),
    )


def sign_passes(metrics: WeightMetrics, sign: str) -> bool:
    if metrics.min_nonzero is None or metrics.max_nonzero is None:
        return False
    if sign == "positive":
        return metrics.min_nonzero > 0.0
    if sign == "negative":
        return metrics.max_nonzero < 0.0
    raise ValidationError(f"Unsupported sign gate: {sign}")


def compute_context_metrics(run: RunData, orientation_deg: float) -> dict[str, float]:
    center_l23e = run.context_rows[("center_only", "l23e")]
    broad_l23e = run.context_rows[("broad_field", "l23e")]
    center_l23som = run.context_rows[("center_only", "l23som")]
    broad_l23som = run.context_rows[("broad_field", "l23som")]

    if orientation_deg not in center_l23e.rates_by_deg:
        raise ValidationError(
            f"Orientation {orientation_deg} not present in center-only L23E context row for {run.prefix}"
        )
    if orientation_deg not in broad_l23e.rates_by_deg:
        raise ValidationError(
            f"Orientation {orientation_deg} not present in broad-field L23E context row for {run.prefix}"
        )
    if orientation_deg not in center_l23som.rates_by_deg or orientation_deg not in broad_l23som.rates_by_deg:
        raise ValidationError(
            f"Orientation {orientation_deg} not present in SOM context rows for {run.prefix}"
        )

    center_pref_l23e = center_l23e.rates_by_deg[orientation_deg]
    broad_pref_l23e = broad_l23e.rates_by_deg[orientation_deg]
    center_pref_l23som = center_l23som.rates_by_deg[orientation_deg]
    broad_pref_l23som = broad_l23som.rates_by_deg[orientation_deg]

    if center_pref_l23e <= 0.0:
        raise ValidationError(
            f"Center-only preferred L23E rate must be positive for suppression computation in {run.prefix}"
        )
    if center_pref_l23som <= 0.0:
        raise ValidationError(
            f"Center-only preferred L23SOM rate must be positive for context validation in {run.prefix}"
        )

    driven_center_threshold_hz = max(10.0, 0.25 * center_pref_l23e)
    relevant_orientations: list[float] = []
    bsi_values: list[float] = []
    min_center_som_hz = math.inf
    min_broad_som_hz = math.inf

    for current_orientation_deg, center_rate in center_l23e.rates_by_deg.items():
        if current_orientation_deg not in broad_l23e.rates_by_deg:
            raise ValidationError(
                f"Orientation {current_orientation_deg} not present in broad-field L23E context row for {run.prefix}"
            )
        if current_orientation_deg not in center_l23som.rates_by_deg:
            raise ValidationError(
                f"Orientation {current_orientation_deg} not present in center-only L23SOM context row for {run.prefix}"
            )
        if current_orientation_deg not in broad_l23som.rates_by_deg:
            raise ValidationError(
                f"Orientation {current_orientation_deg} not present in broad-field L23SOM context row for {run.prefix}"
            )

        if center_rate < driven_center_threshold_hz:
            continue

        relevant_orientations.append(current_orientation_deg)
        broad_rate = broad_l23e.rates_by_deg[current_orientation_deg]
        center_som_rate = center_l23som.rates_by_deg[current_orientation_deg]
        broad_som_rate = broad_l23som.rates_by_deg[current_orientation_deg]
        bsi_values.append((center_rate - broad_rate) / center_rate)
        min_center_som_hz = min(min_center_som_hz, center_som_rate)
        min_broad_som_hz = min(min_broad_som_hz, broad_som_rate)

    if not relevant_orientations:
        raise ValidationError(
            f"No orientations met the driven center threshold {driven_center_threshold_hz:.6f} Hz "
            f"for context validation in {run.prefix}"
        )

    summary_mean_bsi = None
    center_mean_key = "center_only_central_l23e_mean_rate_hz"
    broad_mean_key = "broad_field_central_l23e_mean_rate_hz"
    if center_mean_key in run.summary and broad_mean_key in run.summary:
        center_mean = require_summary_metric(run, center_mean_key)
        broad_mean = require_summary_metric(run, broad_mean_key)
        if center_mean > 0.0:
            summary_mean_bsi = (center_mean - broad_mean) / center_mean

    return {
        "center_pref_l23e_hz": center_pref_l23e,
        "broad_pref_l23e_hz": broad_pref_l23e,
        "center_pref_l23som_hz": center_pref_l23som,
        "broad_pref_l23som_hz": broad_pref_l23som,
        "preferred_bsi": (center_pref_l23e - broad_pref_l23e) / center_pref_l23e,
        "mean_bsi": sum(bsi_values) / len(bsi_values),
        "driven_center_threshold_hz": driven_center_threshold_hz,
        "relevant_orientation_count": float(len(relevant_orientations)),
        "min_center_som_hz": min_center_som_hz,
        "min_broad_som_hz": min_broad_som_hz,
        "summary_mean_bsi": summary_mean_bsi,
    }


def preferred_center_orientation_deg(run: RunData) -> tuple[float, float]:
    center_l23e = run.context_rows[("center_only", "l23e")]
    pref_deg, pref_rate = max(center_l23e.rates_by_deg.items(), key=lambda item: item[1])
    return pref_deg, pref_rate


def circular_orientation_delta_deg(first_deg: float, second_deg: float) -> float:
    delta = abs((first_deg - second_deg) % 180.0)
    return min(delta, 180.0 - delta)


def require_size_rows(run: RunData) -> list[SizeTuningRow]:
    if run.size_tuning_rows is None:
        raise ValidationError(f"Size tuning rows were not loaded for prefix {run.prefix}")
    return run.size_tuning_rows


def require_specificity_rows(run: RunData) -> list[SpecificityRow]:
    if run.specificity_rows is None:
        raise ValidationError(f"L23E specificity rows were not loaded for prefix {run.prefix}")
    return run.specificity_rows


def build_size_tuning_grid(
    run: RunData,
) -> tuple[dict[str, dict[float, dict[float, float]]], list[float], list[float]]:
    rows = require_size_rows(run)
    required_populations = {"l4e", "l23e", "l23pv", "l23som"}
    radii = sorted({row.radius_sites for row in rows})
    orientations = sorted({row.orientation_deg for row in rows})
    if len(radii) < 3:
        raise ValidationError(f"Size tuning requires at least 3 radii for prefix {run.prefix}")
    if not orientations:
        raise ValidationError(f"Size tuning requires at least one orientation for prefix {run.prefix}")

    grid: dict[str, dict[float, dict[float, float]]] = {
        population: {radius: {} for radius in radii}
        for population in required_populations
    }
    seen: set[tuple[str, float, float]] = set()
    for row in rows:
        if row.population not in required_populations:
            continue
        key = (row.population, row.radius_sites, row.orientation_deg)
        if key in seen:
            raise ValidationError(
                f"Duplicate size tuning row for {row.population} radius={row.radius_sites} "
                f"orientation={row.orientation_deg} in prefix {run.prefix}"
            )
        seen.add(key)
        grid[row.population][row.radius_sites][row.orientation_deg] = row.rate_hz

    expected = {(radius, orientation) for radius in radii for orientation in orientations}
    for population in required_populations:
        observed = {
            (radius, orientation)
            for radius, rates_by_orientation in grid[population].items()
            for orientation in rates_by_orientation
        }
        missing = expected.difference(observed)
        if missing:
            raise ValidationError(
                f"Missing size tuning rows for {population} in prefix {run.prefix}: "
                f"{sorted(missing)[:5]}"
            )

    return grid, radii, orientations


def summarize_size_curve(radii: list[float], rates: list[float]) -> dict[str, float]:
    if len(radii) != len(rates) or not rates:
        raise ValidationError("Size curve summary requires aligned non-empty radii and rates.")
    peak_index = max(range(len(rates)), key=lambda index: rates[index])
    peak_rate = rates[peak_index]
    large_rate = rates[-1]
    suppression = ((peak_rate - large_rate) / peak_rate) if peak_rate > 0.0 else -math.inf
    return {
        "peak_index": float(peak_index),
        "peak_radius": radii[peak_index],
        "peak_rate": peak_rate,
        "small_rate": rates[0],
        "large_rate": large_rate,
        "early_delta": peak_rate - rates[0],
        "suppression": suppression,
    }


def compute_size_tuning_metrics(
    run: RunData,
    *,
    selected_orientations: list[float] | None = None,
) -> dict[str, object]:
    grid, radii, orientations = build_size_tuning_grid(run)
    reference_radius = min(radii, key=lambda radius: abs(radius - 2.0))
    reference_rates = grid["l23e"][reference_radius]
    preferred_deg, preferred_rate = max(reference_rates.items(), key=lambda item: item[1])

    if selected_orientations is None:
        driven_threshold_hz = max(1.0, 0.25 * preferred_rate)
        selected_orientations = [
            orientation
            for orientation in orientations
            if circular_orientation_delta_deg(orientation, preferred_deg) <= 22.5
            and reference_rates[orientation] >= driven_threshold_hz
        ]
        if not selected_orientations:
            selected_orientations = [preferred_deg]

    missing_orientations = set(selected_orientations).difference(orientations)
    if missing_orientations:
        raise ValidationError(
            f"Size tuning orientations missing for prefix {run.prefix}: {sorted(missing_orientations)}"
        )

    mean_rates: dict[str, list[float]] = {}
    for population in ("l4e", "l23e", "l23pv", "l23som"):
        mean_rates[population] = [
            sum(grid[population][radius][orientation] for orientation in selected_orientations)
            / len(selected_orientations)
            for radius in radii
        ]

    return {
        "radii": radii,
        "selected_orientations": selected_orientations,
        "preferred_deg": preferred_deg,
        "preferred_rate": preferred_rate,
        "reference_radius": reference_radius,
        "mean_rates": mean_rates,
        "l23e": summarize_size_curve(radii, mean_rates["l23e"]),
        "l4e": summarize_size_curve(radii, mean_rates["l4e"]),
        "l23som": summarize_size_curve(radii, mean_rates["l23som"]),
    }


def format_float_list(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"


def mean(values: list[float]) -> float:
    if not values:
        raise ValidationError("Cannot compute mean of an empty collection.")
    return sum(values) / len(values)


def summarize_specificity_group(rows: list[SpecificityRow], prefix: str) -> dict[str, float]:
    if not rows:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_min_delta_pref_deg": math.nan,
            f"{prefix}_max_delta_pref_deg": math.nan,
            f"{prefix}_mean_delta_w": math.nan,
            f"{prefix}_median_delta_w": math.nan,
            f"{prefix}_mean_w_after": math.nan,
            f"{prefix}_median_w_after": math.nan,
        }
    return {
        f"{prefix}_count": float(len(rows)),
        f"{prefix}_min_delta_pref_deg": min(row.delta_pref_deg for row in rows),
        f"{prefix}_max_delta_pref_deg": max(row.delta_pref_deg for row in rows),
        f"{prefix}_mean_delta_w": mean([row.delta_w for row in rows]),
        f"{prefix}_median_delta_w": median(row.delta_w for row in rows),
        f"{prefix}_mean_w_after": mean([row.w_after for row in rows]),
        f"{prefix}_median_w_after": median(row.w_after for row in rows),
    }


def compute_specificity_metrics(rows: list[SpecificityRow]) -> dict[str, float]:
    if not rows:
        raise ValidationError("Specificity comparison requested for an empty row set.")

    sorted_rows = sorted(rows, key=lambda row: row.delta_pref_deg)
    quantile_count = max(1, len(sorted_rows) // 4)
    min_count = min(50, max(5, len(sorted_rows) // 20))
    low_delta = sorted_rows[:quantile_count]
    high_delta = sorted_rows[-quantile_count:]
    same = [row for row in rows if row.delta_pref_deg <= 22.5]
    orthogonal = [row for row in rows if row.delta_pref_deg >= 67.5]

    metrics = {
        "row_count": float(len(rows)),
        "quantile_fraction": 0.25,
        "min_count": float(min_count),
        "p95_abs_delta_w": percentile([abs(row.delta_w) for row in rows], 95.0),
    }
    metrics.update(summarize_specificity_group(low_delta, "low_delta"))
    metrics.update(summarize_specificity_group(high_delta, "high_delta"))
    metrics.update(summarize_specificity_group(same, "same"))
    metrics.update(summarize_specificity_group(orthogonal, "orthogonal"))
    return metrics


def summarize_correlation_group(rows: list[SpecificityRow], prefix: str) -> dict[str, float]:
    if not rows:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_min_response_corr": math.nan,
            f"{prefix}_max_response_corr": math.nan,
            f"{prefix}_mean_delta_w": math.nan,
            f"{prefix}_median_delta_w": math.nan,
            f"{prefix}_mean_w_after": math.nan,
            f"{prefix}_median_w_after": math.nan,
        }
    return {
        f"{prefix}_count": float(len(rows)),
        f"{prefix}_min_response_corr": min(row.response_corr for row in rows),
        f"{prefix}_max_response_corr": max(row.response_corr for row in rows),
        f"{prefix}_mean_delta_w": mean([row.delta_w for row in rows]),
        f"{prefix}_median_delta_w": median(row.delta_w for row in rows),
        f"{prefix}_mean_w_after": mean([row.w_after for row in rows]),
        f"{prefix}_median_w_after": median(row.w_after for row in rows),
    }


def compute_response_correlation_metrics(rows: list[SpecificityRow]) -> dict[str, float]:
    if not rows:
        raise ValidationError("Response-correlation specificity requested for an empty row set.")

    sorted_rows = sorted(rows, key=lambda row: row.response_corr)
    quantile_count = max(1, len(sorted_rows) // 4)
    min_count = min(50, max(5, len(sorted_rows) // 20))
    low_corr = sorted_rows[:quantile_count]
    high_corr = sorted_rows[-quantile_count:]

    metrics = {
        "row_count": float(len(rows)),
        "quantile_fraction": 0.25,
        "min_count": float(min_count),
        "p95_abs_delta_w": percentile([abs(row.delta_w) for row in rows], 95.0),
    }
    metrics.update(summarize_correlation_group(low_corr, "low_corr"))
    metrics.update(summarize_correlation_group(high_corr, "high_corr"))
    return metrics


def compute_strong_synapse_enrichment(rows: list[SpecificityRow]) -> dict[str, float]:
    if not rows:
        raise ValidationError("Strong-synapse enrichment requested for an empty row set.")

    active_rows = [row for row in rows if row.pre_peak_hz > 0.0 and row.post_peak_hz > 0.0]
    min_active_count = min(100, max(20, len(rows) // 100))
    selected_rows = active_rows if len(active_rows) >= min_active_count else rows
    selected_label = "active_endpoints" if selected_rows is active_rows else "all"

    top_count = max(1, int(math.ceil(0.10 * len(selected_rows))))
    min_top_count = min(20, max(5, len(selected_rows) // 100))
    delta_p25 = percentile([row.delta_pref_deg for row in selected_rows], 25.0)
    sorted_by_weight = sorted(selected_rows, key=lambda row: row.w_after, reverse=True)
    top_rows = sorted_by_weight[:top_count]

    def enrichment_for(predicate) -> tuple[int, int, float, float, float]:
        top_qual = sum(predicate(row) for row in top_rows)
        all_qual = sum(predicate(row) for row in selected_rows)
        non_top_count = len(selected_rows) - top_count
        non_top_qual = all_qual - top_qual
        top_other = top_count - top_qual
        non_top_other = non_top_count - non_top_qual
        odds_ratio = ((top_qual + 0.5) / (top_other + 0.5)) / ((non_top_qual + 0.5) / (non_top_other + 0.5))
        return top_qual, all_qual, top_qual / top_count, all_qual / len(selected_rows), odds_ratio

    corr_top, corr_all, corr_top_frac, corr_all_frac, corr_or = enrichment_for(
        lambda row: row.response_corr > 0.20
    )
    combined_top, combined_all, combined_top_frac, combined_all_frac, combined_or = enrichment_for(
        lambda row: row.response_corr > 0.20 and row.delta_pref_deg <= delta_p25
    )

    return {
        "row_count": float(len(selected_rows)),
        "all_row_count": float(len(rows)),
        "active_endpoint_count": float(len(active_rows)),
        "used_active_endpoints": 1.0 if selected_rows is active_rows else 0.0,
        "min_active_count": float(min_active_count),
        "top_count": float(top_count),
        "min_top_count": float(min_top_count),
        "min_qualifying_count": float(min(20, max(5, len(selected_rows) // 1000))),
        "corr_threshold": 0.20,
        "delta_p25": delta_p25,
        "corr_top_qualifying_count": float(corr_top),
        "corr_all_qualifying_count": float(corr_all),
        "corr_top_qualifying_fraction": corr_top_frac,
        "corr_all_qualifying_fraction": corr_all_frac,
        "corr_odds_ratio": corr_or,
        "combined_top_qualifying_count": float(combined_top),
        "combined_all_qualifying_count": float(combined_all),
        "combined_top_qualifying_fraction": combined_top_frac,
        "combined_all_qualifying_fraction": combined_all_frac,
        "combined_odds_ratio": combined_or,
        "top_weight_min": min(row.w_after for row in top_rows),
        "top_weight_max": max(row.w_after for row in top_rows),
        "selected_label": selected_label,
    }


def pearson_correlation(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValidationError("Correlation requires aligned response vectors.")
    if len(first) < 2:
        return 0.0

    first_mean = mean(first)
    second_mean = mean(second)
    numerator = 0.0
    first_var = 0.0
    second_var = 0.0
    for first_value, second_value in zip(first, second):
        first_centered = first_value - first_mean
        second_centered = second_value - second_mean
        numerator += first_centered * second_centered
        first_var += first_centered * first_centered
        second_var += second_centered * second_centered
    if first_var <= 0.0 or second_var <= 0.0:
        return 0.0
    return numerator / math.sqrt(first_var * second_var)


def tuning_vector(row: CellTuningRow, orientations: list[float]) -> list[float]:
    missing = [orientation for orientation in orientations if orientation not in row.rates_by_deg]
    if missing:
        raise ValidationError(f"Cell {row.cell_id} is missing orientation rates: {missing}")
    return [row.rates_by_deg[orientation] for orientation in orientations]


def compute_recurrence_context_metrics(
    specificity_rows: list[SpecificityRow],
    full_tuning: dict[int, CellTuningRow],
    recoff_tuning: dict[int, CellTuningRow],
) -> dict[str, float]:
    if not specificity_rows:
        raise ValidationError("Recurrence context comparison requires specificity rows.")
    if not full_tuning or not recoff_tuning:
        raise ValidationError("Recurrence context comparison requires non-empty tuning maps.")

    common_orientations = sorted(
        set(next(iter(full_tuning.values())).rates_by_deg).intersection(
            next(iter(recoff_tuning.values())).rates_by_deg
        )
    )
    if len(common_orientations) < 2:
        raise ValidationError("Recurrence context comparison requires at least two common orientations.")

    pair_rows: list[dict[str, float]] = []
    for row in specificity_rows:
        if row.pre_id not in full_tuning or row.post_id not in full_tuning:
            continue
        if row.pre_id not in recoff_tuning or row.post_id not in recoff_tuning:
            continue

        pre_on = full_tuning[row.pre_id]
        post_on = full_tuning[row.post_id]
        pre_off = recoff_tuning[row.pre_id]
        post_off = recoff_tuning[row.post_id]
        corr_on = pearson_correlation(
            tuning_vector(pre_on, common_orientations),
            tuning_vector(post_on, common_orientations),
        )
        corr_off = pearson_correlation(
            tuning_vector(pre_off, common_orientations),
            tuning_vector(post_off, common_orientations),
        )
        pair_rows.append(
            {
                "delta_pref_deg": row.delta_pref_deg,
                "full_specificity_corr": row.response_corr,
                "corr_on": corr_on,
                "corr_off": corr_off,
                "peak_on": 0.5 * (pre_on.peak_rate_hz + post_on.peak_rate_hz),
                "peak_off": 0.5 * (pre_off.peak_rate_hz + post_off.peak_rate_hz),
                "osi_on": 0.5 * (pre_on.osi + post_on.osi),
                "osi_off": 0.5 * (pre_off.osi + post_off.osi),
            }
        )

    if not pair_rows:
        raise ValidationError("No recurrent pairs could be mapped to recurrence context tuning rows.")

    active_pairs = [row for row in pair_rows if row["peak_on"] > 0.0]
    active_pairs = active_pairs if active_pairs else pair_rows
    min_count = min(100, max(20, len(active_pairs) // 100))
    delta_p25 = percentile([row["delta_pref_deg"] for row in active_pairs], 25.0)
    focus_pairs = [
        row
        for row in active_pairs
        if row["delta_pref_deg"] <= delta_p25 or row["corr_on"] > 0.20 or row["full_specificity_corr"] > 0.20
    ]

    def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            "count": float(len(rows)),
            "mean_corr_on": mean([row["corr_on"] for row in rows]),
            "mean_corr_off": mean([row["corr_off"] for row in rows]),
            "frac_corr_gt_0p2_on": sum(row["corr_on"] > 0.20 for row in rows) / len(rows),
            "frac_corr_gt_0p2_off": sum(row["corr_off"] > 0.20 for row in rows) / len(rows),
            "mean_peak_on": mean([row["peak_on"] for row in rows]),
            "mean_peak_off": mean([row["peak_off"] for row in rows]),
            "mean_osi_on": mean([row["osi_on"] for row in rows]),
            "mean_osi_off": mean([row["osi_off"] for row in rows]),
        }

    if not focus_pairs:
        focus_pairs = active_pairs
    focus = summarize(focus_pairs)
    active = summarize(active_pairs)

    full_scales = {
        row.recurrent_output_scale
        for row in full_tuning.values()
        if row.recurrent_output_scale is not None
    }
    recoff_scales = {
        row.recurrent_output_scale
        for row in recoff_tuning.values()
        if row.recurrent_output_scale is not None
    }

    metrics = {
        "mapped_pair_count": float(len(pair_rows)),
        "active_pair_count": float(len(active_pairs)),
        "focus_pair_count": focus["count"],
        "min_count": float(min_count),
        "delta_p25": delta_p25,
        "full_recurrent_scale": next(iter(full_scales)) if len(full_scales) == 1 else math.nan,
        "recoff_recurrent_scale": next(iter(recoff_scales)) if len(recoff_scales) == 1 else math.nan,
    }
    for prefix, values in (("focus", focus), ("active", active)):
        metrics[f"{prefix}_mean_corr_on"] = values["mean_corr_on"]
        metrics[f"{prefix}_mean_corr_off"] = values["mean_corr_off"]
        metrics[f"{prefix}_mean_corr_delta"] = values["mean_corr_on"] - values["mean_corr_off"]
        metrics[f"{prefix}_frac_corr_gt_0p2_on"] = values["frac_corr_gt_0p2_on"]
        metrics[f"{prefix}_frac_corr_gt_0p2_off"] = values["frac_corr_gt_0p2_off"]
        metrics[f"{prefix}_frac_corr_gt_0p2_delta"] = (
            values["frac_corr_gt_0p2_on"] - values["frac_corr_gt_0p2_off"]
        )
        metrics[f"{prefix}_mean_peak_on"] = values["mean_peak_on"]
        metrics[f"{prefix}_mean_peak_off"] = values["mean_peak_off"]
        metrics[f"{prefix}_mean_osi_on"] = values["mean_osi_on"]
        metrics[f"{prefix}_mean_osi_off"] = values["mean_osi_off"]
    return metrics


def format_specificity_distance_bins(rows: list[SpecificityRow]) -> str:
    bins = (
        ("d_le_1", [row for row in rows if row.distance_sites <= 1.0]),
        ("d_1_2", [row for row in rows if 1.0 < row.distance_sites <= 2.0]),
        ("d_gt_2", [row for row in rows if row.distance_sites > 2.0]),
    )
    parts: list[str] = []
    for label, bin_rows in bins:
        if not bin_rows:
            parts.append(f"{label}_count=0")
            continue
        parts.append(
            f"{label}_count={len(bin_rows)} "
            f"{label}_mean_delta_w={mean([row.delta_w for row in bin_rows]):.6f} "
            f"{label}_mean_w_after={mean([row.w_after for row in bin_rows]):.6f}"
        )
    return " ".join(parts)


def print_result(passed: bool, label: str, details: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"{status} {label} {details}")
    return passed


def main() -> int:
    args = parse_args()
    try:
        full = load_run(
            args.genn_dir,
            args.full,
            require_size_tuning=True,
            require_specificity=True,
        )
        control = load_run(args.genn_dir, args.control)
        somoff = load_run(args.genn_dir, args.somoff, require_size_tuning=True)

        overall_ok = True

        full_post_osi = require_summary_metric(full, "post_l23_median_osi")
        control_post_osi = require_summary_metric(control, "post_l23_median_osi")
        osi_delta = full_post_osi - control_post_osi
        overall_ok &= print_result(
            full_post_osi >= 0.70 and osi_delta >= 0.10,
            "osi",
            f"full_post={full_post_osi:.6f} control_post={control_post_osi:.6f} delta={osi_delta:.6f}",
        )

        for spec in WEIGHT_SPECS:
            full_before, full_after = full.weights[spec.name]
            control_before, control_after = control.weights[spec.name]
            full_metrics = compare_weight_series(full_before, full_after, spec.lower, spec.upper)
            control_metrics = compare_weight_series(control_before, control_after, spec.lower, spec.upper)

            full_ok = (
                sign_passes(full_metrics, spec.sign)
                and full_metrics.changed_fraction >= 0.05
                and full_metrics.p95_abs_change >= full_metrics.threshold
                and full_metrics.lower_fraction < 0.10
                and full_metrics.upper_fraction < 0.10
            )
            overall_ok &= print_result(
                full_ok,
                f"weights_full[{spec.name}]",
                (
                    f"active={full_metrics.active_count} changed_frac={full_metrics.changed_fraction:.6f} "
                    f"p95={full_metrics.p95_abs_change:.6f} threshold={full_metrics.threshold:.6f} "
                    f"lower_frac={full_metrics.lower_fraction:.6f} upper_frac={full_metrics.upper_fraction:.6f} "
                    f"min_nonzero={full_metrics.min_nonzero:.6f} max_nonzero={full_metrics.max_nonzero:.6f}"
                ),
            )

            control_ok = control_metrics.max_abs_change <= 1.0e-9
            overall_ok &= print_result(
                control_ok,
                f"weights_control[{spec.name}]",
                f"max_abs_change={control_metrics.max_abs_change:.12f}",
            )

        rate_limits = {"l23e": 100.0, "l23pv": 150.0, "l23som": 150.0}
        for run_label, run in (("full", full), ("control", control), ("somoff", somoff)):
            for population, max_p99 in rate_limits.items():
                metrics = compute_rate_metrics(run.post_site_rates[population])
                if population == "l23e":
                    passed = metrics.p99_hz <= max_p99
                else:
                    passed = (
                        metrics.median_hz >= 1.0
                        and metrics.frac_below_1hz < 0.25
                        and metrics.p99_hz <= max_p99
                    )
                overall_ok &= print_result(
                    passed,
                    f"rates[{run_label}:{population}]",
                    (
                        f"median={metrics.median_hz:.6f} frac_lt1={metrics.frac_below_1hz:.6f} "
                        f"p99={metrics.p99_hz:.6f} limit={max_p99:.1f}"
                    ),
                )

        pref_deg, full_center_pref_rate = preferred_center_orientation_deg(full)
        full_context = compute_context_metrics(full, pref_deg)
        somoff_context = compute_context_metrics(somoff, pref_deg)

        overall_ok &= print_result(
            full_center_pref_rate >= 5.0,
            "som_center_pref",
            f"preferred_deg={pref_deg:.1f} center_pref_l23e_hz={full_center_pref_rate:.6f}",
        )
        overall_ok &= print_result(
            full_context["min_center_som_hz"] > 0.0 and full_context["min_broad_som_hz"] > 0.0,
            "som_sanity",
            (
                f"driven_center_threshold_hz={full_context['driven_center_threshold_hz']:.6f} "
                f"relevant_orientations={int(full_context['relevant_orientation_count'])} "
                f"min_center_som_hz={full_context['min_center_som_hz']:.6f} "
                f"min_broad_som_hz={full_context['min_broad_som_hz']:.6f}"
            ),
        )
        overall_ok &= print_result(
            full_context["mean_bsi"] >= 0.20,
            "som_full",
            (
                f"mean_bsi={full_context['mean_bsi']:.6f} "
                f"driven_center_threshold_hz={full_context['driven_center_threshold_hz']:.6f} "
                f"relevant_orientations={int(full_context['relevant_orientation_count'])}"
            ),
        )
        overall_ok &= print_result(
            ((full_context["mean_bsi"] - somoff_context["mean_bsi"]) >= 0.05)
            or (somoff_context["mean_bsi"] <= (0.5 * full_context["mean_bsi"])),
            "som_somoff",
            (
                f"full_mean_bsi={full_context['mean_bsi']:.6f} "
                f"somoff_mean_bsi={somoff_context['mean_bsi']:.6f} "
                f"delta={(full_context['mean_bsi'] - somoff_context['mean_bsi']):.6f}"
            ),
        )

        summary_mean_bsi_details = []
        if full_context["summary_mean_bsi"] is not None:
            summary_mean_bsi_details.append(f"full_summary_mean_bsi={full_context['summary_mean_bsi']:.6f}")
        if somoff_context["summary_mean_bsi"] is not None:
            summary_mean_bsi_details.append(f"somoff_summary_mean_bsi={somoff_context['summary_mean_bsi']:.6f}")
        if summary_mean_bsi_details:
            print(f"INFO som_mean_bsi_summary {' '.join(summary_mean_bsi_details)}")
        print(
            "INFO som_preferred_bsi "
            f"preferred_deg={pref_deg:.1f} "
            f"full_preferred_bsi={full_context['preferred_bsi']:.6f} "
            f"somoff_preferred_bsi={somoff_context['preferred_bsi']:.6f}"
        )
        print(
            "INFO som_driven_threshold "
            f"full_threshold_hz={full_context['driven_center_threshold_hz']:.6f} "
            f"somoff_threshold_hz={somoff_context['driven_center_threshold_hz']:.6f}"
        )

        full_size = compute_size_tuning_metrics(full)
        somoff_size = compute_size_tuning_metrics(
            somoff,
            selected_orientations=full_size["selected_orientations"],
        )
        full_l23e_size = full_size["l23e"]
        full_l4e_size = full_size["l4e"]
        somoff_l23e_size = somoff_size["l23e"]
        radii = full_size["radii"]
        full_l23e_rates = full_size["mean_rates"]["l23e"]
        somoff_l23e_rates = somoff_size["mean_rates"]["l23e"]

        full_l23e_peak_index = int(full_l23e_size["peak_index"])
        overall_ok &= print_result(
            0 < full_l23e_peak_index < (len(radii) - 1),
            "som_size_interior_optimum",
            (
                f"preferred_deg={full_size['preferred_deg']:.1f} "
                f"selected_orientations={format_float_list(full_size['selected_orientations'])} "
                f"radii={format_float_list(radii)} "
                f"peak_radius={full_l23e_size['peak_radius']:.6f} "
                f"l23e_rates={format_float_list(full_l23e_rates)}"
            ),
        )
        overall_ok &= print_result(
            full_l23e_size["peak_rate"] >= 1.0
            and full_l23e_size["early_delta"] >= max(0.5, 0.05 * full_l23e_size["peak_rate"]),
            "som_size_early_summation",
            (
                f"small_rate={full_l23e_size['small_rate']:.6f} "
                f"peak_rate={full_l23e_size['peak_rate']:.6f} "
                f"early_delta={full_l23e_size['early_delta']:.6f}"
            ),
        )
        overall_ok &= print_result(
            full_l23e_size["suppression"] >= 0.10,
            "som_size_large_suppression",
            (
                f"peak_rate={full_l23e_size['peak_rate']:.6f} "
                f"large_rate={full_l23e_size['large_rate']:.6f} "
                f"suppression={full_l23e_size['suppression']:.6f}"
            ),
        )
        overall_ok &= print_result(
            (full_l23e_size["suppression"] - full_l4e_size["suppression"]) >= 0.03,
            "som_size_l4_vs_l23e",
            (
                f"l23e_suppression={full_l23e_size['suppression']:.6f} "
                f"l4e_suppression={full_l4e_size['suppression']:.6f} "
                f"delta={(full_l23e_size['suppression'] - full_l4e_size['suppression']):.6f}"
            ),
        )
        overall_ok &= print_result(
            (full_l23e_size["suppression"] - somoff_l23e_size["suppression"]) >= 0.03,
            "som_size_somoff",
            (
                f"full_suppression={full_l23e_size['suppression']:.6f} "
                f"somoff_suppression={somoff_l23e_size['suppression']:.6f} "
                f"delta={(full_l23e_size['suppression'] - somoff_l23e_size['suppression']):.6f} "
                f"somoff_l23e_rates={format_float_list(somoff_l23e_rates)}"
            ),
        )

        specificity_rows = require_specificity_rows(full)
        specificity = compute_specificity_metrics(specificity_rows)
        delta_w_mean_margin = specificity["low_delta_mean_delta_w"] - specificity["high_delta_mean_delta_w"]
        delta_w_median_margin = specificity["low_delta_median_delta_w"] - specificity["high_delta_median_delta_w"]
        w_after_mean_margin = specificity["low_delta_mean_w_after"] - specificity["high_delta_mean_w_after"]
        w_after_median_margin = specificity["low_delta_median_w_after"] - specificity["high_delta_median_w_after"]
        specificity_margin = max(
            delta_w_mean_margin,
            delta_w_median_margin,
            w_after_mean_margin,
            w_after_median_margin,
        )
        specificity_margin_threshold = max(1.0e-6, 0.01 * specificity["p95_abs_delta_w"])
        overall_ok &= print_result(
            specificity["low_delta_count"] >= specificity["min_count"]
            and specificity["high_delta_count"] >= specificity["min_count"]
            and specificity["p95_abs_delta_w"] >= 1.0e-6
            and specificity_margin >= specificity_margin_threshold,
            "l23ee_specificity",
            (
                f"row_count={int(specificity['row_count'])} "
                f"low_count={int(specificity['low_delta_count'])} "
                f"high_count={int(specificity['high_delta_count'])} "
                f"min_count={int(specificity['min_count'])} "
                f"low_delta_range=[{specificity['low_delta_min_delta_pref_deg']:.6f},"
                f"{specificity['low_delta_max_delta_pref_deg']:.6f}] "
                f"high_delta_range=[{specificity['high_delta_min_delta_pref_deg']:.6f},"
                f"{specificity['high_delta_max_delta_pref_deg']:.6f}] "
                f"low_mean_delta_w={specificity['low_delta_mean_delta_w']:.6f} "
                f"high_mean_delta_w={specificity['high_delta_mean_delta_w']:.6f} "
                f"low_median_delta_w={specificity['low_delta_median_delta_w']:.6f} "
                f"high_median_delta_w={specificity['high_delta_median_delta_w']:.6f} "
                f"low_mean_w_after={specificity['low_delta_mean_w_after']:.6f} "
                f"high_mean_w_after={specificity['high_delta_mean_w_after']:.6f} "
                f"low_median_w_after={specificity['low_delta_median_w_after']:.6f} "
                f"high_median_w_after={specificity['high_delta_median_w_after']:.6f} "
                f"p95_abs_delta_w={specificity['p95_abs_delta_w']:.6f} "
                f"margin_threshold={specificity_margin_threshold:.6f} "
                f"best_margin={specificity_margin:.6f}"
            ),
        )

        corr_specificity = compute_response_correlation_metrics(specificity_rows)
        corr_delta_w_mean_margin = corr_specificity["high_corr_mean_delta_w"] - corr_specificity["low_corr_mean_delta_w"]
        corr_delta_w_median_margin = corr_specificity["high_corr_median_delta_w"] - corr_specificity["low_corr_median_delta_w"]
        corr_w_after_mean_margin = corr_specificity["high_corr_mean_w_after"] - corr_specificity["low_corr_mean_w_after"]
        corr_w_after_median_margin = corr_specificity["high_corr_median_w_after"] - corr_specificity["low_corr_median_w_after"]
        corr_margin = max(
            corr_delta_w_mean_margin,
            corr_delta_w_median_margin,
            corr_w_after_mean_margin,
            corr_w_after_median_margin,
        )
        corr_margin_threshold = max(1.0e-6, 0.01 * corr_specificity["p95_abs_delta_w"])
        overall_ok &= print_result(
            corr_specificity["low_corr_count"] >= corr_specificity["min_count"]
            and corr_specificity["high_corr_count"] >= corr_specificity["min_count"]
            and corr_specificity["p95_abs_delta_w"] >= 1.0e-6
            and corr_margin >= corr_margin_threshold,
            "l23ee_response_corr_specificity",
            (
                f"row_count={int(corr_specificity['row_count'])} "
                f"low_count={int(corr_specificity['low_corr_count'])} "
                f"high_count={int(corr_specificity['high_corr_count'])} "
                f"min_count={int(corr_specificity['min_count'])} "
                f"low_corr_range=[{corr_specificity['low_corr_min_response_corr']:.6f},"
                f"{corr_specificity['low_corr_max_response_corr']:.6f}] "
                f"high_corr_range=[{corr_specificity['high_corr_min_response_corr']:.6f},"
                f"{corr_specificity['high_corr_max_response_corr']:.6f}] "
                f"high_mean_delta_w={corr_specificity['high_corr_mean_delta_w']:.6f} "
                f"low_mean_delta_w={corr_specificity['low_corr_mean_delta_w']:.6f} "
                f"high_median_delta_w={corr_specificity['high_corr_median_delta_w']:.6f} "
                f"low_median_delta_w={corr_specificity['low_corr_median_delta_w']:.6f} "
                f"high_mean_w_after={corr_specificity['high_corr_mean_w_after']:.6f} "
                f"low_mean_w_after={corr_specificity['low_corr_mean_w_after']:.6f} "
                f"high_median_w_after={corr_specificity['high_corr_median_w_after']:.6f} "
                f"low_median_w_after={corr_specificity['low_corr_median_w_after']:.6f} "
                f"p95_abs_delta_w={corr_specificity['p95_abs_delta_w']:.6f} "
                f"margin_threshold={corr_margin_threshold:.6f} "
                f"best_margin={corr_margin:.6f}"
            ),
        )

        enrichment = compute_strong_synapse_enrichment(specificity_rows)
        corr_enriched = (
            enrichment["corr_top_qualifying_count"] >= enrichment["min_qualifying_count"]
            and enrichment["corr_all_qualifying_count"] >= enrichment["min_qualifying_count"]
            and enrichment["corr_odds_ratio"] >= 1.20
        )
        combined_enriched = (
            enrichment["combined_top_qualifying_count"] >= enrichment["min_qualifying_count"]
            and enrichment["combined_all_qualifying_count"] >= enrichment["min_qualifying_count"]
            and enrichment["combined_odds_ratio"] >= 1.20
        )
        overall_ok &= print_result(
            enrichment["top_count"] >= enrichment["min_top_count"]
            and (corr_enriched or combined_enriched),
            "l23ee_strong_synapse_enrichment",
            (
                f"subset={enrichment['selected_label']} "
                f"row_count={int(enrichment['row_count'])} "
                f"all_row_count={int(enrichment['all_row_count'])} "
                f"active_endpoint_count={int(enrichment['active_endpoint_count'])} "
                f"min_active_count={int(enrichment['min_active_count'])} "
                f"top_count={int(enrichment['top_count'])} "
                f"min_top_count={int(enrichment['min_top_count'])} "
                f"min_qualifying_count={int(enrichment['min_qualifying_count'])} "
                f"corr_threshold={enrichment['corr_threshold']:.6f} "
                f"delta_p25={enrichment['delta_p25']:.6f} "
                f"corr_top_qualifying={int(enrichment['corr_top_qualifying_count'])} "
                f"corr_all_qualifying={int(enrichment['corr_all_qualifying_count'])} "
                f"corr_top_fraction={enrichment['corr_top_qualifying_fraction']:.6f} "
                f"corr_all_fraction={enrichment['corr_all_qualifying_fraction']:.6f} "
                f"corr_odds_ratio={enrichment['corr_odds_ratio']:.6f} "
                f"combined_top_qualifying={int(enrichment['combined_top_qualifying_count'])} "
                f"combined_all_qualifying={int(enrichment['combined_all_qualifying_count'])} "
                f"combined_top_fraction={enrichment['combined_top_qualifying_fraction']:.6f} "
                f"combined_all_fraction={enrichment['combined_all_qualifying_fraction']:.6f} "
                f"combined_odds_ratio={enrichment['combined_odds_ratio']:.6f} "
                f"top_weight_range=[{enrichment['top_weight_min']:.6f},"
                f"{enrichment['top_weight_max']:.6f}]"
            ),
        )
        print(
            "INFO l23ee_specificity_true_bins "
            f"same_count={int(specificity['same_count'])} "
            f"orthogonal_count={int(specificity['orthogonal_count'])} "
            f"same_mean_delta_w={specificity['same_mean_delta_w']:.6f} "
            f"orthogonal_mean_delta_w={specificity['orthogonal_mean_delta_w']:.6f} "
            f"same_mean_w_after={specificity['same_mean_w_after']:.6f} "
            f"orthogonal_mean_w_after={specificity['orthogonal_mean_w_after']:.6f}"
        )
        print(f"INFO l23ee_specificity_distance_bins {format_specificity_distance_bins(specificity_rows)}")

        if args.recoff:
            full_recurrence_tuning = parse_cell_tuning_csv(
                require_file(args.genn_dir / f"{args.full}_l23e_recurrence_context_tuning.csv")
            )
            recoff_recurrence_tuning = parse_cell_tuning_csv(
                require_file(args.genn_dir / f"{args.recoff}_l23e_recurrence_context_tuning.csv")
            )
            recurrence = compute_recurrence_context_metrics(
                specificity_rows,
                full_recurrence_tuning,
                recoff_recurrence_tuning,
            )
            overall_ok &= print_result(
                recurrence["focus_pair_count"] >= recurrence["min_count"]
                and (
                    recurrence["focus_mean_corr_delta"] >= 0.01
                    or recurrence["focus_frac_corr_gt_0p2_delta"] >= 0.01
                ),
                "l23ee_recurrence_corr_contribution",
                (
                    f"mapped_pairs={int(recurrence['mapped_pair_count'])} "
                    f"active_pairs={int(recurrence['active_pair_count'])} "
                    f"focus_pairs={int(recurrence['focus_pair_count'])} "
                    f"min_count={int(recurrence['min_count'])} "
                    f"delta_p25={recurrence['delta_p25']:.6f} "
                    f"full_scale={recurrence['full_recurrent_scale']:.6f} "
                    f"recoff_scale={recurrence['recoff_recurrent_scale']:.6f} "
                    f"mean_corr_on={recurrence['focus_mean_corr_on']:.6f} "
                    f"mean_corr_off={recurrence['focus_mean_corr_off']:.6f} "
                    f"mean_corr_delta={recurrence['focus_mean_corr_delta']:.6f} "
                    f"frac_corr_gt_0p2_on={recurrence['focus_frac_corr_gt_0p2_on']:.6f} "
                    f"frac_corr_gt_0p2_off={recurrence['focus_frac_corr_gt_0p2_off']:.6f} "
                    f"frac_corr_gt_0p2_delta={recurrence['focus_frac_corr_gt_0p2_delta']:.6f}"
                ),
            )
            active_peak_ratio = (
                recurrence["active_mean_peak_off"] / recurrence["active_mean_peak_on"]
                if recurrence["active_mean_peak_on"] > 0.0
                else 0.0
            )
            overall_ok &= print_result(
                recurrence["active_pair_count"] >= recurrence["min_count"]
                and recurrence["active_mean_peak_on"] > 0.0
                and recurrence["active_mean_peak_off"] >= max(0.05, 0.05 * recurrence["active_mean_peak_on"])
                and 0.0 <= recurrence["active_mean_osi_on"] <= 1.05
                and 0.0 <= recurrence["active_mean_osi_off"] <= 1.05,
                "l23ee_recurrence_rate_osi_safety",
                (
                    f"active_pairs={int(recurrence['active_pair_count'])} "
                    f"min_count={int(recurrence['min_count'])} "
                    f"mean_peak_on={recurrence['active_mean_peak_on']:.6f} "
                    f"mean_peak_off={recurrence['active_mean_peak_off']:.6f} "
                    f"peak_ratio_off_on={active_peak_ratio:.6f} "
                    f"mean_osi_on={recurrence['active_mean_osi_on']:.6f} "
                    f"mean_osi_off={recurrence['active_mean_osi_off']:.6f}"
                ),
            )
        else:
            print("INFO l23ee_recurrence_context no_recoff_prefix_provided")

        vip_weight_failures = full.vip_weight_files + control.vip_weight_files + somoff.vip_weight_files
        overall_ok &= print_result(
            not vip_weight_failures,
            "vip_weights",
            "none found" if not vip_weight_failures else ",".join(str(path.name) for path in vip_weight_failures),
        )

        for run_label, run in (("full", full), ("control", control), ("somoff", somoff)):
            vip_metrics = {
                key: value
                for key, value in run.summary.items()
                if "vip" in key.lower() and "rate" in key.lower()
            }
            if vip_metrics:
                formatted = " ".join(f"{key}={value:.6f}" for key, value in sorted(vip_metrics.items()))
                print(f"INFO vip_rates[{run_label}] {formatted}")

        return 0 if overall_ok else 1
    except ValidationError as exc:
        print(f"FAIL input {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
