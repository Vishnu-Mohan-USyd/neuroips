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


def load_run(genn_dir: Path, prefix: str) -> RunData:
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

    return RunData(
        prefix=prefix,
        summary=summary,
        context_rows=context_rows,
        post_site_rates=post_site_rates,
        weights=weights,
        vip_weight_files=vip_weight_files,
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


def print_result(passed: bool, label: str, details: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"{status} {label} {details}")
    return passed


def main() -> int:
    args = parse_args()
    try:
        full = load_run(args.genn_dir, args.full)
        control = load_run(args.genn_dir, args.control)
        somoff = load_run(args.genn_dir, args.somoff)

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
