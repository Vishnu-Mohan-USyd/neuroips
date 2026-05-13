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
import random
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
    validation_site_id: int
    som_output_scale: float
    mean_rate_hz: float
    rates_by_deg: dict[float, float]


@dataclass(frozen=True)
class SizeTuningRow:
    """One central-site size tuning response."""

    radius_sites: float
    population: str
    site_id: int
    validation_site_id: int
    som_output_scale: float
    orientation_deg: float
    rate_hz: float


@dataclass(frozen=True)
class OrientationContextRow:
    """One validation-only orientation-context suppression assay row."""

    condition: str
    site_id: int
    validation_site_id: int
    preferred_orientation_deg: float
    stimulus_orientation_deg: float
    orthogonal_orientation_deg: float
    aperture_radius_sites: float
    inner_radius_sites: float
    som_output_scale: float
    l4e_rate_hz: float
    l23e_rate_hz: float
    l23pv_rate_hz: float
    l23som_rate_hz: float
    si_same_l4e: float
    si_orth_l4e: float
    osd_l4e: float
    si_same_l23e: float
    si_orth_l23e: float
    osd_l23e: float
    surround_same_l23e_ratio: float
    surround_orth_l23e_ratio: float


@dataclass(frozen=True)
class BlankBaselineRow:
    """One validation-only blank-baseline site-rate row."""

    repeat_index: int
    population: str
    site_id: int
    rate_hz: float


@dataclass(frozen=True)
class ContrastSweepRow:
    """One validation-only contrast-sweep central-site response row."""

    contrast: float
    site_id: int
    validation_site_id: int
    population: str
    orientation_deg: float
    aperture_radius_sites: float
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
class MultiPhaseCellTuningRow:
    """One L23E cell response vector pooled across phase slots."""

    cell_id: int
    site_id: int
    site_pref_deg: float
    best_orientation_deg: float
    best_phase_deg: float
    phase_count: int
    peak_rate_any_phase_hz: float
    mean_rate_hz: float
    phase_pooled_osi: float
    phase_mean_rates_by_deg: dict[float, float]


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


@dataclass(frozen=True)
class OsiSiteMetrics:
    """L23E site OSI summaries split by response threshold."""

    total_count: int
    osi_count: int
    active_count: int
    responsive_count: int
    active_fraction: float
    responsive_fraction: float
    all_median_osi: float | None
    active_median_osi: float | None
    responsive_median_osi: float | None
    responsive_threshold_hz: float


@dataclass(frozen=True)
class CellResponsiveMetrics:
    """L23E cell peak-response coverage from held-out tuning curves."""

    total_cells: int
    active_cells: int
    responsive_cells: int
    active_fraction: float
    responsive_fraction: float
    active_median_osi: float | None
    responsive_median_osi: float | None
    total_sites: int
    active_sites: int
    responsive_sites: int
    active_site_fraction: float
    responsive_site_fraction: float
    threshold_hz: float


@dataclass(frozen=True)
class MultiPhaseCellResponsiveMetrics:
    """L23E cell peak-response coverage across held-out phases."""

    total_cells: int
    active_cells: int
    responsive_cells: int
    active_fraction: float
    responsive_fraction: float
    responsive_median_phase_pooled_osi: float | None
    total_sites: int
    active_sites_ge1: int
    responsive_sites_ge1: int
    responsive_sites_ge2: int
    active_site_fraction_ge1: float
    responsive_site_fraction_ge1: float
    responsive_site_fraction_ge2: float
    threshold_hz: float


@dataclass(frozen=True)
class PostSiteMetric:
    """One post-sweep site row with optional spatial and tuning diagnostics."""

    site_id: int
    x: float | None
    y: float | None
    map_pref_deg: float | None
    measured_pref_deg: float | None
    mean_rate_hz: float
    osi: float | None


@dataclass
class RunData:
    """All parsed artifacts for one experiment prefix."""

    prefix: str
    summary: dict[str, float]
    context_rows: dict[tuple[str, str], ContextRow]
    context_rows_by_site: dict[int, dict[tuple[str, str], ContextRow]]
    post_site_rates: dict[str, list[float]]
    l4_post_sites: list[PostSiteMetric] | None
    l23e_post_sites: list[PostSiteMetric]
    weights: dict[str, tuple[WeightSeries, WeightSeries]]
    vip_weight_files: list[Path]
    size_tuning_rows: list[SizeTuningRow] | None = None
    orientation_context_rows: list[OrientationContextRow] | None = None
    blank_baseline_rows: list[BlankBaselineRow] | None = None
    contrast_sweep_rows: list[ContrastSweepRow] | None = None
    specificity_rows: list[SpecificityRow] | None = None
    l23e_cell_tuning: dict[int, CellTuningRow] | None = None
    l23e_cell_tuning_multiphase: dict[int, MultiPhaseCellTuningRow] | None = None


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
    parser.add_argument(
        "--pvweak",
        help="Optional prefix for PV-output weakening run, e.g. v1_fp_pvweak.",
    )
    parser.add_argument(
        "--min-validation-sites",
        type=int,
        default=1,
        help="Minimum required retinotopic validation sites in full and somoff context/size artifacts.",
    )
    parser.add_argument(
        "--require-l4-intersite",
        action="store_true",
        help="Require opt-in L4 inter-site diagnostics and enforce local/bounded spread gates.",
    )
    parser.add_argument(
        "--require-emergent-l23-orientation-suppression",
        action="store_true",
        help=(
            "Require strict no-hardcoded-orientation metadata and the opt-in "
            "L2/3 orientation-context suppression assay gates."
        ),
    )
    parser.add_argument(
        "--require-l23ee-recurrent-biology",
        action="store_true",
        help=(
            "Require opt-in recurrent L23E->L23E biology diagnostics "
            "from the specificity CSV."
        ),
    )
    parser.add_argument(
        "--require-pv-gain-normalization",
        action="store_true",
        help="Require full-vs-pvweak PV gain-normalization causality diagnostics.",
    )
    parser.add_argument(
        "--require-som-size-surround",
        action="store_true",
        help="Require additive SOM size/surround validation diagnostics.",
    )
    parser.add_argument(
        "--require-responsiveness-sparsity",
        action="store_true",
        help="Require additive L23E cell responsiveness/sparsity validation diagnostics.",
    )
    parser.add_argument(
        "--require-scaling-map-consistency",
        action="store_true",
        help="Require additive sheet-scaling map consistency and spatial coverage diagnostics.",
    )
    parser.add_argument(
        "--require-sensory-baseline-contrast-annulus",
        action="store_true",
        help="Require additive sensory blank, contrast, and annular surround validation diagnostics.",
    )
    parser.add_argument(
        "--allow-responsive-osi",
        action="store_true",
        help=(
            "Allow responsive-site L23E OSI to rescue a failed all-site OSI gate "
            "only when all other validator gates pass."
        ),
    )
    parser.add_argument(
        "--responsive-rate-threshold-hz",
        type=float,
        default=1.0,
        help="Mean-rate threshold for responsive-site L23E OSI reporting and optional rescue.",
    )
    parser.add_argument(
        "--cell-responsive-threshold-hz",
        type=float,
        default=1.0,
        help="Peak-rate threshold for reporting responsive L23E cell coverage.",
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


def parse_post_site_metrics_csv(path: Path) -> list[PostSiteMetric]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "mean_rate_hz" not in reader.fieldnames:
            raise ValidationError(f"Missing mean_rate_hz column in {path}")

        rows: list[PostSiteMetric] = []
        for row_number, row in enumerate(reader, start=2):
            for column in reader.fieldnames:
                raw = row.get(column)
                if raw is None:
                    raise ValidationError(f"Missing value in {path} row {row_number} column {column}")
                if column == "site_id":
                    parse_int(raw, path, row_number, column)
                else:
                    parse_float(raw, path, row_number, column)

            site_id = (
                parse_int(row["site_id"], path, row_number, "site_id")
                if "site_id" in reader.fieldnames
                else row_number - 2
            )
            rows.append(
                PostSiteMetric(
                    site_id=site_id,
                    x=parse_float(row["x"], path, row_number, "x") if "x" in reader.fieldnames else None,
                    y=parse_float(row["y"], path, row_number, "y") if "y" in reader.fieldnames else None,
                    map_pref_deg=(
                        parse_float(row["map_pref_deg"], path, row_number, "map_pref_deg")
                        if "map_pref_deg" in reader.fieldnames
                        else None
                    ),
                    measured_pref_deg=(
                        parse_float(row["measured_pref_deg"], path, row_number, "measured_pref_deg")
                        if "measured_pref_deg" in reader.fieldnames
                        else None
                    ),
                    mean_rate_hz=parse_float(row["mean_rate_hz"], path, row_number, "mean_rate_hz"),
                    osi=parse_float(row["osi"], path, row_number, "osi") if "osi" in reader.fieldnames else None,
                )
            )
    if not rows:
        raise ValidationError(f"Post site file is empty: {path}")
    return rows


def parse_rate_column_name(column: str, path: Path) -> float:
    prefix = "rate_"
    suffix = "deg_hz"
    if not column.startswith(prefix) or not column.endswith(suffix):
        raise ValidationError(f"Unexpected rate column in {path}: {column}")
    return float(column[len(prefix) : -len(suffix)])


def parse_context_csv(
    path: Path,
) -> tuple[dict[tuple[str, str], ContextRow], dict[int, dict[tuple[str, str], ContextRow]]]:
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

        rows_by_site: dict[int, dict[tuple[str, str], ContextRow]] = {}
        site_order: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            condition = (row.get("condition") or "").strip()
            population = (row.get("population") or "").strip()
            if not condition or not population:
                raise ValidationError(f"Missing condition/population in {path} row {row_number}")

            site_id = parse_int(row["site_id"], path, row_number, "site_id")
            validation_site_id = (
                parse_int(row["validation_site_id"], path, row_number, "validation_site_id")
                if "validation_site_id" in reader.fieldnames
                else site_id
            )
            som_output_scale = parse_float(row["som_output_scale"], path, row_number, "som_output_scale")
            mean_rate_hz = parse_float(row["mean_rate_hz"], path, row_number, "mean_rate_hz")

            rates_by_deg: dict[float, float] = {}
            for column in rate_columns:
                rates_by_deg[parse_rate_column_name(column, path)] = parse_float(
                    row[column], path, row_number, column
                )

            key = (condition, population)
            if validation_site_id not in rows_by_site:
                rows_by_site[validation_site_id] = {}
                site_order.append(validation_site_id)
            if key in rows_by_site[validation_site_id]:
                raise ValidationError(
                    f"Duplicate context row for site={validation_site_id} "
                    f"condition={condition} population={population} in {path}"
                )
            rows_by_site[validation_site_id][key] = ContextRow(
                condition=condition,
                population=population,
                site_id=site_id,
                validation_site_id=validation_site_id,
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
    if not rows_by_site:
        raise ValidationError(f"Context file is empty: {path}")
    for validation_site_id, rows in rows_by_site.items():
        missing_keys = expected_keys.difference(rows)
        if missing_keys:
            raise ValidationError(
                f"Missing context rows for validation_site_id={validation_site_id} "
                f"in {path}: {sorted(missing_keys)}"
            )
    return rows_by_site[site_order[0]], rows_by_site


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
            site_id = parse_int(row["site_id"], path, row_number, "site_id")
            validation_site_id = (
                parse_int(row["validation_site_id"], path, row_number, "validation_site_id")
                if "validation_site_id" in reader.fieldnames
                else site_id
            )
            rows.append(
                SizeTuningRow(
                    radius_sites=parse_float(row["radius_sites"], path, row_number, "radius_sites"),
                    population=population,
                    site_id=site_id,
                    validation_site_id=validation_site_id,
                    som_output_scale=parse_float(row["som_output_scale"], path, row_number, "som_output_scale"),
                    orientation_deg=parse_float(row["orientation_deg"], path, row_number, "orientation_deg"),
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                )
            )

    if not rows:
        raise ValidationError(f"Size tuning file is empty: {path}")
    return rows


def parse_orientation_context_csv(path: Path) -> list[OrientationContextRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "condition",
            "site_id",
            "preferred_orientation_deg",
            "stimulus_orientation_deg",
            "orthogonal_orientation_deg",
            "aperture_radius_sites",
            "inner_radius_sites",
            "som_output_scale",
            "l4e_rate_hz",
            "l23e_rate_hz",
            "l23pv_rate_hz",
            "l23som_rate_hz",
            "si_same_l4e",
            "si_orth_l4e",
            "osd_l4e",
            "si_same_l23e",
            "si_orth_l23e",
            "osd_l23e",
            "surround_same_l23e_ratio",
            "surround_orth_l23e_ratio",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing orientation-context columns in {path}: {sorted(missing)}")

        rows: list[OrientationContextRow] = []
        for row_number, row in enumerate(reader, start=2):
            condition = (row.get("condition") or "").strip()
            if not condition:
                raise ValidationError(f"Missing condition in {path} row {row_number}")
            site_id = parse_int(row["site_id"], path, row_number, "site_id")
            validation_site_id = (
                parse_int(row["validation_site_id"], path, row_number, "validation_site_id")
                if "validation_site_id" in reader.fieldnames
                else site_id
            )
            rows.append(
                OrientationContextRow(
                    condition=condition,
                    site_id=site_id,
                    validation_site_id=validation_site_id,
                    preferred_orientation_deg=parse_float(
                        row["preferred_orientation_deg"], path, row_number, "preferred_orientation_deg"
                    ),
                    stimulus_orientation_deg=parse_float(
                        row["stimulus_orientation_deg"], path, row_number, "stimulus_orientation_deg"
                    ),
                    orthogonal_orientation_deg=parse_float(
                        row["orthogonal_orientation_deg"], path, row_number, "orthogonal_orientation_deg"
                    ),
                    aperture_radius_sites=parse_float(
                        row["aperture_radius_sites"], path, row_number, "aperture_radius_sites"
                    ),
                    inner_radius_sites=parse_float(row["inner_radius_sites"], path, row_number, "inner_radius_sites"),
                    som_output_scale=parse_float(row["som_output_scale"], path, row_number, "som_output_scale"),
                    l4e_rate_hz=parse_float(row["l4e_rate_hz"], path, row_number, "l4e_rate_hz"),
                    l23e_rate_hz=parse_float(row["l23e_rate_hz"], path, row_number, "l23e_rate_hz"),
                    l23pv_rate_hz=parse_float(row["l23pv_rate_hz"], path, row_number, "l23pv_rate_hz"),
                    l23som_rate_hz=parse_float(row["l23som_rate_hz"], path, row_number, "l23som_rate_hz"),
                    si_same_l4e=parse_float(row["si_same_l4e"], path, row_number, "si_same_l4e"),
                    si_orth_l4e=parse_float(row["si_orth_l4e"], path, row_number, "si_orth_l4e"),
                    osd_l4e=parse_float(row["osd_l4e"], path, row_number, "osd_l4e"),
                    si_same_l23e=parse_float(row["si_same_l23e"], path, row_number, "si_same_l23e"),
                    si_orth_l23e=parse_float(row["si_orth_l23e"], path, row_number, "si_orth_l23e"),
                    osd_l23e=parse_float(row["osd_l23e"], path, row_number, "osd_l23e"),
                    surround_same_l23e_ratio=parse_float(
                        row["surround_same_l23e_ratio"], path, row_number, "surround_same_l23e_ratio"
                    ),
                    surround_orth_l23e_ratio=parse_float(
                        row["surround_orth_l23e_ratio"], path, row_number, "surround_orth_l23e_ratio"
                    ),
                )
            )

    if not rows:
        raise ValidationError(f"Orientation-context file is empty: {path}")
    return rows


def parse_blank_baseline_csv(path: Path) -> list[BlankBaselineRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {"repeat_index", "population", "site_id", "rate_hz"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing blank-baseline columns in {path}: {sorted(missing)}")

        rows: list[BlankBaselineRow] = []
        for row_number, row in enumerate(reader, start=2):
            population = (row.get("population") or "").strip()
            if not population:
                raise ValidationError(f"Missing population in {path} row {row_number}")
            rows.append(
                BlankBaselineRow(
                    repeat_index=parse_int(row["repeat_index"], path, row_number, "repeat_index"),
                    population=population,
                    site_id=parse_int(row["site_id"], path, row_number, "site_id"),
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                )
            )

    if not rows:
        raise ValidationError(f"Blank-baseline file is empty: {path}")
    return rows


def parse_contrast_sweep_csv(path: Path) -> list[ContrastSweepRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "contrast",
            "site_id",
            "validation_site_id",
            "population",
            "orientation_deg",
            "aperture_radius_sites",
            "rate_hz",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing contrast-sweep columns in {path}: {sorted(missing)}")

        rows: list[ContrastSweepRow] = []
        for row_number, row in enumerate(reader, start=2):
            population = (row.get("population") or "").strip()
            if not population:
                raise ValidationError(f"Missing population in {path} row {row_number}")
            rows.append(
                ContrastSweepRow(
                    contrast=parse_float(row["contrast"], path, row_number, "contrast"),
                    site_id=parse_int(row["site_id"], path, row_number, "site_id"),
                    validation_site_id=parse_int(row["validation_site_id"], path, row_number, "validation_site_id"),
                    population=population,
                    orientation_deg=parse_float(row["orientation_deg"], path, row_number, "orientation_deg"),
                    aperture_radius_sites=parse_float(row["aperture_radius_sites"], path, row_number, "aperture_radius_sites"),
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                )
            )

    if not rows:
        raise ValidationError(f"Contrast-sweep file is empty: {path}")
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


def parse_multiphase_cell_tuning_csv(path: Path) -> dict[int, MultiPhaseCellTuningRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "cell_id",
            "site_id",
            "site_pref_deg",
            "best_orientation_deg",
            "best_phase_deg",
            "phase_count",
            "peak_rate_any_phase_hz",
            "mean_rate_hz",
            "phase_pooled_osi",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing multiphase cell tuning columns in {path}: {sorted(missing)}")

        rate_columns = [column for column in reader.fieldnames if column.startswith("rate_")]
        if not rate_columns:
            raise ValidationError(f"No phase-mean orientation rate columns found in {path}")

        rows: dict[int, MultiPhaseCellTuningRow] = {}
        for row_number, row in enumerate(reader, start=2):
            cell_id = parse_int(row["cell_id"], path, row_number, "cell_id")
            if cell_id in rows:
                raise ValidationError(f"Duplicate cell_id {cell_id} in {path}")

            phase_mean_rates_by_deg = {
                parse_rate_column_name(column, path): parse_float(row[column], path, row_number, column)
                for column in rate_columns
            }
            rows[cell_id] = MultiPhaseCellTuningRow(
                cell_id=cell_id,
                site_id=parse_int(row["site_id"], path, row_number, "site_id"),
                site_pref_deg=parse_float(row["site_pref_deg"], path, row_number, "site_pref_deg"),
                best_orientation_deg=parse_float(row["best_orientation_deg"], path, row_number, "best_orientation_deg"),
                best_phase_deg=parse_float(row["best_phase_deg"], path, row_number, "best_phase_deg"),
                phase_count=parse_int(row["phase_count"], path, row_number, "phase_count"),
                peak_rate_any_phase_hz=parse_float(
                    row["peak_rate_any_phase_hz"],
                    path,
                    row_number,
                    "peak_rate_any_phase_hz",
                ),
                mean_rate_hz=parse_float(row["mean_rate_hz"], path, row_number, "mean_rate_hz"),
                phase_pooled_osi=parse_float(row["phase_pooled_osi"], path, row_number, "phase_pooled_osi"),
                phase_mean_rates_by_deg=phase_mean_rates_by_deg,
            )

    if not rows:
        raise ValidationError(f"Multiphase cell tuning file is empty: {path}")
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
    context_rows, context_rows_by_site = parse_context_csv(
        require_file(genn_dir / f"{prefix}_som_context_validation.csv")
    )

    post_site_rates = {
        population: parse_site_rates_csv(require_file(genn_dir / f"{prefix}{suffix}"))
        for population, suffix in POST_SITE_SUFFIXES.items()
    }
    l23e_post_sites = parse_post_site_metrics_csv(
        require_file(genn_dir / f"{prefix}_post_l23_sites.csv")
    )
    l4_post_site_path = genn_dir / f"{prefix}_post_l4_sites.csv"
    l4_post_sites = parse_post_site_metrics_csv(l4_post_site_path) if l4_post_site_path.is_file() else None

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
    orientation_context_path = genn_dir / f"{prefix}_l23_orientation_context_suppression.csv"
    orientation_context_rows = (
        parse_orientation_context_csv(orientation_context_path)
        if orientation_context_path.is_file()
        else None
    )
    blank_baseline_path = genn_dir / f"{prefix}_blank_baseline.csv"
    blank_baseline_rows = (
        parse_blank_baseline_csv(blank_baseline_path)
        if blank_baseline_path.is_file()
        else None
    )
    contrast_sweep_path = genn_dir / f"{prefix}_contrast_sweep.csv"
    contrast_sweep_rows = (
        parse_contrast_sweep_csv(contrast_sweep_path)
        if contrast_sweep_path.is_file()
        else None
    )
    specificity_rows = (
        parse_specificity_csv(require_file(genn_dir / f"{prefix}_l23ee_specificity.csv"))
        if require_specificity
        else None
    )
    cell_tuning_path = genn_dir / f"{prefix}_l23e_cell_tuning.csv"
    l23e_cell_tuning = parse_cell_tuning_csv(cell_tuning_path) if cell_tuning_path.is_file() else None
    multiphase_cell_tuning_path = genn_dir / f"{prefix}_l23e_cell_tuning_multiphase.csv"
    l23e_cell_tuning_multiphase = (
        parse_multiphase_cell_tuning_csv(multiphase_cell_tuning_path)
        if multiphase_cell_tuning_path.is_file()
        else None
    )

    return RunData(
        prefix=prefix,
        summary=summary,
        context_rows=context_rows,
        context_rows_by_site=context_rows_by_site,
        post_site_rates=post_site_rates,
        l4_post_sites=l4_post_sites,
        l23e_post_sites=l23e_post_sites,
        weights=weights,
        vip_weight_files=vip_weight_files,
        size_tuning_rows=size_tuning_rows,
        orientation_context_rows=orientation_context_rows,
        blank_baseline_rows=blank_baseline_rows,
        contrast_sweep_rows=contrast_sweep_rows,
        specificity_rows=specificity_rows,
        l23e_cell_tuning=l23e_cell_tuning,
        l23e_cell_tuning_multiphase=l23e_cell_tuning_multiphase,
    )


def require_summary_metric(run: RunData, metric: str) -> float:
    if metric not in run.summary:
        raise ValidationError(f"Missing summary metric {metric!r} in prefix {run.prefix}")
    value = run.summary[metric]
    if not math.isfinite(value):
        raise ValidationError(f"Non-finite summary metric {metric!r} in prefix {run.prefix}")
    return value


def require_metric(metrics: dict[str, float], metric: str, source: str) -> float:
    if metric not in metrics:
        raise ValidationError(f"Missing metric {metric!r} in {source}")
    value = metrics[metric]
    if not math.isfinite(value):
        raise ValidationError(f"Non-finite metric {metric!r} in {source}")
    return value


def optional_metric(metrics: dict[str, float], metric: str, fallback: float) -> float:
    value = metrics.get(metric, fallback)
    if not math.isfinite(value):
        raise ValidationError(f"Non-finite metric {metric!r}")
    return value


def compute_rate_metrics(rates: list[float]) -> RateMetrics:
    if not rates:
        raise ValidationError("Rate sanity requested for an empty rate vector.")
    return RateMetrics(
        median_hz=median(rates),
        frac_below_1hz=sum(rate < 1.0 for rate in rates) / len(rates),
        p99_hz=percentile(rates, 99.0),
    )


def optional_median(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return median(values_list)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def positive_modulo_degrees(degrees: float) -> float:
    wrapped = degrees % 180.0
    return wrapped + 180.0 if wrapped < 0.0 else wrapped


def orientation_tuning_metrics(rates_by_deg: dict[float, float]) -> tuple[float, float, float]:
    if not rates_by_deg:
        raise ValidationError("Orientation tuning metrics require non-empty rates.")
    total = sum(max(0.0, rate) for rate in rates_by_deg.values())
    if total <= 0.0:
        return 0.0, 0.0, max(rates_by_deg, key=lambda degree: rates_by_deg[degree])

    vector_x = 0.0
    vector_y = 0.0
    for degree, rate in rates_by_deg.items():
        theta = math.radians(2.0 * degree)
        nonnegative_rate = max(0.0, rate)
        vector_x += nonnegative_rate * math.cos(theta)
        vector_y += nonnegative_rate * math.sin(theta)
    osi = min(1.0, math.hypot(vector_x, vector_y) / total)
    preferred_deg = positive_modulo_degrees(0.5 * math.degrees(math.atan2(vector_y, vector_x)))
    return osi, max(rates_by_deg.values()), preferred_deg


def compute_pv_gain_normalization_metrics(full: RunData, pvweak: RunData) -> dict[str, float]:
    pvweak_scale = require_summary_metric(pvweak, "l23pv_context_output_scale")
    pvweak_active = require_summary_metric(pvweak, "l23pv_context_output_ablation_active")

    paired_full_rates: list[float] = []
    paired_pvweak_rates: list[float] = []
    full_osi_by_site: list[float] = []
    pvweak_osi_by_site: list[float] = []
    preferred_shifts: list[float] = []

    common_site_ids = sorted(set(full.context_rows_by_site).intersection(pvweak.context_rows_by_site))
    if not common_site_ids:
        raise ValidationError("PV gain-normalization requires common context validation sites.")

    for validation_site_id in common_site_ids:
        full_rows = full.context_rows_by_site[validation_site_id]
        pvweak_rows = pvweak.context_rows_by_site[validation_site_id]
        full_l23e = full_rows.get(("center_only", "l23e"))
        pvweak_l23e = pvweak_rows.get(("center_only", "l23e"))
        if full_l23e is None or pvweak_l23e is None:
            raise ValidationError(
                f"PV gain-normalization missing center_only/l23e rows for site {validation_site_id}"
            )

        common_orientations = sorted(set(full_l23e.rates_by_deg).intersection(pvweak_l23e.rates_by_deg))
        if not common_orientations:
            raise ValidationError(
                f"PV gain-normalization found no common orientations for site {validation_site_id}"
            )
        peak_full = max(full_l23e.rates_by_deg[orientation] for orientation in common_orientations)
        driven_threshold = max(1.0, 0.25 * peak_full)
        selected_orientations = [
            orientation
            for orientation in common_orientations
            if full_l23e.rates_by_deg[orientation] >= driven_threshold
        ]
        if not selected_orientations:
            selected_orientations = [
                max(common_orientations, key=lambda orientation: full_l23e.rates_by_deg[orientation])
            ]

        for orientation in selected_orientations:
            paired_full_rates.append(full_l23e.rates_by_deg[orientation])
            paired_pvweak_rates.append(pvweak_l23e.rates_by_deg[orientation])

        pvweak_peak = max(pvweak_l23e.rates_by_deg[orientation] for orientation in common_orientations)
        if peak_full >= 1.0 or pvweak_peak >= 1.0:
            full_common_rates = {orientation: full_l23e.rates_by_deg[orientation] for orientation in common_orientations}
            pvweak_common_rates = {orientation: pvweak_l23e.rates_by_deg[orientation] for orientation in common_orientations}
            full_osi, _, full_pref = orientation_tuning_metrics(full_common_rates)
            pvweak_osi, _, pvweak_pref = orientation_tuning_metrics(pvweak_common_rates)
            full_osi_by_site.append(full_osi)
            pvweak_osi_by_site.append(pvweak_osi)
            preferred_shifts.append(circular_orientation_delta_deg(full_pref, pvweak_pref))

    if not paired_full_rates or not paired_pvweak_rates:
        raise ValidationError("PV gain-normalization found no driven L23E context rates.")
    if not full_osi_by_site or not pvweak_osi_by_site:
        raise ValidationError("PV gain-normalization found no responsive sites for OSI safety.")

    full_mean = mean(paired_full_rates)
    pvweak_mean = mean(paired_pvweak_rates)
    full_median = median(paired_full_rates)
    pvweak_median = median(paired_pvweak_rates)
    mean_increase = (pvweak_mean / full_mean) - 1.0 if full_mean > 0.0 else math.inf
    median_increase = (pvweak_median / full_median) - 1.0 if full_median > 0.0 else math.inf
    full_median_osi = median(full_osi_by_site)
    pvweak_median_osi = median(pvweak_osi_by_site)
    full_pv_rate_metrics = compute_rate_metrics(full.post_site_rates["l23pv"])

    return {
        "pvweak_scale": pvweak_scale,
        "pvweak_active": pvweak_active,
        "site_count": float(len(common_site_ids)),
        "driven_rate_count": float(len(paired_full_rates)),
        "full_mean_l23e_hz": full_mean,
        "pvweak_mean_l23e_hz": pvweak_mean,
        "mean_increase_fraction": mean_increase,
        "full_median_l23e_hz": full_median,
        "pvweak_median_l23e_hz": pvweak_median,
        "median_increase_fraction": median_increase,
        "pvweak_l23e_context_p99_hz": percentile(paired_pvweak_rates, 99.0),
        "l23e_p99_limit_hz": 100.0,
        "responsive_site_count": float(len(full_osi_by_site)),
        "full_median_osi": full_median_osi,
        "pvweak_median_osi": pvweak_median_osi,
        "median_osi_drop": full_median_osi - pvweak_median_osi,
        "median_pref_shift_deg": median(preferred_shifts),
        "max_pref_shift_deg": max(preferred_shifts),
        "full_l23pv_post_median_hz": full_pv_rate_metrics.median_hz,
        "full_l23pv_post_frac_lt1": full_pv_rate_metrics.frac_below_1hz,
        "full_l23pv_post_p99_hz": full_pv_rate_metrics.p99_hz,
        "full_l23pv_post_p99_limit_hz": 150.0,
    }


def compute_l23e_osi_site_metrics(
    rows: list[PostSiteMetric],
    responsive_threshold_hz: float,
) -> OsiSiteMetrics:
    total_count = len(rows)
    osi_rows = [row for row in rows if row.osi is not None]
    active_rows = [row for row in osi_rows if row.mean_rate_hz > 0.0]
    responsive_rows = [row for row in osi_rows if row.mean_rate_hz >= responsive_threshold_hz]
    denominator = total_count if total_count > 0 else 1
    return OsiSiteMetrics(
        total_count=total_count,
        osi_count=len(osi_rows),
        active_count=len(active_rows),
        responsive_count=len(responsive_rows),
        active_fraction=len(active_rows) / denominator,
        responsive_fraction=len(responsive_rows) / denominator,
        all_median_osi=optional_median(row.osi for row in osi_rows if row.osi is not None),
        active_median_osi=optional_median(row.osi for row in active_rows if row.osi is not None),
        responsive_median_osi=optional_median(
            row.osi for row in responsive_rows if row.osi is not None
        ),
        responsive_threshold_hz=responsive_threshold_hz,
    )


def print_l23e_osi_site_info(run_label: str, metrics: OsiSiteMetrics) -> None:
    print(
        f"INFO l23e_osi_sites[{run_label}] "
        f"total_count={metrics.total_count} "
        f"osi_count={metrics.osi_count} "
        f"active_count={metrics.active_count} "
        f"active_fraction={metrics.active_fraction:.6f} "
        f"responsive_threshold_hz={metrics.responsive_threshold_hz:.6f} "
        f"responsive_count={metrics.responsive_count} "
        f"responsive_fraction={metrics.responsive_fraction:.6f} "
        f"all_median_osi={format_optional_float(metrics.all_median_osi)} "
        f"active_median_osi={format_optional_float(metrics.active_median_osi)} "
        f"responsive_median_osi={format_optional_float(metrics.responsive_median_osi)}"
    )


def format_l23e_osi_quadrants(
    rows: list[PostSiteMetric],
    responsive_threshold_hz: float,
) -> str:
    usable_rows = [row for row in rows if row.x is not None and row.y is not None and row.osi is not None]
    if not usable_rows:
        return "unavailable=1 reason=missing_x_y_or_osi"

    min_x = min(row.x for row in usable_rows if row.x is not None)
    max_x = max(row.x for row in usable_rows if row.x is not None)
    min_y = min(row.y for row in usable_rows if row.y is not None)
    max_y = max(row.y for row in usable_rows if row.y is not None)
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)
    quadrants: dict[str, list[PostSiteMetric]] = {
        "left_lower": [],
        "left_upper": [],
        "right_lower": [],
        "right_upper": [],
    }
    for row in usable_rows:
        horizontal = "left" if row.x is not None and row.x <= mid_x else "right"
        vertical = "lower" if row.y is not None and row.y <= mid_y else "upper"
        quadrants[f"{horizontal}_{vertical}"].append(row)

    parts = [f"x_mid={mid_x:.6f}", f"y_mid={mid_y:.6f}"]
    for label in sorted(quadrants):
        quadrant_rows = quadrants[label]
        active_rows = [row for row in quadrant_rows if row.mean_rate_hz > 0.0]
        responsive_rows = [
            row for row in quadrant_rows if row.mean_rate_hz >= responsive_threshold_hz
        ]
        parts.extend(
            [
                f"{label}_count={len(quadrant_rows)}",
                f"{label}_active={len(active_rows)}",
                f"{label}_responsive={len(responsive_rows)}",
                f"{label}_median_osi={format_optional_float(optional_median(row.osi for row in quadrant_rows if row.osi is not None))}",
                f"{label}_responsive_median_osi={format_optional_float(optional_median(row.osi for row in responsive_rows if row.osi is not None))}",
            ]
        )
    return " ".join(parts)


def compute_cell_responsive_metrics(
    rows_by_cell: dict[int, CellTuningRow],
    threshold_hz: float,
) -> CellResponsiveMetrics:
    rows = list(rows_by_cell.values())
    total_cells = len(rows)
    denominator = total_cells if total_cells > 0 else 1
    active_rows = [row for row in rows if row.peak_rate_hz > 0.0]
    responsive_rows = [row for row in rows if row.peak_rate_hz >= threshold_hz]
    all_sites = {row.site_id for row in rows}
    active_sites = {row.site_id for row in active_rows}
    responsive_sites = {row.site_id for row in responsive_rows}
    site_denominator = len(all_sites) if all_sites else 1
    return CellResponsiveMetrics(
        total_cells=total_cells,
        active_cells=len(active_rows),
        responsive_cells=len(responsive_rows),
        active_fraction=len(active_rows) / denominator,
        responsive_fraction=len(responsive_rows) / denominator,
        active_median_osi=optional_median(row.osi for row in active_rows),
        responsive_median_osi=optional_median(row.osi for row in responsive_rows),
        total_sites=len(all_sites),
        active_sites=len(active_sites),
        responsive_sites=len(responsive_sites),
        active_site_fraction=len(active_sites) / site_denominator,
        responsive_site_fraction=len(responsive_sites) / site_denominator,
        threshold_hz=threshold_hz,
    )


def print_l23e_cell_coverage_info(
    run_label: str,
    cell_tuning: dict[int, CellTuningRow] | None,
    threshold_hz: float,
) -> None:
    if cell_tuning is None:
        print(f"INFO l23e_cell_responsive_coverage[{run_label}] available=0")
        return
    metrics = compute_cell_responsive_metrics(cell_tuning, threshold_hz)
    print(
        f"INFO l23e_cell_responsive_coverage[{run_label}] "
        f"available=1 "
        f"threshold_hz={metrics.threshold_hz:.6f} "
        f"total_cells={metrics.total_cells} "
        f"active_cells={metrics.active_cells} "
        f"active_fraction={metrics.active_fraction:.6f} "
        f"responsive_cells={metrics.responsive_cells} "
        f"responsive_fraction={metrics.responsive_fraction:.6f} "
        f"active_median_osi={format_optional_float(metrics.active_median_osi)} "
        f"responsive_median_osi={format_optional_float(metrics.responsive_median_osi)} "
        f"total_sites={metrics.total_sites} "
        f"active_sites={metrics.active_sites} "
        f"active_site_fraction={metrics.active_site_fraction:.6f} "
        f"responsive_sites={metrics.responsive_sites} "
        f"responsive_site_fraction={metrics.responsive_site_fraction:.6f}"
    )


def compute_multiphase_cell_responsive_metrics(
    rows_by_cell: dict[int, MultiPhaseCellTuningRow],
    threshold_hz: float,
) -> MultiPhaseCellResponsiveMetrics:
    rows = list(rows_by_cell.values())
    total_cells = len(rows)
    denominator = total_cells if total_cells > 0 else 1
    active_rows = [row for row in rows if row.peak_rate_any_phase_hz > 0.0]
    responsive_rows = [row for row in rows if row.peak_rate_any_phase_hz >= threshold_hz]
    all_sites = {row.site_id for row in rows}
    site_denominator = len(all_sites) if all_sites else 1
    active_sites = {row.site_id for row in active_rows}
    responsive_count_by_site: dict[int, int] = {}
    for row in responsive_rows:
        responsive_count_by_site[row.site_id] = responsive_count_by_site.get(row.site_id, 0) + 1
    responsive_sites_ge1 = sum(count >= 1 for count in responsive_count_by_site.values())
    responsive_sites_ge2 = sum(count >= 2 for count in responsive_count_by_site.values())
    return MultiPhaseCellResponsiveMetrics(
        total_cells=total_cells,
        active_cells=len(active_rows),
        responsive_cells=len(responsive_rows),
        active_fraction=len(active_rows) / denominator,
        responsive_fraction=len(responsive_rows) / denominator,
        responsive_median_phase_pooled_osi=optional_median(
            row.phase_pooled_osi for row in responsive_rows
        ),
        total_sites=len(all_sites),
        active_sites_ge1=len(active_sites),
        responsive_sites_ge1=responsive_sites_ge1,
        responsive_sites_ge2=responsive_sites_ge2,
        active_site_fraction_ge1=len(active_sites) / site_denominator,
        responsive_site_fraction_ge1=responsive_sites_ge1 / site_denominator,
        responsive_site_fraction_ge2=responsive_sites_ge2 / site_denominator,
        threshold_hz=threshold_hz,
    )


def print_l23e_cell_multiphase_coverage_info(
    run_label: str,
    cell_tuning: dict[int, MultiPhaseCellTuningRow] | None,
    threshold_hz: float,
) -> None:
    if cell_tuning is None:
        print(f"INFO l23e_cell_multiphase_coverage[{run_label}] available=0")
        return
    metrics = compute_multiphase_cell_responsive_metrics(cell_tuning, threshold_hz)
    print(
        f"INFO l23e_cell_multiphase_coverage[{run_label}] "
        f"available=1 "
        f"threshold_hz={metrics.threshold_hz:.6f} "
        f"total_cells={metrics.total_cells} "
        f"active_cells={metrics.active_cells} "
        f"active_fraction={metrics.active_fraction:.6f} "
        f"responsive_cells={metrics.responsive_cells} "
        f"responsive_fraction={metrics.responsive_fraction:.6f} "
        f"responsive_median_phase_pooled_osi={format_optional_float(metrics.responsive_median_phase_pooled_osi)} "
        f"total_sites={metrics.total_sites} "
        f"active_sites_ge1={metrics.active_sites_ge1} "
        f"active_site_fraction_ge1={metrics.active_site_fraction_ge1:.6f} "
        f"responsive_sites_ge1={metrics.responsive_sites_ge1} "
        f"responsive_site_fraction_ge1={metrics.responsive_site_fraction_ge1:.6f} "
        f"responsive_sites_ge2={metrics.responsive_sites_ge2} "
        f"responsive_site_fraction_ge2={metrics.responsive_site_fraction_ge2:.6f}"
    )


def fraction_at_least(values: Iterable[float], threshold: float) -> float:
    value_list = list(values)
    if not value_list:
        raise ValidationError("Cannot compute a fraction of an empty collection.")
    return sum(value >= threshold for value in value_list) / len(value_list)


def responsiveness_spatial_balance_metrics(
    cell_tuning: dict[int, MultiPhaseCellTuningRow],
    post_sites: list[PostSiteMetric],
    threshold_hz: float,
) -> dict[str, float]:
    rows = list(cell_tuning.values())
    if not rows:
        raise ValidationError("Spatial responsiveness balance requires multiphase cell tuning rows.")

    coordinate_by_site = {
        row.site_id: (row.x, row.y)
        for row in post_sites
        if row.x is not None and row.y is not None
    }
    site_ids = {row.site_id for row in rows}
    use_post_coordinates = all(site_id in coordinate_by_site for site_id in site_ids)
    if use_post_coordinates:
        xs = [coordinate_by_site[site_id][0] for site_id in site_ids]
        ys = [coordinate_by_site[site_id][1] for site_id in site_ids]
        mid_x = 0.5 * (min(xs) + max(xs))
        mid_y = 0.5 * (min(ys) + max(ys))
    else:
        side = max(1, int(math.ceil(math.sqrt(max(site_ids) + 1)))) if site_ids else 1
        mid_x = 0.5 * (side - 1)
        mid_y = 0.5 * (side - 1)

    quadrant_site_sets: dict[str, set[int]] = {
        "left_lower": set(),
        "left_upper": set(),
        "right_lower": set(),
        "right_upper": set(),
    }
    quadrant_cell_counts = {label: 0 for label in quadrant_site_sets}
    responsive_rows = [row for row in rows if row.peak_rate_any_phase_hz >= threshold_hz]

    for row in responsive_rows:
        if use_post_coordinates:
            x, y = coordinate_by_site[row.site_id]
        else:
            side = max(1, int(math.ceil(math.sqrt(max(site_ids) + 1))))
            x = float(row.site_id % side)
            y = float(row.site_id // side)
        horizontal = "left" if x <= mid_x else "right"
        vertical = "lower" if y <= mid_y else "upper"
        label = f"{horizontal}_{vertical}"
        quadrant_site_sets[label].add(row.site_id)
        quadrant_cell_counts[label] += 1

    responsive_site_count = len({row.site_id for row in responsive_rows})
    responsive_cell_count = len(responsive_rows)
    site_denominator = responsive_site_count if responsive_site_count else 1
    cell_denominator = responsive_cell_count if responsive_cell_count else 1
    min_site_fraction = min(
        len(site_set) / site_denominator for site_set in quadrant_site_sets.values()
    )
    min_cell_fraction = min(
        cell_count / cell_denominator for cell_count in quadrant_cell_counts.values()
    )
    return {
        "responsive_site_count": float(responsive_site_count),
        "responsive_cell_count": float(responsive_cell_count),
        "min_quadrant_site_fraction": min_site_fraction,
        "min_quadrant_cell_fraction": min_cell_fraction,
        "zero_site_quadrants": float(sum(len(site_set) == 0 for site_set in quadrant_site_sets.values())),
        "zero_cell_quadrants": float(sum(count == 0 for count in quadrant_cell_counts.values())),
        "used_post_site_coordinates": 1.0 if use_post_coordinates else 0.0,
        "left_lower_sites": float(len(quadrant_site_sets["left_lower"])),
        "left_upper_sites": float(len(quadrant_site_sets["left_upper"])),
        "right_lower_sites": float(len(quadrant_site_sets["right_lower"])),
        "right_upper_sites": float(len(quadrant_site_sets["right_upper"])),
        "left_lower_cells": float(quadrant_cell_counts["left_lower"]),
        "left_upper_cells": float(quadrant_cell_counts["left_upper"]),
        "right_lower_cells": float(quadrant_cell_counts["right_lower"]),
        "right_upper_cells": float(quadrant_cell_counts["right_upper"]),
    }


def count_blank_or_spontaneous_artifacts(genn_dir: Path, prefix: str) -> int:
    candidates: set[Path] = set()
    for pattern in (f"{prefix}*blank*.csv", f"{prefix}*spont*.csv"):
        candidates.update(genn_dir.glob(pattern))
    return len(candidates)


def post_site_preferences_available(rows: list[PostSiteMetric] | None) -> bool:
    return rows is not None and all(
        row.map_pref_deg is not None and row.measured_pref_deg is not None for row in rows
    )


def post_site_coordinates(
    site_ids: set[int],
    post_sites: list[PostSiteMetric],
) -> tuple[dict[int, tuple[float, float]], bool]:
    coordinate_by_site = {
        row.site_id: (row.x, row.y)
        for row in post_sites
        if row.x is not None and row.y is not None
    }
    if site_ids and all(site_id in coordinate_by_site for site_id in site_ids):
        return {site_id: coordinate_by_site[site_id] for site_id in site_ids}, True

    side = max(1, int(math.ceil(math.sqrt(max(site_ids) + 1)))) if site_ids else 1
    return {
        site_id: (float(site_id % side), float(site_id // side))
        for site_id in site_ids
    }, False


def tile_index_for_coordinate(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> int:
    x_span = max(max_x - min_x, 1.0e-9)
    y_span = max(max_y - min_y, 1.0e-9)
    tile_x = min(3, max(0, int(math.floor(4.0 * (x - min_x) / (x_span + 1.0e-9)))))
    tile_y = min(3, max(0, int(math.floor(4.0 * (y - min_y) / (y_span + 1.0e-9)))))
    return (tile_y * 4) + tile_x


def orientation_bin_index(orientation_deg: float, bin_count: int = 12) -> int:
    wrapped = positive_modulo_degrees(orientation_deg)
    return min(bin_count - 1, int(math.floor((wrapped / 180.0) * bin_count)))


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts)) if len(counts) > 1 else 0.0


def compute_l4_map_consistency_metrics(rows: list[PostSiteMetric]) -> dict[str, float]:
    active_rows = [
        row for row in rows
        if row.mean_rate_hz > 0.0 and row.map_pref_deg is not None and row.measured_pref_deg is not None
    ]
    errors = [
        circular_orientation_delta_deg(row.map_pref_deg, row.measured_pref_deg)
        for row in active_rows
        if row.map_pref_deg is not None and row.measured_pref_deg is not None
    ]
    if not errors:
        raise ValidationError("Scaling L4 map consistency requires active L4 rows with map/measured preferences.")
    return {
        "total_sites": float(len(rows)),
        "active_sites": float(len(active_rows)),
        "active_fraction": len(active_rows) / len(rows) if rows else 0.0,
        "median_error_deg": median(errors),
        "p90_error_deg": percentile(errors, 90.0),
    }


def compute_l23_l4_map_consistency_metrics(
    l23_rows: list[PostSiteMetric],
    l4_rows: list[PostSiteMetric],
    multiphase_cells: dict[int, MultiPhaseCellTuningRow],
) -> dict[str, float]:
    l4_pref_by_site = {
        row.site_id: row.measured_pref_deg
        for row in l4_rows
        if row.mean_rate_hz > 0.0 and row.measured_pref_deg is not None
    }
    active_site_errors = [
        circular_orientation_delta_deg(row.measured_pref_deg, l4_pref_by_site[row.site_id])
        for row in l23_rows
        if row.mean_rate_hz > 0.0
        and row.measured_pref_deg is not None
        and row.site_id in l4_pref_by_site
    ]
    cell5_errors = [
        circular_orientation_delta_deg(row.best_orientation_deg, l4_pref_by_site[row.site_id])
        for row in multiphase_cells.values()
        if row.peak_rate_any_phase_hz >= 5.0 and row.site_id in l4_pref_by_site
    ]
    cell10_errors = [
        circular_orientation_delta_deg(row.best_orientation_deg, l4_pref_by_site[row.site_id])
        for row in multiphase_cells.values()
        if row.peak_rate_any_phase_hz >= 10.0 and row.site_id in l4_pref_by_site
    ]
    active_site_median = optional_median(active_site_errors)
    cell5_median = optional_median(cell5_errors)
    cell10_median = optional_median(cell10_errors)
    return {
        "active_site_count": float(len(active_site_errors)),
        "active_site_median_delta_deg": active_site_median if active_site_median is not None else math.nan,
        "active_site_p90_delta_deg": percentile(active_site_errors, 90.0) if active_site_errors else math.nan,
        "cell5_count": float(len(cell5_errors)),
        "cell5_median_delta_deg": cell5_median if cell5_median is not None else math.nan,
        "cell10_count": float(len(cell10_errors)),
        "cell10_median_delta_deg": cell10_median if cell10_median is not None else math.nan,
    }


def compute_tile_orientation_metrics(
    multiphase_cells: dict[int, MultiPhaseCellTuningRow],
    post_sites: list[PostSiteMetric],
    threshold_hz: float,
) -> dict[str, float]:
    rows = [row for row in multiphase_cells.values() if row.peak_rate_any_phase_hz >= threshold_hz]
    all_site_ids = {row.site_id for row in multiphase_cells.values()}
    coordinates, used_post_coordinates = post_site_coordinates(all_site_ids, post_sites)
    if not coordinates:
        raise ValidationError("Scaling tile metrics require at least one site coordinate.")
    xs = [coordinate[0] for coordinate in coordinates.values()]
    ys = [coordinate[1] for coordinate in coordinates.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    tile_cell_counts = [0 for _ in range(16)]
    tile_bin_counts = [[0 for _ in range(12)] for _ in range(16)]
    global_bins: set[int] = set()
    for row in rows:
        x, y = coordinates[row.site_id]
        tile_index = tile_index_for_coordinate(x, y, min_x, max_x, min_y, max_y)
        orientation_bin = orientation_bin_index(row.best_orientation_deg)
        tile_cell_counts[tile_index] += 1
        tile_bin_counts[tile_index][orientation_bin] += 1
        global_bins.add(orientation_bin)

    tile_occupied_bins = [sum(count > 0 for count in counts) for counts in tile_bin_counts]
    tile_entropies = [normalized_entropy(counts) for counts in tile_bin_counts]
    low_count_threshold = 8
    bin_gate_passes = [
        occupied >= (6 if cell_count < low_count_threshold else 8)
        for cell_count, occupied in zip(tile_cell_counts, tile_occupied_bins)
    ]
    return {
        "threshold_hz": threshold_hz,
        "responsive_cells": float(len(rows)),
        "nonempty_tile_count": float(sum(count > 0 for count in tile_cell_counts)),
        "global_occupied_bins": float(len(global_bins)),
        "min_tile_cell_count": float(min(tile_cell_counts)),
        "max_tile_cell_count": float(max(tile_cell_counts)),
        "low_count_tile_count": float(sum(count < low_count_threshold for count in tile_cell_counts)),
        "min_occupied_bins": float(min(tile_occupied_bins)),
        "median_occupied_bins": median([float(value) for value in tile_occupied_bins]),
        "bin_gate_pass": 1.0 if all(bin_gate_passes) else 0.0,
        "median_entropy": median(tile_entropies),
        "min_entropy": min(tile_entropies),
        "used_post_site_coordinates": 1.0 if used_post_coordinates else 0.0,
    }


def compute_edge_quadrant_balance_metrics(
    multiphase_cells: dict[int, MultiPhaseCellTuningRow],
    post_sites: list[PostSiteMetric],
    threshold_hz: float,
) -> dict[str, float]:
    all_site_ids = {row.site_id for row in multiphase_cells.values()}
    coordinates, used_post_coordinates = post_site_coordinates(all_site_ids, post_sites)
    if not coordinates:
        raise ValidationError("Scaling edge/quadrant balance requires at least one site coordinate.")
    xs = [coordinate[0] for coordinate in coordinates.values()]
    ys = [coordinate[1] for coordinate in coordinates.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)

    edge_sites = {
        site_id
        for site_id, (x, y) in coordinates.items()
        if x == min_x or x == max_x or y == min_y or y == max_y
    }
    quadrant_cell_counts = {
        "left_lower": 0,
        "left_upper": 0,
        "right_lower": 0,
        "right_upper": 0,
    }
    responsive_rows = [
        row for row in multiphase_cells.values() if row.peak_rate_any_phase_hz >= threshold_hz
    ]
    responsive_sites = {row.site_id for row in responsive_rows}
    for row in responsive_rows:
        x, y = coordinates[row.site_id]
        horizontal = "left" if x <= mid_x else "right"
        vertical = "lower" if y <= mid_y else "upper"
        quadrant_cell_counts[f"{horizontal}_{vertical}"] += 1

    responsive_cell_count = len(responsive_rows)
    cell_denominator = responsive_cell_count if responsive_cell_count else 1
    quadrant_fractions = [
        count / cell_denominator for count in quadrant_cell_counts.values()
    ]
    edge_denominator = len(edge_sites) if edge_sites else 1
    return {
        "responsive_cells": float(responsive_cell_count),
        "responsive_sites": float(len(responsive_sites)),
        "edge_sites": float(len(edge_sites)),
        "responsive_edge_sites": float(len(edge_sites.intersection(responsive_sites))),
        "edge_site_coverage": len(edge_sites.intersection(responsive_sites)) / edge_denominator,
        "min_quadrant_cell_fraction": min(quadrant_fractions),
        "zero_quadrants": float(sum(count == 0 for count in quadrant_cell_counts.values())),
        "used_post_site_coordinates": 1.0 if used_post_coordinates else 0.0,
        "left_lower_cells": float(quadrant_cell_counts["left_lower"]),
        "left_upper_cells": float(quadrant_cell_counts["left_upper"]),
        "right_lower_cells": float(quadrant_cell_counts["right_lower"]),
        "right_upper_cells": float(quadrant_cell_counts["right_upper"]),
    }


def sign_passes(metrics: WeightMetrics, sign: str) -> bool:
    if metrics.min_nonzero is None or metrics.max_nonzero is None:
        return False
    if sign == "positive":
        return metrics.min_nonzero > 0.0
    if sign == "negative":
        return metrics.max_nonzero < 0.0
    raise ValidationError(f"Unsupported sign gate: {sign}")


def preferred_center_orientations(run: RunData) -> tuple[dict[int, float], dict[int, float]]:
    preferred_by_site: dict[int, float] = {}
    rates_by_site: dict[int, float] = {}
    for validation_site_id, rows in run.context_rows_by_site.items():
        center_l23e = rows[("center_only", "l23e")]
        pref_deg, pref_rate = max(center_l23e.rates_by_deg.items(), key=lambda item: item[1])
        preferred_by_site[validation_site_id] = pref_deg
        rates_by_site[validation_site_id] = pref_rate
    return preferred_by_site, rates_by_site


def preferred_center_orientation_deg(run: RunData) -> tuple[float, float]:
    center_l23e = run.context_rows[("center_only", "l23e")]
    pref_deg, pref_rate = max(center_l23e.rates_by_deg.items(), key=lambda item: item[1])
    return pref_deg, pref_rate


def compute_context_metrics(run: RunData, orientation_deg: float | dict[int, float]) -> dict[str, float]:
    if isinstance(orientation_deg, dict):
        preferred_by_site = orientation_deg
    else:
        primary_site_id = next(iter(run.context_rows_by_site))
        preferred_by_site = {primary_site_id: orientation_deg}

    site_metrics: list[dict[str, float]] = []
    for validation_site_id, preferred_deg in preferred_by_site.items():
        if validation_site_id not in run.context_rows_by_site:
            raise ValidationError(
                f"Context rows for validation_site_id={validation_site_id} are missing in prefix {run.prefix}"
            )
        rows = run.context_rows_by_site[validation_site_id]
        center_l23e = rows[("center_only", "l23e")]
        broad_l23e = rows[("broad_field", "l23e")]
        center_l23som = rows[("center_only", "l23som")]
        broad_l23som = rows[("broad_field", "l23som")]

        if preferred_deg not in center_l23e.rates_by_deg:
            raise ValidationError(
                f"Orientation {preferred_deg} not present in center-only L23E context row "
                f"for {run.prefix} site {validation_site_id}"
            )
        if preferred_deg not in broad_l23e.rates_by_deg:
            raise ValidationError(
                f"Orientation {preferred_deg} not present in broad-field L23E context row "
                f"for {run.prefix} site {validation_site_id}"
            )
        if preferred_deg not in center_l23som.rates_by_deg or preferred_deg not in broad_l23som.rates_by_deg:
            raise ValidationError(
                f"Orientation {preferred_deg} not present in SOM context rows "
                f"for {run.prefix} site {validation_site_id}"
            )

        center_pref_l23e = center_l23e.rates_by_deg[preferred_deg]
        broad_pref_l23e = broad_l23e.rates_by_deg[preferred_deg]
        center_pref_l23som = center_l23som.rates_by_deg[preferred_deg]
        broad_pref_l23som = broad_l23som.rates_by_deg[preferred_deg]

        if center_pref_l23e <= 0.0:
            raise ValidationError(
                f"Center-only preferred L23E rate must be positive for suppression computation "
                f"in {run.prefix} site {validation_site_id}"
            )
        if center_pref_l23som <= 0.0:
            raise ValidationError(
                f"Center-only preferred L23SOM rate must be positive for context validation "
                f"in {run.prefix} site {validation_site_id}"
            )

        driven_center_threshold_hz = max(10.0, 0.25 * center_pref_l23e)
        relevant_orientations: list[float] = []
        bsi_values: list[float] = []
        min_center_som_hz = math.inf
        min_broad_som_hz = math.inf

        for current_orientation_deg, center_rate in center_l23e.rates_by_deg.items():
            if current_orientation_deg not in broad_l23e.rates_by_deg:
                raise ValidationError(
                    f"Orientation {current_orientation_deg} not present in broad-field L23E context row "
                    f"for {run.prefix} site {validation_site_id}"
                )
            if current_orientation_deg not in center_l23som.rates_by_deg:
                raise ValidationError(
                    f"Orientation {current_orientation_deg} not present in center-only L23SOM context row "
                    f"for {run.prefix} site {validation_site_id}"
                )
            if current_orientation_deg not in broad_l23som.rates_by_deg:
                raise ValidationError(
                    f"Orientation {current_orientation_deg} not present in broad-field L23SOM context row "
                    f"for {run.prefix} site {validation_site_id}"
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
                f"for context validation in {run.prefix} site {validation_site_id}"
            )

        site_metrics.append(
            {
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
            }
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
        "validation_site_count": float(len(site_metrics)),
        "center_pref_l23e_hz": mean([site["center_pref_l23e_hz"] for site in site_metrics]),
        "min_center_pref_l23e_hz": min(site["center_pref_l23e_hz"] for site in site_metrics),
        "broad_pref_l23e_hz": mean([site["broad_pref_l23e_hz"] for site in site_metrics]),
        "center_pref_l23som_hz": mean([site["center_pref_l23som_hz"] for site in site_metrics]),
        "broad_pref_l23som_hz": mean([site["broad_pref_l23som_hz"] for site in site_metrics]),
        "preferred_bsi": mean([site["preferred_bsi"] for site in site_metrics]),
        "mean_bsi": mean([site["mean_bsi"] for site in site_metrics]),
        "driven_center_threshold_hz": mean([site["driven_center_threshold_hz"] for site in site_metrics]),
        "relevant_orientation_count": sum(site["relevant_orientation_count"] for site in site_metrics),
        "min_center_som_hz": min(site["min_center_som_hz"] for site in site_metrics),
        "min_broad_som_hz": min(site["min_broad_som_hz"] for site in site_metrics),
        "summary_mean_bsi": summary_mean_bsi,
    }


def circular_orientation_delta_deg(first_deg: float, second_deg: float) -> float:
    delta = abs((first_deg - second_deg) % 180.0)
    return min(delta, 180.0 - delta)


def require_size_rows(run: RunData) -> list[SizeTuningRow]:
    if run.size_tuning_rows is None:
        raise ValidationError(f"Size tuning rows were not loaded for prefix {run.prefix}")
    return run.size_tuning_rows


def size_validation_site_count(run: RunData) -> int:
    return len({row.validation_site_id for row in require_size_rows(run)})


def require_specificity_rows(run: RunData) -> list[SpecificityRow]:
    if run.specificity_rows is None:
        raise ValidationError(f"L23E specificity rows were not loaded for prefix {run.prefix}")
    return run.specificity_rows


def require_orientation_context_rows(run: RunData) -> list[OrientationContextRow]:
    if run.orientation_context_rows is None:
        raise ValidationError(
            f"Orientation-context suppression rows were not loaded for prefix {run.prefix}"
        )
    return run.orientation_context_rows


def safe_suppression_index(center_rate_hz: float, context_rate_hz: float) -> float:
    if center_rate_hz <= 0.0:
        return 0.0
    return (center_rate_hz - context_rate_hz) / center_rate_hz


def compute_orientation_context_suppression_metrics(
    rows: list[OrientationContextRow],
    *,
    driven_threshold_hz: float = 1.0,
) -> dict[str, float]:
    """Summarize validation-only L2/3 orientation-context suppression.

    The full-vs-control OSD difference is interpreted as plasticity-specific
    enhancement only. A no-learning control can still show real emergent
    suppression from fixed local E/I dynamics, so absence of a large delta is
    not evidence of hardcoded orientation suppression when the no-hardcode and
    absolute L23>L4 suppression gates pass.
    """

    expected_conditions = {
        "center_only",
        "same_surround",
        "orth_surround",
        "surround_same_only",
        "surround_orth_only",
    }
    rows_by_site: dict[int, dict[str, OrientationContextRow]] = {}
    for row in rows:
        site_rows = rows_by_site.setdefault(row.validation_site_id, {})
        if row.condition in site_rows:
            raise ValidationError(
                f"Duplicate orientation-context row for site={row.validation_site_id} "
                f"condition={row.condition}"
            )
        site_rows[row.condition] = row

    if not rows_by_site:
        raise ValidationError("Orientation-context suppression requested for an empty row set.")

    site_metrics: list[dict[str, float]] = []
    for validation_site_id, site_rows in rows_by_site.items():
        missing = expected_conditions.difference(site_rows)
        if missing:
            raise ValidationError(
                f"Missing orientation-context conditions for site={validation_site_id}: {sorted(missing)}"
            )
        center = site_rows["center_only"]
        same = site_rows["same_surround"]
        orth = site_rows["orth_surround"]
        surround_same = site_rows["surround_same_only"]
        surround_orth = site_rows["surround_orth_only"]

        si_same_l23e = safe_suppression_index(center.l23e_rate_hz, same.l23e_rate_hz)
        si_orth_l23e = safe_suppression_index(center.l23e_rate_hz, orth.l23e_rate_hz)
        osd_l23e = si_same_l23e - si_orth_l23e
        si_same_l4e = safe_suppression_index(center.l4e_rate_hz, same.l4e_rate_hz)
        si_orth_l4e = safe_suppression_index(center.l4e_rate_hz, orth.l4e_rate_hz)
        osd_l4e = si_same_l4e - si_orth_l4e
        surround_only_l23e_hz = max(surround_same.l23e_rate_hz, surround_orth.l23e_rate_hz)
        surround_only_ratio = (
            surround_only_l23e_hz / center.l23e_rate_hz if center.l23e_rate_hz > 0.0 else math.inf
        )
        site_metrics.append(
            {
                "validation_site_id": float(validation_site_id),
                "center_l23e_hz": center.l23e_rate_hz,
                "center_l4e_hz": center.l4e_rate_hz,
                "si_same_l23e": si_same_l23e,
                "si_orth_l23e": si_orth_l23e,
                "osd_l23e": osd_l23e,
                "osd_l4e": osd_l4e,
                "osd_l23e_minus_l4e": osd_l23e - osd_l4e,
                "surround_only_l23e_hz": surround_only_l23e_hz,
                "surround_only_l23e_ratio": surround_only_ratio,
            }
        )

    driven_sites = [
        site for site in site_metrics if site["center_l23e_hz"] >= driven_threshold_hz
    ]
    if not driven_sites:
        return {
            "site_count": float(len(site_metrics)),
            "driven_count": 0.0,
            "driven_fraction": 0.0,
            "driven_threshold_hz": driven_threshold_hz,
            "mean_center_l23e_hz": mean([site["center_l23e_hz"] for site in site_metrics]),
            "mean_si_same_l23e": math.nan,
            "median_si_same_l23e": math.nan,
            "mean_osd_l23e": math.nan,
            "median_osd_l23e": math.nan,
            "frac_osd_gt_0p05": 0.0,
            "mean_osd_l4e": math.nan,
            "mean_osd_l23e_minus_l4e": math.nan,
            "mean_surround_only_l23e_hz": math.nan,
            "mean_surround_only_l23e_ratio": math.nan,
            "max_surround_only_l23e_ratio": math.nan,
        }

    return {
        "site_count": float(len(site_metrics)),
        "driven_count": float(len(driven_sites)),
        "driven_fraction": len(driven_sites) / len(site_metrics),
        "driven_threshold_hz": driven_threshold_hz,
        "mean_center_l23e_hz": mean([site["center_l23e_hz"] for site in driven_sites]),
        "mean_si_same_l23e": mean([site["si_same_l23e"] for site in driven_sites]),
        "median_si_same_l23e": median(site["si_same_l23e"] for site in driven_sites),
        "mean_osd_l23e": mean([site["osd_l23e"] for site in driven_sites]),
        "median_osd_l23e": median(site["osd_l23e"] for site in driven_sites),
        "frac_osd_gt_0p05": sum(site["osd_l23e"] > 0.05 for site in driven_sites) / len(driven_sites),
        "mean_osd_l4e": mean([site["osd_l4e"] for site in driven_sites]),
        "mean_osd_l23e_minus_l4e": mean([site["osd_l23e_minus_l4e"] for site in driven_sites]),
        "mean_surround_only_l23e_hz": mean([site["surround_only_l23e_hz"] for site in driven_sites]),
        "mean_surround_only_l23e_ratio": mean([site["surround_only_l23e_ratio"] for site in driven_sites]),
        "max_surround_only_l23e_ratio": max(site["surround_only_l23e_ratio"] for site in driven_sites),
    }


def blank_rows_by_population(rows: list[BlankBaselineRow]) -> dict[str, list[BlankBaselineRow]]:
    by_population: dict[str, list[BlankBaselineRow]] = {}
    for row in rows:
        by_population.setdefault(row.population, []).append(row)
    return by_population


def compute_blank_baseline_metrics(rows: list[BlankBaselineRow]) -> dict[str, float]:
    required_populations = {"l4e", "l23e", "l23pv", "l23som"}
    by_population = blank_rows_by_population(rows)
    missing = required_populations.difference(by_population)
    if missing:
        raise ValidationError(f"Blank baseline rows missing populations: {sorted(missing)}")

    metrics: dict[str, float] = {
        "population_count": float(len(by_population)),
        "row_count": float(len(rows)),
    }
    for population in sorted(required_populations):
        population_rows = by_population[population]
        rates = [row.rate_hz for row in population_rows]
        repeats = {row.repeat_index for row in population_rows}
        sites = {row.site_id for row in population_rows}
        metrics[f"{population}_repeat_count"] = float(len(repeats))
        metrics[f"{population}_site_count"] = float(len(sites))
        metrics[f"{population}_mean_hz"] = mean(rates)
        metrics[f"{population}_p50_hz"] = percentile(rates, 50.0)
        metrics[f"{population}_p95_hz"] = percentile(rates, 95.0)
        metrics[f"{population}_p99_hz"] = percentile(rates, 99.0)
        metrics[f"{population}_max_hz"] = max(rates)
        metrics[f"{population}_frac_lt1"] = sum(rate < 1.0 for rate in rates) / len(rates)
    return metrics


def compute_contrast_sweep_metrics(rows: list[ContrastSweepRow]) -> dict[str, float]:
    required_populations = {"l4e", "l23e", "l23pv", "l23som"}
    populations = {row.population for row in rows}
    missing = required_populations.difference(populations)
    if missing:
        raise ValidationError(f"Contrast sweep rows missing populations: {sorted(missing)}")

    contrasts = sorted({row.contrast for row in rows})
    if len(contrasts) < 2:
        raise ValidationError("Contrast sweep requires at least two contrast levels.")
    low_contrast = contrasts[0]
    high_contrast = contrasts[-1]

    rates_by_population_site_contrast: dict[tuple[str, int, float], list[float]] = {}
    for row in rows:
        if row.population not in required_populations:
            continue
        rates_by_population_site_contrast.setdefault(
            (row.population, row.validation_site_id, row.contrast),
            [],
        ).append(row.rate_hz)

    metrics: dict[str, float] = {
        "contrast_count": float(len(contrasts)),
        "low_contrast": low_contrast,
        "high_contrast": high_contrast,
    }
    for population in sorted(required_populations):
        site_ids = sorted({
            site_id
            for pop, site_id, contrast in rates_by_population_site_contrast
            if pop == population and contrast in {low_contrast, high_contrast}
        })
        paired_low: list[float] = []
        paired_high: list[float] = []
        for site_id in site_ids:
            low_values = rates_by_population_site_contrast.get((population, site_id, low_contrast), [])
            high_values = rates_by_population_site_contrast.get((population, site_id, high_contrast), [])
            if not low_values or not high_values:
                continue
            paired_low.append(mean(low_values))
            paired_high.append(mean(high_values))

        if not paired_low or len(paired_low) != len(paired_high):
            raise ValidationError(f"Contrast sweep lacks paired low/high rows for {population}.")
        high_p99 = percentile(paired_high, 99.0)
        metrics[f"{population}_site_count"] = float(len(paired_low))
        metrics[f"{population}_low_mean_hz"] = mean(paired_low)
        metrics[f"{population}_high_mean_hz"] = mean(paired_high)
        metrics[f"{population}_high_p99_hz"] = high_p99
        metrics[f"{population}_mean_delta_hz"] = mean([high - low for low, high in zip(paired_low, paired_high)])
        metrics[f"{population}_monotonic_fraction"] = (
            sum(high >= low for low, high in zip(paired_low, paired_high)) / len(paired_low)
        )
    return metrics


def compute_annular_protocol_metrics(rows: list[OrientationContextRow]) -> dict[str, float]:
    surround_only = [
        row for row in rows
        if row.condition in {"surround_same_only", "surround_orth_only"}
    ]
    same_or_orth = [
        row for row in rows
        if row.condition in {"same_surround", "orth_surround"}
    ]
    if not surround_only or not same_or_orth:
        return {
            "surround_only_row_count": float(len(surround_only)),
            "same_orth_row_count": float(len(same_or_orth)),
            "min_inner_radius_sites": math.nan,
            "min_outer_minus_inner_sites": math.nan,
            "annular_row_fraction": 0.0,
        }
    annular_rows = [
        row for row in surround_only
        if row.inner_radius_sites > 0.0 and row.aperture_radius_sites > row.inner_radius_sites
    ]
    outer_minus_inner = [
        row.aperture_radius_sites - row.inner_radius_sites
        for row in annular_rows
    ]
    return {
        "surround_only_row_count": float(len(surround_only)),
        "same_orth_row_count": float(len(same_or_orth)),
        "min_inner_radius_sites": min(row.inner_radius_sites for row in surround_only),
        "min_outer_minus_inner_sites": min(outer_minus_inner) if outer_minus_inner else math.nan,
        "annular_row_fraction": len(annular_rows) / len(surround_only),
    }


def build_size_tuning_grid(
    run: RunData,
) -> tuple[dict[int, dict[str, dict[float, dict[float, float]]]], list[float], list[float], list[int]]:
    rows = require_size_rows(run)
    required_populations = {"l4e", "l23e", "l23pv", "l23som"}
    radii = sorted({row.radius_sites for row in rows})
    orientations = sorted({row.orientation_deg for row in rows})
    validation_site_ids: list[int] = []
    for row in rows:
        if row.population in required_populations and row.validation_site_id not in validation_site_ids:
            validation_site_ids.append(row.validation_site_id)
    if len(radii) < 3:
        raise ValidationError(f"Size tuning requires at least 3 radii for prefix {run.prefix}")
    if not orientations:
        raise ValidationError(f"Size tuning requires at least one orientation for prefix {run.prefix}")
    if not validation_site_ids:
        raise ValidationError(f"Size tuning requires at least one validation site for prefix {run.prefix}")

    grid: dict[int, dict[str, dict[float, dict[float, float]]]] = {
        validation_site_id: {
            population: {radius: {} for radius in radii}
            for population in required_populations
        }
        for validation_site_id in validation_site_ids
    }
    seen: set[tuple[int, str, float, float]] = set()
    for row in rows:
        if row.population not in required_populations:
            continue
        key = (row.validation_site_id, row.population, row.radius_sites, row.orientation_deg)
        if key in seen:
            raise ValidationError(
                f"Duplicate size tuning row for site={row.validation_site_id} "
                f"{row.population} radius={row.radius_sites} "
                f"orientation={row.orientation_deg} in prefix {run.prefix}"
            )
        seen.add(key)
        grid[row.validation_site_id][row.population][row.radius_sites][row.orientation_deg] = row.rate_hz

    expected = {(radius, orientation) for radius in radii for orientation in orientations}
    for validation_site_id in validation_site_ids:
        for population in required_populations:
            observed = {
                (radius, orientation)
                for radius, rates_by_orientation in grid[validation_site_id][population].items()
                for orientation in rates_by_orientation
            }
            missing = expected.difference(observed)
            if missing:
                raise ValidationError(
                    f"Missing size tuning rows for site={validation_site_id} {population} "
                    f"in prefix {run.prefix}: {sorted(missing)[:5]}"
                )

    return grid, radii, orientations, validation_site_ids


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
    selected_orientations: list[float] | dict[int, list[float]] | None = None,
) -> dict[str, object]:
    grids, radii, orientations, validation_site_ids = build_size_tuning_grid(run)
    reference_radius = min(radii, key=lambda radius: abs(radius - 2.0))
    selected_by_site: dict[int, list[float]] = {}
    preferred_by_site: dict[int, float] = {}
    preferred_rate_by_site: dict[int, float] = {}

    for validation_site_id in validation_site_ids:
        grid = grids[validation_site_id]
        reference_rates = grid["l23e"][reference_radius]
        preferred_deg, preferred_rate = max(reference_rates.items(), key=lambda item: item[1])
        preferred_by_site[validation_site_id] = preferred_deg
        preferred_rate_by_site[validation_site_id] = preferred_rate

        if selected_orientations is None:
            driven_threshold_hz = max(1.0, 0.25 * preferred_rate)
            site_selected_orientations = [
                orientation
                for orientation in orientations
                if circular_orientation_delta_deg(orientation, preferred_deg) <= 22.5
                and reference_rates[orientation] >= driven_threshold_hz
            ]
            if not site_selected_orientations:
                site_selected_orientations = [preferred_deg]
        elif isinstance(selected_orientations, dict):
            if validation_site_id not in selected_orientations:
                raise ValidationError(
                    f"Size tuning selected orientations missing for prefix {run.prefix} "
                    f"site {validation_site_id}"
                )
            site_selected_orientations = selected_orientations[validation_site_id]
        else:
            site_selected_orientations = selected_orientations

        missing_orientations = set(site_selected_orientations).difference(orientations)
        if missing_orientations:
            raise ValidationError(
                f"Size tuning orientations missing for prefix {run.prefix} "
                f"site {validation_site_id}: {sorted(missing_orientations)}"
            )
        selected_by_site[validation_site_id] = list(site_selected_orientations)

    mean_rates: dict[str, list[float]] = {
        population: [0.0 for _ in radii]
        for population in ("l4e", "l23e", "l23pv", "l23som")
    }
    for validation_site_id in validation_site_ids:
        grid = grids[validation_site_id]
        site_selected_orientations = selected_by_site[validation_site_id]
        for population in ("l4e", "l23e", "l23pv", "l23som"):
            site_rates = [
                sum(grid[population][radius][orientation] for orientation in site_selected_orientations)
                / len(site_selected_orientations)
                for radius in radii
            ]
            for index, rate in enumerate(site_rates):
                mean_rates[population][index] += rate / len(validation_site_ids)

    primary_site_id = validation_site_ids[0]

    return {
        "validation_site_count": float(len(validation_site_ids)),
        "primary_validation_site_id": float(primary_site_id),
        "radii": radii,
        "selected_orientations": selected_by_site[primary_site_id],
        "selected_orientations_by_site": selected_by_site,
        "preferred_deg": preferred_by_site[primary_site_id],
        "preferred_rate": preferred_rate_by_site[primary_site_id],
        "mean_preferred_rate": mean(list(preferred_rate_by_site.values())),
        "min_preferred_rate": min(preferred_rate_by_site.values()),
        "reference_radius": reference_radius,
        "mean_rates": mean_rates,
        "l23e": summarize_size_curve(radii, mean_rates["l23e"]),
        "l4e": summarize_size_curve(radii, mean_rates["l4e"]),
        "l23som": summarize_size_curve(radii, mean_rates["l23som"]),
    }


def site_size_rates(
    grid: dict[str, dict[float, dict[float, float]]],
    population: str,
    radii: list[float],
    selected_orientations: list[float],
) -> list[float]:
    if not selected_orientations:
        raise ValidationError("Site size rates require at least one selected orientation.")
    return [
        sum(grid[population][radius][orientation] for orientation in selected_orientations)
        / len(selected_orientations)
        for radius in radii
    ]


def compute_som_size_surround_metrics(
    full: RunData,
    somoff: RunData,
    full_size: dict[str, object],
    somoff_size: dict[str, object],
    full_context: dict[str, float],
) -> dict[str, float]:
    full_grids, radii, _, validation_site_ids = build_size_tuning_grid(full)
    somoff_grids, somoff_radii, _, somoff_site_ids = build_size_tuning_grid(somoff)
    if radii != somoff_radii:
        raise ValidationError("SOM size/surround requires aligned full and somoff size radii.")
    selected_by_site = full_size["selected_orientations_by_site"]
    if not isinstance(selected_by_site, dict):
        raise ValidationError("SOM size/surround expected selected orientations by site.")

    site_curve_passes = 0
    site_driven_count = 0
    site_large_suppressed_count = 0
    site_rescue_count = 0
    site_large_rate_delta_values: list[float] = []
    site_suppression_reduction_values: list[float] = []
    site_count = 0

    for validation_site_id in validation_site_ids:
        if validation_site_id not in somoff_site_ids:
            raise ValidationError(f"SOM size/surround missing somoff size site {validation_site_id}.")
        if validation_site_id not in selected_by_site:
            raise ValidationError(f"SOM size/surround missing selected orientations for site {validation_site_id}.")
        selected_orientations = selected_by_site[validation_site_id]
        if not isinstance(selected_orientations, list):
            raise ValidationError(f"SOM size/surround malformed selected orientations for site {validation_site_id}.")

        full_l23e_curve = summarize_size_curve(
            radii,
            site_size_rates(full_grids[validation_site_id], "l23e", radii, selected_orientations),
        )
        somoff_l23e_curve = summarize_size_curve(
            radii,
            site_size_rates(somoff_grids[validation_site_id], "l23e", radii, selected_orientations),
        )
        peak_index = int(full_l23e_curve["peak_index"])
        driven = full_l23e_curve["peak_rate"] >= 1.0
        interior = 0 < peak_index < (len(radii) - 1)
        early = full_l23e_curve["early_delta"] > 0.0
        suppressed = full_l23e_curve["large_rate"] < full_l23e_curve["peak_rate"]
        site_count += 1
        site_driven_count += int(driven)
        site_large_suppressed_count += int(suppressed)
        site_curve_passes += int(driven and interior and early and suppressed)

        large_rate_delta = somoff_l23e_curve["large_rate"] - full_l23e_curve["large_rate"]
        suppression_reduction = full_l23e_curve["suppression"] - somoff_l23e_curve["suppression"]
        site_large_rate_delta_values.append(large_rate_delta)
        site_suppression_reduction_values.append(suppression_reduction)
        site_rescue_count += int(large_rate_delta >= 0.0 or suppression_reduction >= 0.03)

    full_l23e = full_size["l23e"]
    full_l4e = full_size["l4e"]
    full_som = full_size["l23som"]
    somoff_l23e = somoff_size["l23e"]
    peak_rate = full_l23e["peak_rate"]
    summation_index = (
        (full_l23e["peak_rate"] - full_l23e["small_rate"]) / peak_rate
        if peak_rate > 0.0
        else -math.inf
    )
    l23e_l4_suppression_delta = full_l23e["suppression"] - full_l4e["suppression"]
    center_or_peak_som = max(
        full_som["small_rate"],
        full_som["peak_rate"],
        full_context["center_pref_l23som_hz"],
    )
    large_or_broad_som = max(full_som["large_rate"], full_context["broad_pref_l23som_hz"])
    som_recruitment_index = (
        (large_or_broad_som - center_or_peak_som) / center_or_peak_som
        if center_or_peak_som > 0.0
        else math.inf
    )
    mean_large_rate_delta = mean(site_large_rate_delta_values)
    mean_suppression_reduction = mean(site_suppression_reduction_values)

    return {
        "site_count": float(site_count),
        "site_curve_pass_fraction": site_curve_passes / site_count if site_count else 0.0,
        "site_driven_fraction": site_driven_count / site_count if site_count else 0.0,
        "site_large_suppressed_fraction": site_large_suppressed_count / site_count if site_count else 0.0,
        "peak_radius": full_l23e["peak_radius"],
        "small_l23e_rate": full_l23e["small_rate"],
        "peak_l23e_rate": full_l23e["peak_rate"],
        "large_l23e_rate": full_l23e["large_rate"],
        "summation_index": summation_index,
        "large_suppression_index": full_l23e["suppression"],
        "l23e_l4_suppression_delta": l23e_l4_suppression_delta,
        "l4_suppression": full_l4e["suppression"],
        "l23e_suppression": full_l23e["suppression"],
        "small_som_rate": full_som["small_rate"],
        "peak_som_rate": full_som["peak_rate"],
        "large_som_rate": full_som["large_rate"],
        "center_context_som_rate": full_context["center_pref_l23som_hz"],
        "broad_context_som_rate": full_context["broad_pref_l23som_hz"],
        "som_center_or_peak_rate": center_or_peak_som,
        "som_large_or_broad_rate": large_or_broad_som,
        "som_recruitment_index": som_recruitment_index,
        "somoff_large_l23e_rate": somoff_l23e["large_rate"],
        "somoff_l23e_suppression": somoff_l23e["suppression"],
        "mean_large_rate_delta_somoff_minus_full": mean_large_rate_delta,
        "mean_suppression_reduction_full_minus_somoff": mean_suppression_reduction,
        "site_rescue_fraction": site_rescue_count / site_count if site_count else 0.0,
    }


def format_float_list(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"


def load_l4_intersite_metrics(genn_dir: Path, prefix: str) -> dict[str, float]:
    return parse_summary_csv(require_file(genn_dir / f"{prefix}_l4_intersite_diagnostics.csv"))


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


def l23ee_upper_bound() -> float:
    for spec in WEIGHT_SPECS:
        if spec.name == "l23ee":
            return spec.upper
    raise ValidationError("Missing l23ee weight spec.")


def gini_coefficient(values: list[float]) -> float:
    if not values:
        raise ValidationError("Cannot compute Gini coefficient for an empty collection.")
    sorted_values = sorted(values)
    total = sum(sorted_values)
    if total <= 0.0:
        return 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(sorted_values))
    count = len(sorted_values)
    return ((2.0 * weighted_sum) / (count * total)) - ((count + 1.0) / count)


def mass_share_top_fraction(values: list[float], fraction: float) -> float:
    if not values:
        raise ValidationError("Cannot compute mass share for an empty collection.")
    total = sum(values)
    if total <= 0.0:
        return 0.0
    top_count = max(1, int(math.ceil(fraction * len(values))))
    return sum(sorted(values, reverse=True)[:top_count]) / total


def compute_l23ee_recurrent_heavy_tail_metrics(rows: list[SpecificityRow]) -> dict[str, float]:
    weights = [row.w_after for row in rows if row.w_after > 0.0]
    if not weights:
        raise ValidationError("L23EE recurrent heavy-tail diagnostics require positive active w_after rows.")

    average = mean(weights)
    variance = mean([(weight - average) ** 2 for weight in weights])
    std = math.sqrt(variance)
    upper = l23ee_upper_bound()
    upper_tolerance = max(1.0e-9, 1.0e-6 * abs(upper))
    return {
        "active_count": float(len(weights)),
        "mean": average,
        "std": std,
        "cv": (std / average) if average > 0.0 else math.inf,
        "p50": percentile(weights, 50.0),
        "p90": percentile(weights, 90.0),
        "p95": percentile(weights, 95.0),
        "p99": percentile(weights, 99.0),
        "max": max(weights),
        "gini": gini_coefficient(weights),
        "top1_mass_share": mass_share_top_fraction(weights, 0.01),
        "top5_mass_share": mass_share_top_fraction(weights, 0.05),
        "top10_mass_share": mass_share_top_fraction(weights, 0.10),
        "upper_bound": upper,
        "upper_cap_fraction": sum(abs(weight - upper) <= upper_tolerance for weight in weights) / len(weights),
    }


def select_recurrent_biology_rows(rows: list[SpecificityRow]) -> tuple[list[SpecificityRow], str, int]:
    active_endpoint_rows = [row for row in rows if row.pre_peak_hz > 0.0 and row.post_peak_hz > 0.0]
    min_active_count = min(100, max(20, len(rows) // 100))
    if len(active_endpoint_rows) >= min_active_count:
        return active_endpoint_rows, "active_endpoints", min_active_count

    peak5_rows = [row for row in rows if row.pre_peak_hz >= 5.0 and row.post_peak_hz >= 5.0]
    if len(peak5_rows) >= min_active_count:
        return peak5_rows, "peak_ge_5hz_endpoints", min_active_count

    return rows, "all", min_active_count


def recurrent_distance_bin(row: SpecificityRow) -> str:
    if row.distance_sites <= 1.0:
        return "d_le_1"
    if row.distance_sites <= 2.0:
        return "d_1_2"
    return "d_gt_2"


def top10_corr_enrichment(rows: list[SpecificityRow], corr_values: list[float]) -> tuple[float, float, float]:
    if len(rows) != len(corr_values):
        raise ValidationError("Correlation enrichment requires aligned rows and correlation values.")
    if not rows:
        raise ValidationError("Correlation enrichment requires non-empty rows.")
    top_count = max(1, int(math.ceil(0.10 * len(rows))))
    top_indices = sorted(range(len(rows)), key=lambda index: rows[index].w_after, reverse=True)[:top_count]
    all_fraction = sum(corr > 0.20 for corr in corr_values) / len(corr_values)
    top_fraction = sum(corr_values[index] > 0.20 for index in top_indices) / top_count
    return top_fraction, all_fraction, top_fraction - all_fraction


def compute_l23ee_recurrent_shuffle_specificity_metrics(
    rows: list[SpecificityRow],
    *,
    shuffle_count: int = 200,
    seed: int = 1729,
) -> dict[str, float | str]:
    selected_rows, selected_label, min_active_count = select_recurrent_biology_rows(rows)
    if not selected_rows:
        raise ValidationError("Shuffle specificity requires non-empty recurrent rows.")

    observed_corr = [row.response_corr for row in selected_rows]
    observed_top_fraction, observed_all_fraction, observed_delta = top10_corr_enrichment(selected_rows, observed_corr)
    top_count = max(1, int(math.ceil(0.10 * len(selected_rows))))
    qualifying_count = sum(corr > 0.20 for corr in observed_corr)

    strata: dict[tuple[str, bool], list[int]] = {}
    for index, row in enumerate(selected_rows):
        active_endpoint = row.pre_peak_hz > 0.0 and row.post_peak_hz > 0.0
        strata.setdefault((recurrent_distance_bin(row), active_endpoint), []).append(index)

    rng = random.Random(seed)
    shuffled_deltas: list[float] = []
    for _ in range(shuffle_count):
        shuffled_corr = list(observed_corr)
        for indices in strata.values():
            values = [shuffled_corr[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                shuffled_corr[index] = value
        _, _, shuffled_delta = top10_corr_enrichment(selected_rows, shuffled_corr)
        shuffled_deltas.append(shuffled_delta)

    shuffle_mean = mean(shuffled_deltas)
    shuffle_std = math.sqrt(mean([(value - shuffle_mean) ** 2 for value in shuffled_deltas]))
    z_score = (
        (observed_delta - shuffle_mean) / shuffle_std
        if shuffle_std > 0.0
        else (math.inf if observed_delta > shuffle_mean else 0.0)
    )
    return {
        "selected_label": selected_label,
        "row_count": float(len(selected_rows)),
        "all_row_count": float(len(rows)),
        "min_active_count": float(min_active_count),
        "top_count": float(top_count),
        "qualifying_count": float(qualifying_count),
        "observed_top_fraction": observed_top_fraction,
        "observed_all_fraction": observed_all_fraction,
        "observed_delta": observed_delta,
        "shuffle_count": float(shuffle_count),
        "shuffle_seed": float(seed),
        "shuffle_mean_delta": shuffle_mean,
        "shuffle_std_delta": shuffle_std,
        "shuffle_q95_delta": percentile(shuffled_deltas, 95.0),
        "z_score": z_score,
        "strata_count": float(len(strata)),
    }


def summarize_recurrent_bin(rows: list[SpecificityRow], prefix: str) -> dict[str, float]:
    if not rows:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean_w_after": math.nan,
            f"{prefix}_median_w_after": math.nan,
            f"{prefix}_mean_delta_w": math.nan,
            f"{prefix}_median_delta_w": math.nan,
        }
    return {
        f"{prefix}_count": float(len(rows)),
        f"{prefix}_mean_w_after": mean([row.w_after for row in rows]),
        f"{prefix}_median_w_after": median(row.w_after for row in rows),
        f"{prefix}_mean_delta_w": mean([row.delta_w for row in rows]),
        f"{prefix}_median_delta_w": median(row.delta_w for row in rows),
    }


def compute_l23ee_recurrent_cotuning_bin_metrics(rows: list[SpecificityRow]) -> dict[str, float]:
    bins = {
        "corr_le_0": [row for row in rows if row.response_corr <= 0.0],
        "corr_0_0p2": [row for row in rows if 0.0 < row.response_corr <= 0.20],
        "corr_0p2_0p5": [row for row in rows if 0.20 < row.response_corr <= 0.50],
        "corr_gt_0p5": [row for row in rows if row.response_corr > 0.50],
    }
    low_rows = bins["corr_le_0"] + bins["corr_0_0p2"]
    high_rows = bins["corr_0p2_0p5"] + bins["corr_gt_0p5"]
    min_count = min(50, max(5, len(rows) // 100))
    metrics = {
        "row_count": float(len(rows)),
        "low_count": float(len(low_rows)),
        "high_count": float(len(high_rows)),
        "min_count": float(min_count),
        "p95_abs_delta_w": percentile([abs(row.delta_w) for row in rows], 95.0),
    }
    for label, bin_rows in bins.items():
        metrics.update(summarize_recurrent_bin(bin_rows, label))
    metrics.update(summarize_recurrent_bin(low_rows, "low_corr_le_0p2"))
    metrics.update(summarize_recurrent_bin(high_rows, "high_corr_gt_0p2"))
    return metrics


def format_l23ee_recurrent_cotuning_bins(metrics: dict[str, float]) -> str:
    labels = ("corr_le_0", "corr_0_0p2", "corr_0p2_0p5", "corr_gt_0p5")
    parts: list[str] = []
    for label in labels:
        parts.append(
            f"{label}_count={int(metrics[f'{label}_count'])} "
            f"{label}_mean_w_after={metrics[f'{label}_mean_w_after']:.6f} "
            f"{label}_median_w_after={metrics[f'{label}_median_w_after']:.6f} "
            f"{label}_mean_delta_w={metrics[f'{label}_mean_delta_w']:.6f} "
            f"{label}_median_delta_w={metrics[f'{label}_median_delta_w']:.6f}"
        )
    return " ".join(parts)


def compute_l23ee_recurrent_reciprocal_metrics(rows: list[SpecificityRow]) -> dict[str, float]:
    active_rows = [row for row in rows if row.w_after > 0.0]
    if not active_rows:
        raise ValidationError("Reciprocal recurrent metrics require positive active rows.")
    pair_set = {(row.pre_id, row.post_id) for row in active_rows}
    reciprocal_rows = [row for row in active_rows if (row.post_id, row.pre_id) in pair_set]
    nonreciprocal_rows = [row for row in active_rows if (row.post_id, row.pre_id) not in pair_set]
    top_count = max(1, int(math.ceil(0.10 * len(active_rows))))
    top_rows = sorted(active_rows, key=lambda row: row.w_after, reverse=True)[:top_count]
    all_fraction = len(reciprocal_rows) / len(active_rows)
    top_fraction = sum((row.post_id, row.pre_id) in pair_set for row in top_rows) / top_count
    return {
        "active_count": float(len(active_rows)),
        "reciprocal_count": float(len(reciprocal_rows)),
        "nonreciprocal_count": float(len(nonreciprocal_rows)),
        "reciprocal_fraction": all_fraction,
        "reciprocal_mean_w_after": mean([row.w_after for row in reciprocal_rows]) if reciprocal_rows else math.nan,
        "nonreciprocal_mean_w_after": mean([row.w_after for row in nonreciprocal_rows]) if nonreciprocal_rows else math.nan,
        "top10_count": float(top_count),
        "top10_reciprocal_fraction": top_fraction,
        "top10_reciprocal_enrichment": top_fraction - all_fraction,
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
        if args.min_validation_sites < 1:
            raise ValidationError("--min-validation-sites must be at least 1.")
        if args.responsive_rate_threshold_hz < 0.0:
            raise ValidationError("--responsive-rate-threshold-hz must be non-negative.")
        if args.cell_responsive_threshold_hz < 0.0:
            raise ValidationError("--cell-responsive-threshold-hz must be non-negative.")
        if args.require_pv_gain_normalization and not args.pvweak:
            raise ValidationError("--require-pv-gain-normalization requires --pvweak PREFIX.")
        full = load_run(
            args.genn_dir,
            args.full,
            require_size_tuning=True,
            require_specificity=True,
        )
        control = load_run(args.genn_dir, args.control)
        somoff = load_run(args.genn_dir, args.somoff, require_size_tuning=True)
        pvweak = load_run(args.genn_dir, args.pvweak) if args.require_pv_gain_normalization else None

        overall_ok = True
        l23e_osi_metrics_by_label = {
            run_label: compute_l23e_osi_site_metrics(
                run.l23e_post_sites,
                args.responsive_rate_threshold_hz,
            )
            for run_label, run in (("full", full), ("control", control), ("somoff", somoff))
        }
        for run_label, run in (("full", full), ("control", control), ("somoff", somoff)):
            print_l23e_osi_site_info(run_label, l23e_osi_metrics_by_label[run_label])
            print(
                f"INFO l23e_osi_quadrants[{run_label}] "
                f"{format_l23e_osi_quadrants(run.l23e_post_sites, args.responsive_rate_threshold_hz)}"
            )
            print_l23e_cell_coverage_info(
                run_label,
                run.l23e_cell_tuning,
                args.cell_responsive_threshold_hz,
            )
            print_l23e_cell_multiphase_coverage_info(
                run_label,
                run.l23e_cell_tuning_multiphase,
                args.cell_responsive_threshold_hz,
            )
            if "l4_l23_orientation_bias_strength" in run.summary:
                print(
                    f"INFO no_hardcode_audit[{run_label}] "
                    f"l4_l23_orientation_bias_strength="
                    f"{run.summary['l4_l23_orientation_bias_strength']:.6f} "
                    f"feedforward_orientation_prior_enabled="
                    f"{run.summary.get('l4_l23_feedforward_orientation_prior_enabled', math.nan):.6f} "
                    f"inhibitory_orientation_rule_enabled="
                    f"{run.summary.get('inhibitory_orientation_rule_enabled', math.nan):.6f} "
                    f"orientation_context_assay_enabled="
                    f"{run.summary.get('orientation_context_assay_enabled', math.nan):.6f}"
                )

        if args.require_sensory_baseline_contrast_annulus:
            sensory_enabled = full.summary.get("sensory_assay_enabled", 0.0)
            blank_available = full.blank_baseline_rows is not None and sensory_enabled == 1.0
            blank_missing = "none" if full.blank_baseline_rows is not None else f"{full.prefix}_blank_baseline.csv"
            overall_ok &= print_result(
                blank_available,
                "sensory_blank_artifacts_available",
                (
                    f"sensory_assay_enabled={sensory_enabled:.6f} "
                    f"blank_rows={len(full.blank_baseline_rows) if full.blank_baseline_rows is not None else 0} "
                    f"missing={blank_missing}"
                ),
            )
            if blank_available and full.blank_baseline_rows is not None:
                blank = compute_blank_baseline_metrics(full.blank_baseline_rows)
                min_repeat_count = min(
                    blank["l4e_repeat_count"],
                    blank["l23e_repeat_count"],
                    blank["l23pv_repeat_count"],
                    blank["l23som_repeat_count"],
                )
                print(
                    "INFO sensory_blank_summary "
                    f"repeat_count={int(min_repeat_count)} "
                    f"l4e_p50={blank['l4e_p50_hz']:.6f} "
                    f"l4e_p95={blank['l4e_p95_hz']:.6f} "
                    f"l4e_p99={blank['l4e_p99_hz']:.6f} "
                    f"l23e_p50={blank['l23e_p50_hz']:.6f} "
                    f"l23e_p95={blank['l23e_p95_hz']:.6f} "
                    f"l23e_p99={blank['l23e_p99_hz']:.6f} "
                    f"l23pv_p99={blank['l23pv_p99_hz']:.6f} "
                    f"l23som_p99={blank['l23som_p99_hz']:.6f}"
                )
                overall_ok &= print_result(
                    min_repeat_count >= 4.0
                    and blank["l4e_p99_hz"] <= 0.05
                    and blank["l4e_max_hz"] <= 0.10,
                    "sensory_blank_l4_low",
                    (
                        f"repeat_count={int(min_repeat_count)} "
                        f"l4e_mean_hz={blank['l4e_mean_hz']:.6f} "
                        f"l4e_p99_hz={blank['l4e_p99_hz']:.6f} "
                        f"l4e_max_hz={blank['l4e_max_hz']:.6f}"
                    ),
                )
                overall_ok &= print_result(
                    blank["l23e_frac_lt1"] >= 0.90
                    and blank["l23e_p95_hz"] <= 1.0
                    and blank["l23e_p99_hz"] <= 5.0,
                    "sensory_blank_l23e_sparse_safe",
                    (
                        f"l23e_frac_lt1={blank['l23e_frac_lt1']:.6f} "
                        f"l23e_p50_hz={blank['l23e_p50_hz']:.6f} "
                        f"l23e_p95_hz={blank['l23e_p95_hz']:.6f} "
                        f"l23e_p99_hz={blank['l23e_p99_hz']:.6f}"
                    ),
                )
                overall_ok &= print_result(
                    blank["l23pv_p99_hz"] <= 75.0
                    and blank["l23som_p99_hz"] <= 75.0,
                    "sensory_blank_interneuron_safe",
                    (
                        f"l23pv_p50_hz={blank['l23pv_p50_hz']:.6f} "
                        f"l23pv_p99_hz={blank['l23pv_p99_hz']:.6f} "
                        f"l23som_p50_hz={blank['l23som_p50_hz']:.6f} "
                        f"l23som_p99_hz={blank['l23som_p99_hz']:.6f} "
                        "p99_limit_hz=75.000000"
                    ),
                )
            else:
                overall_ok = False

            contrast_available = full.contrast_sweep_rows is not None and sensory_enabled == 1.0
            contrast_missing = "none" if full.contrast_sweep_rows is not None else f"{full.prefix}_contrast_sweep.csv"
            overall_ok &= print_result(
                contrast_available,
                "sensory_contrast_artifacts_available",
                (
                    f"sensory_assay_enabled={sensory_enabled:.6f} "
                    f"contrast_rows={len(full.contrast_sweep_rows) if full.contrast_sweep_rows is not None else 0} "
                    f"missing={contrast_missing}"
                ),
            )
            if contrast_available and full.contrast_sweep_rows is not None:
                contrast = compute_contrast_sweep_metrics(full.contrast_sweep_rows)
                overall_ok &= print_result(
                    contrast["contrast_count"] >= 2.0
                    and contrast["l4e_high_mean_hz"] > contrast["l4e_low_mean_hz"]
                    and contrast["l4e_monotonic_fraction"] >= 0.75,
                    "sensory_contrast_l4_monotonic",
                    (
                        f"low_contrast={contrast['low_contrast']:.6f} "
                        f"high_contrast={contrast['high_contrast']:.6f} "
                        f"site_count={int(contrast['l4e_site_count'])} "
                        f"l4e_low_mean_hz={contrast['l4e_low_mean_hz']:.6f} "
                        f"l4e_high_mean_hz={contrast['l4e_high_mean_hz']:.6f} "
                        f"l4e_mean_delta_hz={contrast['l4e_mean_delta_hz']:.6f} "
                        f"l4e_monotonic_fraction={contrast['l4e_monotonic_fraction']:.6f}"
                    ),
                )
                l23e_contrast_delta_floor = max(0.05, 0.02 * contrast["l23e_low_mean_hz"])
                overall_ok &= print_result(
                    contrast["l23e_high_mean_hz"]
                    >= contrast["l23e_low_mean_hz"] + l23e_contrast_delta_floor
                    and contrast["l23e_monotonic_fraction"] >= 0.50
                    and contrast["l23e_high_p99_hz"] <= 100.0,
                    "sensory_contrast_l23e_gain_safe",
                    (
                        f"site_count={int(contrast['l23e_site_count'])} "
                        f"l23e_low_mean_hz={contrast['l23e_low_mean_hz']:.6f} "
                        f"l23e_high_mean_hz={contrast['l23e_high_mean_hz']:.6f} "
                        f"required_delta_hz={l23e_contrast_delta_floor:.6f} "
                        f"l23e_mean_delta_hz={contrast['l23e_mean_delta_hz']:.6f} "
                        f"l23e_monotonic_fraction={contrast['l23e_monotonic_fraction']:.6f} "
                        f"l23e_high_p99_hz={contrast['l23e_high_p99_hz']:.6f} "
                        "p99_limit_hz=100.000000"
                    ),
                )
            else:
                overall_ok = False

            full_annular_rows = full.orientation_context_rows
            somoff_annular_rows = somoff.orientation_context_rows
            annular_artifacts_available = (
                full_annular_rows is not None and somoff_annular_rows is not None
            )
            overall_ok &= print_result(
                annular_artifacts_available,
                "sensory_annular_artifacts_available",
                (
                    f"full_rows={len(full_annular_rows) if full_annular_rows is not None else 0} "
                    f"somoff_rows={len(somoff_annular_rows) if somoff_annular_rows is not None else 0} "
                    f"full_missing={int(full_annular_rows is None)} "
                    f"somoff_missing={int(somoff_annular_rows is None)}"
                ),
            )
            if annular_artifacts_available and full_annular_rows is not None and somoff_annular_rows is not None:
                annular_protocol = compute_annular_protocol_metrics(full_annular_rows)
                full_orientation_context = compute_orientation_context_suppression_metrics(full_annular_rows)
                somoff_orientation_context = compute_orientation_context_suppression_metrics(somoff_annular_rows)
                annular_enabled = full.summary.get("orientation_context_annular_surround_only_enabled", 0.0)
                orientation_context_enabled = full.summary.get("orientation_context_assay_enabled", 0.0)
                protocol_present = (
                    orientation_context_enabled == 1.0
                    and annular_enabled == 1.0
                    and annular_protocol["annular_row_fraction"] >= 1.0
                    and annular_protocol["min_inner_radius_sites"] > 0.0
                    and annular_protocol["min_outer_minus_inner_sites"] > 0.0
                )
                overall_ok &= print_result(
                    protocol_present,
                    "sensory_annular_protocol_present",
                    (
                        f"orientation_context_assay_enabled={orientation_context_enabled:.6f} "
                        f"annular_surround_only_enabled={annular_enabled:.6f} "
                        f"surround_only_rows={int(annular_protocol['surround_only_row_count'])} "
                        f"annular_row_fraction={annular_protocol['annular_row_fraction']:.6f} "
                        f"min_inner_radius_sites={annular_protocol['min_inner_radius_sites']:.6f} "
                        f"min_outer_minus_inner_sites={annular_protocol['min_outer_minus_inner_sites']:.6f}"
                    ),
                )
                annular_surround_rate_guard = max(
                    1.0,
                    0.25 * full_orientation_context["mean_center_l23e_hz"],
                )
                min_driven_sites = max(
                    1,
                    int(math.ceil(0.50 * full_orientation_context["site_count"])),
                )
                overall_ok &= print_result(
                    full_orientation_context["driven_count"] >= min_driven_sites
                    and full_orientation_context["mean_surround_only_l23e_hz"]
                    <= annular_surround_rate_guard,
                    "sensory_annular_surround_only_low",
                    (
                        f"driven={int(full_orientation_context['driven_count'])} "
                        f"required_driven={min_driven_sites} "
                        f"mean_center_l23e_hz={full_orientation_context['mean_center_l23e_hz']:.6f} "
                        f"mean_surround_only_l23e_hz="
                        f"{full_orientation_context['mean_surround_only_l23e_hz']:.6f} "
                        f"mean_surround_only_ratio="
                        f"{full_orientation_context['mean_surround_only_l23e_ratio']:.6f} "
                        f"rate_guard_hz={annular_surround_rate_guard:.6f}"
                    ),
                )
                overall_ok &= print_result(
                    full_orientation_context["driven_count"] >= min_driven_sites
                    and full_orientation_context["mean_si_same_l23e"] >= 0.15
                    and full_orientation_context["mean_osd_l23e"] >= 0.10,
                    "sensory_annular_same_vs_orth_osd",
                    (
                        f"driven={int(full_orientation_context['driven_count'])} "
                        f"required_driven={min_driven_sites} "
                        f"mean_si_same_l23e={full_orientation_context['mean_si_same_l23e']:.6f} "
                        f"median_si_same_l23e={full_orientation_context['median_si_same_l23e']:.6f} "
                        f"mean_osd_l23e={full_orientation_context['mean_osd_l23e']:.6f} "
                        f"median_osd_l23e={full_orientation_context['median_osd_l23e']:.6f} "
                        f"mean_osd_l4e={full_orientation_context['mean_osd_l4e']:.6f}"
                    ),
                )
                osd_reduction = (
                    full_orientation_context["mean_osd_l23e"]
                    - somoff_orientation_context["mean_osd_l23e"]
                )
                same_suppression_reduction = (
                    full_orientation_context["mean_si_same_l23e"]
                    - somoff_orientation_context["mean_si_same_l23e"]
                )
                overall_ok &= print_result(
                    full_orientation_context["driven_count"] >= min_driven_sites
                    and somoff_orientation_context["driven_count"] >= 1.0
                    and (osd_reduction >= 0.05 or same_suppression_reduction >= 0.05),
                    "sensory_annular_som_causality",
                    (
                        f"full_driven={int(full_orientation_context['driven_count'])} "
                        f"somoff_driven={int(somoff_orientation_context['driven_count'])} "
                        f"full_mean_osd_l23e={full_orientation_context['mean_osd_l23e']:.6f} "
                        f"somoff_mean_osd_l23e={somoff_orientation_context['mean_osd_l23e']:.6f} "
                        f"osd_reduction={osd_reduction:.6f} "
                        f"full_mean_si_same_l23e={full_orientation_context['mean_si_same_l23e']:.6f} "
                        f"somoff_mean_si_same_l23e={somoff_orientation_context['mean_si_same_l23e']:.6f} "
                        f"same_suppression_reduction={same_suppression_reduction:.6f}"
                    ),
                )
            else:
                overall_ok = False

        if args.require_responsiveness_sparsity:
            missing_artifacts: list[str] = []
            if full.l23e_cell_tuning is None:
                missing_artifacts.append(f"{full.prefix}_l23e_cell_tuning.csv")
            if full.l23e_cell_tuning_multiphase is None:
                missing_artifacts.append(f"{full.prefix}_l23e_cell_tuning_multiphase.csv")
            artifacts_available = not missing_artifacts
            overall_ok &= print_result(
                artifacts_available,
                "responsiveness_artifacts_available",
                (
                    f"full_cell_tuning={int(full.l23e_cell_tuning is not None)} "
                    f"full_multiphase_tuning={int(full.l23e_cell_tuning_multiphase is not None)} "
                    f"missing={','.join(missing_artifacts) if missing_artifacts else 'none'}"
                ),
            )

            blank_artifact_count = count_blank_or_spontaneous_artifacts(args.genn_dir, full.prefix)
            if blank_artifact_count == 0:
                print(
                    "INFO blank_spontaneous_baseline_missing "
                    "artifact_available=0 hard_fail=0"
                )
            else:
                print(
                    f"INFO blank_spontaneous_baseline artifact_available=1 "
                    f"file_count={blank_artifact_count}"
                )

            if artifacts_available:
                assert full.l23e_cell_tuning is not None
                assert full.l23e_cell_tuning_multiphase is not None
                single_5hz = compute_cell_responsive_metrics(full.l23e_cell_tuning, 5.0)
                single_10hz = compute_cell_responsive_metrics(full.l23e_cell_tuning, 10.0)
                multiphase_5hz = compute_multiphase_cell_responsive_metrics(
                    full.l23e_cell_tuning_multiphase,
                    5.0,
                )
                multiphase_10hz = compute_multiphase_cell_responsive_metrics(
                    full.l23e_cell_tuning_multiphase,
                    10.0,
                )
                phase_mean_peak_values = [
                    max(row.phase_mean_rates_by_deg.values())
                    for row in full.l23e_cell_tuning_multiphase.values()
                ]
                print(
                    "INFO phase_mean_responsiveness "
                    f"cell_count={len(phase_mean_peak_values)} "
                    f"peak_ge5_fraction={fraction_at_least(phase_mean_peak_values, 5.0):.6f} "
                    f"peak_ge10_fraction={fraction_at_least(phase_mean_peak_values, 10.0):.6f} "
                    f"median_phase_mean_peak_hz={median(phase_mean_peak_values):.6f}"
                )
                print(
                    "INFO l23e_cell_peak10_responsiveness "
                    f"peak_ge10_fraction={single_10hz.responsive_fraction:.6f} "
                    f"peak_ge10_cells={single_10hz.responsive_cells} "
                    f"total_cells={single_10hz.total_cells}"
                )

                overall_ok &= print_result(
                    0.10 <= single_5hz.responsive_fraction <= 0.45,
                    "l23e_cell_sparse_responsiveness",
                    (
                        f"peak_ge5_fraction={single_5hz.responsive_fraction:.6f} "
                        f"peak_ge5_bounds=[0.100000,0.450000] "
                        f"peak_ge10_fraction={single_10hz.responsive_fraction:.6f} "
                        f"peak_ge10_cells={single_10hz.responsive_cells} "
                        f"total_cells={single_5hz.total_cells}"
                    ),
                )

                multiphase_osi_ok = (
                    multiphase_5hz.responsive_median_phase_pooled_osi is not None
                    and multiphase_5hz.responsive_median_phase_pooled_osi >= 0.65
                )
                overall_ok &= print_result(
                    0.15 <= multiphase_5hz.responsive_fraction <= 0.55
                    and multiphase_10hz.responsive_fraction >= 0.15
                    and multiphase_osi_ok,
                    "l23e_cell_multiphase_sparse_responsiveness",
                    (
                        f"peak_any_phase_ge5_fraction={multiphase_5hz.responsive_fraction:.6f} "
                        f"peak_any_phase_ge5_bounds=[0.150000,0.550000] "
                        f"peak_any_phase_ge10_fraction={multiphase_10hz.responsive_fraction:.6f} "
                        f"responsive_median_phase_pooled_osi="
                        f"{format_optional_float(multiphase_5hz.responsive_median_phase_pooled_osi)} "
                        f"total_cells={multiphase_5hz.total_cells}"
                    ),
                )

                overall_ok &= print_result(
                    multiphase_5hz.responsive_site_fraction_ge1 >= 0.85
                    and multiphase_5hz.responsive_site_fraction_ge2 >= 0.65
                    and single_5hz.responsive_site_fraction >= 0.70,
                    "l23e_responsive_site_coverage",
                    (
                        f"multiphase_sites_ge1_fraction="
                        f"{multiphase_5hz.responsive_site_fraction_ge1:.6f} "
                        f"multiphase_sites_ge2_fraction="
                        f"{multiphase_5hz.responsive_site_fraction_ge2:.6f} "
                        f"single_phase_site_fraction={single_5hz.responsive_site_fraction:.6f} "
                        f"total_multiphase_sites={multiphase_5hz.total_sites} "
                        f"total_single_phase_sites={single_5hz.total_sites}"
                    ),
                )

                full_l23e_rate_metrics = compute_rate_metrics(full.post_site_rates["l23e"])
                overall_ok &= print_result(
                    full_l23e_rate_metrics.frac_below_1hz >= 0.85
                    and full_l23e_rate_metrics.p99_hz <= 5.0,
                    "l23e_population_sparse_rates",
                    (
                        f"frac_lt1={full_l23e_rate_metrics.frac_below_1hz:.6f} "
                        f"frac_lt1_min=0.850000 "
                        f"p99={full_l23e_rate_metrics.p99_hz:.6f} "
                        f"p99_limit=5.000000"
                    ),
                )

                spatial_balance = responsiveness_spatial_balance_metrics(
                    full.l23e_cell_tuning_multiphase,
                    full.l23e_post_sites,
                    5.0,
                )
                overall_ok &= print_result(
                    spatial_balance["responsive_site_count"] > 0.0
                    and spatial_balance["responsive_cell_count"] > 0.0
                    and spatial_balance["zero_site_quadrants"] == 0.0
                    and spatial_balance["zero_cell_quadrants"] == 0.0
                    and spatial_balance["min_quadrant_site_fraction"] >= 0.10,
                    "l23e_spatial_coverage_balance",
                    (
                        f"responsive_sites={int(spatial_balance['responsive_site_count'])} "
                        f"responsive_cells={int(spatial_balance['responsive_cell_count'])} "
                        f"min_quadrant_site_fraction="
                        f"{spatial_balance['min_quadrant_site_fraction']:.6f} "
                        f"min_quadrant_cell_fraction="
                        f"{spatial_balance['min_quadrant_cell_fraction']:.6f} "
                        f"zero_site_quadrants={int(spatial_balance['zero_site_quadrants'])} "
                        f"zero_cell_quadrants={int(spatial_balance['zero_cell_quadrants'])} "
                        f"used_post_site_coordinates="
                        f"{int(spatial_balance['used_post_site_coordinates'])} "
                        f"left_lower_sites={int(spatial_balance['left_lower_sites'])} "
                        f"left_upper_sites={int(spatial_balance['left_upper_sites'])} "
                        f"right_lower_sites={int(spatial_balance['right_lower_sites'])} "
                        f"right_upper_sites={int(spatial_balance['right_upper_sites'])} "
                        f"left_lower_cells={int(spatial_balance['left_lower_cells'])} "
                        f"left_upper_cells={int(spatial_balance['left_upper_cells'])} "
                        f"right_lower_cells={int(spatial_balance['right_lower_cells'])} "
                        f"right_upper_cells={int(spatial_balance['right_upper_cells'])}"
                    ),
                )

        if args.require_scaling_map_consistency:
            l4is_paths = {
                "full": args.genn_dir / f"{args.full}_l4_intersite_diagnostics.csv",
                "control": args.genn_dir / f"{args.control}_l4_intersite_diagnostics.csv",
                "somoff": args.genn_dir / f"{args.somoff}_l4_intersite_diagnostics.csv",
            }
            missing_scaling_artifacts: list[str] = []
            if full.l4_post_sites is None:
                missing_scaling_artifacts.append(f"{full.prefix}_post_l4_sites.csv")
            if control.l4_post_sites is None:
                missing_scaling_artifacts.append(f"{control.prefix}_post_l4_sites.csv")
            if not post_site_preferences_available(full.l4_post_sites):
                missing_scaling_artifacts.append(f"{full.prefix}_post_l4_sites.csv:map_pref/measured_pref")
            if not post_site_preferences_available(control.l4_post_sites):
                missing_scaling_artifacts.append(f"{control.prefix}_post_l4_sites.csv:map_pref/measured_pref")
            if not post_site_preferences_available(full.l23e_post_sites):
                missing_scaling_artifacts.append(f"{full.prefix}_post_l23_sites.csv:map_pref/measured_pref")
            if not post_site_preferences_available(control.l23e_post_sites):
                missing_scaling_artifacts.append(f"{control.prefix}_post_l23_sites.csv:map_pref/measured_pref")
            if full.l23e_cell_tuning_multiphase is None:
                missing_scaling_artifacts.append(f"{full.prefix}_l23e_cell_tuning_multiphase.csv")
            for label, path in l4is_paths.items():
                if not path.is_file():
                    missing_scaling_artifacts.append(f"{label}:{path.name}")

            scaling_artifacts_available = not missing_scaling_artifacts
            overall_ok &= print_result(
                scaling_artifacts_available,
                "scaling_map_artifacts_available",
                f"missing={','.join(missing_scaling_artifacts) if missing_scaling_artifacts else 'none'}",
            )

            if scaling_artifacts_available:
                assert full.l4_post_sites is not None
                assert control.l4_post_sites is not None
                assert full.l23e_cell_tuning_multiphase is not None
                full_l4_intersite_for_scaling = load_l4_intersite_metrics(args.genn_dir, args.full)
                control_l4_intersite_for_scaling = load_l4_intersite_metrics(args.genn_dir, args.control)
                somoff_l4_intersite_for_scaling = load_l4_intersite_metrics(args.genn_dir, args.somoff)

                l4_map = compute_l4_map_consistency_metrics(full.l4_post_sites)
                l4is_post_map_error = full_l4_intersite_for_scaling.get("post_l4_map_error_deg_median", math.nan)
                l4is_baseline_map_error = full_l4_intersite_for_scaling.get(
                    "baseline_l4_map_error_deg_median",
                    math.nan,
                )
                overall_ok &= print_result(
                    l4_map["active_fraction"] >= 0.95
                    and l4_map["median_error_deg"] <= 5.0
                    and l4_map["p90_error_deg"] <= 10.0,
                    "scaling_l4_map_consistency",
                    (
                        f"active_sites={int(l4_map['active_sites'])} "
                        f"total_sites={int(l4_map['total_sites'])} "
                        f"active_fraction={l4_map['active_fraction']:.6f} "
                        f"median_map_error_deg={l4_map['median_error_deg']:.6f} "
                        f"p90_map_error_deg={l4_map['p90_error_deg']:.6f} "
                        f"summary_post_l4_map_error_deg="
                        f"{require_summary_metric(full, 'post_l4_map_error_deg_median'):.6f} "
                        f"l4is_post_l4_map_error_deg={l4is_post_map_error:.6f} "
                        f"l4is_baseline_l4_map_error_deg={l4is_baseline_map_error:.6f}"
                    ),
                )

                l23_l4_map = compute_l23_l4_map_consistency_metrics(
                    full.l23e_post_sites,
                    full.l4_post_sites,
                    full.l23e_cell_tuning_multiphase,
                )
                active_site_map_ok = (
                    l23_l4_map["active_site_count"] > 0.0
                    and l23_l4_map["active_site_median_delta_deg"] <= 25.0
                )
                cell10_map_ok = (
                    l23_l4_map["cell10_count"] > 0.0
                    and l23_l4_map["cell10_median_delta_deg"] <= 22.5
                )
                overall_ok &= print_result(
                    active_site_map_ok or cell10_map_ok,
                    "scaling_l23_l4_map_consistency",
                    (
                        f"active_site_count={int(l23_l4_map['active_site_count'])} "
                        f"active_site_median_delta_deg="
                        f"{l23_l4_map['active_site_median_delta_deg']:.6f} "
                        f"active_site_p90_delta_deg={l23_l4_map['active_site_p90_delta_deg']:.6f} "
                        f"cell5_count={int(l23_l4_map['cell5_count'])} "
                        f"cell5_median_delta_deg={l23_l4_map['cell5_median_delta_deg']:.6f} "
                        f"cell10_count={int(l23_l4_map['cell10_count'])} "
                        f"cell10_median_delta_deg={l23_l4_map['cell10_median_delta_deg']:.6f} "
                        f"active_site_gate={int(active_site_map_ok)} "
                        f"cell10_gate={int(cell10_map_ok)}"
                    ),
                )

                tile5 = compute_tile_orientation_metrics(
                    full.l23e_cell_tuning_multiphase,
                    full.l23e_post_sites,
                    5.0,
                )
                tile10 = compute_tile_orientation_metrics(
                    full.l23e_cell_tuning_multiphase,
                    full.l23e_post_sites,
                    10.0,
                )
                overall_ok &= print_result(
                    tile5["nonempty_tile_count"] == 16.0
                    and tile5["global_occupied_bins"] == 12.0
                    and tile5["bin_gate_pass"] == 1.0,
                    "scaling_tile_orientation_coverage",
                    (
                        f"threshold_hz={tile5['threshold_hz']:.6f} "
                        f"responsive_cells={int(tile5['responsive_cells'])} "
                        f"nonempty_tiles={int(tile5['nonempty_tile_count'])} "
                        f"global_occupied_bins={int(tile5['global_occupied_bins'])}/12 "
                        f"min_tile_cell_count={int(tile5['min_tile_cell_count'])} "
                        f"max_tile_cell_count={int(tile5['max_tile_cell_count'])} "
                        f"low_count_tile_count={int(tile5['low_count_tile_count'])} "
                        f"min_occupied_bins={int(tile5['min_occupied_bins'])} "
                        f"median_occupied_bins={tile5['median_occupied_bins']:.6f} "
                        f"bin_gate_pass={int(tile5['bin_gate_pass'])} "
                        f"used_post_site_coordinates={int(tile5['used_post_site_coordinates'])}"
                    ),
                )
                overall_ok &= print_result(
                    tile5["median_entropy"] >= 0.70
                    and tile5["min_entropy"] >= 0.55,
                    "scaling_tile_orientation_entropy",
                    (
                        f"threshold5_median_entropy={tile5['median_entropy']:.6f} "
                        f"threshold5_min_entropy={tile5['min_entropy']:.6f} "
                        f"threshold10_responsive_cells={int(tile10['responsive_cells'])} "
                        f"threshold10_median_entropy={tile10['median_entropy']:.6f} "
                        f"threshold10_min_entropy={tile10['min_entropy']:.6f}"
                    ),
                )

                edge_quadrants = compute_edge_quadrant_balance_metrics(
                    full.l23e_cell_tuning_multiphase,
                    full.l23e_post_sites,
                    5.0,
                )
                overall_ok &= print_result(
                    edge_quadrants["edge_site_coverage"] >= 0.60
                    and edge_quadrants["zero_quadrants"] == 0.0
                    and edge_quadrants["min_quadrant_cell_fraction"] >= 0.10,
                    "scaling_edge_quadrant_balance",
                    (
                        f"responsive_cells={int(edge_quadrants['responsive_cells'])} "
                        f"responsive_sites={int(edge_quadrants['responsive_sites'])} "
                        f"responsive_edge_sites={int(edge_quadrants['responsive_edge_sites'])} "
                        f"edge_sites={int(edge_quadrants['edge_sites'])} "
                        f"edge_site_coverage={edge_quadrants['edge_site_coverage']:.6f} "
                        f"min_quadrant_cell_fraction="
                        f"{edge_quadrants['min_quadrant_cell_fraction']:.6f} "
                        f"zero_quadrants={int(edge_quadrants['zero_quadrants'])} "
                        f"used_post_site_coordinates={int(edge_quadrants['used_post_site_coordinates'])} "
                        f"left_lower_cells={int(edge_quadrants['left_lower_cells'])} "
                        f"left_upper_cells={int(edge_quadrants['left_upper_cells'])} "
                        f"right_lower_cells={int(edge_quadrants['right_lower_cells'])} "
                        f"right_upper_cells={int(edge_quadrants['right_upper_cells'])}"
                    ),
                )

                enabled_values = {
                    "full": require_metric(full_l4_intersite_for_scaling, "enabled", args.full),
                    "control": require_metric(control_l4_intersite_for_scaling, "enabled", args.control),
                    "somoff": require_metric(somoff_l4_intersite_for_scaling, "enabled", args.somoff),
                }
                radius_sites = require_metric(full_l4_intersite_for_scaling, "radius_sites", args.full)
                weight_scale = require_metric(full_l4_intersite_for_scaling, "weight_scale", args.full)
                l4ee_scale = optional_metric(full_l4_intersite_for_scaling, "l4ee_scale", weight_scale)
                l4e_to_l4pv_scale = optional_metric(
                    full_l4_intersite_for_scaling,
                    "l4e_to_l4pv_scale",
                    weight_scale,
                )
                l4pv_to_l4e_scale = optional_metric(
                    full_l4_intersite_for_scaling,
                    "l4pv_to_l4e_scale",
                    weight_scale,
                )
                max_projection_scale = max(l4ee_scale, l4e_to_l4pv_scale, l4pv_to_l4e_scale)
                edge_counts = {
                    "l4ee": require_metric(full_l4_intersite_for_scaling, "l4ee_edge_count", args.full),
                    "l4e_to_l4pv": require_metric(
                        full_l4_intersite_for_scaling,
                        "l4e_to_l4pv_edge_count",
                        args.full,
                    ),
                    "l4pv_to_l4e": require_metric(
                        full_l4_intersite_for_scaling,
                        "l4pv_to_l4e_edge_count",
                        args.full,
                    ),
                }
                max_distance_sites = require_metric(
                    full_l4_intersite_for_scaling,
                    "max_projection_distance_sites",
                    args.full,
                )
                max_same_site_fraction = require_metric(
                    full_l4_intersite_for_scaling,
                    "max_same_site_fraction",
                    args.full,
                )
                max_beyond_radius_fraction = require_metric(
                    full_l4_intersite_for_scaling,
                    "max_beyond_radius_fraction",
                    args.full,
                )
                post_l4_osi = require_summary_metric(full, "post_l4_median_osi")
                control_post_l4_osi = require_summary_metric(control, "post_l4_median_osi")
                post_l4_map_error_deg = require_summary_metric(full, "post_l4_map_error_deg_median")
                control_post_l4_map_error_deg = require_summary_metric(control, "post_l4_map_error_deg_median")
                allowed_osi_drop = max(0.02, 0.05 * control_post_l4_osi)
                osi_drop = control_post_l4_osi - post_l4_osi
                map_error_delta = post_l4_map_error_deg - control_post_l4_map_error_deg
                l4_peak_rate = require_metric(full_l4_intersite_for_scaling, "l4_size_peak_rate_hz", args.full)
                l4_small_peak_ratio = require_metric(
                    full_l4_intersite_for_scaling,
                    "l4_size_small_peak_ratio",
                    args.full,
                )
                l4_large_peak_ratio = require_metric(
                    full_l4_intersite_for_scaling,
                    "l4_size_large_peak_ratio",
                    args.full,
                )
                l4is_enabled_ok = all(value == 1.0 for value in enabled_values.values())
                l4is_connectivity_ok = (
                    radius_sites >= 2.0
                    and radius_sites <= 3.0
                    and 0.0 < max_projection_scale <= 0.20
                    and all(count > 0.0 for count in edge_counts.values())
                    and max_distance_sites <= ((2.0 ** 0.5) * radius_sites + 1.0e-6)
                    and max_same_site_fraction <= 1.0e-9
                    and max_beyond_radius_fraction <= 1.0e-9
                )
                l4is_map_ok = (
                    osi_drop <= allowed_osi_drop
                    and post_l4_map_error_deg <= 45.0
                    and map_error_delta <= 5.0
                )
                l4is_spread_ok = (
                    l4_peak_rate > 0.0
                    and 0.0 <= l4_small_peak_ratio <= 1.05
                    and 0.0 <= l4_large_peak_ratio <= 1.20
                )
                overall_ok &= print_result(
                    l4is_enabled_ok and l4is_connectivity_ok and l4is_map_ok and l4is_spread_ok,
                    "scaling_l4is_preservation",
                    (
                        f"enabled_full={enabled_values['full']:.0f} "
                        f"enabled_control={enabled_values['control']:.0f} "
                        f"enabled_somoff={enabled_values['somoff']:.0f} "
                        f"enabled_ok={int(l4is_enabled_ok)} "
                        f"connectivity_ok={int(l4is_connectivity_ok)} "
                        f"map_ok={int(l4is_map_ok)} "
                        f"spread_ok={int(l4is_spread_ok)} "
                        f"radius_sites={radius_sites:.6f} "
                        f"max_projection_scale={max_projection_scale:.6f} "
                        f"max_distance_sites={max_distance_sites:.6f} "
                        f"max_same_site_fraction={max_same_site_fraction:.6f} "
                        f"max_beyond_radius_fraction={max_beyond_radius_fraction:.6f} "
                        f"osi_drop={osi_drop:.6f} "
                        f"allowed_osi_drop={allowed_osi_drop:.6f} "
                        f"post_l4_map_error_deg={post_l4_map_error_deg:.6f} "
                        f"control_post_l4_map_error_deg={control_post_l4_map_error_deg:.6f} "
                        f"map_error_delta={map_error_delta:.6f} "
                        f"l4_peak_rate_hz={l4_peak_rate:.6f} "
                        f"small_peak_ratio={l4_small_peak_ratio:.6f} "
                        f"large_peak_ratio={l4_large_peak_ratio:.6f}"
                    ),
                )

        full_context_site_count = len(full.context_rows_by_site)
        somoff_context_site_count = len(somoff.context_rows_by_site)
        full_size_site_count = size_validation_site_count(full)
        somoff_size_site_count = size_validation_site_count(somoff)
        min_observed_validation_sites = min(
            full_context_site_count,
            somoff_context_site_count,
            full_size_site_count,
            somoff_size_site_count,
        )
        overall_ok &= print_result(
            min_observed_validation_sites >= args.min_validation_sites,
            "validation_sites",
            (
                f"required={args.min_validation_sites} "
                f"full_context={full_context_site_count} "
                f"somoff_context={somoff_context_site_count} "
                f"full_size={full_size_site_count} "
                f"somoff_size={somoff_size_site_count}"
            ),
        )

        if args.require_pv_gain_normalization:
            if pvweak is None:
                raise ValidationError("--require-pv-gain-normalization requires loaded --pvweak data.")
            pv_metrics = compute_pv_gain_normalization_metrics(full, pvweak)
            pv_scale_is_half = abs(pv_metrics["pvweak_scale"] - 0.5) <= 1.0e-6
            gain_floor = 0.20 if pv_scale_is_half else 0.10
            gain_signal = max(
                pv_metrics["mean_increase_fraction"],
                pv_metrics["median_increase_fraction"],
            )
            gain_upper_ok = (
                (not pv_scale_is_half)
                or (
                    pv_metrics["mean_increase_fraction"] <= 1.50
                    and pv_metrics["median_increase_fraction"] <= 1.50
                )
            )
            overall_ok &= print_result(
                pv_metrics["pvweak_active"] == 1.0
                and pv_metrics["pvweak_scale"] < 1.0
                and pv_metrics["driven_rate_count"] > 0.0
                and gain_signal >= gain_floor
                and gain_upper_ok
                and pv_metrics["pvweak_l23e_context_p99_hz"] <= pv_metrics["l23e_p99_limit_hz"],
                "pv_gain_normalization_causality",
                (
                    f"pvweak_scale={pv_metrics['pvweak_scale']:.6f} "
                    f"pvweak_active={pv_metrics['pvweak_active']:.0f} "
                    f"site_count={int(pv_metrics['site_count'])} "
                    f"driven_rate_count={int(pv_metrics['driven_rate_count'])} "
                    f"full_mean_l23e_hz={pv_metrics['full_mean_l23e_hz']:.6f} "
                    f"pvweak_mean_l23e_hz={pv_metrics['pvweak_mean_l23e_hz']:.6f} "
                    f"mean_increase_fraction={pv_metrics['mean_increase_fraction']:.6f} "
                    f"full_median_l23e_hz={pv_metrics['full_median_l23e_hz']:.6f} "
                    f"pvweak_median_l23e_hz={pv_metrics['pvweak_median_l23e_hz']:.6f} "
                    f"median_increase_fraction={pv_metrics['median_increase_fraction']:.6f} "
                    f"required_gain_floor={gain_floor:.6f} "
                    f"scale_half_upper_bound_applies={int(pv_scale_is_half)} "
                    f"pvweak_l23e_context_p99_hz={pv_metrics['pvweak_l23e_context_p99_hz']:.6f} "
                    f"p99_limit_hz={pv_metrics['l23e_p99_limit_hz']:.6f}"
                ),
            )
            overall_ok &= print_result(
                pv_metrics["responsive_site_count"] > 0.0
                and pv_metrics["median_osi_drop"] <= 0.15
                and pv_metrics["median_pref_shift_deg"] <= 22.5,
                "pv_gain_normalization_selectivity_safety",
                (
                    f"responsive_site_count={int(pv_metrics['responsive_site_count'])} "
                    f"full_median_osi={pv_metrics['full_median_osi']:.6f} "
                    f"pvweak_median_osi={pv_metrics['pvweak_median_osi']:.6f} "
                    f"median_osi_drop={pv_metrics['median_osi_drop']:.6f} "
                    f"median_pref_shift_deg={pv_metrics['median_pref_shift_deg']:.6f} "
                    f"max_pref_shift_deg={pv_metrics['max_pref_shift_deg']:.6f}"
                ),
            )
            overall_ok &= print_result(
                pv_metrics["full_l23pv_post_median_hz"] >= 1.0
                and pv_metrics["full_l23pv_post_frac_lt1"] < 0.25
                and pv_metrics["full_l23pv_post_p99_hz"] <= pv_metrics["full_l23pv_post_p99_limit_hz"],
                "pv_gain_normalization_rates",
                (
                    f"full_l23pv_post_median_hz={pv_metrics['full_l23pv_post_median_hz']:.6f} "
                    f"full_l23pv_post_frac_lt1={pv_metrics['full_l23pv_post_frac_lt1']:.6f} "
                    f"full_l23pv_post_p99_hz={pv_metrics['full_l23pv_post_p99_hz']:.6f} "
                    f"p99_limit_hz={pv_metrics['full_l23pv_post_p99_limit_hz']:.6f}"
                ),
            )

        if args.require_emergent_l23_orientation_suppression:
            ff_bias_strength = require_summary_metric(full, "l4_l23_orientation_bias_strength")
            ff_prior_enabled = require_summary_metric(full, "l4_l23_feedforward_orientation_prior_enabled")
            inhibitory_orientation_rule_enabled = require_summary_metric(full, "inhibitory_orientation_rule_enabled")
            orientation_context_enabled = require_summary_metric(full, "orientation_context_assay_enabled")
            overall_ok &= print_result(
                abs(ff_bias_strength) <= 1.0e-12
                and ff_prior_enabled == 0.0
                and inhibitory_orientation_rule_enabled == 0.0,
                "no_hardcode_audit",
                (
                    f"l4_l23_orientation_bias_strength={ff_bias_strength:.6f} "
                    f"feedforward_orientation_prior_enabled={ff_prior_enabled:.6f} "
                    f"inhibitory_orientation_rule_enabled={inhibitory_orientation_rule_enabled:.6f}"
                ),
            )
            overall_ok &= print_result(
                orientation_context_enabled == 1.0,
                "l23_orientation_context_assay_enabled",
                f"orientation_context_assay_enabled={orientation_context_enabled:.6f}",
            )

            full_orientation_context = compute_orientation_context_suppression_metrics(
                require_orientation_context_rows(full)
            )
            control_orientation_context = compute_orientation_context_suppression_metrics(
                require_orientation_context_rows(control)
            )
            min_driven_sites = max(
                1,
                int(math.ceil(0.50 * full_orientation_context["site_count"])),
            )
            full_control_osd_delta = (
                full_orientation_context["mean_osd_l23e"]
                - control_orientation_context["mean_osd_l23e"]
            )
            surround_only_rate_guard = max(
                1.0,
                0.25 * full_orientation_context["mean_center_l23e_hz"],
            )
            overall_ok &= print_result(
                full_orientation_context["driven_count"] >= min_driven_sites,
                "l23_orientation_context_driven_sites",
                (
                    f"driven={int(full_orientation_context['driven_count'])} "
                    f"required={min_driven_sites} "
                    f"site_count={int(full_orientation_context['site_count'])} "
                    f"threshold_hz={full_orientation_context['driven_threshold_hz']:.6f} "
                    f"mean_center_l23e_hz={full_orientation_context['mean_center_l23e_hz']:.6f}"
                ),
            )
            overall_ok &= print_result(
                full_orientation_context["mean_si_same_l23e"] >= 0.15,
                "l23_orientation_context_same_suppression",
                (
                    f"mean_si_same_l23e={full_orientation_context['mean_si_same_l23e']:.6f} "
                    f"median_si_same_l23e={full_orientation_context['median_si_same_l23e']:.6f}"
                ),
            )
            overall_ok &= print_result(
                full_orientation_context["mean_osd_l23e"] >= 0.10,
                "l23_orientation_context_osd",
                (
                    f"mean_osd_l23e={full_orientation_context['mean_osd_l23e']:.6f} "
                    f"median_osd_l23e={full_orientation_context['median_osd_l23e']:.6f} "
                    f"control_mean_osd_l23e={control_orientation_context['mean_osd_l23e']:.6f}"
                ),
            )
            overall_ok &= print_result(
                full_orientation_context["frac_osd_gt_0p05"] >= 0.65,
                "l23_orientation_context_osd_site_fraction",
                f"frac_osd_gt_0p05={full_orientation_context['frac_osd_gt_0p05']:.6f}",
            )
            overall_ok &= print_result(
                full_orientation_context["mean_surround_only_l23e_hz"] <= surround_only_rate_guard,
                "l23_orientation_context_surround_only_low",
                (
                    f"mean_surround_only_l23e_hz="
                    f"{full_orientation_context['mean_surround_only_l23e_hz']:.6f} "
                    f"mean_surround_only_ratio="
                    f"{full_orientation_context['mean_surround_only_l23e_ratio']:.6f} "
                    f"max_surround_only_ratio="
                    f"{full_orientation_context['max_surround_only_l23e_ratio']:.6f} "
                    f"rate_guard_hz={surround_only_rate_guard:.6f}"
                ),
            )
            overall_ok &= print_result(
                full_orientation_context["mean_osd_l23e_minus_l4e"] >= 0.05,
                "l23_orientation_context_l23_minus_l4",
                (
                    f"mean_osd_l23e={full_orientation_context['mean_osd_l23e']:.6f} "
                    f"mean_osd_l4e={full_orientation_context['mean_osd_l4e']:.6f} "
                    f"delta={full_orientation_context['mean_osd_l23e_minus_l4e']:.6f}"
                ),
            )
            overall_ok &= print_result(
                full_control_osd_delta >= -0.02,
                "l23_orientation_context_full_control_delta",
                (
                    f"full_mean_osd_l23e={full_orientation_context['mean_osd_l23e']:.6f} "
                    f"control_mean_osd_l23e={control_orientation_context['mean_osd_l23e']:.6f} "
                    f"delta={full_control_osd_delta:.6f} "
                    "plasticity_enhancement_informational=1 "
                    "minimum_safety_delta=-0.020000"
                ),
            )
            if somoff.orientation_context_rows is not None:
                somoff_orientation_context = compute_orientation_context_suppression_metrics(
                    somoff.orientation_context_rows
                )
                print(
                    "INFO l23_orientation_context_somoff "
                    f"mean_osd_l23e={somoff_orientation_context['mean_osd_l23e']:.6f} "
                    f"mean_si_same_l23e={somoff_orientation_context['mean_si_same_l23e']:.6f} "
                    f"driven={int(somoff_orientation_context['driven_count'])}"
                )

        if args.require_l4_intersite:
            full_l4_intersite = load_l4_intersite_metrics(args.genn_dir, args.full)
            control_l4_intersite = load_l4_intersite_metrics(args.genn_dir, args.control)
            somoff_l4_intersite = load_l4_intersite_metrics(args.genn_dir, args.somoff)
            enabled_values = {
                "full": require_metric(full_l4_intersite, "enabled", args.full),
                "control": require_metric(control_l4_intersite, "enabled", args.control),
                "somoff": require_metric(somoff_l4_intersite, "enabled", args.somoff),
            }
            overall_ok &= print_result(
                all(value == 1.0 for value in enabled_values.values()),
                "l4_intersite_enabled",
                " ".join(f"{label}={value:.0f}" for label, value in enabled_values.items()),
            )

            radius_sites = require_metric(full_l4_intersite, "radius_sites", args.full)
            weight_scale = require_metric(full_l4_intersite, "weight_scale", args.full)
            l4ee_scale = optional_metric(full_l4_intersite, "l4ee_scale", weight_scale)
            l4e_to_l4pv_scale = optional_metric(full_l4_intersite, "l4e_to_l4pv_scale", weight_scale)
            l4pv_to_l4e_scale = optional_metric(full_l4_intersite, "l4pv_to_l4e_scale", weight_scale)
            max_projection_scale = max(l4ee_scale, l4e_to_l4pv_scale, l4pv_to_l4e_scale)
            edge_counts = {
                "l4ee": require_metric(full_l4_intersite, "l4ee_edge_count", args.full),
                "l4e_to_l4pv": require_metric(full_l4_intersite, "l4e_to_l4pv_edge_count", args.full),
                "l4pv_to_l4e": require_metric(full_l4_intersite, "l4pv_to_l4e_edge_count", args.full),
            }
            max_distance_sites = require_metric(full_l4_intersite, "max_projection_distance_sites", args.full)
            max_same_site_fraction = require_metric(full_l4_intersite, "max_same_site_fraction", args.full)
            max_beyond_radius_fraction = require_metric(full_l4_intersite, "max_beyond_radius_fraction", args.full)
            overall_ok &= print_result(
                radius_sites >= 2.0
                and radius_sites <= 3.0
                and 0.0 < max_projection_scale <= 0.20
                and all(count > 0.0 for count in edge_counts.values())
                and max_distance_sites <= ((2.0 ** 0.5) * radius_sites + 1.0e-6)
                and max_same_site_fraction <= 1.0e-9
                and max_beyond_radius_fraction <= 1.0e-9,
                "l4_intersite_connectivity",
                (
                    f"radius_sites={radius_sites:.6f} weight_scale={weight_scale:.6f} "
                    f"l4ee_scale={l4ee_scale:.6f} "
                    f"l4e_to_l4pv_scale={l4e_to_l4pv_scale:.6f} "
                    f"l4pv_to_l4e_scale={l4pv_to_l4e_scale:.6f} "
                    f"l4ee_edges={int(edge_counts['l4ee'])} "
                    f"l4e_to_l4pv_edges={int(edge_counts['l4e_to_l4pv'])} "
                    f"l4pv_to_l4e_edges={int(edge_counts['l4pv_to_l4e'])} "
                    f"max_distance_sites={max_distance_sites:.6f} "
                    f"max_same_site_fraction={max_same_site_fraction:.6f} "
                    f"max_beyond_radius_fraction={max_beyond_radius_fraction:.6f}"
                ),
            )

            post_l4_osi = require_summary_metric(full, "post_l4_median_osi")
            control_post_l4_osi = require_summary_metric(control, "post_l4_median_osi")
            post_l4_map_error_deg = require_summary_metric(full, "post_l4_map_error_deg_median")
            control_post_l4_map_error_deg = require_summary_metric(control, "post_l4_map_error_deg_median")
            allowed_osi_drop = max(0.02, 0.05 * control_post_l4_osi)
            osi_drop = control_post_l4_osi - post_l4_osi
            map_error_delta = post_l4_map_error_deg - control_post_l4_map_error_deg
            overall_ok &= print_result(
                osi_drop <= allowed_osi_drop
                and post_l4_map_error_deg <= 45.0
                and map_error_delta <= 5.0,
                "l4_intersite_map_preservation",
                (
                    f"full_post_l4_median_osi={post_l4_osi:.6f} "
                    f"control_post_l4_median_osi={control_post_l4_osi:.6f} "
                    f"osi_drop={osi_drop:.6f} allowed_osi_drop={allowed_osi_drop:.6f} "
                    f"full_post_l4_map_error_deg={post_l4_map_error_deg:.6f} "
                    f"control_post_l4_map_error_deg={control_post_l4_map_error_deg:.6f} "
                    f"map_error_delta={map_error_delta:.6f}"
                ),
            )

            l4_peak_rate = require_metric(full_l4_intersite, "l4_size_peak_rate_hz", args.full)
            l4_small_peak_ratio = require_metric(full_l4_intersite, "l4_size_small_peak_ratio", args.full)
            l4_large_peak_ratio = require_metric(full_l4_intersite, "l4_size_large_peak_ratio", args.full)
            overall_ok &= print_result(
                l4_peak_rate > 0.0
                and 0.0 <= l4_small_peak_ratio <= 1.05
                and 0.0 <= l4_large_peak_ratio <= 1.20,
                "l4_intersite_spread_bounded",
                (
                    f"l4_peak_rate_hz={l4_peak_rate:.6f} "
                    f"small_peak_ratio={l4_small_peak_ratio:.6f} "
                    f"large_peak_ratio={l4_large_peak_ratio:.6f}"
                ),
            )

        full_post_osi = require_summary_metric(full, "post_l23_median_osi")
        control_post_osi = require_summary_metric(control, "post_l23_median_osi")
        osi_delta = full_post_osi - control_post_osi
        strict_osi_ok = full_post_osi >= 0.70 and osi_delta >= 0.10
        printed_strict_osi_ok = print_result(
            strict_osi_ok,
            "osi",
            f"full_post={full_post_osi:.6f} control_post={control_post_osi:.6f} delta={osi_delta:.6f}",
        )
        if strict_osi_ok or not args.allow_responsive_osi:
            overall_ok &= printed_strict_osi_ok
        else:
            print(
                "INFO osi_responsive_rescue pending "
                "strict_all_site_pass=0 strict_all_site_gate_deferred=1"
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

        preferred_by_site, preferred_rates_by_site = preferred_center_orientations(full)
        primary_validation_site_id = next(iter(preferred_by_site))
        pref_deg = preferred_by_site[primary_validation_site_id]
        full_center_pref_rate = mean(list(preferred_rates_by_site.values()))
        full_min_center_pref_rate = min(preferred_rates_by_site.values())
        full_context = compute_context_metrics(full, preferred_by_site)
        somoff_context = compute_context_metrics(somoff, preferred_by_site)

        overall_ok &= print_result(
            full_center_pref_rate >= 5.0,
            "som_center_pref",
            (
                f"validation_sites={int(full_context['validation_site_count'])} "
                f"primary_site={primary_validation_site_id} "
                f"preferred_deg={pref_deg:.1f} "
                f"mean_center_pref_l23e_hz={full_center_pref_rate:.6f} "
                f"min_center_pref_l23e_hz={full_min_center_pref_rate:.6f}"
            ),
        )
        overall_ok &= print_result(
            full_context["min_center_som_hz"] > 0.0 and full_context["min_broad_som_hz"] > 0.0,
            "som_sanity",
            (
                f"driven_center_threshold_hz={full_context['driven_center_threshold_hz']:.6f} "
                f"validation_sites={int(full_context['validation_site_count'])} "
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
                f"validation_sites={int(full_context['validation_site_count'])} "
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
            f"primary_site={primary_validation_site_id} "
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
            selected_orientations=full_size["selected_orientations_by_site"],
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
                f"validation_sites={int(full_size['validation_site_count'])} "
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

        if args.require_som_size_surround:
            som_size_surround = compute_som_size_surround_metrics(
                full,
                somoff,
                full_size,
                somoff_size,
                full_context,
            )
            peak_index = int(full_l23e_size["peak_index"])
            overall_ok &= print_result(
                som_size_surround["peak_l23e_rate"] >= 1.0
                and 0 < peak_index < (len(radii) - 1)
                and som_size_surround["summation_index"] > 0.0
                and som_size_surround["large_suppression_index"] > 0.0
                and som_size_surround["site_curve_pass_fraction"] >= 0.50,
                "som_size_curve_shape",
                (
                    f"validation_sites={int(som_size_surround['site_count'])} "
                    f"peak_radius={som_size_surround['peak_radius']:.6f} "
                    f"small_l23e_rate={som_size_surround['small_l23e_rate']:.6f} "
                    f"peak_l23e_rate={som_size_surround['peak_l23e_rate']:.6f} "
                    f"large_l23e_rate={som_size_surround['large_l23e_rate']:.6f} "
                    f"summation_index={som_size_surround['summation_index']:.6f} "
                    f"large_suppression_index={som_size_surround['large_suppression_index']:.6f} "
                    f"site_curve_pass_fraction={som_size_surround['site_curve_pass_fraction']:.6f} "
                    f"site_driven_fraction={som_size_surround['site_driven_fraction']:.6f} "
                    f"site_large_suppressed_fraction={som_size_surround['site_large_suppressed_fraction']:.6f}"
                ),
            )
            overall_ok &= print_result(
                som_size_surround["l23e_l4_suppression_delta"] >= 0.05,
                "som_size_l4_vs_l23e",
                (
                    f"l23e_suppression={som_size_surround['l23e_suppression']:.6f} "
                    f"l4e_suppression={som_size_surround['l4_suppression']:.6f} "
                    f"delta={som_size_surround['l23e_l4_suppression_delta']:.6f} "
                    "threshold=0.050000 round4=1"
                ),
            )
            overall_ok &= print_result(
                som_size_surround["som_large_or_broad_rate"] > 0.0
                and (
                    som_size_surround["som_large_or_broad_rate"]
                    >= 0.8 * som_size_surround["som_center_or_peak_rate"]
                    or som_size_surround["som_recruitment_index"] >= -0.20
                ),
                "som_size_som_recruitment",
                (
                    f"small_som_rate={som_size_surround['small_som_rate']:.6f} "
                    f"peak_som_rate={som_size_surround['peak_som_rate']:.6f} "
                    f"large_som_rate={som_size_surround['large_som_rate']:.6f} "
                    f"center_context_som_rate={som_size_surround['center_context_som_rate']:.6f} "
                    f"broad_context_som_rate={som_size_surround['broad_context_som_rate']:.6f} "
                    f"som_center_or_peak_rate={som_size_surround['som_center_or_peak_rate']:.6f} "
                    f"som_large_or_broad_rate={som_size_surround['som_large_or_broad_rate']:.6f} "
                    f"som_recruitment_index={som_size_surround['som_recruitment_index']:.6f}"
                ),
            )
            overall_ok &= print_result(
                som_size_surround["site_rescue_fraction"] >= 0.50
                and (
                    som_size_surround["mean_large_rate_delta_somoff_minus_full"] >= 0.0
                    or som_size_surround["mean_suppression_reduction_full_minus_somoff"] >= 0.03
                ),
                "som_size_somoff_site_rescue",
                (
                    f"site_rescue_fraction={som_size_surround['site_rescue_fraction']:.6f} "
                    f"full_large_l23e_rate={som_size_surround['large_l23e_rate']:.6f} "
                    f"somoff_large_l23e_rate={som_size_surround['somoff_large_l23e_rate']:.6f} "
                    f"mean_large_rate_delta_somoff_minus_full="
                    f"{som_size_surround['mean_large_rate_delta_somoff_minus_full']:.6f} "
                    f"full_l23e_suppression={som_size_surround['l23e_suppression']:.6f} "
                    f"somoff_l23e_suppression={som_size_surround['somoff_l23e_suppression']:.6f} "
                    f"mean_suppression_reduction_full_minus_somoff="
                    f"{som_size_surround['mean_suppression_reduction_full_minus_somoff']:.6f}"
                ),
            )
            full_orientation_context_for_som = compute_orientation_context_suppression_metrics(
                require_orientation_context_rows(full)
            )
            somoff_orientation_context_for_som = compute_orientation_context_suppression_metrics(
                require_orientation_context_rows(somoff)
            )
            osd_reduction = (
                full_orientation_context_for_som["mean_osd_l23e"]
                - somoff_orientation_context_for_som["mean_osd_l23e"]
            )
            same_suppression_reduction = (
                full_orientation_context_for_som["mean_si_same_l23e"]
                - somoff_orientation_context_for_som["mean_si_same_l23e"]
            )
            min_driven_sites = max(
                1,
                int(math.ceil(0.50 * full_orientation_context_for_som["site_count"])),
            )
            overall_ok &= print_result(
                full_orientation_context_for_som["driven_count"] >= min_driven_sites
                and somoff_orientation_context_for_som["driven_count"] >= 1.0
                and (osd_reduction >= 0.05 or same_suppression_reduction >= 0.05),
                "som_orientation_context_somoff_effect",
                (
                    f"full_driven={int(full_orientation_context_for_som['driven_count'])} "
                    f"somoff_driven={int(somoff_orientation_context_for_som['driven_count'])} "
                    f"required_full_driven={min_driven_sites} "
                    f"full_mean_osd_l23e={full_orientation_context_for_som['mean_osd_l23e']:.6f} "
                    f"somoff_mean_osd_l23e={somoff_orientation_context_for_som['mean_osd_l23e']:.6f} "
                    f"osd_reduction={osd_reduction:.6f} "
                    f"full_mean_si_same_l23e={full_orientation_context_for_som['mean_si_same_l23e']:.6f} "
                    f"somoff_mean_si_same_l23e={somoff_orientation_context_for_som['mean_si_same_l23e']:.6f} "
                    f"same_suppression_reduction={same_suppression_reduction:.6f}"
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

        if args.require_l23ee_recurrent_biology:
            heavy_tail = compute_l23ee_recurrent_heavy_tail_metrics(specificity_rows)
            heavy_tail_top1_ok = (
                heavy_tail["active_count"] < 100.0
                or heavy_tail["top1_mass_share"] > 0.01
            )
            overall_ok &= print_result(
                heavy_tail["active_count"] >= 20.0
                and heavy_tail["gini"] >= 0.20
                and heavy_tail["top10_mass_share"] >= 0.12
                and heavy_tail_top1_ok
                and heavy_tail["upper_cap_fraction"] < 0.01,
                "l23ee_recurrent_heavy_tail",
                (
                    f"active_count={int(heavy_tail['active_count'])} "
                    f"p50={heavy_tail['p50']:.6f} "
                    f"p90={heavy_tail['p90']:.6f} "
                    f"p95={heavy_tail['p95']:.6f} "
                    f"p99={heavy_tail['p99']:.6f} "
                    f"max={heavy_tail['max']:.6f} "
                    f"mean={heavy_tail['mean']:.6f} "
                    f"std={heavy_tail['std']:.6f} "
                    f"cv={heavy_tail['cv']:.6f} "
                    f"gini={heavy_tail['gini']:.6f} "
                    f"top1pct_mass_share={heavy_tail['top1_mass_share']:.6f} "
                    f"top5pct_mass_share={heavy_tail['top5_mass_share']:.6f} "
                    f"top10pct_mass_share={heavy_tail['top10_mass_share']:.6f} "
                    f"upper_bound={heavy_tail['upper_bound']:.6f} "
                    f"upper_cap_fraction={heavy_tail['upper_cap_fraction']:.6f} "
                    "description=bounded_heavy_tailed_like"
                ),
            )

            shuffle = compute_l23ee_recurrent_shuffle_specificity_metrics(specificity_rows)
            overall_ok &= print_result(
                shuffle["row_count"] >= shuffle["min_active_count"]
                and shuffle["top_count"] >= 5.0
                and shuffle["qualifying_count"] >= 5.0
                and (
                    shuffle["observed_delta"] > shuffle["shuffle_q95_delta"]
                    or shuffle["z_score"] >= 2.0
                ),
                "l23ee_recurrent_shuffle_specificity",
                (
                    f"subset={shuffle['selected_label']} "
                    f"row_count={int(shuffle['row_count'])} "
                    f"all_row_count={int(shuffle['all_row_count'])} "
                    f"min_count={int(shuffle['min_active_count'])} "
                    f"top_count={int(shuffle['top_count'])} "
                    f"corr_gt_0p2_count={int(shuffle['qualifying_count'])} "
                    f"observed_top_fraction={shuffle['observed_top_fraction']:.6f} "
                    f"observed_all_fraction={shuffle['observed_all_fraction']:.6f} "
                    f"observed_delta={shuffle['observed_delta']:.6f} "
                    f"shuffle_count={int(shuffle['shuffle_count'])} "
                    f"shuffle_seed={int(shuffle['shuffle_seed'])} "
                    f"shuffle_mean_delta={shuffle['shuffle_mean_delta']:.6f} "
                    f"shuffle_std_delta={shuffle['shuffle_std_delta']:.6f} "
                    f"shuffle_q95_delta={shuffle['shuffle_q95_delta']:.6f} "
                    f"z_score={shuffle['z_score']:.6f} "
                    f"strata_count={int(shuffle['strata_count'])}"
                ),
            )

            cotuning_bins = compute_l23ee_recurrent_cotuning_bin_metrics(specificity_rows)
            print(f"INFO l23ee_recurrent_cotuning_bins {format_l23ee_recurrent_cotuning_bins(cotuning_bins)}")
            cotuning_w_mean_margin = (
                cotuning_bins["high_corr_gt_0p2_mean_w_after"]
                - cotuning_bins["low_corr_le_0p2_mean_w_after"]
            )
            cotuning_w_median_margin = (
                cotuning_bins["high_corr_gt_0p2_median_w_after"]
                - cotuning_bins["low_corr_le_0p2_median_w_after"]
            )
            cotuning_delta_mean_margin = (
                cotuning_bins["high_corr_gt_0p2_mean_delta_w"]
                - cotuning_bins["low_corr_le_0p2_mean_delta_w"]
            )
            cotuning_delta_median_margin = (
                cotuning_bins["high_corr_gt_0p2_median_delta_w"]
                - cotuning_bins["low_corr_le_0p2_median_delta_w"]
            )
            cotuning_best_margin = max(
                cotuning_w_mean_margin,
                cotuning_w_median_margin,
                cotuning_delta_mean_margin,
                cotuning_delta_median_margin,
            )
            cotuning_margin_threshold = max(1.0e-6, 0.01 * cotuning_bins["p95_abs_delta_w"])
            overall_ok &= print_result(
                cotuning_bins["high_count"] >= cotuning_bins["min_count"]
                and cotuning_bins["low_count"] >= cotuning_bins["min_count"]
                and cotuning_best_margin >= cotuning_margin_threshold,
                "l23ee_recurrent_cotuning_bins",
                (
                    f"row_count={int(cotuning_bins['row_count'])} "
                    f"low_count={int(cotuning_bins['low_count'])} "
                    f"high_count={int(cotuning_bins['high_count'])} "
                    f"min_count={int(cotuning_bins['min_count'])} "
                    f"high_mean_w_after={cotuning_bins['high_corr_gt_0p2_mean_w_after']:.6f} "
                    f"low_mean_w_after={cotuning_bins['low_corr_le_0p2_mean_w_after']:.6f} "
                    f"high_median_w_after={cotuning_bins['high_corr_gt_0p2_median_w_after']:.6f} "
                    f"low_median_w_after={cotuning_bins['low_corr_le_0p2_median_w_after']:.6f} "
                    f"high_mean_delta_w={cotuning_bins['high_corr_gt_0p2_mean_delta_w']:.6f} "
                    f"low_mean_delta_w={cotuning_bins['low_corr_le_0p2_mean_delta_w']:.6f} "
                    f"high_median_delta_w={cotuning_bins['high_corr_gt_0p2_median_delta_w']:.6f} "
                    f"low_median_delta_w={cotuning_bins['low_corr_le_0p2_median_delta_w']:.6f} "
                    f"best_margin={cotuning_best_margin:.6f} "
                    f"margin_threshold={cotuning_margin_threshold:.6f}"
                ),
            )

            reciprocal = compute_l23ee_recurrent_reciprocal_metrics(specificity_rows)
            print(
                "INFO l23ee_recurrent_reciprocal "
                f"active_count={int(reciprocal['active_count'])} "
                f"reciprocal_count={int(reciprocal['reciprocal_count'])} "
                f"nonreciprocal_count={int(reciprocal['nonreciprocal_count'])} "
                f"reciprocal_fraction={reciprocal['reciprocal_fraction']:.6f} "
                f"reciprocal_mean_w_after={reciprocal['reciprocal_mean_w_after']:.6f} "
                f"nonreciprocal_mean_w_after={reciprocal['nonreciprocal_mean_w_after']:.6f} "
                f"top10_count={int(reciprocal['top10_count'])} "
                f"top10_reciprocal_fraction={reciprocal['top10_reciprocal_fraction']:.6f} "
                f"top10_reciprocal_enrichment={reciprocal['top10_reciprocal_enrichment']:.6f}"
            )

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

        if args.allow_responsive_osi:
            if strict_osi_ok:
                print("INFO osi_responsive_rescue not_needed strict_all_site_pass=1")
            else:
                full_l23e_osi = l23e_osi_metrics_by_label["full"]
                control_l23e_osi = l23e_osi_metrics_by_label["control"]
                full_responsive_osi = full_l23e_osi.responsive_median_osi
                control_responsive_osi = control_l23e_osi.responsive_median_osi
                responsive_delta = (
                    full_responsive_osi - control_responsive_osi
                    if full_responsive_osi is not None and control_responsive_osi is not None
                    else None
                )
                responsive_osi_ok = (
                    full_l23e_osi.responsive_count > 0
                    and control_l23e_osi.responsive_count > 0
                    and full_responsive_osi is not None
                    and control_responsive_osi is not None
                    and responsive_delta is not None
                    and full_responsive_osi >= 0.70
                    and responsive_delta >= 0.10
                )
                downstream_gates_ok = overall_ok
                overall_ok &= print_result(
                    downstream_gates_ok and responsive_osi_ok,
                    "osi_responsive_rescue",
                    (
                        f"strict_all_site_pass=0 "
                        f"downstream_gates_pass={int(downstream_gates_ok)} "
                        f"responsive_gate_pass={int(responsive_osi_ok)} "
                        f"threshold_hz={args.responsive_rate_threshold_hz:.6f} "
                        f"full_responsive_count={full_l23e_osi.responsive_count} "
                        f"control_responsive_count={control_l23e_osi.responsive_count} "
                        f"full_responsive_median_osi={format_optional_float(full_responsive_osi)} "
                        f"control_responsive_median_osi={format_optional_float(control_responsive_osi)} "
                        f"responsive_delta={format_optional_float(responsive_delta)}"
                    ),
                )

        return 0 if overall_ok else 1
    except ValidationError as exc:
        print(f"FAIL input {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
