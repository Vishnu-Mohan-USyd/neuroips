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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal Python installs.
    np = None


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
class VideoPopulationRateRow:
    """One natural-video replay population-rate row."""

    repeat_index: int
    frame_index: int
    population: str
    rate_hz: float
    frame_start_ms: float
    frame_end_ms: float


@dataclass(frozen=True)
class VideoSiteRateRow:
    """One natural-video replay site-rate row."""

    repeat_index: int
    frame_index: int
    population: str
    site_id: int
    rate_hz: float


@dataclass(frozen=True)
class VideoFrameSummaryRow:
    """One natural-video replay frame summary row."""

    repeat_index: int
    frame_index: int
    frame_start_ms: float
    frame_end_ms: float
    l4e_rate_hz: float
    l23e_rate_hz: float
    l23pv_rate_hz: float
    l23som_rate_hz: float
    l4e_drive_min: float
    l4e_drive_mean: float
    l4e_drive_max: float
    l4e_drive_std: float


@dataclass(frozen=True)
class VideoEventBinRow:
    """One event-aligned natural-video timing bin row."""

    condition: str
    repeat_index: int
    event_index: int
    frame_index: int
    population: str
    site_id: int | None
    bin_index: int
    bin_start_ms: float
    bin_end_ms: float
    rate_hz: float
    spike_count: float
    event_start_ms: float
    gray_current: float
    l4e_drive_min: float
    l4e_drive_mean: float
    l4e_drive_max: float
    l4e_drive_std: float


@dataclass(frozen=True)
class HVAPredictorPredictionRow:
    """One host-side HVA sidecar future-tile prediction row."""

    prediction_index: int
    repeat_index: int
    frame_index: int
    target_frame_index: int
    target_channel_index: int
    target_channel: str
    tile_id: int
    split: str
    learning_update_applied: int
    target_state_norm: float
    predicted_state_norm: float
    target_residual_norm: float
    predicted_residual_norm: float
    target_residual_z: float
    predicted_residual_z: float
    train_residual_mean_norm: float
    train_residual_std_norm: float
    persistence_pred_state_norm: float
    train_mean_pred_state_norm: float
    no_learning_pred_state_norm: float
    temporal_block_shift_pred_state_norm: float
    spatial_tile_shuffle_pred_state_norm: float
    target_rate_hz: float
    predicted_rate_hz: float
    error_rate_hz: float
    event_window_target_state_norm: float = 0.0
    event_threshold_norm: float = 0.0
    event_tile_selected: int = 0
    target_event: int = 0
    single_frame_target_event: int = 0
    predicted_event_prob: float = 0.0
    persistence_event_prob: float = 0.0
    train_event_rate: float = 0.0
    no_learning_event_prob: float = 0.0
    temporal_block_shift_event_prob: float = 0.0
    spatial_tile_shuffle_event_prob: float = 0.0
    event_error: float = 0.0
    topk_target_value_norm: float = 0.0
    topk_target: int = 0
    topk_sample_valid: int = 0
    topk_model_score: float = 0.0
    topk_model_prob: float = 0.0
    topk_persistence_score: float = 0.0
    topk_train_frequency_score: float = 0.0
    topk_no_learning_score: float = 0.0
    topk_temporal_block_shift_score: float = 0.0
    topk_spatial_tile_shuffle_score: float = 0.0


@dataclass(frozen=True)
class HVAPredictorEventTileRow:
    """One HVA L23E event-threshold summary row for a predictor tile."""

    target_channel_index: int
    target_channel: str
    tile_id: int
    threshold_norm: float
    threshold_hz: float
    train_count: int
    train_positive_count: int
    train_negative_count: int
    heldout_count: int
    heldout_positive_count: int
    train_positive_fraction: float
    heldout_positive_fraction: float
    selected: int


@dataclass(frozen=True)
class HVAPredictorWeightRow:
    """One host-side HVA sidecar tile-to-tile predictor weight row."""

    pre_tile_id: int
    post_tile_id: int
    target_channel_index: int
    target_channel: str
    pre_tile_x: int
    pre_tile_y: int
    post_tile_x: int
    post_tile_y: int
    distance_tiles: float
    manhattan_distance_tiles: int
    w_before: float
    w_after: float
    delta_w: float
    abs_weight_sum_after: float


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

    genn_dir: Path
    prefix: str
    summary: dict[str, float]
    context_rows: dict[tuple[str, str], ContextRow]
    context_rows_by_site: dict[int, dict[tuple[str, str], ContextRow]]
    post_site_rates: dict[str, list[float]]
    l4_post_sites: list[PostSiteMetric] | None
    l23e_post_sites: list[PostSiteMetric]
    weights: dict[str, tuple[WeightSeries, WeightSeries]]
    vip_weight_files: list[Path]
    final_post_video_site_rates: dict[str, list[float]] | None = None
    final_post_video_l4_sites: list[PostSiteMetric] | None = None
    final_post_video_l23e_sites: list[PostSiteMetric] | None = None
    final_post_video_l23e_cell_tuning: dict[int, CellTuningRow] | None = None
    final_post_video_l23e_cell_tuning_multiphase: dict[int, MultiPhaseCellTuningRow] | None = None
    final_post_video_context_rows: dict[tuple[str, str], ContextRow] | None = None
    final_post_video_context_rows_by_site: dict[int, dict[tuple[str, str], ContextRow]] | None = None
    final_post_video_size_tuning_rows: list[SizeTuningRow] | None = None
    size_tuning_rows: list[SizeTuningRow] | None = None
    orientation_context_rows: list[OrientationContextRow] | None = None
    blank_baseline_rows: list[BlankBaselineRow] | None = None
    contrast_sweep_rows: list[ContrastSweepRow] | None = None
    video_population_rows: list[VideoPopulationRateRow] | None = None
    video_site_rows: list[VideoSiteRateRow] | None = None
    video_frame_summary_rows: list[VideoFrameSummaryRow] | None = None
    video_event_population_bin_rows: list[VideoEventBinRow] | None = None
    video_event_site_bin_rows: list[VideoEventBinRow] | None = None
    hva_predictor_config: dict[str, float] | None = None
    hva_predictor_metrics: dict[str, float] | None = None
    video_consolidation_metrics: dict[str, float] | None = None
    hva_predictor_rate_row_count: int | None = None
    hva_predictor_predictions: list[HVAPredictorPredictionRow] | None = None
    hva_predictor_event_tiles: list[HVAPredictorEventTileRow] | None = None
    hva_predictor_weights: list[HVAPredictorWeightRow] | None = None
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
        "--pvoff",
        help="Optional prefix for PV-output-off ablation run used by L2/3 video reliability reporting.",
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
        "--require-natural-video-physiology",
        action="store_true",
        help="Require opt-in natural-video lower-V1 replay artifacts and bounded physiology gates.",
    )
    parser.add_argument(
        "--require-l23-video-reliability",
        action="store_true",
        help=(
            "Require artifact-only L2/3 natural-video reliability diagnostics, "
            "raw repeat-oracle ceiling, L4 alignment, and ablation reporting."
        ),
    )
    parser.add_argument(
        "--require-l23-activity-reliability",
        action="store_true",
        help=(
            "Require strict raw L2/3 natural-video activity stability gates: "
            "raw top-k repeat oracle, repeat correlation, and bounded active-tile density."
        ),
    )
    parser.add_argument(
        "--l23-video-min-frame-top1-accuracy",
        type=float,
        default=None,
        help=(
            "Optional stricter minimum frame-top1 accuracy for the "
            "l23_video_representational_validity gate. Defaults to the "
            "historical chance-margin threshold when omitted."
        ),
    )
    parser.add_argument(
        "--l23-video-min-raw-oracle-at-k",
        type=float,
        default=0.45,
        help="Minimum no-leak raw L2/3 top-k repeat-oracle recall when activity reliability is required.",
    )
    parser.add_argument(
        "--l23-video-min-raw-oracle-ceiling-fraction",
        type=float,
        default=0.75,
        help="Minimum no-leak raw oracle divided by leaky repeat-oracle ceiling.",
    )
    parser.add_argument(
        "--l23-video-min-l23e-repeat-corr",
        type=float,
        default=0.35,
        help="Minimum L23E repeat correlation for natural-video frame activity.",
    )
    parser.add_argument(
        "--l23-video-max-mean-active-tile-fraction",
        type=float,
        default=0.65,
        help="Maximum mean fraction of active L23E video tiles per sample.",
    )
    parser.add_argument(
        "--l23-video-max-sample-active-tile-fraction",
        type=float,
        default=0.80,
        help="Maximum per-sample fraction of active L23E video tiles.",
    )
    parser.add_argument(
        "--require-emergent-ff-gain",
        action="store_true",
        help=(
            "Require opt-in emergent L4E->L23E feedforward-gain audit: "
            "no transient eval FF replay gain, causal video FF STDP exposure, "
            "and nonzero video-exposure L4E->L23E weight deltas."
        ),
    )
    parser.add_argument(
        "--require-event-driven-ff-plasticity",
        action="store_true",
        help=(
            "Require event-driven local L4E->L23E trace plasticity during video exposure "
            "and reject windowed spike-count-only coactivity rules."
        ),
    )
    parser.add_argument(
        "--require-natural-video-event-timing",
        action="store_true",
        help="Require opt-in millisecond event-aligned natural-video timing artifacts and gates.",
    )
    parser.add_argument(
        "--require-hva-predictor",
        action="store_true",
        help="Require default-off HVA predictor-only sidecar artifacts and isolation/prediction gates.",
    )
    parser.add_argument(
        "--require-hva-population-prediction",
        action="store_true",
        help=(
            "Require heldout HVA L23E population-distribution prediction gates "
            "computed from existing top-k prediction artifacts."
        ),
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


def parse_video_population_rates_csv(path: Path) -> list[VideoPopulationRateRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {"frame_index", "population", "rate_hz", "frame_start_ms", "frame_end_ms"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing video population-rate columns in {path}: {sorted(missing)}")

        rows: list[VideoPopulationRateRow] = []
        for row_number, row in enumerate(reader, start=2):
            population = (row.get("population") or "").strip()
            if not population:
                raise ValidationError(f"Missing population in {path} row {row_number}")
            rows.append(
                VideoPopulationRateRow(
                    repeat_index=(
                        parse_int(row["repeat_index"], path, row_number, "repeat_index")
                        if "repeat_index" in row and row.get("repeat_index", "") != ""
                        else 0
                    ),
                    frame_index=parse_int(row["frame_index"], path, row_number, "frame_index"),
                    population=population,
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                    frame_start_ms=parse_float(row["frame_start_ms"], path, row_number, "frame_start_ms"),
                    frame_end_ms=parse_float(row["frame_end_ms"], path, row_number, "frame_end_ms"),
                )
            )

    if not rows:
        raise ValidationError(f"Video population-rate file is empty: {path}")
    return rows


def parse_video_site_rates_csv(path: Path) -> list[VideoSiteRateRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {"frame_index", "population", "site_id", "rate_hz"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing video site-rate columns in {path}: {sorted(missing)}")

        rows: list[VideoSiteRateRow] = []
        for row_number, row in enumerate(reader, start=2):
            population = (row.get("population") or "").strip()
            if not population:
                raise ValidationError(f"Missing population in {path} row {row_number}")
            rows.append(
                VideoSiteRateRow(
                    repeat_index=(
                        parse_int(row["repeat_index"], path, row_number, "repeat_index")
                        if "repeat_index" in row and row.get("repeat_index", "") != ""
                        else 0
                    ),
                    frame_index=parse_int(row["frame_index"], path, row_number, "frame_index"),
                    population=population,
                    site_id=parse_int(row["site_id"], path, row_number, "site_id"),
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                )
            )

    if not rows:
        raise ValidationError(f"Video site-rate file is empty: {path}")
    return rows


def parse_video_frame_summary_csv(path: Path) -> list[VideoFrameSummaryRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "frame_index",
            "frame_start_ms",
            "frame_end_ms",
            "l4e_rate_hz",
            "l23e_rate_hz",
            "l23pv_rate_hz",
            "l23som_rate_hz",
            "l4e_drive_min",
            "l4e_drive_mean",
            "l4e_drive_max",
            "l4e_drive_std",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing video frame-summary columns in {path}: {sorted(missing)}")

        rows: list[VideoFrameSummaryRow] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                VideoFrameSummaryRow(
                    repeat_index=(
                        parse_int(row["repeat_index"], path, row_number, "repeat_index")
                        if "repeat_index" in row and row.get("repeat_index", "") != ""
                        else 0
                    ),
                    frame_index=parse_int(row["frame_index"], path, row_number, "frame_index"),
                    frame_start_ms=parse_float(row["frame_start_ms"], path, row_number, "frame_start_ms"),
                    frame_end_ms=parse_float(row["frame_end_ms"], path, row_number, "frame_end_ms"),
                    l4e_rate_hz=parse_float(row["l4e_rate_hz"], path, row_number, "l4e_rate_hz"),
                    l23e_rate_hz=parse_float(row["l23e_rate_hz"], path, row_number, "l23e_rate_hz"),
                    l23pv_rate_hz=parse_float(row["l23pv_rate_hz"], path, row_number, "l23pv_rate_hz"),
                    l23som_rate_hz=parse_float(row["l23som_rate_hz"], path, row_number, "l23som_rate_hz"),
                    l4e_drive_min=parse_float(row["l4e_drive_min"], path, row_number, "l4e_drive_min"),
                    l4e_drive_mean=parse_float(row["l4e_drive_mean"], path, row_number, "l4e_drive_mean"),
                    l4e_drive_max=parse_float(row["l4e_drive_max"], path, row_number, "l4e_drive_max"),
                    l4e_drive_std=parse_float(row["l4e_drive_std"], path, row_number, "l4e_drive_std"),
                )
            )

    if not rows:
        raise ValidationError(f"Video frame-summary file is empty: {path}")
    return rows


def parse_video_event_bins_csv(path: Path, *, has_site_id: bool) -> list[VideoEventBinRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "condition",
            "repeat_index",
            "event_index",
            "frame_index",
            "population",
            "bin_index",
            "bin_start_ms",
            "bin_end_ms",
            "rate_hz",
            "spike_count",
            "event_start_ms",
            "gray_current",
            "l4e_drive_min",
            "l4e_drive_mean",
            "l4e_drive_max",
            "l4e_drive_std",
        }
        if has_site_id:
            required.add("site_id")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing video event-bin columns in {path}: {sorted(missing)}")

        rows: list[VideoEventBinRow] = []
        for row_number, row in enumerate(reader, start=2):
            condition = (row.get("condition") or "").strip()
            population = (row.get("population") or "").strip()
            if not condition or not population:
                raise ValidationError(f"Missing condition/population in {path} row {row_number}")
            rows.append(
                VideoEventBinRow(
                    condition=condition,
                    repeat_index=parse_int(row["repeat_index"], path, row_number, "repeat_index"),
                    event_index=parse_int(row["event_index"], path, row_number, "event_index"),
                    frame_index=parse_int(row["frame_index"], path, row_number, "frame_index"),
                    population=population,
                    site_id=(
                        parse_int(row["site_id"], path, row_number, "site_id")
                        if has_site_id
                        else None
                    ),
                    bin_index=parse_int(row["bin_index"], path, row_number, "bin_index"),
                    bin_start_ms=parse_float(row["bin_start_ms"], path, row_number, "bin_start_ms"),
                    bin_end_ms=parse_float(row["bin_end_ms"], path, row_number, "bin_end_ms"),
                    rate_hz=parse_float(row["rate_hz"], path, row_number, "rate_hz"),
                    spike_count=parse_float(row["spike_count"], path, row_number, "spike_count"),
                    event_start_ms=parse_float(row["event_start_ms"], path, row_number, "event_start_ms"),
                    gray_current=parse_float(row["gray_current"], path, row_number, "gray_current"),
                    l4e_drive_min=parse_float(row["l4e_drive_min"], path, row_number, "l4e_drive_min"),
                    l4e_drive_mean=parse_float(row["l4e_drive_mean"], path, row_number, "l4e_drive_mean"),
                    l4e_drive_max=parse_float(row["l4e_drive_max"], path, row_number, "l4e_drive_max"),
                    l4e_drive_std=parse_float(row["l4e_drive_std"], path, row_number, "l4e_drive_std"),
                )
            )

    if not rows:
        raise ValidationError(f"Video event-bin file is empty: {path}")
    return rows


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValidationError(f"CSV file is empty: {path}") from exc
        return sum(1 for _ in reader)


def parse_hva_predictor_predictions_csv(path: Path) -> list[HVAPredictorPredictionRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "prediction_index",
            "repeat_index",
            "frame_index",
            "target_frame_index",
            "target_channel_index",
            "target_channel",
            "tile_id",
            "split",
            "learning_update_applied",
            "target_state_norm",
            "predicted_state_norm",
            "target_residual_norm",
            "predicted_residual_norm",
            "target_residual_z",
            "predicted_residual_z",
            "train_residual_mean_norm",
            "train_residual_std_norm",
            "persistence_pred_state_norm",
            "train_mean_pred_state_norm",
            "no_learning_pred_state_norm",
            "temporal_block_shift_pred_state_norm",
            "spatial_tile_shuffle_pred_state_norm",
            "target_rate_hz",
            "predicted_rate_hz",
            "error_rate_hz",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing HVA prediction columns in {path}: {sorted(missing)}")

        rows: list[HVAPredictorPredictionRow] = []
        for row_number, row in enumerate(reader, start=2):
            def optional_float(column: str, default: float = 0.0) -> float:
                return (
                    parse_float(row[column], path, row_number, column)
                    if column in row and row[column] != ""
                    else default
                )

            def optional_int(column: str, default: int = 0) -> int:
                return (
                    parse_int(row[column], path, row_number, column)
                    if column in row and row[column] != ""
                    else default
                )

            rows.append(
                HVAPredictorPredictionRow(
                    prediction_index=parse_int(row["prediction_index"], path, row_number, "prediction_index"),
                    repeat_index=parse_int(row["repeat_index"], path, row_number, "repeat_index"),
                    frame_index=parse_int(row["frame_index"], path, row_number, "frame_index"),
                    target_frame_index=parse_int(row["target_frame_index"], path, row_number, "target_frame_index"),
                    target_channel_index=parse_int(
                        row["target_channel_index"],
                        path,
                        row_number,
                        "target_channel_index",
                    ),
                    target_channel=(row.get("target_channel") or "").strip(),
                    tile_id=parse_int(row["tile_id"], path, row_number, "tile_id"),
                    split=(row.get("split") or "").strip(),
                    learning_update_applied=parse_int(
                        row["learning_update_applied"],
                        path,
                        row_number,
                        "learning_update_applied",
                    ),
                    target_state_norm=parse_float(row["target_state_norm"], path, row_number, "target_state_norm"),
                    predicted_state_norm=parse_float(row["predicted_state_norm"], path, row_number, "predicted_state_norm"),
                    target_residual_norm=parse_float(
                        row["target_residual_norm"],
                        path,
                        row_number,
                        "target_residual_norm",
                    ),
                    predicted_residual_norm=parse_float(
                        row["predicted_residual_norm"],
                        path,
                        row_number,
                        "predicted_residual_norm",
                    ),
                    target_residual_z=parse_float(
                        row["target_residual_z"],
                        path,
                        row_number,
                        "target_residual_z",
                    ),
                    predicted_residual_z=parse_float(
                        row["predicted_residual_z"],
                        path,
                        row_number,
                        "predicted_residual_z",
                    ),
                    train_residual_mean_norm=parse_float(
                        row["train_residual_mean_norm"],
                        path,
                        row_number,
                        "train_residual_mean_norm",
                    ),
                    train_residual_std_norm=parse_float(
                        row["train_residual_std_norm"],
                        path,
                        row_number,
                        "train_residual_std_norm",
                    ),
                    persistence_pred_state_norm=parse_float(
                        row["persistence_pred_state_norm"],
                        path,
                        row_number,
                        "persistence_pred_state_norm",
                    ),
                    train_mean_pred_state_norm=parse_float(
                        row["train_mean_pred_state_norm"],
                        path,
                        row_number,
                        "train_mean_pred_state_norm",
                    ),
                    no_learning_pred_state_norm=parse_float(
                        row["no_learning_pred_state_norm"],
                        path,
                        row_number,
                        "no_learning_pred_state_norm",
                    ),
                    temporal_block_shift_pred_state_norm=parse_float(
                        row["temporal_block_shift_pred_state_norm"],
                        path,
                        row_number,
                        "temporal_block_shift_pred_state_norm",
                    ),
                    spatial_tile_shuffle_pred_state_norm=parse_float(
                        row["spatial_tile_shuffle_pred_state_norm"],
                        path,
                        row_number,
                        "spatial_tile_shuffle_pred_state_norm",
                    ),
                    target_rate_hz=parse_float(row["target_rate_hz"], path, row_number, "target_rate_hz"),
                    predicted_rate_hz=parse_float(row["predicted_rate_hz"], path, row_number, "predicted_rate_hz"),
                    error_rate_hz=parse_float(row["error_rate_hz"], path, row_number, "error_rate_hz"),
                    event_window_target_state_norm=optional_float("event_window_target_state_norm"),
                    event_threshold_norm=optional_float("event_threshold_norm"),
                    event_tile_selected=optional_int("event_tile_selected"),
                    target_event=optional_int("target_event"),
                    single_frame_target_event=optional_int("single_frame_target_event"),
                    predicted_event_prob=optional_float("predicted_event_prob"),
                    persistence_event_prob=optional_float("persistence_event_prob"),
                    train_event_rate=optional_float("train_event_rate"),
                    no_learning_event_prob=optional_float("no_learning_event_prob"),
                    temporal_block_shift_event_prob=optional_float("temporal_block_shift_event_prob"),
                    spatial_tile_shuffle_event_prob=optional_float("spatial_tile_shuffle_event_prob"),
                    event_error=optional_float("event_error"),
                    topk_target_value_norm=optional_float("topk_target_value_norm"),
                    topk_target=optional_int("topk_target"),
                    topk_sample_valid=optional_int("topk_sample_valid"),
                    topk_model_score=optional_float("topk_model_score"),
                    topk_model_prob=optional_float("topk_model_prob"),
                    topk_persistence_score=optional_float("topk_persistence_score"),
                    topk_train_frequency_score=optional_float("topk_train_frequency_score"),
                    topk_no_learning_score=optional_float("topk_no_learning_score"),
                    topk_temporal_block_shift_score=optional_float("topk_temporal_block_shift_score"),
                    topk_spatial_tile_shuffle_score=optional_float("topk_spatial_tile_shuffle_score"),
                )
            )

    if not rows:
        raise ValidationError(f"HVA prediction file is empty: {path}")
    return rows


def parse_hva_predictor_event_tiles_csv(path: Path) -> list[HVAPredictorEventTileRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "target_channel_index",
            "target_channel",
            "tile_id",
            "threshold_norm",
            "threshold_hz",
            "train_count",
            "train_positive_count",
            "train_negative_count",
            "heldout_count",
            "heldout_positive_count",
            "train_positive_fraction",
            "heldout_positive_fraction",
            "selected",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing HVA event-tile columns in {path}: {sorted(missing)}")

        rows: list[HVAPredictorEventTileRow] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                HVAPredictorEventTileRow(
                    target_channel_index=parse_int(
                        row["target_channel_index"],
                        path,
                        row_number,
                        "target_channel_index",
                    ),
                    target_channel=(row.get("target_channel") or "").strip(),
                    tile_id=parse_int(row["tile_id"], path, row_number, "tile_id"),
                    threshold_norm=parse_float(row["threshold_norm"], path, row_number, "threshold_norm"),
                    threshold_hz=parse_float(row["threshold_hz"], path, row_number, "threshold_hz"),
                    train_count=parse_int(row["train_count"], path, row_number, "train_count"),
                    train_positive_count=parse_int(
                        row["train_positive_count"],
                        path,
                        row_number,
                        "train_positive_count",
                    ),
                    train_negative_count=parse_int(
                        row["train_negative_count"],
                        path,
                        row_number,
                        "train_negative_count",
                    ),
                    heldout_count=parse_int(row["heldout_count"], path, row_number, "heldout_count"),
                    heldout_positive_count=parse_int(
                        row["heldout_positive_count"],
                        path,
                        row_number,
                        "heldout_positive_count",
                    ),
                    train_positive_fraction=parse_float(
                        row["train_positive_fraction"],
                        path,
                        row_number,
                        "train_positive_fraction",
                    ),
                    heldout_positive_fraction=parse_float(
                        row["heldout_positive_fraction"],
                        path,
                        row_number,
                        "heldout_positive_fraction",
                    ),
                    selected=parse_int(row["selected"], path, row_number, "selected"),
                )
            )

    if not rows:
        raise ValidationError(f"HVA event-tile file is empty: {path}")
    return rows


def parse_hva_predictor_weights_csv(path: Path) -> list[HVAPredictorWeightRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing header in {path}")

        required = {
            "target_channel_index",
            "target_channel",
            "pre_tile_id",
            "post_tile_id",
            "pre_tile_x",
            "pre_tile_y",
            "post_tile_x",
            "post_tile_y",
            "distance_tiles",
            "manhattan_distance_tiles",
            "w_before",
            "w_after",
            "delta_w",
            "abs_weight_sum_after",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing HVA weight columns in {path}: {sorted(missing)}")

        rows: list[HVAPredictorWeightRow] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                HVAPredictorWeightRow(
                    pre_tile_id=parse_int(row["pre_tile_id"], path, row_number, "pre_tile_id"),
                    post_tile_id=parse_int(row["post_tile_id"], path, row_number, "post_tile_id"),
                    target_channel_index=parse_int(
                        row["target_channel_index"],
                        path,
                        row_number,
                        "target_channel_index",
                    ),
                    target_channel=(row.get("target_channel") or "").strip(),
                    pre_tile_x=parse_int(row["pre_tile_x"], path, row_number, "pre_tile_x"),
                    pre_tile_y=parse_int(row["pre_tile_y"], path, row_number, "pre_tile_y"),
                    post_tile_x=parse_int(row["post_tile_x"], path, row_number, "post_tile_x"),
                    post_tile_y=parse_int(row["post_tile_y"], path, row_number, "post_tile_y"),
                    distance_tiles=parse_float(row["distance_tiles"], path, row_number, "distance_tiles"),
                    manhattan_distance_tiles=parse_int(
                        row["manhattan_distance_tiles"],
                        path,
                        row_number,
                        "manhattan_distance_tiles",
                    ),
                    w_before=parse_float(row["w_before"], path, row_number, "w_before"),
                    w_after=parse_float(row["w_after"], path, row_number, "w_after"),
                    delta_w=parse_float(row["delta_w"], path, row_number, "delta_w"),
                    abs_weight_sum_after=parse_float(
                        row["abs_weight_sum_after"],
                        path,
                        row_number,
                        "abs_weight_sum_after",
                    ),
                )
            )

    if not rows:
        raise ValidationError(f"HVA weight file is empty: {path}")
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
    final_post_video_site_paths = {
        population: genn_dir / f"{prefix}_final_post_video_{population}_sites.csv"
        for population in ("l23", "l23pv", "l23som")
    }
    final_post_video_site_rates = (
        {
            "l23e": parse_site_rates_csv(final_post_video_site_paths["l23"]),
            "l23pv": parse_site_rates_csv(final_post_video_site_paths["l23pv"]),
            "l23som": parse_site_rates_csv(final_post_video_site_paths["l23som"]),
        }
        if all(path.is_file() for path in final_post_video_site_paths.values())
        else None
    )
    final_post_video_l4_path = genn_dir / f"{prefix}_final_post_video_l4_sites.csv"
    final_post_video_l4_sites = (
        parse_post_site_metrics_csv(final_post_video_l4_path)
        if final_post_video_l4_path.is_file()
        else None
    )
    final_post_video_l23_path = genn_dir / f"{prefix}_final_post_video_l23_sites.csv"
    final_post_video_l23e_sites = (
        parse_post_site_metrics_csv(final_post_video_l23_path)
        if final_post_video_l23_path.is_file()
        else None
    )
    final_post_video_context_path = genn_dir / f"{prefix}_final_post_video_som_context_validation.csv"
    if final_post_video_context_path.is_file():
        final_post_video_context_rows, final_post_video_context_rows_by_site = parse_context_csv(
            final_post_video_context_path
        )
    else:
        final_post_video_context_rows = None
        final_post_video_context_rows_by_site = None

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
    final_post_video_size_tuning_path = genn_dir / f"{prefix}_final_post_video_size_tuning.csv"
    final_post_video_size_tuning_rows = (
        parse_size_tuning_csv(final_post_video_size_tuning_path)
        if final_post_video_size_tuning_path.is_file()
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
    video_population_path = genn_dir / f"{prefix}_video_population_rates.csv"
    video_population_rows = (
        parse_video_population_rates_csv(video_population_path)
        if video_population_path.is_file()
        else None
    )
    video_site_path = genn_dir / f"{prefix}_video_site_rates.csv"
    video_site_rows = (
        parse_video_site_rates_csv(video_site_path)
        if video_site_path.is_file()
        else None
    )
    video_frame_summary_path = genn_dir / f"{prefix}_video_frame_summary.csv"
    video_frame_summary_rows = (
        parse_video_frame_summary_csv(video_frame_summary_path)
        if video_frame_summary_path.is_file()
        else None
    )
    video_event_population_path = genn_dir / f"{prefix}_video_event_population_bins.csv"
    video_event_population_bin_rows = (
        parse_video_event_bins_csv(video_event_population_path, has_site_id=False)
        if video_event_population_path.is_file()
        else None
    )
    video_event_site_path = genn_dir / f"{prefix}_video_event_site_bins.csv"
    video_event_site_bin_rows = (
        parse_video_event_bins_csv(video_event_site_path, has_site_id=True)
        if video_event_site_path.is_file()
        else None
    )
    hva_predictor_config_path = genn_dir / f"{prefix}_hva_predictor_config.csv"
    hva_predictor_config = (
        parse_summary_csv(hva_predictor_config_path)
        if hva_predictor_config_path.is_file()
        else None
    )
    hva_predictor_metrics_path = genn_dir / f"{prefix}_hva_predictor_metrics.csv"
    hva_predictor_metrics = (
        parse_summary_csv(hva_predictor_metrics_path)
        if hva_predictor_metrics_path.is_file()
        else None
    )
    video_consolidation_metrics_path = genn_dir / f"{prefix}_video_consolidation_metrics.csv"
    video_consolidation_metrics = (
        parse_summary_csv(video_consolidation_metrics_path)
        if video_consolidation_metrics_path.is_file()
        else None
    )
    hva_predictor_rates_path = genn_dir / f"{prefix}_hva_predictor_rates.csv"
    hva_predictor_rate_row_count = (
        count_csv_data_rows(hva_predictor_rates_path)
        if hva_predictor_rates_path.is_file()
        else None
    )
    hva_predictor_predictions_path = genn_dir / f"{prefix}_hva_predictor_predictions.csv"
    hva_predictor_predictions = (
        parse_hva_predictor_predictions_csv(hva_predictor_predictions_path)
        if hva_predictor_predictions_path.is_file()
        else None
    )
    hva_predictor_event_tiles_path = genn_dir / f"{prefix}_hva_predictor_event_tiles.csv"
    hva_predictor_event_tiles = (
        parse_hva_predictor_event_tiles_csv(hva_predictor_event_tiles_path)
        if hva_predictor_event_tiles_path.is_file()
        else None
    )
    hva_predictor_weights_path = genn_dir / f"{prefix}_hva_predictor_weights.csv"
    hva_predictor_weights = (
        parse_hva_predictor_weights_csv(hva_predictor_weights_path)
        if hva_predictor_weights_path.is_file()
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
    final_post_video_cell_tuning_path = genn_dir / f"{prefix}_final_post_video_l23e_cell_tuning.csv"
    final_post_video_l23e_cell_tuning = (
        parse_cell_tuning_csv(final_post_video_cell_tuning_path)
        if final_post_video_cell_tuning_path.is_file()
        else None
    )
    final_post_video_multiphase_cell_tuning_path = (
        genn_dir / f"{prefix}_final_post_video_l23e_cell_tuning_multiphase.csv"
    )
    final_post_video_l23e_cell_tuning_multiphase = (
        parse_multiphase_cell_tuning_csv(final_post_video_multiphase_cell_tuning_path)
        if final_post_video_multiphase_cell_tuning_path.is_file()
        else None
    )

    return RunData(
        genn_dir=genn_dir,
        prefix=prefix,
        summary=summary,
        context_rows=context_rows,
        context_rows_by_site=context_rows_by_site,
        post_site_rates=post_site_rates,
        l4_post_sites=l4_post_sites,
        l23e_post_sites=l23e_post_sites,
        weights=weights,
        vip_weight_files=vip_weight_files,
        final_post_video_site_rates=final_post_video_site_rates,
        final_post_video_l4_sites=final_post_video_l4_sites,
        final_post_video_l23e_sites=final_post_video_l23e_sites,
        final_post_video_l23e_cell_tuning=final_post_video_l23e_cell_tuning,
        final_post_video_l23e_cell_tuning_multiphase=final_post_video_l23e_cell_tuning_multiphase,
        final_post_video_context_rows=final_post_video_context_rows,
        final_post_video_context_rows_by_site=final_post_video_context_rows_by_site,
        final_post_video_size_tuning_rows=final_post_video_size_tuning_rows,
        size_tuning_rows=size_tuning_rows,
        orientation_context_rows=orientation_context_rows,
        blank_baseline_rows=blank_baseline_rows,
        contrast_sweep_rows=contrast_sweep_rows,
        video_population_rows=video_population_rows,
        video_site_rows=video_site_rows,
        video_frame_summary_rows=video_frame_summary_rows,
        video_event_population_bin_rows=video_event_population_bin_rows,
        video_event_site_bin_rows=video_event_site_bin_rows,
        hva_predictor_config=hva_predictor_config,
        hva_predictor_metrics=hva_predictor_metrics,
        video_consolidation_metrics=video_consolidation_metrics,
        hva_predictor_rate_row_count=hva_predictor_rate_row_count,
        hva_predictor_predictions=hva_predictor_predictions,
        hva_predictor_event_tiles=hva_predictor_event_tiles,
        hva_predictor_weights=hva_predictor_weights,
        specificity_rows=specificity_rows,
        l23e_cell_tuning=l23e_cell_tuning,
        l23e_cell_tuning_multiphase=l23e_cell_tuning_multiphase,
    )


def try_load_optional_run(genn_dir: Path, prefix: str | None, label: str) -> RunData | None:
    if prefix is None:
        return None
    try:
        return load_run(genn_dir, prefix)
    except ValidationError as exc:
        print(f"INFO optional_run_unavailable[{label}] prefix={prefix} error={str(exc).replace(' ', '_')}")
        return None


def require_summary_metric(run: RunData, metric: str) -> float:
    if metric not in run.summary:
        raise ValidationError(f"Missing summary metric {metric!r} in prefix {run.prefix}")
    value = run.summary[metric]
    if not math.isfinite(value):
        raise ValidationError(f"Non-finite summary metric {metric!r} in prefix {run.prefix}")
    return value


def final_post_video_orientation_missing(run: RunData) -> list[str]:
    missing: list[str] = []
    if run.final_post_video_site_rates is None:
        missing.append(f"{run.prefix}_final_post_video_l23/l23pv/l23som_sites.csv")
    if run.final_post_video_l4_sites is None:
        missing.append(f"{run.prefix}_final_post_video_l4_sites.csv")
    if run.final_post_video_l23e_sites is None:
        missing.append(f"{run.prefix}_final_post_video_l23_sites.csv")
    if run.final_post_video_l23e_cell_tuning is None:
        missing.append(f"{run.prefix}_final_post_video_l23e_cell_tuning.csv")
    if run.final_post_video_l23e_cell_tuning_multiphase is None:
        missing.append(f"{run.prefix}_final_post_video_l23e_cell_tuning_multiphase.csv")
    return missing


def final_post_video_som_missing(run: RunData) -> list[str]:
    missing: list[str] = []
    if run.final_post_video_context_rows is None or run.final_post_video_context_rows_by_site is None:
        missing.append(f"{run.prefix}_final_post_video_som_context_validation.csv")
    if run.final_post_video_size_tuning_rows is None:
        missing.append(f"{run.prefix}_final_post_video_size_tuning.csv")
    return missing


def with_final_post_video_som_artifacts(run: RunData) -> RunData:
    if (
        run.final_post_video_context_rows is None
        or run.final_post_video_context_rows_by_site is None
        or run.final_post_video_size_tuning_rows is None
    ):
        raise ValidationError(f"Final post-video SOM/context artifacts are missing for prefix {run.prefix}")
    return replace(
        run,
        context_rows=run.final_post_video_context_rows,
        context_rows_by_site=run.final_post_video_context_rows_by_site,
        size_tuning_rows=run.final_post_video_size_tuning_rows,
    )


def print_final_post_video_reference_info(run: RunData) -> None:
    pre_single = (
        compute_cell_responsive_metrics(run.l23e_cell_tuning, 5.0)
        if run.l23e_cell_tuning is not None
        else None
    )
    pre_multi = (
        compute_multiphase_cell_responsive_metrics(run.l23e_cell_tuning_multiphase, 5.0)
        if run.l23e_cell_tuning_multiphase is not None
        else None
    )
    final_single = (
        compute_cell_responsive_metrics(run.final_post_video_l23e_cell_tuning, 5.0)
        if run.final_post_video_l23e_cell_tuning is not None
        else None
    )
    final_multi = (
        compute_multiphase_cell_responsive_metrics(
            run.final_post_video_l23e_cell_tuning_multiphase,
            5.0,
        )
        if run.final_post_video_l23e_cell_tuning_multiphase is not None
        else None
    )
    print(
        "INFO final_post_video_orientation_reference "
        f"pre_post_l23_median_osi={run.summary.get('post_l23_median_osi', math.nan):.6f} "
        f"final_post_video_l23_median_osi="
        f"{run.summary.get('final_post_video_l23_median_osi', math.nan):.6f} "
        f"pre_single_site_fraction="
        f"{format_optional_float(pre_single.responsive_site_fraction if pre_single is not None else None)} "
        f"final_single_site_fraction="
        f"{format_optional_float(final_single.responsive_site_fraction if final_single is not None else None)} "
        f"pre_multiphase_ge1="
        f"{format_optional_float(pre_multi.responsive_site_fraction_ge1 if pre_multi is not None else None)} "
        f"final_multiphase_ge1="
        f"{format_optional_float(final_multi.responsive_site_fraction_ge1 if final_multi is not None else None)} "
        f"pre_multiphase_ge2="
        f"{format_optional_float(pre_multi.responsive_site_fraction_ge2 if pre_multi is not None else None)} "
        f"final_multiphase_ge2="
        f"{format_optional_float(final_multi.responsive_site_fraction_ge2 if final_multi is not None else None)}"
    )


def optional_summary_metric(run: RunData, metric: str) -> float | None:
    if metric not in run.summary:
        return None
    value = run.summary[metric]
    if not math.isfinite(value):
        raise ValidationError(f"Non-finite summary metric {metric!r} in prefix {run.prefix}")
    return value


def first_summary_metric(run: RunData, metrics: tuple[str, ...]) -> tuple[str | None, float | None]:
    for metric in metrics:
        value = optional_summary_metric(run, metric)
        if value is not None:
            return metric, value
    return None, None


def summary_metric_values(run: RunData, metrics: tuple[str, ...]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for metric in metrics:
        value = optional_summary_metric(run, metric)
        if value is not None:
            values.append((metric, value))
    return values


def format_metric_values(values: list[tuple[str, float]]) -> str:
    if not values:
        return "missing"
    return "|".join(f"{metric}:{value:.6f}" for metric, value in values)


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

    active_rows = [row for row in rows if row.pre_peak_hz > 0.0 and row.post_peak_hz > 0.0]
    min_active_count = min(100, max(20, len(rows) // 100))
    active_or_all_rows = active_rows if len(active_rows) >= min_active_count else rows
    active_nonzero_corr_rows = [
        row
        for row in active_rows
        if abs(row.response_corr) > 1.0e-9
    ]
    nonzero_corr_rows = [
        row
        for row in active_or_all_rows
        if abs(row.response_corr) > 1.0e-9
    ]
    min_nonzero_count = min(50, max(5, len(active_or_all_rows) // 20))
    nonzero_corr_fraction_floor = 0.80
    nonzero_corr_fraction = len(nonzero_corr_rows) / len(active_or_all_rows)
    active_nonzero_corr_fraction = (
        len(active_nonzero_corr_rows) / len(active_rows)
        if active_rows
        else 0.0
    )
    nonzero_subset_allowed = (
        len(nonzero_corr_rows) >= min_nonzero_count
        and nonzero_corr_fraction >= nonzero_corr_fraction_floor
    )
    if nonzero_subset_allowed:
        selected_rows = nonzero_corr_rows
        selected_label = "nonzero_corr_active_endpoints" if active_or_all_rows is active_rows else "nonzero_corr_all"
        selected_mode_code = 2.0 if active_or_all_rows is active_rows else 3.0
    else:
        selected_rows = active_or_all_rows
        selected_label = "active_endpoints" if active_or_all_rows is active_rows else "all"
        selected_mode_code = 1.0 if active_or_all_rows is active_rows else 0.0

    sorted_rows = sorted(selected_rows, key=lambda row: row.response_corr)
    quantile_count = max(1, len(sorted_rows) // 4)
    min_count = min(50, max(5, len(sorted_rows) // 20))
    low_corr = sorted_rows[:quantile_count]
    high_corr = sorted_rows[-quantile_count:]
    active_sorted_rows = sorted(active_rows, key=lambda row: row.response_corr)
    active_quantile_count = max(1, len(active_sorted_rows) // 4) if active_sorted_rows else 0
    active_low_corr = active_sorted_rows[:active_quantile_count]
    active_high_corr = active_sorted_rows[-active_quantile_count:] if active_quantile_count > 0 else []
    active_min_count = min(50, max(5, len(active_sorted_rows) // 20)) if active_sorted_rows else min_count
    all_sorted_rows = sorted(rows, key=lambda row: row.response_corr)
    all_quantile_count = max(1, len(all_sorted_rows) // 4)
    all_low_corr = all_sorted_rows[:all_quantile_count]
    all_high_corr = all_sorted_rows[-all_quantile_count:]

    metrics = {
        "row_count": float(len(selected_rows)),
        "all_row_count": float(len(rows)),
        "active_endpoint_count": float(len(active_rows)),
        "nonzero_corr_count": float(len(nonzero_corr_rows)),
        "active_nonzero_corr_count": float(len(active_nonzero_corr_rows)),
        "nonzero_corr_fraction": nonzero_corr_fraction,
        "active_nonzero_corr_fraction": active_nonzero_corr_fraction,
        "nonzero_corr_fraction_floor": nonzero_corr_fraction_floor,
        "nonzero_subset_allowed": 1.0 if nonzero_subset_allowed else 0.0,
        "min_active_count": float(min_active_count),
        "min_nonzero_count": float(min_nonzero_count),
        "active_min_count": float(active_min_count),
        "selected_mode_code": selected_mode_code,
        "quantile_fraction": 0.25,
        "min_count": float(min_count),
        "p95_abs_delta_w": percentile([abs(row.delta_w) for row in selected_rows], 95.0),
        "active_p95_abs_delta_w": percentile([abs(row.delta_w) for row in active_rows], 95.0)
        if active_rows
        else math.nan,
    }
    metrics.update(summarize_correlation_group(low_corr, "low_corr"))
    metrics.update(summarize_correlation_group(high_corr, "high_corr"))
    metrics.update(summarize_correlation_group(active_low_corr, "active_low_corr"))
    metrics.update(summarize_correlation_group(active_high_corr, "active_high_corr"))
    metrics.update(summarize_correlation_group(all_low_corr, "all_low_corr"))
    metrics.update(summarize_correlation_group(all_high_corr, "all_high_corr"))
    metrics["selected_label"] = selected_label
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


def standard_deviation(values: list[float]) -> float:
    if not values:
        raise ValidationError("Cannot compute standard deviation of an empty collection.")
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / len(values))


def video_population_rates_by_name(rows: list[VideoPopulationRateRow]) -> dict[str, list[float]]:
    rates: dict[str, list[float]] = {}
    for row in rows:
        rates.setdefault(row.population, []).append(row.rate_hz)
    return rates


def video_site_rates_by_name(rows: list[VideoSiteRateRow]) -> dict[str, list[float]]:
    rates: dict[str, list[float]] = {}
    for row in rows:
        rates.setdefault(row.population, []).append(row.rate_hz)
    return rates


def fraction_less_than(values: list[float], threshold: float) -> float:
    if not values:
        raise ValidationError("Cannot compute fraction of an empty collection.")
    return sum(1 for value in values if value < threshold) / len(values)


def pearson_correlation_optional(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = 0.0
    x_sq = 0.0
    y_sq = 0.0
    for x_value, y_value in zip(x_values, y_values):
        dx = x_value - x_mean
        dy = y_value - y_mean
        numerator += dx * dy
        x_sq += dx * dx
        y_sq += dy * dy
    denominator = math.sqrt(x_sq * y_sq)
    if denominator <= 1.0e-12:
        return None
    return numerator / denominator


def format_optional_float(value: float | None) -> str:
    return "nan" if value is None or not math.isfinite(value) else f"{value:.6f}"


def cosine_similarity_optional(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or not x_values:
        return None
    dot = 0.0
    x_sq = 0.0
    y_sq = 0.0
    for x_value, y_value in zip(x_values, y_values):
        dot += x_value * y_value
        x_sq += x_value * x_value
        y_sq += y_value * y_value
    denominator = math.sqrt(x_sq * y_sq)
    if denominator <= 1.0e-12:
        return None
    return dot / denominator


def vector_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValidationError("Cannot compute a vector mean of an empty collection.")
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValidationError("Vector mean requires aligned non-empty vectors.")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def top_k_indices(values: list[float], k: int) -> list[int]:
    if k <= 0 or k > len(values):
        raise ValidationError("Top-k selection requires 0 < k <= value count.")
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:k]


def topk_recall(predicted: list[int], target: list[int], k: int) -> float:
    if k <= 0:
        raise ValidationError("Top-k recall requires k > 0.")
    return len(set(predicted).intersection(target)) / k


def normalized_entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    active_count = sum(1 for count in counts if count > 0)
    if total <= 0 or active_count <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(active_count)


def numpy_normalized_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 1.0e-12
    normalized = np.zeros_like(matrix, dtype=float)
    if np.any(valid):
        normalized[valid] = matrix[valid] / norms[valid, None]
    return normalized, valid


def numpy_pearson_optional(x_values, y_values) -> float | None:
    if x_values.shape != y_values.shape or x_values.size < 3:
        return None
    x_centered = x_values - np.mean(x_values)
    y_centered = y_values - np.mean(y_values)
    denominator = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denominator <= 1.0e-12:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def compute_l23_video_representational_metrics_numpy(
    vectors: dict[tuple[int, int], list[float]],
    repeats: list[int],
    frames: list[int],
) -> dict[str, float] | None:
    if np is None:
        return None

    same_sum = 0.0
    same_count = 0
    different_sum = 0.0
    different_count = 0
    for first_index, first_repeat in enumerate(repeats):
        for second_repeat in repeats[first_index + 1:]:
            common_frames = [
                frame_index for frame_index in frames
                if (first_repeat, frame_index) in vectors and (second_repeat, frame_index) in vectors
            ]
            if not common_frames:
                continue
            first_matrix = np.asarray([vectors[(first_repeat, frame_index)] for frame_index in common_frames], dtype=float)
            second_matrix = np.asarray([vectors[(second_repeat, frame_index)] for frame_index in common_frames], dtype=float)
            first_normalized, first_valid = numpy_normalized_rows(first_matrix)
            second_normalized, second_valid = numpy_normalized_rows(second_matrix)
            valid_same = first_valid & second_valid
            if np.any(valid_same):
                same_values = np.sum(first_normalized[valid_same] * second_normalized[valid_same], axis=1)
                same_sum += float(np.sum(same_values))
                same_count += int(same_values.size)
            similarity_matrix = first_normalized @ second_normalized.T
            valid_different = np.outer(first_valid, second_valid)
            valid_different &= ~np.eye(len(common_frames), dtype=bool)
            if np.any(valid_different):
                different_values = similarity_matrix[valid_different]
                different_sum += float(np.sum(different_values))
                different_count += int(different_values.size)

    decoded_count = 0
    correct_count = 0
    top5_count = 0
    rank_sum = 0.0
    rank_count = 0
    for repeat_index in repeats:
        other_repeats = [other for other in repeats if other != repeat_index]
        sample_frames = [frame_index for frame_index in frames if (repeat_index, frame_index) in vectors]
        template_frames: list[int] = []
        template_vectors: list[list[float]] = []
        for template_frame in frames:
            source_vectors = [
                vectors[(other_repeat, template_frame)]
                for other_repeat in other_repeats
                if (other_repeat, template_frame) in vectors
            ]
            if source_vectors:
                template_frames.append(template_frame)
                template_vectors.append(vector_mean(source_vectors))
        if not sample_frames or not template_frames:
            continue
        sample_matrix = np.asarray([vectors[(repeat_index, frame_index)] for frame_index in sample_frames], dtype=float)
        template_matrix = np.asarray(template_vectors, dtype=float)
        sample_normalized, sample_valid = numpy_normalized_rows(sample_matrix)
        template_normalized, template_valid = numpy_normalized_rows(template_matrix)
        if not np.any(template_valid):
            continue
        score_matrix = sample_normalized @ template_normalized.T
        score_matrix[:, ~template_valid] = -np.inf
        template_frame_to_col = {template_frame: col for col, template_frame in enumerate(template_frames)}
        for row_index, sample_is_valid in enumerate(sample_valid):
            if not sample_is_valid:
                continue
            predicted_frame = template_frames[int(np.argmax(score_matrix[row_index]))]
            decoded_count += 1
            correct_count += int(predicted_frame == sample_frames[row_index])
            true_col = template_frame_to_col.get(sample_frames[row_index])
            if true_col is None:
                continue
            true_score = float(score_matrix[row_index, true_col])
            if not math.isfinite(true_score):
                continue
            rank = 1 + int(np.sum(score_matrix[row_index] > true_score))
            rank_sum += float(rank)
            rank_count += 1
            top5_count += int(rank <= 5)

    same_mean = (same_sum / same_count) if same_count else math.nan
    different_mean = (different_sum / different_count) if different_count else math.nan
    return {
        "repeat_count": float(len(repeats)),
        "frame_count": float(len(frames)),
        "same_similarity": same_mean,
        "different_similarity": different_mean,
        "same_different_gap": same_mean - different_mean if math.isfinite(same_mean) and math.isfinite(different_mean) else math.nan,
        "frame_top1_accuracy": (correct_count / decoded_count) if decoded_count else math.nan,
        "frame_top5_accuracy": (top5_count / rank_count) if rank_count else math.nan,
        "frame_mean_rank": (rank_sum / rank_count) if rank_count else math.nan,
        "frame_chance": 1.0 / len(frames),
        "decoded_count": float(decoded_count),
    }


def compute_l23_video_l4_alignment_metrics_numpy(
    l4_by_frame: dict[int, list[float]],
    l23_by_frame: dict[int, list[float]],
    frames: list[int],
) -> dict[str, float] | None:
    if np is None:
        return None
    l4_matrix = np.asarray([l4_by_frame[frame_index] for frame_index in frames], dtype=float)
    l23_matrix = np.asarray([l23_by_frame[frame_index] for frame_index in frames], dtype=float)
    l4_normalized, l4_valid = numpy_normalized_rows(l4_matrix)
    l23_normalized, l23_valid = numpy_normalized_rows(l23_matrix)
    cross_similarity = l4_normalized @ l23_normalized.T
    valid_cross = np.outer(l4_valid, l23_valid)
    diagonal_mask = np.eye(len(frames), dtype=bool)
    same_values = cross_similarity[valid_cross & diagonal_mask]
    different_values = cross_similarity[valid_cross & ~diagonal_mask]

    l4_rsm = l4_normalized @ l4_normalized.T
    l23_rsm = l23_normalized @ l23_normalized.T
    upper_mask = np.triu(np.ones((len(frames), len(frames)), dtype=bool), k=1)
    valid_rsm = np.outer(l4_valid, l4_valid) & np.outer(l23_valid, l23_valid) & upper_mask
    rsm_correlation = (
        numpy_pearson_optional(l4_rsm[valid_rsm], l23_rsm[valid_rsm])
        if np.any(valid_rsm)
        else None
    )
    temporal_shuffle_rsm_correlation: float | None = None
    if len(frames) >= 4:
        temporal_indices = np.roll(np.arange(len(frames)), max(1, len(frames) // 2))
        temporal_l23_valid = l23_valid[temporal_indices]
        temporal_l23_rsm = l23_normalized[temporal_indices] @ l23_normalized[temporal_indices].T
        temporal_valid_rsm = np.outer(l4_valid, l4_valid) & np.outer(temporal_l23_valid, temporal_l23_valid) & upper_mask
        temporal_shuffle_rsm_correlation = (
            numpy_pearson_optional(l4_rsm[temporal_valid_rsm], temporal_l23_rsm[temporal_valid_rsm])
            if np.any(temporal_valid_rsm)
            else None
        )

    spatial_shuffle_rsm_correlation: float | None = None
    site_count = l23_matrix.shape[1] if l23_matrix.ndim == 2 else 0
    if site_count > 1:
        spatial_l23_matrix = np.zeros_like(l23_matrix, dtype=float)
        for row_index in range(l23_matrix.shape[0]):
            spatial_l23_matrix[row_index] = np.roll(l23_matrix[row_index], (row_index % (site_count - 1)) + 1)
        spatial_l23_normalized, spatial_l23_valid = numpy_normalized_rows(spatial_l23_matrix)
        spatial_l23_rsm = spatial_l23_normalized @ spatial_l23_normalized.T
        spatial_valid_rsm = np.outer(l4_valid, l4_valid) & np.outer(spatial_l23_valid, spatial_l23_valid) & upper_mask
        spatial_shuffle_rsm_correlation = (
            numpy_pearson_optional(l4_rsm[spatial_valid_rsm], spatial_l23_rsm[spatial_valid_rsm])
            if np.any(spatial_valid_rsm)
            else None
        )

    same_mean = float(np.mean(same_values)) if same_values.size else math.nan
    different_mean = float(np.mean(different_values)) if different_values.size else math.nan
    return {
        "frame_count": float(len(frames)),
        "same_similarity": same_mean,
        "different_similarity": different_mean,
        "same_different_gap": same_mean - different_mean if math.isfinite(same_mean) and math.isfinite(different_mean) else math.nan,
        "rsm_correlation": rsm_correlation if rsm_correlation is not None else math.nan,
        "temporal_shuffle_rsm_correlation": (
            temporal_shuffle_rsm_correlation
            if temporal_shuffle_rsm_correlation is not None
            else math.nan
        ),
        "spatial_shuffle_rsm_correlation": (
            spatial_shuffle_rsm_correlation
            if spatial_shuffle_rsm_correlation is not None
            else math.nan
        ),
    }


def mean_frame_series(frame_rows: list[VideoFrameSummaryRow], attribute: str) -> tuple[list[int], list[float]]:
    values_by_frame: dict[int, list[float]] = {}
    for row in frame_rows:
        values_by_frame.setdefault(row.frame_index, []).append(float(getattr(row, attribute)))
    frame_indices = sorted(values_by_frame)
    return frame_indices, [mean(values_by_frame[frame_index]) for frame_index in frame_indices]


def lagged_correlations(
    source: list[float],
    target: list[float],
    max_lag: int,
) -> dict[int, float | None]:
    correlations: dict[int, float | None] = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            correlations[lag] = pearson_correlation_optional(source, target)
        else:
            correlations[lag] = pearson_correlation_optional(source[:-lag], target[lag:])
    return correlations


def best_lag(correlations: dict[int, float | None]) -> tuple[int | None, float | None]:
    finite = [(lag, corr) for lag, corr in correlations.items() if corr is not None and math.isfinite(corr)]
    if not finite:
        return None, None
    return max(finite, key=lambda item: item[1])


def compute_video_delay_metrics(frame_rows: list[VideoFrameSummaryRow]) -> dict[str, float | None]:
    frame_indices, l4e = mean_frame_series(frame_rows, "l4e_rate_hz")
    _, l23e = mean_frame_series(frame_rows, "l23e_rate_hz")
    _, l23pv = mean_frame_series(frame_rows, "l23pv_rate_hz")
    _, l23som = mean_frame_series(frame_rows, "l23som_rate_hz")
    if len(frame_indices) < 3:
        return {"frame_count": float(len(frame_indices)), "max_lag": 0.0}

    max_lag = min(5, len(frame_indices) - 2)
    metrics: dict[str, float | None] = {
        "frame_count": float(len(frame_indices)),
        "max_lag": float(max_lag),
    }
    for population, target in (("l23e", l23e), ("l23pv", l23pv), ("l23som", l23som)):
        correlations = lagged_correlations(l4e, target, max_lag)
        lag, corr = best_lag(correlations)
        metrics[f"{population}_best_lag_frames"] = float(lag) if lag is not None else None
        metrics[f"{population}_best_corr"] = corr
        metrics[f"{population}_lag0_corr"] = correlations.get(0)
        metrics[f"{population}_lag1_corr"] = correlations.get(1)
    return metrics


def repeat_population_series(
    frame_rows: list[VideoFrameSummaryRow],
    attribute: str,
) -> dict[int, dict[int, float]]:
    series_by_repeat: dict[int, dict[int, float]] = {}
    for row in frame_rows:
        series_by_repeat.setdefault(row.repeat_index, {})[row.frame_index] = float(getattr(row, attribute))
    return series_by_repeat


def pairwise_repeat_reliability(frame_rows: list[VideoFrameSummaryRow], attribute: str) -> float | None:
    series_by_repeat = repeat_population_series(frame_rows, attribute)
    repeats = sorted(series_by_repeat)
    if len(repeats) < 2:
        return None
    correlations: list[float] = []
    for i, first_repeat in enumerate(repeats):
        for second_repeat in repeats[i + 1:]:
            common_frames = sorted(
                set(series_by_repeat[first_repeat]).intersection(series_by_repeat[second_repeat])
            )
            if len(common_frames) < 3:
                continue
            corr = pearson_correlation_optional(
                [series_by_repeat[first_repeat][frame] for frame in common_frames],
                [series_by_repeat[second_repeat][frame] for frame in common_frames],
            )
            if corr is not None and math.isfinite(corr):
                correlations.append(corr)
    return mean(correlations) if correlations else None


def l23e_site_frame_vectors(site_rows: list[VideoSiteRateRow]) -> dict[int, list[float]]:
    values_by_site_frame: dict[int, dict[int, list[float]]] = {}
    frame_ids: set[int] = set()
    for row in site_rows:
        if row.population != "l23e":
            continue
        values_by_site_frame.setdefault(row.site_id, {}).setdefault(row.frame_index, []).append(row.rate_hz)
        frame_ids.add(row.frame_index)
    ordered_frames = sorted(frame_ids)
    vectors: dict[int, list[float]] = {}
    for site_id, by_frame in values_by_site_frame.items():
        if all(frame_index in by_frame for frame_index in ordered_frames):
            vectors[site_id] = [mean(by_frame[frame_index]) for frame_index in ordered_frames]
    return vectors


def video_site_vectors_by_sample(
    site_rows: list[VideoSiteRateRow],
    population: str,
) -> tuple[list[int], dict[tuple[int, int], list[float]]]:
    values_by_sample_site: dict[tuple[int, int], dict[int, list[float]]] = {}
    site_ids: set[int] = set()
    for row in site_rows:
        if row.population != population:
            continue
        sample_key = (row.repeat_index, row.frame_index)
        values_by_sample_site.setdefault(sample_key, {}).setdefault(row.site_id, []).append(row.rate_hz)
        site_ids.add(row.site_id)
    ordered_site_ids = sorted(site_ids)
    if not ordered_site_ids:
        raise ValidationError(f"Video site vectors require rows for population {population!r}.")
    sample_vectors: dict[tuple[int, int], list[float]] = {}
    for sample_key, by_site in values_by_sample_site.items():
        sample_vectors[sample_key] = [
            mean(by_site[site_id]) if site_id in by_site else 0.0
            for site_id in ordered_site_ids
        ]
    return ordered_site_ids, sample_vectors


def infer_video_site_grid_side(site_ids: list[int]) -> int:
    if not site_ids:
        raise ValidationError("Cannot infer video site grid side without site ids.")
    site_count = max(site_ids) + 1
    side = int(round(math.sqrt(site_count)))
    if side * side != site_count:
        raise ValidationError(f"Video site ids do not form a square sheet: max_site_id={max(site_ids)}.")
    return side


def infer_l23_video_tile_grid_side(run: RunData, site_ids: list[int]) -> int:
    candidates = (
        run.summary.get("hva_predictor_tile_grid_side", math.nan),
        (run.hva_predictor_config or {}).get("tile_grid_side", math.nan),
        (run.hva_predictor_metrics or {}).get("topk_tile_grid_side", math.nan),
        (run.hva_predictor_metrics or {}).get("tile_grid_side", math.nan),
    )
    for candidate in candidates:
        if math.isfinite(candidate) and candidate >= 1.0:
            return int(round(candidate))

    sheet_side = infer_video_site_grid_side(site_ids)
    tile_size = run.summary.get(
        "hva_predictor_tile_size_sites",
        (run.hva_predictor_config or {}).get("tile_size_sites", math.nan),
    )
    if math.isfinite(tile_size) and tile_size >= 1.0:
        return max(1, int(math.ceil(sheet_side / tile_size)))
    if sheet_side % 4 == 0:
        return max(1, sheet_side // 4)
    raise ValidationError("Cannot infer L2/3 video tile grid side from artifacts.")


def video_tile_vectors_by_sample(
    run: RunData,
    site_rows: list[VideoSiteRateRow],
    population: str,
) -> tuple[int, dict[tuple[int, int], list[float]]]:
    site_ids, site_vectors = video_site_vectors_by_sample(site_rows, population)
    sheet_side = infer_video_site_grid_side(site_ids)
    tile_grid_side = infer_l23_video_tile_grid_side(run, site_ids)
    if tile_grid_side < 1 or tile_grid_side > sheet_side:
        raise ValidationError(
            f"Invalid L2/3 video tile grid side {tile_grid_side} for sheet side {sheet_side}."
        )
    tile_count = tile_grid_side * tile_grid_side
    tile_for_site: dict[int, int] = {}
    sites_per_tile = [0 for _ in range(tile_count)]
    for site_id in site_ids:
        x = site_id % sheet_side
        y = site_id // sheet_side
        tile_x = min(tile_grid_side - 1, (x * tile_grid_side) // sheet_side)
        tile_y = min(tile_grid_side - 1, (y * tile_grid_side) // sheet_side)
        tile_id = (tile_y * tile_grid_side) + tile_x
        tile_for_site[site_id] = tile_id
        sites_per_tile[tile_id] += 1

    tile_vectors: dict[tuple[int, int], list[float]] = {}
    for sample_key, site_vector in site_vectors.items():
        tile_vector = [0.0 for _ in range(tile_count)]
        for site_id, rate in zip(site_ids, site_vector):
            tile_vector[tile_for_site[site_id]] += rate
        for tile_id, site_count in enumerate(sites_per_tile):
            if site_count > 0:
                tile_vector[tile_id] /= site_count
        tile_vectors[sample_key] = tile_vector
    return tile_grid_side, tile_vectors


def vector_repeat_sets(vectors_by_sample: dict[tuple[int, int], list[float]]) -> tuple[list[int], list[int]]:
    repeats = sorted({repeat_index for repeat_index, _ in vectors_by_sample})
    frames = sorted({frame_index for _, frame_index in vectors_by_sample})
    return repeats, frames


def compute_l23_video_representational_metrics(site_rows: list[VideoSiteRateRow]) -> dict[str, float]:
    _, vectors = video_site_vectors_by_sample(site_rows, "l23e")
    repeats, frames = vector_repeat_sets(vectors)
    if len(repeats) < 2 or len(frames) < 3:
        return {
            "repeat_count": float(len(repeats)),
            "frame_count": float(len(frames)),
            "same_similarity": math.nan,
            "different_similarity": math.nan,
            "same_different_gap": math.nan,
            "frame_top1_accuracy": math.nan,
            "frame_top5_accuracy": math.nan,
            "frame_mean_rank": math.nan,
            "frame_chance": (1.0 / len(frames)) if frames else math.nan,
            "decoded_count": 0.0,
        }
    numpy_metrics = compute_l23_video_representational_metrics_numpy(vectors, repeats, frames)
    if numpy_metrics is not None:
        return numpy_metrics

    same_similarities: list[float] = []
    different_similarities: list[float] = []
    for first_index, first_repeat in enumerate(repeats):
        for second_repeat in repeats[first_index + 1:]:
            common_frames = [
                frame_index for frame_index in frames
                if (first_repeat, frame_index) in vectors and (second_repeat, frame_index) in vectors
            ]
            for frame_index in common_frames:
                similarity = cosine_similarity_optional(
                    vectors[(first_repeat, frame_index)],
                    vectors[(second_repeat, frame_index)],
                )
                if similarity is not None and math.isfinite(similarity):
                    same_similarities.append(similarity)
            for first_frame in common_frames:
                for second_frame in common_frames:
                    if first_frame == second_frame:
                        continue
                    similarity = cosine_similarity_optional(
                        vectors[(first_repeat, first_frame)],
                        vectors[(second_repeat, second_frame)],
                    )
                    if similarity is not None and math.isfinite(similarity):
                        different_similarities.append(similarity)

    decoded_count = 0
    correct_count = 0
    top5_count = 0
    rank_sum = 0.0
    rank_count = 0
    for repeat_index in repeats:
        other_repeats = [other for other in repeats if other != repeat_index]
        for frame_index in frames:
            sample = vectors.get((repeat_index, frame_index))
            if sample is None:
                continue
            scores: list[tuple[float, int]] = []
            for template_frame in frames:
                template_vectors = [
                    vectors[(other_repeat, template_frame)]
                    for other_repeat in other_repeats
                    if (other_repeat, template_frame) in vectors
                ]
                if not template_vectors:
                    continue
                similarity = cosine_similarity_optional(sample, vector_mean(template_vectors))
                if similarity is not None and math.isfinite(similarity):
                    scores.append((similarity, template_frame))
            if not scores:
                continue
            predicted_frame = max(scores, key=lambda item: (item[0], -item[1]))[1]
            decoded_count += 1
            correct_count += int(predicted_frame == frame_index)
            true_scores = [score for score, template_frame in scores if template_frame == frame_index]
            if true_scores:
                true_score = true_scores[0]
                rank = 1 + sum(1 for score, _ in scores if score > true_score)
                rank_sum += float(rank)
                rank_count += 1
                top5_count += int(rank <= 5)

    same_mean = mean(same_similarities) if same_similarities else math.nan
    different_mean = mean(different_similarities) if different_similarities else math.nan
    return {
        "repeat_count": float(len(repeats)),
        "frame_count": float(len(frames)),
        "same_similarity": same_mean,
        "different_similarity": different_mean,
        "same_different_gap": same_mean - different_mean if math.isfinite(same_mean) and math.isfinite(different_mean) else math.nan,
        "frame_top1_accuracy": (correct_count / decoded_count) if decoded_count else math.nan,
        "frame_top5_accuracy": (top5_count / rank_count) if rank_count else math.nan,
        "frame_mean_rank": (rank_sum / rank_count) if rank_count else math.nan,
        "frame_chance": 1.0 / len(frames),
        "decoded_count": float(decoded_count),
    }


def compute_l23_video_l4_alignment_metrics(site_rows: list[VideoSiteRateRow]) -> dict[str, float]:
    _, l4_vectors_by_sample = video_site_vectors_by_sample(site_rows, "l4e")
    _, l23_vectors_by_sample = video_site_vectors_by_sample(site_rows, "l23e")
    frames = sorted({frame_index for _, frame_index in l4_vectors_by_sample}.intersection(
        {frame_index for _, frame_index in l23_vectors_by_sample}
    ))
    l4_by_frame: dict[int, list[float]] = {}
    l23_by_frame: dict[int, list[float]] = {}
    for frame_index in frames:
        l4_samples = [
            vector for (repeat_index, sample_frame), vector in l4_vectors_by_sample.items()
            if sample_frame == frame_index
        ]
        l23_samples = [
            vector for (repeat_index, sample_frame), vector in l23_vectors_by_sample.items()
            if sample_frame == frame_index
        ]
        if l4_samples and l23_samples:
            l4_by_frame[frame_index] = vector_mean(l4_samples)
            l23_by_frame[frame_index] = vector_mean(l23_samples)
    frames = sorted(set(l4_by_frame).intersection(l23_by_frame))
    if len(frames) < 3:
        return {
            "frame_count": float(len(frames)),
            "same_similarity": math.nan,
            "different_similarity": math.nan,
            "same_different_gap": math.nan,
            "rsm_correlation": math.nan,
            "temporal_shuffle_rsm_correlation": math.nan,
            "spatial_shuffle_rsm_correlation": math.nan,
        }
    numpy_metrics = compute_l23_video_l4_alignment_metrics_numpy(l4_by_frame, l23_by_frame, frames)
    if numpy_metrics is not None:
        return numpy_metrics

    same_similarities: list[float] = []
    different_similarities: list[float] = []
    for l4_frame in frames:
        for l23_frame in frames:
            similarity = cosine_similarity_optional(l4_by_frame[l4_frame], l23_by_frame[l23_frame])
            if similarity is None or not math.isfinite(similarity):
                continue
            if l4_frame == l23_frame:
                same_similarities.append(similarity)
            else:
                different_similarities.append(similarity)

    l4_rsm: list[float] = []
    l23_rsm: list[float] = []
    l23_temporal_shuffle_rsm: list[float] = []
    l23_spatial_shuffle_rsm: list[float] = []
    temporal_shift = max(1, len(frames) // 2)
    temporal_frames = frames[temporal_shift:] + frames[:temporal_shift]
    l23_spatial_shuffle_by_frame: dict[int, list[float]] = {}
    for frame_index, frame in enumerate(frames):
        values = l23_by_frame[frame]
        if len(values) > 1:
            shift = (frame_index % (len(values) - 1)) + 1
            l23_spatial_shuffle_by_frame[frame] = values[-shift:] + values[:-shift]
        else:
            l23_spatial_shuffle_by_frame[frame] = list(values)
    for index, first_frame in enumerate(frames):
        for second_index in range(index + 1, len(frames)):
            second_frame = frames[second_index]
            l4_similarity = cosine_similarity_optional(l4_by_frame[first_frame], l4_by_frame[second_frame])
            l23_similarity = cosine_similarity_optional(l23_by_frame[first_frame], l23_by_frame[second_frame])
            first_temporal_frame = temporal_frames[index]
            second_temporal_frame = temporal_frames[second_index]
            l23_temporal_similarity = cosine_similarity_optional(
                l23_by_frame[first_temporal_frame],
                l23_by_frame[second_temporal_frame],
            )
            l23_spatial_similarity = cosine_similarity_optional(
                l23_spatial_shuffle_by_frame[first_frame],
                l23_spatial_shuffle_by_frame[second_frame],
            )
            if (
                l4_similarity is not None
                and l23_similarity is not None
                and math.isfinite(l4_similarity)
                and math.isfinite(l23_similarity)
            ):
                l4_rsm.append(l4_similarity)
                l23_rsm.append(l23_similarity)
                if l23_temporal_similarity is not None and math.isfinite(l23_temporal_similarity):
                    l23_temporal_shuffle_rsm.append(l23_temporal_similarity)
                if l23_spatial_similarity is not None and math.isfinite(l23_spatial_similarity):
                    l23_spatial_shuffle_rsm.append(l23_spatial_similarity)

    same_mean = mean(same_similarities) if same_similarities else math.nan
    different_mean = mean(different_similarities) if different_similarities else math.nan
    rsm_correlation = pearson_correlation_optional(l4_rsm, l23_rsm)
    temporal_shuffle_rsm_correlation = pearson_correlation_optional(l4_rsm, l23_temporal_shuffle_rsm)
    spatial_shuffle_rsm_correlation = pearson_correlation_optional(l4_rsm, l23_spatial_shuffle_rsm)
    return {
        "frame_count": float(len(frames)),
        "same_similarity": same_mean,
        "different_similarity": different_mean,
        "same_different_gap": same_mean - different_mean if math.isfinite(same_mean) and math.isfinite(different_mean) else math.nan,
        "rsm_correlation": rsm_correlation if rsm_correlation is not None else math.nan,
        "temporal_shuffle_rsm_correlation": (
            temporal_shuffle_rsm_correlation
            if temporal_shuffle_rsm_correlation is not None
            else math.nan
        ),
        "spatial_shuffle_rsm_correlation": (
            spatial_shuffle_rsm_correlation
            if spatial_shuffle_rsm_correlation is not None
            else math.nan
        ),
    }


def topk_k_for_l23_video(run: RunData, tile_count: int) -> int:
    candidates = (
        (run.hva_predictor_metrics or {}).get("topk_k", math.nan),
        run.summary.get("hva_predictor_topk_k", math.nan),
        (run.hva_predictor_config or {}).get("topk_k", math.nan),
    )
    for candidate in candidates:
        if math.isfinite(candidate) and candidate >= 1.0:
            return max(1, min(tile_count, int(round(candidate))))
    return min(5, tile_count)


def compute_l23_video_raw_topk_oracle_metrics(run: RunData, site_rows: list[VideoSiteRateRow]) -> dict[str, float]:
    tile_grid_side, tile_vectors = video_tile_vectors_by_sample(run, site_rows, "l23e")
    tile_count = tile_grid_side * tile_grid_side
    topk_k = topk_k_for_l23_video(run, tile_count)
    repeats, frames = vector_repeat_sets(tile_vectors)
    loo_recalls: list[float] = []
    leaky_recalls: list[float] = []
    for repeat_index in repeats:
        other_repeats = [other for other in repeats if other != repeat_index]
        for frame_index in frames:
            target = tile_vectors.get((repeat_index, frame_index))
            if target is None or sum(target) <= 0.0 or max(target) <= 0.0:
                continue
            target_topk = top_k_indices(target, topk_k)
            loo_template_vectors = [
                tile_vectors[(other_repeat, frame_index)]
                for other_repeat in other_repeats
                if (other_repeat, frame_index) in tile_vectors
            ]
            if loo_template_vectors:
                loo_predicted = top_k_indices(vector_mean(loo_template_vectors), topk_k)
                loo_recalls.append(topk_recall(loo_predicted, target_topk, topk_k))
            leaky_template_vectors = [
                tile_vectors[(template_repeat, frame_index)]
                for template_repeat in repeats
                if (template_repeat, frame_index) in tile_vectors
            ]
            if leaky_template_vectors:
                leaky_predicted = top_k_indices(vector_mean(leaky_template_vectors), topk_k)
                leaky_recalls.append(topk_recall(leaky_predicted, target_topk, topk_k))

    return {
        "repeat_count": float(len(repeats)),
        "frame_count": float(len(frames)),
        "tile_grid_side": float(tile_grid_side),
        "tile_count": float(tile_count),
        "topk_k": float(topk_k),
        "loo_no_leak_oracle_recall_at_k": mean(loo_recalls) if loo_recalls else math.nan,
        "leaky_repeat_mean_oracle_recall_at_k": mean(leaky_recalls) if leaky_recalls else math.nan,
        "loo_sample_count": float(len(loo_recalls)),
        "leaky_sample_count": float(len(leaky_recalls)),
    }


def compute_l23_video_tile_entropy_metrics(run: RunData, site_rows: list[VideoSiteRateRow]) -> dict[str, float]:
    tile_grid_side, tile_vectors = video_tile_vectors_by_sample(run, site_rows, "l23e")
    tile_count = tile_grid_side * tile_grid_side
    topk_k = topk_k_for_l23_video(run, tile_count)
    topk_counts = [0 for _ in range(tile_count)]
    active_counts = [0 for _ in range(tile_count)]
    sample_active_fractions: list[float] = []
    valid_sample_count = 0
    for tile_vector in tile_vectors.values():
        if sum(tile_vector) <= 0.0 or max(tile_vector) <= 0.0:
            continue
        valid_sample_count += 1
        active_tile_count = 0
        for tile_id, rate in enumerate(tile_vector):
            if rate > 0.0:
                active_counts[tile_id] += 1
                active_tile_count += 1
        sample_active_fractions.append(active_tile_count / tile_count if tile_count > 0 else math.nan)
        for tile_id in top_k_indices(tile_vector, topk_k):
            topk_counts[tile_id] += 1

    return {
        "tile_grid_side": float(tile_grid_side),
        "tile_count": float(tile_count),
        "topk_k": float(topk_k),
        "valid_sample_count": float(valid_sample_count),
        "topk_entropy_norm": normalized_entropy_from_counts(topk_counts),
        "topk_occupancy_fraction": (
            sum(1 for count in topk_counts if count > 0) / tile_count
            if tile_count > 0
            else math.nan
        ),
        "active_tile_occupancy_fraction": (
            sum(1 for count in active_counts if count > 0) / tile_count
            if tile_count > 0
            else math.nan
        ),
        "mean_active_tile_fraction": mean(sample_active_fractions) if sample_active_fractions else math.nan,
        "max_active_tile_fraction": max(sample_active_fractions) if sample_active_fractions else math.nan,
    }


def l23_video_reliability_summary_for_run(run: RunData) -> dict[str, float] | None:
    if run.video_site_rows is None:
        return None
    representational = compute_l23_video_representational_metrics(run.video_site_rows)
    oracle = compute_l23_video_raw_topk_oracle_metrics(run, run.video_site_rows)
    entropy = compute_l23_video_tile_entropy_metrics(run, run.video_site_rows)
    return {
        "same_different_gap": representational["same_different_gap"],
        "frame_top1_accuracy": representational["frame_top1_accuracy"],
        "frame_top5_accuracy": representational["frame_top5_accuracy"],
        "frame_mean_rank": representational["frame_mean_rank"],
        "loo_oracle": oracle["loo_no_leak_oracle_recall_at_k"],
        "tile_entropy_norm": entropy["topk_entropy_norm"],
        "topk_occupancy_fraction": entropy["topk_occupancy_fraction"],
        "mean_active_tile_fraction": entropy["mean_active_tile_fraction"],
        "max_active_tile_fraction": entropy["max_active_tile_fraction"],
    }


def validate_l23_video_reliability(
    full: RunData,
    control: RunData,
    somoff: RunData,
    recoff: RunData | None,
    pvoff: RunData | None,
    min_frame_top1_accuracy: float | None = None,
) -> bool:
    overall_ok = True
    site_rows = full.video_site_rows
    frame_rows = full.video_frame_summary_rows
    artifacts_available = site_rows is not None and frame_rows is not None
    overall_ok &= print_result(
        artifacts_available,
        "l23_video_reliability_artifacts_available",
        (
            f"video_site_rows={len(site_rows) if site_rows is not None else 0} "
            f"video_frame_summary_rows={len(frame_rows) if frame_rows is not None else 0}"
        ),
    )
    if not artifacts_available or site_rows is None:
        return overall_ok

    representational = compute_l23_video_representational_metrics(site_rows)
    historical_frame_margin_threshold = max(0.10, 5.0 * representational["frame_chance"])
    frame_margin_threshold = historical_frame_margin_threshold
    if min_frame_top1_accuracy is not None:
        frame_margin_threshold = max(frame_margin_threshold, min_frame_top1_accuracy)
    representational_ok = (
        math.isfinite(representational["same_different_gap"])
        and math.isfinite(representational["frame_top1_accuracy"])
        and representational["same_different_gap"] > 0.10
        and representational["frame_top1_accuracy"] >= frame_margin_threshold
        and representational["decoded_count"] >= 20.0
    )
    overall_ok &= print_result(
        representational_ok,
        "l23_video_representational_validity",
        (
            f"repeat_count={representational['repeat_count']:.0f} "
            f"frame_count={representational['frame_count']:.0f} "
            f"same_similarity={representational['same_similarity']:.6f} "
            f"different_similarity={representational['different_similarity']:.6f} "
            f"same_different_gap={representational['same_different_gap']:.6f} "
            f"gap_threshold=0.100000 "
            f"frame_top1_accuracy={representational['frame_top1_accuracy']:.6f} "
            f"frame_top5_accuracy={representational['frame_top5_accuracy']:.6f} "
            f"frame_mean_rank={representational['frame_mean_rank']:.6f} "
            f"frame_chance={representational['frame_chance']:.6f} "
            f"frame_margin_threshold={frame_margin_threshold:.6f} "
            f"historical_frame_margin_threshold={historical_frame_margin_threshold:.6f} "
            f"configured_min_frame_top1_accuracy={format_optional_float(min_frame_top1_accuracy)} "
            f"decoded_count={representational['decoded_count']:.0f}"
        ),
    )

    alignment = compute_l23_video_l4_alignment_metrics(site_rows)
    print(
        "INFO l23_video_l4_l23_alignment "
        f"diagnostic_only=1 "
        f"hard_gate=l23_video_l4_l23_geometry_alignment "
        f"frame_count={alignment['frame_count']:.0f} "
        f"same_l4_l23_similarity={alignment['same_similarity']:.6f} "
        f"different_l4_l23_similarity={alignment['different_similarity']:.6f} "
        f"same_different_gap={alignment['same_different_gap']:.6f}"
    )
    geometry_rsm_threshold = 0.50
    geometry_null_margin = 0.10
    temporal_null = alignment["temporal_shuffle_rsm_correlation"]
    spatial_null = alignment["spatial_shuffle_rsm_correlation"]
    geometry_ok = (
        math.isfinite(alignment["rsm_correlation"])
        and alignment["rsm_correlation"] >= geometry_rsm_threshold
        and math.isfinite(temporal_null)
        and math.isfinite(spatial_null)
        and alignment["rsm_correlation"] >= temporal_null + geometry_null_margin
        and alignment["rsm_correlation"] >= spatial_null + geometry_null_margin
    )
    overall_ok &= print_result(
        geometry_ok,
        "l23_video_l4_l23_geometry_alignment",
        (
            f"frame_count={alignment['frame_count']:.0f} "
            f"rsm_correlation={alignment['rsm_correlation']:.6f} "
            f"rsm_threshold={geometry_rsm_threshold:.6f} "
            f"temporal_shuffle_rsm_correlation={temporal_null:.6f} "
            f"spatial_shuffle_rsm_correlation={spatial_null:.6f} "
            f"null_margin_threshold={geometry_null_margin:.6f} "
            f"rsm_minus_temporal_shuffle={alignment['rsm_correlation'] - temporal_null:.6f} "
            f"rsm_minus_spatial_shuffle={alignment['rsm_correlation'] - spatial_null:.6f}"
        ),
    )

    oracle = compute_l23_video_raw_topk_oracle_metrics(full, site_rows)
    oracle_ok = (
        math.isfinite(oracle["loo_no_leak_oracle_recall_at_k"])
        and oracle["loo_no_leak_oracle_recall_at_k"] >= 0.35
        and oracle["loo_sample_count"] >= 20.0
    )
    overall_ok &= print_result(
        oracle_ok,
        "l23_video_raw_topk_repeat_oracle_ceiling",
        (
            f"repeat_count={oracle['repeat_count']:.0f} "
            f"frame_count={oracle['frame_count']:.0f} "
            f"tile_grid_side={oracle['tile_grid_side']:.0f} "
            f"tile_count={oracle['tile_count']:.0f} "
            f"topk_k={oracle['topk_k']:.0f} "
            f"loo_no_leak_oracle_recall_at_k={oracle['loo_no_leak_oracle_recall_at_k']:.6f} "
            f"loo_threshold=0.350000 "
            f"loo_sample_count={oracle['loo_sample_count']:.0f} "
            f"leaky_repeat_mean_oracle_recall_at_k={oracle['leaky_repeat_mean_oracle_recall_at_k']:.6f} "
            f"leaky_sample_count={oracle['leaky_sample_count']:.0f} "
            f"uses_current_repeat_for_gate=0"
        ),
    )

    entropy = compute_l23_video_tile_entropy_metrics(full, site_rows)
    entropy_ok = (
        entropy["valid_sample_count"] >= 20.0
        and entropy["topk_entropy_norm"] >= 0.35
        and entropy["topk_occupancy_fraction"] >= 0.20
        and entropy["active_tile_occupancy_fraction"] >= entropy["topk_occupancy_fraction"]
    )
    overall_ok &= print_result(
        entropy_ok,
        "l23_video_tile_entropy_occupancy",
        (
            f"tile_grid_side={entropy['tile_grid_side']:.0f} "
            f"tile_count={entropy['tile_count']:.0f} "
            f"topk_k={entropy['topk_k']:.0f} "
            f"valid_sample_count={entropy['valid_sample_count']:.0f} "
            f"topk_entropy_norm={entropy['topk_entropy_norm']:.6f} "
            f"entropy_threshold=0.350000 "
            f"topk_occupancy_fraction={entropy['topk_occupancy_fraction']:.6f} "
            f"occupancy_threshold=0.200000 "
            f"active_tile_occupancy_fraction={entropy['active_tile_occupancy_fraction']:.6f} "
            f"mean_active_tile_fraction={entropy['mean_active_tile_fraction']:.6f} "
            f"max_active_tile_fraction={entropy['max_active_tile_fraction']:.6f}"
        ),
    )

    ablation_runs: list[tuple[str, RunData | None]] = [
        ("full", full),
        ("control", control),
        ("somoff", somoff),
        ("recoff", recoff),
        ("pvoff", pvoff),
    ]
    ablation_parts: list[str] = []
    ablation_ok = True
    full_reliability_summary = {
        "same_different_gap": representational["same_different_gap"],
        "frame_top1_accuracy": representational["frame_top1_accuracy"],
        "loo_oracle": oracle["loo_no_leak_oracle_recall_at_k"],
        "tile_entropy_norm": entropy["topk_entropy_norm"],
        "topk_occupancy_fraction": entropy["topk_occupancy_fraction"],
        "mean_active_tile_fraction": entropy["mean_active_tile_fraction"],
        "max_active_tile_fraction": entropy["max_active_tile_fraction"],
    }
    ablation_summaries: dict[str, dict[str, float]] = {"full": full_reliability_summary}
    for label, run in ablation_runs:
        if run is None:
            ablation_parts.append(f"{label}_available=0")
            continue
        try:
            summary = full_reliability_summary if label == "full" else l23_video_reliability_summary_for_run(run)
        except ValidationError as exc:
            ablation_ok = False
            ablation_parts.append(f"{label}_available=0 {label}_error={str(exc).replace(' ', '_')}")
            continue
        if summary is None:
            ablation_parts.append(f"{label}_available=0")
            continue
        finite = all(math.isfinite(value) for value in summary.values())
        ablation_ok &= finite
        if finite:
            ablation_summaries[label] = summary
        ablation_parts.append(
            f"{label}_available=1 "
            f"{label}_same_diff_gap={summary['same_different_gap']:.6f} "
            f"{label}_frame_top1={summary['frame_top1_accuracy']:.6f} "
            f"{label}_loo_oracle={summary['loo_oracle']:.6f} "
            f"{label}_entropy={summary['tile_entropy_norm']:.6f} "
            f"{label}_occupancy={summary['topk_occupancy_fraction']:.6f} "
            f"{label}_mean_active_tile_fraction={summary['mean_active_tile_fraction']:.6f} "
            f"{label}_max_active_tile_fraction={summary['max_active_tile_fraction']:.6f}"
        )
    for label in ("control", "somoff", "recoff", "pvoff"):
        summary = ablation_summaries.get(label)
        if summary is None:
            ablation_parts.append(f"{label}_directional_density_integrity=unavailable")
            continue
        performance_better = (
            summary["loo_oracle"] > full_reliability_summary["loo_oracle"] + 0.03
            or summary["frame_top1_accuracy"] > full_reliability_summary["frame_top1_accuracy"] + 0.05
            or summary["same_different_gap"] > full_reliability_summary["same_different_gap"] + 0.05
        )
        # Catch near-doubling pathological densification without flagging modest control shifts.
        massive_density_increase = (
            summary["mean_active_tile_fraction"] > full_reliability_summary["mean_active_tile_fraction"] * 1.97
            and summary["mean_active_tile_fraction"] > full_reliability_summary["mean_active_tile_fraction"] + 0.20
        )
        dense_rescue = performance_better and massive_density_increase
        ablation_parts.append(
            f"{label}_directional_density_integrity={0 if dense_rescue else 1} "
            f"{label}_better_than_full={1 if performance_better else 0} "
            f"{label}_massive_density_increase={1 if massive_density_increase else 0}"
        )
    overall_ok &= print_result(
        ablation_ok,
        "l23_video_reliability_ablation_effects",
        (
            " ".join(ablation_parts)
            + " safety_caveat=separate_ablation_artifacts_descriptive_not_standalone_causal_proof"
        ),
    )

    metrics = full.hva_predictor_metrics
    smooth_keys = (
        "topk_heldout_repeat_avg_smooth_model_recall_at_k",
        "topk_heldout_repeat_avg_smooth_model_ndcg_at_k",
        "topk_heldout_repeat_avg_smooth_model_captured_ideal_mass_at_k",
    )
    if metrics is None:
        smooth_details = "hva_smoothed_metrics_available=0 hva_smoothed_metrics_unavailable_reason=missing_hva_predictor_metrics"
    else:
        smooth_details = (
            "hva_smoothed_metrics_available=1 "
            + " ".join(
                f"{key}={metrics.get(key, math.nan):.6f}"
                for key in smooth_keys
            )
        )
    anti_cheat_ok = oracle_ok
    overall_ok &= print_result(
        anti_cheat_ok,
        "l23_video_anti_cheat_separation",
        (
            f"raw_exact_gate=l23_video_raw_topk_repeat_oracle_ceiling "
            f"raw_exact_gate_pass={1 if oracle_ok else 0} "
            f"raw_exact_loo_oracle_recall_at_k={oracle['loo_no_leak_oracle_recall_at_k']:.6f} "
            f"raw_exact_rescued_by_smoothed_population_metrics=0 "
            f"smoothed_population_metrics_diagnostic_only=1 "
            f"{smooth_details}"
        ),
    )
    return overall_ok


def validate_l23_activity_reliability(
    full: RunData,
    min_frame_top1_accuracy: float | None,
    min_raw_oracle_at_k: float,
    min_raw_oracle_ceiling_fraction: float,
    min_l23e_repeat_corr: float,
    max_mean_active_tile_fraction: float,
    max_sample_active_tile_fraction: float,
) -> bool:
    overall_ok = True
    site_rows = full.video_site_rows
    frame_rows = full.video_frame_summary_rows
    artifacts_available = site_rows is not None and frame_rows is not None
    overall_ok &= print_result(
        artifacts_available,
        "l23_activity_reliability_artifacts_available",
        (
            f"video_site_rows={len(site_rows) if site_rows is not None else 0} "
            f"video_frame_summary_rows={len(frame_rows) if frame_rows is not None else 0}"
        ),
    )
    if not artifacts_available or site_rows is None or frame_rows is None:
        return overall_ok

    representational = compute_l23_video_representational_metrics(site_rows)
    historical_frame_margin_threshold = max(0.10, 5.0 * representational["frame_chance"])
    frame_margin_threshold = historical_frame_margin_threshold
    if min_frame_top1_accuracy is not None:
        frame_margin_threshold = max(frame_margin_threshold, min_frame_top1_accuracy)
    frame_gate_ok = (
        math.isfinite(representational["frame_top1_accuracy"])
        and representational["frame_top1_accuracy"] >= frame_margin_threshold
        and representational["decoded_count"] >= 20.0
    )
    overall_ok &= print_result(
        frame_gate_ok,
        "l23_activity_reliability_frame_decoding_gate",
        (
            f"frame_top1_accuracy={representational['frame_top1_accuracy']:.6f} "
            f"frame_top5_accuracy={representational['frame_top5_accuracy']:.6f} "
            f"frame_mean_rank={representational['frame_mean_rank']:.6f} "
            f"frame_chance={representational['frame_chance']:.6f} "
            f"frame_margin_threshold={frame_margin_threshold:.6f} "
            f"configured_min_frame_top1_accuracy={format_optional_float(min_frame_top1_accuracy)} "
            f"decoded_count={representational['decoded_count']:.0f}"
        ),
    )

    oracle = compute_l23_video_raw_topk_oracle_metrics(full, site_rows)
    raw_oracle = oracle["loo_no_leak_oracle_recall_at_k"]
    leaky_oracle = oracle["leaky_repeat_mean_oracle_recall_at_k"]
    raw_oracle_ceiling_fraction = (
        raw_oracle / leaky_oracle
        if math.isfinite(raw_oracle) and math.isfinite(leaky_oracle) and leaky_oracle > 0.0
        else math.nan
    )
    expected_topk_k = 5
    topk_k = oracle["topk_k"]
    topk_k_ok = math.isfinite(topk_k) and topk_k.is_integer() and int(topk_k) == expected_topk_k
    entropy = compute_l23_video_tile_entropy_metrics(full, site_rows)
    l23e_repeat_corr = pairwise_repeat_reliability(frame_rows, "l23e_rate_hz")
    activity_ok = (
        math.isfinite(raw_oracle)
        and raw_oracle >= min_raw_oracle_at_k
        and topk_k_ok
        and math.isfinite(raw_oracle_ceiling_fraction)
        and raw_oracle_ceiling_fraction >= min_raw_oracle_ceiling_fraction
        and l23e_repeat_corr is not None
        and math.isfinite(l23e_repeat_corr)
        and l23e_repeat_corr >= min_l23e_repeat_corr
        and math.isfinite(entropy["mean_active_tile_fraction"])
        and entropy["mean_active_tile_fraction"] <= max_mean_active_tile_fraction
        and math.isfinite(entropy["max_active_tile_fraction"])
        and entropy["max_active_tile_fraction"] <= max_sample_active_tile_fraction
        and oracle["loo_sample_count"] >= 20.0
        and entropy["valid_sample_count"] >= 20.0
    )
    overall_ok &= print_result(
        activity_ok,
        "l23_activity_raw_topk_repeat_stability",
        (
            f"repeat_count={oracle['repeat_count']:.0f} "
            f"frame_count={oracle['frame_count']:.0f} "
            f"tile_grid_side={oracle['tile_grid_side']:.0f} "
            f"tile_count={oracle['tile_count']:.0f} "
            f"topk_k={oracle['topk_k']:.0f} "
            f"expected_topk_k={expected_topk_k} "
            f"raw_oracle_at_k={raw_oracle:.6f} "
            f"min_raw_oracle_at_k={min_raw_oracle_at_k:.6f} "
            f"leaky_repeat_mean_oracle_recall_at_k={leaky_oracle:.6f} "
            f"raw_oracle_ceiling_fraction={raw_oracle_ceiling_fraction:.6f} "
            f"min_raw_oracle_ceiling_fraction={min_raw_oracle_ceiling_fraction:.6f} "
            f"l23e_repeat_corr={format_optional_float(l23e_repeat_corr)} "
            f"min_l23e_repeat_corr={min_l23e_repeat_corr:.6f} "
            f"mean_active_tile_fraction={entropy['mean_active_tile_fraction']:.6f} "
            f"max_mean_active_tile_fraction={max_mean_active_tile_fraction:.6f} "
            f"max_sample_active_tile_fraction={entropy['max_active_tile_fraction']:.6f} "
            f"allowed_max_sample_active_tile_fraction={max_sample_active_tile_fraction:.6f} "
            f"loo_sample_count={oracle['loo_sample_count']:.0f} "
            f"valid_sample_count={entropy['valid_sample_count']:.0f}"
        ),
    )

    milestone_ok = activity_ok and topk_k_ok and min_raw_oracle_at_k >= 0.60
    overall_ok &= print_result(
        milestone_ok,
        "raw_oracle_0p6_milestone",
        (
            f"raw_oracle_at_k={raw_oracle:.6f} "
            "threshold=0.600000 "
            f"min_raw_oracle_at_k={min_raw_oracle_at_k:.6f} "
            f"topk_k={topk_k:.0f} "
            f"expected_topk_k={expected_topk_k} "
            f"repeat_count={oracle['repeat_count']:.0f} "
            f"sample_count={oracle['loo_sample_count']:.0f}"
        ),
    )

    metrics = full.hva_predictor_metrics
    if metrics is None:
        smooth_details = "hva_smoothed_metrics_available=0 hva_smoothed_metrics_unavailable_reason=missing_hva_predictor_metrics"
    else:
        smooth_details = (
            "hva_smoothed_metrics_available=1 "
            f"topk_heldout_repeat_avg_smooth_model_recall_at_k="
            f"{metrics.get('topk_heldout_repeat_avg_smooth_model_recall_at_k', math.nan):.6f}"
        )
    overall_ok &= print_result(
        activity_ok,
        "l23_activity_anti_cheat_separation",
        (
            "raw_exact_gate=l23_activity_raw_topk_repeat_stability "
            f"raw_exact_gate_pass={1 if activity_ok else 0} "
            f"raw_exact_oracle_at_k={raw_oracle:.6f} "
            "raw_exact_rescued_by_frame_decoding_or_smoothed_population_metrics=0 "
            "frame_decoding_and_smoothed_population_metrics_diagnostic_only=1 "
            f"{smooth_details}"
        ),
    )
    return overall_ok


def parse_video_ff_event_trace_edges_csv(path: Path) -> list[dict[str, float]]:
    required_columns = {
        "pre_l4e_id",
        "post_l23e_id",
        "distance_sites",
        "w_before",
        "w_after",
        "delta_w",
        "pre_before_post_event_count",
        "post_before_pre_event_count",
        "event_causal_score",
        "shuffle_causal_score",
        "pre_rate_hz",
        "post_rate_hz",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"Missing event-trace edge audit header in {path}")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValidationError(f"Missing event-trace edge audit columns in {path}: {sorted(missing)}")
        rows: list[dict[str, float]] = []
        for row_number, row in enumerate(reader, start=2):
            parsed: dict[str, float] = {}
            for column in required_columns:
                raw = row.get(column)
                if raw is None:
                    raise ValidationError(f"Missing value in {path} row {row_number} column {column}")
                parsed[column] = parse_float(raw, path, row_number, column)
            rows.append(parsed)
    return rows


def validate_event_driven_ff_plasticity(run: RunData) -> bool:
    """Require local event-trace FF plasticity, not host-side windowed count updates."""

    overall_ok = True

    def event_metric(name: str) -> float | None:
        summary_value = optional_summary_metric(run, f"video_ff_event_trace_{name}")
        if summary_value is not None:
            return summary_value
        if run.video_consolidation_metrics is None:
            return None
        return run.video_consolidation_metrics.get(f"feedforward_l4_l23_event_trace_{name}")

    enabled = event_metric("enabled")
    tau_pre_ms = event_metric("tau_pre_ms")
    tau_post_ms = event_metric("tau_post_ms")
    tau_rate_ms = event_metric("tau_rate_ms")
    hetero_minus = event_metric("hetero_minus")
    post_target_hz = event_metric("post_target_hz")
    application_count = event_metric("application_count")
    active_edge_count = event_metric("active_edge_count")
    changed_frac = event_metric("changed_frac")
    mean_delta = event_metric("mean_delta")
    p95_abs_delta = event_metric("p95_abs_delta")
    max_abs_delta = event_metric("max_abs_delta")
    mean_gain_ratio = event_metric("mean_gain_ratio")
    local_only = event_metric("local_only")
    windowed_count_only = event_metric("windowed_count_only")
    future_frame_used = event_metric("future_frame_used")
    target_label_used = event_metric("target_label_used")
    heldout_frames_used = event_metric("heldout_frames_used")
    hva_feedback_enabled = event_metric("hva_feedback_enabled")

    summary_ok = (
        enabled == 1.0
        and tau_pre_ms is not None
        and 15.0 <= tau_pre_ms <= 25.0
        and tau_post_ms is not None
        and 30.0 <= tau_post_ms <= 50.0
        and tau_rate_ms is not None
        and 1000.0 <= tau_rate_ms <= 5000.0
        and hetero_minus is not None
        and hetero_minus > 0.0
        and post_target_hz is not None
        and post_target_hz >= 0.0
        and application_count is not None
        and application_count > 0.0
        and active_edge_count is not None
        and active_edge_count >= 100.0
        and changed_frac is not None
        and changed_frac >= 0.01
        and p95_abs_delta is not None
        and p95_abs_delta > 1.0e-6
        and max_abs_delta is not None
        and max_abs_delta > 1.0e-6
    )
    overall_ok &= print_result(
        summary_ok,
        "event_driven_ff_plasticity_summary",
        (
            f"enabled={format_optional_float(enabled)} "
            f"tau_pre_ms={format_optional_float(tau_pre_ms)} "
            f"tau_post_ms={format_optional_float(tau_post_ms)} "
            f"tau_rate_ms={format_optional_float(tau_rate_ms)} "
            f"hetero_minus={format_optional_float(hetero_minus)} "
            f"post_target_hz={format_optional_float(post_target_hz)} "
            f"application_count={format_optional_float(application_count)} "
            f"active_edge_count={format_optional_float(active_edge_count)} "
            f"changed_frac={format_optional_float(changed_frac)} "
            f"mean_delta={format_optional_float(mean_delta)} "
            f"p95_abs_delta={format_optional_float(p95_abs_delta)} "
            f"max_abs_delta={format_optional_float(max_abs_delta)} "
            f"mean_gain_ratio={format_optional_float(mean_gain_ratio)}"
        ),
    )

    no_cheat_ok = (
        local_only == 1.0
        and windowed_count_only == 0.0
        and future_frame_used == 0.0
        and target_label_used == 0.0
        and heldout_frames_used == 0.0
        and hva_feedback_enabled == 0.0
    )
    overall_ok &= print_result(
        no_cheat_ok,
        "event_driven_ff_plasticity_no_cheat",
        (
            f"local_only={format_optional_float(local_only)} "
            f"windowed_count_only={format_optional_float(windowed_count_only)} "
            f"future_frame_used={format_optional_float(future_frame_used)} "
            f"target_label_used={format_optional_float(target_label_used)} "
            f"heldout_frames_used={format_optional_float(heldout_frames_used)} "
            f"hva_feedback_enabled={format_optional_float(hva_feedback_enabled)}"
        ),
    )

    coactivity_enabled = optional_summary_metric(run, "video_ff_coactivity_competition_enabled")
    weight_only_enabled = optional_summary_metric(run, "video_ff_heterosynaptic_competition_enabled")
    legacy_disabled_ok = (
        (coactivity_enabled is None or coactivity_enabled == 0.0)
        and (weight_only_enabled is None or weight_only_enabled == 0.0)
    )
    overall_ok &= print_result(
        legacy_disabled_ok,
        "event_driven_ff_plasticity_no_windowed_counts",
        (
            f"video_ff_coactivity_competition_enabled={format_optional_float(coactivity_enabled)} "
            f"video_ff_heterosynaptic_competition_enabled={format_optional_float(weight_only_enabled)} "
            "required_old_windowed_and_weight_only_paths_disabled=1"
        ),
    )

    mass_post_count = event_metric("incoming_mass_post_count")
    mass_min_ratio = event_metric("incoming_mass_min_ratio")
    mass_mean_ratio = event_metric("incoming_mass_mean_ratio")
    mass_max_ratio = event_metric("incoming_mass_max_ratio")
    mass_p95_abs_log_ratio = event_metric("incoming_mass_p95_abs_log_ratio")
    mass_ok = (
        mass_post_count is not None
        and mass_post_count >= 100.0
        and mass_min_ratio is not None
        and mass_min_ratio >= 0.75
        and mass_mean_ratio is not None
        and 0.85 <= mass_mean_ratio <= 1.15
        and mass_max_ratio is not None
        and mass_max_ratio <= 1.25
        and mass_p95_abs_log_ratio is not None
        and mass_p95_abs_log_ratio <= math.log(1.25)
    )
    overall_ok &= print_result(
        mass_ok,
        "event_driven_ff_plasticity_incoming_mass",
        (
            f"post_count={format_optional_float(mass_post_count)} "
            f"min_ratio={format_optional_float(mass_min_ratio)} "
            f"mean_ratio={format_optional_float(mass_mean_ratio)} "
            f"max_ratio={format_optional_float(mass_max_ratio)} "
            f"p95_abs_log_ratio={format_optional_float(mass_p95_abs_log_ratio)}"
        ),
    )

    edge_path = run.genn_dir / f"{run.prefix}_video_ff_event_trace_edges.csv"
    rows = parse_video_ff_event_trace_edges_csv(require_file(edge_path))
    positive_rows = [
        row
        for row in rows
        if row["delta_w"] > 0.0
        and math.isfinite(row["event_causal_score"])
        and math.isfinite(row["shuffle_causal_score"])
    ]
    mean_event_score = mean([row["event_causal_score"] for row in positive_rows]) if positive_rows else math.nan
    mean_shuffle_score = mean([row["shuffle_causal_score"] for row in positive_rows]) if positive_rows else math.nan
    mean_positive_delta = mean([row["delta_w"] for row in positive_rows]) if positive_rows else math.nan
    edge_ok = (
        len(rows) >= 128
        and len(positive_rows) >= 20
        and math.isfinite(mean_event_score)
        and math.isfinite(mean_shuffle_score)
        and mean_event_score > mean_shuffle_score
    )
    overall_ok &= print_result(
        edge_ok,
        "event_driven_ff_plasticity_edge_audit",
        (
            f"edge_csv={edge_path} "
            f"rows={len(rows)} "
            f"positive_delta_rows={len(positive_rows)} "
            f"mean_positive_delta={mean_positive_delta:.6f} "
            f"mean_event_causal_score={mean_event_score:.6f} "
            f"mean_shuffle_causal_score={mean_shuffle_score:.6f} "
            "required_event_score_gt_shuffle=1"
        ),
    )

    return overall_ok


def validate_emergent_ff_gain(run: RunData) -> bool:
    """Validate that video FF gain is learned, not a transient eval-time scale."""

    overall_ok = True
    video_replay_enabled = optional_summary_metric(run, "video_replay_enabled")
    ff_tuning_enabled = optional_summary_metric(run, "video_ff_reliability_tuning_enabled")
    ff_output_scale = optional_summary_metric(run, "video_ff_reliability_l4e_l23e_output_scale")
    frozen_eval_ok = (
        video_replay_enabled == 1.0
        and (
            ff_tuning_enabled == 0.0
            or (ff_output_scale is not None and abs(ff_output_scale - 1.0) <= 1.0e-9)
        )
    )
    overall_ok &= print_result(
        frozen_eval_ok,
        "l23_video_ff_eval_no_transient_gain",
        (
            f"video_replay_enabled={format_optional_float(video_replay_enabled)} "
            f"video_ff_reliability_tuning_enabled={format_optional_float(ff_tuning_enabled)} "
            f"video_ff_reliability_l4e_l23e_output_scale={format_optional_float(ff_output_scale)} "
            "required_tuning_disabled_or_scale_one=1"
        ),
    )

    video_ff_stdp_enabled = optional_summary_metric(run, "video_ff_stdp_enabled")
    present_only_values = summary_metric_values(
        run,
        (
            "video_ff_stdp_present_frame_drive_only",
            "video_ff_homeostatic_scaling_present_frame_drive_only",
            "video_ff_homeostatic_present_frame_drive_only",
            "video_ff_present_frame_drive_only",
        ),
    )
    future_frame_used_values = summary_metric_values(
        run,
        (
            "video_ff_stdp_future_frame_used",
            "video_ff_homeostatic_scaling_future_frame_used",
            "video_ff_homeostatic_future_frame_used",
            "video_ff_future_frame_used",
        ),
    )
    target_label_used_values = summary_metric_values(
        run,
        (
            "video_ff_stdp_target_label_used",
            "video_ff_homeostatic_scaling_target_label_used",
            "video_ff_homeostatic_target_label_used",
            "video_ff_target_label_used",
        ),
    )
    heldout_frames_used_values = summary_metric_values(
        run,
        (
            "video_ff_stdp_heldout_frames_used",
            "video_ff_homeostatic_scaling_heldout_frames_used",
            "video_ff_homeostatic_heldout_frames_used",
            "video_ff_heldout_frames_used",
        ),
    )
    hva_feedback_enabled_values = summary_metric_values(
        run,
        (
            "video_ff_stdp_hva_feedback_enabled",
            "video_ff_homeostatic_scaling_hva_feedback_enabled",
            "video_ff_homeostatic_hva_feedback_enabled",
            "video_ff_hva_feedback_enabled",
        ),
    )
    optional_false_flags_ok = all(
        value == 0.0
        for _, value in (
            future_frame_used_values
            + target_label_used_values
            + heldout_frames_used_values
            + hva_feedback_enabled_values
        )
    )
    present_only_ok = all(value == 1.0 for _, value in present_only_values)
    exposure_audit_ok = (
        video_ff_stdp_enabled == 1.0
        and present_only_ok
        and optional_false_flags_ok
    )
    overall_ok &= print_result(
        exposure_audit_ok,
        "l23_video_ff_plastic_exposure_audit",
        (
            f"video_ff_stdp_enabled={format_optional_float(video_ff_stdp_enabled)} "
            f"video_ff_present_frame_drive_only_values={format_metric_values(present_only_values)} "
            f"video_ff_future_frame_used_values={format_metric_values(future_frame_used_values)} "
            f"video_ff_target_label_used_values={format_metric_values(target_label_used_values)} "
            f"video_ff_heldout_frames_used_values={format_metric_values(heldout_frames_used_values)} "
            f"video_ff_hva_feedback_enabled_values={format_metric_values(hva_feedback_enabled_values)}"
        ),
    )

    min_active_count = 100.0
    min_changed_fraction = 0.01
    min_p95_abs_delta = 1.0e-6
    min_max_abs_delta = 1.0e-6

    def read_ff_delta_evidence(
        label: str,
        *,
        active_aliases: tuple[str, ...],
        changed_aliases: tuple[str, ...],
        p95_aliases: tuple[str, ...],
        max_aliases: tuple[str, ...],
        mean_aliases: tuple[str, ...],
        gain_aliases: tuple[str, ...],
    ) -> dict[str, float | str | None]:
        active_metric, active_count = first_summary_metric(run, active_aliases)
        changed_metric, changed_fraction = first_summary_metric(run, changed_aliases)
        p95_metric, p95_abs_delta = first_summary_metric(run, p95_aliases)
        max_metric, max_abs_delta = first_summary_metric(run, max_aliases)
        mean_metric, mean_delta = first_summary_metric(run, mean_aliases)
        gain_metric, mean_gain_ratio = first_summary_metric(run, gain_aliases)
        direction_ok = (
            (mean_gain_ratio is not None and mean_gain_ratio > 1.0)
            or (mean_delta is not None and mean_delta > 0.0)
        )
        metrics_present = all(value is not None for value in (changed_fraction, p95_abs_delta, max_abs_delta))
        active_ok = active_count is None or active_count >= min_active_count
        thresholds_ok = (
            metrics_present
            and changed_fraction is not None
            and p95_abs_delta is not None
            and max_abs_delta is not None
            and active_ok
            and changed_fraction >= min_changed_fraction
            and p95_abs_delta >= min_p95_abs_delta
            and max_abs_delta >= min_max_abs_delta
        )
        return {
            "label": label,
            "active_metric": active_metric,
            "active_count": active_count,
            "changed_metric": changed_metric,
            "changed_fraction": changed_fraction,
            "p95_metric": p95_metric,
            "p95_abs_delta": p95_abs_delta,
            "max_metric": max_metric,
            "max_abs_delta": max_abs_delta,
            "mean_metric": mean_metric,
            "mean_delta": mean_delta,
            "gain_metric": gain_metric,
            "mean_gain_ratio": mean_gain_ratio,
            "direction_ok": 1.0 if direction_ok else 0.0,
            "thresholds_ok": 1.0 if thresholds_ok else 0.0,
        }

    stdp_evidence = read_ff_delta_evidence(
        "stdp",
        active_aliases=(
            "video_ff_stdp_l4_l23_active_count",
            "video_ff_stdp_l4e_l23e_active_count",
            "video_ff_stdp_active_edge_count",
        ),
        changed_aliases=(
            "video_ff_stdp_l4_l23_changed_frac",
            "video_ff_stdp_l4e_l23e_changed_frac",
            "feedforward_l4_l23_changed_frac",
        ),
        p95_aliases=(
            "video_ff_stdp_l4_l23_p95_abs_delta",
            "video_ff_stdp_l4e_l23e_p95_abs_delta",
            "feedforward_l4_l23_p95_abs_delta",
        ),
        max_aliases=(
            "video_ff_stdp_l4_l23_max_abs_delta",
            "video_ff_stdp_l4e_l23e_max_abs_delta",
            "feedforward_l4_l23_max_abs_delta",
        ),
        mean_aliases=(
            "video_ff_stdp_l4_l23_mean_delta",
            "video_ff_stdp_l4e_l23e_mean_delta",
            "feedforward_l4_l23_mean_delta",
        ),
        gain_aliases=(
            "video_ff_stdp_l4_l23_mean_gain_ratio",
            "video_ff_stdp_l4e_l23e_mean_gain_ratio",
            "feedforward_l4_l23_mean_gain_ratio",
        ),
    )
    homeostatic_evidence = read_ff_delta_evidence(
        "homeostatic",
        active_aliases=(
            "video_ff_homeostatic_scaling_active_edge_count",
            "video_ff_homeostatic_scaling_active_count",
            "video_ff_homeostatic_scaling_l4_l23_active_count",
            "video_ff_homeostatic_l4_l23_active_count",
            "video_ff_homeostatic_active_edge_count",
            "video_ff_homeostatic_active_count",
        ),
        changed_aliases=(
            "video_ff_homeostatic_scaling_changed_frac",
            "video_ff_homeostatic_scaling_l4_l23_changed_frac",
            "video_ff_homeostatic_l4_l23_changed_frac",
            "video_ff_homeostatic_changed_frac",
        ),
        p95_aliases=(
            "video_ff_homeostatic_scaling_p95_abs_delta",
            "video_ff_homeostatic_scaling_l4_l23_p95_abs_delta",
            "video_ff_homeostatic_l4_l23_p95_abs_delta",
            "video_ff_homeostatic_p95_abs_delta",
        ),
        max_aliases=(
            "video_ff_homeostatic_scaling_max_abs_delta",
            "video_ff_homeostatic_scaling_l4_l23_max_abs_delta",
            "video_ff_homeostatic_l4_l23_max_abs_delta",
            "video_ff_homeostatic_max_abs_delta",
        ),
        mean_aliases=(
            "video_ff_homeostatic_scaling_mean_delta",
            "video_ff_homeostatic_scaling_l4_l23_mean_delta",
            "video_ff_homeostatic_l4_l23_mean_delta",
            "video_ff_homeostatic_mean_delta",
        ),
        gain_aliases=(
            "video_ff_homeostatic_scaling_mean_gain_ratio",
            "video_ff_homeostatic_scaling_l4_l23_mean_gain_ratio",
            "video_ff_homeostatic_l4_l23_mean_gain_ratio",
            "video_ff_homeostatic_mean_gain_ratio",
        ),
    )
    combined_evidence = read_ff_delta_evidence(
        "combined",
        active_aliases=(
            "video_ff_post_exposure_l4_l23_active_count",
            "video_ff_post_exposure_active_count",
            "video_ff_combined_l4_l23_active_count",
            "video_ff_total_l4_l23_active_count",
            "video_ff_l4_l23_weight_delta_active_count",
            "video_ff_l4e_l23e_weight_delta_active_count",
            "video_ff_l4_l23_active_count",
            "video_ff_l4e_l23e_active_count",
            "video_ff_exposure_weight_delta_active_count",
        ),
        changed_aliases=(
            "video_ff_post_exposure_l4_l23_changed_frac",
            "video_ff_post_exposure_changed_frac",
            "video_ff_combined_l4_l23_changed_frac",
            "video_ff_total_l4_l23_changed_frac",
            "video_ff_l4_l23_weight_delta_changed_frac",
            "video_ff_l4e_l23e_weight_delta_changed_frac",
            "video_ff_l4_l23_changed_frac",
            "video_ff_l4e_l23e_changed_frac",
            "video_ff_exposure_weight_delta_changed_frac",
        ),
        p95_aliases=(
            "video_ff_post_exposure_l4_l23_p95_abs_delta",
            "video_ff_post_exposure_p95_abs_delta",
            "video_ff_combined_l4_l23_p95_abs_delta",
            "video_ff_total_l4_l23_p95_abs_delta",
            "video_ff_l4_l23_weight_delta_p95_abs",
            "video_ff_l4_l23_weight_delta_p95_abs_delta",
            "video_ff_l4e_l23e_weight_delta_p95_abs",
            "video_ff_l4e_l23e_weight_delta_p95_abs_delta",
            "video_ff_l4_l23_p95_abs_delta",
            "video_ff_l4e_l23e_p95_abs_delta",
            "video_ff_exposure_weight_delta_p95_abs",
        ),
        max_aliases=(
            "video_ff_post_exposure_l4_l23_max_abs_delta",
            "video_ff_post_exposure_max_abs_delta",
            "video_ff_combined_l4_l23_max_abs_delta",
            "video_ff_total_l4_l23_max_abs_delta",
            "video_ff_l4_l23_weight_delta_max_abs",
            "video_ff_l4_l23_weight_delta_max_abs_delta",
            "video_ff_l4e_l23e_weight_delta_max_abs",
            "video_ff_l4e_l23e_weight_delta_max_abs_delta",
            "video_ff_l4_l23_max_abs_delta",
            "video_ff_l4e_l23e_max_abs_delta",
            "video_ff_exposure_weight_delta_max_abs",
        ),
        mean_aliases=(
            "video_ff_post_exposure_l4_l23_mean_delta",
            "video_ff_post_exposure_mean_delta",
            "video_ff_combined_l4_l23_mean_delta",
            "video_ff_total_l4_l23_mean_delta",
            "video_ff_l4_l23_weight_delta_mean",
            "video_ff_l4_l23_weight_delta_mean_delta",
            "video_ff_l4e_l23e_weight_delta_mean",
            "video_ff_l4e_l23e_weight_delta_mean_delta",
            "video_ff_l4_l23_mean_delta",
            "video_ff_l4e_l23e_mean_delta",
            "video_ff_exposure_weight_delta_mean",
        ),
        gain_aliases=(
            "video_ff_post_exposure_l4_l23_mean_gain_ratio",
            "video_ff_post_exposure_mean_gain_ratio",
            "video_ff_combined_l4_l23_mean_gain_ratio",
            "video_ff_total_l4_l23_mean_gain_ratio",
            "video_ff_l4_l23_weight_mean_gain_ratio",
            "video_ff_l4e_l23e_weight_mean_gain_ratio",
            "video_ff_l4_l23_mean_gain_ratio",
            "video_ff_l4e_l23e_mean_gain_ratio",
            "video_ff_exposure_mean_gain_ratio",
        ),
    )

    evidence_sources = (combined_evidence, homeostatic_evidence, stdp_evidence)
    positive_sources = [
        evidence
        for evidence in evidence_sources
        if evidence["thresholds_ok"] == 1.0 and evidence["direction_ok"] == 1.0
    ]
    selected_evidence = positive_sources[0] if positive_sources else evidence_sources[0]
    learned_gain_direction_ok = bool(positive_sources)
    weight_delta_ok = learned_gain_direction_ok
    overall_ok &= print_result(
        weight_delta_ok,
        "l23_video_ff_learned_gain",
        (
            f"learned_gain_source={selected_evidence['label']} "
            f"active_metric={selected_evidence['active_metric'] or 'missing'} "
            f"active_count={format_optional_float(selected_evidence['active_count'])} "
            f"min_active_count={min_active_count:.0f} "
            f"changed_fraction_metric={selected_evidence['changed_metric'] or 'missing'} "
            f"changed_fraction={format_optional_float(selected_evidence['changed_fraction'])} "
            f"min_changed_fraction={min_changed_fraction:.6f} "
            f"p95_abs_delta_metric={selected_evidence['p95_metric'] or 'missing'} "
            f"p95_abs_delta={format_optional_float(selected_evidence['p95_abs_delta'])} "
            f"min_p95_abs_delta={min_p95_abs_delta:.6e} "
            f"max_abs_delta_metric={selected_evidence['max_metric'] or 'missing'} "
            f"max_abs_delta={format_optional_float(selected_evidence['max_abs_delta'])} "
            f"min_max_abs_delta={min_max_abs_delta:.6e} "
            f"mean_delta_metric={selected_evidence['mean_metric'] or 'missing'} "
            f"mean_delta={format_optional_float(selected_evidence['mean_delta'])} "
            f"mean_gain_ratio_metric={selected_evidence['gain_metric'] or 'missing'} "
            f"mean_gain_ratio={format_optional_float(selected_evidence['mean_gain_ratio'])} "
            f"stdp_mean_delta={format_optional_float(stdp_evidence['mean_delta'])} "
            f"stdp_mean_gain_ratio={format_optional_float(stdp_evidence['mean_gain_ratio'])} "
            f"homeostatic_mean_delta={format_optional_float(homeostatic_evidence['mean_delta'])} "
            f"homeostatic_mean_gain_ratio={format_optional_float(homeostatic_evidence['mean_gain_ratio'])} "
            f"combined_mean_delta={format_optional_float(combined_evidence['mean_delta'])} "
            f"combined_mean_gain_ratio={format_optional_float(combined_evidence['mean_gain_ratio'])} "
            f"learned_gain_direction_ok={int(learned_gain_direction_ok)}"
        ),
    )
    return overall_ok


def compute_recurrent_video_metrics(
    site_rows: list[VideoSiteRateRow],
    specificity_rows: list[SpecificityRow] | None,
) -> dict[str, float]:
    if specificity_rows is None:
        raise ValidationError("Natural-video recurrent metrics require l23ee specificity rows.")
    site_vectors = l23e_site_frame_vectors(site_rows)
    edge_metrics: list[tuple[float, float, float, bool]] = []
    for row in specificity_rows:
        if row.w_after <= 0.0:
            continue
        pre_vector = site_vectors.get(row.pre_site)
        post_vector = site_vectors.get(row.post_site)
        if pre_vector is None or post_vector is None:
            continue
        corr = pearson_correlation_optional(pre_vector, post_vector)
        if corr is None or not math.isfinite(corr):
            continue
        edge_metrics.append((corr, row.w_after, row.delta_pref_deg, row.pre_site == row.post_site))

    if not edge_metrics:
        return {
            "edge_count": 0.0,
            "mean_corr": math.nan,
            "median_corr": math.nan,
            "top10_weight_mean_corr": math.nan,
            "low_delta_mean_corr": math.nan,
            "same_site_fraction": math.nan,
        }

    corr_values = [item[0] for item in edge_metrics]
    top_count = max(1, int(math.ceil(0.10 * len(edge_metrics))))
    top_by_weight = sorted(edge_metrics, key=lambda item: item[1], reverse=True)[:top_count]
    delta_p25 = percentile([item[2] for item in edge_metrics], 25.0)
    low_delta = [item for item in edge_metrics if item[2] <= delta_p25]
    return {
        "edge_count": float(len(edge_metrics)),
        "mean_corr": mean(corr_values),
        "median_corr": median(corr_values),
        "top10_weight_mean_corr": mean([item[0] for item in top_by_weight]),
        "low_delta_mean_corr": mean([item[0] for item in low_delta]),
        "same_site_fraction": sum(1 for item in edge_metrics if item[3]) / len(edge_metrics),
    }


def mean_event_bin_series(
    rows: list[VideoEventBinRow],
    condition: str,
    population: str,
) -> list[tuple[int, float, float, float]]:
    values_by_bin: dict[int, list[float]] = {}
    starts_by_bin: dict[int, float] = {}
    ends_by_bin: dict[int, float] = {}
    for row in rows:
        if row.condition != condition or row.population != population:
            continue
        values_by_bin.setdefault(row.bin_index, []).append(row.rate_hz)
        starts_by_bin[row.bin_index] = row.bin_start_ms
        ends_by_bin[row.bin_index] = row.bin_end_ms
    return [
        (bin_index, starts_by_bin[bin_index], ends_by_bin[bin_index], mean(values_by_bin[bin_index]))
        for bin_index in sorted(values_by_bin)
    ]


def matched_event_minus_control_series(
    rows: list[VideoEventBinRow],
    population: str,
    control_condition: str,
) -> list[tuple[int, float, float, float]]:
    event_by_key: dict[tuple[int, int, int], VideoEventBinRow] = {}
    control_by_key: dict[tuple[int, int, int], VideoEventBinRow] = {}
    for row in rows:
        if row.population != population:
            continue
        key = (row.repeat_index, row.event_index, row.bin_index)
        if row.condition == "event":
            event_by_key[key] = row
        elif row.condition == control_condition:
            control_by_key[key] = row

    values_by_bin: dict[int, list[float]] = {}
    starts_by_bin: dict[int, float] = {}
    ends_by_bin: dict[int, float] = {}
    for key in sorted(set(event_by_key).intersection(control_by_key)):
        event_row = event_by_key[key]
        control_row = control_by_key[key]
        values_by_bin.setdefault(event_row.bin_index, []).append(event_row.rate_hz - control_row.rate_hz)
        starts_by_bin[event_row.bin_index] = event_row.bin_start_ms
        ends_by_bin[event_row.bin_index] = event_row.bin_end_ms

    return [
        (bin_index, starts_by_bin[bin_index], ends_by_bin[bin_index], mean(values_by_bin[bin_index]))
        for bin_index in sorted(values_by_bin)
    ]


def event_response_metrics(series: list[tuple[int, float, float, float]]) -> dict[str, float]:
    baseline = [rate for _, start_ms, _, rate in series if start_ms < 0.0]
    post = [(start_ms, rate) for _, start_ms, _, rate in series if start_ms >= 0.0]
    if not baseline or not post:
        return {
            "baseline_mean": math.nan,
            "baseline_std": math.nan,
            "post_peak": math.nan,
            "post_mean": math.nan,
            "onset_latency_ms": math.nan,
            "peak_latency_ms": math.nan,
            "pre_bin_count": float(len(baseline)),
            "post_bin_count": float(len(post)),
        }

    baseline_mean = mean(baseline)
    baseline_std = standard_deviation(baseline) if len(baseline) > 1 else 0.0
    post_rates = [rate for _, rate in post]
    post_peak = max(post_rates)
    peak_index = post_rates.index(post_peak)
    peak_latency_ms = post[peak_index][0]
    threshold = baseline_mean + max(
        2.0 * baseline_std,
        0.05 * max(0.0, post_peak - baseline_mean),
        1.0e-9,
    )
    onset_latency_ms = math.nan
    for start_ms, rate in post:
        if rate >= threshold:
            onset_latency_ms = start_ms
            break
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "post_peak": post_peak,
        "post_mean": mean(post_rates),
        "onset_latency_ms": onset_latency_ms,
        "peak_latency_ms": peak_latency_ms,
        "pre_bin_count": float(len(baseline)),
        "post_bin_count": float(len(post)),
    }


def event_best_lag_correlation(
    source_series: list[tuple[int, float, float, float]],
    target_series: list[tuple[int, float, float, float]],
    max_lag_ms: float,
) -> dict[str, float | None]:
    if len(source_series) != len(target_series) or len(source_series) < 3:
        return {"best_lag_ms": None, "best_corr": None, "lag0_corr": None, "shifted_null_corr": None}
    source = [row[3] for row in source_series]
    target = [row[3] for row in target_series]
    bin_ms = source_series[0][2] - source_series[0][1]
    max_lag_bins = max(0, min(len(source) - 2, int(math.floor(max_lag_ms / max(bin_ms, 1.0e-9)))))
    correlations = lagged_correlations(source, target, max_lag_bins)
    lag, corr = best_lag(correlations)
    shift = max(1, len(source) // 2)
    shifted_source = source[shift:] + source[:shift]
    shifted_null_corr = pearson_correlation_optional(shifted_source, target)
    return {
        "best_lag_ms": (float(lag) * bin_ms) if lag is not None else None,
        "best_corr": corr,
        "lag0_corr": correlations.get(0),
        "shifted_null_corr": shifted_null_corr,
    }


_HVA_POPULATION_SMOOTH_KERNEL_R1: tuple[tuple[float, ...], ...] = (
    (1.0, 2.0, 1.0),
    (2.0, 4.0, 2.0),
    (1.0, 2.0, 1.0),
)

_HVA_POPULATION_SCORERS: tuple[str, ...] = (
    "model",
    "persistence",
    "train_frequency",
    "no_learning",
    "time_shuffle",
    "spatial_shuffle",
)


def _hva_population_score(row: HVAPredictorPredictionRow, scorer: str) -> float:
    if scorer == "model":
        return row.topk_model_score
    if scorer == "persistence":
        return row.topk_persistence_score
    if scorer == "train_frequency":
        return row.topk_train_frequency_score
    if scorer == "no_learning":
        return row.topk_no_learning_score
    if scorer == "time_shuffle":
        return row.topk_temporal_block_shift_score
    if scorer == "spatial_shuffle":
        return row.topk_spatial_tile_shuffle_score
    raise ValidationError(f"Unknown HVA population scorer: {scorer}")


def _smooth_hva_population_target(values: list[float], grid_side: int) -> list[float]:
    smoothed: list[float] = []
    for tile_id in range(len(values)):
        tile_x = tile_id % grid_side
        tile_y = tile_id // grid_side
        weighted_sum = 0.0
        weight_sum = 0.0
        for kernel_y, kernel_row in enumerate(_HVA_POPULATION_SMOOTH_KERNEL_R1):
            offset_y = kernel_y - 1
            source_y = tile_y + offset_y
            if source_y < 0 or source_y >= grid_side:
                continue
            for kernel_x, kernel_weight in enumerate(kernel_row):
                offset_x = kernel_x - 1
                source_x = tile_x + offset_x
                if source_x < 0 or source_x >= grid_side:
                    continue
                weighted_sum += kernel_weight * values[source_y * grid_side + source_x]
                weight_sum += kernel_weight
        smoothed.append(weighted_sum / weight_sum if weight_sum > 0.0 else values[tile_id])
    return smoothed


def _ranked_tile_ids(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda tile_id: (-scores[tile_id], tile_id))


def _weighted_ndcg_at_k(relevance: list[float], scores: list[float], k: int) -> float | None:
    if k <= 0 or not relevance:
        return None
    cutoff = min(k, len(relevance))
    ideal_order = _ranked_tile_ids(relevance)
    ideal_dcg = 0.0
    for rank, tile_id in enumerate(ideal_order[:cutoff]):
        ideal_dcg += relevance[tile_id] / math.log2(rank + 2.0)
    if ideal_dcg <= 1.0e-12:
        return None

    scored_order = _ranked_tile_ids(scores)
    dcg = 0.0
    for rank, tile_id in enumerate(scored_order[:cutoff]):
        dcg += relevance[tile_id] / math.log2(rank + 2.0)
    return dcg / ideal_dcg


def _captured_ideal_mass_at_k(relevance: list[float], scores: list[float], k: int) -> float | None:
    if k <= 0 or not relevance:
        return None
    cutoff = min(k, len(relevance))
    ideal_order = _ranked_tile_ids(relevance)
    ideal_mass = sum(relevance[tile_id] for tile_id in ideal_order[:cutoff])
    if ideal_mass <= 1.0e-12:
        return None
    scored_order = _ranked_tile_ids(scores)
    captured_mass = sum(relevance[tile_id] for tile_id in scored_order[:cutoff])
    return captured_mass / ideal_mass


def _safe_metric_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1.0e-12:
        return math.inf if numerator > 1.0e-12 else 1.0
    return numerator / denominator


def compute_hva_population_prediction_metrics(
    heldout_topk_rows: list[HVAPredictorPredictionRow],
    tile_count_float: float,
) -> dict[str, float]:
    """Evaluate heldout future L23E population-distribution prediction.

    This is intentionally an evaluation-only target smoothing. It repeat-averages
    existing heldout rows by target frame and tile, smooths the target mass over a
    fixed radius-1 retinotopic kernel, and ranks unchanged model/baseline scores.
    """

    tile_count = int(round(tile_count_float))
    grid_side = int(round(math.sqrt(tile_count)))
    if tile_count <= 0 or grid_side * grid_side != tile_count:
        raise ValidationError(f"HVA population prediction requires square tile count, got {tile_count}.")

    target_groups: dict[int, list[list[float]]] = {}
    repeat_indices_by_target_frame: dict[int, set[int]] = {}
    sample_groups: dict[tuple[int, int], dict[str, object]] = {}
    for row in heldout_topk_rows:
        if row.target_channel != "l23e" or row.topk_sample_valid != 1:
            continue
        if row.tile_id < 0 or row.tile_id >= tile_count:
            raise ValidationError(f"HVA population prediction found invalid tile_id={row.tile_id}.")

        target_group = target_groups.get(row.target_frame_index)
        if target_group is None:
            target_group = [[] for _ in range(tile_count)]
            target_groups[row.target_frame_index] = target_group
        target_group[row.tile_id].append(max(0.0, row.topk_target_value_norm))
        repeat_indices_by_target_frame.setdefault(row.target_frame_index, set()).add(row.repeat_index)

        sample_key = (row.repeat_index, row.target_frame_index)
        sample_group = sample_groups.get(sample_key)
        if sample_group is None:
            sample_group = {}
            for scorer in _HVA_POPULATION_SCORERS:
                sample_group[scorer] = [[] for _ in range(tile_count)]
            sample_groups[sample_key] = sample_group
        for scorer in _HVA_POPULATION_SCORERS:
            score_lists = sample_group[scorer]
            if not isinstance(score_lists, list):
                raise ValidationError("HVA population prediction internal score list error.")
            score = _hva_population_score(row, scorer)
            if not math.isfinite(score):
                raise ValidationError(f"HVA population prediction found non-finite {scorer} score.")
            score_lists[row.tile_id].append(score)

    metric_lists: dict[str, list[float]] = {}
    for scorer in _HVA_POPULATION_SCORERS:
        for k in (5, 10):
            metric_lists[f"{scorer}_ndcg_at{k}"] = []
            metric_lists[f"{scorer}_captured_ideal_mass_at{k}"] = []

    smoothed_target_by_frame: dict[int, list[float]] = {}
    skipped_target_frame_count = 0
    for target_frame_index, target_lists in target_groups.items():
        if any(len(values) == 0 for values in target_lists):
            skipped_target_frame_count += 1
            continue
        target_values = [mean(values) for values in target_lists]
        smoothed_target = _smooth_hva_population_target(target_values, grid_side)
        if sum(smoothed_target) <= 1.0e-12:
            skipped_target_frame_count += 1
            continue
        smoothed_target_by_frame[target_frame_index] = smoothed_target

    evaluated_sample_count = 0
    skipped_sample_count = 0
    evaluated_target_frames: set[int] = set()
    for sample_key, sample_group in sample_groups.items():
        target_frame_index = sample_key[1]
        smoothed_target = smoothed_target_by_frame.get(target_frame_index)
        if smoothed_target is None:
            skipped_sample_count += 1
            continue
        scorer_values: dict[str, list[float]] = {}
        scorer_complete = True
        for scorer in _HVA_POPULATION_SCORERS:
            score_lists = sample_group[scorer]
            if not isinstance(score_lists, list) or any(len(values) == 0 for values in score_lists):
                scorer_complete = False
                break
            scorer_values[scorer] = [mean(values) for values in score_lists]
        if not scorer_complete:
            skipped_sample_count += 1
            continue

        for scorer, scores in scorer_values.items():
            for k in (5, 10):
                ndcg = _weighted_ndcg_at_k(smoothed_target, scores, k)
                captured_mass = _captured_ideal_mass_at_k(smoothed_target, scores, k)
                if ndcg is not None:
                    metric_lists[f"{scorer}_ndcg_at{k}"].append(ndcg)
                if captured_mass is not None:
                    metric_lists[f"{scorer}_captured_ideal_mass_at{k}"].append(captured_mass)
        evaluated_sample_count += 1
        evaluated_target_frames.add(target_frame_index)

    if evaluated_sample_count <= 0:
        raise ValidationError("HVA population prediction found no evaluable heldout samples.")

    repeat_counts = [float(len(repeats)) for repeats in repeat_indices_by_target_frame.values()]

    metrics: dict[str, float] = {
        "tile_count": float(tile_count),
        "tile_grid_side": float(grid_side),
        "heldout_topk_row_count": float(len(heldout_topk_rows)),
        "target_frame_count": float(len(target_groups)),
        "evaluated_frame_count": float(len(evaluated_target_frames)),
        "skipped_frame_count": float(skipped_target_frame_count),
        "evaluated_sample_count": float(evaluated_sample_count),
        "skipped_sample_count": float(skipped_sample_count),
        "repeat_count_mean": mean(repeat_counts) if repeat_counts else 1.0,
        "uniform_chance_captured_ideal_mass_at5": min(1.0, 5.0 / max(1.0, float(tile_count))),
        "uniform_chance_captured_ideal_mass_at10": min(1.0, 10.0 / max(1.0, float(tile_count))),
    }
    for key, values in metric_lists.items():
        if not values:
            raise ValidationError(f"HVA population prediction could not compute {key}.")
        metrics[key] = mean(values)

    metrics["model_ndcg_at5_vs_persistence_ratio"] = _safe_metric_ratio(
        metrics["model_ndcg_at5"],
        metrics["persistence_ndcg_at5"],
    )
    metrics["model_captured_ideal_mass_at5_vs_persistence_ratio"] = _safe_metric_ratio(
        metrics["model_captured_ideal_mass_at5"],
        metrics["persistence_captured_ideal_mass_at5"],
    )
    return metrics


def validate_hva_predictor(run: RunData, require_population_prediction: bool = False) -> bool:
    overall_ok = True
    config = run.hva_predictor_config
    metrics = run.hva_predictor_metrics
    predictions = run.hva_predictor_predictions
    event_tiles = run.hva_predictor_event_tiles
    weights = run.hva_predictor_weights
    rate_row_count = run.hva_predictor_rate_row_count
    summary_enabled = run.summary.get("hva_predictor_enabled", 0.0)
    artifacts_available = (
        summary_enabled == 1.0
        and config is not None
        and metrics is not None
        and predictions is not None
        and event_tiles is not None
        and weights is not None
        and rate_row_count is not None
        and len(predictions) > 0
        and len(event_tiles) > 0
        and len(weights) > 0
        and rate_row_count > 0
    )
    overall_ok &= print_result(
        artifacts_available,
        "hva_predictor_artifacts_available",
        (
            f"hva_predictor_enabled={summary_enabled:.6f} "
            f"config_rows={len(config) if config is not None else 0} "
            f"metric_rows={len(metrics) if metrics is not None else 0} "
            f"rate_rows={rate_row_count if rate_row_count is not None else 0} "
            f"prediction_rows={len(predictions) if predictions is not None else 0} "
            f"event_tile_rows={len(event_tiles) if event_tiles is not None else 0} "
            f"weight_rows={len(weights) if weights is not None else 0}"
        ),
    )
    if (
        not artifacts_available
        or config is None
        or metrics is None
        or predictions is None
        or event_tiles is None
        or weights is None
    ):
        return overall_ok

    train_rows = [row for row in predictions if row.split == "train"]
    heldout_rows = [row for row in predictions if row.split == "heldout"]
    required_target_channels = ("l23e",)
    target_channels = sorted({row.target_channel for row in predictions})
    heldout_by_channel = {
        channel: [row for row in heldout_rows if row.target_channel == channel]
        for channel in target_channels
    }
    split_ok = (
        len(train_rows) > 0
        and len(heldout_rows) > 0
        and all(channel in target_channels for channel in required_target_channels)
        and target_channels == ["l23e"]
        and all(row.learning_update_applied == 0 for row in train_rows)
        and all(row.learning_update_applied == 0 for row in heldout_rows)
        and int(round(require_metric(metrics, "train_prediction_count", "HVA predictor metrics"))) == len(train_rows)
        and int(round(require_metric(metrics, "heldout_prediction_count", "HVA predictor metrics"))) == len(heldout_rows)
        and require_metric(metrics, "heldout_mode_code", "HVA predictor metrics") == 2.0
        and all(
            row.target_frame_index < int(round(require_metric(metrics, "heldout_start_frame", "HVA predictor metrics")))
            for row in train_rows
        )
        and all(
            row.frame_index >= int(round(require_metric(metrics, "heldout_start_frame", "HVA predictor metrics")))
            and row.target_frame_index >= int(round(require_metric(metrics, "heldout_start_frame", "HVA predictor metrics")))
            for row in heldout_rows
        )
    )
    overall_ok &= print_result(
        split_ok,
        "hva_predictor_heldout_split",
        (
            f"train_rows={len(train_rows)} heldout_rows={len(heldout_rows)} "
            f"heldout_mode_code={require_metric(metrics, 'heldout_mode_code', 'HVA predictor metrics'):.0f} "
            f"heldout_start_repeat={require_metric(metrics, 'heldout_start_repeat', 'HVA predictor metrics'):.0f} "
            f"heldout_start_frame={require_metric(metrics, 'heldout_start_frame', 'HVA predictor metrics'):.0f} "
            f"train_frame_count={require_metric(metrics, 'train_frame_count', 'HVA predictor metrics'):.0f} "
            f"heldout_frame_count={require_metric(metrics, 'heldout_frame_count', 'HVA predictor metrics'):.0f} "
            f"boundary_gap_prediction_count={require_metric(metrics, 'boundary_gap_prediction_count', 'HVA predictor metrics'):.0f} "
            f"target_channels={','.join(target_channels)} "
            f"evaluation_updates_applied=0"
        ),
    )

    lower_v1_frozen = run.summary.get("hva_predictor_lower_v1_frozen", config.get("lower_v1_frozen", 0.0))
    hva_to_v1_connections = run.summary.get(
        "hva_predictor_hva_to_v1_connection_count",
        config.get("hva_to_v1_connection_count", math.nan),
    )
    hva_to_v1_current = run.summary.get(
        "hva_predictor_hva_to_v1_current_enabled",
        config.get("hva_to_v1_current_enabled", math.nan),
    )
    lower_v1_weight_delta = run.summary.get(
        "hva_predictor_lower_v1_weight_delta_max_after_hva",
        config.get("lower_v1_weight_delta_max_after_hva", math.nan),
    )
    lower_v1_output_delta = run.summary.get(
        "hva_predictor_lower_v1_output_delta_max_after_hva",
        config.get("lower_v1_output_delta_max_after_hva", math.nan),
    )
    v1_mutation_after_hva = run.summary.get(
        "hva_predictor_v1_mutation_after_hva_enabled",
        config.get("v1_mutation_after_hva_enabled", math.nan),
    )
    fingerprint_equal = run.summary.get(
        "hva_predictor_lower_v1_replay_fingerprint_equal",
        config.get("lower_v1_replay_fingerprint_equal", math.nan),
    )
    site_hash_before = run.summary.get(
        "hva_predictor_lower_v1_replay_site_count_fingerprint32_before",
        config.get("lower_v1_replay_site_count_fingerprint32_before", math.nan),
    )
    site_hash_after = run.summary.get(
        "hva_predictor_lower_v1_replay_site_count_fingerprint32_after",
        config.get("lower_v1_replay_site_count_fingerprint32_after", math.nan),
    )
    tile_hash_before = run.summary.get(
        "hva_predictor_lower_v1_replay_tile_rate_fingerprint32_before",
        config.get("lower_v1_replay_tile_rate_fingerprint32_before", math.nan),
    )
    tile_hash_after = run.summary.get(
        "hva_predictor_lower_v1_replay_tile_rate_fingerprint32_after",
        config.get("lower_v1_replay_tile_rate_fingerprint32_after", math.nan),
    )
    site_sum_before = run.summary.get(
        "hva_predictor_lower_v1_replay_site_count_sum_before",
        config.get("lower_v1_replay_site_count_sum_before", math.nan),
    )
    site_sum_after = run.summary.get(
        "hva_predictor_lower_v1_replay_site_count_sum_after",
        config.get("lower_v1_replay_site_count_sum_after", math.nan),
    )
    tile_sum_before = run.summary.get(
        "hva_predictor_lower_v1_replay_tile_rate_sum_before",
        config.get("lower_v1_replay_tile_rate_sum_before", math.nan),
    )
    tile_sum_after = run.summary.get(
        "hva_predictor_lower_v1_replay_tile_rate_sum_after",
        config.get("lower_v1_replay_tile_rate_sum_after", math.nan),
    )
    multitask_target_fingerprint_equal = run.summary.get(
        "hva_predictor_lower_v1_replay_multitask_target_fingerprint_equal",
        config.get("lower_v1_replay_multitask_target_fingerprint_equal", math.nan),
    )
    multitask_target_hash_before = run.summary.get(
        "hva_predictor_lower_v1_replay_multitask_target_fingerprint32_before",
        config.get("lower_v1_replay_multitask_target_fingerprint32_before", math.nan),
    )
    multitask_target_hash_after = run.summary.get(
        "hva_predictor_lower_v1_replay_multitask_target_fingerprint32_after",
        config.get("lower_v1_replay_multitask_target_fingerprint32_after", math.nan),
    )
    multitask_target_sum_before = run.summary.get(
        "hva_predictor_lower_v1_replay_multitask_target_sum_before",
        config.get("lower_v1_replay_multitask_target_sum_before", math.nan),
    )
    multitask_target_sum_after = run.summary.get(
        "hva_predictor_lower_v1_replay_multitask_target_sum_after",
        config.get("lower_v1_replay_multitask_target_sum_after", math.nan),
    )
    video_training_enabled = run.summary.get("video_training_enabled", math.nan)
    video_feedback_disabled = run.summary.get("video_feedback_disabled", math.nan)
    consolidation_summary_enabled = run.summary.get("lower_v1_video_consolidation_enabled", 0.0)
    consolidation_summary_requested = run.summary.get("lower_v1_video_consolidation_requested", 0.0)
    isolation_ok = (
        lower_v1_frozen == 1.0
        and hva_to_v1_connections == 0.0
        and hva_to_v1_current == 0.0
        and lower_v1_weight_delta <= 1.0e-12
        and lower_v1_output_delta <= 1.0e-12
        and v1_mutation_after_hva == 0.0
        and fingerprint_equal == 1.0
        and site_hash_before == site_hash_after
        and tile_hash_before == tile_hash_after
        and site_sum_before == site_sum_after
        and tile_sum_before == tile_sum_after
        and multitask_target_fingerprint_equal == 1.0
        and multitask_target_hash_before == multitask_target_hash_after
        and multitask_target_sum_before == multitask_target_sum_after
        and (video_training_enabled == 0.0 or consolidation_summary_enabled == 1.0)
        and video_feedback_disabled == 1.0
    )
    overall_ok &= print_result(
        isolation_ok,
        "hva_predictor_isolation",
        (
            f"lower_v1_frozen={lower_v1_frozen:.6f} "
            f"hva_to_v1_connection_count={hva_to_v1_connections:.6f} "
            f"hva_to_v1_current_enabled={hva_to_v1_current:.6f} "
            f"lower_v1_weight_delta_max_after_hva={lower_v1_weight_delta:.6e} "
            f"lower_v1_output_delta_max_after_hva={lower_v1_output_delta:.6e} "
            f"v1_mutation_after_hva_enabled={v1_mutation_after_hva:.6f} "
            f"fingerprint_equal={fingerprint_equal:.6f} "
            f"site_fingerprint_before={site_hash_before:.0f} "
            f"site_fingerprint_after={site_hash_after:.0f} "
            f"tile_fingerprint_before={tile_hash_before:.0f} "
            f"tile_fingerprint_after={tile_hash_after:.0f} "
            f"multitask_target_fingerprint_equal={multitask_target_fingerprint_equal:.6f} "
            f"multitask_target_fingerprint_before={multitask_target_hash_before:.0f} "
            f"multitask_target_fingerprint_after={multitask_target_hash_after:.0f} "
            f"video_training_enabled={video_training_enabled:.6f} "
            f"lower_v1_video_consolidation_enabled={consolidation_summary_enabled:.6f} "
            f"video_feedback_disabled={video_feedback_disabled:.6f}"
        ),
    )

    consolidation_metrics = run.video_consolidation_metrics
    if consolidation_summary_enabled == 1.0 or consolidation_metrics is not None:
        consolidation_source = consolidation_metrics if consolidation_metrics is not None else {}

        def consolidation_value(metric: str, summary_metric: str | None = None) -> float:
            if metric in consolidation_source:
                return consolidation_source[metric]
            return run.summary.get(summary_metric or f"lower_v1_video_consolidation_{metric}", math.nan)

        consolidation_enabled = consolidation_value("enabled")
        consolidation_requested = consolidation_value("requested")
        consolidation_repeat_count = consolidation_value("repeat_count")
        consolidation_frame_start = consolidation_value("frame_start_index")
        consolidation_frame_count = consolidation_value("frame_count")
        consolidation_heldout_start = consolidation_value("heldout_start_frame")
        consolidation_heldout_excluded = consolidation_value("heldout_excluded_frame_count")
        consolidation_heldout_used = consolidation_value("heldout_frames_used")
        consolidation_present_only = consolidation_value("present_frame_drive_only")
        consolidation_future_target = consolidation_value("future_frame_target_used")
        consolidation_label_target = consolidation_value("target_label_used")
        consolidation_pre_hva = consolidation_value("pre_hva_stage")
        consolidation_ff_plasticity = consolidation_value("feedforward_l4_l23_plasticity_enabled")
        consolidation_hva_feedback = consolidation_value("hva_feedback_enabled")
        consolidation_pre_trials = consolidation_value("pre_eval_trial_count")
        consolidation_trials = consolidation_value("consolidation_trial_count")
        consolidation_post_trials = consolidation_value("post_eval_trial_count")
        consolidation_pre_corr = consolidation_value("pre_l23e_repeat_corr")
        consolidation_post_corr = consolidation_value("post_l23e_repeat_corr")
        consolidation_pre_top5 = consolidation_value("pre_l23e_repeat_top5_overlap")
        consolidation_post_top5 = consolidation_value("post_l23e_repeat_top5_overlap")
        consolidation_l4_delta = consolidation_value("l4_l23_weight_delta_max")
        consolidation_l23ee_delta = consolidation_value("l23ee_weight_delta_max")
        consolidation_l23pv_delta = consolidation_value("l23pv_weight_delta_max")
        consolidation_l23som_delta = consolidation_value("l23som_weight_delta_max")
        video_repeat_count = run.summary.get("video_repeat_count", math.nan)
        consolidation_trial_counts_ok = (
            math.isfinite(video_repeat_count)
            and math.isfinite(consolidation_repeat_count)
            and abs(consolidation_pre_trials - (consolidation_frame_count * video_repeat_count)) <= 1.0e-6
            and abs(consolidation_post_trials - (consolidation_frame_count * video_repeat_count)) <= 1.0e-6
            and abs(consolidation_trials - (consolidation_frame_count * consolidation_repeat_count)) <= 1.0e-6
        )

        consolidation_audit_ok = (
            consolidation_summary_requested == 1.0
            and consolidation_summary_enabled == 1.0
            and consolidation_requested == 1.0
            and consolidation_enabled == 1.0
            and consolidation_frame_start == 0.0
            and consolidation_frame_count > 0.0
            and consolidation_heldout_start >= consolidation_frame_count
            and consolidation_heldout_excluded > 0.0
            and consolidation_heldout_used == 0.0
            and consolidation_present_only == 1.0
            and consolidation_future_target == 0.0
            and consolidation_label_target == 0.0
            and consolidation_pre_hva == 1.0
            and consolidation_ff_plasticity == 0.0
            and consolidation_hva_feedback == 0.0
            and consolidation_pre_trials > 0.0
            and consolidation_trials > 0.0
            and consolidation_post_trials > 0.0
            and consolidation_trial_counts_ok
            and math.isfinite(consolidation_pre_corr)
            and math.isfinite(consolidation_post_corr)
            and math.isfinite(consolidation_pre_top5)
            and math.isfinite(consolidation_post_top5)
            and consolidation_l4_delta <= 1.0e-12
            and math.isfinite(consolidation_l23ee_delta)
            and math.isfinite(consolidation_l23pv_delta)
            and math.isfinite(consolidation_l23som_delta)
        )
        overall_ok &= print_result(
            consolidation_audit_ok,
            "hva_predictor_lower_v1_video_consolidation_audit",
            (
                f"requested={consolidation_requested:.6f} enabled={consolidation_enabled:.6f} "
                f"frame_start_index={consolidation_frame_start:.0f} "
                f"frame_count={consolidation_frame_count:.0f} "
                f"heldout_start_frame={consolidation_heldout_start:.0f} "
                f"heldout_excluded_frame_count={consolidation_heldout_excluded:.0f} "
                f"heldout_frames_used={consolidation_heldout_used:.0f} "
                f"present_frame_drive_only={consolidation_present_only:.0f} "
                f"future_frame_target_used={consolidation_future_target:.0f} "
                f"target_label_used={consolidation_label_target:.0f} "
                f"pre_hva_stage={consolidation_pre_hva:.6f} "
                f"feedforward_l4_l23_plasticity_enabled={consolidation_ff_plasticity:.6f} "
                f"hva_feedback_enabled={consolidation_hva_feedback:.6f} "
                f"pre_eval_trials={consolidation_pre_trials:.0f} "
                f"consolidation_trials={consolidation_trials:.0f} "
                f"post_eval_trials={consolidation_post_trials:.0f} "
                f"pre_l23e_repeat_corr={consolidation_pre_corr:.6f} "
                f"post_l23e_repeat_corr={consolidation_post_corr:.6f} "
                f"pre_l23e_repeat_top5_overlap={consolidation_pre_top5:.6f} "
                f"post_l23e_repeat_top5_overlap={consolidation_post_top5:.6f} "
                f"l4_l23_weight_delta_max={consolidation_l4_delta:.6e} "
                f"l23ee_weight_delta_max={consolidation_l23ee_delta:.6e} "
                f"l23pv_weight_delta_max={consolidation_l23pv_delta:.6e} "
                f"l23som_weight_delta_max={consolidation_l23som_delta:.6e}"
            ),
        )

    prediction_count = require_metric(metrics, "prediction_count", "HVA predictor metrics")
    sample_count = require_metric(metrics, "sample_count", "HVA predictor metrics")
    tile_count = require_metric(metrics, "tile_count", "HVA predictor metrics")
    train_count = require_metric(metrics, "train_prediction_count", "HVA predictor metrics")
    heldout_count = require_metric(metrics, "heldout_prediction_count", "HVA predictor metrics")
    target_channel_count = require_metric(metrics, "target_channel_count", "HVA predictor metrics")
    required_target_channel_count = require_metric(metrics, "required_target_channel_count", "HVA predictor metrics")
    l23e_target_enabled = require_metric(metrics, "l23e_target_channel_enabled", "HVA predictor metrics")
    l4e_target_enabled = require_metric(metrics, "l4e_target_channel_enabled", "HVA predictor metrics")
    l23pv_target_enabled = require_metric(metrics, "l23pv_target_channel_enabled", "HVA predictor metrics")
    non_l23_required_count = require_metric(
        metrics,
        "non_l23_required_target_channel_count",
        "HVA predictor metrics",
    )
    non_l23_autoreg_baseline = require_metric(
        metrics,
        "non_l23_target_autoregressive_baseline_enabled",
        "HVA predictor metrics",
    )
    l23e_only_input = require_metric(metrics, "input_channel_l23e_only", "HVA predictor metrics")
    l4e_input_enabled = require_metric(metrics, "input_channel_l4e_enabled", "HVA predictor metrics")
    l23pv_input_enabled = require_metric(metrics, "input_channel_l23pv_enabled", "HVA predictor metrics")
    target_mode = require_metric(metrics, "prediction_target_mode_code", "HVA predictor metrics")
    residual_enabled = require_metric(metrics, "residual_prediction_enabled", "HVA predictor metrics")
    residual_rate_head = require_metric(metrics, "l23e_residual_rate_head_enabled", "HVA predictor metrics")
    event_head = require_metric(metrics, "l23e_event_hazard_head_enabled", "HVA predictor metrics")
    event_window_head = require_metric(metrics, "l23e_event_window_hazard_head_enabled", "HVA predictor metrics")
    single_frame_report_only = require_metric(metrics, "l23e_single_frame_event_report_only", "HVA predictor metrics")
    topk_head = require_metric(metrics, "l23e_future_topk_head_enabled", "HVA predictor metrics")
    topk_objective = require_metric(metrics, "topk_objective_enabled", "HVA predictor metrics")
    topk_target_l23e_only = require_metric(metrics, "topk_target_channel_l23e_only", "HVA predictor metrics")
    topk_input_l23e_only = require_metric(metrics, "topk_input_channel_l23e_only", "HVA predictor metrics")
    topk_feedback_enabled = require_metric(metrics, "topk_feedback_enabled", "HVA predictor metrics")
    topk_tile_size_sites = require_metric(metrics, "topk_tile_size_sites", "HVA predictor metrics")
    topk_metric_tile_grid_side = require_metric(metrics, "topk_tile_grid_side", "HVA predictor metrics")
    topk_metric_tile_count = require_metric(metrics, "topk_tile_count", "HVA predictor metrics")
    topk_k = require_metric(metrics, "topk_k", "HVA predictor metrics")
    topk_future_window_frames = require_metric(metrics, "topk_future_window_frames", "HVA predictor metrics")
    topk_future_window_ms = require_metric(metrics, "topk_future_window_ms", "HVA predictor metrics")
    topk_learning_rate = require_metric(metrics, "topk_learning_rate", "HVA predictor metrics")
    topk_weight_decay = require_metric(metrics, "topk_weight_decay", "HVA predictor metrics")
    topk_train_valid_count = require_metric(metrics, "topk_train_valid_sample_count", "HVA predictor metrics")
    topk_heldout_valid_count = require_metric(metrics, "topk_heldout_valid_sample_count", "HVA predictor metrics")
    topk_frequency_valid_count = require_metric(
        metrics,
        "topk_train_frequency_valid_sample_count",
        "HVA predictor metrics",
    )
    train_then_heldout = require_metric(metrics, "train_then_heldout_enabled", "HVA predictor metrics")
    evaluation_updates = require_metric(metrics, "evaluation_updates_enabled", "HVA predictor metrics")
    training_epochs = require_metric(metrics, "training_epoch_count", "HVA predictor metrics")
    training_update_count = require_metric(metrics, "training_update_count", "HVA predictor metrics")
    event_window_frames = require_metric(metrics, "event_window_frames", "HVA predictor metrics")
    event_window_ms = require_metric(metrics, "event_window_ms", "HVA predictor metrics")
    event_window_mode = require_metric(metrics, "event_window_target_mode_code", "HVA predictor metrics")
    event_train_only_threshold = require_metric(
        metrics,
        "event_hazard_train_only_threshold_enabled",
        "HVA predictor metrics",
    )
    event_l23e_only_input = require_metric(metrics, "event_hazard_input_channel_l23e_only", "HVA predictor metrics")
    event_non_l23_target = require_metric(metrics, "event_hazard_non_l23_target_enabled", "HVA predictor metrics")
    normalized_residual_target = require_metric(
        metrics,
        "learning_target_normalized_rate_residual_enabled",
        "HVA predictor metrics",
    )
    zscore_target = require_metric(metrics, "learning_target_zscore_enabled", "HVA predictor metrics")
    train_only_norm = require_metric(metrics, "train_only_normalization_enabled", "HVA predictor metrics")
    signed_weights = require_metric(metrics, "signed_residual_host_weights_enabled", "HVA predictor metrics")
    local_readout = require_metric(metrics, "local_readout_enabled", "HVA predictor metrics")
    dense_readout = require_metric(metrics, "dense_all_to_all_readout_enabled", "HVA predictor metrics")
    feature_count = require_metric(metrics, "feature_channel_count", "HVA predictor metrics")
    base_feature_count = require_metric(metrics, "base_feature_channel_count", "HVA predictor metrics")
    lag_history_frame_count = require_metric(metrics, "lag_history_frame_count", "HVA predictor metrics")
    lag_history_ms = require_metric(metrics, "lag_history_ms", "HVA predictor metrics")
    lag_history_l23e_only = require_metric(metrics, "lag_history_l23e_only", "HVA predictor metrics")
    lag_future_lookahead = require_metric(metrics, "lag_feature_future_lookahead_frames", "HVA predictor metrics")
    local_context_enabled = require_metric(metrics, "local_context_feature_enabled", "HVA predictor metrics")
    local_context_radius = require_metric(metrics, "local_context_radius_tiles", "HVA predictor metrics")
    local_context_summary_count = require_metric(
        metrics,
        "local_context_summary_feature_count",
        "HVA predictor metrics",
    )
    directional_context_enabled = require_metric(
        metrics,
        "directional_context_feature_enabled",
        "HVA predictor metrics",
    )
    directional_context_radius = require_metric(
        metrics,
        "directional_context_radius_tiles",
        "HVA predictor metrics",
    )
    directional_context_count = require_metric(
        metrics,
        "directional_context_feature_count",
        "HVA predictor metrics",
    )
    directional_context_l23e_only = require_metric(
        metrics,
        "directional_context_l23e_only",
        "HVA predictor metrics",
    )
    directional_context_future_lookahead = require_metric(
        metrics,
        "directional_context_future_lookahead_frames",
        "HVA predictor metrics",
    )
    sequence_state_enabled = metrics.get("sequence_state_enabled", 0.0)
    sequence_state_dim = metrics.get("sequence_state_dim", 0.0)
    sequence_state_feature_count = metrics.get("sequence_state_feature_count", 0.0)
    sequence_state_leak = metrics.get("sequence_state_leak", 0.0)
    sequence_state_input_scale = metrics.get("sequence_state_input_scale", 0.0)
    sequence_state_neighbor_scale = metrics.get("sequence_state_neighbor_scale", 0.0)
    sequence_state_neighbor_radius = metrics.get("sequence_state_neighbor_radius_tiles", 0.0)
    sequence_state_l23e_only = metrics.get("sequence_state_l23e_only", 1.0)
    sequence_state_future_lookahead = metrics.get("sequence_state_future_lookahead_frames", 0.0)
    topk_sequence_state_feature_enabled = metrics.get("topk_sequence_state_feature_enabled", 0.0)
    residual_event_sequence_state_feature_enabled = metrics.get(
        "residual_event_sequence_state_feature_enabled",
        0.0,
    )
    topk_repeat_avg_target_enabled = metrics.get("topk_repeat_avg_target_enabled", 0.0)
    topk_repeat_avg_target_train_only = metrics.get("topk_repeat_avg_target_train_only", 1.0)
    topk_repeat_avg_target_frame_count = metrics.get("topk_repeat_avg_target_frame_count", 0.0)
    topk_repeat_avg_target_sample_count = metrics.get("topk_repeat_avg_target_sample_count", 0.0)
    topk_target_smooth_radius = metrics.get("topk_target_smooth_radius_tiles", 0.0)
    topk_target_smoothing_enabled = metrics.get("topk_target_smoothing_enabled", 0.0)
    topk_target_smoothing_kernel_code = metrics.get("topk_target_smoothing_kernel_code", 0.0)
    topk_target_smoothing_target_only = metrics.get("topk_target_smoothing_target_only", 1.0)
    topk_target_smoothing_input_feature_enabled = metrics.get(
        "topk_target_smoothing_input_feature_enabled",
        0.0,
    )
    topk_target_smoothing_train_repeat_avg_only = metrics.get(
        "topk_target_smoothing_train_repeat_avg_only",
        0.0,
    )
    topk_target_smoothing_eval_repeat_avg_only = metrics.get(
        "topk_target_smoothing_eval_repeat_avg_only",
        1.0,
    )
    topk_frequency_balance_enabled = metrics.get("topk_frequency_balance_enabled", 0.0)
    topk_frequency_balance_train_only = metrics.get("topk_frequency_balance_train_only", 1.0)
    topk_frequency_balance_floor = metrics.get("topk_frequency_balance_floor", 0.0)
    local_context_l23e_only = require_metric(metrics, "local_context_l23e_only", "HVA predictor metrics")
    feature_non_l23_inputs = require_metric(metrics, "feature_uses_non_l23_inputs", "HVA predictor metrics")
    feature_future_leakage = require_metric(metrics, "feature_future_leakage_enabled", "HVA predictor metrics")
    derivative_feature = require_metric(metrics, "derivative_feature_enabled", "HVA predictor metrics")
    past_only_lookahead = require_metric(metrics, "past_only_feature_lookahead_frames", "HVA predictor metrics")
    active_pair_fraction = require_metric(metrics, "active_readout_pair_fraction", "HVA predictor metrics")
    residual_std_min = require_metric(metrics, "train_residual_std_min_norm", "HVA predictor metrics")
    target_std = require_metric(metrics, "target_std_norm", "HVA predictor metrics")
    prediction_std = require_metric(metrics, "prediction_std_norm", "HVA predictor metrics")
    weight_l1 = require_metric(metrics, "weight_l1", "HVA predictor metrics")
    event_weight_l1 = require_metric(metrics, "event_weight_l1", "HVA predictor metrics")
    weight_max_abs = require_metric(metrics, "weight_max_abs", "HVA predictor metrics")
    event_weight_max_abs = require_metric(metrics, "event_weight_max_abs", "HVA predictor metrics")
    weight_clip = require_metric(metrics, "weight_clip", "HVA predictor metrics")
    event_selected_tile_count = require_metric(metrics, "event_selected_tile_count", "HVA predictor metrics")
    feature_standardization_enabled = require_metric(
        metrics,
        "feature_standardization_enabled",
        "HVA predictor metrics",
    )
    feature_standardization_train_only = require_metric(
        metrics,
        "feature_standardization_train_only",
        "HVA predictor metrics",
    )
    feature_standardization_count = require_metric(
        metrics,
        "feature_standardization_feature_count",
        "HVA predictor metrics",
    )
    feature_standardization_observations = require_metric(
        metrics,
        "feature_standardization_train_observation_count",
        "HVA predictor metrics",
    )
    feature_std_floor = require_metric(metrics, "feature_standardization_std_floor", "HVA predictor metrics")
    feature_std_floor_count = require_metric(
        metrics,
        "feature_standardization_std_floor_count",
        "HVA predictor metrics",
    )
    feature_std_min = require_metric(metrics, "feature_standardization_std_min", "HVA predictor metrics")
    feature_std_median = require_metric(metrics, "feature_standardization_std_median", "HVA predictor metrics")
    feature_std_max = require_metric(metrics, "feature_standardization_std_max", "HVA predictor metrics")
    residual_learning_rate = require_metric(metrics, "residual_learning_rate", "HVA predictor metrics")
    event_learning_rate = require_metric(metrics, "event_learning_rate", "HVA predictor metrics")
    bias_learning_rate = require_metric(metrics, "bias_learning_rate", "HVA predictor metrics")
    event_bias_learning_rate = require_metric(metrics, "event_bias_learning_rate", "HVA predictor metrics")
    local_l2_decay_enabled = require_metric(metrics, "local_l2_weight_decay_enabled", "HVA predictor metrics")
    local_l2_decay = require_metric(metrics, "local_l2_weight_decay", "HVA predictor metrics")
    event_l2_decay = require_metric(metrics, "event_local_l2_weight_decay", "HVA predictor metrics")
    posthoc_global_norm = require_metric(metrics, "posthoc_global_normalization_enabled", "HVA predictor metrics")
    event_bias_initialized = require_metric(
        metrics,
        "event_bias_initialized_from_train_base_rate",
        "HVA predictor metrics",
    )
    event_base_rate_floor = require_metric(metrics, "event_base_rate_floor", "HVA predictor metrics")
    event_residual_gain = require_metric(metrics, "event_residual_gain", "HVA predictor metrics")
    event_train_rate_min = require_metric(metrics, "event_train_rate_min", "HVA predictor metrics")
    event_train_rate_median = require_metric(metrics, "event_train_rate_median", "HVA predictor metrics")
    event_train_rate_max = require_metric(metrics, "event_train_rate_max", "HVA predictor metrics")
    event_selected_train_rate_median = require_metric(
        metrics,
        "event_selected_train_rate_median",
        "HVA predictor metrics",
    )
    event_bias_min = require_metric(metrics, "event_bias_min", "HVA predictor metrics")
    event_bias_median = require_metric(metrics, "event_bias_median", "HVA predictor metrics")
    event_bias_max = require_metric(metrics, "event_bias_max", "HVA predictor metrics")
    residual_group_total = require_metric(metrics, "residual_weight_abs_group_total", "HVA predictor metrics")
    event_group_total = require_metric(metrics, "event_weight_abs_group_total", "HVA predictor metrics")
    residual_group_values = [
        require_metric(metrics, "residual_weight_abs_current", "HVA predictor metrics"),
        require_metric(metrics, "residual_weight_abs_trace", "HVA predictor metrics"),
        require_metric(metrics, "residual_weight_abs_derivative", "HVA predictor metrics"),
        require_metric(metrics, "residual_weight_abs_lag", "HVA predictor metrics"),
        require_metric(metrics, "residual_weight_abs_context", "HVA predictor metrics"),
        metrics.get("residual_weight_abs_sequence", 0.0),
    ]
    event_group_values = [
        require_metric(metrics, "event_weight_abs_current", "HVA predictor metrics"),
        require_metric(metrics, "event_weight_abs_trace", "HVA predictor metrics"),
        require_metric(metrics, "event_weight_abs_derivative", "HVA predictor metrics"),
        require_metric(metrics, "event_weight_abs_lag", "HVA predictor metrics"),
        require_metric(metrics, "event_weight_abs_context", "HVA predictor metrics"),
        metrics.get("event_weight_abs_sequence", 0.0),
    ]
    topk_sequence_weight_abs = metrics.get("topk_weight_abs_sequence", 0.0)
    expected_feature_count = (
        base_feature_count
        + lag_history_frame_count
        + local_context_summary_count
        + directional_context_count
        + sequence_state_feature_count
    )
    directional_context_ok = (
        directional_context_l23e_only == 1.0
        and directional_context_future_lookahead == 0.0
        and (
            (
                directional_context_enabled == 1.0
                and directional_context_radius == local_context_radius
                and directional_context_count >= (6.0 * (lag_history_frame_count + 1.0))
            )
            or (
                directional_context_enabled == 0.0
                and directional_context_radius == 0.0
                and directional_context_count == 0.0
            )
        )
    )
    sequence_state_ok = (
        sequence_state_l23e_only == 1.0
        and sequence_state_future_lookahead == 0.0
        and residual_event_sequence_state_feature_enabled == 0.0
        and (
            (
                sequence_state_enabled == 1.0
                and sequence_state_dim >= 1.0
                and sequence_state_feature_count == sequence_state_dim
                and topk_sequence_state_feature_enabled == 1.0
                and 0.0 <= sequence_state_leak <= 1.0
                and sequence_state_input_scale >= 0.0
                and sequence_state_neighbor_scale >= 0.0
                and sequence_state_neighbor_radius == 1.0
                and topk_sequence_weight_abs > 0.0
            )
            or (
                sequence_state_enabled == 0.0
                and sequence_state_feature_count == 0.0
                and topk_sequence_state_feature_enabled == 0.0
            )
        )
    )
    repeat_avg_target_ok = (
        topk_repeat_avg_target_train_only == 1.0
        and (
            (
                topk_repeat_avg_target_enabled == 1.0
                and topk_repeat_avg_target_frame_count > 0.0
                and topk_repeat_avg_target_sample_count >= topk_repeat_avg_target_frame_count
            )
            or (
                topk_repeat_avg_target_enabled == 0.0
                and topk_repeat_avg_target_frame_count == 0.0
                and topk_repeat_avg_target_sample_count == 0.0
            )
        )
    )
    frequency_balance_ok = (
        topk_frequency_balance_train_only == 1.0
        and (
            topk_frequency_balance_enabled == 0.0
            or (topk_frequency_balance_floor > 0.0 and topk_frequency_balance_floor <= 1.0)
        )
    )
    target_smoothing_ok = (
        topk_target_smoothing_target_only == 1.0
        and topk_target_smoothing_input_feature_enabled == 0.0
        and topk_target_smoothing_eval_repeat_avg_only == 1.0
        and (
            (
                topk_target_smoothing_enabled == 0.0
                and topk_target_smooth_radius == 0.0
            )
            or (
                topk_target_smoothing_enabled == 1.0
                and topk_target_smooth_radius == 1.0
                and topk_target_smoothing_kernel_code == 121242121.0
                and topk_target_smoothing_train_repeat_avg_only == 1.0
                and topk_repeat_avg_target_enabled == 1.0
            )
        )
    )
    lag_context_feature_ok = (
        feature_count >= expected_feature_count
        and base_feature_count >= 5.0
        and lag_history_frame_count >= 5.0
        and lag_history_ms > 0.0
        and lag_history_l23e_only == 1.0
        and lag_future_lookahead == 0.0
        and local_context_enabled == 1.0
        and local_context_radius >= 1.0
        and local_context_summary_count >= (3.0 * (lag_history_frame_count + 1.0))
        and local_context_l23e_only == 1.0
        and directional_context_ok
        and sequence_state_ok
        and feature_non_l23_inputs == 0.0
        and feature_future_leakage == 0.0
        and past_only_lookahead == 0.0
    )
    overall_ok &= print_result(
        lag_context_feature_ok,
        "hva_predictor_l23e_lag_context_features",
        (
            f"feature_channel_count={feature_count:.0f} "
            f"expected_min_feature_count={expected_feature_count:.0f} "
            f"base_feature_channel_count={base_feature_count:.0f} "
            f"lag_history_frame_count={lag_history_frame_count:.0f} "
            f"lag_history_ms={lag_history_ms:.6f} "
            f"lag_history_l23e_only={lag_history_l23e_only:.0f} "
            f"lag_feature_future_lookahead_frames={lag_future_lookahead:.0f} "
            f"local_context_feature_enabled={local_context_enabled:.0f} "
            f"local_context_radius_tiles={local_context_radius:.0f} "
            f"local_context_summary_feature_count={local_context_summary_count:.0f} "
            f"local_context_l23e_only={local_context_l23e_only:.0f} "
            f"directional_context_feature_enabled={directional_context_enabled:.0f} "
            f"directional_context_radius_tiles={directional_context_radius:.0f} "
            f"directional_context_feature_count={directional_context_count:.0f} "
            f"directional_context_l23e_only={directional_context_l23e_only:.0f} "
            f"directional_context_future_lookahead_frames={directional_context_future_lookahead:.0f} "
            f"sequence_state_enabled={sequence_state_enabled:.0f} "
            f"sequence_state_dim={sequence_state_dim:.0f} "
            f"sequence_state_feature_count={sequence_state_feature_count:.0f} "
            f"sequence_state_l23e_only={sequence_state_l23e_only:.0f} "
            f"sequence_state_future_lookahead_frames={sequence_state_future_lookahead:.0f} "
            f"feature_uses_non_l23_inputs={feature_non_l23_inputs:.0f} "
            f"feature_future_leakage_enabled={feature_future_leakage:.0f}"
        ),
    )
    overall_ok &= print_result(
        sequence_state_ok and repeat_avg_target_ok and frequency_balance_ok and target_smoothing_ok,
        "hva_predictor_sequence_state_target_audit",
        (
            f"sequence_state_enabled={sequence_state_enabled:.0f} "
            f"sequence_state_dim={sequence_state_dim:.0f} "
            f"sequence_state_leak={sequence_state_leak:.6f} "
            f"sequence_state_input_scale={sequence_state_input_scale:.6f} "
            f"sequence_state_neighbor_scale={sequence_state_neighbor_scale:.6f} "
            f"sequence_state_neighbor_radius_tiles={sequence_state_neighbor_radius:.0f} "
            f"topk_sequence_state_feature_enabled={topk_sequence_state_feature_enabled:.0f} "
            f"residual_event_sequence_state_feature_enabled={residual_event_sequence_state_feature_enabled:.0f} "
            f"topk_weight_abs_sequence={topk_sequence_weight_abs:.6f} "
            f"topk_repeat_avg_target_enabled={topk_repeat_avg_target_enabled:.0f} "
            f"topk_repeat_avg_target_train_only={topk_repeat_avg_target_train_only:.0f} "
            f"topk_repeat_avg_target_frame_count={topk_repeat_avg_target_frame_count:.0f} "
            f"topk_repeat_avg_target_sample_count={topk_repeat_avg_target_sample_count:.0f} "
            f"topk_target_smoothing_enabled={topk_target_smoothing_enabled:.0f} "
            f"topk_target_smooth_radius_tiles={topk_target_smooth_radius:.0f} "
            f"topk_target_smoothing_kernel_code={topk_target_smoothing_kernel_code:.0f} "
            f"topk_target_smoothing_target_only={topk_target_smoothing_target_only:.0f} "
            f"topk_target_smoothing_input_feature_enabled="
            f"{topk_target_smoothing_input_feature_enabled:.0f} "
            f"topk_target_smoothing_train_repeat_avg_only="
            f"{topk_target_smoothing_train_repeat_avg_only:.0f} "
            f"topk_target_smoothing_eval_repeat_avg_only="
            f"{topk_target_smoothing_eval_repeat_avg_only:.0f} "
            f"topk_frequency_balance_enabled={topk_frequency_balance_enabled:.0f} "
            f"topk_frequency_balance_train_only={topk_frequency_balance_train_only:.0f} "
            f"topk_frequency_balance_floor={topk_frequency_balance_floor:.6f}"
        ),
    )
    conditioning_ok = (
        feature_standardization_enabled == 1.0
        and feature_standardization_train_only == 1.0
        and feature_standardization_count == feature_count
        and feature_standardization_observations >= train_count
        and feature_std_floor > 0.0
        and 0.0 <= feature_std_floor_count <= feature_count
        and math.isfinite(feature_std_min)
        and math.isfinite(feature_std_median)
        and math.isfinite(feature_std_max)
        and feature_std_min > 0.0
        and feature_std_max >= feature_std_min
        and residual_learning_rate > 0.0
        and event_learning_rate > 0.0
        and residual_learning_rate <= 0.01
        and event_learning_rate <= 0.01
        and event_learning_rate <= residual_learning_rate
        and bias_learning_rate <= 0.01
        and event_bias_learning_rate <= 0.001
        and local_l2_decay_enabled == 1.0
        and 0.0 < local_l2_decay < 0.1
        and event_l2_decay >= local_l2_decay
        and event_l2_decay < 0.1
        and posthoc_global_norm == 0.0
        and event_bias_initialized == 1.0
        and 0.0 < event_base_rate_floor < 0.01
        and 0.0 < event_residual_gain <= 1.0
        and 0.0 <= event_train_rate_min <= event_train_rate_median <= event_train_rate_max <= 1.0
        and 0.0 <= event_selected_train_rate_median <= 1.0
        and math.isfinite(event_bias_min)
        and math.isfinite(event_bias_median)
        and math.isfinite(event_bias_max)
        and event_bias_min <= event_bias_median <= event_bias_max
        and math.isfinite(residual_group_total)
        and math.isfinite(event_group_total)
        and abs(sum(residual_group_values) - residual_group_total) <= max(1.0e-6, 1.0e-6 * max(1.0, residual_group_total))
        and abs(sum(event_group_values) - event_group_total) <= max(1.0e-6, 1.0e-6 * max(1.0, event_group_total))
    )
    overall_ok &= print_result(
        conditioning_ok,
        "hva_predictor_feature_standardization_homeostasis",
        (
            f"feature_standardization_enabled={feature_standardization_enabled:.0f} "
            f"feature_standardization_train_only={feature_standardization_train_only:.0f} "
            f"feature_standardization_feature_count={feature_standardization_count:.0f} "
            f"feature_standardization_train_observation_count={feature_standardization_observations:.0f} "
            f"feature_std_floor={feature_std_floor:.6f} "
            f"feature_std_floor_count={feature_std_floor_count:.0f} "
            f"feature_std_min={feature_std_min:.6f} "
            f"feature_std_median={feature_std_median:.6f} "
            f"feature_std_max={feature_std_max:.6f} "
            f"residual_learning_rate={residual_learning_rate:.6f} "
            f"event_learning_rate={event_learning_rate:.6f} "
            f"bias_learning_rate={bias_learning_rate:.6f} "
            f"event_bias_learning_rate={event_bias_learning_rate:.6f} "
            f"local_l2_weight_decay={local_l2_decay:.6f} "
            f"event_local_l2_weight_decay={event_l2_decay:.6f} "
            f"event_bias_initialized_from_train_base_rate={event_bias_initialized:.0f} "
            f"event_base_rate_floor={event_base_rate_floor:.6f} "
            f"event_residual_gain={event_residual_gain:.6f} "
            f"event_train_rate_range=[{event_train_rate_min:.6f},{event_train_rate_max:.6f}] "
            f"event_train_rate_median={event_train_rate_median:.6f} "
            f"event_selected_train_rate_median={event_selected_train_rate_median:.6f} "
            f"event_bias_range=[{event_bias_min:.6f},{event_bias_max:.6f}] "
            f"event_bias_median={event_bias_median:.6f} "
            f"posthoc_global_normalization_enabled={posthoc_global_norm:.0f} "
            f"residual_group_total={residual_group_total:.6f} "
            f"event_group_total={event_group_total:.6f}"
        ),
    )
    print(
        "INFO hva_predictor_feature_group_weight_norms "
        f"residual_current={residual_group_values[0]:.6f} "
        f"residual_trace={residual_group_values[1]:.6f} "
        f"residual_derivative={residual_group_values[2]:.6f} "
        f"residual_lag={residual_group_values[3]:.6f} "
        f"residual_context={residual_group_values[4]:.6f} "
        f"residual_sequence={residual_group_values[5]:.6f} "
        f"event_current={event_group_values[0]:.6f} "
        f"event_trace={event_group_values[1]:.6f} "
        f"event_derivative={event_group_values[2]:.6f} "
        f"event_lag={event_group_values[3]:.6f} "
        f"event_context={event_group_values[4]:.6f} "
        f"event_sequence={event_group_values[5]:.6f} "
        f"topk_sequence={topk_sequence_weight_abs:.6f}"
    )
    learning_signal_ok = (
        prediction_count >= 16.0
        and train_count >= 8.0
        and heldout_count >= 8.0
        and sample_count >= 2.0
        and tile_count >= 4.0
        and target_channel_count == 1.0
        and required_target_channel_count == 1.0
        and l23e_target_enabled == 1.0
        and l4e_target_enabled == 0.0
        and l23pv_target_enabled == 0.0
        and non_l23_required_count == 0.0
        and non_l23_autoreg_baseline == 0.0
        and l23e_only_input == 1.0
        and l4e_input_enabled == 0.0
        and l23pv_input_enabled == 0.0
        and target_mode == 3.0
        and residual_enabled == 1.0
        and residual_rate_head == 1.0
        and event_head == 1.0
        and event_window_head == 1.0
        and single_frame_report_only == 1.0
        and topk_head == 1.0
        and topk_objective == 1.0
        and topk_target_l23e_only == 1.0
        and topk_input_l23e_only == 1.0
        and topk_feedback_enabled == 0.0
        and topk_tile_size_sites >= 1.0
        and abs((topk_metric_tile_grid_side * topk_metric_tile_grid_side) - topk_metric_tile_count) <= 1.0e-6
        and topk_metric_tile_count == tile_count
        and 1.0 <= topk_k <= tile_count
        and topk_future_window_frames >= 1.0
        and topk_future_window_ms > 0.0
        and topk_learning_rate > 0.0
        and 0.0 <= topk_weight_decay < 0.1
        and topk_train_valid_count > 0.0
        and topk_heldout_valid_count > 0.0
        and topk_frequency_valid_count > 0.0
        and train_then_heldout == 1.0
        and evaluation_updates == 0.0
        and training_epochs >= 1.0
        and training_update_count >= train_count
        and event_window_frames >= 2.0
        and event_window_ms >= 100.0
        and event_window_mode == 1.0
        and event_train_only_threshold == 1.0
        and event_l23e_only_input == 1.0
        and event_non_l23_target == 0.0
        and normalized_residual_target == 1.0
        and zscore_target == 0.0
        and train_only_norm == 1.0
        and signed_weights == 1.0
        and local_readout == 1.0
        and dense_readout == 0.0
        and lag_context_feature_ok
        and conditioning_ok
        and derivative_feature == 1.0
        and past_only_lookahead == 0.0
        and 0.0 < active_pair_fraction < 1.0
        and residual_std_min > 0.0
        and target_std > 1.0e-8
        and prediction_std >= 0.0
        and weight_l1 > 1.0e-9
        and event_weight_l1 > 1.0e-9
        and weight_max_abs <= (weight_clip + 1.0e-6)
        and event_weight_max_abs <= (weight_clip + 1.0e-6)
        and event_selected_tile_count >= 1.0
    )
    overall_ok &= print_result(
        learning_signal_ok,
        "hva_predictor_learning_signal",
        (
            f"sample_count={sample_count:.0f} prediction_count={prediction_count:.0f} "
            f"train_prediction_count={train_count:.0f} heldout_prediction_count={heldout_count:.0f} "
            f"tile_count={tile_count:.0f} target_channel_count={target_channel_count:.0f} "
            f"required_target_channel_count={required_target_channel_count:.0f} "
            f"non_l23_required_target_channel_count={non_l23_required_count:.0f} "
            f"non_l23_target_autoregressive_baseline_enabled={non_l23_autoreg_baseline:.0f} "
            f"target_std_norm={target_std:.6f} "
            f"prediction_std_norm={prediction_std:.6f} weight_l1={weight_l1:.6f} "
            f"weight_max_abs={weight_max_abs:.6f} weight_clip={weight_clip:.6f} "
            f"event_weight_l1={event_weight_l1:.6f} "
            f"event_weight_max_abs={event_weight_max_abs:.6f} "
            f"event_selected_tile_count={event_selected_tile_count:.0f} "
            f"training_epoch_count={training_epochs:.0f} "
            f"training_update_count={training_update_count:.0f} "
            f"train_then_heldout_enabled={train_then_heldout:.0f} "
            f"evaluation_updates_enabled={evaluation_updates:.0f} "
            f"event_window_frames={event_window_frames:.0f} "
            f"event_window_ms={event_window_ms:.6f} "
            f"prediction_target_mode_code={target_mode:.0f} "
            f"residual_prediction_enabled={residual_enabled:.0f} "
            f"l23e_residual_rate_head_enabled={residual_rate_head:.0f} "
            f"l23e_event_hazard_head_enabled={event_head:.0f} "
            f"l23e_event_window_hazard_head_enabled={event_window_head:.0f} "
            f"l23e_future_topk_head_enabled={topk_head:.0f} "
            f"topk_objective_enabled={topk_objective:.0f} "
            f"topk_target_channel_l23e_only={topk_target_l23e_only:.0f} "
            f"topk_input_channel_l23e_only={topk_input_l23e_only:.0f} "
            f"topk_feedback_enabled={topk_feedback_enabled:.0f} "
            f"topk_tile_size_sites={topk_tile_size_sites:.0f} "
            f"topk_tile_grid_side={topk_metric_tile_grid_side:.0f} "
            f"topk_tile_count={topk_metric_tile_count:.0f} "
            f"topk_k={topk_k:.0f} "
            f"topk_future_window_frames={topk_future_window_frames:.0f} "
            f"topk_future_window_ms={topk_future_window_ms:.6f} "
            f"topk_train_valid_sample_count={topk_train_valid_count:.0f} "
            f"topk_heldout_valid_sample_count={topk_heldout_valid_count:.0f} "
            f"learning_target_normalized_rate_residual_enabled={normalized_residual_target:.0f} "
            f"learning_target_zscore_enabled={zscore_target:.0f} "
            f"train_only_normalization_enabled={train_only_norm:.0f} "
            f"input_channel_l23e_only={l23e_only_input:.0f} "
            f"feature_channel_count={feature_count:.0f} "
            f"lag_history_frame_count={lag_history_frame_count:.0f} "
            f"local_context_radius_tiles={local_context_radius:.0f} "
            f"sequence_state_enabled={sequence_state_enabled:.0f} "
            f"sequence_state_dim={sequence_state_dim:.0f} "
            f"topk_repeat_avg_target_enabled={topk_repeat_avg_target_enabled:.0f} "
            f"topk_target_smooth_radius_tiles={topk_target_smooth_radius:.0f} "
            f"topk_target_smoothing_target_only={topk_target_smoothing_target_only:.0f} "
            f"topk_frequency_balance_enabled={topk_frequency_balance_enabled:.0f} "
            f"feature_standardization_train_only={feature_standardization_train_only:.0f} "
            f"residual_learning_rate={residual_learning_rate:.6f} "
            f"event_learning_rate={event_learning_rate:.6f} "
            f"local_l2_weight_decay={local_l2_decay:.6f} "
            f"event_residual_gain={event_residual_gain:.6f} "
            f"active_readout_pair_fraction={active_pair_fraction:.6f}"
        ),
    )

    topk_model_recall = require_metric(metrics, "topk_heldout_model_recall_at_k", "HVA predictor metrics")
    topk_persistence_recall = require_metric(
        metrics,
        "topk_heldout_persistence_recall_at_k",
        "HVA predictor metrics",
    )
    topk_train_frequency_recall = require_metric(
        metrics,
        "topk_heldout_train_frequency_recall_at_k",
        "HVA predictor metrics",
    )
    topk_no_learning_recall = require_metric(
        metrics,
        "topk_heldout_no_learning_recall_at_k",
        "HVA predictor metrics",
    )
    topk_time_recall = require_metric(
        metrics,
        "topk_heldout_temporal_block_shift_recall_at_k",
        "HVA predictor metrics",
    )
    topk_spatial_recall = require_metric(
        metrics,
        "topk_heldout_spatial_tile_shuffle_recall_at_k",
        "HVA predictor metrics",
    )
    topk_chance_recall = require_metric(metrics, "topk_heldout_chance_recall_at_k", "HVA predictor metrics")
    topk_chance_ratio = require_metric(
        metrics,
        "topk_heldout_model_recall_vs_chance_ratio",
        "HVA predictor metrics",
    )
    topk_model_ndcg = require_metric(metrics, "topk_heldout_model_ndcg_at_k", "HVA predictor metrics")
    topk_persistence_ndcg = require_metric(
        metrics,
        "topk_heldout_persistence_ndcg_at_k",
        "HVA predictor metrics",
    )
    topk_train_frequency_ndcg = require_metric(
        metrics,
        "topk_heldout_train_frequency_ndcg_at_k",
        "HVA predictor metrics",
    )
    topk_no_learning_ndcg = require_metric(
        metrics,
        "topk_heldout_no_learning_ndcg_at_k",
        "HVA predictor metrics",
    )
    topk_time_ndcg = require_metric(
        metrics,
        "topk_heldout_temporal_block_shift_ndcg_at_k",
        "HVA predictor metrics",
    )
    topk_spatial_ndcg = require_metric(
        metrics,
        "topk_heldout_spatial_tile_shuffle_ndcg_at_k",
        "HVA predictor metrics",
    )
    topk_model_mrr = require_metric(metrics, "topk_heldout_model_mrr", "HVA predictor metrics")
    topk_train_frequency_mrr = require_metric(
        metrics,
        "topk_heldout_train_frequency_mrr",
        "HVA predictor metrics",
    )
    topk_rel_persistence = require_metric(
        metrics,
        "topk_heldout_relative_improvement_vs_persistence",
        "HVA predictor metrics",
    )
    topk_rel_train_frequency = require_metric(
        metrics,
        "topk_heldout_relative_improvement_vs_train_frequency",
        "HVA predictor metrics",
    )
    topk_rel_no_learning = require_metric(
        metrics,
        "topk_heldout_relative_improvement_vs_no_learning",
        "HVA predictor metrics",
    )
    topk_model_gain = require_metric(
        metrics,
        "topk_heldout_model_gain_vs_train_frequency",
        "HVA predictor metrics",
    )
    topk_time_gain = require_metric(
        metrics,
        "topk_heldout_temporal_block_shift_gain_vs_train_frequency",
        "HVA predictor metrics",
    )
    topk_spatial_gain = require_metric(
        metrics,
        "topk_heldout_spatial_tile_shuffle_gain_vs_train_frequency",
        "HVA predictor metrics",
    )
    topk_time_retained = require_metric(
        metrics,
        "topk_heldout_temporal_block_shift_retained_fraction",
        "HVA predictor metrics",
    )
    topk_spatial_retained = require_metric(
        metrics,
        "topk_heldout_spatial_tile_shuffle_retained_fraction",
        "HVA predictor metrics",
    )
    repeat_avg_recall = metrics.get("topk_heldout_repeat_avg_model_recall_at_k", 0.0)
    repeat_avg_persistence_recall = metrics.get("topk_heldout_repeat_avg_persistence_recall_at_k", 0.0)
    repeat_avg_train_freq_recall = metrics.get("topk_heldout_repeat_avg_train_frequency_recall_at_k", 0.0)
    smooth_recall = metrics.get("topk_heldout_repeat_avg_smooth_model_recall_at_k", 0.0)
    smooth_persistence_recall = metrics.get("topk_heldout_repeat_avg_smooth_persistence_recall_at_k", 0.0)
    smooth_train_freq_recall = metrics.get("topk_heldout_repeat_avg_smooth_train_frequency_recall_at_k", 0.0)
    smooth_ndcg = metrics.get("topk_heldout_repeat_avg_smooth_model_ndcg_at_k", 0.0)
    smooth_persistence_ndcg = metrics.get("topk_heldout_repeat_avg_smooth_persistence_ndcg_at_k", 0.0)
    smooth_train_freq_ndcg = metrics.get("topk_heldout_repeat_avg_smooth_train_frequency_ndcg_at_k", 0.0)
    smooth_mass = metrics.get("topk_heldout_repeat_avg_smooth_model_captured_ideal_mass_at_k", 0.0)
    smooth_persistence_mass = metrics.get(
        "topk_heldout_repeat_avg_smooth_persistence_captured_ideal_mass_at_k",
        0.0,
    )
    smooth_train_freq_mass = metrics.get(
        "topk_heldout_repeat_avg_smooth_train_frequency_captured_ideal_mass_at_k",
        0.0,
    )
    print(
        "INFO hva_predictor_denoised_topk_metrics "
        f"raw_model_recall_at_k={topk_model_recall:.6f} "
        f"raw_persistence_recall_at_k={topk_persistence_recall:.6f} "
        f"raw_train_frequency_recall_at_k={topk_train_frequency_recall:.6f} "
        f"repeat_avg_model_recall_at_k={repeat_avg_recall:.6f} "
        f"repeat_avg_persistence_recall_at_k={repeat_avg_persistence_recall:.6f} "
        f"repeat_avg_train_frequency_recall_at_k={repeat_avg_train_freq_recall:.6f} "
        f"repeat_avg_smooth_model_recall_at_k={smooth_recall:.6f} "
        f"repeat_avg_smooth_persistence_recall_at_k={smooth_persistence_recall:.6f} "
        f"repeat_avg_smooth_train_frequency_recall_at_k={smooth_train_freq_recall:.6f} "
        f"repeat_avg_smooth_model_ndcg_at_k={smooth_ndcg:.6f} "
        f"repeat_avg_smooth_persistence_ndcg_at_k={smooth_persistence_ndcg:.6f} "
        f"repeat_avg_smooth_train_frequency_ndcg_at_k={smooth_train_freq_ndcg:.6f} "
        f"repeat_avg_smooth_model_captured_ideal_mass_at_k={smooth_mass:.6f} "
        f"repeat_avg_smooth_persistence_captured_ideal_mass_at_k={smooth_persistence_mass:.6f} "
        f"repeat_avg_smooth_train_frequency_captured_ideal_mass_at_k={smooth_train_freq_mass:.6f} "
        f"target_smooth_radius_tiles={topk_target_smooth_radius:.0f}"
    )
    topk_weight_l1 = require_metric(metrics, "topk_weight_l1", "HVA predictor metrics")
    topk_weight_max_abs = require_metric(metrics, "topk_weight_max_abs", "HVA predictor metrics")
    topk_bias_l1 = require_metric(metrics, "topk_bias_l1", "HVA predictor metrics")
    future_target_horizon_frames = require_metric(
        metrics,
        "future_target_horizon_frames",
        "HVA predictor metrics",
    )
    topk_split_safety_horizon_frames = require_metric(
        metrics,
        "topk_split_safety_horizon_frames",
        "HVA predictor metrics",
    )
    topk_local_readout = require_metric(metrics, "topk_local_readout_enabled", "HVA predictor metrics")
    topk_dense_readout = require_metric(metrics, "topk_dense_all_to_all_readout_enabled", "HVA predictor metrics")
    topk_local_radius = require_metric(metrics, "topk_local_radius_tiles", "HVA predictor metrics")
    local_radius_for_topk = require_metric(metrics, "local_radius_tiles", "HVA predictor metrics")
    topk_active_pair_fraction = require_metric(
        metrics,
        "topk_active_readout_pair_fraction",
        "HVA predictor metrics",
    )
    topk_local_pair_count = require_metric(metrics, "topk_local_pair_count", "HVA predictor metrics")
    topk_distant_pair_count = require_metric(metrics, "topk_distant_pair_count", "HVA predictor metrics")
    topk_local_nonzero_pair_count = require_metric(
        metrics,
        "topk_local_nonzero_pair_count",
        "HVA predictor metrics",
    )
    topk_distant_nonzero_pair_count = require_metric(
        metrics,
        "topk_distant_nonzero_pair_count",
        "HVA predictor metrics",
    )
    topk_local_abs_weight_sum = require_metric(
        metrics,
        "topk_local_abs_weight_sum",
        "HVA predictor metrics",
    )
    topk_distant_abs_weight_sum = require_metric(
        metrics,
        "topk_distant_abs_weight_sum",
        "HVA predictor metrics",
    )
    topk_distant_abs_weight_max = require_metric(
        metrics,
        "topk_distant_abs_weight_max",
        "HVA predictor metrics",
    )
    topk_local_abs_weight_mean = require_metric(
        metrics,
        "topk_local_abs_weight_mean",
        "HVA predictor metrics",
    )
    topk_distant_abs_weight_mean = require_metric(
        metrics,
        "topk_distant_abs_weight_mean",
        "HVA predictor metrics",
    )
    topk_diagonal_abs_weight_mean = require_metric(
        metrics,
        "topk_diagonal_abs_weight_mean",
        "HVA predictor metrics",
    )
    topk_offdiagonal_abs_weight_mean = require_metric(
        metrics,
        "topk_offdiagonal_abs_weight_mean",
        "HVA predictor metrics",
    )
    heldout_topk_rows = [row for row in heldout_rows if row.topk_sample_valid == 1]
    expected_future_target_horizon = max(event_window_frames, topk_future_window_frames)
    heldout_start_frame_int = int(round(require_metric(metrics, "heldout_start_frame", "HVA predictor metrics")))
    future_target_horizon_int = int(round(future_target_horizon_frames))
    topk_horizon_safety_ok = (
        abs(future_target_horizon_frames - expected_future_target_horizon) <= 1.0e-6
        and abs(topk_split_safety_horizon_frames - future_target_horizon_frames) <= 1.0e-6
        and future_target_horizon_int >= int(round(topk_future_window_frames))
        and future_target_horizon_int >= int(round(event_window_frames))
        and all(
            row.target_frame_index + future_target_horizon_int <= heldout_start_frame_int
            for row in train_rows
        )
        and all(
            row.frame_index >= heldout_start_frame_int
            and row.target_frame_index >= heldout_start_frame_int
            for row in heldout_rows
        )
    )
    overall_ok &= print_result(
        topk_horizon_safety_ok,
        "hva_predictor_topk_horizon_safety",
        (
            f"future_target_horizon_frames={future_target_horizon_frames:.0f} "
            f"expected_future_target_horizon_frames={expected_future_target_horizon:.0f} "
            f"topk_split_safety_horizon_frames={topk_split_safety_horizon_frames:.0f} "
            f"event_window_frames={event_window_frames:.0f} "
            f"topk_future_window_frames={topk_future_window_frames:.0f} "
            f"heldout_start_frame={heldout_start_frame_int} "
            f"train_rows={len(train_rows)} heldout_rows={len(heldout_rows)}"
        ),
    )
    topk_weight_sum_match = abs((topk_local_abs_weight_sum + topk_distant_abs_weight_sum) - topk_weight_l1) <= max(
        1.0e-6,
        1.0e-6 * max(1.0, topk_weight_l1),
    )
    topk_pair_count_match = abs((topk_local_pair_count + topk_distant_pair_count) - (tile_count * tile_count)) <= 1.0e-6
    topk_locality_ok = (
        topk_local_readout == 1.0
        and topk_dense_readout == 0.0
        and abs(topk_local_radius - local_radius_for_topk) <= 1.0e-6
        and topk_pair_count_match
        and topk_local_pair_count > 0.0
        and topk_distant_pair_count > 0.0
        and topk_local_nonzero_pair_count > 0.0
        and topk_distant_nonzero_pair_count == 0.0
        and topk_local_abs_weight_sum > 1.0e-9
        and topk_distant_abs_weight_sum <= 1.0e-9
        and topk_distant_abs_weight_max <= 1.0e-9
        and topk_distant_abs_weight_mean <= 1.0e-9
        and topk_local_abs_weight_mean > 0.0
        and topk_diagonal_abs_weight_mean > 0.0
        and topk_offdiagonal_abs_weight_mean >= 0.0
        and topk_weight_sum_match
        and abs(topk_active_pair_fraction - active_pair_fraction) <= 1.0e-6
    )
    overall_ok &= print_result(
        topk_locality_ok,
        "hva_predictor_topk_head_locality_structure",
        (
            f"topk_local_readout_enabled={topk_local_readout:.0f} "
            f"topk_dense_all_to_all_readout_enabled={topk_dense_readout:.0f} "
            f"topk_local_radius_tiles={topk_local_radius:.0f} "
            f"residual_event_local_radius_tiles={local_radius_for_topk:.0f} "
            f"topk_active_readout_pair_fraction={topk_active_pair_fraction:.6f} "
            f"active_readout_pair_fraction={active_pair_fraction:.6f} "
            f"topk_local_pair_count={topk_local_pair_count:.0f} "
            f"topk_distant_pair_count={topk_distant_pair_count:.0f} "
            f"topk_local_nonzero_pair_count={topk_local_nonzero_pair_count:.0f} "
            f"topk_distant_nonzero_pair_count={topk_distant_nonzero_pair_count:.0f} "
            f"topk_local_abs_weight_sum={topk_local_abs_weight_sum:.6f} "
            f"topk_distant_abs_weight_sum={topk_distant_abs_weight_sum:.6e} "
            f"topk_distant_abs_weight_max={topk_distant_abs_weight_max:.6e} "
            f"topk_local_abs_weight_mean={topk_local_abs_weight_mean:.6f} "
            f"topk_distant_abs_weight_mean={topk_distant_abs_weight_mean:.6e} "
            f"topk_diagonal_abs_weight_mean={topk_diagonal_abs_weight_mean:.6f} "
            f"topk_offdiagonal_abs_weight_mean={topk_offdiagonal_abs_weight_mean:.6f} "
            f"topk_weight_sum_match={1 if topk_weight_sum_match else 0}"
        ),
    )
    topk_prediction_fields_ok = (
        len(heldout_topk_rows) == int(round(topk_heldout_valid_count * tile_count))
        and all(row.target_channel == "l23e" for row in heldout_topk_rows)
        and all(row.topk_target in (0, 1) for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_target_value_norm) and row.topk_target_value_norm >= 0.0 for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_model_score) for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_model_prob) and 0.0 <= row.topk_model_prob <= 1.0 for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_persistence_score) for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_train_frequency_score) for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_temporal_block_shift_score) for row in heldout_topk_rows)
        and all(math.isfinite(row.topk_spatial_tile_shuffle_score) for row in heldout_topk_rows)
    )
    topk_success_ok = (
        topk_prediction_fields_ok
        and topk_horizon_safety_ok
        and topk_locality_ok
        and topk_heldout_valid_count >= 2.0
        and topk_model_recall >= 1.10 * topk_persistence_recall
        and topk_model_recall >= 1.10 * topk_train_frequency_recall
        and topk_model_recall >= 1.10 * topk_no_learning_recall
        and topk_chance_ratio >= 1.50
        and topk_model_ndcg >= 1.10 * topk_train_frequency_ndcg
        and topk_model_mrr >= topk_train_frequency_mrr
        and topk_model_gain > 0.0
        and topk_time_gain <= (0.30 * topk_model_gain)
        and topk_spatial_gain <= (0.70 * topk_model_gain)
        and topk_time_retained <= 0.30
        and topk_spatial_retained <= 0.70
        and topk_weight_l1 > 1.0e-9
        and topk_weight_max_abs <= (weight_clip + 1.0e-6)
        and topk_bias_l1 > 0.0
    )
    overall_ok &= print_result(
        topk_success_ok,
        "hva_predictor_l23e_future_topk_success",
        (
            f"heldout_valid_sample_count={topk_heldout_valid_count:.0f} "
            f"heldout_topk_rows={len(heldout_topk_rows)} "
            f"topk_k={topk_k:.0f} tile_count={tile_count:.0f} "
            f"model_recall_at_k={topk_model_recall:.6f} "
            f"persistence_recall_at_k={topk_persistence_recall:.6f} "
            f"train_frequency_recall_at_k={topk_train_frequency_recall:.6f} "
            f"no_learning_recall_at_k={topk_no_learning_recall:.6f} "
            f"time_shuffle_recall_at_k={topk_time_recall:.6f} "
            f"spatial_shuffle_recall_at_k={topk_spatial_recall:.6f} "
            f"chance_recall_at_k={topk_chance_recall:.6f} "
            f"model_chance_ratio={topk_chance_ratio:.6f} "
            f"model_ndcg_at_k={topk_model_ndcg:.6f} "
            f"persistence_ndcg_at_k={topk_persistence_ndcg:.6f} "
            f"train_frequency_ndcg_at_k={topk_train_frequency_ndcg:.6f} "
            f"no_learning_ndcg_at_k={topk_no_learning_ndcg:.6f} "
            f"time_shuffle_ndcg_at_k={topk_time_ndcg:.6f} "
            f"spatial_shuffle_ndcg_at_k={topk_spatial_ndcg:.6f} "
            f"model_mrr={topk_model_mrr:.6f} "
            f"train_frequency_mrr={topk_train_frequency_mrr:.6f} "
            f"relative_vs_persistence={topk_rel_persistence:.6f} "
            f"relative_vs_train_frequency={topk_rel_train_frequency:.6f} "
            f"relative_vs_no_learning={topk_rel_no_learning:.6f} "
            f"model_gain_vs_train_frequency={topk_model_gain:.6f} "
            f"time_shuffle_gain_vs_train_frequency={topk_time_gain:.6f} "
            f"spatial_shuffle_gain_vs_train_frequency={topk_spatial_gain:.6f} "
            f"time_shuffle_retained_fraction={topk_time_retained:.6f} "
            f"spatial_shuffle_retained_fraction={topk_spatial_retained:.6f} "
            f"topk_weight_l1={topk_weight_l1:.6f} "
            f"topk_weight_max_abs={topk_weight_max_abs:.6f} "
            f"topk_bias_l1={topk_bias_l1:.6f} "
            f"prediction_fields_ok={1 if topk_prediction_fields_ok else 0}"
        ),
    )

    if require_population_prediction:
        population_metrics = compute_hva_population_prediction_metrics(
            heldout_topk_rows,
            tile_count,
        )
        population_ok = (
            topk_prediction_fields_ok
            and topk_horizon_safety_ok
            and topk_locality_ok
            and topk_target_l23e_only == 1.0
            and topk_input_l23e_only == 1.0
            and topk_feedback_enabled == 0.0
            and feature_non_l23_inputs == 0.0
            and feature_future_leakage == 0.0
            and population_metrics["tile_grid_side"] == 10.0
            and population_metrics["model_ndcg_at5"] >= 0.70
            and population_metrics["model_captured_ideal_mass_at5"] >= 0.70
            and population_metrics["model_ndcg_at5_vs_persistence_ratio"] >= 1.20
            and population_metrics["model_captured_ideal_mass_at5_vs_persistence_ratio"] >= 1.20
        )
        overall_ok &= print_result(
            population_ok,
            "hva_population_prediction_repeat_avg_smooth",
            (
                f"heldout_topk_rows={population_metrics['heldout_topk_row_count']:.0f} "
                f"target_frame_count={population_metrics['target_frame_count']:.0f} "
                f"evaluated_frame_count={population_metrics['evaluated_frame_count']:.0f} "
                f"skipped_frame_count={population_metrics['skipped_frame_count']:.0f} "
                f"evaluated_sample_count={population_metrics['evaluated_sample_count']:.0f} "
                f"skipped_sample_count={population_metrics['skipped_sample_count']:.0f} "
                f"repeat_count_mean={population_metrics['repeat_count_mean']:.6f} "
                f"tile_grid_side={population_metrics['tile_grid_side']:.0f} "
                f"smoothing_kernel=radius1_121242121_normalized "
                f"target_repeat_averaged=1 score_repeat_averaged=0 "
                f"target_smoothing_evaluation_only=1 "
                f"model_ndcg_at5={population_metrics['model_ndcg_at5']:.6f} "
                f"persistence_ndcg_at5={population_metrics['persistence_ndcg_at5']:.6f} "
                f"train_frequency_ndcg_at5={population_metrics['train_frequency_ndcg_at5']:.6f} "
                f"no_learning_ndcg_at5={population_metrics['no_learning_ndcg_at5']:.6f} "
                f"time_shuffle_ndcg_at5={population_metrics['time_shuffle_ndcg_at5']:.6f} "
                f"spatial_shuffle_ndcg_at5={population_metrics['spatial_shuffle_ndcg_at5']:.6f} "
                f"model_captured_ideal_mass_at5="
                f"{population_metrics['model_captured_ideal_mass_at5']:.6f} "
                f"persistence_captured_ideal_mass_at5="
                f"{population_metrics['persistence_captured_ideal_mass_at5']:.6f} "
                f"train_frequency_captured_ideal_mass_at5="
                f"{population_metrics['train_frequency_captured_ideal_mass_at5']:.6f} "
                f"no_learning_captured_ideal_mass_at5="
                f"{population_metrics['no_learning_captured_ideal_mass_at5']:.6f} "
                f"time_shuffle_captured_ideal_mass_at5="
                f"{population_metrics['time_shuffle_captured_ideal_mass_at5']:.6f} "
                f"spatial_shuffle_captured_ideal_mass_at5="
                f"{population_metrics['spatial_shuffle_captured_ideal_mass_at5']:.6f} "
                f"model_ndcg_at5_vs_persistence_ratio="
                f"{population_metrics['model_ndcg_at5_vs_persistence_ratio']:.6f} "
                f"model_captured_ideal_mass_at5_vs_persistence_ratio="
                f"{population_metrics['model_captured_ideal_mass_at5_vs_persistence_ratio']:.6f} "
                f"model_ndcg_at10={population_metrics['model_ndcg_at10']:.6f} "
                f"persistence_ndcg_at10={population_metrics['persistence_ndcg_at10']:.6f} "
                f"model_captured_ideal_mass_at10="
                f"{population_metrics['model_captured_ideal_mass_at10']:.6f} "
                f"persistence_captured_ideal_mass_at10="
                f"{population_metrics['persistence_captured_ideal_mass_at10']:.6f} "
                f"uniform_chance_captured_ideal_mass_at5="
                f"{population_metrics['uniform_chance_captured_ideal_mass_at5']:.6f} "
                f"topk_target_channel_l23e_only={topk_target_l23e_only:.0f} "
                f"topk_input_channel_l23e_only={topk_input_l23e_only:.0f} "
                f"feature_uses_non_l23_inputs={feature_non_l23_inputs:.0f} "
                f"feature_future_leakage_enabled={feature_future_leakage:.0f}"
            ),
        )

    model_mse = require_metric(metrics, "heldout_model_mse_norm", "HVA predictor metrics")
    model_raw_mse = require_metric(metrics, "heldout_model_raw_mse_norm", "HVA predictor metrics")
    residual_z_mse = require_metric(metrics, "heldout_model_residual_z_mse", "HVA predictor metrics")
    no_learning_mse = require_metric(metrics, "heldout_no_learning_mse_norm", "HVA predictor metrics")
    persistence_mse = require_metric(metrics, "heldout_persistence_mse_norm", "HVA predictor metrics")
    train_mean_mse = require_metric(metrics, "heldout_train_mean_mse_norm", "HVA predictor metrics")
    time_shuffle_mse = require_metric(metrics, "heldout_temporal_block_shift_mse_norm", "HVA predictor metrics")
    spatial_shuffle_mse = require_metric(metrics, "heldout_spatial_tile_shuffle_mse_norm", "HVA predictor metrics")
    corr = require_metric(metrics, "mean_tile_prediction_corr", "HVA predictor metrics")
    relative_no_learning = require_metric(metrics, "heldout_relative_improvement_vs_no_learning", "HVA predictor metrics")
    relative_persistence = require_metric(metrics, "heldout_relative_improvement_vs_persistence", "HVA predictor metrics")
    relative_train_mean = require_metric(metrics, "heldout_relative_improvement_vs_train_mean", "HVA predictor metrics")
    relative_time_shuffle = require_metric(metrics, "heldout_relative_improvement_vs_temporal_block_shift", "HVA predictor metrics")
    relative_spatial_shuffle = require_metric(metrics, "heldout_relative_improvement_vs_spatial_tile_shuffle", "HVA predictor metrics")
    residual_reporting_ok = (
        math.isfinite(model_mse)
        and math.isfinite(model_raw_mse)
        and math.isfinite(residual_z_mse)
        and abs(model_raw_mse - model_mse) <= 1.0e-12
        and math.isfinite(no_learning_mse)
        and math.isfinite(persistence_mse)
        and math.isfinite(train_mean_mse)
        and math.isfinite(relative_no_learning)
        and math.isfinite(relative_persistence)
        and math.isfinite(relative_train_mean)
        and math.isfinite(corr)
    )
    overall_ok &= print_result(
        residual_reporting_ok,
        "hva_predictor_residual_rate_reporting",
        (
            f"heldout_model_mse_norm={model_mse:.6f} "
            f"heldout_model_raw_mse_norm={model_raw_mse:.6f} "
            f"heldout_model_residual_z_mse={residual_z_mse:.6f} "
            f"heldout_no_learning_mse_norm={no_learning_mse:.6f} "
            f"heldout_persistence_mse_norm={persistence_mse:.6f} "
            f"heldout_train_mean_mse_norm={train_mean_mse:.6f} "
            f"heldout_relative_improvement_vs_no_learning={relative_no_learning:.6f} "
            f"heldout_relative_improvement_vs_persistence={relative_persistence:.6f} "
            f"heldout_relative_improvement_vs_train_mean={relative_train_mean:.6f} "
            f"mean_tile_prediction_corr={corr:.6f}"
        ),
    )

    print(
        "INFO hva_predictor_residual_rate_shuffle_reporting "
        f"heldout_model_mse_norm={model_mse:.6f} "
        f"heldout_temporal_block_shift_mse_norm={time_shuffle_mse:.6f} "
        f"heldout_spatial_tile_shuffle_mse_norm={spatial_shuffle_mse:.6f} "
        f"heldout_relative_improvement_vs_temporal_block_shift={relative_time_shuffle:.6f} "
        f"heldout_relative_improvement_vs_spatial_tile_shuffle={relative_spatial_shuffle:.6f}"
    )

    event_min_train_positive_count = require_metric(
        metrics,
        "event_min_train_positive_count",
        "HVA predictor metrics",
    )
    selected_event_tiles = [
        row
        for row in event_tiles
        if row.target_channel == "l23e" and row.selected == 1
    ]
    event_tile_export_ok = (
        len(event_tiles) == int(round(tile_count))
        and all(row.target_channel == "l23e" for row in event_tiles)
        and len(selected_event_tiles) == int(round(event_selected_tile_count))
        and len(selected_event_tiles) > 0
        and all(math.isfinite(row.threshold_norm) and row.threshold_norm >= 0.0 for row in event_tiles)
        and all(math.isfinite(row.threshold_hz) and row.threshold_hz >= 0.0 for row in event_tiles)
        and all(row.train_count == row.train_positive_count + row.train_negative_count for row in event_tiles)
        and all(row.heldout_count > 0 for row in event_tiles)
        and all(
            row.train_positive_count >= int(round(event_min_train_positive_count))
            and row.train_negative_count >= int(round(event_min_train_positive_count))
            for row in selected_event_tiles
        )
    )
    overall_ok &= print_result(
        event_tile_export_ok,
        "hva_predictor_l23e_event_tile_selection",
        (
            f"event_tile_rows={len(event_tiles)} tile_count={tile_count:.0f} "
            f"selected_event_tile_count={len(selected_event_tiles)} "
            f"metric_selected_event_tile_count={event_selected_tile_count:.0f} "
            f"event_min_train_positive_count={event_min_train_positive_count:.0f} "
            f"selected_train_positive_total={sum(row.train_positive_count for row in selected_event_tiles)} "
            f"selected_train_negative_total={sum(row.train_negative_count for row in selected_event_tiles)} "
            f"selected_heldout_positive_total={sum(row.heldout_positive_count for row in selected_event_tiles)}"
        ),
    )

    selected_event_count = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_prediction_count",
        "HVA predictor metrics",
    )
    selected_event_positive_count = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_positive_count",
        "HVA predictor metrics",
    )
    selected_event_positive_fraction = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_positive_fraction",
        "HVA predictor metrics",
    )
    selected_event_brier = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_model_brier",
        "HVA predictor metrics",
    )
    selected_event_persistence_brier = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_persistence_brier",
        "HVA predictor metrics",
    )
    selected_event_train_mean_brier = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_train_mean_brier",
        "HVA predictor metrics",
    )
    selected_event_no_learning_brier = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_no_learning_brier",
        "HVA predictor metrics",
    )
    selected_event_time_brier = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_temporal_block_shift_brier",
        "HVA predictor metrics",
    )
    selected_event_spatial_brier = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_spatial_tile_shuffle_brier",
        "HVA predictor metrics",
    )
    selected_event_rel_persistence = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_relative_improvement_vs_persistence",
        "HVA predictor metrics",
    )
    selected_event_rel_train_mean = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_relative_improvement_vs_train_mean",
        "HVA predictor metrics",
    )
    selected_event_rel_no_learning = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_relative_improvement_vs_no_learning",
        "HVA predictor metrics",
    )
    selected_event_rel_time = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_relative_improvement_vs_temporal_block_shift",
        "HVA predictor metrics",
    )
    selected_event_rel_spatial = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_relative_improvement_vs_spatial_tile_shuffle",
        "HVA predictor metrics",
    )
    selected_event_corr = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_prediction_corr",
        "HVA predictor metrics",
    )
    selected_event_logloss = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_model_logloss",
        "HVA predictor metrics",
    )
    selected_event_persistence_logloss = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_persistence_logloss",
        "HVA predictor metrics",
    )
    selected_event_train_mean_logloss = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_train_mean_logloss",
        "HVA predictor metrics",
    )
    selected_event_no_learning_logloss = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_no_learning_logloss",
        "HVA predictor metrics",
    )
    selected_event_time_logloss = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_temporal_block_shift_logloss",
        "HVA predictor metrics",
    )
    selected_event_spatial_logloss = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_spatial_tile_shuffle_logloss",
        "HVA predictor metrics",
    )
    selected_event_auc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_model_auc",
        "HVA predictor metrics",
    )
    selected_event_persistence_auc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_persistence_auc",
        "HVA predictor metrics",
    )
    selected_event_train_mean_auc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_train_mean_auc",
        "HVA predictor metrics",
    )
    selected_event_no_learning_auc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_no_learning_auc",
        "HVA predictor metrics",
    )
    selected_event_time_auc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_temporal_block_shift_auc",
        "HVA predictor metrics",
    )
    selected_event_spatial_auc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_spatial_tile_shuffle_auc",
        "HVA predictor metrics",
    )
    selected_event_auprc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_model_auprc",
        "HVA predictor metrics",
    )
    selected_event_persistence_auprc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_persistence_auprc",
        "HVA predictor metrics",
    )
    selected_event_train_mean_auprc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_train_mean_auprc",
        "HVA predictor metrics",
    )
    selected_event_no_learning_auprc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_no_learning_auprc",
        "HVA predictor metrics",
    )
    selected_event_time_auprc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_temporal_block_shift_auprc",
        "HVA predictor metrics",
    )
    selected_event_spatial_auprc = require_metric(
        metrics,
        "l23e_event_selected_tiles_heldout_event_spatial_tile_shuffle_auprc",
        "HVA predictor metrics",
    )
    selected_heldout_rows = [row for row in heldout_rows if row.event_tile_selected == 1]
    event_probs_finite = all(
        math.isfinite(row.predicted_event_prob)
        and 0.0 <= row.predicted_event_prob <= 1.0
        and math.isfinite(row.train_event_rate)
        and 0.0 <= row.train_event_rate <= 1.0
        and row.target_event in (0, 1)
        for row in heldout_rows
    )
    event_success_ok = (
        selected_event_count >= 8.0
        and selected_event_positive_count >= 1.0
        and 0.0 < selected_event_positive_fraction < 1.0
        and len(selected_heldout_rows) == int(round(selected_event_count))
        and event_probs_finite
        and math.isfinite(selected_event_brier)
        and math.isfinite(selected_event_logloss)
        and math.isfinite(selected_event_auc)
        and math.isfinite(selected_event_auprc)
        and math.isfinite(selected_event_corr)
        and selected_event_persistence_brier > 0.0
        and selected_event_train_mean_brier > 0.0
        and selected_event_no_learning_brier > 0.0
        and selected_event_time_brier > 0.0
        and selected_event_spatial_brier > 0.0
        and selected_event_rel_persistence >= 0.001
        and selected_event_rel_train_mean >= 0.001
        and selected_event_rel_no_learning >= 0.001
        and selected_event_rel_time >= 0.001
        and selected_event_rel_spatial >= 0.001
        and selected_event_logloss <= selected_event_persistence_logloss - 0.001
        and selected_event_logloss <= selected_event_train_mean_logloss - 0.001
        and selected_event_logloss <= selected_event_no_learning_logloss - 0.001
        and selected_event_logloss <= selected_event_time_logloss - 0.001
        and selected_event_logloss <= selected_event_spatial_logloss - 0.001
        and selected_event_auc >= 0.52
        and selected_event_auc >= selected_event_persistence_auc + 0.001
        and selected_event_auc >= selected_event_train_mean_auc + 0.001
        and selected_event_auc >= selected_event_no_learning_auc + 0.001
        and selected_event_auc >= selected_event_time_auc + 0.001
        and selected_event_auc >= selected_event_spatial_auc + 0.001
        and selected_event_auprc >= selected_event_positive_fraction + 0.005
        and selected_event_auprc >= selected_event_persistence_auprc + 0.001
        and selected_event_auprc >= selected_event_train_mean_auprc + 0.001
        and selected_event_auprc >= selected_event_no_learning_auprc + 0.001
        and selected_event_auprc >= selected_event_time_auprc + 0.001
        and selected_event_auprc >= selected_event_spatial_auprc + 0.001
    )
    print(
        "INFO hva_predictor_l23e_event_window_hazard_reporting "
        f"legacy_gate_passed={1 if event_success_ok else 0} "
        f"selected_heldout_event_count={selected_event_count:.0f} "
        f"selected_heldout_event_positive_count={selected_event_positive_count:.0f} "
        f"selected_heldout_event_positive_fraction={selected_event_positive_fraction:.6f} "
        f"selected_heldout_event_model_brier={selected_event_brier:.6f} "
        f"selected_heldout_event_persistence_brier={selected_event_persistence_brier:.6f} "
        f"selected_heldout_event_train_mean_brier={selected_event_train_mean_brier:.6f} "
        f"selected_heldout_event_no_learning_brier={selected_event_no_learning_brier:.6f} "
        f"selected_heldout_event_time_shuffle_brier={selected_event_time_brier:.6f} "
        f"selected_heldout_event_spatial_shuffle_brier={selected_event_spatial_brier:.6f} "
        f"selected_heldout_event_logloss={selected_event_logloss:.6f} "
        f"selected_heldout_event_persistence_logloss={selected_event_persistence_logloss:.6f} "
        f"selected_heldout_event_time_shuffle_logloss={selected_event_time_logloss:.6f} "
        f"selected_heldout_event_spatial_shuffle_logloss={selected_event_spatial_logloss:.6f} "
        f"selected_heldout_event_auc={selected_event_auc:.6f} "
        f"persistence_auc={selected_event_persistence_auc:.6f} "
        f"time_shuffle_auc={selected_event_time_auc:.6f} "
        f"spatial_shuffle_auc={selected_event_spatial_auc:.6f} "
        f"selected_heldout_event_auprc={selected_event_auprc:.6f} "
        f"persistence_auprc={selected_event_persistence_auprc:.6f} "
        f"time_shuffle_auprc={selected_event_time_auprc:.6f} "
        f"spatial_shuffle_auprc={selected_event_spatial_auprc:.6f} "
        f"rel_vs_persistence={selected_event_rel_persistence:.6f} "
        f"rel_vs_train_mean={selected_event_rel_train_mean:.6f} "
        f"rel_vs_no_learning={selected_event_rel_no_learning:.6f} "
        f"rel_vs_time_shuffle={selected_event_rel_time:.6f} "
        f"rel_vs_spatial_shuffle={selected_event_rel_spatial:.6f} "
        f"prediction_corr={selected_event_corr:.6f} "
        "gate=topk_primary"
    )

    single_frame_auc = require_metric(
        metrics,
        "l23e_single_frame_event_selected_tiles_heldout_event_model_auc",
        "HVA predictor metrics",
    )
    single_frame_auprc = require_metric(
        metrics,
        "l23e_single_frame_event_selected_tiles_heldout_event_model_auprc",
        "HVA predictor metrics",
    )
    single_frame_brier = require_metric(
        metrics,
        "l23e_single_frame_event_selected_tiles_heldout_event_model_brier",
        "HVA predictor metrics",
    )
    print(
        "INFO hva_predictor_l23e_single_frame_event_reporting "
        f"selected_single_frame_event_auc={single_frame_auc:.6f} "
        f"selected_single_frame_event_auprc={single_frame_auprc:.6f} "
        f"selected_single_frame_event_brier={single_frame_brier:.6f} "
        "gate=event_window_only"
    )

    all_event_count = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_prediction_count",
        "HVA predictor metrics",
    )
    all_event_positive_fraction = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_positive_fraction",
        "HVA predictor metrics",
    )
    all_event_brier = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_model_brier",
        "HVA predictor metrics",
    )
    all_event_logloss = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_model_logloss",
        "HVA predictor metrics",
    )
    all_event_auc = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_model_auc",
        "HVA predictor metrics",
    )
    all_event_auprc = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_model_auprc",
        "HVA predictor metrics",
    )
    all_event_rel_no_learning = require_metric(
        metrics,
        "l23e_event_all_tiles_heldout_event_relative_improvement_vs_no_learning",
        "HVA predictor metrics",
    )
    all_event_reporting_ok = (
        all_event_count == len(heldout_rows)
        and math.isfinite(all_event_brier)
        and math.isfinite(all_event_logloss)
        and math.isfinite(all_event_auc)
        and math.isfinite(all_event_auprc)
        and math.isfinite(all_event_positive_fraction)
        and math.isfinite(all_event_rel_no_learning)
        and 0.0 <= all_event_positive_fraction <= 1.0
    )
    overall_ok &= print_result(
        all_event_reporting_ok,
        "hva_predictor_l23e_event_all_tile_reporting",
        (
            f"all_heldout_event_count={all_event_count:.0f} "
            f"heldout_rows={len(heldout_rows)} "
            f"all_heldout_event_positive_fraction={all_event_positive_fraction:.6f} "
            f"all_heldout_event_model_brier={all_event_brier:.6f} "
            f"all_heldout_event_model_logloss={all_event_logloss:.6f} "
            f"all_heldout_event_model_auc={all_event_auc:.6f} "
            f"all_heldout_event_model_auprc={all_event_auprc:.6f} "
            f"all_heldout_event_relative_improvement_vs_no_learning={all_event_rel_no_learning:.6f}"
        ),
    )

    pred_min = require_metric(metrics, "heldout_prediction_min_norm", "HVA predictor metrics")
    pred_max = require_metric(metrics, "heldout_prediction_max_norm", "HVA predictor metrics")
    bounded_ok = (
        pred_min >= -1.0e-9
        and pred_max <= 1.0 + 1.0e-9
        and all(0.0 <= row.predicted_state_norm <= 1.0 for row in heldout_rows)
        and all(math.isfinite(row.predicted_state_norm) for row in heldout_rows)
        and all(math.isfinite(row.target_residual_z) for row in heldout_rows)
        and all(math.isfinite(row.predicted_residual_z) for row in heldout_rows)
        and all(math.isfinite(row.target_residual_norm) for row in heldout_rows)
        and all(math.isfinite(row.predicted_residual_norm) for row in heldout_rows)
        and all(row.train_residual_std_norm > 0.0 for row in heldout_rows)
        and all(math.isfinite(row.event_window_target_state_norm) and row.event_window_target_state_norm >= 0.0 for row in heldout_rows)
        and all(math.isfinite(row.event_threshold_norm) and row.event_threshold_norm >= 0.0 for row in heldout_rows)
        and all(math.isfinite(row.predicted_event_prob) and 0.0 <= row.predicted_event_prob <= 1.0 for row in heldout_rows)
        and all(row.single_frame_target_event in (0, 1) for row in heldout_rows)
        and all(math.isfinite(row.event_error) for row in heldout_rows)
    )
    overall_ok &= print_result(
        bounded_ok,
        "hva_predictor_output_boundedness",
        (
            f"heldout_prediction_min_norm={pred_min:.6f} "
            f"heldout_prediction_max_norm={pred_max:.6f} "
            f"heldout_rows={len(heldout_rows)} residual_fields_finite=1"
        ),
    )

    changed_weight_count = sum(row.abs_weight_sum_after > 1.0e-9 for row in weights)
    nonzero_before_count = sum(abs(row.w_before) > 1.0e-12 for row in weights)
    target_mean_hz = require_metric(metrics, "target_mean_hz", "HVA predictor metrics")
    prediction_mean_hz = require_metric(metrics, "prediction_mean_hz", "HVA predictor metrics")
    finite_predictions = all(
        math.isfinite(row.predicted_state_norm)
        and math.isfinite(row.target_state_norm)
        and math.isfinite(row.error_rate_hz)
        for row in predictions
    )
    weights_ok = (
        finite_predictions
        and changed_weight_count > 0
        and nonzero_before_count == 0
    )
    overall_ok &= print_result(
        weights_ok,
        "hva_predictor_weights_host_update",
        (
            f"changed_weight_count={changed_weight_count} "
            f"nonzero_before_count={nonzero_before_count} "
            f"target_mean_hz={target_mean_hz:.6f} "
            f"prediction_mean_hz={prediction_mean_hz:.6f}"
        ),
    )

    local_radius = int(round(require_metric(metrics, "local_radius_tiles", "HVA predictor metrics")))
    local_weights = [
        row.abs_weight_sum_after
        for row in weights
        if row.manhattan_distance_tiles <= local_radius
    ]
    distant_weights = [
        row.abs_weight_sum_after
        for row in weights
        if row.manhattan_distance_tiles > local_radius
    ]
    diagonal_weights = [
        row.abs_weight_sum_after
        for row in weights
        if row.pre_tile_id == row.post_tile_id
    ]
    offdiagonal_weights = [
        row.abs_weight_sum_after
        for row in weights
        if row.pre_tile_id != row.post_tile_id
    ]
    local_mean = mean(local_weights) if local_weights else math.nan
    distant_mean = mean(distant_weights) if distant_weights else math.nan
    diagonal_mean = mean(diagonal_weights) if diagonal_weights else math.nan
    offdiagonal_mean = mean(offdiagonal_weights) if offdiagonal_weights else math.nan
    coordinate_structure_ok = bool(
        local_weights
        and distant_weights
        and diagonal_weights
        and offdiagonal_weights
        and all(math.isfinite(row.distance_tiles) and row.distance_tiles >= 0.0 for row in weights)
        and local_mean >= (1.10 * distant_mean)
        and (local_mean - distant_mean) >= max(1.0e-6, 0.05 * max(distant_mean, 1.0e-12))
        and diagonal_mean > 0.0
    )
    overall_ok &= print_result(
        coordinate_structure_ok,
        "hva_predictor_weight_structure",
        (
            f"local_count={len(local_weights)} distant_count={len(distant_weights)} "
            f"diagonal_count={len(diagonal_weights)} offdiagonal_count={len(offdiagonal_weights)} "
            f"local_radius_tiles={local_radius} "
            f"local_abs_weight_mean={local_mean:.6f} "
            f"distant_abs_weight_mean={distant_mean:.6f} "
            f"local_distant_ratio={(local_mean / distant_mean) if distant_mean > 0.0 else math.inf:.6f} "
            f"diagonal_abs_weight_mean={diagonal_mean:.6f} "
            f"offdiagonal_abs_weight_mean={offdiagonal_mean:.6f}"
        ),
    )

    return overall_ok


def validate_natural_video_event_timing(run: RunData) -> bool:
    overall_ok = True
    timing_enabled = run.summary.get("video_event_timing_enabled", 0.0)
    population_rows = run.video_event_population_bin_rows
    site_rows = run.video_event_site_bin_rows
    expected_events = int(round(run.summary.get("video_event_frame_count", 0.0)))
    conditions = {row.condition for row in population_rows} if population_rows is not None else set()
    populations = {row.population for row in population_rows} if population_rows is not None else set()
    unique_events = (
        len({(row.condition, row.repeat_index, row.event_index) for row in population_rows})
        if population_rows is not None
        else 0
    )
    required_populations = {"l4e", "l23e", "l23pv", "l23som"}
    artifacts_available = (
        timing_enabled == 1.0
        and population_rows is not None
        and "event" in conditions
        and required_populations.issubset(populations)
        and expected_events > 0
    )
    overall_ok &= print_result(
        artifacts_available,
        "natural_video_event_artifacts_available",
        (
            f"video_event_timing_enabled={timing_enabled:.6f} "
            f"summary_event_count={expected_events} "
            f"population_rows={len(population_rows) if population_rows is not None else 0} "
            f"conditions={','.join(sorted(conditions)) if conditions else 'none'} "
            f"populations={','.join(sorted(populations)) if populations else 'none'} "
            f"unique_condition_events={unique_events}"
        ),
    )
    site_count = (
        len({row.site_id for row in site_rows if row.site_id is not None})
        if site_rows is not None
        else 0
    )
    overall_ok &= print_result(
        site_rows is not None and site_count > 0,
        "natural_video_event_site_artifacts_available",
        f"site_rows={len(site_rows) if site_rows is not None else 0} site_count={site_count}",
    )
    if not artifacts_available or population_rows is None:
        return overall_ok

    gray_required = run.summary.get("video_event_gray_control_count", 0.0) > 0.0
    blank_required = run.summary.get("video_event_blank_control_count", 0.0) > 0.0
    controls_available = (
        (not gray_required or "gray_control" in conditions)
        and (not blank_required or "blank_control" in conditions)
    )
    overall_ok &= print_result(
        controls_available,
        "natural_video_event_controls_available",
        (
            f"gray_required={1 if gray_required else 0} "
            f"blank_required={1 if blank_required else 0} "
            f"conditions={','.join(sorted(conditions))}"
        ),
    )

    series = {
        population: mean_event_bin_series(population_rows, "event", population)
        for population in sorted(required_populations)
    }
    metrics = {population: event_response_metrics(pop_series) for population, pop_series in series.items()}
    gray_matched_series = {
        population: matched_event_minus_control_series(population_rows, population, "gray_control")
        for population in sorted(required_populations)
    } if "gray_control" in conditions else {}
    causal_series = {
        population: (
            gray_matched_series.get(population)
            if gray_matched_series.get(population)
            else series[population]
        )
        for population in sorted(required_populations)
    }
    causal_metrics = {
        population: event_response_metrics(pop_series)
        for population, pop_series in causal_series.items()
    }
    causal_source = "event_minus_gray_control" if any(gray_matched_series.values()) else "event_minus_baseline"
    l4 = metrics["l4e"]
    bin_ms = run.summary.get("video_event_bin_ms", 0.0)
    post_ms = run.summary.get("video_event_post_ms", 0.0)
    l4_peak_delta = l4["post_peak"] - l4["baseline_mean"]
    l4_onset_ok = (
        math.isfinite(l4["onset_latency_ms"])
        and 0.0 <= l4["onset_latency_ms"] <= max(20.0, 2.0 * bin_ms)
        and l4_peak_delta > max(1.0e-9, l4["baseline_std"])
    )
    overall_ok &= print_result(
        l4_onset_ok,
        "natural_video_event_l4_onset_peak",
        (
            f"bin_ms={bin_ms:.6f} pre_bins={l4['pre_bin_count']:.0f} post_bins={l4['post_bin_count']:.0f} "
            f"l4e_baseline_mean_hz={l4['baseline_mean']:.6f} "
            f"l4e_post_peak_hz={l4['post_peak']:.6f} "
            f"l4e_peak_delta_hz={l4_peak_delta:.6f} "
            f"l4e_onset_latency_ms={l4['onset_latency_ms']:.6f} "
            f"l4e_peak_latency_ms={l4['peak_latency_ms']:.6f}"
        ),
    )

    target_parts = []
    latency_ok = True
    l4_latency_reference = causal_metrics["l4e"] if causal_source == "event_minus_gray_control" else l4
    for population in ("l23e", "l23pv", "l23som"):
        pop_metrics = causal_metrics[population]
        raw_metrics = metrics[population]
        peak_delta = pop_metrics["post_peak"] - pop_metrics["baseline_mean"]
        pop_ok = (
            math.isfinite(pop_metrics["onset_latency_ms"])
            and peak_delta > 0.0
            and math.isfinite(l4_latency_reference["onset_latency_ms"])
            and pop_metrics["onset_latency_ms"] >= max(-bin_ms, l4_latency_reference["onset_latency_ms"] - (2.0 * bin_ms))
            and pop_metrics["onset_latency_ms"] <= post_ms
        )
        latency_ok &= pop_ok
        target_parts.append(
            f"{population}_causal_baseline={pop_metrics['baseline_mean']:.6f} "
            f"{population}_causal_peak={pop_metrics['post_peak']:.6f} "
            f"{population}_causal_onset_ms={pop_metrics['onset_latency_ms']:.6f} "
            f"{population}_causal_peak_ms={pop_metrics['peak_latency_ms']:.6f} "
            f"{population}_raw_baseline={raw_metrics['baseline_mean']:.6f} "
            f"{population}_raw_peak={raw_metrics['post_peak']:.6f} "
            f"{population}_raw_onset_ms={raw_metrics['onset_latency_ms']:.6f}"
        )
    overall_ok &= print_result(
        latency_ok,
        "natural_video_event_l23_interneuron_latency_peak",
        (
            f"timing_source={causal_source} "
            f"l4_reference_onset_ms={l4_latency_reference['onset_latency_ms']:.6f} "
            + " ".join(target_parts)
        ),
    )

    rate_safety_ok = (
        metrics["l23e"]["post_peak"] <= 500.0
        and metrics["l23pv"]["post_peak"] <= 1000.0
        and metrics["l23som"]["post_peak"] <= 1000.0
    )
    overall_ok &= print_result(
        rate_safety_ok,
        "natural_video_event_rate_safety",
        (
            f"l23e_peak_hz={metrics['l23e']['post_peak']:.6f} "
            f"l23pv_peak_hz={metrics['l23pv']['post_peak']:.6f} "
            f"l23som_peak_hz={metrics['l23som']['post_peak']:.6f}"
        ),
    )

    max_lag_ms = min(50.0, post_ms)
    cross = {
        population: event_best_lag_correlation(series["l4e"], series[population], max_lag_ms)
        for population in ("l23e", "l23pv", "l23som")
    }
    l23e_corr = cross["l23e"]["best_corr"]
    l23e_null = cross["l23e"]["shifted_null_corr"]
    cross_ok = (
        l23e_corr is not None
        and math.isfinite(l23e_corr)
        and l23e_corr > 0.0
        and (
            l23e_null is None
            or not math.isfinite(l23e_null)
            or l23e_corr >= (l23e_null - 0.02)
        )
    )
    overall_ok &= print_result(
        cross_ok,
        "natural_video_event_crosscorr_null",
        (
            f"max_lag_ms={max_lag_ms:.6f} "
            f"l23e_best_lag_ms={format_optional_float(cross['l23e']['best_lag_ms'])} "
            f"l23e_best_corr={format_optional_float(cross['l23e']['best_corr'])} "
            f"l23e_lag0_corr={format_optional_float(cross['l23e']['lag0_corr'])} "
            f"l23e_shifted_null_corr={format_optional_float(cross['l23e']['shifted_null_corr'])} "
            f"l23pv_best_lag_ms={format_optional_float(cross['l23pv']['best_lag_ms'])} "
            f"l23pv_best_corr={format_optional_float(cross['l23pv']['best_corr'])} "
            f"l23som_best_lag_ms={format_optional_float(cross['l23som']['best_lag_ms'])} "
            f"l23som_best_corr={format_optional_float(cross['l23som']['best_corr'])}"
        ),
    )

    if controls_available:
        gray_l4 = event_response_metrics(mean_event_bin_series(population_rows, "gray_control", "l4e"))
        gray_l23e = event_response_metrics(mean_event_bin_series(population_rows, "gray_control", "l23e"))
        blank_l4 = event_response_metrics(mean_event_bin_series(population_rows, "blank_control", "l4e"))
        blank_l23e = event_response_metrics(mean_event_bin_series(population_rows, "blank_control", "l23e"))
        control_ok = (
            (not blank_required or blank_l4["post_peak"] <= (1.10 * max(l4["post_peak"], 1.0e-9)))
            and (not blank_required or blank_l23e["post_peak"] <= (1.10 * max(metrics["l23e"]["post_peak"], 1.0e-9)))
            and (not gray_required or math.isfinite(gray_l4["post_peak"]))
            and (not gray_required or math.isfinite(gray_l23e["post_peak"]))
        )
        overall_ok &= print_result(
            control_ok,
            "natural_video_event_gray_blank_controls",
            (
                f"event_l4e_peak_hz={l4['post_peak']:.6f} "
                f"gray_l4e_peak_hz={gray_l4['post_peak']:.6f} "
                f"blank_l4e_peak_hz={blank_l4['post_peak']:.6f} "
                f"event_l23e_peak_hz={metrics['l23e']['post_peak']:.6f} "
                f"gray_l23e_peak_hz={gray_l23e['post_peak']:.6f} "
                f"blank_l23e_peak_hz={blank_l23e['post_peak']:.6f}"
            ),
        )
    return overall_ok


def validate_natural_video_physiology(run: RunData) -> bool:
    overall_ok = True
    video_enabled = run.summary.get("video_replay_enabled", 0.0)
    expected_frame_count = int(round(run.summary.get("video_frame_count", 0.0)))
    population_rows = run.video_population_rows
    site_rows = run.video_site_rows
    frame_rows = run.video_frame_summary_rows
    unique_frames = (
        len({row.frame_index for row in frame_rows})
        if frame_rows is not None
        else 0
    )
    artifacts_available = (
        video_enabled == 1.0
        and population_rows is not None
        and site_rows is not None
        and frame_rows is not None
        and unique_frames > 0
        and (expected_frame_count == 0 or expected_frame_count == unique_frames)
    )
    overall_ok &= print_result(
        artifacts_available,
        "natural_video_artifacts_available",
        (
            f"video_replay_enabled={video_enabled:.6f} "
            f"summary_frame_count={expected_frame_count} "
            f"population_rows={len(population_rows) if population_rows is not None else 0} "
            f"site_rows={len(site_rows) if site_rows is not None else 0} "
            f"frame_summary_rows={len(frame_rows) if frame_rows is not None else 0} "
            f"unique_frames={unique_frames}"
        ),
    )
    if not artifacts_available or population_rows is None or site_rows is None or frame_rows is None:
        return overall_ok

    feedback_disabled = run.summary.get("video_feedback_disabled", 0.0)
    video_training_enabled = run.summary.get("video_training_enabled", 0.0)
    consolidation_enabled = run.summary.get("lower_v1_video_consolidation_enabled", 0.0)
    overall_ok &= print_result(
        feedback_disabled == 1.0 and (video_training_enabled == 0.0 or consolidation_enabled == 1.0),
        "natural_video_feedback_disabled",
        (
            f"video_feedback_disabled={feedback_disabled:.6f} "
            f"video_training_enabled={video_training_enabled:.6f} "
            f"lower_v1_video_consolidation_enabled={consolidation_enabled:.6f}"
        ),
    )

    population_rates = video_population_rates_by_name(population_rows)
    site_rates = video_site_rates_by_name(site_rows)
    required_populations = {"l4e", "l23e", "l23pv", "l23som"}
    missing_populations = sorted(
        population for population in required_populations
        if population not in population_rates or population not in site_rates
    )
    population_available = not missing_populations
    overall_ok &= print_result(
        population_available,
        "natural_video_population_artifacts_available",
        f"missing_populations={missing_populations if missing_populations else 'none'}",
    )
    if not population_available:
        return overall_ok

    l4e_pop = population_rates["l4e"]
    l23e_pop = population_rates["l23e"]
    l23pv_pop = population_rates["l23pv"]
    l23som_pop = population_rates["l23som"]
    l4e_site = site_rates["l4e"]
    l23e_site = site_rates["l23e"]
    l23pv_site = site_rates["l23pv"]
    l23som_site = site_rates["l23som"]

    l4e_mean = mean(l4e_pop)
    l4e_site_p99 = percentile(l4e_site, 99.0)
    l4e_site_p95 = percentile(l4e_site, 95.0)
    overall_ok &= print_result(
        l4e_mean > 0.0 and l4e_site_p99 > 0.0 and l4e_site_p99 <= 500.0,
        "natural_video_l4_responsive_bounded",
        (
            f"l4e_mean_rate_hz={l4e_mean:.6f} "
            f"l4e_site_p95_hz={l4e_site_p95:.6f} "
            f"l4e_site_p99_hz={l4e_site_p99:.6f}"
        ),
    )

    l23e_mean = mean(l23e_pop)
    l23e_site_p95 = percentile(l23e_site, 95.0)
    l23e_site_p99 = percentile(l23e_site, 99.0)
    l23e_frac_lt1 = fraction_less_than(l23e_site, 1.0)
    overall_ok &= print_result(
        max(l23e_pop) > 0.0
        and l23e_site_p99 <= 100.0
        and l23e_site_p95 <= 50.0
        and l23e_frac_lt1 >= 0.25,
        "natural_video_l23e_sparse_safe",
        (
            f"l23e_mean_rate_hz={l23e_mean:.6f} "
            f"l23e_site_p95_hz={l23e_site_p95:.6f} "
            f"l23e_site_p99_hz={l23e_site_p99:.6f} "
            f"l23e_site_frac_lt1={l23e_frac_lt1:.6f}"
        ),
    )

    l23pv_mean = mean(l23pv_pop)
    l23som_mean = mean(l23som_pop)
    l23pv_site_p99 = percentile(l23pv_site, 99.0)
    l23som_site_p99 = percentile(l23som_site, 99.0)
    overall_ok &= print_result(
        max(l23pv_pop) > 0.0
        and max(l23som_pop) > 0.0
        and l23pv_site_p99 <= 200.0
        and l23som_site_p99 <= 200.0,
        "natural_video_interneurons_active_safe",
        (
            f"l23pv_mean_rate_hz={l23pv_mean:.6f} "
            f"l23som_mean_rate_hz={l23som_mean:.6f} "
            f"l23pv_site_p99_hz={l23pv_site_p99:.6f} "
            f"l23som_site_p99_hz={l23som_site_p99:.6f}"
        ),
    )

    l4e_frame_rates = [row.l4e_rate_hz for row in frame_rows]
    l23e_frame_rates = [row.l23e_rate_hz for row in frame_rows]
    l23pv_frame_rates = [row.l23pv_rate_hz for row in frame_rows]
    l23som_frame_rates = [row.l23som_rate_hz for row in frame_rows]
    drive_std_values = [row.l4e_drive_std for row in frame_rows]
    l4e_std = standard_deviation(l4e_frame_rates)
    l23e_std = standard_deviation(l23e_frame_rates)
    population_dynamic = (
        unique_frames >= 2
        and l4e_std > 1.0e-6
        and l23e_std > 1.0e-6
        and max(drive_std_values) > 1.0e-6
        and percentile(l4e_frame_rates, 99.0) <= 500.0
        and percentile(l23e_frame_rates, 99.0) <= 100.0
    )
    overall_ok &= print_result(
        population_dynamic,
        "natural_video_population_dynamic_range",
        (
            f"frame_count={unique_frames} "
            f"l4e_frame_std={l4e_std:.6f} "
            f"l23e_frame_std={l23e_std:.6f} "
            f"l23pv_frame_std={standard_deviation(l23pv_frame_rates):.6f} "
            f"l23som_frame_std={standard_deviation(l23som_frame_rates):.6f} "
            f"drive_std_max={max(drive_std_values):.6f}"
        ),
    )

    delay_metrics = compute_video_delay_metrics(frame_rows)
    delay_available = (
        delay_metrics.get("frame_count", 0.0) is not None
        and float(delay_metrics.get("frame_count", 0.0) or 0.0) >= 3.0
        and delay_metrics.get("l23e_best_corr") is not None
    )
    overall_ok &= print_result(
        delay_available,
        "natural_video_delay_crosscorr",
        (
            f"frame_count={delay_metrics.get('frame_count', 0.0):.0f} "
            f"max_lag_frames={delay_metrics.get('max_lag', 0.0):.0f} "
            f"l23e_best_lag_frames={format_optional_float(delay_metrics.get('l23e_best_lag_frames'))} "
            f"l23e_best_corr={format_optional_float(delay_metrics.get('l23e_best_corr'))} "
            f"l23e_lag0_corr={format_optional_float(delay_metrics.get('l23e_lag0_corr'))} "
            f"l23e_lag1_corr={format_optional_float(delay_metrics.get('l23e_lag1_corr'))} "
            f"l23pv_best_lag_frames={format_optional_float(delay_metrics.get('l23pv_best_lag_frames'))} "
            f"l23pv_best_corr={format_optional_float(delay_metrics.get('l23pv_best_corr'))} "
            f"l23som_best_lag_frames={format_optional_float(delay_metrics.get('l23som_best_lag_frames'))} "
            f"l23som_best_corr={format_optional_float(delay_metrics.get('l23som_best_corr'))}"
        ),
    )

    observed_repeats = sorted({row.repeat_index for row in frame_rows})
    summary_repeat_count = int(round(run.summary.get("video_repeat_count", float(len(observed_repeats)))))
    if len(observed_repeats) < 2:
        print(
            "INFO natural_video_replay_reliability "
            f"available=0 observed_repeat_count={len(observed_repeats)} "
            f"summary_repeat_count={summary_repeat_count} "
            "set_env=V1_VIDEO_REPLAY_REPEAT_COUNT>=2"
        )
    else:
        l4e_reliability = pairwise_repeat_reliability(frame_rows, "l4e_rate_hz")
        l23e_reliability = pairwise_repeat_reliability(frame_rows, "l23e_rate_hz")
        l23pv_reliability = pairwise_repeat_reliability(frame_rows, "l23pv_rate_hz")
        l23som_reliability = pairwise_repeat_reliability(frame_rows, "l23som_rate_hz")
        reliability_available = l4e_reliability is not None and l23e_reliability is not None
        overall_ok &= print_result(
            reliability_available,
            "natural_video_replay_reliability",
            (
                f"observed_repeat_count={len(observed_repeats)} "
                f"summary_repeat_count={summary_repeat_count} "
                f"l4e_repeat_corr={format_optional_float(l4e_reliability)} "
                f"l23e_repeat_corr={format_optional_float(l23e_reliability)} "
                f"l23pv_repeat_corr={format_optional_float(l23pv_reliability)} "
                f"l23som_repeat_corr={format_optional_float(l23som_reliability)}"
            ),
        )

    recurrent_video_metrics = compute_recurrent_video_metrics(site_rows, run.specificity_rows)
    recurrent_video_available = (
        recurrent_video_metrics["edge_count"] >= 10.0
        and math.isfinite(recurrent_video_metrics["mean_corr"])
    )
    overall_ok &= print_result(
        recurrent_video_available,
        "natural_video_l23ee_recurrent_video_metrics",
        (
            f"edge_count={recurrent_video_metrics['edge_count']:.0f} "
            f"mean_site_response_corr={recurrent_video_metrics['mean_corr']:.6f} "
            f"median_site_response_corr={recurrent_video_metrics['median_corr']:.6f} "
            f"top10_weight_mean_site_response_corr={recurrent_video_metrics['top10_weight_mean_corr']:.6f} "
            f"low_delta_mean_site_response_corr={recurrent_video_metrics['low_delta_mean_corr']:.6f} "
            f"same_site_fraction={recurrent_video_metrics['same_site_fraction']:.6f}"
        ),
    )
    return overall_ok


def main() -> int:
    args = parse_args()
    try:
        if args.min_validation_sites < 1:
            raise ValidationError("--min-validation-sites must be at least 1.")
        if args.responsive_rate_threshold_hz < 0.0:
            raise ValidationError("--responsive-rate-threshold-hz must be non-negative.")
        if args.cell_responsive_threshold_hz < 0.0:
            raise ValidationError("--cell-responsive-threshold-hz must be non-negative.")
        if args.l23_video_min_frame_top1_accuracy is not None:
            if not 0.0 <= args.l23_video_min_frame_top1_accuracy <= 1.0:
                raise ValidationError("--l23-video-min-frame-top1-accuracy must be in [0, 1].")
            if not (args.require_l23_video_reliability or args.require_l23_activity_reliability):
                raise ValidationError(
                    "--l23-video-min-frame-top1-accuracy requires --require-l23-video-reliability "
                    "or --require-l23-activity-reliability."
                )
        for flag_name, value in (
            ("--l23-video-min-raw-oracle-at-k", args.l23_video_min_raw_oracle_at_k),
            (
                "--l23-video-min-raw-oracle-ceiling-fraction",
                args.l23_video_min_raw_oracle_ceiling_fraction,
            ),
            (
                "--l23-video-max-mean-active-tile-fraction",
                args.l23_video_max_mean_active_tile_fraction,
            ),
            (
                "--l23-video-max-sample-active-tile-fraction",
                args.l23_video_max_sample_active_tile_fraction,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{flag_name} must be in [0, 1].")
        if not -1.0 <= args.l23_video_min_l23e_repeat_corr <= 1.0:
            raise ValidationError("--l23-video-min-l23e-repeat-corr must be in [-1, 1].")
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
        video_recoff = (
            try_load_optional_run(args.genn_dir, args.recoff, "recoff")
            if args.require_l23_video_reliability
            else None
        )
        video_pvoff = (
            try_load_optional_run(args.genn_dir, args.pvoff, "pvoff")
            if args.require_l23_video_reliability
            else None
        )

        overall_ok = True
        final_post_video_missing = final_post_video_orientation_missing(full)
        use_final_post_video_orientation = (
            args.require_event_driven_ff_plasticity
            and not final_post_video_missing
        )
        final_post_video_som_artifact_missing = final_post_video_som_missing(full)
        use_final_post_video_som = (
            args.require_event_driven_ff_plasticity
            and not final_post_video_som_artifact_missing
        )
        som_validation_full = (
            with_final_post_video_som_artifacts(full)
            if use_final_post_video_som
            else full
        )
        som_validation_source = (
            "final_post_video" if use_final_post_video_som else "post"
        )
        if args.require_event_driven_ff_plasticity:
            overall_ok &= print_result(
                not final_post_video_missing,
                "final_post_video_orientation_artifacts_available",
                (
                    f"source={'final_post_video' if not final_post_video_missing else 'post'} "
                    f"summary_enabled={full.summary.get('final_post_video_assay_enabled', math.nan):.6f} "
                    f"missing={','.join(final_post_video_missing) if final_post_video_missing else 'none'}"
                ),
            )
            print_final_post_video_reference_info(full)
            overall_ok &= print_result(
                not final_post_video_som_artifact_missing,
                "final_post_video_som_artifacts_available",
                (
                    f"source={som_validation_source} "
                    f"missing={','.join(final_post_video_som_artifact_missing) if final_post_video_som_artifact_missing else 'none'}"
                ),
            )
        validation_l23e_post_sites = (
            full.final_post_video_l23e_sites
            if use_final_post_video_orientation and full.final_post_video_l23e_sites is not None
            else full.l23e_post_sites
        )
        validation_l4_post_sites = (
            full.final_post_video_l4_sites
            if use_final_post_video_orientation and full.final_post_video_l4_sites is not None
            else full.l4_post_sites
        )
        validation_post_site_rates = (
            full.final_post_video_site_rates
            if use_final_post_video_orientation and full.final_post_video_site_rates is not None
            else full.post_site_rates
        )
        validation_l23e_cell_tuning = (
            full.final_post_video_l23e_cell_tuning
            if use_final_post_video_orientation
            else full.l23e_cell_tuning
        )
        validation_l23e_cell_tuning_multiphase = (
            full.final_post_video_l23e_cell_tuning_multiphase
            if use_final_post_video_orientation
            else full.l23e_cell_tuning_multiphase
        )
        orientation_validation_source = (
            "final_post_video" if use_final_post_video_orientation else "post"
        )
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

        if args.require_natural_video_physiology:
            overall_ok &= validate_natural_video_physiology(full)

        if args.require_l23_video_reliability:
            overall_ok &= validate_l23_video_reliability(
                full,
                control,
                somoff,
                video_recoff,
                video_pvoff,
                min_frame_top1_accuracy=args.l23_video_min_frame_top1_accuracy,
            )

        if args.require_l23_activity_reliability:
            overall_ok &= validate_l23_activity_reliability(
                full,
                min_frame_top1_accuracy=args.l23_video_min_frame_top1_accuracy,
                min_raw_oracle_at_k=args.l23_video_min_raw_oracle_at_k,
                min_raw_oracle_ceiling_fraction=args.l23_video_min_raw_oracle_ceiling_fraction,
                min_l23e_repeat_corr=args.l23_video_min_l23e_repeat_corr,
                max_mean_active_tile_fraction=args.l23_video_max_mean_active_tile_fraction,
                max_sample_active_tile_fraction=args.l23_video_max_sample_active_tile_fraction,
            )

        if args.require_emergent_ff_gain:
            overall_ok &= validate_emergent_ff_gain(full)

        if args.require_event_driven_ff_plasticity:
            overall_ok &= validate_event_driven_ff_plasticity(full)

        if args.require_natural_video_event_timing:
            overall_ok &= validate_natural_video_event_timing(full)

        if args.require_hva_predictor or args.require_hva_population_prediction:
            overall_ok &= validate_hva_predictor(
                full,
                require_population_prediction=args.require_hva_population_prediction,
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
            if validation_l23e_cell_tuning is None:
                missing_artifacts.append(
                    f"{full.prefix}_{orientation_validation_source}_l23e_cell_tuning.csv"
                )
            if validation_l23e_cell_tuning_multiphase is None:
                missing_artifacts.append(
                    f"{full.prefix}_{orientation_validation_source}_l23e_cell_tuning_multiphase.csv"
                )
            artifacts_available = not missing_artifacts
            overall_ok &= print_result(
                artifacts_available,
                "responsiveness_artifacts_available",
                (
                    f"source={orientation_validation_source} "
                    f"full_cell_tuning={int(validation_l23e_cell_tuning is not None)} "
                    f"full_multiphase_tuning={int(validation_l23e_cell_tuning_multiphase is not None)} "
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
                assert validation_l23e_cell_tuning is not None
                assert validation_l23e_cell_tuning_multiphase is not None
                single_5hz = compute_cell_responsive_metrics(validation_l23e_cell_tuning, 5.0)
                single_10hz = compute_cell_responsive_metrics(validation_l23e_cell_tuning, 10.0)
                multiphase_5hz = compute_multiphase_cell_responsive_metrics(
                    validation_l23e_cell_tuning_multiphase,
                    5.0,
                )
                multiphase_10hz = compute_multiphase_cell_responsive_metrics(
                    validation_l23e_cell_tuning_multiphase,
                    10.0,
                )
                phase_mean_peak_values = [
                    max(row.phase_mean_rates_by_deg.values())
                    for row in validation_l23e_cell_tuning_multiphase.values()
                ]
                print(
                    "INFO phase_mean_responsiveness "
                    f"source={orientation_validation_source} "
                    f"cell_count={len(phase_mean_peak_values)} "
                    f"peak_ge5_fraction={fraction_at_least(phase_mean_peak_values, 5.0):.6f} "
                    f"peak_ge10_fraction={fraction_at_least(phase_mean_peak_values, 10.0):.6f} "
                    f"median_phase_mean_peak_hz={median(phase_mean_peak_values):.6f}"
                )
                print(
                    "INFO l23e_cell_peak10_responsiveness "
                    f"source={orientation_validation_source} "
                    f"peak_ge10_fraction={single_10hz.responsive_fraction:.6f} "
                    f"peak_ge10_cells={single_10hz.responsive_cells} "
                    f"total_cells={single_10hz.total_cells}"
                )

                overall_ok &= print_result(
                    0.10 <= single_5hz.responsive_fraction <= 0.45,
                    "l23e_cell_sparse_responsiveness",
                    (
                        f"source={orientation_validation_source} "
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
                        f"source={orientation_validation_source} "
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
                        f"source={orientation_validation_source} "
                        f"multiphase_sites_ge1_fraction="
                        f"{multiphase_5hz.responsive_site_fraction_ge1:.6f} "
                        f"multiphase_sites_ge2_fraction="
                        f"{multiphase_5hz.responsive_site_fraction_ge2:.6f} "
                        f"single_phase_site_fraction={single_5hz.responsive_site_fraction:.6f} "
                        f"total_multiphase_sites={multiphase_5hz.total_sites} "
                        f"total_single_phase_sites={single_5hz.total_sites}"
                    ),
                )

                full_l23e_rate_metrics = compute_rate_metrics(validation_post_site_rates["l23e"])
                overall_ok &= print_result(
                    full_l23e_rate_metrics.frac_below_1hz >= 0.85
                    and full_l23e_rate_metrics.p99_hz <= 5.0,
                    "l23e_population_sparse_rates",
                    (
                        f"source={orientation_validation_source} "
                        f"frac_lt1={full_l23e_rate_metrics.frac_below_1hz:.6f} "
                        f"frac_lt1_min=0.850000 "
                        f"p99={full_l23e_rate_metrics.p99_hz:.6f} "
                        f"p99_limit=5.000000"
                    ),
                )

                spatial_balance = responsiveness_spatial_balance_metrics(
                    validation_l23e_cell_tuning_multiphase,
                    validation_l23e_post_sites,
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
                        f"source={orientation_validation_source} "
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
            if validation_l4_post_sites is None:
                missing_scaling_artifacts.append(f"{full.prefix}_{orientation_validation_source}_l4_sites.csv")
            if control.l4_post_sites is None:
                missing_scaling_artifacts.append(f"{control.prefix}_post_l4_sites.csv")
            if not post_site_preferences_available(validation_l4_post_sites):
                missing_scaling_artifacts.append(
                    f"{full.prefix}_{orientation_validation_source}_l4_sites.csv:map_pref/measured_pref"
                )
            if not post_site_preferences_available(control.l4_post_sites):
                missing_scaling_artifacts.append(f"{control.prefix}_post_l4_sites.csv:map_pref/measured_pref")
            if not post_site_preferences_available(validation_l23e_post_sites):
                missing_scaling_artifacts.append(
                    f"{full.prefix}_{orientation_validation_source}_l23_sites.csv:map_pref/measured_pref"
                )
            if not post_site_preferences_available(control.l23e_post_sites):
                missing_scaling_artifacts.append(f"{control.prefix}_post_l23_sites.csv:map_pref/measured_pref")
            if validation_l23e_cell_tuning_multiphase is None:
                missing_scaling_artifacts.append(
                    f"{full.prefix}_{orientation_validation_source}_l23e_cell_tuning_multiphase.csv"
                )
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
                assert validation_l4_post_sites is not None
                assert control.l4_post_sites is not None
                assert validation_l23e_cell_tuning_multiphase is not None
                full_l4_intersite_for_scaling = load_l4_intersite_metrics(args.genn_dir, args.full)
                control_l4_intersite_for_scaling = load_l4_intersite_metrics(args.genn_dir, args.control)
                somoff_l4_intersite_for_scaling = load_l4_intersite_metrics(args.genn_dir, args.somoff)

                l4_map = compute_l4_map_consistency_metrics(validation_l4_post_sites)
                l4is_post_map_error = full_l4_intersite_for_scaling.get("post_l4_map_error_deg_median", math.nan)
                l4is_baseline_map_error = full_l4_intersite_for_scaling.get(
                    "baseline_l4_map_error_deg_median",
                    math.nan,
                )
                summary_l4_map_metric = (
                    "final_post_video_l4_map_error_deg_median"
                    if use_final_post_video_orientation
                    else "post_l4_map_error_deg_median"
                )
                overall_ok &= print_result(
                    l4_map["active_fraction"] >= 0.95
                    and l4_map["median_error_deg"] <= 5.0
                    and l4_map["p90_error_deg"] <= 10.0,
                    "scaling_l4_map_consistency",
                    (
                        f"source={orientation_validation_source} "
                        f"active_sites={int(l4_map['active_sites'])} "
                        f"total_sites={int(l4_map['total_sites'])} "
                        f"active_fraction={l4_map['active_fraction']:.6f} "
                        f"median_map_error_deg={l4_map['median_error_deg']:.6f} "
                        f"p90_map_error_deg={l4_map['p90_error_deg']:.6f} "
                        f"summary_post_l4_map_error_deg="
                        f"{require_summary_metric(full, summary_l4_map_metric):.6f} "
                        f"l4is_post_l4_map_error_deg={l4is_post_map_error:.6f} "
                        f"l4is_baseline_l4_map_error_deg={l4is_baseline_map_error:.6f}"
                    ),
                )

                l23_l4_map = compute_l23_l4_map_consistency_metrics(
                    validation_l23e_post_sites,
                    validation_l4_post_sites,
                    validation_l23e_cell_tuning_multiphase,
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
                        f"source={orientation_validation_source} "
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
                    validation_l23e_cell_tuning_multiphase,
                    validation_l23e_post_sites,
                    5.0,
                )
                tile10 = compute_tile_orientation_metrics(
                    validation_l23e_cell_tuning_multiphase,
                    validation_l23e_post_sites,
                    10.0,
                )
                overall_ok &= print_result(
                    tile5["nonempty_tile_count"] == 16.0
                    and tile5["global_occupied_bins"] == 12.0
                    and tile5["bin_gate_pass"] == 1.0,
                    "scaling_tile_orientation_coverage",
                    (
                        f"source={orientation_validation_source} "
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
                        f"source={orientation_validation_source} "
                        f"threshold5_median_entropy={tile5['median_entropy']:.6f} "
                        f"threshold5_min_entropy={tile5['min_entropy']:.6f} "
                        f"threshold10_responsive_cells={int(tile10['responsive_cells'])} "
                        f"threshold10_median_entropy={tile10['median_entropy']:.6f} "
                        f"threshold10_min_entropy={tile10['min_entropy']:.6f}"
                    ),
                )

                edge_quadrants = compute_edge_quadrant_balance_metrics(
                    validation_l23e_cell_tuning_multiphase,
                    validation_l23e_post_sites,
                    5.0,
                )
                overall_ok &= print_result(
                    edge_quadrants["edge_site_coverage"] >= 0.60
                    and edge_quadrants["zero_quadrants"] == 0.0
                    and edge_quadrants["min_quadrant_cell_fraction"] >= 0.10,
                    "scaling_edge_quadrant_balance",
                    (
                        f"source={orientation_validation_source} "
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

        full_post_osi_metric = (
            "final_post_video_l23_median_osi"
            if use_final_post_video_orientation
            else "post_l23_median_osi"
        )
        full_post_osi = require_summary_metric(full, full_post_osi_metric)
        pre_video_full_post_osi = require_summary_metric(full, "post_l23_median_osi")
        control_post_osi = require_summary_metric(control, "post_l23_median_osi")
        osi_delta = full_post_osi - control_post_osi
        strict_osi_ok = full_post_osi >= 0.70 and osi_delta >= 0.10
        printed_strict_osi_ok = print_result(
            strict_osi_ok,
            "osi",
            (
                f"source={orientation_validation_source} "
                f"full_post={full_post_osi:.6f} "
                f"control_post={control_post_osi:.6f} "
                f"delta={osi_delta:.6f} "
                f"pre_video_full_post={pre_video_full_post_osi:.6f}"
            ),
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

        preferred_by_site, preferred_rates_by_site = preferred_center_orientations(som_validation_full)
        primary_validation_site_id = next(iter(preferred_by_site))
        pref_deg = preferred_by_site[primary_validation_site_id]
        full_center_pref_rate = mean(list(preferred_rates_by_site.values()))
        full_min_center_pref_rate = min(preferred_rates_by_site.values())
        full_context = compute_context_metrics(som_validation_full, preferred_by_site)
        somoff_context = compute_context_metrics(somoff, preferred_by_site)
        if use_final_post_video_som:
            pre_video_preferred_by_site, pre_video_preferred_rates_by_site = preferred_center_orientations(full)
            pre_video_context = compute_context_metrics(full, pre_video_preferred_by_site)
            print(
                "INFO final_post_video_som_reference "
                f"pre_mean_center_pref_l23e_hz={mean(list(pre_video_preferred_rates_by_site.values())):.6f} "
                f"final_mean_center_pref_l23e_hz={full_center_pref_rate:.6f} "
                f"pre_mean_bsi={pre_video_context['mean_bsi']:.6f} "
                f"final_mean_bsi={full_context['mean_bsi']:.6f} "
                f"pre_min_center_som_hz={pre_video_context['min_center_som_hz']:.6f} "
                f"final_min_center_som_hz={full_context['min_center_som_hz']:.6f} "
                f"pre_min_broad_som_hz={pre_video_context['min_broad_som_hz']:.6f} "
                f"final_min_broad_som_hz={full_context['min_broad_som_hz']:.6f}"
            )

        overall_ok &= print_result(
            full_center_pref_rate >= 5.0,
            "som_center_pref",
            (
                f"source={som_validation_source} "
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
                f"source={som_validation_source} "
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
                f"source={som_validation_source} "
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
                f"source={som_validation_source} "
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
            f"source={som_validation_source} "
            f"preferred_deg={pref_deg:.1f} "
            f"primary_site={primary_validation_site_id} "
            f"full_preferred_bsi={full_context['preferred_bsi']:.6f} "
            f"somoff_preferred_bsi={somoff_context['preferred_bsi']:.6f}"
        )
        print(
            "INFO som_driven_threshold "
            f"source={som_validation_source} "
            f"full_threshold_hz={full_context['driven_center_threshold_hz']:.6f} "
            f"somoff_threshold_hz={somoff_context['driven_center_threshold_hz']:.6f}"
        )

        full_size = compute_size_tuning_metrics(som_validation_full)
        somoff_size = compute_size_tuning_metrics(
            somoff,
            selected_orientations=full_size["selected_orientations_by_site"],
        )
        if use_final_post_video_som:
            pre_video_size = compute_size_tuning_metrics(full)
            print(
                "INFO final_post_video_size_reference "
                f"pre_peak_rate={pre_video_size['l23e']['peak_rate']:.6f} "
                f"final_peak_rate={full_size['l23e']['peak_rate']:.6f} "
                f"pre_suppression={pre_video_size['l23e']['suppression']:.6f} "
                f"final_suppression={full_size['l23e']['suppression']:.6f} "
                f"pre_l23som_peak_rate={pre_video_size['l23som']['peak_rate']:.6f} "
                f"final_l23som_peak_rate={full_size['l23som']['peak_rate']:.6f}"
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
                som_validation_full,
                somoff,
                full_size,
                somoff_size,
                full_context,
            )
            if use_final_post_video_som:
                pre_video_size = compute_size_tuning_metrics(full)
                pre_video_somoff_size = compute_size_tuning_metrics(
                    somoff,
                    selected_orientations=pre_video_size["selected_orientations_by_site"],
                )
                pre_video_preferred_by_site, _ = preferred_center_orientations(full)
                pre_video_context = compute_context_metrics(full, pre_video_preferred_by_site)
                pre_video_som_size_surround = compute_som_size_surround_metrics(
                    full,
                    somoff,
                    pre_video_size,
                    pre_video_somoff_size,
                    pre_video_context,
                )
                print(
                    "INFO final_post_video_som_recruitment_reference "
                    f"pre_som_large_or_broad_rate={pre_video_som_size_surround['som_large_or_broad_rate']:.6f} "
                    f"final_som_large_or_broad_rate={som_size_surround['som_large_or_broad_rate']:.6f} "
                    f"pre_som_center_or_peak_rate={pre_video_som_size_surround['som_center_or_peak_rate']:.6f} "
                    f"final_som_center_or_peak_rate={som_size_surround['som_center_or_peak_rate']:.6f} "
                    f"pre_som_recruitment_index={pre_video_som_size_surround['som_recruitment_index']:.6f} "
                    f"final_som_recruitment_index={som_size_surround['som_recruitment_index']:.6f}"
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
                    f"source={som_validation_source} "
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
                    f"source={som_validation_source} "
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
                    f"source={som_validation_source} "
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
                    f"source={som_validation_source} "
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
        print(
            "INFO l23ee_response_corr_specificity_all_rows "
            f"all_row_count={int(corr_specificity['all_row_count'])} "
            f"active_endpoint_count={int(corr_specificity['active_endpoint_count'])} "
            f"nonzero_corr_count={int(corr_specificity['nonzero_corr_count'])} "
            f"active_nonzero_corr_count={int(corr_specificity['active_nonzero_corr_count'])} "
            f"active_nonzero_corr_fraction={corr_specificity['active_nonzero_corr_fraction']:.6f} "
            f"all_low_count={int(corr_specificity['all_low_corr_count'])} "
            f"all_high_count={int(corr_specificity['all_high_corr_count'])} "
            f"all_low_corr_range=[{corr_specificity['all_low_corr_min_response_corr']:.6f},"
            f"{corr_specificity['all_low_corr_max_response_corr']:.6f}] "
            f"all_high_corr_range=[{corr_specificity['all_high_corr_min_response_corr']:.6f},"
            f"{corr_specificity['all_high_corr_max_response_corr']:.6f}] "
            f"all_high_mean_delta_w={corr_specificity['all_high_corr_mean_delta_w']:.6f} "
            f"all_low_mean_delta_w={corr_specificity['all_low_corr_mean_delta_w']:.6f} "
            f"all_high_mean_w_after={corr_specificity['all_high_corr_mean_w_after']:.6f} "
            f"all_low_mean_w_after={corr_specificity['all_low_corr_mean_w_after']:.6f}"
        )
        active_corr_delta_w_mean_margin = (
            corr_specificity["active_high_corr_mean_delta_w"] - corr_specificity["active_low_corr_mean_delta_w"]
        )
        active_corr_delta_w_median_margin = (
            corr_specificity["active_high_corr_median_delta_w"] - corr_specificity["active_low_corr_median_delta_w"]
        )
        active_corr_w_after_mean_margin = (
            corr_specificity["active_high_corr_mean_w_after"] - corr_specificity["active_low_corr_mean_w_after"]
        )
        active_corr_w_after_median_margin = (
            corr_specificity["active_high_corr_median_w_after"] - corr_specificity["active_low_corr_median_w_after"]
        )
        active_corr_margin = max(
            active_corr_delta_w_mean_margin,
            active_corr_delta_w_median_margin,
            active_corr_w_after_mean_margin,
            active_corr_w_after_median_margin,
        )
        active_corr_margin_threshold = max(1.0e-6, 0.01 * corr_specificity["active_p95_abs_delta_w"])
        active_corr_margin_ok = (
            corr_specificity["active_endpoint_count"] >= corr_specificity["min_active_count"]
            and corr_specificity["active_low_corr_count"] >= corr_specificity["active_min_count"]
            and corr_specificity["active_high_corr_count"] >= corr_specificity["active_min_count"]
            and corr_specificity["active_p95_abs_delta_w"] >= 1.0e-6
            and active_corr_margin >= active_corr_margin_threshold
        )
        print(
            "INFO l23ee_response_corr_specificity_active_endpoints "
            f"active_endpoint_count={int(corr_specificity['active_endpoint_count'])} "
            f"active_nonzero_corr_count={int(corr_specificity['active_nonzero_corr_count'])} "
            f"active_nonzero_corr_fraction={corr_specificity['active_nonzero_corr_fraction']:.6f} "
            f"nonzero_corr_fraction_floor={corr_specificity['nonzero_corr_fraction_floor']:.6f} "
            f"nonzero_subset_allowed={corr_specificity['nonzero_subset_allowed']:.0f} "
            f"active_low_count={int(corr_specificity['active_low_corr_count'])} "
            f"active_high_count={int(corr_specificity['active_high_corr_count'])} "
            f"active_min_count={int(corr_specificity['active_min_count'])} "
            f"active_low_corr_range=[{corr_specificity['active_low_corr_min_response_corr']:.6f},"
            f"{corr_specificity['active_low_corr_max_response_corr']:.6f}] "
            f"active_high_corr_range=[{corr_specificity['active_high_corr_min_response_corr']:.6f},"
            f"{corr_specificity['active_high_corr_max_response_corr']:.6f}] "
            f"active_high_mean_delta_w={corr_specificity['active_high_corr_mean_delta_w']:.6f} "
            f"active_low_mean_delta_w={corr_specificity['active_low_corr_mean_delta_w']:.6f} "
            f"active_high_mean_w_after={corr_specificity['active_high_corr_mean_w_after']:.6f} "
            f"active_low_mean_w_after={corr_specificity['active_low_corr_mean_w_after']:.6f} "
            f"active_p95_abs_delta_w={corr_specificity['active_p95_abs_delta_w']:.6f} "
            f"active_margin_threshold={active_corr_margin_threshold:.6f} "
            f"active_best_margin={active_corr_margin:.6f} "
            f"active_margin_ok={1 if active_corr_margin_ok else 0}"
        )
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
            and corr_margin >= corr_margin_threshold
            and active_corr_margin_ok,
            "l23ee_response_corr_specificity",
            (
                f"subset={corr_specificity['selected_label']} "
                f"selected_mode_code={corr_specificity['selected_mode_code']:.0f} "
                f"row_count={int(corr_specificity['row_count'])} "
                f"all_row_count={int(corr_specificity['all_row_count'])} "
                f"active_endpoint_count={int(corr_specificity['active_endpoint_count'])} "
                f"nonzero_corr_count={int(corr_specificity['nonzero_corr_count'])} "
                f"active_nonzero_corr_count={int(corr_specificity['active_nonzero_corr_count'])} "
                f"active_nonzero_corr_fraction={corr_specificity['active_nonzero_corr_fraction']:.6f} "
                f"nonzero_corr_fraction_floor={corr_specificity['nonzero_corr_fraction_floor']:.6f} "
                f"nonzero_subset_allowed={corr_specificity['nonzero_subset_allowed']:.0f} "
                f"low_count={int(corr_specificity['low_corr_count'])} "
                f"high_count={int(corr_specificity['high_corr_count'])} "
                f"min_count={int(corr_specificity['min_count'])} "
                f"min_active_count={int(corr_specificity['min_active_count'])} "
                f"min_nonzero_count={int(corr_specificity['min_nonzero_count'])} "
                f"active_low_count={int(corr_specificity['active_low_corr_count'])} "
                f"active_high_count={int(corr_specificity['active_high_corr_count'])} "
                f"active_min_count={int(corr_specificity['active_min_count'])} "
                f"low_corr_range=[{corr_specificity['low_corr_min_response_corr']:.6f},"
                f"{corr_specificity['low_corr_max_response_corr']:.6f}] "
                f"high_corr_range=[{corr_specificity['high_corr_min_response_corr']:.6f},"
                f"{corr_specificity['high_corr_max_response_corr']:.6f}] "
                f"active_low_corr_range=[{corr_specificity['active_low_corr_min_response_corr']:.6f},"
                f"{corr_specificity['active_low_corr_max_response_corr']:.6f}] "
                f"active_high_corr_range=[{corr_specificity['active_high_corr_min_response_corr']:.6f},"
                f"{corr_specificity['active_high_corr_max_response_corr']:.6f}] "
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
                f"best_margin={corr_margin:.6f} "
                f"active_p95_abs_delta_w={corr_specificity['active_p95_abs_delta_w']:.6f} "
                f"active_margin_threshold={active_corr_margin_threshold:.6f} "
                f"active_best_margin={active_corr_margin:.6f} "
                f"active_margin_ok={1 if active_corr_margin_ok else 0}"
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
