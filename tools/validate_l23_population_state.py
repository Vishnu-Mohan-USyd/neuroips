#!/usr/bin/env python3
"""Validate stable L2/3 population-state coding from repeated natural video.

This validator intentionally treats exact top-k site identity as secondary.
Biological V1/L2/3 responses to naturalistic stimuli can be sparse and
trial-variable at single-cell level while preserving a stable distributed
population code. The primary tests here therefore use held-out-repeat
population-vector reliability, representational geometry, and readout
transfer.

Input CSV schema:
    repeat_index,frame_index,population,site_id,rate_hz

The expected source is ``*_video_site_rates.csv`` exported by the GeNN V1
model. Metrics are computed only from the requested population, default
``l23e``. If a central validation core is requested, site IDs are interpreted as
row-major ``y * sheet_side + x`` and cropped before metric calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadedRates:
    """Dense repeated-video rate tensor.

    Attributes:
        rates: Rate array shaped ``[repeat, frame, site]`` in Hz.
        repeats: Original repeat indices represented in the CSV.
        frames: Original frame indices represented in the CSV.
        sites: Original site IDs represented after optional cropping.
    """

    rates: np.ndarray
    repeats: list[int]
    frames: list[int]
    sites: list[int]


def load_site_rates(
    path: Path,
    population: str,
    *,
    sheet_side: int | None = None,
    core_side: int | None = None,
) -> LoadedRates:
    """Load ``population`` site rates as ``[repeat, frame, site]``.

    Args:
        path: GeNN ``*_video_site_rates.csv`` path.
        population: Population label to extract, e.g. ``"l23e"``.
        sheet_side: Optional full sheet side for central-core cropping.
        core_side: Optional central crop side. Requires ``sheet_side``.

    Raises:
        ValueError: If required columns are missing, the population has no
            rows, or crop parameters are inconsistent.
    """

    if core_side is not None and sheet_side is None:
        raise ValueError("--core-side requires --sheet-side")
    if sheet_side is not None and sheet_side <= 0:
        raise ValueError("--sheet-side must be positive")
    if core_side is not None and (core_side <= 0 or core_side > sheet_side):
        raise ValueError("--core-side must be in (0, sheet_side]")

    keep_sites: set[int] | None = None
    if sheet_side is not None and core_side is not None and core_side < sheet_side:
        offset = (sheet_side - core_side) // 2
        keep_sites = {
            (y * sheet_side) + x
            for y in range(offset, offset + core_side)
            for x in range(offset, offset + core_side)
        }

    rows: list[tuple[int, int, int, float]] = []
    repeat_set: set[int] = set()
    frame_set: set[int] = set()
    site_set: set[int] = set()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"repeat_index", "frame_index", "population", "site_id", "rate_hz"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            if row["population"] != population:
                continue
            site = int(row["site_id"])
            if keep_sites is not None and site not in keep_sites:
                continue
            repeat = int(row["repeat_index"])
            frame = int(row["frame_index"])
            rate = float(row["rate_hz"])
            rows.append((repeat, frame, site, rate))
            repeat_set.add(repeat)
            frame_set.add(frame)
            site_set.add(site)

    if not rows:
        crop = f", sheet_side={sheet_side}, core_side={core_side}" if core_side else ""
        raise ValueError(f"{path} contains no rows for population={population!r}{crop}")

    repeats = sorted(repeat_set)
    frames = sorted(frame_set)
    sites = sorted(site_set)
    repeat_to_i = {value: idx for idx, value in enumerate(repeats)}
    frame_to_i = {value: idx for idx, value in enumerate(frames)}
    site_to_i = {value: idx for idx, value in enumerate(sites)}

    rates = np.zeros((len(repeats), len(frames), len(sites)), dtype=np.float64)
    seen = np.zeros_like(rates, dtype=bool)
    for repeat, frame, site, rate in rows:
        r = repeat_to_i[repeat]
        f = frame_to_i[frame]
        s = site_to_i[site]
        rates[r, f, s] = rate
        seen[r, f, s] = True

    if not np.all(seen):
        missing_count = int(seen.size - int(np.sum(seen)))
        raise ValueError(f"{path} has an incomplete repeat/frame/site grid: missing={missing_count}")

    return LoadedRates(rates=rates, repeats=repeats, frames=frames, sites=sites)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation for two vectors, returning NaN for degenerate input."""

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ax = a - float(np.mean(a))
    bx = b - float(np.mean(b))
    denom = float(np.linalg.norm(ax) * np.linalg.norm(bx))
    if denom <= 1.0e-12:
        return math.nan
    return float(np.dot(ax, bx) / denom)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def pairwise_repeat_reliability(rates: np.ndarray, *, rng: np.random.Generator) -> dict[str, float]:
    """Compute matched-repeat vector correlations and deterministic controls."""

    n_repeats, n_frames, n_sites = rates.shape
    if n_repeats < 2:
        raise ValueError("repeat reliability requires at least two repeats")
    if n_frames < 2:
        raise ValueError("repeat reliability requires at least two frames")

    temporal_offset = max(1, n_frames // 2)
    spatial_perm = rng.permutation(n_sites)
    matched: list[float] = []
    temporal: list[float] = []
    spatial: list[float] = []
    nonmatch: list[float] = []

    for r0 in range(n_repeats):
        for r1 in range(r0 + 1, n_repeats):
            for frame in range(n_frames):
                matched.append(safe_pearson(rates[r0, frame], rates[r1, frame]))
                temporal.append(
                    safe_pearson(rates[r0, frame], rates[r1, (frame + temporal_offset) % n_frames])
                )
                spatial.append(safe_pearson(rates[r0, frame], rates[r1, frame, spatial_perm]))
                nonmatch.append(safe_pearson(rates[r0, frame], rates[r1, n_frames - frame - 1]))

    matched_mean = finite_mean(matched)
    temporal_mean = finite_mean(temporal)
    spatial_mean = finite_mean(spatial)
    nonmatch_mean = finite_mean(nonmatch)
    return {
        "matched_repeat_corr_mean": matched_mean,
        "matched_repeat_corr_median": finite_median(matched),
        "temporal_control_corr_mean": temporal_mean,
        "spatial_control_corr_mean": spatial_mean,
        "reverse_frame_control_corr_mean": nonmatch_mean,
        "matched_minus_temporal_control": matched_mean - temporal_mean,
        "matched_minus_spatial_control": matched_mean - spatial_mean,
        "matched_minus_reverse_control": matched_mean - nonmatch_mean,
        "valid_matched_pair_count": float(sum(math.isfinite(v) for v in matched)),
    }


def corr_rows(matrix: np.ndarray) -> np.ndarray:
    """Correlation matrix between row vectors."""

    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denom = np.maximum(norms[:, None] * norms[None, :], 1.0e-12)
    return (centered @ centered.T) / denom


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    row_idx, col_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_idx, col_idx]


def rsm_stability(rates: np.ndarray, *, rng: np.random.Generator) -> dict[str, float]:
    """Odd/even repeat representational similarity stability and controls."""

    n_repeats, n_frames, n_sites = rates.shape
    if n_repeats < 2:
        raise ValueError("RSM stability requires at least two repeats")
    if n_frames < 3:
        raise ValueError("RSM stability requires at least three frames")

    even_repeat_indices = np.arange(0, n_repeats, 2)
    odd_repeat_indices = np.arange(1, n_repeats, 2)
    if len(odd_repeat_indices) == 0:
        odd_repeat_indices = np.array([n_repeats - 1])
        even_repeat_indices = np.arange(0, n_repeats - 1)

    template_a = np.mean(rates[even_repeat_indices], axis=0)
    template_b = np.mean(rates[odd_repeat_indices], axis=0)
    rsm_a = corr_rows(template_a)
    rsm_b = corr_rows(template_b)

    upper_a = upper_triangle_values(rsm_a)
    upper_b = upper_triangle_values(rsm_b)
    observed = safe_pearson(upper_a, upper_b)

    frame_perm = rng.permutation(n_frames)
    temporal_shuffle = safe_pearson(upper_a, upper_triangle_values(rsm_b[frame_perm][:, frame_perm]))

    scrambled = np.empty_like(template_b)
    for frame in range(n_frames):
        scrambled[frame] = template_b[frame, rng.permutation(n_sites)]
    spatial_shuffle = safe_pearson(upper_a, upper_triangle_values(corr_rows(scrambled)))

    return {
        "rsm_odd_even_corr": observed,
        "rsm_temporal_shuffle_corr": temporal_shuffle,
        "rsm_spatial_shuffle_corr": spatial_shuffle,
        "rsm_minus_temporal_shuffle": observed - temporal_shuffle,
        "rsm_minus_spatial_shuffle": observed - spatial_shuffle,
        "rsm_even_repeat_count": float(len(even_repeat_indices)),
        "rsm_odd_repeat_count": float(len(odd_repeat_indices)),
    }


def correlation_template_decoder(rates: np.ndarray, top_k: int) -> dict[str, float]:
    """Held-out-repeat frame decoder using other repeats as templates."""

    n_repeats, n_frames, _ = rates.shape
    if n_repeats < 2:
        raise ValueError("decoder requires at least two repeats")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    effective_top_k = min(top_k, n_frames)

    top1 = 0
    topk = 0
    ranks: list[int] = []
    same: list[float] = []
    different: list[float] = []

    for heldout in range(n_repeats):
        train = [idx for idx in range(n_repeats) if idx != heldout]
        templates = np.mean(rates[train], axis=0)
        samples = rates[heldout]
        for frame in range(n_frames):
            scores = np.asarray([safe_pearson(samples[frame], templates[target]) for target in range(n_frames)])
            scores = np.nan_to_num(scores, nan=-1.0)
            true_score = float(scores[frame])
            rank = int(1 + np.sum(scores > true_score))
            ranks.append(rank)
            top1 += int(rank == 1)
            topk += int(rank <= effective_top_k)
            same.append(true_score)
            different.extend(float(scores[target]) for target in range(n_frames) if target != frame)

    sample_count = n_repeats * n_frames
    return {
        "decoder_top1_accuracy": float(top1 / sample_count),
        f"decoder_top{effective_top_k}_accuracy": float(topk / sample_count),
        "decoder_topk": float(effective_top_k),
        "decoder_chance_top1": float(1.0 / n_frames),
        "decoder_chance_topk": float(effective_top_k / n_frames),
        "decoder_mean_rank": float(np.mean(np.asarray(ranks, dtype=np.float64))),
        "decoder_median_rank": float(np.median(np.asarray(ranks, dtype=np.float64))),
        "decoder_same_similarity": finite_mean(same),
        "decoder_different_similarity": finite_mean(different),
        "decoder_same_different_gap": finite_mean(same) - finite_mean(different),
        "decoder_sample_count": float(sample_count),
    }


def population_snr_and_sparsity(rates: np.ndarray, active_threshold_hz: float) -> dict[str, float]:
    """Compute repeat noise, stimulus signal, and sparse activity diagnostics."""

    frame_means = np.mean(rates, axis=0)
    signal_var = float(np.mean(np.var(frame_means, axis=0)))
    noise_var = float(np.mean(np.var(rates, axis=0)))
    response_mean = float(np.mean(rates))
    repeat_fano_like = float(noise_var / max(response_mean, 1.0e-12))
    active = rates >= active_threshold_hz
    site_peak = np.max(rates, axis=(0, 1))
    site_mean = np.mean(rates, axis=(0, 1))
    sample_active_fraction = np.mean(active, axis=2)
    return {
        "rate_mean_hz": response_mean,
        "rate_p95_hz": float(np.percentile(rates, 95.0)),
        "rate_p99_hz": float(np.percentile(rates, 99.0)),
        "signal_variance": signal_var,
        "repeat_noise_variance": noise_var,
        "signal_noise_ratio": float(signal_var / max(noise_var, 1.0e-12)),
        "repeat_fano_like": repeat_fano_like,
        "mean_sample_active_site_fraction": float(np.mean(sample_active_fraction)),
        "max_sample_active_site_fraction": float(np.max(sample_active_fraction)),
        "lifetime_active_site_fraction": float(np.mean(site_peak >= active_threshold_hz)),
        "mean_active_site_fraction": float(np.mean(site_mean >= active_threshold_hz)),
    }


def emit_result(ok: bool, name: str, details: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"{status} {name} {details}")
    return ok


def format_metric(metrics: dict[str, float], key: str) -> str:
    value = metrics[key]
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def validate(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    """Print PASS/FAIL rows and return aggregate status."""

    ok = True
    ok &= emit_result(
        metrics["repeat_count"] >= args.min_repeat_count and metrics["frame_count"] >= args.min_frame_count,
        "l23_population_state_artifacts",
        (
            f"population={args.population} repeat_count={metrics['repeat_count']:.0f} "
            f"frame_count={metrics['frame_count']:.0f} site_count={metrics['site_count']:.0f} "
            f"sheet_side={args.sheet_side if args.sheet_side is not None else 'none'} "
            f"core_side={args.core_side if args.core_side is not None else 'none'}"
        ),
    )

    ok &= emit_result(
        metrics["matched_repeat_corr_mean"] >= args.min_matched_corr
        and metrics["matched_minus_temporal_control"] >= args.min_control_gap
        and metrics["matched_minus_spatial_control"] >= args.min_control_gap,
        "l23_population_vector_repeat_reliability",
        (
            f"matched_mean={format_metric(metrics, 'matched_repeat_corr_mean')} "
            f"matched_median={format_metric(metrics, 'matched_repeat_corr_median')} "
            f"temporal_control={format_metric(metrics, 'temporal_control_corr_mean')} "
            f"spatial_control={format_metric(metrics, 'spatial_control_corr_mean')} "
            f"reverse_control={format_metric(metrics, 'reverse_frame_control_corr_mean')} "
            f"gap_temporal={format_metric(metrics, 'matched_minus_temporal_control')} "
            f"gap_spatial={format_metric(metrics, 'matched_minus_spatial_control')} "
            f"min_matched={args.min_matched_corr:.6f} min_gap={args.min_control_gap:.6f}"
        ),
    )

    ok &= emit_result(
        metrics["rsm_odd_even_corr"] >= args.min_rsm_corr
        and metrics["rsm_minus_temporal_shuffle"] >= args.min_rsm_shuffle_gap
        and metrics["rsm_minus_spatial_shuffle"] >= args.min_rsm_shuffle_gap,
        "l23_population_rsm_stability",
        (
            f"rsm_corr={format_metric(metrics, 'rsm_odd_even_corr')} "
            f"temporal_shuffle={format_metric(metrics, 'rsm_temporal_shuffle_corr')} "
            f"spatial_shuffle={format_metric(metrics, 'rsm_spatial_shuffle_corr')} "
            f"gap_temporal={format_metric(metrics, 'rsm_minus_temporal_shuffle')} "
            f"gap_spatial={format_metric(metrics, 'rsm_minus_spatial_shuffle')} "
            f"min_rsm={args.min_rsm_corr:.6f} min_gap={args.min_rsm_shuffle_gap:.6f}"
        ),
    )

    topk_key = f"decoder_top{int(metrics['decoder_topk'])}_accuracy"
    ok &= emit_result(
        metrics["decoder_top1_accuracy"] >= args.min_decoder_top1
        and metrics[topk_key] >= args.min_decoder_topk_accuracy
        and metrics["decoder_same_different_gap"] >= args.min_decoder_similarity_gap,
        "l23_population_heldout_decoder_transfer",
        (
            f"top1={format_metric(metrics, 'decoder_top1_accuracy')} "
            f"{topk_key}={format_metric(metrics, topk_key)} "
            f"chance_top1={format_metric(metrics, 'decoder_chance_top1')} "
            f"chance_topk={format_metric(metrics, 'decoder_chance_topk')} "
            f"mean_rank={format_metric(metrics, 'decoder_mean_rank')} "
            f"same_similarity={format_metric(metrics, 'decoder_same_similarity')} "
            f"different_similarity={format_metric(metrics, 'decoder_different_similarity')} "
            f"gap={format_metric(metrics, 'decoder_same_different_gap')} "
            f"min_top1={args.min_decoder_top1:.6f} "
            f"min_topk={args.min_decoder_topk_accuracy:.6f}"
        ),
    )

    ok &= emit_result(
        metrics["signal_noise_ratio"] >= args.min_signal_noise_ratio,
        "l23_population_signal_noise_structure",
        (
            f"signal_variance={format_metric(metrics, 'signal_variance')} "
            f"repeat_noise_variance={format_metric(metrics, 'repeat_noise_variance')} "
            f"signal_noise_ratio={format_metric(metrics, 'signal_noise_ratio')} "
            f"repeat_fano_like={format_metric(metrics, 'repeat_fano_like')} "
            f"min_signal_noise_ratio={args.min_signal_noise_ratio:.6f}"
        ),
    )

    ok &= emit_result(
        args.min_mean_sample_active_fraction
        <= metrics["mean_sample_active_site_fraction"]
        <= args.max_mean_sample_active_fraction
        and metrics["max_sample_active_site_fraction"] <= args.max_sample_active_fraction
        and metrics["lifetime_active_site_fraction"] >= args.min_lifetime_active_fraction,
        "l23_population_sparse_but_distributed",
        (
            f"rate_mean_hz={format_metric(metrics, 'rate_mean_hz')} "
            f"rate_p95_hz={format_metric(metrics, 'rate_p95_hz')} "
            f"rate_p99_hz={format_metric(metrics, 'rate_p99_hz')} "
            f"mean_sample_active_site_fraction={format_metric(metrics, 'mean_sample_active_site_fraction')} "
            f"max_sample_active_site_fraction={format_metric(metrics, 'max_sample_active_site_fraction')} "
            f"lifetime_active_site_fraction={format_metric(metrics, 'lifetime_active_site_fraction')} "
            f"active_threshold_hz={args.active_threshold_hz:.6f}"
        ),
    )
    return ok


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_site_rates_csv", type=Path)
    parser.add_argument("--population", default="l23e")
    parser.add_argument("--sheet-side", type=int, default=None)
    parser.add_argument("--core-side", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--active-threshold-hz", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json-out", type=Path, default=None)

    parser.add_argument("--min-repeat-count", type=float, default=3.0)
    parser.add_argument("--min-frame-count", type=float, default=32.0)
    parser.add_argument("--min-matched-corr", type=float, default=0.60)
    parser.add_argument("--min-control-gap", type=float, default=0.10)
    parser.add_argument("--min-rsm-corr", type=float, default=0.50)
    parser.add_argument("--min-rsm-shuffle-gap", type=float, default=0.10)
    parser.add_argument("--min-decoder-top1", type=float, default=0.25)
    parser.add_argument("--min-decoder-topk-accuracy", type=float, default=0.70)
    parser.add_argument("--min-decoder-similarity-gap", type=float, default=0.10)
    parser.add_argument("--min-signal-noise-ratio", type=float, default=1.0)
    parser.add_argument("--min-mean-sample-active-fraction", type=float, default=0.005)
    parser.add_argument("--max-mean-sample-active-fraction", type=float, default=0.50)
    parser.add_argument("--max-sample-active-fraction", type=float, default=0.80)
    parser.add_argument("--min-lifetime-active-fraction", type=float, default=0.10)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    loaded = load_site_rates(
        args.video_site_rates_csv,
        args.population,
        sheet_side=args.sheet_side,
        core_side=args.core_side,
    )
    rates = loaded.rates
    metrics: dict[str, float | int | str | list[int]] = {
        "source": str(args.video_site_rates_csv),
        "population": args.population,
        "repeat_count": float(rates.shape[0]),
        "frame_count": float(rates.shape[1]),
        "site_count": float(rates.shape[2]),
        "repeat_indices": loaded.repeats,
        "frame_indices_first_last": [loaded.frames[0], loaded.frames[-1]],
        "site_indices_first_last": [loaded.sites[0], loaded.sites[-1]],
    }
    metrics.update(pairwise_repeat_reliability(rates, rng=rng))
    metrics.update(rsm_stability(rates, rng=rng))
    metrics.update(correlation_template_decoder(rates, args.top_k))
    metrics.update(population_snr_and_sparsity(rates, args.active_threshold_hz))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    ok = validate(metrics, args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
