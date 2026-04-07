#!/usr/bin/env python3
"""Anchor-averaged, multi-seed M7 evaluation for publication hardening.

This script reuses the canonical ON/OFF analysis logic from
``scripts.analyze_representation`` and adds the orchestration needed for:

1. repeated-seed M7 evaluation;
2. anchor-averaged decoding summaries;
3. paired ON/OFF contrasts;
4. JSON and CSV output suitable for publication tables.

The implementation intentionally keeps the statistical protocol outside the
human-readable ``analyze_representation.py`` CLI so the canonical report stays
simple while publication hardening remains reproducible and scriptable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_representation import (
    ANALYSIS_ANCHORS_DEG,
    CueConfig,
    load_model,
    metric_match_vs_near_miss_decoding,
    sanity_check_ablation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    """One checkpoint/config/cue-mode evaluation target."""

    label: str
    checkpoint: str
    config: str
    cue_mode: str


def _bootstrap_mean_ci(
    values: Sequence[float],
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Return deterministic bootstrap mean and 95% CI for one metric series."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1D series")

    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n_samples": 1}

    rs = np.random.RandomState(seed)
    boot = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        sample = arr[rs.randint(0, arr.size, size=arr.size)]
        boot[i] = sample.mean()
    return {
        "mean": mean,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_samples": int(arr.size),
    }


def _summarize_delta_samples(
    delta_samples: dict[str, list[float]],
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Summarize repeated-seed M7 delta samples for each tested delta."""
    return {
        delta_key: _bootstrap_mean_ci(values, draws=draws, seed=seed + i)
        for i, (delta_key, values) in enumerate(sorted(delta_samples.items()))
    }


def _evaluate_run(
    spec: RunSpec,
    device: torch.device,
    n_train: int,
    n_test: int,
    noise_std: float,
    readout_noise_std: float,
    resample_seeds: Sequence[int],
    cue_cfg: CueConfig,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[dict, dict]:
    """Run the anchor-averaged M7 metric repeatedly for one checkpoint."""
    net, _, _ = load_model(spec.checkpoint, spec.config, device)
    sanity = sanity_check_ablation(net, device)
    if not sanity["ablation_zero"]:
        raise ValueError(
            f"sanity_check_ablation failed for {spec.label}: {sanity}"
        )

    delta_samples: dict[str, list[float]] = {"delta_3": [], "delta_5": [], "delta_10": []}
    for seed in resample_seeds:
        metric = metric_match_vs_near_miss_decoding(
            net,
            device,
            n_train=n_train,
            n_test=n_test,
            noise_std=noise_std,
            readout_noise_std=readout_noise_std,
            seed=seed,
            anchors=ANALYSIS_ANCHORS_DEG,
            cue_cfg=cue_cfg,
        )
        for delta_key in delta_samples:
            delta_samples[delta_key].append(float(metric[delta_key]["delta_acc"]))

    run_result = {
        "checkpoint": spec.checkpoint,
        "config": spec.config,
        "cue_cfg": {
            "mode": cue_cfg.mode,
            "contrast": cue_cfg.contrast,
            "prestimulus_steps": cue_cfg.prestimulus_steps,
            "offset": cue_cfg.offset,
        },
        "samples": delta_samples,
        "summary": _summarize_delta_samples(
            delta_samples, draws=bootstrap_draws, seed=bootstrap_seed
        ),
    }

    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_result, sanity


def _combine_runs(
    label: str,
    member_labels: Sequence[str],
    runs: dict[str, dict],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict:
    """Combine repeated-seed M7 samples across multiple run labels."""
    delta_samples: dict[str, list[float]] = {"delta_3": [], "delta_5": [], "delta_10": []}
    for member in member_labels:
        for delta_key in delta_samples:
            delta_samples[delta_key].extend(runs[member]["samples"][delta_key])
    return {
        "members": list(member_labels),
        "samples": delta_samples,
        "summary": _summarize_delta_samples(
            delta_samples, draws=bootstrap_draws, seed=bootstrap_seed
        ),
    }


def _pair_runs(
    label: str,
    label_a: str,
    label_b: str,
    runs: dict[str, dict],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict:
    """Compute paired per-seed M7 differences between two run labels."""
    delta_samples: dict[str, list[float]] = {"delta_3": [], "delta_5": [], "delta_10": []}
    for delta_key in delta_samples:
        a = runs[label_a]["samples"][delta_key]
        b = runs[label_b]["samples"][delta_key]
        if len(a) != len(b):
            raise ValueError(
                f"paired labels {label_a}/{label_b} have mismatched sample counts "
                f"for {delta_key}: {len(a)} vs {len(b)}"
            )
        delta_samples[delta_key] = [float(x - y) for x, y in zip(a, b)]
    return {
        "labels": [label_a, label_b],
        "samples": delta_samples,
        "summary": _summarize_delta_samples(
            delta_samples, draws=bootstrap_draws, seed=bootstrap_seed
        ),
    }


def _write_csv(
    output_csv: Path,
    runs: dict[str, dict],
    combined: dict[str, dict],
    pairs: dict[str, dict],
) -> None:
    """Write flat CSV summaries for run/combine/pair publication tables."""
    rows: list[dict[str, object]] = []

    def _append_rows(scope: str, label: str, payload: dict) -> None:
        for delta_key, summary in payload["summary"].items():
            rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "delta": delta_key,
                    "mean": summary["mean"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "n_samples": summary["n_samples"],
                }
            )

    for label, payload in runs.items():
        _append_rows("run", label, payload)
    for label, payload in combined.items():
        _append_rows("combined", label, payload)
    for label, payload in pairs.items():
        _append_rows("pair", label, payload)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["scope", "label", "delta", "mean", "ci_low", "ci_high", "n_samples"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_runs(raw_runs: Sequence[Sequence[str]]) -> list[RunSpec]:
    """Parse ``--run`` CLI records into typed run specifications."""
    specs: list[RunSpec] = []
    seen_labels: set[str] = set()
    for raw in raw_runs:
        label, checkpoint, config, cue_mode = raw
        if label in seen_labels:
            raise ValueError(f"duplicate run label: {label}")
        seen_labels.add(label)
        specs.append(RunSpec(label=label, checkpoint=checkpoint, config=config, cue_mode=cue_mode))
    return specs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for repeated-seed anchor-averaged M7 evaluation."""
    parser = argparse.ArgumentParser(description="Anchor-averaged repeated-seed M7 evaluation")
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("LABEL", "CHECKPOINT", "CONFIG", "CUE_MODE"),
        required=True,
        help="Run specification. Repeat for each checkpoint/cue condition.",
    )
    parser.add_argument(
        "--combine",
        action="append",
        nargs="+",
        default=[],
        metavar=("LABEL", "MEMBER"),
        help="Combined summary: first token is output label, remaining tokens are run labels.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "RUN_A", "RUN_B"),
        help="Paired difference summary RUN_A - RUN_B.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--readout-noise-std", type=float, default=0.3)
    parser.add_argument("--cue-contrast", type=float, default=1.0)
    parser.add_argument("--cue-prestimulus-steps", type=int, default=4)
    parser.add_argument("--cue-offset", type=float, default=0.0)
    parser.add_argument("--resample-seed-start", type=int, default=1000)
    parser.add_argument("--resample-seed-count", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run repeated-seed anchor-averaged M7 evaluation and save JSON/CSV."""
    args = parse_args()
    run_specs = _parse_runs(args.run)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    resample_seeds = list(
        range(args.resample_seed_start, args.resample_seed_start + args.resample_seed_count)
    )

    runs: dict[str, dict] = {}
    sanity_checks: dict[str, dict] = {}
    for spec in run_specs:
        cue_cfg = CueConfig(
            mode=spec.cue_mode,
            contrast=args.cue_contrast,
            prestimulus_steps=args.cue_prestimulus_steps if spec.cue_mode != "none" else 0,
            offset=args.cue_offset,
        )
        logger.info("Evaluating %s", spec.label)
        run_result, sanity = _evaluate_run(
            spec=spec,
            device=device,
            n_train=args.n_train,
            n_test=args.n_test,
            noise_std=args.noise_std,
            readout_noise_std=args.readout_noise_std,
            resample_seeds=resample_seeds,
            cue_cfg=cue_cfg,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        runs[spec.label] = run_result
        sanity_checks[spec.label] = sanity

    combined: dict[str, dict] = {}
    for raw in args.combine:
        if len(raw) < 3:
            raise ValueError("--combine requires one label and at least two member labels")
        label, *members = raw
        combined[label] = _combine_runs(
            label=label,
            member_labels=members,
            runs=runs,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed + 100,
        )

    pairs: dict[str, dict] = {}
    for label, run_a, run_b in args.pair:
        pairs[label] = _pair_runs(
            label=label,
            label_a=run_a,
            label_b=run_b,
            runs=runs,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed + 200,
        )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": {
            "anchors": list(ANALYSIS_ANCHORS_DEG),
            "n_train": args.n_train,
            "n_test": args.n_test,
            "noise_std": args.noise_std,
            "readout_noise_std": args.readout_noise_std,
            "cue_contrast": args.cue_contrast,
            "cue_prestimulus_steps": args.cue_prestimulus_steps,
            "cue_offset": args.cue_offset,
            "resample_seeds": resample_seeds,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "sanity_checks": sanity_checks,
        "runs": runs,
        "combined": combined,
        "pairs": pairs,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_csv, runs=runs, combined=combined, pairs=pairs)
    logger.info("Saved JSON to %s", output_json)
    logger.info("Saved CSV to %s", output_csv)


if __name__ == "__main__":
    main()
