#!/usr/bin/env python3
"""Analyze native Stage1 temporal learning artifacts.

This is an analysis-only harness for the standalone native Stage1 path. It does
not alter training behavior. It consumes one or more native Stage1 checkpoint
JSON files plus their `W_ctx_pred` sidecar binaries and produces a compact JSON
summary covering:

- whether the implemented Stage1 path uses an explicit global loss/optimizer
  versus local plastic updates and gate metrics
- checkpoint-wise gate metrics already emitted by the native executable
- simple readout/sequence metrics derived from the learned `W_ctx_pred`
- extent-wise trends across `n_trials`

Assumptions:
- native Stage1 checkpoints are produced by `expectation_snn_native stage1-train`
- `W_ctx_pred` has shape `(192, 192)` flattened as float64 row-major
- H layout is 12 channels x 16 cells/channel, but Stage1 schedule uses the 6
  orientation channels on the even indices `[0, 2, 4, 6, 8, 10]`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

N_ORIENTATIONS = 6
H_CHANNELS = 12
H_CELLS_PER_CHANNEL = 16
W_SHAPE = (H_CHANNELS * H_CELLS_PER_CHANNEL, H_CHANNELS * H_CELLS_PER_CHANNEL)
ORI_CHANNELS = np.arange(0, H_CHANNELS, 2, dtype=np.int32)
DERANGEMENT = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_path(path: Path | str) -> str:
    return str(Path(path).resolve())


def _resolve_w_path(checkpoint_path: Path, checkpoint_json: dict[str, Any]) -> Path:
    w_path = Path(checkpoint_json["checkpoint"]["W_ctx_pred_path"])
    if not w_path.is_absolute():
        w_path = checkpoint_path.parent / w_path
    return w_path


def _load_w_matrix(w_path: Path) -> np.ndarray:
    w = np.fromfile(w_path, dtype=np.float64)
    if w.size != W_SHAPE[0] * W_SHAPE[1]:
        raise ValueError(
            f"{w_path} has {w.size} float64 values, expected {W_SHAPE[0] * W_SHAPE[1]}"
        )
    return w.reshape(W_SHAPE)


def _orientation_weight_matrix_mean(w: np.ndarray) -> np.ndarray:
    channel_mean = w.reshape(H_CHANNELS, H_CELLS_PER_CHANNEL, H_CHANNELS, H_CELLS_PER_CHANNEL).mean(axis=(1, 3))
    return channel_mean[np.ix_(ORI_CHANNELS, ORI_CHANNELS)]


def _safe_normalize_row(row: np.ndarray) -> np.ndarray:
    total = float(row.sum())
    if total <= 0.0:
        return np.full_like(row, 1.0 / row.size)
    return row / total


def _row_entropy(row: np.ndarray) -> float:
    p = _safe_normalize_row(row)
    return float(-(p * np.log2(np.clip(p, 1e-12, None))).sum())


def _readout_metrics(
    weight_matrix: np.ndarray,
    pairs: np.ndarray,
    expected_trailer_idx: np.ndarray,
    is_expected: np.ndarray,
) -> dict[str, Any]:
    row_argmax = weight_matrix.argmax(axis=1).astype(np.int32)
    expected_margin = []
    expected_fraction = []
    row_entropy = []
    expected_rank = []
    for leader in range(N_ORIENTATIONS):
        row = weight_matrix[leader]
        target = int(DERANGEMENT[leader])
        other = np.delete(row, target)
        expected_margin.append(float(row[target] - other.max()))
        expected_fraction.append(float(row[target] / row.sum()) if row.sum() > 0.0 else 0.0)
        row_entropy.append(_row_entropy(row))
        expected_rank.append(int((-row).argsort().tolist().index(target) + 1))

    trial_pred = row_argmax[pairs[:, 0]]
    trial_expected = expected_trailer_idx
    trial_actual = pairs[:, 1]
    unexpected_mask = is_expected == 0

    unexpected_expected_minus_actual = (
        weight_matrix[pairs[unexpected_mask, 0], trial_expected[unexpected_mask]]
        - weight_matrix[pairs[unexpected_mask, 0], trial_actual[unexpected_mask]]
        if unexpected_mask.any()
        else np.zeros(0, dtype=np.float64)
    )

    confusion = np.zeros((N_ORIENTATIONS, N_ORIENTATIONS), dtype=np.float64)
    for leader in range(N_ORIENTATIONS):
        confusion[leader, row_argmax[leader]] += 1.0

    return {
        "orientation_weight_matrix_mean": weight_matrix.tolist(),
        "expected_argmax_accuracy": float(np.mean(row_argmax == DERANGEMENT)),
        "expected_margin_mean": float(np.mean(expected_margin)),
        "expected_target_fraction_mean": float(np.mean(expected_fraction)),
        "row_entropy_mean_bits": float(np.mean(row_entropy)),
        "expected_rank_mean": float(np.mean(expected_rank)),
        "trial_argmax_matches_expected": float(np.mean(trial_pred == trial_expected)),
        "trial_argmax_matches_actual": float(np.mean(trial_pred == trial_actual)),
        "unexpected_trial_argmax_matches_expected": (
            float(np.mean(trial_pred[unexpected_mask] == trial_expected[unexpected_mask]))
            if unexpected_mask.any()
            else 0.0
        ),
        "unexpected_trial_argmax_matches_actual": (
            float(np.mean(trial_pred[unexpected_mask] == trial_actual[unexpected_mask]))
            if unexpected_mask.any()
            else 0.0
        ),
        "unexpected_expected_minus_actual_weight_mean": (
            float(np.mean(unexpected_expected_minus_actual))
            if unexpected_expected_minus_actual.size > 0
            else 0.0
        ),
        "unexpected_expected_minus_actual_weight_min": (
            float(np.min(unexpected_expected_minus_actual))
            if unexpected_expected_minus_actual.size > 0
            else 0.0
        ),
        "unexpected_expected_minus_actual_weight_max": (
            float(np.max(unexpected_expected_minus_actual))
            if unexpected_expected_minus_actual.size > 0
            else 0.0
        ),
        "predicted_trailer_confusion_by_leader": confusion.tolist(),
    }


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    w_path = _resolve_w_path(path, data)
    w = _load_w_matrix(w_path)
    pairs = np.asarray(data["schedule"]["pairs"], dtype=np.int32)
    expected = np.asarray(data["schedule"]["expected_trailer_idx"], dtype=np.int32)
    is_expected = np.asarray(data["schedule"]["is_expected"], dtype=np.int32)
    weight_matrix = _orientation_weight_matrix_mean(w)
    readout = _readout_metrics(weight_matrix, pairs, expected, is_expected)
    return {
        "checkpoint_path": str(path),
        "w_path": str(w_path),
        "content_hash_fnv1a64": data["content_hash_fnv1a64"],
        "seed": int(data["seed"]),
        "n_trials": int(data["n_trials"]),
        "passed": bool(data["passed"]),
        "native_scientific_stage1_passed": bool(data["native_scientific_stage1_passed"]),
        "ctx_pred_constants": data["ctx_pred_constants"],
        "metrics": data["metrics"],
        "weight_stats": data["weight_stats"],
        "timing_seconds": data["timing_seconds"],
        "schedule_bias_fraction_expected": float(np.mean(is_expected)),
        "readout": readout,
    }


def _heldout_readout_metrics(
    pred_pretrailer_channel_counts: np.ndarray,
    leader_idx: np.ndarray,
    trailer_idx: np.ndarray,
    expected_trailer_idx: np.ndarray,
    is_expected: np.ndarray,
) -> dict[str, Any]:
    ori_counts = pred_pretrailer_channel_counts[:, ORI_CHANNELS]
    if ori_counts.shape[0] == 0:
        raise ValueError("held-out eval must contain at least one trial")

    leader_means = np.zeros((N_ORIENTATIONS, N_ORIENTATIONS), dtype=np.float64)
    leader_trial_counts = np.zeros(N_ORIENTATIONS, dtype=np.int32)
    for leader in range(N_ORIENTATIONS):
        mask = leader_idx == leader
        leader_trial_counts[leader] = int(mask.sum())
        if mask.any():
            leader_means[leader] = ori_counts[mask].mean(axis=0)

    row_argmax = leader_means.argmax(axis=1).astype(np.int32)
    expected_margin = []
    expected_fraction = []
    row_entropy = []
    expected_rank = []
    for leader in range(N_ORIENTATIONS):
        row = leader_means[leader]
        target = int(DERANGEMENT[leader])
        other = np.delete(row, target)
        expected_margin.append(float(row[target] - other.max()))
        expected_fraction.append(float(row[target] / row.sum()) if row.sum() > 0.0 else 0.0)
        row_entropy.append(_row_entropy(row))
        expected_rank.append(int((-row).argsort().tolist().index(target) + 1))

    trial_pred = ori_counts.argmax(axis=1).astype(np.int32)
    unexpected_mask = is_expected == 0
    unexpected_expected_minus_actual = (
        ori_counts[np.where(unexpected_mask)[0], expected_trailer_idx[unexpected_mask]]
        - ori_counts[np.where(unexpected_mask)[0], trailer_idx[unexpected_mask]]
        if unexpected_mask.any()
        else np.zeros(0, dtype=np.float64)
    )

    confusion = np.zeros((N_ORIENTATIONS, N_ORIENTATIONS), dtype=np.float64)
    for leader in range(N_ORIENTATIONS):
        mask = leader_idx == leader
        if not mask.any():
            continue
        for pred in trial_pred[mask]:
            confusion[leader, pred] += 1.0
        confusion[leader] /= float(mask.sum())

    return {
        "orientation_pretrailer_channel_counts_mean": leader_means.tolist(),
        "expected_argmax_accuracy": float(np.mean(row_argmax == DERANGEMENT)),
        "expected_margin_mean": float(np.mean(expected_margin)),
        "expected_target_fraction_mean": float(np.mean(expected_fraction)),
        "row_entropy_mean_bits": float(np.mean(row_entropy)),
        "expected_rank_mean": float(np.mean(expected_rank)),
        "trial_argmax_matches_expected": float(np.mean(trial_pred == expected_trailer_idx)),
        "trial_argmax_matches_actual": float(np.mean(trial_pred == trailer_idx)),
        "unexpected_trial_argmax_matches_expected": (
            float(np.mean(trial_pred[unexpected_mask] == expected_trailer_idx[unexpected_mask]))
            if unexpected_mask.any()
            else 0.0
        ),
        "unexpected_trial_argmax_matches_actual": (
            float(np.mean(trial_pred[unexpected_mask] == trailer_idx[unexpected_mask]))
            if unexpected_mask.any()
            else 0.0
        ),
        "unexpected_expected_minus_actual_count_mean": (
            float(np.mean(unexpected_expected_minus_actual))
            if unexpected_expected_minus_actual.size > 0
            else 0.0
        ),
        "unexpected_expected_minus_actual_count_min": (
            float(np.min(unexpected_expected_minus_actual))
            if unexpected_expected_minus_actual.size > 0
            else 0.0
        ),
        "unexpected_expected_minus_actual_count_max": (
            float(np.max(unexpected_expected_minus_actual))
            if unexpected_expected_minus_actual.size > 0
            else 0.0
        ),
        "predicted_trailer_confusion_by_leader": confusion.tolist(),
    }


def _heldout_eval_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    schedule = data["schedule"]
    pairs = np.asarray(schedule["pairs"], dtype=np.int32)
    leader_idx = pairs[:, 0]
    trailer_idx = pairs[:, 1]
    expected = np.asarray(schedule["expected_trailer_idx"], dtype=np.int32)
    is_expected = np.asarray(schedule["is_expected"], dtype=np.int32)
    pred_pre = np.asarray(
        data["per_trial"]["pred_pretrailer_channel_counts"],
        dtype=np.float64,
    )
    if pred_pre.shape != (len(pairs), H_CHANNELS):
        raise ValueError(
            f"{path} pred_pretrailer_channel_counts shape {pred_pre.shape} "
            f"!= ({len(pairs)}, {H_CHANNELS})"
        )
    readout = _heldout_readout_metrics(
        pred_pre, leader_idx, trailer_idx, expected, is_expected
    )
    metrics = data["metrics"]
    return {
        "artifact_path": str(path),
        "checkpoint_path": data["checkpoint"]["json_path"],
        "heldout_schedule_seed": int(data["heldout_schedule_seed"]),
        "heldout_schedule_hash_fnv1a64": data["heldout_schedule_hash_fnv1a64"],
        "schedule_bias_fraction_expected": float(np.mean(is_expected)),
        "metrics": {
            "h_context_persistence_ms": float(metrics["h_context_persistence_ms"]),
            "forecast_probability": float(
                metrics["h_prediction_pretrailer_forecast_probability"]
            ),
            "no_runaway_population_rate_hz": float(metrics["no_runaway_max_rate_hz"]),
            "max_cpu_cuda_error": float(data["max_cpu_cuda_error"]),
        },
        "readout": readout,
    }


def _trend_summary(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(checkpoints, key=lambda item: item["n_trials"])
    first = ordered[0]
    last = ordered[-1]

    def series(key_fn):
        return [
            {"n_trials": item["n_trials"], "value": key_fn(item)}
            for item in ordered
        ]

    out = {
        "n_trials": [item["n_trials"] for item in ordered],
        "forecast_probability": series(
            lambda item: item["metrics"]["forecast_probability"]
        ),
        "persistence_ms": series(
            lambda item: item["metrics"]["h_context_persistence_ms"]
        ),
        "no_runaway_population_rate_hz": series(
            lambda item: item["metrics"]["no_runaway_population_rate_hz"]
        ),
        "expected_argmax_accuracy": series(
            lambda item: item["readout"]["expected_argmax_accuracy"]
        ),
        "expected_margin_mean": series(
            lambda item: item["readout"]["expected_margin_mean"]
        ),
        "unexpected_expected_minus_actual_weight_mean": series(
            lambda item: item["readout"]["unexpected_expected_minus_actual_weight_mean"]
        ),
        "delta_from_first_to_last": {
            "forecast_probability": (
                last["metrics"]["forecast_probability"]
                - first["metrics"]["forecast_probability"]
            ),
            "persistence_ms": (
                last["metrics"]["h_context_persistence_ms"]
                - first["metrics"]["h_context_persistence_ms"]
            ),
            "expected_argmax_accuracy": (
                last["readout"]["expected_argmax_accuracy"]
                - first["readout"]["expected_argmax_accuracy"]
            ),
            "expected_margin_mean": (
                last["readout"]["expected_margin_mean"]
                - first["readout"]["expected_margin_mean"]
            ),
            "unexpected_expected_minus_actual_weight_mean": (
                last["readout"]["unexpected_expected_minus_actual_weight_mean"]
                - first["readout"]["unexpected_expected_minus_actual_weight_mean"]
            ),
        },
    }
    if all("heldout_eval" in item for item in ordered):
        out["heldout_forecast_probability"] = series(
            lambda item: item["heldout_eval"]["metrics"]["forecast_probability"]
        )
        out["heldout_expected_argmax_accuracy"] = series(
            lambda item: item["heldout_eval"]["readout"]["expected_argmax_accuracy"]
        )
        out["heldout_expected_margin_mean"] = series(
            lambda item: item["heldout_eval"]["readout"]["expected_margin_mean"]
        )
        out["heldout_trial_argmax_matches_expected"] = series(
            lambda item: item["heldout_eval"]["readout"]["trial_argmax_matches_expected"]
        )
        out["heldout_trial_argmax_matches_actual"] = series(
            lambda item: item["heldout_eval"]["readout"]["trial_argmax_matches_actual"]
        )
        out["delta_from_first_to_last"].update(
            {
                "heldout_forecast_probability": (
                    last["heldout_eval"]["metrics"]["forecast_probability"]
                    - first["heldout_eval"]["metrics"]["forecast_probability"]
                ),
                "heldout_expected_argmax_accuracy": (
                    last["heldout_eval"]["readout"]["expected_argmax_accuracy"]
                    - first["heldout_eval"]["readout"]["expected_argmax_accuracy"]
                ),
                "heldout_expected_margin_mean": (
                    last["heldout_eval"]["readout"]["expected_margin_mean"]
                    - first["heldout_eval"]["readout"]["expected_margin_mean"]
                ),
                "heldout_trial_argmax_matches_expected": (
                    last["heldout_eval"]["readout"]["trial_argmax_matches_expected"]
                    - first["heldout_eval"]["readout"]["trial_argmax_matches_expected"]
                ),
                "heldout_trial_argmax_matches_actual": (
                    last["heldout_eval"]["readout"]["trial_argmax_matches_actual"]
                    - first["heldout_eval"]["readout"]["trial_argmax_matches_actual"]
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--heldout-evals",
        nargs="*",
        default=[],
        type=Path,
        help="Optional held-out native Stage1 eval artifacts matching the checkpoints.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--sensory-baseline-grating-rate-hz",
        type=float,
        default=100.0,
        help="Phase 1 accepted sensory operating point. Context only; native Stage1 does not consume this knob.",
    )
    args = parser.parse_args()

    checkpoints = [_checkpoint_summary(path) for path in args.checkpoints]
    checkpoints.sort(key=lambda item: item["n_trials"])
    if args.heldout_evals:
        heldout_by_checkpoint = {}
        for path in args.heldout_evals:
            heldout = _heldout_eval_summary(path)
            key = _normalize_path(heldout["checkpoint_path"])
            heldout_by_checkpoint[key] = heldout
        missing = []
        for checkpoint in checkpoints:
            key = _normalize_path(checkpoint["checkpoint_path"])
            heldout = heldout_by_checkpoint.get(key)
            if heldout is None:
                missing.append(checkpoint["checkpoint_path"])
                continue
            if heldout["metrics"]["max_cpu_cuda_error"] > 1e-8:
                raise ValueError(
                    f"{heldout['artifact_path']} max_cpu_cuda_error "
                    f"{heldout['metrics']['max_cpu_cuda_error']} > 1e-8"
                )
            checkpoint["heldout_eval"] = heldout
        if missing:
            raise ValueError(
                "missing held-out eval artifacts for checkpoints: "
                + ", ".join(missing)
            )

    summary = {
        "schema_version": 2,
        "artifact_kind": "native_stage1_temporal_learning_analysis",
        "baseline_context": {
            "accepted_sensory_grating_rate_hz": args.sensory_baseline_grating_rate_hz,
            "note": "Context only. The current standalone native Stage1 trainer does not consume the sensory grating-rate knob.",
        },
        "training_mechanism": {
            "explicit_global_loss_present": False,
            "optimizer_present": False,
            "implemented_update_rule": {
                "local_pre_post_traces": True,
                "eligibility_trace": True,
                "delayed_gate_update": True,
                "homeostatic_drift_term": True,
                "row_cap": True,
                "weight_clip": True,
            },
            "summary": (
                "Native Stage1 currently performs local ctx->pred plastic updates "
                "with xpre/xpost/eligibility traces, a delayed gate update, row "
                "cap, and gate metrics. It does not implement an explicit global "
                "loss or optimizer."
            ),
        },
        "checkpoints": checkpoints,
        "trends": _trend_summary(checkpoints),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    ordered = summary["checkpoints"]
    first = ordered[0]
    last = ordered[-1]
    print("command=analyze-native-stage1-temporal-learning")
    print("status=PASS")
    print(f"output_path={args.out}")
    print(f"n_checkpoints={len(ordered)}")
    print(
        "forecast_first_last="
        f"{first['metrics']['forecast_probability']:.6f}->"
        f"{last['metrics']['forecast_probability']:.6f}"
    )
    print(
        "persistence_first_last_ms="
        f"{first['metrics']['h_context_persistence_ms']:.6f}->"
        f"{last['metrics']['h_context_persistence_ms']:.6f}"
    )
    print(
        "expected_margin_first_last="
        f"{first['readout']['expected_margin_mean']:.6f}->"
        f"{last['readout']['expected_margin_mean']:.6f}"
    )
    print(
        "unexpected_expected_minus_actual_first_last="
        f"{first['readout']['unexpected_expected_minus_actual_weight_mean']:.6f}->"
        f"{last['readout']['unexpected_expected_minus_actual_weight_mean']:.6f}"
    )
    if "heldout_eval" in first and "heldout_eval" in last:
        print(
            "heldout_forecast_first_last="
            f"{first['heldout_eval']['metrics']['forecast_probability']:.6f}->"
            f"{last['heldout_eval']['metrics']['forecast_probability']:.6f}"
        )
        print(
            "heldout_expected_margin_first_last="
            f"{first['heldout_eval']['readout']['expected_margin_mean']:.6f}->"
            f"{last['heldout_eval']['readout']['expected_margin_mean']:.6f}"
        )
        print(
            "heldout_argmax_expected_first_last="
            f"{first['heldout_eval']['readout']['trial_argmax_matches_expected']:.6f}->"
            f"{last['heldout_eval']['readout']['trial_argmax_matches_expected']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
