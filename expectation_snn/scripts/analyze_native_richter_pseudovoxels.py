#!/usr/bin/env python3
"""Post-hoc pseudo-voxel readout for native Richter JSON artifacts.

Native ``richter-dampening`` artifacts expose V1_E responses as 12 aggregate
channel counts, not per-cell spike counts. To reuse the existing assay forward
model with the smallest compatibility layer, this utility treats each channel
aggregate as one pseudo-cell and assigns channels to four contiguous
pseudo-voxels with the same formula as
``richter_crossover._voxel_spatial_bins``:

    voxel = channel * n_voxels // n_channels

That preserves the native channel-level evidence while matching the spatial
binning used by the Brian/Richter crossover assay.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

# Allow direct path execution from the repo checkout:
#   python expectation_snn/scripts/analyze_native_richter_pseudovoxels.py ...
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expectation_snn.assays.metrics import (
    _VOXEL_MODEL_FAMILIES,
    pseudo_voxel_forward_model,
)


N_CHANNELS = 12
N_VOXELS = 4
N_100MS_BINS = 5
DEFAULT_THINNING_KEEP_P = (0.05, 0.02, 0.015, 0.01)
DEFAULT_N_THINNING_SEEDS = 64
DEFAULT_N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260424
EXPECTED_TRAILER_LABELS = (0, 2, 4, 6, 8, 10)

DEFAULT_STAGE3_DIR = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424"
)

DEFAULT_ARTIFACTS: Mapping[str, str] = {
    "feedback_on_sigma22": (
        "richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_"
        "feedback_r03333333333333333_gtotal20_hpred_replay_som_center010.json"
    ),
    "feedback_on_sigma16": (
        "richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_sigma16_"
        "feedback_r03333333333333333_gtotal20_hpred_replay_som_center010.json"
    ),
    "sensory_only_sigma22": (
        "richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_sigma22_"
        "sensory_only_feedback_g0_r03333333333333333_center010.json"
    ),
    "sensory_only_sigma16": (
        "richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_sigma16_"
        "sensory_only_feedback_g0_r03333333333333333_center010.json"
    ),
}


def _jsonify(value: Any) -> Any:
    """Convert numpy values and non-finite floats into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _voxel_spatial_bins(
    n_channels: int = N_CHANNELS,
    n_voxels: int = N_VOXELS,
) -> np.ndarray:
    """Return contiguous channel-to-voxel bins matching richter_crossover."""
    channel_per_cell = np.arange(n_channels, dtype=np.int64)
    return (channel_per_cell * n_voxels // n_channels).astype(np.int64)


def _condition_masks(is_expected: np.ndarray) -> Mapping[str, np.ndarray]:
    return {
        "expected": is_expected.astype(bool),
        "unexpected": ~is_expected.astype(bool),
    }


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _mean_or_nan(values: np.ndarray, axis: int = 0) -> np.ndarray:
    if values.shape[axis] == 0:
        out_shape = tuple(s for i, s in enumerate(values.shape) if i != axis)
        return np.full(out_shape, np.nan, dtype=np.float64)
    return np.mean(values, axis=axis)


def _load_native_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)

    trial_data = artifact.get("trial_data")
    if not isinstance(trial_data, dict):
        raise ValueError(f"{path}: missing object field trial_data")

    required = (
        "v1_e_trailer_channel_counts",
        "v1_e_trailer_100ms_channel_counts",
        "trailer_channel",
        "is_expected",
    )
    missing = [k for k in required if k not in trial_data]
    if missing:
        raise ValueError(f"{path}: missing trial_data fields {missing}")

    channel_counts = np.asarray(
        trial_data["v1_e_trailer_channel_counts"], dtype=np.float64,
    )
    channel_counts_100ms = np.asarray(
        trial_data["v1_e_trailer_100ms_channel_counts"], dtype=np.float64,
    )
    trailer_channel = np.asarray(trial_data["trailer_channel"], dtype=np.int64)
    is_expected = np.asarray(trial_data["is_expected"], dtype=np.int64)

    if channel_counts.ndim != 2 or channel_counts.shape[1] != N_CHANNELS:
        raise ValueError(
            f"{path}: v1_e_trailer_channel_counts shape "
            f"{channel_counts.shape}, expected (n_trials, {N_CHANNELS})"
        )
    if (
        channel_counts_100ms.ndim != 3
        or channel_counts_100ms.shape[1:] != (N_100MS_BINS, N_CHANNELS)
    ):
        raise ValueError(
            f"{path}: v1_e_trailer_100ms_channel_counts shape "
            f"{channel_counts_100ms.shape}, expected "
            f"(n_trials, {N_100MS_BINS}, {N_CHANNELS})"
        )
    n_trials = channel_counts.shape[0]
    if trailer_channel.shape != (n_trials,):
        raise ValueError(f"{path}: trailer_channel shape mismatch")
    if is_expected.shape != (n_trials,):
        raise ValueError(f"{path}: is_expected shape mismatch")
    if np.any((trailer_channel < 0) | (trailer_channel >= N_CHANNELS)):
        raise ValueError(f"{path}: trailer_channel outside 0..{N_CHANNELS - 1}")

    return {
        "path": path,
        "artifact": artifact,
        "channel_counts": channel_counts,
        "channel_counts_100ms": channel_counts_100ms,
        "trailer_channel": trailer_channel,
        "is_expected": is_expected,
    }


def _pool_voxels(channel_counts: np.ndarray, voxel_bins: np.ndarray) -> np.ndarray:
    pooled = np.zeros((channel_counts.shape[0], N_VOXELS), dtype=np.float64)
    for voxel in range(N_VOXELS):
        pooled[:, voxel] = channel_counts[:, voxel_bins == voxel].sum(axis=1)
    return pooled


def _pool_voxels_100ms(
    channel_counts_100ms: np.ndarray,
    voxel_bins: np.ndarray,
) -> np.ndarray:
    pooled = np.zeros(
        (channel_counts_100ms.shape[0], N_100MS_BINS, N_VOXELS),
        dtype=np.float64,
    )
    for voxel in range(N_VOXELS):
        pooled[:, :, voxel] = channel_counts_100ms[:, :, voxel_bins == voxel].sum(
            axis=2,
        )
    return pooled


def _means_by_condition_channel_voxel(
    voxel_counts: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for condition, cond_mask in _condition_masks(is_expected).items():
        condition_out: Dict[str, Any] = {}
        for ch in range(N_CHANNELS):
            mask = cond_mask & (trailer_channel == ch)
            condition_out[str(ch)] = {
                "n_trials": int(mask.sum()),
                "mean_voxel_response": _mean_or_nan(voxel_counts[mask], axis=0),
                "total_mean_response": _safe_mean(voxel_counts[mask].sum(axis=1)),
            }
        out[condition] = condition_out
    return out


def _means_by_condition_bin_channel_voxel(
    voxel_counts_100ms: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for condition, cond_mask in _condition_masks(is_expected).items():
        condition_out: Dict[str, Any] = {}
        for ch in range(N_CHANNELS):
            mask = cond_mask & (trailer_channel == ch)
            condition_out[str(ch)] = {
                "n_trials": int(mask.sum()),
                "mean_100ms_voxel_response": _mean_or_nan(
                    voxel_counts_100ms[mask],
                    axis=0,
                ),
            }
        out[condition] = condition_out
    return out


def _existing_metric_pair(metrics: Mapping[str, Any], name: str) -> Dict[str, Any]:
    expected = metrics.get(f"expected_{name}")
    unexpected = metrics.get(f"unexpected_{name}")
    expected_f = float(expected) if expected is not None else float("nan")
    unexpected_f = float(unexpected) if unexpected is not None else float("nan")
    return {
        "expected": expected_f,
        "unexpected": unexpected_f,
        "unexpected_minus_expected": unexpected_f - expected_f,
    }


def _expected_unexpected_summary(
    artifact: Mapping[str, Any],
    voxel_counts: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Any]:
    masks = _condition_masks(is_expected)
    response = {
        condition: _safe_mean(voxel_counts[mask].sum(axis=1))
        for condition, mask in masks.items()
    }
    response["unexpected_minus_expected"] = (
        response["unexpected"] - response["expected"]
    )
    metrics = artifact.get("metrics", {})
    return {
        "voxel_response_total_counts_per_trial": response,
        "native_v1_activity_counts_per_trial": _existing_metric_pair(
            metrics,
            "v1_trailer_count_per_trial",
        ),
        "native_v1_q_active_fC_per_trial": _existing_metric_pair(
            metrics,
            "v1_trailer_q_active_fC_per_trial",
        ),
        "native_v1e_q_active_fC_per_trial": _existing_metric_pair(
            metrics,
            "v1e_trailer_q_active_fC_per_trial",
        ),
        "native_v1som_q_active_fC_per_trial": _existing_metric_pair(
            metrics,
            "v1som_trailer_q_active_fC_per_trial",
        ),
    }


def _full_argmax_decoder(
    channel_counts: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Any]:
    argmax = np.argmax(channel_counts, axis=1)
    correct = argmax == trailer_channel
    out: Dict[str, Any] = {
        "overall_actual_channel_accuracy": _safe_mean(correct.astype(np.float64)),
    }
    for condition, mask in _condition_masks(is_expected).items():
        out[condition] = {
            "actual_channel_accuracy": _safe_mean(correct[mask].astype(np.float64)),
            "n_trials": int(mask.sum()),
        }
    return out


def _voxel_argmax_decoder(
    voxel_counts: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Any]:
    actual_voxel = trailer_channel * N_VOXELS // N_CHANNELS
    pred_voxel = np.argmax(voxel_counts, axis=1)
    correct = pred_voxel == actual_voxel
    out: Dict[str, Any] = {
        "overall_actual_voxel_accuracy": _safe_mean(correct.astype(np.float64)),
    }
    for condition, mask in _condition_masks(is_expected).items():
        out[condition] = {
            "actual_voxel_accuracy": _safe_mean(correct[mask].astype(np.float64)),
            "n_trials": int(mask.sum()),
        }
    return out


def _nearest_centroid_leave_one_out(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    if features.ndim != 2:
        raise ValueError("features must be 2-D")
    labels = np.asarray(labels, dtype=np.int64)
    n = features.shape[0]
    if labels.shape != (n,):
        raise ValueError("labels shape mismatch")
    if n < 3 or np.unique(labels).size < 2:
        return {
            "accuracy": float("nan"),
            "n_trials": int(n),
            "n_classes": int(np.unique(labels).size),
            "skipped": "need at least 3 trials and 2 classes",
        }

    predictions = np.full(n, -1, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        train_labels = labels[train_mask]
        classes = np.unique(train_labels)
        if classes.size < 2:
            continue
        centroids = []
        centroid_classes = []
        for cls in classes:
            cls_mask = train_mask & (labels == cls)
            if cls_mask.any():
                centroids.append(features[cls_mask].mean(axis=0))
                centroid_classes.append(cls)
        if len(centroids) < 2:
            continue
        centroids_arr = np.vstack(centroids)
        dist = np.sum((centroids_arr - features[i]) ** 2, axis=1)
        predictions[i] = int(centroid_classes[int(np.argmin(dist))])
        valid[i] = True

    if not valid.any():
        return {
            "accuracy": float("nan"),
            "n_trials": int(n),
            "n_classes": int(np.unique(labels).size),
            "skipped": "no valid leave-one-out folds",
        }
    return {
        "accuracy": _safe_mean((predictions[valid] == labels[valid]).astype(float)),
        "n_trials": int(n),
        "n_valid_folds": int(valid.sum()),
        "n_classes": int(np.unique(labels).size),
    }


def _voxel_cross_validated_decoder(
    voxel_counts: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Any]:
    actual_voxel = trailer_channel * N_VOXELS // N_CHANNELS
    out: Dict[str, Any] = {}
    for condition, mask in _condition_masks(is_expected).items():
        out[condition] = _nearest_centroid_leave_one_out(
            voxel_counts[mask],
            actual_voxel[mask],
        )
    exp = out["expected"]["accuracy"]
    unexp = out["unexpected"]["accuracy"]
    out["expected_lower_than_unexpected"] = (
        bool(exp < unexp)
        if math.isfinite(float(exp)) and math.isfinite(float(unexp))
        else None
    )
    return out


def _fit_nearest_centroid(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    if features.ndim != 2:
        raise ValueError("features must be 2-D")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (features.shape[0],):
        raise ValueError("labels shape mismatch")
    classes = np.asarray(sorted(int(v) for v in np.unique(labels)), dtype=np.int64)
    if classes.size < 2:
        raise ValueError("nearest-centroid decoder needs at least two classes")
    centroids = np.zeros((classes.size, features.shape[1]), dtype=np.float64)
    counts = np.zeros(classes.size, dtype=np.int64)
    for i, cls in enumerate(classes):
        mask = labels == int(cls)
        counts[i] = int(mask.sum())
        if counts[i] == 0:
            raise ValueError(f"class {cls} has zero training examples")
        centroids[i] = features[mask].mean(axis=0)
    return {
        "classes": classes,
        "centroids": centroids,
        "class_counts": counts,
    }


def _predict_nearest_centroid_with_margin(
    model: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    centroids = np.asarray(model["centroids"], dtype=np.float64)
    classes = np.asarray(model["classes"], dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != centroids.shape[1]:
        raise ValueError("feature shape does not match nearest-centroid model")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels shape mismatch")
    dist = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    pred = classes[np.argmin(dist, axis=1)]
    class_to_index = {int(cls): i for i, cls in enumerate(classes)}
    try:
        true_index = np.asarray([class_to_index[int(label)] for label in labels])
    except KeyError as exc:
        raise ValueError(f"test label {exc} absent from centroid model") from exc
    true_dist = dist[np.arange(features.shape[0]), true_index]
    wrong_dist = dist.copy()
    wrong_dist[np.arange(features.shape[0]), true_index] = np.inf
    nearest_wrong_dist = np.min(wrong_dist, axis=1)
    return {
        "prediction": pred,
        "true_class_margin": nearest_wrong_dist - true_dist,
        "true_class_distance": true_dist,
        "nearest_wrong_distance": nearest_wrong_dist,
    }


def _accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(np.mean(np.asarray(pred, dtype=np.int64) == labels))


def _condition_metric_summary(
    pred: np.ndarray,
    labels: np.ndarray,
    margins: np.ndarray,
    idx: np.ndarray,
) -> Dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    return {
        "accuracy": _accuracy(pred[idx], labels[idx]),
        "true_class_margin_mean": _safe_mean(margins[idx]),
        "n_trials": int(idx.size),
    }


def _partitioned_unexpected_metrics(
    pred: np.ndarray,
    labels: np.ndarray,
    margins: np.ndarray,
    unexpected_idx: np.ndarray,
    expected_n: int,
    rng: np.random.Generator | None,
) -> Dict[str, Any]:
    if expected_n <= 0:
        raise ValueError("expected_n must be positive")
    idx = np.asarray(unexpected_idx, dtype=np.int64)
    if rng is not None:
        idx = rng.permutation(idx)
    subsets = [idx[start:start + expected_n] for start in range(0, idx.size, expected_n)]
    subset_metrics = [
        _condition_metric_summary(pred, labels, margins, subset)
        for subset in subsets
        if subset.size == expected_n
    ]
    if not subset_metrics:
        raise ValueError("no full unexpected subsets for matched accuracy")
    subset_acc = [float(metric["accuracy"]) for metric in subset_metrics]
    subset_margin = [
        float(metric["true_class_margin_mean"]) for metric in subset_metrics
    ]
    return {
        "accuracy": float(np.mean(subset_acc)),
        "true_class_margin_mean": float(np.mean(subset_margin)),
        "subset_accuracies": subset_acc,
        "subset_true_class_margin_means": subset_margin,
        "subset_size": int(expected_n),
        "n_subsets": int(len(subset_metrics)),
        "n_unexpected_trials": int(idx.size),
    }


def _transfer_condition_metrics(
    model: Mapping[str, Any],
    feedback_features: np.ndarray,
    feedback_labels: np.ndarray,
    feedback_is_expected: np.ndarray,
    rng: np.random.Generator | None,
) -> Dict[str, Any]:
    labels = np.asarray(feedback_labels, dtype=np.int64)
    is_expected = np.asarray(feedback_is_expected, dtype=bool)
    prediction = _predict_nearest_centroid_with_margin(
        model,
        feedback_features,
        labels,
    )
    pred = prediction["prediction"]
    margins = prediction["true_class_margin"]
    expected_idx = np.flatnonzero(is_expected)
    unexpected_idx = np.flatnonzero(~is_expected)
    expected = _condition_metric_summary(pred, labels, margins, expected_idx)
    unexpected = _partitioned_unexpected_metrics(
        pred,
        labels,
        margins,
        unexpected_idx,
        expected_n=int(expected_idx.size),
        rng=rng,
    )
    return {
        "expected_accuracy": expected["accuracy"],
        "unexpected_accuracy": unexpected["accuracy"],
        "unexpected_minus_expected": (
            unexpected["accuracy"] - expected["accuracy"]
        ),
        "expected_true_class_margin_mean": expected["true_class_margin_mean"],
        "unexpected_true_class_margin_mean": unexpected["true_class_margin_mean"],
        "unexpected_minus_expected_true_class_margin_mean": (
            unexpected["true_class_margin_mean"]
            - expected["true_class_margin_mean"]
        ),
        "expected_n_trials": int(expected_idx.size),
        "expected": expected,
        "unexpected_partitioned": unexpected,
    }


def _thin_after_voxel_pooling(
    rng: np.random.Generator,
    voxel_counts: np.ndarray,
    keep_p: float,
) -> np.ndarray:
    if keep_p < 0.0 or keep_p > 1.0:
        raise ValueError(f"keep_p must be in [0, 1], got {keep_p}")
    counts = np.rint(np.maximum(voxel_counts, 0.0)).astype(np.int64)
    return rng.binomial(counts, keep_p).astype(np.float64)


def _mean_std(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(arr.size)}


def _bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "alpha": float(alpha),
            "low": float("nan"),
            "high": float("nan"),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        }
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    means = arr[sample_idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "alpha": float(alpha),
        "low": float(low),
        "high": float(high),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _mean_std_with_ci(values: Sequence[float]) -> Dict[str, Any]:
    out = _mean_std(values)
    out["bootstrap_mean_ci_95"] = _bootstrap_mean_ci(values)
    return out


def _summarize_decoder_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected = [float(run["expected_accuracy"]) for run in runs]
    unexpected = [float(run["unexpected_accuracy"]) for run in runs]
    delta = [float(run["unexpected_minus_expected"]) for run in runs]
    expected_margin = [
        float(run["expected_true_class_margin_mean"]) for run in runs
    ]
    unexpected_margin = [
        float(run["unexpected_true_class_margin_mean"]) for run in runs
    ]
    delta_margin = [
        float(run["unexpected_minus_expected_true_class_margin_mean"])
        for run in runs
    ]
    return {
        "expected_accuracy": _mean_std(expected),
        "unexpected_accuracy": _mean_std(unexpected),
        "unexpected_minus_expected_accuracy": _mean_std_with_ci(delta),
        "expected_true_class_margin_mean": _mean_std(expected_margin),
        "unexpected_true_class_margin_mean": _mean_std(unexpected_margin),
        "unexpected_minus_expected_true_class_margin_mean": (
            _mean_std_with_ci(delta_margin)
        ),
        "expected_lower_than_unexpected_fraction": float(
            np.mean(np.asarray(delta, dtype=np.float64) > 0.0)
        ),
        "expected_lower_margin_than_unexpected_fraction": float(
            np.mean(np.asarray(delta_margin, dtype=np.float64) > 0.0)
        ),
    }


def _label_counts(labels: np.ndarray) -> Dict[str, int]:
    return {
        str(int(label)): int(np.sum(labels == int(label)))
        for label in sorted(np.unique(labels))
    }


def _label_counts_by_condition(
    labels: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Dict[str, int]]:
    is_expected = np.asarray(is_expected, dtype=bool)
    return {
        "expected": _label_counts(labels[is_expected]),
        "unexpected": _label_counts(labels[~is_expected]),
    }


def _shuffle_test_labels_within_condition(
    labels: np.ndarray,
    is_expected: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.asarray(labels, dtype=np.int64).copy()
    is_expected = np.asarray(is_expected, dtype=bool)
    for mask in (is_expected, ~is_expected):
        idx = np.flatnonzero(mask)
        shuffled[idx] = rng.permutation(shuffled[idx])
    return shuffled


def _six_class_transfer_decoder_pair(
    feedback_name: str,
    feedback_path: Path,
    localizer_name: str,
    localizer_path: Path,
    keep_ps: Sequence[float],
    n_seeds: int,
) -> Dict[str, Any]:
    feedback = _load_native_artifact(feedback_path)
    localizer = _load_native_artifact(localizer_path)
    voxel_bins = _voxel_spatial_bins()
    localizer_voxels = _pool_voxels(localizer["channel_counts"], voxel_bins)
    feedback_voxels = _pool_voxels(feedback["channel_counts"], voxel_bins)
    localizer_labels = np.asarray(localizer["trailer_channel"], dtype=np.int64)
    feedback_labels = np.asarray(feedback["trailer_channel"], dtype=np.int64)
    feedback_expected = np.asarray(feedback["is_expected"], dtype=bool)

    observed_labels = tuple(int(v) for v in sorted(np.unique(localizer_labels)))
    model = _fit_nearest_centroid(localizer_voxels, localizer_labels)
    baseline = _transfer_condition_metrics(
        model,
        feedback_voxels,
        feedback_labels,
        feedback_expected,
        rng=None,
    )
    baseline_localizer_shuffle_runs = []
    baseline_test_shuffle_runs = []
    seed_values = list(range(int(n_seeds)))
    for seed in seed_values:
        localizer_shuffle_rng = np.random.default_rng(1_000_000 + seed)
        test_shuffle_rng = np.random.default_rng(2_000_000 + seed)
        partition_rng = np.random.default_rng(3_000_000 + seed)
        shuffled_localizer_labels = localizer_shuffle_rng.permutation(
            localizer_labels,
        )
        shuffled_model = _fit_nearest_centroid(
            localizer_voxels,
            shuffled_localizer_labels,
        )
        baseline_localizer_shuffle_runs.append(
            _transfer_condition_metrics(
                shuffled_model,
                feedback_voxels,
                feedback_labels,
                feedback_expected,
                rng=partition_rng,
            )
        )
        shuffled_feedback_labels = _shuffle_test_labels_within_condition(
            feedback_labels,
            feedback_expected,
            test_shuffle_rng,
        )
        baseline_test_shuffle_runs.append(
            _transfer_condition_metrics(
                model,
                feedback_voxels,
                shuffled_feedback_labels,
                feedback_expected,
                rng=np.random.default_rng(3_000_000 + seed),
            )
        )

    noisy: Dict[str, Any] = {}
    for keep_p in keep_ps:
        runs = []
        localizer_shuffle_runs = []
        test_shuffle_runs = []
        for seed in seed_values:
            thinning_rng = np.random.default_rng(seed)
            localizer_thin = _thin_after_voxel_pooling(
                thinning_rng,
                localizer_voxels,
                keep_p,
            )
            feedback_thin = _thin_after_voxel_pooling(
                thinning_rng,
                feedback_voxels,
                keep_p,
            )
            run_model = _fit_nearest_centroid(localizer_thin, localizer_labels)
            runs.append(
                _transfer_condition_metrics(
                    run_model,
                    feedback_thin,
                    feedback_labels,
                    feedback_expected,
                    rng=np.random.default_rng(3_000_000 + seed),
                )
            )

            localizer_shuffle_rng = np.random.default_rng(1_000_000 + seed)
            shuffled_localizer_labels = localizer_shuffle_rng.permutation(
                localizer_labels,
            )
            shuffled_model = _fit_nearest_centroid(
                localizer_thin,
                shuffled_localizer_labels,
            )
            localizer_shuffle_runs.append(
                _transfer_condition_metrics(
                    shuffled_model,
                    feedback_thin,
                    feedback_labels,
                    feedback_expected,
                    rng=np.random.default_rng(3_000_000 + seed),
                )
            )

            test_shuffle_rng = np.random.default_rng(2_000_000 + seed)
            shuffled_feedback_labels = _shuffle_test_labels_within_condition(
                feedback_labels,
                feedback_expected,
                test_shuffle_rng,
            )
            test_shuffle_runs.append(
                _transfer_condition_metrics(
                    run_model,
                    feedback_thin,
                    shuffled_feedback_labels,
                    feedback_expected,
                    rng=np.random.default_rng(3_000_000 + seed),
                )
            )
        noisy[str(float(keep_p))] = {
            "keep_p": float(keep_p),
            "n_seeds": int(n_seeds),
            "seed_values": seed_values,
            "quantization_step_after_voxel_pooling_counts": 1.0,
            "summary": _summarize_decoder_runs(runs),
            "chance_controls": {
                "shuffled_localizer_labels": _summarize_decoder_runs(
                    localizer_shuffle_runs,
                ),
                "shuffled_test_labels_conditionwise": _summarize_decoder_runs(
                    test_shuffle_runs,
                ),
            },
        }

    energy_activity = _expected_unexpected_summary(
        feedback["artifact"],
        feedback_voxels,
        feedback_expected,
    )
    return {
        "feedback_artifact": feedback_name,
        "feedback_path": str(feedback_path),
        "localizer_artifact": localizer_name,
        "localizer_path": str(localizer_path),
        "decoder": "pure_numpy_nearest_centroid_transfer",
        "measurement_degradation": (
            "Independent binomial thinning is applied after 12-channel to "
            "4-pseudo-voxel pooling for both localizer and feedback responses."
        ),
        "quantization": {
            "pre_thinning_voxel_counts_quantization_step": 1.0,
            "post_thinning_voxel_counts_quantization_step": 1.0,
            "unit": "retained V1_E spike counts per pooled pseudo-voxel",
        },
        "labels": {
            "required_observed_labels": list(EXPECTED_TRAILER_LABELS),
            "localizer_observed_labels": list(observed_labels),
            "matches_required_observed_labels": (
                observed_labels == EXPECTED_TRAILER_LABELS
            ),
            "feedback_observed_labels": [
                int(v) for v in sorted(np.unique(feedback_labels))
            ],
        },
        "trial_counts": {
            "localizer_total": int(localizer_voxels.shape[0]),
            "localizer_by_label": _label_counts(localizer_labels),
            "feedback_total": int(feedback_voxels.shape[0]),
            "feedback_expected_total": int(feedback_expected.sum()),
            "feedback_unexpected_total": int((~feedback_expected).sum()),
            "feedback_by_condition_label": _label_counts_by_condition(
                feedback_labels,
                feedback_expected,
            ),
            "unexpected_partition_size": int(feedback_expected.sum()),
            "unexpected_n_full_partitions": int(
                np.sum(~feedback_expected) // max(int(feedback_expected.sum()), 1)
            ),
        },
        "training": {
            "n_localizer_trials": int(localizer_voxels.shape[0]),
            "class_counts": {
                str(int(cls)): int(count)
                for cls, count in zip(model["classes"], model["class_counts"])
            },
            "centroids_no_noise": model["centroids"],
        },
        "test": {
            "n_feedback_expected_trials": int(feedback_expected.sum()),
            "n_feedback_unexpected_trials": int((~feedback_expected).sum()),
            "unexpected_partition_size": int(feedback_expected.sum()),
            "unexpected_partitioning": (
                "Unexpected trials are partitioned into 24-trial subsets "
                "per seed and subset accuracies are averaged."
            ),
        },
        "no_noise_baseline": baseline,
        "no_noise_chance_controls": {
            "n_label_shuffle_seeds": int(n_seeds),
            "shuffled_localizer_labels": _summarize_decoder_runs(
                baseline_localizer_shuffle_runs,
            ),
            "shuffled_test_labels_conditionwise": _summarize_decoder_runs(
                baseline_test_shuffle_runs,
            ),
        },
        "binomial_thinning": noisy,
        "energy_activity_from_feedback_artifact": energy_activity,
    }


def _final_transfer_criterion(pair: Mapping[str, Any]) -> Dict[str, Any]:
    ea = pair["energy_activity_from_feedback_artifact"]
    q_pair = ea["native_v1_q_active_fC_per_trial"]
    activity_pair = ea["native_v1_activity_counts_per_trial"]
    q_expected_lower = bool(q_pair["expected"] < q_pair["unexpected"])
    activity_expected_lower = bool(activity_pair["expected"] < activity_pair["unexpected"])
    noisy = pair["binomial_thinning"]
    decoder_expected_lower_by_keep_p = {
        keep_p: bool(
            metrics["summary"]["unexpected_minus_expected_accuracy"]["mean"] > 0.0
        )
        for keep_p, metrics in noisy.items()
    }
    decoder_delta_by_keep_p = {
        keep_p: metrics["summary"]["unexpected_minus_expected_accuracy"]
        for keep_p, metrics in noisy.items()
    }
    margin_expected_lower_by_keep_p = {
        keep_p: bool(
            metrics["summary"][
                "unexpected_minus_expected_true_class_margin_mean"
            ]["mean"] > 0.0
        )
        for keep_p, metrics in noisy.items()
    }
    margin_delta_by_keep_p = {
        keep_p: metrics["summary"][
            "unexpected_minus_expected_true_class_margin_mean"
        ]
        for keep_p, metrics in noisy.items()
    }
    accuracy_ci_excludes_zero_by_keep_p = {
        keep_p: bool(
            metrics["summary"]["unexpected_minus_expected_accuracy"][
                "bootstrap_mean_ci_95"
            ]["low"] > 0.0
        )
        for keep_p, metrics in noisy.items()
    }
    margin_ci_excludes_zero_by_keep_p = {
        keep_p: bool(
            metrics["summary"][
                "unexpected_minus_expected_true_class_margin_mean"
            ]["bootstrap_mean_ci_95"]["low"] > 0.0
        )
        for keep_p, metrics in noisy.items()
    }
    return {
        "q_active_expected_lower_than_unexpected": q_expected_lower,
        "activity_expected_lower_than_unexpected": activity_expected_lower,
        "noisy_decoder_expected_lower_than_unexpected_by_keep_p": (
            decoder_expected_lower_by_keep_p
        ),
        "noisy_decoder_delta_unexpected_minus_expected_by_keep_p": (
            decoder_delta_by_keep_p
        ),
        "noisy_margin_expected_lower_than_unexpected_by_keep_p": (
            margin_expected_lower_by_keep_p
        ),
        "noisy_margin_delta_unexpected_minus_expected_by_keep_p": (
            margin_delta_by_keep_p
        ),
        "accuracy_bootstrap_ci_excludes_zero_by_keep_p": (
            accuracy_ci_excludes_zero_by_keep_p
        ),
        "margin_bootstrap_ci_excludes_zero_by_keep_p": (
            margin_ci_excludes_zero_by_keep_p
        ),
        "satisfies_all_requested_keep_p": bool(
            q_expected_lower
            and activity_expected_lower
            and all(decoder_expected_lower_by_keep_p.values())
        ),
        "satisfies_all_requested_keep_p_with_accuracy_ci": bool(
            q_expected_lower
            and activity_expected_lower
            and all(accuracy_ci_excludes_zero_by_keep_p.values())
        ),
        "satisfies_all_requested_keep_p_with_accuracy_and_margin_ci": bool(
            q_expected_lower
            and activity_expected_lower
            and all(accuracy_ci_excludes_zero_by_keep_p.values())
            and all(margin_ci_excludes_zero_by_keep_p.values())
        ),
        "satisfies_any_requested_keep_p": bool(
            q_expected_lower
            and activity_expected_lower
            and any(decoder_expected_lower_by_keep_p.values())
        ),
    }


def _counts_by_theta(
    channel_counts: np.ndarray,
    trailer_channel: np.ndarray,
    condition_mask: np.ndarray,
    theta_channels: Sequence[int],
) -> np.ndarray:
    by_theta = np.full((N_CHANNELS, len(theta_channels)), np.nan, dtype=np.float64)
    for ti, channel in enumerate(theta_channels):
        mask = condition_mask & (trailer_channel == int(channel))
        if mask.any():
            by_theta[:, ti] = channel_counts[mask].mean(axis=0)
    return by_theta


def _pool_tuning_by_voxel(
    counts_by_theta: np.ndarray,
    voxel_bins: np.ndarray,
) -> np.ndarray:
    tuning = np.zeros((N_VOXELS, counts_by_theta.shape[1]), dtype=np.float64)
    for voxel in range(N_VOXELS):
        tuning[voxel] = np.nanmean(counts_by_theta[voxel_bins == voxel], axis=0)
    return tuning


def _model_family_comparison(
    channel_counts: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
    voxel_bins: np.ndarray,
) -> Dict[str, Any]:
    theta_channels = sorted(int(ch) for ch in np.unique(trailer_channel))
    masks = _condition_masks(is_expected)
    expected_by_theta = _counts_by_theta(
        channel_counts,
        trailer_channel,
        masks["expected"],
        theta_channels,
    )
    unexpected_by_theta = _counts_by_theta(
        channel_counts,
        trailer_channel,
        masks["unexpected"],
        theta_channels,
    )

    valid_theta = ~(np.isnan(expected_by_theta).any(axis=0) |
                    np.isnan(unexpected_by_theta).any(axis=0))
    theta_channels = [ch for ch, ok in zip(theta_channels, valid_theta) if ok]
    expected_by_theta = expected_by_theta[:, valid_theta]
    unexpected_by_theta = unexpected_by_theta[:, valid_theta]

    if expected_by_theta.shape[1] == 0:
        return {
            "skipped": "no trailer channels have both expected and unexpected trials",
        }

    thetas = np.asarray(theta_channels, dtype=np.float64) * (np.pi / N_CHANNELS)
    observed_expected = _pool_tuning_by_voxel(expected_by_theta, voxel_bins)
    observed_unexpected = _pool_tuning_by_voxel(unexpected_by_theta, voxel_bins)

    comparisons: Dict[str, Any] = {}
    n_obs = int(observed_expected.size)
    for family in _VOXEL_MODEL_FAMILIES:
        model = pseudo_voxel_forward_model(
            unexpected_by_theta,
            voxel_bins,
            family,
            thetas,
            effect_size=0.2,
        )
        predicted = np.asarray(model["voxel_tuning_predicted"], dtype=np.float64)
        residual = observed_expected - predicted
        sse = float(np.sum(residual ** 2))
        aic = float(n_obs * math.log(max(sse / max(n_obs, 1), 1e-12)) + 2.0)
        comparisons[family] = {
            "sse_vs_observed_expected": sse,
            "aic_like": aic,
            "preferred_voxel": int(model["preferred_voxel"]),
            "effect_size": float(model["effect_size"]),
            "predicted_expected_voxel_tuning": predicted,
        }

    baseline_residual = observed_expected - observed_unexpected
    baseline_sse = float(np.sum(baseline_residual ** 2))
    baseline_aic = float(
        n_obs * math.log(max(baseline_sse / max(n_obs, 1), 1e-12)) + 2.0
    )
    best_family = min(
        comparisons,
        key=lambda family: comparisons[family]["aic_like"],
    )
    return {
        "compatibility_note": (
            "Native artifacts expose 12 channel-aggregate counts, so each "
            "channel is treated as one pseudo-cell before calling "
            "pseudo_voxel_forward_model."
        ),
        "theta_channels": theta_channels,
        "theta_radians": thetas,
        "observed_expected_voxel_tuning": observed_expected,
        "observed_unexpected_voxel_tuning": observed_unexpected,
        "baseline_unexpected_predicts_expected": {
            "sse_vs_observed_expected": baseline_sse,
            "aic_like": baseline_aic,
        },
        "families": comparisons,
        "best_family_by_aic_like": best_family,
    }


def analyze_artifact(name: str, path: Path) -> Dict[str, Any]:
    loaded = _load_native_artifact(path)
    artifact = loaded["artifact"]
    channel_counts = loaded["channel_counts"]
    channel_counts_100ms = loaded["channel_counts_100ms"]
    trailer_channel = loaded["trailer_channel"]
    is_expected = loaded["is_expected"]
    voxel_bins = _voxel_spatial_bins()
    voxel_counts = _pool_voxels(channel_counts, voxel_bins)
    voxel_counts_100ms = _pool_voxels_100ms(channel_counts_100ms, voxel_bins)

    voxel_argmax = _voxel_argmax_decoder(voxel_counts, trailer_channel, is_expected)
    voxel_cv = _voxel_cross_validated_decoder(
        voxel_counts,
        trailer_channel,
        is_expected,
    )
    full_argmax = _full_argmax_decoder(channel_counts, trailer_channel, is_expected)

    return {
        "name": name,
        "path": str(path),
        "n_trials": int(channel_counts.shape[0]),
        "n_expected_trials": int(is_expected.astype(bool).sum()),
        "n_unexpected_trials": int((~is_expected.astype(bool)).sum()),
        "channel_to_voxel": [int(v) for v in voxel_bins.tolist()],
        "voxel_response_means_by_condition_channel_voxel": (
            _means_by_condition_channel_voxel(
                voxel_counts,
                trailer_channel,
                is_expected,
            )
        ),
        "voxel_response_means_100ms_by_condition_channel_voxel": (
            _means_by_condition_bin_channel_voxel(
                voxel_counts_100ms,
                trailer_channel,
                is_expected,
            )
        ),
        "expected_vs_unexpected_mean_response_energy_activity": (
            _expected_unexpected_summary(artifact, voxel_counts, is_expected)
        ),
        "model_family_fit_comparison": _model_family_comparison(
            channel_counts,
            trailer_channel,
            is_expected,
            voxel_bins,
        ),
        "voxel_argmax_decoder": voxel_argmax,
        "voxel_cross_validated_nearest_centroid_decoder": voxel_cv,
        "full_argmax_decoder": full_argmax,
        "readout_expected_lower_decoding_than_unexpected": {
            "voxel_cross_validated": voxel_cv["expected_lower_than_unexpected"],
            "voxel_argmax": (
                voxel_argmax["expected"]["actual_voxel_accuracy"]
                < voxel_argmax["unexpected"]["actual_voxel_accuracy"]
            ),
            "full_argmax": (
                full_argmax["expected"]["actual_channel_accuracy"]
                < full_argmax["unexpected"]["actual_channel_accuracy"]
            ),
        },
    }


def _artifact_paths(stage3_dir: Path) -> Dict[str, Path]:
    return {name: stage3_dir / rel for name, rel in DEFAULT_ARTIFACTS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the four fixed native Richter artifacts with a "
            "pseudo-voxel fMRI-like readout."
        ),
    )
    parser.add_argument(
        "--stage3-dir",
        type=Path,
        default=DEFAULT_STAGE3_DIR,
        help="Directory containing the stage3 native Richter artifacts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--keep-p",
        type=float,
        nargs="*",
        default=list(DEFAULT_THINNING_KEEP_P),
        help=(
            "Binomial keep probabilities for corrected 6-class transfer "
            "decoder measurement degradation."
        ),
    )
    parser.add_argument(
        "--n-thinning-seeds",
        type=int,
        default=DEFAULT_N_THINNING_SEEDS,
        help="Number of deterministic thinning seeds per keep probability.",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        choices=("sigma22", "sigma16"),
        default=["sigma22", "sigma16"],
        help="Artifact pairs to run for the corrected transfer decoder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_thinning_seeds <= 0:
        raise ValueError("--n-thinning-seeds must be positive")
    if not args.keep_p:
        raise ValueError("--keep-p requires at least one value")
    paths = _artifact_paths(args.stage3_dir)
    selected_pairs = list(dict.fromkeys(args.pairs))
    if "sigma22" not in selected_pairs:
        selected_pairs.insert(0, "sigma22")
    required_artifacts = {"feedback_on_sigma22", "sensory_only_sigma22"}
    if "sigma16" in selected_pairs:
        required_artifacts.update({"feedback_on_sigma16", "sensory_only_sigma16"})
    missing = [
        str(paths[name]) for name in sorted(required_artifacts)
        if not paths[name].exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    analyses = {
        name: analyze_artifact(name, path)
        for name, path in paths.items()
        if name in required_artifacts
    }
    transfer_decoders: Dict[str, Any] = {}
    if "sigma22" in selected_pairs:
        transfer_decoders["sigma22"] = _six_class_transfer_decoder_pair(
            "feedback_on_sigma22",
            paths["feedback_on_sigma22"],
            "sensory_only_sigma22",
            paths["sensory_only_sigma22"],
            keep_ps=args.keep_p,
            n_seeds=args.n_thinning_seeds,
        )
    if "sigma16" in selected_pairs:
        transfer_decoders["sigma16"] = _six_class_transfer_decoder_pair(
            "feedback_on_sigma16",
            paths["feedback_on_sigma16"],
            "sensory_only_sigma16",
            paths["sensory_only_sigma16"],
            keep_ps=args.keep_p,
            n_seeds=args.n_thinning_seeds,
        )
    feedback_sigma22 = analyses["feedback_on_sigma22"]
    corrected_sigma22_criterion = _final_transfer_criterion(
        transfer_decoders["sigma22"],
    )
    summary = {
        "analysis_kind": "native_richter_pseudovoxel_fmri_like_readout",
        "stage3_dir": str(args.stage3_dir),
        "artifacts_analyzed": {
            name: str(paths[name]) for name in sorted(required_artifacts)
        },
        "voxel_binning": {
            "n_channels": N_CHANNELS,
            "n_voxels": N_VOXELS,
            "channel_to_voxel": _voxel_spatial_bins().tolist(),
            "matches": "richter_crossover._voxel_spatial_bins formula",
        },
        "native_forward_model_wrapper": (
            "Uses 12 native channel aggregates as pseudo-cells because native "
            "JSON artifacts do not contain per-cell V1_E spike counts."
        ),
        "corrected_transfer_decoder": {
            "class_labels": list(EXPECTED_TRAILER_LABELS),
            "keep_p_values": [float(v) for v in args.keep_p],
            "n_thinning_seeds": int(args.n_thinning_seeds),
            "selected_pairs": selected_pairs,
            "pairs": transfer_decoders,
            "feedback_on_sigma22_final_criterion": corrected_sigma22_criterion,
        },
        "analyses": analyses,
        "feedback_on_sigma22_verdict": {
            "energy_activity_passing_artifact": "feedback_on_sigma22",
            "voxel_cross_validated_expected_lower_than_unexpected": (
                feedback_sigma22[
                    "readout_expected_lower_decoding_than_unexpected"
                ]["voxel_cross_validated"]
            ),
            "voxel_argmax_expected_lower_than_unexpected": (
                feedback_sigma22[
                    "readout_expected_lower_decoding_than_unexpected"
                ]["voxel_argmax"]
            ),
            "full_argmax_expected_lower_than_unexpected": (
                feedback_sigma22[
                    "readout_expected_lower_decoding_than_unexpected"
                ]["full_argmax"]
            ),
            "voxel_cross_validated_accuracy_expected": (
                feedback_sigma22[
                    "voxel_cross_validated_nearest_centroid_decoder"
                ]["expected"]["accuracy"]
            ),
            "voxel_cross_validated_accuracy_unexpected": (
                feedback_sigma22[
                    "voxel_cross_validated_nearest_centroid_decoder"
                ]["unexpected"]["accuracy"]
            ),
            "voxel_argmax_accuracy_expected": (
                feedback_sigma22["voxel_argmax_decoder"]["expected"][
                    "actual_voxel_accuracy"
                ]
            ),
            "voxel_argmax_accuracy_unexpected": (
                feedback_sigma22["voxel_argmax_decoder"]["unexpected"][
                    "actual_voxel_accuracy"
                ]
            ),
            "full_argmax_accuracy_expected": (
                feedback_sigma22["full_argmax_decoder"]["expected"][
                    "actual_channel_accuracy"
                ]
            ),
            "full_argmax_accuracy_unexpected": (
                feedback_sigma22["full_argmax_decoder"]["unexpected"][
                    "actual_channel_accuracy"
                ]
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2, sort_keys=True)
        f.write("\n")

    verdict = summary["feedback_on_sigma22_verdict"]
    print(json.dumps(_jsonify({
        "out": str(args.out),
        "corrected_transfer_decoder_feedback_on_sigma22": (
            summary["corrected_transfer_decoder"][
                "feedback_on_sigma22_final_criterion"
            ]
        ),
        "feedback_on_sigma22_verdict": verdict,
    }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
