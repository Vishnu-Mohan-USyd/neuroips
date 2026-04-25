#!/usr/bin/env python3
"""Analyze standalone native sensory-stage diagnostics.

This script consumes the JSON artifact emitted by
`expectation_snn_native sensory-diagnostics` and writes a compact analysis JSON
with:

- architecture summary copied from the native artifact
- orientation tuning matrices across the 12 wrapped stimulus channels
- simple geometry statistics (top channel, second-best channel, margin,
  nonzero-channel count)
- leave-one-repeat-out nearest-centroid separability/confusion diagnostics for
  V1_E, V1_SOM, and the concatenated V1_E+V1_SOM response vectors
- the same separability metric under additive Gaussian measurement noise

Assumptions:
- trailer counts are measured over a 500 ms window
- V1_E uses 16 cells/channel
- V1_SOM uses 4 cells/channel
- every repeat contains exactly one trial per stimulus channel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

TRAILER_WINDOW_SECONDS = 0.5
V1_E_CELLS_PER_CHANNEL = 16.0
V1_SOM_CELLS_PER_CHANNEL = 4.0
NOISE_SIGMA_HZ = 8.0
NOISE_REPEATS = 50
NOISE_SEED = 12345


def _mean_tuning_by_stimulus(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_stimuli = int(labels.max()) + 1
    out = np.zeros((n_stimuli, vectors.shape[1]), dtype=np.float64)
    for stimulus in range(n_stimuli):
        out[stimulus] = vectors[labels == stimulus].mean(axis=0)
    return out


def _geometry_stats(vectors: np.ndarray) -> dict[str, Any]:
    top = vectors.max(axis=1)
    second = np.partition(vectors, -2, axis=1)[:, -2]
    margin = top - second
    nonzero = (vectors > 1e-9).sum(axis=1)
    return {
        "top_mean_hz": float(top.mean()),
        "second_mean_hz": float(second.mean()),
        "margin_mean_hz": float(margin.mean()),
        "nonzero_channel_histogram": {
            str(int(value)): int((nonzero == value).sum()) for value in np.unique(nonzero)
        },
    }


def _pairwise_distance_matrix(mean_vectors: np.ndarray) -> list[list[float]]:
    diff = mean_vectors[:, None, :] - mean_vectors[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist.tolist()


def _nearest_centroid_confusion(
    vectors: np.ndarray,
    labels: np.ndarray,
    repeats: np.ndarray,
    noise_sigma_hz: float = 0.0,
    noise_repeats: int = 1,
    noise_seed: int = NOISE_SEED,
) -> dict[str, Any]:
    unique_labels = np.unique(labels)
    unique_repeats = np.unique(repeats)
    confusion = np.zeros((unique_labels.size, unique_labels.size), dtype=np.float64)
    accuracies: list[float] = []
    rng = np.random.default_rng(noise_seed)

    for rep in unique_repeats:
        train_mask = repeats != rep
        test_mask = repeats == rep
        x_train = vectors[train_mask]
        y_train = labels[train_mask]
        x_test = vectors[test_mask]
        y_test = labels[test_mask]
        centroids = np.stack(
            [x_train[y_train == label].mean(axis=0) for label in unique_labels],
            axis=0,
        )
        for _ in range(noise_repeats):
            if noise_sigma_hz > 0.0:
                x_eval = np.clip(
                    x_test + rng.normal(0.0, noise_sigma_hz, size=x_test.shape),
                    0.0,
                    None,
                )
            else:
                x_eval = x_test
            distances = np.sum((x_eval[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            pred_idx = np.argmin(distances, axis=1)
            pred = unique_labels[pred_idx]
            accuracies.append(float(np.mean(pred == y_test)))
            for truth, guess in zip(y_test, pred, strict=True):
                confusion[int(truth), int(guess)] += 1.0

    if confusion.sum() > 0.0:
        confusion /= confusion.sum(axis=1, keepdims=True)
    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "confusion_matrix": confusion.tolist(),
        "noise_sigma_hz": float(noise_sigma_hz),
        "noise_repeats": int(noise_repeats),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text())
    trial_data = artifact["trial_data"]

    stimulus_channel = np.asarray(trial_data["stimulus_channel"], dtype=np.int32)
    repeat = np.asarray(trial_data["rep"], dtype=np.int32)
    v1_e_counts = np.asarray(trial_data["v1_e_trailer_channel_counts"], dtype=np.float64)
    v1_som_counts = np.asarray(
        trial_data["v1_som_trailer_channel_counts"], dtype=np.float64
    )
    hctx_counts = np.asarray(
        trial_data["hctx_e_trailer_total_counts"], dtype=np.float64
    )
    hpred_counts = np.asarray(
        trial_data["hpred_e_trailer_total_counts"], dtype=np.float64
    )

    v1_e_rates = v1_e_counts / (V1_E_CELLS_PER_CHANNEL * TRAILER_WINDOW_SECONDS)
    v1_som_rates = v1_som_counts / (V1_SOM_CELLS_PER_CHANNEL * TRAILER_WINDOW_SECONDS)
    hctx_rates = hctx_counts / (192.0 * TRAILER_WINDOW_SECONDS)
    hpred_rates = hpred_counts / (192.0 * TRAILER_WINDOW_SECONDS)

    v1_e_mean = _mean_tuning_by_stimulus(v1_e_rates, stimulus_channel)
    v1_som_mean = _mean_tuning_by_stimulus(v1_som_rates, stimulus_channel)
    hctx_mean = _mean_tuning_by_stimulus(hctx_rates[:, None], stimulus_channel)[:, 0]
    hpred_mean = _mean_tuning_by_stimulus(hpred_rates[:, None], stimulus_channel)[:, 0]

    combined = np.concatenate([v1_e_rates, v1_som_rates], axis=1)

    analysis = {
        "schema_version": 1,
        "artifact_kind": "native_sensory_stage_analysis",
        "input_artifact_path": str(args.input),
        "input_content_hash_fnv1a64": artifact["content_hash_fnv1a64"],
        "architecture": artifact["architecture"],
        "stimulus": artifact["stimulus"],
        "tuning_curves": {
            "v1_e_channel_rates_hz_by_stimulus": v1_e_mean.tolist(),
            "v1_som_channel_rates_hz_by_stimulus": v1_som_mean.tolist(),
            "hctx_e_trailer_population_rate_hz_by_stimulus": hctx_mean.tolist(),
            "hpred_e_trailer_population_rate_hz_by_stimulus": hpred_mean.tolist(),
        },
        "geometry": {
            "v1_e": _geometry_stats(v1_e_mean),
            "v1_som": _geometry_stats(v1_som_mean),
        },
        "pairwise_distances": {
            "v1_e_euclidean": _pairwise_distance_matrix(v1_e_mean),
            "v1_som_euclidean": _pairwise_distance_matrix(v1_som_mean),
            "v1_e_plus_v1_som_euclidean": _pairwise_distance_matrix(
                np.concatenate([v1_e_mean, v1_som_mean], axis=1)
            ),
        },
        "separability": {
            "v1_e_clean": _nearest_centroid_confusion(
                v1_e_rates, stimulus_channel, repeat, noise_sigma_hz=0.0
            ),
            "v1_e_noisy": _nearest_centroid_confusion(
                v1_e_rates,
                stimulus_channel,
                repeat,
                noise_sigma_hz=NOISE_SIGMA_HZ,
                noise_repeats=NOISE_REPEATS,
            ),
            "v1_som_clean": _nearest_centroid_confusion(
                v1_som_rates, stimulus_channel, repeat, noise_sigma_hz=0.0
            ),
            "v1_som_noisy": _nearest_centroid_confusion(
                v1_som_rates,
                stimulus_channel,
                repeat,
                noise_sigma_hz=NOISE_SIGMA_HZ,
                noise_repeats=NOISE_REPEATS,
            ),
            "v1_e_plus_v1_som_clean": _nearest_centroid_confusion(
                combined, stimulus_channel, repeat, noise_sigma_hz=0.0
            ),
            "v1_e_plus_v1_som_noisy": _nearest_centroid_confusion(
                combined,
                stimulus_channel,
                repeat,
                noise_sigma_hz=NOISE_SIGMA_HZ,
                noise_repeats=NOISE_REPEATS,
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2))

    print("command=analyze-native-sensory-diagnostics")
    print("status=PASS")
    print(f"input_path={args.input}")
    print(f"output_path={args.out}")
    print(f"v1_e_clean_accuracy={analysis['separability']['v1_e_clean']['accuracy_mean']:.6f}")
    print(f"v1_e_noisy_accuracy={analysis['separability']['v1_e_noisy']['accuracy_mean']:.6f}")
    print(f"v1_som_clean_accuracy={analysis['separability']['v1_som_clean']['accuracy_mean']:.6f}")
    print(f"v1_som_noisy_accuracy={analysis['separability']['v1_som_noisy']['accuracy_mean']:.6f}")
    print(
        "v1_e_plus_v1_som_noisy_accuracy="
        f"{analysis['separability']['v1_e_plus_v1_som_noisy']['accuracy_mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
