#!/usr/bin/env python3
"""Independent held-out-repeat decoder for L2/3 natural-video site rates.

This script intentionally does not import ``validate_full_plasticity.py``. It is
meant as an audit path for the L2/3 video decoding numbers reported by the main
validator.

Input CSV schema:
    repeat_index,frame_index,population,site_id,rate_hz

Method:
    For each held-out repeat, average the other repeats frame-by-frame to form
    one template vector per frame. Score held-out L2/3E site vectors against all
    frame templates with cosine similarity, then report top-1, top-5, mean rank,
    and same-vs-different cosine similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_population_site_rates(path: Path, population: str) -> tuple[np.ndarray, list[int], list[int]]:
    """Load site-rate vectors as ``[repeat, frame, site]``.

    Returns:
        rates: Float array with shape ``[n_repeats, n_frames, n_sites]``.
        repeats: Sorted repeat indices represented in the CSV.
        frames: Sorted frame indices represented in the CSV.
    """
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
            repeat = int(row["repeat_index"])
            frame = int(row["frame_index"])
            site = int(row["site_id"])
            rate = float(row["rate_hz"])
            rows.append((repeat, frame, site, rate))
            repeat_set.add(repeat)
            frame_set.add(frame)
            site_set.add(site)

    if not rows:
        raise ValueError(f"{path} contains no rows for population={population!r}")

    repeats = sorted(repeat_set)
    frames = sorted(frame_set)
    sites = sorted(site_set)
    repeat_to_i = {value: idx for idx, value in enumerate(repeats)}
    frame_to_i = {value: idx for idx, value in enumerate(frames)}
    site_to_i = {value: idx for idx, value in enumerate(sites)}

    rates = np.zeros((len(repeats), len(frames), len(sites)), dtype=np.float64)
    for repeat, frame, site, rate in rows:
        rates[repeat_to_i[repeat], frame_to_i[frame], site_to_i[site]] = rate

    return rates, repeats, frames


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarity between row vectors in ``a`` and ``b``."""
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(a_norm[:, None] * b_norm[None, :], 1.0e-12)
    return (a @ b.T) / denom


def decode_heldout_repeats(rates: np.ndarray, top_k: int) -> dict[str, Any]:
    """Compute held-out-repeat nearest-template decoding metrics."""
    n_repeats, n_frames, _ = rates.shape
    if n_repeats < 2:
        raise ValueError("need at least two repeats for held-out-repeat decoding")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    ranks: list[int] = []
    top1 = 0
    topk = 0
    same_values: list[float] = []
    different_values: list[float] = []

    frame_ids = np.arange(n_frames)
    for heldout in range(n_repeats):
        train_indices = [idx for idx in range(n_repeats) if idx != heldout]
        templates = rates[train_indices].mean(axis=0)
        samples = rates[heldout]
        sims = cosine_matrix(samples, templates)

        for frame in range(n_frames):
            true_score = sims[frame, frame]
            rank = int(1 + np.sum(sims[frame] > true_score))
            ranks.append(rank)
            if rank == 1:
                top1 += 1
            if rank <= top_k:
                topk += 1

            same_values.append(float(true_score))
            different_values.extend(float(value) for idx, value in enumerate(sims[frame]) if idx != frame)

    sample_count = n_repeats * n_frames
    same = float(np.mean(same_values))
    different = float(np.mean(different_values))
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return {
        "repeat_count": int(n_repeats),
        "frame_count": int(n_frames),
        "sample_count": int(sample_count),
        "top1_accuracy": float(top1 / sample_count),
        f"top{top_k}_accuracy": float(topk / sample_count),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "same_similarity": same,
        "different_similarity": different,
        "same_different_gap": same - different,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_site_rates_csv", type=Path)
    parser.add_argument("--population", default="l23e")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    rates, repeats, frames = load_population_site_rates(args.video_site_rates_csv, args.population)
    metrics = decode_heldout_repeats(rates, args.top_k)
    metrics["population"] = args.population
    metrics["source"] = str(args.video_site_rates_csv)
    metrics["repeat_indices"] = repeats
    metrics["frame_indices_first_last"] = [frames[0], frames[-1]]

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        for key in [
            "source",
            "population",
            "repeat_count",
            "frame_count",
            "sample_count",
            "top1_accuracy",
            f"top{args.top_k}_accuracy",
            "mean_rank",
            "median_rank",
            "same_similarity",
            "different_similarity",
            "same_different_gap",
        ]:
            print(f"{key}={metrics[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
