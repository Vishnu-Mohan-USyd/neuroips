#!/usr/bin/env python3
"""Build a portable four-seed, six-alpha summary from standalone assays.

This tool aggregates fields already recorded by each seed's
``endpoint_assay.json``. It does not replay checkpoints, reconstruct unavailable
profiles, or feed the endpoint plotter. Paths supplied on the command line are
used only for reading; the output contains logical artifact IDs and SHA-256
digests, never workstation paths.

Generator provenance distinguishes the repository base commit from the exact
generator source: ``repository_base_commit`` records HEAD but is not a clean
snapshot guarantee, ``repository_worktree_dirty_at_generation`` reports local
tracked or untracked changes, and ``source_file_sha256`` pins this file's exact
contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
SEEDS = (0, 1, 2, 3)
ALPHAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
ALPHA_KEYS = tuple(f"{alpha:.1f}" for alpha in ALPHAS)

METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "final_mean_rate_A": (
        "mean_rate_energy_saving",
        "condition_a_mean_rate",
        "mean",
    ),
    "final_mean_rate_B": (
        "mean_rate_energy_saving",
        "condition_b_mean_rate",
        "mean",
    ),
    "saving_B_minus_A_over_B": (
        "mean_rate_energy_saving",
        "relative_saving_ratio_of_means",
    ),
    "fraction_pairs_B_greater_A": (
        "mean_rate_energy_saving",
        "fraction_pairs_unexpected_B_greater_expected_A",
    ),
    "decode_accuracy_A": (
        "condition_blind_held_out_36_class_decoding",
        "expected_A_held_out_top1_accuracy",
    ),
    "decode_accuracy_B": (
        "condition_blind_held_out_36_class_decoding",
        "unexpected_B_held_out_top1_accuracy",
    ),
    "delta_decode_A_minus_B": (
        "condition_blind_held_out_36_class_decoding",
        "expected_A_minus_unexpected_B_accuracy",
    ),
    "delta_C_A_minus_B_over_R_ref": (
        "aligned_center_flank_Q_shape_contrasts",
        "center_a_minus_b_over_R_ref",
        "mean",
    ),
    "delta_F_A_minus_B_over_R_ref": (
        "aligned_center_flank_Q_shape_contrasts",
        "flank_a_minus_b_over_R_ref",
        "mean",
    ),
    "delta_Fq_A_minus_B": (
        "aligned_center_flank_Q_shape_contrasts",
        "Fq_a_minus_b",
        "mean",
    ),
    "delta_Q_A_minus_B": (
        "aligned_center_flank_Q_shape_contrasts",
        "Q_a_minus_b",
        "mean",
    ),
    "population_alignment_A": (
        "circular_population_vector_alignment",
        "expected_A_alignment",
        "mean",
    ),
    "population_alignment_B": (
        "circular_population_vector_alignment",
        "unexpected_B_alignment",
        "mean",
    ),
    "delta_population_alignment_A_minus_B": (
        "circular_population_vector_alignment",
        "expected_A_minus_unexpected_B_alignment",
        "mean",
    ),
}

METRIC_DEFINITIONS: dict[str, str] = {
    "final_mean_rate_A": "mean final L2/3 rate under operational continuation A",
    "final_mean_rate_B": "mean final L2/3 rate under matched operational OOD reversal B",
    "saving_B_minus_A_over_B": "(final_mean_rate_B-final_mean_rate_A)/(final_mean_rate_B+eps_rate)",
    "fraction_pairs_B_greater_A": "fraction of matched pairs with final_mean_rate_B > final_mean_rate_A",
    "decode_accuracy_A": "condition-blind, noise-held-out orientation accuracy for continuation A",
    "decode_accuracy_B": "condition-blind, noise-held-out orientation accuracy for OOD reversal B",
    "delta_decode_A_minus_B": "decode_accuracy_A-decode_accuracy_B",
    "delta_C_A_minus_B_over_R_ref": "(aligned_center_A-aligned_center_B)/R_ref",
    "delta_F_A_minus_B_over_R_ref": "(aligned_flank_A-aligned_flank_B)/R_ref",
    "delta_Fq_A_minus_B": "normalized_flank_A-normalized_flank_B",
    "delta_Q_A_minus_B": "shape_index_Q_A-shape_index_Q_B",
    "population_alignment_A": "circular population-vector alignment for continuation A",
    "population_alignment_B": "circular population-vector alignment for OOD reversal B",
    "delta_population_alignment_A_minus_B": "population_alignment_A-population_alignment_B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Seed run directory containing endpoint_assay.json, "
            "training_summary.json, training.jsonl, and final checkpoints. "
            "Repeat exactly four times."
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--generated-at",
        required=True,
        help=(
            "Explicit ISO-8601 timestamp used for reproducible generation. "
            "The output separately records the repository base commit, "
            "worktree dirty state, and exact generator-source SHA-256."
        ),
    )
    args = parser.parse_args()
    if len(args.run_dir) != len(SEEDS):
        parser.error(f"--run-dir must be supplied exactly {len(SEEDS)} times")
    args.run_dir = [path.expanduser().resolve() for path in args.run_dir]
    if len(set(args.run_dir)) != len(args.run_dir):
        parser.error("--run-dir values must be distinct")
    args.out = args.out.expanduser().resolve()
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def first_event(path: Path, event_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == event_name:
                return event
    raise ValueError(f"{path.name} lacks event {event_name!r}")


def nested_float(payload: dict[str, Any], path: Sequence[str]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric at {'.'.join(path)}")
    return result


def mean_sem(values: Sequence[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.shape != (len(SEEDS),) or not torch.isfinite(tensor).all():
        raise ValueError("aggregate requires four finite seed values")
    return {
        "mean": float(tensor.mean().item()),
        "sem": float((tensor.std(unbiased=True) / math.sqrt(len(SEEDS))).item()),
        "n_seeds": len(SEEDS),
    }


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.1f}".replace(".", "p")


def logical_artifact_id(seed: int, basename: str) -> str:
    return f"seed_{seed}/{basename}"


def repository_base_commit() -> str:
    """Return HEAD as a base commit, without implying a clean snapshot."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def repository_worktree_dirty() -> bool:
    """Return whether tracked or untracked files differ from the base commit."""

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )
    return bool(status.strip())


def generation_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def checkpoint_provenance(
    run_dir: Path,
    seed: int,
    training: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int, int, float]:
    common_path = run_dir / "common_pretrain_final.pt"
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    if common["seed"] != seed or common["stage"] != "common_pretrain":
        raise ValueError(f"invalid common checkpoint for seed {seed}")
    model_config = common["tuned_net_config"]
    common_record = {
        "logical_id": logical_artifact_id(seed, common_path.name),
        "basename": common_path.name,
        "file_sha256": sha256_file(common_path),
        "state_sha256": training["common_pretrain_state_sha256"],
    }

    training_by_alpha = {
        f"{float(row['alpha']):.1f}": row for row in training["alphas"]
    }
    alpha_records: dict[str, Any] = {}
    axis_steps: set[int] = set()
    learning_rates: set[float] = set()
    for alpha, alpha_key in zip(ALPHAS, ALPHA_KEYS, strict=True):
        path = run_dir / f"alpha_{alpha_slug(alpha)}_final.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint["seed"] != seed
            or checkpoint["stage"] != "alpha_axis"
            or float(checkpoint["alpha"]) != alpha
            or checkpoint["tuned_net_config"] != model_config
        ):
            raise ValueError(f"invalid alpha checkpoint {alpha_key} for seed {seed}")
        axis_steps.add(int(checkpoint["target_steps"]))
        for group in checkpoint["optimizer_state_dict"]["param_groups"]:
            learning_rates.add(float(group["lr"]))
        summary = training_by_alpha[alpha_key]
        alpha_records[alpha_key] = {
            "logical_id": logical_artifact_id(seed, path.name),
            "basename": path.name,
            "file_sha256": sha256_file(path),
            "state_sha256": summary["state_sha256"],
        }
    if len(axis_steps) != 1 or len(learning_rates) != 1:
        raise ValueError(f"inconsistent optimizer protocol for seed {seed}")
    return (
        {"common_pretrain": common_record, "alpha_finals": alpha_records},
        model_config,
        int(common["target_steps"]),
        axis_steps.pop(),
        learning_rates.pop(),
    )


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    source_rows: list[dict[str, Any]] = []
    metric_rows: dict[str, list[dict[str, Any]]] = {
        alpha_key: [] for alpha_key in ALPHA_KEYS
    }
    model_configs: list[dict[str, Any]] = []
    pretrain_steps: set[int] = set()
    axis_steps: set[int] = set()
    learning_rates: set[float] = set()
    batch_sizes: set[int] = set()
    sequence_lengths: set[int] = set()

    loaded: list[tuple[int, Path, dict[str, Any], dict[str, Any]]] = []
    for run_dir in args.run_dir:
        assay_path = run_dir / "endpoint_assay.json"
        training_path = run_dir / "training_summary.json"
        log_path = run_dir / "training.jsonl"
        assay = load_json(assay_path)
        training = load_json(training_path)
        seed = int(training["seed"])
        loaded.append((seed, run_dir, assay, training))
    loaded.sort(key=lambda row: row[0])
    if tuple(row[0] for row in loaded) != SEEDS:
        raise ValueError(f"expected seed order {SEEDS}")

    for seed, run_dir, assay, training in loaded:
        if tuple(assay["per_alpha"]) != ALPHA_KEYS:
            raise ValueError(f"seed {seed} alpha order differs from protocol")
        if training["feedback_mode"] != "posterior_prior_excess":
            raise ValueError(f"seed {seed} feedback mode differs from protocol")
        if training["freeze_local_comp"] is not True:
            raise ValueError(f"seed {seed} local competition was not frozen")
        run_start = first_event(run_dir / "training.jsonl", "run_start")
        batch_sizes.add(int(run_start["batch"]))
        sequence_lengths.add(int(run_start["sequence_length"]))
        checkpoints, model_config, pre_steps, arm_steps, lr = checkpoint_provenance(
            run_dir, seed, training
        )
        model_configs.append(model_config)
        pretrain_steps.add(pre_steps)
        axis_steps.add(arm_steps)
        learning_rates.add(lr)

        source_rows.append(
            {
                "seed": seed,
                "references": training["references"],
                "training_device": training["device"],
                "source_artifacts": {
                    "standalone_assay": {
                        "logical_id": logical_artifact_id(seed, "endpoint_assay.json"),
                        "file_sha256": sha256_file(run_dir / "endpoint_assay.json"),
                    },
                    "training_summary_audit_record": {
                        "logical_id": logical_artifact_id(seed, "training_summary.json"),
                        "file_sha256": sha256_file(run_dir / "training_summary.json"),
                    },
                    "training_log_audit_record": {
                        "logical_id": logical_artifact_id(seed, "training.jsonl"),
                        "file_sha256": sha256_file(run_dir / "training.jsonl"),
                    },
                },
                "checkpoints": checkpoints,
            }
        )
        for alpha_key in ALPHA_KEYS:
            source_metrics = assay["per_alpha"][alpha_key]
            metrics = {
                name: nested_float(source_metrics, path)
                for name, path in METRIC_PATHS.items()
            }
            metric_rows[alpha_key].append({"seed": seed, "metrics": metrics})

    if not all(config == model_configs[0] for config in model_configs):
        raise ValueError("model config differs across seeds")
    if not all(len(values) == 1 for values in (
        pretrain_steps,
        axis_steps,
        learning_rates,
        batch_sizes,
        sequence_lengths,
    )):
        raise ValueError("training protocol differs across seeds")

    per_alpha: dict[str, Any] = {}
    for alpha_key in ALPHA_KEYS:
        rows = metric_rows[alpha_key]
        aggregates = {
            metric: mean_sem([row["metrics"][metric] for row in rows])
            for metric in METRIC_PATHS
        }
        per_alpha[alpha_key] = {
            "per_seed": rows,
            "aggregate": aggregates,
        }

    generator_path = Path(__file__).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "id": generator_path.relative_to(ROOT).as_posix(),
            "version": GENERATOR_VERSION,
            "source_file_sha256": sha256_file(generator_path),
            "repository_base_commit": repository_base_commit(),
            "repository_worktree_dirty_at_generation": (
                repository_worktree_dirty()
            ),
            "generated_at": args.generated_at,
        },
        "data_flow": (
            "final checkpoints -> standalone per-seed assay JSON -> "
            "this six-alpha aggregate"
        ),
        "training_protocol": {
            "seeds": list(SEEDS),
            "alphas": list(ALPHAS),
            "common_pretrain_steps": pretrain_steps.pop(),
            "axis_steps_per_arm": axis_steps.pop(),
            "batch_size": batch_sizes.pop(),
            "sequence_length": sequence_lengths.pop(),
            "optimizer": {
                "name": "Adam",
                "learning_rate": learning_rates.pop(),
                "gradient_norm_clip": 5.0,
            },
            "objective": "J_alpha=(1-alpha)*T+alpha*E",
            "rate_cost": "E=mean_{B,S,N}(raw_L23_rate)/R_ref",
            "feedback_mode": "posterior_prior_excess",
            "freeze_local_comp": True,
            "model_config": model_configs[0],
            "trainability": {
                "common_pretrain": ["GRU", "W_fb"],
                "alpha_arms": ["GRU", "W_fb", "five circ_raw motif gains"],
                "fixed_or_frozen": [
                    "L4 code",
                    "L4_to_L23 basis",
                    "local competition",
                    "built-in decode and decoder gain",
                ],
            },
            "rng": {
                "parameter_initialization": "seed",
                "pretrain_data": "200000+seed",
                "pretrain_population_noise": "300000+seed",
                "axis_data_each_arm": "400000+seed",
                "axis_population_noise_each_arm": "500000+seed",
            },
        },
        "input_process": {
            "period_degrees": 180,
            "channels": 36,
            "nominal_degrees_per_channel": 5,
            "sequence_length": 12,
            "initial_channel": "uniform integer 0..35",
            "acceleration_alphabet": [-1, 0, 1],
            "acceleration_persistence": 0.9,
            "otherwise": "uniform resample from {-1,0,1}",
            "initial_velocity": "uniform integer -4..4",
            "velocity_update": "clip(v[t-1]+a[t-1],-4,4)",
            "position_update": "integrate velocity modulo 36",
        },
        "assay_protocol": {
            "condition_A": "operational continuation A",
            "condition_B": "matched operational OOD reversal B",
            "condition_A_history": "[y-4v,y-3v,y-2v,y-v,y] mod 36",
            "condition_B_history": "[y+2v,y+v,y,y-v,y] mod 36",
            "velocities": [-3, -2, -1, 1, 2, 3],
            "pair_count": 216,
            "feedback_execution": "normal feedback-on unroll in both conditions",
            "predictor_probability_gate": False,
            "decoder": {
                "name": "condition-blind, noise-held-out orientation decoding",
                "train_noise_seed": 910001,
                "test_noise_seed": 910002,
                "train_repeats_per_pair_condition": 32,
                "test_repeats_per_pair_condition": 32,
                "features": "ReLU(rate+Gaussian noise), per-trial L1 then L2",
                "paired_noise": "same A/B noise table within each split",
                "fit": "balanced pooled A+B cosine nearest-centroid",
                "held_out_scope": "independent test noise; same histories/orientations",
                "chance_accuracy": 1.0 / 36.0,
            },
            "shape": {
                "aligned_offsets_channels": list(range(-18, 18)),
                "center_offsets_channels": [-1, 0, 1],
                "flank_offsets_channels": [-6, -5, -4, -3, 3, 4, 5, 6],
                "eps_rate": "1e-8*36*R_ref",
                "q": "r/(sum(r)+eps_rate)",
                "Q": "(Cq-Fq)/(Cq+Fq+1e-8)",
            },
            "sign_conventions": {
                "delta_metrics": "A-B",
                "saving": "(mean_rate_B-mean_rate_A)/(mean_rate_B+eps_rate)",
            },
            "units": {
                "raw_rates": "arbitrary activity units",
                "decode_and_normalized_metrics": "dimensionless",
                "alpha": "dimensionless objective coordinate",
            },
            "aggregation": (
                "216 rows are reduced within seed; aggregate mean and sample "
                "SEM use four seed-level values"
            ),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "per_alpha": per_alpha,
        "provenance": {
            "sources": source_rows,
            "generation_environment": generation_environment(),
            "portability": "no absolute source paths or usernames stored",
        },
        "scope_limit": (
            "Only fields recorded by standalone endpoint_assay.json are "
            "aggregated; unavailable all-alpha raw profiles are not reconstructed."
        ),
    }
    atomic_json_save(payload, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": SCHEMA_VERSION,
                "seeds": list(SEEDS),
                "alphas": list(ALPHAS),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
