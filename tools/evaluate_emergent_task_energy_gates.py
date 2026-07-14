#!/usr/bin/env python3
"""Replay and evaluate the frozen task-energy endpoint gates.

This is a portable adaptation of the corrected external v2 evaluator whose
SHA-256 is recorded in :data:`AUTHORITATIVE_EVALUATOR_V2_SHA256`. It performs
no training and changes no checkpoint. Every endpoint is loaded in evaluation
mode and replayed with normal feedback enabled (``fb_scale=1``).

For each seed, the assay supplies 216 matched length-five histories for
operational continuation A and matched operational OOD reversal B. Replayed
L2/3 rates have shape ``[216, 5, 36]``. Whole-profile retention is

``M = AUC(A final aligned 36-bin profile) / AUC(literal-t0 36-bin profile)``.

The literal-t0 profile averages the A and B first responses after aligning each
row to its own presented first orientation. It comes from the normal
feedback-on unroll, whose fed-down state is naturally zero at the first step.
The circular rectangular-rule AUC multiplies both 36-bin sums by the 5-degree
bin width, so M is also checked against the corresponding mean-rate ratio.
No epsilon is added to the M denominator, which must be finite and positive.

The stored assay's rate-saving denominator uses its original
``epsilon_rate = 1e-8 * 36 * R_ref``; that value is identity-checked but is not
used by any frozen gate. Decoder noise is also replayed exactly as the assay
defines it: independent fixed generators with seeds 910001 and 910002 for the
training and test noise tables. There is no other randomness in this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import assay_emergent_task_energy_axis as assay  # noqa: E402
from tools import plot_emergent_reference_figures as figures  # noqa: E402


SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
AUTHORITATIVE_EVALUATOR_V2_SHA256 = (
    "9df9f9f54abcb2ab1d7e175d2498c2af0847f60256941597aa28893ccfe15ef9"
)
REQUIRED_SEED_COUNT = 4
DECODING_ACCURACY_THRESHOLD = 1.0 / 36.0
M_COMPARATOR_RATIO_MIN = 1.25
M_COMPARATOR_DIFFERENCE_MIN = 0.040
FRET_COMPARATOR_RATIO_MIN = 1.15
FRET_COMPARATOR_DIFFERENCE_MIN = 0.040
COHORT_MEAN_M_MIN = 0.250
M_DEFINITION = "whole_36_bin_expected_A_AUC_over_timestep0_AUC"
FLOAT_REL_TOL = 1e-10
FLOAT_ABS_TOL = 1e-10


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without retaining its location."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    """Apply the corrected evaluator's tolerance to an identity check."""

    return math.isclose(
        float(left),
        float(right),
        rel_tol=FLOAT_REL_TOL,
        abs_tol=FLOAT_ABS_TOL,
    )


def atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    """Atomically write a deterministic, pretty-printed JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_artifact_alpha(parser: argparse.ArgumentParser, name: str, alpha: float) -> None:
    """Reject values that cannot identify the repository's one-decimal files."""

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        parser.error(f"{name} must be finite and lie in [0,1]")
    canonical = float(f"{alpha:.1f}")
    if not math.isclose(alpha, canonical, rel_tol=0.0, abs_tol=1e-12):
        parser.error(f"{name} must use the one-decimal checkpoint protocol")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the portable four-seed replay interface."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help=(
            "Seed directory with common_pretrain_final.pt and both requested "
            "alpha final checkpoints; repeat exactly four times."
        ),
    )
    parser.add_argument("--candidate-alpha", required=True, type=float)
    parser.add_argument("--comparator-alpha", required=True, type=float)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch replay device, for example cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Portable JSON decision path; it must be outside every input run directory.",
    )
    args = parser.parse_args(argv)

    if len(args.run_dir) != REQUIRED_SEED_COUNT:
        parser.error(f"--run-dir must be supplied exactly {REQUIRED_SEED_COUNT} times")
    resolved_dirs = [path.expanduser().resolve() for path in args.run_dir]
    if len(set(resolved_dirs)) != len(resolved_dirs):
        parser.error("--run-dir values must be distinct")
    missing = [path for path in resolved_dirs if not path.is_dir()]
    if missing:
        parser.error(f"run directory does not exist: {missing[0]}")

    _validate_artifact_alpha(parser, "--candidate-alpha", args.candidate_alpha)
    _validate_artifact_alpha(parser, "--comparator-alpha", args.comparator_alpha)
    if math.isclose(
        args.candidate_alpha,
        args.comparator_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        parser.error("candidate and comparator alpha values must be distinct")

    output_path = args.out.expanduser().resolve()
    for run_dir in resolved_dirs:
        if output_path == run_dir or run_dir in output_path.parents:
            parser.error("--out must be outside every input run directory")
    args.run_dir = resolved_dirs
    args.out = output_path
    return args


def discover_assay_path(
    run_dir: Path,
    seed: int,
    candidate_alpha: float,
    comparator_alpha: float,
) -> Path:
    """Find one assay using the two proven layouts, rejecting ambiguity.

    Normal runs keep ``endpoint_assay.json`` (or the assay CLI's default
    ``assay.json``) inside the seed directory. The sealed fresh-confirmation
    bundle keeps ``seed_<n>_alpha_<candidate>_<comparator>.json`` in a sibling
    ``assay`` directory.
    """

    candidate_slug = assay.alpha_slug(candidate_alpha)
    comparator_slug = assay.alpha_slug(comparator_alpha)
    possible = [
        run_dir / "endpoint_assay.json",
        run_dir / "assay.json",
        run_dir.parent
        / "assay"
        / (
            f"seed_{seed}_alpha_{candidate_slug}_{comparator_slug}.json"
        ),
    ]
    matches = sorted(
        {path.resolve() for path in possible if path.is_file()},
        key=lambda path: path.as_posix(),
    )
    if not matches:
        checked = ", ".join(path.name for path in possible)
        raise FileNotFoundError(
            f"no assay found for seed {seed}; checked deterministic names: {checked}"
        )
    if len(matches) != 1:
        names = ", ".join(path.as_posix() for path in matches)
        raise ValueError(f"ambiguous assays for seed {seed}: {names}")
    return matches[0]


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other top-level JSON values."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def whole_profile_retention_from_replay(
    theta_a: torch.Tensor,
    theta_b: torch.Tensor,
    finals: torch.Tensor,
    rates_a_all: torch.Tensor,
    rates_b_all: torch.Tensor,
) -> dict[str, float]:
    """Compute corrected whole-profile retention from replay tensors.

    ``theta_a`` and ``theta_b`` are degree-valued ``[P,S]`` histories;
    ``rates_*_all`` are arbitrary-unit L2/3 rates with shape ``[P,S,36]``;
    ``finals`` contains ``[P]`` final-orientation channel indices. Production
    assays use ``P=216`` and ``S=5``. Smaller ``P`` fixtures are accepted for
    focused tests. M has no epsilon: both the AUC and t0 mean-rate denominators
    must be finite and strictly positive.
    """

    if theta_a.ndim != 2 or theta_b.shape != theta_a.shape:
        raise ValueError("A/B theta histories must have the same [P,S] shape")
    pairs, sequence_length = theta_a.shape
    expected_rate_shape = (pairs, sequence_length, assay.N)
    if tuple(rates_a_all.shape) != expected_rate_shape:
        raise ValueError(
            f"A rates must have shape {expected_rate_shape}, got {tuple(rates_a_all.shape)}"
        )
    if tuple(rates_b_all.shape) != expected_rate_shape:
        raise ValueError(
            f"B rates must have shape {expected_rate_shape}, got {tuple(rates_b_all.shape)}"
        )
    if tuple(finals.shape) != (pairs,):
        raise ValueError(f"final channels must have shape {(pairs,)}")
    if pairs < 1 or sequence_length < 1:
        raise ValueError("retention replay requires at least one row and time step")
    if not torch.isfinite(rates_a_all).all() or not torch.isfinite(rates_b_all).all():
        raise ValueError("retention replay rates must be finite")

    rates_a = rates_a_all[:, -1, :]
    first_a = (theta_a[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    first_b = (theta_b[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    aligned_a = assay.align_rates(rates_a, finals).to(torch.float64).mean(dim=0)
    t0_a = (
        assay.align_rates(rates_a_all[:, 0, :], first_a)
        .to(torch.float64)
        .mean(dim=0)
    )
    t0_b = (
        assay.align_rates(rates_b_all[:, 0, :], first_b)
        .to(torch.float64)
        .mean(dim=0)
    )
    t0_curve = 0.5 * (t0_a + t0_b)
    auc = assay.STEP_DEG * aligned_a.sum()
    auc0 = assay.STEP_DEG * t0_curve.sum()
    mean_a = rates_a.to(torch.float64).mean()
    mean_t0 = 0.5 * (
        rates_a_all[:, 0, :].to(torch.float64).mean()
        + rates_b_all[:, 0, :].to(torch.float64).mean()
    )
    if not torch.isfinite(auc0) or not float(auc0.item()) > 0.0:
        raise ValueError("literal-t0 AUC must be finite and positive")
    if not torch.isfinite(mean_t0) or not float(mean_t0.item()) > 0.0:
        raise ValueError("literal-t0 mean rate must be finite and positive")

    m_auc = auc / auc0
    m_mean = mean_a / mean_t0
    if not math.isclose(
        float(m_auc.item()),
        float(m_mean.item()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("whole-profile AUC and mean-rate retention disagree")
    return {
        "M": float(m_auc.item()),
        "M_mean_ratio": float(m_mean.item()),
        "AUC": float(auc.item()),
        "AUC0": float(auc0.item()),
        "mean_rate_t0": float(mean_t0.item()),
    }


@torch.inference_mode()
def whole_profile_retention(
    run_dir: Path,
    alpha: float,
    device: torch.device,
) -> dict[str, float]:
    """Replay one endpoint and compute M from normal feedback-on responses."""

    checkpoint_path = run_dir / f"alpha_{assay.alpha_slug(alpha)}_final.pt"
    net, checkpoint = assay.load_arm(checkpoint_path, device)
    if not close(float(checkpoint["alpha"]), alpha):
        raise ValueError(f"{checkpoint_path.name} has the wrong alpha metadata")
    theta_a, theta_b, finals = assay.matched_pairs(device)
    center_feedback = bool(checkpoint.get("center_feedback", False))
    feedback_mode = assay.tuned.resolve_feedback_mode(
        center_feedback,
        checkpoint.get("feedback_mode"),
    )
    _, rates_a_all = assay.tuned.forward_seq_tuned(
        net,
        theta_a,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
    )
    _, rates_b_all = assay.tuned.forward_seq_tuned(
        net,
        theta_b,
        1.0,
        center_feedback=center_feedback,
        feedback_mode=feedback_mode,
    )
    return whole_profile_retention_from_replay(
        theta_a,
        theta_b,
        finals,
        rates_a_all,
        rates_b_all,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise an identity error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"assay field {name} must be an object")
    return value


def validate_assay_identity(
    payload: Mapping[str, Any],
    candidate_alpha: float,
    comparator_alpha: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate assay protocol identity and return the two endpoint records."""

    if set(payload) != {"metadata", "per_alpha"}:
        raise ValueError("assay must contain exactly metadata and per_alpha")
    metadata = _mapping(payload["metadata"], "metadata")
    per_alpha = _mapping(payload["per_alpha"], "per_alpha")
    expected_pairs = assay.N * len(assay.VELOCITIES)
    if int(metadata.get("pair_count", -1)) != expected_pairs:
        raise ValueError(f"assay pair_count must be {expected_pairs}")
    if int(metadata.get("final_channels", -1)) != assay.N:
        raise ValueError(f"assay final_channels must be {assay.N}")
    if metadata.get("condition_a_history") != "[y-4v,y-3v,y-2v,y-v,y] mod 36":
        raise ValueError("assay condition A history does not match the frozen protocol")
    if metadata.get("condition_b_history") != "[y+2v,y+v,y,y-v,y] mod 36":
        raise ValueError("assay condition B history does not match the frozen protocol")
    decoder = _mapping(
        metadata.get("held_out_decoding_protocol"),
        "metadata.held_out_decoding_protocol",
    )
    if int(decoder.get("train_noise_seed", -1)) != assay.DECODER_TRAIN_NOISE_SEED:
        raise ValueError("assay decoder training-noise seed does not match")
    if int(decoder.get("test_noise_seed", -1)) != assay.DECODER_TEST_NOISE_SEED:
        raise ValueError("assay decoder test-noise seed does not match")

    records: list[Mapping[str, Any]] = []
    for alpha in (candidate_alpha, comparator_alpha):
        key = f"{alpha:.1f}"
        if key not in per_alpha:
            raise ValueError(f"assay lacks requested alpha {key}")
        records.append(_mapping(per_alpha[key], f"per_alpha.{key}"))
    return records[0], records[1]


def metrics_from_measurement(
    measurement: figures.SeedArmMeasurement,
    retention: Mapping[str, float],
    assay_record: Mapping[str, Any],
) -> dict[str, float]:
    """Bind checkpoint replay to stored assay leaves and return gate metrics."""

    rate = _mapping(assay_record["mean_rate_energy_saving"], "mean_rate_energy_saving")
    decoding = _mapping(
        assay_record["condition_blind_held_out_36_class_decoding"],
        "condition_blind_held_out_36_class_decoding",
    )
    shape = _mapping(
        assay_record["aligned_center_flank_Q_shape_contrasts"],
        "aligned_center_flank_Q_shape_contrasts",
    )
    condition_a_rate = _mapping(rate["condition_a_mean_rate"], "condition_a_mean_rate")
    condition_b_rate = _mapping(rate["condition_b_mean_rate"], "condition_b_mean_rate")
    center_delta = _mapping(shape["center_a_minus_b_over_R_ref"], "center delta")
    flank_delta = _mapping(shape["flank_a_minus_b_over_R_ref"], "flank delta")
    q_delta = _mapping(shape["Q_a_minus_b"], "Q delta")

    if measurement.zero_context_center <= 0.0 or measurement.zero_context_flank <= 0.0:
        raise ValueError("literal-t0 center and flank rates must be positive")
    expected_epsilon = 1e-8 * assay.N * measurement.rate_reference
    if not close(measurement.epsilon_rate, expected_epsilon):
        raise ValueError("replayed assay epsilon_rate identity failed")

    current = {
        "M": float(retention["M"]),
        "AUC": float(retention["AUC"]),
        "AUC0": float(retention["AUC0"]),
        "mean_rate_t0": float(retention["mean_rate_t0"]),
        "rate_A": float(measurement.mean_rate_expected),
        "rate_B": float(measurement.mean_rate_unexpected),
        "decode_A": float(measurement.decode_expected),
        "decode_B": float(measurement.decode_unexpected),
        "Cret": float(
            measurement.expected_center / measurement.zero_context_center
        ),
        "Fret": float(
            measurement.expected_flank / measurement.zero_context_flank
        ),
        "dC": float(center_delta["mean"]),
        "dF": float(flank_delta["mean"]),
        "dQ": float(q_delta["mean"]),
    }
    identities = {
        "stored saving": (
            measurement.stored_saving,
            float(rate["relative_saving_ratio_of_means"]),
        ),
        "M mean ratio": (current["M"], float(retention["M_mean_ratio"])),
        "rate A": (current["rate_A"], float(condition_a_rate["mean"])),
        "rate B": (current["rate_B"], float(condition_b_rate["mean"])),
        "decode A": (
            current["decode_A"],
            float(decoding["expected_A_held_out_top1_accuracy"]),
        ),
        "decode B": (
            current["decode_B"],
            float(decoding["unexpected_B_held_out_top1_accuracy"]),
        ),
        "dC": (
            current["dC"],
            (measurement.expected_center - measurement.unexpected_center)
            / measurement.rate_reference,
        ),
        "dF": (
            current["dF"],
            (measurement.expected_flank - measurement.unexpected_flank)
            / measurement.rate_reference,
        ),
        "dQ": (current["dQ"], measurement.delta_q),
    }
    for name, (left, right) in identities.items():
        if not close(left, right):
            raise ValueError(
                f"checkpoint/assay identity mismatch for {name}: {left} != {right}"
            )
    if not all(math.isfinite(value) for value in current.values()):
        raise ValueError("gate metrics must all be finite")
    return current


def evaluate_seed_gates(
    candidate: Mapping[str, float],
    comparator: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, bool]]:
    """Apply the corrected evaluator's exact twelve per-seed gates."""

    if comparator["M"] <= 0.0 or comparator["Fret"] <= 0.0:
        raise ValueError("comparator M and Fret must be positive")
    m_ratio = candidate["M"] / comparator["M"]
    m_difference = candidate["M"] - comparator["M"]
    fret_ratio = candidate["Fret"] / comparator["Fret"]
    fret_difference = candidate["Fret"] - comparator["Fret"]
    comparators = {
        "M_ratio": m_ratio,
        "M_difference": m_difference,
        "Fret_ratio": fret_ratio,
        "Fret_difference": fret_difference,
    }
    gates = {
        "energy_A_less_B": candidate["rate_A"] < candidate["rate_B"],
        "decoding_A_less_B": candidate["decode_A"] < candidate["decode_B"],
        "decoding_A_above_chance": (
            candidate["decode_A"] > DECODING_ACCURACY_THRESHOLD
        ),
        "decoding_B_above_chance": (
            candidate["decode_B"] > DECODING_ACCURACY_THRESHOLD
        ),
        "Cret_less_Fret": candidate["Cret"] < candidate["Fret"],
        "dC_less_dF": candidate["dC"] < candidate["dF"],
        "dQ_less_zero": candidate["dQ"] < 0.0,
        "Cret_less_one": candidate["Cret"] < 1.0,
        "M_comparator_ratio_at_least_1p25": (
            m_ratio >= M_COMPARATOR_RATIO_MIN
        ),
        "M_comparator_difference_at_least_0p040": (
            m_difference >= M_COMPARATOR_DIFFERENCE_MIN
        ),
        "Fret_comparator_ratio_at_least_1p15": (
            fret_ratio >= FRET_COMPARATOR_RATIO_MIN
        ),
        "Fret_comparator_difference_at_least_0p040": (
            fret_difference >= FRET_COMPARATOR_DIFFERENCE_MIN
        ),
    }
    return comparators, gates


def frozen_thresholds() -> dict[str, float]:
    """Return honestly named numeric thresholds for the frozen protocol."""

    return {
        "decoding_accuracy_min_exclusive": DECODING_ACCURACY_THRESHOLD,
        "M_comparator_ratio_min_inclusive": M_COMPARATOR_RATIO_MIN,
        "M_comparator_difference_min_inclusive": M_COMPARATOR_DIFFERENCE_MIN,
        "Fret_comparator_ratio_min_inclusive": FRET_COMPARATOR_RATIO_MIN,
        "Fret_comparator_difference_min_inclusive": FRET_COMPARATOR_DIFFERENCE_MIN,
        "cohort_mean_M_candidate_min_inclusive": COHORT_MEAN_M_MIN,
    }


def evaluate_seed_run(
    run_dir: Path,
    candidate_alpha: float,
    comparator_alpha: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay, identity-check, and evaluate one seed directory."""

    common_seed, common_local = figures.load_common_local_comp(run_dir, device)
    assay_path = discover_assay_path(
        run_dir,
        common_seed,
        candidate_alpha,
        comparator_alpha,
    )
    assay_payload = load_json_object(assay_path)
    candidate_record, comparator_record = validate_assay_identity(
        assay_payload,
        candidate_alpha,
        comparator_alpha,
    )
    metrics: dict[str, dict[str, float]] = {}
    measurements: dict[float, figures.SeedArmMeasurement] = {}
    for alpha, record in (
        (candidate_alpha, candidate_record),
        (comparator_alpha, comparator_record),
    ):
        measurement = figures.measure_seed_arm(
            run_dir,
            alpha,
            device,
            common_local,
        )
        if measurement.seed != common_seed or not close(measurement.alpha, alpha):
            raise ValueError("common/checkpoint seed or alpha metadata disagree")
        retention = whole_profile_retention(run_dir, alpha, device)
        metrics[f"{alpha:.1f}"] = metrics_from_measurement(
            measurement,
            retention,
            record,
        )
        measurements[alpha] = measurement

    candidate = metrics[f"{candidate_alpha:.1f}"]
    comparator = metrics[f"{comparator_alpha:.1f}"]
    comparators, gates = evaluate_seed_gates(candidate, comparator)
    passed = all(gates.values())
    seed_result = {
        "seed": common_seed,
        "metrics": metrics,
        "comparators": comparators,
        "gates": gates,
        "status": "passed" if passed else "failed",
    }
    sources = {
        "seed": common_seed,
        "common_pretrain": {
            "logical_id": f"seed_{common_seed}/common_pretrain_final.pt",
            "file_sha256": sha256_file(run_dir / "common_pretrain_final.pt"),
        },
        "candidate_checkpoint": {
            "alpha": candidate_alpha,
            "logical_id": (
                f"seed_{common_seed}/{measurements[candidate_alpha].checkpoint.name}"
            ),
            "file_sha256": measurements[candidate_alpha].checkpoint_sha256,
        },
        "comparator_checkpoint": {
            "alpha": comparator_alpha,
            "logical_id": (
                f"seed_{common_seed}/{measurements[comparator_alpha].checkpoint.name}"
            ),
            "file_sha256": measurements[comparator_alpha].checkpoint_sha256,
        },
        "assay": {
            "logical_id": f"seed_{common_seed}/endpoint_assay.json",
            "file_sha256": sha256_file(assay_path),
        },
    }
    return seed_result, sources


def build_decision(
    seed_results: Sequence[Mapping[str, Any]],
    source_artifacts: Sequence[Mapping[str, Any]],
    candidate_alpha: float,
    comparator_alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    """Build a deterministic, host-path-free cohort decision."""

    ordered_results = sorted(seed_results, key=lambda result: int(result["seed"]))
    ordered_sources = sorted(source_artifacts, key=lambda result: int(result["seed"]))
    if len(ordered_results) != REQUIRED_SEED_COUNT:
        raise ValueError(f"decision requires exactly {REQUIRED_SEED_COUNT} seed results")
    seeds = [int(result["seed"]) for result in ordered_results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("checkpoint metadata contain duplicate seeds")
    candidate_key = f"{candidate_alpha:.1f}"
    candidate_m = [
        float(_mapping(result["metrics"], "metrics")[candidate_key]["M"])
        for result in ordered_results
    ]
    cohort_mean_m = sum(candidate_m) / len(candidate_m)
    cohort_gate = cohort_mean_m >= COHORT_MEAN_M_MIN
    all_pass = cohort_gate and all(
        result["status"] == "passed" for result in ordered_results
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "id": "tools/evaluate_emergent_task_energy_gates.py",
            "version": GENERATOR_VERSION,
            "source_file_sha256": sha256_file(Path(__file__).resolve()),
            "authoritative_external_evaluator_v2_sha256": (
                AUTHORITATIVE_EVALUATOR_V2_SHA256
            ),
        },
        "status": "passed" if all_pass else "failed",
        "seeds": seeds,
        "candidate_alpha": candidate_alpha,
        "comparator_alpha": comparator_alpha,
        "M_definition": M_DEFINITION,
        "protocol": {
            "conditions": {
                "A": "operational continuation [y-4v,y-3v,y-2v,y-v,y] mod 36",
                "B": "matched operational OOD reversal [y+2v,y+v,y,y-v,y] mod 36",
            },
            "matched_rows_per_seed_condition": assay.N * len(assay.VELOCITIES),
            "history_shape": [assay.N * len(assay.VELOCITIES), 5],
            "rate_replay_shape": [
                assay.N * len(assay.VELOCITIES),
                5,
                assay.N,
            ],
            "feedback": "normal feedback-on unroll with fb_scale=1",
            "literal_t0": (
                "A/B first responses aligned to each presented first orientation "
                "and averaged; fed-down state is naturally zero"
            ),
            "M_denominator_epsilon": 0.0,
            "assay_rate_saving_epsilon_formula": "1e-8 * N * R_ref",
            "randomness": {
                "retention_replay": "none",
                "decoder_train_noise_seed": assay.DECODER_TRAIN_NOISE_SEED,
                "decoder_test_noise_seed": assay.DECODER_TEST_NOISE_SEED,
            },
            "retraining": False,
            "replay_device": str(device),
            "thresholds": frozen_thresholds(),
        },
        "seed_results": ordered_results,
        "cohort": {
            "candidate_mean_M": cohort_mean_m,
            "gate_mean_M_at_least_0p250": cohort_gate,
        },
        "source_artifacts": ordered_sources,
        "source_file_sha256": {
            "assay": sha256_file(ROOT / "tools" / "assay_emergent_task_energy_axis.py"),
            "plotter": sha256_file(ROOT / "tools" / "plot_emergent_reference_figures.py"),
            "tuned_library": sha256_file(ROOT / "tools" / "tuned_emergence_lib.py"),
            "model": sha256_file(ROOT / "simple_net.py"),
        },
        "portability": {
            "absolute_paths_present": False,
            "source_paths": "logical artifact identifiers and SHA-256 digests only",
            "seal_or_ledger_generated": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only four-seed replay and write its portable decision."""

    args = parse_args(argv)
    device = assay.choose_device(args.device)
    seed_results: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for run_dir in args.run_dir:
        seed_result, sources = evaluate_seed_run(
            run_dir,
            args.candidate_alpha,
            args.comparator_alpha,
            device,
        )
        seed = int(seed_result["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        seen_seeds.add(seed)
        seed_results.append(seed_result)
        source_artifacts.append(sources)
        print(json.dumps({"seed": seed, "status": "measured_seed"}, sort_keys=True))

    decision = build_decision(
        seed_results,
        source_artifacts,
        args.candidate_alpha,
        args.comparator_alpha,
        device,
    )
    atomic_json_save(decision, args.out)
    print(
        json.dumps(
            {
                "seed_count": len(seed_results),
                "status": decision["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if decision["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
