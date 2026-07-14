"""Behavioral coverage for the portable frozen-gate evaluator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_emergent_task_energy_gates as evaluator  # noqa: E402


def _gate_metrics(**overrides: float) -> dict[str, float]:
    """Return one complete synthetic metric record for pure gate tests."""

    metrics = {
        "M": 0.25,
        "AUC": 1.0,
        "AUC0": 4.0,
        "mean_rate_t0": 1.0,
        "rate_A": 0.10,
        "rate_B": 0.20,
        "decode_A": evaluator.DECODING_ACCURACY_THRESHOLD + 1e-6,
        "decode_B": 0.30,
        "Cret": 0.40,
        "Fret": 0.575,
        "dC": -0.20,
        "dF": -0.10,
        "dQ": -0.01,
    }
    metrics.update(overrides)
    return metrics


def test_whole_profile_retention_uses_all_bins_and_literal_t0_average() -> None:
    """M uses the full A profile over the pooled literal first response."""

    pairs, sequence_length, channels = 2, 2, evaluator.assay.N
    theta_a = torch.zeros((pairs, sequence_length), dtype=torch.float32)
    theta_b = torch.full(
        (pairs, sequence_length),
        evaluator.assay.STEP_DEG,
        dtype=torch.float32,
    )
    finals = torch.tensor((0, 1), dtype=torch.long)
    rates_a = torch.zeros((pairs, sequence_length, channels), dtype=torch.float32)
    rates_b = torch.zeros_like(rates_a)
    rates_a[:, 0, :] = 1.0
    rates_b[:, 0, :] = 3.0
    rates_a[:, -1, 0] = float(channels)

    result = evaluator.whole_profile_retention_from_replay(
        theta_a,
        theta_b,
        finals,
        rates_a,
        rates_b,
    )

    assert result == {
        "M": 0.5,
        "M_mean_ratio": 0.5,
        "AUC": 180.0,
        "AUC0": 360.0,
        "mean_rate_t0": 2.0,
    }


def test_seed_gate_math_preserves_inclusive_and_exclusive_boundaries() -> None:
    """Comparator gates are inclusive while above-chance gates are strict."""

    candidate = _gate_metrics()
    comparator = _gate_metrics(M=0.20, Fret=0.50)
    comparators, gates = evaluator.evaluate_seed_gates(candidate, comparator)

    assert comparators == {
        "M_ratio": 1.25,
        "M_difference": 0.04999999999999999,
        "Fret_ratio": 1.15,
        "Fret_difference": 0.07499999999999996,
    }
    assert len(gates) == 12
    assert all(gates.values())

    at_chance = _gate_metrics(
        decode_A=evaluator.DECODING_ACCURACY_THRESHOLD,
    )
    _, boundary_gates = evaluator.evaluate_seed_gates(at_chance, comparator)
    assert boundary_gates["decoding_A_above_chance"] is False


def test_cohort_gate_is_inclusive_and_decision_contains_no_input_paths() -> None:
    """Four passing seeds at exactly M=.250 pass the cohort threshold."""

    results = []
    sources = []
    for seed in (8, 9, 10, 11):
        candidate = _gate_metrics(M=0.25)
        comparator = _gate_metrics(M=0.20, Fret=0.50)
        comparators, gates = evaluator.evaluate_seed_gates(candidate, comparator)
        results.append(
            {
                "seed": seed,
                "metrics": {"0.5": candidate, "0.9": comparator},
                "comparators": comparators,
                "gates": gates,
                "status": "passed",
            }
        )
        sources.append({"seed": seed, "logical_id": f"seed_{seed}/fixture"})

    decision = evaluator.build_decision(
        results,
        sources,
        candidate_alpha=0.5,
        comparator_alpha=0.9,
        device=torch.device("cpu"),
    )

    assert decision["status"] == "passed"
    assert decision["cohort"] == {
        "candidate_mean_M": 0.25,
        "gate_mean_M_at_least_0p250": True,
    }
    assert decision["protocol"]["M_denominator_epsilon"] == 0.0
    assert decision["protocol"]["retraining"] is False
    assert "/home/" not in evaluator.json.dumps(decision, sort_keys=True)


def test_cli_accepts_four_distinct_external_output_run_dirs(tmp_path: Path) -> None:
    """The small CLI accepts the complete explicit frozen protocol."""

    run_dirs = [tmp_path / f"seed_{seed}" for seed in (8, 9, 10, 11)]
    for run_dir in run_dirs:
        run_dir.mkdir()
    argv: list[str] = []
    for run_dir in run_dirs:
        argv.extend(("--run-dir", str(run_dir)))
    argv.extend(
        (
            "--candidate-alpha",
            "0.5",
            "--comparator-alpha",
            "0.9",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "decision.json"),
        )
    )

    args = evaluator.parse_args(argv)

    assert args.run_dir == [path.resolve() for path in run_dirs]
    assert args.candidate_alpha == 0.5
    assert args.comparator_alpha == 0.9


@pytest.mark.parametrize(
    ("candidate", "comparator"),
    (("0.51", "0.9"), ("0.5", "0.5"), ("nan", "0.9")),
)
def test_cli_rejects_ambiguous_or_invalid_alphas(
    tmp_path: Path,
    candidate: str,
    comparator: str,
) -> None:
    """Artifact alphas must be finite, distinct one-decimal coordinates."""

    run_dirs = [tmp_path / f"seed_{seed}" for seed in (8, 9, 10, 11)]
    for run_dir in run_dirs:
        run_dir.mkdir()
    argv: list[str] = []
    for run_dir in run_dirs:
        argv.extend(("--run-dir", str(run_dir)))
    argv.extend(
        (
            "--candidate-alpha",
            candidate,
            "--comparator-alpha",
            comparator,
            "--out",
            str(tmp_path / "decision.json"),
        )
    )

    with pytest.raises(SystemExit, match="2"):
        evaluator.parse_args(argv)


def test_assay_discovery_supports_proven_layouts_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    """In-seed and sealed sibling assays cannot silently compete."""

    run_dir = tmp_path / "seed_8"
    run_dir.mkdir()
    in_seed = run_dir / "endpoint_assay.json"
    in_seed.write_text("{}\n", encoding="utf-8")
    assert evaluator.discover_assay_path(run_dir, 8, 0.5, 0.9) == in_seed.resolve()

    sibling = tmp_path / "assay" / "seed_8_alpha_0p5_0p9.json"
    sibling.parent.mkdir()
    sibling.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous assays"):
        evaluator.discover_assay_path(run_dir, 8, 0.5, 0.9)

    in_seed.unlink()
    assert evaluator.discover_assay_path(run_dir, 8, 0.5, 0.9) == sibling.resolve()
