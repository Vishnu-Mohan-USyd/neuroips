"""Schema and scientific-consistency checks for endpoint selection provenance."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_emergent_task_energy_gates as evaluator  # noqa: E402


RECORD_PATH = (
    ROOT
    / "figures"
    / "emergent_reference_comparison"
    / "endpoint_selection_record.json"
)
EVALUATOR_PATH = ROOT / "tools" / "evaluate_emergent_task_energy_gates.py"


def _sha256(path: Path) -> str:
    """Return a test-local source digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_endpoint_selection_schema_and_evaluator_bindings_are_consistent() -> None:
    """The static record binds its schema, evaluator, and numeric thresholds."""

    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    generator = record["generator"]
    per_seed = record["frozen_gates"]["per_seed"]

    assert record["schema_version"] == "2.0.0"
    assert record["schema"] == {
        "id": "emergent_task_energy_endpoint_selection_record",
        "version": "2.0.0",
    }
    assert generator["id"] == "tools/evaluate_emergent_task_energy_gates.py"
    assert generator["version"] == evaluator.GENERATOR_VERSION
    assert generator["source_file_sha256"] == _sha256(EVALUATOR_PATH)
    assert generator["authoritative_external_evaluator_v2_sha256"] == (
        evaluator.AUTHORITATIVE_EVALUATOR_V2_SHA256
    )
    assert re.fullmatch(r"[0-9a-f]{64}", generator["source_file_sha256"])
    assert "decoding_A_above_chance" not in per_seed
    assert "decoding_B_above_chance" not in per_seed
    assert per_seed["decoding_A_accuracy_min_exclusive"] == (
        evaluator.DECODING_ACCURACY_THRESHOLD
    )
    assert per_seed["decoding_B_accuracy_min_exclusive"] == (
        evaluator.DECODING_ACCURACY_THRESHOLD
    )
    assert per_seed["M_comparator_ratio_min"] == evaluator.M_COMPARATOR_RATIO_MIN
    assert per_seed["M_comparator_difference_min"] == (
        evaluator.M_COMPARATOR_DIFFERENCE_MIN
    )
    assert per_seed["Fret_comparator_ratio_min"] == (
        evaluator.FRET_COMPARATOR_RATIO_MIN
    )
    assert per_seed["Fret_comparator_difference_min"] == (
        evaluator.FRET_COMPARATOR_DIFFERENCE_MIN
    )
    assert record["frozen_gates"]["cohort"]["mean_M_candidate_min"] == (
        evaluator.COHORT_MEAN_M_MIN
    )


def test_fresh_confirmation_values_satisfy_the_recorded_frozen_gates() -> None:
    """Stored fresh-confirmation ratios, differences, mean, and counts agree."""

    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    fresh = record["selection_history"]["alpha_0p5_fresh_confirmation"]
    values = fresh["per_seed"]
    candidate_m: list[float] = []
    for seed in (8, 9, 10, 11):
        leaf = values[str(seed)]
        candidate_m.append(leaf["M_alpha_0p5"])
        assert math.isclose(
            leaf["M_ratio"],
            leaf["M_alpha_0p5"] / leaf["M_alpha_0p9"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        assert math.isclose(
            leaf["M_difference"],
            leaf["M_alpha_0p5"] - leaf["M_alpha_0p9"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        assert leaf["M_ratio"] >= evaluator.M_COMPARATOR_RATIO_MIN
        assert leaf["M_difference"] >= evaluator.M_COMPARATOR_DIFFERENCE_MIN
        assert leaf["Fret_ratio"] >= evaluator.FRET_COMPARATOR_RATIO_MIN
        assert leaf["Fret_difference"] >= evaluator.FRET_COMPARATOR_DIFFERENCE_MIN
        assert leaf["gates_passed"] == leaf["gates_total"] == 12

    mean_m = sum(candidate_m) / len(candidate_m)
    assert math.isclose(
        fresh["mean_M_alpha_0p5"],
        mean_m,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert mean_m >= evaluator.COHORT_MEAN_M_MIN
    assert fresh["per_seed_gates_passed"] == fresh["per_seed_gates_total"] == 48
    assert fresh["cohort_mean_M_gate"] is True
