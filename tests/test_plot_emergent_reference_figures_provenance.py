"""Focused regression coverage for portable alpha-driven plot metadata."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import plot_emergent_reference_figures as plotter  # noqa: E402


def test_alpha_0p5_drives_labels_and_portable_command_provenance() -> None:
    labels = plotter.endpoint_labels(task_alpha=0.0, energy_alpha=0.5)

    assert labels == {
        "task": "task-only endpoint (α=0.0)",
        "energy": "rate-cost-weighted endpoint (α=0.5)",
        "task_multiline": "task-only endpoint\n(α=0.0)",
        "energy_multiline": "rate-cost-weighted endpoint\n(α=0.5)",
        "energy_objective": (
            "50% task + 50% normalized L2/3 mean-rate proxy"
        ),
    }

    provenance = plotter.portable_command_provenance(
        seed_ids=[8, 9, 10, 11],
        task_alpha=0.0,
        energy_alpha=0.5,
        device="cuda:0",
        output_directory="figures/emergent_reference_comparison",
    )
    encoded = json.dumps(
        {"labels": labels, "command_provenance": provenance},
        sort_keys=True,
    )
    assert provenance["argv"].count("--run-dir") == 4
    assert "bundle:seed_8" in provenance["argv"]
    assert "bundle:seed_11" in provenance["argv"]
    assert "/home/" not in encoded
    assert "α=0.9" not in encoded
    assert "10% task" not in encoded
    assert "90% normalized" not in encoded

    tuning_source = inspect.getsource(plotter.plot_tuning)
    assert "literal t0 first stimulus" in tuning_source
    assert "normal feedback-on unroll" in tuning_source


def test_phase_rate_metadata_discloses_exact_epsilon_denominator() -> None:
    r_ref = 0.16667839884757996
    epsilon_rate = 1e-8 * plotter.assay.N * r_ref
    mean_rate_a = 0.1
    mean_rate_b = 0.2
    phase_y = (mean_rate_a - mean_rate_b) / (mean_rate_b + epsilon_rate)
    curve = plotter.torch.ones(len(plotter.PLOT_OFFSETS), dtype=plotter.torch.float64)
    measurement = plotter.SeedArmMeasurement(
        seed=8,
        checkpoint=Path("alpha_0p5_final.pt"),
        checkpoint_sha256="0" * 64,
        alpha=0.5,
        feedback_mode="posterior_prior_excess",
        expected_curve=curve,
        unexpected_curve=curve,
        zero_context_curve=curve,
        expected_center=1.0,
        expected_flank=1.0,
        unexpected_center=1.0,
        unexpected_flank=1.0,
        zero_context_center=1.0,
        zero_context_flank=1.0,
        delta_fq=0.0,
        delta_q=0.0,
        decode_expected=0.1,
        decode_unexpected=0.2,
        mean_rate_expected=mean_rate_a,
        mean_rate_unexpected=mean_rate_b,
        rate_reference=r_ref,
        epsilon_rate=epsilon_rate,
        phase_y=phase_y,
        stored_saving=-phase_y,
    )

    rates = plotter.measurement_json(measurement)["rates"]
    assert rates["rate_reference"] == r_ref
    assert rates["epsilon_rate"] == 1e-8 * plotter.assay.N * r_ref
    assert rates[plotter.PHASE_Y_FIELD] == (
        (rates["continuation_A_mean"] - rates["OOD_reversal_B_mean"])
        / (rates["OOD_reversal_B_mean"] + rates["epsilon_rate"])
    )
    assert rates[plotter.PHASE_Y_FIELD] == -rates[plotter.STORED_SAVING_FIELD]
    assert "phase_y_A_minus_B_over_B" not in rates
    assert "stored_saving_B_minus_A_over_B" not in rates

    phase_space = plotter.aggregate_arm([measurement] * 4)["phase_space"]
    assert plotter.PHASE_Y_FIELD in phase_space
    assert plotter.STORED_SAVING_FIELD in phase_space
    assert phase_space["rate_reference"]["mean"] == r_ref
    assert phase_space["epsilon_rate"]["mean"] == epsilon_rate
    assert "y_mean_rate_A_minus_B_over_B" not in phase_space
    assert "stored_saving_B_minus_A_over_B" not in phase_space

    source = Path(plotter.__file__).read_text(encoding="utf-8")
    assert '"formula": "1e-8 * N * R_ref"' in source
    assert '"R_ref_source": "checkpoint.references.R_ref"' in source
    assert "(mean_rate_B + epsilon_rate)" in source
