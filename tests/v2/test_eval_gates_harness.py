"""Smoke tests for Phase-2 Gates 1-5 harness.

Checkpoints a freshly-initialised :class:`V2Network` (untrained) and runs
:func:`run_gates_1_to_5` against it. Pass/fail results are not
meaningful on an untrained network — this only verifies the harness
runs end-to-end and produces the expected JSON shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_gates import (
    gate_1_rate_distribution, gate_2_contrast_response,
    gate_3_surround_suppression, gate_4_prediction_beats_copy_last,
    gate_5a_orientation_localizer, gate_5b_identity_localizer,
    main as eval_gates_main, run_gates_1_to_5,
)
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_fresh_checkpoint(cfg, tmp_path: Path, *, seed: int = 42) -> Path:
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.eval()
    path = tmp_path / "smoke_step_0.pt"
    torch.save({
        "step": 0, "state_dict": net.state_dict(),
        "phase": "phase2", "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path


@pytest.fixture
def untrained_bundle(cfg, tmp_path):
    return load_checkpoint(_write_fresh_checkpoint(cfg, tmp_path), seed=42, device="cpu")


def test_gate_1_returns_rate_distribution_dict(untrained_bundle):
    out = gate_1_rate_distribution(untrained_bundle, n_steps=20, batch_size=2)
    assert out["gate"] == "1_rate_distribution"
    for k in ("median_rate", "acceptable_band", "mean_rate", "std_rate", "passed"):
        assert k in out
    assert isinstance(out["passed"], bool)


def test_gate_2_returns_naka_rushton_fit(untrained_bundle):
    out = gate_2_contrast_response(
        untrained_bundle, contrasts=(0.1, 0.5, 1.0), n_steps_steady=5,
    )
    assert out["gate"] == "2_contrast_response"
    assert "r_squared" in out["fit"]
    assert len(out["responses"]) == 3
    assert isinstance(out["passed"], bool)


def test_gate_3_returns_surround_suppression_fraction(untrained_bundle):
    out = gate_3_surround_suppression(untrained_bundle, n_steps_steady=5)
    assert out["gate"] == "3_surround_suppression"
    assert 0.0 <= out["frac_units_with_si_above_threshold"] <= 1.0
    assert isinstance(out["passed"], bool)


def test_gate_4_returns_pred_vs_copy_metrics(untrained_bundle):
    out = gate_4_prediction_beats_copy_last(
        untrained_bundle, n_trajectories=2, n_steps_per_traj=5,
    )
    assert out["gate"] == "4_prediction_beats_copy_last"
    for k in ("network_mse", "copy_last_mse", "relative_improvement", "passed"):
        assert k in out


def test_gate_5a_returns_fwhm_distribution(untrained_bundle):
    out = gate_5a_orientation_localizer(
        untrained_bundle, n_orientations=4, n_steps_steady=5,
    )
    assert out["gate"] == "5a_orientation_localizer"
    assert 0.0 <= out["frac_units_fwhm_below"] <= 1.0
    assert out["n_orientations"] == 4


def test_gate_5b_returns_identity_localizer_metrics(untrained_bundle):
    out = gate_5b_identity_localizer(
        untrained_bundle, n_noise_samples=4, n_steps_steady=2,
    )
    assert out["gate"] == "5b_identity_localizer"
    if "error" not in out:
        assert 0.0 <= out["svm_accuracy"] <= 1.0
        assert isinstance(out["rsa_passed"], bool)


def test_run_gates_1_to_5_aggregates(untrained_bundle):
    results = run_gates_1_to_5(untrained_bundle)
    expected = {
        "gate_1_rate_distribution", "gate_2_contrast_response",
        "gate_3_surround_suppression",
        "gate_4_next_step_prediction_beats_copy_last",
        "gate_5a_orientation_localizer", "gate_5b_identity_localizer",
        "all_passed",
    }
    assert expected.issubset(results.keys())
    assert isinstance(results["all_passed"], bool)


def test_cli_main_writes_json(cfg, tmp_path):
    path = _write_fresh_checkpoint(cfg, tmp_path)
    out_path = tmp_path / "gates_1_5.json"
    rc = eval_gates_main([
        "--checkpoint", str(path), "--seed", "42",
        "--device", "cpu", "--output", str(out_path),
    ])
    assert rc in (0, 1)
    payload = json.loads(out_path.read_text())
    assert "all_passed" in payload


def test_gate_1_is_deterministic(untrained_bundle):
    a = gate_1_rate_distribution(untrained_bundle, n_steps=20, batch_size=2)
    b = gate_1_rate_distribution(untrained_bundle, n_steps=20, batch_size=2)
    assert a["median_rate"] == pytest.approx(b["median_rate"], rel=0, abs=0)
