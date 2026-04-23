"""Smoke tests for ``scripts.v2.task74_diagnostics``.

Runs :func:`run_diagnostics` on an untrained Phase-3-Kok bundle using a
micro-budget (3 localizer trials/orient, 2 probe trials/condition) and
the Kok tiny-timing fixture. Verifies that the returned dictionary has
the three headline blocks (coverage, rule_magnitude, readout_alignment)
and that each reports numeric fields of the expected type.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.task74_diagnostics import (
    THRESHOLDS, main as diag_main, run_diagnostics,
)
from scripts.v2.train_phase3_kok_learning import KokTiming
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_phase3_kok_checkpoint(cfg, tmp_path: Path, *, seed: int = 42) -> Path:
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase("phase3_kok")
    net.eval()
    path = tmp_path / "phase3_kok_step_0.pt"
    torch.save({
        "step": 0, "state_dict": net.state_dict(),
        "phase": "phase3_kok",
        "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path


@pytest.fixture
def untrained_kok_ckpt(cfg, tmp_path):
    return _write_phase3_kok_checkpoint(cfg, tmp_path)


def test_run_diagnostics_returns_three_blocks(untrained_kok_ckpt):
    """All three metric blocks present with expected scalar types."""
    out = run_diagnostics(
        untrained_kok_ckpt,
        seed=42,
        n_coverage_trials=3,
        n_probe_per_cond=2,
        n_localizer=3,
        noise_std=0.0,
    )
    assert out["task"] == "task74_diagnostics"
    assert out["seed"] == 42
    assert out["phase"] == "phase3_kok"

    cov = out["coverage"]
    assert cov["n_orientations"] == 12
    assert len(cov["per_orientation_n_pref_units"]) == 12
    assert isinstance(cov["entropy_nats"], float)
    assert isinstance(cov["entropy_bits"], float)
    assert isinstance(cov["n_bins_geq_5pct"], int)
    # Histogram must sum to n_units.
    assert sum(cov["per_orientation_n_pref_units"]) == cov["n_units"]
    # Each pass flag is a bool.
    for flag in ("pass_strong", "pass_weak", "pass_bins"):
        assert isinstance(cov[flag], bool)

    rm = out["rule_magnitude"]
    assert isinstance(rm["bias_added_norm_mean"], float)
    assert isinstance(rm["localizer_norm_mean"], float)
    assert isinstance(rm["ratio"], float)
    assert isinstance(rm["pass_gate"], bool)
    # Expected classes present (2 for Kok).
    assert set(rm["per_class"].keys()) == {"0", "1"}

    ra = out["readout_alignment"]
    assert isinstance(ra["mean_cos_same"], float)
    assert len(ra["per_class_cos_same"]) == 2
    assert isinstance(ra["pass"], bool)

    # Thresholds echoed in the output.
    assert out["thresholds"] == THRESHOLDS


def test_skip_phase3_metrics_still_reports_coverage(untrained_kok_ckpt):
    """With --skip-phase3-metrics, coverage still runs; other blocks skipped."""
    out = run_diagnostics(
        untrained_kok_ckpt,
        seed=42,
        n_coverage_trials=3,
        n_probe_per_cond=2, n_localizer=3,
        noise_std=0.0, skip_phase3_metrics=True,
    )
    assert "entropy_nats" in out["coverage"]
    assert out["rule_magnitude"] == {"skipped": True}
    assert out["readout_alignment"] == {"skipped": True}


def test_cli_main_writes_json(cfg, tmp_path):
    """CLI ``main`` end-to-end: runs on an untrained ckpt, writes valid JSON."""
    path = _write_phase3_kok_checkpoint(cfg, tmp_path)
    out_path = tmp_path / "task74.json"
    rc = diag_main([
        "--checkpoint", str(path),
        "--output", str(out_path),
        "--seed", "42",
        "--device", "cpu",
        "--n-coverage-trials", "3",
        "--n-probe-per-cond", "2",
        "--n-localizer", "3",
        "--noise-std", "0.0",
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["task"] == "task74_diagnostics"
    assert "coverage" in payload
    assert "rule_magnitude" in payload
    assert "readout_alignment" in payload
