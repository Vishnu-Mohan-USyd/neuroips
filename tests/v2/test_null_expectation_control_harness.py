"""Smoke tests for the Phase-2 Gate-6 null-expectation control harness.

Exercises :func:`run_kok_null`, :func:`run_richter_null` and
:func:`run_gate_6_null_control` on an untrained network. Only verifies
the harness runs and returns the expected JSON structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.run_null_expectation_control import (
    main as null_main, run_gate_6_null_control,
    run_kok_null, run_richter_null,
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


def test_run_kok_null_returns_structured_dict(untrained_bundle):
    out = run_kok_null(
        untrained_bundle, n_trials=4, n_cells=1,
        n_steps_cue=2, n_steps_delay=2, n_steps_probe=4, seed=42,
    )
    assert out["assay"] == "kok_null"
    assert isinstance(out["passed"], bool)
    for label in ("A_45deg", "B_135deg"):
        entry = out["per_class"][label]
        for k in (
            "n_trials", "mean_expected", "mean_unexpected",
            "delta_amplitude", "sem_pooled", "passed",
        ):
            assert k in entry, f"missing key {k!r} in kok[{label}]"


def test_run_kok_null_is_deterministic(untrained_bundle):
    kwargs = dict(
        n_trials=4, n_cells=1, n_steps_cue=2,
        n_steps_delay=2, n_steps_probe=3, seed=7,
    )
    a = run_kok_null(untrained_bundle, **kwargs)
    b = run_kok_null(untrained_bundle, **kwargs)
    assert a["per_class"]["A_45deg"]["delta_amplitude"] == pytest.approx(
        b["per_class"]["A_45deg"]["delta_amplitude"], rel=0, abs=0,
    )


def test_run_richter_null_returns_structured_dict(untrained_bundle):
    out = run_richter_null(
        untrained_bundle, n_trials_per_token=6,
        n_steps_lead=2, n_steps_probe=3, seed=42,
    )
    assert out["assay"] == "richter_null"
    assert "per_token_amplitude" in out and "svm" in out
    assert isinstance(out["passed"], bool)
    # SVM block either has accuracy keys (sklearn) or an error field.
    svm = out["svm"]
    assert "passed" in svm
    assert "accuracy_expected" in svm or "error" in svm


def test_run_gate_6_null_control_combines_both(untrained_bundle):
    out = run_gate_6_null_control(
        untrained_bundle,
        kok_kwargs=dict(
            n_trials=4, n_cells=1, n_steps_cue=2,
            n_steps_delay=2, n_steps_probe=3, seed=42,
        ),
        richter_kwargs=dict(
            n_trials_per_token=4, n_steps_lead=2, n_steps_probe=3, seed=42,
        ),
    )
    assert out["phase_switched_to"] == "phase3_kok"
    assert "kok" in out and "richter" in out
    assert isinstance(out["passed"], bool)


def test_cli_main_writes_json(cfg, tmp_path):
    path = _write_fresh_checkpoint(cfg, tmp_path)
    out_path = tmp_path / "gate_6.json"
    rc = null_main([
        "--checkpoint", str(path), "--seed", "42", "--device", "cpu",
        "--kok-trials", "4", "--richter-trials-per-token", "4",
        "--output", str(out_path),
    ])
    assert rc in (0, 1)
    payload = json.loads(out_path.read_text())
    assert "kok" in payload and "richter" in payload
