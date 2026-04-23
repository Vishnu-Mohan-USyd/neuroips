"""Smoke tests for the Phase-2 Gate-7 C-load-bearing harness.

Exercises :func:`ablate_context_memory`, :func:`measure_prediction_mse`,
and :func:`run_gate_7_c_load_bearing` on an untrained network. Verifies:

* The ablation context manager zeros ``W_mh_gen`` / ``W_mh_task_exc`` /
  ``W_mh_task_inh`` inside the ``with`` block and restores them on exit.
* The MSE measurement produces a finite non-negative float.
* Aggregate gate returns a dict with the expected keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.run_c_load_bearing_check import (
    ablate_context_memory, main as c_lb_main,
    measure_prediction_mse, run_gate_7_c_load_bearing,
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


def test_ablation_zeros_inside_and_restores_on_exit(untrained_bundle):
    cm = untrained_bundle.net.context_memory
    orig_gen = cm.W_mh_gen.data.detach().clone()
    orig_task_exc = cm.W_mh_task_exc.data.detach().clone()
    orig_task_inh = cm.W_mh_task_inh.data.detach().clone()
    assert torch.any(orig_gen != 0.0), "W_mh_gen is zero at init (precondition broken)"
    with ablate_context_memory(untrained_bundle):
        assert torch.all(cm.W_mh_gen.data == 0.0)
        assert torch.all(cm.W_mh_task_exc.data == 0.0)
        assert torch.all(cm.W_mh_task_inh.data == 0.0)
    assert torch.equal(cm.W_mh_gen.data, orig_gen)
    assert torch.equal(cm.W_mh_task_exc.data, orig_task_exc)
    assert torch.equal(cm.W_mh_task_inh.data, orig_task_inh)


def test_measure_prediction_mse_is_finite_nonneg(untrained_bundle):
    mse = measure_prediction_mse(
        untrained_bundle, n_trajectories=2, n_steps_per_traj=4,
    )
    assert isinstance(mse, float) and mse >= 0.0
    assert mse == mse and mse != float("inf")


def test_gate_7_returns_structured_dict(untrained_bundle):
    out = run_gate_7_c_load_bearing(
        untrained_bundle, n_trajectories=2, n_steps_per_traj=4,
    )
    assert out["gate"] == "7_c_load_bearing"
    for k in (
        "n_trajectories", "n_steps_per_traj", "mse_with_c", "mse_without_c",
        "relative_degradation", "degradation_floor", "passed",
    ):
        assert k in out, f"missing key {k!r}"
    assert isinstance(out["passed"], bool)
    assert out["mse_with_c"] >= 0.0 and out["mse_without_c"] >= 0.0


def test_cli_main_writes_json(cfg, tmp_path):
    path = _write_fresh_checkpoint(cfg, tmp_path)
    out_path = tmp_path / "gate_7.json"
    rc = c_lb_main([
        "--checkpoint", str(path), "--seed", "42", "--device", "cpu",
        "--n-trajectories", "2", "--n-steps-per-traj", "4",
        "--output", str(out_path),
    ])
    assert rc in (0, 1)
    payload = json.loads(out_path.read_text())
    assert payload["gate"] == "7_c_load_bearing"


def test_mse_unchanged_after_ablation_restore(untrained_bundle):
    mse_before = measure_prediction_mse(
        untrained_bundle, n_trajectories=2, n_steps_per_traj=4,
    )
    with ablate_context_memory(untrained_bundle):
        _ = measure_prediction_mse(
            untrained_bundle, n_trajectories=2, n_steps_per_traj=4,
        )
    mse_after = measure_prediction_mse(
        untrained_bundle, n_trajectories=2, n_steps_per_traj=4,
    )
    assert mse_after == pytest.approx(mse_before, rel=1e-10, abs=1e-10)
