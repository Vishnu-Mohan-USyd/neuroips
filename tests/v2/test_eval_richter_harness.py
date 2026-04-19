"""Smoke tests for ``scripts.v2.eval_richter``.

Runs :func:`evaluate_richter` on an untrained Phase-3-Richter bundle with
a micro-budget (1 trial per condition, 6×6 grid, 2-step epochs, 8
pseudo-voxels per model). Verifies the returned dict has the keys
required by the Task #40 spec: ``assay``, ``amplitude_summary``,
``rsa_within_between``, ``preference_rank_curve`` and
``pseudo_voxel_models`` (6 model entries).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_richter import evaluate_richter, main as eval_main
from scripts.v2.train_phase3_richter_learning import (
    N_LEAD_TRAIL, RichterTiming,
)
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_phase3_richter_checkpoint(
    cfg, tmp_path: Path, *, seed: int = 42,
) -> Path:
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase("phase3_richter")
    net.eval()
    path = tmp_path / "phase3_richter_step_0.pt"
    torch.save({
        "step": 0, "state_dict": net.state_dict(),
        "phase": "phase3_richter", "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path


@pytest.fixture
def untrained_richter_bundle(cfg, tmp_path):
    return load_checkpoint(
        _write_phase3_richter_checkpoint(cfg, tmp_path),
        seed=42, device="cpu",
    )


def _tiny_timing() -> RichterTiming:
    return RichterTiming(leader_steps=2, trailer_steps=2, iti_steps=0)


EXPECTED_MODEL_KEYS = {
    "local_gain", "local_tuning",
    "remote_gain", "remote_tuning",
    "global_gain", "global_tuning",
}


def test_evaluate_richter_returns_structured_dict(untrained_richter_bundle):
    """evaluate_richter returns the Task-#40 metrics schema."""
    out = evaluate_richter(
        untrained_richter_bundle,
        n_trials_per_condition=1, noise_std=0.0,
        n_pseudo_voxels_per_model=8,
        seed=42, timing=_tiny_timing(),
    )
    assert out["assay"] == "eval_richter"
    assert out["n_trials_per_condition"] == 1
    assert len(out["permutation"]) == N_LEAD_TRAIL

    amp = out["amplitude_summary"]
    assert len(amp["per_condition_mean_unit0"]) == N_LEAD_TRAIL * N_LEAD_TRAIL
    assert isinstance(amp["grand_mean"], float)
    assert isinstance(amp["grand_std"], float)

    rsa = out["rsa_within_between"]
    assert isinstance(rsa, dict)
    # ≥ 4 trials → RSA reports distances + pair counts; otherwise 'error'.
    assert (
        ("within_mean_distance" in rsa and "between_mean_distance" in rsa)
        or ("error" in rsa)
    )

    pref = out["preference_rank_curve"]
    assert isinstance(pref, dict)
    # Canonical keys from _preference_rank_curve: ranks + mean_rate + sem_rate.
    assert ("mean_rate" in pref) or ("error" in pref)

    models = out["pseudo_voxel_models"]
    assert set(models.keys()) == EXPECTED_MODEL_KEYS, (
        f"expected the six {{local,remote,global}}×{{gain,tuning}} models; "
        f"got {sorted(models.keys())}"
    )


def test_cli_main_writes_json(cfg, tmp_path):
    path = _write_phase3_richter_checkpoint(cfg, tmp_path)
    out_path = tmp_path / "eval_richter.json"
    rc = eval_main([
        "--checkpoint", str(path), "--seed", "42", "--device", "cpu",
        "--n-trials-per-condition", "1", "--n-pseudo-voxels", "8",
        "--noise-std", "0.0", "--output", str(out_path),
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["assay"] == "eval_richter"
    assert "amplitude_summary" in payload
    assert "rsa_within_between" in payload
    assert "preference_rank_curve" in payload
    assert set(payload["pseudo_voxel_models"].keys()) == EXPECTED_MODEL_KEYS
