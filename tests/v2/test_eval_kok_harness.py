"""Smoke tests for ``scripts.v2.eval_kok``.

Runs :func:`evaluate_kok` on an untrained Phase-3-Kok bundle with a
micro-budget (3 trials per condition, 2-step epochs). Verifies the
returned dictionary has the keys required by the Task #40 spec:
``assay``, ``per_cell_mean_l23``, ``svm`` (with ``all``/``expected``/
``unexpected`` sub-blocks) and ``pref_nonpref``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_kok import evaluate_kok, main as eval_main
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
        "phase": "phase3_kok", "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path


@pytest.fixture
def untrained_kok_bundle(cfg, tmp_path):
    return load_checkpoint(
        _write_phase3_kok_checkpoint(cfg, tmp_path), seed=42, device="cpu",
    )


def _tiny_timing() -> KokTiming:
    return KokTiming(
        cue_steps=2, delay_steps=2, probe1_steps=2,
        blank_steps=1, probe2_steps=2,
    )


def test_evaluate_kok_returns_structured_dict(untrained_kok_bundle):
    """evaluate_kok returns the Task-#40 metrics schema."""
    out = evaluate_kok(
        untrained_kok_bundle,
        n_trials_per_condition=3, n_cell_groups=4, noise_std=0.0,
        seed=42, timing=_tiny_timing(),
    )
    assert out["assay"] == "eval_kok"
    assert out["n_trials_per_condition"] == 3
    assert out["n_cell_groups"] == 4
    assert set(out["cue_mapping"].keys()) == {0, 1}

    # Per-cell means: 3 tags × 4 cell groups.
    per_cell = out["per_cell_mean_l23"]
    assert set(per_cell.keys()) == {"expected", "unexpected", "all"}
    for tag, groups in per_cell.items():
        assert len(groups) == 4, (
            f"per_cell_mean_l23[{tag}] has {len(groups)} groups; expected 4"
        )
        for g in groups:
            assert isinstance(g, float)

    # SVM block present with the three required sub-conditions. With 12
    # trials (3/cond × 4 conds) the 10-sample CV threshold is met so we
    # expect accuracy keys, but accept the documented 'error' path too.
    svm = out["svm"]
    assert set(svm.keys()) == {"all", "expected", "unexpected"}
    for sub in svm.values():
        assert "mean_accuracy" in sub or "error" in sub

    # Preference / non-preference asymmetry must report a numeric summary
    # or the documented error path.
    pn = out["pref_nonpref"]
    assert isinstance(pn, dict)
    assert ("asymmetry" in pn) or ("error" in pn)


def test_cli_main_writes_json(cfg, tmp_path):
    """CLI runs end-to-end on an untrained checkpoint and writes valid JSON.

    Uses ``--n-trials-per-condition 2`` (8 trials total) so the SVM
    block gracefully returns its ``insufficient data`` error path rather
    than attempting LinearSVC on full-length, untrained-network rates
    (which can blow up to NaN over the 370-step default timing).
    """
    path = _write_phase3_kok_checkpoint(cfg, tmp_path)
    out_path = tmp_path / "eval_kok.json"
    rc = eval_main([
        "--checkpoint", str(path), "--seed", "42", "--device", "cpu",
        "--n-trials-per-condition", "2", "--n-cell-groups", "4",
        "--noise-std", "0.0", "--output", str(out_path),
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["assay"] == "eval_kok"
    assert "per_cell_mean_l23" in payload
    assert "svm" in payload
    assert "pref_nonpref" in payload
