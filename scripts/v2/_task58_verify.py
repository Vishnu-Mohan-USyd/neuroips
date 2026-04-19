"""Task #58 verification (scratch driver).

Reports for the DM to Lead:
- w_qm_task_max_magnitude_at_init
- w_lm_task_max_magnitude_at_init
- w_mh_task_max_magnitude_at_init
- null_control_kok_delta_amplitude  (pooled mean over Kok classes)
- null_control_kok_within_sem        (pooled passed flag)

Uses the same micro-budget harness as the null-control smoke test so this
can run on a fresh, untrained V2 network in ~a few seconds on CPU.
"""
from __future__ import annotations

from pathlib import Path
import json
import tempfile

import torch

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from scripts.v2._gates_common import load_checkpoint
from scripts.v2.run_null_expectation_control import run_kok_null


def main() -> None:
    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42, device="cpu")
    net.eval()

    w_qm_max = float(net.context_memory.W_qm_task.abs().max().item())
    w_lm_max = float(net.context_memory.W_lm_task.abs().max().item())
    w_mh_max = float(net.context_memory.W_mh_task.abs().max().item())
    print(f"w_qm_task_max_magnitude_at_init = {w_qm_max:.6e}")
    print(f"w_lm_task_max_magnitude_at_init = {w_lm_max:.6e}")
    print(f"w_mh_task_max_magnitude_at_init = {w_mh_max:.6e}")

    # Persist checkpoint so run_kok_null's load_checkpoint API is happy.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "task58_init_ckpt.pt"
        torch.save({
            "step": 0, "state_dict": net.state_dict(),
            "phase": "phase3_kok",
            "frozen_sha": net.frozen_sensory_core_sha(),
        }, ckpt_path)
        bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
        bundle.net.set_phase("phase3_kok")

        out = run_kok_null(
            bundle, n_trials=40, n_cells=4,
            n_steps_cue=8, n_steps_delay=22, n_steps_probe=20,
            orientation_a_deg=45.0, orientation_b_deg=135.0,
            noise_std=0.01, amplitude_tolerance_sem=1.0, seed=42,
        )

    # Report pooled pass/fail and mean |Δ| across the two Kok classes.
    per_class = out["per_class"]
    deltas = [per_class[k]["delta_amplitude"] for k in per_class]
    delta_mean = float(sum(deltas) / len(deltas))
    passed_all = bool(out["passed"])
    print(f"null_control_kok_delta_amplitude = {delta_mean:.6e}")
    print(f"null_control_kok_within_sem      = {passed_all}")
    # Show the per-class detail for sanity.
    print("null_control_kok_per_class_json  =", json.dumps(per_class, sort_keys=True))


if __name__ == "__main__":
    main()
