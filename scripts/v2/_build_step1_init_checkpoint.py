"""Task #74 Step 1: build a Fix-E INIT checkpoint (no training).

Freshly constructs V2Network (seed=42) with Fix E applied (W_l23_som_raw
init_mean=-4.5 — already in src/v2_model/layers.py), sets phase to the
Kok phase for the save and injects a synthetic ``cue_mapping`` of
{0: 45.0, 1: 135.0} into meta so eval_kok / eval_richter can tag probes
without any Phase-2 or Phase-3 training.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def main() -> None:
    seed = 42
    cfg = ModelConfig(seed=seed, device="cpu")
    torch.manual_seed(seed)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase("phase3_kok")

    out = Path("checkpoints/v2/step1_fixE_init.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "state_dict": net.state_dict(),
            "phase": "phase3_kok",
            "frozen_sha": net.frozen_sensory_core_sha(),
            "cue_mapping": {0: 45.0, 1: 135.0},
            "notes": "Task #74 Step 1 init baseline; Fix E only; no training.",
        },
        out,
    )
    print(f"[step1] wrote {out}")


if __name__ == "__main__":
    main()
