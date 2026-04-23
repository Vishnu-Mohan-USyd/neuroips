"""Probe magnitude of cue drive into memory vs other drives (H-W mechanism)."""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
    make_blank_frame,
)


def main():
    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42)
    net.set_phase("phase3_kok")
    blob = torch.load("checkpoints/v2/phase3_kok/phase3_kok_s42.pt",
                      map_location="cpu", weights_only=False)
    net.load_state_dict(blob["state_dict"], strict=False)
    cm = net.context_memory
    cue_mapping = blob.get("cue_mapping", cue_mapping_from_seed(42))

    timing = KokTiming()
    cue_end = timing.cue_steps
    rows = []

    for cue_id in (0, 1):
        q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
        blank = make_blank_frame(1, cfg, device="cpu")
        state = net.initial_state(batch_size=1)
        with torch.no_grad():
            for t in range(cue_end):
                # Step through cue period
                _x, state, info = net(blank, state, q_t=q_cue)
                net.l23_e.homeostasis.update(state.r_l23)
                net.h_e.homeostasis.update(state.r_h)
                # After step: compute drive components at the *next* step's
                # state.
                drive_hm = F.linear(state.r_h, cm.W_hm_gen)
                drive_mm = F.linear(state.m, cm.W_mm_gen)
                drive_qm = F.linear(q_cue, cm.W_qm_task)
                if t in (0, 1, 5, 10, 20, 39):
                    rows.append({
                        "cue_id": cue_id, "t": t,
                        "m_norm": float(state.m.norm().item()),
                        "h_norm": float(state.r_h.norm().item()),
                        "drive_hm_norm": float(drive_hm.norm().item()),
                        "drive_mm_norm": float(drive_mm.norm().item()),
                        "drive_qm_norm": float(drive_qm.norm().item()),
                    })
    # Summary: ratio of cue-drive to total drive
    import numpy as np
    by_cue = {0: [], 1: []}
    for r in rows:
        total = (r["drive_hm_norm"]**2 + r["drive_mm_norm"]**2
                 + r["drive_qm_norm"]**2) ** 0.5
        r["cue_drive_fraction_of_l2"] = r["drive_qm_norm"] / max(total, 1e-12)
        by_cue[r["cue_id"]].append(r)
    print(json.dumps({"rows": rows,
                      "W_qm_task_col0_norm": float(
                          cm.W_qm_task.data[:, 0].norm().item()),
                      "W_qm_task_col1_norm": float(
                          cm.W_qm_task.data[:, 1].norm().item()),
                      "W_hm_gen_norm": float(cm.W_hm_gen.data.norm().item()),
                      "W_mm_gen_norm": float(cm.W_mm_gen.data.norm().item()),
                      }, indent=2))


if __name__ == "__main__":
    main()
