"""Task #73 Kok-at-15° rerun — fresh Phase-3 with cue_mapping={0:15°, 1:75°}.

Motivation (from H5, confirmed by Dx58 smoke test localizer histogram):
  L2/3 preferred-orientation distribution is clustered at 15° (≈202 units)
  and 75° (≈54 units); zero units at 45°/135°. The default cue_mapping
  ({0:135, 1:45}) thus evaluates anchor pref/nonpref splits on a population
  with no preferred units at those anchors → Task#72 null may be caused by
  coverage gap, not by template-replay per se.

Decisive test: retrain Phase-3 with orientations where pref units exist
(15° and 75°). If Δamp / Δsvm / asymmetry emerge at 15° → coverage gap
explains Task#72 null. If still null at 15° → template-replay also fails
on populations that DO have pref units.

Runs:
  1. Load fresh post-Phase-2 Task#70 ckpt (W_mh_task zero).
  2. Phase-3 learning + scan with cue_mapping={0:15, 1:75}.
  3. Save new ckpt at checkpoints/v2/phase3_kok_task73_at15/phase3_kok_s42.pt.
  4. Run standard Kok eval on the retrained ckpt.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import argparse
import json
import torch
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from scripts.v2.train_phase3_kok_learning import (
    run_phase3_kok_training, _save_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase2-ckpt", type=Path,
                   default=ROOT / "checkpoints/v2/phase2/phase2_task70_s42/"
                   "phase2_s42/step_3000.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials-learning", type=int, default=5000)
    p.add_argument("--n-trials-scan", type=int, default=10000)
    p.add_argument("--validity-scan", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--cue0-orient", type=float, default=15.0)
    p.add_argument("--cue1-orient", type=float, default=75.0)
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "checkpoints/v2/phase3_kok_task73_at15")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    cfg = ModelConfig(seed=int(args.seed), device=args.device)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=int(args.seed),
                    device=args.device)
    ckpt = torch.load(args.phase2_ckpt, map_location=args.device,
                      weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.set_phase("phase3_kok")

    cue_mapping = {0: float(args.cue0_orient), 1: float(args.cue1_orient)}
    print(f"[Kok-at-15] cue_mapping={cue_mapping}", file=sys.stderr,
          flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"phase3_kok_s{int(args.seed)}.pt"
    metrics_path = (args.out_dir /
                    f"phase3_kok_s{int(args.seed)}_metrics.jsonl")
    run_phase3_kok_training(
        net=net,
        n_trials_learning=int(args.n_trials_learning),
        n_trials_scan=int(args.n_trials_scan),
        validity_scan=float(args.validity_scan),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        noise_std=float(args.noise_std),
        cue_mapping=cue_mapping,
        metrics_path=metrics_path,
        log_every=int(args.log_every),
    )
    _save_checkpoint(
        net, args.n_trials_learning + args.n_trials_scan,
        out_path, cue_mapping,
    )
    print(f"[Kok-at-15] checkpoint saved to {out_path}", file=sys.stderr,
          flush=True)
    print(json.dumps({
        "ckpt": str(out_path),
        "cue_mapping": cue_mapping,
        "n_trials_learning": int(args.n_trials_learning),
        "n_trials_scan": int(args.n_trials_scan),
    }))


if __name__ == "__main__":
    main()
