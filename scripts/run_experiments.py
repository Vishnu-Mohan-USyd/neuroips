#!/usr/bin/env python3
"""Load a trained model and run all experimental paradigms.

Usage:
    python -m scripts.run_experiments --checkpoint path/to/model.pt
    python -m scripts.run_experiments --checkpoint path/to/model.pt --paradigm hidden_state
    python -m scripts.run_experiments --checkpoint path/to/model.pt --n-trials 50 --output results.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, MechanismType
from src.model.network import LaminarV1V2Network
from src.experiments import ALL_PARADIGMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental paradigms")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--mechanism", type=str, default="center_surround",
                        choices=[m.value for m in MechanismType],
                        help="Mechanism type (must match checkpoint)")
    parser.add_argument("--paradigm", type=str, default=None,
                        choices=list(ALL_PARADIGMS.keys()),
                        help="Run a single paradigm (default: all)")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Trials per condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Reconstruct ModelConfig from checkpoint's saved config, not defaults
    if "config" in ckpt and "model" in ckpt["config"]:
        model_raw = ckpt["config"]["model"]
        cfg = ModelConfig(**model_raw)
        logger.info(f"Loaded config from checkpoint: mechanism={cfg.mechanism.value}")
    else:
        # Fallback for legacy checkpoints without saved config
        cfg = ModelConfig(mechanism=MechanismType(args.mechanism))
        logger.warning("No config in checkpoint, using --mechanism flag")

    net = LaminarV1V2Network(cfg)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    paradigms = ALL_PARADIGMS
    if args.paradigm:
        paradigms = {args.paradigm: paradigms[args.paradigm]}

    results = {}
    for name, ParadigmClass in paradigms.items():
        logger.info(f"Running {name}...")
        paradigm = ParadigmClass(net, cfg)
        result = paradigm.run(
            n_trials=args.n_trials, seed=args.seed, batch_size=args.batch_size)
        results[name] = result

        n_cond = len(result.conditions)
        total = sum(c.r_l23.shape[0] for c in result.conditions.values())
        logger.info(f"  {n_cond} conditions, {total} total trials")

    if args.output:
        torch.save(results, args.output)
        logger.info(f"Results saved to {args.output}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
