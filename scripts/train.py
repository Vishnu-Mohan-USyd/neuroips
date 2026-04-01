#!/usr/bin/env python3
"""Entry point for training the laminar V1-V2 model.

Usage:
    python -m scripts.train                          # full pipeline, default config
    python -m scripts.train --mechanism dampening     # specific mechanism
    python -m scripts.train --stage 1                 # Stage 1 only
    python -m scripts.train --stage 2                 # Stage 2 only (needs Stage 1 checkpoint)
    python -m scripts.train --config config/custom.yaml
    python -m scripts.train --stage2-steps 500        # abbreviated Stage 2 for testing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainingConfig, StimulusConfig, MechanismType, load_config
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.training.stage1_sensory import run_stage1
from src.training.stage2_feedback import run_stage2
from src.training.trainer import freeze_stage1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train laminar V1-V2 model")
    parser.add_argument("--config", type=str, default="config/defaults.yaml",
                        help="Path to config YAML")
    parser.add_argument("--mechanism", type=str, default=None,
                        choices=[m.value for m in MechanismType],
                        help="Override mechanism type")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2],
                        help="Run only a specific stage (default: both)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/cuda:0)")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Path to Stage 1 checkpoint (for Stage 2 only)")
    parser.add_argument("--stage2-steps", type=int, default=None,
                        help="Override Stage 2 step count (for testing)")
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Override sequence length (presentations per training sequence)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--checkpoint-at", type=int, nargs="*", default=None,
                        help="Save intermediate checkpoints at these step numbers")
    parser.add_argument("--v2-input", type=str, default=None,
                        choices=['l23', 'l4', 'l4_l23'],
                        help="Override V2 input mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    model_cfg, train_cfg, stim_cfg = load_config(args.config)

    # Override mechanism if specified
    if args.mechanism:
        model_cfg = ModelConfig(
            **{k: v for k, v in model_cfg.__dict__.items() if k != "mechanism"},
            mechanism=MechanismType(args.mechanism),
        )

    # Override V2 input mode if specified
    if args.v2_input:
        model_cfg.v2_input_mode = args.v2_input

    # Override Stage 2 steps if specified
    if args.stage2_steps is not None:
        train_cfg.stage2_n_steps = args.stage2_steps
    if args.seq_length is not None:
        train_cfg.seq_length = args.seq_length
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    logger.info(f"Mechanism: {model_cfg.mechanism.value}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")

    # Output directory
    out_dir = Path(args.output) / f"{model_cfg.mechanism.value}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build network
    net = LaminarV1V2Network(model_cfg)
    logger.info(f"Network parameters: {sum(p.numel() for p in net.parameters())}")

    # Loss function (shared between stages for decoder continuity)
    loss_fn = CompositeLoss(train_cfg, model_cfg)

    # Stage 1
    if args.stage is None or args.stage == 1:
        logger.info("=" * 60)
        logger.info("STAGE 1: Sensory Scaffold")
        logger.info("=" * 60)

        result1 = run_stage1(net, model_cfg, train_cfg, device, args.seed)

        logger.info(f"Stage 1 complete: loss={result1.final_loss:.4f}, "
                     f"acc={result1.decoder_accuracy:.3f}")
        logger.info(f"Gating checks: {result1.gating_passed}")

        # Save Stage 1 checkpoint
        ckpt_path = out_dir / "stage1_checkpoint.pt"
        torch.save({
            "model_state": net.state_dict(),
            "decoder_state": loss_fn.orientation_decoder.state_dict(),
            "gating": result1.gating_passed,
            "config": {"model": vars(model_cfg), "training": vars(train_cfg)},
        }, ckpt_path)
        logger.info(f"Stage 1 checkpoint saved to {ckpt_path}")

        all_passed = all(result1.gating_passed.values())
        if not all_passed:
            logger.warning("Not all gating checks passed! Proceeding anyway.")

    # Load Stage 1 checkpoint if doing Stage 2 only
    if args.stage == 2:
        if args.stage1_checkpoint:
            ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
            net.load_state_dict(ckpt["model_state"])
            if "decoder_state" in ckpt:
                loss_fn.orientation_decoder.load_state_dict(ckpt["decoder_state"])
            logger.info(f"Loaded Stage 1 checkpoint from {args.stage1_checkpoint}")
        else:
            logger.warning("No Stage 1 checkpoint provided for Stage 2. Using random init.")

    # Stage 2
    if args.stage is None or args.stage == 2:
        logger.info("=" * 60)
        logger.info("STAGE 2: V2 + Feedback")
        logger.info("=" * 60)

        freeze_stage1(net)

        # Intermediate checkpoint callback
        ckpt_steps = args.checkpoint_at or []

        def save_intermediate(step: int):
            ckpt_path = out_dir / f"checkpoint_step{step}.pt"
            torch.save({
                "model_state": net.state_dict(),
                "decoder_state": loss_fn.orientation_decoder.state_dict(),
                "step": step,
                "config": {"model": vars(model_cfg), "training": vars(train_cfg)},
            }, ckpt_path)
            logger.info(f"Intermediate checkpoint saved: {ckpt_path}")

        result2 = run_stage2(
            net, loss_fn, model_cfg, train_cfg, stim_cfg, device, args.seed,
            checkpoint_fn=save_intermediate if ckpt_steps else None,
            checkpoint_steps=ckpt_steps,
        )

        logger.info(f"Stage 2 complete: loss={result2.final_loss:.4f}, "
                     f"s_acc={result2.final_sensory_acc:.3f}, "
                     f"p_acc={result2.final_pred_acc:.3f}")

        # Save final checkpoint
        ckpt_path = out_dir / "checkpoint.pt"
        torch.save({
            "model_state": net.state_dict(),
            "decoder_state": loss_fn.orientation_decoder.state_dict(),
            "history": {"loss": result2.loss_history},
            "config": {"model": vars(model_cfg), "training": vars(train_cfg)},
        }, ckpt_path)
        logger.info(f"Saved to {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
