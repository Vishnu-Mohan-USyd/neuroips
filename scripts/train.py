#!/usr/bin/env python3
"""Entry point for training the laminar V1-V2 model.

Usage:
    python -m scripts.train                          # full pipeline, default config
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

from src.config import ModelConfig, TrainingConfig, StimulusConfig, load_config
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
    parser.add_argument("--allow-gating-fail", action="store_true",
                        help="Allow proceeding to Stage 2 even if Stage 1 gating fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility guard: Stage 2 alone requires a Stage 1 checkpoint.
    if args.stage == 2 and not args.stage1_checkpoint:
        raise ValueError(
            "Stage 2 training requires a Stage 1 checkpoint. "
            "Pass --stage1-checkpoint PATH to load a trained Stage 1 scaffold, "
            "or run the full pipeline without --stage to train both stages."
        )

    # Load config
    model_cfg, train_cfg, stim_cfg = load_config(args.config)

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

    logger.info(f"Feedback mode: {model_cfg.feedback_mode}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")

    # Output directory
    out_dir = Path(args.output) / f"emergent_seed{args.seed}"
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

        # Transfer trained decoder to the shared loss_fn
        if result1.decoder_state_dict is not None:
            loss_fn.orientation_decoder.load_state_dict(result1.decoder_state_dict)
            logger.info("Transferred trained decoder from Stage 1 to shared loss_fn")

        logger.info(f"Stage 1 complete: loss={result1.final_loss:.4f}, "
                     f"acc={result1.decoder_accuracy:.3f}")
        logger.info(f"Gating checks: {result1.gating_passed}")

        # Save Stage 1 checkpoint.
        # `loss_state` captures the *full* CompositeLoss module (including
        # `orientation_decoder`, `mismatch_head`, and any other auxiliary
        # heads allocated by the lambdas in train_cfg). This is the fix for
        # the Task #6 latent bug: previously only `decoder_state` was saved,
        # so any downstream consumer that re-instantiated CompositeLoss and
        # tried to use mismatch_head got randomly-initialised garbage. The
        # standalone `decoder_state` key is retained for backward compat
        # with older eval scripts that read it directly.
        ckpt_path = out_dir / "stage1_checkpoint.pt"
        torch.save({
            "model_state": net.state_dict(),
            "decoder_state": loss_fn.orientation_decoder.state_dict(),
            "loss_state": loss_fn.state_dict(),
            "gating": result1.gating_passed,
            "config": {"model": vars(model_cfg), "training": vars(train_cfg)},
        }, ckpt_path)
        logger.info(f"Stage 1 checkpoint saved to {ckpt_path}")

        all_passed = all(result1.gating_passed.values())
        if not all_passed:
            failed = [k for k, v in result1.gating_passed.items() if not v]
            if args.allow_gating_fail:
                logger.warning(f"Gating checks failed: {failed}. Proceeding (--allow-gating-fail).")
            else:
                raise RuntimeError(
                    f"Stage 1 gating failed: {failed}. "
                    "Use --allow-gating-fail to override."
                )

    # Load Stage 1 checkpoint if doing Stage 2 only.
    # Prefer the full `loss_state` dict (includes all CompositeLoss heads
    # like mismatch_head). Fall back to standalone `decoder_state` for
    # backward compat with older Stage 1 checkpoints. `strict=False` lets
    # us load Stage 1 checkpoints whose head shapes differ slightly from
    # the current CompositeLoss config (e.g. lambdas changed across runs).
    if args.stage == 2:
        ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        if "loss_state" in ckpt:
            loss_fn.load_state_dict(ckpt["loss_state"], strict=False)
        elif "decoder_state" in ckpt:
            loss_fn.orientation_decoder.load_state_dict(ckpt["decoder_state"])
        logger.info(f"Loaded Stage 1 checkpoint from {args.stage1_checkpoint}")

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
                "loss_state": loss_fn.state_dict(),
                "step": step,
                "config": {"model": vars(model_cfg), "training": vars(train_cfg)},
            }, ckpt_path)
            logger.info(f"Intermediate checkpoint saved: {ckpt_path}")

        result2 = run_stage2(
            net, loss_fn, model_cfg, train_cfg, stim_cfg, device, args.seed,
            checkpoint_fn=save_intermediate if ckpt_steps else None,
            checkpoint_steps=ckpt_steps,
            output_dir=str(out_dir),
        )

        logger.info(f"Stage 2 complete: loss={result2.final_loss:.4f}, "
                     f"s_acc={result2.final_sensory_acc:.3f}, "
                     f"p_acc={result2.final_pred_acc:.3f}")

        # Save final checkpoint. Includes the full loss_fn state_dict so
        # the trained mismatch_head (and any other heads) can be reloaded
        # for downstream evaluation. See Task #6 / Fix A.
        ckpt_path = out_dir / "checkpoint.pt"
        torch.save({
            "model_state": net.state_dict(),
            "decoder_state": loss_fn.orientation_decoder.state_dict(),
            "loss_state": loss_fn.state_dict(),
            "history": {"loss": result2.loss_history},
            "config": {"model": vars(model_cfg), "training": vars(train_cfg)},
        }, ckpt_path)
        logger.info(f"Saved to {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
