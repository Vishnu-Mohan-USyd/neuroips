#!/usr/bin/env python3
"""Oracle predictor test: bypass V2 with perfect predictions.

Diagnostic experiment to test whether feedback mechanisms work correctly
when given ground-truth predictions. This isolates feedback learning from
V2 prediction quality.

The oracle injects:
  - q_pred: circular Gaussian centered on true next orientation (sigma=10 deg)
  - pi_pred: high fixed precision (e.g., 3.0)

If feedback mechanisms can learn suppression/sharpening with oracle input,
the bottleneck is V2 prediction, not the feedback pathway.

Usage:
    python -m scripts.oracle_test --mechanism dampening
    python -m scripts.oracle_test --mechanism center_surround --steps 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainingConfig, StimulusConfig, MechanismType, load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.losses import CompositeLoss
from src.training.stage1_sensory import run_stage1
from src.training.trainer import (
    freeze_stage1,
    unfreeze_stage2,
    create_stage2_optimizer,
    make_warmup_cosine_scheduler,
    build_stimulus_sequence,
    compute_readout_indices,
    extract_readout_data,
)
from src.utils import circular_distance_abs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_oracle_q_pred(true_next_thetas: torch.Tensor, n_orientations: int,
                       sigma: float = 10.0, period: float = 180.0) -> torch.Tensor:
    """Build circular Gaussian q_pred centered on true next orientation.

    Args:
        true_next_thetas: [B] — true next orientations in degrees.
        n_orientations: Number of orientation channels.
        sigma: Width of the circular Gaussian in degrees.
        period: Orientation period (180 for orientation).

    Returns:
        q_pred: [B, N] — normalised probability distribution.
    """
    step = period / n_orientations
    preferred = torch.arange(n_orientations, device=true_next_thetas.device).float() * step
    # [B, 1] vs [1, N]
    dists = torch.abs(true_next_thetas.unsqueeze(-1) - preferred.unsqueeze(0))
    dists = torch.min(dists, period - dists)
    q = torch.exp(-dists ** 2 / (2 * sigma ** 2))
    q = q / q.sum(dim=-1, keepdim=True)
    return q


def run_oracle_test(
    mechanism: MechanismType,
    n_steps: int = 5000,
    oracle_pi: float = 3.0,
    device: torch.device | None = None,
    seed: int = 42,
    config_path: str = "config/defaults.yaml",
) -> dict:
    """Run oracle predictor test for one mechanism.

    Trains Stage 1 (sensory scaffold), then runs Stage 2 with oracle
    predictions instead of V2 output.

    Returns dict with training metrics.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    model_cfg = ModelConfig(
        **{k: v for k, v in model_cfg.__dict__.items() if k != "mechanism"},
        mechanism=mechanism,
    )

    N = model_cfg.n_orientations
    net = LaminarV1V2Network(model_cfg)

    # Stage 1: sensory scaffold
    logger.info(f"[{mechanism.value}] Stage 1: sensory scaffold")
    result1 = run_stage1(net, model_cfg, train_cfg, dev, seed)
    logger.info(f"[{mechanism.value}] Stage 1 done: acc={result1.decoder_accuracy:.3f}")

    # Setup Stage 2
    loss_fn = CompositeLoss(train_cfg, model_cfg)
    net = net.to(dev)
    loss_fn = loss_fn.to(dev)
    freeze_stage1(net)
    unfreeze_stage2(net)

    optimizer = create_stage2_optimizer(net, loss_fn, train_cfg)
    scheduler = make_warmup_cosine_scheduler(
        optimizer, train_cfg.stage2_warmup_steps, n_steps
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    hmm_gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=model_cfg.orientation_range,
        contrast_range=train_cfg.stage2_contrast_range,
        ambiguous_fraction=train_cfg.ambiguous_fraction,
    )

    batch_size = train_cfg.batch_size
    seq_length = train_cfg.seq_length
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi

    readout_indices = compute_readout_indices(
        seq_length, steps_on, steps_isi,
        window_start=4, window_end=7,
    )

    # Enable oracle mode
    net.oracle_mode = True

    # Curriculum: start with full feedback (no burn-in needed — oracle is perfect)
    net.feedback_scale.fill_(1.0)

    loss_history = []
    metrics_log = []

    logger.info(f"[{mechanism.value}] Stage 2 ORACLE: {n_steps} steps, pi={oracle_pi}")

    for step in range(n_steps):
        optimizer.zero_grad()

        metadata = hmm_gen.generate(batch_size, seq_length, gen)
        stim_seq, cue_seq, task_seq, true_thetas, true_next_thetas, true_states = (
            build_stimulus_sequence(metadata, model_cfg, train_cfg)
        )
        stim_seq = stim_seq.to(dev, non_blocking=True)
        cue_seq = cue_seq.to(dev, non_blocking=True)
        task_seq = task_seq.to(dev, non_blocking=True)
        true_thetas = true_thetas.to(dev, non_blocking=True)
        true_next_thetas = true_next_thetas.to(dev, non_blocking=True)
        true_states = true_states.to(dev, non_blocking=True)

        packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
        T_total = seq_length * steps_per

        # Pre-compute oracle predictions for entire sequence: [B, T_total, N]
        # Expand true_next_thetas to per-timestep: [B, S] -> [B, T_total]
        true_next_expanded = true_next_thetas.unsqueeze(2).expand(
            -1, -1, steps_per
        ).reshape(batch_size, -1)  # [B, T_total]

        # Build [B, T_total, N] oracle q_pred sequence
        oracle_q_seq = make_oracle_q_pred(
            true_next_expanded.reshape(-1), N, sigma=10.0
        ).reshape(batch_size, T_total, N)

        oracle_pi_seq = torch.full(
            (batch_size, T_total, 1), oracle_pi, device=dev
        )

        # Set sequence oracle mode and run standard forward()
        net.oracle_q_pred = oracle_q_seq
        net.oracle_pi_pred = oracle_pi_seq

        r_l23_all, final_state, aux = net(packed)

        # Build outputs dict matching Stage 2 format
        outputs = {
            "r_l23": r_l23_all,
            "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"],
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
            "state_logits": aux["state_logits_all"],
        }

        # Extract readout windows
        r_l23_windows, q_pred_windows, state_logits_windows = extract_readout_data(
            outputs, readout_indices
        )

        # Compute loss
        loss, loss_dict = loss_fn(
            outputs, true_thetas, true_next_thetas,
            r_l23_windows, q_pred_windows,
            state_logits_windows=state_logits_windows,
            true_states_windows=true_states,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.gradient_clip)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                # Sensory accuracy (same method as stage2_feedback.py)
                logits = loss_fn.orientation_decoder(r_l23_windows)
                B_W = logits.shape[0] * logits.shape[1]
                s_acc = (
                    logits.reshape(B_W, N).argmax(dim=-1)
                    == loss_fn._theta_to_channel(true_thetas).reshape(-1)
                ).float().mean().item()

                # Extract r_l4 at readout windows for suppression measurement
                steps_per_pres = steps_on + steps_isi
                r_l4_all = outputs["r_l4"]
                r_l4_reshaped = r_l4_all.reshape(batch_size, seq_length, steps_per_pres, N)
                _, ts_first = readout_indices[0]
                w_start = ts_first[0]
                w_end = ts_first[-1] + 1
                r_l4_windows = r_l4_reshaped[:, :, w_start:w_end].mean(dim=2)

                # Measure suppression: compare L2/3 peak to L4 peak
                l23_peak = r_l23_windows.max(dim=-1).values.mean().item()
                l4_peak = r_l4_windows.max(dim=-1).values.mean().item()
                suppression = 1.0 - (l23_peak / (l4_peak + 1e-8))

                # SOM activity
                som_mean = outputs["r_som"].mean().item()

            metrics = {
                "step": step + 1,
                "loss": loss.item(),
                "s_acc": s_acc,
                "suppression": suppression,
                "l23_peak": l23_peak,
                "l4_peak": l4_peak,
                "som_mean": som_mean,
            }
            metrics_log.append(metrics)

            logger.info(
                f"[{mechanism.value}] step {step+1}/{n_steps}: "
                f"loss={loss.item():.4f}, s_acc={s_acc:.3f}, "
                f"suppression={suppression:.3f}, som={som_mean:.4f}"
            )

    # Disable oracle mode
    net.oracle_mode = False

    return {
        "mechanism": mechanism.value,
        "loss_history": loss_history,
        "metrics_log": metrics_log,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "final_s_acc": metrics_log[-1]["s_acc"] if metrics_log else 0.0,
        "final_suppression": metrics_log[-1]["suppression"] if metrics_log else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Oracle predictor diagnostic test")
    parser.add_argument("--mechanism", type=str, required=True,
                        choices=[m.value for m in MechanismType],
                        help="Feedback mechanism to test")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of Stage 2 steps (default: 5000)")
    parser.add_argument("--pi", type=float, default=3.0,
                        help="Oracle precision value (default: 3.0)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/defaults.yaml")
    parser.add_argument("--output", type=str, default="checkpoints/oracle",
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mechanism = MechanismType(args.mechanism)

    logger.info(f"Oracle test: mechanism={mechanism.value}, steps={args.steps}, "
                f"pi={args.pi}, device={device}")

    results = run_oracle_test(
        mechanism=mechanism,
        n_steps=args.steps,
        oracle_pi=args.pi,
        device=device,
        seed=args.seed,
        config_path=args.config,
    )

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"oracle_{mechanism.value}_seed{args.seed}.pt"
    torch.save(results, out_path)
    logger.info(f"Results saved to {out_path}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"ORACLE TEST SUMMARY: {mechanism.value}")
    logger.info(f"{'='*60}")
    logger.info(f"  Final loss:        {results['final_loss']:.4f}")
    logger.info(f"  Final s_acc:       {results['final_s_acc']:.3f}")
    logger.info(f"  Final suppression: {results['final_suppression']:.3f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
