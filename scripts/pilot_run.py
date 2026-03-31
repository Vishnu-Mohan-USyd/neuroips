#!/usr/bin/env python3
"""Phase 8A: Pilot science run — A/B/C/D × 1 seed.

Train 4 models (dampening/sharpening/center-surround/adaptation-only),
1 seed each, λ_energy=0.01. Checkpoints at 20K/40K/80K.
Run P1/P2/P3. Run key analyses.
Go/no-go: do mechanisms diverge?

Usage:
    python -m scripts.pilot_run                    # full run (GPU recommended)
    python -m scripts.pilot_run --quick            # smoke test (~100 steps)
    python -m scripts.pilot_run --output results/pilot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainingConfig, StimulusConfig, MechanismType
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.training.stage1_sensory import run_stage1
from src.training.stage2_feedback import run_stage2
from src.training.trainer import freeze_stage1

from src.experiments.hidden_state import HiddenStateParadigm
from src.experiments.omission import OmissionParadigm
from src.experiments.ambiguous import AmbiguousParadigm

from src.analysis.suppression_profile import (
    compute_mean_responses, compute_suppression_profile_from_experiment,
)
from src.analysis.tuning_curves import analyse_tuning_curves
from src.analysis.decoding import cross_validated_decoding, compute_d_prime
from src.analysis.energy import compute_energy
from src.analysis.observation_model import run_observation_model
from src.analysis.omission_analysis import run_omission_analysis
from src.analysis.temporal_analysis import run_temporal_analysis
from src.analysis.v2_probes import run_v2_probes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MECHANISMS = [
    MechanismType.DAMPENING,
    MechanismType.SHARPENING,
    MechanismType.CENTER_SURROUND,
    MechanismType.ADAPTATION_ONLY,
]

CHECKPOINT_STEPS = [20000, 40000, 80000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8A pilot science run")
    parser.add_argument("--output", type=str, default="results/pilot",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if omitted)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (100 stage2 steps)")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Trials per condition for experiments")
    parser.add_argument("--mechanisms", type=str, nargs="+", default=None,
                        help="Subset of mechanisms to run (default: all 4)")
    return parser.parse_args()


def train_model(
    mechanism: MechanismType,
    seed: int,
    device: torch.device,
    out_dir: Path,
    quick: bool = False,
) -> tuple[LaminarV1V2Network, ModelConfig, CompositeLoss, dict]:
    """Train a single model through Stage 1 + Stage 2.

    Returns:
        net, cfg, loss_fn, training_summary
    """
    cfg = ModelConfig(mechanism=mechanism)
    train_cfg = TrainingConfig(lambda_energy=0.01)
    stim_cfg = StimulusConfig()

    if quick:
        train_cfg.stage1_n_steps = 200
        train_cfg.stage2_n_steps = 100
        ckpt_steps = [50, 100]
    else:
        ckpt_steps = CHECKPOINT_STEPS

    torch.manual_seed(seed)
    net = LaminarV1V2Network(cfg)
    loss_fn = CompositeLoss(train_cfg, cfg)

    mech_dir = out_dir / f"{mechanism.value}_seed{seed}"
    mech_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Training {mechanism.value} (seed={seed})")
    logger.info(f"{'='*60}")

    # Stage 1
    t0 = time.time()
    result1 = run_stage1(net, cfg, train_cfg, device, seed)
    stage1_time = time.time() - t0

    logger.info(f"Stage 1 done: loss={result1.final_loss:.4f}, "
                f"acc={result1.decoder_accuracy:.3f}, "
                f"time={stage1_time:.1f}s")
    logger.info(f"Gating: {result1.gating_passed}")

    # Save Stage 1 checkpoint
    torch.save({
        "model_state": net.state_dict(),
        "decoder_state": loss_fn.orientation_decoder.state_dict(),
        "gating": result1.gating_passed,
        "mechanism": mechanism.value,
        "seed": seed,
    }, mech_dir / "stage1_checkpoint.pt")

    # Stage 2 with intermediate checkpoints
    freeze_stage1(net)

    def save_checkpoint(step: int):
        ckpt_path = mech_dir / f"checkpoint_step{step}.pt"
        torch.save({
            "model_state": net.state_dict(),
            "decoder_state": loss_fn.orientation_decoder.state_dict(),
            "step": step,
            "mechanism": mechanism.value,
            "seed": seed,
        }, ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path}")

    t0 = time.time()
    result2 = run_stage2(
        net, loss_fn, cfg, train_cfg, stim_cfg, device, seed,
        checkpoint_fn=save_checkpoint,
        checkpoint_steps=ckpt_steps,
    )
    stage2_time = time.time() - t0

    logger.info(f"Stage 2 done: loss={result2.final_loss:.4f}, "
                f"s_acc={result2.final_sensory_acc:.3f}, "
                f"p_acc={result2.final_pred_acc:.3f}, "
                f"time={stage2_time:.1f}s")

    # Save final checkpoint
    torch.save({
        "model_state": net.state_dict(),
        "decoder_state": loss_fn.orientation_decoder.state_dict(),
        "loss_history": result2.loss_history,
        "mechanism": mechanism.value,
        "seed": seed,
    }, mech_dir / "checkpoint.pt")

    summary = {
        "mechanism": mechanism.value,
        "seed": seed,
        "stage1_loss": result1.final_loss,
        "stage1_acc": result1.decoder_accuracy,
        "stage1_gating": result1.gating_passed,
        "stage1_time_s": stage1_time,
        "stage2_loss": result2.final_loss,
        "stage2_sensory_acc": result2.final_sensory_acc,
        "stage2_pred_acc": result2.final_pred_acc,
        "stage2_time_s": stage2_time,
        "stage2_steps": result2.n_steps_trained,
    }

    return net, cfg, loss_fn, summary


def run_experiments(
    net: LaminarV1V2Network,
    cfg: ModelConfig,
    n_trials: int = 200,
    seed: int = 42,
) -> dict:
    """Run P1, P2, P3 experiments on a trained model."""
    net = net.cpu()
    net.eval()
    results = {}

    # P1: Hidden state
    logger.info("  Running P1 (hidden_state)...")
    p1 = HiddenStateParadigm(net, cfg, probe_deviations=[0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
    results["P1"] = p1.run(n_trials=n_trials, seed=seed, batch_size=32)

    # P2: Omission
    logger.info("  Running P2 (omission)...")
    p2 = OmissionParadigm(net, cfg)
    results["P2"] = p2.run(n_trials=n_trials, seed=seed, batch_size=32)

    # P3: Ambiguous
    logger.info("  Running P3 (ambiguous)...")
    p3 = AmbiguousParadigm(net, cfg)
    results["P3"] = p3.run(n_trials=n_trials, seed=seed, batch_size=32)

    return results


def run_analyses(
    experiment_results: dict,
    mechanism: str,
) -> dict:
    """Run key analyses on experiment results.

    Returns dict of analysis results for this mechanism.
    """
    analysis = {}
    p1 = experiment_results["P1"]
    p2 = experiment_results["P2"]
    p3 = experiment_results["P3"]

    # 1. Suppression + surprise profiles (from P1)
    logger.info("  Analysis: suppression+surprise profiles...")
    supp = compute_suppression_profile_from_experiment(p1, expected_ori=45.0)
    analysis["suppression_profile"] = supp

    # 2. Mean responses
    mean_resp = compute_mean_responses(p1)
    analysis["mean_responses"] = mean_resp

    # 3. Tuning curves (from P1 sustained window)
    logger.info("  Analysis: tuning curves...")
    try:
        start, end = p1.temporal_windows["sustained"]
        cond_responses = {}
        for cond_name, cd in p1.conditions.items():
            cond_responses[cond_name] = cd.r_l23[:, start:end].mean(dim=1)
        tc_result = analyse_tuning_curves(cond_responses)
        analysis["tuning_curves"] = tc_result
    except Exception as e:
        logger.warning(f"  Tuning curves failed: {e}")

    # 4. Decoding (from P1)
    logger.info("  Analysis: decoding...")
    try:
        start, end = p1.temporal_windows["sustained"]
        all_patterns = []
        all_labels = []
        for i, (cond_name, cd) in enumerate(p1.conditions.items()):
            patterns = cd.r_l23[:, start:end].mean(dim=1)
            all_patterns.append(patterns)
            all_labels.append(torch.full((patterns.shape[0],), i, dtype=torch.long))
        X = torch.cat(all_patterns, dim=0)
        y = torch.cat(all_labels, dim=0)
        cv_acc = cross_validated_decoding(X, y, n_folds=5)
        analysis["decoding_accuracy"] = cv_acc

        # d-prime between expected and neutral conditions
        exp_conds = [n for n in p1.conditions if "expected" in n or "dev0" in n]
        neut_conds = [n for n in p1.conditions if "neutral" in n]
        if exp_conds and neut_conds:
            exp_r = p1.conditions[exp_conds[0]].r_l23[:, start:end].mean(dim=1)
            neut_r = p1.conditions[neut_conds[0]].r_l23[:, start:end].mean(dim=1)
            dp = compute_d_prime(exp_r, neut_r)
            analysis["d_prime_exp_neut"] = dp
    except Exception as e:
        logger.warning(f"  Decoding failed: {e}")

    # 5. Energy
    logger.info("  Analysis: energy...")
    energy = compute_energy(p1)
    analysis["energy"] = energy

    # 6. Observation model (from P1)
    logger.info("  Analysis: observation model...")
    try:
        start, end = p1.temporal_windows["sustained"]
        obs_data = {}
        for cond_name, cd in p1.conditions.items():
            obs_data[cond_name] = cd.r_l23[:, start:end].mean(dim=1)
        obs_result = run_observation_model(obs_data, n_voxels=8, snr=10.0)
        analysis["observation_model"] = obs_result
    except Exception as e:
        logger.warning(f"  Observation model failed: {e}")

    # 7. Omission + prestimulus (from P2)
    logger.info("  Analysis: omission + prestimulus...")
    try:
        omission_result = run_omission_analysis(p2)
        analysis["omission"] = omission_result
    except Exception as e:
        logger.warning(f"  Omission analysis failed: {e}")

    # 8. Temporal analysis (from P1)
    logger.info("  Analysis: temporal...")
    temporal = run_temporal_analysis(p1)
    analysis["temporal"] = temporal

    # 9. V2 probes (from P1)
    logger.info("  Analysis: V2 probes...")
    v2 = run_v2_probes(p1)
    analysis["v2_probes"] = v2

    return analysis


def summarize_results(all_results: dict) -> str:
    """Generate a human-readable summary of the pilot run results."""
    lines = []
    lines.append("# Phase 8A: Pilot Science Run Results\n")
    lines.append(f"Mechanisms tested: {', '.join(all_results.keys())}\n")

    # Training summary table
    lines.append("## Training Summary\n")
    lines.append("| Mechanism | S1 Acc | S2 Loss | S2 S_Acc | S2 P_Acc | S1 Time | S2 Time |")
    lines.append("|-----------|--------|---------|----------|----------|---------|---------|")
    for mech, data in all_results.items():
        s = data["training"]
        lines.append(
            f"| {mech} | {s['stage1_acc']:.3f} | {s['stage2_loss']:.4f} | "
            f"{s['stage2_sensory_acc']:.3f} | {s['stage2_pred_acc']:.3f} | "
            f"{s['stage1_time_s']:.0f}s | {s['stage2_time_s']:.0f}s |"
        )

    # Suppression profiles
    lines.append("\n## Suppression Profiles (P1)\n")
    lines.append("| Mechanism | Supp Mean | Supp Std | Surp Mean | Surp Std | Diff Mean |")
    lines.append("|-----------|-----------|----------|-----------|----------|-----------|")
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        sp = a.get("suppression_profile")
        if sp is not None:
            supp_mean = sp.suppression.mean().item()
            supp_std = sp.suppression.std().item()
            surp_mean = sp.surprise.mean().item() if sp.surprise is not None else float("nan")
            surp_std = sp.surprise.std().item() if sp.surprise is not None else float("nan")
            diff_mean = sp.difference.mean().item() if sp.difference is not None else float("nan")
            lines.append(
                f"| {mech} | {supp_mean:.4f} | {supp_std:.4f} | "
                f"{surp_mean:.4f} | {surp_std:.4f} | {diff_mean:.4f} |"
            )

    # Energy
    lines.append("\n## Energy (P1)\n")
    lines.append("| Mechanism | Total Activity | Excitatory | Inhibitory |")
    lines.append("|-----------|---------------|------------|------------|")
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        e = a.get("energy")
        if e is not None:
            lines.append(
                f"| {mech} | {e.total_activity:.4f} | "
                f"{e.excitatory_activity:.4f} | {e.inhibitory_activity:.4f} |"
            )

    # Decoding
    lines.append("\n## Decoding (P1)\n")
    lines.append("| Mechanism | CV Accuracy | d' (exp vs neut) |")
    lines.append("|-----------|-------------|-------------------|")
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        cv = a.get("decoding_accuracy", float("nan"))
        dp = a.get("d_prime_exp_neut", float("nan"))
        lines.append(f"| {mech} | {cv:.3f} | {dp:.3f} |")

    # V2 probes
    lines.append("\n## V2 Probes (P1)\n")
    lines.append("| Mechanism | Cond | q_pred Entropy | pi_pred Mean |")
    lines.append("|-----------|------|----------------|--------------|")
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        v2 = a.get("v2_probes")
        if v2 is not None:
            for cond in sorted(v2.q_pred_entropy.keys())[:3]:  # first 3 conditions
                ent = v2.q_pred_entropy[cond]
                pi = v2.pi_pred_mean[cond]
                lines.append(f"| {mech} | {cond[:20]} | {ent:.3f} | {pi:.3f} |")

    # Observation model
    lines.append("\n## Observation Model (P1)\n")
    lines.append("| Mechanism | MVPA 3-way Acc | MVPA 2-way Acc | Dissociation |")
    lines.append("|-----------|---------------|---------------|-------------|")
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        obs = a.get("observation_model")
        if obs is not None:
            lines.append(
                f"| {mech} | {obs.mvpa_accuracy_3way:.3f} | "
                f"{obs.mvpa_accuracy_2way:.3f} | {obs.dissociation} |"
            )

    # Go/No-Go
    lines.append("\n## Go/No-Go Assessment\n")
    lines.append("### Mechanism Divergence Check\n")

    # Check if suppression profiles differ across mechanisms
    supp_means = {}
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        sp = a.get("suppression_profile")
        if sp is not None:
            supp_means[mech] = sp.suppression.mean().item()

    if len(supp_means) >= 2:
        vals = list(supp_means.values())
        spread = max(vals) - min(vals)
        lines.append(f"- Suppression profile spread: {spread:.4f}")
        lines.append(f"  (range: {min(vals):.4f} to {max(vals):.4f})")
        if spread > 0.01:
            lines.append(f"  **DIVERGENT** — mechanisms produce different suppression profiles")
        else:
            lines.append(f"  Marginal divergence — may need more training or seeds")
    else:
        lines.append("- Insufficient data to assess divergence")

    energy_vals = {}
    for mech, data in all_results.items():
        a = data.get("analysis", {})
        e = a.get("energy")
        if e is not None:
            energy_vals[mech] = e.total_activity

    if len(energy_vals) >= 2:
        vals = list(energy_vals.values())
        spread = max(vals) - min(vals)
        lines.append(f"- Energy spread: {spread:.4f}")
        if spread > 0.01:
            lines.append(f"  **DIVERGENT** — mechanisms have different metabolic costs")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Quick mode: {args.quick}")

    mechanisms = MECHANISMS
    if args.mechanisms:
        mechanisms = [MechanismType(m) for m in args.mechanisms]

    all_results: dict[str, dict] = {}
    total_t0 = time.time()

    for mechanism in mechanisms:
        mech_name = mechanism.value

        # Train
        net, cfg, loss_fn, train_summary = train_model(
            mechanism, args.seed, device, out_dir, quick=args.quick,
        )

        # Run experiments
        logger.info(f"Running experiments for {mech_name}...")
        exp_results = run_experiments(
            net, cfg,
            n_trials=10 if args.quick else args.n_trials,
            seed=args.seed,
        )

        # Save experiment results
        mech_dir = out_dir / f"{mech_name}_seed{args.seed}"
        torch.save(exp_results, mech_dir / "experiments.pt")

        # Run analyses
        logger.info(f"Running analyses for {mech_name}...")
        analysis = run_analyses(exp_results, mech_name)

        # Save analysis results
        torch.save(analysis, mech_dir / "analysis.pt")

        all_results[mech_name] = {
            "training": train_summary,
            "analysis": analysis,
        }

        # Free GPU memory
        del net, loss_fn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total_time = time.time() - total_t0
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Generate summary report
    report = summarize_results(all_results)

    report_path = out_dir / "PILOT_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {report_path}")

    # Also save structured results
    # (can't save full analysis objects to JSON, so save key metrics)
    metrics = {}
    for mech, data in all_results.items():
        m = {"training": data["training"]}
        a = data.get("analysis", {})

        sp = a.get("suppression_profile")
        if sp is not None:
            m["suppression_mean"] = sp.suppression.mean().item()
            m["surprise_mean"] = sp.surprise.mean().item() if sp.surprise is not None else None

        e = a.get("energy")
        if e is not None:
            m["total_activity"] = e.total_activity
            m["excitatory_activity"] = e.excitatory_activity
            m["inhibitory_activity"] = e.inhibitory_activity

        m["decoding_accuracy"] = a.get("decoding_accuracy")
        m["d_prime"] = a.get("d_prime_exp_neut")

        obs = a.get("observation_model")
        if obs is not None:
            m["mvpa_3way"] = obs.mvpa_accuracy_3way
            m["mvpa_2way"] = obs.mvpa_accuracy_2way
            m["dissociation"] = obs.dissociation

        metrics[mech] = m

    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Pilot run complete.")


if __name__ == "__main__":
    main()
