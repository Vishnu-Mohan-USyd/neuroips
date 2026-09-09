"""Refit the five learned-temporal cases from their exact packaged initial states."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

import train_sweep as train

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints" / "temporal_v11"
CONDITIONS = {
    "joint": (.3, "early"),
    "accuracy_only": (0.0, "early"),
    "energy_only": (1.0, "early"),
    "fast_sst": (.3, "early"),
    "sustained": (.3, "sustained"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(8, 9, 10), default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "temporal_v11_training")
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--axis-steps", type=int, default=24000)
    parser.add_argument("--lr", type=float, default=.003)
    parser.set_defaults(
        batch=128, sequence_length=12, mismatch_prob=.02, clip=5.0,
        log_every=100, checkpoint_every=250, freeze_local_comp=True,
        learn_feedback_gain=True, axis_current_readout="population_vector",
        center_feedback=False, feedback_mode=train.tuned.FEEDBACK_MODE_POSTERIOR,
        temporal_v10=True, learn_temporal_kinetics=True, sst_response_gain=True,
        temporal_current_ce_penalty_coefficient=10000.0,
    )
    args = parser.parse_args()
    if args.axis_steps < 1:
        parser.error("axis steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    return args


def main():
    args = parse_args()
    device = train.choose_device(args.device)
    torch.set_num_threads(2)
    train.seed_everything(args.seed)
    calibration = json.loads((CHECKPOINTS / "reference_limits.json").read_text())
    ceiling = calibration[str(args.seed)]["ceiling"]
    for condition in args.conditions:
        alpha, window = CONDITIONS[condition]
        run_dir = args.out / condition / f"seed_{args.seed}"
        prefix = f"alpha_{train.alpha_slug(alpha)}"
        final_path = run_dir / f"{prefix}_final.pt"
        latest_path = run_dir / f"{prefix}_latest.pt"
        resume_path = final_path if final_path.exists() else latest_path
        if resume_path.exists():
            saved = torch.load(resume_path, map_location=device, weights_only=False)
            previous_target = int(saved["target_steps"])
            step = int(saved["step"])
            if args.axis_steps < max(previous_target, step):
                raise ValueError(f"{condition}: cannot decrease the existing step target")
            if resume_path == final_path and step == args.axis_steps:
                continue
            if args.axis_steps > previous_target:
                original_source = resume_path
                if final_path.exists():
                    original_source = run_dir / f"{prefix}_step{step:05d}.pt"
                    if not original_source.exists():
                        train.atomic_torch_save(saved, original_source)
                continued = dict(
                    saved, previous_target_steps=previous_target,
                    continuation_source_checkpoint=str(original_source),
                    target_steps=args.axis_steps,
                )
                train.atomic_torch_save(continued, latest_path)
                if final_path.exists():
                    final_path.unlink()

        source = CHECKPOINTS / condition / f"seed_{args.seed}" / "common_initial.pt"
        common = torch.load(source, map_location=device, weights_only=False)
        config = copy.deepcopy(common["tuned_net_config"])
        train.MODEL_CONFIG.clear()
        train.MODEL_CONFIG.update(config)
        net = train.tuned.build_tuned_from_config(config).to(device)
        assert net.learn_temporal_kinetics and net.sst_response_gain
        net.load_state_dict(common["state_dict"])
        state = copy.deepcopy(net.state_dict())
        condition_args = copy.copy(args)
        condition_args.temporal_current_window = window
        condition_args.temporal_current_ce_ceiling = None if condition == "energy_only" else ceiling
        condition_args.temporal_tau_e_init = config["temporal_tau_e_init"]
        condition_args.temporal_tau_p_init = config["temporal_tau_p_init"]
        run_dir.mkdir(parents=True, exist_ok=True)
        initial_path = run_dir / "common_initial.pt"
        if not initial_path.exists():
            train.atomic_torch_save(common, initial_path)
        log = train.EventLog(run_dir / "training.jsonl")
        try:
            result = train.run_alpha(alpha, state, common["references"], condition_args,
                                     run_dir, device, log)
            train.atomic_json_save(result, run_dir / "training_summary.json")
            log.write({"event": "run_complete", "condition": f"fitted_{condition}"})
        finally:
            log.close()


if __name__ == "__main__":
    main()
