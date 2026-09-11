"""Refit the uncapped peak-readout temporal network from static v9 common weights."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

import train_sweep as train

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints" / "temporal_peak"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-ratio", type=int, choices=(2, 3), default=2,
                        help="Initial tau_SST/tau_E ratio; both taus remain unrestricted.")
    parser.add_argument("--seed", type=int, choices=(8, 9, 10), default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "temporal_peak_training")
    parser.add_argument("--axis-steps", type=int, default=24000)
    parser.add_argument("--lr", type=float, default=.003)
    parser.set_defaults(
        batch=128, sequence_length=12, mismatch_prob=.02, clip=5.0,
        log_every=100, checkpoint_every=250, freeze_local_comp=True,
        learn_feedback_gain=True, axis_current_readout="population_vector",
        center_feedback=False, feedback_mode=train.tuned.FEEDBACK_MODE_POSTERIOR,
        temporal_v10=True, learn_temporal_kinetics=True, sst_response_gain=True,
        temporal_current_window="peak", ce_constraint="dual",
        temporal_current_ce_penalty_coefficient=10000.0,
    )
    args = parser.parse_args()
    if args.axis_steps < 1:
        parser.error("axis steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    return args


def peak_common_initial(seed, init_ratio, device, *, tau_e_init=None, tau_sst_init=None):
    template_path = CHECKPOINTS / f"init_{init_ratio}x" / f"seed_{seed}" / "common_initial.pt"
    template = torch.load(template_path, map_location=device, weights_only=False)
    config = copy.deepcopy(template["tuned_net_config"])
    config["temporal_protocol"]["predictor_readout"] = "peak_evidence_noiseless"
    if tau_e_init is not None:
        config["temporal_tau_e_init"] = tau_e_init
    config["temporal_tau_p_init"] = 1.0 if tau_sst_init is None else tau_sst_init
    net = train.tuned.build_tuned_from_config(config).to(device)
    source = ROOT / "checkpoints" / f"seed{seed}" / "alpha0p07" / "common_pretrain_final.pt"
    static = torch.load(source, map_location=device, weights_only=False)
    state = net.state_dict()
    state.update(static["state_dict"])
    state["w_sf_fixed"] = state["w_sf_fixed"].clone()
    state["w_sf_fixed"].fill_(train.tuned.softplus_inverse(
        float(state["w_sf_fixed"].clamp_min(0.0))
    ))
    net.load_state_dict(state)
    references = dict(static["references"])
    references["J_ref"] = config["temporal_protocol"]["on_duration"] * references["R_ref"]
    return {
        "stage": "common_initial", "seed": seed, "step": 0, "target_steps": 0,
        "source_checkpoint": str(source.relative_to(ROOT)),
        "initialization_note": (
            "Static v9 common-pretrain weights with SST response-gain reparameterization, "
            "two newly initialized time constants, and a noiseless peak readout; "
            "no temporal pretraining."
        ),
        "state_dict": net.state_dict(), "references": references,
        "tuned_net_config": config,
        "model_architecture_version": net.model_architecture_version,
        "training_compatibility_version": train.LEARNED_TEMPORAL_TRAINING_COMPATIBILITY_VERSION,
        "feedback_mode": static["feedback_mode"], "center_feedback": False,
        "temporal_current_window": "peak",
    }


@torch.no_grad()
def calibrate_peak_ce(args, device):
    # The reference always has tau_E=tau_SST=1, independent of the fitted ratio.
    common = peak_common_initial(args.seed, 2, device, tau_e_init=1.0, tau_sst_init=1.0)
    net = train.tuned.build_tuned_from_config(common["tuned_net_config"]).to(device)
    net.load_state_dict(common["state_dict"])
    net.eval()
    data = train.make_generator(device, 920001)
    noise = train.make_generator(device, 920002)
    measurements = {"current_ce": 0.0, "next_ce": 0.0}
    for _ in range(8):
        theta, channels = train.momentum_batch(
            128, 12, device, data, mismatch_prob=.02,
        )
        losses = train.task_activity_losses(
            net, theta, channels, noise, common["references"],
            feedback_mode=args.feedback_mode, temporal_current_window="peak",
        )
        for key in measurements:
            measurements[key] += float(losses[key]) / 8
    return {
        "seed": args.seed,
        "source_checkpoint": common["source_checkpoint"],
        "predictor_readout": "peak_evidence_noiseless",
        "temporal_current_window": "peak",
        "calibration_data_seed": 920001,
        "calibration_noise_seed": 920002,
        "batch": 128, "batches": 8, "sequence_length": 12,
        "mismatch_prob": .02, "device": str(device),
        "measurements": measurements,
        "ceiling": measurements["current_ce"],
        "performance_tolerance": 0.0,
        "penalty_coefficient": args.temporal_current_ce_penalty_coefficient,
    }


def main():
    args = parse_args()
    device = train.choose_device(args.device)
    torch.set_num_threads(2)
    train.seed_everything(args.seed)
    run_dir = args.out / f"init_{args.init_ratio}x" / f"seed_{args.seed}"
    calibration_path = run_dir / "peak_reference_limits.json"
    if calibration_path.exists():
        limits = json.loads(calibration_path.read_text())
        if (
            limits["source_checkpoint"]
            != f"checkpoints/seed{args.seed}/alpha0p07/common_pretrain_final.pt"
            or limits["predictor_readout"] != "peak_evidence_noiseless"
        ):
            raise ValueError("existing calibration uses a different initial source or readout")
    else:
        limits = calibrate_peak_ce(args, device)
        train.atomic_json_save(limits, calibration_path)
    args.temporal_current_ce_ceiling = limits["ceiling"]

    common = peak_common_initial(args.seed, args.init_ratio, device)
    config = copy.deepcopy(common["tuned_net_config"])
    final_path = run_dir / "alpha_0p3_final.pt"
    latest_path = run_dir / "alpha_0p3_latest.pt"
    resume_path = final_path if final_path.exists() else latest_path
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if (
            saved["seed"] != args.seed or float(saved["alpha"]) != .3
            or saved.get("temporal_current_ce_constraint_method") != "dual"
            or saved.get("temporal_current_window") != "peak"
            or saved.get("temporal_current_ce_ceiling") != args.temporal_current_ce_ceiling
            or saved["tuned_net_config"]["temporal_protocol"] != config["temporal_protocol"]
            or any(saved["tuned_net_config"][key] != config[key] for key in (
                "temporal_tau_e_init", "temporal_tau_p_init"
            ))
        ):
            raise ValueError("existing checkpoint uses a different temporal recipe")
        previous_target = int(saved["target_steps"])
        step = int(saved["step"])
        if args.axis_steps < max(previous_target, step):
            raise ValueError("cannot decrease the existing step target")
        if resume_path == final_path and step == args.axis_steps:
            return
        if args.axis_steps > previous_target:
            original_source = resume_path
            if final_path.exists():
                original_source = run_dir / f"alpha_0p3_step{step:05d}.pt"
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

    train.MODEL_CONFIG.clear()
    train.MODEL_CONFIG.update(config)
    args.temporal_tau_e_init = config["temporal_tau_e_init"]
    args.temporal_tau_p_init = config["temporal_tau_p_init"]
    initial_path = run_dir / "common_initial.pt"
    if not initial_path.exists():
        train.atomic_torch_save(common, initial_path)
    log = train.EventLog(run_dir / "training.jsonl")
    try:
        result = train.run_alpha(
            .3, common["state_dict"], common["references"], args, run_dir, device, log,
        )
        train.atomic_json_save(result, run_dir / "training_summary.json")
        log.write({"event": "run_complete", "condition": "fitted_joint", "init_ratio": args.init_ratio})
    finally:
        log.close()


if __name__ == "__main__":
    main()
