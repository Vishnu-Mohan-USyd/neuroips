#!/usr/bin/env python3
"""Validate fixed-basis tuned emergence checkpoints on held-out assays only."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tuned_emergence_lib import (  # noqa: E402
    N,
    STEP_DEG,
    build_tuned_from_config,
    chan,
    device,
    forward_seq_tuned,
    make_sequences,
    model_config,
)


K = 4
VELS = (-3, -2, -1, 1, 2, 3)


def build_pairs():
    rows_e, rows_u, e_vals, u_vals = [], [], [], []
    for c0 in range(N):
        for v in VELS:
            prefix = [int((c0 + v * t) % N) for t in range(K)]
            e = int((c0 + v * K) % N)
            u = int((e + N // 2) % N)
            rows_e.append(prefix + [e])
            rows_u.append(prefix + [u])
            e_vals.append(e)
            u_vals.append(u)
    return (
        torch.tensor(rows_e, device=device).float() * STEP_DEG,
        torch.tensor(rows_u, device=device).float() * STEP_DEG,
        torch.tensor(e_vals, device=device, dtype=torch.long),
        torch.tensor(u_vals, device=device, dtype=torch.long),
    )


def load_net(path):
    obj = torch.load(path, map_location=device)
    config = obj.get("tuned_net_config", {}) if isinstance(obj, dict) else {}
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    training_contract = obj.get("training_contract", {}) if isinstance(obj, dict) else {}
    net = build_tuned_from_config(config).to(device)
    net.load_state_dict(state)
    net.eval()
    return net, training_contract


def local_comp_effective_strength_value(net):
    return float(net.local_comp_effective_strength().detach().cpu().item())


def local_comp_gain_raw_value(net):
    if not getattr(net, "local_comp_trainable", False):
        return None
    return float(net.local_comp_strength_raw.detach().cpu().item())


def stats(x):
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "sem": float(x.std(unbiased=False).item() / math.sqrt(max(1, x.numel()))),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def aligned_stack(r, center_idx):
    return torch.stack([torch.roll(row, shifts=-int(center_idx[i].item()), dims=0) for i, row in enumerate(r)], 0)


def profile_summary(profile):
    center = profile[0]
    near = profile[[1, -1]].mean()
    flank = profile[[2, -2]].mean()
    shoulder = profile[[3, 4, -3, -4]].mean()
    far = profile[[10, 11, 12, -10, -11, -12]].mean()
    return center, near, flank, shoulder, far


@torch.no_grad()
def held_acc(net, seed, batch, S=12):
    torch.manual_seed(seed)
    theta = make_sequences(batch, S, mode="momentum", p_stay=0.9)
    preds, _ = forward_seq_tuned(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


@torch.no_grad()
def noisy_decoder(net, r, target, sigma, repeats, seed):
    torch.manual_seed(seed)
    batch = r.shape[0]
    noisy = r.unsqueeze(0) + sigma * torch.randn(repeats, batch, N, device=device)
    logits = net.decode(noisy.reshape(repeats * batch, N))
    target_rep = target.repeat(repeats)
    ce = F.cross_entropy(logits, target_rep, reduction="none")
    prob = logits.softmax(-1).gather(1, target_rep.view(-1, 1)).squeeze(1)
    acc = logits.argmax(-1) == target_rep
    return {"ce": stats(ce), "target_prob": stats(prob), "acc": float(acc.float().mean().item())}


def shape_previous(mode, r_e, r_floor, e_idx, margin):
    aligned = aligned_stack(r_e, e_idx)
    aligned_floor = aligned_stack(r_floor, e_idx)
    profile = aligned.mean(0)
    floor_profile = aligned_floor.mean(0)
    center, near, flank, _, far = profile_summary(profile)
    floor_center = floor_profile[0]
    if mode == "sharpen":
        checks = {
            "center_above_feedback_off_floor": bool(center.item() >= floor_center.item() + margin),
            "center_above_near": bool(center.item() >= near.item() + margin),
            "center_above_flank": bool(center.item() >= flank.item() + margin),
            "center_above_far": bool(center.item() >= far.item() + margin),
        }
    else:
        checks = {
            "center_below_feedback_off_floor": bool(center.item() <= floor_center.item() - margin),
        }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "expected_center": float(center.item()),
        "feedback_off_floor_center": float(floor_center.item()),
        "expected_near_mean_offsets_1": float(near.item()),
        "expected_flank_mean_offsets_2": float(flank.item()),
        "expected_far_mean_offsets_10_11_12": float(far.item()),
        "expected_profile_aligned_to_expected_channel": profile.detach().float().cpu().tolist(),
        "feedback_off_floor_profile_aligned_to_expected_channel": floor_profile.detach().float().cpu().tolist(),
    }


def shape_strict(mode, r_e, e_idx, args):
    aligned = aligned_stack(r_e, e_idx)
    profile = aligned.mean(0)
    sem = aligned.std(0, unbiased=False) / math.sqrt(aligned.shape[0])
    center, near, flank, shoulder, far = profile_summary(profile)
    if mode == "sharpen":
        checks = {
            "center_above_near": bool(center.item() >= near.item() + args.shape_margin),
            "center_above_flank": bool(center.item() >= flank.item() + args.shape_margin),
            "center_above_far": bool(center.item() >= far.item() + args.shape_margin),
            "far_low_relative_to_center": bool(far.item() <= args.sharpen_far_max_frac * center.item()),
        }
    else:
        checks = {
            "center_nonzero": bool(center.item() >= args.dampen_center_min),
            "center_below_flank": bool(center.item() + args.shape_margin <= flank.item()),
            "center_below_shoulder": bool(center.item() + args.shape_margin <= shoulder.item()),
            "far_not_pathological_vs_flank": bool(far.item() <= flank.item() + args.dampen_far_flank_slack),
        }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "raw_center": float(center.item()),
        "raw_near_mean_offsets_5deg": float(near.item()),
        "raw_flank_mean_offsets_10deg": float(flank.item()),
        "raw_shoulder_mean_offsets_15_20deg": float(shoulder.item()),
        "raw_far_mean_offsets_50_55_60deg": float(far.item()),
        "raw_profile_aligned_to_expected": profile.detach().float().cpu().tolist(),
        "raw_profile_sem_aligned_to_expected": sem.detach().float().cpu().tolist(),
    }


@torch.no_grad()
def validate_one(label, path, mode, args, seed_offset):
    net, training_contract = load_net(path)
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    _, r_all_e = forward_seq_tuned(net, theta_e, 1.0)
    _, r_all_u = forward_seq_tuned(net, theta_u, 1.0)
    _, r_all_floor = forward_seq_tuned(net, theta_e, 0.0)
    r_e = r_all_e[:, K, :]
    r_u = r_all_u[:, K, :]
    r_floor = r_all_floor[:, K, :]

    E_e = r_e.abs().mean(dim=1)
    E_u = r_u.abs().mean(dim=1)
    dE = E_e - E_u
    energy_pass = bool((dE <= -args.energy_margin).all().item())

    logits_e = net.decode(r_e)
    logits_u = net.decode(r_u)
    ce_e = F.cross_entropy(logits_e, e_idx, reduction="none")
    ce_u = F.cross_entropy(logits_u, u_idx, reduction="none")
    prob_e = logits_e.softmax(-1).gather(1, e_idx.view(-1, 1)).squeeze(1)
    prob_u = logits_u.softmax(-1).gather(1, u_idx.view(-1, 1)).squeeze(1)
    if mode == "sharpen":
        decode_pass = bool(ce_e.mean().item() + args.decode_margin <= ce_u.mean().item() and prob_e.mean().item() >= prob_u.mean().item() + args.decode_margin)
    else:
        decode_pass = bool(ce_e.mean().item() >= ce_u.mean().item() + args.decode_margin and prob_e.mean().item() + args.decode_margin <= prob_u.mean().item())

    noisy_e = noisy_decoder(net, r_e, e_idx, args.noise_sigma, args.noise_repeats, args.seed + seed_offset)
    noisy_u = noisy_decoder(net, r_u, u_idx, args.noise_sigma, args.noise_repeats, args.seed + seed_offset + 1)
    if mode == "sharpen":
        noisy_pass = bool(noisy_e["ce"]["mean"] + args.noisy_decode_margin <= noisy_u["ce"]["mean"] and noisy_e["target_prob"]["mean"] >= noisy_u["target_prob"]["mean"] + args.noisy_prob_margin)
        noisy_acc_floor_pass = bool(noisy_e["acc"] >= args.min_sharpen_expected_noisy_acc)
    else:
        noisy_pass = bool(noisy_e["ce"]["mean"] >= noisy_u["ce"]["mean"] + args.noisy_decode_margin and noisy_e["target_prob"]["mean"] + args.noisy_prob_margin <= noisy_u["target_prob"]["mean"])
        noisy_acc_floor_pass = True

    held = held_acc(net, args.seed + seed_offset, args.held_batch)
    previous_shape = shape_previous(mode, r_e, r_floor, e_idx, args.shape_margin)
    strict_shape = shape_strict(mode, r_e, e_idx, args)
    previous_pass = bool(energy_pass and decode_pass and held >= args.held_min and previous_shape["pass"])
    strict_pass = bool(energy_pass and decode_pass and noisy_pass and noisy_acc_floor_pass and held >= args.held_min and strict_shape["pass"])
    return {
        "label": label,
        "path": path,
        "mode": mode,
        "n_pairs": int(theta_e.shape[0]),
        "architecture_contract": {
            "fixed_local_feedforward_basis": True,
            "constrained_orientation_readout": True,
            "readout_contract": str(net.readout),
            "population_vector_readout": bool(net.readout == "population_vector"),
            "population_normalize": bool(getattr(net, "population_normalize", True)),
            "prediction_dependent_inhibition_strength": float(getattr(net, "pred_inhib_strength", 0.0)),
            "predicted_feature_suppression_strength": float(getattr(net, "pred_feature_supp_strength", 0.0)),
            "rate_saturation_r_max": float(getattr(net, "rate_saturation_r_max", 0.0)),
            "rate_saturation_r_half": float(getattr(net, "rate_saturation_r_half", 1.0)),
            "l23_adapt_strength": float(getattr(net, "adapt_strength", 0.0)),
            "l23_adapt_decay": float(getattr(net, "adapt_decay", 0.85)),
            "l23_adapt_sigma_channels": float(getattr(net, "adapt_sigma_channels", 1.0)),
            "l23_local_comp_strength": float(getattr(net, "local_comp_strength", 0.0)),
            "l23_local_comp_learned_strength": local_comp_effective_strength_value(net),
            "l23_local_comp_final_raw_strength": local_comp_gain_raw_value(net),
            "l23_local_comp_trainable": bool(getattr(net, "local_comp_trainable", False)),
            "l23_local_comp_sigma_channels": float(getattr(net, "local_comp_sigma_channels", 1.0)),
            "l23_local_comp_power": float(getattr(net, "local_comp_power", 1.0)),
            "l23_local_comp_mode": str(getattr(net, "local_comp_mode", "divisive")),
            "prediction_confidence_energy_objective_enabled": training_contract.get("prediction_confidence_energy_objective_enabled"),
            "prediction_targeted_activity_penalty_replacement_added": training_contract.get("prediction_targeted_activity_penalty_replacement_added"),
            "model_config": model_config(net),
            "gains_g_v_g_s_g_sv_g_e_g_ps": F.softplus(net.circ_raw).detach().float().cpu().tolist(),
            "initial_state_evidence": training_contract.get("initial_state_evidence", {}),
            "local_comp_training_contract": training_contract.get("shared_current_step_l23_local_competition", {}),
            "trainable_local_competition_strength": training_contract.get("trainable_local_competition_strength", {}),
        },
        "energy_mean_abs": {
            "expected": stats(E_e),
            "unexpected": stats(E_u),
            "delta_expected_minus_unexpected": stats(dE),
            "frac_delta_lt_0": float((dE < 0).float().mean().item()),
            "max_allowed_delta": float(-args.energy_margin),
            "pass": energy_pass,
        },
        "current_population_readout_clean": {
            "ce_expected_target": stats(ce_e),
            "ce_unexpected_target": stats(ce_u),
            "target_prob_expected": stats(prob_e),
            "target_prob_unexpected": stats(prob_u),
            "pass": decode_pass,
        },
        "current_population_readout_noisy": {
            "sigma": args.noise_sigma,
            "repeats": args.noise_repeats,
            "expected": noisy_e,
            "unexpected": noisy_u,
            "direction_pass": noisy_pass,
            "min_sharpen_expected_acc": args.min_sharpen_expected_noisy_acc if mode == "sharpen" else None,
            "expected_acc_floor_pass": noisy_acc_floor_pass if mode == "sharpen" else None,
            "pass": bool(noisy_pass and noisy_acc_floor_pass),
        },
        "prediction": {
            "held_acc_percent": held,
            "held_min_percent": args.held_min,
            "pass": bool(held >= args.held_min),
        },
        "previous_shape": previous_shape,
        "raw_tuning_shape": strict_shape,
        "previous_pass": previous_pass,
        "strict_pass": strict_pass,
        "pass": bool(previous_pass and strict_pass),
    }


def zero_pred_conf_audit(checkpoints):
    enabled = {
        item["mode"]: item["architecture_contract"].get("prediction_confidence_energy_objective_enabled")
        for item in checkpoints
    }
    replacements = {
        item["mode"]: item["architecture_contract"].get("prediction_targeted_activity_penalty_replacement_added")
        for item in checkpoints
    }
    return {
        "pass": bool(
            all(value is False for value in enabled.values())
            and all(value is False for value in replacements.values())
        ),
        "prediction_confidence_energy_objective_enabled": enabled,
        "prediction_targeted_activity_penalty_replacement_added": replacements,
        "requirement": "pred_conf_energy and pred_conf_energy_sq absent/zero with no prediction-targeted replacement activity penalty",
    }


def strict_shared_mechanism_audit(checkpoints):
    configs = {item["mode"]: dict(item["architecture_contract"]["model_config"]) for item in checkpoints}
    ignored_learned_config_keys = {"decoder_gain", "local_comp_learned_strength"}
    keys = sorted((set(configs["sharpen"]) | set(configs["dampen"])) - ignored_learned_config_keys)
    mismatches = {
        key: {"sharpen": configs["sharpen"].get(key), "dampen": configs["dampen"].get(key)}
        for key in keys
        if configs["sharpen"].get(key) != configs["dampen"].get(key)
    }
    contract_mismatches = {}
    for key in (
        "readout_contract",
        "population_vector_readout",
        "population_normalize",
        "prediction_dependent_inhibition_strength",
        "predicted_feature_suppression_strength",
        "rate_saturation_r_max",
        "rate_saturation_r_half",
        "l23_adapt_strength",
        "l23_adapt_decay",
        "l23_adapt_sigma_channels",
        "l23_local_comp_strength",
        "l23_local_comp_trainable",
        "l23_local_comp_sigma_channels",
        "l23_local_comp_power",
        "l23_local_comp_mode",
    ):
        s_val = checkpoints[0]["architecture_contract"].get(key)
        d_val = checkpoints[1]["architecture_contract"].get(key)
        if s_val != d_val:
            contract_mismatches[key] = {"sharpen": s_val, "dampen": d_val}
    learned_local_comp = {
        item["mode"]: item["architecture_contract"].get("l23_local_comp_learned_strength")
        for item in checkpoints
    }
    local_comp_trainable = any(
        bool(item["architecture_contract"].get("l23_local_comp_trainable"))
        for item in checkpoints
    )
    initial_state_mismatches = {}
    initial_state_evidence = {
        item["mode"]: item["architecture_contract"].get("initial_state_evidence", {})
        for item in checkpoints
    }
    if local_comp_trainable:
        s_init = initial_state_evidence["sharpen"]
        d_init = initial_state_evidence["dampen"]
        required = (
            "shared_state_dict_sha256",
            "local_comp_strength_configured",
            "local_comp_strength_effective",
            "local_comp_strength_raw",
            "local_comp_strength_trainable",
        )
        for key in required:
            if s_init.get(key) != d_init.get(key):
                initial_state_mismatches[key] = {"sharpen": s_init.get(key), "dampen": d_init.get(key)}
            if key not in s_init or key not in d_init:
                initial_state_mismatches[key] = {"sharpen": s_init.get(key), "dampen": d_init.get(key)}
        if not bool(s_init.get("regime_hashes_match")) or not bool(d_init.get("regime_hashes_match")):
            initial_state_mismatches["regime_hashes_match"] = {
                "sharpen": s_init.get("regime_hashes_match"),
                "dampen": d_init.get("regime_hashes_match"),
            }
    return {
        "pass": bool(not mismatches and not contract_mismatches and not initial_state_mismatches),
        "compared_model_config_keys": keys,
        "ignored_learned_model_config_keys": sorted(ignored_learned_config_keys),
        "model_config_mismatches": mismatches,
        "architecture_contract_mismatches": contract_mismatches,
        "initial_state_mismatches": initial_state_mismatches,
        "initial_state_evidence": initial_state_evidence,
        "learned_local_comp_strength": learned_local_comp,
        "allowed_regime_differences": "learned parameters after training and scalar objective weights only",
    }


def main():
    ap = argparse.ArgumentParser(description="Validate tuned fixed-basis emergence checkpoints.")
    ap.add_argument("--sharpen", required=True)
    ap.add_argument("--dampen", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--energy-margin", type=float, default=1e-4)
    ap.add_argument("--decode-margin", type=float, default=1e-4)
    ap.add_argument("--noisy-decode-margin", type=float, default=1e-4)
    ap.add_argument("--noisy-prob-margin", type=float, default=1e-4)
    ap.add_argument("--shape-margin", type=float, default=1e-3)
    ap.add_argument("--held-min", type=float, default=75.0)
    ap.add_argument("--held-batch", type=int, default=8192)
    ap.add_argument("--noise-sigma", type=float, default=1.0)
    ap.add_argument("--noise-repeats", type=int, default=16)
    ap.add_argument("--min-sharpen-expected-noisy-acc", type=float, default=0.80)
    ap.add_argument("--sharpen-far-max-frac", type=float, default=0.65)
    ap.add_argument("--dampen-center-min", type=float, default=0.05)
    ap.add_argument("--dampen-far-flank-slack", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--require-strict-shared-mechanism", action="store_true")
    ap.add_argument("--require-zero-pred-conf", action="store_true")
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args()

    checkpoints = [
        validate_one("sharpen", args.sharpen, "sharpen", args, 100),
        validate_one("dampen", args.dampen, "dampen", args, 200),
    ]
    result = {
        "device": device,
        "torch_version": torch.__version__,
        "K": K,
        "velocities": list(VELS),
        "criteria": {
            "n_pairs": 216,
            "energy_every_pair_delta_lte": -args.energy_margin,
            "held_min_percent": args.held_min,
            "shape_margin": args.shape_margin,
            "noise_sigma": args.noise_sigma,
            "noise_repeats": args.noise_repeats,
            "min_sharpen_expected_noisy_acc": args.min_sharpen_expected_noisy_acc,
        },
        "checkpoints": checkpoints,
        "previous_pass": all(item["previous_pass"] for item in checkpoints),
        "strict_pass": all(item["strict_pass"] for item in checkpoints),
        "strict_shared_mechanism_audit": strict_shared_mechanism_audit(checkpoints),
        "zero_pred_conf_audit": zero_pred_conf_audit(checkpoints),
    }
    result["pass"] = bool(result["previous_pass"] and result["strict_pass"])
    if args.require_strict_shared_mechanism:
        result["pass"] = bool(result["pass"] and result["strict_shared_mechanism_audit"]["pass"])
    if args.require_zero_pred_conf:
        result["pass"] = bool(result["pass"] and result["zero_pred_conf_audit"]["pass"])
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    if not result["pass"] and not args.no_fail:
        print("TUNED_VALIDATION_PASS=False", file=sys.stderr)
        return 1
    print(f"TUNED_VALIDATION_PASS={result['pass']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
