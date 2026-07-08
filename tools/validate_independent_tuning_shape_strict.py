#!/usr/bin/env python3
"""Strict raw-tuning validator for independent repaired SOM/VIP checkpoints.

This validator keeps the existing paired expectation-energy contract and adds
raw L2/3 tuning-shape checks. It uses feedback-on raw ``r_all[:, 4, :]`` aligned
to the expected final orientation; no feedback-off floor subtraction is used for
shape pass/fail.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simple_net import N, STEP_DEG, SimpleNet, chan, device, forward_seq, make_sequences  # noqa: E402


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
    theta_e = torch.tensor(rows_e, device=device).float() * STEP_DEG
    theta_u = torch.tensor(rows_u, device=device).float() * STEP_DEG
    e_idx = torch.tensor(e_vals, device=device, dtype=torch.long)
    u_idx = torch.tensor(u_vals, device=device, dtype=torch.long)
    return theta_e, theta_u, e_idx, u_idx


def apply_model_config(net, config):
    for key in (
        "l23_competition_strength",
        "l23_competition_sigma_channels",
        "l23_competition_radius",
        "l23_competition_global_strength",
        "l23_local_inhibition_strength",
        "l23_local_inhibition_sigma_channels",
        "l23_local_inhibition_radius",
        "l23_local_inhibition_center_weight",
        "l23_feedback_gated_inhibition_strength",
        "l23_feedback_gated_inhibition_sigma_channels",
        "l23_feedback_gated_inhibition_radius",
        "l23_feedback_gated_inhibition_center_weight",
        "som_feedback_pool_strength",
        "som_feedback_pool_sigma_channels",
        "som_feedback_pool_radius",
        "som_feedback_pool_center_weight",
        "somvip_topographic_som_strength",
        "somvip_topographic_som_sigma_channels",
        "somvip_topographic_som_radius",
        "somvip_topographic_som_center_weight",
        "somvip_topographic_vip_strength",
        "somvip_topographic_vip_sigma_channels",
        "somvip_topographic_vip_radius",
        "somvip_topographic_vip_center_weight",
        "l23_prediction_error_strength",
    ):
        if key in config:
            setattr(net, key, config[key])


def model_config(net):
    return {
        "l23_competition_strength": float(getattr(net, "l23_competition_strength", 0.0)),
        "l23_competition_sigma_channels": float(getattr(net, "l23_competition_sigma_channels", 2.0)),
        "l23_competition_radius": int(getattr(net, "l23_competition_radius", 4)),
        "l23_competition_global_strength": float(getattr(net, "l23_competition_global_strength", 0.0)),
        "l23_local_inhibition_strength": float(getattr(net, "l23_local_inhibition_strength", 0.0)),
        "l23_local_inhibition_sigma_channels": float(getattr(net, "l23_local_inhibition_sigma_channels", 1.5)),
        "l23_local_inhibition_radius": int(getattr(net, "l23_local_inhibition_radius", 3)),
        "l23_local_inhibition_center_weight": float(getattr(net, "l23_local_inhibition_center_weight", 0.0)),
        "l23_feedback_gated_inhibition_strength": float(getattr(net, "l23_feedback_gated_inhibition_strength", 0.0)),
        "l23_feedback_gated_inhibition_sigma_channels": float(getattr(net, "l23_feedback_gated_inhibition_sigma_channels", 1.5)),
        "l23_feedback_gated_inhibition_radius": int(getattr(net, "l23_feedback_gated_inhibition_radius", 3)),
        "l23_feedback_gated_inhibition_center_weight": float(getattr(net, "l23_feedback_gated_inhibition_center_weight", 1.0)),
        "som_feedback_pool_strength": float(getattr(net, "som_feedback_pool_strength", 0.0)),
        "som_feedback_pool_sigma_channels": float(getattr(net, "som_feedback_pool_sigma_channels", 1.5)),
        "som_feedback_pool_radius": int(getattr(net, "som_feedback_pool_radius", 3)),
        "som_feedback_pool_center_weight": float(getattr(net, "som_feedback_pool_center_weight", 0.0)),
        "somvip_topographic_som_strength": float(getattr(net, "somvip_topographic_som_strength", 0.0)),
        "somvip_topographic_som_sigma_channels": float(getattr(net, "somvip_topographic_som_sigma_channels", 0.75)),
        "somvip_topographic_som_radius": int(getattr(net, "somvip_topographic_som_radius", 2)),
        "somvip_topographic_som_center_weight": float(getattr(net, "somvip_topographic_som_center_weight", 1.0)),
        "somvip_topographic_vip_strength": float(getattr(net, "somvip_topographic_vip_strength", 0.0)),
        "somvip_topographic_vip_sigma_channels": float(getattr(net, "somvip_topographic_vip_sigma_channels", 2.5)),
        "somvip_topographic_vip_radius": int(getattr(net, "somvip_topographic_vip_radius", 5)),
        "somvip_topographic_vip_center_weight": float(getattr(net, "somvip_topographic_vip_center_weight", 0.0)),
        "l23_prediction_error_strength": float(getattr(net, "l23_prediction_error_strength", 0.0)),
    }


def load_net(path):
    obj = torch.load(path, map_location=device)
    config = obj.get("simple_net_config", {}) if isinstance(obj, dict) else {}
    if isinstance(obj, dict) and "net" in obj:
        obj = obj["net"]
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    net = SimpleNet(use_circuit=True).to(device)
    apply_model_config(net, config)
    net.load_state_dict(obj)
    net.eval()
    return net


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
    preds, _ = forward_seq(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


@torch.no_grad()
def noisy_decoder(net, r, target, sigma, repeats, seed):
    torch.manual_seed(seed)
    B = r.shape[0]
    noise = sigma * torch.randn(repeats, B, N, device=device)
    logits = net.decoder((r.unsqueeze(0) + noise).reshape(repeats * B, N))
    target_rep = target.repeat(repeats)
    ce = F.cross_entropy(logits, target_rep, reduction="none")
    prob = logits.softmax(-1).gather(1, target_rep.view(-1, 1)).squeeze(1)
    acc = logits.argmax(-1) == target_rep
    return {
        "ce": stats(ce),
        "target_prob": stats(prob),
        "acc": float(acc.float().mean().item()),
    }


@torch.no_grad()
def validate_one(label, path, mode, args, seed_offset):
    net = load_net(path)
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    _, r_all_e = forward_seq(net, theta_e, 1.0)
    _, r_all_u = forward_seq(net, theta_u, 1.0)
    r_e = r_all_e[:, K, :]
    r_u = r_all_u[:, K, :]

    E_e = r_e.abs().mean(dim=1)
    E_u = r_u.abs().mean(dim=1)
    dE = E_e - E_u
    energy_pass = bool((dE <= -args.energy_margin).all().item())

    logits_e = net.decoder(r_e)
    logits_u = net.decoder(r_u)
    ce_e = F.cross_entropy(logits_e, e_idx, reduction="none")
    ce_u = F.cross_entropy(logits_u, u_idx, reduction="none")
    prob_e = logits_e.softmax(-1).gather(1, e_idx.view(-1, 1)).squeeze(1)
    prob_u = logits_u.softmax(-1).gather(1, u_idx.view(-1, 1)).squeeze(1)
    clean = {
        "ce_expected_target": stats(ce_e),
        "ce_unexpected_target": stats(ce_u),
        "target_prob_expected": stats(prob_e),
        "target_prob_unexpected": stats(prob_u),
    }
    if mode == "sharpen":
        clean_pass = bool(
            ce_e.mean().item() + args.decode_margin <= ce_u.mean().item()
            and prob_e.mean().item() >= prob_u.mean().item() + args.decode_margin
        )
    elif mode == "dampen":
        clean_pass = bool(
            ce_e.mean().item() >= ce_u.mean().item() + args.decode_margin
            and prob_e.mean().item() + args.decode_margin <= prob_u.mean().item()
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    clean["pass"] = clean_pass

    noisy_e = noisy_decoder(net, r_e, e_idx, args.noise_sigma, args.noise_repeats, args.seed + seed_offset)
    noisy_u = noisy_decoder(net, r_u, u_idx, args.noise_sigma, args.noise_repeats, args.seed + seed_offset + 1)
    if mode == "sharpen":
        noisy_pass = bool(
            noisy_e["ce"]["mean"] + args.noisy_decode_margin <= noisy_u["ce"]["mean"]
            and noisy_e["target_prob"]["mean"] >= noisy_u["target_prob"]["mean"] + args.noisy_prob_margin
        )
    else:
        noisy_pass = bool(
            noisy_e["ce"]["mean"] >= noisy_u["ce"]["mean"] + args.noisy_decode_margin
            and noisy_e["target_prob"]["mean"] + args.noisy_prob_margin <= noisy_u["target_prob"]["mean"]
        )

    aligned = aligned_stack(r_e, e_idx)
    profile = aligned.mean(0)
    profile_sem = aligned.std(0, unbiased=False) / math.sqrt(aligned.shape[0])
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

    shape = {
        "pass": all(checks.values()),
        "checks": checks,
        "raw_center": float(center.item()),
        "raw_near_mean_offsets_5deg": float(near.item()),
        "raw_flank_mean_offsets_10deg": float(flank.item()),
        "raw_shoulder_mean_offsets_15_20deg": float(shoulder.item()),
        "raw_far_mean_offsets_50_55_60deg": float(far.item()),
        "raw_profile_aligned_to_expected": profile.detach().float().cpu().tolist(),
        "raw_profile_sem_aligned_to_expected": profile_sem.detach().float().cpu().tolist(),
    }

    held = held_acc(net, args.seed + seed_offset, args.held_batch)
    prediction = {
        "held_acc_percent": held,
        "held_min_percent": args.held_min,
        "pass": bool(held >= args.held_min),
    }
    gains = F.softplus(net.circ_raw).detach().float().cpu().tolist()
    return {
        "label": label,
        "path": path,
        "mode": mode,
        "n_pairs": int(theta_e.shape[0]),
        "architecture_contract": {
            "simple_net_use_circuit": True,
            "simple_net_context": False,
            "simple_net_config": model_config(net),
            "state_dict_keys": sorted(net.state_dict().keys()),
            "gains_g_v_g_s_g_sv_g_e_g_ps": gains,
        },
        "energy_mean_abs": {
            "expected": stats(E_e),
            "unexpected": stats(E_u),
            "delta_expected_minus_unexpected": stats(dE),
            "frac_delta_lt_0": float((dE < 0).float().mean().item()),
            "max_allowed_delta": float(-args.energy_margin),
            "pass": energy_pass,
        },
        "current_decoder_clean": clean,
        "current_decoder_noisy": {
            "sigma": args.noise_sigma,
            "repeats": args.noise_repeats,
            "expected": noisy_e,
            "unexpected": noisy_u,
            "pass": noisy_pass,
        },
        "prediction": prediction,
        "raw_tuning_shape": shape,
        "pass": bool(energy_pass and clean_pass and noisy_pass and prediction["pass"] and shape["pass"]),
    }


def main():
    ap = argparse.ArgumentParser(description="Strict raw-tuning validation for independent repaired checkpoints.")
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
    ap.add_argument("--sharpen-far-max-frac", type=float, default=0.65)
    ap.add_argument("--dampen-center-min", type=float, default=0.05)
    ap.add_argument("--dampen-far-flank-slack", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args()

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
            "sharpen_far_max_frac": args.sharpen_far_max_frac,
            "dampen_center_min": args.dampen_center_min,
            "dampen_far_flank_slack": args.dampen_far_flank_slack,
            "noise_sigma": args.noise_sigma,
            "noise_repeats": args.noise_repeats,
        },
        "checkpoints": [
            validate_one("sharpen", args.sharpen, "sharpen", args, 100),
            validate_one("dampen", args.dampen, "dampen", args, 200),
        ],
    }
    result["pass"] = all(item["pass"] for item in result["checkpoints"])
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    if not result["pass"] and not args.no_fail:
        print("STRICT_TUNING_VALIDATION_PASS=False", file=sys.stderr)
        return 1
    print(f"STRICT_TUNING_VALIDATION_PASS={result['pass']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
