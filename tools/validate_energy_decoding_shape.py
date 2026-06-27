#!/usr/bin/env python3
"""Strict paired energy/decoding validator for repaired Phase-A checkpoints.

The validation set is deterministic: all 36 starts crossed with six velocities,
with an expected continuation at t=4 and a 90-degree orthogonal unexpected
continuation sharing the same prefix. The primary pass criterion is that every
pair has lower mean L2/3 activity for the expected continuation.
"""

import argparse
import json
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


def load_net(path):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "net" in obj:
        obj = obj["net"]
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    net = SimpleNet(use_circuit=True).to(device)
    net.load_state_dict(obj)
    net.eval()
    return net


def stats(x):
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def align_profiles(r, center_idx):
    return aligned_stack(r, center_idx).mean(0).detach().float().cpu().tolist()


def aligned_stack(r, center_idx):
    return torch.stack([torch.roll(row, shifts=-int(center_idx[i].item()), dims=0) for i, row in enumerate(r)], 0)


def shape_metrics(decode_mode, r_e, r_u, r_floor, e_idx, u_idx, margin):
    aligned_e = aligned_stack(r_e, e_idx)
    aligned_u = aligned_stack(r_u, u_idx)
    aligned_floor = aligned_stack(r_floor, e_idx)
    profile = aligned_e.mean(0)
    floor_profile = aligned_floor.mean(0)
    near = profile[[1, -1]].mean()
    flank = profile[[2, -2]].mean()
    far = profile[N // 2]
    center = profile[0]
    floor_center = floor_profile[0]

    if decode_mode == "sharpen":
        checks = {
            "center_above_feedback_off_floor": bool(center.item() >= floor_center.item() + margin),
            "center_above_near": bool(center.item() >= near.item() + margin),
            "center_above_flank": bool(center.item() >= flank.item() + margin),
            "center_above_far": bool(center.item() >= far.item() + margin),
        }
    elif decode_mode == "dampen":
        checks = {
            "center_below_feedback_off_floor": bool(center.item() <= floor_center.item() - margin),
        }
    else:
        raise ValueError(f"unknown decode mode {decode_mode!r}")

    return {
        "pass": all(checks.values()),
        "checks": checks,
        "shape_margin": margin,
        "expected_center": float(center.item()),
        "expected_near_mean_offsets_1": float(near.item()),
        "expected_flank_mean_offsets_2": float(flank.item()),
        "expected_far_offset_18": float(far.item()),
        "feedback_off_floor_center": float(floor_center.item()),
        "feedback_off_floor_near_mean_offsets_1": float(floor_profile[[1, -1]].mean().item()),
        "feedback_off_floor_flank_mean_offsets_2": float(floor_profile[[2, -2]].mean().item()),
        "feedback_off_floor_far_offset_18": float(floor_profile[N // 2].item()),
        "flank_minus_floor_flank": float((flank - floor_profile[[2, -2]].mean()).item()),
        "far_minus_floor_far": float((far - floor_profile[N // 2]).item()),
        "gains_g_v_g_s_g_sv_g_e_g_ps": None,
        "expected_profile_aligned_to_expected_channel": profile.detach().float().cpu().tolist(),
        "unexpected_profile_aligned_to_unexpected_channel": aligned_u.mean(0).detach().float().cpu().tolist(),
        "feedback_off_floor_profile_aligned_to_expected_channel": floor_profile.detach().float().cpu().tolist(),
        "expected_nonzero_count": stats((r_e != 0).float().sum(dim=1)),
        "unexpected_nonzero_count": stats((r_u != 0).float().sum(dim=1)),
    }


@torch.no_grad()
def held_acc(net, seed, batch, S=12):
    torch.manual_seed(seed)
    theta = make_sequences(batch, S, mode="momentum", p_stay=0.9)
    preds, _ = forward_seq(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


@torch.no_grad()
def validate_one(label, path, decode_mode, args):
    net = load_net(path)
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    _, r_all_e = forward_seq(net, theta_e, 1.0)
    _, r_all_u = forward_seq(net, theta_u, 1.0)
    _, r_all_floor = forward_seq(net, theta_e, 0.0)
    r_e = r_all_e[:, K, :]
    r_u = r_all_u[:, K, :]
    r_floor = r_all_floor[:, K, :]

    E_e = r_e.abs().mean(dim=1)
    E_u = r_u.abs().mean(dim=1)
    dE = E_e - E_u
    sq_e = (r_e * r_e).sum(dim=1)
    sq_u = (r_u * r_u).sum(dim=1)
    d_sq = sq_e - sq_u

    logits_e = net.decoder(r_e)
    logits_u = net.decoder(r_u)
    ce_e = F.cross_entropy(logits_e, e_idx, reduction="none")
    ce_u = F.cross_entropy(logits_u, u_idx, reduction="none")
    prob_e = logits_e.softmax(-1).gather(1, e_idx.view(-1, 1)).squeeze(1)
    prob_u = logits_u.softmax(-1).gather(1, u_idx.view(-1, 1)).squeeze(1)
    acc_e = logits_e.argmax(-1) == e_idx
    acc_u = logits_u.argmax(-1) == u_idx

    energy_pass = bool((dE <= -args.energy_margin).all().item())
    if decode_mode == "sharpen":
        decode_pass = bool(
            ce_e.mean().item() + args.decode_margin <= ce_u.mean().item()
            and prob_e.mean().item() >= prob_u.mean().item() + args.decode_margin
        )
    elif decode_mode == "dampen":
        decode_pass = bool(
            ce_e.mean().item() >= ce_u.mean().item() + args.decode_margin
            and prob_e.mean().item() + args.decode_margin <= prob_u.mean().item()
        )
    else:
        raise ValueError(f"unknown decode mode {decode_mode!r}")

    held = held_acc(net, args.seed, args.held_batch)
    held_pass = bool(held >= args.held_min)
    gains = F.softplus(net.circ_raw).detach().float().cpu().tolist()
    shape = shape_metrics(decode_mode, r_e, r_u, r_floor, e_idx, u_idx, args.shape_margin)
    shape["gains_g_v_g_s_g_sv_g_e_g_ps"] = gains

    return {
        "label": label,
        "path": path,
        "decode_mode": decode_mode,
        "n_pairs": int(theta_e.shape[0]),
        "energy_mean_abs": {
            "expected": stats(E_e),
            "unexpected": stats(E_u),
            "delta_expected_minus_unexpected": stats(dE),
            "frac_delta_lt_0": float((dE < 0).float().mean().item()),
            "max_allowed_delta": float(-args.energy_margin),
            "pass": energy_pass,
        },
        "energy_sum_sq": {
            "expected": stats(sq_e),
            "unexpected": stats(sq_u),
            "delta_expected_minus_unexpected": stats(d_sq),
            "frac_delta_lt_0": float((d_sq < 0).float().mean().item()),
        },
        "current_decoder": {
            "ce_expected_target": stats(ce_e),
            "ce_unexpected_target": stats(ce_u),
            "target_prob_expected": stats(prob_e),
            "target_prob_unexpected": stats(prob_u),
            "acc_expected_target": float(acc_e.float().mean().item()),
            "acc_unexpected_target": float(acc_u.float().mean().item()),
            "frac_expected_ce_lt_unexpected_ce": float((ce_e < ce_u).float().mean().item()),
            "pass": decode_pass,
        },
        "prediction": {
            "held_acc_percent": held,
            "held_min_percent": args.held_min,
            "pass": held_pass,
        },
        "shape": shape,
        "pass": energy_pass and decode_pass and held_pass and shape["pass"],
    }


def main():
    ap = argparse.ArgumentParser(description="Validate repaired SOM/VIP checkpoints on paired expected-vs-orthogonal energy and decoding.")
    ap.add_argument("--sharpen", required=True, help="path to sharpen checkpoint")
    ap.add_argument("--dampen", required=True, help="path to dampen checkpoint")
    ap.add_argument("--energy-margin", type=float, default=1e-4, help="require E_expected - E_unexpected <= -margin for every pair")
    ap.add_argument("--decode-margin", type=float, default=1e-4, help="mean decoder CE/probability separation margin")
    ap.add_argument("--shape-margin", type=float, default=1e-3, help="margin for shape pass/fail checks")
    ap.add_argument("--held-min", type=float, default=75.0, help="minimum held-out next-step prediction accuracy in percent")
    ap.add_argument("--held-batch", type=int, default=8192, help="held-out batch size")
    ap.add_argument("--seed", type=int, default=12345, help="seed for held-out prediction check")
    ap.add_argument("--no-fail", action="store_true", help="print JSON but exit 0 even if validation fails")
    args = ap.parse_args()

    result = {
        "device": device,
        "torch_version": torch.__version__,
        "K": K,
        "velocities": list(VELS),
        "criteria": {
            "energy_every_pair_delta_lte": -args.energy_margin,
            "held_min_percent": args.held_min,
            "decode_margin": args.decode_margin,
            "shape_margin": args.shape_margin,
        },
        "checkpoints": [
            validate_one("sharpen", args.sharpen, "sharpen", args),
            validate_one("dampen", args.dampen, "dampen", args),
        ],
    }
    result["pass"] = all(item["pass"] for item in result["checkpoints"])
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"] and not args.no_fail:
        print("VALIDATION_PASS=False", file=sys.stderr)
        return 1
    print(f"VALIDATION_PASS={result['pass']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
