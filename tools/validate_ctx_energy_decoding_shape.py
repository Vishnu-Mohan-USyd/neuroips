#!/usr/bin/env python3
"""Validate a repaired Phase-B context SOM/VIP checkpoint.

The validation contract is the built-in ``net.decoder``. External readout
weights are not used for pass criteria. The paired set is deterministic:
36 starts crossed with velocities {-3,-2,-1,+1,+2,+3}, evaluated at K=4
against a 90-degree orthogonal unexpected continuation sharing the prefix.
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


def ctx_batch(value, batch):
    return torch.full((batch, 1), float(value), device=device)


def load_net(path):
    obj = torch.load(path, map_location=device)
    cfg = {}
    if isinstance(obj, dict) and "net" in obj:
        cfg = obj.get("cfg", {})
        state = obj["net"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        cfg = obj.get("cfg", {})
        state = obj["state_dict"]
    else:
        state = obj
    net = SimpleNet(use_circuit=True, context=True).to(device)
    net.load_state_dict(state)
    net.eval()
    return net, cfg


def stats(x):
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def aligned_stack(r, center_idx):
    return torch.stack([torch.roll(row, shifts=-int(center_idx[i].item()), dims=0) for i, row in enumerate(r)], 0)


def target_logit_margin(logits, target):
    target_logit = logits.gather(1, target.view(-1, 1)).squeeze(1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, target.view(-1, 1), True)
    other = logits.masked_fill(mask, -torch.inf).max(dim=1).values
    return target_logit - other


def decode_arrays(logits, target):
    prob = logits.softmax(-1).gather(1, target.view(-1, 1)).squeeze(1)
    return {
        "ce": F.cross_entropy(logits, target, reduction="none"),
        "prob": prob,
        "margin": target_logit_margin(logits, target),
        "acc": logits.argmax(-1) == target,
    }


def noisy_pair_decode(net, r_a, target_a, r_b, target_b, sigma, repeats, seed):
    chunks_a = {"ce": [], "prob": [], "margin": [], "acc": []}
    chunks_b = {"ce": [], "prob": [], "margin": [], "acc": []}
    for i in range(repeats):
        torch.manual_seed(seed + i)
        eps = torch.randn_like(r_a)
        a = decode_arrays(net.decoder(r_a + sigma * eps), target_a)
        b = decode_arrays(net.decoder(r_b + sigma * eps), target_b)
        for key in chunks_a:
            chunks_a[key].append(a[key].float())
            chunks_b[key].append(b[key].float())
    return {k: torch.cat(v) for k, v in chunks_a.items()}, {k: torch.cat(v) for k, v in chunks_b.items()}


def better_than(a, b, args):
    return bool(
        a["ce"].mean().item() + args.decode_margin <= b["ce"].mean().item()
        and a["prob"].mean().item() >= b["prob"].mean().item() + args.decode_margin
        and a["margin"].mean().item() >= b["margin"].mean().item() + args.logit_margin
    )


def decode_report(label_a, a, label_b, b, args, expect):
    if expect == "a_better":
        passed = better_than(a, b, args)
    elif expect == "b_better":
        passed = better_than(b, a, args)
    else:
        raise ValueError(f"unknown decode expectation {expect!r}")
    return {
        "left": label_a,
        "right": label_b,
        "expectation": expect,
        "pass": passed,
        f"{label_a}_ce": stats(a["ce"]),
        f"{label_b}_ce": stats(b["ce"]),
        f"{label_a}_target_prob": stats(a["prob"]),
        f"{label_b}_target_prob": stats(b["prob"]),
        f"{label_a}_target_logit_margin": stats(a["margin"]),
        f"{label_b}_target_logit_margin": stats(b["margin"]),
        f"{label_a}_acc": float(a["acc"].float().mean().item()),
        f"{label_b}_acc": float(b["acc"].float().mean().item()),
        f"frac_{label_a}_ce_lt_{label_b}_ce": float((a["ce"] < b["ce"]).float().mean().item()),
        f"frac_{label_a}_prob_gt_{label_b}_prob": float((a["prob"] > b["prob"]).float().mean().item()),
        "decode_margin": args.decode_margin,
        "logit_margin": args.logit_margin,
    }


def shape_metrics(mode, r_e, r_u, r_floor, e_idx, u_idx, args):
    aligned_e = aligned_stack(r_e, e_idx)
    aligned_u = aligned_stack(r_u, u_idx)
    aligned_floor = aligned_stack(r_floor, e_idx)
    profile = aligned_e.mean(0)
    floor_profile = aligned_floor.mean(0)
    center = profile[0]
    floor_center = floor_profile[0]
    near = profile[[1, -1]].mean()
    flank = profile[[2, -2]].mean()
    far = profile[N // 2]
    surround = profile[torch.arange(N, device=profile.device) != 0].mean()
    if mode == "attend":
        checks = {
            "center_above_feedback_off_floor": bool(center.item() >= floor_center.item() + args.shape_margin),
            "center_above_near": bool(center.item() >= near.item() + args.shape_margin),
            "center_above_flank": bool(center.item() >= flank.item() + args.shape_margin),
            "center_above_far": bool(center.item() >= far.item() + args.shape_margin),
            "center_above_surround_mean": bool(center.item() >= surround.item() + args.shape_margin),
        }
    elif mode == "save":
        checks = {
            "center_below_feedback_off_floor": bool(center.item() <= floor_center.item() - args.shape_margin),
        }
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "shape_margin": args.shape_margin,
        "expected_center": float(center.item()),
        "expected_near_mean_offsets_1": float(near.item()),
        "expected_flank_mean_offsets_2": float(flank.item()),
        "expected_far_offset_18": float(far.item()),
        "expected_surround_mean": float(surround.item()),
        "feedback_off_floor_center": float(floor_center.item()),
        "feedback_off_floor_near_mean_offsets_1": float(floor_profile[[1, -1]].mean().item()),
        "feedback_off_floor_flank_mean_offsets_2": float(floor_profile[[2, -2]].mean().item()),
        "feedback_off_floor_far_offset_18": float(floor_profile[N // 2].item()),
        "feedback_off_floor_surround_mean": float(
            floor_profile[torch.arange(N, device=floor_profile.device) != 0].mean().item()
        ),
        "flank_minus_floor_flank": float((flank - floor_profile[[2, -2]].mean()).item()),
        "far_minus_floor_far": float((far - floor_profile[N // 2]).item()),
        "expected_profile_aligned_to_expected_channel": profile.detach().float().cpu().tolist(),
        "unexpected_profile_aligned_to_unexpected_channel": aligned_u.mean(0).detach().float().cpu().tolist(),
        "feedback_off_floor_profile_aligned_to_expected_channel": floor_profile.detach().float().cpu().tolist(),
        "expected_nonzero_count": stats((r_e != 0).float().sum(dim=1)),
        "unexpected_nonzero_count": stats((r_u != 0).float().sum(dim=1)),
    }


@torch.no_grad()
def held_acc(net, cv, seed, batch):
    torch.manual_seed(seed)
    theta = make_sequences(batch, 12, mode="momentum", p_stay=0.9)
    preds, _ = forward_seq(net, theta, 1.0, ctx=ctx_batch(cv, batch))
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


@torch.no_grad()
def measure_context(net, cv, label, args, theta_e, theta_u, e_idx, u_idx, seed_offset):
    cb = ctx_batch(cv, theta_e.shape[0])
    _, r_all_e = forward_seq(net, theta_e, 1.0, ctx=cb)
    _, r_all_u = forward_seq(net, theta_u, 1.0, ctx=cb)
    _, r_all_floor = forward_seq(net, theta_e, 0.0, ctx=cb)
    r_e = r_all_e[:, K, :]
    r_u = r_all_u[:, K, :]
    r_floor = r_all_floor[:, K, :]

    E_e = r_e.abs().mean(dim=1)
    E_u = r_u.abs().mean(dim=1)
    dE = E_e - E_u
    sq_e = r_e.square().sum(dim=1)
    sq_u = r_u.square().sum(dim=1)
    d_sq = sq_e - sq_u
    energy_pass = bool((dE <= -args.energy_margin).all().item())

    clean_e = decode_arrays(net.decoder(r_e), e_idx)
    clean_u = decode_arrays(net.decoder(r_u), u_idx)
    noisy_e, noisy_u = noisy_pair_decode(
        net, r_e, e_idx, r_u, u_idx, args.noise_sigma, args.noise_repeats, args.seed + seed_offset
    )
    expect = "a_better" if label == "attend" else "b_better"
    clean_report = decode_report("expected", clean_e, "unexpected", clean_u, args, expect)
    noisy_report = decode_report("expected", noisy_e, "unexpected", noisy_u, args, expect)
    held = held_acc(net, cv, args.seed + seed_offset + 1000, args.held_batch)
    shape = shape_metrics(label, r_e, r_u, r_floor, e_idx, u_idx, args)

    report = {
        "label": label,
        "ctx": float(cv),
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
        "current_decoder_clean": clean_report,
        "current_decoder_noisy": noisy_report,
        "prediction": {
            "held_acc_percent": held,
            "held_min_percent": args.held_min,
            "pass": bool(held >= args.held_min),
        },
        "shape": shape,
    }
    report["pass"] = bool(
        energy_pass
        and clean_report["pass"]
        and noisy_report["pass"]
        and report["prediction"]["pass"]
        and shape["pass"]
    )
    raw = {
        "r_e": r_e,
        "r_u": r_u,
        "r_floor": r_floor,
        "E_e": E_e,
        "E_u": E_u,
        "clean_e": clean_e,
        "clean_u": clean_u,
        "noisy_e": noisy_e,
        "noisy_u": noisy_u,
    }
    return report, raw


@torch.no_grad()
def cross_context_metrics(net, attend_raw, save_raw, e_idx, args):
    d_energy = save_raw["E_e"] - attend_raw["E_e"]
    energy_pass = bool((d_energy <= -args.context_energy_margin).all().item())
    clean = decode_report(
        "attend_expected",
        attend_raw["clean_e"],
        "save_expected",
        save_raw["clean_e"],
        args,
        "a_better",
    )
    noisy_att, noisy_save = noisy_pair_decode(
        net,
        attend_raw["r_e"],
        e_idx,
        save_raw["r_e"],
        e_idx,
        args.noise_sigma,
        args.noise_repeats,
        args.seed + 5000,
    )
    noisy = decode_report("attend_expected", noisy_att, "save_expected", noisy_save, args, "a_better")
    return {
        "expected_energy_save_minus_attend": {
            "delta": stats(d_energy),
            "frac_save_lower": float((d_energy < 0).float().mean().item()),
            "max_allowed_delta": float(-args.context_energy_margin),
            "pass": energy_pass,
        },
        "expected_decoder_clean_attend_vs_save": clean,
        "expected_decoder_noisy_attend_vs_save": noisy,
        "pass": bool(energy_pass and clean["pass"] and noisy["pass"]),
    }


@torch.no_grad()
def legacy_gate_metrics(net, attend_report, save_report, attend_raw, save_raw, e_idx, args):
    floor_delta = attend_raw["r_floor"] - save_raw["r_floor"]
    floor_parity = bool(floor_delta.abs().max().item() <= args.floor_tolerance)
    k1 = attend_report["shape"]["checks"]["center_above_feedback_off_floor"]
    k2 = save_report["shape"]["checks"]["center_below_feedback_off_floor"]
    k3 = attend_report["prediction"]["pass"] and save_report["prediction"]["pass"]
    saved = net.g_ctx_raw.data.clone()
    try:
        net.g_ctx_raw.data.fill_(-20.0)
        theta_e, _, _, _ = build_pairs()
        cb_att = ctx_batch(1.0, theta_e.shape[0])
        cb_save = ctx_batch(-1.0, theta_e.shape[0])
        _, r_att_on = forward_seq(net, theta_e, 1.0, ctx=cb_att)
        _, r_att_floor = forward_seq(net, theta_e, 0.0, ctx=cb_att)
        _, r_save_on = forward_seq(net, theta_e, 1.0, ctx=cb_save)
        _, r_save_floor = forward_seq(net, theta_e, 0.0, ctx=cb_save)
        att_delta = (
            r_att_on[:, K, :].gather(1, e_idx.view(-1, 1))
            - r_att_floor[:, K, :].gather(1, e_idx.view(-1, 1))
        ).squeeze(1)
        save_delta = (
            r_save_on[:, K, :].gather(1, e_idx.view(-1, 1))
            - r_save_floor[:, K, :].gather(1, e_idx.view(-1, 1))
        ).squeeze(1)
        lesion_gap = (att_delta - save_delta).abs()
    finally:
        net.g_ctx_raw.data.copy_(saved)
    lesion_collapses = bool(lesion_gap.max().item() <= args.floor_tolerance)
    k4 = bool(k1 and k2 and floor_parity and lesion_collapses)
    return {
        "K1_attend_sharpens": bool(k1),
        "K2_save_dampens": bool(k2),
        "K3_held_both_contexts": bool(k3),
        "K4_context_flip_floor_parity_gctx_lesion": bool(k4),
        "floor_parity": {
            "max_abs_floor_delta": float(floor_delta.abs().max().item()),
            "mean_abs_floor_delta": float(floor_delta.abs().mean().item()),
            "tolerance": args.floor_tolerance,
            "pass": floor_parity,
        },
        "g_ctx_lesion": {
            "max_abs_attend_save_exp_floor_delta_gap": float(lesion_gap.max().item()),
            "mean_abs_attend_save_exp_floor_delta_gap": float(lesion_gap.mean().item()),
            "tolerance": args.floor_tolerance,
            "pass": lesion_collapses,
        },
        "pass": bool(k1 and k2 and k3 and k4),
    }


def main():
    ap = argparse.ArgumentParser(description="Validate Phase-B context energy/decoding/shape repair.")
    ap.add_argument("--ckpt", required=True, help="path to repaired {'net','cfg'} checkpoint")
    ap.add_argument("--json-out", help="optional path for full JSON report")
    ap.add_argument("--energy-margin", type=float, default=1e-4)
    ap.add_argument("--context-energy-margin", type=float, default=1e-4)
    ap.add_argument("--decode-margin", type=float, default=1e-4)
    ap.add_argument("--logit-margin", type=float, default=1e-4)
    ap.add_argument("--shape-margin", type=float, default=1e-3)
    ap.add_argument("--floor-tolerance", type=float, default=1e-3)
    ap.add_argument("--held-min", type=float, default=75.0)
    ap.add_argument("--held-batch", type=int, default=8192)
    ap.add_argument("--noise-sigma", type=float, default=1.0)
    ap.add_argument("--noise-repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-fail", action="store_true", help="emit report but exit 0 even on validation failure")
    args = ap.parse_args()

    net, cfg = load_net(args.ckpt)
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    attend_report, attend_raw = measure_context(net, 1.0, "attend", args, theta_e, theta_u, e_idx, u_idx, 10)
    save_report, save_raw = measure_context(net, -1.0, "save", args, theta_e, theta_u, e_idx, u_idx, 20)
    cross = cross_context_metrics(net, attend_raw, save_raw, e_idx, args)
    legacy = legacy_gate_metrics(net, attend_report, save_report, attend_raw, save_raw, e_idx, args)

    result = {
        "device": device,
        "torch_version": torch.__version__,
        "checkpoint": args.ckpt,
        "checkpoint_cfg": cfg,
        "decoder_contract": "built_in_net.decoder",
        "external_readout_used_for_pass_criteria": False,
        "K": K,
        "velocities": list(VELS),
        "criteria": {
            "n_pairs": int(theta_e.shape[0]),
            "energy_every_pair_delta_lte": -args.energy_margin,
            "context_energy_every_pair_delta_lte": -args.context_energy_margin,
            "decode_margin": args.decode_margin,
            "logit_margin": args.logit_margin,
            "shape_margin": args.shape_margin,
            "held_min_percent": args.held_min,
            "noise_sigma": args.noise_sigma,
            "noise_repeats": args.noise_repeats,
        },
        "contexts": [attend_report, save_report],
        "cross_context": cross,
        "phaseB_legacy_equivalent": legacy,
        "gains_g_v_g_s_g_sv_g_e_g_ps": [float(x) for x in F.softplus(net.circ_raw).detach().cpu().tolist()],
        "g_ctx": float(F.softplus(net.g_ctx_raw).item()),
    }
    result["pass"] = bool(
        all(item["pass"] for item in result["contexts"]) and cross["pass"] and legacy["pass"]
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    if not result["pass"] and not args.no_fail:
        print("CTX_VALIDATION_PASS=False", file=sys.stderr)
        return 1
    print(f"CTX_VALIDATION_PASS={result['pass']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
