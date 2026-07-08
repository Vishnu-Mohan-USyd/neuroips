#!/usr/bin/env python3
"""Train independent SOM/VIP checkpoints with raw tuning-shape safeguards.

This additive driver keeps the SimpleNet architecture unchanged. It adds
biologically motivated objective terms: local/sparse feedforward representation
pressure, population sparsity, paired expected-vs-orthogonal energy contrast,
and bounded prediction-derived dampen suppression that preserves nonzero
expected-channel evidence.
"""

import argparse
import copy
import json
import math
import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simple_net import N, STEP_DEG, SimpleNet, chan, device, forward_seq, l4_code, make_sequences  # noqa: E402


K = 4
VELS = (-3, -2, -1, 1, 2, 3)


def circular_distance_matrix():
    idx = torch.arange(N, device=device)
    d = (idx[:, None] - idx[None, :]).abs()
    return torch.minimum(d, N - d).float()


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


def aligned_stack(r, center_idx):
    return torch.stack([torch.roll(row, shifts=-int(center_idx[i].item()), dims=0) for i, row in enumerate(r)], 0)


def ff_locality_loss(net, locality_penalty):
    W = net.W_ff.weight
    return (W.square() * locality_penalty).sum() / N + net.W_ff.bias.square().mean()


def locality_penalty(args):
    d = circular_distance_matrix()
    return 1.0 - torch.exp(-0.5 * (d / args.ff_local_sigma_channels).square())


def phase1_local(net, args, loc_penalty):
    opt = torch.optim.Adam(list(net.W_ff.parameters()) + list(net.decoder.parameters()), lr=args.phase1_lr)
    history = []
    print("\n=== PHASE 1 LOCAL/SPARSE REPRESENTATION ===", flush=True)
    for step in range(1, args.phase1_steps + 1):
        theta = torch.randint(0, N, (args.batch,), device=device).float() * STEP_DEG
        r = net.l23(l4_code(theta), torch.zeros(args.batch, N, device=device))
        logits = net.decoder(r)
        target = chan(theta)
        current_ce = F.cross_entropy(logits, target)
        activity = r.abs().mean()
        activity_sq = r.square().mean()
        local = ff_locality_loss(net, loc_penalty)
        loss = (
            current_ce
            + args.phase1_activity_weight * activity
            + args.phase1_activity_sq_weight * activity_sq
            + args.ff_locality_weight * local
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.phase1_steps:
            acc = (logits.argmax(-1) == target).float().mean().item() * 100.0
            row = {
                "step": step,
                "loss": float(loss.item()),
                "current_ce": float(current_ce.item()),
                "rep_acc": float(acc),
                "activity": float(activity.item()),
                "activity_sq": float(activity_sq.item()),
                "ff_locality": float(local.item()),
            }
            history.append(row)
            print(json.dumps({"stage": "phase1", **row}, sort_keys=True), flush=True)
    return history


def prediction_mask(pred, args):
    mask = F.softmax(pred.detach() / args.prediction_mask_temp, dim=1)
    if args.prediction_mask_topk > 0:
        keep = torch.zeros_like(mask)
        keep.scatter_(1, mask.topk(args.prediction_mask_topk, dim=1).indices, 1.0)
        mask = mask * keep
    return mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)


def pair_losses(net, regime, theta_e, theta_u, e_idx, u_idx, args):
    preds_e, r_all_e = forward_seq(net, theta_e, 1.0)
    _, r_all_u = forward_seq(net, theta_u, 1.0)
    _, r_all_floor = forward_seq(net, theta_e, 0.0)
    r_e = r_all_e[:, K, :]
    r_u = r_all_u[:, K, :]
    r_floor = r_all_floor[:, K, :].detach()

    E_e = r_e.abs().mean(dim=1)
    E_u = r_u.abs().mean(dim=1)
    energy_violation = F.relu(E_e - E_u + args.energy_margin)
    energy_loss = energy_violation.mean() + energy_violation.max()

    logits_e = net.decoder(r_e)
    logits_u = net.decoder(r_u)
    ce_e = F.cross_entropy(logits_e, e_idx, reduction="none")
    ce_u = F.cross_entropy(logits_u, u_idx, reduction="none")
    prob_e = logits_e.softmax(-1).gather(1, e_idx.view(-1, 1)).squeeze(1)
    prob_u = logits_u.softmax(-1).gather(1, u_idx.view(-1, 1)).squeeze(1)
    if regime == "sharpen":
        decode_violation = F.relu(ce_e - ce_u + args.decode_margin)
        prob_violation = F.relu(prob_u - prob_e + args.decode_prob_margin)
        decode_loss = ce_e.mean() + args.rank_weight * (
            decode_violation.mean() + decode_violation.max() + prob_violation.mean() + prob_violation.max()
        )
    elif regime == "dampen":
        decode_violation = F.relu(ce_u - ce_e + args.decode_margin)
        prob_violation = F.relu(prob_e - prob_u + args.decode_prob_margin)
        decode_loss = ce_u.mean() + args.rank_weight * (
            decode_violation.mean() + decode_violation.max() + prob_violation.mean() + prob_violation.max()
        )
    else:
        raise ValueError(f"unknown regime {regime!r}")

    pair_pred_ce = F.cross_entropy(
        preds_e[:, :-1, :].reshape(-1, N),
        chan(theta_e[:, 1:]).reshape(-1),
    )
    mask = prediction_mask(preds_e[:, K - 1, :], args)
    predicted_activity = (mask * r_e).sum(dim=1)
    predicted_floor = (mask * r_floor).sum(dim=1).detach()
    lower = args.dampen_lower_floor_frac * predicted_floor
    upper = args.dampen_upper_floor_frac * predicted_floor
    below_lower = F.relu(lower - predicted_activity + args.dampen_bound_margin)
    above_upper = F.relu(predicted_activity - upper + args.dampen_bound_margin)
    bounded_suppression = (
        below_lower.mean()
        + below_lower.max()
        + args.dampen_bound_sq_weight * (below_lower.square().mean() + below_lower.square().max())
        + above_upper.mean()
        + above_upper.max()
        + args.dampen_bound_sq_weight * (above_upper.square().mean() + above_upper.square().max())
    )

    aligned_e = aligned_stack(r_e, e_idx)
    raw_profile = aligned_e.mean(0)
    raw_center = raw_profile[0]
    raw_flank = raw_profile[[2, -2]].mean()
    raw_far = raw_profile[[10, 11, 12, -10, -11, -12]].mean()

    return {
        "energy": energy_loss,
        "decode": decode_loss,
        "pair_pred_ce": pair_pred_ce,
        "bounded_suppression": bounded_suppression,
        "max_delta_mean_abs": (E_e - E_u).detach().max(),
        "mean_delta_mean_abs": (E_e - E_u).detach().mean(),
        "frac_delta_lt_0": ((E_e - E_u).detach() < 0).float().mean(),
        "mean_ce_expected": ce_e.detach().mean(),
        "mean_ce_unexpected": ce_u.detach().mean(),
        "mean_prob_expected": prob_e.detach().mean(),
        "mean_prob_unexpected": prob_u.detach().mean(),
        "mean_predicted_activity": predicted_activity.detach().mean(),
        "mean_predicted_floor": predicted_floor.detach().mean(),
        "min_predicted_activity": predicted_activity.detach().min(),
        "raw_center": raw_center.detach(),
        "raw_flank": raw_flank.detach(),
        "raw_far": raw_far.detach(),
    }


def sequence_losses(net, args):
    theta = make_sequences(args.batch, 12, mode="momentum", p_stay=0.9)
    preds, r_all = forward_seq(net, theta, 1.0)
    pred_ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), chan(theta[:, 1:]).reshape(-1))
    current_ce = F.cross_entropy(net.decoder(r_all.reshape(-1, N)), chan(theta).reshape(-1))
    activity = r_all.abs().mean()
    activity_sq = r_all.square().mean()
    return pred_ce, current_ce, activity, activity_sq


@torch.no_grad()
def held_acc(net, seed, batch=4096):
    torch.manual_seed(seed)
    theta = make_sequences(batch, 12, mode="momentum", p_stay=0.9)
    preds, _ = forward_seq(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


def param_groups(net, args):
    fast = list(net.gru.parameters()) + list(net.W_fb.parameters()) + [net.circ_raw]
    slow = list(net.W_ff.parameters()) + list(net.decoder.parameters())
    return [
        {"params": fast, "lr": args.lr},
        {"params": slow, "lr": args.lr_repr},
    ]


def ramped_weight(final_weight, step, warmup_steps, ramp_steps):
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    return final_weight * min(1.0, (step - warmup_steps) / ramp_steps)


def train_one(regime, base_state, out_path, args, loc_penalty):
    net = SimpleNet(use_circuit=True).to(device)
    net.load_state_dict(copy.deepcopy(base_state))
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(param_groups(net, args))
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    history = []
    print(f"\n=== TUNING SHAPE REPAIR {regime.upper()} steps={args.steps} device={device} ===", flush=True)
    for step in range(1, args.steps + 1):
        pred_ce, current_ce, activity, activity_sq = sequence_losses(net, args)
        pair = pair_losses(net, regime, theta_e, theta_u, e_idx, u_idx, args)
        local = ff_locality_loss(net, loc_penalty)

        energy_weight = ramped_weight(args.energy_weight, step, args.energy_warmup_steps, args.energy_ramp_steps)
        bounded_weight = (
            ramped_weight(args.dampen_bounded_weight, step, args.dampen_bounded_warmup_steps, args.dampen_bounded_ramp_steps)
            if regime == "dampen"
            else 0.0
        )

        loss = (
            args.pred_weight * pred_ce
            + args.current_weight * current_ce
            + args.activity_weight * activity
            + args.activity_sq_weight * activity_sq
            + args.pair_pred_weight * pair["pair_pred_ce"]
            + energy_weight * pair["energy"]
            + args.decode_weight * pair["decode"]
            + args.ff_locality_weight * local
            + bounded_weight * pair["bounded_suppression"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            held = held_acc(net, args.seed + step, args.held_batch)
            gains = [round(x, 4) for x in F.softplus(net.circ_raw).detach().cpu().tolist()]
            row = {
                "step": step,
                "loss": float(loss.item()),
                "pred_ce": float(pred_ce.item()),
                "current_ce": float(current_ce.item()),
                "activity": float(activity.item()),
                "activity_sq": float(activity_sq.item()),
                "ff_locality": float(local.item()),
                "energy_weight": float(energy_weight),
                "bounded_suppression_weight": float(bounded_weight),
                "pair_energy": float(pair["energy"].item()),
                "pair_decode": float(pair["decode"].item()),
                "bounded_suppression": float(pair["bounded_suppression"].item()),
                "max_delta_mean_abs": float(pair["max_delta_mean_abs"].item()),
                "mean_delta_mean_abs": float(pair["mean_delta_mean_abs"].item()),
                "frac_delta_lt_0": float(pair["frac_delta_lt_0"].item()),
                "mean_ce_expected": float(pair["mean_ce_expected"].item()),
                "mean_ce_unexpected": float(pair["mean_ce_unexpected"].item()),
                "mean_prob_expected": float(pair["mean_prob_expected"].item()),
                "mean_prob_unexpected": float(pair["mean_prob_unexpected"].item()),
                "mean_predicted_activity": float(pair["mean_predicted_activity"].item()),
                "mean_predicted_floor": float(pair["mean_predicted_floor"].item()),
                "min_predicted_activity": float(pair["min_predicted_activity"].item()),
                "raw_center": float(pair["raw_center"].item()),
                "raw_flank": float(pair["raw_flank"].item()),
                "raw_far": float(pair["raw_far"].item()),
                "held_acc": held,
                "gains": gains,
            }
            history.append(row)
            print(json.dumps({"regime": regime, **row}, sort_keys=True), flush=True)

    torch.save(net.state_dict(), out_path)
    print(f"SAVED {regime} {out_path}", flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description="Train independent raw-tuning repair checkpoints.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase1-steps", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held-batch", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--phase1-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-repr", type=float, default=1e-4)
    ap.add_argument("--ff-local-sigma-channels", type=float, default=2.5)
    ap.add_argument("--ff-locality-weight", type=float, default=4.0)
    ap.add_argument("--phase1-activity-weight", type=float, default=0.08)
    ap.add_argument("--phase1-activity-sq-weight", type=float, default=0.02)
    ap.add_argument("--activity-weight", type=float, default=0.05)
    ap.add_argument("--activity-sq-weight", type=float, default=0.01)
    ap.add_argument("--energy-margin", type=float, default=0.025)
    ap.add_argument("--decode-margin", type=float, default=0.05)
    ap.add_argument("--decode-prob-margin", type=float, default=0.02)
    ap.add_argument("--pred-weight", type=float, default=2.5)
    ap.add_argument("--current-weight", type=float, default=0.8)
    ap.add_argument("--pair-pred-weight", type=float, default=0.8)
    ap.add_argument("--energy-weight", type=float, default=30.0)
    ap.add_argument("--energy-warmup-steps", type=int, default=3500)
    ap.add_argument("--energy-ramp-steps", type=int, default=2000)
    ap.add_argument("--decode-weight", type=float, default=1.2)
    ap.add_argument("--rank-weight", type=float, default=2.0)
    ap.add_argument("--prediction-mask-temp", type=float, default=0.18)
    ap.add_argument("--prediction-mask-topk", type=int, default=3)
    ap.add_argument("--dampen-bounded-weight", type=float, default=8.0)
    ap.add_argument("--dampen-bounded-warmup-steps", type=int, default=3500)
    ap.add_argument("--dampen-bounded-ramp-steps", type=int, default=2000)
    ap.add_argument("--dampen-lower-floor-frac", type=float, default=0.18)
    ap.add_argument("--dampen-upper-floor-frac", type=float, default=0.58)
    ap.add_argument("--dampen-bound-margin", type=float, default=0.01)
    ap.add_argument("--dampen-bound-sq-weight", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    print(f"device={device}", flush=True)
    print(f"out={args.out}", flush=True)
    print(f"seed={args.seed}", flush=True)

    loc_penalty = locality_penalty(args)
    base = SimpleNet(use_circuit=True).to(device)
    phase1_history = phase1_local(base, args, loc_penalty)
    base_state = copy.deepcopy(base.state_dict())

    summary = {
        "args": vars(args),
        "device": device,
        "checkpoints": {
            "sharpen": os.path.join(args.out, "ckpt_tuning_shape_repair_sharpen.pt"),
            "dampen": os.path.join(args.out, "ckpt_tuning_shape_repair_dampen.pt"),
        },
        "history": {
            "phase1": phase1_history,
        },
    }
    summary["history"]["sharpen"] = train_one("sharpen", base_state, summary["checkpoints"]["sharpen"], args, loc_penalty)
    summary["history"]["dampen"] = train_one("dampen", base_state, summary["checkpoints"]["dampen"], args, loc_penalty)

    summary_path = os.path.join(args.out, "train_tuning_shape_repair_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"SUMMARY {summary_path}", flush=True)
    print("TUNING_SHAPE_REPAIR_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
