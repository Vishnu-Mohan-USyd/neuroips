#!/usr/bin/env python3
"""Train independent SOM/VIP checkpoints with naturalistic energy demands.

This trainer intentionally does not optimize raw tuning shape. It keeps one
shared local/sparse feedforward representation, freezes it, and trains sharpen
and dampen with the same loss families. The regimes differ only by scalar
precision/task and energy/metabolic weights.
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


def locality_penalty(args):
    d = circular_distance_matrix()
    return 1.0 - torch.exp(-0.5 * (d / args.ff_local_sigma_channels).square())


def ff_locality_loss(net, penalty):
    W = net.W_ff.weight
    return (W.square() * penalty).sum() / N + net.W_ff.bias.square().mean()


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


def profile_monitor(r, center_idx):
    profile = aligned_stack(r, center_idx).mean(0)
    return {
        "raw_center": float(profile[0].detach().item()),
        "raw_flank": float(profile[[2, -2]].mean().detach().item()),
        "raw_shoulder": float(profile[[3, 4, -3, -4]].mean().detach().item()),
        "raw_far": float(profile[[10, 11, 12, -10, -11, -12]].mean().detach().item()),
    }


def phase1_local(net, args, loc_penalty):
    opt = torch.optim.Adam(list(net.W_ff.parameters()) + list(net.decoder.parameters()), lr=args.phase1_lr)
    history = []
    print("\n=== PHASE 1 SHARED LOCAL/SPARSE REPRESENTATION ===", flush=True)
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
                "rep_acc": acc,
                "activity": float(activity.item()),
                "activity_sq": float(activity_sq.item()),
                "ff_locality": float(local.item()),
            }
            history.append(row)
            print(json.dumps({"stage": "phase1", **row}, sort_keys=True), flush=True)
    return history


def freeze_shared_representation(net):
    for p in list(net.W_ff.parameters()) + list(net.decoder.parameters()):
        p.requires_grad_(False)


def prediction_mask(pred, args):
    mask = F.softmax(pred.detach() / args.prediction_mask_temp, dim=1)
    if args.prediction_mask_topk > 0:
        keep = torch.zeros_like(mask)
        keep.scatter_(1, mask.topk(args.prediction_mask_topk, dim=1).indices, 1.0)
        mask = mask * keep
    return mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)


def noisy_current_ce(net, r, target, sigma, repeats):
    if sigma <= 0.0 or repeats <= 0:
        return F.cross_entropy(net.decoder(r), target)
    r_rep = r.unsqueeze(0).expand(repeats, *r.shape)
    noisy = r_rep + sigma * torch.randn_like(r_rep)
    logits = net.decoder(noisy.reshape(repeats * r.shape[0], N))
    return F.cross_entropy(logits, target.repeat(repeats))


def sequence_losses(net, args, weights):
    theta = make_sequences(args.batch, 12, mode="momentum", p_stay=0.9)
    preds, r_all = forward_seq(net, theta, 1.0)
    target_next = chan(theta[:, 1:]).reshape(-1)
    target_current = chan(theta).reshape(-1)
    pred_ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), target_next)
    r_flat = r_all.reshape(-1, N)
    current_ce = F.cross_entropy(net.decoder(r_flat), target_current)
    noisy_ce = noisy_current_ce(net, r_flat, target_current, args.noise_sigma, args.noise_repeats)
    activity = r_all.abs().mean()
    activity_sq = r_all.square().mean()
    return {
        "pred_ce": pred_ce,
        "current_ce": current_ce,
        "noisy_current_ce": noisy_ce,
        "activity": activity,
        "activity_sq": activity_sq,
    }


def pair_losses(net, theta_e, theta_u, e_idx, u_idx, args):
    preds_e, r_all_e = forward_seq(net, theta_e, 1.0)
    _, r_all_u = forward_seq(net, theta_u, 1.0)
    r_e = r_all_e[:, K, :]
    r_u = r_all_u[:, K, :]

    E_e = r_e.abs().mean(dim=1)
    E_u = r_u.abs().mean(dim=1)
    dE = E_e - E_u
    energy_violation = F.relu(dE + args.pair_energy_margin)
    paired_energy_contrast = energy_violation.mean() + args.pair_energy_max_weight * energy_violation.max()
    logits_e = net.decoder(r_e)
    logits_u = net.decoder(r_u)
    ce_e = F.cross_entropy(logits_e, e_idx, reduction="none")
    ce_u = F.cross_entropy(logits_u, u_idx, reduction="none")
    prob_e = logits_e.softmax(-1).gather(1, e_idx.view(-1, 1)).squeeze(1)
    prob_u = logits_u.softmax(-1).gather(1, u_idx.view(-1, 1)).squeeze(1)

    pair_pred_ce = F.cross_entropy(
        preds_e[:, :-1, :].reshape(-1, N),
        chan(theta_e[:, 1:]).reshape(-1),
    )
    pair_current_ce = F.cross_entropy(logits_e, e_idx)
    pair_noisy_ce = noisy_current_ce(net, r_e, e_idx, args.noise_sigma, args.noise_repeats)

    mask = prediction_mask(preds_e[:, K - 1, :], args)
    pred_activity = (mask * r_e.abs()).sum(dim=1)
    pred_activity_sq = (mask * r_e.square()).sum(dim=1)
    pred_energy = pred_activity.mean() + args.pred_energy_max_weight * pred_activity.max()
    pred_energy_sq = pred_activity_sq.mean() + args.pred_energy_max_weight * pred_activity_sq.max()

    monitor = profile_monitor(r_e, e_idx)
    return {
        "pair_pred_ce": pair_pred_ce,
        "pair_current_ce": pair_current_ce,
        "pair_noisy_current_ce": pair_noisy_ce,
        "paired_energy_contrast": paired_energy_contrast,
        "paired_energy_violation_mean": energy_violation.detach().mean(),
        "paired_energy_violation_max": energy_violation.detach().max(),
        "expected_activity": E_e.mean(),
        "expected_activity_sq": r_e.square().mean(),
        "prediction_weighted_energy": pred_energy,
        "prediction_weighted_energy_sq": pred_energy_sq,
        "energy_max_delta": dE.detach().max(),
        "energy_mean_delta": dE.detach().mean(),
        "frac_delta_lt_0": (dE.detach() < 0).float().mean(),
        "mean_ce_expected": ce_e.detach().mean(),
        "mean_ce_unexpected": ce_u.detach().mean(),
        "mean_prob_expected": prob_e.detach().mean(),
        "mean_prob_unexpected": prob_u.detach().mean(),
        "mean_pred_activity": pred_activity.detach().mean(),
        "max_pred_activity": pred_activity.detach().max(),
        **monitor,
    }


@torch.no_grad()
def held_acc(net, seed, batch=4096):
    torch.manual_seed(seed)
    theta = make_sequences(batch, 12, mode="momentum", p_stay=0.9)
    preds, _ = forward_seq(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


def ramped_weight(final_weight, step, warmup_steps, ramp_steps):
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    return final_weight * min(1.0, (step - warmup_steps) / ramp_steps)


def regime_weights(regime, args):
    if regime == "sharpen":
        return {
            "pred": args.sharpen_pred_weight,
            "current": args.sharpen_current_weight,
            "noisy_current": args.sharpen_noisy_current_weight,
            "pair_pred": args.sharpen_pair_pred_weight,
            "pair_current": args.sharpen_pair_current_weight,
            "pair_noisy_current": args.sharpen_pair_noisy_current_weight,
            "activity": args.sharpen_activity_weight,
            "activity_sq": args.sharpen_activity_sq_weight,
            "pred_energy": args.sharpen_pred_energy_weight,
            "pred_energy_sq": args.sharpen_pred_energy_sq_weight,
            "pair_energy": args.sharpen_pair_energy_weight,
        }
    if regime == "dampen":
        return {
            "pred": args.dampen_pred_weight,
            "current": args.dampen_current_weight,
            "noisy_current": args.dampen_noisy_current_weight,
            "pair_pred": args.dampen_pair_pred_weight,
            "pair_current": args.dampen_pair_current_weight,
            "pair_noisy_current": args.dampen_pair_noisy_current_weight,
            "activity": args.dampen_activity_weight,
            "activity_sq": args.dampen_activity_sq_weight,
            "pred_energy": args.dampen_pred_energy_weight,
            "pred_energy_sq": args.dampen_pred_energy_sq_weight,
            "pair_energy": args.dampen_pair_energy_weight,
        }
    raise ValueError(f"unknown regime {regime!r}")


def train_one(regime, base_state, out_path, args, loc_penalty):
    net = SimpleNet(use_circuit=True).to(device)
    net.load_state_dict(copy.deepcopy(base_state))
    freeze_shared_representation(net)
    opt = torch.optim.Adam(list(net.gru.parameters()) + list(net.W_fb.parameters()) + [net.circ_raw], lr=args.lr)
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    weights = regime_weights(regime, args)
    history = []

    print(f"\n=== NATURAL TUNING REPAIR {regime.upper()} steps={args.steps} device={device} ===", flush=True)
    print(json.dumps({"regime": regime, "weights": weights, "shared_W_ff_decoder_frozen": True}, sort_keys=True), flush=True)
    for step in range(1, args.steps + 1):
        seq = sequence_losses(net, args, weights)
        pair = pair_losses(net, theta_e, theta_u, e_idx, u_idx, args)
        energy_scale = ramped_weight(1.0, step, args.energy_warmup_steps, args.energy_ramp_steps)
        pred_energy_weight = energy_scale * weights["pred_energy"]
        pred_energy_sq_weight = energy_scale * weights["pred_energy_sq"]
        pair_energy_weight = energy_scale * weights["pair_energy"]
        activity_weight = energy_scale * weights["activity"]
        activity_sq_weight = energy_scale * weights["activity_sq"]

        loss = (
            weights["pred"] * seq["pred_ce"]
            + weights["current"] * seq["current_ce"]
            + weights["noisy_current"] * seq["noisy_current_ce"]
            + weights["pair_pred"] * pair["pair_pred_ce"]
            + weights["pair_current"] * pair["pair_current_ce"]
            + weights["pair_noisy_current"] * pair["pair_noisy_current_ce"]
            + activity_weight * (seq["activity"] + pair["expected_activity"])
            + activity_sq_weight * (seq["activity_sq"] + pair["expected_activity_sq"])
            + pair_energy_weight * pair["paired_energy_contrast"]
            + pred_energy_weight * pair["prediction_weighted_energy"]
            + pred_energy_sq_weight * pair["prediction_weighted_energy_sq"]
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            held = held_acc(net, args.seed + step, args.held_batch)
            local = ff_locality_loss(net, loc_penalty)
            gains = [round(x, 4) for x in F.softplus(net.circ_raw).detach().cpu().tolist()]
            row = {
                "step": step,
                "loss": float(loss.item()),
                "pred_ce": float(seq["pred_ce"].item()),
                "current_ce": float(seq["current_ce"].item()),
                "noisy_current_ce": float(seq["noisy_current_ce"].item()),
                "activity": float(seq["activity"].item()),
                "activity_sq": float(seq["activity_sq"].item()),
                "pair_pred_ce": float(pair["pair_pred_ce"].item()),
                "pair_current_ce": float(pair["pair_current_ce"].item()),
                "pair_noisy_current_ce": float(pair["pair_noisy_current_ce"].item()),
                "paired_energy_contrast": float(pair["paired_energy_contrast"].item()),
                "paired_energy_violation_mean": float(pair["paired_energy_violation_mean"].item()),
                "paired_energy_violation_max": float(pair["paired_energy_violation_max"].item()),
                "prediction_weighted_energy": float(pair["prediction_weighted_energy"].item()),
                "prediction_weighted_energy_sq": float(pair["prediction_weighted_energy_sq"].item()),
                "effective_activity_weight": float(activity_weight),
                "effective_activity_sq_weight": float(activity_sq_weight),
                "effective_pair_energy_weight": float(pair_energy_weight),
                "effective_pred_energy_weight": float(pred_energy_weight),
                "effective_pred_energy_sq_weight": float(pred_energy_sq_weight),
                "ff_locality": float(local.item()),
                "max_delta_mean_abs": float(pair["energy_max_delta"].item()),
                "mean_delta_mean_abs": float(pair["energy_mean_delta"].item()),
                "frac_delta_lt_0": float(pair["frac_delta_lt_0"].item()),
                "mean_ce_expected": float(pair["mean_ce_expected"].item()),
                "mean_ce_unexpected": float(pair["mean_ce_unexpected"].item()),
                "mean_prob_expected": float(pair["mean_prob_expected"].item()),
                "mean_prob_unexpected": float(pair["mean_prob_unexpected"].item()),
                "mean_pred_activity": float(pair["mean_pred_activity"].item()),
                "max_pred_activity": float(pair["max_pred_activity"].item()),
                "held_acc": held,
                "gains": gains,
                "raw_center": pair["raw_center"],
                "raw_flank": pair["raw_flank"],
                "raw_shoulder": pair["raw_shoulder"],
                "raw_far": pair["raw_far"],
            }
            history.append(row)
            print(json.dumps({"regime": regime, **row}, sort_keys=True), flush=True)

    torch.save(net.state_dict(), out_path)
    print(f"SAVED {regime} {out_path}", flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description="Train naturalistic-only independent tuning repair checkpoints.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase1-steps", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held-batch", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--phase1-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise-sigma", type=float, default=0.6)
    ap.add_argument("--noise-repeats", type=int, default=2)
    ap.add_argument("--ff-local-sigma-channels", type=float, default=2.5)
    ap.add_argument("--ff-locality-weight", type=float, default=4.0)
    ap.add_argument("--phase1-activity-weight", type=float, default=0.08)
    ap.add_argument("--phase1-activity-sq-weight", type=float, default=0.02)
    ap.add_argument("--prediction-mask-temp", type=float, default=0.18)
    ap.add_argument("--prediction-mask-topk", type=int, default=3)
    ap.add_argument("--pred-energy-max-weight", type=float, default=0.5)
    ap.add_argument("--pair-energy-margin", type=float, default=0.03)
    ap.add_argument("--pair-energy-max-weight", type=float, default=1.0)
    ap.add_argument("--energy-warmup-steps", type=int, default=3500)
    ap.add_argument("--energy-ramp-steps", type=int, default=2000)

    ap.add_argument("--sharpen-pred-weight", type=float, default=3.0)
    ap.add_argument("--sharpen-current-weight", type=float, default=1.0)
    ap.add_argument("--sharpen-noisy-current-weight", type=float, default=0.6)
    ap.add_argument("--sharpen-pair-pred-weight", type=float, default=0.8)
    ap.add_argument("--sharpen-pair-current-weight", type=float, default=1.0)
    ap.add_argument("--sharpen-pair-noisy-current-weight", type=float, default=0.6)
    # Stage3 keeps the loss families identical across regimes. These scalar
    # defaults rebalance naturalistic task/energy demands only.
    ap.add_argument("--sharpen-activity-weight", type=float, default=0.015)
    ap.add_argument("--sharpen-activity-sq-weight", type=float, default=0.002)
    ap.add_argument("--sharpen-pair-energy-weight", type=float, default=20.0)
    ap.add_argument("--sharpen-pred-energy-weight", type=float, default=0.04)
    ap.add_argument("--sharpen-pred-energy-sq-weight", type=float, default=0.002)

    ap.add_argument("--dampen-pred-weight", type=float, default=2.8)
    ap.add_argument("--dampen-current-weight", type=float, default=0.24)
    ap.add_argument("--dampen-noisy-current-weight", type=float, default=0.08)
    ap.add_argument("--dampen-pair-pred-weight", type=float, default=0.8)
    ap.add_argument("--dampen-pair-current-weight", type=float, default=0.20)
    ap.add_argument("--dampen-pair-noisy-current-weight", type=float, default=0.08)
    ap.add_argument("--dampen-activity-weight", type=float, default=0.04)
    ap.add_argument("--dampen-activity-sq-weight", type=float, default=0.005)
    ap.add_argument("--dampen-pair-energy-weight", type=float, default=24.0)
    ap.add_argument("--dampen-pred-energy-weight", type=float, default=0.35)
    ap.add_argument("--dampen-pred-energy-sq-weight", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    print(f"device={device}", flush=True)
    print(f"out={args.out}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print("training_contract=same_architecture_same_loss_families_scalar_weight_differences_only", flush=True)
    print("shape_losses_used=false", flush=True)

    loc_penalty = locality_penalty(args)
    base = SimpleNet(use_circuit=True).to(device)
    phase1_history = phase1_local(base, args, loc_penalty)
    base_state = copy.deepcopy(base.state_dict())

    summary = {
        "args": vars(args),
        "device": device,
        "contract": {
            "architecture": "SimpleNet(use_circuit=True)",
            "shared_feedforward_representation": "same phase1 W_ff and decoder copied into both regimes and frozen",
            "loss_families": [
                "next_step_prediction_ce",
                "clean_current_decoder_ce",
                "noisy_current_decoder_ce",
                "paired_expected_vs_orthogonal_energy_contrast",
                "global_mean_and_squared_l23_activity",
                "prediction_weighted_metabolic_activity",
                "feedforward_locality_sparsity_phase1",
            ],
            "regime_differences": "scalar weights only",
            "shape_losses_used": False,
        },
        "checkpoints": {
            "sharpen": os.path.join(args.out, "ckpt_tuning_shape_natural_sharpen.pt"),
            "dampen": os.path.join(args.out, "ckpt_tuning_shape_natural_dampen.pt"),
        },
        "history": {
            "phase1": phase1_history,
        },
    }
    summary["history"]["sharpen"] = train_one("sharpen", base_state, summary["checkpoints"]["sharpen"], args, loc_penalty)
    summary["history"]["dampen"] = train_one("dampen", base_state, summary["checkpoints"]["dampen"], args, loc_penalty)

    summary_path = os.path.join(args.out, "train_tuning_shape_natural_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"SUMMARY {summary_path}", flush=True)
    print("TUNING_SHAPE_NATURAL_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
