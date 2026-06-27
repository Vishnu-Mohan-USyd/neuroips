#!/usr/bin/env python3
"""Train repaired Phase-A SOM/VIP checkpoints with paired energy contrast.

This is an additive repair driver. It leaves the original reproduction scripts
and committed checkpoints untouched, and writes repaired checkpoints only to the
explicit --out directory.
"""

import argparse
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simple_net import N, STEP_DEG, SimpleNet, chan, device, forward_seq, make_sequences, phase1  # noqa: E402


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
    if regime == "sharpen":
        decode_violation = F.relu(ce_e - ce_u + args.decode_margin)
        decode_loss = ce_e.mean() + args.rank_weight * (decode_violation.mean() + decode_violation.max())
    elif regime == "dampen":
        decode_violation = F.relu(ce_u - ce_e + args.decode_margin)
        decode_loss = ce_u.mean() + args.rank_weight * (decode_violation.mean() + decode_violation.max())
    else:
        raise ValueError(f"unknown regime {regime!r}")

    pair_pred_ce = F.cross_entropy(
        preds_e[:, :-1, :].reshape(-1, N),
        chan(theta_e[:, 1:]).reshape(-1),
    )
    pred_mask = F.softmax(preds_e[:, K - 1, :].detach() / args.prediction_mask_temp, dim=1)
    if args.prediction_mask_topk > 0:
        keep = torch.zeros_like(pred_mask)
        idx = pred_mask.topk(args.prediction_mask_topk, dim=1).indices
        keep.scatter_(1, idx, 1.0)
        pred_mask = pred_mask * keep
    pred_mask = pred_mask / pred_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    predicted_activity = (pred_mask * r_e).sum(dim=1)
    predicted_activity_sq = (pred_mask * r_e.square()).sum(dim=1)
    predicted_floor = (pred_mask * r_floor).sum(dim=1)
    floor_excess = F.relu(r_e - r_floor + args.predicted_floor_margin)
    predicted_floor_excess = (pred_mask * floor_excess).sum(dim=1)
    predicted_floor_excess_sq = (pred_mask * floor_excess.square()).sum(dim=1)
    predicted_content = (
        predicted_activity.mean()
        + predicted_activity.max()
        + args.predicted_content_sq_weight * (predicted_activity_sq.mean() + predicted_activity_sq.max())
    )
    predicted_floor_excess_loss = (
        predicted_floor_excess.mean()
        + predicted_floor_excess.max()
        + args.predicted_floor_excess_sq_weight * (
            predicted_floor_excess_sq.mean() + predicted_floor_excess_sq.max()
        )
    )
    pred_mask_target_mass = pred_mask.gather(1, e_idx.view(-1, 1)).squeeze(1).detach()
    return {
        "energy": energy_loss,
        "decode": decode_loss,
        "pair_pred_ce": pair_pred_ce,
        "predicted_content": predicted_content,
        "predicted_floor_excess": predicted_floor_excess_loss,
        "mean_predicted_activity": predicted_activity.detach().mean(),
        "max_predicted_activity": predicted_activity.detach().max(),
        "mean_predicted_activity_sq": predicted_activity_sq.detach().mean(),
        "max_predicted_activity_sq": predicted_activity_sq.detach().max(),
        "mean_predicted_floor": predicted_floor.detach().mean(),
        "max_predicted_floor": predicted_floor.detach().max(),
        "mean_predicted_floor_excess": predicted_floor_excess.detach().mean(),
        "max_predicted_floor_excess": predicted_floor_excess.detach().max(),
        "mean_predicted_floor_excess_sq": predicted_floor_excess_sq.detach().mean(),
        "max_predicted_floor_excess_sq": predicted_floor_excess_sq.detach().max(),
        "mean_prediction_mask_target_mass": pred_mask_target_mass.mean(),
        "min_prediction_mask_target_mass": pred_mask_target_mass.min(),
        "max_prediction_mask_target_mass": pred_mask_target_mass.max(),
        "max_delta_mean_abs": (E_e - E_u).detach().max(),
        "mean_delta_mean_abs": (E_e - E_u).detach().mean(),
        "frac_delta_lt_0": ((E_e - E_u).detach() < 0).float().mean(),
        "mean_ce_expected": ce_e.detach().mean(),
        "mean_ce_unexpected": ce_u.detach().mean(),
    }


def sequence_losses(net, args):
    theta = make_sequences(args.batch, 12, mode="momentum", p_stay=0.9)
    preds, r_all = forward_seq(net, theta, 1.0)
    pred_ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), chan(theta[:, 1:]).reshape(-1))
    current_ce = F.cross_entropy(net.decoder(r_all.reshape(-1, N)), chan(theta).reshape(-1))
    activity = r_all.abs().mean()
    return pred_ce, current_ce, activity


@torch.no_grad()
def held_acc(net, seed, batch=8192):
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


def set_feedforward_trainable(net, trainable):
    for p in net.W_ff.parameters():
        p.requires_grad_(trainable)


def ramped_weight(final_weight, step, warmup_steps, ramp_steps):
    """Delayed metabolic pressure: predictor/readout competence first, then ramp."""
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    return final_weight * min(1.0, (step - warmup_steps) / ramp_steps)


def train_one(regime, base_state, out_path, args):
    net = SimpleNet(use_circuit=True).to(device)
    net.load_state_dict(copy.deepcopy(base_state))
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(param_groups(net, args))
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    history = []

    print(f"\n=== ENERGY REPAIR {regime.upper()} steps={args.steps} device={device} ===", flush=True)
    for step in range(1, args.steps + 1):
        w_ff_frozen = regime == "dampen" and step > args.freeze_wff_after_step
        set_feedforward_trainable(net, not w_ff_frozen)
        pred_ce, current_ce, activity = sequence_losses(net, args)
        pair = pair_losses(net, regime, theta_e, theta_u, e_idx, u_idx, args)
        energy_weight = ramped_weight(args.energy_weight, step, args.energy_warmup_steps, args.energy_ramp_steps)
        predicted_content_weight = (
            ramped_weight(
                args.predicted_content_weight,
                step,
                args.predicted_content_warmup_steps,
                args.predicted_content_ramp_steps,
            )
            if regime == "dampen"
            else 0.0
        )
        predicted_floor_excess_weight = (
            ramped_weight(
                args.predicted_floor_excess_weight,
                step,
                args.predicted_floor_excess_warmup_steps,
                args.predicted_floor_excess_ramp_steps,
            )
            if regime == "dampen"
            else 0.0
        )
        loss = (
            args.pred_weight * pred_ce
            + args.current_weight * current_ce
            + args.activity_weight * activity
            + args.pair_pred_weight * pair["pair_pred_ce"]
            + energy_weight * pair["energy"]
            + args.decode_weight * pair["decode"]
            + predicted_content_weight * pair["predicted_content"]
            + predicted_floor_excess_weight * pair["predicted_floor_excess"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            held = held_acc(net, args.seed + step, batch=args.held_batch)
            gains = [round(x, 4) for x in F.softplus(net.circ_raw).detach().cpu().tolist()]
            row = {
                "step": step,
                "loss": float(loss.item()),
                "pred_ce": float(pred_ce.item()),
                "current_ce": float(current_ce.item()),
                "activity": float(activity.item()),
                "effective_energy_weight": float(energy_weight),
                "effective_predicted_content_weight": float(predicted_content_weight),
                "effective_predicted_floor_excess_weight": float(predicted_floor_excess_weight),
                "w_ff_frozen": w_ff_frozen,
                "pair_energy": float(pair["energy"].item()),
                "pair_decode": float(pair["decode"].item()),
                "predicted_content": float(pair["predicted_content"].item()),
                "predicted_floor_excess": float(pair["predicted_floor_excess"].item()),
                "weighted_pair_energy": float(energy_weight * pair["energy"].item()),
                "weighted_predicted_content": float(predicted_content_weight * pair["predicted_content"].item()),
                "weighted_predicted_floor_excess": float(
                    predicted_floor_excess_weight * pair["predicted_floor_excess"].item()
                ),
                "mean_predicted_activity": float(pair["mean_predicted_activity"].item()),
                "max_predicted_activity": float(pair["max_predicted_activity"].item()),
                "mean_predicted_activity_sq": float(pair["mean_predicted_activity_sq"].item()),
                "max_predicted_activity_sq": float(pair["max_predicted_activity_sq"].item()),
                "mean_predicted_floor": float(pair["mean_predicted_floor"].item()),
                "max_predicted_floor": float(pair["max_predicted_floor"].item()),
                "mean_predicted_floor_excess": float(pair["mean_predicted_floor_excess"].item()),
                "max_predicted_floor_excess": float(pair["max_predicted_floor_excess"].item()),
                "mean_predicted_floor_excess_sq": float(pair["mean_predicted_floor_excess_sq"].item()),
                "max_predicted_floor_excess_sq": float(pair["max_predicted_floor_excess_sq"].item()),
                "mean_prediction_mask_target_mass": float(pair["mean_prediction_mask_target_mass"].item()),
                "min_prediction_mask_target_mass": float(pair["min_prediction_mask_target_mass"].item()),
                "max_prediction_mask_target_mass": float(pair["max_prediction_mask_target_mass"].item()),
                "max_delta_mean_abs": float(pair["max_delta_mean_abs"].item()),
                "mean_delta_mean_abs": float(pair["mean_delta_mean_abs"].item()),
                "frac_delta_lt_0": float(pair["frac_delta_lt_0"].item()),
                "mean_ce_expected": float(pair["mean_ce_expected"].item()),
                "mean_ce_unexpected": float(pair["mean_ce_unexpected"].item()),
                "held_acc": held,
                "gains": gains,
            }
            history.append(row)
            print(json.dumps({"regime": regime, **row}, sort_keys=True), flush=True)

    torch.save(net.state_dict(), out_path)
    print(f"SAVED {regime} {out_path}", flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description="Train repaired Phase-A sharpen/dampen checkpoints with paired energy contrast.")
    ap.add_argument("--out", required=True, help="external output directory for repaired checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase1-steps", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held-batch", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-repr", type=float, default=3e-4)
    ap.add_argument("--energy-margin", type=float, default=0.03)
    ap.add_argument("--decode-margin", type=float, default=0.05)
    ap.add_argument("--pred-weight", type=float, default=2.0)
    ap.add_argument("--current-weight", type=float, default=0.5)
    ap.add_argument("--activity-weight", type=float, default=0.02)
    ap.add_argument("--pair-pred-weight", type=float, default=0.5)
    ap.add_argument("--energy-weight", type=float, default=35.0)
    ap.add_argument("--energy-warmup-steps", type=int, default=4000)
    ap.add_argument("--energy-ramp-steps", type=int, default=2000)
    ap.add_argument("--predicted-content-weight", type=float, default=1.0)
    ap.add_argument("--predicted-content-sq-weight", type=float, default=0.25)
    ap.add_argument("--predicted-content-warmup-steps", type=int, default=4000)
    ap.add_argument("--predicted-content-ramp-steps", type=int, default=2000)
    ap.add_argument("--prediction-mask-temp", type=float, default=0.2)
    ap.add_argument("--prediction-mask-topk", type=int, default=3)
    ap.add_argument("--predicted-floor-excess-weight", type=float, default=5.0)
    ap.add_argument("--predicted-floor-excess-sq-weight", type=float, default=0.5)
    ap.add_argument("--predicted-floor-margin", type=float, default=0.02)
    ap.add_argument("--predicted-floor-excess-warmup-steps", type=int, default=4000)
    ap.add_argument("--predicted-floor-excess-ramp-steps", type=int, default=2000)
    ap.add_argument("--freeze-wff-after-step", type=int, default=4000)
    ap.add_argument("--decode-weight", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    print(f"device={device}", flush=True)
    print(f"out={args.out}", flush=True)
    print(f"seed={args.seed}", flush=True)

    base = SimpleNet(use_circuit=True).to(device)
    phase1(base, steps=args.phase1_steps)
    base_state = copy.deepcopy(base.state_dict())

    summary = {
        "args": vars(args),
        "device": device,
        "checkpoints": {
            "sharpen": os.path.join(args.out, "ckpt_energy_repair_sharpen.pt"),
            "dampen": os.path.join(args.out, "ckpt_energy_repair_dampen.pt"),
        },
        "history": {},
    }
    summary["history"]["sharpen"] = train_one("sharpen", base_state, summary["checkpoints"]["sharpen"], args)
    summary["history"]["dampen"] = train_one("dampen", base_state, summary["checkpoints"]["dampen"], args)

    summary_path = os.path.join(args.out, "train_energy_repair_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"SUMMARY {summary_path}", flush=True)
    print("ENERGY_REPAIR_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
