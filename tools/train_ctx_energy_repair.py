#!/usr/bin/env python3
"""Train one context-switching SOM/VIP net with Stage4 energy/shape criteria.

The saved checkpoint uses a Phase-B-style wrapper, but the validation contract
is explicit: the built-in ``net.decoder`` is the primary current-orientation
decoder for both clean and noisy metrics. No external readout is used for pass
criteria.
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


def ctx_batch(value, batch):
    return torch.full((batch, 1), float(value), device=device)


def ramped_weight(final_weight, step, warmup_steps, ramp_steps):
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    return final_weight * min(1.0, (step - warmup_steps) / ramp_steps)


def hard_violation_loss(violation, topk):
    if topk > 0 and topk < violation.numel():
        hard = violation.topk(topk).values
    else:
        hard = violation
    return hard.mean() + hard.max()


def set_feedforward_trainable(net, trainable):
    for p in net.W_ff.parameters():
        p.requires_grad_(trainable)


def load_state_dict(path):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "net" in obj:
        return obj["net"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def prediction_mask(pred, args):
    mask = F.softmax(pred.detach() / args.prediction_mask_temp, dim=1)
    if args.prediction_mask_topk > 0:
        keep = torch.zeros_like(mask)
        keep.scatter_(1, mask.topk(args.prediction_mask_topk, dim=1).indices, 1.0)
        mask = mask * keep
    return mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)


def noisy_ce(net, r, target, sigma):
    return F.cross_entropy(net.decoder(r + sigma * torch.randn_like(r)), target, reduction="none")


def pair_context(net, cv, theta_e, theta_u, e_idx, u_idx, args):
    cb = ctx_batch(cv, theta_e.shape[0])
    preds_e, r_all_e = forward_seq(net, theta_e, 1.0, ctx=cb)
    _, r_all_u = forward_seq(net, theta_u, 1.0, ctx=cb)
    _, r_all_floor = forward_seq(net, theta_e, 0.0, ctx=cb)
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
    nce_e = noisy_ce(net, r_e, e_idx, args.noise_sigma)
    nce_u = noisy_ce(net, r_u, u_idx, args.noise_sigma)

    if cv > 0:
        decode_violation = F.relu(ce_e - ce_u + args.decode_margin)
        noisy_violation = F.relu(nce_e - nce_u + args.noisy_decode_margin)
        decode_loss = ce_e.mean() + nce_e.mean() + args.rank_weight * (
            decode_violation.mean() + decode_violation.max() + noisy_violation.mean() + noisy_violation.max()
        )
    else:
        decode_violation = F.relu(ce_u - ce_e + args.decode_margin)
        noisy_violation = F.relu(nce_u - nce_e + args.noisy_decode_margin)
        decode_loss = ce_u.mean() + nce_u.mean() + args.rank_weight * (
            decode_violation.mean() + decode_violation.max() + noisy_violation.mean() + noisy_violation.max()
        )

    pair_pred_ce = F.cross_entropy(
        preds_e[:, :-1, :].reshape(-1, N),
        chan(theta_e[:, 1:]).reshape(-1),
    )

    mask = prediction_mask(preds_e[:, K - 1, :], args)
    predicted_activity = (mask * r_e).sum(dim=1)
    predicted_activity_sq = (mask * r_e.square()).sum(dim=1)
    predicted_floor = (mask * r_floor).sum(dim=1)
    floor_excess = F.relu(r_e - r_floor + args.predicted_floor_margin)
    predicted_floor_excess = (mask * floor_excess).sum(dim=1)
    predicted_floor_excess_sq = (mask * floor_excess.square()).sum(dim=1)
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
    target_mass = mask.gather(1, e_idx.view(-1, 1)).squeeze(1)

    return {
        "r_e": r_e,
        "r_u": r_u,
        "E_e": E_e,
        "E_u": E_u,
        "energy_violation": energy_violation,
        "energy": energy_loss,
        "decode": decode_loss,
        "pair_pred_ce": pair_pred_ce,
        "predicted_content": predicted_content,
        "predicted_floor_excess": predicted_floor_excess_loss,
        "ce_e": ce_e,
        "ce_u": ce_u,
        "nce_e": nce_e,
        "nce_u": nce_u,
        "max_delta_mean_abs": (E_e - E_u).detach().max(),
        "mean_delta_mean_abs": (E_e - E_u).detach().mean(),
        "frac_delta_lt_0": ((E_e - E_u).detach() < 0).float().mean(),
        "mean_predicted_activity": predicted_activity.detach().mean(),
        "max_predicted_activity": predicted_activity.detach().max(),
        "mean_predicted_floor": predicted_floor.detach().mean(),
        "mean_predicted_floor_excess": predicted_floor_excess.detach().mean(),
        "max_predicted_floor_excess": predicted_floor_excess.detach().max(),
        "mean_prediction_mask_target_mass": target_mass.detach().mean(),
        "min_prediction_mask_target_mass": target_mass.detach().min(),
    }


def sequence_losses(net, args):
    theta = make_sequences(args.batch, 12, mode="momentum", p_stay=0.9)
    losses = []
    current = []
    activity = []
    for cv in (1.0, -1.0):
        preds, r_all = forward_seq(net, theta, 1.0, ctx=ctx_batch(cv, args.batch))
        losses.append(F.cross_entropy(preds[:, :-1, :].reshape(-1, N), chan(theta[:, 1:]).reshape(-1)))
        current.append(F.cross_entropy(net.decoder(r_all.reshape(-1, N)), chan(theta).reshape(-1)))
        activity.append(r_all.abs().mean())
    return torch.stack(losses).mean(), torch.stack(current).mean(), torch.stack(activity).mean()


def prediction_guard_loss(net, args):
    theta = make_sequences(args.pred_guard_batch, 12, mode="momentum", p_stay=0.9)
    losses = []
    for cv in (1.0, -1.0):
        preds, _ = forward_seq(net, theta, 1.0, ctx=ctx_batch(cv, args.pred_guard_batch))
        losses.append(F.cross_entropy(preds[:, :-1, :].reshape(-1, N), chan(theta[:, 1:]).reshape(-1)))
    return torch.stack(losses).mean()


def cycle_active(repair_elapsed, args):
    if repair_elapsed <= 0:
        return False
    if args.repair_cycle_steps <= 0:
        return True
    return ((repair_elapsed - 1) % args.repair_cycle_steps) >= args.pred_guard_cycle_steps


@torch.no_grad()
def held_acc(net, cv, seed, batch):
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        theta = make_sequences(batch, 12, mode="momentum", p_stay=0.9)
        preds, _ = forward_seq(net, theta, 1.0, ctx=ctx_batch(cv, batch))
        ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
        return float(ok.float().mean().item() * 100.0)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def train(args):
    torch.manual_seed(args.seed)
    net = SimpleNet(use_circuit=True, context=True).to(device)
    if args.init_ckpt:
        net.load_state_dict(load_state_dict(args.init_ckpt))
        print(f"INIT_CKPT {args.init_ckpt}", flush=True)
    else:
        phase1(net, steps=args.phase1_steps)
        base_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(base_state)
    for p in net.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(
        [
            {"params": list(net.gru.parameters()) + list(net.W_fb.parameters()) + [net.circ_raw, net.g_ctx_raw], "lr": args.lr},
            {"params": list(net.W_ff.parameters()) + list(net.decoder.parameters()), "lr": args.lr_repr},
        ]
    )
    theta_e, theta_u, e_idx, u_idx = build_pairs()
    history = []
    gate_step = 0 if args.init_ckpt else None
    gate_reason = "init_ckpt" if args.init_ckpt else None
    last_gate_held_attend = None
    last_gate_held_save = None
    gate_guard_ce = None

    print(f"device={device}", flush=True)
    print(f"out={args.out}", flush=True)
    print(f"decoder_contract=built_in_net.decoder", flush=True)
    print(f"seed={args.seed}", flush=True)

    for step in range(1, args.steps + 1):
        if gate_step is None and (
            step == 1 or step % args.held_gate_check_every == 0 or step >= args.repair_warmup_steps
        ):
            last_gate_held_attend = held_acc(net, 1.0, args.seed + 100000 + step, args.held_batch)
            last_gate_held_save = held_acc(net, -1.0, args.seed + 200000 + step, args.held_batch)
            held_ready = (
                step >= args.repair_min_warmup_steps
                and min(last_gate_held_attend, last_gate_held_save) >= args.repair_gate_held_min
            )
            target_ready = step >= args.repair_warmup_steps
            if held_ready or target_ready:
                gate_step = step
                gate_reason = "held_accuracy" if held_ready else "warmup_target"

        repair_elapsed = 0 if gate_step is None else max(0, step - gate_step)
        repair_step_active = gate_step is not None and cycle_active(repair_elapsed, args)
        hard_cleanup_window = (
            repair_step_active
            and repair_elapsed > args.attend_hard_after_gate_steps
            and args.allow_wff_hard_cleanup
        )
        absolute_freeze = args.freeze_wff_after_step >= 0 and step > args.freeze_wff_after_step
        gate_freeze = gate_step is not None and step > gate_step + args.freeze_wff_after_gate_steps
        w_ff_frozen = (absolute_freeze or gate_freeze) and not hard_cleanup_window
        set_feedforward_trainable(net, not w_ff_frozen)

        pred_ce, current_ce, seq_activity = sequence_losses(net, args)
        if gate_step is None:
            guard_pred_ce = pred_ce.detach()
            pred_guard_weight = 0.0
            pred_guard_excess_weight = 0.0
            pred_guard_excess = pred_ce.detach() * 0.0
        else:
            guard_pred_ce = prediction_guard_loss(net, args)
            if gate_guard_ce is None:
                gate_guard_ce = float(guard_pred_ce.detach().item())
            pred_guard_weight = ramped_weight(
                args.pred_guard_weight, repair_elapsed, 0, args.pred_guard_ramp_steps
            )
            pred_guard_limit = gate_guard_ce * args.pred_guard_ce_multiplier + args.pred_guard_ce_margin
            pred_guard_excess = F.relu(guard_pred_ce - pred_guard_limit)
            pred_guard_excess_weight = args.pred_guard_excess_weight

        attend = pair_context(net, 1.0, theta_e, theta_u, e_idx, u_idx, args)
        save = pair_context(net, -1.0, theta_e, theta_u, e_idx, u_idx, args)

        repair_ramp = 0.0 if gate_step is None else ramped_weight(1.0, repair_elapsed, 0, args.repair_ramp_steps)
        repair_scale = 1.0 if repair_step_active else 0.0
        pair_pred_weight = args.pair_pred_weight * repair_ramp
        decode_weight = args.decode_weight * repair_ramp
        context_weight = args.context_weight * repair_ramp
        energy_weight = 0.0 if gate_step is None else ramped_weight(
            args.energy_weight, repair_elapsed, args.energy_warmup_steps, args.energy_ramp_steps
        )
        pred_content_weight = ramped_weight(
            args.predicted_content_weight,
            repair_elapsed,
            args.predicted_content_warmup_steps,
            args.predicted_content_ramp_steps,
        ) if gate_step is not None else 0.0
        floor_excess_weight = ramped_weight(
            args.predicted_floor_excess_weight,
            repair_elapsed,
            args.predicted_floor_excess_warmup_steps,
            args.predicted_floor_excess_ramp_steps,
        ) if gate_step is not None else 0.0
        attend_hard_weight = ramped_weight(
            args.attend_hard_energy_weight,
            repair_elapsed,
            args.attend_hard_after_gate_steps,
            args.attend_hard_ramp_steps,
        ) if gate_step is not None else 0.0
        save_hard_weight = ramped_weight(
            args.save_hard_energy_weight,
            repair_elapsed,
            args.save_hard_after_gate_steps,
            args.save_hard_ramp_steps,
        ) if gate_step is not None else 0.0
        pair_pred_weight *= repair_scale
        decode_weight *= repair_scale
        context_weight *= repair_scale
        energy_weight *= repair_scale
        pred_content_weight *= repair_scale
        floor_excess_weight *= repair_scale
        attend_hard_weight *= repair_scale
        save_hard_weight *= repair_scale
        if args.save_hard_requires_wff_frozen and not w_ff_frozen:
            save_hard_weight = 0.0
        attend_hard_energy = hard_violation_loss(attend["energy_violation"], args.attend_hard_topk)
        save_hard_energy = hard_violation_loss(save["energy_violation"], args.save_hard_topk)
        save_energy_adv = F.relu(save["E_e"] - attend["E_e"] + args.context_energy_margin)
        clean_ctx_decode = F.relu(attend["ce_e"] - save["ce_e"] + args.context_decode_margin)
        noisy_ctx_decode = F.relu(attend["nce_e"] - save["nce_e"] + args.context_decode_margin)
        context_loss = (
            save_energy_adv.mean()
            + save_energy_adv.max()
            + args.context_decode_weight * (clean_ctx_decode.mean() + clean_ctx_decode.max())
            + args.context_noisy_decode_weight * (noisy_ctx_decode.mean() + noisy_ctx_decode.max())
        )

        loss = (
            args.pred_weight * pred_ce
            + args.current_weight * current_ce
            + args.activity_weight * seq_activity
            + pred_guard_weight * guard_pred_ce
            + pred_guard_excess_weight * pred_guard_excess.square()
            + pair_pred_weight * (attend["pair_pred_ce"] + save["pair_pred_ce"]) * 0.5
            + energy_weight * (attend["energy"] + save["energy"])
            + attend_hard_weight * attend_hard_energy
            + save_hard_weight * save_hard_energy
            + decode_weight * (attend["decode"] + save["decode"])
            + context_weight * context_loss
            + pred_content_weight * save["predicted_content"]
            + floor_excess_weight * save["predicted_floor_excess"]
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "pred_ce": float(pred_ce.item()),
                "current_ce": float(current_ce.item()),
                "guard_pred_ce": float(guard_pred_ce.item()),
                "gate_guard_ce": gate_guard_ce,
                "pred_guard_weight": float(pred_guard_weight),
                "pred_guard_excess": float(pred_guard_excess.item()),
                "pred_guard_excess_weight": float(pred_guard_excess_weight),
                "seq_activity": float(seq_activity.item()),
                "energy_weight": float(energy_weight),
                "pair_pred_weight": float(pair_pred_weight),
                "decode_weight": float(decode_weight),
                "context_weight": float(context_weight),
                "predicted_content_weight": float(pred_content_weight),
                "predicted_floor_excess_weight": float(floor_excess_weight),
                "attend_hard_energy_weight": float(attend_hard_weight),
                "attend_hard_energy": float(attend_hard_energy.item()),
                "save_hard_energy_weight": float(save_hard_weight),
                "save_hard_energy": float(save_hard_energy.item()),
                "context_loss": float(context_loss.item()),
                "repair_enabled": gate_step is not None,
                "repair_elapsed": int(repair_elapsed),
                "repair_gate_step": gate_step,
                "repair_gate_reason": gate_reason,
                "repair_ramp": float(repair_ramp),
                "repair_step_active": repair_step_active,
                "hard_cleanup_window": hard_cleanup_window,
                "gate_held_attend": last_gate_held_attend,
                "gate_held_save": last_gate_held_save,
                "w_ff_frozen": w_ff_frozen,
                "held_attend": held_acc(net, 1.0, args.seed + step, args.held_batch),
                "held_save": held_acc(net, -1.0, args.seed + step, args.held_batch),
                "g_ctx": float(F.softplus(net.g_ctx_raw).item()),
                "gains": [round(x, 4) for x in F.softplus(net.circ_raw).detach().cpu().tolist()],
            }
            for prefix, data in (("attend", attend), ("save", save)):
                row.update({
                    f"{prefix}_frac_delta_lt_0": float(data["frac_delta_lt_0"].item()),
                    f"{prefix}_max_delta_mean_abs": float(data["max_delta_mean_abs"].item()),
                    f"{prefix}_mean_delta_mean_abs": float(data["mean_delta_mean_abs"].item()),
                    f"{prefix}_ce_e": float(data["ce_e"].mean().item()),
                    f"{prefix}_ce_u": float(data["ce_u"].mean().item()),
                    f"{prefix}_nce_e": float(data["nce_e"].mean().item()),
                    f"{prefix}_nce_u": float(data["nce_u"].mean().item()),
                    f"{prefix}_mean_predicted_activity": float(data["mean_predicted_activity"].item()),
                    f"{prefix}_max_predicted_activity": float(data["max_predicted_activity"].item()),
                    f"{prefix}_mean_predicted_floor": float(data["mean_predicted_floor"].item()),
                    f"{prefix}_mean_predicted_floor_excess": float(data["mean_predicted_floor_excess"].item()),
                    f"{prefix}_max_predicted_floor_excess": float(data["max_predicted_floor_excess"].item()),
                    f"{prefix}_mask_target_mass": float(data["mean_prediction_mask_target_mass"].item()),
                    f"{prefix}_min_mask_target_mass": float(data["min_prediction_mask_target_mass"].item()),
                })
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    cfg = {
        "seed": args.seed,
        "decoder_contract": "built_in_net.decoder",
        "ctx_coding": "+1 attend / -1 save",
        "noise_sigma": args.noise_sigma,
        "args": vars(args),
    }
    ckpt = {"net": net.state_dict(), "read": None, "cfg": cfg}
    ckpt_path = os.path.join(args.out, "ckpt_ctx_energy_repair.pt")
    torch.save(ckpt, ckpt_path)
    summary_path = os.path.join(args.out, "train_ctx_energy_repair_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "history": history, "checkpoint": ckpt_path}, f, indent=2, sort_keys=True)
    print(f"SAVED {ckpt_path}", flush=True)
    print(f"SUMMARY {summary_path}", flush=True)
    print("CTX_ENERGY_REPAIR_TRAIN_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train one Phase-B context net for Stage4 energy/decoding/shape criteria.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-ckpt", help="optional wrapper/raw checkpoint to continue from instead of running phase1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--phase1-steps", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held-batch", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-repr", type=float, default=3e-4)
    ap.add_argument("--pred-weight", type=float, default=6.0)
    ap.add_argument("--current-weight", type=float, default=0.75)
    ap.add_argument("--activity-weight", type=float, default=0.02)
    ap.add_argument("--pair-pred-weight", type=float, default=0.3)
    ap.add_argument("--energy-weight", type=float, default=18.0)
    ap.add_argument("--energy-margin", type=float, default=0.03)
    ap.add_argument("--energy-warmup-steps", type=int, default=0)
    ap.add_argument("--energy-ramp-steps", type=int, default=4000)
    ap.add_argument("--decode-weight", type=float, default=0.5)
    ap.add_argument("--decode-margin", type=float, default=0.05)
    ap.add_argument("--noisy-decode-margin", type=float, default=0.05)
    ap.add_argument("--noise-sigma", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=2.0)
    ap.add_argument("--context-weight", type=float, default=0.5)
    ap.add_argument("--context-energy-margin", type=float, default=0.05)
    ap.add_argument("--context-decode-margin", type=float, default=0.05)
    ap.add_argument("--context-decode-weight", type=float, default=1.0)
    ap.add_argument("--context-noisy-decode-weight", type=float, default=1.0)
    ap.add_argument("--prediction-mask-temp", type=float, default=0.2)
    ap.add_argument("--prediction-mask-topk", type=int, default=3)
    ap.add_argument("--predicted-content-weight", type=float, default=0.5)
    ap.add_argument("--predicted-content-sq-weight", type=float, default=0.25)
    ap.add_argument("--predicted-content-warmup-steps", type=int, default=0)
    ap.add_argument("--predicted-content-ramp-steps", type=int, default=4000)
    ap.add_argument("--predicted-floor-excess-weight", type=float, default=2.0)
    ap.add_argument("--predicted-floor-excess-sq-weight", type=float, default=0.5)
    ap.add_argument("--predicted-floor-margin", type=float, default=0.02)
    ap.add_argument("--predicted-floor-excess-warmup-steps", type=int, default=0)
    ap.add_argument("--predicted-floor-excess-ramp-steps", type=int, default=4000)
    ap.add_argument("--freeze-wff-after-step", type=int, default=4000)
    ap.add_argument("--freeze-wff-after-gate-steps", type=int, default=0)
    ap.add_argument("--repair-min-warmup-steps", type=int, default=4000)
    ap.add_argument("--repair-warmup-steps", type=int, default=6000)
    ap.add_argument("--repair-gate-held-min", type=float, default=75.0)
    ap.add_argument("--held-gate-check-every", type=int, default=500)
    ap.add_argument("--repair-ramp-steps", type=int, default=4000)
    ap.add_argument("--repair-cycle-steps", type=int, default=4)
    ap.add_argument("--pred-guard-cycle-steps", type=int, default=2)
    ap.add_argument("--pred-guard-batch", type=int, default=512)
    ap.add_argument("--pred-guard-weight", type=float, default=10.0)
    ap.add_argument("--pred-guard-ramp-steps", type=int, default=1000)
    ap.add_argument("--pred-guard-ce-multiplier", type=float, default=1.25)
    ap.add_argument("--pred-guard-ce-margin", type=float, default=0.05)
    ap.add_argument("--pred-guard-excess-weight", type=float, default=30.0)
    ap.add_argument("--attend-hard-energy-weight", type=float, default=20.0)
    ap.add_argument("--attend-hard-after-gate-steps", type=int, default=3500)
    ap.add_argument("--attend-hard-ramp-steps", type=int, default=1000)
    ap.add_argument("--attend-hard-topk", type=int, default=36)
    ap.add_argument("--save-hard-energy-weight", type=float, default=20.0)
    ap.add_argument("--save-hard-after-gate-steps", type=int, default=3500)
    ap.add_argument("--save-hard-ramp-steps", type=int, default=1000)
    ap.add_argument("--save-hard-topk", type=int, default=36)
    ap.add_argument("--save-hard-requires-wff-frozen", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-wff-hard-cleanup", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train(args)


if __name__ == "__main__":
    raise SystemExit(main())
