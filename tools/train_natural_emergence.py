#!/usr/bin/env python3
"""Train independent SOM/VIP checkpoints from natural sequence demands only.

This additive trainer deliberately keeps the validation assay out of training:
it samples only ``make_sequences(..., mode="momentum")`` batches, never builds
expected/unexpected validation pairs, and never optimizes response-shape terms.
Sharpen and dampen use the same architecture and loss families; only scalar
task, precision, and metabolic weights differ.
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

from simple_net import N, SimpleNet, chan, device, forward_seq, l4_code, make_sequences  # noqa: E402


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


def phase1_equalization_losses(net, args):
    """General feedforward-rate equalization over all orientation classes."""
    theta_grid = torch.arange(N, device=device).float() * (180.0 / N)
    r_grid = net.l23(l4_code(theta_grid), torch.zeros(N, N, device=device))
    energy_by_orientation = r_grid.abs().mean(dim=1)
    channel_rate = r_grid.mean(dim=0)
    orient_energy_equal = energy_by_orientation.var(unbiased=False)
    channel_rate_equal = channel_rate.var(unbiased=False)
    if args.phase1_channel_homeostatic_target >= 0.0:
        channel_homeostatic = (channel_rate - args.phase1_channel_homeostatic_target).square().mean()
    else:
        channel_homeostatic = channel_rate_equal
    return {
        "orient_energy_equal": orient_energy_equal,
        "channel_rate_equal": channel_rate_equal,
        "channel_homeostatic": channel_homeostatic,
        "energy_by_orientation_mean": energy_by_orientation.detach().mean(),
        "energy_by_orientation_std": energy_by_orientation.detach().std(unbiased=False),
        "energy_by_orientation_min": energy_by_orientation.detach().min(),
        "energy_by_orientation_max": energy_by_orientation.detach().max(),
        "channel_rate_mean": channel_rate.detach().mean(),
        "channel_rate_std": channel_rate.detach().std(unbiased=False),
        "channel_rate_min": channel_rate.detach().min(),
        "channel_rate_max": channel_rate.detach().max(),
    }


def phase1_local(net, args, loc_penalty):
    opt = torch.optim.Adam(list(net.W_ff.parameters()) + list(net.decoder.parameters()), lr=args.phase1_lr)
    history = []
    print("\n=== PHASE 1 SHARED LOCAL/SPARSE REPRESENTATION ===", flush=True)
    for step in range(1, args.phase1_steps + 1):
        theta = torch.randint(0, N, (args.batch,), device=device).float() * (180.0 / N)
        r = net.l23(l4_code(theta), torch.zeros(args.batch, N, device=device))
        target = chan(theta)
        logits = net.decoder(r)
        current_ce = F.cross_entropy(logits, target)
        activity = r.abs().mean()
        activity_sq = r.square().mean()
        local = ff_locality_loss(net, loc_penalty)
        equal = phase1_equalization_losses(net, args)
        loss = (
            current_ce
            + args.phase1_activity_weight * activity
            + args.phase1_activity_sq_weight * activity_sq
            + args.ff_locality_weight * local
            + args.phase1_orientation_energy_equal_weight * equal["orient_energy_equal"]
            + args.phase1_channel_rate_equal_weight * equal["channel_rate_equal"]
            + args.phase1_channel_homeostatic_weight * equal["channel_homeostatic"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.phase1_steps:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "current_ce": float(current_ce.item()),
                "rep_acc_percent": float((logits.argmax(-1) == target).float().mean().item() * 100.0),
                "activity": float(activity.item()),
                "activity_sq": float(activity_sq.item()),
                "ff_locality": float(local.item()),
                "orient_energy_equal": float(equal["orient_energy_equal"].item()),
                "channel_rate_equal": float(equal["channel_rate_equal"].item()),
                "channel_homeostatic": float(equal["channel_homeostatic"].item()),
                "energy_by_orientation_mean": float(equal["energy_by_orientation_mean"].item()),
                "energy_by_orientation_std": float(equal["energy_by_orientation_std"].item()),
                "energy_by_orientation_min": float(equal["energy_by_orientation_min"].item()),
                "energy_by_orientation_max": float(equal["energy_by_orientation_max"].item()),
                "channel_rate_mean": float(equal["channel_rate_mean"].item()),
                "channel_rate_std": float(equal["channel_rate_std"].item()),
                "channel_rate_min": float(equal["channel_rate_min"].item()),
                "channel_rate_max": float(equal["channel_rate_max"].item()),
            }
            history.append(row)
            print(json.dumps({"stage": "phase1", **row}, sort_keys=True), flush=True)
    return history


def freeze_shared_representation(net):
    for p in list(net.W_ff.parameters()) + list(net.decoder.parameters()):
        p.requires_grad_(False)


def apply_model_config(net, config):
    for key, value in config.items():
        setattr(net, key, value)


def candidate_phase1_config(args):
    config = {
        "phase1_orientation_energy_equal_weight": args.phase1_orientation_energy_equal_weight,
        "phase1_channel_rate_equal_weight": args.phase1_channel_rate_equal_weight,
        "phase1_channel_homeostatic_weight": args.phase1_channel_homeostatic_weight,
        "phase1_channel_homeostatic_target": args.phase1_channel_homeostatic_target,
    }
    if args.candidate in ("floor_fb_gated_inhib_mild", "floor_fb_gated_inhib_error_mild"):
        config.update({
            "phase1_orientation_energy_equal_weight": 4.0,
            "phase1_channel_rate_equal_weight": 1.0,
            "phase1_channel_homeostatic_weight": 0.0,
            "phase1_channel_homeostatic_target": -1.0,
        })
    elif args.candidate == "floor_fb_gated_homeo_energy":
        config.update({
            "phase1_orientation_energy_equal_weight": 6.0,
            "phase1_channel_rate_equal_weight": 1.5,
            "phase1_channel_homeostatic_weight": 0.0,
            "phase1_channel_homeostatic_target": -1.0,
        })
    elif args.candidate in ("som_pool_error_floor", "som_pool_fb_gated_floor"):
        config.update({
            "phase1_orientation_energy_equal_weight": 10.0,
            "phase1_channel_rate_equal_weight": 3.0,
            "phase1_channel_homeostatic_weight": 0.0,
            "phase1_channel_homeostatic_target": -1.0,
        })
    elif args.candidate == "som_pool_homeo_energy":
        config.update({
            "phase1_orientation_energy_equal_weight": 14.0,
            "phase1_channel_rate_equal_weight": 4.0,
            "phase1_channel_homeostatic_weight": 0.0,
            "phase1_channel_homeostatic_target": -1.0,
        })
    elif args.candidate in ("topo_somvip_floor", "topo_somvip_fb_gated_floor"):
        config.update({
            "phase1_orientation_energy_equal_weight": 18.0,
            "phase1_channel_rate_equal_weight": 5.0,
            "phase1_channel_homeostatic_weight": 0.0,
            "phase1_channel_homeostatic_target": -1.0,
        })
    elif args.candidate == "topo_somvip_homeo_energy":
        config.update({
            "phase1_orientation_energy_equal_weight": 24.0,
            "phase1_channel_rate_equal_weight": 6.0,
            "phase1_channel_homeostatic_weight": 0.0,
            "phase1_channel_homeostatic_target": -1.0,
        })
    return config


def candidate_model_config(args):
    config = {
        "l23_competition_strength": args.l23_competition_strength,
        "l23_competition_sigma_channels": args.l23_competition_sigma_channels,
        "l23_competition_radius": args.l23_competition_radius,
        "l23_competition_global_strength": args.l23_competition_global_strength,
        "l23_local_inhibition_strength": args.l23_local_inhibition_strength,
        "l23_local_inhibition_sigma_channels": args.l23_local_inhibition_sigma_channels,
        "l23_local_inhibition_radius": args.l23_local_inhibition_radius,
        "l23_local_inhibition_center_weight": args.l23_local_inhibition_center_weight,
        "l23_feedback_gated_inhibition_strength": args.l23_feedback_gated_inhibition_strength,
        "l23_feedback_gated_inhibition_sigma_channels": args.l23_feedback_gated_inhibition_sigma_channels,
        "l23_feedback_gated_inhibition_radius": args.l23_feedback_gated_inhibition_radius,
        "l23_feedback_gated_inhibition_center_weight": args.l23_feedback_gated_inhibition_center_weight,
        "som_feedback_pool_strength": args.som_feedback_pool_strength,
        "som_feedback_pool_sigma_channels": args.som_feedback_pool_sigma_channels,
        "som_feedback_pool_radius": args.som_feedback_pool_radius,
        "som_feedback_pool_center_weight": args.som_feedback_pool_center_weight,
        "somvip_topographic_som_strength": args.somvip_topographic_som_strength,
        "somvip_topographic_som_sigma_channels": args.somvip_topographic_som_sigma_channels,
        "somvip_topographic_som_radius": args.somvip_topographic_som_radius,
        "somvip_topographic_som_center_weight": args.somvip_topographic_som_center_weight,
        "somvip_topographic_vip_strength": args.somvip_topographic_vip_strength,
        "somvip_topographic_vip_sigma_channels": args.somvip_topographic_vip_sigma_channels,
        "somvip_topographic_vip_radius": args.somvip_topographic_vip_radius,
        "somvip_topographic_vip_center_weight": args.somvip_topographic_vip_center_weight,
        "l23_prediction_error_strength": args.l23_prediction_error_strength,
    }
    if args.candidate == "norm_mild":
        config.update({
            "l23_competition_strength": 0.7,
            "l23_competition_sigma_channels": 2.5,
            "l23_competition_radius": 5,
            "l23_competition_global_strength": 0.10,
        })
    elif args.candidate == "norm_energy":
        config.update({
            "l23_competition_strength": 1.2,
            "l23_competition_sigma_channels": 2.5,
            "l23_competition_radius": 5,
            "l23_competition_global_strength": 0.20,
        })
    elif args.candidate == "norm_precision":
        config.update({
            "l23_competition_strength": 0.9,
            "l23_competition_sigma_channels": 2.0,
            "l23_competition_radius": 4,
            "l23_competition_global_strength": 0.12,
        })
    elif args.candidate == "error_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "error_energy":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.55,
        })
    elif args.candidate == "error_precision":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.45,
        })
    elif args.candidate == "corrupt_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.0,
        })
    elif args.candidate == "corrupt_error_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "corrupt_error_precision":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.45,
        })
    elif args.candidate == "ramp_corrupt_low":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.0,
        })
    elif args.candidate == "ramp_corrupt_error_low":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "ramp_corrupt_error_balanced":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "local_inhib_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.18,
            "l23_local_inhibition_sigma_channels": 1.5,
            "l23_local_inhibition_radius": 3,
            "l23_local_inhibition_center_weight": 0.0,
            "l23_prediction_error_strength": 0.0,
        })
    elif args.candidate == "local_inhib_error_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.18,
            "l23_local_inhibition_sigma_channels": 1.5,
            "l23_local_inhibition_radius": 3,
            "l23_local_inhibition_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "local_inhib_error_balanced":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.26,
            "l23_local_inhibition_sigma_channels": 1.75,
            "l23_local_inhibition_radius": 4,
            "l23_local_inhibition_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "fb_gated_inhib_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.24,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "l23_prediction_error_strength": 0.0,
        })
    elif args.candidate == "fb_gated_inhib_error_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.24,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "fb_gated_inhib_error_balanced":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.36,
            "l23_feedback_gated_inhibition_sigma_channels": 1.75,
            "l23_feedback_gated_inhibition_radius": 4,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "l23_prediction_error_strength": 0.25,
        })
    elif args.candidate == "floor_fb_gated_inhib_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.24,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "l23_prediction_error_strength": 0.0,
        })
    elif args.candidate == "floor_fb_gated_inhib_error_mild":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.24,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "floor_fb_gated_homeo_energy":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.24,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "som_pool_error_floor":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.0,
            "som_feedback_pool_strength": 0.35,
            "som_feedback_pool_sigma_channels": 1.5,
            "som_feedback_pool_radius": 3,
            "som_feedback_pool_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "som_pool_fb_gated_floor":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.16,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "som_feedback_pool_strength": 0.35,
            "som_feedback_pool_sigma_channels": 1.5,
            "som_feedback_pool_radius": 3,
            "som_feedback_pool_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "som_pool_homeo_energy":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.12,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "som_feedback_pool_strength": 0.50,
            "som_feedback_pool_sigma_channels": 1.75,
            "som_feedback_pool_radius": 4,
            "som_feedback_pool_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "topo_somvip_floor":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.0,
            "som_feedback_pool_strength": 0.0,
            "somvip_topographic_som_strength": 0.45,
            "somvip_topographic_som_sigma_channels": 0.65,
            "somvip_topographic_som_radius": 2,
            "somvip_topographic_som_center_weight": 1.0,
            "somvip_topographic_vip_strength": 0.30,
            "somvip_topographic_vip_sigma_channels": 2.25,
            "somvip_topographic_vip_radius": 5,
            "somvip_topographic_vip_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "topo_somvip_fb_gated_floor":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.10,
            "l23_feedback_gated_inhibition_sigma_channels": 1.5,
            "l23_feedback_gated_inhibition_radius": 3,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "som_feedback_pool_strength": 0.0,
            "somvip_topographic_som_strength": 0.45,
            "somvip_topographic_som_sigma_channels": 0.65,
            "somvip_topographic_som_radius": 2,
            "somvip_topographic_som_center_weight": 1.0,
            "somvip_topographic_vip_strength": 0.35,
            "somvip_topographic_vip_sigma_channels": 2.5,
            "somvip_topographic_vip_radius": 5,
            "somvip_topographic_vip_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    elif args.candidate == "topo_somvip_homeo_energy":
        config.update({
            "l23_competition_strength": 0.0,
            "l23_competition_global_strength": 0.0,
            "l23_local_inhibition_strength": 0.0,
            "l23_feedback_gated_inhibition_strength": 0.08,
            "l23_feedback_gated_inhibition_sigma_channels": 1.75,
            "l23_feedback_gated_inhibition_radius": 4,
            "l23_feedback_gated_inhibition_center_weight": 1.0,
            "som_feedback_pool_strength": 0.0,
            "somvip_topographic_som_strength": 0.60,
            "somvip_topographic_som_sigma_channels": 0.85,
            "somvip_topographic_som_radius": 2,
            "somvip_topographic_som_center_weight": 1.0,
            "somvip_topographic_vip_strength": 0.45,
            "somvip_topographic_vip_sigma_channels": 2.75,
            "somvip_topographic_vip_radius": 6,
            "somvip_topographic_vip_center_weight": 0.0,
            "l23_prediction_error_strength": 0.35,
        })
    return config


def candidate_corruption_config(args):
    config = {
        "l4_dropout_start": args.l4_dropout_start,
        "l4_dropout_end": args.l4_dropout_end,
        "l4_noise_start": args.l4_noise_start,
        "l4_noise_end": args.l4_noise_end,
        "l4_corruption_warmup_steps": args.l4_corruption_warmup_steps,
        "l4_corruption_ramp_steps": args.l4_corruption_ramp_steps,
        "l4_corruption_anneal_steps": args.l4_corruption_anneal_steps,
    }
    if args.candidate == "corrupt_mild":
        config.update({"l4_dropout_end": 0.25, "l4_noise_end": 0.02})
    elif args.candidate == "corrupt_error_mild":
        config.update({"l4_dropout_end": 0.25, "l4_noise_end": 0.02})
    elif args.candidate == "corrupt_error_precision":
        config.update({"l4_dropout_end": 0.35, "l4_noise_end": 0.03})
    elif args.candidate == "ramp_corrupt_low":
        config.update({
            "l4_dropout_start": 0.0,
            "l4_dropout_end": 0.08,
            "l4_noise_start": 0.0,
            "l4_noise_end": 0.008,
            "l4_corruption_warmup_steps": 2500,
            "l4_corruption_ramp_steps": 2500,
            "l4_corruption_anneal_steps": 1500,
        })
    elif args.candidate == "ramp_corrupt_error_low":
        config.update({
            "l4_dropout_start": 0.0,
            "l4_dropout_end": 0.08,
            "l4_noise_start": 0.0,
            "l4_noise_end": 0.008,
            "l4_corruption_warmup_steps": 2500,
            "l4_corruption_ramp_steps": 2500,
            "l4_corruption_anneal_steps": 1500,
        })
    elif args.candidate == "ramp_corrupt_error_balanced":
        config.update({
            "l4_dropout_start": 0.0,
            "l4_dropout_end": 0.12,
            "l4_noise_start": 0.0,
            "l4_noise_end": 0.012,
            "l4_corruption_warmup_steps": 2500,
            "l4_corruption_ramp_steps": 2500,
            "l4_corruption_anneal_steps": 1000,
        })
    return config


def candidate_gain_config(args):
    config = {
        "gain_regularization_weight": args.gain_regularization_weight,
        "gain_regularization_warmup_steps": args.gain_regularization_warmup_steps,
        "gain_regularization_ramp_steps": args.gain_regularization_ramp_steps,
        "g_ps_floor": args.g_ps_floor,
    }
    if args.candidate in ("topo_somvip_floor", "topo_somvip_fb_gated_floor"):
        config.update({
            "gain_regularization_weight": 0.08,
            "gain_regularization_warmup_steps": 3000,
            "gain_regularization_ramp_steps": 2500,
            "g_ps_floor": 0.50,
        })
    elif args.candidate == "topo_somvip_homeo_energy":
        config.update({
            "gain_regularization_weight": 0.12,
            "gain_regularization_warmup_steps": 3000,
            "gain_regularization_ramp_steps": 2500,
            "g_ps_floor": 0.52,
        })
    return config


def corruption_schedule(args, step):
    if step is None:
        return args.l4_dropout_end, args.l4_noise_end
    if step <= args.l4_corruption_warmup_steps:
        frac = 0.0
    elif args.l4_corruption_ramp_steps <= 0:
        frac = 1.0
    else:
        frac = min(1.0, (step - args.l4_corruption_warmup_steps) / args.l4_corruption_ramp_steps)
    if args.l4_corruption_anneal_steps > 0:
        anneal_start = max(args.l4_corruption_warmup_steps, args.steps - args.l4_corruption_anneal_steps)
        if step > anneal_start:
            frac *= max(0.0, (args.steps - step) / args.l4_corruption_anneal_steps)
    dropout = args.l4_dropout_start + frac * (args.l4_dropout_end - args.l4_dropout_start)
    noise = args.l4_noise_start + frac * (args.l4_noise_end - args.l4_noise_start)
    return dropout, noise


def corrupt_l4(x, dropout_prob, noise_sigma):
    """Apply general stochastic L4 sensory corruption during training only."""
    if dropout_prob > 0.0:
        keep = (torch.rand_like(x) >= dropout_prob).float()
        x = x * keep
    if noise_sigma > 0.0:
        x = (x + noise_sigma * torch.randn_like(x)).clamp_min(0.0)
    return x


def forward_seq_train(net, theta, args, fb_scale=1.0, step=None):
    """Training unroll with optional stochastic L4 corruption on every timestep."""
    use_signed = getattr(net, "signed_fb", False)
    B = theta.shape[0]
    h = torch.zeros(B, net.hidden, device=device)
    pred_down = torch.zeros(B, N, device=device)
    preds, r_seq = [], []
    dropout_prob, noise_sigma = corruption_schedule(args, step)
    for t in range(theta.shape[1]):
        l4 = corrupt_l4(l4_code(theta[:, t]), dropout_prob, noise_sigma)
        r = net.l23(l4, fb_scale * pred_down)
        r_seq.append(r)
        h = net.gru(r, h)
        pred = net.W_fb(h)
        preds.append(pred)
        pred_down = pred if use_signed else F.relu(pred)
    return torch.stack(preds, 1), torch.stack(r_seq, 1), dropout_prob, noise_sigma


def prediction_confidence_mask(pred_logits, args):
    mask = F.softmax(pred_logits.detach() / args.prediction_mask_temp, dim=-1)
    if args.prediction_mask_topk > 0:
        keep = torch.zeros_like(mask)
        keep.scatter_(-1, mask.topk(args.prediction_mask_topk, dim=-1).indices, 1.0)
        mask = mask * keep
    return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def noisy_current_ce(net, r_flat, target_flat, sigma, repeats):
    if sigma <= 0.0 or repeats <= 0:
        return F.cross_entropy(net.decoder(r_flat), target_flat)
    r_rep = r_flat.unsqueeze(0).expand(repeats, *r_flat.shape)
    noisy = r_rep + sigma * torch.randn_like(r_rep)
    logits = net.decoder(noisy.reshape(repeats * r_flat.shape[0], N))
    return F.cross_entropy(logits, target_flat.repeat(repeats))


def natural_sequence_losses(net, args, step):
    theta = make_sequences(args.batch, args.sequence_len, mode="momentum", p_stay=args.p_stay)
    preds, r_all, dropout_prob, noise_sigma = forward_seq_train(net, theta, args, 1.0, step)
    target_current = chan(theta)
    target_next = target_current[:, 1:].reshape(-1)
    r_flat = r_all.reshape(-1, N)
    current_flat = target_current.reshape(-1)

    pred_ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), target_next)
    current_ce = F.cross_entropy(net.decoder(r_flat), current_flat)
    noisy_ce = noisy_current_ce(net, r_flat, current_flat, args.noise_sigma, args.noise_repeats)
    activity = r_all.abs().mean()
    activity_sq = r_all.square().mean()

    # Prediction made at t-1 is fed back at t. Apply the same metabolic term
    # broadly across every such natural timestep and sample.
    mask = prediction_confidence_mask(preds[:, :-1, :], args)
    r_with_prior = r_all[:, 1:, :]
    pred_conf_activity = (mask * r_with_prior.abs()).sum(dim=-1)
    pred_conf_activity_sq = (mask * r_with_prior.square()).sum(dim=-1)
    pred_conf_energy = (
        pred_conf_activity.mean()
        + args.pred_conf_energy_max_weight * pred_conf_activity.max()
    )
    pred_conf_energy_sq = (
        pred_conf_activity_sq.mean()
        + args.pred_conf_energy_max_weight * pred_conf_activity_sq.max()
    )

    mean_per_channel = r_all.mean(dim=(0, 1))
    homeostatic = (mean_per_channel - args.homeostatic_target).square().mean()

    return {
        "pred_ce": pred_ce,
        "current_ce": current_ce,
        "noisy_current_ce": noisy_ce,
        "activity": activity,
        "activity_sq": activity_sq,
        "pred_conf_energy": pred_conf_energy,
        "pred_conf_energy_sq": pred_conf_energy_sq,
        "homeostatic": homeostatic,
        "mean_channel_rate": mean_per_channel.detach().mean(),
        "std_channel_rate": mean_per_channel.detach().std(unbiased=False),
        "min_channel_rate": mean_per_channel.detach().min(),
        "max_channel_rate": mean_per_channel.detach().max(),
        "mean_pred_conf_activity": pred_conf_activity.detach().mean(),
        "max_pred_conf_activity": pred_conf_activity.detach().max(),
        "mean_pred_conf_activity_sq": pred_conf_activity_sq.detach().mean(),
        "max_pred_conf_activity_sq": pred_conf_activity_sq.detach().max(),
        "l4_dropout_prob": dropout_prob,
        "l4_noise_sigma": noise_sigma,
    }


def gain_regularization_loss(net, args):
    if not hasattr(net, "circ_raw") or args.gain_regularization_weight <= 0.0:
        return torch.zeros((), device=device)
    gains = F.softplus(net.circ_raw)
    g_ps = gains[4]
    return F.relu(args.g_ps_floor - g_ps).square()


@torch.no_grad()
def held_acc(net, seed, args):
    torch.manual_seed(seed)
    theta = make_sequences(args.held_batch, args.sequence_len, mode="momentum", p_stay=args.p_stay)
    preds, _ = forward_seq(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


def ramped_weight(final_weight, step, warmup_steps, ramp_steps):
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    return final_weight * min(1.0, (step - warmup_steps) / ramp_steps)


def candidate_weights(args):
    base = {
        "sharpen": {
            "pred": args.sharpen_pred_weight,
            "current": args.sharpen_current_weight,
            "noisy_current": args.sharpen_noisy_current_weight,
            "activity": args.sharpen_activity_weight,
            "activity_sq": args.sharpen_activity_sq_weight,
            "pred_conf_energy": args.sharpen_pred_conf_energy_weight,
            "pred_conf_energy_sq": args.sharpen_pred_conf_energy_sq_weight,
            "homeostatic": args.sharpen_homeostatic_weight,
        },
        "dampen": {
            "pred": args.dampen_pred_weight,
            "current": args.dampen_current_weight,
            "noisy_current": args.dampen_noisy_current_weight,
            "activity": args.dampen_activity_weight,
            "activity_sq": args.dampen_activity_sq_weight,
            "pred_conf_energy": args.dampen_pred_conf_energy_weight,
            "pred_conf_energy_sq": args.dampen_pred_conf_energy_sq_weight,
            "homeostatic": args.dampen_homeostatic_weight,
        },
    }
    if args.candidate == "base":
        return base
    if args.candidate == "energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.35
        out["sharpen"]["pred_conf_energy"] *= 1.5
        out["sharpen"]["pred_conf_energy_sq"] *= 1.75
        out["dampen"]["activity"] *= 1.35
        out["dampen"]["activity_sq"] *= 1.25
        out["dampen"]["pred_conf_energy"] *= 1.35
        out["dampen"]["pred_conf_energy_sq"] *= 1.25
        return out
    if args.candidate == "precision":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.25
        out["sharpen"]["noisy_current"] *= 1.25
        out["dampen"]["current"] *= 1.35
        out["dampen"]["noisy_current"] *= 1.5
        out["dampen"]["pred_conf_energy"] *= 0.75
        out["dampen"]["pred_conf_energy_sq"] *= 0.65
        return out
    if args.candidate == "homeo":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "homeo_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.9
        return out
    if args.candidate == "homeo_precision":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.25
        out["sharpen"]["noisy_current"] *= 1.25
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.45
        out["dampen"]["current"] *= 1.5
        out["dampen"]["noisy_current"] *= 1.75
        out["dampen"]["pred_conf_energy"] *= 1.5
        out["dampen"]["pred_conf_energy_sq"] *= 1.25
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "norm_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "norm_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.9
        return out
    if args.candidate == "norm_precision":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.25
        out["sharpen"]["noisy_current"] *= 1.25
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.45
        out["dampen"]["current"] *= 1.5
        out["dampen"]["noisy_current"] *= 1.75
        out["dampen"]["pred_conf_energy"] *= 1.5
        out["dampen"]["pred_conf_energy_sq"] *= 1.25
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "error_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "error_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.9
        return out
    if args.candidate == "error_precision":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.25
        out["sharpen"]["noisy_current"] *= 1.25
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.45
        out["dampen"]["current"] *= 1.5
        out["dampen"]["noisy_current"] *= 1.75
        out["dampen"]["pred_conf_energy"] *= 1.5
        out["dampen"]["pred_conf_energy_sq"] *= 1.25
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "corrupt_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "corrupt_error_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "corrupt_error_precision":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.25
        out["sharpen"]["noisy_current"] *= 1.25
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.45
        out["dampen"]["current"] *= 1.5
        out["dampen"]["noisy_current"] *= 1.75
        out["dampen"]["pred_conf_energy"] *= 1.5
        out["dampen"]["pred_conf_energy_sq"] *= 1.25
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "ramp_corrupt_low":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "ramp_corrupt_error_low":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "ramp_corrupt_error_balanced":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.15
        out["sharpen"]["noisy_current"] *= 1.15
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.40
        out["dampen"]["current"] *= 1.35
        out["dampen"]["noisy_current"] *= 1.50
        out["dampen"]["pred_conf_energy"] *= 1.35
        out["dampen"]["pred_conf_energy_sq"] *= 1.15
        out["dampen"]["homeostatic"] = 0.50
        return out
    if args.candidate == "local_inhib_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "local_inhib_error_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "local_inhib_error_balanced":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.15
        out["sharpen"]["noisy_current"] *= 1.15
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.40
        out["dampen"]["current"] *= 1.35
        out["dampen"]["noisy_current"] *= 1.50
        out["dampen"]["pred_conf_energy"] *= 1.35
        out["dampen"]["pred_conf_energy_sq"] *= 1.15
        out["dampen"]["homeostatic"] = 0.50
        return out
    if args.candidate == "fb_gated_inhib_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "fb_gated_inhib_error_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "fb_gated_inhib_error_balanced":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] *= 1.15
        out["sharpen"]["noisy_current"] *= 1.15
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.40
        out["dampen"]["current"] *= 1.35
        out["dampen"]["noisy_current"] *= 1.50
        out["dampen"]["pred_conf_energy"] *= 1.35
        out["dampen"]["pred_conf_energy_sq"] *= 1.15
        out["dampen"]["homeostatic"] = 0.50
        return out
    if args.candidate == "floor_fb_gated_inhib_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "floor_fb_gated_inhib_error_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "floor_fb_gated_homeo_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.9
        return out
    if args.candidate in ("som_pool_error_floor", "som_pool_fb_gated_floor"):
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "som_pool_homeo_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.9
        return out
    if args.candidate in ("topo_somvip_floor", "topo_somvip_fb_gated_floor"):
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if args.candidate == "topo_somvip_homeo_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.9
        return out
    raise ValueError(f"unknown candidate {args.candidate!r}")


def train_one(regime, weights, base_state, out_path, args, model_config):
    net = SimpleNet(use_circuit=True).to(device)
    apply_model_config(net, model_config)
    net.load_state_dict(copy.deepcopy(base_state))
    freeze_shared_representation(net)
    opt = torch.optim.Adam(list(net.gru.parameters()) + list(net.W_fb.parameters()) + [net.circ_raw], lr=args.lr)
    history = []

    print(f"\n=== NATURAL EMERGENCE {args.candidate}:{regime.upper()} steps={args.steps} device={device} ===", flush=True)
    print(json.dumps({"regime": regime, "weights": weights, "model_config": model_config}, sort_keys=True), flush=True)
    for step in range(1, args.steps + 1):
        losses = natural_sequence_losses(net, args, step)
        energy_scale = ramped_weight(1.0, step, args.energy_warmup_steps, args.energy_ramp_steps)
        activity_weight = energy_scale * weights["activity"]
        activity_sq_weight = energy_scale * weights["activity_sq"]
        pred_conf_weight = energy_scale * weights["pred_conf_energy"]
        pred_conf_sq_weight = energy_scale * weights["pred_conf_energy_sq"]
        homeostatic_weight = energy_scale * weights["homeostatic"]
        gain_reg_weight = ramped_weight(
            args.gain_regularization_weight,
            step,
            args.gain_regularization_warmup_steps,
            args.gain_regularization_ramp_steps,
        )
        gain_reg = gain_regularization_loss(net, args)

        loss = (
            weights["pred"] * losses["pred_ce"]
            + weights["current"] * losses["current_ce"]
            + weights["noisy_current"] * losses["noisy_current_ce"]
            + activity_weight * losses["activity"]
            + activity_sq_weight * losses["activity_sq"]
            + pred_conf_weight * losses["pred_conf_energy"]
            + pred_conf_sq_weight * losses["pred_conf_energy_sq"]
            + homeostatic_weight * losses["homeostatic"]
            + gain_reg_weight * gain_reg
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            held = held_acc(net, args.seed + step, args)
            gains = [round(x, 5) for x in F.softplus(net.circ_raw).detach().cpu().tolist()]
            row = {
                "candidate": args.candidate,
                "regime": regime,
                "step": step,
                "loss": float(loss.item()),
                "pred_ce": float(losses["pred_ce"].item()),
                "current_ce": float(losses["current_ce"].item()),
                "noisy_current_ce": float(losses["noisy_current_ce"].item()),
                "activity": float(losses["activity"].item()),
                "activity_sq": float(losses["activity_sq"].item()),
                "pred_conf_energy": float(losses["pred_conf_energy"].item()),
                "pred_conf_energy_sq": float(losses["pred_conf_energy_sq"].item()),
                "homeostatic": float(losses["homeostatic"].item()),
                "gain_regularization": float(gain_reg.item()),
                "mean_channel_rate": float(losses["mean_channel_rate"].item()),
                "std_channel_rate": float(losses["std_channel_rate"].item()),
                "min_channel_rate": float(losses["min_channel_rate"].item()),
                "max_channel_rate": float(losses["max_channel_rate"].item()),
                "mean_pred_conf_activity": float(losses["mean_pred_conf_activity"].item()),
                "max_pred_conf_activity": float(losses["max_pred_conf_activity"].item()),
                "mean_pred_conf_activity_sq": float(losses["mean_pred_conf_activity_sq"].item()),
                "max_pred_conf_activity_sq": float(losses["max_pred_conf_activity_sq"].item()),
                "l4_dropout_prob": float(losses["l4_dropout_prob"]),
                "l4_noise_sigma": float(losses["l4_noise_sigma"]),
                "effective_activity_weight": float(activity_weight),
                "effective_activity_sq_weight": float(activity_sq_weight),
                "effective_pred_conf_energy_weight": float(pred_conf_weight),
                "effective_pred_conf_energy_sq_weight": float(pred_conf_sq_weight),
                "effective_homeostatic_weight": float(homeostatic_weight),
                "effective_gain_regularization_weight": float(gain_reg_weight),
                "held_acc_percent": held,
                "gains_g_v_g_s_g_sv_g_e_g_ps": gains,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    torch.save({
        "state_dict": net.state_dict(),
        "simple_net_config": model_config,
        "training_contract": {
            "training_sequences": "natural momentum batches only",
            "expected_unexpected_pairs_used_for_training": False,
            "expected_unexpected_contrast_losses_used": False,
            "shape_losses_used": False,
            "phase1_equalization": {
                "phase1_orientation_energy_equal_weight": args.phase1_orientation_energy_equal_weight,
                "phase1_channel_rate_equal_weight": args.phase1_channel_rate_equal_weight,
                "phase1_channel_homeostatic_weight": args.phase1_channel_homeostatic_weight,
                "phase1_channel_homeostatic_target": args.phase1_channel_homeostatic_target,
            },
            "sensory_corruption_training_only": {
                "l4_dropout_start": args.l4_dropout_start,
                "l4_dropout_end": args.l4_dropout_end,
                "l4_noise_start": args.l4_noise_start,
                "l4_noise_end": args.l4_noise_end,
                "l4_corruption_warmup_steps": args.l4_corruption_warmup_steps,
                "l4_corruption_ramp_steps": args.l4_corruption_ramp_steps,
                "l4_corruption_anneal_steps": args.l4_corruption_anneal_steps,
            },
            "gain_regularization": {
                "gain_regularization_weight": args.gain_regularization_weight,
                "gain_regularization_warmup_steps": args.gain_regularization_warmup_steps,
                "gain_regularization_ramp_steps": args.gain_regularization_ramp_steps,
                "g_ps_floor": args.g_ps_floor,
            },
            "mechanism": "shared_topographic_somvip_feedback_routing_with_optional_feedback_gated_l23_inhibition_prediction_error_feedback_and_training_only_l4_corruption",
        },
    }, out_path)
    print(f"SAVED {args.candidate}:{regime} {out_path}", flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description="Train natural-sequence-only independent emergence checkpoints.")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--candidate",
        choices=(
            "base",
            "energy",
            "precision",
            "homeo",
            "homeo_energy",
            "homeo_precision",
            "norm_mild",
            "norm_energy",
            "norm_precision",
            "error_mild",
            "error_energy",
            "error_precision",
            "corrupt_mild",
            "corrupt_error_mild",
            "corrupt_error_precision",
            "ramp_corrupt_low",
            "ramp_corrupt_error_low",
            "ramp_corrupt_error_balanced",
            "local_inhib_mild",
            "local_inhib_error_mild",
            "local_inhib_error_balanced",
            "fb_gated_inhib_mild",
            "fb_gated_inhib_error_mild",
            "fb_gated_inhib_error_balanced",
            "floor_fb_gated_inhib_mild",
            "floor_fb_gated_inhib_error_mild",
            "floor_fb_gated_homeo_energy",
            "som_pool_error_floor",
            "som_pool_fb_gated_floor",
            "som_pool_homeo_energy",
            "topo_somvip_floor",
            "topo_somvip_fb_gated_floor",
            "topo_somvip_homeo_energy",
        ),
        default="base",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase1-steps", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held-batch", type=int, default=4096)
    ap.add_argument("--sequence-len", type=int, default=12)
    ap.add_argument("--p-stay", type=float, default=0.9)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--phase1-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise-sigma", type=float, default=0.6)
    ap.add_argument("--noise-repeats", type=int, default=2)
    ap.add_argument("--ff-local-sigma-channels", type=float, default=2.5)
    ap.add_argument("--ff-locality-weight", type=float, default=4.0)
    ap.add_argument("--phase1-activity-weight", type=float, default=0.08)
    ap.add_argument("--phase1-activity-sq-weight", type=float, default=0.02)
    ap.add_argument("--phase1-orientation-energy-equal-weight", type=float, default=0.0)
    ap.add_argument("--phase1-channel-rate-equal-weight", type=float, default=0.0)
    ap.add_argument("--phase1-channel-homeostatic-weight", type=float, default=0.0)
    ap.add_argument("--phase1-channel-homeostatic-target", type=float, default=-1.0)
    ap.add_argument("--prediction-mask-temp", type=float, default=0.22)
    ap.add_argument("--prediction-mask-topk", type=int, default=0)
    ap.add_argument("--pred-conf-energy-max-weight", type=float, default=0.25)
    ap.add_argument("--homeostatic-target", type=float, default=0.12)
    ap.add_argument("--l23-competition-strength", type=float, default=0.0)
    ap.add_argument("--l23-competition-sigma-channels", type=float, default=2.0)
    ap.add_argument("--l23-competition-radius", type=int, default=4)
    ap.add_argument("--l23-competition-global-strength", type=float, default=0.0)
    ap.add_argument("--l23-local-inhibition-strength", type=float, default=0.0)
    ap.add_argument("--l23-local-inhibition-sigma-channels", type=float, default=1.5)
    ap.add_argument("--l23-local-inhibition-radius", type=int, default=3)
    ap.add_argument("--l23-local-inhibition-center-weight", type=float, default=0.0)
    ap.add_argument("--l23-feedback-gated-inhibition-strength", type=float, default=0.0)
    ap.add_argument("--l23-feedback-gated-inhibition-sigma-channels", type=float, default=1.5)
    ap.add_argument("--l23-feedback-gated-inhibition-radius", type=int, default=3)
    ap.add_argument("--l23-feedback-gated-inhibition-center-weight", type=float, default=1.0)
    ap.add_argument("--som-feedback-pool-strength", type=float, default=0.0)
    ap.add_argument("--som-feedback-pool-sigma-channels", type=float, default=1.5)
    ap.add_argument("--som-feedback-pool-radius", type=int, default=3)
    ap.add_argument("--som-feedback-pool-center-weight", type=float, default=0.0)
    ap.add_argument("--somvip-topographic-som-strength", type=float, default=0.0)
    ap.add_argument("--somvip-topographic-som-sigma-channels", type=float, default=0.75)
    ap.add_argument("--somvip-topographic-som-radius", type=int, default=2)
    ap.add_argument("--somvip-topographic-som-center-weight", type=float, default=1.0)
    ap.add_argument("--somvip-topographic-vip-strength", type=float, default=0.0)
    ap.add_argument("--somvip-topographic-vip-sigma-channels", type=float, default=2.5)
    ap.add_argument("--somvip-topographic-vip-radius", type=int, default=5)
    ap.add_argument("--somvip-topographic-vip-center-weight", type=float, default=0.0)
    ap.add_argument("--l23-prediction-error-strength", type=float, default=0.0)
    ap.add_argument("--l4-dropout-start", type=float, default=0.0)
    ap.add_argument("--l4-dropout-end", type=float, default=0.0)
    ap.add_argument("--l4-noise-start", type=float, default=0.0)
    ap.add_argument("--l4-noise-end", type=float, default=0.0)
    ap.add_argument("--l4-corruption-warmup-steps", type=int, default=0)
    ap.add_argument("--l4-corruption-ramp-steps", type=int, default=1)
    ap.add_argument("--l4-corruption-anneal-steps", type=int, default=0)
    ap.add_argument("--energy-warmup-steps", type=int, default=3000)
    ap.add_argument("--energy-ramp-steps", type=int, default=2500)
    ap.add_argument("--gain-regularization-weight", type=float, default=0.0)
    ap.add_argument("--gain-regularization-warmup-steps", type=int, default=3000)
    ap.add_argument("--gain-regularization-ramp-steps", type=int, default=2500)
    ap.add_argument("--g-ps-floor", type=float, default=0.0)

    ap.add_argument("--sharpen-pred-weight", type=float, default=3.0)
    ap.add_argument("--sharpen-current-weight", type=float, default=1.0)
    ap.add_argument("--sharpen-noisy-current-weight", type=float, default=0.7)
    ap.add_argument("--sharpen-activity-weight", type=float, default=0.02)
    ap.add_argument("--sharpen-activity-sq-weight", type=float, default=0.003)
    ap.add_argument("--sharpen-pred-conf-energy-weight", type=float, default=0.08)
    ap.add_argument("--sharpen-pred-conf-energy-sq-weight", type=float, default=0.006)
    ap.add_argument("--sharpen-homeostatic-weight", type=float, default=0.0)

    ap.add_argument("--dampen-pred-weight", type=float, default=3.0)
    ap.add_argument("--dampen-current-weight", type=float, default=0.42)
    ap.add_argument("--dampen-noisy-current-weight", type=float, default=0.16)
    ap.add_argument("--dampen-activity-weight", type=float, default=0.075)
    ap.add_argument("--dampen-activity-sq-weight", type=float, default=0.012)
    ap.add_argument("--dampen-pred-conf-energy-weight", type=float, default=0.16)
    ap.add_argument("--dampen-pred-conf-energy-sq-weight", type=float, default=0.008)
    ap.add_argument("--dampen-homeostatic-weight", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    phase1_config = candidate_phase1_config(args)
    args.phase1_orientation_energy_equal_weight = phase1_config["phase1_orientation_energy_equal_weight"]
    args.phase1_channel_rate_equal_weight = phase1_config["phase1_channel_rate_equal_weight"]
    args.phase1_channel_homeostatic_weight = phase1_config["phase1_channel_homeostatic_weight"]
    args.phase1_channel_homeostatic_target = phase1_config["phase1_channel_homeostatic_target"]
    loc_penalty = locality_penalty(args)
    model_config = candidate_model_config(args)
    corruption_config = candidate_corruption_config(args)
    gain_config = candidate_gain_config(args)
    args.l4_dropout_start = corruption_config["l4_dropout_start"]
    args.l4_dropout_end = corruption_config["l4_dropout_end"]
    args.l4_noise_start = corruption_config["l4_noise_start"]
    args.l4_noise_end = corruption_config["l4_noise_end"]
    args.l4_corruption_warmup_steps = corruption_config["l4_corruption_warmup_steps"]
    args.l4_corruption_ramp_steps = corruption_config["l4_corruption_ramp_steps"]
    args.l4_corruption_anneal_steps = corruption_config["l4_corruption_anneal_steps"]
    args.gain_regularization_weight = gain_config["gain_regularization_weight"]
    args.gain_regularization_warmup_steps = gain_config["gain_regularization_warmup_steps"]
    args.gain_regularization_ramp_steps = gain_config["gain_regularization_ramp_steps"]
    args.g_ps_floor = gain_config["g_ps_floor"]

    print(f"device={device}", flush=True)
    print(f"out={args.out}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"candidate={args.candidate}", flush=True)
    print("training_data=make_sequences(mode='momentum') only", flush=True)
    print("expected_unexpected_pairs_used_for_training=false", flush=True)
    print("shape_losses_used=false", flush=True)
    print(f"model_config={json.dumps(model_config, sort_keys=True)}", flush=True)
    print(f"phase1_equalization={json.dumps(phase1_config, sort_keys=True)}", flush=True)
    print(f"training_l4_corruption={json.dumps(corruption_config, sort_keys=True)}", flush=True)
    print(f"gain_regularization={json.dumps(gain_config, sort_keys=True)}", flush=True)

    base = SimpleNet(use_circuit=True).to(device)
    apply_model_config(base, model_config)
    phase1_history = phase1_local(base, args, loc_penalty)
    base_state = copy.deepcopy(base.state_dict())
    weights = candidate_weights(args)

    checkpoints = {
        "sharpen": os.path.join(args.out, f"ckpt_natural_emergence_{args.candidate}_sharpen.pt"),
        "dampen": os.path.join(args.out, f"ckpt_natural_emergence_{args.candidate}_dampen.pt"),
    }
    history = {
        "sharpen": train_one("sharpen", weights["sharpen"], base_state, checkpoints["sharpen"], args, model_config),
        "dampen": train_one("dampen", weights["dampen"], base_state, checkpoints["dampen"], args, model_config),
    }
    summary = {
        "args": vars(args),
        "device": device,
        "contract": {
            "architecture": "SimpleNet(use_circuit=True)",
            "training_sequences": "natural momentum batches only",
            "expected_unexpected_pairs_used_for_training": False,
            "expected_unexpected_contrast_losses_used": False,
            "shape_losses_used": False,
            "shared_feedforward_representation": "same phase1 W_ff and decoder copied into both regimes and frozen",
            "phase1_equalization": phase1_config,
            "model_config": model_config,
            "sensory_corruption_training_only": corruption_config,
            "shared_l23_feedback": "same optional prediction-error feedback, feedback-to-SOM pool, topographic SOM/VIP feedback routing, and feedback-gated/local recurrent inhibition parameters for both regimes",
            "topographic_somvip_routing": "same fixed circular narrow SOM and broad VIP feedback routing for both regimes when enabled",
            "gain_regularization": gain_config,
            "loss_families": [
                "next_step_prediction_ce_all_sequence_transitions",
                "current_decoder_ce_all_timesteps",
                "noisy_current_decoder_ce_all_timesteps",
                "stochastic_l4_channel_dropout_noise_all_training_timesteps",
                "fixed_circular_local_l23_recurrent_inhibition_all_timesteps",
                "fixed_circular_feedback_gated_l23_inhibition_all_timesteps",
                "fixed_circular_feedback_to_som_pool_all_timesteps",
                "fixed_circular_topographic_somvip_feedback_routing_all_timesteps",
                "same_regime_independent_som_to_pyr_gain_floor_regularization",
                "global_mean_l23_activity_all_timesteps",
                "global_squared_l23_activity_all_timesteps",
                "prediction_confidence_weighted_activity_all_natural_timesteps",
                "prediction_confidence_weighted_squared_activity_all_natural_timesteps",
                "phase1_feedforward_locality_sparsity",
                "phase1_orientation_uniform_energy_equalization",
                "phase1_channel_rate_equalization",
                "optional_homeostatic_rate_stabilization",
            ],
            "regime_differences": "scalar weights only",
        },
        "weights": weights,
        "checkpoints": checkpoints,
        "phase1_history": phase1_history,
        "history": history,
    }
    summary_path = os.path.join(args.out, f"train_natural_emergence_{args.candidate}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"SUMMARY {summary_path}", flush=True)
    print("TRAIN_NATURAL_EMERGENCE_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
