#!/usr/bin/env python3
"""Train fixed-basis tuned networks from ordinary momentum sequences only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tuned_emergence_lib import (  # noqa: E402
    N,
    SimpleTunedNet,
    build_tuned_from_config,
    chan,
    device,
    forward_seq_tuned,
    l4_code,
    make_sequences,
    model_config,
)


def candidate_weights(name: str) -> dict:
    base = {
        "sharpen": {
            "pred": 3.0,
            "current": 1.0,
            "noisy_current": 0.7,
            "activity": 0.02,
            "activity_sq": 0.003,
            "pred_conf_energy": 0.08,
            "pred_conf_energy_sq": 0.006,
            "homeostatic": 0.20,
        },
        "dampen": {
            "pred": 3.0,
            "current": 0.42,
            "noisy_current": 0.16,
            "activity": 0.075,
            "activity_sq": 0.012,
            "pred_conf_energy": 0.16,
            "pred_conf_energy_sq": 0.008,
            "homeostatic": 0.35,
        },
    }
    if name == "tuned_mild":
        return base
    if name == "tuned_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 2.0
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["activity"] *= 1.4
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 1.75
        out["dampen"]["pred_conf_energy_sq"] *= 1.5
        out["dampen"]["homeostatic"] = 0.55
        return out
    if name == "tuned_homeo_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["activity"] *= 1.5
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 3.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.0
        out["sharpen"]["homeostatic"] = 0.75
        out["dampen"]["activity"] *= 1.7
        out["dampen"]["activity_sq"] *= 1.6
        out["dampen"]["pred_conf_energy"] *= 2.6
        out["dampen"]["pred_conf_energy_sq"] *= 2.2
        out["dampen"]["homeostatic"] = 0.9
        return out
    if name == "pop_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.1
        out["sharpen"]["noisy_current"] = 0.8
        out["sharpen"]["pred_conf_energy"] *= 1.5
        out["sharpen"]["pred_conf_energy_sq"] *= 1.5
        out["sharpen"]["homeostatic"] = 0.35
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 1.25
        out["dampen"]["activity_sq"] *= 1.25
        out["dampen"]["pred_conf_energy"] *= 1.7
        out["dampen"]["pred_conf_energy_sq"] *= 1.4
        out["dampen"]["homeostatic"] = 0.55
        return out
    if name == "pop_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.1
        out["sharpen"]["noisy_current"] = 0.8
        out["sharpen"]["activity"] *= 1.4
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 2.4
        out["sharpen"]["pred_conf_energy_sq"] *= 2.4
        out["sharpen"]["homeostatic"] = 0.55
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 1.6
        out["dampen"]["activity_sq"] *= 1.6
        out["dampen"]["pred_conf_energy"] *= 2.4
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.75
        return out
    if name == "pop_homeo_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.1
        out["sharpen"]["noisy_current"] = 0.8
        out["sharpen"]["activity"] *= 1.8
        out["sharpen"]["activity_sq"] *= 1.8
        out["sharpen"]["pred_conf_energy"] *= 3.2
        out["sharpen"]["pred_conf_energy_sq"] *= 3.2
        out["sharpen"]["homeostatic"] = 0.9
        out["dampen"]["current"] = 0.60
        out["dampen"]["noisy_current"] = 0.30
        out["dampen"]["activity"] *= 2.0
        out["dampen"]["activity_sq"] *= 1.9
        out["dampen"]["pred_conf_energy"] *= 3.1
        out["dampen"]["pred_conf_energy_sq"] *= 2.6
        out["dampen"]["homeostatic"] = 1.1
        return out
    if name == "pred_inhib_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.15
        out["sharpen"]["noisy_current"] = 0.85
        out["sharpen"]["activity"] *= 1.2
        out["sharpen"]["activity_sq"] *= 1.2
        out["sharpen"]["pred_conf_energy"] *= 2.0
        out["sharpen"]["pred_conf_energy_sq"] *= 1.8
        out["sharpen"]["homeostatic"] = 0.45
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 1.35
        out["dampen"]["activity_sq"] *= 1.35
        out["dampen"]["pred_conf_energy"] *= 2.2
        out["dampen"]["pred_conf_energy_sq"] *= 1.8
        out["dampen"]["homeostatic"] = 0.65
        return out
    if name == "pred_inhib_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.20
        out["sharpen"]["noisy_current"] = 0.90
        out["sharpen"]["activity"] *= 1.45
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 2.7
        out["sharpen"]["pred_conf_energy_sq"] *= 2.5
        out["sharpen"]["homeostatic"] = 0.65
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 1.75
        out["dampen"]["activity_sq"] *= 1.7
        out["dampen"]["pred_conf_energy"] *= 3.0
        out["dampen"]["pred_conf_energy_sq"] *= 2.4
        out["dampen"]["homeostatic"] = 0.85
        return out
    if name == "pred_inhib_homeo":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.25
        out["sharpen"]["noisy_current"] = 0.95
        out["sharpen"]["activity"] *= 1.7
        out["sharpen"]["activity_sq"] *= 1.8
        out["sharpen"]["pred_conf_energy"] *= 3.4
        out["sharpen"]["pred_conf_energy_sq"] *= 3.2
        out["sharpen"]["homeostatic"] = 0.9
        out["dampen"]["current"] = 0.60
        out["dampen"]["noisy_current"] = 0.30
        out["dampen"]["activity"] *= 2.0
        out["dampen"]["activity_sq"] *= 1.9
        out["dampen"]["pred_conf_energy"] *= 3.6
        out["dampen"]["pred_conf_energy_sq"] *= 2.9
        out["dampen"]["homeostatic"] = 1.1
        return out
    if name == "feature_supp_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.20
        out["sharpen"]["noisy_current"] = 0.90
        out["sharpen"]["activity"] *= 1.2
        out["sharpen"]["activity_sq"] *= 1.2
        out["sharpen"]["pred_conf_energy"] *= 2.1
        out["sharpen"]["pred_conf_energy_sq"] *= 1.9
        out["sharpen"]["homeostatic"] = 0.50
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 1.45
        out["dampen"]["activity_sq"] *= 1.4
        out["dampen"]["pred_conf_energy"] *= 2.5
        out["dampen"]["pred_conf_energy_sq"] *= 2.0
        out["dampen"]["homeostatic"] = 0.70
        return out
    if name == "feature_supp_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.25
        out["sharpen"]["noisy_current"] = 0.95
        out["sharpen"]["activity"] *= 1.45
        out["sharpen"]["activity_sq"] *= 1.5
        out["sharpen"]["pred_conf_energy"] *= 2.8
        out["sharpen"]["pred_conf_energy_sq"] *= 2.5
        out["sharpen"]["homeostatic"] = 0.70
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 1.8
        out["dampen"]["activity_sq"] *= 1.7
        out["dampen"]["pred_conf_energy"] *= 3.2
        out["dampen"]["pred_conf_energy_sq"] *= 2.5
        out["dampen"]["homeostatic"] = 0.90
        return out
    if name == "feature_supp_homeo":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.30
        out["sharpen"]["noisy_current"] = 1.00
        out["sharpen"]["activity"] *= 1.7
        out["sharpen"]["activity_sq"] *= 1.8
        out["sharpen"]["pred_conf_energy"] *= 3.5
        out["sharpen"]["pred_conf_energy_sq"] *= 3.2
        out["sharpen"]["homeostatic"] = 0.95
        out["dampen"]["current"] = 0.60
        out["dampen"]["noisy_current"] = 0.30
        out["dampen"]["activity"] *= 2.0
        out["dampen"]["activity_sq"] *= 1.9
        out["dampen"]["pred_conf_energy"] *= 3.8
        out["dampen"]["pred_conf_energy_sq"] *= 3.0
        out["dampen"]["homeostatic"] = 1.15
        return out
    if name == "feature_supp_boundary_034":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.55
        out["sharpen"]["noisy_current"] = 1.30
        out["sharpen"]["activity"] *= 1.8
        out["sharpen"]["activity_sq"] *= 1.9
        out["sharpen"]["pred_conf_energy"] *= 3.7
        out["sharpen"]["pred_conf_energy_sq"] *= 3.4
        out["sharpen"]["homeostatic"] = 1.0
        return out
    if name == "feature_supp_boundary_038":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.80
        out["sharpen"]["noisy_current"] = 1.55
        out["sharpen"]["activity"] *= 1.9
        out["sharpen"]["activity_sq"] *= 2.0
        out["sharpen"]["pred_conf_energy"] *= 4.0
        out["sharpen"]["pred_conf_energy_sq"] *= 3.7
        out["sharpen"]["homeostatic"] = 1.05
        return out
    if name == "feature_supp_boundary_042":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.05
        out["sharpen"]["noisy_current"] = 1.80
        out["sharpen"]["activity"] *= 2.0
        out["sharpen"]["activity_sq"] *= 2.1
        out["sharpen"]["pred_conf_energy"] *= 4.3
        out["sharpen"]["pred_conf_energy_sq"] *= 4.0
        out["sharpen"]["homeostatic"] = 1.1
        return out
    if name == "feature_supp_boundary_050":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["pred_conf_energy"] *= 4.6
        out["sharpen"]["pred_conf_energy_sq"] *= 4.3
        out["sharpen"]["homeostatic"] = 1.15
        return out
    if name == "sat_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.70
        out["sharpen"]["noisy_current"] = 1.45
        out["sharpen"]["activity"] *= 1.7
        out["sharpen"]["activity_sq"] *= 1.8
        out["sharpen"]["pred_conf_energy"] *= 3.5
        out["sharpen"]["pred_conf_energy_sq"] *= 3.2
        out["sharpen"]["homeostatic"] = 0.95
        out["dampen"]["current"] = 0.60
        out["dampen"]["noisy_current"] = 0.30
        out["dampen"]["activity"] *= 2.0
        out["dampen"]["activity_sq"] *= 1.9
        out["dampen"]["pred_conf_energy"] *= 3.8
        out["dampen"]["pred_conf_energy_sq"] *= 3.0
        out["dampen"]["homeostatic"] = 1.15
        return out
    if name == "sat_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.05
        out["sharpen"]["noisy_current"] = 1.80
        out["sharpen"]["activity"] *= 2.0
        out["sharpen"]["activity_sq"] *= 2.1
        out["sharpen"]["pred_conf_energy"] *= 4.3
        out["sharpen"]["pred_conf_energy_sq"] *= 4.0
        out["sharpen"]["homeostatic"] = 1.10
        out["dampen"]["current"] = 0.58
        out["dampen"]["noisy_current"] = 0.28
        out["dampen"]["activity"] *= 2.1
        out["dampen"]["activity_sq"] *= 2.0
        out["dampen"]["pred_conf_energy"] *= 4.1
        out["dampen"]["pred_conf_energy_sq"] *= 3.3
        out["dampen"]["homeostatic"] = 1.20
        return out
    if name == "sat_strong":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["pred_conf_energy"] *= 4.6
        out["sharpen"]["pred_conf_energy_sq"] *= 4.3
        out["sharpen"]["homeostatic"] = 1.15
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 2.2
        out["dampen"]["activity_sq"] *= 2.1
        out["dampen"]["pred_conf_energy"] *= 4.4
        out["dampen"]["pred_conf_energy_sq"] *= 3.6
        out["dampen"]["homeostatic"] = 1.25
        return out
    if name == "strict_sat_mild":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 1.70
        out["sharpen"]["noisy_current"] = 1.45
        out["sharpen"]["activity"] *= 1.7
        out["sharpen"]["activity_sq"] *= 1.8
        out["sharpen"]["pred_conf_energy"] *= 3.5
        out["sharpen"]["pred_conf_energy_sq"] *= 3.2
        out["sharpen"]["homeostatic"] = 0.95
        out["dampen"]["current"] = 0.60
        out["dampen"]["noisy_current"] = 0.30
        out["dampen"]["activity"] *= 2.0
        out["dampen"]["activity_sq"] *= 1.9
        out["dampen"]["pred_conf_energy"] *= 3.8
        out["dampen"]["pred_conf_energy_sq"] *= 3.0
        out["dampen"]["homeostatic"] = 1.15
        return out
    if name == "strict_sat_energy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.05
        out["sharpen"]["noisy_current"] = 1.80
        out["sharpen"]["activity"] *= 2.0
        out["sharpen"]["activity_sq"] *= 2.1
        out["sharpen"]["pred_conf_energy"] *= 4.3
        out["sharpen"]["pred_conf_energy_sq"] *= 4.0
        out["sharpen"]["homeostatic"] = 1.10
        out["dampen"]["current"] = 0.58
        out["dampen"]["noisy_current"] = 0.28
        out["dampen"]["activity"] *= 2.1
        out["dampen"]["activity_sq"] *= 2.0
        out["dampen"]["pred_conf_energy"] *= 4.1
        out["dampen"]["pred_conf_energy_sq"] *= 3.3
        out["dampen"]["homeostatic"] = 1.20
        return out
    if name == "strict_sat_strong":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["pred_conf_energy"] *= 4.6
        out["sharpen"]["pred_conf_energy_sq"] *= 4.3
        out["sharpen"]["homeostatic"] = 1.15
        out["dampen"]["current"] = 0.55
        out["dampen"]["noisy_current"] = 0.25
        out["dampen"]["activity"] *= 2.2
        out["dampen"]["activity_sq"] *= 2.1
        out["dampen"]["pred_conf_energy"] *= 4.4
        out["dampen"]["pred_conf_energy_sq"] *= 3.6
        out["dampen"]["homeostatic"] = 1.25
        return out
    if name == "strict_no_pred_conf_dampen_heavy":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["homeostatic"] = 1.15
        out["dampen"]["current"] = 0.42
        out["dampen"]["noisy_current"] = 0.14
        out["dampen"]["activity"] *= 4.0
        out["dampen"]["activity_sq"] *= 4.0
        out["dampen"]["homeostatic"] = 1.6
        return out
    if name == "strict_no_pred_conf_dampen_max":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["homeostatic"] = 1.15
        out["dampen"]["current"] = 0.32
        out["dampen"]["noisy_current"] = 0.08
        out["dampen"]["activity"] *= 6.0
        out["dampen"]["activity_sq"] *= 6.0
        out["dampen"]["homeostatic"] = 2.0
        return out
    if name == "strict_no_pred_conf_dampen_ultra":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["homeostatic"] = 1.15
        out["dampen"]["current"] = 0.24
        out["dampen"]["noisy_current"] = 0.04
        out["dampen"]["activity"] *= 10.0
        out["dampen"]["activity_sq"] *= 10.0
        out["dampen"]["homeostatic"] = 3.0
        return out
    if name == "strict_no_pred_conf_dampen_extreme":
        out = copy.deepcopy(base)
        out["sharpen"]["current"] = 2.35
        out["sharpen"]["noisy_current"] = 2.05
        out["sharpen"]["activity"] *= 2.1
        out["sharpen"]["activity_sq"] *= 2.2
        out["sharpen"]["homeostatic"] = 1.15
        out["dampen"]["current"] = 0.18
        out["dampen"]["noisy_current"] = 0.02
        out["dampen"]["activity"] *= 14.0
        out["dampen"]["activity_sq"] *= 14.0
        out["dampen"]["homeostatic"] = 4.0
        return out
    if name == "strict_adapt_no_pred_ultra":
        return candidate_weights("strict_no_pred_conf_dampen_ultra")
    if name == "strict_adapt_no_pred_extreme":
        return candidate_weights("strict_no_pred_conf_dampen_extreme")
    if name == "strict_local_comp_no_pred_ultra":
        return candidate_weights("strict_no_pred_conf_dampen_ultra")
    if name == "strict_local_comp_no_pred_extreme":
        return candidate_weights("strict_no_pred_conf_dampen_extreme")
    raise ValueError(f"unknown candidate {name!r}")


def candidate_model_overrides(name: str) -> dict:
    if name == "feature_supp_mild":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.18},
            "dampen": {"pred_feature_supp_strength": 0.65},
        }
    if name == "feature_supp_energy":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.25},
            "dampen": {"pred_feature_supp_strength": 0.90},
        }
    if name == "feature_supp_homeo":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.32},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "feature_supp_boundary_034":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.34},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "feature_supp_boundary_038":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.38},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "feature_supp_boundary_042":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.42},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "feature_supp_boundary_050":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.50},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "sat_mild":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.38},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "sat_energy":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.42},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    if name == "sat_strong":
        return {
            "sharpen": {"pred_feature_supp_strength": 0.50},
            "dampen": {"pred_feature_supp_strength": 1.15},
        }
    return {"sharpen": {}, "dampen": {}}


def zero_prediction_confidence_weights(weights: dict) -> dict:
    out = copy.deepcopy(weights)
    for regime in ("sharpen", "dampen"):
        out[regime]["pred_conf_energy"] = 0.0
        out[regime]["pred_conf_energy_sq"] = 0.0
    return out


def ramped_weight(final_weight: float, step: int, warmup_steps: int, ramp_steps: int) -> float:
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    return final_weight * min(1.0, (step - warmup_steps) / ramp_steps)


def prediction_confidence_mask(pred_logits: torch.Tensor, temp: float, topk: int) -> torch.Tensor:
    mask = F.softmax(pred_logits.detach() / temp, dim=-1)
    if topk > 0:
        keep = torch.zeros_like(mask)
        keep.scatter_(-1, mask.topk(topk, dim=-1).indices, 1.0)
        mask = mask * keep
    return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def noisy_current_ce(net: SimpleTunedNet, r_flat: torch.Tensor, target: torch.Tensor, sigma: float, repeats: int):
    if sigma <= 0.0 or repeats <= 0:
        return F.cross_entropy(net.decode(r_flat), target)
    r_rep = r_flat.unsqueeze(0).expand(repeats, *r_flat.shape)
    noisy = r_rep + sigma * torch.randn_like(r_rep)
    logits = net.decode(noisy.reshape(repeats * r_flat.shape[0], N))
    return F.cross_entropy(logits, target.repeat(repeats))


def state_dict_sha256(state: dict) -> str:
    """Stable hash for initial-state equality evidence."""
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def local_comp_gain_raw_value(net: SimpleTunedNet) -> float | None:
    if not getattr(net, "local_comp_trainable", False):
        return None
    return float(net.local_comp_strength_raw.detach().cpu().item())


def local_comp_effective_strength_value(net: SimpleTunedNet) -> float:
    return float(net.local_comp_effective_strength().detach().cpu().item())


def natural_sequence_losses(net: SimpleTunedNet, args):
    theta = make_sequences(args.batch, args.sequence_len, mode="momentum", p_stay=args.p_stay)
    preds, r_all = forward_seq_tuned(net, theta, 1.0)
    target_current = chan(theta)
    r_flat = r_all.reshape(-1, N)
    current_flat = target_current.reshape(-1)
    pred_ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), target_current[:, 1:].reshape(-1))
    current_ce = F.cross_entropy(net.decode(r_flat), current_flat)
    noisy_ce = noisy_current_ce(net, r_flat, current_flat, args.noise_sigma, args.noise_repeats)
    activity = r_all.abs().mean()
    activity_sq = r_all.square().mean()

    mask = prediction_confidence_mask(preds[:, :-1, :], args.prediction_mask_temp, args.prediction_mask_topk)
    r_with_prior = r_all[:, 1:, :]
    pred_conf_activity = (mask * r_with_prior.abs()).sum(dim=-1)
    pred_conf_activity_sq = (mask * r_with_prior.square()).sum(dim=-1)
    pred_conf_energy = pred_conf_activity.mean() + args.pred_conf_energy_max_weight * pred_conf_activity.max()
    pred_conf_energy_sq = pred_conf_activity_sq.mean() + args.pred_conf_energy_max_weight * pred_conf_activity_sq.max()

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
    }


@torch.no_grad()
def held_acc(net: SimpleTunedNet, seed: int, args):
    torch.manual_seed(seed)
    theta = make_sequences(args.held_batch, args.sequence_len, mode="momentum", p_stay=args.p_stay)
    preds, _ = forward_seq_tuned(net, theta, 1.0)
    ok = preds[:, :-1].argmax(-1) == chan(theta[:, 1:])
    return float(ok.float().mean().item() * 100.0)


def train_one(
    regime: str,
    weights: dict,
    base_state: dict,
    out_path: str,
    args,
    config: dict,
    initial_state_evidence: dict,
):
    net = build_tuned_from_config(config).to(device)
    net.load_state_dict(copy.deepcopy(base_state))
    loaded_initial_hash = state_dict_sha256(net.state_dict())
    if loaded_initial_hash != initial_state_evidence["shared_state_dict_sha256"]:
        raise ValueError(f"{regime} initial state hash mismatch: {loaded_initial_hash} != {initial_state_evidence['shared_state_dict_sha256']}")
    opt_params = list(net.gru.parameters()) + list(net.W_fb.parameters()) + [net.circ_raw, net.decoder_gain_raw]
    if getattr(net, "local_comp_trainable", False):
        opt_params.append(net.local_comp_strength_raw)
    opt = torch.optim.Adam(opt_params, lr=args.lr)
    history = []
    print(f"\n=== TUNED EMERGENCE {args.candidate}:{regime.upper()} steps={args.steps} device={device} ===", flush=True)
    print(json.dumps({"regime": regime, "weights": weights, "model_config": config}, sort_keys=True), flush=True)
    for step in range(1, args.steps + 1):
        losses = natural_sequence_losses(net, args)
        energy_scale = ramped_weight(1.0, step, args.energy_warmup_steps, args.energy_ramp_steps)
        loss = (
            weights["pred"] * losses["pred_ce"]
            + weights["current"] * losses["current_ce"]
            + weights["noisy_current"] * losses["noisy_current_ce"]
            + energy_scale * weights["activity"] * losses["activity"]
            + energy_scale * weights["activity_sq"] * losses["activity_sq"]
            + energy_scale * weights["pred_conf_energy"] * losses["pred_conf_energy"]
            + energy_scale * weights["pred_conf_energy_sq"] * losses["pred_conf_energy_sq"]
            + energy_scale * weights["homeostatic"] * losses["homeostatic"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
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
                "mean_channel_rate": float(losses["mean_channel_rate"].item()),
                "std_channel_rate": float(losses["std_channel_rate"].item()),
                "min_channel_rate": float(losses["min_channel_rate"].item()),
                "max_channel_rate": float(losses["max_channel_rate"].item()),
                "effective_activity_weight": float(energy_scale * weights["activity"]),
                "effective_activity_sq_weight": float(energy_scale * weights["activity_sq"]),
                "effective_pred_conf_energy_weight": float(energy_scale * weights["pred_conf_energy"]),
                "effective_pred_conf_energy_sq_weight": float(energy_scale * weights["pred_conf_energy_sq"]),
                "effective_homeostatic_weight": float(energy_scale * weights["homeostatic"]),
                "held_acc_percent": held_acc(net, args.seed + step, args),
                "decoder_gain": float(F.softplus(net.decoder_gain_raw).detach().cpu().item()),
                "gains_g_v_g_s_g_sv_g_e_g_ps": gains,
                "local_comp_strength_configured": float(net.local_comp_strength),
                "local_comp_strength_learned": local_comp_effective_strength_value(net),
                "local_comp_strength_raw": local_comp_gain_raw_value(net),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    final_local_comp_strength = local_comp_effective_strength_value(net)
    torch.save({
        "state_dict": net.state_dict(),
        "tuned_net_config": model_config(net),
        "training_contract": {
            "architecture": "SimpleTunedNet fixed local L4->L2/3 basis, constrained orientation readout",
            "readout_contract": config["readout"],
            "shared_prediction_dependent_inhibition": {
                "strength": config["pred_inhib_strength"],
                "sigma_channels": config["pred_inhib_sigma_channels"],
                "driven_by": "top_down_feedback_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
            },
            "shared_predicted_feature_suppression": {
                "strength": config["pred_feature_supp_strength"],
                "driven_by": "same_channel_top_down_feedback_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
            },
            "shared_saturating_l23_rate": {
                "r_max": config["rate_saturation_r_max"],
                "r_half": config["rate_saturation_r_half"],
                "equation": "r_max * relu(preactivation) / (r_half + relu(preactivation)); disabled when r_max <= 0",
                "same_equation_for_sharpen_and_dampen": True,
            },
            "shared_l23_activity_dependent_adaptation": {
                "strength": config["adapt_strength"],
                "decay": config["adapt_decay"],
                "sigma_channels": config["adapt_sigma_channels"],
                "state_update": "adapt = decay * adapt + (1 - decay) * circular_smooth(previous_l23_activity)",
                "effect": "subtract adapt_strength * adapt from L2/3 preactivation before relu/saturation",
                "driven_by": "previous_l23_activity_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
                "uses_prediction_logits_or_masks": False,
            },
            "shared_current_step_l23_local_competition": {
                "strength": config["local_comp_strength"],
                "strength_configured_initial": config["local_comp_strength"],
                "strength_trainable": config["local_comp_trainable"],
                "initial_raw_strength": initial_state_evidence["local_comp_strength_raw"],
                "learned_strength_at_save": final_local_comp_strength,
                "final_raw_strength": local_comp_gain_raw_value(net),
                "sigma_channels": config["local_comp_sigma_channels"],
                "power": config["local_comp_power"],
                "mode": config["local_comp_mode"],
                "equation": "r = relu(pre); pool = circular_smooth(r ** power); r = r / (1 + strength * pool) for divisive mode",
                "driven_by": "current_l23_activity_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
                "uses_prediction_logits_or_masks": False,
            },
            "trainable_local_competition_strength": {
                "enabled": config["local_comp_trainable"],
                "parameterization": "strength = softplus(local_comp_strength_raw)",
                "same_initial_raw_strength_for_regimes": True,
                "same_optimizer_treatment_for_regimes": True,
                "optimizer": "Adam",
                "lr": args.lr,
                "bounds_or_clipping": "none",
                "freeze_schedule": "none",
            },
            "initial_state_evidence": initial_state_evidence,
            "training_sequences": "natural momentum batches only",
            "expected_unexpected_pairs_used_for_training": False,
            "expected_unexpected_contrast_losses_used": False,
            "shape_losses_used": False,
            "loss_families": [
                "next_step_prediction_ce_all_sequence_transitions",
                "current_constrained_readout_ce_all_timesteps",
                "noisy_current_constrained_readout_ce_all_timesteps",
                "global_mean_l23_activity_all_timesteps",
                "global_squared_l23_activity_all_timesteps",
                "homeostatic_rate_stabilization_all_timesteps",
            ] + ([] if args.zero_pred_conf_energy else [
                "prediction_confidence_weighted_activity_all_natural_timesteps",
                "prediction_confidence_weighted_squared_activity_all_natural_timesteps",
            ]),
            "regime_differences": "scalar weights only",
            "strict_shared_mechanism_requested": bool(args.strict_shared_mechanism),
            "training_initial_state_shared": True,
            "prediction_confidence_energy_objective_enabled": not bool(args.zero_pred_conf_energy),
            "prediction_targeted_activity_penalty_replacement_added": False,
        },
    }, out_path)
    print(f"SAVED {args.candidate}:{regime} {out_path}", flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description="Train fixed/local tuned-basis emergence checkpoints.")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--candidate",
        choices=(
            "tuned_mild",
            "tuned_energy",
            "tuned_homeo_energy",
            "pop_mild",
            "pop_energy",
            "pop_homeo_energy",
            "pred_inhib_mild",
            "pred_inhib_energy",
            "pred_inhib_homeo",
            "feature_supp_mild",
            "feature_supp_energy",
            "feature_supp_homeo",
            "feature_supp_boundary_034",
            "feature_supp_boundary_038",
            "feature_supp_boundary_042",
            "feature_supp_boundary_050",
            "sat_mild",
            "sat_energy",
            "sat_strong",
            "strict_sat_mild",
            "strict_sat_energy",
            "strict_sat_strong",
            "strict_no_pred_conf_dampen_heavy",
            "strict_no_pred_conf_dampen_max",
            "strict_no_pred_conf_dampen_ultra",
            "strict_no_pred_conf_dampen_extreme",
            "strict_adapt_no_pred_ultra",
            "strict_adapt_no_pred_extreme",
            "strict_local_comp_no_pred_ultra",
            "strict_local_comp_no_pred_extreme",
        ),
        default="feature_supp_energy",
    )
    ap.add_argument("--only-regime", choices=("both", "sharpen", "dampen"), default="both")
    ap.add_argument("--strict-shared-mechanism", action="store_true")
    ap.add_argument("--zero-pred-conf-energy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held-batch", type=int, default=4096)
    ap.add_argument("--sequence-len", type=int, default=12)
    ap.add_argument("--p-stay", type=float, default=0.9)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ff-sigma-channels", type=float, default=1.1)
    ap.add_argument("--ff-gain", type=float, default=1.6)
    ap.add_argument("--decoder-gain", type=float, default=8.0)
    ap.add_argument("--readout", choices=("channel", "population_vector"), default="population_vector")
    ap.add_argument("--population-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pred-inhib-strength", type=float, default=0.65)
    ap.add_argument("--pred-inhib-sigma-channels", type=float, default=0.65)
    ap.add_argument("--pred-feature-supp-strength", type=float, default=0.0)
    ap.add_argument("--rate-saturation-r-max", type=float, default=0.0)
    ap.add_argument("--rate-saturation-r-half", type=float, default=1.0)
    ap.add_argument("--adapt-strength", type=float, default=0.0)
    ap.add_argument("--adapt-decay", type=float, default=0.85)
    ap.add_argument("--adapt-sigma-channels", type=float, default=1.0)
    ap.add_argument("--local-comp-strength", type=float, default=0.0)
    ap.add_argument("--local-comp-sigma-channels", type=float, default=1.0)
    ap.add_argument("--local-comp-power", type=float, default=1.0)
    ap.add_argument("--local-comp-mode", choices=("divisive", "subtractive"), default="divisive")
    ap.add_argument("--trainable-local-comp-strength", action="store_true")
    ap.add_argument("--noise-sigma", type=float, default=0.6)
    ap.add_argument("--noise-repeats", type=int, default=2)
    ap.add_argument("--prediction-mask-temp", type=float, default=0.22)
    ap.add_argument("--prediction-mask-topk", type=int, default=0)
    ap.add_argument("--pred-conf-energy-max-weight", type=float, default=0.25)
    ap.add_argument("--homeostatic-target", type=float, default=0.12)
    ap.add_argument("--energy-warmup-steps", type=int, default=3000)
    ap.add_argument("--energy-ramp-steps", type=int, default=2500)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    config = {
        "hidden": args.hidden,
        "ff_sigma_channels": args.ff_sigma_channels,
        "ff_gain": args.ff_gain,
        "decoder_gain": args.decoder_gain,
        "readout": args.readout,
        "population_normalize": args.population_normalize,
        "pred_inhib_strength": args.pred_inhib_strength,
        "pred_inhib_sigma_channels": args.pred_inhib_sigma_channels,
        "pred_feature_supp_strength": args.pred_feature_supp_strength,
        "rate_saturation_r_max": args.rate_saturation_r_max,
        "rate_saturation_r_half": args.rate_saturation_r_half,
        "adapt_strength": args.adapt_strength,
        "adapt_decay": args.adapt_decay,
        "adapt_sigma_channels": args.adapt_sigma_channels,
        "local_comp_strength": args.local_comp_strength,
        "local_comp_trainable": args.trainable_local_comp_strength,
        "local_comp_sigma_channels": args.local_comp_sigma_channels,
        "local_comp_power": args.local_comp_power,
        "local_comp_mode": args.local_comp_mode,
    }
    base = build_tuned_from_config(config).to(device)
    base_state = copy.deepcopy(base.state_dict())
    initial_hash = state_dict_sha256(base_state)
    initial_state_evidence = {
        "shared_state_dict_sha256": initial_hash,
        "regime_initial_state_sha256": {
            "sharpen": initial_hash,
            "dampen": initial_hash,
        },
        "regime_hashes_match": True,
        "local_comp_strength_configured": float(args.local_comp_strength),
        "local_comp_strength_effective": local_comp_effective_strength_value(base),
        "local_comp_strength_raw": local_comp_gain_raw_value(base),
        "local_comp_strength_trainable": bool(args.trainable_local_comp_strength),
    }
    weights = candidate_weights(args.candidate)
    if args.zero_pred_conf_energy:
        weights = zero_prediction_confidence_weights(weights)
    model_overrides = {"sharpen": {}, "dampen": {}} if args.strict_shared_mechanism else candidate_model_overrides(args.candidate)
    regime_configs = {
        "sharpen": {**config, **model_overrides["sharpen"]},
        "dampen": {**config, **model_overrides["dampen"]},
    }
    mechanism_audit = {
        "strict_shared_mechanism_requested": bool(args.strict_shared_mechanism),
        "regime_model_configs_identical": bool(regime_configs["sharpen"] == regime_configs["dampen"]),
        "model_overrides": model_overrides,
        "allowed_regime_differences": "scalar objective weights only",
        "training_initial_state_shared": True,
        "initial_state_evidence": initial_state_evidence,
    }
    if args.strict_shared_mechanism and (args.only_regime != "both" or not mechanism_audit["regime_model_configs_identical"]):
        raise ValueError(f"strict shared mechanism audit failed before training: {mechanism_audit}")
    checkpoints = {
        "sharpen": os.path.join(args.out, f"ckpt_tuned_emergence_{args.candidate}_sharpen.pt"),
        "dampen": os.path.join(args.out, f"ckpt_tuned_emergence_{args.candidate}_dampen.pt"),
    }

    print(f"device={device}", flush=True)
    print(f"out={args.out}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"candidate={args.candidate}", flush=True)
    print("training_data=make_sequences(mode='momentum') only", flush=True)
    print("expected_unexpected_pairs_used_for_training=false", flush=True)
    print("shape_losses_used=false", flush=True)
    print(f"readout_contract={args.readout}", flush=True)
    print(f"shared_prediction_dependent_inhibition_strength={args.pred_inhib_strength}", flush=True)
    print(f"shared_prediction_dependent_inhibition_sigma_channels={args.pred_inhib_sigma_channels}", flush=True)
    print(f"base_predicted_feature_suppression_strength={args.pred_feature_supp_strength}", flush=True)
    print(f"shared_rate_saturation_r_max={args.rate_saturation_r_max}", flush=True)
    print(f"shared_rate_saturation_r_half={args.rate_saturation_r_half}", flush=True)
    print(f"shared_l23_adapt_strength={args.adapt_strength}", flush=True)
    print(f"shared_l23_adapt_decay={args.adapt_decay}", flush=True)
    print(f"shared_l23_adapt_sigma_channels={args.adapt_sigma_channels}", flush=True)
    print(f"shared_l23_local_comp_strength={args.local_comp_strength}", flush=True)
    print(f"shared_l23_local_comp_trainable={args.trainable_local_comp_strength}", flush=True)
    print(f"shared_initial_state_evidence={json.dumps(initial_state_evidence, sort_keys=True)}", flush=True)
    print(f"shared_l23_local_comp_sigma_channels={args.local_comp_sigma_channels}", flush=True)
    print(f"shared_l23_local_comp_power={args.local_comp_power}", flush=True)
    print(f"shared_l23_local_comp_mode={args.local_comp_mode}", flush=True)
    print(f"strict_shared_mechanism={args.strict_shared_mechanism}", flush=True)
    print(f"zero_pred_conf_energy={args.zero_pred_conf_energy}", flush=True)
    print(f"regime_model_configs={json.dumps(regime_configs, sort_keys=True)}", flush=True)
    print(f"mechanism_audit={json.dumps(mechanism_audit, sort_keys=True)}", flush=True)
    print(f"model_config={json.dumps(config, sort_keys=True)}", flush=True)
    if args.only_regime == "both":
        regimes = ("sharpen", "dampen")
    else:
        regimes = (args.only_regime,)
    history = {
        regime: train_one(regime, weights[regime], base_state, checkpoints[regime], args, regime_configs[regime], initial_state_evidence)
        for regime in regimes
    }
    summary = {
        "args": vars(args),
        "device": device,
        "contract": {
            "architecture": "SimpleTunedNet fixed/local orientation-tuned L2/3 basis",
            "readout_contract": args.readout,
            "shared_prediction_dependent_inhibition": {
                "strength": args.pred_inhib_strength,
                "sigma_channels": args.pred_inhib_sigma_channels,
                "driven_by": "top_down_feedback_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
            },
            "shared_predicted_feature_suppression": {
                "base_strength": args.pred_feature_supp_strength,
                "regime_configs": regime_configs,
                "driven_by": "same_channel_top_down_feedback_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
            },
            "shared_saturating_l23_rate": {
                "r_max": args.rate_saturation_r_max,
                "r_half": args.rate_saturation_r_half,
                "equation": "r_max * relu(preactivation) / (r_half + relu(preactivation)); disabled when r_max <= 0",
                "same_equation_for_sharpen_and_dampen": True,
            },
            "shared_l23_activity_dependent_adaptation": {
                "strength": args.adapt_strength,
                "decay": args.adapt_decay,
                "sigma_channels": args.adapt_sigma_channels,
                "state_update": "adapt = decay * adapt + (1 - decay) * circular_smooth(previous_l23_activity)",
                "effect": "subtract adapt_strength * adapt from L2/3 preactivation before relu/saturation",
                "driven_by": "previous_l23_activity_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
                "uses_prediction_logits_or_masks": False,
            },
            "shared_current_step_l23_local_competition": {
                "strength": args.local_comp_strength,
                "strength_trainable": args.trainable_local_comp_strength,
                "initial_raw_strength": initial_state_evidence["local_comp_strength_raw"],
                "sigma_channels": args.local_comp_sigma_channels,
                "power": args.local_comp_power,
                "mode": args.local_comp_mode,
                "equation": "r = relu(pre); pool = circular_smooth(r ** power); r = r / (1 + strength * pool) for divisive mode",
                "driven_by": "current_l23_activity_at_every_timestep",
                "same_equation_for_sharpen_and_dampen": True,
                "uses_prediction_logits_or_masks": False,
            },
            "trainable_local_competition_strength": {
                "enabled": args.trainable_local_comp_strength,
                "parameterization": "strength = softplus(local_comp_strength_raw)",
                "same_initial_raw_strength_for_regimes": True,
                "same_optimizer_treatment_for_regimes": True,
                "optimizer": "Adam",
                "lr": args.lr,
                "bounds_or_clipping": "none",
                "freeze_schedule": "none",
            },
            "initial_state_evidence": initial_state_evidence,
            "training_sequences": "natural momentum batches only",
            "expected_unexpected_pairs_used_for_training": False,
            "expected_unexpected_contrast_losses_used": False,
            "shape_losses_used": False,
            "regime_differences": "scalar weights only",
            "model_config": config,
            "regime_model_configs": regime_configs,
            "strict_shared_mechanism_audit": mechanism_audit,
            "prediction_confidence_energy_objective_enabled": not bool(args.zero_pred_conf_energy),
            "prediction_targeted_activity_penalty_replacement_added": False,
        },
        "weights": weights,
        "checkpoints": checkpoints,
        "history": history,
    }
    summary_path = os.path.join(args.out, f"train_tuned_emergence_{args.candidate}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"SUMMARY {summary_path}", flush=True)
    print("TRAIN_TUNED_EMERGENCE_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
