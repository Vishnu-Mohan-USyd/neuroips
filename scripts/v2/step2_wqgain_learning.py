"""Task #74 β-mechanism — Step 2: toy learning rule for W_q_gain.

Closed-loop test of a local three-factor rule that should drive
``W_q_gain`` from the uniform init (all ones) to the hand-crafted
target pattern validated in Step 1:

    W_q_gain[c, j]  →  1 - g0   if unit j prefers cue c's expected orient
                    →  1.0      otherwise

Toy setup (Lead dispatch 2026-04-23):
  * 2 cues × 200 trials = 400 trials total, shuffled.
  * 75% validity — probe matches cue's expected orient on 75% of trials,
    other orient on 25%, per Kok protocol.
  * Phase-2 substrate is loaded and FROZEN; task weights zeroed. Only
    ``W_q_gain`` updates.
  * r_l23e[j] := mean probe-window L23E rate captured during the trial
    forward pass (closed-loop — the update sees the current W_q_gain's
    effect on r).

Rules available via --rule:
  * rule1   : ΔW[c,j] = −η · cue_on[c] · r_j · sign_matched
              sign_matched = +1 matched, −1 mismatched. Clamp W ∈ [0.1, 1.0].
              Lead's candidate rule. Expected failure mode: W drops to
              floor 0.1 on pref units because there is no balancing term
              at the target 0.7 — drive remains net-negative once r ≥
              non-pref r, so only the clamp stops descent.
  * rule2a  : rule1 + γ · (1 − W) (variant a — homeostasis toward 1).
              Gives a non-clamp fixed point at
                  W* = 1 / (1 + (η/γ)·α)
              where α = 0.75·R_matched − 0.25·R_mismatched. Tuning
              η/γ ≈ 0.337 against R_matched≈1.7, R_mismatched≈0.01
              predicts W* ≈ 0.7 on pref units and clamp-at-1.0 on
              non-pref units (α ≈ −0.42 there).

Metrics:
  step2_toy_learning: final_mean_gain_pref=<#> final_mean_gain_nonpref=<#>
     target_pref=<#> target_nonpref=<#> matches_target=<T/F>
     iterations_to_converge=<#> rule=<name>

``matches_target`` is True if |final_pref − target_pref| ≤ 0.1 AND
|final_nonpref − target_nonpref| ≤ 0.1. Convergence = smoothed
|ΔW_per_trial| mean over a 50-trial window falling below 5e-5 for
100 consecutive trials.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


# ---------------------------------------------------------------------------
# Trial forward (closed-loop — current W_q_gain shapes r_l23e)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_probe_trial_with_gate(
    bundle, *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float, generator: torch.Generator,
) -> Tensor:
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)
    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total
    rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, q_cue
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=q_t)
        if delay_end <= t < probe1_end:
            rates.append(info["r_l23"][0].clone())
    return torch.stack(rates, dim=0).mean(dim=0)


@torch.no_grad()
def _run_localizer_trial(
    bundle, *, probe_orientation_deg: float, timing: KokTiming,
    noise_std: float, generator: torch.Generator,
) -> Tensor:
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total
    rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end or cue_end <= t < delay_end:
            frame = blank
        elif delay_end <= t < probe1_end:
            frame = probe
        elif probe1_end <= t < blank2_end:
            frame = blank
        else:
            frame = probe
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=None)
        if delay_end <= t < probe1_end:
            rates.append(info["r_l23"][0].clone())
    return torch.stack(rates, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Learning rule step
# ---------------------------------------------------------------------------


@torch.no_grad()
def _apply_rule(
    l23, *, cue_id: int, r_l23e: Tensor, matched: bool, rule: str,
    lr: float, homeostasis_gamma: float,
    gain_min: float, gain_max: float,
) -> float:
    """Apply the chosen learning rule in-place on ``l23.W_q_gain``.

    Returns the L2 norm of ΔW actually applied (post-clamp), used for
    convergence detection.
    """
    sign = 1.0 if matched else -1.0
    delta = -lr * sign * r_l23e                 # [n_l23_e]
    if rule == "rule2a":
        # Homeostasis pulling W toward 1.0
        delta = delta + homeostasis_gamma * (1.0 - l23.W_q_gain[cue_id])
    elif rule != "rule1":
        raise ValueError(f"unknown rule {rule!r}")
    new_row = torch.clamp(
        l23.W_q_gain[cue_id] + delta, gain_min, gain_max,
    )
    actual_delta = new_row - l23.W_q_gain[cue_id]
    l23.W_q_gain[cue_id].copy_(new_row)
    return float(actual_delta.norm().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials-per-cue", type=int, default=200)
    ap.add_argument("--validity", type=float, default=0.75)
    ap.add_argument("--g0", type=float, default=0.3,
                    help="target gain floor on pref-for-cue units "
                         "(expected W* = 1 − g0)")
    ap.add_argument("--pref-tol-deg", type=float, default=15.0)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-localizer-orients", type=int, default=36)
    ap.add_argument("--n-localizer-trials", type=int, default=8)
    ap.add_argument("--rule", choices=("rule1", "rule2a"), default="rule1")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--homeostasis-gamma", type=float, default=3e-3,
                    help="only used for rule2a")
    ap.add_argument("--gain-min", type=float, default=0.1)
    ap.add_argument("--gain-max", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--conv-window", type=int, default=50)
    ap.add_argument("--conv-eps", type=float, default=5e-5)
    ap.add_argument("--conv-persist", type=int, default=100)
    ap.add_argument(
        "--output", type=Path,
        default=Path("logs/task74/step2_wqgain_learning.json"),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- Load substrate and zero task weights ---------------------------
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cm = bundle.net.context_memory
    with torch.no_grad():
        cm.W_qm_task.data.zero_()
        cm.W_mh_task_exc.data.zero_()
        cm.W_mh_task_inh.data.zero_()

    timing = KokTiming()
    cue_mapping = cue_mapping_from_seed(args.seed)
    l23 = bundle.net.l23_e
    n_l23_e = l23.n_units
    n_cue = l23.n_cue

    # --- Localizer → per-unit preferred orientation (ground truth) ------
    orients = np.linspace(
        0.0, 180.0, args.n_localizer_orients, endpoint=False,
    )
    gen_loc = torch.Generator().manual_seed(args.seed + 1)
    loc_rates, loc_y = [], []
    for theta in orients:
        for _ in range(args.n_localizer_trials):
            r = _run_localizer_trial(
                bundle, probe_orientation_deg=float(theta),
                timing=timing, noise_std=args.noise_std,
                generator=gen_loc,
            )
            loc_rates.append(r.cpu().numpy())
            loc_y.append(float(theta))
    L = np.stack(loc_rates, axis=0)
    L_y = np.asarray(loc_y)
    orient_mean = np.stack([
        L[np.abs(L_y - o) < 1e-6].mean(axis=0) for o in orients
    ], axis=0)
    unit_pref_idx = np.argmax(orient_mean, axis=0)
    unit_pref_deg = orients[unit_pref_idx]

    def _mask(anchor: float, tol: float = args.pref_tol_deg) -> np.ndarray:
        d = np.abs(((unit_pref_deg - anchor + 90.0) % 180.0) - 90.0)
        return d <= tol
    pref_mask_per_cue = np.stack([
        _mask(cue_mapping[0]), _mask(cue_mapping[1]),
    ], axis=0)                                   # [2, n_l23_e] bool

    # --- Initialize W_q_gain = ones (default) ---------------------------
    with torch.no_grad():
        l23.W_q_gain.fill_(1.0)

    # --- Build trial sequence -------------------------------------------
    rng = random.Random(args.seed)
    trials: list[tuple[int, float, bool]] = []
    for c in (0, 1):
        for _ in range(args.n_trials_per_cue):
            if rng.random() < args.validity:
                probe = cue_mapping[c]
                matched = True
            else:
                probe = (
                    CUE_ORIENTATIONS_DEG[1]
                    if abs(cue_mapping[c] - CUE_ORIENTATIONS_DEG[0]) < 1e-6
                    else CUE_ORIENTATIONS_DEG[0]
                )
                matched = False
            trials.append((int(c), float(probe), bool(matched)))
    rng.shuffle(trials)
    n_trials = len(trials)

    # --- Training loop ---------------------------------------------------
    gen = torch.Generator().manual_seed(args.seed + 2)
    history: list[dict] = []
    rolling_dnorm = np.zeros(args.conv_window, dtype=np.float32)
    conv_persist_counter = 0
    iterations_to_converge: int = -1

    for i, (c, probe_deg, matched) in enumerate(trials):
        r = _run_probe_trial_with_gate(
            bundle, cue_id=c, probe_orientation_deg=probe_deg,
            timing=timing, noise_std=args.noise_std, generator=gen,
        )
        dnorm = _apply_rule(
            l23, cue_id=c, r_l23e=r, matched=matched, rule=args.rule,
            lr=args.lr, homeostasis_gamma=args.homeostasis_gamma,
            gain_min=args.gain_min, gain_max=args.gain_max,
        )
        rolling_dnorm[i % args.conv_window] = dnorm
        if i >= args.conv_window:
            mean_dnorm = float(rolling_dnorm.mean())
            if mean_dnorm < args.conv_eps:
                conv_persist_counter += 1
                if (conv_persist_counter >= args.conv_persist
                        and iterations_to_converge < 0):
                    iterations_to_converge = i + 1
            else:
                conv_persist_counter = 0

        if i % args.log_every == 0 or i == n_trials - 1:
            W = l23.W_q_gain.detach().cpu().numpy()
            snapshot = {
                "trial": i,
                "cue": c,
                "matched": matched,
                "dnorm": dnorm,
                "mean_gain_pref_cue0": float(
                    W[0, pref_mask_per_cue[0]].mean()
                ),
                "mean_gain_nonpref_cue0": float(
                    W[0, ~pref_mask_per_cue[0]].mean()
                ),
                "mean_gain_pref_cue1": float(
                    W[1, pref_mask_per_cue[1]].mean()
                ),
                "mean_gain_nonpref_cue1": float(
                    W[1, ~pref_mask_per_cue[1]].mean()
                ),
            }
            history.append(snapshot)

    # --- Final analysis --------------------------------------------------
    W_final = l23.W_q_gain.detach().cpu().numpy()
    mean_gain_pref = float((
        W_final[0, pref_mask_per_cue[0]].mean()
        + W_final[1, pref_mask_per_cue[1]].mean()
    ) / 2.0)
    mean_gain_nonpref = float((
        W_final[0, ~pref_mask_per_cue[0]].mean()
        + W_final[1, ~pref_mask_per_cue[1]].mean()
    ) / 2.0)
    target_pref = 1.0 - args.g0
    target_nonpref = 1.0
    matches_target = (
        abs(mean_gain_pref - target_pref) <= 0.1
        and abs(mean_gain_nonpref - target_nonpref) <= 0.1
    )
    rule_formula = {
        "rule1": "ΔW=-η·cue·r·sign(matched); clamp[gmin,gmax]",
        "rule2a": "rule1 + γ(1-W); clamp[gmin,gmax]",
    }[args.rule]

    print(
        f"step2_toy_learning: "
        f"final_mean_gain_pref={mean_gain_pref:.4f} "
        f"final_mean_gain_nonpref={mean_gain_nonpref:.4f} "
        f"target_pref={target_pref:.2f} "
        f"target_nonpref={target_nonpref:.2f} "
        f"matches_target={'T' if matches_target else 'F'} "
        f"iterations_to_converge={iterations_to_converge} "
        f"rule={args.rule}"
    )
    print(f"  rule_formula: {rule_formula}")
    print(
        f"  cue0: pref_gain={W_final[0, pref_mask_per_cue[0]].mean():.4f} "
        f"nonpref_gain={W_final[0, ~pref_mask_per_cue[0]].mean():.4f} "
        f"(n_pref={int(pref_mask_per_cue[0].sum())})"
    )
    print(
        f"  cue1: pref_gain={W_final[1, pref_mask_per_cue[1]].mean():.4f} "
        f"nonpref_gain={W_final[1, ~pref_mask_per_cue[1]].mean():.4f} "
        f"(n_pref={int(pref_mask_per_cue[1].sum())})"
    )
    print(
        f"  lr={args.lr} homeostasis_gamma={args.homeostasis_gamma} "
        f"gain_clamp=[{args.gain_min},{args.gain_max}] "
        f"n_trials={n_trials} validity={args.validity}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "rule": args.rule,
            "rule_formula": rule_formula,
            "ckpt": str(args.checkpoint),
            "seed": args.seed,
            "n_trials": n_trials,
            "validity": args.validity,
            "lr": args.lr,
            "homeostasis_gamma": args.homeostasis_gamma,
            "gain_clamp": [args.gain_min, args.gain_max],
            "g0": args.g0,
            "target_pref": target_pref,
            "target_nonpref": target_nonpref,
            "mean_gain_pref": mean_gain_pref,
            "mean_gain_nonpref": mean_gain_nonpref,
            "matches_target": bool(matches_target),
            "iterations_to_converge": iterations_to_converge,
            "per_cue_final": {
                "cue0": {
                    "pref_gain": float(
                        W_final[0, pref_mask_per_cue[0]].mean()
                    ),
                    "nonpref_gain": float(
                        W_final[0, ~pref_mask_per_cue[0]].mean()
                    ),
                    "n_pref": int(pref_mask_per_cue[0].sum()),
                },
                "cue1": {
                    "pref_gain": float(
                        W_final[1, pref_mask_per_cue[1]].mean()
                    ),
                    "nonpref_gain": float(
                        W_final[1, ~pref_mask_per_cue[1]].mean()
                    ),
                    "n_pref": int(pref_mask_per_cue[1].sum()),
                },
            },
            "history": history,
        }, f, indent=2)
    print(f"  JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
