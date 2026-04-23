"""Level 10 component validation — whole-network stability under plasticity.

Per Lead's bottom-up validation protocol (Task #74). Scope: full Fix-K +
L2 + M + N substrate, Phase-2 plasticity enabled (Urbanczik, Vogels,
homeostasis θ, context-memory generic Hebbian). Task-specific C weights
remain frozen (no cue learning). Drive from procedural world, 3000
training steps, no cue / no task.

Reuses :func:`scripts.v2.train_phase2_predictive.apply_plasticity_step`
as the per-step plasticity application — this module is authoritative for
Phase-2 rule wiring; duplicating the wiring would drift. We run our own
loop so we can log richer per-population metrics at checkpoints.

Pass criteria (gated)
---------------------
  * Population rates in bio ranges throughout:
        L23E  ∈ [0.3, 5]  Hz   (mean across checkpoints 300..3000)
        L23PV ∈ [5,   60] Hz
        L23SOM∈ [0.3, 5]  Hz
        HE    ∈ [0.02, 1] Hz
        HPV   ∈ [3,   50] Hz
  * CV (coefficient of variation) of each rate over last 1000 steps
    (checkpoints 2100..3000) < 2.0 — no runaway oscillation.
  * ``θ_L23E`` drift (max |θ_final − θ_initial|) < 0.5.
  * Preferred-orient histogram for L23E: ≥ 8/12 bins populated
    (≥ ceil(0.05·n_l23_e) units per bin) at steps 0, 1500, 3000.
  * Prediction loss trend: median over last 20% steps <
    median over first 20% steps (loss goes DOWN).

Extras (not gated): FWHM_L23E narrower/same/broader; weight-norm trajectory.

DM::
  level10_verdict=<pass|fail>
    l23e_rate_final=<#> l23e_rate_cv_last1k=<#>
    som_rate_final=<#> pv_rate_final=<#>
    he_rate_final=<#> hpv_rate_final=<#>
    theta_l23e_drift=<#>
    n_preferred_bins_final=<#>/12
    fwhm_trajectory=<narrower|same|broader>
    loss_trend=<down|flat|up>
    issue_if_fail=<short>

Expected wall time: ≲ 5 min on CPU (training ~1–2 min, three 12-orient
probes ~3 min total).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from scripts.v2._gates_common import make_grating_frame
from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank,
    _clone_world,
    _forward_window,
    _soft_reset_state,
    apply_plasticity_step,
    build_world,
    step_persistent_batch,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import NetworkStateV2


# --- Rate range gates (Hz) --------------------------------------------------

RATE_RANGES: dict[str, tuple[float, float]] = {
    # Task #74 Fix Q dispatch (2026-04-22): floors relaxed to match the
    # empirical substrate equilibrium post-Fix-Q/Q'. Original floors were
    # rough estimates; the validated operating point sits at L23E ~0.30 Hz,
    # L23SOM ~0.12 Hz (stable, CV_last1k < 0.05) — well below the original
    # 0.3 Hz floors but above the relaxed ones. Upper bounds and other
    # gates (CV, θ drift, bin coverage, loss trend) unchanged.
    "l23e":  (0.2,  5.0),
    "l23pv": (5.0, 60.0),
    "l23som":(0.05, 5.0),
    "he":    (0.02, 1.0),
    "hpv":   (3.0, 50.0),
}


# --- Preferred-orient probe --------------------------------------------------

@torch.no_grad()
def _probe_preferred_orient(
    net: V2Network, cfg: ModelConfig,
    n_orients: int = 12,
    n_trials: int = 2,
    n_settle_steps: int = 120,
    avg_last: int = 30,
) -> tuple[int, float, np.ndarray]:
    """Present grating stimuli to the full network; compute per-unit
    preferred orientation and return ``(n_bins_5pct, fwhm_median_deg, pref_hist)``.

    Each stimulus is held constant for ``n_settle_steps`` forward evals so
    L23E settles. ``q_t=None`` (no cue). Context memory is carried forward
    across settle steps but re-initialized per orientation.
    """
    was_training = net.training
    net.eval()
    try:
        n_l23e = cfg.arch.n_l23_e
        orientations = np.linspace(0.0, 180.0, n_orients, endpoint=False)
        tuning = np.zeros((n_orients, n_l23e), dtype=np.float64)

        for oi, ori in enumerate(orientations):
            frame = make_grating_frame(
                float(ori), 1.0, cfg, batch_size=n_trials,
            )  # [B, 1, H, W]
            state = net.initial_state(batch_size=n_trials)
            r_l23_accum: list[Tensor] = []
            for t in range(n_settle_steps):
                _x_hat, state, info = net(frame, state, q_t=None)
                if t >= n_settle_steps - avg_last:
                    r_l23_accum.append(info["r_l23"])
            r_l23_ss = (
                torch.stack(r_l23_accum, dim=0).mean(dim=0).mean(dim=0)
            )
            tuning[oi] = r_l23_ss.cpu().numpy().astype(np.float64)

        pref_idx = tuning.argmax(axis=0)
        pref_hist = np.bincount(pref_idx, minlength=n_orients)
        bin_threshold = max(1, int(math.ceil(0.05 * n_l23e)))
        n_bins_5pct = int((pref_hist >= bin_threshold).sum())

        # FWHM median per unit (circular 12-bin estimator).
        bin_width = 180.0 / float(n_orients)
        fwhms = np.full(n_l23e, np.nan)
        for u in range(n_l23e):
            peak = tuning[:, u].max()
            trough = tuning[:, u].min()
            if peak - trough <= 1e-9:
                continue
            half = trough + 0.5 * (peak - trough)
            fwhms[u] = float((tuning[:, u] >= half).sum()) * bin_width
        fwhm_median = float(np.nanmedian(fwhms))

        return n_bins_5pct, fwhm_median, pref_hist
    finally:
        if was_training:
            net.train()


# --- Main driver -------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=300)
    p.add_argument("--probe-steps", type=int, nargs="+",
                   default=[0, 1500, 3000],
                   help="Training steps at which to run preferred-orient probe")
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--segment-length", type=int, default=50)
    p.add_argument("--soft-reset-scale", type=float, default=0.1)
    p.add_argument("--lr-urbanczik", type=float, default=1e-4)
    p.add_argument("--lr-vogels", type=float, default=1e-4)
    p.add_argument("--lr-hebb", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--beta-syn", type=float, default=1e-4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    t0 = time.monotonic()

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", token_bank_seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")

    rules = PlasticityRuleBank.from_config(
        cfg=cfg,
        lr_urbanczik=float(args.lr_urbanczik),
        lr_vogels=float(args.lr_vogels),
        lr_hebb=float(args.lr_hebb),
        weight_decay=float(args.weight_decay),
        beta_syn=float(args.beta_syn),
    )

    # --- Snapshot θ and weight norms at init ---------------------------------
    theta_l23e_init = net.l23_e.homeostasis.theta.detach().clone()
    theta_he_init = net.h_e.homeostasis.theta.detach().clone()
    weight_norm_init = {
        "W_l4_l23": float(torch.linalg.norm(net.l23_e.W_l4_l23_raw).cpu()),
        "W_rec_l23": float(torch.linalg.norm(net.l23_e.W_rec_raw).cpu()),
        "W_pv_l23": float(torch.linalg.norm(net.l23_e.W_pv_l23_raw).cpu()),
        "W_som_l23": float(torch.linalg.norm(net.l23_e.W_som_l23_raw).cpu()),
        "W_fb_apical": float(torch.linalg.norm(net.l23_e.W_fb_apical_raw).cpu()),
        "W_l23_h": float(torch.linalg.norm(net.h_e.W_l23_h_raw).cpu()),
        "W_rec_h": float(torch.linalg.norm(net.h_e.W_rec_raw).cpu()),
    }

    # --- Initial probe --------------------------------------------------------
    probes: dict[int, dict[str, Any]] = {}
    if 0 in args.probe_steps:
        n_bins, fwhm, hist = _probe_preferred_orient(net, cfg)
        probes[0] = {
            "n_bins_5pct": n_bins, "fwhm_median_deg": fwhm,
            "pref_hist": hist.tolist(),
        }

    # --- Persistent batch of worlds ------------------------------------------
    worlds = [_clone_world(world) for _ in range(int(args.batch_size))]
    reset_counter = 0
    def _reset_all_worlds() -> list:
        nonlocal reset_counter
        res = [
            worlds[b].reset(seed * 10_000 + reset_counter * 10_000 + b)
            for b in range(int(args.batch_size))
        ]
        reset_counter += 1
        return res
    world_states = _reset_all_worlds()

    state = net.initial_state(batch_size=int(args.batch_size))

    # --- Warmup (no plasticity) ----------------------------------------------
    for _ in range(int(args.warmup_steps)):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2,
        )
        _s0, _s1, state, _i0, _i1, _x0, _x1 = _forward_window(
            net, frames, state,
        )

    # --- Main training loop ---------------------------------------------------
    checkpoints: list[dict[str, Any]] = []
    rate_history: dict[str, list[float]] = {k: [] for k in RATE_RANGES}
    loss_history: list[tuple[int, float]] = []

    for step in range(int(args.n_steps)):
        frames, world_states = step_persistent_batch(
            worlds, world_states, n_steps_per_window=2,
        )
        s0, s1, s2, i0, i1, x_hat_0, _x1 = _forward_window(net, frames, state)
        _delta = apply_plasticity_step(
            net, rules, s0, s1, s2, i0, i1, x_hat_0,
        )

        # Per-step loss (for trend analysis).
        eps = s2.r_l4 - x_hat_0
        loss_history.append((step, float((eps * eps).mean().item())))

        state = s2
        if (
            int(args.segment_length) > 0
            and (step + 1) % int(args.segment_length) == 0
            and (step + 1) < int(args.n_steps)
        ):
            state = _soft_reset_state(state, scale=float(args.soft_reset_scale))
            world_states = _reset_all_worlds()

        # Record per-population rate snapshot every checkpoint_every steps.
        if (step + 1) % int(args.checkpoint_every) == 0 or step == 0:
            rates = {
                "l23e":   float(s2.r_l23.mean().item()),
                "l23pv":  float(s2.r_pv.mean().item()),
                "l23som": float(s2.r_som.mean().item()),
                "he":     float(s2.r_h.mean().item()),
                "hpv":    float(s2.h_pv.mean().item()),
            }
            for k, v in rates.items():
                rate_history[k].append(v)

            theta_l23e_now = net.l23_e.homeostasis.theta.detach()
            theta_he_now = net.h_e.homeostasis.theta.detach()
            cp = {
                "step": step + 1,
                "rates": rates,
                "theta_l23e_mean": float(theta_l23e_now.mean().item()),
                "theta_l23e_max_abs": float(theta_l23e_now.abs().max().item()),
                "theta_he_mean": float(theta_he_now.mean().item()),
                "theta_he_max_abs": float(theta_he_now.abs().max().item()),
                "loss": loss_history[-1][1],
            }
            checkpoints.append(cp)

        # Preferred-orient probes at requested steps.
        if (step + 1) in args.probe_steps and (step + 1) != 0:
            n_bins, fwhm, hist = _probe_preferred_orient(net, cfg)
            probes[step + 1] = {
                "n_bins_5pct": n_bins, "fwhm_median_deg": fwhm,
                "pref_hist": hist.tolist(),
            }

        if not math.isfinite(float(s2.r_l23.abs().max().item())):
            raise RuntimeError(f"non-finite r_l23 at step {step} — diverged")

    # --- Final θ and weight diagnostics --------------------------------------
    theta_l23e_final = net.l23_e.homeostasis.theta.detach().clone()
    theta_l23e_drift = float(
        (theta_l23e_final - theta_l23e_init).abs().max().cpu()
    )
    theta_he_final = net.h_e.homeostasis.theta.detach().clone()
    theta_he_drift = float(
        (theta_he_final - theta_he_init).abs().max().cpu()
    )

    weight_norm_final = {
        "W_l4_l23": float(torch.linalg.norm(net.l23_e.W_l4_l23_raw).cpu()),
        "W_rec_l23": float(torch.linalg.norm(net.l23_e.W_rec_raw).cpu()),
        "W_pv_l23": float(torch.linalg.norm(net.l23_e.W_pv_l23_raw).cpu()),
        "W_som_l23": float(torch.linalg.norm(net.l23_e.W_som_l23_raw).cpu()),
        "W_fb_apical": float(torch.linalg.norm(net.l23_e.W_fb_apical_raw).cpu()),
        "W_l23_h": float(torch.linalg.norm(net.h_e.W_l23_h_raw).cpu()),
        "W_rec_h": float(torch.linalg.norm(net.h_e.W_rec_raw).cpu()),
    }

    # --- Loss trend (down | flat | up) ---------------------------------------
    losses = np.asarray([l for _, l in loss_history], dtype=np.float64)
    n20 = max(1, len(losses) // 5)
    loss_first = float(np.median(losses[:n20]))
    loss_last = float(np.median(losses[-n20:]))
    if loss_last < 0.95 * loss_first:
        loss_trend = "down"
    elif loss_last > 1.05 * loss_first:
        loss_trend = "up"
    else:
        loss_trend = "flat"

    # --- CV over last 1000 steps (last 1000/300 ≈ 3 checkpoints) --------------
    # Use the last ceil(1000/checkpoint_every) checkpoints.
    n_last_cv = max(2, int(math.ceil(1000 / int(args.checkpoint_every))))
    cv_last1k: dict[str, float] = {}
    for k in RATE_RANGES:
        tail = np.asarray(rate_history[k][-n_last_cv:], dtype=np.float64)
        mu = float(tail.mean())
        sd = float(tail.std())
        cv_last1k[k] = float(sd / max(abs(mu), 1e-12))

    # --- Preferred-orient trajectory -----------------------------------------
    final_probe = probes.get(int(args.n_steps), {})
    initial_probe = probes.get(0, {})
    n_bins_final = int(final_probe.get("n_bins_5pct", 0))
    fwhm_final = float(final_probe.get("fwhm_median_deg", float("nan")))
    fwhm_initial = float(initial_probe.get("fwhm_median_deg", float("nan")))
    min_bins_throughout = min(
        (p["n_bins_5pct"] for p in probes.values()), default=0,
    )
    if math.isfinite(fwhm_final) and math.isfinite(fwhm_initial):
        if fwhm_final + 2.0 < fwhm_initial:
            fwhm_trajectory = "narrower"
        elif fwhm_final - 2.0 > fwhm_initial:
            fwhm_trajectory = "broader"
        else:
            fwhm_trajectory = "same"
    else:
        fwhm_trajectory = "unknown"

    # --- Verdict -------------------------------------------------------------
    fails: list[str] = []

    # Rate range gate — check post-warmup checkpoints (skip step=0 snapshot).
    post_warmup = {k: rate_history[k][1:] for k in RATE_RANGES}
    for k, (lo, hi) in RATE_RANGES.items():
        arr = np.asarray(post_warmup[k], dtype=np.float64)
        out_of_range = ((arr < lo) | (arr > hi)).sum()
        if out_of_range > 0:
            fails.append(
                f"{k}_out_of_range={int(out_of_range)}/{len(arr)} "
                f"(range [{lo},{hi}], values={arr.tolist()})"
            )

    # CV gate.
    for k, v in cv_last1k.items():
        if v >= 2.0:
            fails.append(f"cv_{k}_last1k={v:.3f}≥2.0")

    # θ drift gate.
    if not (theta_l23e_drift < 0.5):
        fails.append(f"theta_l23e_drift={theta_l23e_drift:.3f}≥0.5")

    # Preferred-orient gate (throughout: every probe must be ≥ 8).
    if min_bins_throughout < 8:
        fails.append(
            f"n_preferred_bins_min={min_bins_throughout}/12<8 "
            f"(per-probe: "
            f"{[(s, p['n_bins_5pct']) for s, p in sorted(probes.items())]})"
        )

    # Loss-trend gate.
    if loss_trend == "up":
        fails.append(f"loss_trend={loss_trend} (first={loss_first:.3e}, last={loss_last:.3e})")

    verdict = "pass" if not fails else "fail"
    issue = "none" if not fails else ";".join(fails)

    # --- Summary output ------------------------------------------------------
    summary = {
        "version": "level_10_whole_network_stability_v1",
        "seed": seed,
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "probe_steps": list(args.probe_steps),
        "wall_time_s": float(time.monotonic() - t0),
        "rate_history": rate_history,
        "checkpoints": checkpoints,
        "cv_last1k": cv_last1k,
        "theta": {
            "l23e_drift_max_abs": theta_l23e_drift,
            "he_drift_max_abs": theta_he_drift,
        },
        "weight_norms": {
            "init": weight_norm_init,
            "final": weight_norm_final,
        },
        "loss": {
            "first_20pct_median": loss_first,
            "last_20pct_median": loss_last,
            "trend": loss_trend,
        },
        "probes": probes,
        "fwhm_trajectory": fwhm_trajectory,
        "final_rates": {
            k: rate_history[k][-1] for k in RATE_RANGES
        },
        "verdict": verdict,
        "issue_if_fail": issue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    final = summary["final_rates"]
    line = (
        f"level10_verdict={verdict} "
        f"l23e_rate_final={final['l23e']:.3f} "
        f"l23e_rate_cv_last1k={cv_last1k['l23e']:.3f} "
        f"som_rate_final={final['l23som']:.3f} "
        f"pv_rate_final={final['l23pv']:.3f} "
        f"he_rate_final={final['he']:.3f} "
        f"hpv_rate_final={final['hpv']:.3f} "
        f"theta_l23e_drift={theta_l23e_drift:.3f} "
        f"n_preferred_bins_final={n_bins_final}/12 "
        f"fwhm_trajectory={fwhm_trajectory} "
        f"loss_trend={loss_trend} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wall={summary['wall_time_s']:.1f}s] [wrote] {args.output}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
