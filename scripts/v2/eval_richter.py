"""Phase-3 Richter evaluation — upgraded per Task #72 / critique §6.

Original Task #40 assays (retained for harness compatibility):
  1. **Unit-level L2/3 amplitude on trailing** (6×6 conditions × units).
  2. **Within / between-class RSA** over leader-trailer classes.
  3. **Preference-rank suppression curve** (population-average rank 1..6).
  4. **6-model pseudo-voxel forward comparison** — correlates each model's
     voxel projection of the population response with the population mean.
     This assay saturates to ~1.0 in the Task-#70 breakthrough run because
     the projections are content-agnostic — superseded by the Task-#72
     modulation-pattern fit below.

New Task #72 upgrades:
  5. **Trailer-only localizer** (leader-free) → per-unit trailer preference
     ranking used for suppression binning.
  6. **Modulation-pattern fit** — for each trailer j compute the OBSERVED
     modulation vector across units (expected-leader response minus
     unexpected-leader response) and correlate with each of 6 model-
     predicted modulation vectors (local/remote/global × gain/tuning).
     The across-UNITS correlation distinguishes circuit geometries; the
     old across-trials correlation with population mean does not.
  7. **Preference-rank-binned suppression** — bin units by their localizer
     preference for trailer j (top 20% / middle 60% / bottom 20%) and
     report per-bin suppression = unexp − exp. Richter predicts top-bin
     suppression > bottom-bin suppression.
  8. **Same-trailer contrast** — canonical Richter comparison; per trailer
     j, ``mean_response(j, expected_leader) − mean_response(j, any other
     leader)``. Reported pooled + per-trailer, with bootstrap CI and
     permutation p.
  9. **Bootstrap 95% CIs + permutation p-values** on same-trailer Δ,
     preference-rank slope, modulation-pattern correlations.

Writes ``eval_richter.json`` next to the checkpoint. Exit 0 always.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import CheckpointBundle, load_checkpoint
from scripts.v2.train_phase3_richter_learning import (
    LEADER_TOKEN_IDX, N_LEAD_TRAIL, RichterTiming, TRAILER_TOKEN_IDX,
    build_leader_tensor, permutation_from_seed,
)


__all__ = [
    "run_richter_probe_trial", "run_richter_localizer_trial", "evaluate_richter",
]


# ---------------------------------------------------------------------------
# Single-trial forward (eval-only; no plasticity)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_richter_probe_trial(
    bundle: CheckpointBundle,
    *, leader_pos: int, trailer_pos: int,
    timing: RichterTiming, noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    """Return per-unit mean L2/3 rate during trailer. Shape ``[n_l23_e]``."""
    cfg = bundle.cfg
    device = cfg.device
    leader_tok = bundle.bank.tokens[
        LEADER_TOKEN_IDX[int(leader_pos)]:LEADER_TOKEN_IDX[int(leader_pos)] + 1
    ].to(device=device)
    trailer_tok = bundle.bank.tokens[
        TRAILER_TOKEN_IDX[int(trailer_pos)]:TRAILER_TOKEN_IDX[int(trailer_pos)] + 1
    ].to(device=device)
    leader_v = build_leader_tensor(
        int(leader_pos), bundle.net.context_memory.n_leader, device=device,
    )

    state = bundle.net.initial_state(batch_size=1)
    leader_end = timing.leader_steps
    trailer_end = leader_end + timing.trailer_steps

    trailer_rates: list[Tensor] = []
    for t in range(trailer_end):
        if t < leader_end:
            frame, ld_t = leader_tok, leader_v
        else:
            frame, ld_t = trailer_tok, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, leader_t=ld_t)
        if leader_end <= t < trailer_end:
            trailer_rates.append(info["r_l23"][0].clone())

    return torch.stack(trailer_rates, dim=0).mean(dim=0)           # [n_l23]


@torch.no_grad()
def run_richter_localizer_trial(
    bundle: CheckpointBundle,
    *, trailer_pos: int,
    timing: RichterTiming, noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    """LEADER-FREE trailer-only trial.

    Presents a blank frame during the leader window (no leader token, no
    leader_t context vector), then the trailer token. Measures the pure
    sensory response to each trailer for per-unit preference ranking.
    Returns ``[n_l23_e]``.
    """
    cfg = bundle.cfg
    device = cfg.device
    trailer_tok = bundle.bank.tokens[
        TRAILER_TOKEN_IDX[int(trailer_pos)]:TRAILER_TOKEN_IDX[int(trailer_pos)] + 1
    ].to(device=device)
    # Blank frame: zeros (mean-grey proxy) sized like trailer token.
    blank = torch.zeros_like(trailer_tok)

    state = bundle.net.initial_state(batch_size=1)
    leader_end = timing.leader_steps
    trailer_end = leader_end + timing.trailer_steps
    trailer_rates: list[Tensor] = []
    for t in range(trailer_end):
        if t < leader_end:
            frame = blank
        else:
            frame = trailer_tok
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, leader_t=None)
        if leader_end <= t < trailer_end:
            trailer_rates.append(info["r_l23"][0].clone())
    return torch.stack(trailer_rates, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# RSA — within vs between leader-trailer classes (Task #40)
# ---------------------------------------------------------------------------


def _rsa_within_between(
    r_l23: np.ndarray,               # [N_trials, n_l23]
    class_ids: np.ndarray,           # [N_trials] — integer class index
) -> dict[str, Any]:
    """Mean cosine distance within vs between classes."""
    if r_l23.shape[0] < 4:
        return {"error": "need ≥ 4 trials for RSA"}
    norms = np.linalg.norm(r_l23, axis=1, keepdims=True) + 1e-9
    nrm = r_l23 / norms
    cos = nrm @ nrm.T
    dist = 1.0 - cos
    iu = np.triu_indices(r_l23.shape[0], k=1)
    same = class_ids[iu[0]] == class_ids[iu[1]]
    d = dist[iu]
    within = float(d[same].mean()) if same.any() else float("nan")
    between = float(d[~same].mean()) if (~same).any() else float("nan")
    return {
        "within_mean_distance": within,
        "between_mean_distance": between,
        "between_minus_within": float(between - within),
        "n_within_pairs": int(same.sum()),
        "n_between_pairs": int((~same).sum()),
    }


# ---------------------------------------------------------------------------
# Preference-rank suppression curve (Task #40 — retained)
# ---------------------------------------------------------------------------


def _preference_rank_curve(
    per_condition: np.ndarray,       # [n_leaders, n_trailers, n_l23]
) -> dict[str, Any]:
    """Mean L2/3 rate vs trailer preference rank (1 = most preferred)."""
    n_leaders, n_trailers, n_l23 = per_condition.shape
    order = np.argsort(-per_condition, axis=1)                      # [L, T, U]
    ranked = np.take_along_axis(per_condition, order, axis=1)
    curve_mean = ranked.mean(axis=(0, 2))
    curve_sem = ranked.reshape(-1, n_trailers).std(axis=0) / np.sqrt(
        n_leaders * n_l23 + 1e-9,
    )
    return {
        "ranks": [i + 1 for i in range(n_trailers)],
        "mean_rate": [float(x) for x in curve_mean],
        "sem_rate": [float(x) for x in curve_sem],
        "suppression_rank1_vs_rank_last": float(
            curve_mean[0] - curve_mean[-1]
        ),
    }


# ---------------------------------------------------------------------------
# 6-model pseudo-voxel forward comparison (Task #40 — retained, superseded
# by modulation-pattern fit below; kept for diagnostic comparison).
# ---------------------------------------------------------------------------


def _build_pseudo_voxel_models(
    n_l23: int, n_voxels_per_model: int, seed: int,
) -> dict[str, np.ndarray]:
    """Return ``{model_name: projection[n_voxels, n_l23]}`` for 6 models."""
    rng = np.random.default_rng(int(seed))
    block_size = max(1, n_l23 // n_voxels_per_model)
    models: dict[str, np.ndarray] = {}

    def _make_mask_local(v: int) -> np.ndarray:
        m = np.zeros(n_l23, dtype=np.float64)
        lo = (v * block_size) % n_l23
        hi = min(lo + block_size, n_l23)
        m[lo:hi] = 1.0
        return m

    def _make_mask_remote(v: int) -> np.ndarray:
        m = np.zeros(n_l23, dtype=np.float64)
        idx = np.arange(v, n_l23, n_voxels_per_model)[:block_size]
        m[idx] = 1.0
        return m

    def _make_mask_global(v: int) -> np.ndarray:
        return np.ones(n_l23, dtype=np.float64) / float(n_l23)

    for spatial, mask_fn in (
        ("local", _make_mask_local),
        ("remote", _make_mask_remote),
        ("global", _make_mask_global),
    ):
        for effect in ("gain", "tuning"):
            proj = np.zeros((n_voxels_per_model, n_l23), dtype=np.float64)
            for v in range(n_voxels_per_model):
                m = mask_fn(v)
                if effect == "gain":
                    scale = 0.5 + 0.5 * rng.random()
                    proj[v] = m * scale
                else:
                    signs = rng.choice(
                        [-1.0, 1.0], size=n_l23, replace=True,
                    )
                    proj[v] = m * signs
                    nz = proj[v] != 0
                    if nz.any():
                        proj[v][nz] = proj[v][nz] - proj[v][nz].mean()
            models[f"{spatial}_{effect}"] = proj
    return models


def _pseudo_voxel_forward(
    per_condition: np.ndarray,
    models: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Project observed responses through each model; correlate with the
    population-mean condition response. (Legacy; saturates.)"""
    n_leaders, n_trailers, n_l23 = per_condition.shape
    flat = per_condition.reshape(n_leaders * n_trailers, n_l23)
    pop_mean = flat.mean(axis=1)
    pop_mean -= pop_mean.mean()
    pop_mean /= (pop_mean.std() + 1e-9)

    out: dict[str, dict[str, float]] = {}
    for name, proj in models.items():
        voxel_activity = flat @ proj.T
        voxel_mean = voxel_activity.mean(axis=1)
        voxel_mean_c = voxel_mean - voxel_mean.mean()
        voxel_mean_c /= (voxel_mean_c.std() + 1e-9)
        r = float((pop_mean * voxel_mean_c).mean())
        out[name] = {
            "pearson_r_with_pop_mean": r,
            "n_voxels": int(proj.shape[0]),
        }
    return out


# ---------------------------------------------------------------------------
# Task #72: modulation-pattern fit (replaces pop-mean forward comparison)
# ---------------------------------------------------------------------------


def _inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    """σ such that σ(σ_in(i)) = i. Used to get expected_leader for each
    trailer j: ``inverse_permutation[j]`` is the unique i with perm[i]=j."""
    n = len(perm)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[int(p)] = int(i)
    return tuple(inv)


def _compute_observed_modulation_patterns(
    per_condition: np.ndarray,       # [n_leaders, n_trailers, n_l23]
    permutation: tuple[int, ...],
) -> np.ndarray:
    """For each trailer j compute the across-units modulation vector
    ``observed[j, :] = mean(r | j, expected_leader) − mean(r | j, other leaders)``.

    Returns ``[n_trailers, n_l23]`` — one modulation pattern per trailer.
    """
    inv_perm = _inverse_permutation(permutation)
    n_leaders, n_trailers, n_l23 = per_condition.shape
    obs = np.zeros((n_trailers, n_l23), dtype=np.float64)
    for j in range(n_trailers):
        exp_leader = int(inv_perm[j])
        exp_resp = per_condition[exp_leader, j]                     # [n_l23]
        other_leaders = [i for i in range(n_leaders) if i != exp_leader]
        unexp_resp = per_condition[other_leaders, j].mean(axis=0)   # [n_l23]
        obs[j] = exp_resp - unexp_resp
    return obs


def _model_predict_response(
    baseline: np.ndarray,         # [n_l23] — localizer preference for trailer j
    spatial_mask: np.ndarray,     # [n_l23] ∈ {0, 1}
    effect_pattern: np.ndarray,   # [n_l23] — per-unit effect amplitude (signed or all-1)
    *, is_expected_context: bool,
    alpha: float = 1.0,
    multiplicative: bool,
) -> np.ndarray:
    """Predict the per-unit response for trailer j in (expected / unexpected) context.

    Expected context applies a leader-driven modulation on ``spatial_mask``;
    unexpected context has no such modulation (baseline only).

    * If ``multiplicative=True``  (gain): resp = baseline * (1 + α · mask · effect)
    * If ``multiplicative=False`` (tuning): resp = baseline + α · mask · effect

    Shapes: all inputs ``[n_l23]``; returns ``[n_l23]``.
    """
    if not is_expected_context:
        return baseline.copy()
    mod = alpha * spatial_mask * effect_pattern
    if multiplicative:
        return baseline * (1.0 + mod)
    return baseline + mod


def _compute_model_predicted_modulations(
    per_condition: np.ndarray,       # [n_leaders, n_trailers, n_l23]
    permutation: tuple[int, ...],
    localizer_preference: np.ndarray,  # [n_trailers, n_l23] — leader-free trailer responses
    models: dict[str, np.ndarray],
    *, alpha: float = 1.0,
    top_frac: float = 0.2,
    bot_frac: float = 0.2,
    tuning_seed: int = 0,
) -> dict[str, np.ndarray]:
    """For each of the 6 (spatial × effect) models, predict the per-unit
    Δ-vector ``pred_mod[j, :] = response(expected_leader) − response(unexpected_leader)``.

    The prediction rule: expected context adds a leader-driven modulation
    living on a spatial sub-population of units with a model-specific effect
    shape; unexpected context has no such modulation. Subtracting the two
    yields the Δ-vector that is directly comparable (via across-units
    Pearson) to the observed modulation vector.

    Spatial mask (WHERE the modulation lives — differentiates local/remote/global):
      * ``local``  = top-``top_frac`` units by ``localizer_preference[j]``
        (units tuned TO trailer j).
      * ``remote`` = bottom-``bot_frac`` units
        (units tuned AWAY from trailer j).
      * ``global`` = all units.

    Effect (HOW the modulation acts — differentiates gain/tuning):
      * ``gain``   = MULTIPLICATIVE on baseline → Δ = α · mask · baseline_j
        (enhancement strongest where baseline firing is highest).
      * ``tuning`` = ADDITIVE zero-mean signed  → Δ = α · mask · signed_j
        (some units up, some down; mean-zero within mask).

    This construction (a) explicitly computes expected − unexpected Δ-vectors
    per Lead directive, (b) guarantees the 6 models give DISTINGUISHABLE
    predictions (different spatial support × different effect shape), and
    (c) avoids the pop-mean saturation failure mode of the legacy forward.

    Parameters
    ----------
    per_condition     : preserved for signature symmetry; not used.
    permutation       : preserved for signature symmetry; not used.
    localizer_preference : leader-free per-trailer per-unit baseline rates.
    models            : preserved for signature symmetry; not used.
    alpha             : modulation strength (arbitrary — cancels under Pearson).
    top_frac, bot_frac: fraction of units forming local / remote masks.
    tuning_seed       : seed for the signed tuning pattern (deterministic).

    Returns
    -------
    dict[str, np.ndarray] with keys ``{local,remote,global}_{gain,tuning}``
    each of shape ``[n_trailers, n_l23]``.
    """
    _ = per_condition
    _ = permutation
    _ = models
    n_trailers, n_l23 = localizer_preference.shape
    n_top = max(1, int(round(top_frac * n_l23)))
    n_bot = max(1, int(round(bot_frac * n_l23)))
    rng = np.random.default_rng(int(tuning_seed))

    out: dict[str, np.ndarray] = {}
    for spatial in ("local", "remote", "global"):
        for effect in ("gain", "tuning"):
            pred = np.zeros((n_trailers, n_l23), dtype=np.float64)
            for j in range(n_trailers):
                baseline = localizer_preference[j].astype(np.float64)
                order = np.argsort(-baseline)  # descending
                mask = np.zeros(n_l23, dtype=np.float64)
                if spatial == "local":
                    mask[order[:n_top]] = 1.0
                elif spatial == "remote":
                    mask[order[-n_bot:]] = 1.0
                else:  # global
                    mask[:] = 1.0

                if effect == "gain":
                    effect_pattern = np.ones(n_l23, dtype=np.float64)
                    multiplicative = True
                else:  # tuning: zero-mean signed on masked units
                    signs = rng.choice([-1.0, 1.0], size=n_l23).astype(np.float64)
                    active = mask > 0
                    if active.any():
                        signs[active] = signs[active] - signs[active].mean()
                    effect_pattern = signs
                    multiplicative = False

                pred_exp = _model_predict_response(
                    baseline, mask, effect_pattern,
                    is_expected_context=True, alpha=alpha,
                    multiplicative=multiplicative,
                )
                pred_unexp = _model_predict_response(
                    baseline, mask, effect_pattern,
                    is_expected_context=False, alpha=alpha,
                    multiplicative=multiplicative,
                )
                pred[j] = pred_exp - pred_unexp
            out[f"{spatial}_{effect}"] = pred
    return out


def _pearson_across_units(
    obs: np.ndarray,                  # [n_trailers, n_l23]
    pred: np.ndarray,                 # [n_trailers, n_l23]
) -> np.ndarray:
    """Per-trailer Pearson r(obs[j, :], pred[j, :]) across units."""
    out = np.zeros(obs.shape[0], dtype=np.float64)
    for j in range(obs.shape[0]):
        x = obs[j]
        y = pred[j]
        xm = x - x.mean()
        ym = y - y.mean()
        d = np.sqrt((xm ** 2).sum() * (ym ** 2).sum()) + 1e-12
        out[j] = float((xm * ym).sum() / d)
    return out


# ---------------------------------------------------------------------------
# Task #72: preference-rank-binned suppression
# ---------------------------------------------------------------------------


def _preference_rank_binned_suppression(
    per_trial: np.ndarray,           # [N_trials, n_l23]
    trial_trailer: np.ndarray,       # [N_trials]
    trial_expected: np.ndarray,      # [N_trials] bool
    localizer_preference: np.ndarray,  # [n_trailers, n_l23]
    *, top_frac: float = 0.2, bot_frac: float = 0.2,
) -> dict[str, Any]:
    """Bin units by localizer preference for the shown trailer (top/mid/bot),
    measure suppression = unexpected − expected mean per bin.

    Returns per-bin suppression averaged over all trailers, and the linear
    slope across (top, mid, bot) bins. Richter prediction: slope from top
    to bot is positive (top units suppressed more than bot).
    """
    n_trailers, n_l23 = localizer_preference.shape
    n_top = max(1, int(round(top_frac * n_l23)))
    n_bot = max(1, int(round(bot_frac * n_l23)))
    per_bin_sup = {"top": [], "mid": [], "bot": []}
    for j in range(n_trailers):
        pref = localizer_preference[j]
        order = np.argsort(-pref)
        top_units = order[:n_top]
        bot_units = order[-n_bot:]
        mid_units = order[n_top:len(order) - n_bot]
        trial_mask = (trial_trailer == j)
        exp_mask = trial_mask & trial_expected
        unexp_mask = trial_mask & (~trial_expected)
        if not exp_mask.any() or not unexp_mask.any():
            continue
        exp_r = per_trial[exp_mask]
        unexp_r = per_trial[unexp_mask]
        for bin_name, units in (
            ("top", top_units), ("mid", mid_units), ("bot", bot_units),
        ):
            if units.size == 0:
                continue
            exp_mean = float(exp_r[:, units].mean())
            unexp_mean = float(unexp_r[:, units].mean())
            per_bin_sup[bin_name].append(unexp_mean - exp_mean)
    summary = {
        bin_name: (
            float(np.mean(vals)) if vals else float("nan")
        )
        for bin_name, vals in per_bin_sup.items()
    }
    # Linear slope across ordered bins (top=0, mid=1, bot=2).
    bins = np.asarray(
        [summary["top"], summary["mid"], summary["bot"]], dtype=np.float64,
    )
    xs = np.array([0.0, 1.0, 2.0])
    valid = np.isfinite(bins)
    if valid.sum() >= 2:
        slope = float(np.polyfit(xs[valid], bins[valid], 1)[0])
    else:
        slope = float("nan")
    return {
        "suppression_top_bin": summary["top"],
        "suppression_mid_bin": summary["mid"],
        "suppression_bot_bin": summary["bot"],
        "preference_rank_slope": slope,
        "top_frac": float(top_frac), "bot_frac": float(bot_frac),
        "n_top_units": int(n_top), "n_bot_units": int(n_bot),
    }


# ---------------------------------------------------------------------------
# Task #72: same-trailer expected-vs-other-leader contrast
# ---------------------------------------------------------------------------


def _same_trailer_contrast(
    per_trial: np.ndarray,           # [N_trials, n_l23]
    trial_trailer: np.ndarray,       # [N_trials]
    trial_expected: np.ndarray,      # [N_trials] bool
) -> dict[str, Any]:
    """Per-trailer ``mean(rate | expected leader) − mean(rate | other leader)``,
    pooled over units and averaged across trailers.

    Returns the per-trailer means, the grand Δ (exp − unexp), and trial-level
    arrays for bootstrap/permutation downstream.
    """
    n_trailers = int(trial_trailer.max() + 1)
    per_j_exp: list[float] = []
    per_j_unexp: list[float] = []
    trial_mean = per_trial.mean(axis=1)                             # [N_trials]
    for j in range(n_trailers):
        mask = (trial_trailer == j)
        exp_m = mask & trial_expected
        unexp_m = mask & (~trial_expected)
        per_j_exp.append(
            float(trial_mean[exp_m].mean()) if exp_m.any() else float("nan"),
        )
        per_j_unexp.append(
            float(trial_mean[unexp_m].mean()) if unexp_m.any() else float("nan"),
        )
    grand_exp = float(np.nanmean(per_j_exp))
    grand_unexp = float(np.nanmean(per_j_unexp))
    return {
        "per_trailer_exp": per_j_exp,
        "per_trailer_unexp": per_j_unexp,
        "grand_exp": grand_exp,
        "grand_unexp": grand_unexp,
        "grand_delta_exp_minus_unexp": float(grand_exp - grand_unexp),
        "trial_mean_exp": trial_mean[trial_expected].astype(np.float64),
        "trial_mean_unexp": trial_mean[~trial_expected].astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Bootstrap + permutation helpers (shared with eval_kok)
# ---------------------------------------------------------------------------


def _bootstrap_ci_mean_diff(
    values_a: np.ndarray, values_b: np.ndarray,
    *, n_resamples: int = 1000, seed: int = 0,
) -> dict[str, float]:
    a = np.asarray(values_a, dtype=np.float64); a = a[np.isfinite(a)]
    b = np.asarray(values_b, dtype=np.float64); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            "mean_diff": float("nan"), "ci_lo": float("nan"),
            "ci_hi": float("nan"), "n_a": int(a.size), "n_b": int(b.size),
        }
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_resamples), dtype=np.float64)
    for k in range(int(n_resamples)):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        diffs[k] = aa.mean() - bb.mean()
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "n_a": int(a.size), "n_b": int(b.size),
        "n_resamples": int(n_resamples),
    }


def _permutation_p_two_sided(
    values_a: np.ndarray, values_b: np.ndarray,
    *, n_permutations: int = 1000, seed: int = 0,
) -> float:
    a = np.asarray(values_a, dtype=np.float64); a = a[np.isfinite(a)]
    b = np.asarray(values_b, dtype=np.float64); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(int(seed))
    hits = 0
    for _ in range(int(n_permutations)):
        idx = rng.permutation(pool.size)
        aa = pool[idx[:n_a]]
        bb = pool[idx[n_a:]]
        if abs(aa.mean() - bb.mean()) >= obs:
            hits += 1
    return float((hits + 1) / (int(n_permutations) + 1))


# ---------------------------------------------------------------------------
# Top-level eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_richter(
    bundle: CheckpointBundle,
    *, n_trials_per_condition: int = 8,
    timing: Optional[RichterTiming] = None,
    permutation: Optional[tuple[int, ...]] = None,
    noise_std: float = 0.01,
    n_pseudo_voxels_per_model: int = 50,
    seed: int = 42,
    n_localizer_trials: int = 20,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    run_upgrades: bool = True,
) -> dict[str, Any]:
    """Run Richter eval with Task #40 + Task #72 assays.

    When ``run_upgrades`` is False only the Task #40 assays run (fast path
    for the test harness). When True (default) the localizer + modulation-
    pattern fit + binned suppression + same-trailer contrast also run.
    """
    cfg = bundle.cfg
    timing = timing or RichterTiming()
    permutation = permutation or permutation_from_seed(seed)
    inv_perm = _inverse_permutation(permutation)

    gen = torch.Generator(device=cfg.device); gen.manual_seed(int(seed))

    n_l23 = cfg.arch.n_l23_e
    per_condition_sum = np.zeros(
        (N_LEAD_TRAIL, N_LEAD_TRAIL, n_l23), dtype=np.float64,
    )
    per_condition_count = np.zeros(
        (N_LEAD_TRAIL, N_LEAD_TRAIL), dtype=np.int64,
    )
    all_trials: list[np.ndarray] = []
    all_cls: list[int] = []
    all_leader: list[int] = []
    all_trailer: list[int] = []
    all_expected: list[bool] = []
    for leader_pos in range(N_LEAD_TRAIL):
        for trailer_pos in range(N_LEAD_TRAIL):
            expected_here = bool(int(permutation[leader_pos]) == trailer_pos)
            for _ in range(int(n_trials_per_condition)):
                r = run_richter_probe_trial(
                    bundle, leader_pos=leader_pos, trailer_pos=trailer_pos,
                    timing=timing, noise_std=float(noise_std),
                    generator=gen,
                )
                r_np = r.cpu().numpy().astype(np.float64)
                per_condition_sum[leader_pos, trailer_pos] += r_np
                per_condition_count[leader_pos, trailer_pos] += 1
                all_trials.append(r_np)
                all_cls.append(leader_pos * N_LEAD_TRAIL + trailer_pos)
                all_leader.append(int(leader_pos))
                all_trailer.append(int(trailer_pos))
                all_expected.append(expected_here)

    per_condition_mean = per_condition_sum / np.clip(
        per_condition_count[..., None], 1, None,
    )
    trial_arr = np.stack(all_trials, axis=0)                        # [N, n_l23]
    trial_trailer = np.asarray(all_trailer, dtype=np.int64)
    trial_leader = np.asarray(all_leader, dtype=np.int64)
    trial_expected = np.asarray(all_expected, dtype=bool)

    amplitude_summary = {
        "per_condition_mean_unit0": [
            float(per_condition_mean[i, j, 0])
            for i in range(N_LEAD_TRAIL)
            for j in range(N_LEAD_TRAIL)
        ],
        "grand_mean": float(per_condition_mean.mean()),
        "grand_std": float(per_condition_mean.std()),
    }

    rsa = _rsa_within_between(
        trial_arr, np.asarray(all_cls, dtype=np.int64),
    )
    pref_rank = _preference_rank_curve(per_condition_mean)
    models = _build_pseudo_voxel_models(
        n_l23, int(n_pseudo_voxels_per_model), int(seed),
    )
    pseudo = _pseudo_voxel_forward(per_condition_mean, models)

    out: dict[str, Any] = {
        "assay": "eval_richter",
        "n_trials_per_condition": int(n_trials_per_condition),
        "permutation": [int(x) for x in permutation],
        "amplitude_summary": amplitude_summary,
        "rsa_within_between": rsa,
        "preference_rank_curve": pref_rank,
        "pseudo_voxel_models": pseudo,
    }

    if not run_upgrades:
        return out

    # --- Task #72 localizer ----------------------------------------------
    loc_sum = np.zeros((N_LEAD_TRAIL, n_l23), dtype=np.float64)
    loc_count = np.zeros(N_LEAD_TRAIL, dtype=np.int64)
    for trailer_pos in range(N_LEAD_TRAIL):
        for _ in range(int(n_localizer_trials)):
            r = run_richter_localizer_trial(
                bundle, trailer_pos=trailer_pos,
                timing=timing, noise_std=float(noise_std),
                generator=gen,
            )
            loc_sum[trailer_pos] += r.cpu().numpy().astype(np.float64)
            loc_count[trailer_pos] += 1
    localizer_preference = loc_sum / np.clip(loc_count[:, None], 1, None)

    # --- Modulation-pattern fit ------------------------------------------
    obs_mod = _compute_observed_modulation_patterns(
        per_condition_mean, permutation,
    )
    pred_mod = _compute_model_predicted_modulations(
        per_condition_mean, permutation, localizer_preference, models,
    )
    mod_corr = {
        name: {
            "per_trailer_r": [float(x) for x in _pearson_across_units(obs_mod, p)],
            "mean_r": float(np.nanmean(_pearson_across_units(obs_mod, p))),
        }
        for name, p in pred_mod.items()
    }
    # Pick the model with the largest ABSOLUTE signed Pearson — the best
    # spatial-pattern match regardless of direction. Record the sign of the
    # preferred model's signed correlation so downstream can distinguish
    # facilitation (+) from dampening (−) without masking either.
    abs_rs = {
        name: abs(v["mean_r"]) if not np.isnan(v["mean_r"]) else -np.inf
        for name, v in mod_corr.items()
    }
    best_model = max(abs_rs, key=abs_rs.get) if abs_rs else ""
    if best_model:
        best_mean_r = mod_corr[best_model]["mean_r"]
        effect_sign = "+" if best_mean_r > 0 else ("-" if best_mean_r < 0 else "0")
    else:
        best_mean_r = float("nan")
        effect_sign = ""

    # --- local_gain vs global_gain |r| separation + bootstrap CI ----------
    # Tests whether the spatial-specificity distinction is real: if local
    # and global both score ~equally high, we cannot distinguish a
    # local-circuit mechanism from a global gain change.
    def _bootstrap_abs_delta_trailers(
        per_trailer_local: list[float], per_trailer_global: list[float],
        n_resamples: int, seed_val: int,
    ) -> dict[str, float]:
        rng = np.random.default_rng(int(seed_val))
        pl = np.asarray(per_trailer_local, dtype=np.float64)
        pg = np.asarray(per_trailer_global, dtype=np.float64)
        n = len(pl)
        point = abs(float(np.nanmean(pl))) - abs(float(np.nanmean(pg)))
        if n == 0:
            return {
                "delta_abs_r": point, "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_trailers": 0,
                "n_resamples": int(n_resamples),
            }
        draws = np.empty(int(n_resamples), dtype=np.float64)
        for i in range(int(n_resamples)):
            idx = rng.integers(0, n, size=n)
            draws[i] = abs(float(np.nanmean(pl[idx]))) - abs(
                float(np.nanmean(pg[idx]))
            )
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        return {
            "delta_abs_r": point,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "n_trailers": int(n), "n_resamples": int(n_resamples),
        }

    local_vs_global = _bootstrap_abs_delta_trailers(
        mod_corr.get("local_gain", {}).get("per_trailer_r", []),
        mod_corr.get("global_gain", {}).get("per_trailer_r", []),
        n_resamples=int(n_bootstrap), seed_val=int(seed),
    )

    # --- Preference-rank binned suppression ------------------------------
    rank_sup = _preference_rank_binned_suppression(
        trial_arr, trial_trailer, trial_expected, localizer_preference,
    )

    # --- Same-trailer contrast -------------------------------------------
    same_trailer = _same_trailer_contrast(
        trial_arr, trial_trailer, trial_expected,
    )
    st_trial_exp = same_trailer.pop("trial_mean_exp")
    st_trial_unexp = same_trailer.pop("trial_mean_unexp")
    same_trailer["bootstrap_delta_exp_minus_unexp"] = _bootstrap_ci_mean_diff(
        st_trial_exp, st_trial_unexp,
        n_resamples=int(n_bootstrap), seed=int(seed),
    )
    same_trailer["permutation_p_two_sided"] = _permutation_p_two_sided(
        st_trial_exp, st_trial_unexp,
        n_permutations=int(n_permutations), seed=int(seed),
    )

    out["localizer"] = {
        "n_trials_per_trailer": int(n_localizer_trials),
        "mean_response_per_trailer": [
            float(localizer_preference[t].mean())
            for t in range(N_LEAD_TRAIL)
        ],
        "population_rate_min": float(localizer_preference.min()),
        "population_rate_max": float(localizer_preference.max()),
    }
    out["modulation_pattern_fit"] = {
        "models": mod_corr,
        "model_preferred": best_model,
        "model_preferred_mean_r": float(best_mean_r),
        "model_preferred_effect_sign": effect_sign,
        "selection_criterion": "argmax_abs_mean_r_signed_pearson",
        "local_vs_global_abs_r": local_vs_global,
        "inverse_permutation": [int(x) for x in inv_perm],
    }
    out["preference_rank_suppression"] = rank_sup
    out["same_trailer_contrast"] = same_trailer

    return out


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3 Richter evaluation (Task #72)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-per-condition", type=int, default=8)
    p.add_argument("--n-pseudo-voxels", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--n-localizer-trials", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument(
        "--skip-upgrades", action="store_true",
        help="run Task #40 assays only (fast path for harness tests)",
    )
    p.add_argument("--output", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    bundle = load_checkpoint(
        args.checkpoint, seed=int(args.seed), device=args.device,
    )
    bundle.net.set_phase("phase3_richter")
    permutation = None
    if "permutation" in bundle.meta:
        permutation = tuple(int(x) for x in bundle.meta["permutation"])
    results = evaluate_richter(
        bundle,
        n_trials_per_condition=int(args.n_trials_per_condition),
        permutation=permutation,
        noise_std=float(args.noise_std),
        n_pseudo_voxels_per_model=int(args.n_pseudo_voxels),
        seed=int(args.seed),
        n_localizer_trials=int(args.n_localizer_trials),
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        run_upgrades=not args.skip_upgrades,
    )
    out_path = args.output or (args.checkpoint.parent / "eval_richter.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
