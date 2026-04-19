"""Phase-3 Richter evaluation (plan v4 / Task #40).

Loads a Phase-3-Richter checkpoint and produces the primary + supplementary
Richter-like analyses.

Primary:
  1. **Unit-level L2/3 amplitude on trailing** — per-(leader, trailer)
     mean L2/3 rate across trailer steps, shape
     ``[n_leaders, n_trailers, n_l23_e]``.
  2. **Within / between-class RSA** — token-identity RSA over the
     trailer response. ``within_mean`` is the mean pairwise distance
     among trials sharing the same ``(leader, trailer)`` condition;
     ``between_mean`` is across different conditions. A lower
     ``within`` than ``between`` implies token-identity information
     survives the leader-conditioned memory bias.
  3. **Preference-rank suppression curve** — for each unit, rank the 6
     trailers by response magnitude (under matched leader). Report the
     curve of mean rate vs rank (1 = most preferred). The population-
     average curve is a canonical Richter-like suppression signature.

Supplementary:
  4. **6-model pseudo-voxel forward comparison** — ``~50`` pseudo-
     voxels drawn per model; 6 models cover
     ``{local, remote, global} × {gain, tuning}``. Each model projects
     the observed L2/3 tensor into a model-specific pseudo-voxel space
     and correlates with the population-mean response. Reports
     Pearson r per model.

Writes ``eval_richter.json`` next to the checkpoint. Exit 0 always.

Usage:
    python -m scripts.v2.eval_richter \\
        --checkpoint checkpoints/v2/phase3_richter/phase3_richter_s42.pt \\
        --seed 42
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
    "run_richter_probe_trial", "evaluate_richter",
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


# ---------------------------------------------------------------------------
# RSA — within vs between leader-trailer classes
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
# Preference-rank suppression curve
# ---------------------------------------------------------------------------


def _preference_rank_curve(
    per_condition: np.ndarray,       # [n_leaders, n_trailers, n_l23]
) -> dict[str, Any]:
    """Mean L2/3 rate vs trailer preference rank (1 = most preferred).

    For each unit and each leader, rank the ``n_trailers`` responses from
    largest to smallest; the rank-1 slot holds the unit's most-preferred
    trailer under that leader. Population-average curve is reported.
    """
    n_leaders, n_trailers, n_l23 = per_condition.shape
    # Rank per (leader, unit) across trailer axis, descending.
    order = np.argsort(-per_condition, axis=1)                      # [L, T, U]
    ranked = np.take_along_axis(per_condition, order, axis=1)       # [L, T, U]
    curve_mean = ranked.mean(axis=(0, 2))                           # [n_trailers]
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
# 6-model pseudo-voxel forward comparison (supplementary)
# ---------------------------------------------------------------------------


def _build_pseudo_voxel_models(
    n_l23: int, n_voxels_per_model: int, seed: int,
) -> dict[str, np.ndarray]:
    """Return ``{model_name: projection[n_voxels, n_l23]}`` for 6 models.

    Models: {local, remote, global} × {gain, tuning}. All are synthetic
    linear projections; differ in spatial support and sign structure:

    * ``local``  — each voxel spans a contiguous block of ``block_size``
      units.
    * ``remote`` — each voxel spans units scattered across the population
      (interleaved stride).
    * ``global`` — every voxel covers all units uniformly.
    * ``gain``   — projection entries are non-negative (pure scaling).
    * ``tuning`` — projection entries are zero-mean (orientation-flip /
      tuning shift pattern; voxel sums to ~0).
    """
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
                else:  # tuning: zero-mean signed pattern
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
    per_condition: np.ndarray,       # [n_leaders, n_trailers, n_l23]
    models: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Project observed responses through each model; correlate with the
    population-mean condition response."""
    n_leaders, n_trailers, n_l23 = per_condition.shape
    flat = per_condition.reshape(n_leaders * n_trailers, n_l23)     # [C, U]
    pop_mean = flat.mean(axis=1)                                    # [C]
    pop_mean -= pop_mean.mean()
    pop_mean /= (pop_mean.std() + 1e-9)

    out: dict[str, dict[str, float]] = {}
    for name, proj in models.items():
        voxel_activity = flat @ proj.T                              # [C, V]
        voxel_mean = voxel_activity.mean(axis=1)                    # [C]
        voxel_mean_c = voxel_mean - voxel_mean.mean()
        voxel_mean_c /= (voxel_mean_c.std() + 1e-9)
        r = float((pop_mean * voxel_mean_c).mean())
        out[name] = {
            "pearson_r_with_pop_mean": r,
            "n_voxels": int(proj.shape[0]),
        }
    return out


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
) -> dict[str, Any]:
    """Run Richter eval: unit-level amplitude, RSA, preference rank, models."""
    cfg = bundle.cfg
    timing = timing or RichterTiming()
    permutation = permutation or permutation_from_seed(seed)

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
    for leader_pos in range(N_LEAD_TRAIL):
        for trailer_pos in range(N_LEAD_TRAIL):
            for _ in range(int(n_trials_per_condition)):
                r = run_richter_probe_trial(
                    bundle, leader_pos=leader_pos, trailer_pos=trailer_pos,
                    timing=timing, noise_std=float(noise_std),
                    generator=gen,
                )
                r_np = r.cpu().numpy().astype(np.float64)           # [n_l23]
                per_condition_sum[leader_pos, trailer_pos] += r_np
                per_condition_count[leader_pos, trailer_pos] += 1
                all_trials.append(r_np)
                all_cls.append(leader_pos * N_LEAD_TRAIL + trailer_pos)

    per_condition_mean = per_condition_sum / np.clip(
        per_condition_count[..., None], 1, None,
    )

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
        np.stack(all_trials, axis=0),
        np.asarray(all_cls, dtype=np.int64),
    )
    pref_rank = _preference_rank_curve(per_condition_mean)
    models = _build_pseudo_voxel_models(
        n_l23, int(n_pseudo_voxels_per_model), int(seed),
    )
    pseudo = _pseudo_voxel_forward(per_condition_mean, models)

    return {
        "assay": "eval_richter",
        "n_trials_per_condition": int(n_trials_per_condition),
        "permutation": [int(x) for x in permutation],
        "amplitude_summary": amplitude_summary,
        "rsa_within_between": rsa,
        "preference_rank_curve": pref_rank,
        "pseudo_voxel_models": pseudo,
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3 Richter evaluation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-per-condition", type=int, default=8)
    p.add_argument("--n-pseudo-voxels", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.01)
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
    )
    out_path = args.output or (args.checkpoint.parent / "eval_richter.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
