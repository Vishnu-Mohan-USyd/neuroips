"""Phase-3 Kok evaluation (plan v4 / Task #40).

Loads a Phase-3-Kok checkpoint and runs the Kok 2012 expected / unexpected
comparison. Primary metrics:

  1. **Mean L2/3 amplitude per cell (4 cells)** — the 256 L2/3 E units
     are split into 4 non-overlapping groups of 64 and their probe-epoch
     mean rates are reported per group. ``n_trials_per_condition``
     defaults to 500.
  2. **5-fold CV ``LinearSVC`` orientation decoding** — per condition
     (expected / unexpected), predict orientation from the probe-epoch
     L2/3 vector. Chance = 0.5 on two orientations.
  3. **Pref / non-pref asymmetry** — classify each L2/3 unit by its
     preferred orientation (higher mean response over all trials),
     then report:
         * ``pref_exp`` / ``pref_unexp``     (mean response of each
           unit to its preferred orientation, split by condition)
         * ``nonpref_exp`` / ``nonpref_unexp``
         * ``asymmetry`` = (nonpref_unexp − nonpref_exp)
                         − (pref_unexp − pref_exp)
       A positive asymmetry is the Kok-2012 sharpening prediction:
       expectation suppresses non-preferred activity more than preferred
       activity, i.e. non-preferred units are suppressed MORE by
       expectation than preferred units. (Critique C3 / Task #68:
       docstring corrected to match the formula — the earlier wording
       described dampening, which is the opposite sign.)

Writes ``eval_kok.json`` next to the checkpoint. Exit 0 always — the
script reports metrics; a separate pass/fail policy lives upstream.

Usage:
    python -m scripts.v2.eval_kok \\
        --checkpoint checkpoints/v2/phase3_kok/phase3_kok_s42.pt \\
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

from scripts.v2._gates_common import (
    CheckpointBundle, load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


__all__ = [
    "run_kok_probe_trial", "evaluate_kok",
]


# ---------------------------------------------------------------------------
# Single-trial forward (eval-only; no plasticity)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_kok_probe_trial(
    bundle: CheckpointBundle,
    *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    """Return the mean L2/3 rate vector during probe1 for one eval trial.

    Shape ``[n_l23_e]`` — used by every downstream metric
    (amp-per-cell-group, decoder, asymmetry).
    """
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

    probe_rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, None
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
            probe_rates.append(info["r_l23"][0].clone())

    return torch.stack(probe_rates, dim=0).mean(dim=0)             # [n_l23_e]


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------


def _per_group_means(
    r_l23: np.ndarray, n_groups: int = 4,
) -> list[float]:
    """Split ``[n_trials, n_l23]`` into ``n_groups`` column blocks; return
    the grand mean of each block (averaged over trials and units)."""
    n_units = r_l23.shape[1]
    if n_units % n_groups != 0:
        raise ValueError(
            f"n_l23={n_units} not divisible by n_groups={n_groups}"
        )
    block = n_units // n_groups
    return [
        float(r_l23[:, g * block:(g + 1) * block].mean())
        for g in range(n_groups)
    ]


def _svm_5fold_cv(
    X: np.ndarray, y: np.ndarray, *, seed: int,
) -> dict[str, Any]:
    """5-fold CV LinearSVC accuracy. Returns mean + per-fold scores."""
    try:
        from sklearn.model_selection import StratifiedKFold            # type: ignore
        from sklearn.svm import LinearSVC                              # type: ignore
    except ImportError:
        return {"error": "sklearn not available"}
    if X.shape[0] < 10 or len(np.unique(y)) < 2:
        return {"error": "insufficient data for 5-fold CV"}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    fold_acc: list[float] = []
    for tr, te in skf.split(X, y):
        clf = LinearSVC(random_state=int(seed), max_iter=5000, dual="auto")
        clf.fit(X[tr], y[tr])
        fold_acc.append(float(clf.score(X[te], y[te])))
    return {
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "per_fold": fold_acc,
    }


def _pref_nonpref_asymmetry(
    r_l23: np.ndarray,               # [n_trials, n_l23]
    orientation_deg: np.ndarray,     # [n_trials] — values in {45, 135}
    expected: np.ndarray,            # [n_trials] — bool, True when cue matches probe
) -> dict[str, Any]:
    """Split units by preferred orientation, measure exp vs unexp effect."""
    orient_vals = tuple(np.unique(orientation_deg))
    if len(orient_vals) != 2:
        return {"error": "need exactly two orientations"}
    o_lo, o_hi = float(orient_vals[0]), float(orient_vals[1])

    r_lo = r_l23[orientation_deg == o_lo].mean(axis=0)              # [n_l23]
    r_hi = r_l23[orientation_deg == o_hi].mean(axis=0)
    pref_is_hi = r_hi >= r_lo                                       # [n_l23]

    def _cell_mean(mask_trials: np.ndarray, unit_mask: np.ndarray) -> float:
        sub = r_l23[np.ix_(mask_trials, unit_mask)]
        return float(sub.mean()) if sub.size else float("nan")

    pref_exp_lo = _cell_mean(
        (orientation_deg == o_lo) & expected, ~pref_is_hi,
    )
    pref_exp_hi = _cell_mean(
        (orientation_deg == o_hi) & expected, pref_is_hi,
    )
    pref_unexp_lo = _cell_mean(
        (orientation_deg == o_lo) & ~expected, ~pref_is_hi,
    )
    pref_unexp_hi = _cell_mean(
        (orientation_deg == o_hi) & ~expected, pref_is_hi,
    )
    nonpref_exp_lo = _cell_mean(
        (orientation_deg == o_lo) & expected, pref_is_hi,
    )
    nonpref_exp_hi = _cell_mean(
        (orientation_deg == o_hi) & expected, ~pref_is_hi,
    )
    nonpref_unexp_lo = _cell_mean(
        (orientation_deg == o_lo) & ~expected, pref_is_hi,
    )
    nonpref_unexp_hi = _cell_mean(
        (orientation_deg == o_hi) & ~expected, ~pref_is_hi,
    )

    pref_exp = float(np.nanmean([pref_exp_lo, pref_exp_hi]))
    pref_unexp = float(np.nanmean([pref_unexp_lo, pref_unexp_hi]))
    nonpref_exp = float(np.nanmean([nonpref_exp_lo, nonpref_exp_hi]))
    nonpref_unexp = float(np.nanmean([nonpref_unexp_lo, nonpref_unexp_hi]))
    asymmetry = (nonpref_unexp - nonpref_exp) - (pref_unexp - pref_exp)

    return {
        "pref_expected": pref_exp,
        "pref_unexpected": pref_unexp,
        "nonpref_expected": nonpref_exp,
        "nonpref_unexpected": nonpref_unexp,
        "asymmetry": float(asymmetry),
        "n_units_pref_hi": int(pref_is_hi.sum()),
        "n_units_pref_lo": int((~pref_is_hi).sum()),
    }


# ---------------------------------------------------------------------------
# Top-level eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_kok(
    bundle: CheckpointBundle,
    *, n_trials_per_condition: int = 500,
    cue_mapping: Optional[dict[int, float]] = None,
    timing: Optional[KokTiming] = None,
    n_cell_groups: int = 4,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Kok expected/unexpected eval, return metrics dict for JSON dump."""
    cfg = bundle.cfg
    timing = timing or KokTiming()
    cue_mapping = cue_mapping or cue_mapping_from_seed(seed)

    gen = torch.Generator(device=cfg.device); gen.manual_seed(int(seed))

    trials_l23: list[Tensor] = []
    trials_orient: list[float] = []
    trials_cue: list[int] = []
    trials_expected: list[bool] = []

    for cue_id in (0, 1):
        cue_probe = cue_mapping[cue_id]
        other_probe = cue_mapping[1 - cue_id]
        for probe_deg, is_expected in (
            (cue_probe, True), (other_probe, False),
        ):
            for _ in range(int(n_trials_per_condition)):
                r = run_kok_probe_trial(
                    bundle, cue_id=cue_id,
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=float(noise_std),
                    generator=gen,
                )
                trials_l23.append(r)
                trials_orient.append(float(probe_deg))
                trials_cue.append(int(cue_id))
                trials_expected.append(bool(is_expected))

    r_arr = torch.stack(trials_l23, dim=0).cpu().numpy()            # [N, n_l23]
    orient = np.asarray(trials_orient, dtype=np.float64)
    expected = np.asarray(trials_expected, dtype=bool)

    per_cell = {}
    for tag, mask in (
        ("expected", expected),
        ("unexpected", ~expected),
        ("all", np.ones_like(expected, dtype=bool)),
    ):
        per_cell[tag] = _per_group_means(r_arr[mask], n_groups=int(n_cell_groups))

    # SVM accuracy: orient ∈ {o_lo, o_hi} → 2-class decoder.
    uniq = np.unique(orient)
    y = np.where(orient == uniq[0], 0, 1).astype(np.int64)
    svm_all = _svm_5fold_cv(r_arr, y, seed=int(seed))
    svm_expected = _svm_5fold_cv(r_arr[expected], y[expected], seed=int(seed))
    svm_unexpected = _svm_5fold_cv(
        r_arr[~expected], y[~expected], seed=int(seed),
    )
    asymmetry = _pref_nonpref_asymmetry(r_arr, orient, expected)

    return {
        "assay": "eval_kok",
        "n_trials_per_condition": int(n_trials_per_condition),
        "cue_mapping": {int(k): float(v) for k, v in cue_mapping.items()},
        "orientations_deg": [float(x) for x in uniq],
        "per_cell_mean_l23": per_cell,
        "n_cell_groups": int(n_cell_groups),
        "svm": {
            "all": svm_all, "expected": svm_expected,
            "unexpected": svm_unexpected,
        },
        "pref_nonpref": asymmetry,
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3 Kok evaluation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-per-condition", type=int, default=500)
    p.add_argument("--n-cell-groups", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--output", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    bundle = load_checkpoint(
        args.checkpoint, seed=int(args.seed), device=args.device,
    )
    bundle.net.set_phase("phase3_kok")
    cue_mapping = None
    if "cue_mapping" in bundle.meta:
        cue_mapping = {int(k): float(v) for k, v in bundle.meta["cue_mapping"].items()}
    results = evaluate_kok(
        bundle, n_trials_per_condition=int(args.n_trials_per_condition),
        cue_mapping=cue_mapping,
        n_cell_groups=int(args.n_cell_groups),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
    )
    out_path = args.output or (args.checkpoint.parent / "eval_kok.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
