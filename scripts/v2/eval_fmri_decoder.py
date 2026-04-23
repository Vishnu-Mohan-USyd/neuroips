"""Task #74 fMRI-style V1 orientation decoder on the V2 substrate.

Purpose
-------
Substrate sanity check: quantify how much orientation information the
L2/3 E population carries when read out fMRI-style — i.e. via
retinotopic **voxel pooling** + a linear SVM — with **no cue** and
**no task weights**. This mirrors Kok 2012 / Richter voxel-level
orientation decoding, where the decoder's success comes from random
orientation-column sampling within spatial voxels rather than from
any top-down gate. Running this on a phase-2 checkpoint tells us
whether the upstream pipeline (LGN → L4 → L2/3 E) carries a readable
orientation signal before any β mechanism or context memory is brought
online.

Design (matches Task #74 dispatch spec)
---------------------------------------
Two decode regimes are run per invocation by default:

* **Coarse**: 8 orientations spanning 0°–180° at 22.5° spacing,
  ``--n-trials-coarse`` trials per orientation (default 200, total
  1600 trials). Tests broadband orientation discriminability.
* **Fine-pair**: 45° vs 135° binary, 200 trials per orientation
  (400 total). Same stimulus pair that ceilings on eval_kok's main
  SVM; we want to see whether voxel pooling + noise brings it below
  ceiling.

Each trial presents a drifting-grating frame at the requested
orientation with per-trial uniform **contrast jitter** (default
[0.7, 1.0]) and **additive Gaussian noise** on the LGN-front-end
input (``--noise-std`` default 0.1). The network is advanced for
``n_warmup_steps`` + ``n_readout_steps`` (defaults 50 + 50) and the
per-trial L2/3 E activity is the mean rate across the readout window.
``q_t=None`` throughout (cue-free) and the phase manifest is left
at ``phase2`` (no task weights).

Pseudo-voxel construction
-------------------------
L2/3 E has 256 units tiled as **16 retinotopic cells (4×4) × 16
orientation slots per cell** (see ``src/v2_model/network._build_l4_l23_mask``
for the fixed tiling rationale). Voxels are built by partitioning
the 16 orientation slots within each retinotopic cell into
``K = n_voxels // 16`` equal groups; each voxel is the mean rate
across the ``256 / n_voxels`` units in its block. This is the
Kok-style voxel: spatially localised, with a weak orientation bias
that arises from uneven orientation-column sampling within the
patch. Supported values of ``--n-voxels`` are therefore
``{16, 32, 64, 128, 256}`` (the default 64 matches the dispatch spec;
256 means "no pooling — raw L2/3 E"). The "no pooling" baseline is
always also computed and reported for reference.

Classifier
----------
``sklearn.svm.LinearSVC`` wrapped in 5-fold stratified CV on the
outer split; within each outer training fold we run an inner 5-fold
CV grid search over ``C ∈ {0.01, 0.1, 1.0, 10.0}``. The final fold
score is produced by fitting ``LinearSVC(C=best_C)`` on the full
outer-training fold and scoring on the outer-held fold. Accuracy is
reported as mean ± std across outer folds; the chance level (1/N)
is printed alongside.

Command-line
------------
::

    PYTHONPATH=. python3 -m scripts.v2.eval_fmri_decoder \\
        --checkpoint checkpoints/v2/phase2/phase2_validated_substrate_s42/phase2_s42/step_3000.pt \\
        --seed 42 --n-voxels 64 --regime both \\
        --output-coarse logs/task74/eval_fmri_decoder_coarse.json \\
        --output-fine   logs/task74/eval_fmri_decoder_fine.json

Do **not** run this on the β-trained checkpoint — the dispatch is
explicit that we need to validate the substrate-side readout first.
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
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.eval_kok import _install_learned_w_q_gain
from scripts.v2.train_phase3_kok_learning import build_cue_tensor, cue_mapping_from_seed

# Fixed L2/3 E retinotopic tiling (from src/v2_model/network._build_l4_l23_mask).
RETINO_SIDE: int = 4
N_RETINO_CELLS: int = RETINO_SIDE * RETINO_SIDE   # 16
N_ORIENT_BINS: int = 16                           # per retino cell
N_L23_E: int = N_RETINO_CELLS * N_ORIENT_BINS     # 256

SUPPORTED_N_VOXELS: tuple[int, ...] = (16, 32, 64, 128, 256)


# ---------------------------------------------------------------------------
# Voxel-pool construction
# ---------------------------------------------------------------------------


def build_voxel_pool_mask(n_voxels: int) -> np.ndarray:
    """Return a ``[n_voxels, N_L23_E]`` binary pooling mask.

    L2/3 E unit ``i`` maps to ``(retino_flat=i//16, orient_bin=i%16)``.
    Voxels partition the 16 orientation bins within each retinotopic
    cell into ``K = n_voxels // 16`` contiguous groups, each
    containing ``16 // K`` orientation bins (and therefore
    ``16 // K`` units). Total voxels = 16·K = ``n_voxels``.

    Parameters
    ----------
    n_voxels : int
        One of :data:`SUPPORTED_N_VOXELS` = {16, 32, 64, 128, 256}.
        ``n_voxels=16``  → one voxel per retino cell (16 units each,
        orientation-agnostic).
        ``n_voxels=64``  → four voxels per retino cell (4 units each,
        weak orientation bias per voxel — Kok-style).
        ``n_voxels=256`` → no pooling (identity; used as "raw L2/3"
        baseline).

    Returns
    -------
    mask : ndarray[float32], shape ``[n_voxels, 256]``
        Binary. Row ``v`` has ones on the units belonging to voxel
        ``v`` and zeros elsewhere.

    Raises
    ------
    ValueError
        If ``n_voxels`` is not in :data:`SUPPORTED_N_VOXELS`.
    """
    if int(n_voxels) not in SUPPORTED_N_VOXELS:
        raise ValueError(
            f"n_voxels must be one of {SUPPORTED_N_VOXELS}; got "
            f"{n_voxels}. The L2/3 E layout is 16 retino × 16 orient "
            f"bins, so voxel counts must split the 16 orient bins "
            f"evenly: K ∈ {{1, 2, 4, 8, 16}} voxels per retino cell."
        )
    n_voxels = int(n_voxels)
    K = n_voxels // N_RETINO_CELLS                # voxels per retino cell
    bins_per_voxel = N_ORIENT_BINS // K           # units per voxel
    mask = np.zeros((n_voxels, N_L23_E), dtype=np.float32)
    for i in range(N_L23_E):
        retino_flat = i // N_ORIENT_BINS
        orient_bin = i % N_ORIENT_BINS
        voxel_idx = retino_flat * K + (orient_bin // bins_per_voxel)
        mask[voxel_idx, i] = 1.0
    return mask


def pool_to_voxels(r_l23: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool L2/3 E activity into voxels.

    Parameters
    ----------
    r_l23 : ndarray, shape ``[N_trials, N_L23_E]``
        Per-trial L2/3 E rates.
    mask : ndarray, shape ``[n_voxels, N_L23_E]``
        Binary pooling mask from :func:`build_voxel_pool_mask`.

    Returns
    -------
    voxels : ndarray, shape ``[N_trials, n_voxels]``
        Per-trial, per-voxel mean rate (BOLD-like amplitude).
    """
    counts = mask.sum(axis=1)                     # [n_voxels]
    if np.any(counts == 0):
        raise ValueError("empty voxel in pooling mask")
    sums = r_l23 @ mask.T                         # [N_trials, n_voxels]
    return sums / counts[None, :]


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_trial(
    bundle,
    *, orientation_deg: float, contrast: float,
    noise_std: float, n_warmup: int, n_readout: int,
    generator: torch.Generator,
    cue_id: Optional[int] = None,
) -> Tensor:
    """Advance the network under a steady grating; return mean L2/3 E
    rate over the readout window.

    A blank frame is never shown — the net sees the grating from t=0.
    Per-step additive Gaussian noise with std ``noise_std`` is applied
    to the frame tensor (upstream of the LGN front end).

    Parameters
    ----------
    cue_id : int | None
        If ``None`` (default — substrate / no-cue mode), ``q_t=None``
        is delivered every step and both β paths (``l23_e.W_q_gain``
        and ``context_memory.W_qm_task``) are bypassed. If an integer,
        ``q_t = build_cue_tensor(cue_id, ...)`` is delivered every
        step (warmup AND readout); the β gate at
        ``src/v2_model/layers.py:680`` modulates the L4→L23E feed
        forward by ``W_q_gain[cue_id]`` during the readout window,
        matching the validated ``step1_beta_level11`` probe-gating
        protocol. "Expected" vs "unexpected" is encoded by the caller
        choosing ``cue_id`` to match or mismatch
        ``orientation_deg`` via the checkpoint's ``cue_mapping``.

    Returns
    -------
    r_l23 : Tensor, shape ``[N_L23_E]``, dtype float32
        Mean L2/3 E rate across the readout window.
    """
    cfg = bundle.cfg
    device = cfg.device
    grating = make_grating_frame(
        float(orientation_deg), float(contrast), cfg, device=device,
    )
    q_cue: Optional[Tensor] = None
    if cue_id is not None:
        q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)
    state = bundle.net.initial_state(batch_size=1)
    rates: list[Tensor] = []
    total_steps = int(n_warmup) + int(n_readout)
    for t in range(total_steps):
        frame = grating
        if noise_std > 0.0:
            frame = frame + float(noise_std) * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=q_cue)
        if t >= int(n_warmup):
            rates.append(info["r_l23"][0].clone())
    return torch.stack(rates, dim=0).mean(dim=0)


def _pick_cue_id_for_orient(
    cue_mode: str, orientation_deg: float,
    cue_mapping: dict[int, float],
) -> Optional[int]:
    """Resolve cue_id from ``(cue_mode, orientation_deg, cue_mapping)``.

    ``cue_mode``:
      * ``"none"``     → returns None (no cue delivered, β bypass).
      * ``"expected"`` → cue whose ``cue_mapping`` value matches
        ``orientation_deg`` within 1e-6°. Raises if no match.
      * ``"unexpected"`` → cue whose value does NOT match
        ``orientation_deg``. For a 2-entry cue_mapping this is the
        other cue; for 3+ entries we'd need a policy (we currently
        only support binary, since that's the Kok stimulus pair).
    """
    mode = cue_mode.strip().lower()
    if mode == "none":
        return None
    match_ids = [
        int(k) for k, v in cue_mapping.items()
        if abs(float(v) - float(orientation_deg)) < 1e-6
    ]
    if mode == "expected":
        if len(match_ids) != 1:
            raise ValueError(
                f"cue_mode=expected needs exactly one cue matching "
                f"{orientation_deg}°, got {match_ids} from "
                f"cue_mapping={cue_mapping}"
            )
        return match_ids[0]
    if mode == "unexpected":
        mismatch_ids = [
            int(k) for k, v in cue_mapping.items()
            if abs(float(v) - float(orientation_deg)) >= 1e-6
        ]
        if len(cue_mapping) != 2 or len(mismatch_ids) != 1:
            raise ValueError(
                f"cue_mode=unexpected currently supports only binary "
                f"cue_mapping (2 entries) with a unique orthogonal "
                f"cue; got cue_mapping={cue_mapping}"
            )
        return mismatch_ids[0]
    raise ValueError(
        f"cue_mode must be one of 'none'/'expected'/'unexpected'; "
        f"got {cue_mode!r}"
    )


def collect_trials(
    bundle,
    *, orientations_deg: list[float], n_trials_per_orient: int,
    noise_std: float, contrast_min: float, contrast_max: float,
    n_warmup: int, n_readout: int, seed: int,
    cue_mode: str = "none",
    cue_mapping: Optional[dict[int, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_trials_per_orient`` trials at each orientation.

    ``cue_mode`` ∈ ``{"none", "expected", "unexpected"}`` selects which
    (if any) cue is delivered via ``q_t`` each step — see
    :func:`_pick_cue_id_for_orient`. ``cue_mapping`` is required unless
    ``cue_mode="none"``.

    Returns
    -------
    r_all : ndarray, shape ``[N_trials, N_L23_E]``, float32
        Trial-by-unit L2/3 E rates.
    y : ndarray, shape ``[N_trials]``, int64
        Orientation class index (``0..len(orientations_deg)-1``).
    contrasts : ndarray, shape ``[N_trials]``, float32
        Per-trial contrast actually used (for audit).
    """
    if cue_mode != "none" and cue_mapping is None:
        raise ValueError(
            f"cue_mode={cue_mode!r} requires cue_mapping; pass the "
            "checkpoint's cue_mapping from bundle.meta (or a dict of "
            "{cue_id: orientation_deg})."
        )
    gen = torch.Generator(device=bundle.cfg.device).manual_seed(int(seed))
    r_list: list[np.ndarray] = []
    y_list: list[int] = []
    c_list: list[float] = []
    for ori_idx, ori_deg in enumerate(orientations_deg):
        cue_id = _pick_cue_id_for_orient(
            cue_mode, float(ori_deg), cue_mapping or {},
        )
        for _ in range(int(n_trials_per_orient)):
            c = float(contrast_min) + (
                float(contrast_max) - float(contrast_min)
            ) * float(torch.rand(1, generator=gen).item())
            r = run_trial(
                bundle, orientation_deg=float(ori_deg), contrast=c,
                noise_std=float(noise_std),
                n_warmup=int(n_warmup), n_readout=int(n_readout),
                generator=gen,
                cue_id=cue_id,
            )
            r_list.append(r.cpu().numpy().astype(np.float32))
            y_list.append(int(ori_idx))
            c_list.append(c)
    return (
        np.stack(r_list, axis=0),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(c_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 5-fold CV with inner C search
# ---------------------------------------------------------------------------


def svm_5fold_cv_with_C(
    X: np.ndarray, y: np.ndarray,
    *, C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Outer 5-fold stratified CV with inner C grid search.

    For each outer fold: run inner 5-fold CV over ``C_grid``, pick the
    best-mean-accuracy C, refit on the whole outer-training fold at
    that C, score on the outer-held fold. Returns fold accuracies,
    the per-fold best C, and mean ± std across folds.
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.svm import LinearSVC
    except ImportError as e:
        return {"error": f"sklearn not available: {e}"}
    if X.shape[0] < 10 or len(np.unique(y)) < 2:
        return {"error": "insufficient data for 5-fold CV"}

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    fold_accs: list[float] = []
    fold_Cs: list[float] = []
    for outer_tr, outer_te in outer.split(X, y):
        X_tr, y_tr = X[outer_tr], y[outer_tr]
        X_te, y_te = X[outer_te], y[outer_te]

        # Inner CV for C selection — use a different random_state so
        # inner folds don't replicate outer splits.
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=int(seed) + 7919,
        )
        best_C = C_grid[0]
        best_score = -1.0
        for C in C_grid:
            inner_accs: list[float] = []
            for inner_tr, inner_te in inner.split(X_tr, y_tr):
                clf = LinearSVC(
                    C=float(C), random_state=int(seed),
                    max_iter=10000, dual="auto",
                )
                clf.fit(X_tr[inner_tr], y_tr[inner_tr])
                inner_accs.append(float(clf.score(X_tr[inner_te], y_tr[inner_te])))
            mean_inner = float(np.mean(inner_accs))
            if mean_inner > best_score:
                best_score = mean_inner
                best_C = float(C)

        clf = LinearSVC(
            C=float(best_C), random_state=int(seed),
            max_iter=10000, dual="auto",
        )
        clf.fit(X_tr, y_tr)
        fold_accs.append(float(clf.score(X_te, y_te)))
        fold_Cs.append(float(best_C))

    return {
        "acc_mean": float(np.mean(fold_accs)),
        "acc_std": float(np.std(fold_accs)),
        "per_fold_acc": [float(x) for x in fold_accs],
        "per_fold_C": [float(x) for x in fold_Cs],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _run_regime(
    bundle, *, orientations_deg: list[float], n_trials_per_orient: int,
    mask: np.ndarray, noise_std: float, contrast_min: float,
    contrast_max: float, n_warmup: int, n_readout: int,
    trial_seed: int, cv_seed: int, regime_label: str,
) -> dict[str, Any]:
    """Collect trials and score both voxel-pooled and raw L2/3 E decoders."""
    print(
        f"[fmri_decoder] {regime_label}: collecting "
        f"{len(orientations_deg)} orient × {n_trials_per_orient} trials "
        f"(noise_std={noise_std}, contrast∈[{contrast_min},{contrast_max}], "
        f"{n_warmup}+{n_readout} steps)...",
        flush=True,
    )
    X, y, contrasts = collect_trials(
        bundle, orientations_deg=orientations_deg,
        n_trials_per_orient=int(n_trials_per_orient),
        noise_std=float(noise_std),
        contrast_min=float(contrast_min),
        contrast_max=float(contrast_max),
        n_warmup=int(n_warmup), n_readout=int(n_readout),
        seed=int(trial_seed),
    )
    print(
        f"[fmri_decoder] {regime_label}: collected X{X.shape} y{y.shape}; "
        f"running 5-fold CV (voxel-pooled + raw).",
        flush=True,
    )
    X_vox = pool_to_voxels(X, mask)
    svm_vox = svm_5fold_cv_with_C(X_vox, y, seed=int(cv_seed))
    svm_raw = svm_5fold_cv_with_C(X, y, seed=int(cv_seed))
    return {
        "orientations_deg": [float(o) for o in orientations_deg],
        "n_orientations": int(len(orientations_deg)),
        "chance": float(1.0 / len(orientations_deg)),
        "n_trials_per_orient": int(n_trials_per_orient),
        "n_trials_total": int(X.shape[0]),
        "n_voxels": int(mask.shape[0]),
        "noise_std": float(noise_std),
        "contrast_range": [float(contrast_min), float(contrast_max)],
        "n_warmup_steps": int(n_warmup),
        "n_readout_steps": int(n_readout),
        "contrast_mean": float(contrasts.mean()),
        "svm_voxel_pooled": svm_vox,
        "svm_raw_l23": svm_raw,
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--n-voxels", type=int, default=64,
        choices=SUPPORTED_N_VOXELS,
        help="Pseudo-voxel count (partition of 256 L2/3 E units)."
    )
    p.add_argument(
        "--regime", choices=("coarse", "fine", "both"), default="both",
        help="Which decode regime to run. Both by default.",
    )
    p.add_argument(
        "--n-trials-coarse", type=int, default=200,
        help="Trials per orientation for the 8-way coarse regime.",
    )
    p.add_argument(
        "--n-trials-fine", type=int, default=200,
        help="Trials per orientation for the 2-way 45°/135° regime.",
    )
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--contrast-min", type=float, default=0.7)
    p.add_argument("--contrast-max", type=float, default=1.0)
    p.add_argument("--n-warmup-steps", type=int, default=50)
    p.add_argument("--n-readout-steps", type=int, default=50)
    p.add_argument(
        "--output-coarse", type=Path,
        default=Path("logs/task74/eval_fmri_decoder_coarse.json"),
    )
    p.add_argument(
        "--output-fine", type=Path,
        default=Path("logs/task74/eval_fmri_decoder_fine.json"),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help=(
            "Pipeline shakeout: override n-trials-coarse=10, "
            "n-trials-fine=20, n-warmup=10, n-readout=10. Does NOT "
            "match the dispatch spec — only use for CI / debugging."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)

    if args.smoke:
        args.n_trials_coarse = 10
        args.n_trials_fine = 20
        args.n_warmup_steps = 10
        args.n_readout_steps = 10

    bundle = load_checkpoint(
        args.checkpoint, seed=int(args.seed), device=args.device,
    )
    # IMPORTANT: do not call set_phase("phase3_*") — we want phase-2
    # substrate behaviour with no task weights in play.
    bundle.net.eval()

    mask = build_voxel_pool_mask(int(args.n_voxels))

    out: dict[str, Any] = {}

    if args.regime in ("coarse", "both"):
        orients_coarse = np.linspace(0.0, 180.0, 8, endpoint=False).tolist()
        out["coarse"] = _run_regime(
            bundle, orientations_deg=orients_coarse,
            n_trials_per_orient=int(args.n_trials_coarse),
            mask=mask,
            noise_std=float(args.noise_std),
            contrast_min=float(args.contrast_min),
            contrast_max=float(args.contrast_max),
            n_warmup=int(args.n_warmup_steps),
            n_readout=int(args.n_readout_steps),
            trial_seed=int(args.seed),
            cv_seed=int(args.seed),
            regime_label="coarse 8-way",
        )
        args.output_coarse.parent.mkdir(parents=True, exist_ok=True)
        args.output_coarse.write_text(json.dumps(out["coarse"], indent=2))

    if args.regime in ("fine", "both"):
        orients_fine = [45.0, 135.0]
        out["fine"] = _run_regime(
            bundle, orientations_deg=orients_fine,
            n_trials_per_orient=int(args.n_trials_fine),
            mask=mask,
            noise_std=float(args.noise_std),
            contrast_min=float(args.contrast_min),
            contrast_max=float(args.contrast_max),
            n_warmup=int(args.n_warmup_steps),
            n_readout=int(args.n_readout_steps),
            # Offset the trial seed for fine so the RNG stream
            # is independent from coarse (avoids an accidental
            # reuse of the same noise realisations).
            trial_seed=int(args.seed) + 1,
            cv_seed=int(args.seed),
            regime_label="fine 2-way",
        )
        args.output_fine.parent.mkdir(parents=True, exist_ok=True)
        args.output_fine.write_text(json.dumps(out["fine"], indent=2))

    # --- Stdout report table -----------------------------------------------
    print("fmri_decoder (phase-2 substrate, no cue, no task weights)")
    for key, label in (("coarse", "coarse (8-way)"), ("fine", "fine   (2-way 45/135)")):
        if key not in out:
            continue
        r = out[key]
        sv = r["svm_voxel_pooled"]
        sr = r["svm_raw_l23"]
        print(
            f"  {label}, chance={r['chance']:.3f}: "
            f"acc_mean={sv['acc_mean']:.4f}, "
            f"acc_std={sv['acc_std']:.4f}, n_voxels={r['n_voxels']}"
        )
        print(
            f"    no-pooling (raw L2/3 E, 256-d): "
            f"acc_mean={sr['acc_mean']:.4f}, acc_std={sr['acc_std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
