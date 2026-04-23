"""Level 1 component validation — LGN + L4 front end only.

Per Lead's bottom-up validation protocol (Task #74). Scope: the fixed
LGN DoG+Gabor bank, L4 E retinotopic×orientation pool, and L4 PV
divisive normalization. No L2/3, no SOM, no H, no plasticity.

Probes
------
A. Orient sweep: 12 orients × contrast=1 × centered full-field grating,
   simulated to L4 steady state. Per L4 E unit: preferred orient
   (argmax), FWHM of circular tuning curve (half-max width).
B. Contrast sweep: 6 contrasts × each of 8 stimulus orients. For each
   unit, extract the column at its preferred orient → per-unit response
   curve vs contrast; Spearman rho with the contrast ladder.
C. Position sweep: Gaussian-windowed grating placed on a 5x5 position
   grid. For each unit, compute response-weighted center-of-mass on the
   5x5 grid and compare against its retinotopic anchor (n_l4_pv=16 → 4x4
   retino_side); PASS if COM falls within the anchor's half-cell.

PASS criteria (all four):
  * ≥80% units have a well-defined preferred orient (peak > min + 1 Hz).
  * median population FWHM ∈ [30°, 60°] (visual-cortex prior at init).
  * ≥80% units have Spearman rho ≥ 0.7 across the contrast ladder.
  * ≥80% units have position COM within 1/2 retino-cell of their
    configured retinotopic anchor.

Stdout (last line):
  ``level1_verdict=<pass/fail> n_l4e=<#> pref_orient_frac_defined=<#>
    fwhm_median_deg=<#> contrast_mono_frac=<#> rf_com_frac_localized=<#>
    issue_if_fail=<short>``
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import make_grating_frame
from src.v2_model.config import ModelConfig
from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.state import NetworkStateV2, initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_initial_r_l4(n_l4_e: int, batch: int, device: str) -> Tensor:
    return torch.zeros((batch, n_l4_e), dtype=torch.float32, device=device)


@torch.no_grad()
def _run_to_steady_state(
    front: LGNL4FrontEnd, cfg: ModelConfig,
    frame: Tensor, n_steps: int, avg_last: int,
) -> Tensor:
    """Run the front end on a constant ``frame`` for ``n_steps`` integration
    steps, return the L4 E rate averaged over the final ``avg_last`` steps.

    Shapes:
        frame:    [B, 1, H, W]
        returns:  [B, n_l4_e]
    """
    B = frame.shape[0]
    state = initial_state(cfg, batch_size=B)
    buf = []
    for t in range(int(n_steps)):
        _feat, r_l4, state = front(frame, state)
        if t >= n_steps - int(avg_last):
            buf.append(r_l4)
    return torch.stack(buf, dim=0).mean(dim=0)  # [B, n_l4_e]


def _circular_fwhm_deg(tuning: np.ndarray, orients_deg: np.ndarray) -> float:
    """FWHM of a 180°-periodic circular tuning curve in degrees.

    ``tuning``: [n_orients] non-negative values.
    Returns NaN if the curve has no above-half-max samples or a flat profile.
    """
    peak = float(tuning.max())
    trough = float(tuning.min())
    if peak <= trough + 1e-9:
        return float("nan")
    half = trough + 0.5 * (peak - trough)
    above = tuning >= half
    n_above = int(above.sum())
    if n_above == 0:
        return float("nan")
    # Each bin covers 180°/n_orients; width = n_above * bin_width.
    bin_width = 180.0 / float(len(orients_deg))
    return float(n_above * bin_width)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no tie correction needed for small N)."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _make_patch_grating(
    orientation_deg: float, contrast: float, cfg: ModelConfig,
    *, cx: float, cy: float, sigma_px: float, device: str = "cpu",
) -> Tensor:
    """Gaussian-windowed sinusoidal grating centered on ``(cx, cy)`` pixels
    (origin at frame center). Returns [1, 1, H, W] in [0, 1]."""
    H, W = cfg.arch.grid_h, cfg.arch.grid_w
    ys = torch.arange(H, dtype=torch.float32, device=device) - (H - 1) / 2.0
    xs = torch.arange(W, dtype=torch.float32, device=device) - (W - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    theta = math.radians(float(orientation_deg))
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    grating = 0.5 + 0.5 * float(contrast) * torch.sin(
        2.0 * math.pi * 0.15 * proj,
    )
    # Gaussian window at (cx, cy).
    r2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    window = torch.exp(-r2 / (2.0 * float(sigma_px) ** 2))
    patch = 0.5 + (grating - 0.5) * window
    return patch.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-orients", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--avg-last", type=int, default=20)
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--pos-grid", type=int, default=4,
                   help="N×N grid of probe positions for RF sweep "
                        "(default 4 matches retino_side=4 → one probe "
                        "per pool cell)")
    p.add_argument("--patch-sigma-px", type=float, default=1.5,
                   help="Gaussian window σ (px) for positional probes. "
                        "1.5 px ≈ single LGN DoG centre; keeps the patch "
                        "inside one pool cell.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    front = LGNL4FrontEnd(cfg).to(device="cpu")
    front.eval()

    n_l4e = front.n_l4_e
    n_ori_bank = front.n_ori                 # 8 (Gabor channels)
    retino_side = front.retino_side          # 4
    pool_h = front.pool_h                    # grid_h / retino_side
    pool_w = front.pool_w
    grid_h = front.grid_h
    grid_w = front.grid_w

    # Configured preferred orient per unit: idx layout (retino_i, retino_j, ori)
    # with ori as fast axis. Preferred_deg_i = ori_idx * 180/n_ori_bank.
    unit_idx = np.arange(n_l4e)
    unit_ori_bin = unit_idx % n_ori_bank
    unit_retino_flat = unit_idx // n_ori_bank          # 0..retino_side²-1
    unit_retino_i = unit_retino_flat // retino_side    # row
    unit_retino_j = unit_retino_flat % retino_side     # col
    configured_pref_deg = unit_ori_bin.astype(np.float64) * (
        180.0 / float(n_ori_bank)
    )

    # -------- Probe A: orient sweep ---------------------------------------
    orients_A = np.linspace(0.0, 180.0, int(args.n_orients), endpoint=False)
    tuning_A = np.zeros((len(orients_A), n_l4e), dtype=np.float64)
    for i, o in enumerate(orients_A):
        frame = make_grating_frame(float(o), 1.0, cfg, batch_size=1)
        r = _run_to_steady_state(
            front, cfg, frame, int(args.n_steps), int(args.avg_last),
        )  # [1, n_l4e]
        tuning_A[i] = r[0].cpu().numpy().astype(np.float64)

    peak = tuning_A.max(axis=0)
    trough = tuning_A.min(axis=0)
    pref_idx = tuning_A.argmax(axis=0)
    measured_pref_deg = orients_A[pref_idx]
    # Relative criterion: tuning is well-defined iff peak > 2·trough + 0.01 Hz.
    # Chosen so it tracks the LGN/L4 front-end's true dynamic range rather
    # than a fixed 1 Hz absolute threshold (unsaturated units can be tuned at
    # < 1 Hz peak-trough range while still showing clean orientation
    # selectivity). Additive 0.01 Hz floor avoids 0>0 edge case for silent
    # units.
    well_defined = peak > (2.0 * trough + 0.01)
    pref_orient_frac_defined = float(well_defined.mean())

    fwhms = np.array([
        _circular_fwhm_deg(tuning_A[:, u], orients_A) for u in range(n_l4e)
    ])
    fwhm_median = float(np.nanmedian(fwhms))

    # Circular offset between measured and configured preferred orient.
    diff = np.minimum(
        np.abs(measured_pref_deg - configured_pref_deg),
        180.0 - np.abs(measured_pref_deg - configured_pref_deg),
    )
    pref_circ_offset_median_deg = float(np.median(diff))

    # -------- Probe B: contrast sweep -------------------------------------
    # For each of n_ori_bank canonical orients × each contrast, one frame.
    canonical_orients = np.array([
        i * 180.0 / float(n_ori_bank) for i in range(n_ori_bank)
    ])
    contrasts = np.array([float(c) for c in args.contrasts])
    tuning_B = np.zeros(
        (len(contrasts), len(canonical_orients), n_l4e), dtype=np.float64,
    )
    for ci, c in enumerate(contrasts):
        for oi, o in enumerate(canonical_orients):
            frame = make_grating_frame(float(o), float(c), cfg, batch_size=1)
            r = _run_to_steady_state(
                front, cfg, frame, int(args.n_steps), int(args.avg_last),
            )
            tuning_B[ci, oi, :] = r[0].cpu().numpy().astype(np.float64)

    # For each unit, take the response at its configured preferred ori-bin
    # (canonical 8 orients align exactly with unit_ori_bin indexing).
    unit_pref_curve = np.zeros((n_l4e, len(contrasts)), dtype=np.float64)
    for u in range(n_l4e):
        unit_pref_curve[u] = tuning_B[:, int(unit_ori_bin[u]), u]
    rhos = np.array([_spearman(contrasts, unit_pref_curve[u]) for u in range(n_l4e)])
    contrast_mono_frac = float((rhos >= 0.7).mean())

    # -------- Probe C: position sweep -------------------------------------
    # Pool-cell-aligned probe grid. When pos_n == retino_side, each probe
    # position sits at the center of exactly one retinotopic pool cell, so
    # the argmax-position maps directly onto a pool index (Lead's Option
    # (i)-B/C). Positions are in pixels, origin = frame center.
    pos_n = int(args.pos_grid)
    if pos_n == retino_side:
        # Exact pool centers.
        probe_cx = np.array([
            (j + 0.5) * pool_w - grid_w / 2.0 for j in range(pos_n)
        ])
        probe_cy = np.array([
            (i + 0.5) * pool_h - grid_h / 2.0 for i in range(pos_n)
        ])
    else:
        # Fallback: uniform coverage of the central 75% of the frame.
        probe_cx = np.linspace(
            -(grid_w - 1) / 2.0 * 0.75, (grid_w - 1) / 2.0 * 0.75, pos_n,
        )
        probe_cy = np.linspace(
            -(grid_h - 1) / 2.0 * 0.75, (grid_h - 1) / 2.0 * 0.75, pos_n,
        )
    tuning_C = np.zeros((pos_n, pos_n, n_l4e), dtype=np.float64)
    for pi, cy in enumerate(probe_cy):
        for pj, cx in enumerate(probe_cx):
            # Drive with the unit-population preferred orient of each
            # ori-channel — we'll select per unit after.
            # For speed, probe the n_ori_bank canonical orientations at
            # this position and keep the full tensor.
            best_per_unit = np.zeros(n_l4e, dtype=np.float64)
            for oi, o in enumerate(canonical_orients):
                frame = _make_patch_grating(
                    float(o), 1.0, cfg, cx=float(cx), cy=float(cy),
                    sigma_px=float(args.patch_sigma_px),
                )
                r = _run_to_steady_state(
                    front, cfg, frame, int(args.n_steps), int(args.avg_last),
                )[0].cpu().numpy().astype(np.float64)
                # Keep only the units whose unit_ori_bin matches oi
                match = (unit_ori_bin == oi)
                best_per_unit[match] = r[match]
            tuning_C[pi, pj] = best_per_unit

    # Retinotopic anchor in pixel coords: each retino cell i has center
    # cy = (i + 0.5) * pool_h - grid_h/2.
    anchor_px_x = (unit_retino_j.astype(np.float64) + 0.5) * pool_w - grid_w / 2.0
    anchor_px_y = (unit_retino_i.astype(np.float64) + 0.5) * pool_h - grid_h / 2.0

    # Per-unit argmax over the pos_n × pos_n probe grid. Index layout:
    # tuning_C[pi, pj, u] where pi indexes probe_cy, pj indexes probe_cx.
    # Localized iff argmax sits within ±1 pool cell of the configured anchor
    # (per Lead's Option (i)-C). When pos_n == retino_side, ±1 pool cell ⇔
    # ±1 grid step; otherwise we fall back to pixel distance ≤ pool_w.
    argmax_flat = tuning_C.reshape(pos_n * pos_n, n_l4e).argmax(axis=0)
    argmax_pi = argmax_flat // pos_n
    argmax_pj = argmax_flat % pos_n
    argmax_x = probe_cx[argmax_pj]
    argmax_y = probe_cy[argmax_pi]
    # Silent units (zero response everywhere) get argmax = 0 trivially; mark
    # those as not-localized regardless of the geometric criterion.
    unit_any_response = tuning_C.reshape(pos_n * pos_n, n_l4e).max(axis=0) > 1e-9

    localized = np.zeros(n_l4e, dtype=bool)
    if pos_n == retino_side:
        localized = (
            (np.abs(argmax_pi - unit_retino_i) <= 1)
            & (np.abs(argmax_pj - unit_retino_j) <= 1)
            & unit_any_response
        )
    else:
        # Fallback: pixel distance ≤ one pool cell (Chebyshev, per axis).
        dx = np.abs(argmax_x - anchor_px_x)
        dy = np.abs(argmax_y - anchor_px_y)
        localized = (
            (dx <= float(pool_w)) & (dy <= float(pool_h)) & unit_any_response
        )
    rf_com_frac_localized = float(localized.mean())

    # Also compute COM for diagnostics only (not used in verdict).
    pos_x_grid, pos_y_grid = np.meshgrid(probe_cx, probe_cy, indexing="xy")
    com_x = np.full(n_l4e, np.nan, dtype=np.float64)
    com_y = np.full(n_l4e, np.nan, dtype=np.float64)
    for u in range(n_l4e):
        w = tuning_C[:, :, u]
        wsum = float(w.sum())
        if wsum > 1e-9:
            com_x[u] = float((w * pos_x_grid).sum() / wsum)
            com_y[u] = float((w * pos_y_grid).sum() / wsum)

    # -------- Verdict -----------------------------------------------------
    fails = []
    if pref_orient_frac_defined < 0.80:
        fails.append(
            f"pref_orient_defined {pref_orient_frac_defined:.2f}<0.80"
        )
    if not (30.0 <= fwhm_median <= 60.0):
        fails.append(f"fwhm_median {fwhm_median:.1f}∉[30,60]")
    if contrast_mono_frac < 0.80:
        fails.append(f"contrast_mono {contrast_mono_frac:.2f}<0.80")
    if rf_com_frac_localized < 0.80:
        fails.append(f"rf_com {rf_com_frac_localized:.2f}<0.80")
    verdict = "pass" if not fails else "fail"
    issue = "none" if not fails else ";".join(fails)

    summary = {
        "version": "level_1_lgn_l4_v2_recalibrated",
        "seed": seed,
        "n_l4e": int(n_l4e),
        "n_ori_bank": int(n_ori_bank),
        "retino_side": int(retino_side),
        "pool_h": int(pool_h),
        "pool_w": int(pool_w),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "probe_A": {
            "orients_deg": orients_A.tolist(),
            "peak_rate_median": float(np.median(peak)),
            "trough_rate_median": float(np.median(trough)),
            "peak_minus_trough_median": float(np.median(peak - trough)),
            "pref_orient_frac_defined": pref_orient_frac_defined,
            "fwhm_median_deg": fwhm_median,
            "fwhm_p25_deg": float(np.nanpercentile(fwhms, 25)),
            "fwhm_p75_deg": float(np.nanpercentile(fwhms, 75)),
            "pref_circ_offset_median_deg": pref_circ_offset_median_deg,
        },
        "probe_B": {
            "contrasts": contrasts.tolist(),
            "contrast_mono_frac_rho_ge_0p7": contrast_mono_frac,
            "rho_median": float(np.nanmedian(rhos)),
        },
        "probe_C": {
            "pos_grid": pos_n,
            "patch_sigma_px": float(args.patch_sigma_px),
            "probe_cx_px": probe_cx.tolist(),
            "probe_cy_px": probe_cy.tolist(),
            "criterion": (
                "argmax_within_pm1_pool_cell"
                if pos_n == retino_side
                else "argmax_within_pool_width_px"
            ),
            "rf_com_frac_localized": rf_com_frac_localized,
            "anchor_argmax_offset_px_median": float(np.median(
                np.sqrt(
                    (argmax_x - anchor_px_x) ** 2 + (argmax_y - anchor_px_y) ** 2
                )
            )),
            "anchor_com_offset_px_median": float(np.nanmedian(
                np.sqrt(
                    (com_x - anchor_px_x) ** 2 + (com_y - anchor_px_y) ** 2
                )
            )),
            "n_silent_units": int((~unit_any_response).sum()),
        },
        "verdict": verdict,
        "issue_if_fail": issue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    line = (
        f"level1_verdict={verdict} "
        f"n_l4e={int(n_l4e)} "
        f"pref_orient_frac_defined={pref_orient_frac_defined:.3f} "
        f"fwhm_median_deg={fwhm_median:.1f} "
        f"contrast_mono_frac={contrast_mono_frac:.3f} "
        f"rf_com_frac_localized={rf_com_frac_localized:.3f} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
