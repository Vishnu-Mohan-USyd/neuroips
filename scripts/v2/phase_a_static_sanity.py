"""PHASE A — static-network sanity protocol (Task #74 pivot, 2026-04-21).

Diagnostic probe of a **freshly-constructed** ``V2Network`` at init —
no Phase-2 training, no plasticity, eval-only forward passes. Answers
the question: before we attribute any pathology to plasticity, does
the init-weight circuit even produce biologically plausible responses
to oriented gratings?

Protocol
--------
1. Orientation tuning — 12 orientations (15° spacing) × 20 trials ×
   500 ms probes (100 steps at dt=5 ms). Per L2/3 E unit compute
   (a) preferred orientation (argmax over 12), (b) circular FWHM
   (existing helper), (c) mean rate at preferred orient. Population:
   preferred-orient histogram (n_bins populated), rate distribution
   (median, 5-95th pct), fraction "responsive" (peak - baseline ≥
   0.05 Hz).
2. Contrast response — 6 contrasts {0.1, 0.2, 0.4, 0.6, 0.8, 1.0} × 3
   orientations × 10 trials. Per unit fit Naka-Rushton (Rmax·c^n /
   (c^n + c50^n)), report median R² across units with non-trivial
   tuning.
3. Surround suppression — 3 center sizes {4, 6, 8} × 2 orientations
   × 10 trials. SI = (R_center_only − R_center_surround) /
   R_center_only, per unit. Report SI median + fraction SI > 0.1.

Trial axis
----------
Trials are parallelised through the batch dimension with small
per-trial initial-state perturbations (low-variance Gaussian). The
network is deterministic given (state, frame), so trials vary by
starting state only — which is sufficient to sample dynamical
variability across transient trajectories during the 500 ms probe.

Output
------
* ``logs/task74/phase_a_static_sanity.json`` — per-gate dicts plus a
  ``summary`` block with the single-line DM metrics.
* Prints the one-line summary (format matches Lead's dispatch).

Wall budget
-----------
~20 min on CPU (single shell, no tmux). Pure diagnostic — no commits.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import (
    make_grating_frame,
    make_surround_grating_frame,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import NetworkStateV2
from src.v2_model.stimuli.feature_tokens import TokenBank


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _build_fresh_network(seed: int = 42) -> tuple[ModelConfig, V2Network]:
    """Fresh V2Network from ModelConfig defaults — no checkpoint load.

    Phase set to ``phase2`` so that generic-weight manifests are the
    ones probed in their init state, but no plasticity rule is called
    here (we only invoke ``net(frame, state)``).
    """
    cfg = ModelConfig(seed=seed)
    torch.manual_seed(seed)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")
    net.eval()
    return cfg, net


def _perturbed_initial_state(
    net: V2Network, batch_size: int, *, noise_std: float = 0.02,
    seed: int = 0,
) -> NetworkStateV2:
    """Zero state with tiny Gaussian noise on each rate field.

    Provides the per-trial variability that makes 'N trials' meaningful
    for a deterministic network. ``noise_std`` is small enough that it
    decays within the first few steps and does not bias the steady-state
    response.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    s = net.initial_state(batch_size=batch_size)
    for field in ("r_l4", "r_l23", "r_pv", "r_som", "r_h", "h_pv"):
        t = getattr(s, field)
        if t.numel() == 0:
            continue
        t.add_(
            torch.randn(t.shape, generator=gen, dtype=t.dtype) * noise_std
        ).clamp_(min=0.0)
    return s


@torch.no_grad()
def _probe_steady_state(
    net: V2Network, frame: Tensor, n_steps: int, *,
    avg_last: int, trial_seed: int,
) -> dict[str, np.ndarray]:
    """Run ``n_steps`` forward; average info rates over the last ``avg_last`` steps.

    Returns a dict of ``[B, n_units]`` numpy arrays for the relevant rate
    fields. The time-average is over the post-transient tail only.
    """
    B = frame.shape[0]
    state = _perturbed_initial_state(net, B, seed=trial_seed)
    buffers: dict[str, list[Tensor]] = {
        "r_l23": [], "r_som": [], "r_pv": [], "r_l4": [], "r_h": [],
    }
    for t in range(int(n_steps)):
        _xh, state, info = net(frame, state)
        if t >= n_steps - int(avg_last):
            for k in buffers:
                buffers[k].append(info[k].detach().clone())
    out: dict[str, np.ndarray] = {}
    for k, ts in buffers.items():
        stacked = torch.stack(ts, dim=0)                       # [Tavg, B, N]
        out[k] = stacked.mean(dim=0).cpu().numpy()             # [B, N]
    return out


# ---------------------------------------------------------------------------
# Gate: orientation tuning (12 × 20 × 500 ms)
# ---------------------------------------------------------------------------


def _fwhm_from_tuning(curve: np.ndarray) -> float:
    """Circular FWHM in degrees (180° period) for a tuning curve."""
    if curve.max() <= 1e-9:
        return 180.0
    c = curve - curve.min()
    half = c.max() / 2.0
    peak_idx = int(np.argmax(c))
    shifted = np.roll(c, -peak_idx)
    step = 180.0 / len(c)
    above = shifted >= half
    if not above[0]:
        return 180.0
    n = len(c)
    right = 0
    for k in range(1, n // 2 + 1):
        if above[k % n]:
            right = k
        else:
            break
    left = 0
    for k in range(1, n // 2 + 1):
        if above[(-k) % n]:
            left = k
        else:
            break
    return float((right + left + 1) * step)


@torch.no_grad()
def orientation_tuning(
    net: V2Network, cfg: ModelConfig, *,
    n_orients: int = 12, n_trials: int = 20,
    n_steps: int = 100, avg_last: int = 50,
    contrast: float = 1.0, noise_floor_hz: float = 0.05,
) -> dict[str, Any]:
    """Per-L2/3 E unit tuning curve + population metrics."""
    orientations = np.linspace(
        0.0, 180.0, n_orients, endpoint=False,
    ).astype(np.float64)

    # Tuning curve accumulators: [n_orients, n_l23_e], [n_orients, n_l23_som].
    tuning_l23 = np.zeros((n_orients, cfg.arch.n_l23_e), dtype=np.float64)
    tuning_som = np.zeros((n_orients, cfg.arch.n_l23_som), dtype=np.float64)

    # Blank-frame baseline (for responsive-unit definition).
    blank = torch.full(
        (n_trials, 1, cfg.arch.grid_h, cfg.arch.grid_w),
        fill_value=0.5, dtype=torch.float32,
    )
    blank_probe = _probe_steady_state(
        net, blank, n_steps, avg_last=avg_last, trial_seed=999,
    )
    baseline_l23 = blank_probe["r_l23"].mean(axis=0)           # [n_l23_e]

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), contrast, cfg, batch_size=n_trials,
        )
        probe = _probe_steady_state(
            net, frame, n_steps, avg_last=avg_last, trial_seed=oi,
        )
        tuning_l23[oi] = probe["r_l23"].mean(axis=0)            # avg over trials
        tuning_som[oi] = probe["r_som"].mean(axis=0)

    # Per-unit stats.
    preferred_idx = np.argmax(tuning_l23, axis=0)              # [n_l23_e]
    peak_rate = tuning_l23[preferred_idx, np.arange(cfg.arch.n_l23_e)]
    fwhms = np.array(
        [_fwhm_from_tuning(tuning_l23[:, u]) for u in range(cfg.arch.n_l23_e)],
    )

    # Responsive units: peak rises clearly above blank baseline.
    responsive = (peak_rate - baseline_l23) > noise_floor_hz
    frac_resp = float(responsive.mean())

    # Preferred-bin coverage: how many of the 12 bins are populated?
    pref_hist = np.bincount(preferred_idx, minlength=n_orients)
    n_pref_bins = int((pref_hist > 0).sum())

    # r_som: aggregate rate, not tuning shape — collapsed over all orients.
    r_som_mean_per_unit = tuning_som.mean(axis=0)              # [n_l23_som]
    r_som_mean = float(r_som_mean_per_unit.mean())

    return {
        "gate": "orientation_tuning",
        "orientations_deg": orientations.tolist(),
        "tuning_l23_mean_peak": float(peak_rate.mean()),
        "rate_median": float(np.median(peak_rate)),
        "rate_p05": float(np.quantile(peak_rate, 0.05)),
        "rate_p95": float(np.quantile(peak_rate, 0.95)),
        "rate_std": float(peak_rate.std()),
        "rate_fraction_responsive": frac_resp,
        "baseline_l23_mean": float(baseline_l23.mean()),
        "fwhm_median_deg": float(np.median(fwhms)),
        "fwhm_p25_deg": float(np.quantile(fwhms, 0.25)),
        "fwhm_p75_deg": float(np.quantile(fwhms, 0.75)),
        "preferred_bin_histogram": pref_hist.tolist(),
        "n_preferred_bins_populated": n_pref_bins,
        "r_som_mean": r_som_mean,
        "r_som_max_unit_mean": float(r_som_mean_per_unit.max()),
    }


# ---------------------------------------------------------------------------
# Gate: contrast response (6 × 3 × 10) with per-unit Naka-Rushton
# ---------------------------------------------------------------------------


def _fit_naka_rushton(
    contrast: np.ndarray, response: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit R(c) = Rmax·c^n/(c^n + c50^n); returns (Rmax, n, c50, r²)."""
    try:
        from scipy.optimize import curve_fit                   # type: ignore

        def _nr(c, rmax, n, c50):
            return rmax * (c ** n) / ((c ** n) + (c50 ** n) + 1e-8)

        popt, _ = curve_fit(
            _nr, contrast, response,
            p0=[max(float(response.max()), 1e-3), 2.0, 0.3],
            bounds=([0.0, 0.1, 0.01], [10.0, 10.0, 1.0]),
            maxfev=2000,
        )
        rmax, n, c50 = [float(p) for p in popt]
        pred = _nr(contrast, rmax, n, c50)
    except Exception:
        rmax = float(response.max())
        c50, n = 0.3, 2.0
        pred = rmax * (contrast ** n) / ((contrast ** n) + (c50 ** n) + 1e-8)
    ss_res = float(((response - pred) ** 2).sum())
    ss_tot = float(((response - response.mean()) ** 2).sum()) + 1e-12
    return rmax, n, c50, 1.0 - ss_res / ss_tot


@torch.no_grad()
def contrast_response(
    net: V2Network, cfg: ModelConfig, *,
    contrasts: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    orientations_deg: tuple[float, ...] = (0.0, 45.0, 90.0),
    n_trials: int = 10, n_steps: int = 100, avg_last: int = 50,
    min_peak_hz: float = 0.05,
) -> dict[str, Any]:
    """Per-unit Naka-Rushton fit across contrasts (pooled over orients)."""
    n_c = len(contrasts)
    # Per-unit contrast curves averaged over orientations and trials.
    curves = np.zeros((n_c, cfg.arch.n_l23_e), dtype=np.float64)
    for ci, c in enumerate(contrasts):
        per_ori = np.zeros((len(orientations_deg), cfg.arch.n_l23_e))
        for oi, ori in enumerate(orientations_deg):
            frame = make_grating_frame(
                float(ori), float(c), cfg, batch_size=n_trials,
            )
            probe = _probe_steady_state(
                net, frame, n_steps, avg_last=avg_last,
                trial_seed=100 + ci * 17 + oi,
            )
            per_ori[oi] = probe["r_l23"].mean(axis=0)
        curves[ci] = per_ori.mean(axis=0)

    c_arr = np.asarray(contrasts, dtype=np.float64)
    r2s: list[float] = []
    rmaxs: list[float] = []
    c50s: list[float] = []
    ns: list[float] = []
    for u in range(cfg.arch.n_l23_e):
        response = curves[:, u]
        if response.max() < min_peak_hz:
            continue                                           # skip silent unit
        rmax, n_exp, c50, r2 = _fit_naka_rushton(c_arr, response)
        r2s.append(r2)
        rmaxs.append(rmax)
        c50s.append(c50)
        ns.append(n_exp)

    r2_arr = np.asarray(r2s, dtype=np.float64) if r2s else np.array([np.nan])
    return {
        "gate": "contrast_response",
        "contrasts": list(c_arr.tolist()),
        "orientations_deg": list(orientations_deg),
        "n_units_fit": len(r2s),
        "n_units_total": int(cfg.arch.n_l23_e),
        "r2_median": float(np.nanmedian(r2_arr)),
        "r2_mean": float(np.nanmean(r2_arr)),
        "r2_frac_above_0.7": float(np.mean(r2_arr > 0.7)) if r2s else 0.0,
        "rmax_median": float(np.median(rmaxs)) if rmaxs else float("nan"),
        "c50_median": float(np.median(c50s)) if c50s else float("nan"),
        "n_exponent_median": float(np.median(ns)) if ns else float("nan"),
    }


# ---------------------------------------------------------------------------
# Gate: surround suppression (3 × 2 × 10)
# ---------------------------------------------------------------------------


@torch.no_grad()
def surround_suppression(
    net: V2Network, cfg: ModelConfig, *,
    center_radii: tuple[int, ...] = (4, 6, 8),
    orientations_deg: tuple[float, ...] = (0.0, 90.0),
    n_trials: int = 10, contrast: float = 1.0,
    n_steps: int = 100, avg_last: int = 50,
    si_threshold: float = 0.1, min_center_hz: float = 0.05,
) -> dict[str, Any]:
    """SI = (R_center_only − R_full) / R_center_only per L2/3 E unit.

    Pools across center sizes and orientations for population summary;
    also reports per-(size, orient) condition SI medians.
    """
    per_cond: list[dict[str, Any]] = []
    si_pool: list[np.ndarray] = []
    for radius in center_radii:
        for ori in orientations_deg:
            center_frame = make_surround_grating_frame(
                float(ori), contrast, cfg,
                center_radius=int(radius), include_surround=False,
                batch_size=n_trials,
            )
            full_frame = make_surround_grating_frame(
                float(ori), contrast, cfg,
                center_radius=int(radius), include_surround=True,
                batch_size=n_trials,
            )
            probe_c = _probe_steady_state(
                net, center_frame, n_steps, avg_last=avg_last,
                trial_seed=int(radius) * 101 + int(ori),
            )
            probe_f = _probe_steady_state(
                net, full_frame, n_steps, avg_last=avg_last,
                trial_seed=int(radius) * 103 + int(ori) + 1,
            )
            r_c = probe_c["r_l23"].mean(axis=0)                # [n_l23_e]
            r_f = probe_f["r_l23"].mean(axis=0)
            # Only score units that have non-trivial center response.
            valid = r_c > min_center_hz
            denom = np.where(valid, r_c, np.nan)
            si = (r_c - r_f) / denom
            si_valid = si[valid]
            per_cond.append({
                "center_radius_px": int(radius),
                "orientation_deg": float(ori),
                "n_valid_units": int(valid.sum()),
                "si_median": float(np.nanmedian(si))
                    if si_valid.size else float("nan"),
                "si_frac_above_threshold": float(
                    np.mean(si_valid > si_threshold),
                ) if si_valid.size else 0.0,
            })
            if si_valid.size:
                si_pool.append(si_valid)

    pooled = (
        np.concatenate(si_pool) if si_pool else np.array([np.nan])
    )
    return {
        "gate": "surround_suppression",
        "center_radii_px": list(center_radii),
        "orientations_deg": list(orientations_deg),
        "si_median_pooled": float(np.nanmedian(pooled)),
        "si_mean_pooled": float(np.nanmean(pooled)),
        "si_frac_above_0.1_pooled": float(np.mean(pooled > si_threshold)),
        "per_condition": per_cond,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    out_dir = Path("logs/task74")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase_a_static_sanity.json"

    print(f"[phase_a] building fresh V2Network (seed=42, no training)")
    cfg, net = _build_fresh_network(seed=42)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[phase_a] network built — n_params={n_params}")

    # --- Gate 1: orientation tuning --------------------------------------
    print(f"[phase_a] orientation tuning (12 × 20 × 500 ms) ...")
    tgate = time.time()
    tune = orientation_tuning(net, cfg)
    print(
        f"  rate_median={tune['rate_median']:.3f} hz  "
        f"fwhm_median={tune['fwhm_median_deg']:.1f}°  "
        f"frac_resp={tune['rate_fraction_responsive']:.2f}  "
        f"n_pref_bins={tune['n_preferred_bins_populated']}/12  "
        f"r_som={tune['r_som_mean']:.2f} hz  "
        f"[{time.time()-tgate:.1f}s]"
    )

    # --- Gate 2: contrast response ---------------------------------------
    print(f"[phase_a] contrast response (6 × 3 × 10) ...")
    tgate = time.time()
    contrast = contrast_response(net, cfg)
    print(
        f"  r2_median={contrast['r2_median']:.3f}  "
        f"r2_frac>0.7={contrast['r2_frac_above_0.7']:.2f}  "
        f"n_units_fit={contrast['n_units_fit']}/{contrast['n_units_total']}  "
        f"[{time.time()-tgate:.1f}s]"
    )

    # --- Gate 3: surround suppression ------------------------------------
    print(f"[phase_a] surround suppression (3 × 2 × 10) ...")
    tgate = time.time()
    surround = surround_suppression(net, cfg)
    print(
        f"  si_median={surround['si_median_pooled']:.3f}  "
        f"si_frac>0.1={surround['si_frac_above_0.1_pooled']:.2f}  "
        f"[{time.time()-tgate:.1f}s]"
    )

    # --- Summary line (matches Lead's dispatch format) --------------------
    summary = {
        "rate_median": tune["rate_median"],
        "rate_fraction_responsive": tune["rate_fraction_responsive"],
        "fwhm_median_deg": tune["fwhm_median_deg"],
        "n_preferred_bins": tune["n_preferred_bins_populated"],
        "contrast_R2_median": contrast["r2_median"],
        "suppression_index_median": surround["si_median_pooled"],
        "suppression_frac_above_0.1": surround["si_frac_above_0.1_pooled"],
        "r_som_mean_hz": tune["r_som_mean"],
    }
    summary_line = (
        f"PhaseA "
        f"rate_median={summary['rate_median']:.3f} "
        f"rate_fraction_responsive={summary['rate_fraction_responsive']:.3f} "
        f"fwhm_median={summary['fwhm_median_deg']:.1f} "
        f"n_preferred_bins={summary['n_preferred_bins']} "
        f"contrast_R2_median={summary['contrast_R2_median']:.3f} "
        f"suppression_index_median={summary['suppression_index_median']:.3f} "
        f"suppression_frac_above_0.1={summary['suppression_frac_above_0.1']:.3f} "
        f"r_som={summary['r_som_mean_hz']:.2f}"
    )

    # --- Write JSON -------------------------------------------------------
    result = {
        "version": "phase_a_v1",
        "seed": 42,
        "n_params": int(n_params),
        "wall_seconds": float(time.time() - t0),
        "orientation_tuning": tune,
        "contrast_response": contrast,
        "surround_suppression": surround,
        "summary": summary,
        "summary_line": summary_line,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"[phase_a] wrote {out_path}  wall={result['wall_seconds']:.1f}s")
    print(summary_line)


if __name__ == "__main__":
    main()
