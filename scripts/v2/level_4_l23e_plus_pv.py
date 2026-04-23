"""Level 4 component validation — L23E + W_rec + PV loop on top of Fix K.

Per Lead's bottom-up validation protocol (Task #74). Scope: Level 3 plus
the co-simulated L23PV population. SOM output and H feedback remain
zeroed. Context bias / som_gain disabled.

Co-simulation
-------------
Sync-Euler (pre-update siblings only):
    r_l23_new = L23E.forward(
        l4_input          = r_l4_ss,                     # Fix-K sparse FF
        l23_recurrent_input = state.r_l23_prev,
        som_input         = zeros,
        pv_input          = state.r_pv_prev,             # <<< L4 addition
        h_apical_input    = zeros,
        context_bias      = zeros,
        state             = state.r_l23_prev,
        som_gain          = None,
    )
    r_pv_new  = L23PV.forward(pre_input=state.r_l23_prev, state=state.r_pv_prev)

Probes
------
1. Orient sweep: 12 orients × n_trials trials at contrast=1. Per-unit
   preferred orient, FWHM, peak/trough/mean. Co-measures mean PV rate.
2. Contrast sweep: 6 contrasts × 12 orients × n_trials_c trials. Per-unit
   contrast response extracted at each unit's preferred orient. Naka-
   Rushton ``R_max · c^n / (c^n + c50^n)`` fit by scipy (n, c50 bounded);
   per-unit R² reported. Gain-compression ratio = mean r(c=1) / r(c=0.1).

Pass criteria (seven):
  * ``n_l23e_defined_frac`` ≥ 0.80
  * ``fwhm_median_deg`` ∈ [30°, 80°]
  * ≥ 8 / 12 preferred-orient bins populated
  * L23E ``rate_mean`` ∈ [0.5, 10] Hz at contrast 1
  * L23PV ``rate_mean`` ∈ [5, 50] Hz at contrast 1
  * contrast Naka-Rushton R² median ≥ 0.7
  * gain compression r(1) / r(0.1) < 8 (genuine divisive normalization)

DM:
  level4_verdict=<pass/fail> n_l23e_defined=<#> fwhm_median=<#>°
  n_preferred_bins=<#>/12 rate_l23e_mean=<#> rate_pv_mean=<#>
  contrast_R2_median=<#> gain_compression=<#> issue_if_fail=<short>
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

try:
    from scipy.optimize import curve_fit as _curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from scripts.v2._gates_common import make_grating_frame
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circular_fwhm_deg(tuning: np.ndarray, orients_deg: np.ndarray) -> float:
    peak = float(tuning.max())
    trough = float(tuning.min())
    if peak <= trough + 1e-9:
        return float("nan")
    half = trough + 0.5 * (peak - trough)
    above = tuning >= half
    if int(above.sum()) == 0:
        return float("nan")
    bin_width = 180.0 / float(len(orients_deg))
    return float(int(above.sum()) * bin_width)


def _naka_rushton(c: np.ndarray, R_max: float, n: float, c50: float) -> np.ndarray:
    return R_max * np.power(c, n) / (np.power(c, n) + np.power(c50, n))


def _fit_naka_rushton_r2(
    contrasts: np.ndarray, response: np.ndarray,
) -> Optional[float]:
    """Fit R = R_max · c^n / (c^n + c50^n); return R².

    Returns None on fit failure or flat response (insufficient dynamic
    range). n bounded to [0.5, 5], c50 bounded to [0.01, 2]."""
    if not _HAS_SCIPY:
        return None
    if response.max() - response.min() < 1e-6:
        return None
    try:
        popt, _ = _curve_fit(
            _naka_rushton, contrasts, response,
            p0=[float(response.max()), 2.0, 0.3],
            bounds=([1e-6, 0.5, 0.01],
                    [np.inf, 5.0, 2.0]),
            maxfev=5000,
        )
        yhat = _naka_rushton(contrasts, *popt)
        ss_res = float(((response - yhat) ** 2).sum())
        ss_tot = float(((response - response.mean()) ** 2).sum())
        if ss_tot < 1e-12:
            return None
        return 1.0 - ss_res / ss_tot
    except Exception:  # noqa: BLE001 — scipy raises many subclasses
        return None


@torch.no_grad()
def _drive_lgn_l4_to_steady(
    net: V2Network, cfg: ModelConfig, frame: Tensor,
    n_steps: int, avg_last: int,
) -> Tensor:
    B = frame.shape[0]
    state = initial_state(cfg, batch_size=B)
    buf: list[Tensor] = []
    for t in range(int(n_steps)):
        _feat, r_l4, _ = net.lgn_l4(frame, state)
        state = state._replace(r_l4=r_l4)
        if t >= n_steps - int(avg_last):
            buf.append(r_l4)
    return torch.stack(buf, dim=0).mean(dim=0)


@torch.no_grad()
def _drive_l23e_pv_from_l4(
    net: V2Network, r_l4_ss: Tensor, n_steps: int, avg_last: int,
) -> tuple[Tensor, Tensor]:
    """Co-simulate L23E + L23PV with sync-Euler (PV sees prev-step L23E).

    Returns (r_l23_mean, r_pv_mean) each averaged over last ``avg_last``
    steps.
    """
    B = r_l4_ss.shape[0]
    device = r_l4_ss.device
    dtype = r_l4_ss.dtype
    l23e = net.l23_e
    l23_pv = net.l23_pv

    def _zeros(n: int) -> Tensor:
        return torch.zeros(B, n, dtype=dtype, device=device)

    zeros_som = _zeros(l23e.n_som)
    zeros_fb = _zeros(l23e.n_h_e)
    zeros_bias = _zeros(l23e.n_units)

    r_l23 = torch.zeros(B, l23e.n_units, dtype=dtype, device=device)
    r_pv = torch.zeros(B, l23_pv.n_units, dtype=dtype, device=device)

    buf_l23: list[Tensor] = []
    buf_pv: list[Tensor] = []
    for t in range(int(n_steps)):
        # Snapshot pre-update state so the L23E and PV updates both read
        # sibling's *previous* rate (strict sync-Euler; matches V2Network).
        r_l23_prev = r_l23
        r_pv_prev = r_pv
        r_l23, _ = l23e(
            l4_input=r_l4_ss,
            l23_recurrent_input=r_l23_prev,
            som_input=zeros_som,
            pv_input=r_pv_prev,
            h_apical_input=zeros_fb,
            context_bias=zeros_bias,
            state=r_l23_prev,
            som_gain=None,
        )
        r_pv, _ = l23_pv(r_l23_prev, r_pv_prev)
        if t >= n_steps - int(avg_last):
            buf_l23.append(r_l23)
            buf_pv.append(r_pv)
    return (
        torch.stack(buf_l23, dim=0).mean(dim=0),
        torch.stack(buf_pv, dim=0).mean(dim=0),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-orients", type=int, default=12)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-trials-contrast", type=int, default=5)
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--n-steps-l4", type=int, default=40)
    p.add_argument("--avg-last-l4", type=int, default=20)
    p.add_argument("--n-steps-l23", type=int, default=200,
                   help="Longer than Level 3: recurrent + PV loop with fast "
                        "PV (τ=5 ms) — ~10 τ_L23 + PV transient = 200 steps.")
    p.add_argument("--avg-last-l23", type=int, default=40)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if not _HAS_SCIPY:
        print("[level4] WARNING: scipy unavailable; Naka-Rushton R² → NaN")

    seed = int(args.seed)
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    net = V2Network(cfg, token_bank=None, seed=seed, device="cpu")
    net.eval()

    n_l23e = cfg.arch.n_l23_e
    n_pv = cfg.arch.n_l23_pv
    orientations = np.linspace(0.0, 180.0, int(args.n_orients), endpoint=False)

    # -------- Probe A: orient sweep at contrast=1 -------------------------
    tuning_l23 = np.zeros((int(args.n_orients), n_l23e), dtype=np.float64)
    tuning_pv_mean = np.zeros(int(args.n_orients), dtype=np.float64)

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), 1.0, cfg, batch_size=int(args.n_trials),
        )
        r_l4_ss = _drive_lgn_l4_to_steady(
            net, cfg, frame, int(args.n_steps_l4), int(args.avg_last_l4),
        )
        r_l23, r_pv = _drive_l23e_pv_from_l4(
            net, r_l4_ss, int(args.n_steps_l23), int(args.avg_last_l23),
        )
        tuning_l23[oi] = r_l23.mean(dim=0).cpu().numpy().astype(np.float64)
        tuning_pv_mean[oi] = float(r_pv.mean().cpu())

    peak = tuning_l23.max(axis=0)
    trough = tuning_l23.min(axis=0)
    mean_unit = tuning_l23.mean(axis=0)
    pref_idx = tuning_l23.argmax(axis=0)

    well_defined = peak > (2.0 * trough + 0.01)
    n_defined_frac = float(well_defined.mean())
    fwhms = np.array([
        _circular_fwhm_deg(tuning_l23[:, u], orientations) for u in range(n_l23e)
    ])
    fwhm_median = float(np.nanmedian(fwhms))

    pref_hist = np.bincount(pref_idx, minlength=int(args.n_orients))
    bin_threshold = max(1, int(math.ceil(0.05 * n_l23e)))
    n_pref_bins_5pct = int((pref_hist >= bin_threshold).sum())

    rate_l23e_mean = float(mean_unit.mean())
    rate_l23e_max = float(mean_unit.max())
    rate_l23e_min = float(mean_unit.min())
    # Highest-contrast PV rate = population mean across orientations (PV
    # fan-in is dense → relatively flat across orient).
    rate_pv_mean = float(tuning_pv_mean.mean())
    n_silent = int((peak <= 1e-9).sum())
    n_runaway = int((mean_unit > 100.0).sum())

    # -------- Probe B: contrast sweep -------------------------------------
    contrasts = np.array([float(c) for c in args.contrasts])
    contrast_tuning = np.zeros(
        (len(contrasts), int(args.n_orients), n_l23e), dtype=np.float64,
    )
    for ci, c in enumerate(contrasts):
        for oi, ori in enumerate(orientations):
            frame = make_grating_frame(
                float(ori), float(c), cfg,
                batch_size=int(args.n_trials_contrast),
            )
            r_l4_ss = _drive_lgn_l4_to_steady(
                net, cfg, frame, int(args.n_steps_l4), int(args.avg_last_l4),
            )
            r_l23, _ = _drive_l23e_pv_from_l4(
                net, r_l4_ss, int(args.n_steps_l23), int(args.avg_last_l23),
            )
            contrast_tuning[ci, oi] = (
                r_l23.mean(dim=0).cpu().numpy().astype(np.float64)
            )

    # Per-unit response at its preferred orient (from Probe A).
    per_unit_cr = np.zeros((n_l23e, len(contrasts)), dtype=np.float64)
    for u in range(n_l23e):
        per_unit_cr[u] = contrast_tuning[:, int(pref_idx[u]), u]

    r2_list: list[float] = []
    for u in range(n_l23e):
        r2 = _fit_naka_rushton_r2(contrasts, per_unit_cr[u])
        if r2 is not None:
            r2_list.append(r2)
    contrast_r2_median = float(np.median(r2_list)) if r2_list else float("nan")
    n_r2_fits = int(len(r2_list))

    # Gain compression ratio r(c=1)/r(c=0.1) — mean across units
    # (skip silent units where r(0.1) ≈ 0 to avoid div-by-zero blow-up).
    idx_lo = int(np.argmin(np.abs(contrasts - 0.1)))
    idx_hi = int(np.argmin(np.abs(contrasts - 1.0)))
    r_lo = per_unit_cr[:, idx_lo]
    r_hi = per_unit_cr[:, idx_hi]
    valid = r_lo > 0.05                                # ≥ 0.05 Hz floor
    if valid.any():
        gain_compression = float(np.median(r_hi[valid] / r_lo[valid]))
    else:
        gain_compression = float("inf")

    # -------- Verdict -----------------------------------------------------
    fails: list[str] = []
    if n_defined_frac < 0.80:
        fails.append(f"n_l23e_defined {n_defined_frac:.2f}<0.80")
    if not (30.0 <= fwhm_median <= 80.0):
        fails.append(f"fwhm_median {fwhm_median:.1f}∉[30,80]")
    if n_pref_bins_5pct < 8:
        fails.append(f"n_pref_bins_5pct {n_pref_bins_5pct}/12<8")
    if not (0.5 <= rate_l23e_mean <= 10.0):
        fails.append(f"rate_l23e_mean {rate_l23e_mean:.3f}∉[0.5,10]")
    if not (5.0 <= rate_pv_mean <= 50.0):
        fails.append(f"rate_pv_mean {rate_pv_mean:.3f}∉[5,50]")
    if not (contrast_r2_median == contrast_r2_median) or contrast_r2_median < 0.7:
        fails.append(f"contrast_R2_median {contrast_r2_median:.3f}<0.7")
    if not math.isfinite(gain_compression) or gain_compression >= 8.0:
        fails.append(f"gain_compression {gain_compression:.2f}≥8")
    if n_silent > 0:
        fails.append(f"n_silent={n_silent}")
    if n_runaway > 0:
        fails.append(f"n_runaway={n_runaway}")

    verdict = "pass" if not fails else "fail"
    issue = "none" if not fails else ";".join(fails)

    summary = {
        "version": "level_4_l23e_plus_pv_v1",
        "seed": seed,
        "n_l23e": int(n_l23e),
        "n_pv": int(n_pv),
        "n_orients": int(args.n_orients),
        "n_trials": int(args.n_trials),
        "n_trials_contrast": int(args.n_trials_contrast),
        "contrasts": contrasts.tolist(),
        "n_steps_l4": int(args.n_steps_l4),
        "n_steps_l23": int(args.n_steps_l23),
        "orients_deg": orientations.tolist(),
        "stats": {
            "n_l23e_defined_frac": n_defined_frac,
            "fwhm_median_deg": fwhm_median,
            "fwhm_p25_deg": float(np.nanpercentile(fwhms, 25)),
            "fwhm_p75_deg": float(np.nanpercentile(fwhms, 75)),
            "pref_hist": pref_hist.tolist(),
            "bin_threshold_units": bin_threshold,
            "n_pref_bins_5pct": n_pref_bins_5pct,
            "rate_l23e_mean": rate_l23e_mean,
            "rate_l23e_min": rate_l23e_min,
            "rate_l23e_max": rate_l23e_max,
            "rate_pv_mean": rate_pv_mean,
            "rate_pv_across_orients": tuning_pv_mean.tolist(),
            "contrast_R2_median": contrast_r2_median,
            "n_r2_fits": n_r2_fits,
            "gain_compression": gain_compression,
            "n_silent": n_silent,
            "n_runaway": n_runaway,
            "peak_rate_median": float(np.median(peak)),
            "trough_rate_median": float(np.median(trough)),
        },
        "verdict": verdict,
        "issue_if_fail": issue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    line = (
        f"level4_verdict={verdict} "
        f"n_l23e_defined={n_defined_frac:.3f} "
        f"fwhm_median={fwhm_median:.1f}° "
        f"n_preferred_bins={n_pref_bins_5pct}/12 "
        f"rate_l23e_mean={rate_l23e_mean:.3f} "
        f"rate_pv_mean={rate_pv_mean:.3f} "
        f"contrast_R2_median={contrast_r2_median:.3f} "
        f"gain_compression={gain_compression:.2f} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
