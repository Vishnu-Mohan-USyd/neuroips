"""Level 6 component validation — enable H (HE + HPV) on top of Fix-K + Fix-L2
+ Fix-M substrate.

Per Lead's bottom-up validation protocol (Task #74). Scope: Level 5 plus the
co-simulated HE and HPV populations and the full H feedback loop into
L2/3 — W_fb_apical (H → L23E apical) and W_fb_som (H → SOM). Context bias
and som_gain remain disabled.

This completes the substrate test: all five intrinsic populations active,
but plasticity off (everything at init).

Co-simulation (strict sync-Euler)
---------------------------------
    r_l23_prev, r_pv_prev, r_som_prev, r_h_prev, r_hpv_prev = <snapshot>
    r_l23, _ = L23E.forward(
        l4_input            = r_l4_ss,
        l23_recurrent_input = r_l23_prev,
        som_input           = r_som_prev,
        pv_input            = r_pv_prev,
        h_apical_input      = r_h_prev,          # <<< Level-6 addition (apical)
        context_bias        = zeros,
        state               = r_l23_prev,
        som_gain            = None,
    )
    r_pv, _  = L23PV.forward(r_l23_prev, r_pv_prev)
    r_som, _ = L23SOM.forward(
        l23e_input            = r_l23_prev,
        h_som_feedback_input  = r_h_prev,        # <<< Level-6 addition (SOM)
        state                 = r_som_prev,
    )
    r_h, _   = HE.forward(
        l23_input         = r_l23_prev,
        h_recurrent_input = r_h_prev,
        h_pv_input        = r_hpv_prev,
        context_bias      = zeros,
        state             = r_h_prev,
    )
    r_hpv, _ = HPV.forward(r_h_prev, r_hpv_prev)

Probes
------
1. Orient sweep at contrast=1, 12 orients × n_trials trials.
2. Per-L23E-unit FWHM + preferred-orient stats (as Levels 2–5).
3. Per-HE-unit FWHM + preferred-orient stats (new).
4. HPV and SOM population rates (sanity-band checks).

Pass criteria (seven gated)
---------------------------
  * ``n_l23e_defined_frac`` ≥ 0.80 (``peak > 2·trough + 0.01``)
  * L23E ``fwhm_median_deg`` ∈ [30°, 80°]
  * ≥ 8 / 12 L23E preferred-orient bins populated (each ≥ 5% n_l23_e)
  * L23E ``rate_mean`` ∈ [0.5, 10] Hz
  * H ``rate_mean``    ∈ [0.05, 5] Hz
  * HPV ``rate_mean``  ∈ [5, 50] Hz
  * SOM ``rate_mean``  ∈ [0.3, 20] Hz  (relaxed floor per Lead)
  * no silent L23E unit / no runaway in any population (> 100 Hz mean)

Diagnostics (reported, not gated)
---------------------------------
  * ``h_snr_median``    — (max−min)/mean tuning SNR across H units
  * ``h_fwhm_median``   — FWHM of per-H tuning
  * ``h_n_pref_bins_5pct``
  * ``delta_rate_l23e_vs_L5``, ``delta_fwhm_l23e_vs_L5``

DM::
  level6_verdict=<pass/fail> n_l23e_defined=<#> fwhm_l23e=<#>°
    n_bins_l23e=<#>/12 rate_l23e=<#> rate_h=<#> rate_hpv=<#> rate_som=<#>
    h_snr_median=<#> Δrate_l23e_vs_L5=<#> Δfwhm_vs_L5=<#> issue_if_fail=<short>
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
    n_above = int(above.sum())
    if n_above == 0:
        return float("nan")
    bin_width = 180.0 / float(len(orients_deg))
    return float(n_above * bin_width)


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
def _drive_full_substrate_from_l4(
    net: V2Network, r_l4_ss: Tensor, n_steps: int, avg_last: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Co-simulate L23E + L23PV + L23SOM + HE + HPV in strict sync-Euler.

    Returns ``(r_l23, r_pv, r_som, r_h, r_hpv)`` each averaged over the
    final ``avg_last`` Euler steps. Context bias zeroed, som_gain=None,
    W_fb_apical (H → L23E) and W_fb_som (H → SOM) live at init.
    """
    B = r_l4_ss.shape[0]
    device = r_l4_ss.device
    dtype = r_l4_ss.dtype
    l23e = net.l23_e
    l23_pv = net.l23_pv
    l23_som = net.l23_som
    h_e = net.h_e
    h_pv = net.h_pv

    def _zeros(n: int) -> Tensor:
        return torch.zeros(B, n, dtype=dtype, device=device)

    zeros_bias_l23 = _zeros(l23e.n_units)
    zeros_bias_h = _zeros(h_e.n_units)

    r_l23 = torch.zeros(B, l23e.n_units, dtype=dtype, device=device)
    r_pv = torch.zeros(B, l23_pv.n_units, dtype=dtype, device=device)
    r_som = torch.zeros(B, l23_som.n_units, dtype=dtype, device=device)
    r_h = torch.zeros(B, h_e.n_units, dtype=dtype, device=device)
    r_hpv = torch.zeros(B, h_pv.n_units, dtype=dtype, device=device)

    buf_l23: list[Tensor] = []
    buf_pv: list[Tensor] = []
    buf_som: list[Tensor] = []
    buf_h: list[Tensor] = []
    buf_hpv: list[Tensor] = []

    for t in range(int(n_steps)):
        r_l23_prev = r_l23
        r_pv_prev = r_pv
        r_som_prev = r_som
        r_h_prev = r_h
        r_hpv_prev = r_hpv

        r_l23, _ = l23e(
            l4_input=r_l4_ss,
            l23_recurrent_input=r_l23_prev,
            som_input=r_som_prev,
            pv_input=r_pv_prev,
            h_apical_input=r_h_prev,
            context_bias=zeros_bias_l23,
            state=r_l23_prev,
            som_gain=None,
        )
        r_pv, _ = l23_pv(r_l23_prev, r_pv_prev)
        r_som, _ = l23_som(
            l23e_input=r_l23_prev,
            h_som_feedback_input=r_h_prev,
            state=r_som_prev,
        )
        r_h, _ = h_e(
            l23_input=r_l23_prev,
            h_recurrent_input=r_h_prev,
            h_pv_input=r_hpv_prev,
            context_bias=zeros_bias_h,
            state=r_h_prev,
        )
        r_hpv, _ = h_pv(r_h_prev, r_hpv_prev)

        if t >= n_steps - int(avg_last):
            buf_l23.append(r_l23)
            buf_pv.append(r_pv)
            buf_som.append(r_som)
            buf_h.append(r_h)
            buf_hpv.append(r_hpv)

    return (
        torch.stack(buf_l23, dim=0).mean(dim=0),
        torch.stack(buf_pv, dim=0).mean(dim=0),
        torch.stack(buf_som, dim=0).mean(dim=0),
        torch.stack(buf_h, dim=0).mean(dim=0),
        torch.stack(buf_hpv, dim=0).mean(dim=0),
    )


def _load_l5_baseline(path: Path) -> tuple[Optional[float], Optional[float]]:
    """Return ``(rate_l23e_mean_L5, fwhm_median_deg_L5)`` from the post-Fix-M
    Level-5 JSON, or ``(None, None)`` on missing/malformed."""
    try:
        if not path.exists():
            return None, None
        j = json.loads(path.read_text())
        s = j.get("stats", {})
        rate = s.get("rate_l23e_mean")
        fwhm = s.get("fwhm_median_deg")
        return (
            float(rate) if rate is not None else None,
            float(fwhm) if fwhm is not None else None,
        )
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-orients", type=int, default=12)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-steps-l4", type=int, default=40)
    p.add_argument("--avg-last-l4", type=int, default=20)
    p.add_argument("--n-steps-l23", type=int, default=300,
                   help="Extended vs Level 5: HE has τ=50 ms (leak=0.9) so "
                        "its transient is ~10 τ_HE = 100 steps — 300 steps "
                        "total gives 2× margin for the HE ↔ HPV + apical-FB "
                        "loops to settle.")
    p.add_argument("--avg-last-l23", type=int, default=60)
    p.add_argument("--level-5-json", type=Path,
                   default=Path("logs/task74/level_5_post_fixM.json"),
                   help="Level-5 post-Fix-M JSON for Δ computations.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    net = V2Network(cfg, token_bank=None, seed=seed, device="cpu")
    net.eval()

    n_l23e = cfg.arch.n_l23_e
    n_pv = cfg.arch.n_l23_pv
    n_som = cfg.arch.n_l23_som
    n_h = cfg.arch.n_h_e
    n_hpv = cfg.arch.n_h_pv
    orientations = np.linspace(0.0, 180.0, int(args.n_orients), endpoint=False)

    tuning_l23 = np.zeros((int(args.n_orients), n_l23e), dtype=np.float64)
    tuning_h = np.zeros((int(args.n_orients), n_h), dtype=np.float64)
    tuning_som_mean = np.zeros(int(args.n_orients), dtype=np.float64)
    tuning_pv_mean = np.zeros(int(args.n_orients), dtype=np.float64)
    tuning_hpv_mean = np.zeros(int(args.n_orients), dtype=np.float64)

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), 1.0, cfg, batch_size=int(args.n_trials),
        )
        r_l4_ss = _drive_lgn_l4_to_steady(
            net, cfg, frame, int(args.n_steps_l4), int(args.avg_last_l4),
        )
        r_l23, r_pv, r_som, r_h, r_hpv = _drive_full_substrate_from_l4(
            net, r_l4_ss, int(args.n_steps_l23), int(args.avg_last_l23),
        )
        tuning_l23[oi] = r_l23.mean(dim=0).cpu().numpy().astype(np.float64)
        tuning_h[oi] = r_h.mean(dim=0).cpu().numpy().astype(np.float64)
        tuning_som_mean[oi] = float(r_som.mean().cpu())
        tuning_pv_mean[oi] = float(r_pv.mean().cpu())
        tuning_hpv_mean[oi] = float(r_hpv.mean().cpu())

    # ---- L23E stats ---------------------------------------------------------
    peak = tuning_l23.max(axis=0)
    trough = tuning_l23.min(axis=0)
    mean_unit = tuning_l23.mean(axis=0)
    pref_idx = tuning_l23.argmax(axis=0)

    well_defined = peak > (2.0 * trough + 0.01)
    n_defined_frac = float(well_defined.mean())
    fwhms_l23 = np.array([
        _circular_fwhm_deg(tuning_l23[:, u], orientations) for u in range(n_l23e)
    ])
    fwhm_median_l23 = float(np.nanmedian(fwhms_l23))

    pref_hist_l23 = np.bincount(pref_idx, minlength=int(args.n_orients))
    bin_threshold_l23 = max(1, int(math.ceil(0.05 * n_l23e)))
    n_pref_bins_5pct_l23 = int((pref_hist_l23 >= bin_threshold_l23).sum())

    rate_l23e_mean = float(mean_unit.mean())
    rate_l23e_min = float(mean_unit.min())
    rate_l23e_max = float(mean_unit.max())
    n_silent_l23e = int((peak <= 1e-9).sum())
    n_runaway_l23e = int((mean_unit > 100.0).sum())

    # ---- H stats ------------------------------------------------------------
    h_peak = tuning_h.max(axis=0)
    h_trough = tuning_h.min(axis=0)
    h_mean_unit = tuning_h.mean(axis=0)
    h_pref_idx = tuning_h.argmax(axis=0)

    fwhms_h = np.array([
        _circular_fwhm_deg(tuning_h[:, u], orientations) for u in range(n_h)
    ])
    fwhm_median_h = float(np.nanmedian(fwhms_h))

    pref_hist_h = np.bincount(h_pref_idx, minlength=int(args.n_orients))
    bin_threshold_h = max(1, int(math.ceil(0.05 * n_h)))
    n_pref_bins_5pct_h = int((pref_hist_h >= bin_threshold_h).sum())

    rate_h_mean = float(h_mean_unit.mean())
    rate_h_min = float(h_mean_unit.min())
    rate_h_max = float(h_mean_unit.max())
    n_silent_h = int((h_peak <= 1e-9).sum())
    n_runaway_h = int((h_mean_unit > 100.0).sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        h_snr_per_unit = np.where(
            h_mean_unit > 1e-6,
            (h_peak - h_trough) / h_mean_unit,
            np.nan,
        )
    h_snr_median = float(np.nanmedian(h_snr_per_unit))

    # ---- SOM / PV / HPV (diagnostic) ---------------------------------------
    rate_som_mean = float(tuning_som_mean.mean())
    rate_pv_mean = float(tuning_pv_mean.mean())
    rate_hpv_mean = float(tuning_hpv_mean.mean())
    n_runaway_som = int((tuning_som_mean > 100.0).sum())
    n_runaway_pv = int((tuning_pv_mean > 100.0).sum())
    n_runaway_hpv = int((tuning_hpv_mean > 100.0).sum())

    # ---- Δ vs Level 5 -------------------------------------------------------
    rate_l5, fwhm_l5 = _load_l5_baseline(args.level_5_json)
    delta_rate_vs_l5 = (
        rate_l23e_mean - rate_l5 if rate_l5 is not None else float("nan")
    )
    delta_fwhm_vs_l5 = (
        fwhm_median_l23 - fwhm_l5 if fwhm_l5 is not None else float("nan")
    )

    # ---- Verdict ------------------------------------------------------------
    fails: list[str] = []
    if n_defined_frac < 0.80:
        fails.append(f"n_l23e_defined {n_defined_frac:.2f}<0.80")
    if not (30.0 <= fwhm_median_l23 <= 80.0):
        fails.append(f"fwhm_l23e {fwhm_median_l23:.1f}∉[30,80]")
    if n_pref_bins_5pct_l23 < 8:
        fails.append(f"n_bins_l23e {n_pref_bins_5pct_l23}/12<8")
    if not (0.5 <= rate_l23e_mean <= 10.0):
        fails.append(f"rate_l23e {rate_l23e_mean:.3f}∉[0.5,10]")
    if not (0.05 <= rate_h_mean <= 5.0):
        fails.append(f"rate_h {rate_h_mean:.3f}∉[0.05,5]")
    if not (5.0 <= rate_hpv_mean <= 50.0):
        fails.append(f"rate_hpv {rate_hpv_mean:.3f}∉[5,50]")
    if not (0.3 <= rate_som_mean <= 20.0):
        fails.append(f"rate_som {rate_som_mean:.3f}∉[0.3,20]")
    if n_silent_l23e > 0:
        fails.append(f"n_silent_l23e={n_silent_l23e}")
    if n_runaway_l23e > 0:
        fails.append(f"n_runaway_l23e={n_runaway_l23e}")
    if n_silent_h > 0:
        fails.append(f"n_silent_h={n_silent_h}")
    if n_runaway_h > 0:
        fails.append(f"n_runaway_h={n_runaway_h}")
    if n_runaway_som > 0:
        fails.append(f"n_runaway_som={n_runaway_som}")
    if n_runaway_pv > 0:
        fails.append(f"n_runaway_pv={n_runaway_pv}")
    if n_runaway_hpv > 0:
        fails.append(f"n_runaway_hpv={n_runaway_hpv}")

    verdict = "pass" if not fails else "fail"
    issue = "none" if not fails else ";".join(fails)

    summary = {
        "version": "level_6_full_substrate_v1",
        "seed": seed,
        "n_l23e": int(n_l23e),
        "n_pv": int(n_pv),
        "n_som": int(n_som),
        "n_h": int(n_h),
        "n_hpv": int(n_hpv),
        "n_orients": int(args.n_orients),
        "n_trials": int(args.n_trials),
        "n_steps_l4": int(args.n_steps_l4),
        "n_steps_l23": int(args.n_steps_l23),
        "orients_deg": orientations.tolist(),
        "stats": {
            # L23E
            "n_l23e_defined_frac": n_defined_frac,
            "fwhm_median_deg": fwhm_median_l23,
            "fwhm_p25_deg": float(np.nanpercentile(fwhms_l23, 25)),
            "fwhm_p75_deg": float(np.nanpercentile(fwhms_l23, 75)),
            "pref_hist": pref_hist_l23.tolist(),
            "bin_threshold_units": bin_threshold_l23,
            "n_pref_bins_5pct": n_pref_bins_5pct_l23,
            "rate_l23e_mean": rate_l23e_mean,
            "rate_l23e_min": rate_l23e_min,
            "rate_l23e_max": rate_l23e_max,
            "n_silent_l23e": n_silent_l23e,
            "n_runaway_l23e": n_runaway_l23e,
            # HE
            "rate_h_mean": rate_h_mean,
            "rate_h_min": rate_h_min,
            "rate_h_max": rate_h_max,
            "h_fwhm_median_deg": fwhm_median_h,
            "h_pref_hist": pref_hist_h.tolist(),
            "h_bin_threshold_units": bin_threshold_h,
            "h_n_pref_bins_5pct": n_pref_bins_5pct_h,
            "h_snr_median": h_snr_median,
            "n_silent_h": n_silent_h,
            "n_runaway_h": n_runaway_h,
            # SOM / PV / HPV
            "rate_som_mean": rate_som_mean,
            "rate_pv_mean": rate_pv_mean,
            "rate_hpv_mean": rate_hpv_mean,
            "n_runaway_som": n_runaway_som,
            "n_runaway_pv": n_runaway_pv,
            "n_runaway_hpv": n_runaway_hpv,
            # Δ vs Level 5
            "level5_rate_l23e_mean": rate_l5,
            "level5_fwhm_median_deg": fwhm_l5,
            "delta_rate_l23e_vs_L5": delta_rate_vs_l5,
            "delta_fwhm_l23e_vs_L5": delta_fwhm_vs_l5,
        },
        "verdict": verdict,
        "issue_if_fail": issue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    line = (
        f"level6_verdict={verdict} "
        f"n_l23e_defined={n_defined_frac:.3f} "
        f"fwhm_l23e={fwhm_median_l23:.1f}° "
        f"n_bins_l23e={n_pref_bins_5pct_l23}/12 "
        f"rate_l23e={rate_l23e_mean:.3f} "
        f"rate_h={rate_h_mean:.3f} "
        f"rate_hpv={rate_hpv_mean:.3f} "
        f"rate_som={rate_som_mean:.3f} "
        f"h_snr_median={h_snr_median:.3f} "
        f"Δrate_l23e_vs_L5={delta_rate_vs_l5:.3f} "
        f"Δfwhm_vs_L5={delta_fwhm_vs_l5:.2f} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
