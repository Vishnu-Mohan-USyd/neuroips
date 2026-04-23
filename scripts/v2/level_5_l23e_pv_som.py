"""Level 5 component validation — enable SOM on top of Fix-K + recurrent + Fix-L2 PV.

Per Lead's bottom-up validation protocol (Task #74). Scope: Level 4 plus the
co-simulated L23SOM population. H feedback (to L23E apical and to SOM) stays
zeroed; context bias / som_gain disabled.

Co-simulation
-------------
Strict sync-Euler (each sibling reads the other's previous-step rate):

    r_l23_prev, r_pv_prev, r_som_prev = r_l23, r_pv, r_som
    r_l23, _ = L23E.forward(
        l4_input          = r_l4_ss,                     # Fix-K sparse FF
        l23_recurrent_input = r_l23_prev,
        som_input         = r_som_prev,                  # <<< Level-5 addition
        pv_input          = r_pv_prev,                   # Fix-L2 PV
        h_apical_input    = zeros,
        context_bias      = zeros,
        state             = r_l23_prev,
        som_gain          = None,
    )
    r_pv, _  = L23PV.forward(r_l23_prev, r_pv_prev)
    r_som, _ = L23SOM.forward(
        l23e_input            = r_l23_prev,
        h_som_feedback_input  = zeros,                   # H still off
        state                 = r_som_prev,
    )

Probes
------
1. Orient sweep at contrast=1, 12 orients × n_trials trials. Per-L23E-unit
   FWHM, peak/trough/mean, preferred orient. Per-SOM-unit tuning across the
   same 12 orientations.

Derived diagnostics
-------------------
* SOM orient-selectivity SNR per unit = (max − min) / mean (set NaN if
  mean ≤ 0). Median reported.
* Δ_l23e_rate_vs_L4  = rate_l23e_mean − (L4 baseline rate_l23e_mean).
* Δ_fwhm_vs_L4       = fwhm_median_deg − (L4 baseline fwhm_median_deg).

  L4 baseline JSON (post-Fix-L2) located at
  ``logs/task74/level_4_post_fixL2.json``; if missing, Δ-fields = NaN.

Pass criteria (six gated + four diagnostics)
-------------------------------------------
Gated:
  * ``n_l23e_defined_frac`` ≥ 0.80 (``peak > 2·trough + 0.01``)
  * ``fwhm_median_deg`` ∈ [30°, 80°]
  * ≥ 8 / 12 preferred-orient bins populated (≥ 5% of n_l23_e each)
  * L23E ``rate_mean`` ∈ [0.5, 10] Hz
  * SOM ``rate_mean`` ∈ [0.5, 20] Hz
  * no silent L23E unit (peak ≤ 0), no runaway (mean > 100 Hz) in either
    L23E or SOM population

Diagnostics (reported, not gated):
  * ``som_snr_median``     — expected low under all-to-all W_l23_som
  * ``Δrate_vs_L4``
  * ``Δfwhm_vs_L4``

DM summary::

  level5_verdict=<pass/fail> n_l23e_defined=<#> fwhm_median=<#>°
    n_bins=<#>/12 rate_l23e=<#> rate_som=<#> som_snr_median=<#>
    Δrate_vs_L4=<#> Δfwhm_vs_L4=<#> issue_if_fail=<short>
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
def _drive_l23e_pv_som_from_l4(
    net: V2Network, r_l4_ss: Tensor, n_steps: int, avg_last: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Co-simulate L23E + L23PV + L23SOM with strict sync-Euler.

    Returns ``(r_l23_mean, r_pv_mean, r_som_mean)`` each averaged across
    the final ``avg_last`` Euler steps. H-feedback paths zeroed so that
    SOM is driven solely by L23E → SOM.
    """
    B = r_l4_ss.shape[0]
    device = r_l4_ss.device
    dtype = r_l4_ss.dtype
    l23e = net.l23_e
    l23_pv = net.l23_pv
    l23_som = net.l23_som

    def _zeros(n: int) -> Tensor:
        return torch.zeros(B, n, dtype=dtype, device=device)

    zeros_fb = _zeros(l23e.n_h_e)
    zeros_bias = _zeros(l23e.n_units)
    zeros_som_fb = _zeros(l23_som.n_h_e)

    r_l23 = torch.zeros(B, l23e.n_units, dtype=dtype, device=device)
    r_pv = torch.zeros(B, l23_pv.n_units, dtype=dtype, device=device)
    r_som = torch.zeros(B, l23_som.n_units, dtype=dtype, device=device)

    buf_l23: list[Tensor] = []
    buf_pv: list[Tensor] = []
    buf_som: list[Tensor] = []
    for t in range(int(n_steps)):
        r_l23_prev = r_l23
        r_pv_prev = r_pv
        r_som_prev = r_som
        r_l23, _ = l23e(
            l4_input=r_l4_ss,
            l23_recurrent_input=r_l23_prev,
            som_input=r_som_prev,
            pv_input=r_pv_prev,
            h_apical_input=zeros_fb,
            context_bias=zeros_bias,
            state=r_l23_prev,
            som_gain=None,
        )
        r_pv, _ = l23_pv(r_l23_prev, r_pv_prev)
        r_som, _ = l23_som(
            l23e_input=r_l23_prev,
            h_som_feedback_input=zeros_som_fb,
            state=r_som_prev,
        )
        if t >= n_steps - int(avg_last):
            buf_l23.append(r_l23)
            buf_pv.append(r_pv)
            buf_som.append(r_som)
    return (
        torch.stack(buf_l23, dim=0).mean(dim=0),
        torch.stack(buf_pv, dim=0).mean(dim=0),
        torch.stack(buf_som, dim=0).mean(dim=0),
    )


def _load_l4_baseline(path: Path) -> tuple[Optional[float], Optional[float]]:
    """Return ``(rate_l23e_mean_L4, fwhm_median_L4)`` or ``(None, None)`` on
    missing / malformed JSON."""
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
    except Exception:  # noqa: BLE001 — best-effort diagnostic load
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
    p.add_argument("--n-steps-l23", type=int, default=200,
                   help="Matches Level 4 (PV + SOM transients both settle "
                        "within ~10 τ_L23; SOM τ=20 ms like L23E).")
    p.add_argument("--avg-last-l23", type=int, default=40)
    p.add_argument("--level-4-json", type=Path,
                   default=Path("logs/task74/level_4_post_fixL2.json"),
                   help="Level-4 post-Fix-L2 JSON for Δ computations.")
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
    orientations = np.linspace(0.0, 180.0, int(args.n_orients), endpoint=False)

    tuning_l23 = np.zeros((int(args.n_orients), n_l23e), dtype=np.float64)
    tuning_som = np.zeros((int(args.n_orients), n_som), dtype=np.float64)
    tuning_pv_mean = np.zeros(int(args.n_orients), dtype=np.float64)

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), 1.0, cfg, batch_size=int(args.n_trials),
        )
        r_l4_ss = _drive_lgn_l4_to_steady(
            net, cfg, frame, int(args.n_steps_l4), int(args.avg_last_l4),
        )
        r_l23, r_pv, r_som = _drive_l23e_pv_som_from_l4(
            net, r_l4_ss, int(args.n_steps_l23), int(args.avg_last_l23),
        )
        tuning_l23[oi] = r_l23.mean(dim=0).cpu().numpy().astype(np.float64)
        tuning_som[oi] = r_som.mean(dim=0).cpu().numpy().astype(np.float64)
        tuning_pv_mean[oi] = float(r_pv.mean().cpu())

    # ---- L23E tuning stats --------------------------------------------------
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
    n_silent_l23e = int((peak <= 1e-9).sum())
    n_runaway_l23e = int((mean_unit > 100.0).sum())

    # ---- SOM stats ----------------------------------------------------------
    som_max = tuning_som.max(axis=0)
    som_min = tuning_som.min(axis=0)
    som_mean_unit = tuning_som.mean(axis=0)
    rate_som_mean = float(som_mean_unit.mean())
    rate_som_max = float(som_mean_unit.max())
    rate_som_min = float(som_mean_unit.min())
    n_runaway_som = int((som_mean_unit > 100.0).sum())

    # SOM SNR per unit: (max − min) / mean (NaN if mean ≤ 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        som_snr_per_unit = np.where(
            som_mean_unit > 1e-6,
            (som_max - som_min) / som_mean_unit,
            np.nan,
        )
    som_snr_median = float(np.nanmedian(som_snr_per_unit))

    # ---- Δ vs Level 4 -------------------------------------------------------
    rate_l4, fwhm_l4 = _load_l4_baseline(args.level_4_json)
    delta_rate_vs_l4 = (
        rate_l23e_mean - rate_l4 if rate_l4 is not None else float("nan")
    )
    delta_fwhm_vs_l4 = (
        fwhm_median - fwhm_l4 if fwhm_l4 is not None else float("nan")
    )

    # ---- Verdict ------------------------------------------------------------
    fails: list[str] = []
    if n_defined_frac < 0.80:
        fails.append(f"n_l23e_defined {n_defined_frac:.2f}<0.80")
    if not (30.0 <= fwhm_median <= 80.0):
        fails.append(f"fwhm_median {fwhm_median:.1f}∉[30,80]")
    if n_pref_bins_5pct < 8:
        fails.append(f"n_pref_bins_5pct {n_pref_bins_5pct}/12<8")
    if not (0.5 <= rate_l23e_mean <= 10.0):
        fails.append(f"rate_l23e_mean {rate_l23e_mean:.3f}∉[0.5,10]")
    if not (0.5 <= rate_som_mean <= 20.0):
        fails.append(f"rate_som_mean {rate_som_mean:.3f}∉[0.5,20]")
    if n_silent_l23e > 0:
        fails.append(f"n_silent_l23e={n_silent_l23e}")
    if n_runaway_l23e > 0:
        fails.append(f"n_runaway_l23e={n_runaway_l23e}")
    if n_runaway_som > 0:
        fails.append(f"n_runaway_som={n_runaway_som}")

    verdict = "pass" if not fails else "fail"
    issue = "none" if not fails else ";".join(fails)

    summary = {
        "version": "level_5_l23e_pv_som_v1",
        "seed": seed,
        "n_l23e": int(n_l23e),
        "n_pv": int(n_pv),
        "n_som": int(n_som),
        "n_orients": int(args.n_orients),
        "n_trials": int(args.n_trials),
        "n_steps_l4": int(args.n_steps_l4),
        "n_steps_l23": int(args.n_steps_l23),
        "orients_deg": orientations.tolist(),
        "stats": {
            # L23E
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
            "n_silent_l23e": n_silent_l23e,
            "n_runaway_l23e": n_runaway_l23e,
            # PV (diagnostic)
            "rate_pv_mean": float(tuning_pv_mean.mean()),
            "rate_pv_across_orients": tuning_pv_mean.tolist(),
            # SOM
            "rate_som_mean": rate_som_mean,
            "rate_som_min": rate_som_min,
            "rate_som_max": rate_som_max,
            "som_snr_median": som_snr_median,
            "n_runaway_som": n_runaway_som,
            # Δ vs Level 4
            "level4_rate_l23e_mean": rate_l4,
            "level4_fwhm_median_deg": fwhm_l4,
            "delta_rate_vs_L4": delta_rate_vs_l4,
            "delta_fwhm_vs_L4": delta_fwhm_vs_l4,
        },
        "verdict": verdict,
        "issue_if_fail": issue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    line = (
        f"level5_verdict={verdict} "
        f"n_l23e_defined={n_defined_frac:.3f} "
        f"fwhm_median={fwhm_median:.1f}° "
        f"n_bins={n_pref_bins_5pct}/12 "
        f"rate_l23e={rate_l23e_mean:.3f} "
        f"rate_som={rate_som_mean:.3f} "
        f"som_snr_median={som_snr_median:.3f} "
        f"Δrate_vs_L4={delta_rate_vs_l4:.3f} "
        f"Δfwhm_vs_L4={delta_fwhm_vs_l4:.2f} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
