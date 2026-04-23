"""Level 3 component validation — L23E + W_rec_l23 on top of Fix-K L4 drive.

Per Lead's bottom-up validation protocol (Task #74). Scope: Level 2 plus
the L23E recurrent pathway ``W_rec_l23``. PV output, SOM output, and H
feedback remain zeroed (zero tensors passed to ``L23E.forward``); context
bias and som_gain are disabled (None / zero).

This isolates whether the sparse like-to-like recurrent mask at init
(``W_rec_raw = softplus(-5.0) ≈ 0.0067`` with ~30 incoming per row)
sharpens or broadens the feedforward tuning established by Fix K, and
whether the recurrent loop stays stable on its own (no divisive PV).

Protocol
--------
1. Build ``V2Network(seed=42)`` (Fix K mask already installed in __init__).
2. For each of 12 orientations × ``n_trials`` trials, drive LGN/L4 to a
   steady L4-E rate (n_steps_l4 forward evals).
3. Iterate ``L23E.forward`` directly with:
     l4_input              = r_l4_ss           (Fix-K pathway)
     l23_recurrent_input   = state.r_l23_prev  (pre-update sync-Euler)
     som_input, pv_input, h_apical_input, context_bias = zeros
     som_gain              = None
   for n_steps_l23 Euler steps; average last avg_last.
4. Per-unit 12-bin tuning curve + FWHM + preferred orient + peak/trough/
   mean. Compare FWHM median to the Level 2 post-Fix-K JSON (narrower /
   same / broader classifier with ±2° tolerance).

Pass criteria (five):
  * ``n_l23e_defined_frac`` ≥ 0.80 (``peak > 2·trough + 0.01``)
  * ``fwhm_median_deg`` ∈ [30°, 80°]
  * preferred-orient histogram ≥ ceil(0.05·n_l23_e) units in ≥ 8 of 12 bins
  * ``rate_mean`` ∈ [0.5, 10] Hz (no recurrent runaway)
  * no unit silent (peak ≤ 0) and no unit > 100 Hz

DM summary:
  ``level3_verdict=<pass/fail> n_l23e_defined_frac=<#> fwhm_median_deg=<#>
    fwhm_vs_level2=<narrower|same|broader> n_preferred_bins_5pct=<#>/12
    rate_mean=<#> rate_max=<#> rate_min=<#> issue_if_fail=<short>``
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
    buf = []
    for t in range(int(n_steps)):
        _feat, r_l4, _ = net.lgn_l4(frame, state)
        state = state._replace(r_l4=r_l4)
        if t >= n_steps - int(avg_last):
            buf.append(r_l4)
    return torch.stack(buf, dim=0).mean(dim=0)


@torch.no_grad()
def _drive_l23e_recurrent_from_l4(
    net: V2Network, r_l4_ss: Tensor, n_steps: int, avg_last: int,
) -> Tensor:
    """Iterate L23E.forward with L4 drive + recurrent feedback ONLY.

    Sync-Euler: ``l23_recurrent_input`` = previous-step rate (no fresh
    sibling read). PV, SOM, H-feedback, context bias all zero. som_gain
    unset. Returns [B, n_l23_e] averaged over the last ``avg_last`` steps.
    """
    B = r_l4_ss.shape[0]
    device = r_l4_ss.device
    dtype = r_l4_ss.dtype
    l23e = net.l23_e

    def _zeros(n: int) -> Tensor:
        return torch.zeros(B, n, dtype=dtype, device=device)

    zeros_pv = _zeros(l23e.n_pv)
    zeros_som = _zeros(l23e.n_som)
    zeros_fb = _zeros(l23e.n_h_e)
    zeros_bias = _zeros(l23e.n_units)

    state = torch.zeros(B, l23e.n_units, dtype=dtype, device=device)
    buf = []
    for t in range(int(n_steps)):
        rate, state = l23e(
            l4_input=r_l4_ss,
            l23_recurrent_input=state,               # ← Level-3 additive
            som_input=zeros_som,
            pv_input=zeros_pv,
            h_apical_input=zeros_fb,
            context_bias=zeros_bias,
            state=state,
            som_gain=None,
        )
        if t >= n_steps - int(avg_last):
            buf.append(rate)
    return torch.stack(buf, dim=0).mean(dim=0)


def _classify_fwhm(fwhm_L3: float, fwhm_L2: Optional[float],
                   tol_deg: float = 2.0) -> str:
    if fwhm_L2 is None or not math.isfinite(fwhm_L2):
        return "unknown"
    if fwhm_L3 + tol_deg < fwhm_L2:
        return "narrower"
    if fwhm_L3 - tol_deg > fwhm_L2:
        return "broader"
    return "same"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-orients", type=int, default=12)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-steps-l4", type=int, default=40)
    p.add_argument("--avg-last-l4", type=int, default=20)
    p.add_argument("--n-steps-l23", type=int, default=160,
                   help="Doubled vs Level 2 so the recurrent loop has time "
                        "to settle — leak=1-dt/τ=0.75, recurrent gain ~0.2 "
                        "at init so ~10 τ ≈ 160 steps to equilibrium.")
    p.add_argument("--avg-last-l23", type=int, default=40)
    p.add_argument("--level-2-json", type=Path,
                   default=Path("logs/task74/level_2_post_fixK.json"),
                   help="Level-2 post-Fix-K JSON, used to compute "
                        "fwhm_vs_level2 classifier.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    net = V2Network(cfg, token_bank=None, seed=seed, device="cpu")
    net.eval()

    n_l23e = cfg.arch.n_l23_e
    orientations = np.linspace(0.0, 180.0, int(args.n_orients), endpoint=False)
    tuning = np.zeros((int(args.n_orients), n_l23e), dtype=np.float64)

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), 1.0, cfg, batch_size=int(args.n_trials),
        )
        r_l4_ss = _drive_lgn_l4_to_steady(
            net, cfg, frame, int(args.n_steps_l4), int(args.avg_last_l4),
        )
        r_l23_ss = _drive_l23e_recurrent_from_l4(
            net, r_l4_ss, int(args.n_steps_l23), int(args.avg_last_l23),
        )
        tuning[oi] = r_l23_ss.mean(dim=0).cpu().numpy().astype(np.float64)

    peak = tuning.max(axis=0)
    trough = tuning.min(axis=0)
    mean_unit = tuning.mean(axis=0)
    pref_idx = tuning.argmax(axis=0)

    well_defined = peak > (2.0 * trough + 0.01)
    n_defined_frac = float(well_defined.mean())

    fwhms = np.array([
        _circular_fwhm_deg(tuning[:, u], orientations) for u in range(n_l23e)
    ])
    fwhm_median = float(np.nanmedian(fwhms))

    # Level 2 comparison.
    fwhm_L2: Optional[float] = None
    try:
        if args.level_2_json.exists():
            j = json.loads(args.level_2_json.read_text())
            fwhm_L2 = float(j.get("stats", {}).get("fwhm_median_deg", float("nan")))
    except Exception:  # noqa: BLE001
        fwhm_L2 = None
    fwhm_vs_L2 = _classify_fwhm(fwhm_median, fwhm_L2)

    pref_hist = np.bincount(pref_idx, minlength=int(args.n_orients))
    bin_threshold = max(1, int(math.ceil(0.05 * n_l23e)))
    n_pref_bins_5pct = int((pref_hist >= bin_threshold).sum())

    rate_mean = float(mean_unit.mean())
    rate_min = float(mean_unit.min())
    rate_max = float(mean_unit.max())
    n_silent = int((peak <= 1e-9).sum())
    n_runaway = int((mean_unit > 100.0).sum())

    fails: list[str] = []
    if n_defined_frac < 0.80:
        fails.append(f"n_l23e_defined {n_defined_frac:.2f}<0.80")
    if not (30.0 <= fwhm_median <= 80.0):
        fails.append(f"fwhm_median {fwhm_median:.1f}∉[30,80]")
    if n_pref_bins_5pct < 8:
        fails.append(f"n_pref_bins_5pct {n_pref_bins_5pct}/12<8")
    if not (0.5 <= rate_mean <= 10.0):
        fails.append(f"rate_mean {rate_mean:.3f}∉[0.5,10]")
    if n_silent > 0:
        fails.append(f"n_silent={n_silent}")
    if n_runaway > 0:
        fails.append(f"n_runaway={n_runaway}")

    verdict = "pass" if not fails else "fail"
    issue = "none" if not fails else ";".join(fails)

    summary = {
        "version": "level_3_l23e_recurrent_v1",
        "seed": seed,
        "n_l23e": int(n_l23e),
        "n_orients": int(args.n_orients),
        "n_trials": int(args.n_trials),
        "n_steps_l4": int(args.n_steps_l4),
        "n_steps_l23": int(args.n_steps_l23),
        "orients_deg": orientations.tolist(),
        "stats": {
            "n_l23e_defined_frac": n_defined_frac,
            "fwhm_median_deg": fwhm_median,
            "fwhm_p25_deg": float(np.nanpercentile(fwhms, 25)),
            "fwhm_p75_deg": float(np.nanpercentile(fwhms, 75)),
            "fwhm_level2_median_deg": fwhm_L2,
            "fwhm_vs_level2": fwhm_vs_L2,
            "pref_hist": pref_hist.tolist(),
            "bin_threshold_units": bin_threshold,
            "n_pref_bins_5pct": n_pref_bins_5pct,
            "rate_mean": rate_mean,
            "rate_min": rate_min,
            "rate_max": rate_max,
            "rate_median": float(np.median(mean_unit)),
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
        f"level3_verdict={verdict} "
        f"n_l23e_defined_frac={n_defined_frac:.3f} "
        f"fwhm_median_deg={fwhm_median:.1f} "
        f"fwhm_vs_level2={fwhm_vs_L2} "
        f"n_preferred_bins_5pct={n_pref_bins_5pct}/12 "
        f"rate_mean={rate_mean:.3f} "
        f"rate_max={rate_max:.3f} "
        f"rate_min={rate_min:.3f} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
