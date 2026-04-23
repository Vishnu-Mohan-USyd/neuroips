"""Level 2 component validation — L23 E response to L4 afferent drive only.

Per Lead's bottom-up validation protocol (Task #74). Scope: evaluate L23E
orientation tuning when **only** the L4→L23E excitatory pathway is active.
Recurrent L23E, PV, SOM, H-feedback, and context bias are all zeroed at
test time by supplying zero tensors to the corresponding forward kwargs
(no weight mutation — stimuli-free isolation).

Protocol
--------
1. Instantiate ``V2Network(seed=42)`` on CPU. Keep the LGN/L4 front end
   (fixed DoG + Gabor filterbank) and ``L23E`` module; ignore the rest.
2. For each of 12 orientations and ``n_trials`` redundant trials,
   generate a full-contrast centered grating frame and drive LGN/L4 to
   a steady L4-E rate (n_steps_l4 forward evaluations).
3. Feed the steady L4 rate into ``L23E.forward`` directly, with every
   other kwarg (``l23_recurrent_input``, ``som_input``, ``pv_input``,
   ``h_apical_input``, ``context_bias``) supplied as an explicit zero
   tensor and ``som_gain=None``. Iterate ``n_steps_l23`` Euler steps and
   average the last ``avg_last`` to a trial-mean L23E rate. This isolates
   the ``L4 → W_l4_l23_raw → φ(drive − θ)`` pathway.
4. Per-unit metrics across the 12-bin tuning curve:
     - preferred orient (argmax)
     - FWHM (circular, same helper as Level 1)
     - peak, trough, mean rate
5. Pass criteria (all five):
     - ``n_l23e_defined_frac`` ≥ 0.80 (``peak > 2·trough + 0.01 Hz``)
     - ``fwhm_median_deg`` ∈ [40°, 80°]
     - preferred-orient histogram has ≥ ceil(0.05·n_l23_e) units in ≥ 8
       of 12 bins (broad orientation coverage)
     - mean-rate across units ∈ [0.5, 10] Hz
     - no unit silent (0 Hz across all 12 bins) and no unit > 100 Hz

Stdout (last line):
  ``level2_verdict=<pass/fail> n_l23e_defined_frac=<#>
    fwhm_median_deg=<#> n_preferred_bins_5pct=<#>/12 rate_mean=<#>
    rate_max=<#> rate_min=<#> issue_if_fail=<short>``
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
    """FWHM of a 180°-periodic tuning curve in degrees.

    ``tuning``: [n_orients] non-negative values. Returns NaN if the curve
    is flat (peak ≈ trough) or all samples below half-max.
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
    bin_width = 180.0 / float(len(orients_deg))
    return float(n_above * bin_width)


@torch.no_grad()
def _drive_lgn_l4_to_steady(
    net: V2Network, cfg: ModelConfig, frame: Tensor,
    n_steps: int, avg_last: int,
) -> Tensor:
    """Run only LGN/L4 on ``frame`` for ``n_steps``; return L4 E rate
    averaged over the final ``avg_last`` steps. Shape [B, n_l4_e].

    ``state.r_l4`` is the only recurrent quantity LGN/L4 reads, so we do
    not need to simulate any downstream population."""
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
def _drive_l23e_from_l4(
    net: V2Network, r_l4_ss: Tensor, n_steps: int, avg_last: int,
) -> Tensor:
    """Iterate L23E.forward with ONLY ``l4_input = r_l4_ss`` active.

    All other input kwargs (l23_recurrent_input, som_input, pv_input,
    h_apical_input, context_bias) are supplied as explicit zero tensors;
    ``som_gain`` is left as ``None``. Returns [B, n_l23_e] rate averaged
    over the last ``avg_last`` steps.

    This is equivalent to the L4-only pathway:
        r_l23_{t+1} = λ·r_l23_t + (1−λ)·φ( softplus(W_l4_l23_raw)·r_l4_ss − θ )
    i.e. the L4→L23 drive passing through L23E's own leak / activation /
    homeostatic threshold θ, with **no** recurrent / interneuron /
    top-down contribution.
    """
    B = r_l4_ss.shape[0]
    device = r_l4_ss.device
    dtype = r_l4_ss.dtype
    l23e = net.l23_e

    def _zeros(n: int) -> Tensor:
        return torch.zeros(B, n, dtype=dtype, device=device)

    zeros_rec = _zeros(l23e.n_units)
    zeros_pv = _zeros(l23e.n_pv)
    zeros_som = _zeros(l23e.n_som)
    zeros_fb = _zeros(l23e.n_h_e)
    zeros_bias = _zeros(l23e.n_units)

    state = torch.zeros(B, l23e.n_units, dtype=dtype, device=device)
    buf = []
    for t in range(int(n_steps)):
        rate, state = l23e(
            l4_input=r_l4_ss,
            l23_recurrent_input=zeros_rec,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-orients", type=int, default=12)
    p.add_argument("--n-trials", type=int, default=10,
                   help="redundant trials per orientation (batch dim). "
                        "LGN/L4 + L23E-from-L4 are deterministic given "
                        "the same frame + seed, so batching serves as a "
                        "shape sanity check; trials are averaged.")
    p.add_argument("--n-steps-l4", type=int, default=40)
    p.add_argument("--avg-last-l4", type=int, default=20)
    p.add_argument("--n-steps-l23", type=int, default=80,
                   help="L23E τ = 20 ms at dt = 5 ms → leak ≈ 0.78; "
                        "80 steps ≈ 400 ms, plenty for steady state.")
    p.add_argument("--avg-last-l23", type=int, default=40)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    net = V2Network(cfg, token_bank=None, seed=seed, device="cpu")
    net.eval()

    n_l23e = cfg.arch.n_l23_e
    orientations = np.linspace(
        0.0, 180.0, int(args.n_orients), endpoint=False,
    )
    tuning = np.zeros((int(args.n_orients), n_l23e), dtype=np.float64)

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), 1.0, cfg, batch_size=int(args.n_trials),
        )
        r_l4_ss = _drive_lgn_l4_to_steady(
            net, cfg, frame,
            int(args.n_steps_l4), int(args.avg_last_l4),
        )  # [B, n_l4_e]
        r_l23_ss = _drive_l23e_from_l4(
            net, r_l4_ss, int(args.n_steps_l23), int(args.avg_last_l23),
        )  # [B, n_l23_e]
        # Trial-mean per unit.
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

    # Preferred-orient histogram (5% threshold per bin → 8 of 12 bins).
    pref_hist = np.bincount(pref_idx, minlength=int(args.n_orients))
    bin_threshold = max(1, int(math.ceil(0.05 * n_l23e)))
    n_pref_bins_5pct = int((pref_hist >= bin_threshold).sum())

    # Rate statistics: population mean, min (silent), max (runaway).
    rate_mean = float(mean_unit.mean())
    rate_min = float(mean_unit.min())
    rate_max = float(mean_unit.max())
    # "Silent" = unit has zero response across every orientation (peak == 0).
    # "Runaway" = any unit mean > 100 Hz.
    n_silent = int((peak <= 1e-9).sum())
    n_runaway = int((mean_unit > 100.0).sum())

    # -------- Verdict -----------------------------------------------------
    fails: list[str] = []
    if n_defined_frac < 0.80:
        fails.append(f"n_l23e_defined {n_defined_frac:.2f}<0.80")
    if not (40.0 <= fwhm_median <= 80.0):
        fails.append(f"fwhm_median {fwhm_median:.1f}∉[40,80]")
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
        "version": "level_2_l23e_l4_only_v1",
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
        f"level2_verdict={verdict} "
        f"n_l23e_defined_frac={n_defined_frac:.3f} "
        f"fwhm_median_deg={fwhm_median:.1f} "
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
