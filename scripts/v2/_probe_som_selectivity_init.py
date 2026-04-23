"""Task #74: SOM orientation-selectivity probe on Fix-E INIT network.

Pre-Phase-3 diagnostic. Phase-3 Fix C-v2 learns from
``modulator = r_som_EMA_expected − r_som_EMA_unexpected``; this requires SOM
units to respond DIFFERENTIALLY to probe orientations at init. If SOM is
orientation-blind, the three-factor rule has no signal to learn from.

Protocol
--------
1. Fresh V2Network(seed=42, Fix-E init — W_l23_som_raw init_mean=-4.5 already
   applied in src/v2_model/layers.py).
2. 12 orientations × 20 trials of single-orientation gratings (contrast=1.0,
   100 forward steps, avg over last 50).
3. Per SOM unit j: collect response over the 12 bins (averaged over 20 trials),
   compute SNR_j = (max − min) / mean.
4. Report: snr_median, snr_p90, frac_snr_above_0.2, n_preferred_bins
   (bins with ≥ ceil(0.05·n_som) = 2 units assigned as preferred).

Output
------
JSON to --output, and a one-line DM summary to stdout in the format:
``SOM_init_selectivity snr_median=<#> snr_p90=<#> frac_snr_above_0.2=<#>
  n_preferred_bins=<#>``
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.v2._gates_common import make_grating_frame
from scripts.v2.phase_a_static_sanity import _probe_steady_state
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-orients", type=int, default=12)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--avg-last", type=int, default=50)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    torch.manual_seed(seed)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.eval()

    n_som = cfg.arch.n_l23_som
    orientations = np.linspace(0.0, 180.0, args.n_orients, endpoint=False)

    # [n_orients, n_som]: trial-averaged rate per SOM unit per orientation.
    tuning_som = np.zeros((args.n_orients, n_som), dtype=np.float64)

    for oi, ori in enumerate(orientations):
        frame = make_grating_frame(
            float(ori), float(args.contrast), cfg, batch_size=int(args.n_trials),
        )
        probe = _probe_steady_state(
            net, frame, int(args.n_steps),
            avg_last=int(args.avg_last), trial_seed=oi,
        )
        tuning_som[oi] = probe["r_som"].mean(axis=0)

    # Per-unit SNR = (max - min) / mean across the 12 orientation bins.
    # Guard: units with mean ~ 0 get snr=0 (they carry no signal).
    unit_max = tuning_som.max(axis=0)                               # [n_som]
    unit_min = tuning_som.min(axis=0)
    unit_mean = tuning_som.mean(axis=0)
    snr = np.where(
        unit_mean > 1e-9,
        (unit_max - unit_min) / unit_mean,
        0.0,
    )

    # Preferred orient per unit; bin counts; bins populated by ≥ threshold units.
    preferred_idx = np.argmax(tuning_som, axis=0)                   # [n_som]
    pref_hist = np.bincount(preferred_idx, minlength=int(args.n_orients))
    pref_threshold = max(1, int(math.ceil(0.05 * n_som)))            # ≥ 2 for 32
    n_pref_bins = int((pref_hist >= pref_threshold).sum())

    snr_median = float(np.median(snr))
    snr_p90 = float(np.quantile(snr, 0.90))
    frac_above_02 = float((snr > 0.2).mean())

    summary_line = (
        f"SOM_init_selectivity "
        f"snr_median={snr_median:.3f} "
        f"snr_p90={snr_p90:.3f} "
        f"frac_snr_above_0.2={frac_above_02:.3f} "
        f"n_preferred_bins={n_pref_bins}"
    )

    result = {
        "version": "som_init_selectivity_v1",
        "seed": seed,
        "n_som": int(n_som),
        "n_orients": int(args.n_orients),
        "n_trials": int(args.n_trials),
        "n_steps": int(args.n_steps),
        "avg_last": int(args.avg_last),
        "contrast": float(args.contrast),
        "orientations_deg": orientations.tolist(),
        "tuning_som_mean": tuning_som.tolist(),
        "snr_per_unit": snr.tolist(),
        "preferred_orient_idx_per_unit": preferred_idx.tolist(),
        "preferred_bin_histogram": pref_hist.tolist(),
        "preferred_bin_threshold_units": pref_threshold,
        "summary": {
            "snr_median": snr_median,
            "snr_p90": snr_p90,
            "frac_snr_above_0.2": frac_above_02,
            "n_preferred_bins": n_pref_bins,
            "snr_mean": float(snr.mean()),
            "snr_max": float(snr.max()),
            "r_som_unit_mean_median": float(np.median(unit_mean)),
            "r_som_unit_mean_max": float(unit_mean.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(summary_line)
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
