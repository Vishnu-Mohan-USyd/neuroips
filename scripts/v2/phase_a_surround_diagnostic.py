"""Does SOM see the surround? Diagnostic probe (Task #74 Fix E follow-up).

Three grating conditions × 5 orientations × 20 trials on a freshly-
constructed V2Network at init (post-Fix-E, W_l23_som_raw init_mean=-4.5):

* center-only  — grating inside a disc of diameter 8 px (radius 4), grey outside
* full-field   — grating everywhere
* surround-only — annulus (grey disc at center, grating outside)

For each condition we record mean population rates for L23 E, L23 SOM,
L23 PV and H E, and a split of L23 E rates by retinotopic RF centroid
(center-preferring vs surround-preferring units). The decision rules
laid out by Lead:

* r_som(full) ≈ r_som(center):          SOM not pooling surround → spatial-RF problem
* r_som(full) meaningfully > r_som(center) but SI < 0:
                                        SOM sees surround; output projection too
                                        weak, OR direct L4→L23E surround drive
                                        outruns SOM-mediated suppression.
* r_som(full) much greater than center but SI negative:
                                        SOM fighting a stronger surround-driven
                                        facilitation pathway.

No training, no plasticity — pure forward probes.
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
from scripts.v2.phase_a_static_sanity import (
    _build_fresh_network,
    _probe_steady_state,
)
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network


def _l23e_rf_centroid_mask(net: V2Network, cfg: ModelConfig) -> np.ndarray:
    """Return a boolean ``[n_l23_e]`` mask: True = center-preferring.

    L4 layout is (retino_i, retino_j, ori) at 4×4 retinotopic grid over
    a 32×32 input (pool=8 px per retino cell). Each L4 unit represents
    a sub-region centered at pixel (retino_i*8 + 4, retino_j*8 + 4).
    We compute each L23E unit's RF centroid (in pixel coordinates) as
    the softplus(W_l4_l23_raw)-weighted average L4 pixel position, then
    classify as center if the centroid is within ``radius=4`` pixels of
    the grid center.
    """
    w = torch.nn.functional.softplus(net.l23_e.W_l4_l23_raw).detach().cpu().numpy()
    # w is [n_l23_e, n_l4_e=128].
    n_ori = cfg.arch.n_orientations
    retino_side = int(round(math.sqrt(cfg.arch.n_l4_e / n_ori)))   # 4
    pool = cfg.arch.grid_h // retino_side                           # 8
    cx = (cfg.arch.grid_w - 1) / 2.0
    cy = (cfg.arch.grid_h - 1) / 2.0

    # Per-L4 pixel center.
    l4_idx = np.arange(cfg.arch.n_l4_e)
    # layout: u = retino_i * (retino_side * n_ori) + retino_j * n_ori + ori
    retino_i = (l4_idx // (retino_side * n_ori))
    retino_j = (l4_idx // n_ori) % retino_side
    px = retino_j * pool + (pool - 1) / 2.0
    py = retino_i * pool + (pool - 1) / 2.0

    weights = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    cx_u = (weights * px[None, :]).sum(axis=1)
    cy_u = (weights * py[None, :]).sum(axis=1)
    dist = np.sqrt((cx_u - cx) ** 2 + (cy_u - cy) ** 2)
    return dist <= 4.0                                             # bool [n_l23_e]


def run(
    *, n_orients: int = 5, n_trials: int = 20,
    contrast: float = 1.0, center_radius: int = 4,
    n_steps: int = 100, avg_last: int = 50,
) -> dict[str, Any]:
    cfg, net = _build_fresh_network(seed=42)           # TokenBank init uses grad
    # All subsequent probes are forward-only.
    center_mask = _l23e_rf_centroid_mask(net, cfg)
    n_center = int(center_mask.sum())
    n_surround = int((~center_mask).sum())

    orientations = np.linspace(0.0, 180.0, n_orients, endpoint=False)

    # Helper: build one trial's frames for each condition.
    def _frame(cond: str, ori: float) -> Tensor:
        if cond == "center_only":
            return make_surround_grating_frame(
                float(ori), contrast, cfg,
                center_radius=center_radius, include_surround=False,
                batch_size=n_trials,
            )
        if cond == "full_field":
            return make_grating_frame(
                float(ori), contrast, cfg, batch_size=n_trials,
            )
        if cond == "surround_only":
            # Annulus: full-field grating minus center disc (grey inside).
            # Reuse make_surround_grating_frame logic by inverting the mask.
            full = make_grating_frame(
                float(ori), contrast, cfg, batch_size=n_trials,
            )
            H, W = cfg.arch.grid_h, cfg.arch.grid_w
            ys = torch.arange(H, dtype=torch.float32) - (H - 1) / 2.0
            xs = torch.arange(W, dtype=torch.float32) - (W - 1) / 2.0
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            r = torch.sqrt(yy * yy + xx * xx)
            center_grey = (r <= float(center_radius)).to(full.dtype)
            grey = torch.full_like(full, 0.5)
            return grey * center_grey + full * (1.0 - center_grey)
        raise ValueError(cond)

    results: dict[str, dict[str, Any]] = {}
    for cond in ("center_only", "full_field", "surround_only"):
        # Accumulators per condition.
        r_l23_per_unit = np.zeros(cfg.arch.n_l23_e, dtype=np.float64)
        r_som_mean = 0.0
        r_pv_mean = 0.0
        r_h_mean = 0.0
        r_l4_mean = 0.0
        for oi, ori in enumerate(orientations):
            frame = _frame(cond, float(ori))
            probe = _probe_steady_state(
                net, frame, n_steps, avg_last=avg_last,
                trial_seed=1000 + oi * 7 + hash(cond) % 31,
            )
            r_l23_per_unit += probe["r_l23"].mean(axis=0)          # over trials
            r_som_mean += probe["r_som"].mean()
            r_pv_mean += probe["r_pv"].mean()
            r_h_mean += probe["r_h"].mean()
            r_l4_mean += probe["r_l4"].mean()
        r_l23_per_unit /= n_orients
        r_som_mean /= n_orients
        r_pv_mean /= n_orients
        r_h_mean /= n_orients
        r_l4_mean /= n_orients

        r_l23e_center = (
            float(r_l23_per_unit[center_mask].mean()) if n_center else float("nan")
        )
        r_l23e_surround = (
            float(r_l23_per_unit[~center_mask].mean()) if n_surround else float("nan")
        )
        results[cond] = {
            "r_l23e_mean": float(r_l23_per_unit.mean()),
            "r_l23e_center_unit": r_l23e_center,
            "r_l23e_surround_unit": r_l23e_surround,
            "r_som_mean": float(r_som_mean),
            "r_pv_mean": float(r_pv_mean),
            "r_h_mean": float(r_h_mean),
            "r_l4_mean": float(r_l4_mean),
        }

    # Derived comparisons (Lead's decision rules).
    r_som_center = results["center_only"]["r_som_mean"]
    r_som_full = results["full_field"]["r_som_mean"]
    r_som_surround = results["surround_only"]["r_som_mean"]
    r_l23_center = results["center_only"]["r_l23e_mean"]
    r_l23_full = results["full_field"]["r_l23e_mean"]

    si_rpop = (
        (r_l23_center - r_l23_full) / r_l23_center if r_l23_center > 1e-6
        else float("nan")
    )

    verdict = {
        "r_som_full_over_center_ratio": (
            r_som_full / r_som_center if r_som_center > 1e-6 else float("nan")
        ),
        "r_som_surround_over_center_ratio": (
            r_som_surround / r_som_center if r_som_center > 1e-6 else float("nan")
        ),
        "si_population_level": si_rpop,
        "l23e_full_over_center_ratio": (
            r_l23_full / r_l23_center if r_l23_center > 1e-6 else float("nan")
        ),
        "n_l23e_center_preferring": n_center,
        "n_l23e_surround_preferring": n_surround,
    }
    return {
        "protocol": "phase_a_surround_diagnostic",
        "config_notes": {
            "fix_E_init_mean_W_l23_som_raw": -4.5,
            "center_radius_px": int(center_radius),
            "n_orients": n_orients,
            "n_trials": n_trials,
            "n_steps": n_steps,
            "avg_last": avg_last,
        },
        "conditions": results,
        "verdict": verdict,
    }


def main() -> None:
    t0 = time.time()
    out_dir = Path("logs/task74")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase_a_surround_diagnostic.json"

    print("[diag] running center/full/surround probes on Fix-E network...")
    result = run()

    # One line per condition.
    for cond, r in result["conditions"].items():
        print(
            f"  {cond:13s} "
            f"r_l23e_mean={r['r_l23e_mean']:.3f} "
            f"r_l23e_center_unit={r['r_l23e_center_unit']:.3f} "
            f"r_l23e_surround_unit={r['r_l23e_surround_unit']:.3f} "
            f"r_som_mean={r['r_som_mean']:.3f} "
            f"r_pv_mean={r['r_pv_mean']:.3f} "
            f"r_h_mean={r['r_h_mean']:.3f}"
        )

    v = result["verdict"]
    print(
        f"[diag] verdict: r_som(full)/r_som(center)={v['r_som_full_over_center_ratio']:.2f} "
        f"r_som(surround)/r_som(center)={v['r_som_surround_over_center_ratio']:.2f} "
        f"l23e(full)/l23e(center)={v['l23e_full_over_center_ratio']:.2f} "
        f"SI_pop={v['si_population_level']:.3f}"
    )

    result["wall_seconds"] = float(time.time() - t0)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[diag] wrote {out_path}  wall={result['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
