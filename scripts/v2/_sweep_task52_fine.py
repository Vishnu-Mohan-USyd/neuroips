"""Task #52 fine-grained sweep — find PV/SOM operating point within stability.

Starts from Phase-C best (l4=+4, b_pred=-10, W_pred_*=-10) and searches a
2D grid over (raise-L23→inh-drive, lower-inh→L23-feedback) so interneurons
fire at blank *and* the Jacobian stays < 1.0.

Design insight
--------------
PV/SOM/HPV are silent because drive = W_l23_pv · r_l23 is below the
target_rate_hz=1.0 threshold at the L23E operating rate (~0.016). Raising
W_l23_pv pushes interneurons above threshold, but also amplifies the
L23↔PV negative-feedback loop, which a prior sweep showed drives ρ(J) >
1.67 when W_l23_pv = -1.0 and W_pv_l23 stays at -3.0. Lowering W_pv_l23
(weaker PV→L23 inhibition per unit) breaks the loop gain while keeping
the drive gain high.

We search (W_l23_pv, W_pv_l23) with the same strategy for (W_l23_som,
W_som_l23) and (W_pre_hpv, W_pv_h).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict

from scripts.v2._calibrate_task52 import (
    CalibReport, run_calibration, _evaluate_targets,
)


BASE_OVERRIDES = {
    "l23_e.W_l4_l23_raw": 4.0,
    # Self-rec and FB-apical: keep prior coder's values (stability-guarded).
    # L23E.W_rec_raw stays at -5.0; L23E.W_fb_apical_raw stays at -5.0.
    "prediction_head.b_pred_raw": -10.0,
    "prediction_head.W_pred_H_raw": -10.0,
    "prediction_head.W_pred_C_raw": -10.0,
    "prediction_head.W_pred_apical_raw": -10.0,
}


def _score(r: CalibReport) -> tuple:
    tgs = _evaluate_targets(r)
    n_pass = sum(1 for ok, _ in tgs.values() if ok)
    # Tie-break: prefer λ_full < 1.0 with some margin; penalise ρ near 1.0.
    margin = 1.0 - r.lambda_max_full
    return (n_pass, margin)


def sweep() -> list[tuple[dict, CalibReport]]:
    results: list[tuple[dict, CalibReport]] = []
    # Grid: raise L23→PV drive and lower PV→L23 feedback in parallel.
    for (w_l23_pv, w_pv_l23) in itertools.product(
        [-1.5, -1.2, -1.0, -0.5], [-5.0, -4.0, -3.0]
    ):
        # Mirror for SOM (SOM threshold logic identical to PV's).
        for (w_l23_som, w_som_l23) in [
            (-1.5, -5.0), (-1.2, -5.0), (-1.0, -4.0),
        ]:
            # HPV: L23E→HE drive is much smaller than L23E→PV (different
            # pool sizes), so HPV drive scales with r_h not r_l23.
            # r_h ~ 0.007, so HPV drive = softplus(W_pre) · 64 · 0.007.
            # To exceed 1.0: softplus(W_pre) > 2.23, i.e. W_pre > 2.
            # Very aggressive — verify loop stability.
            for w_pre_hpv in [-1.0, 0.0, 1.0, 2.0]:
                for w_pv_h in [-3.0, -4.0, -5.0]:
                    ov = dict(BASE_OVERRIDES)
                    ov.update({
                        "l23_pv.W_pre_raw": w_l23_pv,
                        "l23_e.W_pv_l23_raw": w_pv_l23,
                        "l23_som.W_l23_som_raw": w_l23_som,
                        "l23_e.W_som_l23_raw": w_som_l23,
                        "h_pv.W_pre_raw": w_pre_hpv,
                        "h_e.W_pv_h_raw": w_pv_h,
                    })
                    try:
                        r = run_calibration(overrides=ov)
                        results.append((ov, r))
                    except Exception as e:  # noqa: BLE001
                        results.append((ov, f"FAIL: {e}"))  # type: ignore[arg-type]
    return results


def main() -> None:
    res = sweep()
    valid = [(o, r) for o, r in res if isinstance(r, CalibReport)]
    # Filter ones that pass T5 hard (< 1.0) first.
    valid_stable = [(o, r) for o, r in valid if r.lambda_max_full < 1.0]
    print(f"Total probed: {len(res)}, finite: {len(valid)}, stable: {len(valid_stable)}")
    valid_stable.sort(key=lambda x: _score(x[1]), reverse=True)
    print("\nTop 10 stable results (by score):")
    for ov, r in valid_stable[:10]:
        tgs = _evaluate_targets(r)
        passes = "".join("P" if ok else "." for ok, _ in tgs.values())
        print(
            f"  [{passes}]  "
            f"r_l23={r.r_l23_median:.4g} r_h={r.r_h_median:.4g} "
            f"r_pv={r.r_pv_median:.4g} r_som={r.r_som_median:.4g} "
            f"hpv={r.h_pv_median:.4g}  λ={r.lambda_max_full:.4f}  "
            f"x̂/r_l4={r.x_hat_l2/max(r.r_l4_l2,1e-9):.2g}"
        )
        interesting = {k: v for k, v in ov.items() if k not in BASE_OVERRIDES}
        print(f"      | {interesting}")
    if valid_stable:
        best_ov, best_r = valid_stable[0]
        print("\n" + "=" * 80 + "\nBEST\n" + "=" * 80)
        print(json.dumps(best_ov, indent=2))
        print()
        print(json.dumps(asdict(best_r), indent=2))


if __name__ == "__main__":
    main()
