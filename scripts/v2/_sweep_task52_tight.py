"""Task #52 tight sweep — minimal grid to find T4-passing stable config.

Strategy: drop the PV→L23 and SOM→L23 feedback gains (W_pv_l23,
W_som_l23, W_pv_h) to -5 so the inhibitory→excitatory loop is weak, then
raise the L23→PV / L23→SOM / HE→HPV drives to push interneurons above
the target_rate_hz=1.0 threshold at the L23E blank operating rate.
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
    "l23_e.W_pv_l23_raw": -5.0,     # weak PV→L23 feedback (was -3)
    "l23_e.W_som_l23_raw": -5.0,    # weak SOM→L23 feedback (was -2.5)
    "h_e.W_pv_h_raw": -5.0,         # weak HPV→HE feedback (was -3)
    "prediction_head.b_pred_raw": -10.0,
    "prediction_head.W_pred_H_raw": -10.0,
    "prediction_head.W_pred_C_raw": -10.0,
    "prediction_head.W_pred_apical_raw": -10.0,
}


def _score(r: CalibReport) -> tuple:
    tgs = _evaluate_targets(r)
    n_pass = sum(1 for ok, _ in tgs.values() if ok)
    margin = 1.0 - r.lambda_max_full
    return (n_pass, margin)


def sweep() -> list[tuple[dict, CalibReport]]:
    results: list[tuple[dict, CalibReport]] = []
    for w_l23_pv, w_l23_som, w_pre_hpv in itertools.product(
        [-1.3, -1.0, -0.5],
        [-1.3, -1.0, -0.5],
        [1.0, 2.0, 2.5, 3.0],
    ):
        ov = dict(BASE_OVERRIDES)
        ov.update({
            "l23_pv.W_pre_raw": w_l23_pv,
            "l23_som.W_l23_som_raw": w_l23_som,
            "h_pv.W_pre_raw": w_pre_hpv,
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
    valid_stable = [(o, r) for o, r in valid if r.lambda_max_full < 1.0]
    print(f"Total probed: {len(res)}, finite: {len(valid)}, stable: {len(valid_stable)}")
    valid_stable.sort(key=lambda x: _score(x[1]), reverse=True)

    for ov, r in valid_stable:
        tgs = _evaluate_targets(r)
        passes = "".join("P" if ok else "." for ok, _ in tgs.values())
        interesting = {
            "Wl23pv": ov["l23_pv.W_pre_raw"],
            "Wl23som": ov["l23_som.W_l23_som_raw"],
            "Whpv": ov["h_pv.W_pre_raw"],
        }
        print(
            f"  [{passes}]  "
            f"r_l23={r.r_l23_median:.4g} r_h={r.r_h_median:.4g} "
            f"r_pv={r.r_pv_median:.4g} r_som={r.r_som_median:.4g} "
            f"hpv={r.h_pv_median:.4g}  λ={r.lambda_max_full:.4f}  "
            f"| {interesting}"
        )

    if valid_stable:
        best_ov, best_r = valid_stable[0]
        print("\nBEST\n" + "=" * 80)
        print(json.dumps(best_ov, indent=2))
        print()
        print(json.dumps(asdict(best_r), indent=2))


if __name__ == "__main__":
    main()
