"""Task #52 parameter sweep — explores init_mean grid to find an operating point.

Uses :func:`scripts.v2._calibrate_task52.run_calibration` with per-call
`overrides` to scan candidate init_means. Targets (relaxed per Lead
2026-04-19):

* T1: L23E median rate ∈ [0.01, 0.5] at blank.
* T3: HE median rate < L23E median rate.
* T4: PV/SOM/HPV respond when L23E > 0 (probed indirectly by rate > 0).
* T5 (HARD relaxed): ρ(J_full) < 1.0 — the slow-integrator eigenvalue at
  ~0.99 (context memory's exp(-dt/τ_m) leak) is accepted as biologically
  appropriate for Kok-style cue-delay bridging.
* T6: ||x_hat||_2 / ||r_l4||_2 ∈ [0.1, 10].

Output a ranked table of configurations with a single "best" pick.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict

from scripts.v2._calibrate_task52 import (
    CalibReport, run_calibration, _evaluate_targets,
)


def _score(r: CalibReport) -> tuple[int, float]:
    """Simple scoring: (num targets passed, tie-break by L23-in-range closeness).

    Tie-break prefers r_l23 closer to the middle of [0.01, 0.5] (log-geometric
    mid ≈ 0.07) — a stable operating point rather than edge cases.
    """
    tgs = _evaluate_targets(r)
    n_pass = sum(1 for ok, _ in tgs.values() if ok)
    # Target r_l23 ≈ 0.07 (geometric mid of [0.01, 0.5]).
    import math as _m
    l23_penalty = abs(_m.log10(max(r.r_l23_median, 1e-9)) - _m.log10(0.07))
    return (n_pass, -l23_penalty)


def sweep_phase_a() -> list[tuple[dict, CalibReport]]:
    """Phase A — raise L23E drive via W_l4_l23 (L4 is r_l4≈5e-5 locked by LGN)."""
    results: list[tuple[dict, CalibReport]] = []
    # W_l4_l23 raw ∈ {+1, +2, +3, +4}: softplus(3)=3.05, drive=128·3.05·5e-5≈0.02.
    grid = [1.0, 2.0, 3.0, 4.0]
    for w_l4 in grid:
        overrides = {"l23_e.W_l4_l23_raw": w_l4}
        try:
            r = run_calibration(overrides=overrides)
            results.append((overrides, r))
        except Exception as e:  # noqa: BLE001
            results.append((overrides, f"FAIL: {e}"))  # type: ignore[arg-type]
    return results


def sweep_phase_b(w_l4_base: float) -> list[tuple[dict, CalibReport]]:
    """Phase B — given a good W_l4_l23, tune inhibitory W_l23_pv/som/hpv.

    Goal: PV/SOM/HPV respond at the blank operating point without
    destabilising L23/HE. Raise from -3 / -1 toward -1 / 0.
    """
    results: list[tuple[dict, CalibReport]] = []
    for w_pv, w_som in itertools.product(
        [-3.0, -2.0, -1.0], [-3.0, -2.0, -1.0]
    ):
        overrides = {
            "l23_e.W_l4_l23_raw": w_l4_base,
            "l23_pv.W_pre_raw": w_pv,
            "l23_som.W_l23_som_raw": w_som,
            "h_pv.W_pre_raw": w_pv,  # mirror for HPV
        }
        try:
            r = run_calibration(overrides=overrides)
            results.append((overrides, r))
        except Exception as e:  # noqa: BLE001
            results.append((overrides, f"FAIL: {e}"))  # type: ignore[arg-type]
    return results


def sweep_phase_c(base: dict) -> list[tuple[dict, CalibReport]]:
    """Phase C — lower prediction_head bias/weights so x_hat ~ r_l4."""
    results: list[tuple[dict, CalibReport]] = []
    for b_pred, w_pred in itertools.product(
        [-10.0, -9.0, -8.0, -7.0], [-10.0, -8.0, -6.0]
    ):
        ov = dict(base)
        ov.update({
            "prediction_head.b_pred_raw": b_pred,
            "prediction_head.W_pred_H_raw": w_pred,
            "prediction_head.W_pred_C_raw": w_pred,
            "prediction_head.W_pred_apical_raw": w_pred,
        })
        try:
            r = run_calibration(overrides=ov)
            results.append((ov, r))
        except Exception as e:  # noqa: BLE001
            results.append((ov, f"FAIL: {e}"))  # type: ignore[arg-type]
    return results


def _print_row(overrides: dict, r) -> None:
    if not isinstance(r, CalibReport):
        print(f"  {overrides} -> {r}")
        return
    tgs = _evaluate_targets(r)
    passes = "".join("P" if ok else "." for ok, _ in tgs.values())
    print(
        f"  [{passes}]  "
        f"r_l23={r.r_l23_median:.4g} r_h={r.r_h_median:.4g} "
        f"r_pv={r.r_pv_median:.4g} r_som={r.r_som_median:.4g} "
        f"hpv={r.h_pv_median:.4g}  λ={r.lambda_max_full:.3f}  "
        f"x̂/r_l4={r.x_hat_l2/max(r.r_l4_l2,1e-9):.2g}  "
        f"| {overrides}"
    )


def main() -> None:
    print("=" * 80)
    print("Phase A — raise L23E drive via W_l4_l23")
    print("=" * 80)
    a = sweep_phase_a()
    for ov, r in a:
        _print_row(ov, r)
    # Pick best from Phase A.
    valid_a = [(o, r) for o, r in a if isinstance(r, CalibReport)]
    valid_a.sort(key=lambda x: _score(x[1]), reverse=True)
    best_a = valid_a[0]
    print(f"\nBest from A: {best_a[0]}\n")
    w_l4_base = best_a[0]["l23_e.W_l4_l23_raw"]

    print("=" * 80)
    print("Phase B — inhibitory init_means given W_l4_l23 from A")
    print("=" * 80)
    b = sweep_phase_b(w_l4_base)
    for ov, r in b:
        _print_row(ov, r)
    valid_b = [(o, r) for o, r in b if isinstance(r, CalibReport)]
    valid_b.sort(key=lambda x: _score(x[1]), reverse=True)
    best_b = valid_b[0]
    print(f"\nBest from B: {best_b[0]}\n")

    print("=" * 80)
    print("Phase C — prediction head re-scaling given A+B")
    print("=" * 80)
    c = sweep_phase_c(best_b[0])
    for ov, r in c:
        _print_row(ov, r)
    valid_c = [(o, r) for o, r in c if isinstance(r, CalibReport)]
    valid_c.sort(key=lambda x: _score(x[1]), reverse=True)
    best_c = valid_c[0]
    print(f"\nBest from C: {best_c[0]}\n")

    print("=" * 80)
    print("FINAL BEST")
    print("=" * 80)
    print(json.dumps(best_c[0], indent=2))
    print(json.dumps(asdict(best_c[1]), indent=2))


if __name__ == "__main__":
    main()
