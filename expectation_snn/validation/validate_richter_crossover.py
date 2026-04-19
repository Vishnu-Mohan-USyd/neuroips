"""Functional component validation for the Richter cross-over assay.

Per-component rule (task #27): passes BEFORE Sprint 5a run.

Checks (all local; tiny end-to-end smoke):

1. Schedule totals: 12 × reps_per_pair = n_trials at defaults (360).
2. Pair balance: 30 reps each of 12 distinct pair types at defaults.
3. Expected pairs: θ_L = θ_T.
4. Unexpected pairs: θ_L − θ_T ≡ π/2 (mod π).
5. Equal exp/unexp counts (180/180 at defaults).
6. End-to-end smoke: tiny config runs, all 4 primary metrics + 6 voxel
   families return well-formed outputs.

Usage
-----
    python -m expectation_snn.validation.validate_richter_crossover
"""
from __future__ import annotations

import numpy as np

from expectation_snn.assays.richter_crossover import (
    RichterConfig, RichterResult,
    build_richter_schedule, run_richter_crossover,
    _richter_thetas_rad,
)
from expectation_snn.assays.metrics import _VOXEL_MODEL_FAMILIES


def _wrap_pi(x: float) -> float:
    x = x % np.pi
    return min(x, np.pi - x)


def test_schedule_totals() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    assert len(sch) == cfg.n_trials, (len(sch), cfg.n_trials)


def test_pair_balance() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    pair_counts: dict = {}
    for t in sch:
        pid = t["pair_id"]
        pair_counts[pid] = pair_counts.get(pid, 0) + 1
    # Should be exactly 12 unique pair ids, each with reps_per_pair count.
    assert len(pair_counts) == 12, len(pair_counts)
    for pid, c in pair_counts.items():
        assert c == cfg.reps_per_pair, (pid, c, cfg.reps_per_pair)


def test_expected_pairs_same_orientation() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    for t in sch:
        if t["condition"] == 1:
            assert abs(t["theta_L"] - t["theta_T"]) < 1e-9, t


def test_unexpected_pairs_orthogonal() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    for t in sch:
        if t["condition"] == 0:
            d = _wrap_pi(t["theta_L"] - t["theta_T"])
            assert abs(d - np.pi / 2) < 1e-9, t


def test_equal_exp_unexp() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    n_exp = sum(1 for t in sch if t["condition"] == 1)
    n_unexp = sum(1 for t in sch if t["condition"] == 0)
    assert n_exp == n_unexp, (n_exp, n_unexp)
    assert n_exp == cfg.n_trials // 2, n_exp


def test_orientations_set() -> None:
    thetas = _richter_thetas_rad()
    assert thetas.shape == (6,)
    # In radians: {0, 30, 60, 90, 120, 150} deg
    expected = np.deg2rad([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
    assert np.allclose(thetas, expected), (thetas, expected)


def test_end_to_end_smoke() -> None:
    """Tiny 24-trial run exercises each metric path."""
    cfg = RichterConfig(
        n_trials=24, reps_per_pair=2,
        leader_ms=200.0, trailer_ms=200.0, iti_ms=200.0,
        seed=42,
    )
    r = run_richter_crossover(cfg=cfg)
    assert isinstance(r, RichterResult)
    # Metric 1: preference-rank
    bd = r.pref_rank["bin_delta"]
    assert bd.shape == (10,) and np.all(np.isfinite(bd))
    # Metric 2: feature-distance surface
    grid = r.feature_distance["grid"]
    assert grid.shape == (8, 8)
    # Metric 3: cell-type gain matrix
    ctg = r.cell_type_gain["delta_hz"]
    assert ctg.shape == (3, 3)
    assert r.cell_type_gain["pops"] == ("E", "SOM", "PV")
    assert r.cell_type_gain["dists"] == ("local", "nbr", "far")
    # Metric 4: center-vs-flank
    cvf = r.center_vs_flank
    assert np.isfinite(cvf["center_delta"])
    assert np.isfinite(cvf["flank_delta"])
    assert np.isfinite(cvf["redist"])
    # Secondary: 6-family voxel forward
    assert set(r.voxel_forward.keys()) == set(_VOXEL_MODEL_FAMILIES)
    for fam, out in r.voxel_forward.items():
        assert out["voxel_tuning_baseline"].shape == (cfg.n_voxels, 6)
        assert out["voxel_tuning_predicted"].shape == (cfg.n_voxels, 6)


def test_spike_counts_shape() -> None:
    """Shape sanity on raw counts."""
    cfg = RichterConfig(
        n_trials=24, reps_per_pair=2,
        leader_ms=100.0, trailer_ms=100.0, iti_ms=100.0,
        seed=42,
    )
    r = run_richter_crossover(cfg=cfg)
    tc_e = r.raw["trailer_counts_e"]
    tc_som = r.raw["trailer_counts_som"]
    tc_pv = r.raw["trailer_counts_pv"]
    assert tc_e.shape[1] == cfg.n_trials
    assert tc_som.shape[1] == cfg.n_trials
    assert tc_pv.shape[1] == cfg.n_trials
    assert (tc_e >= 0).all() and (tc_som >= 0).all() and (tc_pv >= 0).all()


_CHECKS = [
    ("schedule_totals", test_schedule_totals),
    ("pair_balance", test_pair_balance),
    ("expected_pairs_same_orientation", test_expected_pairs_same_orientation),
    ("unexpected_pairs_orthogonal", test_unexpected_pairs_orthogonal),
    ("equal_exp_unexp", test_equal_exp_unexp),
    ("orientations_set", test_orientations_set),
    ("end_to_end_smoke", test_end_to_end_smoke),
    ("spike_counts_shape", test_spike_counts_shape),
]


def main() -> int:
    n_pass = 0
    n_fail = 0
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            n_fail += 1
    total = n_pass + n_fail
    print(f"\nvalidate_richter_crossover: {n_pass}/{total} PASS")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
