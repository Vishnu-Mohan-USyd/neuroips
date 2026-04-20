"""Functional component validation for the Richter cross-over assay.

Sprint 5c R1 (deranged-permutation) design checks:

1. Schedule totals: 6×reps_expected + 24×reps_unexpected = n_trials (=372 default).
2. Pair balance: 30 pair types total (6 expected + 24 unexpected); reps match cfg.
3. Expected pairs: Δθ_step == 1 and θ_T = θ_L + 30° (mod π).
4. Unexpected pairs: Δθ_step ∈ {2,3,4,5} and θ_T = θ_L + Δθ_step·30° (mod π).
5. Δθ-step balance within unexpected: each k ∈ {2,3,4,5} appears 6×reps_unexpected times.
6. Orientations: 6 evenly spaced 0..150 deg.
7. End-to-end smoke: tiny config runs, all 4 primary metrics + Δθ-stratified output
   + 6 voxel families return well-formed dicts.

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


_STEP_RAD = np.pi / 6.0     # 30° in radians


def _wrap_pi(x: float) -> float:
    """Wrap a difference into [0, π/2] (orientation distance on 0..π ring)."""
    x = x % np.pi
    return min(x, np.pi - x)


def _step_distance(theta_L: float, theta_T: float) -> float:
    """Signed step distance from leader to trailer on the 6-orient ring,
    returned as the positive (mod 6) step (1..5).

    Returns the integer k such that θ_T ≈ θ_L + k·30° (mod π); since the
    ring wraps at 180°, we resolve to k ∈ {1,2,3,4,5} (k=0 / k=6 means
    same orientation, which is excluded from the deranged design).
    """
    raw = (theta_T - theta_L) / _STEP_RAD
    k = int(round(raw)) % 6
    return k


def test_schedule_totals() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    expected_total = cfg.reps_expected * 6 + cfg.reps_unexpected * 24
    assert expected_total == cfg.n_trials, (expected_total, cfg.n_trials)
    assert len(sch) == cfg.n_trials, (len(sch), cfg.n_trials)
    assert cfg.n_trials == 372, f"defaults expected 372 trials, got {cfg.n_trials}"


def test_pair_balance() -> None:
    """Should have 30 unique pair_ids (6 expected + 24 unexpected) at correct reps."""
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    pair_counts: dict = {}
    pair_cond: dict = {}
    for t in sch:
        pid = t["pair_id"]
        pair_counts[pid] = pair_counts.get(pid, 0) + 1
        pair_cond[pid] = t["condition"]
    assert len(pair_counts) == 30, f"expected 30 unique pair_ids, got {len(pair_counts)}"
    n_exp_pairs = sum(1 for c in pair_cond.values() if c == 1)
    n_unexp_pairs = sum(1 for c in pair_cond.values() if c == 0)
    assert n_exp_pairs == 6, n_exp_pairs
    assert n_unexp_pairs == 24, n_unexp_pairs
    for pid, c in pair_counts.items():
        target = cfg.reps_expected if pair_cond[pid] == 1 else cfg.reps_unexpected
        assert c == target, (pid, c, target)


def test_expected_pairs_one_step() -> None:
    """Expected: Δθ_step == 1, i.e. θ_T = θ_L + 30°."""
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    for t in sch:
        if t["condition"] != 1:
            continue
        assert t["dtheta_step"] == 1, t
        k = _step_distance(t["theta_L"], t["theta_T"])
        assert k == 1, (t, k)


def test_unexpected_pairs_step_in_2_to_5() -> None:
    """Unexpected: Δθ_step ∈ {2,3,4,5}; θ_T matches the step distance."""
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    for t in sch:
        if t["condition"] != 0:
            continue
        assert t["dtheta_step"] in (2, 3, 4, 5), t
        k = _step_distance(t["theta_L"], t["theta_T"])
        assert k == t["dtheta_step"], (t, k)


def test_unexpected_step_balance() -> None:
    """Each unexpected step k ∈ {2,3,4,5} appears 6 leaders × reps_unexpected times."""
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    per_k: dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for t in sch:
        per_k[t["dtheta_step"]] += 1
    assert per_k[1] == 6 * cfg.reps_expected, (per_k, cfg.reps_expected)
    for k in (2, 3, 4, 5):
        assert per_k[k] == 6 * cfg.reps_unexpected, (k, per_k[k], cfg.reps_unexpected)


def test_total_exp_unexp_split() -> None:
    cfg = RichterConfig()
    sch = build_richter_schedule(cfg)
    n_exp = sum(1 for t in sch if t["condition"] == 1)
    n_unexp = sum(1 for t in sch if t["condition"] == 0)
    assert n_exp == 6 * cfg.reps_expected, (n_exp, cfg.reps_expected)
    assert n_unexp == 24 * cfg.reps_unexpected, (n_unexp, cfg.reps_unexpected)


def test_orientations_set() -> None:
    thetas = _richter_thetas_rad()
    assert thetas.shape == (6,)
    expected = np.deg2rad([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
    assert np.allclose(thetas, expected), (thetas, expected)


def test_end_to_end_smoke() -> None:
    """Tiny run exercises every metric path including the new Δθ-stratified report."""
    cfg = RichterConfig(
        reps_expected=2, reps_unexpected=1,
        leader_ms=200.0, trailer_ms=200.0, iti_ms=200.0,
        seed=42,
    )
    r = run_richter_crossover(cfg=cfg)
    assert isinstance(r, RichterResult)
    bd = r.pref_rank["bin_delta"]
    assert bd.shape == (10,) and np.all(np.isfinite(bd))
    grid = r.feature_distance["grid"]
    assert grid.shape == (8, 8)
    ctg = r.cell_type_gain["delta_hz"]
    assert ctg.shape == (3, 3)
    assert r.cell_type_gain["pops"] == ("E", "SOM", "PV")
    assert r.cell_type_gain["dists"] == ("local", "nbr", "far")
    cvf = r.center_vs_flank
    assert np.isfinite(cvf["center_delta"])
    assert np.isfinite(cvf["flank_delta"])
    assert np.isfinite(cvf["redist"])
    # Δθ-stratified output: per-step bin_delta + redist_by_step + n_per_step.
    ds = r.dtheta_stratified
    for k in (2, 3, 4, 5):
        assert k in ds["bin_delta_by_step"]
        bdk = ds["bin_delta_by_step"][k]
        assert bdk.shape == (10,)
        rd = ds["redist_by_step"][k]
        assert "center_delta" in rd and "flank_delta" in rd and "redist" in rd
    assert ds["n_trials_per_step"][1] == 6 * cfg.reps_expected
    for k in (2, 3, 4, 5):
        assert ds["n_trials_per_step"][k] == 6 * cfg.reps_unexpected
    # Secondary: 6-family voxel forward
    assert set(r.voxel_forward.keys()) == set(_VOXEL_MODEL_FAMILIES)
    for fam, out in r.voxel_forward.items():
        assert out["voxel_tuning_baseline"].shape == (cfg.n_voxels, 6)
        assert out["voxel_tuning_predicted"].shape == (cfg.n_voxels, 6)


def test_spike_counts_shape() -> None:
    cfg = RichterConfig(
        reps_expected=2, reps_unexpected=1,
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
    assert r.raw["dtheta_step"].shape == (cfg.n_trials,)


_CHECKS = [
    ("schedule_totals", test_schedule_totals),
    ("pair_balance", test_pair_balance),
    ("expected_pairs_one_step", test_expected_pairs_one_step),
    ("unexpected_pairs_step_in_2_to_5", test_unexpected_pairs_step_in_2_to_5),
    ("unexpected_step_balance", test_unexpected_step_balance),
    ("total_exp_unexp_split", test_total_exp_unexp_split),
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
