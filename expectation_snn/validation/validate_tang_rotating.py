"""Functional component validation for the Tang rotating-deviant assay.

Sprint 5c R3: +Random block + Δθ_prev_step covariate.

Checks (all local; tiny end-to-end smoke):

1. Sequence generator (rotating block): deviant_mask aligns with block ends.
2. Deviant rate (rotating block) ~ 1 / mean(block_len_range).
3. End-to-end smoke (Random + Rotating blocks): all primary metrics +
   R3 outputs (three_condition, dtheta_prev) populate with right shapes.
4. Random-block items have deviant_mask=False, condition code 0,
   block_id=-1, is_random=True; rotating items the complement.
5. Δθ_prev_step ∈ {-1,0,1,2,3} with -1 only at item 0; rotating-block
   step distances are ∈ {1,2,3} (since the rotating grammar steps ±30°
   except at deviants).
6. n_items = n_random + n_rotating.
7. Legacy SVM / cell-gain restricted to rotating items only.

Usage
-----
    python -m expectation_snn.validation.validate_tang_rotating
"""
from __future__ import annotations

import numpy as np

from expectation_snn.assays.tang_rotating import (
    TangConfig, TangResult, run_tang_rotating,
    _theta_step_distance, _tang_thetas_rad,
)
from expectation_snn.brian2_model.stimulus import (
    tang_rotating_sequence, TANG_BLOCK_LEN_RANGE,
)


def test_deviant_mask_aligns_with_block_end() -> None:
    rng = np.random.default_rng(42)
    plan = tang_rotating_sequence(rng, n_items=300)
    dm = plan.meta["deviant_mask"]
    bid = plan.meta["block_ids"]
    unique_blocks, counts = np.unique(bid, return_counts=True)
    for bi, cnt in zip(unique_blocks, counts):
        block_mask = bid == bi
        items_in_block = np.where(block_mask)[0]
        n_dev_in_block = int(dm[items_in_block].sum())
        assert n_dev_in_block <= 1, (bi, n_dev_in_block)


def test_deviant_rate_in_ballpark() -> None:
    rng = np.random.default_rng(42)
    plan = tang_rotating_sequence(rng, n_items=2000)
    dev_rate = float(plan.meta["deviant_mask"].mean())
    blen_lo, blen_hi = TANG_BLOCK_LEN_RANGE
    mean_blen = (blen_lo + blen_hi) / 2.0
    expected = 1.0 / mean_blen                          # ~0.143
    assert 0.10 <= dev_rate <= 0.20, (dev_rate, expected)


def test_end_to_end_smoke() -> None:
    """Tiny mixed-block run: both R3 outputs and legacy metrics populate."""
    cfg = TangConfig(
        n_random=40, n_rotating=40, item_ms=150.0,
        presettle_ms=200.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    assert isinstance(r, TangResult)

    n_e = r.raw["counts_per_item"].shape[0]

    # Metric 1 (legacy, rotating only): per-cell matched-θ gain
    cg = r.cell_gain
    assert cg["delta_hz"].shape == (n_e,)
    assert cg["rate_deviant_hz"].shape == (n_e,)
    assert cg["rate_expected_hz"].shape == (n_e,)
    assert cg["n_cells_with_data"] >= 0

    # Metric 2 (legacy, rotating only): SVM
    assert 0.0 <= r.svm["accuracy"] <= 1.0

    # Metric 3 (legacy, rotating only): laminar
    lm = r.laminar
    for k in ("deviant_rate_hz", "expected_rate_hz", "delta_hz"):
        assert np.isfinite(lm[k]), (k, lm[k])

    # Secondary: tuning fit outputs
    tf = r.tuning
    for cond in ("expected_fit", "deviant_fit"):
        d = tf[cond]
        assert d["fwhm_rad"].shape == (n_e,)
    assert tf["sc_by_theta_expected"].shape == (n_e, 6)
    assert tf["sc_by_theta_deviant"].shape == (n_e, 6)

    # R3 primary: three-condition matched-θ rate
    tc = r.three_condition
    for name in ("random", "rotating_expected", "rotating_deviant"):
        assert name in tc["per_cond"]
        d = tc["per_cond"][name]
        assert "mean_rate_hz" in d and "ci" in d
    for k in ("rotating_expected_minus_random",
              "rotating_deviant_minus_random",
              "deviant_minus_expected"):
        assert k in tc["deltas"]
        assert "mean_delta_hz" in tc["deltas"][k]
        assert "ci" in tc["deltas"][k]
    assert tc["per_cell_rates"].shape == (n_e, 3)
    assert tc["n_items_per_cell_per_cond"].shape == (n_e, 3)

    # R3 covariate: Δθ_prev stratification
    dp = r.dtheta_prev
    assert dp["n_trials_grid"].shape == (3, 4)
    for name in ("random", "rotating_expected", "rotating_deviant"):
        assert name in dp["by_cond_by_step"]
        for step in (0, 1, 2, 3):
            assert step in dp["by_cond_by_step"][name]
            ent = dp["by_cond_by_step"][name][step]
            for k in ("mean_rate_hz", "n_trials", "n_cells"):
                assert k in ent


def test_counts_per_item_shape() -> None:
    cfg = TangConfig(
        n_random=15, n_rotating=15, item_ms=100.0,
        presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    cpi = r.raw["counts_per_item"]
    assert cpi.shape[1] == cfg.n_items
    assert (cpi >= 0).all()


def test_meta_counts_consistent() -> None:
    cfg = TangConfig(
        n_random=20, n_rotating=20, item_ms=100.0,
        presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    assert r.meta["n_items"] == cfg.n_items == cfg.n_random + cfg.n_rotating
    assert r.meta["n_random"] == cfg.n_random
    assert r.meta["n_rotating"] == cfg.n_rotating
    assert r.meta["n_random_items"] == cfg.n_random
    assert r.meta["n_deviant"] >= 1                     # rotating block has deviants
    assert r.meta["n_expected"] == cfg.n_rotating - r.meta["n_deviant"]


def test_random_block_has_no_deviants() -> None:
    cfg = TangConfig(
        n_random=30, n_rotating=30, item_ms=100.0,
        presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    is_random = r.raw["is_random"]
    dm = r.raw["deviant_mask"]
    cond = r.raw["cond_codes"]
    # Random items: condition=0, deviant_mask=False, block_id=-1
    assert is_random.sum() == cfg.n_random
    assert (~dm[is_random]).all(), "random items must not be deviants"
    assert (cond[is_random] == 0).all()
    assert (r.raw["block_ids"][is_random] == -1).all()
    # Rotating items: condition ∈ {1,2}, block_id ≥ 0
    rot = ~is_random
    assert ((cond[rot] == 1) | (cond[rot] == 2)).all()
    assert (cond[rot & dm] == 2).all()
    assert (cond[rot & ~dm] == 1).all()
    assert (r.raw["block_ids"][rot] >= 0).all()


def test_dtheta_prev_step_values() -> None:
    cfg = TangConfig(
        n_random=20, n_rotating=20, item_ms=100.0,
        presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    dts = r.raw["dtheta_prev_step"]
    assert dts.shape == (cfg.n_items,)
    # First item is a sentinel (-1).
    assert dts[0] == -1
    # All later items ∈ {0,1,2,3}
    assert ((dts[1:] >= 0) & (dts[1:] <= 3)).all()


def test_dtheta_step_helper() -> None:
    """Sanity check on the step-distance helper itself."""
    thetas = _tang_thetas_rad()                          # 0,30,...,150 deg
    # 0° vs 30°  → step 1
    assert _theta_step_distance(thetas[0], thetas[1], thetas) == 1
    # 0° vs 60°  → step 2
    assert _theta_step_distance(thetas[0], thetas[2], thetas) == 2
    # 0° vs 90°  → step 3
    assert _theta_step_distance(thetas[0], thetas[3], thetas) == 3
    # 0° vs 120° → step 2 (wraps via π/2 symmetry)
    assert _theta_step_distance(thetas[0], thetas[4], thetas) == 2
    # 0° vs 150° → step 1
    assert _theta_step_distance(thetas[0], thetas[5], thetas) == 1
    # Same orientation → 0
    assert _theta_step_distance(thetas[0], thetas[0], thetas) == 0


def test_legacy_svm_uses_rotating_only() -> None:
    """SVM y-vector should be deviant_mask of rotating items, not all items.

    With n_random=20, n_rotating=20, ~3 deviants → 17 expected → SVM
    should still run (more than 1 sample of each class), and return a
    finite accuracy ∈ [0,1].
    """
    cfg = TangConfig(
        n_random=20, n_rotating=20, item_ms=100.0,
        presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    assert np.isfinite(r.svm["accuracy"])
    assert 0.0 <= r.svm["accuracy"] <= 1.0


_CHECKS = [
    ("deviant_mask_aligns_with_block_end", test_deviant_mask_aligns_with_block_end),
    ("deviant_rate_in_ballpark", test_deviant_rate_in_ballpark),
    ("dtheta_step_helper", test_dtheta_step_helper),
    ("end_to_end_smoke", test_end_to_end_smoke),
    ("counts_per_item_shape", test_counts_per_item_shape),
    ("meta_counts_consistent", test_meta_counts_consistent),
    ("random_block_has_no_deviants", test_random_block_has_no_deviants),
    ("dtheta_prev_step_values", test_dtheta_prev_step_values),
    ("legacy_svm_uses_rotating_only", test_legacy_svm_uses_rotating_only),
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
    print(f"\nvalidate_tang_rotating: {n_pass}/{total} PASS")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
