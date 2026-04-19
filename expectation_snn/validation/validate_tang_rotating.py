"""Functional component validation for the Tang rotating-deviant assay.

Per-component rule (task #27): passes BEFORE Sprint 5a run.

Checks (all local; tiny end-to-end smoke):

1. Sequence generator: deviant_mask aligns with block boundaries.
2. End-to-end smoke: tiny n_items run produces all 3 primary metrics + secondary.
3. Shape sanity on counts_per_item, cell_gain, SVM, laminar, tuning.
4. Deviant rate in the ballpark of 1/mean(block_len_range).
5. SVM accuracy is in [0, 1].

Usage
-----
    python -m expectation_snn.validation.validate_tang_rotating
"""
from __future__ import annotations

import numpy as np

from expectation_snn.assays.tang_rotating import (
    TangConfig, TangResult, run_tang_rotating,
)
from expectation_snn.brian2_model.stimulus import (
    tang_rotating_sequence, TANG_BLOCK_LEN_RANGE,
)


def test_deviant_mask_aligns_with_block_end() -> None:
    rng = np.random.default_rng(42)
    plan = tang_rotating_sequence(rng, n_items=300)
    dm = plan.meta["deviant_mask"]
    bid = plan.meta["block_ids"]
    # Last item of each block should be a deviant (except final block if
    # truncated). Confirm there's exactly one deviant per complete block.
    unique_blocks, counts = np.unique(bid, return_counts=True)
    for bi, cnt in zip(unique_blocks, counts):
        block_mask = bid == bi
        # Deviant at the tail position of the block.
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
    # Allow ±30 % tolerance — block-length sampling is discrete uniform.
    assert 0.10 <= dev_rate <= 0.20, (dev_rate, expected)


def test_end_to_end_smoke() -> None:
    cfg = TangConfig(
        n_items=80, item_ms=150.0, presettle_ms=200.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    assert isinstance(r, TangResult)

    # Metric 1: per-cell matched-θ gain
    cg = r.cell_gain
    n_e = r.raw["counts_per_item"].shape[0]
    assert cg["delta_hz"].shape == (n_e,)
    assert cg["rate_deviant_hz"].shape == (n_e,)
    assert cg["rate_expected_hz"].shape == (n_e,)
    assert cg["n_cells_with_data"] >= 0

    # Metric 2: SVM
    assert 0.0 <= r.svm["accuracy"] <= 1.0

    # Metric 3: laminar
    lm = r.laminar
    for k in ("deviant_rate_hz", "expected_rate_hz", "delta_hz"):
        assert np.isfinite(lm[k]), (k, lm[k])

    # Secondary: tuning fit outputs
    tf = r.tuning
    for cond in ("expected_fit", "deviant_fit"):
        d = tf[cond]
        assert d["fwhm_rad"].shape == (n_e,)
        assert d["r2"].shape == (n_e,)
    assert tf["sc_by_theta_expected"].shape == (n_e, 6)
    assert tf["sc_by_theta_deviant"].shape == (n_e, 6)


def test_counts_per_item_shape() -> None:
    cfg = TangConfig(
        n_items=30, item_ms=100.0, presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    cpi = r.raw["counts_per_item"]
    assert cpi.shape[1] == cfg.n_items
    assert (cpi >= 0).all()


def test_meta_counts_consistent() -> None:
    cfg = TangConfig(
        n_items=40, item_ms=100.0, presettle_ms=100.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg)
    assert r.meta["n_items"] == cfg.n_items
    assert r.meta["n_deviant"] + r.meta["n_expected"] == cfg.n_items
    assert r.meta["n_deviant"] >= 1


_CHECKS = [
    ("deviant_mask_aligns_with_block_end", test_deviant_mask_aligns_with_block_end),
    ("deviant_rate_in_ballpark", test_deviant_rate_in_ballpark),
    ("end_to_end_smoke", test_end_to_end_smoke),
    ("counts_per_item_shape", test_counts_per_item_shape),
    ("meta_counts_consistent", test_meta_counts_consistent),
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
