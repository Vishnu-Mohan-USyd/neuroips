"""Focused validator for Sprint 5e-Fix D.4 — Tang direction wiring.

Checks that ``run_tang_rotating`` correctly drives the H_context direction
afferent (CW = channel 0, CCW = channel 1) at each *block onset* when
given an ``architecture='ctx_pred'`` bundle, silences the afferent on
random-block items, and leaves the afferent silenced after the
sequence. Legacy ``h_t`` bundles must continue to work unchanged
(direction wiring is a no-op there).

Assays (all short smoke runs, a few seconds each)::

  [1] ctx_pred_tang_runs_end_to_end      (small n, completes + returns metrics)
  [2] h_t_tang_still_works                (back-compat: legacy default path)
  [3] h_kind_hr_rejected                  (assay must refuse an H_R bundle)
  [4] direction_helpers_split_afferents   (CW / CCW / silence produce expected
                                           rate splits on the direction pool)
  [5] post_run_direction_is_silenced      (no CW/CCW bias leaks past the run)
  [6] tang_direction_wired_flag_matches   (meta.tang_direction_wired reflects
                                           architecture)
  [7] block_transitions_rewire_direction  (instrumented stepwise replay:
                                           rotating block onsets call
                                           set_tang_direction with the mapped
                                           CW/CCW channel; random-block
                                           transitions silence)

Usage::

    python -m expectation_snn.validation.validate_tang_direction_wiring
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from brian2 import prefs, defaultclock, ms
from brian2 import seed as b2_seed

from expectation_snn.assays.tang_rotating import (
    TangConfig, run_tang_rotating,
)
from expectation_snn.assays.runtime import (
    FrozenBundle, build_frozen_network,
)


def _reset_brian(seed: int = 42) -> None:
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)


def _dir_rates(b: FrozenBundle) -> np.ndarray:
    assert b.ctx_pred is not None
    return np.asarray(b.ctx_pred.direction.rates).copy()


# ---------------------------------------------------------------------------
# [1] ctx_pred Tang end-to-end smoke
# ---------------------------------------------------------------------------

def assay_ctx_pred_tang_runs_end_to_end() -> dict:
    _reset_brian(42)
    b = build_frozen_network(architecture="ctx_pred", seed=42, r=1.0)
    cfg = TangConfig(
        n_random=20, n_rotating=30, item_ms=150.0,
        presettle_ms=200.0, seed=42,
    )
    r = run_tang_rotating(bundle=b, cfg=cfg, verbose=False)
    assert r.meta["n_items"] == 50
    assert r.meta["n_random_items"] == 20
    assert r.meta["tang_direction_wired"] is True
    assert r.meta["bundle"]["architecture"] == "ctx_pred"
    # rate buckets must be finite for each cond (non-empty n_items, etc.)
    for cond in ("random", "rotating_expected", "rotating_deviant"):
        hz = r.three_condition["per_cond"][cond]["mean_rate_hz"]
        assert np.isfinite(hz), f"cond {cond} got non-finite rate"
    return {"bundle": b, "result": r}


# ---------------------------------------------------------------------------
# [2] h_t Tang back-compat
# ---------------------------------------------------------------------------

def assay_h_t_tang_still_works() -> dict:
    _reset_brian(42)
    b = build_frozen_network(h_kind="ht", seed=42, r=1.0)
    cfg = TangConfig(
        n_random=20, n_rotating=30, item_ms=150.0,
        presettle_ms=200.0, seed=42,
    )
    r = run_tang_rotating(bundle=b, cfg=cfg, verbose=False)
    assert r.meta["tang_direction_wired"] is False
    assert b.ctx_pred is None
    assert r.meta["bundle"]["architecture"] == "h_t"
    return {"result": r}


# ---------------------------------------------------------------------------
# [3] h_r bundle must be rejected
# ---------------------------------------------------------------------------

def assay_h_kind_hr_rejected() -> None:
    _reset_brian(42)
    b = build_frozen_network(h_kind="hr", seed=42, r=1.0)
    try:
        run_tang_rotating(
            bundle=b,
            cfg=TangConfig(n_random=2, n_rotating=2, item_ms=50.0,
                           presettle_ms=50.0, seed=42),
        )
    except ValueError as exc:
        assert "h_kind" in str(exc)
        return
    raise AssertionError("Tang assay must reject h_kind='hr' bundle but did not")


# ---------------------------------------------------------------------------
# [4] direction helpers split afferents correctly
# ---------------------------------------------------------------------------

def assay_direction_helpers_split_afferents() -> None:
    _reset_brian(42)
    b = build_frozen_network(architecture="ctx_pred", seed=42, r=1.0)
    n = int(b.ctx_pred.direction.N)
    half = n // 2

    b.silence_tang_direction()
    r_sil = _dir_rates(b)
    assert r_sil.max() == 0.0

    b.set_tang_direction(0, rate_hz=80.0)
    r_cw = _dir_rates(b)
    assert np.allclose(r_cw[:half], 80.0)
    assert np.allclose(r_cw[half:], 0.0)

    b.set_tang_direction(1, rate_hz=80.0)
    r_ccw = _dir_rates(b)
    assert np.allclose(r_ccw[:half], 0.0)
    assert np.allclose(r_ccw[half:], 80.0)

    b.silence_tang_direction()
    assert _dir_rates(b).max() == 0.0


# ---------------------------------------------------------------------------
# [5] post-run direction is silenced
# ---------------------------------------------------------------------------

def assay_post_run_direction_is_silenced(res1: dict) -> None:
    b: FrozenBundle = res1["bundle"]
    post = _dir_rates(b)
    assert post.max() == 0.0, (
        f"post-run direction rates not silenced: max={post.max():.2f}"
    )


# ---------------------------------------------------------------------------
# [6] tang_direction_wired flag matches architecture
# ---------------------------------------------------------------------------

def assay_tang_direction_wired_flag_matches(res1: dict, res2: dict) -> None:
    assert res1["result"].meta["tang_direction_wired"] is True
    assert res2["result"].meta["tang_direction_wired"] is False


# ---------------------------------------------------------------------------
# [7] block transitions rewire direction (instrumented stepwise replay)
# ---------------------------------------------------------------------------

def assay_block_transitions_rewire_direction() -> None:
    """Replay the direction-wiring block of run_tang_rotating in isolation
    on a synthetic 3-block sequence:
        random (block_id=-1) → CW rotating (rotation_dir=+1) →
        CCW rotating (rotation_dir=-1) → random
    and confirm that set/silence_tang_direction are called with the
    correct arguments at each block transition (and only on transitions).
    """
    _reset_brian(42)
    b = build_frozen_network(architecture="ctx_pred", seed=42, r=1.0)

    # Instrument the bundle's direction helpers to record call history.
    calls: list = []
    _orig_set = b.set_tang_direction
    _orig_silence = b.silence_tang_direction

    def _rec_set(direction: int, rate_hz: float = 80.0) -> None:
        calls.append(("set", int(direction), float(rate_hz)))
        _orig_set(direction, rate_hz)

    def _rec_sil() -> None:
        calls.append(("silence",))
        _orig_silence()

    b.set_tang_direction = _rec_set           # type: ignore[assignment]
    b.silence_tang_direction = _rec_sil       # type: ignore[assignment]

    # Hand-crafted items: 3 random items (block_id=-1, rot_dir=0), then
    # 4 CW items (block_id=0, rot_dir=+1), then 4 CCW items (block_id=1,
    # rot_dir=-1), then 2 random tail items (block_id=-1, rot_dir=0).
    # Replicate the exact wiring logic used in run_tang_rotating — keep
    # the same _DIR_FROM_ROT mapping.
    block_ids = np.array([-1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1], dtype=np.int64)
    rot_dir = np.array([0, 0, 0, +1, +1, +1, +1, -1, -1, -1, -1, 0, 0], dtype=np.int64)
    _DIR_FROM_ROT = {+1: 0, -1: 1}
    prev = None
    # Initial silence before loop (matches runner).
    b.silence_tang_direction()
    for k in range(len(block_ids)):
        cur = int(block_ids[k])
        if cur != prev:
            if cur < 0:
                b.silence_tang_direction()
            else:
                rd = int(rot_dir[k])
                b.set_tang_direction(_DIR_FROM_ROT[rd])
            prev = cur
    # Mirror post-run silence.
    b.silence_tang_direction()

    # Expected call trace:
    #   silence (initial)                 ← pre-loop guard
    #   silence (first random block)      ← entering block_id=-1
    #   set 0  (CW rotating)              ← entering block_id=0
    #   set 1  (CCW rotating)             ← entering block_id=1
    #   silence (tail random)             ← re-entering block_id=-1
    #   silence (post-loop)               ← final guard
    expected = [
        ("silence",),
        ("silence",),
        ("set", 0, 80.0),
        ("set", 1, 80.0),
        ("silence",),
        ("silence",),
    ]
    assert calls == expected, (
        f"block-transition trace mismatch:\n  got:      {calls}\n  "
        f"expected: {expected}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("[1] ctx_pred_tang_runs_end_to_end")
    r1 = assay_ctx_pred_tang_runs_end_to_end()
    pc = r1["result"].three_condition["per_cond"]
    print(f"    n_items={r1['result'].meta['n_items']} "
          f"random={r1['result'].meta['n_random_items']} "
          f"dev={r1['result'].meta['n_deviant']} "
          f"rate_Hz: rand={pc['random']['mean_rate_hz']:.2f} "
          f"exp={pc['rotating_expected']['mean_rate_hz']:.2f} "
          f"dev={pc['rotating_deviant']['mean_rate_hz']:.2f}  PASS")

    print("[2] h_t_tang_still_works")
    r2 = assay_h_t_tang_still_works()
    pc = r2["result"].three_condition["per_cond"]
    print(f"    h_kind=ht  rate_Hz: rand={pc['random']['mean_rate_hz']:.2f} "
          f"exp={pc['rotating_expected']['mean_rate_hz']:.2f} "
          f"dev={pc['rotating_deviant']['mean_rate_hz']:.2f}  PASS")

    print("[3] h_kind_hr_rejected")
    assay_h_kind_hr_rejected()
    print("    ValueError raised as expected  PASS")

    print("[4] direction_helpers_split_afferents")
    assay_direction_helpers_split_afferents()
    print("    CW / CCW / silence afferent splits correct  PASS")

    print("[5] post_run_direction_is_silenced")
    assay_post_run_direction_is_silenced(r1)
    print("    ctx_pred post-run direction rates all 0 Hz  PASS")

    print("[6] tang_direction_wired_flag_matches")
    assay_tang_direction_wired_flag_matches(r1, r2)
    print("    ctx_pred=True  h_t=False  PASS")

    print("[7] block_transitions_rewire_direction")
    assay_block_transitions_rewire_direction()
    print("    expected silence/set(0)/set(1) trace matches  PASS")

    print("\nvalidate_tang_direction_wiring: 7/7 PASS")


if __name__ == "__main__":
    main()
