"""Functional validation for Sprint 5e-Fix D.3 — the ``ctx_pred`` branch
of :func:`expectation_snn.assays.runtime.build_frozen_network`.

Per the per-component functional-validation rule, every new load-bearing
surface ships with a ``validate_*.py`` that proves the contract.

Contract (from runtime.build_frozen_network docstring + Fix C/D spec):

  [1] ctx_pred_builds_with_default_seed
        Architecture ``"ctx_pred"`` builds without error on seed=42,
        loads the Stage-1 ctx_pred checkpoint, and exposes the two
        rings + the plastic ctx→pred transform via ``bundle.ctx_pred``.

  [2] feedback_source_is_pred_ring
        ``bundle.h_ring is bundle.ctx_pred.pred`` — the H → V1 feedback
        routes are built with pred as the presynaptic ring (predictions
        flow BACK to V1). ctx is NOT the feedback source.

  [3] v1_to_h_teacher_targets_ctx
        When ``with_v1_to_h="continuous"``, the V1 → H teacher projects
        into ``ctx_pred.ctx`` (sensory-driven ring). Pred stays free of
        a bottom-up V1 teacher at assay time (spec: pred is driven only
        by the learned ctx→pred transform during assays).

  [4] ctx_pred_weights_match_checkpoint
        The Stage-1 ctx_pred checkpoint's ``W_ctx_pred_final``,
        ``ctx_ee_w_final``, and ``pred_ee_w_final`` are byte-equal to
        the live synaptic weights after building the bundle.

  [5] with_cue_requires_h_r
        ``build_frozen_network(architecture='ctx_pred', with_cue=True)``
        raises ValueError (Stage-2 cue was trained on H_R only).

  [6] legacy_h_kind_still_works
        Legacy ``h_kind='hr'`` and ``h_kind='ht'`` paths continue to
        return single-ring bundles with ``bundle.ctx_pred is None`` and
        load the correct legacy Stage-1 checkpoints.

  [7] tang_direction_helpers_round_trip
        ``set_tang_direction(0|1, rate_hz)`` activates the requested
        CW/CCW afferent block (all rates of the other block = 0, all
        rates of the active block = rate_hz). ``silence_tang_direction``
        zeros every afferent. Calling ``set_tang_direction`` on a
        legacy (h_r / h_t) bundle raises RuntimeError.

  [8] reset_h_clears_both_rings_on_ctx_pred
        ``reset_h()`` on a ctx_pred bundle zeros V on ctx.e AND pred.e,
        and clears xpre/xpost/elig on the W_ctx_pred synapses.

  [9] short_run_drives_ctx_from_v1
        500 ms grating drive propagates V1 → ctx via the V1→H teacher:
        ctx firing rate > 0.5 Hz. (No threshold on pred; at an
        undertrained checkpoint pred is expected to be sparse.)

  [10] groups_include_both_rings_and_ctx_pred
        ``bundle.groups`` contains ctx-side and pred-side synapse /
        neuron objects (by Brian2 name) plus the plastic ctx_pred
        synapse and the direction PoissonGroup.

  [11] default_ctx_pred_feedback_topology
        The default ctx_pred runtime feedback uses center-only direct
        H_pred→V1_E and wrapped d1/d2 local-surround H_pred→V1_SOM.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from brian2 import (
    Hz, Network, SpikeMonitor, defaultclock, ms, mV, pA, prefs,
    seed as b2_seed, start_scope,
)

from expectation_snn.assays.runtime import (
    DEFAULT_CKPT_DIR, build_frozen_network, set_grating,
)
from expectation_snn.brian2_model.h_context_prediction import HContextPrediction
from expectation_snn.brian2_model.feedback_routes import (
    DIRECT_KERNEL_CENTER, SOM_KERNEL_D1_D2_SURROUND,
)


SEED = 42


def _setup() -> None:
    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)


def assay_ctx_pred_builds_with_default_seed() -> None:
    print("[1] ctx_pred_builds_with_default_seed")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED,
                             r=1.0, g_total=1.0, with_v1_to_h="continuous")
    assert b.meta["architecture"] == "ctx_pred", b.meta["architecture"]
    assert b.h_kind == "ctx_pred"
    assert isinstance(b.ctx_pred, HContextPrediction)
    assert b.ctx_pred.ctx is not None and b.ctx_pred.pred is not None
    print(f"    architecture={b.meta['architecture']}  "
          f"ctx={b.ctx_pred.ctx.name}  pred={b.ctx_pred.pred.name}  PASS")


def assay_feedback_source_is_pred_ring() -> None:
    print("[2] feedback_source_is_pred_ring")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED)
    assert b.h_ring is b.ctx_pred.pred, (
        f"feedback source should be pred, got ring name {b.h_ring.name}"
    )
    # And sanity: pred != ctx.
    assert b.ctx_pred.pred is not b.ctx_pred.ctx
    print(f"    h_ring={b.h_ring.name} (pred={b.ctx_pred.pred.name})  PASS")


def assay_v1_to_h_teacher_targets_ctx() -> None:
    print("[3] v1_to_h_teacher_targets_ctx")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED,
                             with_v1_to_h="continuous")
    # Inspect the Synapses' target group directly.
    syn = b.v1_to_h.v1_to_he
    # In Brian2, syn.target is the post-synaptic NeuronGroup.
    assert syn.target is b.ctx_pred.ctx.e, (
        f"V1→H teacher target is {syn.target.name}, "
        f"expected {b.ctx_pred.ctx.e.name}"
    )
    assert syn.target is not b.ctx_pred.pred.e
    print(f"    v1_to_h.target={syn.target.name} (ctx.e={b.ctx_pred.ctx.e.name})  PASS")


def assay_ctx_pred_weights_match_checkpoint() -> None:
    print("[4] ctx_pred_weights_match_checkpoint")
    _setup()
    ck = np.load(os.path.join(DEFAULT_CKPT_DIR,
                              f"stage_1_ctx_pred_seed{SEED}.npz"))
    b = build_frozen_network(architecture="ctx_pred", seed=SEED)
    w_ctx_pred = np.asarray(b.ctx_pred.ctx_pred.w[:], dtype=np.float64)
    w_ctx_ee = np.asarray(b.ctx_pred.ctx.ee.w[:], dtype=np.float64)
    w_pred_ee = np.asarray(b.ctx_pred.pred.ee.w[:], dtype=np.float64)
    assert np.allclose(w_ctx_pred,
                       np.asarray(ck["W_ctx_pred_final"], dtype=np.float64)), (
        "W_ctx_pred does not match checkpoint W_ctx_pred_final"
    )
    assert np.allclose(w_ctx_ee,
                       np.asarray(ck["ctx_ee_w_final"], dtype=np.float64)), (
        "ctx.ee.w does not match checkpoint ctx_ee_w_final"
    )
    assert np.allclose(w_pred_ee,
                       np.asarray(ck["pred_ee_w_final"], dtype=np.float64)), (
        "pred.ee.w does not match checkpoint pred_ee_w_final"
    )
    print(f"    W_ctx_pred n={w_ctx_pred.size}  ctx.ee n={w_ctx_ee.size}  "
          f"pred.ee n={w_pred_ee.size}  all byte-equal to ckpt  PASS")


def assay_with_cue_requires_h_r() -> None:
    print("[5] with_cue_requires_h_r")
    _setup()
    try:
        build_frozen_network(architecture="ctx_pred", seed=SEED, with_cue=True)
        raise AssertionError(
            "with_cue=True + architecture='ctx_pred' should have raised"
        )
    except ValueError as exc:
        assert "h_r" in str(exc) or "Stage-2" in str(exc), (
            f"expected ValueError about h_r / Stage-2, got: {exc}"
        )
    print("    ValueError raised as expected  PASS")


def assay_legacy_h_kind_still_works() -> None:
    print("[6] legacy_h_kind_still_works")
    _setup()
    b_hr = build_frozen_network(h_kind="hr", seed=SEED)
    assert b_hr.meta["architecture"] == "h_r"
    assert b_hr.h_kind == "hr"
    assert b_hr.ctx_pred is None
    assert b_hr.meta["n_ee_w"] > 0
    b_ht = build_frozen_network(h_kind="ht", seed=SEED)
    assert b_ht.meta["architecture"] == "h_t"
    assert b_ht.h_kind == "ht"
    assert b_ht.ctx_pred is None
    print(f"    hr: architecture={b_hr.meta['architecture']} n_ee_w={b_hr.meta['n_ee_w']}  "
          f"ht: architecture={b_ht.meta['architecture']} n_ee_w={b_ht.meta['n_ee_w']}  PASS")


def assay_tang_direction_helpers_round_trip() -> None:
    print("[7] tang_direction_helpers_round_trip")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED)
    n_dir = int(b.ctx_pred.direction.N)
    half = n_dir // 2
    # CW (direction=0)
    b.set_tang_direction(0, rate_hz=80.0)
    rates = np.asarray(b.ctx_pred.direction.rates / Hz)
    assert np.allclose(rates[:half], 80.0) and np.allclose(rates[half:], 0.0), (
        f"CW rates mis-set: first half mean={rates[:half].mean()}, "
        f"second half mean={rates[half:].mean()}"
    )
    # CCW (direction=1)
    b.set_tang_direction(1, rate_hz=40.0)
    rates = np.asarray(b.ctx_pred.direction.rates / Hz)
    assert np.allclose(rates[:half], 0.0) and np.allclose(rates[half:], 40.0), (
        f"CCW rates mis-set: first half mean={rates[:half].mean()}, "
        f"second half mean={rates[half:].mean()}"
    )
    # Silence
    b.silence_tang_direction()
    rates = np.asarray(b.ctx_pred.direction.rates / Hz)
    assert np.allclose(rates, 0.0), f"silence failed: max rate={rates.max()}"

    # Legacy bundle: set_tang_direction must raise; silence must no-op.
    b2 = build_frozen_network(h_kind="hr", seed=SEED)
    try:
        b2.set_tang_direction(0, rate_hz=10.0)
        raise AssertionError("set_tang_direction on h_r bundle should raise")
    except RuntimeError as exc:
        assert "ctx_pred" in str(exc).lower(), (
            f"expected error to mention ctx_pred, got: {exc}"
        )
    b2.silence_tang_direction()  # must not raise
    print("    CW / CCW / silence work; legacy rejects set; legacy silence no-ops  PASS")


def assay_reset_h_clears_both_rings_on_ctx_pred() -> None:
    print("[8] reset_h_clears_both_rings_on_ctx_pred")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED)
    # Perturb ctx and pred state deliberately.
    b.ctx_pred.ctx.e.V = -55.0 * mV
    b.ctx_pred.pred.e.V = -50.0 * mV
    b.ctx_pred.ctx.e.I_e = 3.0 * pA
    b.ctx_pred.pred.e.I_i = 2.0 * pA
    b.ctx_pred.ctx_pred.elig[:] = 0.5
    b.ctx_pred.ctx_pred.xpre[:] = 0.3
    b.ctx_pred.ctx_pred.xpost[:] = 0.4
    # Reset.
    b.reset_h()
    v_ctx = np.asarray(b.ctx_pred.ctx.e.V / mV)
    v_pred = np.asarray(b.ctx_pred.pred.e.V / mV)
    i_e_ctx = np.asarray(b.ctx_pred.ctx.e.I_e / pA)
    i_i_pred = np.asarray(b.ctx_pred.pred.e.I_i / pA)
    elig = np.asarray(b.ctx_pred.ctx_pred.elig[:])
    xpre = np.asarray(b.ctx_pred.ctx_pred.xpre[:])
    xpost = np.asarray(b.ctx_pred.ctx_pred.xpost[:])
    assert np.allclose(v_ctx, -70.0), f"ctx.e.V not reset: mean={v_ctx.mean()}"
    assert np.allclose(v_pred, -70.0), f"pred.e.V not reset: mean={v_pred.mean()}"
    assert np.allclose(i_e_ctx, 0.0) and np.allclose(i_i_pred, 0.0), (
        "ctx/pred currents not reset"
    )
    assert np.allclose(elig, 0.0) and np.allclose(xpre, 0.0) and np.allclose(xpost, 0.0), (
        "ctx_pred traces not reset"
    )
    print("    ctx.e.V / pred.e.V / currents / xpre / xpost / elig all cleared  PASS")


def assay_short_run_drives_ctx_from_v1() -> None:
    print("[9] short_run_drives_ctx_from_v1")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED,
                             r=1.0, g_total=1.0, with_v1_to_h="continuous")
    ctx_mon = SpikeMonitor(b.ctx_pred.ctx.e, name="dbg_vd9_ctx_mon")
    v1_mon = SpikeMonitor(b.v1_ring.e, name="dbg_vd9_v1_mon")
    net = Network(*b.groups, ctx_mon, v1_mon)
    b.reset_h()
    b.silence_tang_direction()
    set_grating(b.v1_ring, theta_rad=0.0, contrast=1.0)
    net.run(500 * ms)
    set_grating(b.v1_ring, theta_rad=None)
    n_ctx = int(b.ctx_pred.ctx.e.N)
    n_v1 = int(b.v1_ring.e.N)
    ctx_hz = float(len(ctx_mon.i)) / n_ctx / 0.5
    v1_hz = float(len(v1_mon.i)) / n_v1 / 0.5
    assert v1_hz > 1.0, f"V1 silent under grating, {v1_hz:.2f} Hz"
    assert ctx_hz > 0.5, (
        f"ctx silent under grating (V1→ctx teacher broken?), {ctx_hz:.2f} Hz"
    )
    print(f"    V1={v1_hz:.2f}Hz  ctx={ctx_hz:.2f}Hz (>0.5 required)  PASS")


def assay_groups_include_both_rings_and_ctx_pred() -> None:
    print("[10] groups_include_both_rings_and_ctx_pred")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED)
    names = {getattr(g, "name", str(type(g))) for g in b.groups}
    # Must contain ctx and pred ring objects.
    assert any(n.startswith("s5a_ctx_seed42") for n in names), (
        f"no ctx_seed42 objects in groups: {names}"
    )
    assert any(n.startswith("s5a_pred_seed42") for n in names), (
        f"no pred_seed42 objects in groups: {names}"
    )
    # Must contain the plastic ctx→pred synapse.
    assert b.ctx_pred.ctx_pred in b.groups, "ctx_pred plastic synapse not in groups"
    # Must contain the direction PoissonGroup + dir_to_ctx synapse.
    assert b.ctx_pred.direction in b.groups, "direction group not in groups"
    assert b.ctx_pred.dir_to_ctx in b.groups, "dir_to_ctx synapse not in groups"
    print(f"    {len(b.groups)} groups incl. ctx+pred rings, ctx_pred syn, "
          f"direction, dir_to_ctx  PASS")


def assay_default_ctx_pred_feedback_topology() -> None:
    print("[11] default_ctx_pred_feedback_topology")
    _setup()
    b = build_frozen_network(architecture="ctx_pred", seed=SEED)
    assert b.fb.config.direct_kernel_mode == DIRECT_KERNEL_CENTER, (
        b.fb.config.direct_kernel_mode
    )
    assert b.fb.config.som_kernel_mode == SOM_KERNEL_D1_D2_SURROUND, (
        b.fb.config.som_kernel_mode
    )
    expected_direct = np.eye(12, dtype=np.float64)
    expected_som = np.zeros((12, 12), dtype=np.float64)
    for ci in range(12):
        expected_som[ci, (ci - 1) % 12] = 0.4
        expected_som[ci, (ci + 1) % 12] = 0.4
        expected_som[ci, (ci - 2) % 12] = 0.1
        expected_som[ci, (ci + 2) % 12] = 0.1
    assert np.allclose(b.fb.kernel_direct, expected_direct), b.fb.kernel_direct
    assert np.allclose(b.fb.kernel_som, expected_som), b.fb.kernel_som
    assert b.meta["fb_direct_kernel_mode"] == DIRECT_KERNEL_CENTER
    assert b.meta["fb_som_kernel_mode"] == SOM_KERNEL_D1_D2_SURROUND
    print(
        "    direct=center-only SOM=d1/d2 surround "
        f"n_direct={b.meta['n_fb_direct']} n_som={b.meta['n_fb_som']} PASS"
    )


def main() -> int:
    assays = [
        assay_ctx_pred_builds_with_default_seed,
        assay_feedback_source_is_pred_ring,
        assay_v1_to_h_teacher_targets_ctx,
        assay_ctx_pred_weights_match_checkpoint,
        assay_with_cue_requires_h_r,
        assay_legacy_h_kind_still_works,
        assay_tang_direction_helpers_round_trip,
        assay_reset_h_clears_both_rings_on_ctx_pred,
        assay_short_run_drives_ctx_from_v1,
        assay_groups_include_both_rings_and_ctx_pred,
        assay_default_ctx_pred_feedback_topology,
    ]
    failed = []
    for a in assays:
        try:
            a()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{a.__name__}: {exc}")
            print(f"    FAIL — {exc}")
    n = len(assays)
    n_ok = n - len(failed)
    print()
    print(f"validate_runtime_ctx_pred: {n_ok}/{n} PASS")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
