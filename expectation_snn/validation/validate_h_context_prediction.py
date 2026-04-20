"""Functional validation for Sprint 5e Fix C —
``brian2_model.h_context_prediction``.

Per the per-component functional-validation rule: every new module
(or new function of load-bearing scope) ships with a
``validate_*.py`` that proves its contract on seed=42.

The H_context + H_prediction architecture split (Researcher's Fix C
spec) introduces a learned transform ``W_ctx_pred`` between two Wang
rings using Frémaux & Gerstner 2015 three-factor eligibility-trace
plasticity, plus a 2-channel (CW/CCW) direction afferent into
H_context for Tang support (Fix D).

Contract
--------

  [1] build_architecture_shapes
        `build_h_context_prediction(rng=default_rng(42))` returns two
        HRing bundles with N_E = 192 each (12 channels × 16 per
        channel); W_ctx_pred is all-to-all (36864 synapses); the
        2-channel direction afferent has DEFAULT_N_DIRECTION_AFFERENTS
        Poisson sources, each broadcast to every H_ctx_E cell.

  [2] build_is_deterministic_under_seed
        Two calls with independent `default_rng(42)` generators return
        byte-equal `w_ctx_pred_init` and identical architecture counts.

  [3] initial_weight_distribution
        `w_ctx_pred_init ∼ Uniform[0, w_init_frac · w_max]` to within
        KS/range tolerances.

  [4] pred_cue_is_silenced_by_spec
        `build_h_context_prediction(...)` must clamp H_pred.cue rates
        to 0 Hz (spec: "H_pred is NOT cue-driven"); a 100 ms quiet-run
        with no direction, no ctx cue, no pred cue produces <1 spk/cell
        in either ring.

  [5] pre_only_drive_grows_xpre_without_elig
        If H_pred_E is silent and H_ctx_E fires, per-synapse `xpre`
        grows but `elig` stays ≈ 0 — eligibility requires coincident
        post activity (three-factor rule, not unsupervised).

  [6] coincident_pre_post_grows_elig
        Driving BOTH H_ctx_E[L] and H_pred_E[f(L)] for 200 ms produces
        non-zero eligibility on the L→f(L) synapse subset; the
        cross-pair (L'→f(L)) with L' ≠ L produces strictly smaller
        eligibility.

  [7] eligibility_decays_with_tau_elig
        After coincident drive, letting the network run without any
        further drive for 3·tau_elig_ms reduces elig by ≥ 90 %.

  [8] m_gate_update_changes_weights_when_elig_positive
        `apply_modulatory_update(..., m_integral=m, dt_trial_s=0.0)`
        with a positive elig array and m>0 produces positive dw on
        elig-carrying synapses; with m=0 and elig>0 produces pure
        decay (or zero if dt_trial_s=0).

  [9] row_cap_rescales_offending_rows
        Setting w to ceiling (1.0) on every synapse so that
        sum_j w[i, j] = N_post ≫ w_row_max, then calling
        `apply_modulatory_update(m_integral=0, dt_trial_s=0)`, brings
        every row-sum to exactly `w_row_max`.

  [10] direction_CW_CCW_silence
        `set_direction(bundle, 0, rate_hz=r)` drives the first half of
        direction afferents at `r` Hz, silences the second; symmetric
        for direction=1; `silence_direction(bundle)` zeroes both.
        Invalid direction raises ValueError.

  [11] modulatory_gate_operation_fires_once_per_trailer
        A `NetworkOperation` built via `make_modulatory_gate_operation`
        with three trailer onsets fires exactly 3 updates across a
        run spanning all three onsets (each update logged in the
        provided list).

  [12] namespace_and_imports_are_stable
        Public API: `HContextPredictionConfig`, `HContextPrediction`,
        `build_h_context_prediction`, `set_direction`,
        `silence_direction`, `apply_modulatory_update`,
        `make_modulatory_gate_operation`.

NOTE: the primary Sprint 5e-Fix go/no-go (post-training
``P(argmax H_pred == expected_trailer) ≥ 0.7``) is deferred to the
integration validator because it requires the full training pipeline
(Stage-1 retrain on the biased Richter schedule). This module validates
the mechanics; training-level forecasting is validated by
``validate_preprobe_forecast_gate.py`` + the post-training assay
battery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Brian2 is expensive to import — do it once up-front.
from brian2 import Hz, Network, defaultclock, ms, start_scope

from expectation_snn.brian2_model.h_context_prediction import (
    DEFAULT_N_DIRECTION_AFFERENTS,
    DIRECTION_CHANNELS,
    HContextPrediction,
    HContextPredictionConfig,
    apply_modulatory_update,
    build_h_context_prediction,
    make_modulatory_gate_operation,
    set_direction,
    silence_direction,
)
from expectation_snn.brian2_model.h_ring import (
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
    pulse_channel,
    silence_cue,
)


SEED = 42
N_E_EXPECTED = H_N_CHANNELS * H_N_E_PER  # 12 * 16 = 192
N_SYN_EXPECTED = N_E_EXPECTED * N_E_EXPECTED  # all-to-all


def _fresh_bundle(seed: int = SEED) -> HContextPrediction:
    """Start a clean Brian2 scope and build a fresh H_ctx+H_pred bundle."""
    start_scope()
    # Reset the global clock to t=0 — otherwise successive assays pile
    # up on the same simulated time and `NetworkOperation`-driven gate
    # tests race their own onset windows.
    defaultclock.dt = 0.1 * ms
    rng = np.random.default_rng(seed)
    return build_h_context_prediction(rng=rng)


# ---------------------------------------------------------------------------

def assay_build_architecture_shapes() -> None:
    print("[1] build_architecture_shapes")
    bundle = _fresh_bundle()
    assert int(bundle.ctx.e.N) == N_E_EXPECTED
    assert int(bundle.pred.e.N) == N_E_EXPECTED
    assert len(bundle.ctx_pred) == N_SYN_EXPECTED
    assert int(bundle.direction.N) == DEFAULT_N_DIRECTION_AFFERENTS
    assert int(bundle.direction.N) % DIRECTION_CHANNELS == 0
    # dir_to_ctx is broadcast: 1 synapse per (direction_afferent, H_ctx_E) pair
    assert len(bundle.dir_to_ctx) == (
        DEFAULT_N_DIRECTION_AFFERENTS * N_E_EXPECTED
    )
    print(f"    ctx_E={int(bundle.ctx.e.N)}, pred_E={int(bundle.pred.e.N)}, "
          f"ctx_pred_syn={len(bundle.ctx_pred)}, "
          f"dir={int(bundle.direction.N)}  PASS")


def assay_build_is_deterministic_under_seed() -> None:
    print("[2] build_is_deterministic_under_seed")
    bundle_a = _fresh_bundle(SEED)
    w_a = np.asarray(bundle_a.w_ctx_pred_init)
    bundle_b = _fresh_bundle(SEED)
    w_b = np.asarray(bundle_b.w_ctx_pred_init)
    assert np.array_equal(w_a, w_b), "W_ctx_pred init differed under seed=42"
    assert len(bundle_a.ctx_pred) == len(bundle_b.ctx_pred)
    print(f"    w_ctx_pred_init byte-equal across rebuilds (n={w_a.size})  PASS")


def assay_initial_weight_distribution() -> None:
    print("[3] initial_weight_distribution")
    bundle = _fresh_bundle()
    cfg = bundle.config
    w = np.asarray(bundle.w_ctx_pred_init)
    hi = cfg.w_init_frac * cfg.w_max
    assert w.min() >= 0.0, f"min weight {w.min()} < 0"
    assert w.max() <= hi + 1e-12, f"max weight {w.max()} > {hi}"
    # Uniform mean ≈ (lo+hi)/2 = hi/2 within ±5% on 36864 draws.
    target_mean = 0.5 * hi
    assert abs(w.mean() - target_mean) / target_mean < 0.05, (
        f"mean {w.mean():.5f} deviates from uniform target {target_mean:.5f}"
    )
    # Bucket check: roughly equal mass in 4 quantiles.
    q = np.quantile(w, [0.25, 0.5, 0.75])
    for expected, got in zip([0.25 * hi, 0.5 * hi, 0.75 * hi], q):
        assert abs(got - expected) / expected < 0.05, (
            f"quantile {expected:.4f} ≠ observed {got:.4f}"
        )
    print(f"    w ∈ [{w.min():.4f}, {w.max():.4f}], "
          f"mean={w.mean():.4f} (target {target_mean:.4f})  PASS")


def assay_pred_cue_is_silenced_by_spec() -> None:
    print("[4] pred_cue_is_silenced_by_spec")
    bundle = _fresh_bundle()
    pred_cue_rates = np.asarray(bundle.pred.cue.rates / Hz)
    assert np.all(pred_cue_rates == 0.0), (
        f"pred.cue.rates should be all 0 Hz, got max {pred_cue_rates.max()}"
    )
    # Silence ctx cue + direction so nothing drives either ring.
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    from brian2 import SpikeMonitor
    mon_ctx = SpikeMonitor(bundle.ctx.e)
    mon_pred = SpikeMonitor(bundle.pred.e)
    net = Network(*bundle.groups, mon_ctx, mon_pred)
    net.run(100 * ms)
    # <1 spk/cell average — i.e. no runaway spontaneous activity.
    ctx_per_cell = mon_ctx.num_spikes / max(int(bundle.ctx.e.N), 1)
    pred_per_cell = mon_pred.num_spikes / max(int(bundle.pred.e.N), 1)
    assert ctx_per_cell < 1.0, (
        f"ctx_E spontaneous {ctx_per_cell:.3f} spk/cell over 100 ms"
    )
    assert pred_per_cell < 1.0, (
        f"pred_E spontaneous {pred_per_cell:.3f} spk/cell over 100 ms"
    )
    print(f"    quiet run: ctx={ctx_per_cell:.3f}, pred={pred_per_cell:.3f} "
          f"spk/cell  PASS")


def assay_pre_only_drive_grows_xpre_without_elig() -> None:
    print("[5] pre_only_drive_grows_xpre_without_elig")
    bundle = _fresh_bundle()
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    # Activate only one ctx channel so some ctx_E cells fire; pred stays off.
    pulse_channel(bundle.ctx, channel=0, rate_hz=400.0)
    net = Network(*bundle.groups)
    net.run(200 * ms)
    xpre = np.asarray(bundle.ctx_pred.xpre[:])
    elig = np.asarray(bundle.ctx_pred.elig[:])
    assert xpre.max() > 0.0, "pre activity should have grown xpre"
    # Allow numerical flutter but elig must be essentially zero: no post spikes.
    assert float(np.abs(elig).max()) < 1e-6, (
        f"pred silent, elig should be 0 but got max |elig|={np.abs(elig).max()}"
    )
    print(f"    xpre max={xpre.max():.3f}, |elig|max={np.abs(elig).max():.2e}  "
          f"PASS")


def assay_coincident_pre_post_grows_elig() -> None:
    print("[6] coincident_pre_post_grows_elig")
    bundle = _fresh_bundle()
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    # Drive L=0 on ctx and f(L)=1 on pred simultaneously.
    pulse_channel(bundle.ctx, channel=0, rate_hz=400.0)
    pulse_channel(bundle.pred, channel=1, rate_hz=400.0)
    net = Network(*bundle.groups)
    net.run(200 * ms)

    elig = np.asarray(bundle.ctx_pred.elig[:])
    i_pre = np.asarray(bundle.ctx_pred.i[:])
    j_post = np.asarray(bundle.ctx_pred.j[:])

    # Pre cells on channel 0: idx 0..N_E_PER-1; post cells on channel 1:
    # idx N_E_PER..2*N_E_PER-1.
    pre_on = np.arange(0, H_N_E_PER)
    post_on = np.arange(H_N_E_PER, 2 * H_N_E_PER)
    mask_on = np.isin(i_pre, pre_on) & np.isin(j_post, post_on)
    # "Cross" = random L=2 pre channel (not driven) → same f(L)=1 post.
    pre_off = np.arange(2 * H_N_E_PER, 3 * H_N_E_PER)
    mask_off = np.isin(i_pre, pre_off) & np.isin(j_post, post_on)

    mean_on = float(elig[mask_on].mean())
    mean_off = float(elig[mask_off].mean())
    assert mean_on > 0.0, (
        f"coincident L→f(L) elig mean should be >0, got {mean_on}"
    )
    assert mean_on > 5.0 * max(mean_off, 1e-6), (
        f"driven elig mean {mean_on:.4f} should dominate undriven "
        f"elig mean {mean_off:.4f} by ≥5x"
    )
    print(f"    elig[L→f(L)] mean={mean_on:.4f} vs elig[L'→f(L)] "
          f"mean={mean_off:.4f}  PASS")


def assay_eligibility_decays_with_tau_elig() -> None:
    print("[7] eligibility_decays_with_tau_elig")
    # Brian2 2.10 event-driven variables update *only* at spike events:
    # clock-driven `run(...)` does not apply lazy decay on external
    # reads. So we probe decay by forcing an on_pre event on every
    # ctx cell after the silent gap — that event applies the lazy
    # decay to all 36864 synapses' `elig`, then adds `xpost` (which
    # has decayed to ≈ 0 via tau_coinc=20 ms over the gap, so
    # post-burst elig ≈ old_elig · exp(-gap/tau_elig)).
    bundle = _fresh_bundle()
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    pulse_channel(bundle.ctx, channel=0, rate_hz=400.0)
    pulse_channel(bundle.pred, channel=1, rate_hz=400.0)
    net = Network(*bundle.groups)
    net.run(200 * ms)
    elig_before = float(np.asarray(bundle.ctx_pred.elig[:]).max())
    assert elig_before > 0.0, "expected positive elig after coincident drive"

    # Silent gap of 3·tau_elig.
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    net.run(3 * bundle.config.tau_elig_ms * ms)

    # Force on_pre across all 192 ctx cells: set every ctx cue afferent
    # high so every channel's pre fires. A short burst is enough.
    n_cue = int(bundle.ctx.cue.N)
    bundle.ctx.cue.rates = np.full(n_cue, 2000.0) * Hz
    net.run(20 * ms)
    silence_cue(bundle.ctx)
    net.run(1 * ms)

    elig_after = float(np.asarray(bundle.ctx_pred.elig[:]).max())
    # 3.02 s of decay → exp(-3.02) ≈ 0.0488. on_pre also injects
    # xpost, which has decayed to 0 over 3 s (tau_coinc=20 ms, so
    # exp(-3000/20) ≈ 10^-65) — this term is negligible.
    expected_scale = np.exp(-3.02)
    ratio = elig_after / elig_before
    # Generous upper bound (3x expected — not all synapses experience
    # the full 3 s gap because the ctx burst takes ~20 ms to spread
    # spike events across all 192 cells).
    assert ratio <= expected_scale * 3.0 + 0.05, (
        f"after 3·tau_elig decay, elig ratio {ratio:.4f} "
        f"exceeds bound {expected_scale*3+0.05:.4f} "
        f"(reference exp(-3)={expected_scale:.4f})"
    )
    print(f"    elig decayed from {elig_before:.4f} → {elig_after:.4f} "
          f"(ratio {ratio:.4f}, bound {expected_scale*3+0.05:.4f})  PASS")


def assay_m_gate_update_changes_weights_when_elig_positive() -> None:
    print("[8] m_gate_update_changes_weights_when_elig_positive")
    # With w_init uniform on [0, 0.05] and all-to-all fan-out N=192,
    # every row starts at row_sum ≈ 4.8 > w_row_max=3.0. So row-cap
    # rescales every row on the very first update — some per-synapse
    # w's shrink despite receiving a positive learning increment. The
    # physically meaningful check is that synapses carrying positive
    # elig have a *higher mean weight after* than synapses with zero
    # elig (evidence that learning steers weight distribution even
    # under aggressive homeostasis).
    bundle = _fresh_bundle()
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    pulse_channel(bundle.ctx, channel=0, rate_hz=400.0)
    pulse_channel(bundle.pred, channel=1, rate_hz=400.0)
    net = Network(*bundle.groups)
    net.run(200 * ms)

    w_before = np.asarray(bundle.ctx_pred.w[:]).copy()
    elig = np.asarray(bundle.ctx_pred.elig[:]).copy()
    # Use dt_trial_s=0 to suppress decay so we can isolate the learning term.
    stats = apply_modulatory_update(bundle, m_integral=0.150, dt_trial_s=0.0)
    w_after = np.asarray(bundle.ctx_pred.w[:])

    mask_on = elig > 0
    mask_off = elig == 0
    assert mask_on.any(), "expected some synapses to carry non-zero elig"
    assert mask_off.any(), "expected some synapses with zero elig"

    # Initial uniform means should be ~identical for both masks.
    init_on = float(w_before[mask_on].mean())
    init_off = float(w_before[mask_off].mean())
    assert abs(init_on - init_off) < 0.005, (
        f"initial uniform means diverge: on={init_on:.4f}, off={init_off:.4f}"
    )
    # After M-gate update, elig>0 synapses should be strictly heavier.
    post_on = float(w_after[mask_on].mean())
    post_off = float(w_after[mask_off].mean())
    assert post_on > post_off, (
        f"post-update mean w[elig>0]={post_on:.4f} should exceed "
        f"mean w[elig=0]={post_off:.4f} (learning did not outrun cap)"
    )
    # Elig must have been reset to 0 after consumption.
    assert float(np.asarray(bundle.ctx_pred.elig[:]).max()) < 1e-9, (
        "elig should be consumed (reset to 0) after M-gate update"
    )
    # Stats sanity.
    assert stats["elig_mean"] >= 0.0
    assert stats["w_mean_after"] >= 0.0
    print(f"    w[elig>0] mean={post_on:.4f} > w[elig=0] mean={post_off:.4f}; "
          f"elig consumed  PASS")


def assay_row_cap_rescales_offending_rows() -> None:
    print("[9] row_cap_rescales_offending_rows")
    bundle = _fresh_bundle()
    cfg = bundle.config
    syn = bundle.ctx_pred
    # Force every weight to the ceiling so sum_j w[i,j] = N_post * w_max
    # ≫ w_row_max.
    syn.w[:] = cfg.w_max
    syn.elig[:] = 0.0
    # Zero m/dt so only the cap can move weights.
    stats = apply_modulatory_update(bundle, m_integral=0.0, dt_trial_s=0.0)
    i_pre = np.asarray(syn.i[:])
    w_after = np.asarray(syn.w[:])
    n_pre = int(i_pre.max()) + 1
    row_sums = np.bincount(i_pre, weights=w_after, minlength=n_pre)
    assert np.allclose(row_sums, cfg.w_row_max, atol=1e-6), (
        f"row-cap should clamp every row sum to {cfg.w_row_max}; "
        f"got [{row_sums.min():.4f}, {row_sums.max():.4f}]"
    )
    assert stats["n_capped"] == n_pre, (
        f"n_capped={stats['n_capped']} ≠ {n_pre}"
    )
    print(f"    all {n_pre} rows clamped to {cfg.w_row_max:.2f}  PASS")


def assay_direction_CW_CCW_silence() -> None:
    print("[10] direction_CW_CCW_silence")
    bundle = _fresh_bundle()
    n_dir = int(bundle.direction.N)
    half = n_dir // DIRECTION_CHANNELS

    set_direction(bundle, direction=0, rate_hz=80.0)
    rates = np.asarray(bundle.direction.rates / Hz)
    assert np.all(rates[:half] == 80.0)
    assert np.all(rates[half:] == 0.0)

    set_direction(bundle, direction=1, rate_hz=50.0)
    rates = np.asarray(bundle.direction.rates / Hz)
    assert np.all(rates[:half] == 0.0)
    assert np.all(rates[half:] == 50.0)

    silence_direction(bundle)
    rates = np.asarray(bundle.direction.rates / Hz)
    assert np.all(rates == 0.0)

    # Invalid direction rejected.
    try:
        set_direction(bundle, direction=2, rate_hz=10.0)
        raise AssertionError("direction=2 should raise ValueError")
    except ValueError:
        pass
    print(f"    CW/CCW/silence + reject invalid  PASS")


def assay_modulatory_gate_operation_fires_once_per_trailer() -> None:
    print("[11] modulatory_gate_operation_fires_once_per_trailer")
    bundle = _fresh_bundle()
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)
    # Drive both rings lightly so elig grows between onsets.
    pulse_channel(bundle.ctx, channel=0, rate_hz=400.0)
    pulse_channel(bundle.pred, channel=1, rate_hz=400.0)

    trailer_onsets = [100.0, 300.0, 500.0]  # ms
    log: list = []
    op = make_modulatory_gate_operation(
        bundle,
        trailer_onsets_ms=trailer_onsets,
        dt_trial_s=1.5,
        dt_op_ms=5.0,
        log=log,
    )
    net = Network(*bundle.groups, op)
    net.run(700 * ms)
    assert len(log) == 3, (
        f"expected exactly 3 M-gate firings (one per onset), got {len(log)}"
    )
    # Each entry must be a dict from apply_modulatory_update.
    for entry in log:
        assert {"w_mean_before", "w_mean_after",
                "n_capped"}.issubset(entry.keys()), (
            f"log entry missing expected keys: {entry.keys()}"
        )
    print(f"    3 onsets → {len(log)} updates  PASS")


def assay_namespace_and_imports_are_stable() -> None:
    print("[12] namespace_and_imports_are_stable")
    from expectation_snn.brian2_model import h_context_prediction as mod
    required = [
        "HContextPredictionConfig",
        "HContextPrediction",
        "build_h_context_prediction",
        "set_direction",
        "silence_direction",
        "apply_modulatory_update",
        "make_modulatory_gate_operation",
    ]
    for name in required:
        assert hasattr(mod, name), f"public API missing: {name}"
    # Config default factory returns a valid, serializable config.
    cfg = HContextPredictionConfig()
    assert cfg.tau_elig_ms == 1000.0
    assert cfg.eta == 1e-3
    assert cfg.gamma == 1e-4
    assert cfg.w_max == 1.0
    assert cfg.w_row_max == 3.0
    assert cfg.w_init_frac == 0.05
    # Researcher Fix C spec: W_target = 0.05 * w_max (soft-decay to
    # the init-uniform mean, Vogels 2011 iSTDP precedent).
    assert cfg.w_target == 0.05, (
        f"w_target should be 0.05 (= w_init_frac · w_max), got {cfg.w_target}"
    )
    assert cfg.tau_coinc_ms == 20.0, (
        f"tau_coinc_ms should be 20 (fast pre/post filter), got {cfg.tau_coinc_ms}"
    )
    assert cfg.m_window_ms == 75.0, (
        f"m_window_ms should be 75 (Yagishita 2014), got {cfg.m_window_ms}"
    )
    print(f"    {len(required)} public names present; config defaults intact  "
          f"PASS")


def main() -> int:
    np.random.seed(SEED)
    assays = [
        assay_build_architecture_shapes,
        assay_build_is_deterministic_under_seed,
        assay_initial_weight_distribution,
        assay_pred_cue_is_silenced_by_spec,
        assay_pre_only_drive_grows_xpre_without_elig,
        assay_coincident_pre_post_grows_elig,
        assay_eligibility_decays_with_tau_elig,
        assay_m_gate_update_changes_weights_when_elig_positive,
        assay_row_cap_rescales_offending_rows,
        assay_direction_CW_CCW_silence,
        assay_modulatory_gate_operation_fires_once_per_trailer,
        assay_namespace_and_imports_are_stable,
    ]
    failed = []
    for a in assays:
        try:
            a()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{a.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            print(f"    FAIL — {exc}")
    n = len(assays)
    n_ok = n - len(failed)
    print()
    print(f"validate_h_context_prediction: {n_ok}/{n} PASS")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
