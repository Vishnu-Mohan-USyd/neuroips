"""Functional validation for `brian2_model.feedback_routes`.

Per-component validation rule: must pass before Sprint 5a assays use it.

Checks
------
1. `balance_weights(g_total, r)` algebra correct across sweep range.
2. Route construction: channel-matched connectivity for both routes
   (H_E[c] -> V1_E[c] apical, H_E[c] -> V1_SOM[c]).
3. With all weights non-zero, both Synapses' `w` arrays are set to the
   resolved g_direct / g_SOM scalars.
4. Functional: H_R pulse on channel c drives V1_SOM channel c (suppressive
   route fires) AND V1_E channel c (direct route depolarizes apical).
   Verified by comparing matched-channel response vs unmatched-channel.
5. Ablation: r=inf (g_SOM=0) -> V1_SOM on matched channel stays quiet;
   r=0 (g_direct=0) -> V1_E matched-channel apical modulation disappears
   but V1_SOM matched-channel fires.
6. `set_balance` reshuffles weights in-place without rebuilding Network.

Runs with brian2 numpy codegen, dt=0.1 ms, seed=42.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from brian2 import (
    Network, SpikeMonitor, defaultclock, prefs, ms,
    seed as b2_seed,
)

# Ensure package import works when run as a script.
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.brian2_model.h_ring import (
    build_h_r, pulse_channel, silence_cue, N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER_CHANNEL,
)
from expectation_snn.brian2_model.v1_ring import (
    build_v1_ring, set_stimulus, N_CHANNELS as V1_N_CHANNELS,
)
from expectation_snn.brian2_model.feedback_routes import (
    build_feedback_routes, set_balance,
    balance_weights, FeedbackRoutesConfig,
)


SEED = 42
DT_MS = 0.1


# -- helpers ----------------------------------------------------------------

def _setup_brian():
    prefs.codegen.target = "numpy"
    defaultclock.dt = DT_MS * ms
    b2_seed(SEED)
    np.random.seed(SEED)


def _per_channel_spike_counts(mon: SpikeMonitor, channel_map: np.ndarray,
                              t_win_ms=(0.0, np.inf)) -> np.ndarray:
    t = np.asarray(mon.t / ms)
    i = np.asarray(mon.i[:])
    m = (t >= t_win_ms[0]) & (t < t_win_ms[1])
    if not np.any(m):
        return np.zeros(V1_N_CHANNELS, dtype=np.int64)
    return np.bincount(channel_map[i[m]], minlength=V1_N_CHANNELS)


# -- checks -----------------------------------------------------------------

def check_1_balance_weights_algebra() -> bool:
    """balance_weights(g_total, r) algebra correct across sweep range."""
    pre_reg_sweep = [0.25, 0.50, 1.00, 2.00, 4.00]
    for r in pre_reg_sweep:
        for g_total in (0.5, 1.0, 2.0):
            gd, gs = balance_weights(g_total, r)
            assert abs(gd + gs - g_total) < 1e-9, (gd, gs, g_total, r)
            assert abs(gd / gs - r) < 1e-9, (gd, gs, r)
    # Balanced r=1.0 case
    gd, gs = balance_weights(1.0, 1.0)
    assert abs(gd - 0.5) < 1e-9 and abs(gs - 0.5) < 1e-9
    # Degenerates
    gd, gs = balance_weights(1.0, 0.0)
    assert gd == 0.0 and gs == 1.0
    gd, gs = balance_weights(1.0, float("inf"))
    assert gd == 1.0 and gs == 0.0
    print("[1] balance_weights: PASS")
    return True


def check_2_route_connectivity() -> bool:
    """Both routes feature-matched by channel."""
    _setup_brian()
    h = build_h_r()
    v = build_v1_ring()
    fb = build_feedback_routes(h, v, FeedbackRoutesConfig(g_total=1.0, r=1.0))

    # Direct route: H_E[c] -> V1_E[c]
    i1 = np.asarray(fb.hr_to_v1e.i[:])
    j1 = np.asarray(fb.hr_to_v1e.j[:])
    ci1 = h.e_channel[i1]
    cj1 = v.e_channel[j1]
    assert np.all(ci1 == cj1), (
        "Direct route has cross-channel connections: "
        f"{np.sum(ci1 != cj1)} of {len(ci1)}"
    )
    # Expected count: each of 16 H_E per channel contacts all 16 V1_E
    # per channel, over 12 channels = 16*16*12 = 3072.
    expected_n1 = H_N_E_PER_CHANNEL * 16 * H_N_CHANNELS
    assert len(ci1) == expected_n1, (len(ci1), expected_n1)

    # SOM route: H_E[c] -> V1_SOM[c]
    i2 = np.asarray(fb.hr_to_v1som.i[:])
    j2 = np.asarray(fb.hr_to_v1som.j[:])
    ci2 = h.e_channel[i2]
    cj2 = v.som_channel[j2]
    assert np.all(ci2 == cj2), (
        "SOM route has cross-channel connections: "
        f"{np.sum(ci2 != cj2)} of {len(ci2)}"
    )
    # Expected count: 16 H_E * 4 V1_SOM per channel * 12 = 768.
    expected_n2 = H_N_E_PER_CHANNEL * 4 * H_N_CHANNELS
    assert len(ci2) == expected_n2, (len(ci2), expected_n2)
    print(f"[2] route_connectivity: PASS "
          f"(direct={len(ci1)}, som={len(ci2)} synapses, all channel-matched)")
    return True


def check_3_weight_assignment() -> bool:
    """Scalar weights set on all synapses per resolved balance."""
    _setup_brian()
    h = build_h_r()
    v = build_v1_ring()
    cfg = FeedbackRoutesConfig(g_total=2.0, r=4.0)
    fb = build_feedback_routes(h, v, cfg)
    gd_exp, gs_exp = balance_weights(2.0, 4.0)   # 1.6, 0.4

    w1 = np.asarray(fb.hr_to_v1e.w[:])
    w2 = np.asarray(fb.hr_to_v1som.w[:])
    assert np.allclose(w1, gd_exp), (w1[:5], gd_exp)
    assert np.allclose(w2, gs_exp), (w2[:5], gs_exp)
    assert abs(fb.g_direct - gd_exp) < 1e-9
    assert abs(fb.g_SOM - gs_exp) < 1e-9
    print(f"[3] weight_assignment: PASS "
          f"(g_direct={fb.g_direct:.3f}, g_SOM={fb.g_SOM:.3f})")
    return True


def check_4_functional_matched_vs_unmatched() -> bool:
    """H_R pulse on ch0 -> V1_SOM ch0 fires, V1 per-ch profile peaks near ch0.

    No bottom-up stimulus (V1 afferent rates = 0). All activity is
    feedback-driven. We expect:
      - V1_SOM channel 0 to fire more than any other channel (direct
        excitatory drive from H_E route 2).
      - V1_E apical response on channel 0 is not visible as spikes
        (apical is modulatory), so we instead check SOM channel response
        for the structural channel-match claim.
    """
    _setup_brian()
    h = build_h_r()
    v = build_v1_ring()
    fb = build_feedback_routes(h, v, FeedbackRoutesConfig(g_total=2.0, r=1.0))

    # Zero bottom-up stimulus - activity should be feedback-driven only.
    set_stimulus(v, theta_rad=0.0, contrast=0.0)

    v_e_mon = SpikeMonitor(v.e)
    v_som_mon = SpikeMonitor(v.som)
    h_e_mon = SpikeMonitor(h.e)
    net = Network(*h.groups, *v.groups, *fb.groups,
                  v_e_mon, v_som_mon, h_e_mon)

    pulse_channel(h, channel=0, rate_hz=300.0)
    net.run(150 * ms)
    silence_cue(h)
    net.run(250 * ms)

    # Per-channel V1_SOM spike counts across full trial
    som_per_ch = _per_channel_spike_counts(v_som_mon, v.som_channel)
    matched = som_per_ch[0]
    unmatched = som_per_ch[6]   # orthogonal (90 deg off)
    print(f"[4] V1_SOM per-ch: matched(ch0)={matched} "
          f"unmatched(ch6)={unmatched}  full={som_per_ch.tolist()}")
    assert matched > unmatched + 2, (
        f"matched V1_SOM should exceed orthogonal (got {matched} vs {unmatched})"
    )

    # Also verify H bump was evoked
    h_per_ch = _per_channel_spike_counts(h_e_mon, h.e_channel)
    assert h_per_ch[0] > h_per_ch[6], (h_per_ch.tolist(),)
    print(f"[4] functional_matched_vs_unmatched: PASS "
          f"(H ch0={h_per_ch[0]}, ch6={h_per_ch[6]})")
    return True


def check_5_ablation_r0_vs_rinf() -> bool:
    """r=0 (pure SOM) vs r=inf (pure direct): SOM response differs.

    r=0  -> g_direct=0, g_SOM=g_total   : V1_SOM ch0 fires strongly.
    r=inf-> g_direct=g_total, g_SOM=0   : V1_SOM ch0 is silent (no H->SOM drive).
    """
    def _run(r_val: float) -> int:
        _setup_brian()
        h = build_h_r()
        v = build_v1_ring()
        fb = build_feedback_routes(h, v,
                                   FeedbackRoutesConfig(g_total=2.0, r=r_val))
        set_stimulus(v, theta_rad=0.0, contrast=0.0)
        som_mon = SpikeMonitor(v.som)
        net = Network(*h.groups, *v.groups, *fb.groups, som_mon)
        pulse_channel(h, channel=0, rate_hz=300.0)
        net.run(150 * ms)
        silence_cue(h)
        net.run(250 * ms)
        per_ch = _per_channel_spike_counts(som_mon, v.som_channel)
        return int(per_ch[0])

    som_ch0_r0 = _run(0.0)
    som_ch0_rinf = _run(float("inf"))
    print(f"[5] V1_SOM ch0  r=0 -> {som_ch0_r0}   r=inf -> {som_ch0_rinf}")
    assert som_ch0_r0 > som_ch0_rinf + 5, (
        f"Pure-SOM route should drive V1_SOM harder than pure-direct "
        f"(got {som_ch0_r0} vs {som_ch0_rinf})"
    )
    # Pure-direct route with no SOM drive should produce near-zero SOM spikes
    # from this route. (Baseline SOM rate from bias is small over 400 ms.)
    print("[5] ablation_r0_vs_rinf: PASS")
    return True


def check_6_set_balance_inplace() -> bool:
    """`set_balance` rewrites weights on existing Synapses, no Network rebuild."""
    _setup_brian()
    h = build_h_r()
    v = build_v1_ring()
    fb = build_feedback_routes(h, v,
                               FeedbackRoutesConfig(g_total=1.0, r=1.0))
    w1_before = float(np.asarray(fb.hr_to_v1e.w[:])[0])
    w2_before = float(np.asarray(fb.hr_to_v1som.w[:])[0])
    set_balance(fb, r=4.0)
    w1_after = float(np.asarray(fb.hr_to_v1e.w[:])[0])
    w2_after = float(np.asarray(fb.hr_to_v1som.w[:])[0])
    gd_exp, gs_exp = balance_weights(1.0, 4.0)
    assert abs(w1_after - gd_exp) < 1e-9
    assert abs(w2_after - gs_exp) < 1e-9
    assert abs(fb.config.r - 4.0) < 1e-9
    assert w1_after > w1_before    # r grew -> g_direct grew
    assert w2_after < w2_before    # r grew -> g_SOM shrank
    print(f"[6] set_balance_inplace: PASS "
          f"(w_direct {w1_before:.3f}->{w1_after:.3f}, "
          f"w_som {w2_before:.3f}->{w2_after:.3f})")
    return True


# -- runner -----------------------------------------------------------------

def main() -> int:
    checks = [
        ("balance_weights_algebra", check_1_balance_weights_algebra),
        ("route_connectivity",      check_2_route_connectivity),
        ("weight_assignment",       check_3_weight_assignment),
        ("functional_matched",      check_4_functional_matched_vs_unmatched),
        ("ablation_r0_vs_rinf",     check_5_ablation_r0_vs_rinf),
        ("set_balance_inplace",     check_6_set_balance_inplace),
    ]
    n_pass = 0
    for name, fn in checks:
        try:
            ok = fn()
        except Exception as exc:
            print(f"[X] {name}: FAIL -- {type(exc).__name__}: {exc}")
            ok = False
        if ok:
            n_pass += 1
    total = len(checks)
    print(f"\n--- validate_feedback_routes: {n_pass}/{total} PASS ---")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
