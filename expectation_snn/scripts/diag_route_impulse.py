"""Diagnostic D4 — route impulse-response harness (Sprint 5d, task #41 step 5).

Purpose
-------
Sprint 5c meta-review (docs/SPRINT_5C_META_REVIEW.md) identified four
failure modes for the H-as-prediction module; Case B is "H → V1 feedback
routing broken". Diagnostic D4 isolates that by driving **fixed V1
stimulus** (θ=0 grating) and a **fixed H pulse train** (via H-clamp)
through the feedback routes, and recording V1 currents/rates in three
windows × four route configurations. No learning, no plasticity — purely
a mechanical impulse-response test of the wiring.

What this script does NOT do
----------------------------
- No assertions, no science verdict. This harness records data; the
  Debugger (task #42) then interprets it against Case B expectations.
- No sweeps over contrast/noise/topology. Fixed θ=0 grating at nominal
  contrast; fixed clamp channel (0) at 200 Hz.

Protocol
--------
Four route configurations (`set_balance` on the feedback routes):

  config        | r       | g_total | g_direct | g_SOM
  --------------+---------+---------+----------+-------
  off           | 1.0     | 0.0     | 0.00     | 0.00
  som_only      | 0.0     | 1.0     | 0.00     | 1.00
  apical_only   | inf     | 1.0     | 1.00     | 0.00
  both          | 1.0     | 1.0     | 0.50     | 0.50

Three windows (all sequential within one trial; state resets between
configs):

  W1 "baseline"      — blank (no grating), H-clamp OFF. 300 ms.
                       Sets the passive spontaneous floor.
  W2 "grating_only"  — θ=0 grating ON, H-clamp OFF. 300 ms.
                       Pure V1 impulse response; H is silent (V1→H OFF),
                       so feedback routes contribute nothing.
  W3 "grating+clamp" — θ=0 grating ON, H-clamp ON at channel 0. 300 ms.
                       External drive on H_E[ch=0] at 200 Hz; the route
                       configuration determines what reaches V1.

Signals recorded per (config, window)
-------------------------------------
- V1_E spike counts (per cell, shape (N_V1_E,), int64)
- V1_SOM spike counts (per cell)
- V1_PV spike counts (per cell)
- V1_E.I_ap_e mean ± std over the window (pA) — H→V1_E apical current
- V1_E.w_adapt mean ± std over the window (pA) — SFA adaptation
- V1_SOM.I_e mean ± std over the window (pA) — dominant H→SOM drive
                                                  (in `som_only` / `both`)

Output
------
- `expectation_snn/data/diag_route_impulse_seed{seed}.npz` — single
  dict-of-dicts numpy archive, keyed by `{config}/{window}/{signal}`.
- stdout table: per-config per-window matched-channel rates + currents.

Usage
-----
    python -m expectation_snn.scripts.diag_route_impulse [--seed 42]

References
----------
- Sprint 5c meta-review `expectation_snn/docs/SPRINT_5C_META_REVIEW.md`
  — D4 route-impulse-response test.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from brian2 import (
    Network, SpikeMonitor, StateMonitor, defaultclock, prefs, ms, pA,
    seed as b2_seed,
)

from expectation_snn.assays.runtime import (
    build_frozen_network, set_grating,
)
from expectation_snn.brian2_model.feedback_routes import set_balance
from expectation_snn.brian2_model.h_clamp import HClampConfig
from expectation_snn.brian2_model.h_ring import N_CHANNELS as H_N_CHANNELS
from expectation_snn.brian2_model.v1_ring import N_CHANNELS as V1_N_CHANNELS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RouteConfig:
    """One row of the 4-config grid."""
    name: str
    r: float
    g_total: float


ROUTE_CONFIGS: List[RouteConfig] = [
    RouteConfig("off",          r=1.0,           g_total=0.0),
    RouteConfig("som_only",     r=0.0,           g_total=1.0),
    RouteConfig("apical_only",  r=float("inf"),  g_total=1.0),
    RouteConfig("both",         r=1.0,           g_total=1.0),
]

# Windows: (name, grating_theta_rad_or_None, clamp_on_bool, duration_ms)
WINDOWS: List[Tuple[str, object, bool, float]] = [
    ("baseline",      None, False, 300.0),
    ("grating_only",  0.0,  False, 300.0),
    ("grating+clamp", 0.0,  True,  300.0),
]

CLAMP_TARGET_CHANNEL = 0
CLAMP_RATE_HZ = 200.0
STATE_MON_DT_MS = 1.0        # 1 ms sampling for current traces


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _snapshot_counts(mon: SpikeMonitor) -> np.ndarray:
    return np.asarray(mon.count[:], dtype=np.int64).copy()


def _window_counts(mon: SpikeMonitor, pre: np.ndarray) -> np.ndarray:
    return _snapshot_counts(mon) - pre


def _state_window_stats(
    mon: StateMonitor, t_start_ms: float, t_end_ms: float,
    var_name: str,
) -> Dict[str, float]:
    """Mean/std/min/max of a single StateMonitor variable over a time window,
    collapsed across all recorded cells. Values in pA (Brian2 'amp')."""
    t_ms = np.asarray(mon.t / ms)
    mask = (t_ms >= t_start_ms) & (t_ms < t_end_ms)
    if not mask.any():
        return {"mean_pA": float("nan"), "std_pA": float("nan"),
                "min_pA": float("nan"), "max_pA": float("nan")}
    arr = np.asarray(getattr(mon, var_name))[:, mask] / pA  # (n_cells, T)
    return {
        "mean_pA": float(arr.mean()),
        "std_pA":  float(arr.std()),
        "min_pA":  float(arr.min()),
        "max_pA":  float(arr.max()),
    }


def run_one_config(
    rc: RouteConfig, seed: int, verbose: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Run all 3 windows for one route-config. Returns dict keyed by window.

    Brian2 state is isolated per call (fresh Network / fresh bundle).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)

    hclamp_cfg = HClampConfig(
        target_channel=int(CLAMP_TARGET_CHANNEL),
        clamp_rate_hz=float(CLAMP_RATE_HZ),
    )
    bundle = build_frozen_network(
        h_kind="ht", seed=seed, with_cue=False,
        r=rc.r if np.isfinite(rc.r) else 1.0,  # set_balance will refine
        g_total=rc.g_total,
        with_v1_to_h="off",          # isolate: no V1→H confound
        with_feedback_routes=True,   # always build; gains set per config
        with_h_clamp=hclamp_cfg,
    )

    # Apply exact per-config balance (set_balance supports r=inf).
    set_balance(bundle.fb, r=rc.r, g_total=rc.g_total)

    v1 = bundle.v1_ring
    e_mon   = SpikeMonitor(v1.e,   name=f"diag_rip_e_{rc.name}_s{seed}")
    pv_mon  = SpikeMonitor(v1.pv,  name=f"diag_rip_pv_{rc.name}_s{seed}")
    som_mon = SpikeMonitor(v1.som, name=f"diag_rip_som_{rc.name}_s{seed}")

    # State monitors on all V1_E cells (192): I_ap_e and w_adapt. V1_SOM
    # I_e on all SOM cells. Sample every STATE_MON_DT_MS ms (coarse).
    e_state_mon = StateMonitor(
        v1.e, variables=["I_ap_e", "w_adapt"],
        record=True, dt=STATE_MON_DT_MS * ms,
        name=f"diag_rip_e_state_{rc.name}_s{seed}",
    )
    som_state_mon = StateMonitor(
        v1.som, variables=["I_e"],
        record=True, dt=STATE_MON_DT_MS * ms,
        name=f"diag_rip_som_state_{rc.name}_s{seed}",
    )

    groups = list(bundle.groups) + [e_mon, pv_mon, som_mon,
                                     e_state_mon, som_state_mon]
    net = Network(*groups)

    bundle.reset_all()
    assert bundle.h_clamp is not None
    bundle.h_clamp.set_active(False)
    set_grating(v1, theta_rad=None, contrast=0.0)

    # Presettle: 200 ms blank to stabilise baselines.
    net.run(200.0 * ms)

    results: Dict[str, Dict[str, object]] = {}
    t_cursor_ms = 200.0     # post-presettle clock

    # Spike-count snapshot helpers: we snapshot `mon.count[:]` before/after
    # each window.
    for w_name, w_theta, w_clamp_on, w_dur in WINDOWS:
        # Set stim + clamp for this window.
        if w_theta is None:
            set_grating(v1, theta_rad=None, contrast=0.0)
        else:
            set_grating(v1, theta_rad=float(w_theta), contrast=1.0)
        bundle.h_clamp.set_active(bool(w_clamp_on))

        pre_e   = _snapshot_counts(e_mon)
        pre_pv  = _snapshot_counts(pv_mon)
        pre_som = _snapshot_counts(som_mon)
        t_start_ms = t_cursor_ms

        net.run(w_dur * ms)

        t_cursor_ms = t_start_ms + w_dur
        ct_e   = _window_counts(e_mon,   pre_e)
        ct_pv  = _window_counts(pv_mon,  pre_pv)
        ct_som = _window_counts(som_mon, pre_som)

        I_ap_e   = _state_window_stats(e_state_mon, t_start_ms, t_cursor_ms, "I_ap_e")
        w_adapt  = _state_window_stats(e_state_mon, t_start_ms, t_cursor_ms, "w_adapt")
        I_e_som  = _state_window_stats(som_state_mon, t_start_ms, t_cursor_ms, "I_e")

        # Matched-channel convenience (channel 0, since θ=0 aligns to V1
        # channel 0 and clamp target is channel 0). PV is a pool with no
        # per-channel topology, so its "matched" rate is the population mean.
        n_e_per_ch   = int((v1.e_channel   == 0).sum())
        n_som_per_ch = int((v1.som_channel == 0).sum())
        n_pv_pool    = int(v1.pv.N)
        win_s = w_dur * 1e-3
        matched_rate_e_hz   = float(ct_e[v1.e_channel == 0].sum()
                                    / (max(n_e_per_ch, 1) * win_s))
        matched_rate_pv_hz  = float(ct_pv.sum()
                                    / (max(n_pv_pool, 1) * win_s))
        matched_rate_som_hz = float(ct_som[v1.som_channel == 0].sum()
                                    / (max(n_som_per_ch, 1) * win_s))

        results[w_name] = {
            "counts_v1_e":    ct_e,
            "counts_v1_pv":   ct_pv,
            "counts_v1_som":  ct_som,
            "I_ap_e_pA":      I_ap_e,
            "w_adapt_pA":     w_adapt,
            "I_e_som_pA":     I_e_som,
            "matched_rate_v1_e_hz":   matched_rate_e_hz,
            "matched_rate_v1_pv_hz":  matched_rate_pv_hz,
            "matched_rate_v1_som_hz": matched_rate_som_hz,
            "t_start_ms":    float(t_start_ms),
            "t_end_ms":      float(t_cursor_ms),
            "duration_ms":   float(w_dur),
        }

        if verbose:
            print(
                f"  [{rc.name:<11}] {w_name:<14}"
                f"  V1_E={matched_rate_e_hz:6.2f}Hz"
                f"  PV={matched_rate_pv_hz:6.2f}Hz"
                f"  SOM={matched_rate_som_hz:6.2f}Hz"
                f"  I_ap_e={I_ap_e['mean_pA']:+6.2f}pA"
                f"  w_adapt={w_adapt['mean_pA']:+6.2f}pA"
                f"  I_e_som={I_e_som['mean_pA']:+6.2f}pA"
            )

    # Turn everything off at end; free monitors with the network.
    set_grating(v1, theta_rad=None, contrast=0.0)
    bundle.h_clamp.set_active(False)
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir", type=str,
        default=str(_root / "expectation_snn" / "data"),
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"diag_route_impulse_seed{args.seed}.npz",
    )

    print(
        f"diag_route_impulse: seed={args.seed} h=H_T clamp_ch="
        f"{CLAMP_TARGET_CHANNEL} clamp_rate={CLAMP_RATE_HZ:.0f}Hz "
        f"window={WINDOWS[0][3]:.0f}ms"
    )
    print(
        f"  columns: V1_E/PV/SOM matched-channel rate (Hz), I_ap_e, "
        f"w_adapt, I_e_som (pA)"
    )

    all_results: Dict[str, Dict[str, Dict[str, object]]] = {}
    for rc in ROUTE_CONFIGS:
        if verbose:
            print(f"[{rc.name}]  r={rc.r!r}  g_total={rc.g_total:.2f}")
        all_results[rc.name] = run_one_config(rc, seed=args.seed, verbose=verbose)

    # Flatten into a single npz: keys = f"{config}/{window}/{signal}".
    flat: Dict[str, np.ndarray] = {}
    for cname, cres in all_results.items():
        for wname, wres in cres.items():
            for k, v in wres.items():
                key = f"{cname}/{wname}/{k}"
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{key}/{kk}"] = np.asarray(vv)
                else:
                    flat[key] = np.asarray(v)

    np.savez_compressed(out_path, **flat)
    print(f"\nwrote {out_path}  ({len(flat)} arrays)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
