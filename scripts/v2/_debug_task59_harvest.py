"""Task #59 — harvest exact key=value numbers requested by team-lead.

Runs the 4 previously-tested configs + adds a pre-Task-#56 baseline
(segment_length=1, soft_reset_scale=0: state fully reset every window,
matching pre-stateful behaviour).

Extracts:
  * bookkeeping norms at t=100, t=200 (or last-before-halt if NaN earlier)
  * per-config W_pv_l23_raw delta at t=200 (or last-before-halt)
  * first explosive step for the default config
"""
from __future__ import annotations

import math
from dataclasses import asdict

from scripts.v2._debug_task59_explosion import run_instrumented


def _safe_value(snaps, step: int, fn) -> float:
    """Return fn(snap) for the snapshot at `step`, or the last-before-halt."""
    if step < len(snaps):
        return fn(snaps[step])
    # halted before `step` — return last valid value
    return fn(snaps[-1])


def harvest(label: str, **kwargs) -> dict[str, float]:
    print(f"\n[running {label}]")
    snaps = run_instrumented(seed=42, batch_size=4, warmup_steps=30, **kwargs)
    print(f"  halted at step {len(snaps) - 1}")

    def pre_tk(s) -> float: return float(s.pre_traces_nkeys)
    def post_tk(s) -> float: return float(s.post_traces_nkeys)
    def reg_n(s) -> float: return s.regime_post_norm
    def w_pv_d(s) -> float: return s.deltas.get("l23_e.W_pv_l23_raw", float("nan"))
    def maxd(s) -> float:
        return max((v for v in s.deltas.values()
                    if not math.isnan(v) and not math.isinf(v)), default=float("nan"))

    out = {
        "halted_step": len(snaps) - 1,
        "pre_traces_nkeys_t0": pre_tk(snaps[0]),
        "pre_traces_nkeys_t100": _safe_value(snaps, 100, pre_tk),
        "pre_traces_nkeys_t200": _safe_value(snaps, 200, pre_tk),
        "post_traces_nkeys_t0": post_tk(snaps[0]),
        "post_traces_nkeys_t100": _safe_value(snaps, 100, post_tk),
        "post_traces_nkeys_t200": _safe_value(snaps, 200, post_tk),
        "regime_post_norm_t0": reg_n(snaps[0]),
        "regime_post_norm_t100": _safe_value(snaps, 100, reg_n),
        "regime_post_norm_t200": _safe_value(snaps, 200, reg_n),
        "W_pv_l23_delta_t0": w_pv_d(snaps[0]),
        "W_pv_l23_delta_t100": _safe_value(snaps, 100, w_pv_d),
        "W_pv_l23_delta_t200": _safe_value(snaps, 200, w_pv_d),
        "max_delta_t200": _safe_value(snaps, 200, maxd),
    }
    # first step where any rule delta crosses 1e3
    first = None
    for s in snaps:
        for v in s.deltas.values():
            if not math.isnan(v) and abs(v) > 1e3:
                first = s.step
                break
        if first is not None:
            break
    out["first_explosive_step"] = first if first is not None else -1
    return out


def main() -> None:
    configs = {
        "default_seg50_sc0p1":   dict(n_steps=300, segment_length=50, soft_reset_scale=0.1),
        "hard_reset_seg50_sc0": dict(n_steps=300, segment_length=50, soft_reset_scale=0.0),
        "no_reset_seg50_sc1":   dict(n_steps=300, segment_length=50, soft_reset_scale=1.0),
        "short_seg10_sc0p1":    dict(n_steps=300, segment_length=10, soft_reset_scale=0.1),
        "pre_task56_seg1_sc0":  dict(n_steps=300, segment_length=1, soft_reset_scale=0.0),
    }
    results = {}
    for name, kw in configs.items():
        results[name] = harvest(name, **kw)

    print("\n\n" + "=" * 76)
    print("=== HARVEST RESULTS (for team-lead key=value report) ===")
    print("=" * 76)
    for name, r in results.items():
        print(f"\n--- {name} ---")
        for k, v in r.items():
            if isinstance(v, float):
                print(f"{k}= {v:.6e}")
            else:
                print(f"{k}= {v}")


if __name__ == "__main__":
    main()
