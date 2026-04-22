#!/usr/bin/env python
"""Focused ctx_pred Richter feedback-balance diagnostic.

This script is intentionally diagnostic-only: it does not change model
dynamics or production assay code. It asks whether a frozen
H_context -> H_prediction checkpoint can drive V1 expected-vs-unexpected
response changes through the existing fixed H_prediction -> V1 feedback
routes.

Recorded array shapes use trial-major metadata and cell-major spike counts:

- ``trailer_counts_e``: ``(n_v1_e, n_trials)`` spikes during trailer window.
- ``h_pred_preprobe_rate_hz``: ``(n_trials, n_h_channels)`` mean Hz in the
  last ``preprobe_window_ms`` of the leader epoch.
- ``h_pred_trailer_rate_hz``: ``(n_trials, n_h_channels)`` mean Hz during
  the trailer epoch, when H_prediction -> V1 feedback would act.
- ``v1_channel_rate_hz``: ``(n_trials, n_v1_channels)`` mean V1_E Hz/channel
  during the trailer window.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs
from brian2 import pA
from brian2 import seed as b2_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expectation_snn.assays.metrics import suppression_vs_preference
from expectation_snn.assays.richter_crossover import _richter_thetas_rad
from expectation_snn.assays.runtime import (
    build_frozen_network,
    preprobe_h_rate_hz,
    set_grating,
)
from expectation_snn.brian2_model.h_context_prediction import HContextPredictionConfig
from expectation_snn.brian2_model.stimulus import N_CHANNELS as V1_N_CHANNELS


def _jsonable(x: Any) -> Any:
    """Convert numpy/Brian scalar containers into JSON-serializable values."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _snapshot(mon: SpikeMonitor) -> np.ndarray:
    return np.asarray(mon.count[:], dtype=np.int64).copy()


def _parse_r_values(text: str) -> List[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("--r-values must contain at least one float")
    return vals


def _brian_name_tag(r: float, feedback_routes: bool) -> str:
    r_tag = f"{float(r):g}".replace("-", "m").replace(".", "p")
    return f"r{r_tag}_fb{int(feedback_routes)}"


def _channel_for_theta(theta_rad: float, n_channels: int = V1_N_CHANNELS) -> int:
    chans = np.arange(n_channels, dtype=np.float64) * (np.pi / n_channels)
    d = np.abs(chans - float(theta_rad))
    d = np.minimum(d, np.pi - d)
    return int(np.argmin(d))


def _ensure_checkpoints(seed: int, ckpt_dir: Path) -> Dict[str, Any]:
    """Verify ctx_pred checkpoints and record any explicit Stage-0 copy.

    ``build_frozen_network`` expects Stage-0 and Stage-1 ctx_pred checkpoints
    in one directory. Diagnostic Stage-1 outputs are often written under
    ``data/checkpoints_diag`` while the canonical Stage-0 calibration remains
    under ``expectation_snn/data/checkpoints``. If Stage-0 is missing locally
    but present canonically, copy it and record provenance in the JSON output.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stage0 = ckpt_dir / f"stage_0_seed{seed}.npz"
    stage1 = ckpt_dir / f"stage_1_ctx_pred_seed{seed}.npz"
    provenance: Dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir),
        "stage0": str(stage0),
        "stage1_ctx_pred": str(stage1),
        "stage0_copied": False,
    }
    if not stage1.is_file():
        raise FileNotFoundError(f"missing Stage-1 ctx_pred checkpoint: {stage1}")
    if not stage0.is_file():
        fallback = Path("expectation_snn/data/checkpoints") / stage0.name
        if not fallback.is_file():
            raise FileNotFoundError(
                f"missing Stage-0 checkpoint: {stage0}; fallback absent: {fallback}"
            )
        shutil.copy2(fallback, stage0)
        provenance.update({
            "stage0_copied": True,
            "stage0_source": str(fallback),
        })
    return provenance


def _make_schedule(n_trials: int, seed: int) -> List[Dict[str, Any]]:
    """Build a balanced Richter-like expected/unexpected schedule.

    The expected transition is the deranged one-step rotation used by the
    production Richter schedule: ``trailer = leader + 30 deg`` on the six
    30-degree orientations. Unexpected trials use steps 2..5. For every trial
    we keep ``theta_expected`` separate from ``theta_T`` so forecast accuracy
    is judged against the expected next stimulus, not the actually presented
    unexpected trailer.
    """
    if n_trials < 2:
        raise ValueError("--n-trials must be >= 2")
    rng = np.random.default_rng(seed)
    thetas = _richter_thetas_rad()

    expected: List[Dict[str, Any]] = []
    unexpected: List[Dict[str, Any]] = []
    for leader_idx, theta_l in enumerate(thetas):
        exp_idx = (leader_idx + 1) % len(thetas)
        expected.append({
            "theta_L": float(theta_l),
            "theta_expected": float(thetas[exp_idx]),
            "theta_T": float(thetas[exp_idx]),
            "condition": 1,
            "leader_idx6": int(leader_idx),
            "expected_idx6": int(exp_idx),
            "trailer_idx6": int(exp_idx),
            "dtheta_step": 1,
        })
        for step in (2, 3, 4, 5):
            trailer_idx = (leader_idx + step) % len(thetas)
            unexpected.append({
                "theta_L": float(theta_l),
                "theta_expected": float(thetas[exp_idx]),
                "theta_T": float(thetas[trailer_idx]),
                "condition": 0,
                "leader_idx6": int(leader_idx),
                "expected_idx6": int(exp_idx),
                "trailer_idx6": int(trailer_idx),
                "dtheta_step": int(step),
            })

    n_exp = max(1, n_trials // 2)
    n_unexp = max(1, n_trials - n_exp)
    while n_exp + n_unexp > n_trials:
        n_exp -= 1

    def sample_pool(pool: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while len(out) < n:
            order = rng.permutation(len(pool))
            out.extend(dict(pool[int(i)]) for i in order)
        return out[:n]

    items = sample_pool(expected, n_exp) + sample_pool(unexpected, n_unexp)
    rng.shuffle(items)
    for k, item in enumerate(items):
        item["trial"] = int(k)
    return items


def _rate_by_channel(
    counts: np.ndarray,
    channel_per_cell: np.ndarray,
    n_channels: int,
    window_ms: float,
) -> np.ndarray:
    """Return per-channel mean cell rate, shape ``(n_trials, n_channels)``."""
    counts = np.asarray(counts, dtype=np.float64)
    ch = np.asarray(channel_per_cell, dtype=np.int64)
    out = np.zeros((counts.shape[1], n_channels), dtype=np.float64)
    window_s = float(window_ms) * 1e-3
    for c in range(n_channels):
        mask = (ch == c)
        if mask.any():
            out[:, c] = counts[mask, :].sum(axis=0) / mask.sum() / window_s
    return out


def _h_rate_from_counts(
    before: np.ndarray,
    after: np.ndarray,
    h_ring: Any,
    window_ms: float,
) -> np.ndarray:
    return preprobe_h_rate_hz(before, after, h_ring, window_ms)


def _condition_mean(values: np.ndarray, cond_mask: np.ndarray) -> Tuple[float, float, float]:
    exp = values[cond_mask == 1]
    unexp = values[cond_mask == 0]
    exp_mean = float(exp.mean()) if exp.size else float("nan")
    unexp_mean = float(unexp.mean()) if unexp.size else float("nan")
    return exp_mean, unexp_mean, exp_mean - unexp_mean


def _summarize_v1(
    trailer_counts_e: np.ndarray,
    theta_t: np.ndarray,
    cond_mask: np.ndarray,
    channel_per_cell: np.ndarray,
    trailer_ms: float,
) -> Dict[str, Any]:
    pref_rad = v1_e_preferred_thetas_dummy(channel_per_cell)
    pref_rank = suppression_vs_preference(
        trailer_counts_e, pref_rad, theta_t, cond_mask, n_bins=10,
    )
    bin_delta_counts = np.asarray(pref_rank["bin_delta"], dtype=np.float64)
    bin_delta_hz = bin_delta_counts / (float(trailer_ms) * 1e-3)
    flank_hz = float(bin_delta_hz[bin_delta_hz.size // 2:].mean())
    center_hz = float(bin_delta_hz[0])

    total_rate = (
        trailer_counts_e.sum(axis=0)
        / trailer_counts_e.shape[0]
        / (float(trailer_ms) * 1e-3)
    )
    total_exp, total_unexp, total_delta = _condition_mean(total_rate, cond_mask)

    channel_rate = _rate_by_channel(
        trailer_counts_e, channel_per_cell, V1_N_CHANNELS, trailer_ms,
    )
    local = np.zeros(theta_t.shape[0], dtype=np.float64)
    flank = np.zeros(theta_t.shape[0], dtype=np.float64)
    for k, theta in enumerate(theta_t):
        c0 = _channel_for_theta(float(theta), V1_N_CHANNELS)
        local[k] = channel_rate[k, c0]
        flank_ch = [((c0 - 1) % V1_N_CHANNELS), ((c0 + 1) % V1_N_CHANNELS)]
        flank[k] = float(channel_rate[k, flank_ch].mean())
    local_exp, local_unexp, local_delta = _condition_mean(local, cond_mask)
    flank_exp, flank_unexp, flank_delta = _condition_mean(flank, cond_mask)

    return {
        "pref_rank": {
            **pref_rank,
            "bin_delta_hz": bin_delta_hz,
            "bin_expected_hz": np.asarray(pref_rank["bin_expected"]) / (float(trailer_ms) * 1e-3),
            "bin_unexpected_hz": np.asarray(pref_rank["bin_unexpected"]) / (float(trailer_ms) * 1e-3),
        },
        "center_delta_hz": center_hz,
        "flank_delta_hz": flank_hz,
        "redist_hz": center_hz - flank_hz,
        "total_rate_hz_expected": total_exp,
        "total_rate_hz_unexpected": total_unexp,
        "total_delta_hz": total_delta,
        "local_channel_rate_hz_expected": local_exp,
        "local_channel_rate_hz_unexpected": local_unexp,
        "local_channel_delta_hz": local_delta,
        "flank_channel_rate_hz_expected": flank_exp,
        "flank_channel_rate_hz_unexpected": flank_unexp,
        "flank_channel_delta_hz": flank_delta,
        "v1_channel_rate_hz": channel_rate,
    }


def v1_e_preferred_thetas_dummy(channel_per_cell: np.ndarray) -> np.ndarray:
    """Per-cell preferred orientation from stored V1 channel ids."""
    return np.asarray(channel_per_cell, dtype=np.float64) * (np.pi / V1_N_CHANNELS)


def _summarize_h(
    h_pred_preprobe_rate_hz: np.ndarray,
    h_ctx_preprobe_rate_hz: np.ndarray,
    expected_idx6: np.ndarray,
    trailer_idx6: np.ndarray,
) -> Dict[str, Any]:
    expected_ch12 = (2 * expected_idx6.astype(np.int64)) % V1_N_CHANNELS
    trailer_ch12 = (2 * trailer_idx6.astype(np.int64)) % V1_N_CHANNELS
    pred_argmax12 = np.argmax(h_pred_preprobe_rate_hz, axis=1).astype(np.int64)
    pred_argmax6 = (2 * np.argmax(h_pred_preprobe_rate_hz[:, ::2], axis=1)).astype(np.int64)
    ctx_argmax12 = np.argmax(h_ctx_preprobe_rate_hz, axis=1).astype(np.int64)

    trial_idx = np.arange(expected_idx6.shape[0])
    expected_rate = h_pred_preprobe_rate_hz[trial_idx, expected_ch12]
    trailer_rate = h_pred_preprobe_rate_hz[trial_idx, trailer_ch12]

    return {
        "pred_argmax12": pred_argmax12,
        "pred_argmax6_as12": pred_argmax6,
        "ctx_argmax12": ctx_argmax12,
        "expected_ch12": expected_ch12,
        "trailer_ch12": trailer_ch12,
        "forecast_hit12": pred_argmax12 == expected_ch12,
        "forecast_hit6": pred_argmax6 == expected_ch12,
        "forecast_prob12": float(np.mean(pred_argmax12 == expected_ch12)),
        "forecast_prob6": float(np.mean(pred_argmax6 == expected_ch12)),
        "pred_expected_rate_hz_mean": float(expected_rate.mean()),
        "pred_trailer_rate_hz_mean": float(trailer_rate.mean()),
        "pred_peak_rate_hz_mean": float(h_pred_preprobe_rate_hz.max(axis=1).mean()),
        "ctx_peak_rate_hz_mean": float(h_ctx_preprobe_rate_hz.max(axis=1).mean()),
    }


def run_condition(
    *,
    seed: int,
    ckpt_dir: Path,
    r: float,
    g_total: float,
    feedback_routes: bool,
    v1_to_h_mode: str,
    ctx_pred_drive_pA: float | None,
    pred_bias_pA: float,
    schedule: List[Dict[str, Any]],
    leader_ms: float,
    trailer_ms: float,
    iti_ms: float,
    preprobe_window_ms: float,
    contrast: float,
) -> Dict[str, Any]:
    """Run one frozen ctx_pred condition and return raw arrays + summaries."""
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed)
    np.random.seed(seed)

    bundle = build_frozen_network(
        architecture="ctx_pred",
        seed=seed,
        r=r,
        g_total=g_total,
        with_cue=False,
        with_v1_to_h=v1_to_h_mode,
        with_feedback_routes=feedback_routes,
        with_preprobe_h_mon=False,
        ckpt_dir=str(ckpt_dir),
        ctx_pred_cfg=(
            HContextPredictionConfig(drive_amp_ctx_pred_pA=float(ctx_pred_drive_pA))
            if ctx_pred_drive_pA is not None else None
        ),
    )
    if bundle.ctx_pred is None:
        raise RuntimeError("build_frozen_network did not return ctx_pred bundle")
    if pred_bias_pA:
        bundle.ctx_pred.pred.e.I_bias = float(pred_bias_pA) * pA

    tag = _brian_name_tag(r, feedback_routes)
    e_mon = SpikeMonitor(bundle.v1_ring.e, name=f"diag_v1_e_{tag}")
    som_mon = SpikeMonitor(bundle.v1_ring.som, name=f"diag_v1_som_{tag}")
    pv_mon = SpikeMonitor(bundle.v1_ring.pv, name=f"diag_v1_pv_{tag}")
    ctx_mon = SpikeMonitor(bundle.ctx_pred.ctx.e, name=f"diag_h_ctx_{tag}")
    pred_mon = SpikeMonitor(bundle.ctx_pred.pred.e, name=f"diag_h_pred_{tag}")
    net = Network(*bundle.groups, e_mon, som_mon, pv_mon, ctx_mon, pred_mon)

    n_trials = len(schedule)
    n_e = int(bundle.v1_ring.e.N)
    n_som = int(bundle.v1_ring.som.N)
    n_pv = int(bundle.v1_ring.pv.N)
    n_h_ch = int(bundle.ctx_pred.pred.e_channel.max()) + 1

    leader_counts_e = np.zeros((n_e, n_trials), dtype=np.int64)
    trailer_counts_e = np.zeros((n_e, n_trials), dtype=np.int64)
    trailer_counts_som = np.zeros((n_som, n_trials), dtype=np.int64)
    trailer_counts_pv = np.zeros((n_pv, n_trials), dtype=np.int64)
    h_pred_preprobe_rate_hz = np.zeros((n_trials, n_h_ch), dtype=np.float64)
    h_ctx_preprobe_rate_hz = np.zeros((n_trials, n_h_ch), dtype=np.float64)
    h_pred_trailer_rate_hz = np.zeros((n_trials, n_h_ch), dtype=np.float64)
    h_ctx_trailer_rate_hz = np.zeros((n_trials, n_h_ch), dtype=np.float64)

    theta_l = np.zeros(n_trials, dtype=np.float64)
    theta_expected = np.zeros(n_trials, dtype=np.float64)
    theta_t = np.zeros(n_trials, dtype=np.float64)
    cond_mask = np.zeros(n_trials, dtype=np.int64)
    dtheta_step = np.zeros(n_trials, dtype=np.int64)
    expected_idx6 = np.zeros(n_trials, dtype=np.int64)
    trailer_idx6 = np.zeros(n_trials, dtype=np.int64)

    context_only = v1_to_h_mode == "context_only"
    if context_only and bundle.v1_to_h is None:
        raise RuntimeError("context_only requested but V1->H was not built")

    pre_leader_ms = float(leader_ms) - float(preprobe_window_ms)
    if pre_leader_ms < 0:
        raise ValueError("preprobe_window_ms must be <= leader_ms")

    for k, item in enumerate(schedule):
        theta_l[k] = item["theta_L"]
        theta_expected[k] = item["theta_expected"]
        theta_t[k] = item["theta_T"]
        cond_mask[k] = item["condition"]
        dtheta_step[k] = item["dtheta_step"]
        expected_idx6[k] = item["expected_idx6"]
        trailer_idx6[k] = item["trailer_idx6"]

        bundle.reset_all()
        if bundle.v1_to_h is not None:
            bundle.v1_to_h.set_active(True)

        set_grating(bundle.v1_ring, theta_rad=item["theta_L"], contrast=contrast)
        pre_e = _snapshot(e_mon)
        if pre_leader_ms > 0:
            net.run(pre_leader_ms * ms)
        ctx_before = _snapshot(ctx_mon)
        pred_before = _snapshot(pred_mon)
        if preprobe_window_ms > 0:
            net.run(preprobe_window_ms * ms)
        ctx_after = _snapshot(ctx_mon)
        pred_after = _snapshot(pred_mon)
        leader_counts_e[:, k] = _snapshot(e_mon) - pre_e

        h_ctx_preprobe_rate_hz[k, :] = _h_rate_from_counts(
            ctx_before, ctx_after, bundle.ctx_pred.ctx, preprobe_window_ms,
        )
        h_pred_preprobe_rate_hz[k, :] = _h_rate_from_counts(
            pred_before, pred_after, bundle.ctx_pred.pred, preprobe_window_ms,
        )

        set_grating(bundle.v1_ring, theta_rad=item["theta_T"], contrast=contrast)
        if context_only:
            bundle.v1_to_h.set_active(False)
        pre_e = _snapshot(e_mon)
        pre_som = _snapshot(som_mon)
        pre_pv = _snapshot(pv_mon)
        ctx_before_trailer = _snapshot(ctx_mon)
        pred_before_trailer = _snapshot(pred_mon)
        net.run(float(trailer_ms) * ms)
        ctx_after_trailer = _snapshot(ctx_mon)
        pred_after_trailer = _snapshot(pred_mon)
        trailer_counts_e[:, k] = _snapshot(e_mon) - pre_e
        trailer_counts_som[:, k] = _snapshot(som_mon) - pre_som
        trailer_counts_pv[:, k] = _snapshot(pv_mon) - pre_pv
        h_ctx_trailer_rate_hz[k, :] = _h_rate_from_counts(
            ctx_before_trailer, ctx_after_trailer, bundle.ctx_pred.ctx, trailer_ms,
        )
        h_pred_trailer_rate_hz[k, :] = _h_rate_from_counts(
            pred_before_trailer, pred_after_trailer, bundle.ctx_pred.pred, trailer_ms,
        )

        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        if context_only:
            bundle.v1_to_h.set_active(True)
        if iti_ms > 0:
            net.run(float(iti_ms) * ms)

    channel_e = np.asarray(bundle.v1_ring.e_channel, dtype=np.int64)
    v1_summary = _summarize_v1(
        trailer_counts_e, theta_t, cond_mask, channel_e, trailer_ms,
    )
    h_summary = _summarize_h(
        h_pred_preprobe_rate_hz, h_ctx_preprobe_rate_hz,
        expected_idx6, trailer_idx6,
    )
    h_trailer_summary = _summarize_h(
        h_pred_trailer_rate_hz, h_ctx_trailer_rate_hz,
        expected_idx6, trailer_idx6,
    )

    return {
        "condition": {
            "r": float(r),
            "g_total": float(g_total),
            "with_feedback_routes": bool(feedback_routes),
            "with_v1_to_h": str(v1_to_h_mode),
            "ctx_pred_drive_pA": (
                None if ctx_pred_drive_pA is None else float(ctx_pred_drive_pA)
            ),
            "pred_bias_pA": float(pred_bias_pA),
            "g_direct": float(bundle.meta["g_direct"]),
            "g_SOM": float(bundle.meta["g_SOM"]),
        },
        "summary": {
            **{f"v1_{k}": v for k, v in v1_summary.items() if k != "v1_channel_rate_hz" and k != "pref_rank"},
            **{f"h_{k}": v for k, v in h_summary.items() if not isinstance(v, np.ndarray)},
            **{f"h_trailer_{k}": v for k, v in h_trailer_summary.items() if not isinstance(v, np.ndarray)},
        },
        "metrics": {
            "pref_rank": v1_summary["pref_rank"],
            "h": h_summary,
            "h_trailer": h_trailer_summary,
        },
        "raw": {
            "leader_counts_e": leader_counts_e,
            "trailer_counts_e": trailer_counts_e,
            "trailer_counts_som": trailer_counts_som,
            "trailer_counts_pv": trailer_counts_pv,
            "v1_e_channel": channel_e,
            "v1_channel_rate_hz": v1_summary["v1_channel_rate_hz"],
            "h_ctx_preprobe_rate_hz": h_ctx_preprobe_rate_hz,
            "h_pred_preprobe_rate_hz": h_pred_preprobe_rate_hz,
            "h_ctx_trailer_rate_hz": h_ctx_trailer_rate_hz,
            "h_pred_trailer_rate_hz": h_pred_trailer_rate_hz,
            "theta_L": theta_l,
            "theta_expected": theta_expected,
            "theta_T": theta_t,
            "condition_mask": cond_mask,
            "dtheta_step": dtheta_step,
            "expected_idx6": expected_idx6,
            "trailer_idx6": trailer_idx6,
        },
        "bundle_meta": bundle.meta,
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("data/checkpoints_diag"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--r-values", type=str, default="0.25,1,4")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--g-total", type=float, default=1.0)
    ap.add_argument(
        "--ctx-pred-drive-pA",
        type=float,
        default=None,
        help=(
            "Diagnostic-only override for H_context -> H_prediction per-spike "
            "current. Default uses the model config/checkpoint value."
        ),
    )
    ap.add_argument(
        "--pred-bias-pA",
        type=float,
        default=0.0,
        help=(
            "Diagnostic-only uniform tonic bias on H_prediction E cells. "
            "This tests excitability/persistence without channel-specific clamping."
        ),
    )
    ap.add_argument("--leader-ms", type=float, default=500.0)
    ap.add_argument("--trailer-ms", type=float, default=500.0)
    ap.add_argument("--iti-ms", type=float, default=1500.0)
    ap.add_argument("--preprobe-window-ms", type=float, default=100.0)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument(
        "--v1-to-h-mode",
        choices=("continuous", "context_only", "off"),
        default="context_only",
        help="Assay-time V1->H mode. Use off as a no-context control.",
    )
    ap.add_argument(
        "--skip-feedback-off",
        action="store_true",
        help="Only run feedback-on conditions for quick smoke tests.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    provenance = _ensure_checkpoints(args.seed, args.ckpt_dir)
    r_values = _parse_r_values(args.r_values)
    schedule = _make_schedule(args.n_trials, args.seed)

    results: Dict[str, Any] = {
        "meta": {
            "script": "scripts/diag_ctx_pred_richter_balance.py",
            "seed": int(args.seed),
            "checkpoint_provenance": provenance,
            "r_values": r_values,
            "n_trials": int(len(schedule)),
            "leader_ms": float(args.leader_ms),
            "trailer_ms": float(args.trailer_ms),
            "iti_ms": float(args.iti_ms),
            "preprobe_window_ms": float(args.preprobe_window_ms),
            "contrast": float(args.contrast),
            "ctx_pred_drive_pA": (
                None if args.ctx_pred_drive_pA is None
                else float(args.ctx_pred_drive_pA)
            ),
            "pred_bias_pA": float(args.pred_bias_pA),
            "schedule": schedule,
        },
        "conditions": {},
    }

    for r in r_values:
        key_on = f"r={r:g}|feedback=on|v1_to_h={args.v1_to_h_mode}"
        results["conditions"][key_on] = run_condition(
            seed=args.seed,
            ckpt_dir=args.ckpt_dir,
            r=r,
            g_total=args.g_total,
            feedback_routes=True,
            v1_to_h_mode=args.v1_to_h_mode,
            ctx_pred_drive_pA=args.ctx_pred_drive_pA,
            pred_bias_pA=args.pred_bias_pA,
            schedule=schedule,
            leader_ms=args.leader_ms,
            trailer_ms=args.trailer_ms,
            iti_ms=args.iti_ms,
            preprobe_window_ms=args.preprobe_window_ms,
            contrast=args.contrast,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
        print(f"partial wrote {args.out} after {key_on}", flush=True)
        if not args.skip_feedback_off:
            key_off = f"r={r:g}|feedback=off|v1_to_h={args.v1_to_h_mode}"
            results["conditions"][key_off] = run_condition(
                seed=args.seed,
                ckpt_dir=args.ckpt_dir,
                r=r,
                g_total=args.g_total,
                feedback_routes=False,
                v1_to_h_mode=args.v1_to_h_mode,
                ctx_pred_drive_pA=args.ctx_pred_drive_pA,
                pred_bias_pA=args.pred_bias_pA,
                schedule=schedule,
                leader_ms=args.leader_ms,
                trailer_ms=args.trailer_ms,
                iti_ms=args.iti_ms,
                preprobe_window_ms=args.preprobe_window_ms,
                contrast=args.contrast,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
            print(f"partial wrote {args.out} after {key_off}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    for key, res in results["conditions"].items():
        s = res["summary"]
        print(
            key,
            f"forecast6={s['h_forecast_prob6']:.3f}",
            f"trailer_forecast6={s['h_trailer_forecast_prob6']:.3f}",
            f"trailer_peak_hz={s['h_trailer_pred_peak_rate_hz_mean']:.3f}",
            f"center_delta_hz={s['v1_center_delta_hz']:.3f}",
            f"flank_delta_hz={s['v1_flank_delta_hz']:.3f}",
            f"total_delta_hz={s['v1_total_delta_hz']:.3f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
