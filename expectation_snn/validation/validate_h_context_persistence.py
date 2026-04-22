"""Primitive H_context persistence validator.

This is a bottom-up H_context probe only: no full Stage-1 training and no
downstream assay. It drives H_context channel 0, silences all inputs, then
measures post-offset channel rates. The validator is expected to fail under
current code if H_context does not hold the primitive bump in the Stage-1
band.

Run:
    python -m expectation_snn.validation.validate_h_context_persistence
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs, seed as b2_seed
from brian2 import start_scope

from ..assays.runtime import build_frozen_network
from ..brian2_model.h_context_prediction import (
    HContextPrediction,
    HContextPredictionConfig,
    build_h_context_prediction,
    silence_direction,
)
from ..brian2_model.h_ring import (
    HRingConfig,
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER_CHANNEL,
    pulse_channel,
    silence_cue,
)
from ..brian2_model.train import _stage1_h_cfg
from .stage_1_gate import (
    BUMP_PERSISTENCE_BAND_MS,
    BUMP_RATE_FLOOR_HZ,
    RUNAWAY_CEILING_HZ,
    compute_bump_persistence_ms,
)


SEED = 42
SETTLE_MS = 100.0
PULSE_MS = 300.0
POST_MS = 700.0
PULSE_RATE_HZ = 300.0
BIN_MS = 10.0
CTX_CHANNEL = 0
DIAG_CKPT_DIR = Path("data/checkpoints_diag")


@dataclass
class ProbeResult:
    """One H_context persistence probe result."""

    name: str
    skipped: bool
    skip_reason: str = ""
    persistence_ms: float = float("nan")
    max_post_rate_hz: float = float("nan")
    post_population_e_rate_hz: float = float("nan")
    pulse_peak_rate_hz: float = float("nan")
    post_channel_rate_hz: Optional[np.ndarray] = None
    passed: bool = False

    def summary(self) -> str:
        if self.skipped:
            return f"[SKIP] {self.name}: {self.skip_reason}"
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.name}: "
            f"persistence_ms={self.persistence_ms:.1f} "
            f"band=[{BUMP_PERSISTENCE_BAND_MS[0]:.0f},{BUMP_PERSISTENCE_BAND_MS[1]:.0f}] "
            f"post_population_e_rate_hz={self.post_population_e_rate_hz:.2f} "
            f"ceiling<={RUNAWAY_CEILING_HZ:.1f} "
            f"max_post_rate_hz={self.max_post_rate_hz:.2f} diagnostic_only "
            f"pulse_peak_rate_hz={self.pulse_peak_rate_hz:.2f}"
        )


def _setup_brian() -> None:
    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)


def _silence_bundle(bundle: HContextPrediction) -> None:
    silence_cue(bundle.ctx)
    silence_cue(bundle.pred)
    silence_direction(bundle)


def _channel_bin_rates(
    *,
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    e_channel: np.ndarray,
    t0_ms: float,
    duration_ms: float,
    bin_ms: float,
) -> np.ndarray:
    """Return per-bin, per-channel H_E rates in Hz, shape (n_bins, 12)."""
    n_bins = max(1, int(np.ceil(duration_ms / bin_ms)))
    out = np.zeros((n_bins, H_N_CHANNELS), dtype=np.float64)
    for b in range(n_bins):
        lo = t0_ms + b * bin_ms
        hi = min(t0_ms + (b + 1) * bin_ms, t0_ms + duration_ms)
        dur_s = max((hi - lo) * 1e-3, 1e-9)
        mask = (spike_t_ms >= lo) & (spike_t_ms < hi)
        if not np.any(mask):
            continue
        counts = np.bincount(
            e_channel[spike_i[mask]],
            minlength=H_N_CHANNELS,
        )
        out[b] = counts.astype(np.float64) / (H_N_E_PER_CHANNEL * dur_s)
    return out


def _probe_bundle(
    *,
    name: str,
    bundle_factory: Callable[[], Tuple[HContextPrediction, List[object]]],
) -> ProbeResult:
    _setup_brian()
    bundle, groups = bundle_factory()
    return _probe_constructed_bundle(name=name, bundle=bundle, groups=groups)


def _probe_constructed_bundle(
    *,
    name: str,
    bundle: HContextPrediction,
    groups: List[object],
) -> ProbeResult:
    _silence_bundle(bundle)

    ctx_mon = SpikeMonitor(bundle.ctx.e, name=f"{name}_ctx_e_mon")
    net = Network(*groups, ctx_mon)

    net.run(SETTLE_MS * ms)
    pulse_channel(bundle.ctx, channel=CTX_CHANNEL, rate_hz=PULSE_RATE_HZ)
    net.run(PULSE_MS * ms)
    offset_ms = float(net.t / ms)
    _silence_bundle(bundle)
    net.run(POST_MS * ms)

    spike_i = np.asarray(ctx_mon.i[:], dtype=np.int64)
    spike_t_ms = np.asarray(ctx_mon.t / ms, dtype=np.float64)
    e_channel = np.asarray(bundle.ctx.e_channel, dtype=np.int64)

    pulse_rates = _channel_bin_rates(
        spike_i=spike_i,
        spike_t_ms=spike_t_ms,
        e_channel=e_channel,
        t0_ms=SETTLE_MS,
        duration_ms=PULSE_MS,
        bin_ms=BIN_MS,
    )
    post_rates = _channel_bin_rates(
        spike_i=spike_i,
        spike_t_ms=spike_t_ms,
        e_channel=e_channel,
        t0_ms=offset_ms,
        duration_ms=POST_MS,
        bin_ms=BIN_MS,
    )
    driven_post = post_rates[:, CTX_CHANNEL]
    persistence_ms = compute_bump_persistence_ms(
        driven_post,
        offset_idx=0,
        dt_ms=BIN_MS,
        floor_hz=BUMP_RATE_FLOOR_HZ,
    )
    max_post_rate = float(post_rates.max() if post_rates.size else 0.0)
    pulse_peak = float(pulse_rates[:, CTX_CHANNEL].max() if pulse_rates.size else 0.0)
    post_mean = post_rates.mean(axis=0) if post_rates.size else np.zeros(H_N_CHANNELS)
    post_population_e_rate = float(post_mean.mean())
    passed = (
        BUMP_PERSISTENCE_BAND_MS[0] <= persistence_ms <= BUMP_PERSISTENCE_BAND_MS[1]
        and post_population_e_rate <= RUNAWAY_CEILING_HZ
    )
    assert np.isfinite(max_post_rate), "max_post_rate_hz must be finite"
    assert np.isfinite(post_population_e_rate), "post_population_e_rate_hz must be finite"
    assert np.isfinite(pulse_peak), "pulse_peak_rate_hz must be finite"
    return ProbeResult(
        name=name,
        skipped=False,
        persistence_ms=float(persistence_ms),
        max_post_rate_hz=max_post_rate,
        post_population_e_rate_hz=post_population_e_rate,
        pulse_peak_rate_hz=pulse_peak,
        post_channel_rate_hz=post_mean,
        passed=passed,
    )


def _raw_stage1_ctx_bundle() -> Tuple[HContextPrediction, List[object]]:
    h_cfg = _stage1_h_cfg(HRingConfig())
    bundle = build_h_context_prediction(
        config=HContextPredictionConfig(ctx_cfg=h_cfg, pred_cfg=h_cfg),
        rng=np.random.default_rng(SEED),
        ctx_name="raw_h_ctx_persist",
        pred_name="raw_h_pred_persist",
    )
    return bundle, list(bundle.groups)


def _build_runtime_frozen_bundle():
    return build_frozen_network(
        architecture="ctx_pred",
        seed=SEED,
        with_v1_to_h="off",
        with_feedback_routes=False,
        ckpt_dir=str(DIAG_CKPT_DIR),
    )


def _runtime_probe_or_skip() -> ProbeResult:
    stage0 = DIAG_CKPT_DIR / f"stage_0_seed{SEED}.npz"
    stage1 = DIAG_CKPT_DIR / f"stage_1_ctx_pred_seed{SEED}.npz"
    missing = [str(p) for p in (stage0, stage1) if not p.is_file()]
    if missing:
        return ProbeResult(
            name="frozen_runtime_checkpoints_diag",
            skipped=True,
            skip_reason=f"missing checkpoint(s): {missing}",
        )
    with np.load(stage1) as ckpt:
        has_config_metadata = "ctx_pred_config_json" in ckpt.files
    if not has_config_metadata:
        return ProbeResult(
            name="frozen_runtime_checkpoints_diag",
            skipped=True,
            skip_reason=(
                "legacy diagnostic checkpoint lacks ctx_pred config metadata; "
                "runtime would use ctx_pred_config_source=legacy_fallback"
            ),
        )
    _setup_brian()
    frozen = _build_runtime_frozen_bundle()
    source = str(frozen.meta.get("ctx_pred_config_source", "unknown"))
    if source == "legacy_fallback":
        return ProbeResult(
            name="frozen_runtime_checkpoints_diag",
            skipped=True,
            skip_reason="runtime reported ctx_pred_config_source=legacy_fallback",
        )
    if frozen.ctx_pred is None:
        raise RuntimeError("runtime ctx_pred bundle was not constructed")
    frozen.reset_h()
    frozen.silence_tang_direction()
    return _probe_constructed_bundle(
        name="frozen_runtime_checkpoints_diag",
        bundle=frozen.ctx_pred,
        groups=list(frozen.groups),
    )


def main() -> int:
    results: List[ProbeResult] = [
        _probe_bundle(
            name="raw_build_h_context_prediction_stage1_cfg",
            bundle_factory=_raw_stage1_ctx_bundle,
        ),
        _runtime_probe_or_skip(),
    ]

    for res in results:
        print(res.summary())
        if res.post_channel_rate_hz is not None:
            rates = np.array2string(
                res.post_channel_rate_hz,
                precision=2,
                separator=", ",
            )
            print(f"       post_mean_channel_rate_hz={rates}")

    raw = results[0]
    non_skipped_runtime = results[1:]
    runtime_ok = all(res.skipped or res.passed for res in non_skipped_runtime)
    passed = raw.passed and runtime_ok
    if not passed:
        print(
            "validate_h_context_persistence: FAIL "
            f"(requires persistence in [{BUMP_PERSISTENCE_BAND_MS[0]:.0f},"
            f"{BUMP_PERSISTENCE_BAND_MS[1]:.0f}] ms and "
            f"Stage-1-style post_population_e_rate_hz <= {RUNAWAY_CEILING_HZ:.1f}; "
            "max_post_rate_hz is diagnostic-only)"
        )
        return 1
    print(
        "validate_h_context_persistence: PASS "
        f"(raw arm passed; non-skipped runtime arms passed; "
        f"max_post_rate_hz diagnostic-only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
