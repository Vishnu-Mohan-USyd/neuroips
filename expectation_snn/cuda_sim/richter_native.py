"""Native CUDA diagnostic wrapper for bounded frozen ctx_pred Richter runs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from expectation_snn.cuda_sim.export_bundle import export_ctx_pred_manifest
from expectation_snn.cuda_sim.native import (
    backend_info,
    run_frozen_richter_controlled_source_test,
)


RAW_COUNT_KEYS = (
    "v1_e.leader",
    "v1_e.preprobe",
    "v1_e.trailer",
    "hctx_e.leader",
    "hctx_e.preprobe",
    "hctx_e.trailer",
    "hpred_e.leader",
    "hpred_e.preprobe",
    "hpred_e.trailer",
)
RATE_KEYS = (
    "v1_e.leader",
    "v1_e.preprobe",
    "v1_e.trailer",
    "hctx_e.preprobe",
    "hctx_e.trailer",
    "hpred_e.preprobe",
    "hpred_e.trailer",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _npz_key(key: str) -> str:
    return key.replace(".", "_")


def _window_duration_s(phase_steps: dict[str, int], phase: str, dt_ms: float) -> float:
    start = int(phase_steps[f"{phase}_start_step"])
    end = int(phase_steps[f"{phase}_end_step"])
    return float(end - start) * float(dt_ms) / 1000.0


def _channel_rates(
    counts: np.ndarray,
    channels: np.ndarray,
    n_channels: int,
    duration_s: float,
) -> np.ndarray:
    out = np.zeros((n_channels,), dtype=np.float64)
    if duration_s <= 0.0:
        return out
    for channel in range(n_channels):
        mask = channels == channel
        if np.any(mask):
            out[channel] = float(np.mean(counts[mask])) / duration_s
    return out


def load_manifest(path: str | Path) -> dict[str, np.ndarray]:
    """Load a schema-v1 native manifest NPZ."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def export_manifest(
    *,
    ckpt_dir: str | Path,
    out_path: str | Path,
    seed: int = 42,
    r: float = 1.0,
    g_total: float = 1.0,
    v1_to_h_mode: str = "context_only",
    with_feedback_routes: bool = True,
) -> Path:
    """Export the current Brian2-defined frozen ctx_pred topology."""
    return export_ctx_pred_manifest(
        ckpt_dir=ckpt_dir,
        out_path=out_path,
        seed=int(seed),
        r=float(r),
        g_total=float(g_total),
        v1_to_h_mode=str(v1_to_h_mode),
        with_feedback_routes=bool(with_feedback_routes),
    )


def controlled_diagnostic(
    arrays: dict[str, Any],
    *,
    event_steps: Iterable[int],
    event_sources: Iterable[int],
    condition: dict[str, Any] | None = None,
    expected_channel: int = 0,
    unexpected_channel: int = 1,
    n_steps: int = 120,
    leader_start_step: int = 0,
    leader_end_step: int = 30,
    preprobe_start_step: int = 30,
    preprobe_end_step: int = 60,
    trailer_start_step: int = 60,
    trailer_end_step: int = 100,
    iti_start_step: int = 100,
    iti_end_step: int = 120,
) -> dict[str, Any]:
    """Run native CUDA with explicit V1 stimulus source events.

    The source events are afferent-level events. Each event fans out through
    the exported ``v1_stim_to_e`` CSR bank before V1/H state updates continue.
    """
    event_steps_arr = np.asarray([int(step) for step in event_steps], dtype=np.int32)
    event_sources_arr = np.asarray(
        [int(source) for source in event_sources],
        dtype=np.int32,
    )
    if event_steps_arr.shape != event_sources_arr.shape:
        raise ValueError("event_steps and event_sources must have matching shape")
    if event_steps_arr.size == 0:
        raise ValueError("controlled_diagnostic requires at least one source event")

    start = time.perf_counter()
    result = run_frozen_richter_controlled_source_test(
        arrays,
        event_steps=event_steps_arr.tolist(),
        event_sources=event_sources_arr.tolist(),
        expected_channel=int(expected_channel),
        unexpected_channel=int(unexpected_channel),
        n_steps=int(n_steps),
        leader_start_step=int(leader_start_step),
        leader_end_step=int(leader_end_step),
        preprobe_start_step=int(preprobe_start_step),
        preprobe_end_step=int(preprobe_end_step),
        trailer_start_step=int(trailer_start_step),
        trailer_end_step=int(trailer_end_step),
        iti_start_step=int(iti_start_step),
        iti_end_step=int(iti_end_step),
    )
    native_wall_s = time.perf_counter() - start

    phase_steps = {str(k): int(v) for k, v in result["phase_steps"].items()}
    dt_ms = float(result["dt_ms"])
    n_channels = int(np.asarray(arrays["n_channels"]))
    channel_maps = {
        "v1_e": np.asarray(arrays["v1_e_channel"], dtype=np.int32),
        "hctx_e": np.asarray(arrays["ctx_e_channel"], dtype=np.int32),
        "hpred_e": np.asarray(arrays["pred_e_channel"], dtype=np.int32),
    }

    raw_npz: dict[str, np.ndarray] = {
        "source_event_steps": event_steps_arr,
        "source_event_sources": event_sources_arr,
    }
    raw_summary: dict[str, Any] = {}
    for side in ("cpu", "cuda"):
        raw_counts = result[f"{side}_raw_counts"]
        rates = result[f"{side}_diagnostic_rates_hz"]
        for key in RAW_COUNT_KEYS:
            npz_key = _npz_key(f"{side}.{key}.counts")
            counts = np.asarray(raw_counts[key], dtype=np.int32)
            raw_npz[npz_key] = counts[:, None]
            raw_summary[npz_key] = {
                "shape": list(raw_npz[npz_key].shape),
                "sum": int(counts.sum()),
            }
        for key in RATE_KEYS:
            npz_key = _npz_key(f"{side}.{key}.rate_hz")
            rate = np.asarray(rates[key], dtype=np.float64)
            raw_npz[npz_key] = rate[:, None]
            raw_summary[npz_key] = {
                "shape": list(raw_npz[npz_key].shape),
                "mean": float(rate.mean()) if rate.size else 0.0,
            }
            pop, phase = key.split(".")
            duration_s = _window_duration_s(phase_steps, phase, dt_ms)
            raw_npz[_npz_key(f"{side}.{key}.channel_rate_hz")] = _channel_rates(
                np.asarray(raw_counts[key], dtype=np.int32),
                channel_maps[pop],
                n_channels,
                duration_s,
            )
    for side in ("cpu", "cuda"):
        source_counts = result[f"{side}_source_counts"]
        for key, values in source_counts.items():
            raw_npz[_npz_key(f"{side}.{key}")] = np.asarray(values, dtype=np.int32)

    max_error = {
        str(k): float(v) for k, v in result["max_abs_error"].items()
    }
    diagnostic = {
        "meta": {
            "schema": "native_richter_subset_v1",
            "script": "expectation_snn.cuda_sim.richter_native",
            "dt_ms": dt_ms,
            "n_steps": int(result["n_steps"]),
            "phase_steps": phase_steps,
            "expected_channel": int(result["expected_channel"]),
            "unexpected_channel": int(result["unexpected_channel"]),
        },
        "condition": condition or {
            "name": "controlled_source_small",
            "source_mode": "explicit_events",
        },
        "raw": raw_summary,
        "rates": {
            "keys": list(RATE_KEYS),
            "units": "Hz",
            "per_cell_arrays_in_npz": True,
            "per_channel_arrays_in_npz": True,
        },
        "source_events": {
            "mode": "explicit_events",
            "n_events": int(event_steps_arr.size),
            "steps": event_steps_arr.tolist(),
            "sources": event_sources_arr.tolist(),
            "counts": _jsonable(result["source_event_counts"]),
        },
        "backend": {
            "name": "native_cuda",
            "info": backend_info(),
        },
        "performance": {
            "native_wall_s": float(native_wall_s),
            "n_steps": int(result["n_steps"]),
        },
        "timings": {
            "native_wall_s": float(native_wall_s),
        },
        "edge_counts": _jsonable(result["edge_counts"]),
        "max_abs_error": max_error,
        "raw_npz_arrays": sorted(raw_npz),
    }
    return {
        "diagnostic": diagnostic,
        "npz_arrays": raw_npz,
        "native_result": result,
    }


def write_diagnostic(
    payload: dict[str, Any],
    *,
    out_json: str | Path,
    out_npz: str | Path,
) -> tuple[Path, Path]:
    """Write diagnostic JSON metadata and raw NPZ arrays."""
    out_json = Path(out_json)
    out_npz = Path(out_npz)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload["npz_arrays"])
    diagnostic = dict(payload["diagnostic"])
    diagnostic["raw_npz_path"] = str(out_npz)
    out_json.write_text(json.dumps(_jsonable(diagnostic), indent=2, sort_keys=True))
    return out_json, out_npz


def controlled_diagnostic_from_manifest(
    manifest_path: str | Path,
    *,
    event_steps: Iterable[int],
    event_sources: Iterable[int],
    out_json: str | Path | None = None,
    out_npz: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a manifest, run controlled native CUDA, and optionally write output."""
    arrays = load_manifest(manifest_path)
    payload = controlled_diagnostic(
        arrays,
        event_steps=event_steps,
        event_sources=event_sources,
        **kwargs,
    )
    payload["diagnostic"]["meta"]["manifest_path"] = str(manifest_path)
    if out_json is not None and out_npz is not None:
        write_diagnostic(payload, out_json=out_json, out_npz=out_npz)
    return payload


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--event-steps", required=True)
    parser.add_argument("--event-sources", required=True)
    parser.add_argument("--n-steps", type=int, default=120)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = controlled_diagnostic_from_manifest(
        args.manifest,
        event_steps=_parse_csv_ints(args.event_steps),
        event_sources=_parse_csv_ints(args.event_sources),
        n_steps=args.n_steps,
        out_json=args.out_json,
        out_npz=args.out_npz,
    )
    print(
        "richter_native:",
        f"wrote_json={args.out_json}",
        f"wrote_npz={args.out_npz}",
        f"events={payload['diagnostic']['source_events']['n_events']}",
        f"native_wall_s={payload['diagnostic']['performance']['native_wall_s']:.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
