#!/usr/bin/env python3
"""Run and summarize native Richter learned-route impartiality/factorial assay.

This utility deliberately scores learned/static architecture routes with all
explicit engineered suppressors disabled. The previous raw-Ie predicted
suppression condition may be included only as an invalid positive control from
an existing artifact; it is not used as success evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from analyze_native_richter_pseudovoxels import (
    _full_argmax_decoder,
    _jsonify,
    _load_native_artifact,
    _six_class_transfer_decoder_pair,
)


DEFAULT_STAGE3_DIR = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424"
)
DEFAULT_CHECKPOINT = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "expectation_snn/data/checkpoints_native_cpp/"
    "stage1_temporal_sweep_20260424_fix2/stage1_ctx_pred_seed42_n72.json"
)
DEFAULT_BINARY = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "cpp_cuda/build/expectation_snn_native"
)
DEFAULT_LOG_DIR = Path("/workspace/neuroips_gpu_migration_20260422/logs")
DEFAULT_LOCALIZER = DEFAULT_STAGE3_DIR / (
    "richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_sigma22_"
    "sensory_only_feedback_g0_r03333333333333333_center010.json"
)
DEFAULT_INVALID_RAW_IE = DEFAULT_STAGE3_DIR / (
    "predicted_raw_suppression_sigma22_20260424/"
    "richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_feedback_"
    "r03333333333333333_gtotal20_center010_predraw_s16_n0p5.json"
)

KEEP_PS = (0.05, 0.02, 0.015, 0.01)
N_CHANNELS = 12


@dataclass(frozen=True)
class Row:
    condition: str
    name: str
    feedback_g_total: float
    feedback_r: float
    feedback_som_center_weight: float = 0.10
    note: str = ""
    extra_args: Sequence[str] = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        return self.name.replace(".", "p").replace("-", "m")


def _rows() -> List[Row]:
    rows: List[Row] = [
        Row(
            "A_held_replay_trailer_gating_only",
            "held_replay_static_routes_g2_r03333333333333333",
            2.0,
            0.3333333333333333,
            note=(
                "Supported static-route baseline: held replay plus trailer "
                "ctx_to_pred gate, no predicted suppression, no divisive suppressors."
            ),
        ),
    ]
    for g in (0.5, 1.0, 2.0, 4.0):
        rows.append(
            Row(
                "B_direct_apical_only",
                f"direct_only_r1000_g{g:g}",
                g,
                1000.0,
                note="Direct/apical branch dominates; SOM feedback is effectively off.",
            )
        )
    for g in (0.5, 1.0, 2.0, 4.0):
        rows.append(
            Row(
                "C_som_only",
                f"som_only_r0_g{g:g}",
                g,
                0.0,
                note="SOM branch only; direct/apical feedback is off.",
            )
        )
    for r in (0.0, 0.1, 0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 2.0, 10.0, 1000.0):
        rows.append(
            Row(
                "D_direct_som_balance_fixed_total",
                f"balance_g2_r{r:g}",
                2.0,
                r,
                note="Fixed total feedback balance sweep; no engineered suppressors.",
            )
        )
    return rows


def _run_command(
    cmd: Sequence[str],
    log_path: Path,
    cwd: Path,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall = time.monotonic() - t0
    return {
        "cmd": list(cmd),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "wall_seconds": wall,
    }


def _native_command(row: Row, binary: Path, checkpoint: Path, out: Path) -> List[str]:
    return [
        str(binary),
        "richter-dampening",
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
        "--seed",
        "4242",
        "--reps-expected",
        "4",
        "--reps-unexpected",
        "4",
        "--execution-mode",
        "gpu_only_production",
        "--grating-rate-hz",
        "100",
        "--baseline-rate-hz",
        "0",
        "--v1-stim-sigma-deg",
        "22.0",
        "--feedback-g-total",
        repr(float(row.feedback_g_total)),
        "--feedback-r",
        repr(float(row.feedback_r)),
        "--feedback-som-center-weight",
        repr(float(row.feedback_som_center_weight)),
        "--feedback-replay-mode",
        "raw",
        "--v1-predicted-suppression-scale",
        "0",
        "--v1-predicted-suppression-neighbor-weight",
        "0",
        "--v1-som-divisive-scale",
        "0",
        "--v1-direct-divisive-scale",
        "0",
        "--v1-feedforward-divisive-scale",
        "0",
        *row.extra_args,
    ]


def _channel_summary(
    counts: np.ndarray,
    leader_channel: np.ndarray,
    trailer_channel: np.ndarray,
    is_expected: np.ndarray,
) -> Dict[str, Any]:
    counts = np.asarray(counts, dtype=np.float64)
    leader_channel = np.asarray(leader_channel, dtype=np.int64)
    trailer_channel = np.asarray(trailer_channel, dtype=np.int64)
    is_expected = np.asarray(is_expected, dtype=bool)
    predicted_channel = (leader_channel + 2) % N_CHANNELS
    out: Dict[str, Any] = {}
    for cond, mask in (("expected", is_expected), ("unexpected", ~is_expected)):
        if not np.any(mask):
            out[cond] = {}
            continue
        cond_counts = counts[mask]
        cond_pred = predicted_channel[mask]
        cond_actual = trailer_channel[mask]
        pred_vals = cond_counts[np.arange(cond_counts.shape[0]), cond_pred]
        actual_vals = cond_counts[np.arange(cond_counts.shape[0]), cond_actual]
        ranks_desc = np.argsort(-cond_counts, axis=1)
        pred_rank = np.array([
            int(np.flatnonzero(ranks_desc[i] == cond_pred[i])[0] + 1)
            for i in range(cond_counts.shape[0])
        ])
        actual_rank = np.array([
            int(np.flatnonzero(ranks_desc[i] == cond_actual[i])[0] + 1)
            for i in range(cond_counts.shape[0])
        ])
        out[cond] = {
            "predicted_channel_count_mean": float(np.mean(pred_vals)),
            "actual_channel_count_mean": float(np.mean(actual_vals)),
            "predicted_minus_actual_count_mean": float(np.mean(pred_vals - actual_vals)),
            "predicted_rank_mean": float(np.mean(pred_rank)),
            "actual_rank_mean": float(np.mean(actual_rank)),
            "top1_matches_predicted_fraction": float(
                np.mean(np.argmax(cond_counts, axis=1) == cond_pred)
            ),
            "top1_matches_actual_fraction": float(
                np.mean(np.argmax(cond_counts, axis=1) == cond_actual)
            ),
        }
    return out


def _extract_metrics(
    artifact_path: Path,
    localizer_path: Path,
    keep_ps: Sequence[float],
    n_seeds: int,
) -> Dict[str, Any]:
    loaded = _load_native_artifact(artifact_path)
    artifact = loaded["artifact"]
    trial_data = artifact["trial_data"]
    is_expected = np.asarray(trial_data["is_expected"], dtype=bool)
    leader_channel = np.asarray(trial_data["leader_channel"], dtype=np.int64)
    trailer_channel = np.asarray(trial_data["trailer_channel"], dtype=np.int64)
    v1_counts = np.asarray(trial_data["v1_e_trailer_channel_counts"], dtype=np.float64)
    hpred_live = np.asarray(
        trial_data.get("hpred_e_trailer_100ms_channel_counts", []),
        dtype=np.float64,
    )
    hpred_held = np.asarray(
        trial_data.get("hpred_feedback_held_trailer_100ms_channel_counts", []),
        dtype=np.float64,
    )
    v1_som = np.asarray(
        trial_data.get("v1_som_trailer_channel_counts", []),
        dtype=np.float64,
    )

    transfer = _six_class_transfer_decoder_pair(
        artifact_path.stem,
        artifact_path,
        localizer_path.stem,
        localizer_path,
        keep_ps,
        n_seeds,
    )
    full_argmax = _full_argmax_decoder(v1_counts, trailer_channel, is_expected)

    def metric(name: str) -> float:
        return float(artifact["metrics"][name])

    q_e = metric("expected_v1_trailer_q_active_fC_per_trial")
    q_u = metric("unexpected_v1_trailer_q_active_fC_per_trial")
    act_e = metric("expected_v1_trailer_count_per_trial")
    act_u = metric("unexpected_v1_trailer_count_per_trial")

    out: Dict[str, Any] = {
        "artifact_path": str(artifact_path),
        "feedback": artifact.get("feedback", {}),
        "timing_seconds": artifact.get("timing_seconds", {}),
        "native_metrics": {
            "v1_q_active_fC_per_trial": {
                "expected": q_e,
                "unexpected": q_u,
                "unexpected_minus_expected": q_u - q_e,
            },
            "v1e_q_active_fC_per_trial": {
                "expected": metric("expected_v1e_trailer_q_active_fC_per_trial"),
                "unexpected": metric("unexpected_v1e_trailer_q_active_fC_per_trial"),
            },
            "v1som_q_active_fC_per_trial": {
                "expected": metric("expected_v1som_trailer_q_active_fC_per_trial"),
                "unexpected": metric("unexpected_v1som_trailer_q_active_fC_per_trial"),
            },
            "v1_activity_counts_per_trial": {
                "expected": act_e,
                "unexpected": act_u,
                "unexpected_minus_expected": act_u - act_e,
            },
        },
        "decoder": {
            "no_noise_full_argmax_debug_only": {
                "expected": full_argmax["expected"]["actual_channel_accuracy"],
                "unexpected": full_argmax["unexpected"]["actual_channel_accuracy"],
            },
            "noisy_pseudovoxel": {},
        },
        "trial_counts": transfer["trial_counts"],
        "v1_e_predicted_actual": _channel_summary(
            v1_counts,
            leader_channel,
            trailer_channel,
            is_expected,
        ),
        "pathology_flags": [],
    }

    if hpred_live.size:
        out["hpred_live_predicted_actual"] = _channel_summary(
            hpred_live.sum(axis=1),
            leader_channel,
            trailer_channel,
            is_expected,
        )
    if hpred_held.size:
        out["hpred_held_feedback_predicted_actual"] = _channel_summary(
            hpred_held.sum(axis=1),
            leader_channel,
            trailer_channel,
            is_expected,
        )
    if v1_som.size:
        out["v1_som_predicted_actual"] = _channel_summary(
            v1_som,
            leader_channel,
            trailer_channel,
            is_expected,
        )

    for keep_p, item in transfer["binomial_thinning"].items():
        summary = item["summary"]
        chance = item["chance_controls"]
        out["decoder"]["noisy_pseudovoxel"][keep_p] = {
            "unexpected_minus_expected_accuracy": (
                summary["unexpected_minus_expected_accuracy"]
            ),
            "unexpected_minus_expected_true_class_margin_mean": (
                summary["unexpected_minus_expected_true_class_margin_mean"]
            ),
            "expected_accuracy": summary["expected_accuracy"],
            "unexpected_accuracy": summary["unexpected_accuracy"],
            "chance_shuffled_localizer_delta": (
                chance["shuffled_localizer_labels"][
                    "unexpected_minus_expected_accuracy"
                ]
            ),
            "chance_shuffled_test_delta": (
                chance["shuffled_test_labels_conditionwise"][
                    "unexpected_minus_expected_accuracy"
                ]
            ),
        }

    if act_e < 5.0 or act_u < 5.0:
        out["pathology_flags"].append("near_zero_v1_activity")
    return out


def _row_markdown(row: Mapping[str, Any]) -> str:
    m = row["metrics"]["native_metrics"]
    dec = row["metrics"]["decoder"]["noisy_pseudovoxel"]
    return (
        f"| {row['condition']} | {row['name']} | "
        f"{m['v1_q_active_fC_per_trial']['expected']:.4f} | "
        f"{m['v1_q_active_fC_per_trial']['unexpected']:.4f} | "
        f"{m['v1_q_active_fC_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{m['v1_activity_counts_per_trial']['expected']:.4f} | "
        f"{m['v1_activity_counts_per_trial']['unexpected']:.4f} | "
        f"{m['v1_activity_counts_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{dec['0.02']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.015']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{','.join(row['metrics']['pathology_flags']) or 'none'} |"
    )


def _interpret(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid_rows = [
        r for r in rows
        if r.get("evidence_status") == "valid_learned_static_route"
    ]
    natural = []
    for r in valid_rows:
        metrics = r["metrics"]
        native = metrics["native_metrics"]
        dec = metrics["decoder"]["noisy_pseudovoxel"]
        q_ok = native["v1_q_active_fC_per_trial"]["unexpected_minus_expected"] > 0.0
        act_ok = (
            native["v1_activity_counts_per_trial"]["unexpected_minus_expected"] > 0.0
        )
        dec_ok = all(
            dec[str(float(k))]["unexpected_minus_expected_accuracy"]["mean"] > 0.0
            for k in KEEP_PS
        )
        if q_ok and act_ok and dec_ok and not metrics["pathology_flags"]:
            natural.append(r["name"])
    return {
        "natural_dampening_rows": natural,
        "natural_dampening_found": bool(natural),
        "criterion": (
            "valid learned/static route row with expected lower V1 Q, expected "
            "lower V1 activity, and expected lower noisy pseudo-voxel decoding "
            "than unexpected at all requested keep_p values"
        ),
        "raw_ie_policy": (
            "raw_ie predicted suppression is explicit engineered suppression "
            "and is excluded from success evidence"
        ),
    }


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Native Richter Impartiality Factorial Assay - 2026-04-24",
        "",
        "Primary scope: learned H_pred state plus ordinary static feedback routes only.",
        "Explicit predicted suppression and divisive suppressors are disabled for all valid rows.",
        "Raw-Ie predicted suppression, if listed, is an invalid engineered positive control.",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Localizer: `{summary['sensory_only_localizer']}`",
        f"Thinning seeds: `{summary['n_thinning_seeds']}`",
        "",
        "## Result Table",
        "",
        "| Condition | Row | Q E | Q U | Q U-E | Activity E | Activity U | Activity U-E | Decoder U-E kp=.02 | Decoder U-E kp=.015 | Pathology |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_row_markdown(row) for row in summary["rows"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Natural dampening found: `{summary['interpretation']['natural_dampening_found']}`",
        "",
        "Rows meeting all natural criteria:",
        "",
    ])
    natural = summary["interpretation"]["natural_dampening_rows"]
    if natural:
        lines.extend(f"- `{name}`" for name in natural)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "This is not a full fMRI forward model; the pseudo-voxel decoder is a",
        "measurement-degraded population readout. Full argmax is reported only as",
        "deterministic debug telemetry, not biological evidence.",
        "",
        "## Artifacts",
        "",
        f"JSON summary: `{summary['summary_json_path']}`",
    ])
    for row in summary["rows"]:
        lines.append(f"- `{row['name']}`: `{row['artifact_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--localizer", type=Path, default=DEFAULT_LOCALIZER)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--n-thinning-seeds", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.stage3_dir / "impartiality_factorial_20260424"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not args.localizer.exists():
        raise FileNotFoundError(f"sensory-only localizer missing: {args.localizer}")

    rows_out: List[Dict[str, Any]] = []
    for row in _rows():
        artifact = out_dir / f"richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_{row.slug}.json"
        log_path = args.log_dir / f"native_richter_factorial_{row.slug}_20260424.log"
        cmd = _native_command(row, args.binary, args.checkpoint, artifact)
        run_info: Dict[str, Any]
        if args.force or not artifact.exists():
            run_info = _run_command(cmd, log_path, args.repo)
            if run_info["returncode"] != 0:
                raise RuntimeError(f"native row failed: {row.name}; log={log_path}")
        else:
            run_info = {
                "cmd": cmd,
                "log_path": str(log_path),
                "returncode": 0,
                "wall_seconds": None,
                "skipped_existing_artifact": True,
            }
        metrics = _extract_metrics(
            artifact,
            args.localizer,
            KEEP_PS,
            args.n_thinning_seeds,
        )
        rows_out.append({
            "condition": row.condition,
            "name": row.name,
            "evidence_status": "valid_learned_static_route",
            "note": row.note,
            "artifact_path": str(artifact),
            "command": cmd,
            "run": run_info,
            "metrics": metrics,
        })

    invalid_controls: Dict[str, Any] = {}
    if DEFAULT_INVALID_RAW_IE.exists():
        invalid_controls["raw_ie_scale16_neighbor05"] = {
            "evidence_status": "invalid_engineered_positive_control",
            "artifact_path": str(DEFAULT_INVALID_RAW_IE),
            "reason": (
                "Explicit predicted raw-Ie suppression hardcodes the dampening "
                "mechanism and is excluded from natural/emergent evidence."
            ),
            "metrics": _extract_metrics(
                DEFAULT_INVALID_RAW_IE,
                args.localizer,
                KEEP_PS,
                args.n_thinning_seeds,
            ),
        }

    summary_json = args.stage3_dir / "stage3_native_richter_impartiality_factorial_20260424.json"
    summary_md = args.stage3_dir / "stage3_native_richter_impartiality_factorial_20260424.md"
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_impartiality_factorial",
        "checkpoint": str(args.checkpoint),
        "sensory_only_localizer": str(args.localizer),
        "stage3_dir": str(args.stage3_dir),
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "engineered_suppressors_disabled_for_valid_rows": {
            "v1_predicted_suppression_scale": 0.0,
            "v1_som_divisive_scale": 0.0,
            "v1_direct_divisive_scale": 0.0,
            "v1_feedforward_divisive_scale": 0.0,
        },
        "rows": rows_out,
        "invalid_engineered_positive_controls": invalid_controls,
    }
    summary["interpretation"] = _interpret(rows_out)
    summary["summary_json_path"] = str(summary_json)
    summary["summary_md_path"] = str(summary_md)

    summary_json.write_text(
        json.dumps(_jsonify(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, summary_md)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    print(f"natural_dampening_found={summary['interpretation']['natural_dampening_found']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
