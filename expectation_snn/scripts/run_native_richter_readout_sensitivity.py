#!/usr/bin/env python3
"""Frozen-task readout sensitivity sweep for native Richter assay.

This is not a training objective and does not optimize model weights against
dampening, activity, Q, or decoder metrics. It evaluates a frozen Stage1
next-orientation task checkpoint under ordinary static feedback-route readout
parameters only. Explicit predicted suppression and divisive suppressors remain
disabled for every row.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from run_native_richter_impartiality_factorial import (
    DEFAULT_BINARY,
    DEFAULT_CHECKPOINT,
    DEFAULT_LOCALIZER,
    DEFAULT_LOG_DIR,
    DEFAULT_STAGE3_DIR,
    KEEP_PS,
    Row,
    _extract_metrics,
    _jsonify,
    _native_command,
    _run_command,
)


G_TOTAL_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
R_GRID = (0.25, 0.28, 0.30, 1.0 / 3.0, 0.36, 0.40, 0.45, 0.50)


def _slug_float(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def _rows() -> List[Row]:
    rows: List[Row] = []
    for g_total in G_TOTAL_GRID:
        for r in R_GRID:
            rows.append(
                Row(
                    condition="frozen_task_static_feedback_readout_sensitivity",
                    name=f"gtotal{_slug_float(g_total)}_r{_slug_float(r)}",
                    feedback_g_total=float(g_total),
                    feedback_r=float(r),
                    feedback_som_center_weight=0.10,
                    note=(
                        "Frozen Stage1 next-orientation checkpoint; ordinary "
                        "direct/SOM feedback readout only; no engineered "
                        "predicted suppression or divisive suppressors."
                    ),
                )
            )
    return rows


def _criteria(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    native = metrics["native_metrics"]
    decoder = metrics["decoder"]["noisy_pseudovoxel"]
    q_ue = native["v1_q_active_fC_per_trial"]["unexpected_minus_expected"]
    activity_ue = native["v1_activity_counts_per_trial"]["unexpected_minus_expected"]
    decoder_by_keep = {
        keep_p: decoder[str(float(keep_p))][
            "unexpected_minus_expected_accuracy"
        ]["mean"]
        for keep_p in KEEP_PS
    }
    decoder_ci_low_by_keep = {
        keep_p: decoder[str(float(keep_p))][
            "unexpected_minus_expected_accuracy"
        ].get("bootstrap_mean_ci_95", {}).get("low")
        for keep_p in KEEP_PS
    }
    return {
        "q_expected_lower": bool(q_ue > 0.0),
        "activity_expected_lower": bool(activity_ue > 0.0),
        "decoder_expected_lower_by_keep_p": {
            str(float(k)): bool(v > 0.0)
            for k, v in decoder_by_keep.items()
        },
        "decoder_ci_low_positive_by_keep_p": {
            str(float(k)): bool(v is not None and v > 0.0)
            for k, v in decoder_ci_low_by_keep.items()
        },
        "all_directional": bool(
            q_ue > 0.0
            and activity_ue > 0.0
            and all(v > 0.0 for v in decoder_by_keep.values())
        ),
        "all_decoder_ci_low_positive": bool(
            all(v is not None and v > 0.0 for v in decoder_ci_low_by_keep.values())
        ),
    }


def _rank_key(row: Mapping[str, Any]) -> Sequence[float]:
    metrics = row["metrics"]
    native = metrics["native_metrics"]
    decoder = metrics["decoder"]["noisy_pseudovoxel"]
    return (
        float(native["v1_q_active_fC_per_trial"]["unexpected_minus_expected"]),
        float(native["v1_activity_counts_per_trial"]["unexpected_minus_expected"]),
        float(decoder["0.02"]["unexpected_minus_expected_accuracy"]["mean"]),
        float(decoder["0.015"]["unexpected_minus_expected_accuracy"]["mean"]),
    )


def _markdown_row(row: Mapping[str, Any]) -> str:
    native = row["metrics"]["native_metrics"]
    decoder = row["metrics"]["decoder"]["noisy_pseudovoxel"]
    return (
        f"| {row['name']} | {row['feedback_g_total']} | {row['feedback_r']} | "
        f"{native['v1_q_active_fC_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{native['v1_activity_counts_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{decoder['0.05']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{decoder['0.02']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{decoder['0.015']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{decoder['0.01']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{row['criteria']['all_directional']} | "
        f"{','.join(row['metrics']['pathology_flags']) or 'none'} |"
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    ranked = summary["ranked_rows"]
    lines = [
        "# Native Richter Frozen-Task Static-Route Readout Sensitivity - 2026-04-24",
        "",
        "This sweep uses a frozen Stage1 next-orientation task-trained checkpoint.",
        "It does not train on Q/activity/decoder metrics and does not use explicit",
        "predicted-channel suppression, divisive suppressors, or label leakage.",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Localizer: `{summary['sensory_only_localizer']}`",
        f"Rows: `{len(summary['rows'])}`",
        f"Thinning seeds per row: `{summary['n_thinning_seeds']}`",
        "",
        "## Ranked Rows",
        "",
        "| Row | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | All directional | Pathology |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(_markdown_row(row) for row in ranked)
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Natural/static-route all-directional rows: `{summary['interpretation']['all_directional_count']}`",
        f"Rows with positive decoder CI lower bound at all keep_p: `{summary['interpretation']['all_decoder_ci_low_positive_count']}`",
        "",
        "This is a circuit readout sensitivity assay, not a training-selection loop.",
        "Any promising setting must still be replicated across seeds/checkpoints",
        "before being treated as robust emergent dampening.",
    ])
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

    out_dir = args.stage3_dir / "readout_sensitivity_20260424"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict[str, Any]] = []
    for row in _rows():
        artifact = out_dir / (
            "richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_"
            f"{row.slug}.json"
        )
        log_path = args.log_dir / f"native_richter_readout_{row.slug}_20260424.log"
        cmd = _native_command(row, args.binary, args.checkpoint, artifact)
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
        row_out: Dict[str, Any] = {
            "condition": row.condition,
            "name": row.name,
            "feedback_g_total": row.feedback_g_total,
            "feedback_r": row.feedback_r,
            "feedback_som_center_weight": row.feedback_som_center_weight,
            "artifact_path": str(artifact),
            "command": cmd,
            "run": run_info,
            "metrics": metrics,
        }
        row_out["criteria"] = _criteria(metrics)
        rows_out.append(row_out)

    ranked = sorted(rows_out, key=_rank_key, reverse=True)
    all_directional = [r for r in rows_out if r["criteria"]["all_directional"]]
    all_ci = [
        r for r in rows_out
        if r["criteria"]["all_decoder_ci_low_positive"]
    ]
    summary_json = (
        args.stage3_dir
        / "stage3_native_richter_readout_sensitivity_20260424.json"
    )
    summary_md = (
        args.stage3_dir
        / "stage3_native_richter_readout_sensitivity_20260424.md"
    )
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_frozen_task_static_route_readout_sensitivity",
        "checkpoint": str(args.checkpoint),
        "sensory_only_localizer": str(args.localizer),
        "stage3_dir": str(args.stage3_dir),
        "grid": {
            "g_total": list(G_TOTAL_GRID),
            "r": list(R_GRID),
            "feedback_som_center_weight": 0.10,
        },
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "constraints": {
            "training": "frozen Stage1 next-orientation task checkpoint only",
            "no_dampening_metric_training": True,
            "no_explicit_predicted_suppression": True,
            "no_divisive_suppressors": True,
            "no_label_leakage": True,
        },
        "rows": rows_out,
        "ranked_rows": ranked,
        "interpretation": {
            "all_directional_count": len(all_directional),
            "all_directional_rows": [r["name"] for r in all_directional],
            "all_decoder_ci_low_positive_count": len(all_ci),
            "all_decoder_ci_low_positive_rows": [r["name"] for r in all_ci],
        },
        "summary_json_path": str(summary_json),
        "summary_md_path": str(summary_md),
    }
    summary_json.write_text(
        json.dumps(_jsonify(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, summary_md)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    print(f"all_directional_count={len(all_directional)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
