#!/usr/bin/env python3
"""Train a frozen next-V1-state Stage1 predictor and evaluate Richter effects.

Training is restricted to the biological prediction task: learn ctx->pred
weights from leader/context events to a future V1_E-like trailer template.
Q/activity/decoder/Richter dampening metrics are never used for training or
model selection. Frozen evaluation runs only preregistered no-cheat feedback
readout rows with explicit suppressors and divisive gates disabled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from run_native_richter_impartiality_factorial import (
    DEFAULT_BINARY,
    DEFAULT_LOG_DIR,
    DEFAULT_STAGE3_DIR,
    KEEP_PS,
    Row,
    _extract_metrics,
    _jsonify,
    _native_command,
)


DEFAULT_CHECKPOINT_DIR = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "expectation_snn/data/checkpoints_native_cpp/"
    "stage1_v1_state_prediction_20260425"
)
BASELINE = {
    "q_u_minus_e": 21062.79,
    "activity_u_minus_e": 22.33,
    "decoder_u_minus_e_keep_p_0.02": 0.00548,
}


@dataclass(frozen=True)
class EvalRow:
    name: str
    feedback_g_total: float
    feedback_r: float
    analysis_role: str

    @property
    def row(self) -> Row:
        return Row(
            condition="frozen_v1_state_prediction_checkpoint",
            name=self.name,
            feedback_g_total=self.feedback_g_total,
            feedback_r=self.feedback_r,
            feedback_som_center_weight=0.10,
            note=(
                "Frozen next-V1-state predictor; ordinary direct/SOM feedback "
                "readout; no explicit predicted suppression or divisive gates."
            ),
        )


def _run_command(cmd: Sequence[str], log_path: Path, cwd: Path) -> Dict[str, Any]:
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
    return {
        "cmd": list(cmd),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "wall_seconds": time.monotonic() - t0,
    }


def _decoder_delta(metrics: Mapping[str, Any], keep_p: str) -> float:
    return float(
        metrics["decoder"]["noisy_pseudovoxel"][keep_p][
            "unexpected_minus_expected_accuracy"
        ]["mean"]
    )


def _criteria(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    native = metrics["native_metrics"]
    return {
        "q_expected_lower": (
            native["v1_q_active_fC_per_trial"]["unexpected_minus_expected"] > 0.0
        ),
        "activity_expected_lower": (
            native["v1_activity_counts_per_trial"]["unexpected_minus_expected"] > 0.0
        ),
        "decoder_expected_lower_keep_p_0.02": _decoder_delta(metrics, "0.02") > 0.0,
        "beats_previous_baseline_all_three": (
            native["v1_q_active_fC_per_trial"]["unexpected_minus_expected"]
            > BASELINE["q_u_minus_e"]
            and native["v1_activity_counts_per_trial"]["unexpected_minus_expected"]
            > BASELINE["activity_u_minus_e"]
            and _decoder_delta(metrics, "0.02")
            > BASELINE["decoder_u_minus_e_keep_p_0.02"]
        ),
    }


def _row_markdown(row: Mapping[str, Any]) -> str:
    native = row["metrics"]["native_metrics"]
    dec = row["metrics"]["decoder"]["noisy_pseudovoxel"]
    return (
        f"| {row['name']} | {row['analysis_role']} | "
        f"{native['v1_q_active_fC_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{native['v1_activity_counts_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{dec['0.05']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.02']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.015']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.01']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{row['criteria']['beats_previous_baseline_all_three']} | "
        f"{','.join(row['metrics']['pathology_flags']) or 'none'} |"
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Native Richter Task-Trained Next-V1-State Predictor - 2026-04-25",
        "",
        "Training objective: predict the future lower-level V1_E trailer template",
        "from context/leader only. No Q/activity/decoder/Richter dampening metric",
        "is used in training or checkpoint selection.",
        "",
        f"Checkpoint: `{summary['stage1_checkpoint']}`",
        f"Stage1 heldout eval: `{summary['stage1_heldout_eval']}`",
        f"Sensory-only localizer: `{summary['sensory_only_localizer']}`",
        f"Thinning seeds: `{summary['n_thinning_seeds']}`",
        "",
        "## Frozen Evaluation Rows",
        "",
        "| Row | Role | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats previous all three | Pathology |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(_row_markdown(row) for row in summary["rows"])
    lines.extend([
        "",
        "## Constraints",
        "",
        "- Explicit predicted-channel current scaling is disabled.",
        "- V1 divisive suppressors are disabled.",
        "- The frozen readout rows are preregistered; they are not selected by dampening metrics.",
        "- The sigma22 sensory-only localizer is used only for the post-hoc pseudo-voxel transfer decoder.",
        "",
        "## Baseline Comparator",
        "",
        f"Previous no-cheat baseline Q U-E: `{BASELINE['q_u_minus_e']}`",
        f"Previous no-cheat baseline activity U-E: `{BASELINE['activity_u_minus_e']}`",
        f"Previous no-cheat baseline decoder U-E @ keep_p=.02: `{BASELINE['decoder_u_minus_e_keep_p_0.02']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=72)
    parser.add_argument("--heldout-seed", type=int, default=4243)
    parser.add_argument("--n-thinning-seeds", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.stage3_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.stage3_dir / "task_prediction_v1_state_20260425"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint_dir / (
        f"stage1_ctx_pred_v1_template_seed{args.seed}_n{args.n_trials}.json"
    )
    heldout_eval = args.checkpoint_dir / (
        f"stage1_ctx_pred_v1_template_seed{args.seed}_n{args.n_trials}_"
        f"heldout_seed{args.heldout_seed}.json"
    )

    train_cmd = [
        str(args.binary),
        "stage1-train",
        "--fixture",
        "generated",
        "--seed",
        str(args.seed),
        "--n-trials",
        str(args.n_trials),
        "--stage1-prediction-target",
        "v1_template",
        "--out",
        str(checkpoint),
    ]
    train_log = args.log_dir / "native_stage1_v1_template_train_20260425.log"
    if args.force or not checkpoint.exists():
        train_run = _run_command(train_cmd, train_log, args.repo)
        if train_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 v1_template training failed: {train_log}")
    else:
        train_run = {
            "cmd": train_cmd,
            "log_path": str(train_log),
            "returncode": 0,
            "wall_seconds": None,
            "skipped_existing_artifact": True,
        }

    heldout_cmd = [
        str(args.binary),
        "stage1-heldout-eval",
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(args.heldout_seed),
        "--heldout-schedule",
        "generated",
        "--out",
        str(heldout_eval),
    ]
    heldout_log = args.log_dir / "native_stage1_v1_template_heldout_20260425.log"
    if args.force or not heldout_eval.exists():
        heldout_run = _run_command(heldout_cmd, heldout_log, args.repo)
        if heldout_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 v1_template heldout eval failed: {heldout_log}")
    else:
        heldout_run = {
            "cmd": heldout_cmd,
            "log_path": str(heldout_log),
            "returncode": 0,
            "wall_seconds": None,
            "skipped_existing_artifact": True,
        }

    localizer_row = Row(
        condition="sensory_only_localizer",
        name="sensory_only_g0_r03333333333333333",
        feedback_g_total=0.0,
        feedback_r=0.3333333333333333,
        feedback_som_center_weight=0.10,
        note="Frozen checkpoint sensory-only localizer for decoder training.",
    )
    localizer = out_dir / (
        "richter_dampening_stage1_v1template_n72_seed4242_reps4_"
        "rate100_sigma22_sensory_only_g0.json"
    )
    localizer_log = args.log_dir / "native_richter_v1template_sensory_only_20260425.log"
    localizer_cmd = _native_command(localizer_row, args.binary, checkpoint, localizer)
    if args.force or not localizer.exists():
        localizer_run = _run_command(localizer_cmd, localizer_log, args.repo)
        if localizer_run["returncode"] != 0:
            raise RuntimeError(f"sensory-only localizer failed: {localizer_log}")
    else:
        localizer_run = {
            "cmd": localizer_cmd,
            "log_path": str(localizer_log),
            "returncode": 0,
            "wall_seconds": None,
            "skipped_existing_artifact": True,
        }

    rows = [
        EvalRow(
            "balanced_g2_r03333333333333333",
            2.0,
            0.3333333333333333,
            "minimum_required_preregistered_supported_balance",
        ),
        EvalRow(
            "analysis_only_g2p5_r0p5",
            2.5,
            0.5,
            "analysis_only_preregistered_sensitivity_row",
        ),
    ]
    rows_out: List[Dict[str, Any]] = []
    for eval_row in rows:
        artifact = out_dir / (
            "richter_dampening_stage1_v1template_n72_seed4242_reps4_"
            f"rate100_sigma22_{eval_row.name}.json"
        )
        log_path = args.log_dir / f"native_richter_v1template_{eval_row.name}_20260425.log"
        cmd = _native_command(eval_row.row, args.binary, checkpoint, artifact)
        if args.force or not artifact.exists():
            run_info = _run_command(cmd, log_path, args.repo)
            if run_info["returncode"] != 0:
                raise RuntimeError(f"frozen Richter eval failed: {log_path}")
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
            localizer,
            KEEP_PS,
            args.n_thinning_seeds,
        )
        row_out: Dict[str, Any] = {
            "name": eval_row.name,
            "analysis_role": eval_row.analysis_role,
            "feedback_g_total": eval_row.feedback_g_total,
            "feedback_r": eval_row.feedback_r,
            "artifact_path": str(artifact),
            "command": cmd,
            "run": run_info,
            "metrics": metrics,
        }
        row_out["criteria"] = _criteria(metrics)
        rows_out.append(row_out)

    summary_json = (
        args.stage3_dir
        / "stage3_native_richter_task_prediction_v1_state_20260425.json"
    )
    summary_md = (
        args.stage3_dir
        / "stage3_native_richter_task_prediction_v1_state_20260425.md"
    )
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_task_trained_next_v1_state_prediction",
        "stage1_checkpoint": str(checkpoint),
        "stage1_checkpoint_dir": str(args.checkpoint_dir),
        "stage1_heldout_eval": str(heldout_eval),
        "sensory_only_localizer": str(localizer),
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "baseline_comparator": BASELINE,
        "training_constraints": {
            "objective": "predict future lower-level V1_E trailer template from context/leader",
            "uses_q_activity_decoder_dampening_metrics": False,
            "uses_actual_unexpected_trailer_labels": False,
            "checkpoint_selected_by": "predefined seed/n_trials/target only",
        },
        "engineered_suppressors_disabled_for_eval": {
            "v1_predicted_suppression_scale": 0.0,
            "v1_som_divisive_scale": 0.0,
            "v1_direct_divisive_scale": 0.0,
            "v1_feedforward_divisive_scale": 0.0,
        },
        "runs": {
            "stage1_train": train_run,
            "stage1_heldout_eval": heldout_run,
            "sensory_only_localizer": localizer_run,
        },
        "rows": rows_out,
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
    for row in rows_out:
        print(
            f"{row['name']}: q_ue="
            f"{row['metrics']['native_metrics']['v1_q_active_fC_per_trial']['unexpected_minus_expected']:.6f} "
            f"activity_ue="
            f"{row['metrics']['native_metrics']['v1_activity_counts_per_trial']['unexpected_minus_expected']:.6f} "
            f"decoder_ue_0.02={_decoder_delta(row['metrics'], '0.02'):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
