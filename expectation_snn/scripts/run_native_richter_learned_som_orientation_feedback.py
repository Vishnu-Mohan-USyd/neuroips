#!/usr/bin/env python3
"""Prediction-only orientation-cell H_pred->V1_SOM feedback assay.

This is the minimal fair follow-up to the V1-template learned-SOM run: train the
original next-orientation/orientation-cell Stage1 target, add checkpointed
H_pred->V1_SOM weights learned from prediction templates, freeze, and evaluate
only preregistered no-cheat Richter rows.
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
    "stage1_orientation_learned_som_feedback_20260425"
)
BASELINE = {
    "original_natural": {
        "q_u_minus_e": 21062.79,
        "activity_u_minus_e": 22.33,
        "decoder_u_minus_e_keep_p_0.02": 0.00548,
    },
    "v1_template_learned_som": {
        "q_u_minus_e": -145995.230546,
        "activity_u_minus_e": 29.166667,
        "decoder_u_minus_e_keep_p_0.02": 0.006999,
    },
}


@dataclass(frozen=True)
class EvalRow:
    name: str
    feedback_g_total: float
    feedback_r: float
    feedback_som_source: str
    analysis_role: str

    @property
    def row(self) -> Row:
        return Row(
            condition="frozen_orientation_cell_learned_hpred_v1som_feedback",
            name=self.name,
            feedback_g_total=self.feedback_g_total,
            feedback_r=self.feedback_r,
            feedback_som_center_weight=0.10,
            note=(
                "Frozen original next-orientation/orientation-cell predictor "
                "with prediction-only learned H_pred->V1_SOM feedback route."
            ),
            extra_args=("--feedback-som-source", self.feedback_som_source),
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
    q_ue = native["v1_q_active_fC_per_trial"]["unexpected_minus_expected"]
    activity_ue = native["v1_activity_counts_per_trial"]["unexpected_minus_expected"]
    decoder_ue = _decoder_delta(metrics, "0.02")
    original = BASELINE["original_natural"]
    v1_template = BASELINE["v1_template_learned_som"]
    return {
        "q_expected_lower": q_ue > 0.0,
        "activity_expected_lower": activity_ue > 0.0,
        "decoder_expected_lower_keep_p_0.02": decoder_ue > 0.0,
        "beats_original_natural_all_three": (
            q_ue > original["q_u_minus_e"]
            and activity_ue > original["activity_u_minus_e"]
            and decoder_ue > original["decoder_u_minus_e_keep_p_0.02"]
        ),
        "improves_v1_template_learned_som_decoder": (
            decoder_ue > v1_template["decoder_u_minus_e_keep_p_0.02"]
        ),
    }


def _row_markdown(row: Mapping[str, Any]) -> str:
    native = row["metrics"]["native_metrics"]
    dec = row["metrics"]["decoder"]["noisy_pseudovoxel"]
    return (
        f"| {row['name']} | {row['analysis_role']} | {row['feedback_som_source']} | "
        f"{native['v1_q_active_fC_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{native['v1_activity_counts_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{dec['0.05']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.02']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.015']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.01']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{row['criteria']['beats_original_natural_all_three']} | "
        f"{','.join(row['metrics']['pathology_flags']) or 'none'} |"
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Native Richter Orientation-Cell Learned SOM Feedback - 2026-04-25",
        "",
        "Training objective: original next-orientation/orientation-cell prediction",
        "plus a prediction-only H_pred->V1_SOM route. No Q/activity/decoder/Richter",
        "metrics are used for training or checkpoint selection.",
        "",
        f"Checkpoint: `{summary['stage1_checkpoint']}`",
        f"Stage1 heldout eval: `{summary['stage1_heldout_eval']}`",
        f"Sensory-only localizer: `{summary['sensory_only_localizer']}`",
        f"Thinning seeds: `{summary['n_thinning_seeds']}`",
        "",
        "## Frozen Rows",
        "",
        "| Row | Role | SOM source | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(_row_markdown(row) for row in summary["rows"])
    lines.extend([
        "",
        "## Comparators",
        "",
        f"Original natural baseline: `{summary['baseline_comparator']['original_natural']}`",
        f"V1-template learned-SOM result: `{summary['baseline_comparator']['v1_template_learned_som']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    out_dir = args.stage3_dir / "orientation_learned_som_feedback_20260425"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint_dir / (
        f"stage1_ctx_pred_orientation_cell_learned_som_seed{args.seed}_n{args.n_trials}.json"
    )
    heldout_eval = args.checkpoint_dir / (
        f"stage1_ctx_pred_orientation_cell_learned_som_seed{args.seed}_n{args.n_trials}_"
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
        "orientation_cell",
        "--stage1-learn-feedback-som",
        "--out",
        str(checkpoint),
    ]
    train_log = args.log_dir / "native_stage1_orientation_learned_som_train_20260425.log"
    if args.force or not checkpoint.exists():
        train_run = _run_command(train_cmd, train_log, args.repo)
        if train_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 orientation learned-SOM training failed: {train_log}")
    else:
        train_run = {"cmd": train_cmd, "log_path": str(train_log), "returncode": 0, "wall_seconds": None, "skipped_existing_artifact": True}

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
    heldout_log = args.log_dir / "native_stage1_orientation_learned_som_heldout_20260425.log"
    if args.force or not heldout_eval.exists():
        heldout_run = _run_command(heldout_cmd, heldout_log, args.repo)
        if heldout_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 orientation learned-SOM heldout failed: {heldout_log}")
    else:
        heldout_run = {"cmd": heldout_cmd, "log_path": str(heldout_log), "returncode": 0, "wall_seconds": None, "skipped_existing_artifact": True}

    localizer_row = Row(
        condition="sensory_only_localizer",
        name="sensory_only_g0_orientation_learned_som_checkpoint",
        feedback_g_total=0.0,
        feedback_r=0.0,
        feedback_som_center_weight=0.10,
        note="Sensory-only localizer for transfer decoder.",
        extra_args=("--feedback-som-source", "disabled"),
    )
    localizer = out_dir / (
        "richter_dampening_orientation_learned_som_seed4242_reps4_rate100_sigma22_"
        "sensory_only_g0.json"
    )
    localizer_log = args.log_dir / "native_richter_orientation_learned_som_sensory_only_20260425.log"
    localizer_cmd = _native_command(localizer_row, args.binary, checkpoint, localizer)
    if args.force or not localizer.exists():
        localizer_run = _run_command(localizer_cmd, localizer_log, args.repo)
        if localizer_run["returncode"] != 0:
            raise RuntimeError(f"sensory-only localizer failed: {localizer_log}")
    else:
        localizer_run = {"cmd": localizer_cmd, "log_path": str(localizer_log), "returncode": 0, "wall_seconds": None, "skipped_existing_artifact": True}

    rows = [
        EvalRow("orientation_learned_som_only_g2_r0", 2.0, 0.0, "learned", "primary_preregistered_learned_som_only"),
        EvalRow("orientation_learned_som_plus_static_balanced_g2_r03333333333333333", 2.0, 1.0 / 3.0, "learned", "preregistered_learned_som_plus_static_direct_balance"),
        EvalRow("orientation_learned_som_disabled_control_g2_r0", 2.0, 0.0, "disabled", "som_feedback_disabled_control"),
        EvalRow("orientation_learned_som_shifted_control_g2_r0", 2.0, 0.0, "learned-shifted", "wrong_hpred_mapping_control"),
    ]

    rows_out: List[Dict[str, Any]] = []
    for eval_row in rows:
        artifact = out_dir / (
            "richter_dampening_orientation_learned_som_seed4242_reps4_rate100_sigma22_"
            f"{eval_row.name}.json"
        )
        log_path = args.log_dir / f"native_richter_orientation_learned_som_{eval_row.name}_20260425.log"
        cmd = _native_command(eval_row.row, args.binary, checkpoint, artifact)
        if args.force or not artifact.exists():
            run_info = _run_command(cmd, log_path, args.repo)
            if run_info["returncode"] != 0:
                raise RuntimeError(f"frozen Richter eval failed: {log_path}")
        else:
            run_info = {"cmd": cmd, "log_path": str(log_path), "returncode": 0, "wall_seconds": None, "skipped_existing_artifact": True}
        metrics = _extract_metrics(artifact, localizer, KEEP_PS, args.n_thinning_seeds)
        row_out: Dict[str, Any] = {
            "name": eval_row.name,
            "analysis_role": eval_row.analysis_role,
            "feedback_g_total": eval_row.feedback_g_total,
            "feedback_r": eval_row.feedback_r,
            "feedback_som_source": eval_row.feedback_som_source,
            "artifact_path": str(artifact),
            "command": cmd,
            "run": run_info,
            "metrics": metrics,
        }
        row_out["criteria"] = _criteria(metrics)
        rows_out.append(row_out)

    checkpoint_json = _load_json(checkpoint)
    heldout_json = _load_json(heldout_eval)
    summary_json = (
        args.stage3_dir
        / "stage3_native_richter_orientation_learned_som_feedback_20260425.json"
    )
    summary_md = (
        args.stage3_dir
        / "stage3_native_richter_orientation_learned_som_feedback_20260425.md"
    )
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_prediction_only_orientation_cell_learned_hpred_v1som_feedback",
        "stage1_checkpoint": str(checkpoint),
        "stage1_heldout_eval": str(heldout_eval),
        "sensory_only_localizer": str(localizer),
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "baseline_comparator": BASELINE,
        "training_constraints": {
            "ctx_pred_objective": "predict future next-orientation/orientation-cell target from context/leader",
            "feedback_objective": "learn H_pred->V1_SOM future template route from training schedule",
            "uses_q_activity_decoder_dampening_metrics": False,
            "uses_actual_unexpected_trailer_labels": False,
            "checkpoint_selected_by": "predefined seed/n_trials/target only",
        },
        "checkpoint_task_metrics": checkpoint_json.get("target_prediction_metrics", {}),
        "heldout_task_metrics": heldout_json.get("target_prediction_metrics", {}),
        "learned_feedback_manifest": checkpoint_json.get("learned_feedback", {}),
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
            f"decoder_ue_0.02={_decoder_delta(row['metrics'], '0.02'):.6f} "
            f"beats_original_all_three={row['criteria']['beats_original_natural_all_three']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
