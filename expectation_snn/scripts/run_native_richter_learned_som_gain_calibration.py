#!/usr/bin/env python3
"""Task-only gain calibration for learned H_pred->V1_SOM feedback.

This driver calibrates the scalar gain on a checkpointed, prediction-only
H_pred->V1_SOM route using validation metrics that do not inspect Richter
expected-vs-unexpected dampening gaps. The selected gain is then frozen for the
Richter assay with all explicit suppressors/divisive mechanisms disabled.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

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
    "stage1_orientation_learned_som_gain_calibration_20260425"
)
VALIDATION_GAINS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
VALIDATION_SEED = 5252
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
    "orientation_learned_som_prior_gain_g2": {
        "q_u_minus_e": -66208.09986,
        "activity_u_minus_e": 10.166667,
        "decoder_u_minus_e_keep_p_0.02": -0.000885,
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
            condition="frozen_orientation_cell_learned_hpred_v1som_gain_calibrated",
            name=self.name,
            feedback_g_total=self.feedback_g_total,
            feedback_r=self.feedback_r,
            feedback_som_center_weight=0.10,
            note=(
                "Frozen original next-orientation/orientation-cell predictor with "
                "prediction-validation-selected learned H_pred->V1_SOM route gain."
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


def _slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return {
        "q_expected_lower": q_ue > 0.0,
        "activity_expected_lower": activity_ue > 0.0,
        "decoder_expected_lower_keep_p_0.02": decoder_ue > 0.0,
        "beats_original_natural_all_three": (
            q_ue > original["q_u_minus_e"]
            and activity_ue > original["activity_u_minus_e"]
            and decoder_ue > original["decoder_u_minus_e_keep_p_0.02"]
        ),
    }


def _validation_command(
    binary: Path,
    checkpoint: Path,
    out: Path,
    gain: float,
) -> List[str]:
    return [
        str(binary),
        "richter-dampening",
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
        "--seed",
        str(VALIDATION_SEED),
        "--reps-expected",
        "1",
        "--reps-unexpected",
        "1",
        "--execution-mode",
        "gpu_only_production",
        "--grating-rate-hz",
        "100",
        "--baseline-rate-hz",
        "0",
        "--v1-stim-sigma-deg",
        "22.0",
        "--feedback-g-total",
        repr(float(gain)),
        "--feedback-r",
        "0.0",
        "--feedback-som-center-weight",
        "0.10",
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
        "--feedback-som-source",
        "learned",
    ]


def _one_hot_cosine(counts: np.ndarray, predicted_channel: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(counts, axis=1)
    pred_vals = counts[np.arange(counts.shape[0]), predicted_channel]
    out = np.zeros(counts.shape[0], dtype=np.float64)
    mask = norms > 0.0
    out[mask] = pred_vals[mask] / norms[mask]
    return out


def _validation_metrics(artifact_path: Path, gain: float) -> Dict[str, Any]:
    artifact = _load_json(artifact_path)
    trial_data = artifact["trial_data"]
    leader_channel = np.asarray(trial_data["leader_channel"], dtype=np.int64)
    predicted_channel = (leader_channel + 2) % 12
    som_counts = np.asarray(
        trial_data["v1_som_trailer_channel_counts"],
        dtype=np.float64,
    )
    v1_counts = np.asarray(
        trial_data["v1_e_trailer_channel_counts"],
        dtype=np.float64,
    )
    held_counts = np.asarray(
        trial_data["hpred_feedback_held_trailer_100ms_channel_counts"],
        dtype=np.float64,
    ).sum(axis=1)
    pred_som = som_counts[np.arange(som_counts.shape[0]), predicted_channel]
    pred_held = held_counts[np.arange(held_counts.shape[0]), predicted_channel]
    som_total = som_counts.sum(axis=1)
    v1_total = v1_counts.sum(axis=1)
    cosine = _one_hot_cosine(som_counts, predicted_channel)
    top1 = np.argmax(som_counts, axis=1)
    nonzero_trials = som_total > 0.0
    # Counts are over a 500 ms trailer window and V1_SOM has 768 cells.
    som_population_rate_hz = float(np.mean(som_total) / (768.0 * 0.5))
    v1_population_rate_hz = float(np.mean(v1_total) / (192.0 * 0.5))
    predicted_count_mean = float(np.mean(pred_som))
    score = 0.0
    if predicted_count_mean > 0.0:
        score = float(np.mean(cosine) * math.log1p(predicted_count_mean))
    eligible = (
        gain > 0.0
        and predicted_count_mean > 0.0
        and som_population_rate_hz <= 50.0
        and v1_population_rate_hz <= 100.0
        and float(np.mean(v1_total)) >= 5.0
    )
    return {
        "gain": gain,
        "artifact_path": str(artifact_path),
        "som_target_template": (
            "one-hot future next-orientation channel derived from "
            "(leader_channel + 2) % 12; expected/unexpected flags ignored"
        ),
        "trial_count": int(som_counts.shape[0]),
        "som_total_count_mean": float(np.mean(som_total)),
        "som_predicted_channel_count_mean": predicted_count_mean,
        "som_nonzero_trial_fraction": float(np.mean(nonzero_trials)),
        "som_target_cosine_mean": float(np.mean(cosine)),
        "som_top1_matches_predicted_fraction": float(np.mean(top1 == predicted_channel)),
        "som_population_rate_hz": som_population_rate_hz,
        "v1_population_rate_hz": v1_population_rate_hz,
        "held_hpred_predicted_channel_count_mean": float(np.mean(pred_held)),
        "selection_score": score,
        "eligible_under_preregistered_rule": eligible,
    }


def _select_gain(validation_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [
        row for row in validation_rows
        if row.get("status") == "PASS"
        and row["metrics"]["eligible_under_preregistered_rule"]
    ]
    if not eligible:
        raise RuntimeError("no validation gain met the preregistered eligibility rule")
    # Maximize task-only SOM prediction score under rate caps; ties choose lower gain.
    selected = max(
        eligible,
        key=lambda row: (
            row["metrics"]["selection_score"],
            -row["metrics"]["gain"],
        ),
    )
    return {
        "selected_gain": selected["metrics"]["gain"],
        "selected_validation_artifact": selected["metrics"]["artifact_path"],
        "selected_validation_log": selected["run"]["log_path"],
        "selected_score": selected["metrics"]["selection_score"],
        "selection_rule": (
            "Before Richter testing, sweep learned SOM route g_total at r=0 on "
            "validation seed 5252 with reps_expected=1/reps_unexpected=1; ignore "
            "expected/unexpected condition labels and all Q/activity/decoder gap "
            "metrics; keep rows with predicted SOM count > 0, V1_SOM population "
            "rate <= 50 Hz, V1_E population rate <= 100 Hz, and mean V1_E trailer "
            "count >= 5; select the eligible gain maximizing "
            "mean(one-hot future-orientation SOM cosine) * log1p(mean predicted "
            "SOM count), tie-breaking toward lower gain."
        ),
        "invalid_selection_metrics": [
            "Q expected/unexpected gap",
            "V1 activity expected/unexpected gap",
            "pseudo-voxel decoder expected/unexpected gap",
            "full argmax expected/unexpected gap",
        ],
    }


def _row_markdown(row: Mapping[str, Any]) -> str:
    native = row["metrics"]["native_metrics"]
    dec = row["metrics"]["decoder"]["noisy_pseudovoxel"]
    return (
        f"| {row['name']} | {row['analysis_role']} | {row['feedback_som_source']} | "
        f"{row['feedback_g_total']:.6g} | {row['feedback_r']:.6g} | "
        f"{native['v1_q_active_fC_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{native['v1_activity_counts_per_trial']['unexpected_minus_expected']:.4f} | "
        f"{dec['0.05']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.02']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.015']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{dec['0.01']['unexpected_minus_expected_accuracy']['mean']:.6f} | "
        f"{row['criteria']['beats_original_natural_all_three']} | "
        f"{','.join(row['metrics']['pathology_flags']) or 'none'} |"
    )


def _validation_markdown(row: Mapping[str, Any]) -> str:
    if row.get("status") != "PASS":
        return (
            f"| {row['gain']:.6g} | FAIL |  |  |  |  |  |  | "
            f"`{row['run']['log_path']}` |"
        )
    m = row["metrics"]
    return (
        f"| {m['gain']:.6g} | PASS | {m['som_predicted_channel_count_mean']:.6f} | "
        f"{m['som_total_count_mean']:.6f} | {m['som_target_cosine_mean']:.6f} | "
        f"{m['som_top1_matches_predicted_fraction']:.6f} | "
        f"{m['som_population_rate_hz']:.6f} | {m['selection_score']:.6f} | "
        f"{m['eligible_under_preregistered_rule']} |"
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Native Richter Orientation Learned-SOM Gain Calibration - 2026-04-25",
        "",
        "Training objective: original next-orientation/orientation-cell prediction",
        "plus checkpointed H_pred->V1_SOM route learned from future-state templates.",
        "The gain is selected only from validation SOM-recruitment/template metrics.",
        "No Q/activity/decoder/Richter expected-unexpected gap is used for training,",
        "checkpoint selection, or gain selection.",
        "",
        f"Checkpoint: `{summary['stage1_checkpoint']}`",
        f"Heldout eval: `{summary['stage1_heldout_eval']}`",
        f"Sensory-only localizer: `{summary['sensory_only_localizer']}`",
        f"Selected gain: `{summary['selection']['selected_gain']}`",
        "",
        "## Selection Rule",
        "",
        summary["selection"]["selection_rule"],
        "",
        "## Validation Gain Table",
        "",
        "| Gain | Status | Pred SOM count | SOM total | SOM cosine | Top1 pred | SOM rate Hz | Score | Eligible |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_validation_markdown(row) for row in summary["validation_rows"])
    lines.extend([
        "",
        "## Frozen Richter Test Rows",
        "",
        "| Row | Role | SOM source | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    lines.extend(_row_markdown(row) for row in summary["rows"])
    lines.extend([
        "",
        "## Comparators",
        "",
        f"Original natural baseline: `{summary['baseline_comparator']['original_natural']}`",
        f"V1-template learned-SOM result: `{summary['baseline_comparator']['v1_template_learned_som']}`",
        f"Prior orientation learned-SOM g=2 result: `{summary['baseline_comparator']['orientation_learned_som_prior_gain_g2']}`",
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
    out_dir = args.stage3_dir / "orientation_learned_som_gain_calibration_20260425"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint_dir / (
        f"stage1_ctx_pred_orientation_cell_learned_som_gaincal_seed{args.seed}_n{args.n_trials}.json"
    )
    heldout_eval = args.checkpoint_dir / (
        f"stage1_ctx_pred_orientation_cell_learned_som_gaincal_seed{args.seed}_n{args.n_trials}_"
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
    train_log = args.log_dir / "native_stage1_orientation_learned_som_gaincal_train_20260425.log"
    if args.force or not checkpoint.exists():
        train_run = _run_command(train_cmd, train_log, args.repo)
        if train_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 orientation learned-SOM gaincal train failed: {train_log}")
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
    heldout_log = args.log_dir / "native_stage1_orientation_learned_som_gaincal_heldout_20260425.log"
    if args.force or not heldout_eval.exists():
        heldout_run = _run_command(heldout_cmd, heldout_log, args.repo)
        if heldout_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 orientation learned-SOM gaincal heldout failed: {heldout_log}")
    else:
        heldout_run = {
            "cmd": heldout_cmd,
            "log_path": str(heldout_log),
            "returncode": 0,
            "wall_seconds": None,
            "skipped_existing_artifact": True,
        }

    validation_rows: List[Dict[str, Any]] = []
    for gain in VALIDATION_GAINS:
        slug = _slug_float(gain)
        artifact = out_dir / (
            "validation_orientation_learned_som_seed5252_reps1_rate100_sigma22_"
            f"g{slug}_r0.json"
        )
        log_path = args.log_dir / (
            f"native_richter_orientation_learned_som_gaincal_validation_g{slug}_20260425.log"
        )
        cmd = _validation_command(args.binary, checkpoint, artifact, gain)
        if args.force or not artifact.exists():
            run = _run_command(cmd, log_path, args.repo)
        else:
            run = {
                "cmd": cmd,
                "log_path": str(log_path),
                "returncode": 0,
                "wall_seconds": None,
                "skipped_existing_artifact": True,
            }
        row: Dict[str, Any] = {
            "gain": gain,
            "artifact_path": str(artifact),
            "run": run,
            "status": "PASS" if run["returncode"] == 0 and artifact.exists() else "FAIL",
        }
        if row["status"] == "PASS":
            row["metrics"] = _validation_metrics(artifact, gain)
        validation_rows.append(row)

    selection = _select_gain(validation_rows)
    selected_gain = float(selection["selected_gain"])
    selected_slug = _slug_float(selected_gain)

    localizer_row = Row(
        condition="sensory_only_localizer",
        name="sensory_only_g0_orientation_learned_som_gaincal_checkpoint",
        feedback_g_total=0.0,
        feedback_r=0.0,
        feedback_som_center_weight=0.10,
        note="Sensory-only localizer for transfer decoder.",
        extra_args=("--feedback-som-source", "disabled"),
    )
    localizer = out_dir / (
        "richter_dampening_orientation_learned_som_gaincal_seed4242_reps4_"
        "rate100_sigma22_sensory_only_g0.json"
    )
    localizer_log = args.log_dir / (
        "native_richter_orientation_learned_som_gaincal_sensory_only_20260425.log"
    )
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
            f"selected_gain_learned_som_only_g{selected_slug}_r0",
            selected_gain,
            0.0,
            "learned",
            "selected_gain_learned_som_only",
        ),
        EvalRow(
            f"selected_gain_learned_som_plus_static_balanced_g{selected_slug}_r03333333333333333",
            selected_gain,
            1.0 / 3.0,
            "learned",
            "selected_gain_learned_som_plus_static_direct_balance",
        ),
        EvalRow(
            f"selected_gain_disabled_control_g{selected_slug}_r0",
            selected_gain,
            0.0,
            "disabled",
            "selected_gain_som_feedback_disabled_control",
        ),
        EvalRow(
            f"selected_gain_shifted_control_g{selected_slug}_r0",
            selected_gain,
            0.0,
            "learned-shifted",
            "selected_gain_wrong_hpred_mapping_control",
        ),
        EvalRow(
            "prior_gain_learned_som_only_g2_r0",
            2.0,
            0.0,
            "learned",
            "prior_orientation_learned_som_gain_baseline",
        ),
    ]

    rows_out: List[Dict[str, Any]] = []
    for eval_row in rows:
        artifact = out_dir / (
            "richter_dampening_orientation_learned_som_gaincal_seed4242_reps4_"
            f"rate100_sigma22_{eval_row.name}.json"
        )
        log_path = args.log_dir / (
            f"native_richter_orientation_learned_som_gaincal_{eval_row.name}_20260425.log"
        )
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
        / "stage3_native_richter_orientation_learned_som_gain_calibration_20260425.json"
    )
    summary_md = (
        args.stage3_dir
        / "stage3_native_richter_orientation_learned_som_gain_calibration_20260425.md"
    )
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_prediction_only_orientation_cell_learned_som_gain_calibration",
        "stage1_checkpoint": str(checkpoint),
        "stage1_heldout_eval": str(heldout_eval),
        "sensory_only_localizer": str(localizer),
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "validation_seed": VALIDATION_SEED,
        "validation_gains": list(VALIDATION_GAINS),
        "selection": selection,
        "validation_rows": validation_rows,
        "baseline_comparator": BASELINE,
        "training_constraints": {
            "ctx_pred_objective": "predict future next-orientation/orientation-cell target from context/leader",
            "feedback_objective": "learn H_pred->V1_SOM future template route from training schedule",
            "uses_q_activity_decoder_dampening_metrics": False,
            "uses_actual_unexpected_trailer_labels": False,
            "checkpoint_selected_by": "predefined seed/n_trials/target only",
            "gain_selected_by": "validation SOM recruitment/template score under rate cap only",
        },
        "checkpoint_metrics": checkpoint_json.get("metrics", {}),
        "checkpoint_task_metrics": checkpoint_json.get("target_prediction_metrics", {}),
        "heldout_metrics": heldout_json.get("metrics", {}),
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
    print(f"selected_gain={selected_gain}")
    for row in validation_rows:
        if row.get("status") == "PASS":
            m = row["metrics"]
            print(
                f"validation gain={m['gain']:g} pred_som={m['som_predicted_channel_count_mean']:.6f} "
                f"cosine={m['som_target_cosine_mean']:.6f} rate={m['som_population_rate_hz']:.6f} "
                f"score={m['selection_score']:.6f} eligible={m['eligible_under_preregistered_rule']}"
            )
        else:
            print(f"validation gain={row['gain']:g} FAILED log={row['run']['log_path']}")
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
