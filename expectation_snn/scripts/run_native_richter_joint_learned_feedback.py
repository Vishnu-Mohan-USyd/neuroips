#!/usr/bin/env python3
"""Prediction-only joint learned direct+SOM feedback assay.

The native checkpoint learns H_pred->V1_E direct/apical and H_pred->V1_SOM
routes from future lower-level templates only. A held-out validation sweep
selects route gain/balance from template-recruitment metrics under rate caps;
Richter expected/unexpected Q, activity, and decoder gaps are not used until the
final frozen assay.
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
    "stage1_orientation_joint_learned_feedback_20260425"
)
VALIDATION_SEED = 5353
VALIDATION_GRID = (
    (8.0, 0.0),
    (8.0, 1.0 / 3.0),
    (8.0, 1.0),
    (8.0, 3.0),
    (8.0, 1000.0),
    (16.0, 0.0),
    (16.0, 1.0 / 3.0),
    (16.0, 1.0),
    (16.0, 3.0),
    (16.0, 1000.0),
    (32.0, 0.0),
    (32.0, 1.0 / 3.0),
    (32.0, 1.0),
    (32.0, 3.0),
    (32.0, 1000.0),
    (64.0, 0.0),
    (64.0, 1.0 / 3.0),
    (64.0, 1.0),
    (64.0, 3.0),
    (64.0, 1000.0),
    (128.0, 0.0),
    (128.0, 1.0 / 3.0),
    (128.0, 1.0),
    (128.0, 3.0),
    (128.0, 1000.0),
)
BASELINE = {
    "original_natural": {
        "q_u_minus_e": 21062.79,
        "activity_u_minus_e": 22.33,
        "decoder_u_minus_e_keep_p_0.02": 0.00548,
    },
    "orientation_learned_som_gaincal_g64": {
        "q_u_minus_e": -2048606.843238,
        "activity_u_minus_e": 133.833333,
        "decoder_u_minus_e_keep_p_0.02": 0.099080,
    },
}


@dataclass(frozen=True)
class EvalRow:
    name: str
    feedback_g_total: float
    feedback_r: float
    feedback_direct_source: str
    feedback_som_source: str
    analysis_role: str

    @property
    def row(self) -> Row:
        return Row(
            condition="frozen_orientation_cell_joint_learned_feedback",
            name=self.name,
            feedback_g_total=self.feedback_g_total,
            feedback_r=self.feedback_r,
            feedback_som_center_weight=0.10,
            note=(
                "Frozen original next-orientation/orientation-cell predictor with "
                "prediction-validation-selected learned H_pred->V1_E direct and "
                "H_pred->V1_SOM routes."
            ),
            extra_args=(
                "--feedback-direct-source",
                self.feedback_direct_source,
                "--feedback-som-source",
                self.feedback_som_source,
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


def _slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_balance(g_total: float, r: float) -> Dict[str, float]:
    if math.isinf(r):
        return {"g_direct": g_total, "g_som": 0.0}
    g_som = g_total / (1.0 + r)
    return {"g_direct": g_total - g_som, "g_som": g_som}


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
    g_total: float,
    r: float,
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
        repr(float(g_total)),
        "--feedback-r",
        repr(float(r)),
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
        "--feedback-direct-source",
        "learned",
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


def _validation_metrics(
    artifact_path: Path,
    g_total: float,
    r: float,
) -> Dict[str, Any]:
    artifact = _load_json(artifact_path)
    trial_data = artifact["trial_data"]
    leader_channel = np.asarray(trial_data["leader_channel"], dtype=np.int64)
    predicted_channel = (leader_channel + 2) % 12
    v1_counts = np.asarray(
        trial_data["v1_e_trailer_channel_counts"],
        dtype=np.float64,
    )
    som_counts = np.asarray(
        trial_data["v1_som_trailer_channel_counts"],
        dtype=np.float64,
    )
    held_counts = np.asarray(
        trial_data["hpred_feedback_held_trailer_100ms_channel_counts"],
        dtype=np.float64,
    ).sum(axis=1)
    pred_v1 = v1_counts[np.arange(v1_counts.shape[0]), predicted_channel]
    pred_som = som_counts[np.arange(som_counts.shape[0]), predicted_channel]
    pred_held = held_counts[np.arange(held_counts.shape[0]), predicted_channel]
    v1_total = v1_counts.sum(axis=1)
    som_total = som_counts.sum(axis=1)
    v1_cosine = _one_hot_cosine(v1_counts, predicted_channel)
    som_cosine = _one_hot_cosine(som_counts, predicted_channel)
    balance = _resolved_balance(g_total, r)
    v1_population_rate_hz = float(np.mean(v1_total) / (192.0 * 0.5))
    som_population_rate_hz = float(np.mean(som_total) / (768.0 * 0.5))
    v1_pred_mean = float(np.mean(pred_v1))
    som_pred_mean = float(np.mean(pred_som))
    score = (
        float(np.mean(v1_cosine)) * math.log1p(max(v1_pred_mean, 0.0))
        + float(np.mean(som_cosine)) * math.log1p(max(som_pred_mean, 0.0))
    )
    eligible = (
        g_total > 0.0
        and (v1_pred_mean > 0.0 or som_pred_mean > 0.0)
        and v1_population_rate_hz <= 100.0
        and som_population_rate_hz <= 50.0
        and float(np.mean(v1_total)) >= 5.0
    )
    return {
        "g_total": g_total,
        "r": r,
        "resolved_g_direct": balance["g_direct"],
        "resolved_g_som": balance["g_som"],
        "artifact_path": str(artifact_path),
        "target_template": (
            "future next-orientation channel derived from (leader_channel + 2) % 12; "
            "expected/unexpected gap metrics ignored"
        ),
        "trial_count": int(v1_counts.shape[0]),
        "v1_total_count_mean": float(np.mean(v1_total)),
        "v1_predicted_channel_count_mean": v1_pred_mean,
        "v1_target_cosine_mean": float(np.mean(v1_cosine)),
        "v1_top1_matches_predicted_fraction": float(
            np.mean(np.argmax(v1_counts, axis=1) == predicted_channel)
        ),
        "v1_population_rate_hz": v1_population_rate_hz,
        "som_total_count_mean": float(np.mean(som_total)),
        "som_predicted_channel_count_mean": som_pred_mean,
        "som_target_cosine_mean": float(np.mean(som_cosine)),
        "som_top1_matches_predicted_fraction": float(
            np.mean(np.argmax(som_counts, axis=1) == predicted_channel)
        ),
        "som_population_rate_hz": som_population_rate_hz,
        "held_hpred_predicted_channel_count_mean": float(np.mean(pred_held)),
        "selection_score": score,
        "eligible_under_preregistered_rule": eligible,
    }


def _select_route(validation_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [
        row for row in validation_rows
        if row.get("status") == "PASS"
        and row["metrics"]["eligible_under_preregistered_rule"]
    ]
    if not eligible:
        raise RuntimeError("no validation row met the preregistered eligibility rule")
    selected = max(
        eligible,
        key=lambda row: (
            row["metrics"]["selection_score"],
            -row["metrics"]["g_total"],
        ),
    )
    m = selected["metrics"]
    return {
        "selected_g_total": m["g_total"],
        "selected_r": m["r"],
        "selected_resolved_g_direct": m["resolved_g_direct"],
        "selected_resolved_g_som": m["resolved_g_som"],
        "selected_validation_artifact": m["artifact_path"],
        "selected_validation_log": selected["run"]["log_path"],
        "selected_score": m["selection_score"],
        "selection_rule": (
            "Before Richter testing, sweep joint learned direct+SOM g_total/r on "
            "validation seed 5353 with reps_expected=1/reps_unexpected=1; ignore "
            "expected/unexpected condition labels for scoring and do not inspect "
            "Q/activity/decoder gap metrics; keep rows with nonzero predicted "
            "V1 or SOM recruitment, V1_E population rate <= 100 Hz, V1_SOM "
            "population rate <= 50 Hz, and mean V1_E trailer count >= 5; select "
            "the eligible row maximizing V1 one-hot future-orientation cosine * "
            "log1p(predicted V1 count) plus SOM one-hot future-orientation cosine "
            "* log1p(predicted SOM count), tie-breaking toward lower total gain."
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
        f"| {row['name']} | {row['analysis_role']} | "
        f"{row['feedback_direct_source']} | {row['feedback_som_source']} | "
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
            f"| {row['g_total']:.6g} | {row['r']:.6g} | FAIL |  |  |  |  |  |  |  | |"
        )
    m = row["metrics"]
    return (
        f"| {m['g_total']:.6g} | {m['r']:.6g} | PASS | "
        f"{m['resolved_g_direct']:.6g} | {m['resolved_g_som']:.6g} | "
        f"{m['v1_predicted_channel_count_mean']:.6f} | {m['v1_target_cosine_mean']:.6f} | "
        f"{m['som_predicted_channel_count_mean']:.6f} | {m['som_target_cosine_mean']:.6f} | "
        f"{m['v1_population_rate_hz']:.6f} | {m['som_population_rate_hz']:.6f} | "
        f"{m['selection_score']:.6f} | {m['eligible_under_preregistered_rule']} |"
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Native Richter Joint Learned Direct+SOM Feedback - 2026-04-25",
        "",
        "Training objective: original next-orientation/orientation-cell prediction plus",
        "checkpointed H_pred->V1_E direct/apical and H_pred->V1_SOM routes learned",
        "from future-state templates. Gain/balance selection uses validation",
        "template-recruitment metrics only; no Q/activity/decoder/Richter gap metric",
        "is used before the frozen test.",
        "",
        f"Checkpoint: `{summary['stage1_checkpoint']}`",
        f"Heldout eval: `{summary['stage1_heldout_eval']}`",
        f"Sensory-only localizer: `{summary['sensory_only_localizer']}`",
        f"Selected g_total/r: `{summary['selection']['selected_g_total']}` / `{summary['selection']['selected_r']}`",
        "",
        "## Selection Rule",
        "",
        summary["selection"]["selection_rule"],
        "",
        "## Validation Grid",
        "",
        "| g_total | r | Status | g_direct | g_som | V1 pred | V1 cos | SOM pred | SOM cos | V1 Hz | SOM Hz | Score | Eligible |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_validation_markdown(row) for row in summary["validation_rows"])
    lines.extend([
        "",
        "## Frozen Richter Test Rows",
        "",
        "| Row | Role | Direct source | SOM source | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    lines.extend(_row_markdown(row) for row in summary["rows"])
    lines.extend([
        "",
        "## Comparators",
        "",
        f"Original natural baseline: `{summary['baseline_comparator']['original_natural']}`",
        f"Orientation learned-SOM gaincal g64: `{summary['baseline_comparator']['orientation_learned_som_gaincal_g64']}`",
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
    out_dir = args.stage3_dir / "orientation_joint_learned_feedback_20260425"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint_dir / (
        f"stage1_ctx_pred_orientation_cell_joint_learned_feedback_seed{args.seed}_n{args.n_trials}.json"
    )
    heldout_eval = args.checkpoint_dir / (
        f"stage1_ctx_pred_orientation_cell_joint_learned_feedback_seed{args.seed}_n{args.n_trials}_"
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
        "--stage1-learn-feedback-direct",
        "--stage1-learn-feedback-som",
        "--out",
        str(checkpoint),
    ]
    train_log = args.log_dir / "native_stage1_orientation_joint_learned_train_20260425.log"
    if args.force or not checkpoint.exists():
        train_run = _run_command(train_cmd, train_log, args.repo)
        if train_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 joint learned feedback train failed: {train_log}")
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
    heldout_log = args.log_dir / "native_stage1_orientation_joint_learned_heldout_20260425.log"
    if args.force or not heldout_eval.exists():
        heldout_run = _run_command(heldout_cmd, heldout_log, args.repo)
        if heldout_run["returncode"] != 0:
            raise RuntimeError(f"Stage1 joint learned feedback heldout failed: {heldout_log}")
    else:
        heldout_run = {
            "cmd": heldout_cmd,
            "log_path": str(heldout_log),
            "returncode": 0,
            "wall_seconds": None,
            "skipped_existing_artifact": True,
        }

    validation_rows: List[Dict[str, Any]] = []
    for g_total, r in VALIDATION_GRID:
        slug = f"g{_slug_float(g_total)}_r{_slug_float(r)}"
        artifact = out_dir / (
            "validation_orientation_joint_learned_seed5353_reps1_rate100_sigma22_"
            f"{slug}.json"
        )
        log_path = args.log_dir / (
            f"native_richter_orientation_joint_learned_validation_{slug}_20260425.log"
        )
        cmd = _validation_command(args.binary, checkpoint, artifact, g_total, r)
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
            "g_total": g_total,
            "r": r,
            "artifact_path": str(artifact),
            "run": run,
            "status": "PASS" if run["returncode"] == 0 and artifact.exists() else "FAIL",
        }
        if row["status"] == "PASS":
            row["metrics"] = _validation_metrics(artifact, g_total, r)
        validation_rows.append(row)

    selection = _select_route(validation_rows)
    selected_g = float(selection["selected_g_total"])
    selected_r = float(selection["selected_r"])
    selected_direct = float(selection["selected_resolved_g_direct"])
    selected_som = float(selection["selected_resolved_g_som"])
    selected_slug = f"g{_slug_float(selected_g)}_r{_slug_float(selected_r)}"

    localizer_row = Row(
        condition="sensory_only_localizer",
        name="sensory_only_g0_orientation_joint_learned_checkpoint",
        feedback_g_total=0.0,
        feedback_r=0.0,
        feedback_som_center_weight=0.10,
        note="Sensory-only localizer for transfer decoder.",
        extra_args=(
            "--feedback-direct-source",
            "disabled",
            "--feedback-som-source",
            "disabled",
        ),
    )
    localizer = out_dir / (
        "richter_dampening_orientation_joint_learned_seed4242_reps4_"
        "rate100_sigma22_sensory_only_g0.json"
    )
    localizer_log = args.log_dir / (
        "native_richter_orientation_joint_learned_sensory_only_20260425.log"
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

    direct_only_g = selected_direct * (1001.0 / 1000.0) if selected_direct > 0 else 0.0
    rows = [
        EvalRow(
            f"selected_joint_learned_{selected_slug}",
            selected_g,
            selected_r,
            "learned",
            "learned",
            "selected_joint_learned_direct_plus_som",
        ),
        EvalRow(
            f"selected_disabled_control_{selected_slug}",
            selected_g,
            selected_r,
            "disabled",
            "disabled",
            "selected_learned_routes_disabled_control",
        ),
        EvalRow(
            f"selected_shifted_control_{selected_slug}",
            selected_g,
            selected_r,
            "learned-shifted",
            "learned-shifted",
            "selected_wrong_hpred_mapping_control",
        ),
        EvalRow(
            f"selected_som_only_learned_g{_slug_float(selected_som)}_r0",
            selected_som,
            0.0,
            "disabled",
            "learned",
            "component_control_som_only_learned",
        ),
        EvalRow(
            f"selected_direct_only_learned_g{_slug_float(direct_only_g)}_r1000",
            direct_only_g,
            1000.0,
            "learned",
            "disabled",
            "component_control_direct_only_learned",
        ),
    ]

    rows_out: List[Dict[str, Any]] = []
    for eval_row in rows:
        artifact = out_dir / (
            "richter_dampening_orientation_joint_learned_seed4242_reps4_"
            f"rate100_sigma22_{eval_row.name}.json"
        )
        log_path = args.log_dir / (
            f"native_richter_orientation_joint_learned_{eval_row.name}_20260425.log"
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
            "feedback_direct_source": eval_row.feedback_direct_source,
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
        / "stage3_native_richter_orientation_joint_learned_feedback_20260425.json"
    )
    summary_md = (
        args.stage3_dir
        / "stage3_native_richter_orientation_joint_learned_feedback_20260425.md"
    )
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_prediction_only_orientation_cell_joint_learned_direct_som_feedback",
        "stage1_checkpoint": str(checkpoint),
        "stage1_heldout_eval": str(heldout_eval),
        "sensory_only_localizer": str(localizer),
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "validation_seed": VALIDATION_SEED,
        "validation_grid": list(VALIDATION_GRID),
        "selection": selection,
        "validation_rows": validation_rows,
        "baseline_comparator": BASELINE,
        "training_constraints": {
            "ctx_pred_objective": "predict future next-orientation/orientation-cell target from context/leader",
            "direct_feedback_objective": "learn H_pred->V1_E future template route from training schedule",
            "som_feedback_objective": "learn H_pred->V1_SOM future template route from training schedule",
            "uses_q_activity_decoder_dampening_metrics": False,
            "uses_actual_unexpected_trailer_labels": False,
            "checkpoint_selected_by": "predefined seed/n_trials/target only",
            "gains_selected_by": "validation V1/SOM template recruitment score under rate caps only",
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
    print(f"selected_g_total={selected_g}")
    print(f"selected_r={selected_r}")
    print(f"selected_resolved_g_direct={selected_direct}")
    print(f"selected_resolved_g_som={selected_som}")
    for row in validation_rows:
        if row.get("status") == "PASS":
            m = row["metrics"]
            print(
                f"validation g={m['g_total']:g} r={m['r']:g} "
                f"v1_pred={m['v1_predicted_channel_count_mean']:.6f} "
                f"v1_cos={m['v1_target_cosine_mean']:.6f} "
                f"som_pred={m['som_predicted_channel_count_mean']:.6f} "
                f"som_cos={m['som_target_cosine_mean']:.6f} "
                f"v1_hz={m['v1_population_rate_hz']:.6f} "
                f"som_hz={m['som_population_rate_hz']:.6f} "
                f"score={m['selection_score']:.6f} "
                f"eligible={m['eligible_under_preregistered_rule']}"
            )
        else:
            print(
                f"validation g={row['g_total']:g} r={row['r']:g} "
                f"FAILED log={row['run']['log_path']}"
            )
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
