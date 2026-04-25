#!/usr/bin/env python3
"""Condition-blind prediction-efficiency calibration for learned feedback.

This experiment reuses prediction-only learned feedback checkpoints and selects
feedback gains from validation recruitment per active-charge cost. Selection is
condition-blind: it ignores expected-vs-unexpected Q/activity/decoder gaps and
uses only future-orientation template recruitment, global active charge, and
rate/pathology caps before frozen Richter testing.
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


DEFAULT_SOM_CHECKPOINT = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "expectation_snn/data/checkpoints_native_cpp/"
    "stage1_orientation_learned_som_gain_calibration_20260425/"
    "stage1_ctx_pred_orientation_cell_learned_som_gaincal_seed42_n72.json"
)
DEFAULT_JOINT_CHECKPOINT = Path(
    "/workspace/neuroips_gpu_migration_20260422/neuroips/"
    "expectation_snn/data/checkpoints_native_cpp/"
    "stage1_orientation_joint_learned_feedback_20260425/"
    "stage1_ctx_pred_orientation_cell_joint_learned_feedback_seed42_n72.json"
)
VALIDATION_SEED = 5454
SOM_GAINS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
JOINT_GRID = (
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
    "som_gaincal_high_g64": {
        "q_u_minus_e": -2048606.843238,
        "activity_u_minus_e": 133.833333,
        "decoder_u_minus_e_keep_p_0.02": 0.099080,
    },
    "joint_gaincal_high_g128_r1": {
        "q_u_minus_e": -1221796.372786,
        "activity_u_minus_e": 129.833333,
        "decoder_u_minus_e_keep_p_0.02": 0.046692,
    },
}


@dataclass(frozen=True)
class EvalRow:
    family: str
    name: str
    checkpoint: Path
    localizer: Path
    feedback_g_total: float
    feedback_r: float
    feedback_direct_source: str
    feedback_som_source: str
    analysis_role: str

    @property
    def row(self) -> Row:
        return Row(
            condition=f"frozen_{self.family}_prediction_efficiency_calibration",
            name=self.name,
            feedback_g_total=self.feedback_g_total,
            feedback_r=self.feedback_r,
            feedback_som_center_weight=0.10,
            note=(
                "Frozen prediction-only learned feedback route selected by "
                "condition-blind prediction-recruitment per active-charge cost."
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


def _native_validation_command(
    binary: Path,
    checkpoint: Path,
    out: Path,
    g_total: float,
    r: float,
    direct_source: str,
    som_source: str,
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
        direct_source,
        "--feedback-som-source",
        som_source,
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
    family: str,
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
    pred_v1 = v1_counts[np.arange(v1_counts.shape[0]), predicted_channel]
    pred_som = som_counts[np.arange(som_counts.shape[0]), predicted_channel]
    v1_total = v1_counts.sum(axis=1)
    som_total = som_counts.sum(axis=1)
    v1_cosine = _one_hot_cosine(v1_counts, predicted_channel)
    som_cosine = _one_hot_cosine(som_counts, predicted_channel)
    v1_q = np.asarray(trial_data["v1_trailer_q_active_fC"], dtype=np.float64)
    v1e_q = np.asarray(trial_data["v1e_trailer_q_active_fC"], dtype=np.float64)
    som_q = np.asarray(trial_data["v1som_trailer_q_active_fC"], dtype=np.float64)
    v1_population_rate_hz = float(np.mean(v1_total) / (192.0 * 0.5))
    som_population_rate_hz = float(np.mean(som_total) / (768.0 * 0.5))
    pred_v1_mean = float(np.mean(pred_v1))
    pred_som_mean = float(np.mean(pred_som))
    v1_component = float(np.mean(v1_cosine)) * math.log1p(max(pred_v1_mean, 0.0))
    som_component = float(np.mean(som_cosine)) * math.log1p(max(pred_som_mean, 0.0))
    if family == "som":
        prediction_score = som_component
    elif family == "joint":
        prediction_score = v1_component + som_component
    else:
        raise ValueError(f"unknown family: {family}")
    mean_q = float(np.mean(v1_q))
    q_cost_10mfc = max(mean_q / 10_000_000.0, 1e-9)
    efficiency_score = prediction_score / q_cost_10mfc
    balance = _resolved_balance(g_total, r)
    eligible = (
        g_total > 0.0
        and prediction_score > 0.0
        and float(np.mean(v1_total)) >= 5.0
        and v1_population_rate_hz <= 100.0
        and som_population_rate_hz <= 50.0
    )
    return {
        "family": family,
        "g_total": g_total,
        "r": r,
        "resolved_g_direct": balance["g_direct"],
        "resolved_g_som": balance["g_som"],
        "artifact_path": str(artifact_path),
        "target_template": "future next-orientation channel (leader_channel + 2) % 12",
        "trial_count": int(v1_counts.shape[0]),
        "v1_total_count_mean": float(np.mean(v1_total)),
        "v1_predicted_channel_count_mean": pred_v1_mean,
        "v1_target_cosine_mean": float(np.mean(v1_cosine)),
        "v1_prediction_component": v1_component,
        "v1_population_rate_hz": v1_population_rate_hz,
        "som_total_count_mean": float(np.mean(som_total)),
        "som_predicted_channel_count_mean": pred_som_mean,
        "som_target_cosine_mean": float(np.mean(som_cosine)),
        "som_prediction_component": som_component,
        "som_population_rate_hz": som_population_rate_hz,
        "mean_v1_q_active_fC_per_trial": mean_q,
        "mean_v1e_q_active_fC_per_trial": float(np.mean(v1e_q)),
        "mean_v1som_q_active_fC_per_trial": float(np.mean(som_q)),
        "q_cost_10mfc": q_cost_10mfc,
        "prediction_score": prediction_score,
        "efficiency_score": efficiency_score,
        "eligible_under_preregistered_rule": eligible,
    }


def _select_family(family: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [
        row for row in rows
        if row.get("status") == "PASS"
        and row["metrics"]["family"] == family
        and row["metrics"]["eligible_under_preregistered_rule"]
    ]
    if not eligible:
        raise RuntimeError(f"no eligible validation row for {family}")
    selected = max(
        eligible,
        key=lambda row: (
            row["metrics"]["efficiency_score"],
            -row["metrics"]["g_total"],
        ),
    )
    m = selected["metrics"]
    return {
        "family": family,
        "selected_g_total": m["g_total"],
        "selected_r": m["r"],
        "selected_resolved_g_direct": m["resolved_g_direct"],
        "selected_resolved_g_som": m["resolved_g_som"],
        "selected_validation_artifact": m["artifact_path"],
        "selected_validation_log": selected["run"]["log_path"],
        "selected_prediction_score": m["prediction_score"],
        "selected_mean_q_active_fC_per_trial": m["mean_v1_q_active_fC_per_trial"],
        "selected_efficiency_score": m["efficiency_score"],
        "selection_formula": (
            "efficiency_score = prediction_score / max(mean_trial_v1_q_active_fC / 1e7, 1e-9); "
            "SOM family prediction_score = SOM one-hot future-orientation cosine * "
            "log1p(predicted SOM-channel count); joint family prediction_score = "
            "that SOM component plus V1_E one-hot future-orientation cosine * "
            "log1p(predicted V1_E-channel count). Eligibility requires nonzero "
            "prediction_score, V1_E population rate <= 100 Hz, V1_SOM population "
            "rate <= 50 Hz, and mean V1_E trailer count >= 5. Tie-break lower g_total."
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
        f"| {row['family']} | {row['name']} | {row['analysis_role']} | "
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
            f"| {row['family']} | {row['g_total']:.6g} | {row['r']:.6g} | FAIL |  |  |  |  |  |  |  | |"
        )
    m = row["metrics"]
    return (
        f"| {m['family']} | {m['g_total']:.6g} | {m['r']:.6g} | PASS | "
        f"{m['prediction_score']:.6f} | {m['mean_v1_q_active_fC_per_trial']:.2f} | "
        f"{m['efficiency_score']:.6f} | {m['v1_predicted_channel_count_mean']:.6f} | "
        f"{m['som_predicted_channel_count_mean']:.6f} | {m['v1_population_rate_hz']:.6f} | "
        f"{m['som_population_rate_hz']:.6f} | {m['eligible_under_preregistered_rule']} |"
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Native Richter Prediction-Efficiency Calibration - 2026-04-25",
        "",
        "Scope: condition-blind gain calibration for existing prediction-only learned",
        "feedback routes. No suppressors/divisive mechanisms are used. No expected-",
        "vs-unexpected Q/activity/decoder gap metric is used for gain selection.",
        "",
        f"SOM checkpoint: `{summary['checkpoints']['som']}`",
        f"Joint checkpoint: `{summary['checkpoints']['joint']}`",
        "",
        "## Selection Formula",
        "",
        summary["selection"]["som"]["selection_formula"],
        "",
        "## Selected Rows",
        "",
        f"SOM selected: `{summary['selection']['som']['selected_g_total']}`, r=`{summary['selection']['som']['selected_r']}`",
        f"Joint selected: `{summary['selection']['joint']['selected_g_total']}`, r=`{summary['selection']['joint']['selected_r']}`",
        "",
        "## Validation Tradeoff",
        "",
        "| Family | g_total | r | Status | Prediction score | Mean Q fC | Efficiency | V1 pred | SOM pred | V1 Hz | SOM Hz | Eligible |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_validation_markdown(row) for row in summary["validation_rows"])
    lines.extend([
        "",
        "## Frozen Richter Test Rows",
        "",
        "| Family | Row | Role | Direct source | SOM source | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    lines.extend(_row_markdown(row) for row in summary["rows"])
    lines.extend([
        "",
        "## Comparators",
        "",
        f"Original natural baseline: `{summary['baseline_comparator']['original_natural']}`",
        f"SOM high-gain comparator: `{summary['baseline_comparator']['som_gaincal_high_g64']}`",
        f"Joint high-gain comparator: `{summary['baseline_comparator']['joint_gaincal_high_g128_r1']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--som-checkpoint", type=Path, default=DEFAULT_SOM_CHECKPOINT)
    parser.add_argument("--joint-checkpoint", type=Path, default=DEFAULT_JOINT_CHECKPOINT)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--n-thinning-seeds", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.stage3_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.stage3_dir / "prediction_efficiency_calibration_20260425"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.som_checkpoint.exists():
        raise FileNotFoundError(args.som_checkpoint)
    if not args.joint_checkpoint.exists():
        raise FileNotFoundError(args.joint_checkpoint)

    validation_rows: List[Dict[str, Any]] = []
    for gain in SOM_GAINS:
        family = "som"
        slug = f"som_g{_slug_float(gain)}_r0"
        artifact = out_dir / f"validation_efficiency_{slug}.json"
        log_path = args.log_dir / f"native_richter_efficiency_validation_{slug}_20260425.log"
        cmd = _native_validation_command(
            args.binary,
            args.som_checkpoint,
            artifact,
            gain,
            0.0,
            "disabled",
            "learned",
        )
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
            "family": family,
            "g_total": gain,
            "r": 0.0,
            "artifact_path": str(artifact),
            "run": run,
            "status": "PASS" if run["returncode"] == 0 and artifact.exists() else "FAIL",
        }
        if row["status"] == "PASS":
            row["metrics"] = _validation_metrics(artifact, family, gain, 0.0)
        validation_rows.append(row)

    for g_total, r in JOINT_GRID:
        family = "joint"
        slug = f"joint_g{_slug_float(g_total)}_r{_slug_float(r)}"
        artifact = out_dir / f"validation_efficiency_{slug}.json"
        log_path = args.log_dir / f"native_richter_efficiency_validation_{slug}_20260425.log"
        cmd = _native_validation_command(
            args.binary,
            args.joint_checkpoint,
            artifact,
            g_total,
            r,
            "learned",
            "learned",
        )
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
        row = {
            "family": family,
            "g_total": g_total,
            "r": r,
            "artifact_path": str(artifact),
            "run": run,
            "status": "PASS" if run["returncode"] == 0 and artifact.exists() else "FAIL",
        }
        if row["status"] == "PASS":
            row["metrics"] = _validation_metrics(artifact, family, g_total, r)
        validation_rows.append(row)

    selection = {
        "som": _select_family("som", validation_rows),
        "joint": _select_family("joint", validation_rows),
    }
    som_g = float(selection["som"]["selected_g_total"])
    joint_g = float(selection["joint"]["selected_g_total"])
    joint_r = float(selection["joint"]["selected_r"])

    som_localizer = out_dir / "richter_efficiency_som_sensory_only_g0.json"
    joint_localizer = out_dir / "richter_efficiency_joint_sensory_only_g0.json"
    localizer_specs = [
        (
            "som",
            args.som_checkpoint,
            som_localizer,
            Row(
                "sensory_only_localizer",
                "efficiency_som_sensory_only_g0",
                0.0,
                0.0,
                note="SOM checkpoint sensory-only localizer.",
                extra_args=(
                    "--feedback-direct-source",
                    "disabled",
                    "--feedback-som-source",
                    "disabled",
                ),
            ),
        ),
        (
            "joint",
            args.joint_checkpoint,
            joint_localizer,
            Row(
                "sensory_only_localizer",
                "efficiency_joint_sensory_only_g0",
                0.0,
                0.0,
                note="Joint checkpoint sensory-only localizer.",
                extra_args=(
                    "--feedback-direct-source",
                    "disabled",
                    "--feedback-som-source",
                    "disabled",
                ),
            ),
        ),
    ]
    localizer_runs: Dict[str, Any] = {}
    for family, checkpoint, localizer_path, row in localizer_specs:
        log_path = args.log_dir / f"native_richter_efficiency_{family}_sensory_only_20260425.log"
        cmd = _native_command(row, args.binary, checkpoint, localizer_path)
        if args.force or not localizer_path.exists():
            run = _run_command(cmd, log_path, args.repo)
            if run["returncode"] != 0:
                raise RuntimeError(f"{family} localizer failed: {log_path}")
        else:
            run = {
                "cmd": cmd,
                "log_path": str(log_path),
                "returncode": 0,
                "wall_seconds": None,
                "skipped_existing_artifact": True,
            }
        localizer_runs[family] = run

    eval_rows = [
        EvalRow(
            "som",
            f"efficiency_selected_som_g{_slug_float(som_g)}_r0",
            args.som_checkpoint,
            som_localizer,
            som_g,
            0.0,
            "disabled",
            "learned",
            "efficiency_selected_som_only",
        ),
        EvalRow(
            "som",
            f"efficiency_selected_som_disabled_g{_slug_float(som_g)}_r0",
            args.som_checkpoint,
            som_localizer,
            som_g,
            0.0,
            "disabled",
            "disabled",
            "selected_som_disabled_control",
        ),
        EvalRow(
            "som",
            f"efficiency_selected_som_shifted_g{_slug_float(som_g)}_r0",
            args.som_checkpoint,
            som_localizer,
            som_g,
            0.0,
            "disabled",
            "learned-shifted",
            "selected_som_wrong_hpred_control",
        ),
        EvalRow(
            "som",
            "efficiency_prior_high_som_g64_r0",
            args.som_checkpoint,
            som_localizer,
            64.0,
            0.0,
            "disabled",
            "learned",
            "prior_high_gain_som_comparator",
        ),
        EvalRow(
            "joint",
            f"efficiency_selected_joint_g{_slug_float(joint_g)}_r{_slug_float(joint_r)}",
            args.joint_checkpoint,
            joint_localizer,
            joint_g,
            joint_r,
            "learned",
            "learned",
            "efficiency_selected_joint_direct_plus_som",
        ),
        EvalRow(
            "joint",
            f"efficiency_selected_joint_disabled_g{_slug_float(joint_g)}_r{_slug_float(joint_r)}",
            args.joint_checkpoint,
            joint_localizer,
            joint_g,
            joint_r,
            "disabled",
            "disabled",
            "selected_joint_disabled_control",
        ),
        EvalRow(
            "joint",
            f"efficiency_selected_joint_shifted_g{_slug_float(joint_g)}_r{_slug_float(joint_r)}",
            args.joint_checkpoint,
            joint_localizer,
            joint_g,
            joint_r,
            "learned-shifted",
            "learned-shifted",
            "selected_joint_wrong_hpred_control",
        ),
        EvalRow(
            "joint",
            "efficiency_prior_high_joint_g128_r1",
            args.joint_checkpoint,
            joint_localizer,
            128.0,
            1.0,
            "learned",
            "learned",
            "prior_high_gain_joint_comparator",
        ),
    ]

    rows_out: List[Dict[str, Any]] = []
    for eval_row in eval_rows:
        artifact = out_dir / f"richter_dampening_{eval_row.name}.json"
        log_path = args.log_dir / f"native_richter_efficiency_{eval_row.name}_20260425.log"
        cmd = _native_command(eval_row.row, args.binary, eval_row.checkpoint, artifact)
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
            eval_row.localizer,
            KEEP_PS,
            args.n_thinning_seeds,
        )
        row_out: Dict[str, Any] = {
            "family": eval_row.family,
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

    summary_json = (
        args.stage3_dir
        / "stage3_native_richter_prediction_efficiency_calibration_20260425.json"
    )
    summary_md = (
        args.stage3_dir
        / "stage3_native_richter_prediction_efficiency_calibration_20260425.md"
    )
    summary: Dict[str, Any] = {
        "analysis_kind": "native_richter_condition_blind_prediction_efficiency_calibration",
        "checkpoints": {
            "som": str(args.som_checkpoint),
            "joint": str(args.joint_checkpoint),
        },
        "sensory_only_localizers": {
            "som": str(som_localizer),
            "joint": str(joint_localizer),
        },
        "n_thinning_seeds": int(args.n_thinning_seeds),
        "keep_ps": list(KEEP_PS),
        "validation_seed": VALIDATION_SEED,
        "selection": selection,
        "validation_rows": validation_rows,
        "baseline_comparator": BASELINE,
        "training_constraints": {
            "uses_existing_prediction_only_checkpoints": True,
            "uses_q_activity_decoder_dampening_metrics_for_selection": False,
            "uses_actual_unexpected_trailer_labels_for_selection": False,
            "engineered_suppressors_or_divisive_mechanisms": False,
        },
        "runs": {
            "localizers": localizer_runs,
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
    print(f"selected_som_g={som_g}")
    print(f"selected_joint_g={joint_g}")
    print(f"selected_joint_r={joint_r}")
    for key in ("som", "joint"):
        sel = selection[key]
        print(
            f"selection {key}: g={sel['selected_g_total']:g} r={sel['selected_r']:g} "
            f"score={sel['selected_prediction_score']:.6f} "
            f"q={sel['selected_mean_q_active_fC_per_trial']:.2f} "
            f"eff={sel['selected_efficiency_score']:.6f}"
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
