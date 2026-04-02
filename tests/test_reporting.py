from __future__ import annotations

from pathlib import Path
import csv
import json
import tempfile
import unittest

from lvc_expectation.reporting import (
    CLASS_CHALLENGER_NO_ADMISSION,
    CLASS_PHASE1_POSTSTIM_POSITIVE,
    CLASS_PHASE1_PRESTIM_NEGATIVE,
    CLASS_PHASE15_POSITIVE,
    CLASS_PHASE2_CLEAN_NEGATIVE,
    DIAGNOSTIC_CALIBRATION_REPORT_V2_ARTIFACT,
    FROZEN_BENCHMARK_METRIC_VERSION,
    LEGACY_BENCHMARK_METRIC_VERSION,
    METRIC_SCHEMA_VERSION,
    CLASSIFICATION_RULE_VERSION,
    SEEDWISE_CSV_ARTIFACT,
    STATS_JSON_ARTIFACT,
    VERDICT_JSON_ARTIFACT,
    build_seed_aggregate_report,
    discover_archived_run_families,
    load_archived_run_bundle,
    write_diagnostic_calibration_report_v2,
    write_seed_aggregate_artifact_trio,
    write_seed_aggregate_report,
)


class ReportingTests(unittest.TestCase):
    def _write_manifest(
        self,
        run_dir: Path,
        *,
        heldout_seed: int,
        notes: dict[str, object] | None = None,
        heldout_assays: list[str] | None = None,
    ) -> None:
        payload = {
            "run_id": run_dir.name,
            "created_at": "2026-04-01T00:00:00+00:00",
            "config_name": "phase1_core",
            "train_objectives": [],
            "heldout_assays": heldout_assays or [],
            "notes": notes or {},
        }
        payload["notes"]["heldout_seed"] = heldout_seed
        (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_primary_metrics(self, run_dir: Path, learned: float, oracle: float, *, frozen: bool = True) -> None:
        learned_payload = {
            "dampening": {
                "mean_suppression": 0.1,
                "poststimulus_l23_template_specificity": learned,
            }
            if frozen
            else {
                "mean_suppression": 0.1,
            }
        }
        oracle_payload = {
            "dampening": {
                "mean_suppression": 0.2,
                "poststimulus_l23_template_specificity": oracle,
            }
            if frozen
            else {
                "mean_suppression": 0.2,
            }
        }
        (run_dir / "primary_metrics.json").write_text(json.dumps(learned_payload, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "oracle_primary_metrics.json").write_text(json.dumps(oracle_payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_probe_metrics(self, run_dir: Path, learned: float, oracle: float) -> None:
        learned_payload = {
            "probe_target_aligned_specificity_contrast": learned,
            "probe_pooled_target_aligned_specificity_contrast": learned / 2.0,
        }
        oracle_payload = {
            "probe_target_aligned_specificity_contrast": oracle,
            "probe_pooled_target_aligned_specificity_contrast": oracle / 2.0,
        }
        (run_dir / "probe_metrics.json").write_text(json.dumps(learned_payload, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "oracle_probe_metrics.json").write_text(json.dumps(oracle_payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_alignment_report(self, run_dir: Path, learned: float, oracle: float) -> None:
        payload = {
            "learned_probe_alignment_kl": learned,
            "oracle_probe_alignment_kl": oracle,
        }
        (run_dir / "probe_context_alignment_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_prestim_gate(self, run_dir: Path, value: float) -> None:
        payload = {
            "control_mode": "zero_context",
            "subcase": "adaptation_only",
            "intact": {
                "context_only": {
                    "n_steps": 12,
                    "n_trials": 6,
                    "prestimulus_template_specificity": value,
                    "template_peak": 0.03,
                }
            },
            "zero_context": {
                "context_only": {
                    "n_steps": 12,
                    "n_trials": 6,
                    "prestimulus_template_specificity": 0.0,
                    "template_peak": 0.0,
                }
            },
        }
        (run_dir / "prestim_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def test_seed_aggregate_report_is_deterministic_and_includes_oracle_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            values = [0.2, -0.1, 0.4]
            oracle_values = [0.4, 0.2, 0.8]
            run_dirs = []
            for index, (value, oracle_value) in enumerate(zip(values, oracle_values), start=1):
                run_dir = root / f"run_{index}"
                run_dir.mkdir()
                self._write_manifest(
                    run_dir,
                    heldout_seed=100 + index,
                    notes={"controlled_sources": [0, 1, 2, 3, 4, 5]},
                    heldout_assays=["probe_target_aligned_specificity_contrast"],
                )
                self._write_probe_metrics(run_dir, learned=value, oracle=oracle_value)
                run_dirs.append(run_dir)

            report_a = build_seed_aggregate_report(
                run_dirs,
                metric_name="probe_target_aligned_specificity_contrast",
                bootstrap_seed=17,
                bootstrap_resamples=256,
            )
            report_b = build_seed_aggregate_report(
                run_dirs,
                metric_name="probe_target_aligned_specificity_contrast",
                bootstrap_seed=17,
                bootstrap_resamples=256,
            )

            self.assertEqual(report_a, report_b)
            self.assertEqual(report_a["report_version"], "stage1_reporting_v1")
            self.assertEqual(report_a["metric_schema_version"], METRIC_SCHEMA_VERSION)
            self.assertEqual(report_a["classification_rule_version"], CLASSIFICATION_RULE_VERSION)
            self.assertEqual(report_a["benchmark_metric_version"], FROZEN_BENCHMARK_METRIC_VERSION)
            self.assertEqual(report_a["metric_versions"]["probe_target_aligned_specificity_contrast"], "v1")
            self.assertEqual(report_a["classification"], CLASS_PHASE15_POSITIVE)
            self.assertEqual([item["heldout_seed"] for item in report_a["seedwise"]], [101, 102, 103])
            self.assertAlmostEqual(report_a["summary"]["mean"], sum(values) / len(values), places=6)
            self.assertEqual(report_a["summary"]["median"], 0.2)
            self.assertEqual(report_a["summary"]["sign_counts"], {"positive": 2, "negative": 1, "zero": 0})
            self.assertEqual(report_a["summary"]["bootstrap_mean_ci_95"]["seed"], 17)
            self.assertEqual(report_a["summary"]["bootstrap_mean_ci_95"]["resamples"], 256)
            ratio_summary = report_a["oracle_normalized_ratio_summary"]
            self.assertIsNotNone(ratio_summary)
            self.assertAlmostEqual(ratio_summary["mean"], 1.0 / 6.0, places=6)
            self.assertEqual(ratio_summary["sign_counts"], {"positive": 2, "negative": 1, "zero": 0})

            output_path = root / "aggregate_report.json"
            write_seed_aggregate_report(report_a, output_path)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), report_a)

            trio_paths_a = write_seed_aggregate_artifact_trio(report_a, root / "artifacts_a")
            trio_paths_b = write_seed_aggregate_artifact_trio(report_b, root / "artifacts_b")
            self.assertEqual(sorted(path.name for path in trio_paths_a.values()), [
                SEEDWISE_CSV_ARTIFACT,
                STATS_JSON_ARTIFACT,
                VERDICT_JSON_ARTIFACT,
            ])
            for key in trio_paths_a:
                self.assertEqual(trio_paths_a[key].read_bytes(), trio_paths_b[key].read_bytes())

            with trio_paths_a["seedwise_csv"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["heldout_seed"], "101")

            stats_payload = json.loads(trio_paths_a["stats_report"].read_text(encoding="utf-8"))
            verdict_payload = json.loads(trio_paths_a["verdict_report"].read_text(encoding="utf-8"))
            for payload in (stats_payload, verdict_payload):
                self.assertEqual(payload["metric_schema_version"], METRIC_SCHEMA_VERSION)
                self.assertEqual(payload["classification_rule_version"], CLASSIFICATION_RULE_VERSION)
                self.assertIn("metric_versions", payload)

    def test_bundle_classification_and_metric_versions_cover_known_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            phase1_run = root / "phase1_run"
            phase1_run.mkdir()
            self._write_manifest(phase1_run, heldout_seed=11, heldout_assays=["mean_suppression"])
            self._write_primary_metrics(phase1_run, learned=0.4, oracle=0.5, frozen=True)

            legacy_run = root / "legacy_run"
            legacy_run.mkdir()
            self._write_manifest(legacy_run, heldout_seed=12, heldout_assays=["mean_suppression"])
            self._write_primary_metrics(legacy_run, learned=0.0, oracle=0.0, frozen=False)

            phase15_run = root / "phase15_run"
            phase15_run.mkdir()
            self._write_manifest(
                phase15_run,
                heldout_seed=13,
                notes={"gate": "local_global_surprise_probe", "controlled_sources": [0, 1, 2, 3, 4, 5]},
                heldout_assays=["probe_target_aligned_specificity_contrast"],
            )
            self._write_probe_metrics(phase15_run, learned=0.02, oracle=0.02)

            phase2_run = root / "phase2_run"
            phase2_run.mkdir()
            self._write_manifest(
                phase2_run,
                heldout_seed=14,
                notes={
                    "gate": "local_global_surprise_probe",
                    "phase2_regime": "p2a_source_heldout_probe_generalization",
                    "controlled_sources": [0, 2, 4],
                },
                heldout_assays=["probe_target_aligned_specificity_contrast"],
            )
            self._write_probe_metrics(phase2_run, learned=0.0, oracle=0.0)

            challenger_run = root / "challenger_run"
            challenger_run.mkdir()
            self._write_manifest(
                challenger_run,
                heldout_seed=15,
                notes={
                    "gate": "probe_context_alignment",
                    "phase2_regime": "p2a_source_heldout_probe_generalization",
                    "challenger_candidate": "std400_only",
                    "controlled_sources": [1, 3, 5],
                },
                heldout_assays=["probe_context_alignment"],
            )
            self._write_alignment_report(challenger_run, learned=0.1, oracle=0.0)

            prestim_run = root / "prestim_run"
            prestim_run.mkdir()
            self._write_manifest(
                prestim_run,
                heldout_seed=16,
                notes={"gate": "context_only_prestim_template"},
                heldout_assays=["prestimulus_template_specificity"],
            )
            self._write_prestim_gate(prestim_run, value=-0.002)

            self.assertEqual(load_archived_run_bundle(phase1_run).classification, CLASS_PHASE1_POSTSTIM_POSITIVE)
            self.assertEqual(load_archived_run_bundle(phase1_run).source_metric_version, FROZEN_BENCHMARK_METRIC_VERSION)
            self.assertEqual(load_archived_run_bundle(legacy_run).source_metric_version, LEGACY_BENCHMARK_METRIC_VERSION)
            self.assertEqual(load_archived_run_bundle(phase15_run).classification, CLASS_PHASE15_POSITIVE)
            self.assertEqual(load_archived_run_bundle(phase2_run).classification, CLASS_PHASE2_CLEAN_NEGATIVE)
            self.assertEqual(load_archived_run_bundle(challenger_run).classification, CLASS_CHALLENGER_NO_ADMISSION)
            self.assertEqual(load_archived_run_bundle(prestim_run).classification, CLASS_PHASE1_PRESTIM_NEGATIVE)

            prestim_report = build_seed_aggregate_report(
                [prestim_run],
                metric_name="prestimulus_template_specificity",
            )
            self.assertEqual(prestim_report["classification"], CLASS_PHASE1_PRESTIM_NEGATIVE)
            self.assertAlmostEqual(prestim_report["seedwise"][0]["value"], -0.002, places=6)
            self.assertIsNone(prestim_report["seedwise"][0]["oracle_value"])
            self.assertIsNone(prestim_report["oracle_normalized_ratio_summary"])

    def test_discover_archived_run_families_groups_by_classification_and_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for run_name, controlled_sources in (
                ("even_a", [0, 2, 4]),
                ("even_b", [0, 2, 4]),
                ("odd_a", [1, 3, 5]),
            ):
                run_dir = root / run_name
                run_dir.mkdir()
                self._write_manifest(
                    run_dir,
                    heldout_seed=20 + len(controlled_sources),
                    notes={
                        "gate": "local_global_surprise_probe",
                        "phase2_regime": "p2a_source_heldout_probe_generalization",
                        "controlled_sources": controlled_sources,
                    },
                    heldout_assays=["probe_target_aligned_specificity_contrast"],
                )
                self._write_probe_metrics(run_dir, learned=0.0, oracle=0.0)

            families = discover_archived_run_families(root)
            family_sizes = sorted(len(items) for items in families.values())
            self.assertEqual(family_sizes, [1, 2])
            for family_key, bundles in families.items():
                if len(bundles) == 2:
                    self.assertIn('"controlled_sources": [0, 2, 4]', family_key)
                if len(bundles) == 1:
                    self.assertIn('"controlled_sources": [1, 3, 5]', family_key)

    def test_stage2_diagnostic_calibration_report_rescores_frozen_probe_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            positive = write_diagnostic_calibration_report_v2(
                "phase15_probe_positive_v1",
                artifacts_root="artifacts",
                output_dir=output_root / "phase15",
            )
            primary_negative = write_diagnostic_calibration_report_v2(
                "phase2_p2a_primary_negative_v1",
                artifacts_root="artifacts",
                output_dir=output_root / "phase2_primary_negative",
            )
            challenger_negative = write_diagnostic_calibration_report_v2(
                "phase2_challenger_selection_negative_v1",
                artifacts_root="artifacts",
                output_dir=output_root / "phase2_challenger_negative",
            )

            self.assertTrue((output_root / "phase15" / DIAGNOSTIC_CALIBRATION_REPORT_V2_ARTIFACT).exists())
            self.assertTrue(
                (output_root / "phase2_primary_negative" / DIAGNOSTIC_CALIBRATION_REPORT_V2_ARTIFACT).exists()
            )
            self.assertTrue(
                (output_root / "phase2_challenger_negative" / DIAGNOSTIC_CALIBRATION_REPORT_V2_ARTIFACT).exists()
            )

            for report in (positive["report"], primary_negative["report"], challenger_negative["report"]):
                for key in (
                    "metric_schema_version",
                    "metric_versions",
                    "classification_rule_version",
                    "source_metric_versions",
                    "diagnostic_schema_version",
                    "benchmark_registry_version",
                    "summary",
                    "seedwise",
                ):
                    self.assertIn(key, report)

            positive_summary = positive["report"]["summary"]
            primary_negative_summary = primary_negative["report"]["summary"]
            challenger_summary = challenger_negative["report"]["summary"]

            self.assertGreater(positive_summary["probe_correct_pair_flip_rate__v2"]["mean"], 0.9)
            self.assertLess(primary_negative_summary["probe_correct_pair_flip_rate__v2"]["mean"], 0.1)
            self.assertLess(challenger_summary["probe_correct_pair_flip_rate__v2"]["mean"], 0.9)
            self.assertGreater(
                positive_summary["probe_within_pair_mass__v2"]["mean"],
                primary_negative_summary["probe_within_pair_mass__v2"]["mean"],
            )
            self.assertGreater(
                positive_summary["learned_probe_expected_target_margin__v2"]["mean"],
                challenger_summary["learned_probe_expected_target_margin__v2"]["mean"],
            )


if __name__ == "__main__":
    unittest.main()
