from __future__ import annotations

from pathlib import Path
import csv
import json
import tempfile
import unittest

from lvc_expectation.config import ArtifactConfig, ExperimentConfig
from lvc_expectation.diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
    HIDDEN_STATE_PROBE_TABLE_FIELDS,
    HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT,
)
from lvc_expectation.runner import run_stage3_task_definition
from lvc_expectation.stage3 import (
    ORACLE_TASK_FEASIBILITY_REPORT_V2_ARTIFACT,
    SHORTCUT_BASELINE_REPORT_V2_ARTIFACT,
    STAGE3_TASK_VERSION,
    SUPPORT_SUMMARY_V2_ARTIFACT,
    SUPPORT_TABLE_FIELDS,
    SUPPORT_TABLE_V2_ARTIFACT,
    TASK_DEFINITION_MANIFEST_V2_ARTIFACT,
    TASK_DEFINITION_VERDICT_V2_ARTIFACT,
)


_REQUIRED_STAGE3_JSON_FIELDS = (
    "metric_schema_version",
    "metric_versions",
    "classification_rule_version",
    "source_metric_versions",
    "benchmark_registry_version",
    "stage3_task_version",
)


class Stage3TaskDefinitionTests(unittest.TestCase):
    def test_stage3_task_definition_writes_revised_v2_artifacts_and_checks_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_stage3_task_definition(config=config)

            verdict_path = result.output_root / TASK_DEFINITION_VERDICT_V2_ARTIFACT
            self.assertTrue(verdict_path.exists())
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                self.assertIn(field_name, verdict)
            self.assertEqual(verdict["stage3_task_version"], STAGE3_TASK_VERSION)
            self.assertEqual(len(verdict["panel_results"]), 4)
            self.assertTrue(verdict["passes"])

            for direction_id in ("splitA_to_splitB", "splitB_to_splitA"):
                for seed_panel in ("primary", "confirmation"):
                    panel_dir = result.output_root / direction_id / seed_panel
                    self.assertTrue((panel_dir / TASK_DEFINITION_MANIFEST_V2_ARTIFACT).exists())
                    self.assertTrue((panel_dir / SUPPORT_TABLE_V2_ARTIFACT).exists())
                    self.assertTrue((panel_dir / SUPPORT_SUMMARY_V2_ARTIFACT).exists())
                    self.assertTrue((panel_dir / SHORTCUT_BASELINE_REPORT_V2_ARTIFACT).exists())
                    self.assertTrue((panel_dir / ORACLE_TASK_FEASIBILITY_REPORT_V2_ARTIFACT).exists())

                    manifest_payload = json.loads((panel_dir / TASK_DEFINITION_MANIFEST_V2_ARTIFACT).read_text(encoding="utf-8"))
                    for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                        self.assertIn(field_name, manifest_payload)
                    self.assertEqual(manifest_payload["direction_id"], direction_id)
                    self.assertEqual(manifest_payload["seed_panel"], seed_panel)
                    self.assertEqual(manifest_payload["subcase"], "dampening")
                    self.assertEqual(manifest_payload["probe_visible_step_index"], 1)
                    self.assertEqual(manifest_payload["controlled_sources"], [0, 1, 2, 3, 4, 5])
                    self.assertEqual(manifest_payload["split_definition"], "mixed_offset_complementary_pair_sets")

                    with (panel_dir / SUPPORT_TABLE_V2_ARTIFACT).open("r", encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        rows = list(reader)
                    self.assertEqual(tuple(reader.fieldnames or ()), SUPPORT_TABLE_FIELDS)
                    self.assertEqual(len(rows), 48)
                    for required_column in (
                        "pair_id",
                        "symmetry_mate_pair_id",
                        "expected_target_orientation",
                        "local_offset_family",
                    ):
                        self.assertIn(required_column, reader.fieldnames or ())

                    support_summary = json.loads((panel_dir / SUPPORT_SUMMARY_V2_ARTIFACT).read_text(encoding="utf-8"))
                    for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                        self.assertIn(field_name, support_summary)
                    for bool_field in (
                        "all_six_sources_present_in_train",
                        "all_six_sources_present_in_eval",
                        "both_offset_families_present_in_train",
                        "both_offset_families_present_in_eval",
                        "train_eval_pair_disjoint",
                        "pair_sets_complementary",
                        "source_context_ambiguous_in_train",
                        "source_context_ambiguous_in_eval",
                        "symmetry_mates_complete_in_train",
                        "symmetry_mates_complete_in_eval",
                        "contexts_balanced",
                        "expected_unexpected_balanced",
                        "visible_step_fixed",
                        "task_fixed_orientation_relevant",
                        "prestim_fixed_none",
                        "controlled_sources_fixed",
                        "no_blank_scored_rows",
                        "no_uncontrolled_scored_rows",
                    ):
                        self.assertTrue(support_summary[bool_field], msg=f"{direction_id}/{seed_panel}: {bool_field}")

                    shortcut_report = json.loads((panel_dir / SHORTCUT_BASELINE_REPORT_V2_ARTIFACT).read_text(encoding="utf-8"))
                    for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                        self.assertIn(field_name, shortcut_report)
                    self.assertTrue(shortcut_report["passes"])
                    for baseline_name in ("source_only_lookup_v1", "source_context_lookup_v1"):
                        self.assertIn("seedwise", shortcut_report["baselines"][baseline_name])
                        self.assertIn("summary", shortcut_report["baselines"][baseline_name])
                        self.assertIn("passes", shortcut_report["baselines"][baseline_name])
                        self.assertTrue(shortcut_report["baselines"][baseline_name]["passes"])
                        self.assertEqual(len(shortcut_report["baselines"][baseline_name]["seedwise"]), 5)

                    oracle_feasibility = json.loads(
                        (panel_dir / ORACLE_TASK_FEASIBILITY_REPORT_V2_ARTIFACT).read_text(encoding="utf-8")
                    )
                    for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                        self.assertIn(field_name, oracle_feasibility)
                    self.assertTrue(oracle_feasibility["passes"])
                    for check_name in (
                        "oracle_alignment_kl_all_seeds",
                        "latent_contrast_positive_all_seeds",
                        "pooled_contrast_positive_all_seeds",
                        "oracle_symmetry_consistency_all_seeds",
                        "context0_localization_positive_all_seeds",
                        "context1_localization_positive_all_seeds",
                        "block_early_localization_positive_all_seeds",
                        "block_late_localization_positive_all_seeds",
                        "mean_oracle_latent_contrast_floor",
                        "mean_oracle_pooled_contrast_floor",
                        "design_invariants_all_seeds",
                    ):
                        self.assertTrue(oracle_feasibility["checks"][check_name], msg=f"{direction_id}/{seed_panel}: {check_name}")

            self.assertEqual(len(result.run_ids), 20)
            first_run_dir = Path(tmp_dir) / "runs" / result.run_ids[0]
            self.assertTrue((first_run_dir / "probe_design_report.json").exists())
            self.assertTrue((first_run_dir / "probe_table.json").exists())
            self.assertTrue((first_run_dir / "oracle_probe_metrics.json").exists())
            self.assertTrue((first_run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((first_run_dir / "eval" / "oracle_full_trajectories.pt").exists())
            self.assertTrue((first_run_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT).exists())
            self.assertTrue((first_run_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT).exists())
            self.assertFalse((first_run_dir / "probe_metrics.json").exists())
            self.assertFalse((first_run_dir / "eval" / "full_trajectories.pt").exists())

            probe_table_payload = json.loads((first_run_dir / "probe_table.json").read_text(encoding="utf-8"))
            for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                self.assertIn(field_name, probe_table_payload)
            self.assertEqual(probe_table_payload["stage3_task_version"], STAGE3_TASK_VERSION)
            self.assertEqual(len(probe_table_payload["rows"]), 12)

            hidden_state_payload = json.loads((first_run_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT).read_text(encoding="utf-8"))
            for field_name in _REQUIRED_STAGE3_JSON_FIELDS:
                self.assertIn(field_name, hidden_state_payload)
            self.assertEqual(hidden_state_payload["stage3_task_version"], STAGE3_TASK_VERSION)
            self.assertEqual(hidden_state_payload["diagnostic_schema_version"], DIAGNOSTIC_SCHEMA_VERSION)
            self.assertGreater(hidden_state_payload["oracle_probe_expected_target_margin__v2"], 0.0)
            self.assertIsNone(hidden_state_payload["learned_probe_expected_target_margin__v2"])

            with (first_run_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                hidden_rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), HIDDEN_STATE_PROBE_TABLE_FIELDS)
            self.assertEqual(len(hidden_rows), 12)


if __name__ == "__main__":
    unittest.main()
