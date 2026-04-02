from __future__ import annotations

from pathlib import Path
import csv
import json
import tempfile
import unittest

import torch

from lvc_expectation.diagnostics import (
    HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
    HIDDEN_STATE_PROBE_TABLE_FIELDS,
    HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT,
    build_hidden_state_diagnostics_v2,
    build_hidden_state_probe_table_rows_from_payload,
)
from lvc_expectation.reporting import write_hidden_state_diagnostic_artifacts_v2


class DiagnosticsTests(unittest.TestCase):
    def test_full_probe_run_writes_required_v2_hidden_state_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "phase15_positive"
            result = write_hidden_state_diagnostic_artifacts_v2(
                Path("artifacts/runs/4fee7e5512c6"),
                output_dir=output_dir,
                benchmark_anchor_id="phase15_probe_positive_v1",
            )

            self.assertTrue((output_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT).exists())
            self.assertTrue((output_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT).exists())

            with (output_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertGreater(len(rows), 0)
            self.assertEqual(tuple(reader.fieldnames or ()), HIDDEN_STATE_PROBE_TABLE_FIELDS)
            first_row = rows[0]
            self.assertIn(first_row["context_bin__v2"], ("context0", "context1"))
            self.assertIn(first_row["block_position_bin__v2"], ("early", "late"))
            self.assertNotEqual(first_row["l23_target_specificity__v2"], "")
            self.assertNotEqual(first_row["pooled_target_specificity__v2"], "")
            self.assertNotEqual(first_row["context_comparator_nonuniformity__v2"], "")

            diagnostics_payload = json.loads((output_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT).read_text())
            for key in (
                "metric_schema_version",
                "metric_versions",
                "classification_rule_version",
                "source_metric_versions",
                "diagnostic_schema_version",
                "benchmark_registry_version",
                "learned_probe_expected_target_margin__v2",
                "probe_correct_pair_flip_rate__v2",
                "probe_target_aligned_specificity_contrast__context0_v2",
                "probe_target_aligned_specificity_contrast__context1_v2",
            ):
                self.assertIn(key, diagnostics_payload)
            self.assertNotIn("learned_probe_pair_flip_rate__legacy_v1", diagnostics_payload)
            self.assertEqual(diagnostics_payload["benchmark_anchor_id"], "phase15_probe_positive_v1")
            self.assertGreater(diagnostics_payload["probe_correct_pair_flip_rate__v2"], 0.9)

    def test_alignment_only_run_keeps_state_fields_empty_but_emits_v2_prediction_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "challenger_alignment"
            result = write_hidden_state_diagnostic_artifacts_v2(
                Path("artifacts/runs/3004568f8330"),
                output_dir=output_dir,
                benchmark_anchor_id="phase2_challenger_selection_negative_v1",
            )

            self.assertTrue((output_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT).exists())
            self.assertTrue((output_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT).exists())

            with (output_dir / HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                first_row = next(iter(reader))

            self.assertEqual(first_row["l23_target_specificity__v2"], "")
            self.assertEqual(first_row["pooled_target_specificity__v2"], "")
            self.assertEqual(first_row["context_comparator_nonuniformity__v2"], "")

            diagnostics_payload = json.loads((output_dir / HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT).read_text())
            self.assertNotIn("learned_probe_pair_flip_rate__legacy_v1", diagnostics_payload)
            self.assertIsNone(diagnostics_payload["probe_target_aligned_specificity_contrast__context0_v2"])
            self.assertIsNone(diagnostics_payload["probe_target_aligned_specificity_contrast__context1_v2"])
            self.assertGreaterEqual(diagnostics_payload["learned_probe_expected_target_confidence__v2"], 0.0)
            self.assertLess(diagnostics_payload["probe_correct_pair_flip_rate__v2"], 0.9)

    def test_oracle_only_payload_keeps_oracle_v2_metrics_without_fabricating_learned_values(self) -> None:
        heldout_batch = torch.load(Path("artifacts/runs/4fee7e5512c6/eval/heldout_batch.pt"))
        oracle_payload = torch.load(Path("artifacts/runs/4fee7e5512c6/eval/oracle_full_trajectories.pt"))["dampening"]
        rows = build_hidden_state_probe_table_rows_from_payload(
            {
                "run_id": "oracle_only_probe",
                "batch": heldout_batch,
                "learned_logits": None,
                "oracle_logits": oracle_payload["context_predictions"],
                "learned_precision": None,
                "oracle_precision": oracle_payload["precision"],
                "l23_readout": oracle_payload["states"]["l23_readout"],
                "gaussian_orientation_bank": oracle_payload["observations"]["gaussian_orientation_bank"],
                "context_comparator": oracle_payload["states"]["context_comparator"],
            }
        )
        diagnostics_payload = build_hidden_state_diagnostics_v2(rows)

        self.assertGreater(len(rows), 0)
        self.assertIsNone(rows[0]["learned_expected_target_margin__v2"])
        self.assertGreater(rows[0]["oracle_expected_target_margin__v2"], 0.0)
        self.assertIsNone(diagnostics_payload["learned_probe_expected_target_margin__v2"])
        self.assertIsNone(diagnostics_payload["probe_correct_pair_flip_rate__v2"])
        self.assertGreater(diagnostics_payload["oracle_probe_expected_target_margin__v2"], 0.0)
        self.assertGreater(diagnostics_payload["probe_target_aligned_specificity_contrast__context0_v2"], 0.0)
        self.assertGreater(diagnostics_payload["probe_target_aligned_specificity_contrast__context1_v2"], 0.0)


if __name__ == "__main__":
    unittest.main()
