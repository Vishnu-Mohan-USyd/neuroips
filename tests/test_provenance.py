from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from lvc_expectation.provenance import (
    FROZEN_BENCHMARK_REGISTRY,
    FROZEN_METRIC_KEY_DESCRIPTIONS,
    compute_bundle_fingerprint,
    compute_run_fingerprint,
    frozen_benchmark_fingerprints_payload,
    frozen_benchmark_registry_payload,
    frozen_metric_versions_payload,
    verify_frozen_benchmark_registry,
)


class ProvenanceTests(unittest.TestCase):
    def test_run_fingerprint_changes_when_artifact_bytes_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "abc123"
            (run_dir / "eval").mkdir(parents=True)
            (run_dir / "manifest.json").write_text('{"run_id":"abc123"}', encoding="utf-8")
            (run_dir / "eval" / "metrics.json").write_text('{"score":1.0}', encoding="utf-8")

            before = compute_run_fingerprint(run_dir)
            (run_dir / "eval" / "metrics.json").write_text('{"score":2.0}', encoding="utf-8")
            after = compute_run_fingerprint(run_dir)

            self.assertNotEqual(before, after)

    def test_bundle_fingerprint_is_order_invariant_over_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_a = root / "runs" / "a"
            run_b = root / "runs" / "b"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            (run_a / "manifest.json").write_text('{"run_id":"a"}', encoding="utf-8")
            (run_b / "manifest.json").write_text('{"run_id":"b"}', encoding="utf-8")

            forward = compute_bundle_fingerprint(root, ["a", "b"])
            reverse = compute_bundle_fingerprint(root, ["b", "a"])

            self.assertEqual(forward, reverse)

    def test_frozen_benchmark_registry_exposes_required_anchors(self) -> None:
        payload = frozen_benchmark_registry_payload()
        self.assertIn("benchmark_registry_version", payload)
        anchors = payload["anchors"]
        for anchor_id in (
            "phase1_poststim_positive_v1",
            "phase1_prestim_negative_v1",
            "phase15_probe_positive_v1",
            "phase2_p2a_primary_negative_v1",
            "phase2_challenger_selection_negative_v1",
        ):
            self.assertIn(anchor_id, anchors)
            self.assertIn(anchor_id, FROZEN_BENCHMARK_REGISTRY)
            self.assertEqual(len(anchors[anchor_id]["bundle_fingerprint"]), 64)

    def test_corrected_poststim_anchor_uses_contracted_run_ids(self) -> None:
        poststim_anchor = FROZEN_BENCHMARK_REGISTRY["phase1_poststim_positive_v1"]
        self.assertEqual(
            poststim_anchor.run_ids,
            (
                "7ca381644851",
                "c6d4f8c043b9",
                "ff56b2f53b0b",
                "48865dcda306",
                "340c4bbb2344",
            ),
        )

    def test_benchmark_artifact_files_exist_and_match_helper_payloads(self) -> None:
        benchmark_dir = Path("artifacts/benchmarks")
        self.assertTrue((benchmark_dir / "benchmark_registry.v1.json").exists())
        self.assertTrue((benchmark_dir / "benchmark_fingerprints.v1.json").exists())
        self.assertTrue((benchmark_dir / "metric_versions.v1.json").exists())

        registry_payload = json.loads((benchmark_dir / "benchmark_registry.v1.json").read_text(encoding="utf-8"))
        fingerprints_payload = json.loads(
            (benchmark_dir / "benchmark_fingerprints.v1.json").read_text(encoding="utf-8")
        )
        metric_versions_payload = json.loads((benchmark_dir / "metric_versions.v1.json").read_text(encoding="utf-8"))

        self.assertEqual(registry_payload, frozen_benchmark_registry_payload())
        self.assertEqual(fingerprints_payload, frozen_benchmark_fingerprints_payload())
        self.assertEqual(metric_versions_payload, frozen_metric_versions_payload())
        self.assertIn("poststimulus_l23_template_specificity", metric_versions_payload["metric_versions"])
        self.assertIn("probe_target_aligned_specificity_contrast", metric_versions_payload["metric_versions"])
        self.assertNotIn("phase1_poststim_positive_v1", metric_versions_payload["metric_versions"])
        for metric_key, description in FROZEN_METRIC_KEY_DESCRIPTIONS.items():
            self.assertIn(metric_key, metric_versions_payload["metric_versions"])
            metric_entry = metric_versions_payload["metric_versions"][metric_key]
            self.assertEqual(metric_entry["version"], "v1")
            self.assertEqual(metric_entry["benchmark_metric_version"], "benchmark_v1_frozen")
            self.assertEqual(metric_entry["semantic_description"], description)
            self.assertEqual(metric_entry["status"], "frozen_v1_anchor")

    def test_live_benchmark_artifacts_verify_against_frozen_registry(self) -> None:
        verification = verify_frozen_benchmark_registry("artifacts")
        self.assertTrue(all(verification.values()))
        self.assertIn("phase2_p2a_primary_negative_v1", verification)
        self.assertIn("phase2_challenger_selection_negative_v1", verification)


if __name__ == "__main__":
    unittest.main()
