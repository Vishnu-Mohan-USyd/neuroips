from __future__ import annotations

from pathlib import Path
import unittest

from lvc_expectation.config import ExperimentConfig, load_config
from lvc_expectation.io import RunStore
from lvc_expectation.provenance import (
    ARTIFACT_SCHEMA_VERSION,
    BENCHMARK_REGISTRY_VERSION,
    CONTRACT_VERSION,
    METRIC_SCHEMA_VERSION,
)
from lvc_expectation.registry import get_preset, list_presets


class ConfigContractTests(unittest.TestCase):
    def test_default_contract_matches_phase1_plan(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.geometry.n_orientations, 12)
        self.assertEqual(config.geometry.periodicity_deg, 180.0)
        self.assertEqual(config.sequence.n_steps, 12)
        self.assertTrue(config.artifacts.save_full_trajectories)
        self.assertEqual(config.observation.schemes, ("identity", "gaussian_orientation_bank"))
        self.assertEqual(config.training.predictive_objective, "next_orientation_distribution")
        self.assertTrue(config.training.heldout_assays_only)
        self.assertTrue(config.context.causal_mask)
        self.assertTrue(config.assays.heldout_evaluation)

    def test_registry_exposes_phase1_core(self) -> None:
        self.assertIn("phase1_core", list_presets())
        preset = get_preset("phase1_core")
        self.assertEqual(preset.name, "phase1_core")

    def test_load_config_round_trip(self) -> None:
        tmp_dir = Path("tests/_tmp")
        tmp_dir.mkdir(exist_ok=True)
        config_path = tmp_dir / "config.yml"
        config_path.write_text("name: custom\ngeometry:\n  n_orientations: 24\nsequence:\n  n_steps: 16\n", encoding="utf-8")
        config = load_config(config_path)
        self.assertEqual(config.name, "custom")
        self.assertEqual(config.geometry.n_orientations, 24)
        self.assertEqual(config.sequence.n_steps, 16)
        config_path.unlink()
        tmp_dir.rmdir()

    def test_periodicity_guard_rejects_non_orientation_geometry(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(geometry=type(ExperimentConfig().geometry)(periodicity_deg=360.0))

    def test_generic_orientation_and_time_contract_can_change(self) -> None:
        config = ExperimentConfig(
            geometry=type(ExperimentConfig().geometry)(n_orientations=24),
            sequence=type(ExperimentConfig().sequence)(n_steps=16, prestim_steps=3),
        )
        self.assertEqual(config.geometry.n_orientations, 24)
        self.assertEqual(config.geometry.bin_width_deg, 7.5)
        self.assertEqual(config.sequence.n_steps, 16)

    def test_to_dict_records_predictive_only_contract(self) -> None:
        payload = ExperimentConfig().to_dict()
        self.assertEqual(payload["training"]["predictive_objective"], "next_orientation_distribution")
        self.assertTrue(payload["training"]["heldout_assays_only"])
        self.assertTrue(payload["assays"]["heldout_evaluation"])
        self.assertEqual(payload["windows"]["early"], (1, 3))

    def test_run_store_persists_manifest_metadata(self) -> None:
        root = Path("tests/_artifact_tmp")
        store = RunStore(root)
        manifest, run_dir = store.create_run(
            "phase1_core",
            train_objectives=["next_orientation_distribution"],
            heldout_assays=["expected_unexpected_neutral"],
            notes={"scaffold": "config_only"},
        )
        loaded = store.load_manifest(run_dir)
        self.assertEqual(loaded.run_id, manifest.run_id)
        self.assertEqual(loaded.contract_version, CONTRACT_VERSION)
        self.assertEqual(loaded.artifact_schema_version, ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(loaded.metric_schema_version, METRIC_SCHEMA_VERSION)
        self.assertEqual(loaded.benchmark_registry_version, BENCHMARK_REGISTRY_VERSION)
        self.assertEqual(loaded.train_objectives, ["next_orientation_distribution"])
        self.assertEqual(loaded.heldout_assays, ["expected_unexpected_neutral"])
        self.assertEqual(loaded.lineage["parent_run_ids"], [])
        self.assertEqual(loaded.lineage["child_run_ids"], [])
        self.assertEqual(loaded.lineage["benchmark_anchor_refs"], [])
        self.assertEqual(loaded.notes["scaffold"], "config_only")
        store.write_provenance(run_dir, manifest, resolved_config=ExperimentConfig().to_dict())
        manifest_path = run_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertTrue((run_dir / "resolved_config.json").exists())
        self.assertTrue((run_dir / "environment.json").exists())
        self.assertTrue((run_dir / "run_fingerprint.json").exists())
        (run_dir / "resolved_config.json").unlink()
        (run_dir / "environment.json").unlink()
        (run_dir / "run_fingerprint.json").unlink()
        manifest_path.unlink()
        (run_dir / "eval").rmdir()
        (run_dir / "ablations").rmdir()
        run_dir.rmdir()
        (root / "runs").rmdir()
        root.rmdir()


if __name__ == "__main__":
    unittest.main()
