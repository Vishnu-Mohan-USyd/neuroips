from __future__ import annotations

import unittest

import torch

from lvc_expectation.config import ExperimentConfig
from lvc_expectation.geometry import OrientationGeometry
from lvc_expectation.paradigms import (
    CONDITION_CODES,
    PRESTIM_MODES,
    TASK_MODES,
    Phase1ParadigmGenerator,
    build_transition_matrices,
    compute_neutral_match_report,
    generate_trial_batch,
)
from lvc_expectation.stimuli import orientation_to_population_code


class ParadigmTests(unittest.TestCase):
    def _assert_valid_probe_subset_batch(self, batch: torch.Tensor, expected_sources: list[int]) -> None:
        probe_pair_id = batch.metadata["probe_pair_id"]
        probe_source = batch.metadata["probe_source_orientation"]
        probe_target = batch.metadata["probe_target_orientation"]
        probe_step_mask = batch.metadata["probe_step_mask"]
        probe_expected = batch.metadata["probe_global_expected_mask"][probe_step_mask]
        probe_unexpected = batch.metadata["probe_global_unexpected_mask"][probe_step_mask]

        expected_source_tensor = torch.tensor(expected_sources, dtype=torch.long)
        self.assertTrue(torch.equal(batch.metadata["controlled_sources"], expected_source_tensor))
        self.assertTrue(torch.equal(batch.metadata["probe_source_subset"], expected_source_tensor))
        self.assertEqual(batch.metadata["probe_report"]["probe_source_subset"], expected_sources)
        self.assertTrue(torch.all((batch.context_ids == 0) | (batch.context_ids == 1)))
        for pair_id in torch.unique(probe_pair_id).tolist():
            trial_mask = probe_pair_id.eq(int(pair_id))
            self.assertEqual(int(trial_mask.sum().item()), 2)
            self.assertEqual(set(batch.context_ids[trial_mask].tolist()), {0, 1})
            self.assertEqual(len(set(probe_source[trial_mask].tolist())), 1)
            self.assertEqual(len(set(probe_target[trial_mask].tolist())), 1)
            self.assertEqual(int(probe_expected[trial_mask].sum().item()), 1)
            self.assertEqual(int(probe_unexpected[trial_mask].sum().item()), 1)

    def test_transition_matrices_counterbalance_expectedness(self) -> None:
        geometry = OrientationGeometry(12)
        matrices = build_transition_matrices(geometry, controlled_sources=[0, 1, 2])
        self.assertGreater(matrices[0][0, 1].item(), matrices[0][0, 11].item())
        self.assertGreater(matrices[1][0, 11].item(), matrices[1][0, 1].item())
        self.assertAlmostEqual(matrices[0][5, 6].item(), 0.5)
        self.assertAlmostEqual(matrices[1][5, 4].item(), 0.5)

    def test_generation_is_deterministic(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config, controlled_sources=[0, 1, 2, 3])
        ambiguity = torch.ones((8, config.sequence.n_steps, config.geometry.n_orientations), dtype=torch.float32)
        batch_a = generator.generate_batch(batch_size=8, seed=7, ambiguity_weights=ambiguity)
        batch_b = generator.generate_batch(batch_size=8, seed=7, ambiguity_weights=ambiguity)
        self.assertTrue(batch_a.orientations.equal(batch_b.orientations))
        self.assertTrue(batch_a.blank_mask.equal(batch_b.blank_mask))
        self.assertTrue(batch_a.metadata["expected_distribution"].equal(batch_b.metadata["expected_distribution"]))
        self.assertTrue(batch_a.metadata["controlled_sources"].equal(batch_b.metadata["controlled_sources"]))
        self.assertTrue(batch_a.metadata["ambiguity_weights"].equal(batch_b.metadata["ambiguity_weights"]))

    def test_neutral_report_has_expected_fields(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config, controlled_sources=[0, 1, 2, 3, 4, 5])
        batch = generator.generate_batch(batch_size=16, seed=11)
        report = generator.build_neutral_match_report(batch)
        self.assertIn("neutral", report.condition_counts)
        self.assertTrue(len(report.orientation_counts) > 0)
        self.assertTrue(len(report.transition_counts) > 0)
        self.assertIn("orientation_relevant", report.task_counts)
        self.assertIn("cue_only", report.prestim_counts)
        self.assertGreaterEqual(report.orthogonal_event_count, 0)
        self.assertGreaterEqual(report.controlled_source_count, 0)

    def test_population_code_accepts_optional_ambiguity_weights(self) -> None:
        orientations = torch.tensor([[0, 1, -1]], dtype=torch.long)
        ambiguity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.2, 0.3, 0.5]]],
            dtype=torch.float32,
        )
        code = orientation_to_population_code(orientations, n_orientations=3, ambiguity_weights=ambiguity)
        self.assertTrue(torch.allclose(code[0, 0], ambiguity[0, 0]))
        self.assertTrue(torch.allclose(code[0, 1], ambiguity[0, 1]))
        self.assertTrue(code[0, 2].eq(0.0).all())

    def test_local_global_probe_batch_counterbalances_same_physical_pairs_across_contexts(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config, controlled_sources=[0, 1, 2, 3, 4, 5])
        batch = generator.generate_local_global_probe_batch(seed=19)
        self._assert_valid_probe_subset_batch(batch, [0, 1, 2, 3, 4, 5])

    def test_local_global_probe_batch_accepts_even_source_subset(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config)
        batch = generator.generate_local_global_probe_batch(seed=31, probe_source_subset=[0, 2, 4])
        self._assert_valid_probe_subset_batch(batch, [0, 2, 4])

    def test_local_global_probe_batch_accepts_odd_source_subset(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config)
        batch = generator.generate_local_global_probe_batch(seed=37, probe_source_subset=[1, 3, 5])
        self._assert_valid_probe_subset_batch(batch, [1, 3, 5])

    def test_local_global_probe_batch_has_symmetry_mate_and_fixed_scored_constraints(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config)
        batch = generator.generate_local_global_probe_batch(seed=23)

        probe_step_mask = batch.metadata["probe_step_mask"]
        probe_valid_mask = batch.metadata["probe_valid_mask"]
        self.assertTrue(torch.equal(probe_step_mask, probe_valid_mask))
        self.assertTrue(probe_step_mask.sum(dim=1).eq(1).all())
        self.assertEqual(batch.metadata["probe_visible_step_index"], 1)
        self.assertEqual(batch.metadata["probe_report"]["fixed_probe_visible_step_index"], 1)
        self.assertTrue(batch.task_mode.eq(TASK_MODES["orientation_relevant"]).all())
        self.assertTrue(batch.prestim_mode.eq(PRESTIM_MODES["none"]).all())
        self.assertTrue(batch.blank_mask[probe_step_mask].eq(False).all())
        self.assertTrue(batch.metadata["condition_codes"][probe_step_mask].ne(CONDITION_CODES["omission"]).all())

        pair_descriptors = batch.metadata["probe_report"]["pair_descriptors"]
        descriptor_pairs = {
            (descriptor["source"], descriptor["local_offset_bins"])
            for descriptor in pair_descriptors.values()
        }
        for source in range(6):
            self.assertIn((source, 1), descriptor_pairs)
            self.assertIn((source, -1), descriptor_pairs)

    def test_local_global_probe_defaults_remain_reproducible(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config)
        batch = generator.generate_local_global_probe_batch(seed=41)
        self.assertEqual(batch.metadata["probe_report"]["probe_source_subset"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(batch.metadata["probe_visible_step_index"], 1)
        self.assertEqual(int(batch.metadata["probe_pair_id"].max().item()) + 1, 12)
        self.assertIsNone(batch.metadata["probe_report"]["local_offset_family"])
        self.assertIsNone(batch.metadata["probe_offset_family"])

    def test_local_global_probe_batch_supports_plus_one_offset_family_only(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config)
        batch = generator.generate_local_global_probe_batch(seed=43, local_offset_family=1)

        probe_mask = batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]
        self.assertEqual(int(probe_mask.sum().item()), 12)
        self.assertTrue(batch.metadata["probe_local_offset_bins"].eq(1).all())
        self.assertEqual(int(torch.unique(batch.metadata["probe_pair_id"]).numel()), 6)
        self.assertEqual(int(batch.metadata["probe_global_expected_mask"][probe_mask].sum().item()), 6)
        self.assertEqual(int(batch.metadata["probe_global_unexpected_mask"][probe_mask].sum().item()), 6)
        self.assertEqual(batch.metadata["probe_report"]["local_offset_family"], 1)
        self.assertEqual(int(batch.metadata["probe_offset_family"].item()), 1)

    def test_local_global_probe_batch_supports_minus_one_offset_family_only(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config)
        batch = generator.generate_local_global_probe_batch(seed=47, local_offset_family=-1)

        probe_mask = batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]
        self.assertEqual(int(probe_mask.sum().item()), 12)
        self.assertTrue(batch.metadata["probe_local_offset_bins"].eq(-1).all())
        self.assertEqual(int(torch.unique(batch.metadata["probe_pair_id"]).numel()), 6)
        self.assertEqual(int(batch.metadata["probe_global_expected_mask"][probe_mask].sum().item()), 6)
        self.assertEqual(int(batch.metadata["probe_global_unexpected_mask"][probe_mask].sum().item()), 6)
        self.assertEqual(batch.metadata["probe_report"]["local_offset_family"], -1)
        self.assertEqual(int(batch.metadata["probe_offset_family"].item()), -1)

    def test_standard_generator_path_is_still_additive(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config, controlled_sources=[0, 1, 2, 3])
        batch = generator.generate_batch(batch_size=8, seed=29)
        self.assertEqual(tuple(batch.orientations.shape), (8, config.sequence.n_steps))
        self.assertIn("expected_distribution", batch.metadata)
        self.assertNotIn("probe_step_mask", batch.metadata)


if __name__ == "__main__":
    unittest.main()
