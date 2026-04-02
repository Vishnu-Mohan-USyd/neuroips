from __future__ import annotations

import unittest

import torch

from lvc_expectation.assays import AssayRunner
from lvc_expectation.config import ExperimentConfig, GeometryConfig
from lvc_expectation.types import ContextPrediction, SimulationOutput, TrialBatch


class AssayRunnerTests(unittest.TestCase):
    def test_prestim_specificity_uses_expected_target_distribution_for_blank_steps(self) -> None:
        config = ExperimentConfig(geometry=GeometryConfig(n_orientations=3))
        runner = AssayRunner(config)

        deep_template = torch.tensor(
            [[[0.1, 0.1, 0.8], [0.2, 0.6, 0.2]]],
            dtype=torch.float32,
        )
        simulation = SimulationOutput(
            states={
                "l23_readout": deep_template.clone(),
                "deep_template": deep_template,
            },
            observations={},
            context_prediction=ContextPrediction(
                orientation_logits=torch.tensor(
                    [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
                    dtype=torch.float32,
                ),
                precision_logit=None,
            ),
            metadata={},
        )

        expected_distribution = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        batch = TrialBatch(
            orientations=torch.tensor([[-1, 1]], dtype=torch.long),
            blank_mask=torch.tensor([[True, False]], dtype=torch.bool),
            expected_mask=torch.tensor([[False, False]], dtype=torch.bool),
            context_ids=torch.tensor([0], dtype=torch.long),
            task_mode=torch.tensor([0], dtype=torch.long),
            prestim_mode=torch.tensor([1], dtype=torch.long),
            orthogonal_events=torch.tensor([[0, 0]], dtype=torch.long),
            metadata={
                "condition_codes": torch.tensor([[3, 2]], dtype=torch.long),
                "expected_distribution": expected_distribution,
                "omission_targets": torch.tensor([[-1, -1]], dtype=torch.long),
            },
        )

        metrics = runner.compute_primary_metrics(simulation, batch)

        self.assertAlmostEqual(metrics["prestimulus_template_specificity"], 0.7, places=5)

    def test_prestim_template_gate_reports_intact_and_zero_context_conditions(self) -> None:
        config = ExperimentConfig(geometry=GeometryConfig(n_orientations=3))
        runner = AssayRunner(config)

        intact_template = torch.tensor(
            [
                [[0.1, 0.1, 0.8], [0.2, 0.6, 0.2]],
                [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
                [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]],
            ],
            dtype=torch.float32,
        )
        zero_context_template = torch.full_like(intact_template, 1.0 / 3.0)

        intact_simulation = SimulationOutput(
            states={
                "l23_readout": intact_template.clone(),
                "deep_template": intact_template,
            },
            observations={},
            context_prediction=ContextPrediction(
                orientation_logits=torch.zeros((3, 2, 3), dtype=torch.float32),
                precision_logit=None,
            ),
            metadata={},
        )
        zero_context_simulation = SimulationOutput(
            states={
                "l23_readout": zero_context_template.clone(),
                "deep_template": zero_context_template,
            },
            observations={},
            context_prediction=ContextPrediction(
                orientation_logits=torch.zeros((3, 2, 3), dtype=torch.float32),
                precision_logit=None,
            ),
            metadata={},
        )

        expected_distribution = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        batch = TrialBatch(
            orientations=torch.tensor([[-1, 1], [-1, 0], [-1, 1]], dtype=torch.long),
            blank_mask=torch.tensor([[True, False], [True, False], [True, False]], dtype=torch.bool),
            expected_mask=torch.zeros((3, 2), dtype=torch.bool),
            context_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            task_mode=torch.tensor([0, 0, 0], dtype=torch.long),
            prestim_mode=torch.tensor([1, 2, 1], dtype=torch.long),
            orthogonal_events=torch.zeros((3, 2), dtype=torch.long),
            metadata={
                "condition_codes": torch.tensor([[3, 2], [3, 2], [3, 2]], dtype=torch.long),
                "expected_distribution": expected_distribution,
                "omission_targets": torch.full((3, 2), -1, dtype=torch.long),
            },
        )

        gate = runner.compute_prestim_template_gate(intact_simulation, zero_context_simulation, batch)

        self.assertEqual(gate["control_mode"], "zero_context")
        for control_name in ("intact", "zero_context"):
            self.assertIn("cue_only", gate[control_name])
            self.assertIn("context_only", gate[control_name])
            self.assertIn("neutral", gate[control_name])
        self.assertAlmostEqual(gate["intact"]["cue_only"]["prestimulus_template_specificity"], 0.7, places=5)
        self.assertAlmostEqual(gate["intact"]["context_only"]["prestimulus_template_specificity"], 0.7, places=5)
        self.assertAlmostEqual(gate["intact"]["neutral"]["prestimulus_template_specificity"], 0.7, places=5)
        self.assertAlmostEqual(gate["zero_context"]["cue_only"]["prestimulus_template_specificity"], 0.0, places=5)
        self.assertAlmostEqual(gate["zero_context"]["context_only"]["prestimulus_template_specificity"], 0.0, places=5)
        self.assertAlmostEqual(gate["zero_context"]["neutral"]["prestimulus_template_specificity"], 0.0, places=5)

    def test_poststim_metrics_capture_template_specificity_and_comparator_nonuniformity_by_window(self) -> None:
        config = ExperimentConfig(geometry=GeometryConfig(n_orientations=3))
        runner = AssayRunner(config)

        expected_targets = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]], dtype=torch.long)
        expected_distribution = torch.nn.functional.one_hot(expected_targets, num_classes=3).to(torch.float32)

        responses = torch.tensor(
            [[
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6],
                [0.6, 0.2, 0.2],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]],
            dtype=torch.float32,
        )
        comparator = torch.tensor(
            [[
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
            ]],
            dtype=torch.float32,
        )
        simulation = SimulationOutput(
            states={
                "l23_readout": responses,
                "deep_template": torch.zeros_like(responses),
                "context_comparator": comparator,
            },
            observations={
                "identity": responses.clone(),
                "gaussian_orientation_bank": responses.clone(),
            },
            context_prediction=ContextPrediction(
                orientation_logits=expected_distribution.clone(),
                precision_logit=None,
            ),
            metadata={},
        )

        batch = TrialBatch(
            orientations=expected_targets.clone(),
            blank_mask=torch.zeros((1, 12), dtype=torch.bool),
            expected_mask=torch.zeros((1, 12), dtype=torch.bool),
            context_ids=torch.tensor([0], dtype=torch.long),
            task_mode=torch.tensor([0], dtype=torch.long),
            prestim_mode=torch.tensor([0], dtype=torch.long),
            orthogonal_events=torch.zeros((1, 12), dtype=torch.long),
            metadata={
                "condition_codes": torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
                "expected_distribution": expected_distribution,
                "omission_targets": torch.full((1, 12), -1, dtype=torch.long),
            },
        )

        metrics = runner.compute_primary_metrics(simulation, batch)

        self.assertAlmostEqual(metrics["poststimulus_l23_template_specificity_early"], 0.7, places=5)
        self.assertAlmostEqual(metrics["poststimulus_l23_template_specificity_middle"], 0.4, places=5)
        self.assertAlmostEqual(metrics["poststimulus_l23_template_specificity_late"], 0.0, places=5)
        self.assertAlmostEqual(
            metrics["poststimulus_pooled_template_specificity"],
            metrics["poststimulus_l23_template_specificity"],
            places=5,
        )
        self.assertAlmostEqual(metrics["context_comparator_nonuniformity_early"], 0.0, places=5)
        self.assertAlmostEqual(metrics["context_comparator_nonuniformity_middle"], 0.311111, places=5)
        self.assertAlmostEqual(metrics["context_comparator_nonuniformity_late"], 0.177778, places=5)
        self.assertGreater(metrics["poststimulus_l23_template_specificity"], 0.3)
        self.assertGreater(metrics["context_comparator_nonuniformity"], 0.1)
        self.assertIn("rsa_distance", metrics)
        self.assertIn("poststimulus_pooled_template_specificity", metrics)

    def test_local_global_probe_metrics_emit_expected_keys_from_probe_metadata(self) -> None:
        config = ExperimentConfig(geometry=GeometryConfig(n_orientations=3))
        runner = AssayRunner(config)

        responses = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.6, 0.2, 0.2]],
                [[0.0, 0.0, 0.0], [0.9, 0.05, 0.05]],
                [[0.0, 0.0, 0.0], [0.2, 0.6, 0.2]],
                [[0.0, 0.0, 0.0], [0.05, 0.9, 0.05]],
            ],
            dtype=torch.float32,
        )
        comparator = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]],
                [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1]],
                [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]],
                [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1]],
            ],
            dtype=torch.float32,
        )
        simulation = SimulationOutput(
            states={
                "l23_readout": responses,
                "deep_template": torch.zeros_like(responses),
                "context_comparator": comparator,
            },
            observations={
                "identity": responses.clone(),
                "gaussian_orientation_bank": responses.clone(),
            },
            context_prediction=ContextPrediction(
                orientation_logits=torch.zeros_like(responses),
                precision_logit=None,
            ),
            metadata={},
        )
        batch = TrialBatch(
            orientations=torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.long),
            blank_mask=torch.zeros((4, 2), dtype=torch.bool),
            expected_mask=torch.zeros((4, 2), dtype=torch.bool),
            context_ids=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            task_mode=torch.zeros(4, dtype=torch.long),
            prestim_mode=torch.zeros(4, dtype=torch.long),
            orthogonal_events=torch.zeros((4, 2), dtype=torch.long),
            metadata={
                "condition_codes": torch.tensor([[2, 0], [2, 1], [2, 0], [2, 1]], dtype=torch.long),
                "expected_distribution": torch.nn.functional.one_hot(
                    torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.long),
                    num_classes=3,
                ).to(torch.float32),
                "omission_targets": torch.full((4, 2), -1, dtype=torch.long),
                "probe_step_mask": torch.tensor(
                    [[False, True], [False, True], [False, True], [False, True]],
                    dtype=torch.bool,
                ),
                "probe_valid_mask": torch.tensor(
                    [[False, True], [False, True], [False, True], [False, True]],
                    dtype=torch.bool,
                ),
                "probe_target_orientation": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "probe_pair_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "probe_global_expected_mask": torch.tensor(
                    [[False, True], [False, False], [False, True], [False, False]],
                    dtype=torch.bool,
                ),
                "probe_global_unexpected_mask": torch.tensor(
                    [[False, False], [False, True], [False, False], [False, True]],
                    dtype=torch.bool,
                ),
            },
        )

        metrics = runner.compute_local_global_probe_metrics(simulation, batch)

        self.assertEqual(metrics["n_probe_rows"], 4)
        self.assertEqual(metrics["n_probe_pairs_scored"], 2)
        self.assertEqual(metrics["probe_pair_ids_scored"], [0, 1])
        self.assertAlmostEqual(metrics["probe_target_aligned_specificity_contrast"], -0.45, places=5)
        self.assertAlmostEqual(metrics["probe_pooled_target_aligned_specificity_contrast"], -0.45, places=5)
        self.assertAlmostEqual(metrics["probe_context_comparator_nonuniformity_contrast"], -0.244444, places=5)
        self.assertIn("rsa_distance", metrics)

    def test_pooled_target_specificity_and_rsa_distance_can_diverge(self) -> None:
        config = ExperimentConfig(geometry=GeometryConfig(n_orientations=3))
        runner = AssayRunner(config)

        targets = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
        aligned = torch.nn.functional.one_hot(targets, num_classes=3).to(torch.float32)
        expected_distribution = aligned.clone()
        condition_codes = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)

        simulation = SimulationOutput(
            states={
                "l23_readout": aligned.clone(),
                "deep_template": torch.zeros_like(aligned),
                "context_comparator": torch.zeros_like(aligned),
            },
            observations={
                "identity": aligned.clone(),
                "gaussian_orientation_bank": aligned.clone(),
            },
            context_prediction=ContextPrediction(
                orientation_logits=expected_distribution.clone(),
                precision_logit=None,
            ),
            metadata={},
        )
        batch = TrialBatch(
            orientations=targets.clone(),
            blank_mask=torch.zeros((1, 6), dtype=torch.bool),
            expected_mask=torch.zeros((1, 6), dtype=torch.bool),
            context_ids=torch.tensor([0], dtype=torch.long),
            task_mode=torch.tensor([0], dtype=torch.long),
            prestim_mode=torch.tensor([0], dtype=torch.long),
            orthogonal_events=torch.zeros((1, 6), dtype=torch.long),
            metadata={
                "condition_codes": condition_codes,
                "expected_distribution": expected_distribution,
                "omission_targets": torch.full((1, 6), -1, dtype=torch.long),
            },
        )

        metrics = runner.compute_primary_metrics(simulation, batch)

        self.assertAlmostEqual(metrics["rsa_distance"], 0.0, places=6)
        self.assertGreater(metrics["poststimulus_pooled_template_specificity"], 0.9)


if __name__ == "__main__":
    unittest.main()
