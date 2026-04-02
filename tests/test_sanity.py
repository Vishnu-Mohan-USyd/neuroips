from __future__ import annotations

import unittest

import torch

from lvc_expectation.config import ExperimentConfig
from lvc_expectation.metrics import (
    decoder_accuracy,
    mean_response_delta,
    omission_specificity,
    prestimulus_template_specificity,
    trajectory_mean_response_delta,
)
from lvc_expectation.paradigms import Phase1ParadigmGenerator
from lvc_expectation.sanity import (
    nuisance_only_decoder,
    predictive_structure_decoder,
    run_nuisance_only_failure_test,
    run_predictive_structure_success_test,
)


class SanityTests(unittest.TestCase):
    def test_sanity_checks_split_nuisance_and_predictive_structure(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config, controlled_sources=[0, 1, 2, 3, 4, 5])
        batch = generator.generate_batch(batch_size=128, seed=13)
        nuisance = run_nuisance_only_failure_test(batch)
        predictive = run_predictive_structure_success_test(batch)
        self.assertLess(nuisance.accuracy, 0.5)
        self.assertGreater(predictive.accuracy, 0.8)
        self.assertEqual(nuisance.n_examples, predictive.n_examples)
        self.assertEqual(nuisance.accuracy, nuisance_only_decoder(batch).accuracy)
        self.assertEqual(predictive.accuracy, predictive_structure_decoder(batch).accuracy)

    def test_sanity_split_holds_on_tranche_heldout_seeds(self) -> None:
        config = ExperimentConfig()
        generator = Phase1ParadigmGenerator(config, controlled_sources=[0, 1, 2, 3, 4, 5])
        for seed in (10101, 10202, 10303, 10404, 10505):
            batch = generator.generate_batch(batch_size=config.training.batch_size, seed=seed)
            nuisance = run_nuisance_only_failure_test(batch)
            predictive = run_predictive_structure_success_test(batch)
            self.assertLess(nuisance.accuracy, 0.5)
            self.assertGreater(predictive.accuracy, 0.8)

    def test_metric_helpers_are_deterministic_on_controlled_inputs(self) -> None:
        expected = torch.tensor([[0.1, 0.2], [0.0, 0.1]], dtype=torch.float32)
        unexpected = torch.tensor([[0.6, 0.7], [0.5, 0.8]], dtype=torch.float32)
        self.assertGreater(mean_response_delta(expected, unexpected).item(), 0.0)

        activity = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        self.assertGreater(prestimulus_template_specificity(activity, targets).mean().item(), 0.0)
        self.assertGreater(omission_specificity(activity, targets).item(), 0.0)

        logits = torch.tensor([[3.0, 1.0], [0.5, 2.5]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        self.assertEqual(decoder_accuracy(logits, labels).item(), 1.0)

        trajectories = torch.tensor([[0.1, 0.3, 0.5, 0.7]], dtype=torch.float32)
        expected_mask = torch.tensor([[True, True, False, False]])
        unexpected_mask = torch.tensor([[False, False, True, True]])
        self.assertGreater(trajectory_mean_response_delta(trajectories, expected_mask, unexpected_mask).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
