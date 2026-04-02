from __future__ import annotations

import unittest

import torch

from lvc_expectation.config import ExperimentConfig
from lvc_expectation.context import CausalContextPredictor
from lvc_expectation.paradigms import generate_trial_batch


class ContextTests(unittest.TestCase):
    def test_context_predictor_outputs_expected_shapes(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=4, seed=3)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        self.assertEqual(tuple(prediction.orientation_logits.shape), (4, config.sequence.n_steps, config.geometry.n_orientations))
        self.assertEqual(tuple(prediction.precision_logit.shape), (4, config.sequence.n_steps, 1))

    def test_context_predictor_is_causally_masked(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=2, seed=5)
        predictor = CausalContextPredictor(config.geometry, config.context)
        baseline = predictor(batch)
        mutated = generate_trial_batch(config, batch_size=2, seed=5)
        mutated.orientations[:, 4] = (mutated.orientations[:, 4].clamp_min(0) + 1) % config.geometry.n_orientations
        mutated_prediction = predictor(mutated)
        self.assertTrue(torch.allclose(baseline.orientation_logits[:, 4], mutated_prediction.orientation_logits[:, 4]))


if __name__ == "__main__":
    unittest.main()
