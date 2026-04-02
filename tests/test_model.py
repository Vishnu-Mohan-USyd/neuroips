from __future__ import annotations

import unittest

import torch

from lvc_expectation.config import ExperimentConfig
from lvc_expectation.context import CausalContextPredictor
from lvc_expectation.eval import evaluate_with_observations
from lvc_expectation.model import V1ExpectationModel
from lvc_expectation.paradigms import generate_trial_batch


class ModelTests(unittest.TestCase):
    def test_rollout_retains_full_trajectories(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=3, seed=9)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        model = V1ExpectationModel(config)
        for subcase in (
            "adaptation_only",
            "context_global_gain",
            "dampening",
            "sharpening",
            "center_surround",
        ):
            simulation = model.rollout(batch, prediction, subcase=subcase)
            self.assertEqual(tuple(simulation.states["l23_readout"].shape), (3, config.sequence.n_steps, config.geometry.n_orientations))
            self.assertEqual(tuple(simulation.context_predictions.shape), (3, config.sequence.n_steps, config.geometry.n_orientations))

    def test_adaptation_only_removes_context_comparator_effects(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=2, seed=11)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        model = V1ExpectationModel(config)
        simulation = model.rollout(batch, prediction, subcase="adaptation_only")
        comparator = simulation.states["context_comparator"]
        self.assertTrue(torch.equal(comparator, torch.zeros_like(comparator)))

    def test_context_global_gain_comparator_is_uniform_across_channels(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=2, seed=10)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        model = V1ExpectationModel(config)
        simulation = model.rollout(batch, prediction, subcase="context_global_gain")
        comparator = simulation.states["context_comparator"]
        self.assertTrue(torch.allclose(comparator, comparator[..., :1].expand_as(comparator)))

    def test_evaluation_adds_observations_and_window_summaries(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=2, seed=12)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        model = V1ExpectationModel(config)
        simulation = model.rollout(batch, prediction, subcase="center_surround")
        evaluated, summary = evaluate_with_observations(simulation, config)
        self.assertIn("identity", evaluated.observations)
        self.assertIn("gaussian_orientation_bank", evaluated.observations)
        self.assertIn("early", summary.summaries)
        self.assertEqual(tuple(evaluated.observations["identity"].shape), (2, config.sequence.n_steps, config.geometry.n_orientations))


if __name__ == "__main__":
    unittest.main()
