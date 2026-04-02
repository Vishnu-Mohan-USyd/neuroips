from __future__ import annotations

import unittest

import torch

from lvc_expectation.config import ExperimentConfig
from lvc_expectation.context import CausalContextPredictor
from lvc_expectation.eval import evaluate_with_observations
from lvc_expectation.model import V1ExpectationModel
from lvc_expectation.paradigms import generate_trial_batch
from lvc_expectation.train import (
    EXPECTED_DISTRIBUTION_OBJECTIVE,
    evaluate_context_predictor,
    predictive_loss,
    train_context_predictor,
)
from lvc_expectation.types import ContextPrediction, TrialBatch


class TrainingContractTests(unittest.TestCase):
    def test_predictive_loss_uses_orientation_targets_only(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=8, seed=31)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        result = predictive_loss(prediction, batch)
        self.assertGreater(result.predictive_loss.item(), 0.0)
        self.assertEqual(result.energy_penalty.item(), 0.0)
        self.assertEqual(result.homeostasis_penalty.item(), 0.0)

    def test_soft_target_predictive_loss_reads_expected_distribution_and_ignores_blank_steps(self) -> None:
        logits = torch.tensor(
            [[[0.0, 8.0, 0.0], [8.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        prediction = ContextPrediction(
            orientation_logits=logits,
            precision_logit=None,
        )
        batch = TrialBatch(
            orientations=torch.tensor([[0, -1]], dtype=torch.long),
            blank_mask=torch.tensor([[False, True]], dtype=torch.bool),
            expected_mask=torch.zeros((1, 2), dtype=torch.bool),
            context_ids=torch.tensor([0], dtype=torch.long),
            task_mode=torch.tensor([0], dtype=torch.long),
            prestim_mode=torch.tensor([0], dtype=torch.long),
            orthogonal_events=torch.zeros((1, 2), dtype=torch.long),
            metadata={
                "expected_distribution": torch.tensor(
                    [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                    dtype=torch.float32,
                ),
            },
        )

        sampled_label_loss = predictive_loss(prediction, batch)
        soft_target_loss = predictive_loss(
            prediction,
            batch,
            objective_mode=EXPECTED_DISTRIBUTION_OBJECTIVE,
        )
        expected_first_step_loss = -torch.log_softmax(logits[0, 0], dim=-1)[1]

        self.assertAlmostEqual(
            soft_target_loss.predictive_loss.item(),
            expected_first_step_loss.item(),
            places=6,
        )
        self.assertLess(soft_target_loss.predictive_loss.item(), sampled_label_loss.predictive_loss.item())

    def test_soft_target_predictive_loss_accepts_probe_step_mask(self) -> None:
        logits = torch.tensor(
            [[[6.0, 0.0, 0.0], [0.0, 7.0, 0.0]]],
            dtype=torch.float32,
        )
        prediction = ContextPrediction(
            orientation_logits=logits,
            precision_logit=None,
        )
        batch = TrialBatch(
            orientations=torch.tensor([[0, 1]], dtype=torch.long),
            blank_mask=torch.tensor([[False, False]], dtype=torch.bool),
            expected_mask=torch.zeros((1, 2), dtype=torch.bool),
            context_ids=torch.tensor([0], dtype=torch.long),
            task_mode=torch.tensor([0], dtype=torch.long),
            prestim_mode=torch.tensor([0], dtype=torch.long),
            orthogonal_events=torch.zeros((1, 2), dtype=torch.long),
            metadata={
                "expected_distribution": torch.tensor(
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    dtype=torch.float32,
                ),
                "probe_step_mask": torch.tensor([[False, True]], dtype=torch.bool),
                "probe_valid_mask": torch.tensor([[False, True]], dtype=torch.bool),
            },
        )

        probe_step_mask = batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]
        masked_loss = predictive_loss(
            prediction,
            batch,
            objective_mode=EXPECTED_DISTRIBUTION_OBJECTIVE,
            loss_mask=probe_step_mask,
        )
        expected_probe_step_loss = -torch.log_softmax(logits[0, 1], dim=-1)[1]

        self.assertAlmostEqual(
            masked_loss.predictive_loss.item(),
            expected_probe_step_loss.item(),
            places=6,
        )

    def test_window_summaries_are_derived_from_full_trajectories(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=8, seed=32)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        model = V1ExpectationModel(config)
        simulation = model.rollout(batch, prediction, subcase="dampening")
        evaluated, summary = evaluate_with_observations(simulation, config)
        self.assertEqual(evaluated.states["l23_readout"].shape[1], config.sequence.n_steps)
        self.assertEqual(tuple(summary.summaries["late"]["l23_readout"].shape), (8, config.geometry.n_orientations))

    def test_optimizer_loop_and_heldout_evaluation_are_separate(self) -> None:
        config = ExperimentConfig(
            training=type(ExperimentConfig().training)(n_epochs=2, batch_size=4),
        )
        train_batches = [
            generate_trial_batch(config, batch_size=4, seed=41),
            generate_trial_batch(config, batch_size=4, seed=42),
        ]
        heldout_batch = generate_trial_batch(config, batch_size=4, seed=99)
        predictor = CausalContextPredictor(config.geometry, config.context)

        history = train_context_predictor(predictor, train_batches, config.training)
        heldout = evaluate_context_predictor(predictor, heldout_batch)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].epoch, 1)
        self.assertGreater(history[0].predictive_loss, 0.0)
        self.assertGreater(heldout["predictive_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
