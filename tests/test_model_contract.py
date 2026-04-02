from __future__ import annotations

import unittest

from lvc_expectation.config import ExperimentConfig
from lvc_expectation.context import CausalContextPredictor
from lvc_expectation.model import V1ExpectationModel
from lvc_expectation.paradigms import generate_trial_batch


class ModelContractTests(unittest.TestCase):
    def test_oracle_and_learned_context_share_prediction_shape_contract(self) -> None:
        config = ExperimentConfig()
        batch = generate_trial_batch(config, batch_size=3, seed=23)
        predictor = CausalContextPredictor(config.geometry, config.context)
        prediction = predictor(batch)
        model = V1ExpectationModel(config)
        output_a = model.rollout(batch, prediction, subcase="dampening")
        output_b = model.rollout(batch, prediction, subcase="center_surround")
        self.assertEqual(output_a.context_predictions.shape, output_b.context_predictions.shape)


if __name__ == "__main__":
    unittest.main()
