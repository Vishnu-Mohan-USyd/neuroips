"""Tests for HVA distributed population-state prediction validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_full_plasticity.py"
SPEC = importlib.util.spec_from_file_location("validate_full_plasticity", VALIDATOR_PATH)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def make_prediction_row(
    *,
    prediction_index: int,
    frame_index: int,
    target_frame_index: int,
    tile_id: int,
    target: float,
    model: float,
    persistence: float,
    train_mean: float,
    no_learning: float,
    temporal: float,
    spatial: float,
) -> validator.HVAPredictorPredictionRow:
    return validator.HVAPredictorPredictionRow(
        prediction_index=prediction_index,
        repeat_index=0,
        frame_index=frame_index,
        target_frame_index=target_frame_index,
        target_channel_index=0,
        target_channel="l23e",
        tile_id=tile_id,
        split="heldout",
        learning_update_applied=0,
        target_state_norm=target,
        predicted_state_norm=model,
        target_residual_norm=target,
        predicted_residual_norm=model,
        target_residual_z=target,
        predicted_residual_z=model,
        train_residual_mean_norm=0.0,
        train_residual_std_norm=1.0,
        persistence_pred_state_norm=persistence,
        train_mean_pred_state_norm=train_mean,
        no_learning_pred_state_norm=no_learning,
        temporal_block_shift_pred_state_norm=temporal,
        spatial_tile_shuffle_pred_state_norm=spatial,
        target_rate_hz=target,
        predicted_rate_hz=model,
        error_rate_hz=target - model,
    )


def synthetic_rows(*, model_equals_persistence: bool) -> list[validator.HVAPredictorPredictionRow]:
    targets = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    bad_vectors = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rows: list[validator.HVAPredictorPredictionRow] = []
    prediction_index = 0
    for frame_index, target in enumerate(targets):
        bad = bad_vectors[frame_index]
        model = bad if model_equals_persistence else target
        for tile_id, target_value in enumerate(target):
            rows.append(
                make_prediction_row(
                    prediction_index=prediction_index,
                    frame_index=frame_index,
                    target_frame_index=frame_index + 5,
                    tile_id=tile_id,
                    target=target_value,
                    model=model[tile_id],
                    persistence=bad[tile_id],
                    train_mean=bad[tile_id],
                    no_learning=bad[tile_id],
                    temporal=bad[tile_id],
                    spatial=bad[tile_id],
                )
            )
            prediction_index += 1
    return rows


class HVAPopulationStatePredictionTests(unittest.TestCase):
    def test_population_state_metrics_pass_when_model_tracks_target(self) -> None:
        metrics = validator.compute_hva_population_state_prediction_metrics(
            synthetic_rows(model_equals_persistence=False),
            4.0,
        )

        self.assertEqual(metrics["complete_sample_count"], 3.0)
        self.assertAlmostEqual(metrics["model_vector_corr_mean"], 1.0)
        self.assertAlmostEqual(metrics["model_vector_cosine_mean"], 1.0)
        self.assertAlmostEqual(metrics["model_mse_mean"], 0.0)
        self.assertGreater(metrics["model_vs_persistence_vector_corr_delta"], 0.0)
        self.assertGreater(metrics["model_vs_no_learning_mse_delta"], 0.0)
        self.assertTrue(validator.hva_population_state_prediction_passes(metrics))

    def test_population_state_metrics_fail_when_model_equals_persistence(self) -> None:
        metrics = validator.compute_hva_population_state_prediction_metrics(
            synthetic_rows(model_equals_persistence=True),
            4.0,
        )

        self.assertAlmostEqual(metrics["model_vector_corr_mean"], metrics["persistence_vector_corr_mean"])
        self.assertAlmostEqual(metrics["model_vector_cosine_mean"], metrics["no_learning_vector_cosine_mean"])
        self.assertAlmostEqual(metrics["model_mse_mean"], metrics["train_mean_mse_mean"])
        self.assertFalse(validator.hva_population_state_prediction_passes(metrics))


if __name__ == "__main__":
    unittest.main()
