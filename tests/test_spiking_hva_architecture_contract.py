"""Tests for the spiking-HVA architecture source contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_spiking_hva_predictor.py"
SPEC = importlib.util.spec_from_file_location("validate_spiking_hva_predictor", VALIDATOR_PATH)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def base_summary() -> dict[str, float]:
    return {
        "spiking_hva_enabled": 1.0,
        "spiking_hva_scaffold_only": 0.0,
        "spiking_hva_prediction_learning_enabled": 1.0,
        "spiking_hva_feedback_to_v1_enabled": 0.0,
        "spiking_hva_hva_to_v1_connection_count": 0.0,
        "spiking_hva_hva_to_v1_current_enabled": 0.0,
        "spiking_hva_external_v1_input_l23e_only": 1.0,
        "spiking_hva_uses_l4_input": 0.0,
        "spiking_hva_uses_future_features": 0.0,
        "spiking_hva_heldout_updates_applied": 0.0,
        "spiking_hva_predictor_no_future_features_at_prediction": 1.0,
        "spiking_hva_predictor_heldout_update_count": 0.0,
        "spiking_hva_predictor_feedback_to_v1_enabled": 0.0,
        "spiking_hva_predictor_hva_to_v1_connection_count": 0.0,
        "spiking_hva_predictor_lower_v1_mutation_enabled": 0.0,
    }


class SpikingHVAArchitectureContractTests(unittest.TestCase):
    def test_current_direct_l23_readout_style_summary_fails(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_recorded_l23e_hva_e_counts_only": 1.0,
                "spiking_hva_predictor_dale_separated_ei_branches": 1.0,
                "spiking_hva_predictor_suppressive_branch_nonnegative": 1.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "explicit HVA spikes/state"):
            validator.validate_architecture_contract(summary)

    def test_direct_source_flag_fails_even_with_explicit_hva_flag(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_spikes": 1.0,
                "spiking_hva_predictor_prediction_source_direct_l23_readout": 1.0,
                "spiking_hva_predictor_prediction_source_direct_l23e_tile_rates": 1.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "disallowed direct/readout"):
            validator.validate_architecture_contract(summary)

    def test_host_softmax_listwise_readout_flag_fails(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_spikes": 1.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
                "spiking_hva_predictor_host_softmax_listwise_readout_enabled": 1.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "disallowed direct/readout"):
            validator.validate_architecture_contract(summary)

    def test_raw_hva_source_without_prediction_state_fails(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_spikes": 1.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 0.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "explicit_hva_prediction_population"):
            validator.validate_architecture_contract(summary)

    def test_hva_prediction_standin_fails_even_with_population_flag(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_spikes": 1.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
                "spiking_hva_predictor_hva_e_to_prediction_local_synapse_standin": 1.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "disallowed direct/readout"):
            validator.validate_architecture_contract(summary)

    def test_host_reconstructed_multitimescale_primary_fails(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_state": 1.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
                "spiking_hva_predictor_multi_timescale_state_primary_prediction": 1.0,
                "spiking_hva_predictor_multi_timescale_state_host_side_reconstruction": 1.0,
                "spiking_hva_predictor_multi_timescale_state_actual_genn_state": 0.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "host-side reconstruction"):
            validator.validate_architecture_contract(summary)

    def test_multitimescale_primary_requires_actual_genn_state(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_state": 1.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
                "spiking_hva_predictor_multi_timescale_state_primary_prediction": 1.0,
                "spiking_hva_predictor_multi_timescale_state_host_side_reconstruction": 0.0,
                "spiking_hva_predictor_multi_timescale_state_actual_genn_state": 0.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "actual GeNN state"):
            validator.validate_architecture_contract(summary)

    def test_synthetic_proper_spiking_hva_summary_passes(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_spikes": 1.0,
                "spiking_hva_predictor_prediction_source_direct_l23_readout": 0.0,
                "spiking_hva_predictor_prediction_source_direct_l23e_tile_rates": 0.0,
                "spiking_hva_predictor_prediction_source_host_readout": 0.0,
                "spiking_hva_predictor_prediction_source_algorithmic_ei_readout": 0.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
                "spiking_hva_predictor_hva_e_to_prediction_local_synapse_standin": 0.0,
            }
        )

        validator.validate_no_cheat(summary)
        validator.validate_architecture_contract(summary)

    def test_synthetic_actual_genn_multitimescale_summary_passes(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_predictor_prediction_source_explicit_hva_state": 1.0,
                "spiking_hva_predictor_prediction_source_direct_l23_readout": 0.0,
                "spiking_hva_predictor_prediction_source_host_readout": 0.0,
                "spiking_hva_predictor_prediction_source_algorithmic_ei_readout": 0.0,
                "spiking_hva_predictor_explicit_hva_synaptic_learning_enabled": 1.0,
                "spiking_hva_predictor_explicit_hva_prediction_population": 1.0,
                "spiking_hva_predictor_hva_e_to_prediction_local_synapse_standin": 0.0,
                "spiking_hva_predictor_multi_timescale_state_primary_prediction": 1.0,
                "spiking_hva_predictor_multi_timescale_state_host_side_reconstruction": 0.0,
                "spiking_hva_predictor_multi_timescale_state_actual_genn_state": 1.0,
            }
        )

        validator.validate_architecture_contract(summary)

    def test_ctx_transition_runaway_mean_rate_fails_safety(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_ctx_transition_enabled": 1.0,
                "spiking_hva_predictor_hva_ctx_transition_state_source_export_enabled": 1.0,
                "spiking_hva_ctx_transition_prediction_hva_ctx_mean_rate_hz": 25.0,
                "spiking_hva_ctx_transition_transition_state_p99_norm": 1.0,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "mean_rate_hz"):
            validator.validate_physiology_safety(summary)

    def test_ctx_transition_saturated_state_fails_safety(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_ctx_transition_enabled": 1.0,
                "spiking_hva_predictor_hva_ctx_transition_state_source_export_enabled": 1.0,
                "spiking_hva_ctx_transition_prediction_hva_ctx_mean_rate_hz": 5.0,
                "spiking_hva_ctx_transition_transition_state_p99_norm": 1.95,
            }
        )

        with self.assertRaisesRegex(validator.ValidationError, "transition_state_p99"):
            validator.validate_physiology_safety(summary)

    def test_ctx_transition_bounded_state_passes_safety(self) -> None:
        summary = base_summary()
        summary.update(
            {
                "spiking_hva_ctx_transition_enabled": 1.0,
                "spiking_hva_predictor_hva_ctx_transition_state_source_export_enabled": 1.0,
                "spiking_hva_ctx_transition_prediction_hva_ctx_mean_rate_hz": 5.0,
                "spiking_hva_ctx_transition_transition_state_p99_norm": 1.0,
            }
        )

        validator.validate_physiology_safety(summary)

    def test_prediction_gate_rejects_train_mean_only_model(self) -> None:
        targets = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        train_mean = [0.25, 0.25, 0.25, 0.25]
        samples = [
            {
                "target": target,
                "model": train_mean,
                "persistence": [0.0, 0.0, 0.0, 1.0],
                "train_mean": train_mean,
                "no_learning": [0.0, 0.0, 0.0, 0.0],
                "temporal_shuffle": [0.0, 0.0, 1.0, 0.0],
                "spatial_shuffle": [0.0, 0.0, 0.0, 1.0],
            }
            for target in targets
        ]
        metrics = validator.compute_metrics(samples, top_k=2)

        with self.assertRaisesRegex(validator.ValidationError, "train_mean"):
            validator.validate_prediction_gates(
                metrics,
                min_corr=0.70,
                min_delta=0.05,
                min_train_mean_delta=0.03,
            )

    def test_prediction_gate_accepts_model_that_beats_train_mean(self) -> None:
        targets = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        train_mean = [0.25, 0.25, 0.25, 0.25]
        samples = [
            {
                "target": target,
                "model": target,
                "persistence": [0.0, 0.0, 0.0, 1.0],
                "train_mean": train_mean,
                "no_learning": [0.0, 0.0, 0.0, 0.0],
                "temporal_shuffle": [0.0, 0.0, 1.0, 0.0],
                "spatial_shuffle": [0.0, 0.0, 0.0, 1.0],
            }
            for target in targets
        ]
        metrics = validator.compute_metrics(samples, top_k=2)

        validator.validate_prediction_gates(
            metrics,
            min_corr=0.70,
            min_delta=0.05,
            min_train_mean_delta=0.03,
        )


if __name__ == "__main__":
    unittest.main()
