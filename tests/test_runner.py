from __future__ import annotations

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import lvc_expectation.runner as runner_module
from lvc_expectation.config import ArtifactConfig, ExperimentConfig, TrainingConfig
from lvc_expectation.diagnostics import (
    DIAGNOSTIC_CLASSIFICATION_RULE_VERSION,
    DIAGNOSTIC_METRIC_SCHEMA_VERSION,
    DIAGNOSTIC_SCHEMA_VERSION,
    HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
    HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT,
)
from lvc_expectation.provenance import (
    ARTIFACT_SCHEMA_VERSION,
    BENCHMARK_REGISTRY_VERSION,
    CONTRACT_VERSION,
    FROZEN_BENCHMARK_METRIC_VERSION,
    METRIC_SCHEMA_VERSION,
)
from lvc_expectation.runner import (
    run_higher_level_state_train_package,
    run_higher_level_state_package,
    run_context_residual_package,
    run_linear_probe_transplant_package,
    run_predictive_state_space_family_package,
    run_source_relative_commitment_context_package,
    run_source_relative_direction_engagement_package,
    run_source_relative_signed_direction_belief_package,
    run_orientation_head_interpolation_package,
    run_precision_target_package,
    run_readout_class_package,
    run_readout_alignment_package,
    run_context_only_prestim_template_gate,
    run_phase15_continuation_retention,
    run_probe_context_alignment_gate,
    run_local_global_surprise_gate,
    run_standard_end_localization_package,
    run_stage4_package,
    run_tranche1_experiment,
)


class RunnerTests(unittest.TestCase):
    @staticmethod
    def _make_stage4_probe_run_entry(
        *,
        run_id: str,
        train_seed: int,
        heldout_seed: int,
        latent: float = 0.025,
        pooled: float = 0.004,
        oracle_latent: float = 0.04,
        oracle_pooled: float = 0.01,
        learned_alignment_kl: float = 0.1,
        learned_top1: float = 0.8,
        learned_symmetry: float = 1.0,
        oracle_symmetry: float = 1.0,
        correct_pair_flip: float = 0.8,
        within_pair_mass: float = 0.5,
        source_bin_kl: float = 2.5,
        delta_latent: float = 0.01,
        artifacts_complete: bool = True,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "run_dir": Path("/tmp") / run_id,
            "seed_panel": "primary",
            "train_seed": int(train_seed),
            "heldout_seed": int(heldout_seed),
            "probe_metrics": {
                "probe_target_aligned_specificity_contrast": float(latent),
                "probe_pooled_target_aligned_specificity_contrast": float(pooled),
            },
            "oracle_probe_metrics": {
                "probe_target_aligned_specificity_contrast": float(oracle_latent),
                "probe_pooled_target_aligned_specificity_contrast": float(oracle_pooled),
            },
            "probe_context_alignment_report": {
                "learned_probe_alignment_kl": float(learned_alignment_kl),
                "learned_probe_expected_target_top1_rate": float(learned_top1),
                "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                "oracle_probe_pair_flip_symmetry_consistency": float(oracle_symmetry),
            },
            "hidden_state_diagnostics": {
                "probe_correct_pair_flip_rate__v2": float(correct_pair_flip),
                "probe_within_pair_mass__v2": float(within_pair_mass),
                "probe_source_bin_kl__v2": float(source_bin_kl),
                "probe_collapse_index__v2": 0.0,
            },
            "continuation_retention_report": {
                "delta_probe_target_aligned_specificity_contrast__pre_to_post_v2": float(delta_latent),
            },
            "artifacts_complete": bool(artifacts_complete),
        }

    @staticmethod
    def _make_standard_end_localization_entry(
        *,
        run_id: str,
        train_seed: int,
        heldout_seed: int,
        auxiliary_seed: int,
        latent: float = 0.005,
        pooled: float = 0.001,
        oracle_latent: float = 0.04,
        learned_alignment_kl: float = 1.0,
        learned_top1: float = 0.45,
        learned_symmetry: float = 1.0,
        linear_alignment_kl: float = 0.1,
        linear_top1: float = 0.85,
        linear_correct_pair_flip: float = 0.8,
        linear_within_pair_mass: float = 0.5,
        linear_source_bin_kl: float = 2.5,
        native_correct_pair_flip: float = 0.3,
        native_within_pair_mass: float = 0.2,
        native_source_bin_kl: float = 1.0,
        artifacts_complete: bool = True,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "run_dir": Path("/tmp") / run_id,
            "seed_panel": "primary",
            "train_seed": int(train_seed),
            "heldout_seed": int(heldout_seed),
            "auxiliary_seed": int(auxiliary_seed),
            "probe_metrics": {
                "probe_target_aligned_specificity_contrast": float(latent),
                "probe_pooled_target_aligned_specificity_contrast": float(pooled),
            },
            "oracle_probe_metrics": {
                "probe_target_aligned_specificity_contrast": float(oracle_latent),
                "probe_pooled_target_aligned_specificity_contrast": 0.01,
            },
            "probe_context_alignment_report": {
                "learned_probe_alignment_kl": float(learned_alignment_kl),
                "learned_probe_expected_target_top1_rate": float(learned_top1),
                "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
            },
            "hidden_state_diagnostics": {
                "probe_correct_pair_flip_rate__v2": float(native_correct_pair_flip),
                "probe_within_pair_mass__v2": float(native_within_pair_mass),
                "probe_source_bin_kl__v2": float(native_source_bin_kl),
            },
            "linear_probe_report": {
                "linear_probe_alignment_kl": float(linear_alignment_kl),
                "linear_probe_expected_target_top1_rate": float(linear_top1),
                "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                "train_eval_disjoint": True,
                "heldout_generator_seed": int(heldout_seed),
                "auxiliary_generator_seed": int(auxiliary_seed),
            },
            "linear_probe_diagnostics": {},
            "train_eval_disjoint": True,
            "artifacts_complete": bool(artifacts_complete),
        }

    @staticmethod
    def _make_readout_alignment_entry(
        *,
        run_id: str,
        train_seed: int,
        heldout_seed: int,
        auxiliary_probe_fit_seed: int,
        pre_latent: float = 0.005,
        pre_pooled: float = 0.001,
        post_latent: float = 0.02,
        post_pooled: float = 0.003,
        oracle_latent: float = 0.04,
        pre_native_alignment_kl: float = 1.0,
        post_native_alignment_kl: float = 0.2,
        pre_native_top1: float = 0.45,
        post_native_top1: float = 0.82,
        learned_symmetry: float = 1.0,
        linear_alignment_kl: float = 0.1,
        linear_top1: float = 0.85,
        linear_correct_pair_flip: float = 0.8,
        linear_within_pair_mass: float = 0.5,
        linear_source_bin_kl: float = 2.5,
        pre_native_correct_pair_flip: float = 0.3,
        post_native_correct_pair_flip: float = 0.82,
        pre_native_within_pair_mass: float = 0.2,
        post_native_within_pair_mass: float = 0.45,
        pre_native_source_bin_kl: float = 1.0,
        post_native_source_bin_kl: float = 2.1,
        integrity_passes: bool = True,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "run_dir": Path("/tmp") / run_id,
            "seed_panel": "primary",
            "train_seed": int(train_seed),
            "heldout_seed": int(heldout_seed),
            "auxiliary_probe_fit_seed": int(auxiliary_probe_fit_seed),
            "pre_alignment": {
                "probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(pre_latent),
                    "probe_pooled_target_aligned_specificity_contrast": float(pre_pooled),
                },
                "oracle_probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(oracle_latent),
                    "probe_pooled_target_aligned_specificity_contrast": 0.01,
                },
                "probe_context_alignment_report": {
                    "learned_probe_alignment_kl": float(pre_native_alignment_kl),
                    "learned_probe_expected_target_top1_rate": float(pre_native_top1),
                    "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                },
                "hidden_state_diagnostics": {
                    "probe_correct_pair_flip_rate__v2": float(pre_native_correct_pair_flip),
                    "probe_within_pair_mass__v2": float(pre_native_within_pair_mass),
                    "probe_source_bin_kl__v2": float(pre_native_source_bin_kl),
                },
                "linear_probe_report": {
                    "linear_probe_alignment_kl": float(linear_alignment_kl),
                    "linear_probe_expected_target_top1_rate": float(linear_top1),
                    "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                    "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                    "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                    "train_eval_disjoint": True,
                },
                "artifacts_complete": True,
            },
            "post_alignment": {
                "probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(post_latent),
                    "probe_pooled_target_aligned_specificity_contrast": float(post_pooled),
                },
                "oracle_probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(oracle_latent),
                    "probe_pooled_target_aligned_specificity_contrast": 0.01,
                },
                "probe_context_alignment_report": {
                    "learned_probe_alignment_kl": float(post_native_alignment_kl),
                    "learned_probe_expected_target_top1_rate": float(post_native_top1),
                    "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                },
                "hidden_state_diagnostics": {
                    "probe_correct_pair_flip_rate__v2": float(post_native_correct_pair_flip),
                    "probe_within_pair_mass__v2": float(post_native_within_pair_mass),
                    "probe_source_bin_kl__v2": float(post_native_source_bin_kl),
                },
                "linear_probe_report": {
                    "linear_probe_alignment_kl": float(linear_alignment_kl),
                    "linear_probe_expected_target_top1_rate": float(linear_top1),
                    "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                    "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                    "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                    "train_eval_disjoint": True,
                },
                "artifacts_complete": True,
            },
            "alignment_integrity": {
                "heldout_batch_identity_same_pre_post": bool(integrity_passes),
                "auxiliary_batch_identity_same_pre_post": bool(integrity_passes),
                "auxiliary_eval_disjoint_from_heldout": bool(integrity_passes),
                "recurrent_core_parameter_fingerprint_unchanged": bool(integrity_passes),
                "precision_head_parameter_fingerprint_unchanged": bool(integrity_passes),
                "orientation_head_parameter_fingerprint_changed": bool(integrity_passes),
                "heldout_hidden_states_unchanged": bool(integrity_passes),
                "auxiliary_hidden_states_unchanged": bool(integrity_passes),
                "predictive_loss_pre": 0.3,
                "predictive_loss_post": 0.31,
                "predictive_structure_accuracy_pre": 1.0,
                "predictive_structure_accuracy_post": 1.0,
                "nuisance_only_accuracy_pre": 0.0,
                "nuisance_only_accuracy_post": 0.0,
                "delta_predictive_loss": 0.01,
                "delta_predictive_structure_accuracy": 0.0,
                "delta_nuisance_only_accuracy": 0.0,
                "predictive_nonregression_passes": bool(integrity_passes),
            },
            "artifacts_complete": True,
        }

    @staticmethod
    def _make_linear_probe_transplant_entry(
        *,
        run_id: str,
        train_seed: int,
        heldout_seed: int,
        auxiliary_probe_fit_seed: int,
        pre_latent: float = 0.005,
        pre_pooled: float = 0.001,
        post_latent: float = 0.02,
        post_pooled: float = 0.003,
        oracle_latent: float = 0.04,
        pre_native_alignment_kl: float = 1.0,
        post_native_alignment_kl: float = 0.2,
        pre_native_top1: float = 0.45,
        post_native_top1: float = 0.82,
        learned_symmetry: float = 1.0,
        linear_alignment_kl: float = 0.1,
        linear_top1: float = 0.85,
        linear_correct_pair_flip: float = 0.8,
        linear_within_pair_mass: float = 0.5,
        linear_source_bin_kl: float = 2.5,
        pre_native_correct_pair_flip: float = 0.3,
        post_native_correct_pair_flip: float = 0.82,
        pre_native_within_pair_mass: float = 0.2,
        post_native_within_pair_mass: float = 0.45,
        pre_native_source_bin_kl: float = 1.0,
        post_native_source_bin_kl: float = 2.1,
        integrity_passes: bool = True,
        native_logits_match: bool = True,
        no_optimization_after_transplant: bool = True,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "run_dir": Path("/tmp") / run_id,
            "seed_panel": "primary",
            "train_seed": int(train_seed),
            "heldout_seed": int(heldout_seed),
            "auxiliary_probe_fit_seed": int(auxiliary_probe_fit_seed),
            "pre_transplant": {
                "probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(pre_latent),
                    "probe_pooled_target_aligned_specificity_contrast": float(pre_pooled),
                },
                "oracle_probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(oracle_latent),
                    "probe_pooled_target_aligned_specificity_contrast": 0.01,
                },
                "probe_context_alignment_report": {
                    "learned_probe_alignment_kl": float(pre_native_alignment_kl),
                    "learned_probe_expected_target_top1_rate": float(pre_native_top1),
                    "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                },
                "hidden_state_diagnostics": {
                    "probe_correct_pair_flip_rate__v2": float(pre_native_correct_pair_flip),
                    "probe_within_pair_mass__v2": float(pre_native_within_pair_mass),
                    "probe_source_bin_kl__v2": float(pre_native_source_bin_kl),
                },
                "linear_probe_report": {
                    "linear_probe_alignment_kl": float(linear_alignment_kl),
                    "linear_probe_expected_target_top1_rate": float(linear_top1),
                    "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                    "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                    "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                    "train_eval_disjoint": True,
                },
                "artifacts_complete": True,
            },
            "post_transplant": {
                "probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(post_latent),
                    "probe_pooled_target_aligned_specificity_contrast": float(post_pooled),
                },
                "oracle_probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(oracle_latent),
                    "probe_pooled_target_aligned_specificity_contrast": 0.01,
                },
                "probe_context_alignment_report": {
                    "learned_probe_alignment_kl": float(post_native_alignment_kl),
                    "learned_probe_expected_target_top1_rate": float(post_native_top1),
                    "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                },
                "hidden_state_diagnostics": {
                    "probe_correct_pair_flip_rate__v2": float(post_native_correct_pair_flip),
                    "probe_within_pair_mass__v2": float(post_native_within_pair_mass),
                    "probe_source_bin_kl__v2": float(post_native_source_bin_kl),
                },
                "linear_probe_report": {
                    "linear_probe_alignment_kl": float(linear_alignment_kl),
                    "linear_probe_expected_target_top1_rate": float(linear_top1),
                    "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                    "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                    "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                    "train_eval_disjoint": True,
                },
                "artifacts_complete": True,
            },
            "transplant_integrity": {
                "heldout_batch_identity_same_pre_post": bool(integrity_passes),
                "auxiliary_batch_identity_same_pre_post": bool(integrity_passes),
                "auxiliary_eval_disjoint_from_heldout": bool(integrity_passes),
                "recurrent_core_parameter_fingerprint_unchanged": bool(integrity_passes),
                "precision_head_parameter_fingerprint_unchanged": bool(integrity_passes),
                "orientation_head_parameter_fingerprint_changed": bool(integrity_passes),
                "heldout_hidden_states_unchanged": bool(integrity_passes),
                "auxiliary_hidden_states_unchanged": bool(integrity_passes),
                "heldout_hidden_state_max_abs_diff": 0.0,
                "auxiliary_hidden_state_max_abs_diff": 0.0,
                "transplanted_native_orientation_head_equals_donor": bool(integrity_passes),
                "native_orientation_logits_match_standalone_linear_probe": bool(native_logits_match),
                "no_optimization_after_transplant": bool(no_optimization_after_transplant),
                "optimization_steps_after_transplant": 0,
                "predictive_loss_pre": 0.3,
                "predictive_loss_post": 0.3,
                "predictive_structure_accuracy_pre": 0.9,
                "predictive_structure_accuracy_post": 0.9,
                "nuisance_only_accuracy_pre": 0.2,
                "nuisance_only_accuracy_post": 0.2,
                "delta_predictive_loss": 0.0,
                "delta_predictive_structure_accuracy": 0.0,
                "delta_nuisance_only_accuracy": 0.0,
                "predictive_nonregression_passes": True,
            },
            "transplant_logit_match_report": {
                "weight_shape_match": True,
                "bias_shape_match": True,
                "max_abs_weight_diff_after_copy": 0.0,
                "max_abs_bias_diff_after_copy": 0.0,
                "max_abs_logit_diff": 0.0,
                "mean_abs_logit_diff": 0.0,
                "native_logits_match_standalone_linear_probe": bool(native_logits_match),
                "no_optimization_after_transplant": bool(no_optimization_after_transplant),
            },
            "artifacts_complete": True,
        }

    @staticmethod
    def _make_orientation_head_interpolation_entry(
        *,
        run_id: str,
        train_seed: int,
        heldout_seed: int,
        donor_fit_seed: int,
        predictive_rehearsal_seed: int,
        fixed_probe_selection_seed: int,
        selected_alpha: float = 0.375,
        pre_latent: float = 0.005,
        pre_pooled: float = 0.001,
        post_latent: float = 0.02,
        post_pooled: float = 0.003,
        oracle_latent: float = 0.04,
        pre_native_alignment_kl: float = 1.0,
        post_native_alignment_kl: float = 0.2,
        pre_native_top1: float = 0.45,
        post_native_top1: float = 0.82,
        learned_symmetry: float = 1.0,
        linear_alignment_kl: float = 0.1,
        linear_top1: float = 0.85,
        linear_correct_pair_flip: float = 0.8,
        linear_within_pair_mass: float = 0.5,
        linear_source_bin_kl: float = 2.5,
        pre_native_correct_pair_flip: float = 0.3,
        post_native_correct_pair_flip: float = 0.82,
        pre_native_within_pair_mass: float = 0.2,
        post_native_within_pair_mass: float = 0.45,
        pre_native_source_bin_kl: float = 1.0,
        post_native_source_bin_kl: float = 2.1,
        integrity_passes: bool = True,
        predictive_nonregression_passes: bool = True,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "run_dir": Path("/tmp") / run_id,
            "seed_panel": "primary",
            "train_seed": int(train_seed),
            "heldout_seed": int(heldout_seed),
            "auxiliary_probe_fit_seed": int(donor_fit_seed),
            "donor_fit_seed": int(donor_fit_seed),
            "predictive_rehearsal_seed": int(predictive_rehearsal_seed),
            "fixed_probe_selection_seed": int(fixed_probe_selection_seed),
            "selected_alpha": float(selected_alpha),
            "pre_interpolation": {
                "probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(pre_latent),
                    "probe_pooled_target_aligned_specificity_contrast": float(pre_pooled),
                },
                "oracle_probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(oracle_latent),
                    "probe_pooled_target_aligned_specificity_contrast": 0.01,
                },
                "probe_context_alignment_report": {
                    "learned_probe_alignment_kl": float(pre_native_alignment_kl),
                    "learned_probe_expected_target_top1_rate": float(pre_native_top1),
                    "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                },
                "hidden_state_diagnostics": {
                    "probe_correct_pair_flip_rate__v2": float(pre_native_correct_pair_flip),
                    "probe_within_pair_mass__v2": float(pre_native_within_pair_mass),
                    "probe_source_bin_kl__v2": float(pre_native_source_bin_kl),
                },
                "linear_probe_report": {
                    "linear_probe_alignment_kl": float(linear_alignment_kl),
                    "linear_probe_expected_target_top1_rate": float(linear_top1),
                    "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                    "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                    "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                    "train_eval_disjoint": True,
                },
                "artifacts_complete": True,
            },
            "post_interpolation": {
                "probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(post_latent),
                    "probe_pooled_target_aligned_specificity_contrast": float(post_pooled),
                },
                "oracle_probe_metrics": {
                    "probe_target_aligned_specificity_contrast": float(oracle_latent),
                    "probe_pooled_target_aligned_specificity_contrast": 0.01,
                },
                "probe_context_alignment_report": {
                    "learned_probe_alignment_kl": float(post_native_alignment_kl),
                    "learned_probe_expected_target_top1_rate": float(post_native_top1),
                    "learned_probe_pair_flip_symmetry_consistency": float(learned_symmetry),
                },
                "hidden_state_diagnostics": {
                    "probe_correct_pair_flip_rate__v2": float(post_native_correct_pair_flip),
                    "probe_within_pair_mass__v2": float(post_native_within_pair_mass),
                    "probe_source_bin_kl__v2": float(post_native_source_bin_kl),
                },
                "linear_probe_report": {
                    "linear_probe_alignment_kl": float(linear_alignment_kl),
                    "linear_probe_expected_target_top1_rate": float(linear_top1),
                    "linear_probe_correct_pair_flip_rate__v2": float(linear_correct_pair_flip),
                    "linear_probe_within_pair_mass__v2": float(linear_within_pair_mass),
                    "linear_probe_source_bin_kl__v2": float(linear_source_bin_kl),
                    "train_eval_disjoint": True,
                },
                "artifacts_complete": True,
            },
            "interpolation_integrity": {
                "heldout_batch_identity_same_pre_post": bool(integrity_passes),
                "donor_fit_batch_identity_same_pre_post": bool(integrity_passes),
                "predictive_rehearsal_batch_identity_same_pre_post": bool(integrity_passes),
                "fixed_probe_selection_batch_identity_same_pre_post": bool(integrity_passes),
                "all_batch_identities_disjoint": bool(integrity_passes),
                "recurrent_core_parameter_fingerprint_unchanged": bool(integrity_passes),
                "precision_head_parameter_fingerprint_unchanged": bool(integrity_passes),
                "heldout_hidden_states_unchanged": bool(integrity_passes),
                "heldout_hidden_state_max_abs_diff": 0.0,
                "donor_fit_hidden_state_max_abs_diff": 0.0,
                "predictive_rehearsal_hidden_state_max_abs_diff": 0.0,
                "fixed_probe_selection_hidden_state_max_abs_diff": 0.0,
                "selected_head_matches_interpolation_formula": bool(integrity_passes),
                "predictive_feasibility_filter_recorded": True,
                "alpha_selection_table_recorded": True,
                "no_optimization": True,
                "optimization_steps": 0,
                "predictive_loss_pre": 0.3,
                "predictive_loss_post": 0.3,
                "predictive_structure_accuracy_pre": 0.9,
                "predictive_structure_accuracy_post": 0.9,
                "nuisance_only_accuracy_pre": 0.2,
                "nuisance_only_accuracy_post": 0.2,
                "delta_predictive_loss": 0.0,
                "delta_predictive_structure_accuracy": 0.0,
                "delta_nuisance_only_accuracy": 0.0,
                "predictive_nonregression_passes": bool(predictive_nonregression_passes),
                "selected_alpha": float(selected_alpha),
            },
            "interpolation_state_report": {
                "selected_alpha": float(selected_alpha),
                "selected_head_matches_formula": bool(integrity_passes),
                "max_abs_weight_diff_after_copy": 0.0,
                "max_abs_bias_diff_after_copy": 0.0,
            },
            "artifacts_complete": True,
        }

    def test_runner_writes_expected_artifacts_and_separates_train_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=2),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_tranche1_experiment(config=config, train_seed=11, eval_seed=101)

            run_dir = result.run_dir
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "predictive_metrics.json").exists())
            self.assertTrue((run_dir / "sanity_report.json").exists())
            self.assertTrue((run_dir / "neutral_match_report.json").exists())
            self.assertTrue((run_dir / "primary_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_primary_metrics.json").exists())
            self.assertTrue((run_dir / "task_mode_primary_metrics.json").exists())
            self.assertTrue((run_dir / "resolved_config.json").exists())
            self.assertTrue((run_dir / "environment.json").exists())
            self.assertTrue((run_dir / "run_fingerprint.json").exists())
            self.assertTrue((run_dir / "eval" / "window_summaries.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_window_summaries.pt").exists())
            self.assertTrue((run_dir / "eval" / "full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertFalse((run_dir / "probe_context_alignment_report.json").exists())
            self.assertFalse((run_dir / "eval" / "probe_context_predictions.pt").exists())

            predictive_metrics = json.loads((run_dir / "predictive_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(predictive_metrics["train_seeds"], [11, 12])
            self.assertEqual(predictive_metrics["heldout_seed"], 101)
            self.assertNotEqual(predictive_metrics["train_seeds"][0], predictive_metrics["heldout_seed"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["contract_version"], CONTRACT_VERSION)
            self.assertEqual(manifest["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(manifest["metric_schema_version"], METRIC_SCHEMA_VERSION)
            self.assertEqual(manifest["benchmark_registry_version"], BENCHMARK_REGISTRY_VERSION)
            self.assertEqual(manifest["lineage"]["parent_run_ids"], [])
            self.assertEqual(manifest["lineage"]["child_run_ids"], [])
            self.assertEqual(manifest["lineage"]["benchmark_anchor_refs"], [])
            notes = manifest["notes"]
            self.assertNotIn("objective_mode", notes)
            self.assertNotIn("probe_finetune_epochs", notes)
            self.assertNotIn("probe_finetune_mode", notes)
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("challenger_initial_standard_epochs", notes)
            self.assertNotIn("challenger_continuation_probe_updates", notes)
            self.assertNotIn("challenger_continuation_standard_updates", notes)
            self.assertNotIn("challenger_continuation_schedule", notes)
            fingerprint_payload = json.loads((run_dir / "run_fingerprint.json").read_text(encoding="utf-8"))
            self.assertEqual(fingerprint_payload["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(fingerprint_payload["metric_schema_version"], METRIC_SCHEMA_VERSION)
            self.assertEqual(len(fingerprint_payload["run_fingerprint"]), 64)
            resolved_config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
            self.assertEqual(resolved_config["name"], config.name)
            environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
            self.assertIn("python_version", environment)
            self.assertIn("torch_version", environment)
            self.assertIn("git", environment)

            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")
            for field_name in (
                "orientations",
                "blank_mask",
                "expected_mask",
                "context_ids",
                "task_mode",
                "prestim_mode",
                "orthogonal_events",
                "metadata",
            ):
                self.assertIn(field_name, heldout_batch)

    def test_saved_window_summaries_are_derived_from_full_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_tranche1_experiment(config=config, train_seed=21, eval_seed=121)

            summaries = torch.load(result.run_dir / "eval" / "window_summaries.pt")
            trajectories = torch.load(result.run_dir / "eval" / "full_trajectories.pt")

            late_summary = summaries["dampening"]["late"]["l23_readout"]
            late_expected = trajectories["dampening"]["states"]["l23_readout"][:, 7:12].mean(dim=1)

            self.assertTrue(torch.allclose(late_summary, late_expected))

    def test_runner_honors_explicit_controlled_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            controlled_sources = [0, 1, 2, 3, 4, 5]
            result = run_tranche1_experiment(
                config=config,
                train_seed=31,
                eval_seed=131,
                controlled_sources=controlled_sources,
            )

            trajectories = torch.load(result.run_dir / "eval" / "full_trajectories.pt")
            saved_controlled_sources = trajectories["dampening"]["metadata"]["controlled_sources"]

            self.assertTrue(torch.equal(saved_controlled_sources, torch.tensor(controlled_sources)))

    def test_oracle_context_runner_path_uses_same_heldout_batch_and_prediction_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=8, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_tranche1_experiment(
                config=config,
                train_seed=51,
                eval_seed=151,
                subcases=("dampening",),
            )

            run_dir = result.run_dir
            learned_trajectories = torch.load(run_dir / "eval" / "full_trajectories.pt")
            oracle_trajectories = torch.load(run_dir / "eval" / "oracle_full_trajectories.pt")
            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")

            learned = learned_trajectories["dampening"]
            oracle = oracle_trajectories["dampening"]
            self.assertEqual(
                tuple(learned["context_predictions"].shape),
                tuple(oracle["context_predictions"].shape),
            )
            self.assertEqual(
                tuple(learned["states"]["l23_readout"].shape),
                tuple(oracle["states"]["l23_readout"].shape),
            )
            self.assertEqual(
                learned["metadata"]["generator_seed"],
                heldout_batch["metadata"]["generator_seed"],
            )
            self.assertEqual(
                oracle["metadata"]["generator_seed"],
                heldout_batch["metadata"]["generator_seed"],
            )

    def test_task_mode_stratified_metrics_are_emitted_without_replacing_aggregate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=16, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_tranche1_experiment(
                config=config,
                train_seed=61,
                eval_seed=101,
                subcases=("dampening",),
            )

            aggregate = json.loads((result.run_dir / "primary_metrics.json").read_text(encoding="utf-8"))
            oracle = json.loads((result.run_dir / "oracle_primary_metrics.json").read_text(encoding="utf-8"))
            stratified = json.loads((result.run_dir / "task_mode_primary_metrics.json").read_text(encoding="utf-8"))

            self.assertIn("dampening", aggregate)
            self.assertIn("mean_suppression", aggregate["dampening"])
            self.assertIn("poststimulus_pooled_template_specificity", aggregate["dampening"])
            self.assertIn("dampening", oracle)
            self.assertIn("mean_suppression", oracle["dampening"])
            self.assertIn("poststimulus_pooled_template_specificity", oracle["dampening"])
            for context_source in ("learned", "oracle"):
                self.assertIn("orientation_relevant", stratified[context_source])
                self.assertIn("orthogonal_relevant", stratified[context_source])
                for task_name in ("orientation_relevant", "orthogonal_relevant"):
                    self.assertGreater(stratified[context_source][task_name]["n_trials"], 0)
                    self.assertIn("dampening", stratified[context_source][task_name]["metrics"])
                    self.assertIn(
                        "poststimulus_l23_template_specificity",
                        stratified[context_source][task_name]["metrics"]["dampening"],
                    )
                    self.assertIn(
                        "poststimulus_pooled_template_specificity",
                        stratified[context_source][task_name]["metrics"]["dampening"],
                    )

    def test_prestim_gate_runner_writes_zero_context_control_on_same_heldout_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=8, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_context_only_prestim_template_gate(
                config=config,
                train_seed=41,
                eval_seed=141,
                subcase="dampening",
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "prestim_gate.json").exists())
            self.assertTrue((run_dir / "eval" / "prestim_gate_trajectories.pt").exists())

            gate_payload = json.loads((run_dir / "prestim_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_payload["subcase"], "dampening")
            self.assertEqual(gate_payload["control_mode"], "zero_context")
            for control_name in ("intact", "zero_context"):
                self.assertIn("cue_only", gate_payload[control_name])
                self.assertIn("context_only", gate_payload[control_name])
                self.assertIn("neutral", gate_payload[control_name])

            trajectories = torch.load(run_dir / "eval" / "prestim_gate_trajectories.pt")
            self.assertEqual(
                trajectories["intact"]["metadata"]["generator_seed"],
                trajectories["zero_context"]["metadata"]["generator_seed"],
            )
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertNotIn("objective_mode", notes)
            self.assertNotIn("probe_finetune_epochs", notes)
            self.assertNotIn("probe_finetune_mode", notes)
            self.assertNotIn("phase2_regime", notes)
            self.assertNotIn("probe_train_source_subset", notes)
            self.assertNotIn("probe_eval_source_subset", notes)
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("challenger_initial_standard_epochs", notes)
            self.assertNotIn("challenger_continuation_probe_updates", notes)
            self.assertNotIn("challenger_continuation_standard_updates", notes)
            self.assertNotIn("challenger_continuation_schedule", notes)

    def test_local_global_probe_runner_writes_probe_artifacts_and_same_batch_for_oracle_and_learned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_local_global_surprise_gate(
                config=config,
                train_seed=71,
                eval_seed=171,
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_probe_metrics.json").exists())
            self.assertTrue((run_dir / "probe_design_report.json").exists())
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "probe_table.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_full_trajectories.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "local_global_surprise_probe")
            self.assertEqual(notes["probe_visible_step_index"], 1)
            self.assertIn("probe_pairs", notes)
            self.assertIn("controlled_sources", notes)
            self.assertEqual(notes["task_mode_fixed_state"], "orientation_relevant")
            self.assertEqual(notes["prestim_fixed_state"], "none")
            self.assertIn("oracle_precision_mode", notes)
            self.assertNotIn("objective_mode", notes)
            self.assertNotIn("probe_finetune_epochs", notes)
            self.assertNotIn("probe_finetune_mode", notes)
            self.assertNotIn("phase2_regime", notes)
            self.assertNotIn("probe_train_source_subset", notes)
            self.assertNotIn("probe_eval_source_subset", notes)
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("challenger_initial_standard_epochs", notes)
            self.assertNotIn("challenger_continuation_probe_updates", notes)
            self.assertNotIn("challenger_continuation_standard_updates", notes)
            self.assertNotIn("challenger_continuation_schedule", notes)

            probe_metrics = json.loads((run_dir / "probe_metrics.json").read_text(encoding="utf-8"))
            oracle_probe_metrics = json.loads((run_dir / "oracle_probe_metrics.json").read_text(encoding="utf-8"))
            probe_design_report = json.loads((run_dir / "probe_design_report.json").read_text(encoding="utf-8"))
            probe_context_alignment_report = json.loads(
                (run_dir / "probe_context_alignment_report.json").read_text(encoding="utf-8")
            )
            probe_table = json.loads((run_dir / "probe_table.json").read_text(encoding="utf-8"))
            self.assertIn("probe_target_aligned_specificity_contrast", probe_metrics)
            self.assertIn("probe_pooled_target_aligned_specificity_contrast", probe_metrics)
            self.assertIn("probe_context_comparator_nonuniformity_contrast", probe_metrics)
            self.assertIn("probe_target_aligned_specificity_contrast", oracle_probe_metrics)
            for field_name in (
                "probe_visible_step_index",
                "task_mode_fixed_state",
                "prestim_fixed_state",
                "contexts_used",
                "controlled_sources",
                "n_probe_rows",
                "n_probe_global_expected_rows",
                "n_probe_global_unexpected_rows",
                "n_probe_pairs_total",
                "n_probe_pairs_scored",
                "all_probe_rows_nonblank",
                "all_probe_rows_nonomission",
                "all_probe_rows_controlled_source",
                "all_probe_rows_at_fixed_visible_step",
                "all_pairs_have_two_rows",
                "all_pairs_span_contexts_0_1",
                "all_pairs_same_source",
                "all_pairs_same_target",
                "all_pairs_flip_global_expectedness",
                "all_pairs_have_symmetry_mate",
                "same_batch_provenance_ok",
                "pair_balance_rows",
            ):
                self.assertIn(field_name, probe_design_report)
            for field_name in (
                "learned_probe_alignment_kl",
                "oracle_probe_alignment_kl",
                "learned_probe_expected_logprob",
                "oracle_probe_expected_logprob",
                "learned_probe_expected_target_top1_rate",
                "oracle_probe_expected_target_top1_rate",
                "learned_probe_pair_flip_rate",
                "oracle_probe_pair_flip_rate",
                "learned_probe_pair_flip_symmetry_consistency",
                "oracle_probe_pair_flip_symmetry_consistency",
            ):
                self.assertIn(field_name, probe_context_alignment_report)
            self.assertGreater(len(probe_table), 0)
            for field_name in (
                "condition_code",
                "orthogonal_event",
                "repetition_lag",
                "run_length",
                "controlled_source_step",
                "is_symmetry_mate",
            ):
                self.assertIn(field_name, probe_table[0])
            self.assertIsInstance(probe_table[0]["is_symmetry_mate"], bool)
            self.assertEqual(probe_design_report["probe_visible_step_index"], 1)
            self.assertEqual(probe_design_report["task_mode_fixed_state"], "orientation_relevant")
            self.assertEqual(probe_design_report["prestim_fixed_state"], "none")
            self.assertEqual(probe_design_report["contexts_used"], [0, 1])
            self.assertEqual(probe_design_report["controlled_sources"], [0, 1, 2, 3, 4, 5])
            self.assertEqual(probe_design_report["n_probe_rows"], len(probe_table))
            self.assertEqual(
                probe_design_report["n_probe_global_expected_rows"] + probe_design_report["n_probe_global_unexpected_rows"],
                probe_design_report["n_probe_rows"],
            )
            self.assertEqual(
                probe_design_report["n_probe_pairs_total"],
                len(probe_design_report["pair_balance_rows"]),
            )
            self.assertEqual(
                probe_design_report["n_probe_pairs_scored"],
                probe_design_report["n_probe_pairs_total"],
            )
            self.assertTrue(probe_design_report["all_probe_rows_nonblank"])
            self.assertTrue(probe_design_report["all_probe_rows_nonomission"])
            self.assertTrue(probe_design_report["all_probe_rows_controlled_source"])
            self.assertTrue(probe_design_report["all_probe_rows_at_fixed_visible_step"])
            self.assertTrue(probe_design_report["all_pairs_have_two_rows"])
            self.assertTrue(probe_design_report["all_pairs_span_contexts_0_1"])
            self.assertTrue(probe_design_report["all_pairs_same_source"])
            self.assertTrue(probe_design_report["all_pairs_same_target"])
            self.assertTrue(probe_design_report["all_pairs_flip_global_expectedness"])
            self.assertTrue(probe_design_report["all_pairs_have_symmetry_mate"])
            self.assertTrue(probe_design_report["same_batch_provenance_ok"])
            self.assertGreaterEqual(probe_context_alignment_report["learned_probe_alignment_kl"], 0.0)
            self.assertAlmostEqual(probe_context_alignment_report["oracle_probe_alignment_kl"], 0.0, places=6)
            self.assertGreaterEqual(probe_context_alignment_report["learned_probe_expected_target_top1_rate"], 0.0)
            self.assertLessEqual(probe_context_alignment_report["learned_probe_expected_target_top1_rate"], 1.0)
            self.assertGreaterEqual(probe_context_alignment_report["oracle_probe_expected_target_top1_rate"], 0.0)
            self.assertLessEqual(probe_context_alignment_report["oracle_probe_expected_target_top1_rate"], 1.0)
            self.assertGreaterEqual(probe_context_alignment_report["learned_probe_pair_flip_rate"], 0.0)
            self.assertLessEqual(probe_context_alignment_report["learned_probe_pair_flip_rate"], 1.0)
            self.assertGreaterEqual(probe_context_alignment_report["oracle_probe_pair_flip_rate"], 0.0)
            self.assertLessEqual(probe_context_alignment_report["oracle_probe_pair_flip_rate"], 1.0)
            self.assertGreaterEqual(probe_context_alignment_report["learned_probe_pair_flip_symmetry_consistency"], 0.0)
            self.assertLessEqual(probe_context_alignment_report["learned_probe_pair_flip_symmetry_consistency"], 1.0)
            self.assertGreaterEqual(probe_context_alignment_report["oracle_probe_pair_flip_symmetry_consistency"], 0.0)
            self.assertLessEqual(probe_context_alignment_report["oracle_probe_pair_flip_symmetry_consistency"], 1.0)
            self.assertLessEqual(
                probe_context_alignment_report["learned_probe_expected_logprob"],
                0.0,
            )
            self.assertLessEqual(
                probe_context_alignment_report["oracle_probe_expected_logprob"],
                0.0,
            )

            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")
            learned_trajectories = torch.load(run_dir / "eval" / "full_trajectories.pt")
            oracle_trajectories = torch.load(run_dir / "eval" / "oracle_full_trajectories.pt")
            learned = learned_trajectories["dampening"]
            oracle = oracle_trajectories["dampening"]
            self.assertEqual(
                learned["metadata"]["generator_seed"],
                heldout_batch["metadata"]["generator_seed"],
            )
            self.assertEqual(
                oracle["metadata"]["generator_seed"],
                heldout_batch["metadata"]["generator_seed"],
            )
            self.assertEqual(
                tuple(learned["context_predictions"].shape),
                tuple(oracle["context_predictions"].shape),
            )

    def test_local_global_probe_runner_supports_opt_in_recovery_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_local_global_surprise_gate(
                config=config,
                train_seed=73,
                eval_seed=173,
                objective_mode="expected_distribution",
                probe_finetune_epochs=1,
                probe_finetune_mode="probe_step_only",
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_probe_metrics.json").exists())
            self.assertTrue((run_dir / "probe_design_report.json").exists())
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "probe_table.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_full_trajectories.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "local_global_surprise_probe")
            self.assertEqual(notes["objective_mode"], "expected_distribution")
            self.assertEqual(notes["probe_finetune_epochs"], 1)
            self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")

    def test_local_global_probe_runner_supports_source_heldout_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_local_global_surprise_gate(
                config=config,
                train_seed=74,
                eval_seed=174,
                objective_mode="expected_distribution",
                probe_finetune_epochs=1,
                probe_finetune_mode="probe_step_only",
                probe_train_source_subset=[0, 1, 2],
                probe_eval_source_subset=[3, 4, 5],
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_probe_metrics.json").exists())
            self.assertTrue((run_dir / "probe_design_report.json").exists())
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "probe_table.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_full_trajectories.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "local_global_surprise_probe")
            self.assertEqual(notes["phase2_regime"], "p2a_source_heldout_probe_generalization")
            self.assertEqual(notes["probe_train_source_subset"], [0, 1, 2])
            self.assertEqual(notes["probe_eval_source_subset"], [3, 4, 5])
            self.assertEqual(notes["objective_mode"], "expected_distribution")
            self.assertEqual(notes["probe_finetune_epochs"], 1)
            self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")

            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")
            self.assertEqual(heldout_batch["metadata"]["probe_source_subset"].tolist(), [3, 4, 5])

    def test_local_global_probe_runner_supports_challenger_mixed_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_local_global_surprise_gate(
                config=config,
                train_seed=76,
                eval_seed=176,
                objective_mode="expected_distribution",
                challenger_candidate="std400_mix_50p_150s",
                probe_train_source_subset=[0, 1, 2],
                probe_eval_source_subset=[3, 4, 5],
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_probe_metrics.json").exists())
            self.assertTrue((run_dir / "probe_design_report.json").exists())
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "probe_table.json").exists())
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["challenger_candidate"], "std400_mix_50p_150s")
            self.assertEqual(notes["challenger_initial_standard_epochs"], 400)
            self.assertEqual(notes["challenger_continuation_probe_updates"], 50)
            self.assertEqual(notes["challenger_continuation_standard_updates"], 150)
            self.assertEqual(len(notes["challenger_continuation_schedule"]), 200)
            self.assertEqual(notes["challenger_continuation_schedule"].count("probe"), 50)
            self.assertEqual(notes["challenger_continuation_schedule"].count("standard"), 150)
            self.assertEqual(notes["phase2_regime"], "p2a_source_heldout_probe_generalization")
            self.assertEqual(notes["probe_train_source_subset"], [0, 1, 2])
            self.assertEqual(notes["probe_eval_source_subset"], [3, 4, 5])
            self.assertEqual(len(result.train_history), 600)

    def test_probe_context_alignment_gate_supports_challenger_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_probe_context_alignment_gate(
                config=config,
                train_seed=82,
                eval_seed=182,
                objective_mode="expected_distribution",
                challenger_candidate="std400_only",
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "probe_context_predictions.pt").exists())
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["challenger_candidate"], "std400_only")
            self.assertEqual(notes["challenger_initial_standard_epochs"], 400)
            self.assertEqual(notes["challenger_continuation_probe_updates"], 0)
            self.assertEqual(notes["challenger_continuation_standard_updates"], 0)
            self.assertEqual(notes["challenger_continuation_schedule"], [])
            self.assertEqual(len(result.train_history), 400)

    def test_tranche1_runner_supports_opt_in_recovered_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_tranche1_experiment(
                config=config,
                train_seed=75,
                eval_seed=175,
                objective_mode="expected_distribution",
                probe_finetune_epochs=1,
                probe_finetune_mode="probe_step_only",
                subcases=("dampening",),
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "predictive_metrics.json").exists())
            self.assertTrue((run_dir / "sanity_report.json").exists())
            self.assertTrue((run_dir / "neutral_match_report.json").exists())
            self.assertTrue((run_dir / "primary_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_primary_metrics.json").exists())
            self.assertTrue((run_dir / "task_mode_primary_metrics.json").exists())
            self.assertTrue((run_dir / "eval" / "window_summaries.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_window_summaries.pt").exists())
            self.assertTrue((run_dir / "eval" / "full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["objective_mode"], "expected_distribution")
            self.assertEqual(notes["probe_finetune_epochs"], 1)
            self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")

    def test_probe_context_alignment_gate_writes_alignment_artifacts_without_touching_rollout_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_probe_context_alignment_gate(
                config=config,
                train_seed=81,
                eval_seed=181,
                objective_mode="expected_distribution",
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "probe_context_predictions.pt").exists())
            self.assertTrue((run_dir / "resolved_config.json").exists())
            self.assertTrue((run_dir / "environment.json").exists())
            self.assertTrue((run_dir / "run_fingerprint.json").exists())
            self.assertFalse((run_dir / "probe_metrics.json").exists())
            self.assertFalse((run_dir / "oracle_probe_metrics.json").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["contract_version"], CONTRACT_VERSION)
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "probe_context_alignment")
            self.assertEqual(notes["objective_mode"], "expected_distribution")
            self.assertEqual(notes["probe_visible_step_index"], 1)
            self.assertEqual(notes["task_mode_fixed_state"], "orientation_relevant")
            self.assertEqual(notes["prestim_fixed_state"], "none")
            self.assertEqual(notes["controlled_sources"], [0, 1, 2, 3, 4, 5])
            self.assertEqual(notes["contexts_used"], [0, 1])
            self.assertEqual(notes["probe_finetune_epochs"], 0)
            self.assertEqual(notes["probe_finetune_mode"], "disabled")
            self.assertNotIn("phase2_regime", notes)
            self.assertNotIn("probe_train_source_subset", notes)
            self.assertNotIn("probe_eval_source_subset", notes)
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("challenger_initial_standard_epochs", notes)
            self.assertNotIn("challenger_continuation_probe_updates", notes)
            self.assertNotIn("challenger_continuation_standard_updates", notes)
            self.assertNotIn("challenger_continuation_schedule", notes)

            alignment_report = json.loads((run_dir / "probe_context_alignment_report.json").read_text(encoding="utf-8"))
            for field_name in (
                "learned_probe_alignment_kl",
                "oracle_probe_alignment_kl",
                "learned_probe_expected_logprob",
                "oracle_probe_expected_logprob",
                "learned_probe_expected_target_top1_rate",
                "oracle_probe_expected_target_top1_rate",
                "learned_probe_pair_flip_rate",
                "oracle_probe_pair_flip_rate",
                "learned_probe_pair_flip_symmetry_consistency",
                "oracle_probe_pair_flip_symmetry_consistency",
            ):
                self.assertIn(field_name, alignment_report)
            self.assertGreaterEqual(alignment_report["learned_probe_alignment_kl"], 0.0)
            self.assertAlmostEqual(alignment_report["oracle_probe_alignment_kl"], 0.0, places=6)

            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")
            saved_predictions = torch.load(run_dir / "eval" / "probe_context_predictions.pt")
            self.assertIn("learned", saved_predictions)
            self.assertIn("oracle", saved_predictions)
            self.assertEqual(
                tuple(saved_predictions["learned"]["orientation_logits"].shape),
                tuple(saved_predictions["oracle"]["orientation_logits"].shape),
            )
            self.assertEqual(
                tuple(saved_predictions["learned"]["orientation_logits"].shape[:2]),
                tuple(heldout_batch["orientations"].shape),
            )

    def test_probe_context_alignment_gate_supports_source_heldout_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_probe_context_alignment_gate(
                config=config,
                train_seed=83,
                eval_seed=183,
                objective_mode="expected_distribution",
                probe_finetune_epochs=1,
                probe_finetune_mode="probe_step_only",
                probe_train_source_subset=[0, 2, 4],
                probe_eval_source_subset=[1, 3, 5],
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "probe_context_predictions.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "probe_context_alignment")
            self.assertEqual(notes["phase2_regime"], "p2a_source_heldout_probe_generalization")
            self.assertEqual(notes["probe_train_source_subset"], [0, 2, 4])
            self.assertEqual(notes["probe_eval_source_subset"], [1, 3, 5])
            self.assertEqual(notes["probe_finetune_epochs"], 1)
            self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")

            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")
            self.assertEqual(heldout_batch["metadata"]["probe_source_subset"].tolist(), [1, 3, 5])

    def test_probe_context_alignment_gate_supports_opt_in_probe_step_finetune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_probe_context_alignment_gate(
                config=config,
                train_seed=91,
                eval_seed=191,
                objective_mode="expected_distribution",
                probe_finetune_epochs=1,
                probe_finetune_mode="probe_step_only",
            )

            run_dir = result.run_dir
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "probe_context_alignment")
            self.assertEqual(notes["probe_finetune_epochs"], 1)
            self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "eval" / "probe_context_predictions.pt").exists())

    def test_phase15_continuation_retention_runner_writes_paired_v2_report_on_fixed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                training=TrainingConfig(batch_size=4, n_epochs=1),
                artifacts=ArtifactConfig(root_dir=Path(tmp_dir)),
            )
            result = run_phase15_continuation_retention(
                config=config,
                train_seed=93,
                eval_seed=193,
            )

            run_dir = result.run_dir
            self.assertTrue((run_dir / "continuation_retention_report.v2.json").exists())
            self.assertTrue((run_dir / "eval" / "standard_heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "fixed_probe_batch.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "phase15_continuation_retention")
            self.assertEqual(notes["subcase"], "dampening")
            self.assertEqual(notes["objective_mode"], "expected_distribution")
            self.assertEqual(notes["probe_finetune_epochs"], 1)
            self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")
            self.assertEqual(notes["standard_heldout_seed"], 193)
            self.assertEqual(notes["fixed_probe_seed"], 193)
            self.assertEqual(notes["continuation_probe_train_seed"], 194)
            self.assertEqual(notes["probe_visible_step_index"], 1)
            self.assertEqual(notes["task_mode_fixed_state"], "orientation_relevant")
            self.assertEqual(notes["prestim_fixed_state"], "none")
            self.assertNotIn("phase2_regime", notes)
            self.assertNotIn("probe_train_source_subset", notes)
            self.assertNotIn("probe_eval_source_subset", notes)
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("challenger_initial_standard_epochs", notes)
            self.assertNotIn("challenger_continuation_probe_updates", notes)
            self.assertNotIn("challenger_continuation_standard_updates", notes)
            self.assertNotIn("challenger_continuation_schedule", notes)

            report = json.loads((run_dir / "continuation_retention_report.v2.json").read_text(encoding="utf-8"))
            self.assertEqual(report["report_version"], "continuation_retention_v2")
            self.assertEqual(report["metric_schema_version"], DIAGNOSTIC_METRIC_SCHEMA_VERSION)
            self.assertEqual(report["classification_rule_version"], DIAGNOSTIC_CLASSIFICATION_RULE_VERSION)
            self.assertEqual(report["diagnostic_schema_version"], DIAGNOSTIC_SCHEMA_VERSION)
            self.assertEqual(report["benchmark_registry_version"], BENCHMARK_REGISTRY_VERSION)
            self.assertEqual(report["source_metric_versions"], [FROZEN_BENCHMARK_METRIC_VERSION])
            self.assertEqual(
                report["metric_versions"],
                {
                    "delta_probe_target_aligned_specificity_contrast__pre_to_post_v2": "v2",
                    "delta_probe_pooled_target_aligned_specificity_contrast__pre_to_post_v2": "v2",
                    "delta_learned_probe_alignment_kl__pre_to_post_v2": "v2",
                    "delta_predictive_loss__pre_to_post_v2": "v2",
                    "delta_predictive_structure_accuracy__pre_to_post_v2": "v2",
                    "delta_nuisance_only_accuracy__pre_to_post_v2": "v2",
                },
            )
            self.assertEqual(report["schedule_name"], "fixed_phase15_probe_continuation")
            self.assertEqual(report["subcase"], "dampening")
            self.assertEqual(report["objective_mode"], "expected_distribution")
            self.assertEqual(report["probe_finetune_epochs"], 1)
            self.assertEqual(report["probe_finetune_mode"], "probe_step_only")
            self.assertTrue(report["same_train_seed_panel_pre_post"])
            self.assertTrue(report["same_standard_predictive_heldout_batch_pre_post"])
            self.assertTrue(report["same_fixed_probe_batch_pre_post"])
            self.assertTrue(report["probe_batch_seen_source_only"])
            self.assertEqual(report["train_seeds"], [93])
            self.assertEqual(report["standard_heldout_seed"], 193)
            self.assertEqual(report["fixed_probe_seed"], 193)
            self.assertEqual(report["continuation_probe_train_seed"], 194)
            self.assertEqual(report["probe_visible_step_index"], 1)
            self.assertEqual(report["task_mode_fixed_state"], "orientation_relevant")
            self.assertEqual(report["prestim_fixed_state"], "none")
            for field_name in (
                "delta_probe_target_aligned_specificity_contrast__pre_to_post_v2",
                "delta_probe_pooled_target_aligned_specificity_contrast__pre_to_post_v2",
                "delta_learned_probe_alignment_kl__pre_to_post_v2",
                "delta_predictive_loss__pre_to_post_v2",
                "delta_predictive_structure_accuracy__pre_to_post_v2",
                "delta_nuisance_only_accuracy__pre_to_post_v2",
                "predictive_loss_pre",
                "predictive_loss_post",
                "predictive_structure_accuracy_pre",
                "predictive_structure_accuracy_post",
                "nuisance_only_accuracy_pre",
                "nuisance_only_accuracy_post",
            ):
                self.assertIn(field_name, report)
            self.assertEqual(report["delta_predictive_structure_accuracy__pre_to_post_v2"], 0.0)
            self.assertEqual(report["delta_nuisance_only_accuracy__pre_to_post_v2"], 0.0)
            self.assertIn("standard_predictive_metrics", report["pre"])
            self.assertIn("standard_predictive_metrics", report["post"])
            self.assertIn("probe_metrics", report["pre"])
            self.assertIn("probe_context_alignment_report", report["post"])

            standard_batch = torch.load(run_dir / "eval" / "standard_heldout_batch.pt")
            probe_batch = torch.load(run_dir / "eval" / "fixed_probe_batch.pt")
            self.assertEqual(standard_batch["metadata"]["generator_seed"], 193)
            self.assertEqual(probe_batch["metadata"]["generator_seed"], 193)
            self.assertEqual(probe_batch["metadata"]["probe_visible_step_index"], 1)
            self.assertTrue(torch.equal(probe_batch["task_mode"], torch.zeros_like(probe_batch["task_mode"])))
            self.assertTrue(torch.equal(probe_batch["prestim_mode"], torch.zeros_like(probe_batch["prestim_mode"])))

    def test_standard_end_localization_manifest_and_nonregression_stop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-01.standard_end_localization.metric-schema.v1",
                "metric_versions": {"standard_end_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.standard_end_localization.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "standard_end_localization_package_version": "2026-04-01.standard_end_localization.package.v1",
                "checks": {"stage0_registry_verification": False},
                "passes": False,
                "complete": False,
            }
            with patch.object(runner_module, "_standard_end_build_nonregression_report", return_value=failing_nonregression_report):
                result = run_standard_end_localization_package(
                    config=config,
                    primary_seed_panel=((11, 1011),),
                    confirmation_seed_panel=((22, 2022),),
                )

            output_root = result.output_root
            self.assertTrue((output_root / "standard_end_localization_manifest.v1.json").exists())
            self.assertTrue((output_root / "standard_end_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "standard_end_anchor_report.v1.json").exists())
            self.assertTrue((output_root / "linear_probe_primary_report.v1.json").exists())
            self.assertTrue((output_root / "localization_delta_report.v1.json").exists())
            self.assertTrue((output_root / "standard_end_localization_verdict.v1.json").exists())
            self.assertFalse((output_root / "linear_probe_confirmation_report.v1.json").exists())

            manifest = json.loads((output_root / "standard_end_localization_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["standard_end_localization_package_version"],
                "2026-04-01.standard_end_localization.package.v1",
            )
            self.assertEqual(manifest["pretraining"]["objective_mode"], "expected_distribution")
            self.assertEqual(manifest["pretraining"]["n_epochs"], 10)
            self.assertEqual(manifest["linear_probe"]["form"], "ridge_linear_readout")
            self.assertEqual(manifest["linear_probe"]["target"], "oracle_log_expected_distribution")
            self.assertEqual(manifest["linear_probe"]["ridge_lambda"], 1e-4)
            self.assertEqual(manifest["linear_probe"]["hidden_state_tensor_name"], "gru_hidden_sequence")
            self.assertEqual(manifest["linear_probe"]["bootstrap_resamples"], 10000)
            self.assertTrue(manifest["stop_rules"]["halt_on_nonregression_failure"])
            self.assertTrue(manifest["stop_rules"]["stop_on_ambiguous_primary"])
            self.assertTrue(manifest["stop_rules"]["no_second_package"])
            self.assertTrue(manifest["stop_rules"]["no_sweep"])

            verdict = json.loads((output_root / "standard_end_localization_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertEqual(verdict["stop_stage"], "nonregression")
            self.assertEqual(verdict["stop_gate"], "S1 non-regression")
            self.assertTrue(verdict["package_complete"])
            self.assertFalse(verdict["execution_valid"])
            self.assertEqual(result.run_ids, [])

    def test_standard_end_localization_run_exports_hidden_states_and_leak_free_auxiliary_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            resolved_config = runner_module._resolve_config(config=config, preset_name="phase1_core")
            store = runner_module.RunStore(resolved_config.artifacts.root_dir)
            generator = runner_module.Phase1ParadigmGenerator(
                resolved_config,
                controlled_sources=runner_module.STAGE3_FULL_CONTROLLED_SOURCES,
            )
            model = runner_module.V1ExpectationModel(resolved_config)
            model.eval()
            assay_runner = runner_module.AssayRunner(resolved_config)

            entry = runner_module._standard_end_execute_localization_run(
                resolved_config=resolved_config,
                store=store,
                generator=generator,
                model=model,
                assay_runner=assay_runner,
                train_seed=31,
                heldout_seed=1031,
                seed_panel="primary",
            )

            run_dir = Path(entry["run_dir"])
            self.assertTrue((run_dir / "probe_design_report.json").exists())
            self.assertTrue((run_dir / "probe_table.json").exists())
            self.assertTrue((run_dir / "probe_metrics.json").exists())
            self.assertTrue((run_dir / "oracle_probe_metrics.json").exists())
            self.assertTrue((run_dir / "probe_context_alignment_report.json").exists())
            self.assertTrue((run_dir / "eval" / "heldout_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "oracle_full_trajectories.pt").exists())
            self.assertTrue((run_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_predictor_hidden_states.standard_end.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_probe_batch.pt").exists())
            self.assertTrue((run_dir / "linear_probe_report.v1.json").exists())
            self.assertTrue((run_dir / "linear_probe_parameters.pt").exists())
            self.assertTrue((run_dir / "linear_probe_hidden_state_probe_table.v2.csv").exists())
            self.assertTrue((run_dir / "linear_probe_hidden_state_diagnostics.v2.json").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "standard_end_localization_native_probe")
            self.assertEqual(notes["checkpoint"], "standard_end")
            self.assertEqual(notes["objective_mode"], "expected_distribution")
            self.assertEqual(notes["initial_standard_epochs"], 10)
            self.assertEqual(notes["probe_finetune_mode"], "disabled")
            self.assertEqual(notes["continuation_probe_updates"], 0)
            self.assertEqual(notes["continuation_standard_updates"], 0)
            self.assertEqual(notes["hidden_state_tensor_name"], "gru_hidden_sequence")
            self.assertEqual(notes["linear_probe_form"], "ridge_linear_readout")
            self.assertTrue(notes["linear_probe_train_eval_disjoint"])
            self.assertNotIn("phase2_regime", notes)
            self.assertNotIn("challenger_candidate", notes)

            eval_hidden = torch.load(run_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            aux_hidden = torch.load(run_dir / "eval" / "aux_predictor_hidden_states.standard_end.pt")
            aux_batch = torch.load(run_dir / "eval" / "aux_probe_batch.pt")
            heldout_batch = torch.load(run_dir / "eval" / "heldout_batch.pt")
            self.assertEqual(eval_hidden["hidden_state_tensor_name"], "gru_hidden_sequence")
            self.assertEqual(aux_hidden["hidden_state_tensor_name"], "gru_hidden_sequence")
            self.assertEqual(eval_hidden["batch_role"], "heldout_eval")
            self.assertEqual(aux_hidden["batch_role"], "auxiliary_probe_fit")
            self.assertEqual(tuple(eval_hidden["hidden_states"].shape[:2]), tuple(heldout_batch["orientations"].shape))
            self.assertEqual(tuple(aux_hidden["hidden_states"].shape[:2]), tuple(aux_batch["orientations"].shape))
            self.assertNotEqual(eval_hidden["generator_seed"], aux_hidden["generator_seed"])
            self.assertEqual(eval_hidden["generator_seed"], heldout_batch["metadata"]["generator_seed"])
            self.assertEqual(aux_hidden["generator_seed"], aux_batch["metadata"]["generator_seed"])

            linear_probe_report = json.loads((run_dir / "linear_probe_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(linear_probe_report["train_eval_disjoint"])
            self.assertEqual(linear_probe_report["heldout_generator_seed"], heldout_batch["metadata"]["generator_seed"])
            self.assertEqual(linear_probe_report["auxiliary_generator_seed"], aux_batch["metadata"]["generator_seed"])

    def test_standard_end_localization_delta_report_is_deterministic(self) -> None:
        primary_entries = [
            self._make_standard_end_localization_entry(
                run_id=f"delta-{idx}",
                train_seed=51 + idx,
                heldout_seed=1051 + idx,
                auxiliary_seed=2051 + idx,
                learned_alignment_kl=1.0,
                linear_alignment_kl=0.2,
                learned_top1=0.4,
                linear_top1=0.8,
            )
            for idx in range(5)
        ]
        first = runner_module._standard_end_build_localization_delta_report(
            primary_run_entries=primary_entries,
            confirmation_run_entries=None,
        )
        second = runner_module._standard_end_build_localization_delta_report(
            primary_run_entries=primary_entries,
            confirmation_run_entries=None,
        )
        self.assertEqual(first, second)
        primary_summary = first["panels"]["primary"]["summary"]
        self.assertIn("bootstrap_ci_95", primary_summary)
        self.assertIn("delta_expected_target_top1_rate__linear_minus_native", primary_summary["bootstrap_ci_95"])
        self.assertIn("alignment_kl_reduction__native_minus_linear", primary_summary["bootstrap_ci_95"])
        self.assertEqual(
            primary_summary["bootstrap_ci_95"]["delta_expected_target_top1_rate__linear_minus_native"]["n_resamples"],
            10000,
        )

    def test_standard_end_localization_package_runs_confirmation_for_localization_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.standard_end_localization.metric-schema.v1",
                "metric_versions": {"standard_end_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.standard_end_localization.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "standard_end_localization_package_version": "2026-04-01.standard_end_localization.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
                "complete": True,
            }

            def fake_run(
                *,
                resolved_config: ExperimentConfig,
                store,
                generator,
                model,
                assay_runner,
                train_seed: int,
                heldout_seed: int,
                seed_panel: str,
            ) -> dict[str, object]:
                del resolved_config, store, generator, model, assay_runner
                entry = self._make_standard_end_localization_entry(
                    run_id=f"{seed_panel}-{train_seed}-{heldout_seed}",
                    train_seed=train_seed,
                    heldout_seed=heldout_seed,
                    auxiliary_seed=heldout_seed + 1,
                )
                entry["seed_panel"] = seed_panel
                return entry

            with patch.object(runner_module, "_standard_end_build_nonregression_report", return_value=passing_nonregression_report), \
                patch.object(runner_module, "_standard_end_execute_localization_run", side_effect=fake_run):
                result = run_standard_end_localization_package(
                    config=config,
                    primary_seed_panel=((11, 1011), (12, 1012), (13, 1013), (14, 1014), (15, 1015)),
                    confirmation_seed_panel=((21, 2021), (22, 2022), (23, 2023), (24, 2024), (25, 2025)),
                )

            self.assertTrue((result.output_root / "linear_probe_confirmation_report.v1.json").exists())
            anchor_report = json.loads((result.output_root / "standard_end_anchor_report.v1.json").read_text(encoding="utf-8"))
            primary_linear_report = json.loads((result.output_root / "linear_probe_primary_report.v1.json").read_text(encoding="utf-8"))
            confirmation_linear_report = json.loads((result.output_root / "linear_probe_confirmation_report.v1.json").read_text(encoding="utf-8"))
            delta_report = json.loads((result.output_root / "localization_delta_report.v1.json").read_text(encoding="utf-8"))
            verdict = json.loads((result.output_root / "standard_end_localization_verdict.v1.json").read_text(encoding="utf-8"))

            self.assertFalse(anchor_report["primary_passes"])
            self.assertTrue(primary_linear_report["passes"])
            self.assertTrue(confirmation_linear_report["passes"])
            self.assertIn("confirmation", delta_report["panels"])
            self.assertEqual(verdict["classification"], "GO-localization-readout")
            self.assertTrue(verdict["confirmation_executed"])
            self.assertEqual(len(result.run_ids), 10)

    def test_readout_alignment_manifest_and_nonregression_stop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-01.readout_alignment.metric-schema.v1",
                "metric_versions": {"readout_alignment_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.readout_alignment.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "readout_alignment_package_version": "2026-04-01.readout_alignment.package.v1",
                "checks": {"stage0_registry_verification": False},
                "passes": False,
                "complete": False,
            }
            with patch.object(runner_module, "_readout_alignment_build_nonregression_report", return_value=failing_nonregression_report):
                result = run_readout_alignment_package(
                    config=config,
                    primary_seed_panel=((11, 1011),),
                    confirmation_seed_panel=((22, 2022),),
                )

            output_root = result.output_root
            self.assertTrue((output_root / "readout_alignment_manifest.v1.json").exists())
            self.assertTrue((output_root / "readout_alignment_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "pre_alignment_primary_report.v1.json").exists())
            self.assertTrue((output_root / "post_alignment_primary_report.v1.json").exists())
            self.assertTrue((output_root / "alignment_integrity_primary_report.v1.json").exists())
            self.assertTrue((output_root / "readout_gap_report.v1.json").exists())
            self.assertTrue((output_root / "readout_alignment_verdict.v1.json").exists())
            self.assertFalse((output_root / "pre_alignment_confirmation_report.v1.json").exists())

            manifest = json.loads((output_root / "readout_alignment_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["readout_alignment_package_version"], "2026-04-01.readout_alignment.package.v1")
            self.assertEqual(manifest["alignment_phase"]["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertEqual(manifest["alignment_phase"]["objective_mode"], "expected_distribution")
            self.assertEqual(manifest["alignment_phase"]["step_count"], 10)
            self.assertEqual(manifest["linear_probe"]["form"], "ridge_linear_readout")
            self.assertEqual(manifest["bootstrap"]["resamples"], 10000)
            self.assertTrue(manifest["stop_rules"]["halt_on_nonregression_failure"])
            self.assertTrue(manifest["stop_rules"]["no_second_package"])
            self.assertTrue(manifest["explicit_statements"]["linear_probe_instrumentation_only"])

            verdict = json.loads((output_root / "readout_alignment_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertEqual(verdict["stop_stage"], "nonregression")
            self.assertEqual(verdict["stop_gate"], "R1 frozen non-regression")
            self.assertTrue(verdict["package_complete"])
            self.assertFalse(verdict["execution_valid"])
            self.assertEqual(result.run_ids, [])

    def test_readout_alignment_run_updates_only_orientation_head_and_exports_pre_post_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            resolved_config = runner_module._resolve_config(config=config, preset_name="phase1_core")
            store = runner_module.RunStore(resolved_config.artifacts.root_dir)
            generator = runner_module.Phase1ParadigmGenerator(
                resolved_config,
                controlled_sources=runner_module.STAGE3_FULL_CONTROLLED_SOURCES,
            )
            model = runner_module.V1ExpectationModel(resolved_config)
            model.eval()
            assay_runner = runner_module.AssayRunner(resolved_config)

            prepared_run = runner_module._readout_alignment_prepare_run(
                resolved_config=resolved_config,
                store=store,
                generator=generator,
                model=model,
                assay_runner=assay_runner,
                train_seed=41,
                heldout_seed=1041,
                seed_panel="primary",
            )
            entry = runner_module._readout_alignment_complete_run(prepared_run)

            run_dir = Path(entry["run_dir"])
            for checkpoint_dirname in ("standard_end_pre_alignment", "standard_end_post_alignment"):
                checkpoint_dir = run_dir / checkpoint_dirname
                self.assertTrue((checkpoint_dir / "probe_design_report.json").exists())
                self.assertTrue((checkpoint_dir / "probe_table.json").exists())
                self.assertTrue((checkpoint_dir / "probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "oracle_probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "probe_context_alignment_report.json").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "oracle_full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_probe_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_report.v1.json").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_parameters.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_probe_table.v2.csv").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_diagnostics.v2.json").exists())

            self.assertTrue((run_dir / "parameter_scope_fingerprints.v1.json").exists())
            self.assertTrue((run_dir / "eval" / "native_head_refit_predictions.pt").exists())
            self.assertTrue((run_dir / "eval" / "native_head_refit_orientation_head_state.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "readout_alignment")
            self.assertEqual(notes["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertTrue(notes["linear_probe_train_eval_disjoint"])
            self.assertEqual(notes["alignment_steps"], 10)
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("phase2_regime", notes)

            fingerprints = json.loads((run_dir / "parameter_scope_fingerprints.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(fingerprints["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertEqual(
                fingerprints["post_alignment"]["trainable_parameter_names"],
                ["orientation_head.weight", "orientation_head.bias"],
            )
            self.assertTrue(fingerprints["checks"]["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["orientation_head_parameter_fingerprint_changed"])
            self.assertTrue(fingerprints["checks"]["exact_trainable_parameter_scope_during_alignment"])

            pre_hidden = torch.load(run_dir / "standard_end_pre_alignment" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            post_hidden = torch.load(run_dir / "standard_end_post_alignment" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            pre_aux_hidden = torch.load(run_dir / "standard_end_pre_alignment" / "eval" / "aux_predictor_hidden_states.standard_end.pt")
            post_aux_hidden = torch.load(run_dir / "standard_end_post_alignment" / "eval" / "aux_predictor_hidden_states.standard_end.pt")
            self.assertTrue(
                torch.allclose(
                    pre_hidden["hidden_states"],
                    post_hidden["hidden_states"],
                    atol=runner_module.READOUT_ALIGNMENT_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    pre_aux_hidden["hidden_states"],
                    post_aux_hidden["hidden_states"],
                    atol=runner_module.READOUT_ALIGNMENT_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertNotEqual(pre_hidden["generator_seed"], pre_aux_hidden["generator_seed"])
            self.assertEqual(pre_hidden["generator_seed"], post_hidden["generator_seed"])
            self.assertEqual(pre_aux_hidden["generator_seed"], post_aux_hidden["generator_seed"])

            integrity = entry["alignment_integrity"]
            self.assertTrue(integrity["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["orientation_head_parameter_fingerprint_changed"])
            self.assertTrue(integrity["heldout_hidden_states_unchanged"])
            self.assertTrue(integrity["auxiliary_hidden_states_unchanged"])
            self.assertLessEqual(integrity["heldout_hidden_state_max_abs_diff"], runner_module.READOUT_ALIGNMENT_HIDDEN_STATE_ATOL)
            self.assertLessEqual(integrity["auxiliary_hidden_state_max_abs_diff"], runner_module.READOUT_ALIGNMENT_HIDDEN_STATE_ATOL)
            self.assertTrue(integrity["auxiliary_eval_disjoint_from_heldout"])

    def test_readout_alignment_gap_report_is_deterministic(self) -> None:
        primary_entries = [
            self._make_readout_alignment_entry(
                run_id=f"readout-gap-{idx}",
                train_seed=61 + idx,
                heldout_seed=1061 + idx,
                auxiliary_probe_fit_seed=2061 + idx,
            )
            for idx in range(5)
        ]
        first = runner_module._readout_alignment_build_gap_report(
            primary_run_entries=primary_entries,
            confirmation_run_entries=None,
        )
        second = runner_module._readout_alignment_build_gap_report(
            primary_run_entries=primary_entries,
            confirmation_run_entries=None,
        )
        self.assertEqual(first, second)
        post_primary_summary = first["panels"]["post_alignment_primary"]["summary"]
        self.assertIn("bootstrap_ci_95", post_primary_summary)
        self.assertIn("delta_expected_target_top1_rate__linear_minus_native", post_primary_summary["bootstrap_ci_95"])
        self.assertEqual(
            post_primary_summary["bootstrap_ci_95"]["delta_expected_target_top1_rate__linear_minus_native"]["n_resamples"],
            10000,
        )

    def test_readout_alignment_package_runs_confirmation_only_after_primary_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.readout_alignment.metric-schema.v1",
                "metric_versions": {"readout_alignment_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.readout_alignment.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "readout_alignment_package_version": "2026-04-01.readout_alignment.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
                "complete": True,
            }

            def fake_prepare(
                *,
                resolved_config: ExperimentConfig,
                store,
                generator,
                model,
                assay_runner,
                train_seed: int,
                heldout_seed: int,
                seed_panel: str,
            ) -> dict[str, object]:
                del resolved_config, store, generator, model, assay_runner
                return {
                    "run_id": f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "run_dir": Path(tmp_dir) / f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "auxiliary_probe_fit_seed": int(heldout_seed + 1),
                    "pre_alignment": self._make_readout_alignment_entry(
                        run_id=f"{seed_panel}-{train_seed}-{heldout_seed}",
                        train_seed=train_seed,
                        heldout_seed=heldout_seed,
                        auxiliary_probe_fit_seed=heldout_seed + 1,
                    )["pre_alignment"],
                }

            def fake_complete(prepared_run: dict[str, object]) -> dict[str, object]:
                entry = self._make_readout_alignment_entry(
                    run_id=str(prepared_run["run_id"]),
                    train_seed=int(prepared_run["train_seed"]),
                    heldout_seed=int(prepared_run["heldout_seed"]),
                    auxiliary_probe_fit_seed=int(prepared_run["auxiliary_probe_fit_seed"]),
                )
                entry["seed_panel"] = prepared_run["seed_panel"]
                return entry

            with patch.object(runner_module, "_readout_alignment_build_nonregression_report", return_value=passing_nonregression_report), \
                patch.object(runner_module, "_readout_alignment_prepare_run", side_effect=fake_prepare), \
                patch.object(runner_module, "_readout_alignment_complete_run", side_effect=fake_complete):
                result = run_readout_alignment_package(
                    config=config,
                    primary_seed_panel=((11, 1011), (12, 1012), (13, 1013), (14, 1014), (15, 1015)),
                    confirmation_seed_panel=((21, 2021), (22, 2022), (23, 2023), (24, 2024), (25, 2025)),
                )

            self.assertTrue((result.output_root / "pre_alignment_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "post_alignment_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "alignment_integrity_confirmation_report.v1.json").exists())
            pre_primary_report = json.loads((result.output_root / "pre_alignment_primary_report.v1.json").read_text(encoding="utf-8"))
            post_primary_report = json.loads((result.output_root / "post_alignment_primary_report.v1.json").read_text(encoding="utf-8"))
            integrity_primary_report = json.loads((result.output_root / "alignment_integrity_primary_report.v1.json").read_text(encoding="utf-8"))
            verdict = json.loads((result.output_root / "readout_alignment_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(pre_primary_report["passes"])
            self.assertTrue(post_primary_report["passes"])
            self.assertTrue(integrity_primary_report["passes"])
            self.assertEqual(verdict["classification"], "GO-readout-aligned")
            self.assertTrue(verdict["confirmation_executed"])
            self.assertEqual(len(result.run_ids), 10)

    def test_linear_probe_transplant_manifest_and_nonregression_stop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-01.linear_probe_transplant.metric-schema.v1",
                "metric_versions": {"linear_probe_transplant_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.linear_probe_transplant.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "linear_probe_transplant_package_version": "2026-04-01.linear_probe_transplant.package.v1",
                "checks": {"stage0_registry_verification": False},
                "passes": False,
                "complete": False,
            }
            with patch.object(
                runner_module,
                "_linear_probe_transplant_build_nonregression_report",
                return_value=failing_nonregression_report,
            ):
                result = run_linear_probe_transplant_package(
                    config=config,
                    primary_seed_panel=((11, 1011),),
                    confirmation_seed_panel=((22, 2022),),
                )

            output_root = result.output_root
            self.assertTrue((output_root / "linear_probe_transplant_manifest.v1.json").exists())
            self.assertTrue((output_root / "linear_probe_transplant_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "pre_transplant_primary_report.v1.json").exists())
            self.assertTrue((output_root / "post_transplant_primary_report.v1.json").exists())
            self.assertTrue((output_root / "transplant_integrity_primary_report.v1.json").exists())
            self.assertTrue((output_root / "transplant_gap_report.v1.json").exists())
            self.assertTrue((output_root / "linear_probe_transplant_verdict.v1.json").exists())
            self.assertFalse((output_root / "pre_transplant_confirmation_report.v1.json").exists())

            manifest = json.loads((output_root / "linear_probe_transplant_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["linear_probe_transplant_package_version"],
                "2026-04-01.linear_probe_transplant.package.v1",
            )
            self.assertEqual(manifest["transplant"]["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertTrue(manifest["transplant"]["no_optimization_after_transplant"])
            self.assertEqual(manifest["linear_probe"]["form"], "ridge_linear_readout")
            self.assertEqual(manifest["linear_probe"]["train_eval_split"], "auxiliary_probe_fit_batch vs heldout_fixed_probe_batch")
            self.assertEqual(manifest["bootstrap"]["resamples"], 10000)
            self.assertTrue(manifest["stop_rules"]["halt_on_nonregression_failure"])
            self.assertTrue(manifest["stop_rules"]["no_second_package"])
            self.assertTrue(manifest["explicit_statements"]["no_optimization_after_transplant"])

            verdict = json.loads((output_root / "linear_probe_transplant_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertEqual(verdict["stop_stage"], "nonregression")
            self.assertEqual(verdict["stop_gate"], "R1 frozen non-regression")
            self.assertTrue(verdict["package_complete"])
            self.assertFalse(verdict["execution_valid"])
            self.assertEqual(result.run_ids, [])

    def test_linear_probe_transplant_run_copies_exact_donor_and_exports_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            resolved_config = runner_module._resolve_config(config=config, preset_name="phase1_core")
            store = runner_module.RunStore(resolved_config.artifacts.root_dir)
            generator = runner_module.Phase1ParadigmGenerator(
                resolved_config,
                controlled_sources=runner_module.STAGE3_FULL_CONTROLLED_SOURCES,
            )
            model = runner_module.V1ExpectationModel(resolved_config)
            model.eval()
            assay_runner = runner_module.AssayRunner(resolved_config)

            prepared_run = runner_module._linear_probe_transplant_prepare_run(
                resolved_config=resolved_config,
                store=store,
                generator=generator,
                model=model,
                assay_runner=assay_runner,
                train_seed=51,
                heldout_seed=1051,
                seed_panel="primary",
            )
            entry = runner_module._linear_probe_transplant_complete_run(prepared_run)

            run_dir = Path(entry["run_dir"])
            for checkpoint_dirname in ("standard_end_pre_transplant", "standard_end_post_transplant"):
                checkpoint_dir = run_dir / checkpoint_dirname
                self.assertTrue((checkpoint_dir / "probe_design_report.json").exists())
                self.assertTrue((checkpoint_dir / "probe_table.json").exists())
                self.assertTrue((checkpoint_dir / "probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "oracle_probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "probe_context_alignment_report.json").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "oracle_full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_probe_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_report.v1.json").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_parameters.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_probe_table.v2.csv").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_diagnostics.v2.json").exists())

            self.assertTrue((run_dir / "parameter_scope_fingerprints.v1.json").exists())
            self.assertTrue((run_dir / "transplant_logit_match_report.v1.json").exists())
            self.assertTrue((run_dir / "eval" / "linear_probe_transplant_head_state.pt").exists())
            self.assertTrue((run_dir / "eval" / "linear_probe_transplant_predictions.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "linear_probe_transplant")
            self.assertEqual(notes["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertTrue(notes["linear_probe_train_eval_disjoint"])
            self.assertTrue(notes["no_optimization_after_transplant"])
            self.assertEqual(notes["transplant_type"], "exact_affine_copy")
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("phase2_regime", notes)

            fingerprints = json.loads((run_dir / "parameter_scope_fingerprints.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(fingerprints["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertTrue(fingerprints["checks"]["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["orientation_head_parameter_fingerprint_changed"])
            self.assertTrue(fingerprints["checks"]["no_optimization_after_transplant"])

            pre_hidden = torch.load(run_dir / "standard_end_pre_transplant" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            post_hidden = torch.load(run_dir / "standard_end_post_transplant" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            pre_aux_hidden = torch.load(run_dir / "standard_end_pre_transplant" / "eval" / "aux_predictor_hidden_states.standard_end.pt")
            post_aux_hidden = torch.load(run_dir / "standard_end_post_transplant" / "eval" / "aux_predictor_hidden_states.standard_end.pt")
            self.assertTrue(
                torch.allclose(
                    pre_hidden["hidden_states"],
                    post_hidden["hidden_states"],
                    atol=runner_module.LINEAR_PROBE_TRANSPLANT_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    pre_aux_hidden["hidden_states"],
                    post_aux_hidden["hidden_states"],
                    atol=runner_module.LINEAR_PROBE_TRANSPLANT_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertNotEqual(pre_hidden["generator_seed"], pre_aux_hidden["generator_seed"])
            self.assertEqual(pre_hidden["generator_seed"], post_hidden["generator_seed"])
            self.assertEqual(pre_aux_hidden["generator_seed"], post_aux_hidden["generator_seed"])

            donor_probe = torch.load(run_dir / "standard_end_pre_transplant" / "linear_probe_parameters.pt")
            transplanted_head = torch.load(run_dir / "eval" / "linear_probe_transplant_head_state.pt")
            self.assertTrue(torch.equal(transplanted_head["weight"], donor_probe["weight"].transpose(0, 1)))
            self.assertTrue(torch.equal(transplanted_head["bias"], donor_probe["bias"]))

            transplant_predictions = torch.load(run_dir / "eval" / "linear_probe_transplant_predictions.pt")
            self.assertEqual(transplant_predictions["heldout_generator_seed"], 1051)
            self.assertTrue(
                torch.allclose(
                    transplant_predictions["native_orientation_logits"],
                    transplant_predictions["standalone_linear_probe_logits"],
                    atol=runner_module.LINEAR_PROBE_TRANSPLANT_LOGIT_MATCH_ATOL,
                    rtol=0.0,
                )
            )

            logit_match_report = json.loads((run_dir / "transplant_logit_match_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(logit_match_report["weight_shape_match"])
            self.assertTrue(logit_match_report["bias_shape_match"])
            self.assertEqual(logit_match_report["max_abs_weight_diff_after_copy"], 0.0)
            self.assertEqual(logit_match_report["max_abs_bias_diff_after_copy"], 0.0)
            self.assertLessEqual(
                logit_match_report["max_abs_logit_diff"],
                runner_module.LINEAR_PROBE_TRANSPLANT_LOGIT_MATCH_ATOL,
            )
            self.assertTrue(logit_match_report["native_logits_match_standalone_linear_probe"])
            self.assertTrue(logit_match_report["no_optimization_after_transplant"])

            integrity = entry["transplant_integrity"]
            self.assertTrue(integrity["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["orientation_head_parameter_fingerprint_changed"])
            self.assertTrue(integrity["transplanted_native_orientation_head_equals_donor"])
            self.assertTrue(integrity["native_orientation_logits_match_standalone_linear_probe"])
            self.assertTrue(integrity["heldout_hidden_states_unchanged"])
            self.assertTrue(integrity["auxiliary_hidden_states_unchanged"])
            self.assertEqual(integrity["optimization_steps_after_transplant"], 0)
            self.assertTrue(integrity["no_optimization_after_transplant"])
            self.assertLessEqual(
                integrity["heldout_hidden_state_max_abs_diff"],
                runner_module.LINEAR_PROBE_TRANSPLANT_HIDDEN_STATE_ATOL,
            )
            self.assertLessEqual(
                integrity["auxiliary_hidden_state_max_abs_diff"],
                runner_module.LINEAR_PROBE_TRANSPLANT_HIDDEN_STATE_ATOL,
            )
            self.assertEqual(
                integrity["predictive_nonregression_passes"],
                (
                    integrity["delta_predictive_loss"] <= runner_module.LINEAR_PROBE_TRANSPLANT_PREDICTIVE_LOSS_DELTA_MAX
                    and integrity["delta_predictive_structure_accuracy"]
                    >= runner_module.LINEAR_PROBE_TRANSPLANT_PREDICTIVE_STRUCTURE_DELTA_MIN
                    and integrity["delta_nuisance_only_accuracy"]
                    <= runner_module.LINEAR_PROBE_TRANSPLANT_NUISANCE_ONLY_DELTA_MAX
                    and integrity["nuisance_only_accuracy_post"]
                    < runner_module.LINEAR_PROBE_TRANSPLANT_NUISANCE_ONLY_POST_MAX
                ),
            )

    def test_linear_probe_transplant_gap_report_is_deterministic(self) -> None:
        primary_entries = [
            self._make_linear_probe_transplant_entry(
                run_id=f"transplant-gap-{idx}",
                train_seed=71 + idx,
                heldout_seed=1071 + idx,
                auxiliary_probe_fit_seed=2071 + idx,
            )
            for idx in range(5)
        ]
        first = runner_module._linear_probe_transplant_build_gap_report(
            primary_run_entries=primary_entries,
            confirmation_run_entries=None,
        )
        second = runner_module._linear_probe_transplant_build_gap_report(
            primary_run_entries=primary_entries,
            confirmation_run_entries=None,
        )
        self.assertEqual(first, second)
        post_primary_summary = first["panels"]["post_transplant_primary"]["summary"]
        self.assertIn("bootstrap_ci_95", post_primary_summary)
        self.assertIn(
            "delta_expected_target_top1_rate__linear_minus_native",
            post_primary_summary["bootstrap_ci_95"],
        )
        self.assertEqual(
            post_primary_summary["bootstrap_ci_95"]["delta_expected_target_top1_rate__linear_minus_native"]["n_resamples"],
            10000,
        )

    def test_linear_probe_transplant_package_runs_confirmation_only_after_primary_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.linear_probe_transplant.metric-schema.v1",
                "metric_versions": {"linear_probe_transplant_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.linear_probe_transplant.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "linear_probe_transplant_package_version": "2026-04-01.linear_probe_transplant.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
                "complete": True,
            }

            def fake_prepare(
                *,
                resolved_config: ExperimentConfig,
                store,
                generator,
                model,
                assay_runner,
                train_seed: int,
                heldout_seed: int,
                seed_panel: str,
            ) -> dict[str, object]:
                del resolved_config, store, generator, model, assay_runner
                return {
                    "run_id": f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "run_dir": Path(tmp_dir) / f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "auxiliary_probe_fit_seed": int(heldout_seed + 1),
                    "pre_transplant": self._make_linear_probe_transplant_entry(
                        run_id=f"{seed_panel}-{train_seed}-{heldout_seed}",
                        train_seed=train_seed,
                        heldout_seed=heldout_seed,
                        auxiliary_probe_fit_seed=heldout_seed + 1,
                    )["pre_transplant"],
                }

            def fake_complete(prepared_run: dict[str, object]) -> dict[str, object]:
                entry = self._make_linear_probe_transplant_entry(
                    run_id=str(prepared_run["run_id"]),
                    train_seed=int(prepared_run["train_seed"]),
                    heldout_seed=int(prepared_run["heldout_seed"]),
                    auxiliary_probe_fit_seed=int(prepared_run["auxiliary_probe_fit_seed"]),
                    post_latent=0.005,
                    post_pooled=0.001,
                    post_native_alignment_kl=1.0,
                    post_native_top1=0.45,
                    post_native_correct_pair_flip=0.3,
                    post_native_within_pair_mass=0.2,
                    post_native_source_bin_kl=1.0,
                    integrity_passes=True,
                    native_logits_match=True,
                )
                entry["seed_panel"] = prepared_run["seed_panel"]
                return entry

            with patch.object(
                runner_module,
                "_linear_probe_transplant_build_nonregression_report",
                return_value=passing_nonregression_report,
            ), patch.object(
                runner_module,
                "_linear_probe_transplant_prepare_run",
                side_effect=fake_prepare,
            ), patch.object(
                runner_module,
                "_linear_probe_transplant_complete_run",
                side_effect=fake_complete,
            ):
                result = run_linear_probe_transplant_package(
                    config=config,
                    primary_seed_panel=((11, 1011), (12, 1012), (13, 1013), (14, 1014), (15, 1015)),
                    confirmation_seed_panel=((21, 2021), (22, 2022), (23, 2023), (24, 2024), (25, 2025)),
                )

            self.assertTrue((result.output_root / "pre_transplant_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "post_transplant_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "transplant_integrity_confirmation_report.v1.json").exists())
            pre_primary_report = json.loads((result.output_root / "pre_transplant_primary_report.v1.json").read_text(encoding="utf-8"))
            post_primary_report = json.loads((result.output_root / "post_transplant_primary_report.v1.json").read_text(encoding="utf-8"))
            integrity_primary_report = json.loads((result.output_root / "transplant_integrity_primary_report.v1.json").read_text(encoding="utf-8"))
            verdict = json.loads((result.output_root / "linear_probe_transplant_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(pre_primary_report["passes"])
            self.assertFalse(post_primary_report["restoration_passes"])
            self.assertTrue(integrity_primary_report["passes"])
            self.assertEqual(verdict["classification"], "GO-localization-native-path-limitation")
            self.assertTrue(verdict["confirmation_executed"])
            self.assertEqual(len(result.run_ids), 10)

    def test_orientation_head_interpolation_manifest_and_nonregression_stop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-01.orientation_head_interpolation.metric-schema.v1",
                "metric_versions": {"orientation_head_interpolation_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.orientation_head_interpolation.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "orientation_head_interpolation_package_version": "2026-04-01.orientation_head_interpolation.package.v1",
                "checks": {"stage0_registry_verification": False},
                "passes": False,
                "complete": False,
            }
            with patch.object(
                runner_module,
                "_orientation_head_interpolation_build_nonregression_report",
                return_value=failing_nonregression_report,
            ):
                result = run_orientation_head_interpolation_package(
                    config=config,
                    primary_seed_panel=((11, 1011),),
                    confirmation_seed_panel=((22, 2022),),
                )

            output_root = result.output_root
            self.assertTrue((output_root / "orientation_head_interpolation_manifest.v1.json").exists())
            self.assertTrue((output_root / "orientation_head_interpolation_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "pre_interpolation_primary_report.v1.json").exists())
            self.assertTrue((output_root / "interpolation_selection_table.v1.json").exists())
            self.assertTrue((output_root / "interpolation_integrity_primary_report.v1.json").exists())
            self.assertTrue((output_root / "post_interpolation_primary_report.v1.json").exists())
            self.assertTrue((output_root / "interpolation_gap_report.v1.json").exists())
            self.assertTrue((output_root / "orientation_head_interpolation_verdict.v1.json").exists())
            self.assertFalse((output_root / "pre_interpolation_confirmation_report.v1.json").exists())

            manifest = json.loads((output_root / "orientation_head_interpolation_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["orientation_head_interpolation_package_version"],
                "2026-04-01.orientation_head_interpolation.package.v1",
            )
            self.assertEqual(
                manifest["candidate_set"],
                [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            )
            self.assertEqual(
                manifest["interpolation"]["parameter_update_scope"],
                ["orientation_head.weight", "orientation_head.bias"],
            )
            self.assertTrue(manifest["explicit_statements"]["no_optimization"])
            self.assertTrue(manifest["explicit_statements"]["global_alpha_selection_once"])
            self.assertEqual(
                manifest["selection_rule"]["tie_break_order"],
                [
                    "higher mean auxiliary fixed-probe expected-target top1",
                    "lower mean auxiliary fixed-probe alignment KL",
                    "smaller alpha",
                ],
            )
            self.assertTrue(manifest["stop_rules"]["halt_if_no_feasible_alpha"])
            self.assertTrue(manifest["stop_rules"]["no_second_package"])

            verdict = json.loads((output_root / "orientation_head_interpolation_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertEqual(verdict["stop_stage"], "nonregression")
            self.assertEqual(verdict["stop_gate"], "R1 frozen non-regression")
            self.assertTrue(verdict["package_complete"])
            self.assertFalse(verdict["execution_valid"])
            self.assertEqual(result.run_ids, [])

    def test_orientation_head_interpolation_selection_tiebreaks_choose_smallest_alpha_last(self) -> None:
        candidates = [
            {
                "alpha": 0.5,
                "feasible": True,
                "aggregate_score": 40,
                "mean_aux_fixed_probe_expected_target_top1": 0.8,
                "mean_aux_fixed_probe_alignment_kl": 0.2,
                "per_seed": [],
            },
            {
                "alpha": 0.375,
                "feasible": True,
                "aggregate_score": 40,
                "mean_aux_fixed_probe_expected_target_top1": 0.8,
                "mean_aux_fixed_probe_alignment_kl": 0.2,
                "per_seed": [],
            },
            {
                "alpha": 0.625,
                "feasible": False,
                "aggregate_score": 50,
                "mean_aux_fixed_probe_expected_target_top1": 1.0,
                "mean_aux_fixed_probe_alignment_kl": 0.0,
                "per_seed": [],
            },
        ]
        selection = runner_module._orientation_head_interpolation_select_alpha_payload(candidates)
        self.assertEqual(selection["selected_alpha"], 0.375)
        selected = next(candidate for candidate in selection["candidates"] if candidate["selected"])
        self.assertEqual(selected["alpha"], 0.375)
        rejected_feasible = next(candidate for candidate in selection["candidates"] if candidate["alpha"] == 0.5)
        rejected_infeasible = next(candidate for candidate in selection["candidates"] if candidate["alpha"] == 0.625)
        self.assertEqual(rejected_feasible["elimination_reason"], "larger_alpha_tiebreak")
        self.assertEqual(rejected_infeasible["elimination_reason"], "predictive_feasibility_failed")

    def test_orientation_head_interpolation_run_applies_exact_formula_and_exports_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            resolved_config = runner_module._resolve_config(config=config, preset_name="phase1_core")
            store = runner_module.RunStore(resolved_config.artifacts.root_dir)
            generator = runner_module.Phase1ParadigmGenerator(
                resolved_config,
                controlled_sources=runner_module.STAGE3_FULL_CONTROLLED_SOURCES,
            )
            model = runner_module.V1ExpectationModel(resolved_config)
            model.eval()
            assay_runner = runner_module.AssayRunner(resolved_config)

            prepared_run = runner_module._orientation_head_interpolation_prepare_run(
                resolved_config=resolved_config,
                store=store,
                generator=generator,
                model=model,
                assay_runner=assay_runner,
                train_seed=61,
                heldout_seed=1061,
                seed_panel="primary",
            )
            selection_table = runner_module._orientation_head_interpolation_build_selection_table(
                prepared_runs=[prepared_run],
            )
            selected_alpha = 0.5 if selection_table["selected_alpha"] is None else float(selection_table["selected_alpha"])
            if selection_table["selected_alpha"] is None:
                selection_table["selected_alpha"] = selected_alpha
                selection_table["no_feasible_alpha"] = False
            entry = runner_module._orientation_head_interpolation_complete_run(
                prepared_run,
                selected_alpha=selected_alpha,
                selection_table=selection_table,
            )

            run_dir = Path(entry["run_dir"])
            for checkpoint_dirname in ("standard_end_pre_interpolation", "standard_end_post_interpolation"):
                checkpoint_dir = run_dir / checkpoint_dirname
                self.assertTrue((checkpoint_dir / "probe_design_report.json").exists())
                self.assertTrue((checkpoint_dir / "probe_table.json").exists())
                self.assertTrue((checkpoint_dir / "probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "oracle_probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "probe_context_alignment_report.json").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "oracle_full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_donor_fit_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_predictive_rehearsal_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_fixed_probe_selection_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_donor_fit_probe_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_predictive_rehearsal_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_fixed_probe_selection_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_report.v1.json").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_parameters.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_probe_table.v2.csv").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_diagnostics.v2.json").exists())

            self.assertTrue((run_dir / "parameter_scope_fingerprints.v1.json").exists())
            self.assertTrue((run_dir / "eval" / "orientation_head_interpolation_state.pt").exists())
            self.assertTrue((run_dir / "eval" / "orientation_head_interpolation_predictions.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "orientation_head_interpolation")
            self.assertEqual(notes["parameter_update_scope"], ["orientation_head.weight", "orientation_head.bias"])
            self.assertTrue(notes["linear_probe_train_eval_disjoint"])
            self.assertTrue(notes["selection_uses_auxiliary_only"])
            self.assertTrue(notes["global_alpha_selection_once"])
            self.assertTrue(notes["no_optimization"])
            self.assertNotIn("challenger_candidate", notes)
            self.assertNotIn("phase2_regime", notes)

            fingerprints = json.loads((run_dir / "parameter_scope_fingerprints.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(fingerprints["checks"]["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["selected_head_matches_interpolation_formula"])
            self.assertTrue(fingerprints["checks"]["no_optimization"])

            interpolation_state = torch.load(run_dir / "eval" / "orientation_head_interpolation_state.pt")
            expected_weight = (
                (1.0 - selected_alpha) * interpolation_state["base_weight"] + selected_alpha * interpolation_state["donor_weight"]
            )
            expected_bias = (
                (1.0 - selected_alpha) * interpolation_state["base_bias"] + selected_alpha * interpolation_state["donor_bias"]
            )
            self.assertTrue(torch.equal(interpolation_state["weight"], expected_weight))
            self.assertTrue(torch.equal(interpolation_state["bias"], expected_bias))
            expected_head_changed = (
                not torch.equal(interpolation_state["weight"], interpolation_state["base_weight"])
                or not torch.equal(interpolation_state["bias"], interpolation_state["base_bias"])
            )
            self.assertEqual(
                fingerprints["checks"]["orientation_head_parameter_fingerprint_changed"],
                expected_head_changed,
            )

            predictions = torch.load(run_dir / "eval" / "orientation_head_interpolation_predictions.pt")
            self.assertEqual(predictions["selected_alpha"], selected_alpha)
            self.assertEqual(predictions["heldout_generator_seed"], 1061)

            integrity = entry["interpolation_integrity"]
            self.assertTrue(integrity["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["heldout_hidden_states_unchanged"])
            self.assertTrue(integrity["selected_head_matches_interpolation_formula"])
            self.assertTrue(integrity["predictive_feasibility_filter_recorded"])
            self.assertTrue(integrity["alpha_selection_table_recorded"])
            self.assertEqual(integrity["optimization_steps"], 0)
            self.assertTrue(integrity["no_optimization"])
            self.assertLessEqual(
                integrity["heldout_hidden_state_max_abs_diff"],
                runner_module.ORIENTATION_HEAD_INTERPOLATION_HIDDEN_STATE_ATOL,
            )

    def test_orientation_head_interpolation_package_stops_clean_negative_when_no_alpha_is_feasible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.orientation_head_interpolation.metric-schema.v1",
                "metric_versions": {"orientation_head_interpolation_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.orientation_head_interpolation.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "orientation_head_interpolation_package_version": "2026-04-01.orientation_head_interpolation.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
                "complete": True,
            }

            def fake_prepare(
                *,
                resolved_config: ExperimentConfig,
                store,
                generator,
                model,
                assay_runner,
                train_seed: int,
                heldout_seed: int,
                seed_panel: str,
            ) -> dict[str, object]:
                del resolved_config, store, generator, model, assay_runner
                return {
                    "run_id": f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "run_dir": Path(tmp_dir) / f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "donor_fit_seed": int(heldout_seed + 1),
                    "predictive_rehearsal_seed": int(heldout_seed + 2),
                    "fixed_probe_selection_seed": int(heldout_seed + 3),
                    "pre_interpolation": self._make_orientation_head_interpolation_entry(
                        run_id=f"{seed_panel}-{train_seed}-{heldout_seed}",
                        train_seed=train_seed,
                        heldout_seed=heldout_seed,
                        donor_fit_seed=heldout_seed + 1,
                        predictive_rehearsal_seed=heldout_seed + 2,
                        fixed_probe_selection_seed=heldout_seed + 3,
                    )["pre_interpolation"],
                }

            no_feasible_selection = {
                "metric_schema_version": "2026-04-01.orientation_head_interpolation.metric-schema.v1",
                "metric_versions": {"interpolation_selection_table": "v1"},
                "classification_rule_version": "2026-04-01.orientation_head_interpolation.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "orientation_head_interpolation_package_version": "2026-04-01.orientation_head_interpolation.package.v1",
                "candidate_set": [0.0, 0.125, 0.25],
                "selection_uses_auxiliary_only": True,
                "heldout_eval_not_used_in_selection": True,
                "global_alpha_selection_once": True,
                "candidates": [],
                "selected_alpha": None,
                "no_feasible_alpha": True,
            }

            with patch.object(
                runner_module,
                "_orientation_head_interpolation_build_nonregression_report",
                return_value=passing_nonregression_report,
            ), patch.object(
                runner_module,
                "_orientation_head_interpolation_prepare_run",
                side_effect=fake_prepare,
            ), patch.object(
                runner_module,
                "_orientation_head_interpolation_build_selection_table",
                return_value=no_feasible_selection,
            ):
                result = run_orientation_head_interpolation_package(
                    config=config,
                    primary_seed_panel=((11, 1011), (12, 1012), (13, 1013), (14, 1014), (15, 1015)),
                    confirmation_seed_panel=((21, 2021),),
                )

            verdict = json.loads((result.output_root / "orientation_head_interpolation_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO clean negative")
            self.assertEqual(verdict["stop_stage"], "selection")
            self.assertEqual(verdict["stop_gate"], "no feasible alpha")
            self.assertFalse(verdict["confirmation_executed"])
            self.assertFalse((result.output_root / "pre_interpolation_confirmation_report.v1.json").exists())

    def test_orientation_head_interpolation_package_runs_confirmation_only_after_primary_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.orientation_head_interpolation.metric-schema.v1",
                "metric_versions": {"orientation_head_interpolation_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.orientation_head_interpolation.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "orientation_head_interpolation_package_version": "2026-04-01.orientation_head_interpolation.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
                "complete": True,
            }

            def fake_prepare(
                *,
                resolved_config: ExperimentConfig,
                store,
                generator,
                model,
                assay_runner,
                train_seed: int,
                heldout_seed: int,
                seed_panel: str,
            ) -> dict[str, object]:
                del resolved_config, store, generator, model, assay_runner
                return {
                    "run_id": f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "run_dir": Path(tmp_dir) / f"{seed_panel}-{train_seed}-{heldout_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "donor_fit_seed": int(heldout_seed + 1),
                    "predictive_rehearsal_seed": int(heldout_seed + 2),
                    "fixed_probe_selection_seed": int(heldout_seed + 3),
                    "pre_interpolation": self._make_orientation_head_interpolation_entry(
                        run_id=f"{seed_panel}-{train_seed}-{heldout_seed}",
                        train_seed=train_seed,
                        heldout_seed=heldout_seed,
                        donor_fit_seed=heldout_seed + 1,
                        predictive_rehearsal_seed=heldout_seed + 2,
                        fixed_probe_selection_seed=heldout_seed + 3,
                    )["pre_interpolation"],
                }

            def fake_complete(prepared_run: dict[str, object], *, selected_alpha: float, selection_table: dict[str, object]) -> dict[str, object]:
                del selection_table
                entry = self._make_orientation_head_interpolation_entry(
                    run_id=str(prepared_run["run_id"]),
                    train_seed=int(prepared_run["train_seed"]),
                    heldout_seed=int(prepared_run["heldout_seed"]),
                    donor_fit_seed=int(prepared_run["donor_fit_seed"]),
                    predictive_rehearsal_seed=int(prepared_run["predictive_rehearsal_seed"]),
                    fixed_probe_selection_seed=int(prepared_run["fixed_probe_selection_seed"]),
                    selected_alpha=selected_alpha,
                )
                entry["seed_panel"] = prepared_run["seed_panel"]
                return entry

            selection_table = {
                "metric_schema_version": "2026-04-01.orientation_head_interpolation.metric-schema.v1",
                "metric_versions": {"interpolation_selection_table": "v1"},
                "classification_rule_version": "2026-04-01.orientation_head_interpolation.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "orientation_head_interpolation_package_version": "2026-04-01.orientation_head_interpolation.package.v1",
                "candidate_set": [0.0, 0.125, 0.25, 0.375],
                "selection_uses_auxiliary_only": True,
                "heldout_eval_not_used_in_selection": True,
                "global_alpha_selection_once": True,
                "candidates": [{"alpha": 0.375, "feasible": True, "aggregate_score": 50, "selected": True, "per_seed": []}],
                "selected_alpha": 0.375,
                "no_feasible_alpha": False,
            }

            with patch.object(
                runner_module,
                "_orientation_head_interpolation_build_nonregression_report",
                return_value=passing_nonregression_report,
            ), patch.object(
                runner_module,
                "_orientation_head_interpolation_prepare_run",
                side_effect=fake_prepare,
            ), patch.object(
                runner_module,
                "_orientation_head_interpolation_build_selection_table",
                return_value=selection_table,
            ), patch.object(
                runner_module,
                "_orientation_head_interpolation_complete_run",
                side_effect=fake_complete,
            ):
                result = run_orientation_head_interpolation_package(
                    config=config,
                    primary_seed_panel=((11, 1011), (12, 1012), (13, 1013), (14, 1014), (15, 1015)),
                    confirmation_seed_panel=((21, 2021), (22, 2022), (23, 2023), (24, 2024), (25, 2025)),
                )

            self.assertTrue((result.output_root / "pre_interpolation_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "interpolation_integrity_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "post_interpolation_confirmation_report.v1.json").exists())
            selection_payload = json.loads((result.output_root / "interpolation_selection_table.v1.json").read_text(encoding="utf-8"))
            verdict = json.loads((result.output_root / "orientation_head_interpolation_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(selection_payload["selected_alpha"], 0.375)
            self.assertTrue(selection_payload["selection_uses_auxiliary_only"])
            self.assertTrue(selection_payload["heldout_eval_not_used_in_selection"])
            self.assertEqual(verdict["classification"], "GO-orientation-head-compromise-aligned")
            self.assertTrue(verdict["confirmation_executed"])
            self.assertEqual(verdict["selected_alpha"], 0.375)
            self.assertEqual(len(result.run_ids), 10)

    def test_precision_target_package_writes_expected_artifacts_and_reports_clean_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_precision_target_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "precision_target_manifest.v1.json").exists())
            self.assertTrue((output_root / "precision_target_support_report.v1.json").exists())
            self.assertTrue((output_root / "precision_target_leakage_audit.v1.json").exists())
            self.assertTrue((output_root / "precision_target_oracle_feasibility_report.v1.json").exists())
            self.assertTrue((output_root / "precision_target_verdict.v1.json").exists())

            manifest = json.loads((output_root / "precision_target_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["precision_target_package_version"], "2026-04-02.precision_target.package.v1")
            self.assertEqual(manifest["candidate_name"], "transition_confidence_margin_v1")
            self.assertEqual(manifest["formula"], "top1(expected_distribution_row) - top2(expected_distribution_row)")
            self.assertEqual(
                manifest["source_fields_used"],
                [
                    "batch.metadata.expected_distribution",
                    "batch.metadata.probe_step_mask",
                    "batch.metadata.probe_valid_mask",
                    "batch.blank_mask",
                    "batch.orientations",
                ],
            )
            self.assertTrue(manifest["no_training_occurred"])
            self.assertTrue(manifest["no_model_parameters_changed"])
            self.assertTrue(manifest["benchmark_metric_fingerprints_unchanged"]["benchmark_registry_payload_matches_in_code"])
            self.assertTrue(manifest["benchmark_metric_fingerprints_unchanged"]["benchmark_fingerprints_payload_matches_in_code"])
            self.assertTrue(manifest["benchmark_metric_fingerprints_unchanged"]["metric_versions_payload_matches_in_code"])

            support = json.loads((output_root / "precision_target_support_report.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(support["panels"]["primary"]["total_scored_rows"], 120)
            self.assertEqual(support["panels"]["confirmation"]["total_scored_rows"], 120)
            self.assertTrue(support["checks"]["target_defined_on_all_scored_rows"])
            self.assertTrue(support["checks"]["target_semantics_fixed_across_panels"])
            self.assertFalse(support["checks"]["target_nonconstant"])
            self.assertTrue(support["checks"]["target_not_identical_to_row_index"])
            self.assertTrue(support["checks"]["target_not_identical_to_source_id"])
            self.assertTrue(support["checks"]["target_not_identical_to_context_id"])
            self.assertEqual(support["overall"]["unique_target_values"], [0.6000000238418579])
            self.assertTrue(support["tautology_audit_summary"]["tautology_unavoidable_on_scored_domain"])

            leakage = json.loads((output_root / "precision_target_leakage_audit.v1.json").read_text(encoding="utf-8"))
            for key, value in leakage["checks"].items():
                self.assertTrue(value, key)

            oracle = json.loads((output_root / "precision_target_oracle_feasibility_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(oracle["checks"]["oracle_target_defined_on_all_scored_rows"])
            self.assertFalse(oracle["checks"]["oracle_target_nonconstant"])
            self.assertTrue(oracle["checks"]["oracle_target_respects_symmetry_groups"])
            self.assertFalse(oracle["checks"]["oracle_target_not_incompatible_with_positive_anchor"])
            self.assertTrue(oracle["weak_or_nearly_constant_on_probe_rows"])

            verdict = json.loads((output_root / "precision_target_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO clean negative")
            self.assertTrue(verdict["execution_valid"])

    def test_precision_target_package_reports_invalid_execution_on_benchmark_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_precision_target_benchmark_status",
                return_value={
                    "benchmark_registry_path": "artifacts/benchmarks/benchmark_registry.v1.json",
                    "benchmark_fingerprints_path": "artifacts/benchmarks/benchmark_fingerprints.v1.json",
                    "metric_versions_path": "artifacts/benchmarks/metric_versions.v1.json",
                    "benchmark_registry_sha256": "bad",
                    "benchmark_fingerprints_sha256": "bad",
                    "metric_versions_sha256": "bad",
                    "benchmark_registry_payload_matches_in_code": False,
                    "benchmark_fingerprints_payload_matches_in_code": True,
                    "metric_versions_payload_matches_in_code": True,
                },
            ):
                result = run_precision_target_package(config=config)

            verdict = json.loads((result.output_root / "precision_target_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertFalse(verdict["execution_valid"])

    def test_readout_class_package_writes_expected_artifacts_and_reports_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_readout_class_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "readout_class_manifest.v1.json").exists())
            self.assertTrue((output_root / "readout_class_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "readout_class_breadth_report.v1.json").exists())
            self.assertTrue((output_root / "readout_class_native_supervision_report.v1.json").exists())
            self.assertTrue((output_root / "readout_class_support_report.v1.json").exists())
            self.assertTrue((output_root / "readout_class_leakage_audit.v1.json").exists())
            self.assertTrue((output_root / "readout_class_oracle_feasibility_report.v1.json").exists())
            self.assertTrue((output_root / "readout_class_verdict.v1.json").exists())

            manifest = json.loads((output_root / "readout_class_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["readout_class_package_version"], "2026-04-02.readout_class.package.v1")
            self.assertEqual(
                manifest["candidate_name"],
                "context_conditioned_affine_residual_orientation_readout_v1",
            )
            self.assertEqual(manifest["equation"], "z(h, c) = (W0 h + b0) + dW[c] h + db[c]")
            self.assertTrue(manifest["no_training_occurred"])
            self.assertTrue(manifest["no_weights_changed"])
            self.assertTrue(manifest["no_heldout_data_used_for_class_selection"])
            self.assertTrue(manifest["no_second_candidate_allowed"])

            nonregression = json.loads((output_root / "readout_class_nonregression_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(nonregression["passes"])
            self.assertTrue(nonregression["checks"]["benchmark_registry_unchanged"])
            self.assertTrue(nonregression["checks"]["historical_bundle_classifications_unchanged"])
            self.assertEqual(
                nonregression["historical_bundle_classifications"]["stage4"]["observed_classification"],
                "NO-GO clean negative",
            )

            breadth = json.loads((output_root / "readout_class_breadth_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(breadth["checks"]["genuinely_broader_than_shared_affine"])
            self.assertTrue(breadth["checks"]["constructive_witness_exists_on_frozen_hidden_state_support"])
            self.assertTrue(breadth["checks"]["shared_affine_residual_nonzero_all_scored_rows"])
            self.assertTrue(breadth["checks"]["shared_affine_residual_nonzero_primary_panel"])
            self.assertTrue(breadth["checks"]["shared_affine_residual_nonzero_confirmation_panel"])
            self.assertIn("standard_end_localization_20260401_run1/runs", breadth["witness"]["support_runs_root"])
            self.assertGreater(
                breadth["shared_affine_least_squares_residual"]["all_scored_rows"]["frobenius_norm"],
                0.0,
            )
            self.assertGreater(
                breadth["shared_affine_least_squares_residual"]["primary_panel"]["frobenius_norm"],
                0.0,
            )
            self.assertGreater(
                breadth["shared_affine_least_squares_residual"]["confirmation_panel"]["frobenius_norm"],
                0.0,
            )
            self.assertEqual(breadth["witness"]["context0_expected_target_orientation"], 1)
            self.assertEqual(breadth["witness"]["context1_expected_target_orientation"], 11)

            support = json.loads((output_root / "readout_class_support_report.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(support["probe_scored_support"]["primary"]["total_scored_rows"], 120)
            self.assertEqual(support["probe_scored_support"]["confirmation"]["total_scored_rows"], 120)
            self.assertTrue(support["checks"]["context_id_exists_on_standard_batches"])
            self.assertTrue(support["checks"]["context_id_exists_on_probe_batches"])
            self.assertTrue(support["checks"]["context_id_varies_on_scored_probe_rows"])
            self.assertTrue(support["checks"]["target_defined_on_all_scored_rows"])
            self.assertTrue(support["checks"]["target_nonconstant"])
            self.assertTrue(support["checks"]["task_mode_fixed_on_scored_probe_domain"])
            self.assertTrue(support["checks"]["prestim_mode_fixed_on_scored_probe_domain"])
            self.assertTrue(support["checks"]["context_is_smallest_live_conditioner"])

            native_supervision = json.loads((output_root / "readout_class_native_supervision_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(
                native_supervision["checks"]["all_future_trainable_parameter_families_use_expected_distribution_only"]
            )
            self.assertTrue(native_supervision["checks"]["no_new_precision_supervision"])
            self.assertTrue(native_supervision["checks"]["no_probe_only_targets"])

            leakage = json.loads((output_root / "readout_class_leakage_audit.v1.json").read_text(encoding="utf-8"))
            for key, value in leakage["checks"].items():
                self.assertTrue(value, key)

            oracle = json.loads((output_root / "readout_class_oracle_feasibility_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(oracle["checks"]["oracle_target_defined_on_all_scored_rows"])
            self.assertTrue(oracle["checks"]["oracle_target_nonconstant"])
            self.assertTrue(oracle["checks"]["oracle_target_respects_symmetry_groups"])
            self.assertTrue(oracle["checks"]["oracle_target_not_incompatible_with_positive_anchor"])
            self.assertFalse(oracle["weak_or_nearly_constant_on_relevant_rows"])

            verdict = json.loads((output_root / "readout_class_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "GO")
            self.assertTrue(verdict["execution_valid"])

    def test_readout_class_package_reports_invalid_execution_on_nonregression_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_readout_class_nonregression_report",
                return_value={
                    "metric_schema_version": "2026-04-02.readout_class.metric-schema.v1",
                    "metric_versions": {"readout_class_nonregression_report": "v1"},
                    "classification_rule_version": "2026-04-02.readout_class.rules.v1",
                    "source_metric_versions": ["benchmark_v1_frozen"],
                    "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                    "readout_class_package_version": "2026-04-02.readout_class.package.v1",
                    "checks": {
                        "benchmark_registry_unchanged": False,
                        "benchmark_fingerprints_unchanged": True,
                        "metric_versions_unchanged": True,
                        "historical_bundle_classifications_unchanged": True,
                    },
                    "passes": False,
                },
            ):
                result = run_readout_class_package(config=config)

            verdict = json.loads((result.output_root / "readout_class_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertFalse(verdict["execution_valid"])

    def test_higher_level_state_package_writes_expected_artifacts_and_reports_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_higher_level_state_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "higher_level_state_manifest.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_semantics_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_breadth_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_native_supervision_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_support_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_leakage_audit.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_oracle_feasibility_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_verdict.v1.json").exists())

            manifest = json.loads((output_root / "higher_level_state_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["higher_level_state_package_version"], "2026-04-02.higher_level_state.package.v1")
            self.assertEqual(
                manifest["candidate_name"],
                "sticky_three_state_source_relative_latent_context_v1",
            )
            self.assertEqual(manifest["latent_state_vocabulary"], ["CW", "CCW", "Neutral"])
            self.assertEqual(manifest["hard_label_tie_rule"], "ties -> Neutral")
            self.assertEqual(manifest["future_trainable_parameter_families"]["state_transition_family"], [])
            self.assertFalse(
                manifest["fixed_family_constraints"]["state_transition_family"]["serialized_constraints"][
                    "free_stickiness_parameter"
                ]
            )
            self.assertTrue(manifest["no_training_occurred"])
            self.assertTrue(manifest["no_weights_changed"])
            self.assertTrue(manifest["no_heldout_selection_occurred"])
            self.assertTrue(manifest["one_candidate_only"])

            nonregression = json.loads(
                (output_root / "higher_level_state_nonregression_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(nonregression["passes"])
            self.assertTrue(nonregression["checks"]["benchmark_registry_unchanged"])
            self.assertTrue(nonregression["checks"]["historical_bundle_classifications_unchanged"])
            self.assertEqual(
                nonregression["historical_bundle_classifications"]["context_residual"]["observed_classification"],
                "NO-GO clean negative",
            )

            semantics = json.loads((output_root / "higher_level_state_semantics_report.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(semantics["latent_state_semantics"]["CW"], "context favors source-relative +1")
            self.assertEqual(semantics["latent_state_semantics"]["CCW"], "context favors source-relative -1")
            self.assertEqual(semantics["latent_state_semantics"]["Neutral"], "locally symmetric or non-directional commitment")
            self.assertEqual(semantics["precision_definition"]["formula"], "1 - H(state_mass_vector) / log(3)")
            self.assertEqual(
                semantics["future_trainable_parameter_families"]["state_transition_family"]["trainable_parameters"],
                [],
            )
            self.assertFalse(
                semantics["fixed_family_constraints"]["state_transition_family"]["serialized_constraints"][
                    "free_stickiness_parameter"
                ]
            )

            breadth = json.loads((output_root / "higher_level_state_breadth_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(breadth["checks"]["genuinely_above_closed_readout_family"])
            self.assertTrue(breadth["checks"]["constructive_frozen_sequence_witness_defined"])
            self.assertTrue(breadth["checks"]["introduces_explicit_latent_state_variable"])
            self.assertTrue(breadth["checks"]["introduces_explicit_state_transition_family"])
            self.assertTrue(breadth["checks"]["not_reducible_to_orientation_head_variants"])
            self.assertGreater(
                breadth["shared_affine_least_squares_residual"]["all_support"]["frobenius_norm"],
                1e-10,
            )
            self.assertGreater(
                breadth["shared_affine_least_squares_residual"]["primary_support"]["frobenius_norm"],
                1e-10,
            )
            self.assertGreater(
                breadth["shared_affine_least_squares_residual"]["confirmation_support"]["frobenius_norm"],
                1e-10,
            )
            self.assertIn("run_summaries", breadth["constructive_witness"]["frozen_hidden_state_support_provenance"])

            native_supervision = json.loads(
                (output_root / "higher_level_state_native_supervision_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(native_supervision["checks"]["state_mass_vector_comes_from_collapsed_expected_distribution"])
            self.assertTrue(native_supervision["checks"]["transition_targets_come_from_consecutive_causal_hard_labels_only"])
            self.assertTrue(native_supervision["checks"]["emissions_are_tied_to_existing_transition_matrices"])
            self.assertTrue(native_supervision["checks"]["no_new_precision_target"])
            self.assertEqual(native_supervision["future_trainable_parameter_families"]["state_transition_family"], [])
            self.assertIn("fixed family constraint", native_supervision["supervision_by_parameter_family"]["state_transition_family"])

            support = json.loads((output_root / "higher_level_state_support_report.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(support["probe_panel_support"]["primary"]["state_counts"]["Neutral"], 0)
            self.assertEqual(support["probe_panel_support"]["confirmation"]["state_counts"]["Neutral"], 0)
            self.assertTrue(support["checks"]["all_three_states_observed_overall"])
            self.assertTrue(support["checks"]["each_state_occurs_under_more_than_one_absolute_source_orientation_overall"])
            self.assertTrue(support["checks"]["transition_support_includes_self_transitions"])
            self.assertTrue(support["checks"]["transition_support_includes_nonself_transitions_into_out_of_neutral"])
            self.assertTrue(support["checks"]["source_relative_emission_support_and_mirror_checks_pass"])
            self.assertTrue(support["checks"]["precision_defined_on_all_scored_rows"])
            self.assertTrue(support["checks"]["precision_nonconstant"])
            self.assertTrue(support["checks"]["precision_not_just_hard_state_lookup_if_possible"])

            leakage = json.loads((output_root / "higher_level_state_leakage_audit.v1.json").read_text(encoding="utf-8"))
            for key, value in leakage["checks"].items():
                self.assertTrue(value, key)

            oracle = json.loads(
                (output_root / "higher_level_state_oracle_feasibility_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(oracle["checks"]["three_state_collapse_defined_on_all_oracle_scored_rows"])
            self.assertTrue(oracle["checks"]["state_masses_vary_where_oracle_structure_varies"])
            self.assertTrue(oracle["checks"]["cw_ccw_preserve_mirror_symmetry"])
            self.assertTrue(oracle["checks"]["neutral_is_genuinely_non_directional_not_threshold_residue"])
            self.assertTrue(oracle["checks"]["oracle_consistent_state_transition_assignment_exists"])
            self.assertTrue(oracle["checks"]["reconstructed_emissions_not_incompatible_with_positive_anchor"])
            self.assertTrue(oracle["checks"]["precision_compatible_with_oracle_concentration_and_not_flat"])

            verdict = json.loads((output_root / "higher_level_state_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "GO")
            self.assertTrue(verdict["execution_valid"])

    def test_higher_level_state_package_reports_invalid_execution_on_nonregression_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_higher_level_state_nonregression_report",
                return_value={
                    "metric_schema_version": "2026-04-02.higher_level_state.metric-schema.v1",
                    "metric_versions": {"higher_level_state_nonregression_report": "v1"},
                    "classification_rule_version": "2026-04-02.higher_level_state.rules.v1",
                    "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION],
                    "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                    "higher_level_state_package_version": "2026-04-02.higher_level_state.package.v1",
                    "checks": {
                        "benchmark_registry_unchanged": False,
                        "benchmark_fingerprints_unchanged": True,
                        "metric_versions_unchanged": True,
                        "historical_bundle_classifications_unchanged": True,
                    },
                    "passes": False,
                },
            ):
                result = run_higher_level_state_package(config=config)

            verdict = json.loads((result.output_root / "higher_level_state_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertFalse(verdict["execution_valid"])

    def test_predictive_state_space_family_package_writes_expected_artifacts_and_is_internally_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_predictive_state_space_family_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "predictive_state_space_family_definition.v1.json").exists())
            self.assertTrue((output_root / "predictive_state_space_family_audit.v1.json").exists())
            self.assertTrue((output_root / "eval" / "predictive_state_space_family_tensors.pt").exists())

            definition = json.loads(
                (output_root / "predictive_state_space_family_definition.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                definition["predictive_state_space_family_package_version"],
                "2026-04-02.predictive_state_space_family.package.v1",
            )
            self.assertEqual(definition["family_id"], "explicit_predictive_state_space_context_v1")
            self.assertEqual(definition["latent_state_vocabulary"], ["CW", "CCW", "Neutral"])
            self.assertEqual(definition["fixed_stickiness_variant"]["rho"], 0.25)
            self.assertEqual(
                definition["continuous_state_parameterization"]["s_t"],
                "p_t(CW) - p_t(CCW)",
            )
            self.assertTrue(definition["no_training_occurred"])
            self.assertTrue(definition["no_weights_changed"])
            self.assertTrue(definition["no_heldout_selection_occurred"])
            self.assertTrue(definition["one_candidate_only"])

            audit = json.loads(
                (output_root / "predictive_state_space_family_audit.v1.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                audit["classification"],
                {"READY_FOR_INDEPENDENT_VALIDATION", "DEFINITION_WEAK_OR_INCOMPLETE"},
            )
            self.assertFalse(audit["scientific_success_claimed"])
            self.assertTrue(audit["nonregression_report"]["passes"])
            self.assertTrue(audit["checks"]["support"]["s_t_defined_on_all_scored_rows"])
            self.assertTrue(audit["checks"]["support"]["c_t_defined_on_all_scored_rows"])
            self.assertTrue(audit["checks"]["support"]["s_t_in_range"])
            self.assertTrue(audit["checks"]["support"]["c_t_in_range"])
            self.assertTrue(audit["checks"]["support"]["abs_s_t_le_c_t"])
            self.assertTrue(audit["checks"]["support"]["implied_probabilities_are_consistent"])
            self.assertTrue(audit["checks"]["support"]["state_space_not_identical_to_row_index"])
            self.assertTrue(audit["checks"]["support"]["state_space_not_identical_to_source_id"])
            self.assertTrue(audit["checks"]["support"]["state_space_not_identical_to_context_id"])
            self.assertTrue(audit["checks"]["support"]["source_relative_emission_reconstruction_path_exercised"])
            self.assertTrue(audit["checks"]["oracle_feasibility"]["source_relative_emission_basis_only"])

            support = audit["support"]
            self.assertGreater(sum(support["overall"]["sticky_state_counts"].values()), 0)
            self.assertGreater(len(support["standard_panels"]["primary"]["transition_counts"]), 0)
            self.assertGreater(len(support["standard_panels"]["confirmation"]["transition_counts"]), 0)
            self.assertIn("Neutral", support["overall"]["sticky_state_counts"])
            self.assertIn(
                "expected_target_top1_rate",
                support["overall"]["probe_emission_reconstruction"],
            )

            tensor_payload = torch.load(
                output_root / "eval" / "predictive_state_space_family_tensors.pt",
                map_location="cpu",
            )
            for panel_name in (
                "probe_primary",
                "probe_confirmation",
                "standard_primary",
                "standard_confirmation",
            ):
                panel = tensor_payload[panel_name]
                s_t = panel["s_t"]
                c_t = panel["c_t"]
                base_mass = panel["base_state_mass"]
                sticky_mass = panel["sticky_state_mass"]
                implied_cw = (c_t + s_t) / 2.0
                implied_ccw = (c_t - s_t) / 2.0
                implied_neutral = 1.0 - c_t
                self.assertTrue(torch.all(sticky_mass >= -1e-8))
                self.assertTrue(torch.all(sticky_mass <= 1.0 + 1e-8))
                self.assertTrue(torch.allclose(sticky_mass.sum(dim=-1), torch.ones_like(c_t), atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_cw, sticky_mass[:, 0], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_ccw, sticky_mass[:, 1], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_neutral, sticky_mass[:, 2], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.all(c_t >= -1e-8))
                self.assertTrue(torch.all(c_t <= 1.0 + 1e-8))
                self.assertTrue(torch.all(torch.abs(s_t) <= c_t + 1e-8))
                self.assertEqual(base_mass.shape[-1], 3)
                self.assertEqual(sticky_mass.shape[-1], 3)
                self.assertEqual(panel["expected_distribution"].shape, panel["reconstructed_orientation_distribution"].shape)

    def test_predictive_state_space_family_package_reports_invalid_execution_on_nonregression_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_predictive_state_space_family_nonregression_report",
                return_value={
                    "metric_schema_version": "2026-04-02.predictive_state_space_family.metric-schema.v1",
                    "metric_versions": {"predictive_state_space_family_nonregression_report": "v1"},
                    "classification_rule_version": "2026-04-02.predictive_state_space_family.rules.v1",
                    "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION],
                    "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                    "predictive_state_space_family_package_version": "2026-04-02.predictive_state_space_family.package.v1",
                    "checks": {
                        "benchmark_registry_unchanged": False,
                        "benchmark_fingerprints_unchanged": True,
                        "metric_versions_unchanged": True,
                        "historical_bundle_classifications_unchanged": True,
                    },
                    "passes": False,
                },
            ):
                result = run_predictive_state_space_family_package(config=config)

            audit = json.loads(
                (result.output_root / "predictive_state_space_family_audit.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["classification"], "INVALID_EXECUTION")
            self.assertFalse(audit["execution_valid"])

    def test_predictive_state_space_family_neutral_transition_flag_requires_into_and_out_of_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            primary_transition_counts = {
                "CW->CW": 2,
                "CCW->CCW": 2,
                "Neutral->Neutral": 1,
                "Neutral->CW": 1,
                "Neutral->CCW": 1,
            }
            confirmation_transition_counts = {}
            overall_transition_counts = dict(primary_transition_counts)
            with patch.object(
                runner_module,
                "_predictive_state_space_family_transition_counts",
                side_effect=[
                    overall_transition_counts,
                    primary_transition_counts,
                    confirmation_transition_counts,
                ],
            ):
                result = run_predictive_state_space_family_package(config=config)

            audit = json.loads(
                (result.output_root / "predictive_state_space_family_audit.v1.json").read_text(encoding="utf-8")
            )
            support = audit["support"]
            serialized_overall_counts = support["overall"]["transition_counts"]
            self.assertEqual(serialized_overall_counts, overall_transition_counts)
            self.assertEqual(
                support["standard_panels"]["primary"]["transition_counts"],
                primary_transition_counts,
            )
            self.assertEqual(
                support["standard_panels"]["confirmation"]["transition_counts"],
                confirmation_transition_counts,
            )
            into_neutral = sum(
                int(serialized_overall_counts.get(key, 0))
                for key in ("CW->Neutral", "CCW->Neutral")
            )
            out_of_neutral = sum(
                int(serialized_overall_counts.get(key, 0))
                for key in ("Neutral->CW", "Neutral->CCW")
            )
            self.assertEqual(into_neutral, 0)
            self.assertGreater(out_of_neutral, 0)
            self.assertFalse(audit["checks"]["support"]["transitions_into_and_out_of_neutral_are_represented"])

    def test_predictive_state_space_family_neutral_transition_flag_is_true_when_both_directions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            primary_transition_counts = {
                "CW->CW": 2,
                "CCW->CCW": 2,
                "Neutral->Neutral": 1,
                "Neutral->CW": 1,
            }
            confirmation_transition_counts = {
                "CW->Neutral": 1,
            }
            overall_transition_counts = {
                "CW->CW": 2,
                "CCW->CCW": 2,
                "Neutral->Neutral": 1,
                "Neutral->CW": 1,
                "CW->Neutral": 1,
            }
            with patch.object(
                runner_module,
                "_predictive_state_space_family_transition_counts",
                side_effect=[
                    overall_transition_counts,
                    primary_transition_counts,
                    confirmation_transition_counts,
                ],
            ):
                result = run_predictive_state_space_family_package(config=config)

            audit = json.loads(
                (result.output_root / "predictive_state_space_family_audit.v1.json").read_text(encoding="utf-8")
            )
            support = audit["support"]
            serialized_overall_counts = support["overall"]["transition_counts"]
            self.assertEqual(serialized_overall_counts, overall_transition_counts)
            into_neutral = sum(
                int(serialized_overall_counts.get(key, 0))
                for key in ("CW->Neutral", "CCW->Neutral")
            )
            out_of_neutral = sum(
                int(serialized_overall_counts.get(key, 0))
                for key in ("Neutral->CW", "Neutral->CCW")
            )
            self.assertGreater(into_neutral, 0)
            self.assertGreater(out_of_neutral, 0)
            self.assertTrue(audit["checks"]["support"]["transitions_into_and_out_of_neutral_are_represented"])

    def test_source_relative_commitment_context_package_writes_expected_artifacts_and_is_formula_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_source_relative_commitment_context_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "source_relative_commitment_context_definition.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_breadth_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_native_supervision_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_support_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_leakage_audit.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_oracle_feasibility_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_commitment_context_verdict.v1.json").exists())
            self.assertTrue((output_root / "eval" / "source_relative_commitment_context_tensors.pt").exists())

            definition = json.loads(
                (output_root / "source_relative_commitment_context_definition.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                definition["source_relative_commitment_context_package_version"],
                "2026-04-02.source_relative_commitment_context.package.v1",
            )
            self.assertEqual(definition["family_id"], "source_relative_commitment_context_v1")
            self.assertEqual(
                definition["state_vocabulary"],
                ["Undecided", "CW_committed", "CCW_committed"],
            )
            self.assertEqual(
                definition["supported_topology"],
                [
                    "Undecided->Undecided",
                    "Undecided->CW_committed",
                    "Undecided->CCW_committed",
                    "CW_committed->CW_committed",
                    "CCW_committed->CCW_committed",
                ],
            )
            self.assertEqual(
                definition["prohibited_topology"],
                [
                    "CW_committed->CCW_committed",
                    "CCW_committed->CW_committed",
                    "CW_committed->Undecided",
                    "CCW_committed->Undecided",
                ],
            )
            self.assertTrue(definition["no_training_occurred"])
            self.assertTrue(definition["no_weights_changed"])
            self.assertTrue(definition["no_calibration_or_heldout_selection"])
            self.assertTrue(definition["one_candidate_only"])
            self.assertNotIn("rho_present", definition)

            support = json.loads(
                (output_root / "source_relative_commitment_context_support_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                support["claimed_supported_topology"],
                definition["supported_topology"],
            )
            self.assertEqual(
                support["claimed_prohibited_topology"],
                definition["prohibited_topology"],
            )
            self.assertGreater(
                support["standard_panels"]["primary"]["strict_undecided_row_count"],
                0,
            )
            self.assertGreater(
                len(support["probe_panels"]["mirror_symmetry"]["rows"]),
                0,
            )
            self.assertIn(
                "Undecided->CW_committed",
                support["overall"]["topology_edge_counts"],
            )
            self.assertIn(
                "CW_committed->Undecided",
                support["overall"]["topology_edge_counts"],
            )
            self.assertTrue(support["checks"]["direction_t_defined_on_all_scored_rows"])
            self.assertTrue(support["checks"]["commitment_t_defined_on_all_scored_rows"])
            self.assertTrue(support["checks"]["direction_t_in_range"])
            self.assertTrue(support["checks"]["commitment_t_in_range"])
            self.assertTrue(support["checks"]["abs_direction_t_le_commitment_t"])
            self.assertTrue(support["checks"]["implied_probabilities_are_consistent"])
            self.assertTrue(support["checks"]["probe_mirror_symmetry_holds"])

            leakage = json.loads(
                (output_root / "source_relative_commitment_context_leakage_audit.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(leakage["checks"]["no_probe_answer_metadata_dependency"])
            self.assertTrue(leakage["checks"]["source_relative_basis_only"])
            self.assertTrue(leakage["checks"]["context_id_used_only_for_audit_slicing"])

            verdict = json.loads(
                (output_root / "source_relative_commitment_context_verdict.v1.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                verdict["classification"],
                {"READY_FOR_INDEPENDENT_VALIDATION", "DEFINITION_WEAK_OR_INCOMPLETE"},
            )
            self.assertTrue(verdict["execution_valid"])
            self.assertFalse(verdict["scientific_success_claimed"])

            tensor_payload = torch.load(
                output_root / "eval" / "source_relative_commitment_context_tensors.pt",
                map_location="cpu",
            )
            for panel_name in (
                "probe_primary",
                "probe_confirmation",
                "standard_primary",
                "standard_confirmation",
            ):
                panel = tensor_payload[panel_name]
                posterior = panel["posterior_state_mass"]
                direction = panel["direction_t"]
                commitment = panel["commitment_t"]
                precision = panel["precision_t"]
                implied_cw = (commitment + direction) / 2.0
                implied_ccw = (commitment - direction) / 2.0
                implied_u = 1.0 - commitment
                self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones_like(direction), atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_cw, posterior[:, 0], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_ccw, posterior[:, 1], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_u, posterior[:, 2], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.all(direction >= -1.0 - 1e-8))
                self.assertTrue(torch.all(direction <= 1.0 + 1e-8))
                self.assertTrue(torch.all(commitment >= -1e-8))
                self.assertTrue(torch.all(commitment <= 1.0 + 1e-8))
                self.assertTrue(torch.all(torch.abs(direction) <= commitment + 1e-8))
                self.assertTrue(torch.allclose(precision, torch.abs(direction), atol=1e-8, rtol=0.0))
                self.assertEqual(panel["expected_distribution"].shape, panel["reconstructed_orientation_distribution"].shape)

    def test_source_relative_commitment_context_package_reports_invalid_execution_on_nonregression_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_source_relative_commitment_context_nonregression_report",
                return_value={
                    "metric_schema_version": "2026-04-02.source_relative_commitment_context.metric-schema.v1",
                    "metric_versions": {"source_relative_commitment_context_nonregression_report": "v1"},
                    "classification_rule_version": "2026-04-02.source_relative_commitment_context.rules.v1",
                    "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION],
                    "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                    "source_relative_commitment_context_package_version": "2026-04-02.source_relative_commitment_context.package.v1",
                    "checks": {
                        "benchmark_registry_unchanged": False,
                        "benchmark_fingerprints_unchanged": True,
                        "metric_versions_unchanged": True,
                        "historical_bundle_classifications_unchanged": True,
                    },
                    "passes": False,
                },
            ):
                result = run_source_relative_commitment_context_package(config=config)

            verdict = json.loads(
                (result.output_root / "source_relative_commitment_context_verdict.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["classification"], "INVALID_EXECUTION")
            self.assertFalse(verdict["execution_valid"])
            self.assertFalse(verdict["scientific_success_claimed"])

    def test_source_relative_signed_direction_belief_package_writes_expected_artifacts_and_is_formula_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_source_relative_signed_direction_belief_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "source_relative_signed_direction_belief_definition.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_breadth_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_native_supervision_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_support_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_leakage_audit.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_oracle_feasibility_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_signed_direction_belief_verdict.v1.json").exists())
            self.assertTrue((output_root / "eval" / "source_relative_signed_direction_belief_tensors.pt").exists())

            definition = json.loads(
                (output_root / "source_relative_signed_direction_belief_definition.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                definition["source_relative_signed_direction_belief_package_version"],
                "2026-04-02.source_relative_signed_direction_belief.package.v1",
            )
            self.assertEqual(definition["family_id"], "source_relative_signed_direction_belief_context_v1")
            self.assertEqual(definition["latent_coordinate"]["belief_t"], "belief_t in [-1, 1]")
            self.assertEqual(
                definition["source_relative_bases"],
                {
                    "E_CW(source)": "transition_matrices[0][source]",
                    "E_CCW(source)": "transition_matrices[1][source]",
                },
            )
            self.assertTrue(definition["no_hidden_third_basis_or_state_component"])
            self.assertTrue(definition["no_training_occurred"])
            self.assertTrue(definition["no_weights_changed"])
            self.assertTrue(definition["no_calibration_or_heldout_selection"])
            self.assertTrue(definition["one_candidate_only"])

            breadth = json.loads(
                (output_root / "source_relative_signed_direction_belief_breadth_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertIn("adjacent_sign_transition_counts", breadth)
            self.assertTrue(breadth["checks"]["both_positive_and_negative_belief_occur_on_standard_rows"])
            self.assertTrue(breadth["checks"]["more_than_two_unique_belief_values_overall"])
            self.assertTrue(breadth["checks"]["at_least_one_sign_preserving_consecutive_transition"])

            support = json.loads(
                (output_root / "source_relative_signed_direction_belief_support_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(support["ambiguity_support_claimed"])
            self.assertTrue(support["checks"]["belief_t_defined_on_all_scored_rows_primary"])
            self.assertTrue(support["checks"]["belief_t_defined_on_all_scored_rows_confirmation"])
            self.assertTrue(support["checks"]["belief_t_in_range"])
            self.assertTrue(support["checks"]["reconstructed_probabilities_in_unit_interval_and_sum_to_one"])
            self.assertTrue(support["checks"]["precision_equals_abs_belief_exactly"])
            self.assertTrue(support["checks"]["belief_t_nonconstant_overall"])
            self.assertTrue(support["checks"]["zero_belief_rows_exist_if_ambiguity_support_is_claimed"])
            self.assertIn("segment_fit_summary", support["overall"])
            self.assertIn("belief_by_sign_unique_values", support["overall"])

            leakage = json.loads(
                (output_root / "source_relative_signed_direction_belief_leakage_audit.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(leakage["checks"]["no_probe_answer_metadata_dependency"])
            self.assertTrue(leakage["checks"]["source_relative_basis_only"])
            self.assertTrue(leakage["checks"]["no_context_id_use_in_posterior_construction"])

            oracle = json.loads(
                (output_root / "source_relative_signed_direction_belief_oracle_feasibility_report.v1.json").read_text(encoding="utf-8")
            )
            self.assertTrue(oracle["checks"]["probe_mirror_symmetry_holds"])
            self.assertTrue(oracle["checks"]["reconstructed_emissions_preserve_positive_anchor_compatibility"])
            self.assertTrue(oracle["checks"]["zero_belief_rows_are_closer_to_midpoint_if_present"])
            self.assertTrue(oracle["checks"]["high_abs_belief_rows_match_their_directional_endpoint"])

            verdict = json.loads(
                (output_root / "source_relative_signed_direction_belief_verdict.v1.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                verdict["classification"],
                {"READY_FOR_INDEPENDENT_VALIDATION", "DEFINITION_WEAK_OR_INCOMPLETE"},
            )
            self.assertTrue(verdict["execution_valid"])
            self.assertFalse(verdict["scientific_success_claimed"])

            tensor_payload = torch.load(
                output_root / "eval" / "source_relative_signed_direction_belief_tensors.pt",
                map_location="cpu",
            )
            for panel_name in (
                "probe_primary",
                "probe_confirmation",
                "standard_primary",
                "standard_confirmation",
            ):
                panel = tensor_payload[panel_name]
                posterior = panel["posterior_directional_probabilities"]
                belief = panel["belief_t"]
                precision = panel["precision_t"]
                implied_cw = (1.0 + belief) / 2.0
                implied_ccw = (1.0 - belief) / 2.0
                self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones_like(belief), atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_cw, posterior[:, 0], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(implied_ccw, posterior[:, 1], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.all(belief >= -1.0 - 1e-8))
                self.assertTrue(torch.all(belief <= 1.0 + 1e-8))
                self.assertTrue(torch.allclose(precision, torch.abs(belief), atol=1e-8, rtol=0.0))
                self.assertEqual(panel["expected_distribution"].shape, panel["reconstructed_orientation_distribution"].shape)

    def test_source_relative_signed_direction_belief_package_reports_invalid_execution_on_nonregression_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_source_relative_signed_direction_belief_nonregression_report",
                return_value={
                    "metric_schema_version": "2026-04-02.source_relative_signed_direction_belief.metric-schema.v1",
                    "metric_versions": {"source_relative_signed_direction_belief_nonregression_report": "v1"},
                    "classification_rule_version": "2026-04-02.source_relative_signed_direction_belief.rules.v1",
                    "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION],
                    "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                    "source_relative_signed_direction_belief_package_version": "2026-04-02.source_relative_signed_direction_belief.package.v1",
                    "checks": {
                        "benchmark_registry_unchanged": False,
                        "benchmark_fingerprints_unchanged": True,
                        "metric_versions_unchanged": True,
                        "historical_bundle_classifications_unchanged": True,
                    },
                    "passes": False,
                },
            ):
                result = run_source_relative_signed_direction_belief_package(config=config)

            verdict = json.loads(
                (result.output_root / "source_relative_signed_direction_belief_verdict.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["classification"], "INVALID_EXECUTION")
            self.assertFalse(verdict["execution_valid"])
            self.assertFalse(verdict["scientific_success_claimed"])

    def test_source_relative_direction_engagement_package_writes_expected_artifacts_and_is_formula_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            result = run_source_relative_direction_engagement_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "source_relative_direction_engagement_definition.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_breadth_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_native_supervision_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_support_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_leakage_audit.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_oracle_feasibility_report.v1.json").exists())
            self.assertTrue((output_root / "source_relative_direction_engagement_verdict.v1.json").exists())
            self.assertTrue((output_root / "eval" / "source_relative_direction_engagement_tensors.pt").exists())

            definition = json.loads(
                (output_root / "source_relative_direction_engagement_definition.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                definition["source_relative_direction_engagement_package_version"],
                "2026-04-02.source_relative_direction_engagement.package.v2",
            )
            self.assertEqual(definition["family_id"], "source_relative_direction_engagement_context_v1")
            self.assertEqual(
                definition["source_relative_bases"],
                {
                    "E_CW(source)": "transition_matrices[0][source]",
                    "E_CCW(source)": "transition_matrices[1][source]",
                    "E_N(source)": "transition_matrices[2][source]",
                },
            )
            self.assertEqual(definition["posterior"]["formula"], "softmax([score_CW, score_CCW, score_N])")
            self.assertEqual(definition["posterior"]["temperature"], 1.0)
            self.assertFalse(definition["forbidden_components"]["rho"])
            self.assertFalse(definition["forbidden_components"]["midpoint_substitute_for_E_N"])
            self.assertTrue(definition["no_training_occurred"])
            self.assertTrue(definition["no_weights_changed"])
            self.assertTrue(definition["one_candidate_only"])

            breadth = json.loads(
                (output_root / "source_relative_direction_engagement_breadth_report.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("best_two_basis_vs_three_basis", breadth)
            self.assertTrue(breadth["checks"]["strict_neutral_rows_exist_on_frozen_standard_rows"])
            transition_report = breadth["adjacent_transition_report"]
            telemetry = breadth["generator_limited_telemetry"]
            self.assertEqual(
                telemetry["at_least_one_nonzero_to_strict_neutral_consecutive_transition_exists"],
                transition_report["nonzero_to_strict_neutral_count"] > 0,
            )
            self.assertEqual(
                telemetry["at_least_one_strict_neutral_to_nonzero_consecutive_transition_exists"],
                transition_report["strict_neutral_to_nonzero_count"] > 0,
            )
            self.assertTrue(breadth["checks"]["at_least_one_sign_preserving_consecutive_transition_exists"])
            self.assertTrue(breadth["checks"]["irreducibility_witness_present_on_primary_strict_neutral_rows"])
            self.assertTrue(breadth["checks"]["irreducibility_witness_present_on_confirmation_strict_neutral_rows"])
            self.assertTrue(breadth["checks"]["pooled_mean_three_basis_improvement_over_best_two_basis_is_positive"])
            self.assertIn("context_id is sampled once per trial", breadth["support_provenance"]["context_sampling_semantics"])

            support = json.loads(
                (output_root / "source_relative_direction_engagement_support_report.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(support["checks"]["direction_t_defined_on_all_scored_rows_primary"])
            self.assertTrue(support["checks"]["direction_t_defined_on_all_scored_rows_confirmation"])
            self.assertTrue(support["checks"]["engagement_t_defined_on_all_scored_rows_primary"])
            self.assertTrue(support["checks"]["engagement_t_defined_on_all_scored_rows_confirmation"])
            self.assertTrue(support["checks"]["direction_t_in_range"])
            self.assertTrue(support["checks"]["engagement_t_in_range"])
            self.assertTrue(support["checks"]["implied_probabilities_in_unit_interval_and_sum_to_one"])
            self.assertTrue(support["checks"]["precision_equals_engagement_times_abs_direction_exactly"])
            self.assertTrue(support["checks"]["strict_neutral_rows_exist_on_frozen_standard_rows"])
            self.assertIn("basis_separation_summary", support["overall"])

            leakage = json.loads(
                (output_root / "source_relative_direction_engagement_leakage_audit.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(leakage["checks"]["no_context_id_use_in_posterior_construction"])
            self.assertTrue(leakage["checks"]["source_relative_basis_only"])
            self.assertTrue(leakage["checks"]["no_future_observation_dependency"])

            oracle = json.loads(
                (output_root / "source_relative_direction_engagement_oracle_feasibility_report.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(oracle["checks"]["probe_mirror_symmetry_holds"])
            self.assertTrue(oracle["checks"]["strict_neutral_rows_are_closer_to_E_N_than_directional_bases"])
            self.assertTrue(oracle["checks"]["reconstructed_emissions_preserve_positive_anchor_compatibility"])
            self.assertTrue(oracle["checks"]["precision_higher_on_directional_rows_than_strict_neutral_rows"])

            verdict = json.loads(
                (output_root / "source_relative_direction_engagement_verdict.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(verdict["classification"], "READY_FOR_INDEPENDENT_VALIDATION")
            self.assertTrue(verdict["execution_valid"])
            self.assertFalse(verdict["scientific_success_claimed"])

            tensor_payload = torch.load(
                output_root / "eval" / "source_relative_direction_engagement_tensors.pt",
                map_location="cpu",
            )
            for panel_name in (
                "probe_primary",
                "probe_confirmation",
                "standard_primary",
                "standard_confirmation",
            ):
                panel = tensor_payload[panel_name]
                posterior = panel["posterior_probabilities"]
                direction = panel["direction_t"]
                engagement = panel["engagement_t"]
                precision = panel["precision_t"]
                p_cw = engagement * (1.0 + direction) / 2.0
                p_ccw = engagement * (1.0 - direction) / 2.0
                p_n = 1.0 - engagement
                zero_engagement = engagement <= 1e-12
                reconstructed_direction = torch.where(
                    zero_engagement,
                    torch.zeros_like(direction),
                    (posterior[:, 0] - posterior[:, 1]) / engagement,
                )
                self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones_like(direction), atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(p_cw, posterior[:, 0], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(p_ccw, posterior[:, 1], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(p_n, posterior[:, 2], atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(direction, reconstructed_direction, atol=1e-8, rtol=0.0))
                self.assertTrue(torch.all(direction >= -1.0 - 1e-8))
                self.assertTrue(torch.all(direction <= 1.0 + 1e-8))
                self.assertTrue(torch.all(engagement >= -1e-8))
                self.assertTrue(torch.all(engagement <= 1.0 + 1e-8))
                self.assertTrue(torch.allclose(precision, engagement * torch.abs(direction), atol=1e-8, rtol=0.0))
                self.assertTrue(torch.allclose(precision, torch.abs(posterior[:, 0] - posterior[:, 1]), atol=1e-8, rtol=0.0))
                self.assertTrue(
                    torch.equal(panel["strict_neutral_row"], posterior[:, 2] > torch.maximum(posterior[:, 0], posterior[:, 1]))
                )
                self.assertEqual(panel["expected_distribution"].shape, panel["reconstructed_orientation_distribution"].shape)

    def test_source_relative_direction_engagement_package_reports_invalid_execution_on_nonregression_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            with patch.object(
                runner_module,
                "_source_relative_direction_engagement_nonregression_report",
                return_value={
                    "metric_schema_version": "2026-04-02.source_relative_direction_engagement.metric-schema.v2",
                    "metric_versions": {"source_relative_direction_engagement_nonregression_report": "v1"},
                    "classification_rule_version": "2026-04-02.source_relative_direction_engagement.rules.v2",
                    "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION],
                    "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                    "source_relative_direction_engagement_package_version": "2026-04-02.source_relative_direction_engagement.package.v2",
                    "checks": {
                        "benchmark_registry_unchanged": False,
                        "benchmark_fingerprints_unchanged": True,
                        "metric_versions_unchanged": True,
                        "historical_bundle_classifications_unchanged": True,
                    },
                    "passes": False,
                },
            ):
                result = run_source_relative_direction_engagement_package(config=config)

            verdict = json.loads(
                (result.output_root / "source_relative_direction_engagement_verdict.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(verdict["classification"], "INVALID_EXECUTION")
            self.assertFalse(verdict["execution_valid"])
            self.assertFalse(verdict["scientific_success_claimed"])

    def test_higher_level_state_train_manifest_and_nonregression_stop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-02.higher_level_state_train.metric-schema.v1",
                "metric_versions": {"higher_level_state_train_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-02.higher_level_state_train.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "higher_level_state_train_package_version": "2026-04-02.higher_level_state_train.package.v1",
                "checks": {"stage0_registry_verification": False},
                "passes": False,
            }
            with patch.object(
                runner_module,
                "_higher_level_state_train_build_nonregression_report",
                return_value=failing_nonregression_report,
            ):
                result = run_higher_level_state_train_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "higher_level_state_train_manifest.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_train_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "pre_refit_primary_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_integrity_primary_report.v1.json").exists())
            self.assertTrue((output_root / "post_refit_primary_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_structure_primary_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_advantage_gap_report.v1.json").exists())
            self.assertTrue((output_root / "higher_level_state_train_verdict.v1.json").exists())

            manifest = json.loads((output_root / "higher_level_state_train_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["higher_level_state_train_package_version"],
                "2026-04-02.higher_level_state_train.package.v1",
            )
            self.assertEqual(
                manifest["trainable_scope"],
                [
                    "predictor.latent_state_head.weight",
                    "predictor.latent_state_head.bias",
                ],
            )
            self.assertEqual(manifest["objective"]["total"], "L_state + L_emit")
            self.assertTrue(manifest["explicit_statements"]["no_sweep"])
            self.assertTrue(manifest["explicit_statements"]["no_second_package"])
            self.assertTrue(manifest["explicit_statements"]["no_teacher_targets"])
            self.assertTrue(manifest["explicit_statements"]["no_heldout_selection"])

            verdict = json.loads((output_root / "higher_level_state_train_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertEqual(verdict["stop_stage"], "nonregression")
            self.assertEqual(verdict["stop_gate"], "R1 frozen non-regression")
            self.assertTrue(verdict["package_complete"])
            self.assertFalse(verdict["execution_valid"])
            self.assertEqual(result.run_ids, [])

    def test_higher_level_state_train_run_updates_only_state_head_and_exports_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            resolved_config = runner_module._resolve_config(config=config, preset_name="phase1_core")
            store = runner_module.RunStore(resolved_config.artifacts.root_dir)
            generator = runner_module.Phase1ParadigmGenerator(
                resolved_config,
                controlled_sources=runner_module.STAGE3_FULL_CONTROLLED_SOURCES,
            )
            model = runner_module.V1ExpectationModel(resolved_config)
            model.eval()
            assay_runner = runner_module.AssayRunner(resolved_config)

            prepared_run = runner_module._higher_level_state_train_prepare_run(
                resolved_config=resolved_config,
                store=store,
                generator=generator,
                model=model,
                assay_runner=assay_runner,
                train_seed=81,
                heldout_seed=1081,
                seed_panel="primary",
            )
            entry = runner_module._higher_level_state_train_complete_run(prepared_run)

            run_dir = Path(entry["run_dir"])
            for checkpoint_dirname in ("standard_end_pre_refit", "standard_end_post_refit"):
                checkpoint_dir = run_dir / checkpoint_dirname
                self.assertTrue((checkpoint_dir / "probe_design_report.json").exists())
                self.assertTrue((checkpoint_dir / "probe_table.json").exists())
                self.assertTrue((checkpoint_dir / "probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "oracle_probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "probe_context_alignment_report.json").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_predictive_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "oracle_full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_predictive_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_report.v1.json").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_parameters.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_probe_table.v2.csv").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_diagnostics.v2.json").exists())
                self.assertTrue((checkpoint_dir / "eval" / "target_state_mass.probe.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "target_state_mass.standard.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "predicted_state_posterior.probe.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "predicted_state_posterior.standard.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "reconstructed_orientation_distribution.probe.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "reconstructed_orientation_distribution.standard.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "posterior_derived_precision.probe.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "posterior_derived_precision.standard.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "transition_audit.standard.pt").exists())

            self.assertTrue((run_dir / "parameter_scope_fingerprints.v1.json").exists())
            self.assertTrue((run_dir / "standard_pretraining_history.json").exists())
            self.assertTrue((run_dir / "higher_level_state_train_history.json").exists())
            self.assertTrue((run_dir / "eval" / "aux_donor_fit_hidden_states.standard_end.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_probe_train_hidden_states.standard_end.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_predictive_train_hidden_states.standard_end.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_donor_fit_probe_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_probe_train_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "aux_predictive_train_batch.pt").exists())
            self.assertTrue((run_dir / "eval" / "higher_level_state_train_state_head_state.pt").exists())
            self.assertTrue((run_dir / "eval" / "higher_level_state_train_predictions.pre.pt").exists())
            self.assertTrue((run_dir / "eval" / "higher_level_state_train_predictions.post.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "higher_level_state_train")
            self.assertEqual(
                notes["parameter_update_scope"],
                [
                    "predictor.latent_state_head.weight",
                    "predictor.latent_state_head.bias",
                ],
            )
            self.assertTrue(notes["no_teacher_targets"])
            self.assertTrue(notes["no_heldout_selection"])

            fingerprints = json.loads((run_dir / "parameter_scope_fingerprints.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(
                fingerprints["parameter_update_scope"],
                [
                    "predictor.latent_state_head.weight",
                    "predictor.latent_state_head.bias",
                ],
            )
            self.assertEqual(
                fingerprints["post_refit"]["trainable_parameter_names"],
                ["latent_state_head.weight", "latent_state_head.bias"],
            )
            self.assertTrue(fingerprints["checks"]["exact_trainable_parameter_scope_during_refit"])
            self.assertTrue(fingerprints["checks"]["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["orientation_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["state_head_parameter_fingerprint_changed"])

            pre_hidden = torch.load(run_dir / "standard_end_pre_refit" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            post_hidden = torch.load(run_dir / "standard_end_post_refit" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            pre_predictive_hidden = torch.load(
                run_dir / "standard_end_pre_refit" / "eval" / "raw_predictor_predictive_hidden_states.standard_end.pt"
            )
            post_predictive_hidden = torch.load(
                run_dir / "standard_end_post_refit" / "eval" / "raw_predictor_predictive_hidden_states.standard_end.pt"
            )
            self.assertTrue(
                torch.allclose(
                    pre_hidden["hidden_states"],
                    post_hidden["hidden_states"],
                    atol=runner_module.HIGHER_LEVEL_STATE_TRAIN_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    pre_predictive_hidden["hidden_states"],
                    post_predictive_hidden["hidden_states"],
                    atol=runner_module.HIGHER_LEVEL_STATE_TRAIN_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertEqual(pre_hidden["generator_seed"], post_hidden["generator_seed"])
            self.assertEqual(pre_predictive_hidden["generator_seed"], post_predictive_hidden["generator_seed"])

            integrity = entry["refit_integrity"]
            self.assertTrue(integrity["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["orientation_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["state_head_parameter_fingerprint_changed"])
            self.assertTrue(integrity["heldout_probe_hidden_states_unchanged"])
            self.assertTrue(integrity["heldout_predictive_hidden_states_unchanged"])
            self.assertTrue(integrity["exact_trainable_parameter_scope_during_refit"])
            self.assertTrue(integrity["scientific_scoring_uses_reconstructed_emissions_only"])
            self.assertLessEqual(
                integrity["heldout_probe_hidden_state_max_abs_diff"],
                runner_module.HIGHER_LEVEL_STATE_TRAIN_HIDDEN_STATE_ATOL,
            )
            self.assertLessEqual(
                integrity["heldout_predictive_hidden_state_max_abs_diff"],
                runner_module.HIGHER_LEVEL_STATE_TRAIN_HIDDEN_STATE_ATOL,
            )
            self.assertIn("heldout_hidden_states", entry["post_refit"])
            self.assertIn("latent_state_logits", entry["post_refit"])
            structure_report = runner_module._higher_level_state_train_build_structure_report(
                seed_panel="primary",
                run_entries=[entry],
            )
            self.assertIn("shared_affine_least_squares_residual", structure_report)
            self.assertGreater(
                structure_report["shared_affine_least_squares_residual"]["n_rows"],
                0,
            )

    def test_higher_level_state_train_package_runs_confirmation_only_after_primary_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-02.higher_level_state_train.metric-schema.v1",
                "metric_versions": {"higher_level_state_train_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-02.higher_level_state_train.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "higher_level_state_train_package_version": "2026-04-02.higher_level_state_train.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
            }

            def fake_prepare(*, train_seed: int, heldout_seed: int, seed_panel: str, **kwargs) -> dict[str, object]:
                return {
                    "run_id": f"{seed_panel}-{train_seed}",
                    "run_dir": Path(tmp_dir) / f"{seed_panel}-{train_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "auxiliary_probe_fit_seed": int(heldout_seed) + 1,
                    "pre_refit": {
                        "probe_metrics": {
                            "probe_target_aligned_specificity_contrast": 0.001,
                            "probe_pooled_target_aligned_specificity_contrast": 0.0,
                        },
                        "oracle_probe_metrics": {
                            "probe_target_aligned_specificity_contrast": 0.04,
                            "probe_pooled_target_aligned_specificity_contrast": 0.01,
                        },
                        "probe_context_alignment_report": {
                            "learned_probe_alignment_kl": 1.0,
                            "learned_probe_expected_target_top1_rate": 0.1,
                            "learned_probe_pair_flip_symmetry_consistency": 1.0,
                        },
                        "hidden_state_diagnostics": {
                            "probe_correct_pair_flip_rate__v2": 0.1,
                            "probe_within_pair_mass__v2": 0.1,
                            "probe_source_bin_kl__v2": 0.1,
                        },
                        "linear_probe_report": {
                            "linear_probe_alignment_kl": 0.1,
                            "linear_probe_expected_target_top1_rate": 1.0,
                            "linear_probe_correct_pair_flip_rate__v2": 1.0,
                            "linear_probe_within_pair_mass__v2": 1.0,
                            "linear_probe_source_bin_kl__v2": 2.5,
                        },
                    },
                }

            def fake_complete(prepared_run: dict[str, object]) -> dict[str, object]:
                return {
                    "run_id": prepared_run["run_id"],
                    "run_dir": prepared_run["run_dir"],
                    "seed_panel": prepared_run["seed_panel"],
                    "train_seed": prepared_run["train_seed"],
                    "heldout_seed": prepared_run["heldout_seed"],
                    "auxiliary_probe_fit_seed": prepared_run["auxiliary_probe_fit_seed"],
                    "pre_refit": prepared_run["pre_refit"],
                    "post_refit": {
                        "heldout_hidden_states": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        "probe_scoring_mask": torch.tensor([[True, True]]),
                        "latent_state_logits": torch.tensor([[[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]], dtype=torch.float32),
                        "probe_metrics": {
                            "probe_target_aligned_specificity_contrast": 0.03,
                            "probe_pooled_target_aligned_specificity_contrast": 0.004,
                        },
                        "oracle_probe_metrics": {
                            "probe_target_aligned_specificity_contrast": 0.04,
                            "probe_pooled_target_aligned_specificity_contrast": 0.01,
                        },
                        "probe_context_alignment_report": {
                            "learned_probe_alignment_kl": 0.1,
                            "learned_probe_expected_target_top1_rate": 0.9,
                            "learned_probe_pair_flip_symmetry_consistency": 1.0,
                        },
                        "hidden_state_diagnostics": {
                            "probe_correct_pair_flip_rate__v2": 0.9,
                            "probe_within_pair_mass__v2": 0.9,
                            "probe_source_bin_kl__v2": 2.5,
                        },
                        "linear_probe_report": {
                            "linear_probe_alignment_kl": 0.1,
                            "linear_probe_expected_target_top1_rate": 0.9,
                            "linear_probe_correct_pair_flip_rate__v2": 0.9,
                            "linear_probe_within_pair_mass__v2": 0.9,
                            "linear_probe_source_bin_kl__v2": 2.5,
                        },
                        "standard_predictive_metrics": {"predictive_loss": 0.0},
                        "standard_sanity": {
                            "predictive_structure": {"accuracy": 1.0},
                            "nuisance_only": {"accuracy": 0.0},
                        },
                        "state_metrics": {
                            "probe": {
                                "mean_state_kl": 0.01,
                                "hard_state_accuracy": 1.0,
                                "mirror_error": 0.0,
                            },
                            "standard": {
                                "hard_state_accuracy": 1.0,
                                "strict_neutral_recall": 1.0,
                                "transition_hard_accuracy": 1.0,
                                "all_three_predicted_states_present": True,
                                "posterior_precision_defined": True,
                                "posterior_precision_nonconstant": True,
                            },
                            "probe_directional_precision_minus_strict_neutral_standard": 0.1,
                        },
                    },
                    "refit_integrity": {
                        "heldout_probe_batch_identity_same_pre_post": True,
                        "heldout_predictive_batch_identity_same_pre_post": True,
                        "auxiliary_batch_identity_same_pre_post": True,
                        "auxiliary_eval_disjoint_from_heldout": True,
                        "recurrent_core_parameter_fingerprint_unchanged": True,
                        "orientation_head_parameter_fingerprint_unchanged": True,
                        "precision_head_parameter_fingerprint_unchanged": True,
                        "state_head_parameter_fingerprint_changed": True,
                        "heldout_probe_hidden_states_unchanged": True,
                        "heldout_predictive_hidden_states_unchanged": True,
                        "heldout_probe_hidden_state_max_abs_diff": 0.0,
                        "heldout_predictive_hidden_state_max_abs_diff": 0.0,
                        "delta_predictive_loss": 0.0,
                        "delta_predictive_structure_accuracy": 0.0,
                        "delta_nuisance_only_accuracy": 0.0,
                        "nuisance_only_accuracy_post": 0.0,
                        "exact_trainable_parameter_scope_during_refit": True,
                        "scientific_scoring_uses_reconstructed_emissions_only": True,
                    },
                    "artifacts_complete": True,
                }

            passes_report = {
                "metric_schema_version": "2026-04-02.higher_level_state_train.metric-schema.v1",
                "metric_versions": {"panel_report": "v1"},
                "classification_rule_version": "2026-04-02.higher_level_state_train.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "higher_level_state_train_package_version": "2026-04-02.higher_level_state_train.package.v1",
                "passes": True,
                "skipped": False,
            }

            with patch.object(
                runner_module,
                "_higher_level_state_train_build_nonregression_report",
                return_value=passing_nonregression_report,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_prepare_run",
                side_effect=fake_prepare,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_complete_run",
                side_effect=fake_complete,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_build_pre_report",
                return_value=passes_report,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_build_integrity_report",
                return_value=passes_report,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_build_post_report",
                return_value=passes_report,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_build_structure_report",
                return_value=passes_report,
            ), patch.object(
                runner_module,
                "_higher_level_state_train_build_advantage_gap_report",
                return_value={"panels": {}, "metric_schema_version": "2026-04-02.higher_level_state_train.metric-schema.v1", "metric_versions": {"higher_level_state_advantage_gap_report": "v1"}, "classification_rule_version": "2026-04-02.higher_level_state_train.rules.v1", "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION], "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION, "higher_level_state_train_package_version": "2026-04-02.higher_level_state_train.package.v1"},
            ):
                result = run_higher_level_state_train_package(
                    config=config,
                    primary_seed_panel=((91, 1091), (92, 1092), (93, 1093), (94, 1094), (95, 1095)),
                    confirmation_seed_panel=((96, 1096), (97, 1097), (98, 1098), (99, 1099), (100, 1100)),
                )

            self.assertTrue((result.output_root / "pre_refit_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "higher_level_state_integrity_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "post_refit_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "higher_level_state_structure_confirmation_report.v1.json").exists())
            verdict = json.loads(
                (result.output_root / "higher_level_state_train_verdict.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["classification"], "GO-higher-level-state-aligned")
            self.assertTrue(verdict["confirmation_executed"])

    def test_context_residual_manifest_and_nonregression_stop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-02.context_residual.metric-schema.v1",
                "metric_versions": {"context_residual_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-02.context_residual.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "context_residual_package_version": "2026-04-02.context_residual.package.v1",
                "checks": {"stage0_registry_verification": False},
                "passes": False,
            }
            with patch.object(
                runner_module,
                "_context_residual_build_nonregression_report",
                return_value=failing_nonregression_report,
            ):
                result = run_context_residual_package(config=config)

            output_root = result.output_root
            self.assertTrue((output_root / "context_residual_manifest.v1.json").exists())
            self.assertTrue((output_root / "context_residual_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "pre_refit_primary_report.v1.json").exists())
            self.assertTrue((output_root / "refit_integrity_primary_report.v1.json").exists())
            self.assertTrue((output_root / "post_refit_primary_report.v1.json").exists())
            self.assertTrue((output_root / "context_residual_structure_report.v1.json").exists())
            self.assertTrue((output_root / "context_residual_advantage_gap_report.v1.json").exists())
            self.assertTrue((output_root / "context_residual_verdict.v1.json").exists())

            manifest = json.loads((output_root / "context_residual_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["context_residual_package_version"], "2026-04-02.context_residual.package.v1")
            self.assertEqual(
                manifest["trainable_scope"],
                [
                    "predictor.context_residual_orientation_weight",
                    "predictor.context_residual_orientation_bias",
                ],
            )
            self.assertEqual(manifest["objective"]["name"], runner_module.EXPECTED_DISTRIBUTION_OBJECTIVE)
            self.assertTrue(manifest["explicit_statements"]["no_sweep"])
            self.assertTrue(manifest["explicit_statements"]["no_second_package"])
            self.assertTrue(manifest["explicit_statements"]["no_teacher_targets"])
            self.assertTrue(manifest["explicit_statements"]["no_heldout_selection"])

            verdict = json.loads((output_root / "context_residual_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "NO-GO invalid execution")
            self.assertEqual(verdict["stop_stage"], "nonregression")
            self.assertEqual(verdict["stop_gate"], "R1 frozen non-regression")
            self.assertTrue(verdict["package_complete"])
            self.assertFalse(verdict["execution_valid"])
            self.assertEqual(result.run_ids, [])

    def test_context_residual_nonregression_report_includes_later_frozen_bundle_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=output_root))
            base_report = {
                "metric_schema_version": "2026-04-01.stage4.metric-schema.v1",
                "metric_versions": {"stage4_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-01.stage4.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "stage4_package_version": "2026-04-01.stage4.package.v1",
                "checks": {
                    "benchmark_registry_files_present": True,
                    "stage0_registry_verification": True,
                    "archived_classifications_verified": True,
                    "benchmark_fingerprints_verified": True,
                    "metric_version_freeze_verified": True,
                    "stage2_calibration_verified": True,
                    "stage2_retention_panel_verified": True,
                    "stage3_oracle_task_definition_passes": True,
                    "full_test_suite": True,
                    "compileall": True,
                },
                "passes": True,
                "complete": True,
            }
            verdict_dir = output_root / "history"
            verdict_dir.mkdir(parents=True, exist_ok=True)
            stage4_verdict = verdict_dir / "stage4_verdict.json"
            stage4_verdict.write_text(
                json.dumps(
                    {
                        "nonregression_passes": True,
                        "anchor_repass_passes": False,
                        "scientific_success_claimed": False,
                    }
                ),
                encoding="utf-8",
            )
            localization_verdict = verdict_dir / "localization_verdict.json"
            localization_verdict.write_text(
                json.dumps({"classification": "GO-localization-readout"}),
                encoding="utf-8",
            )
            expectations = {
                "stage4": {
                    "path": stage4_verdict,
                    "expected_classification": "NO-GO clean negative",
                },
                "standard_end_localization": {
                    "path": localization_verdict,
                    "expected_classification": "GO-localization-readout",
                },
            }

            with patch.object(
                runner_module,
                "_stage4_build_nonregression_report",
                return_value=base_report,
            ), patch.object(
                runner_module,
                "_readout_class_expected_history",
                return_value=expectations,
            ):
                report = runner_module._context_residual_build_nonregression_report(
                    resolved_config=runner_module._resolve_config(config=config, preset_name="phase1_core"),
                    output_root=output_root / "report",
                    benchmark_artifacts_root="artifacts",
                )

            self.assertTrue(report["passes"])
            self.assertTrue(report["checks"]["historical_bundle_classifications_unchanged"])
            self.assertEqual(
                report["historical_bundle_classifications"]["stage4"]["observed_classification"],
                "NO-GO clean negative",
            )
            self.assertEqual(
                report["historical_bundle_classifications"]["standard_end_localization"]["observed_classification"],
                "GO-localization-readout",
            )

            localization_verdict.write_text(
                json.dumps({"classification": "NO-GO clean negative"}),
                encoding="utf-8",
            )
            with patch.object(
                runner_module,
                "_stage4_build_nonregression_report",
                return_value=base_report,
            ), patch.object(
                runner_module,
                "_readout_class_expected_history",
                return_value=expectations,
            ):
                mismatched_report = runner_module._context_residual_build_nonregression_report(
                    resolved_config=runner_module._resolve_config(config=config, preset_name="phase1_core"),
                    output_root=output_root / "report2",
                    benchmark_artifacts_root="artifacts",
                )

            self.assertFalse(mismatched_report["passes"])
            self.assertFalse(mismatched_report["checks"]["historical_bundle_classifications_unchanged"])

    def test_context_residual_run_updates_only_residual_scope_and_exports_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            resolved_config = runner_module._resolve_config(config=config, preset_name="phase1_core")
            store = runner_module.RunStore(resolved_config.artifacts.root_dir)
            generator = runner_module.Phase1ParadigmGenerator(
                resolved_config,
                controlled_sources=runner_module.STAGE3_FULL_CONTROLLED_SOURCES,
            )
            model = runner_module.V1ExpectationModel(resolved_config)
            model.eval()
            assay_runner = runner_module.AssayRunner(resolved_config)

            prepared_run = runner_module._context_residual_prepare_run(
                resolved_config=resolved_config,
                store=store,
                generator=generator,
                model=model,
                assay_runner=assay_runner,
                train_seed=81,
                heldout_seed=1081,
                seed_panel="primary",
            )
            entry = runner_module._context_residual_complete_run(prepared_run)

            run_dir = Path(entry["run_dir"])
            for checkpoint_dirname in ("standard_end_pre_refit", "standard_end_post_refit"):
                checkpoint_dir = run_dir / checkpoint_dirname
                self.assertTrue((checkpoint_dir / "probe_design_report.json").exists())
                self.assertTrue((checkpoint_dir / "probe_table.json").exists())
                self.assertTrue((checkpoint_dir / "probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "oracle_probe_metrics.json").exists())
                self.assertTrue((checkpoint_dir / "probe_context_alignment_report.json").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "heldout_predictive_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "oracle_full_trajectories.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "raw_predictor_predictive_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_predictor_hidden_states.standard_end.pt").exists())
                self.assertTrue((checkpoint_dir / "eval" / "aux_probe_batch.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_report.v1.json").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_parameters.pt").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_probe_table.v2.csv").exists())
                self.assertTrue((checkpoint_dir / "linear_probe_hidden_state_diagnostics.v2.json").exists())

            self.assertTrue((run_dir / "parameter_scope_fingerprints.v1.json").exists())
            self.assertTrue((run_dir / "context_residual_refit_history.json").exists())
            self.assertTrue((run_dir / "eval" / "context_residual_state.pt").exists())
            self.assertTrue((run_dir / "eval" / "context_residual_native_predictions.pre.pt").exists())
            self.assertTrue((run_dir / "eval" / "context_residual_native_predictions.post.pt").exists())
            self.assertTrue((run_dir / "eval" / "context_residual_zero_ablation_predictions.pt").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            notes = manifest["notes"]
            self.assertEqual(notes["gate"], "context_residual")
            self.assertEqual(
                notes["parameter_update_scope"],
                [
                    "predictor.context_residual_orientation_weight",
                    "predictor.context_residual_orientation_bias",
                ],
            )
            self.assertTrue(notes["no_teacher_targets"])
            self.assertTrue(notes["no_heldout_selection"])

            fingerprints = json.loads((run_dir / "parameter_scope_fingerprints.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(
                fingerprints["parameter_update_scope"],
                [
                    "predictor.context_residual_orientation_weight",
                    "predictor.context_residual_orientation_bias",
                ],
            )
            self.assertEqual(
                fingerprints["post_refit"]["trainable_parameter_names"],
                ["context_residual_orientation_weight", "context_residual_orientation_bias"],
            )
            self.assertTrue(fingerprints["checks"]["exact_trainable_parameter_scope_during_refit"])
            self.assertTrue(fingerprints["checks"]["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["orientation_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(fingerprints["checks"]["residual_parameter_fingerprint_changed"])

            pre_hidden = torch.load(run_dir / "standard_end_pre_refit" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            post_hidden = torch.load(run_dir / "standard_end_post_refit" / "eval" / "raw_predictor_hidden_states.standard_end.pt")
            pre_predictive_hidden = torch.load(
                run_dir / "standard_end_pre_refit" / "eval" / "raw_predictor_predictive_hidden_states.standard_end.pt"
            )
            post_predictive_hidden = torch.load(
                run_dir / "standard_end_post_refit" / "eval" / "raw_predictor_predictive_hidden_states.standard_end.pt"
            )
            self.assertTrue(
                torch.allclose(
                    pre_hidden["hidden_states"],
                    post_hidden["hidden_states"],
                    atol=runner_module.CONTEXT_RESIDUAL_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    pre_predictive_hidden["hidden_states"],
                    post_predictive_hidden["hidden_states"],
                    atol=runner_module.CONTEXT_RESIDUAL_HIDDEN_STATE_ATOL,
                    rtol=0.0,
                )
            )
            self.assertEqual(pre_hidden["generator_seed"], post_hidden["generator_seed"])
            self.assertEqual(pre_predictive_hidden["generator_seed"], post_predictive_hidden["generator_seed"])

            integrity = entry["refit_integrity"]
            self.assertTrue(integrity["recurrent_core_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["orientation_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["precision_head_parameter_fingerprint_unchanged"])
            self.assertTrue(integrity["residual_parameter_fingerprint_changed"])
            self.assertTrue(integrity["heldout_fixed_probe_hidden_states_unchanged"])
            self.assertTrue(integrity["heldout_predictive_hidden_states_unchanged"])
            self.assertTrue(integrity["exact_trainable_parameter_scope_during_refit"])
            self.assertTrue(integrity["no_linear_probe_dependency_in_refit"])
            self.assertLessEqual(
                integrity["heldout_fixed_probe_hidden_state_max_abs_diff"],
                runner_module.CONTEXT_RESIDUAL_HIDDEN_STATE_ATOL,
            )
            self.assertLessEqual(
                integrity["heldout_predictive_hidden_state_max_abs_diff"],
                runner_module.CONTEXT_RESIDUAL_HIDDEN_STATE_ATOL,
            )

    def test_context_residual_package_runs_confirmation_only_after_primary_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-02.context_residual.metric-schema.v1",
                "metric_versions": {"context_residual_nonregression_report": "v1"},
                "classification_rule_version": "2026-04-02.context_residual.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "context_residual_package_version": "2026-04-02.context_residual.package.v1",
                "checks": {"stage0_registry_verification": True},
                "passes": True,
            }

            def fake_prepare(*, train_seed: int, heldout_seed: int, seed_panel: str, **kwargs) -> dict[str, object]:
                return {
                    "run_id": f"{seed_panel}-{train_seed}",
                    "run_dir": Path(tmp_dir) / f"{seed_panel}-{train_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "auxiliary_probe_fit_seed": int(heldout_seed) + 1,
                    "pre_refit": self._make_readout_alignment_entry(
                        run_id=f"{seed_panel}-{train_seed}",
                        train_seed=int(train_seed),
                        heldout_seed=int(heldout_seed),
                        auxiliary_probe_fit_seed=int(heldout_seed) + 1,
                    )["pre_alignment"],
                }

            def fake_complete(prepared_run: dict[str, object]) -> dict[str, object]:
                entry = self._make_readout_alignment_entry(
                    run_id=prepared_run["run_id"],
                    train_seed=prepared_run["train_seed"],
                    heldout_seed=prepared_run["heldout_seed"],
                    auxiliary_probe_fit_seed=prepared_run["auxiliary_probe_fit_seed"],
                )
                return {
                    "run_id": prepared_run["run_id"],
                    "run_dir": prepared_run["run_dir"],
                    "seed_panel": prepared_run["seed_panel"],
                    "train_seed": prepared_run["train_seed"],
                    "heldout_seed": prepared_run["heldout_seed"],
                    "auxiliary_probe_fit_seed": prepared_run["auxiliary_probe_fit_seed"],
                    "pre_refit": entry["pre_alignment"],
                    "post_refit": {
                        **entry["post_alignment"],
                        "heldout_hidden_states": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        "predictive_hidden_states": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        "learned_orientation_logits": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        "probe_scoring_mask": torch.tensor([[True, True]]),
                    },
                    "residual_ablated": {
                        **entry["pre_alignment"],
                    },
                    "refit_integrity": {
                        "heldout_fixed_probe_batch_identity_same_pre_post": True,
                        "heldout_predictive_batch_identity_same_pre_post": True,
                        "auxiliary_batch_identity_same_pre_post": True,
                        "auxiliary_eval_disjoint_from_heldout": True,
                        "recurrent_core_parameter_fingerprint_unchanged": True,
                        "orientation_head_parameter_fingerprint_unchanged": True,
                        "precision_head_parameter_fingerprint_unchanged": True,
                        "residual_parameter_fingerprint_changed": True,
                        "heldout_fixed_probe_hidden_states_unchanged": True,
                        "heldout_predictive_hidden_states_unchanged": True,
                        "heldout_fixed_probe_hidden_state_max_abs_diff": 0.0,
                        "heldout_predictive_hidden_state_max_abs_diff": 0.0,
                        "delta_predictive_loss": 0.0,
                        "delta_predictive_structure_accuracy": 0.0,
                        "delta_nuisance_only_accuracy": 0.0,
                        "nuisance_only_accuracy_post": 0.0,
                        "exact_trainable_parameter_scope_during_refit": True,
                        "no_linear_probe_dependency_in_refit": True,
                    },
                    "artifacts_complete": True,
                }

            passing_structure_report = {
                "metric_schema_version": "2026-04-02.context_residual.metric-schema.v1",
                "metric_versions": {"context_residual_structure_report": "v1"},
                "classification_rule_version": "2026-04-02.context_residual.rules.v1",
                "source_metric_versions": [FROZEN_BENCHMARK_METRIC_VERSION, DIAGNOSTIC_SCHEMA_VERSION],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "context_residual_package_version": "2026-04-02.context_residual.package.v1",
                "panels": {
                    "primary": {"passes": True},
                    "confirmation": {"passes": True},
                },
                "skipped": False,
            }

            with patch.object(
                runner_module,
                "_context_residual_build_nonregression_report",
                return_value=passing_nonregression_report,
            ), patch.object(
                runner_module,
                "_context_residual_prepare_run",
                side_effect=fake_prepare,
            ), patch.object(
                runner_module,
                "_context_residual_complete_run",
                side_effect=fake_complete,
            ), patch.object(
                runner_module,
                "_context_residual_build_structure_report",
                return_value=passing_structure_report,
            ):
                result = run_context_residual_package(
                    config=config,
                    primary_seed_panel=((91, 1091), (92, 1092), (93, 1093), (94, 1094), (95, 1095)),
                    confirmation_seed_panel=((96, 1096), (97, 1097), (98, 1098), (99, 1099), (100, 1100)),
                )

            self.assertTrue((result.output_root / "pre_refit_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "refit_integrity_confirmation_report.v1.json").exists())
            self.assertTrue((result.output_root / "post_refit_confirmation_report.v1.json").exists())
            verdict = json.loads((result.output_root / "context_residual_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["classification"], "GO-context-residual-aligned")
            self.assertTrue(verdict["confirmation_executed"])

    def test_stage4_package_runner_writes_nonregression_contract_fields_and_stops_after_failing_anchor_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            calibration_counter = {"value": 0}
            retention_counter = {"value": 0}
            tranche_counter = {"value": 0}

            def fake_validation_command(command: list[str], *, cwd: Path) -> dict[str, object]:
                return {
                    "command": command,
                    "cwd": str(cwd),
                    "ok": True,
                    "returncode": 0,
                    "inherited_from_parent": False,
                    "stdout_tail": ["ok"],
                    "stderr_tail": [],
                }

            def fake_calibration(anchor_id: str, *, artifacts_root: str | Path = "artifacts", output_dir: str | Path | None = None, subcase: str = "dampening") -> dict[str, object]:
                del artifacts_root, subcase
                assert output_dir is not None
                calibration_counter["value"] += 1
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                report_path = output_path / "diagnostic_calibration_report.v2.json"
                report_payload = {
                    "benchmark_anchor_id": anchor_id,
                    "benchmark_classification": json.loads(
                        (Path("artifacts") / "benchmarks" / "benchmark_registry.v1.json").read_text(encoding="utf-8")
                    )["anchors"][anchor_id]["classification"],
                }
                report_path.write_text(json.dumps(report_payload), encoding="utf-8")
                return {"report": report_payload, "report_path": report_path}

            def fake_retention(*, config: ExperimentConfig, train_seed: int | None = None, eval_seed: int | None = None, **_: object) -> SimpleNamespace:
                assert train_seed is not None and eval_seed is not None
                retention_counter["value"] += 1
                run_id = f"retention-{retention_counter['value']}"
                run_dir = config.artifacts.root_dir / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                return SimpleNamespace(
                    manifest=SimpleNamespace(run_id=run_id),
                    run_dir=run_dir,
                )

            def fake_tranche(*, config: ExperimentConfig, **_: object) -> SimpleNamespace:
                tranche_counter["value"] += 1
                run_id = f"tranche-{tranche_counter['value']}"
                run_dir = config.artifacts.root_dir / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                return SimpleNamespace(
                    manifest=SimpleNamespace(run_id=run_id),
                    run_dir=run_dir,
                )

            def fake_stage3(*, config: ExperimentConfig, **_: object) -> SimpleNamespace:
                output_root = config.artifacts.root_dir / "stage3" / "fake"
                output_root.mkdir(parents=True, exist_ok=True)
                return SimpleNamespace(
                    output_root=output_root,
                    task_definition_verdict={"passes": True},
                )

            with patch.object(runner_module, "_stage4_run_validation_command", side_effect=fake_validation_command), \
                patch.object(runner_module, "write_diagnostic_calibration_report_v2", side_effect=fake_calibration), \
                patch.object(runner_module, "run_phase15_continuation_retention", side_effect=fake_retention), \
                patch.object(runner_module, "run_tranche1_experiment", side_effect=fake_tranche), \
                patch.object(runner_module, "run_stage3_task_definition", side_effect=fake_stage3):
                result = run_stage4_package(
                    config=config,
                    primary_seed_panel=((11, 1011),),
                    confirmation_seed_panel=((22, 2022),),
                    run_confirmation_panel=False,
                )

            output_root = result.output_root
            self.assertTrue((output_root / "stage4_package_manifest.v1.json").exists())
            self.assertTrue((output_root / "stage4_nonregression_report.v1.json").exists())
            self.assertTrue((output_root / "stage4_anchor_repass_report.v1.json").exists())
            self.assertTrue((output_root / "stage4_primary_gate_report.v1.json").exists())
            self.assertTrue((output_root / "stage4_verdict.v1.json").exists())

            package_manifest = json.loads((output_root / "stage4_package_manifest.v1.json").read_text(encoding="utf-8"))
            self.assertEqual(package_manifest["stage4_package_version"], "2026-04-01.stage4.package.v1")
            self.assertEqual(package_manifest["stage3_task_version"], "2026-04-01.stage3.task-definition.v2")
            self.assertEqual(package_manifest["changed_files"], ["src/lvc_expectation/runner.py"])
            self.assertEqual(package_manifest["initial_standard_pretraining"]["n_epochs"], 10)
            self.assertEqual(package_manifest["initial_standard_pretraining"]["batch_size"], 32)
            self.assertEqual(package_manifest["initial_standard_pretraining"]["learning_rate"], 1e-3)
            self.assertEqual(package_manifest["initial_standard_pretraining"]["weight_decay"], 1e-4)
            self.assertEqual(package_manifest["continuation_schedule"]["n_cycles"], 10)
            self.assertEqual(package_manifest["continuation_schedule"]["continuation_probe_updates"], 10)
            self.assertEqual(package_manifest["continuation_schedule"]["continuation_standard_updates"], 10)
            self.assertEqual(package_manifest["continuation_schedule"]["batch_mixing_policy"], "strict_1_to_1_alternation")
            self.assertEqual(package_manifest["continuation_schedule"]["probe_finetune_mode"], "probe_step_only")
            self.assertTrue(package_manifest["stop_rules"]["no_sweep"])
            self.assertTrue(package_manifest["stop_rules"]["no_second_package"])
            self.assertTrue(package_manifest["stop_rules"]["halt_on_nonregression_failure"])
            self.assertTrue(package_manifest["stop_rules"]["halt_on_anchor_repass_failure"])
            self.assertTrue(package_manifest["stop_rules"]["primary_pass_before_confirmation"])

            nonregression_report = json.loads((output_root / "stage4_nonregression_report.v1.json").read_text(encoding="utf-8"))
            self.assertIn("passes", nonregression_report)
            self.assertTrue(nonregression_report["complete"])
            self.assertTrue(nonregression_report["passes"])
            self.assertIn("checks", nonregression_report)
            self.assertTrue(nonregression_report["checks"]["archived_classifications_verified"])
            self.assertTrue(nonregression_report["checks"]["benchmark_fingerprints_verified"])
            self.assertTrue(nonregression_report["checks"]["metric_version_freeze_verified"])
            self.assertTrue(nonregression_report["checks"]["full_test_suite"])
            self.assertTrue(nonregression_report["checks"]["compileall"])
            self.assertIn("stage1_tranche_run_id", nonregression_report)
            self.assertEqual(len(nonregression_report["stage2_retention_panel"]["runs"]), 5)
            self.assertTrue(nonregression_report["stage2_retention_panel"]["complete"])
            self.assertIn("benchmark_registry_files", nonregression_report)
            self.assertTrue(nonregression_report["benchmark_registry_files"]["registry_file_exists"])
            self.assertTrue(nonregression_report["benchmark_registry_files"]["fingerprints_file_exists"])
            self.assertTrue(nonregression_report["benchmark_registry_files"]["metric_versions_file_exists"])
            self.assertIn("archived_classification_verification", nonregression_report)
            self.assertIn("benchmark_fingerprint_verification", nonregression_report)
            self.assertIn("metric_version_freeze_verification", nonregression_report)
            self.assertIn("validation_commands", nonregression_report)
            self.assertIn("full_test_suite", nonregression_report["validation_commands"])
            self.assertIn("compileall", nonregression_report["validation_commands"])
            for anchor_id in (
                "phase15_probe_positive_v1",
                "phase2_p2a_primary_negative_v1",
                "phase2_challenger_selection_negative_v1",
            ):
                self.assertIn(anchor_id, nonregression_report["stage2_calibration_reports"])
                self.assertTrue(nonregression_report["stage2_calibration_reports"][anchor_id]["exists"])
                self.assertTrue(nonregression_report["stage2_calibration_reports"][anchor_id]["classification_matches_registry"])
            self.assertTrue(nonregression_report["stage3_task_definition_verdict"]["passes"])
            self.assertEqual(calibration_counter["value"], 3)
            self.assertEqual(retention_counter["value"], 5)
            self.assertEqual(tranche_counter["value"], 1)

            self.assertFalse((output_root / "panels" / "splitA_to_splitB" / "primary" / "stage4_panel_summary.v1.json").exists())
            self.assertFalse((output_root / "panels" / "splitB_to_splitA" / "primary" / "stage4_panel_summary.v1.json").exists())
            self.assertFalse((output_root / "panels" / "splitA_to_splitB" / "confirmation" / "stage4_panel_summary.v1.json").exists())
            self.assertFalse((output_root / "panels" / "splitB_to_splitA" / "confirmation" / "stage4_panel_summary.v1.json").exists())

            anchor_report = json.loads((output_root / "stage4_anchor_repass_report.v1.json").read_text(encoding="utf-8"))
            self.assertIn("checks", anchor_report)
            self.assertIn("passes", anchor_report)
            self.assertFalse(anchor_report["passes"])
            self.assertIn("panel_summaries", anchor_report)
            self.assertIn("primary_panel_summary", anchor_report)
            self.assertIn("confirmation_panel_summary", anchor_report)
            self.assertEqual(anchor_report["primary_panel_summary"]["required_successes"], 5)
            self.assertEqual(anchor_report["confirmation_panel_summary"]["required_successes"], 4)
            self.assertIn("primary_panel_mean_latent_ge_0_016830", anchor_report["checks"])
            self.assertIn("primary_panel_mean_pooled_ge_0_002843", anchor_report["checks"])
            self.assertIn("primary_panel_counts_pass", anchor_report["checks"])
            self.assertIn("confirmation_panel_counts_pass", anchor_report["checks"])
            self.assertIn("no_latent_pooled_sign_reversal", anchor_report["checks"])
            self.assertIn("probe_target_aligned_specificity_contrast_sign_counts", anchor_report["summary"])
            self.assertIn("learned_symmetry_consistency_counts", anchor_report["summary"])
            self.assertIn("mean_learned_oracle_latent_ratio", anchor_report["summary"])
            self.assertIn("mean_learned_oracle_pooled_ratio", anchor_report["summary"])

            primary_gate_report = json.loads((output_root / "stage4_primary_gate_report.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(primary_gate_report["skipped"])
            self.assertEqual(primary_gate_report["stop_reason"], "anchor_repass_failed")

            verdict = json.loads((output_root / "stage4_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertIn("package_complete", verdict)
            self.assertIn("nonregression_passes", verdict)
            self.assertIn("anchor_repass_passes", verdict)
            self.assertIn("primary_gate_passes", verdict)
            self.assertFalse(verdict["stopped_before_anchor"])
            self.assertTrue(verdict["stopped_before_stage3_learned"])
            self.assertFalse(verdict["confirmation_panel_executed"])
            self.assertFalse(verdict["scientific_success_claimed"])

            self.assertEqual(len(result.run_ids), 2)
            gates = {}
            for run_id in result.run_ids:
                run_dir = Path(tmp_dir) / "runs" / run_id
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                notes = manifest["notes"]
                gates[run_id] = notes["gate"]
                for field_name in (
                    "stage3_task_version",
                    "stage4_package_version",
                    "stage3_direction_id",
                    "stage3_seed_panel",
                    "train_seed",
                    "heldout_seed",
                    "train_split_pair_ids",
                    "eval_split_pair_ids",
                    "controlled_sources",
                    "probe_visible_step_index",
                    "objective_mode",
                    "initial_standard_epochs",
                    "continuation_probe_updates",
                    "continuation_standard_updates",
                    "probe_finetune_mode",
                    "standard_rehearsal_seed_base",
                ):
                    self.assertIn(field_name, notes)
                self.assertEqual(notes["objective_mode"], "expected_distribution")
                self.assertEqual(notes["initial_standard_epochs"], 10)
                self.assertEqual(notes["continuation_probe_updates"], 10)
                self.assertEqual(notes["continuation_standard_updates"], 10)
                self.assertEqual(notes["probe_finetune_mode"], "probe_step_only")
                self.assertNotIn("phase2_regime", notes)
                self.assertNotIn("probe_train_source_subset", notes)
                self.assertNotIn("probe_eval_source_subset", notes)
                self.assertNotIn("challenger_candidate", notes)
                self.assertNotIn("challenger_continuation_schedule", notes)

                for artifact_name in (
                    "probe_design_report.json",
                    "probe_table.json",
                    "probe_metrics.json",
                    "oracle_probe_metrics.json",
                    "probe_context_alignment_report.json",
                    "continuation_retention_report.v2.json",
                    "standard_pretraining_history.json",
                    "continuation_history.json",
                ):
                    self.assertTrue((run_dir / artifact_name).exists(), msg=f"{run_id}: missing {artifact_name}")
                for artifact_name in (
                    "eval/heldout_batch.pt",
                    "eval/standard_heldout_batch.pt",
                    "eval/full_trajectories.pt",
                    "eval/oracle_full_trajectories.pt",
                    HIDDEN_STATE_PROBE_TABLE_V2_ARTIFACT,
                    HIDDEN_STATE_DIAGNOSTICS_V2_ARTIFACT,
                ):
                    self.assertTrue((run_dir / artifact_name).exists(), msg=f"{run_id}: missing {artifact_name}")

                probe_metrics = json.loads((run_dir / "probe_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(probe_metrics["stage4_package_version"], "2026-04-01.stage4.package.v1")
                self.assertEqual(probe_metrics["stage3_task_version"], "2026-04-01.stage3.task-definition.v2")
                retention_report = json.loads((run_dir / "continuation_retention_report.v2.json").read_text(encoding="utf-8"))
                self.assertTrue(retention_report["same_standard_predictive_heldout_batch_pre_post"])
                self.assertTrue(retention_report["same_probe_heldout_batch_pre_post"])
                self.assertEqual(retention_report["initial_standard_epochs"], 10)
                self.assertEqual(retention_report["continuation_probe_updates"], 10)
                self.assertEqual(retention_report["continuation_standard_updates"], 10)

            self.assertEqual(set(gates.values()), {"stage4_fixed_probe_anchor_repass"})

    def test_stage4_package_runner_stops_before_anchor_when_nonregression_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            failing_nonregression_report = {
                "metric_schema_version": "2026-04-01.stage4.metric-schema.v1",
                "metric_versions": {"stage4_nonregression": "v1"},
                "classification_rule_version": "2026-04-01.stage4.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "stage3_task_version": "2026-04-01.stage3.task-definition.v2",
                "stage4_package_version": "2026-04-01.stage4.package.v1",
                "checks": {"full_test_suite": False},
                "passes": False,
                "complete": False,
            }
            with patch.object(runner_module, "_stage4_build_nonregression_report", return_value=failing_nonregression_report):
                result = run_stage4_package(
                    config=config,
                    primary_seed_panel=((31, 1031),),
                    confirmation_seed_panel=((41, 1041),),
                    run_confirmation_panel=True,
                )

            self.assertEqual(result.run_ids, [])
            anchor_report = json.loads((result.output_root / "stage4_anchor_repass_report.v1.json").read_text(encoding="utf-8"))
            primary_gate_report = json.loads((result.output_root / "stage4_primary_gate_report.v1.json").read_text(encoding="utf-8"))
            verdict = json.loads((result.output_root / "stage4_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(anchor_report["skipped"])
            self.assertEqual(anchor_report["stop_reason"], "nonregression_failed")
            self.assertTrue(primary_gate_report["skipped"])
            self.assertEqual(primary_gate_report["stop_reason"], "nonregression_failed")
            self.assertTrue(verdict["stopped_before_anchor"])
            self.assertFalse(verdict["package_complete"])

    def test_stage4_package_runner_stops_before_stage3_learned_when_anchor_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.stage4.metric-schema.v1",
                "metric_versions": {"stage4_nonregression": "v1"},
                "classification_rule_version": "2026-04-01.stage4.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "stage3_task_version": "2026-04-01.stage3.task-definition.v2",
                "stage4_package_version": "2026-04-01.stage4.package.v1",
                "checks": {"full_test_suite": True},
                "passes": True,
                "complete": True,
            }
            original_anchor_builder = runner_module._stage4_build_anchor_repass_report
            with patch.object(runner_module, "_stage4_build_nonregression_report", return_value=passing_nonregression_report), \
                patch.object(
                    runner_module,
                    "_stage4_build_anchor_repass_report",
                    side_effect=lambda *, run_entries: {
                        **original_anchor_builder(run_entries=run_entries),
                        "passes": False,
                    },
                ):
                result = run_stage4_package(
                    config=config,
                    primary_seed_panel=((31, 1031),),
                    confirmation_seed_panel=((41, 1041),),
                    run_confirmation_panel=True,
                )

            gates = {
                json.loads((Path(tmp_dir) / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))["notes"]["gate"]
                for run_id in result.run_ids
            }
            self.assertEqual(gates, {"stage4_fixed_probe_anchor_repass"})
            primary_gate_report = json.loads((result.output_root / "stage4_primary_gate_report.v1.json").read_text(encoding="utf-8"))
            verdict = json.loads((result.output_root / "stage4_verdict.v1.json").read_text(encoding="utf-8"))
            self.assertTrue(primary_gate_report["skipped"])
            self.assertEqual(primary_gate_report["stop_reason"], "anchor_repass_failed")
            self.assertTrue(verdict["stopped_before_stage3_learned"])
            self.assertFalse((result.output_root / "panels" / "splitA_to_splitB" / "primary" / "stage4_panel_summary.v1.json").exists())

    def test_stage4_learned_panel_summary_uses_four_of_five_and_requires_symmetry_count(self) -> None:
        run_entries = [
            self._make_stage4_probe_run_entry(
                run_id=f"good-{idx}",
                train_seed=100 + idx,
                heldout_seed=1000 + idx,
            )
            for idx in range(4)
        ]
        run_entries.append(
            self._make_stage4_probe_run_entry(
                run_id="count-failure",
                train_seed=104,
                heldout_seed=1004,
                latent=-0.005,
                pooled=-0.001,
                learned_alignment_kl=0.9,
                learned_top1=0.6,
                learned_symmetry=0.0,
                correct_pair_flip=0.2,
                within_pair_mass=0.2,
                source_bin_kl=1.0,
            )
        )
        passing_summary = runner_module._stage4_build_panel_summary(
            direction_id="splitA_to_splitB",
            seed_panel="primary",
            run_entries=run_entries,
        )
        self.assertEqual(passing_summary["checks"]["required_successes"], 4)
        self.assertEqual(passing_summary["checks"]["learned_symmetry_consistency_count"], 4)
        self.assertTrue(passing_summary["passes"])

        symmetry_fail_entries = [
            self._make_stage4_probe_run_entry(
                run_id=f"symmetry-{idx}",
                train_seed=110 + idx,
                heldout_seed=1010 + idx,
                learned_symmetry=1.0 if idx < 3 else 0.0,
            )
            for idx in range(5)
        ]
        symmetry_failing_summary = runner_module._stage4_build_panel_summary(
            direction_id="splitA_to_splitB",
            seed_panel="primary",
            run_entries=symmetry_fail_entries,
        )
        self.assertEqual(symmetry_failing_summary["checks"]["learned_symmetry_consistency_count"], 3)
        self.assertFalse(symmetry_failing_summary["checks"]["count_thresholds_met"])
        self.assertFalse(symmetry_failing_summary["passes"])

    def test_stage4_anchor_repass_requires_panel_symmetry_counts(self) -> None:
        confirmation_tolerant_entries = []
        for idx in range(5):
            confirmation_tolerant_entries.append(
                {
                    **self._make_stage4_probe_run_entry(
                        run_id=f"anchor-primary-{idx}",
                        train_seed=200 + idx,
                        heldout_seed=1200 + idx,
                    ),
                    "seed_panel": "primary",
                }
            )
        for idx in range(5):
            confirmation_tolerant_entries.append(
                {
                    **self._make_stage4_probe_run_entry(
                        run_id=f"anchor-confirmation-{idx}",
                        train_seed=300 + idx,
                        heldout_seed=1300 + idx,
                        learned_symmetry=0.0 if idx == 4 else 1.0,
                        oracle_symmetry=0.0 if idx == 4 else 1.0,
                    ),
                    "seed_panel": "confirmation",
                }
            )
        passing_anchor_report = runner_module._stage4_build_anchor_repass_report(
            run_entries=confirmation_tolerant_entries,
        )
        self.assertEqual(passing_anchor_report["primary_panel_summary"]["required_successes"], 5)
        self.assertEqual(passing_anchor_report["confirmation_panel_summary"]["required_successes"], 4)
        self.assertEqual(
            passing_anchor_report["confirmation_panel_summary"]["learned_symmetry_consistency_count"],
            4,
        )
        self.assertTrue(passing_anchor_report["passes"])

        primary_symmetry_fail_entries = []
        for idx in range(5):
            primary_symmetry_fail_entries.append(
                {
                    **self._make_stage4_probe_run_entry(
                        run_id=f"anchor-primary-fail-{idx}",
                        train_seed=400 + idx,
                        heldout_seed=1400 + idx,
                        learned_symmetry=0.0 if idx == 4 else 1.0,
                        oracle_symmetry=0.0 if idx == 4 else 1.0,
                    ),
                    "seed_panel": "primary",
                }
            )
        for idx in range(5):
            primary_symmetry_fail_entries.append(
                {
                    **self._make_stage4_probe_run_entry(
                        run_id=f"anchor-confirmation-fail-{idx}",
                        train_seed=500 + idx,
                        heldout_seed=1500 + idx,
                    ),
                    "seed_panel": "confirmation",
                }
            )
        failing_anchor_report = runner_module._stage4_build_anchor_repass_report(
            run_entries=primary_symmetry_fail_entries,
        )
        self.assertEqual(
            failing_anchor_report["primary_panel_summary"]["learned_symmetry_consistency_count"],
            4,
        )
        self.assertFalse(failing_anchor_report["passes"])

    def test_stage4_primary_gate_report_enforces_cross_panel_closure(self) -> None:
        panel_summaries = []
        for direction_id in ("splitA_to_splitB", "splitB_to_splitA"):
            primary_entries = [
                self._make_stage4_probe_run_entry(
                    run_id=f"{direction_id}-primary-{idx}",
                    train_seed=600 + idx,
                    heldout_seed=1600 + idx,
                )
                for idx in range(5)
            ]
            confirmation_entries = [
                self._make_stage4_probe_run_entry(
                    run_id=f"{direction_id}-confirmation-{idx}",
                    train_seed=700 + idx,
                    heldout_seed=1700 + idx,
                    learned_alignment_kl=0.8 if direction_id == "splitA_to_splitB" and idx == 4 else 0.1,
                )
                for idx in range(5)
            ]
            panel_summaries.append(
                runner_module._stage4_build_panel_summary(
                    direction_id=direction_id,
                    seed_panel="primary",
                    run_entries=primary_entries,
                )
            )
            panel_summaries.append(
                runner_module._stage4_build_panel_summary(
                    direction_id=direction_id,
                    seed_panel="confirmation",
                    run_entries=confirmation_entries,
                )
            )

        report = runner_module._stage4_build_primary_gate_report(
            panel_summaries=panel_summaries,
            run_confirmation_panel=True,
            confirmation_panel_executed=True,
        )
        self.assertTrue(report["primary_panels_pass"])
        self.assertTrue(report["confirmation_panels_pass"])
        self.assertIn("splitA_to_splitB", report["direction_closure_summaries"])
        self.assertIn("splitB_to_splitA", report["direction_closure_summaries"])
        self.assertFalse(
            report["direction_closure_summaries"]["splitA_to_splitB"]["checks"]["no_seed_learned_probe_alignment_kl_gt_0_75"]
        )
        self.assertFalse(report["direction_closure_summaries"]["splitA_to_splitB"]["passes"])
        self.assertTrue(report["direction_closure_summaries"]["splitB_to_splitA"]["passes"])
        self.assertFalse(report["checks"]["stage4d_direction_closure_passes_if_executed"])
        self.assertFalse(report["passes"])

    def test_stage4_package_runner_runs_confirmation_only_after_primary_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(artifacts=ArtifactConfig(root_dir=Path(tmp_dir)))
            passing_nonregression_report = {
                "metric_schema_version": "2026-04-01.stage4.metric-schema.v1",
                "metric_versions": {"stage4_nonregression": "v1"},
                "classification_rule_version": "2026-04-01.stage4.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "stage3_task_version": "2026-04-01.stage3.task-definition.v2",
                "stage4_package_version": "2026-04-01.stage4.package.v1",
                "checks": {"full_test_suite": True},
                "passes": True,
                "complete": True,
            }
            passing_anchor_report = {
                "metric_schema_version": "2026-04-01.stage4.metric-schema.v1",
                "metric_versions": {"stage4_anchor_repass_report": "v1"},
                "classification_rule_version": "2026-04-01.stage4.rules.v1",
                "source_metric_versions": ["benchmark_v1_frozen"],
                "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
                "stage3_task_version": "2026-04-01.stage3.task-definition.v2",
                "stage4_package_version": "2026-04-01.stage4.package.v1",
                "n_runs": 2,
                "seedwise": [],
                "summary": {},
                "checks": {"all_latent_contrasts_positive": True},
                "artifacts_complete": True,
                "passes": True,
                "skipped": False,
            }

            def fake_learned_probe_run(
                *,
                resolved_config: ExperimentConfig,
                store,
                generator,
                model,
                assay_runner,
                train_seed: int,
                heldout_seed: int,
                seed_panel: str,
                stage3_direction_id: str,
                run_index: int,
                eval_mode: str,
            ) -> dict[str, object]:
                del resolved_config, store, generator, model, assay_runner, run_index, eval_mode
                return {
                    "run_id": f"{stage3_direction_id}-{seed_panel}-{train_seed}-{heldout_seed}",
                    "run_dir": Path(tmp_dir) / "runs" / f"{stage3_direction_id}-{seed_panel}-{train_seed}-{heldout_seed}",
                    "seed_panel": seed_panel,
                    "train_seed": int(train_seed),
                    "heldout_seed": int(heldout_seed),
                    "probe_metrics": {
                        "probe_target_aligned_specificity_contrast": 0.02,
                        "probe_pooled_target_aligned_specificity_contrast": 0.003,
                    },
                    "oracle_probe_metrics": {
                        "probe_target_aligned_specificity_contrast": 0.04,
                        "probe_pooled_target_aligned_specificity_contrast": 0.01,
                    },
                    "probe_context_alignment_report": {
                        "learned_probe_alignment_kl": 0.1,
                        "learned_probe_expected_target_top1_rate": 0.8,
                        "learned_probe_pair_flip_symmetry_consistency": 1.0,
                        "oracle_probe_pair_flip_symmetry_consistency": 1.0,
                    },
                    "hidden_state_diagnostics": {
                        "probe_correct_pair_flip_rate__v2": 0.8,
                        "probe_within_pair_mass__v2": 0.5,
                        "probe_source_bin_kl__v2": 2.5,
                        "probe_collapse_index__v2": 0.0,
                    },
                    "continuation_retention_report": {
                        "delta_probe_target_aligned_specificity_contrast__pre_to_post_v2": 0.01,
                    },
                    "artifacts_complete": True,
                }

            with patch.object(runner_module, "_stage4_build_nonregression_report", return_value=passing_nonregression_report), \
                patch.object(runner_module, "_stage4_build_anchor_repass_report", return_value=passing_anchor_report), \
                patch.object(runner_module, "_stage4_execute_learned_probe_run", side_effect=fake_learned_probe_run):
                result = run_stage4_package(
                    config=config,
                    primary_seed_panel=((31, 1031), (32, 1032), (33, 1033), (34, 1034), (35, 1035)),
                    confirmation_seed_panel=((41, 1041), (42, 1042), (43, 1043), (44, 1044), (45, 1045)),
                    run_confirmation_panel=True,
                )

            self.assertTrue((result.output_root / "panels" / "splitA_to_splitB" / "confirmation" / "stage4_panel_summary.v1.json").exists())
            self.assertTrue((result.output_root / "panels" / "splitB_to_splitA" / "confirmation" / "stage4_panel_summary.v1.json").exists())
            verdict = json.loads((result.output_root / "stage4_verdict.v1.json").read_text(encoding="utf-8"))
            primary_gate_report = json.loads((result.output_root / "stage4_primary_gate_report.v1.json").read_text(encoding="utf-8"))
            primary_summary = json.loads(
                (result.output_root / "panels" / "splitA_to_splitB" / "primary" / "stage4_panel_summary.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(primary_summary["passes"])
            self.assertEqual(primary_summary["checks"]["required_successes"], 4)
            self.assertIn("learned_probe_expected_target_top1_rate", primary_summary["seedwise"][0])
            self.assertIn("mean_learned_probe_expected_target_top1_rate", primary_summary["summary"])
            self.assertIn("learned_symmetry_consistency_count", primary_summary["checks"])
            self.assertIn("learned_probe_expected_target_top1_rate_ge_0_75_count", primary_summary["checks"])
            self.assertIn("probe_correct_pair_flip_rate__v2_ge_0_75_count", primary_summary["checks"])
            self.assertIn("probe_within_pair_mass__v2_ge_0_40_count", primary_summary["checks"])
            self.assertIn("probe_source_bin_kl__v2_ge_2_00_count", primary_summary["checks"])
            self.assertIn("no_latent_pooled_sign_reversal", primary_summary["checks"])
            self.assertIn("direction_closure_summaries", primary_gate_report)
            self.assertIn("overall_closure_summary", primary_gate_report)
            self.assertTrue(primary_gate_report["direction_closure_passes"])
            self.assertTrue(primary_gate_report["overall_closure_passes"])
            self.assertTrue(
                primary_gate_report["direction_closure_summaries"]["splitA_to_splitB"]["checks"]["latent_positive_count_ge_9"]
            )
            self.assertTrue(
                primary_gate_report["direction_closure_summaries"]["splitA_to_splitB"]["checks"]["learned_symmetry_consistency_count_ge_9"]
            )
            self.assertTrue(
                primary_gate_report["overall_closure_summary"]["checks"]["latent_positive_count_ge_18"]
            )
            self.assertTrue(primary_gate_report["confirmation_panel_executed"])
            self.assertTrue(primary_gate_report["confirmation_panel_eligible"])
            self.assertTrue(primary_gate_report["passes"])
            self.assertTrue(verdict["direction_closure_passes"])
            self.assertTrue(verdict["overall_closure_passes"])
            self.assertTrue(verdict["confirmation_panel_executed"])


if __name__ == "__main__":
    unittest.main()
