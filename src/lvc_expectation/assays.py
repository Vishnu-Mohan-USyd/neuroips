"""Primary assay metric runner."""

from __future__ import annotations

from .config import ExperimentConfig
import torch
import torch.nn.functional as F

from .geometry import OrientationGeometry
from .metrics import (
    comparator_nonuniformity,
    mean_response_delta,
    masked_template_specificity,
    omission_specificity,
    prestimulus_condition_report,
    simple_decoder_accuracy,
    template_specificity,
)
from .paradigms import PRESTIM_CONTEXT_ONLY, PRESTIM_CUE_ONLY
from .types import SimulationOutput, TrialBatch


class AssayRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.geometry = OrientationGeometry(config.geometry.n_orientations)
        self.primary_metrics = (
            "mean_suppression",
            "poststimulus_l23_template_specificity",
            "poststimulus_pooled_template_specificity",
            "poststimulus_l23_template_specificity_early",
            "poststimulus_l23_template_specificity_middle",
            "poststimulus_l23_template_specificity_late",
            "context_comparator_nonuniformity",
            "context_comparator_nonuniformity_early",
            "context_comparator_nonuniformity_middle",
            "context_comparator_nonuniformity_late",
            "tuning_slope",
            "tuning_peak_change",
            "tuning_width_change",
            "decoder_accuracy",
            "rsa_distance",
            "prestimulus_template_specificity",
            "omission_template_specificity",
            "energy_telemetry",
        )

    @staticmethod
    def _prestim_valid_mask(batch: TrialBatch) -> torch.Tensor:
        return batch.blank_mask & batch.metadata["omission_targets"].lt(0)

    @staticmethod
    def _prestim_targets(batch: TrialBatch) -> torch.Tensor:
        return batch.metadata["expected_distribution"].argmax(dim=-1)

    @staticmethod
    def _visible_poststim_mask(batch: TrialBatch) -> torch.Tensor:
        return batch.orientations.ge(0)

    def _window_mask(self, base_mask: torch.Tensor, window_name: str) -> torch.Tensor:
        start, end = getattr(self.config.windows, window_name)
        time_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        time_mask[:, start - 1 : end] = True
        return base_mask & time_mask

    def _prestim_condition_masks(self, batch: TrialBatch) -> dict[str, torch.Tensor]:
        prestim_valid = self._prestim_valid_mask(batch)
        return {
            "cue_only": prestim_valid & batch.prestim_mode.eq(PRESTIM_CUE_ONLY).unsqueeze(1),
            "context_only": prestim_valid & batch.prestim_mode.eq(PRESTIM_CONTEXT_ONLY).unsqueeze(1),
            "neutral": prestim_valid & batch.context_ids.eq(2).unsqueeze(1),
        }

    def compute_prestim_template_gate(
        self,
        intact_simulation: SimulationOutput,
        zero_context_simulation: SimulationOutput,
        batch: TrialBatch,
    ) -> dict[str, object]:
        targets = self._prestim_targets(batch)
        condition_masks = self._prestim_condition_masks(batch)

        def build_report(simulation: SimulationOutput) -> dict[str, dict[str, float | int]]:
            deep_template = simulation.states["deep_template"]
            return {
                condition: prestimulus_condition_report(deep_template, targets, mask)
                for condition, mask in condition_masks.items()
            }

        return {
            "control_mode": "zero_context",
            "intact": build_report(intact_simulation),
            "zero_context": build_report(zero_context_simulation),
        }

    @staticmethod
    def _probe_step_mask(batch: TrialBatch) -> torch.Tensor:
        return batch.metadata["probe_step_mask"] & batch.metadata["probe_valid_mask"]

    @staticmethod
    def _pair_balanced_probe_summary(
        values: torch.Tensor,
        pair_ids: torch.Tensor,
        expected_mask: torch.Tensor,
        unexpected_mask: torch.Tensor,
    ) -> dict[str, object]:
        scored_pair_ids: list[int] = []
        expected_means: list[torch.Tensor] = []
        unexpected_means: list[torch.Tensor] = []
        contrasts: list[torch.Tensor] = []
        for pair_id in torch.unique(pair_ids).tolist():
            current_pair = pair_ids.eq(int(pair_id))
            pair_expected = values[current_pair & expected_mask]
            pair_unexpected = values[current_pair & unexpected_mask]
            if pair_expected.numel() == 0 or pair_unexpected.numel() == 0:
                continue
            expected_mean = pair_expected.mean()
            unexpected_mean = pair_unexpected.mean()
            scored_pair_ids.append(int(pair_id))
            expected_means.append(expected_mean)
            unexpected_means.append(unexpected_mean)
            contrasts.append(expected_mean - unexpected_mean)

        if not contrasts:
            return {
                "expected_mean": 0.0,
                "unexpected_mean": 0.0,
                "contrast": 0.0,
                "n_pairs_scored": 0,
                "scored_pair_ids": [],
            }

        return {
            "expected_mean": float(torch.stack(expected_means).mean().item()),
            "unexpected_mean": float(torch.stack(unexpected_means).mean().item()),
            "contrast": float(torch.stack(contrasts).mean().item()),
            "n_pairs_scored": len(scored_pair_ids),
            "scored_pair_ids": scored_pair_ids,
        }

    def compute_local_global_probe_metrics(self, simulation: SimulationOutput, batch: TrialBatch) -> dict[str, object]:
        probe_mask = self._probe_step_mask(batch)
        responses = simulation.states["l23_readout"]
        pooled = simulation.observations.get("gaussian_orientation_bank")
        if pooled is None:
            pooled = simulation.observations.get("identity", responses)
        comparator = simulation.states.get("context_comparator", torch.zeros_like(responses))

        target_bins = batch.metadata["probe_target_orientation"].unsqueeze(1).expand_as(batch.orientations)
        pair_ids = batch.metadata["probe_pair_id"].unsqueeze(1).expand_as(batch.orientations)
        expected_rows = batch.metadata["probe_global_expected_mask"][probe_mask]
        unexpected_rows = batch.metadata["probe_global_unexpected_mask"][probe_mask]
        probe_pair_ids = pair_ids[probe_mask]

        response_specificity = template_specificity(responses[probe_mask], target_bins[probe_mask])
        pooled_specificity = template_specificity(pooled[probe_mask], target_bins[probe_mask])
        comparator_rows = comparator[probe_mask]
        comparator_row_nonuniformity = (
            comparator_rows - comparator_rows.mean(dim=-1, keepdim=True)
        ).abs().mean(dim=-1)

        response_summary = self._pair_balanced_probe_summary(
            response_specificity,
            probe_pair_ids,
            expected_rows,
            unexpected_rows,
        )
        pooled_summary = self._pair_balanced_probe_summary(
            pooled_specificity,
            probe_pair_ids,
            expected_rows,
            unexpected_rows,
        )
        comparator_summary = self._pair_balanced_probe_summary(
            comparator_row_nonuniformity,
            probe_pair_ids,
            expected_rows,
            unexpected_rows,
        )

        if expected_rows.any() and unexpected_rows.any():
            expected_mean = pooled[batch.metadata["probe_global_expected_mask"]].mean(dim=0)
            unexpected_mean = pooled[batch.metadata["probe_global_unexpected_mask"]].mean(dim=0)
            rsa_distance = float(
                1.0 - F.cosine_similarity(expected_mean.unsqueeze(0), unexpected_mean.unsqueeze(0)).item()
            )
        else:
            rsa_distance = 0.0

        return {
            "n_probe_rows": int(probe_mask.sum().item()),
            "n_probe_global_expected_rows": int(batch.metadata["probe_global_expected_mask"][probe_mask].sum().item()),
            "n_probe_global_unexpected_rows": int(batch.metadata["probe_global_unexpected_mask"][probe_mask].sum().item()),
            "n_probe_pairs_total": int(torch.unique(batch.metadata["probe_pair_id"]).numel()),
            "n_probe_pairs_scored": int(response_summary["n_pairs_scored"]),
            "probe_pair_ids_scored": response_summary["scored_pair_ids"],
            "probe_target_aligned_specificity_expected": response_summary["expected_mean"],
            "probe_target_aligned_specificity_unexpected": response_summary["unexpected_mean"],
            "probe_target_aligned_specificity_contrast": response_summary["contrast"],
            "probe_pooled_target_aligned_specificity_expected": pooled_summary["expected_mean"],
            "probe_pooled_target_aligned_specificity_unexpected": pooled_summary["unexpected_mean"],
            "probe_pooled_target_aligned_specificity_contrast": pooled_summary["contrast"],
            "probe_context_comparator_nonuniformity_expected": comparator_summary["expected_mean"],
            "probe_context_comparator_nonuniformity_unexpected": comparator_summary["unexpected_mean"],
            "probe_context_comparator_nonuniformity_contrast": comparator_summary["contrast"],
            "rsa_distance": rsa_distance,
        }

    def compute_primary_metrics(self, simulation: SimulationOutput, batch: TrialBatch) -> dict[str, float]:
        responses = simulation.states["l23_readout"]
        comparator = simulation.states.get("context_comparator", torch.zeros_like(responses))
        condition_codes = batch.metadata["condition_codes"]
        expected_mask = condition_codes == 0
        unexpected_mask = condition_codes == 1
        expected = responses[expected_mask]
        unexpected = responses[unexpected_mask]
        mean_suppression = float(mean_response_delta(expected, unexpected).item()) if expected.numel() and unexpected.numel() else 0.0

        predicted_bins = simulation.context_predictions.argmax(dim=-1)
        visible_targets = batch.orientations.clamp_min(0)
        distances = self.geometry.circular_distance_bins(predicted_bins, visible_targets).to(torch.float32)
        response_peak = responses.max(dim=-1).values
        flat_distances = distances.reshape(-1)
        flat_peaks = response_peak.reshape(-1)
        design = torch.stack((torch.ones_like(flat_distances), flat_distances), dim=-1)
        slope = torch.linalg.lstsq(design, flat_peaks.unsqueeze(-1)).solution.squeeze(-1)[1].item()

        tuning_peak_change = float(expected.max().item() - unexpected.max().item()) if expected.numel() and unexpected.numel() else 0.0
        expected_width = (response_peak[expected_mask] * distances[expected_mask]).mean() if expected_mask.any() else responses.new_tensor(0.0)
        unexpected_width = (response_peak[unexpected_mask] * distances[unexpected_mask]).mean() if unexpected_mask.any() else responses.new_tensor(0.0)
        tuning_width_change = float((expected_width - unexpected_width).item())

        if simulation.observations:
            decoder_logits = simulation.observations["identity"]
        else:
            decoder_logits = responses
        decoder_acc = float(simple_decoder_accuracy(decoder_logits.reshape(-1, decoder_logits.shape[-1]), visible_targets.reshape(-1)).item())

        pooled = simulation.observations.get("gaussian_orientation_bank")
        if pooled is None:
            pooled = simulation.observations.get("identity", responses)
        expected_mean = pooled[expected_mask].mean(dim=0) if expected_mask.any() else pooled.new_zeros(pooled.shape[-1])
        unexpected_mean = pooled[unexpected_mask].mean(dim=0) if unexpected_mask.any() else pooled.new_zeros(pooled.shape[-1])
        # Keep rsa_distance as a pooled condition-mean separation metric.
        condition_mean_rsa_distance = float(
            1.0 - F.cosine_similarity(expected_mean.unsqueeze(0), unexpected_mean.unsqueeze(0)).item()
        )

        visible_poststim_mask = self._visible_poststim_mask(batch)
        expected_targets = self._prestim_targets(batch)
        poststimulus_l23_template_specificity = float(
            masked_template_specificity(responses, expected_targets, visible_poststim_mask).item()
        )
        poststimulus_pooled_template_specificity = float(
            masked_template_specificity(pooled, expected_targets, visible_poststim_mask).item()
        )
        context_comparator_nonuniformity = float(
            comparator_nonuniformity(comparator, visible_poststim_mask).item()
        )
        windowed_metrics: dict[str, float] = {}
        for window_name in ("early", "middle", "late"):
            window_mask = self._window_mask(visible_poststim_mask, window_name)
            windowed_metrics[f"poststimulus_l23_template_specificity_{window_name}"] = float(
                masked_template_specificity(responses, expected_targets, window_mask).item()
            )
            windowed_metrics[f"context_comparator_nonuniformity_{window_name}"] = float(
                comparator_nonuniformity(comparator, window_mask).item()
            )

        prestimulus_template_specificity = prestimulus_condition_report(
            simulation.states["deep_template"],
            expected_targets,
            self._prestim_valid_mask(batch),
        )["prestimulus_template_specificity"]

        omission_template_specificity = float(
            omission_specificity(simulation.states["deep_template"], batch.metadata["omission_targets"]).item()
        )
        energy_telemetry = float(responses.abs().mean().item())

        return {
            "mean_suppression": mean_suppression,
            "poststimulus_l23_template_specificity": poststimulus_l23_template_specificity,
            "poststimulus_pooled_template_specificity": poststimulus_pooled_template_specificity,
            "context_comparator_nonuniformity": context_comparator_nonuniformity,
            "tuning_slope": float(slope),
            "tuning_peak_change": tuning_peak_change,
            "tuning_width_change": tuning_width_change,
            "decoder_accuracy": decoder_acc,
            "rsa_distance": condition_mean_rsa_distance,
            "prestimulus_template_specificity": prestimulus_template_specificity,
            "omission_template_specificity": omission_template_specificity,
            "energy_telemetry": energy_telemetry,
            **windowed_metrics,
        }
