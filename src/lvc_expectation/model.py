"""Shared scaffold and angular operator family for phase-1 expectation modeling."""

from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig
from .geometry import OrientationGeometry
from .stimuli import orientation_to_one_hot
from .types import ContextPrediction, SimulationOutput, TrialBatch


class AngularKernelFamily(nn.Module):
    def __init__(self, geometry: OrientationGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        narrow = [geometry.gaussian_kernel(index, width_deg=geometry.step_deg * 1.5) for index in range(geometry.n_orientations)]
        broad = [geometry.gaussian_kernel(index, width_deg=geometry.step_deg * 3.0) for index in range(geometry.n_orientations)]
        self.register_buffer("narrow_bank", torch.stack(narrow, dim=0))
        self.register_buffer("broad_bank", torch.stack(broad, dim=0))

    def forward(self, template: torch.Tensor, precision: torch.Tensor, subcase: str) -> torch.Tensor:
        precision = precision.expand_as(template)
        if subcase == "adaptation_only":
            return torch.zeros_like(template)
        if subcase == "context_global_gain":
            scalar = template.mean(dim=-1, keepdim=True) * precision.mean(dim=-1, keepdim=True)
            return scalar.expand_as(template)
        if subcase == "dampening":
            return precision * template
        if subcase == "sharpening":
            return precision * (1.0 - template)
        if subcase == "center_surround":
            center = torch.matmul(template, self.narrow_bank)
            surround = torch.matmul(template, self.broad_bank)
            return precision * (surround - 0.5 * center)
        raise ValueError(f"unknown operator subcase: {subcase}")


class V1ExpectationModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.geometry = OrientationGeometry(config.geometry.n_orientations)
        self.angular_family = AngularKernelFamily(self.geometry)

    def rollout(self, batch: TrialBatch, context: ContextPrediction, subcase: str) -> SimulationOutput:
        sensory = orientation_to_one_hot(batch.orientations, self.geometry.n_orientations)
        bsz, steps, n = sensory.shape
        device = sensory.device
        l4 = torch.zeros((bsz, n), dtype=torch.float32, device=device)
        l23 = torch.zeros_like(l4)
        template = torch.zeros_like(l4)
        adaptation = torch.zeros_like(l4)
        l4_states = []
        l23_states = []
        template_states = []
        adaptation_states = []
        comparator_states = []
        for step in range(steps):
            current_sensory = sensory[:, step]
            adaptation = self.config.scaffold.adapt_tau * adaptation + self.config.scaffold.adapt_gain * l4
            l4_drive = torch.relu(current_sensory - adaptation)
            l4 = l4_drive / (1.0 + self.config.scaffold.pv_norm_strength * l4_drive.mean(dim=-1, keepdim=True))
            template_target = torch.softmax(context.orientation_logits[:, step], dim=-1)
            precision = torch.sigmoid(context.precision_logit[:, step]) if context.precision_logit is not None else torch.ones((bsz, 1), device=device)
            template = 0.8 * template + self.config.scaffold.template_gain * precision * template_target
            comparator = self.angular_family(template, precision, subcase=subcase)
            l23 = torch.relu(self.config.scaffold.ff_gain * l4 + self.config.scaffold.recurrent_gain * l23 + template - comparator)
            if self.config.scaffold.process_noise_std > 0.0 and self.training:
                l23 = l23 + torch.randn_like(l23) * self.config.scaffold.process_noise_std
            l4_states.append(l4)
            l23_states.append(l23)
            template_states.append(template)
            adaptation_states.append(adaptation)
            comparator_states.append(comparator)
        states = {
            "l4_sensory": torch.stack(l4_states, dim=1),
            "l23_readout": torch.stack(l23_states, dim=1),
            "deep_template": torch.stack(template_states, dim=1),
            "adaptation_state": torch.stack(adaptation_states, dim=1),
            "context_comparator": torch.stack(comparator_states, dim=1),
        }
        return SimulationOutput(
            states=states,
            observations={},
            context_prediction=context,
            metadata=batch.metadata,
        )
