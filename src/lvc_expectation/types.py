"""Shared typed containers used across the phase-1 codebase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ContextPrediction:
    orientation_logits: torch.Tensor
    precision_logit: torch.Tensor | None = None


@dataclass
class SimulationOutput:
    states: dict[str, torch.Tensor]
    observations: dict[str, torch.Tensor]
    context_prediction: ContextPrediction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_predictions(self) -> torch.Tensor | None:
        if self.context_prediction is None:
            return None
        return self.context_prediction.orientation_logits

    @property
    def precision(self) -> torch.Tensor | None:
        if self.context_prediction is None:
            return None
        return self.context_prediction.precision_logit


@dataclass
class WindowSummary:
    summaries: dict[str, dict[str, torch.Tensor]]


@dataclass
class TrialBatch:
    orientations: torch.Tensor
    blank_mask: torch.Tensor | None
    expected_mask: torch.Tensor
    context_ids: torch.Tensor
    task_mode: torch.Tensor
    prestim_mode: torch.Tensor
    orthogonal_events: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prestimulus_mode(self) -> torch.Tensor:
        return self.prestim_mode
