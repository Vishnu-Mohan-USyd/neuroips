"""Causally masked context prediction for phase-1 expectation modeling."""

from __future__ import annotations

import torch
from torch import nn

from .config import ContextConfig, GeometryConfig
from .types import ContextPrediction, TrialBatch


class CausalContextPredictor(nn.Module):
    def __init__(self, geometry: GeometryConfig, context_config: ContextConfig) -> None:
        super().__init__()
        self.n_orientations = geometry.n_orientations
        self.context_dim = context_config.context_dim
        self.hidden_dim = context_config.hidden_dim
        self.use_precision = context_config.use_precision
        self.task_dim = 2
        input_dim = self.n_orientations + self.context_dim + self.task_dim + 1
        self.gru = nn.GRU(input_size=input_dim, hidden_size=context_config.hidden_dim, batch_first=True)
        self.orientation_head = nn.Linear(context_config.hidden_dim, self.n_orientations)
        self.precision_head = nn.Linear(context_config.hidden_dim, 1) if self.use_precision else None
        self.latent_state_head = nn.Linear(context_config.hidden_dim, 3)
        self.context_residual_orientation_weight = nn.Parameter(
            torch.zeros(self.context_dim, self.n_orientations, context_config.hidden_dim)
        )
        self.context_residual_orientation_bias = nn.Parameter(torch.zeros(self.context_dim, self.n_orientations))
        nn.init.zeros_(self.latent_state_head.weight)
        nn.init.zeros_(self.latent_state_head.bias)
        self.latent_state_head.requires_grad_(False)
        self.context_residual_orientation_weight.requires_grad_(False)
        self.context_residual_orientation_bias.requires_grad_(False)

    def _build_inputs(self, batch: TrialBatch) -> torch.Tensor:
        orientations = batch.orientations
        batch_size, n_steps = orientations.shape
        safe = orientations.clamp_min(0)
        prev_one_hot = torch.nn.functional.one_hot(safe, num_classes=self.n_orientations).to(torch.float32)
        prev_one_hot = torch.roll(prev_one_hot, shifts=1, dims=1)
        prev_one_hot[:, 0] = 0.0
        prev_one_hot = torch.where(batch.blank_mask.unsqueeze(-1), torch.zeros_like(prev_one_hot), prev_one_hot)
        context_one_hot = torch.nn.functional.one_hot(batch.context_ids, num_classes=self.context_dim).to(torch.float32)
        context_one_hot = context_one_hot.unsqueeze(1).expand(batch_size, n_steps, self.context_dim)
        task_one_hot = torch.nn.functional.one_hot(batch.task_mode, num_classes=self.task_dim).to(torch.float32)
        task_one_hot = task_one_hot.unsqueeze(1).expand(batch_size, n_steps, self.task_dim)
        previous_orthogonal = torch.roll(batch.orthogonal_events.to(torch.float32), shifts=1, dims=1)
        previous_orthogonal[:, 0] = 0.0
        previous_orthogonal = previous_orthogonal.unsqueeze(-1)
        return torch.cat([prev_one_hot, context_one_hot, task_one_hot, previous_orthogonal], dim=-1)

    def _expand_context_ids(self, context_ids: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Expand per-sequence context ids to the [batch, step] grid used by hidden-state readout."""

        if context_ids.ndim == 1:
            return context_ids.unsqueeze(1).expand(-1, n_steps)
        if context_ids.ndim == 2 and int(context_ids.shape[1]) == int(n_steps):
            return context_ids
        raise ValueError("context_ids must have shape [batch] or [batch, step].")

    def orientation_logits_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Project frozen GRU hidden states through the native head plus optional context residual.

        The residual parameters stay zero-initialized and frozen by default, so existing packages
        remain behaviorally unchanged unless a tranche explicitly enables this scope.
        """

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, step, hidden_dim].")
        batch_size, n_steps, hidden_dim = hidden_states.shape
        if int(hidden_dim) != int(self.hidden_dim):
            raise ValueError("hidden_states last dimension must match predictor hidden_dim.")
        native_logits = self.orientation_head(hidden_states)
        expanded_context_ids = self._expand_context_ids(context_ids, n_steps).to(
            device=hidden_states.device,
            dtype=torch.long,
        )
        if expanded_context_ids.shape != (batch_size, n_steps):
            raise ValueError("expanded context ids must match the leading [batch, step] hidden-state dimensions.")
        residual_weight = self.context_residual_orientation_weight[expanded_context_ids]
        residual_bias = self.context_residual_orientation_bias[expanded_context_ids]
        residual_logits = torch.einsum(
            "bsoh,bsh->bso",
            residual_weight.to(dtype=hidden_states.dtype),
            hidden_states,
        ) + residual_bias.to(dtype=hidden_states.dtype)
        return native_logits + residual_logits

    def latent_state_logits_from_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project frozen GRU hidden states into the 3-state latent family.

        This head is zero-initialized and frozen by default so legacy packages remain unchanged.
        Tranches that explicitly authorize this scope can enable only these parameters.
        """

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, step, hidden_dim].")
        if int(hidden_states.shape[-1]) != int(self.hidden_dim):
            raise ValueError("hidden_states last dimension must match predictor hidden_dim.")
        return self.latent_state_head(hidden_states)

    def forward(self, batch: TrialBatch) -> ContextPrediction:
        inputs = self._build_inputs(batch)
        hidden, _ = self.gru(inputs)
        orientation_logits = self.orientation_logits_from_hidden_states(hidden, batch.context_ids)
        precision = self.precision_head(hidden) if self.precision_head is not None else None
        return ContextPrediction(orientation_logits=orientation_logits, precision_logit=precision)
