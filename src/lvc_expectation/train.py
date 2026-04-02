"""Predictive-only training utilities for the context pathway."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .context import CausalContextPredictor
from .config import TrainingConfig
from .types import ContextPrediction, TrialBatch

SAMPLED_LABEL_OBJECTIVE = "sampled_labels"
EXPECTED_DISTRIBUTION_OBJECTIVE = "expected_distribution"


@dataclass
class LossBundle:
    total: torch.Tensor
    predictive_loss: torch.Tensor
    energy_penalty: torch.Tensor
    homeostasis_penalty: torch.Tensor


@dataclass
class EpochMetrics:
    epoch: int
    total: float
    predictive_loss: float
    energy_penalty: float
    homeostasis_penalty: float


def _visible_nonblank_mask(batch: TrialBatch) -> torch.Tensor:
    visible_mask = batch.orientations.ge(0)
    if batch.blank_mask is None:
        return visible_mask
    return visible_mask & batch.blank_mask.logical_not()


def _coerce_loss_mask(
    prediction: ContextPrediction,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if loss_mask is None:
        return None
    expected_shape = prediction.orientation_logits.shape[:2]
    if tuple(loss_mask.shape) != tuple(expected_shape):
        raise ValueError("loss_mask must match the leading [batch, step] shape of orientation_logits.")
    return loss_mask.to(device=prediction.orientation_logits.device, dtype=torch.bool)


def _sampled_label_predictive_loss(
    prediction: ContextPrediction,
    batch: TrialBatch,
    *,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = batch.orientations
    logits = prediction.orientation_logits
    valid_mask = targets.ge(0)
    if loss_mask is not None:
        valid_mask = valid_mask & loss_mask
    if not valid_mask.any():
        raise ValueError("sampled_labels objective requires at least one supervised step after masking.")
    ignore_index = -100
    safe_targets = torch.where(valid_mask, targets, torch.full_like(targets, ignore_index))
    return F.cross_entropy(logits.transpose(1, 2), safe_targets, ignore_index=ignore_index)


def _expected_distribution_predictive_loss(
    prediction: ContextPrediction,
    batch: TrialBatch,
    *,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = prediction.orientation_logits
    expected_distribution = batch.metadata.get("expected_distribution")
    if expected_distribution is None:
        raise ValueError("expected_distribution objective requires batch.metadata['expected_distribution'].")
    expected_distribution = expected_distribution.to(device=logits.device, dtype=logits.dtype)
    if expected_distribution.shape != logits.shape:
        raise ValueError("expected_distribution must have the same shape as orientation_logits.")
    valid_mask = _visible_nonblank_mask(batch).to(device=logits.device)
    if loss_mask is not None:
        valid_mask = valid_mask & loss_mask
    if valid_mask.shape != logits.shape[:2]:
        raise ValueError("visible-step mask must match the leading orientation_logits dimensions.")
    if not valid_mask.any():
        raise ValueError("expected_distribution objective requires at least one visible, nonblank step.")
    log_probs = torch.log_softmax(logits, dim=-1)
    per_step_loss = -(expected_distribution * log_probs).sum(dim=-1)
    return per_step_loss[valid_mask].mean()


def predictive_loss(
    prediction: ContextPrediction,
    batch: TrialBatch,
    *,
    objective_mode: str = SAMPLED_LABEL_OBJECTIVE,
    loss_mask: torch.Tensor | None = None,
) -> LossBundle:
    resolved_loss_mask = _coerce_loss_mask(prediction, loss_mask)
    if objective_mode == SAMPLED_LABEL_OBJECTIVE:
        predictive = _sampled_label_predictive_loss(prediction, batch, loss_mask=resolved_loss_mask)
    elif objective_mode == EXPECTED_DISTRIBUTION_OBJECTIVE:
        predictive = _expected_distribution_predictive_loss(prediction, batch, loss_mask=resolved_loss_mask)
    else:
        raise ValueError(f"unknown predictive objective mode: {objective_mode}")
    zero = prediction.orientation_logits.new_tensor(0.0)
    return LossBundle(total=predictive, predictive_loss=predictive, energy_penalty=zero, homeostasis_penalty=zero)


def _to_epoch_metrics(epoch: int, losses: LossBundle) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        total=float(losses.total.item()),
        predictive_loss=float(losses.predictive_loss.item()),
        energy_penalty=float(losses.energy_penalty.item()),
        homeostasis_penalty=float(losses.homeostasis_penalty.item()),
    )


def train_context_predictor(
    predictor: CausalContextPredictor,
    train_batches: list[TrialBatch],
    training_config: TrainingConfig,
    *,
    objective_mode: str = SAMPLED_LABEL_OBJECTIVE,
    loss_masks: list[torch.Tensor | None] | None = None,
) -> list[EpochMetrics]:
    """Train the context predictor with Adam on the predictive-only objective.

    Training and held-out evaluation stay separate by construction: this function consumes
    only the provided train batches and returns per-epoch training losses.
    """

    if len(train_batches) < training_config.n_epochs:
        raise ValueError("train_batches must provide at least one batch per configured epoch.")
    if loss_masks is not None and len(loss_masks) < training_config.n_epochs:
        raise ValueError("loss_masks must provide at least one mask per configured epoch when supplied.")

    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[EpochMetrics] = []
    predictor.train()
    for epoch in range(training_config.n_epochs):
        batch = train_batches[epoch]
        loss_mask = None if loss_masks is None else loss_masks[epoch]
        optimizer.zero_grad(set_to_none=True)
        prediction = predictor(batch)
        losses = predictive_loss(
            prediction,
            batch,
            objective_mode=objective_mode,
            loss_mask=loss_mask,
        )
        losses.total.backward()
        optimizer.step()
        history.append(_to_epoch_metrics(epoch + 1, losses))
    return history


@torch.no_grad()
def evaluate_context_predictor(
    predictor: CausalContextPredictor,
    batch: TrialBatch,
    *,
    objective_mode: str = SAMPLED_LABEL_OBJECTIVE,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate predictive-only loss on a held-out batch."""

    predictor.eval()
    prediction = predictor(batch)
    losses = predictive_loss(
        prediction,
        batch,
        objective_mode=objective_mode,
        loss_mask=loss_mask,
    )
    return {
        "total": float(losses.total.item()),
        "predictive_loss": float(losses.predictive_loss.item()),
        "energy_penalty": float(losses.energy_penalty.item()),
        "homeostasis_penalty": float(losses.homeostasis_penalty.item()),
    }
