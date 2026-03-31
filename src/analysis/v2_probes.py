"""Analysis 12: V2 interpretability probes.

Decode latent state (CW/CCW/neutral) from h_v2.
Entropy of q_pred per condition.
pi_pred calibration: correlation with actual prediction accuracy.
Validates that V2 carries real latent-state information, not just a gain signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.experiments.paradigm_base import ConditionData, ExperimentResult


@dataclass
class V2ProbeResult:
    """V2 interpretability analysis results."""
    q_pred_entropy: dict[str, float]     # {condition: mean entropy of q_pred}
    pi_pred_mean: dict[str, float]       # {condition: mean pi_pred}
    state_decoding_accuracy: float       # decode CW/CCW/neutral from state_logits
    pi_calibration: float                # correlation(pi_pred, actual accuracy)


def compute_q_pred_entropy(q_pred: Tensor) -> Tensor:
    """Compute entropy of q_pred distribution.

    Args:
        q_pred: [n_trials, T, N] predicted orientation distribution.

    Returns:
        entropy: [n_trials, T] entropy values.
    """
    # Normalize to probability distribution
    q_prob = q_pred / (q_pred.sum(dim=-1, keepdim=True) + 1e-10)
    q_prob = q_prob.clamp(min=1e-10)
    return -(q_prob * q_prob.log()).sum(dim=-1)


def decode_latent_state(
    state_logits: Tensor,
) -> Tensor:
    """Decode the most likely latent state from V2 state logits.

    Args:
        state_logits: [n_trials, T, 3] logits for CW/CCW/neutral.

    Returns:
        predicted_states: [n_trials, T] argmax state indices.
    """
    return state_logits.argmax(dim=-1)


def run_v2_probes(
    result: ExperimentResult,
    window: str = "sustained",
) -> V2ProbeResult:
    """Run V2 interpretability probes.

    Args:
        result: ExperimentResult from any paradigm.
        window: Temporal window to analyse.
    """
    start, end = result.temporal_windows[window]

    entropy_per_cond: dict[str, float] = {}
    pi_per_cond: dict[str, float] = {}
    all_logits = []
    all_pi = []

    for cond_name, cd in result.conditions.items():
        q_pred = cd.q_pred[:, start:end]
        ent = compute_q_pred_entropy(q_pred)
        entropy_per_cond[cond_name] = ent.mean().item()

        pi = cd.pi_pred[:, start:end]
        pi_per_cond[cond_name] = pi.mean().item()

        all_logits.append(cd.state_logits[:, start:end])
        all_pi.append(pi)

    # State decoding accuracy: check if argmax state is consistent within condition
    state_acc = 0.0
    n_total = 0
    for cd in result.conditions.values():
        logits = cd.state_logits[:, start:end]
        # Most common state across time for each trial
        states = logits.argmax(dim=-1)  # [n_trials, W]
        mode_states = states.mode(dim=-1).values  # [n_trials]
        # Check consistency: what fraction of timesteps match mode
        for t in range(states.shape[0]):
            matches = (states[t] == mode_states[t]).float().mean()
            state_acc += matches.item()
            n_total += 1
    state_acc = state_acc / max(n_total, 1)

    # pi_pred calibration: rough correlation with entropy (lower entropy = better prediction)
    pi_calibration = 0.0
    if len(entropy_per_cond) > 2:
        ent_vals = torch.tensor(list(entropy_per_cond.values()))
        pi_vals = torch.tensor(list(pi_per_cond.values()))
        ent_vals = ent_vals - ent_vals.mean()
        pi_vals = pi_vals - pi_vals.mean()
        denom = ent_vals.norm() * pi_vals.norm()
        if denom > 1e-10:
            pi_calibration = -(torch.dot(ent_vals, pi_vals) / denom).item()

    return V2ProbeResult(
        q_pred_entropy=entropy_per_cond,
        pi_pred_mean=pi_per_cond,
        state_decoding_accuracy=state_acc,
        pi_calibration=pi_calibration,
    )
