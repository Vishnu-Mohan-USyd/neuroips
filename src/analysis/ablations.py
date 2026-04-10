"""Analysis 13: Post-training causal ablations.

On each trained model, apply ablations and re-run P1:
    - Zero SOM feedback -> remove inhibitory feedback
    - Zero PV -> remove normalization
    - Clamp pi_pred to fixed value -> remove precision modulation

Compare ablated vs intact suppression profiles.
Provides causal evidence for mechanism claims, beyond correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import ExperimentResult
from src.experiments.hidden_state import HiddenStateParadigm


@dataclass
class AblationResult:
    """Result of one ablation experiment."""
    ablation_name: str
    experiment_result: ExperimentResult
    description: str


@contextmanager
def _zero_module(module: nn.Module):
    """Temporarily zero all outputs of a module by zeroing weights+biases."""
    original_params = {}
    for name, param in module.named_parameters():
        original_params[name] = param.data.clone()
        param.data.zero_()
    try:
        yield
    finally:
        for name, param in module.named_parameters():
            param.data.copy_(original_params[name])


def run_ablation(
    net: LaminarV1V2Network,
    cfg: ModelConfig,
    ablation_name: str,
    n_trials: int = 50,
    seed: int = 42,
    probe_deviations: list[float] | None = None,
) -> AblationResult:
    """Run a single ablation on a trained model.

    Args:
        net: Trained network (will be temporarily modified).
        cfg: Model configuration.
        ablation_name: One of 'zero_som', 'zero_pv', 'clamp_pi'.
        n_trials: Trials per condition.
        seed: Random seed.
    """
    devs = probe_deviations or [0.0, 45.0, 90.0]
    paradigm = HiddenStateParadigm(net, cfg, probe_deviations=devs)

    if ablation_name == "zero_som":
        with _zero_module(net.som):
            result = paradigm.run(n_trials=n_trials, seed=seed)
        desc = "Zeroed SOM module — removes inhibitory feedback"

    elif ablation_name == "zero_pv":
        with _zero_module(net.pv):
            result = paradigm.run(n_trials=n_trials, seed=seed)
        desc = "Zeroed PV pool — removes divisive normalization"

    elif ablation_name == "clamp_pi":
        # Temporarily replace V2's pi_pred output with a constant
        orig_forward = net.v2.forward

        def patched_forward(*args, **kwargs):
            outputs = orig_forward(*args, **kwargs)
            # Emergent mode: (mu_pred, pi_pred, feedback_signal, h_v2)
            mu_pred, pi_pred, feedback_signal, h_v2 = outputs
            return mu_pred, torch.ones_like(pi_pred), feedback_signal, h_v2

        net.v2.forward = patched_forward
        try:
            result = paradigm.run(n_trials=n_trials, seed=seed)
        finally:
            net.v2.forward = orig_forward
        desc = "Clamped pi_pred=1 — removes precision modulation"

    else:
        # Unknown or inapplicable ablation — run intact
        result = paradigm.run(n_trials=n_trials, seed=seed)
        desc = f"No-op ablation ({ablation_name} not applicable)"

    return AblationResult(
        ablation_name=ablation_name,
        experiment_result=result,
        description=desc,
    )


def run_all_ablations(
    net: LaminarV1V2Network,
    cfg: ModelConfig,
    n_trials: int = 50,
    seed: int = 42,
    probe_deviations: list[float] | None = None,
) -> dict[str, AblationResult]:
    """Run all applicable ablations.

    Runs: intact, zero_som, zero_pv, clamp_pi.
    """
    ablation_names = ["zero_som", "zero_pv", "clamp_pi"]

    results: dict[str, AblationResult] = {}

    # Intact baseline
    devs = probe_deviations or [0.0, 45.0, 90.0]
    paradigm = HiddenStateParadigm(net, cfg, probe_deviations=devs)
    intact = paradigm.run(n_trials=n_trials, seed=seed)
    results["intact"] = AblationResult("intact", intact, "No ablation — intact model")

    for name in ablation_names:
        results[name] = run_ablation(
            net, cfg, name, n_trials=n_trials, seed=seed,
            probe_deviations=probe_deviations)

    return results
