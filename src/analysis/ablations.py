"""Analysis 13: Post-training causal ablations.

On each trained model, apply ablations and re-run P1:
    - Zero SOM feedback -> remove mechanism-specific inhibition
    - Zero PV -> remove normalization
    - Zero deep template pathway -> remove prediction signal
    - Clamp pi_pred to fixed value -> remove precision modulation
    - (Model C only) Zero center / zero surround separately

Compare ablated vs intact suppression profiles.
Provides causal evidence for mechanism claims, beyond correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor

from src.config import ModelConfig, MechanismType
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


@contextmanager
def _clamp_output(module: nn.Module, attr: str, value: float):
    """Temporarily monkey-patch a module attribute to return a fixed value."""
    original_fn = getattr(module, attr)

    def clamped_fn(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        return torch.full_like(result, value)

    setattr(module, attr, clamped_fn)
    try:
        yield
    finally:
        setattr(module, attr, original_fn)


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
        ablation_name: One of 'zero_som', 'zero_pv', 'zero_template',
                       'clamp_pi', 'zero_center', 'zero_surround'.
        n_trials: Trials per condition.
        seed: Random seed.
    """
    devs = probe_deviations or [0.0, 45.0, 90.0]
    paradigm = HiddenStateParadigm(net, cfg, probe_deviations=devs)

    if ablation_name == "zero_som":
        with _zero_module(net.som):
            result = paradigm.run(n_trials=n_trials, seed=seed)
        desc = "Zeroed SOM module — removes mechanism-specific inhibition"

    elif ablation_name == "zero_pv":
        with _zero_module(net.pv):
            result = paradigm.run(n_trials=n_trials, seed=seed)
        desc = "Zeroed PV pool — removes divisive normalization"

    elif ablation_name == "zero_template":
        with _zero_module(net.deep_template):
            result = paradigm.run(n_trials=n_trials, seed=seed)
        desc = "Zeroed deep template — removes prediction signal"

    elif ablation_name == "clamp_pi":
        # Temporarily replace V2's pi_pred output with a constant
        orig_forward = net.v2.forward

        def patched_forward(*args, **kwargs):
            outputs = orig_forward(*args, **kwargs)
            if len(outputs) == 3:
                # Emergent mode: (p_cw, pi_pred, h_v2)
                p_cw, pi_pred, h_v2 = outputs
                return p_cw, torch.ones_like(pi_pred), h_v2
            else:
                # Fixed mode: (q_pred, pi_pred, state_logits, h_v2)
                q_pred, pi_pred, state_logits, h_v2 = outputs
                return q_pred, torch.ones_like(pi_pred), state_logits, h_v2

        net.v2.forward = patched_forward
        try:
            result = paradigm.run(n_trials=n_trials, seed=seed)
        finally:
            net.v2.forward = orig_forward
        desc = "Clamped pi_pred=1 — removes precision modulation"

    elif ablation_name == "zero_center" and cfg.mechanism == MechanismType.CENTER_SURROUND:
        orig_fn = net.feedback.compute_center_excitation

        def zero_center(*args, **kwargs):
            result = orig_fn(*args, **kwargs)
            return torch.zeros_like(result)

        net.feedback.compute_center_excitation = zero_center
        try:
            result = paradigm.run(n_trials=n_trials, seed=seed)
        finally:
            net.feedback.compute_center_excitation = orig_fn
        desc = "Zeroed center excitation — Model C only"

    elif ablation_name == "zero_surround" and cfg.mechanism == MechanismType.CENTER_SURROUND:
        orig_fn = net.feedback.compute_som_drive

        def zero_surround(*args, **kwargs):
            result = orig_fn(*args, **kwargs)
            return torch.zeros_like(result)

        net.feedback.compute_som_drive = zero_surround
        try:
            result = paradigm.run(n_trials=n_trials, seed=seed)
        finally:
            net.feedback.compute_som_drive = orig_fn
        desc = "Zeroed surround SOM drive — Model C only"

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

    Always runs: intact, zero_som, zero_pv, zero_template, clamp_pi.
    For center-surround: also zero_center, zero_surround.
    """
    ablation_names = ["zero_som", "zero_pv", "zero_template", "clamp_pi"]
    if cfg.mechanism == MechanismType.CENTER_SURROUND:
        ablation_names.extend(["zero_center", "zero_surround"])

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
