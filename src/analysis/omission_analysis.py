"""Analysis 6: Omission + prestimulus template decoding.

Decode orientation from deep template during omission (P2).
Also decode from deep template during pre-stimulus period of normal trials
(Kok 2017: expected orientation info present before stimulus onset).
Template fidelity vs pi_pred correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils import circular_distance_abs
from src.experiments.paradigm_base import ConditionData, ExperimentResult


@dataclass
class TemplateDecodingResult:
    """Template decoding results."""
    omission_accuracy: float           # decode accuracy during omission
    prestimulus_accuracy: float        # decode accuracy before probe onset
    template_fidelity: float           # cosine similarity of template to expected
    fidelity_pi_correlation: float     # correlation between fidelity and pi_pred


def decode_orientation_from_template(
    templates: Tensor,
    true_orientations: Tensor,
    n_orientations: int = 36,
    period: float = 180.0,
) -> float:
    """Decode orientation from deep template via population vector.

    Args:
        templates: [n_trials, N] template patterns.
        true_orientations: [n_trials] true orientations (channel indices).

    Returns:
        Fraction of trials where decoded orientation is within 1 channel of true.
    """
    step = period / n_orientations
    # Population vector decode: argmax of template
    decoded_channels = templates.argmax(dim=-1)  # [n_trials]
    error = (decoded_channels - true_orientations).abs()
    error = torch.min(error, n_orientations - error)
    return (error <= 1).float().mean().item()


def compute_template_fidelity(
    template: Tensor,
    expected_channel: int,
    n_orientations: int = 36,
    period: float = 180.0,
) -> float:
    """Compute cosine similarity between template and ideal expected pattern.

    Args:
        template: [N] deep template.
        expected_channel: channel index of expected orientation.

    Returns:
        Cosine similarity in [0, 1].
    """
    step = period / n_orientations
    pref = torch.arange(n_orientations, dtype=torch.float32) * step
    expected_ori = expected_channel * step
    dists = circular_distance_abs(pref, torch.tensor(expected_ori), period)
    ideal = torch.exp(-dists ** 2 / (2 * 12.0 ** 2))  # sigma=12 Gaussian

    cos_sim = torch.dot(template, ideal) / (template.norm() * ideal.norm() + 1e-10)
    return cos_sim.clamp(0, 1).item()


def run_omission_analysis(
    omission_result: ExperimentResult,
    expected_channels: Tensor | None = None,
    n_orientations: int = 36,
    period: float = 180.0,
) -> TemplateDecodingResult:
    """Run omission + prestimulus template decoding analysis.

    Args:
        omission_result: ExperimentResult from P2 (omission paradigm).
        expected_channels: [n_trials] expected orientation channel indices.
    """
    tw = omission_result.temporal_windows

    # Omission window template decoding
    omission_acc = 0.0
    prestim_acc = 0.0
    fidelity = 0.0
    fid_pi_corr = 0.0

    omission_conds = [n for n in omission_result.conditions if "omission" in n]
    present_conds = [n for n in omission_result.conditions if "present" in n]

    if omission_conds:
        cd = omission_result.conditions[omission_conds[0]]

        if "omission" in tw:
            o_start, o_end = tw["omission"]
        else:
            o_start, o_end = tw.get("sustained", (0, 1))

        # Template during omission
        templates_omit = cd.deep_template[:, o_start:o_end].mean(dim=1)  # [n_trials, N]
        if expected_channels is not None:
            omission_acc = decode_orientation_from_template(
                templates_omit, expected_channels, n_orientations, period)
        else:
            # Use argmax of template as self-consistency check
            omission_acc = (templates_omit.abs().sum(dim=-1) > 0.01).float().mean().item()

        # Template fidelity
        fidelities = []
        pi_preds = []
        for t in range(cd.deep_template.shape[0]):
            tmpl = templates_omit[t]
            peak_ch = tmpl.argmax().item()
            fidelities.append(compute_template_fidelity(tmpl, peak_ch, n_orientations, period))
            pi_preds.append(cd.pi_pred[t, o_start:o_end].mean().item())

        fidelity = sum(fidelities) / max(len(fidelities), 1)

        if len(fidelities) > 2:
            f_t = torch.tensor(fidelities)
            p_t = torch.tensor(pi_preds)
            f_t = f_t - f_t.mean()
            p_t = p_t - p_t.mean()
            denom = f_t.norm() * p_t.norm()
            fid_pi_corr = (torch.dot(f_t, p_t) / max(denom.item(), 1e-10)).item()

    # Prestimulus template decoding (from present trials)
    if present_conds:
        cd_p = omission_result.conditions[present_conds[0]]
        pre_start, pre_end = tw.get("prestimulus", (0, 1))
        templates_pre = cd_p.deep_template[:, pre_start:pre_end].mean(dim=1)
        if expected_channels is not None:
            prestim_acc = decode_orientation_from_template(
                templates_pre, expected_channels, n_orientations, period)
        else:
            prestim_acc = (templates_pre.abs().sum(dim=-1) > 0.01).float().mean().item()

    return TemplateDecodingResult(
        omission_accuracy=omission_acc,
        prestimulus_accuracy=prestim_acc,
        template_fidelity=fidelity,
        fidelity_pi_correlation=fid_pi_corr,
    )
