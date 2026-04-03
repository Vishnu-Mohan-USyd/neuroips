"""Feedback profile extraction and classification for the emergent operator.

Given a trained model with EmergentFeedbackOperator, this module:
1. Extracts the learned K_inh and K_exc profiles.
2. Classifies them against known mechanism templates (dampening, sharpening,
   center-surround) using template correlation.
3. Computes the net modulation profile (K_exc - K_inh) for interpretation.

Usage:
    from src.analysis.feedback_discovery import extract_profiles, classify_profile

    K_inh, K_exc = extract_profiles(model)
    result = classify_profile(K_inh, K_exc)
    print(result['winning_class'], result['correlations'])
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils import circular_distance_abs


def extract_profiles(model) -> tuple[Tensor, Tensor]:
    """Get the learned K_inh and K_exc profiles from the emergent operator.

    Args:
        model: LaminarV1V2Network with feedback_mode='emergent'.

    Returns:
        K_inh: [N] inhibitory (SOM) kernel profile.
        K_exc: [N] excitatory (L2/3 center) kernel profile.
    """
    K_inh, K_exc = model.feedback.get_profiles()
    return K_inh.detach().cpu(), K_exc.detach().cpu()


def _make_template(
    N: int,
    period: float,
    template_type: str,
) -> Tensor:
    """Build a canonical mechanism template for correlation analysis.

    Args:
        N: Number of orientation channels.
        period: Orientation range in degrees.
        template_type: One of 'dampening', 'sharpening', 'center_surround'.

    Returns:
        Template [N] centered at channel 0.
    """
    step = period / N
    thetas = torch.arange(N, dtype=torch.float32) * step
    dists = circular_distance_abs(
        thetas.unsqueeze(0), thetas[0:1].unsqueeze(1), period
    ).squeeze(0)

    if template_type == 'dampening':
        # Narrow positive peak at center: SOM inhibits AT expected
        sigma = 10.0
        template = torch.exp(-dists ** 2 / (2 * sigma ** 2))
        template = template / template.sum()

    elif template_type == 'sharpening':
        # DoG: broad - narrow -> minimum at center, maxima at flanks
        sigma_broad = 35.0
        sigma_narrow = 10.0
        broad = torch.exp(-dists ** 2 / (2 * sigma_broad ** 2))
        narrow = torch.exp(-dists ** 2 / (2 * sigma_narrow ** 2))
        template = broad / broad.sum() - narrow / narrow.sum()

    elif template_type == 'center_surround':
        # Broad positive SOM - narrow center excitation
        # Net modulation: negative at center (exc > inh), positive at flanks (inh > exc)
        sigma_broad = 30.0
        sigma_narrow = 10.0
        broad = torch.exp(-dists ** 2 / (2 * sigma_broad ** 2))
        narrow = torch.exp(-dists ** 2 / (2 * sigma_narrow ** 2))
        template = broad / broad.sum() - 0.8 * narrow / narrow.sum()

    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    return template


def classify_profile(
    K_inh: Tensor,
    K_exc: Tensor,
    period: float = 180.0,
) -> dict:
    """Classify the learned feedback profiles against known mechanism templates.

    Computes Pearson correlation between the net modulation profile (K_exc - K_inh)
    and canonical templates for each mechanism type.

    Args:
        K_inh: [N] inhibitory profile.
        K_exc: [N] excitatory profile.
        period: Orientation range in degrees.

    Returns:
        dict with:
            'net_modulation': [N] net effect on L2/3 (K_exc - K_inh)
            'correlations': dict of {mechanism_name: R value}
            'r_squared': dict of {mechanism_name: R^2 value}
            'winning_class': name of the best-matching mechanism
            'winning_r2': R^2 of the best match
            'K_inh': [N] inhibitory profile
            'K_exc': [N] excitatory profile
    """
    N = K_inh.shape[0]
    net_mod = K_exc - K_inh  # [N]

    template_types = ['dampening', 'sharpening', 'center_surround']
    correlations = {}
    r_squared = {}

    for ttype in template_types:
        template = _make_template(N, period, ttype)

        # Pearson correlation
        x = net_mod - net_mod.mean()
        y = template - template.mean()
        r = (x * y).sum() / (x.norm() * y.norm() + 1e-8)
        correlations[ttype] = r.item()
        r_squared[ttype] = r.item() ** 2

    # Find winner
    winning_class = max(r_squared, key=r_squared.get)

    return {
        'net_modulation': net_mod,
        'correlations': correlations,
        'r_squared': r_squared,
        'winning_class': winning_class,
        'winning_r2': r_squared[winning_class],
        'K_inh': K_inh,
        'K_exc': K_exc,
    }
