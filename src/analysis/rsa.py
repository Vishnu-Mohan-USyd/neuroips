"""Analysis 5: Representational Similarity Analysis (RSA).

Pairwise distance matrices per condition.
Kendall's tau between expected and neutral RDMs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RSAResult:
    """Representational similarity analysis results."""
    rdms: dict[str, Tensor]         # {condition: [n_ori, n_ori] RDM}
    kendall_tau: dict[str, float]   # {condition_pair: tau}


def compute_rdm(responses: Tensor) -> Tensor:
    """Compute representational dissimilarity matrix.

    Args:
        responses: [n_stimuli, n_units] mean response per stimulus.

    Returns:
        rdm: [n_stimuli, n_stimuli] pairwise Euclidean distances.
    """
    n = responses.shape[0]
    rdm = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = (responses[i] - responses[j]).norm().item()
            rdm[i, j] = d
            rdm[j, i] = d
    return rdm


def kendall_tau(rdm_a: Tensor, rdm_b: Tensor) -> float:
    """Compute Kendall's tau-a between upper triangles of two RDMs.

    Args:
        rdm_a, rdm_b: [n, n] symmetric distance matrices.

    Returns:
        tau: correlation in [-1, 1].
    """
    n = rdm_a.shape[0]
    # Extract upper triangle
    idx = torch.triu_indices(n, n, offset=1)
    a = rdm_a[idx[0], idx[1]]
    b = rdm_b[idx[0], idx[1]]

    n_pairs = len(a)
    concordant = 0
    discordant = 0
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            sign_a = (a[i] - a[j]).sign()
            sign_b = (b[i] - b[j]).sign()
            product = sign_a * sign_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def run_rsa(
    condition_responses: dict[str, Tensor],
) -> RSAResult:
    """Run RSA across conditions.

    Args:
        condition_responses: {condition_name: [n_stimuli, n_units]}.

    Returns:
        RSAResult with RDMs and pairwise Kendall's tau.
    """
    rdms = {name: compute_rdm(resp) for name, resp in condition_responses.items()}

    tau_results: dict[str, float] = {}
    names = list(rdms.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = f"{names[i]}_vs_{names[j]}"
            tau_results[pair] = kendall_tau(rdms[names[i]], rdms[names[j]])

    return RSAResult(rdms=rdms, kendall_tau=tau_results)
