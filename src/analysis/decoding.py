"""Analysis 4: Decoding, Fisher information, and d-prime.

Linear SVM decoder with 5-fold CV per condition.
d-prime between adjacent orientations.
Fisher information from population response Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DecodingResult:
    """Results from orientation decoding analysis."""
    accuracy: float                    # overall decoding accuracy
    accuracy_per_condition: dict[str, float]  # per condition
    d_prime: dict[str, float]         # d-prime per condition
    fisher_info: Tensor | None         # [n_orientations] Fisher information


def nearest_centroid_decode(
    train_X: Tensor, train_y: Tensor,
    test_X: Tensor, test_y: Tensor,
) -> float:
    """Nearest-centroid classifier. Returns accuracy."""
    classes = train_y.unique()
    centroids = {}
    for c in classes:
        mask = train_y == c.item()
        if mask.sum() > 0:
            centroids[c.item()] = train_X[mask].mean(dim=0)

    correct = 0
    for i in range(len(test_X)):
        dists = {c: ((test_X[i] - cent) ** 2).sum() for c, cent in centroids.items()}
        pred = min(dists, key=dists.get)
        if pred == test_y[i].item():
            correct += 1
    return correct / max(len(test_X), 1)


def cross_validated_decoding(
    patterns: Tensor,
    labels: Tensor,
    n_folds: int = 5,
    seed: int = 42,
) -> float:
    """K-fold cross-validated nearest-centroid decoding.

    Args:
        patterns: [n_trials, n_features].
        labels: [n_trials] integer labels.

    Returns:
        Mean accuracy across folds.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    n = patterns.shape[0]
    perm = torch.randperm(n, generator=gen)
    fold_size = n // n_folds
    accs = []

    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = torch.cat([perm[:fold * fold_size], perm[(fold + 1) * fold_size:]])
        acc = nearest_centroid_decode(
            patterns[train_idx], labels[train_idx],
            patterns[test_idx], labels[test_idx],
        )
        accs.append(acc)

    return sum(accs) / len(accs)


def compute_d_prime(
    responses_a: Tensor, responses_b: Tensor,
) -> float:
    """Compute d-prime between two response distributions.

    Args:
        responses_a, responses_b: [n_trials, n_features].

    Returns:
        d-prime (scalar).
    """
    mean_a = responses_a.mean(dim=0)
    mean_b = responses_b.mean(dim=0)
    var_a = responses_a.var(dim=0).mean()
    var_b = responses_b.var(dim=0).mean()
    pooled_std = ((var_a + var_b) / 2).sqrt()
    if pooled_std < 1e-10:
        return 0.0
    delta = (mean_a - mean_b).norm()
    return (delta / pooled_std).item()


def compute_fisher_information(
    responses: Tensor,
    orientations: Tensor,
    period: float = 180.0,
) -> Tensor:
    """Estimate Fisher information from population response Jacobian.

    Uses finite differences to approximate df/dtheta.

    Args:
        responses: [n_orientations, n_units] mean responses at each orientation.
        orientations: [n_orientations] in degrees.

    Returns:
        fisher: [n_orientations] Fisher information at each orientation.
    """
    n_ori, n_units = responses.shape
    fisher = torch.zeros(n_ori)

    for i in range(n_ori):
        i_next = (i + 1) % n_ori
        i_prev = (i - 1) % n_ori
        d_theta = orientations[i_next] - orientations[i_prev]
        if d_theta <= 0:
            d_theta += period
        d_theta_rad = d_theta * torch.pi / 180.0

        # df/dtheta via central difference
        df = (responses[i_next] - responses[i_prev]) / (2 * d_theta_rad)
        # Variance at this orientation (assume Poisson-like: var ~ mean)
        var = responses[i].clamp(min=1e-6)
        fisher[i] = (df ** 2 / var).sum()

    return fisher
