"""Analysis 9: Synthetic observation model.

Pool L2/3 into synthetic voxels, compute synthetic BOLD + MVPA.
Multiple voxel granularities and noise levels (validated by recovery analysis).
Key test: same ground truth -> different MVPA vs univariate conclusions?
CS uniquely predicts BOLD-dampening + MVPA-sharpening dissociation (Kok 2012).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.experiments.paradigm_base import ConditionData


@dataclass
class ObservationModelResult:
    """Results from synthetic observation model."""
    univariate_effect: dict[str, float]    # {condition: mean voxel response}
    mvpa_accuracy_2way: float
    mvpa_accuracy_3way: float
    n_voxels: int
    snr: float
    dissociation: bool                      # univariate vs MVPA disagree?


def pool_to_voxels(
    r_l23: Tensor,
    n_voxels: int = 8,
) -> Tensor:
    """Pool L2/3 units into synthetic voxels.

    Args:
        r_l23: [n_trials, N] L2/3 responses (time-averaged).

    Returns:
        voxel_responses: [n_trials, n_voxels].
    """
    N = r_l23.shape[1]
    units_per_voxel = N // n_voxels
    pool_w = torch.zeros(n_voxels, N, device=r_l23.device)
    for v in range(n_voxels):
        start = v * units_per_voxel
        end = min(start + units_per_voxel, N)
        pool_w[v, start:end] = 1.0 / max(end - start, 1)
    return r_l23 @ pool_w.T


def add_noise(
    signal: Tensor,
    snr: float,
    seed: int = 42,
) -> Tensor:
    """Add Gaussian noise at specified SNR."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise_std = signal.abs().mean() / max(snr, 1e-6)
    noise = torch.randn_like(signal, generator=gen if signal.device.type == "cpu" else None) * noise_std
    return signal + noise


def run_observation_model(
    condition_data: dict[str, Tensor],
    n_voxels: int = 8,
    snr: float = 5.0,
    n_folds: int = 5,
    seed: int = 42,
) -> ObservationModelResult:
    """Run synthetic observation model on condition responses.

    Args:
        condition_data: {condition_name: [n_trials, N] L2/3 responses}.
        n_voxels: Number of synthetic voxels.
        snr: Signal-to-noise ratio.

    Returns:
        ObservationModelResult.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Pool and add noise
    voxel_data: dict[str, Tensor] = {}
    for name, resp in condition_data.items():
        pooled = pool_to_voxels(resp, n_voxels)
        voxel_data[name] = add_noise(pooled, snr, seed)

    # Univariate: mean voxel response per condition
    univariate: dict[str, float] = {}
    for name, vox in voxel_data.items():
        univariate[name] = vox.mean().item()

    # MVPA: nearest centroid
    cond_names = sorted(voxel_data.keys())
    all_patterns = []
    all_labels = []
    for i, name in enumerate(cond_names):
        all_patterns.append(voxel_data[name])
        all_labels.append(torch.full((voxel_data[name].shape[0],), i, dtype=torch.long))

    patterns = torch.cat(all_patterns)
    labels = torch.cat(all_labels)

    # 3-way classification
    acc_3way = _cv_classify(patterns, labels, n_folds, seed)

    # 2-way (first two conditions)
    if len(cond_names) >= 2:
        mask = labels < 2
        acc_2way = _cv_classify(patterns[mask], labels[mask], n_folds, seed)
    else:
        acc_2way = acc_3way

    # Dissociation: univariate says one thing, MVPA says another
    uni_vals = list(univariate.values())
    uni_suppressed = uni_vals[0] < uni_vals[-1] if len(uni_vals) >= 2 else False
    mvpa_discriminates = acc_2way > 0.6

    return ObservationModelResult(
        univariate_effect=univariate,
        mvpa_accuracy_2way=acc_2way,
        mvpa_accuracy_3way=acc_3way,
        n_voxels=n_voxels,
        snr=snr,
        dissociation=uni_suppressed and mvpa_discriminates,
    )


def _cv_classify(
    patterns: Tensor, labels: Tensor, n_folds: int, seed: int,
) -> float:
    """Cross-validated nearest-centroid classifier."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    n = patterns.shape[0]
    perm = torch.randperm(n, generator=gen)
    fold_size = n // max(n_folds, 1)
    if fold_size == 0:
        return 0.0

    correct = 0
    total = 0
    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = torch.cat([perm[:fold * fold_size], perm[(fold + 1) * fold_size:]])
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        centroids = {}
        for c in labels[train_idx].unique():
            mask = labels[train_idx] == c
            centroids[c.item()] = patterns[train_idx][mask].mean(dim=0)

        for i in range(len(test_idx)):
            dists = {c: ((patterns[test_idx[i]] - cent) ** 2).sum()
                     for c, cent in centroids.items()}
            pred = min(dists, key=dists.get)
            if pred == labels[test_idx[i]].item():
                correct += 1
            total += 1

    return correct / max(total, 1)
