"""Sparse connectivity generation for the V1 L4/L2/3 SNN."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .config import ConnectionRule, ModelConfig
from .layout import NetworkLayout, PopulationLayout


@dataclass(frozen=True)
class SparseProjection:
    """Sparse synaptic projection between two populations."""

    name: str
    source: str
    target: str
    is_excitatory: bool
    matrix: torch.Tensor
    pre_indices: torch.Tensor
    post_indices: torch.Tensor
    weights: torch.Tensor

    @property
    def edge_count(self) -> int:
        return int(self.weights.numel())


def _orientation_similarity(preferred_pre: np.ndarray, preferred_post: float, kappa: float) -> np.ndarray:
    if kappa <= 0.0 or math.isnan(preferred_post):
        return np.ones_like(preferred_pre, dtype=np.float64)
    return np.exp(kappa * np.cos(2.0 * (preferred_pre - preferred_post)))


def _effective_distance_sq(offsets: np.ndarray, theta: float, axial_ratio: float) -> np.ndarray:
    if axial_ratio <= 1.0 or math.isnan(theta):
        return np.sum(offsets * offsets, axis=1)
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    parallel = offsets[:, 0] * ctheta + offsets[:, 1] * stheta
    orthogonal = -offsets[:, 0] * stheta + offsets[:, 1] * ctheta
    return (parallel / axial_ratio) ** 2 + orthogonal**2


def _draw_weights(rule: ConnectionRule, count: int, rng: np.random.Generator) -> np.ndarray:
    if rule.is_excitatory:
        sigma = rule.weight_sigma
        mu = math.log(rule.weight_mean) - 0.5 * sigma**2
        return rng.lognormal(mean=mu, sigma=sigma, size=count)
    std = max(rule.weight_mean * rule.weight_sigma, 1e-6)
    weights = rng.normal(loc=rule.weight_mean, scale=std, size=count)
    return np.clip(weights, 1e-6, None)


def _sample_projection(
    name: str,
    rule: ConnectionRule,
    pre: PopulationLayout,
    post: PopulationLayout,
    site_positions: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
) -> SparseProjection:
    pre_sites = pre.site_ids.cpu().numpy()
    post_sites = post.site_ids.cpu().numpy()
    pre_orient = pre.preferred_orientation.cpu().numpy()
    post_orient = post.preferred_orientation.cpu().numpy()
    pre_by_site = [np.flatnonzero(pre_sites == site_id) for site_id in range(site_positions.shape[0])]

    pre_edges: list[np.ndarray] = []
    post_edges: list[np.ndarray] = []
    syn_weights: list[np.ndarray] = []

    for post_index in range(post.size):
        post_site = post_sites[post_index]
        offsets = site_positions - site_positions[post_site]
        dist_sq = _effective_distance_sq(offsets=offsets, theta=post_orient[post_index], axial_ratio=rule.axial_ratio)
        candidate_sites = np.flatnonzero(dist_sq <= rule.radius_sites**2)
        if candidate_sites.size == 0:
            continue

        candidate_indices = np.concatenate([pre_by_site[site_id] for site_id in candidate_sites])
        if candidate_indices.size == 0:
            continue
        if not rule.allow_self and pre.name == post.name:
            candidate_indices = candidate_indices[candidate_indices != post_index]
        if candidate_indices.size == 0:
            continue

        candidate_site_ids = pre_sites[candidate_indices]
        spatial = np.exp(-0.5 * dist_sq[candidate_site_ids] / (rule.sigma_sites**2))
        feature = _orientation_similarity(pre_orient[candidate_indices], post_orient[post_index], rule.orientation_kappa)
        probability = spatial * feature + 1e-12
        fan_in = min(rule.fan_in, candidate_indices.size)
        chosen = rng.choice(candidate_indices.size, size=fan_in, replace=False, p=probability / probability.sum())
        selected_pre = candidate_indices[chosen]
        weights = _draw_weights(rule=rule, count=fan_in, rng=rng)

        pre_edges.append(selected_pre.astype(np.int64))
        post_edges.append(np.full(fan_in, post_index, dtype=np.int64))
        syn_weights.append(weights.astype(np.float32))

    if not syn_weights:
        raise RuntimeError(f"Projection {name} generated no edges")

    pre_array = np.concatenate(pre_edges)
    post_array = np.concatenate(post_edges)
    weight_array = np.concatenate(syn_weights)
    indices = torch.tensor(np.vstack([post_array, pre_array]), dtype=torch.long, device=device)
    values = torch.tensor(weight_array, dtype=torch.float32, device=device)
    matrix = torch.sparse_coo_tensor(indices, values, (post.size, pre.size), device=device).coalesce()
    return SparseProjection(
        name=name,
        source=rule.source,
        target=rule.target,
        is_excitatory=rule.is_excitatory,
        matrix=matrix,
        pre_indices=torch.tensor(pre_array, dtype=torch.long, device=device),
        post_indices=torch.tensor(post_array, dtype=torch.long, device=device),
        weights=values,
    )


def build_connectivity(config: ModelConfig, layout: NetworkLayout, device: torch.device) -> dict[str, SparseProjection]:
    """Sample all sparse projections for the two-layer network."""

    rng = np.random.default_rng(config.simulation.seed)
    site_positions = layout.site_positions.cpu().numpy()
    projections: dict[str, SparseProjection] = {}
    for name, rule in config.connectivity.items():
        projections[name] = _sample_projection(
            name=name,
            rule=rule,
            pre=layout.populations[rule.source],
            post=layout.populations[rule.target],
            site_positions=site_positions,
            rng=rng,
            device=device,
        )
    return projections
