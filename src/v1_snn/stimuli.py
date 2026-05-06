"""Stimulus generation and fixed L4 Gabor drive."""

from __future__ import annotations

import math

import torch

from .config import ModelConfig
from .layout import NetworkLayout


def generate_grating(
    size_px: int,
    orientation_rad: float,
    spatial_frequency_cycles: float,
    contrast: float,
    phase_rad: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate a full-field sinusoidal grating."""

    coords = torch.linspace(-1.0, 1.0, size_px, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    axis = xx * math.cos(orientation_rad) + yy * math.sin(orientation_rad)
    carrier = torch.cos(2.0 * math.pi * spatial_frequency_cycles * axis + phase_rad)
    return contrast * carrier


def make_l4_gabor_bank(config: ModelConfig, layout: NetworkLayout, device: torch.device) -> torch.Tensor:
    """Create one fixed Gabor kernel per L4 excitatory neuron."""

    l4_e = layout.populations["l4_e"]
    patch = config.sheet.rf_patch_px
    coords = torch.linspace(-(patch // 2), patch // 2, patch, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    theta = l4_e.preferred_orientation[:, None, None]
    phase = l4_e.phases[:, None, None] if l4_e.phases is not None else 0.0

    x_theta = xx * torch.cos(theta) + yy * torch.sin(theta)
    y_theta = -xx * torch.sin(theta) + yy * torch.cos(theta)
    envelope = torch.exp(
        -(x_theta.square() + (config.gabor.aspect_ratio**2) * y_theta.square())
        / (2.0 * config.gabor.sigma_px**2)
    )
    carrier = torch.cos(2.0 * math.pi * x_theta / config.gabor.wavelength_px + phase)
    kernels = envelope * carrier
    kernels = kernels - kernels.mean(dim=(1, 2), keepdim=True)
    kernels = kernels / kernels.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    return kernels.reshape(l4_e.size, -1)


def extract_site_patches(image: torch.Tensor, side: int, patch_px: int) -> torch.Tensor:
    """Extract one local receptive-field patch per retinotopic site."""

    margin = patch_px // 2
    size_px = int(image.shape[-1])
    centers = torch.linspace(margin, size_px - margin - 1, side, device=image.device)
    centers = centers.round().to(torch.long)
    patches = []
    for y in centers.tolist():
        for x in centers.tolist():
            patch = image[y - margin : y + margin + 1, x - margin : x + margin + 1]
            patches.append(patch.reshape(-1))
    return torch.stack(patches, dim=0)


def compute_l4_feedforward_drive(
    image: torch.Tensor,
    config: ModelConfig,
    layout: NetworkLayout,
    gabor_bank: torch.Tensor,
) -> torch.Tensor:
    """Project the image through fixed L4 Gabor filters."""

    site_patches = extract_site_patches(image=image, side=layout.side, patch_px=config.sheet.rf_patch_px)
    l4_e = layout.populations["l4_e"]
    neuron_patches = site_patches.index_select(0, l4_e.site_ids)
    responses = (neuron_patches * gabor_bank).sum(dim=1)
    return torch.relu(responses) * config.gabor.gain + config.gabor.bias


def compute_l4_i_feedforward_drive(
    l4_e_drive: torch.Tensor,
    config: ModelConfig,
    layout: NetworkLayout,
) -> torch.Tensor:
    """Derive a broad untuned drive to L4 inhibition from local L4 excitation."""

    l4_e = layout.populations["l4_e"]
    l4_i = layout.populations["l4_i"]
    site_count = layout.site_count
    site_sum = torch.zeros(site_count, dtype=torch.float32, device=l4_e_drive.device)
    site_sum.index_add_(0, l4_e.site_ids, l4_e_drive)
    counts = torch.bincount(l4_e.site_ids, minlength=site_count).to(torch.float32)
    site_mean = site_sum / counts.clamp_min(1.0)
    return config.gabor.l4_i_pool_gain * site_mean.index_select(0, l4_i.site_ids)
