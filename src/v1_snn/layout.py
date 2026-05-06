"""Network layout generation for the V1 L4/L2/3 SNN."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .config import ModelConfig


@dataclass(frozen=True)
class PopulationLayout:
    """Metadata for a neuron population on the retinotopic sheet."""

    name: str
    site_ids: torch.Tensor
    positions: torch.Tensor
    preferred_orientation: torch.Tensor
    phases: torch.Tensor | None

    @property
    def size(self) -> int:
        return int(self.site_ids.numel())


@dataclass(frozen=True)
class NetworkLayout:
    """Complete geometric layout for the two-sheet network."""

    site_positions: torch.Tensor
    orientation_map: torch.Tensor
    populations: dict[str, PopulationLayout]
    side: int

    @property
    def site_count(self) -> int:
        return int(self.site_positions.shape[0])


def _generate_orientation_map(side: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a smooth orientation map with pinwheel-like singularities."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real_noise = torch.randn((side, side), generator=generator, dtype=torch.float32)
    imag_noise = torch.randn((side, side), generator=generator, dtype=torch.float32)
    complex_noise = torch.complex(real_noise, imag_noise).to(device)

    frequencies = torch.fft.fftfreq(side, device=device)
    ky, kx = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius_sq = kx.square() + ky.square()
    lowpass = torch.exp(-radius_sq / (2.0 * 0.10**2))
    filtered = torch.fft.ifft2(torch.fft.fft2(complex_noise) * lowpass)
    orientation = 0.5 * torch.angle(filtered)
    return torch.remainder(orientation, math.pi)


def _make_population(
    name: str,
    per_site: int,
    site_positions: torch.Tensor,
    site_orientation: torch.Tensor,
    seed: int,
    device: torch.device,
    tuned: bool,
) -> PopulationLayout:
    """Instantiate one population with retinotopic positions and orientation tags."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    site_count = int(site_positions.shape[0])
    site_ids = torch.arange(site_count, dtype=torch.long, device=device).repeat_interleave(per_site)
    positions = site_positions.repeat_interleave(per_site, dim=0)

    if tuned:
        jitter = torch.randn((site_count, per_site), generator=generator, dtype=torch.float32, device=device)
        preferred = torch.remainder(site_orientation[:, None] + jitter * (math.pi / 18.0), math.pi).reshape(-1)
    else:
        preferred = torch.full((site_count * per_site,), float("nan"), dtype=torch.float32, device=device)

    phases: torch.Tensor | None = None
    if name == "l4_e":
        base_phases = torch.tensor([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], dtype=torch.float32, device=device)
        phases = base_phases.repeat(math.ceil(per_site / base_phases.numel()))[:per_site]
        phases = phases.repeat(site_count)

    return PopulationLayout(
        name=name,
        site_ids=site_ids,
        positions=positions,
        preferred_orientation=preferred,
        phases=phases,
    )


def build_layout(config: ModelConfig, device: torch.device) -> NetworkLayout:
    """Build the retinotopic sheet geometry and orientation map."""

    side = config.sheet.side
    grid_y, grid_x = torch.meshgrid(
        torch.arange(side, dtype=torch.float32, device=device),
        torch.arange(side, dtype=torch.float32, device=device),
        indexing="ij",
    )
    site_positions = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    orientation_map = _generate_orientation_map(side=side, seed=config.simulation.seed, device=device)
    site_orientation = orientation_map.reshape(-1)

    populations = {
        "l4_e": _make_population(
            name="l4_e",
            per_site=config.populations.l4_e_per_site,
            site_positions=site_positions,
            site_orientation=site_orientation,
            seed=config.simulation.seed + 11,
            device=device,
            tuned=True,
        ),
        "l4_i": _make_population(
            name="l4_i",
            per_site=config.populations.l4_i_per_site,
            site_positions=site_positions,
            site_orientation=site_orientation,
            seed=config.simulation.seed + 13,
            device=device,
            tuned=False,
        ),
        "l23_e": _make_population(
            name="l23_e",
            per_site=config.populations.l23_e_per_site,
            site_positions=site_positions,
            site_orientation=site_orientation,
            seed=config.simulation.seed + 17,
            device=device,
            tuned=True,
        ),
        "l23_i": _make_population(
            name="l23_i",
            per_site=config.populations.l23_i_per_site,
            site_positions=site_positions,
            site_orientation=site_orientation,
            seed=config.simulation.seed + 19,
            device=device,
            tuned=False,
        ),
    }

    return NetworkLayout(
        site_positions=site_positions,
        orientation_map=orientation_map,
        populations=populations,
        side=side,
    )
