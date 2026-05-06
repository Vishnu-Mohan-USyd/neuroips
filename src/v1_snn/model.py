"""Core two-layer V1 spiking network simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import json

import torch

from .analysis import compute_osi
from .config import ModelConfig
from .connectivity import SparseProjection, build_connectivity
from .layout import NetworkLayout, build_layout
from .stimuli import (
    compute_l4_feedforward_drive,
    compute_l4_i_feedforward_drive,
    generate_grating,
    make_l4_gabor_bank,
)


@dataclass
class PopulationState:
    """Dynamical state for a neuron population."""

    voltage: torch.Tensor
    exc_current: torch.Tensor
    inh_current: torch.Tensor
    adaptation: torch.Tensor
    refractory: torch.Tensor
    spikes: torch.Tensor
    threshold: float
    reset: float
    refractory_ms: float
    tau_m_ms: float
    exc_decay: float
    inh_decay: float
    adapt_decay: float
    adapt_increment: float
    background_current: float


@dataclass(frozen=True)
class SimulationResult:
    """Collected responses from a stimulus protocol."""

    orientations_rad: torch.Tensor
    rates: dict[str, torch.Tensor]
    osi: dict[str, torch.Tensor]
    preferred_orientation: dict[str, torch.Tensor]

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "orientations_rad": self.orientations_rad.cpu(),
                "rates": {name: value.cpu() for name, value in self.rates.items()},
                "osi": {name: value.cpu() for name, value in self.osi.items()},
                "preferred_orientation": {name: value.cpu() for name, value in self.preferred_orientation.items()},
            },
            path / "responses.pt",
        )
        summary = {
            "mean_rate_hz": {name: float(value.mean().item()) for name, value in self.rates.items()},
            "mean_osi": {
                name: float(self.osi[name].mean().item())
                for name in self.osi
                if self.osi[name].numel() > 0
            },
        }
        (path / "summary.json").write_text(json.dumps(summary, indent=2))


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _make_state(size: int, params, dt_ms: float, device: torch.device) -> PopulationState:
    zeros = torch.zeros(size, dtype=torch.float32, device=device)
    return PopulationState(
        voltage=zeros.clone(),
        exc_current=zeros.clone(),
        inh_current=zeros.clone(),
        adaptation=zeros.clone(),
        refractory=zeros.clone(),
        spikes=zeros.clone(),
        threshold=params.threshold,
        reset=params.reset,
        refractory_ms=params.refractory_ms,
        tau_m_ms=params.tau_m_ms,
        exc_decay=math.exp(-dt_ms / params.tau_exc_ms),
        inh_decay=math.exp(-dt_ms / params.tau_inh_ms),
        adapt_decay=math.exp(-dt_ms / params.adapt_tau_ms),
        adapt_increment=params.adapt_increment,
        background_current=params.background_current,
    )


class V1TwoLayerSNN:
    """Scaled-down L4/L2/3 spiking network with fixed L4 Gabor drive."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = _resolve_device(config.simulation.device)
        torch.manual_seed(config.simulation.seed)
        self.layout: NetworkLayout = build_layout(config=config, device=self.device)
        self.gabor_bank = make_l4_gabor_bank(config=config, layout=self.layout, device=self.device)
        self.projections: dict[str, SparseProjection] = build_connectivity(
            config=config,
            layout=self.layout,
            device=self.device,
        )
        self.orientations_rad = torch.deg2rad(
            torch.tensor(config.simulation.stimulus_orientations_deg, dtype=torch.float32, device=self.device)
        )

    def _reset_states(self) -> dict[str, PopulationState]:
        states = {}
        for name, population in self.layout.populations.items():
            states[name] = _make_state(
                size=population.size,
                params=self.config.neuron_params(name),
                dt_ms=self.config.simulation.dt_ms,
                device=self.device,
            )
        return states

    def _external_drive(self, orientation_rad: float) -> dict[str, torch.Tensor]:
        image = generate_grating(
            size_px=self.config.simulation.stimulus_size_px,
            orientation_rad=orientation_rad,
            spatial_frequency_cycles=self.config.simulation.stimulus_spatial_frequency,
            contrast=self.config.simulation.stimulus_contrast,
            phase_rad=math.radians(self.config.simulation.stimulus_phase_deg),
            device=self.device,
        )
        l4_e = compute_l4_feedforward_drive(
            image=image,
            config=self.config,
            layout=self.layout,
            gabor_bank=self.gabor_bank,
        )
        l4_i = compute_l4_i_feedforward_drive(l4_e_drive=l4_e, config=self.config, layout=self.layout)
        zeros_l23_e = torch.zeros(self.layout.populations["l23_e"].size, dtype=torch.float32, device=self.device)
        zeros_l23_i = torch.zeros(self.layout.populations["l23_i"].size, dtype=torch.float32, device=self.device)
        return {
            "l4_e": l4_e,
            "l4_i": l4_i,
            "l23_e": zeros_l23_e,
            "l23_i": zeros_l23_i,
        }

    def _step(self, states: dict[str, PopulationState], external: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        noise_scale = self.config.simulation.background_noise_std
        dt = self.config.simulation.dt_ms

        for state in states.values():
            state.exc_current.mul_(state.exc_decay)
            state.inh_current.mul_(state.inh_decay)
            state.adaptation.mul_(state.adapt_decay)
            state.refractory.sub_(dt).clamp_(min=0.0)

        spike_outputs: dict[str, torch.Tensor] = {}
        for name, state in states.items():
            total_current = (
                external[name]
                + state.background_current
                + state.exc_current
                - state.inh_current
                - state.adaptation
            )
            if noise_scale > 0.0:
                total_current = total_current + noise_scale * torch.randn_like(total_current)
            ready = state.refractory <= 0.0
            dv = dt * (-state.voltage + total_current) / state.tau_m_ms
            state.voltage = torch.where(ready, state.voltage + dv, state.voltage)
            spikes = (state.voltage >= state.threshold) & ready
            state.voltage = torch.where(spikes, torch.full_like(state.voltage, state.reset), state.voltage)
            state.adaptation = state.adaptation + spikes.to(torch.float32) * state.adapt_increment
            state.refractory = torch.where(
                spikes,
                torch.full_like(state.refractory, state.refractory_ms),
                state.refractory,
            )
            state.spikes = spikes.to(torch.float32)
            spike_outputs[name] = state.spikes

        for projection in self.projections.values():
            pre_spikes = spike_outputs[projection.source].unsqueeze(1)
            delta = torch.sparse.mm(projection.matrix, pre_spikes).squeeze(1)
            if projection.is_excitatory:
                states[projection.target].exc_current.add_(delta)
            else:
                states[projection.target].inh_current.add_(delta)

        return spike_outputs

    def run_protocol(self) -> SimulationResult:
        """Present the configured orientation set and collect firing-rate responses."""

        rates: dict[str, list[torch.Tensor]] = {name: [] for name in self.layout.populations}
        response_seconds = max(self.config.simulation.steps_per_stimulus * self.config.simulation.dt_ms / 1000.0, 1e-6)

        for orientation in self.orientations_rad.tolist():
            states = self._reset_states()
            external = self._external_drive(orientation_rad=orientation)

            for _ in range(self.config.simulation.relaxation_steps):
                self._step(states=states, external=external)

            counts = {
                name: torch.zeros(population.size, dtype=torch.float32, device=self.device)
                for name, population in self.layout.populations.items()
            }
            for _ in range(self.config.simulation.steps_per_stimulus):
                spikes = self._step(states=states, external=external)
                for name in counts:
                    counts[name].add_(spikes[name])

            for name in counts:
                rates[name].append(counts[name] / response_seconds)

        stacked_rates = {name: torch.stack(value, dim=0) for name, value in rates.items()}
        osi: dict[str, torch.Tensor] = {}
        preferred: dict[str, torch.Tensor] = {}
        for name, rate in stacked_rates.items():
            if rate.numel() == 0:
                osi[name] = torch.empty(0, dtype=torch.float32, device=self.device)
                preferred[name] = torch.empty(0, dtype=torch.float32, device=self.device)
                continue
            osi[name], preferred[name] = compute_osi(rate, self.orientations_rad)

        return SimulationResult(
            orientations_rad=self.orientations_rad,
            rates=stacked_rates,
            osi=osi,
            preferred_orientation=preferred,
        )
