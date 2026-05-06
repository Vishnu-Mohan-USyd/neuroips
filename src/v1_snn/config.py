"""Configuration loading for the V1 L4/L2/3 SNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    device: str
    dt_ms: float
    stimulus_size_px: int
    steps_per_stimulus: int
    relaxation_steps: int
    stimulus_contrast: float
    stimulus_spatial_frequency: float
    stimulus_phase_deg: float
    background_noise_std: float
    stimulus_orientations_deg: tuple[float, ...]


@dataclass(frozen=True)
class SheetConfig:
    side: int
    rf_patch_px: int


@dataclass(frozen=True)
class PopulationConfig:
    l4_e_per_site: int
    l4_i_per_site: int
    l23_e_per_site: int
    l23_i_per_site: int


@dataclass(frozen=True)
class GaborConfig:
    sigma_px: float
    wavelength_px: float
    aspect_ratio: float
    gain: float
    bias: float
    l4_i_pool_gain: float


@dataclass(frozen=True)
class NeuronParams:
    tau_m_ms: float
    tau_exc_ms: float
    tau_inh_ms: float
    refractory_ms: float
    threshold: float
    reset: float
    adapt_tau_ms: float
    adapt_increment: float
    background_current: float


@dataclass(frozen=True)
class NeuronConfig:
    excitatory: NeuronParams
    inhibitory: NeuronParams


@dataclass(frozen=True)
class ConnectionRule:
    source: str
    target: str
    sign: str
    fan_in: int
    sigma_sites: float
    radius_sites: float
    orientation_kappa: float
    axial_ratio: float
    weight_mean: float
    weight_sigma: float
    allow_self: bool

    @property
    def is_excitatory(self) -> bool:
        return self.sign == "exc"


@dataclass(frozen=True)
class ModelConfig:
    simulation: SimulationConfig
    sheet: SheetConfig
    populations: PopulationConfig
    gabor: GaborConfig
    neurons: NeuronConfig
    connectivity: dict[str, ConnectionRule]

    def neuron_params(self, population_name: str) -> NeuronParams:
        if population_name.endswith("_e"):
            return self.neurons.excitatory
        return self.neurons.inhibitory


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_simulation(raw: dict[str, Any]) -> SimulationConfig:
    return SimulationConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        dt_ms=float(raw["dt_ms"]),
        stimulus_size_px=int(raw["stimulus_size_px"]),
        steps_per_stimulus=int(raw["steps_per_stimulus"]),
        relaxation_steps=int(raw["relaxation_steps"]),
        stimulus_contrast=float(raw["stimulus_contrast"]),
        stimulus_spatial_frequency=float(raw["stimulus_spatial_frequency"]),
        stimulus_phase_deg=float(raw["stimulus_phase_deg"]),
        background_noise_std=float(raw["background_noise_std"]),
        stimulus_orientations_deg=tuple(float(value) for value in raw["stimulus_orientations_deg"]),
    )


def _load_sheet(raw: dict[str, Any]) -> SheetConfig:
    return SheetConfig(side=int(raw["side"]), rf_patch_px=int(raw["rf_patch_px"]))


def _load_populations(raw: dict[str, Any]) -> PopulationConfig:
    return PopulationConfig(
        l4_e_per_site=int(raw["l4_e_per_site"]),
        l4_i_per_site=int(raw["l4_i_per_site"]),
        l23_e_per_site=int(raw["l23_e_per_site"]),
        l23_i_per_site=int(raw["l23_i_per_site"]),
    )


def _load_gabor(raw: dict[str, Any]) -> GaborConfig:
    return GaborConfig(
        sigma_px=float(raw["sigma_px"]),
        wavelength_px=float(raw["wavelength_px"]),
        aspect_ratio=float(raw["aspect_ratio"]),
        gain=float(raw["gain"]),
        bias=float(raw["bias"]),
        l4_i_pool_gain=float(raw["l4_i_pool_gain"]),
    )


def _load_neuron_params(raw: dict[str, Any]) -> NeuronParams:
    return NeuronParams(
        tau_m_ms=float(raw["tau_m_ms"]),
        tau_exc_ms=float(raw["tau_exc_ms"]),
        tau_inh_ms=float(raw["tau_inh_ms"]),
        refractory_ms=float(raw["refractory_ms"]),
        threshold=float(raw["threshold"]),
        reset=float(raw["reset"]),
        adapt_tau_ms=float(raw["adapt_tau_ms"]),
        adapt_increment=float(raw["adapt_increment"]),
        background_current=float(raw["background_current"]),
    )


def _load_connectivity(raw: dict[str, Any]) -> dict[str, ConnectionRule]:
    rules: dict[str, ConnectionRule] = {}
    for name, rule in raw.items():
        rules[name] = ConnectionRule(
            source=str(rule["source"]),
            target=str(rule["target"]),
            sign=str(rule["sign"]),
            fan_in=int(rule["fan_in"]),
            sigma_sites=float(rule["sigma_sites"]),
            radius_sites=float(rule["radius_sites"]),
            orientation_kappa=float(rule["orientation_kappa"]),
            axial_ratio=float(rule["axial_ratio"]),
            weight_mean=float(rule["weight_mean"]),
            weight_sigma=float(rule["weight_sigma"]),
            allow_self=bool(rule["allow_self"]),
        )
    return rules


def load_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    raw = _read_toml(config_path)
    return ModelConfig(
        simulation=_load_simulation(raw["simulation"]),
        sheet=_load_sheet(raw["sheet"]),
        populations=_load_populations(raw["populations"]),
        gabor=_load_gabor(raw["gabor"]),
        neurons=NeuronConfig(
            excitatory=_load_neuron_params(raw["neurons"]["excitatory"]),
            inhibitory=_load_neuron_params(raw["neurons"]["inhibitory"]),
        ),
        connectivity=_load_connectivity(raw["connectivity"]),
    )
