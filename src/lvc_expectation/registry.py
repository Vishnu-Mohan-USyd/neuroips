"""Preset registry for small, explicit experiment configurations."""

from __future__ import annotations

from copy import deepcopy

from .config import ExperimentConfig, make_phase1_core_config

_PRESETS: dict[str, ExperimentConfig] = {
    "phase1_core": make_phase1_core_config(),
}


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def get_preset(name: str) -> ExperimentConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset: {name}")
    return deepcopy(_PRESETS[name])
