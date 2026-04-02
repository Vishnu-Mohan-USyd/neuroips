"""Core package for phase-1 lower visual cortex expectation modeling."""

from .config import ExperimentConfig, load_config, make_phase1_core_config
from .geometry import OrientationGeometry
from .registry import get_preset, list_presets

__all__ = [
    "ExperimentConfig",
    "OrientationGeometry",
    "get_preset",
    "list_presets",
    "load_config",
    "make_phase1_core_config",
]
