"""V2 laminar predictive circuit (scaffold).

Minimal mechanistic model for expectation effects (Kok 2012, Richter 2018/2019)
under local Hebbian + predictive + homeostatic plasticity. Full spec:
`plans/come-to-me-with-streamed-grove.md`.

This `__init__` re-exports the top-level dataclasses + state containers; lower
modules (layers, plasticity, network, context_memory, lgn_l4, ...) are added in
subsequent tasks.
"""

from __future__ import annotations

from src.v2_model.config import (
    ArchitectureConfig,
    ConnectivityConfig,
    EnergyConfig,
    ModelConfig,
    PlasticityConfig,
    RegimeConfig,
    TimeConstantsConfig,
)
from src.v2_model.freeze_manifest import FreezeManifest, PhaseFreezeSpec
from src.v2_model.state import NetworkStateV2, initial_state

__all__ = [
    "ArchitectureConfig",
    "ConnectivityConfig",
    "EnergyConfig",
    "FreezeManifest",
    "ModelConfig",
    "NetworkStateV2",
    "PhaseFreezeSpec",
    "PlasticityConfig",
    "RegimeConfig",
    "TimeConstantsConfig",
    "initial_state",
]
