"""Procedural synthetic world with hidden regime (Phase 1 step 12).

Re-exports :class:`ProceduralWorld` from :mod:`src.v2_model.world.procedural`.
"""

from __future__ import annotations

from src.v2_model.world.procedural import (
    JUMP_PROBS,
    REGIMES,
    REGIME_PERSIST_PROB,
    ProceduralWorld,
    WorldState,
)

__all__ = [
    "JUMP_PROBS",
    "REGIMES",
    "REGIME_PERSIST_PROB",
    "ProceduralWorld",
    "WorldState",
]
