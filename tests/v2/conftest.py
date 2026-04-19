"""Shared fixtures for tests/v2.

Per v4 spec §Learning rules — no autograd in main path. Tests must run
deterministically so that any CUDA non-determinism introduced by local-plasticity
scatter_add (flagged in scope-impact §4(e)) is caught immediately.

Global fixtures here are autouse where it is safe (seed + deterministic mode);
other fixtures are opt-in so tests that need different sizes can override.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

SEED = 42


@pytest.fixture(autouse=True)
def _deterministic_mode():
    """Pin seeds + request deterministic algorithms for every test.

    Uses `warn_only=True` so tests don't fail on ops that have no deterministic
    CUDA kernel today (e.g. `scatter_add` on older CUDA). Task #11+ will tighten
    to `warn_only=False` once plasticity kernels are in place.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def device() -> torch.device:
    """CPU by default; tests that need CUDA should gate themselves explicitly."""
    return torch.device("cpu")


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def cfg():
    """Default v2 ModelConfig (population sizes from the v4 architecture table)."""
    from src.v2_model.config import ModelConfig
    return ModelConfig()
