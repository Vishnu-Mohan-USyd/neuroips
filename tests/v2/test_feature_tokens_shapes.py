"""Shape + dtype + API contract for ``src.v2_model.stimuli.feature_tokens``.

Per v4 plan step 11 / Task #25:
  * ``TokenBank.tokens`` is a buffer of shape ``[12, 1, 32, 32]`` in float32,
    clipped to ``[0, 1]``.
  * The tensor is registered as a buffer (no parameters, no plasticity).
  * ``TokenBank(cfg)`` takes a ``ModelConfig`` and an optional seed; the
    ``tokens`` tensor is deterministic in the seed.
"""

from __future__ import annotations

import torch

from src.v2_model.stimuli.feature_tokens import (
    GRID_SIZE,
    N_TOKENS,
    TokenBank,
)


def test_tokens_shape(cfg) -> None:
    """``tokens.shape == [12, 1, 32, 32]`` — exact spec shape."""
    bank = TokenBank(cfg, seed=0)
    assert bank.tokens.shape == (N_TOKENS, 1, GRID_SIZE, GRID_SIZE)


def test_tokens_dtype_is_float32(cfg) -> None:
    bank = TokenBank(cfg, seed=0)
    assert bank.tokens.dtype == torch.float32


def test_tokens_in_unit_interval(cfg) -> None:
    """Pixel values strictly in ``[0, 1]`` (explicit range clip applied)."""
    bank = TokenBank(cfg, seed=0)
    assert float(bank.tokens.min()) >= 0.0
    assert float(bank.tokens.max()) <= 1.0


def test_tokens_registered_as_buffer(cfg) -> None:
    """``tokens`` must be a *buffer*, not a parameter — no plasticity."""
    bank = TokenBank(cfg, seed=0)
    buffer_names = {name for name, _ in bank.named_buffers()}
    assert "tokens" in buffer_names
    # No parameters allowed at all — the bank is fully frozen.
    assert list(bank.parameters()) == []


def test_tokens_not_in_parameters(cfg) -> None:
    """Defensive: ``tokens`` id must not appear in ``parameters()``."""
    bank = TokenBank(cfg, seed=0)
    tokens_id = id(bank.tokens)
    assert all(id(p) != tokens_id for p in bank.parameters())


def test_tokens_count_is_12(cfg) -> None:
    bank = TokenBank(cfg, seed=0)
    assert bank.tokens.shape[0] == N_TOKENS == 12


def test_tokens_grayscale_channel(cfg) -> None:
    """Single grayscale channel (matches LGNL4FrontEnd's expected input)."""
    bank = TokenBank(cfg, seed=0)
    assert bank.tokens.shape[1] == 1


def test_seed_attribute_preserved(cfg) -> None:
    """``bank.seed`` round-trips the constructor argument."""
    bank = TokenBank(cfg, seed=17)
    assert bank.seed == 17
