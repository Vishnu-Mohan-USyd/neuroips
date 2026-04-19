"""Stimuli subpackage — stimulus banks and procedural-world generators.

Currently houses ``feature_tokens.TokenBank``; ``procedural_world`` lands in
a later task.
"""

from __future__ import annotations

from src.v2_model.stimuli.feature_tokens import TokenBank

__all__ = ["TokenBank"]
