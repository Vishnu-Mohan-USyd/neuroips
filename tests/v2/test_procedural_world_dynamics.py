"""LOAD-BEARING: regime-transition and cause-switch rates match the spec.

The hidden regime ``g`` is what makes context memory C load-bearing in
Phase 2 (Gate 7): the circuit has to infer g from history because it is
invisible in any single frame. For that story to hold, the transition
kernel itself must match the spec:

* ``P(g_{t+1} = g_t) = 0.98`` — measured over 10 000 steps, tolerance ±0.01.
* Per-regime cause-switch (``is_jump``) rates match the spec within ±0.01:
  CW-drift=0.05, CCW-drift=0.05, low-hazard=0.05, high-hazard=0.30.
"""

from __future__ import annotations

import pytest

from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world import (
    JUMP_PROBS, REGIME_PERSIST_PROB, REGIMES, ProceduralWorld,
)


@pytest.fixture(scope="module")
def bank():
    from src.v2_model.config import ModelConfig
    return TokenBank(ModelConfig(), seed=0)


@pytest.fixture
def world(cfg, bank):
    return ProceduralWorld(cfg, bank, seed_family="train")


def test_regime_persistence_rate(world: ProceduralWorld) -> None:
    """Empirical P(regime unchanged step-to-step) ≈ 0.98 ± 0.01."""
    state = world.reset(42)
    n_steps = 10_000
    n_same = 0
    for _ in range(n_steps):
        _, state_n, _ = world.step(state)
        if state_n.regime == state.regime:
            n_same += 1
        state = state_n
    observed = n_same / n_steps
    assert abs(observed - REGIME_PERSIST_PROB) < 0.01, (
        f"persistence {observed:.4f} deviates from spec "
        f"{REGIME_PERSIST_PROB:.4f} by more than 0.01"
    )


def test_per_regime_jump_rates(world: ProceduralWorld) -> None:
    """Cause-switch rate per regime matches ``JUMP_PROBS`` within ±0.01."""
    state = world.reset(42)
    n_steps = 20_000                        # more samples per regime
    regime_count = {r: 0 for r in REGIMES}
    jump_count = {r: 0 for r in REGIMES}
    for _ in range(n_steps):
        _, state, info = world.step(state)
        regime_count[info["regime"]] += 1
        if info["is_jump"]:
            jump_count[info["regime"]] += 1
    for r in REGIMES:
        assert regime_count[r] > 500, (
            f"regime {r!r} under-represented (n={regime_count[r]})"
        )
        observed = jump_count[r] / regime_count[r]
        expected = JUMP_PROBS[r]
        assert abs(observed - expected) < 0.015, (
            f"{r}: jump rate {observed:.4f} deviates from spec "
            f"{expected:.4f} by more than 0.015 (n={regime_count[r]})"
        )


def test_jump_resamples_z(world: ProceduralWorld) -> None:
    """On ``is_jump=True`` the identity z is re-sampled (≠ previous z most of
    the time); on ``is_jump=False`` z is always preserved."""
    state = world.reset(42)
    n_steps = 5000
    no_jump_z_preserved = 0
    no_jump_total = 0
    jump_total = 0
    for _ in range(n_steps):
        prev_z = state.z
        _, state, info = world.step(state)
        if info["is_jump"]:
            jump_total += 1
        else:
            no_jump_total += 1
            if state.z == prev_z:
                no_jump_z_preserved += 1
    # Spec: z unchanged during smooth drift.
    assert no_jump_z_preserved == no_jump_total, (
        f"smooth drift altered z in {no_jump_total - no_jump_z_preserved}"
        f" of {no_jump_total} non-jump steps"
    )
    assert jump_total > 100, f"not enough jumps observed ({jump_total})"


def test_drift_regimes_have_directional_theta(world: ProceduralWorld) -> None:
    """CW-drift mean Δθ ≈ +5°, CCW-drift ≈ -5° (over non-jump steps)."""
    state = world.reset(42)
    n_steps = 20_000
    sum_dtheta = {r: 0.0 for r in REGIMES}
    count = {r: 0 for r in REGIMES}
    for _ in range(n_steps):
        prev_theta = state.theta
        _, state, info = world.step(state)
        if info["is_jump"]:
            continue
        dtheta = (info["theta"] - prev_theta + 180.0) % 360.0 - 180.0
        sum_dtheta[info["regime"]] += dtheta
        count[info["regime"]] += 1
    mean = {r: sum_dtheta[r] / count[r] for r in REGIMES if count[r] > 0}
    assert mean["CW-drift"] > 4.0
    assert mean["CCW-drift"] < -4.0
    # Hazard regimes are zero-mean; tolerance comes from 20k-step std/√n.
    assert abs(mean["low-hazard"]) < 1.0
    assert abs(mean["high-hazard"]) < 1.0
