"""Persistence / decay invariants — the load-bearing test that proves C can
bridge Kok's 550 ms cue-delay interval.

With zero drive (achieved by zeroing `W_mm_gen` + passing zero H/q/leader input
or a properly-silenced φ at the origin), the update reduces to
``m_{t+1} = exp(-dt/τ_m) · m_t`` exactly (exact-ODE leak factor — endorsed in
the Task-16 GO review). Iterating 550 ms / 5 ms = 110 steps should leave the
memory content retrievable — i.e. still significantly above machine zero and
still monotonically correlated with the initial content.
"""

from __future__ import annotations

import math

import torch

from src.v2_model.context_memory import ContextMemory


def _cm(tau_m_ms: float = 500.0, dt_ms: float = 5.0) -> ContextMemory:
    cm = ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        tau_m_ms=tau_m_ms, dt_ms=dt_ms, seed=0,
    )
    # Zero-out the recurrence so the persistence analysis isolates pure leak.
    with torch.no_grad():
        cm.W_mm_gen.zero_()
    return cm


def test_zero_drive_decays_by_exact_leak_factor_one_step() -> None:
    """Single-step: m_next == exp(-dt/τ) · m_t when the drive sums to zero."""
    cm = _cm(tau_m_ms=500.0, dt_ms=5.0)
    B = 4
    m = torch.randn(B, cm.n_m)
    h_zero = torch.zeros(B, cm.n_h)                                # no H drive
    m_next, _ = cm(m, h_zero)                                      # q, leader = None
    expected = math.exp(-5.0 / 500.0) * m
    torch.testing.assert_close(m_next, expected, atol=1e-7, rtol=0.0)


def test_zero_drive_decay_over_550_ms_matches_closed_form() -> None:
    """After 110 steps of zero drive, m_n = exp(-n·dt/τ) · m_0 exactly.

    Iterating exp(-dt/τ) n times equals exp(-n·dt/τ) analytically; numerically
    the two may differ by a few ULPs of accumulated floating-point error, so a
    loose atol is used.
    """
    cm = _cm(tau_m_ms=500.0, dt_ms=5.0)
    B = 4
    m0 = torch.randn(B, cm.n_m)
    m = m0.clone()
    h_zero = torch.zeros(B, cm.n_h)
    n_steps = 110                                                  # 110 · 5 ms = 550 ms

    for _ in range(n_steps):
        m, _ = cm(m, h_zero)

    expected = math.exp(-n_steps * 5.0 / 500.0) * m0
    torch.testing.assert_close(m, expected, atol=1e-5, rtol=0.0)


def test_memory_content_still_retrievable_after_550_ms() -> None:
    """Key bridging claim: at 550 ms the signal is still a meaningful fraction
    of the initial magnitude and perfectly rank-preserves the initial content."""
    cm = _cm(tau_m_ms=500.0, dt_ms=5.0)
    B = 4
    m0 = torch.randn(B, cm.n_m)
    m = m0.clone()
    h_zero = torch.zeros(B, cm.n_h)

    for _ in range(110):                                            # 550 ms
        m, _ = cm(m, h_zero)

    # Closed-form amplitude ratio = exp(-550/500) = exp(-1.1) ≈ 0.333 — well above noise.
    ratio = (m.norm() / m0.norm()).item()
    assert 0.30 < ratio < 0.36, f"ratio={ratio} outside expected 0.30–0.36 band"

    # Rank-preservation: a scalar decay cannot reshuffle sign or relative magnitude.
    # Correlation between initial and final content should be exactly 1.
    m_flat = m.flatten()
    m0_flat = m0.flatten()
    r = (m_flat @ m0_flat) / (m_flat.norm() * m0_flat.norm())
    torch.testing.assert_close(r, torch.tensor(1.0), atol=1e-6, rtol=0.0)


def test_decay_rate_scales_with_tau() -> None:
    """Larger τ ⇒ slower decay. Tests both bounds of the 300–800 ms plan range."""
    m0 = torch.randn(1, 16)
    h_zero = torch.zeros(1, 24)
    n_steps = 60                                                    # 300 ms

    results = {}
    for tau in (300.0, 800.0):
        cm = _cm(tau_m_ms=tau, dt_ms=5.0)
        m = m0.clone()
        for _ in range(n_steps):
            m, _ = cm(m, h_zero)
        results[tau] = (m.norm() / m0.norm()).item()

    # Stronger retention at τ=800 than τ=300.
    assert results[800.0] > results[300.0], (
        f"expected τ=800 retention > τ=300; got {results}"
    )
    # Closed-form sanity: exp(-300/800) ≈ 0.687, exp(-300/300) = exp(-1) ≈ 0.368.
    torch.testing.assert_close(
        torch.tensor(results[800.0]),
        torch.tensor(math.exp(-n_steps * 5.0 / 800.0)),
        atol=1e-5, rtol=0.0,
    )
    torch.testing.assert_close(
        torch.tensor(results[300.0]),
        torch.tensor(math.exp(-n_steps * 5.0 / 300.0)),
        atol=1e-5, rtol=0.0,
    )


def test_positive_initial_memory_stays_positive_under_zero_drive() -> None:
    """Leak factor is in (0, 1) — a non-negative m_t stays non-negative."""
    cm = _cm(tau_m_ms=500.0, dt_ms=5.0)
    m = torch.rand(3, cm.n_m) + 0.01                                # strictly positive
    h_zero = torch.zeros(3, cm.n_h)
    for _ in range(50):
        m, _ = cm(m, h_zero)
    assert (m > 0).all()


def test_memory_state_has_zero_norm_when_driven_only_by_recurrence_from_zero() -> None:
    """Stability check: starting at m=0 with zero external drive, the state
    remains at zero indefinitely (fixed point)."""
    cm = ContextMemory(
        n_m=16, n_h=24, n_cue=6, n_leader=7, n_out=12,
        tau_m_ms=500.0, dt_ms=5.0, seed=0,
    )
    # Do NOT zero out W_mm_gen here — recurrence is live, but starts from 0.
    B = 3
    m = torch.zeros(B, cm.n_m)
    h_zero = torch.zeros(B, cm.n_h)
    for _ in range(20):
        m, _ = cm(m, h_zero)
    # φ(0) = rectified_softplus(0) = 0 exactly; decay factor × 0 = 0.
    assert torch.all(m == 0.0)
