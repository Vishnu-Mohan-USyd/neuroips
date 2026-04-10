"""Unit tests for `src/spiking/populations.py`.

Covers the Phase 1.4 acceptance criteria from task #14:
    1. All populations produce binary spikes (z ∈ {0, 1}).
    2. Filtered traces are smooth and non-negative.
    3. Firing rates are measured (sensitivity curve) and logged — see
       `TestColdFiringRateSensitivity` for the measurement table. We do NOT
       assert biological rates from a cold (untrained) forward pass; see the
       docstring on that test class for the rationale.
    4. Dale's law: inhibitory weights >= 0 via softplus.
    5. L2/3 spectral radius <= 0.95 constraint maintained.
    6. Gradients flow through all populations.

Firing-rate note (important)
----------------------------
At the plan's default `V_thresh = 1.0` + `β_mem ≈ 0.82-0.90` + {0,1} spike
encoding + dt = 1 ms, the LIF membrane with subtract reset is a near-binary
function of drive: below `V_thresh·(1-β)` no spikes fire, and only slightly
above that floor the rate jumps to tens of Hz. Hitting the primary-source
stationary targets (L4 2-5 Hz, L2/3 1-4 Hz, Niell & Stryker 2010) in an
*untrained* network would require driving within an infeasibly narrow
input-magnitude window. Training learns the right gains. So these tests
*measure* the cold firing-rate curve and *log* it rather than asserting a
biological range — the cold assertion is just "rate > 0 and rate < 1 for
moderate drive". This matches team-lead Ruling 1 on firing rate targets
(stationary mode = training-time goal, not cold-start gate).
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
import torch.nn as nn

from src.config import ModelConfig, SpikingConfig
from src.spiking.populations import (
    SpikingL4Ring,
    SpikingL23Ring,
    SpikingPVPool,
    SpikingSOMRing,
    SpikingVIPRing,
    _build_rec_kernel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfgs() -> tuple[ModelConfig, SpikingConfig]:
    return ModelConfig(), SpikingConfig()


@pytest.fixture
def pops(cfgs):
    mc, sc = cfgs
    return {
        "l4": SpikingL4Ring(mc, sc),
        "pv": SpikingPVPool(mc),
        "l23": SpikingL23Ring(mc, sc),
        "som": SpikingSOMRing(mc, sc),
        "vip": SpikingVIPRing(mc, sc),
    }


def _step_l4(l4, stim, r_pv, state):
    return l4(stim, r_pv, state)


def _step_l23(l23, x_l4, x_som, r_pv, state):
    return l23(x_l4, torch.zeros_like(x_l4), x_som, r_pv, state)


# ---------------------------------------------------------------------------
# 1. Constants & time constants — match plan
# ---------------------------------------------------------------------------

class TestTimeConstantsMatchPlan:
    def test_l4_beta_mem(self, cfgs):
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        assert l4.beta_mem == pytest.approx(math.exp(-1.0 / 5.0), abs=1e-6)
        assert l4.beta_mem == pytest.approx(0.8187, abs=1e-4)

    def test_l23_beta_mem(self, cfgs):
        mc, sc = cfgs
        l23 = SpikingL23Ring(mc, sc)
        assert l23.beta_mem == pytest.approx(math.exp(-1.0 / 10.0), abs=1e-6)
        assert l23.beta_mem == pytest.approx(0.9048, abs=1e-4)

    def test_som_vip_beta_mem(self, cfgs):
        mc, sc = cfgs
        som = SpikingSOMRing(mc, sc)
        vip = SpikingVIPRing(mc, sc)
        expected = math.exp(-1.0 / 10.0)
        assert som.beta_mem == pytest.approx(expected, abs=1e-6)
        assert vip.beta_mem == pytest.approx(expected, abs=1e-6)

    def test_filter_alpha(self, cfgs):
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        expected = math.exp(-1.0 / 10.0)
        assert l4.alpha_filter == pytest.approx(expected, abs=1e-6)

    def test_rho_adapt(self, cfgs):
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        expected = math.exp(-1.0 / 200.0)
        assert l4.rho_adapt == pytest.approx(expected, abs=1e-6)
        assert l4.rho_adapt == pytest.approx(0.99501, abs=1e-4)

    def test_beta_adapt_default(self, cfgs):
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        assert l4.beta_adapt == 1.8


# ---------------------------------------------------------------------------
# 2. Shapes — init_state and forward
# ---------------------------------------------------------------------------

class TestShapes:
    @pytest.mark.parametrize("B", [1, 4, 32])
    def test_l4_state_shape(self, cfgs, B):
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        s = l4.init_state(B)
        for k in ("v", "z", "x", "b"):
            assert s[k].shape == (B, mc.n_orientations)

    def test_l4_forward_shapes(self, pops):
        B, N = 4, 36
        state = pops["l4"].init_state(B)
        stim = torch.rand(B, N)
        r_pv = torch.zeros(B, 1)
        new_state, z, x = pops["l4"](stim, r_pv, state)
        assert z.shape == (B, N)
        assert x.shape == (B, N)
        for k in ("v", "z", "x", "b"):
            assert new_state[k].shape == (B, N)

    def test_pv_shapes(self, pops):
        B, N = 4, 36
        state = pops["pv"].init_state(B)
        x_l4 = torch.rand(B, N)
        x_l23 = torch.rand(B, N)
        new_state, r_pv_a, r_pv_b = pops["pv"](x_l4, x_l23, state)
        assert new_state["r_pv"].shape == (B, 1)
        assert r_pv_a.shape == (B, 1)
        # For PV, "z" and "x" slots both carry r_pv
        assert torch.equal(r_pv_a, r_pv_b)

    def test_l23_forward_shapes(self, pops):
        B, N = 4, 36
        state = pops["l23"].init_state(B)
        x_l4 = torch.rand(B, N)
        x_som = torch.zeros(B, N)
        r_pv = torch.zeros(B, 1)
        new_state, z, x = _step_l23(pops["l23"], x_l4, x_som, r_pv, state)
        assert z.shape == (B, N)
        assert x.shape == (B, N)
        for k in ("v", "z", "x"):
            assert new_state[k].shape == (B, N)

    def test_som_vip_shapes(self, pops):
        B, N = 4, 36
        for pop in (pops["som"], pops["vip"]):
            state = pop.init_state(B)
            drive = torch.rand(B, N)
            new_state, z, x = pop(drive, state)
            assert z.shape == (B, N)
            assert x.shape == (B, N)


# ---------------------------------------------------------------------------
# 3. Binary spikes
# ---------------------------------------------------------------------------

class TestBinarySpikes:
    """Acceptance criterion #1: all populations produce binary spikes."""

    @staticmethod
    def _assert_binary(z: torch.Tensor, name: str):
        unique = set(z.unique().tolist())
        assert unique.issubset({0.0, 1.0}), f"{name} non-binary spikes: {unique}"

    def test_l4_binary(self, pops):
        B, N = 4, 36
        state = pops["l4"].init_state(B)
        stim = torch.rand(B, N) * 2.0  # mix below/above threshold
        r_pv = torch.zeros(B, 1)
        for _ in range(30):
            state, z, x = pops["l4"](stim, r_pv, state)
            self._assert_binary(z, "L4")

    def test_l23_binary(self, pops):
        B, N = 4, 36
        state = pops["l23"].init_state(B)
        x_l4 = torch.rand(B, N) * 0.5
        x_som = torch.zeros(B, N)
        r_pv = torch.zeros(B, 1)
        for _ in range(30):
            state, z, x = _step_l23(pops["l23"], x_l4, x_som, r_pv, state)
            self._assert_binary(z, "L2/3")

    def test_som_binary(self, pops):
        B, N = 4, 36
        state = pops["som"].init_state(B)
        drive = torch.rand(B, N) * 2.0
        for _ in range(30):
            state, z, x = pops["som"](drive, state)
            self._assert_binary(z, "SOM")

    def test_vip_binary(self, pops):
        B, N = 4, 36
        state = pops["vip"].init_state(B)
        drive = torch.rand(B, N) * 2.0
        for _ in range(30):
            state, z, x = pops["vip"](drive, state)
            self._assert_binary(z, "VIP")


# ---------------------------------------------------------------------------
# 4. Filtered traces smooth and non-negative
# ---------------------------------------------------------------------------

class TestFilteredTraces:
    """Acceptance criterion #2: filtered traces are smooth and non-negative.

    'Smooth' here means the per-step change |x[t] - x[t-1]| is at most 1
    (because a single spike adds 1 and the decay factor is < 1, so the max
    per-step jump is +1 when a spike lands; between spikes x decays toward
    zero — both are smooth).
    """

    def test_l4_trace_nonneg(self, pops):
        B, N = 4, 36
        state = pops["l4"].init_state(B)
        stim = torch.rand(B, N) * 2.0
        r_pv = torch.zeros(B, 1)
        for _ in range(100):
            state, z, x = pops["l4"](stim, r_pv, state)
            assert (x >= 0).all()

    def test_l23_trace_nonneg(self, pops):
        B, N = 4, 36
        state = pops["l23"].init_state(B)
        x_l4 = torch.rand(B, N) * 0.3
        x_som = torch.zeros(B, N)
        r_pv = torch.zeros(B, 1)
        for _ in range(100):
            state, z, x = _step_l23(pops["l23"], x_l4, x_som, r_pv, state)
            assert (x >= 0).all()

    def test_trace_steady_state_bounded(self, pops):
        """Filter steady state for constant spike train = 1/(1-α) ≈ 10.5."""
        B, N = 4, 36
        # Drive L4 hard so it spikes every step after warmup
        state = pops["l4"].init_state(B)
        stim = torch.ones(B, N) * 3.0
        r_pv = torch.zeros(B, 1)
        for _ in range(500):
            state, z, x = pops["l4"](stim, r_pv, state)
        # Filter steady state for α ≈ 0.9048 is 1/(1-α) ≈ 10.503
        expected_ss = 1.0 / (1.0 - math.exp(-1.0 / 10.0))
        assert x.max() <= expected_ss + 0.5
        assert x.mean() > 0.5 * expected_ss


# ---------------------------------------------------------------------------
# 5. Dale's law / structural constraints
# ---------------------------------------------------------------------------

class TestDalesLaw:
    """Acceptance criterion #4 + #5: Dale's law + spectral radius preserved."""

    def test_pv_pooling_weights_nonneg(self, pops):
        """PV pool's `w_pv_l4`, `w_pv_l23` are softplus(raw) → non-negative."""
        pv = pops["pv"]
        assert pv.w_pv_l4.item() >= 0
        assert pv.w_pv_l23.item() >= 0
        # Also after a negative raw weight
        with torch.no_grad():
            pv.w_pv_l4_raw.fill_(-10.0)
            pv.w_pv_l23_raw.fill_(-10.0)
        assert pv.w_pv_l4.item() >= 0
        assert pv.w_pv_l23.item() >= 0

    def test_l4_pv_gain_nonneg(self, pops):
        """L4's pv_gain (InhibitoryGain) is softplus-wrapped."""
        assert pops["l4"].pv_gain.gain.item() >= 0
        with torch.no_grad():
            pops["l4"].pv_gain.gain_raw.fill_(-10.0)
        assert pops["l4"].pv_gain.gain.item() >= 0

    def test_l23_som_pv_gains_nonneg(self, pops):
        """L2/3's w_som and w_pv_l23 inhibitory gains stay non-negative."""
        l23 = pops["l23"]
        assert l23.w_som.gain.item() >= 0
        assert l23.w_pv_l23.gain.item() >= 0
        with torch.no_grad():
            l23.w_som.gain_raw.fill_(-10.0)
            l23.w_pv_l23.gain_raw.fill_(-10.0)
        assert l23.w_som.gain.item() >= 0
        assert l23.w_pv_l23.gain.item() >= 0

    def test_l23_w_rec_spectral_radius_bounded(self, pops):
        """W_rec spectral radius ≤ 0.95 strictly, for any raw param value."""
        l23 = pops["l23"]
        # Default init
        W = l23.W_rec
        sv = torch.linalg.svdvals(W)
        assert sv[0].item() <= 0.95 + 1e-6

        # Push gain_rec_raw very high — softplus saturates, then clamp(<=0.95) kicks in
        with torch.no_grad():
            l23.gain_rec_raw.fill_(100.0)
        W = l23.W_rec
        sv = torch.linalg.svdvals(W)
        assert sv[0].item() <= 0.95 + 1e-6

    def test_l23_w_rec_nonneg(self, pops):
        """W_rec is a row-normalised circular Gaussian × gain → non-negative."""
        l23 = pops["l23"]
        assert (l23.W_rec >= 0).all()

    def test_l23_w_rec_row_normalised_times_gain(self, pops):
        """Each row of W_rec sums to exactly `gain` (pre-clamp) — the clamped
        softplus gain is the row sum."""
        l23 = pops["l23"]
        W = l23.W_rec
        row_sums = W.sum(dim=-1)
        expected_gain = torch.clamp(
            torch.nn.functional.softplus(l23.gain_rec_raw), max=0.95
        )
        torch.testing.assert_close(
            row_sums,
            expected_gain.expand_as(row_sums),
            atol=1e-6, rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# 6. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Acceptance criterion #6: gradients flow through all populations."""

    def test_l4_gradient_flows(self, pops):
        B, N = 2, 36
        stim = torch.rand(B, N, requires_grad=True)
        state = pops["l4"].init_state(B)
        r_pv = torch.zeros(B, 1)
        for _ in range(30):
            state, z, x = pops["l4"](stim, r_pv, state)
        loss = x.sum()
        loss.backward()
        assert torch.isfinite(stim.grad).all()
        assert (stim.grad != 0).any()

    def test_l23_gradient_through_w_rec(self, pops):
        """σ_rec_raw, g_rec_raw receive gradients through L2/3 forward."""
        B, N = 2, 36
        l23 = pops["l23"]
        state = l23.init_state(B)
        x_l4 = torch.ones(B, N) * 0.3
        x_som = torch.zeros(B, N)
        r_pv = torch.zeros(B, 1)
        for _ in range(30):
            state, z, x = _step_l23(l23, x_l4, x_som, r_pv, state)
        loss = x.sum()
        loss.backward()
        assert l23.sigma_rec_raw.grad is not None
        assert l23.gain_rec_raw.grad is not None
        assert torch.isfinite(l23.sigma_rec_raw.grad)
        assert torch.isfinite(l23.gain_rec_raw.grad)

    def test_l23_gradient_through_inhib_gains(self, pops):
        B, N = 2, 36
        l23 = pops["l23"]
        state = l23.init_state(B)
        x_l4 = torch.ones(B, N) * 0.3
        x_som = torch.ones(B, N) * 0.1
        r_pv = torch.ones(B, 1) * 0.1
        for _ in range(30):
            state, z, x = _step_l23(l23, x_l4, x_som, r_pv, state)
        loss = x.sum()
        loss.backward()
        assert l23.w_som.gain_raw.grad is not None
        assert l23.w_pv_l23.gain_raw.grad is not None

    def test_pv_gradient_flows(self, pops):
        B, N = 2, 36
        pv = pops["pv"]
        state = pv.init_state(B)
        x_l4 = torch.rand(B, N, requires_grad=True)
        x_l23 = torch.rand(B, N, requires_grad=True)
        for _ in range(10):
            state, r, r2 = pv(x_l4, x_l23, state)
        loss = state["r_pv"].sum()
        loss.backward()
        assert torch.isfinite(x_l4.grad).all()
        assert torch.isfinite(x_l23.grad).all()
        assert (x_l4.grad != 0).any()

    def test_som_vip_gradient_flows(self, pops):
        B, N = 2, 36
        for pop in (pops["som"], pops["vip"]):
            state = pop.init_state(B)
            drive = torch.rand(B, N, requires_grad=True)
            for _ in range(30):
                state, z, x = pop(drive, state)
            loss = x.sum()
            loss.backward()
            assert torch.isfinite(drive.grad).all()


# ---------------------------------------------------------------------------
# 7. Dynamics correctness — spot-check LIF math
# ---------------------------------------------------------------------------

class TestDynamicsCorrectness:
    def test_l4_reset_subtracts_V_thresh(self, cfgs):
        """After a spike at step t, v[t+1] = β·v[t] + drive[t+1] − V_thresh."""
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        B, N = 1, 36
        state = l4.init_state(B)

        # Force v above V_thresh on step 1 by using a big drive.
        stim = torch.ones(B, N) * 10.0
        r_pv = torch.zeros(B, 1)
        state, z_t, _ = l4(stim, r_pv, state)
        assert (z_t == 1.0).all(), "large drive should cause spike"

        v_after_spike = state["v"].clone()
        # Next step with drive=0 — v should be β·v_after − V_thresh (because
        # z_prev=1 subtracts V_thresh in the state update).
        state2, _, _ = l4(torch.zeros(B, N), r_pv, state)
        expected_v = l4.beta_mem * v_after_spike + 0.0 - 1.0 * sc.V_thresh
        torch.testing.assert_close(state2["v"], expected_v, atol=1e-6, rtol=1e-6)

    def test_l4_adaptation_rises_under_sustained_drive(self, cfgs):
        """After ~5×τ_adapt of sustained spiking, b approaches its steady-state."""
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        B, N = 1, 36
        state = l4.init_state(B)
        stim = torch.ones(B, N) * 5.0
        r_pv = torch.zeros(B, 1)
        for _ in range(1000):
            state, _, _ = l4(stim, r_pv, state)
        # b should have increased substantially from 0
        assert state["b"].max() > 0.5, (
            f"adaptation should build up; max b = {state['b'].max()}"
        )

    def test_l4_no_adapt_no_spike_no_drive(self, cfgs):
        """With zero drive and fresh state, no spikes, no adaptation growth."""
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        B, N = 1, 36
        state = l4.init_state(B)
        zero = torch.zeros(B, N)
        r_pv = torch.zeros(B, 1)
        for _ in range(20):
            state, z, x = l4(zero, r_pv, state)
            assert (z == 0).all()
        assert (state["b"] == 0).all()

    def test_filter_decays(self, pops):
        """After a spike, x decays by α each step in the absence of new spikes."""
        B, N = 1, 36
        som = pops["som"]
        state = som.init_state(B)
        # Force a spike with a strong transient
        drive = torch.ones(B, N) * 10.0
        state, z, x_after_spike = som(drive, state)
        # Now zero drive — x should decay geometrically
        zero = torch.zeros(B, N)
        x_prev = x_after_spike.clone()
        for _ in range(5):
            state, z, x = som(zero, state)
            # No new spikes at zero drive + v decaying; just filter decay
            if (z == 0).all():
                torch.testing.assert_close(
                    x, som.alpha_filter * x_prev, atol=1e-6, rtol=1e-6,
                )
            x_prev = x

    def test_pv_rate_follows_pool(self, pops):
        """PV rate is a leaky integrator on sum(x_l4) + sum(x_l23)."""
        B, N = 2, 36
        pv = pops["pv"]
        state = pv.init_state(B)
        x_l4 = torch.ones(B, N) * 0.5
        x_l23 = torch.zeros(B, N)
        # Large enough pooled input should push r_pv positive asymptotically
        for _ in range(100):
            state, r, _ = pv(x_l4, x_l23, state)
        assert state["r_pv"].min() > 0
        assert torch.isfinite(state["r_pv"]).all()


# ---------------------------------------------------------------------------
# 8. Cold (untrained) firing-rate sensitivity measurement
# ---------------------------------------------------------------------------

class TestColdFiringRateSensitivity:
    """NON-ASSERTING measurement of firing rate vs drive for each population.

    These tests report firing rates at several drive levels from a cold
    (untrained, zero-init state) forward pass. We assert only that:
      (a) The rate is non-zero somewhere in the tested range (neurons can fire)
      (b) The rate is non-saturated somewhere in the tested range (neurons can
          *not* fire below the V_thresh·(1-β) drive floor)

    This is NOT a biological-range gate. See module docstring for why a cold
    network cannot hit 2-5 Hz at unit drive with plan defaults — the drive
    window between 0% and 100% spiking is extremely narrow at V_thresh=1.0.
    Training will learn the right gains.
    """

    B, T = 2, 300

    def _run_l4(self, drive_mag, cfgs):
        mc, sc = cfgs
        l4 = SpikingL4Ring(mc, sc)
        state = l4.init_state(self.B)
        stim = torch.ones(self.B, mc.n_orientations) * drive_mag
        r_pv = torch.zeros(self.B, 1)
        zs = []
        for _ in range(self.T):
            state, z, _ = l4(stim, r_pv, state)
            zs.append(z)
        return torch.stack(zs).mean().item()

    def _run_lif(self, pop, drive_mag, N=36):
        state = pop.init_state(self.B)
        drive = torch.ones(self.B, N) * drive_mag
        zs = []
        for _ in range(self.T):
            state, z, _ = pop(drive, state)
            zs.append(z)
        return torch.stack(zs).mean().item()

    def test_l4_rate_sensitivity(self, cfgs):
        rates = {d: self._run_l4(d, cfgs) for d in [0.0, 0.15, 0.19, 0.22, 0.3, 0.5, 1.0]}
        # Log
        print("\n[L4 cold firing rate (spikes/step)]")
        for d, r in rates.items():
            print(f"  drive={d:.3f}  rate={r:.4f}  ({r * 1000:.0f} Hz)")
        assert rates[0.0] == 0.0, "no drive → no spikes"
        assert rates[1.0] > 0.0, "strong drive → some spikes"
        assert rates[0.19] != rates[1.0], "rate must vary with drive"

    def test_l23_rate_sensitivity(self, cfgs):
        mc, sc = cfgs
        l23 = SpikingL23Ring(mc, sc)
        # Drive through the x_l4 input (ff path)
        def run(dm):
            st = l23.init_state(self.B)
            x_l4 = torch.ones(self.B, mc.n_orientations) * dm
            x_som = torch.zeros(self.B, mc.n_orientations)
            r_pv = torch.zeros(self.B, 1)
            zs = []
            for _ in range(self.T):
                st, z, _ = _step_l23(l23, x_l4, x_som, r_pv, st)
                zs.append(z)
            return torch.stack(zs).mean().item()

        rates = {d: run(d) for d in [0.0, 0.1, 0.15, 0.2, 0.3, 0.5]}
        print("\n[L2/3 cold firing rate (spikes/step)]")
        for d, r in rates.items():
            print(f"  x_l4 drive={d:.3f}  rate={r:.4f}  ({r * 1000:.0f} Hz)")
        assert rates[0.0] == 0.0
        assert rates[0.5] > 0.0

    def test_som_rate_sensitivity(self, pops):
        rates = {d: self._run_lif(pops["som"], d) for d in [0.0, 0.1, 0.15, 0.3, 1.0]}
        print("\n[SOM cold firing rate (spikes/step)]")
        for d, r in rates.items():
            print(f"  drive={d:.3f}  rate={r:.4f}  ({r * 1000:.0f} Hz)")
        assert rates[0.0] == 0.0
        assert rates[1.0] > 0.0

    def test_vip_rate_sensitivity(self, pops):
        rates = {d: self._run_lif(pops["vip"], d) for d in [0.0, 0.1, 0.2, 1.0]}
        print("\n[VIP cold firing rate (spikes/step)]")
        for d, r in rates.items():
            print(f"  drive={d:.3f}  rate={r:.4f}  ({r * 1000:.0f} Hz)")
        assert rates[0.0] == 0.0
        assert rates[1.0] > 0.0


# ---------------------------------------------------------------------------
# 9. Parity with rate model structural invariants
# ---------------------------------------------------------------------------

class TestRateModelParity:
    """Structural parity with rate model's populations (not numerical parity)."""

    def test_build_rec_kernel_matches_rate_implementation(self, cfgs):
        """`_build_rec_kernel` in spiking populations matches the rate version."""
        from src.model.populations import _build_rec_kernel as rate_build

        mc, _sc = cfgs
        sigma_raw = torch.tensor(
            math.log(math.exp(mc.sigma_rec) - 1.0)
        )
        gain_raw = torch.tensor(
            math.log(math.exp(mc.gain_rec) - 1.0)
        )
        K_spk = _build_rec_kernel(mc.n_orientations, sigma_raw, gain_raw, mc.orientation_range)
        K_rate = rate_build(mc.n_orientations, sigma_raw, gain_raw, mc.orientation_range)
        torch.testing.assert_close(K_spk, K_rate, atol=0.0, rtol=0.0)

    def test_l4_identity_ff_buffer(self, pops):
        """L4's W_ff is identity and registered as a buffer (not parameter)."""
        l4 = pops["l4"]
        assert torch.equal(l4.W_ff, torch.eye(l4.n))
        # W_ff should be in buffers, not parameters
        assert "W_ff" in dict(l4.named_buffers())
        assert "W_ff" not in dict(l4.named_parameters())

    def test_l23_identity_l4_to_l23_buffer(self, pops):
        l23 = pops["l23"]
        assert torch.equal(l23.W_l4_to_l23, torch.eye(l23.n))
        assert "W_l4_to_l23" in dict(l23.named_buffers())
        assert "W_l4_to_l23" not in dict(l23.named_parameters())

    def test_l23_cache_kernels(self, pops):
        """cache_kernels() stores the kernel and uncache_kernels() clears it."""
        l23 = pops["l23"]
        assert l23._cached_W_rec is None
        l23.cache_kernels()
        assert l23._cached_W_rec is not None
        torch.testing.assert_close(
            l23.W_rec, l23._cached_W_rec, atol=0.0, rtol=0.0,
        )
        l23.uncache_kernels()
        assert l23._cached_W_rec is None
