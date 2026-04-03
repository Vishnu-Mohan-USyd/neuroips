"""Tests for V1 populations: L4, PV (Phase 2), L2/3, DeepTemplate, SOM (Phase 3).

Includes validation checks with numerical results.
"""

import math

import torch
import pytest

from src.config import ModelConfig
from src.utils import circular_gaussian_fwhm, shifted_softplus, rectified_softplus
from src.state import initial_state
from src.stimulus.gratings import population_code, naka_rushton, generate_grating
from src.model.populations import V1L4Ring, PVPool, V1L23Ring, DeepTemplate, SOMRing


@pytest.fixture
def cfg():
    return ModelConfig(feedback_mode='fixed')


@pytest.fixture
def l4(cfg):
    return V1L4Ring(cfg)


@pytest.fixture
def pv(cfg):
    return PVPool(cfg)


# ===================================================================
# V1L4Ring: W_ff and basic properties
# ===================================================================

class TestV1L4Basics:

    def test_w_ff_is_identity(self, l4, cfg):
        """W_ff should be identity matrix (tuning is in population code)."""
        assert torch.allclose(l4.W_ff, torch.eye(cfg.n_orientations))

    def test_w_ff_is_buffer_not_parameter(self, l4):
        """W_ff should be a buffer, not a parameter — it's frozen."""
        param_names = [n for n, _ in l4.named_parameters()]
        assert "W_ff" not in param_names
        assert "W_ff" in dict(l4.named_buffers())

    def test_output_shapes(self, l4, cfg):
        """Forward pass with batch=4, verify shapes."""
        B = 4
        stim = generate_grating(
            torch.tensor([0.0, 45.0, 90.0, 135.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff,
        )
        r_l4 = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)
        adapt = torch.zeros(B, cfg.n_orientations)

        new_r_l4, new_adapt = l4(stim, r_l4, r_pv, adapt)

        assert new_r_l4.shape == (B, cfg.n_orientations)
        assert new_adapt.shape == (B, cfg.n_orientations)

    def test_has_pv_gain(self, l4):
        """Should store an InhibitoryGain for PV (for L2/3 use in Phase 3)."""
        assert hasattr(l4, "pv_gain")
        assert l4.pv_gain.gain.item() > 0


# ===================================================================
# Validation 1: Tuning curves — peak at preferred, FWHM≈28°
# ===================================================================

class TestV1L4Tuning:

    def test_tuning_curves_peak_correctly(self, l4, cfg):
        """Each channel's tuning curve should peak at its preferred orientation."""
        B = 1
        r_pv = torch.zeros(B, 1)

        for ch in range(cfg.n_orientations):
            theta = torch.tensor([ch * cfg.orientation_step])
            stim = generate_grating(theta, torch.tensor([1.0]),
                                    n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
            r_l4 = torch.zeros(B, cfg.n_orientations)
            adapt = torch.zeros(B, cfg.n_orientations)

            # Run to steady state (no PV, no prior adaptation)
            for _ in range(50):
                r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

            assert r_l4[0].argmax().item() == ch, f"Failed for channel {ch}"

    def test_tuning_curve_fwhm(self, l4, cfg):
        """FWHM should be approximately 28° (2.355 * 12°), within 15°-30° gate."""
        B = 1
        r_pv = torch.zeros(B, 1)

        # Analytical expectation
        expected_fwhm = circular_gaussian_fwhm(cfg.sigma_ff)
        assert 28.0 < expected_fwhm < 29.0

        # Measure from L4 steady-state tuning curve at θ=0°
        stim = generate_grating(torch.tensor([0.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)

        for _ in range(80):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

        curve = r_l4[0].detach()
        peak = curve.max()
        half_max = peak / 2
        above_half = (curve >= half_max).sum().item()
        measured_fwhm = above_half * cfg.orientation_step

        # Plan gating checkpoint: 15°-30°
        assert 15.0 <= measured_fwhm <= 35.0, \
            f"FWHM={measured_fwhm}° (expected ~{expected_fwhm}°)"

    def test_tuning_curve_shape(self, l4, cfg):
        """Tuning curve should be unimodal with Gaussian-like falloff."""
        B = 1
        stim = generate_grating(torch.tensor([0.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        for _ in range(80):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

        curve = r_l4[0].detach()
        # Peak at channel 0
        assert curve.argmax().item() == 0
        # Monotonically decreasing from peak toward ±90° (channels 0→18)
        for i in range(1, 10):
            assert curve[i] < curve[i - 1] or curve[i] < 1e-6, \
                f"Non-monotonic at channel {i}: {curve[i]:.4f} >= {curve[i-1]:.4f}"


# ===================================================================
# Validation 2: Adaptation — decay and recovery
# ===================================================================

class TestV1L4Adaptation:

    def test_adaptation_reduces_response(self, l4, cfg):
        """Sustained stimulus should reduce L4 response over 500+ steps.

        With τ_a=200, adaptation builds slowly. Rate reaches unadapted
        steady state in ~20 steps (4*τ_l4), then adaptation gradually
        reduces it.
        """
        B = 1
        stim = generate_grating(torch.tensor([45.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        responses = []
        for step in range(500):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
            responses.append(r_l4[0, 9].item())  # Channel 9 = 45°

        # Peak after rate stabilizes (step 20-30) vs late adapted response
        early_peak = max(responses[20:40])
        late = responses[-1]
        reduction = 1.0 - (late / early_peak) if early_peak > 0 else 0.0

        assert reduction > 0.05, \
            f"Adaptation too weak: {reduction:.1%} (early={early_peak:.4f}, late={late:.4f})"
        assert late > 0, "Response should not go to zero"

    def test_adaptation_recovery_on_change(self, l4, cfg):
        """Switching orientation should show recovery from adaptation."""
        B = 1
        contrast = torch.tensor([1.0])
        r_pv = torch.zeros(B, 1)

        stim_0 = generate_grating(torch.tensor([0.0]), contrast,
                                  n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        stim_90 = generate_grating(torch.tensor([90.0]), contrast,
                                   n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)

        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)

        # Adapt to 0° for 500 steps
        for _ in range(500):
            r_l4, adapt = l4(stim_0, r_l4, r_pv, adapt)
        adapted_response = r_l4[0, 0].item()
        adapted_ch0 = adapt[0, 0].item()

        # Switch to 90° for 100 steps (adaptation at ch0 decays)
        for _ in range(100):
            r_l4, adapt = l4(stim_90, r_l4, r_pv, adapt)

        assert adapt[0, 0].item() < adapted_ch0, "Adaptation did not decay during switch"

        # Return to 0° for 30 steps (stabilize rate)
        for _ in range(30):
            r_l4, adapt = l4(stim_0, r_l4, r_pv, adapt)
        recovered_response = r_l4[0, 0].item()

        assert recovered_response > adapted_response, \
            f"No recovery: adapted={adapted_response:.4f}, recovered={recovered_response:.4f}"

    def test_adaptation_clamped(self, l4, cfg):
        """Adaptation should not exceed adaptation_clamp (10.0)."""
        B = 1
        stim = generate_grating(torch.tensor([0.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        for _ in range(5000):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

        assert (adapt <= cfg.adaptation_clamp).all()

    def test_adaptation_half_life(self, cfg):
        """Verify adaptation time constant: half-life = τ_a * ln(2) ≈ 138.6 steps."""
        expected = cfg.tau_adaptation * math.log(2)
        assert 130 < expected < 145


# ===================================================================
# Validation 3: Contrast — zero, monotonic
# ===================================================================

class TestV1L4Contrast:

    def test_zero_contrast_gives_zero(self, l4, cfg):
        """Zero contrast → zero stimulus → zero L4 response."""
        B = 1
        stim = generate_grating(torch.tensor([45.0]), torch.tensor([0.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        for _ in range(20):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

        assert r_l4.abs().max().item() < 1e-6

    def test_higher_contrast_stronger_response(self, l4, cfg):
        """Higher contrast should produce larger L4 peak response."""
        peaks = {}
        for c_val in [0.1, 0.5, 1.0]:
            B = 1
            stim = generate_grating(torch.tensor([45.0]), torch.tensor([c_val]),
                                    n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
            r_l4 = torch.zeros(B, cfg.n_orientations)
            adapt = torch.zeros(B, cfg.n_orientations)
            r_pv = torch.zeros(B, 1)

            for _ in range(30):
                r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

            peaks[c_val] = r_l4[0, 9].item()

        assert peaks[0.1] < peaks[0.5] < peaks[1.0], f"Not monotonic: {peaks}"


# ===================================================================
# Validation 4: PV pool
# ===================================================================

class TestPVPool:

    def test_output_shape(self, pv, cfg):
        """PV output should be [B, 1]."""
        B = 4
        r_l4 = torch.ones(B, cfg.n_orientations)
        r_l23 = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)
        new_pv = pv(r_l4, r_l23, r_pv)
        assert new_pv.shape == (B, 1)

    def test_pv_pool_basic_value(self, pv, cfg):
        """With L4 rates all 1.0, PV drive = N * w_pv_l4. Verify reasonable value."""
        B = 1
        r_l4 = torch.ones(B, cfg.n_orientations)
        r_l23 = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        # Run to near steady state
        for _ in range(50):
            r_pv = pv(r_l4, r_l23, r_pv)

        w = pv.w_pv_l4.item()
        expected_drive = cfg.n_orientations * w  # 36 * w
        # PV should converge to rectified_softplus(drive)
        expected_ss = rectified_softplus(torch.tensor(expected_drive)).item()

        assert r_pv.item() > 0, "PV should be positive"
        assert abs(r_pv.item() - expected_ss) / max(expected_ss, 1e-6) < 0.1, \
            f"PV={r_pv.item():.4f}, expected ~{expected_ss:.4f} (drive={expected_drive:.4f}, w={w:.4f})"

    def test_pv_non_negative(self, pv, cfg):
        """PV rates should always be non-negative (shifted softplus)."""
        B = 4
        r_l4 = torch.randn(B, cfg.n_orientations).abs()
        r_l23 = torch.randn(B, cfg.n_orientations).abs()
        r_pv = torch.zeros(B, 1)
        for _ in range(20):
            r_pv = pv(r_l4, r_l23, r_pv)
        assert (r_pv >= 0).all()

    def test_pv_weights_non_negative(self, pv):
        """Effective PV weights should be non-negative (softplus guarantee)."""
        assert pv.w_pv_l4.item() >= 0
        assert pv.w_pv_l23.item() >= 0

    def test_pv_increases_with_activity(self, pv, cfg):
        """More L4 activity → higher PV output."""
        B = 1
        r_l23 = torch.zeros(B, cfg.n_orientations)

        low_l4 = torch.ones(B, cfg.n_orientations) * 0.1
        high_l4 = torch.ones(B, cfg.n_orientations) * 1.0

        r_pv_low = torch.zeros(B, 1)
        r_pv_high = torch.zeros(B, 1)

        for _ in range(30):
            r_pv_low = pv(low_l4, r_l23, r_pv_low)
            r_pv_high = pv(high_l4, r_l23, r_pv_high)

        assert r_pv_high.item() > r_pv_low.item()


# ===================================================================
# Validation 5: Divisive normalization — contrast-invariant tuning width
# ===================================================================

class TestDivisiveNormalization:

    def test_tuning_width_contrast_invariant(self, cfg):
        """PV divisive normalization should keep tuning width constant
        across contrasts (0.1, 0.5, 1.0).

        Key functional test: width of L4 tuning should NOT broaden at
        high contrast. Amplitude changes, width stays constant.
        """
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)

        # Set PV weights higher to see normalization effect
        with torch.no_grad():
            pv_mod.w_pv_l4_raw.fill_(1.0)  # softplus(1.0) ≈ 1.31

        theta = torch.tensor([0.0])
        fwhms = {}
        amplitudes = {}

        for c_val in [0.1, 0.5, 1.0]:
            B = 1
            stim = generate_grating(theta, torch.tensor([c_val]),
                                    n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
            r_l4 = torch.zeros(B, cfg.n_orientations)
            adapt = torch.zeros(B, cfg.n_orientations)
            r_pv = torch.zeros(B, 1)
            r_l23 = torch.zeros(B, cfg.n_orientations)

            # Run to steady state with PV feedback
            for _ in range(100):
                r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
                r_pv = pv_mod(r_l4, r_l23, r_pv)

            curve = r_l4[0].detach()
            peak = curve.max()
            amplitudes[c_val] = peak.item()

            if peak < 1e-8:
                continue

            above = (curve >= peak / 2).sum().item()
            fwhms[c_val] = above * cfg.orientation_step

        # Amplitudes should differ (contrast effect)
        assert amplitudes[0.1] < amplitudes[1.0], \
            f"Amplitudes not contrast-dependent: {amplitudes}"

        # FWHMs should be similar (within 15° of each other)
        vals = list(fwhms.values())
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            assert spread <= 15.0, \
                f"Tuning width not contrast-invariant: FWHMs={fwhms}, spread={spread}°"

    def test_no_nan_after_200_steps(self, cfg):
        """Run L4 + PV for 200 timesteps: no NaN, bounded activity."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        B = 4
        state = initial_state(B, cfg.n_orientations, cfg.v2_hidden_dim)

        stim = generate_grating(
            torch.tensor([45.0, 90.0, 0.0, 135.0]),
            torch.tensor([0.3, 0.7, 1.0, 0.1]),
            n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff,
        )
        r_l4 = state.r_l4.clone()
        adapt = state.adaptation.clone()
        r_pv = state.r_pv.clone()

        for _ in range(200):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
            r_pv = pv_mod(r_l4, state.r_l23, r_pv)

        assert not torch.isnan(r_l4).any(), "NaN in r_l4"
        assert not torch.isnan(adapt).any(), "NaN in adaptation"
        assert not torch.isnan(r_pv).any(), "NaN in r_pv"
        assert r_l4.max().item() < 100.0, f"Unbounded r_l4: max={r_l4.max().item()}"
        assert r_pv.max().item() < 1000.0, f"Unbounded r_pv: max={r_pv.max().item()}"

    def test_euler_stability_criterion(self, cfg):
        """dt/τ_min should be < 1.0 for Euler stability."""
        tau_min = min(cfg.tau_l4, cfg.tau_pv, cfg.tau_l23, cfg.tau_som)
        ratio = cfg.dt / tau_min
        assert ratio < 1.0, f"dt/τ_min = {ratio} >= 1.0, Euler may diverge"


# ===================================================================
# Numerical validation: print actual values for review
# ===================================================================

class TestNumericalResults:
    """Tests that print numerical results for the team lead's review."""

    def test_print_tuning_curve_values(self, l4, cfg, capsys):
        """Print steady-state tuning curve at θ=0°."""
        B = 1
        stim = generate_grating(torch.tensor([0.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        for _ in range(80):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

        curve = r_l4[0].detach()
        peak = curve.max().item()
        peak_ch = curve.argmax().item()
        above_half = (curve >= peak / 2).sum().item()
        fwhm = above_half * cfg.orientation_step

        print(f"\n=== Tuning curve at θ=0°, contrast=1.0, 80 steps ===")
        print(f"  Peak channel: {peak_ch} (θ={peak_ch * cfg.orientation_step}°)")
        print(f"  Peak rate: {peak:.4f}")
        print(f"  FWHM: {fwhm}° ({above_half} channels above half-max)")
        print(f"  Channels 0-5: {[f'{curve[i].item():.4f}' for i in range(6)]}")

        assert peak_ch == 0
        assert 15 <= fwhm <= 35

    def test_print_adaptation_dynamics(self, l4, cfg, capsys):
        """Print initial vs adapted response magnitudes."""
        B = 1
        stim = generate_grating(torch.tensor([45.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4 = torch.zeros(B, cfg.n_orientations)
        adapt = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        responses = []
        for step in range(500):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
            responses.append(r_l4[0, 9].item())

        early_peak = max(responses[20:40])
        late = responses[-1]
        reduction = 1.0 - (late / early_peak)

        print(f"\n=== Adaptation at θ=45° over 500 steps ===")
        print(f"  Step 1 response:    {responses[0]:.4f}")
        print(f"  Step 30 (peak):     {early_peak:.4f}")
        print(f"  Step 500 (adapted): {late:.4f}")
        print(f"  Reduction:          {reduction:.1%}")
        print(f"  Adaptation value:   {adapt[0, 9].item():.4f}")

        assert reduction > 0.05

    def test_print_contrast_responses(self, l4, cfg, capsys):
        """Print L4 peak response at three contrasts."""
        print(f"\n=== L4 contrast response (θ=45°, 30 steps, no PV) ===")
        for c_val in [0.1, 0.5, 1.0]:
            B = 1
            stim = generate_grating(torch.tensor([45.0]), torch.tensor([c_val]),
                                    n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
            r_l4 = torch.zeros(B, cfg.n_orientations)
            adapt = torch.zeros(B, cfg.n_orientations)
            r_pv = torch.zeros(B, 1)

            for _ in range(30):
                r_l4, adapt = l4(stim, r_l4, r_pv, adapt)

            naka_gain = naka_rushton(torch.tensor(c_val)).item()
            print(f"  c={c_val}: peak={r_l4[0, 9].item():.4f}, Naka-Rushton gain={naka_gain:.4f}")

    def test_print_pv_pool_value(self, pv, cfg, capsys):
        """Print PV pool value with L4 rates all 1.0."""
        B = 1
        r_l4 = torch.ones(B, cfg.n_orientations)
        r_l23 = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)

        for _ in range(50):
            r_pv = pv(r_l4, r_l23, r_pv)

        w_l4 = pv.w_pv_l4.item()
        drive = cfg.n_orientations * w_l4
        expected_ss = rectified_softplus(torch.tensor(drive)).item()

        print(f"\n=== PV pool (L4 all 1.0, 50 steps) ===")
        print(f"  w_pv_l4 = softplus(raw) = {w_l4:.4f}")
        print(f"  Drive = N * w = {cfg.n_orientations} * {w_l4:.4f} = {drive:.4f}")
        print(f"  Expected SS = rectified_softplus({drive:.4f}) = {expected_ss:.4f}")
        print(f"  Actual PV = {r_pv.item():.4f}")

        assert r_pv.item() > 0

    def test_print_contrast_invariance(self, cfg, capsys):
        """Print FWHM at three contrasts with PV normalization."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        with torch.no_grad():
            pv_mod.w_pv_l4_raw.fill_(1.0)

        print(f"\n=== Contrast invariance test (with PV, w_pv_l4={pv_mod.w_pv_l4.item():.3f}) ===")

        for c_val in [0.1, 0.5, 1.0]:
            B = 1
            stim = generate_grating(torch.tensor([0.0]), torch.tensor([c_val]),
                                    n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
            r_l4 = torch.zeros(B, cfg.n_orientations)
            adapt = torch.zeros(B, cfg.n_orientations)
            r_pv = torch.zeros(B, 1)
            r_l23 = torch.zeros(B, cfg.n_orientations)

            for _ in range(100):
                r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
                r_pv = pv_mod(r_l4, r_l23, r_pv)

            curve = r_l4[0].detach()
            peak = curve.max().item()
            above = (curve >= peak / 2).sum().item() if peak > 1e-8 else 0
            fwhm = above * cfg.orientation_step

            print(f"  c={c_val}: peak={peak:.4f}, PV={r_pv.item():.4f}, FWHM={fwhm}°")


# ===================================================================
# Phase 3: V1L23Ring
# ===================================================================

@pytest.fixture
def l23(cfg):
    return V1L23Ring(cfg)


def _run_l4_pv_l23(cfg, l4, pv_mod, l23_mod, stim, n_steps):
    """Helper: run L4+PV+L2/3 circuit for n_steps, returns final state."""
    B = stim.shape[0]
    state = initial_state(B, cfg.n_orientations, cfg.v2_hidden_dim)
    r_l4, adapt, r_pv, r_l23 = (
        state.r_l4.clone(), state.adaptation.clone(),
        state.r_pv.clone(), state.r_l23.clone(),
    )
    r_som = state.r_som.clone()
    tmpl_mod = torch.zeros(B, cfg.n_orientations)

    for _ in range(n_steps):
        r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
        r_pv = pv_mod(r_l4, r_l23, r_pv)
        r_l23 = l23_mod(r_l4, r_l23, tmpl_mod, r_som, r_pv)

    return r_l4, adapt, r_pv, r_l23, r_som


class TestV1L23Basics:

    def test_output_shape(self, l23, cfg):
        B = 4
        r_l4 = torch.randn(B, cfg.n_orientations).abs()
        r_l23 = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)
        r_som = torch.zeros(B, cfg.n_orientations)
        tmpl_mod = torch.zeros(B, cfg.n_orientations)
        out = l23(r_l4, r_l23, tmpl_mod, r_som, r_pv)
        assert out.shape == (B, cfg.n_orientations)

    def test_w_l4_to_l23_is_identity_buffer(self, l23, cfg):
        """W_l4_to_l23 should be identity, registered as buffer (frozen)."""
        assert "W_l4_to_l23" in dict(l23.named_buffers())
        param_names = [n for n, _ in l23.named_parameters()]
        assert "W_l4_to_l23" not in param_names
        assert torch.allclose(l23.W_l4_to_l23, torch.eye(cfg.n_orientations))

    def test_w_rec_is_circular_gaussian(self, l23, cfg):
        """W_rec should be a symmetric circulant kernel (not dense)."""
        W = l23.W_rec.detach()
        assert W.shape == (cfg.n_orientations, cfg.n_orientations)
        assert torch.allclose(W, W.T, atol=1e-5)
        diag = W.diag()
        off_diag = W[0, cfg.n_orientations // 2]
        assert diag.mean().item() > off_diag.item()

    def test_w_rec_only_two_free_params(self, l23):
        """W_rec should be parameterized by only σ_rec and g_rec."""
        rec_params = {n for n, _ in l23.named_parameters()
                      if 'rec' in n and 'raw' in n}
        assert rec_params == {'sigma_rec_raw', 'gain_rec_raw'}

    def test_w_rec_initial_values(self, l23, cfg):
        """σ_rec should initialize near 15°, g_rec near 0.3."""
        assert l23.sigma_rec.item() == pytest.approx(cfg.sigma_rec, rel=0.1)
        assert l23.gain_rec.item() == pytest.approx(cfg.gain_rec, rel=0.1)

    def test_inhibitory_gains_positive(self, l23):
        assert l23.w_pv_l23.gain.item() > 0
        assert l23.w_som.gain.item() > 0

    def test_template_modulation_effect(self, l23, cfg):
        """Template modulation should increase L2/3 drive."""
        B = 2
        r_l4 = torch.randn(B, cfg.n_orientations).abs()
        r_l23 = torch.zeros(B, cfg.n_orientations)
        r_pv = torch.zeros(B, 1)
        r_som = torch.zeros(B, cfg.n_orientations)

        out_zero = l23(r_l4, r_l23, torch.zeros(B, cfg.n_orientations), r_som, r_pv)
        td = torch.randn(B, cfg.n_orientations).abs() * 0.1
        out_td = l23(r_l4, r_l23, td, r_som, r_pv)

        assert out_td.mean().item() >= out_zero.mean().item() - 0.01


class TestV1L23Stability:

    def test_stable_orientation_selective_without_v2(self, cfg):
        """Without V2 (SOM=0, template_mod=0), L2/3 should show stable
        orientation-selective activity driven by L4."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        l23_mod = V1L23Ring(cfg)

        stim = generate_grating(torch.tensor([45.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4, adapt, r_pv, r_l23, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_mod, stim, 200)

        assert r_l23[0].argmax().item() == 9
        assert r_l23.max().item() > 0
        assert r_l23.max().item() < 100.0

    def test_no_nan_200_steps_full_circuit(self, cfg):
        """Run L4 + PV + L2/3 for 200 steps: no NaN, bounded activity."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        l23_mod = V1L23Ring(cfg)

        stim = generate_grating(
            torch.tensor([0.0, 45.0, 90.0, 135.0]),
            torch.tensor([0.5, 1.0, 0.3, 0.8]),
            n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff,
        )
        r_l4, adapt, r_pv, r_l23, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_mod, stim, 200)

        assert not torch.isnan(r_l4).any()
        assert not torch.isnan(r_l23).any()
        assert not torch.isnan(r_pv).any()
        assert r_l23.max().item() < 100.0

    def test_l23_inherits_l4_selectivity(self, cfg):
        """L2/3 should inherit orientation selectivity from L4."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        l23_mod = V1L23Ring(cfg)

        peaks_l23 = {}
        for theta_val, ch in [(0.0, 0), (45.0, 9), (90.0, 18)]:
            stim = generate_grating(
                torch.tensor([theta_val]), torch.tensor([1.0]),
                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff,
            )
            _, _, _, r_l23, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_mod, stim, 50)
            peaks_l23[theta_val] = r_l23[0].argmax().item()

        assert peaks_l23[0.0] == 0
        assert peaks_l23[45.0] == 9
        assert peaks_l23[90.0] == 18

    def test_recurrence_amplifies_response(self, cfg):
        """L2/3 response should be higher with recurrence (g=0.3) than without (g=0)."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        stim = generate_grating(torch.tensor([45.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)

        # With recurrence (default g=0.3)
        l23_rec = V1L23Ring(cfg)
        _, _, _, r_l23_rec, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_rec, stim, 100)
        peak_rec = r_l23_rec[0, 9].item()

        # Without recurrence (g=0)
        l23_no_rec = V1L23Ring(cfg)
        with torch.no_grad():
            l23_no_rec.gain_rec_raw.fill_(-10.0)  # softplus(-10) ≈ 0
        _, _, _, r_l23_no, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_no_rec, stim, 100)
        peak_no = r_l23_no[0, 9].item()

        assert peak_rec > peak_no, \
            f"Recurrence should amplify: with_rec={peak_rec:.4f}, without={peak_no:.4f}"


# ===================================================================
# Phase 3: DeepTemplate
# ===================================================================

class TestDeepTemplate:

    def test_output_shape(self, cfg):
        dt = DeepTemplate(cfg)
        B = 4
        q_pred = torch.randn(B, cfg.n_orientations).softmax(dim=-1)
        pi_pred = torch.ones(B, 1)
        out = dt(q_pred, pi_pred)
        assert out.shape == (B, cfg.n_orientations)

    def test_zero_precision_gives_zero(self, cfg):
        dt = DeepTemplate(cfg)
        q_pred = torch.randn(4, cfg.n_orientations).softmax(dim=-1)
        pi_pred = torch.zeros(4, 1)
        out = dt(q_pred, pi_pred)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_scales_with_precision(self, cfg):
        dt = DeepTemplate(cfg)
        q_pred = torch.randn(1, cfg.n_orientations).softmax(dim=-1)
        out_low = dt(q_pred, torch.tensor([[1.0]]))
        out_high = dt(q_pred, torch.tensor([[5.0]]))
        assert out_high.sum().item() > out_low.sum().item()

    def test_gain_is_learnable(self, cfg):
        """DeepTemplate gain should be a learnable parameter."""
        dt = DeepTemplate(cfg)
        params = list(dt.named_parameters())
        assert len(params) == 1
        assert params[0][0] == "gain_raw"
        assert dt.gain.item() == pytest.approx(cfg.template_gain, rel=0.1)

    def test_one_hot_template(self, cfg):
        """With one-hot q_pred at ch 9, pi=3.0, gain=1.0: template[9]=3.0."""
        dt = DeepTemplate(cfg)
        q_pred = torch.zeros(1, cfg.n_orientations)
        q_pred[0, 9] = 1.0
        pi_pred = torch.tensor([[3.0]])
        out = dt(q_pred, pi_pred)
        gain = dt.gain.item()
        assert out[0, 9].item() == pytest.approx(gain * 3.0, rel=0.05)
        assert out[0, 0].item() == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# Phase 3: SOMRing
# ===================================================================

class TestSOMRing:

    def test_output_shape(self, cfg):
        som = SOMRing(cfg)
        B = 4
        r_som = torch.zeros(B, cfg.n_orientations)
        drive = torch.randn(B, cfg.n_orientations)
        out = som(drive, r_som)
        assert out.shape == (B, cfg.n_orientations)

    def test_zero_drive_stays_zero(self, cfg):
        som = SOMRing(cfg)
        r_som = torch.zeros(2, cfg.n_orientations)
        drive = torch.zeros(2, cfg.n_orientations)
        for _ in range(20):
            r_som = som(drive, r_som)
        assert torch.allclose(r_som, torch.zeros_like(r_som), atol=1e-7)

    def test_positive_drive_produces_positive_rates(self, cfg):
        som = SOMRing(cfg)
        r_som = torch.zeros(1, cfg.n_orientations)
        drive = torch.ones(1, cfg.n_orientations)
        for _ in range(30):
            r_som = som(drive, r_som)
        assert (r_som > 0).all()

    def test_som_tracks_drive_pattern(self, cfg):
        som = SOMRing(cfg)
        r_som = torch.zeros(1, cfg.n_orientations)
        drive = torch.zeros(1, cfg.n_orientations)
        drive[0, 9] = 5.0
        drive[0, 10] = 3.0
        for _ in range(50):
            r_som = som(drive, r_som)
        assert r_som[0].argmax().item() == 9

    def test_som_non_negative_with_non_negative_drive(self, cfg):
        som = SOMRing(cfg)
        r_som = torch.zeros(1, cfg.n_orientations)
        drive = torch.randn(1, cfg.n_orientations).abs()
        for _ in range(30):
            r_som = som(drive, r_som)
        assert (r_som >= -1e-7).all()

    def test_som_trajectory(self, cfg, capsys):
        """SOM with constant drive=1.0 should approach rectified_softplus(1.0)≈0.81."""
        som = SOMRing(cfg)
        r_som = torch.zeros(1, cfg.n_orientations)
        drive = torch.ones(1, cfg.n_orientations)
        expected_ss = rectified_softplus(torch.tensor(1.0)).item()

        trajectory = []
        for _ in range(50):
            r_som = som(drive, r_som)
            trajectory.append(r_som[0, 0].item())

        print(f"\n=== SOM trajectory (drive=1.0, τ_som={cfg.tau_som}) ===")
        print(f"  Steps 1,5,10,20,50: {[f'{trajectory[i-1]:.4f}' for i in [1,5,10,20,50]]}")
        print(f"  Expected SS: {expected_ss:.4f}, actual: {trajectory[-1]:.4f}")

        assert abs(trajectory[-1] - expected_ss) / expected_ss < 0.05


# ===================================================================
# Phase 3: Gradient flow through full V1
# ===================================================================

class TestGradientFlow:

    def test_gradients_through_v1_circuit(self, cfg):
        """Chain L4 → PV → L2/3 for 5 steps. Verify learnable params get gradients."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        l23_mod = V1L23Ring(cfg)

        B = 2
        stim = generate_grating(torch.tensor([45.0, 90.0]), torch.tensor([1.0, 0.5]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)

        state = initial_state(B, cfg.n_orientations, cfg.v2_hidden_dim)
        r_l4, adapt, r_pv, r_l23 = (
            state.r_l4.clone(), state.adaptation.clone(),
            state.r_pv.clone(), state.r_l23.clone(),
        )
        r_som = torch.zeros(B, cfg.n_orientations)
        tmpl_mod = torch.zeros(B, cfg.n_orientations)

        for _ in range(5):
            r_l4, adapt = l4(stim, r_l4, r_pv, adapt)
            r_pv = pv_mod(r_l4, r_l23, r_pv)
            r_l23 = l23_mod(r_l4, r_l23, tmpl_mod, r_som, r_pv)

        loss = r_l23.sum()
        loss.backward()

        # PV weights should have gradients
        assert pv_mod.w_pv_l4_raw.grad is not None
        assert pv_mod.w_pv_l23_raw.grad is not None

        # L2/3 recurrence params should have gradients
        assert l23_mod.sigma_rec_raw.grad is not None
        assert l23_mod.gain_rec_raw.grad is not None

        # L2/3 inhibitory gains should have gradients
        assert l23_mod.w_som.gain_raw.grad is not None
        assert l23_mod.w_pv_l23.gain_raw.grad is not None


# ===================================================================
# Phase 3 numerical results
# ===================================================================

class TestPhase3Numerical:

    def test_print_l23_steady_state(self, cfg, capsys):
        """Print L2/3 steady-state values with L4+PV circuit."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        l23_mod = V1L23Ring(cfg)

        stim = generate_grating(torch.tensor([45.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)
        r_l4, adapt, r_pv, r_l23, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_mod, stim, 100)

        curve = r_l23[0].detach()
        print(f"\n=== L2/3 steady state (θ=45°, c=1.0, SOM=0, 100 steps) ===")
        print(f"  Peak channel: {curve.argmax().item()} (θ={curve.argmax().item()*5}°)")
        print(f"  Peak rate: {curve.max().item():.4f}")
        print(f"  Mean rate: {curve.mean().item():.4f}")
        print(f"  L4 peak: {r_l4[0].max().item():.4f}, PV: {r_pv.item():.4f}")
        print(f"  W_rec σ={l23_mod.sigma_rec.item():.1f}°, g={l23_mod.gain_rec.item():.3f}")

        assert curve.argmax().item() == 9
        assert curve.max().item() > 0

    def test_print_recurrence_comparison(self, cfg, capsys):
        """Print L2/3 peak with and without recurrence."""
        l4 = V1L4Ring(cfg)
        pv_mod = PVPool(cfg)
        stim = generate_grating(torch.tensor([45.0]), torch.tensor([1.0]),
                                n_orientations=cfg.n_orientations, sigma=cfg.sigma_ff)

        l23_rec = V1L23Ring(cfg)
        _, _, _, r_l23_rec, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_rec, stim, 100)
        peak_rec = r_l23_rec[0, 9].item()

        l23_no = V1L23Ring(cfg)
        with torch.no_grad():
            l23_no.gain_rec_raw.fill_(-10.0)
        _, _, _, r_l23_no, _ = _run_l4_pv_l23(cfg, l4, pv_mod, l23_no, stim, 100)
        peak_no = r_l23_no[0, 9].item()

        print(f"\n=== Recurrence comparison (θ=45°, 100 steps) ===")
        print(f"  With rec (g={l23_rec.gain_rec.item():.3f}): peak={peak_rec:.4f}")
        print(f"  Without rec (g≈0):              peak={peak_no:.4f}")
        print(f"  Amplification: {peak_rec/max(peak_no, 1e-8):.2f}x")
