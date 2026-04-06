"""Tests for stimulus system: utils, gratings, sequences, config."""

import math

import torch
import pytest

from src.utils import (
    circular_distance,
    circular_distance_abs,
    circular_gaussian,
    make_circular_gaussian_kernel,
    circular_gaussian_fwhm,
    shifted_softplus,
    ExcitatoryLinear,
    InhibitoryGain,
)
from src.config import ModelConfig, MechanismType, TrainingConfig, StimulusConfig, load_config
from src.state import NetworkState, initial_state
from src.stimulus.gratings import population_code, naka_rushton, generate_grating, make_ambiguous_stimulus
from src.stimulus.sequences import (
    build_transition_matrix,
    sample_state_sequence,
    generate_orientation_sequence,
    generate_batch_sequences,
    HMMSequenceGenerator,
    HMMState,
)


# ===================================================================
# Circular distance
# ===================================================================

class TestCircularDistance:

    def test_zero_distance(self):
        a = torch.tensor([0.0, 45.0, 90.0])
        d = circular_distance(a, a)
        assert torch.allclose(d, torch.zeros(3))

    def test_wraparound(self):
        # 170 deg and 10 deg are 20 deg apart in 180-periodic space
        d = circular_distance_abs(torch.tensor(170.0), torch.tensor(10.0), period=180.0)
        assert torch.allclose(d, torch.tensor(20.0))

    def test_signed(self):
        # CW direction
        d = circular_distance(torch.tensor(10.0), torch.tensor(0.0), period=180.0)
        assert d.item() == pytest.approx(10.0)

        # CCW direction
        d = circular_distance(torch.tensor(0.0), torch.tensor(10.0), period=180.0)
        assert d.item() == pytest.approx(-10.0)

    def test_max_distance(self):
        d = circular_distance_abs(torch.tensor(0.0), torch.tensor(90.0), period=180.0)
        assert d.item() == pytest.approx(90.0)


# ===================================================================
# Circular Gaussian
# ===================================================================

class TestCircularGaussian:

    def test_peak_at_zero(self):
        v = circular_gaussian(torch.tensor(0.0), sigma=12.0)
        assert v.item() == pytest.approx(1.0)

    def test_decays(self):
        v0 = circular_gaussian(torch.tensor(0.0), sigma=12.0)
        v30 = circular_gaussian(torch.tensor(30.0), sigma=12.0)
        assert v30.item() < v0.item()

    def test_fwhm_analytical(self):
        sigma = 12.0
        fwhm = circular_gaussian_fwhm(sigma)
        expected = 2.0 * sigma * math.sqrt(2.0 * math.log(2.0))
        assert fwhm == pytest.approx(expected, rel=1e-6)
        # Should be approximately 28.2 degrees
        assert 28.0 < fwhm < 29.0

    def test_kernel_shape_and_symmetry(self):
        K = make_circular_gaussian_kernel(36, sigma=12.0)
        assert K.shape == (36, 36)
        # Symmetric
        assert torch.allclose(K, K.T)
        # Diagonal is 1 (distance 0)
        assert torch.allclose(K.diag(), torch.ones(36))

    def test_kernel_circulant(self):
        K = make_circular_gaussian_kernel(36, sigma=12.0)
        # Each row is a cyclic shift of row 0
        row0 = K[0]
        row1 = K[1]
        # row1 should equal row0 rolled by +1 (peak shifts from col 0 to col 1)
        assert torch.allclose(row1, row0.roll(1), atol=1e-5)


# ===================================================================
# Shifted softplus
# ===================================================================

class TestShiftedSoftplus:

    def test_zero_at_zero(self):
        v = shifted_softplus(torch.tensor(0.0))
        assert v.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_positive(self):
        v = shifted_softplus(torch.tensor(5.0))
        assert v.item() > 0.0

    def test_near_zero_for_large_negative(self):
        v = shifted_softplus(torch.tensor(-10.0))
        assert v.item() < 0.01

    def test_gradient_exists(self):
        x = torch.tensor(1.0, requires_grad=True)
        y = shifted_softplus(x)
        y.backward()
        assert x.grad is not None and x.grad.item() > 0


# ===================================================================
# Sign-constrained layers
# ===================================================================

class TestSignConstrainedLayers:

    def test_excitatory_weights_positive(self):
        layer = ExcitatoryLinear(10, 5)
        x = torch.randn(2, 10)
        # Effective weights should be non-negative
        w = torch.nn.functional.softplus(layer.weight_raw)
        assert (w >= 0).all()

    def test_inhibitory_gain_positive(self):
        inh = InhibitoryGain(init_gain=1.0)
        assert inh.gain.item() > 0
        # Should initialize near 1.0
        assert inh.gain.item() == pytest.approx(1.0, abs=0.1)

    def test_inhibitory_forward(self):
        inh = InhibitoryGain(init_gain=2.0)
        x = torch.ones(3, 5)
        out = inh(x)
        assert torch.allclose(out, torch.full_like(out, inh.gain.item()), atol=0.01)


# ===================================================================
# Config
# ===================================================================

class TestConfig:

    def test_mechanism_enum(self):
        assert MechanismType("dampening") == MechanismType.DAMPENING
        assert MechanismType("center_surround") == MechanismType.CENTER_SURROUND

    def test_model_config_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_orientations == 36
        assert cfg.orientation_step == 5.0
        assert len(cfg.preferred_orientations) == 36
        assert cfg.preferred_orientations[0] == 0.0
        assert cfg.preferred_orientations[-1] == pytest.approx(175.0)

    def test_load_config(self):
        model_cfg, train_cfg, stim_cfg = load_config("config/defaults.yaml")
        assert isinstance(model_cfg, ModelConfig)
        assert isinstance(train_cfg, TrainingConfig)
        assert isinstance(stim_cfg, StimulusConfig)
        assert model_cfg.mechanism == MechanismType.CENTER_SURROUND
        assert train_cfg.batch_size == 32
        assert stim_cfg.n_states == 3

    def test_stimulus_defaults_preserve_legacy_path(self):
        stim_cfg = StimulusConfig()
        assert stim_cfg.cue_mode == "none"
        assert stim_cfg.cue_prestimulus_steps == 0
        assert stim_cfg.ambiguous_mode == "one_sided"

    def test_option_b_prototype_config_loads(self):
        model_cfg, train_cfg, stim_cfg = load_config("config/cue_local_competitor_oracle.yaml")
        assert model_cfg.vip_enabled is True
        assert train_cfg.freeze_v2 is True
        assert train_cfg.oracle_shift_timing is True
        assert stim_cfg.cue_mode == "current"
        assert stim_cfg.cue_prestimulus_steps == 2
        assert stim_cfg.ambiguous_mode == "symmetric_local_competitor"


# ===================================================================
# NetworkState
# ===================================================================

class TestNetworkState:

    def test_initial_state(self):
        state = initial_state(batch_size=4)
        assert state.r_l4.shape == (4, 36)
        assert state.r_pv.shape == (4, 1)
        assert state.a_apical.shape == (4, 36)
        assert state.h_v2.shape == (4, 16)
        assert (state.r_l4 == 0).all()

    def test_named_tuple_fields(self):
        state = initial_state(batch_size=1)
        assert hasattr(state, "r_l4")
        assert hasattr(state, "r_vip")
        assert hasattr(state, "a_apical")
        assert hasattr(state, "deep_template")
        assert state._fields == (
            "r_l4", "r_l23", "r_pv", "r_som", "r_vip", "a_apical", "adaptation", "h_v2", "deep_template"
        )


# ===================================================================
# Gratings
# ===================================================================

class TestGratings:

    def test_population_code_shape(self):
        ori = torch.tensor([45.0, 90.0])
        pop = population_code(ori)
        assert pop.shape == (2, 36)

    def test_population_code_peak(self):
        # Orientation at 45 degrees should peak at channel 9 (45/5)
        ori = torch.tensor([45.0])
        pop = population_code(ori)
        assert pop[0].argmax().item() == 9

    def test_population_code_circular_gaussian_shape(self):
        # Verify the output follows a circular Gaussian
        ori = torch.tensor([0.0])
        pop = population_code(ori, sigma=12.0)
        # Peak at channel 0
        assert pop[0, 0].item() == pytest.approx(1.0)
        # Verify FWHM: value at half-max distance
        fwhm = circular_gaussian_fwhm(12.0)
        half_fwhm_channel = int(round(fwhm / 2 / 5))  # ~2.8 -> channel 3
        # At half-FWHM, value should be ~0.5
        assert 0.4 < pop[0, half_fwhm_channel].item() < 0.7

    def test_naka_rushton_range(self):
        c = torch.linspace(0.0, 1.0, 100)
        g = naka_rushton(c)
        assert g[0].item() == pytest.approx(0.0, abs=1e-7)
        assert g[-1].item() < 1.0  # Never reaches 1.0 unless c50=0
        # Monotonically increasing
        assert (g[1:] >= g[:-1]).all()

    def test_naka_rushton_half_saturation(self):
        c50 = 0.3
        g = naka_rushton(torch.tensor(c50), c50=c50)
        assert g.item() == pytest.approx(0.5, abs=1e-5)

    def test_generate_grating(self):
        ori = torch.tensor([90.0])
        contrast = torch.tensor([0.5])
        stim = generate_grating(ori, contrast)
        assert stim.shape == (1, 36)
        # Peak at channel 18 (90/5)
        assert stim[0].argmax().item() == 18
        # Amplitude scaled by Naka-Rushton
        gain = naka_rushton(contrast).item()
        assert stim[0].max().item() == pytest.approx(gain, rel=1e-3)

    def test_zero_contrast_gives_zero(self):
        stim = generate_grating(torch.tensor([45.0]), torch.tensor([0.0]))
        assert torch.allclose(stim, torch.zeros_like(stim), atol=1e-7)


# ===================================================================
# Sequences
# ===================================================================

class TestSequences:

    def test_transition_matrix_valid(self):
        T = build_transition_matrix(p_self=0.95)
        assert T.shape == (3, 3)
        # Rows sum to 1
        assert torch.allclose(T.sum(dim=1), torch.ones(3), atol=1e-6)
        # Diagonal entries = p_self
        assert T[0, 0].item() == pytest.approx(0.95)

    def test_state_sequence_length(self):
        gen = torch.Generator().manual_seed(42)
        states = sample_state_sequence(100, generator=gen)
        assert states.shape == (100,)
        assert states.min().item() >= 0
        assert states.max().item() <= 2

    def test_hmm_transition_stats(self):
        gen = torch.Generator().manual_seed(0)
        states = sample_state_sequence(10000, p_self=0.95, generator=gen)
        # Count self-transitions
        same = (states[1:] == states[:-1]).float().mean().item()
        # Should be close to 0.95
        assert 0.90 < same < 0.99

    def test_orientation_sequence_range(self):
        gen = torch.Generator().manual_seed(42)
        oris, states = generate_orientation_sequence(50, generator=gen)
        assert oris.shape == (50,)
        assert (oris >= 0).all() and (oris < 180).all()

    def test_cw_transitions(self):
        gen = torch.Generator().manual_seed(42)
        # Force all CW by making p_self very high
        oris, states = generate_orientation_sequence(
            1000, p_self=0.99, p_transition_cw=1.0,
            p_transition_ccw=1.0, generator=gen,
        )
        # Find segments where state == CW
        cw_mask = states == HMMState.CW
        # In CW segments, orientations should tend to increase by transition_step
        # (not a strict test due to jitter)
        cw_diffs = []
        for t in range(1, len(oris)):
            if cw_mask[t] and cw_mask[t - 1]:
                diff = (oris[t].item() - oris[t - 1].item()) % 180
                if diff > 90:
                    diff -= 180
                cw_diffs.append(diff)
        if len(cw_diffs) > 10:
            mean_diff = sum(cw_diffs) / len(cw_diffs)
            # Mean diff should be near transition_step (15 deg)
            assert 5 < mean_diff < 25

    def test_batch_sequences(self):
        gen = torch.Generator().manual_seed(42)
        oris, states, contrasts = generate_batch_sequences(
            batch_size=4, seq_length=50, generator=gen,
        )
        assert oris.shape == (4, 50)
        assert states.shape == (4, 50)
        assert contrasts.shape == (4, 50)
        assert (contrasts >= 0.15).all() and (contrasts <= 1.0).all()

    def test_circular_distance_wraparound(self):
        # 170 -> 10 should be +20 (wrapping through 180/0)
        d = circular_distance(torch.tensor(10.0), torch.tensor(170.0), period=180.0)
        assert d.item() == pytest.approx(20.0)

        # 10 -> 170 should be -20
        d = circular_distance(torch.tensor(170.0), torch.tensor(10.0), period=180.0)
        assert d.item() == pytest.approx(-20.0)


# ===================================================================
# Ambiguous stimulus
# ===================================================================

class TestAmbiguousStimulus:

    def test_mixture_shape(self):
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([30.0])
        c = torch.tensor([0.5])
        stim = make_ambiguous_stimulus(t1, t2, c)
        assert stim.shape == (1, 36)

    def test_mixture_has_two_peaks(self):
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([45.0])
        c = torch.tensor([1.0])
        stim = make_ambiguous_stimulus(t1, t2, c)
        # Should have elevated responses near both orientations
        assert stim[0, 0].item() > 0.3   # near 0 deg
        assert stim[0, 9].item() > 0.3   # near 45 deg

    def test_equal_weight_symmetric(self):
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([90.0])
        c = torch.tensor([1.0])
        stim = make_ambiguous_stimulus(t1, t2, c, weight=0.5)
        # Peak at channel 0 and channel 18 should be equal
        assert stim[0, 0].item() == pytest.approx(stim[0, 18].item(), rel=1e-3)


# ===================================================================
# Row-normalised kernel
# ===================================================================

class TestKernelNormalisation:

    def test_row_normalised_sums_to_one(self):
        K = make_circular_gaussian_kernel(36, sigma=12.0, row_normalise=True)
        assert torch.allclose(K.sum(dim=1), torch.ones(36), atol=1e-5)

    def test_unnormalised_does_not_sum_to_one(self):
        K = make_circular_gaussian_kernel(36, sigma=12.0, row_normalise=False)
        # Row sums should NOT be 1 for a non-normalised kernel
        assert not torch.allclose(K.sum(dim=1), torch.ones(36), atol=0.01)


# ===================================================================
# HMMSequenceGenerator (full-featured)
# ===================================================================

class TestHMMSequenceGenerator:

    def test_generate_shapes(self):
        gen_rng = torch.Generator().manual_seed(42)
        hmm = HMMSequenceGenerator()
        meta = hmm.generate(batch_size=4, seq_length=50, generator=gen_rng)
        assert meta.orientations.shape == (4, 50)
        assert meta.states.shape == (4, 50)
        assert meta.contrasts.shape == (4, 50)
        assert meta.is_ambiguous.shape == (4, 50)
        assert meta.task_states.shape == (4, 50, 2)
        assert meta.cues.shape == (4, 50, 36)

    def test_task_states_one_hot(self):
        gen_rng = torch.Generator().manual_seed(42)
        hmm = HMMSequenceGenerator()
        meta = hmm.generate(batch_size=20, seq_length=10, generator=gen_rng)
        # Each sequence: task_states should be constant and sum to 1 per position
        for b in range(20):
            row_sums = meta.task_states[b].sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones(10))

    def test_cues_are_zeros(self):
        gen_rng = torch.Generator().manual_seed(42)
        hmm = HMMSequenceGenerator()
        meta = hmm.generate(batch_size=4, seq_length=50, generator=gen_rng)
        assert (meta.cues == 0).all()

    def test_ambiguous_fraction(self):
        gen_rng = torch.Generator().manual_seed(42)
        hmm = HMMSequenceGenerator(ambiguous_fraction=0.15)
        meta = hmm.generate(batch_size=100, seq_length=100, generator=gen_rng)
        frac = meta.is_ambiguous.float().mean().item()
        # Should be roughly 0.15
        assert 0.10 < frac < 0.20

    def test_task_state_balance(self):
        gen_rng = torch.Generator().manual_seed(42)
        hmm = HMMSequenceGenerator()
        meta = hmm.generate(batch_size=200, seq_length=10, generator=gen_rng)
        # Count sequences with task_state [1, 0] vs [0, 1]
        relevant = (meta.task_states[:, 0, 0] == 1.0).float().mean().item()
        assert 0.35 < relevant < 0.65  # roughly 50/50

    def test_mechanism_type_is_str(self):
        # MechanismType(str, Enum) should be usable as string
        assert MechanismType.DAMPENING == "dampening"
        assert str(MechanismType.SHARPENING) == "MechanismType.SHARPENING"
        assert MechanismType.CENTER_SURROUND.value == "center_surround"
