"""Unit tests for SpikingConfig dataclass and load_spiking_config (src/config.py)."""

from __future__ import annotations

import math
import textwrap

import pytest

from src.config import (
    ModelConfig,
    SpikingConfig,
    StimulusConfig,
    TrainingConfig,
    load_config,
    load_spiking_config,
)


# ----------------------------------------------------------------------------
# 1. Defaults
# ----------------------------------------------------------------------------

class TestSpikingConfigDefaults:
    def test_V_thresh_default(self):
        assert SpikingConfig().V_thresh == 1.0

    def test_V_reset_default(self):
        assert SpikingConfig().V_reset == 0.0

    def test_membrane_time_constants_match_plan(self):
        """Defaults must exactly match rate model tau values (plan lines 46-55)."""
        cfg = SpikingConfig()
        assert cfg.tau_mem_l4 == 5.0   # matches ModelConfig.tau_l4
        assert cfg.tau_mem_l23 == 10.0  # matches ModelConfig.tau_l23
        assert cfg.tau_mem_som == 10.0  # matches ModelConfig.tau_som
        assert cfg.tau_mem_vip == 10.0  # matches ModelConfig.tau_vip

    def test_tau_filter_default(self):
        assert SpikingConfig().tau_filter == 10.0

    def test_tau_adapt_default(self):
        """tau_adapt=200 matches rate model SSA and Bellec LSNN."""
        assert SpikingConfig().tau_adapt == 200.0

    def test_surrogate_slope_default(self):
        assert SpikingConfig().surrogate_slope == 25.0

    def test_lsnn_sizes_default(self):
        cfg = SpikingConfig()
        assert cfg.n_lsnn_neurons == 80
        assert cfg.n_lsnn_exc == 40
        assert cfg.n_lsnn_adaptive == 20
        assert cfg.n_lsnn_inh == 20

    def test_lsnn_adapt_beta_default(self):
        assert SpikingConfig().lsnn_adapt_beta == 1.8

    def test_surrogate_dampen_default(self):
        """Default surrogate_dampen = 0.3 (Bellec 2018 §3, LSNN-official)."""
        assert SpikingConfig().surrogate_dampen == 0.3

    def test_stationary_mode_default(self):
        """Default stationary_mode = True (Lead Ruling 1, 2026-04-10)."""
        assert SpikingConfig().stationary_mode is True

    def test_matches_rate_model_membrane_taus(self):
        """Spiking tau_mem_* must equal the rate model's tau_*."""
        m = ModelConfig()
        s = SpikingConfig()
        assert s.tau_mem_l4 == m.tau_l4
        assert s.tau_mem_l23 == m.tau_l23
        assert s.tau_mem_som == m.tau_som
        assert s.tau_mem_vip == m.tau_vip

    def test_tau_mem_v2_default(self):
        """Default tau_mem_v2 = 20 ms (Bellec 2018 LSNN canonical value)."""
        assert SpikingConfig().tau_mem_v2 == 20.0


# ----------------------------------------------------------------------------
# 2. spike_filter_alpha computed from tau_filter (__post_init__)
# ----------------------------------------------------------------------------

class TestSpikeFilterAlphaComputation:
    def test_default_alpha_derived_from_tau_filter(self):
        """spike_filter_alpha = exp(-1/tau_filter) by default."""
        cfg = SpikingConfig()
        expected = math.exp(-1.0 / 10.0)  # tau_filter=10 default
        assert cfg.spike_filter_alpha == pytest.approx(expected, abs=1e-12)
        assert cfg.spike_filter_alpha == pytest.approx(0.9048374180359595, abs=1e-12)

    def test_alpha_updates_with_custom_tau_filter(self):
        cfg = SpikingConfig(tau_filter=5.0)
        expected = math.exp(-1.0 / 5.0)
        assert cfg.spike_filter_alpha == pytest.approx(expected, abs=1e-12)

    def test_explicit_alpha_override_honoured(self):
        """Explicit spike_filter_alpha wins over the auto-derivation."""
        cfg = SpikingConfig(tau_filter=10.0, spike_filter_alpha=0.5)
        assert cfg.spike_filter_alpha == 0.5  # NOT exp(-0.1) = 0.905

    def test_rejects_nonpositive_tau_filter(self):
        """tau_filter <= 0 is invalid and raises."""
        with pytest.raises(ValueError, match="tau_filter must be > 0"):
            SpikingConfig(tau_filter=0.0)
        with pytest.raises(ValueError, match="tau_filter must be > 0"):
            SpikingConfig(tau_filter=-1.0)


# ----------------------------------------------------------------------------
# 3. LSNN decomposition consistency
# ----------------------------------------------------------------------------

class TestLSNNDecomposition:
    def test_subpopulations_sum_to_total(self):
        """40 + 20 + 20 = 80 (default)."""
        cfg = SpikingConfig()
        assert cfg.n_lsnn_exc + cfg.n_lsnn_adaptive + cfg.n_lsnn_inh == cfg.n_lsnn_neurons

    def test_rejects_mismatched_decomposition(self):
        with pytest.raises(ValueError, match="LSNN sub-populations must sum"):
            SpikingConfig(n_lsnn_neurons=80, n_lsnn_exc=40, n_lsnn_adaptive=20, n_lsnn_inh=10)

    def test_accepts_custom_consistent_decomposition(self):
        cfg = SpikingConfig(
            n_lsnn_neurons=100, n_lsnn_exc=50, n_lsnn_adaptive=30, n_lsnn_inh=20,
        )
        assert cfg.n_lsnn_neurons == 100
        assert cfg.n_lsnn_exc == 50


# ----------------------------------------------------------------------------
# 4. Backward compatibility — ModelConfig and load_config untouched
# ----------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_model_config_still_works(self):
        """ModelConfig does not have any new fields or broken defaults."""
        m = ModelConfig()
        assert m.n_orientations == 36
        assert m.tau_l4 == 5
        assert m.tau_l23 == 10

    def test_load_config_still_returns_three_tuple(self):
        """Existing callers unpack load_config as model, train, stim — must still work."""
        result = load_config("config/defaults.yaml")
        assert isinstance(result, tuple)
        assert len(result) == 3
        m, t, s = result
        assert isinstance(m, ModelConfig)
        assert isinstance(t, TrainingConfig)
        assert isinstance(s, StimulusConfig)

    def test_training_config_still_works(self):
        tc = TrainingConfig()
        assert tc.stage1_n_steps == 2000


# ----------------------------------------------------------------------------
# 5. load_spiking_config
# ----------------------------------------------------------------------------

class TestLoadSpikingConfig:
    def test_load_from_yaml_without_spiking_block_returns_defaults(self, tmp_path):
        """YAML with no `spiking:` block -> SpikingConfig with all defaults."""
        p = tmp_path / "no_spiking.yaml"
        p.write_text("model:\n  n_orientations: 36\n")

        cfg = load_spiking_config(p)
        assert isinstance(cfg, SpikingConfig)
        assert cfg.V_thresh == 1.0
        assert cfg.tau_mem_l23 == 10.0

    def test_load_from_yaml_with_partial_overrides(self, tmp_path):
        """Partial YAML `spiking:` block overrides only the listed fields."""
        p = tmp_path / "partial.yaml"
        p.write_text(textwrap.dedent("""\
            spiking:
              V_thresh: 0.5
              tau_mem_l23: 20.0
              surrogate_slope: 10.0
        """))

        cfg = load_spiking_config(p)
        # Overridden
        assert cfg.V_thresh == 0.5
        assert cfg.tau_mem_l23 == 20.0
        assert cfg.surrogate_slope == 10.0
        # Unchanged defaults
        assert cfg.V_reset == 0.0
        assert cfg.tau_mem_l4 == 5.0
        assert cfg.n_lsnn_neurons == 80

    def test_load_full_override(self, tmp_path):
        """All fields override cleanly through YAML."""
        p = tmp_path / "full.yaml"
        p.write_text(textwrap.dedent("""\
            spiking:
              V_thresh: 1.5
              V_reset: 0.1
              tau_mem_l4: 4.0
              tau_mem_l23: 12.0
              tau_mem_som: 8.0
              tau_mem_vip: 12.0
              tau_filter: 15.0
              tau_adapt: 300.0
              surrogate_slope: 30.0
              n_lsnn_neurons: 100
              n_lsnn_exc: 50
              n_lsnn_adaptive: 30
              n_lsnn_inh: 20
              lsnn_adapt_beta: 2.0
        """))

        cfg = load_spiking_config(p)
        assert cfg.V_thresh == 1.5
        assert cfg.tau_mem_l4 == 4.0
        assert cfg.tau_filter == 15.0
        # Derived alpha updates with tau_filter
        assert cfg.spike_filter_alpha == pytest.approx(math.exp(-1.0 / 15.0), abs=1e-12)
        assert cfg.n_lsnn_neurons == 100
        assert cfg.lsnn_adapt_beta == 2.0

    def test_load_explicit_alpha_override(self, tmp_path):
        """Explicit spike_filter_alpha in YAML overrides the auto-derivation."""
        p = tmp_path / "alpha.yaml"
        p.write_text(textwrap.dedent("""\
            spiking:
              tau_filter: 10.0
              spike_filter_alpha: 0.75
        """))

        cfg = load_spiking_config(p)
        assert cfg.spike_filter_alpha == 0.75

    def test_load_surrogate_dampen_override(self, tmp_path):
        """YAML `spiking.surrogate_dampen` overrides the default (0.3)."""
        p = tmp_path / "dampen.yaml"
        p.write_text(textwrap.dedent("""\
            spiking:
              surrogate_dampen: 0.5
        """))
        cfg = load_spiking_config(p)
        assert cfg.surrogate_dampen == 0.5
        # Unchanged defaults
        assert cfg.stationary_mode is True
        assert cfg.surrogate_slope == 25.0

    def test_load_stationary_mode_override(self, tmp_path):
        """YAML `spiking.stationary_mode: false` switches to running-state targets."""
        p = tmp_path / "running.yaml"
        p.write_text(textwrap.dedent("""\
            spiking:
              stationary_mode: false
        """))
        cfg = load_spiking_config(p)
        assert cfg.stationary_mode is False
        # Unchanged defaults
        assert cfg.surrogate_dampen == 0.3

    def test_load_rejects_inconsistent_lsnn_decomposition(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(textwrap.dedent("""\
            spiking:
              n_lsnn_neurons: 80
              n_lsnn_exc: 40
              n_lsnn_adaptive: 20
              n_lsnn_inh: 10
        """))

        with pytest.raises(ValueError, match="LSNN sub-populations must sum"):
            load_spiking_config(p)
