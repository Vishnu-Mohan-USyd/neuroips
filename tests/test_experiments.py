"""Tests for experimental paradigms (Phase 6).

Covers:
    - Trial timing and structure (base class)
    - P1 hidden-state: conditions, counterbalancing, neutral matching
    - P2 omission: blank probe, temporal windows
    - P3 ambiguous: mixture/low-contrast stimuli
    - P4 task relevance: task-state vector encoding
    - P5 surprise dissociation: matched probe/last-context, different context
    - All paradigms: end-to-end smoke test, finite trajectories
"""

from __future__ import annotations

import pytest
import torch

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import (
    TrialConfig, TrialSet, ConditionData, ExperimentResult,
)
from src.experiments.hidden_state import HiddenStateParadigm
from src.experiments.omission import OmissionParadigm
from src.experiments.ambiguous import AmbiguousParadigm
from src.experiments.task_relevance import TaskRelevanceParadigm
from src.experiments.surprise_dissociation import SurpriseDissociationParadigm


@pytest.fixture(scope="module")
def net_and_cfg():
    torch.manual_seed(0)
    cfg = ModelConfig(n_orientations=36)
    net = LaminarV1V2Network(cfg)
    net.eval()
    return net, cfg


# ---------------------------------------------------------------------------
# Base class / trial structure
# ---------------------------------------------------------------------------

class TestTrialConfig:
    def test_defaults(self):
        tc = TrialConfig()
        assert tc.n_context == 8
        assert tc.steps_on == 8
        assert tc.steps_isi == 4
        assert tc.steps_post == 8

    def test_n_timesteps(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        # 8*(8+4) + 8 + 8 = 112
        assert p.n_timesteps == 112

    def test_probe_onset(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        # 8*(8+4) = 96
        assert p.probe_onset == 96

    def test_temporal_windows_valid(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        tw = p.temporal_windows
        assert tw["prestimulus"] == (92, 96)
        assert tw["early"] == (96, 99)
        assert tw["sustained"] == (99, 104)
        assert tw["late"] == (104, 112)

    def test_custom_trial_config(self, net_and_cfg):
        net, cfg = net_and_cfg
        tc = TrialConfig(n_context=4, steps_on=6, steps_isi=2, steps_post=4)
        p = HiddenStateParadigm(net, cfg, trial_cfg=tc)
        # 4*(6+2) + 6 + 4 = 42
        assert p.n_timesteps == 42
        assert p.probe_onset == 32


class TestBaseStimulus:
    def test_make_grating_shape(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        g = p.make_grating(45.0)
        assert g.shape == (1, 36)
        assert g.sum() > 0

    def test_build_stimulus_sequence_shape(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        context = [float(i * 15) for i in range(8)]
        stim = p.build_stimulus_sequence(context, 120.0)
        assert stim.shape == (1, 112, 36)

    def test_isi_is_blank(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        context = [45.0] * 8
        stim = p.build_stimulus_sequence(context, 45.0)
        tc = p.trial_cfg
        # First ISI: after first ON period
        isi_start = tc.steps_on
        isi_slice = stim[0, isi_start:isi_start + tc.steps_isi]
        assert isi_slice.abs().sum() == 0

    def test_on_period_nonzero(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        stim = p.build_stimulus_sequence([45.0] * 8, 45.0)
        tc = p.trial_cfg
        assert stim[0, 0:tc.steps_on].abs().sum() > 0

    def test_post_probe_is_blank(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        stim = p.build_stimulus_sequence([45.0] * 8, 45.0)
        po = p.probe_onset
        tc = p.trial_cfg
        post = stim[0, po + tc.steps_on:]
        assert post.abs().sum() == 0


# ---------------------------------------------------------------------------
# P1: Hidden-State Transitions
# ---------------------------------------------------------------------------

class TestHiddenState:
    def test_condition_count(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        # 2 rules x 6 deviations + 6 neutral = 18
        assert len(trials) == 18

    def test_trial_shape(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        trials = p.generate_trials(n_trials=5, seed=42)
        for ts in trials.values():
            assert ts.stimulus.shape == (5, 112, 36)

    def test_neutral_has_nonzero_last_context(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        tc = p.trial_cfg
        last_onset = (tc.n_context - 1) * (tc.steps_on + tc.steps_isi)
        neutral = trials["neutral_dev0"]
        assert neutral.stimulus[:, last_onset:last_onset + tc.steps_on].abs().sum() > 0

    def test_fewer_deviations(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg, probe_deviations=[0.0, 45.0])
        trials = p.generate_trials(n_trials=2, seed=42)
        # 2 rules x 2 devs + 2 neutral = 6
        assert len(trials) == 6

    def test_run_produces_result(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg, probe_deviations=[0.0, 45.0])
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert isinstance(result, ExperimentResult)
        assert result.paradigm_name == "hidden_state"
        for cd in result.conditions.values():
            assert cd.r_l23.shape == (2, 112, 36)
            assert cd.r_l4.shape == (2, 112, 36)
            assert cd.r_pv.shape == (2, 112, 1)
            assert cd.r_som.shape == (2, 112, 36)
            assert cd.q_pred.shape == (2, 112, 36)
            assert cd.pi_pred.shape == (2, 112, 1)
            assert cd.state_logits.shape == (2, 112, 3)
            assert cd.deep_template.shape == (2, 112, 36)

    def test_trial_info(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = HiddenStateParadigm(net, cfg, probe_deviations=[0.0])
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert "probe_deviations" in result.trial_info
        assert "transition_step" in result.trial_info


# ---------------------------------------------------------------------------
# P2: Omission
# ---------------------------------------------------------------------------

class TestOmission:
    def test_condition_count(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = OmissionParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        # 2 rules x 2 (omission + present) = 4
        assert len(trials) == 4
        assert "cw_omission" in trials
        assert "ccw_present" in trials

    def test_n_context_is_10(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = OmissionParadigm(net, cfg)
        assert p.trial_cfg.n_context == 10

    def test_omission_probe_is_blank(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = OmissionParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        omit = trials["cw_omission"]
        po = p.probe_onset
        tc = p.trial_cfg
        assert omit.stimulus[:, po:po + tc.steps_on].abs().sum() == 0

    def test_present_probe_nonzero(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = OmissionParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        present = trials["cw_present"]
        po = p.probe_onset
        tc = p.trial_cfg
        assert present.stimulus[:, po:po + tc.steps_on].abs().sum() > 0

    def test_omission_temporal_window(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = OmissionParadigm(net, cfg)
        tw = p.temporal_windows
        assert "omission" in tw
        po = p.probe_onset
        tc = p.trial_cfg
        assert tw["omission"] == (po, po + tc.steps_on + tc.steps_post)

    def test_run(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = OmissionParadigm(net, cfg)
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert len(result.conditions) == 4


# ---------------------------------------------------------------------------
# P3: Ambiguous
# ---------------------------------------------------------------------------

class TestAmbiguous:
    def test_condition_count(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = AmbiguousParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        # 2 rules x 3 types = 6
        assert len(trials) == 6
        assert "cw_mixture" in trials
        assert "ccw_low_contrast" in trials
        assert "cw_clear" in trials

    def test_mixture_probe_nonzero(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = AmbiguousParadigm(net, cfg)
        trials = p.generate_trials(n_trials=1, seed=42)
        mix = trials["cw_mixture"]
        po = p.probe_onset
        assert mix.stimulus[0, po].abs().sum() > 0

    def test_low_contrast_less_energy(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = AmbiguousParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        po = p.probe_onset
        clear_e = trials["cw_clear"].stimulus[:, po].sum(dim=-1).mean()
        low_e = trials["cw_low_contrast"].stimulus[:, po].sum(dim=-1).mean()
        assert low_e < clear_e

    def test_run(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = AmbiguousParadigm(net, cfg)
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert len(result.conditions) == 6

    def test_trial_info(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = AmbiguousParadigm(net, cfg)
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert "ambiguous_offset" in result.trial_info


# ---------------------------------------------------------------------------
# P4: Task Relevance
# ---------------------------------------------------------------------------

class TestTaskRelevance:
    def test_condition_count(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = TaskRelevanceParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        # 2 tasks x 2 conditions = 4
        assert len(trials) == 4
        assert "relevant_expected" in trials
        assert "irrelevant_unexpected" in trials

    def test_task_state_relevant(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = TaskRelevanceParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        rel = trials["relevant_expected"]
        assert rel.task_state is not None
        assert rel.task_state[0, 0, 0].item() == 1.0
        assert rel.task_state[0, 0, 1].item() == 0.0

    def test_task_state_irrelevant(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = TaskRelevanceParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        irrel = trials["irrelevant_expected"]
        assert irrel.task_state is not None
        assert irrel.task_state[0, 0, 0].item() == 0.0
        assert irrel.task_state[0, 0, 1].item() == 1.0

    def test_task_state_constant_across_trial(self, net_and_cfg):
        """Task state should be constant throughout a trial."""
        net, cfg = net_and_cfg
        p = TaskRelevanceParadigm(net, cfg)
        trials = p.generate_trials(n_trials=2, seed=42)
        ts = trials["relevant_expected"].task_state
        # All timesteps should be the same
        assert torch.allclose(ts[:, 0], ts[:, -1])

    def test_run(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = TaskRelevanceParadigm(net, cfg)
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert len(result.conditions) == 4


# ---------------------------------------------------------------------------
# P5: Surprise Dissociation
# ---------------------------------------------------------------------------

class TestSurpriseDissociation:
    def test_condition_count(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = SurpriseDissociationParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        assert len(trials) == 2
        assert "high_v2_surprise" in trials
        assert "low_v2_surprise" in trials

    def test_matched_probe(self, net_and_cfg):
        """Probe stimulus should be identical in both conditions."""
        net, cfg = net_and_cfg
        p = SurpriseDissociationParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        po = p.probe_onset
        tc = p.trial_cfg
        probe_high = trials["high_v2_surprise"].stimulus[:, po:po + tc.steps_on]
        probe_low = trials["low_v2_surprise"].stimulus[:, po:po + tc.steps_on]
        assert torch.allclose(probe_high, probe_low, atol=1e-6)

    def test_matched_last_context(self, net_and_cfg):
        """Last context stimulus should be identical in both conditions."""
        net, cfg = net_and_cfg
        p = SurpriseDissociationParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        tc = p.trial_cfg
        last_onset = (tc.n_context - 1) * (tc.steps_on + tc.steps_isi)
        last_high = trials["high_v2_surprise"].stimulus[:, last_onset:last_onset + tc.steps_on]
        last_low = trials["low_v2_surprise"].stimulus[:, last_onset:last_onset + tc.steps_on]
        assert torch.allclose(last_high, last_low, atol=1e-6)

    def test_different_earlier_context(self, net_and_cfg):
        """Context before last should differ (CW vs CCW)."""
        net, cfg = net_and_cfg
        p = SurpriseDissociationParadigm(net, cfg)
        trials = p.generate_trials(n_trials=3, seed=42)
        tc = p.trial_cfg
        first_high = trials["high_v2_surprise"].stimulus[:, 0:tc.steps_on]
        first_low = trials["low_v2_surprise"].stimulus[:, 0:tc.steps_on]
        assert not torch.allclose(first_high, first_low, atol=1e-6)

    def test_run(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = SurpriseDissociationParadigm(net, cfg)
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert len(result.conditions) == 2

    def test_trial_info(self, net_and_cfg):
        net, cfg = net_and_cfg
        p = SurpriseDissociationParadigm(net, cfg)
        result = p.run(n_trials=2, seed=42, batch_size=4)
        assert "transition_step" in result.trial_info


# ---------------------------------------------------------------------------
# All paradigms: smoke test
# ---------------------------------------------------------------------------

class TestAllParadigmsSmoke:
    """End-to-end smoke tests for all 5 paradigms."""

    @pytest.fixture(scope="class")
    def all_results(self, net_and_cfg):
        net, cfg = net_and_cfg
        results = {}
        for ParadigmClass in [HiddenStateParadigm, OmissionParadigm,
                               AmbiguousParadigm, TaskRelevanceParadigm,
                               SurpriseDissociationParadigm]:
            if ParadigmClass == HiddenStateParadigm:
                p = ParadigmClass(net, cfg, probe_deviations=[0.0, 45.0])
            else:
                p = ParadigmClass(net, cfg)
            result = p.run(n_trials=2, seed=42, batch_size=4)
            results[result.paradigm_name] = result
        return results

    def test_all_five_present(self, all_results):
        assert len(all_results) == 5
        expected = {"hidden_state", "omission", "ambiguous",
                    "task_relevance", "surprise_dissociation"}
        assert set(all_results.keys()) == expected

    def test_all_trajectories_finite(self, all_results):
        for name, result in all_results.items():
            for cond, cd in result.conditions.items():
                assert torch.isfinite(cd.r_l23).all(), f"{name}/{cond}: non-finite r_l23"
                assert torch.isfinite(cd.r_l4).all(), f"{name}/{cond}: non-finite r_l4"
                assert torch.isfinite(cd.q_pred).all(), f"{name}/{cond}: non-finite q_pred"

    def test_temporal_windows_within_bounds(self, all_results):
        for name, result in all_results.items():
            any_cd = next(iter(result.conditions.values()))
            T = any_cd.r_l23.shape[1]
            for wname, (start, end) in result.temporal_windows.items():
                assert 0 <= start < end <= T, (
                    f"{name}/{wname}: [{start},{end}) outside [0,{T})")

    def test_condition_data_shapes_consistent(self, all_results):
        """All fields within a ConditionData should have same (n_trials, T)."""
        for name, result in all_results.items():
            for cond, cd in result.conditions.items():
                n, T = cd.r_l23.shape[:2]
                assert cd.r_l4.shape[:2] == (n, T)
                assert cd.r_pv.shape[:2] == (n, T)
                assert cd.r_som.shape[:2] == (n, T)
                assert cd.q_pred.shape[:2] == (n, T)
                assert cd.pi_pred.shape[:2] == (n, T)
                assert cd.state_logits.shape[:2] == (n, T)
                assert cd.deep_template.shape[:2] == (n, T)
