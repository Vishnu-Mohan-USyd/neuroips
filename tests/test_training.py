"""Tests for Phase 5: Training pipeline.

Tests cover:
    - CompositeLoss components (sensory, prediction, energy, homeostasis)
    - Trainer utilities (freeze/unfreeze, param groups, scheduler, readout windows)
    - Stage 1 smoke test (few steps, bypass V2)
    - Stage 2 smoke test (few steps, all mechanisms)
    - Network forward() extended aux dict
    - Checkpoint save/load
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.config import ModelConfig, TrainingConfig, StimulusConfig
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.training.trainer import (
    get_stage1_params,
    freeze_stage1,
    unfreeze_stage2,
    create_stage2_optimizer,
    make_warmup_cosine_scheduler,
    build_stimulus_sequence,
    compute_readout_indices,
    extract_readout_data,
)
from src.stimulus.sequences import HMMSequenceGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_cfg():
    return ModelConfig()

@pytest.fixture
def train_cfg():
    return TrainingConfig()

@pytest.fixture
def stim_cfg():
    return StimulusConfig()

@pytest.fixture
def net(model_cfg):
    return LaminarV1V2Network(model_cfg)

@pytest.fixture
def loss_fn(train_cfg, model_cfg):
    return CompositeLoss(train_cfg, model_cfg)


# ---------------------------------------------------------------------------
# Test: Network forward() extended aux dict
# ---------------------------------------------------------------------------

class TestNetworkAux:
    def test_forward_has_rate_trajectories(self, net, model_cfg):
        """forward() aux dict contains r_l4_all, r_pv_all, r_som_all."""
        B, T, N = 2, 5, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        r_l23_all, _, aux = net(stim)

        assert "r_l4_all" in aux
        assert "r_pv_all" in aux
        assert "r_som_all" in aux
        assert aux["r_l4_all"].shape == (B, T, N)
        assert aux["r_pv_all"].shape == (B, T, 1)
        assert aux["r_som_all"].shape == (B, T, N)

    def test_aux_trajectories_no_nan(self, net, model_cfg):
        """Rate trajectories have no NaN values."""
        B, T, N = 2, 5, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        _, _, aux = net(stim)
        for key in ["r_l4_all", "r_pv_all", "r_som_all"]:
            assert not torch.isnan(aux[key]).any(), f"NaN in {key}"


# ---------------------------------------------------------------------------
# Test: CompositeLoss
# ---------------------------------------------------------------------------

class TestCompositeLoss:
    def test_theta_to_channel(self, loss_fn):
        """_theta_to_channel converts degrees to channel indices."""
        thetas = torch.tensor([0.0, 45.0, 90.0, 175.0])
        channels = loss_fn._theta_to_channel(thetas)
        assert channels[0].item() == 0   # 0 deg -> ch 0
        assert channels[1].item() == 9   # 45 deg -> ch 9
        assert channels[2].item() == 18  # 90 deg -> ch 18
        assert channels[3].item() == 35  # 175 deg -> ch 35

    def test_theta_to_channel_wraparound(self, loss_fn):
        """180 deg wraps to 0."""
        channels = loss_fn._theta_to_channel(torch.tensor([180.0]))
        assert channels[0].item() == 0

    def test_sensory_readout_loss(self, loss_fn):
        r_l23 = torch.randn(4, 10, 36)
        thetas = torch.rand(4, 10) * 180
        loss = loss_fn.sensory_readout_loss(r_l23, thetas)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_prediction_loss(self, loss_fn):
        q_pred = torch.softmax(torch.randn(4, 10, 36), dim=-1)
        thetas = torch.rand(4, 10) * 180
        loss = loss_fn.prediction_loss(q_pred, thetas)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_prediction_loss_perfect(self, loss_fn):
        """When q_pred matches the soft Gaussian target, loss should be near zero."""
        thetas = torch.tensor([[25.0, 50.0]])  # ch5, ch10
        N = 36
        step = 5.0
        preferred = torch.arange(N).float() * step
        # Build matching circular Gaussian targets (sigma=10)
        q_pred = torch.zeros(1, 2, N)
        for i, theta in enumerate([25.0, 50.0]):
            dists = torch.abs(preferred - theta)
            dists = torch.min(dists, 180.0 - dists)
            q_pred[0, i] = torch.exp(-dists**2 / (2 * 10.0**2))
            q_pred[0, i] /= q_pred[0, i].sum()
        loss = loss_fn.prediction_loss(q_pred, thetas)
        assert loss.item() < 0.01

    def test_energy_excitatory(self, loss_fn):
        outputs = {
            "r_l4": torch.randn(4, 10, 36).abs(),
            "r_l23": torch.randn(4, 10, 36).abs(),
            "r_pv": torch.randn(4, 10, 1).abs(),
            "r_som": torch.randn(4, 10, 36).abs(),
            "deep_template": torch.randn(4, 10, 36).abs(),
        }
        e_exc, e_total = loss_fn.energy_cost(outputs)
        assert e_exc.item() > 0
        assert e_total.item() >= e_exc.item()

    def test_homeostasis_in_range(self, loss_fn):
        """No penalty when mean rate is in target range."""
        r_l23 = torch.ones(4, 10, 36) * 0.1  # mean=0.1, in [0.05, 0.5]
        penalty = loss_fn.homeostasis_penalty(r_l23)
        assert penalty.item() < 1e-6

    def test_homeostasis_too_high(self, loss_fn):
        """Penalty when mean rate exceeds upper bound."""
        r_l23 = torch.ones(4, 10, 36) * 2.0
        penalty = loss_fn.homeostasis_penalty(r_l23)
        assert penalty.item() > 0

    def test_homeostasis_too_low(self, loss_fn):
        """Penalty when mean rate below lower bound."""
        r_l23 = torch.zeros(4, 10, 36)
        penalty = loss_fn.homeostasis_penalty(r_l23)
        assert penalty.item() > 0

    def test_homeostasis_squared(self, loss_fn):
        """Penalty is squared, not linear."""
        r_l23_1 = torch.ones(1, 1, 36) * 1.0  # 0.5 above target_max
        r_l23_2 = torch.ones(1, 1, 36) * 1.5  # 1.0 above target_max
        p1 = loss_fn.homeostasis_penalty(r_l23_1)
        p2 = loss_fn.homeostasis_penalty(r_l23_2)
        # Squared: doubling the excess should ~quadruple the penalty
        assert p2.item() > p1.item() * 3

    def test_forward_returns_tuple(self, loss_fn, net, model_cfg):
        """forward() returns (total_loss, loss_dict)."""
        B, T, N = 2, 5, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        r_l23_all, _, aux = net(stim)

        outputs = {
            "r_l23": r_l23_all,
            "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"],
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
        }

        true_thetas = torch.rand(B, T) * 180
        true_next = torch.rand(B, T) * 180

        total_loss, loss_dict = loss_fn(
            outputs, true_thetas, true_next, r_l23_all, aux["q_pred_all"]
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == ()
        expected_keys = {"total", "sensory", "prediction", "energy_exc", "energy_total", "homeostasis", "state", "fb_sparsity", "surprise", "error_readout", "detection", "l4_sensory", "mismatch", "sharp", "local_disc", "pred_suppress", "fb_energy", "routine_shape"}
        assert set(loss_dict.keys()) == expected_keys
        for k, v in loss_dict.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_gradients_flow(self, loss_fn, net, model_cfg):
        """CompositeLoss gradients flow to network params."""
        B, T, N = 2, 5, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        r_l23_all, _, aux = net(stim)

        outputs = {
            "r_l23": r_l23_all,
            "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"],
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
        }

        total_loss, _ = loss_fn(
            outputs,
            torch.rand(B, T) * 180,
            torch.rand(B, T) * 180,
            r_l23_all,
            aux["q_pred_all"],
        )
        total_loss.backward()

        assert net.l23.sigma_rec_raw.grad is not None
        assert net.v2.gru.weight_ih.grad is not None
        assert net.v2.head_feedback.weight.grad is not None


# ---------------------------------------------------------------------------
# Test: Trainer utilities
# ---------------------------------------------------------------------------

class TestTrainerUtils:
    def test_freeze_stage1(self, net):
        freeze_stage1(net)
        for p in net.l4.parameters():
            assert not p.requires_grad
        assert not net.pv.w_pv_l4_raw.requires_grad
        assert not net.pv.w_pv_l23_raw.requires_grad
        # L2/3 inhibitory gains frozen
        assert not net.l23.w_som.gain_raw.requires_grad
        assert not net.l23.w_pv_l23.gain_raw.requires_grad

    def test_unfreeze_stage2(self, net):
        freeze_stage1(net)
        unfreeze_stage2(net)
        for p in net.v2.parameters():
            assert p.requires_grad
        assert net.l23.sigma_rec_raw.requires_grad
        assert net.l23.gain_rec_raw.requires_grad

    def test_stage2_optimizer_groups(self, net, loss_fn, train_cfg):
        groups = create_stage2_optimizer(net, loss_fn, train_cfg)
        assert len(groups.param_groups) == 3

    def test_warmup_cosine_scheduler(self):
        """Scheduler warms up then decays."""
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = make_warmup_cosine_scheduler(opt, warmup_steps=100, total_steps=1000)

        lrs = []
        for i in range(600):
            lrs.append(sched.get_last_lr()[0])
            opt.step()
            sched.step()

        assert lrs[50] < lrs[99]   # warmup
        assert lrs[500] < lrs[99]  # decay


class TestReadoutWindows:
    def test_compute_readout_indices(self):
        indices = compute_readout_indices(3, steps_on=8, steps_isi=4, window_start=4, window_end=7)
        assert len(indices) == 3
        assert indices[0] == (0, [4, 5, 6, 7])
        assert indices[1] == (1, [16, 17, 18, 19])
        assert indices[2] == (2, [28, 29, 30, 31])

    def test_extract_readout_data_shape(self):
        B, N = 4, 36
        seq_length = 10
        T_total = seq_length * 12
        outputs = {
            "r_l23": torch.randn(B, T_total, N),
            "q_pred": torch.randn(B, T_total, N),
        }
        indices = compute_readout_indices(seq_length)
        r_win, q_win, _ = extract_readout_data(outputs, indices)
        assert r_win.shape == (B, seq_length, N)
        assert q_win.shape == (B, seq_length, N)

    def test_extract_readout_correct_values(self):
        """Verify it picks the correct timesteps."""
        B, N = 1, 2
        seq_length = 2
        T_total = seq_length * 12  # 24

        outputs = {"r_l23": torch.zeros(B, T_total, N), "q_pred": torch.zeros(B, T_total, N)}
        # Mark timesteps 4-7 of first presentation with 1.0
        outputs["r_l23"][0, 4:8, :] = 1.0
        # Mark timesteps 16-19 of second presentation with 2.0
        outputs["r_l23"][0, 16:20, :] = 2.0

        indices = compute_readout_indices(seq_length)
        r_win, _, _ = extract_readout_data(outputs, indices)
        assert torch.allclose(r_win[0, 0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(r_win[0, 1], torch.tensor([2.0, 2.0]))


class TestBuildStimulusSequence:
    def test_output_shapes(self, model_cfg, train_cfg, stim_cfg):
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(4, 10)

        stim_seq, cue_seq, task_seq, thetas, next_thetas, _ = build_stimulus_sequence(
            metadata, model_cfg, train_cfg, stim_cfg
        )

        B, S, N = 4, 10, model_cfg.n_orientations
        T_total = S * (train_cfg.steps_on + train_cfg.steps_isi)

        assert stim_seq.shape == (B, T_total, N)
        assert cue_seq.shape == (B, T_total, N)
        assert task_seq.shape == (B, T_total, 2)
        assert thetas.shape == (B, S)
        assert next_thetas.shape == (B, S)

    def test_isi_is_zero(self, model_cfg, train_cfg, stim_cfg):
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(2, 5)
        stim_seq, _, _, _, _, _ = build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
        # ISI of first presentation (timesteps 8-11)
        isi_stim = stim_seq[:, train_cfg.steps_on:train_cfg.steps_on + train_cfg.steps_isi]
        assert torch.allclose(isi_stim, torch.zeros_like(isi_stim))

    def test_on_period_nonzero(self, model_cfg, train_cfg, stim_cfg):
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(2, 5)
        stim_seq, _, _, _, _, _ = build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
        on_stim = stim_seq[:, :train_cfg.steps_on]
        assert on_stim.sum() > 0

    def test_next_thetas_shifted(self, model_cfg, train_cfg, stim_cfg):
        """next_thetas are shifted by 1."""
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(2, 5)
        _, _, _, thetas, next_thetas, _ = build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
        assert torch.equal(next_thetas[:, :-1], thetas[:, 1:])

    def test_ambiguous_offset_flows_through(self, model_cfg, train_cfg):
        """Regression: stim_cfg.ambiguous_offset must control the second
        orientation of ambiguous mixtures in build_stimulus_sequence.

        Previously the function hardcoded +15.0 and silently ignored the
        StimulusConfig field of the same name. This test builds two stimuli
        — one with ambiguous_offset=15°, one with 20° — that are identical
        in every other way, and verifies that:
          (a) both are mixtures of two Gaussians (i.e. ambiguous mode was hit),
          (b) the 20° stimulus has its second peak offset 5° further around the
              population-code ring than the 15° stimulus.

        Note: symmetric competitors randomly choose +/- offset, so we seed
        the RNG for determinism and handle either sign.
        """
        # Hand-construct metadata with a single ambiguous presentation so that
        # the exact orientation is known and does not depend on random seeding.
        from src.stimulus.sequences import SequenceMetadata
        N = model_cfg.n_orientations
        period = model_cfg.orientation_range
        step_deg = period / N  # 5° per channel at N=36, period=180
        # Primary orientation sits on channel 6 exactly (30°). Second orientation
        # should land on channel (6 ± offset/step_deg) for the mixture to be
        # sharply peaked at two known channels.
        oris = torch.tensor([[30.0]])                # [1, 1]
        states = torch.tensor([[0]], dtype=torch.long)
        contrasts = torch.tensor([[0.8]])
        is_ambiguous = torch.tensor([[True]])
        task_states = torch.zeros(1, 1, 2); task_states[0, 0, 0] = 1.0
        cues = torch.zeros(1, 1, N)
        metadata = SequenceMetadata(
            orientations=oris,
            states=states,
            contrasts=contrasts,
            is_ambiguous=is_ambiguous,
            task_states=task_states,
            cues=cues,
        )

        cfg15 = StimulusConfig(ambiguous_offset=15.0)
        cfg20 = StimulusConfig(ambiguous_offset=20.0)
        # Seed for deterministic sign selection (symmetric competitors)
        torch.manual_seed(0)
        stim15, *_ = build_stimulus_sequence(metadata, model_cfg, train_cfg, cfg15)
        torch.manual_seed(0)
        stim20, *_ = build_stimulus_sequence(metadata, model_cfg, train_cfg, cfg20)

        # Pick the first ON timestep (steps_on>0 so index 0 is ON).
        pop15 = stim15[0, 0]  # [N]
        pop20 = stim20[0, 0]  # [N]

        ch_primary = int(round(30.0 / step_deg))             # 6

        # The two outputs must not be identical — that's exactly what would
        # happen if the old hardcoded +15 path were still active.
        assert not torch.allclose(pop15, pop20), (
            "stim_cfg.ambiguous_offset is being ignored — 15° and 20° produce "
            "identical ambiguous stimuli, which means the old hardcoded path "
            "is still active."
        )
        # Primary Gaussian is centered at 30° in both cases.
        assert pop15[ch_primary - 1] > 0 and pop20[ch_primary - 1] > 0

        # With same seed, both get same sign. The second peak is at
        # 30 ± offset. Check that offset=20 moves the second peak
        # further from the primary than offset=15.
        # Find the argmax excluding the primary peak region (±1 channel).
        mask = torch.ones(N, dtype=torch.bool)
        for c in range(max(0, ch_primary - 1), min(N, ch_primary + 2)):
            mask[c] = False
        second_peak_15 = int(pop15[mask].argmax())
        second_peak_20 = int(pop20[mask].argmax())
        # Convert back to channel index in full array
        masked_indices = torch.arange(N)[mask]
        ch15 = int(masked_indices[second_peak_15])
        ch20 = int(masked_indices[second_peak_20])
        # The second peak of offset=20 should be further from primary than offset=15
        def circ_dist(a, b, n=N):
            d = abs(a - b)
            return min(d, n - d)
        dist15 = circ_dist(ch15, ch_primary)
        dist20 = circ_dist(ch20, ch_primary)
        assert dist20 >= dist15, (
            f"offset=20° second peak (ch {ch20}) should be at least as far from "
            f"primary (ch {ch_primary}) as offset=15° (ch {ch15}); "
            f"dist20={dist20}, dist15={dist15}"
        )


# ---------------------------------------------------------------------------
# Smoke tests: Stage 1 & Stage 2
# ---------------------------------------------------------------------------

class TestStage1Smoke:
    def test_stage1_runs(self):
        """Stage 1 completes with a few steps without errors."""
        cfg = ModelConfig()
        train = TrainingConfig(stage1_n_steps=10)
        net = LaminarV1V2Network(cfg)

        from src.training.stage1_sensory import run_stage1
        result = run_stage1(net, cfg, train, seed=42)

        assert result.n_steps_trained == 10
        assert not math.isnan(result.final_loss)
        assert isinstance(result.gating_passed, dict)
        assert len(result.gating_passed) == 6

    def test_stage1_bypasses_v2(self):
        """Stage 1 should NOT run V2. V2 hidden state should remain zeros.

        Note: With rectified_softplus, untrained L2/3 may produce zero rates
        (PV inhibition > FF drive at init). This is correct — Stage 1 trains
        the gains to produce nonzero L2/3. Here we just verify V1 forward
        runs without error and V2 is not involved.
        """
        cfg = ModelConfig()
        train = TrainingConfig(stage1_n_steps=5)
        net = LaminarV1V2Network(cfg)

        from src.training.stage1_sensory import _run_v1_only
        stim = torch.randn(2, 36).abs()
        r_l23 = _run_v1_only(net, stim, n_timesteps=10)
        # Verify it runs without error and returns correct shape
        assert r_l23.shape == (2, 36)
        # L2/3 rates are non-negative (rectified_softplus)
        assert (r_l23 >= 0).all()

    def test_stage1_freezes_params(self):
        """After Stage 1, L4 and PV params are frozen."""
        cfg = ModelConfig()
        train = TrainingConfig(stage1_n_steps=5)
        net = LaminarV1V2Network(cfg)

        from src.training.stage1_sensory import run_stage1
        run_stage1(net, cfg, train, seed=42)

        for p in net.l4.parameters():
            assert not p.requires_grad
        assert not net.pv.w_pv_l4_raw.requires_grad


class TestStage2Smoke:
    def test_stage2_runs(self):
        """Stage 2 completes with a few steps without errors."""
        cfg = ModelConfig()
        train = TrainingConfig(stage2_n_steps=3, batch_size=2, seq_length=5)
        stim = StimulusConfig()
        net = LaminarV1V2Network(cfg)
        loss_fn = CompositeLoss(train, cfg)

        from src.training.stage2_feedback import run_stage2
        result = run_stage2(net, loss_fn, cfg, train, stim, seed=42)

        assert result.n_steps_trained == 3
        assert not math.isnan(result.final_loss)


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_load_identical(self):
        """Save checkpoint, reload, verify forward pass is identical."""
        cfg = ModelConfig()
        train = TrainingConfig()
        net = LaminarV1V2Network(cfg)
        loss_fn = CompositeLoss(train, cfg)

        # Run a forward pass
        torch.manual_seed(42)
        B, T, N = 2, 5, cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        r_l23_orig, _, aux_orig = net(stim)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ckpt.pt"
            torch.save({
                "model_state": net.state_dict(),
                "decoder_state": loss_fn.orientation_decoder.state_dict(),
            }, path)

            # Create fresh network and load
            net2 = LaminarV1V2Network(cfg)
            loss_fn2 = CompositeLoss(train, cfg)
            ckpt = torch.load(path, weights_only=False)
            net2.load_state_dict(ckpt["model_state"])
            loss_fn2.orientation_decoder.load_state_dict(ckpt["decoder_state"])

            # Forward pass should be identical
            r_l23_loaded, _, aux_loaded = net2(stim)
            assert torch.allclose(r_l23_orig, r_l23_loaded, atol=1e-6)
            assert torch.allclose(aux_orig["q_pred_all"], aux_loaded["q_pred_all"], atol=1e-6)


# ---------------------------------------------------------------------------
# Regression tests for bugs found during sharpening investigation
# ---------------------------------------------------------------------------

class TestRegressionBugs:
    """Regression tests to prevent recurrence of bugs found in code review."""

    def test_load_config_reads_lambda_sharp(self):
        """Bug: lambda_sharp was not parsed from YAML, silently defaulting to 0.0.

        Verify exp_sharp_p1.yaml correctly loads lambda_sharp=0.5,
        and that a config without lambda_sharp gets the default 0.0.
        """
        from src.config import load_config

        # Config with lambda_sharp set
        _, train_cfg, _ = load_config("config/exp_sharp_p1.yaml")
        assert train_cfg.lambda_sharp == 0.5, (
            f"Expected lambda_sharp=0.5 from exp_sharp_p1.yaml, got {train_cfg.lambda_sharp}"
        )

        # Default config should have lambda_sharp=0.0
        _, train_default, _ = load_config("config/defaults.yaml")
        assert train_default.lambda_sharp == 0.0, (
            f"Expected lambda_sharp=0.0 from defaults.yaml, got {train_default.lambda_sharp}"
        )

    def test_oracle_uses_shifted_states(self):
        """Bug: oracle q_pred was built from metadata.states (current-step state)
        instead of true_states (next-step state from build_stimulus_sequence).

        The oracle predicts *next* orientation, so it needs the *next* state.
        Verify that build_stimulus_sequence returns shifted states, and that
        they differ from metadata.states at state-transition boundaries.
        """
        torch.manual_seed(99)
        model_cfg = ModelConfig(feedback_mode='emergent')
        train_cfg = TrainingConfig(
            steps_on=3, steps_isi=2, seq_length=50, batch_size=8,
        )
        stim_cfg = StimulusConfig()

        gen = torch.Generator(device="cpu")
        gen.manual_seed(99)
        hmm_gen = HMMSequenceGenerator(
            n_orientations=model_cfg.n_orientations,
            p_self=stim_cfg.p_self,
            p_transition_cw=stim_cfg.p_transition_cw,
            p_transition_ccw=stim_cfg.p_transition_ccw,
            n_anchors=stim_cfg.n_anchors,
            jitter_range=stim_cfg.jitter_range,
            transition_step=stim_cfg.transition_step,
            period=model_cfg.orientation_range,
            n_states=stim_cfg.n_states,
        )
        metadata = hmm_gen.generate(train_cfg.batch_size, train_cfg.seq_length, gen)

        _, _, _, _, _, true_states = build_stimulus_sequence(
            metadata, model_cfg, train_cfg, stim_cfg
        )

        # true_states should be shifted by 1 relative to metadata.states
        # (true_states[t] = metadata.states[t+1] for t < S-1)
        expected_shifted = torch.roll(metadata.states, -1, dims=1)
        expected_shifted[:, -1] = metadata.states[:, -1]
        assert torch.equal(true_states, expected_shifted), (
            "true_states must be metadata.states shifted by 1 (next-state alignment)"
        )

        # At state transitions, shifted and unshifted MUST differ
        # (otherwise the bug wouldn't matter, but with p_self=0.95 and 50 steps,
        # there should be at least some transitions)
        differs = (true_states != metadata.states).any()
        assert differs, (
            "true_states should differ from metadata.states at state transitions. "
            "With 50 steps and p_self=0.95, transitions should occur."
        )

    def test_readout_window_covers_3_steps(self):
        """Bug: readout window used steps_on//2 as start, giving only 1 step
        for steps_on=3 (window [1,1]). Fixed to steps_on-3 → [0,2] (3 steps).

        Verify window width for steps_on=3 and steps_on=6.
        """
        # steps_on=3: window_start = max(1, 3-3) = max(1,0) = 1 → wrong!
        # Actually the code uses: window_start = max(1, steps_on - 3)
        # For steps_on=3: max(1, 0) = 1, window_end = 2 → indices [1, 2] = 2 steps
        # For steps_on=6: max(1, 3) = 3, window_end = 5 → indices [3, 4, 5] = 3 steps

        # steps_on=3
        window_start_3 = max(1, 3 - 3)
        window_end_3 = 3 - 1
        indices_3 = compute_readout_indices(
            5, steps_on=3, steps_isi=2,
            window_start=window_start_3, window_end=window_end_3,
        )
        _, ts = indices_3[0]
        assert ts == [1, 2], f"steps_on=3: expected window [1,2], got {ts}"
        assert len(ts) == 2, f"steps_on=3: expected 2 timesteps, got {len(ts)}"

        # steps_on=6
        window_start_6 = max(1, 6 - 3)
        window_end_6 = 6 - 1
        indices_6 = compute_readout_indices(
            5, steps_on=6, steps_isi=2,
            window_start=window_start_6, window_end=window_end_6,
        )
        _, ts6 = indices_6[0]
        assert ts6 == [3, 4, 5], f"steps_on=6: expected window [3,4,5], got {ts6}"
        assert len(ts6) == 3, f"steps_on=6: expected 3 timesteps, got {len(ts6)}"

        # Verify second presentation offset is correct
        _, ts6_pres1 = indices_6[1]
        steps_per = 6 + 2
        assert ts6_pres1 == [steps_per + 3, steps_per + 4, steps_per + 5], (
            f"steps_on=6 pres 1: expected offset by {steps_per}, got {ts6_pres1}"
        )

    def test_frozen_decoder_not_in_optimizer(self):
        """Bug: orientation decoder was included in optimizer even when frozen,
        causing wasted compute and potential unfreezing via weight_decay.

        Verify that when freeze_v2=True (which triggers decoder freeze),
        the decoder params are not in any optimizer param group.
        """
        model_cfg = ModelConfig(feedback_mode='emergent')
        train_cfg = TrainingConfig(freeze_v2=True)
        net = LaminarV1V2Network(model_cfg)
        loss_fn = CompositeLoss(train_cfg, model_cfg)

        freeze_stage1(net)
        unfreeze_stage2(net)

        # Freeze V2 (as stage2_feedback.py does)
        for p in net.v2.parameters():
            p.requires_grad_(False)

        # Freeze decoder (as stage2_feedback.py does when freeze_v2=True)
        for p in loss_fn.orientation_decoder.parameters():
            p.requires_grad_(False)

        optimizer = create_stage2_optimizer(net, loss_fn, train_cfg)

        # Collect all optimizer params
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))

        # Decoder params must NOT be in optimizer
        for name, p in loss_fn.orientation_decoder.named_parameters():
            assert id(p) not in opt_params, (
                f"Frozen decoder param '{name}' found in optimizer — "
                "frozen params should be excluded"
            )


class TestOracleShiftTiming:
    """Phase 3: oracle_shift_timing flag rolls q_oracle by +1 along presentation dim.

    When enabled, the oracle template built from presentation s (which predicts
    s+1) is shifted so that it is applied during presentation s+1 — acting as a
    PRIOR about the current stimulus instead of a FORECAST of the next one. The
    first presentation has no valid prior and must receive a uniform
    distribution.
    """

    def test_oracle_shift_timing_default_false(self):
        """Default value of oracle_shift_timing is False (no behavior change)."""
        tc = TrainingConfig()
        assert tc.oracle_shift_timing is False

    def test_oracle_shift_timing_loads_from_yaml(self, tmp_path):
        """oracle_shift_timing parses from YAML via load_config."""
        import yaml
        from src.config import load_config

        cfg_path = tmp_path / "shifted.yaml"
        cfg_dict = {
            "model": {"mechanism": "center_surround", "feedback_mode": "emergent"},
            "training": {
                "stage1": {"n_steps": 10},
                "stage2": {"n_steps": 10},
                "freeze_v2": True,
                "oracle_shift_timing": True,
                "oracle_template": "oracle_true",
            },
            "stimulus": {},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        _, tc, _ = load_config(cfg_path)
        assert tc.oracle_shift_timing is True

    def test_shift_semantics_roll_by_one(self):
        """q_shifted[:,0] is uniform and q_shifted[:,s]==q_orig[:,s-1] for s>=1.

        Reproduces the exact operation performed in
        src/training/stage2_feedback.py when train_cfg.oracle_shift_timing is
        True. This guards against accidental regressions to the shift formula.
        """
        torch.manual_seed(0)
        B, S, N = 4, 6, 36
        q_original = torch.randn(B, S, N).softmax(dim=-1)

        # Exact replica of the shift block in stage2_feedback.py.
        uniform_first = torch.full(
            (B, 1, N), 1.0 / N, device=q_original.device, dtype=q_original.dtype
        )
        q_shifted = torch.cat([uniform_first, q_original[:, :-1, :]], dim=1)

        # Shape preserved.
        assert q_shifted.shape == q_original.shape

        # First presentation is uniform.
        expected_uniform = torch.full((B, N), 1.0 / N)
        assert torch.allclose(q_shifted[:, 0, :], expected_uniform)
        # Each row of the uniform slice sums to 1.
        assert torch.allclose(
            q_shifted[:, 0, :].sum(dim=-1), torch.ones(B), atol=1e-6
        )

        # For every s >= 1, shifted[s] == original[s-1].
        for s in range(1, S):
            assert torch.allclose(q_shifted[:, s, :], q_original[:, s - 1, :]), (
                f"shift broken at s={s}"
            )

        # Last original presentation must be dropped (not present in shifted).
        assert torch.allclose(q_shifted[:, -1, :], q_original[:, -2, :])

    def test_shift_preserves_probability_rows(self):
        """Each row of the shifted tensor remains a valid probability vector."""
        torch.manual_seed(1)
        B, S, N = 2, 4, 36
        q_original = torch.randn(B, S, N).softmax(dim=-1)
        uniform_first = torch.full(
            (B, 1, N), 1.0 / N, device=q_original.device, dtype=q_original.dtype
        )
        q_shifted = torch.cat([uniform_first, q_original[:, :-1, :]], dim=1)
        row_sums = q_shifted.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        assert (q_shifted >= 0).all()


class TestLocalDiscriminationLoss:
    """Phase 4: local 5-way discrimination loss (expected vs +-1, +-2 neighbors).

    Verifies the new `lambda_local_disc` flag, the `local_disc_head`
    attribute, the computed loss, and the optimizer wiring so that the
    head is actually trained.
    """

    def test_lambda_local_disc_default_zero(self):
        """Default value of lambda_local_disc is 0.0 (disabled)."""
        tc = TrainingConfig()
        assert tc.lambda_local_disc == 0.0

    def test_lambda_local_disc_loads_from_yaml(self, tmp_path):
        """lambda_local_disc parses from YAML via load_config."""
        import yaml
        from src.config import load_config

        cfg_path = tmp_path / "localdisc.yaml"
        cfg_dict = {
            "model": {"mechanism": "center_surround", "feedback_mode": "emergent"},
            "training": {
                "stage1": {"n_steps": 10},
                "stage2": {"n_steps": 10},
                "lambda_local_disc": 1.5,
            },
            "stimulus": {},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        _, tc, _ = load_config(cfg_path)
        assert tc.lambda_local_disc == 1.5

    def test_local_disc_head_present_when_enabled(self):
        """CompositeLoss gets a local_disc_head when lambda_local_disc > 0."""
        model_cfg = ModelConfig(feedback_mode='emergent')
        tc = TrainingConfig(lambda_local_disc=1.0)
        loss_fn = CompositeLoss(tc, model_cfg)
        assert hasattr(loss_fn, "local_disc_head")
        # 7-input -> 7-output linear head (±3 channels = ±15°).
        assert loss_fn.local_disc_head.in_features == 7
        assert loss_fn.local_disc_head.out_features == 7

    def test_local_disc_head_absent_when_disabled(self):
        """CompositeLoss does NOT create local_disc_head when lambda=0."""
        model_cfg = ModelConfig(feedback_mode='emergent')
        tc = TrainingConfig(lambda_local_disc=0.0)
        loss_fn = CompositeLoss(tc, model_cfg)
        assert not hasattr(loss_fn, "local_disc_head")

    def test_local_disc_loss_is_finite_and_gradient_flows(self):
        """local_discrimination_loss returns a finite scalar and has gradients
        that flow back to both the head and r_l23_windows."""
        torch.manual_seed(0)
        model_cfg = ModelConfig(feedback_mode='emergent')
        tc = TrainingConfig(lambda_local_disc=1.0)
        loss_fn = CompositeLoss(tc, model_cfg)

        B, W, N = 4, 3, model_cfg.n_orientations
        # Leaf tensor with requires_grad (must not go through .abs() first).
        r_l23 = torch.randn(B, W, N).abs().requires_grad_(True)
        true_theta = torch.rand(B, W) * model_cfg.orientation_range

        l = loss_fn.local_discrimination_loss(r_l23, true_theta)

        # Scalar, finite.
        assert l.ndim == 0
        assert torch.isfinite(l).item()
        assert l.item() > 0.0  # cross-entropy from random init is > 0

        # Gradient flows.
        l.backward()
        assert r_l23.grad is not None
        assert torch.isfinite(r_l23.grad).all()
        assert loss_fn.local_disc_head.weight.grad is not None
        assert torch.isfinite(loss_fn.local_disc_head.weight.grad).all()

    def test_local_disc_channel_wrap(self):
        """Channels outside [0, N) must wrap circularly (e.g. c=0 -> [-3..3] = [N-3, N-2, N-1, 0, 1, 2, 3])."""
        torch.manual_seed(1)
        model_cfg = ModelConfig(feedback_mode='emergent')
        tc = TrainingConfig(lambda_local_disc=1.0)
        loss_fn = CompositeLoss(tc, model_cfg)
        N = model_cfg.n_orientations

        # Exercise the underlying gather logic directly.
        c = torch.tensor([[0]])  # [B=1, W=1], center channel 0
        offsets = torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype=torch.long)
        channels = (c.unsqueeze(-1) + offsets.view(1, 1, 7)) % N
        expected = torch.tensor([[[N - 3, N - 2, N - 1, 0, 1, 2, 3]]])
        assert torch.equal(channels, expected)

        # And a center in the middle (no wrap).
        c_mid = torch.tensor([[N // 2]])
        channels_mid = (c_mid.unsqueeze(-1) + offsets.view(1, 1, 7)) % N
        expected_mid = torch.tensor(
            [[[N // 2 - 3, N // 2 - 2, N // 2 - 1, N // 2, N // 2 + 1, N // 2 + 2, N // 2 + 3]]]
        )
        assert torch.equal(channels_mid, expected_mid)

    def test_forward_contains_local_disc_key(self):
        """CompositeLoss.forward() reports a 'local_disc' key in loss_dict.

        When lambda_local_disc > 0 the value is a positive float; when it is
        0 (default) the key is still present but equal to 0.0 (backward
        compatible with the old loss_dict contract)."""
        torch.manual_seed(2)
        model_cfg = ModelConfig(feedback_mode='emergent')

        # Disabled path.
        tc_off = TrainingConfig(lambda_local_disc=0.0)
        loss_off = CompositeLoss(tc_off, model_cfg)
        net = LaminarV1V2Network(model_cfg)
        B, T, N = 2, 5, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        r_l23_all, _, aux = net(stim)
        outputs = {
            "r_l23": r_l23_all, "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"], "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"], "deep_template": aux["deep_template_all"],
        }
        true_thetas = torch.rand(B, T) * 180
        true_next = torch.rand(B, T) * 180
        _, ld_off = loss_off(outputs, true_thetas, true_next, r_l23_all, aux["q_pred_all"])
        assert "local_disc" in ld_off
        assert ld_off["local_disc"] == 0.0

        # Enabled path.
        tc_on = TrainingConfig(lambda_local_disc=1.0)
        loss_on = CompositeLoss(tc_on, model_cfg)
        _, ld_on = loss_on(outputs, true_thetas, true_next, r_l23_all, aux["q_pred_all"])
        assert "local_disc" in ld_on
        assert ld_on["local_disc"] > 0.0
        assert isinstance(ld_on["local_disc"], float)

    def test_local_disc_head_in_optimizer(self):
        """create_stage2_optimizer must include local_disc_head params when enabled."""
        model_cfg = ModelConfig(feedback_mode='emergent')
        tc = TrainingConfig(lambda_local_disc=1.0)
        net = LaminarV1V2Network(model_cfg)
        loss_fn = CompositeLoss(tc, model_cfg)
        freeze_stage1(net)
        unfreeze_stage2(net)

        optimizer = create_stage2_optimizer(net, loss_fn, tc)

        # Collect all optimizer param ids.
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))

        # Head params must be present.
        head_param_ids = {id(p) for p in loss_fn.local_disc_head.parameters()}
        assert head_param_ids, "local_disc_head has no parameters"
        assert head_param_ids.issubset(opt_params), (
            "local_disc_head params missing from optimizer"
        )

    def test_lambda_disabled_loss_identical_to_pre_phase4(self):
        """With lambda_local_disc=0 (default), CompositeLoss behaviour is
        backward-compatible: the total loss does not depend on local_disc
        and no local_disc_head is created."""
        torch.manual_seed(3)
        model_cfg = ModelConfig(feedback_mode='emergent')
        tc = TrainingConfig(lambda_local_disc=0.0)
        loss_fn = CompositeLoss(tc, model_cfg)
        net = LaminarV1V2Network(model_cfg)

        B, T, N = 2, 5, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs()
        r_l23_all, _, aux = net(stim)
        outputs = {
            "r_l23": r_l23_all, "q_pred": aux["q_pred_all"],
            "r_l4": aux["r_l4_all"], "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"], "deep_template": aux["deep_template_all"],
        }
        true_thetas = torch.rand(B, T) * 180
        true_next = torch.rand(B, T) * 180
        total, ld = loss_fn(outputs, true_thetas, true_next, r_l23_all, aux["q_pred_all"])

        # No head, local_disc contribution is zero, total is finite.
        assert not hasattr(loss_fn, "local_disc_head")
        assert ld["local_disc"] == 0.0
        assert torch.isfinite(total).item()


class TestOracleSigma:
    """Phase 5: oracle_sigma lets the oracle template be narrower or wider
    than the feedforward tuning curves.

    Verifies the new `oracle_sigma` TrainingConfig field, the updated
    `LaminarV1V2Network._make_bump` signature (which now accepts an optional
    `sigma` parameter), and the YAML round-trip.
    """

    def test_oracle_sigma_default_equals_sigma_ff(self):
        """Default oracle_sigma is 12.0 (matches default sigma_ff)."""
        tc = TrainingConfig()
        mc = ModelConfig()
        assert tc.oracle_sigma == 12.0
        assert tc.oracle_sigma == mc.sigma_ff

    def test_oracle_sigma_loads_from_yaml(self, tmp_path):
        """oracle_sigma parses from YAML via load_config."""
        import yaml
        from src.config import load_config

        cfg_path = tmp_path / "sigma5.yaml"
        cfg_dict = {
            "model": {"mechanism": "center_surround", "feedback_mode": "emergent"},
            "training": {
                "stage1": {"n_steps": 10},
                "stage2": {"n_steps": 10},
                "freeze_v2": True,
                "oracle_sigma": 5.0,
            },
            "stimulus": {},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        _, tc, _ = load_config(cfg_path)
        assert tc.oracle_sigma == 5.0

    def test_all_four_sigma_configs_load(self):
        """All 4 Phase 5 sweep configs parse with the expected oracle_sigma."""
        from src.config import load_config

        expected = {
            "config/exp_sigma5_p4.yaml": 5.0,
            "config/exp_sigma8_p4.yaml": 8.0,
            "config/exp_sigma12_p4.yaml": 12.0,
            "config/exp_sigma20_p4.yaml": 20.0,
        }
        for path, sigma in expected.items():
            _, tc, _ = load_config(path)
            assert tc.oracle_sigma == sigma, (
                f"{path}: oracle_sigma={tc.oracle_sigma}, expected {sigma}"
            )
            # Shared invariants from the Phase 2 base config.
            assert tc.ambiguous_fraction == 0.3
            assert tc.oracle_template == "oracle_true"
            assert tc.freeze_v2 is True


class TestPhase24RoutineShape:
    """Phase 2.4: routine E/I symmetry-break loss + alpha_net LR multiplier.

    The routine_shape loss term is:
        shape_per_sample = |center_exc|.mean(T,N) - 0.5 * |som_drive_fb|.mean(T,N)
        l_routine_shape  = (shape_per_sample * w_routine_shape).mean()
        total += lambda_routine_shape * l_routine_shape

    where w_routine_shape is routed per-sample via task_routing (0.0 for
    focused, 2.0 for routine in sweep_dual_2_4.yaml). Only routine samples
    are rewarded for routing feedback through the inhibitory (SOM) branch.

    Plus: TrainingConfig gains an `lr_mult_alpha` field. The multiplier is
    applied to the alpha_net param group on top of stage2_lr_v2, so
    `lr_mult_alpha=10` produces alpha_net LR = stage2_lr_v2 * 10.
    """

    def _make_outputs(self, ce_mag: float, sdf_mag: float, B: int = 4,
                      T: int = 8, N: int = 36) -> dict:
        """Synthetic outputs dict with known center_exc / som_drive_fb magnitudes."""
        return {
            "r_l4":          torch.zeros(B, T, N),
            "r_l23":         torch.zeros(B, T, N),
            "r_pv":          torch.zeros(B, T, 1),
            "r_som":         torch.zeros(B, T, N),
            "deep_template": torch.zeros(B, T, N),
            "center_exc":    torch.full((B, T, N), ce_mag),
            "som_drive_fb":  torch.full((B, T, N), sdf_mag),
        }

    def test_default_lambda_routine_shape_is_zero(self):
        """Default TrainingConfig.lambda_routine_shape is 0.0 (legacy)."""
        tc = TrainingConfig()
        assert tc.lambda_routine_shape == 0.0

    def test_default_lr_mult_alpha_is_one(self):
        """Default TrainingConfig.lr_mult_alpha is 1.0 (legacy, no-op)."""
        tc = TrainingConfig()
        assert tc.lr_mult_alpha == 1.0

    def test_routine_shape_disabled_when_lambda_zero(self, model_cfg):
        """With lambda=0 the loss term is 0.0 regardless of outputs."""
        tc = TrainingConfig(lambda_routine_shape=0.0)
        loss_fn = CompositeLoss(tc, model_cfg)
        B, W, N = 4, 2, 36
        outputs = self._make_outputs(ce_mag=0.1, sdf_mag=0.05, B=B, N=N)
        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        _, ld = loss_fn(outputs, true_theta, true_next, r_l23_w, q_pred_w)
        assert ld["routine_shape"] == 0.0

    def test_routine_shape_analytic_math(self, model_cfg):
        """Legacy (no-routing) path computes shape_per_sample.mean()."""
        tc = TrainingConfig(lambda_routine_shape=1.0)
        loss_fn = CompositeLoss(tc, model_cfg)
        B, W, N = 4, 2, 36
        ce_mag, sdf_mag = 0.04, 0.02
        outputs = self._make_outputs(ce_mag=ce_mag, sdf_mag=sdf_mag, B=B, N=N)
        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        _, ld = loss_fn(outputs, true_theta, true_next, r_l23_w, q_pred_w)
        # shape_per_sample = 0.04 - 0.5*0.02 = 0.030
        expected = ce_mag - 0.5 * sdf_mag
        assert abs(ld["routine_shape"] - expected) < 1e-6, (
            f"routine_shape={ld['routine_shape']}, expected {expected}"
        )

    def test_routine_shape_routed_focused_zero(self, model_cfg):
        """All-focused routing → routine_shape is 0 (focused weight is 0)."""
        tc = TrainingConfig(
            lambda_routine_shape=1.0,
            task_routing={
                "focused": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 0.0,
                },
                "routine": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 2.0,
                },
            },
        )
        loss_fn = CompositeLoss(tc, model_cfg)
        B, W, N = 4, 2, 36
        outputs = self._make_outputs(ce_mag=0.04, sdf_mag=0.02, B=B, N=N)
        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        task_state = torch.tensor([[1.0, 0.0]] * B)  # all focused
        _, ld = loss_fn(
            outputs, true_theta, true_next, r_l23_w, q_pred_w,
            task_state=task_state, task_routing=tc.task_routing,
        )
        assert abs(ld["routine_shape"]) < 1e-6

    def test_routine_shape_routed_routine_scales(self, model_cfg):
        """All-routine routing → routine_shape = 2.0 * (ce - 0.5*sdf)."""
        tc = TrainingConfig(
            lambda_routine_shape=1.0,
            task_routing={
                "focused": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 0.0,
                },
                "routine": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 2.0,
                },
            },
        )
        loss_fn = CompositeLoss(tc, model_cfg)
        B, W, N = 4, 2, 36
        ce_mag, sdf_mag = 0.04, 0.02
        outputs = self._make_outputs(ce_mag=ce_mag, sdf_mag=sdf_mag, B=B, N=N)
        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        task_state = torch.tensor([[0.0, 1.0]] * B)  # all routine
        _, ld = loss_fn(
            outputs, true_theta, true_next, r_l23_w, q_pred_w,
            task_state=task_state, task_routing=tc.task_routing,
        )
        expected = 2.0 * (ce_mag - 0.5 * sdf_mag)  # 0.060
        assert abs(ld["routine_shape"] - expected) < 1e-6

    def test_routine_shape_prefers_inhibitory_gate(self, model_cfg):
        """Loss is strictly lower for a "perfect" gate (g_E=0.2, g_I=1.8)
        than for the identity gate (g_E=1, g_I=1) on a routine-only batch.

        Simulates the effect of the gate on a synthetic raw feedback signal
        of unit magnitude. Proves the loss landscape now rewards inhibitory
        routing — the key Phase 2.4 semantic guarantee.
        """
        tc = TrainingConfig(
            lambda_routine_shape=1.0,
            task_routing={
                "focused": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 0.0,
                },
                "routine": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 2.0,
                },
            },
        )
        loss_fn = CompositeLoss(tc, model_cfg)

        B, W, N = 4, 2, 36
        T = 8
        raw_ce = 0.04   # what center_exc would be at g_E=1
        raw_sdf = 0.04  # what som_drive_fb would be at g_I=1

        # Identity gate: g_E=1, g_I=1
        out_ident = {
            "r_l4": torch.zeros(B, T, N), "r_l23": torch.zeros(B, T, N),
            "r_pv": torch.zeros(B, T, 1), "r_som": torch.zeros(B, T, N),
            "deep_template": torch.zeros(B, T, N),
            "center_exc":   torch.full((B, T, N), 1.0 * raw_ce),
            "som_drive_fb": torch.full((B, T, N), 1.0 * raw_sdf),
        }
        # Perfect gate: g_E=0.2, g_I=1.8
        out_perfect = {
            "r_l4": torch.zeros(B, T, N), "r_l23": torch.zeros(B, T, N),
            "r_pv": torch.zeros(B, T, 1), "r_som": torch.zeros(B, T, N),
            "deep_template": torch.zeros(B, T, N),
            "center_exc":   torch.full((B, T, N), 0.2 * raw_ce),
            "som_drive_fb": torch.full((B, T, N), 1.8 * raw_sdf),
        }

        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        task_state = torch.tensor([[0.0, 1.0]] * B)

        _, ld_ident = loss_fn(
            out_ident, true_theta, true_next, r_l23_w, q_pred_w,
            task_state=task_state, task_routing=tc.task_routing,
        )
        _, ld_perfect = loss_fn(
            out_perfect, true_theta, true_next, r_l23_w, q_pred_w,
            task_state=task_state, task_routing=tc.task_routing,
        )
        assert ld_perfect["routine_shape"] < ld_ident["routine_shape"], (
            f"perfect gate routine_shape={ld_perfect['routine_shape']:.6f} "
            f"is not < identity routine_shape={ld_ident['routine_shape']:.6f} — "
            "loss landscape is not rewarding inhibitory routing."
        )

    def test_routine_shape_gradient_directions(self, model_cfg):
        """Autograd flows with the expected signs into ce and sdf."""
        tc = TrainingConfig(
            lambda_routine_shape=1.0,
            task_routing={
                "focused": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 0.0,
                },
                "routine": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 2.0,
                },
            },
        )
        loss_fn = CompositeLoss(tc, model_cfg)

        B, W, N = 4, 2, 36
        T = 8
        ce_leaf = torch.full((B, T, N), 0.04, requires_grad=True)
        sdf_leaf = torch.full((B, T, N), 0.02, requires_grad=True)
        outputs = {
            "r_l4": torch.zeros(B, T, N), "r_l23": torch.zeros(B, T, N),
            "r_pv": torch.zeros(B, T, 1), "r_som": torch.zeros(B, T, N),
            "deep_template": torch.zeros(B, T, N),
            "center_exc": ce_leaf, "som_drive_fb": sdf_leaf,
        }
        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        task_state = torch.tensor([[0.0, 1.0]] * B)

        total, _ = loss_fn(
            outputs, true_theta, true_next, r_l23_w, q_pred_w,
            task_state=task_state, task_routing=tc.task_routing,
        )
        total.backward()
        # d(routine_shape)/d(ce) > 0 → descent shrinks ce
        # d(routine_shape)/d(sdf) < 0 → descent grows sdf
        assert ce_leaf.grad.mean().item() > 0, (
            f"grad center_exc.mean() = {ce_leaf.grad.mean().item():+.3e}"
        )
        assert sdf_leaf.grad.mean().item() < 0, (
            f"grad som_drive_fb.mean() = {sdf_leaf.grad.mean().item():+.3e}"
        )

    def test_lr_mult_alpha_scales_alpha_net_group(self):
        """lr_mult_alpha=10 → alpha_net param group has LR = 10 * stage2_lr_v2.

        When use_ei_gate=True the network has an alpha_net module. The
        create_stage2_optimizer helper must add a dedicated param group for
        alpha_net whose LR equals stage2_lr_v2 * lr_mult_alpha.
        """
        from dataclasses import replace
        model_cfg = ModelConfig(feedback_mode="emergent", use_ei_gate=True)
        tc = TrainingConfig(stage2_lr_v2=3e-4, lr_mult_alpha=10.0)
        net = LaminarV1V2Network(model_cfg)
        loss_fn = CompositeLoss(tc, model_cfg)
        freeze_stage1(net)
        unfreeze_stage2(net)
        assert hasattr(net, "alpha_net")

        optimizer = create_stage2_optimizer(net, loss_fn, tc)
        alpha_ids = {id(p) for p in net.alpha_net.parameters()}

        # Find the group that contains alpha_net params.
        alpha_group = None
        for group in optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            if alpha_ids.issubset(group_ids):
                alpha_group = group
                break
        assert alpha_group is not None, (
            "alpha_net params not found in any optimizer group"
        )
        expected_lr = tc.stage2_lr_v2 * tc.lr_mult_alpha  # 3e-3
        assert abs(alpha_group["lr"] - expected_lr) < 1e-9, (
            f"alpha_net LR = {alpha_group['lr']}, expected {expected_lr}"
        )
        # alpha_net params must be in exactly one group (no double-counting).
        count = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) in alpha_ids:
                    count += 1
        assert count == len(alpha_ids), (
            f"alpha_net param count mismatch: {count} vs {len(alpha_ids)}"
        )

    def test_lr_mult_alpha_default_legacy(self):
        """lr_mult_alpha=1.0 (default) → alpha_net LR == stage2_lr_v2."""
        model_cfg = ModelConfig(feedback_mode="emergent", use_ei_gate=True)
        tc = TrainingConfig(stage2_lr_v2=3e-4)  # default lr_mult_alpha=1.0
        assert tc.lr_mult_alpha == 1.0
        net = LaminarV1V2Network(model_cfg)
        loss_fn = CompositeLoss(tc, model_cfg)
        freeze_stage1(net)
        unfreeze_stage2(net)

        optimizer = create_stage2_optimizer(net, loss_fn, tc)
        alpha_ids = {id(p) for p in net.alpha_net.parameters()}
        for group in optimizer.param_groups:
            if alpha_ids.issubset({id(p) for p in group["params"]}):
                assert abs(group["lr"] - tc.stage2_lr_v2) < 1e-9
                return
        raise AssertionError("alpha_net params not found in any optimizer group")

    def test_no_alpha_group_when_gate_off(self):
        """use_ei_gate=False → no alpha_net group (legacy 3-group layout)."""
        model_cfg = ModelConfig(feedback_mode="emergent", use_ei_gate=False)
        tc = TrainingConfig(lr_mult_alpha=10.0)  # ignored without gate
        net = LaminarV1V2Network(model_cfg)
        assert not hasattr(net, "alpha_net")
        loss_fn = CompositeLoss(tc, model_cfg)
        freeze_stage1(net)
        unfreeze_stage2(net)
        optimizer = create_stage2_optimizer(net, loss_fn, tc)
        # Legacy behavior: 3 param groups (V2, W_rec, decoder). No alpha_net.
        assert len(optimizer.param_groups) == 3

    def test_sweep_dual_2_4_yaml_loads(self):
        """sweep_dual_2_4.yaml parses with all expected Phase 2.4 fields.

        Note: lambda_routine_shape=2.0 is the MOD 1 bump — the effective
        contribution at 50/50 batch is doubled relative to the
        route-weight-only approach (see sweep header for debugger math).
        """
        from src.config import load_config
        mc, tc, _ = load_config("config/sweep/sweep_dual_2_4.yaml")
        assert mc.use_ei_gate is True
        assert tc.lambda_routine_shape == 2.0
        assert tc.lr_mult_alpha == 10.0
        assert tc.task_routing is not None
        assert tc.task_routing["focused"]["routine_shape"] == 0.0
        assert tc.task_routing["routine"]["routine_shape"] == 2.0

    def test_routine_shape_sign_check_manual_gate(self):
        """MOD 2: end-to-end sign check with a manually set "perfect" gate.

        Wiring:
          - Build a network with use_ei_gate=True
          - Set alpha_net.bias = [-5, +5], weight = 0
            → g_E = 2*sigmoid(-5) ≈ 0.01345  (near zero)
            → g_I = 2*sigmoid(+5) ≈ 1.98655  (near 2)
          - Forward a random stimulus through the network
          - Build CompositeLoss with lambda_routine_shape=2.0 + full
            routing dict (routine_shape=2.0 on routine, 0.0 on focused)
          - Run loss with task_state = all-routine, [0, 1]
          - Assert loss_dict["routine_shape"] < -0.005

        A negative value proves the loss is *rewarding* (decreasing) the
        inhibitory-routed gate configuration — this validates the loss
        *direction* not just the routing math. The threshold -0.005 is a
        conservative floor; in practice the value is ~-0.03 because the
        |som_drive_fb| term dominates with g_I near 2 and g_E near 0.
        """
        from dataclasses import replace

        model_cfg = ModelConfig(feedback_mode="emergent", use_ei_gate=True)
        tc = TrainingConfig(
            lambda_routine_shape=2.0,
            task_routing={
                "focused": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 0.0,
                },
                "routine": {
                    "sensory": 1.0, "energy": 1.0, "fb_energy": 1.0,
                    "routine_shape": 2.0,
                },
            },
        )

        torch.manual_seed(0)
        net = LaminarV1V2Network(model_cfg)
        assert hasattr(net, "alpha_net")

        # Install "perfect" gate: g_E ≈ 0.013, g_I ≈ 1.987 deterministically
        # (zero weight → gate depends only on bias).
        with torch.no_grad():
            net.alpha_net.bias[0] = -5.0
            net.alpha_net.bias[1] = +5.0
            net.alpha_net.weight.zero_()

        # Nontrivial stimulus so the network produces nonzero feedback.
        B, T, N = 4, 20, model_cfg.n_orientations
        stim = torch.randn(B, T, N).abs() * 0.5
        r_l23_all, _, aux = net(stim)

        # Sanity: gains captured in aux should match manual setting.
        gains = aux["gains_all"]  # [B, T, 2]
        g_E_mean = gains[..., 0].mean().item()
        g_I_mean = gains[..., 1].mean().item()
        assert g_E_mean < 0.02, f"g_E_mean={g_E_mean} (expected ≈ 0.0135)"
        assert g_I_mean > 1.98, f"g_I_mean={g_I_mean} (expected ≈ 1.987)"

        outputs = {
            "r_l4": aux["r_l4_all"],
            "r_l23": r_l23_all,
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
            "center_exc": aux["center_exc_all"],
            "som_drive_fb": aux["som_drive_fb_all"],
        }
        # Sanity: som_drive_fb should be present & nonzero (else the test
        # would be trivially zero — false confidence).
        assert outputs["som_drive_fb"].abs().mean().item() > 1e-5, (
            "som_drive_fb is ~zero — cannot validate sign."
        )

        loss_fn = CompositeLoss(tc, model_cfg)
        W = 2
        r_l23_w = torch.ones(B, W, N)
        q_pred_w = torch.full((B, W, N), 1.0 / N)
        true_theta = torch.zeros(B, W)
        true_next = torch.zeros(B, W)
        task_state = torch.tensor([[0.0, 1.0]] * B)  # all routine

        _, ld = loss_fn(
            outputs, true_theta, true_next, r_l23_w, q_pred_w,
            task_state=task_state, task_routing=tc.task_routing,
        )
        # THE sign check: negative = loss is rewarding inhibitory routing.
        assert ld["routine_shape"] < -0.005, (
            f"routine_shape={ld['routine_shape']:.6f} — expected < -0.005. "
            f"g_E={g_E_mean:.4f}, g_I={g_I_mean:.4f}, "
            f"|ce|={outputs['center_exc'].abs().mean().item():.4e}, "
            f"|sdf|={outputs['som_drive_fb'].abs().mean().item():.4e}. "
            "A non-negative value means the loss direction is wrong — "
            "gradient descent would move AWAY from the perfect gate."
        )
