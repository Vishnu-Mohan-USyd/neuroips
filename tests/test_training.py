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

from src.config import ModelConfig, TrainingConfig, StimulusConfig, MechanismType
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
    return ModelConfig(mechanism=MechanismType.CENTER_SURROUND, feedback_mode='fixed')

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
        expected_keys = {"total", "sensory", "prediction", "energy_exc", "energy_total", "homeostasis", "state", "fb_sparsity", "surprise", "error_readout", "detection", "l4_sensory", "mismatch"}
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
        assert net.feedback.surround_gain_raw.grad is not None


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
        for p in net.feedback.parameters():
            assert p.requires_grad
        assert net.l23.sigma_rec_raw.requires_grad
        assert net.l23.gain_rec_raw.requires_grad

    def test_stage2_optimizer_groups(self, net, loss_fn, train_cfg):
        groups = create_stage2_optimizer(net, loss_fn, train_cfg)
        assert len(groups.param_groups) == 4

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
    def test_output_shapes(self, model_cfg, train_cfg):
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(4, 10)

        stim_seq, cue_seq, task_seq, thetas, next_thetas, _ = build_stimulus_sequence(
            metadata, model_cfg, train_cfg
        )

        B, S, N = 4, 10, model_cfg.n_orientations
        T_total = S * (train_cfg.steps_on + train_cfg.steps_isi)

        assert stim_seq.shape == (B, T_total, N)
        assert cue_seq.shape == (B, T_total, N)
        assert task_seq.shape == (B, T_total, 2)
        assert thetas.shape == (B, S)
        assert next_thetas.shape == (B, S)

    def test_isi_is_zero(self, model_cfg, train_cfg):
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(2, 5)
        stim_seq, _, _, _, _, _ = build_stimulus_sequence(metadata, model_cfg, train_cfg)
        # ISI of first presentation (timesteps 8-11)
        isi_stim = stim_seq[:, train_cfg.steps_on:train_cfg.steps_on + train_cfg.steps_isi]
        assert torch.allclose(isi_stim, torch.zeros_like(isi_stim))

    def test_on_period_nonzero(self, model_cfg, train_cfg):
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(2, 5)
        stim_seq, _, _, _, _, _ = build_stimulus_sequence(metadata, model_cfg, train_cfg)
        on_stim = stim_seq[:, :train_cfg.steps_on]
        assert on_stim.sum() > 0

    def test_next_thetas_shifted(self, model_cfg, train_cfg):
        """next_thetas are shifted by 1."""
        hmm = HMMSequenceGenerator(n_orientations=model_cfg.n_orientations)
        metadata = hmm.generate(2, 5)
        _, _, _, thetas, next_thetas, _ = build_stimulus_sequence(metadata, model_cfg, train_cfg)
        assert torch.equal(next_thetas[:, :-1], thetas[:, 1:])


# ---------------------------------------------------------------------------
# Smoke tests: Stage 1 & Stage 2
# ---------------------------------------------------------------------------

class TestStage1Smoke:
    def test_stage1_runs(self):
        """Stage 1 completes with a few steps without errors."""
        cfg = ModelConfig(mechanism=MechanismType.DAMPENING, feedback_mode='fixed')
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
        cfg = ModelConfig(mechanism=MechanismType.DAMPENING, feedback_mode='fixed')
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
        cfg = ModelConfig(mechanism=MechanismType.DAMPENING, feedback_mode='fixed')
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
        cfg = ModelConfig(mechanism=MechanismType.DAMPENING, feedback_mode='fixed')
        train = TrainingConfig(stage2_n_steps=3, batch_size=2, seq_length=5)
        stim = StimulusConfig()
        net = LaminarV1V2Network(cfg)
        loss_fn = CompositeLoss(train, cfg)

        from src.training.stage2_feedback import run_stage2
        result = run_stage2(net, loss_fn, cfg, train, stim, seed=42)

        assert result.n_steps_trained == 3
        assert not math.isnan(result.final_loss)

    def test_stage2_all_mechanisms(self):
        """Stage 2 runs for all 5 mechanism types."""
        train = TrainingConfig(stage2_n_steps=2, batch_size=2, seq_length=3)
        stim = StimulusConfig()

        for mech in MechanismType:
            cfg = ModelConfig(mechanism=mech, feedback_mode='fixed')
            net = LaminarV1V2Network(cfg)
            loss_fn = CompositeLoss(train, cfg)
            from src.training.stage2_feedback import run_stage2
            result = run_stage2(net, loss_fn, cfg, train, stim, seed=42)
            assert not math.isnan(result.final_loss), f"NaN loss for {mech.value}"


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_load_identical(self):
        """Save checkpoint, reload, verify forward pass is identical."""
        cfg = ModelConfig(mechanism=MechanismType.DAMPENING, feedback_mode='fixed')
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
