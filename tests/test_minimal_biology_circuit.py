from __future__ import annotations

import copy
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "tools"))

import train_sweep  # noqa: E402
import tuned_emergence_lib as tuned  # noqa: E402
import assay_emergent_task_energy_axis as assay  # noqa: E402

sys.path.insert(0, str(ROOT))
import reproduce_figures  # noqa: E402


simple = train_sweep.simple
DEVICE = torch.device(simple.device)


def make_test_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    return generator


def make_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def set_gain(net: tuned.SimpleTunedNet, name: str, value: float) -> None:
    with torch.no_grad():
        net.circ_raw[tuned.CIRC_INDEX[name]] = torch.tensor(
            tuned.softplus_inverse(value),
            dtype=net.circ_raw.dtype,
            device=net.circ_raw.device,
        )


def test_alpha_tag_mapping_and_legacy_slug_compatibility() -> None:
    assert train_sweep.alpha_tag(0.0) == "0.0"
    assert train_sweep.alpha_tag(0.1) == "0.1"
    assert train_sweep.alpha_tag(0.5) == "0.5"
    assert train_sweep.alpha_tag(1.0) == "1.0"
    assert train_sweep.alpha_tag(0.15) == "0.15"

    assert train_sweep.alpha_slug(0.0) == "0p0"
    assert train_sweep.alpha_slug(0.1) == "0p1"
    assert train_sweep.alpha_slug(0.5) == "0p5"
    assert train_sweep.alpha_slug(1.0) == "1p0"
    assert train_sweep.alpha_slug(0.15) == "0p15"
    assert train_sweep.alpha_slug(0.10) == "0p1"


def test_alpha_slug_paths_and_assay_keys_preserve_precision(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    seen_paths: list[Path] = []
    seen_keys: dict[str, dict] = {}

    monkeypatch.setattr(
        assay,
        "parse_args",
        lambda: SimpleNamespace(
            run_dir=run_dir,
            device="cpu",
            out=tmp_path / "assay.json",
            alphas=[0.10, 0.15],
        ),
    )
    monkeypatch.setattr(assay, "synthetic_construct_check", lambda device: {})
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, map_location=None: {
            "state_dict": {},
            "center_feedback": False,
            "feedback_mode": tuned.FEEDBACK_MODE_POSTERIOR,
        },
    )

    def fake_assay_arm(path, device, common_local_comp_raw):
        seen_paths.append(path)
        return {"checkpoint": path.name}, {
            "feedback_mode": tuned.FEEDBACK_MODE_POSTERIOR,
        }

    def fake_json_save(result, path):
        seen_keys.update(result["per_alpha"])

    monkeypatch.setattr(assay, "assay_arm", fake_assay_arm)
    monkeypatch.setattr(assay, "atomic_json_save", fake_json_save)

    original_simple_device = simple.device
    original_tuned_device = tuned.device
    original_prefs = simple.prefs
    try:
        assay.main()
    finally:
        simple.device = original_simple_device
        simple.prefs = original_prefs
        tuned.device = original_tuned_device

    assert [path.name for path in seen_paths] == [
        "alpha_0p1_final.pt",
        "alpha_0p15_final.pt",
    ]
    assert list(seen_keys) == ["0.1", "0.15"]


def test_training_protocol_defaults_match_v8_audit(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_sweep.py"])

    args = train_sweep.parse_args()

    assert args.alphas == [0.004, 0.5]
    assert args.axis_steps == 32000
    assert args.mismatch_prob == 0.02
    assert args.task_weight is None
    assert args.constrained_efficient_coding is False


def test_core_model_defaults_disable_shortcuts() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)

    assert train_sweep.ALPHAS == (0.004, 0.5)
    assert train_sweep.MODEL_CONFIG["recurrent_cell"] == "rnn_tanh"
    assert isinstance(net.gru, nn.RNNCell)
    assert net.gru.nonlinearity == "tanh"
    assert train_sweep.MODEL_CONFIG["local_comp_strength"] == 0.0
    assert train_sweep.MODEL_CONFIG["local_comp_trainable"] is False
    assert train_sweep.MODEL_CONFIG["som_input_sigma_channels"] == 2.0
    assert math.isclose(
        train_sweep.MODEL_CONFIG["som_output_sigma_channels"],
        math.sqrt(2.0) * train_sweep.MODEL_CONFIG["som_input_sigma_channels"],
    )
    assert (
        train_sweep.MODEL_CONFIG["vip_som_sigma_channels"]
        == train_sweep.MODEL_CONFIG["som_input_sigma_channels"]
    )
    assert not hasattr(net, "local_comp_strength_raw")
    assert net.local_comp_effective_strength().item() == 0.0
    assert net.pred_feature_supp_strength == 0.0
    assert net.adapt_strength == 0.0
    assert tuned.resolve_feedback_mode(False, None) == tuned.FEEDBACK_MODE_POSTERIOR
    assert (
        train_sweep.MODEL_CONFIG["training_compatibility_version"]
        == train_sweep.TRAINING_COMPATIBILITY_VERSION
    )
    assert train_sweep.MODEL_CONFIG["model_architecture_version"] == (
        tuned.MODEL_ARCHITECTURE_VERSION
    )
    assert tuned.MODEL_ARCHITECTURE_VERSION != "shared_som_vip_posterior_v4"
    assert tuned.MODEL_ARCHITECTURE_VERSION != "split_som_vip_posterior_v5"
    assert tuned.MODEL_ARCHITECTURE_VERSION != "shared_divisive_som_posterior_v6"
    assert tuned.MODEL_ARCHITECTURE_VERSION != "split_basal_apical_som_posterior_v7"
    assert isinstance(net.w_sf_fixed, nn.Parameter)
    assert torch.isclose(
        net.w_sf_effective(),
        torch.tensor(math.sqrt(tuned.C_FIELD), device=DEVICE),
    )
    assert train_sweep.MODEL_CONFIG["fixed_canonical_vip_motif_gains"] == {
        "w_vd": 0.1,
        "w_sv": 0.1,
    }
    assert torch.isclose(
        net.circuit_gains()[tuned.CIRC_INDEX["w_vs"]],
        torch.tensor(0.5, device=DEVICE),
    )


def test_build_tuned_from_config_architecture_version_compatibility() -> None:
    current_net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG)
    versionless_net = tuned.build_tuned_from_config({"hidden": 8})

    assert current_net.hidden == train_sweep.MODEL_CONFIG["hidden"]
    assert versionless_net.hidden == 8
    with pytest.raises(ValueError, match="model_architecture_version mismatch"):
        tuned.build_tuned_from_config(
            {
                **train_sweep.MODEL_CONFIG,
                "model_architecture_version": "shared_som_vip_posterior_v4",
            }
        )


def test_som_footprint_convention_broadens_output_only() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    saved_config = tuned.model_config(net)
    input_sigma = train_sweep.MODEL_CONFIG["som_input_sigma_channels"]
    output_sigma = train_sweep.MODEL_CONFIG["som_output_sigma_channels"]
    vip_som_sigma = train_sweep.MODEL_CONFIG["vip_som_sigma_channels"]

    input_sigma_a = input_sigma / math.sqrt(2.0)
    output_sigma_a = output_sigma / math.sqrt(2.0)
    vip_som_sigma_a = vip_som_sigma / math.sqrt(2.0)
    v3_a_in, v3_a_out, v3_a_ss = tuned.population_footprints(input_sigma_a)
    expected_a_in, _, _ = tuned.population_footprints(input_sigma_a)
    _, expected_a_out, _ = tuned.population_footprints(output_sigma_a)
    _, _, expected_a_ss = tuned.population_footprints(vip_som_sigma_a)

    assert math.isclose(output_sigma_a**2, 2.0 * input_sigma_a**2)
    assert saved_config["som_input_sigma_channels"] == input_sigma
    assert saved_config["som_output_sigma_channels"] == output_sigma
    assert saved_config["vip_som_sigma_channels"] == vip_som_sigma
    assert net.A_in.shape == (tuned.N_POP, simple.N)
    assert net.A_out.shape == (simple.N, tuned.N_POP)
    assert net.A_ss.shape == (tuned.N_POP, tuned.N_POP)
    assert torch.allclose(net.A_in, expected_a_in)
    assert torch.allclose(net.A_in, v3_a_in)
    assert torch.allclose(net.A_ss, expected_a_ss)
    assert torch.allclose(net.A_ss, v3_a_ss)
    assert torch.allclose(net.A_out, expected_a_out)
    assert not torch.allclose(net.A_out, v3_a_out)
    assert torch.allclose(
        net.A_in.sum(dim=1),
        torch.ones(tuned.N_POP, device=DEVICE),
        atol=1e-6,
    )
    assert torch.allclose(
        net.A_out.sum(dim=1),
        torch.ones(simple.N, device=DEVICE),
        atol=1e-6,
    )
    assert torch.allclose(
        net.A_ss.sum(dim=1),
        torch.ones(tuned.N_POP, device=DEVICE),
        atol=1e-6,
    )


def test_prediction_som_kernel_is_peak_normalized_not_row_normalized() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    sigma = tuned.PREDICTION_SOM_SIGMA_CHANNELS
    expected = torch.exp(
        -0.5 * (tuned.circular_distance_channels() / sigma).square()
    )
    expected = expected / expected.max().clamp_min(1e-6)

    assert math.isclose(sigma, math.sqrt(2.0))
    assert net.K_pred.shape == (simple.N, simple.N)
    assert torch.allclose(net.K_pred, expected)
    assert torch.allclose(
        torch.diagonal(net.K_pred),
        torch.ones(simple.N, device=DEVICE),
    )
    assert torch.isclose(
        net.K_pred[0, 1],
        torch.tensor(math.exp(-0.25), device=DEVICE),
        atol=1e-7,
    )
    assert torch.isclose(
        net.K_pred[0, 2],
        torch.tensor(math.exp(-1.0), device=DEVICE),
        atol=1e-7,
    )
    assert torch.isclose(net.K_pred[0, -1], net.K_pred[0, 1])
    row_sums = net.K_pred.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums))
    assert torch.all(row_sums > 1.0)


def test_v4_checkpoint_rejected_by_training_resume(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    torch.save(
        {
            "target_steps": 1,
            "seed": 0,
            "center_feedback": False,
            "feedback_mode": tuned.FEEDBACK_MODE_POSTERIOR,
            "mismatch_prob": 0.0,
            "model_architecture_version": "shared_som_vip_posterior_v4",
            "training_compatibility_version": (
                "fixed_vd_sv_adaptive_vs_broad_som_output_v1"
            ),
        },
        run_dir / "common_pretrain_final.pt",
    )
    monkeypatch.setattr(
        train_sweep,
        "reference_values",
        lambda net, device: {"R_ref": 1.0},
    )
    event_log = train_sweep.EventLog(run_dir / "events.jsonl")
    args = SimpleNamespace(
        seed=0,
        pretrain_steps=1,
        lr=1e-3,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
        center_feedback=False,
        mismatch_prob=0.0,
    )

    try:
        with pytest.raises(RuntimeError, match="pretrain checkpoint metadata"):
            train_sweep.run_pretrain(args, run_dir, DEVICE, event_log)
    finally:
        event_log.close()


def test_prior_v5_objective_checkpoint_rejected_by_training_resume(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    torch.save(
        {
            "target_steps": 1,
            "seed": 0,
            "center_feedback": False,
            "feedback_mode": tuned.FEEDBACK_MODE_POSTERIOR,
            "mismatch_prob": 0.0,
            "model_architecture_version": tuned.MODEL_ARCHITECTURE_VERSION,
            "training_compatibility_version": "split_som_activity_work_v1",
        },
        run_dir / "common_pretrain_final.pt",
    )
    monkeypatch.setattr(
        train_sweep,
        "reference_values",
        lambda net, device: {"R_ref": 1.0},
    )
    event_log = train_sweep.EventLog(run_dir / "events.jsonl")
    args = SimpleNamespace(
        seed=0,
        pretrain_steps=1,
        lr=1e-3,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
        center_feedback=False,
        mismatch_prob=0.0,
    )

    try:
        with pytest.raises(RuntimeError, match="pretrain checkpoint metadata"):
            train_sweep.run_pretrain(args, run_dir, DEVICE, event_log)
    finally:
        event_log.close()


def test_choose_device_moves_l4_preferences() -> None:
    original_simple_device = simple.device
    original_tuned_device = tuned.device
    original_prefs = simple.prefs
    try:
        train_sweep.choose_device("cpu")
        assert simple.prefs.device.type == "cpu"
        l4 = tuned.l4_code(torch.tensor([0.0]))
        assert l4.device.type == "cpu"
    finally:
        simple.device = original_simple_device
        simple.prefs = original_prefs
        tuned.device = original_tuned_device


def test_posterior_feedback_default_and_legacy_mode() -> None:
    raw = torch.randn(3, simple.N, device=DEVICE)

    posterior = tuned.predictive_feedback_evidence(raw)
    assert torch.allclose(
        posterior.sum(dim=-1),
        torch.ones(3, device=DEVICE),
        atol=1e-6,
    )

    uniform = tuned.predictive_feedback_evidence(
        torch.zeros(2, simple.N, device=DEVICE)
    )
    assert torch.allclose(
        uniform,
        torch.full_like(uniform, 1.0 / float(simple.N)),
        atol=1e-7,
    )

    legacy = tuned.predictive_feedback_evidence(
        torch.zeros(2, simple.N, device=DEVICE),
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS,
    )
    assert torch.count_nonzero(legacy).item() == 0


def test_first_timestep_uses_exact_zero_feedback_state() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    theta = torch.tensor([[0.0, 5.0], [20.0, 25.0]], device=DEVICE)

    _, rates, internals = tuned.forward_seq_tuned(net, theta, return_internals=True)
    direct_rate, direct_internals = net.l23(
        tuned.l4_code(theta[:, 0]),
        torch.zeros(theta.shape[0], simple.N, device=DEVICE),
        torch.zeros(theta.shape[0], simple.N, device=DEVICE),
        return_internals=True,
    )

    assert rates.shape == (2, 2, simple.N)
    assert internals[0].shape == (2, 2, simple.N)
    assert internals[1].shape == (2, 2, tuned.N_POP)
    assert internals[2].shape == (2, 2, simple.N)
    assert internals[5].shape == (2, 2, simple.N)
    assert torch.allclose(rates[:, 0], direct_rate)
    for stacked, direct in zip(internals, direct_internals, strict=True):
        assert torch.allclose(stacked[:, 0], direct)
    (
        som,
        vip,
        som_gain,
        pre_pv_rate,
        post_pv_rate,
        exc_feedback_work,
        som_b,
        som_p,
    ) = direct_internals
    g = net.circuit_gains()
    drive = net.feedforward(tuned.l4_code(theta[:, 0]))
    b9 = drive @ net.A_in.t()
    b36 = b9 @ net.A_out.t()
    s_ff = torch.relu(
        torch.zeros_like(b9) - g[tuned.CIRC_INDEX["theta_S"]]
    )
    expected_vip = torch.relu(
        g[tuned.CIRC_INDEX["w_vd"]] * b9
        - g[tuned.CIRC_INDEX["w_vs"]] * (s_ff @ net.A_ss.t())
        - g[tuned.CIRC_INDEX["theta_V"]]
    )
    expected_som_b = torch.relu(
        g[tuned.CIRC_INDEX["w_sd"]] * b36
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * (expected_vip @ net.A_out.t())
    )
    expected_som_p = torch.zeros_like(expected_som_b)
    expected_som = 0.5 * (expected_som_b + expected_som_p)
    expected_som_gain = net.m_fixed_effective() * expected_som
    expected_pre_pv_rate = drive / (
        1.0 + net.m_fixed_effective() * expected_som_b
    )
    expected_pv = (
        g[tuned.CIRC_INDEX["w_pv"]]
        * expected_pre_pv_rate.mean(dim=-1, keepdim=True)
    ).expand_as(expected_pre_pv_rate)
    expected_post_pv_rate = expected_pre_pv_rate / (1.0 + expected_pv)
    assert torch.allclose(vip, expected_vip)
    assert torch.allclose(som_b, expected_som_b)
    assert torch.allclose(som_p, expected_som_p)
    assert torch.allclose(som, expected_som)
    assert torch.allclose(som_gain, expected_som_gain)
    assert torch.equal(pre_pv_rate, expected_pre_pv_rate)
    assert torch.allclose(post_pv_rate, expected_post_pv_rate)
    assert torch.count_nonzero(exc_feedback_work).item() == 0
    refs = train_sweep.reference_values(net, DEVICE)
    assert refs["modeled_population_activity_ref_exc_feedback_report"] == 0.0


def test_prediction_only_input_cannot_drive_e_or_vip() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    l4 = torch.zeros(2, simple.N, device=DEVICE)
    fb = torch.zeros(2, simple.N, device=DEVICE)
    fb[:, 7] = 1.0

    rates, internals = net.l23(l4, fb, return_internals=True)
    zero_rates, zero_internals = net.l23(
        l4,
        torch.zeros_like(fb),
        return_internals=True,
    )
    (
        som,
        vip,
        som_gain,
        pre_pv_rate,
        post_pv_rate,
        exc_feedback_work,
        som_b,
        som_p,
    ) = internals

    assert torch.allclose(rates, zero_rates)
    for observed, zero_observed in zip(internals, zero_internals, strict=True):
        assert torch.allclose(observed, zero_observed)
    assert torch.count_nonzero(som).item() == 0
    assert torch.count_nonzero(som_b).item() == 0
    assert torch.count_nonzero(som_p).item() == 0
    assert torch.count_nonzero(vip).item() == 0
    assert torch.count_nonzero(som_gain).item() == 0
    assert torch.count_nonzero(pre_pv_rate).item() == 0
    assert torch.count_nonzero(post_pv_rate).item() == 0
    assert torch.count_nonzero(exc_feedback_work).item() == 0
    assert torch.count_nonzero(rates).item() == 0


def test_split_sst_pools_rectify_basal_and_prediction_drives_separately() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    for name, value in {
        "w_ef": 0.25,
        "theta_S": 0.02,
        "w_vd": 0.2,
        "w_sd": 0.7,
        "w_vs": 0.3,
        "w_sv": 0.05,
        "theta_V": 0.0,
        "w_pv": 0.0,
    }.items():
        set_gain(net, name, value)
    with torch.no_grad():
        net.m_fixed.fill_(0.4)
        net.w_sf_fixed.fill_(0.6)

    l4 = tuned.l4_code(torch.tensor([0.0], device=DEVICE))
    fb = torch.zeros(1, simple.N, device=DEVICE)
    fb[:, 0] = 1.0
    fb[:, 6] = 0.4
    _, internals = net.l23(l4, fb, return_internals=True)
    som, vip, som_gain, _, _, _, som_b, som_p = internals

    g = net.circuit_gains()
    w_sf = net.w_sf_effective()
    drive = net.feedforward(l4)
    b9 = drive @ net.A_in.t()
    b36 = b9 @ net.A_out.t()
    pool_f9 = fb @ net.A_in.t()
    s_ff = torch.relu(
        w_sf * pool_f9 - g[tuned.CIRC_INDEX["theta_S"]]
    )
    expected_vip = torch.relu(
        g[tuned.CIRC_INDEX["w_vd"]] * b9
        - g[tuned.CIRC_INDEX["w_vs"]] * (s_ff @ net.A_ss.t())
        - g[tuned.CIRC_INDEX["theta_V"]]
    )
    v36 = expected_vip @ net.A_out.t()
    p36 = fb @ net.K_pred.t()
    pred_sens = drive * p36
    expected_som_b = torch.relu(
        g[tuned.CIRC_INDEX["w_sd"]] * b36
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * v36
    )
    expected_som_p = torch.relu(
        w_sf * pred_sens
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * v36
    )

    assert torch.allclose(vip, expected_vip)
    assert torch.allclose(som_b, expected_som_b)
    assert torch.allclose(som_p, expected_som_p)
    assert not torch.allclose(som_b, som_p)
    assert som.shape == (1, simple.N)
    assert som_b.shape == (1, simple.N)
    assert som_p.shape == (1, simple.N)
    assert vip.shape == (1, tuned.N_POP)
    assert som_gain.shape == (1, simple.N)


def test_split_sst_pools_use_exact_bounded_tanh_modulation() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    for name, value in {
        "w_ef": 0.25,
        "theta_S": 0.02,
        "w_vd": 0.2,
        "w_sd": 0.7,
        "w_vs": 0.3,
        "w_sv": 0.05,
        "theta_V": 0.0,
        "w_pv": 0.0,
    }.items():
        set_gain(net, name, value)
    with torch.no_grad():
        net.m_fixed.fill_(0.4)
        net.w_sf_fixed.fill_(0.6)

    l4 = tuned.l4_code(torch.tensor([0.0], device=DEVICE))
    fb = torch.zeros(1, simple.N, device=DEVICE)
    fb[:, 0] = 1.0
    fb[:, 6] = 0.4
    _, internals = net.l23(l4, fb, return_internals=True)
    _, _, _, pre_pv_rate, _, exc_feedback_work, som_b, som_p = internals

    drive = net.feedforward(l4)
    m_effective = net.m_fixed_effective()
    w_ef = net.circuit_gains()[tuned.CIRC_INDEX["w_ef"]]
    expected_exc_feedback_work = (
        w_ef * drive * fb
    )
    basal = drive / (1.0 + m_effective * som_b)
    modulation = w_ef * fb - m_effective * som_p
    expected_pre_pv_rate = basal * (
        1.0 + torch.tanh(modulation)
    )

    assert torch.allclose(exc_feedback_work, expected_exc_feedback_work)
    assert torch.allclose(pre_pv_rate, expected_pre_pv_rate)
    assert torch.all(pre_pv_rate >= 0.0)
    assert torch.all(pre_pv_rate <= 2.0 * basal)


def test_prediction_sst_can_suppress_below_basal_despite_positive_feedback() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    for name, value in {
        "w_ef": 0.01,
        "theta_S": 0.0,
        "w_vd": 0.0,
        "w_sd": 0.0,
        "w_vs": 0.0,
        "w_sv": 0.0,
        "theta_V": 0.0,
        "w_pv": 0.0,
    }.items():
        set_gain(net, name, value)
    with torch.no_grad():
        net.m_fixed.fill_(2.0)
        net.w_sf_fixed.fill_(2.0)

    l4 = tuned.l4_code(torch.tensor([0.0], device=DEVICE))
    fb = torch.zeros(1, simple.N, device=DEVICE)
    fb[:, 0] = 1.0
    _, internals = net.l23(l4, fb, return_internals=True)
    _, _, _, pre_pv_rate, _, _, som_b, som_p = internals

    drive = net.feedforward(l4)
    basal = drive / (1.0 + net.m_fixed_effective() * som_b)
    modulation = (
        net.circuit_gains()[tuned.CIRC_INDEX["w_ef"]] * fb
        - net.m_fixed_effective() * som_p
    )

    assert fb[0, 0] > 0.0
    assert som_p[0, 0] > 0.0
    assert modulation[0, 0] < 0.0
    assert pre_pv_rate[0, 0] < basal[0, 0]


def test_sst_activity_uses_equal_mass_mean_across_both_pools() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    l4 = tuned.l4_code(torch.tensor([0.0], device=DEVICE))
    fb = torch.zeros(1, simple.N, device=DEVICE)
    fb[:, 0] = 1.0

    _, internals = net.l23(l4, fb, return_internals=True)
    som, _, _, _, _, _, som_b, som_p = internals
    explicit_72_unit_mean = torch.cat((som_b, som_p), dim=-1).mean()
    activity = train_sweep.modeled_population_activity_components(
        torch.zeros_like(som),
        som,
        torch.zeros(1, tuned.N_POP, device=DEVICE),
        torch.zeros(1, 1, device=DEVICE),
    )

    assert torch.allclose(som, 0.5 * (som_b + som_p))
    assert torch.allclose(som.mean(), explicit_72_unit_mean)
    assert torch.allclose(
        activity["modeled_population_activity_som"],
        explicit_72_unit_mean,
    )
    assert torch.allclose(
        activity["modeled_population_activity_numerator"],
        train_sweep.MODELED_ACTIVITY_WEIGHTS["som"] * explicit_72_unit_mean,
    )


def test_vip_is_higher_for_topographic_mismatch_than_aligned_prediction() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    for name, value in {
        "w_ef": 0.0,
        "theta_S": 0.0,
        "w_vd": 1.0,
        "w_sd": 0.0,
        "w_vs": 4.0,
        "w_sv": 0.0,
        "theta_V": 0.0,
        "w_pv": 0.0,
    }.items():
        set_gain(net, name, value)
    with torch.no_grad():
        net.m_fixed.zero_()

    l4 = tuned.l4_code(torch.tensor([0.0], device=DEVICE))
    aligned = torch.zeros(1, simple.N, device=DEVICE)
    mismatch = torch.zeros(1, simple.N, device=DEVICE)
    aligned[:, 0] = 1.0
    mismatch[:, simple.N // 2] = 1.0

    _, aligned_internals = net.l23(l4, aligned, return_internals=True)
    _, mismatch_internals = net.l23(l4, mismatch, return_internals=True)
    aligned_vip = aligned_internals[1]
    mismatch_vip = mismatch_internals[1]
    driven_pop = (net.feedforward(l4) @ net.A_in.t()).argmax(dim=-1)

    assert (
        mismatch_vip[0, driven_pop.item()] - aligned_vip[0, driven_pop.item()]
    ).item() > 1e-4


def test_axis_optimizer_freezes_shared_anatomy_and_trains_w_sf() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    train_sweep.enforce_fixed_vip_motif(net)
    params = train_sweep.set_axis_parameter_policy(net, freeze_local_comp=True)
    optimizer = torch.optim.Adam(params, lr=0.05)

    trainable_ids = {id(parameter) for parameter in params}
    expected_trainable_ids = {
        id(parameter)
        for parameter in list(net.gru.parameters()) + list(net.W_fb.parameters())
    }
    expected_trainable_ids.add(id(net.w_sf_fixed))
    assert trainable_ids == expected_trainable_ids
    assert {id(parameter) for parameter in net.parameters() if parameter.requires_grad} == (
        expected_trainable_ids
    )
    assert net.circ_raw.requires_grad is False
    assert net.m_fixed.requires_grad is False

    before_circ = net.circ_raw.detach().clone()
    before_m = net.m_fixed.detach().clone()
    before_a_in = net.A_in.detach().clone()
    before_a_out = net.A_out.detach().clone()
    before_a_ss = net.A_ss.detach().clone()
    before_k_pred = net.K_pred.detach().clone()
    before_w_sf = net.w_sf_fixed.detach().clone()

    loss = net.w_sf_fixed.square()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert net.w_sf_fixed.grad is not None
    assert net.w_sf_fixed.grad != 0
    torch.nn.utils.clip_grad_norm_(params, 0.01)
    optimizer.step()

    assert not torch.equal(net.w_sf_fixed.detach(), before_w_sf)
    assert torch.equal(net.circ_raw.detach(), before_circ)
    assert torch.equal(net.m_fixed.detach(), before_m)
    assert torch.equal(net.A_in.detach(), before_a_in)
    assert torch.equal(net.A_out.detach(), before_a_out)
    assert torch.equal(net.A_ss.detach(), before_a_ss)
    assert torch.equal(net.K_pred.detach(), before_k_pred)


def test_confidence_current_ce_is_chance_for_zero_response() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    references = {"R_ref": 1.0, "sigma_train": 0.0}
    labels = torch.tensor([[0, 7]], device=DEVICE)
    zero_rates = torch.zeros(1, 2, simple.N, device=DEVICE)

    zero = train_sweep.confidence_weighted_current_orientation_ce(
        net,
        zero_rates,
        labels,
        make_test_generator(1),
        references,
    )

    chance = torch.tensor(math.log(simple.N), device=DEVICE)
    assert torch.allclose(zero["current_ce"], chance, atol=1e-6)


def test_confidence_current_ce_prefers_correct_aligned_sparse_response() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    references = {"R_ref": 1.0, "sigma_train": 0.0}
    rates = torch.zeros(1, 1, simple.N, device=DEVICE)
    rates[:, :, 0] = 10.0
    correct = torch.tensor([[0]], device=DEVICE)
    wrong = torch.tensor([[18]], device=DEVICE)

    correct_loss = train_sweep.confidence_weighted_current_orientation_ce(
        net,
        rates,
        correct,
        make_test_generator(2),
        references,
    )
    wrong_loss = train_sweep.confidence_weighted_current_orientation_ce(
        net,
        rates,
        wrong,
        make_test_generator(2),
        references,
    )

    assert correct_loss["current_ce"] < wrong_loss["current_ce"]


def test_confidence_current_ce_improves_monotonically_with_aligned_scale() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    references = {"R_ref": 1.0, "sigma_train": 0.01}
    labels = torch.tensor([[0]], device=DEVICE)

    losses = []
    confidences = []
    for scale in (0.05, 0.2, 1.0):
        rates = torch.zeros(1, 1, simple.N, device=DEVICE)
        rates[:, :, 0] = scale
        current = train_sweep.confidence_weighted_current_orientation_ce(
            net,
            rates,
            labels,
            make_test_generator(3),
            references,
        )
        losses.append(current["current_ce"])
        confidences.append(current["current_confidence"])

    assert losses[0] > losses[1] > losses[2]
    assert confidences[0] < confidences[1] < confidences[2]


def test_modeled_population_activity_uses_exact_class_fractions() -> None:
    final_e = torch.full((1, 1, simple.N), 6.0, device=DEVICE)
    som = torch.full((1, 1, simple.N), 3.0, device=DEVICE)
    vip = torch.full((1, 1, tuned.N_POP), 9.0, device=DEVICE)
    pv_scalar = torch.full((1, 1, 1), 12.0, device=DEVICE)
    som_gain = torch.full_like(som, 100.0)
    exc_feedback_work = torch.full_like(som, 200.0)

    activity = train_sweep.modeled_population_activity_components(
        final_e,
        som,
        vip,
        pv_scalar,
        som_gain,
        exc_feedback_work,
    )
    expected = (
        (5.0 / 6.0) * final_e.mean()
        + (37.0 / 480.0) * pv_scalar.mean()
        + (1.0 / 20.0) * som.mean()
        + (19.0 / 480.0) * vip.mean()
    )

    assert train_sweep.MODELED_ACTIVITY_WEIGHTS == {
        "final_e": 5.0 / 6.0,
        "pv": 37.0 / 480.0,
        "som": 1.0 / 20.0,
        "vip": 19.0 / 480.0,
    }
    assert torch.allclose(
        activity["modeled_population_activity_final_e"],
        final_e.mean(),
    )
    assert torch.allclose(activity["modeled_population_activity_pv"], pv_scalar.mean())
    assert torch.allclose(activity["modeled_population_activity_som"], som.mean())
    assert torch.allclose(activity["modeled_population_activity_vip"], vip.mean())
    assert torch.allclose(
        activity["modeled_population_activity_numerator"],
        expected,
    )
    assert torch.allclose(
        activity["modeled_population_activity_som_gain_report"],
        som_gain.mean(),
    )
    assert torch.allclose(
        activity["modeled_population_activity_exc_feedback_report"],
        exc_feedback_work.mean(),
    )


def test_modeled_population_activity_is_basis_size_independent() -> None:
    final_e_36 = torch.full((1, 1, 36), 2.0, device=DEVICE)
    final_e_72 = torch.full((1, 1, 72), 2.0, device=DEVICE)
    som_36 = torch.full((1, 1, 36), 5.0, device=DEVICE)
    som_72 = torch.full((1, 1, 72), 5.0, device=DEVICE)
    vip_9 = torch.full((1, 1, 9), 7.0, device=DEVICE)
    vip_18 = torch.full((1, 1, 18), 7.0, device=DEVICE)
    pv = torch.full((1, 1, 1), 11.0, device=DEVICE)

    activity_36 = train_sweep.modeled_population_activity_components(
        final_e_36,
        som_36,
        vip_9,
        pv,
    )
    activity_72 = train_sweep.modeled_population_activity_components(
        final_e_72,
        som_72,
        vip_18,
        pv.expand(1, 1, 72),
    )

    assert torch.allclose(
        activity_36["modeled_population_activity_numerator"],
        activity_72["modeled_population_activity_numerator"],
    )


def test_modeled_population_activity_uses_actual_pv_scalar_not_pv_work() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    set_gain(net, "w_pv", 2.0)
    with torch.no_grad():
        net.m_fixed.zero_()
    l4 = torch.ones(2, simple.N, device=DEVICE)

    rates, internals = net.l23(
        l4,
        torch.zeros(2, simple.N, device=DEVICE),
        return_internals=True,
    )
    som, vip, som_gain, pre_pv_rate, post_pv_rate, exc_feedback_work, _, _ = (
        internals
    )
    pv_scalar = train_sweep.pv_scalar_from_pre_pv(net, pre_pv_rate)
    activity = train_sweep.modeled_population_activity_components(
        rates,
        som,
        vip,
        pv_scalar,
        som_gain,
        exc_feedback_work,
    )
    expected_pv = (
        net.circuit_gains()[tuned.CIRC_INDEX["w_pv"]]
        * pre_pv_rate.mean(dim=-1, keepdim=True)
    )
    old_pv_work = (pre_pv_rate - post_pv_rate).mean()

    assert torch.allclose(pv_scalar, expected_pv)
    assert torch.allclose(
        activity["modeled_population_activity_pv"],
        expected_pv.mean(),
    )
    assert not torch.allclose(
        activity["modeled_population_activity_pv"],
        old_pv_work,
    )


def test_som_gain_and_exc_feedback_work_do_not_enter_activity_objective() -> None:
    final_e = torch.full((1, 1, simple.N), 1.5, device=DEVICE)
    som = torch.full((1, 1, simple.N), 2.5, device=DEVICE)
    vip = torch.full((1, 1, tuned.N_POP), 3.5, device=DEVICE)
    pv_scalar = torch.full((1, 1, 1), 4.5, device=DEVICE)

    low_reports = train_sweep.modeled_population_activity_components(
        final_e,
        som,
        vip,
        pv_scalar,
        torch.zeros_like(som),
        torch.zeros_like(som),
    )
    high_reports = train_sweep.modeled_population_activity_components(
        final_e,
        som,
        vip,
        pv_scalar,
        torch.full_like(som, 1000.0),
        torch.full_like(som, 2000.0),
    )

    assert torch.allclose(
        low_reports["modeled_population_activity_numerator"],
        high_reports["modeled_population_activity_numerator"],
    )
    assert not torch.allclose(
        low_reports["modeled_population_activity_som_gain_report"],
        high_reports["modeled_population_activity_som_gain_report"],
    )
    assert not torch.allclose(
        low_reports["modeled_population_activity_exc_feedback_report"],
        high_reports["modeled_population_activity_exc_feedback_report"],
    )


def test_task_loss_uses_one_reference_for_modeled_population_activity() -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(123)
    theta = torch.tensor([[0.0, 5.0, 10.0]], device=DEVICE)
    channels = (theta / simple.STEP_DEG).long()
    references = train_sweep.reference_values(net, DEVICE)

    losses = train_sweep.task_activity_losses(
        net,
        theta,
        channels,
        generator,
        references,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
    )

    assert references["R_ref"] == references["modeled_population_activity_ref"]
    _, rates, _ = tuned.forward_seq_tuned(
        net,
        theta,
        1.0,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
        return_internals=True,
    )
    assert torch.allclose(
        losses["modeled_population_activity_final_e"],
        rates.mean(),
    )
    assert torch.allclose(
        losses["modeled_population_activity"],
        losses["modeled_population_activity_numerator"] / references["R_ref"],
    )
    assert torch.allclose(losses["energy"], losses["modeled_population_activity"])


def test_modeled_activity_is_prediction_independent_for_fixed_population_rates(
    monkeypatch,
) -> None:
    net = tuned.build_tuned_from_config(train_sweep.MODEL_CONFIG).to(DEVICE)
    theta = torch.tensor([[0.0, 5.0, 10.0]], device=DEVICE)
    channels = (theta / simple.STEP_DEG).long()
    references = train_sweep.reference_values(net, DEVICE)
    _, fixed_rates, fixed_internals = tuned.forward_seq_tuned(
        net,
        theta,
        1.0,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
        return_internals=True,
    )
    expected_zero = torch.full_like(fixed_rates, -5.0)
    expected_zero[..., 0] = 5.0
    expected_orthogonal = torch.full_like(fixed_rates, -5.0)
    expected_orthogonal[..., simple.N // 2] = 5.0
    active_predictions = [expected_zero]

    def fixed_population_forward(*_args, **_kwargs):
        return active_predictions[0], fixed_rates, fixed_internals

    monkeypatch.setattr(tuned, "forward_seq_tuned", fixed_population_forward)
    zero_activity = train_sweep.task_activity_losses(
        net,
        theta,
        channels,
        make_test_generator(123),
        references,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
    )
    active_predictions[0] = expected_orthogonal
    orthogonal_activity = train_sweep.task_activity_losses(
        net,
        theta,
        channels,
        make_test_generator(123),
        references,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
    )

    assert not torch.equal(expected_zero, expected_orthogonal)
    assert torch.allclose(
        zero_activity["modeled_population_activity_final_e"],
        fixed_rates.mean(),
    )
    assert torch.allclose(
        orthogonal_activity["modeled_population_activity_final_e"],
        fixed_rates.mean(),
    )
    assert torch.allclose(
        zero_activity["modeled_population_activity_numerator"],
        orthogonal_activity["modeled_population_activity_numerator"],
    )
    assert torch.allclose(
        zero_activity["modeled_population_activity"],
        orthogonal_activity["modeled_population_activity"],
    )


def test_constrained_pair_reuses_batch_and_noise_and_detaches_reference(
    monkeypatch,
) -> None:
    candidate_net = nn.Linear(1, 1).to(DEVICE)
    reference_net = nn.Linear(1, 1).to(DEVICE)
    theta = torch.zeros(2, 3, device=DEVICE)
    channels = torch.zeros(2, 3, dtype=torch.long, device=DEVICE)
    generator = make_test_generator(101)
    references = {"sigma_train": 0.25}
    calls = []

    def fake_task_activity_losses(
        net,
        observed_theta,
        observed_channels,
        observed_generator,
        observed_references,
        **kwargs,
    ):
        calls.append(
            {
                "theta": observed_theta,
                "channels": observed_channels,
                "generator": observed_generator,
                "references": observed_references,
                "noise": kwargs["current_decoder_noise"],
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        anchor = next(net.parameters()).sum()
        return {
            "next_ce": anchor + 1.0,
            "current_ce_normalized": anchor + 2.0,
        }

    monkeypatch.setattr(
        train_sweep,
        "task_activity_losses",
        fake_task_activity_losses,
    )
    candidate_losses, reference_losses, noise = (
        train_sweep.paired_constrained_task_losses(
            candidate_net,
            reference_net,
            theta,
            channels,
            generator,
            references,
        )
    )

    assert len(calls) == 2
    assert calls[0]["theta"] is calls[1]["theta"] is theta
    assert calls[0]["channels"] is calls[1]["channels"] is channels
    assert calls[0]["noise"] is calls[1]["noise"] is noise
    assert calls[0]["generator"] is calls[1]["generator"] is generator
    assert calls[0]["references"] is calls[1]["references"] is references
    assert calls[0]["grad_enabled"] is True
    assert calls[1]["grad_enabled"] is False
    assert candidate_losses["next_ce"].requires_grad
    assert all(not value.requires_grad for value in reference_losses.values())


def test_constrained_objective_has_two_detached_generic_constraints_only() -> None:
    log_n = math.log(simple.N)
    activity = torch.tensor(0.7, device=DEVICE, requires_grad=True)
    candidate_next = torch.tensor(
        2.0 * log_n,
        device=DEVICE,
        requires_grad=True,
    )
    candidate_current = torch.tensor(0.4, device=DEVICE, requires_grad=True)
    reference_next = torch.tensor(
        1.0 * log_n,
        device=DEVICE,
        requires_grad=True,
    )
    reference_current = torch.tensor(0.5, device=DEVICE, requires_grad=True)
    terms = train_sweep.constrained_objective_terms(
        {
            "modeled_population_activity": activity,
            "next_ce": candidate_next,
            "current_ce_normalized": candidate_current,
        },
        {
            "next_ce": reference_next,
            "current_ce_normalized": reference_current,
        },
        torch.tensor(0.2, device=DEVICE),
        torch.tensor(0.3, device=DEVICE),
    )

    assert set(terms) == {
        "objective",
        "constraint_next",
        "constraint_current",
        "candidate_next",
        "reference_next",
        "candidate_current",
        "reference_current",
    }
    assert torch.isclose(terms["constraint_next"], torch.tensor(1.0, device=DEVICE))
    assert torch.isclose(
        terms["constraint_current"],
        torch.tensor(-0.1, device=DEVICE),
    )
    assert torch.isclose(terms["objective"], torch.tensor(0.87, device=DEVICE))
    terms["objective"].backward()
    assert activity.grad is not None
    assert candidate_next.grad is not None
    assert candidate_current.grad is not None
    assert reference_next.grad is None
    assert reference_current.grad is None

    source = inspect.getsource(train_sweep.constrained_objective_terms).lower()
    for forbidden in (
        "expected",
        "unexpected",
        "prediction",
        "channel",
        "center",
        "flank",
        "curve",
        "interneuron",
        'candidate_losses["task"]',
    ):
        assert forbidden not in source


def test_projected_duals_update_separately_after_primal_step() -> None:
    lambda_next, lambda_current = train_sweep.projected_dual_ascent(
        torch.tensor(0.0, device=DEVICE),
        torch.tensor(0.0005, device=DEVICE),
        torch.tensor(2.0, device=DEVICE, requires_grad=True),
        torch.tensor(-1.0, device=DEVICE, requires_grad=True),
    )

    assert torch.isclose(lambda_next, torch.tensor(0.002, device=DEVICE))
    assert torch.isclose(lambda_current, torch.tensor(0.0, device=DEVICE))
    assert not lambda_next.requires_grad
    assert not lambda_current.requires_grad
    source = inspect.getsource(train_sweep.run_constrained_efficient_coding)
    assert source.index("optimizer.step()") < source.index("projected_dual_ascent(")


def make_constrained_resume_checkpoint(
    *,
    step: int = 16000,
    target_steps: int = 16000,
) -> tuple[dict, dict[str, float], str]:
    references = {"R_ref": 1.25, "sigma_train": 0.1}
    common_state_hash = "a" * 64
    data_generator = make_cpu_generator(201)
    noise_generator = make_cpu_generator(202)
    state_dict = {"weight": torch.tensor([1.0], device=DEVICE)}
    optimizer_state = {
        "state": {
            0: {
                "step": torch.tensor(3.0, device=DEVICE),
                "exp_avg": torch.tensor([0.2], device=DEVICE),
                "exp_avg_sq": torch.tensor([0.04], device=DEVICE),
            }
        },
        "param_groups": [{"params": [0], "lr": 1e-3}],
    }
    saved = {
        "stage": "constrained_efficient_coding",
        "step": step,
        "target_steps": target_steps,
        "seed": 8,
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer_state,
        "data_generator_state": data_generator.get_state(),
        "noise_generator_state": noise_generator.get_state(),
        "lambda_next": 0.08,
        "lambda_current": 0.72,
        "references": references,
        "tuned_net_config": copy.deepcopy(train_sweep.MODEL_CONFIG),
        "model_architecture_version": tuned.MODEL_ARCHITECTURE_VERSION,
        "training_compatibility_version": (
            train_sweep.TRAINING_COMPATIBILITY_VERSION
        ),
        "fixed_canonical_vip_motif_gains": copy.deepcopy(
            train_sweep.FIXED_CANONICAL_VIP_MOTIF_GAINS
        ),
        "common_state_sha256": common_state_hash,
        "dual_step_size": train_sweep.CONSTRAINED_DUAL_STEP_SIZE,
        "freeze_local_comp": True,
        "center_feedback": False,
        "feedback_mode": tuned.FEEDBACK_MODE_POSTERIOR,
        "mismatch_prob": 0.02,
        "mismatch_stats": {
            "events": 307579,
            "eligible": 15379676,
            "transitions": 20480000,
        },
        "data_generator_backend": "cpu",
        "noise_generator_backend": "cpu",
    }
    saved.update(
        train_sweep.constrained_checkpoint_integrity(
            common_state_hash,
            references,
            "cpu",
            "cpu",
        )
    )
    saved.update(
        {
            "candidate_state_sha256": train_sweep.state_sha256(state_dict),
            "data_generator_state_sha256": train_sweep.tensor_sha256(
                saved["data_generator_state"]
            ),
            "noise_generator_state_sha256": train_sweep.tensor_sha256(
                saved["noise_generator_state"]
            ),
        }
    )
    return saved, references, common_state_hash


def test_constrained_resume_allows_only_monotonic_target_extension(tmp_path) -> None:
    saved, references, common_state_hash = make_constrained_resume_checkpoint()
    data_generator = make_cpu_generator(401)
    noise_generator = make_cpu_generator(402)
    validated = train_sweep.validate_constrained_resume_checkpoint(
        saved,
        requested_target_steps=32000,
        seed=8,
        freeze_local_comp=True,
        feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
        mismatch_prob=0.02,
        common_state_hash=common_state_hash,
        references=references,
        data_generator=data_generator,
        noise_generator=noise_generator,
        legacy_training_log_path=tmp_path / "unused.jsonl",
    )

    assert validated["step"] == 16000
    assert validated["saved_target_steps"] == 16000
    assert validated["mismatch_stats"] == saved["mismatch_stats"]
    assert validated["generator_backends"] == {"data": "cpu", "noise": "cpu"}
    assert validated["legacy"] is False
    with pytest.raises(RuntimeError, match="not monotonic"):
        train_sweep.validate_constrained_resume_checkpoint(
            saved,
            requested_target_steps=15999,
            seed=8,
            freeze_local_comp=True,
            feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
            mismatch_prob=0.02,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=data_generator,
            noise_generator=noise_generator,
            legacy_training_log_path=tmp_path / "unused.jsonl",
        )
    incomplete = copy.deepcopy(saved)
    del incomplete["optimizer_state_dict"]
    with pytest.raises(RuntimeError, match="incomplete"):
        train_sweep.validate_constrained_resume_checkpoint(
            incomplete,
            requested_target_steps=32000,
            seed=8,
            freeze_local_comp=True,
            feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
            mismatch_prob=0.02,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=data_generator,
            noise_generator=noise_generator,
            legacy_training_log_path=tmp_path / "unused.jsonl",
        )
    incomplete_new_format = copy.deepcopy(saved)
    del incomplete_new_format["mismatch_stats"]
    with pytest.raises(RuntimeError, match="incomplete"):
        train_sweep.validate_constrained_resume_checkpoint(
            incomplete_new_format,
            requested_target_steps=32000,
            seed=8,
            freeze_local_comp=True,
            feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
            mismatch_prob=0.02,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=data_generator,
            noise_generator=noise_generator,
            legacy_training_log_path=tmp_path / "unused.jsonl",
        )
    cross_backend = copy.deepcopy(saved)
    cross_backend["data_generator_backend"] = "cuda"
    cross_backend["noise_generator_backend"] = "cuda"
    cross_backend["generator_backends_sha256"] = train_sweep.canonical_json_sha256(
        {"data": "cuda", "noise": "cuda"}
    )
    with pytest.raises(RuntimeError, match="backend mismatch"):
        train_sweep.validate_constrained_resume_checkpoint(
            cross_backend,
            requested_target_steps=32000,
            seed=8,
            freeze_local_comp=True,
            feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
            mismatch_prob=0.02,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=data_generator,
            noise_generator=noise_generator,
            legacy_training_log_path=tmp_path / "unused.jsonl",
        )

    latest_path = tmp_path / "constrained_efficient_coding_latest.pt"
    final_path = tmp_path / "constrained_efficient_coding_final.pt"
    torch.save({"step": 16001}, latest_path)
    torch.save({"step": 16000}, final_path)
    selected_path, selected = train_sweep.load_constrained_resume_checkpoint(
        latest_path,
        final_path,
        DEVICE,
    )
    assert selected_path == latest_path
    assert selected["step"] == 16001


def test_constrained_resume_recovers_only_exact_legacy_counters(tmp_path) -> None:
    saved, references, common_state_hash = make_constrained_resume_checkpoint()
    for name in (
        "mismatch_stats",
        "frozen_theta0_sha256",
        "reference_values_sha256",
        "model_config_sha256",
        "anatomy_sha256",
        "generator_backends_sha256",
        "candidate_state_sha256",
        "data_generator_state_sha256",
        "noise_generator_state_sha256",
        "data_generator_backend",
        "noise_generator_backend",
    ):
        del saved[name]
    saved["data_generator_state"] = torch.zeros(16, dtype=torch.uint8)
    saved["noise_generator_state"] = torch.zeros(16, dtype=torch.uint8)
    run_start = {
        "event": "run_start",
        "training_mode": "constrained_efficient_coding",
        "seed": saved["seed"],
        "axis_steps": saved["target_steps"],
        "feedback_mode": saved["feedback_mode"],
        "freeze_local_comp": saved["freeze_local_comp"],
        "model_config": saved["tuned_net_config"],
        "fixed_canonical_vip_motif_gains": (
            saved["fixed_canonical_vip_motif_gains"]
        ),
        "training_compatibility_version": (
            saved["training_compatibility_version"]
        ),
    }
    step_event = {
        "event": "constrained_step",
        "step": saved["step"],
        "mismatch_prob": saved["mismatch_prob"],
        "lambda_next": saved["lambda_next"],
        "lambda_current": saved["lambda_current"],
        "mismatch_events": 307579,
        "mismatch_eligible": 15379676,
        "mismatch_transitions": 20480000,
    }
    log_path = tmp_path / "training.jsonl"
    log_path.write_text(
        json.dumps(run_start) + "\n" + json.dumps(step_event) + "\n",
        encoding="utf-8",
    )

    recovered = train_sweep.recover_legacy_constrained_mismatch_stats(
        saved,
        log_path,
    )
    assert recovered == {
        "events": 307579,
        "eligible": 15379676,
        "transitions": 20480000,
    }
    with pytest.raises(RuntimeError, match="CUDA-only"):
        train_sweep.validate_constrained_resume_checkpoint(
            saved,
            requested_target_steps=32000,
            seed=8,
            freeze_local_comp=True,
            feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
            mismatch_prob=0.02,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=make_cpu_generator(501),
            noise_generator=make_cpu_generator(502),
            legacy_training_log_path=log_path,
        )
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        cuda_data_generator = train_sweep.make_generator(cuda_device, 501)
        cuda_noise_generator = train_sweep.make_generator(cuda_device, 502)
        saved["data_generator_state"] = cuda_data_generator.get_state()
        saved["noise_generator_state"] = cuda_noise_generator.get_state()
        validated = train_sweep.validate_constrained_resume_checkpoint(
            saved,
            requested_target_steps=32000,
            seed=8,
            freeze_local_comp=True,
            feedback_mode=tuned.FEEDBACK_MODE_POSTERIOR,
            mismatch_prob=0.02,
            common_state_hash=common_state_hash,
            references=references,
            data_generator=cuda_data_generator,
            noise_generator=cuda_noise_generator,
            legacy_training_log_path=log_path,
        )
        assert validated["legacy"] is True

    step_event["lambda_current"] = 0.71
    log_path.write_text(
        json.dumps(run_start) + "\n" + json.dumps(step_event) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="one exact run/step match"):
        train_sweep.recover_legacy_constrained_mismatch_stats(
            saved,
            log_path,
        )


def test_constrained_resume_restores_bit_exact_first_update() -> None:
    source_net = nn.Linear(2, 1, bias=False).to(DEVICE)
    with torch.no_grad():
        source_net.weight.copy_(torch.tensor([[0.25, -0.5]], device=DEVICE))
    source_optimizer = torch.optim.Adam(source_net.parameters(), lr=1e-3)
    initial_x = torch.tensor([[1.0, 2.0]], device=DEVICE)
    source_optimizer.zero_grad(set_to_none=True)
    source_net(initial_x).square().mean().backward()
    source_optimizer.step()
    source_data_generator = make_test_generator(301)
    source_noise_generator = make_test_generator(302)
    saved = {
        "state_dict": copy.deepcopy(source_net.state_dict()),
        "optimizer_state_dict": copy.deepcopy(source_optimizer.state_dict()),
        "data_generator_state": source_data_generator.get_state(),
        "noise_generator_state": source_noise_generator.get_state(),
        "lambda_next": 0.08,
        "lambda_current": 0.72,
    }
    restored_net = nn.Linear(2, 1, bias=False).to(DEVICE)
    restored_optimizer = torch.optim.Adam(restored_net.parameters(), lr=1e-3)
    restored_data_generator = make_test_generator(999)
    restored_noise_generator = make_test_generator(998)
    restored_stats = {"events": 7, "eligible": 11, "transitions": 13}
    target_stats = {"events": 0, "eligible": 0, "transitions": 0}

    lambda_next, lambda_current = (
        train_sweep.restore_constrained_training_state(
            saved,
            restored_net,
            restored_optimizer,
            restored_data_generator,
            restored_noise_generator,
            restored_stats,
            target_stats,
            DEVICE,
        )
    )
    assert train_sweep.state_sha256(restored_net.state_dict()) == (
        train_sweep.state_sha256(source_net.state_dict())
    )
    assert torch.equal(
        restored_data_generator.get_state(),
        source_data_generator.get_state(),
    )
    assert torch.equal(
        restored_noise_generator.get_state(),
        source_noise_generator.get_state(),
    )
    assert target_stats == restored_stats
    assert lambda_next.item() == pytest.approx(0.08)
    assert lambda_current.item() == pytest.approx(0.72)

    source_x = torch.randn(
        4,
        2,
        device=DEVICE,
        generator=source_data_generator,
    )
    restored_x = torch.randn(
        4,
        2,
        device=DEVICE,
        generator=restored_data_generator,
    )
    source_noise = torch.randn(
        4,
        1,
        device=DEVICE,
        generator=source_noise_generator,
    )
    restored_noise = torch.randn(
        4,
        1,
        device=DEVICE,
        generator=restored_noise_generator,
    )
    assert torch.equal(source_x, restored_x)
    assert torch.equal(source_noise, restored_noise)
    for net, optimizer, x, noise in (
        (source_net, source_optimizer, source_x, source_noise),
        (restored_net, restored_optimizer, restored_x, restored_noise),
    ):
        optimizer.zero_grad(set_to_none=True)
        (net(x) + noise).square().mean().backward()
        optimizer.step()
    assert train_sweep.state_sha256(restored_net.state_dict()) == (
        train_sweep.state_sha256(source_net.state_dict())
    )
    source_state = source_optimizer.state_dict()["state"]
    restored_state = restored_optimizer.state_dict()["state"]
    assert source_state.keys() == restored_state.keys()
    for parameter_id in source_state:
        assert source_state[parameter_id].keys() == restored_state[parameter_id].keys()
        for name in source_state[parameter_id]:
            assert torch.equal(
                source_state[parameter_id][name],
                restored_state[parameter_id][name],
            )


def test_vip_targets_both_sst_pools_through_its_local_footprint() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    for name, value in {
        "w_ef": 0.0,
        "theta_S": 0.0,
        "w_vd": 1.0,
        "w_sd": 1.0,
        "w_vs": 0.0,
        "w_sv": 1.0,
        "theta_V": 0.0,
        "w_pv": 0.0,
    }.items():
        set_gain(net, name, value)
    with torch.no_grad():
        net.m_fixed.zero_()

    l4 = tuned.l4_code(torch.tensor([0.0], device=DEVICE))
    fb = torch.zeros(1, simple.N, device=DEVICE)
    fb[:, 0] = 1.0
    rate, internals = net.l23(
        l4,
        fb,
        return_internals=True,
    )
    som, vip, _, _, _, _, som_b, som_p = internals
    g = net.circuit_gains()
    drive = net.feedforward(l4)
    b9 = drive @ net.A_in.t()
    b36 = b9 @ net.A_out.t()
    s_ff = torch.relu(
        net.w_sf_effective() * (fb @ net.A_in.t())
        - g[tuned.CIRC_INDEX["theta_S"]]
    )
    vip_expected = torch.relu(
        g[tuned.CIRC_INDEX["w_vd"]] * b9
        - g[tuned.CIRC_INDEX["w_vs"]] * (s_ff @ net.A_ss.t())
        - g[tuned.CIRC_INDEX["theta_V"]]
    )
    v36 = vip_expected @ net.A_out.t()
    q_b = g[tuned.CIRC_INDEX["w_sd"]] * b36
    q_p = net.w_sf_effective() * drive * (fb @ net.K_pred.t())
    som_b_local = torch.relu(
        q_b
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * v36
    )
    som_p_local = torch.relu(
        q_p
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * v36
    )
    som_b_global = torch.relu(
        q_b
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * vip_expected.mean(dim=-1, keepdim=True)
    )
    som_p_global = torch.relu(
        q_p
        - g[tuned.CIRC_INDEX["theta_S"]]
        - g[tuned.CIRC_INDEX["w_sv"]] * vip_expected.mean(dim=-1, keepdim=True)
    )

    set_gain(net, "w_sv", 0.0)
    no_vip_target_rate, no_vip_target_internals = net.l23(
        l4,
        fb,
        return_internals=True,
    )
    no_vip_target_som_b = no_vip_target_internals[6]
    no_vip_target_som_p = no_vip_target_internals[7]

    assert torch.allclose(vip, vip_expected)
    assert torch.allclose(vip, no_vip_target_internals[1])
    assert torch.allclose(som_b, som_b_local)
    assert torch.allclose(som_p, som_p_local)
    assert torch.allclose(som, 0.5 * (som_b_local + som_p_local))
    assert torch.all(som_b <= no_vip_target_som_b)
    assert torch.all(som_p <= no_vip_target_som_p)
    assert (no_vip_target_som_b - som_b).max().item() > 1e-4
    assert (no_vip_target_som_p - som_p).max().item() > 1e-4
    assert (som_b_local - som_b_global).abs().max().item() > 1e-4
    assert (som_p_local - som_p_global).abs().max().item() > 1e-4
    assert torch.allclose(internals[3], no_vip_target_internals[3])
    assert torch.allclose(rate, no_vip_target_rate)


def test_pv_is_broad_activity_driven_divisor() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
    ).to(DEVICE)
    set_gain(net, "w_pv", 0.5)
    with torch.no_grad():
        net.m_fixed.zero_()
    l4 = torch.ones(2, simple.N, device=DEVICE)

    _, internals = net.l23(
        l4,
        torch.zeros(2, simple.N, device=DEVICE),
        return_internals=True,
    )
    _, _, _, pre_pv_rate, post_pv_rate, exc_feedback_work, _, _ = internals
    inferred_divisor = pre_pv_rate / post_pv_rate - 1.0

    assert torch.all(pre_pv_rate > 0)
    assert torch.count_nonzero(exc_feedback_work).item() == 0
    assert torch.allclose(
        inferred_divisor,
        inferred_divisor.mean(dim=-1, keepdim=True).expand_as(inferred_divisor),
        atol=1e-6,
    )


def test_forward_seq_feeds_final_e_rate_to_native_rnn() -> None:
    net = tuned.SimpleTunedNet(
        local_comp_strength=0.0,
        local_comp_trainable=False,
        recurrent_cell="rnn_tanh",
    ).to(DEVICE)
    set_gain(net, "w_pv", 1.0)
    theta = torch.tensor([[0.0, 5.0]], device=DEVICE)
    seen_inputs: list[torch.Tensor] = []

    def capture_input(_module, args):
        seen_inputs.append(args[0].detach().clone())

    handle = net.gru.register_forward_pre_hook(capture_input)
    try:
        _, rates, internals = tuned.forward_seq_tuned(
            net,
            theta,
            return_internals=True,
        )
    finally:
        handle.remove()

    pre_pv_rate = internals[3]
    post_pv_rate = internals[4]
    assert isinstance(net.gru, nn.RNNCell)
    assert len(seen_inputs) == theta.shape[1]
    assert torch.allclose(torch.stack(seen_inputs, dim=1), rates)
    assert torch.allclose(rates, post_pv_rate)
    assert not torch.allclose(rates, pre_pv_rate)


def test_reproduce_figures_does_not_write_curves_json_on_failures(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    figure_root = tmp_path / "figures"
    checkpoint_path = checkpoint_root / "seed8" / "alpha0p0" / "alpha_0p0_final.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model_architecture_version": "incompatible",
            "tuned_net_config": {},
            "state_dict": {},
            "references": {"R_ref": 1.0},
        },
        checkpoint_path,
    )

    monkeypatch.setattr(reproduce_figures, "CKPT", checkpoint_root)
    monkeypatch.setattr(reproduce_figures, "FIGS", figure_root)
    monkeypatch.setattr(reproduce_figures, "SEEDS", (8,))
    monkeypatch.setattr(
        reproduce_figures,
        "ARMS",
        (("alpha0p0", "0p0", "task optimized", "sharpening", "#2b7bb9"),),
    )

    assert reproduce_figures.main() == 1
    assert not (figure_root / "c6_curves.json").exists()
