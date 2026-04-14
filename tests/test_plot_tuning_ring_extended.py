from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.state import NetworkState


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_tuning_ring_extended.py"
)
SPEC = importlib.util.spec_from_file_location("plot_tuning_ring_extended", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_trial_response_raw_delta_and_baseline():
    r_l23_all = torch.tensor(
        [
            [
                [0.0, 0.5, 1.0],
                [1.0, 1.5, 2.0],
                [3.0, 4.0, 5.0],
            ]
        ],
        dtype=torch.float32,
    )

    raw = MODULE.extract_trial_response(
        r_l23_all=r_l23_all,
        t_readout=2,
        t_isi_last=1,
        response_mode="raw",
    )
    delta = MODULE.extract_trial_response(
        r_l23_all=r_l23_all,
        t_readout=2,
        t_isi_last=1,
        response_mode="delta",
    )
    baseline = MODULE.extract_trial_response(
        r_l23_all=r_l23_all,
        t_readout=2,
        t_isi_last=1,
        response_mode="baseline",
    )

    torch.testing.assert_close(raw, torch.tensor([[3.0, 4.0, 5.0]]))
    torch.testing.assert_close(delta, torch.tensor([[2.0, 2.5, 3.0]]))
    torch.testing.assert_close(baseline, torch.tensor([[1.0, 1.5, 2.0]]))


def test_extract_trial_response_rejects_unknown_mode():
    r_l23_all = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="Unsupported response_mode"):
        MODULE.extract_trial_response(
            r_l23_all=r_l23_all,
            t_readout=1,
            t_isi_last=0,
            response_mode="bad-mode",
        )


def test_compute_matched_context_weights_keeps_only_shared_contexts_and_equalizes_them():
    exp_contexts = [
        (1, 2, 3),
        (1, 2, 3),
        (4, 5, 6),
        (9, 9, 9),
    ]
    unexp_contexts = [
        (1, 2, 3),
        (4, 5, 6),
        (4, 5, 6),
        (8, 8, 8),
    ]

    exp_w, unexp_w, meta = MODULE.compute_matched_context_weights(exp_contexts, unexp_contexts)

    np.testing.assert_allclose(exp_w, np.array([0.5, 0.5, 1.0, 0.0]))
    np.testing.assert_allclose(unexp_w, np.array([1.0, 0.5, 0.5, 0.0]))
    assert meta == {
        "shared_context_count": 2,
        "expected_raw_n": 4,
        "unexpected_raw_n": 4,
        "expected_matched_n": 3,
        "unexpected_matched_n": 3,
    }


def test_recentered_mean_respects_entry_weights():
    entry = {
        "r": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ]
        ),
        "true_ch": np.array([0, 1]),
        "weights": np.array([1.0, 3.0]),
    }

    mean_curve = MODULE.recentered_mean(entry, N=4)

    expected = np.zeros(4)
    expected[MODULE.CENTER_IDX % 4] = 1.75
    np.testing.assert_allclose(mean_curve, expected)


def test_clone_network_state_detaches_and_deep_copies():
    state = NetworkState(
        r_l4=torch.tensor([[1.0, 2.0]], requires_grad=True),
        r_l23=torch.tensor([[3.0, 4.0]], requires_grad=True),
        r_pv=torch.tensor([[5.0]], requires_grad=True),
        r_som=torch.tensor([[6.0, 7.0]], requires_grad=True),
        r_vip=torch.tensor([[8.0, 9.0]], requires_grad=True),
        adaptation=torch.tensor([[10.0, 11.0]], requires_grad=True),
        h_v2=torch.tensor([[12.0, 13.0]], requires_grad=True),
        deep_template=torch.tensor([[14.0, 15.0]], requires_grad=True),
    )

    cloned = MODULE.clone_network_state(state)
    cloned.r_l23[0, 0] = -99.0

    assert state.r_l23[0, 0].item() == 3.0
    assert not cloned.r_l23.requires_grad
    assert cloned.r_l23.data_ptr() != state.r_l23.data_ptr()


def test_counterfactual_probe_channels_use_predicted_and_90deg_offset():
    pred_peak_idx = torch.tensor([0, 5, 35])
    expected_ch, unexpected_ch = MODULE.compute_counterfactual_probe_channels(
        pred_peak_idx,
        n_orientations=36,
        unexpected_offset_deg=90.0,
        step_deg=5.0,
    )

    torch.testing.assert_close(expected_ch, torch.tensor([0, 5, 35]))
    torch.testing.assert_close(unexpected_ch, torch.tensor([18, 23, 17]))


def test_branch_counterfactual_baseline_uses_shared_centering_key():
    expected_ch = np.array([0, 1])
    unexpected_ch = np.array([2, 3])

    exp_center, unexp_center = MODULE.resolve_branch_centering_channels(
        expected_ch,
        unexpected_ch,
        response_mode="baseline",
    )

    np.testing.assert_array_equal(exp_center, expected_ch)
    np.testing.assert_array_equal(unexp_center, expected_ch)

    baseline_rows = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
        ]
    )
    weights = np.ones(2, dtype=float)
    exp_entry = {"r": baseline_rows, "true_ch": exp_center, "weights": weights}
    unexp_entry = {"r": baseline_rows.copy(), "true_ch": unexp_center, "weights": weights}

    np.testing.assert_allclose(
        MODULE.recentered_mean(exp_entry, N=4),
        MODULE.recentered_mean(unexp_entry, N=4),
    )


def test_branch_counterfactual_raw_keeps_distinct_centering_keys():
    expected_ch = np.array([0, 1])
    unexpected_ch = np.array([2, 3])

    exp_center, unexp_center = MODULE.resolve_branch_centering_channels(
        expected_ch,
        unexpected_ch,
        response_mode="raw",
    )

    np.testing.assert_array_equal(exp_center, expected_ch)
    np.testing.assert_array_equal(unexp_center, unexpected_ch)


def test_extract_branch_response_raw_delta_and_baseline():
    frozen = torch.tensor([[1.0, 2.0, 3.0]])
    probe = torch.tensor([[4.0, 6.0, 8.0]])

    raw = MODULE.extract_branch_response(probe, frozen, "raw")
    delta = MODULE.extract_branch_response(probe, frozen, "delta")
    baseline = MODULE.extract_branch_response(None, frozen, "baseline")

    torch.testing.assert_close(raw, probe)
    torch.testing.assert_close(delta, torch.tensor([[3.0, 4.0, 5.0]]))
    torch.testing.assert_close(baseline, frozen)
