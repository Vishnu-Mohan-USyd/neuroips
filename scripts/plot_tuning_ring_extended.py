#!/usr/bin/env python3
"""Extended ring heatmaps for R4 checkpoint — two figures.

Figure 1 (tuning_ring_recentered.png):
    Every trial's r_l23 is rolled so its true-theta channel lands at index 18.
    2×2 grid (Relevant/Irrelevant × Expected/Unexpected) of viridis ring
    heatmaps, shared vmax. Lower-left annotation per panel shows n, total
    L2/3, peak at the true-orientation channel (index 18), and FWHM of the
    mean re-centered curve. Black arrow labels the "true orientation" at
    the channel-18 wedge.

Figure 2 (tuning_ring_allprobes.png):
    Same 4 buckets, but NO re-centering — mean r_l23 across every trial in
    the bucket regardless of stimulus orientation. 2×2 viridis, shared vmax.
    Lower-left box shows n, total L2/3, and mean per channel (total/36).
    No orientation-specific arrows.

Both figures use:
    - FB ON, late-ON readout (t=9) by default, or delta/baseline scoring
      ``r_l23[t_readout] - r_l23[t_isi_last]`` when ``--response-mode=delta``
      ``r_l23[t_isi_last]`` when ``--response-mode=baseline``
    - Expected: V2 ISI pred-error ≤ 10°; Unexpected: pred-error > 20°
    - Ambiguous-stimulus trials excluded
    - Seed 42, n_batches=20 (default), cuda if available

If any Expected bucket has fewer than 500 trials, the script re-runs the
collection with n_batches doubled, up to once. The default comparison mode
uses the original bucket aggregation; an opt-in branch-point counterfactual
mode instead branches identical pre-probe states into expected and unexpected
probe stimuli.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import NetworkState, initial_state
from src.stimulus.gratings import generate_grating
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


# ------------------------------ helpers ------------------------------


T_READ = 9
COUNTERFACTUAL_UNEXPECTED_OFFSET_DEG = 90.0


def circular_distance(a: torch.Tensor, b: torch.Tensor, period: float = 180.0) -> torch.Tensor:
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def fwhm_of_curve(curve: np.ndarray, step_deg: float) -> float:
    """FWHM of a 1D circular response profile via linear interpolation.

    Finds the fractional channel offsets at which the curve crosses
    ``half_max = baseline + 0.5 * (peak - baseline)`` on either side of
    the peak, then returns ``(right_frac + left_frac) * step_deg``. This
    resolves sub-bin precision (bin-counting snaps to multiples of
    step_deg and hides sub-5° shape differences).

    Returns NaN if a crossing cannot be found in either direction
    (e.g. the curve stays above half-max for the full ring). Returns 0
    if the curve is effectively flat (peak < 1e-8).
    """
    arr = np.asarray(curve, dtype=float)
    peak = float(arr.max())
    if peak < 1e-8:
        return 0.0
    peak_idx = int(np.argmax(arr))
    baseline = float(arr.min())
    half_max = baseline + 0.5 * (peak - baseline)
    n = len(arr)

    right_cross: float | None = None
    for i in range(1, n):
        idx = (peak_idx + i) % n
        prev_idx = (peak_idx + i - 1) % n
        if arr[idx] <= half_max < arr[prev_idx]:
            frac = (arr[prev_idx] - half_max) / (arr[prev_idx] - arr[idx])
            right_cross = (i - 1) + frac
            break

    left_cross: float | None = None
    for i in range(1, n):
        idx = (peak_idx - i) % n
        prev_idx = (peak_idx - i + 1) % n
        if arr[idx] <= half_max < arr[prev_idx]:
            frac = (arr[prev_idx] - half_max) / (arr[prev_idx] - arr[idx])
            left_cross = (i - 1) + frac
            break

    if right_cross is None or left_cross is None:
        return float("nan")
    return (right_cross + left_cross) * step_deg


def extract_trial_response(
    r_l23_all: torch.Tensor,
    t_readout: int,
    t_isi_last: int,
    response_mode: str,
) -> torch.Tensor:
    """Return the analysis response vector for one presentation.

    `raw` preserves the legacy late-ON state readout. `delta` measures the
    evoked response on the accepted surface by subtracting the final ISI state
    from the late-ON readout on the same presentation. `baseline` exposes that
    final ISI state directly for comparison on the same bucket/centering path.
    """
    r_readout = r_l23_all[:, t_readout, :]
    if response_mode == "raw":
        return r_readout
    if response_mode == "delta":
        return r_readout - r_l23_all[:, t_isi_last, :]
    if response_mode == "baseline":
        return r_l23_all[:, t_isi_last, :]
    raise ValueError(f"Unsupported response_mode={response_mode!r}")


def clone_network_state(state: NetworkState) -> NetworkState:
    """Deep-clone a recurrent state so branch probes start identically."""
    return NetworkState(**{k: v.detach().clone() for k, v in state._asdict().items()})


def duplicate_network_state(state: NetworkState) -> NetworkState:
    """Duplicate a recurrent state along the batch axis for paired branches."""
    return NetworkState(**{
        k: torch.cat([v.detach().clone(), v.detach().clone()], dim=0)
        for k, v in state._asdict().items()
    })


def compute_counterfactual_probe_channels(
    pred_peak_idx: torch.Tensor,
    n_orientations: int,
    unexpected_offset_deg: float,
    step_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected branch = predicted channel; unexpected branch = +offset."""
    offset_ch = int(round(unexpected_offset_deg / step_deg)) % n_orientations
    expected_ch = pred_peak_idx.long() % n_orientations
    unexpected_ch = (expected_ch + offset_ch) % n_orientations
    return expected_ch, unexpected_ch


def resolve_branch_centering_channels(
    expected_ch: torch.Tensor | np.ndarray,
    unexpected_ch: torch.Tensor | np.ndarray,
    response_mode: str,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """Return the per-branch re-centering channels for counterfactual outputs."""
    if response_mode == "baseline":
        return expected_ch, expected_ch
    return expected_ch, unexpected_ch


def extract_branch_response(
    r_probe_readout: torch.Tensor | None,
    frozen_r_l23: torch.Tensor,
    response_mode: str,
) -> torch.Tensor:
    """Map a branch rollout plus frozen baseline state onto the requested mode."""
    if response_mode == "baseline":
        return frozen_r_l23
    assert r_probe_readout is not None, (
        "raw/delta branch responses require a probe rollout readout."
    )
    if response_mode == "raw":
        return r_probe_readout
    if response_mode == "delta":
        return r_probe_readout - frozen_r_l23
    raise ValueError(f"Unsupported response_mode={response_mode!r}")


def compute_matched_context_weights(
    expected_contexts: list[tuple[int, int, int]],
    unexpected_contexts: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return per-trial weights that equalize shared contexts within each class.

    Only contexts present in both classes receive nonzero weight. Within a
    class, each shared context contributes total weight 1.0, so the final
    weighted class mean is the average of per-context means rather than the
    average of raw trial counts.
    """
    exp_counts = Counter(expected_contexts)
    unexp_counts = Counter(unexpected_contexts)
    shared_contexts = exp_counts.keys() & unexp_counts.keys()

    exp_weights = np.zeros(len(expected_contexts), dtype=float)
    unexp_weights = np.zeros(len(unexpected_contexts), dtype=float)

    for idx, ctx in enumerate(expected_contexts):
        if ctx in shared_contexts:
            exp_weights[idx] = 1.0 / float(exp_counts[ctx])
    for idx, ctx in enumerate(unexpected_contexts):
        if ctx in shared_contexts:
            unexp_weights[idx] = 1.0 / float(unexp_counts[ctx])

    return exp_weights, unexp_weights, {
        "shared_context_count": int(len(shared_contexts)),
        "expected_raw_n": int(len(expected_contexts)),
        "unexpected_raw_n": int(len(unexpected_contexts)),
        "expected_matched_n": int(np.count_nonzero(exp_weights)),
        "unexpected_matched_n": int(np.count_nonzero(unexp_weights)),
    }


def apply_history_matching(
    buckets: dict,
    match_history: str,
) -> tuple[dict, dict[str, dict[str, int]]]:
    """Apply optional context matching and return updated buckets plus metadata."""
    if match_history == "none":
        out = {}
        for key, entry in buckets.items():
            n = int(entry["r"].shape[0])
            out[key] = {
                **entry,
                "weights": np.ones(n, dtype=float),
                "raw_n": n,
                "matched_n": n,
                "matched_context_count": 0,
            }
        return out, {}

    if match_history != "preprobe_observed":
        raise ValueError(f"Unsupported match_history={match_history!r}")

    out = {k: dict(v) for k, v in buckets.items()}
    metadata: dict[str, dict[str, int]] = {}
    for regime in ("relevant", "irrelevant"):
        exp_key = (regime, "expected")
        unexp_key = (regime, "unexpected")
        exp_entry = buckets[exp_key]
        unexp_entry = buckets[unexp_key]
        exp_weights, unexp_weights, regime_meta = compute_matched_context_weights(
            exp_entry["context"],
            unexp_entry["context"],
        )
        metadata[regime] = regime_meta
        for key, weights in ((exp_key, exp_weights), (unexp_key, unexp_weights)):
            entry = buckets[key]
            keep = weights > 0
            out[key] = {
                **entry,
                "r": entry["r"][keep],
                "true_ch": entry["true_ch"][keep],
                "context": [ctx for ctx, keep_i in zip(entry["context"], keep) if keep_i],
                "weights": weights[keep],
                "raw_n": int(entry["r"].shape[0]),
                "matched_n": int(np.count_nonzero(keep)),
                "matched_context_count": int(regime_meta["shared_context_count"]),
            }
    return out, metadata


def run_counterfactual_probe_pair(
    net: LaminarV1V2Network,
    frozen_state: NetworkState,
    expected_theta: torch.Tensor,
    unexpected_theta: torch.Tensor,
    contrast: torch.Tensor,
    task_state: torch.Tensor,
    response_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch a frozen state into expected/unexpected probes with shared inputs."""
    B = expected_theta.shape[0]
    if response_mode == "baseline":
        baseline = frozen_state.r_l23.detach().clone()
        return baseline, baseline

    branch_state = duplicate_network_state(frozen_state)
    frozen_r = branch_state.r_l23
    probe_theta = torch.cat([expected_theta, unexpected_theta], dim=0)
    probe_contrast = torch.cat([contrast, contrast], dim=0)
    probe_task = torch.cat([task_state, task_state], dim=0)
    probe_stim = generate_grating(
        probe_theta,
        probe_contrast,
        n_orientations=net.cfg.n_orientations,
        sigma=net.cfg.sigma_ff,
        n=net.cfg.naka_rushton_n,
        c50=net.cfg.naka_rushton_c50,
        period=net.cfg.orientation_range,
    ).to(expected_theta.device)
    probe_cue = torch.zeros_like(probe_stim)

    readout = None
    state = branch_state
    for _ in range(T_READ + 1):
        state, _ = net.step(probe_stim, probe_cue, probe_task, state)
        readout = state.r_l23

    response = extract_branch_response(readout, frozen_r, response_mode)
    return response[:B], response[B:]


# ------------------------------ collect ------------------------------


def collect(config_path: str, checkpoint_path: str, device: torch.device,
            seed: int, n_batches: int, response_mode: str = "raw",
            match_history: str = "none") -> tuple[dict, int, float, dict]:
    """Run the trial loop and return (buckets_dict, N, step_deg).

    Each bucket holds per-trial analysis responses and true-theta channel
    index. Keys:
        ('relevant',   'expected')
        ('relevant',   'unexpected')
        ('irrelevant', 'expected')
        ('irrelevant', 'unexpected')
    """
    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi

    gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        task_p_switch=stim_cfg.task_p_switch,
    )

    keys = [
        ("relevant",   "expected"),
        ("relevant",   "unexpected"),
        ("irrelevant", "expected"),
        ("irrelevant", "unexpected"),
    ]
    r_rows = {k: [] for k in keys}
    true_chs = {k: [] for k in keys}
    contexts = {k: [] for k in keys}

    rng = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for _ in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg,
            )
            stim_seq = stim_seq.to(device); cue_seq = cue_seq.to(device); ts_seq = ts_seq.to(device)
            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net.forward(packed)
            q_pred_all = aux["q_pred_all"]

            B = r_l23_all.shape[0]
            true_ori = metadata.orientations.to(device)

            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1
                q_pred_isi = q_pred_all[:, t_isi_last, :]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)                      # [B]
                pred_ori = pred_peak_idx.float() * step_deg
                actual_ori = true_ori[:, pres_i]
                is_amb = metadata.is_ambiguous[:, pres_i].to(device)
                pred_error = circular_distance(pred_ori, actual_ori, period)

                is_exp = (pred_error <= 10.0) & (~is_amb)
                is_unexp = (pred_error > 20.0) & (~is_amb)

                prev_ori = true_ori[:, pres_i - 1]
                prev_ori_ch = (prev_ori / step_deg).round().long() % N
                pre_cue_peak_ch = metadata.cues[:, pres_i - 1].argmax(dim=-1).to(device)
                true_ch = (actual_ori / step_deg).round().long() % N            # [B]
                t_readout = pres_i * steps_per + T_READ
                r_l23_t = extract_trial_response(
                    r_l23_all=r_l23_all,
                    t_readout=t_readout,
                    t_isi_last=t_isi_last,
                    response_mode=response_mode,
                )

                ts_this = metadata.task_states[:, pres_i].to(device)            # [B, 2]
                regime_idx = ts_this.argmax(dim=-1)                             # [B]

                r_cpu = r_l23_t.cpu().numpy()
                true_cpu = true_ch.cpu().numpy()
                prev_ori_cpu = prev_ori_ch.cpu().numpy()
                pre_cue_cpu = pre_cue_peak_ch.cpu().numpy()
                qpred_cpu = pred_peak_idx.cpu().numpy()
                reg_cpu = regime_idx.cpu().numpy()
                exp_cpu = is_exp.cpu().numpy()
                unexp_cpu = is_unexp.cpu().numpy()

                for b in range(B):
                    regime_name = "relevant" if int(reg_cpu[b]) == 0 else "irrelevant"
                    if exp_cpu[b]:
                        bucket_name = "expected"
                    elif unexp_cpu[b]:
                        bucket_name = "unexpected"
                    else:
                        continue
                    key = (regime_name, bucket_name)
                    r_rows[key].append(r_cpu[b])
                    true_chs[key].append(int(true_cpu[b]))
                    contexts[key].append(
                        (
                            int(prev_ori_cpu[b]),
                            int(pre_cue_cpu[b]),
                            int(qpred_cpu[b]),
                        )
                    )

    out: dict = {}
    for k in keys:
        if r_rows[k]:
            r_arr = np.stack(r_rows[k])                 # [n, N]
            true_arr = np.array(true_chs[k], dtype=int)  # [n]
        else:
            r_arr = np.zeros((0, N))
            true_arr = np.zeros((0,), dtype=int)
        out[k] = {"r": r_arr, "true_ch": true_arr, "context": list(contexts[k])}
    out, match_meta = apply_history_matching(out, match_history)
    return out, N, step_deg, match_meta


def collect_counterfactual_branch(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    seed: int,
    n_batches: int,
    response_mode: str = "raw",
) -> tuple[dict, int, float, dict]:
    """Collect branch-point counterfactual probe responses from identical states."""
    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    assert steps_on > T_READ, (
        f"steps_on={steps_on} must exceed T_READ={T_READ} for branch probe readout."
    )

    gen = HMMSequenceGenerator(
        n_orientations=N,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
    )

    keys = [
        ("relevant", "expected"),
        ("relevant", "unexpected"),
        ("irrelevant", "expected"),
        ("irrelevant", "unexpected"),
    ]
    r_rows = {k: [] for k in keys}
    true_chs = {k: [] for k in keys}
    regime_counts = {
        "relevant": {"paired_branchpoints": 0},
        "irrelevant": {"paired_branchpoints": 0},
    }

    rng = torch.Generator().manual_seed(seed)
    h_dim = model_cfg.v2_hidden_dim * 2 if getattr(net, "use_dual_v2", False) else model_cfg.v2_hidden_dim

    with torch.no_grad():
        for _ in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, task_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg,
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            task_seq = task_seq.to(device)
            task_states = metadata.task_states.to(device)
            true_ori = metadata.orientations.to(device)
            contrasts = metadata.contrasts.to(device)

            B = stim_seq.shape[0]
            T_total = stim_seq.shape[1]
            state = initial_state(B, N, h_dim, device=device)

            for t in range(T_total):
                state, aux_t = net.step(stim_seq[:, t], cue_seq[:, t], task_seq[:, t], state)
                if (t + 1) % steps_per != 0:
                    continue
                pres_i = (t + 1) // steps_per
                if pres_i >= seq_length:
                    continue

                upcoming_task = task_states[:, pres_i, :]
                regime_idx = upcoming_task.argmax(dim=-1)
                pred_peak_idx = aux_t.q_pred.argmax(dim=-1)
                exp_ch, unexp_ch = compute_counterfactual_probe_channels(
                    pred_peak_idx,
                    n_orientations=N,
                    unexpected_offset_deg=COUNTERFACTUAL_UNEXPECTED_OFFSET_DEG,
                    step_deg=step_deg,
                )
                exp_theta = exp_ch.float() * step_deg
                unexp_theta = unexp_ch.float() * step_deg
                frozen_state = clone_network_state(state)
                exp_resp, unexp_resp = run_counterfactual_probe_pair(
                    net,
                    frozen_state,
                    expected_theta=exp_theta,
                    unexpected_theta=unexp_theta,
                    contrast=contrasts[:, pres_i],
                    task_state=upcoming_task,
                    response_mode=response_mode,
                )

                exp_cpu = exp_resp.cpu().numpy()
                unexp_cpu = unexp_resp.cpu().numpy()
                exp_true_cpu, unexp_true_cpu = resolve_branch_centering_channels(
                    exp_ch.cpu().numpy(),
                    unexp_ch.cpu().numpy(),
                    response_mode=response_mode,
                )
                reg_cpu = regime_idx.cpu().numpy()

                for b in range(B):
                    regime_name = "relevant" if int(reg_cpu[b]) == 0 else "irrelevant"
                    r_rows[(regime_name, "expected")].append(exp_cpu[b])
                    r_rows[(regime_name, "unexpected")].append(unexp_cpu[b])
                    true_chs[(regime_name, "expected")].append(int(exp_true_cpu[b]))
                    true_chs[(regime_name, "unexpected")].append(int(unexp_true_cpu[b]))
                    regime_counts[regime_name]["paired_branchpoints"] += 1

    out: dict = {}
    for k in keys:
        if r_rows[k]:
            r_arr = np.stack(r_rows[k])
            true_arr = np.array(true_chs[k], dtype=int)
        else:
            r_arr = np.zeros((0, N))
            true_arr = np.zeros((0,), dtype=int)
        n = int(r_arr.shape[0])
        out[k] = {
            "r": r_arr,
            "true_ch": true_arr,
            "weights": np.ones(n, dtype=float),
            "raw_n": n,
            "matched_n": n,
            "matched_context_count": 0,
        }

    return out, N, step_deg, regime_counts


# ------------------------------ derive means ------------------------------


CENTER_IDX = 18  # channel index where true orientation lands after re-centering


def recentered_mean(entry: dict, N: int) -> np.ndarray:
    r = entry["r"]; true_ch = entry["true_ch"]
    if r.shape[0] == 0:
        return np.zeros(N)
    rolled = np.stack([np.roll(r[i], shift=CENTER_IDX - int(true_ch[i])) for i in range(r.shape[0])])
    weights = entry.get("weights")
    if weights is None or len(weights) == 0:
        return rolled.mean(axis=0)
    return np.average(rolled, axis=0, weights=weights)


def pooled_mean(entry: dict, N: int) -> np.ndarray:
    r = entry["r"]
    if r.shape[0] == 0:
        return np.zeros(N)
    weights = entry.get("weights")
    if weights is None or len(weights) == 0:
        return r.mean(axis=0)
    return np.average(r, axis=0, weights=weights)


# ------------------------------ plot ---------------------------------


def _plot_ring_base(ax, activity: np.ndarray, vmax: float, cmap) -> None:
    N = len(activity)
    theta_centers = np.arange(N) * (2 * np.pi / N)
    width = 2 * np.pi / N

    inner_r = 0.65
    outer_r = 1.00
    height = outer_r - inner_r

    norm = Normalize(vmin=0.0, vmax=vmax)
    colors = cmap(norm(activity))

    ax.bar(
        theta_centers, height, width=width, bottom=inner_r,
        color=colors, edgecolor="white", linewidth=0.6, align="center",
    )

    # Cardinal labels inside the ring
    for deg in [0, 45, 90, 135]:
        rad = deg * 2 * np.pi / 180.0
        ax.text(rad, 0.28, f"{deg}°", ha="center", va="center",
                color="gray", fontsize=9)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)


def plot_ring_recentered(ax, curve: np.ndarray, vmax: float, cmap, step_deg: float,
                         n: int) -> None:
    _plot_ring_base(ax, curve, vmax, cmap)

    total = float(curve.sum())
    peak_at_true = float(curve[CENTER_IDX])
    fwhm = fwhm_of_curve(curve, step_deg)
    ax.text(
        0.02, 0.02,
        f"n = {n}\n"
        f"total L2/3: {total:.2f}\n"
        f"peak @ true (ch {CENTER_IDX}): {peak_at_true:.3f}\n"
        f"FWHM: {fwhm:.1f}°",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85, linewidth=0.6),
    )


def plot_ring_pooled(ax, curve: np.ndarray, vmax: float, cmap, step_deg: float,
                     n: int) -> None:
    N = len(curve)
    _plot_ring_base(ax, curve, vmax, cmap)
    total = float(curve.sum())
    mean_per_ch = total / N
    ax.text(
        0.02, 0.02,
        f"n = {n}\n"
        f"total L2/3: {total:.2f}\n"
        f"mean per channel: {mean_per_ch:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.85, linewidth=0.6),
    )


# ------------------------------ figure builders ---------------------


def build_figure(
    buckets: dict, N: int, step_deg: float, mean_fn, plot_fn,
    vmax_src: dict, fig_path: str, suptitle: str, subtitle: str,
) -> list[dict[str, float | int | str]]:
    """Render a 2×2 figure from either re-centered or pooled means.

    Returns a list of stats dicts. Peak/fwhm are computed on the mean vector.
    (For pooled, peak/fwhm use the pooled curve's max.)
    """
    cmap = cm.get_cmap("viridis")

    # shared vmax across all 4 panels
    vmax = max(float(vmax_src[k].max()) for k in buckets)
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(10, 10))
    row_titles = ["Relevant", "Irrelevant"]
    col_titles = ["Expected", "Unexpected"]
    regime_keys = ["relevant", "irrelevant"]
    bucket_keys = ["expected", "unexpected"]

    stats = []
    for i, regime in enumerate(regime_keys):
        for j, bucket in enumerate(bucket_keys):
            idx = i * 2 + j + 1
            ax = fig.add_subplot(2, 2, idx, projection="polar")
            entry = buckets[(regime, bucket)]
            curve = mean_fn(entry, N)
            n = int(entry["r"].shape[0])
            plot_fn(ax, curve, vmax, cmap, step_deg, n)
            ax.set_title(f"{row_titles[i]} {col_titles[j]}\n(n={n})",
                         fontsize=12, pad=18)
            stats.append({
                "label": f"{row_titles[i]} {col_titles[j]}",
                "n": int(n),
                "raw_n": int(entry.get("raw_n", n)),
                "matched_n": int(entry.get("matched_n", n)),
                "matched_context_count": int(entry.get("matched_context_count", 0)),
                "weight_sum": float(np.asarray(entry.get("weights", np.ones(n, dtype=float))).sum()),
                "total": float(curve.sum()),
                "peak": float(curve.max()),
                "fwhm_deg": float(fwhm_of_curve(curve, step_deg)),
            })

    cbar_ax = fig.add_axes([0.22, 0.045, 0.56, 0.020])
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("L2/3 activity", fontsize=10)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.985)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top",
                 fontsize=10, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))

    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats


# ------------------------------ main ---------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config/sweep/sweep_rescue_4.yaml")
    p.add_argument("--checkpoint",
                   default="/home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt")
    p.add_argument("--output-dir", default="docs/figures")
    p.add_argument("--fig1-name", default="tuning_ring_recentered.png",
                   help="Filename (within --output-dir) for the re-centered figure.")
    p.add_argument("--fig1-title", default="Tuning curves re-centered on true orientation",
                   help="Main title for the re-centered figure.")
    p.add_argument("--skip-fig2", action="store_true",
                   help="Skip the all-probes-pooled figure (only produce re-centered).")
    p.add_argument("--device", default=None)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-exp-n", type=int, default=500,
                   help="Re-collect with 2× batches once if any Expected bucket has fewer trials.")
    p.add_argument("--json-out", default=None,
                   help="Optional path for machine-readable stats JSON.")
    p.add_argument("--response-mode", choices=("raw", "delta", "baseline"), default="raw",
                   help="Analysis response surface: late-ON state (`raw`), evoked delta (`delta`), or previous-ISI baseline (`baseline`).")
    p.add_argument("--comparison-mode", choices=("bucket", "branch_counterfactual"), default="bucket",
                   help="Original bucket comparison (`bucket`) or identical-state branch-point counterfactual probes (`branch_counterfactual`).")
    p.add_argument("--match-history", choices=("none", "preprobe_observed"), default="none",
                   help="Optional history matching. `preprobe_observed` matches expected/unexpected within regime on (prev orientation, previous cue peak, previous-ISI q_pred peak).")
    args = p.parse_args()

    _, train_cfg, _ = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.comparison_mode == "branch_counterfactual" and args.match_history != "none":
        raise ValueError("branch_counterfactual mode does not support --match-history; branch pairing already fixes the pre-probe state.")
    print(
        f"[collect] R4 — device={device} seed={args.seed} "
        f"n_batches={args.n_batches} response_mode={args.response_mode} "
        f"comparison_mode={args.comparison_mode} match_history={args.match_history}",
        flush=True,
    )

    if args.comparison_mode == "branch_counterfactual":
        buckets, N, step_deg, comparison_meta = collect_counterfactual_branch(
            args.config,
            args.checkpoint,
            device,
            seed=args.seed,
            n_batches=args.n_batches,
            response_mode=args.response_mode,
        )
        match_meta = {}
    else:
        buckets, N, step_deg, match_meta = collect(
            args.config, args.checkpoint, device,
            seed=args.seed, n_batches=args.n_batches,
            response_mode=args.response_mode,
            match_history=args.match_history,
        )
        comparison_meta = {}

    n_rel_exp = int(buckets[("relevant", "expected")]["r"].shape[0])
    n_irr_exp = int(buckets[("irrelevant", "expected")]["r"].shape[0])
    print(f"  Expected n: relevant={n_rel_exp}, irrelevant={n_irr_exp}", flush=True)

    if min(n_rel_exp, n_irr_exp) < args.min_exp_n:
        bumped = args.n_batches * 2
        print(f"  below min-exp-n ({args.min_exp_n}); re-collecting with n_batches={bumped}", flush=True)
        if args.comparison_mode == "branch_counterfactual":
            buckets, N, step_deg, comparison_meta = collect_counterfactual_branch(
                args.config,
                args.checkpoint,
                device,
                seed=args.seed,
                n_batches=bumped,
                response_mode=args.response_mode,
            )
            match_meta = {}
        else:
            buckets, N, step_deg, match_meta = collect(
                args.config, args.checkpoint, device,
                seed=args.seed, n_batches=bumped,
                response_mode=args.response_mode,
                match_history=args.match_history,
            )
            comparison_meta = {}
        n_rel_exp = int(buckets[("relevant", "expected")]["r"].shape[0])
        n_irr_exp = int(buckets[("irrelevant", "expected")]["r"].shape[0])
        print(f"  re-collected Expected n: relevant={n_rel_exp}, irrelevant={n_irr_exp}", flush=True)

    # Precompute both mean vectors (for vmax sharing within each figure)
    recentered = {k: recentered_mean(v, N) for k, v in buckets.items()}
    pooled = {k: pooled_mean(v, N) for k, v in buckets.items()}

    fig1_path = os.path.join(args.output_dir, args.fig1_name)
    fig2_path = os.path.join(args.output_dir, "tuning_ring_allprobes.png")

    stats1 = build_figure(
        buckets=buckets, N=N, step_deg=step_deg,
        mean_fn=recentered_mean, plot_fn=plot_ring_recentered,
        vmax_src=recentered, fig_path=fig1_path,
        suptitle=args.fig1_title,
        subtitle="",
    )
    print(f"[save] {fig1_path}", flush=True)
    print("[fig1 stats]  label               n    total   peak   fwhm(°)", flush=True)
    for stat in stats1:
        print(
            f"  {stat['label']:22s} {stat['n']:5d}  "
            f"{stat['total']:6.3f}  {stat['peak']:5.3f}  {stat['fwhm_deg']:6.2f}",
            flush=True,
        )

    json_payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "seed": int(args.seed),
        "n_batches": int(args.n_batches),
        "step_deg": float(step_deg),
        "comparison_mode": args.comparison_mode,
        "response_mode": args.response_mode,
        "match_history_mode": args.match_history,
        "response_definition": (
            "r_l23[t_readout]"
            if args.response_mode == "raw"
            else (
                "r_l23[t_readout] - r_l23[t_isi_last]"
                if args.response_mode == "delta"
                else "r_l23[t_isi_last]"
            )
        ),
        "t_readout": int(T_READ),
        "t_isi_last_relative": int(train_cfg.steps_on + train_cfg.steps_isi - 1),
        "match_history": {
            "mode": args.match_history,
            "context_definition": (
                [
                    "prev_ori_ch",
                    "pre_cue_peak_ch",
                    "qpred_peak_ch",
                ]
                if args.match_history == "preprobe_observed"
                else []
            ),
            "regime_stats": match_meta,
        },
        "branch_counterfactual": {
            "enabled": args.comparison_mode == "branch_counterfactual",
            "expected_probe_definition": "probe orientation = q_pred peak at the pre-probe last ISI timestep",
            "unexpected_probe_definition": f"probe orientation = expected probe + {COUNTERFACTUAL_UNEXPECTED_OFFSET_DEG:.1f} deg",
            "regime_stats": comparison_meta,
        },
        "figures": {
            "recentered": {
                "path": fig1_path,
                "title": args.fig1_title,
                "stats": stats1,
            }
        },
    }

    if not args.skip_fig2:
        stats2 = build_figure(
            buckets=buckets, N=N, step_deg=step_deg,
            mean_fn=pooled_mean, plot_fn=plot_ring_pooled,
            vmax_src=pooled, fig_path=fig2_path,
            suptitle="Mean L2/3 activity (all probe orientations pooled)",
            subtitle=(
                "No re-centering — average over every trial in the bucket; "
                "reveals the overall expectation-suppression gap per bucket."
            ),
        )
        print(f"[save] {fig2_path}", flush=True)
        print("[fig2 stats]  label               n    total   peak   fwhm(°)", flush=True)
        for stat in stats2:
            print(
                f"  {stat['label']:22s} {stat['n']:5d}  "
                f"{stat['total']:6.3f}  {stat['peak']:5.3f}  {stat['fwhm_deg']:6.2f}",
                flush=True,
            )
        json_payload["figures"]["all_probes"] = {
            "path": fig2_path,
            "stats": stats2,
        }

    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        print(f"[json] {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
