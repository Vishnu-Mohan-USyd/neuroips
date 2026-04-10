"""Phase 1A evaluation: extended M7 with task_state × time window split.

Runs the dual_1a checkpoint through:
  1. Extended M7 match-vs-near-miss decoding across three task_states
     ``{baseline=(0,0), focused=(1,0), routine=(0,1)}`` AND three readout
     time windows ``{default=last-5, early=steps 0..3, late=last-3}``.
     The early/late split is primary per the Phase 1A plan (time-resolved
     effects may reverse within a trial).
  2. M10 amplitude ratio per regime: mean(|r_l23_on|) / mean(|r_l23_off|).
  3. FWHM delta per regime: change in tuning width from FB-OFF to FB-ON,
     computed at oracle=stim for 8 anchors.
  4. Preregistered PASS/PARTIAL/FAIL verdict per Phase 1A criteria.

All computation reuses ``run_full_trajectory_noisy`` (full [B, T, N]
trajectory) and the decoder fit pattern from
``metric_match_vs_near_miss_decoding`` — no modifications to
``analyze_representation.py``.

Outputs:
    - results/dual_1a/m7_extended.json         full metric dump
    - results/dual_1a/dual_1a_comparison.txt   human-readable table + verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_representation import (  # noqa: E402
    EVAL_CONTRAST,
    READ_WINDOW,
    T_STEPS,
    _build_oracle,
    feedback_disabled,
    fwhm_from_curve,
    load_model,
    run_full_trajectory,
    run_full_trajectory_noisy,
)
from src.stimulus.gratings import generate_grating  # noqa: E402


# ---------- Extended M7: match-vs-near-miss with configurable time window ----

_ANCHORS = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
_DELTAS = [3.0, 5.0, 10.0, 15.0]


def _fit_and_score(X: np.ndarray, labels: np.ndarray,
                   train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_idx], labels[train_idx])
    return float(clf.score(X[test_idx], labels[test_idx]))


def _window_mean(traj: torch.Tensor, window: str) -> torch.Tensor:
    """Apply a named readout window to a ``[B, T, ...]`` trajectory.

    Args:
        traj: ``[B, T_STEPS, ...]`` trajectory with time on dim 1. Works for
            any per-population output (r_l23, r_l4, r_som, r_pv, center_exc).
        window: one of 'default' (last READ_WINDOW=5), 'early' (first 4),
                'mid' (centered 3 steps, i.e., T//2-1 .. T//2+1),
                'late' (last 3).

    Returns:
        Tensor with the time axis reduced by mean: ``[B, ...]``.
    """
    if window == "default":
        return traj[:, -READ_WINDOW:].mean(dim=1)
    if window == "early":
        return traj[:, 0:4].mean(dim=1)
    if window == "mid":
        T = traj.shape[1]
        start = max(0, T // 2 - 1)
        end = min(T, T // 2 + 2)  # 3-step centered window
        return traj[:, start:end].mean(dim=1)
    if window == "late":
        return traj[:, -3:].mean(dim=1)
    raise ValueError(f"unknown window {window!r}")


def run_full_trajectory_with_aux(
    net,
    stim_thetas: torch.Tensor,
    oracle_thetas: torch.Tensor,
    device: torch.device,
    contrast: float = EVAL_CONTRAST,
    task_state: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, dict]:
    """Deterministic trajectory that also returns the network's aux dict.

    Identical setup to ``run_full_trajectory`` but captures ``aux`` (which
    holds r_l4, r_pv, r_som, r_vip, center_exc trajectories alongside
    r_l23_all). Needed for the Phase 1C channel_profile / E:I metrics that
    reference inhibitory populations and apical excitation.

    Returns:
        (r_l23_all [B, T_STEPS, N], aux dict with r_l4_all / r_pv_all /
         r_som_all / center_exc_all / ... keys).
    """
    B = stim_thetas.shape[0]
    N = net.cfg.n_orientations

    stim_frame = generate_grating(
        stim_thetas.to(device),
        torch.full((B,), contrast, device=device),
        n_orientations=N,
        sigma=net.cfg.sigma_ff,
        n=net.cfg.naka_rushton_n,
        c50=net.cfg.naka_rushton_c50,
        period=net.cfg.orientation_range,
    )  # [B, N]
    stim_seq = stim_frame.unsqueeze(1).expand(B, T_STEPS, N).contiguous()
    oracle_q, oracle_pi = _build_oracle(net, oracle_thetas.to(device))
    cue_seq = torch.zeros(B, T_STEPS, N, device=device)
    task_seq = torch.zeros(B, T_STEPS, 2, device=device)
    task_seq[..., 0] = task_state[0]
    task_seq[..., 1] = task_state[1]
    packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
    net.oracle_mode = True
    net.oracle_q_pred = oracle_q
    net.oracle_pi_pred = oracle_pi
    try:
        with torch.no_grad():
            r_l23_all, _, aux = net(packed)
    finally:
        net.oracle_mode = False
        net.oracle_q_pred = None
        net.oracle_pi_pred = None
    return r_l23_all, aux


# ---------- Channel profile + E:I ratio (Phase 1C amplitude-shortcut probe) ----

def channel_profile(
    net,
    device: torch.device,
    task_state: tuple[float, float],
    window: str,
) -> dict:
    """Per-channel r_l23 magnitudes + E:I diagnostics per (regime, window).

    For each of the 8 anchors used elsewhere, runs a deterministic trajectory
    (stim == oracle == anchor) under the specified ``task_state``. From the
    windowed r_l23 response, extracts:

    * ``peak``       — r_l23 at the channel whose preferred orientation is
                       nearest the anchor.
    * ``near_flank`` — mean r_l23 at peak±2 channels (±10° in the 36-channel
                       / 180° layout — the ±10° locations where the debugger
                       showed sensory CE pulling up and sharp pushing down).
    * ``far_flank``  — mean r_l23 at peak±10 channels (±50° — the FWHM waist
                       where sharp+l23_energy would cancel sensory).
    * ``c_exc_peak`` — mean ``|center_exc|`` at the peak channel (apical
                       excitation magnitude).
    * ``som_peak``   — mean ``|r_som|`` at the peak channel (SOM inhibitory
                       magnitude — this is the nearest single-channel
                       inhibitory signal).

    All quantities are averaged across the 8 anchors. The resulting numbers
    are regime-specific, window-specific, and fully deterministic (no
    stimulus noise).
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N

    # Channel offsets for near/far flanks (forced to concrete integers)
    # near = ±10° → ±2 channels in 36ch/180° (5° per channel)
    # far  = ±50° → ±10 channels
    assert step > 0
    near_off = int(round(10.0 / step))
    far_off = int(round(50.0 / step))

    peaks: list[float] = []
    near_flanks: list[float] = []
    far_flanks: list[float] = []
    c_exc_peaks: list[float] = []
    som_peaks: list[float] = []

    for anchor in _ANCHORS:
        stim = torch.full((1,), anchor, device=device)
        oracle = torch.full((1,), anchor, device=device)
        r_l23_traj, aux = run_full_trajectory_with_aux(
            net, stim, oracle, device, task_state=task_state,
        )  # r_l23_traj: [1, T, N]; aux: dict
        r_som_traj = aux["r_som_all"]                           # [1, T, N]
        c_exc_traj = aux["center_exc_all"]                      # [1, T, N]

        r_l23_win = _window_mean(r_l23_traj, window).squeeze(0)  # [N]
        r_som_win = _window_mean(r_som_traj, window).squeeze(0)  # [N]
        c_exc_win = _window_mean(c_exc_traj, window).squeeze(0)  # [N]

        peak_idx = int(round(anchor / step)) % N
        near_ccw = (peak_idx - near_off) % N
        near_cw = (peak_idx + near_off) % N
        far_ccw = (peak_idx - far_off) % N
        far_cw = (peak_idx + far_off) % N

        peaks.append(float(r_l23_win[peak_idx].item()))
        near_flanks.append(
            0.5 * (float(r_l23_win[near_ccw].item())
                   + float(r_l23_win[near_cw].item()))
        )
        far_flanks.append(
            0.5 * (float(r_l23_win[far_ccw].item())
                   + float(r_l23_win[far_cw].item()))
        )
        c_exc_peaks.append(float(c_exc_win[peak_idx].abs().item()))
        som_peaks.append(float(r_som_win[peak_idx].abs().item()))

    return {
        "task_state": list(task_state),
        "window": window,
        "peak":         float(np.mean(peaks)),
        "near_flank":   float(np.mean(near_flanks)),
        "far_flank":    float(np.mean(far_flanks)),
        "c_exc_peak":   float(np.mean(c_exc_peaks)),
        "som_peak":     float(np.mean(som_peaks)),
        "near_off_channels": near_off,
        "far_off_channels":  far_off,
    }


def gate_diagnostics(
    net,
    device: torch.device,
    task_state: tuple[float, float],
) -> dict:
    """alpha_net gate-output distribution per regime (Phase 2 E/I gate probe).

    Runs the standard 8-anchor matched-stim trajectory (stim==oracle, no
    noise) under the specified ``task_state`` and extracts all per-timestep
    samples of g_E and g_I from the ``gains_all`` aux tensor ([B=1, T, 2]).
    Gains are already post-``2*sigmoid`` in network.py (line 156), so
    ``g ≈ 1.0`` means the gate is at its identity init.

    Each anchor contributes T samples (one per timestep). With 8 anchors and
    T=25, that's 200 samples per regime — enough to compute a meaningful
    mean/std/min/max/median and to tell whether the gate actually drifted
    from identity or not.

    Per-anchor means are also returned for backward compatibility with the
    earlier (Phase 2.0) report.

    Returns ``{"use_ei_gate": False, ...}`` if the network lacks alpha_net;
    callers should check the flag before indexing numeric fields.
    """
    if not hasattr(net, "alpha_net"):
        return {
            "task_state": list(task_state),
            "use_ei_gate": False,
            "g_E_mean": None, "g_E_std": None,
            "g_E_min": None, "g_E_max": None, "g_E_median": None,
            "g_I_mean": None, "g_I_std": None,
            "g_I_min": None, "g_I_max": None, "g_I_median": None,
            "g_E": None, "g_I": None,
        }

    g_E_samples: list[float] = []     # flattened over anchors × T
    g_I_samples: list[float] = []
    g_E_anchor_means: list[float] = []   # backward compat (1 scalar per anchor)
    g_I_anchor_means: list[float] = []

    for anchor in _ANCHORS:
        stim = torch.full((1,), anchor, device=device)
        oracle = torch.full((1,), anchor, device=device)
        _, aux = run_full_trajectory_with_aux(
            net, stim, oracle, device, task_state=task_state,
        )
        gains = aux["gains_all"]                        # [1, T, 2]
        g_E_flat = gains[..., 0].flatten().cpu().numpy().astype(float)
        g_I_flat = gains[..., 1].flatten().cpu().numpy().astype(float)
        g_E_samples.extend(g_E_flat.tolist())
        g_I_samples.extend(g_I_flat.tolist())
        g_E_anchor_means.append(float(g_E_flat.mean()))
        g_I_anchor_means.append(float(g_I_flat.mean()))

    g_E_arr = np.asarray(g_E_samples, dtype=np.float64)
    g_I_arr = np.asarray(g_I_samples, dtype=np.float64)

    return {
        "task_state": list(task_state),
        "use_ei_gate": True,
        "n_samples": int(g_E_arr.size),
        # Primary distribution stats (over anchors × T)
        "g_E_mean":   float(g_E_arr.mean()),
        "g_E_std":    float(g_E_arr.std(ddof=0)),
        "g_E_min":    float(g_E_arr.min()),
        "g_E_max":    float(g_E_arr.max()),
        "g_E_median": float(np.median(g_E_arr)),
        "g_I_mean":   float(g_I_arr.mean()),
        "g_I_std":    float(g_I_arr.std(ddof=0)),
        "g_I_min":    float(g_I_arr.min()),
        "g_I_max":    float(g_I_arr.max()),
        "g_I_median": float(np.median(g_I_arr)),
        # Backward compat with earlier report (Phase 2.0)
        "g_E":            float(g_E_arr.mean()),
        "g_I":            float(g_I_arr.mean()),
        "g_E_per_anchor": g_E_anchor_means,
        "g_I_per_anchor": g_I_anchor_means,
    }


def ei_ratio(
    net,
    device: torch.device,
    task_state: tuple[float, float],
    window: str,
) -> dict:
    """Canonical E:I ratio per (regime, window) on a matched-stim batch.

    Uses the definition from ``src/analysis/energy.py``:
        E = mean(|r_l4|) + mean(|r_l23|)
        I = mean(|r_pv|) + mean(|r_som|)
        E:I = E / I

    All means are taken over the windowed slice of the trajectory, across
    the 8 anchors (stim == oracle == anchor, deterministic). Used to test
    the Phase 1C hypothesis that L2 energy + higher focused energy routing
    will pull focused E:I below 1.1 (debugger measured 1.31 in Phase 1B).
    """
    exc_list = []
    inh_list = []
    e_l4_list = []
    e_l23_list = []
    i_pv_list = []
    i_som_list = []

    for anchor in _ANCHORS:
        stim = torch.full((1,), anchor, device=device)
        oracle = torch.full((1,), anchor, device=device)
        r_l23_traj, aux = run_full_trajectory_with_aux(
            net, stim, oracle, device, task_state=task_state,
        )
        r_l4_traj = aux["r_l4_all"]       # [1, T, N]
        r_pv_traj = aux["r_pv_all"]       # [1, T, 1]
        r_som_traj = aux["r_som_all"]     # [1, T, N]

        # Apply window on time axis, then mean over channels
        e_l4  = float(_window_mean(r_l4_traj,  window).abs().mean().item())
        e_l23 = float(_window_mean(r_l23_traj, window).abs().mean().item())
        i_pv  = float(_window_mean(r_pv_traj,  window).abs().mean().item())
        i_som = float(_window_mean(r_som_traj, window).abs().mean().item())

        e_l4_list.append(e_l4)
        e_l23_list.append(e_l23)
        i_pv_list.append(i_pv)
        i_som_list.append(i_som)
        exc_list.append(e_l4 + e_l23)
        inh_list.append(i_pv + i_som)

    mean_exc = float(np.mean(exc_list))
    mean_inh = float(np.mean(inh_list))
    return {
        "task_state": list(task_state),
        "window": window,
        "E": mean_exc,
        "I": mean_inh,
        "EI_ratio": mean_exc / max(mean_inh, 1e-8),
        "mean_r_l4":  float(np.mean(e_l4_list)),
        "mean_r_l23": float(np.mean(e_l23_list)),
        "mean_r_pv":  float(np.mean(i_pv_list)),
        "mean_r_som": float(np.mean(i_som_list)),
    }


def extended_m7_single_window(
    net,
    device: torch.device,
    task_state: tuple[float, float],
    window: str,
    n_train: int = 800,
    n_test: int = 200,
    noise_std: float = 0.3,
    readout_noise_std: float = 0.3,
    seed: int = 42,
) -> dict:
    """Single task_state × single time-window M7.

    Returns the same ``delta_{int}`` structure as
    ``metric_match_vs_near_miss_decoding`` but with the readout slice
    set by ``window``. Computes trajectories once per (task_state, anchor,
    delta) pair and reuses them across windows is deferred to the outer
    caller — this function is the inner per-window decoder.
    """
    period = net.cfg.orientation_range
    n_total = n_train + n_test
    assert n_total % 2 == 0
    n_half = n_total // 2

    results: dict = {}
    for delta in _DELTAS:
        anchor_accs_on: list[float] = []
        anchor_accs_off: list[float] = []
        per_anchor: dict = {}
        for anchor in _ANCHORS:
            thetas_match = torch.full((n_half,), anchor, device=device)
            thetas_miss = torch.full((n_half,), (anchor + delta) % period, device=device)
            stim = torch.cat([thetas_match, thetas_miss], dim=0)
            oracle_arr = torch.full_like(stim, anchor)
            labels = np.concatenate([
                np.zeros(n_half, dtype=np.int64),
                np.ones(n_half, dtype=np.int64),
            ])

            trial_seed = seed + int(delta * 10) + int(anchor * 100)

            # Full trajectory ON
            r_on_traj = run_full_trajectory_noisy(
                net, stim, oracle_arr, device,
                noise_std=noise_std, seed=trial_seed,
                task_state=task_state,
            )  # [B, T, N]
            # Full trajectory OFF
            with feedback_disabled(net):
                r_off_traj = run_full_trajectory_noisy(
                    net, stim, oracle_arr, device,
                    noise_std=noise_std, seed=trial_seed,
                    task_state=task_state,
                )

            # Apply requested readout window
            r_on = _window_mean(r_on_traj, window).cpu().numpy()
            r_off = _window_mean(r_off_traj, window).cpu().numpy()

            rs = np.random.RandomState(trial_seed + 7)
            readout_noise = (rs.randn(*r_on.shape).astype(np.float32)
                             * readout_noise_std)
            X_on = r_on + readout_noise
            X_off = r_off + readout_noise

            perm = rs.permutation(n_total)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

            acc_on = _fit_and_score(X_on, labels, train_idx, test_idx)
            acc_off = _fit_and_score(X_off, labels, train_idx, test_idx)

            anchor_accs_on.append(acc_on)
            anchor_accs_off.append(acc_off)
            per_anchor[float(anchor)] = {
                "on": acc_on, "off": acc_off,
                "delta_acc": acc_on - acc_off,
            }

        mean_on = float(np.mean(anchor_accs_on))
        mean_off = float(np.mean(anchor_accs_off))
        results[f"delta_{int(delta)}"] = {
            "on": mean_on,
            "off": mean_off,
            "delta_acc": mean_on - mean_off,
            "per_anchor": per_anchor,
        }
    results["n_train"] = n_train
    results["n_test"] = n_test
    results["window"] = window
    return results


# ---------- M10: global amplitude ratio (FB-ON total / FB-OFF total) ----------


def m10_amplitude_ratio(
    net,
    device: torch.device,
    task_state: tuple[float, float],
    n_trials: int = 80,
    seed: int = 42,
    window: str = "default",
) -> dict:
    """Compute FB-ON/FB-OFF amplitude ratio at matched (stim==oracle) condition.

    For 8 anchor orientations × (n_trials/8) trials, stim_theta == oracle_theta,
    run deterministic (no noise) trajectories, take the mean |r_l23| over the
    specified readout window for both FB-ON and FB-OFF, compute the ratio
    per anchor, then average.

    Args:
        window: one of 'default', 'early', 'late' — passed through to
            ``_window_mean``. The default-window value is the one used by the
            preregistered M10 gate; early/late are diagnostic.
    """
    period = net.cfg.orientation_range
    per_trial = max(1, n_trials // len(_ANCHORS))

    anchor_ratios = []
    on_mags = []
    off_mags = []
    for anchor in _ANCHORS:
        stim = torch.full((per_trial,), anchor, device=device)
        oracle = torch.full((per_trial,), anchor, device=device)
        r_on_traj = run_full_trajectory(net, stim, oracle, device,
                                        task_state=task_state)
        with feedback_disabled(net):
            r_off_traj = run_full_trajectory(net, stim, oracle, device,
                                             task_state=task_state)
        r_on = _window_mean(r_on_traj, window).abs().mean().item()
        r_off = _window_mean(r_off_traj, window).abs().mean().item()
        on_mags.append(r_on)
        off_mags.append(r_off)
        anchor_ratios.append(r_on / max(r_off, 1e-8))
    return {
        "task_state": list(task_state),
        "window": window,
        "mean_ratio": float(np.mean(anchor_ratios)),
        "per_anchor_ratios": anchor_ratios,
        "mean_on_magnitude": float(np.mean(on_mags)),
        "mean_off_magnitude": float(np.mean(off_mags)),
    }


# ---------- FWHM delta: tuning width ON vs OFF ----------


def fwhm_delta(
    net,
    device: torch.device,
    task_state: tuple[float, float],
    window: str = "default",
) -> dict:
    """Mean FWHM (over 8 anchors) for FB-ON and FB-OFF in a time window.

    Args:
        window: one of 'default', 'early', 'late' — which time-slice of the
            L2/3 trajectory to reduce before fitting the FWHM. The default
            window is what the verdict table historically reports; early/late
            are diagnostic for the Phase 1B FWHM paradox follow-up.
    """
    period = net.cfg.orientation_range
    N = net.cfg.n_orientations
    prefs = np.arange(N, dtype=np.float32) * (period / N)

    fwhm_on_list = []
    fwhm_off_list = []
    for anchor in _ANCHORS:
        stim = torch.full((1,), anchor, device=device)
        oracle = torch.full((1,), anchor, device=device)
        r_on = run_full_trajectory(net, stim, oracle, device, task_state=task_state)
        with feedback_disabled(net):
            r_off = run_full_trajectory(net, stim, oracle, device, task_state=task_state)
        resp_on = _window_mean(r_on, window).squeeze(0).cpu().numpy()
        resp_off = _window_mean(r_off, window).squeeze(0).cpu().numpy()
        fwhm_on_list.append(fwhm_from_curve(prefs, resp_on, period))
        fwhm_off_list.append(fwhm_from_curve(prefs, resp_off, period))
    fwhm_on = float(np.mean(fwhm_on_list))
    fwhm_off = float(np.mean(fwhm_off_list))
    return {
        "task_state": list(task_state),
        "window": window,
        "fwhm_on": fwhm_on,
        "fwhm_off": fwhm_off,
        "fwhm_delta": fwhm_on - fwhm_off,
    }


# ---------- Top-level driver + verdict ----------


def _row(tag: str, d10: float, d5: float = None, d15: float = None) -> str:
    parts = [f"{tag:20s}"]
    parts.append(f"δ=10°  {d10:+.4f}")
    if d5 is not None:
        parts.append(f"δ=5°  {d5:+.4f}")
    if d15 is not None:
        parts.append(f"δ=15° {d15:+.4f}")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-txt", required=True)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--noise-std", type=float, default=0.3)
    ap.add_argument("--readout-noise-std", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)

    net, _model_cfg, _train_cfg = load_model(args.checkpoint, args.config, device)

    # All (task_state, window) combinations we will compute.
    regimes = {
        "baseline": (0.0, 0.0),
        "focused":  (1.0, 0.0),
        "routine":  (0.0, 1.0),
    }
    # Phase 1C: report all four windows. 'default' is kept because the
    # preregistered gates key off it; early/mid/late are the diagnostic
    # slices required by the full dispatch.
    windows = ["default", "early", "mid", "late"]

    m7_results: dict = {}
    for rname, rstate in regimes.items():
        m7_results[rname] = {}
        for w in windows:
            print(f"Running M7 {rname:8s} window={w}...", flush=True)
            m7_results[rname][w] = extended_m7_single_window(
                net, device, rstate, w,
                n_train=args.n_train, n_test=args.n_test,
                noise_std=args.noise_std, readout_noise_std=args.readout_noise_std,
                seed=args.seed,
            )
            d10 = m7_results[rname][w]["delta_10"]["delta_acc"]
            print(f"  δ=10° delta_acc = {d10:+.6f}", flush=True)

    print()
    print("Running M10 amplitude ratios per (regime, window)...", flush=True)
    m10_results: dict = {}
    for rname, rstate in regimes.items():
        m10_results[rname] = {}
        for w in windows:
            d = m10_amplitude_ratio(net, device, rstate,
                                    n_trials=80, seed=args.seed, window=w)
            m10_results[rname][w] = d
            print(f"  {rname:8s} window={w:7s} amp_ratio = {d['mean_ratio']:.4f}")

    print()
    print("Running FWHM deltas per (regime, window)...", flush=True)
    fwhm_results: dict = {}
    for rname, rstate in regimes.items():
        fwhm_results[rname] = {}
        for w in windows:
            d = fwhm_delta(net, device, rstate, window=w)
            fwhm_results[rname][w] = d
            print(f"  {rname:8s} window={w:7s} "
                  f"fwhm_on={d['fwhm_on']:.2f}  fwhm_off={d['fwhm_off']:.2f}  "
                  f"Δ={d['fwhm_delta']:+.2f}")

    # --- Phase 1C: channel profile + E:I ratio per (regime, window) ---
    print()
    print("Running channel_profile per (regime, window)...", flush=True)
    channel_results: dict = {}
    for rname, rstate in regimes.items():
        channel_results[rname] = {}
        for w in windows:
            d = channel_profile(net, device, rstate, window=w)
            channel_results[rname][w] = d
            print(f"  {rname:8s} window={w:7s} "
                  f"peak={d['peak']:.4f}  near={d['near_flank']:.4f}  "
                  f"far={d['far_flank']:.4f}  c_exc={d['c_exc_peak']:.4f}  "
                  f"|som|={d['som_peak']:.4f}")

    print()
    print("Running E:I ratio per (regime, window)...", flush=True)
    ei_results: dict = {}
    for rname, rstate in regimes.items():
        ei_results[rname] = {}
        for w in windows:
            d = ei_ratio(net, device, rstate, window=w)
            ei_results[rname][w] = d
            print(f"  {rname:8s} window={w:7s} "
                  f"E={d['E']:.4f}  I={d['I']:.4f}  E/I={d['EI_ratio']:.4f}")

    # --- Phase 2: alpha_net causal E/I gate outputs per regime (if present) ---
    print()
    print("Running gate diagnostics per regime (Phase 2)...", flush=True)
    gate_results: dict = {}
    for rname, rstate in regimes.items():
        gate_results[rname] = gate_diagnostics(net, device, rstate)
        g = gate_results[rname]
        if g["use_ei_gate"]:
            print(f"  {rname:8s} "
                  f"g_E={g['g_E_mean']:.4f}±{g['g_E_std']:.4f} "
                  f"[{g['g_E_min']:.4f},{g['g_E_max']:.4f}]  "
                  f"g_I={g['g_I_mean']:.4f}±{g['g_I_std']:.4f} "
                  f"[{g['g_I_min']:.4f},{g['g_I_max']:.4f}]  "
                  f"n={g['n_samples']}")
        else:
            print(f"  {rname:8s} (use_ei_gate=False — alpha_net absent)")

    # --- Preregistered gates (Phase 1A) — all must hold for PASS ---
    # Primary metric for gates: default-window δ=10° on baseline/focused/routine.
    m7_focused = m7_results["focused"]["default"]["delta_10"]["delta_acc"]
    m7_routine = m7_results["routine"]["default"]["delta_10"]["delta_acc"]
    m7_diff = m7_focused - m7_routine
    m10_amp_routine = m10_results["routine"]["default"]["mean_ratio"]

    gates = {
        "m7_focused_gt_p03":   m7_focused > +0.03,
        "m7_routine_lt_m03":   m7_routine < -0.03,
        "m10_amp_routine_lt_0p9": m10_amp_routine < 0.9,
        "m7_diff_gt_p06":      abs(m7_diff) > 0.06,
    }
    all_pass = all(gates.values())
    diff_gt_p03 = abs(m7_diff) > 0.03
    diff_lt_p02 = abs(m7_diff) < 0.02

    if all_pass:
        verdict = "PASS"
    elif diff_gt_p03:
        verdict = "PARTIAL"
    elif diff_lt_p02:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"   # in the gap 0.02..0.03 — treat as partial (no hypothesis broken)

    # --- Phase 1C: early-minus-late deltas per regime for the core metrics ---
    early_minus_late: dict = {}
    for rname in regimes:
        m7_d10_early = m7_results[rname]["early"]["delta_10"]["delta_acc"]
        m7_d10_late  = m7_results[rname]["late"]["delta_10"]["delta_acc"]
        m10_early    = m10_results[rname]["early"]["mean_ratio"]
        m10_late     = m10_results[rname]["late"]["mean_ratio"]
        fwhm_early   = fwhm_results[rname]["early"]["fwhm_delta"]
        fwhm_late    = fwhm_results[rname]["late"]["fwhm_delta"]
        peak_early   = channel_results[rname]["early"]["peak"]
        peak_late    = channel_results[rname]["late"]["peak"]
        ei_early     = ei_results[rname]["early"]["EI_ratio"]
        ei_late      = ei_results[rname]["late"]["EI_ratio"]
        early_minus_late[rname] = {
            "m7_delta10":    m7_d10_early - m7_d10_late,
            "m10_amp_ratio": m10_early    - m10_late,
            "fwhm_delta":    fwhm_early   - fwhm_late,
            "peak":          peak_early   - peak_late,
            "ei_ratio":      ei_early     - ei_late,
        }

    out = {
        "paradigm": "dual_1a_phase1a_extended_m7",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "noise_std": args.noise_std,
        "readout_noise_std": args.readout_noise_std,
        "m7_by_regime_by_window": m7_results,
        "m10_amplitude_ratio_by_regime": m10_results,
        "fwhm_by_regime": fwhm_results,
        "channel_profile_by_regime": channel_results,
        "ei_ratio_by_regime": ei_results,
        "gate_diagnostics_by_regime": gate_results,
        "early_minus_late_by_regime": early_minus_late,
        "gates": gates,
        "gate_values": {
            "m7_focused": m7_focused,
            "m7_routine": m7_routine,
            "m7_diff": m7_diff,
            "m10_amp_routine": m10_amp_routine,
        },
        "verdict": verdict,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2, default=float)

    # --- Human-readable comparison ---
    lines = []
    lines.append("=" * 78)
    lines.append("Phase 1A dual_1a extended M7 + M10 + FWHM report")
    lines.append("=" * 78)
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Seed:       {args.seed}")
    lines.append("")
    lines.append("M7 delta_acc per (regime, window, delta):")
    header = f"{'regime':10s} {'window':10s} {'δ=3°':>9s} {'δ=5°':>9s} {'δ=10°':>9s} {'δ=15°':>9s}"
    lines.append(header)
    lines.append("-" * len(header))
    for rname in regimes:
        for w in windows:
            row = m7_results[rname][w]
            d3 = row['delta_3']['delta_acc']
            d5 = row['delta_5']['delta_acc']
            d10 = row['delta_10']['delta_acc']
            d15 = row['delta_15']['delta_acc']
            lines.append(f"{rname:10s} {w:10s} {d3:+9.4f} {d5:+9.4f} {d10:+9.4f} {d15:+9.4f}")
    lines.append("")
    lines.append("M10 amplitude ratio (FB-ON / FB-OFF) per (regime, window):")
    m10_header = f"{'regime':10s} {'window':10s} {'ratio':>9s} {'on':>9s} {'off':>9s}"
    lines.append(m10_header)
    lines.append("-" * len(m10_header))
    for rname in regimes:
        for w in windows:
            r = m10_results[rname][w]
            lines.append(f"{rname:10s} {w:10s} {r['mean_ratio']:9.4f} "
                         f"{r['mean_on_magnitude']:9.4f} {r['mean_off_magnitude']:9.4f}")
    lines.append("")
    lines.append("FWHM (mean over 8 anchors) per (regime, window):")
    fwhm_header = f"{'regime':10s} {'window':10s} {'fwhm_on':>9s} {'fwhm_off':>9s} {'Δ':>9s}"
    lines.append(fwhm_header)
    lines.append("-" * len(fwhm_header))
    for rname in regimes:
        for w in windows:
            f_ = fwhm_results[rname][w]
            lines.append(f"{rname:10s} {w:10s} {f_['fwhm_on']:9.2f} "
                         f"{f_['fwhm_off']:9.2f} {f_['fwhm_delta']:+9.2f}")
    lines.append("")
    lines.append("Channel profile per (regime, window):")
    lines.append("  peak = r_l23 at anchor; near_flank = r_l23 at ±2 ch (±10°)")
    lines.append("  far_flank = r_l23 at ±10 ch (±50°); c_exc/|som| at peak channel")
    cp_header = (f"{'regime':10s} {'window':10s} {'peak':>9s} "
                 f"{'near':>9s} {'far':>9s} {'c_exc':>9s} {'|som|':>9s}")
    lines.append(cp_header)
    lines.append("-" * len(cp_header))
    for rname in regimes:
        for w in windows:
            c = channel_results[rname][w]
            lines.append(f"{rname:10s} {w:10s} {c['peak']:9.4f} "
                         f"{c['near_flank']:9.4f} {c['far_flank']:9.4f} "
                         f"{c['c_exc_peak']:9.4f} {c['som_peak']:9.4f}")
    lines.append("")
    lines.append("Canonical E:I ratio per (regime, window)  (E = |r_l4|+|r_l23|, I = |r_pv|+|r_som|):")
    ei_header = (f"{'regime':10s} {'window':10s} {'E':>9s} {'I':>9s} {'E/I':>9s} "
                 f"{'|r_l4|':>9s} {'|r_l23|':>9s} {'|r_pv|':>9s} {'|r_som|':>9s}")
    lines.append(ei_header)
    lines.append("-" * len(ei_header))
    for rname in regimes:
        for w in windows:
            e = ei_results[rname][w]
            lines.append(f"{rname:10s} {w:10s} {e['E']:9.4f} "
                         f"{e['I']:9.4f} {e['EI_ratio']:9.4f} "
                         f"{e['mean_r_l4']:9.4f} {e['mean_r_l23']:9.4f} "
                         f"{e['mean_r_pv']:9.4f} {e['mean_r_som']:9.4f}")
    lines.append("")
    # Phase 2 gate outputs per regime (only when use_ei_gate=True)
    if any(gate_results[r]["use_ei_gate"] for r in regimes):
        lines.append("Phase 2 alpha_net causal E/I gate outputs:")
        lines.append(
            "  (distribution over 8 anchors × T timesteps — identity init ⇒ g ≈ 1.0)"
        )
        g_header = (
            f"{'regime':10s} {'g_E_mean':>9s} {'g_E_std':>9s} "
            f"{'g_I_mean':>9s} {'g_I_std':>9s} "
            f"{'g_E_range':>17s} {'g_I_range':>17s} {'n':>6s}"
        )
        lines.append(g_header)
        lines.append("-" * len(g_header))
        for rname in regimes:
            g = gate_results[rname]
            g_E_range = f"[{g['g_E_min']:.3f},{g['g_E_max']:.3f}]"
            g_I_range = f"[{g['g_I_min']:.3f},{g['g_I_max']:.3f}]"
            lines.append(
                f"{rname:10s} "
                f"{g['g_E_mean']:9.4f} {g['g_E_std']:9.4f} "
                f"{g['g_I_mean']:9.4f} {g['g_I_std']:9.4f} "
                f"{g_E_range:>17s} {g_I_range:>17s} {g['n_samples']:6d}"
            )
        lines.append("")
        # Cross-regime gate contrast (load-bearing for Phase 2 decision tree)
        if all(gate_results[r]["use_ei_gate"] for r in ("focused", "routine")):
            gf = gate_results["focused"]
            gr = gate_results["routine"]
            lines.append(
                "  focused↔routine gate contrast: "
                f"Δg_E = {gf['g_E_mean'] - gr['g_E_mean']:+.4f}   "
                f"Δg_I = {gf['g_I_mean'] - gr['g_I_mean']:+.4f}"
            )
            lines.append(
                "  (identity/no-learning signature: both Δ ≈ 0 AND all means ≈ 1.0)"
            )
            lines.append("")
    lines.append("Early-minus-late deltas per regime (early − late):")
    em_header = (f"{'regime':10s} {'m7_δ10':>10s} {'m10_amp':>10s} "
                 f"{'fwhm_Δ':>10s} {'peak':>10s} {'E/I':>10s}")
    lines.append(em_header)
    lines.append("-" * len(em_header))
    for rname in regimes:
        d = early_minus_late[rname]
        lines.append(f"{rname:10s} {d['m7_delta10']:+10.4f} {d['m10_amp_ratio']:+10.4f} "
                     f"{d['fwhm_delta']:+10.4f} {d['peak']:+10.4f} "
                     f"{d['ei_ratio']:+10.4f}")
    lines.append("")
    lines.append("Preregistered Phase 1A gates (default window, δ=10°):")
    lines.append(f"  m7_focused        = {m7_focused:+.4f}  (need > +0.03)  {'PASS' if gates['m7_focused_gt_p03'] else 'FAIL'}")
    lines.append(f"  m7_routine        = {m7_routine:+.4f}  (need < -0.03)  {'PASS' if gates['m7_routine_lt_m03'] else 'FAIL'}")
    lines.append(f"  |focused-routine| = {abs(m7_diff):+.4f}  (need > +0.06)  {'PASS' if gates['m7_diff_gt_p06'] else 'FAIL'}")
    lines.append(f"  m10_amp_routine   = {m10_amp_routine:.4f}  (need < 0.9)   {'PASS' if gates['m10_amp_routine_lt_0p9'] else 'FAIL'}")
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append("=" * 78)

    with open(args.output_txt, "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_txt}")


if __name__ == "__main__":
    main()
