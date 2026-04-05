#!/usr/bin/env python3
"""Representation analysis: feedback ON vs OFF for a trained laminar V1-V2 model.

Compares 5 representational metrics between feedback-enabled and feedback-ablated
runs of the same trained network, driven by an oracle expectation template:

    1. Tuning curve FWHM (channel tuned to expected orientation)
    2. Preferred-channel peak gain (mean across channels)
    3. Channel-by-distance population profile at stim == expected
    4. Fine-discrimination linear-decoder accuracy at ±5°/±10°/±15°
    5. Total L2/3 energy (sum of activity, feedback-on vs feedback-off ratio)

Feedback is ablated by zeroing the learned operator's inhibitory weights
(alpha_inh) and the delta-SOM baseline (som_baseline), under a context manager
that restores them afterwards. This is only valid for EmergentFeedbackOperator.

Usage:
    python3 scripts/analyze_representation.py \
        --checkpoint results/hardening/deviance_v2_s42/center_surround_seed42/checkpoint.pt \
        --config config/exp_deviance.yaml

    python3 scripts/analyze_representation.py \
        --checkpoint results/hardening/p4_v2_s42/center_surround_seed42/checkpoint.pt \
        --config config/exp_sharp_p4.yaml

The comparison table is printed to stdout. All intermediate values (tuning
curves, discrimination accuracies per delta, FWHM per seed orientation) are
also printed so a reviewer can sanity-check the computation.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainingConfig, load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading & feedback toggle
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> tuple[LaminarV1V2Network, ModelConfig, TrainingConfig]:
    """Load a trained LaminarV1V2Network from a checkpoint.

    Prefers the config saved inside the checkpoint (ModelConfig round-trip);
    falls back to loading from the YAML config file if needed. This matches
    the pattern established in scripts/run_experiments.py.

    Args:
        checkpoint_path: Path to checkpoint.pt.
        config_path: Path to YAML config (used if checkpoint has no saved config).
        device: Torch device for the loaded model.

    Returns:
        (net, model_cfg, train_cfg)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" in ckpt and "model" in ckpt["config"]:
        model_raw = dict(ckpt["config"]["model"])
        train_raw = dict(ckpt["config"]["training"])
        model_cfg = ModelConfig(**model_raw)
        # TrainingConfig round-trip: strip keys that aren't constructor args.
        # This is unused except for delta_som & oracle_pi, but we keep it so
        # the caller can introspect training conditions.
        train_fields = set(TrainingConfig.__dataclass_fields__.keys())
        train_cfg = TrainingConfig(**{k: v for k, v in train_raw.items() if k in train_fields})
        logger.info(f"Loaded config from checkpoint: mechanism={model_cfg.mechanism.value}, "
                    f"feedback_mode={model_cfg.feedback_mode}, delta_som={train_cfg.delta_som}, "
                    f"oracle_pi(train)={train_cfg.oracle_pi}")
    else:
        model_cfg, train_cfg, _ = load_config(config_path)
        logger.warning(f"No config in checkpoint; using {config_path}")

    if model_cfg.feedback_mode != "emergent":
        raise ValueError(
            f"This analysis script only supports emergent feedback; "
            f"got feedback_mode={model_cfg.feedback_mode}"
        )

    net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
    net.load_state_dict(ckpt["model_state"])
    net.to(device)
    net.eval()
    # Make sure the warm-up scale is at its fully-on value (training leaves it at 1.0
    # after ramp, but set it explicitly so we never analyze with a partial scale).
    net.feedback_scale.fill_(1.0)

    logger.info(f"Loaded model from {checkpoint_path}")
    return net, model_cfg, train_cfg


@contextlib.contextmanager
def feedback_disabled(net: LaminarV1V2Network):
    """Temporarily zero the emergent feedback operator.

    Saves and restores alpha_inh (and som_baseline if delta-SOM is enabled).
    With alpha_inh == 0 the circulant kernel is zero, so inh_field == 0 and
    the SOM drive becomes pi_eff * (softplus(baseline) - softplus(baseline)) == 0
    in delta-SOM mode, or pi_eff * softplus(0) = pi_eff * log(2) without delta-SOM.
    We therefore also zero som_baseline (delta-SOM) as a belt-and-braces guarantee
    that the SOM ring receives no drive while feedback is "off".

    Note: this only affects the feedback pathway. L4, PV, L2/3 recurrence, and
    adaptation all remain active — exactly what we want for an "ablate feedback,
    keep everything else" comparison.
    """
    fb = net.feedback
    saved_alpha = fb.alpha_inh.detach().clone()
    saved_baseline = None
    if hasattr(fb, "som_baseline"):
        saved_baseline = fb.som_baseline.detach().clone()
    try:
        with torch.no_grad():
            fb.alpha_inh.zero_()
            if saved_baseline is not None:
                fb.som_baseline.zero_()
        yield
    finally:
        with torch.no_grad():
            fb.alpha_inh.copy_(saved_alpha)
            if saved_baseline is not None:
                fb.som_baseline.copy_(saved_baseline)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

T_STEPS = 25            # total timesteps (stimulus on whole time, no ISI)
READ_WINDOW = 5         # average L2/3 over the last READ_WINDOW steps
EVAL_PI = 5.0           # oracle precision used during analysis (matches HARDENING_RESULTS pi=5)
EVAL_CONTRAST = 1.0     # stimulus contrast used during analysis


def _build_oracle(net: LaminarV1V2Network, oracle_thetas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build oracle_q_pred and oracle_pi_pred tensors for a batch of trials.

    Args:
        net: the network (used only for _make_bump and cfg).
        oracle_thetas: [B] — oracle target orientation (degrees) per trial.

    Returns:
        oracle_q: [B, T_STEPS, N] — bump at oracle_theta, held constant over time,
                  row-normalised to sum to 1.
        oracle_pi: [B, T_STEPS, 1] — constant EVAL_PI.
    """
    N = net.cfg.n_orientations
    B = oracle_thetas.shape[0]
    device = oracle_thetas.device
    q_single = net._make_bump(oracle_thetas)  # [B, N]
    q_single = q_single / (q_single.sum(dim=-1, keepdim=True) + 1e-8)
    oracle_q = q_single.unsqueeze(1).expand(B, T_STEPS, N).contiguous()
    oracle_pi = torch.full((B, T_STEPS, 1), EVAL_PI, device=device)
    return oracle_q, oracle_pi


def run_trials(
    net: LaminarV1V2Network,
    stim_thetas: torch.Tensor,
    oracle_thetas: torch.Tensor,
    device: torch.device,
    contrast: float = EVAL_CONTRAST,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Run the network on a batch of trials and return the readout L2/3 activity.

    Each trial is a single stimulus presentation held for T_STEPS timesteps,
    with the oracle expectation template held constant throughout. We read
    the mean L2/3 activity over the last READ_WINDOW timesteps (by which
    point the recurrent circuit has settled given tau_l23=10).

    Args:
        net: the trained network (in eval mode).
        stim_thetas: [B] — stimulus orientation per trial (degrees).
        oracle_thetas: [B] — oracle orientation per trial (degrees).
        device: torch device.
        contrast: stimulus contrast (default 1.0).
        noise_std: Gaussian noise added to the population-coded stimulus
                   (matches train_cfg.stimulus_noise semantics). 0 disables.
        seed: optional seed for noise reproducibility.

    Returns:
        r_l23: [B, N] — mean L2/3 activity over the read window.
    """
    B = stim_thetas.shape[0]
    N = net.cfg.n_orientations
    assert oracle_thetas.shape == (B,), f"oracle shape {oracle_thetas.shape} != ({B},)"

    # Build single-frame population-coded stimulus [B, N]
    stim_frame = generate_grating(
        stim_thetas.to(device),
        torch.full((B,), contrast, device=device),
        n_orientations=N,
        sigma=net.cfg.sigma_ff,
        n=net.cfg.naka_rushton_n,
        c50=net.cfg.naka_rushton_c50,
        period=net.cfg.orientation_range,
    )  # [B, N]

    # Expand to [B, T, N] with stimulus held on for the full duration (no ISI).
    stim_seq = stim_frame.unsqueeze(1).expand(B, T_STEPS, N).contiguous()

    if noise_std > 0:
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            noise = torch.randn(stim_seq.shape, device=device, generator=gen) * noise_std
        else:
            noise = torch.randn_like(stim_seq) * noise_std
        stim_seq = (stim_seq + noise).clamp(min=0.0)

    # Oracle template held constant over time
    oracle_q, oracle_pi = _build_oracle(net, oracle_thetas.to(device))

    cue_seq = torch.zeros(B, T_STEPS, N, device=device)
    task_seq = torch.zeros(B, T_STEPS, 2, device=device)
    packed = net.pack_inputs(stim_seq, cue_seq, task_seq)

    # Inject oracle
    net.oracle_mode = True
    net.oracle_q_pred = oracle_q
    net.oracle_pi_pred = oracle_pi
    try:
        with torch.no_grad():
            r_l23_all, _, _ = net(packed)
    finally:
        net.oracle_mode = False
        net.oracle_q_pred = None
        net.oracle_pi_pred = None

    # r_l23_all: [B, T, N]. Average the last READ_WINDOW steps.
    r_l23 = r_l23_all[:, -READ_WINDOW:].mean(dim=1)  # [B, N]
    return r_l23


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def fwhm_from_curve(thetas: np.ndarray, response: np.ndarray, period: float) -> float:
    """Compute full-width at half-maximum of a 1D tuning curve.

    Uses linear interpolation between sample points. Assumes a single peak.
    Handles circularity by allowing the half-crossings to wrap around.

    Args:
        thetas: [K] sorted orientation samples (degrees).
        response: [K] response at each orientation.
        period: orientation range (degrees) for circular wrap.

    Returns:
        FWHM in degrees.
    """
    assert thetas.shape == response.shape
    peak_idx = int(np.argmax(response))
    peak_val = float(response[peak_idx])
    baseline = float(response.min())
    half = baseline + 0.5 * (peak_val - baseline)
    if peak_val <= baseline + 1e-8:
        return float("nan")

    # Walk outward from peak on each side, interpolating the crossing.
    K = thetas.shape[0]

    def _cross(direction: int) -> float:
        prev_theta = float(thetas[peak_idx])
        prev_val = peak_val
        wrap_accum = 0.0
        for k in range(1, K):
            idx = (peak_idx + direction * k) % K
            cur_theta = float(thetas[idx])
            cur_val = float(response[idx])
            # Unwrap theta onto a monotonic axis around the peak
            dtheta = (cur_theta - prev_theta)
            if direction > 0 and dtheta < 0:
                dtheta += period
            if direction < 0 and dtheta > 0:
                dtheta -= period
            wrap_accum += dtheta
            if cur_val <= half:
                # Linear interpolation between prev and cur
                frac = (prev_val - half) / (prev_val - cur_val + 1e-12)
                return abs(wrap_accum - dtheta * (1 - frac))
            prev_theta = cur_theta
            prev_val = cur_val
        return float("nan")  # no crossing found

    left = _cross(-1)
    right = _cross(+1)
    if math.isnan(left) or math.isnan(right):
        return float("nan")
    return left + right


def _dists_from(prefs: torch.Tensor, theta: float, period: float) -> torch.Tensor:
    """Signed minimum circular distance [N] from each pref to theta (range -period/2 .. +period/2)."""
    d = (prefs - theta + period / 2.0) % period - period / 2.0
    return d


def _closest_channel(prefs: torch.Tensor, theta: float, period: float) -> int:
    return int(_dists_from(prefs, theta, period).abs().argmin().item())


def metric1_population_bump(
    net: LaminarV1V2Network,
    device: torch.device,
    conditions: list[tuple[float, float]],
) -> dict:
    """Metric 1: FWHM of the *population* bump at a fixed stimulus.

    For each (stim_theta, oracle_theta) condition, present a single stimulus
    with the oracle template held at oracle_theta. Read all 36 L2/3 channels
    — this is the population response to a single stimulus — sort by their
    preferred orientation relative to stim_theta, and compute the FWHM of the
    resulting spatial bump.

    A sharpening operator should narrow this bump when oracle == stim. A
    dampening operator should attenuate it uniformly (peak drops, FWHM
    unchanged — except for any threshold-rectification artifact, see the
    `artifact_check_threshold` metric).

    Args:
        net: trained network.
        device: torch device.
        conditions: list of (stim_theta, oracle_theta) pairs in degrees.

    Returns:
        dict keyed by "(stim,oracle)" with per-condition results.
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    prefs = torch.arange(N, dtype=torch.float32) * step

    results: dict[str, dict] = {}

    for stim_theta, oracle_theta in conditions:
        stim = torch.tensor([stim_theta], device=device)
        oracle = torch.tensor([oracle_theta], device=device)

        r_on = run_trials(net, stim, oracle, device, noise_std=0.0)[0].cpu().numpy()  # [N]
        with feedback_disabled(net):
            r_off = run_trials(net, stim, oracle, device, noise_std=0.0)[0].cpu().numpy()

        # Sort by signed distance from stim_theta so the population bump is centered.
        dists = _dists_from(prefs, stim_theta, period).numpy()  # [N]
        order = np.argsort(dists)
        dists_sorted = dists[order]
        r_on_sorted = r_on[order]
        r_off_sorted = r_off[order]

        # FWHM of the population bump (as a function of signed distance from stim_theta).
        # The distance axis is monotonic (it's a single-period slice), but the
        # signal is still circular, so we use the same fwhm_from_curve routine
        # treating `dists_sorted` as the theta axis (its range covers the full period).
        fwhm_on = fwhm_from_curve(dists_sorted + period / 2.0, r_on_sorted, period)
        fwhm_off = fwhm_from_curve(dists_sorted + period / 2.0, r_off_sorted, period)

        key = f"stim={stim_theta:.0f},ora={oracle_theta:.0f}"
        results[key] = {
            "stim_theta": stim_theta,
            "oracle_theta": oracle_theta,
            "dists": dists_sorted,
            "r_on": r_on_sorted,
            "r_off": r_off_sorted,
            "peak_on": float(r_on_sorted.max()),
            "peak_off": float(r_off_sorted.max()),
            "fwhm_on": fwhm_on,
            "fwhm_off": fwhm_off,
            "delta_fwhm": fwhm_on - fwhm_off,
            "pct_fwhm": 100.0 * (fwhm_on - fwhm_off) / (fwhm_off + 1e-12),
        }
    return results


def metric1b_flanking_tuning(
    net: LaminarV1V2Network,
    device: torch.device,
    oracle_theta: float = 90.0,
    flanking_offsets: list[float] = None,
    n_stim: int = 36,
) -> dict:
    """Metric 1b: tuning curves of FLANKING channels (at fixed distance from oracle).

    For each flanking_offset d in `flanking_offsets`:
      - Target channel = channel whose pref is closest to oracle_theta + d
      - Vary stimulus 0..175° (36 values at 5° spacing)
      - Oracle fixed at oracle_theta
      - Record target channel response → tuning curve
      - Compute FWHM and peak

    P4 sharpening should show the strongest feedback effects here, because the
    learned kernel has mixed-sign basis coefficients that place suppression at
    flanks (not at center).
    """
    if flanking_offsets is None:
        flanking_offsets = [-30.0, -25.0, -20.0, -10.0, 0.0, 10.0, 20.0, 25.0, 30.0]

    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    orient_step = period / n_stim
    stim_grid = (torch.arange(n_stim, dtype=torch.float32) * orient_step).to(device)
    prefs = torch.arange(N, dtype=torch.float32) * step

    oracle_thetas = torch.full((n_stim,), oracle_theta, device=device)

    r_on = run_trials(net, stim_grid, oracle_thetas, device, noise_std=0.0).cpu().numpy()  # [n_stim, N]
    with feedback_disabled(net):
        r_off = run_trials(net, stim_grid, oracle_thetas, device, noise_std=0.0).cpu().numpy()

    stim_grid_np = stim_grid.cpu().numpy()
    results: dict[float, dict] = {}
    for offset in flanking_offsets:
        target_theta = (oracle_theta + offset) % period
        channel = _closest_channel(prefs, target_theta, period)

        curve_on = r_on[:, channel]
        curve_off = r_off[:, channel]
        fwhm_on = fwhm_from_curve(stim_grid_np, curve_on, period)
        fwhm_off = fwhm_from_curve(stim_grid_np, curve_off, period)

        results[offset] = {
            "offset": offset,
            "channel": channel,
            "pref": float(prefs[channel].item()),
            "stim_grid": stim_grid_np,
            "curve_on": curve_on,
            "curve_off": curve_off,
            "peak_on": float(curve_on.max()),
            "peak_off": float(curve_off.max()),
            "peak_ratio": float(curve_on.max() / (curve_off.max() + 1e-12)),
            "fwhm_on": fwhm_on,
            "fwhm_off": fwhm_off,
            "delta_fwhm": fwhm_on - fwhm_off,
        }
    return results


def metric2_peak_gain(net: LaminarV1V2Network, device: torch.device) -> dict:
    """Metric 2: peak-response gain across all 36 channels.

    For each preferred channel i, present stim at pref_i, oracle also at pref_i,
    record r_l23[i] (the peak of that channel's tuning). Average across channels.
    Report fb_on and fb_off peak means, plus ratio.
    """
    N = net.cfg.n_orientations
    step = net.cfg.orientation_range / N
    prefs = torch.arange(N, dtype=torch.float32, device=device) * step

    # Batch: B = N, stim_theta = pref_i, oracle = pref_i
    r_on = run_trials(net, prefs, prefs, device, noise_std=0.0)       # [N, N]
    with feedback_disabled(net):
        r_off = run_trials(net, prefs, prefs, device, noise_std=0.0)

    # For each trial i, read channel i (the channel tuned to the stim/oracle).
    idx = torch.arange(N, device=device)
    peaks_on = r_on[idx, idx].cpu().numpy()   # [N]
    peaks_off = r_off[idx, idx].cpu().numpy()

    return {
        "peaks_on": peaks_on,
        "peaks_off": peaks_off,
        "peak_on_mean": float(peaks_on.mean()),
        "peak_off_mean": float(peaks_off.mean()),
        "ratio_on_over_off": float(peaks_on.mean() / (peaks_off.mean() + 1e-12)),
    }


def metric3_channel_profile(
    net: LaminarV1V2Network,
    device: torch.device,
    stim_theta: float = 90.0,
) -> dict:
    """Metric 3: full population profile at a single stim/oracle orientation.

    Fix stim == oracle == stim_theta. Read all 36 L2/3 channels. Express as
    a function of angular distance from stim_theta.
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    prefs = torch.arange(N, dtype=torch.float32) * step

    stim = torch.tensor([stim_theta], device=device)
    oracle = torch.tensor([stim_theta], device=device)

    r_on = run_trials(net, stim, oracle, device, noise_std=0.0)[0]    # [N]
    with feedback_disabled(net):
        r_off = run_trials(net, stim, oracle, device, noise_std=0.0)[0]

    dists = (((prefs - stim_theta) + period / 2) % period) - period / 2  # [N], signed
    # Sort by distance
    order = torch.argsort(dists)
    dists_sorted = dists[order].numpy()
    r_on_sorted = r_on[order].cpu().numpy()
    r_off_sorted = r_off[order].cpu().numpy()

    # Normalise by peak (for shape comparison)
    r_on_norm = r_on_sorted / (r_on_sorted.max() + 1e-12)
    r_off_norm = r_off_sorted / (r_off_sorted.max() + 1e-12)

    return {
        "stim_theta": stim_theta,
        "dists": dists_sorted,
        "r_on": r_on_sorted,
        "r_off": r_off_sorted,
        "r_on_norm": r_on_norm,
        "r_off_norm": r_off_norm,
    }


def metric4_fine_discrimination(
    net: LaminarV1V2Network,
    device: torch.device,
    deltas: list[float],
    n_trials_per_class: int = 500,
    readout_noise_stds: list[float] = None,
    subset_channels: list[int] = None,
    base_theta: float = 90.0,
    seed: int = 12345,
) -> dict:
    """Metric 4: linear-decoder discrimination with READOUT noise on a subset of channels.

    Previous version added noise to the stimulus, but the L2/3 response was still
    so clean that even 5° discrimination saturated at 100%. This version:

    - Adds Gaussian noise directly to the L2/3 readout (post-network) at several
      levels so we can scan the operating point.
    - Uses only channels near base_theta (default ±15°, i.e. channels 15-21 for
      base_theta=90°) so a trivial population-vector argmax cannot solve it.
    - Runs feedback on/off with identical stimuli and identical readout-noise
      realizations — the only difference between the two conditions is the
      feedback operator itself.

    For each (noise_std, delta):
      - 500 trials at base_theta (class 0), 500 at base_theta+delta (class 1)
      - Oracle held at base_theta for all trials (expectation)
      - Add i.i.d. Gaussian noise to L2/3 readout, take subset of channels
      - Train LogisticRegression on 80/20 split
      - Report test accuracy on vs off

    Returns:
        {(noise_std, delta): {acc_on, acc_off, n_train, n_test, channels}}
    """
    if readout_noise_stds is None:
        readout_noise_stds = [0.01, 0.03, 0.10]
    period = net.cfg.orientation_range
    N = net.cfg.n_orientations
    step = period / N

    if subset_channels is None:
        prefs = torch.arange(N, dtype=torch.float32) * step
        d = _dists_from(prefs, base_theta, period).abs().numpy()
        subset_channels = [int(i) for i in np.where(d <= 15.0)[0]]

    results: dict = {}

    # Generate the clean (noise-free) L2/3 responses ONCE per delta, for both
    # feedback on and feedback off. Readout noise is added afterwards.
    clean_by_delta: dict[float, dict] = {}
    for delta in deltas:
        thetas_a = torch.full((n_trials_per_class,), base_theta, device=device)
        thetas_b = torch.full((n_trials_per_class,), (base_theta + delta) % period, device=device)
        stim = torch.cat([thetas_a, thetas_b], dim=0)
        oracle = torch.full_like(stim, base_theta)
        labels = np.concatenate([
            np.zeros(n_trials_per_class, dtype=np.int64),
            np.ones(n_trials_per_class, dtype=np.int64),
        ])
        r_on = run_trials(net, stim, oracle, device, noise_std=0.0).cpu().numpy()   # [2K, N]
        with feedback_disabled(net):
            r_off = run_trials(net, stim, oracle, device, noise_std=0.0).cpu().numpy()
        clean_by_delta[delta] = {"r_on": r_on, "r_off": r_off, "labels": labels}

    for noise_std in readout_noise_stds:
        for delta in deltas:
            clean = clean_by_delta[delta]
            labels = clean["labels"]
            n_total = labels.shape[0]

            # Same noise realisation for on and off (so only difference is feedback).
            rs = np.random.RandomState(seed + int(delta * 10) + int(noise_std * 1e4))
            noise = rs.randn(*clean["r_on"].shape).astype(np.float32) * noise_std
            X_on = clean["r_on"] + noise
            X_off = clean["r_off"] + noise
            X_on = X_on[:, subset_channels]
            X_off = X_off[:, subset_channels]

            perm = rs.permutation(n_total)
            n_train = int(0.8 * n_total)
            train_idx, test_idx = perm[:n_train], perm[n_train:]

            def _fit_and_score(X):
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X[train_idx], labels[train_idx])
                return float(clf.score(X[test_idx], labels[test_idx]))

            acc_on = _fit_and_score(X_on)
            acc_off = _fit_and_score(X_off)

            results[(noise_std, delta)] = {
                "noise_std": noise_std,
                "delta": delta,
                "acc_on": acc_on,
                "acc_off": acc_off,
                "n_train": n_train,
                "n_test": n_total - n_train,
                "channels": subset_channels,
            }
    return results


def metric5_energy_by_distance(
    net: LaminarV1V2Network,
    device: torch.device,
    oracle_theta: float = 90.0,
    n_stim: int = 36,
) -> dict:
    """Metric 5: L2/3 energy split by angular distance from oracle.

    For each of 36 stimuli (oracle fixed at oracle_theta), sum L2/3 activity
    in three annular bins:
      - expected:  channels within 10° of oracle
      - surround:  channels 10°-45° from oracle
      - far:       channels 45°+ from oracle

    Reports total energy and per-bin energy for on and off; the bin where the
    reduction is concentrated tells you whether feedback targets center or
    surround.
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    orient_step = period / n_stim
    stim_grid = (torch.arange(n_stim, dtype=torch.float32) * orient_step).to(device)
    oracles = torch.full((n_stim,), oracle_theta, device=device)
    prefs = torch.arange(N, dtype=torch.float32) * step

    r_on = run_trials(net, stim_grid, oracles, device, noise_std=0.0).cpu().numpy()   # [n_stim, N]
    with feedback_disabled(net):
        r_off = run_trials(net, stim_grid, oracles, device, noise_std=0.0).cpu().numpy()

    d = _dists_from(prefs, oracle_theta, period).abs().numpy()
    expected_mask = d <= 10.0
    surround_mask = (d > 10.0) & (d <= 45.0)
    far_mask = d > 45.0

    def _sum(arr, mask):
        return float(arr[:, mask].sum())

    total_on = float(r_on.sum())
    total_off = float(r_off.sum())
    return {
        "oracle_theta": oracle_theta,
        "n_expected_channels": int(expected_mask.sum()),
        "n_surround_channels": int(surround_mask.sum()),
        "n_far_channels": int(far_mask.sum()),
        "energy_on_total": total_on,
        "energy_off_total": total_off,
        "energy_on_expected": _sum(r_on, expected_mask),
        "energy_off_expected": _sum(r_off, expected_mask),
        "energy_on_surround": _sum(r_on, surround_mask),
        "energy_off_surround": _sum(r_off, surround_mask),
        "energy_on_far": _sum(r_on, far_mask),
        "energy_off_far": _sum(r_off, far_mask),
        "reduction_total_pct": 100.0 * (total_off - total_on) / (total_off + 1e-12),
        "reduction_expected_pct": 100.0 * (_sum(r_off, expected_mask) - _sum(r_on, expected_mask)) / (_sum(r_off, expected_mask) + 1e-12),
        "reduction_surround_pct": 100.0 * (_sum(r_off, surround_mask) - _sum(r_on, surround_mask)) / (_sum(r_off, surround_mask) + 1e-12),
        "reduction_far_pct": 100.0 * (_sum(r_off, far_mask) - _sum(r_on, far_mask)) / (_sum(r_off, far_mask) + 1e-12),
    }


# ---------------------------------------------------------------------------
# Sanity and artifact checks
# ---------------------------------------------------------------------------

def sanity_check_ablation(net: LaminarV1V2Network, device: torch.device) -> dict:
    """Confirm that inside feedback_disabled(), the feedback operator's output is exactly zero.

    Builds a peaked q_pred (bump at 90°) and a unit pi_pred, then calls
    `net.feedback(q, pi)` both with and without the ablation context manager.
    Returns the max-abs of both outputs; the ablated output must be 0 (or within
    float epsilon).
    """
    N = net.cfg.n_orientations
    theta = torch.tensor([90.0], device=device)
    q = net._make_bump(theta)
    q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)  # [1, N], normalised
    pi = torch.tensor([[EVAL_PI]], device=device)

    net.feedback.cache_kernels()
    try:
        with torch.no_grad():
            drive_on = net.feedback(q, pi)
    finally:
        net.feedback.uncache_kernels()
    with feedback_disabled(net):
        net.feedback.cache_kernels()
        try:
            with torch.no_grad():
                drive_off = net.feedback(q, pi)
        finally:
            net.feedback.uncache_kernels()

    return {
        "drive_on_max_abs": float(drive_on.abs().max().item()),
        "drive_off_max_abs": float(drive_off.abs().max().item()),
        "drive_on_sum": float(drive_on.sum().item()),
        "drive_off_sum": float(drive_off.sum().item()),
        "ablation_zero": bool(drive_off.abs().max().item() < 1e-7),
    }


def artifact_check_threshold(
    net: LaminarV1V2Network,
    device: torch.device,
    stim_theta: float = 90.0,
    oracle_theta: float = 90.0,
) -> dict:
    """Threshold/rectification artifact check.

    A feedback operator that uniformly REDUCES the L2/3 drive can make the
    rectified post-softplus tuning curve look narrower, even when the
    pre-rectification drive is just scaled down. This check disentangles the
    two effects by also reporting the tuning curve of the *pre-rectification*
    L2/3 drive.

    Protocol:
      1. Run a single-stimulus trial and capture the full trajectory.
      2. At the last timestep, reconstruct the L2/3 drive manually:
           l23_drive = r_l4[t] + W_rec @ r_l23[t-1] - w_som * r_som[t] - w_pv_l23 * r_pv[t]
         (template_modulation is zero in emergent mode.)
      3. Compute the FWHM of this drive (across the 36 channels) alongside the
         FWHM of the rectified L2/3 rate.

    If dampening's narrowing is a real shape change, the pre-rect drive should
    also narrow. If it's a threshold artifact, the pre-rect drive should scale
    uniformly (peak down, FWHM unchanged).

    Args:
        net: trained network (emergent feedback).
        stim_theta, oracle_theta: scalar conditions.

    Returns:
        dict with drives and rates (pre-rect vs post-rect), plus FWHMs.
    """
    import torch.nn.functional as F

    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    prefs = torch.arange(N, dtype=torch.float32) * step

    def _run(stim_theta_: float, oracle_theta_: float):
        stim = torch.tensor([stim_theta_], device=device)
        oracle = torch.tensor([oracle_theta_], device=device)

        stim_frame = generate_grating(
            stim, torch.ones(1, device=device),
            n_orientations=N, sigma=net.cfg.sigma_ff,
            n=net.cfg.naka_rushton_n, c50=net.cfg.naka_rushton_c50,
            period=net.cfg.orientation_range,
        )
        stim_seq = stim_frame.unsqueeze(1).expand(1, T_STEPS, N).contiguous()
        oracle_q, oracle_pi = _build_oracle(net, oracle)
        packed = net.pack_inputs(stim_seq,
                                 torch.zeros(1, T_STEPS, N, device=device),
                                 torch.zeros(1, T_STEPS, 2, device=device))
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

    from src.utils import rectified_softplus

    def _extract_drive(r_l23_all, aux):
        """Reconstruct L2/3 drive at the last timestep for a single-trial run."""
        with torch.no_grad():
            r_l4_last = aux["r_l4_all"][0, -1]   # [N]
            r_som_last = aux["r_som_all"][0, -1]  # [N]
            r_pv_last = aux["r_pv_all"][0, -1]    # [1]
            r_l23_prev = r_l23_all[0, -2]         # [N]

            # W_rec from L2/3 (computed on the fly from its raw params).
            W_rec = net.l23.W_rec
            ff = F.linear(r_l4_last.unsqueeze(0), net.l23.W_l4_to_l23).squeeze(0)  # identity
            rec = F.linear(r_l23_prev.unsqueeze(0), W_rec).squeeze(0)

            # Inhibitory gains (InhibitoryGain acts as softplus(raw) * x, but we need
            # the raw gain values; call the module on scalar inputs).
            inh_som = net.l23.w_som(r_som_last.unsqueeze(0)).squeeze(0)   # [N]
            inh_pv = net.l23.w_pv_l23(r_pv_last.unsqueeze(0)).squeeze(0)  # [1]

            drive_raw = ff + rec - inh_som - inh_pv  # template_modulation is 0 in emergent mode
            rate_post = rectified_softplus(drive_raw)
        return drive_raw.detach().cpu().numpy(), rate_post.detach().cpu().numpy(), r_l23_all[0, -1].detach().cpu().numpy()

    # Feedback ON
    r_on, aux_on = _run(stim_theta, oracle_theta)
    drive_on, post_on, rate_on = _extract_drive(r_on, aux_on)

    # Feedback OFF
    with feedback_disabled(net):
        r_off, aux_off = _run(stim_theta, oracle_theta)
        drive_off, post_off, rate_off = _extract_drive(r_off, aux_off)

    dists = _dists_from(prefs, stim_theta, period).numpy()
    order = np.argsort(dists)
    dists_sorted = dists[order] + period / 2.0  # monotonic thetas for fwhm_from_curve

    fwhm_drive_on = fwhm_from_curve(dists_sorted, drive_on[order], period)
    fwhm_drive_off = fwhm_from_curve(dists_sorted, drive_off[order], period)
    fwhm_post_on = fwhm_from_curve(dists_sorted, rate_on[order], period)
    fwhm_post_off = fwhm_from_curve(dists_sorted, rate_off[order], period)

    return {
        "stim_theta": stim_theta,
        "oracle_theta": oracle_theta,
        "dists_signed": dists[order],
        "drive_on": drive_on[order],
        "drive_off": drive_off[order],
        "rate_on": rate_on[order],
        "rate_off": rate_off[order],
        "fwhm_drive_on": fwhm_drive_on,
        "fwhm_drive_off": fwhm_drive_off,
        "fwhm_rate_on": fwhm_post_on,
        "fwhm_rate_off": fwhm_post_off,
        "peak_drive_on": float(drive_on.max()),
        "peak_drive_off": float(drive_off.max()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_curve(xs: np.ndarray, ys: np.ndarray, n: int = 8) -> str:
    k = min(n, xs.shape[0])
    stride = max(1, xs.shape[0] // k)
    sel = list(range(0, xs.shape[0], stride))[:k]
    return "  ".join(f"{xs[i]:5.1f}°→{ys[i]:6.3f}" for i in sel)


def print_report(label: str, results: dict) -> None:
    m1 = results["metric1"]
    m1b = results["metric1b"]
    m2 = results["metric2"]
    m3 = results["metric3"]
    m4 = results["metric4"]
    m5 = results["metric5"]
    sanity = results["sanity"]
    artifact = results["artifact"]

    print(f"\n{'=' * 72}")
    print(f"REPRESENTATION ANALYSIS — {label}")
    print(f"{'=' * 72}")

    # --- Sanity: ablation zeroes the operator ----------------------------
    print("\n[Sanity] feedback_disabled() actually zeroes the operator output")
    print(f"  ON  drive max-abs: {sanity['drive_on_max_abs']:.6e}  (sum={sanity['drive_on_sum']:.4f})")
    print(f"  OFF drive max-abs: {sanity['drive_off_max_abs']:.6e}  (sum={sanity['drive_off_sum']:.4f})")
    print(f"  Ablation produces zero drive: {sanity['ablation_zero']}")

    # --- Metric 1: Population bump FWHM ----------------------------------
    print("\n[Metric 1] Population-bump FWHM (read all 36 L2/3 channels at a single stim)")
    for key, r in m1.items():
        print(f"  {key}: FWHM OFF={r['fwhm_off']:6.2f}°, ON={r['fwhm_on']:6.2f}°, "
              f"delta={r['delta_fwhm']:+6.2f}° ({r['pct_fwhm']:+6.1f}%)   "
              f"peak OFF={r['peak_off']:.4f} ON={r['peak_on']:.4f}")
    first_key = next(iter(m1))
    r0 = m1[first_key]
    print(f"  Full profile for '{first_key}' (dist_from_stim → r_off, r_on):")
    ds = r0['dists']
    # Print the inner ±60° window so we can see shape changes
    mask = np.abs(ds) <= 60.0
    pairs = [(float(ds[i]), float(r0['r_off'][i]), float(r0['r_on'][i]))
             for i in range(len(ds)) if mask[i]]
    for d, ro, rn in pairs:
        print(f"    d={d:+6.1f}°  OFF={ro:.4f}  ON={rn:.4f}")

    # --- Metric 1b: Flanking-channel tuning curves ------------------------
    print("\n[Metric 1b] Tuning curves at channels at various offsets from oracle (oracle=90°)")
    print(f"  {'offset(deg)':>12s} {'ch':>4s} {'pref(deg)':>10s}   "
          f"{'peak OFF':>10s} {'peak ON':>10s} {'peak ratio':>11s}   "
          f"{'FWHM OFF':>10s} {'FWHM ON':>10s} {'delta_FWHM':>11s}")
    for offset in sorted(m1b.keys()):
        r = m1b[offset]
        print(f"  {offset:+12.1f} {r['channel']:4d} {r['pref']:10.1f}   "
              f"{r['peak_off']:10.4f} {r['peak_on']:10.4f} {r['peak_ratio']:11.4f}   "
              f"{r['fwhm_off']:10.2f} {r['fwhm_on']:10.2f} {r['delta_fwhm']:+11.2f}")

    # --- Metric 2: Peak gain ---------------------------------------------
    print("\n[Metric 2] Preferred-channel peak gain (averaged over all 36 channels)")
    print(f"  Peak response OFF: mean = {m2['peak_off_mean']:.4f}")
    print(f"  Peak response ON : mean = {m2['peak_on_mean']:.4f}")
    print(f"  Ratio (ON / OFF):  {m2['ratio_on_over_off']:.4f}")

    # --- Metric 3: Channel profile (stim=oracle=90°) ---------------------
    print(f"\n[Metric 3] Channel-by-distance profile at stim={m3['stim_theta']}°, oracle={m3['stim_theta']}°")
    d = m3['dists']; ron = m3['r_on']; roff = m3['r_off']
    ron_n = m3['r_on_norm']; roff_n = m3['r_off_norm']
    print(f"  r_off peak: {roff.max():.4f} at dist={float(d[int(np.argmax(roff))])}°")
    print(f"  r_on  peak: {ron.max():.4f}  at dist={float(d[int(np.argmax(ron))])}°")
    print(f"  dist(deg):  {[float(round(x, 1)) for x in d[np.abs(d)<=45].tolist()]}")
    print(f"  r_off:      {[round(float(x), 4) for x in roff[np.abs(d)<=45].tolist()]}")
    print(f"  r_on:       {[round(float(x), 4) for x in ron[np.abs(d)<=45].tolist()]}")

    # --- Metric 4: Fine discrimination -----------------------------------
    print("\n[Metric 4] Fine-discrimination with READOUT noise")
    for title, m in [("NARROW subset (±15° around base_theta)", m4),
                     ("WIDE subset (±30° around base_theta)", results["metric4_wide"])]:
        subset = list(m.values())[0]['channels']
        print(f"  {title}: channels {subset} (n={len(subset)})")
        print(f"    {'noise':>8s} {'delta':>7s}  {'acc OFF':>10s} {'acc ON':>10s} {'delta_acc':>11s}")
        for (noise_std, delta) in sorted(m.keys()):
            r = m[(noise_std, delta)]
            print(f"    {noise_std:8.3f} {delta:7.1f}°  {r['acc_off']:10.3f} {r['acc_on']:10.3f} "
                  f"{r['acc_on']-r['acc_off']:+11.3f}")

    # --- Metric 5: Energy split ------------------------------------------
    print(f"\n[Metric 5] L2/3 energy split by distance from oracle (oracle=90°)")
    print(f"  Channels — expected (|d|≤10°): {m5['n_expected_channels']},"
          f"  surround (10<|d|≤45°): {m5['n_surround_channels']},"
          f"  far (|d|>45°): {m5['n_far_channels']}")
    print(f"  Total     : OFF={m5['energy_off_total']:10.4f}  ON={m5['energy_on_total']:10.4f}  "
          f"reduction={m5['reduction_total_pct']:+6.1f}%")
    print(f"  Expected  : OFF={m5['energy_off_expected']:10.4f}  ON={m5['energy_on_expected']:10.4f}  "
          f"reduction={m5['reduction_expected_pct']:+6.1f}%")
    print(f"  Surround  : OFF={m5['energy_off_surround']:10.4f}  ON={m5['energy_on_surround']:10.4f}  "
          f"reduction={m5['reduction_surround_pct']:+6.1f}%")
    print(f"  Far       : OFF={m5['energy_off_far']:10.4f}  ON={m5['energy_on_far']:10.4f}  "
          f"reduction={m5['reduction_far_pct']:+6.1f}%")

    # --- Artifact: pre-rectification drive FWHM --------------------------
    print(f"\n[Artifact] Pre-rectification L2/3 drive at stim={artifact['stim_theta']}°, "
          f"oracle={artifact['oracle_theta']}°")
    print(f"  FWHM of pre-rect DRIVE : OFF={artifact['fwhm_drive_off']:.2f}°  ON={artifact['fwhm_drive_on']:.2f}°"
          f"  delta={artifact['fwhm_drive_on']-artifact['fwhm_drive_off']:+.2f}°")
    print(f"  FWHM of post-rect RATE : OFF={artifact['fwhm_rate_off']:.2f}°  ON={artifact['fwhm_rate_on']:.2f}°"
          f"  delta={artifact['fwhm_rate_on']-artifact['fwhm_rate_off']:+.2f}°")
    print(f"  Peak DRIVE: OFF={artifact['peak_drive_off']:.4f}  ON={artifact['peak_drive_on']:.4f}"
          f"  ratio={artifact['peak_drive_on']/(artifact['peak_drive_off']+1e-12):.4f}")
    print(f"  Pre-rect drive (±45° window, every channel):")
    ds = artifact['dists_signed']
    mask = np.abs(ds) <= 45.0
    for i in np.where(mask)[0]:
        print(f"    d={ds[i]:+6.1f}°  drive OFF={artifact['drive_off'][i]:+.4f}  "
              f"drive ON={artifact['drive_on'][i]:+.4f}   "
              f"rate OFF={artifact['rate_off'][i]:.4f}  rate ON={artifact['rate_on'][i]:.4f}")


def print_comparison_table(results_by_label: dict[str, dict]) -> None:
    labels = list(results_by_label.keys())
    if not labels:
        return

    def row(header: str, fn):
        cells = "  |  ".join(f"{fn(results_by_label[l]):>10}" for l in labels)
        print(f"  {header:<36} | {cells}")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    header_labels = "  |  ".join(f"{l:>10}" for l in labels)
    print(f"  {'Metric':<36} | {header_labels}")
    print(f"  {'-' * 36}-+-{'-+-'.join(['-' * 10 for _ in labels])}")

    # Metric 2: peak gain
    row("Peak gain ratio (ON/OFF)",
        lambda r: f"{r['metric2']['ratio_on_over_off']:.4f}")

    # Metric 1: population bump FWHM (stim=ora=90°)
    key = "stim=90,ora=90"
    row("PopBump FWHM OFF (stim=ora=90)",
        lambda r: f"{r['metric1'][key]['fwhm_off']:.2f}")
    row("PopBump FWHM ON  (stim=ora=90)",
        lambda r: f"{r['metric1'][key]['fwhm_on']:.2f}")
    row("PopBump FWHM delta (stim=ora=90)",
        lambda r: f"{r['metric1'][key]['delta_fwhm']:+.2f}")

    # Metric 1b: flanking channel peak suppression
    for offset in [0.0, 20.0, 25.0, 30.0]:
        row(f"Flank ch offset={offset:+.0f}° peak ratio",
            lambda r, o=offset: f"{r['metric1b'][o]['peak_ratio']:.4f}")

    # Metric 5: energy split
    row("Energy reduction total (%)",
        lambda r: f"{r['metric5']['reduction_total_pct']:+.1f}")
    row("Energy reduction expected (%)",
        lambda r: f"{r['metric5']['reduction_expected_pct']:+.1f}")
    row("Energy reduction surround (%)",
        lambda r: f"{r['metric5']['reduction_surround_pct']:+.1f}")
    row("Energy reduction far (%)",
        lambda r: f"{r['metric5']['reduction_far_pct']:+.1f}")

    # Metric 4: discrimination at a representative noise level + delta
    # Use the highest noise for the most revealing contrast. Report both narrow and wide subsets.
    first_r = next(iter(results_by_label.values()))
    all_keys = sorted(first_r['metric4'].keys())
    if all_keys:
        noise_std = max(k[0] for k in all_keys)
        deltas = sorted({k[1] for k in all_keys if k[0] == noise_std})
        for delta in deltas:
            k = (noise_std, delta)
            row(f"δ={delta:.0f}° n={noise_std:.2f} narrow ON-OFF",
                lambda r, key=k: f"{r['metric4'][key]['acc_on']-r['metric4'][key]['acc_off']:+.3f}")
            row(f"δ={delta:.0f}° n={noise_std:.2f} wide   ON-OFF",
                lambda r, key=k: f"{r['metric4_wide'][key]['acc_on']-r['metric4_wide'][key]['acc_off']:+.3f}")

    # Artifact check: pre-rect drive FWHM
    row("Pre-rect DRIVE FWHM delta",
        lambda r: f"{r['artifact']['fwhm_drive_on']-r['artifact']['fwhm_drive_off']:+.2f}")
    row("Post-rect RATE FWHM delta",
        lambda r: f"{r['artifact']['fwhm_rate_on']-r['artifact']['fwhm_rate_off']:+.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_one(
    checkpoint: str,
    config: str,
    device: torch.device,
) -> dict:
    net, model_cfg, train_cfg = load_model(checkpoint, config, device)

    sanity = sanity_check_ablation(net, device)
    # Metric 1: population bump at matched and mismatched stim/oracle
    m1 = metric1_population_bump(
        net, device,
        conditions=[(90.0, 90.0), (120.0, 90.0), (90.0, 120.0)],
    )
    # Metric 1b: tuning curves at channels at various offsets from oracle
    m1b = metric1b_flanking_tuning(
        net, device, oracle_theta=90.0,
        flanking_offsets=[-30.0, -25.0, -20.0, -10.0, 0.0, 10.0, 20.0, 25.0, 30.0],
    )
    m2 = metric2_peak_gain(net, device)
    m3 = metric3_channel_profile(net, device, stim_theta=90.0)
    # Metric 4: readout noise with noise levels scanned.
    # Run twice: narrow subset (central ±15°, captures peak-gain effects like
    # dampening) and wide subset (±30°, captures flank effects like P4 sharpening).
    N_ = net.cfg.n_orientations
    step_ = net.cfg.orientation_range / N_
    prefs_ = torch.arange(N_, dtype=torch.float32) * step_
    base_theta_ = 90.0
    d_ = _dists_from(prefs_, base_theta_, net.cfg.orientation_range).abs().numpy()
    narrow_subset = [int(i) for i in np.where(d_ <= 15.0)[0]]
    wide_subset = [int(i) for i in np.where(d_ <= 30.0)[0]]
    m4 = metric4_fine_discrimination(
        net, device,
        deltas=[3.0, 5.0, 10.0],
        n_trials_per_class=500,
        readout_noise_stds=[0.01, 0.03, 0.10, 0.30],
        base_theta=base_theta_,
        subset_channels=narrow_subset,
    )
    m4_wide = metric4_fine_discrimination(
        net, device,
        deltas=[3.0, 5.0, 10.0],
        n_trials_per_class=500,
        readout_noise_stds=[0.01, 0.03, 0.10, 0.30],
        base_theta=base_theta_,
        subset_channels=wide_subset,
    )
    # Metric 5: energy split by distance from oracle
    m5 = metric5_energy_by_distance(net, device, oracle_theta=90.0)
    # Threshold artifact check at stim=ora=90°
    artifact = artifact_check_threshold(net, device, stim_theta=90.0, oracle_theta=90.0)

    return {
        "metric1": m1, "metric1b": m1b,
        "metric2": m2, "metric3": m3,
        "metric4": m4, "metric4_wide": m4_wide,
        "metric5": m5,
        "sanity": sanity, "artifact": artifact,
        "train_cfg": train_cfg,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Representation analysis: feedback ON vs OFF")
    p.add_argument("--checkpoint", type=str, action="append", required=True,
                   help="Checkpoint path. Pass multiple times to compare multiple models.")
    p.add_argument("--config", type=str, action="append", required=True,
                   help="Config path. Must match --checkpoint count and order.")
    p.add_argument("--label", type=str, action="append", default=None,
                   help="Optional label per checkpoint (defaults to checkpoint basename).")
    p.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) != len(args.config):
        raise SystemExit("--checkpoint and --config must appear the same number of times")
    labels = args.label or [Path(c).parent.name for c in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise SystemExit("--label count must match --checkpoint count")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    results_by_label: dict[str, dict] = {}
    for ckpt, cfg, label in zip(args.checkpoint, args.config, labels):
        logger.info(f"--- analyzing {label} ---")
        results = analyze_one(ckpt, cfg, device)
        print_report(label, results)
        results_by_label[label] = results

    print_comparison_table(results_by_label)


if __name__ == "__main__":
    main()
