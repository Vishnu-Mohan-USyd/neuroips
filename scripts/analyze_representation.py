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
        # Strip legacy keys from old checkpoints
        for legacy_key in ('mechanism', 'n_basis', 'max_apical_gain', 'tau_vip',
                           'simple_feedback', 'template_gain'):
            model_raw.pop(legacy_key, None)
        model_cfg = ModelConfig(**model_raw)
        train_fields = set(TrainingConfig.__dataclass_fields__.keys())
        train_cfg = TrainingConfig(**{k: v for k, v in train_raw.items() if k in train_fields})
        logger.info(f"Loaded config from checkpoint: "
                    f"feedback_mode={model_cfg.feedback_mode}, "
                    f"oracle_pi(train)={train_cfg.oracle_pi}")
    else:
        model_cfg, train_cfg, _ = load_config(config_path)
        logger.warning(f"No config in checkpoint; using {config_path}")

    net = LaminarV1V2Network(model_cfg)
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
    """Temporarily disable feedback by zeroing the feedback_scale buffer.

    In the current architecture, V2 feedback flows through a single
    head_feedback linear layer, gated by `feedback_scale`. Zeroing this
    buffer disables both the excitatory (center_exc) and inhibitory
    (SOM drive) feedback pathways.

    L4, PV, L2/3 recurrence, and adaptation all remain active — exactly
    what we want for an "ablate feedback, keep everything else" comparison.
    """
    saved_fb_scale = net.feedback_scale.detach().clone()

    try:
        with torch.no_grad():
            net.feedback_scale.zero_()
        yield
    finally:
        with torch.no_grad():
            net.feedback_scale.copy_(saved_fb_scale)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

T_STEPS = 25            # total timesteps (stimulus on whole time, no ISI)
READ_WINDOW = 5         # average L2/3 over the last READ_WINDOW steps
EVAL_PI = 5.0           # oracle precision used during analysis (matches HARDENING_RESULTS pi=5)
EVAL_CONTRAST = 1.0     # stimulus contrast used during analysis


def _make_bump(cfg: ModelConfig, thetas: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Create a Gaussian bump in orientation space.

    Args:
        cfg: Model config (for n_orientations, orientation_range, sigma_ff).
        thetas: [B] — center orientations (degrees).
        sigma: Width of the bump (degrees). Defaults to sigma_ff.

    Returns:
        bump: [B, N] — normalized Gaussian bumps.
    """
    N = cfg.n_orientations
    period = cfg.orientation_range
    step = period / N
    if sigma is None:
        sigma = cfg.sigma_ff
    pref = torch.arange(N, dtype=torch.float32, device=thetas.device) * step
    diff = (pref.unsqueeze(0) - thetas.unsqueeze(1) + period / 2) % period - period / 2
    bump = torch.exp(-diff ** 2 / (2 * sigma ** 2))
    return bump


def _build_oracle(net: LaminarV1V2Network, oracle_thetas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build oracle_q_pred and oracle_pi_pred tensors for a batch of trials.

    Args:
        net: the network (used for cfg).
        oracle_thetas: [B] — oracle target orientation (degrees) per trial.

    Returns:
        oracle_q: [B, T_STEPS, N] — bump at oracle_theta, held constant over time,
                  row-normalised to sum to 1.
        oracle_pi: [B, T_STEPS, 1] — constant EVAL_PI.
    """
    N = net.cfg.n_orientations
    B = oracle_thetas.shape[0]
    device = oracle_thetas.device
    q_single = _make_bump(net.cfg, oracle_thetas)  # [B, N]
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


def run_full_trajectory(
    net: LaminarV1V2Network,
    stim_thetas: torch.Tensor,
    oracle_thetas: torch.Tensor,
    device: torch.device,
    contrast: float = EVAL_CONTRAST,
) -> torch.Tensor:
    """Run the network and return the *full* L2/3 trajectory (all timesteps).

    Same as run_trials but without the windowed mean — returns [B, T_STEPS, N].
    Used by metric_time_resolved to analyze how L2/3 evolves step-by-step.
    No noise is applied (this helper is only used for the deterministic
    time-resolved metric).
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
    packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
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
    return r_l23_all  # [B, T_STEPS, N]


def run_full_trajectory_noisy(
    net: LaminarV1V2Network,
    stim_thetas: torch.Tensor,
    oracle_thetas: torch.Tensor,
    device: torch.device,
    contrast: float = EVAL_CONTRAST,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Like run_full_trajectory but with optional stimulus noise. Returns [B, T_STEPS, N]."""
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
    )
    stim_seq = stim_frame.unsqueeze(1).expand(B, T_STEPS, N).contiguous()

    if noise_std > 0:
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            noise = torch.randn(stim_seq.shape, device=device, generator=gen) * noise_std
        else:
            noise = torch.randn_like(stim_seq) * noise_std
        stim_seq = (stim_seq + noise).clamp(min=0.0)

    oracle_q, oracle_pi = _build_oracle(net, oracle_thetas.to(device))
    cue_seq = torch.zeros(B, T_STEPS, N, device=device)
    task_seq = torch.zeros(B, T_STEPS, 2, device=device)
    packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
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
    return r_l23_all  # [B, T_STEPS, N]


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
# Phase 1 sharpening-detection metrics (6-9)
#
# These metrics target the geometry of near-miss discrimination, which is
# where sharpening (if real) would show up. The original 5 metrics were
# blind to P4's flank-suppression effect because they measured global peak
# gain, bump FWHM, and readout-noise discrimination that saturated. These
# four are designed to distinguish real representational sharpening from
# dampening artefacts.
# ---------------------------------------------------------------------------


def _circ_mean_and_std_deg(thetas_deg: np.ndarray, period: float) -> tuple[float, float]:
    """Circular mean (degrees, in [0, period)) and circular std (degrees).

    Uses the standard circular-statistics definitions on a circle of
    circumference `period`:
      R = |<exp(i · 2π·θ/period)>|
      mean = angle(<·>) · period / (2π), wrapped to [0, period)
      std  = sqrt(-2 ln R) · period / (2π)
    """
    if thetas_deg.size == 0:
        return float("nan"), float("nan")
    angles = 2.0 * math.pi * thetas_deg / period
    c = float(np.mean(np.cos(angles)))
    s = float(np.mean(np.sin(angles)))
    R = math.sqrt(c * c + s * s)
    mean_ang = math.atan2(s, c) % (2.0 * math.pi)
    mean_deg = mean_ang * period / (2.0 * math.pi)
    if R < 1e-12:
        std_deg = period / 2.0
    else:
        std_deg = math.sqrt(-2.0 * math.log(R)) * period / (2.0 * math.pi)
    return mean_deg, std_deg


def _circ_signed_diff_deg(a_deg: float, b_deg: float, period: float) -> float:
    """Signed minimum circular difference (a - b) in (-period/2, period/2]."""
    return ((a_deg - b_deg + period / 2.0) % period) - period / 2.0


def _population_vector_decode(r: torch.Tensor, prefs: torch.Tensor, period: float) -> torch.Tensor:
    """Population-vector decode a batch of L2/3 responses to an orientation.

    Args:
        r: [B, N] — L2/3 activity per channel.
        prefs: [N] — preferred orientations (degrees) of each channel, on device.
        period: orientation period (degrees).

    Returns:
        theta: [B] — decoded orientation in [0, period).

    The decoder uses `exp(i · 2π·pref/period)` (period-scaled circular
    encoding). This is the standard circular population-vector decode for
    an orientation space of period `period`.
    """
    angles = 2.0 * math.pi * prefs / period  # [N]
    cos_p = torch.cos(angles)
    sin_p = torch.sin(angles)
    x = (r * cos_p).sum(dim=-1)  # [B]
    y = (r * sin_p).sum(dim=-1)
    ang = torch.atan2(y, x)  # [-pi, pi]
    theta = (ang * period / (2.0 * math.pi)) % period
    return theta


def metric_local_dprime(
    net: LaminarV1V2Network,
    device: torch.device,
    n_trials: int = 200,
    noise_std: float = 0.3,
    seed: int = 42,
) -> dict:
    """Metric 6: local d' of match vs near-miss, averaged across 8 anchors.

    For each anchor θ ∈ {0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5}°:
      - Set oracle = anchor.
      - For each δ ∈ {5°, 10°, 15°}:
        * Generate n_trials stim at anchor (match) and n_trials at anchor+δ
          (near-miss). Stimulus noise std=noise_std applied to the
          population-coded input.
        * Run the network, extract readout L2/3 [B, N].
        * Population-vector-decode each trial to a single orientation.
        * Compute circular mean and circular std (in degrees) of the
          decoded angles per condition.
        * d' = |circ_diff(μ_match, μ_miss)| / sqrt(½·(σ_match² + σ_miss²))
      - Repeat for feedback ON and OFF (same noise realization).
    Average d' across the 8 anchors per (δ, condition).

    Real representational sharpening should give delta_d > 0 at small δ
    (feedback improves local d'). Dampening should give delta_d ≤ 0
    (feedback kills signal → wider clusters or smaller separation).

    Args:
        net: loaded network.
        device: torch device.
        n_trials: trials per class (default 200).
        noise_std: Gaussian noise added to the population-coded stimulus
                   (same semantics as run_trials). Default 0.3.
        seed: base seed for reproducible noise realizations.

    Returns:
        dict with per-delta summary and per-anchor breakdowns.
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    prefs = torch.arange(N, dtype=torch.float32, device=device) * step

    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    deltas = [5.0, 10.0, 15.0]

    per_anchor: list[dict] = []
    agg: dict[float, dict[str, list[float]]] = {d: {"on": [], "off": []} for d in deltas}

    for ai, anchor in enumerate(anchors):
        anchor_rec: dict = {"anchor": anchor}
        for delta in deltas:
            thetas_match = torch.full((n_trials,), anchor, device=device)
            thetas_miss = torch.full(
                (n_trials,), (anchor + delta) % period, device=device
            )
            stim_all = torch.cat([thetas_match, thetas_miss], dim=0)
            oracle_all = torch.full_like(stim_all, anchor)

            # Same seed for on and off → same stimulus noise realization, so
            # the only difference between conditions is the feedback pathway.
            trial_seed = seed + ai * 1000 + int(delta * 10)
            r_on = run_trials(
                net, stim_all, oracle_all, device,
                noise_std=noise_std, seed=trial_seed,
            )
            with feedback_disabled(net):
                r_off = run_trials(
                    net, stim_all, oracle_all, device,
                    noise_std=noise_std, seed=trial_seed,
                )

            theta_on = _population_vector_decode(r_on, prefs, period).cpu().numpy()
            theta_off = _population_vector_decode(r_off, prefs, period).cpu().numpy()

            theta_on_m = theta_on[:n_trials]
            theta_on_n = theta_on[n_trials:]
            theta_off_m = theta_off[:n_trials]
            theta_off_n = theta_off[n_trials:]

            mu_on_m, sig_on_m = _circ_mean_and_std_deg(theta_on_m, period)
            mu_on_n, sig_on_n = _circ_mean_and_std_deg(theta_on_n, period)
            mu_off_m, sig_off_m = _circ_mean_and_std_deg(theta_off_m, period)
            mu_off_n, sig_off_n = _circ_mean_and_std_deg(theta_off_n, period)

            sep_on = abs(_circ_signed_diff_deg(mu_on_m, mu_on_n, period))
            sep_off = abs(_circ_signed_diff_deg(mu_off_m, mu_off_n, period))
            pooled_on = math.sqrt(0.5 * (sig_on_m ** 2 + sig_on_n ** 2) + 1e-12)
            pooled_off = math.sqrt(0.5 * (sig_off_m ** 2 + sig_off_n ** 2) + 1e-12)
            dp_on = sep_on / pooled_on
            dp_off = sep_off / pooled_off

            anchor_rec[f"delta_{int(delta)}"] = {
                "on": dp_on,
                "off": dp_off,
                "delta_d": dp_on - dp_off,
                "mu_match_off": mu_off_m,
                "mu_miss_off": mu_off_n,
                "sig_match_off": sig_off_m,
                "sig_miss_off": sig_off_n,
                "sep_off": sep_off,
                "pooled_off": pooled_off,
                "mu_match_on": mu_on_m,
                "mu_miss_on": mu_on_n,
                "sig_match_on": sig_on_m,
                "sig_miss_on": sig_on_n,
                "sep_on": sep_on,
                "pooled_on": pooled_on,
            }
            agg[delta]["on"].append(dp_on)
            agg[delta]["off"].append(dp_off)
        per_anchor.append(anchor_rec)

    summary: dict = {}
    for delta in deltas:
        m_on = float(np.mean(agg[delta]["on"]))
        m_off = float(np.mean(agg[delta]["off"]))
        summary[f"delta_{int(delta)}"] = {
            "on": m_on,
            "off": m_off,
            "delta_d": m_on - m_off,
        }
    summary["per_anchor"] = per_anchor
    summary["n_trials_per_class"] = n_trials
    summary["noise_std"] = noise_std
    return summary


def metric_match_vs_near_miss_decoding(
    net: LaminarV1V2Network,
    device: torch.device,
    n_train: int = 800,
    n_test: int = 200,
    noise_std: float = 0.3,
    readout_noise_std: float = 0.3,
    seed: int = 42,
    oracle_theta: float = 90.0,
) -> dict:
    """Metric 7: trained LogReg decoder of match vs near-miss at small deltas.

    For each δ ∈ {3°, 5°, 10°, 15°}, averaged across 8 anchor orientations
    {0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5}:
      - Generate n_train+n_test trials with labels: half at anchor
        (match), half at anchor+δ (near-miss).
      - Stimulus noise std=noise_std is applied to the input.
      - Extract L2/3 readout response, add readout noise std=readout_noise_std.
      - Train sklearn LogisticRegression on first n_train trials using all
        36 channels as features; test on remaining n_test.
      - Report test accuracy for feedback ON and OFF (identical noise),
        averaged across all 8 anchors.

    Differs from metric 4 in three ways: (a) it tests the specific
    match-vs-near-miss discrimination (not general orientation
    discrimination); (b) it applies noise on both input and output;
    (c) it uses all 36 channels rather than a narrow subset.

    Args:
        net: loaded network.
        device: torch device.
        n_train, n_test: sample counts (default 800/200).
        noise_std: stimulus input noise (default 0.3).
        readout_noise_std: output-side L2/3 noise (default 0.3).
        seed: reproducibility seed.
        oracle_theta: legacy param (ignored — now uses 8 anchors).

    Returns:
        dict keyed by 'delta_{int}' with {on, off, delta_acc, per_anchor}.
    """
    period = net.cfg.orientation_range
    deltas = [3.0, 5.0, 10.0, 15.0]
    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    n_total = n_train + n_test
    # Balanced classes — must be even
    if n_total % 2 != 0:
        raise ValueError(f"n_train+n_test must be even, got {n_total}")
    n_half = n_total // 2

    results: dict = {}
    for delta in deltas:
        anchor_accs_on = []
        anchor_accs_off = []
        per_anchor = {}
        for anchor in anchors:
            thetas_match = torch.full((n_half,), anchor, device=device)
            thetas_miss = torch.full(
                (n_half,), (anchor + delta) % period, device=device
            )
            stim = torch.cat([thetas_match, thetas_miss], dim=0)
            oracle_arr = torch.full_like(stim, anchor)
            labels = np.concatenate([
                np.zeros(n_half, dtype=np.int64),
                np.ones(n_half, dtype=np.int64),
            ])

            trial_seed = seed + int(delta * 10) + int(anchor * 100)
            r_on_clean = run_trials(
                net, stim, oracle_arr, device, noise_std=noise_std, seed=trial_seed,
            ).cpu().numpy()
            with feedback_disabled(net):
                r_off_clean = run_trials(
                    net, stim, oracle_arr, device, noise_std=noise_std, seed=trial_seed,
                ).cpu().numpy()

            rs = np.random.RandomState(trial_seed + 7)
            readout_noise = (rs.randn(*r_on_clean.shape).astype(np.float32)
                             * readout_noise_std)
            X_on = r_on_clean + readout_noise
            X_off = r_off_clean + readout_noise

            perm = rs.permutation(n_total)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

            def _fit_and_score(X, _train=train_idx, _test=test_idx, _labels=labels):
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X[_train], _labels[_train])
                return float(clf.score(X[_test], _labels[_test]))

            acc_on = _fit_and_score(X_on)
            acc_off = _fit_and_score(X_off)

            # --- Bootstrap 95% CI on delta_acc ---
            n_bootstrap = 100
            boot_deltas = []
            for b in range(n_bootstrap):
                boot_rs = np.random.RandomState(trial_seed + 1000 + b)
                boot_perm = boot_rs.permutation(n_total)
                b_train = boot_perm[:n_train]
                b_test = boot_perm[n_train:]
                b_acc_on = _fit_and_score(X_on, _train=b_train, _test=b_test,
                                          _labels=labels)
                b_acc_off = _fit_and_score(X_off, _train=b_train, _test=b_test,
                                           _labels=labels)
                boot_deltas.append(b_acc_on - b_acc_off)
            boot_deltas = np.array(boot_deltas)
            ci_low = float(np.percentile(boot_deltas, 2.5))
            ci_high = float(np.percentile(boot_deltas, 97.5))

            # --- Permutation test on delta_acc ---
            observed_delta = acc_on - acc_off
            n_permutations = 500
            perm_deltas = []
            # Stack on/off responses; shuffle condition labels
            X_stacked = np.concatenate([X_on, X_off], axis=0)  # [2*n_total, C]
            labels_stacked = np.concatenate([labels, labels], axis=0)
            for p in range(n_permutations):
                perm_rs = np.random.RandomState(trial_seed + 2000 + p)
                # Shuffle which half is "on" vs "off"
                shuf_idx = perm_rs.permutation(2 * n_total)
                X_shuf_on = X_stacked[shuf_idx[:n_total]]
                X_shuf_off = X_stacked[shuf_idx[n_total:]]
                p_acc_on = _fit_and_score(X_shuf_on)
                p_acc_off = _fit_and_score(X_shuf_off)
                perm_deltas.append(p_acc_on - p_acc_off)
            perm_deltas = np.array(perm_deltas)
            p_value = float((perm_deltas >= observed_delta).mean())

            anchor_accs_on.append(acc_on)
            anchor_accs_off.append(acc_off)
            per_anchor[anchor] = {
                "on": acc_on, "off": acc_off,
                "ci_low": ci_low, "ci_high": ci_high,
                "p_value": p_value,
            }

        mean_on = float(np.mean(anchor_accs_on))
        mean_off = float(np.mean(anchor_accs_off))
        # Average bootstrap CIs and p-values across anchors
        anchor_ci_lows = [per_anchor[a]["ci_low"] for a in anchors]
        anchor_ci_highs = [per_anchor[a]["ci_high"] for a in anchors]
        anchor_p_values = [per_anchor[a]["p_value"] for a in anchors]
        results[f"delta_{int(delta)}"] = {
            "on": mean_on,
            "off": mean_off,
            "delta_acc": mean_on - mean_off,
            "ci_low": float(np.mean(anchor_ci_lows)),
            "ci_high": float(np.mean(anchor_ci_highs)),
            "p_value": float(np.mean(anchor_p_values)),
            "per_anchor": per_anchor,
        }
    results["n_train"] = n_train
    results["n_test"] = n_test
    results["noise_std"] = noise_std
    results["readout_noise_std"] = readout_noise_std
    results["anchors"] = anchors
    return results


def metric_time_resolved(
    net: LaminarV1V2Network,
    device: torch.device,
    oracle_theta: float = 90.0,
    stim_theta: float = 90.0,
) -> dict:
    """Metric 8: per-timestep L2/3 gain, bump FWHM, and flank (±30°) response.

    Runs a single deterministic trial with stim=oracle=`stim_theta` and
    records L2/3 at every one of the T_STEPS timesteps. Reports:

      - peak gain (max response across the N channels)
      - bump FWHM (across the 36 channels as a function of distance from
        stim_theta)
      - response at the ±30° flank channels

    Returns full time series plus early (steps 0..3) and late (last 3)
    summaries. The hypothesis is that an early "sharpening-like"
    transient may exist that gets overwritten by dampening as the circuit
    settles — if so, it should appear in the early summary but not the
    late one.

    Args:
        net: loaded network.
        device: torch device.
        oracle_theta: expected orientation (default 90°).
        stim_theta: presented orientation (default 90°, matches oracle).

    Returns:
        dict with timesteps list and per-step and early/late summaries.
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    prefs = torch.arange(N, dtype=torch.float32) * step

    stim_arr = torch.tensor([stim_theta], device=device)
    oracle_arr = torch.tensor([oracle_theta], device=device)

    r_on_all = run_full_trajectory(net, stim_arr, oracle_arr, device)[0]  # [T, N]
    with feedback_disabled(net):
        r_off_all = run_full_trajectory(net, stim_arr, oracle_arr, device)[0]

    # Flank channels (±30° from stim_theta)
    flank_p30 = _closest_channel(prefs, (stim_theta + 30.0) % period, period)
    flank_m30 = _closest_channel(prefs, (stim_theta - 30.0) % period, period)

    # Sorted distance axis for FWHM calculation
    dists = _dists_from(prefs, stim_theta, period).numpy()
    order = np.argsort(dists)
    dists_axis = dists[order] + period / 2.0  # monotonic thetas for fwhm_from_curve

    T = int(r_on_all.shape[0])
    peak_on, peak_off = [], []
    fwhm_on, fwhm_off = [], []
    fp30_on, fp30_off = [], []
    fm30_on, fm30_off = [], []
    for t in range(T):
        ro = r_on_all[t].cpu().numpy()
        re = r_off_all[t].cpu().numpy()
        peak_on.append(float(ro.max()))
        peak_off.append(float(re.max()))
        fwhm_on.append(fwhm_from_curve(dists_axis, ro[order], period))
        fwhm_off.append(fwhm_from_curve(dists_axis, re[order], period))
        fp30_on.append(float(ro[flank_p30]))
        fp30_off.append(float(re[flank_p30]))
        fm30_on.append(float(ro[flank_m30]))
        fm30_off.append(float(re[flank_m30]))

    def _avg(lst: list[float], slc: slice) -> float:
        vals = [v for v in lst[slc] if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    # "Early" = steps 0..3 inclusive (4 steps after the stimulus onset but
    # before full settling); "late" = last 3 steps.
    early = slice(0, 4)
    late = slice(-3, None)

    return {
        "timesteps": list(range(T)),
        "peak_gain_on": peak_on, "peak_gain_off": peak_off,
        "fwhm_on": fwhm_on, "fwhm_off": fwhm_off,
        "flank_p30_on": fp30_on, "flank_p30_off": fp30_off,
        "flank_m30_on": fm30_on, "flank_m30_off": fm30_off,
        "early_peak_on": _avg(peak_on, early),
        "early_peak_off": _avg(peak_off, early),
        "late_peak_on": _avg(peak_on, late),
        "late_peak_off": _avg(peak_off, late),
        "early_fwhm_on": _avg(fwhm_on, early),
        "early_fwhm_off": _avg(fwhm_off, early),
        "late_fwhm_on": _avg(fwhm_on, late),
        "late_fwhm_off": _avg(fwhm_off, late),
        "early_flank_p30_on": _avg(fp30_on, early),
        "early_flank_p30_off": _avg(fp30_off, early),
        "late_flank_p30_on": _avg(fp30_on, late),
        "late_flank_p30_off": _avg(fp30_off, late),
        "early_flank_m30_on": _avg(fm30_on, early),
        "early_flank_m30_off": _avg(fm30_off, early),
        "late_flank_m30_on": _avg(fm30_on, late),
        "late_flank_m30_off": _avg(fm30_off, late),
        "flank_p30_channel": flank_p30,
        "flank_m30_channel": flank_m30,
        "stim_theta": stim_theta,
        "oracle_theta": oracle_theta,
    }


def metric_energy_by_relative_distance_normalized(
    net: LaminarV1V2Network,
    device: torch.device,
    oracle_theta: float = 90.0,
    n_stim: int = 36,
) -> dict:
    """Metric 9: per-channel relative energy reduction, averaged within bins.

    Reuses the distance-bin structure of metric5 (expected |d|≤10°,
    surround 10<|d|≤45°, far |d|>45°) but computes the PER-CHANNEL
    relative reduction (E_off[c] − E_on[c]) / |E_off[c]| first, then
    averages across channels in each bin. This gives equal weight to each
    channel and is not dominated by the high-energy channels near the
    oracle peak.

    Metric5 can mask a strong flank effect because the expected bin
    carries most of the energy; metric9 normalizes that away.

    Args:
        net: loaded network.
        device: torch device.
        oracle_theta: expected orientation (default 90°).
        n_stim: number of stimulus orientations to sweep (default 36).

    Returns:
        dict with per-bin channel counts and relative reductions, plus
        the raw per-channel energy arrays for debugging.
    """
    N = net.cfg.n_orientations
    period = net.cfg.orientation_range
    step = period / N
    orient_step = period / n_stim
    stim_grid = (torch.arange(n_stim, dtype=torch.float32) * orient_step).to(device)
    oracles = torch.full((n_stim,), oracle_theta, device=device)
    prefs = torch.arange(N, dtype=torch.float32) * step

    r_on = run_trials(net, stim_grid, oracles, device, noise_std=0.0).cpu().numpy()
    with feedback_disabled(net):
        r_off = run_trials(net, stim_grid, oracles, device, noise_std=0.0).cpu().numpy()

    # Sum each channel's activity across stimuli → [N]
    e_on_ch = r_on.sum(axis=0)
    e_off_ch = r_off.sum(axis=0)

    d = _dists_from(prefs, oracle_theta, period).abs().numpy()
    expected_mask = d <= 10.0
    surround_mask = (d > 10.0) & (d <= 45.0)
    far_mask = d > 45.0

    def _bin_rel(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        rel = (e_off_ch[mask] - e_on_ch[mask]) / (np.abs(e_off_ch[mask]) + 1e-12)
        return float(np.mean(rel))

    return {
        "oracle_theta": oracle_theta,
        "n_expected_channels": int(expected_mask.sum()),
        "n_surround_channels": int(surround_mask.sum()),
        "n_far_channels": int(far_mask.sum()),
        "expected_rel_reduction": _bin_rel(expected_mask),
        "surround_rel_reduction": _bin_rel(surround_mask),
        "far_rel_reduction": _bin_rel(far_mask),
        "e_on_ch": e_on_ch,
        "e_off_ch": e_off_ch,
    }


def metric_global_amplitude(
    net: LaminarV1V2Network,
    device: torch.device,
    oracle_theta: float = 90.0,
) -> dict:
    """Metric 10: global amplitude ratio (feedback ON / OFF), averaged across 8 anchors.

    For each anchor θ ∈ {0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5}°:
      - Present stim=anchor, oracle=anchor (matched condition).
      - Read all 36 L2/3 channels, feedback ON and OFF.
      - Compute mean_on, mean_off (mean across 36 channels), sum_on, sum_off.
    Average all quantities across the 8 anchors. Report ratios:
      mean_ratio = mean_on / mean_off  (<1 = Kok-like reduced, >1 = Huang-like boosted)
      sum_ratio  = sum_on / sum_off

    Args:
        net: trained network.
        device: torch device.
        oracle_theta: ignored (uses 8 anchors for averaging).

    Returns:
        dict with per-anchor breakdowns and averaged ratios.
    """
    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    per_anchor: list[dict] = []
    all_mean_on, all_mean_off = [], []
    all_sum_on, all_sum_off = [], []

    for anchor in anchors:
        stim = torch.tensor([anchor], device=device)
        oracle = torch.tensor([anchor], device=device)

        r_on = run_trials(net, stim, oracle, device, noise_std=0.0)[0].cpu().numpy()  # [N]
        with feedback_disabled(net):
            r_off = run_trials(net, stim, oracle, device, noise_std=0.0)[0].cpu().numpy()

        mean_on = float(r_on.mean())
        mean_off = float(r_off.mean())
        sum_on = float(r_on.sum())
        sum_off = float(r_off.sum())

        per_anchor.append({
            "anchor": anchor,
            "mean_on": mean_on,
            "mean_off": mean_off,
            "sum_on": sum_on,
            "sum_off": sum_off,
        })
        all_mean_on.append(mean_on)
        all_mean_off.append(mean_off)
        all_sum_on.append(sum_on)
        all_sum_off.append(sum_off)

    avg_mean_on = float(np.mean(all_mean_on))
    avg_mean_off = float(np.mean(all_mean_off))
    avg_sum_on = float(np.mean(all_sum_on))
    avg_sum_off = float(np.mean(all_sum_off))

    return {
        "mean_on": avg_mean_on,
        "mean_off": avg_mean_off,
        "mean_ratio": avg_mean_on / (avg_mean_off + 1e-12),
        "sum_on": avg_sum_on,
        "sum_off": avg_sum_off,
        "sum_ratio": avg_sum_on / (avg_sum_off + 1e-12),
        "per_anchor": per_anchor,
    }


# ---------------------------------------------------------------------------
# Phase 2 metrics: stronger effect-size measures (11-14)
#
# These metrics measure sharpening in the regime where priors matter most:
# high noise (below ceiling), fixed downstream readouts, and the temporal
# structure matching training.
# ---------------------------------------------------------------------------


def metric_threshold_benefit(
    net: LaminarV1V2Network,
    device: torch.device,
    deltas: list[float] | None = None,
    target_acc: float = 0.70,
    seed: int = 42,
) -> dict:
    """Metric 11: find noise threshold where OFF ≈ target_acc, measure ON benefit.

    Sweeps stimulus noise to find the noise level where the no-feedback network
    is at ~70% accuracy (threshold). At that noise, feedback benefit is largest
    because the input is genuinely ambiguous. Also computes the threshold_shift:
    how much more noise ON can tolerate before reaching the same target accuracy.

    Args:
        net: trained network.
        device: torch device.
        deltas: orientation differences to test (default [10, 15] degrees).
        target_acc: target OFF accuracy for threshold (default 0.70).
        seed: base seed for reproducibility.

    Returns:
        dict keyed by 'delta_{int}' with threshold noise, accuracies, and shift.
    """
    if deltas is None:
        deltas = [10.0, 15.0]
    period = net.cfg.orientation_range
    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    noise_levels = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    n_per_class = 500
    n_total = n_per_class * 2
    n_train = 800
    n_test = 200

    def _find_threshold(accs: list[float], noises: list[float], target: float) -> float:
        """Interpolate to find noise where acc crosses target (descending)."""
        for i in range(len(noises) - 1):
            if (accs[i] >= target >= accs[i + 1]) or (accs[i] <= target <= accs[i + 1]):
                frac = (target - accs[i]) / (accs[i + 1] - accs[i] + 1e-12)
                return noises[i] + frac * (noises[i + 1] - noises[i])
        # Fallback: closest measured point
        return noises[int(np.argmin(np.abs(np.array(accs) - target)))]

    results: dict = {}
    for delta in deltas:
        anchor_results = []
        for ai, anchor in enumerate(anchors):
            on_accs: list[float] = []
            off_accs: list[float] = []
            for ni, noise_std in enumerate(noise_levels):
                trial_seed = seed + ai * 1000 + int(delta * 10) + ni * 100
                thetas_match = torch.full((n_per_class,), anchor, device=device)
                thetas_miss = torch.full(
                    (n_per_class,), (anchor + delta) % period, device=device
                )
                stim = torch.cat([thetas_match, thetas_miss], dim=0)
                oracle = torch.full_like(stim, anchor)
                labels = np.concatenate([
                    np.zeros(n_per_class, dtype=np.int64),
                    np.ones(n_per_class, dtype=np.int64),
                ])

                r_on = run_trials(
                    net, stim, oracle, device, noise_std=noise_std, seed=trial_seed,
                ).cpu().numpy()
                with feedback_disabled(net):
                    r_off = run_trials(
                        net, stim, oracle, device, noise_std=noise_std, seed=trial_seed,
                    ).cpu().numpy()

                rs = np.random.RandomState(trial_seed + 7)
                perm = rs.permutation(n_total)
                train_idx, test_idx = perm[:n_train], perm[n_train:]

                clf_on = LogisticRegression(max_iter=2000)
                clf_on.fit(r_on[train_idx], labels[train_idx])
                acc_on = float(clf_on.score(r_on[test_idx], labels[test_idx]))

                clf_off = LogisticRegression(max_iter=2000)
                clf_off.fit(r_off[train_idx], labels[train_idx])
                acc_off = float(clf_off.score(r_off[test_idx], labels[test_idx]))

                on_accs.append(acc_on)
                off_accs.append(acc_off)

            threshold_off = _find_threshold(off_accs, noise_levels, target_acc)
            threshold_on = _find_threshold(on_accs, noise_levels, target_acc)
            threshold_shift = threshold_on - threshold_off

            # delta_acc at the OFF threshold (use nearest measured noise)
            idx_nearest = int(np.argmin(np.abs(np.array(noise_levels) - threshold_off)))
            delta_acc_at_threshold = on_accs[idx_nearest] - off_accs[idx_nearest]

            anchor_results.append({
                "anchor": anchor,
                "threshold_noise_off": threshold_off,
                "threshold_noise_on": threshold_on,
                "threshold_shift": threshold_shift,
                "delta_acc_at_threshold": delta_acc_at_threshold,
                "on_accs": on_accs,
                "off_accs": off_accs,
            })

        avg_threshold_off = float(np.mean([a["threshold_noise_off"] for a in anchor_results]))
        avg_threshold_on = float(np.mean([a["threshold_noise_on"] for a in anchor_results]))
        avg_shift = float(np.mean([a["threshold_shift"] for a in anchor_results]))
        avg_delta = float(np.mean([a["delta_acc_at_threshold"] for a in anchor_results]))

        results[f"delta_{int(delta)}"] = {
            "threshold_noise_off": avg_threshold_off,
            "threshold_noise_on": avg_threshold_on,
            "threshold_shift": avg_shift,
            "delta_acc_at_threshold": avg_delta,
            "per_anchor": anchor_results,
        }

    results["target_acc"] = target_acc
    results["noise_levels"] = noise_levels
    return results


def metric_fixed_readout(
    net: LaminarV1V2Network,
    device: torch.device,
    deltas: list[float] | None = None,
    noise_std: float = 0.5,
    readout_noise_std: float = 0.3,
    seed: int = 42,
) -> dict:
    """Metric 12: fixed downstream readout — train on neutral, test on valid/invalid.

    Trains a LogReg decoder ONCE on neutral (feedback OFF, noisy) responses,
    then freezes it and tests on:
      - valid: feedback ON, true oracle (expectation matches stimulus class)
      - invalid: feedback ON, wrong oracle (opposite class)

    This is the biologically relevant measure: does the expectation help a
    fixed downstream reader that was calibrated without feedback?

    Args:
        net: trained network.
        device: torch device.
        deltas: orientation differences (default [10, 15] degrees).
        noise_std: stimulus noise (default 0.5).
        seed: reproducibility seed.

    Returns:
        dict keyed by 'delta_{int}' with acc_neutral, acc_valid, acc_invalid, benefit, cost.
    """
    if deltas is None:
        deltas = [10.0, 15.0]
    period = net.cfg.orientation_range
    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    n_per_class = 500
    n_total = n_per_class * 2
    n_train = 800
    n_test = 200

    results: dict = {}
    for delta in deltas:
        anchor_results = []
        for ai, anchor in enumerate(anchors):
            trial_seed = seed + ai * 1000 + int(delta * 10)

            thetas_match = torch.full((n_per_class,), anchor, device=device)
            thetas_miss = torch.full(
                (n_per_class,), (anchor + delta) % period, device=device
            )
            stim = torch.cat([thetas_match, thetas_miss], dim=0)
            labels = np.concatenate([
                np.zeros(n_per_class, dtype=np.int64),
                np.ones(n_per_class, dtype=np.int64),
            ])

            # 1. Neutral: feedback OFF
            oracle_neutral = torch.full_like(stim, anchor)
            with feedback_disabled(net):
                r_neutral = run_trials(
                    net, stim, oracle_neutral, device,
                    noise_std=noise_std, seed=trial_seed,
                ).cpu().numpy()

            # Add readout noise (same realization for all conditions for fair comparison)
            rs_readout = np.random.RandomState(trial_seed + 42)
            readout_noise = (rs_readout.randn(*r_neutral.shape).astype(np.float32)
                             * readout_noise_std)
            r_neutral = r_neutral + readout_noise

            # Train decoder on neutral, freeze it
            rs = np.random.RandomState(trial_seed + 7)
            perm = rs.permutation(n_total)
            train_idx, test_idx = perm[:n_train], perm[n_train:]

            clf = LogisticRegression(max_iter=2000)
            clf.fit(r_neutral[train_idx], labels[train_idx])
            acc_neutral = float(clf.score(r_neutral[test_idx], labels[test_idx]))

            # 2. Valid: feedback ON, true oracle at anchor
            oracle_valid = torch.full_like(stim, anchor)
            r_valid = run_trials(
                net, stim, oracle_valid, device,
                noise_std=noise_std, seed=trial_seed + 100,
            ).cpu().numpy()
            r_valid = r_valid + readout_noise  # same noise for fair comparison
            acc_valid = float(clf.score(r_valid[test_idx], labels[test_idx]))

            # 3. Invalid: feedback ON, wrong oracle (opposite class)
            oracle_invalid = torch.full_like(stim, (anchor + delta) % period)
            r_invalid = run_trials(
                net, stim, oracle_invalid, device,
                noise_std=noise_std, seed=trial_seed + 200,
            ).cpu().numpy()
            r_invalid = r_invalid + readout_noise  # same noise
            acc_invalid = float(clf.score(r_invalid[test_idx], labels[test_idx]))

            anchor_results.append({
                "anchor": anchor,
                "acc_neutral": acc_neutral,
                "acc_valid": acc_valid,
                "acc_invalid": acc_invalid,
                "benefit": acc_valid - acc_neutral,
                "cost": acc_neutral - acc_invalid,
            })

        avg_neutral = float(np.mean([a["acc_neutral"] for a in anchor_results]))
        avg_valid = float(np.mean([a["acc_valid"] for a in anchor_results]))
        avg_invalid = float(np.mean([a["acc_invalid"] for a in anchor_results]))

        results[f"delta_{int(delta)}"] = {
            "acc_neutral": avg_neutral,
            "acc_valid": avg_valid,
            "acc_invalid": avg_invalid,
            "benefit": avg_valid - avg_neutral,
            "cost": avg_neutral - avg_invalid,
            "per_anchor": anchor_results,
        }

    results["noise_std"] = noise_std
    results["readout_noise_std"] = readout_noise_std
    return results


def metric_time_resolved_decoding(
    net: LaminarV1V2Network,
    device: torch.device,
    deltas: list[float] | None = None,
    noise_std: float = 0.5,
    readout_noise_std: float = 0.3,
    seed: int = 42,
) -> dict:
    """Metric 13: time-resolved decoding — at which timestep is sharpening strongest?

    For each timestep t, trains a LogReg decoder on neutral (OFF) L2/3 at time t,
    tests on valid (ON) L2/3 at time t. Reports per-timestep delta_acc and
    early/mid/late summaries.

    Hypothesis: sharpening benefit peaks early (before SOM normalization settles).

    Args:
        net: trained network.
        device: torch device.
        deltas: orientation differences (default [10, 15] degrees).
        noise_std: stimulus noise (default 0.5).
        seed: reproducibility seed.

    Returns:
        dict keyed by 'delta_{int}' with per-timestep curves and summaries.
    """
    if deltas is None:
        deltas = [10.0, 15.0]
    period = net.cfg.orientation_range
    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    n_per_class = 500
    n_total = n_per_class * 2
    n_train = 800
    n_test = 200

    results: dict = {}
    for delta in deltas:
        t_delta_accs: list[list[float]] = [[] for _ in range(T_STEPS)]
        anchor_results = []

        for ai, anchor in enumerate(anchors):
            trial_seed = seed + ai * 1000 + int(delta * 10)

            thetas_match = torch.full((n_per_class,), anchor, device=device)
            thetas_miss = torch.full(
                (n_per_class,), (anchor + delta) % period, device=device
            )
            stim = torch.cat([thetas_match, thetas_miss], dim=0)
            oracle = torch.full_like(stim, anchor)
            labels = np.concatenate([
                np.zeros(n_per_class, dtype=np.int64),
                np.ones(n_per_class, dtype=np.int64),
            ])

            # Full trajectory with noise — ON and OFF
            traj_on = run_full_trajectory_noisy(
                net, stim, oracle, device, noise_std=noise_std, seed=trial_seed,
            ).cpu().numpy()  # [B, T, N]
            with feedback_disabled(net):
                traj_off = run_full_trajectory_noisy(
                    net, stim, oracle, device, noise_std=noise_std, seed=trial_seed,
                ).cpu().numpy()

            # Add readout noise (same realization for ON and OFF)
            if readout_noise_std > 0:
                rs_readout = np.random.RandomState(trial_seed + 42)
                r_noise = (rs_readout.randn(*traj_off.shape).astype(np.float32)
                           * readout_noise_std)
                traj_off = traj_off + r_noise
                traj_on = traj_on + r_noise

            rs = np.random.RandomState(trial_seed + 7)
            perm = rs.permutation(n_total)
            train_idx, test_idx = perm[:n_train], perm[n_train:]

            t_deltas: list[float] = []
            for t in range(T_STEPS):
                X_off_t = traj_off[:, t, :]
                X_on_t = traj_on[:, t, :]

                clf = LogisticRegression(max_iter=2000)
                clf.fit(X_off_t[train_idx], labels[train_idx])
                acc_off_t = float(clf.score(X_off_t[test_idx], labels[test_idx]))
                acc_on_t = float(clf.score(X_on_t[test_idx], labels[test_idx]))
                delta_acc_t = acc_on_t - acc_off_t
                t_deltas.append(delta_acc_t)
                t_delta_accs[t].append(delta_acc_t)

            anchor_results.append({
                "anchor": anchor,
                "delta_acc_per_t": t_deltas,
            })

        avg_per_t = [float(np.mean(t_delta_accs[t])) for t in range(T_STEPS)]
        early_avg = float(np.mean(avg_per_t[:8]))     # steps 0-7
        mid_avg = float(np.mean(avg_per_t[8:18]))     # steps 8-17
        late_avg = float(np.mean(avg_per_t[18:]))      # steps 18-24
        peak_t = int(np.argmax(avg_per_t))
        peak_delta = float(avg_per_t[peak_t])

        results[f"delta_{int(delta)}"] = {
            "delta_acc_per_t": avg_per_t,
            "early_avg": early_avg,
            "mid_avg": mid_avg,
            "late_avg": late_avg,
            "peak_t": peak_t,
            "peak_delta": peak_delta,
            "per_anchor": anchor_results,
        }

    results["noise_std"] = noise_std
    results["readout_noise_std"] = readout_noise_std
    return results


def metric_temporal_evaluation(
    net: LaminarV1V2Network,
    device: torch.device,
    deltas: list[float] | None = None,
    noise_std: float = 0.3,
    readout_noise_std: float = 0.3,
    seed: int = 42,
) -> dict:
    """Metric 14: cue→ISI→stimulus temporal structure matching training.

    Evaluates with the temporal pattern the model was trained on:
      - ISI (t=0,1): cue (population bump at oracle theta), no stimulus
      - ON (t=2-7): stimulus (with noise), no cue
      - Readout: average L2/3 at last 3 ON steps (t=5,6,7)
    Compares feedback ON (with cue+oracle) vs OFF (no cue, no oracle).
    Train decoder on OFF, test on ON.

    Args:
        net: trained network.
        device: torch device.
        deltas: orientation differences (default [10, 15] degrees).
        noise_std: stimulus noise (default 0.3).
        seed: reproducibility seed.

    Returns:
        dict keyed by 'delta_{int}' with acc_off, acc_on, delta_acc.
    """
    if deltas is None:
        deltas = [10.0, 15.0]
    period = net.cfg.orientation_range
    N = net.cfg.n_orientations
    anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    steps_isi = 2
    steps_on = 6
    T = steps_isi + steps_on  # 8
    read_last = 3  # last 3 ON steps
    n_per_class = 500
    n_total = n_per_class * 2
    n_train = 800
    n_test = 200

    results: dict = {}
    for delta in deltas:
        anchor_results = []
        for ai, anchor in enumerate(anchors):
            trial_seed = seed + ai * 1000 + int(delta * 10)
            B = n_total

            thetas_match = torch.full((n_per_class,), anchor, device=device)
            thetas_miss = torch.full(
                (n_per_class,), (anchor + delta) % period, device=device
            )
            stim_all = torch.cat([thetas_match, thetas_miss], dim=0)
            labels = np.concatenate([
                np.zeros(n_per_class, dtype=np.int64),
                np.ones(n_per_class, dtype=np.int64),
            ])

            # Build stimulus grating [B, N]
            stim_frame = generate_grating(
                stim_all.to(device),
                torch.full((B,), EVAL_CONTRAST, device=device),
                n_orientations=N,
                sigma=net.cfg.sigma_ff,
                n=net.cfg.naka_rushton_n,
                c50=net.cfg.naka_rushton_c50,
                period=period,
            )

            # Build cue bump [B, N]
            oracle_arr = torch.full((B,), anchor, device=device)
            cue_bump = net._make_bump(oracle_arr)
            cue_bump = cue_bump / (cue_bump.sum(dim=-1, keepdim=True) + 1e-8)

            # Noisy stimulus for ON period
            stim_on_block = stim_frame.unsqueeze(1).expand(B, steps_on, N).contiguous().clone()
            if noise_std > 0:
                gen = torch.Generator(device=device)
                gen.manual_seed(trial_seed)
                noise = torch.randn(stim_on_block.shape, device=device, generator=gen) * noise_std
                stim_on_block = (stim_on_block + noise).clamp(min=0.0)

            task_seq = torch.zeros(B, T, 2, device=device)

            # --- ON condition: cue during ISI, stim during ON, oracle active ---
            stim_seq_on = torch.zeros(B, T, N, device=device)
            stim_seq_on[:, steps_isi:, :] = stim_on_block
            cue_seq_on = torch.zeros(B, T, N, device=device)
            cue_seq_on[:, :steps_isi, :] = cue_bump.unsqueeze(1).expand(B, steps_isi, N)

            oracle_q = cue_bump.unsqueeze(1).expand(B, T, N).contiguous()
            oracle_pi = torch.full((B, T, 1), EVAL_PI, device=device)

            packed_on = net.pack_inputs(stim_seq_on, cue_seq_on, task_seq)
            net.oracle_mode = True
            net.oracle_q_pred = oracle_q
            net.oracle_pi_pred = oracle_pi
            try:
                with torch.no_grad():
                    r_on_all, _, _ = net(packed_on)
            finally:
                net.oracle_mode = False
                net.oracle_q_pred = None
                net.oracle_pi_pred = None
            r_on = r_on_all[:, -read_last:].mean(dim=1).cpu().numpy()  # [B, N]

            # --- OFF condition: no cue, no oracle, feedback disabled ---
            stim_seq_off = torch.zeros(B, T, N, device=device)
            stim_seq_off[:, steps_isi:, :] = stim_on_block  # same noise
            cue_seq_off = torch.zeros(B, T, N, device=device)  # no cue

            packed_off = net.pack_inputs(stim_seq_off, cue_seq_off, task_seq)
            with feedback_disabled(net):
                net.oracle_mode = False
                with torch.no_grad():
                    r_off_all, _, _ = net(packed_off)
            r_off = r_off_all[:, -read_last:].mean(dim=1).cpu().numpy()  # [B, N]

            # Add readout noise (same realization for ON and OFF)
            if readout_noise_std > 0:
                rs_readout = np.random.RandomState(trial_seed + 42)
                r_noise = (rs_readout.randn(*r_off.shape).astype(np.float32)
                           * readout_noise_std)
                r_off = r_off + r_noise
                r_on = r_on + r_noise

            # Train decoder on OFF, test on both
            rs = np.random.RandomState(trial_seed + 7)
            perm = rs.permutation(n_total)
            train_idx, test_idx = perm[:n_train], perm[n_train:]

            clf = LogisticRegression(max_iter=2000)
            clf.fit(r_off[train_idx], labels[train_idx])
            acc_off = float(clf.score(r_off[test_idx], labels[test_idx]))
            acc_on = float(clf.score(r_on[test_idx], labels[test_idx]))

            anchor_results.append({
                "anchor": anchor,
                "acc_off": acc_off,
                "acc_on": acc_on,
                "delta_acc": acc_on - acc_off,
            })

        avg_off = float(np.mean([a["acc_off"] for a in anchor_results]))
        avg_on = float(np.mean([a["acc_on"] for a in anchor_results]))

        results[f"delta_{int(delta)}"] = {
            "acc_off": avg_off,
            "acc_on": avg_on,
            "delta_acc": avg_on - avg_off,
            "per_anchor": anchor_results,
        }

    results["noise_std"] = noise_std
    results["readout_noise_std"] = readout_noise_std
    results["steps_isi"] = steps_isi
    results["steps_on"] = steps_on
    return results


# ---------------------------------------------------------------------------
# Sanity and artifact checks
# ---------------------------------------------------------------------------

def _unpack_feedback(result):
    """Unpack feedback operator output, handling old and new APIs.

    Returns:
        (som_drive, vip_drive, apical_gain) — any missing element is None.
    """
    if isinstance(result, tuple):
        if len(result) == 3:
            return result[0], result[1], result[2]  # som, vip, apical
        if len(result) == 2:
            return result[0], result[1], None  # som, vip (pre-apical)
    return result, None, None  # som only (very old)


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
            som_on, vip_on, apical_on = _unpack_feedback(net.feedback(q, pi))
    finally:
        net.feedback.uncache_kernels()
    with feedback_disabled(net):
        net.feedback.cache_kernels()
        try:
            with torch.no_grad():
                som_off, vip_off, apical_off = _unpack_feedback(net.feedback(q, pi))
        finally:
            net.feedback.uncache_kernels()

    result = {
        "drive_on_max_abs": float(som_on.abs().max().item()),
        "drive_off_max_abs": float(som_off.abs().max().item()),
        "drive_on_sum": float(som_on.sum().item()),
        "drive_off_sum": float(som_off.sum().item()),
        "ablation_zero": bool(som_off.abs().max().item() < 1e-7),
    }
    if vip_on is not None:
        result["vip_on_max_abs"] = float(vip_on.abs().max().item())
        result["vip_off_max_abs"] = float(vip_off.abs().max().item())
        result["vip_ablation_zero"] = bool(vip_off.abs().max().item() < 1e-7)
    return result


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
    m6 = results["metric6"]
    m7 = results["metric7"]
    m8 = results["metric8"]
    m9 = results["metric9"]
    m10 = results.get("metric10")
    m11 = results.get("metric11")
    m12 = results.get("metric12")
    m13 = results.get("metric13")
    m14 = results.get("metric14")
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

    # --- Metric 6: Local d' (Phase 1 sharpening detection) ----------------
    print(f"\n[Metric 6] Local d' (match vs near-miss, population-vector decode)")
    print(f"  n_trials={m6['n_trials_per_class']} per class, "
          f"noise_std={m6['noise_std']}, 8 anchors averaged")
    dprime_off_h = "d' OFF"
    dprime_on_h = "d' ON"
    dprime_delta_h = "delta_d'"
    print(f"  {'delta':>8s}  {dprime_off_h:>10s}  {dprime_on_h:>10s}  {dprime_delta_h:>10s}")
    for key in ["delta_5", "delta_10", "delta_15"]:
        r = m6[key]
        print(f"  {key:>8s}  {r['off']:10.4f}  {r['on']:10.4f}  {r['delta_d']:+10.4f}")
    print(f"  Per-anchor breakdown (rows=anchors, cols=deltas 5/10/15°):")
    print(f"    {'anchor':>8s}  "
          f"{'d5 OFF':>10s} {'d5 ON':>10s}  "
          f"{'d10 OFF':>10s} {'d10 ON':>10s}  "
          f"{'d15 OFF':>10s} {'d15 ON':>10s}")
    for a in m6['per_anchor']:
        print(f"    {a['anchor']:8.1f}  "
              f"{a['delta_5']['off']:10.4f} {a['delta_5']['on']:10.4f}  "
              f"{a['delta_10']['off']:10.4f} {a['delta_10']['on']:10.4f}  "
              f"{a['delta_15']['off']:10.4f} {a['delta_15']['on']:10.4f}")
    # Show a sample of the raw decoded-angle statistics (for hand verification)
    _anchor_dbg = next(
        (a for a in m6['per_anchor'] if abs(a['anchor'] - 90.0) < 1e-6),
        m6['per_anchor'][0],
    )
    print(f"  Raw decoded stats @ anchor={_anchor_dbg['anchor']}° "
          f"delta=10° feedback OFF (for hand verification):")
    d10 = _anchor_dbg['delta_10']
    print(f"    mu_match_off={d10['mu_match_off']:.4f}°  "
          f"sig_match_off={d10['sig_match_off']:.4f}°")
    print(f"    mu_miss_off ={d10['mu_miss_off']:.4f}°  "
          f"sig_miss_off ={d10['sig_miss_off']:.4f}°")
    print(f"    sep_off    ={d10['sep_off']:.4f}°   "
          f"pooled_sigma_off={d10['pooled_off']:.4f}°   "
          f"d'_off={d10['off']:.4f}")

    # --- Metric 7: Match vs near-miss trained-decoder accuracy ------------
    print(f"\n[Metric 7] Match vs near-miss trained LogReg decoder (all 36 ch)")
    print(f"  n_train={m7['n_train']}, n_test={m7['n_test']}, "
          f"stim_noise={m7['noise_std']}, readout_noise={m7['readout_noise_std']}, "
          f"anchors={m7.get('anchors', 'single')}")
    has_ci = "ci_low" in m7.get("delta_3", {})
    if has_ci:
        print(f"  {'delta':>8s}  {'acc OFF':>10s}  {'acc ON':>10s}  {'delta_acc':>11s}  {'CI_low':>8s}  {'CI_high':>8s}  {'p_value':>8s}")
    else:
        print(f"  {'delta':>8s}  {'acc OFF':>10s}  {'acc ON':>10s}  {'delta_acc':>11s}")
    for key in ["delta_3", "delta_5", "delta_10", "delta_15"]:
        if key not in m7:
            continue
        r = m7[key]
        if has_ci:
            print(f"  {key:>8s}  {r['off']:10.4f}  {r['on']:10.4f}  {r['delta_acc']:+11.4f}"
                  f"  {r['ci_low']:+8.4f}  {r['ci_high']:+8.4f}  {r['p_value']:8.4f}")
        else:
            print(f"  {key:>8s}  {r['off']:10.4f}  {r['on']:10.4f}  {r['delta_acc']:+11.4f}")

    # --- Metric 8: Time-resolved L2/3 dynamics ----------------------------
    print(f"\n[Metric 8] Time-resolved L2/3 dynamics "
          f"(stim={m8['stim_theta']}°, oracle={m8['oracle_theta']}°, no noise)")
    print(f"  Per-timestep trajectory (T={len(m8['timesteps'])} steps):")
    print(f"    {'t':>3s}  {'peak OFF':>10s} {'peak ON':>10s}  "
          f"{'fwhm OFF':>10s} {'fwhm ON':>10s}  "
          f"{'fp30 OFF':>10s} {'fp30 ON':>10s}  "
          f"{'fm30 OFF':>10s} {'fm30 ON':>10s}")
    for t in m8['timesteps']:
        def _fmt(x):
            return "   nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:10.4f}"
        print(f"    {t:3d}  "
              f"{_fmt(m8['peak_gain_off'][t])} {_fmt(m8['peak_gain_on'][t])}  "
              f"{_fmt(m8['fwhm_off'][t])} {_fmt(m8['fwhm_on'][t])}  "
              f"{_fmt(m8['flank_p30_off'][t])} {_fmt(m8['flank_p30_on'][t])}  "
              f"{_fmt(m8['flank_m30_off'][t])} {_fmt(m8['flank_m30_on'][t])}")
    print(f"  Early (steps 0-3) vs Late (last 3):")
    print(f"    peak  : early OFF={m8['early_peak_off']:.4f} ON={m8['early_peak_on']:.4f}  "
          f"late OFF={m8['late_peak_off']:.4f} ON={m8['late_peak_on']:.4f}")
    print(f"    fwhm  : early OFF={m8['early_fwhm_off']:.2f}  ON={m8['early_fwhm_on']:.2f}  "
          f"late OFF={m8['late_fwhm_off']:.2f}  ON={m8['late_fwhm_on']:.2f}")
    print(f"    fp30  : early OFF={m8['early_flank_p30_off']:.4f} ON={m8['early_flank_p30_on']:.4f}  "
          f"late OFF={m8['late_flank_p30_off']:.4f} ON={m8['late_flank_p30_on']:.4f}")
    print(f"    fm30  : early OFF={m8['early_flank_m30_off']:.4f} ON={m8['early_flank_m30_on']:.4f}  "
          f"late OFF={m8['late_flank_m30_off']:.4f} ON={m8['late_flank_m30_on']:.4f}")

    # --- Metric 9: Per-channel normalized energy reduction ----------------
    print(f"\n[Metric 9] Per-channel relative energy reduction by distance bin")
    print(f"  Channels — expected: {m9['n_expected_channels']}, "
          f"surround: {m9['n_surround_channels']}, far: {m9['n_far_channels']}")
    print(f"  Expected (|d|≤10°)  rel_red = {m9['expected_rel_reduction']:+.4f}")
    print(f"  Surround (10<|d|≤45°) rel_red = {m9['surround_rel_reduction']:+.4f}")
    print(f"  Far      (|d|>45°)  rel_red = {m9['far_rel_reduction']:+.4f}")

    # --- Metric 10: Global amplitude ratio --------------------------------
    if m10 is not None:
        print(f"\n[Metric 10] Global amplitude ratio (feedback ON/OFF), averaged across 8 anchors")
        print(f"  Mean activity ON : {m10['mean_on']:.6f}")
        print(f"  Mean activity OFF: {m10['mean_off']:.6f}")
        print(f"  Mean ratio (ON/OFF): {m10['mean_ratio']:.4f}"
              f"  {'(reduced)' if m10['mean_ratio'] < 1.0 else '(boosted)'}")
        print(f"  Sum activity ON : {m10['sum_on']:.4f}")
        print(f"  Sum activity OFF: {m10['sum_off']:.4f}")
        print(f"  Sum ratio  (ON/OFF): {m10['sum_ratio']:.4f}")
        print(f"  Per-anchor breakdown:")
        print(f"    {'anchor':>8s}  {'mean ON':>10s}  {'mean OFF':>10s}  {'sum ON':>10s}  {'sum OFF':>10s}")
        for a in m10['per_anchor']:
            print(f"    {a['anchor']:8.1f}  {a['mean_on']:10.6f}  {a['mean_off']:10.6f}"
                  f"  {a['sum_on']:10.4f}  {a['sum_off']:10.4f}")

    # --- Metric 11: Threshold benefit ------------------------------------
    if m11 is not None:
        print(f"\n[Metric 11] Threshold benefit (noise where OFF ≈ {m11['target_acc']:.0%})")
        for dkey in sorted(k for k in m11 if k.startswith("delta_")):
            r = m11[dkey]
            print(f"  {dkey}: threshold_noise_off={r['threshold_noise_off']:.3f}, "
                  f"threshold_noise_on={r['threshold_noise_on']:.3f}")
            print(f"    delta_acc at threshold: {r['delta_acc_at_threshold']:+.4f}")
            print(f"    threshold_shift (ON-OFF): {r['threshold_shift']:+.3f}")
        print(f"  Noise sweep per anchor (delta_10):")
        if "delta_10" in m11:
            for a in m11["delta_10"]["per_anchor"]:
                print(f"    anchor={a['anchor']:5.1f}°  "
                      f"noise_off={a['threshold_noise_off']:.3f}  "
                      f"noise_on={a['threshold_noise_on']:.3f}  "
                      f"shift={a['threshold_shift']:+.3f}  "
                      f"Δacc={a['delta_acc_at_threshold']:+.4f}")

    # --- Metric 12: Fixed readout ----------------------------------------
    if m12 is not None:
        print(f"\n[Metric 12] Fixed readout (train on neutral, test on valid/invalid)")
        print(f"  stimulus_noise={m12['noise_std']}, readout_noise={m12.get('readout_noise_std', 0)}")
        for dkey in sorted(k for k in m12 if k.startswith("delta_")):
            r = m12[dkey]
            print(f"  {dkey}: neutral={r['acc_neutral']:.4f}  valid={r['acc_valid']:.4f}  "
                  f"invalid={r['acc_invalid']:.4f}")
            print(f"    benefit (valid-neutral): {r['benefit']:+.4f}   "
                  f"cost (neutral-invalid): {r['cost']:+.4f}")
        if "delta_10" in m12:
            print(f"  Per-anchor (delta_10):")
            for a in m12["delta_10"]["per_anchor"]:
                print(f"    anchor={a['anchor']:5.1f}°  "
                      f"neutral={a['acc_neutral']:.4f}  valid={a['acc_valid']:.4f}  "
                      f"invalid={a['acc_invalid']:.4f}  benefit={a['benefit']:+.4f}")

    # --- Metric 13: Time-resolved decoding --------------------------------
    if m13 is not None:
        print(f"\n[Metric 13] Time-resolved decoding (stim_noise={m13['noise_std']}, "
              f"readout_noise={m13.get('readout_noise_std', 0)})")
        for dkey in sorted(k for k in m13 if k.startswith("delta_")):
            r = m13[dkey]
            print(f"  {dkey}: peak_t={r['peak_t']}, peak_delta={r['peak_delta']:+.4f}")
            print(f"    early(0-7)={r['early_avg']:+.4f}  mid(8-17)={r['mid_avg']:+.4f}  "
                  f"late(18-24)={r['late_avg']:+.4f}")
            print(f"    per-timestep: ", end="")
            for t, d in enumerate(r["delta_acc_per_t"]):
                print(f"t{t}={d:+.3f}", end=" ")
            print()

    # --- Metric 14: Temporal evaluation -----------------------------------
    if m14 is not None:
        print(f"\n[Metric 14] Temporal evaluation (cue→ISI→stim, "
              f"ISI={m14['steps_isi']}, ON={m14['steps_on']}, "
              f"stim_noise={m14['noise_std']}, readout_noise={m14.get('readout_noise_std', 0)})")
        for dkey in sorted(k for k in m14 if k.startswith("delta_")):
            r = m14[dkey]
            print(f"  {dkey}: acc_off={r['acc_off']:.4f}  acc_on={r['acc_on']:.4f}  "
                  f"delta_acc={r['delta_acc']:+.4f}")
        if "delta_10" in m14:
            print(f"  Per-anchor (delta_10):")
            for a in m14["delta_10"]["per_anchor"]:
                print(f"    anchor={a['anchor']:5.1f}°  "
                      f"off={a['acc_off']:.4f}  on={a['acc_on']:.4f}  "
                      f"Δ={a['delta_acc']:+.4f}")

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

    # Metric 6: local d' — expect >0 delta_d for true sharpening, ≤0 for dampening
    for dkey, dlabel in [("delta_5", "δ=5"), ("delta_10", "δ=10"), ("delta_15", "δ=15")]:
        row(f"M6 local d' {dlabel}° OFF",
            lambda r, k=dkey: f"{r['metric6'][k]['off']:.4f}")
        row(f"M6 local d' {dlabel}° ON",
            lambda r, k=dkey: f"{r['metric6'][k]['on']:.4f}")
        row(f"M6 local d' {dlabel}° delta",
            lambda r, k=dkey: f"{r['metric6'][k]['delta_d']:+.4f}")

    # Metric 7: match-vs-near-miss LogReg accuracy (8-anchor averaged)
    first_m7 = next(iter(results_by_label.values()))['metric7']
    has_m7_ci = "ci_low" in first_m7.get("delta_3", {})
    for dkey, dlabel in [("delta_3", "δ=3"), ("delta_5", "δ=5"), ("delta_10", "δ=10"), ("delta_15", "δ=15")]:
        row(f"M7 acc {dlabel}° OFF",
            lambda r, k=dkey: f"{r['metric7'][k]['off']:.4f}")
        row(f"M7 acc {dlabel}° ON",
            lambda r, k=dkey: f"{r['metric7'][k]['on']:.4f}")
        row(f"M7 acc {dlabel}° delta",
            lambda r, k=dkey: f"{r['metric7'][k]['delta_acc']:+.4f}")
        if has_m7_ci:
            row(f"M7 {dlabel}° CI",
                lambda r, k=dkey: f"[{r['metric7'][k]['ci_low']:+.3f},{r['metric7'][k]['ci_high']:+.3f}]")
            row(f"M7 {dlabel}° p-val",
                lambda r, k=dkey: f"{r['metric7'][k]['p_value']:.4f}")

    # Metric 8: time-resolved early-vs-late summary
    row("M8 early peak OFF",
        lambda r: f"{r['metric8']['early_peak_off']:.4f}")
    row("M8 early peak ON",
        lambda r: f"{r['metric8']['early_peak_on']:.4f}")
    row("M8 late peak OFF",
        lambda r: f"{r['metric8']['late_peak_off']:.4f}")
    row("M8 late peak ON",
        lambda r: f"{r['metric8']['late_peak_on']:.4f}")
    row("M8 early fwhm OFF",
        lambda r: f"{r['metric8']['early_fwhm_off']:.2f}")
    row("M8 early fwhm ON",
        lambda r: f"{r['metric8']['early_fwhm_on']:.2f}")
    row("M8 late fwhm OFF",
        lambda r: f"{r['metric8']['late_fwhm_off']:.2f}")
    row("M8 late fwhm ON",
        lambda r: f"{r['metric8']['late_fwhm_on']:.2f}")
    row("M8 late fp30 ratio ON/OFF",
        lambda r: f"{r['metric8']['late_flank_p30_on']/(r['metric8']['late_flank_p30_off']+1e-12):.4f}")
    row("M8 late fm30 ratio ON/OFF",
        lambda r: f"{r['metric8']['late_flank_m30_on']/(r['metric8']['late_flank_m30_off']+1e-12):.4f}")

    # Metric 9: per-channel normalized energy reductions
    row("M9 expected rel_red",
        lambda r: f"{r['metric9']['expected_rel_reduction']:+.4f}")
    row("M9 surround rel_red",
        lambda r: f"{r['metric9']['surround_rel_reduction']:+.4f}")
    row("M9 far rel_red",
        lambda r: f"{r['metric9']['far_rel_reduction']:+.4f}")

    # Metric 10: global amplitude ratio
    if "metric10" in next(iter(results_by_label.values())):
        row("M10 global mean ratio (ON/OFF)",
            lambda r: f"{r['metric10']['mean_ratio']:.4f}")
        row("M10 global sum ratio (ON/OFF)",
            lambda r: f"{r['metric10']['sum_ratio']:.4f}")

    # Metric 11: threshold benefit
    first = next(iter(results_by_label.values()))
    if "metric11" in first:
        for dkey, dlabel in [("delta_10", "δ=10"), ("delta_15", "δ=15")]:
            row(f"M11 {dlabel}° threshold_noise_off",
                lambda r, k=dkey: f"{r['metric11'][k]['threshold_noise_off']:.3f}")
            row(f"M11 {dlabel}° Δacc@threshold",
                lambda r, k=dkey: f"{r['metric11'][k]['delta_acc_at_threshold']:+.4f}")
            row(f"M11 {dlabel}° threshold_shift",
                lambda r, k=dkey: f"{r['metric11'][k]['threshold_shift']:+.3f}")

    # Metric 12: fixed readout
    if "metric12" in first:
        for dkey, dlabel in [("delta_10", "δ=10"), ("delta_15", "δ=15")]:
            row(f"M12 {dlabel}° neutral",
                lambda r, k=dkey: f"{r['metric12'][k]['acc_neutral']:.4f}")
            row(f"M12 {dlabel}° valid",
                lambda r, k=dkey: f"{r['metric12'][k]['acc_valid']:.4f}")
            row(f"M12 {dlabel}° invalid",
                lambda r, k=dkey: f"{r['metric12'][k]['acc_invalid']:.4f}")
            row(f"M12 {dlabel}° benefit",
                lambda r, k=dkey: f"{r['metric12'][k]['benefit']:+.4f}")

    # Metric 13: time-resolved decoding
    if "metric13" in first:
        for dkey, dlabel in [("delta_10", "δ=10"), ("delta_15", "δ=15")]:
            row(f"M13 {dlabel}° peak_t",
                lambda r, k=dkey: f"{r['metric13'][k]['peak_t']}")
            row(f"M13 {dlabel}° peak_delta",
                lambda r, k=dkey: f"{r['metric13'][k]['peak_delta']:+.4f}")
            row(f"M13 {dlabel}° early(0-7)",
                lambda r, k=dkey: f"{r['metric13'][k]['early_avg']:+.4f}")
            row(f"M13 {dlabel}° late(18-24)",
                lambda r, k=dkey: f"{r['metric13'][k]['late_avg']:+.4f}")

    # Metric 14: temporal evaluation
    if "metric14" in first:
        for dkey, dlabel in [("delta_10", "δ=10"), ("delta_15", "δ=15")]:
            row(f"M14 {dlabel}° Δacc (temporal)",
                lambda r, k=dkey: f"{r['metric14'][k]['delta_acc']:+.4f}")


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
    # Phase 1 sharpening-detection metrics
    m6 = metric_local_dprime(net, device, n_trials=200, noise_std=0.3, seed=42)
    m7 = metric_match_vs_near_miss_decoding(
        net, device, n_train=800, n_test=200,
        noise_std=0.3, readout_noise_std=0.3, seed=42, oracle_theta=90.0,
    )
    m8 = metric_time_resolved(net, device, oracle_theta=90.0, stim_theta=90.0)
    m9 = metric_energy_by_relative_distance_normalized(
        net, device, oracle_theta=90.0,
    )
    # Threshold artifact check at stim=ora=90°
    artifact = artifact_check_threshold(net, device, stim_theta=90.0, oracle_theta=90.0)
    # Metric 10: global amplitude ratio
    m10 = metric_global_amplitude(net, device)
    # Phase 2 metrics (11-14): stronger effect-size measures
    m11 = metric_threshold_benefit(net, device, deltas=[10.0, 15.0], seed=42)
    m12 = metric_fixed_readout(net, device, deltas=[10.0, 15.0], noise_std=0.5, seed=42)
    m13 = metric_time_resolved_decoding(net, device, deltas=[10.0, 15.0], noise_std=0.5, seed=42)
    m14 = metric_temporal_evaluation(net, device, deltas=[10.0, 15.0], noise_std=0.3, seed=42)

    return {
        "metric1": m1, "metric1b": m1b,
        "metric2": m2, "metric3": m3,
        "metric4": m4, "metric4_wide": m4_wide,
        "metric5": m5,
        "metric6": m6, "metric7": m7, "metric8": m8, "metric9": m9,
        "metric10": m10,
        "metric11": m11, "metric12": m12, "metric13": m13, "metric14": m14,
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
