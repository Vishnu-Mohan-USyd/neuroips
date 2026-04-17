#!/usr/bin/env python3
"""Priming dose-response sweep (Task #38).

Empirical test of the priming hypothesis:

    If the elevated Expected peak in matched-march contexts is driven by
    L2/3 *priming* — residual activity from preceding presentations whose
    orientation overlaps with the upcoming probe — then **varying the
    distance between context and probe orientations** should produce a
    monotonic dose-response curve. Specifically:

      * Expected peak should DECREASE as offset grows from 0° → 90°.
      * Unexpected peak (probe at +90° from Expected) should be roughly
        FLAT, because the probe is always 90° from the context (no
        priming overlap regardless of offset).
      * Omission peak should also be roughly flat (no stim, just whatever
        feedback echo lives in the would-be probe channel).

Design
======
For each ``offset ∈ [0°, 5°, 10°, 15°, 20°, 30°, 45°, 60°, 90°]``:

  1. Sample ``n_trials`` probe orientations from the 36-channel grid
     (uniform with replacement). Use the SAME sample for all 3 passes
     so per-offset stats are paired.

  2. Build context = ``n_context = 10`` presentations of a HOLD at
     ``(probe_ori - offset) mod 180°`` (NOT a march — same orientation
     for all 10 presentations). This isolates "priming from a single
     reference orientation" as the controlled variable.

  3. Run THREE matched-context forward passes:
       Pass A (Expected):   probe at ``probe_ori`` (same contrast as ctx)
       Pass B (Unexpected): probe at ``(probe_ori + 90°) mod 180°``
       Pass C (Omission):   probe at zero contrast (only stim is zeroed,
                            cue and task_state are unchanged).

  4. Per pass, record at the probe ON readout window (steps 9..11):
       - Mean L2/3 ring per trial: ``r_l23 [B, N]``
       - Decoder top-1 channel: ``decoder(r_l23).argmax(-1)``
       - Per-trial scalars:
           peak  = r_l23[trial, target_ch_for_pass]
           total = r_l23[trial].sum()
           correct = (decoder_top1 == target_ch_for_pass)
       Targets:
           Pass A:  target_ch = probe_ori_channel
           Pass B:  target_ch = unexp_ori_channel  ((probe + 90°)/step)
           Pass C:  target_ch = probe_ori_channel  (would-be probe;
                                  decoder accuracy is "n/a interp")

  5. Aggregate per offset, per pass:
       mean(peak), bootstrap 95% CI
       mean(total), bootstrap 95% CI
       mean(correct), bootstrap 95% CI
       mean(pi_pred_eff)  — side quantity, NOT a filter here

Bootstrap CIs use 1000 resamples of the per-trial values, percentile
method (2.5, 97.5).

Output
======
    PNG: 1 row × 3 cols subplot — peak, total, decoder acc — vs offset,
         3 lines per subplot (E blue / U red / O green) with shaded 95%
         bootstrap CI bands.
    JSON: per-offset, per-pass scalar stats and per-trial arrays
          (kept compact; only means and CIs for the 3 metrics).

Notes (no silly mistakes — per Lead's brief)
============================================
1. Context = 10 IDENTICAL presentations at ``probe_ori - offset`` (a HOLD,
   not a march).
2. Probe is presentation index 10 (0-indexed). target_onset =
   ``n_context * (steps_on + steps_isi)``.
3. Unexpected probe = ``(probe_ori + 90°) mod 180°`` — ALWAYS +90° from
   Pass A's probe, NOT from the context.
4. Decoder accuracy:
     Pass A: vs ``round(probe_ori / step_deg) % N``
     Pass B: vs ``round((probe_ori + 90°) % 180 / step_deg) % N``
     Pass C: vs Pass A target (interpretation: "n/a")
5. Probe orientations sampled from the 36 discrete channels (0°..175°,
   step 5°).
6. Bootstrap: 1000 resamples, percentiles 2.5 / 97.5.

Run on the same R1+R2 checkpoint as Task #37.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

# repo root + scripts/ for imports
_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ModelConfig, TrainingConfig, load_config
from src.experiments.paradigm_base import TrialConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating

from matched_quality_sim import _load_decoder


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _filter_dataclass_kwargs(cls, raw: dict | None) -> dict:
    if raw is None:
        return {}
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}


def load_model_decoder(
    checkpoint_path: str, config_path: str | None, device: torch.device
) -> tuple[LaminarV1V2Network, ModelConfig, TrainingConfig, torch.nn.Module]:
    """Load the trained network + decoder.

    If ``config_path`` is provided, use it (this matches matched_probe_3pass.py
    behavior). Otherwise read model/training config from the checkpoint.
    """
    if config_path is not None:
        model_cfg, train_cfg, _ = load_config(config_path)
    else:
        ckpt_peek = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_raw = dict(ckpt_peek.get("config", {}).get("model", {}))
        for legacy_key in ("mechanism", "n_basis", "max_apical_gain", "tau_vip",
                           "simple_feedback", "template_gain"):
            model_raw.pop(legacy_key, None)
        train_raw = dict(ckpt_peek.get("config", {}).get("training", {}))
        model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_raw))
        train_cfg = TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, train_raw))
        del ckpt_peek

    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    # Same defensive resets as matched_probe_3pass.py.
    if hasattr(net, "oracle_mode"):
        net.oracle_mode = False
    if hasattr(net, "feedback_scale"):
        net.feedback_scale.fill_(1.0)

    decoder = _load_decoder(ckpt, model_cfg.n_orientations, device)
    return net, model_cfg, train_cfg, decoder


def make_trial_cfg(train_cfg: TrainingConfig, n_context: int = 10,
                   contrast: float = 0.8) -> TrialConfig:
    """Build a TrialConfig matching the priming-experiment template."""
    return TrialConfig(
        n_context=n_context,
        steps_on=train_cfg.steps_on,
        steps_isi=train_cfg.steps_isi,
        steps_post=12,
        contrast=contrast,
    )


# ---------------------------------------------------------------------------
# Batched trial construction
# ---------------------------------------------------------------------------

def build_batch_trial(
    model_cfg: ModelConfig,
    trial_cfg: TrialConfig,
    probe_oris_deg: torch.Tensor,    # [B] float32 cpu (deg)
    context_oris_deg: torch.Tensor,  # [B] float32 cpu (deg)  — same per ctx slot
    probe_contrast: float,
    context_contrast: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ``stim [B, T, N]`` and ``task_state [B, T, 2]`` for a HOLD-context
    trial batch.

    Each trial:
      - ``n_context`` identical context presentations at ``context_oris[i]``
        with ``context_contrast``.
      - One probe presentation at ``probe_oris[i]`` with ``probe_contrast``
        (skip if probe_contrast == 0 — leaves zeros in the probe ON window;
        cue and task_state are NOT zeroed by this function — they're set to
        "relevant task" throughout).
    """
    B = probe_oris_deg.shape[0]
    N = model_cfg.n_orientations
    sigma = model_cfg.sigma_ff
    nr_n = model_cfg.naka_rushton_n
    nr_c50 = model_cfg.naka_rushton_c50
    period = model_cfg.orientation_range

    steps_per_pres = trial_cfg.steps_on + trial_cfg.steps_isi
    total_steps = (
        trial_cfg.n_context * steps_per_pres
        + trial_cfg.steps_on
        + trial_cfg.steps_post
    )
    stim = torch.zeros(B, total_steps, N, device=device)

    # Context grating (one [B, N] tensor; broadcast across the 10 ON windows).
    ctx_grating = generate_grating(
        context_oris_deg.cpu(),
        torch.full((B,), float(context_contrast)),
        n_orientations=N, sigma=sigma, n=nr_n, c50=nr_c50, period=period,
    ).to(device)  # [B, N]
    for idx in range(trial_cfg.n_context):
        onset = idx * steps_per_pres
        stim[:, onset:onset + trial_cfg.steps_on, :] = ctx_grating.unsqueeze(1)

    # Probe (skip if contrast==0 — Pass C / Omission)
    if probe_contrast > 0:
        probe_grating = generate_grating(
            probe_oris_deg.cpu(),
            torch.full((B,), float(probe_contrast)),
            n_orientations=N, sigma=sigma, n=nr_n, c50=nr_c50, period=period,
        ).to(device)  # [B, N]
        probe_onset = trial_cfg.n_context * steps_per_pres
        stim[:, probe_onset:probe_onset + trial_cfg.steps_on, :] = (
            probe_grating.unsqueeze(1)
        )

    task_state = torch.zeros(B, total_steps, 2, device=device)
    task_state[:, :, 0] = 1.0  # relevant (focused) task

    return stim, task_state


def forward_pack(net: LaminarV1V2Network, stim: torch.Tensor,
                 task_state: torch.Tensor
                 ) -> tuple[torch.Tensor, dict]:
    """One forward pass on a batched trial. Returns (r_l23 [B, T, N], aux)."""
    packed = net.pack_inputs(stim, None, task_state)
    with torch.no_grad():
        r_l23_all, _, aux = net.forward(packed)
    return r_l23_all, aux


# ---------------------------------------------------------------------------
# Per-offset collection
# ---------------------------------------------------------------------------

def run_offset(
    net: LaminarV1V2Network,
    decoder: torch.nn.Module,
    model_cfg: ModelConfig,
    trial_cfg: TrialConfig,
    probe_chs: torch.Tensor,   # [B] int64 — probe orientation channel indices
    offset_deg: float,
    device: torch.device,
    readout_window: tuple[int, int] = (9, 11),
) -> dict[str, np.ndarray]:
    """Run all 3 passes for a single offset with the given probe-channel sample.

    Returns per-trial arrays for each pass:
      r_probe_{A,B,C}  [B, N]
      decoder_top1_{A,B,C}  [B]
      target_ch_{A,B,C}  [B]
      peak_{A,B,C}  [B]
      total_{A,B,C}  [B]
      correct_{A,B,C}  [B]
      pi_at_target  [B] — pi_pred_eff at probe_isi_pre step (Pass A)
      probe_ori, context_ori, unexp_ori  [B] (deg)
    """
    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    B = probe_chs.shape[0]
    steps_per_pres = trial_cfg.steps_on + trial_cfg.steps_isi
    probe_onset = trial_cfg.n_context * steps_per_pres
    w_start, w_end = readout_window  # inclusive
    pi_step = probe_onset - 1        # last ISI step before probe ON

    probe_ori = probe_chs.float() * step_deg                          # [B] cpu
    context_ori = (probe_ori - offset_deg) % period                   # [B] cpu
    unexp_ori = (probe_ori + period / 2.0) % period                   # [B] cpu

    target_ch_A = ((probe_ori / step_deg).round().long()) % N
    target_ch_B = ((unexp_ori / step_deg).round().long()) % N
    target_ch_C = target_ch_A.clone()

    # Pass A: probe at probe_ori (Expected)
    stim_A, ts_A = build_batch_trial(
        model_cfg, trial_cfg,
        probe_oris_deg=probe_ori,
        context_oris_deg=context_ori,
        probe_contrast=trial_cfg.contrast,
        context_contrast=trial_cfg.contrast,
        device=device,
    )
    r_A, aux_A = forward_pack(net, stim_A, ts_A)

    # Pass B: probe at unexp_ori = (probe + 90°) % 180  (Unexpected)
    stim_B, ts_B = build_batch_trial(
        model_cfg, trial_cfg,
        probe_oris_deg=unexp_ori,
        context_oris_deg=context_ori,
        probe_contrast=trial_cfg.contrast,
        context_contrast=trial_cfg.contrast,
        device=device,
    )
    r_B, _ = forward_pack(net, stim_B, ts_B)

    # Pass C: probe at zero contrast (Omission)
    stim_C, ts_C = build_batch_trial(
        model_cfg, trial_cfg,
        probe_oris_deg=probe_ori,           # ignored when contrast=0
        context_oris_deg=context_ori,
        probe_contrast=0.0,
        context_contrast=trial_cfg.contrast,
        device=device,
    )
    r_C, _ = forward_pack(net, stim_C, ts_C)

    # Diagnostic: pre-probe windows must be bit-identical across passes.
    # We sample two windows: the LAST context ON window and the LAST context
    # ISI pre-probe step. (P-2 / P-1 analogues to Task #37's diagnostic.)
    last_ctx_on_start = (trial_cfg.n_context - 1) * steps_per_pres
    last_ctx_on_end = last_ctx_on_start + trial_cfg.steps_on  # exclusive
    diag_ctx_AB = float((r_A[:, last_ctx_on_start:last_ctx_on_end, :]
                         - r_B[:, last_ctx_on_start:last_ctx_on_end, :]).abs().max())
    diag_ctx_AC = float((r_A[:, last_ctx_on_start:last_ctx_on_end, :]
                         - r_C[:, last_ctx_on_start:last_ctx_on_end, :]).abs().max())

    # Probe ON readout window means
    sl = slice(probe_onset + w_start, probe_onset + w_end + 1)
    r_probe_A = r_A[:, sl, :].mean(dim=1)   # [B, N]
    r_probe_B = r_B[:, sl, :].mean(dim=1)
    r_probe_C = r_C[:, sl, :].mean(dim=1)

    # Decoder top-1
    dec_A = decoder(r_probe_A).argmax(dim=-1)
    dec_B = decoder(r_probe_B).argmax(dim=-1)
    dec_C = decoder(r_probe_C).argmax(dim=-1)

    # pi at probe_isi_pre (only need from Pass A — context is held
    # identical, but it's the Pass A trajectory we report).
    pi_at = aux_A["pi_pred_eff_all"][:, pi_step, 0]  # [B]

    # Per-trial scalars
    rng_idx = torch.arange(B, device=device)
    peak_A = r_probe_A[rng_idx, target_ch_A.to(device)]
    peak_B = r_probe_B[rng_idx, target_ch_B.to(device)]
    peak_C = r_probe_C[rng_idx, target_ch_C.to(device)]
    total_A = r_probe_A.sum(dim=-1)
    total_B = r_probe_B.sum(dim=-1)
    total_C = r_probe_C.sum(dim=-1)
    correct_A = (dec_A.cpu() == target_ch_A).to(torch.float32)
    correct_B = (dec_B.cpu() == target_ch_B).to(torch.float32)
    correct_C = (dec_C.cpu() == target_ch_C).to(torch.float32)

    return {
        "probe_ori": probe_ori.numpy().astype(np.float32),
        "context_ori": context_ori.numpy().astype(np.float32),
        "unexp_ori": unexp_ori.numpy().astype(np.float32),
        "target_ch_A": target_ch_A.numpy().astype(np.int64),
        "target_ch_B": target_ch_B.numpy().astype(np.int64),
        "target_ch_C": target_ch_C.numpy().astype(np.int64),
        "r_probe_A": r_probe_A.cpu().numpy().astype(np.float32),
        "r_probe_B": r_probe_B.cpu().numpy().astype(np.float32),
        "r_probe_C": r_probe_C.cpu().numpy().astype(np.float32),
        "decoder_top1_A": dec_A.cpu().numpy().astype(np.int64),
        "decoder_top1_B": dec_B.cpu().numpy().astype(np.int64),
        "decoder_top1_C": dec_C.cpu().numpy().astype(np.int64),
        "peak_A": peak_A.cpu().numpy().astype(np.float32),
        "peak_B": peak_B.cpu().numpy().astype(np.float32),
        "peak_C": peak_C.cpu().numpy().astype(np.float32),
        "total_A": total_A.cpu().numpy().astype(np.float32),
        "total_B": total_B.cpu().numpy().astype(np.float32),
        "total_C": total_C.cpu().numpy().astype(np.float32),
        "correct_A": correct_A.numpy().astype(np.float32),
        "correct_B": correct_B.numpy().astype(np.float32),
        "correct_C": correct_C.numpy().astype(np.float32),
        "pi_at_target": pi_at.cpu().numpy().astype(np.float32),
        "_diag_ctx_AB": diag_ctx_AB,
        "_diag_ctx_AC": diag_ctx_AC,
    }


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: np.ndarray, n_resamples: int = 1000, seed: int = 0,
    pct_lo: float = 2.5, pct_hi: float = 97.5,
) -> tuple[float, float, float]:
    """Bootstrap mean and percentile CI for a 1-D array of per-trial values.

    Returns (mean, lo, hi). For empty arrays returns (nan, nan, nan).
    """
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, pct_lo))
    hi = float(np.percentile(means, pct_hi))
    return mean, lo, hi


def aggregate_offset(per_trial: dict[str, np.ndarray],
                     n_boot: int = 1000, seed: int = 0
                     ) -> dict[str, Any]:
    """Aggregate per-offset stats with bootstrap CIs for the 3 metrics.

    Returns a dict with shape:
        {pass_letter: {metric: {"mean": _, "lo": _, "hi": _}}}
    where pass_letter ∈ {A, B, C} and metric ∈ {peak, total, correct}.
    Also includes mean(pi_at_target).
    """
    out: dict[str, Any] = {}
    for pass_letter in ("A", "B", "C"):
        bucket: dict[str, dict[str, float]] = {}
        for metric_key, arr_key in [
            ("peak", f"peak_{pass_letter}"),
            ("total", f"total_{pass_letter}"),
            ("correct", f"correct_{pass_letter}"),
        ]:
            mean, lo, hi = bootstrap_mean_ci(
                per_trial[arr_key], n_resamples=n_boot, seed=seed,
            )
            bucket[metric_key] = {"mean": mean, "lo": lo, "hi": hi}
        out[pass_letter] = bucket
    out["mean_pi_at_target"] = float(per_trial["pi_at_target"].mean())
    out["n"] = int(per_trial["pi_at_target"].shape[0])
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PASS_LABELS = {"A": "Expected", "B": "Unexpected", "C": "Omission"}
PASS_COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}


def plot_dose_response(
    offsets_deg: list[float],
    aggs: dict[float, dict[str, Any]],
    fig_path: str,
    title: str,
) -> None:
    """1-row × 3-col figure: peak, total, decoder acc vs offset."""
    metrics = [
        ("peak",    "Peak L2/3 at target channel"),
        ("total",   "Total L2/3 (sum over channels)"),
        ("correct", "Decoder accuracy (top-1)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    x = np.array(offsets_deg, dtype=float)

    for ax_idx, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for pass_letter in ("A", "B", "C"):
            color = PASS_COLORS[pass_letter]
            label = PASS_LABELS[pass_letter]
            means = np.array([aggs[off][pass_letter][metric_key]["mean"]
                              for off in offsets_deg])
            los = np.array([aggs[off][pass_letter][metric_key]["lo"]
                            for off in offsets_deg])
            his = np.array([aggs[off][pass_letter][metric_key]["hi"]
                            for off in offsets_deg])
            ax.plot(x, means, "o-", color=color, label=label, linewidth=1.7,
                    markersize=5.5)
            ax.fill_between(x, los, his, color=color, alpha=0.18, linewidth=0)
        ax.set_xlabel("Context−probe offset (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0])
        ax.set_xticks(offsets_deg)
        ax.grid(True, alpha=0.30)
        ax.set_xlim(-3, max(offsets_deg) + 3)
        if metric_key == "correct":
            ax.set_ylim(-0.02, 1.02)
            ax.axhline(1.0 / 36.0, color="0.55", linestyle=":", linewidth=1.0,
                       label="chance")
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.95)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_OFFSETS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None,
                   help="Optional config YAML. If omitted, model+training "
                        "config is read from the checkpoint dict.")
    p.add_argument("--output-fig", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=200,
                   help="Per-offset trial count.")
    p.add_argument("--n-context", type=int, default=10,
                   help="Number of HOLD context presentations before the probe.")
    p.add_argument("--contrast", type=float, default=0.8,
                   help="Stim contrast (used for both context and probe in "
                        "Pass A / Pass B). Pass C uses 0.")
    p.add_argument("--offsets", default=None,
                   help="Comma-separated offsets in deg. Default is "
                        f"{DEFAULT_OFFSETS}.")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--readout-start", type=int, default=9)
    p.add_argument("--readout-end", type=int, default=11,
                   help="Inclusive end of probe ON readout window.")
    p.add_argument("--save-per-trial", action="store_true",
                   help="If set, also dump per-trial arrays into the JSON "
                        "(under 'per_trial'). Off by default to keep JSON small.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    label = args.label or os.path.basename(args.checkpoint)

    if args.offsets is not None:
        offsets_deg = [float(s) for s in args.offsets.split(",") if s.strip()]
    else:
        offsets_deg = list(DEFAULT_OFFSETS)

    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] device={device}  n_trials/offset={args.n_trials}  "
          f"seed={args.rng_seed}  n_context={args.n_context}  "
          f"contrast={args.contrast}", flush=True)
    print(f"[setup] offsets_deg={offsets_deg}", flush=True)

    net, model_cfg, train_cfg, decoder = load_model_decoder(
        args.checkpoint, args.config, device,
    )
    trial_cfg = make_trial_cfg(train_cfg, n_context=args.n_context,
                               contrast=args.contrast)

    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    print(f"[setup] N={N}  period={period}  step_deg={step_deg:g}  "
          f"steps_on={trial_cfg.steps_on}  steps_isi={trial_cfg.steps_isi}",
          flush=True)

    # Sample one shared probe-channel batch using the seed (paired across offsets).
    g = torch.Generator().manual_seed(int(args.rng_seed))
    probe_chs = torch.randint(0, N, (args.n_trials,), generator=g)

    aggs: dict[float, dict[str, Any]] = {}
    per_trial_dump: dict[str, dict[str, list]] = {}
    diag = {"max_abs_ctx_AB": 0.0, "max_abs_ctx_AC": 0.0}

    for offset in offsets_deg:
        print(f"\n[offset {offset:>5.1f}°] running 3 passes "
              f"(B={args.n_trials})...", flush=True)
        per_trial = run_offset(
            net, decoder, model_cfg, trial_cfg,
            probe_chs=probe_chs, offset_deg=float(offset), device=device,
            readout_window=(args.readout_start, args.readout_end),
        )
        diag["max_abs_ctx_AB"] = max(diag["max_abs_ctx_AB"],
                                     per_trial["_diag_ctx_AB"])
        diag["max_abs_ctx_AC"] = max(diag["max_abs_ctx_AC"],
                                     per_trial["_diag_ctx_AC"])
        agg = aggregate_offset(per_trial, n_boot=args.n_boot,
                               seed=int(args.rng_seed) + int(offset))
        aggs[float(offset)] = agg

        # Print one-liner
        for letter in ("A", "B", "C"):
            m = agg[letter]
            print(f"  pass {letter} ({PASS_LABELS[letter]:>10s}): "
                  f"peak={m['peak']['mean']:.4f} "
                  f"[{m['peak']['lo']:.4f}, {m['peak']['hi']:.4f}]  "
                  f"total={m['total']['mean']:.3f} "
                  f"[{m['total']['lo']:.3f}, {m['total']['hi']:.3f}]  "
                  f"acc={m['correct']['mean']:.3f} "
                  f"[{m['correct']['lo']:.3f}, {m['correct']['hi']:.3f}]",
                  flush=True)
        print(f"  (mean pi_pred_eff at probe_isi_pre = "
              f"{agg['mean_pi_at_target']:.3f}, "
              f"n={agg['n']})", flush=True)
        print(f"  diag context-identity (last ctx ON window) "
              f"max abs A vs B = {per_trial['_diag_ctx_AB']:.2e}, "
              f"A vs C = {per_trial['_diag_ctx_AC']:.2e}", flush=True)

        if args.save_per_trial:
            # Strip the "_diag_*" keys and convert numpy → list for JSON
            keep = {k: v.tolist()
                    for k, v in per_trial.items()
                    if isinstance(v, np.ndarray)}
            per_trial_dump[str(float(offset))] = keep

    # Print summary table for quick reading
    print()
    print("=" * 84)
    print(f"Summary table — {label}")
    print("=" * 84)
    headers = ["off°", "pass", "n", "peak", "total", "acc"]
    col_w = [4, 11, 4, 18, 18, 18]
    fmt = " | ".join(f"{{:>{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in col_w))
    for offset in offsets_deg:
        agg = aggs[float(offset)]
        for letter in ("A", "B", "C"):
            m = agg[letter]
            row = [
                f"{offset:.0f}",
                PASS_LABELS[letter],
                str(agg["n"]),
                f"{m['peak']['mean']:.3f} "
                f"[{m['peak']['lo']:.3f},{m['peak']['hi']:.3f}]",
                f"{m['total']['mean']:.2f} "
                f"[{m['total']['lo']:.2f},{m['total']['hi']:.2f}]",
                f"{m['correct']['mean']:.3f} "
                f"[{m['correct']['lo']:.3f},{m['correct']['hi']:.3f}]",
            ]
            print(fmt.format(*row))
    print("=" * 84, flush=True)

    print(f"\n[diag] global max abs(context A vs B) = "
          f"{diag['max_abs_ctx_AB']:.3e}", flush=True)
    print(f"[diag] global max abs(context A vs C) = "
          f"{diag['max_abs_ctx_AC']:.3e}", flush=True)

    # Figure
    sub_title = (
        f"R1+R2 priming dose-response (n={args.n_trials}/offset; "
        f"context = {args.n_context}-hold; readout [9..11] inclusive)"
    )
    title = f"Priming dose-response — {label}\n{sub_title}"
    plot_dose_response(offsets_deg, aggs, args.output_fig, title=title)
    print(f"\n[fig] wrote {args.output_fig}", flush=True)

    # JSON
    out_meta = {
        "N": int(N),
        "period": float(period),
        "step_deg": float(step_deg),
        "steps_on": int(trial_cfg.steps_on),
        "steps_isi": int(trial_cfg.steps_isi),
        "n_context_pres": int(trial_cfg.n_context),
        "probe_onset_step": int(trial_cfg.n_context * (trial_cfg.steps_on
                                                       + trial_cfg.steps_isi)),
        "readout_window": {
            "start": int(args.readout_start),
            "end": int(args.readout_end),
            "inclusive": True,
        },
        "contrast": float(args.contrast),
        "n_trials_per_offset": int(args.n_trials),
        "offsets_deg": [float(o) for o in offsets_deg],
        "rng_seed": int(args.rng_seed),
        "n_boot": int(args.n_boot),
        "max_abs_context_diff_AB": float(diag["max_abs_ctx_AB"]),
        "max_abs_context_diff_AC": float(diag["max_abs_ctx_AC"]),
    }
    result = {
        "label": label,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "meta": out_meta,
        "aggregate_by_offset": {
            str(float(o)): aggs[float(o)] for o in offsets_deg
        },
    }
    if args.save_per_trial:
        result["per_trial_by_offset"] = per_trial_dump

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
