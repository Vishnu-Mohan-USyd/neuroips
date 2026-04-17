#!/usr/bin/env python3
"""Plot 3 example HMM training trials (Task #39).

For each of three representative HMM trials drawn from the training
distribution (no construction, no march-filtering), render:

  Top strip
  ---------
  Orientation trajectory across all 25 presentations.
    X = presentation index (0..24)
    Y = orientation in degrees (0..180, with circular wrap rendered as a
        line break — adjacent points are only joined when the signed
        circular delta has |Δ| < 90°).
    Blue line+dots: true stimulus orientation (`metadata.orientations[s]`).
    Red  line+dots: V2's predicted orientation at the LAST ISI step BEFORE
        each presentation's ON window begins (argmax of `q_pred[B, T, N]`
        at step ``s * steps_per - 1`` for s >= 1; pred at s=0 is undefined
        because there is no preceding ISI).
    Background shading per presentation:
        light grey  for task_state = focused  (relevant; column 0 = 1.0)
        very light  for task_state = routine  (irrelevant; column 1 = 1.0)
    Top annotation per presentation: pred_err in degrees (small text above).

  Bottom ring row
  ---------------
  25 small viridis ring icons, one per presentation, colored by mean L2/3
  activity over that presentation's ON readout window steps [9, 11]
  (inclusive). vmax shared across all 75 rings (3 trials × 25 rings).

Trial selection
===============
Generate a large HMM batch (256 sequences, seed 42), then pick three
"interesting" trials by criteria (no overlap):

  Trial A — Stay-dominant   : longest run of identical orientation.
  Trial B — CW/CCW run      : longest contiguous run of |signed_delta|
                              == transition_step in a single direction (≥3).
  Trial C — Multi-switch    : at least 3 task_state switches across the 25
                              presentations (column 0 of task_states flips).

If criteria don't all coexist in 256 trials, the script reports which
were missing and uses the best-available substitute.

Output
======
Figure : ``--output-fig`` (default
         ``docs/figures/hmm_trial_examples_r1_2.png``)
JSON   : ``--output-json`` (default
         ``results/hmm_trial_examples_r1_2.json``)
         contains for each trial:
           true_theta_deg [S]
           pred_theta_deg [S]  (NaN at s=0)
           pred_err_deg   [S]  (NaN at s=0)
           task_state_focused [S] (0/1)
           pi_pred_eff    [S]  (NaN at s=0)
           ring_window_avg [S, N]
           summary stats (mean pred_err excluding s=0, n switches, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from typing import Any

# repo root + scripts/ on sys.path
_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from src.config import ModelConfig, TrainingConfig, load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence

from matched_quality_sim import circular_distance


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _filter_dataclass_kwargs(cls, raw: dict | None) -> dict:
    if raw is None:
        return {}
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}


def load_model(checkpoint_path: str, config_path: str | None,
               device: torch.device
               ) -> tuple[LaminarV1V2Network, ModelConfig, TrainingConfig, Any]:
    """Load network + model/training/stim configs."""
    if config_path is not None:
        model_cfg, train_cfg, stim_cfg = load_config(config_path)
    else:
        ckpt_peek = torch.load(checkpoint_path, map_location="cpu",
                               weights_only=False)
        model_raw = dict(ckpt_peek.get("config", {}).get("model", {}))
        for legacy_key in ("mechanism", "n_basis", "max_apical_gain", "tau_vip",
                           "simple_feedback", "template_gain"):
            model_raw.pop(legacy_key, None)
        train_raw = dict(ckpt_peek.get("config", {}).get("training", {}))
        model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_raw))
        train_cfg = TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, train_raw))
        # No stim_cfg available without load_config — caller must pass --config.
        raise RuntimeError(
            "load_model: --config is required (stim_cfg cannot be reconstructed "
            "from a bare checkpoint)."
        )
        del ckpt_peek

    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    if hasattr(net, "oracle_mode"):
        net.oracle_mode = False
    if hasattr(net, "feedback_scale"):
        net.feedback_scale.fill_(1.0)

    return net, model_cfg, train_cfg, stim_cfg


# ---------------------------------------------------------------------------
# Per-presentation extraction
# ---------------------------------------------------------------------------

def signed_circ_delta_np(b: np.ndarray, a: np.ndarray, period: float
                         ) -> np.ndarray:
    """Signed circular delta b − a in (−period/2, period/2]."""
    return ((b - a + period / 2.0) % period) - period / 2.0


def collect_trials(net, model_cfg, train_cfg, stim_cfg,
                   n_batch: int, rng_seed: int, device: torch.device,
                   readout_window: tuple[int, int] = (9, 11),
                   ) -> dict[str, np.ndarray]:
    """Run one HMM batch through the network and return per-presentation arrays.

    Returns a dict of numpy arrays:
        true_theta            [B, S]   float (deg)
        pred_theta            [B, S]   float (deg)  — NaN at s=0
        pred_err              [B, S]   float (deg)  — NaN at s=0
        task_state_focused    [B, S]   float (0 or 1)
        contrasts             [B, S]   float
        is_ambiguous          [B, S]   bool
        pi_pred_eff           [B, S]   float        — NaN at s=0
        ring_window_avg       [B, S, N] float
    Plus meta keys.
    """
    seq_length = train_cfg.seq_length
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    w_start, w_end = readout_window  # inclusive

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

    g = torch.Generator().manual_seed(int(rng_seed))
    metadata = gen.generate(n_batch, seq_length, generator=g)
    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
        metadata, model_cfg, train_cfg, stim_cfg,
    )
    stim_seq = stim_seq.to(device)
    cue_seq = cue_seq.to(device)
    ts_seq = ts_seq.to(device)

    with torch.no_grad():
        packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
        r_l23_all, _, aux = net.forward(packed)  # r_l23_all [B, T, N]
        q_pred_all = aux["q_pred_all"]            # [B, T, N]
        pi_pred_eff_all = aux["pi_pred_eff_all"]  # [B, T, 1]

    B = stim_seq.shape[0]
    S = seq_length

    true_theta = metadata.orientations.cpu().numpy().astype(np.float32)  # [B, S]
    contrasts = metadata.contrasts.cpu().numpy().astype(np.float32)
    is_amb = metadata.is_ambiguous.cpu().numpy().astype(bool)
    task_states = metadata.task_states.cpu().numpy().astype(np.float32)  # [B, S, 2]
    task_focused = task_states[:, :, 0]  # [B, S]

    # Pred theta per presentation: argmax(q_pred at the LAST ISI step before
    # presentation s ON window). For s=0, no preceding ISI exists → NaN.
    pred_theta = np.full((B, S), np.nan, dtype=np.float32)
    pred_err = np.full((B, S), np.nan, dtype=np.float32)
    pi_eff = np.full((B, S), np.nan, dtype=np.float32)
    ring_window = np.zeros((B, S, N), dtype=np.float32)

    q_pred_np = q_pred_all.cpu().numpy()
    pi_np = pi_pred_eff_all[:, :, 0].cpu().numpy()
    r_l23_np = r_l23_all.cpu().numpy()

    for s in range(S):
        on_start = s * steps_per
        on_end_excl = on_start + steps_on  # window slice exclusive end
        # Ring window-avg: timesteps [on_start + 9 .. on_start + 11] inclusive.
        sl = slice(on_start + w_start, on_start + w_end + 1)
        ring_window[:, s, :] = r_l23_np[:, sl, :].mean(axis=1)

        if s >= 1:
            isi_pre_step = on_start - 1   # last ISI step before pres s ON
            pred_idx = q_pred_np[:, isi_pre_step, :].argmax(axis=-1)  # [B]
            pred_theta[:, s] = pred_idx.astype(np.float32) * step_deg
            pi_eff[:, s] = pi_np[:, isi_pre_step]
            # Circular distance in [0, period/2]
            d = np.abs(pred_theta[:, s] - true_theta[:, s])
            pred_err[:, s] = np.minimum(d, period - d).astype(np.float32)

    return {
        "true_theta": true_theta,
        "pred_theta": pred_theta,
        "pred_err": pred_err,
        "task_focused": task_focused,
        "contrasts": contrasts,
        "is_ambiguous": is_amb,
        "pi_pred_eff": pi_eff,
        "ring_window_avg": ring_window,
        "_meta": {
            "B": int(B),
            "S": int(S),
            "N": int(N),
            "period": float(period),
            "step_deg": float(step_deg),
            "steps_on": int(steps_on),
            "steps_isi": int(steps_isi),
            "transition_step_deg": float(stim_cfg.transition_step),
            "p_self": float(stim_cfg.p_self),
            "readout_window": (int(w_start), int(w_end)),
            "rng_seed": int(rng_seed),
            "n_batch": int(n_batch),
        },
    }


# ---------------------------------------------------------------------------
# Trial selection
# ---------------------------------------------------------------------------

def find_trial_indices(records: dict) -> dict[str, dict[str, Any]]:
    """Pick three representative trials (with no overlap if possible).

    Trial A — longest stay-run (consecutive same orientation).
    Trial B — longest CW or CCW directional run.
    Trial C — most task_state switches (must be >= 3 switches).
    """
    true_theta = records["true_theta"]   # [B, S]
    task_focused = records["task_focused"]  # [B, S]
    period = records["_meta"]["period"]
    transition_step = records["_meta"]["transition_step_deg"]
    B, S = true_theta.shape

    # Per-trial scores
    longest_stay = np.zeros(B, dtype=int)
    longest_dir_run = np.zeros(B, dtype=int)
    dominant_dir = np.zeros(B, dtype=int)        # +1 or -1
    n_switches = np.zeros(B, dtype=int)

    for i in range(B):
        # Stay run (consecutive identical, exact-equal because orientations
        # are sampled from the discrete grid)
        stays = (true_theta[i, 1:] == true_theta[i, :-1]).astype(int)
        if stays.size > 0:
            cur = best = 0
            for x in stays:
                if x:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            longest_stay[i] = best + 1 if best > 0 else 1  # length includes start

        # Directional run (signed_delta == ±transition_step in same direction
        # for consecutive steps)
        d = signed_circ_delta_np(true_theta[i, 1:], true_theta[i, :-1], period)
        dirs = np.where(np.abs(np.abs(d) - transition_step) < 0.5,
                        np.sign(d).astype(int), 0)
        if dirs.size > 0:
            best_run = 0
            best_sign = 0
            cur_run = 0
            cur_sign = 0
            for sd in dirs:
                if sd != 0 and sd == cur_sign:
                    cur_run += 1
                else:
                    cur_run = 1 if sd != 0 else 0
                    cur_sign = sd if sd != 0 else 0
                if cur_run > best_run:
                    best_run = cur_run
                    best_sign = cur_sign
            longest_dir_run[i] = best_run
            dominant_dir[i] = best_sign

        # Task-state switches (count flips along axis 1 of focused col)
        n_switches[i] = int((task_focused[i, 1:] != task_focused[i, :-1]).sum())

    chosen: dict[str, dict[str, Any]] = {}
    used: set[int] = set()

    # Trial A: longest stay-run (must be > 1)
    order_A = np.argsort(-longest_stay)  # descending
    for i in order_A:
        if longest_stay[i] >= 2 and i not in used:
            chosen["A"] = {
                "idx": int(i),
                "label": "Stay-dominant",
                "criterion": f"longest stay-run = {int(longest_stay[i])} consecutive same-orientation steps",
                "longest_stay": int(longest_stay[i]),
                "longest_dir_run": int(longest_dir_run[i]),
                "n_switches": int(n_switches[i]),
            }
            used.add(int(i))
            break

    # Trial B: longest directional run (≥ 3)
    order_B = np.argsort(-longest_dir_run)
    for i in order_B:
        if i in used:
            continue
        if longest_dir_run[i] >= 3:
            direction = "CW" if dominant_dir[i] > 0 else "CCW"
            chosen["B"] = {
                "idx": int(i),
                "label": f"{direction} transition run",
                "criterion": f"longest directional run = {int(longest_dir_run[i])} steps {direction}",
                "longest_stay": int(longest_stay[i]),
                "longest_dir_run": int(longest_dir_run[i]),
                "dominant_dir": int(dominant_dir[i]),
                "n_switches": int(n_switches[i]),
            }
            used.add(int(i))
            break

    # Trial C: ≥3 task_state switches (and disjoint from A,B if possible)
    order_C = np.argsort(-n_switches)
    for i in order_C:
        if i in used:
            continue
        if n_switches[i] >= 3:
            chosen["C"] = {
                "idx": int(i),
                "label": "Multi-switch (task state)",
                "criterion": f"task_state switches = {int(n_switches[i])} (focused↔routine)",
                "longest_stay": int(longest_stay[i]),
                "longest_dir_run": int(longest_dir_run[i]),
                "n_switches": int(n_switches[i]),
            }
            used.add(int(i))
            break

    # Fallbacks (if any criterion failed)
    fallback_msgs: list[str] = []
    if "A" not in chosen:
        i = int(np.argmax(longest_stay))
        chosen["A"] = {
            "idx": i, "label": "Stay-dominant (best available)",
            "criterion": f"max stay-run = {int(longest_stay[i])} (criterion ≥2 not met)",
            "longest_stay": int(longest_stay[i]),
            "longest_dir_run": int(longest_dir_run[i]),
            "n_switches": int(n_switches[i]),
        }
        fallback_msgs.append("A: best-available; criterion not strictly met.")
    if "B" not in chosen:
        i = int(np.argmax(longest_dir_run))
        chosen["B"] = {
            "idx": i, "label": "Best-available transition run",
            "criterion": f"max directional run = {int(longest_dir_run[i])} (criterion ≥3 not met)",
            "longest_stay": int(longest_stay[i]),
            "longest_dir_run": int(longest_dir_run[i]),
            "dominant_dir": int(dominant_dir[i]),
            "n_switches": int(n_switches[i]),
        }
        fallback_msgs.append("B: best-available; criterion not strictly met.")
    if "C" not in chosen:
        i = int(np.argmax(n_switches))
        chosen["C"] = {
            "idx": i, "label": "Most-switch (best available)",
            "criterion": f"max switches = {int(n_switches[i])} (criterion ≥3 not met)",
            "longest_stay": int(longest_stay[i]),
            "longest_dir_run": int(longest_dir_run[i]),
            "n_switches": int(n_switches[i]),
        }
        fallback_msgs.append("C: best-available; criterion not strictly met.")

    return {"trials": chosen, "fallback_msgs": fallback_msgs}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_trajectory_strip(
    ax_traj, true_theta: np.ndarray, pred_theta: np.ndarray,
    pred_err: np.ndarray, task_focused: np.ndarray, period: float,
    show_pred_err_text: bool = True,
) -> None:
    """Top trajectory strip per trial.

    `true_theta`, `pred_theta`, `pred_err`, `task_focused` are 1-D length S.
    Adjacent dots are connected ONLY when the signed circular delta is
    smaller than period/2 in absolute value (avoids visual jumps across
    the 180° wrap).
    """
    S = true_theta.shape[0]
    x = np.arange(S, dtype=float)

    # Background shading per presentation: focused vs routine.
    for s in range(S):
        col = "#e8e8e8" if task_focused[s] > 0.5 else "#f8f8f8"
        ax_traj.axvspan(s - 0.5, s + 0.5, color=col, alpha=0.85, zorder=0)

    # Helper: plot dots + line segments only when no wrap.
    def _plot_with_wrap_breaks(ax, vals, color, label):
        # Plot dots first.
        valid = ~np.isnan(vals)
        ax.scatter(x[valid], vals[valid], s=22, color=color, zorder=4,
                   edgecolors="white", linewidths=0.6)
        # Line segments only if both endpoints are valid AND
        # circular delta is < period/2 in absolute (i.e., no wrap).
        for s in range(S - 1):
            if not (valid[s] and valid[s + 1]):
                continue
            d = signed_circ_delta_np(np.array([vals[s + 1]]),
                                     np.array([vals[s]]), period)[0]
            if abs(d) < period / 2.0 - 1e-6 and abs(vals[s + 1] - vals[s]) < period / 2.0:
                ax.plot([x[s], x[s + 1]], [vals[s], vals[s + 1]],
                        color=color, linewidth=1.6, zorder=3, alpha=0.85,
                        label=(label if s == 0 else None))
            else:
                # Mark wrap location with a small vertical dashed line.
                ax.axvline(s + 0.5, color=color, linestyle="--",
                           linewidth=0.8, alpha=0.35, zorder=1)

    _plot_with_wrap_breaks(ax_traj, true_theta, "#1f77b4", "true ori")
    _plot_with_wrap_breaks(ax_traj, pred_theta, "#d62728", "V2 pred ori")

    # pred_err text above each presentation (small, only if defined)
    if show_pred_err_text:
        ymax = period
        for s in range(S):
            if np.isfinite(pred_err[s]):
                txt = f"{pred_err[s]:.0f}"
                # color by magnitude
                if pred_err[s] <= 5:
                    col = "#0a6e0a"
                elif pred_err[s] <= 20:
                    col = "#a08000"
                else:
                    col = "#a02020"
                ax_traj.text(s, ymax + 6.0, txt, ha="center", va="bottom",
                             fontsize=6.5, color=col, clip_on=False)

    ax_traj.set_xlim(-0.5, S - 0.5)
    ax_traj.set_ylim(-5, period + 5)
    ax_traj.set_yticks([0, 45, 90, 135, 180])
    ax_traj.set_xticks(np.arange(S))
    ax_traj.set_xticklabels([str(i) for i in range(S)], fontsize=7)
    ax_traj.tick_params(axis='y', labelsize=8)
    ax_traj.set_ylabel("orientation (deg)", fontsize=9)
    ax_traj.grid(True, alpha=0.20, linewidth=0.5)
    ax_traj.set_axisbelow(True)


def _plot_ring_strip(fig, gs_row_for_rings, ring_data: np.ndarray,
                     vmax: float, cmap, S: int, N: int) -> None:
    """Plot S small ring icons in row `gs_row_for_rings`.

    `ring_data` is [S, N] (one ring per presentation).
    `gs_row_for_rings` is a SubGridSpec returned by `gridspec.subgridspec(1, S)`.
    """
    inner_r = 0.62
    outer_r = 1.0
    height = outer_r - inner_r
    theta_centers = np.arange(N) * (2 * np.pi / N)
    width = 2 * np.pi / N
    norm = Normalize(vmin=0.0, vmax=vmax)

    for s in range(S):
        ax = fig.add_subplot(gs_row_for_rings[0, s], projection="polar")
        colors = cmap(norm(ring_data[s]))
        ax.bar(theta_centers, height, width=width, bottom=inner_r,
               color=colors, edgecolor="white", linewidth=0.25,
               align="center")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_ylim(0.0, 1.10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        # Tiny presentation index label below
        ax.text(0.5, -0.12, str(s), transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color="0.45")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(records: dict, chosen: dict[str, dict],
                 fig_path: str, label: str) -> None:
    meta = records["_meta"]
    S = meta["S"]
    N = meta["N"]
    period = meta["period"]

    # Compute shared vmax across all 3 trials' rings
    rings_per_trial = {}
    for tag, info in chosen.items():
        idx = info["idx"]
        rings_per_trial[tag] = records["ring_window_avg"][idx]  # [S, N]
    vmax = max(float(r.max()) for r in rings_per_trial.values())
    if vmax <= 0:
        vmax = 1.0
    cmap = matplotlib.colormaps["viridis"]

    # Layout: 3 trial blocks. Each block = traj strip (height 1.0)
    # + ring row (height 0.55) + caption (height 0.20). Plus colorbar at right.
    fig = plt.figure(figsize=(15.0, 13.5))
    outer_gs = fig.add_gridspec(
        3, 1, hspace=0.55, top=0.88, bottom=0.05, left=0.06, right=0.92,
    )

    for blk_idx, tag in enumerate(("A", "B", "C")):
        info = chosen[tag]
        idx = info["idx"]
        true_theta = records["true_theta"][idx]
        pred_theta = records["pred_theta"][idx]
        pred_err = records["pred_err"][idx]
        task_focused = records["task_focused"][idx]
        pi_eff = records["pi_pred_eff"][idx]
        is_amb = records["is_ambiguous"][idx]

        # Inner grid: 3 sub-rows (traj, rings, caption)
        inner = outer_gs[blk_idx, 0].subgridspec(
            3, 1, height_ratios=[1.0, 0.55, 0.20], hspace=0.18,
        )

        ax_traj = fig.add_subplot(inner[0, 0])
        title = (f"Trial {tag} ({info['label']}) — batch idx {idx}"
                 f"  •  {info['criterion']}")
        ax_traj.set_title(title, fontsize=10.5, fontweight="bold",
                          color="#1a1a1a", loc="left")
        _plot_trajectory_strip(
            ax_traj, true_theta, pred_theta, pred_err, task_focused, period,
        )

        # Rings sub-grid
        ring_grid = inner[1, 0].subgridspec(1, S, wspace=0.05)
        _plot_ring_strip(fig, ring_grid, rings_per_trial[tag], vmax, cmap,
                         S=S, N=N)

        # Caption row (text-only summary)
        ax_cap = fig.add_subplot(inner[2, 0])
        ax_cap.axis("off")
        # Compute summary stats (excluding s=0 where pred is undefined)
        valid = np.isfinite(pred_err)
        mean_err = float(np.nanmean(pred_err))
        median_err = float(np.nanmedian(pred_err))
        mean_pi = float(np.nanmean(pi_eff))
        n_amb = int(is_amb.sum())
        n_focused = int((task_focused > 0.5).sum())
        n_routine = int((task_focused <= 0.5).sum())
        cap = (
            f"mean pred_err = {mean_err:.2f}°  •  median pred_err = "
            f"{median_err:.2f}°  •  mean pi_pred_eff = {mean_pi:.2f}  •  "
            f"longest stay-run = {info['longest_stay']}  •  "
            f"longest dir-run = {info['longest_dir_run']}  •  "
            f"task switches = {info['n_switches']}  •  "
            f"focused/routine = {n_focused}/{n_routine}  •  "
            f"ambiguous presentations = {n_amb}"
        )
        ax_cap.text(0.0, 0.95, cap, transform=ax_cap.transAxes,
                    ha="left", va="top", fontsize=9.0, color="0.20")

        # Add a legend (only on the first block)
        if blk_idx == 0:
            handles = [
                plt.Line2D([0], [0], color="#1f77b4", marker="o",
                           markerfacecolor="#1f77b4",
                           markeredgecolor="white", markersize=6,
                           label="true orientation"),
                plt.Line2D([0], [0], color="#d62728", marker="o",
                           markerfacecolor="#d62728",
                           markeredgecolor="white", markersize=6,
                           label="V2 prediction (last ISI before pres)"),
                Patch(facecolor="#e8e8e8", edgecolor="0.6",
                      label="task = focused"),
                Patch(facecolor="#f8f8f8", edgecolor="0.6",
                      label="task = routine"),
            ]
            ax_traj.legend(handles=handles, loc="lower center",
                           fontsize=8.0, frameon=True, framealpha=0.95,
                           ncol=4, bbox_to_anchor=(0.5, 1.15))

    # Title
    fig.suptitle(
        f"Three example HMM training trials — {label}\n"
        f"Top: true (blue) vs V2-predicted (red) orientation per "
        f"presentation; pred_err° annotated above.\n"
        f"Bottom: 25 viridis rings of mean L2/3 over the [9..11] "
        f"on-window of each presentation (shared vmax).",
        fontsize=11.0, y=0.985,
    )

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.94, 0.08, 0.014, 0.80])
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Mean L2/3 over [9..11] on-window")

    out_dir = os.path.dirname(os.path.abspath(fig_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-fig", default="docs/figures/hmm_trial_examples_r1_2.png")
    p.add_argument("--output-json", default="results/hmm_trial_examples_r1_2.json")
    p.add_argument("--label", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--n-batch", type=int, default=256,
                   help="HMM batch size to draw from (script picks 3 trials).")
    p.add_argument("--readout-start", type=int, default=9)
    p.add_argument("--readout-end", type=int, default=11,
                   help="Inclusive end of presentation ON readout window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    label = args.label or os.path.basename(args.checkpoint)

    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] device={device}  n_batch={args.n_batch}  "
          f"seed={args.rng_seed}", flush=True)

    net, model_cfg, train_cfg, stim_cfg = load_model(
        args.checkpoint, args.config, device,
    )

    records = collect_trials(
        net, model_cfg, train_cfg, stim_cfg,
        n_batch=args.n_batch, rng_seed=args.rng_seed, device=device,
        readout_window=(args.readout_start, args.readout_end),
    )
    meta = records["_meta"]
    print(f"[collect] B={meta['B']}  S={meta['S']}  N={meta['N']}  "
          f"step_deg={meta['step_deg']:g}  "
          f"transition_step={meta['transition_step_deg']:g}  "
          f"p_self={meta['p_self']:g}", flush=True)

    pick = find_trial_indices(records)
    chosen = pick["trials"]
    for tag, info in chosen.items():
        print(f"[pick] Trial {tag}: idx={info['idx']}  label={info['label']}  "
              f"({info['criterion']})", flush=True)
    if pick["fallback_msgs"]:
        for m in pick["fallback_msgs"]:
            print(f"[pick] FALLBACK {m}", flush=True)

    # Build figure
    build_figure(records, chosen, args.output_fig, label)
    print(f"[fig] wrote {args.output_fig}", flush=True)

    # JSON dump (per-trial trajectories + summaries)
    out_meta = {
        "B": int(meta["B"]),
        "S": int(meta["S"]),
        "N": int(meta["N"]),
        "period": float(meta["period"]),
        "step_deg": float(meta["step_deg"]),
        "steps_on": int(meta["steps_on"]),
        "steps_isi": int(meta["steps_isi"]),
        "transition_step_deg": float(meta["transition_step_deg"]),
        "p_self": float(meta["p_self"]),
        "readout_window": list(meta["readout_window"]),
        "rng_seed": int(meta["rng_seed"]),
        "n_batch": int(meta["n_batch"]),
        "criteria": {
            "A": "longest stay-run (≥2)",
            "B": "longest CW or CCW directional run (≥3)",
            "C": "task_state switches (≥3)",
        },
        "fallback_msgs": pick["fallback_msgs"],
    }
    out_trials: dict[str, Any] = {}
    for tag, info in chosen.items():
        idx = info["idx"]
        true_theta = records["true_theta"][idx]
        pred_theta = records["pred_theta"][idx]
        pred_err = records["pred_err"][idx]
        task_focused = records["task_focused"][idx]
        is_amb = records["is_ambiguous"][idx]
        pi_eff = records["pi_pred_eff"][idx]
        ring_window = records["ring_window_avg"][idx]
        out_trials[tag] = {
            "info": info,
            "true_theta_deg": [float(x) for x in true_theta.tolist()],
            "pred_theta_deg": [None if (x is None or np.isnan(x)) else float(x)
                               for x in pred_theta.tolist()],
            "pred_err_deg":   [None if (x is None or np.isnan(x)) else float(x)
                               for x in pred_err.tolist()],
            "task_focused":   [int(x) for x in task_focused.tolist()],
            "is_ambiguous":   [bool(x) for x in is_amb.tolist()],
            "pi_pred_eff":    [None if (x is None or np.isnan(x)) else float(x)
                               for x in pi_eff.tolist()],
            "ring_window_avg": ring_window.astype(np.float32).tolist(),
            "summary": {
                "mean_pred_err_deg": float(np.nanmean(pred_err)),
                "median_pred_err_deg": float(np.nanmedian(pred_err)),
                "mean_pi_pred_eff": float(np.nanmean(pi_eff)),
                "n_focused": int((task_focused > 0.5).sum()),
                "n_routine": int((task_focused <= 0.5).sum()),
                "n_ambiguous": int(is_amb.sum()),
                "longest_stay_run": int(info["longest_stay"]),
                "longest_dir_run": int(info["longest_dir_run"]),
                "n_task_switches": int(info["n_switches"]),
            },
        }

    result = {
        "label": label,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": str(device),
        "meta": out_meta,
        "trials": out_trials,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] wrote {args.output_json}", flush=True)

    # Quick summary table to stdout
    print()
    print("=" * 90)
    for tag in ("A", "B", "C"):
        info = chosen[tag]
        idx = info["idx"]
        mean_err = float(np.nanmean(records["pred_err"][idx]))
        mean_pi = float(np.nanmean(records["pi_pred_eff"][idx]))
        print(f"Trial {tag} (idx={idx}): {info['label']}")
        print(f"  criterion: {info['criterion']}")
        print(f"  mean pred_err = {mean_err:.2f}°    "
              f"mean pi = {mean_pi:.2f}    "
              f"task_switches = {info['n_switches']}    "
              f"longest_stay = {info['longest_stay']}    "
              f"longest_dir_run = {info['longest_dir_run']}")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
