#!/usr/bin/env python3
"""Tuning-curve visualization: expected vs unexpected (Task #23).

For each of the 4 checkpoints (baseline, R1+2, R3, R4), generate HMM
sequences, classify post-first presentations as expected (pred_err <=
10 deg) / unexpected (pred_err > 20 deg) using the V2 prediction at the
last ISI timestep, extract r_l23 at the late-ON readout (t=9), re-center
each trial so true_theta_idx is at channel N//2, then average across
trials in each bucket.

Produces a single 2x4 PNG at docs/figures/exp_vs_unexp_tuning.png:
  Row 1: Cartesian line plots (x = offset from true theta, -90..+85 deg).
  Row 2: Polar plots (radius = mean r_l23, angle = channel).
Shared y-axis within each row; SEM bands shaded.

Inputs fixed by design (match task brief):
  - feedback_scale = 1.0 (FB ON)
  - seed = 42
  - readout timestep t = 9 (late-ON)
  - N batches * batch_size * (seq_length - 1) ~ 8000 presentations
  - expected threshold: pred_err <= 10 deg
  - unexpected threshold: pred_err > 20 deg
  - ambiguous trials dropped
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


# ------------------------------ helpers ------------------------------


def circular_distance(a: torch.Tensor, b: torch.Tensor, period: float = 180.0) -> torch.Tensor:
    """Absolute circular distance on [0, period)."""
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def compute_fwhm_and_peak(mean_curve: np.ndarray, step_deg: float) -> tuple[float, float]:
    """Compute (peak, FWHM_deg) from a centered tuning curve.

    FWHM is the distance between left/right half-max crossings around the
    argmax, using linear interpolation. Returns (peak, NaN) if either
    crossing falls outside the sampled window.
    """
    peak = float(mean_curve.max())
    if peak <= 0:
        return peak, float("nan")
    half = peak / 2.0
    N = len(mean_curve)
    peak_idx = int(np.argmax(mean_curve))

    # walk left
    left_cross = None
    for i in range(peak_idx, 0, -1):
        if mean_curve[i - 1] <= half < mean_curve[i]:
            denom = mean_curve[i] - mean_curve[i - 1]
            frac = (half - mean_curve[i - 1]) / denom if denom > 0 else 0.0
            left_cross = (i - 1) + frac
            break
    # walk right
    right_cross = None
    for i in range(peak_idx, N - 1):
        if mean_curve[i] > half >= mean_curve[i + 1]:
            denom = mean_curve[i] - mean_curve[i + 1]
            frac = (mean_curve[i] - half) / denom if denom > 0 else 0.0
            right_cross = i + frac
            break

    if left_cross is None or right_cross is None:
        return peak, float("nan")
    fwhm = (right_cross - left_cross) * step_deg
    return peak, float(fwhm)


def collect_tuning(config_path: str, checkpoint_path: str, device: torch.device,
                   seed: int, n_batches: int) -> dict:
    """Run the trial-generation loop and return aggregated exp/unexp curves.

    Returns dict with keys:
      exp_mean [N], exp_sem [N], exp_n, unexp_mean [N], unexp_sem [N],
      unexp_n, step_deg, N.
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
    CENTER = N // 2  # 18 for N=36
    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    T_READ = 9
    assert T_READ < steps_on, f"T_READ={T_READ} must be inside steps_on={steps_on}"

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

    exp_curves: list[np.ndarray] = []
    unexp_curves: list[np.ndarray] = []
    rng = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for _ in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg,
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net.forward(packed)  # [B, T, N]
            q_pred_all = aux["q_pred_all"]            # [B, T, N]

            B = r_l23_all.shape[0]
            true_ori = metadata.orientations.to(device)  # [B, S]

            for pres_i in range(1, seq_length):
                # classification via V2 prediction at last ISI timestep
                t_isi_last = pres_i * steps_per - 1
                q_pred_isi = q_pred_all[:, t_isi_last, :]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg
                actual_ori = true_ori[:, pres_i]
                is_amb = metadata.is_ambiguous[:, pres_i].to(device)
                pred_error = circular_distance(pred_ori, actual_ori, period)

                is_exp = (pred_error <= 10.0) & (~is_amb)
                is_unexp = (pred_error > 20.0) & (~is_amb)

                # r_l23 at late-ON readout
                t_readout = pres_i * steps_per + T_READ
                r_l23_t = r_l23_all[:, t_readout, :]  # [B, N]

                true_ch = (actual_ori / step_deg).round().long() % N  # [B]

                # re-center each trial: shift so true_ch ends up at CENTER
                shifts = (CENTER - true_ch).cpu().numpy()  # [B]
                r_np = r_l23_t.cpu().numpy()
                is_exp_np = is_exp.cpu().numpy()
                is_unexp_np = is_unexp.cpu().numpy()

                for b in range(B):
                    if is_exp_np[b]:
                        exp_curves.append(np.roll(r_np[b], int(shifts[b])))
                    elif is_unexp_np[b]:
                        unexp_curves.append(np.roll(r_np[b], int(shifts[b])))

    def agg(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
        if not curves:
            return np.zeros(N), np.zeros(N), 0
        arr = np.stack(curves)
        m = arr.mean(axis=0)
        s = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return m, s, arr.shape[0]

    e_m, e_s, e_n = agg(exp_curves)
    u_m, u_s, u_n = agg(unexp_curves)
    return {
        "exp_mean": e_m, "exp_sem": e_s, "exp_n": e_n,
        "unexp_mean": u_m, "unexp_sem": u_s, "unexp_n": u_n,
        "step_deg": step_deg, "N": N,
    }


# ------------------------------ main ---------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default="docs/figures/exp_vs_unexp_tuning.png",
                   help="Output PNG path (dir created if absent).")
    p.add_argument("--device", default=None,
                   help="Torch device; default cuda if available, else cpu.")
    p.add_argument("--n-batches", type=int, default=10,
                   help="Number of batches of (batch_size x seq_length) trials (default 10).")
    p.add_argument("--seed", type=int, default=42, help="Stimulus RNG seed (default 42).")
    p.add_argument("--baseline-config", default="config/sweep/sweep_simple_dual.yaml")
    p.add_argument("--baseline-checkpoint",
                   default="/home/vishnu/neuroips/simple_dual/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt")
    p.add_argument("--r12-checkpoint",
                   default="/home/vishnu/neuroips/rescue_1_2/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt")
    p.add_argument("--r3-checkpoint",
                   default="/home/vishnu/neuroips/rescue_3/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt")
    p.add_argument("--r4-checkpoint",
                   default="/home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    panels = [
        ("Baseline (simple_dual)", args.baseline_config,       args.baseline_checkpoint),
        ("Rescue 1+2",             "config/sweep/sweep_rescue_1_2.yaml", args.r12_checkpoint),
        ("Rescue 3 (VIP)",         "config/sweep/sweep_rescue_3.yaml",   args.r3_checkpoint),
        ("Rescue 4 (DeepTemplate)","config/sweep/sweep_rescue_4.yaml",   args.r4_checkpoint),
    ]

    data: list[dict] = []
    for label, cfg, ckpt in panels:
        print(f"[collect] {label}", flush=True)
        res = collect_tuning(cfg, ckpt, device, seed=args.seed, n_batches=args.n_batches)
        res["label"] = label
        res["peak_exp"], res["fwhm_exp"] = compute_fwhm_and_peak(res["exp_mean"], res["step_deg"])
        res["peak_unexp"], res["fwhm_unexp"] = compute_fwhm_and_peak(res["unexp_mean"], res["step_deg"])
        print(f"  exp   n={res['exp_n']:5d}  peak={res['peak_exp']:.3f}  FWHM={res['fwhm_exp']:.2f} deg", flush=True)
        print(f"  unexp n={res['unexp_n']:5d}  peak={res['peak_unexp']:.3f}  FWHM={res['fwhm_unexp']:.2f} deg", flush=True)
        data.append(res)

    # Shared y-axis per row (max across panels)
    cart_max = max(
        max((d["exp_mean"] + d["exp_sem"]).max(), (d["unexp_mean"] + d["unexp_sem"]).max())
        for d in data
    )
    cart_ylim = (0.0, 1.1 * cart_max)

    # Create figure: 2 rows x 4 cols
    fig = plt.figure(figsize=(18, 8))

    for j, d in enumerate(data):
        N = d["N"]
        step_deg = d["step_deg"]
        x = (np.arange(N) - N // 2) * step_deg  # -90..+85 deg in 5 deg steps

        # --- Row 1: Cartesian ---
        ax_c = fig.add_subplot(2, 4, j + 1)
        ax_c.fill_between(
            x, d["exp_mean"] - d["exp_sem"], d["exp_mean"] + d["exp_sem"],
            color="tab:blue", alpha=0.25, linewidth=0,
        )
        ax_c.plot(x, d["exp_mean"], color="tab:blue", lw=2.0,
                  label=f'Expected (n={d["exp_n"]})')
        ax_c.fill_between(
            x, d["unexp_mean"] - d["unexp_sem"], d["unexp_mean"] + d["unexp_sem"],
            color="tab:red", alpha=0.25, linewidth=0,
        )
        ax_c.plot(x, d["unexp_mean"], color="tab:red", lw=2.0,
                  label=f'Unexpected (n={d["unexp_n"]})')
        ax_c.set_title(d["label"], fontsize=11)
        ax_c.set_xlabel("Offset from true θ (°)")
        if j == 0:
            ax_c.set_ylabel("Mean $r_{L2/3}$ (re-centered, t=9)")
        ax_c.set_xlim(x[0], x[-1])
        ax_c.set_ylim(cart_ylim)
        ax_c.grid(alpha=0.3)
        ax_c.axvline(0, color="gray", ls="--", lw=0.6)
        if j == 0:
            ax_c.legend(loc="upper left", fontsize=8, framealpha=0.9)

        ann = (
            f"peak_exp   = {d['peak_exp']:.2f}\n"
            f"peak_unexp = {d['peak_unexp']:.2f}\n"
            f"FWHM_exp   = {d['fwhm_exp']:.1f}°\n"
            f"FWHM_unexp = {d['fwhm_unexp']:.1f}°"
        )
        ax_c.text(
            0.98, 0.97, ann, transform=ax_c.transAxes,
            ha="right", va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )

        # --- Row 2: Polar ---
        ax_p = fig.add_subplot(2, 4, 4 + j + 1, projection="polar")
        # Map 36 channels to a full 2π ring (orientation has 180° period;
        # doubling to 360° gives a visually natural polar plot).
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        theta_closed = np.append(theta, theta[0] + 2 * np.pi)
        exp_closed = np.append(d["exp_mean"], d["exp_mean"][0])
        unexp_closed = np.append(d["unexp_mean"], d["unexp_mean"][0])
        exp_sem_cl = np.append(d["exp_sem"], d["exp_sem"][0])
        unexp_sem_cl = np.append(d["unexp_sem"], d["unexp_sem"][0])

        ax_p.fill_between(
            theta_closed, exp_closed - exp_sem_cl, exp_closed + exp_sem_cl,
            color="tab:blue", alpha=0.25, linewidth=0,
        )
        ax_p.plot(theta_closed, exp_closed, color="tab:blue", lw=2.0)
        ax_p.fill_between(
            theta_closed, unexp_closed - unexp_sem_cl, unexp_closed + unexp_sem_cl,
            color="tab:red", alpha=0.25, linewidth=0,
        )
        ax_p.plot(theta_closed, unexp_closed, color="tab:red", lw=2.0)

        # Peak aligned to channel CENTER = N//2, which after the linspace
        # falls at angle π (bottom). Rotate so it points up for clarity.
        ax_p.set_theta_offset(-np.pi / 2 - np.pi)  # peak → top
        ax_p.set_theta_direction(-1)
        ax_p.set_ylim(0.0, 1.1 * cart_max)
        ax_p.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax_p.set_xticklabels([""] * 8)  # declutter; Cartesian panel already labels offsets
        ax_p.grid(alpha=0.4)
        ax_p.set_title("", fontsize=8)

    fig.suptitle(
        "Expected vs Unexpected — mean L2/3 tuning curve (FB ON, t=9, seed 42)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.output}", flush=True)


if __name__ == "__main__":
    main()
