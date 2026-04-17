#!/usr/bin/env python3
"""Task #12 — main ex/unex evaluation on R1+R2 using Decoder C.

Per-trial design (per Lead's clarification, Reading 2):
  - Random S in [0°, 180°), D in [25°, 90°], CW/CCW (50/50) per trial.
    Per-trial RNG seed = SEED_BASE + trial_idx (independent of N — same
    trial_idx gives same S/D/dir across N values for paired comparison).
  - Pre-probe march: N presentations at fixed 5° steps starting at S.
        CW : S, S+5°, S+10°, ..., S+(N-1)·5°
        CCW: S, S-5°, S-10°, ..., S-(N-1)·5°
        All mod 180°.
  - ex probe ori   = S ± N·5°       (continuation)
  - unex probe ori = ex probe + sign·D   (same march direction)
  - Cue: σ=10° normalised Gaussian, ISI-only (steps_on:steps_per).
        cues[s] for s=1..N-1 points at orientations[s] (= march continuation)
        cues[s=N] (probe) points at ex_probe_ori in BOTH branches —
            so unex cue is "wrong" by D degrees.
        cues[s=0] = zeros (no cue at first presentation).
  - task_state = [1, 0] (focused) throughout.
  - contrast = 1.0 throughout (deterministic eval, full contrast).

Sweep:
  - 12 N values: {4, 5, ..., 15}
  - 200 trials per N
  - Per-N batched: one forward pass for ex branch, one for unex branch.
  - Total: 12 × 2 = 24 forward calls (4800 ex/unex paired trials).

Readout (per Lead's correction):
  - Probe ON window timesteps probe_onset+9 and probe_onset+10
    (Python slice [9:11] within probe ON; 2 timesteps; MEAN over the window).
  - r_probe = r_l23[:, probe_onset+9:probe_onset+11, :].mean(dim=1)  → [B, n_ori]
  - dec_acc: Decoder C top-1 vs probe-true channel
  - net L2/3: r_probe.sum(dim=-1)   (sum across 36 channels)

Outputs:
  - results/eval_ex_vs_unex_decC.json   — per-N stats + pooled
  - docs/figures/eval_ex_vs_unex_decC.png — 2-panel: dec_acc vs N, net_l23 vs N
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating

# Re-use canonical helpers: per-trial roll-to-center + linear-interp FWHM.
# Same convention as commit ce1b34e and matched_hmm_ring_sequence.py.
from matched_quality_sim import roll_to_center        # noqa: E402
from plot_tuning_ring_extended import fwhm_of_curve   # noqa: E402

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N_VALUES_DEFAULT = list(range(4, 16))   # {4, 5, ..., 15} → 12 values
N_TRIALS_DEFAULT = 200
SEED_BASE_DEFAULT = 42
MARCH_STEP_DEG = 5.0
D_LOW, D_HIGH = 25.0, 90.0
CUE_SIGMA_DEG = 10.0
READOUT_WIN = (9, 11)                   # Python slice [9:11] → indices 9, 10 (2 timesteps)
TASK_STATE = (1.0, 0.0)                 # focused (column 0 = relevant)
CONTRAST = 1.0

CKPT_PATH_DEFAULT = "results/simple_dual/emergent_seed42/checkpoint.pt"
DECODER_C_PATH_DEFAULT = "checkpoints/decoder_c.pt"
CONFIG_PATH_DEFAULT = "config/sweep/sweep_rescue_1_2.yaml"
OUT_JSON_DEFAULT = "results/eval_ex_vs_unex_decC.json"
OUT_FIG_DEFAULT = "docs/figures/eval_ex_vs_unex_decC.png"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def gaussian_cue_bumps(ori_deg: torch.Tensor, n_orientations: int,
                       period: float, sigma_deg: float) -> torch.Tensor:
    """Build σ-Gaussian cue bumps on the ring, normalised to sum=1.

    Matches the convention in HMMSequenceGenerator (sequences.py:441-444).

    Args:
        ori_deg: [B] orientations in degrees.
        n_orientations: number of channels (36).
        period: orientation wrap (180.0).
        sigma_deg: cue σ (10.0).

    Returns:
        cues: [B, n_orientations], each row sums to ~1.
    """
    prefs = torch.arange(n_orientations, dtype=torch.float32, device=ori_deg.device) \
        * (period / n_orientations)                              # [N]
    diff = ori_deg.unsqueeze(1) - prefs.unsqueeze(0)             # [B, N]
    # Wrap to [-period/2, period/2)
    diff = (diff + period / 2.0) % period - period / 2.0
    bump = torch.exp(-0.5 * (diff / sigma_deg) ** 2)             # [B, N]
    bump = bump / (bump.sum(dim=-1, keepdim=True) + 1e-8)
    return bump


def sample_per_trial_design(n_trials: int, seed_base: int,
                            device: torch.device
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample per-trial S, D, CW/CCW with per-trial seed.

    Per-trial RNG seed = seed_base + trial_idx (INDEPENDENT of N, so the
    same trial_idx gives the same draw across all N values).

    Returns:
        S:    [n_trials] in [0°, 180°)
        D:    [n_trials] in [D_LOW, D_HIGH]
        sign: [n_trials] in {+1, -1}  (+1 = CW, -1 = CCW)
    """
    S_list, D_list, sign_list = [], [], []
    for t in range(n_trials):
        rng = torch.Generator().manual_seed(seed_base + t)
        # Lead spec: "S: uniform in [0°, 360°), then mod 180°. Equivalent to
        # uniform in [0°, 180°)."
        S_list.append(float(torch.rand(1, generator=rng).item()) * 180.0)
        D_list.append(float(torch.rand(1, generator=rng).item()) * (D_HIGH - D_LOW) + D_LOW)
        sign_list.append(1.0 if torch.rand(1, generator=rng).item() < 0.5 else -1.0)
    S = torch.tensor(S_list, dtype=torch.float32, device=device)
    D = torch.tensor(D_list, dtype=torch.float32, device=device)
    sign = torch.tensor(sign_list, dtype=torch.float32, device=device)
    return S, D, sign


def build_trial_batch(N: int, n_trials: int, seed_base: int,
                      model_cfg, train_cfg, device: torch.device
                      ) -> dict[str, Any]:
    """Construct the paired ex/unex stim+cue+task_state batch for one N.

    Returns dict with:
        stim_ex   [B, T, n_ori]
        stim_unex [B, T, n_ori]
        cue       [B, T, n_ori]    (same in both branches)
        ts        [B, T, 2]        (= [1, 0] throughout)
        ex_ori    [B]              (deg)
        unex_ori  [B]              (deg)
        ex_ch     [B] long         (decoder ground-truth for ex)
        unex_ch   [B] long         (decoder ground-truth for unex)
        probe_onset int
        S, D, sign  [B] floats     (per-trial RNG draws)
    """
    n_ori = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / n_ori
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi

    S, D, sign = sample_per_trial_design(n_trials, seed_base, device)        # [B], [B], [B]

    # Build per-presentation orientations: total presentations = N + 1
    # (N pre-probe at indices 0..N-1, plus probe at index N).
    s_indices = torch.arange(N + 1, dtype=torch.float32, device=device)      # [N+1]
    march_oris = (S.unsqueeze(1) + sign.unsqueeze(1) * s_indices.unsqueeze(0) * MARCH_STEP_DEG) \
        % period                                                              # [B, N+1]
    ex_ori = march_oris[:, N]                                                 # [B]  (= S + sign*N*5°)
    unex_ori = (ex_ori + sign * D) % period                                   # [B]

    # ex stim sequence = full march continuation (probe = ex_ori)
    # unex stim sequence = same as ex except probe slot = unex_ori
    ex_oris_per_pres = march_oris.clone()                                     # [B, N+1]
    unex_oris_per_pres = march_oris.clone()
    unex_oris_per_pres[:, N] = unex_ori

    # Generate gratings (vectorized). Use full contrast.
    n_pres = N + 1
    contrast_vec = torch.full((n_trials * n_pres,), CONTRAST, dtype=torch.float32)

    ex_oris_flat = ex_oris_per_pres.reshape(-1).cpu()                          # [B*(N+1)]
    unex_oris_flat = unex_oris_per_pres.reshape(-1).cpu()

    ex_grats = generate_grating(
        ex_oris_flat, contrast_vec,
        n_orientations=n_ori,
        sigma=model_cfg.sigma_ff,
        n=model_cfg.naka_rushton_n,
        c50=model_cfg.naka_rushton_c50,
        period=period,
    ).reshape(n_trials, n_pres, n_ori).to(device)                              # [B, N+1, n_ori]

    unex_grats = generate_grating(
        unex_oris_flat, contrast_vec,
        n_orientations=n_ori,
        sigma=model_cfg.sigma_ff,
        n=model_cfg.naka_rushton_n,
        c50=model_cfg.naka_rushton_c50,
        period=period,
    ).reshape(n_trials, n_pres, n_ori).to(device)

    # Temporal expansion (matches trainer.py:252-254):
    #   stim ON during first steps_on timesteps of each presentation,
    #   ISI is zero in stim channel.
    T_total = n_pres * steps_per
    ex_stim_T = torch.zeros(n_trials, n_pres, steps_per, n_ori, device=device)
    unex_stim_T = torch.zeros(n_trials, n_pres, steps_per, n_ori, device=device)
    ex_stim_T[:, :, :steps_on, :] = ex_grats.unsqueeze(2)
    unex_stim_T[:, :, :steps_on, :] = unex_grats.unsqueeze(2)
    stim_ex = ex_stim_T.reshape(n_trials, T_total, n_ori)
    stim_unex = unex_stim_T.reshape(n_trials, T_total, n_ori)

    # Cue (matches trainer.py:259-261 + sequences.py:419-445):
    #   cues[s] for s>=1 placed in the ISI window of presentation s
    #   (timesteps steps_on:steps_per of presentation s).
    #   cues[s=0] = zeros (no cue at first presentation).
    #   cues[s] for s=1..N-1 points at march_oris[:, s] (= continuation).
    #   cues[s=N] (probe) points at ex_ori in BOTH branches per Lead.
    cue_per_pres = torch.zeros(n_trials, n_pres, n_ori, device=device)
    for s in range(1, n_pres):
        if s < N:
            ori_s = march_oris[:, s]
        else:                                  # s == N (probe)
            ori_s = ex_ori
        cue_per_pres[:, s, :] = gaussian_cue_bumps(ori_s, n_ori, period, CUE_SIGMA_DEG)
    cue_T = torch.zeros(n_trials, n_pres, steps_per, n_ori, device=device)
    cue_T[:, :, steps_on:, :] = cue_per_pres.unsqueeze(2)
    cue = cue_T.reshape(n_trials, T_total, n_ori)

    # task_state = [1, 0] (focused) at every timestep
    ts = torch.zeros(n_trials, T_total, 2, device=device)
    ts[..., 0] = TASK_STATE[0]
    ts[..., 1] = TASK_STATE[1]

    probe_onset = N * steps_per                                                # presentation N starts here

    ex_ch = (ex_ori / step_deg).round().long() % n_ori
    unex_ch = (unex_ori / step_deg).round().long() % n_ori

    return dict(
        stim_ex=stim_ex,
        stim_unex=stim_unex,
        cue=cue,
        ts=ts,
        ex_ori=ex_ori,
        unex_ori=unex_ori,
        ex_ch=ex_ch,
        unex_ch=unex_ch,
        probe_onset=int(probe_onset),
        S=S, D=D, sign=sign,
    )


def run_one_N(N: int, n_trials: int, seed_base: int,
              net: LaminarV1V2Network, decoder: nn.Linear,
              model_cfg, train_cfg, device: torch.device
              ) -> dict[str, Any]:
    """Run paired ex/unex eval for one march length N. Returns per-trial arrays."""
    bd = build_trial_batch(N, n_trials, seed_base, model_cfg, train_cfg, device)

    steps_on = train_cfg.steps_on
    win_lo = bd["probe_onset"] + READOUT_WIN[0]
    win_hi = bd["probe_onset"] + READOUT_WIN[1]
    assert READOUT_WIN[1] - READOUT_WIN[0] == 2, "Readout window must be 2 timesteps"
    assert READOUT_WIN[1] <= steps_on, \
        f"Readout window end {READOUT_WIN[1]} > steps_on {steps_on}"

    # Pass ex
    packed_ex = net.pack_inputs(bd["stim_ex"], bd["cue"], bd["ts"])
    r_l23_ex, _, _ = net.forward(packed_ex)                                    # [B, T, n_ori]
    r_probe_ex = r_l23_ex[:, win_lo:win_hi, :].mean(dim=1)                     # [B, n_ori]

    # Pass unex (cue identical, only stim differs at probe slot)
    packed_unex = net.pack_inputs(bd["stim_unex"], bd["cue"], bd["ts"])
    r_l23_unex, _, _ = net.forward(packed_unex)
    r_probe_unex = r_l23_unex[:, win_lo:win_hi, :].mean(dim=1)

    # Decoder C → top-1
    pred_ex = decoder(r_probe_ex).argmax(dim=-1)
    pred_unex = decoder(r_probe_unex).argmax(dim=-1)
    correct_ex = (pred_ex == bd["ex_ch"]).cpu().numpy().astype(np.float64)
    correct_unex = (pred_unex == bd["unex_ch"]).cpu().numpy().astype(np.float64)

    # Net L2/3 = sum across 36 channels of the readout vector
    net_l23_ex = r_probe_ex.sum(dim=-1).cpu().numpy().astype(np.float64)
    net_l23_unex = r_probe_unex.sum(dim=-1).cpu().numpy().astype(np.float64)

    # Pre-probe diagnostic: r_l23 at the LAST pre-probe presentation's readout.
    # Pass A vs B should be identical here (same context); useful smoke check.
    pre_onset = (N - 1) * (train_cfg.steps_on + train_cfg.steps_isi)
    r_pre_ex = r_l23_ex[:, pre_onset + READOUT_WIN[0]:pre_onset + READOUT_WIN[1], :].mean(dim=1)
    r_pre_unex = r_l23_unex[:, pre_onset + READOUT_WIN[0]:pre_onset + READOUT_WIN[1], :].mean(dim=1)
    pre_max_abs_diff = float((r_pre_ex - r_pre_unex).abs().max())

    # Task #13: per-trial peak-at-stim + FWHM. Re-center each trial's r_probe by
    # its TRUE probe orientation (ex_ch for ex branch, unex_ch for unex branch
    # — sign-aware via the shared sample_per_trial_design draws). Same convention
    # as commit ce1b34e / matched_hmm_ring_sequence.py: roll so true_ch lands at
    # center_idx = N // 2, then peak = re-centered[center_idx], FWHM via linear-
    # interp fwhm_of_curve in degrees.
    n_ori = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg_local = period / n_ori
    center_idx = n_ori // 2
    peak_ex, fwhm_ex = per_trial_peak_and_fwhm(
        r_probe_ex, bd["ex_ch"], center_idx, step_deg_local)
    peak_unex, fwhm_unex = per_trial_peak_and_fwhm(
        r_probe_unex, bd["unex_ch"], center_idx, step_deg_local)

    return dict(
        N=int(N),
        n=int(n_trials),
        correct_ex=correct_ex,
        correct_unex=correct_unex,
        net_l23_ex=net_l23_ex,
        net_l23_unex=net_l23_unex,
        peak_ex=peak_ex,
        peak_unex=peak_unex,
        fwhm_ex=fwhm_ex,
        fwhm_unex=fwhm_unex,
        ex_ch=bd["ex_ch"].cpu().numpy().astype(np.int64),
        unex_ch=bd["unex_ch"].cpu().numpy().astype(np.int64),
        pred_ex=pred_ex.cpu().numpy().astype(np.int64),
        pred_unex=pred_unex.cpu().numpy().astype(np.int64),
        S=bd["S"].cpu().numpy(),
        D=bd["D"].cpu().numpy(),
        sign=bd["sign"].cpu().numpy(),
        pre_max_abs_diff=pre_max_abs_diff,
    )


def mean_sem(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    s = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, s


def nanmean_sem(arr: np.ndarray) -> tuple[float, float, int]:
    """Like mean_sem but ignores NaNs. Returns (mean, sem, n_valid)."""
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    valid = arr[~np.isnan(arr)]
    n = int(len(valid))
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(valid.mean())
    s = float(valid.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, s, n


def per_trial_peak_and_fwhm(r_probe: "torch.Tensor", true_ch: "torch.Tensor",
                            center_idx: int, step_deg: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial re-center + extract peak-at-true and FWHM.

    For each trial i:
      rolled[i] = np.roll(r_probe[i], center_idx - true_ch[i])
                  (so true stim channel lands at center_idx)
      peak[i]   = rolled[i, center_idx]                 (= r_probe[i, true_ch[i]])
      fwhm[i]   = fwhm_of_curve(rolled[i], step_deg)    (linear-interp deg; NaN if no
                                                         crossing — flat ring or two-peak
                                                         ring where half-max never met)

    Args:
        r_probe:    [n, N] float — readout-window mean L2/3 per trial.
        true_ch:    [n]   long  — true probe orientation channel per trial.
        center_idx: int        — target center channel (= N // 2 = 18 for N=36).
        step_deg:   float      — channel spacing (= 5° for N=36, period=180°).

    Returns:
        peak: [n] float
        fwhm: [n] float (deg, NaN where crossing not found)
    """
    r_np = r_probe.detach().cpu().numpy().astype(np.float64)        # [n, N]
    ch_np = true_ch.detach().cpu().numpy().astype(np.int64)         # [n]
    rolled = roll_to_center(r_np, ch_np, center_idx=center_idx)     # [n, N]
    peak = rolled[:, center_idx].astype(np.float64)
    fwhm = np.array([fwhm_of_curve(rolled[i], step_deg)
                     for i in range(rolled.shape[0])], dtype=np.float64)
    return peak, fwhm


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=CKPT_PATH_DEFAULT)
    p.add_argument("--decoder-c", default=DECODER_C_PATH_DEFAULT)
    p.add_argument("--config", default=CONFIG_PATH_DEFAULT)
    p.add_argument("--output-json", default=OUT_JSON_DEFAULT)
    p.add_argument("--output-fig", default=OUT_FIG_DEFAULT)
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    p.add_argument("--n-values", type=int, nargs="+", default=N_VALUES_DEFAULT)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] decoder_c={args.decoder_c}", flush=True)
    print(f"[setup] N values={args.n_values}", flush=True)
    print(f"[setup] n_trials/N={args.n_trials}", flush=True)
    print(f"[setup] seed_base={args.seed_base}", flush=True)

    # Load config + model
    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    print(f"[cfg] n_orientations={model_cfg.n_orientations}  "
          f"period={model_cfg.orientation_range}  "
          f"steps_on={train_cfg.steps_on}  steps_isi={train_cfg.steps_isi}", flush=True)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    print(f"[setup] network loaded (oracle_mode={net.oracle_mode}, "
          f"feedback_scale={float(net.feedback_scale.item()):.3f})", flush=True)

    # Load Decoder C
    dec_ckpt = torch.load(args.decoder_c, map_location=device, weights_only=False)
    decoder = nn.Linear(model_cfg.n_orientations, model_cfg.n_orientations).to(device)
    decoder.load_state_dict(dec_ckpt["state_dict"])
    decoder.eval()
    best_val = dec_ckpt.get("train_meta", {}).get("best_val_acc", float("nan"))
    print(f"[setup] Decoder C loaded (best_val_acc={best_val:.4f})", flush=True)

    # Evaluate per N
    per_N_results = []
    pool_correct_ex, pool_correct_unex = [], []
    pool_net_ex, pool_net_unex = [], []
    pool_peak_ex, pool_peak_unex = [], []
    pool_fwhm_ex, pool_fwhm_unex = [], []
    pre_diff_max_overall = 0.0

    with torch.no_grad():
        for N_val in args.n_values:
            print(f"\n[run] N={N_val}", flush=True)
            res = run_one_N(N_val, args.n_trials, args.seed_base,
                            net, decoder, model_cfg, train_cfg, device)
            m_acc_ex, s_acc_ex = mean_sem(res["correct_ex"])
            m_acc_unex, s_acc_unex = mean_sem(res["correct_unex"])
            m_l23_ex, s_l23_ex = mean_sem(res["net_l23_ex"])
            m_l23_unex, s_l23_unex = mean_sem(res["net_l23_unex"])
            m_peak_ex, s_peak_ex = mean_sem(res["peak_ex"])
            m_peak_unex, s_peak_unex = mean_sem(res["peak_unex"])
            m_fwhm_ex, s_fwhm_ex, n_fwhm_ex = nanmean_sem(res["fwhm_ex"])
            m_fwhm_unex, s_fwhm_unex, n_fwhm_unex = nanmean_sem(res["fwhm_unex"])

            per_N_results.append({
                "N": N_val,
                "n_trials": res["n"],
                "dec_acc_ex_mean": m_acc_ex,
                "dec_acc_ex_sem": s_acc_ex,
                "dec_acc_unex_mean": m_acc_unex,
                "dec_acc_unex_sem": s_acc_unex,
                "net_l23_ex_mean": m_l23_ex,
                "net_l23_ex_sem": s_l23_ex,
                "net_l23_unex_mean": m_l23_unex,
                "net_l23_unex_sem": s_l23_unex,
                "peak_at_stim_ex_mean": m_peak_ex,
                "peak_at_stim_ex_sem": s_peak_ex,
                "peak_at_stim_unex_mean": m_peak_unex,
                "peak_at_stim_unex_sem": s_peak_unex,
                "fwhm_deg_ex_mean": m_fwhm_ex,
                "fwhm_deg_ex_sem": s_fwhm_ex,
                "fwhm_deg_ex_n_valid": n_fwhm_ex,
                "fwhm_deg_unex_mean": m_fwhm_unex,
                "fwhm_deg_unex_sem": s_fwhm_unex,
                "fwhm_deg_unex_n_valid": n_fwhm_unex,
                "delta_dec_acc_ex_minus_unex": m_acc_ex - m_acc_unex,
                "delta_net_l23_ex_minus_unex": m_l23_ex - m_l23_unex,
                "delta_peak_at_stim_ex_minus_unex": m_peak_ex - m_peak_unex,
                "delta_fwhm_deg_ex_minus_unex": m_fwhm_ex - m_fwhm_unex,
                "pre_probe_max_abs_diff": res["pre_max_abs_diff"],
            })
            pool_correct_ex.append(res["correct_ex"])
            pool_correct_unex.append(res["correct_unex"])
            pool_net_ex.append(res["net_l23_ex"])
            pool_net_unex.append(res["net_l23_unex"])
            pool_peak_ex.append(res["peak_ex"])
            pool_peak_unex.append(res["peak_unex"])
            pool_fwhm_ex.append(res["fwhm_ex"])
            pool_fwhm_unex.append(res["fwhm_unex"])
            pre_diff_max_overall = max(pre_diff_max_overall, res["pre_max_abs_diff"])

            print(f"  ex   :  dec_acc={m_acc_ex:.3f}±{s_acc_ex:.3f}   "
                  f"net_l23={m_l23_ex:.3f}±{s_l23_ex:.3f}   "
                  f"peak={m_peak_ex:.3f}±{s_peak_ex:.3f}   "
                  f"FWHM={m_fwhm_ex:.2f}±{s_fwhm_ex:.2f}° (n_valid={n_fwhm_ex})", flush=True)
            print(f"  unex :  dec_acc={m_acc_unex:.3f}±{s_acc_unex:.3f}   "
                  f"net_l23={m_l23_unex:.3f}±{s_l23_unex:.3f}   "
                  f"peak={m_peak_unex:.3f}±{s_peak_unex:.3f}   "
                  f"FWHM={m_fwhm_unex:.2f}±{s_fwhm_unex:.2f}° (n_valid={n_fwhm_unex})", flush=True)
            print(f"  Δ(ex−unex):  dec_acc={m_acc_ex - m_acc_unex:+.3f}   "
                  f"net_l23={m_l23_ex - m_l23_unex:+.3f}   "
                  f"peak={m_peak_ex - m_peak_unex:+.3f}   "
                  f"FWHM={m_fwhm_ex - m_fwhm_unex:+.2f}°", flush=True)
            print(f"  pre-probe max|ex−unex|={res['pre_max_abs_diff']:.2e}  (smoke check)", flush=True)

    # Pool across N values
    pool_correct_ex = np.concatenate(pool_correct_ex)
    pool_correct_unex = np.concatenate(pool_correct_unex)
    pool_net_ex = np.concatenate(pool_net_ex)
    pool_net_unex = np.concatenate(pool_net_unex)
    pool_peak_ex = np.concatenate(pool_peak_ex)
    pool_peak_unex = np.concatenate(pool_peak_unex)
    pool_fwhm_ex = np.concatenate(pool_fwhm_ex)
    pool_fwhm_unex = np.concatenate(pool_fwhm_unex)

    pm_acc_ex, ps_acc_ex = mean_sem(pool_correct_ex)
    pm_acc_unex, ps_acc_unex = mean_sem(pool_correct_unex)
    pm_l23_ex, ps_l23_ex = mean_sem(pool_net_ex)
    pm_l23_unex, ps_l23_unex = mean_sem(pool_net_unex)
    pm_peak_ex, ps_peak_ex = mean_sem(pool_peak_ex)
    pm_peak_unex, ps_peak_unex = mean_sem(pool_peak_unex)
    pm_fwhm_ex, ps_fwhm_ex, n_fwhm_ex_pool = nanmean_sem(pool_fwhm_ex)
    pm_fwhm_unex, ps_fwhm_unex, n_fwhm_unex_pool = nanmean_sem(pool_fwhm_unex)

    pooled = {
        "n_total": int(len(pool_correct_ex)),
        "dec_acc_ex_mean": pm_acc_ex,
        "dec_acc_ex_sem": ps_acc_ex,
        "dec_acc_unex_mean": pm_acc_unex,
        "dec_acc_unex_sem": ps_acc_unex,
        "net_l23_ex_mean": pm_l23_ex,
        "net_l23_ex_sem": ps_l23_ex,
        "net_l23_unex_mean": pm_l23_unex,
        "net_l23_unex_sem": ps_l23_unex,
        "peak_at_stim_ex_mean": pm_peak_ex,
        "peak_at_stim_ex_sem": ps_peak_ex,
        "peak_at_stim_unex_mean": pm_peak_unex,
        "peak_at_stim_unex_sem": ps_peak_unex,
        "fwhm_deg_ex_mean": pm_fwhm_ex,
        "fwhm_deg_ex_sem": ps_fwhm_ex,
        "fwhm_deg_ex_n_valid": n_fwhm_ex_pool,
        "fwhm_deg_unex_mean": pm_fwhm_unex,
        "fwhm_deg_unex_sem": ps_fwhm_unex,
        "fwhm_deg_unex_n_valid": n_fwhm_unex_pool,
        "delta_dec_acc_ex_minus_unex": pm_acc_ex - pm_acc_unex,
        "delta_net_l23_ex_minus_unex": pm_l23_ex - pm_l23_unex,
        "delta_peak_at_stim_ex_minus_unex": pm_peak_ex - pm_peak_unex,
        "delta_fwhm_deg_ex_minus_unex": pm_fwhm_ex - pm_fwhm_unex,
        "pre_probe_max_abs_diff_overall": pre_diff_max_overall,
    }

    print()
    print(f"[overall] n_total={pooled['n_total']}", flush=True)
    print(f"  ex   :  dec_acc={pm_acc_ex:.3f}±{ps_acc_ex:.3f}   "
          f"net_l23={pm_l23_ex:.3f}±{ps_l23_ex:.3f}   "
          f"peak={pm_peak_ex:.3f}±{ps_peak_ex:.3f}   "
          f"FWHM={pm_fwhm_ex:.2f}±{ps_fwhm_ex:.2f}° (n_valid={n_fwhm_ex_pool})", flush=True)
    print(f"  unex :  dec_acc={pm_acc_unex:.3f}±{ps_acc_unex:.3f}   "
          f"net_l23={pm_l23_unex:.3f}±{ps_l23_unex:.3f}   "
          f"peak={pm_peak_unex:.3f}±{ps_peak_unex:.3f}   "
          f"FWHM={pm_fwhm_unex:.2f}±{ps_fwhm_unex:.2f}° (n_valid={n_fwhm_unex_pool})", flush=True)
    print(f"  Δ(ex−unex):  dec_acc={pm_acc_ex - pm_acc_unex:+.3f}   "
          f"net_l23={pm_l23_ex - pm_l23_unex:+.3f}   "
          f"peak={pm_peak_ex - pm_peak_unex:+.3f}   "
          f"FWHM={pm_fwhm_ex - pm_fwhm_unex:+.2f}°", flush=True)
    print(f"  pre-probe max|ex−unex|={pre_diff_max_overall:.2e}  "
          f"(should be ~0 — context bit-identical until probe ON)", flush=True)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    result = {
        "label": "R1+R2 simple_dual emergent_seed42 — Decoder C ex/unex eval",
        "checkpoint": args.checkpoint,
        "decoder_c": args.decoder_c,
        "config": args.config,
        "design": {
            "march_step_deg": MARCH_STEP_DEG,
            "S_range_deg": [0.0, 180.0],
            "D_range_deg": [D_LOW, D_HIGH],
            "cue_sigma_deg": CUE_SIGMA_DEG,
            "task_state": list(TASK_STATE),
            "contrast": CONTRAST,
            "readout_window_steps": list(READOUT_WIN),
            "readout_window_n_steps": READOUT_WIN[1] - READOUT_WIN[0],
            "n_values": list(args.n_values),
            "n_trials_per_N": args.n_trials,
            "seed_base": args.seed_base,
            "per_trial_seed_formula": "seed_base + trial_idx (independent of N)",
        },
        "per_N": per_N_results,
        "pooled": pooled,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[json] wrote {args.output_json}", flush=True)

    # ------------------------------------------------------------------
    # Plot: 4 panels (dec_acc, net_l23, peak-at-stim, FWHM) vs N
    # ------------------------------------------------------------------
    Ns = [r["N"] for r in per_N_results]
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 13.0), sharex=True)

    ax = axes[0]
    ex_m = [r["dec_acc_ex_mean"] for r in per_N_results]
    ex_s = [r["dec_acc_ex_sem"] for r in per_N_results]
    unex_m = [r["dec_acc_unex_mean"] for r in per_N_results]
    unex_s = [r["dec_acc_unex_sem"] for r in per_N_results]
    ax.errorbar(Ns, ex_m, yerr=ex_s, fmt="o-", color="C0",
                label=f"Expected   (pooled {pm_acc_ex:.3f}±{ps_acc_ex:.3f})", capsize=3)
    ax.errorbar(Ns, unex_m, yerr=unex_s, fmt="s-", color="C1",
                label=f"Unexpected (pooled {pm_acc_unex:.3f}±{ps_acc_unex:.3f})", capsize=3)
    ax.set_ylabel("Decoder C top-1 accuracy")
    ax.set_title(f"Decoder accuracy vs march length N — R1+R2 (Decoder C)\n"
                 f"Δ(ex−unex) pooled = {pm_acc_ex - pm_acc_unex:+.3f}",
                 fontsize=11)
    ax.axhline(1.0 / model_cfg.n_orientations, color="gray", lw=0.6, ls=":",
               label=f"chance = 1/{model_cfg.n_orientations}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)

    ax = axes[1]
    ex_m = [r["net_l23_ex_mean"] for r in per_N_results]
    ex_s = [r["net_l23_ex_sem"] for r in per_N_results]
    unex_m = [r["net_l23_unex_mean"] for r in per_N_results]
    unex_s = [r["net_l23_unex_sem"] for r in per_N_results]
    ax.errorbar(Ns, ex_m, yerr=ex_s, fmt="o-", color="C0",
                label=f"Expected   (pooled {pm_l23_ex:.3f}±{ps_l23_ex:.3f})", capsize=3)
    ax.errorbar(Ns, unex_m, yerr=unex_s, fmt="s-", color="C1",
                label=f"Unexpected (pooled {pm_l23_unex:.3f}±{ps_l23_unex:.3f})", capsize=3)
    ax.set_ylabel("Net L2/3 activity (sum over 36 ch)")
    ax.set_title(f"Net L2/3 activity vs march length N\n"
                 f"Δ(ex−unex) pooled = {pm_l23_ex - pm_l23_unex:+.3f}",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ex_m = [r["peak_at_stim_ex_mean"] for r in per_N_results]
    ex_s = [r["peak_at_stim_ex_sem"] for r in per_N_results]
    unex_m = [r["peak_at_stim_unex_mean"] for r in per_N_results]
    unex_s = [r["peak_at_stim_unex_sem"] for r in per_N_results]
    ax.errorbar(Ns, ex_m, yerr=ex_s, fmt="o-", color="C0",
                label=f"Expected   (pooled {pm_peak_ex:.3f}±{ps_peak_ex:.3f})", capsize=3)
    ax.errorbar(Ns, unex_m, yerr=unex_s, fmt="s-", color="C1",
                label=f"Unexpected (pooled {pm_peak_unex:.3f}±{ps_peak_unex:.3f})", capsize=3)
    ax.set_ylabel(f"Peak-at-stim (re-cent. ch{model_cfg.n_orientations // 2})")
    ax.set_title(f"Peak L2/3 at true stim channel vs N\n"
                 f"Δ(ex−unex) pooled = {pm_peak_ex - pm_peak_unex:+.3f}",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ex_m = [r["fwhm_deg_ex_mean"] for r in per_N_results]
    ex_s = [r["fwhm_deg_ex_sem"] for r in per_N_results]
    unex_m = [r["fwhm_deg_unex_mean"] for r in per_N_results]
    unex_s = [r["fwhm_deg_unex_sem"] for r in per_N_results]
    ax.errorbar(Ns, ex_m, yerr=ex_s, fmt="o-", color="C0",
                label=f"Expected   (pooled {pm_fwhm_ex:.2f}±{ps_fwhm_ex:.2f}°, n_valid={n_fwhm_ex_pool})",
                capsize=3)
    ax.errorbar(Ns, unex_m, yerr=unex_s, fmt="s-", color="C1",
                label=f"Unexpected (pooled {pm_fwhm_unex:.2f}±{ps_fwhm_unex:.2f}°, n_valid={n_fwhm_unex_pool})",
                capsize=3)
    ax.set_xlabel("march length N (pre-probe presentations)")
    ax.set_ylabel("FWHM of re-centered tuning (deg)")
    ax.set_title(f"FWHM of L2/3 tuning curve vs N\n"
                 f"Δ(ex−unex) pooled = {pm_fwhm_ex - pm_fwhm_unex:+.2f}°",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(args.output_fig))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {args.output_fig}", flush=True)


if __name__ == "__main__":
    main()
