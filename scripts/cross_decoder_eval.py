#!/usr/bin/env python3
"""Task #17 — Cross-decoder evaluation across 6 strategies on R1+R2.

For each strategy: re-runs the EXACT same forward pass + masking the original
analysis used (so the strategy's NATIVE-decoder Δdec reproduces the published
value), then applies the OPPOSITE decoder to the SAME per-trial r_l23
readouts under the SAME masks. No model retraining; one re-forward per
strategy.

Strategies:
  NEW eval  : eval_ex_vs_unex_decC — paired ex/unex, 12 N × 200 trials
              (decoder native = C; alt = A)
  M3R       : matched_3row_ring (HMM, pred_err ≤ 5° vs > 20°, pi-Q75)
              (decoder native = A; alt = C)
  HMS       : matched_hmm_ring_sequence default mode (3-march vs march-jump)
  HMS-T     : matched_hmm_ring_sequence --tight-expected
  P3P       : matched_probe_3pass (Pass A=Expected probe vs Pass B=Unexpected)
  VCD-test3 : v2_confidence_dissection test3 (1-to-1 pi matched pairing)

Output: results/cross_decoder_r1_2.json — one block per strategy with
n_ex, n_unex, dec{A,C}_acc_ex, dec{A,C}_acc_unex, dec{A,C}_delta.

Decoder A: orientation_decoder loaded from the network checkpoint.
Decoder C: checkpoints/decoder_c.pt (Linear(36,36), trained on synthetic bumps).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Make scripts/ importable so we can pull helpers from sibling scripts
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Project root (parent of scripts/) for src/ imports
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matched_3row_ring as m3r                 # noqa: E402
import matched_hmm_ring_sequence as hms         # noqa: E402
import matched_probe_3pass as p3p               # noqa: E402
import v2_confidence_dissection as vcd          # noqa: E402
import eval_ex_vs_unex_decC as eval_new         # noqa: E402

from src.config import load_config              # noqa: E402
from src.model.network import LaminarV1V2Network  # noqa: E402
from src.stimulus.sequences import HMMSequenceGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Task #18 input overrides: focused task_state + march-following cue.
# ---------------------------------------------------------------------------

def make_march_cue_bumps(orientations: torch.Tensor, period: float, n_ori: int,
                         cue_sigma: float = 10.0) -> torch.Tensor:
    """Build deterministic march-continuation cue bumps for every presentation.

    For p ≥ 2:  predicted[p] = (2 * stim[p-1] - stim[p-2]) mod period
                = stim[p-1] + (stim[p-1] - stim[p-2]) mod period
    For p = 0 or 1: zeros (no march history yet).

    Bump construction matches HMMSequenceGenerator.generate exactly:
      dists  = min(|prefs - cue_ori|, period - |prefs - cue_ori|)
      bump   = exp(-dists**2 / (2 * sigma**2))
      bump  /= bump.sum() + 1e-8

    Args:
        orientations: [B, S] degrees, continuous.
        period:       orientation_range, e.g. 180.0.
        n_ori:        number of orientation channels.
        cue_sigma:    Gaussian sigma in degrees on the ring (default 10.0,
                      matching HMMSequenceGenerator).
    Returns:
        cues: [B, S, n_ori].
    """
    B, S = orientations.shape
    device = orientations.device
    cues = torch.zeros(B, S, n_ori, dtype=torch.float32, device=device)
    if S < 3:
        return cues
    prefs = torch.arange(n_ori, dtype=torch.float32, device=device) * (period / n_ori)
    pred = (2.0 * orientations[:, 1:S - 1] - orientations[:, 0:S - 2]) % period
    pred_flat = pred.reshape(-1)
    dists = torch.abs(prefs.unsqueeze(0) - pred_flat.unsqueeze(1))
    dists = torch.min(dists, period - dists)
    bumps = torch.exp(-(dists ** 2) / (2.0 * cue_sigma ** 2))
    bumps = bumps / (bumps.sum(dim=-1, keepdim=True) + 1e-8)
    cues[:, 2:, :] = bumps.reshape(B, S - 2, n_ori).to(cues.dtype)
    return cues


def apply_input_overrides(metadata, period: float, n_ori: int):
    """Pin task_state to focused [1,0] for every presentation; replace cues
    with deterministic march-continuation bumps. Mutates metadata in-place
    AND returns it (it's a @dataclass)."""
    B, S = metadata.orientations.shape
    new_ts = torch.zeros(B, S, 2, dtype=metadata.task_states.dtype,
                         device=metadata.task_states.device)
    new_ts[..., 0] = 1.0
    metadata.task_states = new_ts
    metadata.cues = make_march_cue_bumps(metadata.orientations, period, n_ori)
    return metadata


_ORIGINAL_GENERATE = None


def install_input_overrides(period: float, n_ori: int) -> None:
    """Monkey-patch HMMSequenceGenerator.generate to apply Task #18 overrides
    after every call. Idempotent — if already installed, refuses to re-wrap."""
    global _ORIGINAL_GENERATE
    if _ORIGINAL_GENERATE is not None:
        return
    _ORIGINAL_GENERATE = HMMSequenceGenerator.generate

    def patched_generate(self, batch_size, seq_length, generator=None):
        metadata = _ORIGINAL_GENERATE(self, batch_size, seq_length,
                                      generator=generator)
        return apply_input_overrides(metadata, period, n_ori)

    HMMSequenceGenerator.generate = patched_generate
    print(f"[override] HMMSequenceGenerator.generate patched: "
          f"task_state=[1,0] focused, cue=march-continuation σ=10°, "
          f"period={period}, n_ori={n_ori}", flush=True)


def uninstall_input_overrides() -> None:
    """Restore the original HMMSequenceGenerator.generate."""
    global _ORIGINAL_GENERATE
    if _ORIGINAL_GENERATE is None:
        return
    HMMSequenceGenerator.generate = _ORIGINAL_GENERATE
    _ORIGINAL_GENERATE = None
    print("[override] uninstalled", flush=True)


# ---------------------------------------------------------------------------
# Decoder loaders
# ---------------------------------------------------------------------------

def load_decoder_a(ckpt: dict, N: int, device: torch.device) -> nn.Linear:
    """Load Decoder A from a network checkpoint (`orientation_decoder`)."""
    dec = nn.Linear(N, N).to(device)
    if ('loss_heads' in ckpt and isinstance(ckpt['loss_heads'], dict)
            and 'orientation_decoder' in ckpt['loss_heads']):
        dec.load_state_dict(ckpt['loss_heads']['orientation_decoder'])
    elif 'decoder_state' in ckpt:
        dec.load_state_dict(ckpt['decoder_state'])
    else:
        raise RuntimeError(
            "Checkpoint has no orientation_decoder weights "
            "(tried ckpt['loss_heads']['orientation_decoder'] and ckpt['decoder_state'])"
        )
    dec.eval()
    return dec


def load_decoder_c(path: str, N: int, device: torch.device) -> nn.Linear:
    """Load Decoder C from `checkpoints/decoder_c.pt`."""
    dc = torch.load(path, map_location=device, weights_only=False)
    sd = dc['state_dict'] if isinstance(dc, dict) and 'state_dict' in dc else dc
    dec = nn.Linear(N, N).to(device)
    dec.load_state_dict(sd)
    dec.eval()
    return dec


def acc_one(decoder: nn.Linear, r: np.ndarray, true_ch: np.ndarray
            ) -> tuple[float | None, int]:
    """Top-1 accuracy of `decoder` on r vs true_ch. r: [n, N]."""
    n = int(r.shape[0])
    if n == 0:
        return None, 0
    with torch.no_grad():
        pred = decoder(torch.from_numpy(np.ascontiguousarray(r)).float()
                       ).argmax(dim=-1).numpy()
    return float((pred == true_ch).mean()), n


# ---------------------------------------------------------------------------
# _Args helper for module helpers that take an argparse-like object
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def eval_NEW(checkpoint: str, config: str, decoder_c_path: str,
             device: torch.device,
             n_trials_per_N: int = 200,
             n_values: list[int] | None = None,
             seed_base: int = 42) -> dict[str, Any]:
    """NEW eval (Tasks #12/#13) under both decoders. One forward per N."""
    if n_values is None:
        n_values = list(eval_new.N_VALUES_DEFAULT)

    model_cfg, train_cfg, _ = load_config(config)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state'])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    n_ori = model_cfg.n_orientations
    decA = load_decoder_a(ckpt, n_ori, torch.device('cpu'))
    decC = load_decoder_c(decoder_c_path, n_ori, torch.device('cpu'))

    cor_ex_A: list[np.ndarray] = []
    cor_ex_C: list[np.ndarray] = []
    cor_unex_A: list[np.ndarray] = []
    cor_unex_C: list[np.ndarray] = []
    per_N: list[dict] = []

    with torch.no_grad():
        for N in n_values:
            bd = eval_new.build_trial_batch(N, n_trials_per_N, seed_base,
                                            model_cfg, train_cfg, device)
            steps_on = train_cfg.steps_on
            win_lo = bd['probe_onset'] + eval_new.READOUT_WIN[0]
            win_hi = bd['probe_onset'] + eval_new.READOUT_WIN[1]
            assert eval_new.READOUT_WIN[1] - eval_new.READOUT_WIN[0] == 2
            assert eval_new.READOUT_WIN[1] <= steps_on

            packed_ex = net.pack_inputs(bd['stim_ex'], bd['cue'], bd['ts'])
            r_l23_ex, _, _ = net.forward(packed_ex)
            r_probe_ex = r_l23_ex[:, win_lo:win_hi, :].mean(dim=1).cpu().numpy()

            packed_unex = net.pack_inputs(bd['stim_unex'], bd['cue'], bd['ts'])
            r_l23_unex, _, _ = net.forward(packed_unex)
            r_probe_unex = r_l23_unex[:, win_lo:win_hi, :].mean(dim=1).cpu().numpy()

            ex_ch = bd['ex_ch'].cpu().numpy()
            unex_ch = bd['unex_ch'].cpu().numpy()

            with torch.no_grad():
                pred_ex_A = decA(torch.from_numpy(r_probe_ex)).argmax(-1).numpy()
                pred_ex_C = decC(torch.from_numpy(r_probe_ex)).argmax(-1).numpy()
                pred_unex_A = decA(torch.from_numpy(r_probe_unex)).argmax(-1).numpy()
                pred_unex_C = decC(torch.from_numpy(r_probe_unex)).argmax(-1).numpy()

            ce_A = (pred_ex_A == ex_ch).astype(np.float64)
            ce_C = (pred_ex_C == ex_ch).astype(np.float64)
            cu_A = (pred_unex_A == unex_ch).astype(np.float64)
            cu_C = (pred_unex_C == unex_ch).astype(np.float64)

            cor_ex_A.append(ce_A); cor_ex_C.append(ce_C)
            cor_unex_A.append(cu_A); cor_unex_C.append(cu_C)
            per_N.append({
                'N': int(N), 'n': int(ex_ch.size),
                'decA_acc_ex': float(ce_A.mean()),
                'decA_acc_unex': float(cu_A.mean()),
                'decC_acc_ex': float(ce_C.mean()),
                'decC_acc_unex': float(cu_C.mean()),
                'decA_delta': float(ce_A.mean() - cu_A.mean()),
                'decC_delta': float(ce_C.mean() - cu_C.mean()),
            })
            print(f"[NEW] N={N:>2}  decA Δ={ce_A.mean()-cu_A.mean():+.4f}  "
                  f"decC Δ={ce_C.mean()-cu_C.mean():+.4f}  n={ex_ch.size}",
                  flush=True)

    cea = np.concatenate(cor_ex_A); cec = np.concatenate(cor_ex_C)
    cua = np.concatenate(cor_unex_A); cuc = np.concatenate(cor_unex_C)
    return {
        'strategy': 'NEW (eval_ex_vs_unex_decC)',
        'native_decoder': 'C',
        'n_ex': int(cea.size), 'n_unex': int(cua.size),
        'decA_acc_ex': float(cea.mean()), 'decA_acc_unex': float(cua.mean()),
        'decC_acc_ex': float(cec.mean()), 'decC_acc_unex': float(cuc.mean()),
        'decA_delta': float(cea.mean() - cua.mean()),
        'decC_delta': float(cec.mean() - cuc.mean()),
        'per_N': per_N,
    }


def eval_M3R(checkpoint: str, config: str, decoder_c_path: str,
             device: torch.device, n_batches: int = 40, seed: int = 42
             ) -> dict[str, Any]:
    """M3R (matched_3row_ring) — Q75 pi, exp_pred_err 5° (with widening cascade)."""
    args = _Args(config=config, checkpoint=checkpoint,
                 device=str(device), n_batches=n_batches, rng_seed=seed,
                 target_idx=None)
    records, meta = m3r.collect_records(args, device)
    N = meta['N']

    exp_max = 5.0
    pi_pct = 75.0
    min_n = 200
    buckets, pi_thr = m3r.make_buckets(records, exp_pred_err_max=exp_max,
                                       pi_q_pct=pi_pct)
    n_e = buckets['expected']['r_win'].shape[0]
    n_u = buckets['unexpected']['r_win'].shape[0]
    n_o = buckets['omission']['r_win'].shape[0]
    if min(n_e, n_u, n_o) < min_n:
        exp_max = 10.0
        buckets, pi_thr = m3r.make_buckets(records, exp_pred_err_max=exp_max,
                                           pi_q_pct=pi_pct)
        n_e = buckets['expected']['r_win'].shape[0]
        n_u = buckets['unexpected']['r_win'].shape[0]
        n_o = buckets['omission']['r_win'].shape[0]
    if min(n_e, n_u, n_o) < min_n:
        pi_pct = 50.0
        buckets, pi_thr = m3r.make_buckets(records, exp_pred_err_max=exp_max,
                                           pi_q_pct=pi_pct)

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    decA = load_decoder_a(ckpt, N, torch.device('cpu'))
    decC = load_decoder_c(decoder_c_path, N, torch.device('cpu'))

    out: dict[str, Any] = {
        'strategy': 'M3R (matched_3row_ring)',
        'native_decoder': 'A',
        'pi_pct_used': pi_pct,
        'exp_pred_err_max_used': exp_max,
    }
    for name in ['expected', 'unexpected', 'omission']:
        b = buckets[name]
        a_acc, n = acc_one(decA, b['r_win'], b['true_ch'])
        c_acc, _ = acc_one(decC, b['r_win'], b['true_ch'])
        out[f'{name}_n'] = n
        out[f'{name}_decA_acc'] = a_acc
        out[f'{name}_decC_acc'] = c_acc
        print(f"[M3R] {name:<11s} n={n}  decA={a_acc}  decC={c_acc}", flush=True)
    out['n_ex'] = out['expected_n']
    out['n_unex'] = out['unexpected_n']
    out['decA_acc_ex'] = out['expected_decA_acc']
    out['decA_acc_unex'] = out['unexpected_decA_acc']
    out['decC_acc_ex'] = out['expected_decC_acc']
    out['decC_acc_unex'] = out['unexpected_decC_acc']
    out['decA_delta'] = out['decA_acc_ex'] - out['decA_acc_unex']
    out['decC_delta'] = out['decC_acc_ex'] - out['decC_acc_unex']
    return out


def _eval_HMS_inner(args, device, tight_expected: bool, decoder_c_path: str,
                    label: str) -> dict[str, Any]:
    """Shared HMS body, used for both default and tight-expected variants."""
    records, meta = hms.collect_records(args, device)
    N = meta['N']

    pi_pct = 75.0
    min_n = 100
    buckets, pi_thr = hms.make_buckets(
        records, pi_q_pct=pi_pct, tight_expected=tight_expected,
        exp_pred_err_max=5.0, unexp_pred_err_min=60.0,
    )
    n_e = buckets['expected']['n']
    n_u = buckets['unexpected']['n']
    n_o = buckets['omission']['n']
    if min(n_e, n_u, n_o) < min_n:
        # Match the script's widening cascade (drop pi pct to 50 if any < 100)
        pi_pct = 50.0
        buckets, pi_thr = hms.make_buckets(
            records, pi_q_pct=pi_pct, tight_expected=tight_expected,
            exp_pred_err_max=5.0, unexp_pred_err_min=60.0,
        )

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    decA = load_decoder_a(ckpt, N, torch.device('cpu'))
    decC = load_decoder_c(decoder_c_path, N, torch.device('cpu'))

    out: dict[str, Any] = {
        'strategy': label,
        'native_decoder': 'A',
        'tight_expected': bool(tight_expected),
        'pi_pct_used': pi_pct,
    }
    for name in ['expected', 'unexpected', 'omission']:
        b = buckets[name]
        # Pass A buckets have 'r_probe'; omission has 'r_target'
        r_key = 'r_target' if b.get('is_omission') else 'r_probe'
        r = b[r_key]
        true_ch = b['probe_true_ch']
        a_acc, n = acc_one(decA, r, true_ch)
        c_acc, _ = acc_one(decC, r, true_ch)
        out[f'{name}_n'] = n
        out[f'{name}_decA_acc'] = a_acc
        out[f'{name}_decC_acc'] = c_acc
        print(f"[{label}] {name:<11s} n={n}  decA={a_acc}  decC={c_acc}",
              flush=True)
    out['n_ex'] = out['expected_n']
    out['n_unex'] = out['unexpected_n']
    out['decA_acc_ex'] = out['expected_decA_acc']
    out['decA_acc_unex'] = out['unexpected_decA_acc']
    out['decC_acc_ex'] = out['expected_decC_acc']
    out['decC_acc_unex'] = out['unexpected_decC_acc']
    out['decA_delta'] = out['decA_acc_ex'] - out['decA_acc_unex']
    out['decC_delta'] = out['decC_acc_ex'] - out['decC_acc_unex']
    return out


def eval_HMS(checkpoint: str, config: str, decoder_c_path: str,
             device: torch.device, n_batches: int = 40, seed: int = 42
             ) -> dict[str, Any]:
    # collect_records reads: target_idx (None=last), jump_min_deg (75.0 default
    # in the script's argparse), step_tol (1.0 default). We must set them.
    args = _Args(config=config, checkpoint=checkpoint, device=str(device),
                 n_batches=n_batches, rng_seed=seed,
                 target_idx=None, jump_min_deg=75.0, step_tol=1.0)
    return _eval_HMS_inner(args, device, tight_expected=False,
                           decoder_c_path=decoder_c_path,
                           label='HMS (matched_hmm_ring_sequence)')


def eval_HMS_T(checkpoint: str, config: str, decoder_c_path: str,
               device: torch.device, n_batches: int = 40, seed: int = 42
               ) -> dict[str, Any]:
    args = _Args(config=config, checkpoint=checkpoint, device=str(device),
                 n_batches=n_batches, rng_seed=seed,
                 target_idx=None, jump_min_deg=75.0, step_tol=1.0)
    return _eval_HMS_inner(args, device, tight_expected=True,
                           decoder_c_path=decoder_c_path,
                           label='HMS-T (matched_hmm_ring_sequence --tight-expected)')


def eval_P3P(checkpoint: str, config: str, decoder_c_path: str,
             device: torch.device, n_batches: int = 40, seed: int = 42
             ) -> dict[str, Any]:
    """P3P: same widening cascade as the script. Pass A=Expected, B=Unexpected,
    C=Omission. Decoder applied per-pass on r_probe_<pass>.
    """
    # P3P collect_records reads: target_idx (None=last), step_tol (1.0).
    # flip_ccw is read in main(), not collect_records, but kept harmless.
    args = _Args(config=config, checkpoint=checkpoint, device=str(device),
                 n_batches=n_batches, rng_seed=seed, flip_ccw=True,
                 target_idx=None, step_tol=1.0)
    records, meta = p3p.collect_records(args, device)
    N = meta['N']

    exp_max = 5.0
    pi_pct = 75.0
    min_n = 100
    cascade: list[str] = []
    mask, pi_thr = p3p.apply_filter(records, pi_q_pct=pi_pct,
                                    exp_pred_err_max=exp_max)
    n_q = int(mask.sum())
    if n_q < min_n:
        exp_max = 10.0
        cascade.append(f"widen exp_pred_err 5°→10°")
        mask, pi_thr = p3p.apply_filter(records, pi_q_pct=pi_pct,
                                        exp_pred_err_max=exp_max)
        n_q = int(mask.sum())
    if n_q < min_n:
        exp_max = 15.0
        cascade.append(f"widen exp_pred_err 10°→15°")
        mask, pi_thr = p3p.apply_filter(records, pi_q_pct=pi_pct,
                                        exp_pred_err_max=exp_max)
        n_q = int(mask.sum())

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    decA = load_decoder_a(ckpt, N, torch.device('cpu'))
    decC = load_decoder_c(decoder_c_path, N, torch.device('cpu'))

    # Per-pass: Pass A vs target_true_ch (Expected), Pass B vs unexp_probe_ch
    # (Unexpected), Pass C vs target_true_ch (Omission, n/a interp).
    out: dict[str, Any] = {
        'strategy': 'P3P (matched_probe_3pass)',
        'native_decoder': 'A',
        'pi_pct_used': pi_pct,
        'exp_pred_err_max_used': exp_max,
        'widening_cascade': cascade,
        'n_qualifying': int(n_q),
    }
    pass_specs = [
        ('expected', 'r_probe_A', 'target_true_ch'),
        ('unexpected', 'r_probe_B', 'unexp_probe_ch'),
        ('omission', 'r_probe_C', 'target_true_ch'),
    ]
    for name, r_key, t_key in pass_specs:
        r = records[r_key][mask]
        true_ch = records[t_key][mask]
        a_acc, n = acc_one(decA, r, true_ch)
        c_acc, _ = acc_one(decC, r, true_ch)
        out[f'{name}_n'] = n
        out[f'{name}_decA_acc'] = a_acc
        out[f'{name}_decC_acc'] = c_acc
        print(f"[P3P] {name:<11s} n={n}  decA={a_acc}  decC={c_acc}",
              flush=True)
    out['n_ex'] = out['expected_n']
    out['n_unex'] = out['unexpected_n']
    out['decA_acc_ex'] = out['expected_decA_acc']
    out['decA_acc_unex'] = out['unexpected_decA_acc']
    out['decC_acc_ex'] = out['expected_decC_acc']
    out['decC_acc_unex'] = out['unexpected_decC_acc']
    out['decA_delta'] = out['decA_acc_ex'] - out['decA_acc_unex']
    out['decC_delta'] = out['decC_acc_ex'] - out['decC_acc_unex']
    return out


def _vcd_test3_indices(records: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reproduce v2_confidence_dissection.test3 matching algorithm; return
    (exp_global, unexp_global, matching_quality)."""
    pred_err = records['pred_err']
    pi = records['pi_pred_eff']

    exp_idx = np.where(pred_err <= 10.0)[0]
    unexp_idx = np.where(pred_err > 20.0)[0]

    if len(exp_idx) <= len(unexp_idx):
        driver_idx, pool_idx = exp_idx, unexp_idx.copy()
        driver_is_exp = True
    else:
        driver_idx, pool_idx = unexp_idx, exp_idx.copy()
        driver_is_exp = False

    pi_driver = pi[driver_idx]
    pi_pool = pi[pool_idx]

    matched_driver: list[int] = []
    matched_pool: list[int] = []
    available = np.ones(len(pool_idx), dtype=bool)
    sort_order = np.argsort(pi_driver)
    for di in sort_order:
        if not available.any():
            break
        avail_pi = pi_pool[available]
        avail_positions = np.where(available)[0]
        diffs = np.abs(avail_pi - pi_driver[di])
        best_local = np.argmin(diffs)
        best_pool = avail_positions[best_local]
        matched_driver.append(int(di))
        matched_pool.append(int(best_pool))
        available[best_pool] = False

    md = np.array(matched_driver)
    mp = np.array(matched_pool)
    if driver_is_exp:
        exp_global = driver_idx[md]
        unexp_global = pool_idx[mp]
    else:
        unexp_global = driver_idx[md]
        exp_global = pool_idx[mp]

    pi_e = pi[exp_global]; pi_u = pi[unexp_global]
    quality = {
        'driver_is_exp': bool(driver_is_exp),
        'mean_abs_pi_diff': float(np.abs(pi_e - pi_u).mean()),
        'max_abs_pi_diff': float(np.abs(pi_e - pi_u).max()),
        'mean_pi_exp': float(pi_e.mean()),
        'mean_pi_unexp': float(pi_u.mean()),
    }
    return exp_global, unexp_global, quality


def eval_VCD(checkpoint: str, config: str, decoder_c_path: str,
             device: torch.device, n_batches: int = 40, seed: int = 42
             ) -> dict[str, Any]:
    """VCD test3 — pi-matched 1-to-1 pairing."""
    args = _Args(config=config, checkpoint=checkpoint, device=str(device),
                 n_batches=n_batches, rng_seed=seed)
    records, meta = vcd.collect_hmm_pool(args, device)
    N = meta['N']

    exp_global, unexp_global, quality = _vcd_test3_indices(records)
    n_pairs = int(len(exp_global))

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    decA = load_decoder_a(ckpt, N, torch.device('cpu'))
    decC = load_decoder_c(decoder_c_path, N, torch.device('cpu'))

    r_l23_win = records['r_l23_win']
    true_ch = records['true_ch']

    r_e = r_l23_win[exp_global]; t_e = true_ch[exp_global]
    r_u = r_l23_win[unexp_global]; t_u = true_ch[unexp_global]

    a_e, _ = acc_one(decA, r_e, t_e)
    a_u, _ = acc_one(decA, r_u, t_u)
    c_e, _ = acc_one(decC, r_e, t_e)
    c_u, _ = acc_one(decC, r_u, t_u)

    print(f"[VCD] n_pairs={n_pairs}  decA Δ={a_e - a_u:+.4f}  "
          f"decC Δ={c_e - c_u:+.4f}", flush=True)

    return {
        'strategy': 'VCD-test3 (v2_confidence_dissection)',
        'native_decoder': 'A',
        'n_pairs': n_pairs,
        'matching_quality': quality,
        'n_ex': n_pairs, 'n_unex': n_pairs,
        'decA_acc_ex': a_e, 'decA_acc_unex': a_u,
        'decC_acc_ex': c_e, 'decC_acc_unex': c_u,
        'decA_delta': a_e - a_u,
        'decC_delta': c_e - c_u,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', required=True,
                   help='R1+R2 simple_dual emergent_seed42 checkpoint.pt')
    p.add_argument('--config', required=True,
                   help='config/sweep/sweep_rescue_1_2.yaml')
    p.add_argument('--decoder-c-path', default='checkpoints/decoder_c.pt')
    p.add_argument('--output-json', default='results/cross_decoder_r1_2.json')
    p.add_argument('--device', default=None)
    p.add_argument('--n-batches', type=int, default=40)
    p.add_argument('--rng-seed', type=int, default=42)
    p.add_argument('--n-trials-per-N', type=int, default=200)
    p.add_argument('--strategies', nargs='+',
                   default=['NEW', 'M3R', 'HMS', 'HMS-T', 'P3P', 'VCD'])
    p.add_argument('--override-task-cue', action='store_true', default=False,
                   help='Task #18: pin task_state to [1,0] (focused) and '
                        'replace cues with deterministic march-continuation '
                        'bumps for all HMM-driven strategies. Applied via '
                        'HMMSequenceGenerator.generate monkey-patch — affects '
                        'every strategy except NEW (which uses build_trial_batch).')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"[setup] checkpoint={args.checkpoint}", flush=True)
    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] decoder_c_path={args.decoder_c_path}", flush=True)
    print(f"[setup] device={device}  n_batches={args.n_batches}  "
          f"seed={args.rng_seed}  n_trials_per_N={args.n_trials_per_N}",
          flush=True)
    print(f"[setup] strategies={args.strategies}", flush=True)
    print(f"[setup] override_task_cue={args.override_task_cue}", flush=True)

    if args.override_task_cue:
        model_cfg, _, _ = load_config(args.config)
        install_input_overrides(period=float(model_cfg.orientation_range),
                                n_ori=int(model_cfg.n_orientations))

    started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    t0 = time.time()
    out: dict[str, Any] = {
        'label': ('Task #18 cross-decoder eval (modified task+cue) — '
                  'R1+R2 simple_dual emergent_seed42'
                  if args.override_task_cue else
                  'Task #17 cross-decoder eval — R1+R2 simple_dual emergent_seed42'),
        'override_task_cue': bool(args.override_task_cue),
        'checkpoint': args.checkpoint,
        'config': args.config,
        'decoder_c_path': args.decoder_c_path,
        'device': str(device),
        'n_batches': int(args.n_batches),
        'rng_seed': int(args.rng_seed),
        'n_trials_per_N': int(args.n_trials_per_N),
        'started_at': started_at,
        'results': {},
    }

    # NEW eval — needs its own forward pattern (per-N paired ex/unex)
    if 'NEW' in args.strategies:
        print("\n=== NEW eval ===", flush=True)
        out['results']['NEW'] = eval_NEW(
            args.checkpoint, args.config, args.decoder_c_path, device,
            n_trials_per_N=args.n_trials_per_N, seed_base=args.rng_seed,
        )

    if 'M3R' in args.strategies:
        print("\n=== M3R ===", flush=True)
        out['results']['M3R'] = eval_M3R(
            args.checkpoint, args.config, args.decoder_c_path, device,
            n_batches=args.n_batches, seed=args.rng_seed,
        )

    if 'HMS' in args.strategies:
        print("\n=== HMS ===", flush=True)
        out['results']['HMS'] = eval_HMS(
            args.checkpoint, args.config, args.decoder_c_path, device,
            n_batches=args.n_batches, seed=args.rng_seed,
        )

    if 'HMS-T' in args.strategies:
        print("\n=== HMS-T ===", flush=True)
        out['results']['HMS-T'] = eval_HMS_T(
            args.checkpoint, args.config, args.decoder_c_path, device,
            n_batches=args.n_batches, seed=args.rng_seed,
        )

    if 'P3P' in args.strategies:
        print("\n=== P3P ===", flush=True)
        out['results']['P3P'] = eval_P3P(
            args.checkpoint, args.config, args.decoder_c_path, device,
            n_batches=args.n_batches, seed=args.rng_seed,
        )

    if 'VCD' in args.strategies:
        print("\n=== VCD test3 ===", flush=True)
        out['results']['VCD'] = eval_VCD(
            args.checkpoint, args.config, args.decoder_c_path, device,
            n_batches=args.n_batches, seed=args.rng_seed,
        )

    out['elapsed_s'] = float(time.time() - t0)
    out['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[json] wrote {args.output_json}  ({out['elapsed_s']:.1f}s)",
          flush=True)

    # --- Summary table ---
    print("\n" + "=" * 78, flush=True)
    print(" Strategy            |   n_ex |  n_unex |  decA Δ  |  decC Δ  | native",
          flush=True)
    print("-" * 78, flush=True)
    for key in ['NEW', 'M3R', 'HMS', 'HMS-T', 'P3P', 'VCD']:
        if key not in out['results']:
            continue
        r = out['results'][key]
        a = r.get('decA_delta')
        c = r.get('decC_delta')
        a_s = f"{a:+.4f}" if a is not None else "  n/a "
        c_s = f"{c:+.4f}" if c is not None else "  n/a "
        ne = r.get('n_ex', '-'); nu = r.get('n_unex', '-')
        print(f" {key:<19s} | {str(ne):>6} | {str(nu):>7} | {a_s:>8} | {c_s:>8} "
              f"| {r.get('native_decoder')}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
