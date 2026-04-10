#!/usr/bin/env python3
"""Compare L2/3 activity for EXPECTED vs UNEXPECTED stimuli (both feedback ON).

KEY DESIGN: Uses V2's PRE-STIMULUS prediction (from last ISI timestep)
to classify the NEXT stimulus as expected or unexpected. This is the
real predictive coding test — does the network suppress responses when
the prediction matches vs mismatches the incoming stimulus?

Expected: V2 prediction from ISI within ±10° of actual next stimulus
Unexpected: V2 prediction from ISI >20° away from actual next stimulus

Then measures L2/3 and L4 activity during the ON period of that stimulus.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


def circular_distance(a, b, period=180.0):
    """Absolute circular distance."""
    d = torch.abs(a - b)
    return torch.min(d, period - d)


def compute_fwhm(response, period=180.0):
    """Compute FWHM of a 1D circular response profile."""
    N = response.shape[0]
    step = period / N
    peak = response.max()
    if peak < 1e-8:
        return 0.0
    half_max = peak / 2.0
    above = (response >= half_max).float()
    return above.sum().item() * step


def parse_args():
    parser = argparse.ArgumentParser(description="Expected vs unexpected stimulus analysis")
    parser.add_argument("--config", type=str, default="config/exp_v2_ei_highenergy.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str,
                        default="results/iter/v2_ei_highenergy/center_surround_seed42/checkpoint.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--label", type=str, default="",
                        help="Label for output header")
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="RNG seed for stimulus generation")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Config and model ---
    config_path = args.config
    ckpt_path = args.checkpoint

    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['model_state'])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N  # 5 deg

    n_batches = 10
    seq_length = train_cfg.seq_length  # 25
    batch_size = train_cfg.batch_size  # 32
    steps_on = train_cfg.steps_on      # 12
    steps_isi = train_cfg.steps_isi    # 4
    steps_per = steps_on + steps_isi   # 16

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

    label = args.label or f"Checkpoint: {ckpt_path}"
    print("=" * 70)
    print(f"EXPECTED vs UNEXPECTED — PRE-STIMULUS PREDICTION")
    print(f"{label}")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"feedback_scale: {net.feedback_scale.item():.3f}")
    print(f"steps_on={steps_on}, steps_isi={steps_isi}, seq_length={seq_length}")
    print()
    print("Method: V2 prediction taken at LAST ISI timestep (pre-stimulus)")
    print("        Stimulus response measured at late ON timestep (t=9 of 12)")
    print("        Expected: prediction error <= 10 deg")
    print("        Unexpected: prediction error > 20 deg")
    print()

    # Collect per-presentation metrics
    metrics_expected = {'l23_total': [], 'l23_peak': [], 'l23_fwhm': [],
                        'l4_total': [], 'l4_peak': [], 'pred_error': [],
                        'center_exc_total': []}
    metrics_unexpected = {'l23_total': [], 'l23_peak': [], 'l23_fwhm': [],
                          'l4_total': [], 'l4_peak': [], 'pred_error': [],
                          'center_exc_total': []}
    # Also track "middle zone" (10-20 deg) for completeness
    metrics_middle = {'l23_total': [], 'l23_peak': [], 'l23_fwhm': [],
                      'l4_total': [], 'l4_peak': [], 'pred_error': [],
                      'center_exc_total': []}

    rng = torch.Generator().manual_seed(args.rng_seed)

    with torch.no_grad():
        for batch_i in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, true_thetas, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net.forward(packed)
            r_l4_all = aux['r_l4_all']
            q_pred_all = aux['q_pred_all']
            center_exc_all = aux['center_exc_all']

            # For each presentation >= 1 (need prior ISI prediction):
            readout_offset = min(9, steps_on - 1)  # late ON timestep

            for pres_i in range(1, seq_length):  # skip first (no ISI prediction)
                # ISI prediction: last ISI timestep of PREVIOUS presentation
                # ISI for pres_i-1 starts at (pres_i-1)*steps_per + steps_on
                # Last ISI timestep: pres_i * steps_per - 1
                t_isi_last = pres_i * steps_per - 1

                # ON readout: late timestep during pres_i's ON period
                t_on_readout = pres_i * steps_per + readout_offset

                # V2 prediction from ISI
                q_pred_isi = q_pred_all[:, t_isi_last, :]  # [B, N]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)   # [B]
                pred_ori = pred_peak_idx.float() * step_deg  # [B]

                # Actual stimulus orientation for this presentation
                actual_ori = metadata.orientations[:, pres_i]  # [B]
                is_amb = metadata.is_ambiguous[:, pres_i]      # [B]

                # Prediction error
                pred_error = circular_distance(pred_ori, actual_ori, period)  # [B]

                # ON period responses
                r_l23_on = r_l23_all[:, t_on_readout, :]       # [B, N]
                r_l4_on = r_l4_all[:, t_on_readout, :]         # [B, N]
                ce_on = center_exc_all[:, t_on_readout, :]     # [B, N]

                # Classify (exclude ambiguous)
                expected_mask = (pred_error <= 10.0) & (~is_amb)
                unexpected_mask = (pred_error > 20.0) & (~is_amb)
                middle_mask = (pred_error > 10.0) & (pred_error <= 20.0) & (~is_amb)

                for b in range(batch_size):
                    m = {
                        'l23_total': r_l23_on[b].sum().item(),
                        'l23_peak': r_l23_on[b].max().item(),
                        'l23_fwhm': compute_fwhm(r_l23_on[b], period),
                        'l4_total': r_l4_on[b].sum().item(),
                        'l4_peak': r_l4_on[b].max().item(),
                        'pred_error': pred_error[b].item(),
                        'center_exc_total': ce_on[b].sum().item(),
                    }

                    if expected_mask[b]:
                        for k, v in m.items():
                            metrics_expected[k].append(v)
                    elif unexpected_mask[b]:
                        for k, v in m.items():
                            metrics_unexpected[k].append(v)
                    elif middle_mask[b]:
                        for k, v in m.items():
                            metrics_middle[k].append(v)

            print(f"  Batch {batch_i+1}/{n_batches}: "
                  f"expected={len(metrics_expected['l23_total'])}, "
                  f"unexpected={len(metrics_unexpected['l23_total'])}, "
                  f"middle={len(metrics_middle['l23_total'])}")

    # --- Report ---
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for label, data in [("EXPECTED (pred err <= 10°)", metrics_expected),
                         ("MIDDLE (10° < pred err <= 20°)", metrics_middle),
                         ("UNEXPECTED (pred err > 20°)", metrics_unexpected)]:
        n = len(data['l23_total'])
        if n == 0:
            print(f"\n{label}: no presentations")
            continue
        print(f"\n{label} (n={n}):")
        for k in ['l23_total', 'l23_peak', 'l23_fwhm', 'l4_total', 'l4_peak',
                   'pred_error', 'center_exc_total']:
            vals = np.array(data[k])
            print(f"  {k:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"median={np.median(vals):.4f}")

    # --- Key comparison ---
    print()
    print("=" * 70)
    print("KEY COMPARISON: Expected vs Unexpected")
    print("=" * 70)

    n_exp = len(metrics_expected['l23_total'])
    n_unexp = len(metrics_unexpected['l23_total'])

    if n_exp > 0 and n_unexp > 0:
        from scipy import stats

        for metric in ['l23_total', 'l23_peak', 'l23_fwhm', 'l4_total', 'l4_peak', 'center_exc_total']:
            exp_vals = np.array(metrics_expected[metric])
            unexp_vals = np.array(metrics_unexpected[metric])

            exp_mean = exp_vals.mean()
            unexp_mean = unexp_vals.mean()
            diff = unexp_mean - exp_mean
            pct = (diff / exp_mean * 100) if abs(exp_mean) > 1e-8 else float('inf')

            t_stat, p_val = stats.ttest_ind(exp_vals, unexp_vals, equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            direction = "UNEXP > EXP" if diff > 0 else "EXP > UNEXP" if diff < 0 else "EQUAL"

            print(f"  {metric:20s}: Exp={exp_mean:.4f}, Unexp={unexp_mean:.4f}, "
                  f"Δ={diff:+.4f} ({pct:+.1f}%), t={t_stat:.2f}, p={p_val:.2e} {sig} → {direction}")

        # Interpretation
        print()
        l23_exp = np.mean(metrics_expected['l23_total'])
        l23_unexp = np.mean(metrics_unexpected['l23_total'])
        l4_exp = np.mean(metrics_expected['l4_total'])
        l4_unexp = np.mean(metrics_unexpected['l4_total'])

        print("INTERPRETATION:")
        if l23_exp < l23_unexp:
            supp = (l23_unexp - l23_exp) / l23_unexp * 100
            print(f"  L2/3 total: LOWER for expected → PREDICTIVE SUPPRESSION ({supp:.1f}% reduction)")
        elif l23_exp > l23_unexp:
            enh = (l23_exp - l23_unexp) / l23_unexp * 100
            print(f"  L2/3 total: HIGHER for expected → PREDICTIVE ENHANCEMENT ({enh:.1f}% increase)")
        else:
            print(f"  L2/3 total: no difference")

        l4_diff = abs(l4_exp - l4_unexp) / max(l4_exp, l4_unexp, 1e-8) * 100
        print(f"  L4 control: |diff| = {l4_diff:.1f}% {'(small → L4 NOT affected)' if l4_diff < 5 else '(large → confound: L4 differs too)'}")

    else:
        print("  Insufficient data")
        if n_unexp == 0:
            print("  WARNING: V2 prediction is always correct (0 unexpected).")
            print("  This means V2 has learned to predict the HMM perfectly.")

    # --- Binned analysis ---
    print()
    print("=" * 70)
    print("L2/3 TOTAL BY PREDICTION ERROR BIN")
    print("=" * 70)

    all_errors = (np.array(metrics_expected['pred_error'] +
                           metrics_middle['pred_error'] +
                           metrics_unexpected['pred_error']))
    all_l23 = (np.array(metrics_expected['l23_total'] +
                        metrics_middle['l23_total'] +
                        metrics_unexpected['l23_total']))
    all_l4 = (np.array(metrics_expected['l4_total'] +
                       metrics_middle['l4_total'] +
                       metrics_unexpected['l4_total']))
    all_ce = (np.array(metrics_expected['center_exc_total'] +
                       metrics_middle['center_exc_total'] +
                       metrics_unexpected['center_exc_total']))

    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 90)]
    print(f"  {'Range':>12s}  {'n':>6s}  {'L23 mean':>10s}  {'L4 mean':>10s}  {'CE mean':>10s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for lo, hi in bins:
        mask = (all_errors >= lo) & (all_errors < hi)
        n = mask.sum()
        if n > 0:
            print(f"  {lo:>3d}-{hi:>3d} deg   {n:>6d}  {all_l23[mask].mean():>10.4f}  "
                  f"{all_l4[mask].mean():>10.4f}  {all_ce[mask].mean():>10.4f}")
        else:
            print(f"  {lo:>3d}-{hi:>3d} deg   {0:>6d}  {'---':>10s}  {'---':>10s}  {'---':>10s}")

    # --- CONTROL: feedback_scale=0 ---
    print()
    print("=" * 70)
    print("CONTROL: feedback_scale=0 (feedback OFF)")
    print("=" * 70)

    net_off = LaminarV1V2Network(model_cfg)
    net_off.load_state_dict(ckpt['model_state'])
    net_off.eval()
    net_off.oracle_mode = False
    net_off.feedback_scale.fill_(0.0)

    off_expected_l23 = []
    off_unexpected_l23 = []
    off_expected_l4 = []
    off_unexpected_l4 = []

    rng2 = torch.Generator().manual_seed(args.rng_seed)  # same seed for identical stimuli

    with torch.no_grad():
        for batch_i in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng2)
            stim_seq, cue_seq, ts_seq, true_thetas, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            packed = net_off.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23_all, _, aux = net_off.forward(packed)
            r_l4_all = aux['r_l4_all']
            q_pred_all = aux['q_pred_all']

            readout_offset = min(9, steps_on - 1)
            for pres_i in range(1, seq_length):
                t_isi_last = pres_i * steps_per - 1
                t_on_readout = pres_i * steps_per + readout_offset

                q_pred_isi = q_pred_all[:, t_isi_last, :]
                pred_peak_idx = q_pred_isi.argmax(dim=-1)
                pred_ori = pred_peak_idx.float() * step_deg
                actual_ori = metadata.orientations[:, pres_i]
                is_amb = metadata.is_ambiguous[:, pres_i]
                pred_error = circular_distance(pred_ori, actual_ori, period)

                expected_mask = (pred_error <= 10.0) & (~is_amb)
                unexpected_mask = (pred_error > 20.0) & (~is_amb)

                for b in range(batch_size):
                    l23_t = r_l23_all[b, t_on_readout].sum().item()
                    l4_t = r_l4_all[b, t_on_readout].sum().item()
                    if expected_mask[b]:
                        off_expected_l23.append(l23_t)
                        off_expected_l4.append(l4_t)
                    elif unexpected_mask[b]:
                        off_unexpected_l23.append(l23_t)
                        off_unexpected_l4.append(l4_t)

    if len(off_expected_l23) > 0 and len(off_unexpected_l23) > 0:
        from scipy import stats
        exp_m = np.mean(off_expected_l23)
        unexp_m = np.mean(off_unexpected_l23)
        t_off, p_off = stats.ttest_ind(off_expected_l23, off_unexpected_l23, equal_var=False)

        print(f"  FB OFF: Expected L23={exp_m:.4f}, Unexpected L23={unexp_m:.4f}, "
              f"Δ={unexp_m-exp_m:+.4f}, t={t_off:.2f}, p={p_off:.2e}")
        print(f"  FB OFF: Expected L4={np.mean(off_expected_l4):.4f}, "
              f"Unexpected L4={np.mean(off_unexpected_l4):.4f}")

        # Compute feedback contribution
        on_exp = np.mean(metrics_expected['l23_total']) if metrics_expected['l23_total'] else 0
        on_unexp = np.mean(metrics_unexpected['l23_total']) if metrics_unexpected['l23_total'] else 0
        on_diff = on_unexp - on_exp
        off_diff = unexp_m - exp_m
        fb_contribution = on_diff - off_diff

        print(f"\n  FB-ON gap  (unexp - exp): {on_diff:+.4f}")
        print(f"  FB-OFF gap (unexp - exp): {off_diff:+.4f}")
        print(f"  Feedback contribution:     {fb_contribution:+.4f}")

        if fb_contribution > 0:
            print("  → Feedback WIDENS the gap (enhances predictive suppression)")
        elif fb_contribution < 0:
            print("  → Feedback NARROWS the gap (opposes predictive coding)")
        else:
            print("  → Feedback has no differential effect")
    else:
        print(f"  FB OFF: expected={len(off_expected_l23)}, unexpected={len(off_unexpected_l23)}")
        print("  Insufficient data for comparison")

    # --- Additional diagnostic: V2 prediction quality during ISI ---
    print()
    print("=" * 70)
    print("V2 PREDICTION QUALITY (ISI predictions)")
    print("=" * 70)
    all_pred_errors_flat = np.array(
        metrics_expected['pred_error'] + metrics_middle['pred_error'] + metrics_unexpected['pred_error']
    )
    total_pres = len(all_pred_errors_flat)
    within_5 = (all_pred_errors_flat <= 5.0).sum()
    within_10 = (all_pred_errors_flat <= 10.0).sum()
    within_15 = (all_pred_errors_flat <= 15.0).sum()
    within_20 = (all_pred_errors_flat <= 20.0).sum()
    beyond_20 = (all_pred_errors_flat > 20.0).sum()

    print(f"  Total non-ambiguous presentations: {total_pres}")
    print(f"  Within 5°:  {within_5:>5d} ({within_5/total_pres*100:.1f}%)")
    print(f"  Within 10°: {within_10:>5d} ({within_10/total_pres*100:.1f}%)")
    print(f"  Within 15°: {within_15:>5d} ({within_15/total_pres*100:.1f}%)")
    print(f"  Within 20°: {within_20:>5d} ({within_20/total_pres*100:.1f}%)")
    print(f"  Beyond 20°: {beyond_20:>5d} ({beyond_20/total_pres*100:.1f}%)")
    print(f"  Mean prediction error: {all_pred_errors_flat.mean():.2f}°")
    print(f"  Median prediction error: {np.median(all_pred_errors_flat):.2f}°")

    print("\nDone.")


if __name__ == "__main__":
    main()
