# Representation Analysis Results (Phase G2)

## Purpose

After external code review, the original sharpening/dampening claims were
characterized only by suppression index (SI) curves at the stimulus channel.
The reviewer correctly noted that the sharpening vs. dampening distinction
should live or die on **representational geometry** — tuning width, gain,
and discriminability — not on mean suppression alone.

This analysis computes proper representational metrics on the two hardened
checkpoints and revisits the earlier claims.

## Checkpoints analyzed

- **Dampening**: `results/hardening/deviance_v2_s42/center_surround_seed42/checkpoint.pt`
  (hardened deviance condition: L2/3 sensory loss off, SOM-only, delta-SOM,
  frozen decoder, 5000 stage-2 steps, oracle V2 at pi=1)
- **P4 "sharpening"**: `results/hardening/p4_v2_s42/center_surround_seed42/checkpoint.pt`
  (hardened fine-discrimination condition: L2/3 sensory loss on, steps_on=6,
  oracle V2 at pi=3, same seed)

Analysis script: `scripts/analyze_representation.py`. Evaluation pi=5.0 for
both (matches earlier SI comparisons).

## Sanity checks (verified)

1. **Ablation mechanism is clean**. Calling `net.feedback(q, pi)` under a
   context manager that zeroes both `alpha_inh` and `som_baseline` gives
   max-abs drive = 0.0 exactly (not "small"). Feedback is truly off.
2. **Hand-computed FWHM matches script output** for dampening ON at stim=ora=90°:
   linear-interp crossings at d = -12.30° and +13.32°, total FWHM = 25.62°.
3. **Both checkpoints confirmed to contain nonzero alpha_inh** (dampening: 2.147,
   P4: 1.511) and the expected kernel shapes (dampening peaked at center,
   P4 DoG).

## Metrics and results

### 1. Peak gain at preferred channel (ON / OFF ratio)

The L2/3 response at channel 18 (pref=90°) when stimulus=90° and oracle=90°.
Expectation: dampening reduces peak (gain reduction), sharpening should preserve
peak (spare center).

| | Dampening | P4 |
|---|---|---|
| Peak (fb off) | 0.2887 | 0.2887 |
| Peak (fb on) | 0.1866 | 0.2887 |
| Ratio ON/OFF | **0.6466** | **1.0000** |

Dampening kills 35% of the peak; P4 leaves it exactly unchanged. This part
matches the original prediction.

### 2. Population-bump FWHM at fixed stimulus

Read all 36 L2/3 channels when stim=90° and oracle=90°. Compute FWHM of the
resulting bump across the population. Sharpening should narrow this; dampening
was expected to leave it unchanged.

| stim, oracle | Dampening OFF | Dampening ON | ΔFWHM | P4 OFF | P4 ON | ΔFWHM |
|---|---|---|---|---|---|---|
| (90°, 90°) | 26.73° | 25.62° | **-1.11°** | 26.72° | 26.69° | -0.03° |
| (120°, 90°) | 26.73° | 25.13° | -1.60° | 26.72° | 26.46° | -0.26° |
| (90°, 120°) | 26.73° | 24.73° | -2.00° | 26.72° | 26.38° | -0.33° |

Dampening narrows the population bump by ~4%; P4 does not narrow it at all
(-0.03° is noise).

### 3. Is the dampening narrowing a rectified-softplus artifact?

If the narrowing were an artifact of rectifier-threshold effects, the
pre-rectifier drive (before passing through rectified_softplus) would stay
wide and only the post-rectifier rate would narrow.

| | DRIVE FWHM OFF | DRIVE FWHM ON | Δ DRIVE | RATE FWHM OFF | RATE FWHM ON | Δ RATE |
|---|---|---|---|---|---|---|
| Dampening | 29.62° | 28.21° | **-1.41°** | 26.83° | 25.67° | -1.16° |
| P4 | 29.61° | 30.75° | **+1.14°** | 26.82° | 26.79° | -0.03° |

Dampening's pre-rect drive narrows by 1.41° — *more* than the rate narrows
(1.16°). This rules out the rectifier artifact: the kernel is genuinely
reshaping the drive.

P4's pre-rect drive actually *widens* by 1.14° (the kernel pushes drives at
±30° more negative), but the rectifier clips them to zero, so the post-rect
rate is unchanged.

### 4. Tuning curves at channels at different offsets from oracle

For each channel at a given offset from oracle=90°, compute its tuning curve
(response vs stimulus orientation), feedback on vs off.

**Dampening:**

| Offset | Channel | Peak OFF | Peak ON | Ratio | FWHM OFF | FWHM ON | ΔFWHM |
|---|---|---|---|---|---|---|---|
| -30° | 12 (60°) | 0.2887 | 0.2798 | 0.9694 | 26.73° | 26.31° | -0.42° |
| -25° | 13 (65°) | 0.2887 | 0.2670 | 0.9250 | 26.73° | 25.65° | -1.08° |
| -20° | 14 (70°) | 0.2887 | 0.2493 | 0.8635 | 26.73° | 24.69° | -2.04° |
| -10° | 16 (80°) | 0.2887 | 0.2064 | 0.7150 | 26.73° | 22.21° | -4.52° |
| **0°** | **18 (90°)** | 0.2887 | 0.1866 | **0.6466** | 26.73° | 20.92° | -5.81° |
| +10° | 20 (100°) | 0.2887 | 0.2156 | 0.7471 | 26.73° | 22.77° | -3.96° |
| +20° | 22 (110°) | 0.2887 | 0.2649 | 0.9177 | 26.73° | 25.55° | -1.18° |
| +25° | 23 (115°) | 0.2887 | 0.2837 | 0.9830 | 26.73° | 26.54° | -0.19° |
| +30° | 24 (120°) | 0.2887 | 0.2863 | 0.9918 | 26.73° | 26.64° | -0.09° |

Maximum gain drop at the preferred-at-expected channel (0°), falling off
monotonically with distance from oracle. Classic dampening shape.

**P4 "sharpening":**

| Offset | Channel | Peak OFF | Peak ON | Ratio | FWHM OFF | FWHM ON | ΔFWHM |
|---|---|---|---|---|---|---|---|
| -30° | 12 (60°) | 0.2887 | 0.2719 | 0.9418 | 26.72° | 25.82° | -0.89° |
| -25° | 13 (65°) | 0.2887 | 0.2799 | 0.9695 | 26.72° | 26.26° | -0.46° |
| -20° | 14 (70°) | 0.2887 | 0.2871 | 0.9942 | 26.72° | 26.65° | -0.07° |
| -10° | 16 (80°) | 0.2887 | 0.2884 | 0.9988 | 26.72° | 26.71° | -0.01° |
| **0°** | **18 (90°)** | 0.2887 | 0.2887 | **1.0000** | 26.72° | 26.73° | +0.01° |
| +10° | 20 (100°) | 0.2887 | 0.2886 | 0.9995 | 26.72° | 26.72° | +0.00° |
| +20° | 22 (110°) | 0.2887 | 0.2879 | 0.9972 | 26.72° | 26.68° | -0.04° |
| +25° | 23 (115°) | 0.2887 | 0.2863 | 0.9916 | 26.72° | 26.60° | -0.12° |
| +30° | 24 (120°) | 0.2887 | 0.2794 | 0.9678 | 26.72° | 26.22° | -0.50° |

P4's effect is visible only at ±30° flanks (~3% gain drop) and essentially
zero at the preferred channel. Inverted U-shape compared to dampening.

### 5. Energy reduction by distance bin

L2/3 activity summed over channels, split into distance-from-oracle bins.

| Bin | Dampening reduction | P4 reduction |
|---|---|---|
| Total | +10.0% | +6.3% |
| Expected (\|d\| ≤ 10°, 5 channels) | **+47.2%** | +0.0% |
| Surround (10° < \|d\| ≤ 45°, 14 channels) | +8.9% | +6.8% |
| Far (\|d\| > 45°, 17 channels) | -0.0% | +7.8% |

Dampening's energy reduction is concentrated at expected channels, as
expected for a peak-centered kernel. P4 reduces energy approximately
uniformly across surround and far channels (6.8% vs 7.8%), with zero effect
at expected — this is broad non-specific suppression, not targeted surround
suppression.

### 6. Fine-discrimination probe

Linear classifier trained on noisy L2/3 readouts distinguishes stimuli at
90° vs 90°+δ. Two channel subsets (narrow ±15° and wide ±30°), three
readout-noise levels, three δ values.

Representative cells (σ=0.30, δ=10°, out of saturation):

| | Dampening OFF | Dampening ON | ΔAcc | P4 OFF | P4 ON | ΔAcc |
|---|---|---|---|---|---|---|
| Narrow subset | 0.700 | 0.680 | -0.020 | 0.700 | 0.700 | +0.000 |
| Wide subset | 0.745 | 0.740 | -0.005 | 0.745 | 0.750 | +0.005 |

All differences are within noise (200 test trials → SE ≈ 3%). No meaningful
discrimination improvement from feedback in either condition.

### 7. Rotational invariance check for dampening

The dampening profile was verified at multiple oracle orientations to confirm
it's a genuine property of the learned kernel, not an artifact of measuring
at one orientation.

SI (suppression index at the stimulus channel) across different oracle
orientations at pi=5:

| Oracle | 0° | 10° | 20° | 30° | 45° |
|---|---|---|---|---|---|
| 0° | +38.0% | +25.6% | +6.5% | +0.7% | +0.0% |
| 45° | +38.0% | +25.6% | +6.5% | +0.7% | +0.0% |
| 90° | +38.0% | +25.6% | +6.5% | +0.7% | +0.0% |
| 135° | +38.0% | +25.6% | +6.5% | +0.7% | +0.0% |

On-grid orientations (aligned with channel centers) give identical results
to 4 decimal places. Off-grid orientations (22.5°, 67.5°, etc.) differ by
±0.1% due to 5° channel discretization (grid aliasing, not a real
asymmetry). Dampening is rotationally invariant.

## Summary: what the metrics actually show

**Dampening** (hardened deviance checkpoint):
- Peak gain at expected: −35% (solid)
- Energy at expected channels: −47% (solid)
- Population bump FWHM: −4% narrowing (small but real, not a rectifier artifact)
- Tuning curve of preferred-at-expected channel: gain drops 35%, FWHM narrows ~22%
- Effect is rotationally invariant across all tested prediction orientations
- Effect falls off monotonically with angular distance from the oracle peak

Dampening in this model is a real representational effect: gain reduction
concentrated at the predicted orientation, with a small but measurable
additional reshaping of the tuning curve.

**"Sharpening"** (hardened P4 checkpoint):
- Peak gain at preferred: 0 change
- Population bump FWHM: 0 narrowing
- Energy at expected: 0 change
- Only effect: ~3% gain drop at flanks at ±30° from oracle (where FF input
  is already near the rectifier threshold), and a ~7% broad energy reduction
  spread roughly equally across surround and far channels
- Fine discrimination: no improvement within measurement noise

The P4 "sharpening" claim does **not** hold up under proper representational
metrics. What P4 actually does is push drive values at extreme flanks
(±30°, where FF input is tiny) slightly below the rectifier threshold,
killing already-near-zero responses. This is a tiny edge effect, not a real
sharpening of the population code. The DoG-shaped kernel learned by P4 does
produce ~4% SI at flanks in the original per-stimulus-channel measurement,
but this does not translate into narrower tuning, improved discrimination,
or any other geometric sharpening signature.

## Corrections to previous summary

The HARDENING_RESULTS.md and earlier summaries described P4 as a "sharpening
regime" based on the DoG kernel shape and flank-suppression SI curves. The
representational metrics show this was an overclaim. The correct description
is:

- P4's learned kernel has a DoG (difference-of-Gaussians) shape.
- This produces a small (~3%) suppression of responses at the extreme
  flanking channels (±25-30° from the expected orientation).
- It does not narrow the population bump, does not narrow any tuning curve
  at meaningfully away from the flank edge, and does not improve
  discrimination.
- Therefore it is NOT a representational sharpening regime.

The dampening result, by contrast, holds up across all metrics and survives
the rectifier artifact test.

## Files

- Script: `scripts/analyze_representation.py`
- Invocation:
  ```
  python3 scripts/analyze_representation.py \
    --checkpoint results/hardening/deviance_v2_s42/center_surround_seed42/checkpoint.pt \
    --config config/exp_deviance.yaml --label dampening \
    --checkpoint results/hardening/p4_v2_s42/center_surround_seed42/checkpoint.pt \
    --config config/exp_sharp_p4.yaml --label p4_sharp \
    --device cpu
  ```
