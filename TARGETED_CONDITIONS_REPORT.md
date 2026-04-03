# Targeted Conditions Experiment Report

Three experiments designed to force specific feedback profiles from the emergent operator.
Each run: 5000 Stage 2 steps, 3 seeds (42, 123, 456), lambda_fb=0.0 (see below).

## Critical Bug Fix: L1 Sparsity Kills Alphas During Burn-in

**Problem:** The L1 sparsity penalty (lambda_fb) was active during burn-in when fb_scale=0,
meaning no task gradient reached the alpha weights but the penalty actively drove them to zero.
By step ~400, alphas were permanently dead (0.000) and never recovered.

**Fix applied:**
1. `losses.py`: L1 sparsity penalty now scaled by `fb_scale` (zero during burn-in, ramps with feedback).
2. All experiment configs: `lambda_fb=0.0` for these short (5000-step) runs.

Without this fix, all three experiments produced zero alphas across all seeds.

---

## Experiment A: Prediction Error Readout (Dampening Target)

**Config:** `config/exp_dampening.yaml`
- lambda_sensory = 0.0 (L2/3 does NOT decode stimulus)
- lambda_error = 1.0 (L2/3 must decode prediction error into 12 bins)
- lambda_pred = 0.5, lambda_energy = 0.01

**Hypothesis:** If L2/3 must encode prediction error rather than stimulus identity,
feedback should sculpt a dampening profile (suppress expected, pass unexpected).

### Results

| Seed | a_inh | a_exc | net_center | R_damp | R_sharp | R_cs   | s_acc |
|------|-------|-------|------------|--------|---------|--------|-------|
| 42   | 0.000 | 0.000 | +0.000     | +0.001 | +0.001  | +0.001 | N/A   |
| 123  | 0.000 | 0.000 | -0.000     | +0.001 | +0.001  | +0.001 | N/A   |
| 456  | 0.000 | 0.000 | -0.000     | +0.002 | +0.002  | +0.002 | N/A   |

### Analysis

**FAILED — No feedback learned.**

All 3 seeds produced zero alpha weights. The error readout loss drives the
`error_decoder` head but creates no gradient through the feedback operator because
`lambda_sensory=0` eliminates the gradient path from L2/3 through feedback.

The feedback operator is a dead end in the computation graph when nothing backprops
through L2/3's main sensory pathway. The error decoder sits downstream of L2/3 but
doesn't create gradient that flows back through the feedback → SOM → L2/3 path.

**Conclusion:** To shape feedback, lambda_sensory must be > 0 so gradient flows
through L2/3 and reaches the feedback operator.

**Cross-seed consistency:** Perfect (all zero).

---

## Experiment B: High Energy + Noise (Sharpening Target)

**Config:** `config/exp_sharpening.yaml`
- lambda_energy = 0.5 (50x baseline — strong energy pressure)
- stimulus_noise = 0.15 (moderate Gaussian noise on population-coded stimuli)
- lambda_sensory = 1.0, lambda_pred = 0.5

**Hypothesis:** High energy cost penalizes total firing, so feedback should suppress
broadly (inhibition dominant). Noise degrades the feedforward signal, so feedback
must sharpen tuning to maintain discrimination.

### Results

| Seed | a_inh | a_exc | net_center | R_damp  | R_sharp  | R_cs    | s_acc |
|------|-------|-------|------------|---------|----------|---------|-------|
| 42   | 1.488 | 1.502 | +0.003     | +0.957  | -0.984   | -0.954  | 0.085 |
| 123  | 1.472 | 1.485 | +0.002     | +0.921  | -0.973   | -0.960  | 0.087 |
| 456  | 1.461 | 1.470 | +0.003     | +0.939  | -0.981   | -0.961  | 0.078 |

### Analysis

**Near-null net profile with marginal dampening. NOT sharpening.**

Both K_inh and K_exc grow to nearly identical values (~1.47-1.50), producing a
near-zero net modulation profile. The net_center is +0.003 (excitation exceeds
inhibition by a tiny margin at the expected orientation).

**Important sign note:** R_sharpening is NEGATIVE (-0.98), meaning the net profile
is anti-correlated with the sharpening template. The positive R_dampening (+0.93-0.96)
indicates slight dampening. The classify_profile function uses R^2 (ignoring sign)
and would misleadingly label this as "sharpening" — it is actually slight dampening.

**Interpretation:** High energy cost drives both inhibitory and excitatory channels
equally. The system doesn't differentiate — it grows both K_inh and K_exc symmetrically.
The net effect is negligible (~0.1% of raw kernel magnitude). Energy pressure alone
is insufficient to break the symmetry between inhibition and excitation.

**Cross-seed consistency:** Excellent. All 3 seeds show a_inh/a_exc ratio ~1.01,
net_center ~+0.003.

---

## Experiment C: Detection + Discrimination (Center-Surround Target)

**Config:** `config/exp_centersurround.yaml`
- lambda_detection = 1.0 (binary: "is expected orientation present?")
- lambda_sensory = 1.0 (orientation discrimination)
- stimulus_noise = 0.1 (mild noise)
- lambda_pred = 0.5, lambda_energy = 0.01

**Hypothesis:** The detection task requires knowing whether the expected orientation
is present. Combined with discrimination (which orientation?), feedback should develop
center-surround: enhance expected (excitation at center) while suppressing flanks
(broad inhibition).

### Results

| Seed | a_inh | a_exc | net_center | R_damp  | R_sharp  | R_cs    | s_acc |
|------|-------|-------|------------|---------|----------|---------|-------|
| 42   | 1.995 | 1.407 | -0.087     | -0.984  | +0.923   | +0.840  | 0.134 |
| 123  | 1.983 | 1.421 | -0.083     | -0.984  | +0.924   | +0.841  | 0.120 |
| 456  | 1.991 | 1.398 | -0.088     | -0.984  | +0.924   | +0.841  | 0.154 |

### Analysis

**Inhibition-dominant expectation suppression profile. Qualitatively distinct from B.**

The detection task breaks the inhibition/excitation symmetry:
- a_inh (~1.99) significantly exceeds a_exc (~1.41), ratio ~1.42
- Net modulation is negative everywhere, strongest at center (-0.087)
- The expected orientation receives the MOST suppression

**Important sign note:** R_dampening is NEGATIVE (-0.984), meaning anti-dampening.
The dampening template has a positive peak at center (excitation at expected); our
profile has a negative peak at center (inhibition at expected). R_sharpening is
POSITIVE (+0.924), indicating the profile partially matches the sharpening template
shape (negative at center, less negative at flanks).

**Interpretation — Expectation Suppression:**
The detection task creates a gradient signal that differentiates expected vs unexpected
orientations. To detect "is expected present?", the system can either:
(a) Enhance expected → look for strong response
(b) Suppress expected → look for absence of response (prediction error signal)

The optimizer chose (b): suppress the expected orientation most strongly. This is
consistent with predictive coding theory where cortical responses encode prediction
error (actual - expected), and expected stimuli are suppressed.

Notably, s_acc is higher in Experiment C (0.120-0.154) than B (0.078-0.087),
suggesting the detection loss also benefits orientation discrimination.

**Cross-seed consistency:** Excellent. All 3 seeds show a_inh/a_exc ratio ~1.41-1.42,
net_center ~-0.085.

---

## Cross-Experiment Summary

| Experiment | Condition                        | Profile             | a_inh/a_exc | net_center | Consistent? |
|------------|----------------------------------|---------------------|-------------|------------|:-----------:|
| A          | lambda_sensory=0, lambda_error=1 | No learning (dead)  | N/A         | 0.000      | 3/3         |
| B          | lambda_energy=0.5, noise=0.15    | Near-null (slight dampening) | ~1.01 | +0.003 | 3/3         |
| C          | lambda_detect=1, lambda_sens=1, noise=0.1 | Expectation suppression | ~1.42 | -0.087 | 3/3 |

### Key Findings

1. **lambda_sensory=0 kills feedback learning.** The sensory loss is the primary
   gradient pathway through the feedback operator. Without it, alphas receive no
   task-relevant gradient signal.

2. **Energy pressure alone doesn't differentiate profiles.** High lambda_energy
   grows both K_inh and K_exc symmetrically, producing a near-null net effect.
   The energy cost gradient affects both channels equally.

3. **The detection task produces expectation suppression.** This is the first
   condition that produces a qualitatively different profile. The binary "is expected
   present?" task creates asymmetric gradient: inhibition grows faster than excitation,
   producing net suppression at the expected orientation.

4. **All results are highly reproducible.** Every experiment shows near-identical
   results across 3 seeds, indicating these are robust training dynamics, not
   stochastic artifacts.

5. **The classify_profile function has a sign bug.** It uses R^2 to select the
   winning class, which ignores whether the correlation is positive or negative.
   This leads to misleading labels (e.g., labeling anti-dampening as "dampening").
   Should be updated to use signed R or to consider the sign when naming the class.

### Classifier Bug Example

For Experiment C:
- R_dampening = -0.984 → R^2 = 0.968 → labeled "dampening"
- But negative R means ANTI-dampening (opposite of dampening template)
- The actual profile is expectation suppression (strongest inhibition at expected)

---

## Code Changes

1. **`src/training/losses.py`**: Added `fb_scale` parameter to `CompositeLoss.forward()`.
   L1 sparsity penalty now multiplied by `fb_scale` to prevent alpha death during burn-in.

2. **`src/training/stage2_feedback.py`**: Passes `fb_scale=net.feedback_scale.item()`
   to the loss function.

3. **Config files created:**
   - `config/exp_dampening.yaml` (Experiment A)
   - `config/exp_sharpening.yaml` (Experiment B)
   - `config/exp_centersurround.yaml` (Experiment C)
   All with `lambda_fb: 0.0` for short runs.

4. **Tests:** All 31 tests pass after changes.

---

## Results Location

All results in `results/targeted/`:

- Experiment A: `results/targeted/expA_s{42,123,456}/` + `.log` files
- Experiment B: `results/targeted/expB_s{42,123,456}/` + `.log` files
- Experiment C: `results/targeted/expC_s{42,123,456}/` + `.log` files

Each directory contains `center_surround_seed{N}/checkpoint.pt`, `metrics.jsonl`, and `stage1_checkpoint.pt`.
