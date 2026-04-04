# SOM-Only Architecture Experiment Report

## Architecture Change

Removed the excitatory pathway entirely from `EmergentFeedbackOperator`.
Feedback now acts ONLY through SOM inhibition — there is no direct excitatory
pathway to L2/3.

**Rationale:** Debugger analysis showed the excitatory pathway had higher
effective gain than the SOM pathway, preventing genuine dampening profiles
even when kernel weights favored inhibition. With SOM-only feedback,
inhibitory kernel shape directly determines the L2/3 modulation effect
(no pathway gain mismatch).

**Changes:**
- `src/model/feedback.py`: Removed `alpha_exc`, single `K_inh` return from `get_profiles()`, `forward()` returns `som_drive` only
- `src/model/network.py`: `center_exc = torch.zeros_like(som_drive)` for emergent mode
- `src/training/losses.py`: `feedback_sparsity_loss` uses only `alpha_inh`
- `src/training/stage2_feedback.py`: Logging references updated (no `a_exc`)
- `src/analysis/feedback_discovery.py`: `extract_profiles()` returns single tensor; `classify_profile()` uses signed R
- `tests/test_network.py`: All 73 tests updated and passing
- Full suite: 322/322 tests pass

---

## Experiments

Two conditions, each with 3 seeds (42, 123, 456), 5000 Stage 2 steps:

1. **Baseline**: `config/simple.yaml` (lambda_sensory=1.0, lambda_pred=0.5, lambda_fb=0.01)
2. **Exp C (Detection)**: `config/exp_centersurround.yaml` (+ lambda_detection=1.0, noise=0.1, lambda_fb=0.0)

All runs use `--feedback-mode emergent --v2-input l4`.

---

## Results

### Baseline (simple.yaml)

| Seed | a_inh | inh_center | inh_flank | R_damp | R_sharp | R_cs   | s_acc | cw_acc |
|------|-------|------------|-----------|--------|---------|--------|-------|--------|
| 42   | 0.000 | -0.00000   | -0.00000  | -0.003 | +0.003  | +0.002 | 0.087 | 0.856  |
| 123  | 0.000 | +0.00000   | +0.00000  | +0.002 | -0.001  | -0.001 | 0.096 | 0.841  |
| 456  | 0.000 | +0.00000   | -0.00000  | +0.001 | -0.002  | -0.002 | 0.082 | 0.842  |

**Analysis: No feedback learned.** All alphas are effectively zero. The L1 penalty
(lambda_fb=0.01) drives alphas to zero during burn-in and they never recover.
This is consistent with the prior finding that lambda_fb > 0 kills alphas in
short runs. The baseline sensory accuracy (~8.2-9.6%) is above chance (2.8%)
and CW accuracy (~84-86%) is well above chance (50%), confirming V2 learns
transition statistics without feedback.

### Experiment C: Detection + Discrimination

| Seed | a_inh | inh_center | inh_flank | R_damp  | R_sharp | R_cs   | s_acc | cw_acc |
|------|-------|------------|-----------|---------|---------|--------|-------|--------|
| 42   | 0.810 | -0.124     | -0.001    | -0.983  | +0.943  | +0.872 | 0.084 | 0.806  |
| 123  | 0.255 | -0.031     | +0.003    | -0.954  | +0.927  | +0.865 | 0.076 | 0.765  |
| 456  | 0.184 | -0.028     | -0.000    | -0.979  | +0.958  | +0.899 | 0.079 | 0.739  |

**Alpha weights (all 7 basis functions):**

| Seed | σ=5   | σ=15  | σ=30  | σ=60  | MexHat | Const | Odd    |
|------|-------|-------|-------|-------|--------|-------|--------|
| 42   | -0.164| -0.157| -0.140| -0.123| -0.179 | 0.010 | 0.038  |
| 123  | -0.041| -0.037| -0.030| -0.025| -0.054 | 0.010 | 0.059  |
| 456  | -0.038| -0.032| -0.023| -0.018| -0.053 | 0.010 | -0.011 |

---

## Key Findings

### 1. Detection task produces expectation suppression (confirmed with SOM-only)

All 3 Exp C seeds produce a profile where the **expected orientation receives
the strongest inhibition** (inh_center is the most negative value). This is
consistent with the pre-architecture-change results.

The profile shape: narrow peak of inhibition at the expected orientation,
falling off toward the flanks. This is the **anti-dampening** pattern —
SOM suppresses the expected stimulus most strongly.

### 2. Profile classification: expectation suppression, not sharpening

The signed correlations tell the full story:

- **R_dampening ≈ -0.98**: Strong anti-correlation with dampening template.
  The dampening template has a positive peak at center (suppress at expected),
  and our profile has a **negative** peak at center. Since our profile IS
  the SOM inhibition profile, negative values mean the *absolute magnitude*
  of inhibition peaks at center — which IS suppression at expected. The sign
  confusion is because the classifier correlates against the *profile shape*,
  not the *effect on L2/3*.

- **R_sharpening ≈ +0.94**: Positive correlation with sharpening template
  (DoG: broad minus narrow). This makes sense because the negative-at-center
  profile is shaped like a negative Gaussian, which correlates with the
  negative center of the sharpening DoG template.

**Interpretation:** The learned profile is best described as **expectation
suppression** — maximal SOM inhibition at the predicted orientation. Through
SOM's inhibition of L2/3, this suppresses L2/3 response at the expected
orientation while leaving flanking orientations relatively uninhibited.

The net effect on L2/3 is analogous to sharpening: reduced response at
expected, preserved response at unexpected. But the mechanism is pure
inhibition (no excitatory counterpart), making it a cleaner prediction
error signal.

### 3. Alpha weight structure is consistent across seeds

All 3 seeds show the same qualitative pattern:
- All Gaussian basis weights (σ=5,15,30,60) are **negative** (suppressive)
- The **Mexican hat** basis has the most negative weight (drives center-peaked
  inhibition relative to flanks)
- The **constant** basis stays at init value (0.01), providing no spatial structure
- The **odd** basis is small and variable (±0.01-0.06)

The relative weighting (MexHat > σ=5 > σ=15 > σ=30 > σ=60) creates a
composite kernel that is narrower than any single Gaussian, with the
Mexican hat contributing the sharp center peak.

### 4. Magnitude varies across seeds but shape is stable

- Seed 42: a_inh = 0.810 (strongest)
- Seed 123: a_inh = 0.255 (moderate)
- Seed 456: a_inh = 0.184 (weakest)

Despite 4× magnitude difference, the **profile shape** is nearly identical
(R_damp = -0.95 to -0.98 across all seeds). The profile correlations are
robust; only the overall gain differs.

### 5. Baseline confirms feedback learning requires lambda_fb=0

With lambda_fb=0.01 (baseline), all alphas are driven to zero. The
L1 penalty during burn-in (when fb_scale=0) has no compensating task
gradient, so it kills the weights permanently. Exp C uses lambda_fb=0.0,
allowing free alpha growth.

---

## Comparison: Dual-pathway vs SOM-only

| Metric | Dual-pathway (old) | SOM-only (new) |
|--------|-------------------|----------------|
| a_inh (Exp C mean) | ~1.99 | ~0.42 |
| a_exc (Exp C mean) | ~1.41 | N/A |
| net_center (Exp C) | -0.087 | -0.061 |
| R_damp (Exp C)     | -0.984 | -0.972 |
| R_sharp (Exp C)    | +0.924 | +0.943 |
| Profile shape       | Identical | Identical |
| Profile class       | Expectation suppression | Expectation suppression |

The dual-pathway system achieved similar suppression through the
*difference* of two large opposing pathways (1.99 inhibitory - 1.41
excitatory = 0.58 net). The SOM-only system achieves it directly
through inhibition alone, with lower absolute magnitudes and no
pathway cancellation.

---

## Results Location

All results in `results/som_only/`:

- Baseline: `results/som_only/baseline_s{42,123,456}/` + `.log` files
- Exp C: `results/som_only/expC_s{42,123,456}/` + `.log` files
- Raw data: `results/som_only/som_only_results.json`

---

## Code Changes Summary

1. `src/model/feedback.py`: Removed `alpha_exc`, `_cached_exc_circulant`; `get_profiles()` returns single `Tensor`; `forward()` returns `som_drive` only
2. `src/model/network.py`: Emergent mode sets `center_exc = zeros`, uses single return from `self.feedback()`
3. `src/training/losses.py`: `feedback_sparsity_loss()` uses only `alpha_inh.abs().sum()`; `fb_scale` parameter in `forward()`
4. `src/training/stage2_feedback.py`: Removed `a_exc_norm` from logging and metrics
5. `src/analysis/feedback_discovery.py`: `extract_profiles()` returns single tensor; `classify_profile()` uses signed R (fixes R² sign bug)
6. `scripts/extract_sweep_results.py`: Updated for SOM-only interface
7. `tests/test_network.py`: Updated all emergent feedback tests for single-return interface

All 322 tests pass.
