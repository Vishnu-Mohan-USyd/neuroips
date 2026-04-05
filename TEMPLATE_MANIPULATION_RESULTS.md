# Phase F1: Template Manipulation — Dampening Diagnostic

## Purpose

Test whether the learned dampening regime in this model depends on the
**correctness** of the supplied prediction template, or only on its
**peakedness**. This is the key diagnostic for whether a dampening signature
should be interpreted as evidence for predictive coding.

## Design

Four oracle template modes, 3 seeds each (42, 123, 456), identical in every
other respect to the hardened deviance condition (SOM-only, delta-SOM, frozen
V2, decoder frozen, lambda_sensory=0, lambda_l4_sensory=1, lambda_mismatch=1,
lambda_state=0, lambda_energy=0.01, lambda_homeo=1.0, 5000 Stage 2 steps, pi=1
during training).

| Mode | q_pred construction | Peak aligned with true next stimulus? |
|---|---|---|
| `oracle_true` | bump at (current_theta + step) given true next-state | **Yes** |
| `oracle_wrong` | bump at the OPPOSITE state's prediction (CW↔CCW swapped) | **No** (anti-correlated) |
| `oracle_random` | bump at a random orientation, independent of stimulus | **No** (uncorrelated) |
| `oracle_uniform` | uniform distribution (1/N everywhere) | **No peak at all** |

All four templates are normalized to sum to 1. Peaked templates (true, wrong,
random) share the same bandwidth (_make_bump with sigma_ff=12°) — they differ
only in WHERE the peak sits relative to the true stimulus. `oracle_uniform`
has no peak.

## Verification of template construction

Direct inspection of q_oracle tensor for each mode (batch=4, seq=25, N=36):

```
oracle_true     min=0.0000 max=0.1662 std_over_N=0.0506 row_sum=1.0000
oracle_wrong    min=0.0000 max=0.1662 std_over_N=0.0506 row_sum=1.0000
oracle_random   min=0.0000 max=0.1662 std_over_N=0.0506 row_sum=1.0000
oracle_uniform  min=0.0278 max=0.0278 std_over_N=0.0000 row_sum=1.0000
```

All three peaked modes produce identical peak height (0.1662) and spread
(std=0.051). Uniform is exactly 1/36 = 0.0278 everywhere.

Peak location verification for oracle_wrong on a known state-switch trial:
`state=CW curr_ori=0.0° true_peak=15.0° wrong_peak=165.0°`. The wrong template
peaks at (0 - 15) mod 180 = 165° — the CCW prediction — confirming the
CW↔CCW inversion is correct.

## Results

### Learned feedback weights (per-seed)

| Condition | seed 42 | seed 123 | seed 456 | Mean | SD |
|---|---|---|---|---|---|
| `oracle_true` | 2.1471 | 2.1467 | 2.1459 | 2.147 | 0.001 |
| `oracle_wrong` | 2.1460 | 2.1465 | 2.1456 | 2.146 | 0.001 |
| `oracle_random` | 2.1150 | 2.1180 | 2.1171 | 2.117 | 0.002 |
| **`oracle_uniform`** | **0.0700** | **0.0700** | **0.0700** | **0.070** | **0.000** |

`oracle_uniform` value of 0.0700 is exactly the initialization (0.01 × 7 basis
functions). The feedback operator **learned nothing** with a uniform template.
All three seeds are identical to 4 decimal places — this is not a lucky seed
effect; it is a gradient vanishing everywhere when q_centered is zero.

### Kernel shape (averaged over seeds)

| Condition | K(0°) | K(45°) | R(dampening template) |
|---|---|---|---|
| oracle_true | +0.27509 | +0.02655 | +0.914 |
| oracle_wrong | +0.27495 | +0.02656 | +0.914 |
| oracle_random | +0.27043 | +0.02637 | +0.913 |
| oracle_uniform | +0.00798 | +0.00102 | +0.913 (but from tiny K) |

Dampening kernels from true/wrong/random are **identical** up to a ~1% reduction
in amplitude for random (because the random peak location averages away across
training batches more than the fixed CW/CCW geometry does). Uniform kernel is
essentially flat at baseline level.

### L2/3 suppression-by-tuning (feedback on vs off, pi=5, true oracle template at test time)

| Condition | 0° | 10° | 20° | 30° | 45° |
|---|---|---|---|---|---|
| oracle_true | +38.0% | +25.6% | +6.5% | +0.7% | +0.0% |
| oracle_wrong | +38.0% | +25.6% | +6.5% | +0.7% | +0.0% |
| oracle_random | +37.2% | +25.1% | +6.4% | +0.7% | +0.0% |
| **oracle_uniform** | **+0.8%** | **+0.6%** | **+0.2%** | **+0.0%** | **-0.0%** |

true, wrong, random produce indistinguishable dampening curves in L2/3.
Uniform produces essentially no modulation.

## Interpretation

The feedback operator in this model learns a dampening kernel (SOM suppression
concentrated at the peak of q_pred) **whenever the supplied template has a peak,
regardless of whether that peak is aligned with the true next stimulus.**

Mechanism: the circular convolution `inh_field = K_inh ⊛ q_centered` produces
its maximum wherever `q_centered` is most concentrated. The energy cost on
L2/3 activity creates a gradient that pushes K_inh to match the peak shape of
q_centered, so that SOM suppression lands on the most active L2/3 channels and
reduces mean activity most efficiently. The operator does not care *which*
orientation the peak sits at — only that there *is* a peak.

When q_pred is uniform, q_centered = 0 everywhere, so inh_field = 0 regardless
of K_inh, so there is no gradient on K_inh, and alpha stays at initialization.

## Consequence for interpretation

A dampening signature in V1 — suppression at the expected orientation under
top-down feedback — is **not diagnostic of predictive coding**. In this model,
the same signature emerges from:

- a correct oracle prediction,
- the exact opposite prediction,
- a random-orientation prediction,

— with no measurable difference in the learned kernel or the resulting L2/3
profile. The only thing that matters is that the top-down signal carry a peak.
Any peaked inhibitory template, combined with an activity-minimization
pressure, will produce this signature.

This does NOT mean biological V1 dampening is not prediction-related. It means
that behavioral or imaging observations of "expected stimulus suppressed"
cannot — in isolation — distinguish between:

1. A prediction-error code (the suppression reflects what the stimulus is NOT)
2. A peaked-template energy-minimization regime (the suppression just follows
   whatever the top-down signal picks out, for reasons unrelated to its
   accuracy)

To distinguish these, an experiment would need to probe whether the
suppression tracks the *content* of the prediction or only its *structure*.
The template manipulation experiment here is the computational analogue of
that distinction.

## Results location

- `results/template_manipulation/{true,wrong,random,uniform}_s{42,123,456}/` — all 12 checkpoints
- Configs: `config/template_{true,wrong,random,uniform}.yaml`
- Analysis code: embedded in the commit's notebook/script

## Status

All 12 runs completed. Cross-seed variance is negligible (SD < 0.002 for all
peaked templates, identical to 4 decimal places for uniform). The result is
reproducible and the conclusion is robust.
