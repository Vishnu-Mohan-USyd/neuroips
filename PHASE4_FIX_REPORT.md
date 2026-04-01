# Phase 4 Fix Report: Model B Sharpening Redefined as Signed DoG

**Status:** 155/155 tests pass (51 model + 58 network + 46 stimulus).

## Problem

Model B (sharpening) used a broad positive Gaussian which still peaked at the expected channel — functionally dampening-with-wide-spread, NOT sharpening. A non-negative symmetric kernel convolved with a peaked signal always preserves the peak location.

## Fix

Redefined Model B as a **signed difference-of-Gaussians (DoG)** on the SOM path:

```
SOM_B = b_som + pi_pred * (g_surr * K_broad - g_ctr * K_narrow) @ q_pred
```

This produces SOM that is **minimum at the expected channel** and **maximum at flanks** — true center-sparing inhibition.

## Changes

### `src/model/feedback.py`

- Model B now has **5 parameters** (was 2):
  - `surround_gain_raw` -> g_surround = softplus(raw), init ~1.0
  - `surround_width_raw` -> sigma_broad = softplus(raw), init ~35 deg
  - `center_gain_raw` -> g_center = softplus(raw), init ~1.5
  - `center_width_raw` -> sigma_narrow = softplus(raw), init ~10 deg
  - `som_baseline_raw` -> b_som = softplus(raw), init ~1.0

- Width constraint: `sigma_broad >= sigma_narrow + 10 deg` (enforced via clamp in `surround_width` property)

- `compute_som_drive()` now has three distinct code paths:
  - A: `g_surr * K_narrow @ q_pred * pi_pred` (narrow positive)
  - B: `b_som + pi_pred * (g_surr * K_broad - g_ctr * K_narrow) @ q_pred` (signed DoG)
  - C: `g_surr * K_broad @ q_pred * pi_pred - g_ctr * K_narrow @ q_pred * pi_pred` (positive broad minus narrow)

### `tests/test_network.py`

- Updated `test_sharpening_lowest_at_expected`: now asserts `SOM.argmin() == expected_channel` and flanks > expected for offsets +-3 to +-6
- Updated `test_sharpening_broad_exceeds_narrow_by_10`: tests new width constraint
- Updated golden trial for sharpening: prints new DoG profile values
- Added `TestModelCNetEffect`: verifies net effect is positive at center, negative at flanks for Model C
- Added `test_print_som_profile_vs_delta_theta`: prints SOM drive vs angular offset for A, B, C

## What did NOT change

- Model A (dampening): narrow positive SOM, 2 params — unchanged
- Model C (center-surround): broad SOM + narrow L2/3 excitation, 4 params — unchanged
- Model D (adaptation): zero SOM — unchanged
- Model E (predictive error): zero SOM, error signal path — unchanged
- `compute_center_excitation()`: B still returns zeros, only C is nonzero — unchanged
- `compute_error_signal()`: unchanged

## B vs C Distinction

| | Model B (Sharpening) | Model C (Center-Surround) |
|--|---------------------|--------------------------|
| SOM drive | Signed DoG: min at expected, max at flanks | Positive broad - narrow: low at expected, high at flanks |
| Template to L2/3 excitation | **None** | **Yes** — narrow center excitation |
| Net effect on L2/3 | Suppress flanks via SOM only | Boost center via excitation + suppress surround via SOM |
| Params | 5 (incl. som_baseline) | 4 |

## SOM Drive Profiles (one-hot at ch9=45 deg, pi=3.0)

| Mechanism | ch9 (expected) | ch0 (far) | ch15 (flank) | max | min |
|-----------|---------------|-----------|-------------|-----|-----|
| A Dampening | +3.000 | +0.000 | +0.033 | +3.000 | +0.000 |
| B Sharpening | **-0.500** | +2.313 | +3.028 | +3.127 | **-0.500** |
| C Center-surround | +0.000 | +0.974 | +1.786 | +1.996 | +0.000 |
| D Adaptation | 0 | 0 | 0 | 0 | 0 |
| E Pred Error | 0 | 0 | 0 | 0 | 0 |

Key: Model B now shows the correct inverted profile — minimum at expected (-0.500), maximum at flanks (+3.127).

## SOM Drive vs Delta-Theta (diagnostic profile)

```
dtheta   A(damp)  B(sharp)    C(c-s)
    0°   +3.0000   -0.5000   +0.0000
    5°   +2.6475   -0.0017   +0.3111
   10°   +1.8196   +1.1506   +1.0183
   15°   +0.9740   +2.2758   +1.6735
   20°   +0.4060   +2.9391   +1.9962
   25°   +0.1318   +3.1268   +1.9881
   30°   +0.0333   +3.0277   +1.7863
   45°   +0.0001   +2.3125   +0.9738
   60°   +0.0000   +1.6902   +0.4060
   90°   +0.0000   +1.1100   +0.0333
```

- A: sharp peak at 0 deg, rapid falloff
- B: dip at 0 deg, peak at ~25 deg, gradual falloff — classic DoG center-sparing profile
- C: low at 0 deg, peak at ~20 deg, moderate falloff

## Golden Trial Values (updated for Model B)

Sharpening golden SOM (ch9, pi=3.0):
- Min (ch9): -0.500000
- Max: +3.126796 at ch4 (offset ~25 deg)
- Far (ch0): +2.312514
- Flank (ch12): +2.275826
- Flank (ch15): +3.027718
- Deterministic: re-run identical to atol=1e-6

## Parameter Count Summary

| Model | Feedback params | Total |
|-------|----------------|-------|
| A (Dampening) | 2 | 2 |
| B (Sharpening) | 5 | 5 |
| C (Center-Surround) | 4 | 4 |
| D (Adaptation) | 0 | 0 |
| E (Pred Error) | 0 | 0 |

## Gradient Flow

All parameters receive gradients for all 5 mechanisms (verified). Model B's 5 feedback params (including `som_baseline_raw`) all get gradients through the comprehensive loss.
