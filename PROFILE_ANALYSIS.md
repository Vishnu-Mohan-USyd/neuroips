# Feedback Mechanism Profile Analysis

## Failure characterization

This is not a failure investigation — it is a diagnostic analysis of feedback mechanism suppression-by-tuning profiles. The question: do the three feedback mechanisms (dampening, sharpening, center-surround) produce the theoretically expected spatial profiles when driven by an oracle predictor?

## Experimental setup

- **Oracle prediction**: Circular Gaussian peaked at 90° (channel 18), σ=10°, normalized to distribution. Precision π=3.0.
- **Part 1**: Direct kernel analysis — computed `compute_som_drive()` and `compute_center_excitation()` outputs for each mechanism at initialization parameters (no training).
- **Part 2**: Full network simulation — loaded Stage 1 checkpoints (damp_l4l23, cs_l4l23), ran 30-step simulations with oracle ON vs OFF for all 36 stimulus orientations. Suppression index = (neutral - oracle) / neutral for the unit tuned to each stimulus.
- **Part 3**: Sharpening full simulation — no Stage 1 checkpoint available, ran at init params.

---

## Results

### DAMPENING (Model A)

**Mechanism**: Narrow positive SOM kernel peaked AT expected orientation.

**Direct kernel (Part 1):**
| Δθ from expected | SOM drive | Center exc | Net inhib |
|---|---|---|---|
| 0° | +0.340 | 0.000 | +0.340 |
| 5° | +0.314 | 0.000 | +0.314 |
| 10° | +0.246 | 0.000 | +0.246 |
| 15° | +0.158 | 0.000 | +0.158 |
| 20° | +0.072 | 0.000 | +0.072 |
| 25° | +0.005 | 0.000 | +0.005 |
| 30° | −0.039 | 0.000 | −0.039 |
| 45° | −0.081 | 0.000 | −0.081 |
| 90° | −0.083 | 0.000 | −0.083 |

**Full network suppression-by-tuning (Part 2, Stage 1 checkpoint):**
| Δθ_stim | R_oracle | R_neutral | Suppression index |
|---|---|---|---|
| 0° | 0.1295 | 0.2348 | **+0.448 (44.8% suppression)** |
| 5° | 0.1370 | 0.2347 | +0.416 |
| 10° | 0.1572 | 0.2347 | +0.330 |
| 15° | 0.1835 | 0.2347 | +0.218 |
| 20° | 0.2088 | 0.2348 | +0.111 |
| 25° | 0.2285 | 0.2349 | +0.027 |
| 30° | 0.2325 | 0.2350 | +0.011 |
| 45° | 0.2352 | 0.2352 | ~0.000 |
| 90° | 0.2353 | 0.2353 | ~0.000 |

**VERDICT: CORRECT.** Profile shows monotonically decreasing suppression from Δθ=0° (strongest) to Δθ≥30° (zero). Matches the expected "strongest suppression at expected orientation" signature.

---

### SHARPENING (Model B)

**Mechanism**: Signed DoG SOM kernel (broad − narrow): minimum AT expected, maximum at flanks.

**Direct kernel (Part 1):**
| Δθ from expected | SOM drive | Center exc | Net inhib |
|---|---|---|---|
| 0° | **−0.427** | 0.000 | −0.427 |
| 5° | −0.390 | 0.000 | −0.390 |
| 10° | −0.293 | 0.000 | −0.293 |
| 15° | −0.167 | 0.000 | −0.167 |
| 20° | −0.049 | 0.000 | −0.049 |
| 25° | +0.040 | 0.000 | +0.040 |
| 30° | +0.093 | 0.000 | +0.093 |
| 40° | **+0.121** | 0.000 | +0.121 |
| 45° | +0.115 | 0.000 | +0.115 |
| 90° | +0.053 | 0.000 | +0.053 |

**Full network suppression-by-tuning (Part 3, init params — no Stage 1 checkpoint):**

Responses are extremely small (~0.0004) because no V1 training occurred, but the profile shape is still visible:

| Δθ_stim | Suppression index |
|---|---|
| 0° | ~0.000 (no suppression) |
| 5° | ~0.000 |
| 10° | ~0.000 |
| 20° | −0.002 |
| 25° | **+0.121** |
| 30° | **+0.234** |
| 45° | **+0.268** (peak) |
| 60° | +0.220 |
| 90° | +0.165 |

**VERDICT: CORRECT.** SOM drive is negative (→ SOM stays at ~0 via rectified_softplus) at expected orientation, positive (→ SOM activated → inhibition) at flanks. The suppression-by-tuning profile shows NO suppression at Δθ=0° and PEAK suppression at intermediate offsets (~30-45°). This is the "flank suppression" signature that sharpens tuning.

**Note on SOM rectification**: The negative SOM drive at Δθ=0° means SOM stays at baseline (≈0). The mechanism works because flanks get ADDED inhibition, not because the center gets facilitation. The L2/3 response at expected is unchanged (SOM≈0), while flanks are suppressed (SOM>0). Net effect: sharpened tuning.

---

### CENTER-SURROUND (Model C)

**Mechanism**: Broad positive SOM (surround inhibition) + narrow center excitation to L2/3.

**Direct kernel (Part 1):**
| Δθ from expected | SOM drive | Center exc | Net inhib (SOM−CtrExc) |
|---|---|---|---|
| 0° | −0.233 | **+0.340** | **−0.573 (facilitation)** |
| 5° | −0.210 | +0.314 | −0.524 |
| 10° | −0.149 | +0.246 | −0.395 |
| 15° | −0.072 | +0.158 | −0.229 |
| 20° | −0.000 | +0.072 | −0.073 |
| 25° | +0.050 | +0.005 | +0.045 |
| 30° | +0.076 | 0.000 | **+0.076 (suppression)** |
| 35° | +0.083 | 0.000 | **+0.083 (peak suppression)** |
| 45° | +0.066 | 0.000 | +0.066 |
| 60° | +0.031 | 0.000 | +0.031 |
| 90° | +0.005 | 0.000 | +0.005 |

**Full network suppression-by-tuning (Part 2, Stage 1 checkpoint):**
| Δθ_stim | R_oracle | R_neutral | Suppression index |
|---|---|---|---|
| 0° | 0.5311 | 0.2366 | **−1.245 (124.5% facilitation!)** |
| 5° | 0.5092 | 0.2366 | −1.152 |
| 10° | 0.4515 | 0.2367 | −0.907 |
| 15° | 0.3773 | 0.2368 | −0.593 |
| 20° | 0.3063 | 0.2366 | −0.294 |
| 25° | 0.2388 | 0.2363 | −0.010 |
| 30° | 0.2172 | 0.2359 | **+0.079 (suppression)** |
| 35° | — | — | **+0.119** |
| 40° | — | — | **+0.131 (peak)** |
| 45° | 0.2054 | 0.2353 | +0.127 |
| 60° | 0.2143 | 0.2351 | +0.089 |
| 90° | 0.2226 | 0.2353 | +0.054 |

**VERDICT: CORRECT.** Strong facilitation at Δθ=0° (response MORE THAN DOUBLES), crossing zero at ~25° offset, peak suppression at ~35-40° offset, slowly recovering at large offsets. This is the classic center-surround profile: center facilitation + surround suppression.

---

## Summary table

| Mechanism | Expected profile | Observed profile | Match? |
|---|---|---|---|
| **Dampening** (A) | Strongest suppression at Δθ=0°, monotonic decay | ✓ 44.8% at 0°, monotonic → 0% at 30°+ | **YES** |
| **Sharpening** (B) | No suppression at 0°, peak at intermediate Δθ | ✓ ~0% at 0°, peak ~27% at 45° | **YES** |
| **Center-surround** (C) | Facilitation at 0°, suppression at flanks | ✓ +124% facilitation at 0°, peak 13% suppression at 40° | **YES** |

## Key observations

1. **All three mechanisms produce the correct profile shapes at initialization.** The mathematical formulations are sound.

2. **The q_centered trick works correctly.** By centering q_pred around 1/N, the mechanism drives SOM negative when q is uninformative (uniform), and the rectified_softplus activation prevents this from producing negative SOM rates. The net effect is that feedback = 0 when q is uniform.

3. **Center-surround shows dramatic facilitation** (>2× response amplification at Δθ=0°). This comes from the center excitation term feeding directly into L2/3 drive. The surround inhibition via SOM is comparatively weaker at the center because the SOM drive is also negative there (SOM stays at zero), so the center excitation acts unopposed.

4. **Sharpening operates by flank suppression, not center facilitation.** The negative SOM drive at center just keeps SOM at baseline; the tuning sharpening comes entirely from elevated SOM at flanks.

5. **Dampening is the cleanest signal** — simple monotonic suppression with σ≈10° half-width matching the kernel width. No secondary effects.

6. **The sharpening Stage 1 issue**: Without a Stage 1 checkpoint, V1 responses are extremely weak (~0.0004), making the sharpening full-network profile marginal. The kernel analysis (Part 1) and the profile shape are still correct in direction.

## Confirmed root cause analysis status

This was a diagnostic analysis, not a failure investigation. **All three feedback mechanisms produce theoretically correct suppression-by-tuning profiles.** No anomalies detected.
