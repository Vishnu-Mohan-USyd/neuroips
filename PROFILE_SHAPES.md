# Emergent Feedback Profile Analysis: Learned Kernel Shapes

## Summary

Two emergent feedback models were analyzed: **noise=0.30** (stimulus_noise=0.3, `results/experiments/exp2_noise/`) and **noise=0.00** (stimulus_noise=0.0, `results/emergent/run1/`). Both use the `EmergentFeedbackOperator` with 7 learnable basis functions.

**Key results:**
- **noise=0.30** → Pure dampening: SOM-only inhibition, no excitation. ~0.5% suppression at predicted orientation.
- **noise=0.00** → Center-surround: Both SOM inhibition and center excitation. ~3.3% facilitation at predicted, ~1% suppression at flanks.
- The softplus(0) = ln(2) bias **perfectly cancels** in L2/3 (both pathways receive identical constant → net effect = 0).
- Both models learned narrow kernels (FWHM ≈ 20°) from only 2 of 7 basis functions (σ=5° and σ=15° Gaussians).

---

## Analysis 1: Learned kernel weights

### Basis function labels
| Index | Name | Description |
|---|---|---|
| 0 | G(σ=5°) | Narrow Gaussian |
| 1 | G(σ=15°) | Medium Gaussian |
| 2 | G(σ=30°) | Broad Gaussian |
| 3 | G(σ=60°) | Very broad Gaussian |
| 4 | MexHat | Mexican hat (narrow − broad) |
| 5 | Const | Constant / global gain |
| 6 | Odd/Sin | Odd (sine) basis for asymmetry |

### noise=0.30 model

| Basis | α_inh | α_exc |
|---|---|---|
| G(σ=5°) | **+0.0334** | ~0 (10⁻¹⁰) |
| G(σ=15°) | **+0.0163** | ~0 (10⁻¹⁰) |
| G(σ=30°) | ~0 (10⁻⁹) | ~0 |
| G(σ=60°) | ~0 (10⁻¹¹) | ~0 |
| MexHat | ~0 (10⁻¹⁰) | ~0 |
| Const | ~0 (10⁻¹²) | ~0 |
| Odd/Sin | ~0 (10⁻¹⁰) | ~0 |

- **‖α_inh‖₁ = 0.0497**, **‖α_exc‖₁ ≈ 0** (10⁻¹⁰)
- L1 sparsity drove all excitatory weights to zero. Only narrow inhibition survived.
- Interpretation: **Pure SOM-mediated dampening** — predicted orientation gets extra SOM inhibition, no compensatory excitation.

### noise=0.00 model

| Basis | α_inh | α_exc |
|---|---|---|
| G(σ=5°) | **+0.0462** | **+0.1131** |
| G(σ=15°) | **+0.0289** | **+0.0991** |
| G(σ=30°) | ~0 (10⁻¹⁰) | **+0.0448** |
| G(σ=60°) | ~0 (10⁻¹⁰) | ~0 (10⁻¹⁰) |
| MexHat | ~0 (10⁻¹¹) | ~0 (10⁻¹⁰) |
| Const | ~0 (10⁻¹¹) | ~0 (10⁻¹¹) |
| Odd/Sin | ~0 (10⁻¹⁰) | ~0 (10⁻¹⁰) |

- **‖α_inh‖₁ = 0.0752**, **‖α_exc‖₁ = 0.2569**
- Excitation uses 3 Gaussian bases (σ=5°, 15°, 30°); inhibition uses 2 (σ=5°, 15°).
- Excitation is ~3.4× stronger than inhibition (by L1 norm).
- Interpretation: **Center-surround** — strong narrow excitation at predicted, weaker narrow inhibition (net: facilitation at center, suppression at flanks where excitation decays faster).

---

## Analysis 2: Kernel profiles (computed as α · basis)

Kernel values at key angular offsets from predicted orientation (Δθ):

| Δθ | noise=0.30 K_inh | noise=0.30 K_exc | noise=0.00 K_inh | noise=0.00 K_exc |
|---|---|---|---|---|
| 0° | +0.01548 | ~0 | +0.02228 | **+0.06127** |
| 5° | +0.01013 | ~0 | +0.01482 | +0.04277 |
| 10° | +0.00354 | ~0 | +0.00558 | +0.01948 |
| 15° | +0.00146 | ~0 | +0.00254 | +0.01113 |
| 20° | +0.00090 | ~0 | +0.00159 | +0.00782 |
| 30° | +0.00029 | ~0 | +0.00052 | +0.00360 |
| 45° | +0.00002 | ~0 | +0.00004 | +0.00112 |
| 60° | ~0 | ~0 | ~0 | +0.00041 |
| 90° | ~0 | ~0 | ~0 | +0.00003 |

- **noise=0.30**: Inhibitory FWHM ≈ 20°. No excitation. Pure narrow dampening kernel.
- **noise=0.00**: Both kernels have FWHM ≈ 20°, but excitation is ~2.7× stronger at center (0.061 vs 0.022). Excitation also has a broader tail (via σ=30° component), but it decays faster than inhibition at large offsets, creating the surround.

---

## Analysis 3: Net effect on L2/3 (excitation − inhibition signal)

Bias-subtracted orientation-specific signals (uniform q_pred subtracted), at π_eff = 3.0:

### noise=0.30

| Δθ | SOM signal | Exc signal | Net (exc−som) | Effect |
|---|---|---|---|---|
| 0° | +0.00958 | 0 | **−0.00958** | Suppression |
| 5° | +0.00863 | 0 | −0.00863 | Suppression |
| 10° | +0.00624 | 0 | −0.00624 | Suppression |
| 15° | +0.00348 | 0 | −0.00348 | Suppression |
| 20° | +0.00120 | 0 | −0.00120 | Suppression |
| 30° | −0.00115 | 0 | +0.00115 | Facilitation |
| 45° | −0.00195 | 0 | +0.00195 | Facilitation |
| 90° | −0.00207 | 0 | +0.00207 | Facilitation |

**Classification: DAMPENING** — monotonically decreasing suppression from Δθ=0° (strongest) to ~25° (zero-crossing), then slight facilitation at flanks (from SOM q_centered going negative at off-channels).

### noise=0.00

| Δθ | SOM signal | Exc signal | Net (exc−som) | Effect |
|---|---|---|---|---|
| 0° | +0.01407 | **+0.04052** | **+0.02644** | Facilitation |
| 5° | +0.01271 | +0.03692 | +0.02421 | Facilitation |
| 10° | +0.00929 | +0.02785 | +0.01856 | Facilitation |
| 15° | +0.00530 | +0.01708 | +0.01178 | Facilitation |
| 20° | +0.00196 | +0.00778 | +0.00582 | Facilitation |
| 30° | −0.00159 | **−0.00305** | −0.00146 | Suppression |
| 45° | −0.00291 | −0.00841 | **−0.00550** | Suppression |
| 60° | −0.00311 | −0.00992 | −0.00681 | Suppression |
| 90° | −0.00313 | −0.01057 | **−0.00744** | Suppression |

**Classification: CENTER-SURROUND** — strong facilitation at predicted orientation (+0.026 at Δθ=0°), zero-crossing at ~25°, peak suppression at orthogonal (−0.007 at Δθ=90°).

---

## Analysis 4: Softplus bias — perfect cancellation

**The softplus(0) = ln(2) ≈ 0.693 bias does NOT affect L2/3 responses.**

When q_pred = uniform (1/N for all channels):
- q_centered = 0 for all channels
- inh_field = exc_field = 0 for all channels  
- softplus(0) = ln(2) → som_drive = center_exc = π_eff × ln(2) = 2.079 (constant)

Both SOM and center_exc pathways receive **identical** constant bias. At steady state, SOM equilibrates to match this bias, and the SOM→L2/3 subtraction exactly cancels the center_exc addition:

| Condition | L2/3 at stim (noise=0.30) | L2/3 at stim (noise=0.00) |
|---|---|---|
| A: α=0 (no feedback at all) | 1.21717 | 1.21777 |
| B: uniform q (bias only) | 1.21717 | 1.21777 |
| C: oracle q (full feedback) | 1.21052 | 1.25758 |
| **B − A (bias effect)** | **0.000000** | **0.000000** |
| **C − B (signal effect)** | **−0.00665** | **+0.03981** |

The bias has **zero net effect** on L2/3. Only the orientation-specific signal (kernel shape × q_centered) drives changes.

---

## Analysis 5: Full-network suppression-by-tuning profiles

Oracle predicts 90° (channel 18), stimulus orientation varies. π_eff = 3.0, 20 steps to steady state.

### noise=0.30

| Δθ (stim−pred) | R_on | R_off | Delta | SI |
|---|---|---|---|---|
| 0° (match) | 1.21052 | 1.21717 | −0.00665 | **+0.55%** |
| 5° | 1.21109 | 1.21717 | −0.00608 | +0.50% |
| 10° | 1.21253 | 1.21717 | −0.00463 | +0.38% |
| 15° | 1.21428 | 1.21717 | −0.00289 | +0.24% |
| 20° | 1.21582 | 1.21717 | −0.00134 | +0.11% |
| 30° | 1.21770 | 1.21717 | +0.00054 | −0.04% |
| 45° | 1.21863 | 1.21717 | +0.00147 | −0.12% |
| 60° | 1.21883 | 1.21717 | +0.00166 | −0.14% |
| 90° (orthogonal) | 1.21887 | 1.21717 | +0.00170 | **−0.14%** |

**Profile: DAMPENING** — strongest suppression at predicted orientation (SI = +0.55%), zero-crossing at ~25°, slight facilitation at flanks (−0.14%).

### noise=0.00

| Δθ (stim−pred) | R_on | R_off | Delta | SI |
|---|---|---|---|---|
| 0° (match) | 1.25758 | 1.21777 | +0.03981 | **−3.27%** (facilitation) |
| 5° | 1.25478 | 1.21777 | +0.03700 | −3.04% |
| 10° | 1.24752 | 1.21777 | +0.02974 | −2.44% |
| 15° | 1.23838 | 1.21777 | +0.02061 | −1.69% |
| 20° | 1.22973 | 1.21777 | +0.01195 | −0.98% |
| 30° | 1.21743 | 1.21777 | −0.00034 | +0.03% |
| 45° | 1.20897 | 1.21777 | −0.00880 | **+0.72%** |
| 60° | 1.20604 | 1.21777 | −0.01173 | +0.96% |
| 90° (orthogonal) | 1.20479 | 1.21777 | −0.01298 | **+1.07%** |

**Profile: CENTER-SURROUND** — strong facilitation at predicted (SI = −3.27%), zero-crossing at ~25°, surround suppression peaking at orthogonal (SI = +1.07%).

---

## Comparison summary

| Property | noise=0.30 | noise=0.00 |
|---|---|---|
| **Learned kernel type** | Pure dampening (SOM only) | Center-surround (SOM + exc) |
| Active bases (inh) | σ=5° + σ=15° | σ=5° + σ=15° |
| Active bases (exc) | None | σ=5° + σ=15° + σ=30° |
| ‖α_inh‖₁ | 0.050 | 0.075 |
| ‖α_exc‖₁ | 0 | 0.257 |
| Kernel FWHM (inh) | 20° | 20° |
| Kernel FWHM (exc) | — | 20° (but broader tail) |
| Peak SI at Δθ=0° | +0.55% (suppression) | −3.27% (facilitation) |
| Flank SI at Δθ=45° | −0.12% (facilitation) | +0.72% (suppression) |
| Effect magnitude | Small (~0.5%) | Moderate (~3%) |
| Softplus bias effect | Zero | Zero |

---

## Interpretation

1. **Stimulus noise selects mechanism type.** With noise (0.30), the model learns pure dampening — suppress the expected orientation to boost relative salience of unexpected stimuli. Without noise (0.00), the model learns center-surround — amplify the expected orientation while suppressing alternatives.

2. **Both models converge to narrow kernels.** Despite having 7 basis functions available (including broad Gaussians, Mexican hat, constant, and odd bases), both models use only the 2 narrowest Gaussians (σ=5° and σ=15°). The L1 sparsity penalty (λ_fb = 0.01) drives unused bases to exactly zero (~10⁻¹⁰).

3. **The excitation pathway (center_exc) is the key differentiator.** The inhibitory kernels are qualitatively similar (narrow, σ=5°+15°). The difference is entirely in the excitation: noise=0.30 has no excitation, noise=0.00 has strong excitation (3.4× inhibition).

4. **Effect magnitudes are small but orientation-specific.** The full-network suppression indices are 0.5% (noise=0.30) to 3.3% (noise=0.00). These are consistent with the small learned α weights and the fact that the signal (kernel × q_centered) is ~100–1000× smaller than the softplus bias (which cancels).

5. **The softplus bias is architecturally harmless.** Despite softplus(0) = 0.693 creating a large constant drive (~2.08 at π=3.0), this bias is identical in both SOM and center_exc pathways and cancels exactly in L2/3 at steady state. The bias serves only to keep gradients flowing through the softplus (vs. ReLU which would have zero gradient at 0).

---

## Experimental methodology

All experiments used:
- Oracle q_pred: Gaussian bump (σ=10°) centered at predicted orientation, normalized to distribution
- π_eff = 3.0 (typical trained precision)
- 20 timesteps from zero initial state to steady state
- Feedback ON vs OFF comparison: alpha weights intact vs zeroed
- Bias subtraction: uniform q_pred (all 1/N) baseline subtracted to isolate orientation-specific signal
- Suppression index: SI = (R_off − R_on) / R_off (positive = suppression, negative = facilitation)
