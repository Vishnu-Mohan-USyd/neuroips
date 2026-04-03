# All Experiments: Learned Kernel Shapes and L2/3 Effects

## Overview

Four emergent feedback models analyzed (all seed 42). Each model uses the `EmergentFeedbackOperator` with 7 basis functions.

| Model | Checkpoint | noise | λ_fb | λ_det | λ_energy |
|---|---|---|---|---|---|
| Baseline (noise=0.00) | results/emergent/run1/ | 0.0 | 0.01 | 0.0 | 0.01 |
| Exp B (high energy+noise) | checkpoints/exp_sharpening_v3/ | 0.15 | 0.0 | 0.0 | **0.5** |
| Exp C (detection+discrim) | checkpoints/exp_centersurround/ | 0.1 | 0.0 | **1.0** | 0.01 |
| Noise=0.30 | results/experiments/exp2_noise/ | 0.3 | 0.01 | 0.0 | 0.01 |

**Skipped**: Exp A (targeted), Exp B v1/v2 — all have α_inh = α_exc = 0 (L1 or other forces drove feedback to zero).

---

## Summary table

| Model | Kernel shape | L2/3 shape | SI(0°) | SI(45°) | SI(90°) |
|---|---|---|---|---|---|
| **Baseline** | Center-surround | CENTER-SURROUND | **−3.3%** | +0.7% | +1.1% |
| **Exp B** | Center-surround | CENTER-SURROUND | **−7.0%** | +1.6% | +2.3% |
| **Exp C** | Dampening | CENTER-SURROUND | **−2.8%** | +0.7% | +1.0% |
| **Noise=0.30** | Dampening | **DAMPENING** | **+0.5%** | −0.1% | −0.1% |

SI convention: positive = suppression, negative = facilitation.

**Critical finding: 3 of 4 models produce center-surround L2/3 behavior, regardless of kernel shape.** Only noise=0.30 (which has α_exc = 0, SOM-only) produces actual dampening in L2/3. Whenever both pathways are active, center_exc structurally dominates SOM at the L2/3 level.

---

## Detailed analysis per model

### 1. Baseline (noise=0.00)

**Alpha weights:**

| Basis | α_inh | α_exc | net |
|---|---|---|---|
| G(σ=5°) | **+0.046** | **+0.113** | +0.067 |
| G(σ=15°) | **+0.029** | **+0.099** | +0.070 |
| G(σ=30°) | ~0 | **+0.045** | +0.045 |
| G(σ=60°) | ~0 | ~0 | ~0 |
| MexHat | ~0 | ~0 | ~0 |
| Const | ~0 | ~0 | ~0 |
| Odd/Sin | ~0 | ~0 | ~0 |

‖α_inh‖₁ = 0.075, ‖α_exc‖₁ = 0.257. Excitation 3.4× stronger. L1 sparsity (λ_fb=0.01) drives 4 of 7 bases to zero.

**Kernel profiles:**

| Δθ | K_inh | K_exc | K_net |
|---|---|---|---|
| 0° | +0.0223 | +0.0613 | **+0.0390** |
| 10° | +0.0056 | +0.0195 | +0.0139 |
| 20° | +0.0016 | +0.0078 | +0.0062 |
| 30° | +0.0005 | +0.0036 | +0.0031 |
| 45° | ~0 | +0.0011 | +0.0011 |
| 90° | ~0 | ~0 | ~0 |

**K_net is POSITIVE everywhere** — net excitation at all offsets, peaking at center. Kernel shape: excitation-dominant, center-peaked.
FWHM: K_inh = 20°, K_exc = 20° (similar shapes, different magnitudes).

**L2/3 profile**: CENTER-SURROUND. SI(0°) = −3.3% (facilitation), SI(90°) = +1.1% (suppression). Zero crossing ~30°.

---

### 2. Exp B (high energy + noise)

**Alpha weights:**

| Basis | α_inh | α_exc | net |
|---|---|---|---|
| G(σ=5°) | **+0.293** | **+0.296** | +0.004 |
| G(σ=15°) | **+0.294** | **+0.296** | +0.002 |
| G(σ=30°) | **+0.297** | **+0.297** | ~0 |
| G(σ=60°) | **+0.298** | **+0.297** | −0.001 |
| MexHat | **+0.286** | **+0.295** | **+0.010** |
| Const | **+0.010** | **+0.010** | ~0 |
| Odd/Sin | **+0.011** | **+0.010** | −0.001 |

‖α_inh‖₁ = 1.488, ‖α_exc‖₁ = 1.502. **Nearly identical magnitudes.** All 7 bases active (λ_fb = 0.0).

**Answer to the lead's key question**: K_inh and K_exc have **the same shape** (FWHM both 20°, normalized MAE = 0.0013). The non-trivial net profile comes NOT from different shapes but from **small systematic offsets** across bases. The MexHat basis has the largest net contribution (+0.010), which adds a center-positive, flank-negative modulation — creating the center-surround net profile.

**Kernel profiles:**

| Δθ | K_inh | K_exc | K_net |
|---|---|---|---|
| 0° | +0.2253 | +0.2282 | **+0.0030** |
| 10° | +0.0941 | +0.0950 | +0.0009 |
| 20° | +0.0358 | +0.0356 | **−0.0002** |
| 30° | +0.0172 | +0.0168 | −0.0004 |
| 45° | +0.0101 | +0.0098 | −0.0003 |
| 90° | +0.0040 | +0.0040 | ~0 |

K_net is +0.003 at center (tiny net excitation), −0.0004 at flanks (tiny net inhibition). The kernel is center-surround but with **extremely small net values** — the net is only ~1% of the gross kernel amplitude.

**L2/3 profile**: CENTER-SURROUND, **strongest of all models**. SI(0°) = −7.0% (facilitation), SI(90°) = +2.3% (suppression). Zero crossing ~30°.

**Why is the L2/3 effect so large (~7%) when the net kernel is so small (~1%)?** Because the GROSS kernels are large (K_inh ≈ K_exc ≈ 0.225 at center). After softplus and multiplication by π_eff, both pathways produce large drives (~2.2). The small net difference (0.003) is amplified by the pathway gain asymmetry (center_exc has gain 1.0 to L2/3, SOM pathway has gain < 1.0). The gain mismatch amplifies the small net excitation into a 7% facilitation.

---

### 3. Exp C (detection + discrimination)

**Alpha weights:**

| Basis | α_inh | α_exc | net |
|---|---|---|---|
| G(σ=5°) | **+0.395** | **+0.283** | **−0.112** |
| G(σ=15°) | **+0.396** | **+0.277** | −0.119 |
| G(σ=30°) | **+0.398** | **+0.270** | −0.128 |
| G(σ=60°) | **+0.399** | **+0.265** | −0.134 |
| MexHat | **+0.389** | **+0.295** | −0.093 |
| Const | **+0.011** | **+0.009** | −0.002 |
| Odd/Sin | **+0.006** | **+0.008** | +0.002 |

‖α_inh‖₁ = 1.995, ‖α_exc‖₁ = 1.407. **Inhibition 42% stronger.** All 7 bases active (λ_fb = 0.0).

**Kernel profiles:**

| Δθ | K_inh | K_exc | K_net |
|---|---|---|---|
| 0° | +0.304 | +0.217 | **−0.087** |
| 10° | +0.127 | +0.089 | −0.037 |
| 20° | +0.048 | +0.032 | −0.016 |
| 30° | +0.022 | +0.014 | −0.008 |
| 45° | +0.013 | +0.008 | −0.005 |
| 90° | +0.005 | +0.004 | −0.002 |

K_net is **NEGATIVE everywhere** — net inhibition at all offsets, strongest at center. Kernel shape: pure dampening.
FWHM: K_inh = 20°, K_exc = 20° (same shapes, different amplitudes).

**L2/3 profile**: Despite the dampening kernel, L2/3 shows **CENTER-SURROUND**. SI(0°) = −2.8% (facilitation), SI(90°) = +1.0% (suppression). This is the kernel-to-L2/3 mismatch documented in EXPERIMENT_C_VERIFICATION.md. The center_exc pathway has higher effective gain to L2/3 than the SOM pathway, so the weaker excitation still wins at the L2/3 level.

---

### 4. Noise=0.30

**Alpha weights:**

| Basis | α_inh | α_exc | net |
|---|---|---|---|
| G(σ=5°) | **+0.033** | ~0 | −0.033 |
| G(σ=15°) | **+0.016** | ~0 | −0.016 |
| G(σ=30°) | ~0 | ~0 | ~0 |
| G(σ=60°) | ~0 | ~0 | ~0 |
| MexHat | ~0 | ~0 | ~0 |
| Const | ~0 | ~0 | ~0 |
| Odd/Sin | ~0 | ~0 | ~0 |

‖α_inh‖₁ = 0.050, ‖α_exc‖₁ = 0.000. **Pure SOM inhibition, no excitation.** L1 sparsity (λ_fb=0.01) killed the excitation pathway entirely.

**Kernel profiles:**

| Δθ | K_inh | K_exc | K_net |
|---|---|---|---|
| 0° | +0.0155 | 0 | **−0.0155** |
| 10° | +0.0035 | 0 | −0.0035 |
| 20° | +0.0009 | 0 | −0.0009 |
| 30° | +0.0003 | 0 | −0.0003 |
| 45° | ~0 | 0 | ~0 |
| 90° | ~0 | 0 | ~0 |

**L2/3 profile**: DAMPENING — the **only model where kernel and L2/3 agree**. SI(0°) = +0.5% (suppression), SI(90°) = −0.1% (slight facilitation). Small effect because α weights are small and no excitation to amplify via pathway gain mismatch.

---

## Key findings

### 1. The architecture has a structural bias toward center-surround L2/3 behavior

When BOTH pathways are active (α_exc > 0), the center_exc pathway has higher effective gain to L2/3 than the SOM-mediated inhibition pathway. This means:
- Even when K_inh > K_exc (Exp C: dampening kernel), L2/3 shows facilitation at center
- The only way to get dampening in L2/3 is to have α_exc = 0 (SOM-only, as in noise=0.30)

| Pathway | Gain to L2/3 |
|---|---|
| center_exc → L2/3 | **Direct** (gain ≈ 1.0) |
| SOM drive → SOM → L2/3 | **Indirect** (SOM dynamics with τ_som=10 + w_som gain → effective gain < 1.0) |

### 2. K_inh and K_exc always have the SAME shape (FWHM = 20°)

Across all 4 models, both kernel profiles have FWHM ≈ 20°. The difference between models is in the **amplitude ratio**, not the shape:

| Model | K_inh(0°) | K_exc(0°) | Ratio K_exc/K_inh | K shape |
|---|---|---|---|---|
| Baseline | 0.022 | 0.061 | 2.75 | Net excitation |
| Exp B | 0.225 | 0.228 | 1.01 | Nearly equal (tiny net exc) |
| Exp C | 0.304 | 0.217 | 0.71 | Net inhibition |
| Noise=0.30 | 0.015 | 0.000 | 0.00 | Pure inhibition |

The net profile is always the same SHAPE (peaked at center, monotonically decaying) — only the SIGN changes (positive net = excitation-dominant, negative net = inhibition-dominant).

### 3. Exp B: Nearly identical kernels produce the largest L2/3 effect

Exp B has K_inh ≈ K_exc (ratio 1.01), with K_net only 1.3% of the gross kernel. Yet the L2/3 effect is the **strongest** of all models (7.0% facilitation at center, 2.3% suppression at flanks). This is because:
- The GROSS kernels are large (0.225) → large absolute drives through softplus
- The pathway gain asymmetry amplifies the tiny net excitation (center_exc gains 1.0 vs SOM gains < 1.0)
- High λ_energy = 0.5 creates pressure to reduce overall activity, which the surround suppression helps achieve

The MexHat basis (net = +0.010) is the critical differentiator — it adds center-positive, flank-negative modulation that creates the center-surround profile in K_net.

### 4. No model learned sharpening

None of the 4 models produce a sharpening profile (suppression peaked at flanks, sparing the center). The architecture's basis functions and pathway structure may make sharpening difficult to achieve — it would require K_inh to be broader than K_exc, but all learned kernels have the same FWHM.

### 5. Softplus bias perfectly cancels in all models

Verified: uniform q_pred produces som_drive = center_exc = π×ln(2) = 2.0794 in all 4 models. Net L2/3 effect of bias = 0.000000.

---

## Methodology

- Oracle q_pred: Gaussian bump (σ=10°) at predicted orientation, normalized
- π_eff = 3.0 (fixed)
- 20 timesteps from zero initial state to steady state
- Feedback ON (trained weights) vs OFF (α = 0)
- SI = (R_off − R_on) / R_off (positive = suppression, negative = facilitation)
- Oracle predicts 90° (ch 18), stimulus varies to build suppression-by-tuning profile
- FWHM computed from kernel profile centered at Δθ=0°
