# Noise Sweep Report

**Date**: 2026-04-04
**Base config**: `config/simple.yaml` (emergent mode, v2-input=l4, 5000 Stage 2 steps)
**Seed**: 42 (all runs except reproducibility check)

## 1. Full Results Table

| Noise | s_acc | cw_acc | a_inh | a_exc | net_center | net_flank | Profile (signed R) |
|-------|-------|--------|-------|-------|------------|-----------|-------------------|
| 0.00 | 0.300 | 0.846 | 0.075 | 0.257 | +0.03898 | +0.00857 | dampening (R=+0.972) |
| 0.05 | 0.286 | 0.836 | 0.023 | 0.140 | +0.03155 | +0.00521 | dampening (R=+0.962) |
| 0.10 | 0.275 | 0.835 | 0.099 | 0.264 | +0.03596 | +0.00808 | dampening (R=+0.971) |
| 0.15 | 0.284 | 0.809 | 0.000 | 0.054 | +0.01863 | +0.00115 | dampening (R=+0.920) |
| **0.20** | **0.236** | **0.786** | **0.007** | **0.000** | **-0.00275** | **-0.00025** | **sharpening (R=+0.901)** |
| 0.25 | 0.224 | 0.595 | 0.036 | 0.000 | -0.01195 | -0.00113 | sharpening (R=+0.920) |
| 0.30 | 0.203 | 0.572 | 0.050 | 0.000 | -0.01548 | -0.00149 | sharpening (R=+0.925) |
| 0.40 | 0.179 | 0.517 | 0.015 | 0.000 | -0.00587 | -0.00046 | sharpening (R=+0.903) |
| 0.50 | 0.188 | 0.516 | 0.008 | 0.000 | -0.00326 | -0.00026 | sharpening (R=+0.901) |

Notes:
- `net_center`: K_exc[0] - K_inh[0] at 0° offset from predicted orientation. Positive = excitation, negative = suppression.
- `net_flank`: K_exc[3] - K_inh[3] at 15° offset.
- Profile classification uses signed Pearson R against templates (dampening, sharpening, center-surround).

## 2. Crossover Point

The excitation→suppression transition occurs between **noise=0.15** and **noise=0.20**:

| Noise | net_center | Profile |
|-------|------------|---------|
| 0.15 | +0.01863 | dampening |
| **0.20** | **-0.00275** | **sharpening** |
| 0.25 | -0.01195 | sharpening |

At noise=0.20, net_center switches sign from positive to negative. a_exc drops to ~0 and a_inh begins growing. The transition is sharp — not gradual.

## 3. Profile Evolution (Basis Function Weights)

Basis functions: [σ=5° Gaussian, σ=15° Gaussian, σ=30° Gaussian, σ=60° Gaussian, Mexican hat, constant, sin-like odd]

### alpha_inh (7 weights per noise level)

| Noise | b0 (σ=5°) | b1 (σ=15°) | b2 (σ=30°) | b3 (σ=60°) | b4 (MH) | b5 (const) | b6 (odd) |
|-------|-----------|------------|------------|------------|---------|------------|----------|
| 0.00 | 0.0462 | 0.0289 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.05 | 0.0198 | 0.0033 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.10 | 0.0570 | 0.0417 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.15 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.20 | 0.0072 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.25 | 0.0269 | 0.0090 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.30 | 0.0334 | 0.0163 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.40 | 0.0146 | 0.0004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.50 | 0.0082 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### alpha_exc (7 weights per noise level)

| Noise | b0 (σ=5°) | b1 (σ=15°) | b2 (σ=30°) | b3 (σ=60°) | b4 (MH) | b5 (const) | b6 (odd) |
|-------|-----------|------------|------------|------------|---------|------------|----------|
| 0.00 | 0.1131 | 0.0991 | 0.0448 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.05 | 0.0800 | 0.0599 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.10 | 0.1197 | 0.1042 | 0.0399 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.15 | 0.0428 | 0.0116 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.20 | 0.0003 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.40 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Observations:
- Only narrow basis functions (b0=σ=5°, b1=σ=15°, sometimes b2=σ=30°) are ever used.
- Mexican hat, constant, and odd basis functions are always zero.
- Below crossover: both pathways use narrow Gaussians, exc >> inh.
- Above crossover: exc drops to zero, inh uses only narrow Gaussians.

## 4. Reproducibility Check

noise=0.30 with seed=123 (vs seed=42):

| Seed | s_acc | cw_acc | a_inh | a_exc | net_center | Profile |
|------|-------|--------|-------|-------|------------|---------|
| 42 | 0.203 | 0.572 | 0.050 | 0.000 | -0.01548 | sharpening (R=+0.925) |
| 123 | 0.190 | 0.652 | 0.006 | 0.175 | +0.04797 | dampening (R=+0.955) |

**Result: NOT reproducible at noise=0.30.** Seed 42 converges to inhibition-only (suppression), seed 123 converges to excitation-dominant (dampening). The two seeds reach opposite local minima.

alpha weights for seed=123:
- alpha_inh: [0.005, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
- alpha_exc: [0.101, 0.075, 0.0, 0.0, 0.0, 0.0, 0.0]

## 5. No-Feedback Control Comparison

| Noise | With feedback (s_acc) | Without feedback (s_acc) | Feedback benefit |
|-------|----------------------|--------------------------|-----------------|
| 0.00 | 0.300 | 0.082* | +0.218 (3.7x) |
| 0.30 | 0.203 | — | — |

*Control run from adaptation-only experiment (no emergent feedback).

No-feedback control for noise=0.30 was not run separately. The noise=0.00 control establishes that feedback provides 3.7x s_acc improvement in the clean case.
