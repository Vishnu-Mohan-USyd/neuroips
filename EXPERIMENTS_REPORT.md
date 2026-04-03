# Emergent Feedback Experiments Report

**Date**: 2026-04-03
**Seed**: 42 for all runs
**Base config**: `config/simple.yaml` (2-state HMM, no jitter, p_transition=0.95)

## Baseline Reference

| Run | s_acc | cw_acc | a_inh | a_exc | Profile |
|-----|-------|--------|-------|-------|---------|
| Emergent (run1) | 0.300 | 0.846 | 0.075 | 0.257 | Dampening (R=+0.97) |
| Control (no feedback) | 0.082 | 0.727* | — | — | None |

*Control uses state_acc (3-way), not cw_acc (binary).

---

## Experiment 1: High Energy Pressure (lambda_energy 0.01 → 0.1)

**Config**: `config/exp1_energy.yaml`
**Hypothesis**: Energy pressure shifts profile toward suppression.

### Metrics at Checkpoints

| Step | s_acc | cw_acc | ang_err | a_inh | a_exc |
|------|-------|--------|---------|-------|-------|
| 1000 | 0.085 | 0.745 | 16.9 | 0.000 | 0.000 |
| 2000 | 0.167 | 0.844 | 13.8 | 0.000 | 0.167 |
| 3000 | 0.209 | 0.854 | 12.8 | 0.000 | 0.110 |
| 5000 | 0.257 | 0.860 | 13.8 | 0.000 | 0.065 |

### Learned Profile

```
alpha_inh: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
alpha_exc: [0.042, 0.024, 0.0, 0.0, 0.0, 0.0, 0.0]
```

| Template | R | R² |
|----------|---|-----|
| Dampening | +0.944 | 0.892 |
| Sharpening | -0.927 | 0.859 |
| Center-surround | -0.874 | 0.763 |

**Winner: Dampening (R²=0.892)**

### Interpretation

High energy pressure **killed the inhibitory pathway entirely** (a_inh=0.000) and reduced excitation by 75% (a_exc: 0.257→0.065). The profile is still dampening but much weaker. s_acc dropped from 0.300 to 0.257 — the model sacrifices feedback strength to minimize firing rates. Energy pressure does NOT shift toward suppression; it just weakens everything, with inhibition being the first casualty.

---

## Experiment 2: Noisy Stimuli (stimulus_noise=0.3)

**Config**: `config/exp2_noise.yaml`
**Hypothesis**: Noise forces feature-specific feedback (suppress flanks, boost expected).

### Metrics at Checkpoints

| Step | s_acc | cw_acc | ang_err | a_inh | a_exc |
|------|-------|--------|---------|-------|-------|
| 1000 | 0.084 | 0.504 | 23.7 | 0.000 | 0.000 |
| 2000 | 0.110 | 0.491 | 23.6 | 0.000 | 0.065 |
| 3000 | 0.170 | 0.561 | 22.5 | 0.025 | 0.000 |
| 5000 | 0.203 | 0.572 | 22.7 | 0.050 | 0.000 |

### Learned Profile

```
alpha_inh: [0.033, 0.016, 0.0, 0.0, 0.0, 0.0, 0.0]
alpha_exc: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Net modulation at center: **-0.016** (NEGATIVE = suppression)

| Template | R | R² |
|----------|---|-----|
| Dampening | **-0.938** | 0.880 |
| Sharpening | **+0.925** | 0.855 |
| Center-surround | +0.875 | 0.765 |

**Winner by R²: Dampening (R²=0.880), but R is NEGATIVE → profile is anti-dampening = SUPPRESSION**
**Actual best match by signed R: Sharpening (R=+0.925)**

### Interpretation

**THIS IS THE KEY FINDING.** Noise completely **inverted the feedback profile**:
- Excitation pathway died (a_exc=0.000)
- Inhibition pathway is the only survivor (a_inh=0.050)
- Net modulation is NEGATIVE at the predicted orientation = **suppression**
- Profile positively correlates with **sharpening** (R=+0.925)

With noisy stimuli, the optimal strategy shifts from "amplify what you expect" to "suppress what you expect to reduce noise" — the network discovers that suppressing the predictable signal reveals the unpredictable deviation more clearly.

**However**: V2 learning was severely impaired (cw_acc=0.572 vs 0.846 baseline). The V2 predictor cannot learn well from noisy L4 input. This limits the effectiveness of the feedback even though its profile is correct.

---

## Experiment 3: Surprise Detection (lambda_surprise=0.5, 20% violations)

**Config**: `config/exp3_surprise.yaml`
**Hypothesis**: Surprise detection forces differential modulation.

### Metrics at Checkpoints

| Step | s_acc | cw_acc | ang_err | a_inh | a_exc |
|------|-------|--------|---------|-------|-------|
| 1000 | 0.116 | 0.584 | 24.3 | 0.000 | 0.000 |
| 2000 | 0.112 | 0.792 | 19.4 | 0.000 | 0.193 |
| 3000 | 0.201 | 0.809 | 20.3 | 0.011 | 0.312 |
| 5000 | 0.203 | 0.829 | 18.4 | 0.038 | 0.293 |

### Learned Profile

```
alpha_inh: [0.025, 0.013, 0.0, 0.0, 0.0, 0.0, 0.0]
alpha_exc: [0.131, 0.114, 0.048, 0.0, 0.0, 0.0, 0.0]
```

Net modulation at center: +0.059 (excitation)

| Template | R | R² |
|----------|---|-----|
| Dampening | +0.969 | 0.938 |
| Sharpening | -0.920 | 0.846 |
| Center-surround | -0.844 | 0.712 |

**Winner: Dampening (R²=0.938)**

### Interpretation

Surprise detection with 20% violations did NOT shift the profile toward suppression. The profile is very similar to baseline — dampening/center excitation. The higher violation rate made V2 learning harder (cw_acc=0.829 vs 0.846, ang_err=18.4 vs 14.0), and s_acc was lower (0.203 vs 0.300). The surprise head is trained but doesn't change the feedback operator's preferred solution.

---

## Summary Table

| Experiment | s_acc | cw_acc | a_inh | a_exc | Profile | Key Finding |
|------------|-------|--------|-------|-------|---------|-------------|
| **Baseline** | **0.300** | **0.846** | **0.075** | **0.257** | Dampening (R=+0.97) | Center excitation dominant |
| Control | 0.082 | 0.727 | — | — | None | No feedback, no improvement |
| Exp1: Energy | 0.257 | 0.860 | 0.000 | 0.065 | Dampening (R=+0.94) | Energy kills inhibition first |
| **Exp2: Noise** | **0.203** | **0.572** | **0.050** | **0.000** | **Sharpening (R=+0.93)** | **INVERTED: inhibition-only, suppression** |
| Exp3: Surprise | 0.203 | 0.829 | 0.038 | 0.293 | Dampening (R=+0.97) | No change from baseline |

## Key Conclusions

1. **Noise is the key manipulation that produces suppression.** Only Experiment 2 produced a suppressive (sharpening-like) profile. When stimuli are noisy, suppressing the predicted orientation helps the readout discriminate signal from noise.

2. **Energy pressure suppresses ALL feedback**, not selectively. High energy penalty weakens both pathways but kills inhibition first (it's the weaker of the two).

3. **Surprise detection alone is insufficient.** Adding an expected/unexpected readout doesn't change the feedback operator's solution — the model finds the same local minimum.

4. **The excitation-dominance in baseline is an artifact of clean stimuli.** With perfectly predictable, noise-free inputs, the optimal strategy is to amplify the predicted signal. Only when noise corrupts the input does suppression become advantageous.

5. **V2 predictor is the bottleneck for noise experiments.** cw_acc=0.572 vs 0.846 means the V2 can't learn transition structure well from noisy L4. A potential fix: use L2/3 (post-feedback) input to V2 so it gets cleaned-up representations.
