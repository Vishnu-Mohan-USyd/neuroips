# Results

## Summary

This model produces one robust, well-characterized feedback regime:
**template-conditioned dampening**. Representational sharpening does not
emerge from any tested combination of task, loss, timing, or template-width
modifications. The SOM-only inhibitory architecture appears to be a
fundamental constraint.

---

## 1. Dampening: Robust and Well-Characterized

### What it is

When L2/3 sensory supervision is removed (lambda_sensory=0) and a global
energy cost is present (lambda_energy=0.01), the feedback operator learns
a kernel that suppresses L2/3 activity at the predicted orientation.

### Evidence

| Metric | Value |
|---|---|
| Peak gain at predicted orientation (ON/OFF) | 0.65 (35% reduction) |
| SI(0°) at pi=5 | +38% (maximal suppression at center) |
| Population FWHM narrowing | −4% (1.1° narrower; real, not a rectifier artifact) |
| Energy reduction at expected channels | −47% |
| Cross-seed consistency | Identical to 4 decimal places (3 seeds) |
| Rotational invariance | Identical SI curves across 8 tested oracle orientations |

### Template Manipulation (Key Diagnostic)

Tested whether dampening requires the CORRECTNESS of the prediction or only
its PEAKEDNESS. Four oracle template modes, 3 seeds each (12 runs):

| Template | ‖alpha_inh‖ | SI(0°) at pi=5 |
|---|---|---|
| True oracle (correct prediction) | 2.147 ± 0.001 | +38.0% |
| Wrong oracle (CW↔CCW swapped) | 2.146 ± 0.001 | +38.0% |
| Random oracle (uncorrelated peak) | 2.117 ± 0.002 | +37.2% |
| **Uniform oracle (no peak)** | **0.070 ± 0.000** | **+0.8%** |

**True/wrong/random produce indistinguishable dampening. Uniform produces
nothing.** Dampening depends on template PEAKEDNESS, not CORRECTNESS. A
V1 dampening signature is therefore not diagnostic of predictive coding.

### Confound Controls

| Control | ‖alpha‖ | SI(0°) | Survives? |
|---|---|---|---|
| Original | 2.147 | +38.0% | — |
| No adaptation (alpha_adaptation=0) | 2.148 | +37.7% | Yes |
| 50%-reliability transitions | 2.178 | +38.6% | Yes |
| End-to-end learned V2 (cw_acc=44%) | 2.336 | +42.4% | Yes |

Dampening survives all confound controls. It is not driven by adaptation,
not driven by predictable transitions, and not dependent on oracle V2.

### Mechanism

The energy cost penalizes total L2/3 activity. The feedback operator reduces
activity via SOM inhibition. It concentrates suppression at the peak of
q_pred because that's where the circular convolution K_inh ⊛ q_centered
produces the largest SOM drive. Without a sensory loss on L2/3 to resist
this suppression, dampening is the path of least resistance. Any peaked
top-down template suffices — correctness is irrelevant.

---

## 2. Sharpening: Not Observed in This Architecture

### The systematic investigation (Phases 2–5)

Five ingredients were tested, individually and in combination, to produce
Kok-style representational sharpening (narrower tuning + improved
fine discrimination):

| Phase | Ingredient | M6 Δd' (δ=10°) | M7 ΔAcc (δ=10°) | Pop FWHM Δ |
|---|---|---|---|---|
| P4 baseline | Fine disc + noise | +2.09 | −0.005 | −0.03° |
| Phase 2 | + Ambiguous competitors | **+2.65** | −0.010 | −0.01° |
| Phase 3 | + Shifted timing (prior) | +1.87 | −0.005 | 0° |
| Phase 4 | + Local 5-way disc loss | +2.41 | −0.010 | 0° |
| Phase 5 (σ=5) | + Narrow oracle template | +2.53 | −0.010 | 0° |

**M6** = local d' via population-vector decode (1D circular projection).
**M7** = accuracy of a trained LogReg decoder on all 36 channels.
**Pop FWHM** = width of the population response bump.

### What the metrics show

**M6 (popvec d'):** Positive delta in all conditions except dampening.
Feedback tightens population-vector orientation estimates by suppressing
noisy far-flank channels. This is a form of representational noise
reduction, but it is specific to the popvec decoder geometry.

**M7 (trained LogReg):** Flat or slightly negative across ALL conditions.
A trained linear decoder, which can learn to optimally weight channels,
sees NO improvement from feedback. Whatever the operator is doing, it
does not create new information that a flexible downstream decoder can
exploit. This is the key negative result.

**Population FWHM:** Unchanged (within 0.03°) in all non-dampening
conditions. The feedback does not narrow the population response bump.

**Peak gain:** Unchanged (ratio = 1.000) in all non-dampening conditions.
No channel is boosted.

### Why sharpening fails in this architecture

Representational sharpening requires at least one of:

1. **An excitatory mechanism** to boost the expected channel above its
   feedforward-driven level (e.g., disinhibition via VIP→SOM, or apical
   excitatory input)
2. **A multiplicative gain mechanism** that narrows the effective tuning
   curve width, not just suppresses additive activity

The SOM-only inhibitory pathway provides neither. SOM can only SUBTRACT
from L2/3 drive. Subtracting more at flanks than at the center pushes
flank responses below the rectifier threshold (killing already-weak
responses) but cannot CREATE sharper responses at the center. The
population code at the center is unchanged — same peak, same width —
regardless of what happens at the flanks.

### What the feedback operator DOES learn

Under fine-discrimination conditions (P4 and derivatives), the operator
learns a DoG-like kernel (negative center, positive surround) that
suppresses L2/3 channels at ±25–30° from the predicted orientation.
This produces:

- ~3–7% energy reduction in surround/far channels
- ~1–3 d' improvement in popvec estimation (M6)
- Zero improvement in trained-decoder accuracy (M7)
- Zero change in peak gain or population FWHM

This is best described as **weak competitor noise suppression**, not
representational sharpening.

---

## 3. Other Findings

### Sensory loss on L2/3 blocks dampening

With lambda_sensory > 0 on L2/3, the feedback operator learns only weak
alpha (||alpha|| ≈ 0.3–0.4) and produces no measurable L2/3 modulation.
The sensory loss provides a brake that prevents suppression of expected
stimuli — because suppressing expected channels hurts orientation decoding.

### Dampening is the default with or without prediction

The ablation (no mismatch loss, no sensory loss on L2/3, just energy) gives
identical dampening to the full deviance condition. The mismatch detection
objective has zero effect on the learned kernel.

### Dampening's FWHM narrowing is genuine but small

Dampening narrows the population bump by ~4% (1.1°). This is confirmed
NOT to be a rectifier clipping artifact: the pre-rectifier drive also
narrows (by 1.41°), and the narrowing is already present in the drive
before any threshold effect.

### End-to-end V2

With learned V2 (instead of oracle), dampening survives robustly (V2
converges to ~44% state accuracy; dampening kernel is unchanged).
Under the P4 sharpening condition, learned V2 causes the profile to
flip to dampening — the sharpening kernel is not a stable attractor
when predictions are imprecise.

---

## 4. Representational Metrics Used

| Metric | What it measures | Key for |
|---|---|---|
| Peak gain (ON/OFF) | Response at preferred channel with/without feedback | Dampening (gain ↓) |
| Population FWHM | Width of the population response bump | Sharpening (FWHM ↓) |
| M6: Local d' (popvec) | Orientation discrimination via 1D circular decode | Noise suppression |
| M7: Match-vs-near-miss (LogReg) | Trained linear decoder, δ∈{3,5,10}° | True sharpening |
| M8: Time-resolved | Per-timestep peak/FWHM/flank response | Temporal dynamics |
| M9: Normalized energy by distance | Per-channel suppression in expected/surround/far bins | Suppression geometry |
| Pre-rect drive FWHM | Drive width before rectified softplus | Artifact check |
| SI curve | Suppression at stimulus channel across offsets | Profile shape |
| Template manipulation | True/wrong/random/uniform templates | Peakedness vs correctness |

---

## 5. Configs and Results Location

### Key configs

| Config | Description |
|---|---|
| `exp_deviance.yaml` | Dampening: sensory off L2/3, mismatch on, oracle V2 |
| `exp_sensory_control.yaml` | Control: sensory on L2/3, oracle V2 |
| `exp_ambig_p4.yaml` | Phase 2: ambiguous competitors |
| `exp_shifted_p4.yaml` | Phase 3: shifted timing + ambiguous |
| `exp_localdisc_p4.yaml` | Phase 4: local discrimination loss + ambiguous |
| `exp_sigma{5,8,12,20}_p4.yaml` | Phase 5: oracle sigma sweep |
| `template_{true,wrong,random,uniform}.yaml` | Template manipulation experiment |
| `confound_damp_no_adapt.yaml` | No-adaptation control |
| `confound_damp_50reliable.yaml` | 50%-reliability control |
| `e2e_deviance.yaml` | End-to-end learned V2, dampening |

### Results directories

| Directory | Contents |
|---|---|
| `results/deviance_2x2/` | Dampening + control + ablation (3 seeds each) |
| `results/template_manipulation/` | Template manipulation (4 modes × 3 seeds) |
| `results/confounds/` | Adaptation-off + 50%-reliability (dampening + P4) |
| `results/e2e/` | End-to-end learned V2 |
| `results/hardening/` | Hardened dampening + P4 (post-bugfix) |
| `results/sharpening/` | Original P3/P4 runs |
| `results/phase2_ambig/` | Phase 2: ambiguous competitors |
| `results/phase3_shifted/` | Phase 3: shifted timing |
| `results/phase4_localdisc/` | Phase 4: local discrimination loss |
| `results/phase5_sigma/` | Phase 5: oracle sigma sweep |

---

## 6. What This Means

The model supports a single defensible claim:

> In a minimal V1–V2 inhibitory feedback model with laminar populations,
> dampening (suppression at the predicted orientation) is the default and
> only robust regime. It emerges from global activity minimization using
> any peaked top-down template — not from prediction error cancellation.
> Representational sharpening does not emerge from any tested combination
> of task modifications, loss functions, prediction timing, or template
> width changes. The SOM-only inhibitory architecture is insufficient for
> sharpening; it would require additional excitatory or multiplicative
> gain mechanisms.

This is a **negative-constraint result**: it tells us what dampening IS NOT
(not diagnostic of predictive coding) and what this architecture CANNOT DO
(cannot produce sharpening). Both findings are scientifically informative,
because the field currently treats dampening as evidence for prediction
error and debates whether sharpening or dampening is the "true" mechanism
— this model suggests the answer depends on the circuit architecture.
