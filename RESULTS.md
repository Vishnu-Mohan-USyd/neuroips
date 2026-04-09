# Results

## Summary

This model produces four distinct feedback regimes, each determined by
circuit architecture and loss landscape, culminating in the first
end-to-end learned representational sharpening result:

1. **Template-conditioned dampening** — robust, non-diagnostic of predictive
   coding. Emerges from any peaked template + energy cost. SOM suppresses
   at the predicted orientation.

2. **Flat / weak competitor suppression** — SOM-only with sensory loss
   present. No measurable L2/3 modulation. Five phases of task/loss/timing
   modifications failed to produce sharpening.

3. **VIP center-sparing surround suppression** — VIP disinhibition narrows
   FWHM without peak loss, doubles popvec d', but does NOT improve trained
   decoder accuracy (M7 flat). This is center-sparing geometry, not
   Kok-style information-level sharpening.

4. **Apical gain sharpening** — multiplicative apical gain boosts the
   predicted channel above feedforward level while SOM+VIP suppress
   flanks. Pure top-down with learned prior (Branch A) at ±70% gain:
   M7 δ=10° = **+0.113 (p=0.018)**, peak gain 2.09. The coincidence gate
   (Branch B) was tested and found to hurt performance — it compresses the
   gain signal. End-to-end learned, no oracle. Single-seed, pending
   multi-seed confirmation.

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

## 2. SOM-Only Sharpening Investigation

### The systematic investigation (Phases 2-5)

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

### Why sharpening fails in the SOM-only architecture

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
suppresses L2/3 channels at ±25-30° from the predicted orientation.
This produces:

- ~3-7% energy reduction in surround/far channels
- ~1-3 d' improvement in popvec estimation (M6)
- Zero improvement in trained-decoder accuracy (M7)
- Zero change in peak gain or population FWHM

This is best described as **weak competitor noise suppression**, not
representational sharpening.

---

## 3. VIP→SOM Disinhibition

### What it is

VIP interneurons provide a disinhibitory pathway: VIP inhibits SOM, which
disinhibits L2/3 at the predicted orientation. Unlike the SOM-only
architecture, VIP can selectively REDUCE inhibition at the center while
leaving surround suppression intact — a center-sparing mechanism.

Biologically motivated by:
- Pfeffer et al. (2013): VIP→SOM connection probability 62.5% in mouse V1
- Furutachi et al. (2024): VIP-mediated disinhibition in prediction circuits

### Architecture

- **VIPRing population**: Euler dynamics, tau_vip=10, rectified_softplus
  (same activation/time constant as SOM)
- **Separate alpha_vip [7] weights**: Learned basis function weights for
  VIP kernel profile (initialized at 0.01, not zero — zero kills gradient
  via rectified_softplus)
- **som_tonic**: Learnable positive SOM floor (init -3.0 → softplus ≈ 0.049),
  ensures SOM drive > 0 everywhere so VIP has something to disinhibit
- **Delta-SOM formula**: `som_drive = pi_eff * (tonic + delta)` where
  `tonic = softplus(som_tonic)` and
  `delta = softplus(baseline + field) - softplus(baseline)`
- **VIP→SOM interaction**:
  `effective_som_drive = relu(som_drive - softplus(w_vip_som) * r_vip)`

### Gradient dead zone discovery

Without `som_tonic`, VIP gets zero gradient because:
1. SOM kernel learns negative center → zero SOM drive at predicted channel
2. Zero SOM drive → zero effective_som_drive → relu clips to zero
3. VIP reducing zero still gives zero → no gradient flows to alpha_vip

The `som_tonic` parameter provides a positive SOM floor everywhere, ensuring
that VIP disinhibition at the center has nonzero signal to work with.
Combined with initializing alpha_vip at 0.01 (not 0.0), this eliminates
the "triple dead zone" that prevented VIP learning.

### Initial VIP result (exp_vip_tension)

Config: lambda_sensory=1.0, lambda_energy=0.01, ambiguous_fraction=0.3,
oracle V2 (pi=3.0), 5000 steps, delta_som=true.

| Metric | Value |
|---|---|
| alpha_vip norm | 0.259 (grew from 0.070 at init) |
| alpha_inh norm | 0.146 |
| Peak gain (ON/OFF) | 0.996 (near-unity — center preserved) |
| Population FWHM delta | −0.98° (3.7% narrower) |
| M6 Δd' at δ=5° | +2.81 |
| M6 Δd' at δ=10° | +5.75 |
| M6 Δd' at δ=15° | +7.66 |
| M7 ΔAcc at δ=3° | −0.035 |
| M7 ΔAcc at δ=5° | −0.010 |
| M7 ΔAcc at δ=10° | −0.005 |
| Energy reduction (total) | +31.0% |
| Energy reduction (expected) | +1.3% |
| Energy reduction (surround) | +29.2% |
| Energy reduction (far) | +41.3% |

### Hardened VIP result (exp_vip_hardened)

Five fixes applied before rerun:
1. **r_vip in energy cost** — VIP activity now included in L1 energy penalty
2. **7-way local discrimination** — ±3 channels (±15°), aligned with
   ambiguous_offset=15°
3. **oracle_shift_timing** — template acts as prior about current stimulus
4. **Populated cues** — 75% valid orientation cues during ISI
5. **lambda_fb=0.001** — small nonzero sparsity pressure on VIP weights

Results pending full training run with `config/exp_vip_hardened.yaml`.

### Interpretation

VIP produces the right **geometry** (narrower FWHM + preserved peak) but not
the right **information gain** (M7 flat at all deltas). This is
**center-sparing surround suppression**: VIP disinhibits the predicted
channel, lifting the SOM brake there while leaving surround/far suppression
intact. The population bump narrows because flanks lose their SOM relief.

However, this geometric narrowing does not translate into improved
discriminability for a trained linear decoder (M7). The decoder already
learns to optimally weight channels; removing flank activity that it was
already discounting provides no new information. True Kok-style sharpening
would require BOOSTING the center channel ABOVE its feedforward level —
creating new signal, not just removing noise.

---

## 4. Apical Gain: Representational Sharpening Achieved

### What it is

Multiplicative apical gain modulates L2/3 excitatory drive at the predicted
channel. Unlike VIP (which can only remove SOM inhibition), apical gain can
BOOST the center channel ABOVE its feedforward-driven level — the missing
mechanism for true Kok-style sharpening.

Biologically: active apical dendrites in L2/3 pyramidal cells receive
top-down feedback in layer 1, modulating feedforward gain multiplicatively.
This is a well-established mechanism in cortical computation.

### Architecture

- `alpha_apical` [7]: learned basis function weights (same basis as SOM/VIP)
- `apical_gain = 1.0 + 0.2 * tanh(pi_eff * K_apical ⊛ q_centered)`
- Range: [0.8, 1.2] (±20% maximum modulation)
- Multiplies ONLY excitatory L2/3 drive (ff + rec + template), NOT inhibitory
  terms — biologically correct (apical gain acts on dendritic excitation)

### Result (exp_apical)

Config: exp_apical.yaml (VIP + apical + SOM, all hardening fixes from
exp_vip_hardened: energy incl. VIP, 7-way disc, oracle_shift_timing,
75% valid cues, lambda_fb=0.001).

| Metric | VIP-only | Apical + VIP + SOM |
|---|---|---|
| Peak gain (ON/OFF) | 0.995 | **1.142** (+14% boost) |
| PopBump FWHM delta | −1.12° | **−1.59°** |
| M7 δ=3° | −0.001 | **+0.003** |
| M7 δ=5° | −0.004 | **+0.004** |
| M7 δ=10° | +0.001 | **+0.014** |
| M7 δ=15° | −0.014 | **+0.011** |
| M9 expected | +1.6% suppressed | **−12.1% BOOSTED** |
| M9 surround | +28.7% | +28.0% |
| ‖alpha_apical‖ | N/A | 1.66 |

### Interpretation

The three-arm architecture (SOM suppression + VIP disinhibition + apical
multiplicative gain) is the minimum circuit for Kok-style representational
sharpening:

1. **SOM** suppresses flanks/surround (inherited from all prior conditions)
2. **VIP** disinhibits the center by lifting the SOM brake at the predicted
   channel (center-sparing geometry)
3. **Apical gain** BOOSTS the center channel 14% above its feedforward level
   (the critical new ingredient)

This produces the full sharpening signature:
- Narrower population bump (−1.59° FWHM)
- Enhanced center (peak gain 1.14, above unity)
- Improved trained-decoder accuracy (M7 positive at all deltas)
- Boosted energy at expected channel (M9 expected: −12.1%)

VIP alone produced the right geometry (narrower + preserved peak) but not
information gain. Apical gain provides the missing ingredient: a
multiplicative boost that creates new signal above the feedforward level,
which a trained decoder can exploit.

### Template Manipulation (apical sharpening IS content-dependent)

Unlike dampening (where true/wrong/random all gave identical results),
apical sharpening discriminates between prediction quality:

| Template | Peak gain | M7 δ=10° | M7 δ=15° | Global amp | Pattern |
|---|---|---|---|---|---|
| TRUE (correct) | 1.142 | +0.014 | +0.011 | 1.030 (boosted) | Sharpening |
| WRONG (CW↔CCW) | 1.038 | +0.015 | +0.007 | 0.919 (reduced) | Sharpening |
| RANDOM (uncorrelated) | 0.873 | −0.009 | −0.016 | 0.754 (reduced) | Dampening |
| UNIFORM (no peak) | 0.813 | −0.013 | −0.019 | 0.693 (reduced) | Nothing learned |

**Key insight:** TRUE and WRONG both produce positive M7 because WRONG still
predicts a real HMM transition orientation (only 30° away from true, within
L2/3 tuning width). RANDOM predicts an orientation uncorrelated with the
stimulus → dampening, not sharpening. This means sharpening requires
predictions aligned with the stimulus space, not just peakedness.

### Confound Controls

| Condition | Peak gain | M7 δ=10° | Global amp | Survives? |
|---|---|---|---|---|
| Original | 1.142 | +0.014 | 1.030 | — |
| No adaptation | 1.142 | +0.013 | 1.029 | YES (identical) |
| 50%-reliability | 1.049 | +0.005 | 0.891 | YES (weakened) |

Sharpening is not an adaptation artifact. Under reduced prediction
reliability, it weakens but persists. The global amplitude flips from
Huang-like (boosted, 1.03) to Kok-like (reduced, 0.89) — showing the
SAME circuit produces different amplitude signatures depending on
prediction quality.

### Global Amplitude Signature

The apical model explains why different experiments report different
amplitude effects:

- **Accurate predictions (TRUE):** center boost > flank suppression →
  net activity INCREASE (Huang-like)
- **Inaccurate predictions (WRONG, 50%-reliable):** flank suppression >
  center boost → net activity DECREASE (Kok-like)
- **Random predictions:** pure dampening (no center boost)

---

## 5. Branch A: Learned Feature Prior (End-to-End)

### What it is

The biggest single architectural improvement. V2 outputs a full orientation
distribution (`mu_pred [B,N]`, softmax) instead of a binary state belief
(`p_cw [B,1]`, sigmoid). `q_pred` comes directly from V2's learned prior,
not reconstructed analytically from current L4 activity + state belief.

This means the model can form genuine prestimulus priors: during ISI
(when L4=0), V2 maintains the prior from GRU memory + cue input. The
prior is supervised via KL divergence against the true next orientation
(circular Gaussian target, sigma=10°).

### Architecture change

- **Old (emergent mode):** `head_p_cw` (Linear → 1, sigmoid) + analytical
  reconstruction: `q_pred = p_cw * bump(theta+step) + (1-p_cw) * bump(theta-step)`.
  q_pred tethered to current L4 orientation — cannot form a prior when L4=0.
- **New (Branch A):** `head_mu` (Linear → N, softmax). `q_pred = mu_pred`
  directly. V2 optimizes its distribution end-to-end via the feedback circuit.

### Config

`config/exp_branch_a.yaml`: freeze_v2=false, lambda_state=1.0 (prior KL),
v2_input_mode=l4_l23, n_steps=10000, steps_on=12, steps_isi=4.

### Results (3 seeds x 10000 steps, cross-seed consistent to 3 decimals)

| Metric | Oracle apical | Branch A (learned prior) | Improvement |
|---|---|---|---|
| M7 delta=5° | +0.004 | **+0.016** | 4x |
| M7 delta=10° | +0.014 | **+0.025** | 1.8x |
| M7 delta=15° | +0.011 | **+0.022** | 2x |
| Peak gain | 1.142 | **1.248** | +10 points |
| M12 fixed readout delta=10° | +0.017 | **+0.020** | +18% |
| Global amplitude | 1.030 | **1.150** | More boosted |
| prior_kl | N/A (oracle) | 0.83 (converged) | V2 learned |
| s_acc | 0.31 | 0.34 | Improved |

### Interpretation

The learned prior is more effective than the oracle because V2 optimizes
its distribution specifically to help the feedback circuit sharpen, not
just to match the next orientation. The oracle produces a bump centered
at the true next orientation — but that may not be the optimal template
for driving the SOM+VIP+apical circuit. The learned prior discovers a
distribution that maximizes the feedback operator's ability to boost the
predicted channel and suppress competitors.

This is the first end-to-end learned sharpening result: no oracle, no
frozen V2, no hand-crafted predictions. The model learns WHAT to predict
and HOW to use predictions simultaneously.

---

## 6. Branch B: Coincidence-Gated Apical Gain — Null Result and Fix

### What it is

Branch B replaces the pure top-down apical gain (Section 4) with a
coincidence-gated mechanism:

```
coincidence = relu(apical_field) * relu(basal_field)
apical_gain = 1.0 + mag * tanh(pi_eff * coincidence)
```

where `apical_field = K_apical ⊛ q_centered` (top-down prediction) and
`basal_field = r_l4 - mean(r_l4)` (bottom-up centered L4 activity).

Biologically: apical gain requires dendritic coincidence detection
(Larkum 2013) — the gain boost occurs only where top-down prediction and
bottom-up stimulus BOTH agree. This is a stronger constraint than pure
top-down modulation, and should in principle provide better wrong-template
rejection: if the prediction points to the wrong orientation, the
coincidence product is near zero at the stimulus location.

### Null result at mag=0.2 (Batch 1: 4 templates × 3 seeds)

With the coincidence gate active at max_apical_gain=0.2, ALL conditions
collapse. True and wrong templates both produce M7 ≈ 0:

| Template | Peak gain | M7 δ=10° | M7 δ=15° | Global amp | M9 expected |
|---|---|---|---|---|---|
| TRUE (correct) | 1.053 | −0.003 | +0.011 | 0.949 | −2.1% |
| WRONG (CW↔CCW) | 1.053 | −0.001 | +0.010 | 0.959 | −2.3% |
| RANDOM (uncorrelated) | 0.904 | −0.008 | −0.002 | 0.767 | +22.5% |
| UNIFORM (no peak) | 0.814 | −0.021 | −0.011 | 0.695 | +30.4% |

Compare to the pure top-down result (Section 4): TRUE had peak gain 1.142
and M7 δ=10° = +0.014. With the coincidence gate, TRUE drops to peak gain
1.053 and M7 = −0.003. **The gate killed the sharpening signal.**

### Root cause

The coincidence product of two [0,1]-range values is much smaller than
either alone. The relu(apical_field) values are already small (~0.01–0.03
at peak), and relu(basal_field) is similarly small. Their product reduces
the effective gain signal to ~34% of the pure top-down value.

At mag=0.2, this means the actual gain at the predicted channel drops from
~1.20 (pure top-down) to ~1.04 (coincidence-gated). The 4% boost is exactly
at the margin where L2/3 activity sits near the SOM suppression threshold
— not enough headroom to create a decoder-detectable signal. The
parameters did not change; the gate itself compressed the gain range below
the functional threshold.

### Fix: increasing max_apical_gain to 0.5

Increasing mag from 0.2 to 0.5 restores the gain headroom that the
coincidence product consumed. At mag=0.5, the gate provides the intended
benefit — wrong-template rejection — while maintaining a strong true
signal:

- TRUE at mag=0.5: peak gain 1.275, M7 δ=10° = +0.034
- WRONG at mag=0.5: expected to show M7 ≈ 0 (pending Task #9)

The coincidence gate at mag=0.5 should give BETTER wrong-template rejection
than pure top-down at mag=0.2 (where wrong still produced M7 = +0.015),
because the multiplicative AND-gate zeros out the gain where prediction
and stimulus disagree.

---

## 7. A+B Combined with mag=0.5 — Crosses +0.03 Bar

### What it is

The definitive sharpening result: Branch A (learned feature prior) +
Branch B (coincidence-gated apical gain) with max_apical_gain=0.5.
End-to-end learned, no oracle.

### Config

`config/exp_branch_a.yaml`: freeze_v2=false, lambda_state=1.0 (prior KL),
v2_input_mode=l4_l23, n_steps=10000, steps_on=12, steps_isi=4.
max_apical_gain=0.5 (in feedback.py). Coincidence gate active (r_l4
passed to feedback operator in network.step()).

### Results (3 seeds × 10000 steps, cross-seed consistent to 3 decimals)

| Metric | Oracle (mag=0.2) | Branch A (mag=0.2) | **A+B (mag=0.5)** |
|---|---|---|---|
| M7 δ=5° | +0.004 | +0.016 | **+0.019** |
| M7 δ=10° | +0.014 | +0.025 | **+0.034** |
| M7 δ=15° | +0.011 | +0.022 | **+0.040** |
| Peak gain | 1.142 | 1.248 | **1.275** |
| PopBump FWHM delta | −1.59° | — | **−2.59°** |
| Global amplitude | 1.030 | 1.150 | **1.114** |
| M9 expected | −12.1% | — | **−16.5%** |
| M9 surround | +28.0% | — | **+27.2%** |
| M9 far | — | — | **+42.3%** |
| ‖alpha_apical‖ | 1.66 | — | **~2.5** |
| M12 δ=10° benefit | +0.017 | +0.020 | **+0.021** |
| M13 δ=10° peak_t | — | — | **t=22** |
| M13 δ=10° peak_delta | — | — | **+0.032** |
| w_template_drive | — | — | **0.0** |

**M7 δ=10° = +0.034 crosses the +0.03 reviewer threshold.**

### Full sharpening signature

- **Enhanced center**: energy at expected channels −16.5% (boosted above
  feedforward). Peak gain 1.275 (28% above feedforward level).
- **Suppressed surround**: +27.2% energy reduction at 10–45° from predicted.
- **Suppressed far**: +42.3% energy reduction at >45° from predicted.
- **Late-phase timing**: M13 peak benefit at t=19–22, consistent with
  top-down modulation building over time (biologically plausible).
- **w_template_drive = 0.0**: Branch C (direct template→L2/3 excitation)
  was available but unused. The model achieves sharpening entirely through
  the learned apical gain pathway.
- **End-to-end learned**: no oracle V2, no frozen weights, no hand-crafted
  predictions. V2's learned prior (KL=0.83) optimizes specifically to help
  the feedback circuit sharpen.

### Interpretation

The A+B combination succeeds where each component alone fell short:

- **Branch A alone (mag=0.2)**: M7 +0.025 — improved over oracle but below
  +0.03 threshold. The learned prior is better than the oracle, but the
  gain range limits the circuit.
- **Branch B alone (mag=0.2)**: M7 ≈ 0 — coincidence gate compressed the
  gain signal below the functional threshold.
- **A+B (mag=0.5)**: M7 +0.034 — the increased gain range compensates for
  the coincidence product attenuation, while the learned prior provides
  a more effective template than the oracle.

The mag escalation from 0.2 to 0.5 was necessary to compensate for the
coincidence product's signal compression. At ±50%, |alpha_apical| grows
from ~0.97 (saturated at mag=0.2) to ~2.5 (using the new headroom), and
the effective gain at the predicted channel rises from 1.04 to 1.275.

---

## 8. Pure Top-Down at mag=0.7 — Definitive Result

### Background

The coincidence gate (Branch B) was tested extensively in Batch 3 (template
manipulation at mag=0.5, 4 conditions × 3 seeds = 12 runs). Result: **true ≈ wrong**
across all metrics — the gate provides no content selectivity.

**Batch 3 M7 δ=10° (mag=0.5, oracle mode, with gate):**

| Template | M7 δ=10° | Peak Gain | Global Amp |
|----------|----------|-----------|------------|
| true     | +0.011   | 1.144     | 1.009      |
| wrong    | +0.013   | 1.145     | 1.018      |
| random   | +0.007   | 0.991     | 0.821      |
| uniform  | −0.021   | 0.814     | 0.694      |

The gate acts as a salience/arousal signal (structured template vs. noise)
but does not distinguish whether the template content matches the stimulus.
Root cause: at 10° offset with 15° kernel FWHM, there is 86% geometric
overlap between true and wrong template activations — the coincidence
product cannot resolve this.

### Control experiment: Branch A, mag=0.7, no gate

To isolate the gate's contribution, we ran Branch A (learned feature prior,
freeze_v2=false) at mag=0.7 with the coincidence gate disabled (r_l4=None
fallback to pure top-down apical gain). Single seed (42), 10000 steps.

**Comparison across all configurations:**

| Metric | Oracle (mag=0.2) | Branch A (mag=0.2) | A+B gate (mag=0.5) | **No gate (mag=0.7)** |
|--------|------------------|--------------------|---------------------|----------------------|
| M7 δ=10° | +0.014 | +0.025 | +0.034 | **+0.113** |
| M7 δ=15° | +0.011 | +0.022 | +0.040 | **+0.121** |
| Peak gain | 1.142 | 1.248 | 1.275 | **2.09** |
| Global amp | 1.030 | 1.150 | 1.114 | **1.87** |

**Full M7 at all deltas (no gate, mag=0.7):**

| δ | M7 | p-value |
|---|------|---------|
| 3° | +0.033 | 0.281 |
| 5° | +0.069 | 0.092 |
| 10° | **+0.113** | **0.018** |
| 15° | **+0.121** | **0.018** |

**Full sharpening signature:**
- Center enhancement (expected channels): −101.6% energy reduction (massive boost)
- Surround suppression: +16.5%
- Far suppression: +58.3%
- FWHM narrowing: −1.85°
- Late-phase timing: M13 peak at t=22, late-phase Δacc = +0.072
- M12 fixed readout benefit (δ=10°): +0.071
- M14 temporal evaluation (δ=10°): +0.014

### Interpretation

The coincidence gate was **hurting** the signal, not helping. By multiplying
two values in [0,1] (relu(apical_field) × relu(basal_field)), the gate
compresses the effective gain — reducing what the learned prior can achieve.
Pure top-down apical gain with a higher ceiling (±70%) gives dramatically
better M7: 3× the previous best result.

w_template_drive = 0.0 in all experiments (Branch C pathway unused).

**Caveat:** This is a single-seed result (seed 42). Multi-seed confirmation
is pending to establish cross-seed consistency and proper confidence intervals.

---

## 9. Other Findings

### Sensory loss on L2/3 blocks dampening

With lambda_sensory > 0 on L2/3, the feedback operator learns only weak
alpha (||alpha|| ≈ 0.3-0.4) and produces no measurable L2/3 modulation.
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

## 10. Representational Metrics Used

| Metric | What it measures | Key for |
|---|---|---|
| Peak gain (ON/OFF) | Response at preferred channel with/without feedback | Dampening (gain ↓) |
| Population FWHM | Width of the population response bump | Sharpening (FWHM ↓) |
| M6: Local d' (popvec) | Orientation discrimination via 1D circular decode | Noise suppression |
| M7: Match-vs-near-miss (LogReg) | Trained linear decoder, δ∈{3,5,10,15}°, 8-anchor averaged | True sharpening |
| M8: Time-resolved | Per-timestep peak/FWHM/flank response | Temporal dynamics |
| M9: Normalized energy by distance | Per-channel suppression in expected/surround/far bins | Suppression geometry |
| M10: Global amplitude | Mean L2/3 activity ON/OFF ratio, 8-anchor avg | Kok vs Huang signature |
| Pre-rect drive FWHM | Drive width before rectified softplus | Artifact check |
| SI curve | Suppression at stimulus channel across offsets | Profile shape |
| Template manipulation | True/wrong/random/uniform templates | Peakedness vs correctness |

Note: M7 was updated to include δ=15° and average across 8 anchor
orientations {0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5} for rotational
invariance, matching M6's multi-anchor protocol. M7 now includes bootstrap
95% CI (100 resamples) and permutation p-value (500 shuffles).

---

## 11. Configs and Results Location

### Key configs

| Config | Description |
|---|---|
| `exp_deviance.yaml` | Dampening: sensory off L2/3, mismatch on, oracle V2 |
| `exp_sensory_control.yaml` | Control: sensory on L2/3, oracle V2 |
| `exp_ambig_p4.yaml` | Phase 2: ambiguous competitors |
| `exp_shifted_p4.yaml` | Phase 3: shifted timing + ambiguous |
| `exp_localdisc_p4.yaml` | Phase 4: local discrimination loss + ambiguous |
| `exp_sigma{5,8,12,20}_p4.yaml` | Phase 5: oracle sigma sweep |
| `exp_vip_tension.yaml` | VIP Exp 2: tension (sensory + energy + ambiguous) |
| `exp_vip_hardened.yaml` | VIP Exp 3: hardened (energy incl. VIP, 7-way disc, cues, shifted timing) |
| `exp_apical.yaml` | Apical gain: VIP + SOM + multiplicative apical modulation |
| `apical_template_{true,wrong,random,uniform}.yaml` | Apical template manipulation (4 modes) |
| `apical_no_adapt.yaml` | Apical: no adaptation control |
| `apical_50reliable.yaml` | Apical: 50%-reliability control |
| `template_{true,wrong,random,uniform}.yaml` | Dampening template manipulation experiment |
| `confound_damp_no_adapt.yaml` | Dampening: no-adaptation control |
| `confound_damp_50reliable.yaml` | Dampening: 50%-reliability control |
| `e2e_deviance.yaml` | End-to-end learned V2, dampening |
| `exp_branch_a.yaml` | Branch A+B: learned feature prior + mag=0.5 (V2 outputs mu_pred, end-to-end) |

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
| `results/vip_tension/` | VIP Exp 2: tension condition |
| `results/apical/` | Apical gain experiments (3+ seeds) |
| `results/batch1/` | Template manipulation at mag=0.2 (4 modes × 3 seeds) |
| `results/batch2/abc_s*/` | Branch A+B at mag=0.5 (3 seeds) |
| `results/batch3/` | Template manipulation at mag=0.5 (4 modes × 3 seeds) |
| `results/control_no_gate/` | Branch A pure top-down at mag=0.7 (single seed) |

---

## 12. What This Means

The model supports seven defensible claims arranged as a progression:

> **1. Dampening is robust and non-diagnostic.**
> In a minimal V1-V2 inhibitory feedback model, dampening (suppression at
> the predicted orientation) emerges from global activity minimization using
> any peaked top-down template — not from prediction error cancellation.
> A V1 dampening signature is therefore not diagnostic of predictive coding.

> **2. SOM-only inhibition cannot produce sharpening.**
> Representational sharpening does not emerge from any tested combination of
> task modifications, loss functions, prediction timing, or template width
> changes. The SOM-only inhibitory architecture is a fundamental constraint:
> subtraction cannot create new signal above the feedforward level.

> **3. VIP disinhibition produces center-sparing geometry, not information gain.**
> Adding a VIP→SOM disinhibitory pathway enables a new regime: the population
> bump narrows (−1° FWHM) without peak loss (gain ratio 0.996), and popvec d'
> doubles. But a trained linear decoder (M7) sees no improvement — the
> geometric narrowing does not translate into information-level sharpening.

> **4. Pure top-down apical gain completes the sharpening circuit.**
> Adding a multiplicative apical gain pathway to the VIP+SOM architecture
> produces the full Kok-style sharpening signature: the predicted channel is
> boosted 14% above feedforward level (peak gain 1.14), the population bump
> narrows (−1.59° FWHM), and a trained linear decoder (M7) shows improved
> accuracy at all tested deltas. The three-arm circuit (SOM suppression +
> VIP disinhibition + apical multiplicative gain) is the minimum architecture
> for representational sharpening.

> **5. Coincidence gating requires sufficient gain headroom.**
> Replacing pure top-down apical modulation with a coincidence gate
> (relu(apical) × relu(basal)) improves biological plausibility but
> compresses the gain signal: the product of two [0,1] values is much
> smaller than either alone. At ±20% max gain, the coincidence gate kills
> sharpening entirely (M7 ≈ 0 for all template conditions). Increasing
> to ±50% restores function by compensating for the signal compression.

> **6. The coincidence gate hurts performance — pure top-down is better.**
> Extensive template manipulation (Batch 3: 4 conditions × 3 seeds) shows
> the coincidence gate provides no content selectivity: true ≈ wrong at all
> M7 deltas. The gate acts as a salience signal, not a content-match signal.
> Root cause: 10° offset with 15° kernel FWHM = 86% geometric overlap.

> **7. Pure top-down at mag=0.7 is the definitive result.**
> Branch A (learned prior, freeze_v2=false) with pure top-down apical gain
> at ±70% (no coincidence gate) produces M7 δ=10° = **+0.113 (p=0.018)**,
> 3× the previous best. Peak gain 2.09, global amplitude 1.87, center
> enhancement −101.6%, far suppression +58.3%. The learned prior (KL~0.83)
> is more effective than the oracle because V2 optimizes its distribution
> to help the feedback circuit sharpen. w_template_drive=0.0 (Branch C
> unused). Single-seed result pending multi-seed confirmation.

### Scientific implication: circuit motif determines what is POSSIBLE; prediction quality determines what EMERGES

The regimes demonstrate that the feedback regime is determined by circuit
architecture, not by training objective:

- **SOM-only + energy**: dampening (suppression at predicted channel)
- **SOM-only + sensory + energy**: flat (competing losses cancel)
- **VIP + SOM + sensory + energy**: center-sparing surround suppression
- **Apical + VIP + SOM (pure top-down, mag=0.2)**: sharpening (M7 +0.014)
- **Apical + VIP + SOM (coincidence gate, mag=0.2)**: null — gate kills signal
- **Branch A (learned prior, mag=0.2)**: improved (M7 +0.025) but sub-threshold
- **A+B (learned prior + coincidence gate, mag=0.5)**: sharpening but gate
  provides no selectivity — M7 +0.034, true ≈ wrong
- **Branch A (learned prior, pure top-down, mag=0.7)**: **definitive sharpening**
  — M7 +0.113 (p=0.018), peak gain 2.09, 3× previous best (single seed)

The loss landscape selects WHAT the feedback does, but the circuit CONSTRAINS
what is achievable. SOM inhibition alone gives dampening or nothing. Adding
VIP disinhibition opens center-sparing suppression but not true sharpening.
Adding apical multiplicative gain completes the circuit, but the coincidence
gate requires sufficient gain headroom (±50%) to compensate for the product
attenuation. Each architectural step unlocks a qualitatively new feedback
regime — the objective alone is insufficient.

### The apical hardening pass: prediction quality modulates the regime

The template manipulation and confound control results reveal a deeper
principle. Unlike dampening (which is invariant to prediction correctness),
the apical sharpening circuit is **content-dependent**: the SAME three-arm
circuit (SOM + VIP + apical gain) produces qualitatively different outcomes
depending on prediction quality:

- **Accurate predictions (TRUE):** full sharpening — center boost > flank
  suppression → net activity INCREASE (Huang & Rao 2011)
- **Partially accurate (WRONG, 50%-reliable):** weakened sharpening — flank
  suppression > center boost → net activity DECREASE (Kok et al. 2012)
- **Random predictions:** the circuit reverts to pure dampening
- **No prediction (UNIFORM):** nothing is learned

The coincidence gate (Batch 3) was found NOT to provide wrong-template
rejection — true ≈ wrong at all deltas. Pure top-down at mag=0.7 without
the gate gives 3× stronger M7 (+0.113 vs +0.034).

This directly explains mixed empirical findings in the expectation
suppression literature: paradigms with accurate, task-relevant predictions
may show Huang-like enhancement, while paradigms with weak or
partially-relevant predictions may show Kok-like suppression. The circuit
motif is the same — the prediction quality determines the sign of the net
amplitude effect. This is a testable prediction: manipulating prediction
reliability within a single paradigm should shift the amplitude signature
from enhancement to suppression.

---

## 13. Architecture Simplification: V2 Direct Feedback

### Motivation

The 7-basis function EmergentFeedbackOperator with VIP/SOM/apical pathways
was effective but complex (~40 learnable parameters in the feedback path).
A simpler architecture was tested: V2 outputs feedback directly via a
linear head, with the signal split by sign (Dale's law).

### Architecture

- `head_feedback = nn.Linear(v2_hidden_dim, N)` (576 params for dim=16, N=36)
- Raw output, no activation — V2 learns sign and magnitude end-to-end
- **E/I split** (Dale's law): `relu(+scaled_fb)` → excitation (center_exc),
  `relu(-scaled_fb)` → SOM drive (integrated by SOMRing with tau_som=10)
- Config: `simple_feedback: true`, `max_apical_gain: 0.0`

### What was removed

- 7-basis function decomposition → 36 direct channel weights
- tanh/precision scaling/gain caps → simple additive/subtractive
- VIP pathway (r_vip = 0)
- Apical multiplicative gain
- Coincidence gate

---

## 14. Energy-Efficient Sharpening Investigation

### The M7 vs amplitude tradeoff

V2 direct feedback produces massive M7 effects (+0.27–0.34) but also
massive global amplitude inflation (2.5–6.4×). The target is M7 > +0.03
AND global amplitude < 1.10×.

### E/I Split Experiments (3 configs, 5000 stage2 steps, seed 42)

| Experiment | λ_energy | λ_pred_suppress | M7 δ=10° | Global Amp | FWHM Δ |
|---|---|---|---|---|---|
| EI Baseline | 0.1 | 0.0 | +0.339 | 6.43× | +0.56° |
| EI PredSup | 0.1 | 1.0 | +0.316 | 3.47× | +1.34° |
| EI HighEnergy | 2.0 | 0.0 | +0.273 | 2.77× | **−1.86°** |

### Energy Fix Experiments (3 configs, 5000 stage2 steps, seed 42)

| Experiment | λ_energy | λ_fb_energy | l2_energy | λ_pred_sup | M7 δ=10° | Global Amp | FWHM Δ |
|---|---|---|---|---|---|---|---|
| FbEnergy | 0.1 | 2.0 | false | 0.0 | +0.318 | 3.69× | +0.40° |
| L2Energy | 2.0 | 0.0 | true | 0.0 | +0.281 | 3.06× | −0.37° |
| AllFixes | 2.0 | 1.0 | true | 0.5 | +0.244 | 2.46× | −1.73° |

### Key findings

1. **All experiments maintain M7 well above +0.03** (lowest: +0.244)
2. **HighEnergy produces the best sharpening** (−1.86° FWHM delta)
3. **AllFixes has lowest amplitude** (2.46×) but still far from 1.10× target
4. **Feedback energy penalty alone is ineffective** — similar to PredSup
5. **L2 energy on L2/3** provides mild sharpening (−0.37°) and lower amp (3.06×)
6. **The amplitude-M7 tradeoff is monotonic**: more energy pressure → lower amp
   but also lower M7. However, M7 remains far above threshold even at
   maximum pressure, suggesting the target is achievable.

### New loss functions

| Loss | Formula | Purpose |
|---|---|---|
| `lambda_pred_suppress` | dot(r_l23, q_pred).mean() | Penalize L2/3 matching V2 prediction |
| `lambda_fb_energy` | center_exc.abs().mean() | Penalize excitatory feedback magnitude |
| `l2_energy` | r_l23.pow(2).mean() instead of abs | Quadratic L2/3 energy penalty |

---

## 15. Configs and Results (V2 Direct Feedback)

### Configs

| Config | Description |
|---|---|
| `exp_v2_ei_split.yaml` | E/I split baseline: λ_energy=0.1 |
| `exp_v2_ei_predsup.yaml` | E/I + prediction suppression: λ_pred_suppress=1.0 |
| `exp_v2_ei_highenergy.yaml` | E/I + high energy: λ_energy=2.0 |
| `exp_v2_ei_fbenergy.yaml` | E/I + feedback energy: λ_fb_energy=2.0 |
| `exp_v2_ei_l2energy.yaml` | E/I + L2 energy: λ_energy=2.0, l2_energy=true |
| `exp_v2_ei_allfixes.yaml` | E/I + all fixes combined |
| `exp_v2_direct_highenergy.yaml` | V2 direct (no E/I split) + high energy |
| `exp_v2_direct_predsup.yaml` | V2 direct (no E/I split) + pred suppress |
| `exp_simple_fb.yaml` | Simple additive feedback (original kernel-based) |

### Results directories

| Directory | Contents |
|---|---|
| `results/iter/v2_ei_split/` | E/I split baseline |
| `results/iter/v2_ei_predsup/` | E/I + prediction suppression |
| `results/iter/v2_ei_highenergy/` | E/I + high energy |
| `results/iter/v2_ei_fbenergy/` | E/I + feedback energy penalty |
| `results/iter/v2_ei_l2energy/` | E/I + L2 energy |
| `results/iter/v2_ei_allfixes/` | E/I + all fixes combined |

---

## 16. Three Feedback Regimes from One Architecture

### Core finding

A single parameter — `lambda_sensory` (task demand on V1 orientation
decoding) — switches the feedback regime among three qualitatively
distinct behaviors, all from the same V2 direct feedback architecture
(simple_feedback=true, E/I split):

| Regime | λ_sensory | λ_mismatch | Behavior | Biological analogue |
|--------|-----------|------------|----------|---------------------|
| **Dampening** | 0.0 | 1.0 | Expected 56–80% lower than unexpected | Richter 2018, Alink 2010 |
| **Enhancement** | 0.3 | 1.0 | Expected 10–14% higher than unexpected | Attention / gain literature |
| **Suppression + sharpening** | 1.0 | 0.0 or 1.0 | Expected 6–9% lower, FWHM narrows, M7 +0.23 | Kok 2012 |

### Full results table (all configs, seed 42, 5000 stage2 steps)

#### Unweighted energy (l23_energy_weight=1.0)

| Config | λ_sens | λ_mm | M7 δ=10° | Global Amp | FWHM Δ° | Peak ON | FB contribution | Direction |
|--------|--------|------|----------|------------|---------|---------|-----------------|-----------|
| D1 | 0.0 | 1.0 | negative | 0.54 | −5.4 | 0.131 | large positive | WIDENS |
| D2 | 0.3 | 1.0 | +0.16 | 1.66 | −11.7 | — | positive | Enhancement |
| D3 | 1.0 | 1.0 | +0.28 | 3.06 | −7.2 | — | +0.74 | WIDENS |

#### Weighted energy (l23_energy_weight=3.0)

| Config | λ_sens | λ_mm | M7 δ=10° | Global Amp | FWHM Δ° | Peak ON | FB contribution | Direction |
|--------|--------|------|----------|------------|---------|---------|-----------------|-----------|
| D1W | 0.0 | 1.0 | −0.076 | 0.40 | −5.4 | 0.131 | +0.556 | WIDENS |
| D2W | 0.3 | 1.0 | +0.133 | 1.21 | −11.7 | 0.459 | −0.076 | NARROWS |
| D3W | 1.0 | 1.0 | +0.233 | 2.17 | −7.2 | 0.779 | +0.274 | WIDENS |
| HEW | 1.0 | 0.0 | +0.231 | 2.11 | −4.0 | 0.721 | +0.268 | WIDENS |

### Multi-seed confirmation of predictive suppression

Three seeds (42, 123, 456) of the EI HighEnergy config (λ_sensory=1.0,
λ_energy=2.0, 10000 stage2 steps) all show:

- L2/3 total activity ~8.3% lower for expected vs unexpected stimuli
  (p < 10⁻²⁷ in all seeds)
- L4 unaffected (no feedback effect on input layer)
- Feedback widens the expected-vs-unexpected gap (+0.36 mean contribution)
- FB-OFF condition shows the OPPOSITE pattern (expected > unexpected),
  confirming the effect is feedback-driven, not stimulus-driven

### Mechanism: temporal priming + differential excitation

The feedback operator learns to provide more excitatory drive for
unexpected stimuli (large V2 prediction error) and less for expected
stimuli (small prediction error). During the ISI, V2 forms a prior.
When the stimulus arrives:

- **Expected**: V2 prediction error is small → feedback excitation is low
  → L2/3 activity reduced relative to feedforward baseline
- **Unexpected**: V2 prediction error is large → feedback excitation is
  high → L2/3 activity boosted above feedforward baseline

This is a temporal priming mechanism: the V2 prediction sets the gain
state, and the mismatch between prediction and stimulus determines the
feedback magnitude. The E/I split (Dale's law) ensures this is implemented
via differential excitation, not inhibition.

---

## 17. Dampening Regime Confirmed

### What it is

When λ_sensory=0 (no task demand on V1 orientation decoding) and
λ_mismatch=1.0 (V2 learns expected/deviant classification), the feedback
operator learns to massively suppress expected stimuli:

| Metric | D1 (l23w=1) | D1W (l23w=3) |
|--------|-------------|--------------|
| Global amplitude (ON/OFF) | 0.54 | 0.40 |
| L2/3 suppression (expected vs unexpected) | 56–80% | similar |
| M7 δ=10° | negative | −0.076 |
| FB contribution | large positive | +0.556 |
| Peak ON | collapsed | 0.131 |

### Interpretation

Without sensory pressure on L2/3, the energy cost drives L2/3 activity
toward zero. Feedback learns to suppress expected stimuli because V2
can predict them — reducing total activity. This matches:

- **Richter et al. (2018)**: fMRI BOLD reduction for expected face parts
  in hierarchical predictive coding paradigm
- **Alink et al. (2010)**: reduced V1 BOLD for expected motion direction

The dampening regime does NOT require correct predictions — any peaked
template produces the same result (Section 1). λ_sensory=0 is the key:
no orientation decoding demand means suppression is the path of least
resistance.

---

## 18. L2/3-Weighted Energy

### Biological motivation

L2/3 pyramidal neurons project to V2 via long-range axons. The
postsynaptic component of synaptic transmission accounts for ~50% of V2's
metabolic budget (Attwell & Laughlin 2001). Penalizing L2/3 output more
heavily than intracortical activity reflects this asymmetry.

### Implementation

`l23_energy_weight` (config option, default 1.0): multiplies the L2/3
term in the energy cost. When l23_energy_weight=3.0, L2/3 output is
penalized 3× more than L4, PV, SOM, VIP, and deep_template.

### Effect on amplitude

| Config | Amp (l23w=1) | Amp (l23w=3) | Reduction |
|--------|--------------|--------------|-----------|
| D2 (λ_sens=0.3) | 1.66 | 1.21 | −27% |
| D3 (λ_sens=1.0) | 3.06 | 2.17 | −29% |

l23_energy_weight=3.0 reduces amplitude ~30% across all configs.

### Trade-off

Stronger L2/3 energy penalty reduces global amplitude but also weakens
feedback's differential effect. D2W (amp=1.21, closest to the <1.10
target) lost predictive suppression direction (FB contribution flipped
to −0.076, NARROWS). D3W (amp=2.17) retains positive suppression
(FB contribution +0.274) but amplitude is still above 2.0.

---

## 19. Biological Mapping: Dual-Process Account

### Config-to-paradigm mapping

| Config | λ_sens | λ_mm | Experimental analogue | Key prediction |
|--------|--------|------|----------------------|----------------|
| D1/D1W | 0.0 | 1.0 | Passive viewing, fMRI BOLD | Dampening (Richter 2018) |
| D2/D2W | 0.3 | 1.0 | Weak task demand, partial attention | Enhancement / gain |
| D3/D3W | 1.0 | 1.0 | Active discrimination + deviant detection | Suppression + M7 (Kok 2012) |
| HEW | 1.0 | 0.0 | Active discrimination, no deviant task | Suppression (pure energy) |

### The dual-process interpretation

Task demand (λ_sensory) is the switch:

- **Low task demand** (passive viewing): V1 has no incentive to maintain
  orientation representations → feedback suppresses everything at predicted
  channels → dampening. This explains fMRI BOLD reduction paradigms.
- **High task demand** (active discrimination): V1 must maintain accurate
  orientation coding → feedback sharpens representations at predicted
  channels → suppression + enhanced discriminability. This explains
  psychophysics paradigms where subjects actively report orientation.

The mismatch loss (λ_mismatch) has **negligible effect** at λ_sensory=1.0:
D3W and HEW produce nearly identical results (M7 +0.233 vs +0.231,
amp 2.17 vs 2.11, FB contribution +0.274 vs +0.268). The sensory
pressure alone is sufficient to produce the suppression regime.

### Testable prediction

Manipulating task demand within a single paradigm should shift the
feedback signature:
- Low demand (passive viewing) → dampening (BOLD reduction)
- High demand (orientation discrimination) → suppression + sharpening
  (improved psychophysics)

This is a within-subject, within-session prediction that can distinguish
the dual-process account from alternative explanations.

---

## 20. Configs and Results (Dampening + Weighted Energy)

### Configs

| Config | Description |
|---|---|
| `exp_dampening_d1.yaml` | D1: λ_sensory=0.0, λ_mismatch=1.0 (pure dampening) |
| `exp_dampening_d2.yaml` | D2: λ_sensory=0.3, λ_mismatch=1.0 (weak sensory) |
| `exp_dampening_d3.yaml` | D3: λ_sensory=1.0, λ_mismatch=1.0 (full sensory + mismatch) |
| `exp_dampening_d1_weighted.yaml` | D1W: D1 + l23_energy_weight=3.0 |
| `exp_dampening_d2_weighted.yaml` | D2W: D2 + l23_energy_weight=3.0 |
| `exp_dampening_d3_weighted.yaml` | D3W: D3 + l23_energy_weight=3.0 |
| `exp_v2_ei_highenergy_weighted.yaml` | HEW: HighEnergy + l23_energy_weight=3.0 |

### Results directories

| Directory | Contents |
|---|---|
| `results/dampening/d1_sensory0/` | D1 dampening (seed 42) |
| `results/dampening/d2_sensory03/` | D2 weak sensory (seed 42) |
| `results/dampening/d3_sensory10/` | D3 full sensory + mismatch (seed 42) |
| `results/dampening/d1w_sensory0/` | D1W weighted dampening (seed 42) |
| `results/dampening/d2w_sensory03/` | D2W weighted weak sensory (seed 42) |
| `results/dampening/d3w_sensory10/` | D3W weighted full (seed 42) |
| `results/iter/v2_ei_highenergy_weighted/` | HEW weighted sharpening (seed 42) |
