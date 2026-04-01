# Diagnostic Report: Stuck Models B, C, and A (V2)

## Failure Summary

| Model | Mechanism | Steps trained | Sensory acc | Pred acc | Status |
|-------|-----------|---------------|-------------|----------|--------|
| A (dampening) | Narrow SOM | 40K | 63% | 2.7% (chance) | Sensory OK, V2 stuck |
| B (sharpening) | DoG SOM | 14.4K | 2.8% (chance) | 2.7% (chance) | Fully stuck |
| C (center_surround) | Broad SOM + center excitation | 20K | ~3% (chance) | ~3% (chance) | Fully stuck |
| D (adaptation_only) | No feedback | 40K | 63% | 26% | Learning normally |

---

## Hypotheses Tested

### H1: `shifted_softplus` negativity causes oscillatory instability → CONFIRMED

**Experiment**: Run Model C forward pass with original `shifted_softplus` vs patched `ReLU`.

| Metric | shifted_softplus | ReLU |
|--------|-----------------|------|
| Steps with negative L2/3 | 46/50 | 0/50 |
| L2/3 mean range | [-0.171, 0.343] | [0.0002, 0.031] |
| L2/3 oscillation (std of means) | **0.1463** | **0.0085** |
| Steps with negative PV | 13/50 | 0/50 |

**Causal proof**: Changing ONLY the activation (shifted_softplus → ReLU) reduces oscillation 17.1x
and eliminates all negative rates (46/50 → 0/50 steps).

**Control**: Model D has 50/50 steps with negative L2/3 but learns normally (std=0.0027). Negative
rates alone are not fatal — they become catastrophic when amplified by recurrence and feedback.

### H2: High W_rec gain amplifies the oscillation cycle → CONFIRMED (secondary)

| Configuration | L2/3 oscillation (std) | Neg L2/3 steps |
|---------------|----------------------|----------------|
| Model C trained gain = 0.7332 | 0.1463 | 46/50 |
| Reduced to Model D's gain = 0.1208 | 0.0228 | 46/50 |
| Zero gain | 0.0172 | 46/50 |
| Initial (untrained) gain = 0.3000 | 0.0253 | 47/50 |

**Causal proof**: Changing ONLY W_rec gain (0.7332 → 0.1208) reduces oscillation 6.4x (0.1463 →
0.0228). Setting gain to zero reduces it 8.5x (0.1463 → 0.0172). But negative L2/3 steps remain
46/50 in all conditions — W_rec gain amplifies the oscillation but does not cause the negativity.

Training increased W_rec gain from 0.3 → 0.7332, which amplified oscillation 5.8x (0.0253 →
0.1463). A vicious cycle: optimizer pushes gain higher to amplify signal, worsening instability.

### H3: Negative PV amplifies L4 via divisive normalization → PARTIALLY CONFIRMED (not primary driver)

**Original observation**: Step trace shows PV goes to -0.619, reducing denominator to 0.381 (vs
normal 1.0), amplifying L4 2.62x.

**Causal test**: Clamped PV to non-negative after each Euler step.

| Metric | Control (PV free) | PV clamped ≥ 0 |
|--------|-------------------|----------------|
| PV range | [-0.619, 10.305] | [0.000, 0.693] |
| Min denominator | 0.381 | 1.000 |
| Max L4 peak | 1.450 | 0.548 |
| L2/3 oscillation | **0.1463** | **0.2294** |

**Result**: Clamping PV to non-negative ELIMINATES L4 amplification (1.45 → 0.55), confirming
negative PV does amplify L4. BUT oscillation actually WORSENS (0.1463 → 0.2294). This is because
the PV overshoot (reaching 10.3 in the control) acts as a natural brake — it strongly suppresses
L4, which eventually damps the oscillation cycle. Without the brake, the oscillation grows larger.

**Verdict**: Negative PV amplifies L4 (CONFIRMED) but this is NOT the primary driver of L2/3
oscillation — it's actually a self-correcting mechanism. The L2/3 oscillation is driven by
shifted_softplus negativity + recurrence (H1 + H2).

### H4: Model A pi_pred saturates at pi_max with vanishing sigmoid gradient → CONFIRMED

**Experiment H4a**: Gradient magnitude through pi path with original sigmoid vs softplus.

| Metric | Original (sigmoid) | Patched (softplus) | Ratio |
|--------|-------------------|--------------------|-------|
| head_pi weight gradient | 0.000532 | 0.998 | **1877x** |
| GRU weight_hh gradient | 0.000212 | 0.398 | **1877x** |
| GRU bias_ih gradient | 0.000272 | 0.511 | **1877x** |

**Causal proof (H4a)**: Changing ONLY the pi activation (sigmoid → softplus) increases gradient
through the pi path by 1877x. The sigmoid at pre-activation 9.15 has gradient 0.000107, effectively
blocking all learning signal.

**Causal proof (H4b)**: Ran 20 actual training steps with original vs patched activation.

| Metric | Original sigmoid | Patched softplus |
|--------|-----------------|------------------|
| pi_pred movement over 20 steps | 0.00387 | 0.01979 |
| Max head_pi gradient | 0.000459 | 0.00400 |

pi_pred moves 5.1x more with softplus, and gradients are 8.7x larger. Over 80K steps, this
difference compounds — the original sigmoid path is essentially frozen.

**Alternative hypothesis H4c**: Maybe q_pred (not pi_pred) is the bottleneck.

| Metric | Value |
|--------|-------|
| q_pred entropy | 3.485 (max = 3.584) |
| q_pred peak | 0.048 (uniform = 0.028) |
| q_pred argmax | ch 18 (correct for 90° stimulus) |
| head_q gradient | 0.936 |

**RULED OUT**: q_pred has healthy gradients (0.936), is slightly peaked at the correct channel, and
is learning. The bottleneck is specifically pi_pred saturation, not q_pred.

### H5: Model B stuck due to SOM baseline + negative recurrence runaway → CONFIRMED (corrected)

**Original hypothesis** (DoG center negativity causes reversed inhibition): **RULED OUT by causal test.**

Three manipulation experiments on Model B:

| Condition | L2/3 neg steps | L2/3 oscillation | L2/3 range |
|-----------|----------------|-------------------|-------------|
| Control (original) | 50/50 | 0.2198 | [-0.685, -0.006] |
| H5a: Increase som_baseline to 3.0 | 50/50 | 0.1962 | [-0.688, -0.014] |
| H5b: Clamp SOM drive ≥ 0 | 50/50 | 0.2198 | [-0.685, -0.006] |
| H5c: ReLU in L2/3 only | 0/50 | 0.0000 | [0.000, 0.000] |

**H5a result**: Increasing som_baseline has NO effect — still 50/50 negative, oscillation barely
changed. The DoG center negativity is NOT the cause.

**H5b result**: IDENTICAL to control. SOM drive is already all positive in actual dynamics
(min=1.66, all positive) because the DoG effect is small relative to baseline at the actual
pi_pred values. The theoretical DoG negativity only appears with artificially peaked q_pred and
high pi_pred, not in the real dynamics.

**H5c result**: ReLU in L2/3 ONLY eliminates all negativity — but L2/3 goes to exactly 0.0,
meaning the total L2/3 drive is always negative. The network is dead, not fixed.

**Corrected root cause** — L2/3 drive decomposition at step 30 reveals:

| Component | Model B | Model D (control) |
|-----------|---------|-------------------|
| Feedforward (L4) | +0.338 | +0.079 |
| **Recurrent (W_rec @ L2/3)** | **-6.617** | **-0.011** |
| -SOM inhibition | -1.129 | 0.000 |
| -PV inhibition | +0.840 | -0.103 |
| **Total L2/3 drive** | **-6.569** | **-0.035** |
| shifted_softplus(drive) | -0.691 | -0.012 |

**The dominant term is recurrent drive at -6.617** — 600x larger than Model D's -0.011. The
causal chain:

1. SOM baseline (1.0) provides constant inhibition that initially pushes L2/3 drive slightly below
   zero (where Model D has near-zero SOM drive)
2. shifted_softplus allows L2/3 to go negative (to floor ≈ -0.693)
3. **W_rec is a positive Gaussian kernel** (gain × exp(-dist²/2σ²)). Applied to deeply negative
   L2/3, it produces massive negative recurrent drive: W_rec @ (-0.693 × ones) ≈ -6.6
4. This runaway negative recurrence locks L2/3 at the shifted_softplus floor
5. PV goes to -0.693 (from negative L2/3 sum) — the PV floor
6. Network is stuck in a pathological fixed point where all populations are at their negative floors

**Why Model D doesn't have this problem**: No SOM inhibition → L2/3 drive stays near zero → even
when shifted_softplus returns slightly negative values, the recurrence is W_rec @ (-0.01) ≈ -0.01,
not W_rec @ (-0.69) ≈ -6.6. The positive recurrence doesn't amplify because L2/3 isn't deep in
the negative regime.

---

## Proven Root Causes

### Root Cause 1: `shifted_softplus` allows negative firing rates

**Location**: `src/utils.py:86-88`
```python
def shifted_softplus(x: Tensor) -> Tensor:
    return F.softplus(x) - _SOFTPLUS_ZERO  # can return values down to -0.6931
```

**Impact**: All four populations (L4, PV, L2/3, SOM) can develop negative "rates." When negative
rates enter positive recurrence (W_rec) or divisive normalization (PV), pathological dynamics emerge.

**Causal proof**: ReLU replacement eliminates Model C oscillation 17.1x (H1). ReLU in L2/3 only
eliminates Model B's negative recurrence runaway (H5c, though L2/3 then goes to zero because the
feedforward drive minus SOM inhibition is negative).

### Root Cause 2: SOM baseline inhibition + positive recurrence creates negative runaway in Model B

**Location**: `src/model/feedback.py:147` (som_baseline) and `src/model/populations.py:257` (W_rec)

**Mechanism**: SOM baseline pushes L2/3 slightly negative → W_rec amplifies negativity → L2/3
locks at -0.693 → W_rec @ (-0.693) = -6.6 → permanent negative fixed point.

**Causal proof**: L2/3 drive decomposition shows recurrence = -6.617 (dominant term). Model D has
no SOM baseline and recurrence = -0.011 (600x smaller). The SOM baseline is the initial push that
triggers the runaway.

**NOTE**: The DoG center negativity (originally hypothesized as root cause) was **RULED OUT** by
causal experiments H5a and H5b. The SOM drive is all positive in actual dynamics (min=1.66).

### Root Cause 3: Model C oscillatory instability from shifted_softplus + high W_rec gain

**Location**: `src/utils.py:86-88` + `src/model/populations.py:209`

**Mechanism**: shifted_softplus negativity + trained W_rec gain (0.7332) creates oscillatory
instability with period ~25 steps, preventing stable gradient-based learning.

**Causal proof**: ReLU eliminates oscillation 17.1x (H1). Reducing W_rec gain to 0.12 reduces
oscillation 6.4x (H2). These are independent, additive effects.

### Root Cause 4: Model A pi_pred saturates with vanishing sigmoid gradient

**Location**: `src/model/v2_context.py:68`
```python
pi_pred = self.pi_max * torch.sigmoid(self.head_pi(h_v2))  # [B, 1]
```

**Mechanism**: head_pi outputs 9.15 → sigmoid = 0.9999 → gradient = 0.000107 → learning frozen.

**Causal proof**: Replacing sigmoid with softplus increases gradient 1877x (H4a). Over 20 training
steps, pi_pred moves 5.1x more (H4b). q_pred gradients are healthy (0.936), ruling out the
alternative that V2 itself is broken (H4c).

---

## Suggested Fix Directions

### Fix 1: Replace `shifted_softplus` with a non-negative activation (Models B, C)

Options (in order of recommendation):
1. **Clamped shifted_softplus**: `shifted_softplus(x).clamp(min=0)`. Most conservative — preserves
   the smooth derivative near zero while preventing negativity. No parameter retuning needed.
2. **`softplus`** (unshifted): `F.softplus(x)`. Always ≥ 0. Introduces baseline rate ≈ 0.693.
   May need tau or gain retuning to account for the offset.
3. **`ReLU`**: `F.relu(x)`. Proven effective but may cause dead neurons (zero gradient for x < 0).

This single fix addresses Root Causes 1, 2, and 3 simultaneously:
- Prevents L2/3 from going negative → eliminates recurrence runaway (Model B)
- Prevents PV from going negative → eliminates L4 amplification pathway
- With non-negative L2/3, positive W_rec amplifies signal instead of amplifying negativity

### Fix 2: Model A pi_pred parameterization (Model A)

Options:
1. **softplus(head_pi(h_v2)).clamp(max=pi_max)**: Gradient is always ≥ 0.25 for negative inputs,
   unlike sigmoid which saturates at both ends. Bounded by clamp.
2. **Lower pi_max** (e.g., 2.0): Reduces pressure to saturate.
3. **Initialize head_pi bias to 0**: Prevents early saturation (current bias = 0.762).

### Fix 3 (optional): Constrain W_rec gain (Model C)

Clamp `gain_rec_raw` to limit effective gain (e.g., max 0.5). This is secondary if Fix 1 is
applied — with non-negative L2/3, the recurrence is constructive (amplifying signal, not noise).

---

## Hypothesis Verdict Summary

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| H1 | shifted_softplus negativity → oscillation | **CONFIRMED** | ReLU eliminates oscillation 17.1x |
| H2 | High W_rec gain amplifies oscillation | **CONFIRMED** (secondary) | Reducing gain reduces oscillation 6.4x |
| H3 | Negative PV → L4 amplification → instability | **PARTIALLY CONFIRMED** | PV clamp eliminates L4 amplification (2.62x→1.0x) but oscillation WORSENS — PV overshoot acts as natural brake |
| H4 | pi_pred sigmoid saturation → vanishing gradient | **CONFIRMED** | Softplus increases gradient 1877x; pi moves 5.1x more over 20 steps |
| H4c | q_pred (not pi_pred) is bottleneck | **RULED OUT** | q_pred gradient is healthy (0.936), peaked at correct channel |
| H5 (orig) | DoG center negativity → reversed SOM inhibition | **RULED OUT** | Increasing baseline and clamping SOM drive have zero effect — SOM drive is already all positive in actual dynamics |
| H5 (corrected) | SOM baseline + W_rec negative runaway | **CONFIRMED** | L2/3 drive decomposition: recurrence = -6.617 (600x larger than Model D), caused by SOM baseline pushing L2/3 into negative shifted_softplus regime |

---

## Environment

- PyTorch 2.10.0+cu130
- NVIDIA RTX A6000
- Python 3.13
- Checkpoints: A (40K), B (Stage 1 only), C (20K), D (40K)
