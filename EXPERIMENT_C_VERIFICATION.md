# Experiment C Verification: L2/3 Population Response Analysis

## Failure characterization

This is a verification analysis, not a failure investigation. The question: **Does the learned "expectation suppression" kernel (a_inh > a_exc, net center = −0.087) actually produce expectation suppression in L2/3 population responses?**

**Answer: NO.** The kernel has net inhibition (dampening), but the actual L2/3 response shows CENTER-SURROUND behavior — **facilitation** at the predicted orientation (+2.8%), **suppression** at flanks (+1.0%). This is because the center_exc and SOM pathways have different effective gains in the full network.

---

## Experiment setup

- **Checkpoints**: `checkpoints/exp_centersurround/center_surround_seed{42,123,456}/checkpoint.pt`
- **Config**: `config/exp_centersurround.yaml`
- **Key settings**: feedback_mode=emergent, lambda_fb=0.0 (no L1 sparsity), lambda_detection=1.0, stimulus_noise=0.1, v2_input_mode=l23
- **Methodology**: Oracle q_pred (Gaussian bump σ=10°), π_eff=3.0, 20 timesteps to steady state. Feedback ON vs OFF comparison (alpha zeroed).

---

## Analysis 1: Kernel weights (all 3 seeds)

| Basis | Seed 42 α_inh | Seed 42 α_exc | Seed 42 net | Seed 123 net | Seed 456 net |
|---|---|---|---|---|---|
| G(σ=5°) | +0.395 | +0.283 | −0.112 | −0.108 | −0.114 |
| G(σ=15°) | +0.396 | +0.277 | −0.119 | −0.114 | −0.119 |
| G(σ=30°) | +0.398 | +0.270 | −0.128 | −0.122 | −0.128 |
| G(σ=60°) | +0.399 | +0.265 | −0.134 | −0.128 | −0.134 |
| MexHat | +0.389 | +0.295 | −0.093 | −0.090 | −0.095 |
| Const | +0.011 | +0.009 | −0.002 | −0.002 | −0.001 |
| Odd/Sin | +0.006 | +0.008 | +0.002 | +0.002 | −0.001 |

**Seed consistency**: Net kernel at center (Δθ=0°): −0.087 (s42), −0.083 (s123), −0.088 (s456). CV = 2.8%. Highly reproducible.

**Key difference from earlier models**: With lambda_fb=0.0 (no L1 sparsity), ALL 7 basis functions are active (‖α_inh‖₁ ≈ 1.99, ‖α_exc‖₁ ≈ 1.41). The earlier noise=0.00 and noise=0.30 models (lambda_fb=0.01) used only 2–3 bases.

### Kernel profile (seed 42)

| Δθ | K_inh | K_exc | K_net (exc−inh) |
|---|---|---|---|
| 0° | +0.304 | +0.217 | **−0.087** |
| 5° | +0.230 | +0.164 | −0.066 |
| 10° | +0.127 | +0.089 | −0.037 |
| 15° | +0.075 | +0.052 | −0.023 |
| 20° | +0.048 | +0.032 | −0.016 |
| 30° | +0.022 | +0.014 | −0.008 |
| 45° | +0.013 | +0.008 | −0.005 |
| 60° | +0.010 | +0.006 | −0.003 |
| 90° | +0.005 | +0.004 | −0.002 |

**K_net is NEGATIVE at ALL angular offsets.** Inhibition exceeds excitation everywhere. The kernel is pure dampening — strongest net suppression at center, monotonically decaying.

---

## Analysis 2: Feedback drive profiles (bias-subtracted signals)

Uniform q_pred baseline: som_drive = center_exc = π×ln(2) = 2.0794 (confirmed, bias cancels exactly as in PROFILE_SHAPES.md).

### Seed 42, oracle q_pred at ch 18, π = 3.0

| Δθ | SOM signal | Exc signal | Net (exc−som) | Effect |
|---|---|---|---|---|
| 0° | +0.2185 | +0.1550 | **−0.0635** | Net suppression |
| 10° | +0.1519 | +0.1075 | −0.0444 | Net suppression |
| 20° | +0.0432 | +0.0297 | −0.0136 | Net suppression |
| 30° | −0.0178 | −0.0140 | +0.0038 | Net facilitation |
| 45° | −0.0452 | −0.0329 | +0.0123 | Net facilitation |
| 60° | −0.0517 | −0.0368 | +0.0149 | Net facilitation |
| 90° | −0.0566 | −0.0394 | +0.0172 | Net facilitation |

In the feedback drive signals: center is NET SUPPRESSIVE (more SOM than excitation), flanks are NET FACILITATIVE. This is consistent with the dampening kernel.

---

## Analysis 3: Full-network L2/3 suppression-by-tuning profile

**Critical finding: the L2/3 response does NOT match the kernel pattern.**

### Seed 42

| Δθ (stim−pred) | R_on | R_off | Delta | SI |
|---|---|---|---|---|
| 0° (match) | 1.2521 | 1.2185 | +0.0336 | **−2.8% (facilitation)** |
| 5° | 1.2503 | 1.2185 | +0.0318 | −2.6% |
| 10° | 1.2453 | 1.2185 | +0.0267 | −2.2% |
| 15° | 1.2382 | 1.2185 | +0.0196 | −1.6% |
| 20° | 1.2306 | 1.2185 | +0.0121 | −1.0% |
| 30° | 1.2185 | 1.2185 | −0.0001 | +0.0% |
| 45° | 1.2100 | 1.2185 | −0.0086 | **+0.7% (suppression)** |
| 60° | 1.2076 | 1.2185 | −0.0109 | +0.9% |
| 90° | 1.2066 | 1.2185 | −0.0119 | **+1.0% (suppression)** |

### Cross-seed consistency

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean ± SD |
|---|---|---|---|---|
| SI at Δθ=0° | −2.8% | −3.0% | −2.7% | −2.8% ± 0.15% |
| SI at Δθ=45° | +0.7% | +0.8% | +0.7% | +0.7% ± 0.05% |
| SI at Δθ=90° | +1.0% | +1.1% | +1.0% | +1.0% ± 0.05% |
| Zero crossing | ~30° | ~30° | ~30° | ~30° |

**All 3 seeds show identical CENTER-SURROUND L2/3 behavior despite the dampening kernel.**

---

## Analysis 4: Expected vs unexpected vs neutral

Stimulus at 90° (ch 18), seed 42.

| Condition | R at stim ch | vs Neutral |
|---|---|---|
| **Expected** (q_pred = 90°, match) | 1.2521 | **+2.8% (facilitation)** |
| **Unexpected** (q_pred = 0°, mismatch) | 1.2066 | **−1.0% (suppression)** |
| **Neutral** (α = 0, no feedback) | 1.2185 | baseline |

Expected − Unexpected = **+0.0455 (+3.8%)**

The L2/3 response is **HIGHER** for expected stimuli and **LOWER** for unexpected stimuli, relative to neutral. This is the opposite of expectation suppression — it is **expectation facilitation** (or equivalently, **surprise suppression**).

---

## Analysis 5: Detection head

**The detection head weights (nn.Linear(36, 1)) are NOT saved in the checkpoint.** Only the orientation decoder weights are saved (`decoder_state`). This is a gap in the checkpoint-saving code.

To test whether L2/3 carries expected/unexpected information, I trained a fresh logistic regression classifier on L2/3 readouts:

| Metric | Value |
|---|---|
| Training samples | 800 (400 expected, 400 unexpected) |
| Test samples | 200 (100 expected, 100 unexpected) |
| Train accuracy | 0.516 |
| Test accuracy | **0.505 (chance)** |
| Chance level | 0.500 |

**The L2/3 representation does NOT carry enough information to distinguish expected from unexpected stimuli.** The ~2.8% facilitation effect is too small relative to the total L2/3 signal (~1.25) for a linear readout to exploit.

**Note**: This tests the oracle-driven condition. During actual training, the V2's noisy predictions and the readout window dynamics may create different patterns. But the oracle condition represents the best case, and even there the information is at chance.

---

## Analysis 6: Comparison to earlier models

| Property | Exp C (this) | noise=0.00 (PROFILE_SHAPES) | noise=0.30 (PROFILE_SHAPES) |
|---|---|---|---|
| lambda_fb (L1) | **0.0** | 0.01 | 0.01 |
| lambda_detection | **1.0** | 0.0 | 0.0 |
| stimulus_noise | 0.1 | 0.0 | 0.3 |
| v2_input_mode | **l23** | l4 | l4 |
| Active bases | **7/7** | 3/7 (exc), 2/7 (inh) | 2/7 (inh only) |
| ‖α_inh‖₁ | **1.99** | 0.075 | 0.050 |
| ‖α_exc‖₁ | **1.41** | 0.257 | 0.000 |
| Kernel pattern | Dampening | Center-surround | Dampening |
| **L2/3 pattern** | **Center-surround** | Center-surround | Dampening |
| SI at Δθ=0° | −2.8% | −3.3% | +0.6% |
| SI at Δθ=90° | +1.0% | +1.1% | −0.1% |
| Detection acc | 50% (chance) | — | — |

**Key observations:**
1. Exp C has ~40× larger alpha weights than the noise=0.00 model (1.99 vs 0.075 for inh), due to lambda_fb=0.0 removing the L1 penalty.
2. Despite larger weights, the L2/3 effect magnitudes are comparable (2.8% vs 3.3% facilitation). The softplus nonlinearity saturates, limiting the effective gain.
3. The kernel pattern (dampening) does NOT predict the L2/3 pattern (center-surround) in Exp C. This mismatch does not occur in the other two models.

---

## Why kernel ≠ L2/3: the pathway gain mismatch

The kernel K_net = K_exc − K_inh is negative everywhere (dampening). But the actual L2/3 effect is center-surround (facilitation at center). The reason:

1. **center_exc feeds DIRECTLY into L2/3 drive** (additive term in L2/3 equation)
2. **SOM drive goes through the SOM population** (with its own tau_som=10 dynamics and gains) **then through w_som into L2/3** (subtractive)
3. The effective gain from SOM signal → L2/3 suppression is **less than 1.0** (SOM dynamics + w_som gain attenuate the signal)
4. The effective gain from center_exc → L2/3 facilitation is **1.0** (direct addition)

So even though K_inh > K_exc at center (more SOM drive than center_exc), the center_exc has higher effective gain to L2/3. The net L2/3 effect is facilitation.

At the flanks (Δθ>25°), both signals go negative (less than baseline). The negative exc signal (less excitation) has a larger magnitude than the negative SOM signal (less inhibition). With the SOM pathway having lower gain, the net effect is: less excitation wins → L2/3 is suppressed at flanks.

**This is a critical architectural finding: the kernel weights alone do NOT determine the L2/3 effect. The pathway gains (direct vs SOM-mediated) must be accounted for.**

---

## Confirmed findings

1. **Kernel pattern: DAMPENING** — a_inh > a_exc at all 7 basis functions. K_net is negative everywhere. Net center = −0.087 ± 0.003 across 3 seeds.

2. **L2/3 pattern: CENTER-SURROUND** — facilitation at predicted (−2.8% SI), suppression at flanks (+1.0% SI). Consistent across all 3 seeds (CV < 6%).

3. **Kernel-to-L2/3 mismatch** — The dampening kernel produces center-surround L2/3 behavior due to the SOM pathway having lower effective gain than the direct center_exc pathway.

4. **Expected > Neutral > Unexpected** — at the stimulus channel: expected is 2.8% above neutral, unexpected is 1.0% below. This is **expectation facilitation**, not expectation suppression.

5. **Detection: at chance** — L2/3 readout cannot distinguish expected from unexpected (acc = 50.5%). The effect magnitude (~3%) is too small for linear readout. Detection head weights are not saved in checkpoint.

6. **Lambda_fb=0 allows large weights** — all 7 bases active, ‖α‖₁ ~1.4–2.0 (vs 0.05–0.26 with lambda_fb=0.01). But L2/3 effect magnitudes are similar due to softplus saturation.

---

## Experimental methodology

- Oracle q_pred: Gaussian bump (σ=10°) centered at predicted orientation, normalized to distribution
- π_eff = 3.0 (fixed)
- 20 timesteps from zero initial state to steady state
- Feedback ON vs OFF: trained alpha weights vs zeroed alpha weights
- Suppression index: SI = (R_off − R_on) / R_off (positive = suppression, negative = facilitation)
- Detection test: 500 expected + 500 unexpected L2/3 samples, logistic regression, 80/20 train/test split
- All 3 seeds (42, 123, 456) tested for cross-seed consistency
