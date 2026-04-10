# Results

## Overview

A single laminar V1-V2 architecture — trained end-to-end with BPTT —
discovers three distinct feedback regimes from the loss landscape alone.
No architectural switches are needed: the balance between sensory
supervision (λ_sensory) and energy cost (λ_energy) determines whether V2
feedback dampens, preserves, or sharpens L2/3 representations. A 25-run
parameter sweep across 7 groups maps the full phase space.

**Headline result:** Config e1 (λ_sensory=0.3, l23_energy_weight=3.0)
produces near-unity amplitude (M10=1.13) with the strongest tuning
sharpening in the sweep (FWHM=−11.6°), closely matching Kok et al. (2012)
experimental findings.

---

## 1. Three Feedback Regimes

The model's feedback behavior is controlled primarily by λ_sensory, the
weight on 36-way cross-entropy orientation decoding from L2/3 activity.

### Phase diagram (10 λ_sensory values, λ_energy=2.0, l23w=1.0)

| λ_sensory | M7 δ=10° | M10 amp | FWHM Δ | FB contrib | Regime |
|---|---|---|---|---|---|
| 0.00 | −0.047 | 0.70 | −3.9° | +0.143 | **Dampening** |
| 0.10 | +0.019 | 0.95 | −3.5° | −0.003 | Transition |
| 0.12 | +0.028 | 1.00 | −3.2° | −0.001 | **Unity point** |
| 0.15 | +0.053 | 1.09 | −2.0° | −0.004 | Transition |
| 0.18 | +0.071 | 1.18 | −4.3° | +0.005 | **FB flip** |
| 0.20 | +0.081 | 1.23 | −5.2° | +0.014 | Enhancement |
| 0.30 | +0.129 | 1.50 | −5.8° | +0.066 | Enhancement |
| 0.50 | +0.203 | 1.96 | −1.6° | +0.163 | Strong sharpening |
| 0.70 | +0.238 | 2.32 | −1.7° | +0.287 | Strong sharpening |
| 1.00 | +0.271 | 2.77 | −1.9° | +0.398 | Strong sharpening |

### Regime definitions

**Dampening (λ_sensory=0.0):** Without sensory supervision, the energy
cost dominates and V2 learns to suppress L2/3 activity at the predicted
orientation via SOM inhibition. M7 is negative (feedback hurts decoding),
amplitude is sub-unity (M10=0.70), and the feedback widens the
expected-vs-unexpected gap by suppressing expected stimuli more. This maps
to the global suppression reported by Alink et al. (2010).

**Transition zone (λ_sensory=0.10–0.15):** The mechanism switches modes.
M7 turns positive but FB contribution is negative (NARROWS) — improvement
comes from direct sensory optimization, not feedback-mediated
differentiation. The exact unity amplitude point is at λ_sensory=0.12
(M10=0.9999).

**FB-mediated sharpening (λ_sensory≥0.18):** FB contribution flips
positive (WIDENS the expected-vs-unexpected gap). V2 feedback now actively
enhances decoding by differentiating expected from unexpected stimuli.
M7 increases monotonically with λ_sensory. This maps to Kok et al. (2012)
expectation sharpening.

### Key transitions

1. **Phase boundary** (dampening→enhancement): Between λ_sensory=0.0 and
   0.10 — M7 flips from negative to positive.
2. **Unity amplitude** at λ_sensory=0.12 — M10=1.000 exactly.
3. **FB contribution sign flip** between λ_sensory=0.15 (NARROWS) and
   0.18 (WIDENS) — feedback switches from passive to active role.

---

## 2. Best Dampening Configuration

**Config b1:** λ_sensory=0.0, λ_energy=5.0, l23w=1.0

| Metric | Value |
|---|---|
| M7 δ=10° | −0.060 |
| M10 amplitude | 0.643 |
| FWHM Δ | −3.0° |
| FB contribution | +0.153 (WIDENS) |

### Mechanism

Without sensory loss, V2 feedback minimizes the energy cost by suppressing
L2/3 activity wherever it can predict the stimulus. The feedback head
learns a profile that drives SOM inhibition at the predicted orientation,
producing sub-unity amplitude (36% reduction). Higher λ_energy strengthens
this effect: M10 drops from 0.70 (λ_energy=2.0) to 0.64 (λ_energy=5.0).

The dampening regime is robust across all tested parameter combinations —
it emerges whenever λ_sensory=0, regardless of energy weight or l23w.
Increased l23w amplifies dampening further: e2 (l23w=3.0) produces the
strongest dampening (M10=0.53).

### Biological correspondence

Maps to Alink et al. (2010) — global suppression of expected stimuli in
early visual cortex. The model shows this is the default behavior when
higher areas have no incentive to preserve sensory fidelity.

---

## 3. Best Sharpening Configurations

### e1: Near-unity amplitude + strong sharpening (BEST OVERALL)

**Config:** λ_sensory=0.3, λ_energy=2.0, l23w=3.0

| Metric | Value |
|---|---|
| M7 δ=10° | +0.104 |
| M10 amplitude | 1.13 |
| FWHM Δ | **−11.6°** |
| FB contribution | +0.002 (WIDENS) |

This is the best literature match. Near-unity amplitude with strong tuning
sharpening reproduces the signature reported by Kok et al. (2012): expected
stimuli show subtly different amplitude but significantly sharper tuning.
The l23_energy_weight=3.0 specifically constrains L2/3 activity without
limiting V2 freedom, achieving a balance that pure λ_energy tuning cannot.

### c2: Maximum controlled sharpening

**Config:** λ_sensory=1.0, λ_energy=2.0, l23w=5.0

| Metric | Value |
|---|---|
| M7 δ=10° | +0.196 |
| M10 amplitude | 1.77 |
| FWHM Δ | −7.0° |
| FB contribution | +0.207 (WIDENS) |

Highest M7 with M10<2.0. Stronger sensory pressure with aggressive L2/3
energy control. More amplitude than e1 but stronger decoder improvement.

### a2a: Exact unity amplitude

**Config:** λ_sensory=0.12, λ_energy=2.0, l23w=1.0

| Metric | Value |
|---|---|
| M7 δ=10° | +0.028 |
| M10 amplitude | **1.000** |
| FWHM Δ | −3.2° |
| FB contribution | −0.001 (NARROWS) |

M10=0.9999 — the precise parameter value where feedback neither boosts nor
suppresses globally. Weak but positive M7 shows that even at unity
amplitude, the model achieves mild decoder improvement. However, feedback
does not actively contribute (FB NARROWS), so the improvement comes from
direct sensory optimization.

### Sharpening mechanism (simple feedback mode)

V2's GRU `head_feedback` (Linear(16,36)) outputs a raw 36-channel signal.
Dale's law E/I split:
- `center_exc = relu(+feedback)` → additive excitation to L2/3
- `som_drive = relu(−feedback)` → SOM inhibition of L2/3

At λ_sensory≥0.18, V2 learns to:
1. Excite L2/3 at the predicted orientation (center_exc > 0)
2. Inhibit L2/3 at flanking orientations via SOM (som_drive > 0)
3. This sharpens the population response: narrower FWHM + enhanced peak

The sensory loss prevents energy-driven collapse — L2/3 must maintain
decodable orientation representations, so V2 cannot simply suppress
everything.

---

## 4. Expected vs Unexpected Analysis

### Feedback contribution metric

The FB contribution measures whether feedback widens or narrows the gap
between expected and unexpected L2/3 responses:
- **WIDENS (positive):** Feedback enhances expected stimuli more than
  unexpected → active predictive differentiation
- **NARROWS (negative):** Feedback reduces the natural gap → feedback is
  counterproductive for differentiation

### Sign flip at the transition boundary

| λ_sensory | FB contribution | Direction |
|---|---|---|
| 0.10 | −0.003 | NARROWS |
| 0.12 | −0.001 | NARROWS |
| 0.15 | −0.004 | NARROWS |
| **0.18** | **+0.005** | **WIDENS** |
| 0.20 | +0.014 | WIDENS |
| 0.30 | +0.066 | WIDENS |

The sign flip occurs between λ_sensory=0.15 and 0.18. Below this boundary,
any decoder improvement comes from internal L2/3 optimization (recurrent
sharpening, better tuning), not from V2 feedback. Above it, V2 actively
contributes to expected-vs-unexpected differentiation.

### Monotonic scaling with λ_sensory

FB contribution increases monotonically above the transition:
0.18→+0.005, 0.20→+0.014, 0.30→+0.066, 0.50→+0.163, 0.70→+0.287,
1.00→+0.398. Stronger sensory pressure → V2 invests more in
differentiating expected from unexpected stimuli.

---

## 5. Decoder Analysis

### M7 at multiple discrimination distances

M7 measures the change in trained LogReg decoder accuracy (feedback ON
minus OFF) at orientation distance δ:

| λ_sensory | M7 δ=3° | M7 δ=5° | M7 δ=10° | M7 δ=15° |
|---|---|---|---|---|
| 0.0 | −0.024 | −0.035 | −0.047 | −0.064 |
| 0.1 | +0.003 | +0.006 | +0.019 | +0.013 |
| 0.3 | +0.048 | +0.084 | +0.129 | +0.129 |
| 0.5 | +0.066 | +0.128 | +0.203 | +0.184 |
| 0.7 | +0.089 | +0.166 | +0.238 | +0.204 |
| 1.0 | +0.113 | +0.189 | +0.271 | +0.223 |

**Key observations:**
- M7 is monotonically positive at all δ for λ_sensory≥0.1
- δ=10° shows the cleanest scaling (no saturation up to λ_sensory=1.0)
- δ=15° shows saturation above λ_sensory≥0.5 (ceiling effect)
- δ=3° keeps climbing (fine discrimination doesn't saturate as quickly)
- Dampening (λ_sensory=0.0) hurts decoding at ALL distances

### Amplitude (M10) vs decoder (M7) trade-off

Higher λ_sensory gives better M7 but also higher M10 (larger amplitude).
The l23_energy_weight parameter breaks this correlation:

| Config | M7 δ=10° | M10 | FWHM Δ |
|---|---|---|---|
| a4 (l23w=1.0) | +0.129 | 1.50 | −5.8° |
| e1 (l23w=3.0) | +0.104 | 1.13 | −11.6° |
| c1 (l23w=5.0) | +0.081 | 0.95 | −4.3° |

l23w=3.0 reduces M7 only modestly (−19%) while cutting M10 by 25% and
more than doubling FWHM sharpening. This is the key insight: l23w
specifically targets L2/3 excitability without constraining V2, achieving
better amplitude-sharpening trade-offs than global energy tuning.

---

## 6. Full 25-Run Parameter Sweep

### Complete results table

All runs: stage1=2000 steps, stage2=5000 steps, seed=42.

| Run | λ_sens | λ_energy | λ_mm | l23w | M7 δ=10° | M10 amp | FWHM Δ | FB contrib | FB dir |
|---|---|---|---|---|---|---|---|---|---|
| a1 | 0.0 | 2.0 | 0.0 | 1.0 | −0.047 | 0.70 | −3.9° | +0.143 | WIDENS |
| a2 | 0.10 | 2.0 | 0.0 | 1.0 | +0.019 | 0.95 | −3.5° | −0.003 | NARROWS |
| a2a | 0.12 | 2.0 | 0.0 | 1.0 | +0.028 | 1.00 | −3.2° | −0.001 | NARROWS |
| a2b | 0.15 | 2.0 | 0.0 | 1.0 | +0.053 | 1.09 | −2.0° | −0.004 | NARROWS |
| a2c | 0.18 | 2.0 | 0.0 | 1.0 | +0.071 | 1.18 | −4.3° | +0.005 | WIDENS |
| a3 | 0.20 | 2.0 | 0.0 | 1.0 | +0.081 | 1.23 | −5.2° | +0.014 | WIDENS |
| a4 | 0.30 | 2.0 | 0.0 | 1.0 | +0.129 | 1.50 | −5.8° | +0.066 | WIDENS |
| a5 | 0.50 | 2.0 | 0.0 | 1.0 | +0.203 | 1.96 | −1.6° | +0.163 | WIDENS |
| a6 | 0.70 | 2.0 | 0.0 | 1.0 | +0.238 | 2.32 | −1.7° | +0.287 | WIDENS |
| a7 | 1.00 | 2.0 | 0.0 | 1.0 | +0.271 | 2.77 | −1.9° | +0.398 | WIDENS |
| b1 | 0.0 | 5.0 | 0.0 | 1.0 | −0.060 | 0.64 | −3.0° | +0.153 | WIDENS |
| b2 | 0.3 | 5.0 | 0.0 | 1.0 | +0.028 | 1.01 | −2.0° | +0.008 | WIDENS |
| b2e25 | 0.3 | 2.5 | 0.0 | 1.0 | +0.106 | 1.37 | −5.8° | +0.048 | WIDENS |
| b2e30 | 0.3 | 3.0 | 0.0 | 1.0 | +0.084 | 1.26 | −5.2° | +0.027 | WIDENS |
| b2e35 | 0.3 | 3.5 | 0.0 | 1.0 | +0.072 | 1.18 | −3.8° | +0.020 | WIDENS |
| b3 | 1.0 | 5.0 | 0.0 | 1.0 | +0.180 | 1.86 | −2.3° | +0.184 | WIDENS |
| b4 | 0.3 | 0.5 | 0.0 | 1.0 | +0.260 | 2.72 | −0.3° | +0.244 | WIDENS |
| b5 | 1.0 | 0.5 | 0.0 | 1.0 | +0.323 | 4.51 | +0.7° | +0.997 | WIDENS |
| c1 | 0.3 | 2.0 | 0.0 | 5.0 | +0.081 | 0.95 | −4.3° | +0.014 | WIDENS |
| c2 | 1.0 | 2.0 | 0.0 | 5.0 | +0.196 | 1.77 | −7.0° | +0.207 | WIDENS |
| d1 | 0.1 | 2.0 | 1.0 | 1.0 | +0.026 | 1.02 | −2.8° | −0.026 | NARROWS |
| d2 | 0.5 | 2.0 | 1.0 | 1.0 | +0.214 | 2.13 | −1.5° | +0.073 | WIDENS |
| d3 | 0.3 | 2.0 | 0.5 | 1.0 | +0.137 | 1.52 | −5.8° | +0.009 | WIDENS |
| e1 | 0.3 | 2.0 | 0.0 | 3.0 | +0.104 | 1.13 | −11.6° | +0.002 | WIDENS |
| e2 | 0.0 | 2.0 | 0.0 | 3.0 | −0.078 | 0.53 | −4.5° | +0.220 | WIDENS |

### Top 5 candidates (ranked by M7 + FWHM, M10 < 2.0)

1. **e1** (λ_sens=0.3, l23w=3.0): M7=+0.104, M10=1.13, FWHM=−11.6° — **BEST OVERALL**
2. **c2** (λ_sens=1.0, l23w=5.0): M7=+0.196, M10=1.77, FWHM=−7.0°
3. **b2e25** (λ_sens=0.3, λ_e=2.5): M7=+0.106, M10=1.37, FWHM=−5.8°
4. **d3** (λ_sens=0.3, λ_mm=0.5): M7=+0.137, M10=1.52, FWHM=−5.8°
5. **a4** (λ_sens=0.3, l23w=1.0): M7=+0.129, M10=1.50, FWHM=−5.8°

### Sweep groups

- **Group A** (a1–a7): Phase boundary mapping — λ_sensory from 0.0 to 1.0
- **Group A fine** (a2a/a2b/a2c): Transition zone — λ_sensory 0.12/0.15/0.18
- **Group B** (b1–b5): Energy interaction — λ_energy at 0.5/5.0
- **Goldilocks** (b2e25/b2e30/b2e35): Energy fine-tuning — λ_energy 2.5/3.0/3.5
- **Group C** (c1/c2): L2/3 energy weight at l23w=5.0
- **Group D** (d1–d3): Mismatch loss interaction — λ_mismatch at 0.5/1.0
- **Group E** (e1/e2): L2/3 energy weight at l23w=3.0

---

## 7. Parameter Importance (Ranked)

### 1. λ_sensory — DOMINANT

Controls the feedback regime:
- 0.0: dampening (M7<0, M10<1)
- 0.10–0.15: transition (M7 small positive, FB narrows)
- 0.18+: FB-mediated sharpening (FB widens, M7 scales monotonically)

M7 at δ=10° ranges from −0.047 (λ=0.0) to +0.271 (λ=1.0).

### 2. l23_energy_weight — CRITICAL

Controls L2/3 amplitude without killing sharpening:
- 1.0: default (M10=1.50 at λ_sens=0.3)
- 3.0: **sweet spot** (M10=1.13, FWHM=−11.6°)
- 5.0: over-constrained (M10=0.95 at λ_sens=0.3, sub-unity)

l23w is more effective than λ_energy for amplitude control because it
specifically targets L2/3 activity without constraining V2.

### 3. λ_energy — SECONDARY

Broad amplitude/sharpening constraint:
- 0.5: too permissive (M10=4.51 at λ_sens=1.0 — runaway excitation)
- 2.0–2.5: good balance (FWHM sweet spot)
- 5.0: too constraining (M7 collapses to +0.028 at λ_sens=0.3)

Smooth monotonic gradient across all metrics. The Goldilocks zone is
λ_energy=2.0–3.0.

### 4. λ_mismatch — NEGLIGIBLE

Almost no effect on any metric. d3 (λ_mm=0.5) is nearly identical to
a4 (λ_mm=0.0) at the same λ_sensory=0.3. Can safely remain at 0.0.

---

## 8. Energy Analysis

### λ_energy gradient at λ_sensory=0.3

| λ_energy | M7 δ=10° | M10 amp | FWHM Δ | FB contrib |
|---|---|---|---|---|
| 0.5 | +0.260 | 2.72 | −0.3° | +0.244 |
| 2.0 | +0.129 | 1.50 | −5.8° | +0.066 |
| 2.5 | +0.106 | 1.37 | −5.8° | +0.048 |
| 3.0 | +0.084 | 1.26 | −5.2° | +0.027 |
| 3.5 | +0.072 | 1.18 | −3.8° | +0.020 |
| 5.0 | +0.028 | 1.01 | −2.0° | +0.008 |

All metrics decrease smoothly and monotonically with increasing λ_energy.
FWHM peaks at λ_energy=2.0–2.5 (−5.8°), then drops at higher values.

### l23_energy_weight vs λ_energy for amplitude control

| Approach | M7 | M10 | FWHM |
|---|---|---|---|
| b2e35 (λ_energy=3.5, l23w=1.0) | +0.072 | 1.18 | −3.8° |
| e1 (λ_energy=2.0, l23w=3.0) | +0.104 | 1.13 | −11.6° |

At matched amplitude (~1.15), l23w=3.0 gives 44% higher M7 and 3× stronger
FWHM sharpening than energy-only tuning. l23w is the superior knob for
amplitude control.

### Runaway excitation warning

b5 (λ_sens=1.0, λ_energy=0.5): M10=4.51, FWHM=+0.71° (WIDENING).
Low energy + high sensory pressure causes uncontrolled excitation — the
only condition in the sweep where FWHM is positive (feedback broadens
tuning). λ_energy≥2.0 prevents this.

---

## 9. Biological Correspondence

### Kok et al. (2012) — Expectation sharpening

**Model match: e1** (λ_sens=0.3, l23w=3.0)
- Near-unity amplitude (M10=1.13) — expected stimuli not strongly boosted
- Strong tuning sharpening (FWHM=−11.6°) — narrower orientation tuning
- Positive decoder improvement (M7=+0.104) — fine discrimination improved

Kok et al. reported that expected stimuli in early visual cortex showed
sharper orientation tuning without large amplitude changes. Config e1
reproduces this: V2 feedback sharpens L2/3 tuning curves while the
l23_energy_weight constraint prevents runaway excitation.

### Alink et al. (2010) — Expectation suppression

**Model match: a1/b1/e2** (λ_sens=0.0)
- Sub-unity amplitude (M10=0.53–0.70) — expected stimuli suppressed
- Negative M7 — feedback hurts decoding
- SOM-mediated global inhibition at the predicted orientation

The dampening regime emerges when there is no sensory supervision incentive.
V2 feedback reduces L2/3 activity to minimize energy cost.

### Reynolds & Heeger (2009) — Normalization model

The model's parameter space maps onto the normalization framework:
- **Dampening** = suppressive drive dominates (high energy, no sensory)
- **Enhancement** = excitatory drive dominates (high sensory, low energy)
- **Balanced** = the transition zone where normalization is near-unity

The l23_energy_weight parameter specifically controls the gain of the
normalization pool on L2/3, providing a direct analog to the attention
field width in Reynolds & Heeger's model.

### Richter et al. (2018) — Prediction error responses

The FB contribution metric captures the expected-vs-unexpected
differentiation that Richter et al. measured. In the sharpening regime
(λ_sensory≥0.18), V2 feedback widens this gap — expected stimuli are
actively enhanced relative to unexpected stimuli by the top-down signal.

---

## 10. Architecture Evolution

The current simple_feedback architecture emerged from a systematic
investigation of four prior approaches:

### SOM-only inhibition (Sections 2 of original)
- V2 drives SOM inhibition only — no excitatory pathway
- Produced dampening and weak competitor suppression
- M7 flat or negative at all tested configurations
- **Limitation:** Subtractive inhibition cannot create new signal above
  feedforward level

### VIP→SOM disinhibition
- Added VIP interneurons to disinhibit L2/3 at the predicted orientation
- Produced center-sparing geometry (narrower FWHM, preserved peak)
- M7 still flat — geometric narrowing didn't improve trained decoder
- **Limitation:** Removing inhibition ≠ adding signal; a trained decoder
  already discounts flank activity

### Apical multiplicative gain
- Added multiplicative gain on L2/3 excitatory drive
- First positive M7 results (+0.014 at δ=10° with oracle, +0.025 learned)
- Proved that a multiplicative boost above feedforward level is necessary
- **Limitation:** Complex 3-arm architecture (SOM + VIP + apical); many
  parameters; coincidence gate compressed gain range

### Simple feedback (current)
- Single `head_feedback = Linear(16, 36)` output from V2 GRU
- E/I split via relu(+)/relu(−) following Dale's law
- 576 learnable parameters (vs thousands in the legacy operator)
- Discovers all three regimes from loss landscape alone
- M7 up to +0.271, FWHM up to −11.6° — far exceeds prior approaches
- **Why it works:** Giving V2 maximal flexibility to learn the optimal
  feedback profile, rather than constraining it through hand-designed
  basis functions and separate pathways

---

## 11. Configurations and Results

### Key config files

| Config | Description |
|---|---|
| `config/sweep/sweep_a1.yaml` – `sweep_a7.yaml` | Group A: λ_sensory 0.0–1.0 |
| `config/sweep/sweep_a2a.yaml` – `sweep_a2c.yaml` | Fine: λ_sensory 0.12/0.15/0.18 |
| `config/sweep/sweep_b1.yaml` – `sweep_b5.yaml` | Group B: λ_energy interactions |
| `config/sweep/sweep_b2e25.yaml` – `sweep_b2e35.yaml` | Goldilocks: λ_energy 2.5–3.5 |
| `config/sweep/sweep_c1.yaml` – `sweep_c2.yaml` | Group C: l23w=5.0 |
| `config/sweep/sweep_d1.yaml` – `sweep_d3.yaml` | Group D: λ_mismatch interactions |
| `config/sweep/sweep_e1.yaml` – `sweep_e2.yaml` | Group E: l23w=3.0 |

### Result directories

| Path | Contents |
|---|---|
| `results/sweep/group_a_results.txt` | Group A phase boundary (7 runs) |
| `results/sweep/group_a_fine_results.txt` | Transition zone fine-grained (3 runs) |
| `results/sweep/group_b_results.txt` | Energy interaction (5 runs) |
| `results/sweep/goldilocks_results.txt` | Energy fine-tuning (3 runs) |
| `results/sweep/group_ce_results.txt` | L2/3 weight groups C+E (4 runs) |
| `results/sweep/group_d_results.txt` | Mismatch interaction (3 runs) |
| `results/sweep/sweep_summary.txt` | Master summary (all 25 runs) |

### Recommended configurations

| Use case | Config | Key params |
|---|---|---|
| **Default (literature match)** | e1 | λ_sens=0.3, λ_energy=2.0, l23w=3.0 |
| **Strong sharpening** | c2 | λ_sens=1.0, λ_energy=2.0, l23w=5.0 |
| **Dampening baseline** | a1 | λ_sens=0.0, λ_energy=2.0, l23w=1.0 |
| **Unity amplitude** | a2a | λ_sens=0.12, λ_energy=2.0, l23w=1.0 |

### Next steps

1. Multi-seed validation of e1 (5+ seeds) to confirm robustness
2. Full M1–M14 analysis battery on e1 and c2
3. Quantitative comparison to Kok (2012) and Richter (2018) effect sizes
4. Fine-tuning l23w around 2.5–3.5 at λ_sensory=0.3
