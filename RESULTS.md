# Results

## Overview

A single laminar V1-V2 architecture -- trained end-to-end with BPTT --
discovers three distinct feedback regimes from the loss landscape alone.
No architectural switches are needed: the balance between sensory
supervision (lambda_sensory) and energy cost (lambda_energy) determines
whether V2 feedback dampens, preserves, or sharpens L2/3 representations.
A 25-run parameter sweep across 7 groups maps the full phase space.

**Headline result:** Config e1 (lambda_sensory=0.3, l23_energy_weight=3.0)
produces near-unity amplitude (M10=1.13) with the strongest tuning
sharpening in the sweep (FWHM=-11.6 deg), closely matching Kok et al.
(2012) experimental findings.

---

## 1. Three Feedback Regimes

The model's feedback behavior is controlled primarily by lambda_sensory,
the weight on 36-way cross-entropy orientation decoding from L2/3 activity.

### Phase diagram (10 lambda_sensory values, lambda_energy=2.0, l23w=1.0)

| lambda_sensory | M7 d=10 | M10 amp | FWHM D | FB contrib | Regime |
|---|---|---|---|---|---|
| 0.00 | -0.047 | 0.70 | -3.9 deg | +0.143 | **Dampening** |
| 0.10 | +0.019 | 0.95 | -3.5 deg | -0.003 | Transition |
| 0.12 | +0.028 | 1.00 | -3.2 deg | -0.001 | **Unity point** |
| 0.15 | +0.053 | 1.09 | -2.0 deg | -0.004 | Transition |
| 0.18 | +0.071 | 1.18 | -4.3 deg | +0.005 | **FB flip** |
| 0.20 | +0.081 | 1.23 | -5.2 deg | +0.014 | Enhancement |
| 0.30 | +0.129 | 1.50 | -5.8 deg | +0.066 | Enhancement |
| 0.50 | +0.203 | 1.96 | -1.6 deg | +0.163 | Strong sharpening |
| 0.70 | +0.238 | 2.32 | -1.7 deg | +0.287 | Strong sharpening |
| 1.00 | +0.271 | 2.77 | -1.9 deg | +0.398 | Strong sharpening |

### Regime definitions

**Dampening (lambda_sensory=0.0):** Without sensory supervision, the energy
cost dominates and V2 learns to suppress L2/3 activity at the predicted
orientation via SOM inhibition. M7 is negative (feedback hurts decoding),
amplitude is sub-unity (M10=0.70), and the feedback widens the
expected-vs-unexpected gap by suppressing expected stimuli more. This maps
to the global suppression reported by Alink et al. (2010).

**Transition zone (lambda_sensory=0.10-0.15):** The mechanism switches modes.
M7 turns positive but FB contribution is negative (NARROWS) -- improvement
comes from direct sensory optimization, not feedback-mediated
differentiation. The exact unity amplitude point is at lambda_sensory=0.12
(M10=0.9999).

**FB-mediated sharpening (lambda_sensory>=0.18):** FB contribution flips
positive (WIDENS the expected-vs-unexpected gap). V2 feedback now actively
enhances decoding by differentiating expected from unexpected stimuli.
M7 increases monotonically with lambda_sensory. This maps to Kok et al.
(2012) expectation sharpening.

### Key transitions

1. **Phase boundary** (dampening to enhancement): Between lambda_sensory=0.0
   and 0.10 -- M7 flips from negative to positive.
2. **Unity amplitude** at lambda_sensory=0.12 -- M10=1.000 exactly.
3. **FB contribution sign flip** between lambda_sensory=0.15 (NARROWS) and
   0.18 (WIDENS) -- feedback switches from passive to active role.

---

## 2. Best Dampening Configuration

**Config b1:** lambda_sensory=0.0, lambda_energy=5.0, l23w=1.0

| Metric | Value |
|---|---|
| M7 d=10 | -0.060 |
| M10 amplitude | 0.643 |
| FWHM D | -3.0 deg |
| FB contribution | +0.153 (WIDENS) |

### Mechanism

Without sensory loss, V2 feedback minimizes the energy cost by suppressing
L2/3 activity wherever it can predict the stimulus. The feedback head
learns a profile that drives SOM inhibition at the predicted orientation,
producing sub-unity amplitude (36% reduction). Higher lambda_energy
strengthens this effect: M10 drops from 0.70 (lambda_energy=2.0) to 0.64
(lambda_energy=5.0).

The dampening regime is robust across all tested parameter combinations --
it emerges whenever lambda_sensory=0, regardless of energy weight or l23w.
Increased l23w amplifies dampening further: e2 (l23w=3.0) produces the
strongest dampening (M10=0.53).

### Biological correspondence

Maps to Alink et al. (2010) -- global suppression of expected stimuli in
early visual cortex. The model shows this is the default behavior when
higher areas have no incentive to preserve sensory fidelity.

---

## 3. Best Sharpening Configurations

### e1: Near-unity amplitude + strong sharpening (BEST OVERALL)

**Config:** lambda_sensory=0.3, lambda_energy=2.0, l23w=3.0

| Metric | Value |
|---|---|
| M7 d=10 | +0.104 |
| M10 amplitude | 1.13 |
| FWHM D | **-11.6 deg** |
| FB contribution | +0.002 (WIDENS) |

This is the best literature match. Near-unity amplitude with strong tuning
sharpening reproduces the signature reported by Kok et al. (2012): expected
stimuli show subtly different amplitude but significantly sharper tuning.
The l23_energy_weight=3.0 specifically constrains L2/3 activity without
limiting V2 freedom, achieving a balance that pure lambda_energy tuning
cannot.

### Sharpening mechanism

V2's GRU `head_feedback` (Linear(16,36)) outputs a raw 36-channel signal.
Dale's law E/I split:
- `center_exc = relu(+feedback)` -> additive excitation to L2/3
- `som_drive = relu(-feedback)` -> SOM inhibition of L2/3

At lambda_sensory>=0.18, V2 learns to:
1. Excite L2/3 at the predicted orientation (center_exc > 0)
2. Inhibit L2/3 at flanking orientations via SOM (som_drive > 0)
3. This sharpens the population response: narrower FWHM + enhanced peak

The sensory loss prevents energy-driven collapse -- L2/3 must maintain
decodable orientation representations, so V2 cannot simply suppress
everything.

---

## 4. Expected vs Unexpected Analysis

### Feedback contribution metric

The FB contribution measures whether feedback widens or narrows the gap
between expected and unexpected L2/3 responses:
- **WIDENS (positive):** Feedback enhances expected stimuli more than
  unexpected -> active predictive differentiation
- **NARROWS (negative):** Feedback reduces the natural gap -> feedback is
  counterproductive for differentiation

### Sign flip at the transition boundary

| lambda_sensory | FB contribution | Direction |
|---|---|---|
| 0.10 | -0.003 | NARROWS |
| 0.12 | -0.001 | NARROWS |
| 0.15 | -0.004 | NARROWS |
| **0.18** | **+0.005** | **WIDENS** |
| 0.20 | +0.014 | WIDENS |
| 0.30 | +0.066 | WIDENS |

The sign flip occurs between lambda_sensory=0.15 and 0.18. Below this
boundary, any decoder improvement comes from internal L2/3 optimization
(recurrent sharpening, better tuning), not from V2 feedback. Above it, V2
actively contributes to expected-vs-unexpected differentiation.

---

## 5. Full 25-Run Parameter Sweep

### Complete results table

All runs: stage1=2000 steps, stage2=5000 steps, seed=42.

| Run | l_sens | l_energy | l_mm | l23w | M7 d=10 | M10 amp | FWHM D | FB contrib | FB dir |
|---|---|---|---|---|---|---|---|---|---|
| a1 | 0.0 | 2.0 | 0.0 | 1.0 | -0.047 | 0.70 | -3.9 | +0.143 | WIDENS |
| a2 | 0.10 | 2.0 | 0.0 | 1.0 | +0.019 | 0.95 | -3.5 | -0.003 | NARROWS |
| a2a | 0.12 | 2.0 | 0.0 | 1.0 | +0.028 | 1.00 | -3.2 | -0.001 | NARROWS |
| a2b | 0.15 | 2.0 | 0.0 | 1.0 | +0.053 | 1.09 | -2.0 | -0.004 | NARROWS |
| a2c | 0.18 | 2.0 | 0.0 | 1.0 | +0.071 | 1.18 | -4.3 | +0.005 | WIDENS |
| a3 | 0.20 | 2.0 | 0.0 | 1.0 | +0.081 | 1.23 | -5.2 | +0.014 | WIDENS |
| a4 | 0.30 | 2.0 | 0.0 | 1.0 | +0.129 | 1.50 | -5.8 | +0.066 | WIDENS |
| a5 | 0.50 | 2.0 | 0.0 | 1.0 | +0.203 | 1.96 | -1.6 | +0.163 | WIDENS |
| a6 | 0.70 | 2.0 | 0.0 | 1.0 | +0.238 | 2.32 | -1.7 | +0.287 | WIDENS |
| a7 | 1.00 | 2.0 | 0.0 | 1.0 | +0.271 | 2.77 | -1.9 | +0.398 | WIDENS |
| b1 | 0.0 | 5.0 | 0.0 | 1.0 | -0.060 | 0.64 | -3.0 | +0.153 | WIDENS |
| b2 | 0.3 | 5.0 | 0.0 | 1.0 | +0.028 | 1.01 | -2.0 | +0.008 | WIDENS |
| b2e25 | 0.3 | 2.5 | 0.0 | 1.0 | +0.106 | 1.37 | -5.8 | +0.048 | WIDENS |
| b2e30 | 0.3 | 3.0 | 0.0 | 1.0 | +0.084 | 1.26 | -5.2 | +0.027 | WIDENS |
| b2e35 | 0.3 | 3.5 | 0.0 | 1.0 | +0.072 | 1.18 | -3.8 | +0.020 | WIDENS |
| b3 | 1.0 | 5.0 | 0.0 | 1.0 | +0.180 | 1.86 | -2.3 | +0.184 | WIDENS |
| b4 | 0.3 | 0.5 | 0.0 | 1.0 | +0.260 | 2.72 | -0.3 | +0.244 | WIDENS |
| b5 | 1.0 | 0.5 | 0.0 | 1.0 | +0.323 | 4.51 | +0.7 | +0.997 | WIDENS |
| c1 | 0.3 | 2.0 | 0.0 | 5.0 | +0.081 | 0.95 | -4.3 | +0.014 | WIDENS |
| c2 | 1.0 | 2.0 | 0.0 | 5.0 | +0.196 | 1.77 | -7.0 | +0.207 | WIDENS |
| d1 | 0.1 | 2.0 | 1.0 | 1.0 | +0.026 | 1.02 | -2.8 | -0.026 | NARROWS |
| d2 | 0.5 | 2.0 | 1.0 | 1.0 | +0.214 | 2.13 | -1.5 | +0.073 | WIDENS |
| d3 | 0.3 | 2.0 | 0.5 | 1.0 | +0.137 | 1.52 | -5.8 | +0.009 | WIDENS |
| e1 | 0.3 | 2.0 | 0.0 | 3.0 | +0.104 | 1.13 | -11.6 | +0.002 | WIDENS |
| e2 | 0.0 | 2.0 | 0.0 | 3.0 | -0.078 | 0.53 | -4.5 | +0.220 | WIDENS |

### Top 5 candidates (ranked by M7 + FWHM, M10 < 2.0)

1. **e1** (l_sens=0.3, l23w=3.0): M7=+0.104, M10=1.13, FWHM=-11.6 deg -- **BEST OVERALL**
2. **c2** (l_sens=1.0, l23w=5.0): M7=+0.196, M10=1.77, FWHM=-7.0 deg
3. **b2e25** (l_sens=0.3, l_e=2.5): M7=+0.106, M10=1.37, FWHM=-5.8 deg
4. **d3** (l_sens=0.3, l_mm=0.5): M7=+0.137, M10=1.52, FWHM=-5.8 deg
5. **a4** (l_sens=0.3, l23w=1.0): M7=+0.129, M10=1.50, FWHM=-5.8 deg

---

## 6. Parameter Importance (Ranked)

### 1. lambda_sensory -- DOMINANT

Controls the feedback regime:
- 0.0: dampening (M7<0, M10<1)
- 0.10-0.15: transition (M7 small positive, FB narrows)
- 0.18+: FB-mediated sharpening (FB widens, M7 scales monotonically)

### 2. l23_energy_weight -- CRITICAL

Controls L2/3 amplitude without killing sharpening:
- 1.0: default (M10=1.50 at l_sens=0.3)
- 3.0: **sweet spot** (M10=1.13, FWHM=-11.6 deg)
- 5.0: over-constrained (M10=0.95 at l_sens=0.3, sub-unity)

### 3. lambda_energy -- SECONDARY

Broad amplitude/sharpening constraint:
- 0.5: too permissive (runaway excitation)
- 2.0-2.5: good balance
- 5.0: too constraining

### 4. lambda_mismatch -- NEGLIGIBLE

Almost no effect on any metric.

---

## 7. Biological Correspondence

### Kok et al. (2012) -- Expectation sharpening

**Model match: e1** (l_sens=0.3, l23w=3.0)
- Near-unity amplitude (M10=1.13) -- expected stimuli not strongly boosted
- Strong tuning sharpening (FWHM=-11.6 deg) -- narrower orientation tuning
- Positive decoder improvement (M7=+0.104) -- fine discrimination improved

### Alink et al. (2010) -- Expectation suppression

**Model match: a1/b1/e2** (l_sens=0.0)
- Sub-unity amplitude (M10=0.53-0.70) -- expected stimuli suppressed
- Negative M7 -- feedback hurts decoding
- SOM-mediated global inhibition at the predicted orientation

### Reynolds & Heeger (2009) -- Normalization model

The model's parameter space maps onto the normalization framework:
- **Dampening** = suppressive drive dominates
- **Enhancement** = excitatory drive dominates
- **Balanced** = transition zone where normalization is near-unity

---

## 8. Recommended Configurations

| Use case | Config | Key params |
|---|---|---|
| **Default (literature match)** | e1 | l_sens=0.3, l_energy=2.0, l23w=3.0 |
| **Strong sharpening** | c2 | l_sens=1.0, l_energy=2.0, l23w=5.0 |
| **Dampening baseline** | a1 | l_sens=0.0, l_energy=2.0, l23w=1.0 |
| **Unity amplitude** | a2a | l_sens=0.12, l_energy=2.0, l23w=1.0 |

### Next steps

1. Multi-seed validation of e1 (5+ seeds) to confirm robustness
2. Full M1-M14 analysis battery on e1 and c2
3. Quantitative comparison to Kok (2012) and Richter (2018) effect sizes
4. Fine-tuning l23w around 2.5-3.5 at lambda_sensory=0.3

---

## 9. Rescue chain results (failed-dual-regime-experiments branch)

The 25-run sweep above maps the 3 feedback regimes that emerge from the
**baseline** single-network architecture by varying loss weights alone.
Subsequent work on branch `failed-dual-regime-experiments` attempted to
recover **task-state-selective** sharpening AND dampening from a single
checkpoint by adding architectural rescues (R1+R2 / R3 / R4 / R5).

### Cross-checkpoint summary (re-centered tuning, Relevant task_state, late-ON t=9)

Per-trial `r_l23` curves are stimulus-aligned (`np.roll` so true_ch lands
at the array midpoint), then averaged within (regime × bucket) buckets.
FWHM uses linear interpolation at half-max crossings. ≥ 1500 trials per
panel.

| Checkpoint | Rel-Exp total | Rel-Unexp total | Δ total | Rel-Exp peak | Rel-Unexp peak | Δ peak | Exp FWHM | Unexp FWHM | Δ FWHM |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (simple_dual) | 6.93 | 7.17 | +3.5% | 0.987 | 0.992 | +0.5% | 29.3° | 28.2° | +1.1° |
| Rescue 1+2 | 4.05 | 4.78 | +18%  | 0.564 | 0.639 | +13%  | 33.3° | 31.8° | +1.5° |
| Rescue 3   | 4.00 | 4.61 | +15%  | 0.602 | 0.605 | +0.5% | 31.7° | 30.6° | +1.1° |
| Rescue 4   | 4.21 | 5.02 | +19%  | 0.578 | 0.662 | +15%  | 33.6° | 32.1° | +1.5° |

(Δ is unexpected − expected; positive Δ means expected is **lower / narrower** than unexpected.)

### Take-away

**R4 (DeepTemplate + error-mismatch) exhibits the cleanest preserved-shape
dampening signature** (peak −15%, total −19%, FWHM matched within 1.5°
between expected and unexpected trials). This is the closest match to
Richter (2018) preserved-shape dampening among the four checkpoints. R1+R2
shows similar peak+total dampening. R3 (VIP) shows total-only dampening
(divisive). Baseline shows essentially no expectation modulation.

The dampening signature appears in **both** Relevant and Irrelevant
task_state with similar magnitude. The preregistered BOTH-regime criterion
(focused → Kok sharpening, routine → Richter dampening from one network)
is **not** met — task_state does not gate the representational mode.

### Figures

- `docs/figures/tuning_ring_recentered_baseline.png`
- `docs/figures/tuning_ring_recentered_r1_2.png`
- `docs/figures/tuning_ring_recentered_r3.png`
- `docs/figures/tuning_ring_recentered_r4.png` ← main visual result

### Full writeup

`docs/rescues_1_to_4_summary.md` (and its 2026-04-13 update section
correcting the earlier "subtractive predictive coding" interpretation).

### 2026-04-14 dampening-analysis follow-up (aligned pure R1+2 only)

The `dampening-analysis` branch now also carries a narrower follow-up on the
aligned pure-R1+2 checkpoint
`r12_fb24_sharp_050_width_075_rec11_aligned`. This is **not** a new all-rescue
comparison; it adds raw/delta/baseline surfaces plus paired-state
branch-counterfactual analysis for that one aligned checkpoint only.

Source-of-truth artefacts:
- `results/r12_fb24_sharp_050_width_075_rec11_aligned/tuning_ring_recentered_raw_branch_counterfactual.json`
- `results/r12_fb24_sharp_050_width_075_rec11_aligned/tuning_ring_recentered_delta_branch_counterfactual.json`
- `results/r12_fb24_sharp_050_width_075_rec11_aligned/tuning_ring_recentered_baseline_branch_counterfactual.json`

Relevant branch-counterfactual summary:
- `baseline`: expected and unexpected are identical after the centering fix
  (`total=3.517256`, `peak=0.375691`, `FWHM=39.389691°` for both).
- `raw`: expected is lower and slightly narrower than unexpected
  (`peak 0.449558 vs 0.507532`, `FWHM 33.237447° vs 33.515513°`).
- `delta`: expected is much lower and much narrower than unexpected
  (`peak 0.072438 vs 0.491614`, `FWHM 27.581737° vs 45.064195°`).

The baseline-centering bug was analysis-only. Branch-counterfactual baseline
mode returned the same pre-probe tensor for both branches, but the collector
used different recentering channels, rotating identical baselines into
artificially opposite shapes. Baseline mode now uses the shared
predicted/expected channel for both branches.
