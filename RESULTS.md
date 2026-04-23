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

---

## 10. R1+R2 paired ex/unex eval — Decoder C (`dampening-analysis` branch, 2026-04-17)

**Default checkpoint on `dampening-analysis`:** R1+R2 simple_dual emergent_seed42
(`results/simple_dual/emergent_seed42/checkpoint.pt` on the remote). All
expectation-suppression / dampening-vs-sharpening analyses on this branch
are anchored to this checkpoint from 2026-04-17 onwards.

### Decoder C (post-hoc, used here)

Standalone `Linear(36, 36)` trained on 100k synthetic orientation-bump
patterns (50k single-orientation σ=3 ch with amplitudes ∈ [0.1, 2.0]; 50k
multi-orientation K∈{2,3} with strictly-max amplitude as the label;
Gaussian noise σ=0.02). Adam lr=1e-3, batch 256, ≤30 epochs, early-stop
patience 3, seed 42. Saved at `checkpoints/decoder_c.pt`. Held-out
synthetic accuracy 0.81 (single 0.98 / multi 0.65); real-network
natural-HMM R1+R2 accuracy 0.66 non-amb / 0.53 all. **Preferred decoder
for expectation-suppression analyses.** See `ARCHITECTURE.md` § "Decoders"
and the 2026-04-17 Decoder A artefact note in
`docs/rescues_1_to_4_summary.md`.

### Paired ex/unex design

12 N values (4..15) × 200 trials/N = 2400 paired ex/unex trials, run on
R1+R2. Per-trial RNG seed = `42 + trial_idx` (independent of N → bit-
identical pre-probe march for the same trial_idx across N values). Random
S ∈ [0°, 180°), D ∈ [25°, 90°], CW/CCW 50/50 per trial.
`task_state = [1, 0]` (focused) throughout, contrast 1.0. Cue at the
expected-next orientation in **both** branches (so unex cue is "wrong" by
D degrees). Pre-probe state shared across branches — only the probe-ON
window diverges. Readout: probe-ON window steps `[9:11]` mean-pooled,
followed by per-trial roll-to-center on the true probe channel
(peak at ch18) and linear-interpolation FWHM (same convention as
commit `ce1b34e` / `matched_hmm_ring_sequence.py`).

### Pooled across N (n=2400 paired trials)

| Metric | Expected | Unexpected | Δ (ex − unex) |
|---|---:|---:|---:|
| Decoder C top-1 accuracy | 0.707 ± 0.009 | 0.581 ± 0.010 | +0.125 |
| Net L2/3 (sum 36 channels) | 4.99 ± 0.01 | 6.13 ± 0.02 | −1.15 |
| Peak at true channel (re-centered ch18) | 0.773 ± 0.003 | 0.626 ± 0.004 | +0.147 |
| FWHM (linear-interp half-max) | 28.4° ± 0.10 | 29.8° ± 0.19 | −1.33° |

Sign-consistency: all four signs hold at every N from 4 to 15 (no
exceptions). Pre-probe state is bit-identical across branches
(`pre_probe_max_abs_diff = 0.00e+00` per N, and overall). All 2400 trials
produced valid FWHM crossings in both branches.

In plain language: expected trials show **lower net L2/3 activity, higher
peak at the stimulus channel, narrower tuning, and higher decoding
accuracy** than unexpected trials.

### Interpretation framing (literal)

- **Operational dampening (lower activity AND lower decoding on expected):**
  Passes on net L2/3 (expected lower) but fails on decoding (expected
  higher), peak (expected higher), and FWHM (expected narrower).
- **Kok 2012 sharpening (narrower tuning, higher peak, better decoding,
  lower total activity on expected):** Matches on all four axes.
- **Richter 2018 preserved-shape dampening (lower peak, preserved FWHM,
  preserved decoding on expected):** Does not match — the peak goes the
  wrong direction (expected has the higher peak, not the lower one).

No mechanism interpretation beyond these literal comparisons.

### Reproducibility

- Eval script: `scripts/eval_ex_vs_unex_decC.py`.
- Decoder training: `scripts/train_decoder_c.py` → `checkpoints/decoder_c.pt`.
- Result JSON: `results/eval_ex_vs_unex_decC.json` (per-N entries + pooled,
  including `peak_at_stim_*`, `fwhm_deg_*` with `n_valid` counts, and
  `delta_*` for each metric).
- Figure: `docs/figures/eval_ex_vs_unex_decC.png` (4-panel: dec_acc,
  net_L2/3, peak_at_stim, FWHM vs N).
- Run log: `logs/eval_ex_vs_unex_decC_t13.log`.

---

## 11. Cross-decoder comprehensive matrix (Task #26, 2026-04-22)

One forward pass per trial; three orientation decoders (A / B / C) applied to
the same `r_l23`. See `ARCHITECTURE.md` § "Decoders" for decoder definitions.
A single 17-row matrix aggregates:

- 4 paired-HMM-fork conditions on R1+R2 (see § 12)
- HMM C1 (focused + HMM cue) on legacy reference networks a1, b1, c1, e1 (see § 13)
- 6 native-strategy evaluations on R1+R2 (NEW paired march, M3R, HMS, HMS-T, P3P, VCD)
- 3 modified-input (focused + march cue) evaluations on R1+R2 (M3R, HMS-T, VCD)

### Compact Δ = acc(expected) − acc(unexpected)

| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A′ | Δ_B | Δ_C | majority | outlier |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 | 1000 | 1000 | +0.3090 | +0.3070 | +0.0150 | +0.0500 | + | — |
| 2 | HMM C2 (routine + HMM cue) | R1+R2 | 1000 | 1000 | +0.1580 | +0.1690 | −0.0170 | +0.0060 | + | B |
| 3 | HMM C3 (focused + zero cue) | R1+R2 | 1000 | 1000 | +0.3000 | +0.2820 | −0.0070 | +0.0380 | + | B |
| 4 | HMM C4 (routine + zero cue) | R1+R2 | 1000 | 1000 | +0.1550 | +0.1560 | −0.0360 | +0.0390 | + | B |
| 5 | HMM C1 | a1 legacy | 1000 | 1000 | −0.0220 | — | +0.0000 | −0.0090 | − | B |
| 6 | HMM C1 | b1 legacy | 1000 | 1000 | −0.0320 | — | −0.0150 | −0.0230 | − | — |
| 7 | HMM C1 | c1 legacy | 1000 | 1000 | +0.1870 | — | +0.0370 | −0.0070 | + | C |
| 8 | HMM C1 | e1 legacy | 1000 | 1000 | +0.2130 | — | +0.0510 | +0.0110 | + | — |
| 9 | NEW (paired march) | R1+R2 | 2400 | 2400 | +0.3871 | +0.3888 | +0.0854 | +0.1254 | + | — |
| 10 | M3R (matched_3row_ring) | R1+R2 | 1084 | 3302 | −0.1496 | −0.0887 | −0.0082 | −0.0294 | − | — |
| 11 | HMS (matched_hmm_ring_sequence) | R1+R2 | 3074 | 153 | −0.1850 | −0.1556 | −0.1103 | +0.0790 | − | C |
| 12 | HMS-T (--tight-expected) | R1+R2 | 793 | 101 | −0.2919 | −0.2112 | −0.1818 | −0.0631 | − | — |
| 13 | P3P (matched_probe_3pass) | R1+R2 | 38 | 38 | +0.3684 | +0.3902 | −0.1714 | +0.0526 | + | B |
| 14 | VCD-test3 (v2_confidence_dissection) | R1+R2 | 8025 | 8025 | −0.1655 | −0.1987 | −0.0984 | −0.0703 | − | — |
| 15 | M3R (modified: focused + march cue) | R1+R2 | 3260 | 6486 | −0.1373 | −0.1122 | −0.0336 | −0.0176 | − | — |
| 16 | HMS-T (modified: focused + march cue) | R1+R2 | 1139 | 111 | −0.2968 | −0.2031 | −0.1222 | +0.0443 | − | C |
| 17 | VCD-test3 (modified: focused + march cue) | R1+R2 | 6998 | 6998 | −0.0840 | −0.1165 | −0.0264 | −0.0130 | − | — |

(Majority and outlier columns are computed from the **original Δ_A / Δ_B / Δ_C**
triple — the 2026-04-22 matrix. `Δ_A′` values come from
`results/cross_decoder_comprehensive_decAprime.json` (2026-04-23 rerun with
Dec A′ in place of Dec A on R1+R2 rows; legacy rows retain their own stored
Dec A and so have `—` in the Δ_A′ column). "Outlier" = decoder whose sign
disagrees with the 2/3 majority; "—" in that column means all three agree.
Per-row raw ex/unex accuracies are in
`results/cross_decoder_comprehensive.md` (original) and
`results/cross_decoder_comprehensive_decAprime.md` (Dec A′ rerun).)

### Dec A → Dec A′ swap summary (13 R1+R2 rows, 2026-04-23)

Dec A′ is Dec A retrained for 5000 Adam steps on `r_l23` streamed through the
fully-trained, frozen R1+R2 network (stable-target retrain; see
`ARCHITECTURE.md` § "Decoders" and § "Stable-target decoder sanity check").
Swapping Dec A → Dec A′ on the 13 R1+R2 rows (legacy a1/b1/c1/e1 retain their
own stored Dec A):

- **Zero Δ-sign flips** across all 13 R1+R2 rows (Δ_A′ and Δ_A agree on sign
  everywhere).
- `|Δ_A′ − Δ_A|` ≤ 0.094; median 0.025; mean 0.032. The three largest shifts
  are HMS-T native (+0.081), HMS-T modified (+0.094), and M3R native
  (+0.061) — all on sharpening-side rows, all toward smaller |Δ|.
- Holding Δ_B / Δ_C fixed at their original-run values, zero rows change
  sign-agreement class under the Dec A → Dec A′ swap.
- The Dec A′ rerun's Δ_B / Δ_C are within ≤ 0.03 of the original matrix values
  (same seed, same script; residual CPU FP run-to-run noise). Under the
  run-matched Δ_B / Δ_C two rows shift sign-agreement class: HMM C3 tightens
  to ALL-agree (run Δ_B = +0.024 vs original −0.007 moves it positive) and
  M3R native loosens to B-outlier (run Δ_B = +0.003 vs original −0.008 moves
  it positive). Both shifts are driven by Δ_B noise of ±0.03, not by the
  Dec A → Dec A′ swap. Source:
  `results/cross_decoder_comprehensive_decAprime_diff.{json,md}`.

### Per-decoder profile

| Decoder | n rows | mean \|Δ\| | max \|Δ\| | rows ALL agree | rows w/ majority | rows disagreeing | rows where this is outlier |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 17 | 0.2024 | 0.3871 | 9 | 17 | 0 | — |
| A′ (R1+R2 only) | 13 | 0.2138 | 0.3902 | — | 13 (same signs as A on all 13) | 0 | — |
| B | 17 | 0.0598 | 0.1818 | 9 | 12 | 5 | HMM C2/C3/C4 on R1+R2; HMM C1 on a1; P3P on R1+R2 |
| C | 17 | 0.0399 | 0.1254 | 9 | 14 | 3 | HMM C1 on c1; HMS on R1+R2; HMS-T modified on R1+R2 |

Decoder A produces consistently larger-magnitude Δs (mean |Δ| ≈ 5× that of B/C)
and always agrees with the 2/3 majority sign. Decoder C produces the
smallest-magnitude Δs and agrees with the majority in 14/17 rows. Decoder B is
outlier in 5/17 rows. No row has Decoder A as outlier. Decoder A′ (trained on
stable, post-training L2/3) produces Δ of near-identical magnitude to Dec A
on the same 13 R1+R2 rows (mean |Δ| 0.2138 vs 0.2298 restricted to the same 13
rows) and identical sign on all 13 — the Dec A training-schedule concern
(moving target vs stable target) does not change the 13-row sign pattern.

### Rows where all three decoders agree (9/17)

| # | Assay | Network | Δ_A | Δ_B | Δ_C | common sign |
|---|---|---|---:|---:|---:|:--:|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 | +0.3090 | +0.0150 | +0.0500 | + |
| 6 | HMM C1 | b1 legacy | −0.0320 | −0.0150 | −0.0230 | − |
| 8 | HMM C1 | e1 legacy | +0.2130 | +0.0510 | +0.0110 | + |
| 9 | NEW (paired march) | R1+R2 | +0.3871 | +0.0854 | +0.1254 | + |
| 10 | M3R | R1+R2 | −0.1496 | −0.0082 | −0.0294 | − |
| 12 | HMS-T | R1+R2 | −0.2919 | −0.1818 | −0.0631 | − |
| 14 | VCD-test3 | R1+R2 | −0.1655 | −0.0984 | −0.0703 | − |
| 15 | M3R modified | R1+R2 | −0.1373 | −0.0336 | −0.0176 | − |
| 17 | VCD-test3 modified | R1+R2 | −0.0840 | −0.0264 | −0.0130 | − |

On R1+R2, **two assays are decoder-robust sharpening** (Δ > 0 on all three
decoders): row 1 (HMM C1, focused + HMM cue) and row 9 (NEW paired march).
On the same R1+R2 checkpoint, **five assays are decoder-robust dampening**
(Δ < 0 on all three decoders): rows 10 (M3R), 12 (HMS-T), 14 (VCD), 15
(M3R modified), and 17 (VCD modified). Note that row 11 (plain HMS) and
row 16 (HMS-T modified) are NOT decoder-robust — Dec C flips positive on
both — so they are not in the dampening list. The legacy reference rows
(6: b1 dampening; 8: e1 sharpening) are described in § 13.

### Reproducibility

- Aggregator: `scripts/aggregate_cross_decoder_matrix.py`
- Per-strategy evaluator: `scripts/cross_decoder_eval.py` (contains Dec B 5-fold nearest-centroid helper + per-strategy ex/unex evals)
- Result JSONs: `results/cross_decoder_comprehensive.{json,md}`
- Per-source JSONs: `/tmp/task26_paradigm_R1R2.json`, `/tmp/task26_legacy/{a1,b1,c1,e1}_C1.json`, `/tmp/task26_xdec_native.json`, `/tmp/task26_xdec_modified.json`
- Dec A′ rerun (2026-04-23): training `scripts/train_decoder_a_prime.py` → `checkpoints/decoder_a_prime.pt`; stratified eval `scripts/eval_decoder_a_prime_stratified.py` → `results/decoder_a_prime_stratified_eval.json`; ckpt patch helper `scripts/_make_decAprime_ckpt.py` → `/tmp/r1r2_ckpt_decAprime.pt`; matrix rerun sources `/tmp/task26_paradigm_R1R2_decAprime.json`, `/tmp/task26_xdec_native_decAprime.json`, `/tmp/task26_xdec_modified_decAprime.json`; aggregated matrix `results/cross_decoder_comprehensive_decAprime.{json,md}`; diff `scripts/diff_decAprime_matrix.py` → `results/cross_decoder_comprehensive_decAprime_diff.{json,md}`.

---

## 12. Paired HMM fork paradigm × readout analysis (Tasks #19–#22, 2026-04-19)

Paired-state HMM fork: one shared pre-probe march, two branches (ex / unex) with
bit-identical pre-probe state, then a probe consistent (ex) or 90°-shifted
(unex) from the HMM-predicted orientation. Task #22 adds a 4-condition
factorial — focused/routine × HMM cue / neutral cue — each 1000 paired ex/unex
trials on R1+R2. Task #19 probes the adjacent-channel structure of the tuning
curve on the NEW paired march assay, with march-signed offsets.

### Four-condition decoder accuracy (R1+R2, from `results/r1r2_paradigm_readout.json`)

| Condition | n | decB ex | decB unex | Δ_B | decC ex | decC unex | Δ_C |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 focused + HMM cue | 1000 | 0.604 | 0.586 | +0.018 | 0.626 | 0.538 | +0.088 |
| C2 routine + HMM cue | 1000 | 0.435 | 0.463 | −0.028 | 0.469 | 0.456 | +0.013 |
| C3 focused + neutral cue | 1000 | 0.543 | 0.539 | +0.004 | 0.547 | 0.502 | +0.045 |
| C4 routine + neutral cue | 1000 | 0.435 | 0.471 | −0.036 | 0.457 | 0.416 | +0.041 |

Decoder C shows Δ > 0 (expected > unexpected) in **all four conditions**.
Decoder B shows Δ > 0 only in focused conditions (C1, C3); Δ < 0 in routine
(C2, C4). The Decoder B sign is task-state dependent on this paradigm; the
Decoder C sign is not.

(Note: the cross-decoder matrix § 11 rows 1–4 come from a separate
paradigm-R1+R2 evaluation run (Task #26) and include Decoder A alongside B/C;
numerical values differ slightly from the Task #22 numbers above because they
are independent runs of the same 4-condition protocol. The sign pattern is
consistent across both runs.)

### Peak / net-L2/3 / FWHM (R1+R2, paired HMM fork 4 conditions)

| Condition | Branch | Peak (re-centered true-ch) | Net L2/3 (sum 36 ch) | FWHM (linear-interp) |
|---|---|---:|---:|---:|
| C1 focused + HMM cue | ex | 0.5465 ± 0.006 | 4.19 ± 0.03 | 29.82° ± 0.26 |
| | unex | 0.4769 ± 0.006 | 4.87 ± 0.03 | 28.92° ± 0.26 |
| C2 routine + HMM cue | ex | 0.2653 ± 0.004 | 1.93 ± 0.02 | 29.47° ± 0.21 |
| | unex | 0.2079 ± 0.004 | 2.12 ± 0.02 | 27.47° ± 0.22 |
| C3 focused + neutral | ex | 0.5667 ± 0.007 | 4.33 ± 0.03 | 29.08° ± 0.27 |
| | unex | 0.4588 ± 0.007 | 4.91 ± 0.04 | 27.90° ± 0.27 |
| C4 routine + neutral | ex | 0.2842 ± 0.004 | 2.07 ± 0.02 | 28.50° ± 0.21 |
| | unex | 0.2188 ± 0.004 | 2.27 ± 0.02 | 27.14° ± 0.23 |

Across all 4 conditions on the paired HMM fork: ex **peak** is higher than
unex, ex **net L2/3** is lower than unex, ex **FWHM** is **wider** than unex by
+0.90° (C1) to +2.00° (C2).

### FWHM-sign reversal vs NEW paired march (§ 10)

The FWHM sign is the **opposite** of the NEW paired-march result in § 10,
where ex FWHM (28.44°) was **narrower** than unex (29.77°): Δ_FWHM = −1.33°
(ex < unex). In the paired HMM fork 4-condition eval, ex FWHM is **wider**
than unex by +0.9°–2.0° across all conditions. The peak sign (ex > unex) is
consistent across both paradigms; the net-L2/3 sign (ex < unex) is also
consistent. Only the FWHM sign flips between the two paradigms.

### Adjacent-channel signed-offset curve (NEW paired march, Task #19)

From `results/eval_ex_vs_unex_decC_adjacent.json`. Per-trial roll-to-center
(peak at ch18); offsets sign-flipped by march direction so +offset = ahead in
march direction (the side the march is heading toward).

| Offset (ch) | Offset (°) | ex peak | unex peak | Δ (ex − unex) |
|---:|---:|---:|---:|---:|
| −5 | −25 | 0.131 | 0.163 | −0.031 |
| −4 | −20 | 0.268 | 0.245 | +0.023 |
| −3 | −15 | 0.448 | 0.359 | +0.089 |
| −2 | −10 | 0.629 | 0.485 | +0.144 |
| −1 | −5 | 0.752 | 0.586 | +0.165 |
| 0 | 0 | 0.773 | 0.626 | +0.147 |
| +1 | +5 | 0.689 | 0.589 | +0.100 |
| +2 | +10 | 0.528 | 0.488 | +0.041 |
| +3 | +15 | 0.349 | 0.357 | −0.008 |
| +4 | +20 | 0.191 | 0.230 | −0.039 |
| +5 | +25 | 0.078 | 0.126 | −0.047 |

### Flank-asymmetry diagnostic (expected branch, NEW paired march)

ex(+k) − ex(−k) (march-signed offset, paired-SEM):

| k | ex flank Δ | ± sem |
|---|---:|---:|
| 1 | −0.0631 | 0.0037 |
| 2 | −0.1002 | 0.0036 |
| 3 | −0.0996 | 0.0031 |

unex(+k) − unex(−k) (same convention):

| k | unex flank Δ | ± sem |
|---|---:|---:|
| 1 | +0.0022 | 0.0041 |
| 2 | +0.0030 | 0.0039 |
| 3 | −0.0024 | 0.0034 |

Expected branch shows systematic flank asymmetry: the leading flank (ahead in
march direction, positive offsets) is suppressed relative to the trailing
flank (negative offsets) by 0.06–0.10. Unexpected branch is symmetric within
SEM. **UNTESTED MECHANISM HYPOTHESIS** (no isolating experiment yet): the
ex-branch asymmetry could be consistent with V2 feedback subtracting the
ahead-in-march (predicted-next) orientation from L2/3 on expected trials,
but this has not been confirmed by ablating feedback or by direct
inspection of feedback weights against the asymmetry pattern. The
observation itself — the asymmetry — is empirical; the feedback-subtraction
explanation is a candidate hypothesis only.

### Reproducibility

- Result JSONs: `results/eval_ex_vs_unex_decC_adjacent.json` (Task #19), `results/r1r2_paired_hmm_fork.json` (Task #20), `results/r1r2_paradigm_readout.json` (Task #22).
- Figures: `docs/figures/tuning_ring_matched_3row_r1_2.png`, `docs/figures/tuning_ring_hmm_3row_multicol_r1_2.png`, `docs/figures/tuning_ring_hmm_3row_multicol_tightexp_r1_2.png`, `docs/figures/priming_dose_response_r1_2.png`.

---

## 13. Legacy dampening/sharpening reference networks (Tasks #23–#24, 2026-04-21)

Four checkpoints from the legacy 25-run sweep (§ 5) were reloaded on the
`dampening-analysis` branch and re-evaluated under the full three-decoder
protocol (A / B / C) on HMM C1 (focused + HMM cue), 1000 paired ex/unex
trials per network.

### Legacy § 5 regime classification (from the 25-run sweep)

| Tag | λ_sensory | λ_energy | l23w | Regime | M7 d=10° | M10 amp | FWHM Δ |
|---|---:|---:|---:|---|---:|---:|---:|
| a1 | 0.0 | 2.0 | 1.0 | Dampening | −0.047 | 0.70 | −3.9° |
| b1 | 0.0 | 5.0 | 1.0 | Stronger dampening | −0.060 | 0.64 | −3.0° |
| c1 | 0.3 | 2.0 | 5.0 | Near-unity / transition | +0.081 | 0.95 | −4.3° |
| e1 | 0.3 | 2.0 | 3.0 | BEST sharpening | +0.104 | 1.13 | −11.6° |

### HMM C1 paired-fork Δ on legacy refs (rows 5–8 of § 11)

| Network | n_ex | n_unex | Δ_A | Δ_B | Δ_C | decoder-robust? |
|---|---:|---:|---:|---:|---:|---|
| a1 | 1000 | 1000 | −0.0220 | +0.0000 | −0.0090 | **B outlier** (B = 0.0; A and C both < 0) — matches § 11 row 5 |
| b1 | 1000 | 1000 | −0.0320 | −0.0150 | −0.0230 | **All three < 0 (decoder-robust dampening)** |
| c1 | 1000 | 1000 | +0.1870 | +0.0370 | −0.0070 | **C outlier** — A and B > 0, C slightly < 0 |
| e1 | 1000 | 1000 | +0.2130 | +0.0510 | +0.0110 | **All three > 0 (decoder-robust sharpening)** |

Legacy dampening configs (a1, b1) give Δ ≤ 0 under the paired-fork readout
(b1 is decoder-robust dampening on all three decoders; a1 has B = 0.0 but A
and C both < 0 — a1 is therefore not strictly decoder-robust under the
all-three-agree definition, but A and C agree). Legacy sharpening config
e1 gives decoder-robust Δ > 0. c1 (near-unity, on the § 1 transition
boundary) is mixed — A/B positive but C slightly negative. The loss-weight
regime classification from § 5 carries over to the paired-fork ex/unex
readout under Dec A and Dec C in 3 of 4 rows (a1, b1, e1); only c1 has
A vs C disagreement.

R1+R2 on the same HMM C1 assay (row 1 of § 11) sits on the **sharpening**
side of this legacy axis: Δ_A = +0.309, Δ_B = +0.015, Δ_C = +0.050 (all ≥ 0).

### Checkpoint loading

Legacy checkpoints unpickle with `torch.load(..., strict=False)` using a small
`MechanismType` enum shim in `src/config.py` (DAMPENING / SHARPENING /
CENTER_SURROUND / ADAPTATION_ONLY). The shim exists only to allow
unpickling; it is not used by any current code path (per its docstring) and
does not re-introduce any legacy runtime behaviour.

### Reproducibility

- Loader shim: `src/config.py` (MechanismType enum, 17 lines, docstring marks it inert)
- Evaluator: same cross-decoder protocol as § 11 (`scripts/cross_decoder_eval.py`)
- Result JSONs: `/tmp/task26_legacy/{a1,b1,c1,e1}_C1.json`

---

## 14. Robust findings under decoder agreement (Tasks #15–#26 synthesis, 2026-04-22)

The cross-decoder comprehensive matrix (§ 11) establishes which ex-vs-unex
findings on the `dampening-analysis` branch are **decoder-robust** (all three
decoders agree on the sign of Δ) and which are decoder-dependent.

### Decoder-robust sharpening (Δ > 0 on all three decoders)

- **NEW paired-march assay on R1+R2** (row 9): Δ_A = +0.3871, Δ_B = +0.0854, Δ_C = +0.1254
- **HMM C1 (focused + HMM cue) on R1+R2** (row 1): Δ_A = +0.3090, Δ_B = +0.0150, Δ_C = +0.0500
- **HMM C1 on e1 legacy** (row 8): Δ_A = +0.2130, Δ_B = +0.0510, Δ_C = +0.0110

### Decoder-robust dampening (Δ < 0 on all three decoders)

- **HMM C1 on b1 legacy** (row 6): Δ_A = −0.0320, Δ_B = −0.0150, Δ_C = −0.0230
- **M3R on R1+R2** (row 10): Δ_A = −0.1496, Δ_B = −0.0082, Δ_C = −0.0294
- **HMS-T on R1+R2** (row 12): Δ_A = −0.2919, Δ_B = −0.1818, Δ_C = −0.0631
- **VCD-test3 on R1+R2** (row 14): Δ_A = −0.1655, Δ_B = −0.0984, Δ_C = −0.0703
- **M3R modified on R1+R2** (row 15): Δ_A = −0.1373, Δ_B = −0.0336, Δ_C = −0.0176
- **VCD-test3 modified on R1+R2** (row 17): Δ_A = −0.0840, Δ_B = −0.0264, Δ_C = −0.0130

### Decoder-dependent (≥ one decoder disagrees on sign)

- **HMM C2, C3, C4 on R1+R2** (rows 2–4): A and C both positive, B near-zero or slightly negative. The B outlier is consistent across these three conditions, not random.
- **HMM C1 on a1 legacy** (row 5): A and C both negative, B exactly 0.0.
- **HMM C1 on c1 legacy** (row 7): A and B positive, C slightly negative — c1 sits on the § 1 transition boundary.
- **HMS on R1+R2** (row 11): A and B both negative, C positive.
- **P3P on R1+R2** (row 13): A and C positive, B negative — small n (38/branch).
- **HMS-T modified on R1+R2** (row 16): A and B negative, C positive.

### R1+R2 is hybrid, not single-regime

On the **paired-fork paradigm** (rows 1–4 of § 11), R1+R2 shows decoder-robust
ex > unex — consistent with **sharpening**. On **observational HMM-trajectory
paradigms** (rows 10, 12, 14 and their modified variants on R1+R2), the same
R1+R2 checkpoint shows decoder-robust ex < unex — consistent with
**dampening**. The regime label for R1+R2 depends on both the assay paradigm
and, for several assays, the specific decoder. Previous single-paradigm
framings of R1+R2 as purely sharpening or purely dampening are not supported
by the cross-decoder matrix.

### Stable-target Dec A′ sanity check (2026-04-23)

The decoder-robust sharpening and dampening categories above hold under the
Dec A → Dec A′ swap on all 13 R1+R2 rows (Dec A′ trained on the
post-training, frozen R1+R2 `r_l23` — eliminating the "moving target during
Stage 1" concern for Dec A). Zero Δ-sign flips in Δ_A → Δ_A′, and zero
sign-agreement class changes when Δ_B / Δ_C are held fixed. See § 11 "Dec A →
Dec A′ swap summary" and `ARCHITECTURE.md` § "Stable-target decoder sanity
check" for the full diff.

See § 11 for the full 17-row matrix, § 12 for the paradigm × readout analysis
(including the FWHM-sign reversal and flank-asymmetry diagnostic), and § 13
for the legacy-ref reproduction anchoring the "sharpening" / "dampening"
labels to the 25-run sweep regime classes.
