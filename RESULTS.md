# Results

## Summary

This branch now supports **two distinct regimes**, depending on which
top-down pathway family is enabled:

1. **Template-conditioned dampening** in the original SOM-dominated
   inhibitory architecture.
2. **Cue-dependent representational sharpening** after adding a persistent,
   prediction-coupled **apical multiplicative gain** path.

The key negative result still holds for the earlier mechanism family:
**VIP-only disinhibition, additive center-support, and recurrent-gain
variants did not produce canonical sharpening on M6/M7.** The first
claim-grade sharpening result on the canonical readouts appears only after
the apical multiplicative branch is enabled and evaluated against a
corrected OFF baseline that zeros all added top-down branches.

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

## 2. Mechanism Progression Toward Sharpening

### Earlier sharpening attempts that failed on canonical metrics

The original P2–P5 line of work improved population-vector readouts (M6) but
never produced a positive trained-decoder effect (M7):

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

Those runs established an important constraint: **flank suppression alone is
not enough**. M6 can look better because far-flank noise is reduced, while M7
remains flat or negative because no new locally separable information is
created for a flexible decoder.

### Post-P5 mechanism progression on this branch

After that negative result, the branch added progressively more local and
prediction-coupled mechanisms:

| Revision | High-level mechanism | Canonical outcome |
|---|---|---|
| VIP-only | Cue-driven VIP→SOM subtraction | Cue trace works, but late M6/M7 remain flat or negative; effect looks like broad suppression |
| Center-support | Weak additive center excitation from prediction | Mechanistically active, but negligible causal gain over ablation on M6/M7 |
| Recurrent-gain | Narrow gain on the recurrent L2/3 term | Small cue-local-competitor boost, still no convincing canonical sharpening |
| **Apical multiplicative gain** | Persistent, prediction-coupled multiplicative gain on **ff + rec** | **First clear positive M6/M7 result on canonical cued readouts** |

### Why the earlier mechanisms failed

The failed mechanisms all shared the same core weakness: they mostly
reweighted suppression or added weak local support without creating a
persistent multiplicative advantage at the expected feature during the late
read window. In practice, they produced one of two outcomes:

- **Broad suppressive / dampening-like profiles** that lowered energy and peak
  gain but did not improve trained decoding.
- **Tiny local-support effects** that were visible in cue-local-competitor
  summaries but too weak to move canonical M6/M7.

The apical branch succeeds because it is both:

1. **Prediction-coupled**: built from `q_pred` and bounded by `pi_pred_eff`
2. **Persistent into the probe window**: gated by the carried cue trace via
   VIP state rather than the instantaneous raw cue tensor

That combination is the first one in this repo that changes the late
population code in a way a trained linear decoder can exploit.

---

## 3. Apical Multiplicative Gain: Positive Sharpening Result

### Mechanism

The winning revision keeps the existing inhibitory SOM pathway and cue-driven
VIP→SOM subtraction, but adds a new **persistent apical multiplicative gain**
path:

- narrow predicted-feature profile from `q_pred`
- bounded by `pi_pred_eff`
- persisted through a slow `a_apical` state
- cue-gated via the carried VIP trace
- applied multiplicatively to the combined excitatory drive `ff + rec`

The additive center-support branch is turned off in the main winning screen
so the apical effect is cleanly interpretable.

### Corrected OFF baseline

The standard analysis ON/OFF comparison was tightened before calling this
result:

- OFF now zeros **all** added top-down branches, not just inhibitory SOM
  feedback.
- Specifically, OFF ablates:
  - SOM / inhibitory feedback
  - VIP→SOM gain
  - additive center-support
  - recurrent-gain
  - apical multiplicative gain
- `sanity_check_ablation(...)` verifies that every branch output is zero in
  the OFF condition before trusting the readout.

### Seed-42 main result (`beta=0.16`)

Main screen config:
- `config/option_b_apical_gain_screen_beta016.yaml`

Primary artifacts:
- `results/option_b_apical_gain_beta016/screen_on/center_surround_seed42/checkpoint.pt`
- `results/option_b_apical_gain_beta016/screen_on/analyze_seed42_uncued.log`
- `results/option_b_apical_gain_beta016/screen_on/analyze_seed42_cued.log`
- `results/option_b_apical_gain_beta016/m7_resolution_pass.json`

Canonical readouts:

| Metric | Uncued | Cued |
|---|---|---|
| Peak ratio (ON/OFF) | 0.9386 | **1.1657** |
| Population FWHM Δ | −0.18° | **−2.15°** |
| M6 Δd' (δ=5°) | −0.0383 | **+0.1749** |
| M6 Δd' (δ=10°) | −0.0500 | **+0.3272** |
| M7 ΔAcc coarse (δ=10°) | −0.0050 | **+0.0150** |

High-resolution M7 on the saved checkpoint (`n_train=n_test=4000`,
32 resample seeds):

| Metric | Mean | 95% bootstrap CI |
|---|---|---|
| M7 ΔAcc δ=5° | **+0.00743** | **[+0.00690, +0.00798]** |
| M7 ΔAcc δ=10° | **+0.01299** | **[+0.01215, +0.01386]** |

Cue-local-competitor:
- valid-neutral competitor-gap delta: **+0.000855**

### Seed-42 matched OFF ablation

Matched comparator:
- `config/option_b_apical_gain_ablation.yaml`
- `results/option_b_apical_gain/screen_off/center_surround_seed42/checkpoint.pt`

High-resolution M7, cued:

| Metric | Mean | 95% bootstrap CI |
|---|---|---|
| OFF ΔAcc δ=5° | −0.00063 | [−0.00084, −0.00044] |
| OFF ΔAcc δ=10° | −0.00117 | [−0.00134, −0.00100] |

Paired seed-42 main-minus-OFF:

| Metric | Mean | 95% bootstrap CI |
|---|---|---|
| ΔAcc difference δ=5° | **+0.00806** | **[+0.00752, +0.00863]** |
| ΔAcc difference δ=10° | **+0.01416** | **[+0.01330, +0.01505]** |

This is the causal contrast that matters most: the sharpening signal is
present in the main apical run and absent in the OFF ablation under the same
evaluation protocol.

### Seed-43 replication

Replication artifacts:
- `results/option_b_apical_gain_beta016/screen_on_seed43/center_surround_seed43/checkpoint.pt`
- `results/option_b_apical_gain_beta016/m7_resolution_pass_with_seed43.json`

High-resolution M7:

| Metric | Mean | 95% bootstrap CI |
|---|---|---|
| M7 ΔAcc δ=5° | **+0.00744** | **[+0.00691, +0.00798]** |
| M7 ΔAcc δ=10° | **+0.01298** | **[+0.01213, +0.01383]** |

### Multi-seed strengthened evidence

Combined seed-42 + seed-43 cued M7:

| Metric | Mean | 95% bootstrap CI |
|---|---|---|
| M7 ΔAcc δ=5° | **+0.00743** | **[+0.00705, +0.00782]** |
| M7 ΔAcc δ=10° | **+0.01299** | **[+0.01239, +0.01359]** |

Combined uncued control:

| Metric | Mean | 95% bootstrap CI |
|---|---|---|
| Uncued ΔAcc δ=5° | −0.00092 | [−0.00106, −0.00077] |
| Uncued ΔAcc δ=10° | −0.00170 | [−0.00185, −0.00156] |

This satisfies the branch’s claim-grade decision rule: positive cued M7 at
both deltas, mean size above threshold, uncued control remaining non-positive,
and positive paired seed-42 main-minus-OFF contrasts.

---

## 4. Other Findings

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

## 5. Representational Metrics Used

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

## 6. Configs and Results Location

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
| `results/option_b_screen/` | VIP-only / cue-first sharpening screens |
| `results/option_b_center_support/` | Additive center-support screens |
| `results/option_b_recurrent_gain/` | Recurrent-gain screens |
| `results/option_b_apical_gain/` | Initial apical gain screen + OFF ablation + seed-43 replication |
| `results/option_b_apical_gain_beta012/` | Tuned `beta=0.12` apical run + high-resolution M7 |
| `results/option_b_apical_gain_beta016/` | Final `beta=0.16` run + claim-grade M7 resolution pass |

---

## 7. What This Means

The codebase now supports a narrower, more specific claim than the earlier
SOM-only version:

> In the original SOM-dominated feedback architecture, dampening is the
> default robust regime and earlier local-sparing variants are not sufficient
> for canonical sharpening. However, once a persistent prediction-coupled
> apical multiplicative gain path is added, the model shows reproducible,
> cue-dependent representational sharpening on the canonical M6/M7 readouts,
> with a matched OFF ablation and a second seed supporting the effect.

Current caveats:

- The strongest positive result is **cue-dependent**; uncued behavior remains
  suppressive.
- The positive claim is validated with the current canonical analysis stack
  and high-resolution saved-checkpoint M7 evaluation, not with a new learned-V2
  rescue.
- The winning mechanism is no longer SOM-only; it explicitly depends on an
  added multiplicative apical branch.
