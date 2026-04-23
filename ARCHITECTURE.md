# Network Architecture

## Overview

A minimal laminar V1-V2 rate model in PyTorch, trained with BPTT. The model
studies how top-down expectation modulates early visual cortex activity.
One architecture produces both dampening and sharpening feedback regimes,
controlled by the balance of computational objectives (loss weights).

## V1 Circuit (4 populations)

### L4 (Input Layer)
- 36 orientation channels (5 deg spacing, 0-180 deg)
- Identity feedforward weights (frozen after Stage 1)
- Stimulus-specific adaptation: tau_adaptation=200, alpha_adaptation=0.3
- Naka-Rushton nonlinearity for contrast normalization (n=2.0, c50=0.3)
- Time constant: tau_l4=5

### PV (Parvalbumin Interneurons)
- Divisive normalization pool: averages across L4 channels
- Provides contrast invariance via subtractive inhibition of L2/3
- Time constant: tau_pv=5

### L2/3 (Superficial Pyramidal -- Main Output)
- Receives: L4 input, PV inhibition (subtractive), SOM inhibition
  (subtractive), recurrent connections (W_rec), center_exc (additive)
- Activation: rectified_softplus = ReLU(softplus(x) - ln(2))
- W_rec: constrained spectral radius <= 0.95 (sigma_rec=15 deg, gain_rec=0.3)
- Time constant: tau_l23=10

### SOM (Somatostatin Interneurons -- Top-Down Inhibition)
- Receives `som_drive` from V2 feedback (inhibitory channel)
- Inhibits L2/3 (subtractive)
- Integrated with tau_som=10 (not instantaneous)

## V2 (Higher-Order Area)

GRU with 16 hidden units, three output heads:

### Prediction Head
- `head_mu = nn.Linear(16, 36)` -> softmax -> mu_pred (orientation prior)
- `head_pi = nn.Linear(16, 1)` -> softplus + clamp -> pi_pred (precision)
- Prior supervised via KL divergence against true next orientation

### Feedback Head
- `head_feedback = nn.Linear(16, 36)` -- 576 learnable parameters
- Raw output, no activation function -- V2 learns sign and magnitude
- E/I split following Dale's law:
  - `center_exc = relu(+feedback_signal)` -> additive excitation to L2/3
  - `som_drive = relu(-feedback_signal)` -> SOM drive (inhibition to L2/3)
- Feedback scaled by `feedback_scale` (ramps 0->1 during burn-in)

### V2 Input
- Default: L2/3 previous timestep (`v2_input_mode: l23`)
- Alternatives: L4 only (`l4`), or L4+L2/3 concatenated (`l4_l23`)

## Signal Flow

```
Stimulus -> L4 -> PV (normalization)
            L4 + L2/3 -> V2 (GRU) -> mu_pred (prediction)
                                   -> feedback_signal [B, 36]
                                        |
                          feedback_signal * feedback_scale = scaled_fb
                          relu(+scaled_fb) = center_exc -> L2/3 (additive)
                          relu(-scaled_fb) = som_drive  -> SOM (tau=10) -> L2/3 (subtractive)
                                                              |
            L4 ---------> L2/3 <- PV (subtractive)
                           L2/3 <- SOM (subtractive)
                           L2/3 += center_exc (additive)
                           L2/3 <- W_rec (recurrence)
                           L2/3 -> decoder
```

## Training Protocol

### Stage 1: Sensory Scaffold (2000 steps)
- Random gratings, variable contrast
- Trains L2/3 ff weights, PV gains, W_rec
- Gating: decoder accuracy >= 90%, unimodal tuning, FWHM 15-30 deg

### Stage 2: V2 + Feedback (5000+ steps)
- HMM-generated orientation sequences
- Curriculum: burn-in (fb=0, 1000 steps) -> ramp (fb 0->1, 1000 steps) -> full
- LR: V2 3e-4, feedback 1e-4, weight decay 1e-4
- Gradient clipping: 1.0
- Oracle mode available: V2 frozen, q_pred from ground-truth state

## Stimulus Generation

- HMM with 2 states (CW, CCW): p_self=0.95, p_transition=0.80
- 12 anchor orientations (every 15 deg from 0-180 deg)
- transition_step=5.0 deg per presentation
- 25 presentations per sequence, steps_on=12, steps_isi=4
- Gaussian noise on population code (stimulus_noise=0.25)
- Optional ambiguous stimuli (ambiguous_fraction=0.3, offset=15 deg)

## Loss Functions

| Loss | Weight | Target |
|---|---|---|
| `lambda_sensory` | **regime-dependent** | 36-way CE orientation decode on L2/3 |
| `lambda_state` | 1.0 | KL(target \|\| mu_pred) -- V2 prior quality |
| `lambda_energy` | 0.01 (exp: 2.0) | L1 on population rates; L2/3 term multiplied by `l23_energy_weight` |
| `lambda_homeo` | 1.0 | Homeostasis (L2/3 mean in [0.05, 0.5]) |
| `lambda_mismatch` | 0.0 | Binary BCE expected/deviant classification |
| `lambda_pred_suppress` | 0.0 | Penalize L2/3 activity at predicted orientation |
| `lambda_fb_energy` | 0.0 | Penalize excitatory feedback magnitude |

## Key Config Parameters

| Config | Default | Role |
|---|---|---|
| `lambda_sensory` | 0.3 | **Dominant**: controls feedback regime (0=dampen, 0.3=sharpen) |
| `l23_energy_weight` | 1.0 | **Critical**: amplitude control (3.0=near-neutral, 5.0=sub-unity) |
| `lambda_energy` | 0.01 (exp: 2.0) | **Secondary**: broad energy constraint |
| `feedback_mode` | emergent | V2 prediction mode |
| `freeze_v2` | false | Oracle mode (ground-truth predictions) |
| `oracle_pi` | 1.0 | Pi value in oracle mode |
| `oracle_template` | oracle_true | Template mode in oracle |
| `stimulus_noise` | 0.25 | Gaussian noise std on population code |
| `transition_step` | 5.0 | CW/CCW orientation step (degrees) |

## Code Structure

| File | Contents |
|---|---|
| `src/model/populations.py` | V1L4Ring, PVPool, V1L23Ring, SOMRing |
| `src/model/v2_context.py` | V2ContextModule (GRU + heads) |
| `src/model/network.py` | LaminarV1V2Network (composes all modules) |
| `src/training/losses.py` | CompositeLoss with all loss functions |
| `src/training/stage1_sensory.py` | Stage 1 training loop |
| `src/training/stage2_feedback.py` | Stage 2 training loop + oracle mode |
| `src/training/trainer.py` | build_stimulus_sequence, optimizer creation |
| `src/stimulus/sequences.py` | HMMSequenceGenerator |
| `src/config.py` | ModelConfig, TrainingConfig, StimulusConfig |
| `scripts/train.py` | CLI entry point |
| `scripts/analyze_representation.py` | Post-hoc representation analysis |

## Tests

245 tests covering V1 circuit, V2 learned prior, feedback pathway,
training pipeline, config parsing, experimental paradigms, analysis suite,
and regression tests. Run with `python3 -m pytest tests/ -v`.

## Optional rescue-chain modules (failed-dual-regime-experiments branch)

The architecture above describes the **baseline** model on `main` /
`single-network-dual-regime`. On branch `failed-dual-regime-experiments`,
five optional config flags add rescue-chain mechanisms layered on top of
the baseline. Each rescue isolates a specific architectural change in an
attempt to recover task-state-selective sharpening AND dampening.

| Flag | Rescue | Mechanism |
|---|---|---|
| `use_precision_gating` | R2 | Scales feedback by `pi_pred / pi_max` so V2-uncertain feedback is attenuated. |
| `use_vip` | R3 | Adds VIPRing population + structured center-surround SOM disinhibition kernel. |
| `use_deep_template` | R4 | Adds DeepTemplate leaky-integrator population maintaining a V1-side expectation template. |
| `use_error_mismatch` | R4 | Routes the mismatch head from `r_error = relu(r_l23 - r_template)` instead of raw `r_l23`. |
| `use_shape_matched_prediction` | R5 | Projects `q_pred` through a fixed Stage-1-calibrated buffer `T_stage1` before subtractive suppression. |

R1 is `lambda_expected_suppress > 0` (loss-only; no architectural flag). All
five flags default to `false` so the baseline architecture is unchanged when
they are absent. See `docs/rescues_1_to_4_summary.md` for the full
per-rescue rationale, results, and the 2026-04-13 re-centered analysis
correcting the earlier "subtractive predictive coding" interpretation.

## Decoders (orientation readout from L2/3)

Three orientation decoders exist in this project. Only Decoder A is part of
the trained network; B and C are post-hoc analysis tools. All three are
applied to the same per-trial `r_l23` (single forward pass, three readouts)
whenever cross-decoder evaluation is performed.

### Full taxonomy (updated 2026-04-22 from Tasks #15–#26)

| Decoder | Type | Training data | Samples | Sees unexpected in training? | 10k natural-HMM top-1 (R1+R2) | Stratified strengths |
|---|---|---|---|---|---|---|
| **A** | `Linear(36, 36)` saved with each Stage-1 checkpoint, frozen in Stage 2. | Natural HMM-march `r_l23` activations during Stage-1 training. | All Stage-1 training trials. | **Yes** — natural march already includes jumps/unexpected transitions at the task-state switching rate. | **0.5413** | Best on `jump` stratum (top-1 0.742 vs decC 0.424); best overall `within3` on non-ambiguous trials; weaker than decC on `pi_low_Q1` (0.464 vs 0.502). |
| **A′** | Standalone `Linear(36, 36)` with bias, trained **only after the network was fully trained and frozen** (stable-target retrain of Dec A). | Natural-HMM `r_l23` streamed through the frozen R1+R2 network (`results/simple_dual/emergent_seed42/checkpoint.pt`). Per-step: batch 32 × seq 25 = 800 readouts at `t∈[9,11]`, ambiguous kept in, 50/50 focused/routine task_state. Adam lr=1e-3, 5000 gradient steps, seed 42; val pool seed 1234 (~8k readouts). Saved at `checkpoints/decoder_a_prime.pt`. Training script: `scripts/train_decoder_a_prime.py`. | 5000 gradient steps × 800 readouts = 4 M readouts total. | **Yes** — same natural HMM stream, but on **stable, post-training L2/3** (net.eval(), all params `requires_grad_(False)` verified at setup). | **0.5486** | `frac_same_pred(A, A′) = 0.8200`; mean circular distance A vs A′ = 0.25 ch. Slight edges on `clean / pi_high_Q4 / high_pred_err_gt20deg` (+1.6 / +2.6 / +2.5 pp vs A); slight deficits on `ambiguous / pi_low_Q1 / low_pred_err_le5deg` (−1.4 / −2.9 / −3.9 pp vs A). |
| **B** | 5-fold nearest-centroid CV, computed on demand per analysis. | The same `r_l23` activations being analysed (no separate training). | Varies per-assay (set under evaluation). | Inherits exposure from the assay itself (sees whatever trials the analysis contains). | — (not applicable — CV over analysis set) | Robustness control; flips sign in 5 of 17 cross-decoder rows, so not a safe stand-alone reference. |
| **C** | Standalone `Linear(36, 36)` with bias. | 100k synthetic orientation-bump patterns (50k single-orientation σ=3 ch, amplitudes ∈ [0.1, 2.0]; 50k multi-orientation K∈{2,3} with strictly-max amplitude as the label; Gaussian noise σ=0.02). Trained Adam lr=1e-3, batch 256, ≤30 epochs, early-stop patience 3, seed 42. Saved at `checkpoints/decoder_c.pt`. | 100k synthetic. | **No** — trained on clean synthetic bumps only; never sees network `r_l23` or HMM-march context. | **0.5345** | Best on `ambiguous within1` (0.725 vs decA 0.703); best on `pi_low_Q1` (0.502 vs decA 0.464); weaker on `jump` (0.424 vs decA 0.742); best `top1` on `march_smooth` (0.552 vs decA 0.510). |

Agreement between Dec A and Dec C on the 10k natural HMM stream:
`frac_same_pred = 0.6691`, mean circular distance 0.43 channels (≈ 2.1°).
Agreement between Dec A and Dec A′ on the same stream:
`frac_same_pred(A, A′) = 0.8200`, mean circular distance 0.25 channels
(`frac_same_pred(A′, C) = 0.6367`). **Chance baseline = 1/36 ≈ 0.028** for
36-way orientation classification; all three decoders (top-1 0.5413 / 0.5486 /
0.5345) operate ≈ 19× above chance — they are NOT at chance, they are
well-trained classifiers whose predictions agree on 67–82% of trials. Sources:
`/tmp/task25_dec_av_c_summary.json` (Task #25; 10 000 trials, seed 42,
readout window `t∈[9,11]`) and `results/decoder_a_prime_stratified_eval.json`
(Task #1 Dec A′; same 10 000 trials, same design, adds Dec A′).

### Stable-target decoder sanity check (Dec A′ vs Dec A, 2026-04-23)

Dec A is trained jointly with L2/3 during Stage 1 (cf.
`src/training/stage1_sensory.py:127-163`) — it fits a moving target.
Dec A′ was retrained for 5000 Adam steps on `r_l23` streamed through the
**fully-trained, frozen** R1+R2 network (net.eval(), `requires_grad_(False)`
on every network param, verified by assertion). Applying Dec A′ in place of
Dec A on the 13 R1+R2 rows of the Task #26 17-row matrix (legacy
a1/b1/c1/e1 rows retained their own stored Dec A):

- **Zero Δ-sign flips** across all 13 R1+R2 rows (Dec A′ agrees with Dec A
  on the sign of Δ = acc_ex − acc_unex everywhere).
- `|Δ_A′ − Δ_A|` ≤ 0.094; median 0.025; mean 0.032. Largest shifts
  concentrate on sharpening-side rows (HMS-T native +0.081, HMS-T modified
  +0.094, M3R native +0.061) where Dec A′ consistently produces a
  less-extreme negative Δ than Dec A.
- Two sign-agreement class changes: HMM C3 tightens to unanimous `+`
  (Dec B had been the outlier); M3R native loosens to majority `−` with B
  becoming the outlier (|Δ_B| moves from −0.008 to +0.003). No outlier
  identity changes among rows that were already split.
- Dec A′ per-row-magnitude profile on the 13 R1+R2 rows: `mean |Δ| = 0.2138,
  max |Δ| = 0.3902` (Dec A same rows: `mean |Δ| = 0.2298, max = 0.3871`).

**Interpretation.** The "moving target during Stage 1" concern for Dec A
does not materially change the 13-row dampening-vs-sharpening sign pattern.
Dec A's extra training exposure to early-training (pre-feedback) L2/3
distributions appears to amplify sharpening-side magnitudes slightly, not
flip signs. Full matrix rerun: `results/cross_decoder_comprehensive_decAprime.{json,md}`;
row-by-row diff: `results/cross_decoder_comprehensive_decAprime_diff.{json,md}`.

### Cross-decoder bias flags (from Task #26, 17-row matrix)

Measured on `results/cross_decoder_comprehensive.json` (17 ex/unex
comparisons spanning HMM C1–C4, legacy a1/b1/c1/e1, paired-fork, and four
observational assays M3R / HMS / HMS-T / P3P / VCD on R1+R2):

- **Dec A:** `mean |Δ| = 0.2024`, `max |Δ| = 0.3871`; never the single
  sign-outlier. Always aligns with the row majority — but largest Δ of the
  three decoders in every row, which means A consistently amplifies
  whatever effect is present.
- **Dec B:** `mean |Δ| = 0.0598`, `max |Δ| = 0.1818`; outlier in **5 of 17
  rows** (HMM C2 / C3 / C4 on R1+R2, a1 legacy, P3P on R1+R2). B is the
  noisiest sign-carrier.
- **Dec C:** `mean |Δ| = 0.0399`, `max |Δ| = 0.1254`; outlier in **3 of 17
  rows** (c1 legacy, HMS on R1+R2, HMS-T with focused+march cue on R1+R2).
  Smallest-magnitude Δ in most rows, consistent with "untrained-on-network,
  no march exposure."

All three decoders agree on sign in 9 of 17 rows. See RESULTS.md §11 for
the full matrix.

### Decoder magnitude amplification (originally framed as "Decoder A artefact" 2026-04-17; re-evaluated 2026-04-22 under Task #26)

**Original 2026-04-17 framing.** On R1+R2, the matched-probe-3pass
Δdec(ex−unex)=+0.32 measured under Decoder A collapsed to Δ≈+0.04 under
Decoder B (5-fold nearest-centroid CV). The original interpretation was
that Decoder A's fixed templates, trained on the natural-march
distribution, were out-of-distribution for the synthetic Pass B compound
bumps used in matched-probe-3pass — and that A's number was therefore
artefactual.

**Task #26 update (2026-04-22).** The cross-decoder matrix
(`results/cross_decoder_comprehensive.json`, row 13: P3P on R1+R2) measured
the full three-decoder profile on the same matched-probe-3pass evaluation:
**Δ_A = +0.3684, Δ_B = −0.1714, Δ_C = +0.0526** (n_ex = n_unex = 38).
Decoders A and C agree on sign (positive); Decoder B is the single
sign-outlier. Across the full 17-row matrix, Dec A is the largest-magnitude
Δ in every row (mean |Δ| = 0.20, vs B mean = 0.06, C mean = 0.04) but is
**never the sign-outlier** — A consistently amplifies whatever effect is
present, rather than producing artefactual sign-flips.

**Current operational rule.** Treat Dec A as the magnitude-amplifying
readout (largest |Δ|) and Dec C as the conservative sign-only check
(smallest |Δ|). Dec B has the highest sign-outlier rate (5 of 17 rows) and
should not be used as a stand-alone reference. Cross-decoder sign
agreement (the all-agree rows; 9 of 17 in the current matrix) is the
operational definition of a "decoder-robust" finding on this branch.

### Decoder C accuracy summary

- Held-out synthetic test: 0.81 (single-orientation 0.98 / multi-orientation
  0.65).
- R1+R2 natural-HMM (10k, Task #25): **0.5345 top-1 / 0.896 within1 /
  0.956 within2**; 0.66 top-1 on the non-ambiguous slice (7040 trials).
