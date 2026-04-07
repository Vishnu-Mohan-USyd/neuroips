# Network Architecture

## Overview

A minimal laminar V1-V2 rate model in PyTorch, trained with BPTT. The model
studies how top-down expectation modulates early visual cortex activity, and
what feedback regimes emerge under different computational objectives.
On this branch, the emergent-feedback stack has grown beyond the original
SOM-only pathway: cue-driven VIP disinhibition, additive center-support,
recurrent-gain, and a persistent apical multiplicative gain branch have all
been implemented and evaluated. The final sharpening result on this branch
comes from the apical multiplicative path, not from the earlier local-sparing
mechanisms.

## V1 Circuit (6 populations + slow traces)

### L4 (Input Layer)
- 36 orientation channels (5° spacing, 0–180°)
- Identity feedforward weights (frozen after Stage 1)
- Stimulus-specific adaptation: tau_adaptation=200, alpha_adaptation=0.3
- Naka-Rushton nonlinearity for contrast normalization (n=2.0, c50=0.3)
- Time constant: tau_l4=5

### PV (Parvalbumin Interneurons)
- Divisive normalization pool: averages across L4 channels
- Provides contrast invariance via subtractive inhibition of L2/3
- Time constant: tau_pv=5

### L2/3 (Superficial Pyramidal — Main Output)
- Receives: L4 input, PV inhibition (subtractive), SOM inhibition
  (subtractive), recurrent connections (W_rec)
- Activation: rectified_softplus = ReLU(softplus(x) − ln(2))
- W_rec: constrained spectral radius ≤ 0.95 (sigma_rec=15°, gain_rec=0.3)
- Time constant: tau_l23=10

### SOM (Somatostatin Interneurons — Top-Down Inhibition)
- Receives `som_drive` from the feedback operator
- Inhibits L2/3 (subtractive)
- Time constant: tau_som=10

### VIP (Vasoactive Intestinal Peptide — Cue Trace)
- Optional cue-driven ring population
- Reads the existing orientation-channel cue tensor
- Subtracts from SOM drive via VIP→SOM disinhibition
- Provides the persistent cue trace used by later top-down branches
- Time constant: `tau_vip`

### Deep Template
- Slow integrator tracking recent stimulus history
- Modulated by V2 precision: deep_template = gain × q_pred × pi_eff

### Apical Trace
- Slow state `a_apical` carrying the prediction-coupled apical gain
- Built from predicted orientation profile, bounded precision, and persistent
  cue gating
- Used only when the apical multiplicative branch is enabled

## V2 (Higher-Order Area)

GRU with 16 hidden units. Two output modes:

### Emergent Mode (used in all current experiments)
- Outputs: `p_cw` [B,1] (P(clockwise rule), sigmoid), `pi_pred` [B,1]
  (precision, softplus + clamp to [0, pi_max])
- `q_pred` constructed analytically: given current L4 orientation + p_cw,
  compute expected CW (+step°) and CCW (−step°) bumps, mix by p_cw
- Oracle mode: q_pred from ground-truth HMM state, V2 frozen

### Fixed Mode (legacy, not used in current results)
- Outputs: q_pred [B,36] (softmax), pi_pred, state_logits [B,3], h_v2

## Feedback Operator (EmergentFeedbackOperator)

### Core inhibitory pathway
- Base emergent pathway remains SOM-centered.
- `alpha_inh` parameterizes the inhibitory kernel basis.
- Circular convolution of that kernel with `q_pred - uniform` yields
  prediction-shaped SOM drive.
- `delta_som` removes the tonic softplus bias.

### Optional added branches (all default-off unless config enables them)
- **VIP→SOM subtraction**: cue-driven disinhibition that reduces SOM drive
  locally.
- **Center-support**: weak additive center excitation derived from the
  predicted feature profile.
- **Recurrent-gain**: narrow multiplicative gain on the L2/3 recurrent term.
- **Apical multiplicative gain**: persistent, prediction-coupled gain applied
  to the combined excitatory drive `ff + rec`.

These branches were evaluated progressively. On this branch:
- VIP-only, center-support, and recurrent-gain were mechanistically active but
  insufficient on canonical M6/M7.
- The first robust sharpening result came from the apical multiplicative path.

### Basis Functions (7)
1. G(σ=5°) — narrow Gaussian
2. G(σ=15°) — medium Gaussian
3. G(σ=30°) — broad Gaussian
4. G(σ=60°) — very broad Gaussian
5. Mexican hat — G(10°)/sum − G(30°)/sum
6. Constant — 1/N (flat)
7. Odd/sine — sin(θ · π/90°), for tuning shift detection

### Learnable Parameters
- `alpha_inh` [7]: one weight per basis function
- `som_baseline` [scalar]: learned operating point for delta-SOM
- optional scalar gains / widths for center-support, recurrent-gain, and
  apical-gain branches (all config-gated)

### Kernel Computation
- K_inh = Σ(alpha_inh_k × basis_k) — weighted sum → 36-channel profile
- Circular convolution: inh_field = K_inh ⊛ q_centered
  where q_centered = q_pred − 1/36

### Delta-SOM (Bias-Corrected Softplus)
- `som_drive = pi × (softplus(baseline + inh_field) − softplus(baseline))`
- Zero drive when inh_field=0; positive when >0; negative when <0
- Removes the constant softplus bias (softplus(0) = ln(2) ≈ 0.693) that
  previously created tonic SOM drive killing L2/3 at high precision

### Apical multiplicative gain
- Built from a narrow predicted-feature profile plus bounded precision
- Persisted through `a_apical`
- Cue-gated by the carried VIP trace rather than the instantaneous raw cue
- Applied multiplicatively to the excitatory term entering L2/3:
  `exc_eff = (ff + rec) * (1 + apical_gain)`
- This is the branch responsible for the final cue-dependent sharpening result
  on the canonical readouts

## Signal Flow Per Timestep

```
Stimulus → L4 → PV (normalization)
                L4 → V2 → p_cw, pi_pred
                          p_cw + L4 → q_pred (analytical)
                          q_pred → K_inh ⊛ q_centered → inh_field
                          inh_field → delta-SOM → som_drive
                          cue → VIP ───────────────┘
                          q_pred + pi + VIP → a_apical / apical_gain
                                                    ↓
                L4 ──────→ L2/3 ← PV (subtractive)
                           L2/3 ← SOM (subtractive) ← som_drive
                           L2/3 ← W_rec (recurrence)
                           (ff + rec) × (1 + apical_gain)
                           L2/3 → readout decoder
```

## Training Protocol

### Stage 1: Sensory Scaffold (2000 steps)
- Random gratings, variable contrast
- Trains L2/3 ff weights, PV gains, W_rec
- Gating: decoder accuracy ≥87%, unimodal tuning, FWHM 15–30°
- Trained decoder transferred to Stage 2

### Stage 2: V2 + Feedback (5000 steps)
- HMM-generated orientation sequences
- Curriculum: burn-in (fb=0, 500–1000 steps) → ramp (fb 0→1, 500–1000
  steps) → full feedback
- Oracle mode: V2 frozen, q_pred from ground-truth state
- Orientation decoder frozen during mechanistic runs (freeze_decoder=True
  when freeze_v2=True)

## Stimulus Generation

- HMM with 2 states (CW, CCW): p_self=0.95
- 12 anchor orientations (every 15° from 0–180°)
- CW: next = current + transition_step°; CCW: next = current − step°
- Within-state transition fidelity: p_transition=0.80
- 25 presentations per sequence, each steps_on + steps_isi timesteps
- Gaussian noise on population code (stimulus_noise configurable)
- Optional ambiguous stimuli: mixture of anchor + (anchor + ambiguous_offset°)
- Optional cue-first regime: orientation-channel cues placed in prestimulus
  timesteps using the existing cue tensor convention

## Loss Functions

| Loss | Component | Target |
|---|---|---|
| `lambda_sensory` | 36-way CE orientation decode | L2/3 readout |
| `lambda_l4_sensory` | 36-way CE orientation decode | L4 readout |
| `lambda_mismatch` | Binary BCE mismatch detection (ground-truth) | L2/3 readout |
| `lambda_local_disc` | 5-way CE local competitor discrimination | L2/3 readout (±2 channels) |
| `lambda_sharp` | Distance-weighted activity penalty | L2/3 (penalize flanks) |
| `lambda_state` | BCE on p_cw vs true CW/CCW | V2 output |
| `lambda_energy` | L1 on all population rates | All |
| `lambda_homeo` | Homeostasis (L2/3 mean in [0.05, 0.5]) | L2/3 |
| `lambda_fb` | L1 sparsity on alpha_inh | Feedback operator |

## Key Config Options

| Config | Default | Description |
|---|---|---|
| `feedback_mode` | emergent | 'emergent' or 'fixed' |
| `delta_som` | false | Bias-corrected softplus in feedback |
| `freeze_v2` | false | Use oracle V2 (ground-truth predictions) |
| `freeze_decoder` | false | Freeze orientation decoder in Stage 2 |
| `oracle_pi` | 1.0 | Pi value in oracle mode |
| `oracle_sigma` | 12.0 | Width of oracle prediction bump (degrees) |
| `oracle_template` | oracle_true | Template mode: true/wrong/random/uniform |
| `oracle_shift_timing` | false | Shift oracle by +1 presentation (prior about current) |
| `transition_step` | 15.0 | CW/CCW orientation step (degrees) |
| `stimulus_noise` | 0.0 | Gaussian noise std on population code |
| `ambiguous_fraction` | 0.0 | Fraction of presentations that are mixtures |
| `ambiguous_offset` | 15.0 | Angular offset of competitor in mixtures |
| `vip_enabled` | false | Enable cue-driven VIP ring |
| `vip_gain` | 0.0 | VIP→SOM subtractive gain |
| `cue_mode` | none | Training-time cue generation (`none`/`current`) |
| `cue_prestimulus_steps` | 0 | Prestimulus cue duration |
| `ambiguous_mode` | one_sided | `one_sided` or `symmetric_local_competitor` |
| `emergent_center_support_enabled` | false | Additive center-support branch |
| `emergent_recurrent_gain_enabled` | false | Multiplicative recurrent-gain branch |
| `apical_gain_enabled` | false | Persistent apical multiplicative branch |
| `apical_gain_beta` | 0.0 | Strength of apical multiplicative gain |
| `apical_gain_sigma` | 5.0 | Width of predicted-feature apical profile |

## Code Structure

| File | Contents |
|---|---|
| `src/model/populations.py` | V1L4Ring, PVPool, V1L23Ring, SOMRing, DeepTemplate |
| `src/model/v2_context.py` | V2ContextModule (GRU + heads) |
| `src/model/feedback.py` | EmergentFeedbackOperator + legacy FeedbackMechanism |
| `src/model/network.py` | LaminarV1V2Network (composes all modules) |
| `src/training/losses.py` | CompositeLoss with all loss functions |
| `src/training/stage1_sensory.py` | Stage 1 training loop |
| `src/training/stage2_feedback.py` | Stage 2 training loop + oracle mode |
| `src/training/trainer.py` | build_stimulus_sequence, optimizer creation |
| `src/stimulus/sequences.py` | HMMSequenceGenerator |
| `src/config.py` | ModelConfig, TrainingConfig, StimulusConfig |
| `scripts/train.py` | CLI entry point |
| `scripts/analyze_representation.py` | Post-hoc representation analysis (9 metrics) |

## Tests

408 tests covering V1 circuit, V2 factorization, feedback operator, training
pipeline, config parsing, and regression tests for all bugs discovered during
the project. Run with `python -m pytest tests/ -v`.

## Analysis OFF Baseline

The standard analysis ON/OFF comparison now uses a central OFF ablation that
zeros every added top-down branch, not just inhibitory SOM feedback:

- inhibitory feedback kernel / SOM drive
- VIP→SOM gain
- center-support
- recurrent-gain
- apical multiplicative gain

`sanity_check_ablation(...)` verifies that those branch outputs are zero before
the OFF condition is trusted.

## Hardening / Evaluation Surfaces

The branch now has three distinct evaluation layers:

- [scripts/analyze_representation.py](/home/vysoforlife/code_files/fs_codex/neuroips/scripts/analyze_representation.py):
  canonical ON/OFF M6/M7 analysis, now with optional anchor-averaged M7
- [scripts/run_m7_resolution.py](/home/vysoforlife/code_files/fs_codex/neuroips/scripts/run_m7_resolution.py):
  repeated-seed, paired ON/OFF, bootstrap M7 hardening
- [scripts/eval_cue_local_competitor.py](/home/vysoforlife/code_files/fs_codex/neuroips/scripts/eval_cue_local_competitor.py):
  cue-validity summaries from saved experiment artifacts or checkpoints

### Publication-hardened result surfaces

Key saved artifacts for the final apical family are:

- existing-checkpoint reproduction:
  - `results/hardening_apical_eval/m7_full_gpu_attached/beta016_existing_checkpoints.json`
- new full-pipeline seeds:
  - `results/hardening_apical_eval/full_seed44/`
  - `results/hardening_apical_eval/full_seed45/`
- final four-seed full-resolution aggregate:
  - `results/hardening_apical_eval/m7_full_four_seed/four_seed_full.json`

### Experiment CUDA-path fix

`ParadigmBase._run_trial_set()` now moves each trial batch to the model device
before `pack_inputs(...)` and `self.net.forward(...)`, then moves stored
outputs back to CPU for the `ExperimentResult`. This fixed checkpoint-backed
CUDA execution for `cue_local_competitor`, which previously failed when trial
tensors stayed on CPU while the network weights were on CUDA.

### Current claim boundary

The architecture and evaluation stack now support a strong four-seed ON-family
claim for the apical multiplicative branch, but the **matched trained OFF
ablation remains only seed42**. That caveat is scientific rather than
architectural: the code can support more OFF seeds, but they have not yet been
run on this branch.

## Post-Apical Strengthening Extensions

After the four-seed apical result was hardened, the branch added a second
layer of exploratory mechanism work aimed at increasing **direct local flank
suppression** while preserving center gain. These additions are now present in
the code/config surface, but they do **not** replace the validated apical
claim.

Implemented exploratory families now include:

- flank-only SOM supplement
- signed recurrent center-surround modulation
- combined ranked recurrent + flank variants
- reduced-apical sweeps
- direct shunt / center-recruited normalization probes
- cue-conditioned normalization-pool evaluation probes
- SOM-regime gate on the learned SOM field

### New architectural/code surfaces

The latest exploratory pass added:

- new config knobs in [src/config.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/config.py)
  for signed recurrent gain, flank SOM, flank shunt, SOM-regime gating, and
  local ranking loss / sampled local competitors
- new feedback-operator branches in [src/model/feedback.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/model/feedback.py)
  for signed center-surround recurrent modulation, flank-only SOM supplements,
  flank-only divisive shunts, and cue-conditioned SOM regime scaling
- a scalar SOM-regime head in [src/model/v2_context.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/model/v2_context.py)
- corresponding integration surfaces in
  [src/model/network.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/model/network.py),
  [src/model/populations.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/model/populations.py),
  [src/training/losses.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/training/losses.py),
  [src/training/trainer.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/training/trainer.py),
  and [src/training/stage2_feedback.py](/home/vysoforlife/code_files/fs_codex/neuroips/src/training/stage2_feedback.py)
- new triage / evaluation entry points:
  [scripts/run_m7_resolution.py](/home/vysoforlife/code_files/fs_codex/neuroips/scripts/run_m7_resolution.py)
  and
  [scripts/eval_cue_local_competitor.py](/home/vysoforlife/code_files/fs_codex/neuroips/scripts/eval_cue_local_competitor.py)

### Diagnostic takeaway

The central diagnostic result from the strengthening pass was that the uncued
baseline suppression in the best pre-SOM-regime checkpoint was traced to the
always-on learned SOM pathway, specifically the `alpha_inh` + `som_baseline`
route, rather than to:

- apical gain
- recurrent modulation
- flank shunt
- PV normalization
- template-driven branches

This diagnosis is what justified the SOM-regime family.

### Eval-only motivation for SOM-regime gating

Before training a new branch, the codebase was probed with evaluation-only
SOM scaling. A representative passing row was:

- `uncued_scale=0.25`
- `cued_scale=1.10`
- cued `+0=1.1098`
- cued `+20/+25/+30=0.9672/0.9651/0.9676`
- uncued `+20/+25/+30=0.9801/0.9796/0.9776`
- `Δ20=0.0129`, `Δ25=0.0145`, `Δ30=0.0100`

That eval-only geometry showed that a cue-conditioned SOM regime split could
plausibly produce stronger center-plus-flank separation in principle.

### Current exploratory status

The first trained SOM-regime checkpoint did not reproduce the eval-only split
strongly enough. It raised center gain and kept decoder deltas positive, but
remained center-gain dominated and failed the stricter bar for clear
cued-over-uncued flank suppression.

So the architecture currently has:

- a validated apical sharpening path
- exploratory stronger-effect branches that are implemented and test-covered
- an unresolved debugger question about why trained SOM-regime gating does not
  yet reproduce the earlier eval-only split strongly enough

### Current acceptance bar for future mechanisms

Future strengthening work on this branch only counts if it produces:

- strong center gain
- strong local flank suppression
- clear cued-over-uncued separation
- decoder improvement above the user's noise floor
- no broad-gain cheating
- correct cue-validity ordering
- matched ablation evidence
