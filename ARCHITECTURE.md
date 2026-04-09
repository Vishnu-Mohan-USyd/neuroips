# Network Architecture

## Overview

A minimal laminar V1-V2 rate model in PyTorch, trained with BPTT. The model
studies how top-down expectation modulates early visual cortex activity, and
what feedback regimes emerge under different computational objectives.

## V1 Circuit (5 populations)

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

### VIP (Vasoactive Intestinal Peptide Interneurons — Disinhibition)
- Receives `vip_drive` from the feedback operator (separate alpha_vip weights)
- Inhibits SOM: `effective_som_drive = relu(som_drive - softplus(w_vip_som) * r_vip)`
- Disinhibits L2/3 at the predicted orientation by reducing SOM suppression there
- Time constant: tau_vip=10 (same as SOM)
- Activation: rectified_softplus (same as SOM)
- Biologically motivated: VIP→SOM connection probability 62.5% (Pfeffer et al. 2013)

### Apical Gain (Multiplicative Top-Down Modulation)
- Receives `apical_gain` from the feedback operator (separate alpha_apical [7] weights)
- Multiplies ONLY excitatory L2/3 drive: `excitatory_drive = apical_gain * (ff + rec + template)`
- Does NOT affect inhibitory terms (SOM, PV)
- Constrained: `apical_gain = 1.0 + 0.7 * tanh(pi_eff * apical_field)`
- Range: [0.3, 1.7] (±70% modulation maximum)
- **Recommended mode: pure top-down** (r_l4=None fallback). The coincidence
  gate exists in code but was found to hurt performance — it compresses the
  gain signal without providing content selectivity (see RESULTS.md Section 8).
- **Coincidence gate** (code retained, not recommended): `coincidence = relu(apical_field) * relu(basal_field)`,
  where `apical_field = K_apical ⊛ q_centered` (top-down) and
  `basal_field = r_l4 - mean(r_l4)` (bottom-up centered L4). Tested extensively
  in Batch 3 — true ≈ wrong at all deltas, no content selectivity.
- Biologically motivated: active apical dendrites in L2/3 pyramidal cells receive
  top-down feedback in layer 1, modulating the gain of feedforward drive
  (multiplicative interaction, not additive excitation)
- No separate population — computed directly from the prediction template

### Deep Template
- Slow integrator tracking recent stimulus history
- Modulated by V2 precision: deep_template = gain × q_pred × pi_eff

## V2 (Higher-Order Area)

GRU with 16 hidden units. Two output modes:

### Emergent Mode — Learned Feature Prior (Branch A)
- Outputs: `mu_pred` [B,N] (full orientation prior distribution, softmax),
  `pi_pred` [B,1] (precision, softplus + clamp to [0, pi_max])
- `q_pred = mu_pred` directly — V2 outputs the prior, not a state belief
- head_mu: `nn.Linear(v2_hidden_dim, N)` → softmax
- Enables genuine prestimulus priors: during ISI (when L4=0), V2 maintains
  the prior from GRU memory + cue input
- Prior supervised via KL divergence against true next orientation
  (circular Gaussian target, sigma=10°)
- v2_input_mode: `l4_l23` — V2 sees both L4 and L2/3 for temporal context
- Oracle mode: q_pred from ground-truth HMM state, V2 frozen (bypasses mu_pred)

### Fixed Mode (legacy, not used in current results)
- Outputs: q_pred [B,36] (softmax), pi_pred, state_logits [B,3], h_v2

## Feedback Modes

### Simple Feedback Mode (`simple_feedback: true`) — CURRENT

V2 outputs feedback directly via a linear head. No basis functions, no
tanh/precision scaling, no multiplicative gain machinery. Signal is split
into excitatory and inhibitory pathways following Dale's law.

**Architecture:**
- `head_feedback = nn.Linear(v2_hidden_dim, N)` on V2 (576 params for dim=16, N=36)
- Raw output, no activation function — V2 learns the sign and magnitude
- Signal split: `center_exc = relu(+scaled_fb)` → excitation to L2/3,
  `som_drive = relu(-scaled_fb)` → SOM inhibition (integrated by SOMRing with tau_som=10)
- `scaled_fb = feedback_signal * feedback_scale` (feedback_scale ramps 0→1 during burn-in)
- Config flag: `simple_feedback: true` enables this mode
- `max_apical_gain: 0.0` (configurable from YAML, set to 0 to disable apical)

**Key differences from EmergentFeedbackOperator:**
- 36 direct channel weights replace 7-basis function decomposition
- No tanh/gain caps — additive (excitation) and subtractive (via SOM)
- No VIP pathway (r_vip = 0), no apical gain
- SOM receives V2 feedback through actual SOMRing dynamics (tau_som=10),
  not instantaneous drive

### Emergent Feedback Operator (legacy, `simple_feedback: false`)

#### Architecture: SOM-Only
- No excitatory pathway. `center_exc = 0`. Feedback acts exclusively
  through SOM inhibition of L2/3.
- Removes the pathway gain asymmetry that confounded earlier dual-pathway
  results (direct excitatory gain ≈ 1.0 dominated indirect SOM gain < 1.0).

#### Basis Functions (7)
1. G(σ=5°) — narrow Gaussian
2. G(σ=15°) — medium Gaussian
3. G(σ=30°) — broad Gaussian
4. G(σ=60°) — very broad Gaussian
5. Mexican hat — G(10°)/sum − G(30°)/sum
6. Constant — 1/N (flat)
7. Odd/sine — sin(θ · π/90°), for tuning shift detection

#### Learnable Parameters
- `alpha_inh` [7]: one weight per basis function (SOM pathway)
- `som_baseline` [scalar]: learned operating point for delta-SOM
- `som_tonic` [scalar]: learned positive SOM floor (init -3.0 → softplus≈0.049)
- `alpha_vip` [7]: one weight per basis function (VIP pathway, init 0.01)
- `vip_baseline` [scalar]: VIP operating point (delta-style)
- `w_vip_som` [scalar, on network]: VIP→SOM coupling gain (softplus)
- `alpha_apical` [7]: one weight per basis function (apical gain pathway, init 0.01)
- `max_apical_gain` [attribute, configurable from YAML]: maximum gain modulation
- `w_template_drive` [scalar, on network]: Branch C template→L2/3 center excitation
  weight (init 0.0 = off, in optimizer feedback group). `center_exc = w_template_drive * deep_tmpl`.
  Learned to 0.0 in all A+B runs — the model prefers apical gain over direct excitation

#### Kernel Computation
- K_inh = Σ(alpha_inh_k × basis_k) — weighted sum → 36-channel profile
- K_apical = Σ(alpha_apical_k × basis_k) — apical gain kernel
- Circular convolution: inh_field = K_inh ⊛ q_centered
  where q_centered = q_pred − 1/36
- Apical gain (pure top-down, recommended): `1.0 + mag * tanh(pi_eff * apical_field)`
  where `apical_field = K_apical ⊛ q_centered`
- Returns 3-tuple: (som_drive, vip_drive, apical_gain)

#### Delta-SOM (Bias-Corrected Softplus)
- `som_drive = pi × (softplus(baseline + inh_field) − softplus(baseline))`
- Zero drive when inh_field=0; positive when >0; negative when <0
- Removes the constant softplus bias (softplus(0) = ln(2) ≈ 0.693) that
  previously created tonic SOM drive killing L2/3 at high precision

## Signal Flow Per Timestep

### Simple Feedback Mode (current)

```
Stimulus → L4 → PV (normalization)
                L4 → V2 (GRU) → head_feedback → feedback_signal [B, 36]
                                  ↓
                     feedback_signal × feedback_scale = scaled_fb
                     relu(+scaled_fb) = center_exc → L2/3 (additive excitation)
                     relu(-scaled_fb) = som_drive  → SOM (tau_som=10) → L2/3 (subtractive)
                                                        ↓
                L4 ──────→ L2/3 ← PV (subtractive)
                           L2/3 ← SOM (subtractive)
                           L2/3 += center_exc (additive)
                           L2/3 ← W_rec (recurrence)
                           L2/3 → readout decoder
```

### Emergent Feedback Mode (legacy)

```
Stimulus → L4 → PV (normalization)
                L4 → V2 → mu_pred (= q_pred), pi_pred   [learned orientation prior]
                          q_pred → K_inh ⊛ q_centered → inh_field → delta-SOM → som_drive
                          q_pred → K_vip ⊛ q_centered → vip_field → vip_drive
                          q_pred → K_apical ⊛ q_centered → apical_field → apical_gain
                                                                        ↓
                                                    VIP (tau=10, rectified_softplus)
                                                        ↓
                                   effective_som = relu(som_drive - softplus(w_vip_som) × r_vip)
                                                        ↓
                L4 ──────→ L2/3 ← PV (subtractive)
                           L2/3 ← SOM (subtractive) ← effective_som
                           L2/3 ← W_rec (recurrence)
                           L2/3 ×= apical_gain (multiplicative, excitatory drive only)
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

## Loss Functions

| Loss | Component | Target |
|---|---|---|
| `lambda_sensory` | 36-way CE orientation decode | L2/3 readout |
| `lambda_l4_sensory` | 36-way CE orientation decode | L4 readout |
| `lambda_mismatch` | Binary BCE mismatch detection (ground-truth) | L2/3 readout |
| `lambda_local_disc` | 7-way CE local competitor discrimination | L2/3 readout (±3 channels = ±15°) |
| `lambda_sharp` | Distance-weighted activity penalty | L2/3 (penalize flanks) |
| `lambda_state` | KL(target ‖ mu_pred) — prior vs true next orientation | V2 output (learned prior) |
| `lambda_energy` | L1 on all population rates (L2 on r_l23 when `l2_energy: true`); L2/3 term multiplied by `l23_energy_weight` | All |
| `lambda_homeo` | Homeostasis (L2/3 mean in [0.05, 0.5]) | L2/3 |
| `lambda_fb` | L1 sparsity on alpha_inh + alpha_vip + alpha_apical | Feedback operator |
| `lambda_pred_suppress` | Penalize L2/3 activity matching V2 prediction: dot(r_l23, q_pred) | L2/3 vs V2 |
| `lambda_fb_energy` | Penalize excitatory feedback magnitude: L1 on center_exc | Feedback signal |
| `l2_energy` | When true, use quadratic penalty on r_l23 in energy cost | L2/3 |

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
| `simple_feedback` | false | V2 direct feedback mode (bypasses SOM/VIP/apical machinery) |
| `max_apical_gain` | 0.7 | Maximum ±% apical gain modulation (configurable from YAML) |
| `l23_energy_weight` | 1.0 | Multiplier on L2/3 term in energy cost (>1 penalizes L2/3 output more) |
| `lambda_mismatch` | 0.0 | Binary BCE expected/deviant classification from L2/3 |

## Code Structure

| File | Contents |
|---|---|
| `src/model/populations.py` | V1L4Ring, PVPool, V1L23Ring, SOMRing, VIPRing, DeepTemplate |
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

374+ tests covering V1 circuit, V2 learned prior (mu_pred distribution properties),
feedback operator (including VIP and apical gain pathways), simple feedback mode,
training pipeline, config parsing, and regression tests for all bugs discovered
during the project. Run with `python -m pytest tests/ -v`.
