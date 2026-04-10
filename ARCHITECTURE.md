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
