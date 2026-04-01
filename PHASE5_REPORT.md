# Phase 5 Report: Training Pipeline

**Status:** 186/186 tests pass (51 model + 58 network + 46 stimulus + 31 training).

## 1. Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `src/model/network.py` | Modified | Extended `forward()` aux dict with r_l4_all, r_pv_all, r_som_all trajectories |
| `src/training/losses.py` | Created | CompositeLoss with 4 components: sensory readout, prediction, energy, homeostasis |
| `src/training/trainer.py` | Created | Shared utilities: freeze/unfreeze, param groups, scheduler, stimulus builder, readout windows |
| `src/training/stage1_sensory.py` | Created | Stage 1 sensory scaffold (2K steps, V1-only bypass, gating checks) |
| `src/training/stage2_feedback.py` | Created | Stage 2 V2+feedback (80K steps, HMM sequences, BPTT) |
| `scripts/train.py` | Created | Entry point with CLI args |
| `tests/test_training.py` | Created | 31 tests |

## 2. Network Extension

Added 3 new trajectory keys to `forward()` aux dict:
- `r_l4_all`: [B, T, N], `r_pv_all`: [B, T, 1], `r_som_all`: [B, T, N]

## 3. CompositeLoss (`src/training/losses.py`)

### API

```python
CompositeLoss(cfg: TrainingConfig, model_cfg: ModelConfig)
```

Contains `_theta_to_channel(theta)` for degree-to-channel conversion and `orientation_decoder` (Linear 36->36).

### Components
1. **`sensory_readout_loss(r_l23_windows, true_theta_windows)`**: CE on L2/3 -> decoded orientation. Accepts degrees directly.
2. **`prediction_loss(q_pred_windows, true_next_theta_windows)`**: NLL on log(q_pred + eps). Accepts degrees directly.
3. **`energy_cost(outputs)`**: Returns `(E_excitatory, E_total)` from the outputs dict.
4. **`homeostasis_penalty(r_l23)`**: Squared penalty when mean rate outside [0.05, 0.5].

### forward() signature
```python
forward(outputs, true_theta_windows, true_next_theta_windows, r_l23_windows, q_pred_windows)
  -> (total_loss: Tensor, loss_dict: dict[str, float])
```

Weights: λ_sensory=1.0, λ_pred=0.5, λ_energy=0.01, λ_homeo=1.0

## 4. Trainer Utilities (`src/training/trainer.py`)

### Parameter management
- `freeze_stage1()`: Freezes L4, PV, L2/3 inhibitory gains (w_som, w_pv_l23). W_rec stays trainable.
- `unfreeze_stage2()`: Unfreezes V2, feedback, deep_template, W_rec.
- `create_stage2_optimizer()`: 4 param groups — V2 (3e-4), feedback (1e-4), W_rec+template (1e-4), decoder (1e-3).

### Scheduler
- `make_warmup_cosine_scheduler()`: Linear warmup then cosine decay.

### Stimulus building
- `build_stimulus_sequence(metadata, model_cfg, train_cfg)`: Returns `(stimulus_seq, cue_seq, task_state_seq, true_thetas, true_next_thetas)` — all in degrees, timestep-level.
- `compute_readout_indices(seq_length, ...)`: Returns list of `(pres_idx, [timestep_indices])`.
- `extract_readout_data(outputs, readout_indices)`: Extracts windowed averages from outputs dict.

## 5. Stage 1: Sensory Scaffold

### Key design: V1-only bypass
Stage 1 runs `_run_v1_only(net, stimulus, n_timesteps=20)` which calls L4 -> PV -> L2/3 directly, with zero SOM drive and zero template modulation. This ensures the sensory scaffold is built without any feedback influence.

### Training
- 2000 steps, Adam lr=1e-3
- Random orientations (uniform 0-180°), variable contrast (0.1-1.0)
- 20 timesteps per presentation, readout from final state
- Loss: CE(decoder(r_l23), target) + λ_homeo * homeostasis

### Gating checkpoints (6 checks)
1. Decoder accuracy ≥ 90%
2. Unimodal tuning (≥ 80% of units)
3. Preferred orientations tile (≥ 70% unique)
4. FWHM 15-40°
5. Mean activity in [0.001, 2.0]
6. Contrast-invariant tuning width (FWHM ratio < 2.0)

## 6. Stage 2: V2 + Feedback

### Training
- 80K steps with BPTT over HMM sequences
- AdamW with 4 LR groups
- Gradient clipping at 1.0
- Linear warmup (1K steps) + cosine decay
- Batch size 32, 50 presentations per sequence
- Readout window: timesteps 4-7 of each ON period

### Data generation
- HMM sequences via `HMMSequenceGenerator`
- Variable contrast, ambiguous presentations, both task states
- Temporal structure: 8 ON + 4 ISI per presentation

### Loss accepts degrees
`true_thetas` and `true_next_thetas` passed in degrees — CompositeLoss converts internally via `_theta_to_channel`.

### Logging
Every 100 steps: total loss, component losses, sensory accuracy, prediction accuracy, gradient norms.

## 7. Entry Point (`scripts/train.py`)

```
python -m scripts.train                          # full pipeline
python -m scripts.train --mechanism dampening
python -m scripts.train --stage 1
python -m scripts.train --stage 2 --stage1-checkpoint ckpt.pt
python -m scripts.train --stage2-steps 500        # abbreviated for testing
```

Saves model + decoder + history to `checkpoints/{mechanism}_seed{seed}/`.

## 8. Test Summary (31 tests)

- TestNetworkAux: 2 (rate trajectories shape, no NaN)
- TestCompositeLoss: 12 (_theta_to_channel, losses, energy, homeostasis squared, forward returns tuple, gradients)
- TestTrainerUtils: 4 (freeze, unfreeze, optimizer groups, scheduler)
- TestReadoutWindows: 3 (indices, shape, correct values)
- TestBuildStimulusSequence: 4 (shapes, ISI zero, ON nonzero, next shifted)
- TestStage1Smoke: 3 (runs, bypasses V2, freezes params)
- TestStage2Smoke: 2 (runs, all 5 mechanisms)
- TestCheckpoint: 1 (save/load identical forward pass)

## 9. Energy Cost Design

Two variants computed in every step:
- **E_excitatory**: L1(r_l4 + r_l23 + deep_template)
- **E_total**: E_excitatory + L1(r_pv + r_som)

Default training uses E_total. Both logged. Energy sweep (Phase 8) varies λ_energy.

## 10. Alignment with Brief

- CompositeLoss accepts degrees, converts internally via `_theta_to_channel`
- forward() returns `(total_loss, loss_dict)` with `.item()` scalar values
- Homeostasis uses **squared** penalty with `target_min=0.05, target_max=0.5`
- Stage 1 **bypasses V2**: runs L4 -> PV -> L2/3 directly for 20 timesteps
- Stage 2 accepts shared `loss_fn` from Stage 1 for decoder continuity
- `create_stage2_optimizer()` has 4 param groups matching the brief
- Checkpoint includes both model and decoder state dicts
