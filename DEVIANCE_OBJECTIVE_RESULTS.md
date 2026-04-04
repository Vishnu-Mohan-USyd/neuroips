# Deviance-Objective Experiment Results

## Summary

Removing sensory supervision from L2/3 causes the emergent feedback operator to learn a dampening profile (suppress expected orientation via SOM inhibition). With sensory supervision on L2/3, the operator learns weak/negligible feedback. The mismatch detection objective on L2/3 is NOT the driver — an ablation removing it produces identical results.

## Architecture

### SOM-only feedback (no excitatory pathway)
- `EmergentFeedbackOperator` has only `alpha_inh` [7] (no `alpha_exc`)
- 7 circular basis functions: G(5), G(15), G(30), G(60), MexHat, Const, Odd
- `forward()` returns `som_drive` only; `center_exc = zeros` in network
- Feedback acts exclusively through SOM inhibition of L2/3

### Delta-SOM (bias-corrected softplus)
Previous softplus computation: `som_drive = pi * softplus(inh_field)` had a constant bias (`softplus(0) = ln(2) = 0.693`) that created tonic SOM drive even with zero prediction, killing L2/3 at high precision.

Delta-SOM fix: `som_drive = pi * (softplus(baseline + inh_field) - softplus(baseline))` where `baseline` is a learnable parameter (init 0.0). This gives:
- `inh_field = 0` -> `som_drive = 0` (no feedback effect)
- `inh_field > 0` -> `som_drive > 0` (more SOM -> suppress L2/3)
- `inh_field < 0` -> `som_drive < 0` (less SOM -> facilitate L2/3)

### Oracle V2
V2 is frozen; `q_pred` is constructed from ground-truth HMM state + current L4 orientation. `pi_pred` fixed at 1.0. This isolates the feedback operator's learning from V2 quality.

### Other fixes applied
1. **Decoder transfer**: Stage 1 trained decoder now transferred to Stage 2 loss_fn
2. **Readout window**: Shifted to last 3 steps of ON period (was middle of ON period)
3. **L4 decoder + mismatch head**: Added to optimizer param groups in trainer.py

## Experimental Conditions

All: SOM-only, delta-SOM, oracle V2 (pi=1.0), 5000 Stage 2 steps, 3 seeds (42, 123, 456).

| Condition | L2/3 objective | L4 objective | lambda_sensory | lambda_l4_sensory | lambda_mismatch |
|-----------|---------------|-------------|----------------|-------------------|-----------------|
| Control | Orientation decode | None | 1.0 | 0.0 | 0.0 |
| Deviance | Mismatch detect | Orientation decode | 0.0 | 1.0 | 1.0 |
| Ablation | None | Orientation decode | 0.0 | 1.0 | 0.0 |

Common: lambda_state=0.25, lambda_energy=0.01, lambda_fb=0.0, freeze_v2=true, delta_som=true.

## Results

### Learned feedback profiles

| Condition | ||alpha_inh|| | K_inh(0 deg) | K_inh(45 deg) | R(dampening) | R(sharpening) | Cross-seed CV |
|-----------|-------------|-------------|--------------|-------------|--------------|---------------|
| Control | 0.35 | -0.051 | +0.005 | -0.83 | +0.97 | <3% |
| Deviance | 2.15 | +0.275 | +0.027 | +0.91 | -0.90 | <0.1% |
| Ablation | 2.15 | +0.275 | +0.027 | +0.91 | -0.90 | <0.1% |

Control: K_inh dipped at center (negative = less SOM at expected). Anti-dampening.
Deviance/Ablation: K_inh peaked at center (positive = more SOM at expected). Dampening.

### L2/3 suppression-by-tuning (SI = (R_off - R_on) / R_off, positive = suppression)

#### Control (seed 42)
| offset | pi=0.5 | pi=1.0 | pi=2.0 | pi=3.0 |
|--------|--------|--------|--------|--------|
| 0 deg | -0.0% | -0.0% | -0.0% | -0.0% |
| 10 deg | +0.0% | +0.0% | +0.0% | +0.0% |
| 20 deg | +0.0% | +0.0% | +0.1% | +0.1% |
| 30 deg | +0.1% | +0.3% | +0.5% | +0.8% |
| 45 deg | +0.2% | +0.3% | +0.6% | +1.0% |
| 90 deg | +0.1% | +0.2% | +0.3% | +0.5% |

Flat/negligible at all pi values.

#### Deviance (seed 42, identical for ablation)
| offset | pi=0.5 | pi=1.0 | pi=2.0 | pi=3.0 |
|--------|--------|--------|--------|--------|
| 0 deg | +3.7% | +7.4% | +14.9% | +22.5% |
| 10 deg | +2.5% | +5.0% | +10.1% | +15.2% |
| 20 deg | +0.7% | +1.3% | +2.6% | +3.9% |
| 30 deg | +0.1% | +0.1% | +0.3% | +0.4% |
| 45 deg | -0.0% | -0.0% | -0.0% | +0.0% |
| 90 deg | +0.0% | +0.0% | +0.0% | +0.0% |

Strong dampening: maximal suppression at predicted orientation, monotonically decreasing.

### Expected vs Neutral vs Unexpected (pi=3.0, stimulus=90 deg)

| | Control | Deviance |
|---|---------|----------|
| R_expected | 0.3264 | 0.2530 |
| R_neutral | 0.3264 | 0.3264 |
| R_unexpected | 0.3248 | 0.3264 |
| Exp vs Neutral | +0.00% | **-22.5%** |
| Unexp vs Neutral | -0.46% | +0.00% |
| Pattern | Flat | **Expected < Neutral = Unexpected** |

Deviance model suppresses expected by 22.5% below neutral. Unexpected equals neutral.

### Post-hoc L2/3 readout (5-fold CV, pi=3.0)

| | Control | Deviance |
|---|---------|----------|
| Accuracy | 0.510 (chance) | **0.916** |

Deviance model's L2/3 carries discriminable expected/unexpected information.

### Training metrics

| Condition | s_acc (L2/3) | l4_acc | mm_acc | cw_acc |
|-----------|-------------|--------|--------|--------|
| Control | 12% | -- | -- | ~55% |
| Deviance | 2% | 81% | 19% (trivial) | ~57% |
| Ablation | 2% | 78% | -- | ~57% |

## Key Findings

1. **Removing sensory supervision from L2/3 is necessary and sufficient for dampening.** The mismatch loss makes no difference (ablation = deviance). The energy cost drives feedback to suppress L2/3 at the predicted channel, and only the sensory loss on L2/3 prevents this.

2. **Sensory supervision on L2/3 produces flat/no feedback**, not center-surround. Previous center-surround results were architectural artifacts (pathway gain asymmetry in dual-pathway, or softplus bias allowing facilitation via negative alpha). With clean architecture (SOM-only + delta-SOM), sensory loss keeps alpha weak.

3. **No condition produces sharpening or center-surround** with the corrected architecture. This remains an open question.

4. **Dampening is extremely reproducible**: all 3 seeds agree to 4 decimal places in the deviance and ablation conditions.

5. **The mismatch head doesn't learn** (mm_acc = trivial baseline), but its gradient still flows. However, since the ablation (no mismatch) produces identical results, this gradient is irrelevant.

## Config files
- `config/exp_sensory_control.yaml` — Control condition
- `config/exp_deviance.yaml` — Deviance condition

## Results location
- `results/deviance_2x2/control_s{42,123,456}/` — Control runs
- `results/deviance_2x2/deviance_s{42,123,456}/` — Deviance runs
- `results/deviance_2x2/ablation_no_mm_s{42,123,456}/` — Ablation runs
