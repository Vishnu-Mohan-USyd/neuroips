# Phase 4 Report: V2 + Feedback + Full Network

**Status:** 153/153 tests pass (51 model + 56 network + 46 stimulus). Task #7 completed.

## 1. Files Created

| File | Description |
|------|-------------|
| `src/model/v2_context.py` | V2ContextModule — GRU-based latent-state inference |
| `src/model/feedback.py` | FeedbackMechanism — unified kernel family for Models A-E |
| `src/model/network.py` | LaminarV1V2Network — top-level composer with step() and forward() |
| `tests/test_network.py` | 56 tests covering all validation checks from brief |

No existing files were modified.

## 2. FeedbackMechanism Implementation

### Kernel Infrastructure

All models A/B/C share **one code path**. A `dists_sq` buffer holds precomputed `[N, N]` pairwise squared circular distances (registered once at init). `_make_kernel(sigma)` builds a circular Gaussian kernel: `exp(-dists_sq / (2 * sigma^2))`.

### Parameters Per Mechanism

| Mechanism | Parameters | Init Width | Constraint |
|-----------|-----------|------------|------------|
| A (dampening) | surround_gain_raw, surround_width_raw | sigma ~10 deg | width clamped <= 15 deg |
| B (sharpening) | surround_gain_raw, surround_width_raw | sigma ~50 deg | width clamped >= 45 deg |
| C (center-surround) | surround_gain/width + center_gain/width | surround ~30 deg, center ~10 deg | all free |
| D (adaptation_only) | none | — | returns zeros |
| E (predictive_error) | none | — | SOM zeros; error = shifted_softplus(l4 - template) |

### Three Methods

- `compute_som_drive(q_pred, pi_pred)` -> `[B, N]`: For A/B, applies `surround_gain * K(surround_width) @ q_pred * pi_pred`. For C, subtracts center component. For D/E, returns zeros.
- `compute_center_excitation(q_pred, pi_pred)` -> `[B, N]`: Only nonzero for Model C. Returns `center_gain * K(center_width) @ q_pred * pi_pred`.
- `compute_error_signal(r_l4, deep_template)` -> `[B, N]`: For Model E, `shifted_softplus(r_l4 - deep_template)`. For all others, returns r_l4 unchanged.

### Width Constraint Properties

Implemented via properties with clamp:
- `surround_width`: softplus(raw) then clamp(max=15) for A, clamp(min=45) for B, unclamped for C
- Verified: forcing raw=10.0 on dampening -> effective 15.0 deg. Forcing raw=-10.0 on sharpening -> effective 45.0 deg.

## 3. V2 Context Module

- `GRUCell(74, 16)`: input = L2/3 (36) + cue (36) + task_state (2)
- Three output heads:
  - `head_q`: Linear(16, 36) -> softmax -> `q_pred [B, N]` (sums to 1, non-negative)
  - `head_pi`: Linear(16, 1) -> sigmoid * pi_max -> `pi_pred [B, 1]` (bounded [0, 5.0])
  - `head_state`: Linear(16, 3) -> raw logits `[B, 3]` (no softmax)

## 4. Network step() and forward() Design

### step() signature
```
step(stimulus, cue, task_state, state) -> (NetworkState, aux_dict)
```
- `aux_dict` contains: `q_pred [B, N]`, `pi_pred [B, 1]`, `state_logits [B, 3]`
- V2 outputs are NOT in NetworkState — they're transient per-step outputs for loss computation

### forward() signature
```
forward(stimulus_seq, cue_seq=None, task_state_seq=None, state=None) -> (r_l23_all, final_state, aux)
```
- `r_l23_all`: `[B, T, N]`
- `aux` stacks over time: `q_pred_all [B, T, N]`, `pi_pred_all [B, T, 1]`, `state_logits_all [B, T, 3]`, `deep_template_all [B, T, N]`
- cue_seq and task_state_seq default to zeros if None

### Dependency Order
L4 -> PV -> V2 (uses L2/3_{t-1}) -> deep_template -> SOM -> L2/3

## 5. Numerical Results

### SOM Drive Profiles (one-hot q_pred at channel 9 = 45 deg, pi_pred = 3.0)

| Mechanism | ch9 (expected) | ch0 (far) | ch15 (flank) | max | min | Behavior |
|-----------|---------------|-----------|-------------|-----|-----|----------|
| Dampening | +3.000 | +0.000 | +0.033 | +3.000 | +0.000 | Narrow peak AT expected |
| Sharpening | +3.000 | +2.001 | +2.506 | +3.000 | +0.594 | Broad, high everywhere |
| Center-surround | +0.000 | +0.974 | +1.786 | +1.996 | +0.000 | Zero center, high flanks |
| Adaptation-only | 0 | 0 | 0 | 0 | 0 | No feedback |
| Predictive-error | 0 | 0 | 0 | 0 | 0 | SOM inactive |

Key discriminations:
- **Dampening**: peak at ch9, ratio to far = 24,959x — narrow, targeted inhibition
- **Sharpening**: flank/peak ratio much higher than dampening — broad inhibition sparing nothing
- **Center-surround**: ch15 (flank, 1.786) > ch9 (center, 0.000) — spares expected, inhibits surround

### Model E Error Signal (r_l4 = 1.0 everywhere, template[9] = 3.0)

| Channel | Value | Meaning |
|---------|-------|---------|
| error[9] (predicted) | -0.566 | shifted_softplus(1-3) — strongly suppressed |
| error[0] (unpredicted) | +0.620 | shifted_softplus(1-0) — passed through |

### Center Excitation

Only Model C produces nonzero center excitation. All other mechanisms return zeros. Verified for all 5 mechanisms.

### V2 Output Bounds

- q_pred: sum = 1.0000 (exact via softmax), all values >= 0
- pi_pred: bounded in [0, 5.0] (sigmoid * pi_max) — stress-tested with extreme inputs
- state_logits: raw (do NOT sum to 1) — verified

### V2 Initial Output (all-zero inputs)

- q_pred: near-uniform, max=0.035, min=0.021
- pi_pred: 2.469
- state_logits: [-0.157, -0.191, 0.070]

### Golden Trial (dampening, theta=45 deg, contrast=0.8, T=10, seed=42)

| Metric | Value |
|--------|-------|
| L2/3 peak channel | 9 (45 deg) |
| L2/3 peak rate | 0.016 |
| PV rate | 0.019 |
| q_pred entropy | 3.573 (near-uniform after 10 steps) |
| pi_pred | 2.659 |
| Determinism | Re-run identical to atol=1e-6 |

### Gradient Flow

All trainable params receive gradients for all 5 mechanisms using comprehensive loss (r_l23 + state_logits + template + q_pred + pi_pred). One excluded param: `l4.pv_gain.gain_raw` — stored on L4 for convenience but unused in the network forward path.

### State Detachment

Confirmed: detaching NetworkState between forward() calls prevents gradient leakage from trial 2 back through trial 1.

## 6. Deviations from Brief

1. **Init values fixed**: Brief had `tensor(1.5)`, `tensor(4.0)`, `tensor(3.0)` for width raws, but `softplus(3.0) = 3.05 deg`, not ~30 deg. Added `_inv_softplus()` helper for correct initialization at 10/50/30 deg. Same pattern as `populations.py` uses for `sigma_rec_raw`.

2. **step() returns tuple**: Brief noted V2 outputs aren't in NetworkState and suggested returning them separately. Implemented as `step() -> (NetworkState, aux_dict)`.

3. **Gradient test uses comprehensive loss**: Loss on `r_l23_all.sum()` alone leaves `deep_template.gain_raw` (for models A-D) and `v2.head_state.weight` without gradients, since those output paths don't feed into r_l23. The training loss will include prediction and energy terms, so the test uses a comprehensive loss matching that.

No other deviations. All 8 validation checks from the brief are covered.

## 7. Test Summary (56 tests in test_network.py)

- TestV2Context: 6 tests (shapes, q_pred sums to 1, pi_pred bounded, state_logits raw, GRU updates)
- TestFeedbackMechanismBasics: 15 tests (shapes for all 5 mechanisms x 3 methods)
- TestMechanismSpecificSOM: 5 tests (pattern verification per mechanism)
- TestCenterExcitation: 1 test (only C is nonzero)
- TestErrorSignal: 2 tests (E suppresses predicted; non-E pass through)
- TestWidthConstraints: 2 tests (dampening max, sharpening min)
- TestNetworkForward: 13 tests (all mechanisms forward, no NaN, step returns, defaults)
- TestGradientFlowNetwork: 5 tests (all mechanisms)
- TestStateDetachment: 1 test
- TestGoldenTrials: 4 tests (dampening/sharpening/center-surround SOM + full network)
- TestPhase4Numerical: 3 print tests (SOM profiles, V2 initial, error signal)
