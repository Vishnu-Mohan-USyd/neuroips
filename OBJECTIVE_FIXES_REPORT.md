# Training Objective & Metrics Fixes Report

## Fixes Implemented

### Fix A: State target alignment bug
- **File**: `src/training/trainer.py`
- `true_states` shifted by 1 presentation using `torch.roll(..., -1, dims=1)` to align with prediction targets
- Both V2 outputs now target the **future**: q_pred predicts next orientation, state_logits predicts next state
- Last position keeps current state (no valid "next")

### Fix B: Distributional prediction target
- **File**: `src/training/losses.py`
- Replaced exact-channel NLL loss with **circular Gaussian KL divergence**
- `sigma_target = 10.0` (~2 channels) — V2 gets partial credit for being close
- Old exact-channel accuracy kept as logged metric (`p_acc`)
- KL: `F.kl_div(log_q, target_dist, reduction='batchmean', log_target=False)`

### Fix C: Better metrics added to Stage 2 logging
- **File**: `src/training/stage2_feedback.py`
- `state_acc`: 3-way latent state accuracy (CW/CCW/neutral)
- `ang_err`: circular angular error in degrees
- `top3`: top-3 channel accuracy
- `anchor`: 12-anchor accuracy (within ±1 channel of nearest anchor)

### Fix D: Reference baselines logged at training start
- **File**: `src/training/stage2_feedback.py`
- `uniform = 0.028` (1/36 chance)
- `same_as_current ≈ 0.20` (only correct when transition fails)
- `oracle_with_state ≈ 0.75` (within ±1 channel, knowing true state)

### Fix E: V2 curriculum — predictor burn-in before feedback
- **File**: `src/training/stage2_feedback.py`
- Sub-phase 2a: **Hard zero feedback** for first 5000 steps (predictor burn-in)
  - All models run as adaptation-only during this phase
  - V2 learns HMM from clean L2/3 signal
- Sub-phase 2b: **Feedback ramp** 0→1 over steps 5000-10000
  - W_rec gain unfrozen at step 5000 (aligned with end of burn-in)
- Full feedback from step 10000 onward

## Test Results

**307 passed, 0 failed** (26.6s)

### Test updated:
- `tests/test_training.py::TestCompositeLoss::test_prediction_loss_perfect`: Updated to build soft circular Gaussian q_pred matching the new KL target distribution (sigma=10.0)

## Files Modified

| File | Changes |
|------|---------|
| `src/training/trainer.py` | Fix A: shifted true_states by 1 (`torch.roll`) |
| `src/training/losses.py` | Fix B: NLL → circular Gaussian KL divergence |
| `src/training/stage2_feedback.py` | Fix C: 4 new metrics, Fix D: baselines, Fix E: burn-in + ramp curriculum |
| `tests/test_training.py` | Updated prediction_loss_perfect test for KL target |

## Verification

| # | Check | Result |
|---|-------|--------|
| 1 | All tests pass | PASS (307/307) |
| 2 | Smoke test: no NaN with new objective | PASS (5-step test, loss=4.61) |
| 3 | Baselines logged at training start | PASS |
| 4 | New metrics computed in log block | PASS (state_acc, ang_err, top3, anchor) |
| 5 | Feedback scale = 0.0 for first 5000 steps | PASS (hard burn-in) |
| 6 | Feedback ramps 0→1 over steps 5000-10000 | PASS |
