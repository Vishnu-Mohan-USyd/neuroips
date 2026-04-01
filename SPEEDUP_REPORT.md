# V2 Training Speedup Report

## Summary

Comprehensive optimization of the neuroips Stage 2 training loop, targeting Python dispatch overhead (83% of per-step time) without changing any scientific content or model behavior. All 307 tests pass. All 5 feedback mechanisms verified.

## Baseline Performance

| Metric | Value |
|--------|-------|
| Baseline (uncompiled) | 2.042s/step |
| Previous best (torch.compile step) | 0.575s/step |
| 80K steps at 0.575s/step | ~12.8 hours |
| Bottleneck | 83% Python dispatch + autograd overhead, not GPU math |
| Kernel launches per step | 113 (67,800 over 600-step unroll) |

## What Was Changed

### Phase 1: Kernel Caching
**Files:** `src/model/populations.py`, `src/model/feedback.py`, `src/model/network.py`

- `V1L23Ring.W_rec` property was rebuilt from 2 scalar params every timestep (600x per forward pass). Added `cache_kernels()` / `uncache_kernels()` to compute once per forward pass.
- `FeedbackMechanism._make_kernel()` was called 1-4 times per timestep for mechanisms A/B/C. Added `cache_kernels()` / `uncache_kernels()` with `_get_surround_kernel()` / `_get_center_kernel()` helpers.
- `LaminarV1V2Network.forward()` brackets the timestep loop with cache/uncache in a try/finally.
- **Impact:** Eliminates ~2,400 redundant 36x36 kernel builds per forward pass.

### Phase 2+3: Vectorized Data Pipeline + Readout
**Files:** `src/training/trainer.py`, `src/stimulus/gratings.py`

- `build_stimulus_sequence()` had nested Python loops (outer over S=50 presentations, inner over steps_on=8 and steps_per=12). Replaced with a single vectorized `generate_grating()` call for all B*S stimuli, then `unsqueeze+expand+reshape` for temporal expansion.
- `population_code()` recreated preferred orientations via `torch.arange()` every call. Added `_prefs_cache` keyed on `(n_orientations, period, device)`.
- Added `non_blocking=True` to all `.to(dev)` calls in the training loop.

### Phase 4: Preallocated Forward Loop
**Files:** `src/model/network.py`, `src/state.py`, `src/training/stage2_feedback.py`

- `forward()` used 9 Python lists with `.append()` per timestep (5,400 appends) + 9 `torch.stack()` calls. Replaced with `torch.empty()` preallocated tensors and `tensor[:, t] = value` index writes.
- `step()` returned a Python dict (`aux_t`), which is a graph break for `torch.compile`. Replaced with `StepAux` NamedTuple.
- `feedback_scale` was a Python float attribute accessed via `getattr()` (graph break). Registered as a buffer via `register_buffer()`.
- `stage2_feedback.py` updated to use `feedback_scale.fill_()` and `feedback_scale.item()`.

### Phase 5: Full-Model Compile
**Files:** `src/training/stage2_feedback.py`

- Previous: `torch.compile(net.step, mode='max-autotune-no-cudagraphs')` — compiled individual timestep.
- New: `torch.compile(net, mode='reduce-overhead')` — compiles entire model including the 600-step forward loop. Falls back to `max-autotune-no-cudagraphs` if reduce-overhead fails.
- Uses `compiled_net` for forward passes, `net` for parameter access and buffer updates.
- **One-time compilation cost:** ~30-40 minutes (amortized over 80K steps).

### Phase 6: Batch-Vectorized HMM Generator
**Files:** `src/stimulus/sequences.py`

- `sample_state_sequence()` had a scalar Python loop with `.item()` per timestep. Added `batch_sample_state_sequence()` that vectorizes across the batch dimension.
- `generate_orientation_sequence()` had scalar `.item()` loops with Python if-statements. Added `batch_generate_orientation_sequence()` using pre-sampled random values and masked tensor operations (`torch.where`).
- `HMMSequenceGenerator.generate()` had a per-batch-element loop. Now uses the batch-vectorized functions with vectorized reliability drops.
- Task state assignment loop replaced with boolean mask indexing.
- **Note:** RNG stream differs from the scalar version (acceptable — statistical properties preserved).

## Post-Optimization Performance

| Metric | Value |
|--------|-------|
| Uncompiled forward pass (B=32, T=600) | 548ms |
| Per timestep | 0.913ms |
| Throughput | 35,036 timesteps/sec |
| Projected with full compile (reduce-overhead) | ~0.10-0.17s/step |
| Projected 80K steps | ~2.2-3.8 hours |
| torch.compile one-time cost | ~30-40 minutes |

The uncompiled forward pass (548ms) is already faster than the previous *compiled* baseline (575ms), confirming Phases 1-4 deliver meaningful speedup independently of torch.compile.

## Validation

| Check | Result |
|-------|--------|
| Test suite (307 tests) | All pass |
| All 5 mechanisms train without NaN | Pass |
| All losses finite (~5.67) | Pass |
| Gradients flow through all trainable params | Pass |
| No NaN in any output | Pass |
| Numerical equivalence (kernel caching) | max diff = 0.00e+00 |

### Mechanisms Verified
- CENTER_SURROUND: loss=5.683, nan=False, grads_ok=True
- DAMPENING: loss=5.683, nan=False, grads_ok=True
- SHARPENING: loss=5.683, nan=False, grads_ok=True
- ADAPTATION_ONLY: loss=5.683, nan=False, grads_ok=True
- PREDICTIVE_ERROR: loss=5.683, nan=False, grads_ok=True

## What Was NOT Changed

- No model architecture changes
- No loss function changes
- No training hyperparameter changes
- No test modifications (except updating dict access to NamedTuple attribute access for StepAux)
- No changes to experimental paradigms or analysis modules
- Old scalar HMM functions preserved for backward compatibility
- Old `compute_readout_indices()` / `extract_readout_data()` preserved for experiments

## Files Modified

| File | Changes |
|------|---------|
| `src/model/populations.py` | Kernel caching for V1L23Ring |
| `src/model/feedback.py` | Kernel caching for FeedbackMechanism |
| `src/model/network.py` | Preallocated forward, StepAux, feedback_scale buffer, cache bracket |
| `src/state.py` | StepAux NamedTuple definition |
| `src/training/trainer.py` | Vectorized build_stimulus_sequence |
| `src/training/stage2_feedback.py` | Full-model compile, buffer updates, non_blocking transfers |
| `src/stimulus/sequences.py` | Batch-vectorized HMM generation |
| `src/stimulus/gratings.py` | Preferred orientations cache |
