# Merge Report: v2-speedup into speedup-integration

## Merge Summary

Merged `origin/v2-speedup` (commit `200dc04`) into `speedup-integration` (commit `933a2c8`).
Both branches diverged from `40917ea` (training dynamics fixes).

**Conflict**: `src/training/stage2_feedback.py` — feedback_scale assignment.
- Speedup branch: `net.feedback_scale.fill_(...)` (buffer-based for torch.compile)
- Our branch: `net.feedback_scale = ...` (plain attribute, curriculum logic)
- **Resolution**: Keep our curriculum logic (Fix E: burn-in + ramp), use `.fill_()` for buffer compatibility

## Objective Fixes Preserved (Fixes A-E)

| Fix | Feature | Verified |
|-----|---------|----------|
| A | State targets shifted by 1 (`torch.roll`) | YES — `trainer.py` |
| B | KL distributional prediction loss (σ=10°) | YES — `losses.py` |
| C | New metrics (state_acc, ang_err, top3, anchor) | YES — `stage2_feedback.py` |
| D | Reference baselines logged | YES — `stage2_feedback.py` |
| E | V2 curriculum (5K burn-in + 5K ramp) | YES — `stage2_feedback.py` |

## Speedup Features Active

| Feature | File(s) | Verified |
|---------|---------|----------|
| Kernel caching (`cache_kernels()`/`uncache_kernels()`) | `populations.py`, `feedback.py` | YES |
| Preallocated forward loop (`torch.empty`) | `network.py` | YES |
| `StepAux` NamedTuple (replaces dict in inner loop) | `state.py`, `network.py` | YES |
| `NetworkState` NamedTuple | `state.py`, `network.py` | YES |
| `feedback_scale` as registered buffer | `network.py` | YES |
| Batch-vectorised HMM generation | `sequences.py` | YES |
| Vectorised `build_stimulus_sequence` | `trainer.py` | YES |
| Reshape-based `extract_readout_data` | `trainer.py` | YES |
| Full-model `torch.compile(net, mode='reduce-overhead')` | `stage2_feedback.py` | YES |
| `non_blocking=True` GPU transfers | `stage2_feedback.py` | YES |

## Test Results

**307 passed, 0 failed** (42.8s)

## Timing

Full-model `reduce-overhead` compile has very long warmup (~10+ min for the 600-step graph).
Falls back to `max-autotune-no-cudagraphs` if reduce-overhead fails.
Actual per-step timing will be measured from the pilot v3 training logs.

## Files Modified by Merge

| File | Changes from speedup branch |
|------|----------------------------|
| `src/model/feedback.py` | Kernel caching |
| `src/model/network.py` | NamedTuple state/aux, preallocated tensors, buffer feedback_scale |
| `src/model/populations.py` | Kernel caching for W_rec |
| `src/state.py` | New file: NetworkState + StepAux NamedTuples |
| `src/stimulus/gratings.py` | Vectorised grating generation |
| `src/stimulus/sequences.py` | Batch-vectorised HMM |
| `src/training/stage2_feedback.py` | Full-model compile, non_blocking transfers, **merged with curriculum** |
| `src/training/trainer.py` | Vectorised build_stimulus_sequence + reshape extract_readout_data |
| `tests/test_network.py` | Updated for NamedTuple returns |
| `SPEEDUP_REPORT.md` | New file from speedup branch |
