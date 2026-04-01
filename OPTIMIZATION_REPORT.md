# Training Speed Optimization Report

## Problem

Training step for LaminarV1V2Network: ~2.04s per step on RTX A6000 (B=32, T=600, N=36, ~5K params).
At 80K steps = ~45 hours. Target: <0.3s per step (~7 hours total).

## Root Cause: CPU Dispatch Overhead

**Proven via `torch.profiler`**:

| Metric | Value |
|---|---|
| Self CUDA time (actual GPU compute) | **338 ms** |
| Self CPU time (Python dispatch + autograd) | **3,405 ms** |
| CUDA kernel launches per step | 113 |
| Total kernel launches (600 steps) | 67,800 |
| Average kernel duration | ~5 μs |

The GPU does only **338ms of real work** in a 2,042ms training step. The remaining 83% is Python
interpreter overhead dispatching 67,800 tiny CUDA kernels (each ~5μs), plus autograd bookkeeping.

**Secondary cause**: Gaussian kernels (`W_rec`, feedback `K_surround`, `K_center`) are reconstructed
from learnable parameters at **every timestep** even though they don't change within a forward pass.
Cost: ~400ms total (measured: `W_rec` 117ms + `K_surround/K_center` ~300ms over 600 steps).

## Hypotheses Tested

### H1: torch.compile fuses small kernels, reducing dispatch overhead → CONFIRMED

| Configuration | Forward (no grad) | Training (fwd+bwd) | Speedup |
|---|---|---|---|
| **Baseline (no compile)** | 0.646s | 2.042s | 1.0x |
| `torch.compile(step, mode='default')` | 0.216s | 0.732s | 2.79x |
| `torch.compile(step, mode='max-autotune-no-cudagraphs')` | — | **0.575s** | **3.55x** |
| `torch.compile(step, mode='max-autotune')` | — | 0.422s* | 4.84x* |
| `torch.compile(step, mode='reduce-overhead')` | — | FAILS | — |

\* `max-autotune` uses CUDA graphs internally and **fails for B≠32** — unreliable.
| `torch.compile(model, mode='reduce-overhead')` | **0.028s** | 0.169s (1-arg only) | 12.1x* |
| `torch.compile(model, mode='default')` | — | Still compiling after 3+ min | — |

**Evidence**: Compiled profiler shows CUDA events drop from 67,800 → 11,749 (5.8x fewer).
Self CUDA time: 338ms → 31ms. Self CPU time: 3,405ms → 372ms. Inductor generates fused
Triton kernels (e.g., `triton_per_fused_add_clamp_div_mul_neg_softplus_sub`).

### H2: CUDA graphs (reduce-overhead) eliminate kernel launch overhead → PARTIALLY CONFIRMED

**Step-level compile** (`torch.compile(step, mode='reduce-overhead')`): CUDA graphs fail during
backward pass because the autograd tape references tensors overwritten by subsequent replays.
Error: `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`.

**Full-model compile** (`torch.compile(model, mode='reduce-overhead')`): When forward() is
compiled as a whole (including the 600-step loop), the CUDA graph captures ALL steps internally.
State tensors are internal to the graph, avoiding the overwrite issue.

Results with default args (`compiled(stim)`, cue/task=None):
- Forward: 0.028s (22.7x), Training: **0.169s (12.1x)**
- Correctness verified: output diff < 7.5e-8, gradient diff < 1.2e-4
- Memory: 0.09 GB

**CRITICAL CAVEAT**: When all 3 inputs are passed explicitly (`compiled(stim, cue, task)` — as
in the actual training loop), the compiler must trace a much larger graph (3 dynamic input
tensors × 600 unrolled steps). Compilation took 25+ minutes and 9+ GB RAM before being killed.
This makes full-model compile impractical for the current training loop.

**Possible workaround**: Modify forward() to always generate cue_seq and task_state_seq
internally (appending to stimulus_seq or creating from a flag), so the compiler only sees one
dynamic input. This would recover the 12x speedup but requires model changes.

### H3: Pre-caching invariant kernels reduces redundant computation → CONFIRMED (minor)

| Configuration | Training (fwd+bwd) | Speedup |
|---|---|---|
| Kernel caching only (no compile) | 1.305s | 1.56x |
| Kernel caching + torch.compile | 0.653s | 3.13x |
| torch.compile alone | 0.732s | 2.79x |

Caching adds 12% improvement on top of compile (0.732s → 0.653s). Not worth the code complexity.
torch.compile already makes kernel construction cheap (~1.7μs per call via fused Triton kernel).

### H4: TF32 matmul precision improves throughput → RULED OUT

| Configuration | Training (fwd+bwd) | Speedup |
|---|---|---|
| TF32 (high precision) | 2.043s | 1.00x |
| TF32 + torch.compile | 0.736s | 2.77x |

No effect. Matrix dimensions ([32,36] × [36,36]) are too small for TF32 tensor cores to help.
TF32 benefits require matrices ≥256 in at least one dimension.

### H5: max-autotune finds better kernel configurations → CONFIRMED (with caveat)

`max-autotune` (with CUDA graphs): 2.042s → 0.422s (4.84x). BUT **unreliable** — uses
CUDA graphs internally and fails with `RuntimeError: accessing tensor output of CUDAGraphs
that has been overwritten` for batch sizes other than B=32. Not safe for production use.

`max-autotune-no-cudagraphs` (autotuning only): 2.042s → **0.575s (3.55x)**. Stable across
all batch sizes. Gradient correctness verified (max diff 3.66e-04, within atol=1e-3).
Compilation takes several minutes but is a one-time amortized cost.

## Correctness Verification

`torch.compile(step, mode='default')` produces **identical results** to baseline:

| Output | Max absolute difference |
|---|---|
| r_l23 | 5.96e-08 |
| q_pred | 9.31e-09 |
| pi_pred | 4.77e-07 |
| state_logits | 4.47e-08 |
| r_l4 | 8.94e-08 |
| r_pv | 2.68e-07 |
| r_som | 1.49e-07 |

All gradient differences < 6.1e-05 (expected float32 accumulation variance over 600 steps).

## Timing Breakdown (compiled, training)

| Phase | Time | Per step |
|---|---|---|
| Forward | 0.272s | 0.45ms |
| Backward (BPTT) | 0.450s | 0.75ms |
| **Total** | **0.722s** | **1.20ms** |

Backward is 62% of total time. This is the autograd tape unwinding through 600 compiled steps.

## Recommended Implementation

### Primary: One-line change (3.55x speedup)

In `src/training/stage2_feedback.py`, add before the training loop:

```python
# After freeze_stage1(net) / unfreeze_stage2(net), before optimizer creation:
net.step = torch.compile(net.step, mode='max-autotune-no-cudagraphs')
```

**Result**: 2.042s → 0.575s per step. 80K steps: ~45 hours → ~12.8 hours.

First step will be slow (several minutes for autotuning compilation). All subsequent steps
run at compiled speed. The one-time compilation cost is negligible over 80K steps.

**Fallback**: If `max-autotune-no-cudagraphs` causes issues (e.g., compilation errors with
other mechanism types), use `mode='default'` instead (0.732s, 2.79x speedup, ~30s compilation).

**WARNING**: Do NOT use `mode='max-autotune'` — it includes CUDA graphs which cause
`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten` for certain
batch sizes due to recurrent state reuse between timesteps.

Consider adding `torch._dynamo.config.cache_size_limit = 8` if multiple model variants
trigger recompilation.

### Memory impact

Peak GPU memory: 0.12 GB. RTX A6000 has 48 GB. No memory concern.

### What this does NOT achieve

Target was <0.3s per step. We achieve 0.575s with max-autotune-no-cudagraphs (3.55x). The remaining bottleneck
is the Python for-loop calling the compiled step 600 times. Each call has ~0.6ms of
CPU-side dispatch overhead (600 × 0.6ms = 360ms). Eliminating this would require:

1. **Compiling the full forward()** — works for inference (0.028s!) but compilation
   takes minutes and fails for backward due to enormous 600-step unrolled graph.
2. **Custom CUDA extension** — fuse the entire 600-step recurrence into one kernel.
   High implementation effort, fragile to model changes.
3. **Scan-based parallelism** — not applicable to this architecture because each step's
   output depends on the previous step's state (truly sequential recurrence).

The 2.8x speedup from `torch.compile(step)` is the best achievable with a simple change.

## Full Results Summary

| Approach | Train time | Speedup | Complexity | Status |
|---|---|---|---|---|
| Baseline | 2.042s | 1.0x | — | — |
| `torch.compile(step, max-autotune-no-cudagraphs)` | **0.575s** | **3.55x** | 1 line | **RECOMMENDED** |
| `torch.compile(step, default)` | 0.732s | 2.79x | 1 line | Faster compilation, slightly slower |
| `torch.compile(step, max-autotune)` | 0.422s* | 4.84x* | 1 line | UNRELIABLE (CUDA graph failures) |
| Cached kernels + compile | 0.653s | 3.13x | ~100 lines | Marginal gain |
| Cached kernels only | 1.305s | 1.56x | ~80 lines | Superseded by compile |
| TF32 | 2.043s | 1.00x | 1 line | No effect |
| Full model compile (default) | >3 min compile | — | 1 line | Impractical |
| Full model compile (reduce-overhead) | 0.169s (1-arg) | 12.1x (1-arg) | 1 line + model change | Needs forward() refactor |
| reduce-overhead + clone | 0.113s fwd | 5.7x fwd | 5 lines | Inference only |
| reduce-overhead + checkpointing | FAILS | — | ~30 lines | CUDA graph bug |
| max-autotune | Very slow compile | ~same as default | 1 line | Not worth it |

## Environment

- PyTorch 2.10.0+cu130
- NVIDIA RTX A6000 (48 GB)
- CUDA 13.0
- Python 3.13
- Model: 5,108 parameters
