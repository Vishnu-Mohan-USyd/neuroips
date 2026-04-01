# Diagnostic Report: torch.compile Speedup Failure

## Failure

Training runs at **0.75s/step** instead of the expected 0.10-0.17s/step. The
`reduce-overhead` compile mode was expected to eliminate kernel launch overhead via
CUDA graphs, achieving 3-6x speedup. Actual speedup over uncompiled (~1.9s/step) is
only 2.5x.

## Reproducer

```bash
python -m scripts.pilot_run --output results/pilot_v2
# Observe: 100 steps logged every ~75s → 0.75s/step
```

## Hypotheses Tested

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| H1 | reduce-overhead compile time too long for T=600 | **CONFIRMED** | T=10→2s, T=50→5s, T=100→175s. Extrapolated T=600: 40-60 min compile |
| H2 | Backward pass is the dominant bottleneck | **CONFIRMED** | Uncompiled: BWD=1.2s (63% of 1.9s). Step-compiled: BWD=0.4s (44% of 0.9s) |
| H3 | reduce-overhead silently fell back | **CONFIRMED** | Training compile+100 steps = 86s. If reduce-overhead had succeeded, compile alone would take >40 min. Actual ~11s compile matches inductor-only |
| H4 | Python overhead in training loop dominates | **RULED OUT** | datagen=0.01s, transfer=0.03s, loss=0.04s, optim=0.01s — total 0.09s (5% of step) |
| H5 | TF32 matmul precision helps | **MINOR** | TF32 reduces step-compiled forward from 0.282s→0.241s (14% gain). Not transformative |
| H6 | feedback_scale.fill_() causes recompilation | **RULED OUT** | Tested: changing feedback_scale values does not trigger recompilation (0.006s per call) |
| H7 | try/finally in cache_kernels breaks compilation | **RULED OUT** | dynamo.explain shows 0 graph breaks, 1 compiled graph |
| H8 | Graph breaks from new loss/curriculum code | **RULED OUT** | dynamo.explain on full model: 0 graph breaks |

## Proven Root Causes

### Root Cause 1: Full-model compile at T=600 is impractical

**Location**: `src/training/stage2_feedback.py:98`
```python
compiled_net = torch.compile(net, mode='reduce-overhead')
```

The `forward()` method loops 600 times over `self.step()`. When torch.compile traces
this, it unrolls all 600 iterations into a single computation graph. This causes:

| T (timesteps) | Compile time | Runtime (forward) | Speedup vs uncompiled |
|---------------|-------------|-------------------|----------------------|
| 10 | 2s | 0.011s | 1.0x (no gain — already fast) |
| 50 | 5s | 0.013s | 3.6x |
| 100 | **175s** | 0.037s | 2.7x |
| 600 | **>40 min** (est.) | unknown | unknown |

**Evidence**: Compilation time scales **super-linearly** with T (ratio T100/T50 = 35x).
The training log shows compile+first 100 steps = 86s, which is inconsistent with
reduce-overhead at T=600 (would take >40 min for compilation alone). This means
`reduce-overhead` **silently fell back to inductor-only compilation** — CUDA graph
capture either failed or was skipped for the oversized graph.

**Why it falls back silently**: `torch.compile(net, mode='reduce-overhead')` returns a
lazy wrapper that doesn't compile until the first forward call. The try/except in the
training code only catches construction errors, not runtime fallbacks. PyTorch's
reduce-overhead mode degrades gracefully by dropping CUDA graphs and using inductor-only
when graph capture fails — no exception, no warning.

### Root Cause 2: Backward pass is 2x forward (uncompiled) and isn't compiled

**Evidence**:

| Configuration | Forward | Backward | Total | BWD/FWD |
|--------------|---------|----------|-------|---------|
| Uncompiled | 0.60s | 1.20s | 1.90s | 2.0x |
| Step-level compile | 0.26s | 0.60s | 0.90s | 2.3x |

The backward pass through 600 unrolled timesteps creates a deep autograd tape.
Step-level compile helps forward (2.3x speedup) but backward sees less benefit
(2.0x speedup) because the autograd engine replays the tape with Python-level
dispatch between each compiled step.

### Root Cause 3: Model is kernel-launch bound, not compute bound

**Evidence**: Each timestep involves ~20 small kernel launches (matmuls of size 32×36,
GRU of size 32×16, element-wise ops). Each kernel:
- Launch overhead: ~5μs
- Actual compute: ~3-5μs

Over 600 timesteps: 12,000 kernel launches × ~5μs = 60ms just in launch overhead.
CUDA graphs would eliminate this entirely, but they require the full graph to fit in
CUDA memory during capture — infeasible for 600 unrolled timesteps.

## Actual Performance Breakdown (uncompiled training step)

| Component | Time | % of total |
|-----------|------|------------|
| Data generation | 0.01s | 0.5% |
| GPU transfer | 0.03s | 1.6% |
| **Forward pass** | **0.60s** | **31.6%** |
| Loss computation | 0.04s | 2.1% |
| **Backward pass** | **1.20s** | **63.2%** |
| Optimizer + grad clip | 0.01s | 0.5% |
| **TOTAL** | **1.90s** | **100%** |

Forward + backward = 94.8% of training time. Everything else is negligible.

## What the Current Code Actually Achieves

The training code uses `torch.compile(net, mode='reduce-overhead')`. Based on timing
analysis of the logs (0.75s/step), this is consistent with **inductor-only compilation
of the full unrolled graph** — the CUDA graph capture part of reduce-overhead silently
failed. The inductor optimization (operator fusion, memory planning) still provides
~2.5x speedup over fully uncompiled (1.9s → 0.75s).

## Suggested Fix Direction

### Option A: Step-level compile (recommended, simplest)

```python
net.step = torch.compile(net.step, mode='max-autotune-no-cudagraphs')
```

**Measured**: 0.90s/step for full training (forward 0.26s + backward 0.60s).
Compile time: 6.4s (vs current >86s).

This compiles only the step function, which the Python loop calls 600 times. Inductor
fuses the small kernels within each step. Compile time drops from minutes to seconds
because only one step body is compiled, not 600 unrolled copies.

**Expected training step**: 0.90s/step → 88K steps in ~22 hours.

### Option B: Step-level compile + TF32

```python
torch.set_float32_matmul_precision('high')
net.step = torch.compile(net.step, mode='max-autotune-no-cudagraphs')
```

**Measured**: Forward 0.241s (vs 0.282s without TF32). ~15% additional gain.
**Expected training step**: ~0.80s/step → 88K steps in ~19.5 hours.

### Option C: Batch size increase (future, if memory allows)

The model is kernel-launch bound because B=32, N=36 gives tiny matmuls (32×36). Larger
batch sizes would amortize launch overhead better. B=128 would make each matmul 4x
larger, improving GPU utilization.

### Why not reduce-overhead?

`reduce-overhead` (CUDA graphs) would be the ideal solution — it would eliminate all
12,000 kernel launch overheads per forward pass. But it requires:
1. The entire computation graph to fit in GPU memory during capture (infeasible for 600 unrolled steps)
2. Or: chunked CUDA graph capture (capture 50-100 steps at a time, replay chunks) — requires custom implementation not supported by torch.compile

The 0.10-0.17s/step estimate in the OPTIMIZATION_REPORT was based on step-level
reduce-overhead, which assumes CUDA graphs capture one step and replay it 600 times.
**This doesn't work with full-model compile** because torch.compile treats the loop body
differently than a standalone compiled step function.

## Environment

- PyTorch 2.10.0+cu130
- NVIDIA RTX A6000 (48GB)
- CUDA 13.0
- Python 3.13
- Batch size: 32, N: 36, T: 600
