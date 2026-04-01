# Forward() Refactor Report

## 1. Approach Used

**Option A: Packed tensor input.**

`forward()` now accepts a single `packed_input: Tensor` of shape `[B, T, N+N+2]` (stimulus + cue + task_state concatenated along the last dimension). A static method `pack_inputs()` handles the concatenation. Backward compatible: raw `[B, T, N]` stimulus input is auto-detected and zeros are generated for cue/task_state.

Files changed:
- `src/model/network.py` — Added `pack_inputs()`, changed `forward()` signature
- `src/training/stage2_feedback.py` — Uses `torch.compile(net, mode='reduce-overhead')` + packed input
- `src/experiments/paradigm_base.py` — Updated to use `pack_inputs()`
- `tests/test_network.py` — One line fix (old keyword args)

## 2. torch.compile Compilation

**Full-model compile (`torch.compile(net)`) is IMPRACTICAL** for this architecture.

The 600-step `for t in range(T)` loop in `forward()` causes torch.compile to unroll the entire recurrence into a single computation graph. Compilation time:
- `mode='reduce-overhead'`: >30 minutes, did not complete (killed after 30min)
- `mode='default'`: >5 minutes, did not complete (killed)

**Step-level compile (`torch.compile(net.step)`) works well.** Since `step()` processes a single timestep with fixed-shape inputs (B=32, N=36), the graph is small and compilation is fast.

## 3. Timing Results (RTX A6000, B=32, T=600)

| Compile Mode | Step Time | Speedup | 80K Projected | Compile Overhead |
|---|---|---|---|---|
| Uncompiled (baseline) | 1.498s | 1.0x | 33.3h | — |
| `step(default)` | 1.512s | 1.0x | 33.6h | ~2s (hit recompile limit, fell back to eager) |
| `step(max-autotune-no-cudagraphs)` | 0.546s | **2.7x** | **12.1h** | ~13s first step |
| `step(reduce-overhead)` | 0.364s | **4.1x** | **8.1h** | ~10min first 2 steps (CUDA graph capture) |

**Note:** The `default` mode hits dynamo's recompile limit (8) because the `step()` function sees different `requires_grad` states during the first vs subsequent timesteps, causing guard failures.

### Why not 0.169s/step (12x)?

The original debugger estimate of 0.169s/step assumed full-model compilation with `reduce-overhead` (CUDA graphs over the entire 600-step forward). This is impractical — the graph is too large to compile. Step-level compilation captures each individual timestep, which still has 600 Python→CUDA kernel launch transitions per forward pass.

### Root cause of the bottleneck

Each timestep operates on tiny tensors (B=32 × N=36 = 1,152 elements). The A6000 has 10,752 CUDA cores — the GPU is vastly underutilized. The bottleneck is **600 sequential kernel launches** with microscopic per-kernel work, not compute.

## 4. Numerical Correctness

| Check | Result |
|---|---|
| Packed vs raw (zero cue/task) | Exact match (max diff = 0.00e+00) |
| Packed with nonzero cue differs from raw | Yes (max diff = 0.0006) |
| Compiled step vs uncompiled | Match at atol=1e-5 (max diff = 4.47e-08) |
| No NaN in any output | Confirmed |

## 5. Test Suite

**307 tests passed, 0 failed** (25s runtime).

## 6. Recommendation

**Use `step(max-autotune-no-cudagraphs)` for training.** Rationale:
- 2.7x speedup (12.1h for 80K steps) with only 13s compile overhead
- No CUDA graph issues — robust across different batch sizes and sequence lengths
- The `reduce-overhead` mode (4.1x, 8.1h) is faster but has a 10-minute warmup penalty and CUDA graph fragility concerns

### Potential further optimizations (require lead approval):

1. **Increase batch size** (e.g., B=128): More work per kernel, better GPU utilization. *Scientific parameter — requires approval.*
2. **Manual CUDA graph capture**: Capture the full forward loop once, replay for all steps. Complex but could achieve the 12x target.
3. **Vectorize recurrence**: Restructure computation to process multiple timesteps simultaneously where state dependencies allow. Major architecture change.

## 7. Current Stage 2 Training Config

The training loop in `stage2_feedback.py` currently uses:
```python
compiled_net = torch.compile(net, mode='reduce-overhead')
```

This should be changed to step-level compilation based on the findings above. Awaiting lead decision on which mode to use.
