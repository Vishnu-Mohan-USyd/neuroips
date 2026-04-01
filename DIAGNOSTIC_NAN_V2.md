# Diagnostic Report: NaN in Pilot v2 Training

## Failure

Training hits NaN at the very first Stage 2 forward pass (training step 0). Loss = NaN from step 50
onwards. All V2, feedback, and recurrence parameters become NaN. `results/pilot_v2/` is empty
because no checkpoint is reached before the crash.

## Reproducer

```python
python -m scripts.pilot_run --output results/pilot_v2
# NaN at Stage 2 step 1 — loss=nan in the first logged step
```

Minimal reproducer: `python /tmp/debug_nan5.py` — runs Stage 1, then traces Stage 2 forward
pass step-by-step.

## Root Cause: W_rec spectral radius > 1 → exponential blowup (unmasked by rectified_softplus)

### What happens

L2/3 activity explodes exponentially during the 600-timestep forward pass:

```
t=  0: L23_max=5.16e-03
t= 34: L23_max=1.20e+03     ← detectable blowup starts here
t= 40: L23_max=1.29e+04
t= 50: L23_max=6.71e+05
t= 52: L23_max=1.48e+06     ← float32 overflow imminent
```

At t=52, the L2/3 drive decomposition shows:

| Component | Value |
|-----------|-------|
| Recurrent (W_rec @ L2/3) | **1.026e+07** |
| Feedforward (L4) | 4.09e-04 |
| PV inhibition | 1.07e+06 |
| Template / SOM | 0.0 |

**Recurrence dominates by 25 billion to one** over feedforward input.

### Why it happens

The Euler update for L2/3 is:
```
r_l23 += (dt/tau) * (-r_l23 + rectified_softplus(W_rec @ r_l23 + ff - inhib))
```

For the discrete Euler step, stability requires:
```
|1 - dt/τ × (1 - λ_max(W_rec))| < 1
```

where `λ_max` is the spectral radius (approximated by max row sum for non-negative kernels).

| State | W_rec gain | Row sum (≈ spectral radius) | Euler growth factor | Stable? |
|-------|-----------|---------------------------|--------------------|---------| 
| **Before Stage 1** | 0.300 | 2.256 | **1.126** | **NO** |
| **After Stage 1** | 1.026 | 8.097 | **1.710** | **NO** |

After Stage 1: growth factor = 1.710 per timestep. Over 600 timesteps: **1.710^600 = 5.6e+139**.
This overflows float32 (max ≈ 3.4e+38) by step ~165.

### Why the old code didn't blow up

The old `shifted_softplus` had a **negative floor at -0.6931**. When L2/3 activity grew, PV
increased, driving L2/3 drive negative. `shifted_softplus(negative_drive) ≈ -0.693`, which
created a natural rate limiter. The Euler step became:

```
r_l23 += 0.1 * (-r_l23 + (-0.693))   →   r_l23 converges toward -0.693
```

This negative floor acted as an **accidental stability mechanism** — it prevented unbounded growth
by bounding the activation output. The system oscillated (as diagnosed in DIAGNOSTIC_STUCK_MODELS.md)
but never exploded.

The new `rectified_softplus` (clamped to ≥ 0) removed this floor. For large positive inputs,
`rectified_softplus(x) ≈ x - 0.693`, which is essentially linear and provides no growth limiting.
The underlying instability (spectral radius > 1) is now fully exposed.

**The W_rec was unstable even before Stage 1** (spectral radius 2.26, growth factor 1.13). Stage 1
uses only 20 timesteps (via `_run_v1_only`), so the growth is ~1.13^20 ≈ 11x — bounded by
stimulus amplitude. Stage 2 uses 600 timesteps, where 1.13^600 = 3.7e+31 — immediate overflow.

### Evidence chain

1. **Reproduced**: NaN at training step 0, first forward pass. Command: `python /tmp/debug_nan5.py`
2. **Localized**: L2/3 max grows exponentially starting at t≈30, reaches 1.48e6 by t=52
3. **Drive decomposition**: Recurrent component = 1.026e+07, feedforward = 4.09e-04 (25B:1 ratio)
4. **Stability analysis**: W_rec row sum = 8.10 after Stage 1, Euler growth factor = 1.71/step
5. **Causal proof**: The instability exists even with initial W_rec (growth factor 1.13), but was
   masked by shifted_softplus's negative floor

## Suggested Fix Direction

The W_rec recurrent kernel must be **stability-constrained**. Options:

1. **Row-normalize W_rec**: `W_rec = gain * K / K.sum(dim=-1, keepdim=True)`. This ensures row
   sums equal `gain`, so setting `gain < tau/dt = 10` guarantees stability. With gain=0.3 and
   row-normalized K, spectral radius = 0.3, growth factor = 1 - 0.1*(1-0.3) = 0.93 (stable).

2. **Clamp gain to enforce stability**: `gain = softplus(gain_raw).clamp(max=tau/dt - epsilon)`.
   This ensures W_rec spectral radius < tau/dt, which makes the Euler step stable.

3. **Row-normalize AND limit gain**: Most robust. Row-normalize so row_sum = gain, then clamp
   gain < 1.0 so the recurrence is contractive. This guarantees stability for any sigma value.

**Critical note**: Fix 1 from the previous diagnostic (replacing shifted_softplus with
rectified_softplus) is necessary but not sufficient. It prevents negative rates but exposes the
recurrence instability. Both fixes must be applied together.

## Environment

- PyTorch 2.10.0+cu130
- NVIDIA RTX A6000
- Python 3.13
- ModelConfig: dt=1, tau_l23=10, gain_rec init=0.3, sigma_rec init=15.0
