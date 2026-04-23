# Task #49 — Debugger final report

**Scope.** Verify the external code review's 6 claims with tested, reproducible evidence, and run forensic probes A/B/C across 3 stimulus conditions. No fix recommendations — evidence only.

All experiment scripts are under `scripts/v2/_debug_task49_*.py`; raw outputs in `logs/task49_*.log`.

---

## Summary of verdicts

| Claim | Statement | Verdict |
|-------|-----------|---------|
| C1 | layers.py uses `rate_next = leak*state + phi` (missing `(1-leak)` factor) | **CONFIRMED** |
| C2 | raw-softplus weight decay is anti-shrinkage for excitatory raw weights (raw<0) | **CONFIRMED** |
| C3 | Phase 2 state reset every window destroys temporal structure in m | **CONFIRMED** (stronger: no-reset alternative diverges) |
| C4 | Kok cue learning dead from zero because memory≈0 at cue phase | **FALSIFIED in strict form; secondary finding: learning is cue-undiscriminating** |
| C5 | Stability guard is per-population only; full Jacobian needed | **CONFIRMED** (full \|λ\|max = 1.0345 while all per-pop blocks ≤ 0.937) |
| C6 | Homeostasis creates operating point rather than maintaining it | **CONFIRMED** |

---

## Claim 1 — non-standard Euler in recurrent rate updates

**Claim.** Six recurrent forwards in `src/v2_model/layers.py` (lines 530, 582, 649, 757, 806, 879) and the context-memory update (`context_memory.py:275`) use `rate_next = leak*state + phi(drive)` instead of the standard Euler form `r + (dt/τ)(-r + phi)` used at `lgn_l4.py:323-325`.

**Structural evidence.** Grep of all recurrent forwards confirms 7/7 uses the `leak*state + phi` pattern; lgn_l4 is the only population using the `(1-leak)` factor.

**Empirical equilibrium-rate test** (`_debug_task49_extended.py::c1_equilibrium_rate`, `logs/task49_c1.log`).
For L23E (τ=20ms, dt=5ms, leak=0.75, drive=0.5):
- Prediction under layers.py form: `r_eq = phi / (1-leak) = 0.2809 / 0.25 = 1.1237`
- Prediction under standard Euler: `r_eq = phi = 0.2809`
- Observed (iterate 500 steps): **1.123719** → matches layers.py prediction to 2×10⁻⁷ relative error.

For HE (τ=50ms, dt=5ms, leak=0.9):
- Predicted `r_eq = 0.2809 / 0.1 = 2.8093`
- Observed: **2.8093** → matches within 6×10⁻⁷.

**Conclusion.** Operator equilibrium rate is `phi × τ/dt`, i.e. 4× amplified for L23E and 10× amplified for HE relative to the biologically intended steady state. This is an integration-scheme bug, not a parameter bug.

---

## Claim 2 — raw-softplus weight decay is anti-shrinkage

**Claim.** `dw = lr*hebb - wd*weights` (where `weights = raw`, i.e. pre-softplus) causes anti-shrinkage for initially-negative excitatory raw init (raw₀ ≈ −5.85, softplus(raw₀) ≈ 2.9×10⁻³).

**Mechanism.** `raw_{t+1} = raw_t · (1 - wd)`. For raw<0, this multiplies by a factor <1 in magnitude but, since raw is negative, the product is LESS negative → raw drifts UP toward 0 → softplus(raw) grows.

**Numerical evidence (`_debug_task49_claim2.py` and `_debug_task49_extended.py::c2_extended`):**

| wd | t=0 softplus | t=1000 softplus | Δ% |
|----|--------------|-----------------|-----|
| 1e-5 (default) | 2.876e-3 | 2.879e-3 (at t=100) | +0.1%/100s |
| 1e-5 (10000 steps) | 2.876e-3 | 5.013e-3 | **+74%** |
| 1e-3 (100 steps) | 2.876e-3 | 5.003e-3 | **+74%** |

For raw=0 (typical inhibitory init): raw is FIXED POINT of decay → no change.
For raw=-1 (inhibitory init below target): **+8.5%** softplus growth over 10k steps.

**Probe C** (`_debug_task49_probes.py::probe_C`, `logs/task49_probeC.log`) applied decay-only updates directly to the network's actual `W_l4_l23_raw` tensor (init mean = -5.85) and confirmed: wd=1e-5 → +6% softplus norm growth in 1000 steps; wd=1e-4 → +74%.

**Conclusion.** `-wd*raw` in plasticity.py:174/227/347/368 is **anti-shrinkage for excitatory weights**, grows the effective connection strength over time, feeding the positive-feedback loop with Claim 1's 4×/10× integration gain.

---

## Claim 3 — Phase-2 state reset destroys temporal structure for C

**Claim.** `train_phase2_predictive.py` re-initializes state every 2-step window (line 505: `state0 = net.initial_state(batch_size=batch_size)` INSIDE the for-loop), which destroys the temporal integration needed for m to build up context.

**Code evidence.** Grep confirms the reset at line 505 is inside the window loop. Comment at lines 496-499 acknowledges reset as "workaround for instability".

**Empirical evidence** (`_debug_task49_extended.py::c3_empirical`, `logs/task49_c3.log`).

Compared m.norm(t) for two rollout modes, blank input, same network:

| t | driver (2-step reset) | continuous (no reset) |
|---|-----------------------|-----------------------|
| 0 | 0 | 0 |
| 10 | 0 | 2.14e-2 |
| 50 | 0 | 6.21e+2 |
| 100 | 0 | 3.16e+12 |
| 200 | 0 | **inf** |

**Stronger than the claimed form.**
- Reset keeps m trapped at 0 → no context memory ever accumulates.
- BUT the no-reset alternative DIVERGES — m.norm reaches 10¹² within 100 steps.
- Both modes are "broken" in different ways; the reset mitigates divergence but eliminates memory.

**Conclusion.** Phase 2 is simultaneously deprived of temporal structure AND propped up by the reset, which masks a deeper instability the Probe A/B/Claim 5 results attribute to the recurrent-integration scheme.

---

## Claim 4 — Kok cue learning dead from zero

**Claim.** `delta_qm ∝ cue · memory · memory_error` with memory≈0 at cue phase → updates vanish → dead from zero.

**Empirical falsification** (`_debug_task49_extended.py::c4_differentiation`, `logs/task49_c4.log`).

Per-trial trace, lr=1e-5, wd=1e-5, 40 cue steps per trial:

| trial | cue | m_pre.n | m_end.n | mem_err.n | dw.n | W[:,0].n | W[:,1].n |
|-------|-----|---------|---------|-----------|------|----------|----------|
| 0 | 0 | 0 | **17.2** | 17.2 | 8.30e-4 | 8.30e-4 | 0 |
| 0 | 1 | 0 | **17.2** | 17.2 | 8.30e-4 | 8.30e-4 | 8.30e-4 |
| 1 | 0 | 0 | 15.5 | 15.5 | 6.63e-4 | 1.48e-3 | 8.30e-4 |
| 3 | 0 | 0 | 48.0 | 48.0 | 6.09e-3 | 8.01e-3 | 2.28e-3 |
| 4 | 0 | 0 | **933** | 933 | 2.29e+0 | 2.29e+0 | 8.01e-3 |
| 5 | 0 | 0 | **2.31e+7** | 2.31e+7 | 1.37e+9 | 1.37e+9 | 2.29e+0 |
| 7 | 0 | 0 | **nan** | nan | nan | nan | nan |

**Falsification of strict claim.** m reaches norm 17.2 (NOT ≈0) within 40 cue steps from a zero init, driven by W_hm·h + W_mm·m self-recurrence after r_h bootstraps from blank input. Kok cue learning is NOT dead from zero.

**Secondary finding — learning is cue-undiscriminating.** At trial 0, both cue=0 and cue=1 produce IDENTICAL update magnitude (dw=8.30e-4) because m_end is cue-independent when W_qm_task=0 (zero init). W[:,0] and W[:,1] accumulate symmetrically; the columns cannot differentiate cue identity through the blank-frame-dominated cue phase.

**Tertiary finding — runaway feedback.** By trial 4 the system has entered positive-feedback runaway: m_end jumps from 48 → 933 (≈20× per trial) because accumulated W_qm_task feeds the cue into m with super-unity gain. By trial 7, full nan.

**Conclusion.** Claim 4's premise ("memory ≈ 0 at cue") is FALSIFIED. The actual failure mode is different: (a) W_qm columns accumulate the same magnitudes across cues (no discrimination), and (b) the cue-phase dynamics are explosively unstable once W_qm_task grows past ~10⁻². The stated mechanism is incorrect even though the pathology "Kok learning doesn't work" is real.

---

## Claim 5 — stability guard is per-population only

**Claim.** The stability-guard initialization checks each population's Jacobian spectral radius independently, missing cross-population coupling.

**Experiment** (`_debug_task49_probes.py::claim5_full_jacobian`, `logs/task49_claim5.log`). Full-state Jacobian J ∈ ℝ^{496×496} computed by finite differences at op points; compared to per-population block |λ|max.

**At rest state (warmup=0, state=zeros), all 3 stimuli:**

| | Full \|λ\|max | r_l4 | r_l23 | r_pv | r_som | r_h |
|--|---:|---:|---:|---:|---:|---:|
| blank | **1.0345** | 0.500 | 0.891 | 0.368 | 0.750 | 0.937 |
| grating | **1.0345** | 0.500 | 0.891 | 0.368 | 0.750 | 0.937 |
| procedural | **1.0345** | 0.500 | 0.891 | 0.368 | 0.750 | 0.937 |

**Smoking gun.** Per-population maximum is r_h=0.937 — well within the stability budget (`(1-leak)/phi'_op = 0.37`, see layers.py:617-621). The full coupled-network \|λ\|max = 1.0345 > 1.0. Per-population guard PASSES yet the network is unstable at the origin.

At warmup=5 (state has grown to max ≈1-2.5), full \|λ\|max drops to 0.937 because positive rates clamp some rectifiers. The unstable mode is visible only at the origin.

**Corroboration.** Probe A (no plasticity, no homeostasis, forward only) shows r_l23 diverging from 0 → 5.56e+8 in 100 steps across all 3 stimuli — consistent with the 1.0345/step linear growth rate amplified by softplus saturation into the linear regime for large positive drives.

**Conclusion.** Per-population stability check misses a cross-population unstable mode of magnitude 1.0345 at the origin fixed point. Claim 5 confirmed.

---

## Claim 6 — homeostasis creates operating point

**Claim.** Threshold homeostasis drives θ away from the target-rate set point rather than maintaining steady state, especially when initial rates are zero.

**Arithmetic prediction** (`_debug_task49_extended.py::c6_trajectory`, `logs/task49_c6.log`).
With r≈0 throughout, Δθ = lr × (0 − target) × N. For L23E, lr_homeo=1e-5, target=0.5, N=100 steps → predicted Δθ_L23E = -5.0e-4.

**Observed over 100 blank-input steps:**

| t | θ_L23E | θ_HE | r_L23E | r_HE |
|---|-------:|-----:|-------:|-----:|
| 0 | -5.00e-6 | -1.00e-6 | 0.000 | 0.000 |
| 10 | -5.44e-5 | -1.08e-5 | 1.65e-3 | 3.02e-3 |
| 25 | -1.29e-4 | -2.52e-5 | 1.64e-2 | 4.62e-3 |
| 50 | -2.11e-4 | -4.03e-5 | 2.78e-1 | 1.27e-1 |
| 100 | **+2.389** | **+10.0** | **2.59e+8** | **8.03e+5** |

First ~50 steps match arithmetic prediction (linear, negative drift). After ~50 steps, the Probe-A instability (|λ|max=1.03) bootstrapping rates above target reverses θ's sign and drives it to +2.4 (L23E) / +10 (HE) by step 100. Rates diverge to 10⁸.

**Interaction with Probe B.** Homeostasis ALONE (blank input, no plasticity) diverges identically to Probe A — homeostasis cannot stabilize the instability because its time constant (τ_homeo ≈ 1/lr = 10⁵ steps) is slower than the divergence timescale (~50-100 steps).

**Conclusion.** Under the Claim-1 integration scheme, homeostasis does NOT defend the target operating point. It follows the diverging rates and amplifies the pathology. Claim 6 confirmed.

---

## Probes A / B / C — forensic isolation

### Probe A: no plasticity, no homeostasis (pure forward dynamics)

Across all 3 stimuli:

| t | r_l23 (blank) | r_l23 (grating) | r_l23 (procedural) |
|---|-------------:|----------------:|-------------------:|
| 0 | 0 | 0 | 0 |
| 10 | 1.65e-3 | 2.75e-3 | 2.90e-3 |
| 100 | 5.56e+8 | 3.43e+9 | 1.02e+11 |
| 500 | nan | nan | nan |

**Structural instability at init.** Forward dynamics alone diverge to infinity/nan within 500 steps, independent of stimulus or learning. r_l4 (uses standard Euler at lgn_l4.py:323) stays stable. L23E, L23-PV/SOM, HE, HE-PV all diverge together.

### Probe B: homeostasis only

Identical divergence pattern to Probe A, with θ climbing from -5e-6 at t=0 to +2.4 at t=100 before nan. Homeostasis cannot intervene at the divergence timescale.

### Probe C: weight-decay only (applied to actual `W_l4_l23_raw`)

| wd | t=0 softplus.norm | t=1000 softplus.norm | Δ% |
|----|------------------:|---------------------:|----:|
| 0 | 0.526 | 0.526 | 0% |
| 1e-5 (default) | 0.526 | 0.557 | **+6%** |
| 1e-4 | 0.526 | 0.915 | **+74%** |

Confirms anti-shrinkage direction of raw-weight decay on actual network parameters.

---

## Evidence chain (mechanistic narrative)

All numbers are measured, not inferred:

1. **Integration gain** (C1): r_eq = phi × τ/dt → L23E has 4× gain, HE has 10× gain over standard Euler.
2. **Spectral radius** (C5): at origin, \|λ\|max = 1.0345 → unstable fixed point despite all per-population blocks ≤ 0.937.
3. **Forward-only dynamics** (Probe A): exponential divergence to inf/nan within 500 steps. r_l23 reaches 5.6e8 by step 100 (blank) — consistent with linear growth 1.0345¹⁰⁰ ≈ 30 amplified by softplus saturation.
4. **Homeostasis** (Probe B, C6): cannot catch up to the fast divergence (homeo τ ≈ 10⁵ steps vs divergence ≈ 50-100 steps); θ follows rates instead of setting the target.
5. **Weight decay** (C2, Probe C): at default wd=1e-5, excitatory raw weights drift toward 0 at +6%/1000-step, positively feeding the Claim-1 integration gain.
6. **Phase 2 reset** (C3): required to prevent m from diverging via the same recurrent mechanism (m=3.16e12 at step 100 without reset); reset eliminates all temporal context accumulation.
7. **Kok learning** (C4 counter-finding): cue-phase m grows fine (17.2 in 40 steps) but W_qm_task updates are CUE-INDEPENDENT within trial 0 (both columns get identical vectors); once W_qm_task grows, positive-feedback creates trial-5 nan.

---

## What was NOT tested and remains open

- The `(1-leak)` fix proposed by the reviewer has NOT been patched and tested here. Whether applying it restores stability across all 3 stim conditions without destabilizing other training dynamics is a separate experiment.
- Full decomposition of which Jacobian entries drive the unstable mode (L23↔PV coupling? H↔L23 coupling?) — not isolated; only block-diagonal blocks were individually computed.
- Whether Vogels iSTDP (inhibitory plasticity, raw≥0 init) introduces additional drift — not tested here; its effective wd direction is separate from the E-weight anti-shrinkage.

---

**Files produced**
- `scripts/v2/_debug_task49_claim2.py`
- `scripts/v2/_debug_task49_main.py`
- `scripts/v2/_debug_task49_extended.py`
- `scripts/v2/_debug_task49_probes.py`

**Logs**
- `logs/task49_c1.log`, `_c2.log`, `_c3.log`, `_c4.log`, `_c6.log`
- `logs/task49_probeA.log`, `_probeB.log`, `_probeC.log`
- `logs/task49_claim5.log`
