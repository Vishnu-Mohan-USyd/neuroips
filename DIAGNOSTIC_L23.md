# Diagnostic: L2/3 Activity Crushed to Near-Zero

## Failure characterization

At initialization (before Stage 1 training), L2/3 peak activity = **0.000284** when L4 peak = **0.282**. This is a **~1000x drop**. The L2/3 drive goes negative by timestep 3 and stays deeply negative thereafter, with `rectified_softplus` clamping the activation to zero.

---

## Hypotheses under investigation

1. PV divisive normalization on L4 is too strong — Status: **FALSIFIED**
2. PV subtractive inhibition on L2/3 exceeds feedforward input — Status: **CONFIRMED (PRIMARY)**
3. InhibitoryGain on PV→L2/3 initialized too high — Status: **CONFIRMED (same as #2)**
4. W_rec gain cap prevents recurrent amplification — Status: **CONFIRMED (SECONDARY)**

---

## Experiments run

### Experiment 1: L2/3 drive decomposition (20 timesteps, init params)

- **Command**: Ran full V1 circuit, printed every component of L2/3 drive at each timestep
- **Observed**:

| t | r_l4 peak | r_pv | ff peak | pv_inh | total_drive peak | r_l23 peak |
|---|---|---|---|---|---|---|
| 0 | 0.112 | 0.053 | 0.112 | 0.053 | **+0.059** | 0.000 |
| 1 | 0.195 | 0.143 | 0.195 | 0.143 | **+0.052** | 0.003 |
| 2 | 0.252 | 0.251 | 0.252 | 0.251 | **+0.001** | 0.005 |
| 3 | 0.288 | 0.361 | 0.288 | 0.361 | **−0.072** | 0.005 |
| 5 | 0.319 | 0.551 | 0.319 | 0.551 | **−0.232** | 0.004 |
| 10 | 0.310 | 0.786 | 0.310 | 0.786 | **−0.476** | 0.002 |
| 19 | 0.284 | 0.809 | 0.284 | 0.809 | **−0.524** | 0.001 |

**Finding**: By t=3, PV subtractive inhibition (0.361) exceeds feedforward input (0.288). Drive goes permanently negative. L2/3 decays exponentially toward zero (only route is the leaky Euler decay, no positive input).

### Experiment 2: PV clamped to zero on L2/3

- **Command**: Ran same circuit but set `r_pv = 0` for L2/3 computation only
- **Result at t=29**: r_l23 peak = **0.138** (vs 0.000284 with PV — a **486x recovery**)
- **Verdict**: CONFIRMED — PV subtractive inhibition is the primary cause of L2/3 death.

### Experiment 3: PV gain reduced to 0.1

- **Command**: Set `w_pv_l23.gain = 0.1` (from 1.0)
- **Result at t=29**: r_l23 peak = **0.093** (healthy, ~33% of L4 peak)
- **Verdict**: Reducing PV gain by 10x restores healthy L2/3.

### Experiment 4: Critical PV gain threshold

- **Command**: Swept `w_pv_l23` from 0.01 to 1.0
- **Results**:

| w_pv_l23 | r_l23 peak (t=29) | Status |
|---|---|---|
| 0.01 | 0.133 | Healthy |
| 0.05 | 0.114 | Healthy |
| 0.10 | 0.093 | Healthy |
| 0.20 | 0.055 | Marginal |
| 0.30 | 0.021 | Weak |
| **0.35** | **0.004** | **Nearly dead** |
| 0.40 | 0.003 | Dead |
| 0.50 | 0.002 | Dead |
| **1.00** | **0.000284** | **Dead (init value)** |

**Critical threshold**: L2/3 dies at `w_pv_l23 ≈ 0.35`. Initialized at 1.0 — nearly 3x above the death threshold.

### Experiment 5: Dimensional mismatch analysis

- **Command**: Decomposed PV magnitude pathway
- **Finding**: PV pools ALL 36 channels via sum:
  ```
  PV = rectified_softplus(w_pv_l4 * Σ_i(r_l4[i]) + w_pv_l23 * Σ_i(r_l23[i]))
  ```
  - `r_l4.sum() = 1.64` (36 channels)
  - `r_l4.max() = 0.28` (single channel peak)
  - Ratio: **5.8x** — PV magnitude reflects population sum, not single-channel peak
  - PV is then subtracted uniformly from EACH channel: `drive[i] = ff[i] - w_pv_l23 * PV`
  - At init: `PV_sub = 1.0 * 0.79 = 0.79`, but `ff_peak = 0.28`
  - **Drive = 0.28 − 0.79 = −0.51** → clamped to zero

### Experiment 6: Stage 1 training resolves it

- **Command**: Loaded all 6 Stage 1 checkpoints, compared learned parameter values
- **Results** (all checkpoints converge to nearly identical values):

| Parameter | Init value | After Stage 1 | Change |
|---|---|---|---|
| `l23.w_pv_l23.gain` | **1.000** | **0.175** | ↓ 5.7× |
| `l23.gain_rec` | **0.300** | **0.965*** | ↑ 3.2× |
| `pv.w_pv_l4` | **0.744** | **0.097** | ↓ 7.7× |
| `pv.w_pv_l23` | **0.744** | **0.072** | ↓ 10.3× |
| `l23.w_som.gain` | 1.000 | 1.000 | unchanged |
| `l23.sigma_rec` | 15.0 | 11.5 | minor |

\*`gain_rec` raw value = 0.965, but clamped to 0.95 in the kernel → recurrence pushes against the stability cap.

- After Stage 1: `PV_sub = 0.175 * 0.203 = 0.036`, `ff_peak = 0.446` → drive = +0.41 → healthy L2/3 = **0.326**

---

## Confirmed root cause

**The PV subtractive inhibition on L2/3 is initialized pathologically high, causing L2/3 to be effectively dead at initialization.**

The causal chain:
1. `InhibitoryGain(init_gain=1.0)` sets `w_pv_l23 = 1.0`
2. PV pools all 36 channels via sum → PV ≈ 0.79 at steady state
3. PV subtractive term on L2/3 = `1.0 * 0.79 = 0.79`
4. Peak feedforward to L2/3 = `0.28` (single channel)
5. **0.28 − 0.79 = −0.51** → `rectified_softplus(−0.51) = 0` → L2/3 is dead
6. Dead L2/3 → zero recurrence (W_rec @ 0 = 0) → no self-rescue possible
7. The only surviving signal is the tiny transient in the first 2 timesteps before PV ramps up

**Secondary contributing factor**: W_rec gain initialized at 0.3 (capped at 0.95). Even if L2/3 had some activity, the recurrent amplification is too weak at init to compensate for the PV overdrive.

**Stage 1 training compensates** by learning:
- PV gains down by ~6-10× (both on L4 and L2/3)
- Recurrent gain up by ~3× (hitting the 0.95 stability cap)

But this means:
- L2/3 is **dead for the first ~hundreds of training steps** until PV gains decrease enough
- Gradient signal through L2/3 is essentially zero during this period
- Stage 1 training wastes capacity learning to undo a bad initialization
- Any model that skips Stage 1 (e.g. the sharpening network from the profile analysis) has completely dead L2/3

---

## Secondary findings

1. **SOM gain stays at 1.0 after Stage 1**: SOM is not trained during Stage 1 (no feedback signal). This means SOM inhibition with gain=1.0 will add to the suppression problem when feedback activates SOM in Stage 2.

2. **W_rec saturates at the 0.95 cap**: The raw value (0.965) exceeds the cap, meaning the network wants more recurrent amplification than allowed. The cap may be too conservative.

3. **All 6 checkpoints converge to identical values**: The Stage 1 solution is deterministic/unique regardless of mechanism type or V2 input mode (as expected — Stage 1 doesn't use V2/feedback).

4. **The problem is purely about initialization, not architecture**: The subtractive PV→L2/3 pathway is architecturally sound — it just needs proper gain calibration. The population-sum-to-single-channel mismatch means `w_pv_l23` must be initialized well below `ff_peak / PV ≈ 0.28 / 0.79 ≈ 0.35`.
