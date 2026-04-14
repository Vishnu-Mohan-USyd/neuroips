# Project Summary: Laminar V1-V2 Expectation Suppression Model

**Last updated:** 2026-04-14
**Branch:** `dampening-analysis`
**Repository:** `/mnt/c/Users/User/codingproj/freshstart`
**Remote GPU:** `vishnu@100.123.25.88` (reuben-ml, Tailscale)

---

## 1. Variable and Term Definitions

### Neural Populations
- **r_l4** `[B, N]`: V1 Layer 4 excitatory ring firing rates. N=36 channels spanning 0–175° in 5° steps. Driven by population-coded grating input with Naka-Rushton contrast gain, divisive normalization by PV, and stimulus-specific adaptation (τ_a=200).
- **r_pv** `[B, N]`: PV (parvalbumin) interneuron pool. Provides divisive normalization to L4. Integrates L4 and L2/3 with τ_pv=5.
- **r_l23** `[B, N]`: V1 Layer 2/3 excitatory ring firing rates. The primary readout population. Receives L4 input, recurrent excitation (W_rec, σ_rec=15°, gain_rec=0.3), feedback excitation (center_exc), PV inhibition, and SOM inhibition.
- **r_som** `[B, N]`: SOM (somatostatin) interneuron firing rates. Integrates som_drive_fb (feedback-driven inhibition) with τ_som=10. Provides subtractive surround inhibition to L2/3.
- **h_v2** `[B, H]`: V2 context module GRU hidden state (H=16). Encodes the latent temporal context inferred from bottom-up signals.

### V2 Feedback Pathway
- **V2ContextModule**: GRU(input_dim=110, hidden_dim=16) → head_mu (softmax orientation prior), head_pi (precision), head_feedback (raw feedback signal). Input: r_l4 + r_l23_prev + cue + task_state = 36+36+36+2=110.
- **feedback_signal** `[B, N]`: Raw 36-channel output of head_feedback(h_v2). NOT clipped — can be positive or negative.
- **fb_scale** `[0, 1]`: Feedback warmup multiplier. Ramps from 0→1 during burn-in/ramp in Stage 2.
- **scaled_fb** = feedback_signal × fb_scale.
- **center_exc** = relu(+scaled_fb): Positive part → additive excitation to L2/3 (Dale's law E pathway).
- **som_drive_fb** = relu(−scaled_fb): Negative part → drives SOM interneurons → subtractive inhibition (Dale's law I pathway).

### Task and Regime
- **task_state** `[B, 2]`: One-hot task relevance signal. [1,0]=focused (relevant), [0,1]=routine (irrelevant), [0,0]=baseline.
- **Focused / Relevant regime**: task_state=[1,0]. The stimulus orientation is task-relevant (e.g., must be discriminated). Trained with high sensory loss weight (3.0×), zero mismatch weight.
- **Routine / Irrelevant regime**: task_state=[0,1]. The stimulus orientation is task-irrelevant (e.g., ignored grating). Trained with low sensory weight (0.3×), high mismatch weight (1.0×).
- **Baseline**: task_state=[0,0]. No task signal. In Network_both (per-regime feedback), this zeros both feedback heads → equivalent to feedback-off. In Network_mm (shared head), feedback is still active.
- **Markov task_state**: Per-presentation Markov process with p_switch=0.2. Each presentation independently switches regime with 20% probability.

### Metrics
- **s_acc** (sensory accuracy): 36-way orientation decoding accuracy from L2/3 using a trained linear decoder.
- **s_acc_rel** / **s_acc_irr**: Sensory accuracy computed only on focused (relevant) or routine (irrelevant) presentations, respectively.
- **Δ_sens** = s_acc_rel − s_acc_irr: Sensory dissociation between regimes. Positive = better decoding when relevant.
- **mm_acc** (mismatch accuracy): Binary classification accuracy for expected vs. deviant stimuli, using a 2-layer MLP head on L2/3.
- **mm_acc_rel** / **mm_acc_irr**: Mismatch accuracy on focused / routine presentations.
- **Δ_mm** = mm_acc_irr − mm_acc_rel: Mismatch dissociation. Positive = better mismatch detection when irrelevant.
- **M7 (delta_acc)**: Change in decoder accuracy when feedback is ON vs. OFF, at a given orientation perturbation δ (e.g., δ=10° = neighboring channel). M7>0 means feedback improves fine discrimination.
- **M10 (amplitude ratio)**: Ratio of mean r_l23 magnitude with feedback ON vs. OFF. M10>1 = feedback amplifies; M10<1 = feedback dampens.
- **FWHM**: Full-width at half-maximum of the stim-centered r_l23 population profile (8-anchor average). Narrower = sharper tuning.
- **FWHM_Δ** = FWHM_on − FWHM_off: Negative = feedback sharpens tuning; positive = feedback broadens.
- **OSI** (Orientation Selectivity Index): (R_pref − R_orth) / (R_pref + R_orth). Higher = more selective.
- **E/I ratio**: (|r_l4| + |r_l23|) / (|r_pv| + |r_som|). Excitatory vs. inhibitory population balance.

### Expected vs. Unexpected
- **Expected**: stim_offset=0°. The actual stimulus orientation matches the prediction (stim = oracle). Feedback and stimulus are aligned → constructive interference.
- **Unexpected**: stim_offset=±30°. The actual stimulus is shifted 30° from the prediction. Feedback pushes at a different location → destructive interference at stim channel.
- **Mismatch labels**: Binary (0=expected, 1=deviant). Computed by comparing the presented stimulus orientation to the V2 prediction (oracle_theta) with a threshold of 5° (transition_step). First presentation in each sequence is masked (no prior prediction available).

### Training Pipeline
- **Stage 1 (Sensory scaffold)**: 2000 steps, Adam lr=1e-3. Trains L2/3 recurrent weights + PV + decoder with random gratings at variable contrast, no V2/feedback. V1-only forward pass: L4 → PV → L2/3.
- **Stage 1 gating criteria** (all 6 must pass):
  1. L2/3 decoder accuracy ≥ 90% (36-way)
  2. Each L2/3 unit has unimodal tuning
  3. Preferred orientations tile 0–175°
  4. FWHM 15–30°
  5. Mean activity in target range [0.05, 0.5]
  6. Contrast-invariant tuning width (PV normalization working)
- **Stage 2 (V2 + feedback)**: 5000 steps (simple-dual) or up to 80000 (earlier Phase 2.x). AdamW with separate LR groups: V2 params (lr_v2=3e-4), W_rec (lr_feedback=1e-4), decoder (lr=1e-3). Gradient clip=1.0. Warmup (500 steps) + burn-in (1000 steps, fb_scale=0) + ramp (1000 steps, fb_scale: 0→1).

---

## 2. Three-Regime Predecessor

**Branch:** Predates `single-network-dual-regime`. Commit `93e8d76`.

### Architecture
Full laminar model with 8 modules: V1L4Ring, PVPool, V1L23Ring, SOMRing, VIPRing (now deleted), DeepTemplate (now deleted), V2ContextModule, EmergentFeedbackOperator (now deleted). VIP provided disinhibition via VIP→SOM pathway. DeepTemplate was a parametric feedback kernel (7-basis, then 36-weight direct). EmergentFeedbackOperator was a learned feedback→V1 mapping.

### Training Setup
**Three separate checkpoints**, one per regime, each with different loss hyperparameters:
- **Dampening** (λ_sensory=0.0, λ_energy=2.0): Energy-only objective penalizes all L2/3 activity. The network learns to suppress via SOM.
- **Sharpening** (λ_sensory=1.0, λ_energy=0.01): Sensory objective forces the network to maintain decodable tuning. Feedback sharpens peaks and suppresses flanks.
- **Enhancement** (λ_sensory=1.0, no energy constraint): Feedback amplifies activity without constraint.

### Results
Individual networks DID fall into distinct regimes:
- Dampening net: M10=0.3–0.5 (suppression), mean r_l23 reduced 40–60%, decoder preserved (Kok-compatible).
- Sharpening net: M10=3–5× (amplification), FWHM narrowed 5–8°, flanks suppressed (McAdams & Maunsell compatible).

### Limitation
Cross-checkpoint comparison was confounded by seed × hyperparameter interaction. Each network trained independently — differences could be due to random initialization, not loss structure. A single network receiving interleaved task signals was needed for a controlled comparison.

### Rationale for Moving On
Commit `1987fab` removed all legacy code (VIP, apical gain, basis functions, coincidence gate, MechanismType). The single-network-dual-regime design was adopted to eliminate seed confounds and test whether one architecture can produce both regimes.

---

## 3. Single-Network-Dual-Regime Architecture

**Branch:** `single-network-dual-regime`. Commit `194a3c7` (initial), through current HEAD.

### Architecture (5 modules)
1. **V1L4Ring**: 36-channel excitatory ring. W_ff=identity. Divisive normalization by PV. Adaptation τ_a=200, α=0.3.
2. **PVPool**: Divisive normalization interneurons. τ_pv=5, σ_norm=1.0.
3. **V1L23Ring**: 36-channel excitatory ring. Recurrent excitation (σ_rec=15°, gain_rec=0.3). Receives center_exc (feedback E), SOM inhibition, PV inhibition.
4. **SOMRing**: 36-channel inhibitory ring. Driven by som_drive_fb (feedback I pathway). τ_som=10.
5. **V2ContextModule**: GRU(110→16) + head_mu(16→36) + head_pi(16→1) + head_feedback(16→36). Uses `v2_input_mode='l4_l23'` (receives both L4 and L2/3).

**Deleted from predecessor:** VIPRing, DeepTemplate, EmergentFeedbackOperator.

### Key Architectural Finding: V2 GRU Does NOT Receive Oracle
In evaluation (oracle_mode=True), the V2 GRU is called with actual bottom-up inputs (r_l4, r_l23_prev, cue, task_state, h_v2_prev) — NOT with oracle_q_pred. The oracle only sets mu_pred/pi_pred for loss computation. This means: **feedback_signal is computed entirely from what V2 observes, not what it's told to expect.** V2 learns to infer the stimulus orientation from bottom-up signals and generates its own feedback accordingly. (Source: `network.py:139–152`)

### Loss Formula (User's Target Spec)
```
loss(b, s) = task_state[b,s,0] * (3.0 * sensory + 0.0 * mismatch + 1.0 * energy)
           + task_state[b,s,1] * (0.3 * sensory + 1.0 * mismatch + 1.0 * energy)
```
Three loss terms only: sensory (cross-entropy decoder), mismatch (BCE with MLP head), energy (L1 on r_l23). Per-presentation routing by Markov task_state. All other lambdas zeroed.

### Monitored Metrics
Four accuracies logged every 100 steps: s_acc_rel, s_acc_irr, mm_acc_rel, mm_acc_irr.

---

## 4. Phase 2.x Arc (Alpha_net Causal E/I Gate)

This entire arc preceded the simple-dual-regime experiment. It explored adding a learnable E/I gate (`alpha_net`) to modulate center_exc and som_drive_fb by task_state.

### Phase 1A — Per-sample loss routing (commit `25e7034`)
Added per-sample loss routing by task_state: focused gets higher sensory weight, routine gets mismatch weight. **Result:** PARTIAL — 1/4 preregistered gates passed (m7_focused > +0.03: PASS). m7_routine remained positive (FAIL), |focused−routine| too small (FAIL), m10_amp_routine > 0.9 (FAIL).

### Phase 1B — Additional routing (commit `348c4c9`)
Added sharp/local_disc/pred_suppress routing by task_state. **Result:** PARTIAL — 2/4 gates passed. Same core failure: routine didn't dampen.

### Phase 1C — L2 energy + 3× focused energy (commit `2806101`)
Switched to L2 (quadratic) energy penalty with 3× weight for focused. **Result:** PARTIAL — 2/4 gates. Amplitude shortcut survived: network reduced mean activity without changing tuning shape.

### Phase 2 — Causal E/I gate (commit `4ae515d`)
Added `alpha_net = Linear(3, 2)`: takes (task_state[0:2], pi_pred_raw) → 2×sigmoid → (g_E, g_I). Multiplies center_exc by g_E and som_drive_fb by g_I. Init near-identity (bias=0, weight std=0.01). **Result:** PARTIAL — 2/4 gates. Gate stuck at identity (Δg_E = +0.0038).

### Debugger Finding: Alpha_net Freeze Bug
**Root cause (confirmed, commit `cde523f`):** `stage1_sensory.py` runs `for p in net.parameters(): p.requires_grad_(False)` before Stage 1, which blanket-freezes ALL parameters including alpha_net. Stage 1 then only re-enables L2/3 + PV. `unfreeze_stage2()` did not include alpha_net. `create_stage2_optimizer()` filters by `requires_grad`, silently dropping the frozen alpha_net group. Result: alpha_net received gradients but AdamW never called `.step()` on frozen params. **Fix:** Added explicit `alpha_net.requires_grad_(True)` in `unfreeze_stage2()`.

### Phase 2.4 — Routine shape loss (commit `ef4f102`)
Added `lambda_routine_shape`: penalizes center_exc while rewarding som_drive_fb on routine presentations. `lr_mult_alpha=10`. **With freeze bug (pre-fix):** gate dead, pathway collapse (r_l23=0 for routine). **Without freeze bug (Phase 2.4.1):** energy runaway with λ=2.0 (loss diverged).

### Phase 2.4.2 — Gentler hyperparameters (commit `2077ac7`)
λ_routine_shape=1.0, lr_mult_alpha=5. Bounded dynamics but routine "dampening" was pathway collapse (r_l23→0), not functional dampening (reduced-but-decodable activity).

### Key Conclusion
The alpha_net gate hypothesis was architecturally limited: a scalar E/I gain can't produce per-channel selectivity. It can only scale the overall magnitude of center_exc and som_drive_fb, not reshape the tuning curve. The entire Phase 2.x arc was chasing the wrong mechanism. This led to the simple-dual-regime restart.

---

## 5. Simple-Dual-Regime Experiment

### Network_mm (Fix A — MLP Mismatch Head)

**Config:** `sweep_simple_dual.yaml`. **Commit:** `9d1e954`.
**Checkpoint:** `simple_dual_v2/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt`

Architecture: Single shared V2 with single head_feedback (shared across regimes). use_per_regime_feedback=False.

**Debugger finding (Task #6):** The original linear mismatch head plateaued at val_acc ~0.74. An MLP (Linear(36,64)→ReLU→Linear(64,1)) reached 0.89 on the same input. Fix A replaced the linear head with the MLP and added loss_fn state saving to checkpoints.

**Final metrics (step 5000):**
- s_acc_rel = 0.652, s_acc_irr = 0.651, **Δ_sens = +0.001** (near-zero dissociation)
- mm_acc_rel = 0.694, mm_acc_irr = 0.906, **Δ_mm = +0.212**
- M7 focused δ=10° = +0.334, M7 routine δ=10° = +0.329
- M10 focused = 5.169, M10 routine = 4.776
- FWHM focused = 23.59°, FWHM routine = 23.25°
- Baseline peak = 1.506, Focused peak = 1.546, Routine peak = 1.443

**Preregistered gates (default window, δ=10°):**
- m7_focused = +0.334 > +0.03: **PASS**
- m7_routine = +0.329 < −0.03: **FAIL** (positive, not negative)
- |focused−routine| = 0.005 > 0.06: **FAIL** (near-zero)
- m10_amp_routine = 4.776 < 0.9: **FAIL** (amplification, not dampening)

**Verdict: FAIL.** All three regimes show amplification. No dampening in routine. The shared head_feedback produces a broadband gain at the predicted orientation regardless of task_state.

### Network_both (Fix A + Fix 2 — Per-Regime Head Feedback)

**Config:** `sweep_simple_dual_fix2.yaml`. **Commit:** `fb7a1a1`.
**Checkpoint:** `simple_dual_v3/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt`

Architecture: Single V2 GRU with two feedback heads: head_feedback_focused (16→36), head_feedback_routine (16→36), gated by task_state[0:1] and task_state[1:2]. use_per_regime_feedback=True.

**Final metrics:**
- s_acc_rel oscillated ±0.05 around s_acc_irr
- mm_acc_irr = 0.906, mm_acc_rel = 0.694, Δ_mm = +0.212
- M7 focused δ=10° = +0.342, M7 routine δ=10° = +0.320
- M10 focused = 5.496, M10 routine = 4.327
- FWHM focused = 22.87°, FWHM routine = 19.44°
- Baseline peak = 0.286 (FB off — per-regime heads zero at task_state=[0,0]), Focused peak = 1.627, Routine peak = 1.295

**Preregistered gates:** Same pattern — m7_focused PASS, all others FAIL. Routine still amplifies.

---

## 6. Debugger Investigation — Sensory Dissociation Failure

### Task #8 Findings
1. **Gradient dominance:** 99% of total gradient is aligned with the focused-sensory direction. Routine mismatch gradient is ~100× smaller in magnitude.
2. **V2 IS encoding task_state:** h_v2 cosine similarity between regimes = 0.85 — V2's hidden state differentiates regimes. Not the bottleneck.
3. **Feedback IS differentiated:** 22% norm difference in feedback_signal between focused and routine. Not the bottleneck.
4. **The bottleneck:** A single shared Linear(16→36) head_feedback can only produce small orientation-decoding differential. The 10× loss weight ratio IS applied end-to-end, but both regimes hit the same shared-network ceiling.
5. **Feedback is a volume knob:** Feedback provides +0.37 decoder accuracy improvement via overall MAGNITUDE boost, not via task-conditional tuning shape changes.
6. **Anti-phase trade-off:** s_acc_rel and mm_acc dissociation trade off with conserved sum ~0.25. Improving one degrades the other.

---

## 7. Literature-Grounded Criteria

### Sharpening (Attention-Enhanced Processing)
- **McAdams & Maunsell 1999** (J Neurophysiol): Attention increases gain of V4 neurons, multiplicative scaling of tuning curves. Criterion: gain ≥ +5% at preferred.
- **Martinez-Trujillo & Treue 2004** (Curr Biol): Attention shifts tuning toward attended feature. Criterion: selectivity increase.
- **Reynolds & Heeger 2009** (Neuron): Normalization model of attention — attention acts as an input gain that interacts with normalization. Criterion: narrowed tuning when attention is narrow, broadened when attention is broad.
- Applied criterion: M10 > 1 (amplification) + flank suppression (FWHM narrower) + decoder improvement (M7 > 0) + selectivity increase (OSI up).

### Dampening (Expectation Suppression)
- **Kok et al. 2012** (J Neurosci): Expected stimuli produce lower BOLD in V1 than unexpected. Criterion: mean activity reduced 10–50%.
- **Meyer & Olson 2011** (J Neurophysiol): Repeated/expected stimuli produce lower firing rates in IT cortex. Criterion: stimulus-specific suppression.
- **Alink et al. 2010** (J Neurosci): Expected motion direction produces lower BOLD. Criterion: suppression is content-specific.
- **Summerfield & de Lange 2014** (Trends Cogn Sci): Expectation suppression is stimulus-specific, not a global gain reduction.
- Applied criterion: M10 < 0.9 (dampening) + decoder preserved (s_acc_off ≈ s_acc_on) + stimulus-specific (suppression at expected channel, not global) + redistribution not destruction (variance preserved).

### Applied to Network_mm
- Sharpening criteria: 2/4 met (M10 > 1, M7 > 0). Missing: flank suppression, selectivity increase. FAIL.
- Dampening criteria: 0/4 met (M10 > 4, not < 0.9). FAIL.
- Both regimes show the same amplification pattern. The network converges on **predictive facilitation** — feedback boosts activity at the predicted location regardless of task relevance.

---

## 8. Expected vs. Unexpected Analysis

### Biological Framing
The user rejected the "FB-on vs FB-off" comparison as non-biological (ablation doesn't exist in biology). The correct measurement: **Expected vs. Unexpected stimuli, crossed with Relevant vs. Irrelevant task, all with feedback always on.**

### Key Architectural Finding
V2 GRU does NOT receive oracle — feedback_signal depends on V2's autonomous bottom-up inference, not on the oracle prediction. The "expected vs. unexpected" contrast in the evaluator measures "stim aligned with V2's learned prediction vs. stim misaligned."

### Universal Pattern (All Networks)
- **Expected (offset=0°):** Feedback constructively adds to stim channel. Higher peak activity, narrower FWHM, symmetric population profile.
- **Unexpected (offset=±30°):** Feedback pushes at wrong location (30° away from stim). Asymmetric profile skewed toward prediction direction. Lower peak at stim channel. Broader FWHM.
- This is **predictive facilitation**, not expectation suppression. The network learned that feedback should point at where V2 thinks the stimulus is — when V2 is right, this helps; when V2 is wrong, it distorts.

### Network_mm 2×2 Tables (default window)

**Activity at stim channel (peak r_l23):**

|  | Expected | Unexpected (±30°) | Δ (E−U) |
|---|---|---|---|
| Relevant | 1.546 | 1.309 | +0.237 |
| Irrelevant | 1.443 | 1.222 | +0.221 |

**Decoding accuracy (actual stim, FB on):**

|  | Expected | Unexpected | Δ |
|---|---|---|---|
| Relevant | 0.852 | 0.856 | −0.004 |
| Irrelevant | 0.836 | 0.842 | −0.006 |

**FWHM (°):**

|  | Expected | Unexpected | Δ |
|---|---|---|---|
| Relevant | 23.59 | 25.69 | −2.10 |
| Irrelevant | 23.25 | 25.52 | −2.27 |

**Interaction terms (Relevant − Irrelevant):** Activity +0.016, Decoding +0.002, FWHM +0.17. All near-zero. **No regime × expectation interaction.**

---

## 9. Four Parallel Fix Experiments

All trained on remote GPU, 5000 Stage 2 steps each. Commit `8e13d69`.

### Fix 1 — Dual V2 (`sweep_fix1_dual_v2.yaml`)

**Change:** `use_dual_v2: true`. Two independent V2 GRUs (v2_focused, v2_routine), each with own GRU/head_mu/head_pi/head_feedback. Selected by task_state at runtime. Everything else (L4, PV, L2/3, SOM) shared.

**Hypothesis:** The combined network fails because V2's shared GRU is dominated by the focused-amplification gradient. Two independent V2 modules reproduce the gradient isolation of individual networks architecturally.

**Final metrics (step 5000):**
- s_acc_rel = 0.669, s_acc_irr = 0.494, **Δ_sens = +0.175** (largest dissociation)
- mm_acc_rel = 0.657, mm_acc_irr = 0.898, **Δ_mm = +0.241**

**Issue:** s_acc_irr collapsed to 0.494 (near chance). This is architecturally "two networks in a trenchcoat" — the V2 modules can't share representations. Not a meaningful single-network result.

**Evaluation BLOCKED:** `oracle_mode + use_dual_v2` bug. The oracle_mode branch (`network.py:150`) passes `state.h_v2[B,32]` (packed dual hidden states) to `self.v2.gru` which expects `h_v2[B,16]`. Needs code fix to split h_v2 before passing to the correct V2 module(s) in oracle_mode.

### Fix 2 — Gradient Isolation (`sweep_fix2_grad_isolation.yaml`)

**Change:** `gradient_isolation: true`, `isolation_period: 100`. Alternates between focused-only and routine-only task_state overrides every 100 steps. Stimulus HMM runs normally; only task_state is overridden.

**Hypothesis:** Alternating regime-only batches gives each regime an uncontested gradient window, like training individual networks.

**Final metrics (step 5000, using step 4900 for _rel metrics):**
- s_acc_rel = 0.666*, s_acc_irr = 0.635, **Δ_sens = +0.031** (closest to Network_mm)
- mm_acc_rel = 0.418*, mm_acc_irr = 0.905, **Δ_mm = +0.487** (largest — but because mm_acc_rel collapsed)

**2×2 tables:**

Activity: Rel Exp=1.549, Unexp=1.283, Δ=+0.266 | Irr Exp=1.446, Unexp=1.191, Δ=+0.255
Decoding: Rel Exp=0.836, Unexp=0.842, Δ=−0.006 | Irr Exp=0.816, Unexp=0.830, Δ=−0.014
FWHM: Rel Exp=23.11, Unexp=26.03, Δ=−2.93 | Irr Exp=22.89, Unexp=25.67, Δ=−2.78

**Interpretation:** Sensory nearly matched (good), but mm_acc_rel degraded badly — alternating batches caused regime forgetting on mismatch detection. Largest FWHM broadening for unexpected stimuli.

### Fix 3 — Expected-Suppress Loss (`sweep_fix3_pred_err.yaml`)

**Change:** `lambda_expected_suppress: 2.0`. New loss term penalizes mean |r_l23| ONLY on routine presentations where mismatch_label=0 (stimulus matched prediction). Direct gradient toward Kok-style expectation suppression.

**Hypothesis:** The routine regime needs a direct gradient signal to suppress activity on expected stimuli. λ=2.0 matches the individual dampening net's energy strength.

**Final metrics (step 5000):**
- s_acc_rel = 0.672, s_acc_irr = 0.608, **Δ_sens = +0.064**
- mm_acc_rel = 0.545, mm_acc_irr = 0.893, **Δ_mm = +0.348**

**2×2 tables:**

Activity: Rel Exp=1.344, Unexp=1.169, Δ=+0.175 | Irr Exp=1.193, Unexp=1.010, Δ=+0.183
Decoding: Rel Exp=0.891, Unexp=0.893, Δ=−0.002 | Irr Exp=0.863, Unexp=0.875, Δ=−0.012
FWHM: Rel Exp=23.04, Unexp=24.92, Δ=−1.88 | Irr Exp=22.21, Unexp=23.93, Δ=−1.72

**Interpretation:** Lower overall amplitudes (expected-suppress working), but expected > unexpected activity persists in both regimes. No sign of dampening. mm_acc_rel weakened to 0.545.

### Fix 4 — Match Individual Recipe (`sweep_fix4_match_recipe.yaml`)

**Change:** routine sensory: 0.3→0.0, routine energy: 1.0→2.0. Matches the exact loss recipe that produced strong dampening in the individual three-regime network.

**Hypothesis:** The combined network's routine never dampened because residual sensory loss (0.3) pulled against energy. Zeroing routine sensory and doubling energy should match the individual dampening result.

**Final metrics (step 5000):**
- s_acc_rel = 0.654, s_acc_irr = 0.600, **Δ_sens = +0.054**
- mm_acc_rel = 0.571, mm_acc_irr = 0.890, **Δ_mm = +0.319**

**2×2 tables:**

Activity: Rel Exp=1.462, Unexp=1.268, Δ=+0.194 | Irr Exp=1.282, Unexp=1.101, Δ=+0.181
Decoding: Rel Exp=0.875, Unexp=0.863, **Δ=+0.012** | Irr Exp=0.832, Unexp=0.830, Δ=+0.002
FWHM: Rel Exp=23.78, Unexp=25.47, Δ=−1.69 | Irr Exp=23.13, Unexp=25.08, Δ=−1.95

**Interpretation:** Only fix with a positive decoding Δ in relevant (+0.012), but marginal. Expected > unexpected activity in both regimes. No dampening. The individual dampening recipe does NOT transfer to the combined network.

### Side-by-Side Comparison

| Metric | Network_mm (ref) | Fix 2 | Fix 3 | Fix 4 |
|---|---|---|---|---|
| Activity Δ, Relevant | +0.237 | +0.266 | +0.175 | +0.194 |
| Activity Δ, Irrelevant | +0.221 | +0.255 | +0.183 | +0.181 |
| Decoding Δ, Relevant | −0.004 | −0.006 | −0.002 | +0.012 |
| Decoding Δ, Irrelevant | −0.006 | −0.014 | −0.012 | +0.002 |
| FWHM Δ, Relevant | −2.10 | −2.93 | −1.88 | −1.69 |
| FWHM Δ, Irrelevant | −2.27 | −2.78 | −1.72 | −1.95 |

### Conclusion
No fix produces dampening (expected < unexpected activity) or sharpening (flank suppression, selectivity increase). All show predictive facilitation: feedback boosts at the predicted location regardless of regime. The focused-regime gradient dominates shared parameters even with loss, schedule, and architectural interventions.

---

## 10. Key Bugs Found and Fixed

### Alpha_net Freeze Bug (commit `cde523f`)
`stage1_sensory.py` blanket-froze all network parameters before Stage 1. `unfreeze_stage2()` didn't re-enable `alpha_net`. AdamW silently dropped the frozen param group. **Fix:** Added explicit `alpha_net.requires_grad_(True)` in `unfreeze_stage2()`.

### Linear Mismatch Head Capacity (commit `9d1e954`)
Linear mismatch head (Linear(36→1)) plateaued at val_acc ~0.74. **Fix:** Replaced with 2-layer MLP (Linear(36,64)→ReLU→Linear(64,1)) reaching ~0.89.

### Checkpoint Save Bug (commit `9d1e954`)
`mismatch_head` (and other loss_fn submodules) were not saved in checkpoints. **Fix:** Added `loss_fn.state_dict()` as `'loss_heads'` key in checkpoint.

### SIGHUP Vulnerability (commit `69dee73`)
`run_sweep.sh` launched training inside tmux but bare `bash -c` inside pty died silently on SIGHUP when the SSH session disconnected. **Fix:** Added `setsid` self-re-exec pattern so the training process survives terminal hangup.

### Per-Presentation Loss Gating (commit `194a3c7`)
Refactored `losses.py` from per-sample to per-presentation loss routing. `task_routing` config maps regime to per-term multipliers. Applied per-presentation rather than per-batch to support Markov task_state switching within sequences.

### Dual V2 + Oracle Mode Bug (discovered 2026-04-12, UNFIXED)
When `use_dual_v2=True` and `oracle_mode=True`, the oracle_mode branch passes `state.h_v2[B,2*H=32]` to `self.v2.gru` (expects H=16). **Root cause:** oracle_mode branch doesn't handle dual V2 state splitting. Blocks all evaluation for Fix 1. Needs: split h_v2, run both V2 modules, recombine feedback.

---

## 11. Current State and Open Questions

### What Works
- **V2 learns broadband gain:** V2 reliably learns to produce a peaked feedback signal centered on its best estimate of the stimulus orientation. This boosts r_l23 at the predicted channel and improves decoder accuracy (M7 > +0.3 in all networks).
- **Mismatch detection works:** The MLP mismatch head achieves ~0.90 binary accuracy on expected vs. deviant classification in the routine regime.
- **Task_state gating is real but weak:** V2 hidden state differentiates regimes (cos=0.85). Feedback norms differ ~22% between regimes. But this doesn't translate to qualitatively different tuning shapes.

### What Doesn't Work
- **No dampening in any network:** No checkpoint shows M10 < 1.0, expected < unexpected activity, or mean suppression in routine. Every network amplifies in both regimes.
- **No sharpening beyond baseline:** No checkpoint shows flank suppression or increased selectivity beyond what's already present in the FB-off baseline. FWHM narrows by ~3° relative to FB-off, but this is uniform across regimes.
- **Feedback is a volume knob, not a shape operator:** The single Linear(16→36) head produces a gain bump at the predicted orientation. Its positive/negative split via Dale's law creates center_exc/som_drive_fb, but both are peaked at the same location — the SOM pathway is negligible (r_som < 0.001 in Network_mm).

### The Core Problem
The focused-regime gradient (3.0× sensory) dominates the V2 optimization landscape. Even with per-regime heads, gradient isolation, or explicit suppression losses, the network converges on the same strategy: produce a broadband gain bump at the predicted orientation. This is the path of least resistance for maximizing sensory accuracy, and the routine-regime gradient (0.3× sensory + 1.0× mismatch) is too weak to push the shared L2/3 recurrent dynamics into a qualitatively different regime.

### Open Questions
1. **Is a different architecture needed?** The shared L4→PV→L2/3→SOM pathway may be too constrained for one linear feedback head to produce both amplification and suppression. A non-shared pathway (e.g., separate SOM populations per regime) might be required.
2. **Is the loss formulation fundamentally wrong?** The Kok-style dampening criterion (mean suppression) may not be achievable with a sensory+energy loss alone — the network has no incentive to suppress expected stimuli specifically (only to reduce total energy).
3. **Does gradient isolation need longer training?** Fix 2 showed the closest sensory match (Δ_sens=+0.031) but regime forgetting on mismatch. Longer training with periodic consolidation might help.
4. **Is the oracle_mode the right evaluation paradigm?** Since V2 GRU doesn't receive the oracle, the "expected vs. unexpected" contrast depends on how accurately V2 tracks the true stimulus — which may differ from the oracle prediction.

---

## 12. All Checkpoints

| Tag | Path (remote) | Config | Architecture | Key Metrics |
|---|---|---|---|---|
| Network_mm | `simple_dual_v2/.../checkpoint.pt` | `sweep_simple_dual.yaml` | Single V2, shared head_feedback | s_acc_rel=0.652, s_acc_irr=0.651, mm_acc_irr=0.906, M10_foc=5.17, FWHM_foc=23.59° |
| Network_both | `simple_dual_v3/.../checkpoint.pt` | `sweep_simple_dual_fix2.yaml` | Single V2, per-regime head_feedback | mm_acc_irr=0.906, M10_foc=5.50, M10_rou=4.33, FWHM_rou=19.44° |
| Fix 1 | `fix1_dual_v2/.../checkpoint.pt` | `sweep_fix1_dual_v2.yaml` | Dual V2 (v2_focused + v2_routine) | s_acc_rel=0.669, s_acc_irr=0.494, Δ_sens=+0.175. **Eval blocked by oracle_mode bug.** |
| Fix 2 | `fix2_grad_iso/.../checkpoint.pt` | `sweep_fix2_grad_isolation.yaml` | Single V2, gradient isolation (period=100) | s_acc_rel=0.666, s_acc_irr=0.635, mm_acc_irr=0.905, mm_acc_rel=0.418 |
| Fix 3 | `fix3_pred_err/.../checkpoint.pt` | `sweep_fix3_pred_err.yaml` | Single V2, lambda_expected_suppress=2.0 | s_acc_rel=0.672, s_acc_irr=0.608, mm_acc_irr=0.893 |
| Fix 4 | `fix4_recipe/.../checkpoint.pt` | `sweep_fix4_match_recipe.yaml` | Single V2, routine sens=0 energy=2× | s_acc_rel=0.654, s_acc_irr=0.600, mm_acc_irr=0.890 |

All checkpoints at: `/home/vishnu/neuroips/{tag}/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt`

---

## 13. Rescue chain (failed-dual-regime-experiments branch)

**Branch:** `failed-dual-regime-experiments` (analysis figures + updated docs on `dampening-analysis`).

After training the baseline single-network-dual-regime checkpoints (Section 5), four architectural rescues — **R1+R2, R3, R4, R5** — were attempted to recover the preregistered task-state-selective sharpening AND dampening from one architecture. Each rescue is gated by config flags so the baseline is unchanged when the flags are absent.

| Rescue | Config flags added | Mechanism in one line |
|---|---|---|
| R1 | `lambda_expected_suppress > 0` (loss-only) | Subtract a feature-specific predicted-orientation profile from L2/3 activity. |
| R2 | `use_precision_gating: true` | Scale feedback by `pi_pred / pi_max` so V2-uncertain feedback is attenuated. |
| R3 | `use_vip: true` | Add VIPRing population + structured center-surround SOM disinhibition kernel. |
| R4 | `use_deep_template: true`, `use_error_mismatch: true` | DeepTemplate leaky integrator for the V1 expectation template; mismatch head reads `r_error = relu(r_l23 - r_template)`. |
| R5 | `use_shape_matched_prediction: true` | Project `q_pred` through a fixed Stage-1-calibrated buffer `T_stage1` before subtractive suppression. |

### Headline result

The rescues produce increasingly strong expectation suppression (Expected L2/3 activity < Unexpected) but the dampening is **task-state-invariant**, not regime-selective. **R4 (DeepTemplate + error-mismatch)** shows the cleanest Richter (2018) preserved-shape dampening pattern (peak −15%, total −19%, FWHM matched within 1.5° between expected and unexpected). Main visual: `docs/figures/tuning_ring_recentered_r4.png`.

The preregistered BOTH-regime criterion (focused → Kok sharpening, routine → Richter dampening from one checkpoint) is **not** met by any of the four rescues. The dampening signature appears in both Relevant and Irrelevant task_state with similar magnitude — task_state does not gate the representational mode.

### Branch / artefact map

- **Baseline (Section 5):** branch `single-network-dual-regime`, original simple_dual checkpoints.
- **Rescues R1+R2, R3, R4, R5:** branch `failed-dual-regime-experiments`. Configs: `config/sweep/sweep_rescue_{1_2,3,4,5}.yaml`.
- **Re-centered analysis + figures:** branch `dampening-analysis`. Scripts: `scripts/plot_tuning_ring_extended.py`, `scripts/plot_tuning_ring_heatmap.py`, `scripts/plot_tuning_exp_vs_unexp.py`. This branch now also carries raw/delta/baseline surfaces and paired-state branch-counterfactual analysis. Figures in `docs/figures/`.

### Where to read the details

`docs/rescues_1_to_4_summary.md` — full per-rescue rationale, metrics, and the 2026-04-13 update section that corrects the earlier "subtractive predictive coding" interpretation (which was based on non-re-centered, bin-counted FWHM and exaggerated the expected/unexpected FWHM gap).

`RESULTS.md` § 9 — cross-checkpoint summary table and headline take-away.

### 2026-04-14 aligned pure-R1+2 branch-counterfactual follow-up

The current `dampening-analysis` branch also contains a targeted follow-up on
the aligned pure-R1+2 checkpoint
`r12_fb24_sharp_050_width_075_rec11_aligned`. This is deliberately narrower
than the cross-checkpoint rescue summary: it does **not** re-rank R1–R4, and
it does not broaden into a full all-rescue reanalysis.

The added analysis surfaces are:
- `raw`: the legacy late-ON `r_l23[t_readout]`
- `delta`: the evoked response `r_l23[t_readout] - r_l23[t_isi_last]`
- `baseline`: the inherited pre-probe state `r_l23[t_isi_last]`
- `branch_counterfactual`: branch the exact same frozen pre-probe recurrent
  state into an expected probe and an unexpected probe

On the paired-state **Relevant** surface for the aligned checkpoint:
- `baseline` is identical after the centering fix
  (`peak=0.375691`, `FWHM=39.389691°` for both expected and unexpected).
- `raw` shows expected lower and slightly narrower than unexpected
  (`peak 0.449558 vs 0.507532`, `FWHM 33.237447° vs 33.515513°`).
- `delta` shows expected much lower and much narrower than unexpected
  (`peak 0.072438 vs 0.491614`, `FWHM 27.581737° vs 45.064195°`).

The baseline-centering fix is analysis-only. Before the fix, the
branch-counterfactual baseline path returned the same pre-probe tensor for
both branches but re-centered them by different branch probe channels, which
made identical baselines look artificially opposite. Baseline mode now uses
the shared predicted/expected channel for both branches.
