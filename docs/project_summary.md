# Project Summary: Laminar V1-V2 Expectation Suppression Model

**Last updated:** 2026-04-24
**Branch:** `dampening-analysis` (active for analysis); `single-network-dual-regime` (training baselines).
**Default checkpoint for ex-vs-unex / dampening-vs-sharpening analysis (set 2026-04-17):** R1+R2 simple_dual emergent_seed42 (`results/simple_dual/emergent_seed42/checkpoint.pt` on the remote). Network_mm / Network_both remain valid for the per-regime feedback question but are no longer the default model for ex-vs-unex analysis.
**R1+R2 regime characterisation (2026-04-22, Tasks #15–#26):** **hybrid, not single-regime.** On the paired-fork paradigm (constructive probe, bit-identical pre-probe state), R1+R2 shows **decoder-robust sharpening** in the decoding sign (Δdec_A=+0.387, Δdec_B=+0.085, Δdec_C=+0.125 on the NEW eval; all signs positive). FWHM behaviour on the paired-fork is paradigm-internally split: the NEW eval has ex FWHM **narrower** than unex (Δ = −1.33°), but the 4-condition `paradigm_readout` paired HMM fork has ex FWHM **wider** than unex (+0.9° to +2.0°) — same paradigm family, different cue/task-state combinations. On observational paradigms (matched_3row_ring, matched_hmm_ring_sequence with --tight-expected, v2_confidence_dissection, plus M3R and VCD with focused+march cue), R1+R2 shows **decoder-robust dampening** in the decoding sign — Δdec_C ∈ {−0.029, −0.063, −0.070, −0.018, −0.013} on those 5 rows. The sign difference between paired-fork (sharpening on decoding) and observational (dampening on decoding) is real, not a decoder artefact. See RESULTS.md §11–§14 for the full evidence.
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
- **Re-centered analysis + figures:** branch `dampening-analysis`. Scripts: `scripts/plot_tuning_ring_extended.py`, `scripts/plot_tuning_ring_heatmap.py`, `scripts/plot_tuning_exp_vs_unexp.py`. Figures in `docs/figures/`.

### Where to read the details

`docs/rescues_1_to_4_summary.md` — full per-rescue rationale, metrics, and the 2026-04-13 update section that corrects the earlier "subtractive predictive coding" interpretation (which was based on non-re-centered, bin-counted FWHM and exaggerated the expected/unexpected FWHM gap). The 2026-04-17 update section adds a Decoder A artefact note and sets R1+R2 as the canonical default checkpoint for ex-vs-unex analysis.

`RESULTS.md` § 9 — cross-checkpoint summary table and headline take-away.

`RESULTS.md` § 10 — R1+R2 paired ex/unex eval under Decoder C (2026-04-17).

---

## 14. R1+R2 paired ex/unex eval — Decoder C (2026-04-17)

**Branch:** `dampening-analysis`. **Checkpoint:** R1+R2 simple_dual emergent_seed42 (now the canonical default).

### Decoder A artefact

Earlier sections of this doc (and `RESULTS.md` § 9) report Δdec(ex−unex) numbers measured with **Decoder A** — the linear sensory readout trained alongside the network in Stage 2 and frozen thereafter. On R1+R2, the matched-probe-3pass **Δdec(ex−unex) = +0.32 under Decoder A collapses to Δ ≈ +0.04 (within per-fold noise) under Decoder B** (5-fold nearest-centroid CV on the same `r_l23` activations). Root cause: Decoder A's fixed templates, trained on the natural-march distribution, are out-of-distribution for the synthetic Pass B compound bumps used in matched-probe-3pass. Network_mm / Network_both / HMM Expected-vs-Unexpected numbers that used Decoder A are decoder-dependent and should be re-checked under Decoder C before being used as evidence for or against Kok / Richter / dampening signatures.

### Decoder C (preferred decoder for ex-vs-unex analyses)

Standalone `Linear(36, 36)` (bias on) trained on 100k synthetic orientation-bump patterns: 50k single-orientation σ=3 ch with amplitudes ∈ [0.1, 2.0]; 50k multi-orientation K∈{2,3} with strictly-max amplitude as the label; Gaussian noise σ=0.02. Adam lr=1e-3, batch 256, ≤30 epochs, early-stop patience 3, seed 42. Saved at `checkpoints/decoder_c.pt`. Held-out synthetic accuracy 0.81 (single 0.98 / multi 0.65); real-network natural-HMM R1+R2 accuracy 0.66 non-amb / 0.53 all. Source: `scripts/train_decoder_c.py`.

### Paired ex/unex eval design (Tasks #12/#13)

12 N values (4..15) × 200 trials/N = 2400 paired ex/unex trials, run on R1+R2. Per-trial RNG seed = `42 + trial_idx` (independent of N → bit-identical pre-probe march for the same trial_idx across N values). Random S ∈ [0°, 180°), D ∈ [25°, 90°], CW/CCW 50/50 per trial. `task_state = [1, 0]` (focused) throughout, contrast 1.0. Cue at the expected-next orientation in **both** branches (so unex cue is "wrong" by D degrees). Pre-probe state shared across branches — only the probe-ON window diverges. Readout: probe-ON window steps `[9:11]` mean-pooled, followed by per-trial roll-to-center on the true probe channel (peak at ch18) with linear-interpolation FWHM (same convention as commit `ce1b34e`).

### Pooled across N (n=2400 paired trials)

| Metric | Expected | Unexpected | Δ (ex − unex) |
|---|---:|---:|---:|
| Decoder C top-1 accuracy | 0.707 ± 0.009 | 0.581 ± 0.010 | +0.125 |
| Net L2/3 (sum 36 channels) | 4.99 ± 0.01 | 6.13 ± 0.02 | −1.15 |
| Peak at true channel (re-centered ch18) | 0.773 ± 0.003 | 0.626 ± 0.004 | +0.147 |
| FWHM (linear-interp half-max) | 28.4° ± 0.10 | 29.8° ± 0.19 | −1.33° |

All four signs hold at every N from 4 to 15. Pre-probe state is bit-identical across branches (`pre_probe_max_abs_diff = 0.00e+00`). All 2400 trials produced valid FWHM crossings in both branches. In plain language: expected trials show **lower net L2/3 activity, higher peak at the stimulus channel, narrower tuning, and higher decoding accuracy** than unexpected trials.

### Interpretation framing (literal)

- **Operational dampening (lower activity AND lower decoding on expected):** Passes on net L2/3 (expected lower) but fails on decoding (expected higher), peak (expected higher), and FWHM (expected narrower).
- **Kok 2012 sharpening (narrower tuning, higher peak, better decoding, lower total activity on expected):** Matches on all four axes.
- **Richter 2018 preserved-shape dampening (lower peak, preserved FWHM, preserved decoding on expected):** Does not match — the peak goes the wrong direction.

No mechanism interpretation beyond these literal comparisons.

### Reproducibility

- Eval script: `scripts/eval_ex_vs_unex_decC.py`.
- Decoder training: `scripts/train_decoder_c.py` → `checkpoints/decoder_c.pt`.
- Result JSON: `results/eval_ex_vs_unex_decC.json` (per-N entries + pooled, including `peak_at_stim_*`, `fwhm_deg_*` with `n_valid` counts, and `delta_*` for each metric).
- Figure: `docs/figures/eval_ex_vs_unex_decC.png` (4-panel: dec_acc, net_L2/3, peak_at_stim, FWHM vs N).
- Run log: `logs/eval_ex_vs_unex_decC_t13.log`.
- See `ARCHITECTURE.md` § "Decoders" for the full A/B/C decoder taxonomy and `docs/rescues_1_to_4_summary.md` § "Update (2026-04-17)" for the Decoder A artefact note.

---

## 15. Cross-decoder comprehensive matrix (Task #26)

**Purpose.** All ex-vs-unex Δ values collected on R1+R2 and the four legacy reference networks (a1, b1, c1, e1), re-evaluated through the same forward pass under Dec A, Dec B and Dec C. This is the sign-agreement audit for every decoder-dependent claim in the doc set.

**Source:** `results/cross_decoder_comprehensive.json` and its markdown companion `results/cross_decoder_comprehensive.md`.

**Coverage.** 17 rows — HMM C1/C2/C3/C4 on R1+R2 (4); HMM C1 on legacy a1/b1/c1/e1 (4); the paired-fork assay on R1+R2 (1, the NEW eval); four observational assays on R1+R2 — M3R, HMS, HMS-T, P3P, VCD (5); three of those observational assays re-run with `focused + march cue` modification (3).

**Per-decoder profile (2026-04-24, 17-row matrix).**

| Decoder | n rows | mean \|Δ\| | max \|Δ\| | agrees with row-majority sign (ABC) | rows where this decoder is the single outlier (ABC) |
|---|---:|---:|---:|---:|---:|
| A | 17 | 0.2056 | 0.3871 | 17 / 17 | 0 |
| A′ (R1+R2 only) | 13 | 0.1902 | 0.3902 | 13 / 13 (matches Dec A sign on all 13) | — |
| B | 17 | 0.0485 | 0.1434 | 15 / 17 | 2 (HMM C2 / C4 on R1+R2) |
| C | 17 | 0.0416 | 0.1254 | 13 / 17 | 4 (c1 / e1 legacy; HMS native; HMS-T modified) |
| D-raw | 17 | 0.0520 | 0.2308 | 10 / 17 | — (not part of ABC triple) |
| D-shape | 17 | 0.0585 | 0.1656 | 12 / 17 | — |
| E | 17 | 0.1934 | 0.4359 | 13 R1+R2 sign-match Dec A; 2 legacy sign-flips (a1 / b1 HMM C1) | — |

11 of 17 rows are ABC all-agree in the 2026-04-24 rerun. Dec A amplifies
effects the most; Dec C is the most conservative. See RESULTS.md §11 for
the full 7-column matrix and per-row flags.

**Dec A → Dec A′ swap (2026-04-23).** Dec A is trained jointly with L2/3
during Stage 1 (moving target — L2/3 changes each step). Dec A′ is a fresh
`Linear(36, 36)` retrained for 5000 Adam steps on `r_l23` streamed through
the fully-trained, **frozen** R1+R2 network with **50/50** focused/routine
task_state (stable target; 10k HMM top-1 = 0.5486 vs Dec A 0.5413;
`frac_same_pred(A, A′) = 0.820`). Replacing Dec A with Dec A′ on the 13
R1+R2 rows produces **zero Δ-sign flips**; `|Δ_A′ − Δ_A|` ≤ 0.094 with
median 0.025. The Dec A training-schedule concern does not materially
change the 13-row dampening-vs-sharpening pattern. Sources:
`checkpoints/decoder_a_prime.pt`, `results/decoder_a_prime_training.json`,
`results/decoder_a_prime_stratified_eval.json`,
`results/cross_decoder_comprehensive_decAprime.{json,md}`,
`results/cross_decoder_comprehensive_decAprime_diff.{json,md}`.

**Dec D (2026-04-24, FB-ON paired-fork neutral, per-ckpt; 5k variant).** Two variants:
D-raw on raw `r_l23` and D-shape on `r_l23 / (r_l23.sum(1) + 1e-8)`. Paired-fork
training with normal feedback (`feedback_scale = 1.0`), balanced per (target_ch × branch)
at 900 train + 100 val per cell (72 k total) per ckpt. Adam lr=1e-3 wd=1e-4 seed=42,
early-stop patience 3 max 30 epochs. Headline signature: **row 12 (HMS-T native on R1+R2)**
gives Δ_D-shape = +0.166 while Δ_A = −0.303 — a Kok-framework "amplitude dampens,
orientation shape sharpens" co-occurrence; only row in the 17-row matrix with this
divergence at material magnitude. Artefacts:
`checkpoints/decoder_d_fbON_neutral_{raw,shape}_{r1r2,a1,b1,c1,e1}.pt`,
`results/cross_decoder_comprehensive_withD_fbON.{json,md}`,
`results/decoder_d_fbON_all_eval.json`.

**Dec D 20k variant (2026-04-27, Task #8).** Same paired-fork balanced training
data but 20 000 Adam steps lr=1e-3 wd=0 (no early stopping, no regularization)
— matches Dec A′ 20k regime exactly. Per legacy net (a1 / b1 / c1 / e1). Used as
the disambiguation control for the 20k Dec A′ Δ on a1 / b1 (which read +0.21 / +0.18,
opposite to Dec A): Dec D 20k (balanced ex+unex training, no natural-HMM prior bias
to exploit) reads Δ_ex_unex = −0.024 / −0.046 (raw) / −0.052 / −0.044 (shape) on
a1 / b1, agreeing with Dec A's small-dampening direction. The 20k Dec A′ positive Δ
was prior-bias overfitting at large ||W||. On c1 / e1 Dec D 20k reads positive
Δ (raw +0.069 / +0.040; shape +0.084 / +0.067), agreeing with Dec A's small
sharpening. Final classification reconfirms Section 5. Artefacts:
`checkpoints/decoder_d_20k_{raw,shape}_{a1,b1,c1,e1}.pt`,
`results/decoder_d_20k_training_{net}.json`,
`results/decoder_d_20k_raw_stratified_eval_{net}.json`,
`results/task8_decD_20k_legacy/{net}_C1.json`. Training script:
`scripts/train_decoder_d_20k_adam.py`.

**Dec E (2026-04-24, Dec-A-spec post-Stage-2 retrain, per-ckpt; 2026-04-25 retraction).**
Same arch as Dec A (`Linear(36, 36)+bias`), same lr=1e-3 no-weight-decay,
seed 42, 5000 gradient steps, trained after Stage 2 on the natural HMM
stream with the HMM's own stochastic task_state. On R1+R2 Dec E is
effectively isomorphic to Dec A′ (`frac_same_pred(A′, E) = 0.9722`; both
0.547 vs Dec A 0.541 top-1). On a1 / b1 Dec E caps at top-1 ≈ 0.35 vs
Dec A 0.59. The earlier "real dissociation" reading has been retracted:
Debugger Task #5 (`/tmp/debug_dec_a_advantage_report.md`) and Coder Task #6
Part A jointly show this is an Adam @ 5 000-step optimisation-insufficiency
artefact on dampened frozen L2/3, not a representational disagreement. At
20 000 Adam steps Dec A′ reaches 0.6709 on a1 (+30.5 pp vs 5k 0.3659; +8.0 pp
above Dec A original) and 0.6625 on b1 (+30.6 pp vs 5k 0.3562; +7.9 pp
above Dec A); on r1r2 Dec A′ at 20k = 0.5729 (stable, +2.4 pp vs 5k 0.5486)
because r1r2's r_l23 has sharper per-orientation signal so Adam already
saturates at 5k. Stage-1 co-training of L2/3 + decoder shapes Dec A's
small-||W||=82.5 solution (H1 confirmed) but does NOT prevent retrains
from recovering. The Δ_E sign flags on a1 / b1 HMM C1 are artefacts of
the same under-training and not decoder disagreements. Dec E ckpts for
a1 / b1 / c1 are **step-4000-recovered** due to an unrelated post-training
legacy-ckpt `loss_heads` bug (fix landed in trainer); orthogonal to this
retraction. Artefacts:
`checkpoints/decoder_e_{r1r2,a1,b1,c1,e1}.pt`,
`results/decoder_e_training_{net}.json`,
`results/decoder_e_stratified_eval_{net}.json`,
`results/cross_decoder_comprehensive_with_all_decoders.{json,md}`,
`checkpoints/decoder_a_prime_20k_{r1r2,a1,b1}.pt`,
`results/decoder_a_prime_20k_stratified_eval_{net}.json`.

Full taxonomy now in `ARCHITECTURE.md` § "Decoders" (six rows: A / A′ / B / C
/ D-raw / D-shape / E).

---

## 16. Paired HMM fork paradigm × readout analysis (Tasks #19–#22)

**Design.** Shared pre-probe HMM march, then branch expected vs unexpected at the probe channel; bit-identical pre-probe state across branches. 4 conditions crossing `task_state` (focused vs routine) × cue (HMM-supplied vs neutral zero cue). Readout at probe-ON steps `[9, 11]`, re-centered per-trial on the true probe channel (peak at ch 18).

**Decoder-C Δ across 4 conditions (ex − unex, paired):**

| Condition | Δ decC (ex − unex) |
|---|---:|
| C1: focused + HMM cue | +0.088 |
| C2: routine + HMM cue | +0.013 |
| C3: focused + zero cue | +0.045 |
| C4: routine + zero cue | +0.041 |

All four Δ are positive → **sharpening signature persists on the paired-fork paradigm across every task-state × cue combination under Dec C.**

**FWHM-sign reversal between two paired-fork variants.** On the 4-condition `paradigm_readout` paired HMM fork, ex FWHM is **wider** than unex by +0.90° (C1) to +2.00° (C2) across all four conditions. On the NEW paired-march eval (§ 14 / RESULTS.md § 10), the same R1+R2 checkpoint gives ex FWHM (28.44°) **narrower** than unex (29.77°): Δ = −1.33°. The peak sign (ex > unex) and the net-L2/3 sign (ex < unex) are consistent across both paired-fork variants; only FWHM flips between them. The decoding sign is also consistent (Δdec_C > 0 on all of NEW + C1–C4). The decoding-sign comparison against the observational paradigms (which give Δdec_C < 0 on M3R / HMS-T / VCD / their modified variants) is the separate paradigm effect noted in §15 / §18.

**Adjacent-channel signed-offset curve (Task #19).** Per-trial re-centering on the march direction (CW/CCW flipped to a single convention) reveals **flank asymmetry on expected trials only**: on expected, the `+k` flank (leading edge of the march) is lower than the `−k` flank by ≈ 0.06–0.10 for k ∈ {1, 2, 3} (Decoder C readout); on unexpected trials, both flanks are near-symmetric. **UNTESTED MECHANISM HYPOTHESIS** (no isolating experiment yet): a march-direction-aligned pre-probe deformation of the population profile, possibly mediated by V2 feedback subtracting the predicted-next orientation, would produce this signature — but neither feedback ablation nor direct inspection of feedback weights against the asymmetry pattern has been run. The asymmetry itself is the empirical observation. Source: `results/eval_ex_vs_unex_decC_adjacent.json`. Full signed-offset table and per-flank diagnostics in RESULTS.md §12.

**Reproducibility.** `scripts/eval_r1r2_paradigm_readout.py` (4-condition sweep), `scripts/eval_ex_vs_unex_decC_adjacent.py` (adjacent-channel analysis). Result JSONs: `results/r1r2_paradigm_readout.json`, `results/r1r2_paired_hmm_fork.json`, `results/eval_ex_vs_unex_decC_adjacent.json`.

---

## 17. Legacy reference networks (Tasks #23–#24)

Four reference checkpoints from Section 5 re-evaluated under the three-decoder protocol on HMM C1 (focused + HMM cue) for ex-vs-unex Δ, using a `MechanismType` enum shim in `src/config.py` and `torch.load(..., weights_only=False)` / `strict=False` for the pickle-compat load:

| Network | Section-5 regime | Δ_A | Δ_B | Δ_C | sign-agreement |
|---|---|---:|---:|---:|---|
| a1 | baseline dampening (wider on ex) | −0.022 | 0.000 | −0.009 | **B outlier** (B = 0.0; A and C both < 0) |
| b1 | baseline dampening (wider on ex) | −0.032 | −0.015 | −0.023 | ALL agree |
| c1 | transition / mixed | +0.187 | +0.037 | −0.007 | C outlier |
| e1 | sharpening (best) | +0.213 | +0.051 | +0.011 | ALL agree |

Dec A amplifies the sharpening-vs-dampening gap (−0.03 → +0.21 across the four networks), but all three decoders agree on sign in 2 of 4 rows (b1, e1). Under the looser A-vs-C-only check, 3 of 4 rows agree (a1 + b1 + e1; only c1 disagrees). The Section-5 regime classification (a1/b1 dampening, e1 sharpening) replicates under Dec C in those same 3 rows. See RESULTS.md §13 for checkpoint-loading shim details.

---

## 18. Robust findings summary (updated 2026-04-24)

**Decoder-robust sharpening signature (A / B / C all > 0, per 2026-04-24 matrix).**

1. NEW paired-march eval on R1+R2: Δ_A = +0.387, Δ_B = +0.085, Δ_C = +0.125.
2. HMM C1 (focused + HMM cue) on R1+R2: Δ_A = +0.315, Δ_B = +0.020, Δ_C = +0.066.
3. HMM C3 (focused + zero cue) on R1+R2: Δ_A = +0.312, Δ_B = +0.014, Δ_C = +0.041.
4. P3P on R1+R2 (n = 39/branch small): Δ_A = +0.385, Δ_B = +0.029, Δ_C = +0.051.

**Decoder-robust dampening signature (A / B / C all < 0).**

1. HMM C1 on a1 legacy: Δ_A = −0.031, Δ_B = −0.006, Δ_C = −0.010. **NEW in 2026-04-24 matrix** (was B-outlier in 2026-04-22 with Δ_B = +0.000; run-to-run drift flipped it to ALL-agree).
2. HMM C1 on b1 legacy: Δ_A = −0.033, Δ_B = −0.023, Δ_C = −0.028.
3. M3R on R1+R2: Δ_A = −0.155, Δ_B = −0.011, Δ_C = −0.027.
4. HMS-T on R1+R2: Δ_A = −0.303, Δ_B = −0.143, Δ_C = −0.078.
5. VCD-test3 on R1+R2: Δ_A = −0.167, Δ_B = −0.081, Δ_C = −0.071.
6. M3R modified on R1+R2: Δ_A = −0.136, Δ_B = −0.037, Δ_C = −0.020.
7. VCD-test3 modified on R1+R2: Δ_A = −0.084, Δ_B = −0.028, Δ_C = −0.010.

**Decoder-dependent on the A / B / C triple.**

- HMM C2 / C4 on R1+R2 (B-outlier in both; A and C positive, B slightly negative).
- HMM C1 on c1 legacy (C-outlier; A and B positive, C slightly negative — c1 is §1 transition-boundary).
- **HMM C1 on e1 legacy (re-classified as decoder-dependent in 2026-04-24)**: Δ_A = +0.199, Δ_B = +0.041, **Δ_C = −0.002**. Was "ALL-agree sharpening" in 2026-04-22 with Δ_C = +0.011; run-to-run drift pushed Δ_C across zero. Adding Dec D and Dec E confirms the training-regime-dependence: Dec A / A′ (none for legacy) / Dec E all strongly positive (A = +0.199, E = +0.230), Dec C / Dec D-raw / Dec D-shape near zero. See § 17 "Legacy reference networks" and RESULTS.md §14 "e1 reclassification" for the full breakdown.
- HMS on R1+R2 (C-outlier; A and B negative, C positive at +0.053).
- HMS-T modified on R1+R2 (C-outlier; A and B negative, C positive at +0.051).

**Dec A vs retrain-decoder closure on dampening legacy rows (Tasks #5–#8,
2026-04-25 → 2026-05-03).** Three findings, each refining the previous:
(i) Task #5 — 5k retrains were Adam-undertrained (sklearn LBFGS reaches 0.70
on a1; Adam at 5k stalls at 0.36 with ||W||=143 vs 0.66 at 20k with ||W||=488).
(ii) Tasks #6–#7 — 20k Dec A′ exceeds Dec A on every net (a1: 0.36 → 0.67;
+8 pp above Dec A); but on the 17-row matrix at 20k, rows 5–6 (a1 / b1 HMM C1)
read Δ_A′(20k) = +0.21 / +0.18 — opposite to Dec A's −0.03. (iii) Task #8 —
disambiguation: trained Linear(36,36)+bias at 20 000 Adam lr=1e-3 (matching
Dec A′ regime) on Dec D's PAIRED-FORK BALANCED ex+unex data, no natural-HMM
prior asymmetry to exploit. On a1 / b1 HMM C1, Δ_D-raw(20k) = −0.024 / −0.046;
Δ_D-shape(20k) = −0.052 / −0.044 — agreeing with Dec A's small-dampening
direction. The 20k Dec A′ positive Δ on a1 / b1 was natural-HMM prior-bias
overfitting at large ||W||, not a hidden sharpening signal. **Section 5's
25-run-sweep regime labels (a1 / b1 dampening, c1 transitional, e1 best
sharpener) are reconfirmed.** Dec A's 0.59 with ||W||=82.5 is the small-norm
solution from Stage-1 co-training of L2/3 + decoder (`src/training/stage1_sensory.py:120-129`),
which Task #5 confirmed as the H1 mechanism. Sources:
`results/cross_decoder_comprehensive_20k_final.{json,md}`,
`results/task8_decD_20k_legacy/{net}_C1.json`,
`checkpoints/decoder_a_prime_20k_{net}.pt`,
`checkpoints/decoder_d_20k_{raw,shape}_{net}.pt`,
`/tmp/debug_dec_a_advantage_report.md`. Full account:
`docs/R1R2_full_report.md` § 9.6, `docs/research_log.md` 2026-05-03 entry.

**Dec D Kok-style signature on row 12 (2026-04-24).** HMS-T native on R1+R2
gives Δ_D-shape = +0.166 while Δ_A = −0.303. Amplitude-sensitive decoders
report dampening; the shape-normalised Dec D-shape reports sharpening on
the same `r_l23`. This is the Kok-framework co-occurrence — expectation
suppresses amplitude while sharpening the orientation-pattern shape. No
other row in the 17-row matrix shows this divergence at material magnitude.

**R1+R2 is not a single-regime network.** It is decoder-robust-sharpening
on the paired-fork paradigm (rows 1, 3, 9, 13 in the 2026-04-24 matrix;
Dec A′ and Dec E confirm) and decoder-robust-dampening on the matched-probe
observational paradigms (rows 10, 12, 14, 15, 17; plus row 16 / 11 flipping
under Dec C). The paradigm choice, not the decoder choice, drives the sign
on R1+R2. This remains the load-bearing finding. See RESULTS.md §14 and
ARCHITECTURE.md § "Decoders" for cross-references.
