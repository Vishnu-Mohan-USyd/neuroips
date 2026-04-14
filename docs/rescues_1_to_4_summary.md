# Single-Network Dual-Regime Rescues 1–4 — Summary and Results

**Date:** 2026-04-13
**Branch:** `failed-dual-regime-experiments` (stacked on `single-network-dual-regime`)
**Scope:** Baseline simple_dual + 3 rescues (R1+2, R3, R4), all trained seed=42, same Stage 1 + Stage 2 pipeline.

---

## Executive summary

The goal of these rescues was to make a single trained checkpoint of the V1–V2 laminar network exhibit **both** sharpening-like behaviour (Kok 2012) in focused/relevant task mode **and** dampening-like behaviour (Alink 2010 / Richter 2018) in routine/irrelevant task mode. Three successive architectural rescues were trained on top of the `simple_dual` emergent baseline. None of the three rescues produced a Richter-style dampening signature (activity reduction with preserved decoding). Instead, all three rescues (and baseline weakly) show a **subtractive predictive-coding** pattern: expected stimuli produce lower L2/3 activity AND lower decoding accuracy, with a broader (not narrower) tuning curve. The expectation-suppression effect is feedback-driven (FB-OFF control collapses the effect) and grows with each rescue (R1+2: +33%, R3: +19%, R4: +34% L2/3 activity gap). Sharpening metrics (M7 match-vs-near-miss, FWHM narrowing under FB ON) are present across all three rescues, but they reflect feedback-on-vs-off comparisons, not expected-vs-unexpected comparisons. The mismatch accuracy dissociation (mm_acc_irr vs mm_acc_rel) was actually strongest in **R1+2** (Δ=+0.220), weakened in R3 (+0.155) and R4 (+0.168). No single checkpoint cleanly satisfies the preregistered BOTH-regime criterion.

### 2026-04-14 dampening-analysis addendum

The current `dampening-analysis` branch carries a narrower follow-up that is
easy to confuse with this document's cross-checkpoint rescue summary. To keep
scope honest: that follow-up is **aligned pure-R1+2 only** on checkpoint
`r12_fb24_sharp_050_width_075_rec11_aligned`; it is **not** a fresh all-rescue
reanalysis.

That follow-up adds:
- `raw`, `delta`, and `baseline` response surfaces
- paired-state `branch_counterfactual` analysis from identical frozen
  pre-probe recurrent states
- an analysis-only baseline-centering fix

Relevant branch-counterfactual outcome for the aligned pure-R1+2 checkpoint:
- `baseline`: identical after the fix
  (`peak=0.375691`, `FWHM=39.389691°` for both expected and unexpected).
- `raw`: expected lower and slightly narrower than unexpected
  (`peak 0.449558 vs 0.507532`, `FWHM 33.237447° vs 33.515513°`).
- `delta`: expected much lower and much narrower than unexpected
  (`peak 0.072438 vs 0.491614`, `FWHM 27.581737° vs 45.064195°`).

Baseline-centering bug and fix, briefly:
the branch-counterfactual baseline path already returned the same pre-probe
tensor for both branches, but it used different branch probe channels for
re-centering. That rotated identical baselines into artificially opposite
shapes. Baseline mode now uses the shared predicted/expected channel for both
branches.

---

## Context

### Target experimental setup

The network is the single V1–V2 laminar circuit trained on the three-loss task (see [target_experimental_setup memory](../../../home/vysoforlife/.claude/projects/-mnt-c-Users-User-codingproj-freshstart/memory/target_experimental_setup.md)):

- **Loss:** `task_state[0]*(3.0*sensory + 0*mismatch + 1.0*energy) + task_state[1]*(0.3*sensory + 1.0*mismatch + 1.0*energy)`
- **task_state:** Markov chain with `p_switch=0.2` per presentation.
- **Monitored accuracies (every 100 steps):** `s_acc_rel`, `s_acc_irr`, `mm_acc_rel`, `mm_acc_irr`.
- **Preregistered BOTH-regime criterion:** `s_acc_rel > s_acc_irr` AND `mm_acc_irr > mm_acc_rel`, both with clear margins, from a single checkpoint.

### Expected vs unexpected protocol

The `scripts/analysis_expected_vs_unexpected` pipeline (used for all 4 checkpoints) extracts the V2 prediction at the last ISI timestep (pre-stimulus), measures the angular error between that prediction and the subsequent stimulus orientation, and classifies each trial as:
- **Expected:** pred_err ≤ 10°
- **Unexpected:** pred_err > 20°
- **Middle:** 10° < pred_err ≤ 20° (excluded from primary comparisons).

Feedback is applied at full strength (`feedback_scale=1.0`). Stimulus response is measured at a late ON timestep (t=9 of 12). N per condition is 1000–3000 presentations per checkpoint.

### Architecturally prior work (summary)

Before the R-series rescues, the following milestones were documented:

- **Phase 2.4.1 alpha_net causal E/I gate** (commit `ef4f102` + `cde523f`): learned focused/routine differentiation in the E/I feedback split (Δg_E=+1.34, Δg_I=−1.60). Confirmed after the `unfreeze_stage2` freeze-bug fix. Separate architectural pathway, not on the R-series branch.
- **Network_mm / Fix A** (commit `9d1e954`, simple_dual_v2): MLP mismatch_head (`Linear(36,64)→ReLU→Linear(64,1)`). Result: mm_acc_irr 0.906 / mm_acc_rel 0.694 (Δmm +0.212), sensory Δ +0.001 (flat). Mismatch dissociation only.
- **Network_both / Fix A + Fix 2** (commit `fb7a1a1`, simple_dual_v3): per-regime head_feedback (two `Linear(16,36)` gated by task_state). Result: Δsens +0.052 (weak), Δmm +0.140 (regressed from Fix-A). Reallocated signal from mm to partial sensory without expanding the dissociation budget.

These predecessors fed the external review that identified the 5 architectural gaps the rescues were designed to address.

---

## The 5 architectural gaps (external review)

1. **Precision as bookkeeping, not gain control** — `pi_pred` was computed but not used to modulate feedback amplitude.
2. **No disinhibitory mechanism** — VIP population was stripped; feedback had no gating pathway beyond raw E/I split.
3. **No laminar separation of expectation from evidence** — predicted orientation not represented by a dedicated template population.
4. **Global (not feature-specific) losses** — expected_suppress penalized mean activity, not activity at the predicted channel.
5. **Shared-decoder bottleneck** — sensory and mismatch heads read the same feature (`r_l23`), forcing the network to balance two objectives on a single representation.

Rescues 1–4 each addressed one or more of these gaps incrementally.

---

## Rescue 1+2: precision gating + feature-specific expected_suppress

### Architecture

Commit `761a5e8`. Two minimal architectural changes that compose into a single sweep (`config/sweep/sweep_rescue_1_2.yaml`):

**Change A — precision gating on the feedback split** (gap 1). In `src/model/network.py`:
```python
if self.cfg.use_precision_gating:
    precision_gate = pi_pred_raw / self.cfg.pi_max     # [B, 1] in [0, 1]
    scaled_fb = feedback_signal * self.feedback_scale * precision_gate
else:
    scaled_fb = feedback_signal * self.feedback_scale   # baseline
```
The feedback is now multiplicatively modulated by the network's own precision estimate. When precision is low, feedback attenuates.

**Change B — feature-specific expected_suppress loss** (gap 4). In `src/training/losses.py::expected_suppress_loss`:
```python
# Before (global): mean(|r_l23|)
# After (feature-specific):
(r_l23_windows * q_pred_windows).sum(dim=-1)
```
Penalize only the component of L2/3 activity aligned with the predicted orientation, not overall activity.

Also enabled: `lambda_state = 1.0` (restoring the prior_kl loss so V2 keeps learning a coherent task-state prior).

### Training end-state (step 5000)

| metric | value |
|---|---|
| s_acc_rel | 0.590 |
| s_acc_irr | 0.342 |
| **Δ_sens** | **+0.248** |
| mm_acc_rel | 0.657 |
| mm_acc_irr | 0.877 |
| **Δ_mm** | **+0.220** |
| training time | 3285 s |

### Representation analysis

| metric | value |
|---|---|
| M7 δ=5° | +0.228 (p<1e-4) |
| M7 δ=10° | +0.311 (p<1e-4) |
| M7 δ=15° | +0.241 (p<1e-4) |
| FWHM ON | 21.59° |
| FWHM OFF | 26.69° |
| FWHM Δ (ON−OFF) | −5.09° (narrowing) |
| peak ON | 0.94 |
| peak OFF | 0.29 |

### Expected vs unexpected

| metric | Exp (n=2074) | Unexp (n=2479) | Δ | % | p |
|---|---|---|---|---|---|
| L2/3 total | 2.87 | 3.82 | +0.96 | **+33%** | 1.5e-130 |
| L2/3 peak | 0.43 | 0.60 | +0.17 | +39% | 8.4e-139 |
| L2/3 FWHM | 31.25° | 29.35° | −1.90° | −6.1% | 2.3e-18 |
| L4 total | 1.82 | 1.81 | −0.01 | −0.6% | ns |
| center_exc | 2.30 | 4.20 | +1.90 | +82% | 9.0e-270 |
| FB-OFF gap | — | — | −0.10 | — | — |

Feedback WIDENS the gap (expectation-suppression signal exists only with FB ON). L4 is invariant, confirming the effect is feedback-driven.

### Decoding accuracy (expected vs unexpected, FB ON)

| slice | acc_exp | acc_unexp | Δ | p |
|---|---|---|---|---|
| overall | 0.255 | 0.339 | **−0.083** | 9.4e-10 |
| focused | (same sign) | — | −0.081 | — |
| routine | — | — | −0.083 | — |

Unexpected trials decode better than expected, with matched sign in both task_state conditions.

---

## Rescue 3: VIP-SOM disinhibition

### Architecture

Commit `b76001b`. Stacked on top of Rescue 1+2 — keeps precision gating and feature-specific expected_suppress, adds a disinhibitory pathway (gap 2).

**New V2 head:** `head_vip = Linear(16, 36)` produces a VIP drive vector from the V2 GRU hidden state.

**New V1 population:** `VIPRing` (leaky integrator, τ=10) receives the VIP drive and then inhibits the SOM population via a softplus-positive learnable weight:
```python
r_vip = self.vip(vip_drive, state.r_vip)
w_vip = F.softplus(self.w_vip_som_raw)
effective_som_drive = relu(som_drive_fb - w_vip * r_vip)
r_som = self.som(effective_som_drive, state.r_som)
```

**Structured surround kernel:** `make_circular_gaussian_kernel(n=36, sigma=20.0, period=180)` applied to the SOM drive (not a flat inhibition).

Config file: `config/sweep/sweep_rescue_3.yaml`.

### Training end-state (step 5000)

| metric | value |
|---|---|
| s_acc_rel | 0.620 |
| s_acc_irr | 0.410 |
| **Δ_sens** | **+0.210** |
| mm_acc_rel | 0.722 |
| mm_acc_irr | 0.877 |
| **Δ_mm** | **+0.155** |
| training time | 3470 s |

### Representation analysis

| metric | value |
|---|---|
| M7 δ=5° | +0.228 |
| M7 δ=10° | +0.309 |
| M7 δ=15° | +0.244 |
| FWHM ON | **17.92°** (strongest narrowing) |
| FWHM OFF | 26.68° |
| FWHM Δ | **−8.76°** |
| peak ON | 0.94 |
| peak OFF | 0.29 |

R3 produced the **narrowest tuning curves** of the entire rescue series under FB ON.

### Expected vs unexpected

| metric | Exp (n=2515) | Unexp (n=1909) | Δ | % | p |
|---|---|---|---|---|---|
| L2/3 total | 3.08 | 3.67 | +0.60 | **+19%** | 1.1e-57 |
| L2/3 peak | 0.52 | 0.63 | +0.11 | +21% | 2.4e-46 |
| L2/3 FWHM | 28.47° | 26.74° | −1.73° | −6.1% | 1.1e-12 |
| L4 total | 1.82 | 1.83 | +0.02 | +0.9% | ns |
| center_exc | 2.78 | 4.44 | +1.66 | +60% | 5.4e-198 |
| FB-OFF gap | — | — | −0.12 | — | — |

### Decoding accuracy

| slice | acc_exp | acc_unexp | Δ | p |
|---|---|---|---|---|
| overall | 0.291 | 0.393 | **−0.103** | 8.2e-13 |
| focused | — | — | −0.111 | — |
| routine | — | — | −0.096 | — |

---

## Rescue 4: DeepTemplate + error-based mismatch readout

### Architecture

Commit `564ae1b`. Stacked on top of Rescue 1+2 (keeps precision gating + feature-specific expected_suppress + lambda_state=1.0). Does NOT include R3's VIP pathway. Adds laminar separation (gap 3) and breaks the shared-decoder bottleneck (gap 5).

**New V1 population:** `DeepTemplate` leaky integrator in `src/model/populations.py`:
```python
class DeepTemplate(nn.Module):
    def __init__(self, cfg):
        self.tau = cfg.tau_template   # 10
        raw_init = math.log(math.exp(cfg.template_gain) - 1.0)
        self.gain_raw = nn.Parameter(torch.tensor(raw_init))

    @property
    def gain(self) -> Tensor:
        return F.softplus(self.gain_raw)

    def forward(self, q_pred, pi_pred_eff, r_prev):
        drive = self.gain * q_pred * pi_pred_eff   # [B, 36]
        return r_prev + (drive - r_prev) / self.tau
```
The template is driven by the V2 prediction `q_pred` (softmax over 36 orientations) weighted by the gated precision `pi_pred_eff`.

**Error population:** `r_error = relu(r_l23 - r_template)` computed per-timestep on the raw `[B, T, N]` trajectory before window averaging. Implemented in `src/training/stage2_feedback.py`.

**Mismatch head rewiring:** When `cfg.use_error_mismatch=True`, the `mismatch_head` (MLP 36→64→1) reads from `r_error_windows` instead of `r_l23_windows`. The sensory decoder (`orientation_decoder`) continues to read `r_l23` — sensory code is preserved.

Flags added to `ModelConfig`: `use_deep_template: bool = False`, `template_gain: float = 1.0`, `tau_template: int = 10`, `use_error_mismatch: bool = False`. `template_gain` unstripped from the legacy strip list at `src/config.py:260`. `unfreeze_stage2` and `create_stage2_optimizer` both gated on `hasattr(net, 'deep_template_pop')` to avoid the Phase 2.4.1 freeze-bug pattern.

Config file: `config/sweep/sweep_rescue_4.yaml`. Regression tests: `tests/test_network.py::TestRescue4DeepTemplate` (bit-identity with flag off; template differs + r_l23 preserved with flag on). All 299/299 tests pass.

### Training end-state (step 5000)

| metric | value |
|---|---|
| s_acc_rel | 0.593 |
| s_acc_irr | 0.352 |
| **Δ_sens** | **+0.241** |
| mm_acc_rel | 0.540 |
| mm_acc_irr | 0.708 |
| **Δ_mm** | **+0.168** |
| training time | 2811 s |

Network parameters: 7393 (= baseline 7392 + 1 for DeepTemplate `gain_raw`).

### Representation analysis

| metric | value |
|---|---|
| M7 δ=5° | +0.238 (CI 0.185–0.295, p<1e-4) |
| M7 δ=10° | +0.315 (CI 0.244–0.358, p<1e-4) |
| M7 δ=15° | +0.244 (CI 0.205–0.309, p<1e-4) |
| FWHM ON | 20.53° |
| FWHM OFF | 26.69° |
| FWHM Δ | −6.16° |
| peak ON | 1.05 |
| peak OFF | 0.29 |

### Expected vs unexpected

| metric | Exp (n=1938) | Unexp (n=2669) | Δ | % | p |
|---|---|---|---|---|---|
| L2/3 total | 2.97 | 3.99 | +1.02 | **+34%** | 4.6e-134 |
| L2/3 peak | 0.44 | 0.63 | +0.19 | +42% | 1.1e-159 |
| L2/3 FWHM | 31.55° | 28.74° | −2.81° | −8.9% | 1.6e-38 |
| L4 total | 1.83 | 1.79 | −0.04 | −1.9% | 0.019 |
| center_exc | 2.37 | 4.39 | +2.01 | +85% | 3.9e-281 |
| FB-OFF gap | — | — | −0.10 | — | — |

### Decoding accuracy

| slice | acc_exp | acc_unexp | Δ | p |
|---|---|---|---|---|
| overall | 0.262 | 0.331 | **−0.069** | 4.1e-07 |
| focused | — | — | −0.072 | — |
| routine | — | — | −0.066 | — |

---

## Cross-rescue summary tables

### Table 1 — Training end-state (step 5000)

| checkpoint | s_acc_rel | s_acc_irr | Δ_sens | mm_acc_rel | mm_acc_irr | Δ_mm | Parameters |
|---|---|---|---|---|---|---|---|
| baseline (simple_dual) | 0.696 | 0.673 | +0.023 | 0.462 | 0.533 | +0.071 | 7392 |
| **Rescue 1+2** | 0.590 | 0.342 | **+0.248** | 0.657 | 0.877 | **+0.220** | 7392 |
| Rescue 3 (VIP) | 0.620 | 0.410 | +0.210 | 0.722 | 0.877 | +0.155 | 7447 (VIP adds ~55) |
| Rescue 4 (DeepTemplate) | 0.593 | 0.352 | +0.241 | 0.540 | 0.708 | +0.168 | 7393 (+1 gain) |

**Observation:** R1+2 has the strongest mm dissociation (+0.220). R4 regressed on mm (+0.168). Baseline has almost no task-state dissociation on either s_acc or mm_acc.

### Table 2 — Sharpening metrics (FB ON vs FB OFF, from representation analysis)

| checkpoint | M7 δ=10° | FWHM ON | FWHM OFF | FWHM Δ | peak ON | peak OFF | amp ratio |
|---|---|---|---|---|---|---|---|
| baseline | +0.339 | 24.09° | 26.68° | −2.59° | 1.52 | 0.29 | 5.2× |
| Rescue 1+2 | +0.311 | 21.59° | 26.69° | −5.09° | 0.94 | 0.29 | 3.2× |
| Rescue 3 | +0.309 | **17.92°** | 26.68° | **−8.76°** | 0.94 | 0.29 | 3.2× |
| Rescue 4 | +0.315 | 20.53° | 26.69° | −6.16° | 1.05 | 0.29 | 3.6× |

All checkpoints show M7 near-miss decoding improvement under FB ON. **R3 has the narrowest FWHM** (17.92°) but comparable M7 to others. Rescues reduced peak amplitude relative to baseline (from 5.2× to 3.2–3.6×).

### Table 3 — Expected vs unexpected activity (L2/3 total)

| checkpoint | Exp | Unexp | Δ | % | FB-OFF Δ | feedback-driven? |
|---|---|---|---|---|---|---|
| baseline | 6.22 | 6.39 | +0.17 | +2.7% | +0.002 (ns) | no (feedback doesn't widen gap) |
| Rescue 1+2 | 2.87 | 3.82 | +0.96 | **+33%** | −0.10 | **yes** |
| Rescue 3 | 3.08 | 3.67 | +0.60 | +19% | −0.12 | **yes** |
| Rescue 4 | 2.97 | 3.99 | +1.02 | **+34%** | −0.10 | **yes** |

All 3 rescues show feedback-driven expectation suppression. R1+2 and R4 are strongest (tied at ~34%); R3 intermediate (+19%).

### Table 4 — Expected vs unexpected decoding accuracy

| checkpoint | n_exp | n_unexp | acc_exp | acc_unexp | Δ | p | margin_exp | margin_unexp |
|---|---|---|---|---|---|---|---|---|
| baseline | 937 | 3619 | 0.572 | 0.644 | **−0.072** | 4.4e-05 | 0.250 | 0.276 |
| Rescue 1+2 | 2071 | 2506 | 0.255 | 0.339 | **−0.083** | 9.4e-10 | 0.107 | 0.159 |
| Rescue 3 | 2506 | 1905 | 0.291 | 0.393 | **−0.103** | 8.2e-13 | 0.115 | 0.194 |
| Rescue 4 | 1946 | 2666 | 0.262 | 0.331 | **−0.069** | 4.1e-07 | 0.107 | 0.159 |

**Unexpected decodes better than expected in every checkpoint**, significant in every case, and persists within both task_states (focused and routine).

### Table 5 — Information efficiency (accuracy per unit L2/3 activity)

| checkpoint | acc_exp / act_exp | acc_unexp / act_unexp | direction |
|---|---|---|---|
| baseline | 0.572/6.22 = 0.092 | 0.644/6.39 = 0.101 | unexp more efficient |
| Rescue 1+2 | 0.255/2.87 = 0.089 | 0.339/3.82 = 0.089 | tied |
| Rescue 3 | 0.291/3.08 = 0.094 | 0.393/3.67 = 0.107 | unexp more efficient |
| **Rescue 4** | 0.262/2.97 = **0.088** | 0.331/3.99 = 0.083 | **expected marginally more efficient** |

R4 is the only checkpoint where expected has higher acc-per-spike, but the advantage is small (6%) and does not meet a Richter-dampening threshold.

### Table 6 — Tuning curve shape (peak + width for expected vs unexpected)

| checkpoint | peak_exp | peak_unexp | FWHM_exp | FWHM_unexp | shape interpretation |
|---|---|---|---|---|---|
| baseline | 1.03 | 0.97 | 26.6° | 27.4° | slight sharpening direction (exp peak > unexp peak) |
| Rescue 1+2 | 0.43 | 0.60 | 31.2° | 29.4° | **anti-sharpening** (exp lower peak AND wider) |
| Rescue 3 | 0.52 | 0.63 | 28.5° | 26.7° | anti-sharpening |
| Rescue 4 | 0.44 | 0.63 | 31.6° | 28.7° | anti-sharpening |

In all three rescues, expected trials have both a lower peak and a broader tuning curve — the opposite of Kok 2012 sharpening. This is the signature of subtractive interference, not representational refinement.

---

## Interpretation

### Richter/Alink dampening criterion

The Richter 2018 / Alink 2010 dampening pattern is:
- Expected stimuli produce **reduced** population activity.
- Tuning curve **shape** is preserved (same preferred orientation, similar FWHM).
- Decoding accuracy is **preserved** or nearly preserved despite the amplitude reduction.

Evaluating against our data:
- ✓ Reduced activity for expected — met by all 3 rescues (R1+2, R3, R4).
- ✗ Preserved tuning shape — violated: FWHM is BROADER for expected in all 3 rescues.
- ✗ Preserved decoding — violated: acc_exp < acc_unexp by 7–10 pp in every checkpoint.

**None of the three rescues match Richter-style dampening.**

### Kok 2012 sharpening criterion

The Kok sharpening pattern is:
- Expected stimuli produce **narrower** tuning curves (FWHM decreases).
- Decoding accuracy is **higher** for expected stimuli.
- This can be accompanied by roughly preserved or slightly reduced amplitude.

Evaluating:
- ✗ Narrower FWHM for expected — violated: FWHM is broader for expected in all 3 rescues.
- ✗ Higher decoding accuracy for expected — violated in every checkpoint.

**None of the checkpoints (including baseline) match Kok sharpening at the expected-vs-unexpected axis.** However, at the feedback-on-vs-off axis, all rescues DO show M7-style sharpening (FWHM narrows under FB, match-vs-near-miss improves). The two axes tell different stories.

### What the pattern actually looks like: subtractive predictive coding

The observed R-series pattern is consistent with subtractive predictive coding (Rao & Ballard 1999; Friston 2010):
- Expected stimulus: the template (driven by q_pred) partially cancels the input at the predicted channel → L2/3 peak lower, residual distributes more diffusely across the population → broader FWHM, lower discriminability.
- Unexpected stimulus: template is misaligned, fails to cancel input → L2/3 peak higher at the true channel → narrower effective response, higher decoding.
- Feedback removes the signal it itself predicted — classical prediction-error signalling.

This is NOT the dissociation the project set out to produce. The hypothesis was that task_state would route the circuit into one regime (sharpening-by-expectation, focused) or the other (dampening-by-expectation, routine). Instead, the same subtractive pattern appears in both task_states (per-regime decoding Δ is negative and similar-sized in focused and routine for all 3 rescues).

### Feedback-on vs feedback-off sharpening IS present (but orthogonal)

The M7 and FWHM-narrowing results under FB ON are real and significant:
- M7 δ=10° > +0.30 in all rescues and baseline.
- FWHM narrows by −2.6° (baseline) to −8.76° (R3) with FB on.

But these compare FB ON vs FB OFF on the same stimulus distribution — not expected vs unexpected stimuli under FB ON. The M7 effect says "feedback improves fine discrimination overall", not "feedback improves discrimination of expected stimuli more than unexpected stimuli". The published Kok signature requires the latter.

### Mismatch dissociation is architecturally fragile

R4's architectural innovation (error-based mismatch readout) was intended to DECOUPLE the sensory and mismatch codes — giving each head a different feature basis. Instead, mm dissociation REGRESSED from R1+2's +0.220 down to +0.168. Hypothesis: `r_error = relu(r_l23 - r_template)` is a WEAKER feature for mismatch than `r_l23` itself, because the ReLU clips half the signal (the template-over-predicts case) and the feedback-driven template is often weak or stale at the moment mismatch needs to be detected. R1+2's globally-weighted expected_suppress + direct `r_l23` readout remains the best mm-dissociation architecture.

### Summary verdict

- **Sharpening axis (FB on vs off):** present in all 3 rescues + baseline.
- **Dampening axis (expected vs unexpected activity):** feedback-driven expectation suppression present in all 3 rescues.
- **Preserved-representation dampening (Richter):** absent in all 3 rescues.
- **Kok-style sharpening by expectation:** absent in all 3 rescues.
- **Preregistered BOTH-regime criterion:** closest is R1+2 (strongest mm dissociation, strong Δ_sens, strong expectation suppression), but neither of the two regime-specific representational signatures (Kok / Richter) cleanly appears in either task_state.

---

## Outstanding scientific questions

1. **Is the subtractive-coding pattern inevitable from this architecture?** The feedback operator is `scaled_fb = feedback_signal * feedback_scale * precision_gate`, split by sign into `center_exc` and `som_drive_fb`. `center_exc` enters L2/3 additively at the predicted channel. There is no mechanism for feedback to SHARPEN tuning around the predicted orientation without also amplifying activity there — so when q_pred is correct, we'd expect peak UP not DOWN. Yet we see expected peak DOWN. This suggests the expected_suppress loss is overpowering center_exc's constructive contribution and producing net subtractive behaviour. Rebalancing these could matter (Rescue 5 candidate).

2. **Does task_state actually route feedback differently?** Per-regime decoding deltas (focused vs routine) are similar across conditions. The Phase 2.4.1 alpha_net gate DID produce focused/routine E/I differentiation — but that was a separate pathway not in the R-series branch. Merging R-series with alpha_net may be a necessary next step.

3. **Is 10° / 20° the right expected/unexpected threshold?** With p_switch=0.2 and 36 orientation channels, the Markov baseline predicts substantial angular error even for "stable" task_state. The pred_err histogram is bimodal with a cluster near 0° and another near 180/2 = 90°. A stricter expected definition (pred_err ≤ 5°) might unmask effects currently averaged out.

4. **Does the mismatch head architecture need fundamental redesign?** R1+2's simple direct readout outperforms R4's error-based readout. The scientific question is whether there's a principled feature that beats raw `r_l23` for mismatch — candidates: max-pooling a local window, decorrelated residual, or a V2-side mismatch head that reads h_v2 directly.

5. **Can a rescue produce Richter dampening at all with this architecture?** The preservation criterion requires that the template subtract the amplitude but not the tuning shape. The current `r_template = gain * q_pred * pi_pred_eff` has a narrower spatial shape than `r_l23` (one-hot-like vs broad tuning curve), so subtraction flattens the output tuning. A tuned-template that matched the shape of `r_l23`'s predicted-channel response might preserve shape on subtraction. (Candidate: train the template population with a shape-matching loss against a leave-one-out r_l23 prediction.)

---

## Artefacts

### Local (WSL, `/mnt/c/Users/User/codingproj/freshstart`)

- Branch: `failed-dual-regime-experiments`
- Commits:
  - `761a5e8` — Rescue 1+2: precision gating + feature-specific expected_suppress
  - `b76001b` — Rescue 3: VIP-SOM disinhibition with surround kernel
  - `564ae1b` — Rescue 4: DeepTemplate + error-based mismatch
- Remote: `origin/failed-dual-regime-experiments`
- Regression tests (relevant): `tests/test_network.py::TestRescue4DeepTemplate`
- Sweep configs: `config/sweep/sweep_rescue_1_2.yaml`, `config/sweep/sweep_rescue_3.yaml`, `config/sweep/sweep_rescue_4.yaml`
- Analysis script (new, Task #20): `scripts/decoding_by_expected.py`

### Remote (reuben-ml, vishnu@reuben-ml)

| checkpoint | path | size (approx) |
|---|---|---|
| baseline (simple_dual) | `/home/vishnu/neuroips/simple_dual/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt` | ~115 KB |
| Rescue 1+2 | `/home/vishnu/neuroips/rescue_1_2/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt` | ~115 KB |
| Rescue 3 | `/home/vishnu/neuroips/rescue_3/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt` | ~115 KB |
| Rescue 4 | `/home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt` | ~115 KB |

### Analysis output files (on reuben-ml)

- Representation analysis logs: `.../results/<label>/analysis_representation.log`
- Expected-vs-unexpected activity logs: `.../results/<label>/analysis_expected_vs_unexpected.log`
- Summary jsons: `.../results/<label>/summary.json`
- Decoding-by-expected jsons (Task #20): `/home/vishnu/neuroips/rescue_4/freshstart/results/decoding_by_exp/{baseline,rescue_1_2,rescue_3,rescue_4}.json`
- Training logs: `.../results/<label>/sweep.log`

### Training pipeline

- Launcher: `scripts/run_sweep.sh <config.yaml> <outdir> <seed> <device>`
- SIGHUP-hardened via `exec setsid --wait bash "$0" "$@" < /dev/null > /dev/null 2>>"$_RSH_LOGFILE"` (commit `69dee73`).
- Deploy pattern: `git archive HEAD | ssh vishnu@reuben-ml 'tar xf - -C /home/vishnu/neuroips/<label>/freshstart'` (no GitHub auth needed on remote).

---

## Decision points for next step

Options identified from the current data:

A. **Accept the subtractive-coding result and write it up.** The finding is scientifically meaningful: a standard laminar V1–V2 architecture with precision-gated feedback produces subtractive prediction-error signalling, NOT Kok-style sharpening or Richter-style dampening. That's publishable — a negative result against both standard hypotheses.

B. **Rescue 5: merge R-series with Phase 2.4.1 alpha_net gate.** The alpha_net causal E/I gate was independently shown to produce focused/routine differentiation. It's never been combined with precision gating + feature-specific expected_suppress + VIP + DeepTemplate. The combined architecture might finally dissociate the two regimes in the way task_state should in principle allow.

C. **Rescue 5 (alternative): shape-preserving template.** Replace the one-hot-like `gain * q_pred * pi_pred_eff` with a learned projection from `q_pred` into a broad tuning curve matching `r_l23`'s expected shape. This could let the template subtract amplitude without flattening tuning.

D. **Deeper diagnosis on the existing 4 checkpoints.** Fisher information per-channel, margin-landscape analysis across the (pred_err × true_theta) plane, or a decoder-free mutual-information test. This may reveal whether any subtle Richter signal is present but masked by the top-1 accuracy metric.

E. **Shift the axis of analysis.** Re-evaluate all 4 checkpoints on task_state (focused vs routine) splits of the activity and decoding metrics — the "expected vs unexpected" axis may be the wrong test for the "dampening" hypothesis if the original mapping of dampening→routine was correct.

No option is chosen yet. Pending user direction.

---

## Update (2026-04-13): Re-centered tuning curve analysis — R4 resembles Richter preserved-shape dampening

This section **revises (does not replace)** the interpretation captured above in *"What the pattern actually looks like: subtractive predictive coding"* (lines 382–389). The original analysis collected per-trial `r_l23[t=9]` curves indexed by **physical orientation channel** and averaged within (regime × bucket) buckets. Because each bucket pools trials at many different true orientations, the average smears the population peak across channels — the bucket-mean curve is a circular average of curves whose peaks are scattered around the ring. This smearing inflates apparent FWHM and can hide whether the per-trial response is preserved-shape dampening or genuine subtractive central clipping.

### What the re-centered analysis does

For each trial we compute the true-stimulus channel `true_ch = round(actual_ori / step_deg) % N`, then `np.roll(r_l23[t=9], shift=CENTER_IDX − true_ch)` so that every trial's stimulus peak lands at the same array index (`CENTER_IDX = 18`, the array midpoint). We then average within (regime × bucket) buckets. The resulting curve is the **stimulus-aligned population tuning curve**: peak amplitude, baseline, and FWHM all describe the per-trial response shape rather than a circular smear.

FWHM is computed via **linear interpolation at half-max crossings** (`half_max = baseline + 0.5 × (peak − baseline)`, then linear interpolation on both flanks). The earlier bin-counting method snapped every panel to integer multiples of 5°, hiding sub-bin differences.

### Cross-checkpoint summary (Relevant task_state, late-ON t=9)

All four checkpoints were re-analysed with the same re-centering pipeline (≥ 2000 trials per panel; n per bucket varies but each Expected n ≥ 1500). Numbers below are **Relevant** task_state; the **Irrelevant** numbers (not shown) are quantitatively similar.

| checkpoint | Rel-Exp total | Rel-Unexp total | Δ total | Rel-Exp peak | Rel-Unexp peak | Δ peak | Exp FWHM | Unexp FWHM | Δ FWHM | figure |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 6.93 | 7.17 | +3.5% | 0.987 | 0.992 | +0.5% | 29.3° | 28.2° | +1.1° | `tuning_ring_recentered_baseline.png` |
| R1+R2 | 4.05 | 4.78 | +18% | 0.564 | 0.639 | +13% | 33.3° | 31.8° | +1.5° | `tuning_ring_recentered_r1_2.png` |
| R3 | 4.00 | 4.61 | +15% | 0.602 | 0.605 | +0.5% | 31.7° | 30.6° | +1.1° | `tuning_ring_recentered_r3.png` |
| R4 | 4.21 | 5.02 | +19% | 0.578 | 0.662 | +15% | 33.6° | 32.1° | +1.5° | `tuning_ring_recentered_r4.png` |

(Δ is unexpected−expected; positive Δ means expected is **lower/narrower** than unexpected. FWHM rounded to 0.1°.)

### Per-checkpoint interpretation (re-centered)

- **Baseline (no rescue).** Total +3.5%, peak +0.5%, FWHM matched within 1.1°. Effectively **no expectation modulation** — sensory drive dominates and feedback contributes negligibly to the tuning curve. This re-centered analysis confirms baseline is the right null for the rescues.

- **Rescue 1+2 (precision gating + feature-specific expected_suppress).** Total drops 18%, peak drops 13%, FWHM nearly matched (Δ = +1.5°). Both **amplitude AND peak** are dampened on expected trials, with shape essentially preserved. This is a Richter-style preserved-shape dampening signature.

- **Rescue 3 (VIP-SOM disinhibition).** Total drops 15%, peak essentially tied (Δ = +0.5%), FWHM matched within 1.1°. R3 produces **total-only dampening** — the population is reduced uniformly, not by clipping the peak. This is a distinct profile: a divisive/SOM-like gain rescaling rather than a peak-targeted subtraction.

- **Rescue 4 (DeepTemplate + error-mismatch).** Total drops 19%, peak drops 15%, FWHM matched within 1.5°. R4 shows the **largest preserved-shape dampening** of the four checkpoints: substantial peak reduction, substantial total reduction, with the response shape (FWHM) closely matched between expected and unexpected. This is the closest match to the published Richter (2018) preserved-shape dampening signature among the four checkpoints.

### Corrective note on the earlier "subtractive predictive coding" interpretation

The earlier section (*"What the pattern actually looks like: subtractive predictive coding"*, lines 382–389) reported that **expected FWHM was broader than unexpected FWHM** in all rescues, leading to the conclusion that the rescues produce subtractive central clipping (peak removal that pushes residual into the flanks → broader curve → lower discriminability). That conclusion was based on **non-re-centered, bin-counted FWHM** of the bucket-mean curve. Both methodological choices inflated apparent FWHM and exaggerated the expected/unexpected FWHM gap.

The re-centered, interpolation-based analysis here shows the per-trial response actually **preserves shape**: FWHM differs by ≤ 1.5° across all rescues, while peak and total drop substantially on R1+R2 and R4. This matches the Richter "preserved-shape dampening" signature, not the Rao-Ballard subtractive-coding prediction. The earlier section is preserved as a record of the investigation but should be read together with this update.

Both observations are correct, but they answer different questions: the non-re-centered curve describes how a population of trials with scattered peaks looks on average; the re-centered curve describes how a single trial's tuning response is shaped by feedback. The re-centered view is the appropriate one for comparison against the Richter / Kok representational signatures.

### Caveat: the BOTH-regime preregistered criterion is still NOT met

Critically, the re-centered dampening signature appears in **both** Relevant **and** Irrelevant task_state with similar magnitude on all rescues. The preregistered hypothesis was that task_state should **route** the circuit into one regime (Kok sharpening for focused) or the other (Richter dampening for routine). What we observe instead is **task-state-invariant** preserved-shape dampening on R4 (and to a lesser extent R1+R2). The task_state input does not gate the representational mode; both regimes produce the same dampening profile.

So the headline correction is narrow: **R4 looks like Richter preserved-shape dampening per se**, but it does NOT produce the **dissociation** between regimes that the project set out to demonstrate. The decision-point summary in the previous section (Options A–E) is unaffected by this update — Option A (write up as a negative result on the dissociation hypothesis) and Options B/C (architectural changes to recover dissociation) remain the open paths.

### Reproducibility

- Re-centered ring figures: `scripts/plot_tuning_ring_extended.py` (uses `np.roll` per trial and interpolated FWHM; saved as `docs/figures/tuning_ring_recentered_*.png`).
- Per-orientation pooled ring (no re-centering, all probes): `tuning_ring_allprobes.png`.
- Original probe-60° ring with annotation overlays: `tuning_ring_heatmap.png`.
- Expected vs Unexpected per-bucket curves: `scripts/plot_tuning_exp_vs_unexp.py` → `docs/figures/exp_vs_unexp_tuning.png`.
