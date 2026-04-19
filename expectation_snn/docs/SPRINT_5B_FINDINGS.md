# Sprint 5b findings — H1 regime-dependency NOT supported

**Branch:** `expectation-snn-v1h` &nbsp;·&nbsp; **Commit of record:** `6355c8b`
**Date:** 2026-04-19 &nbsp;·&nbsp; **Seed:** 42

---

## Overview

This report summarises the Sprint 5b balance sweep for the Brian2-based
expectation SNN model — a three-ring cortical surrogate (V1 E/PV/SOM +
two higher-area "H" populations, H_R for Richter grammars and H_T for
Tang rotating-deviant grammars) with bottom-up tuning learned in Stage 0,
grammar-specific H-ring dynamics in Stage 1, and cue-gated three-factor
plasticity in Stage 2. Feedback from H → V1 runs through two fixed-weight
Gaussian routes (direct-to-apical and SOM-mediated) whose balance is
parameterised by `r = g_direct / g_SOM` at a fixed total gain `g_total`.

Hypothesis H1 states that this balance ratio controls the *sign* of
predictive modulation at V1: the SOM-dominant regime (low r) and the
direct-apical-dominant regime (high r) should push primary metrics
(Kok amplitude modulation, Richter redistribution, Tang sharpening) in
opposite directions. Sprint 5b tested H1 by sweeping r across two decades
{0.25, 0.5, 1.0, 2.0, 4.0} at `g_total = 1.0`, seed 42, on the intact
frozen network.

**Result: H1 is not supported in this configuration.** Across 13 primary
metrics, zero exhibit the predicted sign-flip between r=0.25 and r=4.0;
seven are monotone in r (magnitude only), one is null, five are
non-monotone. r modulates *magnitude* but never *direction*:
direct-apical excitation dominates the entire sweep.

---

## Pipeline status

| Phase | Gate | Commit | Status |
|---|---|---|---|
| Stage 0 — bottom-up V1 tuning | tuning FWHM, rate bands, WTA, no-runaway | (pre-branch, tag `phase-stage-0-passed`) | PASSED |
| Stage 1 — H_R grammar (Richter 6×6 crossover) | contextual gating, NMDA persistence, cross-channel inh | `ba3dcec`, `77d6e6d` | PASSED |
| Stage 1 — H_T grammar (Tang rotating-deviant) | sequence-selective H peaks | `ba3dcec` | PASSED |
| Stage 2 — cue learning on H_R (three-factor) | 4/4 gates seed 42 | `13bbac1` | PASSED |
| Sprint 3 backfill — 5 component validators | unit-level biology checks | `a99a8a7`, `4b1f57b`, `57ce01e`, `2731fdf`, `79daa87` | PASSED |
| Sprint 4.5 — feedback_routes (H→V1 direct + SOM) | fixed Gaussian weights, balance param | `7c87f95`, `6cb20f5` | PASSED |
| Sprint 5a — assay framework + first-pass metrics (r=1.0) | Kok + Richter + Tang assays | `526ff25`, `bf39f55` | PASSED |
| Sprint 5b — initial balance sweep | r ∈ {0.25, 0.5, 1.0, 2.0, 4.0} | (superseded; all-zero artifact) | INVALID — see below |
| Sprint 5.5 — V1→H feedforward fix (assay-time only) | V1 drives H during assay windows | `6ba542c` | PASSED |
| Sprint 5b — rerun balance sweep | r ∈ {0.25, 0.5, 1.0, 2.0, 4.0} | `6355c8b` | COMPLETE; H1 not supported |

### The Sprint 5b rerun was necessary

The first Sprint 5b pass reported complete r-invariance across every
metric. Debugger (task #30) identified the root cause: H rings emitted
zero spikes during every assay measurement window, so both feedback
routes multiplied by zero (`g_direct · H = g_SOM · H = 0`) and the
balance parameter had no effect on anything measured. The Stage 1 H
rings rely on a cue/grammar drive not present during the brief grating
windows used by the Kok/Richter/Tang assays.

Task #31 added a V1_E → H_E feedforward afferent
(`brian2_model/feedforward_v1_to_h.py`, feature-matched row-normalised
Gaussian over channel distance, `g_v1_to_h = 1.5`,
`drive_amp_v1_to_h_pA = 80.0`, `sigma_channels = 1.0`) wired into the
assay-time network builder only (training-time plasticity remains
unmodified). Diagnostic confirmed H rings are now active during assay
windows and that V1_E grating counts differ between r=0.25 and r=4.0,
so the pathway is engaged. All five r values were then re-run fresh
(five parallel tmux sessions, ~74 min wall each).

---

## H1 verdict — NOT supported

### Primary metrics vs r

| metric                       | r=0.25   | r=0.5    | r=1      | r=2      | r=4      | verdict       |
|---                           |---       |---       |---       |---       |---       |---            |
| kok_amp_delta_hz             | −0.022   | −0.023   | −0.013   | +0.000   | −0.003   | NONMONOTONIC  |
| kok_omission_mean_hz         | +1.942   | +1.955   | +1.987   | +2.035   | +2.072   | MONOTONIC     |
| kok_svm                      | +0.742   | +0.738   | +0.733   | +0.729   | +0.737   | NONMONOTONIC  |
| kok_bin0_delta               | −0.171   | −0.150   | −0.151   | −0.083   | −0.072   | NONMONOTONIC  |
| richter_redist               | −1.113   | −1.106   | −1.204   | −1.265   | −1.447   | MONOTONIC     |
| richter_center_delta         | −1.117   | −1.108   | −1.210   | −1.269   | −1.454   | MONOTONIC     |
| richter_flank_delta          | −0.004   | −0.002   | −0.006   | −0.004   | −0.007   | NULL          |
| richter_E_local_delta        | −2.522   | −2.467   | −2.700   | −2.811   | −3.211   | MONOTONIC     |
| tang_mean_delta_hz           | −1.585   | −1.308   | −1.590   | −1.606   | −1.778   | MONOTONIC     |
| tang_svm                     | +0.858   | +0.858   | +0.858   | +0.858   | +0.858   | NONMONOTONIC  |
| tang_laminar_delta_hz        | −0.058   | −0.029   | −0.044   | −0.092   | −0.101   | MONOTONIC     |
| tang_fwhm_expected           | +0.339   | +0.345   | +0.387   | +0.347   | +0.348   | MONOTONIC     |
| tang_fwhm_deviant            | +0.180   | +0.239   | +0.180   | +0.180   | +0.180   | NONMONOTONIC  |

### Classifier breakdown

H1 predicts **regime-switch**: `sign(metric @ r=0.25) ≠ sign(metric @ r=4.0)`
on non-null metrics (`|v(0.25)| > 0.01` AND `|v(4.0)| > 0.01`).

| category       | count | interpretation |
|---             |---    |---             |
| REGIME-SWITCH  | **0** | H1's positive prediction never fires |
| MONOTONIC      | 7     | sign preserved across r; magnitude grows or shrinks with r |
| NULL           | 1     | metric is near-zero everywhere (richter_flank_delta) |
| NONMONOTONIC   | 5     | sign preserved but magnitude peaks or troughs mid-sweep |

Put simply: r is a magnitude dial, not a sign flip.

---

## Robust positive findings

Independent of H1, the sweep surfaces several well-behaved effects
consistent with the predictive-coding framing:

1. **Richter local gain dampening is robust.**
   `richter_center_delta ∈ [−1.45, −1.11] Hz`, `richter_redist ∈ [−1.45, −1.11]`,
   `richter_E_local_delta ∈ [−3.21, −2.47] Hz`. Matched-channel suppression
   under the expected grammar is present at every r and grows *more* negative
   with r. Flank cells do not redistribute
   (`|richter_flank_delta| < 0.008 Hz` everywhere). Interpretation:
   suppression of the predicted cell is local — the feedback routes dampen
   gain at the matched channel without compensatory flank excitation.

2. **Kok omission response is strong and monotone.**
   `kok_omission_mean_hz ∈ [+1.94, +2.07] Hz`; the model emits a
   positive response when the expected stimulus is omitted, and that
   response grows slightly with r. This is the closest analogue in the
   dataset to an apical "prediction-error burst".

3. **Tang mean delta and FWHM-expected are monotone in r.**
   `tang_mean_delta_hz ∈ [−1.78, −1.31] Hz` and
   `tang_fwhm_expected ∈ [+0.34, +0.39] rad`. Expected-tuning curves are
   broader than deviant-tuning curves everywhere (expected cells are
   less sharply tuned), and the gap grows modestly with r.

4. **r now matters.** The first-pass Sprint 5b showed bit-exact
   invariance; after the V1→H fix, every primary metric responds to r
   (null row aside). Engagement of the feedback loop is thus *confirmed*;
   the negative H1 result is not a trivially silent circuit.

---

## Interpretation

**Direct-apical excitation wins across the entire r range.** The
direction of the monotone shifts (more r → more negative Richter /
Tang suppression, more positive Kok omission) is the direction one
would expect from stronger direct-apical drive: the direct route
excites locally-predicted E cells on the apical dendrite, which
through EI balance reduces somatic rates more under the expected
grammar, sharpens expected vs deviant contrast, and boosts the
mismatch response during omission. None of these flip sign at low r.

**The SOM route is present but subordinate.** At r=0.25 the SOM
pathway carries 4× the gain of the direct route, yet every metric
still points in the "direct-apical-winning" direction. Two explanations
are consistent with the data:

- *Gain asymmetry at the effector level.* Even when `g_SOM` is larger
  nominally, the downstream drive on V1 from the SOM route may be
  weaker per unit of gain — e.g., because SOM→E is inhibitory and
  operates at lower steady-state conductance than the direct apical
  excitation per unit weight. Without an internal calibration the r
  ratio is nominal, not functional.
- *Bottleneck upstream of the balance knob.* If feedback engagement
  (how much H modulates V1) is below the scale at which the two routes'
  qualitative differences dominate, the entire sweep sits inside the
  "whatever signal does get through is dominated by direct" regime.
  Increasing `g_total` or `g_v1_to_h` could disclose a regime-switch
  window; the current configuration apparently does not.

**Tang SVM saturates at 0.858 across the sweep** (five identical
values). SVM accuracy is insensitive to r; it is a poor
regime-discrimination metric at this operating point (deviant vs
expected remains linearly separable no matter how r rebalances the
feedback), though it confirms the readout is not at floor.

---

## Open questions and Sprint 5c options

1. **Ablation contrasts — direct-only vs SOM-only at fixed g_total.**
   Replace the r-ratio sweep with two canonical A/B runs:
   `(g_direct, g_SOM) = (g_total, 0)` vs `(0, g_total)`. This is the
   cleanest mechanistic discriminator: if SOM-only still produces the
   same-sign Richter/Tang suppressions as direct-only, H1's mechanistic
   premise (the two routes do *qualitatively different* things) is
   falsified in this model. If SOM-only flips sign versus direct-only,
   H1 is salvageable under a gain rescaling. Cost: ~2 runs × 75 min
   + the r=1.0 baseline already in hand. **Recommended default.**

2. **Feedback-engagement audit.** Quantify what fraction of H_E's
   current during grating epochs comes from the V1→H feedforward
   versus the H→V1→H round-trip. If the loop is near-open (H is
   driven almost entirely by V1→H), r cannot matter for V1 no matter
   its value because H's modulation of V1 is a small fraction of the
   bottom-up drive. A diagnostic at (r=0.25, r=4.0) × (g_v1_to_h ∈
   {0.5, 1.0, 1.5}) would tell us whether the feedback ratio has any
   chance of switching direction at lower feedforward gains. Cost:
   ~6 short runs.

3. **(g_total, r) 2D sweep.** The sign-flip may only appear when both
   feedback magnitude and balance move together; the H1 regime-switch
   may live outside g_total=1.0. Scan g_total ∈ {0.5, 1.0, 2.0} ×
   r ∈ {0.25, 1, 4} (9 runs minimum, 15 for full parity). Most
   expensive (~1 tmux day); useful *after* (1) and (2) indicate
   whether the geometry exists at all.

**Recommended sequence.** (1) → (2) → (3) if still motivated.
(1) is the cheapest test with the strongest discrimination power;
(2) tells us whether (3) would even be expected to reveal a flip.

A companion consideration: r as currently parameterised is the ratio
of nominal conductance *gains* on the two routes. An alternative
internal metric — ratio of actual post-synaptic currents during the
grating window — would make r biologically interpretable and may not
coincide with the nominal r in our sweep.

---

## Technical provenance

- **Repo root:** `/home/vysoforlife/code_files/und_OSI_formation_extend/Fold6_pvfix`
- **Branch:** `expectation-snn-v1h`
- **Key commits (parent → child):**
  - `7c87f95` feedback_routes skeleton
  - `6cb20f5` feedback_routes with Gaussian topology + r
  - `364bada` assay runtime builder
  - `526ff25` Kok/Richter/Tang assays + validators + driver
  - `bf39f55` Sprint 5a evidence log (r=1.0, seed 42)
  - `6ba542c` V1→H feedforward pathway (Sprint 5.5 fix)
  - `6355c8b` Sprint 5b balance sweep + H1 verdict (this report's source)
- **Environment:**
  - `/home/vysoforlife/miniconda3/envs/expectation_snn/bin/python` (3.12.13)
  - Brian2 2.10.1 (numpy codegen, dt = 0.1 ms)
  - numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0
- **Simulation params (assay time):**
  - Seed 42 single-seed first-pass (multi-seed replication deferred)
  - g_total = 1.0, r ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
  - V1→H feedforward: `g_v1_to_h = 1.5`,
    `drive_amp_v1_to_h_pA = 80.0`, `sigma_channels = 1.0`
  - Runtime ≈ 74 min wall per r (Kok ≈ 38, Richter ≈ 27, Tang ≈ 9)
- **Run command** (per r ∈ {0.25, 0.5, 1.0, 2.0, 4.0}):
  ```
  python -m expectation_snn.scripts.run_sprint_5a \
      --seed 42 --r <R> \
      --out expectation_snn/data/checkpoints/sprint_5b_intact_r<R>_seed42.npz
  ```
- **Artifacts:**
  - `expectation_snn/data/checkpoints/sprint_5b_intact_r{0.25,0.5,1.0,2.0,4.0}_seed42.npz`
    (82 arrays each; gitignored per `data/checkpoints/` rule — re-derivable)
  - `expectation_snn/data/figures/sprint_5b_balance_sweep.png`
  - `expectation_snn/data/figures/sprint_5b_tang_fwhm.png`
  - `expectation_snn/data/figures/sprint_5b_summary.md`
  - `expectation_snn/data/figures/sprint_5b_h1_verdict.md`
- **Aggregator:** `expectation_snn/analysis/balance_sweep.py`
- **Evidence-log section:** `expectation_snn/docs/phase_gate_evidence.md`,
  Sprint 5b.

---

*This report is reviewer-facing and summarises a negative primary
hypothesis under a single seed. Multi-seed replication and the
Sprint 5c ablations above are required before the negative verdict
is considered final.*
