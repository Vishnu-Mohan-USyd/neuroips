# Sprint 5c findings — does the signal survive context_only?

**Branch:** `expectation-snn-v1h`
**Commits of record:**
  · `90a75f3` review docs + pre-audit verdicts (step 1)
  · `952498d` V1→H runtime toggle — continuous / context_only / off (step 2)
  · `06230d6` R1/R2/R3 assay rewrites (step 3)
  · `82149cd` Sprint 5c dual-mode driver (step 4)
**Date:** 2026-04-20 &nbsp;·&nbsp; **Seed:** 42 &nbsp;·&nbsp; **r:** 1.0 &nbsp;·&nbsp; **g_total:** 1.0
**Wall-clock:** 155 min total (continuous: 80 min, context_only: 75 min)
**Output:**
  `data/checkpoints/sprint_5c_intact_r1.0_seed42_continuous_full.npz` (133 arrays)
  `data/checkpoints/sprint_5c_intact_r1.0_seed42_context_only_full.npz` (89 arrays)

---

## Overview

Sprint 5c addressed three reviewer concerns about the Sprint 5b analysis
of the H→V1 expectation pathway:

  - **C2 (Richter)** — the original 6×6 design used θ_L = θ_T for expected
    pairs, conflating expectation with same-orientation adaptation.
  - **C3 (Kok)**     — the validity SVM may have been picking up cue
    identity (channel 3 vs channel 9) rather than expectation per se.
  - **C5 (Tang)**    — the deviant gain may merely reflect a larger-than
    -expected Δθ from the previous item (SSA confound).

Each concern was operationalised as one assay rewrite (steps R1/R2/R3,
commit `06230d6`, 33/33 component validators PASS), then re-run at r=1.0
seed=42 in **two V1→H feedforward modes**:

  - **continuous** — V1→H always on (Sprint 5b mode).
  - **context_only** — V1→H gated to cue intervals only; turned off at
    the onset of the assay's measurement window. Tests whether each
    effect requires *contemporaneous* V1→H drive during the probe, or
    only requires the H ring to have been *primed* by the cue.

Tang has no natural context window (items are 250 ms back-to-back), so
it is skipped in `context_only` and the assay raises `ValueError` if
the bundle is constructed in that mode (validated by
`validate_v1_to_h_toggle.py: tang_rejects_context_only PASS`).

---

## Headline summary table

| Metric                                           |  continuous | context_only | Survives context_only? | Interpretation |
|---|---:|---:|---|---|
| **Kok** orientation-MVPA Δ_decoding              |  +0.000     | +0.000       | n/a (null both modes) | Decoder ceilings at 100 % under both validity conditions — no signal to gate. |
| Kok legacy validity SVM (cue+grating channels)   |  +0.733     | +0.742       | UNCHANGED             | Slight increase under context_only. Signal is **V1-local cue residual**, not feedback-driven. C3 confirmed. |
| Kok mean amplitude (valid Hz, invalid Hz)        |  3.97 / 3.99 | 4.01 / 4.03 | UNCHANGED             | No amplitude modulation in either mode. |
| **Richter** redistribution (preferred-bin Δ)     |  +0.084     | +0.066       | PARTIAL (78 %)        | Modest collapse — feedback contributes ~22 % of the redistribution. |
| Richter step-2 (Δθ=60°) redist                   |  +0.041     | +0.077       | **GROWS** under context_only | Step-2 redistribution is *suppressed* by V1→H feedback. |
| Richter step-3 (Δθ=90°) redist                   |  +0.096     | +0.048       | HALVES                | Step-3 redistribution is ~50 % feedback-driven. |
| Richter step-4 (Δθ=120°) redist                  |  +0.154     | +0.115       | PARTIAL (74 %)        | Largest absolute effect, mostly intrinsic. |
| Richter step-5 (Δθ=150°) redist                  |  +0.046     | +0.023       | HALVES                | ~50 % feedback-driven, but small. |
| **Tang** raw deviant − expected                  |  −0.99      | n/a          | (Tang skipped ctx)    | Wrong sign vs Tang prediction; explained by Δθ_prev confound — see below. |
| Tang rotating-expected − random                  |  +1.17      | n/a          | (Tang skipped ctx)    | Apparent rotational gain — also Δθ_prev confound. |

CIs and per-bin profiles in `expectation_snn/data/checkpoints/sprint_5c_*.npz`.

---

## C3 / Kok — orientation-MVPA collapses to null; legacy SVM is V1-local

The new R2 orientation MVPA decoder achieves
`acc_valid_mean = acc_invalid_mean = 1.000` under **both** modes
(20 subsamples, 30 trials per class per subsample, 5-fold linear SVM,
1 000-bootstrap CI). The Δ_decoding signal is therefore identically
zero in both modes — there is no expectation modulation of orientation
decodability that the network can express, and there is no signal for
the V1→H toggle to gate.

The legacy validity SVM still scores 0.733 (continuous) and 0.742
(context_only). Cutting V1→H feedback to "cue only" *slightly raises*
the SVM accuracy, the opposite of what a feedback-mediated expectation
signal would predict. The legacy SVM is therefore reading
**cue-channel residual activity in V1** (channels 3 vs 9 still differ
in baseline drive after the cue concludes), not an expectation effect.
**Reviewer concern C3 is confirmed.** The legacy validity SVM should
be retired as a decoding endpoint for this paradigm; if a Kok-style
expectation effect exists in this network it is too small for the
orientation-MVPA decoder to see at the present r-balance.

### Caveat — orientation MVPA ceiling

With only 2 orientations (45° vs 135°) and a 96-cell V1 E population,
linear SVM saturates trivially at 100 % accuracy on 30 trials per class.
Future work should either (a) degrade the decoder via per-trial added
noise / feature subsampling, or (b) move to a 6-orientation MVPA on
items drawn from the H_T-trained ring.

---

## C2 / Richter — redistribution PARTIALLY feedback-dependent, with step structure

Per-Δθ-step redistribution (positive values = preference-rank shift
*towards* the trailer's orientation; the sign Sprint 5b reported on the
collapsed metric):

| Δθ_step | continuous redist | context_only redist | Δ (cont − ctx) | survives? |
|---:|---:|---:|---:|---|
| 2 (60°)  | +0.041 | +0.077 | **−0.036** | feedback **suppresses** redistribution |
| 3 (90°)  | +0.096 | +0.048 | +0.048 | feedback ~doubles step-3 redist |
| 4 (120°) | +0.154 | +0.115 | +0.039 | mostly intrinsic, modest feedback boost |
| 5 (150°) | +0.046 | +0.023 | +0.023 | feedback ~doubles step-5 redist |

Center-bin deltas (where the redistribution lives — flank deltas are
~0 across all steps in both modes) follow the same pattern. The
redistribution is therefore not a single homogenous "expectation"
signal: V1→H feedback **suppresses** redistribution at the smallest
unexpected step (Δθ=60°), and **enhances** it at intermediate /
large unexpected steps (Δθ=90°, 150°), with a near-null contribution
at Δθ=120° where the effect is largest in absolute terms.

The collapsed Sprint-5b-style metric (+0.084 cont vs +0.066 ctx, 78 %
retained) **averages over a sign-changing pattern** and obscures the
step-dependent feedback contribution. **Reviewer concern C2 is at
least partially addressed** by the deranged-permutation design — the
expected pairs no longer have θ_L = θ_T, so the residual redistribution
is not a same-θ adaptation artifact. The remaining feedback-suppressed
step-2 effect requires a mechanistic explanation (likely surround
suppression on the matched H channel).

---

## C5 / Tang — Δθ_prev stratification reveals SSA, not predictive coding

The raw Tang contrasts are **opposite to the Tang prediction**:

  - rotating-deviant − rotating-expected = **−0.99 Hz** (deviants *lower*
    than expected, not higher)
  - rotating-expected − random = +1.17 Hz (apparent rotational gain)

But the new R3 Δθ_prev × condition stratification shows this is almost
entirely a **stimulus-specific adaptation (SSA) confound**, not a
predictive-coding signature.

### Δθ_prev × condition rate grid (Hz, matched-θ, continuous mode)

| Condition \\ Δθ_prev | 0 (same θ) | 1 (30°) | 2 (60°) | 3 (90°) |
|---|---:|---:|---:|---:|
| random            | **19.01** (n=99) | 24.67 (n=152) | 23.74 (n=172) | 23.77 (n=76) |
| rotating_expected | **19.50** (n=12) | 24.50 (n=380) | 23.26 (n=24)  | 22.94 (n=13) |
| rotating_deviant  | **20.24** (n=11) | 23.77 (n=18)  | 24.10 (n=27)  | 23.50 (n=15) |

(`tang.dtheta_prev_rate_hz_grid` and `tang.dtheta_prev_n_trials_grid`.)

Two clear patterns:

1. **At Δθ_prev = 0 (same orientation as previous item) every condition
   collapses by ~4-5 Hz** — pure SSA. The depression has nothing to do
   with rotational expectation; it's there in the random block too.
2. **At matched Δθ_prev steps (1, 2, 3), expected and deviant rates are
   nearly identical, and at step 2 the deviant is HIGHER than the
   expected** (24.10 vs 23.26 = +0.84 Hz, n_dev=27 trials).

The "rotating_expected − random = +1.17 Hz" gain is therefore a *Δθ_prev
distribution* artifact: random items have 99/500 = **20 % at the
adapted step=0 bin**, while rotating_expected items have 12/429 = **3 %
at step=0** (380/429 = 89 % at step=1, the canonical rotation step).
After Δθ_prev stratification, there is no rotational expectation gain
at all (random = 24.67, rot_exp = 24.50 at step=1).

The "rotating_deviant − rotating_expected = −0.99 Hz" deficit is the
same artifact: deviants are spread across Δθ_prev bins (mostly steps
2-3, only 18/71 = 25 % at step=1), while expected items live almost
entirely at step=1 — the *least adapted* bin — pushing the expected
mean upward and creating an apparent deviant deficit.

**Reviewer concern C5 is confirmed in the strongest possible form.**
The raw Tang contrasts are dominated by SSA / adaptation confounds.
After stratification, the only residual signal is a step-2 deviant gain
of +0.84 Hz (n=27 trials, no CI computed in the stratified grid yet —
needs follow-up bootstrap), which would be the genuine Tang signature
if it survives proper inference.

---

## Decision gate

The dual-mode rerun **does not rescue the predictive-coding signature**
in this network at r=1.0:

- **Kok**: orientation-MVPA null in both modes; legacy SVM signal is
  V1-local. No expectation effect this network expresses.
- **Richter**: redistribution is real but step-structured, with V1→H
  feedback making mixed-sign per-step contributions. The collapsed
  metric is misleading.
- **Tang**: raw contrasts are SSA artifacts. Genuine deviant gain (if
  present) is only visible after Δθ_prev stratification and is small
  (+0.84 Hz at Δθ_prev=2, n=27 trials).

### Recommended next steps

The Sprint 5c rewrites successfully **diagnosed** the issues (C2/C3/C5
all confirmed at varying strengths) but the underlying network does not
appear to express the expected predictive-coding signatures at r=1.0.
The next sprint should **not** be another assay rerun. Two structural
hypotheses are worth testing before declaring H1 dead at the model
level:

1. **H_T persistence** — Tang items are 250 ms back-to-back. The H_T
   ring's NMDA decay constant (and the V1→H feedforward time constant)
   may be too fast to maintain a *predictive prior* across items. If
   H_T spike rate decays to baseline within an item, it cannot bias the
   *next* item's V1 response. A diagnostic with a `SpikeMonitor` on
   H_T and per-item rate logging would settle this in ~10 min.
2. **V1→H gain** — `g_v1_to_h = 1.5` was set in Sprint 5.5 to ensure
   non-zero H spike rate during assay windows, not to maximise the
   prediction signal. A small gain sweep (g ∈ {0.5, 1.5, 4.5}) at the
   Richter assay only would test whether the step-2 feedback-suppression
   pattern strengthens or inverts at higher gain.

Both are smaller-scope investigations than another r-sweep and can be
fit into a single Sprint 5d before considering whether to re-design
either the H ring or the feedback routes.

### What's defensible to publish from Sprint 5b/5c

- The **infrastructure** (frozen-bundle assay framework, three primary
  assays with per-component validators, V1→H runtime toggle, dual-mode
  driver) is solid and reusable.
- The **diagnostic finding** that the original Sprint 5a/5b metrics
  contain confounds (C2/C3/C5) is a publishable methodological
  contribution.
- The **r-invariance** result from Sprint 5b stands but is now
  contextualised: the model's H→V1 feedback simply does not produce
  the modulation that Kok / Richter / Tang were designed to detect at
  the present configuration.

The honest framing for a paper: this is a *negative methodological
finding* on this specific cortical-surrogate model — predictive-coding
assays applied naively yield apparent effects that disappear under
proper baseline / decoder / covariate controls. The model is a useful
test bed for identifying these confounds, not yet a working
predictive-coding instantiation.
