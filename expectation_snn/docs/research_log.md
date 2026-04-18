# Research log

## Phase 0 (2026-04-18)

- Branch `expectation-snn-v1h` created as orphan from `main` (no `v2-module-implementation` history).
- Skeleton tree per plan v5 sec 10.
- Single Brian2 env `expectation_snn` (Py 3.12, numpy >= 2.0, brian2 2.10.1).

## Tang 2023 paradigm (PDF-verified in Phase 0, 2026-04-18)

**Source**: Tang, M.F., Smout, C.A., Arabzadeh, E., Mattingley, J.B. "Expectation violations
enhance neuronal encoding of sensory information in mouse primary visual cortex."
*Nat Commun* 14:1196 (2023). DOI: 10.1038/s41467-023-36608-8.
PMID: 36864037. PMC: PMC9981605.

**Species**: male mice; two-photon calcium imaging (GCaMP6f); both awake and
anaesthetised mice.

### Stimulus paradigm — numbers verified from PMC full text

| Parameter | Plan v5 assumption | Paper value | Status |
|---|---|---|---|
| Orientation set | 6 at 30° steps, 0–150° | "uniform probability from 0 to 150° in 30° steps" | ✅ match |
| Per-item duration | 250 ms | "displayed for 250 ms" | ✅ match |
| Presentation rate | 4 Hz | "4 Hz presentation rate" | ✅ match |
| ISI | none | "no inter-stimulus blank interval" | ✅ match (back-to-back) |
| Rotation block length | 5–9 items before deviant | "rotated … for 5–9 presentations before jumping to an unexpected random orientation" | ✅ match (selection method in [5,9] not specified explicitly) |
| Deviant rate | ~12% | **implied** ~10–17% (1 deviant per block of 5–9; mean block ~7 → ~13%); not stated verbatim as a percentage | ≈ match, derive from block length |
| Rotation direction | n/a | clockwise OR anti-clockwise per block (e.g., 0→30→60 or 0→150→120) | **add to plan** |
| Random condition | n/a | orientations drawn pseudo-randomly uniform 0–150° in 30° steps | contrastive baseline |

### Additional parameters from Methods (for completeness)

- **Grating contrast**: 50 % (Michelson).
- **Spatial frequency**: 0.034 cycles/degree.
- **Stimulus extent**: full-screen gratings on monitor subtending 76.8° × 43.2°.
- **Trials per run**: 1800. Two runs each of Rotating and Random per session.

### Implications for H_T stimulus builder (brian2_model/stimulus.py)

- Sample rotation direction per block uniformly from {clockwise, anti-clockwise};
  block length uniformly from {5, 6, 7, 8, 9}; on block end, sample an unexpected
  orientation uniformly from the 5 remaining orientations (i.e., excluding the one
  the rotation would have produced next).
- No ISI between items.
- Per-item duration 250 ms ⇒ 4 Hz.
- At training scale for H_T: with mean block 7 + 1 deviant = 8 items per mini-sequence,
  a 1800-item run is ~225 mini-sequences. We can match Tang's within-session trial count.


## Pre-registration — Plan v5 frozen at 2026-04-18

This pre-registration is committed before any assay runs. Any deviation after this
commit must be documented as an amendment with date + rationale + what it touches.

### Seeds

| Stage | Seed set | Status |
|---|---|---|
| **Current stage** — all development + first-pass Kok/Richter/Tang assays | **{42}** | **active** |
| Paper main (reported in tables/figures) | {42, 7, 123, 2024, 11} | **deferred** — only runs once a first-pass finding is worth replicating |
| Held-out (robustness) | {99, 314} | **deferred** — touched once at end, only if paper-main battery confirms |

Rule (amended 2026-04-18): current stage runs seed=42 only. Multi-seed
replication is pre-registered here so the seed ladder is locked, but the
decision to execute the {7, 123, 2024, 11} battery is conditional on seed 42
producing an interpretable first-pass signal on Kok / Richter / Tang. If that
signal is absent or equivocal, we revisit the architecture / calibration
before spending additional seed budget. {99, 314} are touched once at the
end, only if the paper-main battery confirms.

Consequently, the §5 evidence package's held-out-seed replication is
downgraded from a first-pass gate to a deferred replication check.

### Trial counts per seed

| Assay | Count per seed | Note |
|---|---|---|
| Kok 2012 passive cue (H2, H3) | 240 trials | 180 expected-valid + 60 unexpected-invalid (3:1). |
| Richter 2022 6 L × 6 T cross-over | 360 trials | Raised from plan's 240 to enable stratified splits for suppression-vs-distance and pseudo-voxel FM. |
| Tang 2023 rotating (H1, H4) | 1000 items | ~120 deviants at 12% deviant rate. |
| Stage-1 H_R training (incidental pair co-occurrence) | 900 pair trials | — |
| Stage-1 H_T training (rotation-statistic) | 1500 items | Matches Tang run-length at training scale. |

### Balance sweep ladder

r = g_direct / g_SOM (feedback routed to V1_E directly vs via SOM) takes the
values {0.25, 0.50, 1.00, 2.00, 4.00}. g_total = g_direct + g_SOM held constant
across rungs (value calibrated in Stage 0 to hit target V1_E rate band 2–8 Hz
with baseline tuning FWHM 30–60°). Labels: S1 = r=0.25, S2 = 0.50, S3 = 1.00,
S4 = 2.00, S5 = 4.00. S1 is SOM-dominated, S5 is direct-dominated.

### H4 FWHM null test

TOST (two one-sided t-tests) equivalence at Cohen's d < 0.30 between
expected-valid and unexpected-invalid FWHM distributions on Tang rotating. The
threshold is relaxed from strict d < 0.15 for feasibility per power calculation:
at n=5 seeds × 1000 items × 6 orientations, d = 0.30 is a reachable null band
given per-seed FWHM noise floor, while d = 0.15 would demand ≥20 seeds. d < 0.30
is still tight enough to distinguish H4 (gain-only) from an FWHM-narrowing
sharpening account.

### Primary metrics

1. **Suppression-vs-preference deciles** (Kok/Richter) — rank V1_E cells by
   tuning preference relative to expected orientation (10 bins), report mean
   response in expected-valid vs unexpected-invalid; sharpening predicts
   suppression concentrated in off-preference deciles, dampening predicts
   uniform suppression.
2. **Suppression-vs-distance-from-expected 8 × 8 grid** (Richter cross-over) —
   joint factor of (leader distance, trailer distance) from cue-congruent
   orientation; reveals whether suppression tracks absolute mismatch or
   relative-to-expectation.
3. **Total population activity** — summed V1_E spikes in 50–250 ms window
   post-stimulus, per condition.
4. **Preferred-channel gain** — mean response of top-decile cells (those
   preferring the shown orientation) relative to baseline; pure gain signature
   if FWHM null passes.
5. **Tuning fit** — fitted von-Mises FWHM per cell per condition; used in the
   H4 TOST null.
6. **Omission-subtracted response** (Tang) — deviant-minus-expected response per
   cell at the deviant position; principal differentiator between sharpening
   (positive) and dampening (negative) accounts.
7. **4 virtual voxels** — spatially pooled V1_E groups of n_cells/4 each for
   pseudo-voxel forward-model analyses bridging to Richter's fMRI protocol.

### Secondary metrics

- **5-fold linear-SVM decoding** of stimulus orientation from V1_E population
  vector, per condition; reports decoding accuracy, not representation shape.
- **6-model pseudo-voxel forward-model family** (Richter-matched): sharpening,
  dampening, prediction-error, gain-only, FWHM-only, null — fit to virtual-voxel
  responses and compared by AIC.

### Expected A1–A4 ablation patterns per hypothesis

Ablations from plan §6: A1 = feedback off (V2→V1 and H→V1 both cut), A2 = H
recurrent memory off (E↔E within H zeroed during delay window), A3 = direct
route off (g_direct = 0, all feedback via SOM), A4 = SOM route off (g_SOM = 0,
all feedback direct).

| Hypothesis | Prediction | Ablation signature |
|---|---|---|
| H1 (balance → regime) | Regime label is a monotone function of r across S1..S5. | S1 (SOM-dominated) loses its dampening signature under A4; S5 (direct-dominated) loses its sharpening signature under A3. |
| H2 (dual-route dissociation) | Sharpening depends on direct route, dampening on SOM route. | Sharpening reverses under A3 but survives A4; dampening reverses under A4 but survives A3. |
| H3 (statefulness) | Kok/Richter expectation effects require H recurrent persistence during the cue→target delay. | Both Kok 180/60 suppression-vs-preference and Richter 6 × 6 cross-over collapse under A2. |
| H4 (Tang gain not FWHM) | Tang deviant response is a preferred-channel gain change, not an FWHM narrowing. | Gain metric is reliable across seeds; FWHM TOST passes at d < 0.30. |

### Bibliographic anchors

| Tag | Citation | ID |
|---|---|---|
| Tang 2023 | Expectation violations enhance neuronal encoding … | PMID 36864037 |
| Kok 2012 | Less is more: expectation sharpens representations … | PMID 22841311 |
| Richter 2022 | Predictive coding of natural images … | DOI 10.1093/oons/kvac013 |
| Kim 2022 | Prediction-error neurons in mouse visual cortex | PMID 36302912 |
| Vogels 2011 | Inhibitory plasticity balances excitation … | PMID 22075724 |
| Clopath 2010 | Voltage and spike timing interact in STDP | PMID 20098420 |
| Wang 2001 | Synaptic reverberation underlying working memory | PMID 11476885 |
| Frémaux & Gerstner 2015 | Neuromodulated STDP and three-factor rules | PMID 26834568 |

### Resource limits (session)

- RAM ≤ 10 GB (per-process peak).
- GPU ≤ 10 GB (VRAM peak).
- Do not kill non-session processes.
- Long runs (> 5 min wall-clock) proceed in autonomous mode but are still
  reported to team-lead with a sized-up proposal before launch.
