# Sprint 5d Diagnostics — Pre-Registration

**Author**: Debugger (Task #42)
**Date locked**: 2026-04-20
**Status**: DRAFT — awaiting team-lead GO before any run.

This document pre-registers the six diagnostics, their pass/fail thresholds,
analysis pipelines, and the verdict mapping BEFORE any data is collected.
Any post-hoc deviation MUST be explicitly flagged in the final report.

Context: `expectation_snn/docs/SPRINT_5C_META_REVIEW.md` (reviewer critique)
and task #42 description (decision tree + diagnostic protocols).

---

## 0. Scope and non-goals

### In scope
- Six diagnostics D1–D6 producing a single case verdict (A / B / C / D / mixed).
- Verdict based on pre-registered threshold checks, not post-hoc inspection.
- All scripts land in `scripts/diag_sprint5d_<n>.py`; outputs under `data/diag_sprint5d/`.

### Out of scope (explicitly)
- No model-code edits. Read-only on `brian2_model/` and `assays/`.
- No fix implementation. Fix *proposal* in final report; Coder implements under a separate task.
- No re-training of Stage-1/Stage-2. Diagnostics run against existing `seed=42` checkpoints.
- No broad `(g_total, r)` or V1→H gain sweeps. Reviewer explicitly blocked.

---

## 1. Fixed parameters and assumptions (locked)

| Quantity | Value | Justification |
| --- | --- | --- |
| Primary seed | 42 | Sprint 5c/5b convention; only trained checkpoints. |
| Replication seeds | {42, 43, 44} | 3-seed cross-check on all primary verdicts. Brian2 re-seeds but checkpoint is fixed at 42 — so replication randomness only affects trial-generation RNG, not weights. Any verdict divergence across seeds → mark INCONCLUSIVE. |
| r (direct/SOM balance) | 1.0 (default) | Sprint 5c default. D3 sweeps r explicitly. |
| g_total | 1.0 | Sprint 5a default. |
| N_CHANNELS | 12 (both H and V1) | Code constant. |
| Tang orientation channels | [0, 2, 4, 6, 8, 10] (6-orient × 30°) | `H_ORIENT_CHANNELS` in `train.py:449`. |
| V1→H mode | per-diagnostic (see below) | Uses Sprint 5c toggle. |
| Brian2 codegen | `numpy` | Matches prior diagnostics; no JIT variance. |
| dt | 0.1 ms | Brian2 default in assays. |
| Budget | < 3 h total CPU, < 10 GB RAM | Per task description. |

### Randomness discipline
- `np.random.default_rng(seed)` for all trial-schedule RNGs.
- `brian2.seed(seed)` + `np.random.seed(seed)` before each network run.
- Bootstrap CI uses fixed RNG seed `99999` inside analysis scripts so re-analysis is deterministic.
- Record full diag-script argv + git SHA into each output npz.

---

## 2. Execution order and rationale (locked)

Order: **D1 → D5 → D6 → D3 → D4 → D2.** Cheapest-isolating first.

1. **D1** uses Sprint 5c infra directly; a null result (no prior) already narrows the decision tree (excludes B).
2. **D5** uses existing `with_v1_to_h="off"` + `with_feedback_routes=False`; if D5 survives, Case C is provisionally on the table and later diagnostics must be interpreted in light of it.
3. **D6** — Kok SNR curve — independent of H decision; runs even if D5 flips everything.
4. **D3** — most compute-heavy, but gates Case A vs Case B.
5. **D4** — impulse response; informs D3 interpretation and Case B fix specificity.
6. **D2** — re-analysis of D1 spike data; free compute. Last because it refines the mechanism call after other diagnostics have pinned the case.

STOP CONDITION: If D5 shows ≥ 70% of continuous effect magnitude preserved with H fully OFF, Case C becomes the leading candidate. D3/D4 still run (they clarify the *amount* of residual feedback contribution), but verdict weighting shifts.

---

## 3. Per-diagnostic specifications

### D1 — Pre-probe prior index

#### Inputs / infra
- `build_frozen_network(with_v1_to_h="continuous", ...)` (full model as Sprint 5c).
- Coder (`task #41 step 4`) `record_preprobe_state(bundle, window_ms)` helper per trial.
- Window definitions:
  - **Kok**: final 100 ms of the cue→stim gap (t ∈ [900, 1000] ms post-cue-onset, with cue=500 + gap=500).
  - **Richter**: final 100 ms of leader (t ∈ [400, 500] ms of leader epoch).
  - **Tang**: final 50 ms of item *t* (per-item; t ∈ [200, 250] ms within each 250 ms item).

#### Trial counts
- Kok: 60 valid + 60 invalid trials per seed (matches Sprint 5c).
- Richter: 60 expected + 60 unexpected per seed.
- Tang: 360 items per seed (6 blocks × 60 items, matching existing `TangRotatingConfig`).
- 3 seeds × above.

#### Metrics
Per assay, per trial:

```
PI_trial = H[expected_next_channel] - H[reference_channel]   # in Hz
```

- Kok reference = channel of distractor cue (the one NOT at θ).
- Richter reference = channel of leader orientation.
- Tang reference = channel of current item.

Aggregate over trials → `PI_mean`, `PI_sd`, `PI_cohens_d = PI_mean / PI_sd`.
Bootstrap 95% CI on `PI_mean` via 10,000 resamples of trial-level `PI_trial`.

Also compute:
- `H_expected_abs` = mean H rate (Hz) in expected-next channel (across trials, pre-probe window).
- `H_far` = mean H rate in θ+π/2 channel (channel offset 6 of 12).
- `d_expected_vs_current` = Cohen's d of H[expected_next] − H[reference].
- `d_expected_vs_far` = Cohen's d of H[expected_next] − H[far].

#### Pre-registered pass criteria (ALL four must clear)

| Check | Threshold |
| --- | --- |
| Absolute H in expected channel | `H_expected_abs ≥ 5 Hz` |
| Relative PI magnitude | `PI_mean ≥ 2 Hz` and 95% CI excludes 0 in expected direction |
| Effect size on PI | `PI_cohens_d ≥ 0.5` |
| Channel specificity | `d_expected_vs_current ≥ 0.5` AND `d_expected_vs_far ≥ 0.5` |

Any failure → "no meaningful pre-probe prior" for that assay.

Per-seed verdict: all 4 clear on ≥ 2 of 3 seeds → "prior present (PASS)".

#### Failure modes to guard against
- H rate near zero across all channels — record as "prior cannot be tested" rather than false negative. Flag raised if `max_H_any_channel < 1 Hz`.
- Channel-alignment bug (expected-next channel wrongly computed): sanity check: after probe, H argmax should correlate with probe θ if V1→H is active. Fail the sanity check → abort diagnostic, investigate.

---

### D2 — Forecast-vs-memory confusion matrix

#### Inputs / infra
Reuses D1 spike data. No new runs.

#### Windows (per assay trial)
Richter (main focus):
- `early_leader` (0–250 ms of leader)
- `late_leader` (250–500 ms of leader)
- `pre_trailer` (last 100 ms of leader; overlap with late_leader — treat as zoom-in, not independent)
- `early_probe` (0–100 ms of trailer)
- `late_probe` (100–500 ms of trailer)

Tang: same windows adapted to 250 ms items (early_item, late_item, pre_next, early_next_item, late_next_item).
Kok: cue_early, cue_late, gap_early, gap_late, grating.

#### Metrics
For each window, compute H argmax per trial → channel label. Classify against:
- `leader` channel
- `expected_next` channel (= trailer channel for expected trials; rotation-prediction for Tang)
- `current_sensory` channel (= whatever V1 is currently seeing)
- `other`

Aggregate → 3×4 confusion matrix (true label × H-argmax label) per window.

Report probabilities `p(H_argmax = leader | window)`, `p(expected_next | ...)`, `p(current | ...)`.

#### Pre-registered dominance criteria

Chance level = 1/12 ≈ 8.3%. Floor for "non-chance" = 2× chance = 17%.

For each critical window (late_leader, pre_trailer, pre_next in Tang):
- **Forecast dominance**: `p(expected_next) > p(leader) + 0.15` AND `p(expected_next) > 17%`.
- **Memory dominance**: `p(leader) > p(expected_next) + 0.15` AND `p(leader) > 17%`.
- **Amplifier dominance** (only meaningful during probe windows): `p(current) > both others by 0.15`.
- **Silent / diffuse**: no class exceeds 17%.

Permutation test: shuffle H argmax across trials (10,000 perms); p-value on dominant-class fraction vs null. p < 0.01 required to claim a dominance call.

#### D2 contributes to case call via:
- "Forecast dominance at late_leader" → supports Case B or genuine predictive H.
- "Memory dominance at late_leader" + D1 PASS → Stage-1 trained memory but not forecast → supports Case A with a specific mechanism (memory-only, no transform).
- "Amplifier dominance at probe windows + silent/diffuse at late_leader" → confirms Sprint 5b/5c amplifier finding → supports Case A.

---

### D5 — H-off adaptation baseline

Runs early (step 2 in order). Can dominate the verdict.

#### Inputs / infra
- `build_frozen_network(with_v1_to_h="off")`.
- Coder (`task #41 step 6`) confirmed independent toggle of `with_feedback_routes` and `with_v1_to_h`. If not exposed, diagnostic uses zero-weight feedback_routes as a proxy — flagged in report.
- Run corrected Richter + Tang at r=1.0, seed ∈ {42, 43, 44}.

#### Trial counts
- Richter: 60 expected + 60 unexpected per seed (matches Sprint 5c).
- Tang: 360 items per seed.

#### Metrics (matches Sprint 5c primary metrics)
- Richter: per-Δθ-step trailer V1_E rate (matched channel), redistribution index, expected-vs-unexpected contrast.
- Tang: per-Δθ_prev × condition (deviant / expected) matrix of V1_E rates.

Reference: Sprint 5c continuous-mode values from `data/checkpoints/sprint_5c_intact_r1.0_seed42_continuous.npz` (and replicates at 43/44 if available; else Sprint 5c at 42 only, with caveat).

#### Pre-registered pass criteria (team-lead specified)

Define:
```
retention = |effect_Hoff| / |effect_continuous|
```
per primary metric (Richter expected−unexpected, Tang deviant−expected, Tang rotational gain).

| Retention | Interpretation |
| --- | --- |
| `retention ≥ 0.70` | Effect is primarily intrinsic V1 (Case C candidate). |
| `0.30 ≤ retention < 0.70` | Mixed: partial intrinsic + partial feedback. |
| `retention < 0.30` | Effect requires H feedback to exist (Case C excluded for that metric). |

Bootstrap 95% CI on `retention` (paired trial-level resample, 10,000 perms).

Cross-seed: retention band must be consistent across ≥ 2 seeds to support Case C.

#### Failure modes
- If Sprint 5c `continuous` effect magnitude itself has CI overlapping zero, retention ratio is ill-defined. Flag metric as "no baseline effect" and move on.

---

### D6 — Kok SNR curve

#### Inputs / infra
Coder (`task #41 step 3`) `KokSNRConfig`:
- `contrast_multiplier ∈ {1.0, 0.5, 0.3}`
- `n_cells_subsampled ∈ {96, 48, 24, 12}` (96 = full V1_E set of 8 cells × 12 channels).
- `n_orientations ∈ {2, 6}`
- `input_noise_std_hz`: fixed at 0 in primary run; sweep only if the 4×3×2 = 24-combo grid fails to produce a non-saturated band.

Primary sweep grid: 3 × 4 × 2 = 24 combos × 60 valid + 60 invalid trials × 3 seeds.
Estimated cost: ~30–45 min (short trials, no H plasticity).

#### Metrics
Per combo:
- Baseline orientation-decoding accuracy `Acc_base` (chance: 1/n_orient).
- Valid-cue orientation accuracy `Acc_valid`.
- Invalid-cue orientation accuracy `Acc_invalid`.
- `Δ_decoding = Acc_valid − Acc_invalid`.
- Bootstrap 95% CI on `Δ_decoding` (stratified by cue × orientation, 10,000 resamples).

#### Pre-registered non-saturated regime
`Acc_base ∈ [55%, 80%]` (team-lead specified band).

Any combo whose `Acc_base` falls in-band AND `Acc_valid − Acc_invalid` CI excludes 0 → "Kok effect detected at [combo]".

If **at least one** non-saturated combo shows `Δ_decoding > 0` with CI excluding 0 → Case D candidate confirmed.

If **all** non-saturated combos show `Δ_decoding` CI overlapping 0 → Kok genuinely null (Case D falsified).

If **no** combo lands in [55, 80]% band → Kok SNR curve is INCONCLUSIVE; flag for grid refinement (add `input_noise_std_hz` sweep or contrast ∈ {0.2, 0.15}).

#### Failure modes
- SVM class imbalance with `n_orientations=6` and few trials → use stratified split + report class-balanced accuracy.
- Subsampling collapses a channel's representation (e.g., 12 cells × 12 channels → 1 cell/ch): acceptable, but report per-channel cell count.

---

### D3 — Controlled H-clamp test

#### Inputs / infra
Coder (`task #41 step 2`) `with_h_clamp={target_channel, clamp_rate_hz, window_start_ms, window_end_ms}` mode.

Protocol per trial (Richter):
- Leader run (500 ms, V1→H ON).
- Clamp H[expected_trailer_channel] at 50 Hz for 100 ms immediately before trailer onset (window: [400, 500] ms of leader epoch, or [0, 100] ms of pre-trailer gap if inserted — to be decided with coder).
- Set V1→H mode to "off" for the duration of the trailer → H during probe comes ONLY from clamp + NMDA recurrence.
- Run trailer 500 ms.
- Record V1_E rates in 3 windows post-trailer-onset: [0, 50], [50, 150], [150, 500] ms.

Route conditions: 4 (no-feedback, direct-only r→∞, SOM-only r→0, both r=1.0).
- "no-feedback" implemented as `with_feedback_routes=False` or zero-weight.
- "direct-only": `r = 1e6` (or equivalently `g_SOM=0, g_direct=g_total`).
- "SOM-only": `r = 1e-6` (or `g_direct=0, g_SOM=g_total`).
- "both": `r = 1.0, g_total = 1.0`.

Same protocol for Kok: clamp at expected-grating channel in final 100 ms of gap, V1→H off during grating.

Tang: **skipped** (no context window per task spec).

#### Trial counts
- Richter: 30 trials × 4 routes × 3 time-windows × 3 seeds. Time windows are within-trial, so `30 × 4 × 3 = 360` Brian2 runs per assay per seed; 2 assays (Kok+Richter) × 3 seeds = 2160 runs total. Each ~ 1 s → ~ 35–40 min.

#### Metrics
Per (route × window × seed):
- `V1_E_matched` = mean rate at expected-trailer channel (Hz).
- `V1_E_nonpref` = mean rate at orthogonal channel (±6 offset).
- `Δ_V1 = V1_E_matched − V1_E_nonpref`.
- Cohen's d of `Δ_V1` vs no-feedback condition (paired trial-level).
- Redistribution index (as in Sprint 5c): `(matched − nonpref) / (matched + nonpref)`.

#### Pre-registered thresholds (team-lead specified)

Null floor = no-feedback condition values.

| Route | "Works" if (in 0–50 ms window) |
| --- | --- |
| direct-only | `Δ_V1_matched > Δ_V1_nonpref` with Cohen's d ≥ 0.3 vs null floor |
| SOM-only | `Δ_V1_matched < Δ_V1_nonpref` (dampening) with d ≥ 0.3 vs null floor |
| both | either or both above |

Verdict per route: "works" / "null" / "inconclusive" (CI overlaps null floor).

#### D3 contributes to case call via:
- Direct and/or SOM route "works" + D1 "no prior" → **Case A confirmed** (feedback interface capable, learning absent).
- All routes "null" + D1 "prior present" → **Case B confirmed** (learning OK, feedback broken).
- Both D1 and D3 fail → combined / architecture-wide failure → Case A+B mixed.

---

### D4 — Route impulse-response transfer function

#### Inputs / infra
Coder (`task #41 step 5`) `scripts/diag_route_impulse.py` deterministic probe:
- Fixed V1 grating θ=0, contrast=1.0.
- Fixed H pulse train into H_E[ch0] at 50 Hz for 500 ms.
- 4 route configs (off / direct-only / SOM-only / both).
- Measure in 3 windows (0–50, 50–150, 150–500 ms): V1 E per-channel rate, SOM rate, PV rate, apical current `I_H_apical`, SOM input current, V1 adaptation current `I_adapt`.

Since this is deterministic, run 10 seeds (different `brian2.seed`) to bootstrap CI on rate measurements.

#### Metrics
Per (route × window):
- `V1_E_matched` vs `V1_E_nonpref` (channel 0 vs channel 6).
- `SOM_rate`, `PV_rate`.
- `I_H_apical` (integrated current over window, pA·ms).
- `I_adapt_magnitude` (V1 adaptation current).

#### Pre-registered signatures

"Early sharpening" (0–50 ms window, per route):
- `d(V1_E_matched − baseline) ≥ 0.5` AND `d(V1_E_nonpref − baseline) ≤ −0.5`.

"Late dampening" (150–500 ms window, per route):
- `d(V1_E_matched − baseline) ≤ −0.5`.

"Kim 2022 temporal crossover": early sharpening AND late dampening in the SAME route config.

#### D4 contributes to case call via:
- If crossover present in direct-only → 500 ms integration window in Sprint 5c hides early sharpening → supports "assay windowing is wrong" contributing to null.
- If no crossover in any route → pure static feedback → Case B fix needs temporal structure, not route redirection.

---

## 4. Verdict mapping (LOCKED)

Using Sprint 5c meta-review decision tree:

| D1 prior | D3 clamp | D5 H-off retention | D6 Kok @ low SNR | **Verdict** |
| --- | --- | --- | --- | --- |
| FAIL | ≥1 route "works" | < 0.70 | any | **Case A** — H learning/memory is wrong. Fix: split H into context+prediction. |
| PASS | all null | < 0.70 | any | **Case B** — feedback interface is wrong. Fix: apical/SOM/timing. |
| any | any | ≥ 0.70 on ≥2 metrics | any | **Case C** — intrinsic V1 adaptation dominates. Fix: tighter controls, not expectation. |
| any | any | any | Kok Δ > 0 at in-band SNR | **Case D** — decoder saturation (Kok-specific only, does NOT resolve Richter/Tang). |
| combined failures | — | — | — | **Mixed** — enumerate all cases with evidence. |

Special rule: Case C dominates Case A/B if retention ≥ 0.70 on BOTH Richter and Tang primary metrics. In that scenario, A/B are still reported but flagged "effects not attributable to H under H-off control".

Case D is orthogonal: it can co-exist with A/B/C and is reported independently.

---

## 5. Deliverables (locked)

1. Per-diagnostic npz: `data/diag_sprint5d/D{1,2,3,4,5,6}_seed{42,43,44}.npz`. Schema: all raw metrics + thresholds evaluated + pass/fail flags + git SHA + argv.
2. Per-diagnostic analysis notebook output: stdout summary text saved to `data/diag_sprint5d/D{n}_summary.txt`.
3. Combined verdict report via `SendMessage(team-lead, ...)`, under 1500 words, including:
   - Per-diagnostic pass/fail flag.
   - Case verdict (A/B/C/D/mixed).
   - Evidence chain per claim.
   - Proposed fix contingent on case, with file paths + estimated LOC + re-training requirement.
4. No commits of diagnostic results until team-lead approves the verdict report.

---

## 6. Failure / abort conditions

- Any infrastructure from #41 missing at start → diagnostic blocked; flag to team-lead, do not proceed on substitute paths without approval.
- Any diagnostic violates the RAM (< 10 GB) or time budget (total < 3 h) → stop, reduce trial count on the still-pending diagnostics proportionally, document in report.
- D1 sanity check fails (H argmax uncorrelated with probe θ in continuous mode) → abort entire protocol; investigate instrumentation bug before continuing.

---

## 7. Scientific integrity safeguards

- All thresholds are frozen at doc-commit time. Any post-hoc change is FLAGGED in the report in a dedicated "Post-hoc deviations" section.
- No selective seed reporting. If 1 of 3 seeds diverges, the verdict is "mixed/inconclusive" for that metric unless the divergence is explainable by an instrumentation / determinism bug (in which case, investigate + report).
- No p-hacking via multiple-comparison inflation. Bonferroni correction is NOT applied — instead, each diagnostic has exactly one pre-registered primary metric per assay; secondary metrics are exploratory and labeled as such.
- No weakening of thresholds to force a verdict. If thresholds force "INCONCLUSIVE", the report says "INCONCLUSIVE" and proposes what additional infrastructure/data would resolve it.

---

## 8. Outstanding design questions for team-lead review

1. **D1 Tang late-item window (50 ms)** — 50 ms at 0.1 ms dt is 500 samples; OK for rate estimation if H fires at ≥ 2 Hz. If rates below that, consider 100 ms window. Accept?
2. **D3 clamp pre-probe insertion for Kok** — does coder's `with_h_clamp` allow overlapping with the cue/gap epoch directly, or do we need a dedicated pre-probe hook? Confirms once #41 lands.
3. **D4 seed count** — 10 seeds for bootstrap on deterministic impulse response; OK or overkill?
4. **Sprint 5c replicates at seeds 43/44** — do they exist on disk? If only seed 42 is available for Sprint 5c continuous, D5 retention baseline is single-seed. Flag caveat; not a blocker.
5. **Case D interaction with Cases A/B/C** — the table treats D as orthogonal; confirm that's correct (a Kok-specific saturation confound doesn't flip the Richter/Tang call).

Lock-in when team-lead signs off on §4 + §8 resolutions.
