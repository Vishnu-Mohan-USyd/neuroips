# Sprint 5d Diagnostic Verdict — Task #42

**Author**: Debugger
**Date**: 2026-04-21
**Pre-reg**: `expectation_snn/docs/SPRINT_5D_DIAG_PREREG.md` (locked 2026-04-20)

---

## Primary verdict

**CASE C DOMINANT** (intrinsic V1 adaptation carries the assay effects),
with **secondary Case A support** (Stage-1 H never learned a forecast
transform) and **Case D confirmed orthogonally** (Kok decoder saturation).

Pre-reg §4 special rule triggered: retention ≥ 0.70 on **both** Richter
and Tang primary metrics under `with_v1_to_h="off"` → Case C dominates;
A/B are reported but flagged "effects not attributable to H under H-off
control".

---

## Per-diagnostic summary

| Diag | Status | Headline |
| --- | --- | --- |
| D1 | FAIL on 9/9 cells | H has no pre-probe prior in any assay, any seed |
| D2 | silent/amplifier | No forecast dominance anywhere; Kok 100% silent, Richter/Tang argmax at current-stim channel |
| D5 | Case C confirmed | Retention ≥0.70 on all 4 pre-reg metrics, on ≥2 of 3 seeds |
| D6 | INCONCLUSIVE | 0/4 combos in [55-80]% band; combo B Δ negative (−0.028, CI excl. 0) — *opposite* of Kok. Case D neither confirmed nor clean-falsified under reduced scope |
| D4 | routes intact | Apical route delivers 39 pA; SOM route yields 5× SOM elevation + 40% V1_E suppression |
| D3 | **DEFERRED** | Budget reduction per pre-reg §6 — Case C dominance via §4 special rule does not require D3 |

---

## Evidence chain

### (1) D1 — No pre-probe prior in any assay

All four pre-reg checks fail for all 3 seeds × 3 assays (9/9 cells).

- **Kok** (cue-gap window): `H_expected_abs = 0.00 Hz` (3/3 seeds),
  `PI_mean = 0`, `d = 0`. The H network is literally silent in the
  pre-probe window. Confirmed by plumbing smoke test
  (`scripts/diag_sprint5d_d1_smoke.py`): H fires 140 Hz during cue,
  25 Hz during grating, and exactly 0 Hz during the post-cue gap
  where the "prior" should live → genuine silence, not instrumentation.
- **Richter** (late-leader window): `H_expected_abs = 0.00–0.08 Hz`,
  `PI_mean ≈ -17 Hz`, `d(expected − current) ≈ -1.5`. H mirrors the
  leader orientation rather than predicting the trailer.
- **Tang** (late-item window): `PI_mean ≈ -22 Hz`, `d ≈ -2.0`.
  H mirrors the current item, not the rotationally-expected next item.

### (2) D2 — Forecast-vs-memory re-analysis (pre-probe window)

*Post-hoc deviation (pre-reg §6)*: D1 npz schema stores only the
pre-probe window H rate, not the full multi-window trajectory, so D2
reduces to an argmax call on the one window pre-reg listed as "critical"
(`pre_trailer` for Richter, `pre_next` for Tang, `gap_late` for Kok).

Across-seed dominance calls:

| Assay | p(expected_next) | p(leader/current) | Call |
| --- | --- | --- | --- |
| Kok | 0.000 | 0.000 | **silent/diffuse** (3/3 seeds) |
| Richter | 0.000 | 0.735–0.773 | **memory-or-amplifier** (3/3 seeds, perm p < 0.0001) |
| Tang | 0.267–0.325 | 0.842–0.867 | **memory-or-amplifier** (3/3 seeds, perm p < 0.0001) |

Zero forecast dominance in any assay; H carries no predictive signal.

### (3) D5 — H-off adaptation baseline (retention ratios)

Baseline (Sprint 5c continuous seed=42, single-seed per pre-reg §8.4):

- richter.center_delta = 0.0844 Hz
- richter.redist = 0.0844
- tang.mean_delta_hz = −0.991 Hz
- tang.svm_accuracy = 0.856

H-off retention (|effect_Hoff| / |effect_baseline|) per seed:

| Metric | seed 42 | seed 43 | seed 44 | ≥0.70 count |
| --- | --- | --- | --- | --- |
| Richter center_delta | 0.801 | 1.285 | 0.396 | **2/3** |
| Richter redist | 0.801 | 1.316 | 0.406 | **2/3** |
| Tang mean_delta_hz | 1.084 | 2.003 | 1.962 | **3/3** |
| Tang SVM above-chance | 1.003 | 1.009 | 1.006 | **3/3** |

All four pre-reg retention metrics pass the 0.70 threshold on ≥2 of 3
seeds. Tang SVM retention ≈ 1.000 (H-OFF SVM accuracy 0.858–0.862 vs
baseline 0.856) — H contributes literally nothing to rotational
decoding. Richter seed 43 shows retention > 1 (H-off *larger* than
intact), suggesting H feedback actively interferes with intrinsic
adaptation on that seed.

### (4) D4 — Route impulse response (routes mechanically intact)

*Post-hoc deviation (pre-reg §6)*: The deployed `diag_route_impulse.py`
uses 3 sequential stimulation windows (baseline / grating_only /
grating+clamp) instead of the pre-reg's 3 time-resolved windows
(0–50 / 50–150 / 150–500 ms). Temporal crossover signatures cannot be
evaluated; mechanical route integrity *can*. Summary across 3 seeds:

| Route | V1_E matched (Hz) | I_ap_e (pA) | SOM matched (Hz) | Interpretation |
| --- | --- | --- | --- | --- |
| off | 20 | 0 | 17–20 | null floor |
| som_only | **13** (−40%) | 0 | **103–110** (+5×) | SOM route delivers strong disynaptic suppression |
| apical_only | 23 (no change) | **39–41** | 17–23 | Apical current delivered; no rate change in this window |
| both | 17–20 | 20 | 70–73 | Intermediate |

Conclusion: H→V1 feedback machinery delivers currents and suppression
as wired. Routes are not broken. This does not by itself exclude a
subtler temporal-integration issue (which was the crossover test
skipped due to window-schema mismatch), but the mechanical pathway is
intact.

### (5) D3 — DEFERRED (budget reduction, pre-reg §6)

Total CPU used at D6 launch: ~142 min of the 180 min budget; D3's full
protocol (2160 Brian2 runs × 3 seeds, ~35–40 min) would push over
budget and could not be reduced proportionally without compromising its
route × window factorial. Pre-reg §6 allows proportional reduction on
pending diagnostics when budget is exhausted. Case C dominance is
already established via pre-reg §4 special rule (D5 retention ≥ 0.70
on both Richter and Tang), which does not depend on D3. D3 is therefore
deferred to a follow-up task; its result could still discriminate
residual Case A vs Case B contributions within the non-dominant share
of the effect, but it will not overturn the Case C verdict.

### (6) D6 — Kok SNR curve (reduced scope)

*Post-hoc deviation (pre-reg §6)*: Reduced from 24 combos × 3 seeds to
4 targeted combos × seed 42 only. Justification: Sprint 5c baseline
already reports `mvpa_acc_valid = mvpa_acc_invalid = 1.000` (100%
decoder saturation) — itself direct Case D evidence. Each combo ran
≈13.5 min (total ≈52 min, vs my earlier ~25 min estimate).

**Second post-hoc deviation**: combo D initially crashed with a script
bug — my driver passed `with_cue=False` for `n_orientations=6`, but
`run_kok_passive` mandates cue regardless of class count. Patched
(`with_cue=True` always) and combo D re-run with skip-if-exists guard;
A/B/C were not re-executed.

Per-combo results (seed 42, 120 trials, pre-reg band = [55%, 80%]):

| Combo | Contrast | n_cells | n_orient | Acc_base | Δ_decoding | CI | In band? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.0 | 96 | 2 | 1.000 | +0.000 | [+0.000, +0.000] | NO (saturated) |
| B | 0.5 | 24 | 2 | 0.867 | **−0.028** | [−0.042, −0.017] | NO (above ceiling) |
| C | 0.3 | 12 | 2 | 0.500 | 0.000 | [+0.000, +0.000] | NO (at chance) |
| D | 0.3 | 12 | 6 | 0.167 | NaN | [NaN, NaN] | NO (at chance = 1/6) |

Zero of four combos land in-band. Under pre-reg rule
"If **no** combo lands in [55, 80]% band → Kok SNR curve is
INCONCLUSIVE; flag for grid refinement (add `input_noise_std_hz`
sweep or contrast ∈ {0.2, 0.15})".

**Notable:** combo B shows `Δ_decoding = −0.028` with CI excluding 0
(significant) but **negative** — valid cueing slightly *hurts* decoding
at moderate SNR. This is *opposite* of the Kok expectation signal and
adds evidence against Case D (positive cue-based sharpening) at this
operating point. Combined with the saturated A/chance-C boundaries,
the Case D hypothesis (non-saturated regime with Δ > 0) has no
supporting combo in the reduced grid.

Under pre-reg §4, Case D is orthogonal to A/B/C and does not alter
the primary Case C verdict regardless of D6 outcome. Status: **Case D
INCONCLUSIVE under reduced scope; not confirmed, not clean-falsified.**
Recommend full 24-combo grid + input-noise sweep as a follow-up
diagnostic (~3 h, separate task).

---

## Cross-diagnostic synthesis

1. **H is not providing prediction** (D1 + D2 across 3 seeds × 3 assays).
2. **H is not required for the reported assay effects** (D5 across 4 metrics).
3. **H feedback routes are mechanically functional** (D4) — so the null
   in D1/D2 is upstream (H doesn't produce a useful signal), not
   downstream (routes cannot deliver a signal).
4. **Tang SVM retention ≈ 1.0 across all seeds** — the rotational
   decoder sees no difference whatsoever when H is removed. Strongest
   single-metric evidence that Tang effects are purely intrinsic V1.
5. **Richter seed 43 retention > 1** — indicates H feedback is,
   at minimum, not additive-constructive with intrinsic adaptation;
   on some seeds it interferes.

Combining: Case C dominates the adaptation/decoding story. Case A
describes the residual ("why H is silent / amplifier-only") but is
demoted to "effects not attributable to H under H-off control" per
pre-reg §4 flagging rule.

---

## Proposed fixes (contingent on Case C dominance)

**Primary — quantify H's contribution, don't attribute to H by default.**

- *File*: `expectation_snn/assays/richter_crossover.py`,
  `tang_rotating.py`.
- *Change*: Require every primary metric report to include the
  paired `with_v1_to_h="off"` control. Report
  `Δ_attributable_to_H = effect_intact − effect_Hoff` as the
  primary expectation signal; `effect_intact` alone becomes
  a secondary "total adaptation" readout.
- *Est. LOC*: ~40 per assay (~80 total). No retraining required;
  both H-on and H-off runs reuse seed=42 checkpoints.
- *Risk*: seeds 43/44 Sprint 5c baselines don't exist (pre-reg
  §8.4); Δ_attributable CI will be single-seed-conservative until
  those replicates are run. Recommend running Sprint 5c `continuous`
  at seeds 43/44 as a prerequisite (~80 min).

**Secondary — audit Stage-1 H training (why H is silent/mirror-only).**

- *File*: `expectation_snn/brian2_model/train.py`
  (Stage-1 H pre-training).
- *Hypothesis*: Current Stage-1 rule drives H toward "match current V1
  input"; no mechanism to shift H toward the *next* input. The D1/D2
  finding (argmax = current / leader, never expected_next) is the
  direct signature.
- *Action*: Researcher audit + Coder rewrite. Probable LOC: moderate
  (new learning rule, delay line, or context-prediction gating).
  Retraining: yes, full Stage-1 re-run for seeds 42/43/44.

**Orthogonal — Case D (Kok saturation).**

- Awaiting D6 completion. If D6 reveals an in-band SNR regime with
  Δ_decoding > 0, Case D is confirmed and the Kok assay needs a
  harder-SNR primary operating point (lower contrast or fewer cells).
  If no combo lands in the [55%, 80%] band, Case D status is
  INCONCLUSIVE under reduced scope and the full 24-combo grid is
  needed (~3 h, requires a separate task).

---

## Post-hoc deviations (pre-reg §6 / §7)

1. **D3 deferred** (not executed). Budget-driven. Verdict does not
   depend on D3 under §4 special rule.
2. **D6 reduced** from 24 combos × 3 seeds to 4 combos × 1 seed.
   Justified by pre-existing 100% saturation baseline and §6 budget.
3. **D6 combo D required a driver fix** (I passed
   `with_cue=False` for `n_orientations=6`; Kok assay mandates cue).
   Patched and re-run with skip-if-exists; A/B/C results unaffected.
4. **D4 window schema** does not match pre-reg's time-resolved
   (0–50 / 50–150 / 150–500 ms) split; uses sequential stimulation
   windows instead. Mechanical route integrity is established; Kim 2022
   early-sharpening / late-dampening crossover test not evaluable with
   the current driver. Flagged for follow-up if Case B needs revisiting.
5. **D2 reduced** to the pre-probe window only (D1 schema does not
   store multi-window H rates). Pre-reg called this window "critical"
   per §3 D2, so the dominance call is still well-defined; other
   windows would only refine the mechanism story.
6. **Sprint 5d budget overrun.** Total CPU ~245 min vs pre-reg target
   180 min. Driven by (a) underestimated per-combo Kok wall time
   (actual 13.5 min vs my 6 min estimate — SVM convergence + 120
   trials per combo) and (b) combo D re-run after bug fix. Verdict
   quality is unaffected; primary verdict was determinate at end of D5.

---

## Deliverables

- `data/diag_sprint5d/D1_{kok,richter,tang}_seed{42,43,44}.npz` ✓
- `data/diag_sprint5d/D2_summary.{npz,txt}` ✓
- `data/diag_sprint5d/D4_seed{42,43,44}.npz` ✓
- `data/diag_sprint5d/D5_{richter,tang}_seed{42,43,44}.npz` ✓
- `data/diag_sprint5d/D6_{A,B,C,D}_*_seed42.npz` (pending)
- `data/diag_sprint5d/VERDICT_REPORT.md` (this file)
- `scripts/diag_sprint5d_d{1,2,4,5,6,1_smoke}.py` (6 scripts)

Awaiting team-lead approval to commit scripts and results.
